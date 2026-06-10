"""Hold-and-merge dispatch for ASR finals (turn merging off the audio thread).

Two live failures share one root (run-20260609-234435 / run-20260610-003800):
the endpoint commits a final at a mid-thought pause, and the brain ANSWERS the
fragment -- "A long story about" got "Please let me know what topic...",
"So they" got "are probably asking about someone.", and each spoken answer's
echo then destabilized barge-in. The existing ADD-ON/continuation layer cannot
help: it merges only while the prior task is still IN FLIGHT, and the fast tier
answers in ~0.6s, so by the time the user's next words arrive the merge window
is gone.

This module is the missing layer at the final-dispatch seam:

* A final that READS INCOMPLETE -- ends on a conjunction/article/preposition or
  is a tiny fragment -- is HELD for a short bounded window instead of being
  dispatched. If the user keeps talking (a partial arrives / the next final
  lands), the texts are MERGED into one query and the window extends (bounded
  by ``max_hold_sec``); when the user is actually done, the merged turn
  dispatches once. A complete-reading final dispatches with NO added latency.
* Control words ("yes", "stop", "never mind", ...) are exempt -- a confirm or
  reset must never wait out a hold window.
* Dispatch happens on the :class:`FinalDispatcher`'s own worker thread, taking
  the addressing-gate -> cleaner -> router -> publish chain OFF the audio
  capture thread (review finding rc-2: up to three blocking LLM calls per final
  ran on the capture thread, starving mic reads and the KWS poll).

The decision logic (:class:`FinalCoalescer`) is pure + deterministic; the
threading wrapper (:class:`FinalDispatcher`) is a single daemon worker + a
condition variable, with a ``flush``/``stop`` contract for shutdown. Everything
is config-gated (``turn_merge`` block; dataclass default OFF -> byte-identical
behavior for programmatic construction; shipped config.json opts in).
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

from always_on_agent.text import normalize_text

from .endpointing import DEFAULT_INCOMPLETE_ENDINGS

log = logging.getLogger("speaker.turn_merge")

# Words that, when they END a committed final, mean the thought very likely
# continues. SUPERSET of the endpoint detector's conservative list: a false
# "incomplete" there wrongly EXTENDS a finished turn at the recognizer (hard to
# recover); here it only costs ``hold_sec`` of added latency on that turn (the
# final still dispatches when the window expires), so stranding-prone
# prepositions are worth including -- "A long story about" (the live failure)
# ends on "about", which the endpoint list deliberately excludes.
DEFAULT_HOLD_ENDINGS: frozenset[str] = DEFAULT_INCOMPLETE_ENDINGS | frozenset({
    "about", "of", "with", "to", "into", "onto", "than", "so", "if",
    "that", "whose", "versus",
    # Romanian (de-diacritic'd): despre (about), cu (with), decat (than),
    # daca (if), care (which/that).
    "despre", "cu", "decat", "daca", "care",
})

# Short utterances that are COMPLETE by convention -- confirms, denials,
# commands, greetings, resets. Never held, even at 1-2 words: a "yes" answering
# a CONFIRM prompt or a "stop" must act immediately. EN + RO.
DEFAULT_EXEMPT_PHRASES: tuple[str, ...] = (
    "yes", "no", "yeah", "yep", "nope", "ok", "okay", "sure", "stop",
    "cancel", "thanks", "thank you", "never mind", "nevermind",
    "start again", "start over", "hello", "hi", "hey", "go on", "continue",
    "why", "how", "what", "really",
    # Romanian
    "da", "nu", "bine", "opreste", "multumesc", "mersi", "salut", "buna",
    "continua", "de ce",
)


@dataclass
class TurnMergeConfig:
    """The ``turn_merge`` config block. ``enabled`` defaults OFF so programmatic
    construction is byte-identical; the shipped ``config.json`` opts in."""

    enabled: bool = False
    # How long an incomplete-reading final waits for the user's next words.
    hold_sec: float = 1.2
    # Hard cap on the TOTAL hold (from the first held final) across merges and
    # partial-driven extensions -- a mis-scored turn still dispatches.
    max_hold_sec: float = 6.0
    # A final at/below this many words is a fragment -> held (unless exempt).
    max_fragment_words: int = 2
    hold_endings: frozenset[str] = field(default=DEFAULT_HOLD_ENDINGS)
    exempt_phrases: tuple[str, ...] = field(default=DEFAULT_EXEMPT_PHRASES)

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, object]]) -> "TurnMergeConfig":
        data = data or {}
        endings = data.get("hold_endings")
        exempt = data.get("exempt_phrases")
        return cls(
            enabled=bool(data.get("enabled", False)),
            hold_sec=float(data.get("hold_sec", 1.2) or 1.2),
            max_hold_sec=float(data.get("max_hold_sec", 6.0) or 6.0),
            max_fragment_words=int(data.get("max_fragment_words", 2) or 2),
            hold_endings=(
                frozenset(str(w) for w in endings)
                if isinstance(endings, (list, tuple))
                else DEFAULT_HOLD_ENDINGS
            ),
            exempt_phrases=(
                tuple(str(p) for p in exempt)
                if isinstance(exempt, (list, tuple))
                else DEFAULT_EXEMPT_PHRASES
            ),
        )


class FinalCoalescer:
    """Pure hold/merge decisions over final texts. No threads, no I/O."""

    def __init__(self, config: Optional[TurnMergeConfig] = None) -> None:
        self._c = config or TurnMergeConfig()
        self._exempt = frozenset(
            n for p in self._c.exempt_phrases if (n := normalize_text(p))
        )

    def should_hold(self, text: str) -> bool:
        """True iff this final reads as a mid-thought fragment worth waiting on.

        Held: ends on a hold word ("A long story about") or is a tiny fragment
        ("So they", "Dear me"). Never held: empty, an exempt control phrase, or
        a normal complete-reading utterance ("Can you hear me")."""
        words = normalize_text(text).split()
        if not words:
            return False
        if " ".join(words) in self._exempt:
            return False
        if words[-1] in self._c.hold_endings:
            return True
        return len(words) <= self._c.max_fragment_words

    @staticmethod
    def merge(prev: str, addition: str) -> str:
        """One query from a held final + the user's continuation."""
        return f"{prev.strip()} {addition.strip()}".strip()


class FinalDispatcher:
    """Single-worker dispatch queue for finals, with hold-and-merge.

    ``submit`` is called from the engine/audio thread and only takes a lock +
    notifies (cheap, honoring core/engine.py's callback contract); the worker
    thread runs ``dispatch`` -- the full addressing/cleaner/router/publish
    chain -- when the hold window (0 for complete finals) expires. ``note_
    partial`` extends an open hold (the user resumed speaking; their next final
    will merge), bounded by ``max_hold_sec``. ``stop`` flushes a pending final
    through dispatch so shutdown never silently drops the user's words."""

    def __init__(
        self,
        dispatch: Callable[[str], None],
        config: Optional[TurnMergeConfig] = None,
    ) -> None:
        self._dispatch = dispatch
        self._c = config or TurnMergeConfig()
        self._coalescer = FinalCoalescer(self._c)
        self._cv = threading.Condition()
        self._pending: Optional[str] = None
        self._deadline = 0.0
        self._hold_started = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        # Diagnostics (read by tests/run-bundle debugging).
        self.merged_count = 0
        self.held_count = 0

    @property
    def has_pending(self) -> bool:
        with self._cv:
            return self._pending is not None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="speaker-final-dispatch", daemon=True
        )
        self._thread.start()

    def submit(self, text: str) -> None:
        """Engine-thread entry: queue a final, merging into an open hold."""
        now = time.monotonic()
        with self._cv:
            if self._pending is None:
                self._pending = text
                self._hold_started = now
                if self._coalescer.should_hold(text):
                    self.held_count += 1
                    self._deadline = now + self._c.hold_sec
                    log.debug("holding incomplete final %r for up to %.1fs",
                              text, self._c.hold_sec)
                else:
                    self._deadline = now  # complete -> dispatch at once
            else:
                merged = self._coalescer.merge(self._pending, text)
                self.merged_count += 1
                log.info("merged held final with continuation -> %r", merged)
                self._pending = merged
                extend = (
                    self._c.hold_sec if self._coalescer.should_hold(merged) else 0.0
                )
                self._deadline = min(
                    now + extend, self._hold_started + self._c.max_hold_sec
                )
            self._cv.notify()

    def note_partial(self) -> None:
        """The user resumed speaking while a final is held: keep holding until
        their next final lands (bounded by ``max_hold_sec``)."""
        with self._cv:
            if self._pending is None:
                return
            self._deadline = min(
                time.monotonic() + self._c.hold_sec,
                self._hold_started + self._c.max_hold_sec,
            )
            self._cv.notify()

    def flush(self) -> None:
        """Dispatch any held final immediately (e.g. before shutdown)."""
        with self._cv:
            self._deadline = 0.0
            self._cv.notify()

    def stop(self, timeout: float = 2.0) -> None:
        """Flush + stop the worker. A pending final is dispatched, not dropped."""
        self.flush()
        with self._cv:
            self._running = False
            self._cv.notify()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        # The worker exited; if a straggler final is still pending (raced the
        # shutdown), dispatch it synchronously so the user's words aren't lost.
        with self._cv:
            text, self._pending = self._pending, None
        if text is not None:
            try:
                self._dispatch(text)
            except Exception:  # noqa: BLE001 - shutdown path, never raise
                log.exception("final dispatch failed during stop()")

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._running and self._pending is None:
                    self._cv.wait()
                if not self._running:
                    return
                # A hold may be extended by submit/note_partial while we wait;
                # loop until the (current) deadline truly passed.
                now = time.monotonic()
                if now < self._deadline:
                    self._cv.wait(timeout=self._deadline - now)
                    continue
                text, self._pending = self._pending, None
            try:
                self._dispatch(text)
            except Exception:  # noqa: BLE001 - a turn must never kill the worker
                log.exception("final dispatch raised; turn dropped")
