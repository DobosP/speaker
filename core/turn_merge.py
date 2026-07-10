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

The decision logic (:class:`FinalCoalescer`) is pure + deterministic. The
threading wrapper (:class:`FinalDispatcher`) uses one coordinator plus a bounded
provider bulkhead: a newer final can retire an uncommitted preprocessing lease
without letting cancellation-ignoring providers create unbounded threads.
Hold-and-merge remains config-gated; runtimes with LLM preprocessing also use
the cancellable dispatcher when merging is off so provider waits never block an
engine callback.
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

DEFAULT_MAX_ACTIVE_DISPATCHES = 6
_DISPATCH_CANCEL_POLL_SEC = 0.01

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


class FinalDispatchLease:
    """Cancellation and terminal-commit token for one preprocessing chain.

    A newer final can retire an uncommitted lease. ``claim_commit`` linearizes
    the old terminal effects before or after that newer submission, preventing a
    late gate result from publishing a stale task or mutating memory.
    """

    def __init__(
        self,
        owner: "FinalDispatcher",
        generation: int,
        *,
        merge_next: bool = False,
        coalesced: bool = False,
        submitted_at: Optional[float] = None,
        input_generation: Optional[int] = None,
        input_epoch: Optional[int] = None,
    ) -> None:
        self._owner = owner
        self.generation = generation
        self.merge_next = bool(merge_next)
        self.coalesced = bool(coalesced)
        self.submitted_at = submitted_at
        self.input_generation = input_generation
        self.input_epoch = input_epoch
        self.cancel_event = threading.Event()
        self._committed = False

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def claim_commit(self) -> bool:
        return self._owner._claim_lease_commit(self)


class FinalDispatcher:
    """Off-thread final dispatch with optional hold/merge and cancellation.

    ``submit`` is called from the engine/audio thread and only takes a lock +
    notifies (cheap, honoring core/engine.py's callback contract); the worker
    thread runs ``dispatch`` -- the full addressing/cleaner/router/publish
    chain -- when the hold window (0 for complete finals) expires. ``note_
    partial`` extends an open hold (the user resumed speaking; their next final
    will merge), bounded by ``max_hold_sec``. In cancellable mode, a coordinator
    owns generation leases and provider calls run behind a bounded bulkhead;
    shutdown retires uncommitted work instead of starting another LLM call."""

    def __init__(
        self,
        dispatch: Callable[..., None],
        config: Optional[TurnMergeConfig] = None,
        on_hold: Optional[Callable[[], None]] = None,
        *,
        cancellable: bool = False,
        max_active_dispatches: int = DEFAULT_MAX_ACTIVE_DISPATCHES,
    ) -> None:
        self._dispatch = dispatch
        self._c = config or TurnMergeConfig()
        self._on_hold = on_hold
        self._coalescer = FinalCoalescer(self._c)
        self._cv = threading.Condition()
        self._lifecycle_lock = threading.Lock()
        self._stop_in_progress = False
        self._pending: Optional[str] = None
        self._pending_submitted_at: Optional[float] = None
        self._pending_input_generation: Optional[int] = None
        self._pending_input_epoch: Optional[int] = None
        self._deadline = 0.0
        self._hold_started = 0.0
        self._dispatching = False
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cancellable = bool(cancellable)
        self._dispatch_slots = threading.BoundedSemaphore(
            max(1, int(max_active_dispatches))
        )
        self._next_generation = 0
        self._active_lease: Optional[FinalDispatchLease] = None
        self._active_text: Optional[str] = None
        self._pending_merge_next = False
        self._pending_coalesced = False
        # Diagnostics (read by tests/run-bundle debugging).
        self.merged_count = 0
        self.held_count = 0
        self.superseded_count = 0

    @property
    def has_pending(self) -> bool:
        with self._cv:
            return self._pending is not None or self._dispatching

    def start(self) -> None:
        with self._cv:
            if self._stop_in_progress:
                return
            if self._thread is not None and self._thread.is_alive():
                return
            self._running = True
            thread = threading.Thread(
                target=self._run, name="speaker-final-dispatch", daemon=True
            )
            self._thread = thread
            try:
                # Publish + start atomically against stop()/a second start().
                # The new coordinator simply waits on this condition until the
                # lock is released.
                thread.start()
            except BaseException:
                if self._thread is thread:
                    self._thread = None
                    self._running = False
                raise

    def submit(
        self,
        text: str,
        *,
        submitted_at: Optional[float] = None,
        input_generation: Optional[int] = None,
        input_epoch: Optional[int] = None,
    ) -> None:
        """Engine-thread entry: queue a final, merging into an open hold."""
        now = time.monotonic()
        with self._cv:
            if not self._running:
                log.debug("dropping final submitted to stopped dispatcher: %r", text)
                return
            active_hold_started: Optional[float] = None
            coalesced_with_active = False
            if (
                self._cancellable
                and self._active_lease is not None
                and not self._active_lease._committed
            ):
                was_cancelled = self._active_lease.cancel_event.is_set()
                self._active_lease.cancel_event.set()
                active_text = self._active_text
                if (
                    not was_cancelled
                    and self._pending is None
                    and active_text
                    and now <= self._hold_started + self._c.max_hold_sec
                    and (
                        self._active_lease.merge_next
                        or (
                            self._c.enabled
                            and self._coalescer.should_hold(active_text)
                        )
                    )
                ):
                    text = self._coalescer.merge(active_text, text)
                    active_hold_started = self._hold_started
                    coalesced_with_active = True
                    self.merged_count += 1
                    log.info(
                        "merged active fragment with continuation -> %r", text
                    )
            if self._pending is None:
                self._pending = text
                self._pending_submitted_at = submitted_at
                self._pending_input_generation = input_generation
                self._pending_input_epoch = input_epoch
                self._pending_merge_next = False
                self._pending_coalesced = coalesced_with_active
                self._hold_started = (
                    active_hold_started
                    if active_hold_started is not None
                    else now
                )
                if self._c.enabled and self._coalescer.should_hold(text):
                    self.held_count += 1
                    self._deadline = min(
                        now + self._c.hold_sec,
                        self._hold_started + self._c.max_hold_sec,
                    )
                    self._note_hold()
                    log.debug("holding incomplete final %r for up to %.1fs",
                              text, self._c.hold_sec)
                else:
                    self._deadline = now  # complete -> dispatch at once
            elif (
                not self._cancellable
                or self._pending_merge_next
                or (
                    self._c.enabled
                    and self._coalescer.should_hold(self._pending)
                )
            ):
                merged = self._coalescer.merge(self._pending, text)
                self.merged_count += 1
                log.info("merged held final with continuation -> %r", merged)
                self._pending = merged
                self._pending_submitted_at = submitted_at
                self._pending_input_generation = input_generation
                self._pending_input_epoch = input_epoch
                self._pending_merge_next = False
                self._pending_coalesced = True
                extend = (
                    self._c.hold_sec if self._coalescer.should_hold(merged) else 0.0
                )
                self._deadline = min(
                    now + extend, self._hold_started + self._c.max_hold_sec
                )
                if extend > 0.0:
                    self._note_hold()
            else:
                # A complete final waiting only for preprocessing capacity is
                # not a fragment. Newest input supersedes it; concatenating two
                # complete utterances would manufacture a query the user never made.
                self.superseded_count += 1
                self._pending = text
                self._pending_submitted_at = submitted_at
                self._pending_input_generation = input_generation
                self._pending_input_epoch = input_epoch
                self._pending_merge_next = False
                self._pending_coalesced = False
                self._hold_started = now
                if self._c.enabled and self._coalescer.should_hold(text):
                    self.held_count += 1
                    self._deadline = min(
                        now + self._c.hold_sec,
                        self._hold_started + self._c.max_hold_sec,
                    )
                    self._note_hold()
                else:
                    self._deadline = now
            self._cv.notify()

    def note_partial(self) -> None:
        """The user resumed speaking while a final is held: keep holding until
        their next final lands (bounded by ``max_hold_sec``)."""
        with self._cv:
            if not self._running:
                return
            if not self._c.enabled:
                # A dispatcher may exist solely to move LLM preprocessing off
                # the audio thread. Without turn merging, resumed speech simply
                # retires the premature final and waits for the next one.
                if (
                    self._cancellable
                    and self._active_lease is not None
                    and not self._active_lease._committed
                ):
                    self._active_lease.cancel_event.set()
                self._pending = None
                self._pending_submitted_at = None
                self._pending_input_generation = None
                self._pending_input_epoch = None
                self._pending_merge_next = False
                self._pending_coalesced = False
                self._cv.notify_all()
                return
            if (
                self._cancellable
                and self._pending is None
                and self._active_lease is not None
                and not self._active_lease._committed
                and not self._active_lease.cancel_event.is_set()
                and self._active_text
            ):
                # Speech resumed before preprocessing committed. Retire that
                # premature final and put its text back into the bounded hold so
                # the next final can merge with it instead of losing the prefix.
                self._active_lease.cancel_event.set()
                self._pending = self._active_text
                self._pending_submitted_at = self._active_lease.submitted_at
                self._pending_input_generation = self._active_lease.input_generation
                self._pending_input_epoch = self._active_lease.input_epoch
                self._pending_merge_next = True
                self._pending_coalesced = self._active_lease.coalesced
                now = time.monotonic()
                self._deadline = min(
                    now + self._c.hold_sec,
                    self._hold_started + self._c.max_hold_sec,
                )
                self.held_count += 1
                self._note_hold()
            if self._pending is None:
                return
            self._pending_merge_next = True
            self._deadline = min(
                time.monotonic() + self._c.hold_sec,
                self._hold_started + self._c.max_hold_sec,
            )
            self._cv.notify()

    def cancel_pending(self) -> None:
        """Retire all pre-task work without blocking the caller.

        Used by barge-in/stop control paths before they cut playback. Setting a
        committed lease is still useful to downstream retirement checks; the
        commit itself remains the point after which already-applied effects
        cannot be rolled back.
        """
        with self._cv:
            if self._active_lease is not None:
                self._active_lease.cancel_event.set()
            self._active_text = None
            self._pending = None
            self._pending_submitted_at = None
            self._pending_input_generation = None
            self._pending_input_epoch = None
            self._pending_merge_next = False
            self._pending_coalesced = False
            self._cv.notify_all()

    def _note_hold(self) -> None:
        if self._on_hold is None:
            return
        try:
            self._on_hold()
        except Exception:  # noqa: BLE001 - diagnostics must not break dispatch
            log.exception("final hold callback failed")

    def flush(self) -> None:
        """Dispatch any held final immediately (e.g. before shutdown)."""
        with self._cv:
            self._deadline = 0.0
            self._cv.notify()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the worker; reject a concurrent restart until stop returns."""
        with self._lifecycle_lock:
            with self._cv:
                self._stop_in_progress = True
            try:
                self._stop_once(timeout)
            finally:
                with self._cv:
                    self._stop_in_progress = False
                    self._cv.notify_all()

    def _stop_once(self, timeout: float) -> None:
        """One serialized stop pass."""
        if self._cancellable:
            with self._cv:
                self._running = False
                if (
                    self._active_lease is not None
                    and not self._active_lease._committed
                ):
                    self._active_lease.cancel_event.set()
                # Runtime shutdown must not start another potentially blocking
                # LLM preprocessor. Uncommitted speech is retired with the session.
                self._pending = None
                self._pending_submitted_at = None
                self._pending_input_generation = None
                self._pending_input_epoch = None
                self._pending_merge_next = False
                self._pending_coalesced = False
                self._cv.notify_all()
            thread = self._thread
            if thread is not None:
                thread.join(timeout=timeout)
                if not thread.is_alive() and self._thread is thread:
                    self._thread = None
            return
        self.flush()
        with self._cv:
            self._running = False
            self._cv.notify()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
            if not thread.is_alive() and self._thread is thread:
                self._thread = None
        # The worker exited; if a straggler final is still pending (raced the
        # shutdown), dispatch it synchronously so the user's words aren't lost.
        with self._cv:
            text, self._pending = self._pending, None
            self._pending_submitted_at = None
            self._pending_input_generation = None
            self._pending_input_epoch = None
            self._pending_merge_next = False
            self._pending_coalesced = False
        if text is not None:
            try:
                self._dispatch(text)
            except Exception:  # noqa: BLE001 - shutdown path, never raise
                log.exception("final dispatch failed during stop()")

    def _run(self) -> None:
        if self._cancellable:
            self._run_cancellable()
            return
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
                self._pending_submitted_at = None
                self._pending_input_generation = None
                self._pending_input_epoch = None
                self._pending_merge_next = False
                self._pending_coalesced = False
                self._dispatching = True
            try:
                self._dispatch(text)
            except Exception:  # noqa: BLE001 - a turn must never kill the worker
                log.exception("final dispatch raised; turn dropped")
            finally:
                with self._cv:
                    self._dispatching = False
                    self._cv.notify_all()

    def _claim_lease_commit(self, lease: FinalDispatchLease) -> bool:
        """Atomically linearize terminal effects against a newer submission."""
        with self._cv:
            if (
                not self._running
                or self._active_lease is not lease
                or lease.cancel_event.is_set()
                or lease._committed
            ):
                return False
            lease._committed = True
            return True

    def _lease_can_start(self, lease: FinalDispatchLease) -> bool:
        with self._cv:
            return (
                self._running
                and self._active_lease is lease
                and not lease.cancel_event.is_set()
            )

    def _run_cancellable(self) -> None:
        """Coordinate bounded provider threads while newest input stays live."""
        while True:
            with self._cv:
                while self._running and self._pending is None:
                    self._cv.wait()
                if not self._running:
                    return
                now = time.monotonic()
                if now < self._deadline:
                    self._cv.wait(timeout=self._deadline - now)
                    continue
                text, self._pending = self._pending, None
                submitted_at, self._pending_submitted_at = (
                    self._pending_submitted_at,
                    None,
                )
                input_generation, self._pending_input_generation = (
                    self._pending_input_generation,
                    None,
                )
                input_epoch, self._pending_input_epoch = (
                    self._pending_input_epoch,
                    None,
                )
                merge_next, self._pending_merge_next = (
                    self._pending_merge_next,
                    False,
                )
                coalesced, self._pending_coalesced = (
                    self._pending_coalesced,
                    False,
                )
                self._next_generation += 1
                lease = FinalDispatchLease(
                    self,
                    self._next_generation,
                    merge_next=merge_next,
                    coalesced=coalesced,
                    submitted_at=submitted_at,
                    input_generation=input_generation,
                    input_epoch=input_epoch,
                )
                self._active_lease = lease
                self._active_text = text
                self._dispatching = True

            acquired = False
            while not acquired:
                if not self._lease_can_start(lease):
                    break
                acquired = self._dispatch_slots.acquire(
                    timeout=_DISPATCH_CANCEL_POLL_SEC
                )
            if not acquired or not self._lease_can_start(lease):
                if acquired:
                    self._dispatch_slots.release()
                with self._cv:
                    if self._active_lease is lease:
                        self._active_lease = None
                        self._active_text = None
                        self._dispatching = False
                        self._cv.notify_all()
                continue

            done = threading.Event()

            def run_provider(
                provider_text: str,
                provider_lease: FinalDispatchLease,
                provider_done: threading.Event,
            ) -> None:
                from .llm import LLMCallCancelled, capability_context

                context = dict(capability_context.get())
                context["cancel_event"] = provider_lease.cancel_event
                context["final_dispatch_generation"] = provider_lease.generation
                context_token = capability_context.set(context)
                try:
                    if not self._lease_can_start(provider_lease):
                        return
                    self._dispatch(provider_text, provider_lease)
                except LLMCallCancelled:
                    # Normal newest-input/shutdown control flow.
                    pass
                except Exception:  # noqa: BLE001 - one final must not kill dispatch
                    log.exception("final dispatch raised; turn dropped")
                finally:
                    capability_context.reset(context_token)
                    self._dispatch_slots.release()
                    provider_done.set()
                    with self._cv:
                        self._cv.notify_all()

            provider = threading.Thread(
                target=run_provider,
                args=(text, lease, done),
                name=f"speaker-final-provider-{lease.generation}",
                daemon=True,
            )
            try:
                provider.start()
            except BaseException:  # noqa: BLE001 - keep the coordinator alive
                self._dispatch_slots.release()
                log.exception("failed to start final provider thread")
                done.set()

            with self._cv:
                while (
                    not done.is_set()
                    and (
                        lease._committed
                        or not lease.cancel_event.is_set()
                    )
                    and (self._running or lease._committed)
                ):
                    self._cv.wait(timeout=_DISPATCH_CANCEL_POLL_SEC)
                if not self._running and not lease._committed:
                    lease.cancel_event.set()
                if self._active_lease is lease:
                    self._active_lease = None
                    self._active_text = None
                    self._dispatching = False
                    self._cv.notify_all()
