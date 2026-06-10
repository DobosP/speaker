"""Resume-after-interrupt + self-echo text guard (owner feedback 2026-06-10).

Two live failures from the same run (run-20260610-124002):

* **"Start again" must RESUME, not reset.** The owner stopped a story mid-way
  and said "start again" expecting the story to CONTINUE from where it was cut;
  the assistant greeted them fresh instead. :class:`ResumeTracker` remembers the
  current turn's originating query + the sentences that were actually SPOKEN
  (not merely generated), and when a resume phrase arrives after a cut it
  synthesizes a "continue from where you stopped" prompt for the brain.
* **The assistant answered its own echo.** The transcript shows the assistant's
  own TTS coming back as a *user* turn verbatim ("Okay, let's begin. How can I
  help you today?") right after playback ended -- the echo TAIL reaches the
  recognizer once ``_speaking`` clears, and at higher speaker volume it beats
  the energy floor (L1). :meth:`ResumeTracker.is_self_echo` is the L4 guard the
  energy layers cannot provide: a final that arrives shortly after playback and
  reads ~like a recently spoken sentence is the assistant hearing ITSELF --
  volume-independent, pure text. A user genuinely repeating the assistant's
  words outside that short window is unaffected.

Pure logic + a lock (called from the audio, bus, and dispatcher threads); the
runtime wires it at ``_process_final`` / ``TTS_REQUEST`` / barge-stop /
``on_speech_end``. Config block ``resume`` (default ON -- cheap, local,
deterministic; ``enabled``/``echo_guard_enabled`` switch each half off).
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Mapping, Optional

from always_on_agent.text import normalize_text

# Short utterances that mean "pick up where you left off". Only consulted when
# the previous reply was actually CUT (barge/stop) -- otherwise they pass
# through to the brain unchanged ("continue" mid-conversation is a normal
# request). EN + RO (de-diacritic'd to match normalize_text).
DEFAULT_RESUME_PHRASES: tuple[str, ...] = (
    "start again", "start over", "continue", "go on", "keep going", "resume",
    "carry on", "keep talking", "finish the story", "continue the story",
    "where were you",
    # Romanian
    "continua", "mai departe", "zi mai departe", "spune mai departe",
    "de unde ai ramas",
)

DEFAULT_RESUME_TEMPLATE = (
    'The user asked: "{query}". You were answering aloud and were INTERRUPTED; '
    'the last words you actually spoke were: "...{spoken_tail}". The user now '
    "asked you to continue. Pick up the SAME answer exactly where it stopped -- "
    "do not start over, do not repeat what was already said, do not greet."
)

# Control words the FUZZY echo rule must never eat: "no" char-matches "now" at
# 0.8, so a denial within the echo window of a reply ending "...now" would be
# dropped. These only count as echo via an EXACT last-sentence match. EN + RO.
_ECHO_FUZZY_EXEMPT: frozenset[str] = frozenset({
    "yes", "no", "yeah", "yep", "nope", "ok", "okay", "sure", "stop", "what",
    "why", "how", "da", "nu", "bine", "ce", "cum",
})


@dataclass
class ResumeConfig:
    """The ``resume`` config block. Dataclass defaults OFF so programmatic
    construction (tests, the bench harness) is byte-identical; the shipped
    ``config.json`` opts both halves in. NB the echo guard MUST default off
    here: an LLM reply often embeds the user's words verbatim (EchoLLM always
    does), and a default-on text guard would eat legitimate repeat queries in
    harnesses that never see real echo."""

    enabled: bool = False
    resume_phrases: tuple[str, ...] = field(default=DEFAULT_RESUME_PHRASES)
    template: str = DEFAULT_RESUME_TEMPLATE
    # How much of the spoken tail the continue-prompt embeds.
    spoken_tail_chars: int = 360
    # Cap on tracked spoken text per turn (memory bound; the tail is what matters).
    max_spoken_chars: int = 2000
    # L4 self-echo guard: a final within echo_window_sec of playback end whose
    # normalized tokens overlap a recently spoken sentence >= min_overlap is the
    # assistant's own echo -> dropped. NB the window must cover the FINAL's
    # arrival, not the echo audio itself: the recognizer adds endpoint trailing
    # silence (~1-2s) plus the utterance length before the final fires, so a 3s
    # window missed the recorded verbatim echo finals ("Okay, let's begin. How
    # can I help you today?" landed ~4s after playback end). 8s default.
    echo_guard_enabled: bool = False
    echo_window_sec: float = 8.0
    echo_min_overlap: float = 0.75
    # Don't echo-drop very short finals on overlap alone ("yes" overlaps lots);
    # they must read as the LAST spoken sentence's tail -- exactly, or fuzzily
    # (the recognizer garbles echo: live 'Leleep.' from "...spin and leap",
    # 'Loly.' from "lovely"). Char-level similarity vs the tail tokens.
    # Measured on the recorded garbles: 'wecome'/'welcome'=0.92,
    # 'loly'/'lovely'=0.8, 'leleep'/'leap'=0.60 -- 0.6 catches them all, while
    # real reactions stay clear ('great'/'hear'=0.44) and control words are
    # exempt outright.
    echo_min_words: int = 3
    echo_fuzzy_ratio: float = 0.6
    echo_tail_tokens: int = 8

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, object]]) -> "ResumeConfig":
        data = data or {}
        phrases = data.get("resume_phrases")
        return cls(
            enabled=bool(data.get("enabled", False)),
            resume_phrases=(
                tuple(str(p) for p in phrases)
                if isinstance(phrases, (list, tuple))
                else DEFAULT_RESUME_PHRASES
            ),
            template=str(data.get("template") or DEFAULT_RESUME_TEMPLATE),
            spoken_tail_chars=int(data.get("spoken_tail_chars", 360) or 360),
            max_spoken_chars=int(data.get("max_spoken_chars", 2000) or 2000),
            echo_guard_enabled=bool(data.get("echo_guard_enabled", False)),
            echo_window_sec=float(data.get("echo_window_sec", 8.0) or 8.0),
            echo_min_overlap=float(data.get("echo_min_overlap", 0.75) or 0.75),
            echo_min_words=int(data.get("echo_min_words", 3) or 3),
            echo_fuzzy_ratio=float(data.get("echo_fuzzy_ratio", 0.6) or 0.6),
            echo_tail_tokens=int(data.get("echo_tail_tokens", 8) or 8),
        )


def _fuzzy_eq(a: str, b: str, ratio: float) -> bool:
    """Char-level similarity match (difflib) -- echo gets GARBLED by the
    recognizer ('Leleep' from 'leap', 'Wecome' from 'welcome'), so exact token
    equality misses real echo. ``ratio <= 0`` disables fuzziness."""
    if a == b:
        return True
    if ratio <= 0.0:
        return False
    from difflib import SequenceMatcher

    return SequenceMatcher(None, a, b).ratio() >= ratio


def _token_overlap(a_tokens: list[str], b_tokens: list[str], *, fuzzy: float = 0.0) -> float:
    """Fraction of ``a``'s tokens found in ``b`` (multiset containment),
    optionally counting fuzzy char-level matches (garbled echo)."""
    if not a_tokens:
        return 0.0
    pool = list(b_tokens)
    hit = 0
    for t in a_tokens:
        match_i = next((i for i, p in enumerate(pool) if _fuzzy_eq(t, p, fuzzy)), None)
        if match_i is not None:
            pool.pop(match_i)
            hit += 1
    return hit / len(a_tokens)


class ResumeTracker:
    """Tracks what the assistant is saying so the runtime can (a) RESUME a cut
    reply on request and (b) DROP its own echo when the mic feeds it back.

    Thread-safe (one lock; every method is a short critical section). The
    runtime calls:

    * ``note_query(text)``    -- a normal final was dispatched (a new turn)
    * ``note_spoken(text)``   -- a TTS sentence was handed to the engine
    * ``note_playback_end()`` -- the engine reported speech end (echo anchor)
    * ``note_cut()``          -- barge-in / stop interrupted playback
    * ``is_self_echo(text)``  -- L4 guard at the final seam
    * ``resume_prompt(text)`` -- non-None iff ``text`` is a resume request for
                                 a turn that was genuinely cut
    """

    def __init__(self, config: Optional[ResumeConfig] = None) -> None:
        self._c = config or ResumeConfig()
        self._phrases = frozenset(
            n for p in self._c.resume_phrases if (n := normalize_text(p))
        )
        self._lock = threading.Lock()
        self._query: str = ""
        self._spoken: str = ""
        self._sentences: deque[str] = deque(maxlen=12)  # recent spoken sentences
        self._cut = False
        self._playback_end_at: float = 0.0
        # Diagnostics
        self.echo_dropped = 0
        self.resumed = 0

    # --- writers (audio/bus/dispatcher threads) -------------------------------

    def note_query(self, text: str) -> None:
        with self._lock:
            self._query = text
            self._spoken = ""
            self._cut = False

    def note_spoken(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            self._sentences.append(text)
            if len(self._spoken) < self._c.max_spoken_chars:
                self._spoken = (self._spoken + " " + text).strip()

    def note_playback_end(self) -> None:
        with self._lock:
            self._playback_end_at = time.monotonic()

    def note_cut(self) -> None:
        with self._lock:
            if self._spoken:  # a cut with nothing spoken is not a resumable turn
                self._cut = True

    # --- readers (the final-dispatch seam) -------------------------------------

    def is_self_echo(self, text: str) -> bool:
        """True iff ``text`` is (almost certainly) the assistant's own TTS echo:
        it arrived within ``echo_window_sec`` of playback ending AND reads like
        a recently spoken sentence (or their concatenation)."""
        if not self._c.echo_guard_enabled:
            return False
        tokens = normalize_text(text).split()
        if not tokens:
            return False
        with self._lock:
            in_window = (
                self._playback_end_at > 0.0
                and time.monotonic() - self._playback_end_at <= self._c.echo_window_sec
            )
            if not in_window:
                return False
            recent = list(self._sentences)
        if not recent:
            return False
        spoken_tokens = normalize_text(" ".join(recent)).split()
        # EXACT token overlap for multi-word finals: an acoustic echo of a full
        # sentence comes back near-verbatim, while a garbled USER request can
        # share many fuzzy-similar words with a long recent reply -- live, the
        # owner's "Tell me a long story about my gun [cat]" was wrongly eaten
        # when this used fuzzy matching. Fuzziness belongs only to the
        # short-tail branch below (where it is bounded to the LAST sentence).
        overlap = _token_overlap(tokens, spoken_tokens)
        if len(tokens) < self._c.echo_min_words:
            # A tiny final counts as echo only against the LAST spoken
            # sentence's TAIL -- the echo tail is the END of playback. Exact
            # whole-sentence match, or EVERY final token fuzzily matching a
            # tail token (live: 'Leleep.' from "...spin and leap", 'Loly.'
            # from "That sounds lovely!"). Matching any recent sentence (or
            # plain overlap) would eat a legitimate short follow-up that
            # parrots a word from earlier in the reply.
            joined = " ".join(tokens)
            last_tokens = normalize_text(recent[-1]).split()
            tail = last_tokens[-max(1, self._c.echo_tail_tokens):]
            fuzz_ok = not all(t in _ECHO_FUZZY_EXEMPT for t in tokens)
            hit = joined == " ".join(last_tokens) or (
                fuzz_ok
                and self._c.echo_fuzzy_ratio > 0.0
                and all(
                    any(_fuzzy_eq(t, p, self._c.echo_fuzzy_ratio) for p in tail)
                    for t in tokens
                )
            )
        else:
            hit = overlap >= self._c.echo_min_overlap
        if hit:
            with self._lock:
                self.echo_dropped += 1
        return hit

    def resume_prompt(self, text: str) -> Optional[str]:
        """The synthesized continue-prompt iff ``text`` is a resume request and
        the previous reply was CUT mid-way; ``None`` otherwise (normal flow)."""
        if not self._c.enabled:
            return None
        if normalize_text(text) not in self._phrases:
            return None
        with self._lock:
            if not (self._cut and self._query and self._spoken):
                return None
            tail = self._spoken[-self._c.spoken_tail_chars:]
            query = self._query
            self._cut = False  # consumed: a second "continue" continues the NEW turn
            self.resumed += 1
        return self._c.template.format(query=query, spoken_tail=tail)
