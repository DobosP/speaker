"""Semantic turn-completion endpointing.

The endpoint decision -- "has the user finished their turn?" -- was a single
fixed acoustic timer: commit a final after ``asr_rule2_min_trailing_silence``
(0.8s) of silence, identical on every device and blind to WHAT was said. That
cuts off a slow speaker mid-thought and adds a fixed ~0.8s tail to a turn that
obviously ended.

This adds a pluggable *turn-completion detector* + an *adaptive policy* layered
on top of the acoustic timer:

* when the partial clearly reads as a COMPLETE turn, commit EARLY (short silence)
  -- the latency win;
* when it ends mid-phrase ("...and", "...the", "...I want to"), HOLD past the
  acoustic timer (bounded) so the pause isn't mistaken for the end;
* otherwise fall back to the unchanged acoustic decision.

The shipped v1 detector is :class:`LexicalTurnCompletionDetector` -- cheap,
deterministic, no model, no audio work on the capture thread. The
:class:`TurnCompletionDetector` protocol also takes the recent audio, so a
prosodic audio model (e.g. Pipecat Smart Turn v3, ~8MB ONNX) can drop in through
the same seam later. A premature early-endpoint degrades gracefully: the user's
continued words become a new final that the ADD-ON / continuation layer merges
back in.

Everything here is pure + deterministic; the engine integration is a single
``_decide_endpoint`` call (see ``core/engines/sherpa.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, runtime_checkable

from always_on_agent.text import normalize_text

# Words that, when they END an utterance, ALMOST CERTAINLY mean "more is coming".
# Deliberately CONSERVATIVE: only words that essentially never end a real
# sentence. Stranding-prone prepositions ("what are you waiting for"),
# object/terminal pronouns ("what time is it"), and auxiliaries ("here it is")
# are EXCLUDED -- a false "incomplete" would wrongly EXTEND a finished turn with
# no recovery, whereas a missed "incomplete" just falls back to the acoustic
# timer (and a too-early commit is merged back by the continuation layer). So we
# err toward "complete". EN + RO (de-diacritic'd to match normalize_text).
DEFAULT_INCOMPLETE_ENDINGS: frozenset[str] = frozenset({
    # coordinating conjunctions -- a sentence ~never ends on these
    "and", "but", "or", "nor",
    # subordinators that demand a following clause. NB "while"/"though" are
    # excluded: they're commonly terminal ("wait a while", "i'd like to though"),
    # and a false-incomplete wrongly EXTENDS a finished turn (non-recoverable).
    "because", "although", "unless", "whether",
    # articles / possessive determiners -- always precede a noun
    "the", "a", "an", "my", "your", "our", "their", "his", "its",
    # fillers
    "um", "uh", "er", "hmm",
    # Romanian (de-diacritic'd): and / or / but / for / to(subjunctive) / a-an
    "si", "sau", "dar", "pentru", "sa", "un", "o",
})


@runtime_checkable
class TurnCompletionDetector(Protocol):
    """Returns 0..1: how likely the utterance is a COMPLETE turn.

    ``text`` is the latest partial transcript; ``samples`` (optional) is the
    recent utterance audio for a prosodic model. An implementation uses whichever
    it needs and sets ``needs_audio`` so the engine only pays to assemble the
    audio buffer when a detector actually consumes it."""

    needs_audio: bool

    def completion_score(
        self, text: str, *, samples: Optional[Sequence[float]] = None, sample_rate: int = 16000
    ) -> float: ...


class LexicalTurnCompletionDetector:
    """Cheap, deterministic turn-completion from the partial TEXT alone.

    No model, no audio: a normalized last-word check. An utterance ending on a
    conjunction/preposition/article/filler scores LOW (mid-thought); a few words
    not ending that way score HIGH (likely complete); a very short utterance is
    left mid (too little to be confident)."""

    needs_audio = False  # text-only; the engine skips assembling the audio buffer

    def __init__(
        self,
        *,
        min_words: int = 3,
        incomplete_endings: Optional[frozenset[str]] = None,
        complete_score: float = 0.75,
        incomplete_score: float = 0.1,
        short_score: float = 0.4,
    ) -> None:
        self._min_words = max(1, int(min_words))
        self._incomplete = incomplete_endings or DEFAULT_INCOMPLETE_ENDINGS
        self._complete = complete_score
        self._incomplete_score = incomplete_score
        self._short = short_score

    def completion_score(
        self, text: str, *, samples: Optional[Sequence[float]] = None, sample_rate: int = 16000
    ) -> float:
        words = normalize_text(text).split()
        if not words:
            return 0.5
        if words[-1] in self._incomplete:
            return self._incomplete_score
        if len(words) < self._min_words:
            return self._short
        return self._complete


class ScriptedTurnCompletionDetector:
    """Test fake: maps partial text -> score (default 0.5). Records ``.calls``."""

    needs_audio = False

    def __init__(self, mapping: Optional[dict[str, float]] = None, *, default: float = 0.5) -> None:
        self._mapping = dict(mapping or {})
        self._default = default
        self.calls: list[str] = []

    def completion_score(
        self, text: str, *, samples: Optional[Sequence[float]] = None, sample_rate: int = 16000
    ) -> float:
        self.calls.append(text)
        return self._mapping.get(text, self._default)


@dataclass(frozen=True)
class EndpointConfig:
    """Adaptive-endpoint thresholds (the ``sherpa`` endpoint_* config fields).

    ``enabled`` default false -> the engine uses the pure acoustic decision and
    behaviour is byte-identical."""

    enabled: bool = False
    # Earliest trailing silence at which a confident-complete turn may end early.
    # MUST exceed the streaming decoder's right-context/lookahead latency, or the
    # early commit clips the last word(s) the decoder hasn't emitted yet (and the
    # following reset discards them irrecoverably). 0.5s is a safe default for a
    # typical streaming zipformer; validate against your model on device.
    min_silence_sec: float = 0.5
    # Cap on how long the "looks mid-phrase, hold on" extension may suppress the
    # acoustic endpoint -- a hard floor so a mis-scored turn still commits.
    max_silence_sec: float = 1.6
    complete_threshold: float = 0.6
    incomplete_threshold: float = 0.3

    @classmethod
    def from_sherpa(cls, c: object) -> "EndpointConfig":
        return cls(
            enabled=bool(getattr(c, "endpoint_enabled", False)),
            min_silence_sec=float(getattr(c, "endpoint_min_silence_sec", 0.2)),
            max_silence_sec=float(getattr(c, "endpoint_max_silence_sec", 1.6)),
            complete_threshold=float(getattr(c, "endpoint_complete_threshold", 0.6)),
            incomplete_threshold=float(getattr(c, "endpoint_incomplete_threshold", 0.3)),
        )


class AdaptiveEndpointPolicy:
    """Combines the acoustic endpoint, a completion score, and trailing silence
    into one endpoint decision. Pure + deterministic."""

    def __init__(self, config: Optional[EndpointConfig] = None) -> None:
        self._c = config or EndpointConfig()

    def decide(self, *, acoustic_endpoint: bool, completion_score: float, silence_sec: float) -> bool:
        c = self._c
        # SHORTEN: the turn clearly reads as complete and we've had at least the
        # minimum settle -> commit now, ahead of the acoustic timer.
        if completion_score >= c.complete_threshold and silence_sec >= c.min_silence_sec:
            return True
        # EXTEND (bounded): the acoustic timer wants to commit, but the partial
        # ends mid-phrase and we haven't waited too long -> hold for more speech.
        if (
            acoustic_endpoint
            and completion_score <= c.incomplete_threshold
            and silence_sec < c.max_silence_sec
        ):
            return False
        # Otherwise the acoustic decision stands (the safe default / hard backstop).
        return acoustic_endpoint
