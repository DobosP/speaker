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


def match_onnx_input_shape(feats, expected_shape):
    """Adapt a ``(1, n_mels, frames)`` feature tensor to the ONNX graph's declared
    input layout. The Whisper extractor emits mels-then-frames; some Smart Turn
    exports declare ``(1, frames, n_mels)`` instead, so transpose the last two
    axes when the model pins a concrete final dim that matches our mel axis.
    Pure + numpy-free in signature (works on any object with ``.ndim``/``.shape``/
    ``.transpose``) so it is unit-testable without a model. Dynamic axes (non-int,
    e.g. a symbolic batch like ``'s6'``) are left untouched."""
    if getattr(feats, "ndim", None) != 3 or len(expected_shape) != 3:
        return feats
    last = expected_shape[-1]
    if isinstance(last, int) and feats.shape[-1] != last and feats.shape[1] == last:
        return feats.transpose(0, 2, 1)
    return feats


class SmartTurnCompletionDetector:
    """Prosodic turn-completion via the Smart Turn v3 ONNX model (pipecat-ai).

    Audio-only (~8M params, ~8.7 MB ONNX, ~12 ms CPU): 16 kHz mono, up to 8 s of
    the most-recent utterance audio, a Whisper-tiny log-mel front-end + a sigmoid
    head returning P(the turn is complete). Heavier deps (``onnxruntime`` + the
    Whisper feature extractor) are imported LAZILY in ``__init__`` so merely
    importing this module costs the default/phone build nothing -- the detector is
    only constructed when ``sherpa.smart_turn_enabled`` + a model path are set
    (see ``core/engines/sherpa.py``). A load failure raises here (the engine
    catches it and falls back to the lexical detector); a per-call inference error
    is caught by the engine's ``_decide_endpoint`` try/except -> the acoustic
    endpoint, so a bad block can never break capture.

    Preprocessing follows pipecat-ai/smart-turn-v3 ``inference.py`` exactly
    (the canonical extractor, not a hand-rolled mel): ``WhisperFeatureExtractor(
    chunk_length=8)``, pad to ``8*16000`` samples, ``do_normalize=True``; ONNX
    input ``input_features``; ``output[0][0]`` is the sigmoid probability. The
    extractor's native ``(1, n_mels, frames)`` tensor is adapted to whatever the
    loaded ONNX graph declares (some exports want ``(1, frames, n_mels)``)."""

    needs_audio = True
    SAMPLE_RATE = 16000
    MAX_SAMPLES = 8 * 16000  # Smart Turn judges <=8s of the turn's tail

    def __init__(self, model_path: str, *, providers: Optional[Sequence[str]] = None) -> None:
        import numpy as np  # local: keep the module import numpy-free for phone
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor

        self._np = np
        self._extractor = WhisperFeatureExtractor(chunk_length=8)
        self._session = ort.InferenceSession(
            model_path, providers=list(providers) if providers else ["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        # Declared dims (symbolic strings/None for dynamic axes) so we can adapt
        # the extractor output to a (1, frames, n_mels) export if one needs it.
        self._input_shape = tuple(self._session.get_inputs()[0].shape)

    def completion_score(
        self, text: str, *, samples: Optional[Sequence[float]] = None, sample_rate: int = 16000
    ) -> float:
        np = self._np
        if samples is None:
            return 0.5  # no audio assembled -> neutral; policy uses the acoustic decision
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if audio.shape[0] == 0:
            return 0.5
        if audio.shape[0] > self.MAX_SAMPLES:
            audio = audio[-self.MAX_SAMPLES :]
        feats = self._extractor(
            audio,
            sampling_rate=self.SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=self.MAX_SAMPLES,
            truncation=True,
            do_normalize=True,
        )["input_features"]
        feats = np.asarray(feats, dtype=np.float32)
        feats = match_onnx_input_shape(feats, self._input_shape)
        out = self._session.run(None, {self._input_name: feats})
        return float(np.asarray(out[0]).reshape(-1)[0])


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
