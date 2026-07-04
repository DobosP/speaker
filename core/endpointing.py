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

from collections import deque
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


def _slaney_mel_filters(n_freqs: int = 201, n_mels: int = 80,
                        fmin: float = 0.0, fmax: float = 8000.0, sr: int = 16000):
    """The Slaney mel filterbank Whisper uses, in pure numpy. Verified bit-exact
    (<1e-15) against transformers WhisperFeatureExtractor.mel_filters, so the
    prosody features match the model's training preprocessing without pulling in
    transformers/torch. Returns (n_freqs, n_mels)."""
    import numpy as np

    fft_freqs = np.linspace(0, sr / 2, n_freqs)
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0

    def h2m(f):
        f = np.asarray(f, float)
        return np.where(f >= min_log_hz, min_log_mel + np.log(f / min_log_hz) / logstep, f / f_sp)

    def m2h(m):
        m = np.asarray(m, float)
        return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f_sp * m)

    mel_pts = np.linspace(h2m(fmin), h2m(fmax), n_mels + 2)
    freq_pts = m2h(mel_pts)
    fdiff = np.diff(freq_pts)
    ramps = freq_pts[:, None] - fft_freqs[None, :]
    weights = np.zeros((n_mels, n_freqs))
    for i in range(n_mels):
        weights[i] = np.maximum(0.0, np.minimum(-ramps[i] / fdiff[i], ramps[i + 2] / fdiff[i + 1]))
    enorm = 2.0 / (freq_pts[2:n_mels + 2] - freq_pts[:n_mels])
    weights *= enorm[:, None]
    return weights.T.astype("float64")  # (n_freqs, n_mels)


class ProsodyTurnCompletionDetector:
    """Audio turn-completion from PROSODY via the Smart Turn v3 ONNX
    (pipecat-ai/smart-turn-v3, BSD-2, ~8MB, ~9ms CPU). Returns P(turn complete)
    from the recent audio waveform -- catching the rising/sustained intonation of
    a mid-thought trailing-off that the lexical detector (text-only) cannot, so
    the endpoint floor can drop further without splitting.

    The feature extraction is the Whisper log-mel (80 mels x 800 frames over the
    last 8s @ 16 kHz), reimplemented in pure numpy and verified bit-exact against
    transformers' WhisperFeatureExtractor -- no transformers/torch dependency. The
    ONNX output is read directly as the completion probability (the upstream
    contract). Validated on the user's REAL voice 2026-06-01: complete turns
    0.74-0.98, incomplete 0.01-0.56 (margin 0.18). NB the model is human-audio
    only -- it returns a flat ~0.97 on TTS, so it cannot be validated with the
    synthetic-user harness; validate on real speech.
    """

    needs_audio = True  # the engine assembles the utterance audio for this detector

    def __init__(self, model_path: str, *, num_threads: int = 1, min_audio_sec: float = 0.3) -> None:
        self._model_path = model_path
        self._num_threads = max(1, int(num_threads))
        self._min_audio_sec = float(min_audio_sec)
        self._session = None
        self._mel = None  # (201, 80), lazy

    def _ensure(self):
        if self._session is None:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = self._num_threads
            opts.inter_op_num_threads = 1
            self._session = ort.InferenceSession(
                self._model_path, sess_options=opts, providers=["CPUExecutionProvider"]
            )
            self._mel = _slaney_mel_filters()

    def _logmel(self, samples, sample_rate: int):
        """Whisper log-mel (1, 80, 800) for the last 8s of ``samples``. Pure numpy,
        bit-exact to transformers WhisperFeatureExtractor(chunk_length=8,
        do_normalize=True)."""
        import numpy as np

        a = np.asarray(samples, dtype="float64").reshape(-1)
        if sample_rate != 16000 and a.size:
            idx = np.linspace(0, a.size - 1, int(round(a.size * 16000 / sample_rate)))
            a = np.interp(idx, np.arange(a.size), a)
        n = 8 * 16000
        a = a[-n:]
        a = (a - a.mean()) / np.sqrt(a.var() + 1e-7)  # do_normalize (zero-mean unit-var)
        a = np.pad(a, (0, n - a.size)) if a.size < n else a[:n]
        a = np.pad(a, (200, 200), mode="reflect")     # center pad n_fft//2
        win = np.hanning(401)[:-1]                     # periodic hann
        # Vectorized framing (stride tricks) -- a Python per-frame loop here is
        # ~40ms; this keeps the detector ONNX-bound (~10ms).
        from numpy.lib.stride_tricks import sliding_window_view

        frames = sliding_window_view(a, 400)[::160] * win  # (nfr, 400)
        spec = np.abs(np.fft.rfft(frames, n=400, axis=1)) ** 2  # power (nfr, 201)
        mel = spec @ self._mel                                   # (nfr, 80)
        lm = np.log10(np.clip(mel, 1e-10, None)).T              # (80, nfr)
        lm = lm[:, :-1]                                          # transformers drops the last frame
        lm = np.maximum(lm, lm.max() - 8.0)
        lm = (lm + 4.0) / 4.0
        return lm[None].astype("float32")

    def completion_score(self, text: str, *, samples=None, sample_rate: int = 16000) -> float:
        """P(turn complete) from the audio. Returns 0.5 (no opinion -> the policy
        falls back to the acoustic decision) when there is too little audio to
        judge; raises on a real error (the engine catches it and uses acoustic)."""
        import numpy as np

        if samples is None:
            return 0.5
        a = np.asarray(samples, dtype="float32").reshape(-1)
        if a.size < int(self._min_audio_sec * sample_rate):
            return 0.5
        self._ensure()
        out = self._session.run(None, {"input_features": self._logmel(a, sample_rate)})[0]
        return float(np.asarray(out).reshape(-1)[0])


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


# Linguistic floor (NOT per-machine): a gap under ~0.2s is a within-word / breath
# micro-pause, never a real inter-phrase pause -- feeding it would drag the learned
# floor toward zero and re-open the chopping. A universal speech-timing bound, not
# a tuned operating point.
_MIN_PAUSE_SEC: float = 0.2


class SessionPauseModel:
    """Learns THIS session's mid-utterance pause timing so the endpoint floor
    self-calibrates to the SPEAKER instead of a fixed silence number.

    The fixed ``min_silence`` floor over-fragments a speaker who pauses between
    phrases: one sentence "the mere story ... about ... my cat Biki" commits three
    times because each inter-phrase pause crosses the fixed timer. Feeding those
    mid-utterance gaps in here (via :meth:`observe_pause`) lets :meth:`floor`
    return a learned trailing-silence floor ``Lp`` -- a high quantile of the recent
    pauses plus a margin -- so the endpoint waits out THIS speaker's own pauses
    while still committing a clearly-complete turn promptly. No per-machine number:
    ``Lp`` is derived at runtime from observed gaps and clamped to physical bounds.

    Pure stdlib (a manual sorted-index percentile), deterministic, no numpy."""

    def __init__(
        self,
        *,
        window: int = 64,
        pause_quantile: float = 0.85,
        pause_margin: float = 0.15,
        min_samples: int = 8,
        floor_lo: float = 0.0,
        floor_hi: float = 1.6,
        cold_start_sec: float = 0.5,
    ) -> None:
        self._samples: deque[float] = deque(maxlen=max(1, int(window)))
        self._q = float(pause_quantile)
        self._margin = float(pause_margin)
        self._min_samples = max(1, int(min_samples))
        self._lo = float(floor_lo)
        self._hi = float(floor_hi)
        self._cold = float(cold_start_sec)

    def observe_pause(self, gap_sec: float) -> None:
        """Record a mid-utterance pause. Gaps below ``_MIN_PAUSE_SEC`` (micro-pauses)
        and above ``floor_hi`` (an end-of-turn silence, not a mid-phrase pause --
        feeding it would inflate the floor) are ignored."""
        g = float(gap_sec)
        if g < _MIN_PAUSE_SEC or g > self._hi:
            return
        self._samples.append(g)

    def _clamp(self, v: float) -> float:
        return min(max(v, self._lo), self._hi)

    def _quantile(self, q: float) -> float:
        """Linear-interpolated percentile on the sorted samples (numpy 'linear'
        method), done by hand to stay stdlib-only + deterministic."""
        xs = sorted(self._samples)
        n = len(xs)
        if n == 1:
            return xs[0]
        pos = q * (n - 1)
        lo = int(pos)
        if lo + 1 >= n:
            return xs[-1]
        return xs[lo] + (pos - lo) * (xs[lo + 1] - xs[lo])

    def floor(self) -> float:
        """The learned trailing-silence floor ``Lp``. Until ``min_samples`` pauses
        have been seen, the (clamped) cold-start floor; then a high quantile of the
        recent pauses plus a margin, clamped to ``[floor_lo, floor_hi]``."""
        if len(self._samples) < self._min_samples:
            return self._clamp(self._cold)
        return self._clamp(self._quantile(self._q) * (1.0 + self._margin))


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
    # Adaptive confidence-tiered SHORTEN floor. When > 0 (and below
    # ``min_silence_sec``), a HIGH-confidence completion -- ``completion_score >=
    # high_confidence_score``, i.e. the lexical detector's 0.75 bin: a normal
    # ending word, never a conjunction/article/filler -- commits at this LOWER
    # trailing silence instead of ``min_silence_sec``, reclaiming latency on the
    # common, well-formed-turn case. A medium-confidence complete (the short-
    # utterance 0.4 bin, or a custom score in [complete_threshold, high)) keeps the
    # full ``min_silence_sec`` floor. 0.0 (default) -> uniform ``min_silence_sec``,
    # byte-identical to before. The floor MUST still exceed the decoder lookahead
    # (or the early commit clips the last word) AND a typical intra-sentence comma
    # pause (or a run-on splits); validate on device before lowering.
    high_confidence_floor: float = 0.0
    high_confidence_score: float = 0.75
    # Adaptive learned-pause floor (FIX-4). When true, the SHORTEN/anti-chop floor
    # is the learned ``Lp`` from :class:`SessionPauseModel` (calibrated to THIS
    # speaker's own inter-phrase pauses) instead of the fixed ``min_silence_sec``.
    # The pause_* knobs configure that model; ``Lp`` is clamped to
    # ``[high_confidence_floor, max_silence_sec]`` and cold-starts at
    # ``min_silence_sec``. False (default) -> byte-identical to the fixed-floor
    # policy.
    adaptive_floor: bool = False
    pause_window: int = 64
    pause_quantile: float = 0.85
    pause_margin: float = 0.15
    pause_min_samples: int = 8

    @classmethod
    def from_sherpa(cls, c: object) -> "EndpointConfig":
        return cls(
            enabled=bool(getattr(c, "endpoint_enabled", False)),
            min_silence_sec=float(getattr(c, "endpoint_min_silence_sec", 0.2)),
            max_silence_sec=float(getattr(c, "endpoint_max_silence_sec", 1.6)),
            complete_threshold=float(getattr(c, "endpoint_complete_threshold", 0.6)),
            incomplete_threshold=float(getattr(c, "endpoint_incomplete_threshold", 0.3)),
            high_confidence_floor=float(getattr(c, "endpoint_high_confidence_floor", 0.0)),
            high_confidence_score=float(getattr(c, "endpoint_high_confidence_score", 0.75)),
            adaptive_floor=bool(getattr(c, "endpoint_adaptive_floor", False)),
            pause_window=int(getattr(c, "endpoint_pause_window", 64)),
            pause_quantile=float(getattr(c, "endpoint_pause_quantile", 0.85)),
            pause_margin=float(getattr(c, "endpoint_pause_margin", 0.15)),
            pause_min_samples=int(getattr(c, "endpoint_pause_min_samples", 8)),
        )


class AdaptiveEndpointPolicy:
    """Combines the acoustic endpoint, a completion score, and trailing silence
    into one endpoint decision. Pure + deterministic."""

    def __init__(self, config: Optional[EndpointConfig] = None) -> None:
        self._c = config or EndpointConfig()
        c = self._c
        # The learned trailing-silence floor Lp is clamped to the SAME physical
        # bounds the fixed policy uses: floor_lo = high_confidence_floor (the
        # decoder-lookahead guard), floor_hi = max_silence_sec (the hard backstop),
        # cold-starting at min_silence_sec until enough pauses are seen.
        self._pause = SessionPauseModel(
            window=c.pause_window,
            pause_quantile=c.pause_quantile,
            pause_margin=c.pause_margin,
            min_samples=c.pause_min_samples,
            floor_lo=c.high_confidence_floor,
            floor_hi=c.max_silence_sec,
            cold_start_sec=c.min_silence_sec,
        )

    def observe_pause(self, gap_sec: float) -> None:
        """Feed a mid-utterance pause to the session pause model (the capture loop
        calls this on each resume-after-gap). No-op for the floor until
        ``pause_min_samples`` gaps have been seen."""
        self._pause.observe_pause(gap_sec)

    def decide(self, *, acoustic_endpoint: bool, completion_score: float, silence_sec: float) -> bool:
        c = self._c
        if c.adaptive_floor:
            return self._decide_adaptive(
                acoustic_endpoint=acoustic_endpoint,
                completion_score=completion_score,
                silence_sec=silence_sec,
            )
        # SHORTEN: the turn clearly reads as complete and we've had at least the
        # minimum settle -> commit now, ahead of the acoustic timer. A HIGH-
        # confidence completion may use a LOWER floor when configured (still above
        # the decoder lookahead + comma pause) -- the adaptive latency win on the
        # common case; 0.0 disables it (uniform min_silence_sec).
        floor = c.min_silence_sec
        if c.high_confidence_floor > 0.0 and completion_score >= c.high_confidence_score:
            floor = c.high_confidence_floor
        if completion_score >= c.complete_threshold and silence_sec >= floor:
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

    def _decide_adaptive(self, *, acoustic_endpoint: bool, completion_score: float, silence_sec: float) -> bool:
        """Learned-floor variant (``adaptive_floor``). Identical tiers to
        :meth:`decide`, but the SHORTEN/anti-chop floor is the learned ``Lp``
        (this speaker's own pause timing) instead of the fixed ``min_silence_sec``.
        Lp ONLY moves the floor -- the completion-score tiers are unchanged."""
        c = self._c
        lp = self._pause.floor()  # clamped to [high_confidence_floor, max_silence_sec]
        # ANTI-CHOP: below the learned pause floor, HOLD -- even when the acoustic
        # timer or a (possibly false) lexical "complete" wants to commit. This is
        # the fragmentation fix: a mid-utterance pause under Lp must not commit;
        # the speaker's resume lands as a new partial. Bounded by max_silence_sec
        # (Lp itself is clamped there, so this only defers to the hard backstop).
        if silence_sec < lp and silence_sec < c.max_silence_sec:
            return False
        # SHORTEN: a clearly-complete turn commits as soon as trailing silence has
        # reached Lp (not the fixed floor) -- the latency win, self-calibrated.
        if completion_score >= c.complete_threshold and silence_sec >= lp:
            return True
        # EXTEND (bounded): mid-phrase + the acoustic timer fired + not waited too
        # long -> hold for more speech (unchanged incomplete behaviour).
        if (
            acoustic_endpoint
            and completion_score <= c.incomplete_threshold
            and silence_sec < c.max_silence_sec
        ):
            return False
        # Otherwise the acoustic decision stands (the safe default / hard backstop).
        return acoustic_endpoint
