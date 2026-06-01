from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

Embedding = Sequence[float]
EmbedFn = Callable[[Sequence[float], int], Optional[Embedding]]


def cosine_similarity(a: Embedding, b: Embedding) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def rms(samples: Sequence[float]) -> float:
    """Root-mean-square level of a mono float block (0.0 for empty)."""
    n = len(samples)
    if n == 0:
        return 0.0
    return math.sqrt(sum(float(x) * float(x) for x in samples) / n)


def loudness_admits(
    speech_level: float, ambient_level: float, *, margin_db: float
) -> bool:
    """Near-field 'is this the user' check by LOUDNESS (a secondary signal to the
    voice-identity gate). The user is CLOSE to the mic -> loud; a TV / another
    person across the room sits near the ambient noise floor. Admit when the
    speech sits at least ``margin_db`` dB above the running ambient floor.

    ``margin_db <= 0`` DISABLES it (returns True -> the loudness signal abstains,
    leaving the decision to identity). No ambient floor yet (``ambient_level <=
    0``) also abstains (True) so a cold start never wrongly rejects."""
    if margin_db <= 0.0 or ambient_level <= 0.0:
        return True
    if speech_level <= 0.0:
        return False
    return 20.0 * math.log10(speech_level / ambient_level) >= margin_db


def passes_output_margin(
    speech_level: float, playback_level: float, *, margin_db: float
) -> bool:
    """Conservative self-interruption guard for the *unenrolled* gate.

    Without speaker-ID or AEC, the assistant's own TTS bleeding into the mic
    looks like the user talking. When playback is active we therefore require
    the detected speech to sit ``margin_db`` dB *above* the current playback
    level before we treat it as a genuine barge-in -- residual echo is at or
    below the playback level, a real user speaking over the assistant is
    louder. ``playback_level <= 0`` means nothing is playing, so there is no
    self-interruption risk and we fail open (return True).
    """
    if playback_level <= 0.0:
        return True
    if speech_level <= 0.0:
        return False
    # Compare in dB: speech must exceed playback by at least margin_db.
    ratio_db = 20.0 * math.log10(speech_level / playback_level)
    return ratio_db >= margin_db


class SpeakerGate:
    """Decides whether detected speech is the enrolled user (=> real barge-in).

    Without echo cancellation, the assistant's own TTS leaking into the mic can
    look like the user talking and cause false self-interruption. This gate
    compares a speaker embedding of the detected speech against the enrolled
    user's voice; only a match above ``threshold`` counts as barge-in.

    The embedding function is injectable so the *decision logic* can be tested
    without any model. :func:`sherpa_speaker_gate` builds one backed by
    sherpa-onnx speaker embeddings for production.
    """

    def __init__(self, *, threshold: float = 0.5, embed_fn: Optional[EmbedFn] = None):
        self.threshold = threshold
        self._embed_fn = embed_fn
        self._enrolled: Optional[list[float]] = None

    def set_embed_fn(self, embed_fn: EmbedFn) -> None:
        self._embed_fn = embed_fn

    @property
    def is_enrolled(self) -> bool:
        return self._enrolled is not None

    def enroll_embedding(self, embedding: Embedding) -> None:
        self._enrolled = list(embedding)

    def enroll(self, samples: Sequence[float], sample_rate: int) -> bool:
        embedding = self._embed(samples, sample_rate)
        if embedding is not None:
            self._enrolled = list(embedding)
        return self.is_enrolled

    def accept(
        self,
        samples: Sequence[float],
        sample_rate: int,
        *,
        playback_level: float = 0.0,
        output_margin_db: float = 0.0,
    ) -> bool:
        """Return True if this audio should be treated as the user barging in.

        When *enrolled*, only the user's own voice (cosine >= ``threshold``)
        counts; an unusable embedding fails open.

        When *unenrolled* (no speaker-ID / no enrollment), we used to blindly
        fail open, which lets the assistant's own TTS echo self-interrupt
        (realtime-concurrency-5). With no AEC the conservative fallback instead
        gates on output activity: if ``playback_level`` is provided and the
        assistant is currently outputting audio, the detected speech must sit
        ``output_margin_db`` dB above that playback level to count. A genuine
        user talking over the assistant clears the margin; residual TTS echo
        does not. With nothing playing (or no margin configured) we still fail
        open so a real interrupt is never lost."""
        if not self.is_enrolled:
            if output_margin_db <= 0.0:
                return True  # no conservative guard requested -> legacy fail-open
            return passes_output_margin(
                rms(samples), playback_level, margin_db=output_margin_db
            )
        embedding = self._embed(samples, sample_rate)
        if embedding is None:
            return True
        return cosine_similarity(embedding, self._enrolled) >= self.threshold

    def similarity(self, samples: Sequence[float], sample_rate: int) -> float:
        if not self.is_enrolled:
            return 0.0
        embedding = self._embed(samples, sample_rate)
        if embedding is None:
            return 0.0
        return cosine_similarity(embedding, self._enrolled)

    def embed(self, samples: Sequence[float], sample_rate: int) -> Optional[Embedding]:
        """Public embedding accessor used by the enrollment flow (core.enroll).

        Returns the raw speaker embedding for ``samples`` (or ``None`` if the
        model couldn't produce one), without touching the enrolled reference --
        enrollment needs the per-recording vectors to average them itself."""
        return self._embed(samples, sample_rate)

    def _embed(self, samples: Sequence[float], sample_rate: int) -> Optional[Embedding]:
        if self._embed_fn is None:
            raise RuntimeError("SpeakerGate has no embed_fn configured")
        return self._embed_fn(samples, sample_rate)


def sherpa_speaker_gate(
    model_path: str, *, threshold: float = 0.5, num_threads: int = 1, provider: str = "cpu"
) -> SpeakerGate:
    """Build a :class:`SpeakerGate` backed by a sherpa-onnx speaker-embedding
    model (e.g. a 3D-Speaker / WeSpeaker ONNX export). Imported lazily."""

    extractor_holder: dict[str, object] = {}

    def embed_fn(samples: Sequence[float], sample_rate: int):
        import numpy as np
        import sherpa_onnx

        extractor = extractor_holder.get("extractor")
        if extractor is None:
            config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=model_path, num_threads=num_threads, provider=provider
            )
            extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
            extractor_holder["extractor"] = extractor

        stream = extractor.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=np.asarray(samples, dtype="float32"))
        stream.input_finished()
        if not extractor.is_ready(stream):
            return None
        return list(extractor.compute(stream))

    return SpeakerGate(threshold=threshold, embed_fn=embed_fn)
