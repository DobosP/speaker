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

    def accept(self, samples: Sequence[float], sample_rate: int) -> bool:
        """Return True if this audio should be treated as the user barging in.

        Fail-open: if there's no enrollment or no usable embedding, don't block
        barge-in (better to over-interrupt than to ignore the user)."""
        if not self.is_enrolled:
            return True
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

    def _embed(self, samples: Sequence[float], sample_rate: int) -> Optional[Embedding]:
        if self._embed_fn is None:
            raise RuntimeError("SpeakerGate has no embed_fn configured")
        return self._embed_fn(samples, sample_rate)


def sherpa_speaker_gate(model_path: str, *, threshold: float = 0.5, num_threads: int = 1) -> SpeakerGate:
    """Build a :class:`SpeakerGate` backed by a sherpa-onnx speaker-embedding
    model (e.g. a 3D-Speaker / WeSpeaker ONNX export). Imported lazily."""

    extractor_holder: dict[str, object] = {}

    def embed_fn(samples: Sequence[float], sample_rate: int):
        import numpy as np
        import sherpa_onnx

        extractor = extractor_holder.get("extractor")
        if extractor is None:
            config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=model_path, num_threads=num_threads
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
