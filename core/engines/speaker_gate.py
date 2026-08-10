from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

Embedding = Sequence[float]
EmbedFn = Callable[[Sequence[float], int], Optional[Embedding]]


@dataclass(frozen=True)
class SpeakerIdentityActivation:
    """Why the live runtime must allocate speaker-identity resources."""

    enrollment_reference_available: bool
    word_cut_requires_speaker: bool

    @property
    def active(self) -> bool:
        return self.enrollment_reference_available or self.word_cut_requires_speaker


@dataclass(frozen=True, slots=True)
class SpeakerGateAuthority:
    """Opaque generation receipt for one enrolled speaker policy."""

    gate_token: object
    model_generation: int
    enrollment_generation: int
    policy_generation: int
    enrolled: bool
    threshold: float


@dataclass(frozen=True, slots=True)
class SpeakerSimilarityBatchReceipt:
    """Scores produced under one nonblocking model/enrollment snapshot."""

    authority: SpeakerGateAuthority
    similarities: tuple[float, ...]


def resolve_speaker_identity_activation(
    *,
    speaker_enroll_embedding: str = "",
    speaker_enroll_wav: str = "",
    barge_in_enabled: bool = True,
    barge_word_cut_enabled: bool = False,
    aec_enabled: bool = False,
    barge_word_cut_require_speaker: bool = False,
    exists: Callable[[str], bool] = os.path.exists,
) -> SpeakerIdentityActivation:
    """Resolve whether the live session needs its optional speaker-ID model.

    A configured model is only an available capability.  It becomes active
    when an enrollment reference is actually present, or when the no-AEC
    playback word-cut path explicitly requests the multi-voice identity filter.
    Keeping this pure and injectable gives runtime allocation and readiness one
    exact rule without importing the native Sherpa engine into doctor checks.
    """

    enrollment_paths = (
        str(speaker_enroll_embedding or ""),
        str(speaker_enroll_wav or ""),
    )
    enrollment_reference_available = any(
        path and exists(path) for path in enrollment_paths
    )
    word_cut_requires_speaker = bool(
        barge_in_enabled
        and barge_word_cut_enabled
        and not aec_enabled
        and barge_word_cut_require_speaker
    )
    return SpeakerIdentityActivation(
        enrollment_reference_available=bool(enrollment_reference_available),
        word_cut_requires_speaker=word_cut_requires_speaker,
    )


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


def trim_to_voiced_region(
    samples: Sequence[float],
    sample_rate: int,
    *,
    win: float = 0.02,
    thresh_ratio: float = 0.15,
    pad: float = 0.1,
):
    """Return the energy-voiced region plus a small symmetric pad.

    Speaker embeddings shift when fixed recording silence or endpoint padding
    dominates the clip. Enrollment and live final gating must therefore use the
    same temporal envelope as well as the same capture front end. Pure silence
    and clips too short to classify are returned unchanged (fail open).
    """
    import numpy as np

    a = np.asarray(samples, dtype="float32").reshape(-1)
    w = max(1, int(sample_rate * win))
    if a.size < 2 * w:
        return a
    n = (a.size // w) * w
    e = np.sqrt((a[:n].reshape(-1, w) ** 2).mean(axis=1))
    peak = float(e.max()) if e.size else 0.0
    if peak <= 0.0:
        return a
    voiced = np.where(e >= peak * thresh_ratio)[0]
    if voiced.size == 0:
        return a
    start = max(0, int(voiced[0]) * w - int(pad * sample_rate))
    end = min(a.size, (int(voiced[-1]) + 1) * w + int(pad * sample_rate))
    return a[start:end]


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
        self._authority_token = object()
        self._policy_lock = threading.Lock()
        self._threshold = float(threshold)
        self._policy_generation = 0
        self._embed_fn = embed_fn
        self._model_generation = 0
        self._enrolled: Optional[list[float]] = None
        # Embedding inference may run on the async final worker while capture
        # recovery clears/reloads enrollment. Keep mutations short and versioned:
        # inference stays outside the lock, then a changed version makes that
        # in-flight decision fail open instead of comparing against stale state.
        self._enrollment_lock = threading.Lock()
        self._enrollment_generation = 0
        # The sherpa extractor is shared by playback-time word-cut and the async
        # final worker and is not documented as re-entrant. Serialize inference;
        # the capture path uses try_similarity() and defers instead of blocking.
        self._inference_lock = threading.Lock()

    def set_embed_fn(self, embed_fn: EmbedFn) -> None:
        with self._inference_lock:
            with self._policy_lock:
                self._embed_fn = embed_fn
                self._model_generation += 1

    @property
    def threshold(self) -> float:
        with self._policy_lock:
            return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        threshold = float(value)
        with self._policy_lock:
            self._threshold = threshold
            self._policy_generation += 1

    @property
    def is_enrolled(self) -> bool:
        with self._enrollment_lock:
            return self._enrolled is not None

    def enroll_embedding(self, embedding: Embedding) -> None:
        enrolled = list(embedding)
        with self._enrollment_lock:
            self._enrolled = enrolled
            self._enrollment_generation += 1

    def clear_enrollment(self) -> None:
        """Disable identity rejection until a compatible reference is loaded."""
        with self._enrollment_lock:
            self._enrolled = None
            self._enrollment_generation += 1

    def _enrollment_snapshot(self) -> tuple[Optional[list[float]], int]:
        with self._enrollment_lock:
            return self._enrolled, self._enrollment_generation

    def _enrollment_is_current(
        self, enrolled: list[float], generation: int
    ) -> bool:
        with self._enrollment_lock:
            return (
                self._enrollment_generation == generation
                and self._enrolled is enrolled
            )

    def authority_state(self) -> SpeakerGateAuthority:
        """Return the current gate/model/enrollment/policy generation tuple."""

        with self._enrollment_lock:
            enrolled = self._enrolled
            enrollment_generation = self._enrollment_generation
            with self._policy_lock:
                threshold = self._threshold
                policy_generation = self._policy_generation
                model_generation = self._model_generation
        return SpeakerGateAuthority(
            gate_token=self._authority_token,
            model_generation=model_generation,
            enrollment_generation=enrollment_generation,
            policy_generation=policy_generation,
            enrolled=enrolled is not None,
            threshold=threshold,
        )

    def authority_is_current(self, authority: SpeakerGateAuthority) -> bool:
        """Revalidate an opaque authority receipt without running the model."""

        if type(authority) is not SpeakerGateAuthority:
            return False
        if (
            type(authority.model_generation) is not int
            or type(authority.enrollment_generation) is not int
            or type(authority.policy_generation) is not int
            or type(authority.enrolled) is not bool
            or type(authority.threshold) is not float
            or not math.isfinite(authority.threshold)
            or not 0.0 <= authority.threshold <= 1.0
        ):
            return False
        with self._enrollment_lock:
            enrolled = self._enrolled
            enrollment_generation = self._enrollment_generation
            with self._policy_lock:
                threshold = self._threshold
                policy_generation = self._policy_generation
                model_generation = self._model_generation
        return bool(
            authority.gate_token is self._authority_token
            and authority.model_generation == model_generation
            and authority.enrollment_generation == enrollment_generation
            and authority.policy_generation == policy_generation
            and authority.enrolled is (enrolled is not None)
            and authority.threshold == threshold
        )

    def enroll(self, samples: Sequence[float], sample_rate: int) -> bool:
        embedding = self._embed(samples, sample_rate)
        if embedding is not None:
            self.enroll_embedding(embedding)
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
        enrolled, generation = self._enrollment_snapshot()
        if enrolled is None:
            if output_margin_db <= 0.0:
                return True  # no conservative guard requested -> legacy fail-open
            return passes_output_margin(
                rms(samples), playback_level, margin_db=output_margin_db
            )
        embedding = self._embed(samples, sample_rate)
        if embedding is None:
            return True
        if not self._enrollment_is_current(enrolled, generation):
            return True
        return cosine_similarity(embedding, enrolled) >= self.threshold

    def similarity(self, samples: Sequence[float], sample_rate: int) -> float:
        """Compatibility score; unusable/stale inference collapses to ``0``."""
        similarity = self.verification_similarity(samples, sample_rate)
        return 0.0 if similarity is None else similarity

    def verification_similarity(
        self, samples: Sequence[float], sample_rate: int
    ) -> Optional[float]:
        """Exact identity score, or ``None`` when no verdict was possible.

        Unlike :meth:`accept`, this seam never fail-opens an unusable embedding
        or enrollment-generation race. Security callers may mint VERIFIED only
        from a finite score that clears the enrolled threshold; usability callers
        retain the historical boolean/zero-score APIs.
        """
        enrolled, generation = self._enrollment_snapshot()
        if enrolled is None:
            return None
        embedding = self._embed(samples, sample_rate)
        if embedding is None:
            return None
        if not self._enrollment_is_current(enrolled, generation):
            return None
        return cosine_similarity(embedding, enrolled)

    def try_similarity(
        self, samples: Sequence[float], sample_rate: int
    ) -> Optional[float]:
        """Return similarity without waiting for another model inference.

        ``None`` means the shared extractor is busy. Capture-thread callers retain
        bounded PCM and retry later; a completed unusable embedding returns 0.0.
        """
        enrolled, generation = self._enrollment_snapshot()
        if enrolled is None:
            return 0.0
        if not self._inference_lock.acquire(blocking=False):
            return None
        try:
            embedding = self._embed_unlocked(samples, sample_rate)
        finally:
            self._inference_lock.release()
        if embedding is None:
            return 0.0
        if not self._enrollment_is_current(enrolled, generation):
            return 0.0
        return cosine_similarity(embedding, enrolled)

    def try_similarity_batch(
        self,
        samples: tuple[Sequence[float], ...],
        sample_rate: int,
    ) -> Optional[SpeakerSimilarityBatchReceipt]:
        """Score one or two clips under one nonblocking authority snapshot.

        ``None`` retains :meth:`try_similarity`'s exact busy meaning. An
        unusable embedding produces ``0.0``; the caller must additionally
        revalidate the returned receipt before acting.
        """

        if type(samples) is not tuple or not 0 < len(samples) <= 2:
            raise ValueError("speaker batch must contain one or two clips")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
            raise TypeError("speaker sample rate must be an integer")
        if sample_rate <= 0:
            raise ValueError("speaker sample rate must be positive")
        if not self._inference_lock.acquire(blocking=False):
            return None
        try:
            with self._enrollment_lock:
                enrolled = self._enrolled
                enrollment_generation = self._enrollment_generation
                with self._policy_lock:
                    threshold = self._threshold
                    policy_generation = self._policy_generation
                    model_generation = self._model_generation
            authority = SpeakerGateAuthority(
                gate_token=self._authority_token,
                model_generation=model_generation,
                enrollment_generation=enrollment_generation,
                policy_generation=policy_generation,
                enrolled=enrolled is not None,
                threshold=threshold,
            )
            similarities: list[float] = []
            for clip in samples:
                if enrolled is None:
                    similarities.append(0.0)
                    continue
                embedding = self._embed_unlocked(clip, sample_rate)
                similarities.append(
                    0.0
                    if embedding is None
                    else float(cosine_similarity(embedding, enrolled))
                )
        finally:
            self._inference_lock.release()
        if enrolled is not None and not self._enrollment_is_current(
            enrolled,
            enrollment_generation,
        ):
            similarities = [0.0] * len(samples)
        return SpeakerSimilarityBatchReceipt(
            authority=authority,
            similarities=tuple(similarities),
        )

    def embed(self, samples: Sequence[float], sample_rate: int) -> Optional[Embedding]:
        """Public embedding accessor used by the enrollment flow (core.enroll).

        Returns the raw speaker embedding for ``samples`` (or ``None`` if the
        model couldn't produce one), without touching the enrolled reference --
        enrollment needs the per-recording vectors to average them itself."""
        return self._embed(samples, sample_rate)

    def _embed(self, samples: Sequence[float], sample_rate: int) -> Optional[Embedding]:
        with self._inference_lock:
            return self._embed_unlocked(samples, sample_rate)

    def _embed_unlocked(
        self, samples: Sequence[float], sample_rate: int
    ) -> Optional[Embedding]:
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
