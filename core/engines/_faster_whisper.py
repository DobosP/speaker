"""Local-only Faster-Whisper decoding for complete, endpointed utterances.

This adapter is the shared local boundary for offline evaluation and the
explicitly configured final verifier: callers provide one complete mono PCM
utterance, and the recognizer returns text plus aggregate confidence
diagnostics. Transcript text is excluded from the result's representation.
"""
from __future__ import annotations

import math
import operator
import threading
from dataclasses import dataclass, field
from os import PathLike, fspath
from pathlib import Path
from typing import Any, Callable

from core.audio_frontend import AudioResampler


_MODEL_SAMPLE_RATE = 16_000
_ModelFactory = Callable[..., Any]
_ResamplerFactory = Callable[..., Any]


@dataclass(frozen=True)
class FasterWhisperResult:
    """One transcription and repr-safe, uncalibrated model diagnostics.

    ``confidence`` is a bounded heuristic composed from log probability and
    no-speech evidence.  It is not a calibrated probability and must not be
    used directly for runtime selection, authorization, or promotion.
    """

    text: str = field(repr=False)
    confidence: float
    avg_logprob: float | None
    no_speech_probability: float | None
    compression_ratio: float | None
    segment_count: int
    language: str
    language_probability: float | None
    audio_seconds: float


class FasterWhisperEndpointRecognizer:
    """Decode complete mono utterances with a cached CUDA Faster-Whisper model.

    ``model_path`` must already be a local directory.  Model import and
    construction are deferred until the first non-empty transcription, and the
    loader is always told to operate on local files only.
    """

    def __init__(
        self,
        model_path: str | PathLike[str],
        *,
        model_factory: _ModelFactory | None = None,
        resampler_factory: _ResamplerFactory | None = None,
    ) -> None:
        self._model_path = _local_model_directory(model_path)
        self._model_factory = model_factory
        self._resampler_factory = resampler_factory or AudioResampler
        self._model: Any | None = None
        self._load_lock = threading.Lock()
        self._decode_lock = threading.Lock()

    def warm(self) -> None:
        """Load the local model and its CUDA runtime without decoding audio."""
        self._get_model()

    def transcribe(self, pcm: Any, sample_rate: int) -> FasterWhisperResult:
        """Transcribe one complete float32 mono PCM utterance.

        Other floating-point one-dimensional inputs are safely converted to
        float32.  Integer, non-finite, or multi-channel input is rejected
        rather than silently changing the audio domain or amplitude scale.
        """
        rate = _positive_integer_sample_rate(sample_rate)
        samples = _mono_float32(pcm)
        audio_seconds = samples.size / float(rate)
        if samples.size == 0:
            return _empty_result(audio_seconds)

        model_samples = self._resample(samples, rate)
        if model_samples.size == 0:
            return _empty_result(audio_seconds)

        with self._decode_lock:
            segments, info = self._get_model().transcribe(
                model_samples,
                task="transcribe",
                language="en",
                beam_size=5,
                patience=1.0,
                temperature=0.0,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                vad_filter=False,
                vad_parameters=None,
                condition_on_previous_text=False,
                initial_prompt=None,
                prefix=None,
                hotwords=None,
                suppress_blank=True,
                suppress_tokens=[-1],
                without_timestamps=True,
                word_timestamps=False,
            )
            # Faster-Whisper performs decoding while its iterable is consumed.
            # Materialize it while the model's decode lock is still held.
            decoded_segments = tuple(segments)

        return _result_from_segments(decoded_segments, info, audio_seconds)

    def _resample(self, samples: Any, sample_rate: int) -> Any:
        if sample_rate == _MODEL_SAMPLE_RATE:
            return samples
        resampler = self._resampler_factory(
            sample_rate,
            _MODEL_SAMPLE_RATE,
            quality="VHQ",
        )
        return _mono_float32(resampler.process(samples, last=True))

    def _get_model(self) -> Any:
        model = self._model
        if model is not None:
            return model
        with self._load_lock:
            model = self._model
            if model is None:
                factory = self._model_factory
                if factory is None:
                    # Keep this optional GPU dependency out of module import and
                    # empty-input paths.
                    from core.engines._cuda_wheels import (
                        preload_cuda_wheel_libraries,
                    )

                    # CTranslate2 discovers cuBLAS/cuDNN dynamically.  Load the
                    # pinned wheel-owned CUDA runtime before Faster-Whisper can
                    # import CTranslate2; normal live startup therefore needs no
                    # LD_LIBRARY_PATH mutation or system CUDA toolkit.
                    preload_cuda_wheel_libraries()
                    from faster_whisper import WhisperModel

                    factory = WhisperModel
                model = factory(
                    str(self._model_path),
                    device="cuda",
                    device_index=0,
                    compute_type="float16",
                    num_workers=1,
                    local_files_only=True,
                )
                if model is None:
                    raise RuntimeError("Faster-Whisper model construction failed")
                self._model = model
        return model


def _local_model_directory(model_path: str | PathLike[str]) -> Path:
    if isinstance(model_path, (str, PathLike)):
        try:
            raw_path = fspath(model_path)
        except TypeError:
            raw_path = ""
    else:
        raw_path = ""
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("model_path must name an existing local directory")
    try:
        path = Path(raw_path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        raise ValueError("model_path must name an existing local directory") from None
    if not path.is_dir():
        raise ValueError("model_path must name an existing local directory")
    return path


def _positive_integer_sample_rate(sample_rate: int) -> int:
    if isinstance(sample_rate, bool):
        raise ValueError("sample_rate must be a positive integer")
    try:
        value = operator.index(sample_rate)
    except TypeError:
        raise ValueError("sample_rate must be a positive integer") from None
    if value <= 0:
        raise ValueError("sample_rate must be a positive integer")
    return value


def _mono_float32(pcm: Any) -> Any:
    import numpy as np

    try:
        source = np.asarray(pcm)
    except Exception:
        raise ValueError("PCM must be finite one-dimensional real audio") from None
    if source.ndim != 1 or not np.issubdtype(source.dtype, np.floating):
        raise ValueError("PCM must be finite one-dimensional real audio")
    try:
        samples = np.array(source, dtype=np.float32, copy=True, order="C")
    except (TypeError, ValueError, OverflowError):
        raise ValueError("PCM must be finite one-dimensional real audio") from None
    if not np.isfinite(samples).all():
        raise ValueError("PCM must be finite one-dimensional real audio")
    return samples


def _empty_result(audio_seconds: float) -> FasterWhisperResult:
    return FasterWhisperResult(
        text="",
        confidence=0.0,
        avg_logprob=None,
        no_speech_probability=None,
        compression_ratio=None,
        segment_count=0,
        language="en",
        language_probability=None,
        audio_seconds=audio_seconds,
    )


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if math.isfinite(numeric) else None


def _segment_weight(segment: Any) -> float:
    start = _finite_float(getattr(segment, "start", None))
    end = _finite_float(getattr(segment, "end", None))
    if start is not None and end is not None and end > start:
        return end - start
    return 1.0


def _weighted_average(values: list[tuple[float, float]]) -> float | None:
    if not values:
        return None
    total_weight = sum(weight for _, weight in values)
    if total_weight <= 0.0 or not math.isfinite(total_weight):
        return None
    return sum(value * weight for value, weight in values) / total_weight


def _result_from_segments(
    segments: tuple[Any, ...],
    info: Any,
    audio_seconds: float,
) -> FasterWhisperResult:
    text_parts: list[str] = []
    logprobs: list[tuple[float, float]] = []
    no_speech_probabilities: list[tuple[float, float]] = []
    compression_ratios: list[tuple[float, float]] = []

    for segment in segments:
        text = getattr(segment, "text", None)
        if not isinstance(text, str):
            raise RuntimeError("Faster-Whisper returned an invalid segment")
        text_parts.append(text)
        weight = _segment_weight(segment)

        avg_logprob = _finite_float(getattr(segment, "avg_logprob", None))
        if avg_logprob is not None:
            logprobs.append((avg_logprob, weight))

        no_speech = _finite_float(getattr(segment, "no_speech_prob", None))
        if no_speech is not None:
            no_speech_probabilities.append((min(1.0, max(0.0, no_speech)), weight))

        compression_ratio = _finite_float(
            getattr(segment, "compression_ratio", None)
        )
        if compression_ratio is not None:
            compression_ratios.append((max(0.0, compression_ratio), weight))

    text = "".join(text_parts).strip()
    avg_logprob = _weighted_average(logprobs)
    no_speech_probability = _weighted_average(no_speech_probabilities)
    compression_ratio = _weighted_average(compression_ratios)
    confidence = 0.0
    if text and avg_logprob is not None:
        speech_factor = (
            1.0
            if no_speech_probability is None
            else 1.0 - no_speech_probability
        )
        confidence = min(1.0, max(0.0, math.exp(min(0.0, avg_logprob)) * speech_factor))

    language_probability = _finite_float(
        getattr(info, "language_probability", None)
    )
    if language_probability is not None:
        language_probability = min(1.0, max(0.0, language_probability))

    return FasterWhisperResult(
        text=text,
        confidence=confidence,
        avg_logprob=avg_logprob,
        no_speech_probability=no_speech_probability,
        compression_ratio=compression_ratio,
        segment_count=len(segments),
        language="en",
        language_probability=language_probability,
        audio_seconds=audio_seconds,
    )
