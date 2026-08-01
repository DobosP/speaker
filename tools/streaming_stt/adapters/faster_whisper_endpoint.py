"""Final-only Faster-Whisper adapter for the isolated benchmark worker.

This is deliberately not a streaming recognizer. The adapter validates and
buffers one complete bounded PCM case, honors the requested replay pace, then
performs exactly one local CTranslate2 decode and emits no partial hypotheses.
Candidate packages are imported only after the worker verifies both the
runtime-tree and model-tree receipts.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib
from importlib import metadata
from importlib.util import find_spec
import math
import os
from pathlib import Path
import platform
import sys
import threading
import time
from dataclasses import dataclass, field
from collections.abc import Callable, Mapping
from typing import TypeVar

from ..bounded_io import BoundedReadError, read_regular_bounded
from ..manifest import FasterWhisperEndpointConfig
from ..protocol import (
    MAX_HYPOTHESIS_CHARS,
    MAX_PCM_BYTES,
    MAX_RESOURCE_MB,
    MAX_STREAM_SAMPLES,
    ResourceUsage,
    TranscribeRequest,
)


_MAX_SEGMENTS = 4096
_WHEEL_LIBRARIES = (
    ("nvidia.cublas", "libcublasLt.so.12"),
    ("nvidia.cublas", "libcublas.so.12"),
    ("nvidia.cuda_nvrtc", "libnvrtc.so.12"),
    ("nvidia.cudnn", "libcudnn.so.9"),
)
_LOAD_MODE = getattr(os, "RTLD_NOW", 0) | getattr(
    os,
    "RTLD_GLOBAL",
    getattr(ctypes, "RTLD_GLOBAL", 0),
)
_EXPECTED_DISTRIBUTIONS = {
    "faster-whisper": "faster_whisper_version",
    "ctranslate2": "ctranslate2_version",
    "numpy": "numpy_version",
    "nvidia-cublas-cu12": "cublas_version",
    "nvidia-cudnn-cu12": "cudnn_version",
    "nvidia-cuda-nvrtc-cu12": "cuda_nvrtc_version",
}
_T = TypeVar("_T")


class FasterWhisperEndpointAdapterError(RuntimeError):
    """A detail-free candidate load, PCM, or final-decode failure."""


@dataclass(frozen=True)
class FasterWhisperEndpointReady:
    model_load_ms: float
    resources: ResourceUsage


@dataclass(frozen=True)
class FasterWhisperEndpointCase:
    """One endpoint decode with an intentionally empty partial sequence."""

    partials: tuple[object, ...] = field(default=(), repr=False)
    final: str = field(default="", repr=False)
    elapsed_ms: float = 0.0
    finalization_ms: float = 0.0
    compute_ms: float = 0.0
    deadline_misses: int = 0
    max_backlog_ms: float = 0.0
    resources: ResourceUsage = field(
        default_factory=lambda: ResourceUsage(
            rss_mb=0.0,
            threads=1,
            vram_mb=None,
        )
    )


@dataclass(frozen=True)
class _CandidateApi:
    versions: Mapping[str, str]
    numpy: object
    whisper_model: Callable[..., object]
    supported_compute_types: Callable[[str, int], object]
    cuda_handles: tuple[object, ...] = field(repr=False)


def _wheel_lib_directory(package: str) -> Path:
    try:
        spec = find_spec(package)
        if len(sys.path) < 2:
            raise FasterWhisperEndpointAdapterError()
        runtime_root = Path(sys.path[1]).resolve(strict=True)
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError):
        raise FasterWhisperEndpointAdapterError() from None
    if not runtime_root.is_dir():
        raise FasterWhisperEndpointAdapterError()
    locations = None if spec is None else spec.submodule_search_locations
    if not locations:
        raise FasterWhisperEndpointAdapterError()
    candidates: list[Path] = []
    for raw_location in locations:
        try:
            directory = (Path(raw_location) / "lib").resolve(strict=True)
            directory.relative_to(runtime_root)
        except (OSError, RuntimeError, ValueError):
            continue
        if directory.is_dir() and directory not in candidates:
            candidates.append(directory)
    if len(candidates) != 1:
        raise FasterWhisperEndpointAdapterError()
    return candidates[0]


def _preload_cuda_wheel_libraries() -> tuple[object, ...]:
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise FasterWhisperEndpointAdapterError()
    directories: dict[str, Path] = {}
    handles: list[object] = []
    try:
        for package, soname in _WHEEL_LIBRARIES:
            directory = directories.get(package)
            if directory is None:
                directory = _wheel_lib_directory(package)
                directories[package] = directory
            library = (directory / soname).resolve(strict=True)
            if library.parent != directory:
                raise FasterWhisperEndpointAdapterError()
            handles.append(ctypes.CDLL(str(library), mode=_LOAD_MODE))
    except (OSError, TypeError, ValueError, FasterWhisperEndpointAdapterError):
        raise FasterWhisperEndpointAdapterError() from None
    return tuple(handles)


def _candidate_api(config: FasterWhisperEndpointConfig) -> _CandidateApi:
    versions: dict[str, str] = {}
    try:
        for distribution, field_name in _EXPECTED_DISTRIBUTIONS.items():
            actual = metadata.version(distribution)
            expected = getattr(config, field_name)
            if actual != expected:
                raise FasterWhisperEndpointAdapterError()
            versions[distribution] = actual
        # Retain the handles locally until candidate modules have been loaded;
        # their CUDA dependencies then remain process-global for worker life.
        handles = _preload_cuda_wheel_libraries()
        numpy = importlib.import_module("numpy")
        ctranslate2 = importlib.import_module("ctranslate2")
        faster_whisper = importlib.import_module("faster_whisper")
        if (
            not handles
            or getattr(numpy, "__version__", None) != config.numpy_version
            or getattr(ctranslate2, "__version__", None) != config.ctranslate2_version
            or getattr(faster_whisper, "__version__", None)
            != config.faster_whisper_version
        ):
            raise FasterWhisperEndpointAdapterError()
        factory = getattr(faster_whisper, "WhisperModel", None)
        supported = getattr(ctranslate2, "get_supported_compute_types", None)
        if not callable(factory) or not callable(supported):
            raise FasterWhisperEndpointAdapterError()
    except (
        ImportError,
        metadata.PackageNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        FasterWhisperEndpointAdapterError,
    ):
        raise FasterWhisperEndpointAdapterError() from None
    return _CandidateApi(
        versions=versions,
        numpy=numpy,
        whisper_model=factory,
        supported_compute_types=supported,
        cuda_handles=handles,
    )


def _resource_usage() -> ResourceUsage:
    rss_mb: float | None = None
    threads: int | None = None
    try:
        status = Path("/proc/self/status").read_text(
            encoding="utf-8",
            errors="strict",
        )
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                fields = line.split()
                if len(fields) == 3 and fields[2] == "kB":
                    rss_mb = int(fields[1]) / 1024.0
            elif line.startswith("Threads:"):
                fields = line.split()
                if len(fields) == 2:
                    threads = int(fields[1])
    except (OSError, UnicodeError, ValueError):
        pass
    if rss_mb is None:
        try:
            import resource

            maximum_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            rss_mb = maximum_rss / (1024.0 if sys.platform != "darwin" else 1024**2)
        except (ImportError, OSError, ValueError):
            rss_mb = 0.0
    if threads is None:
        threads = max(1, threading.active_count())
    if (
        not math.isfinite(rss_mb)
        or not 0.0 <= rss_mb <= MAX_RESOURCE_MB
        or not 1 <= threads <= 4096
    ):
        raise FasterWhisperEndpointAdapterError()
    # VRAM is intentionally unknown here rather than self-invented. Independent
    # GPU telemetry can be joined to a future private benchmark run.
    return ResourceUsage(rss_mb=rss_mb, threads=threads, vram_mb=None)


def _runtime_versions(api: object) -> Mapping[str, str]:
    value = getattr(api, "versions", None)
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) or not isinstance(version, str)
        for key, version in value.items()
    ):
        raise FasterWhisperEndpointAdapterError()
    return value


class FasterWhisperEndpointAdapter:
    """Buffer complete PCM and make one deterministic local final decode."""

    def __init__(
        self,
        model_root: Path,
        *,
        config: FasterWhisperEndpointConfig,
        api: object | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(config, FasterWhisperEndpointConfig):
            raise FasterWhisperEndpointAdapterError()
        try:
            resolved_model_root = Path(model_root).resolve(strict=True)
        except (OSError, RuntimeError):
            raise FasterWhisperEndpointAdapterError() from None
        if not resolved_model_root.is_dir():
            raise FasterWhisperEndpointAdapterError()
        self.config = config
        self.model_root = resolved_model_root
        self._clock = clock
        self._sleeper = sleeper
        self._closed = False
        self._model: object | None = None
        self._candidate_api: object | None = None

        load_started = self._now()
        candidate = api if api is not None else _candidate_api(config)
        expected_versions = {
            distribution: getattr(config, field_name)
            for distribution, field_name in _EXPECTED_DISTRIBUTIONS.items()
        }
        if dict(_runtime_versions(candidate)) != expected_versions:
            raise FasterWhisperEndpointAdapterError()
        numpy = getattr(candidate, "numpy", None)
        factory = getattr(candidate, "whisper_model", None)
        supported = getattr(candidate, "supported_compute_types", None)
        if numpy is None or not callable(factory) or not callable(supported):
            raise FasterWhisperEndpointAdapterError()
        try:
            compute_types = set(supported(config.device, config.device_index))
            if config.compute_type not in compute_types:
                raise FasterWhisperEndpointAdapterError()
            self._model = factory(
                str(self.model_root),
                device=config.device,
                device_index=config.device_index,
                compute_type=config.compute_type,
                cpu_threads=config.cpu_threads,
                num_workers=config.num_workers,
                local_files_only=True,
            )
            if self._model is None:
                raise FasterWhisperEndpointAdapterError()
            self._candidate_api = candidate
            self._numpy = numpy
            self.ready = FasterWhisperEndpointReady(
                model_load_ms=self._duration_ms(load_started, self._now()),
                resources=_resource_usage(),
            )
        except Exception as error:
            self._model = None
            self._candidate_api = None
            if isinstance(error, FasterWhisperEndpointAdapterError):
                raise
            raise FasterWhisperEndpointAdapterError() from None

    def __enter__(self) -> FasterWhisperEndpointAdapter:
        if self._closed:
            raise FasterWhisperEndpointAdapterError()
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._model = None
        self._candidate_api = None

    def _now(self) -> float:
        try:
            value = float(self._clock())
        except (TypeError, ValueError, OverflowError):
            raise FasterWhisperEndpointAdapterError() from None
        if not math.isfinite(value):
            raise FasterWhisperEndpointAdapterError()
        return value

    @staticmethod
    def _duration_ms(start: float, end: float) -> float:
        duration = end - start
        if not math.isfinite(duration) or duration < 0.0:
            raise FasterWhisperEndpointAdapterError()
        return duration * 1000.0

    def _read_pcm(self, request: TranscribeRequest) -> object:
        try:
            snapshot = read_regular_bounded(
                request.pcm.path,
                maximum_bytes=MAX_PCM_BYTES,
                expected_bytes=request.pcm.size_bytes,
            )
        except BoundedReadError:
            raise FasterWhisperEndpointAdapterError() from None
        raw = snapshot.data
        if (
            len(raw) != request.pcm.size_bytes
            or hashlib.sha256(raw).hexdigest() != request.pcm.sha256
            or request.pcm.sample_rate != self.config.sample_rate
            or request.pcm.channels != 1
            or request.pcm.encoding != "f32le"
        ):
            raise FasterWhisperEndpointAdapterError()
        try:
            samples = self._numpy.frombuffer(raw, dtype="<f4").astype(
                self._numpy.float32,
                copy=True,
            )
            finite = bool(self._numpy.isfinite(samples).all())
        except Exception:
            raise FasterWhisperEndpointAdapterError() from None
        if (
            getattr(samples, "ndim", None) != 1
            or getattr(samples, "size", None) != request.pcm.samples
            or not finite
        ):
            raise FasterWhisperEndpointAdapterError()
        return samples

    def _decode(self, samples: object) -> str:
        model = self._model
        if model is None:
            raise FasterWhisperEndpointAdapterError()
        try:
            segments, _info = model.transcribe(
                samples,
                task=self.config.task,
                language=self.config.language,
                beam_size=self.config.beam_size,
                patience=self.config.patience,
                temperature=self.config.temperature,
                compression_ratio_threshold=(self.config.compression_ratio_threshold),
                log_prob_threshold=self.config.log_prob_threshold,
                no_speech_threshold=self.config.no_speech_threshold,
                vad_filter=self.config.vad_filter,
                vad_parameters=None,
                condition_on_previous_text=self.config.condition_on_previous_text,
                initial_prompt=None,
                prefix=None,
                hotwords=None,
                suppress_blank=True,
                suppress_tokens=[-1],
                without_timestamps=self.config.without_timestamps,
                word_timestamps=self.config.word_timestamps,
            )
            parts: list[str] = []
            characters = 0
            for index, segment in enumerate(segments):
                if index >= _MAX_SEGMENTS:
                    raise FasterWhisperEndpointAdapterError()
                text = getattr(segment, "text", None)
                if not isinstance(text, str):
                    raise FasterWhisperEndpointAdapterError()
                characters += len(text)
                if characters > MAX_HYPOTHESIS_CHARS:
                    raise FasterWhisperEndpointAdapterError()
                parts.append(text)
            final = "".join(parts).strip()
        except FasterWhisperEndpointAdapterError:
            raise
        except Exception:
            raise FasterWhisperEndpointAdapterError() from None
        if len(final) > MAX_HYPOTHESIS_CHARS:
            raise FasterWhisperEndpointAdapterError()
        return final

    def transcribe(
        self,
        request: TranscribeRequest,
        *,
        emit_partial: Callable[[int, object], None],
    ) -> FasterWhisperEndpointCase:
        del emit_partial
        if self._closed or self._model is None:
            raise FasterWhisperEndpointAdapterError()
        samples = self._read_pcm(request)
        total_samples = request.pcm.samples + request.stream.tail_padding_samples
        if total_samples > MAX_STREAM_SAMPLES:
            raise FasterWhisperEndpointAdapterError()

        case_started = self._now()
        stream_started = self._now()
        samples_seen = 0
        for offset in range(0, total_samples, request.stream.chunk_samples):
            count = min(request.stream.chunk_samples, total_samples - offset)
            samples_seen += count
            if request.stream.pace == "realtime":
                target = stream_started + samples_seen / self.config.sample_rate
                before_wait = self._now()
                if before_wait < target:
                    self._sleeper(target - before_wait)
                if self._now() < target:
                    raise FasterWhisperEndpointAdapterError()
        if samples_seen != total_samples:
            raise FasterWhisperEndpointAdapterError()

        finalization_started = self._now()
        final = self._decode(samples)
        finished = self._now()
        finalization_ms = self._duration_ms(finalization_started, finished)
        return FasterWhisperEndpointCase(
            partials=(),
            final=final,
            elapsed_ms=self._duration_ms(case_started, finished),
            finalization_ms=finalization_ms,
            compute_ms=finalization_ms,
            # These numeric fields are mandatory in the worker protocol. They
            # are placeholders, not successful streaming-deadline evidence:
            # this adapter performs no chunk-time decode. The evaluator marks
            # all deadline/backlog aggregates not applicable and emits nulls.
            deadline_misses=0,
            max_backlog_ms=0.0,
            resources=_resource_usage(),
        )


__all__ = [
    "FasterWhisperEndpointAdapter",
    "FasterWhisperEndpointAdapterError",
    "FasterWhisperEndpointCase",
    "FasterWhisperEndpointConfig",
    "FasterWhisperEndpointReady",
]
