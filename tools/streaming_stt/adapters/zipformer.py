"""Isolated streaming adapter for the selected production GigaSpeech Zipformer.

The native runtime is imported lazily inside the worker. Model acquisition,
fallback filenames, and network-backed model identifiers are deliberately not
supported: the schema-v4 manifest binds the four selected local files first.
The production endpoint-rule values are constructor inputs, but this after-PCM
benchmark finalizes at declared end-of-input and does not score endpointing.
"""

from __future__ import annotations

from array import array
from collections.abc import Callable
from dataclasses import dataclass, field
import hashlib
import importlib
from importlib import metadata
import math
from pathlib import Path
import sys
import threading
import time
from typing import TypeVar

from ..bounded_io import BoundedReadError, read_regular_bounded
from ..manifest import (
    MOBILE_ZIPFORMER_ARTIFACT_SPECS,
    MobileZipformerConfig,
    SHERPA_ZIPFORMER_ARTIFACT_SPECS,
    SherpaZipformerConfig,
)
from ..protocol import (
    MAX_HYPOTHESIS_CHARS,
    MAX_MOBILE_ZIPFORMER_ENDPOINT_OBSERVATIONS,
    MAX_PARTIALS_PER_CASE,
    MAX_PCM_BYTES,
    MAX_RESOURCE_MB,
    MAX_STREAM_SAMPLES,
    MOBILE_ZIPFORMER_ENDPOINT_PROTOCOL_VERSION,
    MobileZipformerEndpointObservation,
    ResourceUsage,
    TranscribeRequest,
)


_SHERPA_DISTRIBUTION = "sherpa-onnx"
_SHERPA_CORE_DISTRIBUTION = "sherpa-onnx-core"
_NUMPY_DISTRIBUTION = "numpy"
_MAX_DECODE_CALLS_PER_DRAIN = 4096
_MODEL_FILENAMES = {
    name.removeprefix("model-"): basename
    for name, basename, _sha256, _size_bytes in SHERPA_ZIPFORMER_ARTIFACT_SPECS
}
_MOBILE_MODEL_FILENAMES = {
    name.removeprefix("model-"): basename
    for name, basename, _precision, _sha256, _size_bytes in (
        MOBILE_ZIPFORMER_ARTIFACT_SPECS
    )
}
_T = TypeVar("_T")


class SherpaZipformerAdapterError(RuntimeError):
    """A detail-free runtime, model, or streaming-contract failure."""


class MobileZipformerAdapterError(RuntimeError):
    """A detail-free mobile-runtime or endpoint-contract failure."""


@dataclass(frozen=True)
class SherpaZipformerReady:
    model_load_ms: float
    resources: ResourceUsage


@dataclass(frozen=True)
class SherpaZipformerPartial:
    after_samples: int
    text: str = field(repr=False)
    elapsed_ms: float
    decode_ms: float


@dataclass(frozen=True)
class SherpaZipformerCase:
    partials: tuple[SherpaZipformerPartial, ...] = field(repr=False)
    final: str = field(repr=False)
    elapsed_ms: float
    finalization_ms: float
    compute_ms: float
    deadline_misses: int
    max_backlog_ms: float
    resources: ResourceUsage


@dataclass(frozen=True)
class MobileZipformerReady:
    model_load_ms: float
    resources: ResourceUsage


@dataclass(frozen=True)
class MobileZipformerPartial:
    after_samples: int
    text: str = field(repr=False)
    elapsed_ms: float
    decode_ms: float


@dataclass(frozen=True)
class MobileZipformerCase:
    partials: tuple[MobileZipformerPartial, ...] = field(repr=False)
    final: str = field(repr=False)
    elapsed_ms: float
    finalization_ms: float
    compute_ms: float
    deadline_misses: int
    max_backlog_ms: float
    resources: ResourceUsage
    source_samples: int
    source_samples_consumed: int
    declared_tail_samples: int
    tail_samples_consumed: int
    chunks_yielded: int
    model_padding_samples: int
    endpoint_observations: tuple[MobileZipformerEndpointObservation, ...] = field(
        repr=False
    )
    endpoint_reason: str
    native_endpoint: bool
    endpoint_sample: int | None
    endpoint_latency_ms: float | None
    authoritative: bool


def _candidate_apis(config: SherpaZipformerConfig) -> tuple[object, object]:
    """Import only exact installed production packages; never resolve remotely."""

    try:
        if metadata.version(_SHERPA_DISTRIBUTION) != config.package_version:
            raise SherpaZipformerAdapterError()
        if metadata.version(_NUMPY_DISTRIBUTION) != config.numpy_version:
            raise SherpaZipformerAdapterError()
        sherpa = importlib.import_module("sherpa_onnx")
        numpy = importlib.import_module("numpy")
    except (
        ImportError,
        metadata.PackageNotFoundError,
        SherpaZipformerAdapterError,
    ):
        raise SherpaZipformerAdapterError() from None
    if (
        getattr(sherpa, "__version__", None) != config.package_version
        or getattr(numpy, "__version__", None) != config.numpy_version
    ):
        raise SherpaZipformerAdapterError()
    return sherpa, numpy


def _mobile_candidate_apis(
    config: MobileZipformerConfig,
) -> tuple[object, object]:
    """Import the exact local mobile-compatible runtime and no model resolver."""

    try:
        if (
            metadata.version(_SHERPA_DISTRIBUTION) != config.package_version
            or metadata.version(_SHERPA_CORE_DISTRIBUTION)
            != config.core_package_version
            or metadata.version(_NUMPY_DISTRIBUTION) != config.numpy_version
        ):
            raise MobileZipformerAdapterError()
        sherpa = importlib.import_module("sherpa_onnx")
        numpy = importlib.import_module("numpy")
    except (
        ImportError,
        metadata.PackageNotFoundError,
        MobileZipformerAdapterError,
    ):
        raise MobileZipformerAdapterError() from None
    if (
        getattr(sherpa, "__version__", None) != config.package_version
        or getattr(numpy, "__version__", None) != config.numpy_version
    ):
        raise MobileZipformerAdapterError()
    return sherpa, numpy


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
        raise SherpaZipformerAdapterError()
    return ResourceUsage(rss_mb=rss_mb, threads=threads, vram_mb=None)


def _read_pcm(request: TranscribeRequest) -> array[float]:
    try:
        snapshot = read_regular_bounded(
            request.pcm.path,
            maximum_bytes=MAX_PCM_BYTES,
            expected_bytes=request.pcm.size_bytes,
        )
    except BoundedReadError:
        raise SherpaZipformerAdapterError() from None
    raw = snapshot.data
    if (
        len(raw) != request.pcm.size_bytes
        or hashlib.sha256(raw).hexdigest() != request.pcm.sha256
    ):
        raise SherpaZipformerAdapterError()
    samples = array("f")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    if len(samples) != request.pcm.samples or any(
        not math.isfinite(sample) for sample in samples
    ):
        raise SherpaZipformerAdapterError()
    return samples


def _audio_chunk(
    source: array[float],
    *,
    offset: int,
    count: int,
) -> list[float]:
    source_start = min(offset, len(source))
    source_end = min(offset + count, len(source))
    chunk = list(source[source_start:source_end])
    chunk.extend([0.0] * (count - len(chunk)))
    if len(chunk) != count:
        raise SherpaZipformerAdapterError()
    return chunk


class SherpaZipformerAdapter:
    """One exact recognizer with a fresh, explicitly reset stream per case."""

    def __init__(
        self,
        model_root: Path,
        *,
        config: SherpaZipformerConfig,
        api: object | None = None,
        numpy_api: object | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(config, SherpaZipformerConfig):
            raise SherpaZipformerAdapterError()
        try:
            resolved_model_root = Path(model_root).resolve(strict=True)
        except (OSError, RuntimeError):
            raise SherpaZipformerAdapterError() from None
        if not resolved_model_root.is_dir():
            raise SherpaZipformerAdapterError()
        model_paths = {
            name: resolved_model_root / filename
            for name, filename in _MODEL_FILENAMES.items()
        }
        if any(not path.is_file() for path in model_paths.values()):
            raise SherpaZipformerAdapterError()

        self.config = config
        self.model_root = resolved_model_root
        self._clock = clock
        self._sleeper = sleeper
        self._closed = False
        self._recognizer: object | None = None
        load_started = self._now()
        if (api is None) != (numpy_api is None):
            raise SherpaZipformerAdapterError()
        candidate, numpy = (
            _candidate_apis(config)
            if api is None
            else (api, numpy_api)
        )
        if (
            getattr(candidate, "__version__", None) != config.package_version
            or getattr(numpy, "__version__", None) != config.numpy_version
        ):
            raise SherpaZipformerAdapterError()
        recognizer_type = getattr(candidate, "OnlineRecognizer", None)
        factory = getattr(recognizer_type, "from_transducer", None)
        asarray = getattr(numpy, "asarray", None)
        if not callable(factory) or not callable(asarray):
            raise SherpaZipformerAdapterError()
        self._numpy = numpy
        try:
            self._recognizer = factory(
                tokens=str(model_paths["tokens"]),
                encoder=str(model_paths["encoder"]),
                decoder=str(model_paths["decoder"]),
                joiner=str(model_paths["joiner"]),
                num_threads=config.num_threads,
                provider=config.provider,
                sample_rate=config.sample_rate,
                feature_dim=config.feature_dim,
                enable_endpoint_detection=config.enable_endpoint_detection,
                decoding_method=config.decoding_method,
                max_active_paths=config.max_active_paths,
                rule1_min_trailing_silence=config.rule1_min_trailing_silence,
                rule2_min_trailing_silence=config.rule2_min_trailing_silence,
                rule3_min_utterance_length=config.rule3_min_utterance_length,
            )
            if self._recognizer is None:
                raise SherpaZipformerAdapterError()
            load_finished = self._now()
            self.ready = SherpaZipformerReady(
                model_load_ms=self._duration_ms(load_started, load_finished),
                resources=_resource_usage(),
            )
        except Exception as error:
            self._recognizer = None
            if isinstance(error, SherpaZipformerAdapterError):
                raise
            raise SherpaZipformerAdapterError() from None

    def __enter__(self) -> SherpaZipformerAdapter:
        if self._closed:
            raise SherpaZipformerAdapterError()
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

    def close(self) -> None:
        self._closed = True
        self._recognizer = None

    def _now(self) -> float:
        try:
            value = float(self._clock())
        except (TypeError, ValueError, OverflowError):
            raise SherpaZipformerAdapterError() from None
        if not math.isfinite(value):
            raise SherpaZipformerAdapterError()
        return value

    @staticmethod
    def _duration_ms(start: float, end: float) -> float:
        duration = end - start
        if not math.isfinite(duration) or duration < 0.0:
            raise SherpaZipformerAdapterError()
        return duration * 1000.0

    def _timed(self, function: Callable[[], _T]) -> tuple[_T, float]:
        started = self._now()
        result = function()
        finished = self._now()
        return result, self._duration_ms(started, finished)

    def _drain(self, recognizer: object, stream: object) -> float:
        started = self._now()
        for _ in range(_MAX_DECODE_CALLS_PER_DRAIN):
            if not recognizer.is_ready(stream):
                return self._duration_ms(started, self._now())
            recognizer.decode_stream(stream)
        raise SherpaZipformerAdapterError()

    def _result(self, recognizer: object, stream: object) -> tuple[str, float]:
        value, elapsed_ms = self._timed(lambda: recognizer.get_result(stream))
        if not isinstance(value, str):
            raise SherpaZipformerAdapterError()
        text = value.strip()
        if len(text) > MAX_HYPOTHESIS_CHARS:
            raise SherpaZipformerAdapterError()
        return text, elapsed_ms

    def transcribe(
        self,
        request: TranscribeRequest,
        *,
        emit_partial: Callable[[int, SherpaZipformerPartial], None],
    ) -> SherpaZipformerCase:
        recognizer = self._recognizer
        if self._closed or recognizer is None:
            raise SherpaZipformerAdapterError()
        source = _read_pcm(request)
        total_samples = request.pcm.samples + request.stream.tail_padding_samples
        if total_samples > MAX_STREAM_SAMPLES:
            raise SherpaZipformerAdapterError()

        case_started = self._now()
        stream: object | None = None
        compute_ms = 0.0
        finalization_started = case_started
        partials: list[SherpaZipformerPartial] = []
        final_text = ""
        deadline_misses = 0
        max_backlog_ms = 0.0
        try:
            stream, create_ms = self._timed(recognizer.create_stream)
            compute_ms += create_ms
            if stream is None:
                raise SherpaZipformerAdapterError()
            stream_started = self._now()
            partial_interval_samples = (
                request.stream.partial_interval_ms * self.config.sample_rate + 999
            ) // 1000
            next_partial_samples = partial_interval_samples
            samples_seen = 0
            last_hypothesis = ""

            for offset in range(0, total_samples, request.stream.chunk_samples):
                count = min(
                    request.stream.chunk_samples,
                    total_samples - offset,
                )
                if request.stream.pace == "realtime":
                    target = stream_started + offset / self.config.sample_rate
                    before_wait = self._now()
                    if before_wait < target:
                        self._sleeper(target - before_wait)
                    if self._now() < target:
                        raise SherpaZipformerAdapterError()
                chunk = _audio_chunk(source, offset=offset, count=count)
                native = self._numpy.asarray(chunk, dtype="float32")
                _, accept_ms = self._timed(
                    lambda native=native: stream.accept_waveform(
                        self.config.sample_rate,
                        native,
                    )
                )
                decode_ms = self._drain(recognizer, stream)
                compute_ms += accept_ms + decode_ms
                samples_seen += count

                if samples_seen >= next_partial_samples:
                    text, result_ms = self._result(recognizer, stream)
                    compute_ms += result_ms
                    if text and text != last_hypothesis:
                        if len(partials) >= MAX_PARTIALS_PER_CASE:
                            raise SherpaZipformerAdapterError()
                        partial = SherpaZipformerPartial(
                            after_samples=samples_seen,
                            text=text,
                            elapsed_ms=self._duration_ms(case_started, self._now()),
                            decode_ms=decode_ms + result_ms,
                        )
                        emit_partial(len(partials), partial)
                        partials.append(partial)
                    last_hypothesis = text
                    while next_partial_samples <= samples_seen:
                        next_partial_samples += partial_interval_samples

                if request.stream.pace == "realtime":
                    deadline = stream_started + samples_seen / self.config.sample_rate
                    backlog_ms = max(
                        0.0,
                        self._duration_ms(deadline, max(deadline, self._now())),
                    )
                    max_backlog_ms = max(max_backlog_ms, backlog_ms)
                    if backlog_ms > 0.0:
                        deadline_misses += 1

            if samples_seen != total_samples:
                raise SherpaZipformerAdapterError()
            finalization_started = self._now()
            input_finished = getattr(stream, "input_finished", None)
            if not callable(input_finished):
                raise SherpaZipformerAdapterError()
            _, input_finished_ms = self._timed(input_finished)
            final_decode_ms = self._drain(recognizer, stream)
            final_text, final_result_ms = self._result(recognizer, stream)
            compute_ms += input_finished_ms + final_decode_ms + final_result_ms
        except SherpaZipformerAdapterError:
            raise
        except Exception:
            raise SherpaZipformerAdapterError() from None
        finally:
            if stream is not None:
                active_exception = sys.exc_info()[0] is not None
                try:
                    _, reset_ms = self._timed(lambda: recognizer.reset(stream))
                    compute_ms += reset_ms
                except Exception:
                    if not active_exception:
                        raise SherpaZipformerAdapterError() from None

        finished = self._now()
        return SherpaZipformerCase(
            partials=tuple(partials),
            final=final_text,
            elapsed_ms=self._duration_ms(case_started, finished),
            finalization_ms=self._duration_ms(finalization_started, finished),
            compute_ms=compute_ms,
            deadline_misses=deadline_misses,
            max_backlog_ms=max_backlog_ms,
            resources=_resource_usage(),
        )


class MobileZipformerAdapter:
    """Endpoint-faithful mobile Zipformer decode with reset-committed epochs."""

    def __init__(
        self,
        model_root: Path,
        *,
        config: MobileZipformerConfig,
        api: object | None = None,
        numpy_api: object | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if type(config) is not MobileZipformerConfig:
            raise MobileZipformerAdapterError()
        try:
            resolved_model_root = Path(model_root).resolve(strict=True)
        except (OSError, RuntimeError):
            raise MobileZipformerAdapterError() from None
        if not resolved_model_root.is_dir():
            raise MobileZipformerAdapterError()
        model_paths = {
            name: resolved_model_root / filename
            for name, filename in _MOBILE_MODEL_FILENAMES.items()
        }
        if any(not path.is_file() for path in model_paths.values()):
            raise MobileZipformerAdapterError()

        self.config = config
        self.model_root = resolved_model_root
        self._clock = clock
        self._sleeper = sleeper
        self._closed = False
        self._recognizer: object | None = None
        load_started = self._now()
        if (api is None) != (numpy_api is None):
            raise MobileZipformerAdapterError()
        candidate, numpy = (
            _mobile_candidate_apis(config) if api is None else (api, numpy_api)
        )
        if (
            getattr(candidate, "__version__", None) != config.package_version
            or getattr(numpy, "__version__", None) != config.numpy_version
        ):
            raise MobileZipformerAdapterError()
        recognizer_type = getattr(candidate, "OnlineRecognizer", None)
        factory = getattr(recognizer_type, "from_transducer", None)
        asarray = getattr(numpy, "asarray", None)
        if not callable(factory) or not callable(asarray):
            raise MobileZipformerAdapterError()
        self._numpy = numpy
        try:
            self._recognizer = factory(
                tokens=str(model_paths["tokens"]),
                encoder=str(model_paths["encoder"]),
                decoder=str(model_paths["decoder"]),
                joiner=str(model_paths["joiner"]),
                num_threads=config.num_threads,
                provider=config.provider,
                sample_rate=config.sample_rate,
                feature_dim=config.feature_dim,
                enable_endpoint_detection=config.enable_endpoint_detection,
                decoding_method=config.decoding_method,
                max_active_paths=config.max_active_paths,
                rule1_min_trailing_silence=config.rule1_min_trailing_silence,
                rule2_min_trailing_silence=config.rule2_min_trailing_silence,
                rule3_min_utterance_length=config.rule3_min_utterance_length,
                debug=config.debug,
                model_type=config.model_type,
            )
            if self._recognizer is None:
                raise MobileZipformerAdapterError()
            self.ready = MobileZipformerReady(
                model_load_ms=self._duration_ms(load_started, self._now()),
                resources=_resource_usage(),
            )
        except Exception as error:
            self._recognizer = None
            if isinstance(error, MobileZipformerAdapterError):
                raise
            raise MobileZipformerAdapterError() from None

    def __enter__(self) -> MobileZipformerAdapter:
        if self._closed:
            raise MobileZipformerAdapterError()
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

    def close(self) -> None:
        self._closed = True
        self._recognizer = None

    def _now(self) -> float:
        try:
            value = float(self._clock())
        except (TypeError, ValueError, OverflowError):
            raise MobileZipformerAdapterError() from None
        if not math.isfinite(value):
            raise MobileZipformerAdapterError()
        return value

    @staticmethod
    def _duration_ms(start: float, end: float) -> float:
        duration = end - start
        if not math.isfinite(duration) or duration < 0.0:
            raise MobileZipformerAdapterError()
        return duration * 1000.0

    def _timed(self, function: Callable[[], _T]) -> tuple[_T, float]:
        started = self._now()
        result = function()
        finished = self._now()
        return result, self._duration_ms(started, finished)

    def _drain(self, recognizer: object, stream: object) -> float:
        started = self._now()
        for _ in range(_MAX_DECODE_CALLS_PER_DRAIN):
            if not recognizer.is_ready(stream):
                return self._duration_ms(started, self._now())
            recognizer.decode_stream(stream)
        raise MobileZipformerAdapterError()

    def _result(self, recognizer: object, stream: object) -> tuple[str, float]:
        value, elapsed_ms = self._timed(lambda: recognizer.get_result(stream))
        if not isinstance(value, str):
            raise MobileZipformerAdapterError()
        text = value.strip()
        if len(text) > MAX_HYPOTHESIS_CHARS:
            raise MobileZipformerAdapterError()
        return text, elapsed_ms

    def transcribe(
        self,
        request: TranscribeRequest,
        *,
        emit_partial: Callable[[int, MobileZipformerPartial], None],
    ) -> MobileZipformerCase:
        recognizer = self._recognizer
        config = self.config
        if (
            self._closed
            or recognizer is None
            or request.protocol_version != MOBILE_ZIPFORMER_ENDPOINT_PROTOCOL_VERSION
            or request.stream.chunk_samples != config.native_chunk_samples
            or request.stream.tail_padding_samples > config.maximum_tail_padding_samples
        ):
            raise MobileZipformerAdapterError()
        try:
            source = _read_pcm(request)
        except SherpaZipformerAdapterError:
            raise MobileZipformerAdapterError() from None
        if (
            request.pcm.samples + request.stream.tail_padding_samples
            > MAX_STREAM_SAMPLES
        ):
            raise MobileZipformerAdapterError()

        case_started = self._now()
        stream: object | None = None
        compute_ms = 0.0
        partials: list[MobileZipformerPartial] = []
        observations: list[MobileZipformerEndpointObservation] = []
        deadline_misses = 0
        max_backlog_ms = 0.0
        source_consumed = 0
        tail_consumed = 0
        chunks_yielded = 0
        terminal_endpoint = False
        last_hypothesis = ""
        finalization_started = case_started
        try:
            stream, create_ms = self._timed(recognizer.create_stream)
            compute_ms += create_ms
            if stream is None:
                raise MobileZipformerAdapterError()
            stream_started = self._now()

            def process_chunk(chunk: list[float], *, source_complete: bool) -> bool:
                nonlocal compute_ms
                nonlocal chunks_yielded
                nonlocal deadline_misses
                nonlocal last_hypothesis
                nonlocal max_backlog_ms
                samples_before = source_consumed + tail_consumed - len(chunk)
                if request.stream.pace == "realtime":
                    target = stream_started + samples_before / config.sample_rate
                    before_wait = self._now()
                    if before_wait < target:
                        self._sleeper(target - before_wait)
                    if self._now() < target:
                        raise MobileZipformerAdapterError()
                native = self._numpy.asarray(chunk, dtype="float32")
                _, accept_ms = self._timed(
                    lambda: stream.accept_waveform(config.sample_rate, native)
                )
                drain_ms = self._drain(recognizer, stream)
                text, result_ms = self._result(recognizer, stream)
                endpoint, endpoint_ms = self._timed(
                    lambda: recognizer.is_endpoint(stream)
                )
                if type(endpoint) is not bool:
                    raise MobileZipformerAdapterError()
                compute_ms += accept_ms + drain_ms + result_ms + endpoint_ms
                chunks_yielded += 1
                observed_sample = source_consumed + tail_consumed
                reset_ms = 0.0
                if endpoint:
                    if len(observations) >= MAX_MOBILE_ZIPFORMER_ENDPOINT_OBSERVATIONS:
                        raise MobileZipformerAdapterError()
                    try:
                        _, reset_ms = self._timed(lambda: recognizer.reset(stream))
                    except Exception:
                        raise MobileZipformerAdapterError() from None
                    compute_ms += reset_ms
                    observations.append(
                        MobileZipformerEndpointObservation(
                            text=text,
                            observed_sample=observed_sample,
                            source_complete=source_complete,
                        )
                    )
                if text != last_hypothesis:
                    if len(partials) >= MAX_PARTIALS_PER_CASE:
                        raise MobileZipformerAdapterError()
                    partial = MobileZipformerPartial(
                        after_samples=observed_sample,
                        text=text,
                        elapsed_ms=self._duration_ms(case_started, self._now()),
                        decode_ms=drain_ms + result_ms + endpoint_ms + reset_ms,
                    )
                    emit_partial(len(partials), partial)
                    partials.append(partial)
                last_hypothesis = text
                if request.stream.pace == "realtime":
                    deadline = stream_started + observed_sample / config.sample_rate
                    backlog_ms = max(
                        0.0,
                        self._duration_ms(deadline, max(deadline, self._now())),
                    )
                    max_backlog_ms = max(max_backlog_ms, backlog_ms)
                    if backlog_ms > 0.0:
                        deadline_misses += 1
                return endpoint

            for offset in range(0, request.pcm.samples, config.native_chunk_samples):
                count = min(
                    config.native_chunk_samples,
                    request.pcm.samples - offset,
                )
                source_consumed += count
                endpoint = process_chunk(
                    list(source[offset : offset + count]),
                    source_complete=source_consumed == request.pcm.samples,
                )
                if endpoint and source_consumed == request.pcm.samples:
                    terminal_endpoint = True
                    break
            if source_consumed != request.pcm.samples:
                raise MobileZipformerAdapterError()

            finalization_started = self._now()
            if not terminal_endpoint:
                for offset in range(
                    0,
                    request.stream.tail_padding_samples,
                    config.native_chunk_samples,
                ):
                    count = min(
                        config.native_chunk_samples,
                        request.stream.tail_padding_samples - offset,
                    )
                    tail_consumed += count
                    if process_chunk([0.0] * count, source_complete=True):
                        terminal_endpoint = True
                        break
        except MobileZipformerAdapterError:
            raise
        except Exception:
            raise MobileZipformerAdapterError() from None

        final_text = " ".join(
            observation.text for observation in observations if observation.text
        )
        if len(final_text) > MAX_HYPOTHESIS_CHARS:
            raise MobileZipformerAdapterError()
        finished = self._now()
        endpoint_sample = (
            observations[-1].observed_sample if terminal_endpoint else None
        )
        try:
            resources = _resource_usage()
        except SherpaZipformerAdapterError:
            raise MobileZipformerAdapterError() from None
        return MobileZipformerCase(
            partials=tuple(partials),
            final=final_text,
            elapsed_ms=self._duration_ms(case_started, finished),
            finalization_ms=self._duration_ms(finalization_started, finished),
            compute_ms=compute_ms,
            deadline_misses=deadline_misses,
            max_backlog_ms=max_backlog_ms,
            resources=resources,
            source_samples=request.pcm.samples,
            source_samples_consumed=source_consumed,
            declared_tail_samples=request.stream.tail_padding_samples,
            tail_samples_consumed=tail_consumed,
            chunks_yielded=chunks_yielded,
            model_padding_samples=0,
            endpoint_observations=tuple(observations),
            endpoint_reason=("endpoint" if terminal_endpoint else "tail_exhausted"),
            native_endpoint=terminal_endpoint,
            endpoint_sample=endpoint_sample,
            endpoint_latency_ms=(tail_consumed / 16.0 if terminal_endpoint else None),
            authoritative=terminal_endpoint,
        )


__all__ = [
    "MobileZipformerAdapter",
    "MobileZipformerAdapterError",
    "MobileZipformerCase",
    "MobileZipformerConfig",
    "MobileZipformerPartial",
    "MobileZipformerReady",
    "SherpaZipformerAdapter",
    "SherpaZipformerAdapterError",
    "SherpaZipformerCase",
    "SherpaZipformerConfig",
    "SherpaZipformerPartial",
    "SherpaZipformerReady",
]
