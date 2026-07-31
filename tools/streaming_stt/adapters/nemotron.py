"""Exact, bounded adapter for Nemotron 3.5 cache-aware streaming ASR.

The disposable candidate runtime is imported only while constructing an
adapter.  Streaming callbacks are synchronous: there is no adapter-owned
producer thread or hypothesis queue, and the worker supervisor remains the
hard case-timeout and process-reaping boundary.
"""

from __future__ import annotations

from array import array
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import importlib
from importlib import metadata
import math
from pathlib import Path
import platform
import sys
import threading
import time
from typing import TypeVar

from ..bounded_io import BoundedReadError, read_regular_bounded
from ..manifest import NemotronConfig
from ..protocol import (
    MAX_HYPOTHESIS_CHARS,
    MAX_PARTIALS_PER_CASE,
    MAX_PCM_BYTES,
    MAX_RESOURCE_MB,
    MAX_STREAM_SAMPLES,
    ResourceUsage,
    TranscribeRequest,
)


_EXPECTED_TRANSFORMERS_DISTRIBUTION = "transformers"
_EXPECTED_LIBROSA_DISTRIBUTION = "librosa"
_SAMPLE_RATE = 16_000
_SUPPORTED_LOOKAHEADS = frozenset({0, 3, 6, 13})
_MAX_STREAMER_TOKENS = 65_536
_MAX_TOKEN_ID = (1 << 31) - 1
_EXPECTED_PARAMETERS = 637_997_088
_EXPECTED_COMPUTE_CAPABILITY = (8, 9)
_MEBIBYTE = 1024.0 * 1024.0
_T = TypeVar("_T")


class NemotronAdapterError(RuntimeError):
    """A detail-free candidate load or streaming failure."""


@dataclass(frozen=True)
class NemotronStreamGeometry:
    sample_rate_hz: int
    hop_length_samples: int
    n_fft_samples: int
    win_length_samples: int
    first_chunk_frames: int
    chunk_frames: int
    first_window_samples: int
    window_samples: int
    stride_samples: int
    streaming_latency_ms: int

    def as_dict(self) -> dict[str, int]:
        return {
            "sample_rate_hz": self.sample_rate_hz,
            "hop_length_samples": self.hop_length_samples,
            "n_fft_samples": self.n_fft_samples,
            "win_length_samples": self.win_length_samples,
            "first_chunk_frames": self.first_chunk_frames,
            "chunk_frames": self.chunk_frames,
            "first_window_samples": self.first_window_samples,
            "window_samples": self.window_samples,
            "stride_samples": self.stride_samples,
            "streaming_latency_ms": self.streaming_latency_ms,
        }


@dataclass(frozen=True)
class NemotronReady:
    model_load_ms: float
    resources: ResourceUsage
    stream_geometry: NemotronStreamGeometry


@dataclass(frozen=True)
class NemotronPartial:
    """One provisional cumulative decode from the synchronous streamer."""

    after_samples: int
    text: str = field(repr=False)
    elapsed_ms: float
    decode_ms: float


@dataclass(frozen=True)
class NemotronCase:
    """One authoritative-final candidate result."""

    partials: tuple[NemotronPartial, ...] = field(repr=False)
    final: str = field(repr=False)
    elapsed_ms: float
    finalization_ms: float
    compute_ms: float
    deadline_misses: int
    max_backlog_ms: float
    model_padding_samples: int
    chunks_yielded: int
    resources: ResourceUsage


@dataclass(frozen=True)
class _CandidateApi:
    numpy: object
    torch: object
    transformers: object


@dataclass(frozen=True)
class _ChunkSpec:
    source_start: int
    source_count: int
    unique_offset: int
    after_samples: int
    first: bool

    @property
    def source_end(self) -> int:
        return self.source_start + self.source_count


@dataclass
class _CaseState:
    samples_seen: int = 0
    chunks_yielded: int = 0
    pacing_sleep_sec: float = 0.0
    emit_sec: float = 0.0
    deadline_misses: int = 0
    max_backlog_ms: float = 0.0
    audio_exhausted_at: float | None = None


def _candidate_api(
    expected_transformers_version: str,
    expected_librosa_version: str,
) -> _CandidateApi:
    """Import the exact disposable runtime only inside its worker process."""

    try:
        installed = metadata.version(_EXPECTED_TRANSFORMERS_DISTRIBUTION)
        installed_librosa = metadata.version(_EXPECTED_LIBROSA_DISTRIBUTION)
        if (
            installed != expected_transformers_version
            or installed_librosa != expected_librosa_version
        ):
            raise NemotronAdapterError()
        numpy = importlib.import_module("numpy")
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except (
        ImportError,
        metadata.PackageNotFoundError,
        NemotronAdapterError,
    ):
        raise NemotronAdapterError() from None
    if getattr(transformers, "__version__", None) != expected_transformers_version:
        raise NemotronAdapterError()
    return _CandidateApi(
        numpy=numpy,
        torch=torch,
        transformers=transformers,
    )


def _host_resources() -> tuple[float, int]:
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
        raise NemotronAdapterError()
    return rss_mb, threads


def _read_pcm(request: TranscribeRequest) -> array[float]:
    try:
        snapshot = read_regular_bounded(
            request.pcm.path,
            maximum_bytes=MAX_PCM_BYTES,
            expected_bytes=request.pcm.size_bytes,
        )
    except BoundedReadError:
        raise NemotronAdapterError() from None
    raw = snapshot.data
    if (
        len(raw) != request.pcm.size_bytes
        or hashlib.sha256(raw).hexdigest() != request.pcm.sha256
    ):
        raise NemotronAdapterError()
    samples = array("f")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    if len(samples) != request.pcm.samples or any(
        not math.isfinite(sample) for sample in samples
    ):
        raise NemotronAdapterError()
    return samples


def _strict_positive_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NemotronAdapterError()
    return value


def _strict_nonnegative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise NemotronAdapterError()
    return value


def _token_ids(value: object) -> list[int]:
    """Normalize one batch-one streamer callback without importing torch."""

    try:
        detach = getattr(value, "detach", None)
        if detach is not None:
            value = detach()
        cpu = getattr(value, "cpu", None)
        if cpu is not None:
            value = cpu()
        tolist = getattr(value, "tolist", None)
        if tolist is not None:
            value = tolist()
    except Exception:
        raise NemotronAdapterError() from None

    if not isinstance(value, (list, tuple)) or not value:
        raise NemotronAdapterError()
    if isinstance(value[0], (list, tuple)):
        if len(value) != 1 or not value[0]:
            raise NemotronAdapterError()
        raw_ids: Sequence[object] = value[0]
    else:
        # After the initial prompt, GenerationMixin supplies one token per
        # batch row.  More than one value would mean batch size is not one.
        if len(value) != 1:
            raise NemotronAdapterError()
        raw_ids = value
    result: list[int] = []
    for raw_id in raw_ids:
        if (
            isinstance(raw_id, bool)
            or not isinstance(raw_id, int)
            or not 0 <= raw_id <= _MAX_TOKEN_ID
        ):
            raise NemotronAdapterError()
        result.append(raw_id)
    return result


class _SynchronousSnapshotStreamer:
    """Bound cumulative token state and synchronously publish full snapshots."""

    def __init__(
        self,
        *,
        decode: Callable[[list[int]], str],
        should_decode: Callable[[], bool],
        publish: Callable[[str, float], None],
        clock: Callable[[], float],
    ) -> None:
        self._decode = decode
        self._should_decode = should_decode
        self._publish = publish
        self._clock = clock
        self._tokens: list[int] = []
        self._ended = False

    def _now(self) -> float:
        try:
            value = float(self._clock())
        except (TypeError, ValueError, OverflowError):
            raise NemotronAdapterError() from None
        if not math.isfinite(value):
            raise NemotronAdapterError()
        return value

    def put(self, value: object) -> None:
        if self._ended:
            raise NemotronAdapterError()
        additions = _token_ids(value)
        if len(self._tokens) + len(additions) > _MAX_STREAMER_TOKENS:
            raise NemotronAdapterError()
        self._tokens.extend(additions)
        if not self._should_decode():
            return
        started = self._now()
        try:
            text = self._decode(list(self._tokens))
        except NemotronAdapterError:
            raise
        except Exception:
            raise NemotronAdapterError() from None
        finished = self._now()
        if (
            not isinstance(text, str)
            or len(text) > MAX_HYPOTHESIS_CHARS
            or finished < started
        ):
            raise NemotronAdapterError()
        self._publish(text.strip(), (finished - started) * 1000.0)

    def end(self) -> None:
        self._ended = True

    @property
    def ended(self) -> bool:
        return self._ended


class NemotronAdapter:
    """One exact CUDA model serving one bounded case synchronously at a time."""

    def __init__(
        self,
        model_root: Path,
        *,
        config: NemotronConfig,
        api: object | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(config, NemotronConfig):
            raise NemotronAdapterError()
        try:
            resolved_model_root = Path(model_root).resolve(strict=True)
        except (OSError, RuntimeError):
            raise NemotronAdapterError() from None
        if not resolved_model_root.is_dir():
            raise NemotronAdapterError()

        self.config = config
        self.model_root = resolved_model_root
        self._clock = clock
        self._sleeper = sleeper
        self._closed = False
        self._failed = False
        self._case_lock = threading.Lock()
        self._api: object | None = None
        self._processor: object | None = None
        self._model: object | None = None
        self._device: object | None = None
        self._dtype: object | None = None

        load_started = self._now()
        candidate = (
            api
            if api is not None
            else _candidate_api(
                config.transformers_version,
                config.librosa_version,
            )
        )
        # Retain the runtime before loading so any partial CUDA allocation can
        # be released on a fail-closed identity or geometry check.
        self._api = candidate
        try:
            self._load_candidate(candidate)
            self._cuda_synchronize()
            load_finished = self._now()
            self.ready = NemotronReady(
                model_load_ms=self._duration_ms(load_started, load_finished),
                resources=self._resources(),
                stream_geometry=self._stream_geometry(),
            )
        except Exception as error:
            self._release_candidate()
            if isinstance(error, NemotronAdapterError):
                raise
            raise NemotronAdapterError() from None

    def _load_candidate(self, candidate: object) -> None:
        numpy = getattr(candidate, "numpy", None)
        torch = getattr(candidate, "torch", None)
        transformers = getattr(candidate, "transformers", None)
        if (
            numpy is None
            or torch is None
            or transformers is None
            or getattr(transformers, "__version__", None)
            != self.config.transformers_version
            or platform.python_version() != self.config.python_version
            or getattr(torch, "__version__", None) != self.config.torch_version
            or getattr(getattr(torch, "version", None), "cuda", None)
            != self.config.cuda_version
            or self.config.device != "cuda:0"
            or self.config.dtype != "float32"
        ):
            raise NemotronAdapterError()
        cuda = getattr(torch, "cuda", None)
        if cuda is None or cuda.is_available() is not True:
            raise NemotronAdapterError()
        device_count = _strict_positive_int(cuda.device_count())
        current_device = _strict_nonnegative_int(cuda.current_device())
        if device_count != 1 or current_device != 0:
            raise NemotronAdapterError()
        try:
            device = torch.device(self.config.device)
            dtype = getattr(torch, self.config.dtype)
            capability = tuple(cuda.get_device_capability(device))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            raise NemotronAdapterError() from None
        if capability != _EXPECTED_COMPUTE_CAPABILITY:
            raise NemotronAdapterError()

        processor_type = getattr(transformers, "Nemotron3_5AsrProcessor", None)
        model_type = getattr(transformers, "Nemotron3_5AsrForRNNT", None)
        if processor_type is None or model_type is None:
            raise NemotronAdapterError()
        load_kwargs = {
            "revision": self.config.model_revision,
            "local_files_only": True,
            "trust_remote_code": False,
        }
        try:
            processor = processor_type.from_pretrained(
                str(self.model_root),
                **load_kwargs,
            )
            model = model_type.from_pretrained(
                str(self.model_root),
                **load_kwargs,
                use_safetensors=True,
                weights_only=True,
                device_map=None,
                dtype=dtype,
            )
            model = model.to(device=device, dtype=dtype)
            model = model.eval()
        except Exception:
            raise NemotronAdapterError() from None
        if (
            getattr(model, "training", None) is not False
            or not self._same_device(getattr(model, "device", None), device)
            or getattr(model, "dtype", None) != dtype
        ):
            raise NemotronAdapterError()
        try:
            parameters = tuple(model.parameters())
            buffers = tuple(model.buffers())
            parameter_count = sum(
                _strict_nonnegative_int(parameter.numel()) for parameter in parameters
            )
        except Exception:
            raise NemotronAdapterError() from None
        if parameter_count != _EXPECTED_PARAMETERS or any(
            not self._same_device(getattr(tensor, "device", None), device)
            for tensor in (*parameters, *buffers)
        ):
            raise NemotronAdapterError()

        supported = getattr(processor, "supported_num_lookahead_tokens", None)
        if (
            not isinstance(supported, (list, tuple))
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in supported
            )
            or len(supported) != len(_SUPPORTED_LOOKAHEADS)
            or frozenset(supported) != _SUPPORTED_LOOKAHEADS
        ):
            raise NemotronAdapterError()
        prompt_dictionary = getattr(processor, "prompt_dictionary", None)
        if (
            not isinstance(prompt_dictionary, Mapping)
            or self.config.language not in prompt_dictionary
        ):
            raise NemotronAdapterError()
        try:
            processor.set_num_lookahead_tokens(self.config.lookahead_tokens)
        except Exception:
            raise NemotronAdapterError() from None
        if (
            getattr(processor, "default_num_lookahead_tokens", None)
            != self.config.lookahead_tokens
        ):
            raise NemotronAdapterError()

        encoder_config = getattr(
            getattr(model, "config", None),
            "encoder_config",
            None,
        )
        model_supported = getattr(
            encoder_config,
            "supported_num_lookahead_tokens",
            None,
        )
        if (
            not isinstance(model_supported, (list, tuple))
            or len(model_supported) != len(_SUPPORTED_LOOKAHEADS)
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in model_supported
            )
            or frozenset(model_supported) != _SUPPORTED_LOOKAHEADS
        ):
            raise NemotronAdapterError()

        self._api = candidate
        self._processor = processor
        self._model = model
        self._device = device
        self._dtype = dtype
        self._validate_native_geometry()

    @staticmethod
    def _same_device(actual: object, expected: object) -> bool:
        try:
            return str(actual) == str(expected)
        except Exception:
            return False

    def _validate_native_geometry(self) -> None:
        processor = self._processor
        if processor is None:
            raise NemotronAdapterError()
        extractor = getattr(processor, "feature_extractor", None)
        if extractor is None:
            raise NemotronAdapterError()
        sample_rate = _strict_positive_int(getattr(extractor, "sampling_rate", None))
        hop_length = _strict_positive_int(getattr(extractor, "hop_length", None))
        n_fft = _strict_positive_int(getattr(extractor, "n_fft", None))
        win_length = _strict_positive_int(getattr(extractor, "win_length", None))
        first_frames = _strict_positive_int(
            getattr(processor, "num_mel_frames_first_audio_chunk", None)
        )
        per_frames = _strict_positive_int(
            getattr(processor, "num_mel_frames_per_audio_chunk", None)
        )
        first_samples = _strict_positive_int(
            getattr(processor, "num_samples_first_audio_chunk", None)
        )
        per_samples = _strict_positive_int(
            getattr(processor, "num_samples_per_audio_chunk", None)
        )
        streaming_latency_ms = _strict_positive_int(
            getattr(processor, "streaming_latency_ms", None)
        )
        if (
            sample_rate != self.config.native_sample_rate_hz
            or hop_length != self.config.native_hop_length_samples
            or n_fft != self.config.native_n_fft_samples
            or win_length != self.config.native_win_length_samples
            or first_frames != self.config.native_first_chunk_frames
            or per_frames != self.config.native_chunk_frames
            or first_samples != self.config.native_first_window_samples
            or per_samples != self.config.native_window_samples
            or streaming_latency_ms != self.config.streaming_latency_ms
            or sample_rate != _SAMPLE_RATE
            or first_samples != (first_frames - 1) * hop_length + win_length // 2
            or per_samples != per_frames * hop_length + win_length
            or per_frames * hop_length != self.config.native_stride_samples
        ):
            raise NemotronAdapterError()

    def _stream_geometry(self) -> NemotronStreamGeometry:
        return NemotronStreamGeometry(
            sample_rate_hz=_strict_positive_int(self.config.native_sample_rate_hz),
            hop_length_samples=_strict_positive_int(
                self.config.native_hop_length_samples
            ),
            n_fft_samples=_strict_positive_int(self.config.native_n_fft_samples),
            win_length_samples=_strict_positive_int(
                self.config.native_win_length_samples
            ),
            first_chunk_frames=_strict_positive_int(
                self.config.native_first_chunk_frames
            ),
            chunk_frames=_strict_positive_int(self.config.native_chunk_frames),
            first_window_samples=_strict_positive_int(
                self.config.native_first_window_samples
            ),
            window_samples=_strict_positive_int(self.config.native_window_samples),
            stride_samples=_strict_positive_int(self.config.native_stride_samples),
            streaming_latency_ms=_strict_positive_int(self.config.streaming_latency_ms),
        )

    def __enter__(self) -> NemotronAdapter:
        if self._closed or self._failed:
            raise NemotronAdapterError()
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

    def _release_candidate(self) -> None:
        api = self._api
        self._model = None
        self._processor = None
        self._device = None
        self._dtype = None
        self._api = None
        if api is None:
            return
        try:
            api.torch.cuda.empty_cache()
        except Exception:
            pass

    def close(self) -> None:
        if self._closed:
            return
        if not self._case_lock.acquire(blocking=False):
            raise NemotronAdapterError()
        try:
            self._closed = True
            api = self._api
            self._model = None
            self._processor = None
            self._device = None
            self._dtype = None
            self._api = None
            if api is not None:
                try:
                    api.torch.cuda.empty_cache()
                except Exception:
                    raise NemotronAdapterError() from None
        finally:
            self._case_lock.release()

    def _now(self) -> float:
        try:
            value = float(self._clock())
        except (TypeError, ValueError, OverflowError):
            raise NemotronAdapterError() from None
        if not math.isfinite(value):
            raise NemotronAdapterError()
        return value

    @staticmethod
    def _duration_ms(start: float, end: float) -> float:
        duration = end - start
        if not math.isfinite(duration) or duration < 0.0:
            raise NemotronAdapterError()
        return duration * 1000.0

    def _timed(self, function: Callable[[], _T]) -> tuple[_T, float]:
        started = self._now()
        result = function()
        finished = self._now()
        return result, self._duration_ms(started, finished)

    def _cuda_synchronize(self) -> None:
        if self._api is None or self._device is None:
            return
        try:
            self._api.torch.cuda.synchronize(self._device)
        except Exception:
            raise NemotronAdapterError() from None

    def _reset_peak_vram(self) -> None:
        if self._api is None or self._device is None:
            raise NemotronAdapterError()
        try:
            self._api.torch.cuda.reset_peak_memory_stats(self._device)
        except Exception:
            raise NemotronAdapterError() from None

    def _resources(self) -> ResourceUsage:
        if self._api is None or self._device is None:
            raise NemotronAdapterError()
        rss_mb, threads = _host_resources()
        try:
            allocated = self._api.torch.cuda.max_memory_allocated(self._device)
        except Exception:
            raise NemotronAdapterError() from None
        if (
            isinstance(allocated, bool)
            or not isinstance(allocated, (int, float))
            or not math.isfinite(float(allocated))
            or allocated < 0
        ):
            raise NemotronAdapterError()
        vram_mb = float(allocated) / _MEBIBYTE
        if vram_mb > MAX_RESOURCE_MB:
            raise NemotronAdapterError()
        return ResourceUsage(
            rss_mb=rss_mb,
            threads=threads,
            vram_mb=vram_mb,
        )

    def _native_geometry(self) -> tuple[int, int, int, int, int, int]:
        processor = self._processor
        if processor is None:
            raise NemotronAdapterError()
        self._validate_native_geometry()
        extractor = processor.feature_extractor
        return (
            _strict_positive_int(processor.num_mel_frames_first_audio_chunk),
            _strict_positive_int(processor.num_mel_frames_per_audio_chunk),
            _strict_positive_int(processor.num_samples_first_audio_chunk),
            _strict_positive_int(processor.num_samples_per_audio_chunk),
            _strict_positive_int(extractor.hop_length),
            _strict_positive_int(extractor.n_fft),
        )

    def _chunk_specs(self, total_samples: int) -> tuple[_ChunkSpec, ...]:
        (
            first_frames,
            per_frames,
            first_samples,
            per_samples,
            hop_length,
            n_fft,
        ) = self._native_geometry()
        specs = [
            _ChunkSpec(
                source_start=0,
                source_count=first_samples,
                unique_offset=0,
                after_samples=min(total_samples, first_samples),
                first=True,
            )
        ]
        mel_frame_index = first_frames
        while specs[-1].source_end < total_samples:
            source_start = mel_frame_index * hop_length - n_fft // 2
            source_end = source_start + per_samples
            if source_end <= specs[-1].source_end:
                raise NemotronAdapterError()
            specs.append(
                _ChunkSpec(
                    source_start=source_start,
                    source_count=per_samples,
                    # This is the first sample not already covered by the
                    # preceding native window, not its overlapped raw start.
                    unique_offset=specs[-1].source_end,
                    after_samples=min(total_samples, source_end),
                    first=False,
                )
            )
            mel_frame_index += per_frames
            if len(specs) > MAX_STREAM_SAMPLES:
                raise NemotronAdapterError()
        return tuple(specs)

    @staticmethod
    def _sample_window(
        source: array[float],
        *,
        spec: _ChunkSpec,
    ) -> list[float]:
        left_padding = max(0, -spec.source_start)
        source_start = min(max(0, spec.source_start), len(source))
        source_end = min(max(0, spec.source_end), len(source))
        window = [0.0] * left_padding
        window.extend(source[source_start:source_end])
        window.extend([0.0] * (spec.source_count - len(window)))
        if len(window) != spec.source_count:
            raise NemotronAdapterError()
        return window

    def _processor_inputs(
        self,
        source: array[float],
        *,
        spec: _ChunkSpec,
    ) -> object:
        if (
            self._api is None
            or self._processor is None
            or self._device is None
            or self._dtype is None
        ):
            raise NemotronAdapterError()
        window = self._sample_window(source, spec=spec)
        try:
            audio = self._api.numpy.asarray(
                window,
                dtype=self._api.numpy.float32,
            )
            inputs = self._processor(
                audio,
                sampling_rate=_SAMPLE_RATE,
                is_streaming=True,
                is_first_audio_chunk=spec.first,
                language=self.config.language,
                return_tensors="pt",
            )
            inputs = inputs.to(device=self._device, dtype=self._dtype)
        except Exception:
            raise NemotronAdapterError() from None
        return inputs

    @staticmethod
    def _input_features(inputs: object) -> object:
        try:
            return inputs["input_features"]
        except (KeyError, TypeError):
            try:
                return inputs.input_features
            except AttributeError:
                raise NemotronAdapterError() from None

    @staticmethod
    def _feature_shape(features: object) -> tuple[int, int, int]:
        raw_shape = getattr(features, "shape", None)
        if not isinstance(raw_shape, Sequence) or len(raw_shape) != 3:
            raise NemotronAdapterError()
        try:
            shape = tuple(int(dimension) for dimension in raw_shape)
        except (TypeError, ValueError, OverflowError):
            raise NemotronAdapterError() from None
        if shape[0] != 1 or shape[1] <= 0 or shape[2] <= 0:
            raise NemotronAdapterError()
        return shape

    def _prepared_features(self, inputs: object, *, first: bool) -> object:
        processor = self._processor
        if processor is None:
            raise NemotronAdapterError()
        features = self._input_features(inputs)
        shape = self._feature_shape(features)
        required = (
            processor.num_mel_frames_first_audio_chunk
            if first
            else processor.num_mel_frames_per_audio_chunk
        )
        required = _strict_positive_int(required)
        if first:
            if shape[1] < required:
                raise NemotronAdapterError()
            try:
                features = features[:, :required, :]
            except Exception:
                raise NemotronAdapterError() from None
            if self._feature_shape(features)[1] != required:
                raise NemotronAdapterError()
        elif shape[1] != required:
            raise NemotronAdapterError()
        return features

    def _pace_chunk(
        self,
        *,
        state: _CaseState,
        stream_started: float,
        spec: _ChunkSpec,
        pace: str,
    ) -> None:
        if pace != "realtime":
            return
        target = stream_started + spec.unique_offset / _SAMPLE_RATE
        before = self._now()
        if before < target:
            try:
                self._sleeper(target - before)
            except Exception:
                raise NemotronAdapterError() from None
            after = self._now()
            if after < target:
                raise NemotronAdapterError()
            state.pacing_sleep_sec += after - before

    def _account_chunk_deadline(
        self,
        *,
        state: _CaseState,
        stream_started: float,
        spec: _ChunkSpec,
        pace: str,
    ) -> None:
        if pace != "realtime":
            return
        deadline = stream_started + spec.after_samples / _SAMPLE_RATE
        now = self._now()
        backlog_ms = self._duration_ms(deadline, max(deadline, now))
        if backlog_ms > 0.0:
            state.deadline_misses += 1
            state.max_backlog_ms = max(state.max_backlog_ms, backlog_ms)

    def transcribe(
        self,
        request: TranscribeRequest,
        *,
        emit_partial: Callable[[int, NemotronPartial], None],
    ) -> NemotronCase:
        if self._closed or self._failed:
            raise NemotronAdapterError()
        if not self._case_lock.acquire(blocking=False):
            raise NemotronAdapterError()
        try:
            return self._transcribe_locked(request, emit_partial=emit_partial)
        except Exception as error:
            self._failed = True
            if isinstance(error, NemotronAdapterError):
                raise
            raise NemotronAdapterError() from None
        finally:
            self._case_lock.release()

    def _transcribe_locked(
        self,
        request: TranscribeRequest,
        *,
        emit_partial: Callable[[int, NemotronPartial], None],
    ) -> NemotronCase:
        if (
            self._api is None
            or self._processor is None
            or self._model is None
            or self._device is None
            or self._dtype is None
        ):
            raise NemotronAdapterError()
        source = _read_pcm(request)
        total_samples = request.pcm.samples + request.stream.tail_padding_samples
        if total_samples > MAX_STREAM_SAMPLES:
            raise NemotronAdapterError()
        specs = self._chunk_specs(total_samples)
        self._validate_native_geometry()
        unique_stride = _strict_positive_int(self.config.native_stride_samples)
        if request.stream.chunk_samples != unique_stride:
            raise NemotronAdapterError()

        case_started = self._now()
        stream_started = case_started
        state = _CaseState()
        self._reset_peak_vram()
        self._pace_chunk(
            state=state,
            stream_started=stream_started,
            spec=specs[0],
            pace=request.stream.pace,
        )
        first_inputs, first_processor_ms = self._timed(
            lambda: self._processor_inputs(source, spec=specs[0])
        )
        first_features = self._prepared_features(first_inputs, first=True)

        partials: list[NemotronPartial] = []
        last_partial = ""
        interval_samples = (
            request.stream.partial_interval_ms * _SAMPLE_RATE + 999
        ) // 1000
        next_partial_samples = interval_samples

        def publish_snapshot(text: str, decode_ms: float) -> None:
            nonlocal last_partial, next_partial_samples
            if state.samples_seen < next_partial_samples:
                return
            # The initial decoder-start/blank callback often decodes empty.
            # Keep the interval eligible until a usable cumulative snapshot
            # arrives instead of accidentally suppressing that whole chunk.
            if not text:
                return
            while next_partial_samples <= state.samples_seen:
                next_partial_samples += interval_samples
            if text == last_partial or len(partials) >= MAX_PARTIALS_PER_CASE:
                return
            last_partial = text
            partial = NemotronPartial(
                after_samples=state.samples_seen,
                text=text,
                elapsed_ms=self._duration_ms(case_started, self._now()),
                decode_ms=decode_ms,
            )
            emit_started = self._now()
            try:
                emit_partial(len(partials), partial)
            except Exception:
                raise NemotronAdapterError() from None
            emit_finished = self._now()
            emit_duration = emit_finished - emit_started
            if not math.isfinite(emit_duration) or emit_duration < 0.0:
                raise NemotronAdapterError()
            state.emit_sec += emit_duration
            partials.append(partial)

        def provisional_decode(token_ids: list[int]) -> str:
            try:
                text = self._processor.decode(
                    token_ids,
                    skip_special_tokens=True,
                )
            except Exception:
                raise NemotronAdapterError() from None
            if not isinstance(text, str):
                raise NemotronAdapterError()
            return text

        streamer = _SynchronousSnapshotStreamer(
            decode=provisional_decode,
            should_decode=lambda: (
                state.samples_seen >= next_partial_samples
                and len(partials) < MAX_PARTIALS_PER_CASE
            ),
            publish=publish_snapshot,
            clock=self._clock,
        )

        def input_features_generator() -> Iterator[object]:
            for index, spec in enumerate(specs):
                if index == 0:
                    features = first_features
                else:
                    self._pace_chunk(
                        state=state,
                        stream_started=stream_started,
                        spec=spec,
                        pace=request.stream.pace,
                    )
                    inputs = self._processor_inputs(source, spec=spec)
                    features = self._prepared_features(inputs, first=False)
                state.samples_seen = spec.after_samples
                state.chunks_yielded += 1
                yield features
                self._account_chunk_deadline(
                    state=state,
                    stream_started=stream_started,
                    spec=spec,
                    pace=request.stream.pace,
                )
                if index == len(specs) - 1:
                    state.audio_exhausted_at = self._now()

        try:
            generate_kwargs = dict(first_inputs)
        except (TypeError, ValueError):
            raise NemotronAdapterError() from None
        generate_kwargs["input_features"] = input_features_generator()
        generate_kwargs["num_lookahead_tokens"] = self.config.lookahead_tokens
        generate_kwargs["streamer"] = streamer
        generate_kwargs["return_dict_in_generate"] = True

        self._cuda_synchronize()
        generation_started = self._now()
        try:
            with self._api.torch.inference_mode():
                output = self._model.generate(**generate_kwargs)
        except NemotronAdapterError:
            raise
        except Exception:
            raise NemotronAdapterError() from None
        self._cuda_synchronize()
        generation_finished = self._now()
        if (
            not streamer.ended
            or state.chunks_yielded != len(specs)
            or state.audio_exhausted_at is None
            or state.samples_seen != total_samples
        ):
            raise NemotronAdapterError()

        sequences = getattr(output, "sequences", None)
        if sequences is None:
            raise NemotronAdapterError()
        decoded, final_decode_ms = self._timed(
            lambda: self._processor.batch_decode(
                sequences,
                skip_special_tokens=True,
            )
        )
        if (
            not isinstance(decoded, (list, tuple))
            or len(decoded) != 1
            or not isinstance(decoded[0], str)
        ):
            raise NemotronAdapterError()
        final = decoded[0].strip()
        if len(final) > MAX_HYPOTHESIS_CHARS:
            raise NemotronAdapterError()
        finished = self._now()

        generation_wall_ms = self._duration_ms(
            generation_started,
            generation_finished,
        )
        excluded_ms = (state.pacing_sleep_sec + state.emit_sec) * 1000.0
        generation_compute_ms = generation_wall_ms - excluded_ms
        if not math.isfinite(generation_compute_ms) or generation_compute_ms < 0.0:
            raise NemotronAdapterError()
        model_padding_samples = specs[-1].source_end - total_samples
        if model_padding_samples < 0:
            raise NemotronAdapterError()
        return NemotronCase(
            partials=tuple(partials),
            final=final,
            elapsed_ms=self._duration_ms(case_started, finished),
            finalization_ms=self._duration_ms(
                state.audio_exhausted_at,
                finished,
            ),
            compute_ms=(first_processor_ms + generation_compute_ms + final_decode_ms),
            deadline_misses=state.deadline_misses,
            max_backlog_ms=state.max_backlog_ms,
            model_padding_samples=model_padding_samples,
            chunks_yielded=state.chunks_yielded,
            resources=self._resources(),
        )


__all__ = [
    "NemotronAdapter",
    "NemotronAdapterError",
    "NemotronCase",
    "NemotronConfig",
    "NemotronPartial",
    "NemotronReady",
    "NemotronStreamGeometry",
]
