"""Strict local Moshi 0.2.11 adapter for Kyutai's semantic streaming STT.

The four semantic heads are checked as finite diagnostic observations only.
They never end a stream, authorize a tool, or suppress the exact terminal tail.
The first evaluation cell resamples each complete bounded case before the true
Moshi frame loop; it therefore measures model streaming, not causal transport.
"""

from __future__ import annotations

from array import array
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
import hashlib
import importlib
from importlib import metadata
import math
import os
from pathlib import Path
import platform
import stat
import sys
import threading
import time
from typing import TypeVar

from ..bounded_io import BoundedReadError, read_regular_bounded
from ..manifest import KyutaiConfig
from ..protocol import (
    MAX_HYPOTHESIS_CHARS,
    MAX_PARTIALS_PER_CASE,
    MAX_PCM_BYTES,
    MAX_RESOURCE_MB,
    PROTOCOL_VERSION,
    ResourceUsage,
    TranscribeRequest,
)


_MEBIBYTE = 1024 * 1024
_T = TypeVar("_T")


class KyutaiAdapterError(RuntimeError):
    """A detail-free candidate load or streaming failure."""


@dataclass(frozen=True)
class KyutaiStreamGeometry:
    input_sample_rate_hz: int
    mimi_sample_rate_hz: int
    input_chunk_samples: int
    mimi_frame_samples: int
    resampling_mode: str
    terminal_tail_samples: int


@dataclass(frozen=True)
class KyutaiReady:
    model_load_ms: float
    resources: ResourceUsage
    stream_geometry: KyutaiStreamGeometry


@dataclass(frozen=True)
class KyutaiPartial:
    after_samples: int
    text: str = field(repr=False)
    elapsed_ms: float
    decode_ms: float


@dataclass(frozen=True)
class KyutaiCase:
    partials: tuple[KyutaiPartial, ...] = field(repr=False)
    final: str = field(repr=False)
    elapsed_ms: float
    finalization_ms: float
    compute_ms: float
    deadline_misses: int
    max_backlog_ms: float
    model_padding_samples: int
    chunks_yielded: int
    resources: ResourceUsage
    source_samples: int
    source_samples_consumed: int
    declared_tail_samples: int
    tail_samples_consumed: int
    semantic_head_steps: int


@dataclass(frozen=True)
class _CandidateApi:
    torch: object
    julius: object
    moshi_models: object
    moshi_version: str
    julius_version: str


def _candidate_api(config: KyutaiConfig) -> _CandidateApi:
    """Import only the exact, wheel-receipted runtime in the worker process."""

    try:
        moshi_version = metadata.version("moshi")
        julius_version = metadata.version("julius")
        if (
            moshi_version != config.moshi_version
            or julius_version != config.julius_version
        ):
            raise KyutaiAdapterError()
        torch = importlib.import_module("torch")
        julius = importlib.import_module("julius")
        moshi_models = importlib.import_module("moshi.models")
    except (ImportError, metadata.PackageNotFoundError, KyutaiAdapterError):
        raise KyutaiAdapterError() from None
    return _CandidateApi(
        torch=torch,
        julius=julius,
        moshi_models=moshi_models,
        moshi_version=moshi_version,
        julius_version=julius_version,
    )


def _read_pcm(request: TranscribeRequest) -> array[float]:
    try:
        snapshot = read_regular_bounded(
            request.pcm.path,
            maximum_bytes=MAX_PCM_BYTES,
            expected_bytes=request.pcm.size_bytes,
        )
    except BoundedReadError:
        raise KyutaiAdapterError() from None
    raw = snapshot.data
    if hashlib.sha256(raw).hexdigest() != request.pcm.sha256:
        raise KyutaiAdapterError()
    samples = array("f")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    if len(samples) != request.pcm.samples or any(
        not math.isfinite(sample) for sample in samples
    ):
        raise KyutaiAdapterError()
    return samples


def _bounded_proc_ascii(path: str, *, maximum_bytes: int) -> str:
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        metadata_value = os.fstat(descriptor)
        raw = os.read(descriptor, maximum_bytes + 1)
    except OSError:
        raise KyutaiAdapterError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                raise KyutaiAdapterError() from None
    if not stat.S_ISREG(metadata_value.st_mode) or len(raw) > maximum_bytes:
        raise KyutaiAdapterError()
    try:
        return raw.decode("ascii", errors="strict")
    except UnicodeError:
        raise KyutaiAdapterError() from None


def _host_available_bytes() -> int:
    lines = _bounded_proc_ascii(
        "/proc/meminfo",
        maximum_bytes=16 * 1024,
    ).splitlines()
    matches = [line for line in lines if line.startswith("MemAvailable:")]
    if len(matches) != 1:
        raise KyutaiAdapterError()
    fields = matches[0].split()
    if len(fields) != 3 or fields[2] != "kB":
        raise KyutaiAdapterError()
    try:
        kilobytes = int(fields[1])
    except ValueError:
        raise KyutaiAdapterError() from None
    if kilobytes <= 0:
        raise KyutaiAdapterError()
    return kilobytes * 1024


def _host_resources(torch_api: object, device: object) -> ResourceUsage:
    rss_mb: float | None = None
    threads: int | None = None
    try:
        status = _bounded_proc_ascii(
            "/proc/self/status",
            maximum_bytes=64 * 1024,
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
    except (KyutaiAdapterError, ValueError):
        pass
    if rss_mb is None:
        rss_mb = 0.0
    if threads is None:
        threads = max(1, threading.active_count())
    try:
        cuda = getattr(torch_api, "cuda")
        allocated = float(cuda.max_memory_allocated(device)) / _MEBIBYTE
        reserved = float(cuda.max_memory_reserved(device)) / _MEBIBYTE
        vram_mb = max(allocated, reserved)
    except (AttributeError, RuntimeError, TypeError, ValueError, OverflowError):
        raise KyutaiAdapterError() from None
    if (
        not math.isfinite(rss_mb)
        or not 0.0 <= rss_mb <= MAX_RESOURCE_MB
        or not 1 <= threads <= 4096
        or not math.isfinite(vram_mb)
        or not 0.0 <= vram_mb <= MAX_RESOURCE_MB
    ):
        raise KyutaiAdapterError()
    return ResourceUsage(rss_mb=rss_mb, threads=threads, vram_mb=vram_mb)


def _shape(value: object) -> tuple[int, ...]:
    raw = getattr(value, "shape", None)
    if not isinstance(raw, Sequence):
        raise KyutaiAdapterError()
    try:
        return tuple(int(dimension) for dimension in raw)
    except (TypeError, ValueError, OverflowError):
        raise KyutaiAdapterError() from None


def _scalar_item(value: object) -> object:
    item = getattr(value, "item", None)
    if not callable(item):
        raise KyutaiAdapterError()
    try:
        return item()
    except Exception:
        raise KyutaiAdapterError() from None


class KyutaiAdapter:
    """One shared model with fresh Mimi and LMGen streaming state per case."""

    def __init__(
        self,
        config_path: Path,
        weights_path: Path,
        mimi_path: Path,
        tokenizer_path: Path,
        *,
        config: KyutaiConfig,
        api: object | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleeper: Callable[[float], None] = time.sleep,
        host_available_reader: Callable[[], int] = _host_available_bytes,
        resource_reader: Callable[[object, object], ResourceUsage] = _host_resources,
    ) -> None:
        if not isinstance(config, KyutaiConfig):
            raise KyutaiAdapterError()
        expected_paths = (
            (config_path, config.model_config_filename),
            (weights_path, config.model_weights_filename),
            (mimi_path, config.mimi_weights_filename),
            (tokenizer_path, config.tokenizer_filename),
        )
        resolved: list[Path] = []
        try:
            for path, basename in expected_paths:
                candidate = Path(path).resolve(strict=True)
                if not candidate.is_file() or candidate.name != basename:
                    raise KyutaiAdapterError()
                resolved.append(candidate)
        except (OSError, RuntimeError):
            raise KyutaiAdapterError() from None
        if len({path.parent for path in resolved}) != 1:
            raise KyutaiAdapterError()
        if (
            os.environ.get("NO_TORCH_COMPILE") != config.no_torch_compile_env
            or os.environ.get("NO_CUDA_GRAPH") != config.no_cuda_graph_env
        ):
            raise KyutaiAdapterError()

        self.config = config
        (
            self.config_path,
            self.weights_path,
            self.mimi_path,
            self.tokenizer_path,
        ) = resolved
        self._clock = clock
        self._sleeper = sleeper
        self._resource_reader = resource_reader
        self._closed = False
        self._poisoned = False
        self._case_lock = threading.Lock()
        self._api: object | None = None
        self._device: object | None = None
        self._dtype: object | None = None
        self._mimi: object | None = None
        self._tokenizer: object | None = None
        self._lm: object | None = None
        self._resampler: object | None = None

        load_started = self._now()
        candidate_api = api if api is not None else _candidate_api(config)
        self._api = candidate_api
        try:
            available = host_available_reader()
            if (
                type(available) is not int
                or available < config.minimum_host_available_bytes
            ):
                raise KyutaiAdapterError()
            self._load_candidate(candidate_api)
            self._cuda_synchronize()
            load_finished = self._now()
            self.ready = KyutaiReady(
                model_load_ms=self._duration_ms(load_started, load_finished),
                resources=self._resources(),
                stream_geometry=KyutaiStreamGeometry(
                    input_sample_rate_hz=config.input_sample_rate_hz,
                    mimi_sample_rate_hz=config.mimi_sample_rate_hz,
                    input_chunk_samples=config.input_chunk_samples,
                    mimi_frame_samples=config.mimi_frame_samples,
                    resampling_mode=config.resampling_mode,
                    terminal_tail_samples=config.terminal_tail_samples,
                ),
            )
        except Exception as error:
            self.close()
            if isinstance(error, KyutaiAdapterError):
                raise
            raise KyutaiAdapterError() from None

    def _load_candidate(self, api: object) -> None:
        torch = getattr(api, "torch", None)
        julius = getattr(api, "julius", None)
        moshi_models = getattr(api, "moshi_models", None)
        if (
            torch is None
            or julius is None
            or moshi_models is None
            or getattr(api, "moshi_version", None) != self.config.moshi_version
            or getattr(api, "julius_version", None) != self.config.julius_version
            or platform.python_version() != self.config.python_version
            or getattr(torch, "__version__", None) != self.config.torch_version
            or getattr(getattr(torch, "version", None), "cuda", None)
            != self.config.cuda_version
            or self.config.torch_compile
            or self.config.cuda_graph
            or self.config.local_files_only is not True
        ):
            raise KyutaiAdapterError()
        cuda = getattr(torch, "cuda", None)
        if cuda is None or cuda.is_available() is not True:
            raise KyutaiAdapterError()
        try:
            device = torch.device(self.config.device)
            dtype = getattr(torch, self.config.dtype)
            if cuda.is_bf16_supported() is not True:
                raise KyutaiAdapterError()
            free_bytes, total_bytes = cuda.mem_get_info(device)
        except Exception:
            raise KyutaiAdapterError() from None
        if (
            type(free_bytes) is not int
            or type(total_bytes) is not int
            or free_bytes < self.config.minimum_free_vram_mb * _MEBIBYTE
            or total_bytes < free_bytes
        ):
            raise KyutaiAdapterError()
        try:
            torch.set_num_threads(self.config.num_threads)
            torch.set_num_interop_threads(self.config.num_threads)
            if (
                torch.get_num_threads() != self.config.num_threads
                or torch.get_num_interop_threads() != self.config.num_threads
            ):
                raise KyutaiAdapterError()
            cuda.set_per_process_memory_fraction(
                self.config.maximum_vram_fraction,
                device,
            )
        except Exception:
            raise KyutaiAdapterError() from None

        loaders = getattr(moshi_models, "loaders", None)
        checkpoint_type = getattr(loaders, "CheckpointInfo", None)
        from_hf_repo = getattr(checkpoint_type, "from_hf_repo", None)
        lm_gen_type = getattr(moshi_models, "LMGen", None)
        resampler_type = getattr(julius, "ResampleFrac", None)
        if (
            not callable(from_hf_repo)
            or not callable(lm_gen_type)
            or not callable(resampler_type)
        ):
            raise KyutaiAdapterError()
        try:
            resampler = resampler_type(
                self.config.input_sample_rate_hz,
                self.config.mimi_sample_rate_hz,
            ).to(device=device, dtype=torch.float32)
            checkpoint = from_hf_repo(
                self.config.model_repo_id,
                moshi_weights=self.weights_path,
                mimi_weights=self.mimi_path,
                tokenizer=self.tokenizer_path,
                config_path=self.config_path,
            )
            self._validate_checkpoint(checkpoint)
            mimi = checkpoint.get_mimi(device=self.config.device)
            tokenizer = checkpoint.get_text_tokenizer()
            lm = checkpoint.get_moshi(
                device=self.config.device,
                dtype=dtype,
            )
        except KyutaiAdapterError:
            raise
        except Exception:
            raise KyutaiAdapterError() from None
        if (
            getattr(mimi, "sample_rate", None) != self.config.mimi_sample_rate_hz
            or getattr(mimi, "frame_size", None) != self.config.mimi_frame_samples
            or float(getattr(mimi, "frame_rate", float("nan"))) != 12.5
            or not callable(getattr(mimi, "streaming", None))
            or not callable(getattr(mimi, "encode", None))
            or not callable(getattr(tokenizer, "id_to_piece", None))
            or not callable(getattr(lm, "parameters", None))
            or not callable(resampler)
        ):
            raise KyutaiAdapterError()
        try:
            first_parameter = next(iter(lm.parameters()))
        except Exception:
            raise KyutaiAdapterError() from None
        if (
            getattr(first_parameter, "dtype", None) != dtype
            or str(getattr(first_parameter, "device", None)) != self.config.device
        ):
            raise KyutaiAdapterError()
        self._device = device
        self._dtype = dtype
        self._mimi = mimi
        self._tokenizer = tokenizer
        self._lm = lm
        self._resampler = resampler

    def _validate_checkpoint(self, checkpoint: object) -> None:
        raw = getattr(checkpoint, "raw_config", None)
        stt = getattr(checkpoint, "stt_config", None)
        if not isinstance(raw, Mapping) or not isinstance(stt, Mapping):
            raise KyutaiAdapterError()
        lm_gen = raw.get("lm_gen_config")
        if (
            raw.get("extra_heads_num_heads") != self.config.semantic_head_count
            or raw.get("extra_heads_dim") != self.config.semantic_head_dim
            or raw.get("existing_text_padding_id") != self.config.text_padding_token_id
            or raw.get("model_type") != "stt"
            or raw.get("n_q") != 32
            or raw.get("dep_q") != 0
            or raw.get("delays") != [0] * 33
            or not isinstance(lm_gen, Mapping)
            or type(lm_gen.get("temp")) not in {int, float}
            or float(lm_gen.get("temp")) != self.config.temperature
            or type(lm_gen.get("temp_text")) not in {int, float}
            or float(lm_gen.get("temp_text")) != self.config.text_temperature
            or stt.get("audio_delay_seconds") != self.config.audio_delay_seconds
            or stt.get("audio_silence_prefix_seconds")
            != self.config.audio_silence_prefix_seconds
        ):
            raise KyutaiAdapterError()

    def __enter__(self) -> KyutaiAdapter:
        if self._closed:
            raise KyutaiAdapterError()
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        api = self._api
        device = self._device
        self._lm = None
        self._resampler = None
        self._tokenizer = None
        self._mimi = None
        self._dtype = None
        self._device = None
        self._api = None
        if api is not None:
            try:
                cuda = getattr(getattr(api, "torch"), "cuda")
                if device is not None:
                    cuda.synchronize(device)
                cuda.empty_cache()
            except Exception:
                self._poisoned = True

    def _now(self) -> float:
        try:
            value = float(self._clock())
        except (TypeError, ValueError, OverflowError):
            raise KyutaiAdapterError() from None
        if not math.isfinite(value):
            raise KyutaiAdapterError()
        return value

    @staticmethod
    def _duration_ms(start: float, end: float) -> float:
        duration = end - start
        if not math.isfinite(duration) or duration < 0.0:
            raise KyutaiAdapterError()
        return duration * 1000.0

    def _cuda_synchronize(self) -> None:
        if self._api is None or self._device is None:
            return
        try:
            self._api.torch.cuda.synchronize(self._device)
        except Exception:
            raise KyutaiAdapterError() from None

    def _timed(self, function: Callable[[], _T]) -> tuple[_T, float]:
        self._cuda_synchronize()
        started = self._now()
        value = function()
        self._cuda_synchronize()
        finished = self._now()
        return value, self._duration_ms(started, finished)

    def _resources(self) -> ResourceUsage:
        if self._api is None or self._device is None:
            raise KyutaiAdapterError()
        try:
            value = self._resource_reader(self._api.torch, self._device)
        except Exception:
            raise KyutaiAdapterError() from None
        if not isinstance(value, ResourceUsage):
            raise KyutaiAdapterError()
        return value

    def _validate_heads(self, heads: object) -> None:
        if (
            not isinstance(heads, Sequence)
            or len(heads) != self.config.semantic_head_count
        ):
            raise KyutaiAdapterError()
        assert self._api is not None
        for head in heads:
            if _shape(head) != (1, 1, self.config.semantic_head_dim):
                raise KyutaiAdapterError()
            try:
                finite = self._api.torch.isfinite(head).all()
            except Exception:
                raise KyutaiAdapterError() from None
            if _scalar_item(finite) is not True:
                raise KyutaiAdapterError()

    def _token_id(self, tokens: object) -> int:
        if _shape(tokens) != (1, 1, 1) or self._tokenizer is None:
            raise KyutaiAdapterError()
        try:
            token_id = _scalar_item(tokens[0, 0, 0])
        except Exception:
            raise KyutaiAdapterError() from None
        if type(token_id) is not int or not 0 <= token_id < 8_000:
            raise KyutaiAdapterError()
        return token_id

    def _piece(self, tokens: object) -> str:
        token_id = self._token_id(tokens)
        if token_id in {
            self.config.end_of_padding_token_id,
            self.config.text_padding_token_id,
        }:
            return ""
        try:
            piece = self._tokenizer.id_to_piece(token_id)
        except Exception:
            raise KyutaiAdapterError() from None
        if not isinstance(piece, str):
            raise KyutaiAdapterError()
        return piece.replace("▁", " ")

    def transcribe(
        self,
        request: TranscribeRequest,
        *,
        emit_partial: Callable[[int, KyutaiPartial], None],
    ) -> KyutaiCase:
        if self._closed or self._poisoned:
            raise KyutaiAdapterError()
        if not self._case_lock.acquire(blocking=False):
            raise KyutaiAdapterError()
        try:
            return self._transcribe_locked(request, emit_partial=emit_partial)
        except Exception as error:
            self._poisoned = True
            if isinstance(error, KyutaiAdapterError):
                raise
            raise KyutaiAdapterError() from None
        finally:
            self._case_lock.release()

    def _transcribe_locked(
        self,
        request: TranscribeRequest,
        *,
        emit_partial: Callable[[int, KyutaiPartial], None],
    ) -> KyutaiCase:
        if (
            self._api is None
            or self._device is None
            or self._mimi is None
            or self._lm is None
            or self._resampler is None
            or request.protocol_version != PROTOCOL_VERSION
            or request.pcm.sample_rate != self.config.input_sample_rate_hz
            or request.pcm.channels != 1
            or request.pcm.encoding != "f32le"
            or request.pcm.samples > self.config.maximum_source_samples
            or request.stream.chunk_samples != self.config.input_chunk_samples
            or request.stream.partial_interval_ms != self.config.partial_interval_ms
            or request.stream.tail_padding_samples != self.config.terminal_tail_samples
        ):
            raise KyutaiAdapterError()
        source = _read_pcm(request)
        declared_source = request.pcm.samples
        declared_tail = request.stream.tail_padding_samples
        declared_total = declared_source + declared_tail
        if declared_total <= self.config.input_chunk_samples:
            raise KyutaiAdapterError()

        lm_gen_type = getattr(self._api.moshi_models, "LMGen", None)
        if not callable(lm_gen_type):
            raise KyutaiAdapterError()
        try:
            lm_gen = lm_gen_type(
                self._lm,
                temp=self.config.temperature,
                temp_text=self.config.text_temperature,
                use_sampling=self.config.use_sampling,
            )
        except Exception:
            raise KyutaiAdapterError() from None
        if (
            not callable(getattr(lm_gen, "streaming", None))
            or not callable(getattr(lm_gen, "step_with_extra_heads", None))
            or (
                self.config.initial_frame_policy != "duplicate-first-frame-prime"
                or self.config.initial_frame_prime_steps != 1
            )
        ):
            raise KyutaiAdapterError()

        case_started = self._now()
        partials: list[KyutaiPartial] = []
        text = ""
        compute_ms = 0.0
        finalization_ms = 0.0
        deadline_misses = 0
        max_backlog_ms = 0.0
        chunks = 0
        source_seen = 0
        tail_seen = 0
        semantic_head_steps = 0
        model_padding_samples = (-declared_total) % self.config.input_chunk_samples
        aligned_total = declared_total + model_padding_samples
        expected_chunks = aligned_total // self.config.input_chunk_samples
        expected_resampled_samples = expected_chunks * self.config.mimi_frame_samples
        next_partial_samples = (
            self.config.partial_interval_ms * self.config.input_sample_rate_hz + 999
        ) // 1000

        try:

            def prepare_audio() -> object:
                torch = self._api.torch
                samples = list(source)
                samples.extend([0.0] * (declared_tail + model_padding_samples))
                if len(samples) != aligned_total:
                    raise KyutaiAdapterError()
                audio = torch.tensor(
                    samples,
                    dtype=torch.float32,
                    device=self._device,
                ).view(1, -1)
                value = self._resampler(audio)
                if _shape(value) != (1, expected_resampled_samples):
                    raise KyutaiAdapterError()
                return value

            with self._api.torch.inference_mode():
                resampled, resample_ms = self._timed(prepare_audio)
            compute_ms += resample_ms
            stream_started = case_started
            with ExitStack() as stack:
                stack.enter_context(self._mimi.streaming(self.config.batch_size))
                stack.enter_context(lm_gen.streaming(self.config.batch_size))
                for chunk_index in range(expected_chunks):
                    offset = chunk_index * self.config.input_chunk_samples
                    if request.stream.pace == "realtime":
                        target = (
                            stream_started + offset / self.config.input_sample_rate_hz
                        )
                        before_wait = self._now()
                        if before_wait < target:
                            self._sleeper(target - before_wait)
                        if self._now() < target:
                            raise KyutaiAdapterError()
                    actual_count = min(
                        self.config.input_chunk_samples,
                        declared_total - offset,
                    )
                    source_start = min(offset, declared_source)
                    source_end = min(offset + actual_count, declared_source)
                    source_count = max(0, source_end - source_start)
                    tail_count = actual_count - source_count
                    output_start = chunk_index * self.config.mimi_frame_samples
                    output_end = output_start + self.config.mimi_frame_samples
                    frame = resampled[:, output_start:output_end]
                    if _shape(frame) != (1, self.config.mimi_frame_samples):
                        raise KyutaiAdapterError()

                    def step_frame() -> tuple[object | None, object]:
                        audio_tokens = self._mimi.encode(
                            frame.view(
                                self.config.batch_size,
                                1,
                                self.config.mimi_frame_samples,
                            )
                        )
                        priming = (
                            lm_gen.step_with_extra_heads(audio_tokens)
                            if chunk_index == 0
                            else None
                        )
                        result = lm_gen.step_with_extra_heads(audio_tokens)
                        if (
                            (
                                priming is not None
                                and (
                                    not isinstance(priming, tuple) or len(priming) != 2
                                )
                            )
                            or not isinstance(result, tuple)
                            or len(result) != 2
                        ):
                            raise KyutaiAdapterError()
                        return priming, result

                    with self._api.torch.inference_mode():
                        (priming, (tokens, heads)), decode_ms = self._timed(step_frame)
                    if priming is not None:
                        priming_tokens, priming_heads = priming
                        self._validate_heads(priming_heads)
                        self._token_id(priming_tokens)
                        semantic_head_steps += 1
                    self._validate_heads(heads)
                    semantic_head_steps += 1
                    piece = self._piece(tokens)
                    if len(text) + len(piece) > MAX_HYPOTHESIS_CHARS:
                        raise KyutaiAdapterError()
                    text += piece
                    compute_ms += decode_ms
                    finalization_ms = decode_ms
                    chunks += 1
                    source_seen += source_count
                    tail_seen += tail_count
                    samples_seen = source_seen + tail_seen

                    if request.stream.pace == "realtime":
                        deadline = (
                            stream_started
                            + samples_seen / self.config.input_sample_rate_hz
                        )
                        now = self._now()
                        backlog_ms = self._duration_ms(deadline, max(deadline, now))
                        max_backlog_ms = max(max_backlog_ms, backlog_ms)
                        if backlog_ms > 0.0:
                            deadline_misses += 1

                    visible = text.strip()
                    if samples_seen >= next_partial_samples:
                        while next_partial_samples <= samples_seen:
                            next_partial_samples += (
                                self.config.partial_interval_ms
                                * self.config.input_sample_rate_hz
                                + 999
                            ) // 1000
                        if visible and (not partials or partials[-1].text != visible):
                            if len(partials) >= MAX_PARTIALS_PER_CASE:
                                raise KyutaiAdapterError()
                            partial = KyutaiPartial(
                                after_samples=samples_seen,
                                text=visible,
                                elapsed_ms=self._duration_ms(
                                    case_started,
                                    self._now(),
                                ),
                                decode_ms=decode_ms,
                            )
                            emit_partial(len(partials), partial)
                            partials.append(partial)
        except KyutaiAdapterError:
            raise
        except Exception:
            raise KyutaiAdapterError() from None

        if (
            source_seen != declared_source
            or tail_seen != declared_tail
            or semantic_head_steps != chunks + self.config.initial_frame_prime_steps
            or chunks != expected_chunks
        ):
            raise KyutaiAdapterError()
        finished = self._now()
        return KyutaiCase(
            partials=tuple(partials),
            final=text.strip(),
            elapsed_ms=self._duration_ms(case_started, finished),
            finalization_ms=finalization_ms,
            compute_ms=compute_ms,
            deadline_misses=deadline_misses,
            max_backlog_ms=max_backlog_ms,
            model_padding_samples=model_padding_samples,
            chunks_yielded=chunks,
            resources=self._resources(),
            source_samples=declared_source,
            source_samples_consumed=source_seen,
            declared_tail_samples=declared_tail,
            tail_samples_consumed=tail_seen,
            semantic_head_steps=semantic_head_steps,
        )


__all__ = [
    "KyutaiAdapter",
    "KyutaiAdapterError",
    "KyutaiCase",
    "KyutaiPartial",
    "KyutaiReady",
    "KyutaiStreamGeometry",
]
