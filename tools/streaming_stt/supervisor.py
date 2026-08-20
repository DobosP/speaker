"""Bounded process supervision for one isolated streaming recognizer."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import queue
import re
import signal
import stat
import subprocess
import threading
import time
from typing import Mapping, Sequence

from .bounded_io import (
    BoundedReadError,
    hash_regular_bounded,
    opened_directory_nofollow,
)
from .cgroup_observation import (
    CgroupObservationError,
    CgroupResourceSnapshot,
    CgroupScopeObserver,
)
from .manifest import (
    FASTER_WHISPER_ENDPOINT_ADAPTER,
    FASTER_WHISPER_REQUIRED_MODEL_FILES,
    FASTER_WHISPER_VOCABULARY_FILES,
    KYUTAI_ADAPTER,
    MAX_PYTHON_BYTES,
    MAX_MOBILE_WHISPER_SOURCE_LOCK_BYTES,
    MAX_MOBILE_ZIPFORMER_SOURCE_LOCK_BYTES,
    MAX_WORKER_BYTES,
    MOBILE_WHISPER_ADAPTER,
    MOBILE_ZIPFORMER_ADAPTER,
    MOONSHINE_ADAPTER,
    MOONSHINE_EXTERNAL_ENDPOINT_ADAPTER,
    NEMOTRON_ADAPTER,
    PARAKEET_CPP_ADAPTER,
    PARAKEET_REALTIME_EOU_ADAPTER,
    SHERPA_ZIPFORMER_ADAPTER,
    FasterWhisperEndpointConfig,
    KyutaiConfig,
    MobileWhisperConfig,
    MobileZipformerConfig,
    MoonshineExternalEndpointConfig,
    NemotronConfig,
    ParakeetCppConfig,
    ParakeetRealtimeEouConfig,
    WorkerManifest,
    artifact_maximum_bytes,
)
from .protocol import (
    ByeEvent,
    CaseTrace,
    ErrorEvent,
    FinalEvent,
    MAX_HYPOTHESIS_CHARS,
    MAX_LINE_BYTES,
    MAX_MOBILE_ZIPFORMER_ENDPOINT_OBSERVATIONS,
    MAX_PARAKEET_CPP_OBSERVED_EVENTS,
    MAX_PARTIALS_PER_CASE,
    MAX_PCM_BYTES,
    MAX_STDERR_BYTES,
    MAX_STDOUT_BYTES,
    MOBILE_ZIPFORMER_ENDPOINT_PROTOCOL_VERSION,
    NATIVE_ENDPOINT_PROTOCOL_VERSION,
    NativeFinalEvent,
    PARAKEET_CPP_ENDPOINT_PROTOCOL_VERSION,
    ParakeetCppFinalEvent,
    ParakeetCppObservedEvent,
    MobileZipformerEndpointObservation,
    MobileZipformerFinalEvent,
    PROTOCOL_VERSION,
    PartialEvent,
    ProtocolError,
    ReadyEvent,
    Response,
    ShutdownEvent,
    ShutdownRequest,
    TranscribeRequest,
    encode_message,
    parse_request,
    parse_response,
)
from .source_bundle import (
    SourceBundle,
    SourceBundleError,
    verify_source_bundle,
)
from .runtime_receipt import (
    FASTER_WHISPER_MODEL_TREE_LIMITS,
    FASTER_WHISPER_RUNTIME_TREE_LIMITS,
    KYUTAI_RUNTIME_TREE_LIMITS,
    NEMOTRON_RUNTIME_TREE_LIMITS,
    PARAKEET_RUNTIME_TREE_LIMITS,
    RuntimeTreeReceipt,
    load_runtime_tree_receipt,
    verify_moonshine_wheel_install,
    verify_venv_runtime_location,
)


_TERM_GRACE_SEC = 5.0
_KILL_GRACE_SEC = 2.0
_QUEUE_CAPACITY = MAX_PARTIALS_PER_CASE + 16
_BWRAP_PATH = Path("/usr/bin/bwrap")
_SYSTEMD_RUN_PATH = Path("/usr/bin/systemd-run")
_CANONICAL_REPO_ROOT = Path(__file__).resolve().parents[2]
_USER_RUNTIME_ROOT = Path("/run/user")
_CGROUP_ROOT = Path("/sys/fs/cgroup")
_PROC_ROOT = Path("/proc")
_CGROUP_SETUP_TIMEOUT_SEC = 5.0
_RESOURCE_PHASES = (
    ("startup", "validated_ready"),
    ("first_use", "first_validated_terminal"),
    ("resident", "pre_shutdown_after_suite"),
)
_PARAKEET_MEMORY_HIGH_BYTES = 6 * 1024**3
_PARAKEET_MEMORY_MAX_BYTES = 8 * 1024**3
_PARAKEET_CPU_QUOTA_PERCENT = 200
_PARAKEET_TASKS_MAX = 128
_PARAKEET_SCOPE_PROPERTIES = (
    f"MemoryHigh={_PARAKEET_MEMORY_HIGH_BYTES}",
    f"MemoryMax={_PARAKEET_MEMORY_MAX_BYTES}",
    "MemorySwapMax=0",
    f"CPUQuota={_PARAKEET_CPU_QUOTA_PERCENT}%",
    f"TasksMax={_PARAKEET_TASKS_MAX}",
    "OOMPolicy=kill",
)
_PARAKEET_CPP_MEMORY_HIGH_BYTES = 2 * 1024**3
_PARAKEET_CPP_MEMORY_MAX_BYTES = 3 * 1024**3
_PARAKEET_CPP_CPU_QUOTA_PERCENT = 100
_PARAKEET_CPP_TASKS_MAX = 64
_PARAKEET_CPP_SCOPE_PROPERTIES = (
    f"MemoryHigh={_PARAKEET_CPP_MEMORY_HIGH_BYTES}",
    f"MemoryMax={_PARAKEET_CPP_MEMORY_MAX_BYTES}",
    "MemorySwapMax=0",
    f"CPUQuota={_PARAKEET_CPP_CPU_QUOTA_PERCENT}%",
    f"TasksMax={_PARAKEET_CPP_TASKS_MAX}",
    "OOMPolicy=kill",
)
_KYUTAI_MEMORY_HIGH_BYTES = 5 * 1024**3
_KYUTAI_MEMORY_MAX_BYTES = 6 * 1024**3
_KYUTAI_CPU_QUOTA_PERCENT = 100
_KYUTAI_TASKS_MAX = 128
_KYUTAI_IO_WEIGHT = 1
# Keep one full worker ceiling available to unrelated host workloads even if
# the candidate reaches its cgroup limit.
_KYUTAI_MINIMUM_HOST_AVAILABLE_BYTES = 12 * 1024**3
_KYUTAI_SCOPE_PROPERTIES = (
    f"MemoryHigh={_KYUTAI_MEMORY_HIGH_BYTES}",
    f"MemoryMax={_KYUTAI_MEMORY_MAX_BYTES}",
    "MemorySwapMax=0",
    f"CPUQuota={_KYUTAI_CPU_QUOTA_PERCENT}%",
    f"TasksMax={_KYUTAI_TASKS_MAX}",
    f"IOWeight={_KYUTAI_IO_WEIGHT}",
    "IOSchedulingClass=idle",
    "OOMPolicy=kill",
)
_NVIDIA_REQUIRED_DEVICE_NAMES = ("nvidia0", "nvidiactl", "nvidia-uvm")
_NVIDIA_OPTIONAL_DEVICE_NAMES = ("nvidia-modeset", "nvidia-uvm-tools")
_NVIDIA_CAP_RE = re.compile(r"nvidia-cap[0-9]{1,3}\Z")
_NVIDIA_PCI_RE = re.compile(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]\Z")
_NEMOTRON_CORE_RO_PATHS = (
    Path("/usr"),
    Path("/lib"),
    Path("/lib64"),
    Path("/etc/ld.so.cache"),
    # Numba asks libc for the current account while constructing its cache
    # locator. Keep the namespace closed, but provide the two read-only local
    # account databases needed to resolve the already-running uid/gid.
    Path("/etc/passwd"),
    Path("/etc/group"),
)
_NVIDIA_DRIVER_ROOT = Path("/sys/bus/pci/drivers/nvidia")
_NVIDIA_MODULE_PATHS = (
    Path("/sys/module/nvidia"),
    Path("/sys/module/nvidia_uvm"),
    Path("/sys/module/nvidia_modeset"),
    Path("/sys/module/nvidia_drm"),
)
_NVIDIA_PROC_DRIVER = Path("/proc/driver/nvidia")


class WorkerError(RuntimeError):
    """A detail-free worker lifecycle or protocol failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class _ResourceBoundary:
    phase: str
    boundary: str
    completed_cases: int
    wall_elapsed_ms: float
    snapshot: CgroupResourceSnapshot

    def as_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "boundary": self.boundary,
            "completed_cases": self.completed_cases,
            "wall_elapsed_ms": round(self.wall_elapsed_ms, 3),
            **self.snapshot.as_dict(),
        }


def _external_moonshine_config(
    manifest: WorkerManifest,
) -> MoonshineExternalEndpointConfig:
    config = manifest.adapter_config
    if (
        type(config) is not MoonshineExternalEndpointConfig
        or config.segmentation_mode != "external-presegmented"
        or config.endpoint_owner != "external-input-boundary"
        or type(config.vad_threshold) is not float
        or config.vad_threshold != 0.0
        or math.copysign(1.0, config.vad_threshold) != 1.0
        or type(config.vad_max_segment_duration_sec) is not float
        or config.vad_max_segment_duration_sec != 136.0
        or type(config.vad_hop_size_samples) is not int
        or config.vad_hop_size_samples != 512
        or type(config.streaming_chunk_samples) is not int
        or config.streaming_chunk_samples != 1_280
        or type(config.online_partial_interval_ms) is not int
        or config.online_partial_interval_ms != 500
        or type(config.authoritative_alignment_samples) is not int
        or config.authoritative_alignment_samples
        != math.lcm(
            config.vad_hop_size_samples,
            config.streaming_chunk_samples,
        )
        or config.tail_alignment_policy != "zero-pad-to-vad-model-lcm"
        or config.finalization_policy != "verified-native-free-authoritative-batch-v2"
        or type(config.maximum_source_samples) is not int
        or config.maximum_source_samples != 2_097_152
    ):
        raise WorkerError("worker_protocol")
    return config


def sanitized_worker_environment(scratch_root: Path) -> dict[str, str]:
    """Return a minimal environment with numerical fan-out disabled."""

    environment = {
        "HOME": str(scratch_root),
        "TMPDIR": str(scratch_root),
        "XDG_CACHE_HOME": str(scratch_root / "xdg-cache"),
        "HF_HOME": str(scratch_root / "hf-home"),
        "HF_HUB_CACHE": str(scratch_root / "hf-hub-cache"),
        "HF_ASSETS_CACHE": str(scratch_root / "hf-assets-cache"),
        "HF_MODULES_CACHE": str(scratch_root / "hf-modules-cache"),
        "TRANSFORMERS_CACHE": str(scratch_root / "transformers-cache"),
        "TORCH_HOME": str(scratch_root / "torch-home"),
        "TRITON_CACHE_DIR": str(scratch_root / "triton-cache"),
        "NUMBA_CACHE_DIR": str(scratch_root / "numba-cache"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "MOONSHINE_ORT_SINGLE_THREAD": "1",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    if os.name == "nt":
        for name in ("SYSTEMROOT", "WINDIR"):
            value = os.environ.get(name)
            if value:
                environment[name] = value
    return environment


def _expected_final_evidence(
    manifest: WorkerManifest,
    request: TranscribeRequest,
) -> tuple[int, int | None]:
    """Derive independently checkable final chunk and padding evidence."""

    total_samples = request.pcm.samples + request.stream.tail_padding_samples
    if manifest.adapter == MOONSHINE_EXTERNAL_ENDPOINT_ADAPTER:
        config = _external_moonshine_config(manifest)
        if (
            request.stream.tail_padding_samples != 0
            or request.stream.chunk_samples != config.streaming_chunk_samples
            or request.stream.partial_interval_ms != config.online_partial_interval_ms
            or request.pcm.samples > config.maximum_source_samples
        ):
            raise WorkerError("invalid_request")
        chunks = (
            total_samples + config.streaming_chunk_samples - 1
        ) // config.streaming_chunk_samples
        return (
            chunks,
            (-request.pcm.samples) % config.authoritative_alignment_samples,
        )
    if manifest.adapter == KYUTAI_ADAPTER:
        config = manifest.adapter_config
        if (
            not isinstance(config, KyutaiConfig)
            or manifest.schema_version != 9
            or request.stream.chunk_samples != config.input_chunk_samples
            or request.stream.partial_interval_ms != config.partial_interval_ms
            or request.stream.tail_padding_samples != config.terminal_tail_samples
            or request.pcm.samples > config.maximum_source_samples
        ):
            raise WorkerError("invalid_request")
        chunks = (
            total_samples + config.input_chunk_samples - 1
        ) // config.input_chunk_samples
        return chunks, (-total_samples) % config.input_chunk_samples
    if manifest.adapter == MOBILE_WHISPER_ADAPTER:
        config = manifest.adapter_config
        if (
            not isinstance(config, MobileWhisperConfig)
            or manifest.schema_version != 11
            or request.stream.chunk_samples != config.native_chunk_samples
            or request.stream.tail_padding_samples
            != config.maximum_tail_padding_samples
            or request.stream.pace != "burst"
        ):
            raise WorkerError("invalid_request")
        chunks = (
            total_samples + config.native_chunk_samples - 1
        ) // config.native_chunk_samples
        # This is only adapter/evaluator-appended PCM alignment padding. Sherpa's
        # internal tail_paddings=-1 behavior is intentionally opaque here.
        return chunks, 0
    if manifest.adapter != NEMOTRON_ADAPTER:
        chunks = (
            total_samples + request.stream.chunk_samples - 1
        ) // request.stream.chunk_samples
        return chunks, None

    config = manifest.adapter_config
    if not isinstance(config, NemotronConfig):
        raise WorkerError("worker_protocol")
    geometry = (
        config.native_first_chunk_frames,
        config.native_hop_length_samples,
        config.native_n_fft_samples,
        config.native_first_window_samples,
        config.native_window_samples,
        config.native_stride_samples,
    )
    if any(type(value) is not int or value <= 0 for value in geometry):
        raise WorkerError("worker_protocol")
    (
        first_frames,
        hop_length,
        n_fft,
        first_window,
        window,
        stride,
    ) = geometry
    if total_samples <= first_window:
        final_window_end = first_window
        chunks = 1
    else:
        final_window_end = first_frames * hop_length - n_fft // 2 + window
        if final_window_end <= first_window:
            raise WorkerError("worker_protocol")
        additional = max(
            0,
            (total_samples - final_window_end + stride - 1) // stride,
        )
        final_window_end += additional * stride
        chunks = 2 + additional
    model_padding_samples = final_window_end - total_samples
    if model_padding_samples < 0:
        raise WorkerError("worker_protocol")
    return chunks, model_padding_samples


def _wire_protocol_version(manifest: WorkerManifest) -> int:
    if manifest.adapter == PARAKEET_REALTIME_EOU_ADAPTER:
        return NATIVE_ENDPOINT_PROTOCOL_VERSION
    if manifest.adapter == PARAKEET_CPP_ADAPTER:
        return PARAKEET_CPP_ENDPOINT_PROTOCOL_VERSION
    if manifest.adapter == MOBILE_ZIPFORMER_ADAPTER:
        return MOBILE_ZIPFORMER_ENDPOINT_PROTOCOL_VERSION
    return PROTOCOL_VERSION


def _worker_interpreter_flags(manifest: WorkerManifest) -> tuple[str, ...]:
    """Keep schema-11 startup hooks disabled without changing older adapters."""

    if manifest.adapter == MOBILE_WHISPER_ADAPTER:
        return ("-I", "-S", "-B")
    if manifest.adapter in {MOBILE_ZIPFORMER_ADAPTER, SHERPA_ZIPFORMER_ADAPTER}:
        return ("-I", "-B")
    return ("-I", "-S", "-B")


def _validate_native_final(
    manifest: WorkerManifest,
    request: TranscribeRequest,
    event: NativeFinalEvent,
    *,
    expected_seq: int,
    last_samples: int,
    last_elapsed: float,
) -> None:
    config = manifest.adapter_config
    if not isinstance(config, ParakeetRealtimeEouConfig):
        raise WorkerError("worker_protocol")
    if (
        request.protocol_version != NATIVE_ENDPOINT_PROTOCOL_VERSION
        or request.stream.chunk_samples != config.native_chunk_samples
        or request.stream.tail_padding_samples > config.maximum_tail_padding_samples
    ):
        raise WorkerError("worker_protocol")
    samples_seen = event.source_samples_consumed + event.tail_samples_consumed
    expected_chunks = (
        samples_seen + config.native_chunk_samples - 1
    ) // config.native_chunk_samples
    expected_padding = expected_chunks * config.native_chunk_samples - samples_seen
    expected_endpoint_sample = samples_seen + expected_padding
    source_complete = event.source_samples_consumed == request.pcm.samples
    expected_authoritative = (
        event.native_endpoint and event.endpoint_reason == "eou" and source_complete
    )
    expected_latency = (
        (event.tail_samples_consumed + expected_padding)
        * 1000.0
        / request.pcm.sample_rate
        if expected_authoritative
        else None
    )
    common_invalid = (
        event.protocol_version != NATIVE_ENDPOINT_PROTOCOL_VERSION
        or event.request_id != request.request_id
        or event.seq != expected_seq
        or event.source_samples != request.pcm.samples
        or event.declared_tail_samples != request.stream.tail_padding_samples
        or event.source_samples_consumed > request.pcm.samples
        or event.tail_samples_consumed > request.stream.tail_padding_samples
        or (
            event.source_samples_consumed < request.pcm.samples
            and event.tail_samples_consumed != 0
        )
        or event.samples_seen != samples_seen
        or event.samples_seen < last_samples
        or event.elapsed_ms < last_elapsed
        or event.finalization_ms > event.elapsed_ms
        or event.chunks != expected_chunks
        or event.model_padding_samples != expected_padding
        or abs(event.audio_seconds - request.pcm.samples / request.pcm.sample_rate)
        > 1e-6
    )
    if common_invalid:
        raise WorkerError("worker_protocol")
    if event.native_endpoint:
        if (
            event.endpoint_reason not in {"eou", "eob"}
            or isinstance(event.endpoint_probability, bool)
            or not isinstance(event.endpoint_probability, (int, float))
            or not math.isfinite(float(event.endpoint_probability))
            or not 0.0 <= float(event.endpoint_probability) <= 1.0
            or event.endpoint_sample != expected_endpoint_sample
            or event.authoritative != expected_authoritative
            or (expected_latency is None and event.endpoint_latency_ms is not None)
            or (
                expected_latency is not None
                and (
                    isinstance(event.endpoint_latency_ms, bool)
                    or not isinstance(event.endpoint_latency_ms, (int, float))
                    or not math.isfinite(float(event.endpoint_latency_ms))
                    or abs(float(event.endpoint_latency_ms) - expected_latency) > 1e-6
                )
            )
        ):
            raise WorkerError("worker_protocol")
        return
    if (
        event.endpoint_reason != "tail_exhausted"
        or event.endpoint_probability is not None
        or event.endpoint_sample is not None
        or event.endpoint_latency_ms is not None
        or event.authoritative
        or not source_complete
        or event.tail_samples_consumed != request.stream.tail_padding_samples
    ):
        raise WorkerError("worker_protocol")


def _validate_parakeet_cpp_final(
    manifest: WorkerManifest,
    request: TranscribeRequest,
    event: ParakeetCppFinalEvent,
    *,
    expected_seq: int,
    last_samples: int,
    last_elapsed: float,
) -> None:
    """Independently validate typed parakeet.cpp endpoint evidence."""

    config = manifest.adapter_config
    if not isinstance(config, ParakeetCppConfig):
        raise WorkerError("worker_protocol")
    if (
        request.protocol_version != PARAKEET_CPP_ENDPOINT_PROTOCOL_VERSION
        or request.stream.chunk_samples != config.native_chunk_samples
        or request.stream.tail_padding_samples > config.maximum_tail_padding_samples
    ):
        raise WorkerError("worker_protocol")
    samples_seen = event.source_samples_consumed + event.tail_samples_consumed
    expected_chunks = (
        samples_seen + config.native_chunk_samples - 1
    ) // config.native_chunk_samples
    source_complete = event.source_samples_consumed == event.source_samples_offered
    if (
        event.protocol_version != PARAKEET_CPP_ENDPOINT_PROTOCOL_VERSION
        or event.request_id != request.request_id
        or event.seq != expected_seq
        or event.source_samples_offered != request.pcm.samples
        or event.tail_samples_offered != request.stream.tail_padding_samples
        or event.source_samples_consumed > request.pcm.samples
        or event.tail_samples_consumed > request.stream.tail_padding_samples
        or (
            event.source_samples_consumed < request.pcm.samples
            and event.tail_samples_consumed != 0
        )
        or event.samples_seen != samples_seen
        or event.samples_seen < last_samples
        or event.elapsed_ms < last_elapsed
        or event.finalization_ms > event.elapsed_ms
        or event.chunks != expected_chunks
        or event.model_padding_samples != 0
        or abs(event.audio_seconds - request.pcm.samples / request.pcm.sample_rate)
        > 1e-6
        or not isinstance(event.observed_events, tuple)
        or len(event.observed_events) > MAX_PARAKEET_CPP_OBSERVED_EVENTS
    ):
        raise WorkerError("worker_protocol")

    seen_observations: set[tuple[str, int]] = set()
    last_frame = -1
    last_time = -1.0
    last_observed_sample = -1
    finalize_seen = False
    first_eou_event: ParakeetCppObservedEvent | None = None
    for observation in event.observed_events:
        if type(observation) is not ParakeetCppObservedEvent:
            raise WorkerError("worker_protocol")
        observation_time = observation.encoder_time_seconds
        observation_key = (
            observation.event_type,
            observation.encoder_frame,
        )
        if (
            observation.event_type not in {"eou", "eob"}
            or observation.event_origin not in {"feed", "finalize"}
            or type(observation.observed_sample) is not int
            or not 0 < observation.observed_sample <= samples_seen
            or type(observation.encoder_frame) is not int
            or observation.encoder_frame < 0
            or isinstance(observation_time, bool)
            or not isinstance(observation_time, (int, float))
            or not math.isfinite(float(observation_time))
            or not math.isclose(
                float(observation_time),
                observation.encoder_frame * config.frame_sec,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            or observation.encoder_frame < last_frame
            or float(observation_time) + 1e-9 < last_time
            or observation.observed_sample < last_observed_sample
            or observation_key in seen_observations
            or (
                observation.event_origin == "feed"
                and (
                    finalize_seen
                    or (
                        observation.observed_sample
                        != event.source_samples_offered + event.tail_samples_offered
                        and observation.observed_sample % config.native_chunk_samples
                        != 0
                    )
                    or float(observation_time)
                    > observation.observed_sample / request.pcm.sample_rate + 1e-9
                )
            )
            or (
                observation.event_origin == "finalize"
                and observation.observed_sample != samples_seen
            )
        ):
            raise WorkerError("worker_protocol")
        if observation.event_origin == "finalize":
            finalize_seen = True
        if first_eou_event is None and observation.event_type == "eou":
            first_eou_event = observation
        seen_observations.add(observation_key)
        last_frame = observation.encoder_frame
        last_time = float(observation_time)
        last_observed_sample = observation.observed_sample

    accepted_event = (
        first_eou_event
        if first_eou_event is not None
        and first_eou_event.event_origin == "feed"
        and first_eou_event.observed_sample >= event.source_samples_offered
        else None
    )
    if accepted_event is not None:
        expected_latency = (
            event.tail_samples_consumed * 1000.0 / request.pcm.sample_rate
        )
        if (
            not source_complete
            or event.samples_seen != accepted_event.observed_sample
            or event.endpoint_reason != "eou"
            or event.native_endpoint is not True
            or event.encoder_frame != accepted_event.encoder_frame
            or isinstance(event.encoder_time_seconds, bool)
            or not isinstance(event.encoder_time_seconds, (int, float))
            or not math.isclose(
                float(event.encoder_time_seconds),
                accepted_event.encoder_time_seconds,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            or event.event_origin != "feed"
            or event.endpoint_sample != accepted_event.observed_sample
            or event.authoritative is not True
            or isinstance(event.endpoint_latency_ms, bool)
            or not isinstance(event.endpoint_latency_ms, (int, float))
            or not math.isfinite(float(event.endpoint_latency_ms))
            or not math.isclose(
                float(event.endpoint_latency_ms),
                expected_latency,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            raise WorkerError("worker_protocol")
        return
    if (
        event.native_endpoint is not False
        or event.endpoint_reason != "tail_exhausted"
        or event.event_origin != "none"
        or event.encoder_frame is not None
        or event.encoder_time_seconds is not None
        or event.endpoint_sample is not None
        or event.endpoint_latency_ms is not None
        or event.authoritative is not False
        or not source_complete
        or event.tail_samples_consumed != event.tail_samples_offered
    ):
        raise WorkerError("worker_protocol")


def _validate_mobile_zipformer_final(
    manifest: WorkerManifest,
    request: TranscribeRequest,
    event: MobileZipformerFinalEvent,
    *,
    expected_seq: int,
    last_samples: int,
    last_elapsed: float,
) -> None:
    """Independently validate reset-committed mobile endpoint epochs."""

    config = manifest.adapter_config
    if (
        not isinstance(config, MobileZipformerConfig)
        or request.protocol_version != MOBILE_ZIPFORMER_ENDPOINT_PROTOCOL_VERSION
        or request.stream.chunk_samples != config.native_chunk_samples
        or request.stream.tail_padding_samples > config.maximum_tail_padding_samples
    ):
        raise WorkerError("worker_protocol")
    expected_chunks = (
        request.pcm.samples + config.native_chunk_samples - 1
    ) // config.native_chunk_samples + (
        event.tail_samples_consumed + config.native_chunk_samples - 1
    ) // config.native_chunk_samples
    if (
        event.protocol_version != MOBILE_ZIPFORMER_ENDPOINT_PROTOCOL_VERSION
        or event.request_id != request.request_id
        or event.seq != expected_seq
        or event.source_samples != request.pcm.samples
        or event.source_samples_consumed != request.pcm.samples
        or event.declared_tail_samples != request.stream.tail_padding_samples
        or event.tail_samples_consumed > request.stream.tail_padding_samples
        or event.samples_seen != request.pcm.samples + event.tail_samples_consumed
        or event.samples_seen < last_samples
        or event.elapsed_ms < last_elapsed
        or event.finalization_ms > event.elapsed_ms
        or event.chunks != expected_chunks
        or event.model_padding_samples != 0
        or abs(event.audio_seconds - request.pcm.samples / request.pcm.sample_rate)
        > 1e-6
        or type(event.endpoint_observations) is not tuple
        or len(event.endpoint_observations) > MAX_MOBILE_ZIPFORMER_ENDPOINT_OBSERVATIONS
    ):
        raise WorkerError("worker_protocol")

    last_observed_sample = 0
    source_complete_seen = False
    total_text_chars = 0
    for observation in event.endpoint_observations:
        if (
            type(observation) is not MobileZipformerEndpointObservation
            or len(observation.text) > MAX_HYPOTHESIS_CHARS
            or type(observation.observed_sample) is not int
            or not 0 < observation.observed_sample <= event.samples_seen
            or type(observation.source_complete) is not bool
            or observation.observed_sample <= last_observed_sample
            or observation.source_complete
            != (observation.observed_sample >= request.pcm.samples)
            or source_complete_seen
        ):
            raise WorkerError("worker_protocol")
        total_text_chars += len(observation.text)
        if total_text_chars > MAX_HYPOTHESIS_CHARS:
            raise WorkerError("worker_protocol")
        last_observed_sample = observation.observed_sample
        source_complete_seen = observation.source_complete

    expected_text = " ".join(
        observation.text
        for observation in event.endpoint_observations
        if observation.text
    )
    terminal = event.endpoint_observations[-1] if event.endpoint_observations else None
    if event.text != expected_text:
        raise WorkerError("worker_protocol")
    if event.native_endpoint:
        expected_latency = event.tail_samples_consumed / 16.0
        if (
            event.endpoint_reason != "endpoint"
            or terminal is None
            or not terminal.source_complete
            or event.endpoint_sample != terminal.observed_sample
            or event.endpoint_sample != event.samples_seen
            or event.authoritative is not True
            or event.endpoint_latency_ms is None
            or not math.isclose(
                event.endpoint_latency_ms,
                expected_latency,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            raise WorkerError("worker_protocol")
        return
    if (
        event.endpoint_reason != "tail_exhausted"
        or event.endpoint_sample is not None
        or event.endpoint_latency_ms is not None
        or event.authoritative is not False
        or event.tail_samples_consumed != request.stream.tail_padding_samples
        or any(
            observation.source_complete for observation in event.endpoint_observations
        )
    ):
        raise WorkerError("worker_protocol")


def _stable_path_metadata(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _open_pinned_executable_descriptor(path: Path) -> int:
    """Open one stable executable without following a path replacement."""

    descriptor = -1
    try:
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink < 1
            or stat.S_IMODE(before.st_mode) & 0o111 == 0
        ):
            raise WorkerError("worker_prerequisite")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        current = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink < 1
            or stat.S_IMODE(opened.st_mode) & 0o111 == 0
            or _stable_path_metadata(opened) != _stable_path_metadata(before)
            or _stable_path_metadata(current) != _stable_path_metadata(before)
        ):
            raise WorkerError("worker_prerequisite")
        result = descriptor
        descriptor = -1
        return result
    except (OSError, RuntimeError, ValueError, OverflowError):
        raise WorkerError("worker_prerequisite") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _open_bwrap_descriptor() -> int:
    """Open the exact Bubblewrap executable without following a replacement."""

    return _open_pinned_executable_descriptor(_BWRAP_PATH)


def _open_systemd_run_descriptor() -> int:
    """Open the exact systemd-run executable used for the bounded scope."""

    return _open_pinned_executable_descriptor(_SYSTEMD_RUN_PATH)


def _verified_user_runtime_directory(
    *,
    root: Path | None = None,
) -> Path:
    """Return the private runtime directory needed by the local user manager."""

    if os.name != "posix" or not hasattr(os, "getuid"):
        raise WorkerError("worker_prerequisite")
    uid = os.getuid()
    selected_root = _USER_RUNTIME_ROOT if root is None else root
    runtime = selected_root / str(uid)
    try:
        before = runtime.lstat()
        resolved = runtime.resolve(strict=True)
        current = runtime.lstat()
        if (
            resolved != runtime.absolute()
            or not stat.S_ISDIR(before.st_mode)
            or before.st_uid != uid
            or stat.S_IMODE(before.st_mode) & 0o077 != 0
            or _stable_path_metadata(before) != _stable_path_metadata(current)
        ):
            raise WorkerError("worker_prerequisite")
    except (OSError, RuntimeError, ValueError, OverflowError):
        raise WorkerError("worker_prerequisite") from None
    return runtime


def _read_small_pseudo_file(path: Path, *, maximum_bytes: int = 4096) -> bytes:
    """Read one no-follow procfs/cgroupfs value under a fixed small bound."""

    descriptor = -1
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise WorkerError("worker_prerequisite")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise WorkerError("worker_prerequisite")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(
                descriptor,
                min(1024, maximum_bytes + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise WorkerError("worker_prerequisite")
        payload = b"".join(chunks)
        current = path.lstat()
        if _stable_path_metadata(opened) != _stable_path_metadata(
            before
        ) or _stable_path_metadata(current) != _stable_path_metadata(before):
            raise WorkerError("worker_prerequisite")
        return payload
    except (OSError, RuntimeError, ValueError, OverflowError):
        raise WorkerError("worker_prerequisite") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _single_ascii_line(payload: bytes) -> str:
    try:
        value = payload.decode("ascii", errors="strict")
    except UnicodeError:
        raise WorkerError("worker_prerequisite") from None
    if not value.endswith("\n") or "\n" in value[:-1] or "\r" in value:
        raise WorkerError("worker_prerequisite")
    return value[:-1]


def _parse_unified_cgroup_path(payload: bytes) -> Path:
    """Parse one cgroup-v2 proc line for an automatic systemd-run scope."""

    line = _single_ascii_line(payload)
    if not line.startswith("0::/") or len(line) > 4096:
        raise WorkerError("worker_prerequisite")
    raw_path = line[3:]
    parts = raw_path.split("/")
    if (
        not raw_path.startswith("/")
        or len(parts) < 2
        or any(
            not part
            or part in {".", ".."}
            or len(part) > 255
            or any(ord(character) < 0x21 or ord(character) > 0x7E for character in part)
            for part in parts[1:]
        )
        or re.fullmatch(r"run-[A-Za-z0-9_.:@-]{1,128}\.scope", parts[-1]) is None
    ):
        raise WorkerError("worker_prerequisite")
    return Path(raw_path)


def _verify_parakeet_cgroup_evidence(
    proc_cgroup: bytes,
    *,
    cgroup_root: Path | None = None,
) -> dict[str, object]:
    """Verify exact effective cgroup-v2 limits and return stable evidence."""

    selected_root = _CGROUP_ROOT if cgroup_root is None else cgroup_root
    relative = _parse_unified_cgroup_path(proc_cgroup).relative_to("/")
    try:
        root = selected_root.resolve(strict=True)
        controllers = set(
            _single_ascii_line(
                _read_small_pseudo_file(root / "cgroup.controllers")
            ).split()
        )
        scope = (root / relative).resolve(strict=True)
        scope.relative_to(root)
        if not scope.is_dir() or not {"cpu", "memory", "pids"} <= controllers:
            raise WorkerError("worker_prerequisite")
        values = {
            name: _single_ascii_line(_read_small_pseudo_file(scope / name))
            for name in (
                "memory.high",
                "memory.max",
                "memory.swap.max",
                "memory.oom.group",
                "cpu.max",
                "pids.max",
            )
        }
        quota_fields = values["cpu.max"].split()
        if len(quota_fields) != 2 or "max" in quota_fields:
            raise WorkerError("worker_prerequisite")
        quota, period = (int(field, 10) for field in quota_fields)
        if (
            values["memory.high"] != str(_PARAKEET_MEMORY_HIGH_BYTES)
            or values["memory.max"] != str(_PARAKEET_MEMORY_MAX_BYTES)
            or values["memory.swap.max"] != "0"
            or values["memory.oom.group"] != "1"
            or values["pids.max"] != str(_PARAKEET_TASKS_MAX)
            or quota <= 0
            or period <= 0
            or quota * 100 != _PARAKEET_CPU_QUOTA_PERCENT * period
        ):
            raise WorkerError("worker_prerequisite")
    except (OSError, RuntimeError, ValueError, OverflowError):
        raise WorkerError("worker_prerequisite") from None
    return {
        "kind": "systemd-user-scope-cgroup-v2",
        "memory_high_bytes": _PARAKEET_MEMORY_HIGH_BYTES,
        "memory_max_bytes": _PARAKEET_MEMORY_MAX_BYTES,
        "memory_swap_max_bytes": 0,
        "cpu_quota_percent": _PARAKEET_CPU_QUOTA_PERCENT,
        "tasks_max": _PARAKEET_TASKS_MAX,
        "oom_policy": "kill",
        "verified": True,
    }


def _verify_parakeet_cpp_cgroup_evidence(
    proc_cgroup: bytes,
    *,
    cgroup_root: Path | None = None,
) -> dict[str, object]:
    """Verify the smaller CPU-only candidate's exact effective limits."""

    selected_root = _CGROUP_ROOT if cgroup_root is None else cgroup_root
    relative = _parse_unified_cgroup_path(proc_cgroup).relative_to("/")
    try:
        root = selected_root.resolve(strict=True)
        controllers = set(
            _single_ascii_line(
                _read_small_pseudo_file(root / "cgroup.controllers")
            ).split()
        )
        scope = (root / relative).resolve(strict=True)
        scope.relative_to(root)
        if not scope.is_dir() or not {"cpu", "memory", "pids"} <= controllers:
            raise WorkerError("worker_prerequisite")
        values = {
            name: _single_ascii_line(_read_small_pseudo_file(scope / name))
            for name in (
                "memory.high",
                "memory.max",
                "memory.swap.max",
                "memory.oom.group",
                "cpu.max",
                "pids.max",
            )
        }
        quota_fields = values["cpu.max"].split()
        if len(quota_fields) != 2 or "max" in quota_fields:
            raise WorkerError("worker_prerequisite")
        quota, period = (int(field, 10) for field in quota_fields)
        if (
            values["memory.high"] != str(_PARAKEET_CPP_MEMORY_HIGH_BYTES)
            or values["memory.max"] != str(_PARAKEET_CPP_MEMORY_MAX_BYTES)
            or values["memory.swap.max"] != "0"
            or values["memory.oom.group"] != "1"
            or values["pids.max"] != str(_PARAKEET_CPP_TASKS_MAX)
            or quota <= 0
            or period <= 0
            or quota * 100 != _PARAKEET_CPP_CPU_QUOTA_PERCENT * period
        ):
            raise WorkerError("worker_prerequisite")
    except (OSError, RuntimeError, ValueError, OverflowError):
        raise WorkerError("worker_prerequisite") from None
    return {
        "kind": "systemd-user-scope-cgroup-v2",
        "memory_high_bytes": _PARAKEET_CPP_MEMORY_HIGH_BYTES,
        "memory_max_bytes": _PARAKEET_CPP_MEMORY_MAX_BYTES,
        "memory_swap_max_bytes": 0,
        "cpu_quota_percent": _PARAKEET_CPP_CPU_QUOTA_PERCENT,
        "tasks_max": _PARAKEET_CPP_TASKS_MAX,
        "oom_policy": "kill",
        "verified": True,
    }


def _verify_kyutai_cgroup_evidence(
    proc_cgroup: bytes,
    *,
    cgroup_root: Path | None = None,
) -> dict[str, object]:
    """Verify Kyutai's exact hard cgroup-v2 and lowest I/O weight."""

    selected_root = _CGROUP_ROOT if cgroup_root is None else cgroup_root
    relative = _parse_unified_cgroup_path(proc_cgroup).relative_to("/")
    try:
        root = selected_root.resolve(strict=True)
        controllers = set(
            _single_ascii_line(
                _read_small_pseudo_file(root / "cgroup.controllers")
            ).split()
        )
        scope = (root / relative).resolve(strict=True)
        scope.relative_to(root)
        if (
            not scope.is_dir()
            or not {
                "cpu",
                "io",
                "memory",
                "pids",
            }
            <= controllers
        ):
            raise WorkerError("worker_prerequisite")
        values = {
            name: _single_ascii_line(_read_small_pseudo_file(scope / name))
            for name in (
                "memory.high",
                "memory.max",
                "memory.swap.max",
                "memory.oom.group",
                "cpu.max",
                "pids.max",
                "io.weight",
            )
        }
        quota_fields = values["cpu.max"].split()
        if len(quota_fields) != 2 or "max" in quota_fields:
            raise WorkerError("worker_prerequisite")
        quota, period = (int(field, 10) for field in quota_fields)
        if (
            values["memory.high"] != str(_KYUTAI_MEMORY_HIGH_BYTES)
            or values["memory.max"] != str(_KYUTAI_MEMORY_MAX_BYTES)
            or values["memory.swap.max"] != "0"
            or values["memory.oom.group"] != "1"
            or values["pids.max"] != str(_KYUTAI_TASKS_MAX)
            or values["io.weight"] != f"default {_KYUTAI_IO_WEIGHT}"
            or quota <= 0
            or period <= 0
            or quota * 100 != _KYUTAI_CPU_QUOTA_PERCENT * period
        ):
            raise WorkerError("worker_prerequisite")
    except (OSError, RuntimeError, ValueError, OverflowError):
        raise WorkerError("worker_prerequisite") from None
    return {
        "kind": "systemd-user-scope-cgroup-v2",
        "memory_high_bytes": _KYUTAI_MEMORY_HIGH_BYTES,
        "memory_max_bytes": _KYUTAI_MEMORY_MAX_BYTES,
        "memory_swap_max_bytes": 0,
        "cpu_quota_percent": _KYUTAI_CPU_QUOTA_PERCENT,
        "tasks_max": _KYUTAI_TASKS_MAX,
        "io_weight": _KYUTAI_IO_WEIGHT,
        "oom_policy": "kill",
        "verified": True,
    }


def _read_parakeet_cgroup_evidence(
    pid: int,
    *,
    proc_root: Path | None = None,
    cgroup_root: Path | None = None,
) -> dict[str, object]:
    if type(pid) is not int or pid <= 0:
        raise WorkerError("worker_prerequisite")
    selected_proc = _PROC_ROOT if proc_root is None else proc_root
    return _verify_parakeet_cgroup_evidence(
        _read_small_pseudo_file(selected_proc / str(pid) / "cgroup"),
        cgroup_root=cgroup_root,
    )


def _read_parakeet_cpp_cgroup_evidence(
    pid: int,
    *,
    proc_root: Path | None = None,
    cgroup_root: Path | None = None,
) -> dict[str, object]:
    if type(pid) is not int or pid <= 0:
        raise WorkerError("worker_prerequisite")
    selected_proc = _PROC_ROOT if proc_root is None else proc_root
    return _verify_parakeet_cpp_cgroup_evidence(
        _read_small_pseudo_file(selected_proc / str(pid) / "cgroup"),
        cgroup_root=cgroup_root,
    )


def _read_kyutai_cgroup_evidence(
    pid: int,
    *,
    proc_root: Path | None = None,
    cgroup_root: Path | None = None,
) -> dict[str, object]:
    if type(pid) is not int or pid <= 0:
        raise WorkerError("worker_prerequisite")
    selected_proc = _PROC_ROOT if proc_root is None else proc_root
    return _verify_kyutai_cgroup_evidence(
        _read_small_pseudo_file(selected_proc / str(pid) / "cgroup"),
        cgroup_root=cgroup_root,
    )


def _bind_cgroup_observer(
    pid: int,
    *,
    cpp: bool,
    kyutai: bool = False,
    proc_root: Path | None = None,
    cgroup_root: Path | None = None,
) -> tuple[dict[str, object], CgroupScopeObserver]:
    """Pin the same exact scope whose effective limits were verified."""

    if type(pid) is not int or pid <= 0 or (cpp and kyutai):
        raise WorkerError("worker_prerequisite")
    selected_proc = _PROC_ROOT if proc_root is None else proc_root
    selected_cgroup = _CGROUP_ROOT if cgroup_root is None else cgroup_root
    proc_path = selected_proc / str(pid) / "cgroup"
    payload = _read_small_pseudo_file(proc_path)
    verifier = (
        _verify_kyutai_cgroup_evidence
        if kyutai
        else (
            _verify_parakeet_cpp_cgroup_evidence
            if cpp
            else _verify_parakeet_cgroup_evidence
        )
    )
    evidence = verifier(payload, cgroup_root=selected_cgroup)
    try:
        root = selected_cgroup.resolve(strict=True)
        relative = _parse_unified_cgroup_path(payload).relative_to("/")
        scope = root / relative
        resolved = scope.resolve(strict=True)
        resolved.relative_to(root)
        if resolved != scope.absolute():
            raise WorkerError("worker_prerequisite")
        observer = CgroupScopeObserver.bind(
            scope_path=scope,
            proc_cgroup_path=proc_path,
            expected_proc_cgroup=payload,
        )
    except CgroupObservationError:
        raise WorkerError("worker_prerequisite") from None
    except (OSError, RuntimeError, ValueError, OverflowError):
        raise WorkerError("worker_prerequisite") from None
    return evidence, observer


def _await_parakeet_cgroup(
    process: subprocess.Popen[bytes],
    *,
    monotonic=time.monotonic,
    sleeper=time.sleep,
) -> dict[str, object]:
    """Wait briefly for systemd's atomic PID migration, then fail closed."""

    deadline = monotonic() + _CGROUP_SETUP_TIMEOUT_SEC
    while True:
        if process.poll() is not None:
            raise WorkerError("worker_prerequisite")
        try:
            return _read_parakeet_cgroup_evidence(process.pid)
        except WorkerError:
            remaining = deadline - monotonic()
            if remaining <= 0.0:
                raise WorkerError("worker_prerequisite") from None
            sleeper(min(0.01, remaining))


def _await_parakeet_cpp_cgroup(
    process: subprocess.Popen[bytes],
    *,
    monotonic=time.monotonic,
    sleeper=time.sleep,
) -> dict[str, object]:
    """Wait for the CPU-only worker's exact systemd cgroup migration."""

    deadline = monotonic() + _CGROUP_SETUP_TIMEOUT_SEC
    while True:
        if process.poll() is not None:
            raise WorkerError("worker_prerequisite")
        try:
            return _read_parakeet_cpp_cgroup_evidence(process.pid)
        except WorkerError:
            remaining = deadline - monotonic()
            if remaining <= 0.0:
                raise WorkerError("worker_prerequisite") from None
            sleeper(min(0.01, remaining))


def _await_kyutai_cgroup(
    process: subprocess.Popen[bytes],
    *,
    monotonic=time.monotonic,
    sleeper=time.sleep,
) -> dict[str, object]:
    """Wait for the Kyutai worker's exact hard systemd scope."""

    deadline = monotonic() + _CGROUP_SETUP_TIMEOUT_SEC
    while True:
        if process.poll() is not None:
            raise WorkerError("worker_prerequisite")
        try:
            return _read_kyutai_cgroup_evidence(process.pid)
        except WorkerError:
            remaining = deadline - monotonic()
            if remaining <= 0.0:
                raise WorkerError("worker_prerequisite") from None
            sleeper(min(0.01, remaining))


def _verify_kyutai_host_memory_available(
    *,
    meminfo_path: Path | None = None,
    minimum_bytes: int = _KYUTAI_MINIMUM_HOST_AVAILABLE_BYTES,
) -> None:
    """Fail before launch unless Linux reports the frozen RAM headroom."""

    if (
        type(minimum_bytes) is not int
        or minimum_bytes != _KYUTAI_MINIMUM_HOST_AVAILABLE_BYTES
    ):
        raise WorkerError("worker_prerequisite")
    selected = Path("/proc/meminfo") if meminfo_path is None else meminfo_path
    payload = _read_small_pseudo_file(selected, maximum_bytes=16 * 1024)
    try:
        text = payload.decode("ascii", errors="strict")
        matches = re.findall(r"(?m)^MemAvailable:[ \t]+([0-9]+)[ \t]+kB$", text)
        if len(matches) != 1:
            raise WorkerError("worker_prerequisite")
        available_kib = int(matches[0], 10)
        if available_kib <= 0 or available_kib * 1024 < minimum_bytes:
            raise WorkerError("worker_prerequisite")
    except (UnicodeError, RuntimeError, ValueError, OverflowError):
        raise WorkerError("worker_prerequisite") from None


def _require_mount_source(path: Path, *, directory: bool) -> None:
    """Require one stable system mount source of the expected broad type."""

    try:
        before = path.lstat()
        resolved = path.resolve(strict=True)
        current = path.lstat()
        target = resolved.stat()
    except (OSError, RuntimeError, ValueError, OverflowError):
        raise WorkerError("worker_prerequisite") from None
    if _stable_path_metadata(before) != _stable_path_metadata(current):
        raise WorkerError("worker_prerequisite")
    if directory:
        if not stat.S_ISDIR(target.st_mode):
            raise WorkerError("worker_prerequisite")
    elif not stat.S_ISREG(target.st_mode):
        raise WorkerError("worker_prerequisite")


def _nemotron_device_nodes() -> tuple[Path, ...]:
    """Return the closed set of NVIDIA character devices exposed to the worker."""

    paths = tuple(
        Path("/dev") / name
        for name in (
            *_NVIDIA_REQUIRED_DEVICE_NAMES,
            *_NVIDIA_OPTIONAL_DEVICE_NAMES,
        )
    )
    selected: list[Path] = []
    for index, path in enumerate(paths):
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            if index < len(_NVIDIA_REQUIRED_DEVICE_NAMES):
                raise WorkerError("worker_prerequisite") from None
            continue
        except OSError:
            raise WorkerError("worker_prerequisite") from None
        if not stat.S_ISCHR(metadata.st_mode):
            raise WorkerError("worker_prerequisite")
        selected.append(path)

    caps_root = Path("/dev/nvidia-caps")
    try:
        entries = tuple(sorted(caps_root.iterdir(), key=lambda item: item.name))
    except FileNotFoundError:
        entries = ()
    except OSError:
        raise WorkerError("worker_prerequisite") from None
    if len(entries) > 64:
        raise WorkerError("worker_prerequisite")
    for path in entries:
        if _NVIDIA_CAP_RE.fullmatch(path.name) is None:
            continue
        try:
            metadata = path.lstat()
        except OSError:
            raise WorkerError("worker_prerequisite") from None
        if not stat.S_ISCHR(metadata.st_mode):
            raise WorkerError("worker_prerequisite")
        selected.append(path)
    return tuple(selected)


def _nemotron_system_ro_paths() -> tuple[Path, ...]:
    """Return the narrow host runtime and NVIDIA sysfs closure."""

    core_paths = _core_system_ro_paths()
    _require_mount_source(_NVIDIA_DRIVER_ROOT, directory=True)
    _require_mount_source(_NVIDIA_MODULE_PATHS[0], directory=True)
    _require_mount_source(_NVIDIA_MODULE_PATHS[1], directory=True)

    pci_targets: list[Path] = []
    try:
        entries = tuple(
            sorted(_NVIDIA_DRIVER_ROOT.iterdir(), key=lambda item: item.name)
        )
    except OSError:
        raise WorkerError("worker_prerequisite") from None
    for entry in entries:
        if _NVIDIA_PCI_RE.fullmatch(entry.name) is None:
            continue
        try:
            target = entry.resolve(strict=True)
        except (OSError, RuntimeError):
            raise WorkerError("worker_prerequisite") from None
        _require_mount_source(target, directory=True)
        pci_targets.append(target)
    if not pci_targets:
        raise WorkerError("worker_prerequisite")

    modules: list[Path] = []
    for path in _NVIDIA_MODULE_PATHS:
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        except OSError:
            raise WorkerError("worker_prerequisite") from None
        _require_mount_source(path, directory=True)
        modules.append(path)

    return (
        *core_paths,
        _NVIDIA_DRIVER_ROOT,
        *tuple(dict.fromkeys(pci_targets)),
        *modules,
    )


def _core_system_ro_paths() -> tuple[Path, ...]:
    """Return the minimal local interpreter/native-library mount closure."""

    for path in _NEMOTRON_CORE_RO_PATHS[:3]:
        _require_mount_source(path, directory=True)
    for path in _NEMOTRON_CORE_RO_PATHS[3:]:
        _require_mount_source(path, directory=False)
    return _NEMOTRON_CORE_RO_PATHS


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _paths_overlap(left: Path, right: Path) -> bool:
    return _path_is_within(left, right) or _path_is_within(right, left)


def _validate_mobile_whisper_scratch_authorities(
    manifest: WorkerManifest,
    scratch: Path,
    source_bundle: SourceBundle,
) -> None:
    """Keep schema-11 writable scratch outside every external authority."""

    if manifest.adapter != MOBILE_WHISPER_ADAPTER:
        return
    try:
        model_roots = {
            artifact.path.parent
            for artifact in manifest.artifacts
            if artifact.name.startswith("model-")
        }
        venv_root = manifest.python.path.parent.parent
        provision_root = manifest.path.parent
        protected_roots = (
            venv_root,
            provision_root,
            *model_roots,
            _CANONICAL_REPO_ROOT,
        )
        protected_paths = (
            manifest.worker.path,
            *(control.path for control in manifest.control_files),
        )
        required = (
            scratch,
            source_bundle.root,
            source_bundle.worker_path,
            *protected_roots,
            *protected_paths,
        )
        if (
            manifest.schema_version != 11
            or type(manifest.adapter_config) is not MobileWhisperConfig
            or len(model_roots) != 1
            or any(
                not path.is_absolute() or "\x00" in str(path)
                for path in required
            )
            or any(_paths_overlap(scratch, path) for path in protected_roots)
            or any(_paths_overlap(scratch, path) for path in protected_paths)
            # The verified staged bundle is intentionally placed beneath
            # scratch and overlaid read-only. The reverse topology is unsafe.
            or _path_is_within(scratch, source_bundle.root)
        ):
            raise WorkerError("worker_prerequisite")
    except WorkerError:
        raise
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        raise WorkerError("worker_prerequisite") from None


def _verify_nemotron_runtime_layout(
    manifest: WorkerManifest,
    receipt: RuntimeTreeReceipt,
) -> None:
    config = manifest.adapter_config
    wheel_lock = manifest.artifact_by_name.get("runtime-wheel-lock")
    venv_root = manifest.python.path.parent.parent
    python_root = receipt.root.parent
    maximum_file = max((item.size_bytes for item in receipt.files), default=0)
    try:
        if (
            not isinstance(
                config,
                (KyutaiConfig, NemotronConfig, ParakeetRealtimeEouConfig),
            )
            or wheel_lock is None
            or wheel_lock.sha256 != config.wheel_lock_sha256
            or receipt.content_digest != config.runtime_content_sha256
            or receipt.file_count != config.runtime_file_count
            or receipt.total_size_bytes != config.runtime_total_size_bytes
            or maximum_file != config.runtime_maximum_file_bytes
            or set(os.listdir(venv_root)) != {"bin", "lib", "pyvenv.cfg"}
            or set(os.listdir(venv_root / "bin")) != {"python"}
            or set(os.listdir(venv_root / "lib")) != {"python3.12"}
            or set(os.listdir(python_root)) != {"site-packages"}
            or not manifest.python.path.is_symlink()
            or manifest.python.path.resolve(strict=True).name != "python3.12"
        ):
            raise WorkerError("worker_artifact_changed")
    except (OSError, RuntimeError, ValueError):
        raise WorkerError("worker_artifact_changed") from None


def _faster_whisper_tree_matches(
    receipt: RuntimeTreeReceipt,
    *,
    content_sha256: str,
    file_count: int,
    total_size_bytes: int,
    maximum_file_bytes: int,
) -> bool:
    return (
        receipt.content_digest == content_sha256
        and receipt.file_count == file_count
        and receipt.total_size_bytes == total_size_bytes
        and max((item.size_bytes for item in receipt.files), default=0)
        == maximum_file_bytes
    )


def _load_faster_whisper_model_receipt(
    manifest: WorkerManifest,
) -> RuntimeTreeReceipt:
    config = manifest.adapter_config
    artifact = manifest.artifact_by_name.get("model-receipt")
    if not isinstance(config, FasterWhisperEndpointConfig) or artifact is None:
        raise WorkerError("worker_artifact_changed")
    try:
        receipt = load_runtime_tree_receipt(
            artifact.path,
            expected_digest=artifact.sha256,
            limits=FASTER_WHISPER_MODEL_TREE_LIMITS,
        )
    except RuntimeError:
        raise WorkerError("worker_artifact_changed") from None
    paths = set(receipt.file_by_path)
    if (
        receipt.digest != artifact.sha256
        or not FASTER_WHISPER_REQUIRED_MODEL_FILES.issubset(paths)
        or not FASTER_WHISPER_VOCABULARY_FILES.intersection(paths)
        or not _faster_whisper_tree_matches(
            receipt,
            content_sha256=config.model_content_sha256,
            file_count=config.model_file_count,
            total_size_bytes=config.model_total_size_bytes,
            maximum_file_bytes=config.model_maximum_file_bytes,
        )
    ):
        raise WorkerError("worker_artifact_changed")
    return receipt


def _verify_faster_whisper_runtime_layout(
    manifest: WorkerManifest,
    receipt: RuntimeTreeReceipt,
) -> RuntimeTreeReceipt:
    config = manifest.adapter_config
    if not isinstance(
        config, FasterWhisperEndpointConfig
    ) or not _faster_whisper_tree_matches(
        receipt,
        content_sha256=config.runtime_content_sha256,
        file_count=config.runtime_file_count,
        total_size_bytes=config.runtime_total_size_bytes,
        maximum_file_bytes=config.runtime_maximum_file_bytes,
    ):
        raise WorkerError("worker_artifact_changed")
    model = _load_faster_whisper_model_receipt(manifest)
    try:
        receipt.root.relative_to(model.root)
    except ValueError:
        pass
    else:
        raise WorkerError("worker_artifact_changed")
    try:
        model.root.relative_to(receipt.root)
    except ValueError:
        pass
    else:
        raise WorkerError("worker_artifact_changed")
    return model


def _append_mount(
    command: list[str],
    option: str,
    source: Path,
    destination: Path | None = None,
) -> None:
    command.extend((option, str(source), str(destination or source)))


def _direct_model_gpu_bwrap_command(
    manifest: WorkerManifest,
    scratch: Path,
    source_bundle: SourceBundle,
    *,
    bundle_descriptor: int,
    worker_descriptor: int,
    python_descriptor: int,
    device_nodes: Sequence[Path],
    system_ro_paths: Sequence[Path],
    proc_driver_available: bool,
) -> list[str]:
    """Build a sealed GPU command for an exact direct-artifact model root."""

    artifacts = manifest.artifact_by_name
    try:
        model_artifacts = tuple(
            artifact
            for artifact in manifest.artifacts
            if artifact.name.startswith("model-")
        )
        model_roots = {artifact.path.parent for artifact in model_artifacts}
        if len(model_roots) != 1:
            raise WorkerError("worker_prerequisite")
        model_root = model_roots.pop()
        runtime_receipt = artifacts["runtime-receipt"].path
        runtime_wheel_lock = artifacts["runtime-wheel-lock"].path
        if set(os.listdir(model_root)) != {
            artifact.path.name for artifact in model_artifacts
        }:
            raise WorkerError("worker_prerequisite")
    except (KeyError, OSError, RuntimeError, ValueError):
        raise WorkerError("worker_prerequisite") from None
    venv_root = manifest.python.path.parent.parent
    required_paths = (
        scratch,
        source_bundle.root,
        source_bundle.worker_path,
        manifest.path,
        manifest.worker.path,
        venv_root,
        model_root,
        runtime_receipt,
        runtime_wheel_lock,
    )
    if any(not path.is_absolute() or "\x00" in str(path) for path in required_paths):
        raise WorkerError("worker_prerequisite")
    if any(
        _path_is_within(model_root, root) or _path_is_within(root, model_root)
        for root in (scratch, source_bundle.root, venv_root)
    ):
        raise WorkerError("worker_prerequisite")

    command = [
        str(_BWRAP_PATH),
        "--unshare-user-try",
        "--unshare-ipc",
        "--unshare-pid",
        "--unshare-net",
        "--unshare-uts",
        "--unshare-cgroup-try",
        "--die-with-parent",
        "--new-session",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
    ]
    for path in system_ro_paths:
        _append_mount(command, "--ro-bind", path)
    if proc_driver_available:
        _append_mount(command, "--ro-bind", _NVIDIA_PROC_DRIVER)
    for path in device_nodes:
        _append_mount(command, "--dev-bind", path)

    # Scratch is the only host-backed writable mount. More-specific candidate
    # and source mounts are overlaid read-only after it.
    _append_mount(command, "--bind", scratch)
    _append_mount(command, "--ro-bind", source_bundle.root)
    _append_mount(
        command,
        "--ro-bind",
        source_bundle.worker_path,
        manifest.worker.path,
    )
    _append_mount(command, "--ro-bind", manifest.path)
    _append_mount(command, "--ro-bind", venv_root)
    _append_mount(command, "--ro-bind", model_root)
    read_only_roots = (source_bundle.root, venv_root, model_root)
    if not any(_path_is_within(runtime_receipt, root) for root in read_only_roots):
        _append_mount(command, "--ro-bind", runtime_receipt)
    if not any(_path_is_within(runtime_wheel_lock, root) for root in read_only_roots):
        _append_mount(command, "--ro-bind", runtime_wheel_lock)

    bundle_root = Path(f"/proc/self/fd/{bundle_descriptor}")
    command.extend(
        (
            "--chdir",
            str(scratch),
            "--argv0",
            str(manifest.python.path),
            "--",
            f"/proc/self/fd/{python_descriptor}",
            "-I",
            "-S",
            "-B",
            f"/proc/self/fd/{worker_descriptor}",
            "--manifest",
            str(manifest.path),
            "--scratch-root",
            str(scratch),
            "--source-bundle-manifest",
            str(bundle_root / source_bundle.manifest_path.name),
            "--source-bundle-sha256",
            source_bundle.tree_sha256,
            "--source-bundle-fd",
            str(bundle_descriptor),
        )
    )
    return command


def _nemotron_bwrap_command(
    manifest: WorkerManifest,
    scratch: Path,
    source_bundle: SourceBundle,
    *,
    bundle_descriptor: int,
    worker_descriptor: int,
    python_descriptor: int,
    device_nodes: Sequence[Path],
    system_ro_paths: Sequence[Path],
    proc_driver_available: bool,
) -> list[str]:
    """Retain the stable Nemotron helper seam."""

    return _direct_model_gpu_bwrap_command(
        manifest,
        scratch,
        source_bundle,
        bundle_descriptor=bundle_descriptor,
        worker_descriptor=worker_descriptor,
        python_descriptor=python_descriptor,
        device_nodes=device_nodes,
        system_ro_paths=system_ro_paths,
        proc_driver_available=proc_driver_available,
    )


def _kyutai_bwrap_command(
    manifest: WorkerManifest,
    scratch: Path,
    source_bundle: SourceBundle,
    *,
    bundle_descriptor: int,
    worker_descriptor: int,
    python_descriptor: int,
    device_nodes: Sequence[Path],
    system_ro_paths: Sequence[Path],
    proc_driver_available: bool,
) -> list[str]:
    """Build Kyutai's networkless exact-four-file model sandbox."""

    return _direct_model_gpu_bwrap_command(
        manifest,
        scratch,
        source_bundle,
        bundle_descriptor=bundle_descriptor,
        worker_descriptor=worker_descriptor,
        python_descriptor=python_descriptor,
        device_nodes=device_nodes,
        system_ro_paths=system_ro_paths,
        proc_driver_available=proc_driver_available,
    )


def _faster_whisper_bwrap_command(
    manifest: WorkerManifest,
    scratch: Path,
    source_bundle: SourceBundle,
    *,
    bundle_descriptor: int,
    worker_descriptor: int,
    python_descriptor: int,
    device_nodes: Sequence[Path],
    system_ro_paths: Sequence[Path],
    proc_driver_available: bool,
) -> list[str]:
    """Build the networkless, receipt-mounted Faster-Whisper command."""

    artifacts = manifest.artifact_by_name
    try:
        runtime_artifact = artifacts["runtime-receipt"]
        model_artifact = artifacts["model-receipt"]
        runtime = load_runtime_tree_receipt(
            runtime_artifact.path,
            expected_digest=runtime_artifact.sha256,
            limits=FASTER_WHISPER_RUNTIME_TREE_LIMITS,
        )
        verify_venv_runtime_location(runtime, manifest.python.path)
        model = _verify_faster_whisper_runtime_layout(manifest, runtime)
    except (KeyError, RuntimeError):
        raise WorkerError("worker_prerequisite") from None
    venv_root = manifest.python.path.parent.parent
    model_root = model.root
    required_paths = (
        scratch,
        source_bundle.root,
        source_bundle.worker_path,
        manifest.path,
        manifest.worker.path,
        venv_root,
        model_root,
        runtime_artifact.path,
        model_artifact.path,
    )
    if any(not path.is_absolute() or "\x00" in str(path) for path in required_paths):
        raise WorkerError("worker_prerequisite")

    command = [
        str(_BWRAP_PATH),
        "--unshare-user-try",
        "--unshare-ipc",
        "--unshare-pid",
        "--unshare-net",
        "--unshare-uts",
        "--unshare-cgroup-try",
        "--die-with-parent",
        "--new-session",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
    ]
    for path in system_ro_paths:
        _append_mount(command, "--ro-bind", path)
    if proc_driver_available:
        _append_mount(command, "--ro-bind", _NVIDIA_PROC_DRIVER)
    for path in device_nodes:
        _append_mount(command, "--dev-bind", path)

    _append_mount(command, "--bind", scratch)
    _append_mount(command, "--ro-bind", source_bundle.root)
    _append_mount(
        command,
        "--ro-bind",
        source_bundle.worker_path,
        manifest.worker.path,
    )
    _append_mount(command, "--ro-bind", manifest.path)
    _append_mount(command, "--ro-bind", venv_root)
    _append_mount(command, "--ro-bind", model_root)
    read_only_roots = (source_bundle.root, venv_root, model_root)
    for receipt_path in (runtime_artifact.path, model_artifact.path):
        if not any(_path_is_within(receipt_path, root) for root in read_only_roots):
            _append_mount(command, "--ro-bind", receipt_path)

    bundle_root = Path(f"/proc/self/fd/{bundle_descriptor}")
    command.extend(
        (
            "--chdir",
            str(scratch),
            "--argv0",
            str(manifest.python.path),
            "--",
            f"/proc/self/fd/{python_descriptor}",
            "-I",
            "-S",
            "-B",
            f"/proc/self/fd/{worker_descriptor}",
            "--manifest",
            str(manifest.path),
            "--scratch-root",
            str(scratch),
            "--source-bundle-manifest",
            str(bundle_root / source_bundle.manifest_path.name),
            "--source-bundle-sha256",
            source_bundle.tree_sha256,
            "--source-bundle-fd",
            str(bundle_descriptor),
        )
    )
    return command


def _parakeet_cpp_bwrap_command(
    manifest: WorkerManifest,
    scratch: Path,
    source_bundle: SourceBundle,
    *,
    bundle_descriptor: int,
    worker_descriptor: int,
    python_descriptor: int,
    system_ro_paths: Sequence[Path],
) -> list[str]:
    """Build the closed, networkless CPU-only parakeet.cpp command."""

    config = manifest.adapter_config
    if (
        manifest.schema_version != 8
        or manifest.adapter != PARAKEET_CPP_ADAPTER
        or not isinstance(config, ParakeetCppConfig)
        or config.requested_device != "cpu"
        or config.actual_device != "cpu"
        or config.num_threads != 1
    ):
        raise WorkerError("worker_prerequisite")
    artifacts = manifest.artifact_by_name
    native_names = (
        "source-receipt",
        "build-receipt",
        "libparakeet",
        "bridge-library",
    )
    model_names = ("model-receipt", "model-gguf")
    try:
        native_roots = {artifacts[name].path.parent for name in native_names}
        model_roots = {artifacts[name].path.parent for name in model_names}
        if len(native_roots) != 1 or len(model_roots) != 1:
            raise WorkerError("worker_prerequisite")
        native_root = native_roots.pop()
        model_root = model_roots.pop()
        if (
            _path_is_within(native_root, model_root)
            or _path_is_within(model_root, native_root)
            or set(os.listdir(native_root))
            != {artifacts[name].path.name for name in native_names}
            or set(os.listdir(model_root))
            != {artifacts[name].path.name for name in model_names}
        ):
            raise WorkerError("worker_prerequisite")
    except (KeyError, OSError, RuntimeError, ValueError):
        raise WorkerError("worker_prerequisite") from None
    venv_root = manifest.python.path.parent.parent
    required_paths = (
        scratch,
        source_bundle.root,
        source_bundle.worker_path,
        manifest.path,
        manifest.worker.path,
        venv_root,
        native_root,
        model_root,
    )
    if any(not path.is_absolute() or "\x00" in str(path) for path in required_paths):
        raise WorkerError("worker_prerequisite")

    command = [
        str(_BWRAP_PATH),
        "--unshare-user-try",
        "--unshare-ipc",
        "--unshare-pid",
        "--unshare-net",
        "--unshare-uts",
        "--unshare-cgroup-try",
        "--die-with-parent",
        "--new-session",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
    ]
    for path in system_ro_paths:
        _append_mount(command, "--ro-bind", path)
    _append_mount(command, "--bind", scratch)
    _append_mount(command, "--ro-bind", source_bundle.root)
    _append_mount(
        command,
        "--ro-bind",
        source_bundle.worker_path,
        manifest.worker.path,
    )
    _append_mount(command, "--ro-bind", manifest.path)
    if not any(_path_is_within(venv_root, root) for root in system_ro_paths):
        _append_mount(command, "--ro-bind", venv_root)
    _append_mount(command, "--ro-bind", native_root)
    _append_mount(command, "--ro-bind", model_root)

    bundle_root = Path(f"/proc/self/fd/{bundle_descriptor}")
    command.extend(
        (
            "--chdir",
            str(scratch),
            "--argv0",
            str(manifest.python.path),
            "--",
            f"/proc/self/fd/{python_descriptor}",
            "-I",
            "-S",
            "-B",
            f"/proc/self/fd/{worker_descriptor}",
            "--manifest",
            str(manifest.path),
            "--scratch-root",
            str(scratch),
            "--source-bundle-manifest",
            str(bundle_root / source_bundle.manifest_path.name),
            "--source-bundle-sha256",
            source_bundle.tree_sha256,
            "--source-bundle-fd",
            str(bundle_descriptor),
        )
    )
    return command


def _parakeet_systemd_scope_command(
    bwrap_command: Sequence[str],
    *,
    bwrap_descriptor: int,
) -> list[str]:
    """Place the unchanged Parakeet sandbox beneath one hard user scope."""

    if (
        type(bwrap_descriptor) is not int
        or bwrap_descriptor < 0
        or not bwrap_command
        or bwrap_command[0] != str(_BWRAP_PATH)
        or any(not isinstance(value, str) or "\x00" in value for value in bwrap_command)
    ):
        raise WorkerError("worker_prerequisite")
    inner = [
        f"/proc/self/fd/{bwrap_descriptor}",
        "--unsetenv",
        "XDG_RUNTIME_DIR",
        "--unsetenv",
        "INVOCATION_ID",
        *bwrap_command[1:],
    ]
    command = [
        str(_SYSTEMD_RUN_PATH),
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        "--no-ask-password",
        "--expand-environment=no",
        "--slice-inherit",
        "--nice=15",
        "--description=speaker-parakeet-realtime-eou-worker",
    ]
    for value in _PARAKEET_SCOPE_PROPERTIES:
        command.extend(("--property", value))
    command.extend(("--", *inner))
    return command


def _parakeet_cpp_systemd_scope_command(
    bwrap_command: Sequence[str],
    *,
    bwrap_descriptor: int,
) -> list[str]:
    """Place the CPU-only candidate beneath its smaller hard user scope."""

    if (
        type(bwrap_descriptor) is not int
        or bwrap_descriptor < 0
        or not bwrap_command
        or bwrap_command[0] != str(_BWRAP_PATH)
        or any(not isinstance(value, str) or "\x00" in value for value in bwrap_command)
    ):
        raise WorkerError("worker_prerequisite")
    inner = [
        f"/proc/self/fd/{bwrap_descriptor}",
        "--unsetenv",
        "XDG_RUNTIME_DIR",
        "--unsetenv",
        "INVOCATION_ID",
        *bwrap_command[1:],
    ]
    command = [
        str(_SYSTEMD_RUN_PATH),
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        "--no-ask-password",
        "--expand-environment=no",
        "--slice-inherit",
        "--nice=15",
        "--description=speaker-parakeet-cpp-worker",
    ]
    for value in _PARAKEET_CPP_SCOPE_PROPERTIES:
        command.extend(("--property", value))
    command.extend(("--", *inner))
    return command


def _kyutai_systemd_scope_command(
    bwrap_command: Sequence[str],
    *,
    bwrap_descriptor: int,
) -> list[str]:
    """Place the sealed Kyutai GPU worker beneath its dedicated hard scope."""

    if (
        type(bwrap_descriptor) is not int
        or bwrap_descriptor < 0
        or not bwrap_command
        or bwrap_command[0] != str(_BWRAP_PATH)
        or any(not isinstance(value, str) or "\x00" in value for value in bwrap_command)
    ):
        raise WorkerError("worker_prerequisite")
    inner = [
        f"/proc/self/fd/{bwrap_descriptor}",
        "--unsetenv",
        "XDG_RUNTIME_DIR",
        "--unsetenv",
        "INVOCATION_ID",
        *bwrap_command[1:],
    ]
    command = [
        str(_SYSTEMD_RUN_PATH),
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        "--no-ask-password",
        "--expand-environment=no",
        "--slice-inherit",
        "--nice=15",
        "--description=speaker-kyutai-stt-worker",
    ]
    for value in _KYUTAI_SCOPE_PROPERTIES:
        command.extend(("--property", value))
    command.extend(("--", *inner))
    return command


def _zipformer_bwrap_command(
    manifest: WorkerManifest,
    scratch: Path,
    source_bundle: SourceBundle,
    *,
    bundle_descriptor: int,
    worker_descriptor: int,
    python_descriptor: int,
    system_ro_paths: Sequence[Path],
) -> list[str]:
    """Build the networkless, read-only production-runtime baseline command."""

    try:
        model_artifacts = tuple(
            artifact
            for artifact in manifest.artifacts
            if artifact.name.startswith("model-")
        )
        model_roots = {artifact.path.parent for artifact in model_artifacts}
        if len(model_roots) != 1 or not model_artifacts:
            raise WorkerError("worker_prerequisite")
        model_root = model_roots.pop()
    except (OSError, RuntimeError, ValueError):
        raise WorkerError("worker_prerequisite") from None
    _validate_mobile_whisper_scratch_authorities(
        manifest,
        scratch,
        source_bundle,
    )
    venv_root = manifest.python.path.parent.parent
    required_paths = (
        scratch,
        source_bundle.root,
        source_bundle.worker_path,
        manifest.path,
        manifest.worker.path,
        venv_root,
        model_root,
        *(control.path for control in manifest.control_files),
    )
    if any(not path.is_absolute() or "\x00" in str(path) for path in required_paths):
        raise WorkerError("worker_prerequisite")

    command = [
        str(_BWRAP_PATH),
        "--unshare-user-try",
        "--unshare-ipc",
        "--unshare-pid",
        "--unshare-net",
        "--unshare-uts",
        "--unshare-cgroup-try",
        "--die-with-parent",
        "--new-session",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
    ]
    for path in system_ro_paths:
        _append_mount(command, "--ro-bind", path)
    _append_mount(command, "--bind", scratch)
    _append_mount(command, "--ro-bind", source_bundle.root)
    _append_mount(
        command,
        "--ro-bind",
        source_bundle.worker_path,
        manifest.worker.path,
    )
    _append_mount(command, "--ro-bind", manifest.path)
    for control_file in manifest.control_files:
        _append_mount(command, "--ro-bind", control_file.path)
    _append_mount(command, "--ro-bind", venv_root)
    _append_mount(command, "--ro-bind", model_root)

    bundle_root = Path(f"/proc/self/fd/{bundle_descriptor}")
    interpreter_flags = _worker_interpreter_flags(manifest)
    command.extend(
        (
            "--chdir",
            str(scratch),
            "--argv0",
            str(manifest.python.path),
            "--",
            f"/proc/self/fd/{python_descriptor}",
            *interpreter_flags,
            f"/proc/self/fd/{worker_descriptor}",
            "--manifest",
            str(manifest.path),
            "--scratch-root",
            str(scratch),
            "--source-bundle-manifest",
            str(bundle_root / source_bundle.manifest_path.name),
            "--source-bundle-sha256",
            source_bundle.tree_sha256,
            "--source-bundle-fd",
            str(bundle_descriptor),
        )
    )
    return command


class StreamingWorker:
    """Own one sequential worker and enforce the bounded JSONL lifecycle."""

    def __init__(
        self,
        manifest: WorkerManifest,
        scratch_root: Path | str,
        source_bundle: SourceBundle,
        *,
        popen_factory=subprocess.Popen,
        monotonic=time.monotonic,
    ) -> None:
        self.manifest = manifest
        self.scratch_root = Path(scratch_root).expanduser()
        self.source_bundle = source_bundle
        self._popen_factory = popen_factory
        self._monotonic = monotonic
        self._process: subprocess.Popen[bytes] | None = None
        self._bundle_descriptor = -1
        self._worker_descriptor = -1
        self._python_descriptor = -1
        self._bwrap_descriptor = -1
        self._systemd_run_descriptor = -1
        self._cgroup_evidence: dict[str, object] | None = None
        self._cgroup_observer: CgroupScopeObserver | None = None
        self._launch_started_monotonic: float | None = None
        self._resource_boundaries: list[_ResourceBoundary] = []
        self._completed_cases = 0
        self._stdout_queue: queue.Queue[bytes | None] = queue.Queue(
            maxsize=_QUEUE_CAPACITY
        )
        self._stream_failure = threading.Event()
        self._stdout_bytes = 0
        self._stderr_bytes = 0
        self._stderr_digest = hashlib.sha256()
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._state_lock = threading.Lock()
        self._active = False
        self._closed = False
        self.ready: ReadyEvent | None = None

    @property
    def stderr_summary(self) -> Mapping[str, object]:
        return {
            "bytes": self._stderr_bytes,
            "sha256": self._stderr_digest.hexdigest(),
            "truncated": self._stderr_bytes > MAX_STDERR_BYTES,
        }

    @property
    def cgroup_evidence(self) -> Mapping[str, object] | None:
        return None if self._cgroup_evidence is None else dict(self._cgroup_evidence)

    @property
    def resource_observations(self) -> Mapping[str, object] | None:
        if self._cgroup_evidence is None:
            return None
        if (
            tuple((item.phase, item.boundary) for item in self._resource_boundaries)
            != _RESOURCE_PHASES
        ):
            return None
        return {
            "schema_version": 1,
            "source": "supervisor_cgroup_v2",
            "scope": "systemd_user_scope_all_descendants",
            "os_cache_control": "uncontrolled",
            "cache_state_claim": "none",
            "cache_eviction_performed": False,
            "wall_clock_origin": "supervisor_immediately_before_popen",
            "memory_current_scope": ("point_sample_including_cgroup_charged_cache"),
            "memory_peak_scope": ("cumulative_since_scope_creation_not_phase_local"),
            "cpu_stat_scope": "cumulative_since_scope_creation",
            "snapshot_atomicity": "sequential_cgroup_file_reads_not_atomic",
            "resident_scope": ("pre_shutdown_after_requested_cases_no_idle_settle"),
            "boundaries": [item.as_dict() for item in self._resource_boundaries],
        }

    @property
    def pid(self) -> int | None:
        return None if self._process is None else self._process.pid

    @property
    def fully_stopped(self) -> bool:
        process = self._process
        drainers_stopped = all(
            thread is None or not thread.is_alive()
            for thread in (self._stdout_thread, self._stderr_thread)
        )
        return drainers_stopped and (
            process is None or (process.poll() is not None and not self._group_exists())
        )

    def start(self) -> ReadyEvent:
        if self._process is not None or self._closed:
            raise WorkerError("worker_state")
        if self.manifest.adapter == MOONSHINE_EXTERNAL_ENDPOINT_ADAPTER:
            if self.manifest.schema_version != 7:
                raise WorkerError("worker_protocol")
            _external_moonshine_config(self.manifest)
        elif (
            self.manifest.schema_version == 7
            and self.manifest.adapter != MOONSHINE_EXTERNAL_ENDPOINT_ADAPTER
        ):
            raise WorkerError("worker_protocol")
        if self.manifest.adapter == PARAKEET_CPP_ADAPTER:
            if self.manifest.schema_version != 8 or not isinstance(
                self.manifest.adapter_config, ParakeetCppConfig
            ):
                raise WorkerError("worker_protocol")
        elif self.manifest.schema_version == 8:
            raise WorkerError("worker_protocol")
        if self.manifest.adapter == KYUTAI_ADAPTER:
            config = self.manifest.adapter_config
            if (
                self.manifest.schema_version != 9
                or not isinstance(config, KyutaiConfig)
                or config.minimum_host_available_bytes
                != _KYUTAI_MINIMUM_HOST_AVAILABLE_BYTES
                or config.minimum_free_vram_mb != 8_192
                or config.maximum_vram_fraction != 0.5
                or config.no_torch_compile_env != "1"
                or config.cuda_graph is not False
                or config.no_cuda_graph_env != "1"
            ):
                raise WorkerError("worker_protocol")
        elif self.manifest.schema_version == 9:
            raise WorkerError("worker_protocol")
        if self.manifest.adapter == MOBILE_ZIPFORMER_ADAPTER:
            config = self.manifest.adapter_config
            if (
                self.manifest.schema_version != 10
                or type(config) is not MobileZipformerConfig
                or config.native_chunk_samples != 1_600
                or config.maximum_tail_padding_samples != 48_000
            ):
                raise WorkerError("worker_protocol")
        elif self.manifest.schema_version == 10:
            raise WorkerError("worker_protocol")
        if self.manifest.adapter == MOBILE_WHISPER_ADAPTER:
            config = self.manifest.adapter_config
            if (
                self.manifest.schema_version != 11
                or type(config) is not MobileWhisperConfig
                or config.native_chunk_samples != 1_600
                or config.maximum_tail_padding_samples != 0
                or config.execution_mode != "complete-endpoint-epoch-final-only"
                or config.endpoint_owner != "mobile-zipformer"
                or config.control_authority is not False
            ):
                raise WorkerError("worker_protocol")
        elif self.manifest.schema_version == 11:
            raise WorkerError("worker_protocol")
        try:
            lexical_scratch = Path(os.path.abspath(self.scratch_root))
            with opened_directory_nofollow(
                lexical_scratch,
                require_private=True,
            ) as (scratch, _descriptor):
                pass
        except (OSError, RuntimeError, BoundedReadError):
            raise WorkerError("worker_prerequisite") from None
        _validate_mobile_whisper_scratch_authorities(
            self.manifest,
            scratch,
            self.source_bundle,
        )
        device_nodes: tuple[Path, ...] = ()
        system_ro_paths: tuple[Path, ...] = ()
        proc_driver_available = False
        user_runtime_directory: Path | None = None
        if self.manifest.adapter in {
            FASTER_WHISPER_ENDPOINT_ADAPTER,
            KYUTAI_ADAPTER,
            MOBILE_WHISPER_ADAPTER,
            NEMOTRON_ADAPTER,
            PARAKEET_CPP_ADAPTER,
            PARAKEET_REALTIME_EOU_ADAPTER,
            MOBILE_ZIPFORMER_ADAPTER,
            SHERPA_ZIPFORMER_ADAPTER,
        }:
            if os.name != "posix" or not Path("/proc/self/fd").is_dir():
                raise WorkerError("worker_prerequisite")
        if self.manifest.adapter in {
            MOBILE_WHISPER_ADAPTER,
            MOBILE_ZIPFORMER_ADAPTER,
            SHERPA_ZIPFORMER_ADAPTER,
        }:
            system_ro_paths = _core_system_ro_paths()
        if self.manifest.adapter == PARAKEET_CPP_ADAPTER:
            system_ro_paths = _core_system_ro_paths()
        if self.manifest.adapter in {
            FASTER_WHISPER_ENDPOINT_ADAPTER,
            KYUTAI_ADAPTER,
            NEMOTRON_ADAPTER,
            PARAKEET_REALTIME_EOU_ADAPTER,
        }:
            device_nodes = _nemotron_device_nodes()
            system_ro_paths = _nemotron_system_ro_paths()
            try:
                _NVIDIA_PROC_DRIVER.lstat()
            except FileNotFoundError:
                pass
            except OSError:
                raise WorkerError("worker_prerequisite") from None
            else:
                _require_mount_source(_NVIDIA_PROC_DRIVER, directory=True)
                proc_driver_available = True
        if self.manifest.adapter in {
            KYUTAI_ADAPTER,
            PARAKEET_CPP_ADAPTER,
            PARAKEET_REALTIME_EOU_ADAPTER,
        }:
            user_runtime_directory = _verified_user_runtime_directory()
        if self.manifest.adapter == KYUTAI_ADAPTER:
            _verify_kyutai_host_memory_available(
                minimum_bytes=_KYUTAI_MINIMUM_HOST_AVAILABLE_BYTES,
            )
        self._verify_bound_files()
        if os.name != "posix" or not Path("/proc/self/fd").is_dir():
            raise WorkerError("worker_prerequisite")
        try:
            with opened_directory_nofollow(
                self.source_bundle.root,
                require_private=True,
            ) as (_bundle_root, descriptor):
                self._bundle_descriptor = os.dup(descriptor)
            self._worker_descriptor = self._open_staged_worker_descriptor()
            self._python_descriptor = self._open_bound_python_descriptor()
            if self.manifest.adapter in {
                FASTER_WHISPER_ENDPOINT_ADAPTER,
                KYUTAI_ADAPTER,
                NEMOTRON_ADAPTER,
                PARAKEET_CPP_ADAPTER,
                PARAKEET_REALTIME_EOU_ADAPTER,
                MOBILE_WHISPER_ADAPTER,
                MOBILE_ZIPFORMER_ADAPTER,
                SHERPA_ZIPFORMER_ADAPTER,
            }:
                self._bwrap_descriptor = _open_bwrap_descriptor()
            if self.manifest.adapter in {
                KYUTAI_ADAPTER,
                PARAKEET_CPP_ADAPTER,
                PARAKEET_REALTIME_EOU_ADAPTER,
            }:
                self._systemd_run_descriptor = _open_systemd_run_descriptor()
        except (OSError, BoundedReadError, WorkerError):
            self._close_bundle_descriptors()
            raise WorkerError("worker_prerequisite") from None
        bundle_root = Path(f"/proc/self/fd/{self._bundle_descriptor}")
        interpreter_flags = _worker_interpreter_flags(self.manifest)
        worker_command = [
            str(self.manifest.python.path),
            *interpreter_flags,
            f"/proc/self/fd/{self._worker_descriptor}",
            "--manifest",
            str(self.manifest.path),
            "--scratch-root",
            str(scratch),
            "--source-bundle-manifest",
            str(bundle_root / self.source_bundle.manifest_path.name),
            "--source-bundle-sha256",
            self.source_bundle.tree_sha256,
            "--source-bundle-fd",
            str(self._bundle_descriptor),
        ]
        command = worker_command
        executable = f"/proc/self/fd/{self._python_descriptor}"
        pass_fds = (
            self._bundle_descriptor,
            self._worker_descriptor,
            self._python_descriptor,
        )
        start_new_session = True
        if self.manifest.adapter == PARAKEET_CPP_ADAPTER:
            try:
                command = _parakeet_cpp_bwrap_command(
                    self.manifest,
                    scratch,
                    self.source_bundle,
                    bundle_descriptor=self._bundle_descriptor,
                    worker_descriptor=self._worker_descriptor,
                    python_descriptor=self._python_descriptor,
                    system_ro_paths=system_ro_paths,
                )
            except WorkerError:
                self._close_bundle_descriptors()
                raise
            executable = f"/proc/self/fd/{self._bwrap_descriptor}"
            pass_fds = (*pass_fds, self._bwrap_descriptor)
            command = _parakeet_cpp_systemd_scope_command(
                command,
                bwrap_descriptor=self._bwrap_descriptor,
            )
            executable = f"/proc/self/fd/{self._systemd_run_descriptor}"
            pass_fds = (*pass_fds, self._systemd_run_descriptor)
        elif self.manifest.adapter in {
            FASTER_WHISPER_ENDPOINT_ADAPTER,
            KYUTAI_ADAPTER,
            NEMOTRON_ADAPTER,
            PARAKEET_REALTIME_EOU_ADAPTER,
        }:
            try:
                if self.manifest.adapter == FASTER_WHISPER_ENDPOINT_ADAPTER:
                    builder = _faster_whisper_bwrap_command
                elif self.manifest.adapter == KYUTAI_ADAPTER:
                    builder = _kyutai_bwrap_command
                else:
                    builder = _nemotron_bwrap_command
                command = builder(
                    self.manifest,
                    scratch,
                    self.source_bundle,
                    bundle_descriptor=self._bundle_descriptor,
                    worker_descriptor=self._worker_descriptor,
                    python_descriptor=self._python_descriptor,
                    device_nodes=device_nodes,
                    system_ro_paths=system_ro_paths,
                    proc_driver_available=proc_driver_available,
                )
            except WorkerError:
                self._close_bundle_descriptors()
                raise
            executable = f"/proc/self/fd/{self._bwrap_descriptor}"
            pass_fds = (*pass_fds, self._bwrap_descriptor)
            if self.manifest.adapter in {
                KYUTAI_ADAPTER,
                PARAKEET_REALTIME_EOU_ADAPTER,
            }:
                scope_builder = (
                    _kyutai_systemd_scope_command
                    if self.manifest.adapter == KYUTAI_ADAPTER
                    else _parakeet_systemd_scope_command
                )
                command = scope_builder(
                    command, bwrap_descriptor=self._bwrap_descriptor
                )
                executable = f"/proc/self/fd/{self._systemd_run_descriptor}"
                pass_fds = (*pass_fds, self._systemd_run_descriptor)
            # Popen's session owns the outer Bubblewrap monitor as a bounded
            # process group. Bubblewrap forks its namespace child before the
            # required --new-session setsid(), so the inner session remains
            # independent and die-with-parent links both lifecycle layers.
        elif self.manifest.adapter in {
            MOBILE_WHISPER_ADAPTER,
            MOBILE_ZIPFORMER_ADAPTER,
            SHERPA_ZIPFORMER_ADAPTER,
        }:
            try:
                command = _zipformer_bwrap_command(
                    self.manifest,
                    scratch,
                    self.source_bundle,
                    bundle_descriptor=self._bundle_descriptor,
                    worker_descriptor=self._worker_descriptor,
                    python_descriptor=self._python_descriptor,
                    system_ro_paths=system_ro_paths,
                )
            except WorkerError:
                self._close_bundle_descriptors()
                raise
            executable = f"/proc/self/fd/{self._bwrap_descriptor}"
            pass_fds = (*pass_fds, self._bwrap_descriptor)
        environment = sanitized_worker_environment(scratch)
        if self.manifest.adapter in {
            KYUTAI_ADAPTER,
            PARAKEET_CPP_ADAPTER,
            PARAKEET_REALTIME_EOU_ADAPTER,
        }:
            if user_runtime_directory is None:
                self._close_bundle_descriptors()
                raise WorkerError("worker_prerequisite")
            environment["XDG_RUNTIME_DIR"] = str(user_runtime_directory)
        if self.manifest.adapter == PARAKEET_CPP_ADAPTER:
            environment["PARAKEET_DEVICE"] = "cpu"
            environment["CUDA_VISIBLE_DEVICES"] = ""
            environment.pop("CUDA_DEVICE_ORDER", None)
        if self.manifest.adapter in {
            MOBILE_WHISPER_ADAPTER,
            MOBILE_ZIPFORMER_ADAPTER,
        }:
            environment["CUDA_VISIBLE_DEVICES"] = ""
            environment.pop("CUDA_DEVICE_ORDER", None)
        if self.manifest.adapter == PARAKEET_REALTIME_EOU_ADAPTER:
            # NeMo imports lhotse, whose environment probe requires the normal
            # host binary directory even though the worker has no network and
            # executes only receipt-bound Python/source artifacts.
            environment["PATH"] = "/usr/bin"
        if self.manifest.adapter == KYUTAI_ADAPTER:
            config = self.manifest.adapter_config
            if not isinstance(config, KyutaiConfig):
                self._close_bundle_descriptors()
                raise WorkerError("worker_protocol")
            environment["NO_TORCH_COMPILE"] = config.no_torch_compile_env
            environment["NO_CUDA_GRAPH"] = config.no_cuda_graph_env
        try:
            launch_started = self._monotonic()
        except Exception:
            self._close_bundle_descriptors()
            raise WorkerError("worker_start") from None
        if not math.isfinite(launch_started):
            self._close_bundle_descriptors()
            raise WorkerError("worker_start")
        self._launch_started_monotonic = launch_started
        try:
            process = self._popen_factory(
                command,
                cwd=str(scratch),
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=start_new_session,
                close_fds=True,
                executable=executable,
                pass_fds=pass_fds,
            )
        except Exception:
            self._close_bundle_descriptors()
            raise WorkerError("worker_start") from None
        self._process = process
        try:
            try:
                self._stdout_thread = threading.Thread(
                    target=self._drain_stdout,
                    name="streaming-stt-stdout",
                    daemon=True,
                )
                self._stdout_thread.start()
                self._stderr_thread = threading.Thread(
                    target=self._drain_stderr,
                    name="streaming-stt-stderr",
                    daemon=True,
                )
                self._stderr_thread.start()
            except Exception:
                raise WorkerError("worker_start") from None
            if self.manifest.adapter == PARAKEET_REALTIME_EOU_ADAPTER:
                awaited = _await_parakeet_cgroup(
                    process,
                    monotonic=self._monotonic,
                )
                evidence, observer = _bind_cgroup_observer(
                    process.pid,
                    cpp=False,
                )
                if evidence != awaited:
                    observer.close()
                    raise WorkerError("worker_prerequisite")
                self._cgroup_evidence = evidence
                self._cgroup_observer = observer
            if self.manifest.adapter == PARAKEET_CPP_ADAPTER:
                awaited = _await_parakeet_cpp_cgroup(
                    process,
                    monotonic=self._monotonic,
                )
                evidence, observer = _bind_cgroup_observer(
                    process.pid,
                    cpp=True,
                )
                if evidence != awaited:
                    observer.close()
                    raise WorkerError("worker_prerequisite")
                self._cgroup_evidence = evidence
                self._cgroup_observer = observer
            if self.manifest.adapter == KYUTAI_ADAPTER:
                awaited = _await_kyutai_cgroup(
                    process,
                    monotonic=self._monotonic,
                )
                evidence, observer = _bind_cgroup_observer(
                    process.pid,
                    cpp=False,
                    kyutai=True,
                )
                if evidence != awaited:
                    observer.close()
                    raise WorkerError("worker_prerequisite")
                self._cgroup_evidence = evidence
                self._cgroup_observer = observer
            event = self._next_event(self.manifest.limits.startup_timeout_sec)
            if (
                type(event) is not ReadyEvent
                or event.protocol_version != _wire_protocol_version(self.manifest)
                or event.model_id != self.manifest.model_id
                or event.manifest_sha256 != self.manifest.digest
                or event.source_bundle_sha256 != self.source_bundle.tree_sha256
                or event.adapter != self.manifest.adapter
            ):
                raise WorkerError("worker_protocol")
            self.ready = event
            if self._cgroup_observer is not None:
                self._record_resource_boundary(
                    "startup",
                    "validated_ready",
                    completed_cases=0,
                )
            return event
        except Exception:
            try:
                self._terminate()
            finally:
                self._close_cgroup_observer()
                self._close_process_pipes()
                self._close_bundle_descriptors()
            raise

    def _record_resource_boundary(
        self,
        phase: str,
        boundary: str,
        *,
        completed_cases: int,
    ) -> None:
        observer = self._cgroup_observer
        process = self._process
        launch_started = self._launch_started_monotonic
        index = len(self._resource_boundaries)
        if (
            observer is None
            or process is None
            or process.poll() is not None
            or launch_started is None
            or index >= len(_RESOURCE_PHASES)
            or (phase, boundary) != _RESOURCE_PHASES[index]
            or type(completed_cases) is not int
            or completed_cases < 0
            or (phase == "startup" and completed_cases != 0)
            or (phase == "first_use" and completed_cases != 1)
            or (phase == "resident" and completed_cases != self._completed_cases)
        ):
            raise WorkerError("worker_prerequisite")
        try:
            if self.manifest.adapter == PARAKEET_CPP_ADAPTER:
                current_evidence = _read_parakeet_cpp_cgroup_evidence(process.pid)
            elif self.manifest.adapter == KYUTAI_ADAPTER:
                current_evidence = _read_kyutai_cgroup_evidence(process.pid)
            else:
                current_evidence = _read_parakeet_cgroup_evidence(process.pid)
            observed_at = self._monotonic()
            if (
                current_evidence != self._cgroup_evidence
                or not math.isfinite(observed_at)
                or observed_at < launch_started
            ):
                raise WorkerError("worker_prerequisite")
            elapsed_ms = (observed_at - launch_started) * 1000.0
            if (
                self._resource_boundaries
                and elapsed_ms < self._resource_boundaries[-1].wall_elapsed_ms
            ):
                raise WorkerError("worker_prerequisite")
            snapshot = observer.sample()
        except CgroupObservationError:
            raise WorkerError("worker_prerequisite") from None
        except WorkerError:
            raise
        except (OSError, RuntimeError, ValueError, OverflowError):
            raise WorkerError("worker_prerequisite") from None
        self._resource_boundaries.append(
            _ResourceBoundary(
                phase=phase,
                boundary=boundary,
                completed_cases=completed_cases,
                wall_elapsed_ms=elapsed_ms,
                snapshot=snapshot,
            )
        )

    def _complete_trace(self, trace: CaseTrace) -> CaseTrace:
        completed_cases = self._completed_cases + 1
        if self._cgroup_observer is not None and completed_cases == 1:
            self._record_resource_boundary(
                "first_use",
                "first_validated_terminal",
                completed_cases=completed_cases,
            )
        self._completed_cases = completed_cases
        return trace

    def _close_cgroup_observer(self) -> None:
        observer = self._cgroup_observer
        self._cgroup_observer = None
        if observer is not None:
            try:
                observer.close()
            except OSError:
                pass

    def transcribe(self, request: TranscribeRequest) -> CaseTrace:
        with self._state_lock:
            if self.ready is None or self._closed or self._active:
                raise WorkerError("worker_state")
            self._active = True
        try:
            # Round-trip through the parser so programmatic callers cannot
            # bypass the same strict schema enforced at the worker boundary.
            try:
                encoded = encode_message(request.as_dict())
                checked = parse_request(encoded)
            except ProtocolError:
                raise WorkerError("invalid_request") from None
            if not isinstance(checked, TranscribeRequest):
                raise WorkerError("invalid_request")
            expected_protocol = _wire_protocol_version(self.manifest)
            if checked.protocol_version != expected_protocol:
                raise WorkerError("invalid_request")
            final_only = self.manifest.adapter in {
                FASTER_WHISPER_ENDPOINT_ADAPTER,
                MOBILE_WHISPER_ADAPTER,
            }
            if final_only != (self.manifest.schema_version in {6, 11}):
                raise WorkerError("worker_protocol")
            if (
                self.manifest.schema_version == 6
                and self.manifest.adapter != FASTER_WHISPER_ENDPOINT_ADAPTER
            ) or (
                self.manifest.schema_version == 11
                and self.manifest.adapter != MOBILE_WHISPER_ADAPTER
            ):
                raise WorkerError("worker_protocol")
            if self.manifest.adapter == MOONSHINE_EXTERNAL_ENDPOINT_ADAPTER:
                _expected_final_evidence(self.manifest, checked)
            if self.manifest.adapter == MOBILE_ZIPFORMER_ADAPTER:
                config = self.manifest.adapter_config
                if (
                    not isinstance(config, MobileZipformerConfig)
                    or checked.stream.chunk_samples != config.native_chunk_samples
                    or checked.stream.tail_padding_samples
                    > config.maximum_tail_padding_samples
                ):
                    raise WorkerError("invalid_request")
            if self.manifest.adapter == MOBILE_WHISPER_ADAPTER:
                config = self.manifest.adapter_config
                if (
                    not isinstance(config, MobileWhisperConfig)
                    or checked.stream.chunk_samples != config.native_chunk_samples
                    or checked.stream.tail_padding_samples
                    != config.maximum_tail_padding_samples
                    or checked.stream.pace != "burst"
                ):
                    raise WorkerError("invalid_request")
            self._verify_pcm(checked)
            self._send(encoded)
            partials: list[PartialEvent] = []
            last_samples = -1
            last_elapsed = -1.0
            case_deadline = self._monotonic() + self.manifest.limits.case_timeout_sec
            while True:
                event = self._next_event(max(0.0, case_deadline - self._monotonic()))
                if type(event) is PartialEvent:
                    if (
                        final_only
                        or event.protocol_version != expected_protocol
                        or event.request_id != checked.request_id
                        or event.seq != len(partials)
                        or event.samples_seen < last_samples
                        or (
                            self.manifest.adapter == MOBILE_ZIPFORMER_ADAPTER
                            and (
                                event.samples_seen <= last_samples
                                or (partials and event.text == partials[-1].text)
                            )
                        )
                        or event.samples_seen
                        > checked.pcm.samples + checked.stream.tail_padding_samples
                        or event.elapsed_ms < last_elapsed
                        or len(partials) >= MAX_PARTIALS_PER_CASE
                    ):
                        raise WorkerError("worker_protocol")
                    partials.append(event)
                    last_samples = event.samples_seen
                    last_elapsed = event.elapsed_ms
                    continue
                if type(event) is ErrorEvent:
                    if (
                        event.protocol_version != expected_protocol
                        or event.request_id != checked.request_id
                    ):
                        raise WorkerError("worker_protocol")
                    if event.fatal:
                        self._terminate()
                    raise WorkerError("worker_case")
                if self.manifest.adapter == PARAKEET_REALTIME_EOU_ADAPTER:
                    if type(event) is not NativeFinalEvent:
                        raise WorkerError("worker_protocol")
                    _validate_native_final(
                        self.manifest,
                        checked,
                        event,
                        expected_seq=len(partials),
                        last_samples=last_samples,
                        last_elapsed=last_elapsed,
                    )
                    return self._complete_trace(
                        CaseTrace(partials=tuple(partials), final=event)
                    )
                if self.manifest.adapter == PARAKEET_CPP_ADAPTER:
                    if type(event) is not ParakeetCppFinalEvent:
                        raise WorkerError("worker_protocol")
                    _validate_parakeet_cpp_final(
                        self.manifest,
                        checked,
                        event,
                        expected_seq=len(partials),
                        last_samples=last_samples,
                        last_elapsed=last_elapsed,
                    )
                    return self._complete_trace(
                        CaseTrace(partials=tuple(partials), final=event)
                    )
                if self.manifest.adapter == MOBILE_ZIPFORMER_ADAPTER:
                    if type(event) is not MobileZipformerFinalEvent:
                        raise WorkerError("worker_protocol")
                    _validate_mobile_zipformer_final(
                        self.manifest,
                        checked,
                        event,
                        expected_seq=len(partials),
                        last_samples=last_samples,
                        last_elapsed=last_elapsed,
                    )
                    return self._complete_trace(
                        CaseTrace(partials=tuple(partials), final=event)
                    )
                if type(event) is not FinalEvent:
                    raise WorkerError("worker_protocol")
                expected_chunks, expected_model_padding = _expected_final_evidence(
                    self.manifest,
                    checked,
                )
                if (
                    event.protocol_version != PROTOCOL_VERSION
                    or event.request_id != checked.request_id
                    or event.seq != len(partials)
                    or (final_only and event.seq != 0)
                    or event.samples_seen
                    != checked.pcm.samples + checked.stream.tail_padding_samples
                    or event.samples_seen < last_samples
                    or event.elapsed_ms < last_elapsed
                    or event.chunks != expected_chunks
                    or (
                        expected_model_padding is not None
                        and event.model_padding_samples != expected_model_padding
                    )
                    or abs(
                        event.audio_seconds
                        - checked.pcm.samples / checked.pcm.sample_rate
                    )
                    > 1e-6
                ):
                    raise WorkerError("worker_protocol")
                return self._complete_trace(
                    CaseTrace(partials=tuple(partials), final=event)
                )
        except WorkerError:
            if self._process is not None and self._process.poll() is None:
                # A protocol/time failure makes the stream untrustworthy.
                self._terminate()
            raise
        finally:
            with self._state_lock:
                self._active = False

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            active = self._active
        process = self._process
        if process is None:
            self._close_cgroup_observer()
            self._close_bundle_descriptors()
            return
        try:
            if active:
                self._terminate()
                self._verify_bound_files()
                return
            if process.poll() is None and self.ready is not None:
                try:
                    if self._cgroup_observer is not None and self._completed_cases > 0:
                        self._record_resource_boundary(
                            "resident",
                            "pre_shutdown_after_suite",
                            completed_cases=self._completed_cases,
                        )
                    expected_protocol = _wire_protocol_version(self.manifest)
                    request = ShutdownRequest(
                        "shutdown",
                        protocol_version=expected_protocol,
                    )
                    self._send(encode_message(request.as_dict()))
                    acknowledgement = self._next_event(_TERM_GRACE_SEC)
                    goodbye = self._next_event(_TERM_GRACE_SEC)
                    if (
                        type(acknowledgement) is not ShutdownEvent
                        or acknowledgement.protocol_version != expected_protocol
                        or acknowledgement.request_id != request.request_id
                        or type(goodbye) is not ByeEvent
                        or goodbye.protocol_version != expected_protocol
                    ):
                        raise WorkerError("worker_protocol")
                    process.wait(timeout=_KILL_GRACE_SEC)
                except Exception:
                    self._terminate()
            elif process.poll() is None:
                self._terminate()
            if process.poll() is None or self._group_exists():
                self._terminate()
            else:
                self._join_drainers(require_stopped=True)
            self._verify_bound_files()
        finally:
            self._close_cgroup_observer()
            self._close_process_pipes()
            self._close_bundle_descriptors()

    def __enter__(self) -> StreamingWorker:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _verify_pcm(self, request: TranscribeRequest) -> None:
        candidate = request.pcm.path
        try:
            digest = hash_regular_bounded(
                candidate,
                maximum_bytes=MAX_PCM_BYTES,
                expected_bytes=request.pcm.size_bytes,
            )
            digest.path.relative_to(self.scratch_root.resolve(strict=True))
        except (OSError, RuntimeError, ValueError, BoundedReadError):
            raise WorkerError("worker_prerequisite") from None
        if digest.sha256 != request.pcm.sha256:
            raise WorkerError("worker_prerequisite")

    def _verify_bound_files(self) -> None:
        try:
            worker = hash_regular_bounded(
                self.manifest.worker.path,
                maximum_bytes=MAX_WORKER_BYTES,
                expected_bytes=self.manifest.worker.size_bytes,
            )
            python = hash_regular_bounded(
                self.manifest.python.path,
                maximum_bytes=MAX_PYTHON_BYTES,
                expected_bytes=self.manifest.python.size_bytes,
                allow_final_symlink=True,
            )
            if (
                worker.sha256 != self.manifest.worker.sha256
                or python.sha256 != self.manifest.python.sha256
                or any(
                    hash_regular_bounded(
                        artifact.path,
                        maximum_bytes=artifact_maximum_bytes(
                            self.manifest.adapter,
                            artifact.name,
                        ),
                        expected_bytes=artifact.size_bytes,
                    ).sha256
                    != artifact.sha256
                    for artifact in self.manifest.artifacts
                )
                or any(
                    hash_regular_bounded(
                        control.path,
                        maximum_bytes=(
                            MAX_MOBILE_WHISPER_SOURCE_LOCK_BYTES
                            if self.manifest.adapter == MOBILE_WHISPER_ADAPTER
                            else MAX_MOBILE_ZIPFORMER_SOURCE_LOCK_BYTES
                        ),
                        expected_bytes=control.size_bytes,
                    ).sha256
                    != control.sha256
                    for control in self.manifest.control_files
                )
            ):
                raise WorkerError("worker_artifact_changed")
            if self.manifest.adapter in {
                FASTER_WHISPER_ENDPOINT_ADAPTER,
                KYUTAI_ADAPTER,
                MOONSHINE_ADAPTER,
                MOONSHINE_EXTERNAL_ENDPOINT_ADAPTER,
                NEMOTRON_ADAPTER,
                PARAKEET_REALTIME_EOU_ADAPTER,
            }:
                receipt_artifact = self.manifest.artifact_by_name.get("runtime-receipt")
                if receipt_artifact is None:
                    raise WorkerError("worker_artifact_changed")
                limits = {
                    MOONSHINE_ADAPTER: None,
                    MOONSHINE_EXTERNAL_ENDPOINT_ADAPTER: None,
                    KYUTAI_ADAPTER: KYUTAI_RUNTIME_TREE_LIMITS,
                    NEMOTRON_ADAPTER: NEMOTRON_RUNTIME_TREE_LIMITS,
                    PARAKEET_REALTIME_EOU_ADAPTER: PARAKEET_RUNTIME_TREE_LIMITS,
                    FASTER_WHISPER_ENDPOINT_ADAPTER: (
                        FASTER_WHISPER_RUNTIME_TREE_LIMITS
                    ),
                }[self.manifest.adapter]
                receipt = load_runtime_tree_receipt(
                    receipt_artifact.path,
                    expected_digest=receipt_artifact.sha256,
                    **({} if limits is None else {"limits": limits}),
                )
                verify_venv_runtime_location(
                    receipt,
                    self.manifest.python.path,
                )
                if self.manifest.adapter in {
                    KYUTAI_ADAPTER,
                    NEMOTRON_ADAPTER,
                    PARAKEET_REALTIME_EOU_ADAPTER,
                }:
                    _verify_nemotron_runtime_layout(self.manifest, receipt)
                if self.manifest.adapter == FASTER_WHISPER_ENDPOINT_ADAPTER:
                    _verify_faster_whisper_runtime_layout(self.manifest, receipt)
                if self.manifest.adapter in {
                    MOONSHINE_ADAPTER,
                    MOONSHINE_EXTERNAL_ENDPOINT_ADAPTER,
                }:
                    wheel_artifact = self.manifest.artifact_by_name.get("release-wheel")
                    if wheel_artifact is None:
                        raise WorkerError("worker_artifact_changed")
                    verify_moonshine_wheel_install(
                        receipt,
                        wheel_artifact.path,
                        expected_sha256=wheel_artifact.sha256,
                        expected_size_bytes=wheel_artifact.size_bytes,
                    )
            verify_source_bundle(self.source_bundle)
            staged_worker = self.source_bundle.file_by_path.get(
                self.source_bundle.worker_relative_path
            )
            if (
                staged_worker is None
                or staged_worker.sha256 != self.manifest.worker.sha256
                or staged_worker.size_bytes != self.manifest.worker.size_bytes
            ):
                raise WorkerError("worker_artifact_changed")
        except (OSError, RuntimeError, BoundedReadError, SourceBundleError):
            raise WorkerError("worker_artifact_changed") from None

    def _open_staged_worker_descriptor(self) -> int:
        root_descriptor = self._bundle_descriptor
        if root_descriptor < 0:
            raise WorkerError("worker_prerequisite")
        directory_descriptor = os.dup(root_descriptor)
        worker_descriptor = -1
        try:
            directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            directory_flags |= getattr(os, "O_DIRECTORY", 0)
            directory_flags |= getattr(os, "O_NOFOLLOW", 0)
            parts = self.source_bundle.worker_relative_path.split("/")
            for part in parts[:-1]:
                child = os.open(
                    part,
                    directory_flags,
                    dir_fd=directory_descriptor,
                )
                os.close(directory_descriptor)
                directory_descriptor = child
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            worker_descriptor = os.open(
                parts[-1],
                flags,
                dir_fd=directory_descriptor,
            )
            metadata = os.fstat(worker_descriptor)
            expected = self.source_bundle.file_by_path.get(
                self.source_bundle.worker_relative_path
            )
            if (
                expected is None
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_size != expected.size_bytes
            ):
                raise WorkerError("worker_prerequisite")
            digest = hashlib.sha256()
            consumed = 0
            while consumed <= expected.size_bytes:
                chunk = os.read(
                    worker_descriptor,
                    min(1024 * 1024, expected.size_bytes + 1 - consumed),
                )
                if not chunk:
                    break
                consumed += len(chunk)
                digest.update(chunk)
            after = os.fstat(worker_descriptor)
            root_metadata = os.fstat(root_descriptor)
            current_root = self.source_bundle.root.lstat()
            if (
                consumed != expected.size_bytes
                or digest.hexdigest() != expected.sha256
                or (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                    after.st_nlink,
                )
                != (
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_size,
                    metadata.st_mtime_ns,
                    metadata.st_ctime_ns,
                    1,
                )
                or (
                    current_root.st_dev,
                    current_root.st_ino,
                    stat.S_IFMT(current_root.st_mode),
                )
                != (
                    root_metadata.st_dev,
                    root_metadata.st_ino,
                    stat.S_IFMT(root_metadata.st_mode),
                )
            ):
                raise WorkerError("worker_prerequisite")
            os.lseek(worker_descriptor, 0, os.SEEK_SET)
            result = worker_descriptor
            worker_descriptor = -1
            return result
        finally:
            os.close(directory_descriptor)
            if worker_descriptor >= 0:
                os.close(worker_descriptor)

    def _open_bound_python_descriptor(self) -> int:
        descriptor = -1
        try:
            resolved = self.manifest.python.path.resolve(strict=True)
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(resolved, flags)
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_size != self.manifest.python.size_bytes
            ):
                raise WorkerError("worker_prerequisite")
            digest = hashlib.sha256()
            consumed = 0
            while consumed <= self.manifest.python.size_bytes:
                chunk = os.read(
                    descriptor,
                    min(
                        1024 * 1024,
                        self.manifest.python.size_bytes + 1 - consumed,
                    ),
                )
                if not chunk:
                    break
                consumed += len(chunk)
                digest.update(chunk)
            after = os.fstat(descriptor)
            if (
                consumed != self.manifest.python.size_bytes
                or digest.hexdigest() != self.manifest.python.sha256
                or (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                    after.st_nlink,
                )
                != (
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_size,
                    metadata.st_mtime_ns,
                    metadata.st_ctime_ns,
                    1,
                )
            ):
                raise WorkerError("worker_prerequisite")
            os.lseek(descriptor, 0, os.SEEK_SET)
            result = descriptor
            descriptor = -1
            return result
        except (OSError, RuntimeError, ValueError, OverflowError):
            raise WorkerError("worker_prerequisite") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def _close_bundle_descriptors(self) -> None:
        systemd_run_descriptor = self._systemd_run_descriptor
        self._systemd_run_descriptor = -1
        if systemd_run_descriptor >= 0:
            try:
                os.close(systemd_run_descriptor)
            except OSError:
                pass
        bwrap_descriptor = self._bwrap_descriptor
        self._bwrap_descriptor = -1
        if bwrap_descriptor >= 0:
            try:
                os.close(bwrap_descriptor)
            except OSError:
                pass
        python_descriptor = self._python_descriptor
        self._python_descriptor = -1
        if python_descriptor >= 0:
            try:
                os.close(python_descriptor)
            except OSError:
                pass
        worker_descriptor = self._worker_descriptor
        self._worker_descriptor = -1
        if worker_descriptor >= 0:
            try:
                os.close(worker_descriptor)
            except OSError:
                pass
        descriptor = self._bundle_descriptor
        self._bundle_descriptor = -1
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass

    def _close_process_pipes(self) -> None:
        process = self._process
        if process is None:
            return
        streams = (
            (process.stdin, None),
            (process.stdout, self._stdout_thread),
            (process.stderr, self._stderr_thread),
        )
        for stream, drainer in streams:
            if stream is not None:
                if (
                    drainer is not None
                    and drainer.ident is not None
                    and drainer.is_alive()
                ):
                    continue
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass

    def _send(self, encoded: bytes) -> None:
        process = self._process
        if process is None or process.stdin is None or process.poll() is not None:
            raise WorkerError("worker_dead")
        try:
            process.stdin.write(encoded)
            process.stdin.flush()
        except (BrokenPipeError, OSError):
            raise WorkerError("worker_dead") from None

    def _next_event(self, timeout: float) -> Response:
        deadline = self._monotonic() + timeout
        while True:
            if self._stream_failure.is_set():
                raise WorkerError("worker_output_limit")
            remaining = deadline - self._monotonic()
            if remaining <= 0.0:
                if self._stream_failure.is_set():
                    raise WorkerError("worker_output_limit")
                raise WorkerError("worker_timeout")
            try:
                raw = self._stdout_queue.get(timeout=min(0.05, remaining))
            except queue.Empty:
                if self._stream_failure.is_set():
                    raise WorkerError("worker_output_limit")
                process = self._process
                if process is not None and process.poll() is not None:
                    raise WorkerError("worker_dead")
                continue
            if raw is None:
                if self._stream_failure.is_set():
                    raise WorkerError("worker_output_limit")
                raise WorkerError("worker_dead")
            try:
                return parse_response(raw)
            except ProtocolError:
                raise WorkerError("worker_protocol") from None

    def _drain_stdout(self) -> None:
        process = self._process
        assert process is not None and process.stdout is not None
        try:
            while True:
                raw = process.stdout.readline(MAX_LINE_BYTES + 1)
                if not raw:
                    break
                self._stdout_bytes += len(raw)
                if (
                    len(raw) > MAX_LINE_BYTES
                    or not raw.endswith(b"\n")
                    or self._stdout_bytes > MAX_STDOUT_BYTES
                ):
                    self._stream_failure.set()
                    break
                try:
                    self._stdout_queue.put_nowait(raw)
                except queue.Full:
                    self._stream_failure.set()
                    break
        except Exception:
            self._stream_failure.set()
        finally:
            try:
                self._stdout_queue.put_nowait(None)
            except queue.Full:
                self._stream_failure.set()

    def _drain_stderr(self) -> None:
        process = self._process
        assert process is not None and process.stderr is not None
        read_chunk = getattr(process.stderr, "read1", process.stderr.read)
        try:
            while True:
                remaining = MAX_STDERR_BYTES + 1 - self._stderr_bytes
                if remaining <= 0:
                    break
                chunk = read_chunk(min(4096, remaining))
                if not chunk:
                    break
                self._stderr_bytes += len(chunk)
                self._stderr_digest.update(chunk)
                if self._stderr_bytes > MAX_STDERR_BYTES:
                    self._stream_failure.set()
                    break
        except Exception:
            self._stream_failure.set()

    def _terminate(self) -> None:
        process = self._process
        if process is None:
            self._join_drainers(require_stopped=True)
            return

        term_deadline = self._monotonic() + _TERM_GRACE_SEC
        if process.poll() is None or self._group_exists():
            self._signal_group(signal.SIGTERM)
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.0, term_deadline - self._monotonic()))
            except subprocess.TimeoutExpired:
                pass

        while self._group_exists() and self._monotonic() < term_deadline:
            time.sleep(0.01)

        if process.poll() is None or self._group_exists():
            self._signal_group(signal.SIGKILL)
        if process.poll() is None:
            try:
                process.wait(timeout=_KILL_GRACE_SEC)
            except subprocess.TimeoutExpired:
                raise WorkerError("worker_unstoppable") from None
        if self._group_exists():
            self._wait_group_gone(_KILL_GRACE_SEC)
        if process.poll() is None:
            raise WorkerError("worker_unstoppable")
        # Descendants may inherit the pipe file descriptors. Do not join these
        # readers until the original worker process group is gone.
        self._join_drainers(require_stopped=True)

    def _signal_group(self, signum: int) -> None:
        process = self._process
        if process is None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signum)
            elif signum == signal.SIGKILL:
                process.kill()
            else:
                process.terminate()
        except ProcessLookupError:
            # Before Bubblewrap has completed its required --new-session
            # setsid(), its PID is not yet a process-group ID. Signal that
            # short pre-boundary process directly instead of waiting for a
            # group which cannot exist yet.
            if process.poll() is None:
                try:
                    process.send_signal(signum)
                except ProcessLookupError:
                    pass

    def _group_exists(self) -> bool:
        process = self._process
        if process is None or os.name != "posix":
            return False
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _wait_group_gone(self, timeout: float) -> None:
        deadline = self._monotonic() + timeout
        while self._group_exists() and self._monotonic() < deadline:
            time.sleep(0.01)
        if self._group_exists():
            raise WorkerError("worker_unstoppable")

    def _join_drainers(self, *, require_stopped: bool = False) -> None:
        for thread in (self._stdout_thread, self._stderr_thread):
            if (
                thread is not None
                and thread.ident is not None
                and thread is not threading.current_thread()
            ):
                thread.join(timeout=1.0)
                if require_stopped and thread.is_alive():
                    raise WorkerError("worker_unstoppable")
