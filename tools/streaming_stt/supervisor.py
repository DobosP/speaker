"""Bounded process supervision for one isolated streaming recognizer."""

from __future__ import annotations

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
from .manifest import (
    FASTER_WHISPER_ENDPOINT_ADAPTER,
    FASTER_WHISPER_REQUIRED_MODEL_FILES,
    FASTER_WHISPER_VOCABULARY_FILES,
    MAX_PYTHON_BYTES,
    MAX_WORKER_BYTES,
    MOONSHINE_ADAPTER,
    MOONSHINE_EXTERNAL_ENDPOINT_ADAPTER,
    NEMOTRON_ADAPTER,
    PARAKEET_REALTIME_EOU_ADAPTER,
    SHERPA_ZIPFORMER_ADAPTER,
    FasterWhisperEndpointConfig,
    MoonshineExternalEndpointConfig,
    NemotronConfig,
    ParakeetRealtimeEouConfig,
    WorkerManifest,
    artifact_maximum_bytes,
)
from .protocol import (
    ByeEvent,
    CaseTrace,
    ErrorEvent,
    FinalEvent,
    MAX_LINE_BYTES,
    MAX_PARTIALS_PER_CASE,
    MAX_PCM_BYTES,
    MAX_STDERR_BYTES,
    MAX_STDOUT_BYTES,
    NATIVE_ENDPOINT_PROTOCOL_VERSION,
    NativeFinalEvent,
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
_USER_RUNTIME_ROOT = Path("/run/user")
_CGROUP_ROOT = Path("/sys/fs/cgroup")
_PROC_ROOT = Path("/proc")
_CGROUP_SETUP_TIMEOUT_SEC = 5.0
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
        or config.finalization_policy
        != "verified-native-free-authoritative-batch-v2"
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
            or request.stream.partial_interval_ms
            != config.online_partial_interval_ms
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
    return (
        NATIVE_ENDPOINT_PROTOCOL_VERSION
        if manifest.adapter == PARAKEET_REALTIME_EOU_ADAPTER
        else PROTOCOL_VERSION
    )


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
        payload = os.read(descriptor, maximum_bytes + 1)
        if len(payload) > maximum_bytes:
            raise WorkerError("worker_prerequisite")
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
            not isinstance(config, (NemotronConfig, ParakeetRealtimeEouConfig))
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
    """Build the closed Nemotron-only Bubblewrap command."""

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
    venv_root = manifest.python.path.parent.parent
    required_paths = (
        scratch,
        source_bundle.root,
        source_bundle.worker_path,
        manifest.path,
        manifest.worker.path,
        venv_root,
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
    _append_mount(command, "--ro-bind", venv_root)
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
        try:
            lexical_scratch = Path(os.path.abspath(self.scratch_root))
            with opened_directory_nofollow(
                lexical_scratch,
                require_private=True,
            ) as (scratch, _descriptor):
                pass
        except (OSError, RuntimeError, BoundedReadError):
            raise WorkerError("worker_prerequisite") from None
        device_nodes: tuple[Path, ...] = ()
        system_ro_paths: tuple[Path, ...] = ()
        proc_driver_available = False
        user_runtime_directory: Path | None = None
        if self.manifest.adapter in {
            FASTER_WHISPER_ENDPOINT_ADAPTER,
            NEMOTRON_ADAPTER,
            PARAKEET_REALTIME_EOU_ADAPTER,
            SHERPA_ZIPFORMER_ADAPTER,
        }:
            if os.name != "posix" or not Path("/proc/self/fd").is_dir():
                raise WorkerError("worker_prerequisite")
        if self.manifest.adapter == SHERPA_ZIPFORMER_ADAPTER:
            system_ro_paths = _core_system_ro_paths()
        if self.manifest.adapter in {
            FASTER_WHISPER_ENDPOINT_ADAPTER,
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
        if self.manifest.adapter == PARAKEET_REALTIME_EOU_ADAPTER:
            user_runtime_directory = _verified_user_runtime_directory()
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
                NEMOTRON_ADAPTER,
                PARAKEET_REALTIME_EOU_ADAPTER,
                SHERPA_ZIPFORMER_ADAPTER,
            }:
                self._bwrap_descriptor = _open_bwrap_descriptor()
            if self.manifest.adapter == PARAKEET_REALTIME_EOU_ADAPTER:
                self._systemd_run_descriptor = _open_systemd_run_descriptor()
        except (OSError, BoundedReadError, WorkerError):
            self._close_bundle_descriptors()
            raise WorkerError("worker_prerequisite") from None
        bundle_root = Path(f"/proc/self/fd/{self._bundle_descriptor}")
        interpreter_flags = (
            ["-I", "-B"]
            if self.manifest.adapter == SHERPA_ZIPFORMER_ADAPTER
            else ["-I", "-S", "-B"]
        )
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
        if self.manifest.adapter in {
            FASTER_WHISPER_ENDPOINT_ADAPTER,
            NEMOTRON_ADAPTER,
            PARAKEET_REALTIME_EOU_ADAPTER,
        }:
            try:
                builder = (
                    _faster_whisper_bwrap_command
                    if self.manifest.adapter == FASTER_WHISPER_ENDPOINT_ADAPTER
                    else _nemotron_bwrap_command
                )
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
            if self.manifest.adapter == PARAKEET_REALTIME_EOU_ADAPTER:
                command = _parakeet_systemd_scope_command(
                    command,
                    bwrap_descriptor=self._bwrap_descriptor,
                )
                executable = f"/proc/self/fd/{self._systemd_run_descriptor}"
                pass_fds = (*pass_fds, self._systemd_run_descriptor)
            # Popen's session owns the outer Bubblewrap monitor as a bounded
            # process group. Bubblewrap forks its namespace child before the
            # required --new-session setsid(), so the inner session remains
            # independent and die-with-parent links both lifecycle layers.
        elif self.manifest.adapter == SHERPA_ZIPFORMER_ADAPTER:
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
        if self.manifest.adapter == PARAKEET_REALTIME_EOU_ADAPTER:
            if user_runtime_directory is None:
                self._close_bundle_descriptors()
                raise WorkerError("worker_prerequisite")
            environment["XDG_RUNTIME_DIR"] = str(user_runtime_directory)
            # NeMo imports lhotse, whose environment probe requires the normal
            # host binary directory even though the worker has no network and
            # executes only receipt-bound Python/source artifacts.
            environment["PATH"] = "/usr/bin"
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
                self._cgroup_evidence = _await_parakeet_cgroup(
                    process,
                    monotonic=self._monotonic,
                )
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
            return event
        except Exception:
            try:
                self._terminate()
            finally:
                self._close_process_pipes()
                self._close_bundle_descriptors()
            raise

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
            schema_v6_final_only = self.manifest.schema_version == 6
            if schema_v6_final_only != (
                self.manifest.adapter == FASTER_WHISPER_ENDPOINT_ADAPTER
            ):
                raise WorkerError("worker_protocol")
            if self.manifest.adapter == MOONSHINE_EXTERNAL_ENDPOINT_ADAPTER:
                _expected_final_evidence(self.manifest, checked)
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
                        schema_v6_final_only
                        or event.protocol_version != expected_protocol
                        or event.request_id != checked.request_id
                        or event.seq != len(partials)
                        or event.samples_seen < last_samples
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
                    return CaseTrace(partials=tuple(partials), final=event)
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
                    or (schema_v6_final_only and event.seq != 0)
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
                return CaseTrace(partials=tuple(partials), final=event)
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
            self._close_bundle_descriptors()
            return
        try:
            if active:
                self._terminate()
                self._verify_bound_files()
                return
            if process.poll() is None and self.ready is not None:
                try:
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
            ):
                raise WorkerError("worker_artifact_changed")
            if self.manifest.adapter in {
                FASTER_WHISPER_ENDPOINT_ADAPTER,
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
