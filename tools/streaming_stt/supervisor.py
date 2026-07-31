"""Bounded process supervision for one isolated streaming recognizer."""

from __future__ import annotations

import hashlib
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
    MAX_PYTHON_BYTES,
    MAX_WORKER_BYTES,
    MOONSHINE_ADAPTER,
    NEMOTRON_ADAPTER,
    NemotronConfig,
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
    NEMOTRON_RUNTIME_TREE_LIMITS,
    RuntimeTreeReceipt,
    load_runtime_tree_receipt,
    verify_moonshine_wheel_install,
    verify_runtime_tree_receipt,
    verify_venv_runtime_location,
)


_TERM_GRACE_SEC = 5.0
_KILL_GRACE_SEC = 2.0
_QUEUE_CAPACITY = MAX_PARTIALS_PER_CASE + 16
_BWRAP_PATH = Path("/usr/bin/bwrap")
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


def _stable_path_metadata(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _open_bwrap_descriptor() -> int:
    """Open the exact Bubblewrap executable without following a replacement."""

    descriptor = -1
    try:
        before = _BWRAP_PATH.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink < 1
            or stat.S_IMODE(before.st_mode) & 0o111 == 0
        ):
            raise WorkerError("worker_prerequisite")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(_BWRAP_PATH, flags)
        opened = os.fstat(descriptor)
        current = _BWRAP_PATH.lstat()
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

    for path in _NEMOTRON_CORE_RO_PATHS[:3]:
        _require_mount_source(path, directory=True)
    for path in _NEMOTRON_CORE_RO_PATHS[3:]:
        _require_mount_source(path, directory=False)
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
        *_NEMOTRON_CORE_RO_PATHS,
        _NVIDIA_DRIVER_ROOT,
        *tuple(dict.fromkeys(pci_targets)),
        *modules,
    )


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
            not isinstance(config, NemotronConfig)
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
        if self.manifest.adapter == NEMOTRON_ADAPTER:
            if os.name != "posix" or not Path("/proc/self/fd").is_dir():
                raise WorkerError("worker_prerequisite")
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
            if self.manifest.adapter == NEMOTRON_ADAPTER:
                self._bwrap_descriptor = _open_bwrap_descriptor()
        except (OSError, BoundedReadError, WorkerError):
            self._close_bundle_descriptors()
            raise WorkerError("worker_prerequisite") from None
        bundle_root = Path(f"/proc/self/fd/{self._bundle_descriptor}")
        worker_command = [
            str(self.manifest.python.path),
            "-I",
            "-S",
            "-B",
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
        if self.manifest.adapter == NEMOTRON_ADAPTER:
            try:
                command = _nemotron_bwrap_command(
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
            # Popen's session owns the outer Bubblewrap monitor as a bounded
            # process group. Bubblewrap forks its namespace child before the
            # required --new-session setsid(), so the inner session remains
            # independent and die-with-parent links both lifecycle layers.
        try:
            process = self._popen_factory(
                command,
                cwd=str(scratch),
                env=sanitized_worker_environment(scratch),
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
            event = self._next_event(self.manifest.limits.startup_timeout_sec)
            if (
                not isinstance(event, ReadyEvent)
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
            self._verify_pcm(checked)
            self._send(encoded)
            partials: list[PartialEvent] = []
            last_samples = -1
            last_elapsed = -1.0
            case_deadline = self._monotonic() + self.manifest.limits.case_timeout_sec
            while True:
                event = self._next_event(max(0.0, case_deadline - self._monotonic()))
                if isinstance(event, PartialEvent):
                    if (
                        event.request_id != checked.request_id
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
                if isinstance(event, ErrorEvent):
                    if event.request_id != checked.request_id:
                        raise WorkerError("worker_protocol")
                    if event.fatal:
                        self._terminate()
                    raise WorkerError("worker_case")
                if not isinstance(event, FinalEvent):
                    raise WorkerError("worker_protocol")
                expected_chunks, expected_model_padding = _expected_final_evidence(
                    self.manifest,
                    checked,
                )
                if (
                    event.request_id != checked.request_id
                    or event.seq != len(partials)
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
                    request = ShutdownRequest("shutdown")
                    self._send(encode_message(request.as_dict()))
                    acknowledgement = self._next_event(_TERM_GRACE_SEC)
                    goodbye = self._next_event(_TERM_GRACE_SEC)
                    if (
                        not isinstance(acknowledgement, ShutdownEvent)
                        or acknowledgement.request_id != request.request_id
                        or not isinstance(goodbye, ByeEvent)
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
            if self.manifest.adapter in {MOONSHINE_ADAPTER, NEMOTRON_ADAPTER}:
                receipt_artifact = self.manifest.artifact_by_name.get("runtime-receipt")
                if receipt_artifact is None:
                    raise WorkerError("worker_artifact_changed")
                limits = (
                    NEMOTRON_RUNTIME_TREE_LIMITS
                    if self.manifest.adapter == NEMOTRON_ADAPTER
                    else None
                )
                receipt = load_runtime_tree_receipt(
                    receipt_artifact.path,
                    expected_digest=receipt_artifact.sha256,
                    **({} if limits is None else {"limits": limits}),
                )
                verify_venv_runtime_location(
                    receipt,
                    self.manifest.python.path,
                )
                verify_runtime_tree_receipt(
                    receipt,
                    **({} if limits is None else {"limits": limits}),
                )
                if self.manifest.adapter == NEMOTRON_ADAPTER:
                    _verify_nemotron_runtime_layout(self.manifest, receipt)
                if self.manifest.adapter == MOONSHINE_ADAPTER:
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
