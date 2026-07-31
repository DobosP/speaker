"""Bounded process supervision for one isolated streaming recognizer."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import queue
import signal
import stat
import subprocess
import threading
import time
from typing import Mapping

from .bounded_io import (
    BoundedReadError,
    hash_regular_bounded,
    opened_directory_nofollow,
)
from .manifest import (
    MAX_PYTHON_BYTES,
    MAX_WORKER_BYTES,
    MOONSHINE_ADAPTER,
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
    load_runtime_tree_receipt,
    verify_moonshine_wheel_install,
    verify_runtime_tree_receipt,
    verify_venv_runtime_location,
)


_TERM_GRACE_SEC = 5.0
_KILL_GRACE_SEC = 2.0
_QUEUE_CAPACITY = MAX_PARTIALS_PER_CASE + 16


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
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "MOONSHINE_ORT_SINGLE_THREAD": "1",
    }
    if os.name == "nt":
        for name in ("SYSTEMROOT", "WINDIR"):
            value = os.environ.get(name)
            if value:
                environment[name] = value
    return environment


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
        except (OSError, BoundedReadError, WorkerError):
            self._close_bundle_descriptors()
            raise WorkerError("worker_prerequisite") from None
        bundle_root = Path(f"/proc/self/fd/{self._bundle_descriptor}")
        command = [
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
        try:
            process = self._popen_factory(
                command,
                cwd=str(scratch),
                env=sanitized_worker_environment(scratch),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                close_fds=True,
                executable=f"/proc/self/fd/{self._python_descriptor}",
                pass_fds=(
                    self._bundle_descriptor,
                    self._worker_descriptor,
                    self._python_descriptor,
                ),
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
                expected_chunks = (
                    checked.pcm.samples
                    + checked.stream.tail_padding_samples
                    + checked.stream.chunk_samples
                    - 1
                ) // checked.stream.chunk_samples
                if (
                    event.request_id != checked.request_id
                    or event.seq != len(partials)
                    or event.samples_seen
                    != checked.pcm.samples + checked.stream.tail_padding_samples
                    or event.samples_seen < last_samples
                    or event.elapsed_ms < last_elapsed
                    or event.chunks != expected_chunks
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
            if self.manifest.adapter == MOONSHINE_ADAPTER:
                receipt_artifact = self.manifest.artifact_by_name.get("runtime-receipt")
                wheel_artifact = self.manifest.artifact_by_name.get("release-wheel")
                if receipt_artifact is None or wheel_artifact is None:
                    raise WorkerError("worker_artifact_changed")
                receipt = load_runtime_tree_receipt(
                    receipt_artifact.path,
                    expected_digest=receipt_artifact.sha256,
                )
                verify_venv_runtime_location(
                    receipt,
                    self.manifest.python.path,
                )
                verify_runtime_tree_receipt(receipt)
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
