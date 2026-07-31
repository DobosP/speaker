"""Aggregate-only streaming-STT evaluation through an isolated JSONL worker.

This first implementation intentionally supports only the deterministic fake
adapter. It validates the process, provenance, replay, privacy, and metric
contracts without importing or installing any candidate model runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pwd
import secrets
import shutil
import stat
import tempfile
from typing import Mapping, Sequence

try:
    import fcntl
except ImportError:  # pragma: no cover - the benchmark controller is Linux-first
    fcntl = None  # type: ignore[assignment]

from tools.streaming_stt.bounded_io import (
    BoundedReadError,
    hash_regular_bounded,
    opened_directory_nofollow,
)
from tools.streaming_stt.corpus import (
    LoadedCorpus,
    load_corpus,
    verify_corpus_snapshot,
)
from tools.streaming_stt.manifest import (
    MAX_WORKER_BYTES,
    WorkerManifest,
    load_worker_manifest,
)
from tools.streaming_stt.metrics import RunRecord, aggregate_metrics
from tools.streaming_stt.protocol import (
    MAX_PCM_BYTES,
    PcmInput,
    StreamConfig,
    TranscribeRequest,
    encode_message,
    parse_request,
)
from tools.streaming_stt.supervisor import StreamingWorker
from tools.streaming_stt.source_bundle import (
    WORKER_SOURCE_FILES,
    SourceBundle,
    stage_worker_source_bundle,
    verify_source_bundle,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SCRATCH = _REPO_ROOT / ".cache" / "streaming-stt"
_HOST_UID = os.getuid() if hasattr(os, "getuid") else -1
_HOST_HOME = Path(pwd.getpwuid(_HOST_UID).pw_dir) if _HOST_UID >= 0 else Path.home()


def _trusted_host_lock_path() -> Path:
    return _HOST_HOME / ".local" / "state" / "speaker" / "streaming-stt-benchmark.lock"


_HOST_LOCK_PATH = _trusted_host_lock_path()
_FAKE_WORKER = (_REPO_ROOT / "tools/streaming_stt/worker.py").resolve(strict=True)
_SAFE_ERROR = {
    "ok": False,
    "error": "streaming_stt_prerequisites_unavailable",
}
_EVALUATOR_FILES = (
    "tools/streaming_stt_eval.py",
    "tools/streaming_stt/bounded_io.py",
    "tools/streaming_stt/corpus.py",
    "tools/streaming_stt/manifest.py",
    "tools/streaming_stt/metrics.py",
    "tools/streaming_stt/protocol.py",
    "tools/streaming_stt/source_bundle.py",
    "tools/streaming_stt/supervisor.py",
    "tools/streaming_stt/worker.py",
    "tools/streaming_stt/__init__.py",
    "tools/streaming_stt/adapters/__init__.py",
    "tools/streaming_stt/adapters/fake.py",
    "tools/__init__.py",
    "tools/recorded_stt_eval.py",
    "core/wer.py",
)


class _BenchmarkRunLock:
    """One fixed host lease plus a pathname-bound advisory lock."""

    def __init__(
        self,
        path: Path,
        descriptor: int,
        parent_descriptor: int,
    ) -> None:
        self._path = path
        self._descriptor = descriptor
        self._parent_descriptor = parent_descriptor
        self._identity = self._file_identity(os.fstat(descriptor))
        self._parent_identity = self._directory_identity(os.fstat(parent_descriptor))

    @staticmethod
    def _file_identity(metadata: os.stat_result) -> tuple[int, int, int]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            stat.S_IFMT(metadata.st_mode),
        )

    @staticmethod
    def _directory_identity(metadata: os.stat_result) -> tuple[int, int, int]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            stat.S_IFMT(metadata.st_mode),
        )

    @classmethod
    def acquire(cls, path: Path) -> _BenchmarkRunLock:
        if fcntl is None or os.name != "posix":
            raise ValueError
        candidate = Path(os.path.abspath(path))
        descriptor = -1
        parent_descriptor = -1
        try:
            with opened_directory_nofollow(
                candidate.parent,
                create=True,
                require_private=True,
            ) as (parent, stable_parent):
                if parent != candidate.parent:
                    raise ValueError
                parent_descriptor = os.dup(stable_parent)
            fcntl.flock(
                parent_descriptor,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
            flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(
                candidate.name,
                flags,
                0o600,
                dir_fd=parent_descriptor,
            )
            metadata = os.fstat(descriptor)
            pathname_metadata = os.stat(
                candidate.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or cls._file_identity(metadata) != cls._file_identity(pathname_metadata)
                or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
            ):
                raise ValueError
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = cls(
                candidate,
                descriptor,
                parent_descriptor,
            )
            acquired.assert_bound()
            return acquired
        except (OSError, ValueError, BoundedReadError):
            if descriptor >= 0:
                os.close(descriptor)
            if parent_descriptor >= 0:
                os.close(parent_descriptor)
            raise ValueError from None

    def assert_bound(self) -> None:
        descriptor = self._descriptor
        parent_descriptor = self._parent_descriptor
        if descriptor < 0 or parent_descriptor < 0:
            raise ValueError
        try:
            metadata = os.fstat(descriptor)
            parent_metadata = os.fstat(parent_descriptor)
            parent_path_metadata = self._path.parent.lstat()
            relative_metadata = os.stat(
                self._path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            full_path_metadata = self._path.lstat()
        except OSError:
            raise ValueError from None
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
            or self._file_identity(metadata) != self._identity
            or self._file_identity(relative_metadata) != self._identity
            or self._file_identity(full_path_metadata) != self._identity
            or self._directory_identity(parent_metadata) != self._parent_identity
            or self._directory_identity(parent_path_metadata) != self._parent_identity
        ):
            raise ValueError

    def close(self) -> None:
        descriptor = self._descriptor
        parent_descriptor = self._parent_descriptor
        self._descriptor = -1
        self._parent_descriptor = -1
        if descriptor < 0:
            return
        changed = False
        try:
            try:
                self._descriptor = descriptor
                self._parent_descriptor = parent_descriptor
                self.assert_bound()
            except ValueError:
                changed = True
            finally:
                self._descriptor = -1
                self._parent_descriptor = -1
            assert fcntl is not None
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
            if parent_descriptor >= 0:
                try:
                    assert fcntl is not None
                    fcntl.flock(parent_descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(parent_descriptor)
        if changed:
            raise ValueError


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, OverflowError):
        raise ValueError from None


def _write_strict_report(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically write one private standards-compliant JSON report."""

    encoded = (_strict_json(payload) + "\n").encode("utf-8")
    selected = path.expanduser()
    selected.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".streaming-stt-report-",
        dir=selected.parent,
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, selected)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _evaluator_binding() -> dict[str, object]:
    files: dict[str, str] = {}
    for relative in _EVALUATOR_FILES:
        path = (_REPO_ROOT / relative).resolve(strict=True)
        path.relative_to(_REPO_ROOT.resolve(strict=True))
        files[relative] = hash_regular_bounded(
            path,
            maximum_bytes=MAX_WORKER_BYTES,
            expected_bytes=path.stat().st_size,
        ).sha256
    return {
        "schema_version": 1,
        "kind": "isolated_streaming_stt_fake_harness",
        "production_model": False,
        "files": files,
    }


def _worker_binding(
    manifest: WorkerManifest,
    source_bundle: SourceBundle,
) -> dict[str, object]:
    artifacts = [
        {
            "name": artifact.name,
            "sha256": artifact.sha256,
            "size_bytes": artifact.size_bytes,
        }
        for artifact in manifest.artifacts
    ]
    return {
        "manifest_sha256": manifest.digest,
        "model_id": manifest.model_id,
        "adapter": manifest.adapter,
        "python_sha256": manifest.python.sha256,
        "worker_sha256": manifest.worker.sha256,
        "source_bundle_sha256": source_bundle.tree_sha256,
        "source_bundle_files": len(source_bundle.files),
        "artifact_set_sha256": _canonical_digest(artifacts),
        "artifacts": artifacts,
    }


def _corpus_binding(corpus: LoadedCorpus) -> dict[str, object]:
    case_set = [
        {
            "sha256": case.sha256,
            "samples": case.samples,
            "assertion": case.assertion,
            "commands": len(case.commands),
            "tags": list(case.tags),
        }
        for case in corpus.cases
    ]
    return {
        "manifest_sha256": corpus.digest,
        "case_set_sha256": _canonical_digest(case_set),
        "cases": len(corpus.cases),
        "audio_bytes": corpus.audio_bytes,
        "purpose_sha256": hashlib.sha256(corpus.purpose.encode("utf-8")).hexdigest(),
    }


def _prepare_scratch(parent: Path) -> Path:
    try:
        with opened_directory_nofollow(
            Path(os.path.abspath(parent)),
            create=True,
            require_private=True,
        ) as (stable_parent, descriptor):
            for _attempt in range(32):
                name = f".streaming-stt-{secrets.token_hex(16)}"
                try:
                    os.mkdir(name, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    continue
                path = stable_parent / name
                with opened_directory_nofollow(
                    path,
                    require_private=True,
                ):
                    pass
                return path
    except (OSError, BoundedReadError):
        raise ValueError from None
    raise ValueError


def _chmod_owned_directory_at(parent_descriptor: int, name: str) -> None:
    metadata = os.stat(
        name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    if not stat.S_ISDIR(metadata.st_mode) or (
        hasattr(os, "geteuid") and metadata.st_uid != os.geteuid()
    ):
        raise ValueError
    os.chmod(
        name,
        0o700,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    current = os.stat(
        name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    if (
        current.st_dev,
        current.st_ino,
        stat.S_IFMT(current.st_mode),
    ) != (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
    ):
        raise ValueError


def _unlock_private_directory(descriptor: int) -> None:
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode) or (
        hasattr(os, "geteuid") and metadata.st_uid != os.geteuid()
    ):
        raise ValueError
    os.fchmod(descriptor, 0o700)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    for name in os.listdir(descriptor):
        child_metadata = os.stat(
            name,
            dir_fd=descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(child_metadata.st_mode):
            continue
        _chmod_owned_directory_at(descriptor, name)
        child = os.open(name, flags, dir_fd=descriptor)
        try:
            opened = os.fstat(child)
            if (
                opened.st_dev,
                opened.st_ino,
                stat.S_IFMT(opened.st_mode),
            ) != (
                child_metadata.st_dev,
                child_metadata.st_ino,
                stat.S_IFMT(child_metadata.st_mode),
            ):
                raise ValueError
            _unlock_private_directory(child)
        finally:
            os.close(child)


def _remove_private_scratch(
    root: Path,
    *,
    expected_identity: tuple[int, int],
) -> None:
    """Remove private scratch without chmod-following links or hard-linked files."""

    if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
        raise ValueError
    root = Path(os.path.abspath(root))
    parent = root.parent
    try:
        with opened_directory_nofollow(parent) as (_parent, parent_descriptor):
            initial = os.stat(
                root.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (initial.st_dev, initial.st_ino) != expected_identity:
                raise ValueError
            _chmod_owned_directory_at(parent_descriptor, root.name)
        with opened_directory_nofollow(root) as (_stable_root, root_descriptor):
            root_identity = (
                os.fstat(root_descriptor).st_dev,
                os.fstat(root_descriptor).st_ino,
            )
            _unlock_private_directory(root_descriptor)
            current = root.lstat()
            if (current.st_dev, current.st_ino) != root_identity:
                raise ValueError
    except (OSError, BoundedReadError):
        raise ValueError from None
    shutil.rmtree(root)
    if root.exists() or root.is_symlink():
        raise ValueError


def _validated_stream(stream: StreamConfig) -> StreamConfig:
    probe = TranscribeRequest(
        request_id="stream-validation",
        pcm=PcmInput(
            path=(Path.cwd() / ".stream-validation.f32le").resolve(),
            sha256="0" * 64,
            samples=1,
        ),
        stream=stream,
    )
    try:
        parsed = parse_request(encode_message(probe.as_dict()))
    except Exception:
        raise ValueError from None
    if not isinstance(parsed, TranscribeRequest):
        raise ValueError
    return parsed.stream


def _write_pcm_snapshot(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def run_benchmark(
    worker_manifest_path: Path | str,
    corpus_path: Path | str,
    *,
    scratch_parent: Path | str,
    repeats: int = 3,
    stream: StreamConfig | None = None,
) -> dict[str, object]:
    """Run one exact fake worker/corpus tuple and return no transcript rows."""

    if (
        isinstance(repeats, bool)
        or not isinstance(repeats, int)
        or not 1 <= repeats <= 8
    ):
        raise ValueError
    selected_stream = _validated_stream(
        stream
        or StreamConfig(
            chunk_samples=1600,
            pace="burst",
            partial_interval_ms=200,
            tail_padding_samples=0,
        )
    )
    run_lock = _BenchmarkRunLock.acquire(_HOST_LOCK_PATH)
    try:
        run_lock.assert_bound()
        result = _run_benchmark_locked(
            worker_manifest_path,
            corpus_path,
            scratch_parent=Path(scratch_parent).expanduser(),
            repeats=repeats,
            selected_stream=selected_stream,
        )
        run_lock.assert_bound()
        return result
    finally:
        run_lock.close()


def _run_benchmark_locked(
    worker_manifest_path: Path | str,
    corpus_path: Path | str,
    *,
    scratch_parent: Path,
    repeats: int,
    selected_stream: StreamConfig,
) -> dict[str, object]:
    manifest = load_worker_manifest(worker_manifest_path)
    try:
        if manifest.worker.path.resolve(strict=True) != _FAKE_WORKER:
            raise ValueError
    except (OSError, RuntimeError):
        raise ValueError from None
    corpus = load_corpus(corpus_path)
    evaluator_binding = _evaluator_binding()
    corpus_binding = _corpus_binding(corpus)
    scratch = _prepare_scratch(scratch_parent)
    scratch_metadata = scratch.lstat()
    scratch_identity = (scratch_metadata.st_dev, scratch_metadata.st_ino)
    records: list[RunRecord] = []
    stderr_summary: Mapping[str, object] = {}
    worker: StreamingWorker | None = None
    source_bundle: SourceBundle | None = None
    try:
        source_bundle = stage_worker_source_bundle(
            _REPO_ROOT,
            scratch / "source-bundle",
        )
        worker_binding = _worker_binding(manifest, source_bundle)
        snapshots: list[Path] = []
        for index, case in enumerate(corpus.cases):
            path = scratch / f"case-{index:04d}.f32le"
            _write_pcm_snapshot(path, case.audio_bytes)
            if (
                hash_regular_bounded(
                    path,
                    maximum_bytes=MAX_PCM_BYTES,
                    expected_bytes=len(case.audio_bytes),
                ).sha256
                != case.sha256
            ):
                raise ValueError
            snapshots.append(path)

        worker = StreamingWorker(manifest, scratch, source_bundle)
        with worker:
            assert worker.ready is not None
            for repeat in range(repeats):
                for case_index, (case, pcm_path) in enumerate(
                    zip(corpus.cases, snapshots)
                ):
                    request = TranscribeRequest(
                        request_id=f"case-{case_index:04d}-r{repeat}",
                        pcm=PcmInput(
                            path=pcm_path.resolve(strict=True),
                            sha256=case.sha256,
                            samples=case.samples,
                        ),
                        stream=selected_stream,
                    )
                    records.append(
                        RunRecord(
                            case_index=case_index,
                            repeat=repeat,
                            trace=worker.transcribe(request),
                        )
                    )
            ready = worker.ready
        stderr_summary = worker.stderr_summary
        verify_corpus_snapshot(corpus)
        verify_source_bundle(
            source_bundle,
            required_files=WORKER_SOURCE_FILES,
        )
        current_manifest = load_worker_manifest(manifest.path)
        if (
            current_manifest.digest != manifest.digest
            or current_manifest.worker.path.resolve(strict=True) != _FAKE_WORKER
            or _worker_binding(current_manifest, source_bundle) != worker_binding
            or _evaluator_binding() != evaluator_binding
        ):
            raise ValueError
        metrics = aggregate_metrics(corpus, records, ready, repeats=repeats)
        worker_report = dict(worker_binding)
        worker_report["runtime"] = dict(ready.runtime)
        result = {
            "ok": bool(metrics["coverage_complete"]),
            "evidence": {
                "kind": "bounded_pcm_streaming_harness",
                "fake_adapter": True,
                "production_model": False,
                "vad": False,
                "endpointing": False,
                "aec": False,
                "live_hardware": False,
                "adoption_authority": False,
            },
            "worker": worker_report,
            "corpus": corpus_binding,
            "config": {
                **selected_stream.as_dict(),
                "repeats": repeats,
                "contract_sha256": _canonical_digest(
                    {
                        **selected_stream.as_dict(),
                        "repeats": repeats,
                    }
                ),
            },
            "metrics": metrics,
            "worker_stderr": dict(stderr_summary),
            "evaluator": evaluator_binding,
        }
        _strict_json(result)
        return result
    finally:
        if worker is not None and not worker.fully_stopped:
            raise RuntimeError
        _remove_private_scratch(
            scratch,
            expected_identity=scratch_identity,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate-only isolated streaming-STT harness. The current "
            "implementation accepts deterministic fake worker receipts only."
        )
    )
    parser.add_argument("--worker-manifest", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, default=_DEFAULT_SCRATCH)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--chunk-samples", type=int, default=1600)
    parser.add_argument("--pace", choices=("burst", "realtime"), default="burst")
    parser.add_argument("--partial-interval-ms", type=int, default=200)
    parser.add_argument("--tail-padding-samples", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = run_benchmark(
            args.worker_manifest,
            args.corpus,
            scratch_parent=args.scratch_root,
            repeats=args.repeats,
            stream=StreamConfig(
                chunk_samples=args.chunk_samples,
                pace=args.pace,
                partial_interval_ms=args.partial_interval_ms,
                tail_padding_samples=args.tail_padding_samples,
            ),
        )
        if args.output is not None:
            _write_strict_report(args.output, payload)
        print(_strict_json(payload))
        return 0 if payload["ok"] else 1
    except Exception:  # noqa: BLE001 - private/native details never reach the CLI
        print(_strict_json(_SAFE_ERROR))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
