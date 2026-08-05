"""Validate and publish a private aggregate inventory of LiveKit English EoT.

This command has no downloader or network client.  Production use requires the
exact pinned English validation Parquet shard, an explicit isolated PyArrow 25
interpreter, explicit CC-BY-4.0 acknowledgement, and a new report path outside
every recognized Git worktree.  The report contains fixed public contract
metadata plus aggregate counts and digests; row IDs, audio, text, messages,
words, and local paths never leave the isolated worker.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
from typing import Callable, Iterator, Mapping, Sequence

from tools import livekit_eot_parquet_worker as worker_contract
from tools import public_voice_fixtures as public_fixtures
from tools.streaming_stt.bounded_io import BoundedReadError, opened_directory_nofollow


REPORT_SCHEMA_VERSION = 1
REPORT_KIND = "livekit-eot-english-aggregate-inventory-v1"
DATASET = "livekit/eot-bench-data"
CONFIG = "en"
SPLIT = "validation"
LICENSE = "CC-BY-4.0"
SOURCE_REVISION = worker_contract.SOURCE_REVISION
SOURCE_SIZE_BYTES = worker_contract.SOURCE_SIZE_BYTES
SOURCE_SHA256 = worker_contract.SOURCE_SHA256
SOURCE_ROWS = worker_contract.SOURCE_ROWS
PYARROW_VERSION = worker_contract.PYARROW_VERSION
PROTOCOL_VERSION = worker_contract.PROTOCOL_VERSION
SCHEMA_CONTRACT_SHA256 = worker_contract.SCHEMA_CONTRACT_SHA256
LABEL_CONVENTION = worker_contract.LABEL_CONVENTION
DURATION_EXACT_SAMPLE_COUNT = worker_contract.DURATION_EXACT_SAMPLE_COUNT
DURATION_MICROSECOND_QUANTIZED_COUNT = (
    worker_contract.DURATION_MICROSECOND_QUANTIZED_COUNT
)
FINAL_GAP_ZERO_SAMPLE_COUNT = worker_contract.FINAL_GAP_ZERO_SAMPLE_COUNT
FINAL_GAP_ONE_SAMPLE_COUNT = worker_contract.FINAL_GAP_ONE_SAMPLE_COUNT

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKER_PATH = Path(worker_contract.__file__).resolve(strict=True)
_EXECUTION_CLOSURE_RELATIVE_PATHS = (
    "tools/inventory_livekit_eot_english.py",
    "tools/livekit_eot_parquet_worker.py",
    "tools/public_voice_fixtures.py",
    "tools/streaming_stt/bounded_io.py",
)
_MAX_WORKER_BYTES = 1024 * 1024
_MAX_WORKER_RESULT_BYTES = 64 * 1024
_MAX_REPORT_BYTES = 64 * 1024
_WORKER_TIMEOUT_SECONDS = 720.0
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,95}\Z")
_REPORT_FIELDS = {
    "schema_version",
    "kind",
    "dataset",
    "config",
    "split",
    "license",
    "attribution",
    "evaluation_scope",
    "source_revision",
    "source_size_bytes",
    "source_sha256",
    "schema_contract_sha256",
    "execution_closure",
    "worker",
    "inventory",
    "production_evidence",
}
_WORKER_RESULT_FIELDS = {
    "schema_version",
    "pyarrow_version",
    "source_revision",
    "source_size_bytes",
    "source_sha256",
    "schema_contract_sha256",
    "row_count",
    "row_group_count",
    "audio_file_count",
    "audio_bytes_total",
    "audio_samples_total",
    "silence_span_count",
    "hold_label_count",
    "eot_label_count",
    "minimum_eot_silence_samples",
    "maximum_silence_span_samples",
    "duration_exact_sample_count",
    "duration_microsecond_quantized_count",
    "final_gap_zero_sample_count",
    "final_gap_one_sample_count",
    "label_convention",
    "audio_set_sha256",
    "silence_set_sha256",
    "inventory_sha256",
}
_INVENTORY_FIELDS = _WORKER_RESULT_FIELDS - {
    "schema_version",
    "pyarrow_version",
    "source_revision",
    "source_size_bytes",
    "source_sha256",
    "schema_contract_sha256",
}
_SAFE_ERROR = {
    "ok": False,
    "error": "livekit_eot_inventory_prerequisites_unavailable",
}


class LiveKitEotInventoryError(RuntimeError):
    """A detail-free prerequisite, source, worker, or publication failure."""


@dataclass(frozen=True, slots=True)
class TestWorkerInjection:
    """Synthetic-only seam which can never produce production evidence."""

    run: Callable[[int, Path], None] = field(repr=False)
    fixture: str


@dataclass(frozen=True, slots=True)
class InventoryResult:
    report_path: Path = field(repr=False)
    report_sha256: str
    row_count: int
    silence_span_count: int
    production_evidence: bool


@dataclass(frozen=True, slots=True)
class _SourceHandle:
    path: Path = field(repr=False)
    descriptor: int = field(repr=False)
    identity: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _WorkerRun:
    output_dir: Path = field(repr=False)
    binding: Mapping[str, object]


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, UnicodeError, ValueError, OverflowError):
        raise LiveKitEotInventoryError("report contract failed") from None


def _strict_json(raw: bytes) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise LiveKitEotInventoryError("worker result failed")
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except LiveKitEotInventoryError:
        raise
    except (UnicodeError, ValueError, OverflowError):
        raise LiveKitEotInventoryError("worker result failed") from None


def _snapshot_execution_closure() -> dict[str, str]:
    closure: dict[str, str] = {}
    try:
        for relative_path in _EXECUTION_CLOSURE_RELATIVE_PATHS:
            source = _REPO_ROOT / relative_path
            if source.resolve(strict=True) != source:
                raise LiveKitEotInventoryError("execution closure failed")
            payload = public_fixtures._read_stable_repo_worker(source)
            closure[relative_path] = hashlib.sha256(payload).hexdigest()
    except LiveKitEotInventoryError:
        raise
    except (OSError, RuntimeError, ValueError, public_fixtures.PublicFixtureError):
        raise LiveKitEotInventoryError("execution closure failed") from None
    if tuple(closure) != _EXECUTION_CLOSURE_RELATIVE_PATHS or any(
        _SHA256_RE.fullmatch(digest) is None for digest in closure.values()
    ):
        raise LiveKitEotInventoryError("execution closure failed")
    return closure


def _rebind_execution_closure(expected: Mapping[str, str]) -> None:
    if (
        not isinstance(expected, dict)
        or tuple(expected) != _EXECUTION_CLOSURE_RELATIVE_PATHS
        or any(
            not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None
            for digest in expected.values()
        )
        or _snapshot_execution_closure() != expected
    ):
        raise LiveKitEotInventoryError("execution closure changed")


def _validate_execution_worker_binding(
    worker_binding: Mapping[str, object],
    execution_closure: Mapping[str, str],
) -> bool:
    production_evidence = worker_binding.get("production_evidence") is True
    if production_evidence and worker_binding.get("worker_sha256") != execution_closure[
        "tools/livekit_eot_parquet_worker.py"
    ]:
        raise LiveKitEotInventoryError("execution closure changed")
    return production_evidence


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_uid,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _directory_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        stat.S_IFMT(info.st_mode),
        stat.S_IMODE(info.st_mode),
        info.st_uid,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _directory_entry_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        stat.S_IFMT(info.st_mode),
        stat.S_IMODE(info.st_mode),
        info.st_uid,
    )


def _git_marker_present(directory_descriptor: int) -> bool:
    marker_descriptor = -1
    try:
        try:
            marker = os.stat(
                ".git",
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            try:
                os.stat(
                    ".git",
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return False
            raise LiveKitEotInventoryError("Git boundary verification failed")
        if stat.S_ISREG(marker.st_mode):
            if marker.st_size > 0:
                return True
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            marker_descriptor = os.open(".git", flags, dir_fd=directory_descriptor)
            opened = os.fstat(marker_descriptor)
            if (
                _identity(opened) != _identity(marker)
                or not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or os.read(marker_descriptor, 1)
            ):
                raise LiveKitEotInventoryError("Git boundary verification failed")
            current = os.stat(
                ".git",
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if (
                _identity(os.fstat(marker_descriptor)) != _identity(opened)
                or _identity(current) != _identity(marker)
            ):
                raise LiveKitEotInventoryError("Git boundary verification failed")
            return False
        if stat.S_ISLNK(marker.st_mode) or not stat.S_ISDIR(marker.st_mode):
            return True
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        marker_descriptor = os.open(".git", flags, dir_fd=directory_descriptor)
        opened = os.fstat(marker_descriptor)
        if _directory_identity(opened) != _directory_identity(marker):
            raise LiveKitEotInventoryError("Git boundary verification failed")
        try:
            head = os.stat("HEAD", dir_fd=marker_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            try:
                os.stat("HEAD", dir_fd=marker_descriptor, follow_symlinks=False)
            except FileNotFoundError:
                has_head = False
            else:
                raise LiveKitEotInventoryError(
                    "Git boundary verification failed"
                )
        else:
            try:
                current_head = os.stat(
                    "HEAD",
                    dir_fd=marker_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                raise LiveKitEotInventoryError(
                    "Git boundary verification failed"
                ) from None
            if _identity(current_head) != _identity(head):
                raise LiveKitEotInventoryError(
                    "Git boundary verification failed"
                )
            has_head = True
        current = os.stat(
            ".git",
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            _directory_identity(os.fstat(marker_descriptor))
            != _directory_identity(opened)
            or _directory_identity(current) != _directory_identity(marker)
        ):
            raise LiveKitEotInventoryError("Git boundary verification failed")
        return has_head
    except LiveKitEotInventoryError:
        raise
    except (OSError, OverflowError, ValueError):
        raise LiveKitEotInventoryError("Git boundary verification failed") from None
    finally:
        if marker_descriptor >= 0:
            try:
                os.close(marker_descriptor)
            except OSError:
                pass


def _has_git_ancestor(directory: Path) -> bool:
    try:
        for ancestor in (directory, *directory.parents):
            with opened_directory_nofollow(ancestor) as (stable, descriptor):
                opened = os.fstat(descriptor)
                if (
                    stable != ancestor
                    or _directory_identity(ancestor.lstat())
                    != _directory_identity(opened)
                ):
                    raise LiveKitEotInventoryError(
                        "Git boundary verification failed"
                    )
                has_marker = _git_marker_present(descriptor)
                if (
                    _directory_identity(os.fstat(descriptor))
                    != _directory_identity(opened)
                    or _directory_identity(ancestor.lstat())
                    != _directory_identity(opened)
                ):
                    raise LiveKitEotInventoryError(
                        "Git boundary verification failed"
                    )
                if has_marker:
                    return True
        return False
    except LiveKitEotInventoryError:
        raise
    except (BoundedReadError, OSError, RuntimeError, ValueError):
        raise LiveKitEotInventoryError("Git boundary verification failed") from None


def _outside_repo(path: Path) -> bool:
    return path != _REPO_ROOT and _REPO_ROOT not in path.parents


def _canonical_existing_path(path: Path | str) -> Path:
    supplied = Path(path).expanduser()
    if not supplied.is_absolute():
        raise LiveKitEotInventoryError("path prerequisite failed")
    lexical = Path(os.path.abspath(supplied))
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        raise LiveKitEotInventoryError("path prerequisite failed") from None
    if (
        resolved != lexical
        or not _outside_repo(lexical)
        or _has_git_ancestor(lexical.parent)
    ):
        raise LiveKitEotInventoryError("path prerequisite failed")
    return lexical


def _output_path(path: Path | str) -> tuple[Path, os.stat_result]:
    supplied = Path(path).expanduser()
    if not supplied.is_absolute():
        raise LiveKitEotInventoryError("output prerequisite failed")
    lexical = Path(os.path.abspath(supplied))
    if not lexical.name or lexical.name in {".", "..", ".git"}:
        raise LiveKitEotInventoryError("output prerequisite failed")
    try:
        parent = lexical.parent.resolve(strict=True)
        parent_info = parent.lstat()
    except (OSError, RuntimeError, ValueError):
        raise LiveKitEotInventoryError("output prerequisite failed") from None
    if (
        parent != lexical.parent
        or not _outside_repo(parent)
        or _has_git_ancestor(parent)
        or not stat.S_ISDIR(parent_info.st_mode)
        or stat.S_IMODE(parent_info.st_mode) & 0o077
        or (hasattr(os, "geteuid") and parent_info.st_uid != os.geteuid())
    ):
        raise LiveKitEotInventoryError("output prerequisite failed")
    try:
        lexical.lstat()
    except FileNotFoundError:
        return lexical, parent_info
    except OSError:
        raise LiveKitEotInventoryError("output prerequisite failed") from None
    raise LiveKitEotInventoryError("output prerequisite failed")


def _source_path(path: Path | str) -> tuple[Path, os.stat_result]:
    candidate = _canonical_existing_path(path)
    try:
        metadata = candidate.lstat()
    except OSError:
        raise LiveKitEotInventoryError("source prerequisite failed") from None
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size != SOURCE_SIZE_BYTES
        or stat.S_IMODE(metadata.st_mode) & 0o077
        or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
    ):
        raise LiveKitEotInventoryError("source prerequisite failed")
    return candidate, metadata


@contextmanager
def _opened_source(path: Path | str) -> Iterator[_SourceHandle]:
    descriptor = -1
    try:
        candidate, before = _source_path(path)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(candidate, flags)
        opened = os.fstat(descriptor)
        if _identity(opened) != _identity(before):
            raise LiveKitEotInventoryError("source prerequisite failed")
        handle = _SourceHandle(candidate, descriptor, _identity(opened))
        yield handle
        _verify_source(handle, hash_content=False)
    except LiveKitEotInventoryError:
        raise
    except (OSError, OverflowError, ValueError):
        raise LiveKitEotInventoryError("source prerequisite failed") from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _descriptor_sha256(descriptor: int, expected_size: int) -> str:
    try:
        digest = hashlib.sha256()
        consumed = 0
        while consumed <= expected_size:
            payload = os.pread(
                descriptor,
                min(1024 * 1024, expected_size + 1 - consumed),
                consumed,
            )
            if not payload:
                break
            consumed += len(payload)
            if consumed > expected_size:
                raise LiveKitEotInventoryError("source verification failed")
            digest.update(payload)
        if consumed != expected_size:
            raise LiveKitEotInventoryError("source verification failed")
        return digest.hexdigest()
    except LiveKitEotInventoryError:
        raise
    except (OSError, OverflowError, ValueError):
        raise LiveKitEotInventoryError("source verification failed") from None


def _verify_source(handle: _SourceHandle, *, hash_content: bool) -> None:
    try:
        current = handle.path.lstat()
        opened = os.fstat(handle.descriptor)
        if (
            _identity(current) != handle.identity
            or _identity(opened) != handle.identity
            or handle.path.resolve(strict=True) != handle.path
            or _has_git_ancestor(handle.path.parent)
            or (
                hash_content
                and _descriptor_sha256(handle.descriptor, SOURCE_SIZE_BYTES)
                != SOURCE_SHA256
            )
        ):
            raise LiveKitEotInventoryError("source verification failed")
    except LiveKitEotInventoryError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise LiveKitEotInventoryError("source verification failed") from None


def _write_private_json(path: Path, value: object) -> None:
    payload = _canonical_json_bytes(value)
    try:
        with path.open("xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        path.chmod(0o400)
    except OSError:
        raise LiveKitEotInventoryError("worker prerequisite failed") from None


def _worker_environment(root: Path) -> dict[str, str]:
    return {
        "HOME": str(root),
        "TMPDIR": str(root),
        "PATH": os.defpath,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "ARROW_NUM_THREADS": "1",
    }


@contextmanager
def _run_worker(
    handle: _SourceHandle,
    *,
    output_parent: Path,
    parquet_python: Path | str | None,
    test_worker_injection: TestWorkerInjection | None,
) -> Iterator[_WorkerRun]:
    root = Path(tempfile.mkdtemp(prefix=".speaker-livekit-eot-", dir=output_parent))
    root.chmod(0o700)
    output = root / "output"
    output.mkdir(mode=0o700)
    process: subprocess.Popen[bytes] | None = None
    group_reaped = False
    try:
        if test_worker_injection is not None:
            if (
                type(test_worker_injection) is not TestWorkerInjection
                or not callable(test_worker_injection.run)
                or _SAFE_TOKEN_RE.fullmatch(test_worker_injection.fixture) is None
            ):
                raise LiveKitEotInventoryError("worker prerequisite failed")
            try:
                test_worker_injection.run(handle.descriptor, output)
            except LiveKitEotInventoryError:
                raise
            except Exception:
                raise LiveKitEotInventoryError("worker failed") from None
            yield _WorkerRun(
                output,
                {
                    "contract": "synthetic-injected-livekit-eot-worker-v1",
                    "fixture": test_worker_injection.fixture,
                    "production_evidence": False,
                },
            )
            return

        try:
            interpreter = public_fixtures._validate_parquet_python(parquet_python)
            worker_payload = public_fixtures._read_stable_repo_worker(_WORKER_PATH)
        except public_fixtures.PublicFixtureError:
            raise LiveKitEotInventoryError("worker prerequisite failed") from None
        if not worker_payload or len(worker_payload) > _MAX_WORKER_BYTES:
            raise LiveKitEotInventoryError("worker prerequisite failed")
        worker_sha256 = hashlib.sha256(worker_payload).hexdigest()
        worker_snapshot = root / "livekit_eot_parquet_worker.py"
        public_fixtures._write_private_worker_snapshot(worker_snapshot, worker_payload)
        request_path = root / "request.json"
        _write_private_json(
            request_path,
            {
                "schema_version": PROTOCOL_VERSION,
                "source_revision": SOURCE_REVISION,
                "source_descriptor": handle.descriptor,
                "source_size_bytes": SOURCE_SIZE_BYTES,
                "source_sha256": SOURCE_SHA256,
            },
        )
        try:
            process = subprocess.Popen(
                [
                    str(interpreter),
                    "-I",
                    "-B",
                    str(worker_snapshot),
                    "--request",
                    str(request_path),
                    "--output-dir",
                    str(output),
                ],
                cwd=root,
                env=_worker_environment(root),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                pass_fds=(handle.descriptor,),
                start_new_session=True,
            )
            return_code = process.wait(timeout=_WORKER_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as exc:
            raise LiveKitEotInventoryError("worker timed out") from exc
        except OSError as exc:
            raise LiveKitEotInventoryError("worker failed") from exc
        public_fixtures._terminate_worker_process_group(process)
        group_reaped = True
        if return_code != 0:
            raise LiveKitEotInventoryError("worker failed")
        try:
            snapshot_after = public_fixtures._read_stable_worker_file(
                worker_snapshot,
                maximum=_MAX_WORKER_BYTES,
            )
            repository_after = public_fixtures._read_stable_repo_worker(_WORKER_PATH)
        except public_fixtures.PublicFixtureError:
            raise LiveKitEotInventoryError("worker changed") from None
        if (
            hashlib.sha256(snapshot_after).hexdigest() != worker_sha256
            or hashlib.sha256(repository_after).hexdigest() != worker_sha256
        ):
            raise LiveKitEotInventoryError("worker changed")
        yield _WorkerRun(
            output,
            {
                "contract": "isolated-pyarrow-livekit-eot-worker-v1",
                "protocol_version": PROTOCOL_VERSION,
                "worker_sha256": worker_sha256,
                "pyarrow_version": PYARROW_VERSION,
                "isolated_python": True,
                "single_thread_environment": True,
                "source_transport": "inherited-verified-file-descriptor",
                "production_evidence": True,
            },
        )
    except public_fixtures.PublicFixtureError:
        raise LiveKitEotInventoryError("worker failed") from None
    finally:
        if process is not None and not group_reaped:
            public_fixtures._terminate_worker_process_group(process)
        public_fixtures._unlock_private_tree(root)
        shutil.rmtree(root, ignore_errors=True)


def _private_worker_result(path: Path) -> Mapping[str, object]:
    try:
        with public_fixtures._stable_regular_reader(
            path,
            field="private aggregate worker result",
            maximum=_MAX_WORKER_RESULT_BYTES,
        ) as (handle, opened):
            if stat.S_IMODE(opened.st_mode) & 0o077:
                raise LiveKitEotInventoryError("worker result failed")
            payload = handle.read(opened.st_size + 1)
            if len(payload) != opened.st_size:
                raise LiveKitEotInventoryError("worker result failed")
    except LiveKitEotInventoryError:
        raise
    except public_fixtures.PublicFixtureError:
        raise LiveKitEotInventoryError("worker result failed") from None
    result = _strict_json(payload)
    if not isinstance(result, dict):
        raise LiveKitEotInventoryError("worker result failed")
    return result


def _positive_int(value: object, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= maximum:
        raise LiveKitEotInventoryError("worker result failed")
    return value


def _sha256(value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise LiveKitEotInventoryError("worker result failed")
    return value


def _exact_profile_count(value: object, expected: int) -> int:
    if type(value) is not int or value != expected:
        raise LiveKitEotInventoryError("worker provenance failed")
    return value


def _validate_worker_result(value: Mapping[str, object]) -> Mapping[str, object]:
    if set(value) != _WORKER_RESULT_FIELDS:
        raise LiveKitEotInventoryError("worker result failed")
    if (
        value.get("schema_version") != PROTOCOL_VERSION
        or value.get("pyarrow_version") != PYARROW_VERSION
        or value.get("source_revision") != SOURCE_REVISION
        or value.get("source_size_bytes") != SOURCE_SIZE_BYTES
        or value.get("source_sha256") != SOURCE_SHA256
        or value.get("schema_contract_sha256") != SCHEMA_CONTRACT_SHA256
        or value.get("label_convention") != LABEL_CONVENTION
        or value.get("row_count") != SOURCE_ROWS
        or value.get("audio_file_count") != SOURCE_ROWS
        or value.get("eot_label_count") != SOURCE_ROWS
        or value.get("duration_exact_sample_count")
        != DURATION_EXACT_SAMPLE_COUNT
        or value.get("duration_microsecond_quantized_count")
        != DURATION_MICROSECOND_QUANTIZED_COUNT
        or value.get("final_gap_zero_sample_count")
        != FINAL_GAP_ZERO_SAMPLE_COUNT
        or value.get("final_gap_one_sample_count")
        != FINAL_GAP_ONE_SAMPLE_COUNT
    ):
        raise LiveKitEotInventoryError("worker provenance failed")
    duration_exact = _exact_profile_count(
        value.get("duration_exact_sample_count"),
        DURATION_EXACT_SAMPLE_COUNT,
    )
    duration_quantized = _exact_profile_count(
        value.get("duration_microsecond_quantized_count"),
        DURATION_MICROSECOND_QUANTIZED_COUNT,
    )
    gap_zero = _exact_profile_count(
        value.get("final_gap_zero_sample_count"),
        FINAL_GAP_ZERO_SAMPLE_COUNT,
    )
    gap_one = _exact_profile_count(
        value.get("final_gap_one_sample_count"),
        FINAL_GAP_ONE_SAMPLE_COUNT,
    )
    row_groups = _positive_int(value.get("row_group_count"), maximum=SOURCE_ROWS)
    audio_bytes = _positive_int(
        value.get("audio_bytes_total"), maximum=512 * 1024 * 1024
    )
    audio_samples = _positive_int(
        value.get("audio_samples_total"), maximum=SOURCE_ROWS * 180 * 16_000
    )
    spans = _positive_int(
        value.get("silence_span_count"), maximum=SOURCE_ROWS * 64
    )
    hold = value.get("hold_label_count")
    if isinstance(hold, bool) or not isinstance(hold, int) or hold < 0:
        raise LiveKitEotInventoryError("worker result failed")
    minimum_eot = _positive_int(
        value.get("minimum_eot_silence_samples"), maximum=5 * 16_000
    )
    maximum_span = _positive_int(
        value.get("maximum_silence_span_samples"), maximum=5 * 16_000
    )
    if (
        hold != spans - SOURCE_ROWS
        or duration_exact + duration_quantized != SOURCE_ROWS
        or gap_zero + gap_one != SOURCE_ROWS
        or minimum_eot < int(0.2 * 16_000)
        or maximum_span < minimum_eot
    ):
        raise LiveKitEotInventoryError("worker result failed")
    for field_name in ("audio_set_sha256", "silence_set_sha256", "inventory_sha256"):
        _sha256(value.get(field_name))
    inventory = {field_name: value[field_name] for field_name in _INVENTORY_FIELDS}
    if (
        inventory["row_group_count"] != row_groups
        or inventory["audio_bytes_total"] != audio_bytes
        or inventory["audio_samples_total"] != audio_samples
    ):
        raise LiveKitEotInventoryError("worker result failed")
    return inventory


def _publish_new_private_file(
    destination: Path,
    parent_info: os.stat_result,
    payload: bytes,
    execution_closure: Mapping[str, str],
) -> str:
    parent_descriptor = -1
    temporary_name: str | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        parent_descriptor = os.open(destination.parent, flags)
        if _directory_entry_identity(
            os.fstat(parent_descriptor)
        ) != _directory_entry_identity(parent_info) or _has_git_ancestor(
            destination.parent
        ):
            raise LiveKitEotInventoryError("output publication failed")
        try:
            os.stat(destination.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise LiveKitEotInventoryError("output publication failed")
        temporary_name = f".{destination.name}.{os.urandom(12).hex()}.part"
        file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        file_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        temporary_descriptor = os.open(
            temporary_name,
            file_flags,
            0o600,
            dir_fd=parent_descriptor,
        )
        try:
            with os.fdopen(temporary_descriptor, "wb", buffering=0) as output:
                written = output.write(payload)
                if written != len(payload):
                    raise LiveKitEotInventoryError("output publication failed")
                output.flush()
                os.fsync(output.fileno())
        except Exception:
            try:
                os.close(temporary_descriptor)
            except OSError:
                pass
            raise
        temporary_read_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        temporary_read_flags |= getattr(os, "O_NOFOLLOW", 0)
        temporary_read = os.open(
            temporary_name,
            temporary_read_flags,
            dir_fd=parent_descriptor,
        )
        try:
            temporary_info = os.fstat(temporary_read)
            if (
                not stat.S_ISREG(temporary_info.st_mode)
                or temporary_info.st_nlink != 1
                or stat.S_IMODE(temporary_info.st_mode) != 0o600
                or temporary_info.st_size != len(payload)
                or (hasattr(os, "geteuid") and temporary_info.st_uid != os.geteuid())
                or _descriptor_sha256(temporary_read, len(payload))
                != hashlib.sha256(payload).hexdigest()
            ):
                raise LiveKitEotInventoryError("output publication failed")
        finally:
            os.close(temporary_read)
        if (
            _directory_entry_identity(os.fstat(parent_descriptor))
            != _directory_entry_identity(parent_info)
            or _directory_entry_identity(destination.parent.lstat())
            != _directory_entry_identity(parent_info)
            or _has_git_ancestor(destination.parent)
        ):
            raise LiveKitEotInventoryError("output publication failed")
        _rebind_execution_closure(execution_closure)
        try:
            os.link(
                temporary_name,
                destination.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            raise LiveKitEotInventoryError("output publication failed") from None
        os.unlink(temporary_name, dir_fd=parent_descriptor)
        temporary_name = None
        os.fsync(parent_descriptor)
        output_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        output_flags |= getattr(os, "O_NOFOLLOW", 0)
        published_descriptor = os.open(
            destination.name,
            output_flags,
            dir_fd=parent_descriptor,
        )
        try:
            published = os.fstat(published_descriptor)
            current = destination.lstat()
            if (
                _identity(published) != _identity(current)
                or not stat.S_ISREG(published.st_mode)
                or published.st_nlink != 1
                or stat.S_IMODE(published.st_mode) != 0o600
                or published.st_size != len(payload)
                or (hasattr(os, "geteuid") and published.st_uid != os.geteuid())
                or _descriptor_sha256(published_descriptor, len(payload))
                != hashlib.sha256(payload).hexdigest()
            ):
                raise LiveKitEotInventoryError("output publication failed")
        finally:
            os.close(published_descriptor)
        _rebind_execution_closure(execution_closure)
        return hashlib.sha256(payload).hexdigest()
    except LiveKitEotInventoryError:
        raise
    except (OSError, OverflowError, ValueError):
        raise LiveKitEotInventoryError("output publication failed") from None
    finally:
        if parent_descriptor >= 0:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name, dir_fd=parent_descriptor)
                except OSError:
                    pass
            os.close(parent_descriptor)


def inventory_livekit_eot_english(
    *,
    source_parquet: Path | str,
    parquet_python: Path | str | None,
    output: Path | str,
    accepted_terms: frozenset[str],
    test_worker_injection: TestWorkerInjection | None = None,
) -> InventoryResult:
    """Validate the exact shard and publish one aggregate-only private report."""

    if not isinstance(accepted_terms, frozenset) or LICENSE not in accepted_terms:
        raise LiveKitEotInventoryError("license acknowledgement failed")
    execution_closure = _snapshot_execution_closure()
    destination, parent_info = _output_path(output)
    with _opened_source(source_parquet) as source:
        _verify_source(source, hash_content=True)
        with _run_worker(
            source,
            output_parent=destination.parent,
            parquet_python=parquet_python,
            test_worker_injection=test_worker_injection,
        ) as worker_run:
            inventory = _validate_worker_result(
                _private_worker_result(worker_run.output_dir / "result.json")
            )
            _verify_source(source, hash_content=True)
            production_evidence = _validate_execution_worker_binding(
                worker_run.binding,
                execution_closure,
            )
            report = {
                "schema_version": REPORT_SCHEMA_VERSION,
                "kind": REPORT_KIND,
                "dataset": DATASET,
                "config": CONFIG,
                "split": SPLIT,
                "license": LICENSE,
                "attribution": {
                    "creator": "LiveKit",
                    "source_url": (
                        "https://huggingface.co/datasets/livekit/eot-bench-data"
                    ),
                    "license_url": (
                        "https://creativecommons.org/licenses/by/4.0/legalcode"
                    ),
                    "use_note": (
                        "Aggregate endpoint validation only; source bytes are not "
                        "rehosted or modified."
                    ),
                },
                "evaluation_scope": "endpoint-turn-taking-only-no-wer",
                "source_revision": SOURCE_REVISION,
                "source_size_bytes": SOURCE_SIZE_BYTES,
                "source_sha256": SOURCE_SHA256,
                "schema_contract_sha256": SCHEMA_CONTRACT_SHA256,
                "execution_closure": dict(execution_closure),
                "worker": dict(worker_run.binding),
                "inventory": dict(inventory),
                "production_evidence": production_evidence,
            }
            if set(report) != _REPORT_FIELDS:
                raise LiveKitEotInventoryError("report contract failed")
            payload = _canonical_json_bytes(report)
            if len(payload) > _MAX_REPORT_BYTES:
                raise LiveKitEotInventoryError("report contract failed")
            report_sha256 = _publish_new_private_file(
                destination,
                parent_info,
                payload,
                execution_closure,
            )
    return InventoryResult(
        destination,
        report_sha256,
        int(inventory["row_count"]),
        int(inventory["silence_span_count"]),
        production_evidence,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inventory an explicit local pinned LiveKit English EoT shard."
    )
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--parquet-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accept-license", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        result = inventory_livekit_eot_english(
            source_parquet=args.source_parquet,
            parquet_python=args.parquet_python,
            output=args.output,
            accepted_terms=frozenset(args.accept_license),
        )
    except (LiveKitEotInventoryError, OSError, ValueError):
        print(json.dumps(_SAFE_ERROR, sort_keys=True, separators=(",", ":")))
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "row_count": result.row_count,
                "silence_span_count": result.silence_span_count,
                "production_evidence": result.production_evidence,
                "report_sha256": result.report_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
