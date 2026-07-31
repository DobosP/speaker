"""Isolated JSONL worker for the streaming-STT benchmark.

Only the deterministic fake adapter is present in this first landing. Real
recognizer imports will live behind this process boundary in later branches.
"""

from __future__ import annotations

from array import array
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import sys
from typing import Mapping


_BOOTSTRAP_MAX_MANIFEST_BYTES = 128 * 1024
_BOOTSTRAP_MAX_SOURCE_BYTES = 4 * 1024 * 1024
_BOOTSTRAP_BUNDLE_MANIFEST = "source-bundle.json"
_BOOTSTRAP_WORKER_PATH = "tools/streaming_stt/worker.py"
_BOOTSTRAP_SOURCE_FILES = (
    "tools/__init__.py",
    "tools/streaming_stt/__init__.py",
    "tools/streaming_stt/adapters/__init__.py",
    "tools/streaming_stt/adapters/fake.py",
    "tools/streaming_stt/bounded_io.py",
    "tools/streaming_stt/manifest.py",
    "tools/streaming_stt/protocol.py",
    "tools/streaming_stt/source_bundle.py",
    "tools/streaming_stt/supervisor.py",
    _BOOTSTRAP_WORKER_PATH,
)
_BOOTSTRAP_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _bootstrap_argument(name: str) -> str:
    try:
        index = sys.argv.index(name)
        value = sys.argv[index + 1]
    except (ValueError, IndexError):
        raise RuntimeError from None
    if not value or "\x00" in value:
        raise RuntimeError
    return value


def _bootstrap_json(raw: bytes) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise RuntimeError
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(RuntimeError()),
        )
    except (UnicodeError, ValueError, OverflowError, RuntimeError):
        raise RuntimeError from None


def _bootstrap_read_descriptor(descriptor: int, *, maximum_bytes: int) -> bytes:
    before = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size <= 0
        or before.st_size > maximum_bytes
    ):
        raise RuntimeError
    payload = bytearray()
    while len(payload) <= maximum_bytes:
        chunk = os.read(
            descriptor,
            min(1024 * 1024, maximum_bytes + 1 - len(payload)),
        )
        if not chunk:
            break
        payload.extend(chunk)
    after = os.fstat(descriptor)
    if (
        len(payload) != before.st_size
        or len(payload) > maximum_bytes
        or after.st_nlink != 1
        or (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        != (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
    ):
        raise RuntimeError
    return bytes(payload)


def _bootstrap_read_at(
    root_descriptor: int,
    relative_path: str,
    *,
    maximum_bytes: int,
) -> bytes:
    parts = relative_path.split("/")
    directory_descriptor = os.dup(root_descriptor)
    file_descriptor = -1
    try:
        directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        directory_flags |= getattr(os, "O_DIRECTORY", 0)
        directory_flags |= getattr(os, "O_NOFOLLOW", 0)
        for part in parts[:-1]:
            before = os.stat(
                part,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(before.st_mode):
                raise RuntimeError
            child = os.open(
                part,
                directory_flags,
                dir_fd=directory_descriptor,
            )
            opened = os.fstat(child)
            if (
                opened.st_dev,
                opened.st_ino,
                stat.S_IFMT(opened.st_mode),
            ) != (
                before.st_dev,
                before.st_ino,
                stat.S_IFMT(before.st_mode),
            ):
                os.close(child)
                raise RuntimeError
            os.close(directory_descriptor)
            directory_descriptor = child

        name = parts[-1]
        before = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        file_descriptor = os.open(name, flags, dir_fd=directory_descriptor)
        opened = os.fstat(file_descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
            )
            != (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
        ):
            raise RuntimeError
        payload = _bootstrap_read_descriptor(
            file_descriptor,
            maximum_bytes=maximum_bytes,
        )
        current = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
            current.st_nlink,
        ) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            1,
        ):
            raise RuntimeError
        return payload
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        os.close(directory_descriptor)


def _bootstrap_read_path(path: Path, *, maximum_bytes: int) -> bytes:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise RuntimeError
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    opened = os.fstat(descriptor)
    if (
        opened.st_dev,
        opened.st_ino,
        opened.st_size,
        opened.st_mtime_ns,
        opened.st_ctime_ns,
    ) != (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ):
        os.close(descriptor)
        raise RuntimeError
    try:
        payload = _bootstrap_read_descriptor(
            descriptor,
            maximum_bytes=maximum_bytes,
        )
        current = path.lstat()
        if (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
            current.st_nlink,
        ) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            1,
        ):
            raise RuntimeError
        return payload
    finally:
        os.close(descriptor)


def _bootstrap_tree_files(
    directory_descriptor: int,
    *,
    prefix: str = "",
) -> set[str]:
    files: set[str] = set()
    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    for name in os.listdir(directory_descriptor):
        metadata = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        relative = f"{prefix}/{name}" if prefix else name
        if stat.S_ISDIR(metadata.st_mode):
            child = os.open(name, directory_flags, dir_fd=directory_descriptor)
            try:
                files.update(_bootstrap_tree_files(child, prefix=relative))
            finally:
                os.close(child)
        elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
            files.add(relative)
        else:
            raise RuntimeError
    return files


def _bootstrap_source_bundle() -> tuple[int, str, Path]:
    try:
        descriptor = int(_bootstrap_argument("--source-bundle-fd"))
        expected_digest = _bootstrap_argument("--source-bundle-sha256")
        manifest_path = Path(_bootstrap_argument("--manifest"))
    except (ValueError, OverflowError):
        raise RuntimeError from None
    if (
        descriptor < 0
        or _BOOTSTRAP_SHA256_RE.fullmatch(expected_digest) is None
        or not manifest_path.is_absolute()
    ):
        raise RuntimeError
    root_metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or stat.S_IMODE(root_metadata.st_mode) & 0o077
        or (hasattr(os, "geteuid") and root_metadata.st_uid != os.geteuid())
    ):
        raise RuntimeError

    receipt = _bootstrap_read_at(
        descriptor,
        _BOOTSTRAP_BUNDLE_MANIFEST,
        maximum_bytes=_BOOTSTRAP_MAX_MANIFEST_BYTES,
    )
    if hashlib.sha256(receipt).hexdigest() != expected_digest:
        raise RuntimeError
    value = _bootstrap_json(receipt)
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "worker", "files"}
        or value.get("schema_version") != 1
        or type(value.get("schema_version")) is not int
        or value.get("worker") != _BOOTSTRAP_WORKER_PATH
        or not isinstance(value.get("files"), list)
    ):
        raise RuntimeError
    entries = value["files"]
    assert isinstance(entries, list)
    if len(entries) != len(_BOOTSTRAP_SOURCE_FILES):
        raise RuntimeError
    file_receipts: dict[str, tuple[str, int]] = {}
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "sha256", "size_bytes"}
            or entry.get("path") not in _BOOTSTRAP_SOURCE_FILES
            or not isinstance(entry.get("sha256"), str)
            or _BOOTSTRAP_SHA256_RE.fullmatch(entry["sha256"]) is None
            or isinstance(entry.get("size_bytes"), bool)
            or not isinstance(entry.get("size_bytes"), int)
            or not 0 < entry["size_bytes"] <= _BOOTSTRAP_MAX_SOURCE_BYTES
            or entry["path"] in file_receipts
        ):
            raise RuntimeError
        file_receipts[entry["path"]] = (
            entry["sha256"],
            entry["size_bytes"],
        )
    if tuple(file_receipts) != _BOOTSTRAP_SOURCE_FILES:
        raise RuntimeError
    if _bootstrap_tree_files(descriptor) != {
        _BOOTSTRAP_BUNDLE_MANIFEST,
        *_BOOTSTRAP_SOURCE_FILES,
    }:
        raise RuntimeError
    for relative_path, (expected_sha256, expected_size) in file_receipts.items():
        payload = _bootstrap_read_at(
            descriptor,
            relative_path,
            maximum_bytes=_BOOTSTRAP_MAX_SOURCE_BYTES,
        )
        if (
            len(payload) != expected_size
            or hashlib.sha256(payload).hexdigest() != expected_sha256
        ):
            raise RuntimeError

    manifest_raw = _bootstrap_read_path(
        manifest_path,
        maximum_bytes=_BOOTSTRAP_MAX_MANIFEST_BYTES,
    )
    manifest_value = _bootstrap_json(manifest_raw)
    if not isinstance(manifest_value, dict):
        raise RuntimeError
    worker_receipt = manifest_value.get("worker")
    if (
        not isinstance(worker_receipt, dict)
        or set(worker_receipt) != {"path", "sha256", "size_bytes"}
        or (
            worker_receipt.get("sha256"),
            worker_receipt.get("size_bytes"),
        )
        != file_receipts[_BOOTSTRAP_WORKER_PATH]
    ):
        raise RuntimeError
    return descriptor, expected_digest, Path(f"/proc/self/fd/{descriptor}")


# Executing the worker verifies the inherited private tree before any project
# package can execute. A normal module import is intentionally inert so the
# repository import-smoke contract can inspect this file; imported callers
# cannot enter ``main`` because the launch-bound values remain absent.
if __name__ == "__main__":
    _SOURCE_BUNDLE_FD, _SOURCE_BUNDLE_SHA256, _REPO_ROOT = _bootstrap_source_bundle()
    sys.path.insert(0, str(_REPO_ROOT))
else:
    _SOURCE_BUNDLE_FD = None
    _SOURCE_BUNDLE_SHA256 = None

from tools.streaming_stt.adapters import FakeJsonAdapter  # noqa: E402
from tools.streaming_stt.adapters.fake import FakeAdapterError, FakePartial  # noqa: E402
from tools.streaming_stt.bounded_io import (  # noqa: E402
    BoundedReadError,
    hash_regular_bounded,
    opened_directory_nofollow,
    read_regular_bounded,
)
from tools.streaming_stt.manifest import (  # noqa: E402
    MAX_PYTHON_BYTES,
    ManifestError,
    load_worker_manifest,
)
from tools.streaming_stt.protocol import (  # noqa: E402
    MAX_CORPUS_BYTES,
    MAX_LINE_BYTES,
    MAX_PCM_BYTES,
    PROTOCOL_VERSION,
    ProtocolError,
    ShutdownRequest,
    TranscribeRequest,
    encode_message,
    parse_request,
)
from tools.streaming_stt.source_bundle import (  # noqa: E402
    WORKER_SOURCE_FILES,
    load_source_bundle,
    verify_source_bundle,
)


_SAFE_FATAL = {
    "v": PROTOCOL_VERSION,
    "id": "invalid",
    "type": "error",
    "code": "invalid_request",
    "fatal": True,
}


def _write(value: Mapping[str, object]) -> None:
    sys.stdout.buffer.write(encode_message(value))
    sys.stdout.buffer.flush()


def _resource_dict(value) -> dict[str, object]:
    return {
        "rss_mb": value.rss_mb,
        "threads": value.threads,
        "vram_mb": value.vram_mb,
    }


def _load_pcm(request: TranscribeRequest, scratch_root: Path) -> bytes:
    try:
        snapshot = read_regular_bounded(
            request.pcm.path,
            maximum_bytes=MAX_PCM_BYTES,
            expected_bytes=request.pcm.size_bytes,
        )
        snapshot.path.relative_to(scratch_root)
        raw = snapshot.data
    except (OSError, RuntimeError, ValueError, BoundedReadError):
        raise ProtocolError() from None
    if (
        len(raw) != request.pcm.size_bytes
        or hashlib.sha256(raw).hexdigest() != request.pcm.sha256
    ):
        raise ProtocolError()
    samples = array("f")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    if len(samples) != request.pcm.samples or any(
        not math.isfinite(sample) for sample in samples
    ):
        raise ProtocolError()
    return raw


def _run(
    manifest_path: Path,
    scratch_root: Path,
    *,
    source_bundle_manifest: Path,
    source_bundle_sha256: str,
) -> int:
    source_bundle = load_source_bundle(
        source_bundle_manifest,
        expected_tree_sha256=source_bundle_sha256,
        required_files=WORKER_SOURCE_FILES,
    )
    manifest = load_worker_manifest(manifest_path)
    staged_worker = source_bundle.file_by_path.get(source_bundle.worker_relative_path)
    if (
        source_bundle.tree_sha256 != _SOURCE_BUNDLE_SHA256
        or source_bundle.worker_path.resolve(strict=True)
        != Path(__file__).resolve(strict=True)
        or staged_worker is None
        or staged_worker.sha256 != manifest.worker.sha256
        or staged_worker.size_bytes != manifest.worker.size_bytes
        or hash_regular_bounded(
            Path(sys.executable),
            maximum_bytes=MAX_PYTHON_BYTES,
            expected_bytes=manifest.python.size_bytes,
            allow_final_symlink=True,
        ).sha256
        != manifest.python.sha256
    ):
        raise ManifestError()
    try:
        artifact = manifest.artifact_by_name["fake-script"]
    except KeyError:
        raise ManifestError() from None
    adapter = FakeJsonAdapter(
        artifact.path.resolve(strict=True),
        expected_sha256=artifact.sha256,
        expected_size_bytes=artifact.size_bytes,
    )
    _write(
        {
            "v": PROTOCOL_VERSION,
            "type": "ready",
            "model_id": manifest.model_id,
            "manifest_sha256": manifest.digest,
            "source_bundle_sha256": source_bundle.tree_sha256,
            "adapter": manifest.adapter,
            "model_load_ms": adapter.ready.model_load_ms,
            "resources": _resource_dict(adapter.ready.resources),
            "runtime": {
                "python": platform.python_version(),
                "platform": sys.platform,
            },
        }
    )

    consumed_bytes = 0
    consumed_inputs: set[str] = set()
    while True:
        raw = sys.stdin.buffer.readline(MAX_LINE_BYTES + 1)
        if not raw:
            return 0
        try:
            request = parse_request(raw)
        except ProtocolError:
            _write(_SAFE_FATAL)
            return 2
        try:
            if isinstance(request, ShutdownRequest):
                verify_source_bundle(
                    source_bundle,
                    required_files=WORKER_SOURCE_FILES,
                )
                _write(
                    {
                        "v": PROTOCOL_VERSION,
                        "id": request.request_id,
                        "type": "shutdown",
                    }
                )
                _write({"v": PROTOCOL_VERSION, "type": "bye"})
                return 0

            pcm = _load_pcm(request, scratch_root)
            if request.pcm.sha256 not in consumed_inputs:
                consumed_inputs.add(request.pcm.sha256)
                consumed_bytes += len(pcm)
            if consumed_bytes > MAX_CORPUS_BYTES:
                raise ProtocolError()

            def emit_partial(seq: int, partial: FakePartial) -> None:
                _write(
                    {
                        "v": PROTOCOL_VERSION,
                        "id": request.request_id,
                        "type": "partial",
                        "seq": seq,
                        "text": partial.text,
                        "samples_seen": partial.after_samples,
                        "elapsed_ms": partial.elapsed_ms,
                        "decode_ms": partial.decode_ms,
                    }
                )

            case = adapter.transcribe(request, emit_partial=emit_partial)
            total_samples = request.pcm.samples + request.stream.tail_padding_samples
            chunks = (
                total_samples + request.stream.chunk_samples - 1
            ) // request.stream.chunk_samples
            _write(
                {
                    "v": PROTOCOL_VERSION,
                    "id": request.request_id,
                    "type": "final",
                    "seq": len(case.partials),
                    "text": case.final,
                    "samples_seen": total_samples,
                    "elapsed_ms": case.elapsed_ms,
                    "finalization_ms": case.finalization_ms,
                    "compute_ms": case.compute_ms,
                    "audio_seconds": request.pcm.samples / 16_000,
                    "chunks": chunks,
                    "deadline_misses": case.deadline_misses,
                    "max_backlog_ms": case.max_backlog_ms,
                    "resources": _resource_dict(case.resources),
                }
            )
        except (FakeAdapterError, ProtocolError):
            request_id = request.request_id
            _write(
                {
                    "v": PROTOCOL_VERSION,
                    "id": request_id,
                    "type": "error",
                    "code": "invalid_case",
                    "fatal": False,
                }
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--source-bundle-manifest", type=Path, required=True)
    parser.add_argument("--source-bundle-sha256", required=True)
    parser.add_argument("--source-bundle-fd", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        if (
            args.source_bundle_fd != _SOURCE_BUNDLE_FD
            or args.source_bundle_sha256 != _SOURCE_BUNDLE_SHA256
        ):
            raise ProtocolError()
        lexical_scratch = Path(os.path.abspath(args.scratch_root))
        try:
            with opened_directory_nofollow(
                lexical_scratch,
                require_private=True,
            ) as (scratch, _descriptor):
                pass
        except BoundedReadError:
            raise ProtocolError() from None
        return _run(
            args.manifest,
            scratch,
            source_bundle_manifest=args.source_bundle_manifest,
            source_bundle_sha256=args.source_bundle_sha256,
        )
    except Exception:  # noqa: BLE001 - worker stderr/stdout stay detail-free
        try:
            _write(_SAFE_FATAL)
        except Exception:  # noqa: BLE001 - the protocol itself may be unavailable
            pass
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
