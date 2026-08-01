"""Isolated JSONL worker for the streaming-STT benchmark.

Candidate recognizers remain lazy and execute only after source, manifest,
runtime, interpreter, and model receipts have been verified.
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
    "tools/streaming_stt/adapters/faster_whisper_endpoint.py",
    "tools/streaming_stt/adapters/moonshine.py",
    "tools/streaming_stt/adapters/nemotron.py",
    "tools/streaming_stt/adapters/parakeet_realtime_eou.py",
    "tools/streaming_stt/adapters/zipformer.py",
    "tools/streaming_stt/bounded_io.py",
    "tools/streaming_stt/manifest.py",
    "tools/streaming_stt/protocol.py",
    "tools/streaming_stt/runtime_receipt.py",
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
from tools.streaming_stt.adapters.fake import FakeAdapterError  # noqa: E402
from tools.streaming_stt.bounded_io import (  # noqa: E402
    BoundedReadError,
    hash_regular_bounded,
    opened_directory_nofollow,
    read_regular_bounded,
)
from tools.streaming_stt.manifest import (  # noqa: E402
    FASTER_WHISPER_ENDPOINT_ADAPTER,
    FASTER_WHISPER_REQUIRED_MODEL_FILES,
    FASTER_WHISPER_VOCABULARY_FILES,
    MAX_PYTHON_BYTES,
    ManifestError,
    MOONSHINE_ADAPTER,
    NEMOTRON_ADAPTER,
    PARAKEET_REALTIME_EOU_ADAPTER,
    SHERPA_ZIPFORMER_ADAPTER,
    FasterWhisperEndpointConfig,
    NemotronConfig,
    ParakeetRealtimeEouConfig,
    WorkerManifest,
    load_worker_manifest,
)
from tools.streaming_stt.protocol import (  # noqa: E402
    MAX_CORPUS_BYTES,
    MAX_LINE_BYTES,
    MAX_MODEL_PADDING_SAMPLES,
    MAX_PARTIALS_PER_CASE,
    MAX_PCM_BYTES,
    MAX_STREAM_SAMPLES,
    NATIVE_ENDPOINT_PROTOCOL_VERSION,
    PROTOCOL_VERSION,
    ProtocolError,
    ShutdownRequest,
    TranscribeRequest,
    encode_message,
    parse_request,
)
from tools.streaming_stt.runtime_receipt import (  # noqa: E402
    FASTER_WHISPER_MODEL_TREE_LIMITS,
    FASTER_WHISPER_RUNTIME_TREE_LIMITS,
    NEMOTRON_RUNTIME_TREE_LIMITS,
    PARAKEET_RUNTIME_TREE_LIMITS,
    RuntimeTreeReceipt,
    load_runtime_tree_receipt,
    verify_moonshine_wheel_install,
    verify_venv_runtime_location,
)
from tools.streaming_stt.source_bundle import (  # noqa: E402
    WORKER_SOURCE_FILES,
    load_source_bundle,
    verify_source_bundle,
)


def _wire_protocol_version(manifest: WorkerManifest) -> int:
    return (
        NATIVE_ENDPOINT_PROTOCOL_VERSION
        if manifest.adapter == PARAKEET_REALTIME_EOU_ADAPTER
        else PROTOCOL_VERSION
    )


def _safe_fatal(protocol_version: int) -> dict[str, object]:
    return {
        "v": protocol_version,
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


def _native_final_payload(
    request: TranscribeRequest,
    case,
    *,
    partial_count: int,
) -> dict[str, object]:
    """Validate and serialize exact adapter-owned native endpoint evidence."""

    integer_fields = {
        "source_samples": getattr(case, "source_samples", None),
        "source_samples_consumed": getattr(case, "source_samples_consumed", None),
        "declared_tail_samples": getattr(case, "declared_tail_samples", None),
        "tail_samples_consumed": getattr(case, "tail_samples_consumed", None),
        "chunks_yielded": getattr(case, "chunks_yielded", None),
        "model_padding_samples": getattr(case, "model_padding_samples", None),
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in integer_fields.values()
    ):
        raise ProtocolError()
    source_seen = integer_fields["source_samples_consumed"]
    tail_seen = integer_fields["tail_samples_consumed"]
    source_samples = integer_fields["source_samples"]
    declared_tail_samples = integer_fields["declared_tail_samples"]
    chunks = integer_fields["chunks_yielded"]
    model_padding = integer_fields["model_padding_samples"]
    assert isinstance(source_seen, int)
    assert isinstance(tail_seen, int)
    assert isinstance(source_samples, int)
    assert isinstance(declared_tail_samples, int)
    assert isinstance(chunks, int)
    assert isinstance(model_padding, int)
    samples_seen = source_seen + tail_seen
    reason = getattr(case, "endpoint_reason", None)
    native = getattr(case, "native_endpoint", None)
    probability = getattr(case, "endpoint_probability", None)
    endpoint_sample = getattr(case, "endpoint_sample", None)
    endpoint_latency_ms = getattr(case, "endpoint_latency_ms", None)
    authoritative = getattr(case, "authoritative", None)
    expected_chunks = (
        samples_seen + request.stream.chunk_samples - 1
    ) // request.stream.chunk_samples
    expected_model_padding = (
        expected_chunks * request.stream.chunk_samples - samples_seen
    )
    expected_endpoint_sample = samples_seen + expected_model_padding
    if (
        source_samples != request.pcm.samples
        or declared_tail_samples != request.stream.tail_padding_samples
        or source_seen > request.pcm.samples
        or tail_seen > request.stream.tail_padding_samples
        or (source_seen < request.pcm.samples and tail_seen != 0)
        or samples_seen <= 0
        or chunks != expected_chunks
        or model_padding != expected_model_padding
        or model_padding > MAX_MODEL_PADDING_SAMPLES
        or type(native) is not bool
        or type(authoritative) is not bool
    ):
        raise ProtocolError()
    expected_authoritative = (
        native and reason == "eou" and source_seen == request.pcm.samples
    )
    expected_latency = (
        (tail_seen + expected_model_padding) * 1000.0 / request.pcm.sample_rate
        if expected_authoritative
        else None
    )
    if native:
        if (
            reason not in {"eou", "eob"}
            or isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(float(probability))
            or not 0.0 <= float(probability) <= 1.0
            or endpoint_sample != expected_endpoint_sample
            or authoritative != expected_authoritative
            or (expected_latency is None and endpoint_latency_ms is not None)
            or (
                expected_latency is not None
                and (
                    not isinstance(endpoint_latency_ms, (int, float))
                    or isinstance(endpoint_latency_ms, bool)
                    or abs(float(endpoint_latency_ms) - expected_latency) > 1e-6
                )
            )
        ):
            raise ProtocolError()
    elif (
        reason != "tail_exhausted"
        or probability is not None
        or endpoint_sample is not None
        or endpoint_latency_ms is not None
        or authoritative
        or source_seen != request.pcm.samples
        or tail_seen != request.stream.tail_padding_samples
    ):
        raise ProtocolError()
    return {
        "v": NATIVE_ENDPOINT_PROTOCOL_VERSION,
        "id": request.request_id,
        "type": "native_final",
        "seq": partial_count,
        "text": case.final,
        "samples_seen": samples_seen,
        "elapsed_ms": case.elapsed_ms,
        "finalization_ms": case.finalization_ms,
        "compute_ms": case.compute_ms,
        "audio_seconds": request.pcm.samples / request.pcm.sample_rate,
        "chunks": chunks,
        "deadline_misses": case.deadline_misses,
        "max_backlog_ms": case.max_backlog_ms,
        "model_padding_samples": model_padding,
        "resources": _resource_dict(case.resources),
        "source_samples": request.pcm.samples,
        "source_samples_consumed": source_seen,
        "declared_tail_samples": request.stream.tail_padding_samples,
        "tail_samples_consumed": tail_seen,
        "endpoint_reason": reason,
        "native_endpoint": native,
        "endpoint_probability": probability,
        "endpoint_sample": endpoint_sample,
        "endpoint_latency_ms": endpoint_latency_ms,
        "authoritative": authoritative,
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


def _verify_faster_whisper_model_receipt(
    manifest: WorkerManifest,
) -> RuntimeTreeReceipt:
    artifact = manifest.artifact_by_name.get("model-receipt")
    config = manifest.adapter_config
    if artifact is None or not isinstance(config, FasterWhisperEndpointConfig):
        raise ManifestError()
    try:
        receipt = load_runtime_tree_receipt(
            artifact.path,
            expected_digest=artifact.sha256,
            limits=FASTER_WHISPER_MODEL_TREE_LIMITS,
        )
    except RuntimeError:
        raise ManifestError() from None
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
        raise ManifestError()
    return receipt


def _verify_runtime_receipt(
    manifest: WorkerManifest,
) -> RuntimeTreeReceipt | None:
    if manifest.adapter == "fake-json-v1":
        return None
    if manifest.adapter == SHERPA_ZIPFORMER_ADAPTER:
        return None
    if manifest.adapter not in {
        FASTER_WHISPER_ENDPOINT_ADAPTER,
        MOONSHINE_ADAPTER,
        NEMOTRON_ADAPTER,
        PARAKEET_REALTIME_EOU_ADAPTER,
    }:
        raise ManifestError()
    artifact = manifest.artifact_by_name.get("runtime-receipt")
    if artifact is None:
        raise ManifestError()
    limits = {
        MOONSHINE_ADAPTER: None,
        NEMOTRON_ADAPTER: NEMOTRON_RUNTIME_TREE_LIMITS,
        PARAKEET_REALTIME_EOU_ADAPTER: PARAKEET_RUNTIME_TREE_LIMITS,
        FASTER_WHISPER_ENDPOINT_ADAPTER: FASTER_WHISPER_RUNTIME_TREE_LIMITS,
    }[manifest.adapter]
    try:
        receipt = load_runtime_tree_receipt(
            artifact.path,
            expected_digest=artifact.sha256,
            **({} if limits is None else {"limits": limits}),
        )
        verify_venv_runtime_location(receipt, manifest.python.path)
        if manifest.adapter == MOONSHINE_ADAPTER:
            wheel = manifest.artifact_by_name.get("release-wheel")
            if wheel is None:
                raise ManifestError()
            verify_moonshine_wheel_install(
                receipt,
                wheel.path,
                expected_sha256=wheel.sha256,
                expected_size_bytes=wheel.size_bytes,
            )
    except RuntimeError:
        raise ManifestError() from None
    if (
        receipt.digest != artifact.sha256
        or Path(os.path.abspath(sys.executable)) != manifest.python.path
    ):
        raise ManifestError()
    if manifest.adapter in {NEMOTRON_ADAPTER, PARAKEET_REALTIME_EOU_ADAPTER}:
        config = manifest.adapter_config
        wheel_lock = manifest.artifact_by_name.get("runtime-wheel-lock")
        maximum_file = max(
            (item.size_bytes for item in receipt.files),
            default=0,
        )
        if (
            not isinstance(config, (NemotronConfig, ParakeetRealtimeEouConfig))
            or wheel_lock is None
            or wheel_lock.sha256 != config.wheel_lock_sha256
            or receipt.content_digest != config.runtime_content_sha256
            or receipt.file_count != config.runtime_file_count
            or receipt.total_size_bytes != config.runtime_total_size_bytes
            or maximum_file != config.runtime_maximum_file_bytes
        ):
            raise ManifestError()
        _verify_python312_venv_layout(manifest, receipt)
    if manifest.adapter == FASTER_WHISPER_ENDPOINT_ADAPTER:
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
            raise ManifestError()
        model_receipt = _verify_faster_whisper_model_receipt(manifest)
        try:
            receipt.root.relative_to(model_receipt.root)
        except ValueError:
            pass
        else:
            raise ManifestError()
        try:
            model_receipt.root.relative_to(receipt.root)
        except ValueError:
            pass
        else:
            raise ManifestError()
    return receipt


def _verify_python312_venv_layout(
    manifest: WorkerManifest,
    receipt: RuntimeTreeReceipt,
) -> None:
    root = manifest.python.path.parent.parent
    python_root = receipt.root.parent
    try:
        if (
            set(os.listdir(root)) != {"bin", "lib", "pyvenv.cfg"}
            or set(os.listdir(root / "bin")) != {"python"}
            or set(os.listdir(root / "lib")) != {"python3.12"}
            or set(os.listdir(python_root)) != {"site-packages"}
            or not manifest.python.path.is_symlink()
            or manifest.python.path.resolve(strict=True).name != "python3.12"
        ):
            raise ManifestError()
    except (OSError, RuntimeError, ValueError, ManifestError):
        raise ManifestError() from None


def _verify_worker_state(
    manifest: WorkerManifest,
    source_bundle,
) -> RuntimeTreeReceipt | None:
    current = load_worker_manifest(manifest.path)
    if current != manifest:
        raise ManifestError()
    verify_source_bundle(
        source_bundle,
        required_files=WORKER_SOURCE_FILES,
    )
    interpreter = hash_regular_bounded(
        Path(sys.executable),
        maximum_bytes=MAX_PYTHON_BYTES,
        expected_bytes=manifest.python.size_bytes,
        allow_final_symlink=True,
    )
    if interpreter.sha256 != manifest.python.sha256:
        raise ManifestError()
    return _verify_runtime_receipt(manifest)


def _verify_candidate_python_version(manifest: WorkerManifest) -> None:
    if manifest.adapter not in {
        FASTER_WHISPER_ENDPOINT_ADAPTER,
        NEMOTRON_ADAPTER,
        PARAKEET_REALTIME_EOU_ADAPTER,
    }:
        return
    config = manifest.adapter_config
    if (
        not isinstance(
            config,
            (
                FasterWhisperEndpointConfig,
                NemotronConfig,
                ParakeetRealtimeEouConfig,
            ),
        )
        or platform.python_version() != config.python_version
    ):
        raise ManifestError()


def _verify_nemotron_python_version(manifest: WorkerManifest) -> None:
    """Retain the existing test seam while applying the shared GPU check."""

    _verify_candidate_python_version(manifest)


def _activate_candidate_runtime(
    manifest: WorkerManifest,
    receipt: RuntimeTreeReceipt | None,
) -> None:
    if manifest.adapter == "fake-json-v1":
        if receipt is not None:
            raise ManifestError()
        return
    if manifest.adapter == SHERPA_ZIPFORMER_ADAPTER:
        module_prefixes = ("numpy", "sherpa_onnx", "_sherpa_onnx")
        if (
            receipt is not None
            or sys.flags.no_site != 0
            or sys.flags.no_user_site != 1
            or sys.flags.ignore_environment != 1
            or sys.flags.isolated != 1
            or sys.flags.dont_write_bytecode != 1
            or any(
                any(
                    name == prefix or name.startswith(prefix + ".")
                    for prefix in module_prefixes
                )
                for name in sys.modules
            )
        ):
            raise ManifestError()
        return
    if manifest.adapter not in {
        FASTER_WHISPER_ENDPOINT_ADAPTER,
        MOONSHINE_ADAPTER,
        NEMOTRON_ADAPTER,
        PARAKEET_REALTIME_EOU_ADAPTER,
    }:
        raise ManifestError()
    if manifest.adapter == FASTER_WHISPER_ENDPOINT_ADAPTER:
        module_prefixes = (
            "av",
            "ctranslate2",
            "faster_whisper",
            "huggingface_hub",
            "nvidia",
            "numpy",
            "tokenizers",
        )
    elif manifest.adapter == MOONSHINE_ADAPTER:
        module_prefixes = ("moonshine_voice",)
    elif manifest.adapter == NEMOTRON_ADAPTER:
        module_prefixes = (
            "librosa",
            "llvmlite",
            "numba",
            "numpy",
            "scipy",
            "sklearn",
            "soundfile",
            "soxr",
            "torch",
            "transformers",
            "tokenizers",
            "safetensors",
            "huggingface_hub",
        )
    else:
        module_prefixes = (
            "hydra",
            "lightning",
            "librosa",
            "lhotse",
            "nemo",
            "numpy",
            "omegaconf",
            "sentencepiece",
            "soundfile",
            "torch",
            "torchaudio",
            "transformers",
        )
    if (
        receipt is None
        or sys.flags.no_site != 1
        or sys.flags.no_user_site != 1
        or sys.flags.ignore_environment != 1
        or sys.flags.isolated != 1
        or sys.flags.dont_write_bytecode != 1
        or not sys.path
        or any(
            any(
                name == prefix or name.startswith(prefix + ".")
                for prefix in module_prefixes
            )
            for name in sys.modules
        )
    ):
        raise ManifestError()
    runtime_root = str(receipt.root)
    if runtime_root in sys.path:
        raise ManifestError()
    # Direct insertion does not process .pth files or sitecustomize. The
    # verified source bundle retains import precedence at index zero.
    sys.path.insert(1, runtime_root)


def _create_adapter(manifest: WorkerManifest):
    if manifest.adapter == "fake-json-v1":
        try:
            artifact = manifest.artifact_by_name["fake-script"]
        except KeyError:
            raise ManifestError() from None
        return (
            FakeJsonAdapter(
                artifact.path,
                expected_sha256=artifact.sha256,
                expected_size_bytes=artifact.size_bytes,
            ),
            FakeAdapterError,
        )
    if manifest.adapter == FASTER_WHISPER_ENDPOINT_ADAPTER:
        if not isinstance(manifest.adapter_config, FasterWhisperEndpointConfig):
            raise ManifestError()
        model_receipt = _verify_faster_whisper_model_receipt(manifest)
        from tools.streaming_stt.adapters.faster_whisper_endpoint import (  # noqa: PLC0415
            FasterWhisperEndpointAdapter,
            FasterWhisperEndpointAdapterError,
        )

        return (
            FasterWhisperEndpointAdapter(
                model_receipt.root,
                config=manifest.adapter_config,
            ),
            FasterWhisperEndpointAdapterError,
        )
    model_artifacts = tuple(
        artifact
        for artifact in manifest.artifacts
        if artifact.name.startswith("model-")
    )
    model_roots = {artifact.path.parent for artifact in model_artifacts}
    if len(model_roots) != 1:
        raise ManifestError()
    model_root = model_roots.pop()
    if manifest.adapter in {NEMOTRON_ADAPTER, PARAKEET_REALTIME_EOU_ADAPTER}:
        try:
            expected_model_files = {artifact.path.name for artifact in model_artifacts}
            if set(os.listdir(model_root)) != expected_model_files:
                raise ManifestError()
        except (OSError, RuntimeError, ValueError, ManifestError):
            raise ManifestError() from None
    if manifest.adapter == MOONSHINE_ADAPTER:
        if manifest.adapter_config is None:
            raise ManifestError()
        # This verified source imports the external package only here.
        from tools.streaming_stt.adapters.moonshine import (  # noqa: PLC0415
            MoonshineAdapter,
            MoonshineAdapterError,
        )

        return (
            MoonshineAdapter(model_root, config=manifest.adapter_config),
            MoonshineAdapterError,
        )
    if manifest.adapter == NEMOTRON_ADAPTER:
        if manifest.adapter_config is None:
            raise ManifestError()
        from tools.streaming_stt.adapters.nemotron import (  # noqa: PLC0415
            NemotronAdapter,
            NemotronAdapterError,
        )

        return (
            NemotronAdapter(model_root, config=manifest.adapter_config),
            NemotronAdapterError,
        )
    if manifest.adapter == SHERPA_ZIPFORMER_ADAPTER:
        if manifest.adapter_config is None:
            raise ManifestError()
        from tools.streaming_stt.adapters.zipformer import (  # noqa: PLC0415
            SherpaZipformerAdapter,
            SherpaZipformerAdapterError,
        )

        return (
            SherpaZipformerAdapter(model_root, config=manifest.adapter_config),
            SherpaZipformerAdapterError,
        )
    if manifest.adapter == PARAKEET_REALTIME_EOU_ADAPTER:
        if not isinstance(manifest.adapter_config, ParakeetRealtimeEouConfig):
            raise ManifestError()
        if len(model_artifacts) != 1:
            raise ManifestError()
        from tools.streaming_stt.adapters.parakeet_realtime_eou import (  # noqa: PLC0415
            ParakeetRealtimeEouAdapter,
            ParakeetRealtimeEouAdapterError,
        )

        return (
            ParakeetRealtimeEouAdapter(
                model_artifacts[0].path,
                config=manifest.adapter_config,
            ),
            ParakeetRealtimeEouAdapterError,
        )
    raise ManifestError()


def _close_adapter(adapter) -> None:
    close = getattr(adapter, "close", None)
    if close is not None:
        close()


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
    _verify_candidate_python_version(manifest)
    protocol_version = _wire_protocol_version(manifest)
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
    runtime_receipt = _verify_worker_state(manifest, source_bundle)
    _activate_candidate_runtime(manifest, runtime_receipt)
    final_only_endpoint = manifest.adapter == FASTER_WHISPER_ENDPOINT_ADAPTER
    if final_only_endpoint and manifest.schema_version != 6:
        raise ManifestError()
    adapter, adapter_error = _create_adapter(manifest)
    _write(
        {
            "v": protocol_version,
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
    adapter_closed = False
    try:
        while True:
            raw = sys.stdin.buffer.readline(MAX_LINE_BYTES + 1)
            if not raw:
                return 0
            try:
                request = parse_request(raw)
            except ProtocolError:
                _write(_safe_fatal(protocol_version))
                return 2
            if request.protocol_version != protocol_version:
                _write(_safe_fatal(protocol_version))
                return 2
            try:
                if isinstance(request, ShutdownRequest):
                    _close_adapter(adapter)
                    adapter_closed = True
                    _verify_worker_state(manifest, source_bundle)
                    _write(
                        {
                            "v": protocol_version,
                            "id": request.request_id,
                            "type": "shutdown",
                        }
                    )
                    _write({"v": protocol_version, "type": "bye"})
                    return 0

                pcm = _load_pcm(request, scratch_root)
                if request.pcm.sha256 not in consumed_inputs:
                    consumed_inputs.add(request.pcm.sha256)
                    consumed_bytes += len(pcm)
                if consumed_bytes > MAX_CORPUS_BYTES:
                    raise ProtocolError()

                partial_count = 0

                def emit_partial(_adapter_seq: int, partial) -> None:
                    nonlocal partial_count
                    if final_only_endpoint or partial_count >= MAX_PARTIALS_PER_CASE:
                        raise ProtocolError()
                    _write(
                        {
                            "v": protocol_version,
                            "id": request.request_id,
                            "type": "partial",
                            "seq": partial_count,
                            "text": partial.text,
                            "samples_seen": partial.after_samples,
                            "elapsed_ms": partial.elapsed_ms,
                            "decode_ms": partial.decode_ms,
                        }
                    )
                    partial_count += 1

                case = adapter.transcribe(request, emit_partial=emit_partial)
                if len(case.partials) != partial_count or (
                    final_only_endpoint and partial_count != 0
                ):
                    raise ProtocolError()
                if manifest.adapter == PARAKEET_REALTIME_EOU_ADAPTER:
                    _write(
                        _native_final_payload(
                            request,
                            case,
                            partial_count=partial_count,
                        )
                    )
                    continue
                model_padding_samples = getattr(case, "model_padding_samples", 0)
                if (
                    isinstance(model_padding_samples, bool)
                    or not isinstance(model_padding_samples, int)
                    or model_padding_samples < 0
                    or model_padding_samples > MAX_MODEL_PADDING_SAMPLES
                ):
                    raise ProtocolError()
                total_samples = (
                    request.pcm.samples + request.stream.tail_padding_samples
                )
                chunks = (
                    total_samples + request.stream.chunk_samples - 1
                ) // request.stream.chunk_samples
                if manifest.adapter == NEMOTRON_ADAPTER:
                    chunks = getattr(case, "chunks_yielded", None)
                    if (
                        isinstance(chunks, bool)
                        or not isinstance(chunks, int)
                        or not 1 <= chunks <= MAX_STREAM_SAMPLES
                    ):
                        raise ProtocolError()
                _write(
                    {
                        "v": protocol_version,
                        "id": request.request_id,
                        "type": "final",
                        "seq": 0 if final_only_endpoint else partial_count,
                        "text": case.final,
                        "samples_seen": total_samples,
                        "elapsed_ms": case.elapsed_ms,
                        "finalization_ms": case.finalization_ms,
                        "compute_ms": case.compute_ms,
                        "audio_seconds": request.pcm.samples / 16_000,
                        "chunks": chunks,
                        "deadline_misses": case.deadline_misses,
                        "max_backlog_ms": case.max_backlog_ms,
                        "model_padding_samples": model_padding_samples,
                        "resources": _resource_dict(case.resources),
                    }
                )
            except (adapter_error, ProtocolError):
                fatal = manifest.adapter != "fake-json-v1"
                _write(
                    {
                        "v": protocol_version,
                        "id": request.request_id,
                        "type": "error",
                        "code": ("candidate_failure" if fatal else "invalid_case"),
                        "fatal": fatal,
                    }
                )
                if fatal:
                    return 2
    finally:
        if not adapter_closed:
            _close_adapter(adapter)


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
            _write(_safe_fatal(PROTOCOL_VERSION))
        except Exception:  # noqa: BLE001 - the protocol itself may be unavailable
            pass
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
