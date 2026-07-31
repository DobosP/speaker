"""Private, hash-bound source snapshots for isolated benchmark workers."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Mapping, Sequence

from .bounded_io import (
    BoundedReadError,
    opened_directory_nofollow,
    read_regular_bounded,
)
from .manifest import MAX_WORKER_BYTES


SOURCE_BUNDLE_SCHEMA_VERSION = 1
SOURCE_BUNDLE_MANIFEST = "source-bundle.json"
WORKER_RELATIVE_PATH = "tools/streaming_stt/worker.py"
WORKER_SOURCE_FILES = (
    "tools/__init__.py",
    "tools/streaming_stt/__init__.py",
    "tools/streaming_stt/adapters/__init__.py",
    "tools/streaming_stt/adapters/fake.py",
    "tools/streaming_stt/adapters/moonshine.py",
    "tools/streaming_stt/adapters/nemotron.py",
    "tools/streaming_stt/bounded_io.py",
    "tools/streaming_stt/manifest.py",
    "tools/streaming_stt/protocol.py",
    "tools/streaming_stt/runtime_receipt.py",
    "tools/streaming_stt/source_bundle.py",
    "tools/streaming_stt/supervisor.py",
    WORKER_RELATIVE_PATH,
)
_MAX_BUNDLE_MANIFEST_BYTES = 128 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ROOT_FIELDS = {"schema_version", "worker", "files"}
_FILE_FIELDS = {"path", "sha256", "size_bytes"}


class SourceBundleError(RuntimeError):
    """A source snapshot was unsafe, unstable, or did not match its receipt."""


@dataclass(frozen=True)
class BoundSource:
    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class SourceBundle:
    root: Path = field(repr=False)
    manifest_path: Path = field(repr=False)
    tree_sha256: str
    worker_relative_path: str
    files: tuple[BoundSource, ...]

    @property
    def worker_path(self) -> Path:
        return self.root / self.worker_relative_path

    @property
    def file_by_path(self) -> Mapping[str, BoundSource]:
        return {item.relative_path: item for item in self.files}


def _bad() -> object:
    raise SourceBundleError()


def _strict_json(raw: bytes) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise SourceBundleError()
            result[key] = value
        return result

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: _bad())
    except (UnicodeError, ValueError, OverflowError, SourceBundleError):
        raise SourceBundleError() from None


def _relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise SourceBundleError()
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != value
    ):
        raise SourceBundleError()
    return value


def _encoded_manifest(
    files: Sequence[BoundSource],
    *,
    worker_relative_path: str,
) -> bytes:
    payload = {
        "schema_version": SOURCE_BUNDLE_SCHEMA_VERSION,
        "worker": worker_relative_path,
        "files": [
            {
                "path": item.relative_path,
                "sha256": item.sha256,
                "size_bytes": item.size_bytes,
            }
            for item in files
        ],
    }
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _write_new_regular(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o400)
    try:
        with os.fdopen(descriptor, "wb", buffering=0) as handle:
            descriptor = -1
            handle.write(payload)
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _stage_payload_bundle(
    payloads: Mapping[str, bytes],
    destination: Path,
    *,
    worker_relative_path: str,
    required_files: Sequence[str] | None = None,
) -> SourceBundle:
    worker_relative_path = _relative_path(worker_relative_path)
    normalized_payloads = {
        _relative_path(relative): payload for relative, payload in payloads.items()
    }
    expected = (
        tuple(sorted(required_files))
        if required_files is not None
        else tuple(sorted(normalized_payloads))
    )
    if (
        not normalized_payloads
        or tuple(sorted(normalized_payloads)) != expected
        or worker_relative_path not in normalized_payloads
        or any(
            not payload or len(payload) > MAX_WORKER_BYTES
            for payload in normalized_payloads.values()
        )
    ):
        raise SourceBundleError()

    root = Path(os.path.abspath(destination))
    try:
        root.mkdir(mode=0o700)
        with opened_directory_nofollow(root, require_private=True):
            pass
    except (OSError, BoundedReadError):
        raise SourceBundleError() from None

    staged: list[BoundSource] = []
    try:
        for relative in expected:
            payload = normalized_payloads[relative]
            target = root.joinpath(*PurePosixPath(relative).parts)
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            _write_new_regular(target, payload)
            staged.append(
                BoundSource(
                    relative_path=relative,
                    sha256=hashlib.sha256(payload).hexdigest(),
                    size_bytes=len(payload),
                )
            )

        encoded = _encoded_manifest(
            staged,
            worker_relative_path=worker_relative_path,
        )
        if len(encoded) > _MAX_BUNDLE_MANIFEST_BYTES:
            raise SourceBundleError()
        manifest_path = root / SOURCE_BUNDLE_MANIFEST
        _write_new_regular(manifest_path, encoded)
        tree_sha256 = hashlib.sha256(encoded).hexdigest()
        return load_source_bundle(
            manifest_path,
            expected_tree_sha256=tree_sha256,
            required_files=expected,
        )
    except (OSError, RuntimeError, ValueError, OverflowError):
        raise SourceBundleError() from None


def stage_source_bundle(
    sources: Mapping[str, Path],
    destination: Path,
    *,
    worker_relative_path: str,
    required_files: Sequence[str] | None = None,
) -> SourceBundle:
    """Copy stable single-link source files into one new private tree."""

    payloads: dict[str, bytes] = {}
    try:
        for relative, source in sources.items():
            normalized = _relative_path(relative)
            snapshot = read_regular_bounded(
                Path(source),
                maximum_bytes=MAX_WORKER_BYTES,
            )
            payloads[normalized] = snapshot.data
    except BoundedReadError:
        raise SourceBundleError() from None
    return _stage_payload_bundle(
        payloads,
        destination,
        worker_relative_path=worker_relative_path,
        required_files=required_files,
    )


def _directory_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
    )


def _open_child_directory(parent_descriptor: int, name: str) -> int:
    before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if not stat.S_ISDIR(before.st_mode):
        raise SourceBundleError()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    if _directory_identity(os.fstat(descriptor)) != _directory_identity(before):
        os.close(descriptor)
        raise SourceBundleError()
    return descriptor


def _read_source_at(parent_descriptor: int, name: str) -> bytes:
    descriptor = -1
    try:
        before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > MAX_WORKER_BYTES
        ):
            raise SourceBundleError()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or opened.st_size != before.st_size
            or opened.st_mtime_ns != before.st_mtime_ns
            or opened.st_ctime_ns != before.st_ctime_ns
        ):
            raise SourceBundleError()
        with os.fdopen(descriptor, "rb", buffering=0) as handle:
            descriptor = -1
            payload = handle.read(MAX_WORKER_BYTES + 1)
            after = os.fstat(handle.fileno())
        current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (
            len(payload) != opened.st_size
            or len(payload) > MAX_WORKER_BYTES
            or after.st_nlink != 1
            or current.st_nlink != 1
            or (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            != (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
            )
            or (
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
                current.st_ctime_ns,
            )
            != (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
            )
        ):
            raise SourceBundleError()
        return payload
    except (OSError, RuntimeError, ValueError, OverflowError):
        raise SourceBundleError() from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _enumerate_bundle_files(
    directory_descriptor: int,
    *,
    prefix: str = "",
) -> set[str]:
    files: set[str] = set()
    for name in os.listdir(directory_descriptor):
        if not name or "/" in name or "\x00" in name:
            raise SourceBundleError()
        metadata = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        relative = f"{prefix}/{name}" if prefix else name
        if stat.S_ISDIR(metadata.st_mode):
            child = _open_child_directory(directory_descriptor, name)
            try:
                files.update(
                    _enumerate_bundle_files(
                        child,
                        prefix=relative,
                    )
                )
            finally:
                os.close(child)
        elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
            files.add(relative)
        else:
            raise SourceBundleError()
    return files


def stage_worker_source_bundle(
    repository_root: Path, destination: Path
) -> SourceBundle:
    """Stage the complete fixed source closure imported by the real worker."""

    root = Path(os.path.abspath(repository_root))
    descriptors: list[int] = []
    payloads: dict[str, bytes] = {}
    try:
        with opened_directory_nofollow(root) as (stable_root, root_descriptor):
            tools_descriptor = _open_child_directory(root_descriptor, "tools")
            descriptors.append(tools_descriptor)
            package_descriptor = _open_child_directory(
                tools_descriptor,
                "streaming_stt",
            )
            descriptors.append(package_descriptor)
            adapter_descriptor = _open_child_directory(
                package_descriptor,
                "adapters",
            )
            descriptors.append(adapter_descriptor)
            parents = {
                "tools/__init__.py": (tools_descriptor, "__init__.py"),
                "tools/streaming_stt/__init__.py": (
                    package_descriptor,
                    "__init__.py",
                ),
                "tools/streaming_stt/adapters/__init__.py": (
                    adapter_descriptor,
                    "__init__.py",
                ),
                "tools/streaming_stt/adapters/fake.py": (
                    adapter_descriptor,
                    "fake.py",
                ),
                "tools/streaming_stt/adapters/moonshine.py": (
                    adapter_descriptor,
                    "moonshine.py",
                ),
                "tools/streaming_stt/adapters/nemotron.py": (
                    adapter_descriptor,
                    "nemotron.py",
                ),
            }
            for relative in WORKER_SOURCE_FILES:
                parent_descriptor, name = parents.get(
                    relative,
                    (package_descriptor, PurePosixPath(relative).name),
                )
                payloads[relative] = _read_source_at(parent_descriptor, name)

            if (
                stable_root.lstat().st_dev != os.fstat(root_descriptor).st_dev
                or stable_root.lstat().st_ino != os.fstat(root_descriptor).st_ino
                or _directory_identity(
                    os.stat(
                        "tools",
                        dir_fd=root_descriptor,
                        follow_symlinks=False,
                    )
                )
                != _directory_identity(os.fstat(tools_descriptor))
                or _directory_identity(
                    os.stat(
                        "streaming_stt",
                        dir_fd=tools_descriptor,
                        follow_symlinks=False,
                    )
                )
                != _directory_identity(os.fstat(package_descriptor))
                or _directory_identity(
                    os.stat(
                        "adapters",
                        dir_fd=package_descriptor,
                        follow_symlinks=False,
                    )
                )
                != _directory_identity(os.fstat(adapter_descriptor))
            ):
                raise SourceBundleError()
    except (OSError, BoundedReadError):
        raise SourceBundleError() from None
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass

    return _stage_payload_bundle(
        payloads,
        destination,
        worker_relative_path=WORKER_RELATIVE_PATH,
        required_files=WORKER_SOURCE_FILES,
    )


def load_source_bundle(
    manifest_path: Path,
    *,
    expected_tree_sha256: str,
    required_files: Sequence[str] | None = None,
) -> SourceBundle:
    """Load and fully reverify one private source-tree receipt."""

    if (
        not isinstance(expected_tree_sha256, str)
        or _SHA256_RE.fullmatch(expected_tree_sha256) is None
    ):
        raise SourceBundleError()
    try:
        manifest = read_regular_bounded(
            Path(manifest_path),
            maximum_bytes=_MAX_BUNDLE_MANIFEST_BYTES,
        )
    except BoundedReadError:
        raise SourceBundleError() from None
    raw = manifest.data
    if hashlib.sha256(raw).hexdigest() != expected_tree_sha256:
        raise SourceBundleError()
    value = _strict_json(raw)
    if (
        not isinstance(value, dict)
        or set(value) != _ROOT_FIELDS
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") != SOURCE_BUNDLE_SCHEMA_VERSION
    ):
        raise SourceBundleError()
    worker_relative_path = _relative_path(value.get("worker"))
    raw_files = value.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise SourceBundleError()

    files: list[BoundSource] = []
    for raw_file in raw_files:
        if not isinstance(raw_file, dict) or set(raw_file) != _FILE_FIELDS:
            raise SourceBundleError()
        relative = _relative_path(raw_file.get("path"))
        sha256 = raw_file.get("sha256")
        size_bytes = raw_file.get("size_bytes")
        if (
            not isinstance(sha256, str)
            or _SHA256_RE.fullmatch(sha256) is None
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes <= 0
            or size_bytes > MAX_WORKER_BYTES
        ):
            raise SourceBundleError()
        files.append(BoundSource(relative, sha256, size_bytes))

    relative_paths = tuple(item.relative_path for item in files)
    expected = (
        tuple(sorted(required_files))
        if required_files is not None
        else tuple(sorted(relative_paths))
    )
    if (
        relative_paths != expected
        or len(set(relative_paths)) != len(relative_paths)
        or worker_relative_path not in relative_paths
    ):
        raise SourceBundleError()

    root = manifest.path.parent
    try:
        with opened_directory_nofollow(
            root,
            require_private=True,
        ) as (_stable_root, root_descriptor):
            discovered = _enumerate_bundle_files(root_descriptor)
        if discovered != {SOURCE_BUNDLE_MANIFEST, *relative_paths}:
            raise SourceBundleError()
        for item in files:
            snapshot = read_regular_bounded(
                root.joinpath(*PurePosixPath(item.relative_path).parts),
                maximum_bytes=MAX_WORKER_BYTES,
                expected_bytes=item.size_bytes,
            )
            snapshot.path.relative_to(root)
            if hashlib.sha256(snapshot.data).hexdigest() != item.sha256:
                raise SourceBundleError()
    except (OSError, RuntimeError, ValueError, BoundedReadError):
        raise SourceBundleError() from None

    return SourceBundle(
        root=root,
        manifest_path=manifest.path,
        tree_sha256=expected_tree_sha256,
        worker_relative_path=worker_relative_path,
        files=tuple(files),
    )


def verify_source_bundle(
    bundle: SourceBundle,
    *,
    required_files: Sequence[str] | None = None,
) -> None:
    """Reject any bundle path, receipt, or source file changed after staging."""

    current = load_source_bundle(
        bundle.manifest_path,
        expected_tree_sha256=bundle.tree_sha256,
        required_files=required_files,
    )
    if current != bundle:
        raise SourceBundleError()
