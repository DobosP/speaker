"""Exact, bounded receipts for private candidate runtime trees.

The generator is a provisioning helper for trusted local input. Loading and
verification are fail-closed because a receipt may be supplied to a benchmark
worker long after provisioning.
"""

from __future__ import annotations

import base64
import binascii
import csv
from dataclasses import dataclass, field
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from types import MappingProxyType
from typing import Mapping
import zipfile

from .bounded_io import (
    BoundedReadError,
    opened_directory_nofollow,
    read_regular_bounded,
)


RUNTIME_TREE_RECEIPT_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ROOT_FIELDS = {"schema_version", "root", "files"}
_FILE_FIELDS = {"path", "sha256", "size_bytes"}
_HASH_CHUNK_BYTES = 1024 * 1024
_MAX_VENV_MARKER_BYTES = 64 * 1024
_MAX_RELEASE_WHEEL_BYTES = 128 * 1024 * 1024
_MAX_WHEEL_RECORD_BYTES = 64 * 1024
_MOONSHINE_DIST_INFO = "moonshine_voice-0.1.0.dist-info"
_MOONSHINE_WHEEL_RECORD = f"{_MOONSHINE_DIST_INFO}/RECORD"
_MOONSHINE_REQUIRED_WHEEL_FILES = {
    "moonshine_voice/__init__.py",
    "moonshine_voice/libmoonshine.so",
    "moonshine_voice.libs/libonnxruntime-13ab8084.so.1",
    f"{_MOONSHINE_DIST_INFO}/METADATA",
    f"{_MOONSHINE_DIST_INFO}/WHEEL",
    f"{_MOONSHINE_DIST_INFO}/entry_points.txt",
    f"{_MOONSHINE_DIST_INFO}/top_level.txt",
    f"{_MOONSHINE_DIST_INFO}/licenses/LICENSE",
}


class RuntimeTreeReceiptError(RuntimeError):
    """A detail-free runtime receipt or tree verification failure."""


@dataclass(frozen=True)
class RuntimeTreeLimits:
    """Resource ceilings applied before or while inspecting a runtime tree."""

    maximum_receipt_bytes: int = 4 * 1024 * 1024
    maximum_files: int = 4096
    maximum_directories: int = 4096
    maximum_file_bytes: int = 128 * 1024 * 1024
    maximum_aggregate_bytes: int = 512 * 1024 * 1024
    maximum_relative_path_bytes: int = 1024
    maximum_depth: int = 32


DEFAULT_RUNTIME_TREE_LIMITS = RuntimeTreeLimits()
NEMOTRON_RUNTIME_TREE_LIMITS = RuntimeTreeLimits(
    maximum_receipt_bytes=32 * 1024 * 1024,
    maximum_files=50_000,
    maximum_directories=10_000,
    maximum_file_bytes=2 * 1024 * 1024 * 1024,
    maximum_aggregate_bytes=12 * 1024 * 1024 * 1024,
    maximum_relative_path_bytes=1024,
    maximum_depth=32,
)
PARAKEET_RUNTIME_TREE_LIMITS = RuntimeTreeLimits(
    maximum_receipt_bytes=32 * 1024 * 1024,
    maximum_files=100_000,
    maximum_directories=20_000,
    maximum_file_bytes=4 * 1024 * 1024 * 1024,
    maximum_aggregate_bytes=20 * 1024 * 1024 * 1024,
    maximum_relative_path_bytes=1024,
    maximum_depth=32,
)


@dataclass(frozen=True)
class RuntimeFileReceipt:
    """One relative, content-bound regular file in a runtime tree."""

    relative_path: str
    sha256: str
    size_bytes: int

    @property
    def path(self) -> str:
        """Schema-compatible alias for the canonical relative path."""

        return self.relative_path


@dataclass(frozen=True)
class RuntimeTreeReceipt:
    """One immutable verified runtime-tree receipt."""

    receipt_path: Path = field(repr=False)
    digest: str
    root: Path = field(repr=False)
    files: tuple[RuntimeFileReceipt, ...]
    _mapping: Mapping[str, RuntimeFileReceipt] = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_mapping",
            MappingProxyType({item.relative_path: item for item in self.files}),
        )

    @property
    def file_by_path(self) -> Mapping[str, RuntimeFileReceipt]:
        return self._mapping

    @property
    def mapping(self) -> Mapping[str, RuntimeFileReceipt]:
        return self._mapping

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def total_size_bytes(self) -> int:
        return sum(item.size_bytes for item in self.files)

    @property
    def content_digest(self) -> str:
        """Root-independent digest of the exact ordered runtime file closure."""

        payload = [
            {
                "path": item.relative_path,
                "sha256": item.sha256,
                "size_bytes": item.size_bytes,
            }
            for item in self.files
        ]
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass
class _ScanState:
    limits: RuntimeTreeLimits
    files: list[RuntimeFileReceipt] = field(default_factory=list)
    directories: set[str] = field(default_factory=set)
    entry_count: int = 0
    aggregate_bytes: int = 0


def _fail() -> object:
    raise RuntimeTreeReceiptError()


def _validated_limits(limits: RuntimeTreeLimits) -> RuntimeTreeLimits:
    if not isinstance(limits, RuntimeTreeLimits):
        raise RuntimeTreeReceiptError()
    values = (
        limits.maximum_receipt_bytes,
        limits.maximum_files,
        limits.maximum_directories,
        limits.maximum_file_bytes,
        limits.maximum_aggregate_bytes,
        limits.maximum_relative_path_bytes,
        limits.maximum_depth,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in values
    ):
        raise RuntimeTreeReceiptError()
    return limits


def _strict_json(raw: bytes) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise RuntimeTreeReceiptError()
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda _value: _fail(),
        )
    except (
        UnicodeError,
        ValueError,
        OverflowError,
        RuntimeTreeReceiptError,
    ):
        raise RuntimeTreeReceiptError() from None


def _sha256(value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RuntimeTreeReceiptError()
    return value


def verify_isolated_venv_marker(
    path: Path | str,
    *,
    expected_digest: str | None = None,
    expected_bytes: int | None = None,
) -> None:
    """Require an exact virtualenv marker with system packages disabled."""

    if expected_digest is not None:
        expected_digest = _sha256(expected_digest)
    try:
        snapshot = read_regular_bounded(
            Path(path),
            maximum_bytes=_MAX_VENV_MARKER_BYTES,
            expected_bytes=expected_bytes,
        )
        text = snapshot.data.decode("utf-8", errors="strict")
    except (UnicodeError, ValueError, BoundedReadError):
        raise RuntimeTreeReceiptError() from None
    if "\x00" in text or (
        expected_digest is not None
        and hashlib.sha256(snapshot.data).hexdigest() != expected_digest
    ):
        raise RuntimeTreeReceiptError()
    settings: list[str] = []
    for raw_line in text.splitlines():
        if "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        if key.strip().casefold() == "include-system-site-packages":
            settings.append(value.strip().casefold())
    if settings != ["false"]:
        raise RuntimeTreeReceiptError()


def _bounded_size(value: object, *, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > maximum
    ):
        raise RuntimeTreeReceiptError()
    return value


def _relative_path(value: object, *, limits: RuntimeTreeLimits) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise RuntimeTreeReceiptError()
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeError:
        raise RuntimeTreeReceiptError() from None
    pure = PurePosixPath(value)
    if (
        len(encoded) > limits.maximum_relative_path_bytes
        or pure.is_absolute()
        or len(pure.parts) > limits.maximum_depth
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != value
    ):
        raise RuntimeTreeReceiptError()
    return value


def _record_sha256(value: str) -> str:
    prefix = "sha256="
    if not value.startswith(prefix):
        raise RuntimeTreeReceiptError()
    encoded = value[len(prefix) :].encode("ascii", errors="strict")
    padding = b"=" * ((4 - len(encoded) % 4) % 4)
    try:
        digest = base64.b64decode(
            encoded + padding,
            altchars=b"-_",
            validate=True,
        )
    except (binascii.Error, ValueError):
        raise RuntimeTreeReceiptError() from None
    if len(digest) != hashlib.sha256().digest_size:
        raise RuntimeTreeReceiptError()
    return digest.hex()


def verify_moonshine_wheel_install(
    receipt: RuntimeTreeReceipt,
    wheel_path: Path | str,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
) -> None:
    """Tie installed Moonshine-owned files to the exact release wheel."""

    if (
        not isinstance(receipt, RuntimeTreeReceipt)
        or isinstance(expected_size_bytes, bool)
        or not isinstance(expected_size_bytes, int)
        or not 0 < expected_size_bytes <= _MAX_RELEASE_WHEEL_BYTES
    ):
        raise RuntimeTreeReceiptError()
    expected_sha256 = _sha256(expected_sha256)
    try:
        snapshot = read_regular_bounded(
            Path(wheel_path),
            maximum_bytes=_MAX_RELEASE_WHEEL_BYTES,
            expected_bytes=expected_size_bytes,
        )
    except BoundedReadError:
        raise RuntimeTreeReceiptError() from None
    if hashlib.sha256(snapshot.data).hexdigest() != expected_sha256:
        raise RuntimeTreeReceiptError()

    try:
        with zipfile.ZipFile(io.BytesIO(snapshot.data), mode="r") as archive:
            infos: dict[str, zipfile.ZipInfo] = {}
            for info in archive.infolist():
                if (
                    not isinstance(info.filename, str)
                    or not info.filename
                    or "\x00" in info.filename
                    or info.filename in infos
                ):
                    raise RuntimeTreeReceiptError()
                infos[info.filename] = info
            record_info = infos.get(_MOONSHINE_WHEEL_RECORD)
            if (
                record_info is None
                or record_info.is_dir()
                or record_info.file_size <= 0
                or record_info.file_size > _MAX_WHEEL_RECORD_BYTES
            ):
                raise RuntimeTreeReceiptError()
            record_raw = archive.read(record_info)
    except (
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
        zipfile.BadZipFile,
        RuntimeTreeReceiptError,
    ):
        raise RuntimeTreeReceiptError() from None
    if len(record_raw) != record_info.file_size:
        raise RuntimeTreeReceiptError()

    expected: dict[str, tuple[str, int]] = {}
    release_record_rows: dict[str, tuple[str, str]] = {}
    try:
        rows = csv.reader(
            io.StringIO(record_raw.decode("utf-8", errors="strict"), newline=""),
            strict=True,
        )
        record_seen = False
        for row in rows:
            if len(row) != 3:
                raise RuntimeTreeReceiptError()
            relative = _relative_path(row[0], limits=DEFAULT_RUNTIME_TREE_LIMITS)
            if relative == _MOONSHINE_WHEEL_RECORD:
                if record_seen or row[1:] != ["", ""]:
                    raise RuntimeTreeReceiptError()
                record_seen = True
                release_record_rows[relative] = ("", "")
                continue
            if relative in expected:
                raise RuntimeTreeReceiptError()
            size = _bounded_size(
                int(row[2]),
                maximum=DEFAULT_RUNTIME_TREE_LIMITS.maximum_file_bytes,
            )
            info = infos.get(relative)
            if info is None or info.is_dir() or info.file_size != size:
                raise RuntimeTreeReceiptError()
            expected[relative] = (_record_sha256(row[1]), size)
            release_record_rows[relative] = (row[1], row[2])
    except (
        UnicodeError,
        ValueError,
        OverflowError,
        csv.Error,
        RuntimeTreeReceiptError,
    ):
        raise RuntimeTreeReceiptError() from None
    installed_record = receipt.file_by_path.get(_MOONSHINE_WHEEL_RECORD)
    if (
        not record_seen
        or not _MOONSHINE_REQUIRED_WHEEL_FILES.issubset(expected)
        or installed_record is None
    ):
        raise RuntimeTreeReceiptError()
    try:
        installed_snapshot = read_regular_bounded(
            receipt.root / _MOONSHINE_WHEEL_RECORD,
            maximum_bytes=_MAX_WHEEL_RECORD_BYTES,
            expected_bytes=installed_record.size_bytes,
        )
        if (
            installed_record.sha256
            != hashlib.sha256(installed_snapshot.data).hexdigest()
        ):
            raise RuntimeTreeReceiptError()
        installed_rows: dict[str, tuple[str, str]] = {}
        rows = csv.reader(
            io.StringIO(
                installed_snapshot.data.decode("utf-8", errors="strict"),
                newline="",
            ),
            strict=True,
        )
        for row in rows:
            if (
                len(row) != 3
                or not row[0]
                or "\x00" in row[0]
                or row[0] in installed_rows
            ):
                raise RuntimeTreeReceiptError()
            installed_rows[row[0]] = (row[1], row[2])
    except (
        UnicodeError,
        ValueError,
        csv.Error,
        BoundedReadError,
        RuntimeTreeReceiptError,
    ):
        raise RuntimeTreeReceiptError() from None
    if any(
        installed_rows.get(relative) != fields
        for relative, fields in release_record_rows.items()
    ):
        raise RuntimeTreeReceiptError()

    expected_owned = {
        path
        for path in expected
        if path.startswith(("moonshine_voice/", "moonshine_voice.libs/"))
    }
    installed_owned = {
        path
        for path in receipt.file_by_path
        if path.startswith(("moonshine_voice/", "moonshine_voice.libs/"))
    }
    shadowing = {
        path
        for path in receipt.file_by_path
        if "/" not in path
        and (path == "moonshine_voice" or path.startswith("moonshine_voice."))
    }
    if installed_owned != expected_owned or shadowing:
        raise RuntimeTreeReceiptError()
    for relative, (sha256, size_bytes) in expected.items():
        installed = receipt.file_by_path.get(relative)
        if (
            installed is None
            or installed.sha256 != sha256
            or installed.size_bytes != size_bytes
        ):
            raise RuntimeTreeReceiptError()


def _absolute_root(value: object) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise RuntimeTreeReceiptError()
    candidate = Path(value)
    if (
        not candidate.is_absolute()
        or candidate.as_posix() != value
        or Path(os.path.abspath(candidate)) != candidate
    ):
        raise RuntimeTreeReceiptError()
    return candidate


def _metadata_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        stat.S_IMODE(metadata.st_mode),
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _directory_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return _metadata_identity(metadata)


def _open_child_directory(
    parent_descriptor: int, name: str
) -> tuple[int, tuple[int, ...]]:
    descriptor = -1
    try:
        before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if not stat.S_ISDIR(before.st_mode):
            raise RuntimeTreeReceiptError()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        opened = os.fstat(descriptor)
        identity = _directory_identity(opened)
        if not stat.S_ISDIR(opened.st_mode) or identity != _directory_identity(before):
            raise RuntimeTreeReceiptError()
        return descriptor, identity
    except (OSError, ValueError, OverflowError, RuntimeTreeReceiptError):
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise RuntimeTreeReceiptError() from None


def _hash_file_at(
    parent_descriptor: int,
    name: str,
    relative_path: str,
    *,
    state: _ScanState,
) -> RuntimeFileReceipt:
    descriptor = -1
    try:
        before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        size = _bounded_size(
            before.st_size,
            maximum=state.limits.maximum_file_bytes,
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or state.aggregate_bytes + size > state.limits.maximum_aggregate_bytes
        ):
            raise RuntimeTreeReceiptError()

        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        opened = os.fstat(descriptor)
        opened_identity = _metadata_identity(opened)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened_identity != _metadata_identity(before)
        ):
            raise RuntimeTreeReceiptError()

        digest = hashlib.sha256()
        remaining = size
        while remaining:
            chunk = os.read(descriptor, min(_HASH_CHUNK_BYTES, remaining))
            if not chunk:
                raise RuntimeTreeReceiptError()
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeTreeReceiptError()

        after = os.fstat(descriptor)
        current = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            after.st_nlink != 1
            or current.st_nlink != 1
            or _metadata_identity(after) != opened_identity
            or _metadata_identity(current) != opened_identity
        ):
            raise RuntimeTreeReceiptError()
        state.aggregate_bytes += size
        return RuntimeFileReceipt(
            relative_path=relative_path,
            sha256=digest.hexdigest(),
            size_bytes=size,
        )
    except (
        OSError,
        ValueError,
        OverflowError,
        RuntimeTreeReceiptError,
    ):
        raise RuntimeTreeReceiptError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _directory_names(
    directory_descriptor: int,
    *,
    state: _ScanState,
) -> list[str]:
    names: list[str] = []
    maximum_entries = state.limits.maximum_files + state.limits.maximum_directories
    try:
        with os.scandir(directory_descriptor) as entries:
            for entry in entries:
                name = entry.name
                if (
                    not isinstance(name, str)
                    or not name
                    or name in {".", ".."}
                    or "/" in name
                    or "\x00" in name
                ):
                    raise RuntimeTreeReceiptError()
                state.entry_count += 1
                if state.entry_count > maximum_entries:
                    raise RuntimeTreeReceiptError()
                names.append(name)
    except (OSError, ValueError, OverflowError, RuntimeTreeReceiptError):
        raise RuntimeTreeReceiptError() from None
    names.sort()
    return names


def _scan_directory(
    directory_descriptor: int,
    *,
    prefix: str,
    state: _ScanState,
) -> None:
    opened_identity = _directory_identity(os.fstat(directory_descriptor))
    for name in _directory_names(directory_descriptor, state=state):
        relative = f"{prefix}/{name}" if prefix else name
        relative = _relative_path(relative, limits=state.limits)
        try:
            metadata = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if stat.S_ISDIR(metadata.st_mode):
                if len(state.directories) >= state.limits.maximum_directories:
                    raise RuntimeTreeReceiptError()
                child_descriptor, child_identity = _open_child_directory(
                    directory_descriptor,
                    name,
                )
                state.directories.add(relative)
                try:
                    _scan_directory(
                        child_descriptor,
                        prefix=relative,
                        state=state,
                    )
                    if (
                        _directory_identity(os.fstat(child_descriptor))
                        != child_identity
                    ):
                        raise RuntimeTreeReceiptError()
                finally:
                    os.close(child_descriptor)
                current = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if _directory_identity(current) != child_identity:
                    raise RuntimeTreeReceiptError()
            elif stat.S_ISREG(metadata.st_mode):
                if len(state.files) >= state.limits.maximum_files:
                    raise RuntimeTreeReceiptError()
                state.files.append(
                    _hash_file_at(
                        directory_descriptor,
                        name,
                        relative,
                        state=state,
                    )
                )
            else:
                raise RuntimeTreeReceiptError()
        except (
            OSError,
            ValueError,
            OverflowError,
            RuntimeTreeReceiptError,
        ):
            raise RuntimeTreeReceiptError() from None

    if _directory_identity(os.fstat(directory_descriptor)) != opened_identity:
        raise RuntimeTreeReceiptError()


def _expected_directories(
    files: tuple[RuntimeFileReceipt, ...],
) -> set[str]:
    expected: set[str] = set()
    for item in files:
        parts = PurePosixPath(item.relative_path).parts
        for end in range(1, len(parts)):
            expected.add("/".join(parts[:end]))
    return expected


def _scan_runtime_tree(
    root: Path,
    *,
    limits: RuntimeTreeLimits,
) -> tuple[Path, tuple[RuntimeFileReceipt, ...]]:
    state = _ScanState(limits=limits)
    try:
        with opened_directory_nofollow(
            root,
            require_private=True,
        ) as (stable_root, root_descriptor):
            root_identity = _directory_identity(os.fstat(root_descriptor))
            _scan_directory(root_descriptor, prefix="", state=state)
            current = stable_root.lstat()
            if (
                _directory_identity(os.fstat(root_descriptor)) != root_identity
                or _directory_identity(current) != root_identity
                or stable_root.resolve(strict=True) != stable_root
            ):
                raise RuntimeTreeReceiptError()
        files = tuple(sorted(state.files, key=lambda item: item.relative_path))
        paths = tuple(item.relative_path for item in files)
        if (
            not files
            or len(paths) != len(set(paths))
            or state.directories != _expected_directories(files)
        ):
            raise RuntimeTreeReceiptError()
        return stable_root, files
    except (
        OSError,
        RuntimeError,
        ValueError,
        OverflowError,
        BoundedReadError,
        RuntimeTreeReceiptError,
    ):
        raise RuntimeTreeReceiptError() from None


def _parse_files(
    value: object,
    *,
    limits: RuntimeTreeLimits,
) -> tuple[RuntimeFileReceipt, ...]:
    if not isinstance(value, list) or not value or len(value) > limits.maximum_files:
        raise RuntimeTreeReceiptError()
    files: list[RuntimeFileReceipt] = []
    aggregate_bytes = 0
    for raw_file in value:
        if not isinstance(raw_file, dict) or set(raw_file) != _FILE_FIELDS:
            raise RuntimeTreeReceiptError()
        relative = _relative_path(raw_file.get("path"), limits=limits)
        size = _bounded_size(
            raw_file.get("size_bytes"),
            maximum=limits.maximum_file_bytes,
        )
        aggregate_bytes += size
        if aggregate_bytes > limits.maximum_aggregate_bytes:
            raise RuntimeTreeReceiptError()
        files.append(
            RuntimeFileReceipt(
                relative_path=relative,
                sha256=_sha256(raw_file.get("sha256")),
                size_bytes=size,
            )
        )

    paths = tuple(item.relative_path for item in files)
    if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
        raise RuntimeTreeReceiptError()
    return tuple(files)


def _encoded_receipt(
    root: Path,
    files: tuple[RuntimeFileReceipt, ...],
) -> bytes:
    payload = {
        "schema_version": RUNTIME_TREE_RECEIPT_SCHEMA_VERSION,
        "root": root.as_posix(),
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


def _private_receipt(path: Path) -> None:
    try:
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
        ):
            raise RuntimeTreeReceiptError()
    except (OSError, ValueError, OverflowError, RuntimeTreeReceiptError):
        raise RuntimeTreeReceiptError() from None


def load_runtime_tree_receipt(
    path: Path | str,
    *,
    expected_digest: str | None = None,
    limits: RuntimeTreeLimits = DEFAULT_RUNTIME_TREE_LIMITS,
) -> RuntimeTreeReceipt:
    """Load a receipt and verify exact equality with its private runtime tree."""

    limits = _validated_limits(limits)
    if expected_digest is not None:
        expected_digest = _sha256(expected_digest)
    candidate = Path(os.path.abspath(Path(path).expanduser()))
    try:
        with opened_directory_nofollow(candidate.parent):
            pass
        snapshot = read_regular_bounded(
            candidate,
            maximum_bytes=limits.maximum_receipt_bytes,
        )
        if snapshot.path != candidate:
            raise RuntimeTreeReceiptError()
        _private_receipt(snapshot.path)
    except (OSError, ValueError, BoundedReadError, RuntimeTreeReceiptError):
        raise RuntimeTreeReceiptError() from None

    digest = hashlib.sha256(snapshot.data).hexdigest()
    if expected_digest is not None and digest != expected_digest:
        raise RuntimeTreeReceiptError()
    value = _strict_json(snapshot.data)
    if (
        not isinstance(value, dict)
        or set(value) != _ROOT_FIELDS
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") != RUNTIME_TREE_RECEIPT_SCHEMA_VERSION
    ):
        raise RuntimeTreeReceiptError()
    root = _absolute_root(value.get("root"))
    files = _parse_files(value.get("files"), limits=limits)
    stable_root, actual_files = _scan_runtime_tree(root, limits=limits)
    if actual_files != files:
        raise RuntimeTreeReceiptError()
    return RuntimeTreeReceipt(
        receipt_path=snapshot.path,
        digest=digest,
        root=stable_root,
        files=files,
    )


def verify_runtime_tree_receipt(
    receipt: RuntimeTreeReceipt,
    *,
    limits: RuntimeTreeLimits = DEFAULT_RUNTIME_TREE_LIMITS,
) -> None:
    """Recheck the receipt bytes, root topology, identities, and file hashes."""

    if not isinstance(receipt, RuntimeTreeReceipt):
        raise RuntimeTreeReceiptError()
    current = load_runtime_tree_receipt(
        receipt.receipt_path,
        expected_digest=receipt.digest,
        limits=limits,
    )
    if current != receipt:
        raise RuntimeTreeReceiptError()


def verify_venv_runtime_location(
    receipt: RuntimeTreeReceipt,
    python_path: Path | str,
) -> None:
    """Bind a received site-packages tree to the launched virtual environment."""

    if not isinstance(receipt, RuntimeTreeReceipt):
        raise RuntimeTreeReceiptError()
    lexical_python = Path(python_path)
    canonical_python = Path(os.path.abspath(lexical_python))
    if (
        not lexical_python.is_absolute()
        or lexical_python != canonical_python
        or canonical_python.parent.name != "bin"
    ):
        raise RuntimeTreeReceiptError()
    venv_root = canonical_python.parent.parent
    try:
        relative = receipt.root.relative_to(venv_root)
    except ValueError:
        raise RuntimeTreeReceiptError() from None
    if (
        venv_root.resolve(strict=True) != venv_root
        or len(relative.parts) != 3
        or relative.parts[0] != "lib"
        or re.fullmatch(r"python[0-9]+\.[0-9]+", relative.parts[1]) is None
        or relative.parts[2] != "site-packages"
    ):
        raise RuntimeTreeReceiptError()


def _write_new_private_receipt(path: Path, payload: bytes) -> None:
    descriptor = -1
    try:
        with opened_directory_nofollow(path.parent) as (
            _stable_parent,
            parent_descriptor,
        ):
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(
                path.name,
                flags,
                0o600,
                dir_fd=parent_descriptor,
            )
            os.fchmod(descriptor, 0o600)
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
            ):
                raise RuntimeTreeReceiptError()
            with os.fdopen(descriptor, "wb", buffering=0) as handle:
                descriptor = -1
                handle.write(payload)
                os.fsync(handle.fileno())
            os.fsync(parent_descriptor)
    except (
        OSError,
        ValueError,
        OverflowError,
        BoundedReadError,
        RuntimeTreeReceiptError,
    ):
        raise RuntimeTreeReceiptError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def generate_runtime_tree_receipt(
    root: Path | str,
    destination: Path | str,
    *,
    limits: RuntimeTreeLimits = DEFAULT_RUNTIME_TREE_LIMITS,
) -> RuntimeTreeReceipt:
    """Write one new mode-600 receipt for a trusted local provision tree.

    The destination must not already exist and must be outside the received
    tree. This function does not sanitize or copy untrusted provisioning input.
    """

    limits = _validated_limits(limits)
    candidate_root = Path(os.path.abspath(Path(root).expanduser()))
    candidate_destination = Path(os.path.abspath(Path(destination).expanduser()))
    if (
        not candidate_destination.name
        or candidate_destination.name in {".", ".."}
        or "\x00" in candidate_destination.name
    ):
        raise RuntimeTreeReceiptError()
    stable_root, files = _scan_runtime_tree(candidate_root, limits=limits)
    try:
        candidate_destination.relative_to(stable_root)
    except ValueError:
        pass
    else:
        raise RuntimeTreeReceiptError()
    payload = _encoded_receipt(stable_root, files)
    if not payload or len(payload) > limits.maximum_receipt_bytes:
        raise RuntimeTreeReceiptError()
    _write_new_private_receipt(candidate_destination, payload)
    return load_runtime_tree_receipt(
        candidate_destination,
        expected_digest=hashlib.sha256(payload).hexdigest(),
        limits=limits,
    )
