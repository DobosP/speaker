"""Exact wheel-lock and installed-runtime ownership verification.

This module deliberately does not resolve, download, install, or import a
candidate package.  A lock becomes meaningful only after every selected wheel
has been downloaded and its exact filename, byte size, and SHA-256 have been
measured.  The verifier then ties a private wheelhouse and an installed
``site-packages`` :class:`RuntimeTreeReceipt` to that measured closure.
"""

from __future__ import annotations

import base64
import binascii
import csv
from dataclasses import dataclass, field
from email import policy
from email.parser import BytesParser
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import sys
import tomllib
from types import MappingProxyType
from typing import Iterable, Mapping
import unicodedata
from urllib.parse import unquote, urlsplit
import zipfile

from packaging.tags import parse_tag
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

from .bounded_io import (
    BoundedReadError,
    opened_directory_nofollow,
    read_regular_bounded,
)
from .runtime_receipt import (
    RuntimeFileReceipt,
    RuntimeTreeLimits,
    RuntimeTreeReceipt,
    RuntimeTreeReceiptError,
    verify_runtime_tree_receipt,
)


WHEEL_LOCK_SCHEMA_VERSION = 1
MAXIMUM_WHEELS = 96
MAXIMUM_LOCK_BYTES = 256 * 1024
MAXIMUM_PYLOCK_BYTES = 8 * 1024 * 1024
MAXIMUM_WHEELHOUSE_BYTES = 8 * 1024 * 1024 * 1024
MAXIMUM_INSTALLED_BYTES = 12 * 1024 * 1024 * 1024
MAXIMUM_INSTALLED_FILE_BYTES = 2 * 1024 * 1024 * 1024
MAXIMUM_INSTALLED_FILES = 50_000
MAXIMUM_ARCHIVE_MEMBERS = 100_000
MAXIMUM_MEMBERS_PER_WHEEL = 50_000
MAXIMUM_RECORD_BYTES = 8 * 1024 * 1024
MAXIMUM_METADATA_BYTES = 4 * 1024 * 1024
MAXIMUM_CENTRAL_DIRECTORY_BYTES = 128 * 1024 * 1024
MAXIMUM_PATH_BYTES = 1024
MAXIMUM_PATH_DEPTH = 32
MAXIMUM_INSTALLER_FILES = MAXIMUM_WHEELS * 2
MAXIMUM_INSTALLER_BYTES = 16

_LOCK_FIELDS = {"schema_version", "wheels"}
_WHEEL_FIELDS = {
    "distribution",
    "version",
    "filename",
    "wheel_tag",
    "source_url",
    "size_bytes",
    "sha256",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_WHEEL_TAG_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.]*-"
    r"[A-Za-z0-9][A-Za-z0-9_.]*-"
    r"[A-Za-z0-9][A-Za-z0-9_.]*\Z"
)
_WINDOWS_RESERVED = {
    "aux",
    "clock$",
    "con",
    "nul",
    "prn",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}
_STDLIB_NAMES = frozenset(name.casefold() for name in sys.stdlib_module_names)
_HASH_CHUNK_BYTES = 1024 * 1024
_ZIP_BOMB_RATIO_FLOOR = 1024 * 1024
_ZIP_BOMB_MAX_RATIO = 32
_ALLOWED_ZIP_COMPRESSION = {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
_END_CENTRAL_DIRECTORY = b"PK\x05\x06"
_CENTRAL_DIRECTORY_ENTRY = b"PK\x01\x02"
_MAXIMUM_ZIP_COMMENT_BYTES = 65_535
_INSTALLER_CONTENT = b"uv"
_REQUESTED_CONTENT = b""
_NEMOTRON_EXTERNAL_RUNTIME_EXCLUSIONS = {
    (
        "numba",
        "0.66.0",
        "numba-0.66.0.data/scripts/numba",
    ): (
        "f2744e313bbffe42661c58d2dc5066cc32e0470e403ce8afc615db3a94d39d71",
        178,
    ),
    (
        "sympy",
        "1.14.0",
        "sympy-1.14.0.data/data/share/man/man1/isympy.1",
    ): (
        "f4365d48e2102e292b0131e56e4743674e0b0508a0643504d3fa025c10ef741b",
        6_659,
    ),
}
_NEMOTRON_INERT_PTH_FILES = {
    (
        "cuda-bindings",
        "12.9.7",
        "_cuda_bindings_redirector.pth",
    ): (
        "28d86507e791da835cf8b9dd21fe98b579b77b3ca27e3ad27bae6239bc94f769",
        195,
    ),
    (
        "setuptools",
        "81.0.0",
        "distutils-precedence.pth",
    ): (
        "2638ce9e2500e572a5e0de7faed6661eb569d1b696fcba07b0dd223da5f5d224",
        151,
    ),
}
_NEMOTRON_SHARED_RUNTIME_FILES = {
    "nvidia/__init__.py": {
        "receipt": (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            0,
        ),
        "owners": frozenset(
            {
                ("nvidia-cublas-cu12", "12.6.4.1"),
                ("nvidia-cuda-cupti-cu12", "12.6.80"),
                ("nvidia-cuda-nvrtc-cu12", "12.6.85"),
                ("nvidia-cuda-runtime-cu12", "12.6.77"),
                ("nvidia-cufft-cu12", "11.3.0.4"),
                ("nvidia-cufile-cu12", "1.11.1.6"),
                ("nvidia-curand-cu12", "10.3.7.77"),
                ("nvidia-cusolver-cu12", "11.7.1.2"),
                ("nvidia-cusparse-cu12", "12.5.4.2"),
                ("nvidia-nvjitlink-cu12", "12.6.85"),
                ("nvidia-nvtx-cu12", "12.6.77"),
            }
        ),
    }
}

WHEEL_LOCK_RUNTIME_TREE_LIMITS = RuntimeTreeLimits(
    maximum_receipt_bytes=32 * 1024 * 1024,
    maximum_files=MAXIMUM_INSTALLED_FILES,
    maximum_directories=10_000,
    maximum_file_bytes=MAXIMUM_INSTALLED_FILE_BYTES,
    maximum_aggregate_bytes=MAXIMUM_INSTALLED_BYTES,
    maximum_relative_path_bytes=MAXIMUM_PATH_BYTES,
    maximum_depth=MAXIMUM_PATH_DEPTH,
)


class WheelLockError(RuntimeError):
    """A detail-free invalid lock, wheelhouse, archive, or installation."""


@dataclass(frozen=True)
class WheelLockEntry:
    """One exact, already measured wheel selected for the candidate runtime."""

    distribution: str
    version: str
    filename: str
    wheel_tag: str
    source_url: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class WheelLock:
    """A strict lock whose byte digest binds its ordered wheel selection."""

    path: Path = field(repr=False)
    digest: str
    wheels: tuple[WheelLockEntry, ...]
    _by_filename: Mapping[str, WheelLockEntry] = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_by_filename",
            MappingProxyType({wheel.filename: wheel for wheel in self.wheels}),
        )

    @property
    def by_filename(self) -> Mapping[str, WheelLockEntry]:
        return self._by_filename

    @property
    def total_size_bytes(self) -> int:
        return sum(wheel.size_bytes for wheel in self.wheels)


@dataclass(frozen=True)
class InstallerFile:
    """One explicitly expected, deterministic installer-created metadata file."""

    relative_path: str
    content: bytes = field(repr=False)


@dataclass(frozen=True)
class _RecordRow:
    installed_path: str
    sha256: str | None
    size_bytes: int | None


@dataclass(frozen=True)
class VerifiedWheel:
    """Parsed RECORD ownership from one exact wheel."""

    entry: WheelLockEntry
    dist_info_directory: str
    record_path: str
    archive_member_count: int
    release_files: tuple[RuntimeFileReceipt, ...]
    excluded_external_files: tuple[RuntimeFileReceipt, ...]
    release_rows: tuple[_RecordRow, ...] = field(repr=False)


@dataclass(frozen=True)
class VerifiedWheelhouse:
    """The exact parsed closure of one private wheelhouse."""

    root: Path = field(repr=False)
    wheels: tuple[VerifiedWheel, ...]


@dataclass(frozen=True)
class InstalledContentExpectation:
    """Root-independent exact content expected in ``site-packages``."""

    files: tuple[RuntimeFileReceipt, ...]

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def total_size_bytes(self) -> int:
        return sum(item.size_bytes for item in self.files)

    @property
    def maximum_file_bytes(self) -> int:
        return max((item.size_bytes for item in self.files), default=0)

    @property
    def content_digest(self) -> str:
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


@dataclass(frozen=True)
class _MeasuredWheel:
    filename: str
    size_bytes: int
    sha256: str


def _fail() -> object:
    raise WheelLockError()


def _strict_json(raw: bytes) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise WheelLockError()
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
        WheelLockError,
    ):
        raise WheelLockError() from None


def _sha256(value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise WheelLockError()
    return value


def _positive_size(value: object, *, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > maximum
    ):
        raise WheelLockError()
    return value


def _safe_component(component: str) -> bool:
    try:
        encoded = component.encode("utf-8", errors="strict")
    except UnicodeError:
        return False
    stem = component.split(".", 1)[0].casefold()
    return (
        bool(component)
        and len(encoded) <= 255
        and unicodedata.normalize("NFC", component) == component
        and component not in {".", ".."}
        and not component.endswith((".", " "))
        and stem not in _WINDOWS_RESERVED
        and not any(
            ord(character) < 32 or ord(character) == 127 for character in component
        )
        and all(character not in "\\:" for character in component)
    )


def _safe_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise WheelLockError()
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeError:
        raise WheelLockError() from None
    pure = PurePosixPath(value)
    if (
        len(encoded) > MAXIMUM_PATH_BYTES
        or pure.is_absolute()
        or len(pure.parts) > MAXIMUM_PATH_DEPTH
        or pure.as_posix() != value
        or any(not _safe_component(part) for part in pure.parts)
    ):
        raise WheelLockError()
    return value


def _wheel_tag(filename: str) -> str:
    try:
        stem, extension = filename.rsplit(".", 1)
        _prefix, python_tag, abi_tag, platform_tag = stem.rsplit("-", 3)
    except ValueError:
        raise WheelLockError() from None
    tag = f"{python_tag}-{abi_tag}-{platform_tag}"
    if extension != "whl" or _WHEEL_TAG_RE.fullmatch(tag) is None:
        raise WheelLockError()
    return tag


def _https_wheel_url(value: object, *, filename: str) -> str:
    try:
        encoded = (
            value.encode("utf-8", errors="strict") if isinstance(value, str) else b""
        )
    except UnicodeError:
        raise WheelLockError() from None
    if (
        not isinstance(value, str)
        or not value
        or len(encoded) > 4096
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
        or "\\" in value
    ):
        raise WheelLockError()
    try:
        parsed = urlsplit(value)
        port = parsed.port
        basename = unquote(PurePosixPath(parsed.path).name)
    except (UnicodeError, ValueError):
        raise WheelLockError() from None
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or port not in {None, 443}
        or basename != filename
    ):
        raise WheelLockError()
    return value


def _parse_lock_entry(raw: object) -> WheelLockEntry:
    if not isinstance(raw, dict) or set(raw) != _WHEEL_FIELDS:
        raise WheelLockError()
    distribution = raw.get("distribution")
    version = raw.get("version")
    filename = raw.get("filename")
    try:
        filename_bytes = (
            filename.encode("ascii", errors="strict")
            if isinstance(filename, str)
            else b""
        )
    except UnicodeError:
        raise WheelLockError() from None
    if (
        not isinstance(distribution, str)
        or not distribution
        or len(distribution) > 128
        or not distribution.isascii()
        or canonicalize_name(distribution) != distribution
        or not isinstance(version, str)
        or not version
        or len(version) > 128
        or not version.isascii()
        or not isinstance(filename, str)
        or not filename
        or len(filename_bytes) > 255
        or PurePosixPath(filename).name != filename
    ):
        raise WheelLockError()
    try:
        normalized_version = str(Version(version))
        parsed_name, parsed_version, _build, _tags = parse_wheel_filename(filename)
    except (InvalidVersion, ValueError):
        raise WheelLockError() from None
    exact_tag = _wheel_tag(filename)
    if (
        normalized_version != version
        or str(parsed_name) != distribution
        or str(parsed_version) != version
        or raw.get("wheel_tag") != exact_tag
    ):
        raise WheelLockError()
    return WheelLockEntry(
        distribution=distribution,
        version=version,
        filename=filename,
        wheel_tag=exact_tag,
        source_url=_https_wheel_url(raw.get("source_url"), filename=filename),
        size_bytes=_positive_size(
            raw.get("size_bytes"),
            maximum=MAXIMUM_WHEELHOUSE_BYTES,
        ),
        sha256=_sha256(raw.get("sha256")),
    )


def load_wheel_lock(
    path: Path | str,
    *,
    expected_digest: str | None = None,
) -> WheelLock:
    """Load one strict, private, duplicate-key-free measured wheel lock."""

    if expected_digest is not None:
        expected_digest = _sha256(expected_digest)
    try:
        candidate = Path(os.path.abspath(Path(path).expanduser()))
        stable_path, raw = _read_private_lock(candidate)
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        BoundedReadError,
        WheelLockError,
    ):
        raise WheelLockError() from None
    digest = hashlib.sha256(raw).hexdigest()
    if expected_digest is not None and digest != expected_digest:
        raise WheelLockError()
    value = _strict_json(raw)
    if (
        not isinstance(value, dict)
        or set(value) != _LOCK_FIELDS
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") != WHEEL_LOCK_SCHEMA_VERSION
        or not isinstance(value.get("wheels"), list)
        or not 1 <= len(value["wheels"]) <= MAXIMUM_WHEELS
    ):
        raise WheelLockError()
    wheels = tuple(_parse_lock_entry(raw) for raw in value["wheels"])
    order = tuple(
        (wheel.distribution, wheel.version, wheel.filename) for wheel in wheels
    )
    filenames = tuple(wheel.filename for wheel in wheels)
    distributions = tuple(wheel.distribution for wheel in wheels)
    sources = tuple(wheel.source_url for wheel in wheels)
    if (
        order != tuple(sorted(order))
        or len(filenames) != len(set(filenames))
        or len(distributions) != len(set(distributions))
        or len(sources) != len(set(sources))
        or len({filename.casefold() for filename in filenames}) != len(filenames)
        or sum(wheel.size_bytes for wheel in wheels) > MAXIMUM_WHEELHOUSE_BYTES
    ):
        raise WheelLockError()
    return WheelLock(path=stable_path, digest=digest, wheels=wheels)


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


def _read_private_lock(candidate: Path) -> tuple[Path, bytes]:
    descriptor = -1
    try:
        if (
            not candidate.name
            or candidate.name in {".", ".."}
            or "\x00" in candidate.name
        ):
            raise WheelLockError()
        with opened_directory_nofollow(candidate.parent) as (
            stable_parent,
            parent_descriptor,
        ):
            stable_path = stable_parent / candidate.name
            if stable_path != candidate:
                raise WheelLockError()
            parent_identity = _metadata_identity(os.fstat(parent_descriptor))
            before = os.stat(
                candidate.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or stat.S_IMODE(before.st_mode) != 0o600
                or before.st_size <= 0
                or before.st_size > MAXIMUM_LOCK_BYTES
                or (hasattr(os, "geteuid") and before.st_uid != os.geteuid())
            ):
                raise WheelLockError()
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(
                candidate.name,
                flags,
                dir_fd=parent_descriptor,
            )
            opened = os.fstat(descriptor)
            identity = _metadata_identity(opened)
            if identity != _metadata_identity(before):
                raise WheelLockError()
            with os.fdopen(descriptor, "rb", buffering=0) as handle:
                descriptor = -1
                raw = handle.read(MAXIMUM_LOCK_BYTES + 1)
                after = os.fstat(handle.fileno())
            current = os.stat(
                candidate.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                not raw
                or len(raw) != before.st_size
                or len(raw) > MAXIMUM_LOCK_BYTES
                or _metadata_identity(after) != identity
                or _metadata_identity(current) != identity
                or _metadata_identity(os.fstat(parent_descriptor)) != parent_identity
            ):
                raise WheelLockError()
            return stable_path, raw
    except (
        OSError,
        RuntimeError,
        ValueError,
        OverflowError,
        BoundedReadError,
        WheelLockError,
    ):
        raise WheelLockError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _bounded_directory_names(
    descriptor: int,
    *,
    maximum_entries: int,
) -> tuple[str, ...]:
    names: list[str] = []
    try:
        with os.scandir(descriptor) as entries:
            for entry in entries:
                if len(names) >= maximum_entries:
                    raise WheelLockError()
                name = entry.name
                if (
                    not isinstance(name, str)
                    or not name
                    or name in {".", ".."}
                    or "/" in name
                    or "\x00" in name
                ):
                    raise WheelLockError()
                names.append(name)
    except (OSError, ValueError, OverflowError, WheelLockError):
        raise WheelLockError() from None
    names.sort()
    return tuple(names)


def _measure_wheel_at(
    root_descriptor: int,
    filename: str,
) -> _MeasuredWheel:
    descriptor = -1
    try:
        before = os.stat(
            filename,
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )
        size = _positive_size(
            before.st_size,
            maximum=MAXIMUM_WHEELHOUSE_BYTES,
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (hasattr(os, "geteuid") and before.st_uid != os.geteuid())
        ):
            raise WheelLockError()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(filename, flags, dir_fd=root_descriptor)
        opened = os.fstat(descriptor)
        identity = _metadata_identity(opened)
        if identity != _metadata_identity(before):
            raise WheelLockError()
        digest = hashlib.sha256()
        remaining = size
        while remaining:
            chunk = os.read(descriptor, min(_HASH_CHUNK_BYTES, remaining))
            if not chunk:
                raise WheelLockError()
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise WheelLockError()
        after = os.fstat(descriptor)
        current = os.stat(
            filename,
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )
        if (
            _metadata_identity(after) != identity
            or _metadata_identity(current) != identity
        ):
            raise WheelLockError()
        return _MeasuredWheel(filename, size, digest.hexdigest())
    except (
        OSError,
        RuntimeError,
        ValueError,
        OverflowError,
        WheelLockError,
    ):
        raise WheelLockError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _measure_private_wheelhouse(
    wheelhouse: Path | str,
) -> tuple[Path, tuple[_MeasuredWheel, ...]]:
    try:
        candidate = Path(os.path.abspath(Path(wheelhouse).expanduser()))
        with opened_directory_nofollow(
            candidate,
            require_private=True,
        ) as (stable_root, descriptor):
            root_identity = _metadata_identity(os.fstat(descriptor))
            names = _bounded_directory_names(
                descriptor,
                maximum_entries=MAXIMUM_WHEELS,
            )
            if not 1 <= len(names) <= MAXIMUM_WHEELS or len(
                {name.casefold() for name in names}
            ) != len(names):
                raise WheelLockError()
            measured: list[_MeasuredWheel] = []
            aggregate = 0
            for name in names:
                if (
                    not isinstance(name, str)
                    or not name
                    or not name.isascii()
                    or PurePosixPath(name).name != name
                    or len(name.encode("ascii")) > 255
                ):
                    raise WheelLockError()
                try:
                    parse_wheel_filename(name)
                except ValueError:
                    raise WheelLockError() from None
                item = _measure_wheel_at(descriptor, name)
                aggregate += item.size_bytes
                if aggregate > MAXIMUM_WHEELHOUSE_BYTES:
                    raise WheelLockError()
                measured.append(item)
            if (
                _metadata_identity(os.fstat(descriptor)) != root_identity
                or _bounded_directory_names(
                    descriptor,
                    maximum_entries=MAXIMUM_WHEELS,
                )
                != names
                or stable_root.resolve(strict=True) != stable_root
            ):
                raise WheelLockError()
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        OverflowError,
        BoundedReadError,
        WheelLockError,
    ):
        raise WheelLockError() from None
    return stable_root, tuple(measured)


def _load_uv_pylock(path: Path | str) -> tuple[Mapping[str, object], ...]:
    try:
        candidate = Path(os.path.abspath(Path(path).expanduser()))
        with opened_directory_nofollow(candidate.parent):
            pass
        snapshot = read_regular_bounded(
            candidate,
            maximum_bytes=MAXIMUM_PYLOCK_BYTES,
        )
        if snapshot.path != candidate:
            raise WheelLockError()
        value = tomllib.loads(snapshot.data.decode("utf-8", errors="strict"))
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        UnicodeError,
        tomllib.TOMLDecodeError,
        BoundedReadError,
        WheelLockError,
    ):
        raise WheelLockError() from None
    packages = value.get("packages") if isinstance(value, dict) else None
    if (
        value.get("lock-version") != "1.0"
        or not isinstance(packages, list)
        or not 1 <= len(packages) <= MAXIMUM_WHEELS
        or any(not isinstance(package, dict) for package in packages)
    ):
        raise WheelLockError()
    return tuple(packages)


def _measured_lock_entries(
    packages: tuple[Mapping[str, object], ...],
    measured: tuple[_MeasuredWheel, ...],
) -> tuple[WheelLockEntry, ...]:
    measured_by_identity: dict[tuple[str, str], _MeasuredWheel] = {}
    for item in measured:
        try:
            name, version, _build, _tags = parse_wheel_filename(item.filename)
        except ValueError:
            raise WheelLockError() from None
        identity = str(name), str(version)
        if identity in measured_by_identity:
            raise WheelLockError()
        measured_by_identity[identity] = item

    entries: list[WheelLockEntry] = []
    package_identities: list[tuple[str, str]] = []
    for package in packages:
        name = package.get("name")
        version = package.get("version")
        wheels = package.get("wheels")
        if (
            not isinstance(name, str)
            or not name
            or not name.isascii()
            or canonicalize_name(name) != name
            or not isinstance(version, str)
            or not version
            or not version.isascii()
            or not isinstance(wheels, list)
            or not 1 <= len(wheels) <= 256
            or any(not isinstance(wheel, dict) for wheel in wheels)
        ):
            raise WheelLockError()
        try:
            normalized_version = str(Version(version))
        except InvalidVersion:
            raise WheelLockError() from None
        if normalized_version != version:
            raise WheelLockError()
        identity = name, version
        package_identities.append(identity)
        selected = measured_by_identity.get(identity)
        if selected is None:
            raise WheelLockError()
        matching: list[Mapping[str, object]] = []
        for wheel in wheels:
            url = wheel.get("url")
            if not isinstance(url, str):
                raise WheelLockError()
            try:
                basename = unquote(PurePosixPath(urlsplit(url).path).name)
            except (UnicodeError, ValueError):
                raise WheelLockError() from None
            if basename == selected.filename:
                matching.append(wheel)
        if len(matching) != 1:
            raise WheelLockError()
        selected_context = matching[0]
        hashes = selected_context.get("hashes")
        if (
            (
                "size" in selected_context
                and (
                    isinstance(selected_context.get("size"), bool)
                    or not isinstance(selected_context.get("size"), int)
                    or selected_context.get("size") != selected.size_bytes
                )
            )
            or not isinstance(hashes, dict)
            or set(hashes) != {"sha256"}
            or hashes.get("sha256") != selected.sha256
        ):
            raise WheelLockError()
        entry = _parse_lock_entry(
            {
                "distribution": name,
                "version": version,
                "filename": selected.filename,
                "wheel_tag": _wheel_tag(selected.filename),
                "source_url": selected_context.get("url"),
                "size_bytes": selected.size_bytes,
                "sha256": selected.sha256,
            }
        )
        entries.append(entry)

    expected_identities = tuple(sorted(measured_by_identity))
    if (
        tuple(package_identities) != tuple(sorted(package_identities))
        or len(package_identities) != len(set(package_identities))
        or tuple(sorted(package_identities)) != expected_identities
    ):
        raise WheelLockError()
    entries.sort(
        key=lambda entry: (
            entry.distribution,
            entry.version,
            entry.filename,
        )
    )
    return tuple(entries)


def _encoded_lock(entries: tuple[WheelLockEntry, ...]) -> bytes:
    payload = {
        "schema_version": WHEEL_LOCK_SCHEMA_VERSION,
        "wheels": [
            {
                "distribution": entry.distribution,
                "version": entry.version,
                "filename": entry.filename,
                "wheel_tag": entry.wheel_tag,
                "source_url": entry.source_url,
                "size_bytes": entry.size_bytes,
                "sha256": entry.sha256,
            }
            for entry in entries
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


def _write_new_private(path: Path, payload: bytes) -> None:
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
                raise WheelLockError()
            with os.fdopen(descriptor, "wb", buffering=0) as handle:
                descriptor = -1
                handle.write(payload)
                os.fsync(handle.fileno())
            os.fsync(parent_descriptor)
    except (
        OSError,
        RuntimeError,
        ValueError,
        OverflowError,
        BoundedReadError,
        WheelLockError,
    ):
        raise WheelLockError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def generate_measured_wheel_lock(
    uv_pylock: Path | str,
    wheelhouse: Path | str,
    destination: Path | str,
) -> WheelLock:
    """Write a strict lock from uv resolution context and measured wheels.

    The uv pylock supplies the selected wheel's exact HTTPS source URL, but its
    size and digest are never trusted without equality to the private
    wheelhouse measurement.  Every pylock package must have exactly one
    selected wheel, and every wheelhouse file must satisfy exactly one package.
    Archive/RECORD verification remains a separate ``verify_wheelhouse`` step.
    """

    stable_wheelhouse, measured = _measure_private_wheelhouse(wheelhouse)
    packages = _load_uv_pylock(uv_pylock)
    entries = _measured_lock_entries(packages, measured)
    payload = _encoded_lock(entries)
    if not payload or len(payload) > MAXIMUM_LOCK_BYTES:
        raise WheelLockError()
    try:
        output = Path(os.path.abspath(Path(destination).expanduser()))
    except (OSError, RuntimeError, TypeError, ValueError):
        raise WheelLockError() from None
    if not output.name or output.name in {".", ".."} or "\x00" in output.name:
        raise WheelLockError()
    try:
        output.relative_to(stable_wheelhouse)
    except ValueError:
        pass
    else:
        raise WheelLockError()
    _write_new_private(output, payload)
    lock = load_wheel_lock(
        output,
        expected_digest=hashlib.sha256(payload).hexdigest(),
    )
    if lock.wheels != entries:
        raise WheelLockError()
    return lock


def _zip_member_path(info: zipfile.ZipInfo) -> str:
    value = info.filename
    original = info.orig_filename
    if (
        not isinstance(value, str)
        or not isinstance(original, str)
        or not value
        or "\x00" in original
        or original != value
    ):
        raise WheelLockError()
    directory = value.endswith("/")
    candidate = value[:-1] if directory else value
    relative = _safe_relative_path(candidate)
    if info.is_dir() != directory:
        raise WheelLockError()
    mode = info.external_attr >> 16
    file_type = stat.S_IFMT(mode)
    if directory:
        if file_type not in {0, stat.S_IFDIR} or info.file_size != 0:
            raise WheelLockError()
    elif file_type not in {0, stat.S_IFREG}:
        raise WheelLockError()
    return relative


def _record_digest(value: str) -> str:
    encoded_value = value.removeprefix("sha256=")
    if (
        not value.startswith("sha256=")
        or re.fullmatch(r"[A-Za-z0-9_-]{43}", encoded_value) is None
    ):
        raise WheelLockError()
    try:
        encoded = encoded_value.encode("ascii", errors="strict")
        padding = b"=" * ((4 - len(encoded) % 4) % 4)
        digest = base64.b64decode(
            encoded + padding,
            altchars=b"-_",
            validate=True,
        )
    except (UnicodeError, binascii.Error, ValueError):
        raise WheelLockError() from None
    if len(digest) != hashlib.sha256().digest_size:
        raise WheelLockError()
    return digest.hex()


def _read_zip_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    maximum_bytes: int,
) -> bytes:
    if info.file_size < 0 or info.file_size > maximum_bytes:
        raise WheelLockError()
    try:
        with archive.open(info, mode="r") as handle:
            data = handle.read(info.file_size + 1)
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile):
        raise WheelLockError() from None
    if len(data) != info.file_size or len(data) > maximum_bytes:
        raise WheelLockError()
    return data


def _read_exact(handle: io.FileIO, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise WheelLockError()
    return data


def _preflight_zip(handle: io.FileIO, entry: WheelLockEntry) -> tuple[int, int]:
    """Bound the ZIP central directory before ``zipfile`` allocates entries."""

    tail_size = min(
        entry.size_bytes,
        22 + _MAXIMUM_ZIP_COMMENT_BYTES,
    )
    try:
        handle.seek(entry.size_bytes - tail_size)
        tail = _read_exact(handle, tail_size)
        eocd_relative = tail.rfind(_END_CENTRAL_DIRECTORY)
        if eocd_relative < 0 or len(tail) - eocd_relative < 22:
            raise WheelLockError()
        eocd_offset = entry.size_bytes - tail_size + eocd_relative
        (
            disk_number,
            central_disk,
            entries_on_disk,
            entry_count,
            central_size,
            central_offset,
            comment_size,
        ) = struct.unpack_from("<4H2LH", tail, eocd_relative + 4)
        if (
            disk_number != 0
            or central_disk != 0
            or entries_on_disk != entry_count
            or not 1 <= entry_count <= MAXIMUM_MEMBERS_PER_WHEEL
            or central_size <= 0
            or central_size > MAXIMUM_CENTRAL_DIRECTORY_BYTES
            or central_offset < 0
            or central_offset + central_size != eocd_offset
            or eocd_offset + 22 + comment_size != entry.size_bytes
            or entry_count == 0xFFFF
            or central_size == 0xFFFFFFFF
            or central_offset == 0xFFFFFFFF
        ):
            raise WheelLockError()

        handle.seek(central_offset)
        consumed = 0
        for _index in range(entry_count):
            header = _read_exact(handle, 46)
            consumed += len(header)
            if header[:4] != _CENTRAL_DIRECTORY_ENTRY:
                raise WheelLockError()
            filename_size, extra_size, member_comment_size = struct.unpack_from(
                "<3H",
                header,
                28,
            )
            member_disk = struct.unpack_from("<H", header, 34)[0]
            variable_size = filename_size + extra_size + member_comment_size
            if (
                filename_size == 0
                or filename_size > MAXIMUM_PATH_BYTES
                or member_disk != 0
                or consumed + variable_size > central_size
            ):
                raise WheelLockError()
            _read_exact(handle, variable_size)
            consumed += variable_size
        if consumed != central_size or handle.tell() != eocd_offset:
            raise WheelLockError()
        handle.seek(0)
        return entry_count, central_offset
    except (OSError, ValueError, OverflowError, struct.error, WheelLockError):
        raise WheelLockError() from None


def _installed_path(
    archive_path: str,
    *,
    entry: WheelLockEntry,
    dist_info: str,
) -> tuple[str, bool]:
    parts = PurePosixPath(archive_path).parts
    data_directory = f"{dist_info[: -len('.dist-info')]}.data"
    if parts[0] == data_directory:
        if len(parts) >= 3 and parts[1] in {"purelib", "platlib"}:
            return _safe_relative_path("/".join(parts[2:])), True
        if (
            entry.distribution,
            entry.version,
            archive_path,
        ) in _NEMOTRON_EXTERNAL_RUNTIME_EXCLUSIONS:
            # This exact SymPy man page is a CLI/documentation scheme output.
            # Its original RECORD row stays bound while the isolated ASR
            # runtime deliberately exposes no external data/scripts scheme.
            return archive_path, False
        raise WheelLockError()
    if parts[0].endswith(".data"):
        raise WheelLockError()
    return archive_path, True


def _dist_info_identity(
    infos: Mapping[str, zipfile.ZipInfo],
    entry: WheelLockEntry,
) -> tuple[str, str]:
    records = sorted(
        path
        for path, info in infos.items()
        if (
            not info.is_dir()
            and len(PurePosixPath(path).parts) == 2
            and PurePosixPath(path).parts[0].endswith(".dist-info")
            and PurePosixPath(path).parts[1] == "RECORD"
        )
    )
    if len(records) != 1:
        raise WheelLockError()
    record_path = records[0]
    dist_info = record_path.rsplit("/", 1)[0]
    stem = dist_info[: -len(".dist-info")]
    try:
        raw_name, raw_version = stem.rsplit("-", 1)
        normalized_name = canonicalize_name(raw_name)
        normalized_version = str(Version(raw_version))
    except (InvalidVersion, ValueError):
        raise WheelLockError() from None
    if (
        normalized_name != entry.distribution
        or normalized_version != entry.version
        or "/" in dist_info
    ):
        raise WheelLockError()
    return dist_info, record_path


def _header_values(message: object, name: str) -> tuple[str, ...]:
    try:
        values = message.get_all(name, [])  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError):
        raise WheelLockError() from None
    if not isinstance(values, list):
        raise WheelLockError()
    normalized: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or "\x00" in text or "\r" in text or "\n" in text:
            raise WheelLockError()
        normalized.append(text)
    return tuple(normalized)


def _parsed_metadata(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
) -> object:
    raw = _read_zip_member(
        archive,
        info,
        maximum_bytes=MAXIMUM_METADATA_BYTES,
    )
    if b"\x00" in raw:
        raise WheelLockError()
    try:
        message = BytesParser(policy=policy.default).parsebytes(raw)
    except (TypeError, ValueError, OverflowError):
        raise WheelLockError() from None
    if message.defects:
        raise WheelLockError()
    return message


def _verify_internal_wheel_identity(
    archive: zipfile.ZipFile,
    infos: Mapping[str, zipfile.ZipInfo],
    *,
    entry: WheelLockEntry,
    dist_info: str,
) -> None:
    metadata = _parsed_metadata(
        archive,
        infos[f"{dist_info}/METADATA"],
    )
    names = _header_values(metadata, "Name")
    versions = _header_values(metadata, "Version")
    if len(names) != 1 or len(versions) != 1:
        raise WheelLockError()
    try:
        metadata_name = canonicalize_name(names[0])
        metadata_version = str(Version(versions[0]))
    except InvalidVersion:
        raise WheelLockError() from None
    if (
        metadata_name != entry.distribution
        or metadata_version != entry.version
        or versions[0] != metadata_version
    ):
        raise WheelLockError()

    wheel_metadata = _parsed_metadata(
        archive,
        infos[f"{dist_info}/WHEEL"],
    )
    wheel_versions = _header_values(wheel_metadata, "Wheel-Version")
    root_modes = _header_values(wheel_metadata, "Root-Is-Purelib")
    raw_tags = _header_values(wheel_metadata, "Tag")
    if (
        wheel_versions != ("1.0",)
        or len(root_modes) != 1
        or root_modes[0].casefold() not in {"true", "false"}
        or not raw_tags
    ):
        raise WheelLockError()
    try:
        internal_tags = frozenset(
            tag for raw_tag in raw_tags for tag in parse_tag(raw_tag)
        )
        _name, _version, _build, filename_tags = parse_wheel_filename(entry.filename)
    except ValueError:
        raise WheelLockError() from None
    if internal_tags != filename_tags:
        raise WheelLockError()


def _parse_release_record(
    archive: zipfile.ZipFile,
    infos: Mapping[str, zipfile.ZipInfo],
    *,
    entry: WheelLockEntry,
    dist_info: str,
    record_path: str,
) -> tuple[
    tuple[RuntimeFileReceipt, ...],
    tuple[RuntimeFileReceipt, ...],
    tuple[_RecordRow, ...],
]:
    raw = _read_zip_member(
        archive,
        infos[record_path],
        maximum_bytes=MAXIMUM_RECORD_BYTES,
    )
    expected: list[RuntimeFileReceipt] = []
    excluded: list[RuntimeFileReceipt] = []
    rows_out: list[_RecordRow] = []
    archive_paths: set[str] = set()
    installed_paths: set[str] = set()
    installed_casefolded: set[str] = set()
    archive_by_installed: dict[str, str] = {}
    record_seen = False
    try:
        rows = csv.reader(
            io.StringIO(raw.decode("utf-8", errors="strict"), newline=""),
            strict=True,
        )
        for row_number, row in enumerate(rows, start=1):
            if row_number > MAXIMUM_ARCHIVE_MEMBERS or len(row) != 3:
                raise WheelLockError()
            archive_path = _safe_relative_path(row[0])
            if archive_path in archive_paths:
                raise WheelLockError()
            archive_paths.add(archive_path)
            info = infos.get(archive_path)
            if info is None or info.is_dir():
                raise WheelLockError()
            installed_path, runtime_owned = _installed_path(
                archive_path,
                entry=entry,
                dist_info=dist_info,
            )
            installed_folded = installed_path.casefold()
            if (
                installed_path in installed_paths
                or installed_folded in installed_casefolded
            ):
                raise WheelLockError()
            installed_paths.add(installed_path)
            installed_casefolded.add(installed_folded)
            archive_by_installed[installed_path] = archive_path
            if archive_path == record_path:
                if record_seen or row[1:] != ["", ""]:
                    raise WheelLockError()
                record_seen = True
                rows_out.append(_RecordRow(installed_path, None, None))
                continue
            if not row[1] or not row[2]:
                raise WheelLockError()
            try:
                size = int(row[2], 10)
            except (TypeError, ValueError, OverflowError):
                raise WheelLockError() from None
            if (
                size < 0
                or str(size) != row[2]
                or size > MAXIMUM_INSTALLED_FILE_BYTES
                or info.file_size != size
            ):
                raise WheelLockError()
            digest = _record_digest(row[1])
            receipt = RuntimeFileReceipt(
                relative_path=installed_path,
                sha256=digest,
                size_bytes=size,
            )
            if runtime_owned:
                expected.append(receipt)
            else:
                exact_exclusion = _NEMOTRON_EXTERNAL_RUNTIME_EXCLUSIONS.get(
                    (entry.distribution, entry.version, archive_path)
                )
                if exact_exclusion != (digest, size):
                    raise WheelLockError()
                excluded.append(receipt)
            rows_out.append(_RecordRow(installed_path, digest, size))
    except (UnicodeError, csv.Error, WheelLockError):
        raise WheelLockError() from None
    archive_regular = {path for path, info in infos.items() if not info.is_dir()}
    if (
        not record_seen
        or archive_paths != archive_regular
        or not expected
        or sum(item.size_bytes for item in expected) > MAXIMUM_INSTALLED_BYTES
    ):
        raise WheelLockError()

    for receipt in (*expected, *excluded):
        archive_path = archive_by_installed[receipt.relative_path]
        info = infos[archive_path]
        digest = hashlib.sha256()
        remaining = info.file_size
        try:
            with archive.open(info, mode="r") as handle:
                while remaining:
                    chunk = handle.read(min(_HASH_CHUNK_BYTES, remaining))
                    if not chunk:
                        raise WheelLockError()
                    digest.update(chunk)
                    remaining -= len(chunk)
                if handle.read(1):
                    raise WheelLockError()
        except (OSError, RuntimeError, ValueError, zipfile.BadZipFile):
            raise WheelLockError() from None
        if digest.hexdigest() != receipt.sha256:
            raise WheelLockError()
    expected.sort(key=lambda item: item.relative_path)
    excluded.sort(key=lambda item: item.relative_path)
    rows_out.sort(key=lambda row: row.installed_path)
    return tuple(expected), tuple(excluded), tuple(rows_out)


def _inspect_wheel(handle: io.FileIO, entry: WheelLockEntry) -> VerifiedWheel:
    try:
        expected_members, central_offset = _preflight_zip(handle, entry)
        with zipfile.ZipFile(handle, mode="r") as archive:
            raw_infos = archive.infolist()
            if len(raw_infos) != expected_members:
                raise WheelLockError()
            infos: dict[str, zipfile.ZipInfo] = {}
            casefolded: set[str] = set()
            local_offsets: set[int] = set()
            expanded_bytes = 0
            compressed_bytes = 0
            for info in raw_infos:
                path = _zip_member_path(info)
                folded = path.casefold()
                if (
                    path in infos
                    or folded in casefolded
                    or info.flag_bits & ~0x808
                    or info.header_offset < 0
                    or info.header_offset >= central_offset
                    or info.header_offset in local_offsets
                ):
                    raise WheelLockError()
                infos[path] = info
                casefolded.add(folded)
                local_offsets.add(info.header_offset)
                if info.compress_type not in _ALLOWED_ZIP_COMPRESSION:
                    raise WheelLockError()
                if (
                    info.file_size < 0
                    or info.compress_size < 0
                    or info.file_size > MAXIMUM_INSTALLED_FILE_BYTES
                ):
                    raise WheelLockError()
                expanded_bytes += info.file_size
                compressed_bytes += info.compress_size
                if (
                    expanded_bytes > MAXIMUM_INSTALLED_BYTES
                    or compressed_bytes > entry.size_bytes
                    or (
                        info.file_size > _ZIP_BOMB_RATIO_FLOOR
                        and (
                            info.compress_size == 0
                            or info.file_size > info.compress_size * _ZIP_BOMB_MAX_RATIO
                        )
                    )
                ):
                    raise WheelLockError()
            dist_info, record_path = _dist_info_identity(infos, entry)
            dist_info_directories = {
                PurePosixPath(path).parts[0]
                for path in infos
                if PurePosixPath(path).parts[0].endswith(".dist-info")
            }
            required_metadata = {
                f"{dist_info}/METADATA",
                f"{dist_info}/WHEEL",
                record_path,
            }
            if dist_info_directories != {dist_info} or not required_metadata.issubset(
                infos
            ):
                raise WheelLockError()
            _verify_internal_wheel_identity(
                archive,
                infos,
                entry=entry,
                dist_info=dist_info,
            )
            (
                release_files,
                excluded_external_files,
                release_rows,
            ) = _parse_release_record(
                archive,
                infos,
                entry=entry,
                dist_info=dist_info,
                record_path=record_path,
            )
    except (
        OSError,
        RuntimeError,
        ValueError,
        zipfile.BadZipFile,
        WheelLockError,
    ):
        raise WheelLockError() from None
    return VerifiedWheel(
        entry=entry,
        dist_info_directory=dist_info,
        record_path=record_path,
        archive_member_count=len(raw_infos),
        release_files=release_files,
        excluded_external_files=excluded_external_files,
        release_rows=release_rows,
    )


def _hash_open_wheel(
    root_descriptor: int,
    entry: WheelLockEntry,
) -> VerifiedWheel:
    descriptor = -1
    try:
        before = os.stat(
            entry.filename,
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size != entry.size_bytes
            or (hasattr(os, "geteuid") and before.st_uid != os.geteuid())
        ):
            raise WheelLockError()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(entry.filename, flags, dir_fd=root_descriptor)
        opened = os.fstat(descriptor)
        identity = _metadata_identity(opened)
        if identity != _metadata_identity(before):
            raise WheelLockError()
        digest = hashlib.sha256()
        remaining = entry.size_bytes
        with os.fdopen(descriptor, "rb", buffering=0) as handle:
            descriptor = -1
            while remaining:
                chunk = handle.read(min(_HASH_CHUNK_BYTES, remaining))
                if not chunk:
                    raise WheelLockError()
                digest.update(chunk)
                remaining -= len(chunk)
            if handle.read(1) or digest.hexdigest() != entry.sha256:
                raise WheelLockError()
            verified = _inspect_wheel(handle, entry)
            after = os.fstat(handle.fileno())
        current = os.stat(
            entry.filename,
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )
        if (
            _metadata_identity(after) != identity
            or _metadata_identity(current) != identity
        ):
            raise WheelLockError()
        return verified
    except (
        OSError,
        RuntimeError,
        ValueError,
        OverflowError,
        WheelLockError,
    ):
        raise WheelLockError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _runtime_release_file_union(
    wheels: Iterable[VerifiedWheel],
) -> tuple[RuntimeFileReceipt, ...]:
    by_path: dict[str, list[tuple[VerifiedWheel, RuntimeFileReceipt]]] = {}
    for wheel in wheels:
        for receipt in wheel.release_files:
            by_path.setdefault(receipt.relative_path, []).append((wheel, receipt))
    union: list[RuntimeFileReceipt] = []
    for path, owned in by_path.items():
        receipts = {(receipt.sha256, receipt.size_bytes) for _wheel, receipt in owned}
        if len(owned) == 1:
            union.append(owned[0][1])
            continue
        profile = _NEMOTRON_SHARED_RUNTIME_FILES.get(path)
        identities = frozenset(
            (wheel.entry.distribution, wheel.entry.version) for wheel, _receipt in owned
        )
        if (
            profile is None
            or receipts != {profile["receipt"]}
            or identities != profile["owners"]
        ):
            raise WheelLockError()
        union.append(owned[0][1])
    union.sort(key=lambda receipt: receipt.relative_path)
    paths = tuple(receipt.relative_path for receipt in union)
    if len(paths) != len(set(paths)) or len({path.casefold() for path in paths}) != len(
        paths
    ):
        raise WheelLockError()
    return tuple(union)


def verify_wheelhouse(
    lock: WheelLock,
    wheelhouse: Path | str,
) -> VerifiedWheelhouse:
    """Verify and parse exactly the locked files in one private wheelhouse."""

    if not isinstance(lock, WheelLock):
        raise WheelLockError()
    current_lock = load_wheel_lock(lock.path, expected_digest=lock.digest)
    if current_lock != lock:
        raise WheelLockError()
    lock = current_lock
    try:
        candidate = Path(os.path.abspath(Path(wheelhouse).expanduser()))
        with opened_directory_nofollow(
            candidate,
            require_private=True,
        ) as (stable_root, descriptor):
            root_identity = _metadata_identity(os.fstat(descriptor))
            names = _bounded_directory_names(
                descriptor,
                maximum_entries=MAXIMUM_WHEELS,
            )
            if (
                len(names) != len(lock.wheels)
                or set(names) != set(lock.by_filename)
                or len({name.casefold() for name in names}) != len(names)
            ):
                raise WheelLockError()
            parsed_wheels: list[VerifiedWheel] = []
            aggregate_members = 0
            aggregate_installed_bytes = 0
            aggregate_installed_files = 0
            for entry in lock.wheels:
                wheel = _hash_open_wheel(descriptor, entry)
                aggregate_members += wheel.archive_member_count
                aggregate_installed_files += len(wheel.release_files)
                aggregate_installed_bytes += sum(
                    item.size_bytes for item in wheel.release_files
                )
                if (
                    aggregate_members > MAXIMUM_ARCHIVE_MEMBERS
                    or aggregate_installed_files
                    > WHEEL_LOCK_RUNTIME_TREE_LIMITS.maximum_files
                    or aggregate_installed_bytes > MAXIMUM_INSTALLED_BYTES
                ):
                    raise WheelLockError()
                parsed_wheels.append(wheel)
            wheels = tuple(parsed_wheels)
            if (
                _metadata_identity(os.fstat(descriptor)) != root_identity
                or set(
                    _bounded_directory_names(
                        descriptor,
                        maximum_entries=MAXIMUM_WHEELS,
                    )
                )
                != set(lock.by_filename)
                or stable_root.resolve(strict=True) != stable_root
            ):
                raise WheelLockError()
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        OverflowError,
        BoundedReadError,
        WheelLockError,
    ):
        raise WheelLockError() from None
    release_union = _runtime_release_file_union(wheels)
    all_paths = [item.relative_path for item in release_union]
    if (
        len(all_paths) > WHEEL_LOCK_RUNTIME_TREE_LIMITS.maximum_files
        or sum(item.size_bytes for item in release_union) > MAXIMUM_INSTALLED_BYTES
    ):
        raise WheelLockError()
    _reject_import_interposition(wheels, all_paths)
    return VerifiedWheelhouse(root=stable_root, wheels=wheels)


def _reject_import_interposition(
    wheels: Iterable[VerifiedWheel],
    paths: Iterable[str],
) -> None:
    allowed_inert_pth: set[str] = set()
    for wheel in wheels:
        for receipt in wheel.release_files:
            if not receipt.relative_path.casefold().endswith(".pth"):
                continue
            expected = _NEMOTRON_INERT_PTH_FILES.get(
                (
                    wheel.entry.distribution,
                    wheel.entry.version,
                    receipt.relative_path,
                )
            )
            if expected != (receipt.sha256, receipt.size_bytes):
                raise WheelLockError()
            allowed_inert_pth.add(receipt.relative_path)
    for path in paths:
        pure = PurePosixPath(path)
        folded = path.casefold()
        top = pure.parts[0]
        top_stem = top.split(".", 1)[0].casefold()
        if (
            (folded.endswith(".pth") and path not in allowed_inert_pth)
            or "__pycache__" in {part.casefold() for part in pure.parts}
            or folded in {"sitecustomize.py", "usercustomize.py"}
            or top.casefold() in {"sitecustomize", "usercustomize"}
            or (not top.casefold().endswith(".dist-info") and top_stem in _STDLIB_NAMES)
        ):
            raise WheelLockError()


def _validated_installer_files(
    wheelhouse: VerifiedWheelhouse,
    installer_files: Iterable[InstallerFile],
) -> tuple[RuntimeFileReceipt, ...]:
    provided: list[InstallerFile] = []
    try:
        for item in installer_files:
            if len(provided) >= MAXIMUM_INSTALLER_FILES:
                raise WheelLockError()
            provided.append(item)
    except (TypeError, ValueError, RuntimeError, WheelLockError):
        raise WheelLockError() from None
    dist_infos = {wheel.dist_info_directory: wheel for wheel in wheelhouse.wheels}
    receipts: list[RuntimeFileReceipt] = []
    for item in provided:
        if not isinstance(item, InstallerFile) or not isinstance(item.content, bytes):
            raise WheelLockError()
        relative = _safe_relative_path(item.relative_path)
        try:
            dist_info, basename = relative.rsplit("/", 1)
        except ValueError:
            raise WheelLockError() from None
        if (
            dist_info not in dist_infos
            or len(item.content) > MAXIMUM_INSTALLER_BYTES
            or (basename == "INSTALLER" and item.content != _INSTALLER_CONTENT)
            or (basename == "REQUESTED" and item.content != _REQUESTED_CONTENT)
            or basename not in {"INSTALLER", "REQUESTED"}
        ):
            raise WheelLockError()
        receipts.append(
            RuntimeFileReceipt(
                relative_path=relative,
                sha256=hashlib.sha256(item.content).hexdigest(),
                size_bytes=len(item.content),
            )
        )
    paths = tuple(item.relative_path for item in receipts)
    if (
        len(paths) != len(set(paths))
        or tuple(paths) != tuple(sorted(paths))
        or len({path.casefold() for path in paths}) != len(paths)
    ):
        raise WheelLockError()
    return tuple(receipts)


def _parse_installed_record(
    receipt: RuntimeTreeReceipt,
    wheel: VerifiedWheel,
) -> tuple[_RecordRow, ...]:
    record_receipt = receipt.file_by_path.get(wheel.record_path)
    if record_receipt is None or record_receipt.size_bytes > MAXIMUM_RECORD_BYTES:
        raise WheelLockError()
    try:
        snapshot = read_regular_bounded(
            receipt.root / wheel.record_path,
            maximum_bytes=MAXIMUM_RECORD_BYTES,
            expected_bytes=record_receipt.size_bytes,
        )
    except BoundedReadError:
        raise WheelLockError() from None
    if hashlib.sha256(snapshot.data).hexdigest() != record_receipt.sha256:
        raise WheelLockError()
    parsed: list[_RecordRow] = []
    seen: set[str] = set()
    try:
        rows = csv.reader(
            io.StringIO(
                snapshot.data.decode("utf-8", errors="strict"),
                newline="",
            ),
            strict=True,
        )
        for row_number, row in enumerate(rows, start=1):
            if row_number > MAXIMUM_ARCHIVE_MEMBERS or len(row) != 3 or row[0] in seen:
                raise WheelLockError()
            relative = _safe_relative_path(row[0])
            seen.add(relative)
            if relative == wheel.record_path:
                if row[1:] != ["", ""]:
                    raise WheelLockError()
                parsed.append(_RecordRow(relative, None, None))
                continue
            if not row[1] or not row[2]:
                raise WheelLockError()
            try:
                size = int(row[2], 10)
            except (TypeError, ValueError, OverflowError):
                raise WheelLockError() from None
            if size < 0 or str(size) != row[2] or size > MAXIMUM_INSTALLED_FILE_BYTES:
                raise WheelLockError()
            parsed.append(_RecordRow(relative, _record_digest(row[1]), size))
    except (UnicodeError, csv.Error, WheelLockError):
        raise WheelLockError() from None
    parsed.sort(key=lambda row: row.installed_path)
    return tuple(parsed)


def verify_installed_wheels(
    wheelhouse: VerifiedWheelhouse,
    receipt: RuntimeTreeReceipt,
    *,
    installer_files: Iterable[InstallerFile] = (),
) -> InstalledContentExpectation:
    """Require exact wheel RECORD ownership of an installed runtime receipt.

    Release RECORD rows own package content.  The only permitted installer
    additions are explicitly supplied, exact ``INSTALLER`` (``b"uv"``) and
    empty ``REQUESTED`` files inside the corresponding ``.dist-info``.
    """

    if (
        not isinstance(wheelhouse, VerifiedWheelhouse)
        or not isinstance(receipt, RuntimeTreeReceipt)
        or receipt.root.name != "site-packages"
    ):
        raise WheelLockError()
    installer_receipts = _validated_installer_files(
        wheelhouse,
        installer_files,
    )
    release_files = list(_runtime_release_file_union(wheelhouse.wheels))
    release_paths = {item.relative_path for item in release_files}
    installer_paths = {item.relative_path for item in installer_receipts}
    record_paths = {wheel.record_path for wheel in wheelhouse.wheels}
    if (
        release_paths & installer_paths
        or installer_paths & record_paths
        or release_paths & record_paths
    ):
        raise WheelLockError()

    installer_by_record: dict[str, list[_RecordRow]] = {
        wheel.record_path: [] for wheel in wheelhouse.wheels
    }
    dist_info_to_record = {
        wheel.dist_info_directory: wheel.record_path for wheel in wheelhouse.wheels
    }
    for item in installer_receipts:
        dist_info = item.relative_path.rsplit("/", 1)[0]
        installer_by_record[dist_info_to_record[dist_info]].append(
            _RecordRow(
                item.relative_path,
                item.sha256,
                item.size_bytes,
            )
        )

    record_receipts: list[RuntimeFileReceipt] = []
    for wheel in wheelhouse.wheels:
        actual_rows = _parse_installed_record(receipt, wheel)
        expected_rows = tuple(
            sorted(
                (*wheel.release_rows, *installer_by_record[wheel.record_path]),
                key=lambda row: row.installed_path,
            )
        )
        if actual_rows != expected_rows:
            raise WheelLockError()
        record_receipt = receipt.file_by_path.get(wheel.record_path)
        if record_receipt is None:
            raise WheelLockError()
        record_receipts.append(record_receipt)

    files = tuple(
        sorted(
            (*release_files, *installer_receipts, *record_receipts),
            key=lambda item: item.relative_path,
        )
    )
    paths = tuple(item.relative_path for item in files)
    if (
        not files
        or len(paths) != len(set(paths))
        or len({path.casefold() for path in paths}) != len(paths)
        or len(files) > WHEEL_LOCK_RUNTIME_TREE_LIMITS.maximum_files
        or sum(item.size_bytes for item in files) > MAXIMUM_INSTALLED_BYTES
        or max(item.size_bytes for item in files) > MAXIMUM_INSTALLED_FILE_BYTES
        or files != receipt.files
    ):
        raise WheelLockError()
    _reject_import_interposition(wheelhouse.wheels, paths)
    try:
        verify_runtime_tree_receipt(
            receipt,
            limits=WHEEL_LOCK_RUNTIME_TREE_LIMITS,
        )
    except RuntimeTreeReceiptError:
        raise WheelLockError() from None
    expectation = InstalledContentExpectation(files=files)
    if expectation.content_digest != receipt.content_digest:
        raise WheelLockError()
    return expectation


def verify_locked_wheel_install(
    lock: WheelLock,
    wheelhouse: Path | str,
    receipt: RuntimeTreeReceipt,
    *,
    installer_files: Iterable[InstallerFile] = (),
) -> InstalledContentExpectation:
    """Convenience wrapper for exact wheelhouse and installation verification."""

    verified = verify_wheelhouse(lock, wheelhouse)
    return verify_installed_wheels(
        verified,
        receipt,
        installer_files=installer_files,
    )
