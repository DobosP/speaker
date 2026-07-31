"""Race-aware bounded reads for untrusted local benchmark inputs."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import stat
from typing import Iterator


class BoundedReadError(RuntimeError):
    """A local file was unsafe, unstable, or outside its byte budget."""


@dataclass(frozen=True)
class FileSnapshot:
    """One stable regular-file snapshot retained entirely in memory."""

    path: Path = field(repr=False)
    data: bytes = field(repr=False)


@dataclass(frozen=True)
class FileDigest:
    """One stable regular-file identity and digest without retaining its bytes."""

    path: Path = field(repr=False)
    sha256: str
    size_bytes: int


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _directory_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
    )


@contextmanager
def opened_directory_nofollow(
    path: Path,
    *,
    create: bool = False,
    require_private: bool = False,
) -> Iterator[tuple[Path, int]]:
    """Open an absolute directory through a no-symlink component walk.

    Missing components are created mode 700 only when explicitly requested.
    The final lexical path must equal its strict resolution, and its live
    pathname identity is bound to the returned descriptor.
    """

    candidate = Path(os.path.abspath(path))
    if not candidate.is_absolute():
        raise BoundedReadError()
    descriptor = -1
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(candidate.anchor, flags)
        current = Path(candidate.anchor)
        for component in candidate.parts[1:]:
            try:
                before = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, mode=0o700, dir_fd=descriptor)
                before = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            if not stat.S_ISDIR(before.st_mode):
                raise BoundedReadError()
            child = os.open(component, flags, dir_fd=descriptor)
            opened = os.fstat(child)
            if _directory_identity(opened) != _directory_identity(before):
                os.close(child)
                raise BoundedReadError()
            os.close(descriptor)
            descriptor = child
            current /= component

        opened = os.fstat(descriptor)
        current_metadata = candidate.lstat()
        if (
            current != candidate
            or not stat.S_ISDIR(opened.st_mode)
            or _directory_identity(current_metadata) != _directory_identity(opened)
            or candidate.resolve(strict=True) != candidate
            or (
                require_private
                and (
                    stat.S_IMODE(opened.st_mode) & 0o077
                    or (hasattr(os, "geteuid") and opened.st_uid != os.geteuid())
                )
            )
        ):
            raise BoundedReadError()
    except (OSError, RuntimeError, ValueError, OverflowError):
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
            descriptor = -1
        raise BoundedReadError() from None
    try:
        yield candidate, descriptor
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def read_regular_bounded(
    path: Path,
    *,
    maximum_bytes: int,
    expected_bytes: int | None = None,
    allow_empty: bool = False,
) -> FileSnapshot:
    """Read at most the declared budget and reject file/symlink races.

    Size is checked before allocation. The extra byte in the sole read detects
    growth that races the initial metadata check, while before/after identities
    reject replacement or in-place mutation during the snapshot.
    """

    if (
        isinstance(maximum_bytes, bool)
        or not isinstance(maximum_bytes, int)
        or maximum_bytes <= 0
        or (
            expected_bytes is not None
            and (
                isinstance(expected_bytes, bool)
                or not isinstance(expected_bytes, int)
                or expected_bytes < 0
                or expected_bytes > maximum_bytes
            )
        )
    ):
        raise BoundedReadError()

    candidate = Path(path)
    descriptor = -1
    try:
        before = candidate.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > maximum_bytes
            or (not allow_empty and before.st_size == 0)
            or (expected_bytes is not None and before.st_size != expected_bytes)
        ):
            raise BoundedReadError()

        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(candidate, flags)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or _identity(opened) != _identity(before)
        ):
            raise BoundedReadError()

        read_budget = maximum_bytes if expected_bytes is None else expected_bytes
        with os.fdopen(descriptor, "rb", buffering=0) as handle:
            descriptor = -1
            data = handle.read(read_budget + 1)
            after = os.fstat(handle.fileno())

        current = candidate.lstat()
        resolved = candidate.resolve(strict=True)
        resolved_metadata = resolved.stat()
        if (
            _identity(after) != _identity(opened)
            or _identity(current) != _identity(opened)
            or _identity(resolved_metadata) != _identity(opened)
            or after.st_nlink != 1
            or current.st_nlink != 1
            or resolved_metadata.st_nlink != 1
            or len(data) != opened.st_size
            or len(data) > read_budget
            or (not allow_empty and not data)
        ):
            raise BoundedReadError()
        return FileSnapshot(path=resolved, data=data)
    except (OSError, RuntimeError, ValueError, OverflowError):
        raise BoundedReadError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def hash_regular_bounded(
    path: Path,
    *,
    maximum_bytes: int,
    expected_bytes: int,
    allow_final_symlink: bool = False,
) -> FileDigest:
    """Hash one size-bound stable file through a single descriptor.

    A Python virtual-environment executable is commonly a final-component
    symlink, so callers may explicitly retain that lexical launch path. The
    target and the symlink identity are both rechecked after hashing.
    """

    if (
        isinstance(maximum_bytes, bool)
        or not isinstance(maximum_bytes, int)
        or maximum_bytes <= 0
        or isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes <= 0
        or expected_bytes > maximum_bytes
    ):
        raise BoundedReadError()

    candidate = Path(path)
    descriptor = -1
    try:
        lexical_before = candidate.lstat()
        if stat.S_ISLNK(lexical_before.st_mode):
            if not allow_final_symlink or lexical_before.st_nlink != 1:
                raise BoundedReadError()
        elif not stat.S_ISREG(lexical_before.st_mode) or lexical_before.st_nlink != 1:
            raise BoundedReadError()

        resolved = candidate.resolve(strict=True)
        target_before = resolved.lstat()
        if (
            not stat.S_ISREG(target_before.st_mode)
            or target_before.st_nlink != 1
            or target_before.st_size != expected_bytes
        ):
            raise BoundedReadError()

        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(resolved, flags)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or _identity(opened) != _identity(target_before)
            or opened.st_size != expected_bytes
        ):
            raise BoundedReadError()

        digest = hashlib.sha256()
        consumed = 0
        while consumed <= expected_bytes:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, expected_bytes + 1 - consumed),
            )
            if not chunk:
                break
            consumed += len(chunk)
            if consumed > expected_bytes:
                raise BoundedReadError()
            digest.update(chunk)

        after = os.fstat(descriptor)
        lexical_after = candidate.lstat()
        current_resolved = candidate.resolve(strict=True)
        target_after = current_resolved.lstat()
        if (
            consumed != expected_bytes
            or _identity(after) != _identity(opened)
            or _identity(lexical_after) != _identity(lexical_before)
            or current_resolved != resolved
            or _identity(target_after) != _identity(opened)
            or after.st_nlink != 1
            or lexical_after.st_nlink != 1
            or target_after.st_nlink != 1
        ):
            raise BoundedReadError()
        return FileDigest(
            path=resolved,
            sha256=digest.hexdigest(),
            size_bytes=consumed,
        )
    except (OSError, RuntimeError, ValueError, OverflowError):
        raise BoundedReadError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
