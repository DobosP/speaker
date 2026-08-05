"""Pinned, bounded cgroup-v2 resource observations for supervised workers."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import stat
from typing import Mapping


MAX_CGROUP_FILE_BYTES = 4096
MAX_CGROUP_COUNTER = (1 << 63) - 1
_COUNTER_KEY_RE = re.compile(r"[a-z][a-z0-9_.-]{0,63}\Z")
_REQUIRED_CPU_COUNTERS = ("usage_usec", "user_usec", "system_usec")


class CgroupObservationError(RuntimeError):
    """A detail-free cgroup observation failure."""


def _metadata_identity(metadata: os.stat_result) -> tuple[int, ...]:
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


def _read_descriptor_bounded(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(
            descriptor,
            min(1024, MAX_CGROUP_FILE_BYTES + 1 - total),
        )
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        total += len(chunk)
        if total > MAX_CGROUP_FILE_BYTES:
            raise CgroupObservationError("cgroup_observation_invalid")


def _read_path_bounded(path: Path) -> bytes:
    descriptor = -1
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise CgroupObservationError("cgroup_observation_invalid")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise CgroupObservationError("cgroup_observation_invalid")
        payload = _read_descriptor_bounded(descriptor)
        current = path.lstat()
        if _metadata_identity(before) != _metadata_identity(
            opened
        ) or _metadata_identity(before) != _metadata_identity(current):
            raise CgroupObservationError("cgroup_observation_invalid")
        return payload
    except CgroupObservationError:
        raise
    except (OSError, RuntimeError, ValueError, OverflowError):
        raise CgroupObservationError("cgroup_observation_invalid") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_at_bounded(directory_descriptor: int, name: str) -> bytes:
    descriptor = -1
    try:
        before = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(before.st_mode):
            raise CgroupObservationError("cgroup_observation_invalid")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise CgroupObservationError("cgroup_observation_invalid")
        payload = _read_descriptor_bounded(descriptor)
        current = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if _metadata_identity(before) != _metadata_identity(
            opened
        ) or _metadata_identity(before) != _metadata_identity(current):
            raise CgroupObservationError("cgroup_observation_invalid")
        return payload
    except CgroupObservationError:
        raise
    except (OSError, RuntimeError, ValueError, OverflowError):
        raise CgroupObservationError("cgroup_observation_invalid") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _ascii_lines(payload: bytes) -> tuple[str, ...]:
    try:
        value = payload.decode("ascii", errors="strict")
    except UnicodeError:
        raise CgroupObservationError("cgroup_observation_invalid") from None
    if not value or not value.endswith("\n") or "\r" in value:
        raise CgroupObservationError("cgroup_observation_invalid")
    lines = tuple(value[:-1].split("\n"))
    if not lines or any(not line for line in lines):
        raise CgroupObservationError("cgroup_observation_invalid")
    return lines


def _counter(value: str) -> int:
    if not value or not value.isascii() or not value.isdecimal():
        raise CgroupObservationError("cgroup_observation_invalid")
    try:
        parsed = int(value, 10)
    except (ValueError, OverflowError):
        raise CgroupObservationError("cgroup_observation_invalid") from None
    if parsed > MAX_CGROUP_COUNTER:
        raise CgroupObservationError("cgroup_observation_invalid")
    return parsed


def _single_counter(payload: bytes) -> int:
    lines = _ascii_lines(payload)
    if len(lines) != 1:
        raise CgroupObservationError("cgroup_observation_invalid")
    return _counter(lines[0])


def _cpu_counters(payload: bytes) -> Mapping[str, int]:
    counters: dict[str, int] = {}
    # Kernel releases may add keyed statistics. Validate every row, but retain
    # only the three counters the cgroup-v2 contract guarantees universally.
    for line in _ascii_lines(payload):
        fields = line.split(" ")
        if (
            len(fields) != 2
            or not _COUNTER_KEY_RE.fullmatch(fields[0])
            or fields[0] in counters
        ):
            raise CgroupObservationError("cgroup_observation_invalid")
        counters[fields[0]] = _counter(fields[1])
    if any(name not in counters for name in _REQUIRED_CPU_COUNTERS):
        raise CgroupObservationError("cgroup_observation_invalid")
    return counters


@dataclass(frozen=True)
class CgroupResourceSnapshot:
    memory_current_bytes: int
    memory_peak_bytes: int
    cpu_usage_usec: int
    cpu_user_usec: int
    cpu_system_usec: int

    def as_dict(self) -> dict[str, int]:
        return {
            "memory_current_bytes": self.memory_current_bytes,
            "memory_peak_bytes": self.memory_peak_bytes,
            "cpu_usage_usec": self.cpu_usage_usec,
            "cpu_user_usec": self.cpu_user_usec,
            "cpu_system_usec": self.cpu_system_usec,
        }


@dataclass
class CgroupScopeObserver:
    """Own a pinned scope descriptor and reject migration or replacement."""

    _scope_path: Path = field(repr=False)
    _proc_cgroup_path: Path = field(repr=False)
    _expected_proc_cgroup: bytes = field(repr=False)
    _directory_descriptor: int = field(repr=False)
    _scope_identity: tuple[int, int, int] = field(repr=False)
    _previous: CgroupResourceSnapshot | None = field(default=None, repr=False)
    _closed: bool = field(default=False, repr=False)

    @classmethod
    def bind(
        cls,
        *,
        scope_path: Path,
        proc_cgroup_path: Path,
        expected_proc_cgroup: bytes,
    ) -> CgroupScopeObserver:
        descriptor = -1
        try:
            before = scope_path.lstat()
            resolved = scope_path.resolve(strict=True)
            if resolved != scope_path.absolute() or not stat.S_ISDIR(before.st_mode):
                raise CgroupObservationError("cgroup_observation_invalid")
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_DIRECTORY", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(scope_path, flags)
            opened = os.fstat(descriptor)
            current = scope_path.lstat()
            if (
                not stat.S_ISDIR(opened.st_mode)
                or _directory_identity(before) != _directory_identity(opened)
                or _directory_identity(before) != _directory_identity(current)
                or _read_path_bounded(proc_cgroup_path) != expected_proc_cgroup
            ):
                raise CgroupObservationError("cgroup_observation_invalid")
            result = cls(
                _scope_path=scope_path,
                _proc_cgroup_path=proc_cgroup_path,
                _expected_proc_cgroup=bytes(expected_proc_cgroup),
                _directory_descriptor=descriptor,
                _scope_identity=_directory_identity(opened),
            )
            descriptor = -1
            return result
        except CgroupObservationError:
            raise
        except (OSError, RuntimeError, ValueError, OverflowError):
            raise CgroupObservationError("cgroup_observation_invalid") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def sample(self) -> CgroupResourceSnapshot:
        if self._closed or self._directory_descriptor < 0:
            raise CgroupObservationError("cgroup_observation_invalid")
        self._assert_membership_and_identity()
        current = _single_counter(
            _read_at_bounded(self._directory_descriptor, "memory.current")
        )
        peak = _single_counter(
            _read_at_bounded(self._directory_descriptor, "memory.peak")
        )
        cpu = _cpu_counters(_read_at_bounded(self._directory_descriptor, "cpu.stat"))
        self._assert_membership_and_identity()
        snapshot = CgroupResourceSnapshot(
            memory_current_bytes=current,
            memory_peak_bytes=peak,
            cpu_usage_usec=cpu["usage_usec"],
            cpu_user_usec=cpu["user_usec"],
            cpu_system_usec=cpu["system_usec"],
        )
        previous = self._previous
        if peak < current or (
            previous is not None
            and (
                snapshot.memory_peak_bytes < previous.memory_peak_bytes
                or snapshot.cpu_usage_usec < previous.cpu_usage_usec
                or snapshot.cpu_user_usec < previous.cpu_user_usec
                or snapshot.cpu_system_usec < previous.cpu_system_usec
            )
        ):
            raise CgroupObservationError("cgroup_observation_invalid")
        self._previous = snapshot
        return snapshot

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        descriptor = self._directory_descriptor
        self._directory_descriptor = -1
        if descriptor >= 0:
            os.close(descriptor)

    def _assert_membership_and_identity(self) -> None:
        try:
            current = self._scope_path.lstat()
            opened = os.fstat(self._directory_descriptor)
            if (
                _directory_identity(current) != self._scope_identity
                or _directory_identity(opened) != self._scope_identity
                or _read_path_bounded(self._proc_cgroup_path)
                != self._expected_proc_cgroup
            ):
                raise CgroupObservationError("cgroup_observation_invalid")
        except CgroupObservationError:
            raise
        except (OSError, RuntimeError, ValueError, OverflowError):
            raise CgroupObservationError("cgroup_observation_invalid") from None

    def __enter__(self) -> CgroupScopeObserver:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
