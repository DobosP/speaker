"""Build one script-free candidate runtime from an exact verified wheelhouse.

The runtime worker imports packages by inserting a receipt-verified
``site-packages`` directory while Python runs with ``-I -S``.  It therefore
does not need installer-generated console launchers, cache metadata, activation
scripts, or external documentation schemes.  This builder extracts only the
release-owned import runtime, retains each wheel's original RECORD, and then
requires the exact installed-wheel ownership verifier to accept the result.
"""

from __future__ import annotations

import base64
import csv
from dataclasses import dataclass
import hashlib
import io
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Mapping
import zipfile

from .runtime_receipt import (
    RuntimeFileReceipt,
    RuntimeTreeReceipt,
    RuntimeTreeReceiptError,
    generate_runtime_tree_receipt,
    verify_venv_runtime_location,
)
from .wheel_lock import (
    InstalledContentExpectation,
    VerifiedWheel,
    WHEEL_LOCK_RUNTIME_TREE_LIMITS,
    NEMOTRON_WHEEL_ARCHIVE_POLICY,
    WheelArchivePolicy,
    WheelLock,
    WheelLockError,
    verify_installed_wheels,
    verify_wheelhouse,
)


_COPY_CHUNK_BYTES = 1024 * 1024
_MAXIMUM_PYTHON_BYTES = 512 * 1024 * 1024
_PYTHON_VERSION_RE = re.compile(
    r"(?P<major>0|[1-9][0-9]*)\."
    r"(?P<minor>0|[1-9][0-9]*)\."
    r"(?P<patch>0|[1-9][0-9]*)\Z"
)


class RuntimeBuildError(RuntimeError):
    """A detail-free unsafe input, archive change, or incomplete runtime."""


@dataclass(frozen=True)
class RuntimeBuildPolicy:
    """Exact interpreter identity and closed wheel archive policy."""

    python_version: str
    python_directory: str
    wheel_archive_policy: WheelArchivePolicy

    def __post_init__(self) -> None:
        match = (
            _PYTHON_VERSION_RE.fullmatch(self.python_version)
            if isinstance(self.python_version, str)
            else None
        )
        if (
            match is None
            or self.python_directory
            != f"python{match.group('major')}.{match.group('minor')}"
            or not isinstance(self.wheel_archive_policy, WheelArchivePolicy)
        ):
            raise RuntimeBuildError()


NEMOTRON_RUNTIME_BUILD_POLICY = RuntimeBuildPolicy(
    python_version="3.12.3",
    python_directory="python3.12",
    wheel_archive_policy=NEMOTRON_WHEEL_ARCHIVE_POLICY,
)


@dataclass(frozen=True)
class PreparedRuntime:
    """One exact minimal venv plus its verified installed-content expectation."""

    root: Path
    python: Path
    site_packages: Path
    receipt: RuntimeTreeReceipt
    expectation: InstalledContentExpectation
    wheel_lock_sha256: str


def _absolute(value: Path | str) -> Path:
    try:
        candidate = Path(os.path.abspath(Path(value).expanduser()))
    except (OSError, RuntimeError, TypeError, ValueError):
        raise RuntimeBuildError() from None
    if not candidate.is_absolute() or "\x00" in str(candidate):
        raise RuntimeBuildError()
    return candidate


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _identity(metadata: os.stat_result) -> tuple[int, ...]:
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


def _new_private_directory(path: Path) -> Path:
    if not path.name or path.name in {".", ".."}:
        raise RuntimeBuildError()
    try:
        parent = path.parent.resolve(strict=True)
        if parent != path.parent or not parent.is_dir():
            raise RuntimeBuildError()
        os.mkdir(path, mode=0o700)
        path.chmod(0o700)
        metadata = path.lstat()
    except (OSError, RuntimeError, ValueError, RuntimeBuildError):
        raise RuntimeBuildError() from None
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
    ):
        raise RuntimeBuildError()
    return path


def _mkdir_private(path: Path) -> None:
    try:
        path.mkdir(mode=0o700)
        metadata = path.lstat()
    except (OSError, RuntimeError, ValueError):
        raise RuntimeBuildError() from None
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise RuntimeBuildError()


def _write_new_private(path: Path, data: bytes) -> None:
    descriptor = -1
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", buffering=0) as handle:
            descriptor = -1
            handle.write(data)
            os.fsync(handle.fileno())
    except (OSError, RuntimeError, ValueError, OverflowError):
        raise RuntimeBuildError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _validated_python(
    path: Path | str,
    *,
    policy: RuntimeBuildPolicy,
) -> Path:
    candidate = _absolute(path)
    try:
        canonical = candidate.resolve(strict=True)
        metadata = canonical.stat()
    except (OSError, RuntimeError, ValueError):
        raise RuntimeBuildError() from None
    if (
        candidate != canonical
        or canonical.name != policy.python_directory
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink < 1
        or metadata.st_size <= 0
        or metadata.st_size > _MAXIMUM_PYTHON_BYTES
        or stat.S_IMODE(metadata.st_mode) & 0o111 == 0
    ):
        raise RuntimeBuildError()
    return canonical


def _create_minimal_venv(
    root: Path,
    system_python: Path,
    *,
    policy: RuntimeBuildPolicy,
) -> tuple[Path, Path]:
    bin_root = root / "bin"
    lib_root = root / "lib"
    python_root = lib_root / policy.python_directory
    site_packages = python_root / "site-packages"
    for path in (bin_root, lib_root, python_root, site_packages):
        _mkdir_private(path)
    python = bin_root / "python"
    try:
        python.symlink_to(system_python)
    except (OSError, RuntimeError, ValueError):
        raise RuntimeBuildError() from None
    marker = (
        f"home = {system_python.parent}\n"
        "include-system-site-packages = false\n"
        f"version = {policy.python_version}\n"
        f"executable = {system_python}\n"
    ).encode("utf-8")
    _write_new_private(root / "pyvenv.cfg", marker)
    return python, site_packages


def _ensure_private_parents(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or pure.as_posix() != relative:
        raise RuntimeBuildError()
    current = root
    for component in pure.parts[:-1]:
        candidate = current / component
        try:
            candidate.mkdir(mode=0o700)
        except FileExistsError:
            pass
        except OSError:
            raise RuntimeBuildError() from None
        try:
            metadata = candidate.lstat()
        except OSError:
            raise RuntimeBuildError() from None
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) & 0o077:
            raise RuntimeBuildError()
        current = candidate
    return current / pure.name


def _archive_destination(
    wheel: VerifiedWheel,
    archive_path: str,
) -> tuple[str, bool]:
    parts = PurePosixPath(archive_path).parts
    data_directory = f"{wheel.dist_info_directory.removesuffix('.dist-info')}.data"
    if parts[0] == data_directory:
        if len(parts) >= 3 and parts[1] in {"purelib", "platlib"}:
            return "/".join(parts[2:]), True
        exclusions = {
            receipt.relative_path for receipt in wheel.excluded_external_files
        }
        if archive_path in exclusions:
            return archive_path, False
        raise RuntimeBuildError()
    if parts[0].endswith(".data"):
        raise RuntimeBuildError()
    return archive_path, True


def _write_archive_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    destination: Path,
    *,
    expected: RuntimeFileReceipt | None,
) -> None:
    descriptor = -1
    digest = hashlib.sha256()
    written = 0
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(destination, flags, 0o600)
        os.fchmod(descriptor, 0o600)
        with archive.open(info, mode="r") as source:
            with os.fdopen(descriptor, "wb", buffering=0) as output:
                descriptor = -1
                while True:
                    chunk = source.read(_COPY_CHUNK_BYTES)
                    if not chunk:
                        break
                    written += len(chunk)
                    if (
                        written > info.file_size
                        or written > WHEEL_LOCK_RUNTIME_TREE_LIMITS.maximum_file_bytes
                    ):
                        raise RuntimeBuildError()
                    digest.update(chunk)
                    output.write(chunk)
                os.fsync(output.fileno())
    except (
        OSError,
        RuntimeError,
        ValueError,
        OverflowError,
        zipfile.BadZipFile,
        RuntimeBuildError,
    ):
        raise RuntimeBuildError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
    if written != info.file_size or (
        expected is not None
        and (written != expected.size_bytes or digest.hexdigest() != expected.sha256)
    ):
        raise RuntimeBuildError()


def _installed_record_bytes(wheel: VerifiedWheel) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    rows: list[list[str]] = []
    for row in wheel.release_rows:
        if row.sha256 is None:
            if row.installed_path != wheel.record_path or row.size_bytes is not None:
                raise RuntimeBuildError()
            rows.append([row.installed_path, "", ""])
            continue
        if row.size_bytes is None:
            raise RuntimeBuildError()
        encoded = base64.urlsafe_b64encode(bytes.fromhex(row.sha256)).rstrip(b"=")
        rows.append(
            [
                row.installed_path,
                f"sha256={encoded.decode('ascii')}",
                str(row.size_bytes),
            ]
        )
    rows.sort(key=lambda row: row[0])
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _extract_wheel(
    verified: VerifiedWheel,
    wheel_path: Path,
    site_packages: Path,
    *,
    expected_by_path: Mapping[str, RuntimeFileReceipt],
    written_paths: set[str],
) -> None:
    descriptor = -1
    try:
        before = wheel_path.lstat()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(wheel_path, flags)
        opened = os.fstat(descriptor)
        identity = _identity(opened)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size != verified.entry.size_bytes
            or identity != _identity(before)
        ):
            raise RuntimeBuildError()
        with os.fdopen(descriptor, "rb", buffering=0) as handle:
            descriptor = -1
            with zipfile.ZipFile(handle, mode="r") as archive:
                infos = archive.infolist()
                if len(infos) != verified.archive_member_count:
                    raise RuntimeBuildError()
                for info in infos:
                    if info.is_dir():
                        continue
                    archive_path = info.filename
                    relative, runtime_owned = _archive_destination(
                        verified,
                        archive_path,
                    )
                    if not runtime_owned:
                        continue
                    expected = expected_by_path.get(relative)
                    if relative == verified.record_path:
                        if expected is not None:
                            raise RuntimeBuildError()
                        continue
                    elif expected is None:
                        raise RuntimeBuildError()
                    if relative in written_paths:
                        # verify_wheelhouse permits overlap only for the exact
                        # lock-pinned, byte-identical NVIDIA namespace marker.
                        continue
                    destination = _ensure_private_parents(site_packages, relative)
                    _write_archive_member(
                        archive,
                        info,
                        destination,
                        expected=expected,
                    )
                    written_paths.add(relative)
                record_destination = _ensure_private_parents(
                    site_packages,
                    verified.record_path,
                )
                _write_new_private(
                    record_destination,
                    _installed_record_bytes(verified),
                )
                written_paths.add(verified.record_path)
            after = os.fstat(handle.fileno())
        current = wheel_path.lstat()
        if _identity(after) != identity or _identity(current) != identity:
            raise RuntimeBuildError()
    except (
        OSError,
        RuntimeError,
        ValueError,
        OverflowError,
        zipfile.BadZipFile,
        RuntimeBuildError,
    ):
        raise RuntimeBuildError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _verify_minimal_layout(
    root: Path,
    *,
    system_python: Path,
    policy: RuntimeBuildPolicy,
) -> None:
    try:
        if set(path.name for path in root.iterdir()) != {
            "bin",
            "lib",
            "pyvenv.cfg",
        }:
            raise RuntimeBuildError()
        if set(path.name for path in (root / "bin").iterdir()) != {"python"}:
            raise RuntimeBuildError()
        if set(path.name for path in (root / "lib").iterdir()) != {
            policy.python_directory
        }:
            raise RuntimeBuildError()
        if set(
            path.name
            for path in (root / "lib" / policy.python_directory).iterdir()
        ) != {"site-packages"}:
            raise RuntimeBuildError()
        python = root / "bin" / "python"
        if not python.is_symlink() or Path(os.readlink(python)) != system_python:
            raise RuntimeBuildError()
    except (OSError, RuntimeError, ValueError, RuntimeBuildError):
        raise RuntimeBuildError() from None


def prepare_locked_runtime(
    lock: WheelLock,
    wheelhouse: Path | str,
    *,
    system_python: Path | str,
    output_root: Path | str,
    receipt_path: Path | str,
    policy: RuntimeBuildPolicy = NEMOTRON_RUNTIME_BUILD_POLICY,
) -> PreparedRuntime:
    """Create and verify a new minimal runtime without running an installer."""

    if not isinstance(lock, WheelLock) or not isinstance(policy, RuntimeBuildPolicy):
        raise RuntimeBuildError()
    selected_wheelhouse = _absolute(wheelhouse)
    root = _absolute(output_root)
    selected_receipt_path = _absolute(receipt_path)
    python_target = _validated_python(system_python, policy=policy)
    production_venv = Path(__file__).resolve().parents[2] / ".venv"
    forbidden = (selected_wheelhouse, production_venv)
    if (
        any(_inside(root, item) or _inside(item, root) for item in forbidden)
        or any(
            _inside(selected_receipt_path, item) or _inside(item, selected_receipt_path)
            for item in forbidden
        )
        or _inside(selected_receipt_path, root)
        or selected_receipt_path.exists()
    ):
        raise RuntimeBuildError()
    try:
        verified = verify_wheelhouse(
            lock,
            selected_wheelhouse,
            archive_policy=policy.wheel_archive_policy,
        )
    except WheelLockError:
        raise RuntimeBuildError() from None

    root = _new_private_directory(root)
    python, site_packages = _create_minimal_venv(
        root,
        python_target,
        policy=policy,
    )
    expected_by_path = {
        receipt.relative_path: receipt
        for wheel in verified.wheels
        for receipt in wheel.release_files
    }
    expected_paths = {
        *expected_by_path,
        *(wheel.record_path for wheel in verified.wheels),
    }
    written_paths: set[str] = set()
    for wheel in verified.wheels:
        _extract_wheel(
            wheel,
            verified.root / wheel.entry.filename,
            site_packages,
            expected_by_path=expected_by_path,
            written_paths=written_paths,
        )
    if written_paths != expected_paths:
        raise RuntimeBuildError()

    try:
        receipt = generate_runtime_tree_receipt(
            site_packages,
            selected_receipt_path,
            limits=WHEEL_LOCK_RUNTIME_TREE_LIMITS,
        )
        verify_venv_runtime_location(receipt, python)
        expectation = verify_installed_wheels(verified, receipt)
    except (RuntimeTreeReceiptError, WheelLockError):
        raise RuntimeBuildError() from None
    if (
        expectation.content_digest != receipt.content_digest
        or expectation.file_count != receipt.file_count
        or expectation.total_size_bytes != receipt.total_size_bytes
    ):
        raise RuntimeBuildError()
    _verify_minimal_layout(
        root,
        system_python=python_target,
        policy=policy,
    )
    return PreparedRuntime(
        root=root,
        python=python,
        site_packages=site_packages,
        receipt=receipt,
        expectation=expectation,
        wheel_lock_sha256=lock.digest,
    )


__all__ = [
    "PreparedRuntime",
    "RuntimeBuildPolicy",
    "RuntimeBuildError",
    "NEMOTRON_RUNTIME_BUILD_POLICY",
    "prepare_locked_runtime",
]
