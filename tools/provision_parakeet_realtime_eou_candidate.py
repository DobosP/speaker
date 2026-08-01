"""Bind an existing Parakeet runtime and model without installing either.

The caller supplies a private Python 3.12 runtime, its already measured
wheelhouse, and the already downloaded exact ``.nemo`` artifact.  This tool
only verifies and receipts those inputs.  It never resolves, downloads,
installs, imports, or loads candidate code or model weights.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Iterator, Mapping, Sequence

from tools.streaming_stt.bounded_io import (
    BoundedReadError,
    hash_regular_bounded,
    opened_directory_nofollow,
    read_regular_bounded,
)
from tools.streaming_stt.manifest import (
    MAX_PYTHON_BYTES,
    MAX_WORKER_BYTES,
    PARAKEET_REALTIME_EOU_ADAPTER,
    PARAKEET_REALTIME_EOU_ARTIFACT_NAMES,
    PARAKEET_REALTIME_EOU_CUDA_VERSION,
    PARAKEET_REALTIME_EOU_MODEL_RECEIPT,
    PARAKEET_REALTIME_EOU_NEMO_VERSION,
    PARAKEET_REALTIME_EOU_NUMPY_VERSION,
    PARAKEET_REALTIME_EOU_PYTHON_VERSION,
    PARAKEET_REALTIME_EOU_RUNTIME_CONTENT_SHA256,
    PARAKEET_REALTIME_EOU_RUNTIME_FILE_COUNT,
    PARAKEET_REALTIME_EOU_RUNTIME_MAXIMUM_FILE_BYTES,
    PARAKEET_REALTIME_EOU_RUNTIME_TOTAL_SIZE_BYTES,
    PARAKEET_REALTIME_EOU_TORCH_VERSION,
    PARAKEET_REALTIME_EOU_WHEEL_LOCK_SHA256,
    PARAKEET_REALTIME_EOU_WHEEL_LOCK_SIZE_BYTES,
    ManifestError,
    ParakeetRealtimeEouConfig,
    WorkerManifest,
    artifact_maximum_bytes,
    load_worker_manifest,
)
from tools.streaming_stt.runtime_receipt import (
    PARAKEET_RUNTIME_TREE_LIMITS,
    RuntimeTreeReceiptError,
    generate_runtime_tree_receipt,
    load_runtime_tree_receipt,
    verify_isolated_venv_marker,
    verify_runtime_tree_receipt,
    verify_venv_runtime_location,
)
from tools.streaming_stt.wheel_lock import (
    PARAKEET_WHEEL_ARCHIVE_POLICY,
    WheelLockError,
    load_wheel_lock,
    verify_locked_wheel_install,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_PRODUCTION_VENV = _REPO_ROOT / ".venv"
_FIXED_WORKER = _REPO_ROOT / "tools" / "streaming_stt" / "worker.py"
_REPOSITORY_WHEEL_LOCK = (
    _REPO_ROOT
    / "tools"
    / "streaming_stt"
    / "parakeet-realtime-eou-runtime-wheels.lock.json"
)
_PYTHON_DIRECTORY_RE = re.compile(r"python[0-9]+\.[0-9]+\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_MODEL_FILENAME = "parakeet_realtime_eou_120m-v1.nemo"
_EXPECTED_INTERPRETER = Path("/usr/bin/python3.12")
_TRUSTED_INTERPRETER = Path(sys.executable).resolve(strict=True)
_PENDING_MANIFEST_FILENAME = ".worker-manifest.pending"
_OUTPUT_FILENAMES = {
    "parakeet-realtime-eou-runtime-wheels.lock.json",
    "runtime-receipt.json",
    "worker-manifest.json",
}
_PREPUBLICATION_FILENAMES = {
    "parakeet-realtime-eou-runtime-wheels.lock.json",
    "runtime-receipt.json",
    _PENDING_MANIFEST_FILENAME,
}
_SAFE_ERROR = {
    "ok": False,
    "error": "parakeet_candidate_prerequisites_unavailable",
}


class ProvisionError(RuntimeError):
    """A detail-free unsafe, incomplete, or changed provision input."""


@dataclass(frozen=True)
class _MarkerBinding:
    path: Path = field(repr=False)
    data: bytes = field(repr=False)
    identity: tuple[int, ...] = field(repr=False)
    sha256: str
    size_bytes: int

    def as_payload(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class _PendingManifestBinding:
    path: Path = field(repr=False)
    parent_descriptor: int = field(repr=False)
    descriptor: int = field(repr=False)
    data: bytes = field(repr=False)
    identity: tuple[int, ...] = field(repr=False)
    sha256: str


def _absolute(path: Path | str) -> Path:
    try:
        candidate = Path(os.path.abspath(Path(path).expanduser()))
    except (OSError, RuntimeError, TypeError, ValueError):
        raise ProvisionError() from None
    if not candidate.is_absolute():
        raise ProvisionError()
    return candidate


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _require_outside_production(path: Path) -> None:
    production = _absolute(_PRODUCTION_VENV)
    if _inside(path, production) or _inside(production, path):
        raise ProvisionError()


def _canonical_directory(path: Path | str, *, require_private: bool) -> Path:
    candidate = _absolute(path)
    _require_outside_production(candidate)
    try:
        with opened_directory_nofollow(
            candidate,
            require_private=require_private,
        ) as (stable, _descriptor):
            return stable
    except BoundedReadError:
        raise ProvisionError() from None


def _directory_names(path: Path) -> tuple[str, ...]:
    try:
        with opened_directory_nofollow(path) as (_stable, descriptor):
            names = tuple(os.listdir(descriptor))
    except (OSError, BoundedReadError):
        raise ProvisionError() from None
    if any(
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\x00" in name
        for name in names
    ):
        raise ProvisionError()
    return names


def _runtime_layout(venv_root: Path) -> tuple[Path, Path, Path]:
    if set(_directory_names(venv_root)) != {"bin", "lib", "pyvenv.cfg"}:
        raise ProvisionError()
    binary_root = _canonical_directory(venv_root / "bin", require_private=False)
    if set(_directory_names(binary_root)) != {"python"}:
        raise ProvisionError()
    lib = _canonical_directory(venv_root / "lib", require_private=False)
    python_directories = {
        name for name in _directory_names(lib) if _PYTHON_DIRECTORY_RE.fullmatch(name)
    }
    if python_directories != {"python3.12"}:
        raise ProvisionError()
    python_root = _canonical_directory(lib / "python3.12", require_private=False)
    if set(_directory_names(python_root)) != {"site-packages"}:
        raise ProvisionError()
    site_packages = _canonical_directory(
        python_root / "site-packages",
        require_private=True,
    )
    return binary_root / "python", venv_root / "pyvenv.cfg", site_packages


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
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


def _stable_inode_identity(metadata: os.stat_result) -> tuple[int, ...]:
    """Return identity fields that do not change when one hard link is added."""

    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        stat.S_IMODE(metadata.st_mode),
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _directory_inode_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
    )


def _exact_marker_values(data: bytes) -> dict[str, str]:
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeError:
        raise ProvisionError() from None
    expected = {
        "home": str(_EXPECTED_INTERPRETER.parent),
        "include-system-site-packages": "false",
        "version": PARAKEET_REALTIME_EOU_PYTHON_VERSION,
        "executable": str(_EXPECTED_INTERPRETER),
    }
    values: dict[str, str] = {}
    lines = text.splitlines()
    if len(lines) != len(expected):
        raise ProvisionError()
    for raw_line in lines:
        if not raw_line or "=" not in raw_line:
            raise ProvisionError()
        key, value = raw_line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key not in expected or key in values or not value:
            raise ProvisionError()
        values[key] = value
    if values != expected:
        raise ProvisionError()
    return values


def _exact_marker_binding(path: Path) -> _MarkerBinding:
    try:
        before = path.lstat()
        verify_isolated_venv_marker(path)
        snapshot = read_regular_bounded(
            path,
            maximum_bytes=artifact_maximum_bytes(
                PARAKEET_REALTIME_EOU_ADAPTER,
                "venv-marker",
            ),
        )
        after = path.lstat()
    except (
        OSError,
        RuntimeError,
        ValueError,
        BoundedReadError,
        ManifestError,
        RuntimeTreeReceiptError,
    ):
        raise ProvisionError() from None
    identity = _file_identity(before)
    if (
        snapshot.path != path
        or identity != _file_identity(after)
        or not stat.S_ISREG(after.st_mode)
        or after.st_nlink != 1
        or stat.S_IMODE(after.st_mode) & 0o077
        or (hasattr(os, "geteuid") and after.st_uid != os.geteuid())
    ):
        raise ProvisionError()
    _exact_marker_values(snapshot.data)
    return _MarkerBinding(
        path=path,
        data=snapshot.data,
        identity=identity,
        sha256=hashlib.sha256(snapshot.data).hexdigest(),
        size_bytes=len(snapshot.data),
    )


def _runtime_python_binding(path: Path) -> dict[str, object]:
    try:
        trusted = _TRUSTED_INTERPRETER.resolve(strict=True)
        resolved = path.resolve(strict=True)
        trusted_metadata = trusted.lstat()
        resolved_metadata = resolved.lstat()
        lexical_metadata = path.lstat()
        if (
            trusted != _TRUSTED_INTERPRETER
            or trusted != _EXPECTED_INTERPRETER
            or not stat.S_ISLNK(lexical_metadata.st_mode)
            or lexical_metadata.st_nlink != 1
            or not stat.S_ISREG(resolved_metadata.st_mode)
            or stat.S_IMODE(resolved_metadata.st_mode) & 0o111 == 0
            or not os.access(path, os.X_OK)
            or resolved != trusted
            or _file_identity(resolved_metadata) != _file_identity(trusted_metadata)
        ):
            raise ProvisionError()
        return _bound_payload(
            path,
            maximum_bytes=MAX_PYTHON_BYTES,
            allow_final_symlink=True,
        )
    except (OSError, RuntimeError, ValueError, BoundedReadError, ProvisionError):
        raise ProvisionError() from None


def _bound_payload(
    path: Path,
    *,
    maximum_bytes: int,
    allow_final_symlink: bool = False,
) -> dict[str, object]:
    try:
        size = path.resolve(strict=True).stat().st_size
        digest = hash_regular_bounded(
            path,
            maximum_bytes=maximum_bytes,
            expected_bytes=size,
            allow_final_symlink=allow_final_symlink,
        )
    except (OSError, RuntimeError, ValueError, BoundedReadError):
        raise ProvisionError() from None
    return {
        "path": str(path),
        "sha256": digest.sha256,
        "size_bytes": digest.size_bytes,
    }


def _exact_model_binding(model_root: Path) -> dict[str, object]:
    if set(_directory_names(model_root)) != {_MODEL_FILENAME}:
        raise ProvisionError()
    path = model_root / _MODEL_FILENAME
    expected_sha256, expected_size = PARAKEET_REALTIME_EOU_MODEL_RECEIPT
    if (
        not isinstance(expected_sha256, str)
        or _SHA256_RE.fullmatch(expected_sha256) is None
        or isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size <= 0
        or expected_size
        > artifact_maximum_bytes(PARAKEET_REALTIME_EOU_ADAPTER, "model-nemo")
    ):
        raise ProvisionError()
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ProvisionError()
        digest = hash_regular_bounded(
            path,
            maximum_bytes=artifact_maximum_bytes(
                PARAKEET_REALTIME_EOU_ADAPTER,
                "model-nemo",
            ),
            expected_bytes=expected_size,
        )
    except (OSError, RuntimeError, ValueError, BoundedReadError, ProvisionError):
        raise ProvisionError() from None
    if (
        digest.sha256 != expected_sha256
        or digest.path != path
        or set(_directory_names(model_root)) != {_MODEL_FILENAME}
    ):
        raise ProvisionError()
    return {
        "path": str(path),
        "sha256": digest.sha256,
        "size_bytes": digest.size_bytes,
    }


def _repository_wheel_lock_bytes() -> bytes:
    try:
        snapshot = read_regular_bounded(
            _REPOSITORY_WHEEL_LOCK,
            maximum_bytes=artifact_maximum_bytes(
                PARAKEET_REALTIME_EOU_ADAPTER,
                "runtime-wheel-lock",
            ),
        )
        metadata = snapshot.path.lstat()
    except (OSError, RuntimeError, ValueError, BoundedReadError, ManifestError):
        raise ProvisionError() from None
    if (
        snapshot.path != _REPOSITORY_WHEEL_LOCK
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or len(snapshot.data) != PARAKEET_REALTIME_EOU_WHEEL_LOCK_SIZE_BYTES
        or hashlib.sha256(snapshot.data).hexdigest()
        != PARAKEET_REALTIME_EOU_WHEEL_LOCK_SHA256
    ):
        raise ProvisionError()
    return snapshot.data


def _write_new_private(path: Path, payload: bytes) -> tuple[int, ...]:
    descriptor = -1
    identity: tuple[int, ...] | None = None
    try:
        with opened_directory_nofollow(path.parent) as (_parent, parent_descriptor):
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
                raise ProvisionError()
            with os.fdopen(descriptor, "wb", buffering=0) as handle:
                descriptor = -1
                if handle.write(payload) != len(payload):
                    raise ProvisionError()
                os.fsync(handle.fileno())
                written = os.fstat(handle.fileno())
                if (
                    not stat.S_ISREG(written.st_mode)
                    or written.st_nlink != 1
                    or stat.S_IMODE(written.st_mode) != 0o600
                    or written.st_size != len(payload)
                    or (
                        hasattr(os, "geteuid")
                        and written.st_uid != os.geteuid()
                    )
                ):
                    raise ProvisionError()
                identity = _stable_inode_identity(written)
            os.fsync(parent_descriptor)
    except (OSError, ValueError, OverflowError, BoundedReadError, ProvisionError):
        raise ProvisionError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
    if identity is None:
        raise ProvisionError()
    return identity


def _post_validation_fault_point() -> None:
    """Test seam reached after linking but before accepting publication."""


def _private_regular_at(descriptor: int, name: str) -> os.stat_result:
    try:
        metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
    except OSError:
        raise ProvisionError() from None
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
    ):
        raise ProvisionError()
    return metadata


def _read_bound_manifest_descriptor(
    descriptor: int,
    expected_data: bytes,
) -> os.stat_result:
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
            or (hasattr(os, "geteuid") and before.st_uid != os.geteuid())
            or before.st_size != len(expected_data)
        ):
            raise ProvisionError()
        chunks: list[bytes] = []
        offset = 0
        remaining = len(expected_data) + 1
        while remaining:
            chunk = os.pread(descriptor, min(remaining, 64 * 1024), offset)
            if not chunk:
                break
            chunks.append(chunk)
            offset += len(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    except (OSError, OverflowError, ProvisionError):
        raise ProvisionError() from None
    if (
        _file_identity(after) != _file_identity(before)
        or b"".join(chunks) != expected_data
    ):
        raise ProvisionError()
    return after


def _require_bound_parent(binding: _PendingManifestBinding) -> None:
    try:
        lexical = binding.path.parent.lstat()
        opened = os.fstat(binding.parent_descriptor)
        resolved = binding.path.parent.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        raise ProvisionError() from None
    if (
        resolved != binding.path.parent
        or not stat.S_ISDIR(opened.st_mode)
        or _directory_inode_identity(lexical)
        != _directory_inode_identity(opened)
    ):
        raise ProvisionError()


def _verify_pending_manifest_binding(
    binding: _PendingManifestBinding,
    *,
    expected_links: int,
) -> None:
    _require_bound_parent(binding)
    held = _read_bound_manifest_descriptor(binding.descriptor, binding.data)
    current_descriptor = -1
    try:
        current = _private_regular_at(
            binding.parent_descriptor,
            binding.path.name,
        )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        current_descriptor = os.open(
            binding.path.name,
            flags,
            dir_fd=binding.parent_descriptor,
        )
        opened = _read_bound_manifest_descriptor(
            current_descriptor,
            binding.data,
        )
    except (OSError, OverflowError, ProvisionError):
        raise ProvisionError() from None
    finally:
        if current_descriptor >= 0:
            try:
                os.close(current_descriptor)
            except OSError:
                pass
    if (
        held.st_nlink != expected_links
        or current.st_nlink != expected_links
        or opened.st_nlink != expected_links
        or _stable_inode_identity(held) != binding.identity
        or _stable_inode_identity(current) != binding.identity
        or _stable_inode_identity(opened) != binding.identity
    ):
        raise ProvisionError()


def _verify_canonical_manifest_binding(
    binding: _PendingManifestBinding,
    canonical: Path,
    *,
    expected_links: int,
) -> None:
    _require_bound_parent(binding)
    descriptor = -1
    try:
        current = _private_regular_at(
            binding.parent_descriptor,
            canonical.name,
        )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(
            canonical.name,
            flags,
            dir_fd=binding.parent_descriptor,
        )
        opened = _read_bound_manifest_descriptor(descriptor, binding.data)
    except (OSError, OverflowError, ProvisionError):
        raise ProvisionError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
    if (
        current.st_nlink != expected_links
        or opened.st_nlink != expected_links
        or _stable_inode_identity(current) != binding.identity
        or _stable_inode_identity(opened) != binding.identity
    ):
        raise ProvisionError()


@contextmanager
def _opened_pending_manifest_binding(
    path: Path,
    expected_data: bytes,
) -> Iterator[_PendingManifestBinding]:
    descriptor = -1
    try:
        with opened_directory_nofollow(
            path.parent,
            require_private=True,
        ) as (parent, parent_descriptor):
            if parent != path.parent or not expected_data:
                raise ProvisionError()
            before = _private_regular_at(parent_descriptor, path.name)
            if before.st_nlink != 1 or before.st_size != len(expected_data):
                raise ProvisionError()
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path.name, flags, dir_fd=parent_descriptor)
            opened = _read_bound_manifest_descriptor(descriptor, expected_data)
            current = _private_regular_at(parent_descriptor, path.name)
            if (
                _file_identity(opened) != _file_identity(before)
                or _file_identity(current) != _file_identity(before)
            ):
                raise ProvisionError()
            binding = _PendingManifestBinding(
                path=path,
                parent_descriptor=parent_descriptor,
                descriptor=descriptor,
                data=expected_data,
                identity=_stable_inode_identity(opened),
                sha256=hashlib.sha256(expected_data).hexdigest(),
            )
            _verify_pending_manifest_binding(binding, expected_links=1)
            yield binding
    except (OSError, RuntimeError, ValueError, BoundedReadError, ProvisionError):
        raise ProvisionError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _same_inode(metadata: os.stat_result, identity: tuple[int, ...]) -> bool:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
    ) == identity[:3]


def _unlink_bound_output_entry(
    descriptor: int,
    name: str,
    identity: tuple[int, ...],
) -> bool:
    """Unlink only when the directory entry still names the held inode."""

    try:
        metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError:
        raise ProvisionError() from None
    if not _same_inode(metadata, identity):
        return False
    try:
        os.unlink(name, dir_fd=descriptor)
    except OSError:
        raise ProvisionError() from None
    return True


def _discard_pending_manifest(path: Path, identity: tuple[int, ...]) -> None:
    try:
        with opened_directory_nofollow(
            path.parent,
            require_private=True,
        ) as (_parent, parent_descriptor):
            if _unlink_bound_output_entry(
                parent_descriptor,
                path.name,
                identity,
            ):
                os.fsync(parent_descriptor)
    except (OSError, BoundedReadError, ProvisionError):
        raise ProvisionError() from None


def _publish_pending_manifest(
    binding: _PendingManifestBinding,
    canonical: Path,
) -> None:
    pending = binding.path
    if pending.parent != canonical.parent or pending.name == canonical.name:
        raise ProvisionError()
    try:
        _verify_pending_manifest_binding(binding, expected_links=1)
        parent_descriptor = binding.parent_descriptor
        try:
            os.stat(
                canonical.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise ProvisionError()
        if set(os.listdir(parent_descriptor)) != _PREPUBLICATION_FILENAMES:
            raise ProvisionError()
        success = False
        canonical_linked = False
        try:
            os.link(
                f"/proc/self/fd/{binding.descriptor}",
                canonical.name,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=True,
            )
            canonical_linked = True
            _post_validation_fault_point()
            _verify_pending_manifest_binding(binding, expected_links=2)
            _verify_canonical_manifest_binding(
                binding,
                canonical,
                expected_links=2,
            )
            if set(os.listdir(parent_descriptor)) != (
                _PREPUBLICATION_FILENAMES | {canonical.name}
            ):
                raise ProvisionError()
            os.fsync(parent_descriptor)
            if not _unlink_bound_output_entry(
                parent_descriptor,
                pending.name,
                binding.identity,
            ):
                raise ProvisionError()
            os.fsync(parent_descriptor)
            held = _read_bound_manifest_descriptor(
                binding.descriptor,
                binding.data,
            )
            _verify_canonical_manifest_binding(
                binding,
                canonical,
                expected_links=1,
            )
            if (
                held.st_nlink != 1
                or _stable_inode_identity(held) != binding.identity
                or set(os.listdir(parent_descriptor)) != _OUTPUT_FILENAMES
            ):
                raise ProvisionError()
            success = True
        finally:
            if not success:
                if canonical_linked:
                    _unlink_bound_output_entry(
                        parent_descriptor,
                        canonical.name,
                        binding.identity,
                    )
                _unlink_bound_output_entry(
                    parent_descriptor,
                    pending.name,
                    binding.identity,
                )
                try:
                    os.fsync(parent_descriptor)
                except OSError:
                    pass
    except (OSError, RuntimeError, ValueError, BoundedReadError, ProvisionError):
        raise ProvisionError() from None


def _new_private_output(
    path: Path | str,
    *,
    forbidden_roots: tuple[Path, ...],
) -> Path:
    candidate = _absolute(path)
    _require_outside_production(candidate)
    if (
        not candidate.name
        or candidate.name in {".", ".."}
        or "\x00" in candidate.name
        or any(_inside(candidate, root) for root in forbidden_roots)
    ):
        raise ProvisionError()
    try:
        with opened_directory_nofollow(candidate.parent) as (
            _parent,
            parent_descriptor,
        ):
            os.mkdir(candidate.name, mode=0o700, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
        stable = _canonical_directory(candidate, require_private=True)
        if stat.S_IMODE(stable.stat().st_mode) != 0o700:
            raise ProvisionError()
        return stable
    except (OSError, BoundedReadError, ProvisionError):
        raise ProvisionError() from None


def _manifest_bytes(payload: Mapping[str, object]) -> bytes:
    try:
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
    except (TypeError, ValueError, OverflowError):
        raise ProvisionError() from None


def provision_candidate(
    *,
    venv_root: Path | str,
    model_root: Path | str,
    wheelhouse: Path | str,
    output_dir: Path | str,
) -> WorkerManifest:
    """Verify supplied offline inputs and create one schema-v5 manifest."""

    try:
        config = ParakeetRealtimeEouConfig(
            python_version=PARAKEET_REALTIME_EOU_PYTHON_VERSION,
            nemo_version=PARAKEET_REALTIME_EOU_NEMO_VERSION,
            torch_version=PARAKEET_REALTIME_EOU_TORCH_VERSION,
            cuda_version=PARAKEET_REALTIME_EOU_CUDA_VERSION,
            numpy_version=PARAKEET_REALTIME_EOU_NUMPY_VERSION,
            wheel_lock_sha256=PARAKEET_REALTIME_EOU_WHEEL_LOCK_SHA256,
            runtime_content_sha256=PARAKEET_REALTIME_EOU_RUNTIME_CONTENT_SHA256,
            runtime_file_count=PARAKEET_REALTIME_EOU_RUNTIME_FILE_COUNT,
            runtime_total_size_bytes=PARAKEET_REALTIME_EOU_RUNTIME_TOTAL_SIZE_BYTES,
            runtime_maximum_file_bytes=(
                PARAKEET_REALTIME_EOU_RUNTIME_MAXIMUM_FILE_BYTES
            ),
        )
    except (TypeError, ValueError, ManifestError):
        raise ProvisionError() from None
    venv = _canonical_directory(venv_root, require_private=True)
    selected_model_root = _canonical_directory(model_root, require_private=True)
    selected_wheelhouse = _canonical_directory(wheelhouse, require_private=True)
    roots = (venv, selected_model_root, selected_wheelhouse)
    if any(
        left != right and (_inside(left, right) or _inside(right, left))
        for index, left in enumerate(roots)
        for right in roots[index + 1 :]
    ):
        raise ProvisionError()
    python, marker, site_packages = _runtime_layout(venv)
    python_binding = _runtime_python_binding(python)
    marker_binding = _exact_marker_binding(marker)
    model_binding = _exact_model_binding(selected_model_root)
    repository_lock = _repository_wheel_lock_bytes()
    worker_binding = _bound_payload(
        _FIXED_WORKER,
        maximum_bytes=MAX_WORKER_BYTES,
    )

    destination = _new_private_output(output_dir, forbidden_roots=roots)
    runtime_wheel_lock = (
        destination / "parakeet-realtime-eou-runtime-wheels.lock.json"
    )
    _write_new_private(runtime_wheel_lock, repository_lock)
    try:
        lock = load_wheel_lock(
            runtime_wheel_lock,
            expected_digest=PARAKEET_REALTIME_EOU_WHEEL_LOCK_SHA256,
        )
        receipt = generate_runtime_tree_receipt(
            site_packages,
            destination / "runtime-receipt.json",
            limits=PARAKEET_RUNTIME_TREE_LIMITS,
        )
        verify_venv_runtime_location(receipt, python)
        expectation = verify_locked_wheel_install(
            lock,
            selected_wheelhouse,
            receipt,
            archive_policy=PARAKEET_WHEEL_ARCHIVE_POLICY,
        )
        verify_runtime_tree_receipt(
            receipt,
            limits=PARAKEET_RUNTIME_TREE_LIMITS,
        )
    except (RuntimeTreeReceiptError, WheelLockError):
        raise ProvisionError() from None
    if (
        lock.digest != PARAKEET_REALTIME_EOU_WHEEL_LOCK_SHA256
        or receipt.content_digest != PARAKEET_REALTIME_EOU_RUNTIME_CONTENT_SHA256
        or expectation.content_digest
        != PARAKEET_REALTIME_EOU_RUNTIME_CONTENT_SHA256
        or expectation.file_count != PARAKEET_REALTIME_EOU_RUNTIME_FILE_COUNT
        or expectation.total_size_bytes
        != PARAKEET_REALTIME_EOU_RUNTIME_TOTAL_SIZE_BYTES
        or expectation.maximum_file_bytes
        != PARAKEET_REALTIME_EOU_RUNTIME_MAXIMUM_FILE_BYTES
    ):
        raise ProvisionError()

    artifact_paths = {
        "runtime-receipt": receipt.receipt_path,
        "runtime-wheel-lock": runtime_wheel_lock,
    }
    artifact_bindings = {
        name: _bound_payload(
            artifact_paths[name],
            maximum_bytes=artifact_maximum_bytes(
                PARAKEET_REALTIME_EOU_ADAPTER,
                name,
            ),
        )
        for name in ("runtime-receipt", "runtime-wheel-lock")
    }
    artifact_bindings["venv-marker"] = marker_binding.as_payload()
    artifact_bindings["model-nemo"] = model_binding
    artifacts = [
        {"name": name, **artifact_bindings[name]}
        for name in PARAKEET_REALTIME_EOU_ARTIFACT_NAMES
    ]
    manifest_payload = {
        "schema_version": 5,
        "model_id": "parakeet-realtime-eou-120m-v1-en-cuda",
        "adapter": PARAKEET_REALTIME_EOU_ADAPTER,
        "adapter_config": config.as_dict(),
        "python": python_binding,
        "worker": worker_binding,
        "artifacts": artifacts,
        "limits": {
            "startup_timeout_sec": 300.0,
            "case_timeout_sec": 900.0,
        },
    }
    pending_manifest_path = destination / _PENDING_MANIFEST_FILENAME
    manifest_path = destination / "worker-manifest.json"
    expected_manifest_bytes = _manifest_bytes(manifest_payload)
    expected_manifest_digest = hashlib.sha256(expected_manifest_bytes).hexdigest()
    pending_identity = _write_new_private(
        pending_manifest_path,
        expected_manifest_bytes,
    )
    published = False
    try:
        with _opened_pending_manifest_binding(
            pending_manifest_path,
            expected_manifest_bytes,
        ) as pending_binding:
            pending_manifest = load_worker_manifest(pending_manifest_path)
            _verify_pending_manifest_binding(pending_binding, expected_links=1)
            current_receipt = load_runtime_tree_receipt(
                receipt.receipt_path,
                expected_digest=receipt.digest,
                limits=PARAKEET_RUNTIME_TREE_LIMITS,
            )
            verify_runtime_tree_receipt(
                current_receipt,
                limits=PARAKEET_RUNTIME_TREE_LIMITS,
            )
            current_worker = _bound_payload(
                _FIXED_WORKER,
                maximum_bytes=MAX_WORKER_BYTES,
            )
            current_python = _runtime_python_binding(python)
            current_marker = _exact_marker_binding(marker)
            current_model = _exact_model_binding(selected_model_root)
            if (
                pending_binding.sha256 != expected_manifest_digest
                or pending_manifest.digest != expected_manifest_digest
                or pending_manifest.path != pending_manifest_path
                or pending_manifest.schema_version != 5
                or pending_manifest.model_id
                != "parakeet-realtime-eou-120m-v1-en-cuda"
                or pending_manifest.adapter != PARAKEET_REALTIME_EOU_ADAPTER
                or pending_manifest.adapter_config != config
                or pending_manifest.limits.startup_timeout_sec != 300.0
                or pending_manifest.limits.case_timeout_sec != 900.0
                or pending_manifest.worker.sha256 != worker_binding["sha256"]
                or pending_manifest.worker.size_bytes != worker_binding["size_bytes"]
                or current_worker != worker_binding
                or current_python != python_binding
                or current_marker != marker_binding
                or current_model != model_binding
                or pending_manifest.artifact_by_name["runtime-receipt"].sha256
                != receipt.digest
                or pending_manifest.artifact_by_name["runtime-wheel-lock"].sha256
                != PARAKEET_REALTIME_EOU_WHEEL_LOCK_SHA256
                or set(_directory_names(destination)) != _PREPUBLICATION_FILENAMES
            ):
                raise ProvisionError()
            _publish_pending_manifest(pending_binding, manifest_path)
        published = True
        return replace(pending_manifest, path=manifest_path)
    except (ManifestError, RuntimeTreeReceiptError, ProvisionError):
        raise ProvisionError() from None
    finally:
        if not published:
            _discard_pending_manifest(pending_manifest_path, pending_identity)


def _safe_result(manifest: WorkerManifest) -> dict[str, object]:
    return {
        "ok": True,
        "adapter": manifest.adapter,
        "model_id": manifest.model_id,
        "manifest_sha256": manifest.digest,
        "runtime_receipt_sha256": manifest.artifact_by_name[
            "runtime-receipt"
        ].sha256,
        "worker_sha256": manifest.worker.sha256,
        "artifact_set_sha256": hashlib.sha256(
            json.dumps(
                [
                    {
                        "name": artifact.name,
                        "sha256": artifact.sha256,
                        "size_bytes": artifact.size_bytes,
                    }
                    for artifact in manifest.artifacts
                ],
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bind an existing private Python 3.12 Parakeet runtime, exact "
            "measured wheelhouse, and exact local .nemo artifact into a "
            "schema-v5 worker manifest. This command does not resolve, "
            "install, download, import, or load candidate code or weights."
        )
    )
    parser.add_argument("--venv-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = provision_candidate(
            venv_root=args.venv_root,
            model_root=args.model_root,
            wheelhouse=args.wheelhouse,
            output_dir=args.output_dir,
        )
        result: Mapping[str, object] = _safe_result(manifest)
        code = 0
    except Exception:  # noqa: BLE001 - private local paths remain undisclosed
        result = _SAFE_ERROR
        code = 2
    print(
        json.dumps(
            result,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
