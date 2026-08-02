"""Bind an existing parakeet.cpp CPU candidate without executing it.

The caller supplies two closed private directories containing already-built
native libraries, their exact receipts, and the exact already-downloaded GGUF
model.  This verifier only hashes and receipts those inputs.  It never builds,
downloads, imports, executes, or dynamically loads candidate code or weights.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Mapping, Sequence

from tools.streaming_stt.bounded_io import (
    BoundedReadError,
    hash_regular_bounded,
    opened_directory_nofollow,
    read_regular_bounded,
)
from tools.streaming_stt.manifest import (
    MAX_PARAKEET_CPP_BRIDGE_SOURCE_BYTES,
    MAX_PARAKEET_CPP_CONTROL_ARTIFACT_BYTES,
    MAX_PYTHON_BYTES,
    MAX_WORKER_BYTES,
    PARAKEET_CPP_ADAPTER,
    PARAKEET_CPP_ARTIFACT_NAMES,
    PARAKEET_CPP_BRIDGE_NEEDED,
    PARAKEET_CPP_BRIDGE_RECEIPT,
    PARAKEET_CPP_BRIDGE_RUNPATH,
    PARAKEET_CPP_BRIDGE_SOURCE_SHA256,
    PARAKEET_CPP_GGML_GIT_TREE,
    PARAKEET_CPP_LIBPARAKEET_RECEIPT,
    PARAKEET_CPP_LIBPARAKEET_NEEDED,
    PARAKEET_CPP_MODEL_RECEIPT,
    PARAKEET_CPP_PARENT_GIT_TREE,
    PARAKEET_CPP_PATCHED_DIFF_SHA256,
    PARAKEET_CPP_PATCH_SPECS,
    ManifestError,
    ParakeetCppConfig,
    WorkerManifest,
    artifact_maximum_bytes,
    inspect_elf,
    load_worker_manifest,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXED_WORKER = _REPO_ROOT / "tools" / "streaming_stt" / "worker.py"
_FIXED_BRIDGE_SOURCE = (
    _REPO_ROOT / "tools" / "streaming_stt" / "native" / "parakeet_cpp_bridge.cpp"
)
_PENDING_MANIFEST_FILENAME = ".worker-manifest.pending"
_MANIFEST_FILENAME = "worker-manifest.json"
_NATIVE_FILENAMES = frozenset(
    {
        "source-receipt.json",
        "build-receipt.json",
        "libparakeet.so",
        "libspeaker_parakeet_bridge.so",
    }
)
_MODEL_FILENAMES = frozenset(
    {"model-receipt.json", "realtime_eou_120m-v1-f16.gguf"}
)
_ARTIFACT_FILENAMES = {
    "source-receipt": "source-receipt.json",
    "build-receipt": "build-receipt.json",
    "model-receipt": "model-receipt.json",
    "libparakeet": "libparakeet.so",
    "bridge-library": "libspeaker_parakeet_bridge.so",
    "model-gguf": "realtime_eou_120m-v1-f16.gguf",
}
_MODEL_ID = "parakeet-cpp-realtime-eou-120m-v1-f16-cpu"
_STARTUP_TIMEOUT_SEC = 120.0
_CASE_TIMEOUT_SEC = 300.0
_SAFE_ERROR = {
    "ok": False,
    "error": "parakeet_cpp_candidate_prerequisites_unavailable",
}


class ProvisionError(RuntimeError):
    """A detail-free unsafe, incomplete, or changed provision input."""


@dataclass(frozen=True)
class _DirectoryBinding:
    path: Path = field(repr=False)
    expected_names: frozenset[str]
    identity: tuple[int, ...] = field(repr=False)


@dataclass(frozen=True)
class _FileBinding:
    path: Path = field(repr=False)
    resolved_path: Path = field(repr=False)
    lexical_identity: tuple[int, ...] = field(repr=False)
    target_identity: tuple[int, ...] = field(repr=False)
    sha256: str
    size_bytes: int
    maximum_bytes: int = field(repr=False)
    allow_final_symlink: bool = field(repr=False)
    require_private: bool = field(repr=False)
    require_executable: bool = field(repr=False)
    expected_dependencies: tuple[str, ...] | None = field(repr=False)
    expected_runpath: str | None = field(repr=False)
    require_relro: bool = field(repr=False)
    require_bind_now: bool = field(repr=False)
    require_noexecstack: bool = field(repr=False)

    def as_payload(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class _InputState:
    config: ParakeetCppConfig
    native_root: _DirectoryBinding
    model_root: _DirectoryBinding
    python: _FileBinding
    worker: _FileBinding
    bridge_source: _FileBinding
    artifacts: tuple[tuple[str, _FileBinding], ...]

    @property
    def artifact_by_name(self) -> Mapping[str, _FileBinding]:
        return dict(self.artifacts)


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


def _overlaps(left: Path, right: Path) -> bool:
    return _inside(left, right) or _inside(right, left)


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


def _stable_file_identity(metadata: os.stat_result) -> tuple[int, ...]:
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


def _directory_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        stat.S_IMODE(metadata.st_mode),
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _bind_private_directory(
    path: Path | str,
    *,
    expected_names: frozenset[str],
) -> _DirectoryBinding:
    candidate = _absolute(path)
    try:
        with opened_directory_nofollow(
            candidate,
            require_private=True,
        ) as (stable, descriptor):
            before = os.fstat(descriptor)
            names = frozenset(os.listdir(descriptor))
            after = os.fstat(descriptor)
    except (OSError, RuntimeError, ValueError, BoundedReadError):
        raise ProvisionError() from None
    if (
        stable != candidate
        or not stat.S_ISDIR(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o700
        or (hasattr(os, "geteuid") and before.st_uid != os.geteuid())
        or _directory_identity(after) != _directory_identity(before)
        or names != expected_names
    ):
        raise ProvisionError()
    return _DirectoryBinding(
        path=stable,
        expected_names=expected_names,
        identity=_directory_identity(before),
    )


def _revalidate_directory(binding: _DirectoryBinding) -> None:
    current = _bind_private_directory(
        binding.path,
        expected_names=binding.expected_names,
    )
    if current != binding:
        raise ProvisionError()


def _bind_file(
    path: Path | str,
    *,
    maximum_bytes: int,
    allow_final_symlink: bool = False,
    require_private: bool = False,
    require_executable: bool = False,
    expected_dependencies: tuple[str, ...] | None = None,
    expected_runpath: str | None = None,
    require_relro: bool = False,
    require_bind_now: bool = False,
    require_noexecstack: bool = False,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
) -> _FileBinding:
    candidate = _absolute(path)
    try:
        lexical_before = candidate.lstat()
        if stat.S_ISLNK(lexical_before.st_mode):
            if not allow_final_symlink or lexical_before.st_nlink != 1:
                raise ProvisionError()
        elif (
            not stat.S_ISREG(lexical_before.st_mode)
            or lexical_before.st_nlink != 1
        ):
            raise ProvisionError()
        resolved = candidate.resolve(strict=True)
        target_before = resolved.lstat()
        size_bytes = target_before.st_size
        if (
            not stat.S_ISREG(target_before.st_mode)
            or target_before.st_nlink != 1
            or size_bytes <= 0
            or size_bytes > maximum_bytes
            or (
                expected_size_bytes is not None
                and size_bytes != expected_size_bytes
            )
            or (
                require_private
                and (
                    candidate != resolved
                    or stat.S_IMODE(target_before.st_mode) != 0o600
                    or (
                        hasattr(os, "geteuid")
                        and target_before.st_uid != os.geteuid()
                    )
                )
            )
            or (
                require_executable
                and stat.S_IMODE(target_before.st_mode) & 0o111 == 0
            )
        ):
            raise ProvisionError()
        digest = hash_regular_bounded(
            candidate,
            maximum_bytes=maximum_bytes,
            expected_bytes=size_bytes,
            allow_final_symlink=allow_final_symlink,
        )
        lexical_after = candidate.lstat()
        current_resolved = candidate.resolve(strict=True)
        target_after = current_resolved.lstat()
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        BoundedReadError,
        ProvisionError,
    ):
        raise ProvisionError() from None
    lexical_identity = _file_identity(lexical_before)
    target_identity = _file_identity(target_before)
    if (
        digest.path != resolved
        or digest.size_bytes != size_bytes
        or current_resolved != resolved
        or _file_identity(lexical_after) != lexical_identity
        or _file_identity(target_after) != target_identity
        or (expected_sha256 is not None and digest.sha256 != expected_sha256)
    ):
        raise ProvisionError()
    if expected_dependencies is not None:
        try:
            inspection = inspect_elf(
                resolved,
                expected_sha256=digest.sha256,
                expected_size_bytes=digest.size_bytes,
                maximum_bytes=maximum_bytes,
            )
        except ManifestError:
            raise ProvisionError() from None
        if (
            inspection.dependencies != expected_dependencies
            or (
                expected_runpath is not None
                and inspection.runpath != expected_runpath
            )
            or (require_relro and not inspection.relro)
            or (require_bind_now and not inspection.bind_now)
            or (require_noexecstack and not inspection.noexecstack)
        ):
            raise ProvisionError()
    return _FileBinding(
        path=candidate,
        resolved_path=resolved,
        lexical_identity=lexical_identity,
        target_identity=target_identity,
        sha256=digest.sha256,
        size_bytes=digest.size_bytes,
        maximum_bytes=maximum_bytes,
        allow_final_symlink=allow_final_symlink,
        require_private=require_private,
        require_executable=require_executable,
        expected_dependencies=expected_dependencies,
        expected_runpath=expected_runpath,
        require_relro=require_relro,
        require_bind_now=require_bind_now,
        require_noexecstack=require_noexecstack,
    )


def _revalidate_file(binding: _FileBinding) -> None:
    current = _bind_file(
        binding.path,
        maximum_bytes=binding.maximum_bytes,
        allow_final_symlink=binding.allow_final_symlink,
        require_private=binding.require_private,
        require_executable=binding.require_executable,
        expected_dependencies=binding.expected_dependencies,
        expected_runpath=binding.expected_runpath,
        require_relro=binding.require_relro,
        require_bind_now=binding.require_bind_now,
        require_noexecstack=binding.require_noexecstack,
        expected_sha256=binding.sha256,
        expected_size_bytes=binding.size_bytes,
    )
    if current != binding:
        raise ProvisionError()


def _bad_json() -> object:
    raise ProvisionError()


def _strict_json(raw: bytes) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ProvisionError()
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda _: _bad_json(),
        )
    except (UnicodeError, ValueError, OverflowError, ProvisionError):
        raise ProvisionError() from None


def _exact_receipt(
    binding: _FileBinding,
    expected: Mapping[str, object],
) -> None:
    try:
        snapshot = read_regular_bounded(
            binding.path,
            maximum_bytes=MAX_PARAKEET_CPP_CONTROL_ARTIFACT_BYTES,
            expected_bytes=binding.size_bytes,
        )
    except BoundedReadError:
        raise ProvisionError() from None
    value = _strict_json(snapshot.data)
    if (
        snapshot.path != binding.resolved_path
        or hashlib.sha256(snapshot.data).hexdigest() != binding.sha256
        or not isinstance(value, dict)
        or set(value) != set(expected)
        or not _exact_json_value(value, expected)
    ):
        raise ProvisionError()


def _exact_json_value(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        return set(actual) == set(expected) and all(
            _exact_json_value(actual[name], expected_value)
            for name, expected_value in expected.items()
        )
    if isinstance(expected, list):
        assert isinstance(actual, list)
        return len(actual) == len(expected) and all(
            _exact_json_value(actual_value, expected_value)
            for actual_value, expected_value in zip(actual, expected, strict=True)
        )
    return actual == expected


def _receipt_expectations(
    config: ParakeetCppConfig,
    artifacts: Mapping[str, _FileBinding],
) -> Mapping[str, Mapping[str, object]]:
    return {
        "source-receipt": {
            "schema_version": 1,
            "kind": "parakeet-cpp-source-receipt-v1",
            "repo_id": config.upstream_repo_id,
            "commit": config.upstream_commit,
            "tag": config.upstream_tag,
            "ggml_commit": config.ggml_commit,
            "parent_pristine_git_tree": PARAKEET_CPP_PARENT_GIT_TREE,
            "ggml_pristine_git_tree": PARAKEET_CPP_GGML_GIT_TREE,
            "ordered_patches": [
                {
                    "order": order,
                    "filename": filename,
                    "sha256": sha256,
                }
                for order, (filename, sha256) in enumerate(
                    PARAKEET_CPP_PATCH_SPECS,
                    start=1,
                )
            ],
            "patched_diff_sha256": PARAKEET_CPP_PATCHED_DIFF_SHA256,
            "license": config.upstream_license,
            "license_sha256": config.upstream_license_sha256,
        },
        "build-receipt": {
            "schema_version": 1,
            "kind": "parakeet-cpp-build-receipt-v1",
            "source_receipt_sha256": artifacts["source-receipt"].sha256,
            "c_api_version": config.c_api_version,
            "bridge_abi_version": config.bridge_abi_version,
            "requested_device": config.requested_device,
            "actual_device": config.actual_device,
            "num_threads": config.num_threads,
            "bridge_source_sha256": PARAKEET_CPP_BRIDGE_SOURCE_SHA256,
            "compiler_id": "GNU",
            "compiler_version": "13.3.0",
            "compiler_package": "Ubuntu 13.3.0-6ubuntu2~24.04.1",
            "cmake_version": "3.31.10",
            "ninja_version": "1.13.0.git.kitware.jobserver-pipe-1",
            "system_processor": "x86_64",
            "ggml_system_arch": "x86",
            "cpu_variant_flags": [
                "-msse4.2",
                "-mf16c",
                "-mfma",
                "-mbmi2",
                "-mavx",
                "-mavx2",
            ],
            "cmake": {
                "PARAKEET_SHARED": "ON",
                "BUILD_SHARED_LIBS": "OFF",
                "GGML_STATIC": "ON",
                "GGML_NATIVE": "OFF",
                "GGML_OPENMP": "OFF",
                "PARAKEET_GGML_CUDA": "OFF",
                "PARAKEET_GGML_METAL": "OFF",
                "PARAKEET_GGML_VULKAN": "OFF",
                "PARAKEET_GGML_HIP": "OFF",
            },
            "libparakeet_sha256": artifacts["libparakeet"].sha256,
            "libparakeet_size_bytes": artifacts["libparakeet"].size_bytes,
            "bridge_library_sha256": artifacts["bridge-library"].sha256,
            "bridge_library_size_bytes": artifacts[
                "bridge-library"
            ].size_bytes,
            "libparakeet_needed": list(PARAKEET_CPP_LIBPARAKEET_NEEDED),
            "bridge_needed": list(PARAKEET_CPP_BRIDGE_NEEDED),
            "bridge_runpath": PARAKEET_CPP_BRIDGE_RUNPATH,
            "bridge_relro": True,
            "bridge_bind_now": True,
            "bridge_noexecstack": True,
        },
        "model-receipt": {
            "schema_version": 1,
            "kind": "parakeet-cpp-model-receipt-v1",
            "model_repo_id": config.model_repo_id,
            "model_revision": config.model_revision,
            "source_model_repo_id": config.source_model_repo_id,
            "source_model_revision": config.source_model_revision,
            "filename": config.model_filename,
            "sha256": config.model_sha256,
            "size_bytes": config.model_size_bytes,
            "dtype": config.model_dtype,
            "license": config.model_license,
        },
    }


def _validate_receipts(state: _InputState) -> None:
    artifacts = state.artifact_by_name
    for name, expected in _receipt_expectations(state.config, artifacts).items():
        _exact_receipt(artifacts[name], expected)


def _revalidate_inputs(state: _InputState) -> None:
    _revalidate_directory(state.native_root)
    _revalidate_directory(state.model_root)
    _revalidate_file(state.python)
    _revalidate_file(state.worker)
    _revalidate_file(state.bridge_source)
    for _name, binding in state.artifacts:
        _revalidate_file(binding)
    _validate_receipts(state)
    _revalidate_directory(state.native_root)
    _revalidate_directory(state.model_root)


def _collect_inputs(
    *,
    python: Path | str,
    native_root: Path | str,
    model_root: Path | str,
) -> _InputState:
    try:
        config = ParakeetCppConfig()
    except (TypeError, ValueError, ManifestError):
        raise ProvisionError() from None
    native = _bind_private_directory(
        native_root,
        expected_names=_NATIVE_FILENAMES,
    )
    model = _bind_private_directory(
        model_root,
        expected_names=_MODEL_FILENAMES,
    )
    if _overlaps(native.path, model.path):
        raise ProvisionError()

    paths = {
        "source-receipt": native.path / _ARTIFACT_FILENAMES["source-receipt"],
        "build-receipt": native.path / _ARTIFACT_FILENAMES["build-receipt"],
        "model-receipt": model.path / _ARTIFACT_FILENAMES["model-receipt"],
        "libparakeet": native.path / _ARTIFACT_FILENAMES["libparakeet"],
        "bridge-library": native.path / _ARTIFACT_FILENAMES["bridge-library"],
        "model-gguf": model.path / _ARTIFACT_FILENAMES["model-gguf"],
    }
    artifacts: list[tuple[str, _FileBinding]] = []
    for name in PARAKEET_CPP_ARTIFACT_NAMES:
        expected_sha256: str | None = None
        expected_size_bytes: int | None = None
        if name == "model-gguf":
            expected_sha256, expected_size_bytes = PARAKEET_CPP_MODEL_RECEIPT
        elif name == "libparakeet":
            expected_sha256, expected_size_bytes = (
                PARAKEET_CPP_LIBPARAKEET_RECEIPT
            )
        elif name == "bridge-library":
            expected_sha256, expected_size_bytes = PARAKEET_CPP_BRIDGE_RECEIPT
        artifacts.append(
            (
                name,
                _bind_file(
                    paths[name],
                    maximum_bytes=artifact_maximum_bytes(
                        PARAKEET_CPP_ADAPTER,
                        name,
                    ),
                    require_private=True,
                    expected_dependencies={
                        "libparakeet": PARAKEET_CPP_LIBPARAKEET_NEEDED,
                        "bridge-library": PARAKEET_CPP_BRIDGE_NEEDED,
                    }.get(name),
                    expected_runpath=(
                        PARAKEET_CPP_BRIDGE_RUNPATH
                        if name == "bridge-library"
                        else None
                    ),
                    require_relro=name == "bridge-library",
                    require_bind_now=name == "bridge-library",
                    require_noexecstack=name == "bridge-library",
                    expected_sha256=expected_sha256,
                    expected_size_bytes=expected_size_bytes,
                ),
            )
        )
    state = _InputState(
        config=config,
        native_root=native,
        model_root=model,
        python=_bind_file(
            python,
            maximum_bytes=MAX_PYTHON_BYTES,
            allow_final_symlink=True,
            require_executable=True,
        ),
        worker=_bind_file(
            _FIXED_WORKER,
            maximum_bytes=MAX_WORKER_BYTES,
        ),
        bridge_source=_bind_file(
            _FIXED_BRIDGE_SOURCE,
            maximum_bytes=MAX_PARAKEET_CPP_BRIDGE_SOURCE_BYTES,
            expected_sha256=PARAKEET_CPP_BRIDGE_SOURCE_SHA256,
        ),
        artifacts=tuple(artifacts),
    )
    _validate_receipts(state)
    _revalidate_inputs(state)
    return state


def _new_private_output(
    path: Path | str,
    *,
    forbidden_roots: tuple[Path, ...],
) -> Path:
    candidate = _absolute(path)
    if (
        not candidate.name
        or candidate.name in {".", ".."}
        or "\x00" in candidate.name
        or any(_overlaps(candidate, root) for root in forbidden_roots)
    ):
        raise ProvisionError()
    try:
        with opened_directory_nofollow(candidate.parent) as (
            _parent,
            parent_descriptor,
        ):
            os.mkdir(candidate.name, mode=0o700, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
        with opened_directory_nofollow(
            candidate,
            require_private=True,
        ) as (stable, descriptor):
            metadata = os.fstat(descriptor)
            if (
                stable != candidate
                or stat.S_IMODE(metadata.st_mode) != 0o700
                or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
                or os.listdir(descriptor)
            ):
                raise ProvisionError()
            return stable
    except (OSError, RuntimeError, ValueError, BoundedReadError, ProvisionError):
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


def _same_inode(metadata: os.stat_result, identity: tuple[int, ...]) -> bool:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
    ) == (identity[0], identity[1], identity[2])


def _unlink_bound_entry(
    descriptor: int,
    name: str,
    identity: tuple[int, ...],
) -> bool:
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


def _read_descriptor(descriptor: int, expected: bytes) -> os.stat_result:
    try:
        before = os.fstat(descriptor)
        data = os.pread(descriptor, len(expected) + 1, 0)
        after = os.fstat(descriptor)
    except (OSError, OverflowError, ValueError):
        raise ProvisionError() from None
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o600
        or (hasattr(os, "geteuid") and before.st_uid != os.geteuid())
        or before.st_size != len(expected)
        or _file_identity(after) != _file_identity(before)
        or data != expected
    ):
        raise ProvisionError()
    return after


def _verify_output_entry(
    directory_descriptor: int,
    name: str,
    expected: bytes,
    identity: tuple[int, ...],
    *,
    expected_links: int,
) -> None:
    descriptor = -1
    try:
        metadata = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
        opened = _read_descriptor(descriptor, expected)
    except (OSError, OverflowError, ValueError, ProvisionError):
        raise ProvisionError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
    if (
        metadata.st_nlink != expected_links
        or opened.st_nlink != expected_links
        or _stable_file_identity(metadata) != identity
        or _stable_file_identity(opened) != identity
    ):
        raise ProvisionError()


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    try:
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise ProvisionError()
            offset += written
        os.fsync(descriptor)
    except (OSError, OverflowError, ValueError, ProvisionError):
        raise ProvisionError() from None


def _post_validation_fault_point() -> None:
    """Test seam after canonical linking and before candidate acceptance."""


def _validate_loaded_manifest(
    manifest: WorkerManifest,
    state: _InputState,
    *,
    path: Path,
    expected_digest: str,
) -> None:
    artifacts = state.artifact_by_name
    if (
        manifest.path != path
        or manifest.digest != expected_digest
        or manifest.schema_version != 8
        or manifest.model_id != _MODEL_ID
        or manifest.adapter != PARAKEET_CPP_ADAPTER
        or manifest.adapter_config != state.config
        or manifest.python.path != state.python.path
        or manifest.python.sha256 != state.python.sha256
        or manifest.python.size_bytes != state.python.size_bytes
        or manifest.worker.path != state.worker.resolved_path
        or manifest.worker.sha256 != state.worker.sha256
        or manifest.worker.size_bytes != state.worker.size_bytes
        or state.bridge_source.sha256 != PARAKEET_CPP_BRIDGE_SOURCE_SHA256
        or manifest.limits.startup_timeout_sec != _STARTUP_TIMEOUT_SEC
        or manifest.limits.case_timeout_sec != _CASE_TIMEOUT_SEC
        or tuple(manifest.artifact_by_name) != PARAKEET_CPP_ARTIFACT_NAMES
        or any(
            manifest.artifact_by_name[name].path != binding.resolved_path
            or manifest.artifact_by_name[name].sha256 != binding.sha256
            or manifest.artifact_by_name[name].size_bytes != binding.size_bytes
            for name, binding in artifacts.items()
        )
    ):
        raise ProvisionError()


def _publish_manifest(
    destination: Path,
    payload: bytes,
    state: _InputState,
) -> WorkerManifest:
    pending_path = destination / _PENDING_MANIFEST_FILENAME
    canonical_path = destination / _MANIFEST_FILENAME
    expected_digest = hashlib.sha256(payload).hexdigest()
    pending_descriptor = -1
    pending_identity: tuple[int, ...] | None = None
    published = False
    try:
        with opened_directory_nofollow(
            destination,
            require_private=True,
        ) as (stable, directory_descriptor):
            if stable != destination or os.listdir(directory_descriptor):
                raise ProvisionError()
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            pending_descriptor = os.open(
                _PENDING_MANIFEST_FILENAME,
                flags,
                0o600,
                dir_fd=directory_descriptor,
            )
            os.fchmod(pending_descriptor, 0o600)
            _write_all(pending_descriptor, payload)
            pending_metadata = _read_descriptor(pending_descriptor, payload)
            if pending_metadata.st_nlink != 1:
                raise ProvisionError()
            pending_identity = _stable_file_identity(pending_metadata)
            _verify_output_entry(
                directory_descriptor,
                _PENDING_MANIFEST_FILENAME,
                payload,
                pending_identity,
                expected_links=1,
            )
            if set(os.listdir(directory_descriptor)) != {
                _PENDING_MANIFEST_FILENAME
            }:
                raise ProvisionError()

            manifest = load_worker_manifest(pending_path)
            _validate_loaded_manifest(
                manifest,
                state,
                path=pending_path,
                expected_digest=expected_digest,
            )
            _revalidate_inputs(state)
            _verify_output_entry(
                directory_descriptor,
                _PENDING_MANIFEST_FILENAME,
                payload,
                pending_identity,
                expected_links=1,
            )
            try:
                os.link(
                    f"/proc/self/fd/{pending_descriptor}",
                    _MANIFEST_FILENAME,
                    dst_dir_fd=directory_descriptor,
                    follow_symlinks=True,
                )
            except OSError:
                raise ProvisionError() from None
            _post_validation_fault_point()
            _revalidate_inputs(state)
            _verify_output_entry(
                directory_descriptor,
                _PENDING_MANIFEST_FILENAME,
                payload,
                pending_identity,
                expected_links=2,
            )
            _verify_output_entry(
                directory_descriptor,
                _MANIFEST_FILENAME,
                payload,
                pending_identity,
                expected_links=2,
            )
            if set(os.listdir(directory_descriptor)) != {
                _PENDING_MANIFEST_FILENAME,
                _MANIFEST_FILENAME,
            }:
                raise ProvisionError()
            os.fsync(directory_descriptor)
            if not _unlink_bound_entry(
                directory_descriptor,
                _PENDING_MANIFEST_FILENAME,
                pending_identity,
            ):
                raise ProvisionError()
            os.fsync(directory_descriptor)
            _verify_output_entry(
                directory_descriptor,
                _MANIFEST_FILENAME,
                payload,
                pending_identity,
                expected_links=1,
            )
            if set(os.listdir(directory_descriptor)) != {_MANIFEST_FILENAME}:
                raise ProvisionError()
            published = True
            return replace(manifest, path=canonical_path)
    except (
        OSError,
        RuntimeError,
        ValueError,
        BoundedReadError,
        ManifestError,
        ProvisionError,
    ):
        raise ProvisionError() from None
    finally:
        if pending_descriptor >= 0:
            try:
                os.close(pending_descriptor)
            except OSError:
                pass
        if not published and pending_identity is not None:
            try:
                with opened_directory_nofollow(
                    destination,
                    require_private=True,
                ) as (_stable, directory_descriptor):
                    _unlink_bound_entry(
                        directory_descriptor,
                        _MANIFEST_FILENAME,
                        pending_identity,
                    )
                    _unlink_bound_entry(
                        directory_descriptor,
                        _PENDING_MANIFEST_FILENAME,
                        pending_identity,
                    )
                    os.fsync(directory_descriptor)
            except (OSError, BoundedReadError, ProvisionError):
                pass


def provision_candidate(
    *,
    python: Path | str,
    native_root: Path | str,
    model_root: Path | str,
    output_dir: Path | str,
) -> WorkerManifest:
    """Verify supplied offline inputs and publish one schema-v8 manifest."""

    state = _collect_inputs(
        python=python,
        native_root=native_root,
        model_root=model_root,
    )
    destination = _new_private_output(
        output_dir,
        forbidden_roots=(state.native_root.path, state.model_root.path),
    )
    artifact_bindings = state.artifact_by_name
    manifest_payload = {
        "schema_version": 8,
        "model_id": _MODEL_ID,
        "adapter": PARAKEET_CPP_ADAPTER,
        "adapter_config": state.config.as_dict(),
        "python": state.python.as_payload(),
        "worker": state.worker.as_payload(),
        "artifacts": [
            {"name": name, **artifact_bindings[name].as_payload()}
            for name in PARAKEET_CPP_ARTIFACT_NAMES
        ],
        "limits": {
            "startup_timeout_sec": _STARTUP_TIMEOUT_SEC,
            "case_timeout_sec": _CASE_TIMEOUT_SEC,
        },
    }
    return _publish_manifest(
        destination,
        _manifest_bytes(manifest_payload),
        state,
    )


def _safe_result(manifest: WorkerManifest) -> dict[str, object]:
    artifacts = manifest.artifact_by_name
    return {
        "ok": True,
        "adapter": manifest.adapter,
        "model_id": manifest.model_id,
        "manifest_sha256": manifest.digest,
        "source_receipt_sha256": artifacts["source-receipt"].sha256,
        "build_receipt_sha256": artifacts["build-receipt"].sha256,
        "model_receipt_sha256": artifacts["model-receipt"].sha256,
        "bridge_source_sha256": PARAKEET_CPP_BRIDGE_SOURCE_SHA256,
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
            "Bind exact private parakeet.cpp CPU libraries, receipts, and "
            "GGUF model into a schema-v8 worker manifest without building, "
            "downloading, importing, executing, or loading the candidate."
        )
    )
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = provision_candidate(
            python=args.python,
            native_root=args.native_root,
            model_root=args.model_root,
            output_dir=args.output_dir,
        )
        result: Mapping[str, object] = _safe_result(manifest)
        code = 0
    except Exception:  # noqa: BLE001 - local paths remain undisclosed
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
