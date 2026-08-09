"""Bind an existing Kyutai STT runtime and model without executing either.

The caller supplies the exact private Python 3.12.3 runtime produced by the
offline runtime builder, its already verified wheelhouse, and the four staged
model artifacts.  This command only verifies and receipts those inputs.  It
never resolves, downloads, installs, imports, or loads candidate code or model
weights.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Mapping, Sequence

from tools.streaming_stt.bounded_io import (
    BoundedReadError,
    hash_regular_bounded,
    opened_directory_nofollow,
    read_regular_bounded,
)
from tools.streaming_stt.manifest import (
    KYUTAI_ADAPTER,
    KYUTAI_ARTIFACT_NAMES,
    KYUTAI_MODEL_RECEIPTS,
    KYUTAI_RUNTIME_CONTENT_SHA256,
    KYUTAI_RUNTIME_FILE_COUNT,
    KYUTAI_RUNTIME_MAXIMUM_FILE_BYTES,
    KYUTAI_RUNTIME_TOTAL_SIZE_BYTES,
    KYUTAI_RUNTIME_WHEEL_LOCK_SHA256,
    KYUTAI_RUNTIME_WHEEL_LOCK_SIZE_BYTES,
    MAX_PYTHON_BYTES,
    MAX_WORKER_BYTES,
    KyutaiConfig,
    ManifestError,
    WorkerManifest,
    artifact_maximum_bytes,
    load_worker_manifest,
)
from tools.streaming_stt.runtime_receipt import (
    KYUTAI_RUNTIME_TREE_LIMITS,
    RuntimeTreeReceiptError,
    generate_runtime_tree_receipt,
    load_runtime_tree_receipt,
    verify_isolated_venv_marker,
    verify_runtime_tree_receipt,
    verify_venv_runtime_location,
)
from tools.streaming_stt.wheel_lock import (
    KYUTAI_WHEEL_ARCHIVE_POLICY,
    WheelLockError,
    load_wheel_lock,
    verify_locked_wheel_install,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_PRODUCTION_VENV = _REPO_ROOT / ".venv"
_FIXED_WORKER = _REPO_ROOT / "tools" / "streaming_stt" / "worker.py"
_REPOSITORY_WHEEL_LOCK = (
    _REPO_ROOT / "tools" / "streaming_stt" / "kyutai-stt-runtime-wheels.lock.json"
)
_PYTHON_DIRECTORY_RE = re.compile(r"python[0-9]+\.[0-9]+\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_EXPECTED_INTERPRETER = Path("/usr/bin/python3.12")
_TRUSTED_INTERPRETER = Path(sys.executable).resolve(strict=True)
_MODEL_FILES = {
    "model-config": "config.json",
    "model-weights": "model.safetensors",
    "model-mimi": "mimi-pytorch-e351c8d8@125.safetensors",
    "model-tokenizer": "tokenizer_en_fr_audio_8000.model",
}
_MODEL_RECEIPTS = dict(KYUTAI_MODEL_RECEIPTS)
_OUTPUT_FILENAMES = {
    "kyutai-stt-runtime-wheels.lock.json",
    "runtime-receipt.json",
    "worker-manifest.json",
}
_SAFE_ERROR = {
    "ok": False,
    "error": "kyutai_candidate_prerequisites_unavailable",
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


def _require_outside_production(path: Path) -> None:
    production = _absolute(_PRODUCTION_VENV)
    if _overlaps(path, production):
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
    if python_directories != {"python3.12"} or set(_directory_names(lib)) != {
        "python3.12"
    }:
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


def _exact_marker_values(data: bytes) -> None:
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeError:
        raise ProvisionError() from None
    expected = {
        "home": str(_EXPECTED_INTERPRETER.parent),
        "include-system-site-packages": "false",
        "version": "3.12.3",
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


def _exact_marker_binding(path: Path) -> _MarkerBinding:
    try:
        before = path.lstat()
        verify_isolated_venv_marker(path)
        snapshot = read_regular_bounded(
            path,
            maximum_bytes=artifact_maximum_bytes(KYUTAI_ADAPTER, "venv-marker"),
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
    marker_identity = _file_identity(before)
    if (
        snapshot.path != path
        or marker_identity != _file_identity(after)
        or not stat.S_ISREG(after.st_mode)
        or after.st_nlink != 1
        or stat.S_IMODE(after.st_mode) != 0o600
        or (hasattr(os, "geteuid") and after.st_uid != os.geteuid())
    ):
        raise ProvisionError()
    _exact_marker_values(snapshot.data)
    return _MarkerBinding(
        path=path,
        data=snapshot.data,
        identity=marker_identity,
        sha256=hashlib.sha256(snapshot.data).hexdigest(),
        size_bytes=len(snapshot.data),
    )


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
    except (OSError, RuntimeError, ValueError, ProvisionError):
        raise ProvisionError() from None


def _exact_model_bindings(model_root: Path) -> dict[str, dict[str, object]]:
    model_names = {name for name in KYUTAI_ARTIFACT_NAMES if name.startswith("model-")}
    if (
        set(_MODEL_FILES) != model_names
        or set(_MODEL_RECEIPTS) != set(_MODEL_FILES)
        or set(_directory_names(model_root)) != set(_MODEL_FILES.values())
    ):
        raise ProvisionError()

    bindings: dict[str, dict[str, object]] = {}
    for name, basename in _MODEL_FILES.items():
        path = model_root / basename
        expected_sha256, expected_size = _MODEL_RECEIPTS[name]
        if (
            not isinstance(expected_sha256, str)
            or _SHA256_RE.fullmatch(expected_sha256) is None
            or isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size <= 0
            or expected_size > artifact_maximum_bytes(KYUTAI_ADAPTER, name)
        ):
            raise ProvisionError()
        try:
            before = path.lstat()
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or stat.S_IMODE(before.st_mode) != 0o600
                or (hasattr(os, "geteuid") and before.st_uid != os.geteuid())
            ):
                raise ProvisionError()
            digest = hash_regular_bounded(
                path,
                maximum_bytes=artifact_maximum_bytes(KYUTAI_ADAPTER, name),
                expected_bytes=expected_size,
            )
            after = path.lstat()
        except (OSError, RuntimeError, ValueError, BoundedReadError, ProvisionError):
            raise ProvisionError() from None
        if (
            _file_identity(before) != _file_identity(after)
            or digest.sha256 != expected_sha256
            or digest.path.parent != model_root
        ):
            raise ProvisionError()
        bindings[name] = {
            "path": str(path),
            "sha256": digest.sha256,
            "size_bytes": digest.size_bytes,
        }

    if set(_directory_names(model_root)) != set(_MODEL_FILES.values()):
        raise ProvisionError()
    return bindings


def _repository_wheel_lock_bytes() -> bytes:
    try:
        snapshot = read_regular_bounded(
            _REPOSITORY_WHEEL_LOCK,
            maximum_bytes=artifact_maximum_bytes(
                KYUTAI_ADAPTER,
                "runtime-wheel-lock",
            ),
            expected_bytes=KYUTAI_RUNTIME_WHEEL_LOCK_SIZE_BYTES,
        )
        metadata = snapshot.path.lstat()
    except (OSError, RuntimeError, ValueError, BoundedReadError, ManifestError):
        raise ProvisionError() from None
    if (
        snapshot.path != _REPOSITORY_WHEEL_LOCK
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or hashlib.sha256(snapshot.data).hexdigest() != KYUTAI_RUNTIME_WHEEL_LOCK_SHA256
    ):
        raise ProvisionError()
    return snapshot.data


def _write_new_private(path: Path, payload: bytes) -> None:
    descriptor = -1
    try:
        with opened_directory_nofollow(path.parent) as (_parent, parent_descriptor):
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path.name, flags, 0o600, dir_fd=parent_descriptor)
            os.fchmod(descriptor, 0o600)
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
                    or (hasattr(os, "geteuid") and written.st_uid != os.geteuid())
                ):
                    raise ProvisionError()
                written_identity = _file_identity(written)
            current = os.stat(
                path.name, dir_fd=parent_descriptor, follow_symlinks=False
            )
            if _file_identity(current) != written_identity:
                raise ProvisionError()
            os.fsync(parent_descriptor)
    except (OSError, ValueError, OverflowError, BoundedReadError, ProvisionError):
        raise ProvisionError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


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
    """Verify supplied offline inputs and create one schema-v9 manifest."""

    try:
        config = KyutaiConfig(
            python_version="3.12.3",
            moshi_version="0.2.11",
            torch_version="2.7.1+cu126",
            cuda_version="12.6",
            julius_version="0.2.8",
            wheel_lock_sha256=KYUTAI_RUNTIME_WHEEL_LOCK_SHA256,
            runtime_content_sha256=KYUTAI_RUNTIME_CONTENT_SHA256,
            runtime_file_count=KYUTAI_RUNTIME_FILE_COUNT,
            runtime_total_size_bytes=KYUTAI_RUNTIME_TOTAL_SIZE_BYTES,
            runtime_maximum_file_bytes=KYUTAI_RUNTIME_MAXIMUM_FILE_BYTES,
        )
    except (TypeError, ValueError, ManifestError):
        raise ProvisionError() from None

    venv = _canonical_directory(venv_root, require_private=True)
    selected_model_root = _canonical_directory(model_root, require_private=True)
    selected_wheelhouse = _canonical_directory(wheelhouse, require_private=True)
    roots = (venv, selected_model_root, selected_wheelhouse)
    if any(
        left != right and _overlaps(left, right)
        for index, left in enumerate(roots)
        for right in roots[index + 1 :]
    ):
        raise ProvisionError()

    python, marker, site_packages = _runtime_layout(venv)
    python_binding = _runtime_python_binding(python)
    marker_binding = _exact_marker_binding(marker)
    model_bindings = _exact_model_bindings(selected_model_root)
    repository_lock = _repository_wheel_lock_bytes()
    worker_binding = _bound_payload(_FIXED_WORKER, maximum_bytes=MAX_WORKER_BYTES)

    destination = _new_private_output(output_dir, forbidden_roots=roots)
    runtime_wheel_lock = destination / "kyutai-stt-runtime-wheels.lock.json"
    _write_new_private(runtime_wheel_lock, repository_lock)
    try:
        lock = load_wheel_lock(
            runtime_wheel_lock,
            expected_digest=KYUTAI_RUNTIME_WHEEL_LOCK_SHA256,
        )
        receipt = generate_runtime_tree_receipt(
            site_packages,
            destination / "runtime-receipt.json",
            limits=KYUTAI_RUNTIME_TREE_LIMITS,
        )
        verify_venv_runtime_location(receipt, python)
        expectation = verify_locked_wheel_install(
            lock,
            selected_wheelhouse,
            receipt,
            archive_policy=KYUTAI_WHEEL_ARCHIVE_POLICY,
        )
        verify_runtime_tree_receipt(receipt, limits=KYUTAI_RUNTIME_TREE_LIMITS)
        receipt_metadata = receipt.receipt_path.lstat()
    except (OSError, RuntimeTreeReceiptError, WheelLockError):
        raise ProvisionError() from None

    if (
        lock.digest != KYUTAI_RUNTIME_WHEEL_LOCK_SHA256
        or not stat.S_ISREG(receipt_metadata.st_mode)
        or receipt_metadata.st_nlink != 1
        or stat.S_IMODE(receipt_metadata.st_mode) != 0o600
        or receipt_metadata.st_size <= 0
        or receipt_metadata.st_size > KYUTAI_RUNTIME_TREE_LIMITS.maximum_receipt_bytes
        or (hasattr(os, "geteuid") and receipt_metadata.st_uid != os.geteuid())
        or receipt.content_digest != KYUTAI_RUNTIME_CONTENT_SHA256
        or receipt.file_count != KYUTAI_RUNTIME_FILE_COUNT
        or receipt.total_size_bytes != KYUTAI_RUNTIME_TOTAL_SIZE_BYTES
        or max(item.size_bytes for item in receipt.files)
        != KYUTAI_RUNTIME_MAXIMUM_FILE_BYTES
        or expectation.content_digest != KYUTAI_RUNTIME_CONTENT_SHA256
        or expectation.file_count != KYUTAI_RUNTIME_FILE_COUNT
        or expectation.total_size_bytes != KYUTAI_RUNTIME_TOTAL_SIZE_BYTES
        or expectation.maximum_file_bytes != KYUTAI_RUNTIME_MAXIMUM_FILE_BYTES
    ):
        raise ProvisionError()

    artifact_paths = {
        "runtime-receipt": receipt.receipt_path,
        "runtime-wheel-lock": runtime_wheel_lock,
    }
    artifact_bindings = {
        name: _bound_payload(
            artifact_paths[name],
            maximum_bytes=artifact_maximum_bytes(KYUTAI_ADAPTER, name),
        )
        for name in ("runtime-receipt", "runtime-wheel-lock")
    }
    artifact_bindings["venv-marker"] = marker_binding.as_payload()
    artifact_bindings.update(model_bindings)
    artifacts = [
        {"name": name, **artifact_bindings[name]} for name in KYUTAI_ARTIFACT_NAMES
    ]
    manifest_payload = {
        "schema_version": 9,
        "model_id": "kyutai-stt-1b-en-fr-candle-095e-cuda-bf16",
        "adapter": KYUTAI_ADAPTER,
        "adapter_config": config.as_dict(),
        "python": python_binding,
        "worker": worker_binding,
        "artifacts": artifacts,
        "limits": {
            "startup_timeout_sec": 300.0,
            "case_timeout_sec": 900.0,
        },
    }
    manifest_path = destination / "worker-manifest.json"
    _write_new_private(manifest_path, _manifest_bytes(manifest_payload))
    try:
        manifest = load_worker_manifest(manifest_path)
        current_receipt = load_runtime_tree_receipt(
            receipt.receipt_path,
            expected_digest=receipt.digest,
            limits=KYUTAI_RUNTIME_TREE_LIMITS,
        )
        verify_runtime_tree_receipt(
            current_receipt,
            limits=KYUTAI_RUNTIME_TREE_LIMITS,
        )
    except (ManifestError, RuntimeTreeReceiptError):
        raise ProvisionError() from None
    if (
        manifest.schema_version != 9
        or manifest.adapter != KYUTAI_ADAPTER
        or manifest.adapter_config != config
        or manifest.python.path != python
        or manifest.worker.sha256 != worker_binding["sha256"]
        or manifest.artifact_by_name["runtime-receipt"].sha256 != receipt.digest
        or manifest.artifact_by_name["runtime-wheel-lock"].sha256
        != KYUTAI_RUNTIME_WHEEL_LOCK_SHA256
        or current_receipt != receipt
        or set(_directory_names(destination)) != _OUTPUT_FILENAMES
    ):
        raise ProvisionError()
    return manifest


def _safe_result(manifest: WorkerManifest) -> dict[str, object]:
    return {
        "ok": True,
        "adapter": manifest.adapter,
        "model_id": manifest.model_id,
        "manifest_sha256": manifest.digest,
        "runtime_receipt_sha256": manifest.artifact_by_name["runtime-receipt"].sha256,
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
            "Bind an existing private Python 3.12.3 Kyutai runtime, exact "
            "four-file Candle model, and verified offline wheelhouse into a "
            "schema-v9 worker manifest. This command does not install, "
            "download, import, execute, or load the candidate runtime or model."
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
    except Exception:  # noqa: BLE001 - private paths and failures remain private
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
