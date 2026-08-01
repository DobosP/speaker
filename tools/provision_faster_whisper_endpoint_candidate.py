"""Bind a local Faster-Whisper runtime and model without executing either.

The command hashes an existing virtual-environment package tree and one
already-downloaded CTranslate2 model tree into private receipts. It never
installs, downloads, imports, or loads the candidate model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Mapping, Sequence

from tools.streaming_stt.bounded_io import (
    BoundedReadError,
    hash_regular_bounded,
    opened_directory_nofollow,
    read_regular_bounded,
)
from tools.streaming_stt.manifest import (
    FASTER_WHISPER_ARTIFACT_NAMES,
    FASTER_WHISPER_ENDPOINT_ADAPTER,
    FASTER_WHISPER_REQUIRED_MODEL_FILES,
    FASTER_WHISPER_VOCABULARY_FILES,
    MAX_PYTHON_BYTES,
    MAX_WORKER_BYTES,
    ManifestError,
    WorkerManifest,
    artifact_maximum_bytes,
    load_worker_manifest,
)
from tools.streaming_stt.runtime_receipt import (
    FASTER_WHISPER_MODEL_TREE_LIMITS,
    FASTER_WHISPER_RUNTIME_TREE_LIMITS,
    RuntimeTreeReceipt,
    RuntimeTreeReceiptError,
    generate_runtime_tree_receipt,
    verify_isolated_venv_marker,
    verify_venv_runtime_location,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXED_WORKER = _REPO_ROOT / "tools" / "streaming_stt" / "worker.py"
_PYTHON_DIRECTORY_RE = re.compile(r"python[0-9]+\.[0-9]+\Z")
_PYTHON_DIRECTORY = "python3.12"
_PYTHON_VERSION = "3.12.3"
_SAFE_MODEL_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}\Z")
_EXPECTED_RUNTIME_PATHS = frozenset(
    {
        "faster_whisper/__init__.py",
        "faster_whisper/transcribe.py",
        "faster_whisper-1.2.1.dist-info/METADATA",
        "ctranslate2/__init__.py",
        "ctranslate2-4.8.1.dist-info/METADATA",
        "numpy/__init__.py",
        "numpy-2.4.6.dist-info/METADATA",
        "nvidia_cublas_cu12-12.9.2.10.dist-info/METADATA",
        "nvidia_cudnn_cu12-9.24.0.43.dist-info/METADATA",
        "nvidia_cuda_nvrtc_cu12-12.9.86.dist-info/METADATA",
        "nvidia/cublas/lib/libcublasLt.so.12",
        "nvidia/cublas/lib/libcublas.so.12",
        "nvidia/cuda_nvrtc/lib/libnvrtc.so.12",
        "nvidia/cudnn/lib/libcudnn.so.9",
    }
)
_SAFE_ERROR = {
    "ok": False,
    "error": "faster_whisper_endpoint_prerequisites_unavailable",
}


class ProvisionError(RuntimeError):
    """A detail-free unsafe, incomplete, or changed provision input."""


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


def _canonical_directory(path: Path | str, *, require_private: bool) -> Path:
    candidate = _absolute(path)
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


def _verify_python3123_marker(path: Path) -> None:
    try:
        verify_isolated_venv_marker(path)
        snapshot = read_regular_bounded(
            path,
            maximum_bytes=artifact_maximum_bytes(
                FASTER_WHISPER_ENDPOINT_ADAPTER,
                "venv-marker",
            ),
        )
        text = snapshot.data.decode("utf-8", errors="strict")
    except (
        UnicodeError,
        BoundedReadError,
        ManifestError,
        RuntimeTreeReceiptError,
    ):
        raise ProvisionError() from None
    versions: list[str] = []
    for raw_line in text.splitlines():
        if "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        if key.strip().casefold() == "version":
            versions.append(value.strip())
    if versions != [_PYTHON_VERSION]:
        raise ProvisionError()


def _runtime_layout(venv_root: Path) -> tuple[Path, Path, Path]:
    binary_root = _canonical_directory(venv_root / "bin", require_private=False)
    lib = _canonical_directory(venv_root / "lib", require_private=False)
    python_directories = {
        name for name in _directory_names(lib) if _PYTHON_DIRECTORY_RE.fullmatch(name)
    }
    if python_directories != {_PYTHON_DIRECTORY}:
        raise ProvisionError()
    python_root = _canonical_directory(
        lib / _PYTHON_DIRECTORY,
        require_private=False,
    )
    site_packages = _canonical_directory(
        python_root / "site-packages",
        require_private=True,
    )
    python = binary_root / "python"
    marker = venv_root / "pyvenv.cfg"
    try:
        lexical = python.lstat()
        resolved_python = python.resolve(strict=True)
    except (OSError, RuntimeError):
        raise ProvisionError() from None
    if (
        not stat.S_ISLNK(lexical.st_mode)
        or lexical.st_nlink != 1
        or resolved_python.name != _PYTHON_DIRECTORY
    ):
        raise ProvisionError()
    _verify_python3123_marker(marker)
    return python, marker, site_packages


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


def _inode_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        stat.S_IMODE(metadata.st_mode),
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
    )


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        *_inode_identity(metadata),
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _private_regular(metadata: os.stat_result, *, size_bytes: int) -> bool:
    return (
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_nlink == 1
        and stat.S_IMODE(metadata.st_mode) == 0o600
        and metadata.st_size == size_bytes
        and (not hasattr(os, "geteuid") or metadata.st_uid == os.geteuid())
    )


def _write_new_private(path: Path, payload: bytes) -> None:
    if not isinstance(payload, bytes) or not payload:
        raise ProvisionError()
    descriptor = -1
    try:
        if (
            not path.is_absolute()
            or not path.name
            or path.name in {".", ".."}
            or "/" in path.name
            or "\x00" in path.name
        ):
            raise ProvisionError()
        with opened_directory_nofollow(
            path.parent,
            require_private=True,
        ) as (stable_parent, parent_descriptor):
            if stable_parent != path.parent:
                raise ProvisionError()
            parent_identity = _inode_identity(os.fstat(parent_descriptor))
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
            created = os.fstat(descriptor)
            if not _private_regular(created, size_bytes=0):
                raise ProvisionError()
            created_identity = _inode_identity(created)
            with os.fdopen(descriptor, "wb", buffering=0) as handle:
                descriptor = -1
                if handle.write(payload) != len(payload):
                    raise ProvisionError()
                os.fsync(handle.fileno())
                written = os.fstat(handle.fileno())
                named = os.stat(
                    path.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                if (
                    not _private_regular(written, size_bytes=len(payload))
                    or not _private_regular(named, size_bytes=len(payload))
                    or _inode_identity(written) != created_identity
                    or _file_identity(named) != _file_identity(written)
                ):
                    raise ProvisionError()
                os.fsync(parent_descriptor)
                final_descriptor = os.fstat(handle.fileno())
                final_named = os.stat(
                    path.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                if (
                    _file_identity(final_descriptor) != _file_identity(written)
                    or _file_identity(final_named) != _file_identity(written)
                    or _inode_identity(os.fstat(parent_descriptor)) != parent_identity
                ):
                    raise ProvisionError()
    except (
        OSError,
        RuntimeError,
        ValueError,
        OverflowError,
        BoundedReadError,
        ProvisionError,
    ):
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


def _tree_summary(receipt: RuntimeTreeReceipt, *, prefix: str) -> dict[str, object]:
    maximum_file = max((item.size_bytes for item in receipt.files), default=0)
    if maximum_file <= 0:
        raise ProvisionError()
    return {
        f"{prefix}_content_sha256": receipt.content_digest,
        f"{prefix}_file_count": receipt.file_count,
        f"{prefix}_total_size_bytes": receipt.total_size_bytes,
        f"{prefix}_maximum_file_bytes": maximum_file,
    }


def _verify_runtime_closure(receipt: RuntimeTreeReceipt) -> None:
    paths = set(receipt.file_by_path)
    extension_modules = [
        path
        for path in paths
        if path.startswith("ctranslate2/_ext") and path.endswith(".so")
    ]
    if not _EXPECTED_RUNTIME_PATHS.issubset(paths) or len(extension_modules) != 1:
        raise ProvisionError()


def _verify_model_closure(receipt: RuntimeTreeReceipt) -> None:
    paths = set(receipt.file_by_path)
    if not FASTER_WHISPER_REQUIRED_MODEL_FILES.issubset(
        paths
    ) or not FASTER_WHISPER_VOCABULARY_FILES.intersection(paths):
        raise ProvisionError()


def provision_candidate(
    *,
    venv_root: Path | str,
    model_root: Path | str,
    output_dir: Path | str,
    model_id: str = "faster-whisper-endpoint-local",
    language: str = "en",
) -> WorkerManifest:
    """Create a schema-6 receipt without importing or loading candidate code."""

    if _SAFE_MODEL_ID_RE.fullmatch(model_id) is None or language not in {"en", "ro"}:
        raise ProvisionError()
    venv = _canonical_directory(venv_root, require_private=False)
    python, marker, site_packages = _runtime_layout(venv)
    selected_model_root = _canonical_directory(model_root, require_private=True)
    if _overlaps(venv, selected_model_root) or _overlaps(
        site_packages,
        selected_model_root,
    ):
        raise ProvisionError()

    # Bind all small, fixed inputs before creating the output directory. Invalid
    # Python layout/marker or an unsafe fixed worker therefore has zero output
    # mutation, like invalid output/root overlap below.
    python_binding = _bound_payload(
        python,
        maximum_bytes=MAX_PYTHON_BYTES,
        allow_final_symlink=True,
    )
    marker_binding = _bound_payload(
        marker,
        maximum_bytes=artifact_maximum_bytes(
            FASTER_WHISPER_ENDPOINT_ADAPTER,
            "venv-marker",
        ),
    )
    worker_binding = _bound_payload(
        _FIXED_WORKER,
        maximum_bytes=MAX_WORKER_BYTES,
    )
    destination = _new_private_output(
        output_dir,
        forbidden_roots=(venv, site_packages, selected_model_root),
    )
    try:
        runtime_receipt = generate_runtime_tree_receipt(
            site_packages,
            destination / "runtime-receipt.json",
            limits=FASTER_WHISPER_RUNTIME_TREE_LIMITS,
        )
        verify_venv_runtime_location(runtime_receipt, python)
        model_receipt = generate_runtime_tree_receipt(
            selected_model_root,
            destination / "model-receipt.json",
            limits=FASTER_WHISPER_MODEL_TREE_LIMITS,
        )
    except RuntimeTreeReceiptError:
        raise ProvisionError() from None
    _verify_runtime_closure(runtime_receipt)
    _verify_model_closure(model_receipt)

    artifact_bindings = {
        "runtime-receipt": _bound_payload(
            runtime_receipt.receipt_path,
            maximum_bytes=artifact_maximum_bytes(
                FASTER_WHISPER_ENDPOINT_ADAPTER,
                "runtime-receipt",
            ),
        ),
        "model-receipt": _bound_payload(
            model_receipt.receipt_path,
            maximum_bytes=artifact_maximum_bytes(
                FASTER_WHISPER_ENDPOINT_ADAPTER,
                "model-receipt",
            ),
        ),
        "venv-marker": marker_binding,
    }
    artifacts = [
        {"name": name, **artifact_bindings[name]}
        for name in FASTER_WHISPER_ARTIFACT_NAMES
    ]
    manifest_payload = {
        "schema_version": 6,
        "model_id": model_id,
        "adapter": FASTER_WHISPER_ENDPOINT_ADAPTER,
        "adapter_config": {
            "python_version": _PYTHON_VERSION,
            "faster_whisper_version": "1.2.1",
            "ctranslate2_version": "4.8.1",
            "numpy_version": "2.4.6",
            "cublas_version": "12.9.2.10",
            "cudnn_version": "9.24.0.43",
            "cuda_nvrtc_version": "12.9.86",
            "language": language,
            "task": "transcribe",
            "device": "cuda",
            "device_index": 0,
            "compute_type": "float16",
            "cpu_threads": 1,
            "num_workers": 1,
            "sample_rate": 16_000,
            "execution_mode": "endpoint-final-only",
            "partial_hypotheses": False,
            "tail_padding_policy": "pace-only-not-decoded",
            "beam_size": 5,
            "patience": 1.0,
            "temperature": 0.0,
            "compression_ratio_threshold": 2.4,
            "log_prob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "vad_filter": False,
            "condition_on_previous_text": False,
            "without_timestamps": True,
            "word_timestamps": False,
            **_tree_summary(runtime_receipt, prefix="runtime"),
            **_tree_summary(model_receipt, prefix="model"),
        },
        "python": python_binding,
        "worker": worker_binding,
        "artifacts": artifacts,
        "limits": {
            "startup_timeout_sec": 300.0,
            "case_timeout_sec": 600.0,
        },
    }
    manifest_path = destination / "worker-manifest.json"
    _write_new_private(manifest_path, _manifest_bytes(manifest_payload))
    try:
        manifest = load_worker_manifest(manifest_path)
    except ManifestError:
        raise ProvisionError() from None
    if (
        manifest.schema_version != 6
        or manifest.adapter != FASTER_WHISPER_ENDPOINT_ADAPTER
        or manifest.artifact_by_name["runtime-receipt"].sha256 != runtime_receipt.digest
        or manifest.artifact_by_name["model-receipt"].sha256 != model_receipt.digest
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
        "model_receipt_sha256": manifest.artifact_by_name["model-receipt"].sha256,
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
            "Bind an existing Python 3.12 Faster-Whisper 1.2.1 environment and "
            "local CTranslate2 model into a final-only isolated-worker "
            "manifest. This command does not install, download, import, or "
            "load the candidate."
        )
    )
    parser.add_argument("--venv-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="faster-whisper-endpoint-local")
    parser.add_argument("--language", choices=("en", "ro"), default="en")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = provision_candidate(
            venv_root=args.venv_root,
            model_root=args.model_root,
            output_dir=args.output_dir,
            model_id=args.model_id,
            language=args.language,
        )
        result: Mapping[str, object] = _safe_result(manifest)
        code = 0
    except Exception:  # noqa: BLE001 - local paths and failures remain private
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
