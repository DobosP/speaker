"""Create an exact isolated Moonshine worker receipt without installing it.

The caller supplies a disposable virtual environment, the official release
wheel, and an already downloaded official model directory. This tool verifies
and binds them; it never invokes a package manager or imports Moonshine.
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
)
from tools.streaming_stt.manifest import (
    MAX_PYTHON_BYTES,
    MAX_WORKER_BYTES,
    MOONSHINE_ADAPTER,
    MOONSHINE_ARTIFACT_NAMES,
    ManifestError,
    WorkerManifest,
    artifact_maximum_bytes,
    load_worker_manifest,
)
from tools.streaming_stt.runtime_receipt import (
    RuntimeTreeReceiptError,
    generate_runtime_tree_receipt,
    verify_isolated_venv_marker,
    verify_moonshine_wheel_install,
    verify_venv_runtime_location,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXED_WORKER = _REPO_ROOT / "tools" / "streaming_stt" / "worker.py"
_PYTHON_DIRECTORY_RE = re.compile(r"python[0-9]+\.[0-9]+\Z")
_MODEL_FILES = {
    "model-adapter": "adapter.ort",
    "model-cross-kv": "cross_kv.ort",
    "model-decoder-kv": "decoder_kv.ort",
    "model-encoder": "encoder.ort",
    "model-frontend": "frontend.ort",
    "model-config": "streaming_config.json",
    "model-tokenizer": "tokenizer.bin",
}
_WHEEL_NAME = "moonshine_voice-0.1.0-py3-none-manylinux_2_34_x86_64.whl"
_SAFE_ERROR = {
    "ok": False,
    "error": "moonshine_candidate_prerequisites_unavailable",
}


class ProvisionError(RuntimeError):
    """A detail-free unsafe, incomplete, or changed provision input."""


def _canonical_directory(path: Path | str, *, require_private: bool) -> Path:
    candidate = Path(os.path.abspath(Path(path).expanduser()))
    try:
        with opened_directory_nofollow(
            candidate,
            require_private=require_private,
        ) as (stable, _descriptor):
            return stable
    except BoundedReadError:
        raise ProvisionError() from None


def _site_packages(venv_root: Path) -> Path:
    lib = venv_root / "lib"
    try:
        with opened_directory_nofollow(lib) as (stable_lib, descriptor):
            names = os.listdir(descriptor)
    except (OSError, BoundedReadError):
        raise ProvisionError() from None
    candidates = [
        stable_lib / name / "site-packages"
        for name in names
        if _PYTHON_DIRECTORY_RE.fullmatch(name) is not None
        and (stable_lib / name / "site-packages").is_dir()
    ]
    if len(candidates) != 1:
        raise ProvisionError()
    return _canonical_directory(candidates[0], require_private=True)


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


def _write_new_private(path: Path, payload: bytes) -> None:
    descriptor = -1
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", buffering=0) as handle:
            descriptor = -1
            handle.write(payload)
            os.fsync(handle.fileno())
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
        ):
            raise ProvisionError()
    except (OSError, ValueError, OverflowError, ProvisionError):
        raise ProvisionError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _new_private_output(path: Path | str) -> Path:
    candidate = Path(os.path.abspath(Path(path).expanduser()))
    if not candidate.name or candidate.name in {".", ".."}:
        raise ProvisionError()
    try:
        with opened_directory_nofollow(
            candidate.parent,
        ) as (_parent, descriptor):
            os.mkdir(candidate.name, mode=0o700, dir_fd=descriptor)
        return _canonical_directory(candidate, require_private=True)
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
    wheel: Path | str,
    model_root: Path | str,
    output_dir: Path | str,
    model_arch: str = "tiny-streaming",
) -> WorkerManifest:
    """Verify supplied inputs and create one new private model receipt."""

    if model_arch not in {"tiny-streaming", "small-streaming"}:
        raise ProvisionError()
    venv = _canonical_directory(venv_root, require_private=True)
    site_packages = _site_packages(venv)
    selected_model_root = _canonical_directory(
        model_root,
        require_private=False,
    )
    selected_wheel = Path(os.path.abspath(Path(wheel).expanduser()))
    if selected_wheel.name != _WHEEL_NAME:
        raise ProvisionError()
    python = venv / "bin" / "python"
    marker = venv / "pyvenv.cfg"
    try:
        verify_isolated_venv_marker(marker)
    except RuntimeTreeReceiptError:
        raise ProvisionError() from None
    model_files = {
        name: selected_model_root / basename for name, basename in _MODEL_FILES.items()
    }

    destination = _new_private_output(output_dir)
    try:
        receipt = generate_runtime_tree_receipt(
            site_packages,
            destination / "runtime-receipt.json",
        )
        verify_venv_runtime_location(receipt, python)
    except RuntimeTreeReceiptError:
        raise ProvisionError() from None

    artifact_paths = {
        "runtime-receipt": receipt.receipt_path,
        "venv-marker": marker,
        "release-wheel": selected_wheel,
        **model_files,
    }
    artifact_bindings = {
        name: _bound_payload(
            artifact_paths[name],
            maximum_bytes=artifact_maximum_bytes(MOONSHINE_ADAPTER, name),
        )
        for name in MOONSHINE_ARTIFACT_NAMES
    }
    wheel_binding = artifact_bindings["release-wheel"]
    try:
        verify_moonshine_wheel_install(
            receipt,
            selected_wheel,
            expected_sha256=str(wheel_binding["sha256"]),
            expected_size_bytes=int(wheel_binding["size_bytes"]),
        )
    except RuntimeTreeReceiptError:
        raise ProvisionError() from None
    artifacts = [
        {
            "name": name,
            **artifact_bindings[name],
        }
        for name in MOONSHINE_ARTIFACT_NAMES
    ]
    manifest_payload = {
        "schema_version": 2,
        "model_id": f"moonshine-voice-0.1.0-{model_arch}-en-cpu",
        "adapter": MOONSHINE_ADAPTER,
        "adapter_config": {
            "package_version": "0.1.0",
            "api_version": 30000,
            "model_arch": model_arch,
            "provider": "cpu",
            "language": "en",
        },
        "python": _bound_payload(
            python,
            maximum_bytes=MAX_PYTHON_BYTES,
            allow_final_symlink=True,
        ),
        "worker": _bound_payload(
            _FIXED_WORKER,
            maximum_bytes=MAX_WORKER_BYTES,
        ),
        "artifacts": artifacts,
        "limits": {
            "startup_timeout_sec": 120.0,
            "case_timeout_sec": 300.0,
        },
    }
    manifest_path = destination / "worker-manifest.json"
    _write_new_private(manifest_path, _manifest_bytes(manifest_payload))
    try:
        manifest = load_worker_manifest(manifest_path)
    except ManifestError:
        raise ProvisionError() from None
    if (
        manifest.adapter != MOONSHINE_ADAPTER
        or manifest.artifact_by_name["runtime-receipt"].sha256 != receipt.digest
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
            "Bind an existing disposable Moonshine 0.1.0 environment and an "
            "official streaming model into a private isolated-worker manifest. "
            "This command does not install or download anything."
        )
    )
    parser.add_argument("--venv-root", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-arch",
        choices=("tiny-streaming", "small-streaming"),
        default="tiny-streaming",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = provision_candidate(
            venv_root=args.venv_root,
            wheel=args.wheel,
            model_root=args.model_root,
            output_dir=args.output_dir,
            model_arch=args.model_arch,
        )
        result: Mapping[str, object] = _safe_result(manifest)
        code = 0
    except Exception:  # noqa: BLE001 - local runtime paths remain private
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
