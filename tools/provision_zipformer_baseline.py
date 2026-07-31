"""Bind the installed production Zipformer into a network-disabled benchmark.

This command imports no ASR runtime, loads no model, downloads nothing, and
does not modify the production virtual environment or model directory. It
hashes the selected files and creates a new private schema-v4 worker receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Callable, Mapping, Sequence

from tools.streaming_stt.bounded_io import (
    BoundedReadError,
    hash_regular_bounded,
    opened_directory_nofollow,
)
from tools.streaming_stt.manifest import (
    MAX_PYTHON_BYTES,
    MAX_WORKER_BYTES,
    SHERPA_ZIPFORMER_ADAPTER,
    SHERPA_ZIPFORMER_ARTIFACT_SPECS,
    ManifestError,
    SherpaZipformerConfig,
    WorkerManifest,
    artifact_maximum_bytes,
    load_worker_manifest,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXED_WORKER = _REPO_ROOT / "tools" / "streaming_stt" / "worker.py"
_SAFE_ERROR = {
    "ok": False,
    "error": "zipformer_baseline_prerequisites_unavailable",
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


def _new_private_output(path: Path | str) -> Path:
    candidate = Path(os.path.abspath(Path(path).expanduser()))
    if not candidate.name or candidate.name in {".", ".."}:
        raise ProvisionError()
    try:
        with opened_directory_nofollow(candidate.parent) as (_parent, descriptor):
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
        metadata_value = path.lstat()
        if (
            not stat.S_ISREG(metadata_value.st_mode)
            or metadata_value.st_nlink != 1
            or stat.S_IMODE(metadata_value.st_mode) != 0o600
            or (
                hasattr(os, "geteuid")
                and metadata_value.st_uid != os.geteuid()
            )
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


def _verify_runtime_versions(
    python: Path,
    config: SherpaZipformerConfig,
) -> None:
    """Probe only installed distribution metadata with a closed offline env."""

    script = (
        "import importlib.metadata as m,sys;"
        f"sys.exit(0 if m.version('sherpa-onnx')=={config.package_version!r} "
        f"and m.version('numpy')=={config.numpy_version!r} else 2)"
    )
    environment = {
        "HOME": "/nonexistent",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    try:
        completed = subprocess.run(
            [str(python), "-I", "-B", "-c", script],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=environment,
            timeout=30.0,
            close_fds=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        raise ProvisionError() from None
    if completed.returncode != 0:
        raise ProvisionError()


def provision_baseline(
    *,
    python: Path | str,
    model_root: Path | str,
    output_dir: Path | str,
    version_probe: Callable[[Path, SherpaZipformerConfig], None] = (
        _verify_runtime_versions
    ),
) -> WorkerManifest:
    """Create a new manifest for the exact installed fp32 production model."""

    selected_model_root = _canonical_directory(model_root, require_private=False)
    selected_python = Path(os.path.abspath(Path(python).expanduser()))
    config = SherpaZipformerConfig()
    version_probe(selected_python, config)

    artifacts = []
    for name, basename, _expected_sha256, _expected_size in (
        SHERPA_ZIPFORMER_ARTIFACT_SPECS
    ):
        artifacts.append(
            {
                "name": name,
                **_bound_payload(
                    selected_model_root / basename,
                    maximum_bytes=artifact_maximum_bytes(
                        SHERPA_ZIPFORMER_ADAPTER,
                        name,
                    ),
                ),
            }
        )

    manifest_payload = {
        "schema_version": 4,
        "model_id": "production-gigaspeech-zipformer-fp32-benchmark-1t",
        "adapter": SHERPA_ZIPFORMER_ADAPTER,
        "adapter_config": config.as_dict(),
        "python": _bound_payload(
            selected_python,
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
    destination = _new_private_output(output_dir)
    manifest_path = destination / "worker-manifest.json"
    _write_new_private(manifest_path, _manifest_bytes(manifest_payload))
    try:
        manifest = load_worker_manifest(manifest_path)
    except ManifestError:
        raise ProvisionError() from None
    if (
        manifest.schema_version != 4
        or manifest.adapter != SHERPA_ZIPFORMER_ADAPTER
        or manifest.adapter_config != config
    ):
        raise ProvisionError()
    return manifest


def _safe_result(manifest: WorkerManifest) -> dict[str, object]:
    artifacts = [
        {
            "name": artifact.name,
            "sha256": artifact.sha256,
            "size_bytes": artifact.size_bytes,
        }
        for artifact in manifest.artifacts
    ]
    return {
        "ok": True,
        "adapter": manifest.adapter,
        "model_id": manifest.model_id,
        "manifest_sha256": manifest.digest,
        "artifact_set_sha256": hashlib.sha256(
            json.dumps(
                artifacts,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "benchmark_num_threads": 1,
        "production_num_threads": 4,
        "runtime_equivalent_to_production": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bind the installed fp32 production GigaSpeech Zipformer into a "
            "one-thread local streaming benchmark manifest. No install or download."
        )
    )
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = provision_baseline(
            python=args.python,
            model_root=args.model_root,
            output_dir=args.output_dir,
        )
        result: Mapping[str, object] = _safe_result(manifest)
        code = 0
    except Exception:  # noqa: BLE001 - local paths remain private
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
