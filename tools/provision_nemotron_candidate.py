"""Bind an existing Nemotron runtime and model without installing either.

The caller supplies a pre-existing private Python 3.12 virtual environment and
the already downloaded official Transformers model directory. This tool checks
and binds those inputs; it never invokes a package manager, downloads a file, or
imports Torch, Transformers, or model code.
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
    MAX_PYTHON_BYTES,
    MAX_WORKER_BYTES,
    NEMOTRON_ADAPTER,
    NEMOTRON_ARTIFACT_NAMES,
    NEMOTRON_RUNTIME_CONTENT_SHA256,
    NEMOTRON_RUNTIME_FILE_COUNT,
    NEMOTRON_RUNTIME_MAXIMUM_FILE_BYTES,
    NEMOTRON_RUNTIME_TOTAL_SIZE_BYTES,
    NEMOTRON_WHEEL_LOCK_SHA256,
    ManifestError,
    NemotronConfig,
    WorkerManifest,
    artifact_maximum_bytes,
    load_worker_manifest,
)
from tools.streaming_stt.runtime_receipt import (
    NEMOTRON_RUNTIME_TREE_LIMITS,
    RuntimeTreeReceiptError,
    generate_runtime_tree_receipt,
    verify_isolated_venv_marker,
    verify_venv_runtime_location,
)
from tools.streaming_stt.wheel_lock import (
    WheelLockError,
    load_wheel_lock,
    verify_locked_wheel_install,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_PRODUCTION_VENV = _REPO_ROOT / ".venv"
_FIXED_WORKER = _REPO_ROOT / "tools" / "streaming_stt" / "worker.py"
_REPOSITORY_WHEEL_LOCK = (
    _REPO_ROOT / "tools" / "streaming_stt" / "nemotron_runtime_wheels.lock.json"
)
_PYTHON_DIRECTORY_RE = re.compile(r"python[0-9]+\.[0-9]+\Z")
_PYTHON_VERSION_RE = re.compile(r"3\.12(?:\.[0-9]+)?\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_MODEL_FILES = {
    "model-config": "config.json",
    "model-generation-config": "generation_config.json",
    "model-weights": "model.safetensors",
    "model-processor-config": "processor_config.json",
    "model-tokenizer": "tokenizer.json",
    "model-tokenizer-config": "tokenizer_config.json",
}
_MODEL_RECEIPTS = {
    "model-config": (
        "62d186fd91f518e00e7867500f1f5819225e8ee95ea3e21b546514bf2048e845",
        1_376,
    ),
    "model-generation-config": (
        "993e5d4cb74a6fe9d6e7084a76b3313c1446740679be4676570c23b664fdc07e",
        193,
    ),
    "model-weights": (
        "9eebdd6590289cb3030f310858f3df93256600a800a3e8200c5993d5f967e174",
        2_552_062_944,
    ),
    "model-processor-config": (
        "ec47870f1091ea4f25539208387b45b902c92d0e3f997a30061ef88f73437ab0",
        2_519,
    ),
    "model-tokenizer": (
        "3f3d481deb073b64c2082e8c7860d487a3a62774bf4e9e4faac83007e181f246",
        752_051,
    ),
    "model-tokenizer-config": (
        "5c641c5b3f50702a60082690d27c1ce7fcb5a92c4a624793bcae0f21eda3d6e0",
        881,
    ),
}
_SAFE_ERROR = {
    "ok": False,
    "error": "nemotron_candidate_prerequisites_unavailable",
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


def _site_packages(venv_root: Path) -> Path:
    lib = _canonical_directory(venv_root / "lib", require_private=False)
    python_directories = {
        name for name in _directory_names(lib) if _PYTHON_DIRECTORY_RE.fullmatch(name)
    }
    if python_directories != {"python3.12"}:
        raise ProvisionError()
    python_root = _canonical_directory(lib / "python3.12", require_private=False)
    return _canonical_directory(
        python_root / "site-packages",
        require_private=True,
    )


def _verify_python312_marker(path: Path) -> None:
    try:
        verify_isolated_venv_marker(path)
        snapshot = read_regular_bounded(
            path,
            maximum_bytes=artifact_maximum_bytes(
                NEMOTRON_ADAPTER,
                "venv-marker",
            ),
        )
        text = snapshot.data.decode("utf-8", errors="strict")
    except (UnicodeError, BoundedReadError, ManifestError, RuntimeTreeReceiptError):
        raise ProvisionError() from None
    versions: list[str] = []
    for raw_line in text.splitlines():
        if "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        if key.strip().casefold() in {"version", "version_info"}:
            versions.append(value.strip())
    if len(versions) != 1 or _PYTHON_VERSION_RE.fullmatch(versions[0]) is None:
        raise ProvisionError()


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


def _exact_model_bindings(model_root: Path) -> dict[str, dict[str, object]]:
    if set(_MODEL_FILES) != {
        name for name in NEMOTRON_ARTIFACT_NAMES if name.startswith("model-")
    } or set(_MODEL_RECEIPTS) != set(_MODEL_FILES):
        raise ProvisionError()
    if set(_directory_names(model_root)) != set(_MODEL_FILES.values()):
        raise ProvisionError()

    bindings: dict[str, dict[str, object]] = {}
    for name, basename in _MODEL_FILES.items():
        path = model_root / basename
        expected_sha256, expected_size = _MODEL_RECEIPTS[name]
        if (
            _SHA256_RE.fullmatch(expected_sha256) is None
            or isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size <= 0
            or expected_size > artifact_maximum_bytes(NEMOTRON_ADAPTER, name)
        ):
            raise ProvisionError()
        try:
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ProvisionError()
            digest = hash_regular_bounded(
                path,
                maximum_bytes=artifact_maximum_bytes(NEMOTRON_ADAPTER, name),
                expected_bytes=expected_size,
            )
        except (OSError, RuntimeError, ValueError, BoundedReadError, ProvisionError):
            raise ProvisionError() from None
        if digest.sha256 != expected_sha256 or digest.path.parent != model_root:
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
                NEMOTRON_ADAPTER,
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
        or len(snapshot.data) != 32_099
        or hashlib.sha256(snapshot.data).hexdigest() != NEMOTRON_WHEEL_LOCK_SHA256
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
                handle.write(payload)
                os.fsync(handle.fileno())
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
    lookahead_tokens: int = 3,
    language: str = "en-US",
) -> WorkerManifest:
    """Verify supplied inputs and create one new private worker receipt."""

    try:
        config = NemotronConfig(
            transformers_version="5.13.1",
            librosa_version="0.11.0",
            torch_version="2.12.1+cu126",
            cuda_version="12.6",
            model_revision="f3d333391852ba876df169dcc9ba902d25b6ab0b",
            lookahead_tokens=lookahead_tokens,
            language=language,
            device="cuda:0",
            dtype="float32",
            wheel_lock_sha256=NEMOTRON_WHEEL_LOCK_SHA256,
            runtime_content_sha256=NEMOTRON_RUNTIME_CONTENT_SHA256,
            runtime_file_count=NEMOTRON_RUNTIME_FILE_COUNT,
            runtime_total_size_bytes=NEMOTRON_RUNTIME_TOTAL_SIZE_BYTES,
            runtime_maximum_file_bytes=NEMOTRON_RUNTIME_MAXIMUM_FILE_BYTES,
        )
    except (TypeError, ValueError, ManifestError):
        raise ProvisionError() from None
    venv = _canonical_directory(venv_root, require_private=True)
    selected_model_root = _canonical_directory(
        model_root,
        require_private=True,
    )
    selected_wheelhouse = _canonical_directory(
        wheelhouse,
        require_private=True,
    )
    roots = (venv, selected_model_root, selected_wheelhouse)
    if any(
        left != right and (_inside(left, right) or _inside(right, left))
        for index, left in enumerate(roots)
        for right in roots[index + 1 :]
    ):
        raise ProvisionError()
    python = venv / "bin" / "python"
    marker = venv / "pyvenv.cfg"
    _verify_python312_marker(marker)
    site_packages = _site_packages(venv)
    model_bindings = _exact_model_bindings(selected_model_root)

    destination = _new_private_output(
        output_dir,
        forbidden_roots=roots,
    )
    runtime_wheel_lock = destination / "nemotron-runtime-wheels.lock.json"
    _write_new_private(runtime_wheel_lock, _repository_wheel_lock_bytes())
    try:
        lock = load_wheel_lock(
            runtime_wheel_lock,
            expected_digest=NEMOTRON_WHEEL_LOCK_SHA256,
        )
        receipt = generate_runtime_tree_receipt(
            site_packages,
            destination / "runtime-receipt.json",
            limits=NEMOTRON_RUNTIME_TREE_LIMITS,
        )
        verify_venv_runtime_location(receipt, python)
        expectation = verify_locked_wheel_install(
            lock,
            selected_wheelhouse,
            receipt,
        )
    except (RuntimeTreeReceiptError, WheelLockError):
        raise ProvisionError() from None
    if (
        lock.digest != NEMOTRON_WHEEL_LOCK_SHA256
        or receipt.content_digest != NEMOTRON_RUNTIME_CONTENT_SHA256
        or expectation.content_digest != NEMOTRON_RUNTIME_CONTENT_SHA256
        or expectation.file_count != NEMOTRON_RUNTIME_FILE_COUNT
        or expectation.total_size_bytes != NEMOTRON_RUNTIME_TOTAL_SIZE_BYTES
        or expectation.maximum_file_bytes != NEMOTRON_RUNTIME_MAXIMUM_FILE_BYTES
    ):
        raise ProvisionError()

    artifact_paths = {
        "runtime-receipt": receipt.receipt_path,
        "runtime-wheel-lock": runtime_wheel_lock,
        "venv-marker": marker,
    }
    artifact_bindings = {
        name: _bound_payload(
            artifact_paths[name],
            maximum_bytes=artifact_maximum_bytes(NEMOTRON_ADAPTER, name),
        )
        for name in ("runtime-receipt", "runtime-wheel-lock", "venv-marker")
    }
    artifact_bindings.update(model_bindings)
    artifacts = [
        {
            "name": name,
            **artifact_bindings[name],
        }
        for name in NEMOTRON_ARTIFACT_NAMES
    ]
    model_id = (
        "nemotron-3.5-asr-streaming-0.6b-"
        f"{language.casefold()}-la{lookahead_tokens}-cuda"
    )
    manifest_payload = {
        "schema_version": 3,
        "model_id": model_id,
        "adapter": NEMOTRON_ADAPTER,
        "adapter_config": config.as_dict(),
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
            "startup_timeout_sec": 300.0,
            "case_timeout_sec": 900.0,
        },
    }
    manifest_path = destination / "worker-manifest.json"
    _write_new_private(manifest_path, _manifest_bytes(manifest_payload))
    try:
        manifest = load_worker_manifest(manifest_path)
    except ManifestError:
        raise ProvisionError() from None
    if (
        manifest.schema_version != 3
        or manifest.adapter != NEMOTRON_ADAPTER
        or manifest.adapter_config != config
        or manifest.artifact_by_name["runtime-receipt"].sha256 != receipt.digest
        or manifest.artifact_by_name["runtime-wheel-lock"].sha256
        != NEMOTRON_WHEEL_LOCK_SHA256
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
            "Bind an existing private Python 3.12 Nemotron environment and the "
            "official six-file model into an isolated worker manifest. This "
            "command does not install or download anything and does not import "
            "the candidate runtime."
        )
    )
    parser.add_argument("--venv-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--lookahead-tokens",
        type=int,
        choices=(0, 3, 6, 13),
        default=3,
    )
    parser.add_argument(
        "--language",
        choices=("en-US", "en-GB", "ro-RO"),
        default="en-US",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = provision_candidate(
            venv_root=args.venv_root,
            model_root=args.model_root,
            wheelhouse=args.wheelhouse,
            output_dir=args.output_dir,
            lookahead_tokens=args.lookahead_tokens,
            language=args.language,
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
