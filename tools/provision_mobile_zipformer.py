"""Provision the exact mobile Zipformer tuple without executing model code.

Only pre-existing, owner-private model bytes are accepted.  The command does
not download, import sherpa-onnx or NumPy, load an ONNX graph, use a GPU, or
open an audio device.  Runtime versions are read only through distribution
metadata in an isolated child interpreter.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

from tools.streaming_stt.bounded_io import (
    BoundedReadError,
    hash_regular_bounded,
    opened_directory_nofollow,
    read_regular_bounded,
)
from tools.streaming_stt.manifest import (
    MAX_PYTHON_BYTES,
    MAX_WORKER_BYTES,
    BoundArtifact,
    BoundFile,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCK_PATH = (
    _REPO_ROOT
    / "tools"
    / "streaming_stt"
    / "mobile-zipformer-en-2023-06-26-v1.lock.json"
)
_FIXED_WORKER = _REPO_ROOT / "tools" / "streaming_stt" / "worker.py"

LOCK_RAW_SHA256 = "3b928f709a1c50f426291968bedff67811f7eed0e1c48e0773df475558520efc"
LOCK_SIZE_BYTES = 2_335
LOCK_RECIPE_SHA256 = "736d516ceada4bc8864ca0b5de9f03167a415cb93ad66252ab99351028a15c49"
MODEL_ID = "sherpa-onnx-streaming-zipformer-en-2023-06-26-mobile-hybrid-v1"
ADAPTER = "sherpa-onnx-mobile-zipformer-endpoint-v1"
MANIFEST_KIND = "mobile-zipformer-provision-v1"
RECEIPT_KIND = "mobile-zipformer-provision-receipt-v1"
MANIFEST_FILENAME = "worker-manifest.json"
RECEIPT_FILENAME = "provision-receipt.json"
LOCK_COPY_FILENAME = "source-lock.json"
MODEL_DIRECTORY = "model"
TOTAL_SIZE_BYTES = 74_207_237

_MAX_METADATA_BYTES = 256 * 1024
_COPY_CHUNK_BYTES = 1024 * 1024
_ARTIFACT_SET_DOMAIN = b"speaker-mobile-zipformer-artifact-set-v1\0"
_LOCK_RECIPE_DOMAIN = b"speaker-mobile-zipformer-source-lock-recipe-v1\0"
_PROVISION_SET_DOMAIN = b"speaker-mobile-zipformer-provision-set-v1\0"
_SAFE_ERROR = {
    "ok": False,
    "error": "mobile_zipformer_prerequisites_unavailable",
}
_EXPECTED_RUNTIME_DISTRIBUTIONS = {
    "numpy": "2.4.6",
    "sherpa-onnx": "1.13.3",
    "sherpa-onnx-core": "1.13.3",
}
_RUNTIME_METADATA_MAX_BYTES = 128 * 1024
_RUNTIME_METADATA_SCRIPT = (
    """\
import os
import stat
import sys
import sysconfig

EXPECTED = (
    (b"numpy", "numpy", "2.4.6"),
    (b"sherpa-onnx", "sherpa_onnx", "1.13.3"),
    (b"sherpa-onnx-core", "sherpa_onnx_core", "1.13.3"),
)
MAXIMUM_BYTES = %d
BLOCKED_MODULES = {
    "numpy",
    "sherpa_onnx",
    "site",
    "sitecustomize",
    "usercustomize",
}


def fail():
    raise SystemExit(2)


def identity(metadata):
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def metadata_roots():
    executable = os.path.abspath(sys.executable)
    environment_root = os.path.dirname(os.path.dirname(executable))
    configuration = os.path.join(environment_root, "pyvenv.cfg")
    try:
        configuration_metadata = os.lstat(configuration)
    except FileNotFoundError:
        configured = {
            sysconfig.get_path("purelib"),
            sysconfig.get_path("platlib"),
        }
        roots = sorted(path for path in configured if path)
    else:
        if (
            not stat.S_ISREG(configuration_metadata.st_mode)
            or configuration_metadata.st_size <= 0
            or configuration_metadata.st_size > MAXIMUM_BYTES
        ):
            fail()
        roots = [
            os.path.join(
                environment_root,
                "lib",
                f"python{sys.version_info.major}.{sys.version_info.minor}",
                "site-packages",
            )
        ]
    if not roots:
        fail()
    return roots


def read_metadata(root, directory_name):
    directory_flags = (
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    root_descriptor = os.open(root, directory_flags)
    metadata_directory = -1
    descriptor = -1
    try:
        root_metadata = os.fstat(root_descriptor)
        if not stat.S_ISDIR(root_metadata.st_mode):
            fail()
        metadata_directory = os.open(
            directory_name,
            directory_flags,
            dir_fd=root_descriptor,
        )
        descriptor = os.open(
            "METADATA",
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=metadata_directory,
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > MAXIMUM_BYTES
        ):
            fail()
        remaining = before.st_size
        chunks = []
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                fail()
            chunks.append(chunk)
            remaining -= len(chunk)
        if (
            os.read(descriptor, 1)
            or identity(os.fstat(descriptor)) != identity(before)
        ):
            fail()
        return b"".join(chunks)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if metadata_directory >= 0:
            os.close(metadata_directory)
        os.close(root_descriptor)


def normalize_name(value):
    normalized = bytearray()
    separator = False
    for character in value.lower():
        if character in b"-_.":
            if not separator:
                normalized.append(ord("-"))
            separator = True
        else:
            normalized.append(character)
            separator = False
    return bytes(normalized)


def verify_headers(raw, expected_name, expected_version):
    names = []
    versions = []
    for line in raw.splitlines():
        if not line:
            break
        key, separator, value = line.partition(b":")
        if not separator:
            continue
        if key.lower() == b"name":
            names.append(value.strip())
        elif key.lower() == b"version":
            versions.append(value.strip())
    if (
        len(names) != 1
        or normalize_name(names[0]) != expected_name
        or versions != [expected_version.encode("ascii")]
    ):
        fail()


if not sys.flags.isolated or not sys.flags.no_site:
    fail()
if BLOCKED_MODULES.intersection(sys.modules):
    fail()
try:
    roots = metadata_roots()
    for expected_name, stem, expected_version in EXPECTED:
        directory_name = f"{stem}-{expected_version}.dist-info"
        matches = []
        for root in roots:
            try:
                matches.append(read_metadata(root, directory_name))
            except FileNotFoundError:
                continue
        if len(matches) != 1:
            fail()
        verify_headers(matches[0], expected_name, expected_version)
except (OSError, ValueError):
    fail()
if BLOCKED_MODULES.intersection(sys.modules):
    fail()
"""
    % _RUNTIME_METADATA_MAX_BYTES
)
_EXPECTED_MOBILE_CONFIG = {
    "core_package_version": "1.13.3",
    "debug": True,
    "decoding_method": "greedy_search",
    "enable_endpoint_detection": True,
    "feature_dim": 80,
    "language": "en",
    "max_active_paths": 4,
    "maximum_tail_padding_samples": 48_000,
    "model_type": "zipformer2",
    "native_chunk_samples": 1_600,
    "num_threads": 1,
    "numpy_version": "2.4.6",
    "package_version": "1.13.3",
    "provider": "cpu",
    "rule1_min_trailing_silence": 2.4,
    "rule2_min_trailing_silence": 0.8,
    "rule3_min_utterance_length": 20.0,
    "sample_rate": 16_000,
    "source_repo_id": ("csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26"),
    "source_revision": "672fbf1b30579d6585301139bb363f42a0ad4a24",
    "variant": "epoch-99-avg-1-chunk-16-left-128-mobile-hybrid",
}
_EXPECTED_EVIDENCE_SCOPE = {
    "audio_device_opened": False,
    "downloaded": False,
    "evaluation_result_authority": False,
    "gpu_used": False,
    "mobile_device_evidence": False,
    "model_executed": False,
    "model_loaded": False,
    "packages_imported": False,
}
_EXPECTED_ARTIFACTS = (
    (
        "model-encoder",
        "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
        "int8",
        71_083_163,
        "563fde436d16cf7607cf408cd6b30909819d03162652ef389c2450ced3f45ac1",
    ),
    (
        "model-decoder",
        "decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
        "fp32",
        2_092_621,
        "7bf787f90b194b307e5a4ad6a34fadb4e748304c35f78a8d66358a05b13ee6ef",
    ),
    (
        "model-joiner",
        "joiner-epoch-99-avg-1-chunk-16-left-128.onnx",
        "fp32",
        1_026_405,
        "210591f72b3c56b8364f85f345dca240bc2b4c00632848f4aa923630d5639d3b",
    ),
    (
        "model-tokens",
        "tokens.txt",
        "text",
        5_048,
        "49e3c2646595fd907228b3c6787069658f67b17377c60aeb8619c4551b2316fb",
    ),
)


class MobileZipformerProvisionError(RuntimeError):
    """A detail-free unsafe, incomplete, or changed provision input."""


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    filename: str
    precision: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class MobileZipformerConfig:
    package_version: str
    core_package_version: str
    numpy_version: str
    source_repo_id: str
    source_revision: str
    variant: str
    language: str
    sample_rate: int
    feature_dim: int
    num_threads: int
    provider: str
    debug: bool
    decoding_method: str
    max_active_paths: int
    model_type: str
    enable_endpoint_detection: bool
    rule1_min_trailing_silence: float
    rule2_min_trailing_silence: float
    rule3_min_utterance_length: float
    native_chunk_samples: int
    maximum_tail_padding_samples: int

    def as_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in _EXPECTED_MOBILE_CONFIG}


@dataclass(frozen=True)
class SourceLock:
    path: Path
    raw: bytes
    raw_sha256: str
    recipe_sha256: str
    repo_id: str
    revision: str
    artifacts: tuple[ArtifactSpec, ...]
    runtime_distributions: Mapping[str, str]
    mobile_config: MobileZipformerConfig
    total_size_bytes: int


@dataclass(frozen=True)
class _SourceArtifact:
    spec: ArtifactSpec
    path: Path
    identity: tuple[int, ...]


@dataclass(frozen=True)
class MobileZipformerPreflight:
    lock: SourceLock
    python: BoundFile
    worker: BoundFile
    source_root: Path
    source_root_identity: tuple[int, ...]
    source_artifacts: tuple[_SourceArtifact, ...]
    artifact_set_sha256: str


@dataclass(frozen=True)
class MobileZipformerProvision:
    path: Path
    digest: str
    receipt_digest: str
    model_id: str
    adapter: str
    python: BoundFile
    worker: BoundFile
    artifacts: tuple[BoundArtifact, ...]
    mobile_config: MobileZipformerConfig
    artifact_set_sha256: str
    total_size_bytes: int


@dataclass(frozen=True)
class _StagedReceipt:
    size_bytes: int
    sha256: str
    identity: tuple[int, ...]


@dataclass
class _TerminalCommitState:
    committed: bool = False


@dataclass(frozen=True)
class _ParsedCommand:
    action: str
    python: str
    model_root: str
    output_dir: str | None


def _fail() -> None:
    raise MobileZipformerProvisionError()


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (OverflowError, TypeError, ValueError):
        _fail()


def _strict_json(raw: bytes) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if type(key) is not str or key in result:
                _fail()
            result[key] = value
        return result

    def reject_constant(_value: str) -> object:
        _fail()

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except MobileZipformerProvisionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, TypeError):
        _fail()


def _exact_dict(value: object, keys: set[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        _fail()
    return value


def _exact_string(value: object) -> str:
    if type(value) is not str or not value or len(value) > 512:
        _fail()
    return value


def _exact_sha256(value: object) -> str:
    text = _exact_string(value)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        _fail()
    return text


def _exact_positive_int(value: object) -> int:
    if type(value) is not int or value <= 0:
        _fail()
    return value


def _absolute(path: Path | str) -> Path:
    try:
        supplied = Path(path).expanduser()
        candidate = Path(os.path.abspath(supplied))
    except (OSError, RuntimeError, TypeError, ValueError):
        _fail()
    if not supplied.is_absolute() or candidate != supplied:
        _fail()
    return candidate


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


def _directory_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        stat.S_IMODE(metadata.st_mode),
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _owned(metadata: os.stat_result) -> bool:
    return not hasattr(os, "geteuid") or metadata.st_uid == os.geteuid()


def _private_regular(metadata: os.stat_result, *, size_bytes: int) -> bool:
    return (
        stat.S_ISREG(metadata.st_mode)
        and stat.S_IMODE(metadata.st_mode) == 0o600
        and metadata.st_nlink == 1
        and metadata.st_size == size_bytes
        and _owned(metadata)
    )


def _private_directory_metadata(metadata: os.stat_result) -> bool:
    return (
        stat.S_ISDIR(metadata.st_mode)
        and stat.S_IMODE(metadata.st_mode) == 0o700
        and _owned(metadata)
    )


def _safe_leaf(name: object) -> str:
    value = _exact_string(name)
    if value in {".", ".."} or "/" in value or "\x00" in value:
        _fail()
    return value


def _has_git_ancestor(path: Path) -> bool:
    try:
        current = path.resolve(strict=True)
        while True:
            try:
                (current / ".git").lstat()
            except FileNotFoundError:
                pass
            else:
                return True
            if current.parent == current:
                return False
            current = current.parent
    except (OSError, RuntimeError, ValueError):
        _fail()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _overlaps(left: Path, right: Path) -> bool:
    return _inside(left, right) or _inside(right, left)


def _mobile_config(value: object) -> MobileZipformerConfig:
    mapping = _exact_dict(value, set(_EXPECTED_MOBILE_CONFIG))
    for key, expected in _EXPECTED_MOBILE_CONFIG.items():
        actual = mapping[key]
        if type(actual) is not type(expected) or actual != expected:
            _fail()
    return MobileZipformerConfig(**mapping)


def _artifact_set_sha256(artifacts: Sequence[ArtifactSpec]) -> str:
    rows = [
        {
            "filename": item.filename,
            "name": item.name,
            "precision": item.precision,
            "sha256": item.sha256,
            "size_bytes": item.size_bytes,
        }
        for item in artifacts
    ]
    return hashlib.sha256(_ARTIFACT_SET_DOMAIN + _canonical_json(rows)).hexdigest()


def _load_source_lock(
    path: Path = _LOCK_PATH,
    *,
    expected_raw_sha256: str = LOCK_RAW_SHA256,
    expected_size_bytes: int = LOCK_SIZE_BYTES,
) -> SourceLock:
    try:
        snapshot = read_regular_bounded(
            path,
            maximum_bytes=expected_size_bytes,
            expected_bytes=expected_size_bytes,
        )
        raw = snapshot.data
    except BoundedReadError:
        _fail()
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    if raw_sha256 != expected_raw_sha256:
        _fail()
    value = _exact_dict(
        _strict_json(raw),
        {
            "artifacts",
            "kind",
            "mobile_config",
            "model_id",
            "recipe_sha256",
            "runtime_distributions",
            "schema_version",
            "source",
            "total_size_bytes",
        },
    )
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["kind"] != "mobile-zipformer-source-lock-v1"
        or value["model_id"] != MODEL_ID
        or value["total_size_bytes"] != TOTAL_SIZE_BYTES
    ):
        _fail()
    source = _exact_dict(value["source"], {"repo_id", "revision"})
    repo_id = _exact_string(source["repo_id"])
    revision = _exact_string(source["revision"])
    if (
        repo_id != _EXPECTED_MOBILE_CONFIG["source_repo_id"]
        or revision != _EXPECTED_MOBILE_CONFIG["source_revision"]
    ):
        _fail()
    runtime = _exact_dict(
        value["runtime_distributions"],
        set(_EXPECTED_RUNTIME_DISTRIBUTIONS),
    )
    if any(
        type(runtime[key]) is not str or runtime[key] != expected
        for key, expected in _EXPECTED_RUNTIME_DISTRIBUTIONS.items()
    ):
        _fail()
    rows = value["artifacts"]
    if type(rows) is not list or len(rows) != len(_EXPECTED_ARTIFACTS):
        _fail()
    artifacts: list[ArtifactSpec] = []
    for row, expected in zip(rows, _EXPECTED_ARTIFACTS, strict=True):
        item = _exact_dict(
            row,
            {"filename", "name", "precision", "sha256", "size_bytes"},
        )
        spec = ArtifactSpec(
            name=_safe_leaf(item["name"]),
            filename=_safe_leaf(item["filename"]),
            precision=_exact_string(item["precision"]),
            size_bytes=_exact_positive_int(item["size_bytes"]),
            sha256=_exact_sha256(item["sha256"]),
        )
        if (
            spec.name,
            spec.filename,
            spec.precision,
            spec.size_bytes,
            spec.sha256,
        ) != expected:
            _fail()
        artifacts.append(spec)
    recipe = _exact_sha256(value["recipe_sha256"])
    recipe_value = dict(value)
    del recipe_value["recipe_sha256"]
    calculated_recipe = hashlib.sha256(
        _LOCK_RECIPE_DOMAIN + _canonical_json(recipe_value)
    ).hexdigest()
    if recipe != LOCK_RECIPE_SHA256 or recipe != calculated_recipe:
        _fail()
    config = _mobile_config(value["mobile_config"])
    if sum(item.size_bytes for item in artifacts) != TOTAL_SIZE_BYTES:
        _fail()
    return SourceLock(
        path=snapshot.path,
        raw=raw,
        raw_sha256=raw_sha256,
        recipe_sha256=recipe,
        repo_id=repo_id,
        revision=revision,
        artifacts=tuple(artifacts),
        runtime_distributions=MappingProxyType(dict(runtime)),
        mobile_config=config,
        total_size_bytes=TOTAL_SIZE_BYTES,
    )


def _bind_file(
    path: Path,
    *,
    maximum_bytes: int,
    allow_final_symlink: bool,
) -> BoundFile:
    candidate = _absolute(path)
    try:
        lexical = candidate.lstat()
        if stat.S_ISLNK(lexical.st_mode):
            if not allow_final_symlink or lexical.st_nlink != 1:
                _fail()
        elif not stat.S_ISREG(lexical.st_mode) or lexical.st_nlink != 1:
            _fail()
        resolved = candidate.resolve(strict=True)
        target = resolved.lstat()
        if (
            not stat.S_ISREG(target.st_mode)
            or target.st_nlink != 1
            or target.st_size <= 0
            or target.st_size > maximum_bytes
        ):
            _fail()
        digest = hash_regular_bounded(
            candidate,
            maximum_bytes=maximum_bytes,
            expected_bytes=target.st_size,
            allow_final_symlink=allow_final_symlink,
        )
        if digest.path != resolved:
            _fail()
        return BoundFile(
            path=candidate,
            sha256=digest.sha256,
            size_bytes=digest.size_bytes,
        )
    except MobileZipformerProvisionError:
        raise
    except (BoundedReadError, OSError, RuntimeError, ValueError):
        _fail()


def _same_bound_file(left: BoundFile, right: BoundFile) -> bool:
    return (
        left.path == right.path
        and left.sha256 == right.sha256
        and left.size_bytes == right.size_bytes
    )


def _bind_private_artifact(path: Path, spec: ArtifactSpec) -> _SourceArtifact:
    candidate = _absolute(path)
    try:
        before = candidate.lstat()
        if candidate.resolve(strict=True) != candidate or not _private_regular(
            before, size_bytes=spec.size_bytes
        ):
            _fail()
        digest = hash_regular_bounded(
            candidate,
            maximum_bytes=spec.size_bytes,
            expected_bytes=spec.size_bytes,
        )
        after = candidate.lstat()
        if (
            digest.path != candidate
            or digest.sha256 != spec.sha256
            or digest.size_bytes != spec.size_bytes
            or _identity(after) != _identity(before)
        ):
            _fail()
        return _SourceArtifact(
            spec=spec,
            path=candidate,
            identity=_identity(after),
        )
    except MobileZipformerProvisionError:
        raise
    except (BoundedReadError, OSError, RuntimeError, ValueError):
        _fail()


def _private_directory(
    path: Path | str,
    *,
    expected_names: set[str] | None = None,
) -> tuple[Path, tuple[int, ...]]:
    candidate = _absolute(path)
    try:
        with opened_directory_nofollow(
            candidate,
            require_private=True,
        ) as (stable, descriptor):
            opened = os.fstat(descriptor)
            named = candidate.lstat()
            if (
                stable != candidate
                or not _private_directory_metadata(opened)
                or _directory_identity(named) != _directory_identity(opened)
                or (
                    expected_names is not None
                    and set(os.listdir(descriptor)) != expected_names
                )
            ):
                _fail()
            return stable, _directory_identity(opened)
    except MobileZipformerProvisionError:
        raise
    except (BoundedReadError, OSError, RuntimeError, ValueError):
        _fail()


def _verify_runtime_versions(
    python: Path,
    expected: Mapping[str, str],
) -> None:
    if dict(expected) != _EXPECTED_RUNTIME_DISTRIBUTIONS:
        _fail()
    environment = {
        "HOME": "/nonexistent",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    try:
        completed = subprocess.run(
            [str(python), "-I", "-B", "-S", "-c", _RUNTIME_METADATA_SCRIPT],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=environment,
            timeout=30.0,
            close_fds=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        _fail()
    if completed.returncode != 0:
        _fail()


def _revalidate_preflight(preflight: MobileZipformerPreflight) -> None:
    root, identity = _private_directory(
        preflight.source_root,
        expected_names={item.spec.filename for item in preflight.source_artifacts},
    )
    if root != preflight.source_root or identity != preflight.source_root_identity:
        _fail()
    for expected in preflight.source_artifacts:
        current = _bind_private_artifact(expected.path, expected.spec)
        if current.identity != expected.identity:
            _fail()
    python = _bind_file(
        preflight.python.path,
        maximum_bytes=MAX_PYTHON_BYTES,
        allow_final_symlink=True,
    )
    worker = _bind_file(
        preflight.worker.path,
        maximum_bytes=MAX_WORKER_BYTES,
        allow_final_symlink=False,
    )
    if not _same_bound_file(python, preflight.python) or not _same_bound_file(
        worker,
        preflight.worker,
    ):
        _fail()


def _preflight_with_lock(
    *,
    python: Path | str,
    model_root: Path | str,
    lock: SourceLock,
    version_probe: Callable[[Path, Mapping[str, str]], None],
) -> MobileZipformerPreflight:
    selected_python = _absolute(python)
    selected_root, root_identity = _private_directory(
        model_root,
        expected_names={item.filename for item in lock.artifacts},
    )
    if _has_git_ancestor(selected_root):
        _fail()
    python_before = _bind_file(
        selected_python,
        maximum_bytes=MAX_PYTHON_BYTES,
        allow_final_symlink=True,
    )
    worker_before = _bind_file(
        _FIXED_WORKER,
        maximum_bytes=MAX_WORKER_BYTES,
        allow_final_symlink=False,
    )
    source_artifacts = tuple(
        _bind_private_artifact(selected_root / spec.filename, spec)
        for spec in lock.artifacts
    )
    try:
        version_probe(selected_python, lock.runtime_distributions)
    except MobileZipformerProvisionError:
        raise
    except BaseException as error:
        if isinstance(error, (GeneratorExit, KeyboardInterrupt, SystemExit)):
            raise
        _fail()
    preflight = MobileZipformerPreflight(
        lock=lock,
        python=python_before,
        worker=worker_before,
        source_root=selected_root,
        source_root_identity=root_identity,
        source_artifacts=source_artifacts,
        artifact_set_sha256=_artifact_set_sha256(lock.artifacts),
    )
    _revalidate_preflight(preflight)
    return preflight


def preflight_mobile_zipformer(
    *,
    python: Path | str,
    model_root: Path | str,
) -> MobileZipformerPreflight:
    """Hash and validate the exact staged model and metadata-only runtime."""

    try:
        lock = _load_source_lock()
        return _preflight_with_lock(
            python=python,
            model_root=model_root,
            lock=lock,
            version_probe=_verify_runtime_versions,
        )
    except MobileZipformerProvisionError:
        raise
    except (BoundedReadError, OSError, RuntimeError, TypeError, ValueError):
        _fail()


def _new_private_output(
    path: Path | str,
    *,
    forbidden: Sequence[Path],
) -> Path:
    candidate = _absolute(path)
    if (
        not candidate.name
        or candidate.name in {".", ".."}
        or "\x00" in candidate.name
        or any(_overlaps(candidate, item) for item in forbidden)
    ):
        _fail()
    parent, parent_identity = _private_directory(candidate.parent)
    if _has_git_ancestor(parent):
        _fail()
    try:
        with opened_directory_nofollow(
            parent,
            require_private=True,
        ) as (stable_parent, descriptor):
            before = os.fstat(descriptor)
            if (
                stable_parent != parent
                or _directory_identity(before) != parent_identity
            ):
                _fail()
            os.mkdir(candidate.name, mode=0o700, dir_fd=descriptor)
            created = os.stat(
                candidate.name,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
            if not _private_directory_metadata(created):
                _fail()
            os.fsync(descriptor)
            after = os.stat(
                candidate.name,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
            if _directory_identity(after) != _directory_identity(created):
                _fail()
        root, _identity_value = _private_directory(candidate, expected_names=set())
        return root
    except MobileZipformerProvisionError:
        raise
    except (BoundedReadError, OSError, RuntimeError, ValueError):
        _fail()


def _mkdir_private(parent: Path, name: str) -> Path:
    leaf = _safe_leaf(name)
    try:
        with opened_directory_nofollow(
            parent,
            require_private=True,
        ) as (stable, descriptor):
            if stable != parent:
                _fail()
            os.mkdir(leaf, mode=0o700, dir_fd=descriptor)
            created = os.stat(leaf, dir_fd=descriptor, follow_symlinks=False)
            if not _private_directory_metadata(created):
                _fail()
            os.fsync(descriptor)
        child, _identity_value = _private_directory(
            parent / leaf,
            expected_names=set(),
        )
        return child
    except MobileZipformerProvisionError:
        raise
    except (BoundedReadError, OSError, RuntimeError, ValueError):
        _fail()


def _write_new_private(parent: Path, name: str, payload: bytes) -> BoundFile:
    leaf = _safe_leaf(name)
    if type(payload) is not bytes or not payload or len(payload) > _MAX_METADATA_BYTES:
        _fail()
    descriptor = -1
    try:
        with opened_directory_nofollow(
            parent,
            require_private=True,
        ) as (stable, parent_descriptor):
            if stable != parent:
                _fail()
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(leaf, flags, 0o600, dir_fd=parent_descriptor)
            os.fchmod(descriptor, 0o600)
            created = os.fstat(descriptor)
            if not _private_regular(created, size_bytes=0):
                _fail()
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    _fail()
                view = view[written:]
            os.fsync(descriptor)
            final_descriptor = os.fstat(descriptor)
            final_named = os.stat(
                leaf,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if not _private_regular(
                final_descriptor, size_bytes=len(payload)
            ) or _identity(final_named) != _identity(final_descriptor):
                _fail()
            os.fsync(parent_descriptor)
        return BoundFile(
            path=parent / leaf,
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
        )
    except MobileZipformerProvisionError:
        raise
    except (BoundedReadError, OSError, RuntimeError, ValueError):
        _fail()
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _copy_artifact(
    source: _SourceArtifact,
    destination: Path,
) -> BoundArtifact:
    source_descriptor = -1
    destination_descriptor = -1
    try:
        with (
            opened_directory_nofollow(
                source.path.parent,
                require_private=True,
            ) as (stable_source, source_root_descriptor),
            opened_directory_nofollow(
                destination,
                require_private=True,
            ) as (stable_destination, destination_root_descriptor),
        ):
            if stable_source != source.path.parent or stable_destination != destination:
                _fail()
            source_before = os.stat(
                source.spec.filename,
                dir_fd=source_root_descriptor,
                follow_symlinks=False,
            )
            if _identity(source_before) != source.identity or not _private_regular(
                source_before,
                size_bytes=source.spec.size_bytes,
            ):
                _fail()
            read_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            read_flags |= getattr(os, "O_NOFOLLOW", 0)
            source_descriptor = os.open(
                source.spec.filename,
                read_flags,
                dir_fd=source_root_descriptor,
            )
            source_opened = os.fstat(source_descriptor)
            if _identity(source_opened) != source.identity:
                _fail()

            write_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            write_flags |= getattr(os, "O_CLOEXEC", 0)
            write_flags |= getattr(os, "O_NOFOLLOW", 0)
            destination_descriptor = os.open(
                source.spec.filename,
                write_flags,
                0o600,
                dir_fd=destination_root_descriptor,
            )
            os.fchmod(destination_descriptor, 0o600)
            created = os.fstat(destination_descriptor)
            if not _private_regular(created, size_bytes=0):
                _fail()

            digest = hashlib.sha256()
            consumed = 0
            while consumed <= source.spec.size_bytes:
                chunk = os.read(
                    source_descriptor,
                    min(
                        _COPY_CHUNK_BYTES,
                        source.spec.size_bytes + 1 - consumed,
                    ),
                )
                if not chunk:
                    break
                consumed += len(chunk)
                if consumed > source.spec.size_bytes:
                    _fail()
                digest.update(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(destination_descriptor, view)
                    if written <= 0:
                        _fail()
                    view = view[written:]
            os.fsync(destination_descriptor)

            source_after = os.fstat(source_descriptor)
            source_named = os.stat(
                source.spec.filename,
                dir_fd=source_root_descriptor,
                follow_symlinks=False,
            )
            destination_after = os.fstat(destination_descriptor)
            destination_named = os.stat(
                source.spec.filename,
                dir_fd=destination_root_descriptor,
                follow_symlinks=False,
            )
            if (
                consumed != source.spec.size_bytes
                or digest.hexdigest() != source.spec.sha256
                or _identity(source_after) != source.identity
                or _identity(source_named) != source.identity
                or not _private_regular(
                    destination_after,
                    size_bytes=source.spec.size_bytes,
                )
                or _identity(destination_named) != _identity(destination_after)
            ):
                _fail()
            os.fsync(destination_root_descriptor)
        bound = _bind_private_artifact(
            destination / source.spec.filename,
            source.spec,
        )
        return BoundArtifact(
            name=source.spec.name,
            path=bound.path,
            sha256=source.spec.sha256,
            size_bytes=source.spec.size_bytes,
        )
    except MobileZipformerProvisionError:
        raise
    except (BoundedReadError, OSError, RuntimeError, ValueError):
        _fail()
    finally:
        for descriptor in (source_descriptor, destination_descriptor):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass


def _bound_payload(item: BoundFile) -> dict[str, object]:
    return {
        "path": str(item.path),
        "sha256": item.sha256,
        "size_bytes": item.size_bytes,
    }


def _artifact_payload(item: BoundArtifact) -> dict[str, object]:
    return {
        "name": item.name,
        **_bound_payload(item),
    }


def _manifest_payload(
    preflight: MobileZipformerPreflight,
    retained_lock: BoundFile,
    artifacts: tuple[BoundArtifact, ...],
) -> dict[str, object]:
    python_payload = _bound_payload(preflight.python)
    worker_payload = _bound_payload(preflight.worker)
    return {
        "adapter": ADAPTER,
        "artifact_set_sha256": preflight.artifact_set_sha256,
        "artifacts": [_artifact_payload(item) for item in artifacts],
        "evidence_scope": dict(_EXPECTED_EVIDENCE_SCOPE),
        "kind": MANIFEST_KIND,
        "limits": {
            "case_timeout_sec": 300.0,
            "startup_timeout_sec": 120.0,
        },
        "mobile_config": preflight.lock.mobile_config.as_dict(),
        "model_id": MODEL_ID,
        "python": python_payload,
        "runtime": {
            "distributions": dict(preflight.lock.runtime_distributions),
            "metadata_only_verified": True,
            "model_loaded": False,
            "packages_imported": False,
            "python": python_payload,
            "worker": worker_payload,
        },
        "schema_version": 10,
        "source": {
            "lock": _bound_payload(retained_lock),
            "lock_recipe_sha256": preflight.lock.recipe_sha256,
            "repo_id": preflight.lock.repo_id,
            "revision": preflight.lock.revision,
        },
        "total_size_bytes": preflight.lock.total_size_bytes,
        "worker": worker_payload,
    }


def _provision_set_sha256(
    *,
    manifest_sha256: str,
    manifest_size_bytes: int,
    lock_sha256: str,
    lock_size_bytes: int,
    artifact_set_sha256: str,
    total_size_bytes: int,
) -> str:
    value = {
        "artifact_set_sha256": artifact_set_sha256,
        "lock_sha256": lock_sha256,
        "lock_size_bytes": lock_size_bytes,
        "manifest_sha256": manifest_sha256,
        "manifest_size_bytes": manifest_size_bytes,
        "total_size_bytes": total_size_bytes,
    }
    return hashlib.sha256(_PROVISION_SET_DOMAIN + _canonical_json(value)).hexdigest()


def _receipt_payload(
    preflight: MobileZipformerPreflight,
    retained_lock: BoundFile,
    manifest: BoundFile,
) -> dict[str, object]:
    return {
        "adapter": ADAPTER,
        "artifact_set_sha256": preflight.artifact_set_sha256,
        "complete": True,
        "kind": RECEIPT_KIND,
        "lock": {
            "filename": LOCK_COPY_FILENAME,
            "sha256": retained_lock.sha256,
            "size_bytes": retained_lock.size_bytes,
        },
        "manifest": {
            "filename": MANIFEST_FILENAME,
            "sha256": manifest.sha256,
            "size_bytes": manifest.size_bytes,
        },
        "model_id": MODEL_ID,
        "provision_set_sha256": _provision_set_sha256(
            manifest_sha256=manifest.sha256,
            manifest_size_bytes=manifest.size_bytes,
            lock_sha256=retained_lock.sha256,
            lock_size_bytes=retained_lock.size_bytes,
            artifact_set_sha256=preflight.artifact_set_sha256,
            total_size_bytes=preflight.lock.total_size_bytes,
        ),
        "schema_version": 1,
        "total_size_bytes": preflight.lock.total_size_bytes,
    }


def _verify_output_layout(
    root: Path,
    lock: SourceLock,
    *,
    terminal: bool,
) -> None:
    expected_root = {LOCK_COPY_FILENAME, MANIFEST_FILENAME, MODEL_DIRECTORY}
    if terminal:
        expected_root.add(RECEIPT_FILENAME)
    try:
        selected_root, _root_identity = _private_directory(
            root,
            expected_names=expected_root,
        )
        selected_model, _model_identity = _private_directory(
            root / MODEL_DIRECTORY,
            expected_names={item.filename for item in lock.artifacts},
        )
        if selected_root != root or selected_model != root / MODEL_DIRECTORY:
            _fail()
        identities: set[tuple[int, int]] = set()
        for path, expected_size in (
            (root / LOCK_COPY_FILENAME, len(lock.raw)),
            (root / MANIFEST_FILENAME, None),
            (root / RECEIPT_FILENAME, None if terminal else -1),
        ):
            if expected_size == -1:
                continue
            metadata = path.lstat()
            if (
                path.resolve(strict=True) != path
                or not _private_regular(
                    metadata,
                    size_bytes=(
                        metadata.st_size if expected_size is None else expected_size
                    ),
                )
                or metadata.st_size <= 0
                or metadata.st_size > _MAX_METADATA_BYTES
                or (metadata.st_dev, metadata.st_ino) in identities
            ):
                _fail()
            identities.add((metadata.st_dev, metadata.st_ino))
        for spec in lock.artifacts:
            metadata = (root / MODEL_DIRECTORY / spec.filename).lstat()
            if (
                not _private_regular(metadata, size_bytes=spec.size_bytes)
                or (metadata.st_dev, metadata.st_ino) in identities
            ):
                _fail()
            identities.add((metadata.st_dev, metadata.st_ino))
    except MobileZipformerProvisionError:
        raise
    except (OSError, RuntimeError, ValueError):
        _fail()


def _stage_terminal_receipt(
    root_descriptor: int,
    raw: bytes,
) -> tuple[int, _StagedReceipt]:
    descriptor = -1
    prepared = False
    try:
        if not raw or len(raw) > _MAX_METADATA_BYTES:
            _fail()
        try:
            os.stat(
                RECEIPT_FILENAME,
                dir_fd=root_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            _fail()
        temporary = getattr(os, "O_TMPFILE", 0)
        if not temporary:
            _fail()
        descriptor = os.open(
            ".",
            os.O_RDWR | temporary | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=root_descriptor,
        )
        os.fchmod(descriptor, 0o600)
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                _fail()
            view = view[written:]
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 0
            or metadata.st_size != len(raw)
            or not _owned(metadata)
        ):
            _fail()
        staged = _StagedReceipt(
            size_bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
            identity=_identity(metadata),
        )
        _verify_staged_receipt(descriptor, staged, raw)
        prepared = True
        return descriptor, staged
    except MobileZipformerProvisionError:
        raise
    except (OSError, RuntimeError, ValueError):
        _fail()
    finally:
        if descriptor >= 0 and not prepared:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _verify_staged_receipt(
    descriptor: int,
    staged: _StagedReceipt,
    expected: bytes,
) -> None:
    try:
        before = os.fstat(descriptor)
        if (
            _identity(before) != staged.identity
            or before.st_nlink != 0
            or staged.size_bytes != len(expected)
            or staged.sha256 != hashlib.sha256(expected).hexdigest()
        ):
            _fail()
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        consumed = 0
        while consumed <= staged.size_bytes:
            chunk = os.read(
                descriptor,
                min(65_536, staged.size_bytes + 1 - consumed),
            )
            if not chunk:
                break
            chunks.append(chunk)
            consumed += len(chunk)
        if (
            consumed != staged.size_bytes
            or b"".join(chunks) != expected
            or _identity(os.fstat(descriptor)) != staged.identity
        ):
            _fail()
    except MobileZipformerProvisionError:
        raise
    except (OSError, RuntimeError, ValueError):
        _fail()


def _commit_signal_numbers() -> set[int]:
    values = (
        getattr(signal, "SIGHUP", None),
        getattr(signal, "SIGINT", None),
        getattr(signal, "SIGTERM", None),
    )
    if any(not isinstance(value, int) for value in values):
        _fail()
    numbers = {int(value) for value in values}
    if len(numbers) != len(values):
        _fail()
    return numbers


def _commit_terminal_receipt(
    root_descriptor: int,
    descriptor: int,
    staged: _StagedReceipt,
    raw: bytes,
    *,
    state: _TerminalCommitState,
    commit_guard: Callable[[], None],
) -> None:
    if not hasattr(signal, "pthread_sigmask") or state.committed:
        _fail()
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, _commit_signal_numbers())
    expected_identity: tuple[int, ...] | None = None
    try:
        commit_guard()
        _verify_staged_receipt(descriptor, staged, raw)
        opened = os.fstat(descriptor)
        expected_identity = (
            opened.st_dev,
            opened.st_ino,
            stat.S_IFMT(opened.st_mode),
            stat.S_IMODE(opened.st_mode),
            opened.st_uid,
            opened.st_size,
        )
        os.link(
            f"/proc/self/fd/{descriptor}",
            RECEIPT_FILENAME,
            dst_dir_fd=root_descriptor,
            follow_symlinks=True,
        )
        state.committed = True
        published = os.stat(
            RECEIPT_FILENAME,
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )
        if (
            not _private_regular(published, size_bytes=staged.size_bytes)
            or (
                published.st_dev,
                published.st_ino,
                stat.S_IFMT(published.st_mode),
                stat.S_IMODE(published.st_mode),
                published.st_uid,
                published.st_size,
            )
            != expected_identity
        ):
            _fail()
        os.fsync(root_descriptor)
    except MobileZipformerProvisionError:
        raise
    except (OSError, RuntimeError, ValueError):
        _fail()
    finally:
        if not state.committed and expected_identity is not None:
            try:
                opened = os.fstat(descriptor)
                published = os.stat(
                    RECEIPT_FILENAME,
                    dir_fd=root_descriptor,
                    follow_symlinks=False,
                )
            except OSError:
                pass
            else:
                current_identity = (
                    published.st_dev,
                    published.st_ino,
                    stat.S_IFMT(published.st_mode),
                    stat.S_IMODE(published.st_mode),
                    published.st_uid,
                    published.st_size,
                )
                if (
                    opened.st_nlink == 1
                    and published.st_nlink == 1
                    and current_identity == expected_identity
                ):
                    state.committed = True
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def _read_private(
    path: Path,
    *,
    maximum_bytes: int,
    expected_bytes: int | None = None,
) -> bytes:
    try:
        snapshot = read_regular_bounded(
            path,
            maximum_bytes=maximum_bytes,
            expected_bytes=expected_bytes,
        )
        metadata = path.lstat()
        if snapshot.path != path or not _private_regular(
            metadata, size_bytes=len(snapshot.data)
        ):
            _fail()
        return snapshot.data
    except MobileZipformerProvisionError:
        raise
    except (BoundedReadError, OSError, RuntimeError, ValueError):
        _fail()


def _bound_from_payload(
    value: object,
    *,
    expected_path: Path | None,
    maximum_bytes: int,
    allow_final_symlink: bool,
) -> BoundFile:
    mapping = _exact_dict(value, {"path", "sha256", "size_bytes"})
    path = _absolute(_exact_string(mapping["path"]))
    expected_sha256 = _exact_sha256(mapping["sha256"])
    expected_size = _exact_positive_int(mapping["size_bytes"])
    if expected_path is not None and path != expected_path:
        _fail()
    current = _bind_file(
        path,
        maximum_bytes=maximum_bytes,
        allow_final_symlink=allow_final_symlink,
    )
    if current.sha256 != expected_sha256 or current.size_bytes != expected_size:
        _fail()
    return current


def _root_from_input(path: Path | str) -> Path:
    candidate = _absolute(path)
    try:
        metadata = candidate.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            return candidate
        if (
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_nlink == 1
            and candidate.name == MANIFEST_FILENAME
        ):
            return candidate.parent
        _fail()
    except MobileZipformerProvisionError:
        raise
    except (OSError, RuntimeError, ValueError):
        _fail()


def _parse_receipt(
    raw: bytes,
    *,
    lock: SourceLock,
    manifest_raw: bytes,
    artifact_set_sha256: str,
) -> None:
    value = _exact_dict(
        _strict_json(raw),
        {
            "adapter",
            "artifact_set_sha256",
            "complete",
            "kind",
            "lock",
            "manifest",
            "model_id",
            "provision_set_sha256",
            "schema_version",
            "total_size_bytes",
        },
    )
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["kind"] != RECEIPT_KIND
        or value["model_id"] != MODEL_ID
        or value["adapter"] != ADAPTER
        or type(value["complete"]) is not bool
        or not value["complete"]
        or value["artifact_set_sha256"] != artifact_set_sha256
        or value["total_size_bytes"] != lock.total_size_bytes
    ):
        _fail()
    lock_row = _exact_dict(
        value["lock"],
        {"filename", "sha256", "size_bytes"},
    )
    manifest_row = _exact_dict(
        value["manifest"],
        {"filename", "sha256", "size_bytes"},
    )
    manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    if (
        lock_row["filename"] != LOCK_COPY_FILENAME
        or lock_row["sha256"] != lock.raw_sha256
        or lock_row["size_bytes"] != len(lock.raw)
        or manifest_row["filename"] != MANIFEST_FILENAME
        or manifest_row["sha256"] != manifest_sha256
        or manifest_row["size_bytes"] != len(manifest_raw)
    ):
        _fail()
    expected_provision_set = _provision_set_sha256(
        manifest_sha256=manifest_sha256,
        manifest_size_bytes=len(manifest_raw),
        lock_sha256=lock.raw_sha256,
        lock_size_bytes=len(lock.raw),
        artifact_set_sha256=artifact_set_sha256,
        total_size_bytes=lock.total_size_bytes,
    )
    if value["provision_set_sha256"] != expected_provision_set:
        _fail()


def _parse_manifest(
    raw: bytes,
    *,
    root: Path,
    lock: SourceLock,
) -> tuple[
    BoundFile,
    BoundFile,
    tuple[BoundArtifact, ...],
    MobileZipformerConfig,
    str,
]:
    value = _exact_dict(
        _strict_json(raw),
        {
            "adapter",
            "artifact_set_sha256",
            "artifacts",
            "evidence_scope",
            "kind",
            "limits",
            "mobile_config",
            "model_id",
            "python",
            "runtime",
            "schema_version",
            "source",
            "total_size_bytes",
            "worker",
        },
    )
    artifact_set = _artifact_set_sha256(lock.artifacts)
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 10
        or value["kind"] != MANIFEST_KIND
        or value["model_id"] != MODEL_ID
        or value["adapter"] != ADAPTER
        or value["artifact_set_sha256"] != artifact_set
        or value["total_size_bytes"] != lock.total_size_bytes
    ):
        _fail()
    config = _mobile_config(value["mobile_config"])
    if config != lock.mobile_config:
        _fail()
    evidence = _exact_dict(value["evidence_scope"], set(_EXPECTED_EVIDENCE_SCOPE))
    if any(
        type(evidence[key]) is not bool or evidence[key] is not expected
        for key, expected in _EXPECTED_EVIDENCE_SCOPE.items()
    ):
        _fail()
    limits = _exact_dict(
        value["limits"],
        {"case_timeout_sec", "startup_timeout_sec"},
    )
    if (
        type(limits["startup_timeout_sec"]) is not float
        or limits["startup_timeout_sec"] != 120.0
        or type(limits["case_timeout_sec"]) is not float
        or limits["case_timeout_sec"] != 300.0
    ):
        _fail()

    python = _bound_from_payload(
        value["python"],
        expected_path=None,
        maximum_bytes=MAX_PYTHON_BYTES,
        allow_final_symlink=True,
    )
    worker = _bound_from_payload(
        value["worker"],
        expected_path=_FIXED_WORKER,
        maximum_bytes=MAX_WORKER_BYTES,
        allow_final_symlink=False,
    )
    runtime = _exact_dict(
        value["runtime"],
        {
            "distributions",
            "metadata_only_verified",
            "model_loaded",
            "packages_imported",
            "python",
            "worker",
        },
    )
    distributions = _exact_dict(
        runtime["distributions"],
        set(_EXPECTED_RUNTIME_DISTRIBUTIONS),
    )
    if (
        dict(distributions) != dict(lock.runtime_distributions)
        or runtime["metadata_only_verified"] is not True
        or runtime["packages_imported"] is not False
        or runtime["model_loaded"] is not False
        or runtime["python"] != value["python"]
        or runtime["worker"] != value["worker"]
    ):
        _fail()

    source = _exact_dict(
        value["source"],
        {"lock", "lock_recipe_sha256", "repo_id", "revision"},
    )
    if (
        source["repo_id"] != lock.repo_id
        or source["revision"] != lock.revision
        or source["lock_recipe_sha256"] != lock.recipe_sha256
    ):
        _fail()
    retained_lock = _bound_from_payload(
        source["lock"],
        expected_path=root / LOCK_COPY_FILENAME,
        maximum_bytes=len(lock.raw),
        allow_final_symlink=False,
    )
    if retained_lock.sha256 != lock.raw_sha256 or retained_lock.size_bytes != len(
        lock.raw
    ):
        _fail()

    rows = value["artifacts"]
    if type(rows) is not list or len(rows) != len(lock.artifacts):
        _fail()
    artifacts: list[BoundArtifact] = []
    for row, spec in zip(rows, lock.artifacts, strict=True):
        mapping = _exact_dict(row, {"name", "path", "sha256", "size_bytes"})
        if mapping["name"] != spec.name:
            _fail()
        expected_path = root / MODEL_DIRECTORY / spec.filename
        path = _absolute(_exact_string(mapping["path"]))
        if (
            path != expected_path
            or mapping["sha256"] != spec.sha256
            or mapping["size_bytes"] != spec.size_bytes
        ):
            _fail()
        bound = _bind_private_artifact(path, spec)
        artifacts.append(
            BoundArtifact(
                name=spec.name,
                path=bound.path,
                sha256=spec.sha256,
                size_bytes=spec.size_bytes,
            )
        )
    return python, worker, tuple(artifacts), config, artifact_set


def _load_provision_with_lock(
    path: Path | str,
    *,
    lock: SourceLock,
) -> MobileZipformerProvision:
    root = _root_from_input(path)
    _verify_output_layout(root, lock, terminal=True)
    lock_raw = _read_private(
        root / LOCK_COPY_FILENAME,
        maximum_bytes=len(lock.raw),
        expected_bytes=len(lock.raw),
    )
    if lock_raw != lock.raw or hashlib.sha256(lock_raw).hexdigest() != lock.raw_sha256:
        _fail()
    manifest_raw = _read_private(
        root / MANIFEST_FILENAME,
        maximum_bytes=_MAX_METADATA_BYTES,
    )
    receipt_raw = _read_private(
        root / RECEIPT_FILENAME,
        maximum_bytes=_MAX_METADATA_BYTES,
    )
    python, worker, artifacts, config, artifact_set = _parse_manifest(
        manifest_raw,
        root=root,
        lock=lock,
    )
    _parse_receipt(
        receipt_raw,
        lock=lock,
        manifest_raw=manifest_raw,
        artifact_set_sha256=artifact_set,
    )

    _verify_output_layout(root, lock, terminal=True)
    if (
        _read_private(
            root / LOCK_COPY_FILENAME,
            maximum_bytes=len(lock.raw),
            expected_bytes=len(lock.raw),
        )
        != lock_raw
        or _read_private(
            root / MANIFEST_FILENAME,
            maximum_bytes=_MAX_METADATA_BYTES,
            expected_bytes=len(manifest_raw),
        )
        != manifest_raw
        or _read_private(
            root / RECEIPT_FILENAME,
            maximum_bytes=_MAX_METADATA_BYTES,
            expected_bytes=len(receipt_raw),
        )
        != receipt_raw
    ):
        _fail()
    for artifact, spec in zip(artifacts, lock.artifacts, strict=True):
        current = _bind_private_artifact(artifact.path, spec)
        if current.path != artifact.path:
            _fail()
    current_python = _bind_file(
        python.path,
        maximum_bytes=MAX_PYTHON_BYTES,
        allow_final_symlink=True,
    )
    current_worker = _bind_file(
        worker.path,
        maximum_bytes=MAX_WORKER_BYTES,
        allow_final_symlink=False,
    )
    if not _same_bound_file(current_python, python) or not _same_bound_file(
        current_worker,
        worker,
    ):
        _fail()
    return MobileZipformerProvision(
        path=root / MANIFEST_FILENAME,
        digest=hashlib.sha256(manifest_raw).hexdigest(),
        receipt_digest=hashlib.sha256(receipt_raw).hexdigest(),
        model_id=MODEL_ID,
        adapter=ADAPTER,
        python=python,
        worker=worker,
        artifacts=artifacts,
        mobile_config=config,
        artifact_set_sha256=artifact_set,
        total_size_bytes=lock.total_size_bytes,
    )


def load_mobile_zipformer_provision(
    path: Path | str,
) -> MobileZipformerProvision:
    """Strictly reopen one complete terminal-receipt-bound provision."""

    try:
        return _load_provision_with_lock(path, lock=_load_source_lock())
    except MobileZipformerProvisionError:
        raise
    except (BoundedReadError, OSError, RuntimeError, TypeError, ValueError):
        _fail()


def _precommit_guard(
    *,
    root: Path,
    preflight: MobileZipformerPreflight,
    manifest_raw: bytes,
) -> None:
    _revalidate_preflight(preflight)
    _verify_output_layout(root, preflight.lock, terminal=False)
    retained_lock = _read_private(
        root / LOCK_COPY_FILENAME,
        maximum_bytes=len(preflight.lock.raw),
        expected_bytes=len(preflight.lock.raw),
    )
    retained_manifest = _read_private(
        root / MANIFEST_FILENAME,
        maximum_bytes=_MAX_METADATA_BYTES,
        expected_bytes=len(manifest_raw),
    )
    if retained_lock != preflight.lock.raw or retained_manifest != manifest_raw:
        _fail()
    for spec in preflight.lock.artifacts:
        _bind_private_artifact(root / MODEL_DIRECTORY / spec.filename, spec)


def _publish_preflight(
    *,
    preflight: MobileZipformerPreflight,
    output_dir: Path | str,
) -> MobileZipformerProvision:
    _revalidate_preflight(preflight)
    root = _new_private_output(
        output_dir,
        forbidden=(preflight.source_root, preflight.lock.path),
    )
    staged_descriptor = -1
    state = _TerminalCommitState()
    try:
        model_directory = _mkdir_private(root, MODEL_DIRECTORY)
        retained_lock = _write_new_private(
            root,
            LOCK_COPY_FILENAME,
            preflight.lock.raw,
        )
        artifacts = tuple(
            _copy_artifact(source, model_directory)
            for source in preflight.source_artifacts
        )
        manifest_raw = (
            _canonical_json(_manifest_payload(preflight, retained_lock, artifacts))
            + b"\n"
        )
        manifest = _write_new_private(root, MANIFEST_FILENAME, manifest_raw)
        receipt_raw = (
            _canonical_json(_receipt_payload(preflight, retained_lock, manifest))
            + b"\n"
        )
        _precommit_guard(
            root=root,
            preflight=preflight,
            manifest_raw=manifest_raw,
        )
        with opened_directory_nofollow(
            root,
            require_private=True,
        ) as (stable, root_descriptor):
            if stable != root:
                _fail()
            staged_descriptor, staged = _stage_terminal_receipt(
                root_descriptor,
                receipt_raw,
            )
            _commit_terminal_receipt(
                root_descriptor,
                staged_descriptor,
                staged,
                receipt_raw,
                state=state,
                commit_guard=lambda: _precommit_guard(
                    root=root,
                    preflight=preflight,
                    manifest_raw=manifest_raw,
                ),
            )
            if not state.committed:
                _fail()
        _verify_output_layout(root, preflight.lock, terminal=True)
        provision = _load_provision_with_lock(root, lock=preflight.lock)
        if (
            provision.digest != manifest.sha256
            or provision.artifact_set_sha256 != preflight.artifact_set_sha256
        ):
            _fail()
        return provision
    except BaseException as error:
        if state.committed:
            try:
                recovered = _load_provision_with_lock(root, lock=preflight.lock)
            except Exception:
                if isinstance(error, Exception):
                    _fail()
                raise error
            return recovered
        if isinstance(error, MobileZipformerProvisionError):
            raise
        if isinstance(error, (GeneratorExit, KeyboardInterrupt, SystemExit)):
            raise
        _fail()
    finally:
        if staged_descriptor >= 0:
            try:
                os.close(staged_descriptor)
            except OSError:
                pass


def provision_mobile_zipformer(
    *,
    python: Path | str,
    model_root: Path | str,
    output_dir: Path | str,
) -> MobileZipformerProvision:
    """Preflight and publish one exact private, self-contained provision."""

    try:
        preflight = preflight_mobile_zipformer(
            python=python,
            model_root=model_root,
        )
        return _publish_preflight(
            preflight=preflight,
            output_dir=output_dir,
        )
    except MobileZipformerProvisionError:
        raise
    except (BoundedReadError, OSError, RuntimeError, TypeError, ValueError):
        _fail()


def _safe_preflight_result(
    preflight: MobileZipformerPreflight,
) -> dict[str, object]:
    return {
        "artifact_set_sha256": preflight.artifact_set_sha256,
        "metadata_only_runtime_verified": True,
        "model_id": MODEL_ID,
        "ok": True,
        "ready_to_publish": True,
        "total_size_bytes": preflight.lock.total_size_bytes,
    }


def _safe_provision_result(
    provision: MobileZipformerProvision,
) -> dict[str, object]:
    return {
        "artifact_set_sha256": provision.artifact_set_sha256,
        "manifest_sha256": provision.digest,
        "model_id": provision.model_id,
        "ok": True,
        "receipt_sha256": provision.receipt_digest,
        "total_size_bytes": provision.total_size_bytes,
    }


def _parse_cli(argv: Sequence[str]) -> _ParsedCommand | None:
    if list(argv) in (["--help"], ["help"]):
        return None
    if not argv or argv[0] not in {"preflight", "publish"}:
        _fail()
    action = argv[0]
    tail = list(argv[1:])
    if len(tail) % 2:
        _fail()
    allowed = {"--python", "--model-root"}
    if action == "publish":
        allowed.add("--output-dir")
    values: dict[str, str] = {}
    for index in range(0, len(tail), 2):
        flag, value = tail[index : index + 2]
        if (
            flag not in allowed
            or flag in values
            or not value
            or "\x00" in value
            or value.startswith("--")
        ):
            _fail()
        values[flag] = value
    if set(values) != allowed:
        _fail()
    return _ParsedCommand(
        action=action,
        python=values["--python"],
        model_root=values["--model-root"],
        output_dir=values.get("--output-dir"),
    )


def _help_result() -> dict[str, object]:
    return {
        "actions": ["preflight", "publish"],
        "common_flags": ["--python", "--model-root"],
        "kind": "mobile-zipformer-provision-help-v1",
        "ok": True,
        "publish_only_flag": "--output-dir",
    }


def _emit(value: object) -> None:
    sys.stdout.buffer.write(_canonical_json(value) + b"\n")
    sys.stdout.buffer.flush()


def main(argv: Sequence[str] | None = None) -> int:
    """Run bounded path-free preflight or publication."""

    try:
        command = _parse_cli(tuple(sys.argv[1:] if argv is None else argv))
        if command is None:
            _emit(_help_result())
            return 0
        if command.action == "preflight":
            result = _safe_preflight_result(
                preflight_mobile_zipformer(
                    python=command.python,
                    model_root=command.model_root,
                )
            )
        else:
            if command.output_dir is None:
                _fail()
            result = _safe_provision_result(
                provision_mobile_zipformer(
                    python=command.python,
                    model_root=command.model_root,
                    output_dir=command.output_dir,
                )
            )
        _emit(result)
        return 0
    except Exception:
        _emit(_SAFE_ERROR)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
