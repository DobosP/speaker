"""Provision the exact mobile Whisper tuple without importing model packages.

Only pre-existing owner-private bytes are accepted. The command performs no
download, package import, ONNX load, GPU use, or audio-device access. Runtime
versions are verified by reading distribution metadata in an isolated child.
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
    MAX_MOBILE_WHISPER_ARTIFACT_BYTES,
    MAX_PYTHON_BYTES,
    MAX_WORKER_BYTES,
    MOBILE_WHISPER_ADAPTER,
    MOBILE_WHISPER_ARTIFACT_SET_SHA256,
    MOBILE_WHISPER_ARTIFACT_SPECS,
    MOBILE_WHISPER_TOTAL_SIZE_BYTES,
    BoundArtifact,
    BoundFile,
    MobileWhisperConfig,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCK_PATH = _REPO_ROOT / "tools/streaming_stt/mobile-whisper-base-en-v1.lock.json"
_FIXED_WORKER = _REPO_ROOT / "tools/streaming_stt/worker.py"

LOCK_RAW_SHA256 = "0fac7fee4e4b73f29b23130e1ee449501c4324211bae9e5da888b10483a3e496"
LOCK_SIZE_BYTES = 3_314
LOCK_RECIPE_SHA256 = "aeb7b5bbba9c558120bf68ac7a9d7dca671e75115c2f42d8897bfa31153b6087"
SOURCE_REVISION = "59eea950fc76df2453efb57e6c0fd334548e8ffe"
MODEL_ID = "sherpa-onnx-whisper-base.en-mobile-int8-v1"
ADAPTER = MOBILE_WHISPER_ADAPTER
MANIFEST_KIND = "mobile-whisper-provision-v1"
RECEIPT_KIND = "mobile-whisper-provision-receipt-v1"
MANIFEST_FILENAME = "worker-manifest.json"
RECEIPT_FILENAME = "provision-receipt.json"
LOCK_COPY_FILENAME = "source-lock.json"
MODEL_DIRECTORY = "model"
TOTAL_SIZE_BYTES = MOBILE_WHISPER_TOTAL_SIZE_BYTES

_MAX_METADATA_BYTES = 256 * 1024
_COPY_CHUNK_BYTES = 1024 * 1024
_ARTIFACT_SET_DOMAIN = b"speaker-mobile-whisper-artifact-set-v1\0"
_LOCK_RECIPE_DOMAIN = b"speaker-mobile-whisper-source-lock-recipe-v1\0"
_PROVISION_SET_DOMAIN = b"speaker-mobile-whisper-provision-set-v1\0"
_EXPECTED_RUNTIME_DISTRIBUTIONS = {
    "numpy": "2.4.6",
    "sherpa-onnx": "1.13.3",
    "sherpa-onnx-core": "1.13.3",
}
_EXPECTED_EVIDENCE_SCOPE = {
    "downloaded": False,
    "packages_imported": False,
    "model_loaded": False,
    "model_executed": False,
    "gpu_used": False,
    "audio_device_opened": False,
    "evaluation_result_authority": False,
    "mobile_device_evidence": False,
}
_EXPECTED_SOURCE = {
    "artifact_license_status": "unverified",
    "conversion_license_artifact_present": False,
    "conversion_origin_revision_status": "unverified",
    "exact_conversion_recipe_status": "unavailable",
    "repo_id": "csukuangfj/sherpa-onnx-whisper-base.en",
    "revision": SOURCE_REVISION,
    "upstream_base_en_checkpoint_sha256": (
        "25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead"
    ),
    "upstream_code_license": "MIT",
    "upstream_license_sha256": (
        "b5d65a59060e68c4ff940e1eddfa6f94b2d68fdf58ed7f4dd57721c997e35e9d"
    ),
    "upstream_license_size_bytes": 1_063,
    "upstream_license_url": (
        "https://raw.githubusercontent.com/openai/whisper/"
        "5f86d1d86363843179951550570367b37c5d6f78/LICENSE"
    ),
    "upstream_model_map_sha256": (
        "79ecbed714783b6230931a3dba6d5da6d386e2d44f9fc27256e4fc72751e1f62"
    ),
    "upstream_model_map_size_bytes": 7_432,
    "upstream_model_map_url": (
        "https://raw.githubusercontent.com/openai/whisper/"
        "5f86d1d86363843179951550570367b37c5d6f78/whisper/__init__.py"
    ),
    "upstream_project": "openai/whisper",
    "upstream_evidence_revision": "5f86d1d86363843179951550570367b37c5d6f78",
    "upstream_evidence_scope": "license-and-base.en-checkpoint-map-only",
}
_SAFE_ERROR = {"ok": False, "error": "mobile_whisper_prerequisites_unavailable"}


class MobileWhisperProvisionError(RuntimeError):
    """A detail-free unsafe, incomplete, or changed provision input."""


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    filename: str
    precision: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class SourceLock:
    path: Path
    raw: bytes
    raw_sha256: str
    recipe_sha256: str
    artifacts: tuple[ArtifactSpec, ...]
    runtime_distributions: Mapping[str, str]
    mobile_config: MobileWhisperConfig
    source: Mapping[str, object]
    total_size_bytes: int


@dataclass(frozen=True)
class _SourceArtifact:
    spec: ArtifactSpec
    path: Path
    identity: tuple[int, ...]


@dataclass(frozen=True)
class MobileWhisperPreflight:
    lock: SourceLock
    python: BoundFile
    worker: BoundFile
    runtime_root: Path
    runtime_root_identity: tuple[int, ...]
    source_root: Path
    source_root_identity: tuple[int, ...]
    source_artifacts: tuple[_SourceArtifact, ...]
    artifact_set_sha256: str


@dataclass(frozen=True)
class MobileWhisperProvision:
    path: Path
    digest: str
    receipt_digest: str
    model_id: str
    adapter: str
    python: BoundFile
    worker: BoundFile
    artifacts: tuple[BoundArtifact, ...]
    mobile_config: MobileWhisperConfig
    artifact_set_sha256: str
    total_size_bytes: int


@dataclass(frozen=True)
class _StagedReceipt:
    size_bytes: int
    sha256: str
    identity: tuple[int, ...]


@dataclass(frozen=True)
class _BoundOutputRoot:
    path: Path
    identity: tuple[int, ...]


@dataclass
class _TerminalCommitState:
    linked: bool = False
    committed: bool = False


@dataclass(frozen=True)
class _ParsedCommand:
    action: str
    python: str
    model_root: str
    output_dir: str | None


def _fail() -> None:
    raise MobileWhisperProvisionError()


def _bad() -> object:
    _fail()


def _strict_json(raw: bytes) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                _fail()
            result[key] = value
        return result

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: _bad())
    except (UnicodeError, ValueError, OverflowError, MobileWhisperProvisionError):
        _fail()


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


def _exact_dict(value: object, fields: set[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        _fail()
    return value


def _exact_string(value: object) -> str:
    if type(value) is not str or not value or "\x00" in value:
        _fail()
    return value


def _exact_sha256(value: object) -> str:
    text = _exact_string(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        _fail()
    return text


def _exact_positive_int(value: object) -> int:
    if type(value) is not int or value <= 0:
        _fail()
    return value


def _safe_leaf(value: object) -> str:
    text = _exact_string(value)
    if text in {".", ".."} or Path(text).name != text or "/" in text or "\\" in text:
        _fail()
    return text


def _absolute(value: Path | str) -> Path:
    try:
        text = os.fspath(value)
        if not isinstance(text, str) or not text or "\x00" in text:
            _fail()
        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            _fail()
        return Path(os.path.abspath(candidate))
    except (OSError, TypeError, ValueError):
        _fail()


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


def _directory_object_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        stat.S_IMODE(metadata.st_mode),
        metadata.st_uid,
        metadata.st_gid,
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


def _venv_root_from_python(path: Path) -> tuple[Path, tuple[int, ...]]:
    """Bind the lexical venv root for an exact ``venv/bin/python`` path."""

    candidate = _absolute(path)
    binary_root = candidate.parent
    runtime_root = binary_root.parent
    if (
        candidate.name != "python"
        or binary_root.name != "bin"
        or runtime_root == runtime_root.parent
        or candidate != runtime_root / "bin" / "python"
    ):
        _fail()
    try:
        with opened_directory_nofollow(runtime_root) as (stable, descriptor):
            metadata = os.fstat(descriptor)
            lexical = runtime_root.lstat()
            if (
                stable != runtime_root
                or not stat.S_ISDIR(metadata.st_mode)
                or not _owned(metadata)
                or _directory_identity(metadata) != _directory_identity(lexical)
            ):
                _fail()
            identity = _directory_identity(metadata)
        with opened_directory_nofollow(binary_root) as (stable, descriptor):
            metadata = os.fstat(descriptor)
            if (
                stable != binary_root
                or not stat.S_ISDIR(metadata.st_mode)
                or not _owned(metadata)
            ):
                _fail()
        return runtime_root, identity
    except MobileWhisperProvisionError:
        raise
    except (BoundedReadError, OSError, RuntimeError, ValueError):
        _fail()


def _validate_preflight_domains(
    *,
    source_root: Path,
    runtime_root: Path,
    lock_path: Path,
    worker_path: Path,
) -> None:
    protected = (lock_path, worker_path, _REPO_ROOT)
    if (
        _overlaps(source_root, runtime_root)
        or any(_overlaps(source_root, path) for path in protected)
        or any(_overlaps(runtime_root, path) for path in protected)
    ):
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
            metadata = os.fstat(descriptor)
            lexical = candidate.lstat()
            if (
                stable != candidate
                or not _private_directory_metadata(metadata)
                or _directory_identity(metadata) != _directory_identity(lexical)
                or (
                    expected_names is not None
                    and set(os.listdir(descriptor)) != expected_names
                )
            ):
                _fail()
            return stable, _directory_identity(metadata)
    except MobileWhisperProvisionError:
        raise
    except (BoundedReadError, OSError, RuntimeError, ValueError):
        _fail()


def _bind_file(path: Path, *, maximum_bytes: int, allow_final_symlink: bool) -> BoundFile:
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
        return BoundFile(candidate, digest.sha256, digest.size_bytes)
    except MobileWhisperProvisionError:
        raise
    except (BoundedReadError, OSError, RuntimeError, ValueError):
        _fail()


def _same_bound_file(left: BoundFile, right: BoundFile) -> bool:
    return left == right


def _bind_private_artifact(path: Path, spec: ArtifactSpec) -> _SourceArtifact:
    candidate = _absolute(path)
    try:
        before = candidate.lstat()
        if (
            candidate.resolve(strict=True) != candidate
            or not _private_regular(before, size_bytes=spec.size_bytes)
        ):
            _fail()
        digest = hash_regular_bounded(
            candidate,
            maximum_bytes=min(spec.size_bytes, MAX_MOBILE_WHISPER_ARTIFACT_BYTES),
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
        return _SourceArtifact(spec, candidate, _identity(after))
    except MobileWhisperProvisionError:
        raise
    except (BoundedReadError, OSError, RuntimeError, ValueError):
        _fail()


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
    except BoundedReadError:
        _fail()
    raw = snapshot.data
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
        or value["kind"] != "mobile-whisper-source-lock-v1"
        or value["model_id"] != MODEL_ID
        or value["total_size_bytes"] != TOTAL_SIZE_BYTES
    ):
        _fail()
    raw_source = _exact_dict(value["source"], set(_EXPECTED_SOURCE))
    if any(
        type(raw_source[name]) is not type(expected)
        or raw_source[name] != expected
        for name, expected in _EXPECTED_SOURCE.items()
    ):
        _fail()
    runtime = _exact_dict(
        value["runtime_distributions"],
        set(_EXPECTED_RUNTIME_DISTRIBUTIONS),
    )
    if any(
        type(runtime[name]) is not str or runtime[name] != expected
        for name, expected in _EXPECTED_RUNTIME_DISTRIBUTIONS.items()
    ):
        _fail()
    config_value = _exact_dict(
        value["mobile_config"],
        set(MobileWhisperConfig().as_dict()),
    )
    try:
        config = MobileWhisperConfig(**config_value)
    except (TypeError, ValueError, RuntimeError):
        _fail()
    rows = value["artifacts"]
    if type(rows) is not list or len(rows) != len(MOBILE_WHISPER_ARTIFACT_SPECS):
        _fail()
    artifacts: list[ArtifactSpec] = []
    for row, expected in zip(rows, MOBILE_WHISPER_ARTIFACT_SPECS, strict=True):
        item = _exact_dict(
            row,
            {"filename", "name", "precision", "sha256", "size_bytes"},
        )
        spec = ArtifactSpec(
            name=_safe_leaf(item["name"]),
            filename=_safe_leaf(item["filename"]),
            precision=_exact_string(item["precision"]),
            sha256=_exact_sha256(item["sha256"]),
            size_bytes=_exact_positive_int(item["size_bytes"]),
        )
        manifest_expected = (
            expected[0],
            expected[1],
            expected[2],
            expected[3],
            expected[4],
        )
        if (
            spec.name,
            spec.filename,
            spec.precision,
            spec.sha256,
            spec.size_bytes,
        ) != manifest_expected:
            _fail()
        artifacts.append(spec)
    recipe = _exact_sha256(value["recipe_sha256"])
    recipe_value = dict(value)
    del recipe_value["recipe_sha256"]
    calculated = hashlib.sha256(
        _LOCK_RECIPE_DOMAIN + _canonical_json(recipe_value)
    ).hexdigest()
    if recipe != LOCK_RECIPE_SHA256 or recipe != calculated:
        _fail()
    if (
        sum(item.size_bytes for item in artifacts) != TOTAL_SIZE_BYTES
        or _artifact_set_sha256(artifacts) != MOBILE_WHISPER_ARTIFACT_SET_SHA256
    ):
        _fail()
    return SourceLock(
        path=snapshot.path,
        raw=raw,
        raw_sha256=raw_sha256,
        recipe_sha256=recipe,
        artifacts=tuple(artifacts),
        runtime_distributions=MappingProxyType(dict(runtime)),
        mobile_config=config,
        source=MappingProxyType(dict(raw_source)),
        total_size_bytes=TOTAL_SIZE_BYTES,
    )


_RUNTIME_METADATA_SCRIPT = r'''\
import os
import stat
import sys

EXPECTED = (
    (b"numpy", "numpy", "2.4.6"),
    (b"sherpa-onnx", "sherpa_onnx", "1.13.3"),
    (b"sherpa-onnx-core", "sherpa_onnx_core", "1.13.3"),
)
MAXIMUM_BYTES = 131072
BLOCKED = {
    "_sherpa_onnx", "numpy", "sherpa_onnx", "sherpa_onnx.lib._sherpa_onnx",
    "sherpa_onnx_core", "site", "sitecustomize", "usercustomize",
}

def fail():
    raise SystemExit(2)

def blocked():
    return any(
        name == prefix or name.startswith(prefix + ".")
        for name in sys.modules for prefix in BLOCKED
    )

def identity(value):
    return (
        value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode), value.st_size,
        value.st_mtime_ns, value.st_ctime_ns,
    )

def normalized(value):
    result = bytearray()
    separator = False
    for character in value.lower():
        if character in b"-_.":
            if not separator:
                result.append(ord("-"))
            separator = True
        else:
            result.append(character)
            separator = False
    return bytes(result)

def read_metadata(root, directory_name):
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    root_fd = os.open(root, flags)
    directory_fd = -1
    file_fd = -1
    try:
        directory_fd = os.open(directory_name, flags, dir_fd=root_fd)
        file_fd = os.open(
            "METADATA", os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        before = os.fstat(file_fd)
        if (
            not stat.S_ISREG(before.st_mode) or before.st_size <= 0
            or before.st_size > MAXIMUM_BYTES
        ):
            fail()
        remaining = before.st_size
        chunks = []
        while remaining:
            chunk = os.read(file_fd, min(65536, remaining))
            if not chunk:
                fail()
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(file_fd, 1) or identity(os.fstat(file_fd)) != identity(before):
            fail()
        return b"".join(chunks)
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        if directory_fd >= 0:
            os.close(directory_fd)
        os.close(root_fd)

def verify(raw, expected_name, expected_version):
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
        len(names) != 1 or normalized(names[0]) != expected_name
        or versions != [expected_version.encode("ascii")]
    ):
        fail()

if not sys.flags.isolated or not sys.flags.no_site or blocked():
    fail()
root = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(sys.executable))),
    "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages",
)
try:
    for expected_name, stem, version in EXPECTED:
        verify(read_metadata(root, f"{stem}-{version}.dist-info"), expected_name, version)
except (OSError, ValueError):
    fail()
if blocked():
    fail()
'''


def _verify_runtime_versions(python: Path, expected: Mapping[str, str]) -> None:
    if dict(expected) != _EXPECTED_RUNTIME_DISTRIBUTIONS:
        _fail()
    environment = {
        "HOME": "/nonexistent",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
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


def _revalidate_preflight(preflight: MobileWhisperPreflight) -> None:
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
    runtime_root, runtime_root_identity = _venv_root_from_python(python.path)
    worker = _bind_file(
        preflight.worker.path,
        maximum_bytes=MAX_WORKER_BYTES,
        allow_final_symlink=False,
    )
    if not _same_bound_file(python, preflight.python) or not _same_bound_file(
        worker, preflight.worker
    ) or (
        runtime_root != preflight.runtime_root
        or runtime_root_identity != preflight.runtime_root_identity
    ):
        _fail()
    _validate_preflight_domains(
        source_root=root,
        runtime_root=runtime_root,
        lock_path=preflight.lock.path,
        worker_path=worker.path,
    )


def _preflight_with_lock(
    *,
    python: Path | str,
    model_root: Path | str,
    lock: SourceLock,
    version_probe: Callable[[Path, Mapping[str, str]], None],
) -> MobileWhisperPreflight:
    selected_python = _absolute(python)
    root, root_identity = _private_directory(
        model_root,
        expected_names={item.filename for item in lock.artifacts},
    )
    if _has_git_ancestor(root):
        _fail()
    bound_python = _bind_file(
        selected_python,
        maximum_bytes=MAX_PYTHON_BYTES,
        allow_final_symlink=True,
    )
    runtime_root, runtime_root_identity = _venv_root_from_python(
        bound_python.path
    )
    worker = _bind_file(
        _FIXED_WORKER,
        maximum_bytes=MAX_WORKER_BYTES,
        allow_final_symlink=False,
    )
    _validate_preflight_domains(
        source_root=root,
        runtime_root=runtime_root,
        lock_path=lock.path,
        worker_path=worker.path,
    )
    artifacts = tuple(
        _bind_private_artifact(root / spec.filename, spec)
        for spec in lock.artifacts
    )
    try:
        version_probe(selected_python, lock.runtime_distributions)
    except MobileWhisperProvisionError:
        raise
    except BaseException as error:
        if isinstance(error, (GeneratorExit, KeyboardInterrupt, SystemExit)):
            raise
        _fail()
    preflight = MobileWhisperPreflight(
        lock=lock,
        python=bound_python,
        worker=worker,
        runtime_root=runtime_root,
        runtime_root_identity=runtime_root_identity,
        source_root=root,
        source_root_identity=root_identity,
        source_artifacts=artifacts,
        artifact_set_sha256=_artifact_set_sha256(lock.artifacts),
    )
    _revalidate_preflight(preflight)
    return preflight


def preflight_mobile_whisper(
    *,
    python: Path | str,
    model_root: Path | str,
) -> MobileWhisperPreflight:
    """Bind exact local artifacts and metadata without importing the runtime."""

    try:
        lock = _load_source_lock()
        preflight = _preflight_with_lock(
            python=python,
            model_root=model_root,
            lock=lock,
            version_probe=_verify_runtime_versions,
        )
        if preflight.artifact_set_sha256 != MOBILE_WHISPER_ARTIFACT_SET_SHA256:
            _fail()
        return preflight
    except MobileWhisperProvisionError:
        raise
    except (BoundedReadError, OSError, RuntimeError, TypeError, ValueError):
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
                final_descriptor,
                size_bytes=len(payload),
            ) or _identity(final_named) != _identity(final_descriptor):
                _fail()
            os.fsync(parent_descriptor)
        return BoundFile(
            path=parent / leaf,
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
        )
    except MobileWhisperProvisionError:
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
            if _identity(os.fstat(source_descriptor)) != source.identity:
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
            if not _private_regular(os.fstat(destination_descriptor), size_bytes=0):
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
            path=bound.path,
            sha256=source.spec.sha256,
            size_bytes=source.spec.size_bytes,
            name=source.spec.name,
        )
    except MobileWhisperProvisionError:
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
    return {"name": item.name, **_bound_payload(item)}


def _manifest_payload(
    preflight: MobileWhisperPreflight,
    *,
    output_root: Path,
    retained_lock: BoundFile,
) -> dict[str, object]:
    python_payload = _bound_payload(preflight.python)
    worker_payload = _bound_payload(preflight.worker)
    artifacts = tuple(
        BoundArtifact(
            path=output_root / MODEL_DIRECTORY / item.spec.filename,
            sha256=item.spec.sha256,
            size_bytes=item.spec.size_bytes,
            name=item.spec.name,
        )
        for item in preflight.source_artifacts
    )
    source = dict(preflight.lock.source)
    source.update(
        {
            "lock": _bound_payload(retained_lock),
            "lock_recipe_sha256": preflight.lock.recipe_sha256,
        }
    )
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
        "schema_version": 11,
        "source": source,
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
    preflight: MobileWhisperPreflight,
    *,
    manifest_raw: bytes,
) -> dict[str, object]:
    manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    return {
        "adapter": ADAPTER,
        "artifact_set_sha256": preflight.artifact_set_sha256,
        "complete": True,
        "kind": RECEIPT_KIND,
        "lock": {
            "filename": LOCK_COPY_FILENAME,
            "sha256": preflight.lock.raw_sha256,
            "size_bytes": len(preflight.lock.raw),
        },
        "manifest": {
            "filename": MANIFEST_FILENAME,
            "sha256": manifest_sha256,
            "size_bytes": len(manifest_raw),
        },
        "model_id": MODEL_ID,
        "provision_set_sha256": _provision_set_sha256(
            manifest_sha256=manifest_sha256,
            manifest_size_bytes=len(manifest_raw),
            lock_sha256=preflight.lock.raw_sha256,
            lock_size_bytes=len(preflight.lock.raw),
            artifact_set_sha256=preflight.artifact_set_sha256,
            total_size_bytes=preflight.lock.total_size_bytes,
        ),
        "schema_version": 1,
        "total_size_bytes": preflight.lock.total_size_bytes,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        os.fsync(descriptor)
    except OSError:
        _fail()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _new_private_output(
    path: Path | str,
    *,
    forbidden: Sequence[Path],
) -> _BoundOutputRoot:
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
        root, _identity_value = _private_directory(
            candidate,
            expected_names=set(),
        )
        reopened = root.lstat()
        if _directory_object_identity(reopened) != _directory_object_identity(
            created
        ):
            _fail()
        return _BoundOutputRoot(
            path=root,
            identity=_directory_object_identity(reopened),
        )
    except MobileWhisperProvisionError:
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
    except MobileWhisperProvisionError:
        raise
    except (BoundedReadError, OSError, RuntimeError, ValueError):
        _fail()


def _read_private(
    path: Path,
    *,
    maximum_bytes: int,
    expected_bytes: int | None = None,
) -> bytes:
    try:
        metadata = path.lstat()
        if (
            path.resolve(strict=True) != path
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            _fail()
        snapshot = read_regular_bounded(
            path,
            maximum_bytes=maximum_bytes,
            expected_bytes=expected_bytes,
        )
        after = path.lstat()
        if snapshot.path != path or _identity(after) != _identity(metadata):
            _fail()
        return snapshot.data
    except MobileWhisperProvisionError:
        raise
    except (BoundedReadError, OSError, RuntimeError, ValueError):
        _fail()


def _verify_output_layout(
    root: Path,
    lock: SourceLock,
    *,
    terminal: bool,
    expected_root_identity: tuple[int, ...] | None = None,
) -> None:
    expected = {LOCK_COPY_FILENAME, MANIFEST_FILENAME, MODEL_DIRECTORY}
    if terminal:
        expected.add(RECEIPT_FILENAME)
    _private_directory(root, expected_names=expected)
    if (
        expected_root_identity is not None
        and _directory_object_identity(root.lstat()) != expected_root_identity
    ):
        _fail()
    model_root = root / MODEL_DIRECTORY
    _private_directory(
        model_root,
        expected_names={item.filename for item in lock.artifacts},
    )
    for spec in lock.artifacts:
        _bind_private_artifact(model_root / spec.filename, spec)


def _precommit_guard(
    *,
    root: Path,
    root_identity: tuple[int, ...],
    preflight: MobileWhisperPreflight,
    manifest_raw: bytes,
) -> None:
    _revalidate_preflight(preflight)
    _verify_output_layout(
        root,
        preflight.lock,
        terminal=False,
        expected_root_identity=root_identity,
    )
    if (
        _read_private(
            root / LOCK_COPY_FILENAME,
            maximum_bytes=len(preflight.lock.raw),
            expected_bytes=len(preflight.lock.raw),
        )
        != preflight.lock.raw
        or _read_private(
            root / MANIFEST_FILENAME,
            maximum_bytes=_MAX_METADATA_BYTES,
            expected_bytes=len(manifest_raw),
        )
        != manifest_raw
    ):
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
            or metadata.st_uid != os.getuid()
            or metadata.st_size != len(raw)
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
    except MobileWhisperProvisionError:
        raise
    except (OSError, RuntimeError, ValueError):
        _fail()
    finally:
        if descriptor >= 0 and not prepared:
            os.close(descriptor)


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
    except MobileWhisperProvisionError:
        raise
    except (OSError, RuntimeError, ValueError):
        _fail()


def _commit_signal_numbers() -> set[int]:
    raw = (
        getattr(signal, "SIGHUP", None),
        getattr(signal, "SIGINT", None),
        getattr(signal, "SIGTERM", None),
    )
    if any(value is None for value in raw):
        _fail()
    try:
        numbers = tuple(int(value) for value in raw)
    except (TypeError, ValueError, OverflowError):
        _fail()
    if any(type(value) is not int or value <= 0 for value in numbers):
        _fail()
    result = set(numbers)
    if len(result) != len(numbers):
        _fail()
    return result


def _commit_terminal_receipt(
    root: Path,
    root_identity: tuple[int, ...],
    root_descriptor: int,
    descriptor: int,
    staged: _StagedReceipt,
    raw: bytes,
    *,
    state: _TerminalCommitState,
    commit_guard: Callable[[], None],
) -> None:
    if (
        not hasattr(signal, "pthread_sigmask")
        or state.linked
        or state.committed
    ):
        _fail()
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, _commit_signal_numbers())
    expected_identity: tuple[int, ...] | None = None
    try:
        commit_guard()
        if (
            _directory_object_identity(os.fstat(root_descriptor))
            != root_identity
            or _directory_object_identity(root.lstat()) != root_identity
        ):
            _fail()
        _verify_staged_receipt(descriptor, staged, raw)
        if (
            _directory_object_identity(os.fstat(root_descriptor))
            != root_identity
            or _directory_object_identity(root.lstat()) != root_identity
        ):
            _fail()
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
        state.linked = True
        published = os.stat(
            RECEIPT_FILENAME,
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(published.st_mode)
            or stat.S_IMODE(published.st_mode) != 0o600
            or published.st_nlink != 1
            or published.st_uid != os.getuid()
            or published.st_size != staged.size_bytes
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
        if (
            _directory_object_identity(os.fstat(root_descriptor))
            != root_identity
            or _directory_object_identity(root.lstat()) != root_identity
        ):
            _fail()
        os.fsync(root_descriptor)
        if (
            _directory_object_identity(os.fstat(root_descriptor))
            != root_identity
            or _directory_object_identity(root.lstat()) != root_identity
        ):
            _fail()
        state.committed = True
    except MobileWhisperProvisionError:
        raise
    except (OSError, RuntimeError, ValueError):
        _fail()
    finally:
        if not state.linked and expected_identity is not None:
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
                    state.linked = True
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def _publish_preflight(
    *,
    preflight: MobileWhisperPreflight,
    output_dir: Path | str,
) -> MobileWhisperProvision:
    _revalidate_preflight(preflight)
    bound_root = _new_private_output(
        output_dir,
        forbidden=(
            preflight.source_root,
            preflight.runtime_root,
            preflight.lock.path,
            preflight.worker.path,
            _REPO_ROOT,
        ),
    )
    root = bound_root.path
    root_identity = bound_root.identity
    state = _TerminalCommitState()
    staged_descriptor = -1
    manifest_raw = b""
    try:
        model_root = _mkdir_private(root, MODEL_DIRECTORY)
        retained_lock = _write_new_private(
            root,
            LOCK_COPY_FILENAME,
            preflight.lock.raw,
        )
        for source in preflight.source_artifacts:
            _copy_artifact(source, model_root)
        manifest_raw = (
            _canonical_json(
                _manifest_payload(
                    preflight,
                    output_root=root,
                    retained_lock=retained_lock,
                )
            )
            + b"\n"
        )
        _write_new_private(root, MANIFEST_FILENAME, manifest_raw)
        _precommit_guard(
            root=root,
            root_identity=root_identity,
            preflight=preflight,
            manifest_raw=manifest_raw,
        )
        # Every nonterminal byte and directory entry is durable before the
        # single terminal receipt link can become visible.
        _fsync_directory(model_root)
        _fsync_directory(root)
        receipt_raw = (
            _canonical_json(
                _receipt_payload(preflight, manifest_raw=manifest_raw)
            )
            + b"\n"
        )
        with opened_directory_nofollow(
            root,
            require_private=True,
        ) as (stable, root_descriptor):
            if (
                stable != root
                or _directory_object_identity(os.fstat(root_descriptor))
                != root_identity
                or _directory_object_identity(root.lstat()) != root_identity
            ):
                _fail()
            staged_descriptor, staged = _stage_terminal_receipt(
                root_descriptor,
                receipt_raw,
            )
            _commit_terminal_receipt(
                root,
                root_identity,
                root_descriptor,
                staged_descriptor,
                staged,
                receipt_raw,
                state=state,
                commit_guard=lambda: _precommit_guard(
                    root=root,
                    root_identity=root_identity,
                    preflight=preflight,
                    manifest_raw=manifest_raw,
                ),
            )
            if not state.committed:
                _fail()
        _fsync_directory(root.parent)
        provision = _load_provision_with_lock(
            root,
            lock=preflight.lock,
            expected_root_identity=root_identity,
        )
        if (
            provision.digest != hashlib.sha256(manifest_raw).hexdigest()
            or provision.artifact_set_sha256 != preflight.artifact_set_sha256
        ):
            _fail()
        return provision
    except BaseException as error:
        if state.committed:
            try:
                return _load_provision_with_lock(
                    root,
                    lock=preflight.lock,
                    expected_root_identity=root_identity,
                )
            except Exception:
                if isinstance(error, Exception):
                    _fail()
                raise
        if isinstance(error, MobileWhisperProvisionError):
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


def _root_from_input(path: Path | str) -> Path:
    candidate = _absolute(path)
    try:
        metadata = candidate.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            return candidate.resolve(strict=True)
        if (
            stat.S_ISREG(metadata.st_mode)
            and candidate.name == MANIFEST_FILENAME
            and candidate.resolve(strict=True) == candidate
        ):
            return candidate.parent
        _fail()
    except MobileWhisperProvisionError:
        raise
    except (OSError, RuntimeError, ValueError):
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
    sha256 = _exact_sha256(mapping["sha256"])
    size_bytes = _exact_positive_int(mapping["size_bytes"])
    if expected_path is not None and path != expected_path:
        _fail()
    bound = _bind_file(
        path,
        maximum_bytes=maximum_bytes,
        allow_final_symlink=allow_final_symlink,
    )
    if bound.sha256 != sha256 or bound.size_bytes != size_bytes:
        _fail()
    return bound


def _parse_manifest(
    raw: bytes,
    *,
    root: Path,
    lock: SourceLock,
) -> tuple[BoundFile, BoundFile, tuple[BoundArtifact, ...], str]:
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
        or value["schema_version"] != 11
        or value["kind"] != MANIFEST_KIND
        or value["model_id"] != MODEL_ID
        or value["adapter"] != ADAPTER
        or value["artifact_set_sha256"] != artifact_set
        or value["total_size_bytes"] != lock.total_size_bytes
        or value["mobile_config"] != lock.mobile_config.as_dict()
    ):
        _fail()
    evidence = _exact_dict(value["evidence_scope"], set(_EXPECTED_EVIDENCE_SCOPE))
    if evidence != _EXPECTED_EVIDENCE_SCOPE:
        _fail()
    limits = _exact_dict(
        value["limits"], {"case_timeout_sec", "startup_timeout_sec"}
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
    runtime_root, _runtime_root_identity = _venv_root_from_python(python.path)
    worker = _bound_from_payload(
        value["worker"],
        expected_path=_FIXED_WORKER,
        maximum_bytes=MAX_WORKER_BYTES,
        allow_final_symlink=False,
    )
    if (
        _overlaps(root, runtime_root)
        or any(
            _overlaps(runtime_root, path)
            for path in (lock.path, worker.path, _REPO_ROOT)
        )
    ):
        _fail()
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
        runtime["distributions"], set(_EXPECTED_RUNTIME_DISTRIBUTIONS)
    )
    if (
        distributions != dict(lock.runtime_distributions)
        or runtime["metadata_only_verified"] is not True
        or runtime["packages_imported"] is not False
        or runtime["model_loaded"] is not False
        or runtime["python"] != value["python"]
        or runtime["worker"] != value["worker"]
    ):
        _fail()
    source_fields = {"lock", "lock_recipe_sha256", *lock.source.keys()}
    source = _exact_dict(value["source"], source_fields)
    if (
        {name: source[name] for name in lock.source} != dict(lock.source)
        or source["lock_recipe_sha256"] != lock.recipe_sha256
    ):
        _fail()
    retained_lock = _bound_from_payload(
        source["lock"],
        expected_path=root / LOCK_COPY_FILENAME,
        maximum_bytes=len(lock.raw),
        allow_final_symlink=False,
    )
    if retained_lock.sha256 != lock.raw_sha256 or retained_lock.size_bytes != len(lock.raw):
        _fail()
    rows = value["artifacts"]
    if type(rows) is not list or len(rows) != len(lock.artifacts):
        _fail()
    artifacts: list[BoundArtifact] = []
    for row, spec in zip(rows, lock.artifacts, strict=True):
        mapping = _exact_dict(row, {"name", "path", "sha256", "size_bytes"})
        expected_path = root / MODEL_DIRECTORY / spec.filename
        if (
            mapping["name"] != spec.name
            or _absolute(_exact_string(mapping["path"])) != expected_path
            or mapping["sha256"] != spec.sha256
            or mapping["size_bytes"] != spec.size_bytes
        ):
            _fail()
        current = _bind_private_artifact(expected_path, spec)
        artifacts.append(
            BoundArtifact(
                path=current.path,
                sha256=spec.sha256,
                size_bytes=spec.size_bytes,
                name=spec.name,
            )
        )
    return python, worker, tuple(artifacts), artifact_set


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
        or value["complete"] is not True
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
        lock_row
        != {
            "filename": LOCK_COPY_FILENAME,
            "sha256": lock.raw_sha256,
            "size_bytes": len(lock.raw),
        }
        or manifest_row
        != {
            "filename": MANIFEST_FILENAME,
            "sha256": manifest_sha256,
            "size_bytes": len(manifest_raw),
        }
        or value["provision_set_sha256"]
        != _provision_set_sha256(
            manifest_sha256=manifest_sha256,
            manifest_size_bytes=len(manifest_raw),
            lock_sha256=lock.raw_sha256,
            lock_size_bytes=len(lock.raw),
            artifact_set_sha256=artifact_set_sha256,
            total_size_bytes=lock.total_size_bytes,
        )
    ):
        _fail()


def _load_provision_with_lock(
    path: Path | str,
    *,
    lock: SourceLock,
    expected_root_identity: tuple[int, ...] | None = None,
) -> MobileWhisperProvision:
    root = _root_from_input(path)
    if _has_git_ancestor(root):
        _fail()
    _verify_output_layout(
        root,
        lock,
        terminal=True,
        expected_root_identity=expected_root_identity,
    )
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
    python, worker, artifacts, artifact_set = _parse_manifest(
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
    _verify_output_layout(
        root,
        lock,
        terminal=True,
        expected_root_identity=expected_root_identity,
    )
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
        current_worker, worker
    ):
        _fail()
    for artifact, spec in zip(artifacts, lock.artifacts, strict=True):
        current = _bind_private_artifact(
            root / MODEL_DIRECTORY / spec.filename,
            spec,
        )
        if current.path != artifact.path:
            _fail()
    if (
        expected_root_identity is not None
        and _directory_object_identity(root.lstat()) != expected_root_identity
    ):
        _fail()
    return MobileWhisperProvision(
        path=root / MANIFEST_FILENAME,
        digest=hashlib.sha256(manifest_raw).hexdigest(),
        receipt_digest=hashlib.sha256(receipt_raw).hexdigest(),
        model_id=MODEL_ID,
        adapter=ADAPTER,
        python=python,
        worker=worker,
        artifacts=artifacts,
        mobile_config=lock.mobile_config,
        artifact_set_sha256=artifact_set,
        total_size_bytes=lock.total_size_bytes,
    )


def load_mobile_whisper_provision(path: Path | str) -> MobileWhisperProvision:
    """Strictly reopen one terminal-receipt-bound private provision."""

    try:
        return _load_provision_with_lock(path, lock=_load_source_lock())
    except MobileWhisperProvisionError:
        raise
    except (BoundedReadError, OSError, RuntimeError, TypeError, ValueError):
        _fail()


def provision_mobile_whisper(
    *,
    python: Path | str,
    model_root: Path | str,
    output_dir: Path | str,
) -> MobileWhisperProvision:
    """Preflight and publish one exact private, self-contained provision."""

    try:
        preflight = preflight_mobile_whisper(
            python=python,
            model_root=model_root,
        )
        return _publish_preflight(preflight=preflight, output_dir=output_dir)
    except MobileWhisperProvisionError:
        raise
    except (BoundedReadError, OSError, RuntimeError, TypeError, ValueError):
        _fail()


def _safe_preflight_result(preflight: MobileWhisperPreflight) -> dict[str, object]:
    return {
        "artifact_set_sha256": preflight.artifact_set_sha256,
        "metadata_only_runtime_verified": True,
        "model_id": MODEL_ID,
        "ok": True,
        "ready_to_publish": True,
        "total_size_bytes": preflight.lock.total_size_bytes,
    }


def _safe_provision_result(provision: MobileWhisperProvision) -> dict[str, object]:
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
        "kind": "mobile-whisper-provision-help-v1",
        "ok": True,
        "publish_only_flag": "--output-dir",
    }


def _emit(value: object) -> bool:
    """Attempt one bounded result write without retrying a broken stdout."""

    try:
        payload = _canonical_json(value) + b"\n"
        if len(payload) > _MAX_METADATA_BYTES:
            return False
        sys.stdout.buffer.write(payload)
        sys.stdout.buffer.flush()
        return True
    except (BrokenPipeError, OSError, ValueError, MobileWhisperProvisionError):
        return False


def main(argv: Sequence[str] | None = None) -> int:
    """Run bounded path-free preflight or publication."""

    try:
        command = _parse_cli(tuple(sys.argv[1:] if argv is None else argv))
        if command is None:
            result = _help_result()
        elif command.action == "preflight":
            result = _safe_preflight_result(
                preflight_mobile_whisper(
                    python=command.python,
                    model_root=command.model_root,
                )
            )
        else:
            if command.output_dir is None:
                _fail()
            result = _safe_provision_result(
                provision_mobile_whisper(
                    python=command.python,
                    model_root=command.model_root,
                    output_dir=command.output_dir,
                )
            )
        exit_code = 0
    except Exception:
        result = _SAFE_ERROR
        exit_code = 2
    return exit_code if _emit(result) else 2


if __name__ == "__main__":
    raise SystemExit(main())
