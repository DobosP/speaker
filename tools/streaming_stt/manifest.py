"""Strict, hash-bound worker manifests for the streaming-STT harness."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .bounded_io import (
    BoundedReadError,
    hash_regular_bounded,
    read_regular_bounded,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}\Z")
_MAX_MANIFEST_BYTES = 64 * 1024
MAX_PYTHON_BYTES = 512 * 1024 * 1024
MAX_WORKER_BYTES = 4 * 1024 * 1024
MAX_FAKE_ARTIFACT_BYTES = 4 * 1024 * 1024
_MANIFEST_FIELDS = {
    "schema_version",
    "model_id",
    "adapter",
    "python",
    "worker",
    "artifacts",
    "limits",
}
_FILE_FIELDS = {"path", "sha256", "size_bytes"}
_ARTIFACT_FIELDS = {"name", *_FILE_FIELDS}
_LIMIT_FIELDS = {"startup_timeout_sec", "case_timeout_sec"}


class ManifestError(RuntimeError):
    """A detail-free manifest or local-artifact validation failure."""


@dataclass(frozen=True)
class BoundFile:
    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class BoundArtifact(BoundFile):
    name: str


@dataclass(frozen=True)
class WorkerLimits:
    startup_timeout_sec: float
    case_timeout_sec: float


@dataclass(frozen=True)
class WorkerManifest:
    path: Path
    digest: str
    model_id: str
    adapter: str
    python: BoundFile
    worker: BoundFile
    artifacts: tuple[BoundArtifact, ...]
    limits: WorkerLimits

    @property
    def artifact_by_name(self) -> Mapping[str, BoundArtifact]:
        return {artifact.name: artifact for artifact in self.artifacts}


def _bad() -> object:
    raise ManifestError()


def _strict_json(raw: bytes) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ManifestError()
            result[key] = value
        return result

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: _bad())
    except (UnicodeError, ValueError, OverflowError, ManifestError):
        raise ManifestError() from None


def _safe_id(value: object) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise ManifestError()
    return value


def _sha256(value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ManifestError()
    return value


def _positive_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ManifestError()
    return value


def _bounded_seconds(value: object, *, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError()
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise ManifestError() from None
    if not math.isfinite(number) or number <= 0.0 or number > maximum:
        raise ManifestError()
    return number


def _bound_file(
    value: object,
    *,
    expected_fields: set[str],
    allow_symlink: bool,
    maximum_bytes: int,
) -> BoundFile:
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ManifestError()
    raw_path = value.get("path")
    if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
        raise ManifestError()
    candidate = Path(raw_path)
    if not candidate.is_absolute() or (candidate.is_symlink() and not allow_symlink):
        raise ManifestError()
    size = _positive_int(value.get("size_bytes"))
    expected_hash = _sha256(value.get("sha256"))
    try:
        digest = hash_regular_bounded(
            candidate,
            maximum_bytes=maximum_bytes,
            expected_bytes=size,
            allow_final_symlink=allow_symlink,
        )
    except BoundedReadError:
        raise ManifestError() from None
    if digest.sha256 != expected_hash:
        raise ManifestError()
    return BoundFile(path=candidate, sha256=expected_hash, size_bytes=size)


def _artifact(value: object) -> BoundArtifact:
    bound = _bound_file(
        value,
        expected_fields=_ARTIFACT_FIELDS,
        allow_symlink=False,
        maximum_bytes=MAX_FAKE_ARTIFACT_BYTES,
    )
    assert isinstance(value, dict)
    return BoundArtifact(
        path=bound.path,
        sha256=bound.sha256,
        size_bytes=bound.size_bytes,
        name=_safe_id(value.get("name")),
    )


def load_worker_manifest(path: Path | str) -> WorkerManifest:
    """Load and verify one immutable, machine-local worker receipt."""

    candidate = Path(path).expanduser()
    try:
        snapshot = read_regular_bounded(
            candidate,
            maximum_bytes=_MAX_MANIFEST_BYTES,
        )
    except BoundedReadError:
        raise ManifestError() from None
    resolved = snapshot.path
    raw = snapshot.data
    value = _strict_json(raw)
    if not isinstance(value, dict) or set(value) != _MANIFEST_FIELDS:
        raise ManifestError()
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ManifestError()

    python = _bound_file(
        value.get("python"),
        expected_fields=_FILE_FIELDS,
        allow_symlink=True,
        maximum_bytes=MAX_PYTHON_BYTES,
    )
    worker = _bound_file(
        value.get("worker"),
        expected_fields=_FILE_FIELDS,
        allow_symlink=False,
        maximum_bytes=MAX_WORKER_BYTES,
    )
    raw_artifacts = value.get("artifacts")
    if not isinstance(raw_artifacts, list) or len(raw_artifacts) != 1:
        raise ManifestError()
    artifacts = tuple(_artifact(item) for item in raw_artifacts)
    artifact_names = [artifact.name for artifact in artifacts]
    if artifact_names != ["fake-script"]:
        raise ManifestError()

    raw_limits = value.get("limits")
    if not isinstance(raw_limits, dict) or set(raw_limits) != _LIMIT_FIELDS:
        raise ManifestError()
    limits = WorkerLimits(
        startup_timeout_sec=_bounded_seconds(
            raw_limits.get("startup_timeout_sec"),
            maximum=600.0,
        ),
        case_timeout_sec=_bounded_seconds(
            raw_limits.get("case_timeout_sec"),
            maximum=3600.0,
        ),
    )
    adapter = _safe_id(value.get("adapter"))
    if adapter != "fake-json-v1":
        # Real model adapters are deliberately absent from this first landing.
        raise ManifestError()
    return WorkerManifest(
        path=resolved,
        digest=hashlib.sha256(raw).hexdigest(),
        model_id=_safe_id(value.get("model_id")),
        adapter=adapter,
        python=python,
        worker=worker,
        artifacts=artifacts,
        limits=limits,
    )
