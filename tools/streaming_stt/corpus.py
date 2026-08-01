"""Strict local f32le corpus loading for streaming benchmark replay."""

from __future__ import annotations

from array import array
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
from dataclasses import dataclass, field

from core.wer import normalize

from .bounded_io import BoundedReadError, read_regular_bounded
from .protocol import MAX_CORPUS_BYTES, MAX_PCM_BYTES


_SAFE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ROOT_V1_FIELDS = {"schema_version", "purpose", "cases"}
_ROOT_V2_FIELDS = {*_ROOT_V1_FIELDS, "provenance"}
_ROOT_V3_FIELDS = _ROOT_V2_FIELDS
_CASE_FIELDS = {
    "id",
    "file",
    "sha256",
    "samples",
    "expected_text",
    "assertion",
    "commands",
    "tags",
}
_PROVENANCE_FIELDS = {
    "kind",
    "suite",
    "manifest_sha256",
    "metadata_sha256",
    "source_set_sha256",
}
_PUBLIC_PROVENANCE_KIND = "public-voice-v1"
_PRIVATE_DIAGNOSTIC_PROVENANCE_KIND = "private-diagnostic-v1"
_MAX_CORPUS_MANIFEST_BYTES = 256 * 1024
_MAX_CASES = 512
_MAX_REFERENCE_CHARS = 4096


class CorpusError(RuntimeError):
    """A detail-free corpus validation failure."""


@dataclass(frozen=True)
class CorpusCase:
    case_id: str
    source_path: Path = field(repr=False)
    sha256: str
    samples: int
    expected_text: str = field(repr=False)
    assertion: str
    commands: tuple[str, ...] = field(repr=False)
    tags: tuple[str, ...]
    audio_bytes: bytes = field(repr=False)


@dataclass(frozen=True)
class CorpusProvenance:
    kind: str
    suite: str
    manifest_sha256: str
    metadata_sha256: str
    source_set_sha256: str

    def as_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "suite": self.suite,
            "manifest_sha256": self.manifest_sha256,
            "metadata_sha256": self.metadata_sha256,
            "source_set_sha256": self.source_set_sha256,
        }


@dataclass(frozen=True)
class LoadedCorpus:
    path: Path
    digest: str
    schema_version: int
    purpose: str
    provenance: CorpusProvenance | None
    cases: tuple[CorpusCase, ...] = field(repr=False)
    audio_bytes: int


def _bad() -> object:
    raise CorpusError()


def _strict_json(raw: bytes) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise CorpusError()
            result[key] = value
        return result

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: _bad())
    except (UnicodeError, ValueError, OverflowError, CorpusError):
        raise CorpusError() from None


def _safe_id(value: object) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise CorpusError()
    return value


def _sha256(value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CorpusError()
    return value


def _provenance(value: object, *, schema_version: int) -> CorpusProvenance:
    if not isinstance(value, dict) or set(value) != _PROVENANCE_FIELDS:
        raise CorpusError()
    kind = value.get("kind")
    expected_kind = {
        2: _PUBLIC_PROVENANCE_KIND,
        3: _PRIVATE_DIAGNOSTIC_PROVENANCE_KIND,
    }.get(schema_version)
    if kind != expected_kind:
        raise CorpusError()
    return CorpusProvenance(
        kind=str(kind),
        suite=_safe_id(value.get("suite")),
        manifest_sha256=_sha256(value.get("manifest_sha256")),
        metadata_sha256=_sha256(value.get("metadata_sha256")),
        source_set_sha256=_sha256(value.get("source_set_sha256")),
    )


def _string_list(
    value: object,
    *,
    maximum_items: int,
    maximum_chars: int,
    safe_ids: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > maximum_items:
        raise CorpusError()
    result: list[str] = []
    for item in value:
        if (
            not isinstance(item, str)
            or not item
            or len(item) > maximum_chars
            or (safe_ids and _SAFE_ID_RE.fullmatch(item) is None)
        ):
            raise CorpusError()
        result.append(item)
    if len(set(result)) != len(result):
        raise CorpusError()
    return tuple(result)


def _finite_f32le(raw: bytes, samples: int) -> bool:
    values = array("f")
    try:
        values.frombytes(raw)
    except (BufferError, ValueError):
        return False
    if sys.byteorder != "little":
        values.byteswap()
    return len(values) == samples and all(math.isfinite(value) for value in values)


def _load_case(
    root: Path,
    value: object,
    *,
    remaining_bytes: int,
) -> CorpusCase:
    if not isinstance(value, dict) or set(value) != _CASE_FIELDS:
        raise CorpusError()
    case_id = _safe_id(value.get("id"))
    filename = value.get("file")
    if not isinstance(filename, str) or not filename or "\x00" in filename:
        raise CorpusError()
    pure = PurePosixPath(filename)
    if pure.is_absolute() or len(pure.parts) != 1 or pure.name != filename:
        raise CorpusError()
    candidate = root / filename
    samples = value.get("samples")
    if (
        isinstance(samples, bool)
        or not isinstance(samples, int)
        or samples <= 0
        or samples * 4 > MAX_PCM_BYTES
        or samples * 4 > remaining_bytes
    ):
        raise CorpusError()
    expected_hash = value.get("sha256")
    if (
        not isinstance(expected_hash, str)
        or _SHA256_RE.fullmatch(expected_hash) is None
    ):
        raise CorpusError()
    try:
        snapshot = read_regular_bounded(
            candidate,
            maximum_bytes=MAX_PCM_BYTES,
            expected_bytes=samples * 4,
        )
        path = snapshot.path
        path.relative_to(root)
        raw = snapshot.data
    except (BoundedReadError, ValueError):
        raise CorpusError() from None
    if (
        len(raw) != samples * 4
        or hashlib.sha256(raw).hexdigest() != expected_hash
        or not _finite_f32le(raw, samples)
    ):
        raise CorpusError()
    expected_text = value.get("expected_text")
    assertion = value.get("assertion")
    if (
        not isinstance(expected_text, str)
        or len(expected_text) > _MAX_REFERENCE_CHARS
        or assertion not in {"transcript", "silence"}
    ):
        raise CorpusError()
    commands = _string_list(
        value.get("commands"),
        maximum_items=16,
        maximum_chars=128,
    )
    tags = _string_list(
        value.get("tags"),
        maximum_items=32,
        maximum_chars=64,
        safe_ids=True,
    )
    if (assertion == "transcript" and not expected_text.strip()) or (
        assertion == "silence" and (expected_text != "" or commands)
    ):
        raise CorpusError()
    if assertion == "transcript" and (
        not normalize(expected_text)
        or any(not normalize(command) for command in commands)
    ):
        raise CorpusError()
    return CorpusCase(
        case_id=case_id,
        source_path=path,
        sha256=expected_hash,
        samples=samples,
        expected_text=expected_text,
        assertion=str(assertion),
        commands=commands,
        tags=tags,
        audio_bytes=raw,
    )


def load_corpus(path: Path | str) -> LoadedCorpus:
    candidate = Path(path).expanduser()
    try:
        snapshot = read_regular_bounded(
            candidate,
            maximum_bytes=_MAX_CORPUS_MANIFEST_BYTES,
        )
    except BoundedReadError:
        raise CorpusError() from None
    resolved = snapshot.path
    raw = snapshot.data
    value = _strict_json(raw)
    if (
        not isinstance(value, dict)
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") not in {1, 2, 3}
        or not isinstance(value.get("purpose"), str)
        or not str(value.get("purpose")).strip()
        or len(str(value.get("purpose"))) > 512
    ):
        raise CorpusError()
    schema_version = value["schema_version"]
    expected_fields = {
        1: _ROOT_V1_FIELDS,
        2: _ROOT_V2_FIELDS,
        3: _ROOT_V3_FIELDS,
    }[schema_version]
    if set(value) != expected_fields:
        raise CorpusError()
    provenance = (
        None
        if schema_version == 1
        else _provenance(value.get("provenance"), schema_version=schema_version)
    )
    raw_cases = value.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases or len(raw_cases) > _MAX_CASES:
        raise CorpusError()
    root = resolved.parent.resolve(strict=True)
    loaded_cases: list[CorpusCase] = []
    total = 0
    for item in raw_cases:
        case = _load_case(
            root,
            item,
            remaining_bytes=MAX_CORPUS_BYTES - total,
        )
        loaded_cases.append(case)
        total += len(case.audio_bytes)
    cases = tuple(loaded_cases)
    case_ids = [case.case_id for case in cases]
    filenames = [case.source_path.name.casefold() for case in cases]
    if len(set(case_ids)) != len(case_ids) or len(set(filenames)) != len(filenames):
        raise CorpusError()
    return LoadedCorpus(
        path=resolved,
        digest=hashlib.sha256(raw).hexdigest(),
        schema_version=schema_version,
        purpose=str(value["purpose"]),
        provenance=provenance,
        cases=cases,
        audio_bytes=total,
    )


def verify_corpus_snapshot(corpus: LoadedCorpus) -> None:
    """Reject a source manifest or PCM changed after the in-memory snapshot."""

    try:
        manifest = read_regular_bounded(
            corpus.path,
            maximum_bytes=_MAX_CORPUS_MANIFEST_BYTES,
        )
        if hashlib.sha256(manifest.data).hexdigest() != corpus.digest:
            raise CorpusError()
        for case in corpus.cases:
            snapshot = read_regular_bounded(
                case.source_path,
                maximum_bytes=MAX_PCM_BYTES,
                expected_bytes=len(case.audio_bytes),
            )
            if hashlib.sha256(snapshot.data).hexdigest() != case.sha256:
                raise CorpusError()
    except (OSError, BoundedReadError):
        raise CorpusError() from None
