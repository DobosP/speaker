"""Prepare and verify the bounded Microsoft AEC 32-by-four fixture.

The command is deliberately offline.  It accepts an owner-private materialized
snapshot labelled with the pinned revision, verifies ``meta.csv`` and all 128
selected Git-LFS payload bytes against the committed lock, and copies those
bytes into an opaque private bundle.  Git tree and pointer-object provenance is
statically lock-tested rather than recertified from the supplied checkout.  The
terminal preparation receipt is an anonymous ``O_TMPFILE`` linked only after
every source/output/lock/closure recheck.

This fixture contains synchronized component-test signals.  It is not a live
microphone, room, endpoint, latency, conversation, or product-quality result.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import struct
from typing import Callable, Mapping, Sequence

import numpy as np

from tools.streaming_stt.bounded_io import BoundedReadError, read_regular_bounded


SCHEMA_VERSION = 1
FIXTURE_ID = "microsoft_aec_challenge_synthetic_hash32_v1"
LOCK_KIND = "microsoft-aec-32x4-fixture-lock-v1"
MANIFEST_KIND = "microsoft-aec-private-fixture-v1"
RECEIPT_KIND = "microsoft-aec-private-preparation-receipt-v1"
REVISION = "6c633d0a9d2a143a0e364899b91b06f127315b18"
SAMPLE_RATE_HZ = 16_000
FRAME_SAMPLES = 160
CAPTURE_BLOCK_SAMPLES = 1_600
CASE_COUNT = 32
ARTIFACT_COUNT = 128
META_RELATIVE_PATH = "datasets/synthetic/meta.csv"
META_SIZE_BYTES = 3_243_688
META_SHA256 = "6b867aa68f510e4bafc9ffa05202930ffc1bb54bb0cffe66aa1ec3bcc606f3e5"
SELECTED_ROWS_SHA256 = (
    "542d6d102cad3f47c7c5aad96bbf04c7796654ba13573068bc930232b38ba331"
)
SELECTION_SEED = "speaker-public-eval-v1-2026-08-01"
SELECTION_DESCRIPTION = (
    "Rank synthetic/meta.csv rows by SHA-256(UTF8(seed) || 0x00 || compact "
    "canonical JSON([fileid])), sort by (rank, identity), take 32, and bind all "
    "four LFS WAVs per selected row."
)
SELECTION_POLICY_SHA256 = (
    "abd006fe51ba980bdabc197a0ac40a47a5c5728b964badf26b43a5aa9e1e10bd"
)
DEFAULT_LOCK = (
    Path(__file__).resolve().parent / "audio_eval" / "microsoft_aec_32.lock.json"
)
EXPECTED_LOCK_RECIPE_SHA256 = (
    "a1c6c65f8d40dd147bb7ba13fc0e389bfe33adecf411b3200031215ba65cf198"
)
MANIFEST_FILENAME = "fixture.json"
PREPARATION_RECEIPT_FILENAME = "preparation-receipt.json"
LICENSE_ACCEPTANCE_IDS = (
    "LIBRIVOX-PUBLIC-DOMAIN",
    "EDINBURGH-VCTK-DATA-TERMS",
    "AUDIOSET-CC-BY-4.0",
    "FREESOUND-SELECTED-CC0-1.0",
    "DEMAND-CC-BY-SA-3.0",
)
SIGNAL_ROLES = (
    "echo_signal",
    "farend_speech",
    "nearend_mic_signal",
    "nearend_speech",
)
META_COLUMNS = (
    "nearend_speaker",
    "nearend_wav_path",
    "nearend_wav_path_noisy",
    "farend_speaker",
    "farend_wav_path",
    "farend_wav_path_noisy",
    "ser",
    "is_farend_nonlinear",
    "is_farend_noisy",
    "is_nearend_noisy",
    "split",
    "fileid",
    "nearend_scale",
)
_PREPARER_FILES = (
    "tools/__init__.py",
    "tools/prepare_microsoft_aec_fixture.py",
    "tools/streaming_stt/__init__.py",
    "tools/streaming_stt/bounded_io.py",
)
_MAX_LOCK_BYTES = 512 * 1024
_MAX_MANIFEST_BYTES = 256 * 1024
_MAX_RECEIPT_BYTES = 256 * 1024
_MAX_META_BYTES = 4 * 1024 * 1024
_MAX_WAV_BYTES = 512 * 1024
_MAX_JSON_DEPTH = 16
_MAX_JSON_ITEMS = 100_000
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SHA1_RE = re.compile(r"[0-9a-f]{40}\Z")
_SAFE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,95}\Z")
_SAFE_PATH_PART_RE = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z")
_SAFE_ERROR = {"error": "microsoft_aec_fixture_prerequisites_unavailable", "ok": False}


class MicrosoftAecFixtureError(RuntimeError):
    """A detail-free source, lock, private-bundle, or publication failure."""


class _LifecycleSignal(BaseException):
    def __init__(self, signum: int) -> None:
        self.signum = int(signum)


@dataclass(frozen=True, slots=True)
class TestFixtureInjection:
    """Explicitly non-production lock surface for small generated tests."""

    fixture_id: str
    lock_value: Mapping[str, object] = field(repr=False)


@dataclass(frozen=True, slots=True)
class _ArtifactPin:
    role: str
    relative_path: str = field(repr=False)
    git_pointer_blob_sha1: str
    lfs_oid_sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class _CasePin:
    ordinal: int
    fileid: str = field(repr=False)
    metadata: Mapping[str, str] = field(repr=False)
    artifacts: tuple[_ArtifactPin, ...]


@dataclass(frozen=True, slots=True)
class _FixtureLock:
    fixture_id: str
    recipe_sha256: str
    meta_git_blob_sha1: str
    metadata_sha256: str
    metadata_size_bytes: int
    metadata_row_count: int
    selected_rows_sha256: str
    cases: tuple[_CasePin, ...]
    production_evidence: bool
    raw_sha256: str
    raw_size_bytes: int


@dataclass(frozen=True, slots=True)
class _BoundSource:
    relative_path: str = field(repr=False)
    raw: bytes = field(repr=False)
    sha256: str
    size_bytes: int
    samples: int
    snapshot: tuple[int, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class PrivateFileIdentity:
    name: str = field(repr=False)
    size_bytes: int
    sha256: str
    snapshot: tuple[int, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class MicrosoftAecArtifact:
    name: str = field(repr=False)
    role: str
    size_bytes: int
    sha256: str
    samples: int
    snapshot: tuple[int, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class MicrosoftAecCase:
    case_id: str
    ordinal: int
    rank_sha256: str
    nearend_scale: str
    ser: str
    is_farend_nonlinear: bool
    is_farend_noisy: bool
    is_nearend_noisy: bool
    split: str
    artifacts: tuple[MicrosoftAecArtifact, ...]


@dataclass(frozen=True, slots=True)
class LoadedMicrosoftAecBundle:
    root: Path = field(repr=False)
    fixture_id: str
    production_evidence: bool
    lock_recipe_sha256: str
    manifest_sha256: str
    receipt_sha256: str
    source_contract_sha256: str
    cases: tuple[MicrosoftAecCase, ...]
    identities: tuple[tuple[int, ...], ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class PreparedMicrosoftAecFixture:
    path: Path = field(repr=False)
    fixture_id: str
    production_evidence: bool
    manifest_sha256: str
    receipt_sha256: str
    source_contract_sha256: str
    selected_rows_sha256: str
    artifact_count: int
    audio_bytes: int


@dataclass(slots=True)
class _CommitState:
    committed: bool = False


def _canonical_json(value: object, *, newline: bool = False) -> bytes:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (MemoryError, RecursionError, TypeError, UnicodeError, ValueError):
        raise MicrosoftAecFixtureError() from None
    return raw + (b"\n" if newline else b"")


def _strict_json(raw: bytes, *, maximum_bytes: int) -> object:
    if not raw or len(raw) > maximum_bytes:
        raise MicrosoftAecFixtureError()

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise MicrosoftAecFixtureError()
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_float=Decimal,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                MicrosoftAecFixtureError()
            ),
        )
    except (UnicodeError, ValueError, OverflowError, MicrosoftAecFixtureError):
        raise MicrosoftAecFixtureError() from None
    count = 0
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        if depth > _MAX_JSON_DEPTH:
            raise MicrosoftAecFixtureError()
        if isinstance(current, dict):
            count += len(current)
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            count += len(current)
            stack.extend((item, depth + 1) for item in current)
        elif isinstance(current, str):
            if len(current) > 16_384:
                raise MicrosoftAecFixtureError()
        elif current is not None and not isinstance(current, (bool, int, Decimal)):
            raise MicrosoftAecFixtureError()
        if count > _MAX_JSON_ITEMS:
            raise MicrosoftAecFixtureError()
    return value


def _sha256(value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise MicrosoftAecFixtureError()
    return value


def _sha1(value: object) -> str:
    if not isinstance(value, str) or _SHA1_RE.fullmatch(value) is None:
        raise MicrosoftAecFixtureError()
    return value


def _string(value: object, *, maximum: int = 512, allow_empty: bool = False) -> str:
    if (
        not isinstance(value, str)
        or len(value) > maximum
        or (not allow_empty and not value)
        or "\x00" in value
        or "\r" in value
        or "\n" in value
    ):
        raise MicrosoftAecFixtureError()
    return value


def _integer(value: object, *, minimum: int = 0, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise MicrosoftAecFixtureError()
    return value


def _bool_text(value: str) -> bool:
    if value not in {"0", "1"}:
        raise MicrosoftAecFixtureError()
    return value == "1"


def _decimal_text(value: str, *, minimum: Decimal, maximum: Decimal) -> str:
    text = _string(value, maximum=64)
    try:
        parsed = Decimal(text)
    except (InvalidOperation, ValueError, OverflowError):
        raise MicrosoftAecFixtureError() from None
    if not parsed.is_finite() or not minimum <= parsed <= maximum:
        raise MicrosoftAecFixtureError()
    return text


def _file_snapshot(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_nlink,
        stat.S_IMODE(metadata.st_mode),
        metadata.st_uid,
    )


def _directory_locator(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_uid,
    )


def _safe_relative(value: object) -> str:
    text = _string(value, maximum=256)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or not path.parts
        or any(
            part in {"", ".", ".."} or _SAFE_PATH_PART_RE.fullmatch(part) is None
            for part in path.parts
        )
    ):
        raise MicrosoftAecFixtureError()
    return text


def _safe_leaf(value: object) -> str:
    text = _string(value, maximum=128)
    if Path(text).name != text or _SAFE_ID_RE.fullmatch(text) is None:
        raise MicrosoftAecFixtureError()
    return text


def _rank(seed: str, fileid: str) -> str:
    identity = json.dumps([fileid], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(
        seed.encode("utf-8") + b"\0" + identity.encode("utf-8")
    ).hexdigest()


def _selected_rows_digest(rows: Sequence[Mapping[str, str]]) -> str:
    identities = [
        json.dumps([row["fileid"]], ensure_ascii=False, separators=(",", ":"))
        for row in rows
    ]
    return hashlib.sha256(
        json.dumps(
            identities,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _select_production_rows(
    rows: Sequence[Mapping[str, str]],
) -> tuple[Mapping[str, str], ...]:
    ranked: list[tuple[str, str, Mapping[str, str]]] = []
    identities: set[str] = set()
    for row in rows:
        fileid = row["fileid"]
        identity = json.dumps(
            [fileid], ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
        if identity in identities:
            raise MicrosoftAecFixtureError()
        identities.add(identity)
        ranked.append((_rank(SELECTION_SEED, fileid), identity, row))
    ranked.sort(key=lambda item: (item[0], item[1]))
    selected = tuple(row for _rank_value, _identity, row in ranked[:CASE_COUNT])
    if len(selected) != CASE_COUNT:
        raise MicrosoftAecFixtureError()
    return selected


def _selection_policy_digest() -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "algorithm": "sha256_top_k_v1",
                "description": SELECTION_DESCRIPTION,
                "expected_examples": CASE_COUNT,
                "fixed_ids": [],
                "identity_fields": ["fileid"],
                "seed": SELECTION_SEED,
                "speaker_disjoint": False,
                "speaker_field": None,
                "split": "all-metadata-rows",
                "strata_fields": [],
            },
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _mapping(value: object, fields: set[str]) -> Mapping[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise MicrosoftAecFixtureError()
    return value


def _canonical_self_digest(value: Mapping[str, object]) -> str:
    copied = dict(value)
    raw_self = copied.get("self_digest")
    if not isinstance(raw_self, dict):
        raise MicrosoftAecFixtureError()
    self_digest = dict(raw_self)
    self_digest.pop("value", None)
    copied["self_digest"] = self_digest
    try:
        encoded = json.dumps(
            copied,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (MemoryError, RecursionError, TypeError, UnicodeError, ValueError):
        raise MicrosoftAecFixtureError() from None
    return hashlib.sha256(encoded).hexdigest()


def _expected_artifact_path(role: str, fileid: str) -> str:
    names = {
        "echo_signal": f"datasets/synthetic/echo_signal/echo_fileid_{fileid}.wav",
        "farend_speech": (
            f"datasets/synthetic/farend_speech/farend_speech_fileid_{fileid}.wav"
        ),
        "nearend_mic_signal": (
            f"datasets/synthetic/nearend_mic_signal/nearend_mic_fileid_{fileid}.wav"
        ),
        "nearend_speech": (
            f"datasets/synthetic/nearend_speech/nearend_speech_fileid_{fileid}.wav"
        ),
    }
    try:
        return names[role]
    except KeyError:
        raise MicrosoftAecFixtureError() from None


def _parse_lock_value(
    value: object,
    *,
    raw_sha256: str,
    raw_size_bytes: int,
    production_evidence: bool,
) -> _FixtureLock:
    root = _mapping(
        value,
        {
            "schema_version",
            "fixture_id",
            "self_digest",
            "source",
            "license",
            "usage_constraints",
            "selection",
            "layout",
            "metadata",
            "metric_evidence",
            "cases",
        },
    )
    self_digest = _mapping(
        root["self_digest"],
        {"algorithm", "canonicalization", "excluded_json_pointer", "value"},
    )
    recipe = _sha256(self_digest["value"])
    if (
        self_digest["algorithm"] != "sha256"
        or self_digest["canonicalization"]
        != (
            "UTF-8 JSON with sort_keys=true, separators=(',', ':'), "
            "ensure_ascii=false, allow_nan=false"
        )
        or self_digest["excluded_json_pointer"] != "/self_digest/value"
        or _canonical_self_digest(root) != recipe
    ):
        raise MicrosoftAecFixtureError()
    fixture_id = _string(root["fixture_id"], maximum=96)
    if _SAFE_ID_RE.fullmatch(fixture_id) is None:
        raise MicrosoftAecFixtureError()
    source = _mapping(
        root["source"],
        {"repository_url", "commit", "root_tree_sha1"},
    )
    license_value = _mapping(
        root["license"], {"gate_id", "acceptance_ids", "redistribution"}
    )
    selection = _mapping(
        root["selection"],
        {
            "algorithm",
            "description",
            "split",
            "identity_fields",
            "strata_fields",
            "seed",
            "expected_examples",
            "speaker_disjoint",
            "speaker_field",
            "fixed_ids",
            "policy_sha256",
            "selected_rows_sha256",
            "source_row_domain",
            "selected_upstream_split_counts",
        },
    )
    layout = _mapping(
        root["layout"],
        {
            "case_count",
            "signal_roles",
            "artifact_count",
            "audio_bytes",
            "metadata_bytes",
            "total_pinned_bytes",
        },
    )
    metadata = _mapping(
        root["metadata"],
        {
            "relative_path",
            "git_blob_sha1",
            "sha256",
            "size_bytes",
            "row_count",
            "columns",
        },
    )
    if not isinstance(root["usage_constraints"], list) or not root["usage_constraints"]:
        raise MicrosoftAecFixtureError()
    if not isinstance(root["metric_evidence"], dict):
        raise MicrosoftAecFixtureError()
    cases_raw = root["cases"]
    case_count = _integer(layout["case_count"], minimum=1, maximum=CASE_COUNT)
    artifact_count = _integer(
        layout["artifact_count"], minimum=4, maximum=ARTIFACT_COUNT
    )
    audio_bytes = _integer(layout["audio_bytes"], minimum=1, maximum=64 * 1024 * 1024)
    metadata_bytes = _integer(
        layout["metadata_bytes"], minimum=1, maximum=_MAX_META_BYTES
    )
    common_invalid = (
        root["schema_version"] != SCHEMA_VERSION
        or source["repository_url"] != "https://github.com/microsoft/AEC-Challenge"
        or source["commit"] != REVISION
        or _SHA1_RE.fullmatch(_string(source["root_tree_sha1"], maximum=40)) is None
        or license_value["acceptance_ids"] != list(LICENSE_ACCEPTANCE_IDS)
        or license_value["redistribution"] != "cache-only"
        or selection["identity_fields"] != ["fileid"]
        or selection["expected_examples"] != case_count
        or layout["signal_roles"] != list(SIGNAL_ROLES)
        or artifact_count != case_count * len(SIGNAL_ROLES)
        or metadata["relative_path"] != META_RELATIVE_PATH
        or _sha1(metadata["git_blob_sha1"]) == "0" * 40
        or _sha256(metadata["sha256"]) == "0" * 64
        or metadata["size_bytes"] != metadata_bytes
        or metadata["columns"] != list(META_COLUMNS)
        or layout["total_pinned_bytes"] != audio_bytes + metadata_bytes
        or not isinstance(cases_raw, list)
        or len(cases_raw) != case_count
    )
    if common_invalid:
        raise MicrosoftAecFixtureError()
    selected_digest = _sha256(selection["selected_rows_sha256"])
    policy_digest = _sha256(selection["policy_sha256"])
    metadata_digest = _sha256(metadata["sha256"])
    metadata_row_count = _integer(
        metadata["row_count"], minimum=case_count, maximum=10_000
    )
    if production_evidence:
        if (
            selection["algorithm"] != "sha256_top_k_v1"
            or selection["description"] != SELECTION_DESCRIPTION
            or selection["split"] != "all-metadata-rows"
            or selection["strata_fields"] != []
            or selection["seed"] != SELECTION_SEED
            or selection["speaker_disjoint"] is not False
            or selection["speaker_field"] is not None
            or selection["fixed_ids"] != []
            or selection["source_row_domain"] != "all_metadata_rows"
            or selected_digest != SELECTED_ROWS_SHA256
            or policy_digest != SELECTION_POLICY_SHA256
            or _selection_policy_digest() != SELECTION_POLICY_SHA256
            or selection["selected_upstream_split_counts"] != {"test": 1, "train": 31}
            or metadata_digest != META_SHA256
            or metadata_bytes != META_SIZE_BYTES
            or metadata_row_count != 10_000
        ):
            raise MicrosoftAecFixtureError()
    elif (
        selection["algorithm"] != "test-fixed-v1"
        or selection["split"] != "synthetic"
        or selection["source_row_domain"] != "test_injection"
        or selection["selected_upstream_split_counts"] != {"synthetic": case_count}
        or metadata_row_count != case_count
    ):
        raise MicrosoftAecFixtureError()
    parsed_cases: list[_CasePin] = []
    seen_ids: set[str] = set()
    observed_audio_bytes = 0
    for ordinal, raw_case in enumerate(cases_raw):
        case = _mapping(raw_case, {"ordinal", "fileid", "metadata", "artifacts"})
        fileid = _string(case["fileid"], maximum=8)
        if not fileid.isdecimal() or str(int(fileid)) != fileid or fileid in seen_ids:
            raise MicrosoftAecFixtureError()
        seen_ids.add(fileid)
        raw_metadata = _mapping(case["metadata"], set(META_COLUMNS))
        metadata_row = {
            column: _string(raw_metadata[column], maximum=512, allow_empty=True)
            for column in META_COLUMNS
        }
        artifacts_raw = case["artifacts"]
        if (
            case["ordinal"] != ordinal + 1
            or metadata_row["fileid"] != fileid
            or metadata_row["split"] not in {"train", "test"}
            or not isinstance(artifacts_raw, list)
            or len(artifacts_raw) != len(SIGNAL_ROLES)
        ):
            raise MicrosoftAecFixtureError()
        _decimal_text(
            metadata_row["ser"], minimum=Decimal("-10"), maximum=Decimal("10")
        )
        _decimal_text(
            metadata_row["nearend_scale"], minimum=Decimal("0"), maximum=Decimal("100")
        )
        for key in ("is_farend_nonlinear", "is_farend_noisy", "is_nearend_noisy"):
            _bool_text(metadata_row[key])
        artifacts: list[_ArtifactPin] = []
        for expected_role, raw_artifact in zip(SIGNAL_ROLES, artifacts_raw):
            artifact = _mapping(
                raw_artifact,
                {
                    "role",
                    "relative_path",
                    "git_pointer_blob_sha1",
                    "lfs_oid_sha256",
                    "size_bytes",
                },
            )
            role = _string(artifact["role"], maximum=32)
            relative = _safe_relative(artifact["relative_path"])
            size_bytes = _integer(
                artifact["size_bytes"], minimum=44, maximum=_MAX_WAV_BYTES
            )
            pin = _ArtifactPin(
                role=role,
                relative_path=relative,
                git_pointer_blob_sha1=_sha1(artifact["git_pointer_blob_sha1"]),
                lfs_oid_sha256=_sha256(artifact["lfs_oid_sha256"]),
                size_bytes=size_bytes,
            )
            if role != expected_role or relative != _expected_artifact_path(
                role, fileid
            ):
                raise MicrosoftAecFixtureError()
            observed_audio_bytes += size_bytes
            artifacts.append(pin)
        parsed_cases.append(_CasePin(ordinal, fileid, metadata_row, tuple(artifacts)))
    if observed_audio_bytes != audio_bytes:
        raise MicrosoftAecFixtureError()
    identities = [
        json.dumps([case.fileid], ensure_ascii=False, separators=(",", ":"))
        for case in parsed_cases
    ]
    selected = hashlib.sha256(
        json.dumps(
            identities, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if selected != selected_digest:
        raise MicrosoftAecFixtureError()
    if production_evidence:
        if (
            fixture_id != FIXTURE_ID
            or case_count != CASE_COUNT
            or artifact_count != ARTIFACT_COUNT
            or audio_bytes != 40_965_456
            or raw_size_bytes <= 0
            or recipe != EXPECTED_LOCK_RECIPE_SHA256
        ):
            raise MicrosoftAecFixtureError()
    elif fixture_id == FIXTURE_ID:
        raise MicrosoftAecFixtureError()
    return _FixtureLock(
        fixture_id=fixture_id,
        recipe_sha256=recipe,
        meta_git_blob_sha1=_sha1(metadata["git_blob_sha1"]),
        metadata_sha256=metadata_digest,
        metadata_size_bytes=metadata_bytes,
        metadata_row_count=metadata_row_count,
        selected_rows_sha256=selected_digest,
        cases=tuple(parsed_cases),
        production_evidence=production_evidence,
        raw_sha256=raw_sha256,
        raw_size_bytes=raw_size_bytes,
    )


def _load_lock(
    path: Path | str = DEFAULT_LOCK,
    *,
    test_injection: TestFixtureInjection | None = None,
) -> _FixtureLock:
    supplied = Path(path).expanduser()
    production = test_injection is None
    if production:
        try:
            expected = DEFAULT_LOCK.resolve(strict=True)
            candidate = supplied.resolve(strict=True)
            loaded = read_regular_bounded(candidate, maximum_bytes=_MAX_LOCK_BYTES)
        except (OSError, RuntimeError, ValueError, BoundedReadError):
            raise MicrosoftAecFixtureError() from None
        if candidate != expected or supplied.absolute() != candidate:
            raise MicrosoftAecFixtureError()
        raw = loaded.data
        value = _strict_json(raw, maximum_bytes=_MAX_LOCK_BYTES)
    else:
        if (
            type(test_injection) is not TestFixtureInjection
            or _SAFE_ID_RE.fullmatch(test_injection.fixture_id) is None
            or not test_injection.fixture_id.startswith("synthetic-")
        ):
            raise MicrosoftAecFixtureError()
        value = test_injection.lock_value
        raw = _canonical_json(value, newline=True)
    parsed = _parse_lock_value(
        value,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        raw_size_bytes=len(raw),
        production_evidence=production,
    )
    if test_injection is not None and parsed.fixture_id != test_injection.fixture_id:
        raise MicrosoftAecFixtureError()
    return parsed


def _open_private_root(path: Path | str) -> tuple[Path, int, tuple[int, ...]]:
    candidate = Path(os.path.abspath(Path(path).expanduser()))
    descriptor = -1
    try:
        if candidate.resolve(strict=True) != candidate:
            raise MicrosoftAecFixtureError()
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(candidate, flags)
        opened = os.fstat(descriptor)
        lexical = candidate.lstat()
        snapshot = _file_snapshot(opened)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o700
            or opened.st_uid != os.getuid()
            or _file_snapshot(lexical) != snapshot
        ):
            raise MicrosoftAecFixtureError()
        return candidate, descriptor, snapshot
    except MicrosoftAecFixtureError:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise
    except (OSError, RuntimeError, ValueError):
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise MicrosoftAecFixtureError() from None


def _verify_private_root(
    path: Path, descriptor: int, expected: tuple[int, ...]
) -> None:
    try:
        opened = os.fstat(descriptor)
        lexical = path.lstat()
        if (
            not stat.S_ISDIR(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o700
            or opened.st_uid != os.getuid()
            or _file_snapshot(opened) != expected
            or _file_snapshot(lexical) != expected
            or path.resolve(strict=True) != path
        ):
            raise MicrosoftAecFixtureError()
    except MicrosoftAecFixtureError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise MicrosoftAecFixtureError() from None


def _read_relative_private(
    root_fd: int,
    relative: str,
    *,
    expected_size: int,
    expected_sha256: str,
    maximum_bytes: int,
) -> _BoundSource:
    path = PurePosixPath(_safe_relative(relative))
    directory_fd = os.dup(root_fd)
    descriptor = -1
    try:
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        for component in path.parts[:-1]:
            before = os.stat(component, dir_fd=directory_fd, follow_symlinks=False)
            child = os.open(component, directory_flags, dir_fd=directory_fd)
            opened = os.fstat(child)
            if (
                not stat.S_ISDIR(before.st_mode)
                or _file_snapshot(before) != _file_snapshot(opened)
                or stat.S_IMODE(opened.st_mode) != 0o700
                or opened.st_uid != os.getuid()
            ):
                os.close(child)
                raise MicrosoftAecFixtureError()
            os.close(directory_fd)
            directory_fd = child
        leaf = path.parts[-1]
        before = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
            or before.st_size != expected_size
            or not 0 < before.st_size <= maximum_bytes
        ):
            raise MicrosoftAecFixtureError()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(leaf, flags, dir_fd=directory_fd)
        opened = os.fstat(descriptor)
        if _file_snapshot(opened) != _file_snapshot(before):
            raise MicrosoftAecFixtureError()
        raw = bytearray()
        while len(raw) <= maximum_bytes:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
        entry = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        digest = hashlib.sha256(raw).hexdigest()
        snapshot = _file_snapshot(after)
        if (
            len(raw) != expected_size
            or digest != expected_sha256
            or _file_snapshot(before) != snapshot
            or _file_snapshot(entry) != snapshot
        ):
            raise MicrosoftAecFixtureError()
        return _BoundSource(
            relative_path=relative,
            raw=bytes(raw),
            sha256=digest,
            size_bytes=len(raw),
            samples=0,
            snapshot=snapshot,
        )
    except MicrosoftAecFixtureError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise MicrosoftAecFixtureError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            os.close(directory_fd)
        except OSError:
            pass


def _riff_pcm16_mono(raw: bytes) -> np.ndarray:
    if len(raw) < 42 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise MicrosoftAecFixtureError()
    if struct.unpack_from("<I", raw, 4)[0] + 8 != len(raw):
        raise MicrosoftAecFixtureError()
    offset = 12
    format_chunk: bytes | None = None
    data_chunk: bytes | None = None
    while offset < len(raw):
        if offset + 8 > len(raw):
            raise MicrosoftAecFixtureError()
        chunk_id = raw[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", raw, offset + 4)[0]
        offset += 8
        end = offset + chunk_size
        padded_end = end + (chunk_size & 1)
        if end > len(raw) or padded_end > len(raw):
            raise MicrosoftAecFixtureError()
        payload = raw[offset:end]
        if chunk_id == b"fmt ":
            if format_chunk is not None:
                raise MicrosoftAecFixtureError()
            format_chunk = payload
        elif chunk_id == b"data":
            if data_chunk is not None:
                raise MicrosoftAecFixtureError()
            data_chunk = payload
        offset = padded_end
    if (
        offset != len(raw)
        or format_chunk is None
        or data_chunk is None
        or len(format_chunk) != 16
        or not data_chunk
        or len(data_chunk) % 2
    ):
        raise MicrosoftAecFixtureError()
    encoding, channels, rate, byte_rate, align, bits = struct.unpack(
        "<HHIIHH", format_chunk
    )
    if (
        encoding != 1
        or channels != 1
        or rate != SAMPLE_RATE_HZ
        or byte_rate != SAMPLE_RATE_HZ * 2
        or align != 2
        or bits != 16
    ):
        raise MicrosoftAecFixtureError()
    pcm = np.frombuffer(data_chunk, dtype="<i2")
    if pcm.size <= 0:
        raise MicrosoftAecFixtureError()
    return pcm


def decode_microsoft_aec_wav(raw: bytes) -> np.ndarray:
    """Decode one fixture WAV to owned finite float32 in [-1, 1)."""

    pcm = _riff_pcm16_mono(raw)
    result = pcm.astype(np.float32)
    result *= np.float32(1.0 / 32768.0)
    if not np.all(np.isfinite(result)):
        raise MicrosoftAecFixtureError()
    return result


def _parse_meta(
    raw: bytes, *, expected_rows: int, production_evidence: bool
) -> tuple[Mapping[str, str], ...]:
    try:
        text = raw.decode("utf-8")
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        header = next(reader)
        rows: list[Mapping[str, str]] = []
        identities: set[str] = set()
        if tuple(header) != META_COLUMNS:
            raise MicrosoftAecFixtureError()
        for values in reader:
            if len(values) != len(META_COLUMNS):
                raise MicrosoftAecFixtureError()
            row = {
                column: _string(value, maximum=1024, allow_empty=True)
                for column, value in zip(META_COLUMNS, values)
            }
            fileid = row["fileid"]
            if (
                not fileid.isdecimal()
                or str(int(fileid)) != fileid
                or fileid in identities
            ):
                raise MicrosoftAecFixtureError()
            identities.add(fileid)
            rows.append(row)
    except MicrosoftAecFixtureError:
        raise
    except (csv.Error, StopIteration, UnicodeError, ValueError, OverflowError):
        raise MicrosoftAecFixtureError() from None
    if len(rows) != expected_rows or (
        production_evidence and identities != {str(index) for index in range(10_000)}
    ):
        raise MicrosoftAecFixtureError()
    return tuple(rows)


def _select_locked_meta(
    rows: tuple[Mapping[str, str], ...], fixture_lock: _FixtureLock
) -> tuple[Mapping[str, str], ...]:
    if not fixture_lock.production_evidence:
        selected = rows
        if len(selected) != len(fixture_lock.cases) or any(
            dict(row) != dict(pin.metadata)
            for row, pin in zip(selected, fixture_lock.cases)
        ):
            raise MicrosoftAecFixtureError()
        return selected
    selected = _select_production_rows(rows)
    if (
        len(selected) != len(fixture_lock.cases)
        or _selected_rows_digest(selected) != fixture_lock.selected_rows_sha256
        or any(
            dict(row) != dict(pin.metadata)
            for row, pin in zip(selected, fixture_lock.cases)
        )
    ):
        raise MicrosoftAecFixtureError()
    return selected


def _source_contract_sha256(
    fixture_lock: _FixtureLock,
    metadata: _BoundSource,
    sources: Mapping[str, _BoundSource],
) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "fixture_id": fixture_lock.fixture_id,
                "lock_recipe_sha256": fixture_lock.recipe_sha256,
                "metadata": {
                    "sha256": metadata.sha256,
                    "size_bytes": metadata.size_bytes,
                },
                "selected_rows_sha256": fixture_lock.selected_rows_sha256,
                "signals": [
                    {
                        "role": artifact.role,
                        "sha256": sources[artifact.relative_path].sha256,
                        "size_bytes": sources[artifact.relative_path].size_bytes,
                        "samples": sources[artifact.relative_path].samples,
                    }
                    for case in fixture_lock.cases
                    for artifact in case.artifacts
                ],
            }
        )
    ).hexdigest()


def _preparer_closure() -> tuple[dict[str, object], ...]:
    root = Path(__file__).resolve().parents[1]
    rows: list[dict[str, object]] = []
    for relative in _PREPARER_FILES:
        try:
            loaded = read_regular_bounded(
                root.joinpath(*PurePosixPath(relative).parts),
                maximum_bytes=4 * 1024 * 1024,
            )
        except BoundedReadError:
            raise MicrosoftAecFixtureError() from None
        rows.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(loaded.data).hexdigest(),
                "size_bytes": len(loaded.data),
            }
        )
    return tuple(rows)


@dataclass(slots=True)
class _SourceInventory:
    root: Path = field(repr=False)
    descriptor: int = field(repr=False)
    root_snapshot: tuple[int, ...] = field(repr=False)
    head: _BoundSource = field(repr=False)
    metadata: _BoundSource = field(repr=False)
    sources: Mapping[str, _BoundSource] = field(repr=False)


def _load_source_inventory(
    source_dir: Path | str, fixture_lock: _FixtureLock
) -> _SourceInventory:
    root, descriptor, root_snapshot = _open_private_root(source_dir)
    try:
        head_raw = (REVISION + "\n").encode("ascii")
        head = _read_relative_private(
            descriptor,
            ".git/HEAD",
            expected_size=len(head_raw),
            expected_sha256=hashlib.sha256(head_raw).hexdigest(),
            maximum_bytes=128,
        )
        metadata = _read_relative_private(
            descriptor,
            META_RELATIVE_PATH,
            expected_size=fixture_lock.metadata_size_bytes,
            expected_sha256=fixture_lock.metadata_sha256,
            maximum_bytes=_MAX_META_BYTES,
        )
        selected = _select_locked_meta(
            _parse_meta(
                metadata.raw,
                expected_rows=fixture_lock.metadata_row_count,
                production_evidence=fixture_lock.production_evidence,
            ),
            fixture_lock,
        )
        if any(
            dict(row) != dict(pin.metadata)
            for row, pin in zip(selected, fixture_lock.cases)
        ):
            raise MicrosoftAecFixtureError()
        sources: dict[str, _BoundSource] = {}
        for case in fixture_lock.cases:
            case_samples: int | None = None
            for artifact in case.artifacts:
                loaded = _read_relative_private(
                    descriptor,
                    artifact.relative_path,
                    expected_size=artifact.size_bytes,
                    expected_sha256=artifact.lfs_oid_sha256,
                    maximum_bytes=_MAX_WAV_BYTES,
                )
                samples = int(_riff_pcm16_mono(loaded.raw).size)
                if (
                    samples <= 0
                    or samples > SAMPLE_RATE_HZ * 11
                    or (
                        fixture_lock.production_evidence
                        and samples not in {159_999, 160_000}
                    )
                    or (-samples) % CAPTURE_BLOCK_SAMPLES > 1
                ):
                    raise MicrosoftAecFixtureError()
                if case_samples is None:
                    case_samples = samples
                elif samples != case_samples:
                    raise MicrosoftAecFixtureError()
                sources[artifact.relative_path] = _BoundSource(
                    relative_path=loaded.relative_path,
                    raw=loaded.raw,
                    sha256=loaded.sha256,
                    size_bytes=loaded.size_bytes,
                    samples=samples,
                    snapshot=loaded.snapshot,
                )
        if len(sources) != len(fixture_lock.cases) * len(SIGNAL_ROLES):
            raise MicrosoftAecFixtureError()
        _verify_private_root(root, descriptor, root_snapshot)
        return _SourceInventory(
            root,
            descriptor,
            root_snapshot,
            head,
            metadata,
            sources,
        )
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _verify_source_inventory(
    inventory: _SourceInventory, fixture_lock: _FixtureLock
) -> None:
    _verify_private_root(inventory.root, inventory.descriptor, inventory.root_snapshot)
    head_raw = (REVISION + "\n").encode("ascii")
    head = _read_relative_private(
        inventory.descriptor,
        ".git/HEAD",
        expected_size=len(head_raw),
        expected_sha256=hashlib.sha256(head_raw).hexdigest(),
        maximum_bytes=128,
    )
    metadata = _read_relative_private(
        inventory.descriptor,
        META_RELATIVE_PATH,
        expected_size=fixture_lock.metadata_size_bytes,
        expected_sha256=fixture_lock.metadata_sha256,
        maximum_bytes=_MAX_META_BYTES,
    )
    if head != inventory.head or metadata != inventory.metadata:
        raise MicrosoftAecFixtureError()
    for case in fixture_lock.cases:
        for artifact in case.artifacts:
            current = _read_relative_private(
                inventory.descriptor,
                artifact.relative_path,
                expected_size=artifact.size_bytes,
                expected_sha256=artifact.lfs_oid_sha256,
                maximum_bytes=_MAX_WAV_BYTES,
            )
            saved = inventory.sources[artifact.relative_path]
            if (
                current.raw != saved.raw
                or current.sha256 != saved.sha256
                or current.snapshot != saved.snapshot
                or _riff_pcm16_mono(current.raw).size != saved.samples
            ):
                raise MicrosoftAecFixtureError()
    _verify_private_root(inventory.root, inventory.descriptor, inventory.root_snapshot)


def _has_git_ancestor(path: Path) -> bool:
    for parent in (path, *path.parents):
        marker = parent / ".git"
        try:
            metadata = marker.lstat()
        except FileNotFoundError:
            continue
        except OSError:
            raise MicrosoftAecFixtureError() from None
        if stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode):
            return True
    return False


def _new_private_output(
    output_dir: Path | str, *, source_root: Path
) -> tuple[Path, int, tuple[int, int, int, int]]:
    supplied = Path(output_dir).expanduser()
    if not supplied.is_absolute():
        raise MicrosoftAecFixtureError()
    candidate = Path(os.path.abspath(supplied))
    parent_fd = -1
    descriptor = -1
    try:
        parent = candidate.parent.resolve(strict=True)
        if (
            not candidate.name
            or candidate.name in {".", ".."}
            or candidate.exists()
            or candidate.is_symlink()
            or candidate == source_root
            or candidate in source_root.parents
            or source_root in candidate.parents
            or _has_git_ancestor(parent)
        ):
            raise MicrosoftAecFixtureError()
        stable_parent, parent_fd, _parent_snapshot = _open_private_root(parent)
        if stable_parent != parent or candidate.parent != parent:
            raise MicrosoftAecFixtureError()
        os.mkdir(candidate.name, mode=0o700, dir_fd=parent_fd)
        created = os.stat(candidate.name, dir_fd=parent_fd, follow_symlinks=False)
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(candidate.name, flags, dir_fd=parent_fd)
        os.fchmod(descriptor, 0o700)
        opened = os.fstat(descriptor)
        identity = _directory_locator(opened)
        if (
            not stat.S_ISDIR(created.st_mode)
            or _directory_locator(created) != identity
            or stat.S_IMODE(opened.st_mode) != 0o700
            or opened.st_uid != os.getuid()
        ):
            raise MicrosoftAecFixtureError()
        os.fsync(descriptor)
        os.fsync(parent_fd)
        return candidate, descriptor, identity
    except MicrosoftAecFixtureError:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise
    except (OSError, RuntimeError, ValueError):
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise MicrosoftAecFixtureError() from None
    finally:
        if parent_fd >= 0:
            try:
                os.close(parent_fd)
            except OSError:
                pass


def _verify_output_directory(
    path: Path, descriptor: int, identity: tuple[int, int, int, int]
) -> None:
    try:
        opened = os.fstat(descriptor)
        lexical = path.lstat()
        if (
            _directory_locator(opened) != identity
            or _directory_locator(lexical) != identity
            or not stat.S_ISDIR(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o700
            or opened.st_uid != os.getuid()
            or path.resolve(strict=True) != path
        ):
            raise MicrosoftAecFixtureError()
    except MicrosoftAecFixtureError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise MicrosoftAecFixtureError() from None


def _read_private_at(
    directory_fd: int,
    name: str,
    *,
    maximum_bytes: int,
    expected: PrivateFileIdentity | None = None,
) -> tuple[bytes, PrivateFileIdentity]:
    leaf = _safe_leaf(name)
    descriptor = -1
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(leaf, flags, dir_fd=directory_fd)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
            or not 0 < before.st_size <= maximum_bytes
        ):
            raise MicrosoftAecFixtureError()
        raw = bytearray()
        while len(raw) <= maximum_bytes:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
        entry = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        current = PrivateFileIdentity(
            leaf,
            len(raw),
            hashlib.sha256(raw).hexdigest(),
            _file_snapshot(after),
        )
        if (
            len(raw) != before.st_size
            or _file_snapshot(before) != _file_snapshot(after)
            or _file_snapshot(entry) != _file_snapshot(after)
            or (expected is not None and current != expected)
        ):
            raise MicrosoftAecFixtureError()
        return bytes(raw), current
    except MicrosoftAecFixtureError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise MicrosoftAecFixtureError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _write_private_at(directory_fd: int, name: str, raw: bytes) -> PrivateFileIdentity:
    leaf = _safe_leaf(name)
    if not isinstance(raw, bytes) or not raw:
        raise MicrosoftAecFixtureError()
    descriptor = -1
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(leaf, flags, 0o600, dir_fd=directory_fd)
        os.fchmod(descriptor, 0o600)
        view = memoryview(raw)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise MicrosoftAecFixtureError()
            written += count
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        result = PrivateFileIdentity(
            leaf,
            len(raw),
            hashlib.sha256(raw).hexdigest(),
            _file_snapshot(opened),
        )
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_uid != os.getuid()
            or opened.st_nlink != 1
            or opened.st_size != len(raw)
        ):
            raise MicrosoftAecFixtureError()
    except MicrosoftAecFixtureError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise MicrosoftAecFixtureError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
    _read_private_at(directory_fd, leaf, maximum_bytes=len(raw), expected=result)
    return result


def _bundle_leaf(case_index: int, role: str) -> str:
    suffix = {
        "echo_signal": "echo",
        "farend_speech": "far",
        "nearend_mic_signal": "mic",
        "nearend_speech": "near",
    }.get(role)
    if suffix is None or not 0 <= case_index < CASE_COUNT:
        raise MicrosoftAecFixtureError()
    return f"case-{case_index:02d}-{suffix}.wav"


def _artifact_set_sha256(cases: Sequence[MicrosoftAecCase]) -> str:
    return hashlib.sha256(
        _canonical_json(
            [
                {
                    "case_id": case.case_id,
                    "files": [
                        {
                            "name": artifact.name,
                            "role": artifact.role,
                            "samples": artifact.samples,
                            "sha256": artifact.sha256,
                            "size_bytes": artifact.size_bytes,
                        }
                        for artifact in case.artifacts
                    ],
                }
                for case in cases
            ]
        )
    ).hexdigest()


def _expected_entry_names(case_count: int, *, receipt: bool) -> set[str]:
    result = {MANIFEST_FILENAME}
    for index in range(case_count):
        result.update(_bundle_leaf(index, role) for role in SIGNAL_ROLES)
    if receipt:
        result.add(PREPARATION_RECEIPT_FILENAME)
    return result


def _exact_entries(directory_fd: int, expected: set[str]) -> None:
    try:
        entries = os.listdir(directory_fd)
    except OSError:
        raise MicrosoftAecFixtureError() from None
    if len(entries) != len(set(entries)) or set(entries) != expected:
        raise MicrosoftAecFixtureError()


def _build_manifest(
    fixture_lock: _FixtureLock,
    source_contract_sha256: str,
    written_cases: Sequence[MicrosoftAecCase],
) -> dict[str, object]:
    audio_bytes = sum(
        artifact.size_bytes for case in written_cases for artifact in case.artifacts
    )
    return {
        "artifact_count": len(written_cases) * len(SIGNAL_ROLES),
        "audio_bytes": audio_bytes,
        "case_count": len(written_cases),
        "cases": [
            {
                "artifacts": [
                    {
                        "file": artifact.name,
                        "role": artifact.role,
                        "samples": artifact.samples,
                        "sha256": artifact.sha256,
                        "size_bytes": artifact.size_bytes,
                    }
                    for artifact in case.artifacts
                ],
                "case_id": case.case_id,
                "is_farend_noisy": case.is_farend_noisy,
                "is_farend_nonlinear": case.is_farend_nonlinear,
                "is_nearend_noisy": case.is_nearend_noisy,
                "nearend_scale": case.nearend_scale,
                "ordinal": case.ordinal,
                "rank_sha256": case.rank_sha256,
                "ser": case.ser,
                "source_samples": case.artifacts[0].samples,
                "split": case.split,
                "terminal_padding_samples": (-case.artifacts[0].samples)
                % CAPTURE_BLOCK_SAMPLES,
            }
            for case in written_cases
        ],
        "evidence_scope": {
            "aec_component_replay": True,
            "aecmos": False,
            "capture_or_device": False,
            "live_barge_or_endpoint": False,
            "promotion_or_default_authority": False,
        },
        "fixture_id": fixture_lock.fixture_id,
        "kind": MANIFEST_KIND,
        "lock_recipe_sha256": fixture_lock.recipe_sha256,
        "production_evidence": fixture_lock.production_evidence,
        "sample_format": {
            "channels": 1,
            "encoding": "pcm_s16le",
            "sample_rate_hz": SAMPLE_RATE_HZ,
        },
        "schema_version": SCHEMA_VERSION,
        "selected_rows_sha256": fixture_lock.selected_rows_sha256,
        "source_contract_sha256": source_contract_sha256,
    }


def _stage_terminal_receipt(
    directory_fd: int, raw: bytes
) -> tuple[int, PrivateFileIdentity]:
    descriptor = -1
    prepared = False
    try:
        try:
            os.stat(
                PREPARATION_RECEIPT_FILENAME,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise MicrosoftAecFixtureError()
        temporary = getattr(os, "O_TMPFILE", 0)
        if not temporary:
            raise MicrosoftAecFixtureError()
        descriptor = os.open(
            ".",
            os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | temporary,
            0o600,
            dir_fd=directory_fd,
        )
        os.fchmod(descriptor, 0o600)
        view = memoryview(raw)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise MicrosoftAecFixtureError()
            written += count
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_uid != os.getuid()
            or opened.st_nlink != 0
            or opened.st_size != len(raw)
        ):
            raise MicrosoftAecFixtureError()
        staged = PrivateFileIdentity(
            PREPARATION_RECEIPT_FILENAME,
            len(raw),
            hashlib.sha256(raw).hexdigest(),
            _file_snapshot(opened),
        )
        _verify_staged_receipt(descriptor, staged, raw)
        prepared = True
        return descriptor, staged
    except MicrosoftAecFixtureError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise MicrosoftAecFixtureError() from None
    finally:
        if not prepared and descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _verify_staged_receipt(
    descriptor: int, expected: PrivateFileIdentity, encoded: bytes
) -> None:
    try:
        before = os.fstat(descriptor)
        if (
            expected.name != PREPARATION_RECEIPT_FILENAME
            or not 0 < expected.size_bytes <= _MAX_RECEIPT_BYTES
            or not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_uid != os.getuid()
            or before.st_nlink != 0
            or _file_snapshot(before) != expected.snapshot
        ):
            raise MicrosoftAecFixtureError()
        os.lseek(descriptor, 0, os.SEEK_SET)
        raw = bytearray()
        while len(raw) <= _MAX_RECEIPT_BYTES:
            chunk = os.read(
                descriptor,
                min(65_536, _MAX_RECEIPT_BYTES + 1 - len(raw)),
            )
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
        current = PrivateFileIdentity(
            PREPARATION_RECEIPT_FILENAME,
            len(raw),
            hashlib.sha256(raw).hexdigest(),
            _file_snapshot(after),
        )
        if (
            bytes(raw) != encoded
            or current != expected
            or _file_snapshot(before) != _file_snapshot(after)
        ):
            raise MicrosoftAecFixtureError()
    except MicrosoftAecFixtureError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise MicrosoftAecFixtureError() from None


def _commit_signal_numbers() -> tuple[int, ...]:
    return tuple(
        signum
        for signum in (
            getattr(signal, "SIGINT", None),
            getattr(signal, "SIGHUP", None),
            getattr(signal, "SIGTERM", None),
        )
        if isinstance(signum, int)
    )


def _terminal_commit_guard(callback: Callable[[], None]) -> None:
    callback()


def _commit_terminal_receipt(
    output_path: Path,
    directory_fd: int,
    directory_identity: tuple[int, int, int, int],
    staged_fd: int,
    staged: PrivateFileIdentity,
    encoded: bytes,
    state: _CommitState,
    *,
    case_count: int,
    commit_guard: Callable[[], None],
) -> None:
    if state.committed or not hasattr(signal, "pthread_sigmask"):
        raise MicrosoftAecFixtureError()
    _terminal_commit_guard(commit_guard)
    _verify_staged_receipt(staged_fd, staged, encoded)
    _verify_output_directory(output_path, directory_fd, directory_identity)
    expected_entries = _expected_entry_names(case_count, receipt=False)
    _exact_entries(directory_fd, expected_entries)
    current = os.fstat(staged_fd)
    if (
        not stat.S_ISREG(current.st_mode)
        or stat.S_IMODE(current.st_mode) != 0o600
        or current.st_uid != os.getuid()
        or current.st_nlink != 0
        or current.st_size != len(encoded)
        or _file_snapshot(current) != staged.snapshot
    ):
        raise MicrosoftAecFixtureError()
    try:
        # Make every already-published artifact name durable before the terminal
        # receipt can become visible.  The receipt link remains the commit point;
        # a directory-fsync failure is therefore a pre-commit failure.
        os.fsync(directory_fd)
    except OSError:
        raise MicrosoftAecFixtureError() from None
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, _commit_signal_numbers())
    link_returned = False
    try:
        try:
            os.link(
                f"/proc/self/fd/{staged_fd}",
                PREPARATION_RECEIPT_FILENAME,
                dst_dir_fd=directory_fd,
                follow_symlinks=True,
            )
            link_returned = True
        finally:
            if link_returned:
                state.committed = True
            else:
                try:
                    published = os.stat(
                        PREPARATION_RECEIPT_FILENAME,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                    current = os.fstat(staged_fd)
                except OSError:
                    pass
                else:
                    if (
                        (published.st_dev, published.st_ino)
                        == (current.st_dev, current.st_ino)
                        and stat.S_ISREG(published.st_mode)
                        and stat.S_IMODE(published.st_mode) == 0o600
                        and published.st_uid == os.getuid()
                        and published.st_nlink == 1
                        and published.st_size == len(encoded)
                        and current.st_nlink == 1
                    ):
                        state.committed = True
    finally:
        if state.committed:
            try:
                os.fsync(directory_fd)
            except BaseException:
                # The receipt link is already the irreversible commit point.
                # Best-effort durability failure cannot turn a valid published
                # bundle into a reported pre-commit failure.
                pass
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    if not state.committed:
        raise MicrosoftAecFixtureError()


def _verify_written_files(
    output_path: Path,
    directory_fd: int,
    directory_identity: tuple[int, int, int, int],
    written: Mapping[str, PrivateFileIdentity],
    *,
    case_count: int,
) -> None:
    _verify_output_directory(output_path, directory_fd, directory_identity)
    _exact_entries(directory_fd, _expected_entry_names(case_count, receipt=False))
    if set(written) != _expected_entry_names(case_count, receipt=False):
        raise MicrosoftAecFixtureError()
    for name, identity in written.items():
        maximum = _MAX_MANIFEST_BYTES if name == MANIFEST_FILENAME else _MAX_WAV_BYTES
        _read_private_at(
            directory_fd,
            name,
            maximum_bytes=maximum,
            expected=identity,
        )
    _verify_output_directory(output_path, directory_fd, directory_identity)


def prepare_microsoft_aec_fixture(
    source_dir: Path | str,
    output_dir: Path | str,
    *,
    accepted_terms: Sequence[str],
    lock_path: Path | str = DEFAULT_LOCK,
    _test_source: TestFixtureInjection | None = None,
    _commit_state: _CommitState | None = None,
) -> PreparedMicrosoftAecFixture:
    """Verify and privately copy the exact selected synchronized signals."""

    if (
        isinstance(accepted_terms, (str, bytes))
        or tuple(dict.fromkeys(accepted_terms)) != tuple(accepted_terms)
        or set(accepted_terms) != set(LICENSE_ACCEPTANCE_IDS)
    ):
        raise MicrosoftAecFixtureError()
    fixture_lock = _load_lock(lock_path, test_injection=_test_source)
    closure = _preparer_closure()
    inventory = _load_source_inventory(source_dir, fixture_lock)
    output_fd = -1
    staged_fd = -1
    state = _CommitState() if _commit_state is None else _commit_state
    if type(state) is not _CommitState or state.committed:
        raise MicrosoftAecFixtureError()
    prepared: PreparedMicrosoftAecFixture | None = None
    try:
        source_contract = _source_contract_sha256(
            fixture_lock,
            inventory.metadata,
            inventory.sources,
        )
        output_path, output_fd, output_identity = _new_private_output(
            output_dir, source_root=inventory.root
        )
        written: dict[str, PrivateFileIdentity] = {}
        written_cases: list[MicrosoftAecCase] = []
        for index, pin in enumerate(fixture_lock.cases):
            artifacts: list[MicrosoftAecArtifact] = []
            source_samples: int | None = None
            for artifact_pin in pin.artifacts:
                source = inventory.sources[artifact_pin.relative_path]
                leaf = _bundle_leaf(index, artifact_pin.role)
                identity = _write_private_at(output_fd, leaf, source.raw)
                written[leaf] = identity
                artifacts.append(
                    MicrosoftAecArtifact(
                        name=leaf,
                        role=artifact_pin.role,
                        size_bytes=identity.size_bytes,
                        sha256=identity.sha256,
                        samples=source.samples,
                        snapshot=identity.snapshot,
                    )
                )
                if source_samples is None:
                    source_samples = source.samples
                elif source.samples != source_samples:
                    raise MicrosoftAecFixtureError()
            metadata = pin.metadata
            written_cases.append(
                MicrosoftAecCase(
                    case_id=f"aec-case-{index:02d}",
                    ordinal=index,
                    rank_sha256=_rank(SELECTION_SEED, pin.fileid),
                    nearend_scale=_decimal_text(
                        metadata["nearend_scale"],
                        minimum=Decimal("0"),
                        maximum=Decimal("100"),
                    ),
                    ser=_decimal_text(
                        metadata["ser"],
                        minimum=Decimal("-10"),
                        maximum=Decimal("10"),
                    ),
                    is_farend_nonlinear=_bool_text(metadata["is_farend_nonlinear"]),
                    is_farend_noisy=_bool_text(metadata["is_farend_noisy"]),
                    is_nearend_noisy=_bool_text(metadata["is_nearend_noisy"]),
                    split=metadata["split"],
                    artifacts=tuple(artifacts),
                )
            )
        manifest = _build_manifest(fixture_lock, source_contract, written_cases)
        manifest_raw = _canonical_json(manifest, newline=True)
        manifest_identity = _write_private_at(
            output_fd, MANIFEST_FILENAME, manifest_raw
        )
        written[MANIFEST_FILENAME] = manifest_identity
        artifact_set = _artifact_set_sha256(written_cases)
        audio_bytes = sum(
            artifact.size_bytes for case in written_cases for artifact in case.artifacts
        )
        receipt = {
            "accepted_terms": list(LICENSE_ACCEPTANCE_IDS),
            "artifact_count": len(written_cases) * len(SIGNAL_ROLES),
            "artifact_set_sha256": artifact_set,
            "audio_bytes": audio_bytes,
            "evidence_scope": manifest["evidence_scope"],
            "fixture_id": fixture_lock.fixture_id,
            "kind": RECEIPT_KIND,
            "lock": {
                "file": DEFAULT_LOCK.name,
                "raw_sha256": fixture_lock.raw_sha256,
                "recipe_sha256": fixture_lock.recipe_sha256,
                "size_bytes": fixture_lock.raw_size_bytes,
            },
            "manifest": {
                "file": MANIFEST_FILENAME,
                "sha256": manifest_identity.sha256,
                "size_bytes": manifest_identity.size_bytes,
            },
            "preparer_files": list(closure),
            "production_evidence": fixture_lock.production_evidence,
            "schema_version": SCHEMA_VERSION,
            "selected_rows_sha256": fixture_lock.selected_rows_sha256,
            "source_contract_sha256": source_contract,
        }
        receipt_raw = _canonical_json(receipt, newline=True)
        staged_fd, staged = _stage_terminal_receipt(output_fd, receipt_raw)
        prepared = PreparedMicrosoftAecFixture(
            path=output_path,
            fixture_id=fixture_lock.fixture_id,
            production_evidence=fixture_lock.production_evidence,
            manifest_sha256=manifest_identity.sha256,
            receipt_sha256=staged.sha256,
            source_contract_sha256=source_contract,
            selected_rows_sha256=fixture_lock.selected_rows_sha256,
            artifact_count=len(written_cases) * len(SIGNAL_ROLES),
            audio_bytes=audio_bytes,
        )

        def close_guard() -> None:
            if _load_lock(lock_path, test_injection=_test_source) != fixture_lock:
                raise MicrosoftAecFixtureError()
            _verify_source_inventory(inventory, fixture_lock)
            if _preparer_closure() != closure:
                raise MicrosoftAecFixtureError()
            _verify_written_files(
                output_path,
                output_fd,
                output_identity,
                written,
                case_count=len(fixture_lock.cases),
            )

        _commit_terminal_receipt(
            output_path,
            output_fd,
            output_identity,
            staged_fd,
            staged,
            receipt_raw,
            state,
            case_count=len(fixture_lock.cases),
            commit_guard=close_guard,
        )
        return prepared
    except BaseException as error:
        if state.committed and prepared is not None:
            return prepared
        if isinstance(error, (KeyboardInterrupt, _LifecycleSignal)):
            raise
        if isinstance(error, MicrosoftAecFixtureError):
            raise
        raise MicrosoftAecFixtureError() from None
    finally:
        for descriptor in (staged_fd, output_fd, inventory.descriptor):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass


def _parse_manifest_case(
    raw: object,
    *,
    index: int,
    directory_fd: int,
    fixture_lock: _FixtureLock | None,
) -> MicrosoftAecCase:
    value = _mapping(
        raw,
        {
            "artifacts",
            "case_id",
            "is_farend_noisy",
            "is_farend_nonlinear",
            "is_nearend_noisy",
            "nearend_scale",
            "ordinal",
            "rank_sha256",
            "ser",
            "source_samples",
            "split",
            "terminal_padding_samples",
        },
    )
    case_id = _string(value["case_id"], maximum=32)
    source_samples = _integer(
        value["source_samples"], minimum=1, maximum=SAMPLE_RATE_HZ * 11
    )
    padding = _integer(
        value["terminal_padding_samples"], minimum=0, maximum=CAPTURE_BLOCK_SAMPLES - 1
    )
    artifacts_raw = value["artifacts"]
    if (
        case_id != f"aec-case-{index:02d}"
        or value["ordinal"] != index
        or _sha256(value["rank_sha256"]) == "0" * 64
        or value["split"] not in {"train", "test"}
        or not isinstance(value["is_farend_noisy"], bool)
        or not isinstance(value["is_farend_nonlinear"], bool)
        or not isinstance(value["is_nearend_noisy"], bool)
        or padding != (-source_samples) % CAPTURE_BLOCK_SAMPLES
        or padding > 1
        or not isinstance(artifacts_raw, list)
        or len(artifacts_raw) != len(SIGNAL_ROLES)
    ):
        raise MicrosoftAecFixtureError()
    nearend_scale = _decimal_text(
        _string(value["nearend_scale"], maximum=64),
        minimum=Decimal("0"),
        maximum=Decimal("100"),
    )
    ser = _decimal_text(
        _string(value["ser"], maximum=64),
        minimum=Decimal("-10"),
        maximum=Decimal("10"),
    )
    artifacts: list[MicrosoftAecArtifact] = []
    pin_case = fixture_lock.cases[index] if fixture_lock is not None else None
    for role, raw_artifact in zip(SIGNAL_ROLES, artifacts_raw):
        artifact = _mapping(
            raw_artifact, {"file", "role", "samples", "sha256", "size_bytes"}
        )
        name = _safe_leaf(artifact["file"])
        size_bytes = _integer(
            artifact["size_bytes"], minimum=44, maximum=_MAX_WAV_BYTES
        )
        if (
            artifact["role"] != role
            or name != _bundle_leaf(index, role)
            or artifact["samples"] != source_samples
        ):
            raise MicrosoftAecFixtureError()
        raw_audio, identity = _read_private_at(
            directory_fd, name, maximum_bytes=_MAX_WAV_BYTES
        )
        samples = int(_riff_pcm16_mono(raw_audio).size)
        if (
            identity.size_bytes != size_bytes
            or identity.sha256 != _sha256(artifact["sha256"])
            or samples != source_samples
        ):
            raise MicrosoftAecFixtureError()
        if pin_case is not None:
            pin = pin_case.artifacts[SIGNAL_ROLES.index(role)]
            if (
                identity.size_bytes != pin.size_bytes
                or identity.sha256 != pin.lfs_oid_sha256
            ):
                raise MicrosoftAecFixtureError()
        artifacts.append(
            MicrosoftAecArtifact(
                name=name,
                role=role,
                size_bytes=identity.size_bytes,
                sha256=identity.sha256,
                samples=samples,
                snapshot=identity.snapshot,
            )
        )
    if pin_case is not None:
        if (
            value["rank_sha256"] != _rank(SELECTION_SEED, pin_case.fileid)
            or nearend_scale != pin_case.metadata["nearend_scale"]
            or ser != pin_case.metadata["ser"]
            or value["split"] != pin_case.metadata["split"]
            or value["is_farend_noisy"]
            != _bool_text(pin_case.metadata["is_farend_noisy"])
            or value["is_farend_nonlinear"]
            != _bool_text(pin_case.metadata["is_farend_nonlinear"])
            or value["is_nearend_noisy"]
            != _bool_text(pin_case.metadata["is_nearend_noisy"])
        ):
            raise MicrosoftAecFixtureError()
    return MicrosoftAecCase(
        case_id=case_id,
        ordinal=index,
        rank_sha256=_sha256(value["rank_sha256"]),
        nearend_scale=nearend_scale,
        ser=ser,
        is_farend_nonlinear=value["is_farend_nonlinear"],
        is_farend_noisy=value["is_farend_noisy"],
        is_nearend_noisy=value["is_nearend_noisy"],
        split=_string(value["split"], maximum=8),
        artifacts=tuple(artifacts),
    )


def _expected_source_contract_from_bundle(
    fixture_lock: _FixtureLock, cases: Sequence[MicrosoftAecCase]
) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "fixture_id": fixture_lock.fixture_id,
                "lock_recipe_sha256": fixture_lock.recipe_sha256,
                "metadata": {
                    "sha256": META_SHA256,
                    "size_bytes": META_SIZE_BYTES,
                },
                "selected_rows_sha256": SELECTED_ROWS_SHA256,
                "signals": [
                    {
                        "role": artifact.role,
                        "sha256": artifact.sha256,
                        "size_bytes": artifact.size_bytes,
                        "samples": artifact.samples,
                    }
                    for case in cases
                    for artifact in case.artifacts
                ],
            }
        )
    ).hexdigest()


def load_microsoft_aec_bundle(
    path: Path | str,
) -> LoadedMicrosoftAecBundle:
    """Strictly reopen a complete private bundle without exposing source IDs."""

    root, directory_fd, root_snapshot = _open_private_root(path)
    try:
        receipt_raw, receipt_identity = _read_private_at(
            directory_fd,
            PREPARATION_RECEIPT_FILENAME,
            maximum_bytes=_MAX_RECEIPT_BYTES,
        )
        receipt = _mapping(
            _strict_json(receipt_raw, maximum_bytes=_MAX_RECEIPT_BYTES),
            {
                "accepted_terms",
                "artifact_count",
                "artifact_set_sha256",
                "audio_bytes",
                "evidence_scope",
                "fixture_id",
                "kind",
                "lock",
                "manifest",
                "preparer_files",
                "production_evidence",
                "schema_version",
                "selected_rows_sha256",
                "source_contract_sha256",
            },
        )
        if receipt_raw != _canonical_json(receipt, newline=True):
            raise MicrosoftAecFixtureError()
        production = receipt["production_evidence"]
        if not isinstance(production, bool):
            raise MicrosoftAecFixtureError()
        fixture_lock = _load_lock() if production else None
        preparer_closure = _preparer_closure()
        lock_value = _mapping(
            receipt["lock"],
            {"file", "raw_sha256", "recipe_sha256", "size_bytes"},
        )
        manifest_value = _mapping(receipt["manifest"], {"file", "sha256", "size_bytes"})
        evidence_scope = {
            "aec_component_replay": True,
            "aecmos": False,
            "capture_or_device": False,
            "live_barge_or_endpoint": False,
            "promotion_or_default_authority": False,
        }
        receipt_selected_rows = _sha256(receipt["selected_rows_sha256"])
        if (
            receipt["schema_version"] != SCHEMA_VERSION
            or receipt["kind"] != RECEIPT_KIND
            or receipt["accepted_terms"] != list(LICENSE_ACCEPTANCE_IDS)
            or receipt["evidence_scope"] != evidence_scope
            or receipt["preparer_files"] != list(preparer_closure)
            or manifest_value["file"] != MANIFEST_FILENAME
            or lock_value["file"] != DEFAULT_LOCK.name
            or not isinstance(receipt["fixture_id"], str)
            or (production and receipt["fixture_id"] != FIXTURE_ID)
            or (not production and receipt["fixture_id"] == FIXTURE_ID)
        ):
            raise MicrosoftAecFixtureError()
        if fixture_lock is not None and (
            lock_value["recipe_sha256"] != fixture_lock.recipe_sha256
            or lock_value["raw_sha256"] != fixture_lock.raw_sha256
            or lock_value["size_bytes"] != fixture_lock.raw_size_bytes
            or receipt_selected_rows != fixture_lock.selected_rows_sha256
        ):
            raise MicrosoftAecFixtureError()
        manifest_raw, manifest_identity = _read_private_at(
            directory_fd, MANIFEST_FILENAME, maximum_bytes=_MAX_MANIFEST_BYTES
        )
        if manifest_identity.sha256 != _sha256(
            manifest_value["sha256"]
        ) or manifest_identity.size_bytes != _integer(
            manifest_value["size_bytes"], minimum=1, maximum=_MAX_MANIFEST_BYTES
        ):
            raise MicrosoftAecFixtureError()
        manifest = _mapping(
            _strict_json(manifest_raw, maximum_bytes=_MAX_MANIFEST_BYTES),
            {
                "artifact_count",
                "audio_bytes",
                "case_count",
                "cases",
                "evidence_scope",
                "fixture_id",
                "kind",
                "lock_recipe_sha256",
                "production_evidence",
                "sample_format",
                "schema_version",
                "selected_rows_sha256",
                "source_contract_sha256",
            },
        )
        sample_format = _mapping(
            manifest["sample_format"], {"channels", "encoding", "sample_rate_hz"}
        )
        cases_raw = manifest["cases"]
        case_count = _integer(manifest["case_count"], minimum=1, maximum=CASE_COUNT)
        if (
            manifest_raw != _canonical_json(manifest, newline=True)
            or manifest["schema_version"] != SCHEMA_VERSION
            or manifest["kind"] != MANIFEST_KIND
            or manifest["fixture_id"] != receipt["fixture_id"]
            or manifest["production_evidence"] is not production
            or manifest["selected_rows_sha256"] != receipt_selected_rows
            or manifest["lock_recipe_sha256"] != lock_value["recipe_sha256"]
            or manifest["source_contract_sha256"] != receipt["source_contract_sha256"]
            or manifest["evidence_scope"] != evidence_scope
            or sample_format
            != {
                "channels": 1,
                "encoding": "pcm_s16le",
                "sample_rate_hz": SAMPLE_RATE_HZ,
            }
            or not isinstance(cases_raw, list)
            or len(cases_raw) != case_count
            or (production and case_count != CASE_COUNT)
        ):
            raise MicrosoftAecFixtureError()
        cases = tuple(
            _parse_manifest_case(
                raw_case,
                index=index,
                directory_fd=directory_fd,
                fixture_lock=fixture_lock,
            )
            for index, raw_case in enumerate(cases_raw)
        )
        artifact_count = len(cases) * len(SIGNAL_ROLES)
        audio_bytes = sum(
            artifact.size_bytes for case in cases for artifact in case.artifacts
        )
        source_contract = _sha256(receipt["source_contract_sha256"])
        if (
            manifest["artifact_count"] != artifact_count
            or receipt["artifact_count"] != artifact_count
            or manifest["audio_bytes"] != audio_bytes
            or receipt["audio_bytes"] != audio_bytes
            or receipt["artifact_set_sha256"] != _artifact_set_sha256(cases)
            or (
                fixture_lock is not None
                and source_contract
                != _expected_source_contract_from_bundle(fixture_lock, cases)
            )
        ):
            raise MicrosoftAecFixtureError()
        _exact_entries(directory_fd, _expected_entry_names(case_count, receipt=True))
        _verify_private_root(root, directory_fd, root_snapshot)
        if _preparer_closure() != preparer_closure or (
            fixture_lock is not None and _load_lock() != fixture_lock
        ):
            raise MicrosoftAecFixtureError()
        identities = (
            root_snapshot,
            receipt_identity.snapshot,
            manifest_identity.snapshot,
            *(artifact.snapshot for case in cases for artifact in case.artifacts),
        )
        return LoadedMicrosoftAecBundle(
            root=root,
            fixture_id=_string(receipt["fixture_id"], maximum=96),
            production_evidence=production,
            lock_recipe_sha256=_sha256(lock_value["recipe_sha256"]),
            manifest_sha256=manifest_identity.sha256,
            receipt_sha256=receipt_identity.sha256,
            source_contract_sha256=source_contract,
            cases=cases,
            identities=identities,
        )
    finally:
        try:
            os.close(directory_fd)
        except OSError:
            pass


def verify_microsoft_aec_bundle(bundle: LoadedMicrosoftAecBundle) -> None:
    if type(bundle) is not LoadedMicrosoftAecBundle:
        raise MicrosoftAecFixtureError()
    current = load_microsoft_aec_bundle(bundle.root)
    if current != bundle:
        raise MicrosoftAecFixtureError()


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise MicrosoftAecFixtureError()


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description="Prepare the pinned private Microsoft AEC 32-by-four fixture."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--accept-term",
        action="append",
        default=[],
        help="repeat once for each exact mixed-source acceptance identifier",
    )
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    return parser


def _lifecycle_handler(signum: int, _frame: object) -> None:
    raise _LifecycleSignal(signum)


def main(argv: Sequence[str] | None = None) -> int:
    previous: dict[int, object] = {}
    state = _CommitState()
    try:
        for signum in (
            getattr(signal, "SIGHUP", None),
            getattr(signal, "SIGTERM", None),
        ):
            if isinstance(signum, int):
                previous[signum] = signal.signal(signum, _lifecycle_handler)
        args = _parser().parse_args(argv)
        prepared = prepare_microsoft_aec_fixture(
            args.source_dir,
            args.output_dir,
            accepted_terms=args.accept_term,
            lock_path=args.lock,
            _commit_state=state,
        )
        os.write(
            1,
            _canonical_json(
                {
                    "artifact_count": prepared.artifact_count,
                    "audio_bytes": prepared.audio_bytes,
                    "kind": MANIFEST_KIND,
                    "manifest_sha256": prepared.manifest_sha256,
                    "ok": True,
                    "production_evidence": prepared.production_evidence,
                    "receipt_sha256": prepared.receipt_sha256,
                },
                newline=True,
            ),
        )
        return 0
    except KeyboardInterrupt:
        return 0 if state.committed else 130
    except _LifecycleSignal as interrupted:
        return 0 if state.committed else 128 + interrupted.signum
    except BaseException:
        if state.committed:
            return 0
        try:
            os.write(1, _canonical_json(_SAFE_ERROR, newline=True))
        except BaseException:
            pass
        return 2
    finally:
        for signum, handler in reversed(tuple(previous.items())):
            try:
                signal.signal(signum, handler)
            except BaseException:
                pass


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_LOCK",
    "FIXTURE_ID",
    "LICENSE_ACCEPTANCE_IDS",
    "LoadedMicrosoftAecBundle",
    "MANIFEST_FILENAME",
    "MicrosoftAecArtifact",
    "MicrosoftAecCase",
    "MicrosoftAecFixtureError",
    "PREPARATION_RECEIPT_FILENAME",
    "PreparedMicrosoftAecFixture",
    "TestFixtureInjection",
    "decode_microsoft_aec_wav",
    "load_microsoft_aec_bundle",
    "main",
    "prepare_microsoft_aec_fixture",
    "verify_microsoft_aec_bundle",
]
