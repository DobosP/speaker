"""Prepare one bounded, paired-device NOTSOFAR-1 overlap fixture.

The command is deliberately offline.  It verifies the already-retained exact
``MTG_32006`` source, selects seven source-timing two-speaker windows, and
publishes a private custom bundle whose terminal preparation receipt contains
no transcripts or raw speaker/device identities.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import signal
import stat
from typing import Callable, Final, Mapping, Sequence

import numpy as np

from tools import prepare_notsofar1_far_field_fixture as inherited
from tools.streaming_stt.bounded_io import (
    BoundedReadError,
    opened_directory_nofollow,
    read_regular_bounded,
)


SCHEMA_VERSION: Final = 1
FIXTURE_ID: Final = "notsofar1-mtg32006-natural-overlap-v1"
LICENSE_ID: Final = inherited.LICENSE_ID
REVISION: Final = inherited.REVISION
SAMPLE_RATE_HZ: Final = inherited.SAMPLE_RATE_HZ
SELECTION_SEED: Final = "speaker-notsofar1-natural-overlap-v1-2026-08-12"
MANIFEST_KIND: Final = "notsofar1-natural-overlap-paired-device-v1"
RECEIPT_KIND: Final = "notsofar1-natural-overlap-preparation-receipt-v1"
OVERLAP_METRIC: Final = "not-implemented"
MANIFEST_FILENAME: Final = "notsofar1-natural-overlap-v1.json"
PREPARATION_RECEIPT_FILENAME: Final = "preparation-receipt.json"
DEFAULT_LOCK: Final = (
    Path(__file__).resolve().parent
    / "streaming_stt"
    / "notsofar1-mtg32006-natural-overlap-v1.lock.json"
)
EXPECTED_LOCK_RECIPE_SHA256: Final = (
    "ec6bf59b21e7fe59d7ea432fd4a6ac17e965fb3289e2523121576f71636150a5"
)

MIN_SEGMENT_SECONDS: Final = Decimal("0.600000")
MAX_SEGMENT_SECONDS: Final = Decimal("8.000000")
MIN_OVERLAP_SECONDS: Final = Decimal("0.300000")
MINIMUM_OVERLAP_SECONDS: Final = MIN_OVERLAP_SECONDS
MAX_ENVELOPE_SECONDS: Final = Decimal("8.000000")
MAX_QUANTIZED_INTERVAL_SAMPLES: Final = int(MAX_ENVELOPE_SECONDS * SAMPLE_RATE_HZ) + 1
WINDOW_COUNT: Final = 7
CHANNEL_COUNT: Final = 2
MAX_DERIVED_PCM_BYTES: Final = 8 * 1024 * 1024
PRODUCTION_DERIVED_PCM_BYTES: Final = 2_416_640
PRODUCTION_SELECTION_GEOMETRY: Final = (
    (528_320, 552_800, 9_600, 23_040),
    (5_870_720, 5_901_600, 9_280, 25_920),
    (569_280, 611_360, 1_600, 10_720),
    (5_574_880, 5_642_720, 39_680, 57_920),
    (2_278_720, 2_317_600, 7_360, 37_920),
    (2_321_440, 2_378_240, 40_960, 56_000),
    (3_919_200, 3_960_320, 16_640, 36_480),
)
_SELECTION_COMMITMENT_DOMAIN: Final = (
    "speaker-notsofar1-natural-overlap-private-selection-commitment-v1"
)

_MAX_LOCK_BYTES: Final = 256 * 1024
_MAX_MANIFEST_BYTES: Final = 256 * 1024
_MAX_RECEIPT_BYTES: Final = 256 * 1024
_MAX_AUDIO_BYTES: Final = MAX_DERIVED_PCM_BYTES
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID_RE: Final = re.compile(r"[a-z0-9][a-z0-9_.-]{0,95}\Z")
_SOURCE_PATHS: Final = tuple(inherited._SOURCE_PINS)
_DEVICE_PATHS: Final = tuple(inherited._DEVICE_PATHS)
_PREPARER_FILES: Final = (
    "core/wer.py",
    "tools/prepare_ami_capture_replay.py",
    "tools/prepare_ami_conversation_fixture.py",
    "tools/prepare_notsofar1_far_field_fixture.py",
    "tools/prepare_notsofar1_natural_overlap_fixture.py",
    "tools/streaming_stt/bounded_io.py",
)
_SAFE_ERROR: Final = {
    "error": "notsofar1_natural_overlap_fixture_prerequisites_unavailable",
    "ok": False,
}


class NotsofarNaturalOverlapPreparationError(RuntimeError):
    """A detail-free lock, source, selection, or publication failure."""


@dataclass(frozen=True, slots=True)
class TestSourceInjection:
    """Explicit non-production identities for bounded generated tests."""

    fixture_id: str
    file_pins: Mapping[str, tuple[int, str]] = field(repr=False)
    meeting_duration_seconds: Decimal


@dataclass(frozen=True, slots=True)
class LoadedNotsofarNaturalOverlapBundle:
    path: Path = field(repr=False)
    digest: str
    receipt_sha256: str
    source_contract_sha256: str
    production_evidence: bool
    windows: tuple[Mapping[str, object], ...] = field(repr=False)
    identities: tuple[tuple[int, ...], ...] = field(repr=False)


PreparedNotsofarNaturalOverlapFixture = LoadedNotsofarNaturalOverlapBundle


@dataclass(frozen=True, slots=True)
class _Candidate:
    left: inherited._Segment = field(repr=False)
    right: inherited._Segment = field(repr=False)
    rank: str


@dataclass(frozen=True, slots=True)
class _Lock:
    fixture_id: str
    recipe_sha256: str
    overlap_metric: str
    source_lock_recipe_sha256: str
    source_pins: Mapping[str, tuple[str, int, str]] = field(repr=False)
    selection_rows: tuple[Mapping[str, object], ...] = field(repr=False)
    selection_private_commitment_sha256: str
    artifact_pins: tuple[Mapping[str, object], ...] = field(repr=False)
    production_evidence: bool


@dataclass(frozen=True, slots=True)
class _PrivateFile:
    name: str
    size_bytes: int
    sha256: str
    snapshot: tuple[int, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class _BundleLocator:
    root: Path = field(repr=False)
    root_identity: tuple[int, int, int, int] = field(repr=False)
    manifest_snapshot: tuple[int, ...] | None = field(repr=False)


@dataclass(slots=True)
class _CommitState:
    committed: bool = False
    receipt_snapshot: tuple[int, ...] | None = None


def _exact_json_tree(value: object) -> None:
    stack = [value]
    while stack:
        current = stack.pop()
        current_type = type(current)
        if current_type is dict:
            if any(type(key) is not str for key in current):
                raise NotsofarNaturalOverlapPreparationError()
            stack.extend(current.values())
        elif current_type is list:
            stack.extend(current)
        elif current_type not in {str, int, bool, type(None), Decimal}:
            raise NotsofarNaturalOverlapPreparationError()


def _strict_json(raw: bytes, *, maximum: int) -> object:
    if (
        type(raw) is not bytes
        or type(maximum) is not int
        or not 1 <= maximum <= MAX_DERIVED_PCM_BYTES
        or not raw
        or len(raw) > maximum
    ):
        raise NotsofarNaturalOverlapPreparationError()

    def pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in rows:
            if type(key) is not str or key in result:
                raise NotsofarNaturalOverlapPreparationError()
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_float=Decimal,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                NotsofarNaturalOverlapPreparationError()
            ),
        )
    except (
        MemoryError,
        RecursionError,
        UnicodeError,
        ValueError,
        OverflowError,
        NotsofarNaturalOverlapPreparationError,
    ):
        raise NotsofarNaturalOverlapPreparationError() from None
    _exact_json_tree(value)
    items = 0
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        if depth > 16:
            raise NotsofarNaturalOverlapPreparationError()
        if type(current) is dict:
            items += len(current)
            stack.extend((item, depth + 1) for item in current.values())
        elif type(current) is list:
            items += len(current)
            stack.extend((item, depth + 1) for item in current)
        elif type(current) is str and len(current) > 16_384:
            raise NotsofarNaturalOverlapPreparationError()
        if items > 100_000:
            raise NotsofarNaturalOverlapPreparationError()
    return value


def _canonical_json(value: object, *, newline: bool = False) -> bytes:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (MemoryError, RecursionError, TypeError, UnicodeError, ValueError):
        raise NotsofarNaturalOverlapPreparationError() from None
    return raw + (b"\n" if newline else b"")


def _sha256(value: object) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise NotsofarNaturalOverlapPreparationError()
    return value


def _integer(value: object, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise NotsofarNaturalOverlapPreparationError()
    return value


def _expect_exact_json(value: object, expected: object) -> None:
    """Reject JSON scalar/container type confusion before any equality hook."""

    if type(value) is not type(expected):
        raise NotsofarNaturalOverlapPreparationError()
    if type(expected) is dict:
        actual = value
        if any(type(key) is not str for key in actual) or set(actual) != set(expected):
            raise NotsofarNaturalOverlapPreparationError()
        for key, expected_item in expected.items():
            _expect_exact_json(actual[key], expected_item)
        return
    if type(expected) is list:
        actual = value
        if len(actual) != len(expected):
            raise NotsofarNaturalOverlapPreparationError()
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _expect_exact_json(actual_item, expected_item)
        return
    if value != expected:
        raise NotsofarNaturalOverlapPreparationError()


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


def _rank(left: inherited._Segment, right: inherited._Segment) -> str:
    members = sorted(
        (left, right),
        key=lambda item: (item.speaker, item.start, item.end, item.reference or ""),
    )
    parts = [SELECTION_SEED, "pair"]
    for item in members:
        if item.reference is None:
            raise NotsofarNaturalOverlapPreparationError()
        parts.extend((item.speaker, str(item.start), str(item.end), item.reference))
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def _semantic_identity(segment: inherited._Segment) -> tuple[object, ...]:
    return (segment.speaker, segment.start, segment.end, segment.reference)


def _select(
    segments: tuple[inherited._Segment, ...],
) -> tuple[_Candidate, ...]:
    if not 1 <= len(segments) <= 512:
        raise NotsofarNaturalOverlapPreparationError()
    identities = [_semantic_identity(item) for item in segments]
    if len(set(identities)) != len(identities):
        raise NotsofarNaturalOverlapPreparationError()
    # A third annotation invalidates the complete pair envelope, including a
    # possible gap between the two candidate activities.  Precompute the two
    # half-open interval predicates as bitsets so each pair can apply that
    # rule in O(1), after an O(n^2) bounded setup.
    starts_before_end_masks = [0] * len(segments)
    ends_after_start_masks = [0] * len(segments)
    for boundary_index, boundary in enumerate(segments):
        starts_before_end = 0
        ends_after_start = 0
        for segment_index, segment in enumerate(segments):
            if segment.start < boundary.end:
                starts_before_end |= 1 << segment_index
            if boundary.start < segment.end:
                ends_after_start |= 1 << segment_index
        starts_before_end_masks[boundary_index] = starts_before_end
        ends_after_start_masks[boundary_index] = ends_after_start
    candidates: list[_Candidate] = []
    ranks: set[str] = set()
    for index, left in enumerate(segments):
        for right_index, right in enumerate(segments[index + 1 :], start=index + 1):
            if (
                left.reference is None
                or right.reference is None
                or left.speaker == right.speaker
                or not MIN_SEGMENT_SECONDS
                <= left.end - left.start
                <= MAX_SEGMENT_SECONDS
                or not MIN_SEGMENT_SECONDS
                <= right.end - right.start
                <= MAX_SEGMENT_SECONDS
            ):
                continue
            overlap_start = max(left.start, right.start)
            overlap_end = min(left.end, right.end)
            envelope_start = min(left.start, right.start)
            envelope_end = max(left.end, right.end)
            earliest_start_index = index if left.start <= right.start else right_index
            latest_end_index = index if left.end >= right.end else right_index
            envelope_intersections = (
                starts_before_end_masks[latest_end_index]
                & ends_after_start_masks[earliest_start_index]
            )
            if (
                overlap_end - overlap_start < MIN_OVERLAP_SECONDS
                or envelope_end - envelope_start > MAX_ENVELOPE_SECONDS
                or envelope_intersections & ~((1 << index) | (1 << right_index))
            ):
                continue
            rank = _rank(left, right)
            if rank in ranks:
                raise NotsofarNaturalOverlapPreparationError()
            ranks.add(rank)
            candidates.append(_Candidate(left, right, rank))
    if len(candidates) < WINDOW_COUNT:
        raise NotsofarNaturalOverlapPreparationError()
    selected = tuple(sorted(candidates, key=lambda item: item.rank)[:WINDOW_COUNT])
    intervals = sorted(
        (min(item.left.start, item.right.start), max(item.left.end, item.right.end))
        for item in selected
    )
    if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
        raise NotsofarNaturalOverlapPreparationError()
    return selected


def _selection_row(
    candidate: _Candidate,
    *,
    include_references: bool,
) -> dict[str, object]:
    if type(include_references) is not bool:
        raise NotsofarNaturalOverlapPreparationError()
    left, right = sorted(
        (candidate.left, candidate.right),
        key=lambda item: (item.start, item.end, item.speaker, item.reference or ""),
    )
    if left.reference is None or right.reference is None:
        raise NotsofarNaturalOverlapPreparationError()
    start = min(
        inherited._floor_sample(left.start), inherited._floor_sample(right.start)
    )
    end = max(inherited._ceil_sample(left.end), inherited._ceil_sample(right.end))
    overlap_start = max(
        inherited._floor_sample(left.start), inherited._floor_sample(right.start)
    )
    overlap_end = min(
        inherited._ceil_sample(left.end), inherited._ceil_sample(right.end)
    )
    row: dict[str, object] = {
        "end_sample": end,
        "overlap_relative_end_sample": overlap_end - start,
        "overlap_relative_start_sample": overlap_start - start,
        "role_a_relative_end_sample": inherited._ceil_sample(left.end) - start,
        "role_a_relative_start_sample": inherited._floor_sample(left.start) - start,
        "role_b_relative_end_sample": inherited._ceil_sample(right.end) - start,
        "role_b_relative_start_sample": inherited._floor_sample(right.start) - start,
        "start_sample": start,
    }
    if include_references:
        row.update(
            {
                "reference_a_sha256": hashlib.sha256(
                    left.reference.encode("utf-8")
                ).hexdigest(),
                "reference_b_sha256": hashlib.sha256(
                    right.reference.encode("utf-8")
                ).hexdigest(),
            }
        )
    return row


def _safe_selection_rows(
    value: object,
    *,
    include_references: bool = False,
) -> tuple[Mapping[str, object], ...]:
    if type(include_references) is not bool:
        raise NotsofarNaturalOverlapPreparationError()
    fields = {
        "end_sample",
        "overlap_relative_end_sample",
        "overlap_relative_start_sample",
        "role_a_relative_end_sample",
        "role_a_relative_start_sample",
        "role_b_relative_end_sample",
        "role_b_relative_start_sample",
        "start_sample",
    }
    if include_references:
        fields.update({"reference_a_sha256", "reference_b_sha256"})
    if type(value) is not list or len(value) != WINDOW_COUNT:
        raise NotsofarNaturalOverlapPreparationError()
    result: list[Mapping[str, object]] = []
    for row in value:
        if (
            type(row) is not dict
            or any(type(key) is not str for key in row)
            or set(row) != fields
        ):
            raise NotsofarNaturalOverlapPreparationError()
        start = _integer(row["start_sample"], minimum=0, maximum=10_000_000)
        end = _integer(row["end_sample"], minimum=start + 1, maximum=10_000_000)
        samples = end - start
        if samples > MAX_QUANTIZED_INTERVAL_SAMPLES:
            raise NotsofarNaturalOverlapPreparationError()
        role_a_start = _integer(
            row["role_a_relative_start_sample"], minimum=0, maximum=samples - 1
        )
        role_a_end = _integer(
            row["role_a_relative_end_sample"],
            minimum=role_a_start + 1,
            maximum=samples,
        )
        role_b_start = _integer(
            row["role_b_relative_start_sample"], minimum=0, maximum=samples - 1
        )
        role_b_end = _integer(
            row["role_b_relative_end_sample"],
            minimum=role_b_start + 1,
            maximum=samples,
        )
        overlap_start = _integer(
            row["overlap_relative_start_sample"], minimum=0, maximum=samples - 1
        )
        overlap_end = _integer(
            row["overlap_relative_end_sample"],
            minimum=overlap_start + int(MIN_OVERLAP_SECONDS * SAMPLE_RATE_HZ),
            maximum=samples,
        )
        if overlap_start != max(role_a_start, role_b_start) or overlap_end != min(
            role_a_end, role_b_end
        ):
            raise NotsofarNaturalOverlapPreparationError()
        if (
            role_a_end - role_a_start > MAX_QUANTIZED_INTERVAL_SAMPLES
            or role_b_end - role_b_start > MAX_QUANTIZED_INTERVAL_SAMPLES
        ):
            raise NotsofarNaturalOverlapPreparationError()
        clean: dict[str, object] = {
            key: row[key] for key in fields if key.endswith("sample")
        }
        if include_references:
            clean.update(
                {
                    "reference_a_sha256": _sha256(row["reference_a_sha256"]),
                    "reference_b_sha256": _sha256(row["reference_b_sha256"]),
                }
            )
        result.append(clean)
    return tuple(result)


def _source_pin_rows(
    pins: Mapping[str, tuple[str, int, str]],
) -> list[dict[str, object]]:
    return [
        {
            "path": relative,
            "role": pins[relative][0],
            "size_bytes": pins[relative][1],
            "sha256": pins[relative][2],
        }
        for relative in _SOURCE_PATHS
    ]


def _selection_private_commitment_sha256(
    fixture_id: str,
    rows: tuple[Mapping[str, object], ...] | list[Mapping[str, object]],
) -> str:
    if (
        type(fixture_id) is not str
        or _SAFE_ID_RE.fullmatch(fixture_id) is None
        or type(rows) not in {tuple, list}
        or len(rows) != WINDOW_COUNT
    ):
        raise NotsofarNaturalOverlapPreparationError()
    public_fields = {
        "end_sample",
        "overlap_relative_end_sample",
        "overlap_relative_start_sample",
        "role_a_relative_end_sample",
        "role_a_relative_start_sample",
        "role_b_relative_end_sample",
        "role_b_relative_start_sample",
        "start_sample",
    }
    public_rows: list[dict[str, object]] = []
    references: list[tuple[str, str]] = []
    for row in rows:
        if (
            type(row) is not dict
            or any(type(key) is not str for key in row)
            or set(row) != public_fields | {"reference_a", "reference_b"}
        ):
            raise NotsofarNaturalOverlapPreparationError()
        public_rows.append({key: row[key] for key in public_fields})
        reference_a = row["reference_a"]
        reference_b = row["reference_b"]
        if (
            type(reference_a) is not str
            or type(reference_b) is not str
            or not reference_a.strip()
            or not reference_b.strip()
            or len(reference_a) > 4096
            or len(reference_b) > 4096
        ):
            raise NotsofarNaturalOverlapPreparationError()
        references.append((reference_a, reference_b))
    safe_public_rows = _safe_selection_rows(
        public_rows,
        include_references=False,
    )
    safe_rows = [
        {
            **public,
            "reference_a": reference_a,
            "reference_b": reference_b,
        }
        for public, (reference_a, reference_b) in zip(
            safe_public_rows,
            references,
            strict=True,
        )
    ]
    return hashlib.sha256(
        _canonical_json(
            {
                "domain": _SELECTION_COMMITMENT_DOMAIN,
                "fixture_id": fixture_id,
                "ordered_private_selection_rows": list(safe_rows),
                "version": 1,
            }
        )
    ).hexdigest()


def _selection_commitment_rows(
    selected: Sequence[_Candidate],
) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    for candidate in selected:
        left, right = sorted(
            (candidate.left, candidate.right),
            key=lambda item: (
                item.start,
                item.end,
                item.speaker,
                item.reference or "",
            ),
        )
        if left.reference is None or right.reference is None:
            raise NotsofarNaturalOverlapPreparationError()
        rows.append(
            {
                **_selection_row(candidate, include_references=False),
                "reference_a": left.reference,
                "reference_b": right.reference,
            }
        )
    return tuple(rows)


def _load_lock(
    path: Path | str,
    *,
    test_source_injection: TestSourceInjection | None,
) -> _Lock:
    supplied_lock = Path(os.path.abspath(Path(path).expanduser()))
    try:
        resolved_lock = supplied_lock.resolve(strict=True)
        default_lock = DEFAULT_LOCK.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        raise NotsofarNaturalOverlapPreparationError() from None
    if supplied_lock != resolved_lock or resolved_lock != default_lock:
        raise NotsofarNaturalOverlapPreparationError()
    try:
        snapshot = read_regular_bounded(Path(path), maximum_bytes=_MAX_LOCK_BYTES)
        value = _strict_json(snapshot.data, maximum=_MAX_LOCK_BYTES)
    except BoundedReadError:
        raise NotsofarNaturalOverlapPreparationError() from None
    _exact_json_tree(value)
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise NotsofarNaturalOverlapPreparationError()
    recipe = _sha256(value.get("recipe_sha256"))
    body = dict(value)
    body.pop("recipe_sha256", None)
    if hashlib.sha256(_canonical_json(body)).hexdigest() != recipe:
        raise NotsofarNaturalOverlapPreparationError()
    if test_source_injection is None and recipe != EXPECTED_LOCK_RECIPE_SHA256:
        raise NotsofarNaturalOverlapPreparationError()
    source = value.get("source")
    selection = value.get("selection")
    artifacts = value.get("output_artifacts")
    expected_source_rows = _source_pin_rows(inherited._SOURCE_PINS)
    if (
        set(value)
        != {
            "evidence_scope",
            "fixture_id",
            "kind",
            "license_id",
            "overlap_metric",
            "output_artifacts",
            "privacy",
            "recipe_digest_rule",
            "recipe_sha256",
            "repository",
            "revision",
            "sample_format",
            "schema_version",
            "selection",
            "source",
        }
        or type(selection) is not dict
        or any(type(key) is not str for key in selection)
        or set(selection)
        != {
            "algorithm",
            "cases",
            "channels",
            "maximum_envelope_seconds",
            "maximum_segment_seconds",
            "minimum_overlap_seconds",
            "minimum_segment_seconds",
            "rank_preimage",
            "reference_transform",
            "seed",
            "selection_private_commitment_sha256",
            "third_annotation_policy",
            "windows",
        }
        or type(artifacts) is not list
        or len(artifacts) != WINDOW_COUNT * CHANNEL_COUNT
    ):
        raise NotsofarNaturalOverlapPreparationError()
    for key, expected in (
        ("schema_version", SCHEMA_VERSION),
        ("kind", "notsofar1-natural-overlap-fixture-lock-v1"),
        ("fixture_id", FIXTURE_ID),
        ("license_id", LICENSE_ID),
        ("overlap_metric", OVERLAP_METRIC),
        ("recipe_digest_rule", "sha256-canonical-json-without-recipe_sha256-v1"),
        ("repository", "microsoft/NOTSOFAR"),
        ("revision", REVISION),
    ):
        _expect_exact_json(value[key], expected)
    _expect_exact_json(value["evidence_scope"], _evidence_scope())
    _expect_exact_json(value["privacy"], _receipt_privacy())
    _expect_exact_json(value["sample_format"], _sample_format())
    _expect_exact_json(
        source,
        {
            "files": expected_source_rows,
            "maximum_total_size_bytes": inherited.SOURCE_MAXIMUM_BYTES,
            "meeting_id": "MTG_32006",
            "source_lock_recipe_sha256": inherited.EXPECTED_RECIPE_SHA256,
            "total_size_bytes": sum(row[1] for row in inherited._SOURCE_PINS.values()),
        },
    )
    for key, expected in (
        ("seed", SELECTION_SEED),
        ("algorithm", "sha256-ranked-two-speaker-natural-overlap-v1"),
        (
            "rank_preimage",
            "nul-join(seed,pair,sorted-members(speaker,start,end,reference))"
            "-utf8-sha256-v1",
        ),
        (
            "reference_transform",
            "marker-stripped-row-text-and-core-wer-token-equality-v1",
        ),
        (
            "third_annotation_policy",
            "reject-any-third-half-open-envelope-intersection-v1",
        ),
        ("minimum_overlap_seconds", "0.300000"),
        ("minimum_segment_seconds", "0.600000"),
        ("maximum_segment_seconds", "8.000000"),
        ("maximum_envelope_seconds", "8.000000"),
        ("windows", WINDOW_COUNT),
        ("channels", CHANNEL_COUNT),
    ):
        _expect_exact_json(selection[key], expected)
    selection_rows = _safe_selection_rows(selection.get("cases"))
    selection_commitment = _sha256(selection.get("selection_private_commitment_sha256"))
    if selection_commitment == "0" * 64:
        raise NotsofarNaturalOverlapPreparationError()
    artifact_rows: list[Mapping[str, object]] = []
    for index, row in enumerate(artifacts):
        if (
            type(row) is not dict
            or any(type(key) is not str for key in row)
            or set(row) != {"file", "samples", "sha256", "size_bytes"}
        ):
            raise NotsofarNaturalOverlapPreparationError()
        samples = _integer(row["samples"], minimum=1, maximum=_MAX_AUDIO_BYTES // 4)
        expected_file = (
            f"notsofar-overlap-w{index // 2:02d}-channel-"
            f"{'a' if index % 2 == 0 else 'b'}.f32le"
        )
        _expect_exact_json(row.get("file"), expected_file)
        _expect_exact_json(row.get("size_bytes"), samples * 4)
        artifact_sha256 = _sha256(row["sha256"])
        if artifact_sha256 == "0" * 64:
            raise NotsofarNaturalOverlapPreparationError()
        artifact_rows.append(
            {
                "file": expected_file,
                "samples": samples,
                "sha256": artifact_sha256,
                "size_bytes": samples * 4,
            }
        )
    if sum(int(row["size_bytes"]) for row in artifact_rows) != (
        PRODUCTION_DERIVED_PCM_BYTES
    ):
        raise NotsofarNaturalOverlapPreparationError()
    production_lock = _Lock(
        FIXTURE_ID,
        recipe,
        OVERLAP_METRIC,
        inherited.EXPECTED_RECIPE_SHA256,
        inherited._SOURCE_PINS,
        selection_rows,
        selection_commitment,
        tuple(artifact_rows),
        True,
    )
    if test_source_injection is None:
        return production_lock
    if type(test_source_injection) is not TestSourceInjection:
        raise NotsofarNaturalOverlapPreparationError()
    fixture_id = test_source_injection.fixture_id
    file_pins = test_source_injection.file_pins
    if (
        type(fixture_id) is not str
        or not fixture_id.startswith("synthetic-")
        or _SAFE_ID_RE.fullmatch(fixture_id) is None
        or fixture_id == FIXTURE_ID
        or type(test_source_injection.meeting_duration_seconds) is not Decimal
        or not test_source_injection.meeting_duration_seconds.is_finite()
        or test_source_injection.meeting_duration_seconds <= 0
        or type(file_pins) is not dict
        or any(type(key) is not str for key in file_pins)
        or set(file_pins) != set(_SOURCE_PATHS)
    ):
        raise NotsofarNaturalOverlapPreparationError()
    pins: dict[str, tuple[str, int, str]] = {}
    for relative, (role, _size, _digest) in inherited._SOURCE_PINS.items():
        supplied = file_pins[relative]
        maximum = (
            inherited._MAX_WAV_BYTES
            if relative.endswith(".wav")
            else inherited._MAX_JSON_BYTES
        )
        if (
            type(supplied) is not tuple
            or len(supplied) != 2
            or type(supplied[0]) is not int
            or not 0 < supplied[0] <= maximum
            or type(supplied[1]) is not str
            or _SHA256_RE.fullmatch(supplied[1]) is None
        ):
            raise NotsofarNaturalOverlapPreparationError()
        pins[relative] = (role, supplied[0], supplied[1])
    if sum(row[1] for row in pins.values()) > inherited.SOURCE_MAXIMUM_BYTES:
        raise NotsofarNaturalOverlapPreparationError()
    synthetic_recipe = hashlib.sha256(
        _canonical_json(
            {
                "fixture_id": fixture_id,
                "meeting_duration_seconds": str(
                    test_source_injection.meeting_duration_seconds
                ),
                "source_files": _source_pin_rows(pins),
            }
        )
    ).hexdigest()
    return _Lock(
        fixture_id,
        synthetic_recipe,
        OVERLAP_METRIC,
        inherited.EXPECTED_RECIPE_SHA256,
        pins,
        (),
        synthetic_recipe,
        (),
        False,
    )


def _source_root(
    value: Path | str,
) -> tuple[Path, tuple[int, int, int, int]]:
    candidate = Path(os.path.abspath(Path(value).expanduser()))
    try:
        resolved = candidate.resolve(strict=True)
        with opened_directory_nofollow(resolved) as (stable, descriptor):
            metadata = os.fstat(descriptor)
            if (
                stable != resolved
                or candidate != resolved
                or not stat.S_ISDIR(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o700
                or metadata.st_uid != os.getuid()
            ):
                raise NotsofarNaturalOverlapPreparationError()
            identity = _directory_identity(metadata)
    except NotsofarNaturalOverlapPreparationError:
        raise
    except (OSError, RuntimeError, ValueError, BoundedReadError):
        raise NotsofarNaturalOverlapPreparationError() from None
    return resolved, identity


def _source_budget(
    pins: Mapping[str, tuple[str, int, str]],
) -> int:
    """Validate the complete declared source budget before any leaf is opened."""

    if (
        type(pins) is not dict
        or any(type(key) is not str for key in pins)
        or set(pins) != set(_SOURCE_PATHS)
    ):
        raise NotsofarNaturalOverlapPreparationError()
    total = 0
    for relative in _SOURCE_PATHS:
        pin = pins[relative]
        maximum = (
            inherited._MAX_WAV_BYTES
            if relative.endswith(".wav")
            else inherited._MAX_JSON_BYTES
        )
        if (
            type(pin) is not tuple
            or len(pin) != 3
            or type(pin[0]) is not str
            or type(pin[1]) is not int
            or not 0 < pin[1] <= maximum
            or type(pin[2]) is not str
            or _SHA256_RE.fullmatch(pin[2]) is None
        ):
            raise NotsofarNaturalOverlapPreparationError()
        total += pin[1]
        if total > inherited.SOURCE_MAXIMUM_BYTES:
            raise NotsofarNaturalOverlapPreparationError()
    return total


def _open_source_parent(
    root_fd: int,
    relative: str,
) -> tuple[int, str, tuple[tuple[int, int, int, int], ...]]:
    """Walk every source directory component from the bound root descriptor."""

    path = PurePosixPath(relative)
    if (
        path.is_absolute()
        or path.as_posix() != relative
        or len(path.parts) not in {1, 2}
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in relative
    ):
        raise NotsofarNaturalOverlapPreparationError()
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    identities: list[tuple[int, int, int, int]] = []
    try:
        descriptor = os.open(".", flags, dir_fd=root_fd)
        for component in path.parts[:-1]:
            before = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            if (
                not stat.S_ISDIR(before.st_mode)
                or stat.S_IMODE(before.st_mode) != 0o700
                or before.st_uid != os.getuid()
            ):
                raise NotsofarNaturalOverlapPreparationError()
            child = os.open(component, flags, dir_fd=descriptor)
            opened = os.fstat(child)
            identity = _directory_identity(opened)
            if identity != _directory_identity(before):
                os.close(child)
                raise NotsofarNaturalOverlapPreparationError()
            os.close(descriptor)
            descriptor = child
            identities.append(identity)
        return descriptor, path.parts[-1], tuple(identities)
    except NotsofarNaturalOverlapPreparationError:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except (OSError, RuntimeError, ValueError):
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise NotsofarNaturalOverlapPreparationError() from None


def _verify_source_root(
    root: Path,
    root_fd: int,
    root_identity: tuple[int, int, int, int],
) -> None:
    try:
        opened = os.fstat(root_fd)
        lexical = root.lstat()
        if (
            _directory_identity(opened) != root_identity
            or _directory_identity(lexical) != root_identity
            or not stat.S_ISDIR(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o700
            or opened.st_uid != os.getuid()
            or root.resolve(strict=True) != root
        ):
            raise NotsofarNaturalOverlapPreparationError()
    except NotsofarNaturalOverlapPreparationError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise NotsofarNaturalOverlapPreparationError() from None


def _read_source(
    root: Path,
    root_identity: tuple[int, int, int, int],
    relative: str,
    pin: tuple[str, int, str],
) -> inherited._BoundSource:
    role, expected_size, expected_sha = pin
    maximum = (
        inherited._MAX_WAV_BYTES
        if relative.endswith(".wav")
        else inherited._MAX_JSON_BYTES
    )
    root_fd = -1
    parent_fd = -1
    leaf_fd = -1
    try:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        root_fd = os.open(root, flags)
        _verify_source_root(root, root_fd, root_identity)
        parent_fd, leaf, parent_identities = _open_source_parent(root_fd, relative)
        before = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
            or before.st_size != expected_size
            or before.st_size > maximum
        ):
            raise NotsofarNaturalOverlapPreparationError()
        leaf_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        leaf_flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        leaf_fd = os.open(leaf, leaf_flags, dir_fd=parent_fd)
        opened = os.fstat(leaf_fd)
        if not stat.S_ISREG(opened.st_mode) or _file_snapshot(opened) != _file_snapshot(
            before
        ):
            raise NotsofarNaturalOverlapPreparationError()
        raw = bytearray()
        while len(raw) <= expected_size:
            chunk = os.read(
                leaf_fd,
                min(65_536, expected_size + 1 - len(raw)),
            )
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(leaf_fd)
        reopened_parent_fd, reopened_leaf, reopened_identities = _open_source_parent(
            root_fd,
            relative,
        )
        try:
            current = os.stat(
                reopened_leaf,
                dir_fd=reopened_parent_fd,
                follow_symlinks=False,
            )
        finally:
            os.close(reopened_parent_fd)
        digest = hashlib.sha256(raw).hexdigest()
        snapshot = _file_snapshot(after)
        _verify_source_root(root, root_fd, root_identity)
        if (
            parent_identities != reopened_identities
            or _file_snapshot(before) != snapshot
            or _file_snapshot(current) != snapshot
            or len(raw) != expected_size
            or digest != expected_sha
        ):
            raise NotsofarNaturalOverlapPreparationError()
        return inherited._BoundSource(
            relative=relative,
            role=role,
            raw=bytes(raw),
            size_bytes=expected_size,
            sha256=digest,
            snapshot=snapshot,
        )
    except NotsofarNaturalOverlapPreparationError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise NotsofarNaturalOverlapPreparationError() from None
    finally:
        for descriptor in (leaf_fd, parent_fd, root_fd):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass


def _verify_sources(
    root: Path,
    root_identity: tuple[int, int, int, int],
    sources: Mapping[str, inherited._BoundSource],
    pins: Mapping[str, tuple[str, int, str]],
) -> None:
    for relative in _SOURCE_PATHS:
        current = _read_source(root, root_identity, relative, pins[relative])
        if current.snapshot != sources[relative].snapshot:
            raise NotsofarNaturalOverlapPreparationError()


def _preparer_closure() -> tuple[dict[str, object], ...]:
    root = Path(__file__).resolve().parents[1]
    rows: list[dict[str, object]] = []
    for relative in _PREPARER_FILES:
        try:
            loaded = read_regular_bounded(
                root.joinpath(*PurePosixPath(relative).parts),
                maximum_bytes=1024 * 1024,
            )
        except BoundedReadError:
            raise NotsofarNaturalOverlapPreparationError() from None
        rows.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(loaded.data).hexdigest(),
                "size_bytes": len(loaded.data),
            }
        )
    return tuple(rows)


def _has_git_ancestor(path: Path) -> bool:
    for parent in (path, *path.parents):
        try:
            (parent / ".git").lstat()
        except FileNotFoundError:
            continue
        except OSError:
            raise NotsofarNaturalOverlapPreparationError() from None
        return True
    return False


def _directory_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_uid,
    )


def _new_private_output(
    output_dir: Path | str,
    *,
    source_root: Path,
) -> tuple[Path, int, tuple[int, int, int, int]]:
    supplied = Path(output_dir).expanduser()
    if not supplied.is_absolute():
        raise NotsofarNaturalOverlapPreparationError()
    candidate = Path(os.path.abspath(supplied))
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
            raise NotsofarNaturalOverlapPreparationError()
        with opened_directory_nofollow(parent) as (stable, parent_fd):
            if stable != parent or candidate.parent != parent:
                raise NotsofarNaturalOverlapPreparationError()
            os.mkdir(candidate.name, mode=0o700, dir_fd=parent_fd)
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(candidate.name, flags, dir_fd=parent_fd)
            os.fchmod(descriptor, 0o700)
            opened = os.fstat(descriptor)
            identity = _directory_identity(opened)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or stat.S_IMODE(opened.st_mode) != 0o700
                or opened.st_uid != os.getuid()
            ):
                raise NotsofarNaturalOverlapPreparationError()
            os.fsync(descriptor)
            os.fsync(parent_fd)
            return candidate, descriptor, identity
    except NotsofarNaturalOverlapPreparationError:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except (OSError, RuntimeError, ValueError):
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise NotsofarNaturalOverlapPreparationError() from None


def _verify_output(
    path: Path,
    descriptor: int,
    identity: tuple[int, int, int, int],
) -> None:
    try:
        opened = os.fstat(descriptor)
        lexical = path.lstat()
        if (
            _directory_identity(opened) != identity
            or _directory_identity(lexical) != identity
            or stat.S_IMODE(opened.st_mode) != 0o700
            or path.resolve(strict=True) != path
            or _has_git_ancestor(path)
        ):
            raise NotsofarNaturalOverlapPreparationError()
    except NotsofarNaturalOverlapPreparationError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise NotsofarNaturalOverlapPreparationError() from None


def _safe_leaf(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "/" in value
        or "\\" in value
        or value in {".", ".."}
        or len(value) > 128
    ):
        raise NotsofarNaturalOverlapPreparationError()
    return value


def _write_private(directory_fd: int, name: str, raw: bytes) -> _PrivateFile:
    leaf = _safe_leaf(name)
    if not isinstance(raw, bytes) or not raw:
        raise NotsofarNaturalOverlapPreparationError()
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
                raise NotsofarNaturalOverlapPreparationError()
            written += count
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        result = _PrivateFile(
            leaf,
            len(raw),
            hashlib.sha256(raw).hexdigest(),
            _file_snapshot(metadata),
        )
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
            or metadata.st_size != len(raw)
        ):
            raise NotsofarNaturalOverlapPreparationError()
        return result
    except NotsofarNaturalOverlapPreparationError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise NotsofarNaturalOverlapPreparationError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _read_private(
    directory_fd: int,
    name: str,
    *,
    maximum: int,
    expected: _PrivateFile | None = None,
) -> tuple[bytes, _PrivateFile]:
    leaf = _safe_leaf(name)
    descriptor = -1
    try:
        entry_before = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(entry_before.st_mode)
            or stat.S_IMODE(entry_before.st_mode) != 0o600
            or entry_before.st_uid != os.getuid()
            or entry_before.st_nlink != 1
            or not 0 < entry_before.st_size <= maximum
        ):
            raise NotsofarNaturalOverlapPreparationError()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(leaf, flags, dir_fd=directory_fd)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or _file_snapshot(before) != _file_snapshot(
            entry_before
        ):
            raise NotsofarNaturalOverlapPreparationError()
        raw = bytearray()
        while len(raw) <= maximum:
            chunk = os.read(descriptor, min(65_536, maximum + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
        current = _PrivateFile(
            leaf,
            len(raw),
            hashlib.sha256(raw).hexdigest(),
            _file_snapshot(after),
        )
        entry = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not 0 < len(raw) <= maximum
            or _file_snapshot(before) != _file_snapshot(after)
            or _file_snapshot(entry) != _file_snapshot(after)
            or stat.S_IMODE(after.st_mode) != 0o600
            or after.st_uid != os.getuid()
            or after.st_nlink != 1
            or (expected is not None and current != expected)
        ):
            raise NotsofarNaturalOverlapPreparationError()
        return bytes(raw), current
    except NotsofarNaturalOverlapPreparationError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise NotsofarNaturalOverlapPreparationError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _expected_entries(*, receipt: bool) -> set[str]:
    names = {MANIFEST_FILENAME}
    for index in range(WINDOW_COUNT):
        for channel in ("a", "b"):
            names.add(f"notsofar-overlap-w{index:02d}-channel-{channel}.f32le")
    if receipt:
        names.add(PREPARATION_RECEIPT_FILENAME)
    return names


def _exact_entries(directory_fd: int, expected: set[str]) -> None:
    rows: list[str] = []
    try:
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                if len(rows) >= len(expected) or type(entry.name) is not str:
                    raise NotsofarNaturalOverlapPreparationError()
                rows.append(entry.name)
    except NotsofarNaturalOverlapPreparationError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise NotsofarNaturalOverlapPreparationError() from None
    if len(rows) != len(set(rows)) or set(rows) != expected:
        raise NotsofarNaturalOverlapPreparationError()


def _stage_terminal_receipt(directory_fd: int, raw: bytes) -> tuple[int, _PrivateFile]:
    descriptor = -1
    ready = False
    try:
        temporary = getattr(os, "O_TMPFILE", 0)
        if not temporary or not raw or len(raw) > _MAX_RECEIPT_BYTES:
            raise NotsofarNaturalOverlapPreparationError()
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
                raise NotsofarNaturalOverlapPreparationError()
            written += count
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        staged = _PrivateFile(
            PREPARATION_RECEIPT_FILENAME,
            len(raw),
            hashlib.sha256(raw).hexdigest(),
            _file_snapshot(opened),
        )
        if (
            stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_uid != os.getuid()
            or opened.st_nlink != 0
            or opened.st_size != len(raw)
        ):
            raise NotsofarNaturalOverlapPreparationError()
        ready = True
        return descriptor, staged
    except NotsofarNaturalOverlapPreparationError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise NotsofarNaturalOverlapPreparationError() from None
    finally:
        if not ready and descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _verify_staged(descriptor: int, staged: _PrivateFile, raw: bytes) -> None:
    try:
        before = os.fstat(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        loaded = bytearray()
        while len(loaded) <= _MAX_RECEIPT_BYTES:
            chunk = os.read(
                descriptor,
                min(65_536, _MAX_RECEIPT_BYTES + 1 - len(loaded)),
            )
            if not chunk:
                break
            loaded.extend(chunk)
        after = os.fstat(descriptor)
        if (
            bytes(loaded) != raw
            or staged.sha256 != hashlib.sha256(loaded).hexdigest()
            or staged.snapshot != _file_snapshot(after)
            or _file_snapshot(before) != _file_snapshot(after)
            or after.st_nlink != 0
        ):
            raise NotsofarNaturalOverlapPreparationError()
    except NotsofarNaturalOverlapPreparationError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise NotsofarNaturalOverlapPreparationError() from None


def _commit_terminal_receipt(
    output_path: Path,
    directory_fd: int,
    directory_identity: tuple[int, int, int, int],
    staged_fd: int,
    staged: _PrivateFile,
    raw: bytes,
    state: _CommitState,
    *,
    commit_guard: Callable[[], None],
) -> None:
    if state.committed or not hasattr(signal, "pthread_sigmask"):
        raise NotsofarNaturalOverlapPreparationError()
    commit_guard()
    _verify_output(output_path, directory_fd, directory_identity)
    _exact_entries(directory_fd, _expected_entries(receipt=False))
    _verify_staged(staged_fd, staged, raw)
    os.fsync(directory_fd)
    signums = tuple(
        value
        for value in (
            getattr(signal, "SIGINT", None),
            getattr(signal, "SIGHUP", None),
            getattr(signal, "SIGTERM", None),
        )
        if isinstance(value, int)
    )
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, signums)
    linked = False
    try:
        try:
            os.link(
                f"/proc/self/fd/{staged_fd}",
                PREPARATION_RECEIPT_FILENAME,
                dst_dir_fd=directory_fd,
                follow_symlinks=True,
            )
            linked = True
        finally:
            if linked:
                state.committed = True
            else:
                try:
                    published = os.stat(
                        PREPARATION_RECEIPT_FILENAME,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                    opened = os.fstat(staged_fd)
                except OSError:
                    pass
                else:
                    state.committed = (
                        (published.st_dev, published.st_ino)
                        == (opened.st_dev, opened.st_ino)
                        and published.st_nlink == opened.st_nlink == 1
                        and stat.S_IMODE(published.st_mode) == 0o600
                        and published.st_uid == os.getuid()
                    )
    finally:
        if state.committed:
            try:
                published = os.stat(
                    PREPARATION_RECEIPT_FILENAME,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                opened = os.fstat(staged_fd)
                if (
                    (published.st_dev, published.st_ino)
                    == (opened.st_dev, opened.st_ino)
                    and published.st_nlink == opened.st_nlink == 1
                    and stat.S_IMODE(opened.st_mode) == 0o600
                    and opened.st_uid == os.getuid()
                ):
                    state.receipt_snapshot = _file_snapshot(opened)
                os.fsync(directory_fd)
            except BaseException:
                pass
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)
    if not state.committed:
        raise NotsofarNaturalOverlapPreparationError()


def _source_contract_sha256(
    lock: _Lock,
    source_rows: list[dict[str, object]],
    selection_rows: tuple[Mapping[str, object], ...],
    closure: tuple[dict[str, object], ...] | list[dict[str, object]],
) -> str:
    if (
        type(lock) is not _Lock
        or type(source_rows) is not list
        or type(selection_rows) is not tuple
        or type(closure) not in {tuple, list}
    ):
        raise NotsofarNaturalOverlapPreparationError()
    if (
        type(lock.fixture_id) is not str
        or _SAFE_ID_RE.fullmatch(lock.fixture_id) is None
        or type(lock.recipe_sha256) is not str
        or _SHA256_RE.fullmatch(lock.recipe_sha256) is None
        or type(lock.overlap_metric) is not str
        or lock.overlap_metric != OVERLAP_METRIC
        or type(lock.production_evidence) is not bool
    ):
        raise NotsofarNaturalOverlapPreparationError()
    clean_sources: list[dict[str, object]] = []
    for row in source_rows:
        if (
            type(row) is not dict
            or any(type(key) is not str for key in row)
            or set(row) != {"role", "sha256", "size_bytes"}
        ):
            raise NotsofarNaturalOverlapPreparationError()
        role = row["role"]
        size = row["size_bytes"]
        digest = _sha256(row["sha256"])
        if type(role) is not str or type(size) is not int:
            raise NotsofarNaturalOverlapPreparationError()
        clean_sources.append({"role": role, "sha256": digest, "size_bytes": size})
    selection_values = list(selection_rows)
    include_reference_hashes = all(
        type(row) is dict and {"reference_a_sha256", "reference_b_sha256"}.issubset(row)
        for row in selection_values
    )
    clean_selection = _safe_selection_rows(
        selection_values,
        include_references=include_reference_hashes,
    )
    clean_closure: list[dict[str, object]] = []
    for row in closure:
        if (
            type(row) is not dict
            or any(type(key) is not str for key in row)
            or set(row) != {"path", "sha256", "size_bytes"}
        ):
            raise NotsofarNaturalOverlapPreparationError()
        path = row["path"]
        size = row["size_bytes"]
        digest = _sha256(row["sha256"])
        if type(path) is not str or type(size) is not int:
            raise NotsofarNaturalOverlapPreparationError()
        clean_closure.append({"path": path, "sha256": digest, "size_bytes": size})
    return hashlib.sha256(
        _canonical_json(
            {
                "accepted_license": LICENSE_ID,
                "fixture_id": lock.fixture_id,
                "lock_recipe_sha256": lock.recipe_sha256,
                "overlap_metric": lock.overlap_metric,
                "preparer_files": clean_closure,
                "production_evidence": lock.production_evidence,
                "selection_rows": list(clean_selection),
                "source_files": clean_sources,
            }
        )
    ).hexdigest()


def _decode(raw: bytes, *, expected_samples: int) -> np.ndarray:
    try:
        pcm = inherited._riff_pcm16_mono(raw)
    except Exception:
        raise NotsofarNaturalOverlapPreparationError() from None
    if pcm.dtype != np.dtype("<i2") or pcm.ndim != 1 or pcm.size != expected_samples:
        raise NotsofarNaturalOverlapPreparationError()
    return pcm


def _pcm_f32le(pcm: np.ndarray, start: int, end: int) -> bytes:
    if not 0 <= start < end <= pcm.size:
        raise NotsofarNaturalOverlapPreparationError()
    result = pcm[start:end].astype("<f4")
    result *= np.float32(1.0 / 32768.0)
    if not np.isfinite(result).all():
        raise NotsofarNaturalOverlapPreparationError()
    raw = result.tobytes(order="C")
    if not raw or len(raw) != (end - start) * 4:
        raise NotsofarNaturalOverlapPreparationError()
    return raw


def _source_rows(
    sources: Mapping[str, inherited._BoundSource],
) -> list[dict[str, object]]:
    return [
        {
            "role": sources[relative].role,
            "sha256": sources[relative].sha256,
            "size_bytes": sources[relative].size_bytes,
        }
        for relative in _SOURCE_PATHS
    ]


def _evidence_scope() -> dict[str, bool]:
    return {
        "diagnostic_only": True,
        "natural_overlap": True,
        "ordinary_wer": False,
        "original_device_audio": True,
        "promotion_authority": False,
        "qualification_authority": False,
        "speaker_authority": False,
    }


def _receipt_privacy() -> dict[str, bool]:
    return {
        "case_rows_in_receipt": False,
        "local_paths_in_receipt": False,
        "raw_device_ids_in_receipt": False,
        "raw_speaker_ids_in_receipt": False,
        "transcripts_in_receipt": False,
    }


def _sample_format() -> dict[str, object]:
    return {
        "channels": 1,
        "encoding": "f32le",
        "sample_conversion": "pcm-s16le-to-f32le-divide-32768-v1",
        "sample_rate_hz": SAMPLE_RATE_HZ,
    }


def _build_windows(
    selected: Sequence[_Candidate],
    channels: Mapping[str, np.ndarray],
    *,
    production_evidence: bool,
) -> tuple[list[dict[str, object]], dict[str, bytes]]:
    if type(production_evidence) is not bool:
        raise NotsofarNaturalOverlapPreparationError()
    windows: list[dict[str, object]] = []
    artifacts: dict[str, bytes] = {}
    total_bytes = 0
    for index, candidate in enumerate(selected):
        role_a, role_b = sorted(
            (candidate.left, candidate.right),
            key=lambda item: (item.start, item.end, item.speaker, item.reference or ""),
        )
        if role_a.reference is None or role_b.reference is None:
            raise NotsofarNaturalOverlapPreparationError()
        start = min(
            inherited._floor_sample(role_a.start),
            inherited._floor_sample(role_b.start),
        )
        end = max(
            inherited._ceil_sample(role_a.end),
            inherited._ceil_sample(role_b.end),
        )
        samples = end - start
        window_id = f"notsofar-overlap-w{index:02d}"
        channel_rows: list[dict[str, object]] = []
        for channel_name in ("channel-a", "channel-b"):
            raw = _pcm_f32le(channels[channel_name], start, end)
            case_id = f"{window_id}-{channel_name}"
            filename = f"{case_id}.f32le"
            artifacts[filename] = raw
            total_bytes += len(raw)
            channel_rows.append(
                {
                    "case_id": case_id,
                    "channel": channel_name,
                    "file": filename,
                    "samples": samples,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "size_bytes": len(raw),
                }
            )

        def role_row(segment: inherited._Segment) -> dict[str, object]:
            if segment.reference is None:
                raise NotsofarNaturalOverlapPreparationError()
            return {
                "activity_relative_end_sample": inherited._ceil_sample(segment.end)
                - start,
                "activity_relative_start_sample": inherited._floor_sample(segment.start)
                - start,
                "reference": segment.reference,
                "reference_sha256": hashlib.sha256(
                    segment.reference.encode("utf-8")
                ).hexdigest(),
            }

        overlap_start = (
            max(
                inherited._floor_sample(role_a.start),
                inherited._floor_sample(role_b.start),
            )
            - start
        )
        overlap_end = (
            min(
                inherited._ceil_sample(role_a.end),
                inherited._ceil_sample(role_b.end),
            )
            - start
        )
        windows.append(
            {
                "channels": channel_rows,
                "envelope": {
                    "samples": samples,
                    "source_end_sample": end,
                    "source_start_sample": start,
                },
                "overlap": {
                    "relative_end_sample": overlap_end,
                    "relative_start_sample": overlap_start,
                    "samples": overlap_end - overlap_start,
                },
                "role_a": role_row(role_a),
                "role_b": role_row(role_b),
                "window_id": window_id,
            }
        )
    if (
        len(windows) != WINDOW_COUNT
        or len(artifacts) != WINDOW_COUNT * CHANNEL_COUNT
        or total_bytes > MAX_DERIVED_PCM_BYTES
        or (production_evidence and total_bytes != PRODUCTION_DERIVED_PCM_BYTES)
    ):
        raise NotsofarNaturalOverlapPreparationError()
    return windows, artifacts


def _selection_rows(
    selected: Sequence[_Candidate],
    *,
    include_references: bool,
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        _selection_row(item, include_references=include_references) for item in selected
    )


def _validate_against_lock(
    lock: _Lock,
    public_rows: tuple[Mapping[str, object], ...],
    commitment_rows: tuple[Mapping[str, object], ...],
    artifacts: dict[str, bytes],
) -> None:
    if (
        type(lock) is not _Lock
        or type(lock.production_evidence) is not bool
        or type(public_rows) is not tuple
        or type(commitment_rows) is not tuple
        or type(artifacts) is not dict
    ):
        raise NotsofarNaturalOverlapPreparationError()
    if lock.production_evidence is False:
        return
    if (
        type(lock.fixture_id) is not str
        or type(lock.selection_rows) is not tuple
        or type(lock.selection_private_commitment_sha256) is not str
        or _SHA256_RE.fullmatch(lock.selection_private_commitment_sha256) is None
        or type(lock.artifact_pins) is not tuple
    ):
        raise NotsofarNaturalOverlapPreparationError()
    if any(
        type(name) is not str or type(raw) is not bytes
        for name, raw in artifacts.items()
    ):
        raise NotsofarNaturalOverlapPreparationError()
    safe_public = _safe_selection_rows(list(public_rows))
    safe_lock_rows = _safe_selection_rows(list(lock.selection_rows))
    _expect_exact_json(list(safe_public), list(safe_lock_rows))
    if (
        sum(len(raw) for raw in artifacts.values()) != PRODUCTION_DERIVED_PCM_BYTES
        or _selection_private_commitment_sha256(
            lock.fixture_id,
            commitment_rows,
        )
        != lock.selection_private_commitment_sha256
    ):
        raise NotsofarNaturalOverlapPreparationError()
    actual = tuple(
        {
            "file": name,
            "samples": len(raw) // 4,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }
        for name, raw in artifacts.items()
    )
    safe_artifact_pins: list[dict[str, object]] = []
    for row in lock.artifact_pins:
        if (
            type(row) is not dict
            or any(type(key) is not str for key in row)
            or set(row) != {"file", "samples", "sha256", "size_bytes"}
            or type(row["file"]) is not str
            or type(row["samples"]) is not int
            or type(row["size_bytes"]) is not int
        ):
            raise NotsofarNaturalOverlapPreparationError()
        safe_artifact_pins.append(
            {
                "file": row["file"],
                "samples": row["samples"],
                "sha256": _sha256(row["sha256"]),
                "size_bytes": row["size_bytes"],
            }
        )
    _expect_exact_json(list(actual), safe_artifact_pins)


def prepare_notsofar1_natural_overlap_fixture(
    *,
    source_dir: Path | str,
    output_dir: Path | str,
    accepted_license: str,
    lock_path: Path | str = DEFAULT_LOCK,
    test_source_injection: TestSourceInjection | None = None,
) -> PreparedNotsofarNaturalOverlapFixture:
    """Verify retained inputs and publish the exact grouped overlap bundle."""

    if type(accepted_license) is not str or accepted_license != LICENSE_ID:
        raise NotsofarNaturalOverlapPreparationError()
    lock = _load_lock(
        lock_path,
        test_source_injection=test_source_injection,
    )
    _source_budget(lock.source_pins)
    state = _CommitState()
    root, root_identity = _source_root(source_dir)
    closure = _preparer_closure()
    sources = {
        relative: _read_source(
            root,
            root_identity,
            relative,
            lock.source_pins[relative],
        )
        for relative in _SOURCE_PATHS
    }
    duration = (
        inherited.MEETING_DURATION
        if test_source_injection is None
        else test_source_injection.meeting_duration_seconds
    )
    try:
        meeting = inherited._meeting(
            sources["gt_meeting_metadata.json"].raw,
            duration,
        )
        inherited._devices(sources["devices.json"].raw)
        segments = inherited._segments(
            sources["gt_transcription.json"].raw,
            meeting,
        )
    except Exception:
        raise NotsofarNaturalOverlapPreparationError() from None
    selected = _select(segments)
    selection_rows = _selection_rows(selected, include_references=True)
    public_selection_rows = _selection_rows(selected, include_references=False)
    commitment_rows = _selection_commitment_rows(selected)
    expected_samples_decimal = duration * SAMPLE_RATE_HZ
    if expected_samples_decimal != expected_samples_decimal.to_integral_value():
        raise NotsofarNaturalOverlapPreparationError()
    expected_samples = int(expected_samples_decimal)
    channels = {
        "channel-a": _decode(
            sources[_DEVICE_PATHS[0]].raw,
            expected_samples=expected_samples,
        ),
        "channel-b": _decode(
            sources[_DEVICE_PATHS[1]].raw,
            expected_samples=expected_samples,
        ),
    }
    windows, artifact_bytes = _build_windows(
        selected,
        channels,
        production_evidence=lock.production_evidence,
    )
    _validate_against_lock(
        lock,
        public_selection_rows,
        commitment_rows,
        artifact_bytes,
    )
    source_rows = _source_rows(sources)
    source_contract = _source_contract_sha256(
        lock,
        source_rows,
        selection_rows,
        closure,
    )
    manifest = {
        "evidence_scope": _evidence_scope(),
        "fixture_id": lock.fixture_id,
        "kind": MANIFEST_KIND,
        "lock_recipe_sha256": lock.recipe_sha256,
        "overlap_metric": lock.overlap_metric,
        "production_evidence": lock.production_evidence,
        "sample_format": _sample_format(),
        "schema_version": SCHEMA_VERSION,
        "source_contract_sha256": source_contract,
        "windows": windows,
    }
    manifest_raw = _canonical_json(manifest, newline=True)
    if len(manifest_raw) > _MAX_MANIFEST_BYTES:
        raise NotsofarNaturalOverlapPreparationError()
    output_fd = -1
    staged_fd = -1
    prepared: PreparedNotsofarNaturalOverlapFixture | None = None
    try:
        output_path, output_fd, output_identity = _new_private_output(
            output_dir,
            source_root=root,
        )
        written: dict[str, _PrivateFile] = {}
        for filename, raw in artifact_bytes.items():
            written[filename] = _write_private(output_fd, filename, raw)
        written[MANIFEST_FILENAME] = _write_private(
            output_fd,
            MANIFEST_FILENAME,
            manifest_raw,
        )
        manifest_identity = written[MANIFEST_FILENAME]
        receipt = {
            "accepted_license": LICENSE_ID,
            "evidence_scope": manifest["evidence_scope"],
            "fixture_id": lock.fixture_id,
            "kind": RECEIPT_KIND,
            "lock_recipe_sha256": lock.recipe_sha256,
            "overlap_metric": lock.overlap_metric,
            "manifest": {
                "file": MANIFEST_FILENAME,
                "sha256": manifest_identity.sha256,
                "size_bytes": manifest_identity.size_bytes,
            },
            "preparer_files": list(closure),
            "privacy": _receipt_privacy(),
            "production_evidence": lock.production_evidence,
            "schema_version": SCHEMA_VERSION,
            "source_contract_sha256": source_contract,
            "source_files": source_rows,
            "totals": {
                "audio_bytes": sum(len(raw) for raw in artifact_bytes.values()),
                "channels": CHANNEL_COUNT,
                "pcm_files": len(artifact_bytes),
                "windows": WINDOW_COUNT,
            },
        }
        receipt_raw = _canonical_json(receipt, newline=True)
        staged_fd, staged = _stage_terminal_receipt(output_fd, receipt_raw)

        def committed_result() -> LoadedNotsofarNaturalOverlapBundle:
            receipt_snapshot = state.receipt_snapshot
            if receipt_snapshot is None:
                raise NotsofarNaturalOverlapPreparationError()
            return LoadedNotsofarNaturalOverlapBundle(
                path=output_path / MANIFEST_FILENAME,
                digest=manifest_identity.sha256,
                receipt_sha256=staged.sha256,
                source_contract_sha256=source_contract,
                production_evidence=lock.production_evidence,
                windows=tuple(windows),
                identities=(
                    manifest_identity.snapshot,
                    receipt_snapshot,
                    *(written[name].snapshot for name in artifact_bytes),
                ),
            )

        def close_guard() -> None:
            current_lock = _load_lock(
                lock_path,
                test_source_injection=test_source_injection,
            )
            if current_lock != lock or _preparer_closure() != closure:
                raise NotsofarNaturalOverlapPreparationError()
            _verify_sources(root, root_identity, sources, lock.source_pins)
            if _has_git_ancestor(output_path.parent):
                raise NotsofarNaturalOverlapPreparationError()
            _verify_output(output_path, output_fd, output_identity)
            _exact_entries(output_fd, _expected_entries(receipt=False))
            for name, identity in written.items():
                maximum = (
                    _MAX_MANIFEST_BYTES
                    if name == MANIFEST_FILENAME
                    else _MAX_AUDIO_BYTES
                )
                _read_private(output_fd, name, maximum=maximum, expected=identity)

        _commit_terminal_receipt(
            output_path,
            output_fd,
            output_identity,
            staged_fd,
            staged,
            receipt_raw,
            state,
            commit_guard=close_guard,
        )
        prepared = committed_result()
        return prepared
    except BaseException as error:
        if state.committed:
            return committed_result()
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(error, NotsofarNaturalOverlapPreparationError):
            raise
        raise NotsofarNaturalOverlapPreparationError() from None
    finally:
        for descriptor in (staged_fd, output_fd):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass


def _bundle_root(value: Path | str) -> _BundleLocator:
    candidate = Path(value).expanduser()
    candidate = Path(os.path.abspath(candidate))
    try:
        supplied = candidate.lstat()
        manifest_snapshot: tuple[int, ...] | None = None
        if stat.S_ISDIR(supplied.st_mode):
            root = candidate
        elif stat.S_ISREG(supplied.st_mode) and candidate.name == MANIFEST_FILENAME:
            if (
                stat.S_IMODE(supplied.st_mode) != 0o600
                or supplied.st_uid != os.getuid()
                or supplied.st_nlink != 1
            ):
                raise NotsofarNaturalOverlapPreparationError()
            manifest_snapshot = _file_snapshot(supplied)
            root = candidate.parent
        else:
            raise NotsofarNaturalOverlapPreparationError()
        resolved = root.resolve(strict=True)
        if root != resolved:
            raise NotsofarNaturalOverlapPreparationError()
        metadata = resolved.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or metadata.st_uid != os.getuid()
            or _has_git_ancestor(resolved)
            or (
                stat.S_ISDIR(supplied.st_mode)
                and _directory_identity(supplied) != _directory_identity(metadata)
            )
        ):
            raise NotsofarNaturalOverlapPreparationError()
        if manifest_snapshot is not None:
            current = candidate.lstat()
            if (
                candidate.resolve(strict=True) != candidate
                or _file_snapshot(current) != manifest_snapshot
            ):
                raise NotsofarNaturalOverlapPreparationError()
        return _BundleLocator(
            resolved,
            _directory_identity(metadata),
            manifest_snapshot,
        )
    except NotsofarNaturalOverlapPreparationError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise NotsofarNaturalOverlapPreparationError() from None


def _role_payload(
    value: object,
    *,
    samples: int,
) -> dict[str, object]:
    fields = {
        "activity_relative_end_sample",
        "activity_relative_start_sample",
        "reference",
        "reference_sha256",
    }
    if (
        type(value) is not dict
        or any(type(key) is not str for key in value)
        or set(value) != fields
    ):
        raise NotsofarNaturalOverlapPreparationError()
    start = _integer(
        value["activity_relative_start_sample"],
        minimum=0,
        maximum=samples - 1,
    )
    end = _integer(
        value["activity_relative_end_sample"],
        minimum=start + 1,
        maximum=samples,
    )
    activity_samples = end - start
    if not (
        int(MIN_SEGMENT_SECONDS * SAMPLE_RATE_HZ)
        <= activity_samples
        <= MAX_QUANTIZED_INTERVAL_SAMPLES
    ):
        raise NotsofarNaturalOverlapPreparationError()
    reference = value.get("reference")
    reference_sha256 = _sha256(value["reference_sha256"])
    if (
        type(reference) is not str
        or not reference.strip()
        or len(reference) > 4096
        or hashlib.sha256(reference.encode("utf-8")).hexdigest() != reference_sha256
    ):
        raise NotsofarNaturalOverlapPreparationError()
    return {
        "activity_relative_end_sample": end,
        "activity_relative_start_sample": start,
        "reference": reference,
        "reference_sha256": reference_sha256,
    }


def _parse_manifest(
    raw: bytes,
) -> tuple[dict[str, object], tuple[Mapping[str, object], ...]]:
    value = _strict_json(raw, maximum=_MAX_MANIFEST_BYTES)
    _exact_json_tree(value)
    expected_keys = {
        "evidence_scope",
        "fixture_id",
        "kind",
        "lock_recipe_sha256",
        "overlap_metric",
        "production_evidence",
        "sample_format",
        "schema_version",
        "source_contract_sha256",
        "windows",
    }
    if (
        type(value) is not dict
        or any(type(key) is not str for key in value)
        or set(value) != expected_keys
    ):
        raise NotsofarNaturalOverlapPreparationError()
    _expect_exact_json(value["schema_version"], SCHEMA_VERSION)
    _expect_exact_json(value["kind"], MANIFEST_KIND)
    _expect_exact_json(value["evidence_scope"], _evidence_scope())
    _expect_exact_json(value["overlap_metric"], OVERLAP_METRIC)
    _expect_exact_json(value["sample_format"], _sample_format())
    fixture_id = value["fixture_id"]
    if type(fixture_id) is not str or _SAFE_ID_RE.fullmatch(fixture_id) is None:
        raise NotsofarNaturalOverlapPreparationError()
    production_evidence = value["production_evidence"]
    if type(production_evidence) is not bool:
        raise NotsofarNaturalOverlapPreparationError()
    windows_value = value["windows"]
    if type(windows_value) is not list or len(windows_value) != WINDOW_COUNT:
        raise NotsofarNaturalOverlapPreparationError()
    _sha256(value.get("lock_recipe_sha256"))
    _sha256(value.get("source_contract_sha256"))
    if _canonical_json(value, newline=True) != raw:
        raise NotsofarNaturalOverlapPreparationError()
    if (production_evidence is True) != (fixture_id == FIXTURE_ID):
        raise NotsofarNaturalOverlapPreparationError()
    if production_evidence is False and not fixture_id.startswith("synthetic-"):
        raise NotsofarNaturalOverlapPreparationError()
    # Sanitize the complete metadata tree before opening any audio leaf.
    declared_audio_bytes = 0
    window_keys = {"channels", "envelope", "overlap", "role_a", "role_b", "window_id"}
    channel_keys = {"case_id", "channel", "file", "samples", "sha256", "size_bytes"}
    for item in windows_value:
        if type(item) is not dict or set(item) != window_keys:
            raise NotsofarNaturalOverlapPreparationError()
    windows: list[Mapping[str, object]] = []
    for index, item in enumerate(windows_value):
        if type(item) is not dict or set(item) != window_keys:
            raise NotsofarNaturalOverlapPreparationError()
        window_id = f"notsofar-overlap-w{index:02d}"
        _expect_exact_json(item["window_id"], window_id)
        envelope = item["envelope"]
        overlap = item["overlap"]
        if (
            type(envelope) is not dict
            or set(envelope)
            != {
                "samples",
                "source_end_sample",
                "source_start_sample",
            }
            or type(overlap) is not dict
            or set(overlap)
            != {"relative_end_sample", "relative_start_sample", "samples"}
        ):
            raise NotsofarNaturalOverlapPreparationError()
        start = _integer(envelope["source_start_sample"], minimum=0, maximum=10_000_000)
        end = _integer(
            envelope["source_end_sample"], minimum=start + 1, maximum=10_000_000
        )
        samples = end - start
        declared_samples = _integer(
            envelope["samples"], minimum=1, maximum=MAX_QUANTIZED_INTERVAL_SAMPLES
        )
        if declared_samples != samples or samples > MAX_QUANTIZED_INTERVAL_SAMPLES:
            raise NotsofarNaturalOverlapPreparationError()
        role_a = _role_payload(item["role_a"], samples=samples)
        role_b = _role_payload(item["role_b"], samples=samples)
        overlap_start = _integer(
            overlap["relative_start_sample"], minimum=0, maximum=samples - 1
        )
        overlap_end = _integer(
            overlap["relative_end_sample"],
            minimum=overlap_start + int(MIN_OVERLAP_SECONDS * SAMPLE_RATE_HZ),
            maximum=samples,
        )
        overlap_samples = _integer(
            overlap["samples"], minimum=1, maximum=MAX_QUANTIZED_INTERVAL_SAMPLES
        )
        if (
            overlap_samples != overlap_end - overlap_start
            or overlap_start
            != max(
                role_a["activity_relative_start_sample"],
                role_b["activity_relative_start_sample"],
            )
            or overlap_end
            != min(
                role_a["activity_relative_end_sample"],
                role_b["activity_relative_end_sample"],
            )
        ):
            raise NotsofarNaturalOverlapPreparationError()
        channels = item["channels"]
        if type(channels) is not list or len(channels) != CHANNEL_COUNT:
            raise NotsofarNaturalOverlapPreparationError()
        parsed_channels: list[dict[str, object]] = []
        for channel_index, channel in enumerate(channels):
            channel_name = "channel-a" if channel_index == 0 else "channel-b"
            case_id = f"{window_id}-{channel_name}"
            filename = f"{case_id}.f32le"
            if type(channel) is not dict or set(channel) != channel_keys:
                raise NotsofarNaturalOverlapPreparationError()
            _expect_exact_json(channel["case_id"], case_id)
            _expect_exact_json(channel["channel"], channel_name)
            _expect_exact_json(channel["file"], filename)
            _expect_exact_json(channel["samples"], samples)
            _expect_exact_json(channel["size_bytes"], samples * 4)
            expected_sha = _sha256(channel["sha256"])
            declared_audio_bytes += samples * 4
            if declared_audio_bytes > MAX_DERIVED_PCM_BYTES:
                raise NotsofarNaturalOverlapPreparationError()
            parsed_channels.append(
                {
                    "case_id": case_id,
                    "channel": channel_name,
                    "file": filename,
                    "samples": samples,
                    "sha256": expected_sha,
                    "size_bytes": samples * 4,
                }
            )
        windows.append(
            {
                "channels": parsed_channels,
                "envelope": {
                    "samples": samples,
                    "source_end_sample": end,
                    "source_start_sample": start,
                },
                "overlap": {
                    "relative_end_sample": overlap_end,
                    "relative_start_sample": overlap_start,
                    "samples": overlap_samples,
                },
                "role_a": role_a,
                "role_b": role_b,
                "window_id": window_id,
            }
        )
    if (
        production_evidence is True
        and declared_audio_bytes != PRODUCTION_DERIVED_PCM_BYTES
    ):
        raise NotsofarNaturalOverlapPreparationError()

    return value, tuple(windows)


def _open_manifest_audio(
    directory_fd: int,
    windows: tuple[Mapping[str, object], ...],
    *,
    production_evidence: bool,
) -> tuple[_PrivateFile, ...]:
    if type(directory_fd) is not int or type(production_evidence) is not bool:
        raise NotsofarNaturalOverlapPreparationError()
    _validate_loaded_windows(windows)
    identities: list[_PrivateFile] = []
    total_bytes = 0
    for window in windows:
        channels = window["channels"]
        if type(channels) is not list:
            raise NotsofarNaturalOverlapPreparationError()
        for channel in channels:
            if type(channel) is not dict:
                raise NotsofarNaturalOverlapPreparationError()
            filename = channel["file"]
            samples = channel["samples"]
            expected_sha = channel["sha256"]
            if (
                type(filename) is not str
                or type(samples) is not int
                or type(expected_sha) is not str
            ):
                raise NotsofarNaturalOverlapPreparationError()
            audio_raw, identity = _read_private(
                directory_fd,
                filename,
                maximum=_MAX_AUDIO_BYTES,
            )
            try:
                values = np.frombuffer(audio_raw, dtype="<f4")
            except (BufferError, ValueError):
                raise NotsofarNaturalOverlapPreparationError() from None
            if (
                identity.sha256 != expected_sha
                or identity.size_bytes != samples * 4
                or values.size != samples
                or not np.isfinite(values).all()
            ):
                raise NotsofarNaturalOverlapPreparationError()
            identities.append(identity)
            total_bytes += identity.size_bytes
    if total_bytes > MAX_DERIVED_PCM_BYTES or (
        production_evidence is True and total_bytes != PRODUCTION_DERIVED_PCM_BYTES
    ):
        raise NotsofarNaturalOverlapPreparationError()
    return tuple(identities)


def _parse_receipt(
    raw: bytes,
    *,
    manifest: _PrivateFile,
    manifest_value: Mapping[str, object],
) -> dict[str, object]:
    value = _strict_json(raw, maximum=_MAX_RECEIPT_BYTES)
    _exact_json_tree(value)
    _exact_json_tree(manifest_value)
    manifest_keys = {
        "evidence_scope",
        "fixture_id",
        "kind",
        "lock_recipe_sha256",
        "overlap_metric",
        "production_evidence",
        "sample_format",
        "schema_version",
        "source_contract_sha256",
        "windows",
    }
    expected_keys = {
        "accepted_license",
        "evidence_scope",
        "fixture_id",
        "kind",
        "lock_recipe_sha256",
        "manifest",
        "overlap_metric",
        "preparer_files",
        "privacy",
        "production_evidence",
        "schema_version",
        "source_contract_sha256",
        "source_files",
        "totals",
    }
    if (
        type(value) is not dict
        or any(type(key) is not str for key in value)
        or set(value) != expected_keys
        or type(manifest_value) is not dict
        or any(type(key) is not str for key in manifest_value)
        or set(manifest_value) != manifest_keys
        or type(manifest) is not _PrivateFile
        or type(manifest.name) is not str
        or manifest.name != MANIFEST_FILENAME
        or type(manifest.size_bytes) is not int
        or not 1 <= manifest.size_bytes <= _MAX_MANIFEST_BYTES
        or type(manifest.sha256) is not str
        or _SHA256_RE.fullmatch(manifest.sha256) is None
        or type(manifest.snapshot) is not tuple
        or len(manifest.snapshot) != 9
        or any(type(item) is not int for item in manifest.snapshot)
    ):
        raise NotsofarNaturalOverlapPreparationError()
    _expect_exact_json(manifest_value["schema_version"], SCHEMA_VERSION)
    _expect_exact_json(manifest_value["kind"], MANIFEST_KIND)
    _expect_exact_json(manifest_value["evidence_scope"], _evidence_scope())
    _expect_exact_json(manifest_value["overlap_metric"], OVERLAP_METRIC)
    _expect_exact_json(manifest_value["sample_format"], _sample_format())
    manifest_fixture_id = manifest_value["fixture_id"]
    manifest_production = manifest_value["production_evidence"]
    manifest_windows = manifest_value["windows"]
    if (
        type(manifest_fixture_id) is not str
        or _SAFE_ID_RE.fullmatch(manifest_fixture_id) is None
        or type(manifest_production) is not bool
        or type(manifest_windows) is not list
        or len(manifest_windows) != WINDOW_COUNT
    ):
        raise NotsofarNaturalOverlapPreparationError()
    _sha256(manifest_value["lock_recipe_sha256"])
    _sha256(manifest_value["source_contract_sha256"])
    _expect_exact_json(value["accepted_license"], LICENSE_ID)
    _expect_exact_json(value["evidence_scope"], _evidence_scope())
    _expect_exact_json(value["fixture_id"], manifest_value["fixture_id"])
    _expect_exact_json(value["kind"], RECEIPT_KIND)
    lock_recipe = _sha256(value["lock_recipe_sha256"])
    _expect_exact_json(lock_recipe, manifest_value["lock_recipe_sha256"])
    _expect_exact_json(value["overlap_metric"], OVERLAP_METRIC)
    _expect_exact_json(value["overlap_metric"], manifest_value["overlap_metric"])
    production_evidence = value["production_evidence"]
    if (
        type(production_evidence) is not bool
        or production_evidence is not manifest_value["production_evidence"]
    ):
        raise NotsofarNaturalOverlapPreparationError()
    _expect_exact_json(value["schema_version"], SCHEMA_VERSION)
    source_contract = _sha256(value["source_contract_sha256"])
    _expect_exact_json(source_contract, manifest_value["source_contract_sha256"])
    _expect_exact_json(value["privacy"], _receipt_privacy())
    _expect_exact_json(
        value["manifest"],
        {
            "file": MANIFEST_FILENAME,
            "sha256": manifest.sha256,
            "size_bytes": manifest.size_bytes,
        },
    )
    if _canonical_json(value, newline=True) != raw:
        raise NotsofarNaturalOverlapPreparationError()
    preparer_files = value["preparer_files"]
    source_files = value["source_files"]
    totals = value["totals"]
    if (
        type(preparer_files) is not list
        or len(preparer_files) != len(_PREPARER_FILES)
        or type(source_files) is not list
        or len(source_files) != len(_SOURCE_PATHS)
        or type(totals) is not dict
        or any(type(key) is not str for key in totals)
        or set(totals) != {"audio_bytes", "channels", "pcm_files", "windows"}
    ):
        raise NotsofarNaturalOverlapPreparationError()
    audio_bytes = _integer(
        totals["audio_bytes"], minimum=1, maximum=MAX_DERIVED_PCM_BYTES
    )
    _expect_exact_json(totals["channels"], CHANNEL_COUNT)
    _expect_exact_json(totals["pcm_files"], WINDOW_COUNT * CHANNEL_COUNT)
    _expect_exact_json(totals["windows"], WINDOW_COUNT)
    if production_evidence is True and audio_bytes != PRODUCTION_DERIVED_PCM_BYTES:
        raise NotsofarNaturalOverlapPreparationError()
    for row, expected_path in zip(preparer_files, _PREPARER_FILES, strict=True):
        if (
            type(row) is not dict
            or any(type(key) is not str for key in row)
            or set(row) != {"path", "sha256", "size_bytes"}
            or type(row.get("size_bytes")) is not int
            or not 0 < row["size_bytes"] <= 1024 * 1024
        ):
            raise NotsofarNaturalOverlapPreparationError()
        _expect_exact_json(row["path"], expected_path)
        _sha256(row.get("sha256"))
    expected_roles = [inherited._SOURCE_PINS[path][0] for path in _SOURCE_PATHS]
    source_total = 0
    for row, role, relative in zip(
        source_files, expected_roles, _SOURCE_PATHS, strict=True
    ):
        maximum = (
            inherited._MAX_WAV_BYTES
            if relative.endswith(".wav")
            else inherited._MAX_JSON_BYTES
        )
        if (
            type(row) is not dict
            or any(type(key) is not str for key in row)
            or set(row) != {"role", "sha256", "size_bytes"}
            or type(row.get("size_bytes")) is not int
            or not 0 < row["size_bytes"] <= maximum
        ):
            raise NotsofarNaturalOverlapPreparationError()
        _expect_exact_json(row["role"], role)
        _sha256(row.get("sha256"))
        source_total += row["size_bytes"]
    if source_total > inherited.SOURCE_MAXIMUM_BYTES:
        raise NotsofarNaturalOverlapPreparationError()
    return value


def _validate_loaded_windows(
    windows: tuple[Mapping[str, object], ...],
) -> None:
    if type(windows) is not tuple or len(windows) != WINDOW_COUNT:
        raise NotsofarNaturalOverlapPreparationError()
    window_keys = {"channels", "envelope", "overlap", "role_a", "role_b", "window_id"}
    channel_keys = {"case_id", "channel", "file", "samples", "sha256", "size_bytes"}
    for index, window in enumerate(windows):
        if (
            type(window) is not dict
            or any(type(key) is not str for key in window)
            or set(window) != window_keys
        ):
            raise NotsofarNaturalOverlapPreparationError()
        _expect_exact_json(window["window_id"], f"notsofar-overlap-w{index:02d}")
        envelope = window["envelope"]
        overlap = window["overlap"]
        if (
            type(envelope) is not dict
            or any(type(key) is not str for key in envelope)
            or set(envelope) != {"samples", "source_end_sample", "source_start_sample"}
            or type(overlap) is not dict
            or any(type(key) is not str for key in overlap)
            or set(overlap)
            != {"relative_end_sample", "relative_start_sample", "samples"}
        ):
            raise NotsofarNaturalOverlapPreparationError()
        start = _integer(envelope["source_start_sample"], minimum=0, maximum=10_000_000)
        end = _integer(
            envelope["source_end_sample"], minimum=start + 1, maximum=10_000_000
        )
        samples = end - start
        _expect_exact_json(envelope["samples"], samples)
        role_a = _role_payload(window["role_a"], samples=samples)
        role_b = _role_payload(window["role_b"], samples=samples)
        overlap_start = _integer(
            overlap["relative_start_sample"], minimum=0, maximum=samples - 1
        )
        overlap_end = _integer(
            overlap["relative_end_sample"],
            minimum=overlap_start + int(MIN_OVERLAP_SECONDS * SAMPLE_RATE_HZ),
            maximum=samples,
        )
        _expect_exact_json(overlap["samples"], overlap_end - overlap_start)
        if overlap_start != max(
            role_a["activity_relative_start_sample"],
            role_b["activity_relative_start_sample"],
        ) or overlap_end != min(
            role_a["activity_relative_end_sample"],
            role_b["activity_relative_end_sample"],
        ):
            raise NotsofarNaturalOverlapPreparationError()
        channels = window["channels"]
        if type(channels) is not list or len(channels) != CHANNEL_COUNT:
            raise NotsofarNaturalOverlapPreparationError()
        for channel_index, channel in enumerate(channels):
            if (
                type(channel) is not dict
                or any(type(key) is not str for key in channel)
                or set(channel) != channel_keys
            ):
                raise NotsofarNaturalOverlapPreparationError()
            channel_name = "channel-a" if channel_index == 0 else "channel-b"
            case_id = f"notsofar-overlap-w{index:02d}-{channel_name}"
            _expect_exact_json(channel["case_id"], case_id)
            _expect_exact_json(channel["channel"], channel_name)
            _expect_exact_json(channel["file"], f"{case_id}.f32le")
            _expect_exact_json(channel["samples"], samples)
            _expect_exact_json(channel["size_bytes"], samples * 4)
            _sha256(channel["sha256"])


def _loaded_selection_rows(
    windows: tuple[Mapping[str, object], ...],
    *,
    include_references: bool = True,
) -> tuple[Mapping[str, object], ...]:
    if type(include_references) is not bool:
        raise NotsofarNaturalOverlapPreparationError()
    _validate_loaded_windows(windows)
    rows: list[Mapping[str, object]] = []
    for window in windows:
        if type(window) is not dict or any(type(key) is not str for key in window):
            raise NotsofarNaturalOverlapPreparationError()
        envelope = window["envelope"]
        overlap = window["overlap"]
        role_a = window["role_a"]
        role_b = window["role_b"]
        if any(type(item) is not dict for item in (envelope, overlap, role_a, role_b)):
            raise NotsofarNaturalOverlapPreparationError()
        row: dict[str, object] = {
            "end_sample": envelope["source_end_sample"],
            "overlap_relative_end_sample": overlap["relative_end_sample"],
            "overlap_relative_start_sample": overlap["relative_start_sample"],
            "role_a_relative_end_sample": role_a["activity_relative_end_sample"],
            "role_a_relative_start_sample": role_a["activity_relative_start_sample"],
            "role_b_relative_end_sample": role_b["activity_relative_end_sample"],
            "role_b_relative_start_sample": role_b["activity_relative_start_sample"],
            "start_sample": envelope["source_start_sample"],
        }
        if include_references:
            row.update(
                {
                    "reference_a_sha256": role_a["reference_sha256"],
                    "reference_b_sha256": role_b["reference_sha256"],
                }
            )
        rows.append(row)
    return tuple(rows)


def _loaded_artifact_rows(
    windows: tuple[Mapping[str, object], ...],
) -> tuple[Mapping[str, object], ...]:
    _validate_loaded_windows(windows)
    rows: list[Mapping[str, object]] = []
    for window in windows:
        if type(window) is not dict or any(type(key) is not str for key in window):
            raise NotsofarNaturalOverlapPreparationError()
        channels = window.get("channels")
        if type(channels) is not list:
            raise NotsofarNaturalOverlapPreparationError()
        for channel in channels:
            if type(channel) is not dict or any(
                type(key) is not str for key in channel
            ):
                raise NotsofarNaturalOverlapPreparationError()
            rows.append(
                {
                    "file": channel["file"],
                    "samples": channel["samples"],
                    "sha256": channel["sha256"],
                    "size_bytes": channel["size_bytes"],
                }
            )
    return tuple(rows)


def _loaded_commitment_rows(
    windows: tuple[Mapping[str, object], ...],
) -> tuple[Mapping[str, object], ...]:
    _validate_loaded_windows(windows)
    public_rows = _loaded_selection_rows(windows, include_references=False)
    rows: list[Mapping[str, object]] = []
    for public, window in zip(public_rows, windows, strict=True):
        if type(window) is not dict or any(type(key) is not str for key in window):
            raise NotsofarNaturalOverlapPreparationError()
        role_a = window.get("role_a")
        role_b = window.get("role_b")
        if type(role_a) is not dict or type(role_b) is not dict:
            raise NotsofarNaturalOverlapPreparationError()
        rows.append(
            {
                **public,
                "reference_a": role_a.get("reference"),
                "reference_b": role_b.get("reference"),
            }
        )
    return tuple(rows)


def _validate_authority_inputs(
    manifest: dict[str, object],
    receipt: dict[str, object],
    windows: tuple[Mapping[str, object], ...],
) -> None:
    if type(manifest) is not dict or type(receipt) is not dict:
        raise NotsofarNaturalOverlapPreparationError()
    _exact_json_tree(manifest)
    _exact_json_tree(receipt)
    _validate_loaded_windows(windows)
    manifest_keys = {
        "evidence_scope",
        "fixture_id",
        "kind",
        "lock_recipe_sha256",
        "overlap_metric",
        "production_evidence",
        "sample_format",
        "schema_version",
        "source_contract_sha256",
        "windows",
    }
    receipt_keys = {
        "accepted_license",
        "evidence_scope",
        "fixture_id",
        "kind",
        "lock_recipe_sha256",
        "manifest",
        "overlap_metric",
        "preparer_files",
        "privacy",
        "production_evidence",
        "schema_version",
        "source_contract_sha256",
        "source_files",
        "totals",
    }
    if (
        any(type(key) is not str for key in manifest)
        or set(manifest) != manifest_keys
        or any(type(key) is not str for key in receipt)
        or set(receipt) != receipt_keys
    ):
        raise NotsofarNaturalOverlapPreparationError()
    fixture_id = manifest["fixture_id"]
    if type(fixture_id) is not str or _SAFE_ID_RE.fullmatch(fixture_id) is None:
        raise NotsofarNaturalOverlapPreparationError()
    _expect_exact_json(manifest["kind"], MANIFEST_KIND)
    _expect_exact_json(manifest["schema_version"], SCHEMA_VERSION)
    _expect_exact_json(manifest["overlap_metric"], OVERLAP_METRIC)
    _expect_exact_json(manifest["evidence_scope"], _evidence_scope())
    _expect_exact_json(manifest["sample_format"], _sample_format())
    _sha256(manifest["lock_recipe_sha256"])
    _sha256(manifest["source_contract_sha256"])
    production = manifest["production_evidence"]
    if type(production) is not bool:
        raise NotsofarNaturalOverlapPreparationError()
    _expect_exact_json(manifest["windows"], list(windows))

    _expect_exact_json(receipt["accepted_license"], LICENSE_ID)
    _expect_exact_json(receipt["kind"], RECEIPT_KIND)
    _expect_exact_json(receipt["schema_version"], SCHEMA_VERSION)
    _expect_exact_json(receipt["overlap_metric"], OVERLAP_METRIC)
    _expect_exact_json(receipt["evidence_scope"], _evidence_scope())
    _expect_exact_json(receipt["privacy"], _receipt_privacy())
    _expect_exact_json(receipt["fixture_id"], fixture_id)
    _expect_exact_json(receipt["lock_recipe_sha256"], manifest["lock_recipe_sha256"])
    _expect_exact_json(
        receipt["source_contract_sha256"], manifest["source_contract_sha256"]
    )
    if receipt["production_evidence"] is not production:
        raise NotsofarNaturalOverlapPreparationError()
    manifest_row = receipt["manifest"]
    if (
        type(manifest_row) is not dict
        or any(type(key) is not str for key in manifest_row)
        or set(manifest_row) != {"file", "sha256", "size_bytes"}
    ):
        raise NotsofarNaturalOverlapPreparationError()
    _expect_exact_json(manifest_row["file"], MANIFEST_FILENAME)
    _sha256(manifest_row["sha256"])
    _integer(manifest_row["size_bytes"], minimum=1, maximum=_MAX_MANIFEST_BYTES)
    totals = receipt["totals"]
    if (
        type(totals) is not dict
        or any(type(key) is not str for key in totals)
        or set(totals) != {"audio_bytes", "channels", "pcm_files", "windows"}
    ):
        raise NotsofarNaturalOverlapPreparationError()
    _integer(totals["audio_bytes"], minimum=1, maximum=MAX_DERIVED_PCM_BYTES)
    _expect_exact_json(totals["channels"], CHANNEL_COUNT)
    _expect_exact_json(totals["pcm_files"], WINDOW_COUNT * CHANNEL_COUNT)
    _expect_exact_json(totals["windows"], WINDOW_COUNT)
    for name, expected_length, fields in (
        ("preparer_files", len(_PREPARER_FILES), {"path", "sha256", "size_bytes"}),
        ("source_files", len(_SOURCE_PATHS), {"role", "sha256", "size_bytes"}),
    ):
        rows = receipt[name]
        if type(rows) is not list or len(rows) != expected_length:
            raise NotsofarNaturalOverlapPreparationError()
        for row in rows:
            if (
                type(row) is not dict
                or any(type(key) is not str for key in row)
                or set(row) != fields
            ):
                raise NotsofarNaturalOverlapPreparationError()
            label = row["path"] if name == "preparer_files" else row["role"]
            if type(label) is not str:
                raise NotsofarNaturalOverlapPreparationError()
            _sha256(row["sha256"])
            _integer(
                row["size_bytes"], minimum=1, maximum=inherited.SOURCE_MAXIMUM_BYTES
            )


def _validate_loaded_authority(
    manifest: dict[str, object],
    receipt: dict[str, object],
    windows: tuple[Mapping[str, object], ...],
) -> None:
    _validate_authority_inputs(manifest, receipt, windows)
    selection_rows = _loaded_selection_rows(windows)
    artifact_rows = _loaded_artifact_rows(windows)
    source_rows = receipt["source_files"]
    preparer_rows = receipt["preparer_files"]
    if type(source_rows) is not list or type(preparer_rows) is not list:
        raise NotsofarNaturalOverlapPreparationError()
    closure = _preparer_closure()
    _expect_exact_json(preparer_rows, list(closure))
    if type(manifest.get("production_evidence")) is not bool:
        raise NotsofarNaturalOverlapPreparationError()

    production = manifest["production_evidence"] is True
    if production:
        lock = _load_lock(DEFAULT_LOCK, test_source_injection=None)
        expected_source_rows = tuple(
            {
                "role": lock.source_pins[relative][0],
                "sha256": lock.source_pins[relative][2],
                "size_bytes": lock.source_pins[relative][1],
            }
            for relative in _SOURCE_PATHS
        )
        public_selection_rows = _loaded_selection_rows(
            windows,
            include_references=False,
        )
        _expect_exact_json(manifest["fixture_id"], lock.fixture_id)
        _expect_exact_json(manifest["lock_recipe_sha256"], lock.recipe_sha256)
        _expect_exact_json(manifest["overlap_metric"], lock.overlap_metric)
        _expect_exact_json(source_rows, list(expected_source_rows))
        _expect_exact_json(list(public_selection_rows), list(lock.selection_rows))
        _expect_exact_json(list(artifact_rows), list(lock.artifact_pins))
        if (
            _selection_private_commitment_sha256(
                lock.fixture_id,
                _loaded_commitment_rows(windows),
            )
            != lock.selection_private_commitment_sha256
            or sum(row["size_bytes"] for row in artifact_rows)
            != PRODUCTION_DERIVED_PCM_BYTES
            or receipt["totals"]["audio_bytes"] != PRODUCTION_DERIVED_PCM_BYTES
        ):
            raise NotsofarNaturalOverlapPreparationError()
    else:
        lock = _Lock(
            fixture_id=manifest["fixture_id"],
            recipe_sha256=manifest["lock_recipe_sha256"],
            overlap_metric=manifest["overlap_metric"],
            source_lock_recipe_sha256="",
            source_pins={},
            selection_rows=selection_rows,
            selection_private_commitment_sha256=(
                _selection_private_commitment_sha256(
                    manifest["fixture_id"],
                    _loaded_commitment_rows(windows),
                )
            ),
            artifact_pins=artifact_rows,
            production_evidence=False,
        )
        if lock.recipe_sha256 == EXPECTED_LOCK_RECIPE_SHA256:
            raise NotsofarNaturalOverlapPreparationError()

    expected_contract = _source_contract_sha256(
        lock,
        source_rows,
        selection_rows,
        closure,
    )
    if expected_contract != receipt["source_contract_sha256"]:
        raise NotsofarNaturalOverlapPreparationError()


def load_notsofar1_natural_overlap_bundle(
    value: Path | str,
) -> LoadedNotsofarNaturalOverlapBundle:
    """Strictly reopen a terminal receipt, manifest, and all 14 PCM leaves."""

    locator = _bundle_root(value)
    root = locator.root
    try:
        with opened_directory_nofollow(root) as (stable, directory_fd):
            if stable != root:
                raise NotsofarNaturalOverlapPreparationError()
            root_identity = _directory_identity(os.fstat(directory_fd))
            if root_identity != locator.root_identity:
                raise NotsofarNaturalOverlapPreparationError()
            _exact_entries(directory_fd, _expected_entries(receipt=True))
            manifest_raw, manifest_identity = _read_private(
                directory_fd,
                MANIFEST_FILENAME,
                maximum=_MAX_MANIFEST_BYTES,
            )
            manifest_value, windows = _parse_manifest(manifest_raw)
            if (
                locator.manifest_snapshot is not None
                and manifest_identity.snapshot != locator.manifest_snapshot
            ):
                raise NotsofarNaturalOverlapPreparationError()
            receipt_raw, receipt_identity = _read_private(
                directory_fd,
                PREPARATION_RECEIPT_FILENAME,
                maximum=_MAX_RECEIPT_BYTES,
            )
            receipt = _parse_receipt(
                receipt_raw,
                manifest=manifest_identity,
                manifest_value=manifest_value,
            )
            _validate_loaded_authority(manifest_value, receipt, windows)
            artifact_identities = _open_manifest_audio(
                directory_fd,
                windows,
                production_evidence=manifest_value["production_evidence"],
            )
            if receipt["totals"]["audio_bytes"] != sum(
                item.size_bytes for item in artifact_identities
            ):
                raise NotsofarNaturalOverlapPreparationError()
            identities = (
                manifest_identity.snapshot,
                receipt_identity.snapshot,
                *(item.snapshot for item in artifact_identities),
            )
            _verify_output(root, directory_fd, root_identity)
            _exact_entries(directory_fd, _expected_entries(receipt=True))
            for identity in (
                manifest_identity,
                receipt_identity,
                *artifact_identities,
            ):
                if identity.name == MANIFEST_FILENAME:
                    maximum = _MAX_MANIFEST_BYTES
                elif identity.name == PREPARATION_RECEIPT_FILENAME:
                    maximum = _MAX_RECEIPT_BYTES
                else:
                    maximum = _MAX_AUDIO_BYTES
                _read_private(
                    directory_fd,
                    identity.name,
                    maximum=maximum,
                    expected=identity,
                )
            _verify_output(root, directory_fd, root_identity)
            _exact_entries(directory_fd, _expected_entries(receipt=True))
            return LoadedNotsofarNaturalOverlapBundle(
                path=root / MANIFEST_FILENAME,
                digest=manifest_identity.sha256,
                receipt_sha256=receipt_identity.sha256,
                source_contract_sha256=receipt["source_contract_sha256"],
                production_evidence=receipt["production_evidence"],
                windows=windows,
                identities=identities,
            )
    except NotsofarNaturalOverlapPreparationError:
        raise
    except (OSError, RuntimeError, ValueError, BoundedReadError):
        raise NotsofarNaturalOverlapPreparationError() from None


def verify_notsofar1_natural_overlap_bundle(
    bundle: LoadedNotsofarNaturalOverlapBundle,
) -> None:
    """Reject any mutation of an already-loaded bundle snapshot."""

    if type(bundle) is not LoadedNotsofarNaturalOverlapBundle:
        raise NotsofarNaturalOverlapPreparationError()
    path_type = type(Path("."))
    if (
        type(bundle.path) is not path_type
        or type(bundle.digest) is not str
        or _SHA256_RE.fullmatch(bundle.digest) is None
        or type(bundle.receipt_sha256) is not str
        or _SHA256_RE.fullmatch(bundle.receipt_sha256) is None
        or type(bundle.source_contract_sha256) is not str
        or _SHA256_RE.fullmatch(bundle.source_contract_sha256) is None
        or type(bundle.production_evidence) is not bool
        or type(bundle.windows) is not tuple
        or len(bundle.windows) != WINDOW_COUNT
        or type(bundle.identities) is not tuple
        or len(bundle.identities) != 2 + WINDOW_COUNT * CHANNEL_COUNT
    ):
        raise NotsofarNaturalOverlapPreparationError()

    _validate_loaded_windows(bundle.windows)
    items = 0
    stack: list[tuple[object, int]] = [(bundle.windows, 0)]
    while stack:
        current, depth = stack.pop()
        if depth > 16:
            raise NotsofarNaturalOverlapPreparationError()
        current_type = type(current)
        if current_type is dict:
            items += len(current)
            for key, value in current.items():  # type: ignore[union-attr]
                if type(key) is not str or len(key) > 128:
                    raise NotsofarNaturalOverlapPreparationError()
                stack.append((value, depth + 1))
        elif current_type in {list, tuple}:
            items += len(current)  # type: ignore[arg-type]
            stack.extend(
                (value, depth + 1)
                for value in current  # type: ignore[union-attr]
            )
        elif current_type is str:
            if len(current) > 16_384:
                raise NotsofarNaturalOverlapPreparationError()
        elif current_type is int:
            if not -(10**12) <= current <= 10**12:
                raise NotsofarNaturalOverlapPreparationError()
        elif current_type not in {bool, type(None)}:
            raise NotsofarNaturalOverlapPreparationError()
        if items > 100_000:
            raise NotsofarNaturalOverlapPreparationError()
    for identity in bundle.identities:
        if (
            type(identity) is not tuple
            or len(identity) != 9
            or any(type(value) is not int for value in identity)
        ):
            raise NotsofarNaturalOverlapPreparationError()
    reopened = load_notsofar1_natural_overlap_bundle(bundle.path)
    if (
        reopened.path != bundle.path
        or reopened.digest != bundle.digest
        or reopened.receipt_sha256 != bundle.receipt_sha256
        or reopened.source_contract_sha256 != bundle.source_contract_sha256
        or reopened.production_evidence != bundle.production_evidence
        or reopened.windows != bundle.windows
        or reopened.identities != bundle.identities
    ):
        raise NotsofarNaturalOverlapPreparationError()


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise NotsofarNaturalOverlapPreparationError()


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description="Prepare the pinned offline NOTSOFAR-1 natural-overlap fixture."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--accept-license", required=True)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        prepared = prepare_notsofar1_natural_overlap_fixture(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            accepted_license=args.accept_license,
            lock_path=args.lock,
        )
        print(
            json.dumps(
                {
                    "manifest_sha256": prepared.digest,
                    "ok": True,
                    "production_evidence": prepared.production_evidence,
                    "receipt_sha256": prepared.receipt_sha256,
                    "windows": len(prepared.windows),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    except Exception:
        print(json.dumps(_SAFE_ERROR, sort_keys=True, separators=(",", ":")))
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_LOCK",
    "FIXTURE_ID",
    "LICENSE_ID",
    "LoadedNotsofarNaturalOverlapBundle",
    "MANIFEST_FILENAME",
    "MAX_DERIVED_PCM_BYTES",
    "MINIMUM_OVERLAP_SECONDS",
    "NotsofarNaturalOverlapPreparationError",
    "PREPARATION_RECEIPT_FILENAME",
    "PRODUCTION_DERIVED_PCM_BYTES",
    "PRODUCTION_SELECTION_GEOMETRY",
    "PreparedNotsofarNaturalOverlapFixture",
    "SAMPLE_RATE_HZ",
    "SELECTION_SEED",
    "TestSourceInjection",
    "load_notsofar1_natural_overlap_bundle",
    "main",
    "prepare_notsofar1_natural_overlap_fixture",
    "verify_notsofar1_natural_overlap_bundle",
]
