"""Prepare the pinned, offline PriMock57 two-role conversation fixture.

The hard-WER and overlap products are deliberately separate.  The former is
an ordinary schema-v2 streaming corpus containing three annotation-proven
single-speaker intervals.  The latter is a custom diagnostic bundle whose two
references remain separate and for which no ordinary WER is defined.

No source is downloaded and neither product grants speaker identity, model
qualification, promotion, live-device, or natural mixed-microphone authority.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import struct
from typing import Callable, Iterator, Mapping, Sequence

import numpy as np

from tools.streaming_stt.bounded_io import (
    BoundedReadError,
    opened_directory_nofollow,
    read_regular_bounded,
)
from tools.streaming_stt.corpus import (
    CorpusError,
    CorpusProvenance,
    LoadedCorpus,
    load_corpus,
    verify_corpus_snapshot,
)
from tools.streaming_stt.corpus_writer import (
    CorpusWriteCase,
    CorpusWriterError,
    publish_private_corpus,
)


DEFAULT_LOCK = (
    Path(__file__).resolve().parent
    / "streaming_stt"
    / "primock57-consultation01-v1.lock.json"
)
FIXTURE_ID = "primock57-consultation01-v1"
LICENSE_ID = "CC-BY-4.0"
REVISION = "cd2ac707ad03cb4d2531f4ec6b90c659bf4357c5"
EXPECTED_RECIPE_SHA256 = (
    "2bc14c4114959fe1323d843ed6bb1b17cf81aa46f620b6805a98dd380a1e17ef"
)
SAMPLE_RATE_HZ = 16_000
OVERLAP_MANIFEST_FILENAME = "primock57-two-role-overlap-diagnostic-v1.json"
PREPARATION_RECEIPT_FILENAME = "preparation-receipt.json"
OVERLAP_KIND = "primock57-two-role-overlap-diagnostic-v1"
ISOLATED_RECEIPT_KIND = "primock57-isolated-preparation-receipt-v1"
OVERLAP_RECEIPT_KIND = "primock57-overlap-preparation-receipt-v1"
_ISOLATED_PURPOSE = (
    "PriMock57 consultation 01 marker-free, zero-other-role-intersection "
    "isolated hard WER"
)

_SOURCE_PATHS = (
    "audio/day1_consultation01_doctor.wav",
    "audio/day1_consultation01_patient.wav",
    "transcripts/day1_consultation01_doctor.TextGrid",
    "transcripts/day1_consultation01_patient.TextGrid",
    "LICENSE.md",
)
_SOURCE_ROLES = (
    "role-a-audio",
    "role-b-audio",
    "role-a-annotation",
    "role-b-annotation",
    "license",
)
_MAX_SOURCE_BYTES = 32 * 1024 * 1024
_MAX_WAV_BYTES = 16 * 1024 * 1024
_MAX_TEXTGRID_BYTES = 64 * 1024
_MAX_LICENSE_BYTES = 64 * 1024
_MAX_LOCK_BYTES = 64 * 1024
_MAX_RECEIPT_BYTES = 256 * 1024
_MAX_OVERLAP_MANIFEST_BYTES = 256 * 1024
_MAX_JSON_DEPTH = 16
_MAX_JSON_ITEMS = 100_000
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}\Z")
_INTERVAL_RE = re.compile(
    r"^[ \t]*intervals \[(?P<index>[1-9][0-9]*)\]:[ \t]*\r?\n"
    r"^[ \t]*xmin = (?P<start>[0-9]+(?:\.[0-9]+)?)[ \t]*\r?\n"
    r"^[ \t]*xmax = (?P<end>[0-9]+(?:\.[0-9]+)?)[ \t]*\r?\n"
    r'^[ \t]*text = "(?P<text>(?:""|[^"\r\n])*)"[ \t]*\r?$',
    re.MULTILINE,
)
_SAFE_ERROR = {
    "error": "primock57_conversation_fixture_prerequisites_unavailable",
    "ok": False,
}
_PREPARER_FILES = (
    "tools/__init__.py",
    "tools/streaming_stt/__init__.py",
    "tools/prepare_primock57_conversation_fixture.py",
    "core/__init__.py",
    "core/wer.py",
    "tools/streaming_stt/bounded_io.py",
    "tools/streaming_stt/corpus.py",
    "tools/streaming_stt/corpus_writer.py",
    "tools/streaming_stt/private_diagnostic_receipt.py",
    "tools/streaming_stt/protocol.py",
)


class Primock57PreparationError(RuntimeError):
    """A detail-free lock, source, annotation, or publication failure."""


@dataclass(frozen=True, slots=True)
class TestIntervalSelection:
    """One explicit, complete test-only replacement for a locked interval."""

    role: str
    interval_index: int
    start_seconds: Decimal
    end_seconds: Decimal
    reference_sha256: str


@dataclass(frozen=True, slots=True)
class TestOverlapPairSelection:
    """One explicit test-only replacement for a locked overlap pair."""

    role_a: TestIntervalSelection
    role_b: TestIntervalSelection
    overlap_start_seconds: Decimal
    overlap_end_seconds: Decimal


@dataclass(frozen=True, slots=True)
class TestSourceInjection:
    """A full-shape, unmistakably non-production synthetic source contract."""

    fixture_id: str
    file_pins: Mapping[str, tuple[int, str]] = field(repr=False)
    role_a_duration_seconds: Decimal
    role_b_duration_seconds: Decimal
    isolated_cases: tuple[TestIntervalSelection, ...] = field(repr=False)
    overlap_cases: tuple[TestOverlapPairSelection, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class LoadedPrimock57IsolatedBundle:
    corpus: LoadedCorpus
    receipt_sha256: str
    source_contract_sha256: str
    production_evidence: bool
    identities: tuple[tuple[int, ...], ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class LoadedPrimock57OverlapBundle:
    path: Path
    digest: str
    cases: tuple[Mapping[str, object], ...] = field(repr=False)
    receipt_sha256: str
    source_contract_sha256: str
    production_evidence: bool


@dataclass(frozen=True, slots=True)
class PreparedPrimock57Fixture:
    isolated_corpus: LoadedCorpus
    isolated_receipt_sha256: str
    overlap_bundle: LoadedPrimock57OverlapBundle
    overlap_receipt_sha256: str
    production_evidence: bool


@dataclass(slots=True)
class _BundleCommitState:
    """Share the terminal overlap-receipt boundary with the CLI frame."""

    pending_prepared: PreparedPrimock57Fixture | None = None
    pending_stdout: bytes | None = field(default=None, repr=False)
    committed: bool = False


class _LifecycleSignal(BaseException):
    """Translate HUP/TERM into bounded, commit-aware Python control flow."""

    def __init__(self, signum: int) -> None:
        super().__init__()
        self.signum = signum


@dataclass(frozen=True, slots=True)
class _Interval:
    role: str
    index: int
    start: Decimal
    end: Decimal
    raw_text: str = field(repr=False)
    reference: str | None = field(repr=False)


@dataclass(frozen=True, slots=True)
class _TextGrid:
    role: str
    duration: Decimal
    intervals: tuple[_Interval, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class _BoundSource:
    relative_path: str = field(repr=False)
    role: str
    raw: bytes = field(repr=False)
    size_bytes: int
    sha256: str
    snapshot: tuple[int, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class _SourceContract:
    fixture_id: str
    pins: Mapping[str, tuple[int, str]] = field(repr=False)
    role_a_duration: Decimal
    role_b_duration: Decimal
    isolated: tuple[TestIntervalSelection, ...] = field(repr=False)
    overlaps: tuple[TestOverlapPairSelection, ...] = field(repr=False)
    artifact_pins: Mapping[str, tuple[int, str]] = field(repr=False)
    production_evidence: bool
    lock_recipe_sha256: str


@dataclass(frozen=True, slots=True)
class _PrivateFile:
    name: str
    size_bytes: int
    sha256: str
    snapshot: tuple[int, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class _SourceDirectory:
    name: str
    descriptor: int = field(repr=False)
    snapshot: tuple[int, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class _SourceRoot:
    path: Path = field(repr=False)
    descriptor: int = field(repr=False)
    snapshot: tuple[int, ...] = field(repr=False)
    directories: Mapping[str, _SourceDirectory] = field(repr=False)


def _canonical_json(value: object, *, newline: bool = False) -> bytes:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (MemoryError, TypeError, UnicodeError, ValueError, OverflowError):
        raise Primock57PreparationError() from None
    return raw + (b"\n" if newline else b"")


def _strict_json(raw: bytes, *, maximum_bytes: int) -> object:
    if not raw or len(raw) > maximum_bytes:
        raise Primock57PreparationError()

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise Primock57PreparationError()
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_float=Decimal,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                Primock57PreparationError()
            ),
        )
    except (UnicodeError, ValueError, OverflowError, Primock57PreparationError):
        raise Primock57PreparationError() from None
    count = 0
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        if depth > _MAX_JSON_DEPTH:
            raise Primock57PreparationError()
        if isinstance(current, dict):
            count += len(current)
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            count += len(current)
            stack.extend((item, depth + 1) for item in current)
        elif isinstance(current, str):
            if len(current) > 16_384:
                raise Primock57PreparationError()
        elif current is not None and not isinstance(current, (bool, int, Decimal)):
            raise Primock57PreparationError()
        if count > _MAX_JSON_ITEMS:
            raise Primock57PreparationError()
    return value


def _decimal(value: object) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, Decimal)):
        raise Primock57PreparationError()
    try:
        result = Decimal(value)
    except (InvalidOperation, ValueError, OverflowError):
        raise Primock57PreparationError() from None
    if not result.is_finite():
        raise Primock57PreparationError()
    return result


def _sha256(value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise Primock57PreparationError()
    return value


def _selection(
    value: object, expected_role: str | None = None
) -> TestIntervalSelection:
    expected = {
        "interval_index",
        "start_seconds",
        "end_seconds",
        "reference_sha256",
    }
    if expected_role is None:
        expected.add("role")
    if not isinstance(value, dict) or set(value) != expected:
        raise Primock57PreparationError()
    role = expected_role if expected_role is not None else value.get("role")
    index = value.get("interval_index")
    start = _decimal(value.get("start_seconds"))
    end = _decimal(value.get("end_seconds"))
    if (
        role not in {"role-a", "role-b"}
        or (expected_role is not None and role != expected_role)
        or type(index) is not int
        or index <= 0
        or not Decimal(0) <= start < end
    ):
        raise Primock57PreparationError()
    return TestIntervalSelection(
        role, index, start, end, _sha256(value.get("reference_sha256"))
    )


def _locked_contract(value: object) -> _SourceContract:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "kind",
        "fixture_id",
        "license_id",
        "repository",
        "revision",
        "sample_format",
        "selection",
        "output_artifacts",
        "source",
        "privacy",
        "evidence_scope",
        "recipe_digest_rule",
        "recipe_sha256",
    }:
        raise Primock57PreparationError()
    recipe = _sha256(value.get("recipe_sha256"))
    body = dict(value)
    body.pop("recipe_sha256")
    if (
        recipe != EXPECTED_RECIPE_SHA256
        or hashlib.sha256(_canonical_json(body)).hexdigest() != recipe
        or value.get("schema_version") != 1
        or value.get("kind") != "primock57-conversation-fixture-lock-v1"
        or value.get("fixture_id") != FIXTURE_ID
        or value.get("license_id") != LICENSE_ID
        or value.get("repository") != "babylonhealth/primock57"
        or value.get("revision") != REVISION
        or value.get("recipe_digest_rule")
        != "sha256-canonical-json-without-recipe_sha256-v1"
        or value.get("sample_format")
        != {
            "channels": 1,
            "encoding": "f32le",
            "sample_conversion": "pcm-s16le-to-f32le-divide-32768-v1",
            "sample_rate_hz": SAMPLE_RATE_HZ,
        }
    ):
        raise Primock57PreparationError()
    source = value.get("source")
    selection = value.get("selection")
    output_artifacts = value.get("output_artifacts")
    if (
        not isinstance(source, dict)
        or not isinstance(selection, dict)
        or not isinstance(output_artifacts, dict)
        or set(output_artifacts) != {"isolated", "overlap"}
    ):
        raise Primock57PreparationError()
    files = source.get("files")
    if (
        source.get("consultation") != "day1_consultation01"
        or source.get("maximum_total_size_bytes") != 31_457_280
        or source.get("total_size_bytes") != 29_358_062
        or source.get("input_sample_format")
        != {
            "bits_per_sample": 16,
            "channels": 1,
            "encoding": "pcm-signed-little-endian",
            "sample_rate_hz": SAMPLE_RATE_HZ,
        }
        or not isinstance(files, list)
        or len(files) != len(_SOURCE_PATHS)
    ):
        raise Primock57PreparationError()
    pins: dict[str, tuple[int, str]] = {}
    for row, relative, role in zip(files, _SOURCE_PATHS, _SOURCE_ROLES, strict=True):
        if not isinstance(row, dict) or set(row) != {
            "path",
            "role",
            "size_bytes",
            "sha256",
            "git_blob",
        }:
            raise Primock57PreparationError()
        size = row.get("size_bytes")
        if (
            row.get("path") != relative
            or row.get("role") != role
            or type(size) is not int
            or size <= 0
            or not isinstance(row.get("git_blob"), str)
            or re.fullmatch(r"[0-9a-f]{40}", str(row.get("git_blob"))) is None
        ):
            raise Primock57PreparationError()
        pins[relative] = (size, _sha256(row.get("sha256")))
    if sum(size for size, _digest in pins.values()) != source.get("total_size_bytes"):
        raise Primock57PreparationError()
    isolated_root = selection.get("isolated")
    overlap_root = selection.get("overlap")
    if not isinstance(isolated_root, dict) or not isinstance(overlap_root, dict):
        raise Primock57PreparationError()
    isolated_rows = isolated_root.get("cases")
    overlap_rows = overlap_root.get("cases")
    if not isinstance(isolated_rows, list) or len(isolated_rows) != 3:
        raise Primock57PreparationError()
    isolated = tuple(_selection(row) for row in isolated_rows)
    if not isinstance(overlap_rows, list) or len(overlap_rows) != 3:
        raise Primock57PreparationError()
    overlaps: list[TestOverlapPairSelection] = []
    for row in overlap_rows:
        if not isinstance(row, dict) or set(row) != {
            "role_a",
            "role_b",
            "overlap_start_seconds",
            "overlap_end_seconds",
        }:
            raise Primock57PreparationError()
        overlaps.append(
            TestOverlapPairSelection(
                role_a=_selection(row.get("role_a"), "role-a"),
                role_b=_selection(row.get("role_b"), "role-b"),
                overlap_start_seconds=_decimal(row.get("overlap_start_seconds")),
                overlap_end_seconds=_decimal(row.get("overlap_end_seconds")),
            )
        )
    if (
        isolated_root.get("algorithm") != "fixed-zero-other-role-intersection-v1"
        or isolated_root.get("other_role_intersection_seconds") != "0"
        or isolated_root.get("reference_transform")
        != "primock57-marker-strip-collapse-space-v1"
        or isolated_root.get("scoreability") != "marker-free-nonempty-hard-wer-v1"
        or overlap_root.get("algorithm")
        != "fixed-marker-free-nonempty-disjoint-overlap-pairs-v1"
        or overlap_root.get("mix") != "average-two-masked-full-utterance-stems-v1"
        or overlap_root.get("mix_arithmetic")
        != "float32-add-then-multiply-float32-0.5-v1"
        or overlap_root.get("ordinary_wer") is not False
        or overlap_root.get("reference_contract") != "two-role-separate-references-v1"
        or overlap_root.get("reference_transform")
        != "primock57-marker-strip-collapse-space-v1"
        or overlap_root.get("scoreability") != "diagnostic-references-only-no-metric-v1"
        or selection.get("sample_rounding") != "floor-start-ceil-end-v1"
    ):
        raise Primock57PreparationError()
    isolated_artifacts = output_artifacts.get("isolated")
    overlap_artifacts = output_artifacts.get("overlap")
    if (
        not isinstance(isolated_artifacts, list)
        or len(isolated_artifacts) != 3
        or not isinstance(overlap_artifacts, list)
        or len(overlap_artifacts) != 3
    ):
        raise Primock57PreparationError()
    artifact_pins: dict[str, tuple[int, str]] = {}
    for index, (row, selected) in enumerate(
        zip(isolated_artifacts, isolated, strict=True)
    ):
        expected_name = f"primock57-isolated-{index:02d}.f32le"
        expected_samples = _sample_ceil(selected.end_seconds) - _sample_floor(
            selected.start_seconds
        )
        if (
            not isinstance(row, dict)
            or set(row) != {"file", "samples", "sha256"}
            or row.get("file") != expected_name
            or type(row.get("samples")) is not int
            or row.get("samples") != expected_samples
        ):
            raise Primock57PreparationError()
        artifact_pins[expected_name] = (
            expected_samples,
            _sha256(row.get("sha256")),
        )
    for index, (row, pair) in enumerate(zip(overlap_artifacts, overlaps, strict=True)):
        case_id = f"primock57-overlap-{index:02d}"
        source_start = min(
            _sample_floor(pair.role_a.start_seconds),
            _sample_floor(pair.role_b.start_seconds),
        )
        source_end = max(
            _sample_ceil(pair.role_a.end_seconds),
            _sample_ceil(pair.role_b.end_seconds),
        )
        samples = source_end - source_start
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "case_id",
                "mix_sha256",
                "role_a_sha256",
                "role_b_sha256",
                "samples",
            }
            or row.get("case_id") != case_id
            or type(row.get("samples")) is not int
            or row.get("samples") != samples
        ):
            raise Primock57PreparationError()
        for suffix, digest_key in (
            ("role-a", "role_a_sha256"),
            ("role-b", "role_b_sha256"),
            ("mix", "mix_sha256"),
        ):
            name = f"{case_id}-{suffix}.f32le"
            artifact_pins[name] = (samples, _sha256(row.get(digest_key)))
    if len(artifact_pins) != 12:
        raise Primock57PreparationError()
    return _SourceContract(
        fixture_id=FIXTURE_ID,
        pins=pins,
        role_a_duration=_decimal(source.get("role_a_duration_seconds")),
        role_b_duration=_decimal(source.get("role_b_duration_seconds")),
        isolated=isolated,
        overlaps=tuple(overlaps),
        artifact_pins=artifact_pins,
        production_evidence=True,
        lock_recipe_sha256=EXPECTED_RECIPE_SHA256,
    )


def _load_lock(path: Path | str) -> _SourceContract:
    try:
        raw = read_regular_bounded(Path(path), maximum_bytes=_MAX_LOCK_BYTES).data
    except BoundedReadError:
        raise Primock57PreparationError() from None
    return _locked_contract(_strict_json(raw, maximum_bytes=_MAX_LOCK_BYTES))


def _test_contract(
    injection: TestSourceInjection, locked: _SourceContract
) -> _SourceContract:
    if (
        type(injection) is not TestSourceInjection
        or not isinstance(injection.fixture_id, str)
        or _SAFE_ID_RE.fullmatch(injection.fixture_id) is None
        or not injection.fixture_id.startswith("synthetic-")
        or injection.fixture_id == FIXTURE_ID
        or set(injection.file_pins) != set(_SOURCE_PATHS)
        or type(injection.role_a_duration_seconds) is not Decimal
        or type(injection.role_b_duration_seconds) is not Decimal
        or not injection.role_a_duration_seconds.is_finite()
        or not injection.role_b_duration_seconds.is_finite()
        or injection.role_a_duration_seconds <= 0
        or injection.role_b_duration_seconds <= 0
        or type(injection.isolated_cases) is not tuple
        or len(injection.isolated_cases) != 3
        or type(injection.overlap_cases) is not tuple
        or len(injection.overlap_cases) != 3
    ):
        raise Primock57PreparationError()
    pins: dict[str, tuple[int, str]] = {}
    for relative in _SOURCE_PATHS:
        try:
            size, digest = injection.file_pins[relative]
        except (KeyError, TypeError, ValueError):
            raise Primock57PreparationError() from None
        maximum = (
            _MAX_WAV_BYTES
            if relative.endswith(".wav")
            else _MAX_TEXTGRID_BYTES
            if relative.endswith(".TextGrid")
            else _MAX_LICENSE_BYTES
        )
        if (
            type(size) is not int
            or not 0 < size <= maximum
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
        ):
            raise Primock57PreparationError()
        pins[relative] = (size, digest)
    if sum(size for size, _digest in pins.values()) > _MAX_SOURCE_BYTES:
        raise Primock57PreparationError()
    for item in injection.isolated_cases:
        _validate_test_interval_type(item)
    for pair in injection.overlap_cases:
        if type(pair) is not TestOverlapPairSelection:
            raise Primock57PreparationError()
        _validate_test_interval_type(pair.role_a, expected_role="role-a")
        _validate_test_interval_type(pair.role_b, expected_role="role-b")
        if (
            type(pair.overlap_start_seconds) is not Decimal
            or type(pair.overlap_end_seconds) is not Decimal
            or not pair.overlap_start_seconds.is_finite()
            or not pair.overlap_end_seconds.is_finite()
            or not Decimal(0) <= pair.overlap_start_seconds < pair.overlap_end_seconds
        ):
            raise Primock57PreparationError()
    return _SourceContract(
        fixture_id=injection.fixture_id,
        pins=pins,
        role_a_duration=injection.role_a_duration_seconds,
        role_b_duration=injection.role_b_duration_seconds,
        isolated=injection.isolated_cases,
        overlaps=injection.overlap_cases,
        artifact_pins={},
        production_evidence=False,
        lock_recipe_sha256=hashlib.sha256(
            _canonical_json(
                {
                    "fixture_id": injection.fixture_id,
                    "kind": "primock57-test-injection-lock-v1",
                }
            )
        ).hexdigest(),
    )


def _validate_test_interval_type(
    value: object, *, expected_role: str | None = None
) -> None:
    if (
        type(value) is not TestIntervalSelection
        or value.role not in {"role-a", "role-b"}
        or (expected_role is not None and value.role != expected_role)
        or type(value.interval_index) is not int
        or value.interval_index <= 0
        or type(value.start_seconds) is not Decimal
        or type(value.end_seconds) is not Decimal
        or not value.start_seconds.is_finite()
        or not value.end_seconds.is_finite()
        or not Decimal(0) <= value.start_seconds < value.end_seconds
        or not isinstance(value.reference_sha256, str)
        or _SHA256_RE.fullmatch(value.reference_sha256) is None
    ):
        raise Primock57PreparationError()


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


def _git_marker_present(directory_fd: int) -> bool:
    """Recognize a real Git marker while ignoring an empty test sentinel."""

    marker_fd = -1
    try:
        try:
            marker = os.stat(".git", dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        if stat.S_ISREG(marker.st_mode):
            return marker.st_size > 0
        if stat.S_ISLNK(marker.st_mode) or not stat.S_ISDIR(marker.st_mode):
            return True
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        marker_fd = os.open(".git", flags, dir_fd=directory_fd)
        opened = os.fstat(marker_fd)
        if _directory_identity(opened) != _directory_identity(marker):
            raise Primock57PreparationError()
        try:
            os.stat("HEAD", dir_fd=marker_fd, follow_symlinks=False)
        except FileNotFoundError:
            has_head = False
        else:
            has_head = True
        current = os.stat(".git", dir_fd=directory_fd, follow_symlinks=False)
        if _directory_identity(os.fstat(marker_fd)) != _directory_identity(
            opened
        ) or _directory_identity(current) != _directory_identity(marker):
            raise Primock57PreparationError()
        return has_head
    except Primock57PreparationError:
        raise
    except (OSError, OverflowError, ValueError):
        raise Primock57PreparationError() from None
    finally:
        if marker_fd >= 0:
            try:
                os.close(marker_fd)
            except OSError:
                pass


def _has_git_ancestor(directory: Path) -> bool:
    try:
        for ancestor in (directory, *directory.parents):
            with opened_directory_nofollow(ancestor) as (_stable, descriptor):
                if _git_marker_present(descriptor):
                    return True
    except Primock57PreparationError:
        raise
    except (OSError, RuntimeError, ValueError, BoundedReadError):
        raise Primock57PreparationError() from None
    return False


def _riff_pcm16_mono(raw: bytes) -> np.ndarray:
    """Decode one exact 16-kHz mono PCM16 RIFF payload without an audio backend."""

    if len(raw) < 44 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise Primock57PreparationError()
    if struct.unpack_from("<I", raw, 4)[0] + 8 != len(raw):
        raise Primock57PreparationError()
    offset = 12
    format_chunk: bytes | None = None
    data_chunk: bytes | None = None
    while offset < len(raw):
        if offset + 8 > len(raw):
            raise Primock57PreparationError()
        chunk_id = raw[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", raw, offset + 4)[0]
        offset += 8
        end = offset + chunk_size
        padded_end = end + (chunk_size & 1)
        if end > len(raw) or padded_end > len(raw):
            raise Primock57PreparationError()
        payload = raw[offset:end]
        if chunk_id == b"fmt ":
            if format_chunk is not None:
                raise Primock57PreparationError()
            format_chunk = payload
        elif chunk_id == b"data":
            if data_chunk is not None:
                raise Primock57PreparationError()
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
        raise Primock57PreparationError()
    encoding, channels, rate, byte_rate, block_align, bits = struct.unpack(
        "<HHIIHH", format_chunk
    )
    if (
        encoding != 1
        or channels != 1
        or rate != SAMPLE_RATE_HZ
        or byte_rate != SAMPLE_RATE_HZ * 2
        or block_align != 2
        or bits != 16
    ):
        raise Primock57PreparationError()
    pcm = np.frombuffer(data_chunk, dtype="<i2")
    if pcm.size <= 0:
        raise Primock57PreparationError()
    return pcm


def _private_source_directory(metadata: os.stat_result) -> bool:
    return (
        stat.S_ISDIR(metadata.st_mode)
        and stat.S_IMODE(metadata.st_mode) == 0o700
        and metadata.st_uid == os.getuid()
    )


def _verify_source_root(root: _SourceRoot) -> None:
    try:
        opened = os.fstat(root.descriptor)
        lexical = root.path.lstat()
        if (
            not _private_source_directory(opened)
            or _file_snapshot(opened) != root.snapshot
            or _file_snapshot(lexical) != root.snapshot
            or root.path.resolve(strict=True) != root.path
        ):
            raise Primock57PreparationError()
        for name, retained in root.directories.items():
            current = os.fstat(retained.descriptor)
            entry = os.stat(name, dir_fd=root.descriptor, follow_symlinks=False)
            if (
                retained.name != name
                or not _private_source_directory(current)
                or _file_snapshot(current) != retained.snapshot
                or _file_snapshot(entry) != retained.snapshot
            ):
                raise Primock57PreparationError()
    except Primock57PreparationError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise Primock57PreparationError() from None


def _source_root(value: Path | str) -> _SourceRoot:
    supplied = Path(value).expanduser()
    candidate = Path(os.path.abspath(supplied))
    root_fd = -1
    child_fds: list[int] = []
    prepared = False
    try:
        resolved = candidate.resolve(strict=True)
        if resolved != candidate:
            raise Primock57PreparationError()
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        root_fd = os.open(resolved, flags)
        opened = os.fstat(root_fd)
        lexical = resolved.lstat()
        root_snapshot = _file_snapshot(opened)
        if (
            not _private_source_directory(opened)
            or _file_snapshot(lexical) != root_snapshot
        ):
            raise Primock57PreparationError()
        directories: dict[str, _SourceDirectory] = {}
        for name in ("audio", "transcripts"):
            descriptor = os.open(name, flags, dir_fd=root_fd)
            child_fds.append(descriptor)
            child = os.fstat(descriptor)
            entry = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            snapshot = _file_snapshot(child)
            if (
                not _private_source_directory(child)
                or _file_snapshot(entry) != snapshot
            ):
                raise Primock57PreparationError()
            directories[name] = _SourceDirectory(name, descriptor, snapshot)
        binding = _SourceRoot(resolved, root_fd, root_snapshot, directories)
        _verify_source_root(binding)
        prepared = True
        return binding
    except Primock57PreparationError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise Primock57PreparationError() from None
    finally:
        if not prepared:
            for descriptor in reversed(child_fds):
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if root_fd >= 0:
                try:
                    os.close(root_fd)
                except OSError:
                    pass


def _close_source_root(root: _SourceRoot) -> None:
    for retained in reversed(tuple(root.directories.values())):
        try:
            os.close(retained.descriptor)
        except OSError:
            pass
    try:
        os.close(root.descriptor)
    except OSError:
        pass


def _read_source(
    root: _SourceRoot,
    relative: str,
    role: str,
    pin: tuple[int, str],
) -> _BoundSource:
    size, expected_digest = pin
    parts = PurePosixPath(relative).parts
    if relative not in _SOURCE_PATHS or len(parts) not in {1, 2}:
        raise Primock57PreparationError()
    parent_name = parts[0] if len(parts) == 2 else ""
    parent_fd = (
        root.directories[parent_name].descriptor if parent_name else root.descriptor
    )
    leaf = parts[-1]
    maximum = (
        _MAX_WAV_BYTES
        if relative.endswith(".wav")
        else _MAX_TEXTGRID_BYTES
        if relative.endswith(".TextGrid")
        else _MAX_LICENSE_BYTES
    )
    descriptor = -1
    try:
        _verify_source_root(root)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(leaf, flags, dir_fd=parent_fd)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
            or before.st_size != size
            or before.st_size <= 0
            or before.st_size > maximum
        ):
            raise Primock57PreparationError()
        raw = bytearray()
        while len(raw) <= maximum:
            chunk = os.read(descriptor, min(65_536, maximum + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
        entry = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    except Primock57PreparationError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise Primock57PreparationError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
    digest = hashlib.sha256(raw).hexdigest()
    if (
        _file_snapshot(before) != _file_snapshot(after)
        or _file_snapshot(entry) != _file_snapshot(after)
        or len(raw) != size
        or digest != expected_digest
    ):
        raise Primock57PreparationError()
    return _BoundSource(relative, role, bytes(raw), size, digest, _file_snapshot(after))


def _verify_sources(
    root: _SourceRoot,
    saved: Mapping[str, _BoundSource],
    pins: Mapping[str, tuple[int, str]],
) -> None:
    _verify_source_root(root)
    for relative, role in zip(_SOURCE_PATHS, _SOURCE_ROLES, strict=True):
        current = _read_source(root, relative, role, pins[relative])
        previous = saved[relative]
        if current.snapshot != previous.snapshot or current.sha256 != previous.sha256:
            raise Primock57PreparationError()
    _verify_source_root(root)


def _reference(raw: str) -> str | None:
    if len(raw) > 4096 or any(ord(character) < 32 for character in raw):
        raise Primock57PreparationError()
    output: list[str] = []
    unsure = False
    position = 0
    while position < len(raw):
        marker = raw.find("<", position)
        stray = raw.find(">", position)
        if marker < 0:
            if stray >= 0:
                raise Primock57PreparationError()
            output.append(raw[position:])
            break
        if 0 <= stray < marker:
            raise Primock57PreparationError()
        output.append(raw[position:marker])
        end = raw.find(">", marker + 1)
        if end < 0:
            raise Primock57PreparationError()
        tag = raw[marker : end + 1]
        if tag == "<UNSURE>" and not unsure:
            unsure = True
        elif tag == "</UNSURE>" and unsure:
            unsure = False
        elif tag in {"<UNIN/>", "<INAUDIBLE_SPEECH/>"}:
            output.append(" ")
        else:
            raise Primock57PreparationError()
        position = end + 1
    if unsure:
        raise Primock57PreparationError()
    cleaned = " ".join("".join(output).split())
    return cleaned or None


def _parse_textgrid(
    raw: bytes,
    *,
    role: str,
    expected_duration: Decimal,
    production: bool,
) -> _TextGrid:
    if role not in {"role-a", "role-b"} or not raw or len(raw) > _MAX_TEXTGRID_BYTES:
        raise Primock57PreparationError()
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError:
        raise Primock57PreparationError() from None
    if (
        "\x00" in text
        or not text.endswith("\r\n")
        or "\n" in text.replace("\r\n", "")
        or text.count('class = "IntervalTier"') != 1
    ):
        raise Primock57PreparationError()
    expected_name = "Doctor" if role == "role-a" else "Patient"
    name_match = re.search(r'^[ \t]*name = "([^"\r\n]+)"[ \t]*\r?$', text, re.MULTILINE)
    size_match = re.search(
        r"^[ \t]*intervals: size = ([1-9][0-9]*)[ \t]*\r?$",
        text,
        re.MULTILINE,
    )
    root_max = re.search(r"^xmax = ([0-9]+(?:\.[0-9]+)?)[ \t]*\r?$", text, re.MULTILINE)
    tier_max = re.search(
        r"^[ \t]+xmax = ([0-9]+(?:\.[0-9]+)?)[ \t]*\r?$",
        text,
        re.MULTILINE,
    )
    if (
        not text.startswith('File type = "ooTextFile"')
        or 'Object class = "TextGrid"' not in text[:128]
        or name_match is None
        or (production and name_match.group(1) != expected_name)
        or (not production and len(name_match.group(1)) > 128)
        or size_match is None
        or root_max is None
        or tier_max is None
        or _decimal(root_max.group(1)) != expected_duration
        or _decimal(tier_max.group(1)) != expected_duration
    ):
        raise Primock57PreparationError()
    expected_count = int(size_match.group(1))
    matches = tuple(_INTERVAL_RE.finditer(text))
    if len(matches) != expected_count:
        raise Primock57PreparationError()
    intervals: list[_Interval] = []
    previous = Decimal(0)
    for expected_index, match in enumerate(matches, start=1):
        index = int(match.group("index"))
        start = _decimal(match.group("start"))
        end = _decimal(match.group("end"))
        raw_text = match.group("text").replace('""', '"')
        if (
            index != expected_index
            or start != previous
            or not Decimal(0) <= start < end <= expected_duration
        ):
            raise Primock57PreparationError()
        intervals.append(
            _Interval(role, index, start, end, raw_text, _reference(raw_text))
        )
        previous = end
    if previous != expected_duration:
        raise Primock57PreparationError()
    return _TextGrid(role, expected_duration, tuple(intervals))


def _output_path(value: Path | str, *, source_root: Path) -> Path:
    supplied = Path(value).expanduser()
    if not supplied.is_absolute():
        raise Primock57PreparationError()
    candidate = Path(os.path.abspath(supplied))
    try:
        parent = candidate.parent.resolve(strict=True)
        if (
            not candidate.name
            or candidate.name in {".", ".."}
            or candidate.exists()
            or candidate == source_root
            or source_root in candidate.parents
            or candidate in source_root.parents
            or _has_git_ancestor(parent)
        ):
            raise Primock57PreparationError()
        with opened_directory_nofollow(parent) as (stable, _descriptor):
            if stable != parent:
                raise Primock57PreparationError()
    except Primock57PreparationError:
        raise
    except (
        OSError,
        RuntimeError,
        ValueError,
        BoundedReadError,
    ):
        raise Primock57PreparationError() from None
    return candidate


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
            raise Primock57PreparationError() from None
        rows.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(loaded.data).hexdigest(),
                "size_bytes": len(loaded.data),
            }
        )
    return tuple(rows)


def _sample_floor(value: Decimal) -> int:
    return int((value * SAMPLE_RATE_HZ).to_integral_value(rounding=ROUND_FLOOR))


def _sample_ceil(value: Decimal) -> int:
    return int((value * SAMPLE_RATE_HZ).to_integral_value(rounding=ROUND_CEILING))


def _selected_interval(
    grids: Mapping[str, _TextGrid],
    selection: TestIntervalSelection,
) -> _Interval:
    rows = grids[selection.role].intervals
    if selection.interval_index > len(rows):
        raise Primock57PreparationError()
    row = rows[selection.interval_index - 1]
    if (
        row.index != selection.interval_index
        or row.start != selection.start_seconds
        or row.end != selection.end_seconds
        or row.reference is None
        or "<" in row.raw_text
        or ">" in row.raw_text
        or hashlib.sha256(row.reference.encode("utf-8")).hexdigest()
        != selection.reference_sha256
    ):
        raise Primock57PreparationError()
    return row


def _intersects(left: _Interval, right: _Interval) -> bool:
    return max(left.start, right.start) < min(left.end, right.end)


def _validate_selections(
    grids: Mapping[str, _TextGrid],
    contract: _SourceContract,
) -> tuple[tuple[_Interval, ...], tuple[tuple[_Interval, _Interval], ...]]:
    isolated: list[_Interval] = []
    seen: set[tuple[str, int]] = set()
    for selection in contract.isolated:
        row = _selected_interval(grids, selection)
        key = (row.role, row.index)
        other_role = "role-b" if row.role == "role-a" else "role-a"
        start = _sample_floor(row.start)
        end = _sample_ceil(row.end)
        if (
            key in seen
            or start >= end
            or any(
                other.reference is not None and _intersects(row, other)
                for other in grids[other_role].intervals
            )
            or any(
                other.reference is not None
                and max(start, _sample_floor(other.start))
                < min(end, _sample_ceil(other.end))
                for other in grids[other_role].intervals
            )
        ):
            raise Primock57PreparationError()
        seen.add(key)
        isolated.append(row)
    pairs: list[tuple[_Interval, _Interval]] = []
    previous_envelope_end: Decimal | None = None
    for pair in contract.overlaps:
        left = _selected_interval(grids, pair.role_a)
        right = _selected_interval(grids, pair.role_b)
        overlap_start = max(left.start, right.start)
        overlap_end = min(left.end, right.end)
        envelope_start = min(left.start, right.start)
        envelope_end = max(left.end, right.end)
        if (
            overlap_start != pair.overlap_start_seconds
            or overlap_end != pair.overlap_end_seconds
            or overlap_end <= overlap_start
            or (
                previous_envelope_end is not None
                and envelope_start < previous_envelope_end
            )
            or _sample_ceil(overlap_end) <= _sample_floor(overlap_start)
        ):
            raise Primock57PreparationError()
        previous_envelope_end = envelope_end
        pairs.append((left, right))
    if len(isolated) != 3 or len(pairs) != 3:
        raise Primock57PreparationError()
    return tuple(isolated), tuple(pairs)


def _isolated_selection_rows(
    isolated: Sequence[_Interval],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, row in enumerate(isolated):
        start = _sample_floor(row.start)
        end = _sample_ceil(row.end)
        rows.append(
            {
                "case_id": f"primock57-isolated-{index:02d}",
                "end_sample": end,
                "interval_index": row.index,
                "reference_sha256": hashlib.sha256(
                    (row.reference or "").encode("utf-8")
                ).hexdigest(),
                "role": row.role,
                "samples": end - start,
                "start_sample": start,
            }
        )
    return rows


def _overlap_selection_rows(
    pairs: Sequence[tuple[_Interval, _Interval]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, (left, right) in enumerate(pairs):
        left_start = _sample_floor(left.start)
        left_end = _sample_ceil(left.end)
        right_start = _sample_floor(right.start)
        right_end = _sample_ceil(right.end)
        source_start = min(left_start, right_start)
        source_end = max(left_end, right_end)
        overlap_start = max(left_start, right_start)
        overlap_end = min(left_end, right_end)
        rows.append(
            {
                "case_id": f"primock57-overlap-{index:02d}",
                "overlap_end_sample": overlap_end,
                "overlap_relative_end_sample": overlap_end - source_start,
                "overlap_relative_start_sample": overlap_start - source_start,
                "overlap_samples": overlap_end - overlap_start,
                "overlap_start_sample": overlap_start,
                "role_a": {
                    "end_sample": left_end,
                    "interval_index": left.index,
                    "reference_sha256": hashlib.sha256(
                        (left.reference or "").encode("utf-8")
                    ).hexdigest(),
                    "role": left.role,
                    "start_sample": left_start,
                },
                "role_b": {
                    "end_sample": right_end,
                    "interval_index": right.index,
                    "reference_sha256": hashlib.sha256(
                        (right.reference or "").encode("utf-8")
                    ).hexdigest(),
                    "role": right.role,
                    "start_sample": right_start,
                },
                "source_end_sample": source_end,
                "source_start_sample": source_start,
            }
        )
    return rows


def _selection_sha256(
    isolated: Sequence[_Interval],
    pairs: Sequence[tuple[_Interval, _Interval]],
) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "isolated": _isolated_selection_rows(isolated),
                "overlap": _overlap_selection_rows(pairs),
            }
        )
    ).hexdigest()


def _isolated_selection_sha256(isolated: Sequence[_Interval]) -> str:
    return hashlib.sha256(
        _canonical_json(_isolated_selection_rows(isolated))
    ).hexdigest()


def _overlap_selection_sha256(
    pairs: Sequence[tuple[_Interval, _Interval]],
) -> str:
    return hashlib.sha256(_canonical_json(_overlap_selection_rows(pairs))).hexdigest()


def _contract_isolated_selection_sha256(contract: _SourceContract) -> str:
    rows = []
    for index, row in enumerate(contract.isolated):
        start = _sample_floor(row.start_seconds)
        end = _sample_ceil(row.end_seconds)
        rows.append(
            {
                "case_id": f"primock57-isolated-{index:02d}",
                "end_sample": end,
                "interval_index": row.interval_index,
                "reference_sha256": row.reference_sha256,
                "role": row.role,
                "samples": end - start,
                "start_sample": start,
            }
        )
    return hashlib.sha256(_canonical_json(rows)).hexdigest()


def _contract_overlap_selection_sha256(contract: _SourceContract) -> str:
    rows = []
    for index, pair in enumerate(contract.overlaps):
        left_start = _sample_floor(pair.role_a.start_seconds)
        left_end = _sample_ceil(pair.role_a.end_seconds)
        right_start = _sample_floor(pair.role_b.start_seconds)
        right_end = _sample_ceil(pair.role_b.end_seconds)
        source_start = min(left_start, right_start)
        source_end = max(left_end, right_end)
        overlap_start = max(left_start, right_start)
        overlap_end = min(left_end, right_end)
        rows.append(
            {
                "case_id": f"primock57-overlap-{index:02d}",
                "overlap_end_sample": overlap_end,
                "overlap_relative_end_sample": overlap_end - source_start,
                "overlap_relative_start_sample": overlap_start - source_start,
                "overlap_samples": overlap_end - overlap_start,
                "overlap_start_sample": overlap_start,
                "role_a": {
                    "end_sample": left_end,
                    "interval_index": pair.role_a.interval_index,
                    "reference_sha256": pair.role_a.reference_sha256,
                    "role": pair.role_a.role,
                    "start_sample": left_start,
                },
                "role_b": {
                    "end_sample": right_end,
                    "interval_index": pair.role_b.interval_index,
                    "reference_sha256": pair.role_b.reference_sha256,
                    "role": pair.role_b.role,
                    "start_sample": right_start,
                },
                "source_end_sample": source_end,
                "source_start_sample": source_start,
            }
        )
    return hashlib.sha256(_canonical_json(rows)).hexdigest()


def _directory_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_uid,
    )


def _safe_leaf(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or Path(value).name != value
        or value in {".", ".."}
        or re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", value) is None
    ):
        raise Primock57PreparationError()
    return value


@contextmanager
def _open_private_directory(
    path: Path | str,
) -> Iterator[tuple[Path, int, tuple[int, ...]]]:
    candidate = Path(os.path.abspath(Path(path).expanduser()))
    descriptor = -1
    try:
        if candidate.resolve(strict=True) != candidate:
            raise Primock57PreparationError()
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
            raise Primock57PreparationError()
        yield candidate, descriptor, snapshot
        after = os.fstat(descriptor)
        lexical_after = candidate.lstat()
        if (
            not stat.S_ISDIR(after.st_mode)
            or stat.S_IMODE(after.st_mode) != 0o700
            or after.st_uid != os.getuid()
            or _file_snapshot(after) != snapshot
            or _file_snapshot(lexical_after) != snapshot
            or candidate.resolve(strict=True) != candidate
        ):
            raise Primock57PreparationError()
    except Primock57PreparationError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise Primock57PreparationError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


@contextmanager
def _new_private_directory(
    path: Path,
) -> Iterator[tuple[Path, int, tuple[int, int, int, int]]]:
    descriptor = -1
    try:
        with opened_directory_nofollow(path.parent) as (parent, parent_fd):
            if parent != path.parent:
                raise Primock57PreparationError()
            os.mkdir(path.name, mode=0o700, dir_fd=parent_fd)
            created = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path.name, flags, dir_fd=parent_fd)
            os.fchmod(descriptor, 0o700)
            opened = os.fstat(descriptor)
            identity = _directory_identity(opened)
            if (
                not stat.S_ISDIR(created.st_mode)
                or _directory_identity(created) != identity
                or stat.S_IMODE(opened.st_mode) != 0o700
                or opened.st_uid != os.getuid()
            ):
                raise Primock57PreparationError()
            os.fsync(descriptor)
            os.fsync(parent_fd)
            yield path, descriptor, identity
    except Primock57PreparationError:
        raise
    except (OSError, RuntimeError, ValueError, BoundedReadError):
        raise Primock57PreparationError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _read_private_at(
    directory_fd: int,
    name: str,
    *,
    maximum_bytes: int,
    expected: _PrivateFile | None = None,
) -> tuple[bytes, _PrivateFile]:
    leaf = _safe_leaf(name)
    descriptor = -1
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(leaf, flags, dir_fd=directory_fd)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > maximum_bytes
        ):
            raise Primock57PreparationError()
        raw = bytearray()
        while len(raw) <= maximum_bytes:
            chunk = os.read(descriptor, min(65_536, maximum_bytes + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
        entry = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        current = _PrivateFile(
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
            raise Primock57PreparationError()
        return bytes(raw), current
    except Primock57PreparationError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise Primock57PreparationError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _verify_directory_binding(
    path: Path,
    directory_fd: int,
    identity: tuple[int, int, int, int],
) -> None:
    try:
        opened = os.fstat(directory_fd)
        lexical = path.lstat()
        if (
            _directory_identity(opened) != identity
            or _directory_identity(lexical) != identity
            or path.resolve(strict=True) != path
            or stat.S_IMODE(opened.st_mode) != 0o700
        ):
            raise Primock57PreparationError()
    except Primock57PreparationError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise Primock57PreparationError() from None


def _write_private_at(directory_fd: int, name: str, raw: bytes) -> _PrivateFile:
    leaf = _safe_leaf(name)
    if not isinstance(raw, bytes) or not raw:
        raise Primock57PreparationError()
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
                raise Primock57PreparationError()
            written += count
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_uid != os.getuid()
            or opened.st_nlink != 1
            or opened.st_size != len(raw)
        ):
            raise Primock57PreparationError()
        result = _PrivateFile(
            leaf,
            len(raw),
            hashlib.sha256(raw).hexdigest(),
            _file_snapshot(opened),
        )
    except Primock57PreparationError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise Primock57PreparationError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
    _read_private_at(directory_fd, leaf, maximum_bytes=len(raw), expected=result)
    return result


def _stage_terminal_receipt(directory_fd: int, raw: bytes) -> tuple[int, _PrivateFile]:
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
            raise Primock57PreparationError()
        temporary = getattr(os, "O_TMPFILE", 0)
        if not temporary:
            raise Primock57PreparationError()
        flags = os.O_RDWR | temporary | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(".", flags, 0o600, dir_fd=directory_fd)
        os.fchmod(descriptor, 0o600)
        view = memoryview(raw)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise Primock57PreparationError()
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
            raise Primock57PreparationError()
        os.lseek(descriptor, 0, os.SEEK_SET)
        check = bytearray()
        while len(check) <= len(raw):
            chunk = os.read(descriptor, min(65_536, len(raw) + 1 - len(check)))
            if not chunk:
                break
            check.extend(chunk)
        after = os.fstat(descriptor)
        if bytes(check) != raw or _file_snapshot(after) != _file_snapshot(opened):
            raise Primock57PreparationError()
        staged = _PrivateFile(
            PREPARATION_RECEIPT_FILENAME,
            len(raw),
            hashlib.sha256(raw).hexdigest(),
            _file_snapshot(opened),
        )
        prepared = True
        return descriptor, staged
    except Primock57PreparationError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise Primock57PreparationError() from None
    finally:
        if not prepared and descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _verify_staged_receipt(descriptor: int, expected: _PrivateFile) -> None:
    try:
        before = os.fstat(descriptor)
        if (
            expected.name != PREPARATION_RECEIPT_FILENAME
            or expected.size_bytes <= 0
            or expected.size_bytes > _MAX_RECEIPT_BYTES
            or not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_uid != os.getuid()
            or before.st_nlink != 0
            or _file_snapshot(before) != expected.snapshot
        ):
            raise Primock57PreparationError()
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
        current = _PrivateFile(
            PREPARATION_RECEIPT_FILENAME,
            len(raw),
            hashlib.sha256(raw).hexdigest(),
            _file_snapshot(after),
        )
        if current != expected or _file_snapshot(before) != _file_snapshot(after):
            raise Primock57PreparationError()
    except Primock57PreparationError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise Primock57PreparationError() from None


def _terminal_commit_guard(callback: Callable[[], None]) -> None:
    callback()


def _link_terminal_receipt(directory_fd: int, descriptor: int) -> None:
    try:
        os.link(
            f"/proc/self/fd/{descriptor}",
            PREPARATION_RECEIPT_FILENAME,
            dst_dir_fd=directory_fd,
            follow_symlinks=True,
        )
    except (OSError, RuntimeError, ValueError):
        raise Primock57PreparationError() from None


def _terminal_entry_identity(
    metadata: os.stat_result,
) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        stat.S_IMODE(metadata.st_mode),
        metadata.st_uid,
        metadata.st_size,
    )


def _commit_signal_numbers() -> set[int]:
    return {
        signum
        for signum in (
            getattr(signal, "SIGHUP", None),
            getattr(signal, "SIGINT", None),
            getattr(signal, "SIGTERM", None),
        )
        if type(signum) is int
    }


def _lifecycle_signal_numbers() -> tuple[int, ...]:
    return tuple(
        signum
        for signum in (
            getattr(signal, "SIGHUP", None),
            getattr(signal, "SIGTERM", None),
        )
        if type(signum) is int
    )


def _lifecycle_signal_handler(signum: int, _frame: object) -> None:
    raise _LifecycleSignal(signum)


def _commit_terminal_receipt(
    directory_fd: int,
    descriptor: int,
    state: _BundleCommitState,
    *,
    commit_guard: Callable[[], None],
    staged_receipt: _PrivateFile,
) -> None:
    if not hasattr(signal, "pthread_sigmask") or state.committed:
        raise Primock57PreparationError()
    try:
        staged = os.fstat(descriptor)
        expected_identity = _terminal_entry_identity(staged)
        if (
            not stat.S_ISREG(staged.st_mode)
            or stat.S_IMODE(staged.st_mode) != 0o600
            or staged.st_uid != os.getuid()
            or staged.st_nlink != 0
            or staged.st_size <= 0
            or _file_snapshot(staged) != staged_receipt.snapshot
        ):
            raise Primock57PreparationError()
    except Primock57PreparationError:
        raise
    except OSError:
        raise Primock57PreparationError() from None
    previous_mask = signal.pthread_sigmask(
        signal.SIG_BLOCK,
        _commit_signal_numbers(),
    )
    link_returned = False
    try:
        try:
            _terminal_commit_guard(commit_guard)
            _verify_staged_receipt(descriptor, staged_receipt)
            _link_terminal_receipt(directory_fd, descriptor)
            link_returned = True
        finally:
            if link_returned:
                state.committed = True
            else:
                try:
                    opened = os.fstat(descriptor)
                    published = os.stat(
                        PREPARATION_RECEIPT_FILENAME,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                except OSError:
                    pass
                else:
                    if (
                        _terminal_entry_identity(opened) == expected_identity
                        and _terminal_entry_identity(published) == expected_identity
                        and opened.st_nlink == 1
                        and published.st_nlink == 1
                    ):
                        state.committed = True
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def _source_rows(sources: Mapping[str, _BoundSource]) -> list[dict[str, object]]:
    return [
        {
            "role": sources[relative].role,
            "sha256": sources[relative].sha256,
            "size_bytes": sources[relative].size_bytes,
        }
        for relative in _SOURCE_PATHS
    ]


def _contract_source_rows(contract: _SourceContract) -> list[dict[str, object]]:
    return [
        {
            "role": role,
            "sha256": contract.pins[relative][1],
            "size_bytes": contract.pins[relative][0],
        }
        for relative, role in zip(_SOURCE_PATHS, _SOURCE_ROLES, strict=True)
    ]


def _validated_source_rows(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or len(value) != len(_SOURCE_ROLES):
        raise Primock57PreparationError()
    rows: list[dict[str, object]] = []
    for index, (row, role) in enumerate(zip(value, _SOURCE_ROLES, strict=True)):
        if not isinstance(row, dict) or set(row) != {"role", "sha256", "size_bytes"}:
            raise Primock57PreparationError()
        size = row.get("size_bytes")
        maximum = (
            _MAX_WAV_BYTES
            if index < 2
            else _MAX_TEXTGRID_BYTES
            if index < 4
            else _MAX_LICENSE_BYTES
        )
        if row.get("role") != role or type(size) is not int or not 0 < size <= maximum:
            raise Primock57PreparationError()
        rows.append(
            {
                "role": role,
                "sha256": _sha256(row.get("sha256")),
                "size_bytes": size,
            }
        )
    if sum(int(row["size_bytes"]) for row in rows) > _MAX_SOURCE_BYTES:
        raise Primock57PreparationError()
    return rows


def _validated_preparer_rows(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list) or len(value) != len(_PREPARER_FILES):
        raise Primock57PreparationError()
    rows: list[dict[str, object]] = []
    for row, expected_path in zip(value, _PREPARER_FILES, strict=True):
        if not isinstance(row, dict) or set(row) != {"path", "sha256", "size_bytes"}:
            raise Primock57PreparationError()
        size = row.get("size_bytes")
        if (
            row.get("path") != expected_path
            or type(size) is not int
            or not 0 < size <= 1024 * 1024
        ):
            raise Primock57PreparationError()
        rows.append(
            {
                "path": expected_path,
                "sha256": _sha256(row.get("sha256")),
                "size_bytes": size,
            }
        )
    return tuple(rows)


def _metadata_sha256(source_rows: Sequence[Mapping[str, object]]) -> str:
    return hashlib.sha256(_canonical_json(list(source_rows[2:4]))).hexdigest()


def _validate_evidence_identity(
    receipt: Mapping[str, object],
    *,
    expected_selection_sha256: str,
) -> _SourceContract | None:
    production = receipt.get("production_evidence") is True
    if production:
        locked = _load_lock(DEFAULT_LOCK)
        if (
            receipt.get("fixture_id") != locked.fixture_id
            or receipt.get("lock_recipe_sha256") != locked.lock_recipe_sha256
            or receipt.get("selection_sha256") != expected_selection_sha256
            or receipt.get("source_files") != _contract_source_rows(locked)
        ):
            raise Primock57PreparationError()
        return locked
    if (
        not isinstance(receipt.get("fixture_id"), str)
        or not str(receipt["fixture_id"]).startswith("synthetic-")
        or receipt.get("fixture_id") == FIXTURE_ID
        or receipt.get("lock_recipe_sha256") == EXPECTED_RECIPE_SHA256
    ):
        raise Primock57PreparationError()
    return None


def _source_contract_sha256(
    *,
    fixture_id: str,
    production_evidence: bool,
    source_rows: list[dict[str, object]],
    selection_sha256: str,
    preparer_files: tuple[dict[str, object], ...],
    lock_recipe_sha256: str = EXPECTED_RECIPE_SHA256,
) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "accepted_license": LICENSE_ID,
                "fixture_id": fixture_id,
                "lock_recipe_sha256": lock_recipe_sha256,
                "preparer_files": list(preparer_files),
                "production_evidence": production_evidence,
                "selection_sha256": selection_sha256,
                "source_files": source_rows,
            }
        )
    ).hexdigest()


def _isolated_receipt_payload(
    *,
    fixture_id: str,
    production_evidence: bool,
    source_rows: list[dict[str, object]],
    selection_sha256: str,
    preparer_files: tuple[dict[str, object], ...],
    lock_recipe_sha256: str,
) -> bytes:
    return _canonical_json(
        {
            "accepted_license": LICENSE_ID,
            "fixture_id": fixture_id,
            "kind": ISOLATED_RECEIPT_KIND,
            "lock_recipe_sha256": lock_recipe_sha256,
            "preparer_files": list(preparer_files),
            "privacy": {
                "local_paths_in_receipt": False,
                "raw_role_labels_in_receipt": False,
                "transcripts_in_receipt": False,
            },
            "production_evidence": production_evidence,
            "schema_version": 1,
            "selection_sha256": selection_sha256,
            "source_files": source_rows,
            "totals": {"cases": 3},
        },
        newline=True,
    )


def _receipt_payload(
    *,
    kind: str,
    fixture_id: str,
    production_evidence: bool,
    source_rows: list[dict[str, object]],
    selection_sha256: str,
    preparer_files: tuple[dict[str, object], ...],
    source_contract_sha256: str,
    manifest: _PrivateFile,
    totals: Mapping[str, int],
    lock_recipe_sha256: str,
) -> bytes:
    return _canonical_json(
        {
            "accepted_license": LICENSE_ID,
            "fixture_id": fixture_id,
            "kind": kind,
            "lock_recipe_sha256": lock_recipe_sha256,
            "manifest": {
                "bytes": manifest.size_bytes,
                "file": manifest.name,
                "sha256": manifest.sha256,
            },
            "preparer_files": list(preparer_files),
            "privacy": {
                "local_paths_in_receipt": False,
                "raw_role_labels_in_receipt": False,
                "transcripts_in_receipt": False,
            },
            "production_evidence": production_evidence,
            "schema_version": 1,
            "selection_sha256": selection_sha256,
            "source_contract_sha256": source_contract_sha256,
            "source_files": source_rows,
            "totals": dict(totals),
        },
        newline=True,
    )


def _parse_receipt(
    raw: bytes,
    *,
    expected_kind: str,
    expected_manifest_name: str,
) -> tuple[dict[str, object], _PrivateFile]:
    value = _strict_json(raw, maximum_bytes=_MAX_RECEIPT_BYTES)
    if not isinstance(value, dict) or set(value) != {
        "accepted_license",
        "fixture_id",
        "kind",
        "lock_recipe_sha256",
        "manifest",
        "preparer_files",
        "privacy",
        "production_evidence",
        "schema_version",
        "selection_sha256",
        "source_contract_sha256",
        "source_files",
        "totals",
    }:
        raise Primock57PreparationError()
    manifest = value.get("manifest")
    if (
        _canonical_json(value, newline=True) != raw
        or value.get("schema_version") != 1
        or value.get("kind") != expected_kind
        or value.get("accepted_license") != LICENSE_ID
        or not isinstance(value.get("lock_recipe_sha256"), str)
        or _SHA256_RE.fullmatch(str(value.get("lock_recipe_sha256"))) is None
        or not isinstance(value.get("fixture_id"), str)
        or _SAFE_ID_RE.fullmatch(str(value.get("fixture_id"))) is None
        or type(value.get("production_evidence")) is not bool
        or value.get("privacy")
        != {
            "local_paths_in_receipt": False,
            "raw_role_labels_in_receipt": False,
            "transcripts_in_receipt": False,
        }
        or not isinstance(value.get("totals"), dict)
        or not isinstance(value.get("selection_sha256"), str)
        or _SHA256_RE.fullmatch(str(value.get("selection_sha256"))) is None
        or not isinstance(value.get("source_contract_sha256"), str)
        or _SHA256_RE.fullmatch(str(value.get("source_contract_sha256"))) is None
        or not isinstance(manifest, dict)
        or set(manifest) != {"bytes", "file", "sha256"}
        or manifest.get("file") != expected_manifest_name
        or type(manifest.get("bytes")) is not int
        or manifest.get("bytes", 0) <= 0
        or not isinstance(manifest.get("sha256"), str)
        or _SHA256_RE.fullmatch(str(manifest.get("sha256"))) is None
    ):
        raise Primock57PreparationError()
    source_rows = _validated_source_rows(value["source_files"])
    preparer_rows = _validated_preparer_rows(value["preparer_files"])
    value["source_files"] = source_rows
    value["preparer_files"] = list(preparer_rows)
    source_contract = _source_contract_sha256(
        fixture_id=str(value["fixture_id"]),
        production_evidence=bool(value["production_evidence"]),
        source_rows=source_rows,
        selection_sha256=str(value["selection_sha256"]),
        preparer_files=preparer_rows,
        lock_recipe_sha256=str(value["lock_recipe_sha256"]),
    )
    if source_contract != value["source_contract_sha256"]:
        raise Primock57PreparationError()
    return value, _PrivateFile(
        expected_manifest_name,
        int(manifest["bytes"]),
        str(manifest["sha256"]),
        (),
    )


def _normalized_pcm(pcm: np.ndarray) -> np.ndarray:
    if pcm.dtype != np.dtype("<i2") or pcm.ndim != 1 or pcm.size <= 0:
        raise Primock57PreparationError()
    normalized = pcm.astype(np.float32) / np.float32(32768.0)
    normalized = np.asarray(normalized, dtype="<f4")
    if normalized.shape != pcm.shape or not np.isfinite(normalized).all():
        raise Primock57PreparationError()
    return normalized


def _pcm_bytes(value: np.ndarray) -> bytes:
    if value.ndim != 1 or value.size <= 0 or not np.isfinite(value).all():
        raise Primock57PreparationError()
    raw = np.ascontiguousarray(value, dtype="<f4").tobytes()
    if len(raw) != value.size * 4 or len(raw) > _MAX_WAV_BYTES:
        raise Primock57PreparationError()
    return raw


def _build_overlap_manifest(
    *,
    fixture_id: str,
    production_evidence: bool,
    source_contract_sha256: str,
    lock_recipe_sha256: str,
    pairs: Sequence[tuple[_Interval, _Interval]],
    audio: Mapping[str, np.ndarray],
) -> tuple[bytes, dict[str, bytes]]:
    artifacts: dict[str, bytes] = {}
    cases: list[dict[str, object]] = []
    for index, (left, right) in enumerate(pairs):
        left_start = _sample_floor(left.start)
        left_end = _sample_ceil(left.end)
        right_start = _sample_floor(right.start)
        right_end = _sample_ceil(right.end)
        envelope_start = min(left_start, right_start)
        envelope_end = max(left_end, right_end)
        samples = envelope_end - envelope_start
        if (
            samples <= 0
            or left_end > audio["role-a"].size
            or right_end > audio["role-b"].size
        ):
            raise Primock57PreparationError()
        stem_a = np.zeros(samples, dtype=np.float32)
        stem_b = np.zeros(samples, dtype=np.float32)
        a_start = left_start - envelope_start
        a_end = left_end - envelope_start
        b_start = right_start - envelope_start
        b_end = right_end - envelope_start
        stem_a[a_start:a_end] = audio["role-a"][left_start:left_end]
        stem_b[b_start:b_end] = audio["role-b"][right_start:right_end]
        summed = np.add(stem_a, stem_b, dtype=np.float32)
        mix = np.multiply(summed, np.float32(0.5), dtype=np.float32)
        case_id = f"primock57-overlap-{index:02d}"
        names = {
            "role_a": f"{case_id}-role-a.f32le",
            "role_b": f"{case_id}-role-b.f32le",
            "mix": f"{case_id}-mix.f32le",
        }
        raw_a = _pcm_bytes(stem_a)
        raw_b = _pcm_bytes(stem_b)
        raw_mix = _pcm_bytes(mix)
        artifacts[names["role_a"]] = raw_a
        artifacts[names["role_b"]] = raw_b
        artifacts[names["mix"]] = raw_mix

        def role_payload(
            *,
            interval: _Interval,
            start: int,
            end: int,
            name: str,
            raw: bytes,
        ) -> dict[str, object]:
            if interval.reference is None:
                raise Primock57PreparationError()
            return {
                "activity_end_sample": end,
                "activity_start_sample": start,
                "bytes": len(raw),
                "file": name,
                "interval_index": interval.index,
                "reference": interval.reference,
                "reference_sha256": hashlib.sha256(
                    interval.reference.encode("utf-8")
                ).hexdigest(),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }

        overlap_start = max(left_start, right_start) - envelope_start
        overlap_end = min(left_end, right_end) - envelope_start
        cases.append(
            {
                "case_id": case_id,
                "envelope": {
                    "samples": samples,
                    "source_end_sample": envelope_end,
                    "source_start_sample": envelope_start,
                },
                "mix": {
                    "arithmetic": "float32-add-then-multiply-float32-0.5-v1",
                    "bytes": len(raw_mix),
                    "file": names["mix"],
                    "sha256": hashlib.sha256(raw_mix).hexdigest(),
                },
                "overlap": {
                    "relative_end_sample": overlap_end,
                    "relative_start_sample": overlap_start,
                    "samples": overlap_end - overlap_start,
                },
                "role_a": role_payload(
                    interval=left,
                    start=a_start,
                    end=a_end,
                    name=names["role_a"],
                    raw=raw_a,
                ),
                "role_b": role_payload(
                    interval=right,
                    start=b_start,
                    end=b_end,
                    name=names["role_b"],
                    raw=raw_b,
                ),
            }
        )
    manifest = {
        "cases": cases,
        "evidence_scope": {
            "diagnostic_only": True,
            "natural_conversation_alignment": True,
            "ordinary_wer": False,
            "original_device_mix": False,
            "overlap_metric": "not-implemented",
            "promotion_authority": False,
            "qualification_authority": False,
        },
        "fixture_id": fixture_id,
        "kind": OVERLAP_KIND,
        "lock_recipe_sha256": lock_recipe_sha256,
        "production_evidence": production_evidence,
        "sample_format": {
            "channels": 1,
            "encoding": "f32le",
            "sample_conversion": "pcm-s16le-to-f32le-divide-32768-v1",
            "sample_rate_hz": SAMPLE_RATE_HZ,
        },
        "schema_version": 1,
        "source_contract_sha256": source_contract_sha256,
    }
    return _canonical_json(manifest, newline=True), artifacts


def _publish_overlap_bundle(
    *,
    output_dir: Path,
    manifest_raw: bytes,
    artifacts: Mapping[str, bytes],
    receipt_raw: bytes,
    sources_guard: Callable[[], None],
    closure_guard: Callable[[], None],
    source_contract_sha256: str,
    production_evidence: bool,
    finalize: Callable[[LoadedPrimock57OverlapBundle], PreparedPrimock57Fixture],
    commit_state: _BundleCommitState | None = None,
) -> PreparedPrimock57Fixture:
    state = commit_state or _BundleCommitState()
    receipt_fd = -1
    try:
        if state.pending_prepared is not None or state.committed:
            raise Primock57PreparationError()
        _receipt, expected_manifest = _parse_receipt(
            receipt_raw,
            expected_kind=OVERLAP_RECEIPT_KIND,
            expected_manifest_name=OVERLAP_MANIFEST_FILENAME,
        )
        if (
            expected_manifest.size_bytes != len(manifest_raw)
            or expected_manifest.sha256 != hashlib.sha256(manifest_raw).hexdigest()
        ):
            raise Primock57PreparationError()
        with _new_private_directory(output_dir) as (path, directory_fd, identity):
            written: dict[str, _PrivateFile] = {}
            for name in sorted(artifacts):
                written[name] = _write_private_at(directory_fd, name, artifacts[name])
            manifest = _write_private_at(
                directory_fd,
                OVERLAP_MANIFEST_FILENAME,
                manifest_raw,
            )
            receipt_fd, staged_receipt = _stage_terminal_receipt(
                directory_fd,
                receipt_raw,
            )
            value = _strict_json(
                manifest_raw, maximum_bytes=_MAX_OVERLAP_MANIFEST_BYTES
            )
            if not isinstance(value, dict) or not isinstance(value.get("cases"), list):
                raise Primock57PreparationError()
            loaded = LoadedPrimock57OverlapBundle(
                path=path / OVERLAP_MANIFEST_FILENAME,
                digest=manifest.sha256,
                cases=tuple(value["cases"]),  # type: ignore[arg-type]
                receipt_sha256=staged_receipt.sha256,
                source_contract_sha256=source_contract_sha256,
                production_evidence=production_evidence,
            )
            prepared = finalize(loaded)
            if type(prepared) is not PreparedPrimock57Fixture:
                raise Primock57PreparationError()
            summary = _success_summary(prepared)
            state.pending_prepared = prepared
            state.pending_stdout = summary

            def close_guard() -> None:
                sources_guard()
                closure_guard()
                _verify_directory_binding(path, directory_fd, identity)
                for name, retained in written.items():
                    _read_private_at(
                        directory_fd,
                        name,
                        maximum_bytes=_MAX_WAV_BYTES,
                        expected=retained,
                    )
                _read_private_at(
                    directory_fd,
                    OVERLAP_MANIFEST_FILENAME,
                    maximum_bytes=_MAX_OVERLAP_MANIFEST_BYTES,
                    expected=manifest,
                )
                if set(os.listdir(directory_fd)) != {
                    *written,
                    OVERLAP_MANIFEST_FILENAME,
                }:
                    raise Primock57PreparationError()
                _verify_staged_receipt(receipt_fd, staged_receipt)

            try:
                os.fsync(directory_fd)
            except OSError:
                raise Primock57PreparationError() from None
            _commit_terminal_receipt(
                directory_fd,
                receipt_fd,
                state,
                commit_guard=close_guard,
                staged_receipt=staged_receipt,
            )
            return prepared
    except BaseException as error:
        if state.committed and state.pending_prepared is not None:
            return state.pending_prepared
        if isinstance(error, Primock57PreparationError):
            raise
        if isinstance(error, Exception):
            raise Primock57PreparationError() from None
        raise
    finally:
        if receipt_fd >= 0:
            try:
                os.close(receipt_fd)
            except OSError:
                pass


def _parse_isolated_receipt(raw: bytes) -> dict[str, object]:
    value = _strict_json(raw, maximum_bytes=_MAX_RECEIPT_BYTES)
    if not isinstance(value, dict) or set(value) != {
        "accepted_license",
        "fixture_id",
        "kind",
        "lock_recipe_sha256",
        "preparer_files",
        "privacy",
        "production_evidence",
        "schema_version",
        "selection_sha256",
        "source_files",
        "totals",
    }:
        raise Primock57PreparationError()
    if (
        _canonical_json(value, newline=True) != raw
        or value.get("schema_version") != 1
        or value.get("kind") != ISOLATED_RECEIPT_KIND
        or value.get("accepted_license") != LICENSE_ID
        or not isinstance(value.get("lock_recipe_sha256"), str)
        or _SHA256_RE.fullmatch(str(value.get("lock_recipe_sha256"))) is None
        or not isinstance(value.get("fixture_id"), str)
        or _SAFE_ID_RE.fullmatch(str(value.get("fixture_id"))) is None
        or type(value.get("production_evidence")) is not bool
        or value.get("privacy")
        != {
            "local_paths_in_receipt": False,
            "raw_role_labels_in_receipt": False,
            "transcripts_in_receipt": False,
        }
        or value.get("totals") != {"cases": 3}
        or not isinstance(value.get("selection_sha256"), str)
        or _SHA256_RE.fullmatch(str(value.get("selection_sha256"))) is None
    ):
        raise Primock57PreparationError()
    value["source_files"] = _validated_source_rows(value["source_files"])
    value["preparer_files"] = list(_validated_preparer_rows(value["preparer_files"]))
    return value


def load_primock57_isolated_bundle(
    output_dir: Path | str,
) -> LoadedPrimock57IsolatedBundle:
    try:
        with _open_private_directory(output_dir) as (path, directory_fd, identity):
            receipt_raw, receipt_file = _read_private_at(
                directory_fd,
                PREPARATION_RECEIPT_FILENAME,
                maximum_bytes=_MAX_RECEIPT_BYTES,
            )
            receipt = _parse_isolated_receipt(receipt_raw)
            manifest_raw, manifest_file = _read_private_at(
                directory_fd,
                "corpus.json",
                maximum_bytes=_MAX_OVERLAP_MANIFEST_BYTES,
            )
            corpus = load_corpus(path / "corpus.json")
            verify_corpus_snapshot(corpus)
            source_rows = receipt["source_files"]
            if (
                manifest_file.sha256 != corpus.digest
                or hashlib.sha256(manifest_raw).hexdigest() != corpus.digest
                or corpus.schema_version != 2
                or len(corpus.cases) != 3
                or corpus.purpose != _ISOLATED_PURPOSE
                or corpus.provenance is None
                or corpus.provenance.kind != "public-voice-v1"
                or corpus.provenance.suite != receipt["fixture_id"]
                or corpus.provenance.manifest_sha256 != receipt["lock_recipe_sha256"]
                or corpus.provenance.metadata_sha256 != _metadata_sha256(source_rows)  # type: ignore[arg-type]
                or corpus.provenance.source_set_sha256 != receipt_file.sha256
                or list(receipt["preparer_files"]) != list(_preparer_closure())
            ):
                raise Primock57PreparationError()
            expected_names = {
                PREPARATION_RECEIPT_FILENAME,
                "corpus.json",
            }
            identities: list[tuple[int, ...]] = [
                identity,
                receipt_file.snapshot,
                manifest_file.snapshot,
            ]
            for case in corpus.cases:
                expected_name = f"{case.case_id}.f32le"
                if case.source_path.name != expected_name:
                    raise Primock57PreparationError()
                raw, private_file = _read_private_at(
                    directory_fd,
                    expected_name,
                    maximum_bytes=_MAX_WAV_BYTES,
                )
                if (
                    private_file.size_bytes != case.samples * 4
                    or private_file.sha256 != case.sha256
                    or raw != case.audio_bytes
                ):
                    raise Primock57PreparationError()
                expected_names.add(expected_name)
                identities.append(private_file.snapshot)
            if set(os.listdir(directory_fd)) != expected_names:
                raise Primock57PreparationError()
            source_contract_sha256 = _source_contract_sha256(
                fixture_id=str(receipt["fixture_id"]),
                production_evidence=bool(receipt["production_evidence"]),
                source_rows=list(receipt["source_files"]),  # type: ignore[arg-type]
                selection_sha256=str(receipt["selection_sha256"]),
                preparer_files=tuple(receipt["preparer_files"]),  # type: ignore[arg-type]
                lock_recipe_sha256=str(receipt["lock_recipe_sha256"]),
            )
            locked = _validate_evidence_identity(
                receipt,
                expected_selection_sha256=str(receipt["selection_sha256"]),
            )
            if locked is not None:
                if receipt["selection_sha256"] != _contract_isolated_selection_sha256(
                    locked
                ):
                    raise Primock57PreparationError()
                for index, (case, selected) in enumerate(
                    zip(corpus.cases, locked.isolated, strict=True)
                ):
                    expected_samples = _sample_ceil(
                        selected.end_seconds
                    ) - _sample_floor(selected.start_seconds)
                    if (
                        case.case_id != f"primock57-isolated-{index:02d}"
                        or case.samples != expected_samples
                        or hashlib.sha256(
                            case.expected_text.encode("utf-8")
                        ).hexdigest()
                        != selected.reference_sha256
                        or case.assertion != "transcript"
                        or case.commands
                        or case.forbidden_commands
                        or case.tags != ("primock57", "isolated", selected.role)
                        or locked.artifact_pins.get(case.source_path.name)
                        != (case.samples, case.sha256)
                    ):
                        raise Primock57PreparationError()
            return LoadedPrimock57IsolatedBundle(
                corpus=corpus,
                receipt_sha256=receipt_file.sha256,
                source_contract_sha256=source_contract_sha256,
                production_evidence=bool(receipt["production_evidence"]),
                identities=tuple(identities),
            )
    except Primock57PreparationError:
        raise
    except (OSError, CorpusError, BoundedReadError, RuntimeError, ValueError):
        raise Primock57PreparationError() from None


def verify_primock57_isolated_bundle(
    bundle: LoadedPrimock57IsolatedBundle,
) -> None:
    """Reopen one retained bundle and reject content or inode replacement."""

    if type(bundle) is not LoadedPrimock57IsolatedBundle:
        raise Primock57PreparationError()
    current = load_primock57_isolated_bundle(bundle.corpus.path.parent)
    if current != bundle:
        raise Primock57PreparationError()


def _int_field(value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Primock57PreparationError()
    return value


def _artifact_fields(value: object, *, reference: bool) -> dict[str, object]:
    expected = {
        "activity_end_sample",
        "activity_start_sample",
        "bytes",
        "file",
        "sha256",
    }
    if reference:
        expected.update({"interval_index", "reference", "reference_sha256"})
    if not isinstance(value, dict) or set(value) != expected:
        raise Primock57PreparationError()
    _safe_leaf(value.get("file"))
    _sha256(value.get("sha256"))
    _int_field(value.get("bytes"), minimum=1)
    _int_field(value.get("activity_start_sample"))
    _int_field(value.get("activity_end_sample"), minimum=1)
    if reference:
        _int_field(value.get("interval_index"), minimum=1)
        text = value.get("reference")
        if (
            not isinstance(text, str)
            or not text
            or len(text) > 4096
            or hashlib.sha256(text.encode("utf-8")).hexdigest()
            != _sha256(value.get("reference_sha256"))
        ):
            raise Primock57PreparationError()
    return value


def load_primock57_overlap_bundle(
    output_dir: Path | str,
) -> LoadedPrimock57OverlapBundle:
    try:
        with _open_private_directory(output_dir) as (path, directory_fd, _identity):
            receipt_raw, receipt_file = _read_private_at(
                directory_fd,
                PREPARATION_RECEIPT_FILENAME,
                maximum_bytes=_MAX_RECEIPT_BYTES,
            )
            receipt, expected_manifest = _parse_receipt(
                receipt_raw,
                expected_kind=OVERLAP_RECEIPT_KIND,
                expected_manifest_name=OVERLAP_MANIFEST_FILENAME,
            )
            manifest_raw, manifest_file = _read_private_at(
                directory_fd,
                OVERLAP_MANIFEST_FILENAME,
                maximum_bytes=_MAX_OVERLAP_MANIFEST_BYTES,
            )
            if (
                manifest_file.size_bytes != expected_manifest.size_bytes
                or manifest_file.sha256 != expected_manifest.sha256
                or list(receipt["preparer_files"]) != list(_preparer_closure())
                or receipt.get("totals") != {"cases": 3, "pcm_files": 9}
            ):
                raise Primock57PreparationError()
            manifest = _strict_json(
                manifest_raw,
                maximum_bytes=_MAX_OVERLAP_MANIFEST_BYTES,
            )
            if not isinstance(manifest, dict) or set(manifest) != {
                "cases",
                "evidence_scope",
                "fixture_id",
                "kind",
                "lock_recipe_sha256",
                "production_evidence",
                "sample_format",
                "schema_version",
                "source_contract_sha256",
            }:
                raise Primock57PreparationError()
            cases = manifest.get("cases")
            if (
                _canonical_json(manifest, newline=True) != manifest_raw
                or manifest.get("schema_version") != 1
                or manifest.get("kind") != OVERLAP_KIND
                or manifest.get("fixture_id") != receipt["fixture_id"]
                or manifest.get("production_evidence")
                is not receipt["production_evidence"]
                or manifest.get("source_contract_sha256")
                != receipt["source_contract_sha256"]
                or manifest.get("lock_recipe_sha256") != receipt["lock_recipe_sha256"]
                or manifest.get("sample_format")
                != {
                    "channels": 1,
                    "encoding": "f32le",
                    "sample_conversion": "pcm-s16le-to-f32le-divide-32768-v1",
                    "sample_rate_hz": SAMPLE_RATE_HZ,
                }
                or manifest.get("evidence_scope")
                != {
                    "diagnostic_only": True,
                    "natural_conversation_alignment": True,
                    "ordinary_wer": False,
                    "original_device_mix": False,
                    "overlap_metric": "not-implemented",
                    "promotion_authority": False,
                    "qualification_authority": False,
                }
                or not isinstance(cases, list)
                or len(cases) != 3
            ):
                raise Primock57PreparationError()
            expected_names = {
                PREPARATION_RECEIPT_FILENAME,
                OVERLAP_MANIFEST_FILENAME,
            }
            observed_artifact_pins: dict[str, tuple[int, str]] = {}
            selection_rows: list[dict[str, object]] = []
            previous_source_end = -1
            for index, case in enumerate(cases):
                if not isinstance(case, dict) or set(case) != {
                    "case_id",
                    "envelope",
                    "mix",
                    "overlap",
                    "role_a",
                    "role_b",
                }:
                    raise Primock57PreparationError()
                if case.get("case_id") != f"primock57-overlap-{index:02d}":
                    raise Primock57PreparationError()
                envelope = case.get("envelope")
                overlap = case.get("overlap")
                mix = case.get("mix")
                if (
                    not isinstance(envelope, dict)
                    or set(envelope)
                    != {"samples", "source_end_sample", "source_start_sample"}
                    or not isinstance(overlap, dict)
                    or set(overlap)
                    != {"relative_end_sample", "relative_start_sample", "samples"}
                    or not isinstance(mix, dict)
                    or set(mix) != {"arithmetic", "bytes", "file", "sha256"}
                ):
                    raise Primock57PreparationError()
                samples = _int_field(envelope.get("samples"), minimum=1)
                source_start = _int_field(envelope.get("source_start_sample"))
                source_end = _int_field(envelope.get("source_end_sample"), minimum=1)
                overlap_start = _int_field(overlap.get("relative_start_sample"))
                overlap_end = _int_field(overlap.get("relative_end_sample"), minimum=1)
                overlap_samples = _int_field(overlap.get("samples"), minimum=1)
                if (
                    source_end - source_start != samples
                    or source_start < previous_source_end
                    or overlap_end - overlap_start != overlap_samples
                    or not 0 <= overlap_start < overlap_end <= samples
                    or mix.get("arithmetic")
                    != "float32-add-then-multiply-float32-0.5-v1"
                    or _int_field(mix.get("bytes"), minimum=1) != samples * 4
                ):
                    raise Primock57PreparationError()
                previous_source_end = source_end
                role_a = _artifact_fields(case.get("role_a"), reference=True)
                role_b = _artifact_fields(case.get("role_b"), reference=True)
                mix_name = _safe_leaf(mix.get("file"))
                mix_sha = _sha256(mix.get("sha256"))
                case_id = str(case["case_id"])
                if (
                    role_a["file"] != f"{case_id}-role-a.f32le"
                    or role_b["file"] != f"{case_id}-role-b.f32le"
                    or mix_name != f"{case_id}-mix.f32le"
                ):
                    raise Primock57PreparationError()
                arrays: dict[str, np.ndarray] = {}
                for key, row in (("role_a", role_a), ("role_b", role_b)):
                    name = _safe_leaf(row["file"])
                    raw, file = _read_private_at(
                        directory_fd,
                        name,
                        maximum_bytes=_MAX_WAV_BYTES,
                    )
                    if (
                        file.size_bytes != samples * 4
                        or file.size_bytes != row["bytes"]
                        or file.sha256 != row["sha256"]
                    ):
                        raise Primock57PreparationError()
                    array = np.frombuffer(raw, dtype="<f4")
                    start = _int_field(row["activity_start_sample"])
                    end = _int_field(row["activity_end_sample"], minimum=1)
                    if (
                        array.size != samples
                        or not np.isfinite(array).all()
                        or not 0 <= start < end <= samples
                        or np.any(array[:start] != 0)
                        or np.any(array[end:] != 0)
                    ):
                        raise Primock57PreparationError()
                    arrays[key] = array
                    expected_names.add(name)
                    observed_artifact_pins[name] = (samples, file.sha256)
                if (
                    max(
                        _int_field(role_a["activity_start_sample"]),
                        _int_field(role_b["activity_start_sample"]),
                    )
                    != overlap_start
                    or min(
                        _int_field(role_a["activity_end_sample"], minimum=1),
                        _int_field(role_b["activity_end_sample"], minimum=1),
                    )
                    != overlap_end
                ):
                    raise Primock57PreparationError()
                role_a_start = source_start + _int_field(
                    role_a["activity_start_sample"]
                )
                role_a_end = source_start + _int_field(
                    role_a["activity_end_sample"], minimum=1
                )
                role_b_start = source_start + _int_field(
                    role_b["activity_start_sample"]
                )
                role_b_end = source_start + _int_field(
                    role_b["activity_end_sample"], minimum=1
                )
                selection_rows.append(
                    {
                        "case_id": case_id,
                        "overlap_end_sample": source_start + overlap_end,
                        "overlap_relative_end_sample": overlap_end,
                        "overlap_relative_start_sample": overlap_start,
                        "overlap_samples": overlap_samples,
                        "overlap_start_sample": source_start + overlap_start,
                        "role_a": {
                            "end_sample": role_a_end,
                            "interval_index": _int_field(
                                role_a["interval_index"], minimum=1
                            ),
                            "reference_sha256": _sha256(role_a["reference_sha256"]),
                            "role": "role-a",
                            "start_sample": role_a_start,
                        },
                        "role_b": {
                            "end_sample": role_b_end,
                            "interval_index": _int_field(
                                role_b["interval_index"], minimum=1
                            ),
                            "reference_sha256": _sha256(role_b["reference_sha256"]),
                            "role": "role-b",
                            "start_sample": role_b_start,
                        },
                        "source_end_sample": source_end,
                        "source_start_sample": source_start,
                    }
                )
                mix_raw, mix_file = _read_private_at(
                    directory_fd,
                    mix_name,
                    maximum_bytes=_MAX_WAV_BYTES,
                )
                recomputed = np.multiply(
                    np.add(arrays["role_a"], arrays["role_b"], dtype=np.float32),
                    np.float32(0.5),
                    dtype=np.float32,
                )
                if (
                    mix_file.size_bytes != samples * 4
                    or mix_file.sha256 != mix_sha
                    or mix_raw != _pcm_bytes(recomputed)
                ):
                    raise Primock57PreparationError()
                expected_names.add(mix_name)
                observed_artifact_pins[mix_name] = (samples, mix_file.sha256)
            if set(os.listdir(directory_fd)) != expected_names:
                raise Primock57PreparationError()
            observed_selection_sha256 = hashlib.sha256(
                _canonical_json(selection_rows)
            ).hexdigest()
            if observed_selection_sha256 != receipt["selection_sha256"]:
                raise Primock57PreparationError()
            locked = _validate_evidence_identity(
                receipt,
                expected_selection_sha256=observed_selection_sha256,
            )
            if locked is not None and (
                observed_selection_sha256 != _contract_overlap_selection_sha256(locked)
                or observed_artifact_pins
                != {
                    name: pin
                    for name, pin in locked.artifact_pins.items()
                    if name.startswith("primock57-overlap-")
                }
            ):
                raise Primock57PreparationError()
            return LoadedPrimock57OverlapBundle(
                path=path / OVERLAP_MANIFEST_FILENAME,
                digest=manifest_file.sha256,
                cases=tuple(cases),
                receipt_sha256=receipt_file.sha256,
                source_contract_sha256=str(receipt["source_contract_sha256"]),
                production_evidence=bool(receipt["production_evidence"]),
            )
    except Primock57PreparationError:
        raise
    except (OSError, BoundedReadError, RuntimeError, ValueError, TypeError):
        raise Primock57PreparationError() from None


def prepare_primock57_conversation_fixture(
    *,
    source_dir: Path | str,
    isolated_output_dir: Path | str,
    overlap_output_dir: Path | str,
    accepted_license: str,
    lock_path: Path | str = DEFAULT_LOCK,
    test_source_injection: TestSourceInjection | None = None,
    _commit_state: _BundleCommitState | None = None,
) -> PreparedPrimock57Fixture:
    """Materialize the two independent, bounded PriMock57 products."""

    root_binding: _SourceRoot | None = None
    try:
        if accepted_license != LICENSE_ID:
            raise Primock57PreparationError()
        locked = _load_lock(lock_path)
        contract = (
            locked
            if test_source_injection is None
            else _test_contract(test_source_injection, locked)
        )
        root_binding = _source_root(source_dir)
        root = root_binding.path
        isolated_destination = _output_path(isolated_output_dir, source_root=root)
        overlap_destination = _output_path(overlap_output_dir, source_root=root)
        if (
            isolated_destination == overlap_destination
            or isolated_destination in overlap_destination.parents
            or overlap_destination in isolated_destination.parents
        ):
            raise Primock57PreparationError()

        sources = {
            relative: _read_source(
                root_binding,
                relative,
                role,
                contract.pins[relative],
            )
            for relative, role in zip(_SOURCE_PATHS, _SOURCE_ROLES, strict=True)
        }
        grids = {
            "role-a": _parse_textgrid(
                sources[_SOURCE_PATHS[2]].raw,
                role="role-a",
                expected_duration=contract.role_a_duration,
                production=contract.production_evidence,
            ),
            "role-b": _parse_textgrid(
                sources[_SOURCE_PATHS[3]].raw,
                role="role-b",
                expected_duration=contract.role_b_duration,
                production=contract.production_evidence,
            ),
        }
        pcm_a = _riff_pcm16_mono(sources[_SOURCE_PATHS[0]].raw)
        pcm_b = _riff_pcm16_mono(sources[_SOURCE_PATHS[1]].raw)
        expected_a = contract.role_a_duration * SAMPLE_RATE_HZ
        expected_b = contract.role_b_duration * SAMPLE_RATE_HZ
        if (
            expected_a != expected_a.to_integral_value()
            or expected_b != expected_b.to_integral_value()
            or pcm_a.size != int(expected_a)
            or pcm_b.size != int(expected_b)
        ):
            raise Primock57PreparationError()
        normalized = {
            "role-a": _normalized_pcm(pcm_a),
            "role-b": _normalized_pcm(pcm_b),
        }
        isolated, pairs = _validate_selections(grids, contract)
        preparer_files = _preparer_closure()
        source_rows = _source_rows(sources)
        isolated_selection_sha = _isolated_selection_sha256(isolated)
        overlap_selection_sha = _overlap_selection_sha256(pairs)
        overlap_source_contract = _source_contract_sha256(
            fixture_id=contract.fixture_id,
            production_evidence=contract.production_evidence,
            source_rows=source_rows,
            selection_sha256=overlap_selection_sha,
            preparer_files=preparer_files,
            lock_recipe_sha256=contract.lock_recipe_sha256,
        )

        isolated_receipt_raw = _isolated_receipt_payload(
            fixture_id=contract.fixture_id,
            production_evidence=contract.production_evidence,
            source_rows=source_rows,
            selection_sha256=isolated_selection_sha,
            preparer_files=preparer_files,
            lock_recipe_sha256=contract.lock_recipe_sha256,
        )
        isolated_receipt_sha = hashlib.sha256(isolated_receipt_raw).hexdigest()
        metadata_sha = _metadata_sha256(source_rows)
        write_cases: list[CorpusWriteCase] = []
        for index, row in enumerate(isolated):
            if row.reference is None:
                raise Primock57PreparationError()
            start = _sample_floor(row.start)
            end = _sample_ceil(row.end)
            write_cases.append(
                CorpusWriteCase(
                    case_id=f"primock57-isolated-{index:02d}",
                    audio_bytes=_pcm_bytes(normalized[row.role][start:end]),
                    reference=row.reference,
                    tags=("primock57", "isolated", row.role),
                )
            )
        provenance = CorpusProvenance(
            kind="public-voice-v1",
            suite=contract.fixture_id,
            manifest_sha256=contract.lock_recipe_sha256,
            metadata_sha256=metadata_sha,
            source_set_sha256=isolated_receipt_sha,
        )
        try:
            isolated_corpus = publish_private_corpus(
                cases=write_cases,
                provenance=provenance,
                output_dir=isolated_destination,
                purpose=_ISOLATED_PURPOSE,
                sidecars={PREPARATION_RECEIPT_FILENAME: isolated_receipt_raw},
            )
        except (CorpusWriterError, CorpusError):
            raise Primock57PreparationError() from None
        _verify_sources(root_binding, sources, contract.pins)
        if _preparer_closure() != preparer_files:
            raise Primock57PreparationError()
        isolated_bundle = load_primock57_isolated_bundle(isolated_destination)
        if (
            isolated_bundle.corpus.digest != isolated_corpus.digest
            or isolated_bundle.receipt_sha256 != isolated_receipt_sha
            or isolated_bundle.production_evidence != contract.production_evidence
        ):
            raise Primock57PreparationError()

        manifest_raw, overlap_artifacts = _build_overlap_manifest(
            fixture_id=contract.fixture_id,
            production_evidence=contract.production_evidence,
            source_contract_sha256=overlap_source_contract,
            lock_recipe_sha256=contract.lock_recipe_sha256,
            pairs=pairs,
            audio=normalized,
        )
        manifest_receipt = _PrivateFile(
            OVERLAP_MANIFEST_FILENAME,
            len(manifest_raw),
            hashlib.sha256(manifest_raw).hexdigest(),
            (),
        )
        overlap_receipt_raw = _receipt_payload(
            kind=OVERLAP_RECEIPT_KIND,
            fixture_id=contract.fixture_id,
            production_evidence=contract.production_evidence,
            source_rows=source_rows,
            selection_sha256=overlap_selection_sha,
            preparer_files=preparer_files,
            source_contract_sha256=overlap_source_contract,
            manifest=manifest_receipt,
            totals={"cases": 3, "pcm_files": 9},
            lock_recipe_sha256=contract.lock_recipe_sha256,
        )

        def sources_guard() -> None:
            _verify_sources(root_binding, sources, contract.pins)
            verify_primock57_isolated_bundle(isolated_bundle)

        def closure_guard() -> None:
            if _preparer_closure() != preparer_files:
                raise Primock57PreparationError()

        state = _commit_state or _BundleCommitState()

        def finalize(
            overlap_bundle: LoadedPrimock57OverlapBundle,
        ) -> PreparedPrimock57Fixture:
            return PreparedPrimock57Fixture(
                isolated_corpus=isolated_bundle.corpus,
                isolated_receipt_sha256=isolated_bundle.receipt_sha256,
                overlap_bundle=overlap_bundle,
                overlap_receipt_sha256=overlap_bundle.receipt_sha256,
                production_evidence=contract.production_evidence,
            )

        return _publish_overlap_bundle(
            output_dir=overlap_destination,
            manifest_raw=manifest_raw,
            artifacts=overlap_artifacts,
            receipt_raw=overlap_receipt_raw,
            sources_guard=sources_guard,
            closure_guard=closure_guard,
            source_contract_sha256=overlap_source_contract,
            production_evidence=contract.production_evidence,
            finalize=finalize,
            commit_state=state,
        )
    except Primock57PreparationError:
        raise
    except Exception:
        raise Primock57PreparationError() from None
    finally:
        if root_binding is not None:
            _close_source_root(root_binding)


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise Primock57PreparationError()


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description="Prepare the pinned offline PriMock57 conversation fixture."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--isolated-output-dir", type=Path, required=True)
    parser.add_argument("--overlap-output-dir", type=Path, required=True)
    parser.add_argument("--accept-license", required=True)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    return parser


def _success_summary(prepared: PreparedPrimock57Fixture) -> bytes:
    return _canonical_json(
        {
            "isolated_cases": len(prepared.isolated_corpus.cases),
            "isolated_manifest_sha256": prepared.isolated_corpus.digest,
            "isolated_receipt_sha256": prepared.isolated_receipt_sha256,
            "ok": True,
            "overlap_cases": len(prepared.overlap_bundle.cases),
            "overlap_manifest_sha256": prepared.overlap_bundle.digest,
            "overlap_receipt_sha256": prepared.overlap_receipt_sha256,
            "production_evidence": prepared.production_evidence,
        },
        newline=True,
    )


def _print_safely(raw: bytes) -> None:
    try:
        print(raw.decode("ascii"), end="", flush=True)
    except BaseException:
        pass


def _main(argv: Sequence[str] | None = None) -> int:
    state = _BundleCommitState()
    prepared: PreparedPrimock57Fixture | None = None
    try:
        args = _parser().parse_args(argv)
        prepared = prepare_primock57_conversation_fixture(
            source_dir=args.source_dir,
            isolated_output_dir=args.isolated_output_dir,
            overlap_output_dir=args.overlap_output_dir,
            accepted_license=args.accept_license,
            lock_path=args.lock,
            _commit_state=state,
        )
    except _LifecycleSignal as interrupted:
        if state.committed and state.pending_prepared is not None:
            prepared = state.pending_prepared
        else:
            _print_safely(_canonical_json(_SAFE_ERROR, newline=True))
            return 128 + interrupted.signum
    except KeyboardInterrupt:
        if state.committed and state.pending_prepared is not None:
            prepared = state.pending_prepared
        else:
            _print_safely(_canonical_json(_SAFE_ERROR, newline=True))
            return 130
    except Exception:
        if state.committed and state.pending_prepared is not None:
            prepared = state.pending_prepared
        else:
            _print_safely(_canonical_json(_SAFE_ERROR, newline=True))
            return 2

    if prepared is None:
        _print_safely(_canonical_json(_SAFE_ERROR, newline=True))
        return 2
    summary = state.pending_stdout or _success_summary(prepared)
    _print_safely(summary)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    previous_handlers: dict[int, object] = {}
    try:
        for signum in _lifecycle_signal_numbers():
            previous_handlers[signum] = signal.signal(
                signum,
                _lifecycle_signal_handler,
            )
    except _LifecycleSignal as interrupted:
        result = 128 + interrupted.signum
    except KeyboardInterrupt:
        result = 130
    except Exception:
        result = 2
    else:
        try:
            return _main(argv)
        finally:
            for signum, previous in reversed(tuple(previous_handlers.items())):
                try:
                    signal.signal(signum, previous)
                except BaseException:
                    pass

    for signum, previous in reversed(tuple(previous_handlers.items())):
        try:
            signal.signal(signum, previous)
        except BaseException:
            pass
    _print_safely(_canonical_json(_SAFE_ERROR, newline=True))
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_LOCK",
    "EXPECTED_RECIPE_SHA256",
    "FIXTURE_ID",
    "LICENSE_ID",
    "LoadedPrimock57IsolatedBundle",
    "LoadedPrimock57OverlapBundle",
    "OVERLAP_MANIFEST_FILENAME",
    "PreparedPrimock57Fixture",
    "Primock57PreparationError",
    "REVISION",
    "TestIntervalSelection",
    "TestOverlapPairSelection",
    "TestSourceInjection",
    "load_primock57_isolated_bundle",
    "load_primock57_overlap_bundle",
    "main",
    "prepare_primock57_conversation_fixture",
    "verify_primock57_isolated_bundle",
]
