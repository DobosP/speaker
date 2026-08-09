"""Materialize one bounded, paired NOTSOFAR-1 far-field STT fixture.

The command is deliberately offline: callers provide the already-present
``MTG_32006`` directory.  All five inputs are bound to the lock before the
18-case schema-v2 corpus is published with the hardened no-clobber writer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Mapping, Sequence
import unicodedata

import numpy as np

from core.wer import normalize
from tools.prepare_ami_capture_replay import AmiPreparationError, _riff_pcm16_mono
from tools.prepare_ami_conversation_fixture import (
    AmiConversationPreparationError,
    _has_git_ancestor,
)
from tools.streaming_stt.bounded_io import (
    BoundedReadError,
    opened_directory_nofollow,
    read_regular_bounded,
)
from tools.streaming_stt.corpus import CorpusProvenance, LoadedCorpus, load_corpus
from tools.streaming_stt.corpus_writer import CorpusWriteCase, publish_private_corpus


DEFAULT_LOCK = (
    Path(__file__).resolve().parent
    / "streaming_stt"
    / "notsofar1-mtg32006-v1.lock.json"
)
LICENSE_ID = "CC-BY-4.0"
REVISION = "bb97bae9458f477a85dfd6e8aee92e35d5887dd3"
MEETING_ID = "MTG_32006"
MEETING_DURATION = Decimal("382.992625")
MEETING_PARTICIPANTS = 4
MEETING_HASHTAGS = "#DebateOverlaps"
SAMPLE_RATE_HZ = 16_000
EXPECTED_RECIPE_SHA256 = (
    "bce20c48c793a4c793f5361abe837a5119c4c66a6fbb48c7716b423eba1e7c4e"
)
SELECTION_SEED = "speaker-notsofar1-mtg32006-v1-2026-08-09"
SOURCE_RELATIVE_ROOT = (
    "benchmark-datasets/eval_set/240629.1_eval_small_with_GT/MTG/MTG_32006"
)
SOURCE_TOTAL_BYTES = 24_738_302
SOURCE_MAXIMUM_BYTES = 24 * 1024 * 1024
_MAX_JSON_BYTES = 1024 * 1024
_MAX_JSON_DEPTH = 12
_MAX_JSON_ITEMS = 100_000
_MAX_WAV_BYTES = 16 * 1024 * 1024
_MIN_SECONDS = Decimal("0.600000")
_MAX_SECONDS = Decimal("8.000000")
_CONTEXT_SECONDS = Decimal("0.200000")
_CONTEXT_SAMPLES = int(_CONTEXT_SECONDS * SAMPLE_RATE_HZ)
_SAFE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_NONLEXICAL_TAGS = frozenset(
    {
        "<BA/>",
        "<FILL/>",
        "<FILLlaugh/>",
        "<FL/>",
        "<ST/>",
        "<UNKNOWN/>",
        "<ba/>",
        "<fill/>",
        "<filllaugh/>",
        "<fl/>",
        "<st/>",
        "<unknown/>",
    }
)
_PNAME_OPEN = "<PName>"
_PNAME_CLOSE = "</PName>"
_UNKNOWN_TAGS = frozenset({"<UNKNOWN/>", "<unknown/>"})
_SAFE_ERROR = {
    "error": "notsofar1_far_field_fixture_prerequisites_unavailable",
    "ok": False,
}
_PREPARER_FILES = (
    "core/wer.py",
    "tools/prepare_notsofar1_far_field_fixture.py",
    "tools/prepare_ami_capture_replay.py",
    "tools/streaming_stt/corpus_writer.py",
    "tools/streaming_stt/corpus.py",
)

_SOURCE_PINS: Mapping[str, tuple[str, int, str]] = {
    "gt_transcription.json": (
        "transcription",
        224_525,
        "323cb14d9bb129d67645128bc5a65559c830b54982541bc577de86f199ce955b",
    ),
    "gt_meeting_metadata.json": (
        "meeting-metadata",
        469,
        "43cf2473716aa5dc958f16dfef82a5ea431a6ce5800cd695114de61b219187d0",
    ),
    "devices.json": (
        "devices-metadata",
        1_692,
        "6877215356b488462bf3bcc16fea76983a42d89dd2221c4dbff86aca3dea580f",
    ),
    "sc_meetup_0/ch0.wav": (
        "channel-a",
        12_255_808,
        "02f49d286fbb3ada1cb5ec49e076a6d498d140124d11b420cfcee94b7ecc5ec6",
    ),
    "sc_rockfall_2/ch0.wav": (
        "channel-b",
        12_255_808,
        "b37c3d34462ca56006d67bd0aba434fc8e7df95534d971ac66ecb46a7b908805",
    ),
}
_DEVICE_PATHS = ("sc_meetup_0/ch0.wav", "sc_rockfall_2/ch0.wav")
_MEETING_FIELDS = {
    "meeting_id",
    "MtgType",
    "MeetingDurationSec",
    "Room",
    "NumParticipants",
    "ParticipantAliases",
    "Topic",
    "Hashtags",
    "ParticipantAliasToCtDevice",
}
_DEVICE_FIELDS = {
    "device_name",
    "is_mc",
    "is_close_talk",
    "channels_num",
    "wav_file_names",
}
_SEGMENT_FIELDS = {
    "speaker_id",
    "ct_wav_file_name",
    "start_time",
    "end_time",
    "text",
    "word_timing",
}


class NotsofarPreparationError(RuntimeError):
    """A detail-free lock, source, selection, or publication failure."""


@dataclass(frozen=True, slots=True)
class TestSourceInjection:
    """Explicit non-production identities for small synthetic tests."""

    fixture_id: str
    file_pins: Mapping[str, tuple[int, str]] = field(repr=False)
    meeting_duration_seconds: Decimal


@dataclass(frozen=True, slots=True)
class PreparedNotsofarCorpus:
    corpus: LoadedCorpus
    receipt_sha256: str
    metadata_sha256: str
    selected_windows_sha256: str
    production_evidence: bool


@dataclass(frozen=True, slots=True)
class _Contract:
    pins: Mapping[str, tuple[str, int, str]] = field(repr=False)
    duration: Decimal
    production_evidence: bool


@dataclass(frozen=True, slots=True)
class _BoundSource:
    relative: str = field(repr=False)
    role: str
    raw: bytes = field(repr=False)
    size_bytes: int
    sha256: str
    snapshot: tuple[int, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class _Meeting:
    duration: Decimal
    aliases: tuple[str, ...] = field(repr=False)
    alias_to_device: Mapping[str, str] = field(repr=False)


@dataclass(frozen=True, slots=True)
class _Segment:
    speaker: str = field(repr=False)
    reference: str | None = field(repr=False)
    start: Decimal
    end: Decimal
    ordinal: int


@dataclass(frozen=True, slots=True)
class _Window:
    speaker: str = field(repr=False)
    reference: str = field(repr=False)
    start_sample: int
    end_sample: int
    ordinal: int
    rank: str


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
        raise NotsofarPreparationError() from None
    return raw + (b"\n" if newline else b"")


def _strict_json(raw: bytes) -> object:
    if not raw or len(raw) > _MAX_JSON_BYTES:
        raise NotsofarPreparationError()

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise NotsofarPreparationError()
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_float=Decimal,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                NotsofarPreparationError()
            ),
        )
    except (UnicodeError, ValueError, OverflowError, NotsofarPreparationError):
        raise NotsofarPreparationError() from None
    items = 0
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        if depth > _MAX_JSON_DEPTH:
            raise NotsofarPreparationError()
        if isinstance(current, dict):
            items += len(current)
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            items += len(current)
            stack.extend((item, depth + 1) for item in current)
        elif isinstance(current, str):
            if len(current) > 16_384:
                raise NotsofarPreparationError()
        elif current is not None and not isinstance(current, (bool, int, Decimal)):
            raise NotsofarPreparationError()
        if items > _MAX_JSON_ITEMS:
            raise NotsofarPreparationError()
    return value


def _decimal(value: object) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, Decimal)):
        raise NotsofarPreparationError()
    try:
        result = Decimal(value)
    except (InvalidOperation, ValueError, OverflowError):
        raise NotsofarPreparationError() from None
    if not result.is_finite():
        raise NotsofarPreparationError()
    return result


def _safe_text(value: object, *, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or value != unicodedata.normalize("NFC", value)
        or any(ord(character) < 32 for character in value)
    ):
        raise NotsofarPreparationError()
    return value


def _snapshot(metadata: os.stat_result) -> tuple[int, ...]:
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


def _source_root(value: Path | str) -> Path:
    candidate = Path(os.path.abspath(Path(value).expanduser()))
    try:
        resolved = candidate.resolve(strict=True)
        with opened_directory_nofollow(resolved) as (stable, descriptor):
            metadata = os.fstat(descriptor)
            if (
                stable != resolved
                or candidate != resolved
                or not stat.S_ISDIR(metadata.st_mode)
            ):
                raise NotsofarPreparationError()
    except NotsofarPreparationError:
        raise
    except (OSError, RuntimeError, ValueError, BoundedReadError):
        raise NotsofarPreparationError() from None
    return resolved


def _output_path(value: Path | str, *, source_root: Path) -> Path:
    supplied = Path(value).expanduser()
    if not supplied.is_absolute():
        raise NotsofarPreparationError()
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
            raise NotsofarPreparationError()
        with opened_directory_nofollow(parent) as (stable, _descriptor):
            if stable != parent:
                raise NotsofarPreparationError()
    except NotsofarPreparationError:
        raise
    except (
        OSError,
        RuntimeError,
        ValueError,
        BoundedReadError,
        AmiConversationPreparationError,
    ):
        raise NotsofarPreparationError() from None
    return candidate


def _contract(injection: TestSourceInjection | None) -> _Contract:
    if injection is None:
        return _Contract(_SOURCE_PINS, MEETING_DURATION, True)
    if (
        type(injection) is not TestSourceInjection
        or _SAFE_ID_RE.fullmatch(injection.fixture_id) is None
        or type(injection.meeting_duration_seconds) is not Decimal
        or not injection.meeting_duration_seconds.is_finite()
        or injection.meeting_duration_seconds <= 0
        or set(injection.file_pins) != set(_SOURCE_PINS)
    ):
        raise NotsofarPreparationError()
    pins: dict[str, tuple[str, int, str]] = {}
    for relative, (role, _size, _digest) in _SOURCE_PINS.items():
        try:
            size, digest = injection.file_pins[relative]
        except (KeyError, TypeError, ValueError):
            raise NotsofarPreparationError() from None
        maximum = _MAX_WAV_BYTES if relative.endswith(".wav") else _MAX_JSON_BYTES
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or not 0 < size <= maximum
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
        ):
            raise NotsofarPreparationError()
        pins[relative] = (role, size, digest)
    return _Contract(pins, injection.meeting_duration_seconds, False)


def _load_lock(path: Path | str) -> str:
    try:
        snapshot = read_regular_bounded(Path(path), maximum_bytes=64 * 1024)
    except BoundedReadError:
        raise NotsofarPreparationError() from None
    value = _strict_json(snapshot.data)
    if not isinstance(value, dict):
        raise NotsofarPreparationError()
    recipe = value.get("recipe_sha256")
    if not isinstance(recipe, str) or _SHA256_RE.fullmatch(recipe) is None:
        raise NotsofarPreparationError()
    body = dict(value)
    body.pop("recipe_sha256", None)
    if hashlib.sha256(_canonical_json(body)).hexdigest() != recipe:
        raise NotsofarPreparationError()
    source = value.get("source")
    selection = value.get("selection")
    files = source.get("files") if isinstance(source, dict) else None
    expected_files = [
        {"path": path, "role": role, "size_bytes": size, "sha256": digest}
        for path, (role, size, digest) in _SOURCE_PINS.items()
    ]
    if (
        recipe != EXPECTED_RECIPE_SHA256
        or value.get("schema_version") != 1
        or value.get("kind") != "notsofar1-far-field-fixture-lock-v1"
        or value.get("fixture_id") != "notsofar1-mtg32006-v1"
        or value.get("license_id") != LICENSE_ID
        or value.get("revision") != REVISION
        or value.get("repository") != "microsoft/NOTSOFAR"
        or not isinstance(source, dict)
        or source.get("meeting_relative_root") != SOURCE_RELATIVE_ROOT
        or source.get("subset") != "240629.1_eval_small_with_GT"
        or source.get("total_size_bytes") != SOURCE_TOTAL_BYTES
        or source.get("maximum_total_size_bytes") != SOURCE_MAXIMUM_BYTES
        or source.get("devices") != list(_DEVICE_PATHS)
        or files != expected_files
        or not isinstance(selection, dict)
        or selection.get("cases") != 18
        or selection.get("windows") != 9
        or selection.get("speakers") != 3
        or selection.get("windows_per_speaker") != 3
        or selection.get("seed") != SELECTION_SEED
        or selection.get("reference_transform") != "marker-stripped-row-text-v1"
        or selection.get("timing_agreement") != "core-wer-token-equality-v1"
        or selection.get("unknown_marker_policy") != "segment-ineligible-v1"
    ):
        raise NotsofarPreparationError()
    return recipe


def _read_source(root: Path, relative: str, pin: tuple[str, int, str]) -> _BoundSource:
    role, expected_size, expected_sha = pin
    path = root.joinpath(*PurePosixPath(relative).parts)
    maximum = _MAX_WAV_BYTES if relative.endswith(".wav") else _MAX_JSON_BYTES
    try:
        before = path.lstat()
        loaded = read_regular_bounded(
            path, maximum_bytes=maximum, expected_bytes=expected_size
        )
        after = path.lstat()
    except (OSError, BoundedReadError):
        raise NotsofarPreparationError() from None
    digest = hashlib.sha256(loaded.data).hexdigest()
    if (
        not stat.S_ISREG(before.st_mode)
        or _snapshot(before) != _snapshot(after)
        or len(loaded.data) != expected_size
        or digest != expected_sha
    ):
        raise NotsofarPreparationError()
    return _BoundSource(
        relative=relative,
        role=role,
        raw=loaded.data,
        size_bytes=expected_size,
        sha256=digest,
        snapshot=_snapshot(after),
    )


def _verify_sources(
    root: Path,
    sources: Mapping[str, _BoundSource],
    pins: Mapping[str, tuple[str, int, str]],
) -> None:
    for relative in _SOURCE_PINS:
        current = _read_source(root, relative, pins[relative])
        saved = sources[relative]
        if current.snapshot != saved.snapshot or current.sha256 != saved.sha256:
            raise NotsofarPreparationError()


def _preparer_closure() -> tuple[dict[str, object], ...]:
    root = Path(__file__).resolve().parents[1]
    rows: list[dict[str, object]] = []
    for relative in _PREPARER_FILES:
        try:
            loaded = read_regular_bounded(
                root.joinpath(*PurePosixPath(relative).parts),
                maximum_bytes=512 * 1024,
            )
        except BoundedReadError:
            raise NotsofarPreparationError() from None
        rows.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(loaded.data).hexdigest(),
                "size_bytes": len(loaded.data),
            }
        )
    return tuple(rows)


def _meeting(raw: bytes, expected_duration: Decimal) -> _Meeting:
    value = _strict_json(raw)
    if not isinstance(value, dict) or set(value) != _MEETING_FIELDS:
        raise NotsofarPreparationError()
    aliases = value.get("ParticipantAliases")
    mapping = value.get("ParticipantAliasToCtDevice")
    if (
        value.get("meeting_id") != MEETING_ID
        or value.get("MtgType") != "mtg"
        or _decimal(value.get("MeetingDurationSec")) != expected_duration
        or value.get("NumParticipants") != MEETING_PARTICIPANTS
        or value.get("Hashtags") != MEETING_HASHTAGS
        or not isinstance(aliases, list)
        or len(aliases) != MEETING_PARTICIPANTS
        or not isinstance(mapping, dict)
    ):
        raise NotsofarPreparationError()
    clean_aliases = tuple(_safe_text(item, maximum=128) for item in aliases)
    if len(set(clean_aliases)) != 4 or set(mapping) != set(clean_aliases):
        raise NotsofarPreparationError()
    clean_mapping = {
        alias: _safe_text(mapping[alias], maximum=128) for alias in clean_aliases
    }
    _safe_text(value.get("Room"), maximum=128)
    _safe_text(value.get("Topic"), maximum=1024)
    return _Meeting(expected_duration, clean_aliases, clean_mapping)


def _safe_relative(value: object) -> str:
    text = _safe_text(value, maximum=256)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or path.as_posix() != text
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in text
    ):
        raise NotsofarPreparationError()
    return text


def _strip_notsofar_markers(value: object) -> tuple[str | None, bool, bool]:
    """Apply the fixture's frozen scorer-like marker-removal step only."""

    text = _safe_text(value, maximum=4096)
    if "[" in text or "]" in text:
        raise NotsofarPreparationError()
    exact_nonlexical = text in _NONLEXICAL_TAGS
    contains_unknown = False
    output: list[str] = []
    p_name_open = False
    position = 0
    while position < len(text):
        marker_start = text.find("<", position)
        stray_end = text.find(">", position)
        if marker_start < 0:
            if stray_end >= 0:
                raise NotsofarPreparationError()
            output.append(text[position:])
            break
        if 0 <= stray_end < marker_start:
            raise NotsofarPreparationError()
        output.append(text[position:marker_start])
        marker_end = text.find(">", marker_start + 1)
        if marker_end < 0:
            raise NotsofarPreparationError()
        marker = text[marker_start : marker_end + 1]
        if marker in _NONLEXICAL_TAGS:
            contains_unknown = contains_unknown or marker in _UNKNOWN_TAGS
            output.append(" ")
        elif marker == _PNAME_OPEN and not p_name_open:
            p_name_open = True
        elif marker == _PNAME_CLOSE and p_name_open:
            p_name_open = False
        else:
            raise NotsofarPreparationError()
        position = marker_end + 1
    if p_name_open:
        raise NotsofarPreparationError()
    normalized = " ".join("".join(output).split())
    return (normalized or None), exact_nonlexical, contains_unknown


def _devices(raw: bytes) -> None:
    value = _strict_json(raw)
    if not isinstance(value, list) or not 1 <= len(value) <= 64:
        raise NotsofarPreparationError()
    far: list[str] = []
    names: set[str] = set()
    paths: set[str] = set()
    for row in value:
        if not isinstance(row, dict) or set(row) != _DEVICE_FIELDS:
            raise NotsofarPreparationError()
        name = _safe_text(row.get("device_name"), maximum=128)
        wav = _safe_relative(row.get("wav_file_names"))
        is_mc = row.get("is_mc")
        is_close = row.get("is_close_talk")
        channels = row.get("channels_num")
        if (
            name in names
            or wav in paths
            or type(is_mc) is not bool
            or type(is_close) is not bool
            or isinstance(channels, bool)
            or not isinstance(channels, int)
            or not 1 <= channels <= 32
        ):
            raise NotsofarPreparationError()
        names.add(name)
        paths.add(wav)
        if not is_mc and not is_close and channels == 1:
            if PurePosixPath(wav).parent.name != f"sc_{name}":
                raise NotsofarPreparationError()
            far.append(wav)
    if len(far) != 2 or set(far) != set(_DEVICE_PATHS):
        raise NotsofarPreparationError()


def _segments(raw: bytes, meeting: _Meeting) -> tuple[_Segment, ...]:
    value = _strict_json(raw)
    if not isinstance(value, list) or not 1 <= len(value) <= 4096:
        raise NotsofarPreparationError()
    result: list[_Segment] = []
    for ordinal, row in enumerate(value):
        if not isinstance(row, dict) or set(row) != _SEGMENT_FIELDS:
            raise NotsofarPreparationError()
        speaker = _safe_text(row.get("speaker_id"), maximum=128)
        reference, _row_is_marker, contains_unknown = _strip_notsofar_markers(
            row.get("text")
        )
        close_path = _safe_relative(row.get("ct_wav_file_name"))
        start = _decimal(row.get("start_time"))
        end = _decimal(row.get("end_time"))
        timings = row.get("word_timing")
        if (
            speaker not in meeting.aliases
            or PurePosixPath(close_path).stem != meeting.alias_to_device[speaker]
            or not Decimal(0) <= start < end <= meeting.duration
            or not isinstance(timings, list)
            or len(timings) > 2048
        ):
            raise NotsofarPreparationError()
        previous = start
        normalized_words: list[str] = []
        for timing in timings:
            if not isinstance(timing, list) or len(timing) != 3:
                raise NotsofarPreparationError()
            token, exact_nonlexical, token_unknown = _strip_notsofar_markers(timing[0])
            contains_unknown = contains_unknown or token_unknown
            word_start = _decimal(timing[1])
            word_end = _decimal(timing[2])
            # The pinned annotation contains eight explicit zero-duration word
            # timing entries.  They are valid metadata anchors, not candidate
            # segment durations; retain them while keeping bounds and order strict.
            if (
                not start <= word_start <= word_end <= end
                or word_start < previous
                or (word_start == word_end and not exact_nonlexical)
            ):
                raise NotsofarPreparationError()
            previous = word_start
            if token is not None:
                normalized_words.append(token)
        normalized_word_text = " ".join(" ".join(normalized_words).split()) or None
        reference_tokens = tuple(normalize(reference or ""))
        word_tokens = tuple(normalize(normalized_word_text or ""))
        if contains_unknown or not reference_tokens or reference_tokens != word_tokens:
            reference = None
        result.append(_Segment(speaker, reference, start, end, ordinal))
    if set(item.speaker for item in result) != set(meeting.aliases):
        raise NotsofarPreparationError()
    return tuple(result)


def _floor_sample(value: Decimal) -> int:
    return int((value * SAMPLE_RATE_HZ).to_integral_value(rounding=ROUND_FLOOR))


def _ceil_sample(value: Decimal) -> int:
    return int((value * SAMPLE_RATE_HZ).to_integral_value(rounding=ROUND_CEILING))


def _intersects(left: _Segment, right: _Segment) -> bool:
    return left.start < right.end and right.start < left.end


def _select(segments: tuple[_Segment, ...], duration: Decimal) -> tuple[_Window, ...]:
    total_samples = _floor_sample(duration)
    eligible: dict[str, list[_Window]] = {}
    for segment in segments:
        length = segment.end - segment.start
        if (
            segment.reference is None
            or length < _MIN_SECONDS
            or length > _MAX_SECONDS
            or any(
                other.ordinal != segment.ordinal and _intersects(segment, other)
                for other in segments
            )
        ):
            continue
        start_sample = _floor_sample(segment.start)
        end_sample = _ceil_sample(segment.end)
        left_boundary = 0
        right_boundary = total_samples
        for other in segments:
            if other.ordinal == segment.ordinal:
                continue
            other_start = _floor_sample(other.start)
            other_end = _ceil_sample(other.end)
            if other_end <= start_sample:
                left_boundary = max(left_boundary, other_end)
            if other_start >= end_sample:
                right_boundary = min(right_boundary, other_start)
        window_start = max(left_boundary, start_sample - _CONTEXT_SAMPLES)
        window_end = min(right_boundary, end_sample + _CONTEXT_SAMPLES)
        if (
            not 0
            <= window_start
            <= start_sample
            < end_sample
            <= window_end
            <= total_samples
        ):
            raise NotsofarPreparationError()
        identity = (
            f"{SELECTION_SEED}\0segment\0{segment.speaker}\0{segment.ordinal}\0"
            f"{segment.start}\0{segment.end}\0{segment.reference}"
        )
        rank = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        eligible.setdefault(segment.speaker, []).append(
            _Window(
                segment.speaker,
                segment.reference,
                window_start,
                window_end,
                segment.ordinal,
                rank,
            )
        )
    speakers = [speaker for speaker, rows in eligible.items() if len(rows) >= 3]
    speakers.sort(
        key=lambda speaker: hashlib.sha256(
            f"{SELECTION_SEED}\0speaker\0{speaker}".encode("utf-8")
        ).hexdigest()
    )
    if len(speakers) != 3:
        raise NotsofarPreparationError()
    selected: list[_Window] = []
    for speaker in speakers:
        selected.extend(sorted(eligible[speaker], key=lambda item: item.rank)[:3])
    if len(selected) != 9:
        raise NotsofarPreparationError()
    intervals = sorted((item.start_sample, item.end_sample) for item in selected)
    if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
        raise NotsofarPreparationError()
    return tuple(selected)


def _decode(raw: bytes, *, expected_samples: int) -> np.ndarray:
    try:
        pcm = _riff_pcm16_mono(raw)
    except (AmiPreparationError, MemoryError, ValueError):
        raise NotsofarPreparationError() from None
    if pcm.size != expected_samples:
        raise NotsofarPreparationError()
    return pcm


def _pcm_f32le(pcm: np.ndarray, start: int, end: int) -> bytes:
    if not 0 <= start < end <= pcm.size:
        raise NotsofarPreparationError()
    result = pcm[start:end].astype("<f4")
    result *= np.float32(1.0 / 32768.0)
    raw = result.tobytes(order="C")
    if not raw or len(raw) != (end - start) * 4:
        raise NotsofarPreparationError()
    return raw


def prepare_notsofar1_far_field_fixture(
    *,
    source_dir: Path | str,
    output_dir: Path | str,
    accepted_license: str,
    lock_path: Path | str = DEFAULT_LOCK,
    test_source_injection: TestSourceInjection | None = None,
) -> PreparedNotsofarCorpus:
    """Verify local pinned inputs and publish the deterministic paired corpus."""

    if accepted_license != LICENSE_ID:
        raise NotsofarPreparationError()
    recipe_sha = _load_lock(lock_path)
    contract = _contract(test_source_injection)
    root = _source_root(source_dir)
    destination = _output_path(output_dir, source_root=root)
    preparer_files = _preparer_closure()
    sources = {
        relative: _read_source(root, relative, contract.pins[relative])
        for relative in _SOURCE_PINS
    }
    meeting = _meeting(sources["gt_meeting_metadata.json"].raw, contract.duration)
    _devices(sources["devices.json"].raw)
    segments = _segments(sources["gt_transcription.json"].raw, meeting)
    windows = _select(segments, meeting.duration)
    expected_samples_decimal = meeting.duration * SAMPLE_RATE_HZ
    if expected_samples_decimal != expected_samples_decimal.to_integral_value():
        raise NotsofarPreparationError()
    expected_samples = int(expected_samples_decimal)
    channels = {
        "channel-a": _decode(
            sources[_DEVICE_PATHS[0]].raw, expected_samples=expected_samples
        ),
        "channel-b": _decode(
            sources[_DEVICE_PATHS[1]].raw, expected_samples=expected_samples
        ),
    }
    speaker_order = tuple(dict.fromkeys(item.speaker for item in windows))
    speaker_slots = {speaker: index for index, speaker in enumerate(speaker_order)}
    write_cases: list[CorpusWriteCase] = []
    receipt_cases: list[dict[str, object]] = []
    for window_index, window in enumerate(windows):
        slot = speaker_slots[window.speaker]
        reference_sha = hashlib.sha256(window.reference.encode("utf-8")).hexdigest()
        pair_rows: list[dict[str, object]] = []
        for channel_name in ("channel-a", "channel-b"):
            raw_pcm = _pcm_f32le(
                channels[channel_name], window.start_sample, window.end_sample
            )
            case_id = f"notsofar-w{window_index:02d}-{channel_name}"
            common_tags = (
                "public-voice",
                "notsofar1",
                "far-field",
                "hard-wer",
                "nonoverlap",
                f"speaker-slot-{slot}",
                f"pair-{window_index:02d}",
            )
            tags = (*common_tags, channel_name)
            write_cases.append(
                CorpusWriteCase(
                    case_id=case_id,
                    audio_bytes=raw_pcm,
                    reference=window.reference,
                    tags=tags,
                )
            )
            pair_rows.append(
                {
                    "case_id": case_id,
                    "channel": channel_name,
                    "end_sample": window.end_sample,
                    "pcm_sha256": hashlib.sha256(raw_pcm).hexdigest(),
                    "reference_sha256": reference_sha,
                    "samples": window.end_sample - window.start_sample,
                    "speaker_slot": slot,
                    "start_sample": window.start_sample,
                    "tags": list(tags),
                }
            )
        left, right = pair_rows
        if (
            left["start_sample"] != right["start_sample"]
            or left["end_sample"] != right["end_sample"]
            or left["reference_sha256"] != right["reference_sha256"]
            or left["samples"] != right["samples"]
            or left["speaker_slot"] != right["speaker_slot"]
            or left["tags"][:-1] != right["tags"][:-1]
        ):
            raise NotsofarPreparationError()
        receipt_cases.extend(pair_rows)

    metadata_sha = hashlib.sha256(
        _canonical_json(
            [
                {
                    "role": sources[name].role,
                    "sha256": sources[name].sha256,
                    "size_bytes": sources[name].size_bytes,
                }
                for name in (
                    "gt_transcription.json",
                    "gt_meeting_metadata.json",
                    "devices.json",
                )
            ]
        )
    ).hexdigest()
    selected_sha = hashlib.sha256(_canonical_json(receipt_cases)).hexdigest()
    receipt = {
        "accepted_license": LICENSE_ID,
        "cases": receipt_cases,
        "evidence_scope": {
            "hard_wer_nonoverlap": True,
            "overlap_cases": 0,
            "overlap_evidence": False,
            "speaker_authority": False,
        },
        "fixture_id": "notsofar1-mtg32006-v1",
        "kind": "notsofar1-far-field-preparation-receipt-v1",
        "lock_recipe_sha256": recipe_sha,
        "metadata_sha256": metadata_sha,
        "privacy": {
            "local_paths_in_receipt": False,
            "raw_device_ids_in_receipt": False,
            "raw_speaker_ids_in_receipt": False,
            "transcripts_in_receipt": False,
        },
        "preparer_files": list(preparer_files),
        "production_evidence": contract.production_evidence,
        "reference_policy": {
            "timing_agreement": "core-wer-token-equality-v1",
            "transform": "marker-stripped-row-text-v1",
            "unknown_marker": "segment-ineligible-v1",
        },
        "revision": REVISION,
        "schema_version": 1,
        "selected_windows_sha256": selected_sha,
        "source_files": [
            {
                "role": sources[name].role,
                "sha256": sources[name].sha256,
                "size_bytes": sources[name].size_bytes,
            }
            for name in _SOURCE_PINS
        ],
        "totals": {"cases": 18, "channels": 2, "speakers": 3, "windows": 9},
    }
    receipt_raw = _canonical_json(receipt, newline=True)
    receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
    provenance = CorpusProvenance(
        kind="public-voice-v1",
        suite="notsofar1-mtg32006-v1",
        manifest_sha256=recipe_sha,
        metadata_sha256=metadata_sha,
        source_set_sha256=receipt_sha,
    )
    _verify_sources(root, sources, contract.pins)
    if _preparer_closure() != preparer_files:
        raise NotsofarPreparationError()
    try:
        corpus = publish_private_corpus(
            cases=write_cases,
            provenance=provenance,
            output_dir=destination,
            purpose="NOTSOFAR-1 MTG_32006 paired far-field isolated-segment hard WER",
            sidecars={"preparation-receipt.json": receipt_raw},
        )
    except Exception:
        raise NotsofarPreparationError() from None
    _verify_sources(root, sources, contract.pins)
    if _preparer_closure() != preparer_files:
        raise NotsofarPreparationError()
    if load_corpus(corpus.path).digest != corpus.digest:
        raise NotsofarPreparationError()
    return PreparedNotsofarCorpus(
        corpus=corpus,
        receipt_sha256=receipt_sha,
        metadata_sha256=metadata_sha,
        selected_windows_sha256=selected_sha,
        production_evidence=contract.production_evidence,
    )


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise NotsofarPreparationError()


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description="Prepare the pinned offline NOTSOFAR-1 far-field fixture."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--accept-license", required=True)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        prepared = prepare_notsofar1_far_field_fixture(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            accepted_license=args.accept_license,
            lock_path=args.lock,
        )
        print(
            json.dumps(
                {
                    "cases": len(prepared.corpus.cases),
                    "corpus_manifest_sha256": prepared.corpus.digest,
                    "ok": True,
                    "production_evidence": prepared.production_evidence,
                    "receipt_sha256": prepared.receipt_sha256,
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
    "LICENSE_ID",
    "NotsofarPreparationError",
    "PreparedNotsofarCorpus",
    "REVISION",
    "TestSourceInjection",
    "main",
    "prepare_notsofar1_far_field_fixture",
]
