"""Strict, hash-bound corpus loading for production-path capture replay.

The prepared corpus is deliberately smaller and stricter than the public-source
manifest from which it may be derived.  Every waveform is local mono 16 kHz
``f32le`` PCM, every coordinate is expressed in source samples, and the entire
input is retained as an immutable in-memory snapshot before replay starts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType
from typing import Mapping

import numpy as np

from core.wer import normalize
from tools.streaming_stt.bounded_io import BoundedReadError, read_regular_bounded


SAMPLE_RATE_HZ = 16_000
FRAME_SAMPLES = 1_600
MAX_MANIFEST_BYTES = 256 * 1024
MAX_TRACK_BYTES = 8 * 1024 * 1024
MAX_CORPUS_AUDIO_BYTES = 64 * 1024 * 1024
MAX_CASES = 128
MAX_SPEECH_INTERVALS = 4_096
MAX_SPEAKER_INTERVALS = 8_192
MAX_WORD_INTERVALS = 16_384

_SAFE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,95}\Z")
_SPEAKER_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ROOT_FIELDS = {
    "schema_version",
    "purpose",
    "sample_rate_hz",
    "frame_samples",
    "cases",
}
_CASE_FIELDS = {
    "id",
    "tracks",
    "speech_intervals",
    "speaker_intervals",
    "word_intervals",
    "assertion",
    "expected_text",
    "commands",
    "tags",
    "aec_delay_samples",
}
_TRACK_FIELDS = {"file", "sha256", "encoding", "samples"}
_TRACK_ROLES = frozenset(
    {
        "mic",
        "far_reference_zero",
        "far_reference_delayed",
        "coherence_reference",
    }
)
_SAMPLE_INTERVAL_FIELDS = {"start_sample", "end_sample"}
_SPEAKER_INTERVAL_FIELDS = {
    "speaker_id",
    "role",
    "start_sample",
    "end_sample",
}
_WORD_INTERVAL_FIELDS = {
    "speaker_id",
    "text",
    "start_sample",
    "end_sample",
}


class ReplayCorpusError(RuntimeError):
    """A detail-free prepared-corpus validation failure."""


class ReplayAssertion(str, Enum):
    """Closed assertions understood by production capture replay."""

    TRANSCRIPT = "transcript"
    SILENCE = "silence"
    NEAR_END_ONLY = "near_end_only"
    FAR_END_ONLY = "far_end_only"
    DOUBLE_TALK = "double_talk"
    EVENT_ONLY = "event_only"


class SpeakerRole(str, Enum):
    """Ground-truth relation of one microphone speaker to the requested turn."""

    TARGET = "target"
    BYSTANDER = "bystander"
    OWNER = "owner"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ReplayTrack:
    """One verified, aligned PCM track."""

    role: str
    path: Path = field(repr=False)
    sha256: str
    samples: int
    pcm_bytes: bytes = field(repr=False)


@dataclass(frozen=True, order=True)
class SampleInterval:
    """Half-open source-sample interval."""

    start_sample: int
    end_sample: int


@dataclass(frozen=True)
class SpeakerInterval:
    """One half-open active-speaker interval on the microphone timeline."""

    speaker_id: str
    role: SpeakerRole
    start_sample: int
    end_sample: int


@dataclass(frozen=True)
class WordInterval:
    """One ground-truth word interval on the microphone timeline."""

    speaker_id: str
    text: str = field(repr=False)
    start_sample: int
    end_sample: int


@dataclass(frozen=True)
class ReplayCase:
    """One fully verified production capture replay case."""

    case_id: str
    sample_rate_hz: int
    frame_samples: int
    tracks: Mapping[str, ReplayTrack] = field(repr=False)
    speech_intervals: tuple[SampleInterval, ...]
    speaker_intervals: tuple[SpeakerInterval, ...]
    word_intervals: tuple[WordInterval, ...] = field(repr=False)
    assertion: ReplayAssertion
    expected_text: str = field(repr=False)
    commands: tuple[str, ...] = field(repr=False)
    tags: tuple[str, ...]
    aec_delay_samples: int

    @property
    def mic(self) -> ReplayTrack:
        return self.tracks["mic"]

    @property
    def samples(self) -> int:
        return self.mic.samples

    @property
    def duration_seconds(self) -> float:
        return float(self.samples) / float(self.sample_rate_hz)

    def track(self, role: str) -> ReplayTrack | None:
        return self.tracks.get(role)


@dataclass(frozen=True)
class LoadedReplayCorpus:
    """A manifest and every binary it binds, retained as one stable snapshot."""

    path: Path = field(repr=False)
    digest: str
    schema_version: int
    purpose: str
    sample_rate_hz: int
    frame_samples: int
    cases: tuple[ReplayCase, ...] = field(repr=False)
    audio_bytes: int


def _bad() -> object:
    raise ReplayCorpusError()


def _strict_json(raw: bytes) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ReplayCorpusError()
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda _value: _bad(),
        )
    except (UnicodeError, ValueError, OverflowError, ReplayCorpusError):
        raise ReplayCorpusError() from None


def _lexical_absolute(path: Path) -> Path:
    try:
        candidate = Path(os.path.abspath(os.fspath(path)))
        if candidate.resolve(strict=True) != candidate:
            raise ReplayCorpusError()
        return candidate
    except (OSError, RuntimeError, ValueError, ReplayCorpusError):
        raise ReplayCorpusError() from None


def _read_nofollow(
    path: Path,
    *,
    maximum_bytes: int,
    expected_bytes: int | None = None,
) -> tuple[Path, bytes]:
    candidate = _lexical_absolute(path)
    try:
        snapshot = read_regular_bounded(
            candidate,
            maximum_bytes=maximum_bytes,
            expected_bytes=expected_bytes,
        )
    except BoundedReadError:
        raise ReplayCorpusError() from None
    if snapshot.path != candidate:
        raise ReplayCorpusError()
    return snapshot.path, snapshot.data


def _safe_id(value: object) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise ReplayCorpusError()
    return value


def _speaker_id(value: object) -> str:
    if not isinstance(value, str) or _SPEAKER_ID_RE.fullmatch(value) is None:
        raise ReplayCorpusError()
    return value


def _bounded_int(
    value: object,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        raise ReplayCorpusError()
    return value


def _string_list(
    value: object,
    *,
    maximum_items: int,
    maximum_chars: int,
    safe_ids: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > maximum_items:
        raise ReplayCorpusError()
    result: list[str] = []
    for item in value:
        if (
            not isinstance(item, str)
            or not item
            or item.strip() != item
            or len(item) > maximum_chars
            or any(ord(character) < 32 for character in item)
            or (safe_ids and _SAFE_ID_RE.fullmatch(item) is None)
        ):
            raise ReplayCorpusError()
        result.append(item)
    if len(set(result)) != len(result):
        raise ReplayCorpusError()
    return tuple(result)


def _load_track(
    root: Path,
    role: str,
    value: object,
    *,
    expected_samples: int | None,
    remaining_bytes: int,
) -> ReplayTrack:
    if not isinstance(value, dict) or set(value) != _TRACK_FIELDS:
        raise ReplayCorpusError()
    filename = value.get("file")
    if not isinstance(filename, str) or not filename or "\x00" in filename:
        raise ReplayCorpusError()
    pure = PurePosixPath(filename)
    if (
        pure.is_absolute()
        or len(pure.parts) != 1
        or pure.name != filename
        or pure.name in {".", ".."}
    ):
        raise ReplayCorpusError()
    expected_hash = value.get("sha256")
    if (
        not isinstance(expected_hash, str)
        or _SHA256_RE.fullmatch(expected_hash) is None
        or value.get("encoding") != "f32le"
    ):
        raise ReplayCorpusError()
    samples = _bounded_int(
        value.get("samples"),
        minimum=FRAME_SAMPLES,
        maximum=MAX_TRACK_BYTES // 4,
    )
    if samples % FRAME_SAMPLES != 0 or (
        expected_samples is not None and samples != expected_samples
    ):
        raise ReplayCorpusError()
    expected_bytes = samples * 4
    if expected_bytes > remaining_bytes:
        raise ReplayCorpusError()
    path, raw = _read_nofollow(
        root / filename,
        maximum_bytes=MAX_TRACK_BYTES,
        expected_bytes=expected_bytes,
    )
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ReplayCorpusError()
    pcm = np.frombuffer(raw, dtype="<f4")
    if (
        pcm.size != samples
        or not bool(np.all(np.isfinite(pcm)))
        or bool(np.any(np.abs(pcm) > 1.0))
    ):
        raise ReplayCorpusError()
    return ReplayTrack(
        role=role,
        path=path,
        sha256=expected_hash,
        samples=samples,
        pcm_bytes=raw,
    )


def _sample_interval(value: object, *, samples: int) -> SampleInterval:
    if not isinstance(value, dict) or set(value) != _SAMPLE_INTERVAL_FIELDS:
        raise ReplayCorpusError()
    start = _bounded_int(value.get("start_sample"), minimum=0, maximum=samples)
    end = _bounded_int(value.get("end_sample"), minimum=0, maximum=samples)
    if start >= end:
        raise ReplayCorpusError()
    return SampleInterval(start, end)


def _load_speech_intervals(
    value: object,
    *,
    samples: int,
) -> tuple[SampleInterval, ...]:
    if not isinstance(value, list) or len(value) > MAX_SPEECH_INTERVALS:
        raise ReplayCorpusError()
    intervals = tuple(_sample_interval(item, samples=samples) for item in value)
    if any(
        current.start_sample < previous.end_sample
        for previous, current in zip(intervals, intervals[1:])
    ):
        raise ReplayCorpusError()
    return intervals


def _load_speaker_intervals(
    value: object,
    *,
    samples: int,
) -> tuple[SpeakerInterval, ...]:
    if not isinstance(value, list) or len(value) > MAX_SPEAKER_INTERVALS:
        raise ReplayCorpusError()
    result: list[SpeakerInterval] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != _SPEAKER_INTERVAL_FIELDS:
            raise ReplayCorpusError()
        try:
            role = SpeakerRole(item.get("role"))
        except (TypeError, ValueError):
            raise ReplayCorpusError() from None
        start = _bounded_int(
            item.get("start_sample"),
            minimum=0,
            maximum=samples,
        )
        end = _bounded_int(item.get("end_sample"), minimum=0, maximum=samples)
        if start >= end:
            raise ReplayCorpusError()
        result.append(
            SpeakerInterval(
                speaker_id=_speaker_id(item.get("speaker_id")),
                role=role,
                start_sample=start,
                end_sample=end,
            )
        )
    ordered = sorted(
        result,
        key=lambda item: (
            item.start_sample,
            item.end_sample,
            item.speaker_id,
            item.role.value,
        ),
    )
    if ordered != result or len(set(result)) != len(result):
        raise ReplayCorpusError()
    by_speaker: dict[str, list[SpeakerInterval]] = {}
    for interval in result:
        by_speaker.setdefault(interval.speaker_id, []).append(interval)
    if any(
        current.start_sample < previous.end_sample
        for intervals in by_speaker.values()
        for previous, current in zip(intervals, intervals[1:])
    ):
        raise ReplayCorpusError()
    speaker_roles: dict[str, SpeakerRole] = {}
    for interval in result:
        existing = speaker_roles.setdefault(interval.speaker_id, interval.role)
        if existing is not interval.role:
            raise ReplayCorpusError()
    return tuple(result)


def _load_word_intervals(
    value: object,
    *,
    samples: int,
) -> tuple[WordInterval, ...]:
    if not isinstance(value, list) or len(value) > MAX_WORD_INTERVALS:
        raise ReplayCorpusError()
    result: list[WordInterval] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != _WORD_INTERVAL_FIELDS:
            raise ReplayCorpusError()
        text = item.get("text")
        if (
            not isinstance(text, str)
            or not text
            or text.strip() != text
            or len(text) > 128
            or any(ord(character) < 32 for character in text)
        ):
            raise ReplayCorpusError()
        start = _bounded_int(
            item.get("start_sample"),
            minimum=0,
            maximum=samples,
        )
        end = _bounded_int(item.get("end_sample"), minimum=0, maximum=samples)
        if start >= end:
            raise ReplayCorpusError()
        result.append(
            WordInterval(
                speaker_id=_speaker_id(item.get("speaker_id")),
                text=text,
                start_sample=start,
                end_sample=end,
            )
        )
    ordered = sorted(
        result,
        key=lambda item: (
            item.start_sample,
            item.end_sample,
            item.speaker_id,
            item.text,
        ),
    )
    if ordered != result or len(set(result)) != len(result):
        raise ReplayCorpusError()
    return tuple(result)


def _is_contained(
    start: int,
    end: int,
    intervals: tuple[SampleInterval, ...],
) -> bool:
    return any(
        interval.start_sample <= start and end <= interval.end_sample
        for interval in intervals
    )


def _validate_annotations(
    *,
    speech: tuple[SampleInterval, ...],
    speakers: tuple[SpeakerInterval, ...],
    words: tuple[WordInterval, ...],
) -> None:
    if any(
        not _is_contained(
            interval.start_sample,
            interval.end_sample,
            speech,
        )
        for interval in speakers
    ):
        raise ReplayCorpusError()
    if any(
        not any(
            speaker.speaker_id == word.speaker_id
            and speaker.start_sample <= word.start_sample
            and word.end_sample <= speaker.end_sample
            for speaker in speakers
        )
        for word in words
    ):
        raise ReplayCorpusError()
    if any(
        not any(
            speaker.start_sample < interval.end_sample
            and interval.start_sample < speaker.end_sample
            for speaker in speakers
        )
        for interval in speech
    ):
        raise ReplayCorpusError()


def _load_case(
    root: Path,
    value: object,
    *,
    remaining_bytes: int,
) -> ReplayCase:
    if not isinstance(value, dict) or set(value) != _CASE_FIELDS:
        raise ReplayCorpusError()
    case_id = _safe_id(value.get("id"))
    raw_tracks = value.get("tracks")
    if (
        not isinstance(raw_tracks, dict)
        or "mic" not in raw_tracks
        or not set(raw_tracks).issubset(_TRACK_ROLES)
    ):
        raise ReplayCorpusError()
    loaded_tracks: dict[str, ReplayTrack] = {}
    total = 0
    expected_samples: int | None = None
    for role in (
        "mic",
        "far_reference_zero",
        "far_reference_delayed",
        "coherence_reference",
    ):
        if role not in raw_tracks:
            continue
        track = _load_track(
            root,
            role,
            raw_tracks[role],
            expected_samples=expected_samples,
            remaining_bytes=remaining_bytes - total,
        )
        if expected_samples is None:
            expected_samples = track.samples
        loaded_tracks[role] = track
        total += len(track.pcm_bytes)
    assert expected_samples is not None
    filenames = [track.path.name.casefold() for track in loaded_tracks.values()]
    if len(set(filenames)) != len(filenames):
        raise ReplayCorpusError()

    speech = _load_speech_intervals(
        value.get("speech_intervals"),
        samples=expected_samples,
    )
    speakers = _load_speaker_intervals(
        value.get("speaker_intervals"),
        samples=expected_samples,
    )
    words = _load_word_intervals(
        value.get("word_intervals"),
        samples=expected_samples,
    )
    _validate_annotations(speech=speech, speakers=speakers, words=words)

    try:
        assertion = ReplayAssertion(value.get("assertion"))
    except (TypeError, ValueError):
        raise ReplayCorpusError() from None
    expected_text = value.get("expected_text")
    if (
        not isinstance(expected_text, str)
        or len(expected_text) > 4_096
        or expected_text.strip() != expected_text
        or any(ord(character) < 32 for character in expected_text)
    ):
        raise ReplayCorpusError()
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
    aec_delay_samples = _bounded_int(
        value.get("aec_delay_samples"),
        minimum=0,
        maximum=min(expected_samples, 2 * SAMPLE_RATE_HZ),
    )
    has_reference = any(role != "mic" for role in loaded_tracks)
    text_assertions = {
        ReplayAssertion.TRANSCRIPT,
        ReplayAssertion.NEAR_END_ONLY,
        ReplayAssertion.DOUBLE_TALK,
    }
    if assertion in text_assertions:
        if (
            not expected_text
            or not normalize(expected_text)
            or not speech
            or not speakers
            or not words
            or normalize(" ".join(word.text for word in words))
            != normalize(expected_text)
            or any(not normalize(command) for command in commands)
        ):
            raise ReplayCorpusError()
    elif expected_text or commands:
        raise ReplayCorpusError()
    if assertion in {ReplayAssertion.SILENCE, ReplayAssertion.FAR_END_ONLY} and (
        speech or speakers or words
    ):
        raise ReplayCorpusError()
    if assertion is ReplayAssertion.EVENT_ONLY and (not speech or not speakers):
        raise ReplayCorpusError()
    if assertion is ReplayAssertion.SILENCE and has_reference:
        raise ReplayCorpusError()
    if assertion is ReplayAssertion.NEAR_END_ONLY and has_reference:
        raise ReplayCorpusError()
    if assertion in {
        ReplayAssertion.FAR_END_ONLY,
        ReplayAssertion.DOUBLE_TALK,
    } and "far_reference_zero" not in loaded_tracks:
        raise ReplayCorpusError()
    if not has_reference and aec_delay_samples != 0:
        raise ReplayCorpusError()

    return ReplayCase(
        case_id=case_id,
        sample_rate_hz=SAMPLE_RATE_HZ,
        frame_samples=FRAME_SAMPLES,
        tracks=MappingProxyType(dict(loaded_tracks)),
        speech_intervals=speech,
        speaker_intervals=speakers,
        word_intervals=words,
        assertion=assertion,
        expected_text=expected_text,
        commands=commands,
        tags=tags,
        aec_delay_samples=aec_delay_samples,
    )


def load_corpus(path: Path | str) -> LoadedReplayCorpus:
    """Load and verify a complete prepared capture-replay corpus."""

    manifest_path, raw = _read_nofollow(
        Path(path).expanduser(),
        maximum_bytes=MAX_MANIFEST_BYTES,
    )
    value = _strict_json(raw)
    if (
        not isinstance(value, dict)
        or set(value) != _ROOT_FIELDS
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or value.get("sample_rate_hz") != SAMPLE_RATE_HZ
        or value.get("frame_samples") != FRAME_SAMPLES
        or not isinstance(value.get("purpose"), str)
        or not str(value.get("purpose")).strip()
        or str(value.get("purpose")).strip() != value.get("purpose")
        or len(str(value.get("purpose"))) > 512
    ):
        raise ReplayCorpusError()
    raw_cases = value.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases or len(raw_cases) > MAX_CASES:
        raise ReplayCorpusError()

    root = manifest_path.parent
    cases: list[ReplayCase] = []
    audio_bytes = 0
    for raw_case in raw_cases:
        case = _load_case(
            root,
            raw_case,
            remaining_bytes=MAX_CORPUS_AUDIO_BYTES - audio_bytes,
        )
        cases.append(case)
        audio_bytes += sum(len(track.pcm_bytes) for track in case.tracks.values())
    case_ids = [case.case_id for case in cases]
    all_filenames = [
        track.path.name.casefold()
        for case in cases
        for track in case.tracks.values()
    ]
    if (
        len(set(case_ids)) != len(case_ids)
        or len(set(all_filenames)) != len(all_filenames)
    ):
        raise ReplayCorpusError()
    return LoadedReplayCorpus(
        path=manifest_path,
        digest=hashlib.sha256(raw).hexdigest(),
        schema_version=1,
        purpose=str(value["purpose"]),
        sample_rate_hz=SAMPLE_RATE_HZ,
        frame_samples=FRAME_SAMPLES,
        cases=tuple(cases),
        audio_bytes=audio_bytes,
    )


def verify_corpus_snapshot(corpus: LoadedReplayCorpus) -> None:
    """Reject a manifest or bound PCM file changed after initial loading."""

    try:
        manifest_path, raw = _read_nofollow(
            corpus.path,
            maximum_bytes=MAX_MANIFEST_BYTES,
        )
        if (
            manifest_path != corpus.path
            or hashlib.sha256(raw).hexdigest() != corpus.digest
        ):
            raise ReplayCorpusError()
        seen: set[Path] = set()
        for case in corpus.cases:
            for track in case.tracks.values():
                if track.path in seen:
                    raise ReplayCorpusError()
                seen.add(track.path)
                path, pcm = _read_nofollow(
                    track.path,
                    maximum_bytes=MAX_TRACK_BYTES,
                    expected_bytes=len(track.pcm_bytes),
                )
                if (
                    path != track.path
                    or hashlib.sha256(pcm).hexdigest() != track.sha256
                    or pcm != track.pcm_bytes
                ):
                    raise ReplayCorpusError()
    except (OSError, ReplayCorpusError):
        raise ReplayCorpusError() from None
