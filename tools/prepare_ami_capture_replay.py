"""Prepare a small, hash-bound AMI ES2004a capture-replay corpus.

This module deliberately has no downloader.  It accepts one already present
AMI manual-annotation ZIP and one already present ``ES2004a.Array1-01.wav``,
verifies caller-supplied byte sizes and SHA-256 identities, and emits four
private production-replay cases.  Source paths and transcript text are never
included in CLI receipts or failures.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import string
import struct
from typing import Iterable, Mapping, Sequence
import unicodedata
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from tools.capture_replay import (
    FRAME_SAMPLES,
    SAMPLE_RATE_HZ,
    LoadedReplayCorpus,
    ReplayCorpusError,
    load_corpus,
)
from tools.streaming_stt.bounded_io import (
    BoundedReadError,
    opened_directory_nofollow,
    read_regular_bounded,
)


_MEETING_ID = "ES2004a"
_SPEAKERS = ("A", "B", "C", "D")
_ANNOTATION_ARCHIVE_MAX_BYTES = 32 * 1024 * 1024
_AUDIO_MAX_BYTES = 64 * 1024 * 1024
_MAX_ZIP_MEMBERS = 8_192
_MAX_ZIP_MEMBER_BYTES = 16 * 1024 * 1024
_MAX_ZIP_EXPANDED_BYTES = 256 * 1024 * 1024
_MAX_WORDS_PER_SPEAKER = 32_000
_MAX_ANNOTATION_ELEMENTS_PER_SPEAKER = 64_000
_CALIBRATION_SAMPLES = 2 * SAMPLE_RATE_HZ
_MIN_REAL_SILENCE_SAMPLES = _CALIBRATION_SAMPLES
_MIN_CASE_SAMPLES = 8 * SAMPLE_RATE_HZ
_MAX_CASE_SAMPLES = 15 * SAMPLE_RATE_HZ
_EDGE_PAD_SAMPLES = 2 * SAMPLE_RATE_HZ
_MAX_SOURCE_SAMPLES = _MAX_CASE_SAMPLES - 2 * _EDGE_PAD_SAMPLES
_MAX_TRANSITION_GAP_SAMPLES = 2 * SAMPLE_RATE_HZ
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_WORD_MEMBER_RE = re.compile(
    rf"(?:^|/)words/{re.escape(_MEETING_ID)}\.([ABCD])\.words\.xml\Z"
)
_SAFE_ERROR: Mapping[str, object] = {
    "ok": False,
    "error": "ami_capture_replay_prerequisites_unavailable",
}


class AmiPreparationError(RuntimeError):
    """A detail-free source, selection, or destination failure."""


@dataclass(frozen=True)
class _SourceSnapshot:
    data: bytes
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class _Word:
    speaker: str
    text: str
    start_sample: int
    end_sample: int
    ordinal: int


@dataclass(frozen=True)
class _Activity:
    speaker: str
    kind: str
    start_sample: int
    end_sample: int
    ordinal: int


@dataclass(frozen=True)
class _SpeakerAnnotations:
    words: tuple[_Word, ...]
    activities: tuple[_Activity, ...]


@dataclass(frozen=True)
class _Component:
    start_sample: int
    end_sample: int
    words: tuple[_Word, ...]


@dataclass(frozen=True)
class _Window:
    start_sample: int
    end_sample: int
    words: tuple[_Word, ...]


@dataclass(frozen=True)
class _PreparedCase:
    case_id: str
    filename: str
    pcm: bytes
    manifest: Mapping[str, object]


def _fail() -> object:
    raise AmiPreparationError()


def _expected_identity(sha256: object, size_bytes: object, maximum: int) -> tuple[str, int]:
    if (
        not isinstance(sha256, str)
        or _SHA256_RE.fullmatch(sha256) is None
        or isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes <= 0
        or size_bytes > maximum
    ):
        raise AmiPreparationError()
    return sha256, size_bytes


def _verified_source(
    path: Path | str,
    *,
    expected_sha256: str,
    expected_bytes: int,
    maximum_bytes: int,
) -> _SourceSnapshot:
    digest, size = _expected_identity(expected_sha256, expected_bytes, maximum_bytes)
    try:
        snapshot = read_regular_bounded(
            Path(path).expanduser(),
            maximum_bytes=maximum_bytes,
            expected_bytes=size,
        )
    except BoundedReadError:
        raise AmiPreparationError() from None
    if len(snapshot.data) != size or hashlib.sha256(snapshot.data).hexdigest() != digest:
        raise AmiPreparationError()
    return _SourceSnapshot(data=snapshot.data, sha256=digest, size_bytes=size)


def _safe_zip_name(name: str) -> PurePosixPath:
    directory = name.endswith("/")
    lexical = name[:-1] if directory else name
    parts = lexical.split("/")
    if (
        not name
        or not lexical
        or "\x00" in name
        or "\\" in name
        or any(ord(character) < 32 for character in name)
        or unicodedata.normalize("NFC", name) != name
        or any(part in {"", ".", ".."} for part in parts)
        or ":" in parts[0]
    ):
        raise AmiPreparationError()
    path = PurePosixPath(lexical)
    if (
        path.is_absolute()
        or not path.parts
        or path.as_posix() != lexical
    ):
        raise AmiPreparationError()
    return path


def _zip_member_kind(info: zipfile.ZipInfo) -> None:
    mode = (info.external_attr >> 16) & 0xFFFF
    kind = stat.S_IFMT(mode)
    is_directory = info.is_dir()
    if kind not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise AmiPreparationError()
    if (is_directory and kind == stat.S_IFREG) or (
        not is_directory and kind == stat.S_IFDIR
    ):
        raise AmiPreparationError()


def _read_zip_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
) -> bytes:
    if info.is_dir() or info.file_size <= 0 or info.file_size > _MAX_ZIP_MEMBER_BYTES:
        raise AmiPreparationError()
    try:
        with archive.open(info, "r") as handle:
            data = handle.read(info.file_size + 1)
            trailing = handle.read(1)
    except (
        OSError,
        EOFError,
        RuntimeError,
        NotImplementedError,
        zipfile.BadZipFile,
    ):
        raise AmiPreparationError() from None
    if len(data) != info.file_size or trailing:
        raise AmiPreparationError()
    return data


def _annotation_members(raw: bytes) -> Mapping[str, bytes]:
    selected: dict[str, zipfile.ZipInfo] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(raw), "r", allowZip64=False) as archive:
            infos = archive.infolist()
            if not infos or len(infos) > _MAX_ZIP_MEMBERS:
                raise AmiPreparationError()
            seen: set[str] = set()
            expanded = 0
            for info in infos:
                path = _safe_zip_name(info.filename)
                folded = info.filename.casefold()
                if folded in seen:
                    raise AmiPreparationError()
                seen.add(folded)
                _zip_member_kind(info)
                if (
                    info.flag_bits & 0x1
                    or info.compress_type
                    not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                    or info.file_size < 0
                    or info.file_size > _MAX_ZIP_MEMBER_BYTES
                    or info.compress_size < 0
                    or info.compress_size > len(raw)
                ):
                    raise AmiPreparationError()
                expanded += info.file_size
                if expanded > _MAX_ZIP_EXPANDED_BYTES:
                    raise AmiPreparationError()
                if info.is_dir():
                    continue
                match = _WORD_MEMBER_RE.search(path.as_posix())
                if match is not None:
                    speaker = match.group(1)
                    if speaker in selected:
                        raise AmiPreparationError()
                    selected[speaker] = info
            if set(selected) != set(_SPEAKERS):
                raise AmiPreparationError()
            return {
                speaker: _read_zip_member(archive, selected[speaker])
                for speaker in _SPEAKERS
            }
    except (
        OSError,
        EOFError,
        RuntimeError,
        NotImplementedError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
    ):
        raise AmiPreparationError() from None


def _riff_pcm16_mono(raw: bytes) -> np.ndarray:
    if len(raw) < 44 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise AmiPreparationError()
    declared = struct.unpack_from("<I", raw, 4)[0]
    if declared + 8 != len(raw):
        raise AmiPreparationError()
    offset = 12
    format_chunk: bytes | None = None
    data_chunk: bytes | None = None
    while offset < len(raw):
        if offset + 8 > len(raw):
            raise AmiPreparationError()
        chunk_id = raw[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", raw, offset + 4)[0]
        offset += 8
        end = offset + chunk_size
        padded_end = end + (chunk_size & 1)
        if end > len(raw) or padded_end > len(raw):
            raise AmiPreparationError()
        payload = raw[offset:end]
        if chunk_id == b"fmt ":
            if format_chunk is not None:
                raise AmiPreparationError()
            format_chunk = payload
        elif chunk_id == b"data":
            if data_chunk is not None:
                raise AmiPreparationError()
            data_chunk = payload
        offset = padded_end
    if offset != len(raw) or format_chunk is None or data_chunk is None:
        raise AmiPreparationError()
    if len(format_chunk) != 16 or not data_chunk or len(data_chunk) % 2:
        raise AmiPreparationError()
    (
        encoding,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        sample_bits,
    ) = struct.unpack("<HHIIHH", format_chunk)
    if (
        encoding != 1
        or channels != 1
        or sample_rate != SAMPLE_RATE_HZ
        or byte_rate != SAMPLE_RATE_HZ * 2
        or block_align != 2
        or sample_bits != 16
    ):
        raise AmiPreparationError()
    pcm = np.frombuffer(data_chunk, dtype="<i2")
    if pcm.size <= 0:
        raise AmiPreparationError()
    return pcm


def _local_name(value: str) -> str:
    return value.rsplit("}", 1)[-1].rsplit(":", 1)[-1]


def _local_attributes(element: ET.Element) -> Mapping[str, str]:
    result: dict[str, str] = {}
    for raw_name, value in element.attrib.items():
        name = _local_name(raw_name)
        if name in result:
            raise AmiPreparationError()
        result[name] = value
    return result


def _clean_token(value: object, *, punctuation: bool) -> str:
    if not isinstance(value, str):
        raise AmiPreparationError()
    token = value.strip()
    if (
        not token
        or token != value
        or len(token) > (16 if punctuation else 128)
        or any(ord(character) < 32 for character in token)
        or (not punctuation and any(character.isspace() for character in token))
    ):
        raise AmiPreparationError()
    if punctuation and any(
        character not in string.punctuation
        and not unicodedata.category(character).startswith("P")
        for character in token
    ):
        raise AmiPreparationError()
    return token


def _time_value(value: object) -> Decimal:
    if not isinstance(value, str) or not value or len(value) > 32:
        raise AmiPreparationError()
    try:
        result = Decimal(value)
    except (InvalidOperation, ValueError):
        raise AmiPreparationError() from None
    if not result.is_finite() or result < 0:
        raise AmiPreparationError()
    return result


def _time_to_sample(value: Decimal, *, end: bool) -> int:
    rounding = ROUND_CEILING if end else ROUND_FLOOR
    try:
        return int(
            (value * SAMPLE_RATE_HZ).to_integral_value(rounding=rounding)
        )
    except (InvalidOperation, OverflowError, ValueError):
        raise AmiPreparationError() from None


def _parse_speaker_annotations(
    raw: bytes,
    *,
    speaker: str,
    audio_samples: int,
) -> _SpeakerAnnotations:
    if (
        speaker not in _SPEAKERS
        or not raw
        or len(raw) > _MAX_ZIP_MEMBER_BYTES
        or b"<!doctype" in raw.lower()
        or b"<!entity" in raw.lower()
    ):
        raise AmiPreparationError()
    try:
        root = ET.fromstring(raw)
    except (ET.ParseError, UnicodeError, ValueError):
        raise AmiPreparationError() from None
    duration = Decimal(audio_samples) / Decimal(SAMPLE_RATE_HZ)
    words: list[_Word] = []
    activities: list[_Activity] = []
    pending_prefix = ""
    previous_word_end: Decimal | None = None
    previous_timed_start: Decimal | None = None
    for element_index, element in enumerate(root.iter()):
        if element_index >= _MAX_ANNOTATION_ELEMENTS_PER_SPEAKER:
            raise AmiPreparationError()
        name = _local_name(element.tag) if isinstance(element.tag, str) else ""
        attributes = _local_attributes(element)
        has_start = "starttime" in attributes
        has_end = "endtime" in attributes
        if has_start != has_end:
            raise AmiPreparationError()
        start: Decimal | None = None
        end: Decimal | None = None
        if has_start:
            start = _time_value(attributes["starttime"])
            end = _time_value(attributes["endtime"])
            if (
                end < start
                or end > duration
                or (
                    previous_timed_start is not None
                    and start < previous_timed_start
                )
            ):
                raise AmiPreparationError()
            previous_timed_start = start

        if name not in {"w", "punc"}:
            if start is not None and end is not None and end > start:
                activities.append(
                    _Activity(
                        speaker=speaker,
                        kind=name,
                        start_sample=_time_to_sample(start, end=False),
                        end_sample=_time_to_sample(end, end=True),
                        ordinal=len(activities),
                    )
                )
            continue
        if list(element):
            raise AmiPreparationError()
        text = "".join(element.itertext())
        punctuation_word = name == "w" and attributes.get("punc") == "true"
        if name == "w" and "punc" in attributes and not punctuation_word:
            raise AmiPreparationError()
        if name == "punc":
            if (start is None) != (end is None):
                raise AmiPreparationError()
            token = _clean_token(text, punctuation=True)
            if (
                start is not None
                and end is not None
                and (
                    start != end
                    or previous_word_end is None
                    or start != previous_word_end
                )
            ):
                raise AmiPreparationError()
            if words:
                words[-1] = replace(words[-1], text=words[-1].text + token)
            else:
                pending_prefix += token
                if len(pending_prefix) > 16:
                    raise AmiPreparationError()
            continue
        if punctuation_word:
            if (
                start is None
                or end is None
                or start != end
                or previous_word_end is None
                or start != previous_word_end
                or not words
            ):
                raise AmiPreparationError()
            token = _clean_token(text, punctuation=True)
            words[-1] = replace(words[-1], text=words[-1].text + token)
            continue
        token = _clean_token(text, punctuation=False)
        if start is None or end is None:
            raise AmiPreparationError()
        if (
            end <= start
            or (
                previous_word_end is not None
                and start < previous_word_end
            )
        ):
            raise AmiPreparationError()
        start_sample = _time_to_sample(start, end=False)
        end_sample = _time_to_sample(end, end=True)
        if (
            start_sample < 0
            or start_sample >= end_sample
            or end_sample > audio_samples
        ):
            raise AmiPreparationError()
        rendered = pending_prefix + token
        pending_prefix = ""
        words.append(
            _Word(
                speaker=speaker,
                text=rendered,
                start_sample=start_sample,
                end_sample=end_sample,
                ordinal=len(words),
            )
        )
        if len(words) > _MAX_WORDS_PER_SPEAKER:
            raise AmiPreparationError()
        previous_word_end = end
    if not words or pending_prefix:
        raise AmiPreparationError()
    return _SpeakerAnnotations(tuple(words), tuple(activities))


def _ordered_annotations(
    annotations: Mapping[str, bytes],
    *,
    audio_samples: int,
) -> tuple[tuple[_Word, ...], tuple[_Activity, ...]]:
    parsed = {
        speaker: _parse_speaker_annotations(
            annotations[speaker],
            speaker=speaker,
            audio_samples=audio_samples,
        )
        for speaker in _SPEAKERS
    }
    words = [
        word
        for speaker in _SPEAKERS
        for word in parsed[speaker].words
    ]
    activities = [
        activity
        for speaker in _SPEAKERS
        for activity in parsed[speaker].activities
    ]
    words.sort(
        key=lambda word: (
            word.start_sample,
            word.end_sample,
            word.speaker,
            word.ordinal,
            word.text,
        )
    )
    activities.sort(
        key=lambda activity: (
            activity.start_sample,
            activity.end_sample,
            activity.speaker,
            activity.kind,
            activity.ordinal,
        )
    )
    if not words:
        raise AmiPreparationError()
    return tuple(words), tuple(activities)


def _speech_components(words: Sequence[_Word]) -> tuple[_Component, ...]:
    components: list[_Component] = []
    active: list[_Word] = []
    start = -1
    end = -1
    for word in words:
        if not active or word.start_sample >= end:
            if active:
                components.append(_Component(start, end, tuple(active)))
            active = [word]
            start = word.start_sample
            end = word.end_sample
        else:
            active.append(word)
            end = max(end, word.end_sample)
    if active:
        components.append(_Component(start, end, tuple(active)))
    return tuple(components)


def _merged_ranges(
    intervals: Iterable[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    result: list[list[int]] = []
    for start, end in sorted(intervals):
        if start >= end:
            raise AmiPreparationError()
        if not result or start > result[-1][1]:
            result.append([start, end])
        else:
            result[-1][1] = max(result[-1][1], end)
    return tuple((start, end) for start, end in result)


def _speech_samples(words: Sequence[_Word]) -> int:
    return sum(
        end - start
        for start, end in _merged_ranges(
            (word.start_sample, word.end_sample) for word in words
        )
    )


def _different_speaker_overlap(words: Sequence[_Word]) -> int:
    events: dict[int, list[tuple[int, str]]] = {}
    for word in words:
        events.setdefault(word.start_sample, []).append((1, word.speaker))
        events.setdefault(word.end_sample, []).append((-1, word.speaker))
    active: dict[str, int] = {}
    previous: int | None = None
    overlap = 0
    for sample in sorted(events):
        if previous is not None and len(active) >= 2:
            overlap += sample - previous
        for delta, speaker in sorted(events[sample], key=lambda item: item[0]):
            count = active.get(speaker, 0) + delta
            if count < 0:
                raise AmiPreparationError()
            if count:
                active[speaker] = count
            else:
                active.pop(speaker, None)
        previous = sample
    if active:
        raise AmiPreparationError()
    return overlap


def _transition_gap(words: Sequence[_Word]) -> int | None:
    result: list[int] = []
    for previous, current in zip(words, words[1:]):
        gap = current.start_sample - previous.end_sample
        if (
            previous.speaker != current.speaker
            and 0 <= gap <= _MAX_TRANSITION_GAP_SAMPLES
        ):
            result.append(gap)
    return min(result) if result else None


def _select_windows(
    words: tuple[_Word, ...],
) -> Mapping[str, _Window]:
    components = _speech_components(words)
    if not components:
        raise AmiPreparationError()
    best: dict[str, tuple[tuple[int, ...], _Window]] = {}
    for left in range(len(components)):
        selected: list[_Word] = []
        for right in range(left, len(components)):
            start = components[left].start_sample
            end = components[right].end_sample
            if end - start > _MAX_SOURCE_SAMPLES:
                break
            selected.extend(components[right].words)
            ordered = tuple(
                sorted(
                    selected,
                    key=lambda word: (
                        word.start_sample,
                        word.end_sample,
                        word.speaker,
                        word.ordinal,
                        word.text,
                    ),
                )
            )
            window = _Window(start, end, ordered)
            speakers = {word.speaker for word in ordered}
            speech = _speech_samples(ordered)
            overlap = _different_speaker_overlap(ordered)
            duration = end - start
            if len(speakers) == 1:
                score = (speech, len(ordered), duration, -start)
                if "single" not in best or score > best["single"][0]:
                    best["single"] = (score, window)
            if overlap > 0:
                score = (overlap, speech, len(ordered), duration, -start)
                if "overlap" not in best or score > best["overlap"][0]:
                    best["overlap"] = (score, window)
            transition = _transition_gap(ordered)
            if overlap == 0 and len(speakers) >= 2 and transition is not None:
                score = (speech, len(ordered), -transition, duration, -start)
                if "transition" not in best or score > best["transition"][0]:
                    best["transition"] = (score, window)
    if not {"single", "overlap", "transition"}.issubset(best):
        raise AmiPreparationError()
    return {kind: best[kind][1] for kind in ("single", "overlap", "transition")}


def _select_silence(
    words: Sequence[_Word],
    activities: Sequence[_Activity],
    *,
    audio_samples: int,
) -> _Window:
    occupied = _merged_ranges(
        [
            (word.start_sample, word.end_sample)
            for word in words
        ]
        + [
            (activity.start_sample, activity.end_sample)
            for activity in activities
        ]
    )
    gaps: list[tuple[int, int]] = []
    cursor = 0
    for start, end in occupied:
        if start - cursor >= _MIN_REAL_SILENCE_SAMPLES:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if audio_samples - cursor >= _MIN_REAL_SILENCE_SAMPLES:
        gaps.append((cursor, audio_samples))
    if not gaps:
        raise AmiPreparationError()
    gap_start, gap_end = min(
        gaps,
        key=lambda item: (-(item[1] - item[0]), item[0], item[1]),
    )
    if gap_end - gap_start > _MAX_SOURCE_SAMPLES:
        extra = gap_end - gap_start - _MAX_SOURCE_SAMPLES
        gap_start += extra // 2
        gap_end = gap_start + _MAX_SOURCE_SAMPLES
    return _Window(gap_start, gap_end, ())


def _calibration_roomtone(silence: _Window) -> _Window:
    duration = silence.end_sample - silence.start_sample
    if duration < _CALIBRATION_SAMPLES or silence.words:
        raise AmiPreparationError()
    start = silence.start_sample + (duration - _CALIBRATION_SAMPLES) // 2
    return _Window(start, start + _CALIBRATION_SAMPLES, ())


def _case_pcm(
    source: np.ndarray,
    window: _Window,
    roomtone: _Window,
) -> tuple[bytes, int, int, int]:
    source_samples = window.end_sample - window.start_sample
    if (
        source_samples <= 0
        or source_samples > _MAX_SOURCE_SAMPLES
        or window.start_sample < 0
        or window.end_sample > source.size
        or roomtone.end_sample - roomtone.start_sample != _CALIBRATION_SAMPLES
        or roomtone.start_sample < 0
        or roomtone.end_sample > source.size
    ):
        raise AmiPreparationError()
    minimum = source_samples + _CALIBRATION_SAMPLES + _EDGE_PAD_SAMPLES
    samples = max(_MIN_CASE_SAMPLES, minimum)
    samples = ((samples + FRAME_SAMPLES - 1) // FRAME_SAMPLES) * FRAME_SAMPLES
    if samples > _MAX_CASE_SAMPLES:
        raise AmiPreparationError()
    left_padding = _CALIBRATION_SAMPLES
    right_padding = samples - source_samples - left_padding
    if (
        left_padding != _CALIBRATION_SAMPLES
        or right_padding < _EDGE_PAD_SAMPLES
    ):
        raise AmiPreparationError()
    result = np.zeros(samples, dtype="<f4")
    roomtone_pcm = source[
        roomtone.start_sample : roomtone.end_sample
    ].astype(np.float32) * np.float32(1.0 / 32768.0)
    result[:left_padding] = roomtone_pcm
    result[left_padding : left_padding + source_samples] = source[
        window.start_sample : window.end_sample
    ].astype(np.float32) * np.float32(1.0 / 32768.0)
    tail_start = left_padding + source_samples
    result[tail_start:] = np.resize(roomtone_pcm, right_padding)
    if (
        result.size != samples
        or not bool(np.all(np.isfinite(result)))
        or bool(np.any(np.abs(result) > 1.0))
    ):
        raise AmiPreparationError()
    return result.tobytes(order="C"), samples, left_padding, right_padding


def _case_words(window: _Window, *, left_padding: int) -> tuple[_Word, ...]:
    result: list[_Word] = []
    for word in window.words:
        if (
            word.start_sample < window.start_sample
            or word.end_sample > window.end_sample
        ):
            raise AmiPreparationError()
        result.append(
            replace(
                word,
                start_sample=left_padding
                + word.start_sample
                - window.start_sample,
                end_sample=left_padding + word.end_sample - window.start_sample,
            )
        )
    result.sort(
        key=lambda word: (
            word.start_sample,
            word.end_sample,
            word.speaker,
            word.text,
        )
    )
    return tuple(result)


def _annotation_rows(
    words: Sequence[_Word],
) -> tuple[list[dict[str, int]], list[dict[str, object]], list[dict[str, object]]]:
    speech = [
        {"start_sample": start, "end_sample": end}
        for start, end in _merged_ranges(
            (word.start_sample, word.end_sample) for word in words
        )
    ]
    speaker_ranges: list[tuple[int, int, str]] = []
    for speaker in sorted({word.speaker for word in words}):
        for start, end in _merged_ranges(
            (word.start_sample, word.end_sample)
            for word in words
            if word.speaker == speaker
        ):
            speaker_ranges.append((start, end, speaker))
    speaker_rows = [
        {
            "speaker_id": speaker,
            "role": "unknown",
            "start_sample": start,
            "end_sample": end,
        }
        for start, end, speaker in sorted(
            speaker_ranges,
            key=lambda item: (item[0], item[1], item[2]),
        )
    ]
    word_rows = [
        {
            "speaker_id": word.speaker,
            "text": word.text,
            "start_sample": word.start_sample,
            "end_sample": word.end_sample,
        }
        for word in words
    ]
    return speech, speaker_rows, word_rows


def _prepare_case(
    *,
    kind: str,
    source: np.ndarray,
    window: _Window,
    roomtone: _Window,
) -> _PreparedCase:
    if kind not in {"single", "overlap", "transition", "silence"}:
        raise AmiPreparationError()
    pcm, samples, left_padding, _right_padding = _case_pcm(
        source,
        window,
        roomtone,
    )
    words = _case_words(window, left_padding=left_padding)
    speech, speakers, word_rows = _annotation_rows(words)
    is_silence = kind == "silence"
    if is_silence != (not words):
        raise AmiPreparationError()
    filename = f"ami-es2004a-{kind}.f32le"
    digest = hashlib.sha256(pcm).hexdigest()
    tags = [
        "ami",
        "es2004a",
        "far-field",
        kind,
        "calibration-roomtone-2s",
        "postroll-roomtone-2s",
    ]
    if kind == "single":
        tags.append("single-speaker")
    elif kind == "overlap":
        tags.append("overlapping-speakers")
    elif kind == "transition":
        tags.append("turn-transition")
    else:
        tags.append("room-silence")
    manifest: dict[str, object] = {
        "id": f"ami-es2004a-{kind}",
        "tracks": {
            "mic": {
                "file": filename,
                "sha256": digest,
                "encoding": "f32le",
                "samples": samples,
            }
        },
        "speech_intervals": speech,
        "speaker_intervals": speakers,
        "word_intervals": word_rows,
        "assertion": "silence" if is_silence else "transcript",
        "expected_text": "" if is_silence else " ".join(word.text for word in words),
        "commands": [],
        "tags": tags,
        "aec_delay_samples": 0,
    }
    return _PreparedCase(
        case_id=f"ami-es2004a-{kind}",
        filename=filename,
        pcm=pcm,
        manifest=manifest,
    )


def _new_private_output(path: Path | str) -> Path:
    candidate = Path(os.path.abspath(Path(path).expanduser()))
    if not candidate.name or candidate.name in {".", ".."}:
        raise AmiPreparationError()
    try:
        with opened_directory_nofollow(candidate.parent) as (_stable, descriptor):
            os.mkdir(candidate.name, mode=0o700, dir_fd=descriptor)
        with opened_directory_nofollow(
            candidate,
            require_private=True,
        ) as (stable, _descriptor):
            return stable
    except (OSError, BoundedReadError):
        raise AmiPreparationError() from None


def _write_new_private(path: Path, payload: bytes) -> None:
    descriptor = -1
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", buffering=0) as handle:
            descriptor = -1
            handle.write(payload)
            os.fsync(handle.fileno())
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
        ):
            raise AmiPreparationError()
    except (OSError, ValueError, OverflowError, AmiPreparationError):
        raise AmiPreparationError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def prepare_ami_capture_replay(
    *,
    annotations_zip: Path | str,
    annotations_sha256: str,
    annotations_bytes: int,
    audio_wav: Path | str,
    audio_sha256: str,
    audio_bytes: int,
    output_dir: Path | str,
) -> LoadedReplayCorpus:
    """Verify local AMI inputs and create one private no-overwrite corpus."""

    annotations_source = _verified_source(
        annotations_zip,
        expected_sha256=annotations_sha256,
        expected_bytes=annotations_bytes,
        maximum_bytes=_ANNOTATION_ARCHIVE_MAX_BYTES,
    )
    audio_source = _verified_source(
        audio_wav,
        expected_sha256=audio_sha256,
        expected_bytes=audio_bytes,
        maximum_bytes=_AUDIO_MAX_BYTES,
    )
    source_pcm = _riff_pcm16_mono(audio_source.data)
    annotations = _annotation_members(annotations_source.data)
    words, activities = _ordered_annotations(
        annotations,
        audio_samples=int(source_pcm.size),
    )
    windows = dict(_select_windows(words))
    windows["silence"] = _select_silence(
        words,
        activities,
        audio_samples=int(source_pcm.size),
    )
    roomtone = _calibration_roomtone(windows["silence"])
    cases = tuple(
        _prepare_case(
            kind=kind,
            source=source_pcm,
            window=windows[kind],
            roomtone=roomtone,
        )
        for kind in ("single", "overlap", "transition", "silence")
    )
    purpose = (
        "AMI ES2004a Array1-01 bounded capture replay v1; "
        f"annotations_sha256={annotations_source.sha256};"
        f"annotations_bytes={annotations_source.size_bytes};"
        f"audio_sha256={audio_source.sha256};"
        f"audio_bytes={audio_source.size_bytes};"
        "selection=phenomena-v1"
    )
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "purpose": purpose,
                "sample_rate_hz": SAMPLE_RATE_HZ,
                "frame_samples": FRAME_SAMPLES,
                "cases": [case.manifest for case in cases],
            },
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )

    destination = _new_private_output(output_dir)
    for case in cases:
        _write_new_private(destination / case.filename, case.pcm)
    manifest_path = destination / "capture-replay.json"
    _write_new_private(manifest_path, payload)
    try:
        loaded = load_corpus(manifest_path)
    except ReplayCorpusError:
        raise AmiPreparationError() from None
    if (
        len(loaded.cases) != 4
        or loaded.digest != hashlib.sha256(payload).hexdigest()
        or any(
            case.samples < _MIN_CASE_SAMPLES
            or case.samples > _MAX_CASE_SAMPLES
            for case in loaded.cases
        )
    ):
        raise AmiPreparationError()
    return loaded


def _safe_result(corpus: LoadedReplayCorpus) -> Mapping[str, object]:
    total_samples = sum(case.samples for case in corpus.cases)
    seconds = total_samples / SAMPLE_RATE_HZ
    if not math.isfinite(seconds):
        raise AmiPreparationError()
    return {
        "ok": True,
        "schema_version": corpus.schema_version,
        "manifest_sha256": corpus.digest,
        "cases": len(corpus.cases),
        "audio_bytes": corpus.audio_bytes,
        "total_seconds": round(seconds, 3),
    }


def _positive_int(value: str) -> int:
    try:
        result = int(value, 10)
    except ValueError:
        raise argparse.ArgumentTypeError("expected a positive byte count") from None
    if result <= 0:
        raise argparse.ArgumentTypeError("expected a positive byte count")
    return result


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise AmiPreparationError()


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description=(
            "Prepare four private AMI ES2004a production capture-replay cases "
            "from hash-pinned local inputs; no download, model, network, or "
            "audio-device access."
        )
    )
    parser.add_argument("--annotations-zip", type=Path, required=True)
    parser.add_argument("--annotations-sha256", required=True)
    parser.add_argument("--annotations-bytes", type=_positive_int, required=True)
    parser.add_argument("--audio-wav", type=Path, required=True)
    parser.add_argument("--audio-sha256", required=True)
    parser.add_argument("--audio-bytes", type=_positive_int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        corpus = prepare_ami_capture_replay(
            annotations_zip=args.annotations_zip,
            annotations_sha256=args.annotations_sha256,
            annotations_bytes=args.annotations_bytes,
            audio_wav=args.audio_wav,
            audio_sha256=args.audio_sha256,
            audio_bytes=args.audio_bytes,
            output_dir=args.output_dir,
        )
        result = _safe_result(corpus)
        code = 0
    except Exception:  # noqa: BLE001 - no paths or transcripts may escape
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
