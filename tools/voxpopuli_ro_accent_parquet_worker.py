#!/usr/bin/env python3
"""Isolated selector for the pinned VoxPopuli Romanian-accented English test.

The parent passes two already-open, hash-pinned Parquet descriptors.  This
worker imports PyArrow only in the explicitly selected preparation environment,
validates the exact physical and Hugging Face feature schema, scans with one
thread, and writes only selected exact IEEE-float32 WAVs plus a private result.
It never opens an audio ``path`` value and never writes to stdout or stderr.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import struct
from typing import BinaryIO, Iterator, Mapping, Sequence


PROTOCOL_VERSION = 2
PYARROW_VERSION = "25.0.0"
SOURCE_REVISION = "42f01879c780b4a2e90ec0b4f616c2ece526e4f1"
SELECTION_SEED = "speaker-voxpopuli-en-ro-accent-v1"
SOURCE_TOTAL_ROWS = 8_387
EXPECTED_EN_RO_SPEAKERS = 27
LANGUAGE_ID = 16
ACCENT = "en_ro"
SELECTION_COUNT = 24
SOURCE_SHARDS = (
    {
        "shard_index": 0,
        "filename": "test-00000-of-00002.parquet",
        "size_bytes": 2_471_743_754,
        "sha256": "202d71bde2680f30aa7ae1050a98cdf85972458f94e6493db2aa7b8dd8ff46ec",
    },
    {
        "shard_index": 1,
        "filename": "test-00001-of-00002.parquet",
        "size_bytes": 2_474_983_297,
        "sha256": "7d61b4c7a2e5c565a382543236442a7a8dd081c985e52dd1084905e4becc3901",
    },
)
DURATION_BUCKETS = (
    ("d0_2_6s", 2 * 16_000, 6 * 16_000, False, 6),
    ("d1_6_10s", 6 * 16_000, 10 * 16_000, False, 6),
    ("d2_10_15s", 10 * 16_000, 15 * 16_000, False, 6),
    ("d3_15_25s", 15 * 16_000, 25 * 16_000, True, 6),
)
LANGUAGE_NAMES = (
    "en",
    "de",
    "fr",
    "es",
    "pl",
    "it",
    "ro",
    "hu",
    "cs",
    "nl",
    "fi",
    "hr",
    "sk",
    "sl",
    "et",
    "lt",
    "en_accented",
)
FEATURE_NAMES = (
    "audio_id",
    "language",
    "audio",
    "raw_text",
    "normalized_text",
    "gender",
    "speaker_id",
    "is_gold_transcript",
    "accent",
)
METADATA_COLUMNS = tuple(name for name in FEATURE_NAMES if name != "audio")
SCHEMA_CONTRACT = {
    "fields": [
        ["audio_id", "string"],
        ["language", "int64:class-label"],
        ["audio", "struct<bytes:binary,path:string>:audio-16000"],
        ["raw_text", "string"],
        ["normalized_text", "string"],
        ["gender", "string"],
        ["speaker_id", "string"],
        ["is_gold_transcript", "bool"],
        ["accent", "string"],
    ],
    "huggingface_metadata": "semantic-exact-v1",
    "embedded_audio": {
        "container": "exact-riff-wave-v2",
        "chunk_order": [["fmt ", 18], ["fact", 4], ["data", "4*n"]],
        "fmt_le": [3, 1, 16_000, 64_000, 4, 32],
        "fmt_extra_hex": "0000",
        "fact_le": "uint32-samples-equals-data-bytes-div-4",
        "data": "little-endian-ieee-binary32",
        "empty_data": "valid-container-ineligible-row",
        "selected_values": "finite-no-amplitude-bound-no-transform",
    },
}

_MAX_REQUEST_BYTES = 64 * 1024
_MAX_RESULT_BYTES = 256 * 1024
_MAX_TEXT_BYTES = 16 * 1024
_MAX_AUDIO_BYTES = 16 * 1024 * 1024
_MAX_TARGET_ROWS = 4_096
_MAX_TARGET_AUDIO_BYTES = 512 * 1024 * 1024
_METADATA_BATCH_ROWS = 256
_AUDIO_BATCH_ROWS = 8
_REQUEST_FIELDS = {
    "schema_version",
    "source_revision",
    "selection_seed",
    "source_descriptors",
}
_REQUEST_SOURCE_FIELDS = {
    "shard_index",
    "descriptor",
    "filename",
    "size_bytes",
    "sha256",
}
_RESULT_FIELDS = {
    "schema_version",
    "pyarrow_version",
    "source_revision",
    "selection_seed",
    "schema_contract_sha256",
    "source_total_rows",
    "source_shards",
    "target_accent_speakers",
    "eligible_rows",
    "eligible_speakers",
    "eligible_set_sha256",
    "bucket_counts",
    "cases",
}
_RESULT_SOURCE_FIELDS = {
    "shard_index",
    "filename",
    "size_bytes",
    "sha256",
    "rows",
    "row_groups",
}
_RESULT_CASE_FIELDS = {
    "selection_index",
    "shard_index",
    "row_index",
    "audio_id",
    "speaker_id",
    "gender",
    "transcription",
    "transcription_sha256",
    "duration_samples",
    "duration_bucket",
    "row_rank_sha256",
    "speaker_bucket_rank_sha256",
    "audio_file",
    "audio_sha256",
    "audio_size_bytes",
}


class WorkerError(RuntimeError):
    """A detail-free source, schema, selection, or output failure."""


@dataclass(frozen=True, slots=True)
class _MetadataRow:
    shard_index: int
    row_index: int
    audio_id: str
    speaker_id: str
    gender: str
    transcription: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class Candidate:
    shard_index: int
    row_index: int
    audio_id: str
    speaker_id: str
    gender: str
    transcription: str = field(repr=False)
    duration_samples: int
    duration_bucket: str
    row_rank_sha256: str
    speaker_bucket_rank_sha256: str
    audio_sha256: str
    audio_size_bytes: int


@dataclass(slots=True)
class _OpenedShard:
    record: Mapping[str, object]
    original_fd: int = field(repr=False)
    original_identity: tuple[int, ...]
    handle: BinaryIO = field(repr=False)
    parquet: object = field(repr=False)
    rows: int
    row_groups: int


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, UnicodeError, ValueError, OverflowError):
        raise WorkerError() from None


SCHEMA_CONTRACT_SHA256 = hashlib.sha256(
    _canonical_json_bytes(SCHEMA_CONTRACT)
).hexdigest()


def _strict_json_loads(raw: bytes | str) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate")
            result[key] = value
        return result

    return json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("constant")),
    )


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _text(value: object, *, maximum: int = _MAX_TEXT_BYTES) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise WorkerError()
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeError:
        raise WorkerError() from None
    if len(encoded) > maximum:
        raise WorkerError()
    return value


def _sha256(value: object) -> str:
    text = _text(value, maximum=64)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise WorkerError()
    return text


def _load_request(path: Path) -> Mapping[str, object]:
    try:
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > _MAX_REQUEST_BYTES
            or stat.S_IMODE(before.st_mode) & 0o077
        ):
            raise WorkerError()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if _identity(opened) != _identity(before):
                raise WorkerError()
            payload = os.read(descriptor, before.st_size + 1)
            after = os.fstat(descriptor)
            current = path.lstat()
            if (
                len(payload) != before.st_size
                or _identity(after) != _identity(opened)
                or _identity(current) != _identity(opened)
            ):
                raise WorkerError()
        finally:
            os.close(descriptor)
        request = _strict_json_loads(payload)
    except WorkerError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise WorkerError() from exc
    if not isinstance(request, dict) or set(request) != _REQUEST_FIELDS:
        raise WorkerError()
    if (
        request.get("schema_version") != PROTOCOL_VERSION
        or request.get("source_revision") != SOURCE_REVISION
        or request.get("selection_seed") != SELECTION_SEED
    ):
        raise WorkerError()
    records = request.get("source_descriptors")
    if not isinstance(records, list) or len(records) != len(SOURCE_SHARDS):
        raise WorkerError()
    seen_descriptors: set[int] = set()
    for expected, record in zip(SOURCE_SHARDS, records, strict=True):
        if not isinstance(record, dict) or set(record) != _REQUEST_SOURCE_FIELDS:
            raise WorkerError()
        descriptor = record.get("descriptor")
        if (
            record.get("shard_index") != expected["shard_index"]
            or record.get("filename") != expected["filename"]
            or record.get("size_bytes") != expected["size_bytes"]
            or record.get("sha256") != expected["sha256"]
            or isinstance(descriptor, bool)
            or not isinstance(descriptor, int)
            or descriptor < 3
            or descriptor in seen_descriptors
        ):
            raise WorkerError()
        seen_descriptors.add(descriptor)
    return request


def _descriptor_sha256(descriptor: int, expected_size: int) -> str:
    try:
        digest = hashlib.sha256()
        consumed = 0
        while consumed <= expected_size:
            chunk = os.pread(
                descriptor,
                min(1024 * 1024, expected_size + 1 - consumed),
                consumed,
            )
            if not chunk:
                break
            consumed += len(chunk)
            if consumed > expected_size:
                raise WorkerError()
            digest.update(chunk)
        if consumed != expected_size:
            raise WorkerError()
        return digest.hexdigest()
    except WorkerError:
        raise
    except (OSError, OverflowError, ValueError):
        raise WorkerError() from None


def _value_feature(value: object, dtype: str) -> None:
    if not isinstance(value, dict) or value.get("_type") != "Value":
        raise WorkerError()
    if set(value) != {"dtype", "_type"} or value.get("dtype") != dtype:
        raise WorkerError()


def _validate_huggingface_metadata(metadata: Mapping[bytes, bytes] | None) -> None:
    if not isinstance(metadata, dict) or set(metadata) != {b"huggingface"}:
        raise WorkerError()
    try:
        payload = _strict_json_loads(metadata[b"huggingface"])
    except Exception as exc:  # noqa: BLE001
        raise WorkerError() from exc
    if not isinstance(payload, dict) or set(payload) != {"info"}:
        raise WorkerError()
    info = payload.get("info")
    if not isinstance(info, dict) or set(info) != {"features"}:
        raise WorkerError()
    features = info.get("features")
    if not isinstance(features, dict) or tuple(features) != FEATURE_NAMES:
        raise WorkerError()
    for name in (
        "audio_id",
        "raw_text",
        "normalized_text",
        "gender",
        "speaker_id",
        "accent",
    ):
        _value_feature(features[name], "string")
    _value_feature(features["is_gold_transcript"], "bool")
    language = features["language"]
    if (
        not isinstance(language, dict)
        or language.get("_type") != "ClassLabel"
        or set(language) - {"names", "_type", "id"}
        or tuple(language.get("names") or ()) != LANGUAGE_NAMES
        or language.get("id") not in {None, "language"}
    ):
        raise WorkerError()
    audio = features["audio"]
    if (
        not isinstance(audio, dict)
        or audio.get("_type") != "Audio"
        or audio.get("sampling_rate") != 16_000
        or set(audio)
        - {"sampling_rate", "decode", "num_channels", "streamable", "id", "_type"}
        or audio.get("decode", True) is not True
        or audio.get("num_channels") not in {None, 1}
        or audio.get("streamable", False) is not False
        or audio.get("id") not in {None, "audio"}
    ):
        raise WorkerError()


def _validate_arrow_schema(schema: object) -> None:
    import pyarrow as pa

    expected = pa.schema(
        [
            pa.field("audio_id", pa.string()),
            pa.field("language", pa.int64()),
            pa.field(
                "audio",
                pa.struct(
                    [
                        pa.field("bytes", pa.binary()),
                        pa.field("path", pa.string()),
                    ]
                ),
            ),
            pa.field("raw_text", pa.string()),
            pa.field("normalized_text", pa.string()),
            pa.field("gender", pa.string()),
            pa.field("speaker_id", pa.string()),
            pa.field("is_gold_transcript", pa.bool_()),
            pa.field("accent", pa.string()),
        ]
    )
    if not hasattr(schema, "remove_metadata") or not schema.remove_metadata().equals(
        expected
    ):
        raise WorkerError()
    _validate_huggingface_metadata(schema.metadata)


@contextmanager
def _opened_shards(
    pq, records: Sequence[Mapping[str, object]]
) -> Iterator[list[_OpenedShard]]:
    opened_shards: list[_OpenedShard] = []
    try:
        for expected, record in zip(SOURCE_SHARDS, records, strict=True):
            descriptor = int(record["descriptor"])
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_size != expected["size_bytes"]
                or stat.S_IMODE(metadata.st_mode) & 0o077
                or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
                or _descriptor_sha256(descriptor, int(expected["size_bytes"]))
                != expected["sha256"]
            ):
                raise WorkerError()
            duplicate = os.dup(descriptor)
            handle = os.fdopen(duplicate, "rb", buffering=0)
            try:
                parquet = pq.ParquetFile(handle)
                _validate_arrow_schema(parquet.schema_arrow)
                rows = parquet.metadata.num_rows
                row_groups = parquet.metadata.num_row_groups
                if (
                    isinstance(rows, bool)
                    or not isinstance(rows, int)
                    or rows <= 0
                    or rows > SOURCE_TOTAL_ROWS
                    or isinstance(row_groups, bool)
                    or not isinstance(row_groups, int)
                    or row_groups <= 0
                    or row_groups > rows
                ):
                    raise WorkerError()
            except Exception:
                handle.close()
                raise
            opened_shards.append(
                _OpenedShard(
                    record=record,
                    original_fd=descriptor,
                    original_identity=_identity(metadata),
                    handle=handle,
                    parquet=parquet,
                    rows=rows,
                    row_groups=row_groups,
                )
            )
        if sum(shard.rows for shard in opened_shards) != SOURCE_TOTAL_ROWS:
            raise WorkerError()
        yield opened_shards
        for shard in opened_shards:
            if _identity(os.fstat(shard.original_fd)) != shard.original_identity:
                raise WorkerError()
    except WorkerError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise WorkerError() from exc
    finally:
        for shard in opened_shards:
            try:
                shard.handle.close()
            except OSError:
                pass


def _metadata_rows(shard: _OpenedShard) -> Iterator[Mapping[str, object]]:
    consumed = 0
    try:
        batches = shard.parquet.iter_batches(
            batch_size=_METADATA_BATCH_ROWS,
            columns=list(METADATA_COLUMNS),
            use_threads=False,
        )
        for batch in batches:
            rows = batch.to_pylist()
            if not isinstance(rows, list) or not rows:
                raise WorkerError()
            for row in rows:
                if not isinstance(row, dict) or set(row) != set(METADATA_COLUMNS):
                    raise WorkerError()
                consumed += 1
                if consumed > shard.rows:
                    raise WorkerError()
                yield row
        if consumed != shard.rows:
            raise WorkerError()
    except WorkerError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise WorkerError() from exc


def _wav_samples(payload: bytes) -> int:
    """Validate the pinned embedded IEEE-float WAV shape and return samples."""

    if (
        not isinstance(payload, bytes)
        or len(payload) < 58
        or len(payload) > _MAX_AUDIO_BYTES
        or payload[:4] != b"RIFF"
        or payload[8:12] != b"WAVE"
        or struct.unpack_from("<I", payload, 4)[0] + 8 != len(payload)
    ):
        raise WorkerError()
    fmt_size = struct.unpack_from("<I", payload, 16)[0]
    fmt = struct.unpack_from("<HHIIHH", payload, 20)
    fact_size = struct.unpack_from("<I", payload, 42)[0]
    fact_samples = struct.unpack_from("<I", payload, 46)[0]
    data_size = struct.unpack_from("<I", payload, 54)[0]
    if (
        payload[12:16] != b"fmt "
        or fmt_size != 18
        or fmt != (3, 1, 16_000, 64_000, 4, 32)
        or payload[36:38] != b"\x00\x00"
        or payload[38:42] != b"fact"
        or fact_size != 4
        or payload[50:54] != b"data"
        or data_size % 4
        or 58 + data_size != len(payload)
        or fact_samples != data_size // 4
    ):
        raise WorkerError()
    return data_size // 4


def duration_bucket(samples: int) -> str | None:
    if isinstance(samples, bool) or not isinstance(samples, int):
        return None
    for name, lower, upper, inclusive_upper, _quota in DURATION_BUCKETS:
        if samples >= lower and (
            samples <= upper if inclusive_upper else samples < upper
        ):
            return name
    return None


def _rank(kind: str, *values: str) -> str:
    payload = SELECTION_SEED.encode("ascii") + b"\0" + SOURCE_REVISION.encode("ascii")
    payload += b"\0" + kind.encode("ascii")
    for value in values:
        payload += b"\0" + value.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def row_rank(speaker_id: str, audio_id: str) -> str:
    return _rank("row", speaker_id, audio_id)


def speaker_bucket_rank(bucket: str, speaker_id: str) -> str:
    return _rank("speaker-bucket", bucket, speaker_id)


def _row_group_offsets(parquet: object) -> tuple[tuple[int, int], ...]:
    offsets: list[tuple[int, int]] = []
    start = 0
    try:
        for index in range(parquet.metadata.num_row_groups):
            rows = parquet.metadata.row_group(index).num_rows
            if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
                raise WorkerError()
            offsets.append((start, start + rows))
            start += rows
        if start != parquet.metadata.num_rows:
            raise WorkerError()
    except WorkerError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise WorkerError() from exc
    return tuple(offsets)


def _audio_rows(
    shard: _OpenedShard,
    requested_rows: set[int],
) -> Iterator[tuple[int, bytes, str]]:
    if not requested_rows or any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < shard.rows
        for index in requested_rows
    ):
        if requested_rows:
            raise WorkerError()
        return
    found: set[int] = set()
    try:
        for group_index, (start, end) in enumerate(_row_group_offsets(shard.parquet)):
            group_targets = {index for index in requested_rows if start <= index < end}
            if not group_targets:
                continue
            local = 0
            batches = shard.parquet.iter_batches(
                batch_size=_AUDIO_BATCH_ROWS,
                row_groups=[group_index],
                columns=["audio"],
                use_threads=False,
            )
            for batch in batches:
                rows = batch.to_pylist()
                if not isinstance(rows, list) or not rows:
                    raise WorkerError()
                for row in rows:
                    absolute = start + local
                    local += 1
                    if absolute not in group_targets:
                        continue
                    if not isinstance(row, dict) or set(row) != {"audio"}:
                        raise WorkerError()
                    audio = row["audio"]
                    if not isinstance(audio, dict) or set(audio) != {"bytes", "path"}:
                        raise WorkerError()
                    payload = audio["bytes"]
                    path = _text(audio["path"], maximum=4096)
                    if not isinstance(payload, bytes):
                        raise WorkerError()
                    found.add(absolute)
                    yield absolute, payload, path
            if local != end - start:
                raise WorkerError()
        if found != requested_rows:
            raise WorkerError()
    except WorkerError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise WorkerError() from exc


def _audio_path_matches(path: str, audio_id: str) -> bool:
    try:
        name = PurePosixPath(path).name
    except (TypeError, ValueError):
        return False
    return name == f"{audio_id}.wav"


def select_candidates(candidates: Sequence[Candidate]) -> tuple[Candidate, ...]:
    """Select a deterministic feasible 6-per-band, one-speaker assignment."""

    bucket_names = tuple(item[0] for item in DURATION_BUCKETS)
    quota_by_bucket = {item[0]: item[4] for item in DURATION_BUCKETS}
    best: dict[tuple[str, str], Candidate] = {}
    seen_audio: set[str] = set()
    for candidate in candidates:
        if (
            type(candidate) is not Candidate
            or candidate.duration_bucket not in bucket_names
        ):
            raise WorkerError()
        if candidate.audio_id in seen_audio:
            raise WorkerError()
        seen_audio.add(candidate.audio_id)
        key = (candidate.speaker_id, candidate.duration_bucket)
        current = best.get(key)
        if current is None or (
            candidate.row_rank_sha256,
            candidate.audio_id,
            candidate.shard_index,
            candidate.row_index,
        ) < (
            current.row_rank_sha256,
            current.audio_id,
            current.shard_index,
            current.row_index,
        ):
            best[key] = candidate

    slots = tuple(
        (bucket, slot)
        for bucket in bucket_names
        for slot in range(quota_by_bucket[bucket])
    )
    choices: dict[str, tuple[Candidate, ...]] = {}
    for bucket in bucket_names:
        choices[bucket] = tuple(
            sorted(
                (
                    candidate
                    for (_speaker, band), candidate in best.items()
                    if band == bucket
                ),
                key=lambda item: (
                    item.speaker_bucket_rank_sha256,
                    item.row_rank_sha256,
                    item.speaker_id,
                ),
            )
        )

    assignments: list[Candidate | None] = [None] * len(slots)
    speaker_slot: dict[str, int] = {}

    def assign(
        slot_index: int, visited_speakers: set[str], visited_slots: set[int]
    ) -> bool:
        if slot_index in visited_slots:
            return False
        visited_slots.add(slot_index)
        bucket = slots[slot_index][0]
        for candidate in choices[bucket]:
            speaker = candidate.speaker_id
            if speaker in visited_speakers:
                continue
            visited_speakers.add(speaker)
            previous = speaker_slot.get(speaker)
            if previous is None or assign(previous, visited_speakers, visited_slots):
                assignments[slot_index] = candidate
                speaker_slot[speaker] = slot_index
                return True
        return False

    for slot_index in range(len(slots)):
        if not assign(slot_index, set(), set()):
            raise WorkerError()
    if any(item is None for item in assignments):
        raise WorkerError()
    selected = [item for item in assignments if item is not None]
    if (
        len(selected) != SELECTION_COUNT
        or len({item.speaker_id for item in selected}) != SELECTION_COUNT
    ):
        raise WorkerError()
    ordered: list[Candidate] = []
    for bucket in bucket_names:
        rows = sorted(
            (item for item in selected if item.duration_bucket == bucket),
            key=lambda item: (
                item.speaker_bucket_rank_sha256,
                item.row_rank_sha256,
                item.speaker_id,
            ),
        )
        if len(rows) != quota_by_bucket[bucket]:
            raise WorkerError()
        ordered.extend(rows)
    return tuple(ordered)


def _collect_candidates(
    shards: Sequence[_OpenedShard],
) -> tuple[tuple[Candidate, ...], int, int, str]:
    target_rows: dict[int, dict[int, _MetadataRow]] = {0: {}, 1: {}}
    accent_speakers: set[str] = set()
    seen_audio_ids: set[str] = set()
    for shard in shards:
        shard_index = int(shard.record["shard_index"])
        for row_index, row in enumerate(_metadata_rows(shard)):
            audio_id = _text(row.get("audio_id"), maximum=512)
            speaker_id = _text(row.get("speaker_id"), maximum=512)
            raw_text = row.get("raw_text")
            normalized_text = row.get("normalized_text")
            gender = row.get("gender")
            gold = row.get("is_gold_transcript")
            accent = row.get("accent")
            language = row.get("language")
            if (
                audio_id in seen_audio_ids
                or not isinstance(raw_text, str)
                or "\x00" in raw_text
                or len(raw_text.encode("utf-8")) > _MAX_TEXT_BYTES
                or not isinstance(normalized_text, str)
                or "\x00" in normalized_text
                or len(normalized_text.encode("utf-8")) > _MAX_TEXT_BYTES
                or not isinstance(gender, str)
                or "\x00" in gender
                or len(gender.encode("utf-8")) > 128
                or type(gold) is not bool
                or not isinstance(accent, str)
                or "\x00" in accent
                or len(accent.encode("utf-8")) > 128
                or isinstance(language, bool)
                or not isinstance(language, int)
                or language != LANGUAGE_ID
            ):
                raise WorkerError()
            seen_audio_ids.add(audio_id)
            if accent == ACCENT:
                accent_speakers.add(speaker_id)
            if accent == ACCENT and gold:
                transcription = _text(normalized_text, maximum=4096)
                target_rows[shard_index][row_index] = _MetadataRow(
                    shard_index=shard_index,
                    row_index=row_index,
                    audio_id=audio_id,
                    speaker_id=speaker_id,
                    gender=gender,
                    transcription=transcription,
                )
                if sum(len(rows) for rows in target_rows.values()) > _MAX_TARGET_ROWS:
                    raise WorkerError()
    if (
        len(seen_audio_ids) != SOURCE_TOTAL_ROWS
        or len(accent_speakers) != EXPECTED_EN_RO_SPEAKERS
    ):
        raise WorkerError()

    candidates: list[Candidate] = []
    inspected_audio_bytes = 0
    for shard in shards:
        shard_index = int(shard.record["shard_index"])
        metadata_by_row = target_rows[shard_index]
        for row_index, payload, audio_path in _audio_rows(shard, set(metadata_by_row)):
            metadata = metadata_by_row[row_index]
            if not _audio_path_matches(audio_path, metadata.audio_id):
                raise WorkerError()
            inspected_audio_bytes += len(payload)
            if inspected_audio_bytes > _MAX_TARGET_AUDIO_BYTES:
                raise WorkerError()
            samples = _wav_samples(payload)
            bucket = duration_bucket(samples)
            if bucket is None:
                continue
            candidates.append(
                Candidate(
                    shard_index=shard_index,
                    row_index=row_index,
                    audio_id=metadata.audio_id,
                    speaker_id=metadata.speaker_id,
                    gender=metadata.gender,
                    transcription=metadata.transcription,
                    duration_samples=samples,
                    duration_bucket=bucket,
                    row_rank_sha256=row_rank(metadata.speaker_id, metadata.audio_id),
                    speaker_bucket_rank_sha256=speaker_bucket_rank(
                        bucket, metadata.speaker_id
                    ),
                    audio_sha256=hashlib.sha256(payload).hexdigest(),
                    audio_size_bytes=len(payload),
                )
            )
    eligible_speakers = len({candidate.speaker_id for candidate in candidates})
    if len(candidates) < SELECTION_COUNT or eligible_speakers < SELECTION_COUNT:
        raise WorkerError()
    eligible_binding = sorted(
        (
            {
                "identity_sha256": _rank("identity", item.speaker_id, item.audio_id),
                "transcription_sha256": hashlib.sha256(
                    item.transcription.encode("utf-8")
                ).hexdigest(),
                "duration_samples": item.duration_samples,
                "duration_bucket": item.duration_bucket,
                "audio_sha256": item.audio_sha256,
            }
            for item in candidates
        ),
        key=lambda item: str(item["identity_sha256"]),
    )
    eligible_sha256 = hashlib.sha256(
        _canonical_json_bytes(eligible_binding)
    ).hexdigest()
    return tuple(candidates), len(accent_speakers), eligible_speakers, eligible_sha256


def _write_selected_audio(
    shards: Sequence[_OpenedShard],
    selected: Sequence[Candidate],
    output_dir: Path,
) -> list[dict[str, object]]:
    by_shard: dict[int, dict[int, tuple[int, Candidate]]] = {0: {}, 1: {}}
    for selection_index, candidate in enumerate(selected):
        by_shard[candidate.shard_index][candidate.row_index] = (
            selection_index,
            candidate,
        )
    result_by_index: dict[int, dict[str, object]] = {}
    for shard in shards:
        shard_index = int(shard.record["shard_index"])
        requested = by_shard[shard_index]
        for row_index, payload, audio_path in _audio_rows(shard, set(requested)):
            selection_index, candidate = requested[row_index]
            if (
                not _audio_path_matches(audio_path, candidate.audio_id)
                or len(payload) != candidate.audio_size_bytes
                or hashlib.sha256(payload).hexdigest() != candidate.audio_sha256
            ):
                raise WorkerError()
            selected_samples = _wav_samples(payload)
            if selected_samples <= 0 or selected_samples != candidate.duration_samples:
                raise WorkerError()
            filename = f"case-{selection_index:02d}.wav"
            destination = output_dir / filename
            with destination.open("xb") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            destination.chmod(0o400)
            transcription_sha256 = hashlib.sha256(
                candidate.transcription.encode("utf-8")
            ).hexdigest()
            result_by_index[selection_index] = {
                "selection_index": selection_index,
                "shard_index": candidate.shard_index,
                "row_index": candidate.row_index,
                "audio_id": candidate.audio_id,
                "speaker_id": candidate.speaker_id,
                "gender": candidate.gender,
                "transcription": candidate.transcription,
                "transcription_sha256": transcription_sha256,
                "duration_samples": candidate.duration_samples,
                "duration_bucket": candidate.duration_bucket,
                "row_rank_sha256": candidate.row_rank_sha256,
                "speaker_bucket_rank_sha256": candidate.speaker_bucket_rank_sha256,
                "audio_file": filename,
                "audio_sha256": candidate.audio_sha256,
                "audio_size_bytes": candidate.audio_size_bytes,
            }
    if set(result_by_index) != set(range(SELECTION_COUNT)):
        raise WorkerError()
    return [result_by_index[index] for index in range(SELECTION_COUNT)]


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    encoded = _canonical_json_bytes(payload) + b"\n"
    if not encoded or len(encoded) > _MAX_RESULT_BYTES:
        raise WorkerError()
    temporary = path.with_name(f".{path.name}.part")
    with temporary.open("xb") as output:
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    path.chmod(0o400)


def _set_resource_limits() -> None:
    try:
        import resource
    except ImportError:  # pragma: no cover - Linux/POSIX is the reference
        return
    limits = (
        ("RLIMIT_CORE", 0),
        ("RLIMIT_CPU", 840),
        ("RLIMIT_FSIZE", 32 * 1024 * 1024),
        ("RLIMIT_NOFILE", 96),
        ("RLIMIT_AS", 3 * 1024 * 1024 * 1024),
    )
    try:
        for name, desired in limits:
            resource_id = getattr(resource, name, None)
            if resource_id is None:
                continue
            _soft, hard = resource.getrlimit(resource_id)
            capped = desired if hard == resource.RLIM_INFINITY else min(desired, hard)
            resource.setrlimit(resource_id, (capped, hard))
    except (OSError, ValueError):
        raise WorkerError() from None


def _select(request: Mapping[str, object], output_dir: Path) -> Mapping[str, object]:
    try:
        import pyarrow
        import pyarrow.parquet as pq
    except Exception as exc:  # noqa: BLE001
        raise WorkerError() from exc
    if pyarrow.__version__ != PYARROW_VERSION:
        raise WorkerError()
    records = request["source_descriptors"]
    if not isinstance(records, list):
        raise WorkerError()
    with _opened_shards(pq, records) as shards:
        candidates, accent_speakers, eligible_speakers, eligible_sha256 = (
            _collect_candidates(shards)
        )
        selected = select_candidates(candidates)
        cases = _write_selected_audio(shards, selected, output_dir)
        source_results = [
            {
                "shard_index": int(shard.record["shard_index"]),
                "filename": shard.record["filename"],
                "size_bytes": shard.record["size_bytes"],
                "sha256": shard.record["sha256"],
                "rows": shard.rows,
                "row_groups": shard.row_groups,
            }
            for shard in shards
        ]
    bucket_counts = {
        name: sum(case["duration_bucket"] == name for case in cases)
        for name, *_rest in DURATION_BUCKETS
    }
    return {
        "schema_version": PROTOCOL_VERSION,
        "pyarrow_version": PYARROW_VERSION,
        "source_revision": SOURCE_REVISION,
        "selection_seed": SELECTION_SEED,
        "schema_contract_sha256": SCHEMA_CONTRACT_SHA256,
        "source_total_rows": SOURCE_TOTAL_ROWS,
        "source_shards": source_results,
        "target_accent_speakers": accent_speakers,
        "eligible_rows": len(candidates),
        "eligible_speakers": eligible_speakers,
        "eligible_set_sha256": eligible_sha256,
        "bucket_counts": bucket_counts,
        "cases": cases,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        os.umask(0o077)
        _set_resource_limits()
        args = _parser().parse_args(argv)
        if (
            not args.request.is_file()
            or args.request.is_symlink()
            or not args.output_dir.is_dir()
            or args.output_dir.is_symlink()
            or any(args.output_dir.iterdir())
        ):
            raise WorkerError()
        request = _load_request(args.request)
        result = _select(request, args.output_dir)
        if (
            not isinstance(result, dict)
            or set(result) != _RESULT_FIELDS
            or not isinstance(result.get("source_shards"), list)
            or any(
                not isinstance(item, dict) or set(item) != _RESULT_SOURCE_FIELDS
                for item in result["source_shards"]
            )
            or not isinstance(result.get("cases"), list)
            or any(
                not isinstance(item, dict) or set(item) != _RESULT_CASE_FIELDS
                for item in result["cases"]
            )
        ):
            raise WorkerError()
        _atomic_json(args.output_dir / "result.json", result)
        return 0
    except Exception:  # noqa: BLE001 - source paths/transcripts never escape
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
