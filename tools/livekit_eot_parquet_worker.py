#!/usr/bin/env python3
"""Isolated aggregate inventory worker for the pinned LiveKit English EoT shard.

The parent passes one already-open, hash-pinned Parquet descriptor.  This
preparation-only worker imports PyArrow from an explicitly selected isolated
environment, validates the publisher schema and endpoint-label invariants, and
writes one aggregate-only result.  It never reads the ``id``, ``words``, or
``messages`` columns, never follows an audio path, and never emits source paths
or conversation content.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
from typing import BinaryIO, Iterator, Mapping, Sequence


PROTOCOL_VERSION = 1
PYARROW_VERSION = "25.0.0"
SOURCE_REVISION = "ca9d98a9686b920a2d8c9eb984224ba9be74e4dd"
SOURCE_SIZE_BYTES = 162_406_142
SOURCE_SHA256 = "e475d435c8e693912243e8a810bf7e34a59e01e208551ce5b362c9c778b0fdc4"
SOURCE_ROWS = 400
SAMPLE_RATE_HZ = 16_000
LANGUAGE = "en"
LABEL_CONVENTION = "last-silence-eot-earlier-silences-hold"
DURATION_EXACT_SAMPLE_COUNT = 394
DURATION_MICROSECOND_QUANTIZED_COUNT = 6
FINAL_GAP_ZERO_SAMPLE_COUNT = 394
FINAL_GAP_ONE_SAMPLE_COUNT = 6

FEATURE_NAMES = (
    "id",
    "audio",
    "language",
    "duration",
    "silence_spans",
    "words",
    "messages",
)
SCHEMA_CONTRACT = {
    "fields": [
        ["id", "string"],
        ["audio", "struct<bytes:binary,path:string>:audio-16000"],
        ["language", "string"],
        ["duration", "float64"],
        ["silence_spans", "list<struct<start:float64,end:float64>>"],
        ["words", "list<struct<start:float64,end:float64,word:string>>"],
        ["messages", "list<struct<role:string,content:string>>"],
    ],
    "scan_columns": ["audio", "language", "duration", "silence_spans"],
    "audio": {
        "container": "riff-wave",
        "audio_format": 1,
        "encoding": "pcm-signed-16-little-endian",
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "channels": 1,
        "bits_per_sample": 16,
        "block_align_bytes": 2,
        "byte_rate": 32_000,
        "embedded_bytes_only": True,
    },
    "labels": {
        "convention": LABEL_CONVENTION,
        "endpoint_units": "exact-integer-samples-from-decimal-string",
        "minimum_any_silence_samples": 1_600,
        "minimum_final_eot_silence_samples": 3_200,
        "maximum_silence_span_samples": 80_000,
        "final_gap_samples": [0, 1],
    },
    "duration_metadata": {
        "conversion": "decimal-string-times-16000",
        "rounding": "ROUND_HALF_EVEN",
        "maximum_absolute_residual_samples": "0.008",
        "nearest_sample_must_equal_audio_samples": True,
    },
    "source_profile": {
        "duration_exact_sample_count": DURATION_EXACT_SAMPLE_COUNT,
        "duration_microsecond_quantized_count": (
            DURATION_MICROSECOND_QUANTIZED_COUNT
        ),
        "final_gap_zero_sample_count": FINAL_GAP_ZERO_SAMPLE_COUNT,
        "final_gap_one_sample_count": FINAL_GAP_ONE_SAMPLE_COUNT,
    },
}

_SCAN_COLUMNS = ("audio", "language", "duration", "silence_spans")
_MAX_REQUEST_BYTES = 32 * 1024
_MAX_RESULT_BYTES = 64 * 1024
_MAX_AUDIO_BYTES = 16 * 1024 * 1024
_MAX_AUDIO_SECONDS = 180.0
_MAX_TOTAL_AUDIO_BYTES = 512 * 1024 * 1024
_MAX_SILENCE_SPANS = 64
_MAX_ROW_GROUPS = SOURCE_ROWS
_BATCH_ROWS = 4
_SAMPLE_RATE_DECIMAL = Decimal(SAMPLE_RATE_HZ)
_MAX_DURATION_DECIMAL = Decimal("180")
_MAX_DURATION_RESIDUAL_SAMPLES = Decimal("0.008")
_REQUEST_FIELDS = {
    "schema_version",
    "source_revision",
    "source_descriptor",
    "source_size_bytes",
    "source_sha256",
}
_RESULT_FIELDS = {
    "schema_version",
    "pyarrow_version",
    "source_revision",
    "source_size_bytes",
    "source_sha256",
    "schema_contract_sha256",
    "row_count",
    "row_group_count",
    "audio_file_count",
    "audio_bytes_total",
    "audio_samples_total",
    "silence_span_count",
    "hold_label_count",
    "eot_label_count",
    "minimum_eot_silence_samples",
    "maximum_silence_span_samples",
    "duration_exact_sample_count",
    "duration_microsecond_quantized_count",
    "final_gap_zero_sample_count",
    "final_gap_one_sample_count",
    "label_convention",
    "audio_set_sha256",
    "silence_set_sha256",
    "inventory_sha256",
}


class WorkerError(RuntimeError):
    """A detail-free source, schema, audio, label, or output failure."""


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, UnicodeError, ValueError, OverflowError):
        raise WorkerError() from None


SCHEMA_CONTRACT_SHA256 = hashlib.sha256(
    _canonical_json_bytes(SCHEMA_CONTRACT)
).hexdigest()


def _strict_json_loads(raw: bytes) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    return json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("constant")),
    )


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_uid,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _read_private_file(path: Path, *, maximum: int) -> bytes:
    descriptor = -1
    handle: BinaryIO | None = None
    try:
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > maximum
            or stat.S_IMODE(before.st_mode) & 0o077
            or (hasattr(os, "geteuid") and before.st_uid != os.geteuid())
        ):
            raise WorkerError()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if _identity(opened) != _identity(before):
            raise WorkerError()
        handle = os.fdopen(descriptor, "rb", buffering=0)
        descriptor = -1
        payload = handle.read(opened.st_size + 1)
        if len(payload) != opened.st_size:
            raise WorkerError()
        after = path.lstat()
        if (
            _identity(os.fstat(handle.fileno())) != _identity(opened)
            or _identity(after) != _identity(opened)
        ):
            raise WorkerError()
        return payload
    except WorkerError:
        raise
    except OSError as exc:
        raise WorkerError() from exc
    finally:
        if handle is not None:
            handle.close()
        elif descriptor >= 0:
            os.close(descriptor)


def _load_request(path: Path) -> Mapping[str, object]:
    try:
        value = _strict_json_loads(_read_private_file(path, maximum=_MAX_REQUEST_BYTES))
    except Exception as exc:  # noqa: BLE001
        raise WorkerError() from exc
    if not isinstance(value, dict) or set(value) != _REQUEST_FIELDS:
        raise WorkerError()
    descriptor = value.get("source_descriptor")
    if (
        value.get("schema_version") != PROTOCOL_VERSION
        or value.get("source_revision") != SOURCE_REVISION
        or value.get("source_size_bytes") != SOURCE_SIZE_BYTES
        or value.get("source_sha256") != SOURCE_SHA256
        or isinstance(descriptor, bool)
        or not isinstance(descriptor, int)
        or descriptor < 3
    ):
        raise WorkerError()
    return value


def _descriptor_sha256(descriptor: int, expected_size: int) -> str:
    try:
        digest = hashlib.sha256()
        consumed = 0
        while consumed <= expected_size:
            payload = os.pread(
                descriptor,
                min(1024 * 1024, expected_size + 1 - consumed),
                consumed,
            )
            if not payload:
                break
            consumed += len(payload)
            if consumed > expected_size:
                raise WorkerError()
            digest.update(payload)
        if consumed != expected_size:
            raise WorkerError()
        return digest.hexdigest()
    except WorkerError:
        raise
    except (OSError, OverflowError, ValueError):
        raise WorkerError() from None


def _value_feature(value: object, dtype: str) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != {"dtype", "_type"}
        or value.get("dtype") != dtype
        or value.get("_type") != "Value"
    ):
        raise WorkerError()


def _sequence_feature(value: object, fields: tuple[tuple[str, str], ...]) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != {"feature", "_type"}
        or value.get("_type") != "List"
        or not isinstance(value.get("feature"), dict)
    ):
        raise WorkerError()
    item = value["feature"]
    if tuple(item) != tuple(name for name, _dtype in fields):
        raise WorkerError()
    for name, dtype in fields:
        _value_feature(item[name], dtype)


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
    _value_feature(features["id"], "string")
    audio = features["audio"]
    if (
        not isinstance(audio, dict)
        or audio.get("_type") != "Audio"
        or audio.get("sampling_rate") != SAMPLE_RATE_HZ
        or set(audio)
        - {"sampling_rate", "decode", "num_channels", "streamable", "id", "_type"}
        or audio.get("decode", True) is not True
        or audio.get("num_channels") not in {None, 1}
        or audio.get("streamable", False) is not False
        or audio.get("id") not in {None, "audio"}
    ):
        raise WorkerError()
    _value_feature(features["language"], "string")
    _value_feature(features["duration"], "float64")
    _sequence_feature(features["silence_spans"], (("start", "float64"), ("end", "float64")))
    _sequence_feature(
        features["words"],
        (("start", "float64"), ("end", "float64"), ("word", "string")),
    )
    _sequence_feature(features["messages"], (("role", "string"), ("content", "string")))


def _struct_type(value: object, names: tuple[str, ...], predicates: tuple[object, ...]) -> bool:
    try:
        import pyarrow as pa

        if not pa.types.is_struct(value) or tuple(field.name for field in value) != names:
            return False
        return all(predicate(value[index].type) for index, predicate in enumerate(predicates))
    except Exception:  # noqa: BLE001
        return False


def _list_struct_type(
    value: object,
    names: tuple[str, ...],
    predicates: tuple[object, ...],
) -> bool:
    try:
        import pyarrow as pa

        return pa.types.is_list(value) and _struct_type(value.value_type, names, predicates)
    except Exception:  # noqa: BLE001
        return False


def _validate_arrow_schema(schema: object) -> None:
    try:
        import pyarrow as pa

        if not hasattr(schema, "names") or tuple(schema.names) != FEATURE_NAMES:
            raise WorkerError()
        fields = tuple(schema.field(name) for name in FEATURE_NAMES)
        if not all(field.nullable for field in fields):
            raise WorkerError()
        by_name = {field.name: field.type for field in fields}
        scalar_ok = (
            pa.types.is_string(by_name["id"])
            and pa.types.is_string(by_name["language"])
            and pa.types.is_float64(by_name["duration"])
        )
        audio_ok = _struct_type(
            by_name["audio"],
            ("bytes", "path"),
            (pa.types.is_binary, pa.types.is_string),
        )
        spans_ok = _list_struct_type(
            by_name["silence_spans"],
            ("start", "end"),
            (pa.types.is_float64, pa.types.is_float64),
        )
        words_ok = _list_struct_type(
            by_name["words"],
            ("start", "end", "word"),
            (pa.types.is_float64, pa.types.is_float64, pa.types.is_string),
        )
        messages_ok = _list_struct_type(
            by_name["messages"],
            ("role", "content"),
            (pa.types.is_string, pa.types.is_string),
        )
        if not (scalar_ok and audio_ok and spans_ok and words_ok and messages_ok):
            raise WorkerError()
        _validate_huggingface_metadata(schema.metadata)
    except WorkerError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise WorkerError() from exc


def _wav_samples(payload: bytes) -> int:
    """Validate a bounded embedded mono 16 kHz RIFF/WAVE and return samples."""

    if (
        not isinstance(payload, bytes)
        or len(payload) < 44
        or len(payload) > _MAX_AUDIO_BYTES
        or payload[:4] != b"RIFF"
        or payload[8:12] != b"WAVE"
        or struct.unpack_from("<I", payload, 4)[0] + 8 != len(payload)
    ):
        raise WorkerError()
    offset = 12
    fmt_payload: bytes | None = None
    data_payload: bytes | None = None
    while offset < len(payload):
        if offset + 8 > len(payload):
            raise WorkerError()
        chunk_id = payload[offset : offset + 4]
        size = struct.unpack_from("<I", payload, offset + 4)[0]
        start = offset + 8
        end = start + size
        padded_end = end + (size & 1)
        if end > len(payload) or padded_end > len(payload):
            raise WorkerError()
        if chunk_id == b"fmt ":
            if fmt_payload is not None or not 16 <= size <= 64:
                raise WorkerError()
            fmt_payload = payload[start:end]
        elif chunk_id == b"data":
            if data_payload is not None or size <= 0:
                raise WorkerError()
            data_payload = payload[start:end]
        offset = padded_end
    if offset != len(payload) or fmt_payload is None or data_payload is None:
        raise WorkerError()
    audio_format, channels, sample_rate, byte_rate, block_align, bits = struct.unpack_from(
        "<HHIIHH", fmt_payload, 0
    )
    if (
        (
            audio_format,
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits,
        )
        != (1, 1, SAMPLE_RATE_HZ, 32_000, 2, 16)
        or len(data_payload) % 2
    ):
        raise WorkerError()
    samples = len(data_payload) // block_align
    if samples <= 0 or samples > int(_MAX_AUDIO_SECONDS * SAMPLE_RATE_HZ):
        raise WorkerError()
    return samples


def _decimal(value: object) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkerError()
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise WorkerError() from None
    if not result.is_finite():
        raise WorkerError()
    return result


def _endpoint_sample(value: object) -> int:
    decimal = _decimal(value)
    scaled = decimal * _SAMPLE_RATE_DECIMAL
    integral = scaled.to_integral_value()
    if decimal < 0 or scaled != integral:
        raise WorkerError()
    return int(integral)


def _duration_has_quantized_residual(value: object, audio_samples: int) -> bool:
    decimal = _decimal(value)
    if not Decimal(0) < decimal <= _MAX_DURATION_DECIMAL:
        raise WorkerError()
    scaled = decimal * _SAMPLE_RATE_DECIMAL
    nearest = scaled.to_integral_value(rounding=ROUND_HALF_EVEN)
    residual = abs(scaled - nearest)
    if int(nearest) != audio_samples or residual > _MAX_DURATION_RESIDUAL_SAMPLES:
        raise WorkerError()
    return residual != 0


def _validate_row(row: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(row, dict) or set(row) != set(_SCAN_COLUMNS):
        raise WorkerError()
    if row.get("language") != LANGUAGE:
        raise WorkerError()
    duration = row.get("duration")
    audio = row.get("audio")
    if not isinstance(audio, dict) or set(audio) != {"bytes", "path"}:
        raise WorkerError()
    payload = audio.get("bytes")
    audio_path = audio.get("path")
    if (
        not isinstance(payload, bytes)
        or not isinstance(audio_path, str)
        or not audio_path
        or audio_path in {".", ".."}
        or "\x00" in audio_path
        or "/" in audio_path
        or "\\" in audio_path
        or Path(audio_path).name != audio_path
        or len(audio_path.encode("utf-8")) > 128
    ):
        raise WorkerError()
    samples = _wav_samples(payload)
    duration_quantized = _duration_has_quantized_residual(duration, samples)
    spans = row.get("silence_spans")
    if not isinstance(spans, list) or not 1 <= len(spans) <= _MAX_SILENCE_SPANS:
        raise WorkerError()
    prior_end = -1
    packed_spans = bytearray()
    maximum_span_samples = 0
    minimum_eot_samples = 0
    for index, raw in enumerate(spans):
        if not isinstance(raw, dict) or set(raw) != {"start", "end"}:
            raise WorkerError()
        start = _endpoint_sample(raw.get("start"))
        end = _endpoint_sample(raw.get("end"))
        span_samples = end - start
        if (
            end <= start
            or start < prior_end
            or end > samples
            or span_samples < 1_600
            or span_samples > 80_000
        ):
            raise WorkerError()
        maximum_span_samples = max(maximum_span_samples, span_samples)
        if index == len(spans) - 1:
            final_gap_samples = samples - end
            if span_samples < 3_200 or final_gap_samples not in {0, 1}:
                raise WorkerError()
            minimum_eot_samples = span_samples
        packed_spans.extend(struct.pack(">QQ", start, end))
        prior_end = end
    audio_sha256 = hashlib.sha256(payload).hexdigest()
    silence_sha256 = hashlib.sha256(bytes(packed_spans)).hexdigest()
    return {
        "audio_bytes": len(payload),
        "audio_samples": samples,
        "silence_spans": len(spans),
        "hold_labels": len(spans) - 1,
        "eot_labels": 1,
        "minimum_eot_silence_samples": minimum_eot_samples,
        "maximum_silence_span_samples": maximum_span_samples,
        "duration_quantized": duration_quantized,
        "final_gap_samples": final_gap_samples,
        "audio_sha256": audio_sha256,
        "silence_sha256": silence_sha256,
    }


@contextmanager
def _opened_parquet(pq: object, descriptor: int) -> Iterator[tuple[object, int]]:
    handle: BinaryIO | None = None
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size != SOURCE_SIZE_BYTES
            or stat.S_IMODE(metadata.st_mode) & 0o077
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
            or _descriptor_sha256(descriptor, SOURCE_SIZE_BYTES) != SOURCE_SHA256
        ):
            raise WorkerError()
        identity = _identity(metadata)
        handle = os.fdopen(os.dup(descriptor), "rb", buffering=0)
        parquet = pq.ParquetFile(handle)
        _validate_arrow_schema(parquet.schema_arrow)
        rows = parquet.metadata.num_rows
        row_groups = parquet.metadata.num_row_groups
        if (
            isinstance(rows, bool)
            or not isinstance(rows, int)
            or rows != SOURCE_ROWS
            or isinstance(row_groups, bool)
            or not isinstance(row_groups, int)
            or not 1 <= row_groups <= _MAX_ROW_GROUPS
        ):
            raise WorkerError()
        yield parquet, row_groups
        if _identity(os.fstat(descriptor)) != identity:
            raise WorkerError()
    except WorkerError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise WorkerError() from exc
    finally:
        if handle is not None:
            handle.close()


def _inventory(request: Mapping[str, object]) -> Mapping[str, object]:
    try:
        import pyarrow
        import pyarrow.parquet as pq
    except Exception as exc:  # noqa: BLE001
        raise WorkerError() from exc
    if pyarrow.__version__ != PYARROW_VERSION:
        raise WorkerError()
    descriptor = int(request["source_descriptor"])
    totals = {
        "audio_file_count": 0,
        "audio_bytes_total": 0,
        "audio_samples_total": 0,
        "silence_span_count": 0,
        "hold_label_count": 0,
        "eot_label_count": 0,
        "duration_exact_sample_count": 0,
        "duration_microsecond_quantized_count": 0,
        "final_gap_zero_sample_count": 0,
        "final_gap_one_sample_count": 0,
    }
    minimum_eot: int | None = None
    maximum_silence = 0
    audio_set = hashlib.sha256()
    silence_set = hashlib.sha256()
    inventory = hashlib.sha256()
    with _opened_parquet(pq, descriptor) as (parquet, row_groups):
        consumed = 0
        batches = parquet.iter_batches(
            batch_size=_BATCH_ROWS,
            columns=list(_SCAN_COLUMNS),
            use_threads=False,
        )
        for batch in batches:
            rows = batch.to_pylist()
            if not isinstance(rows, list) or not rows:
                raise WorkerError()
            for row in rows:
                summary = _validate_row(row)
                consumed += 1
                if consumed > SOURCE_ROWS:
                    raise WorkerError()
                totals["audio_file_count"] += 1
                totals["audio_bytes_total"] += int(summary["audio_bytes"])
                totals["audio_samples_total"] += int(summary["audio_samples"])
                totals["silence_span_count"] += int(summary["silence_spans"])
                totals["hold_label_count"] += int(summary["hold_labels"])
                totals["eot_label_count"] += int(summary["eot_labels"])
                if summary["duration_quantized"] is True:
                    totals["duration_microsecond_quantized_count"] += 1
                elif summary["duration_quantized"] is False:
                    totals["duration_exact_sample_count"] += 1
                else:
                    raise WorkerError()
                final_gap_samples = int(summary["final_gap_samples"])
                if final_gap_samples == 0:
                    totals["final_gap_zero_sample_count"] += 1
                elif final_gap_samples == 1:
                    totals["final_gap_one_sample_count"] += 1
                else:
                    raise WorkerError()
                if totals["audio_bytes_total"] > _MAX_TOTAL_AUDIO_BYTES:
                    raise WorkerError()
                eot_samples = int(summary["minimum_eot_silence_samples"])
                minimum_eot = (
                    eot_samples if minimum_eot is None else min(minimum_eot, eot_samples)
                )
                maximum_silence = max(
                    maximum_silence,
                    int(summary["maximum_silence_span_samples"]),
                )
                audio_digest = bytes.fromhex(str(summary["audio_sha256"]))
                silence_digest = bytes.fromhex(str(summary["silence_sha256"]))
                audio_set.update(audio_digest)
                silence_set.update(silence_digest)
                inventory.update(audio_digest)
                inventory.update(silence_digest)
                inventory.update(
                    struct.pack(
                        ">QQQBB",
                        int(summary["audio_bytes"]),
                        int(summary["audio_samples"]),
                        int(summary["silence_spans"]),
                        int(summary["duration_quantized"]),
                        final_gap_samples,
                    )
                )
        if consumed != SOURCE_ROWS:
            raise WorkerError()
    if (
        minimum_eot is None
        or totals["audio_file_count"] != SOURCE_ROWS
        or totals["eot_label_count"] != SOURCE_ROWS
        or totals["hold_label_count"] != totals["silence_span_count"] - SOURCE_ROWS
        or totals["duration_exact_sample_count"] != DURATION_EXACT_SAMPLE_COUNT
        or totals["duration_microsecond_quantized_count"]
        != DURATION_MICROSECOND_QUANTIZED_COUNT
        or totals["final_gap_zero_sample_count"] != FINAL_GAP_ZERO_SAMPLE_COUNT
        or totals["final_gap_one_sample_count"] != FINAL_GAP_ONE_SAMPLE_COUNT
    ):
        raise WorkerError()
    return {
        "schema_version": PROTOCOL_VERSION,
        "pyarrow_version": PYARROW_VERSION,
        "source_revision": SOURCE_REVISION,
        "source_size_bytes": SOURCE_SIZE_BYTES,
        "source_sha256": SOURCE_SHA256,
        "schema_contract_sha256": SCHEMA_CONTRACT_SHA256,
        "row_count": SOURCE_ROWS,
        "row_group_count": row_groups,
        **totals,
        "minimum_eot_silence_samples": minimum_eot,
        "maximum_silence_span_samples": maximum_silence,
        "label_convention": LABEL_CONVENTION,
        "audio_set_sha256": audio_set.hexdigest(),
        "silence_set_sha256": silence_set.hexdigest(),
        "inventory_sha256": inventory.hexdigest(),
    }


def _atomic_result(path: Path, value: Mapping[str, object]) -> None:
    payload = _canonical_json_bytes(value)
    if not payload or len(payload) > _MAX_RESULT_BYTES:
        raise WorkerError()
    temporary = path.with_name(f".{path.name}.part")
    with temporary.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    path.chmod(0o400)


def _set_resource_limits() -> None:
    try:
        import resource
    except ImportError:  # pragma: no cover - POSIX is the production path
        return
    limits = (
        ("RLIMIT_CORE", 0),
        ("RLIMIT_CPU", 600),
        ("RLIMIT_FSIZE", 1024 * 1024),
        ("RLIMIT_NOFILE", 64),
        ("RLIMIT_AS", 2 * 1024 * 1024 * 1024),
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
        result = _inventory(_load_request(args.request))
        if not isinstance(result, dict) or set(result) != _RESULT_FIELDS:
            raise WorkerError()
        _atomic_result(args.output_dir / "result.json", result)
        return 0
    except Exception:  # noqa: BLE001 - never expose source/content details
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
