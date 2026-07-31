#!/usr/bin/env python3
"""Isolated MInDS-14 Parquet selector for public fixture preparation.

This worker is intentionally preparation-only.  It imports ``pyarrow`` from an
explicitly selected interpreter, validates the exact pinned MInDS-14 en-US
schema, and copies only the selected embedded WAV bytes into a private
directory.  It never opens the dataset's ``path`` or ``audio.path`` values.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import struct
from typing import BinaryIO, Iterator, Mapping, Sequence


_PROTOCOL_VERSION = 1
_PYARROW_VERSION = "25.0.0"
_SOURCE_SHA256 = "37004471dc896ce20771b3fdda0ee8fb33ec7a030fb4e2fd047c561ec0a1ee30"
_SOURCE_SIZE_BYTES = 34_196_221
_SOURCE_ROWS = 563
_SOURCE_ROW_GROUPS = 6
_SOURCE_ID = "minds14-en-us-train"
_SELECTION_SEED = "speaker-public-minds14-v1"
_LANGUAGE_ID = 4
_MAX_REQUEST_BYTES = 64 * 1024
_MAX_AUDIO_BYTES = 8 * 1024 * 1024
_MAX_TOTAL_AUDIO_BYTES = 32 * 1024 * 1024
_MAX_AUDIO_DATA_BYTES = 30 * 8000
_MAX_TEXT_BYTES = 4096
_INTENTS = (
    "abroad",
    "address",
    "app_error",
    "atm_limit",
    "balance",
    "business_loan",
    "card_issues",
    "cash_deposit",
    "direct_debit",
    "freeze",
    "high_value_payment",
    "joint_account",
    "latest_transactions",
    "pay_bill",
)
_MAX_TOTAL_AUDIO_DATA_BYTES = len(_INTENTS) * _MAX_AUDIO_DATA_BYTES
_LANGUAGES = (
    "cs-CZ",
    "de-DE",
    "en-AU",
    "en-GB",
    "en-US",
    "es-ES",
    "fr-FR",
    "it-IT",
    "ko-KR",
    "nl-NL",
    "pl-PL",
    "pt-PT",
    "ru-RU",
    "zh-CN",
)
_REQUEST_FIELDS = {
    "schema_version",
    "source_path",
    "source_sha256",
    "source_size_bytes",
    "selection_seed",
    "source_id",
    "cases",
}
_REQUEST_CASE_FIELDS = {"fixture_id", "intent_id", "intent_label"}
_RESULT_FIELDS = {
    "schema_version",
    "source_sha256",
    "source_size_bytes",
    "source_row_count",
    "source_row_groups",
    "pyarrow_version",
    "cases",
}
_RESULT_CASE_FIELDS = {
    "fixture_id",
    "intent_id",
    "intent_label",
    "language_id",
    "language",
    "row_index",
    "source_path",
    "transcription",
    "transcription_sha256",
    "audio_file",
    "audio_sha256",
    "audio_size_bytes",
    "selection_rank_sha256",
}


class WorkerError(RuntimeError):
    """A detail-free preparation worker failure."""


def _strict_json_loads(raw: bytes | str) -> object:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    return json.loads(
        raw,
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("constant")),
    )


def _file_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


@contextmanager
def _stable_regular_reader(
    path: Path,
    *,
    maximum: int,
    expected_size: int | None = None,
) -> Iterator[tuple[BinaryIO, os.stat_result]]:
    descriptor = -1
    handle: BinaryIO | None = None
    try:
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > maximum
            or (expected_size is not None and before.st_size != expected_size)
        ):
            raise WorkerError()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if _file_identity(opened) != _file_identity(before):
            raise WorkerError()
        handle = os.fdopen(descriptor, "rb", buffering=0)
        descriptor = -1
        try:
            yield handle, opened
        finally:
            try:
                current = path.lstat()
            except OSError as exc:
                raise WorkerError() from exc
            if _file_identity(os.fstat(handle.fileno())) != _file_identity(
                opened
            ) or _file_identity(current) != _file_identity(opened):
                raise WorkerError()
    except WorkerError:
        raise
    except OSError as exc:
        raise WorkerError() from exc
    finally:
        if handle is not None:
            handle.close()
        elif descriptor >= 0:
            os.close(descriptor)


def _read_stable_regular_bytes(path: Path, *, maximum: int) -> bytes:
    with _stable_regular_reader(path, maximum=maximum) as (source, opened):
        payload = source.read(opened.st_size + 1)
        if len(payload) != opened.st_size:
            raise WorkerError()
        return payload


def _text(value: object, *, maximum: int = _MAX_TEXT_BYTES) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise WorkerError()
    if len(value.encode("utf-8")) > maximum:
        raise WorkerError()
    return value


def _flat_name(value: object) -> str:
    text = _text(value, maximum=128)
    if Path(text).name != text or text.startswith("."):
        raise WorkerError()
    return text


def _source_audio_basename(value: object, intent_label: str) -> str:
    text = _text(value, maximum=2048)
    path = PurePosixPath(text)
    expected_directory = f"en-US~{intent_label.upper()}"
    if (
        path.is_absolute()
        or len(path.parts) != 2
        or path.parts[0] != expected_directory
        or path.parts[1].startswith(".")
        or not path.parts[1].lower().endswith(".wav")
        or text != f"{expected_directory}/{path.parts[1]}"
    ):
        raise WorkerError()
    return path.parts[1]


def _embedded_audio_basename(value: object) -> str:
    text = _flat_name(value)
    if not text.lower().endswith(".wav"):
        raise WorkerError()
    return text


def _load_request(path: Path) -> Mapping[str, object]:
    try:
        raw = _read_stable_regular_bytes(path, maximum=_MAX_REQUEST_BYTES)
        request = _strict_json_loads(raw)
    except Exception as exc:  # noqa: BLE001
        raise WorkerError() from exc
    if not isinstance(request, dict) or set(request) != _REQUEST_FIELDS:
        raise WorkerError()
    if request.get("schema_version") != _PROTOCOL_VERSION:
        raise WorkerError()
    if request.get("source_sha256") != _SOURCE_SHA256:
        raise WorkerError()
    if request.get("source_size_bytes") != _SOURCE_SIZE_BYTES:
        raise WorkerError()
    if request.get("selection_seed") != _SELECTION_SEED:
        raise WorkerError()
    if request.get("source_id") != _SOURCE_ID:
        raise WorkerError()
    cases = request.get("cases")
    if not isinstance(cases, list) or len(cases) != len(_INTENTS):
        raise WorkerError()
    seen_ids: set[str] = set()
    seen_intents: set[int] = set()
    for item in cases:
        if not isinstance(item, dict) or set(item) != _REQUEST_CASE_FIELDS:
            raise WorkerError()
        fixture_id = _flat_name(item.get("fixture_id"))
        intent_id = item.get("intent_id")
        if (
            fixture_id in seen_ids
            or isinstance(intent_id, bool)
            or not isinstance(intent_id, int)
            or not 0 <= intent_id < len(_INTENTS)
            or intent_id in seen_intents
            or item.get("intent_label") != _INTENTS[intent_id]
        ):
            raise WorkerError()
        seen_ids.add(fixture_id)
        seen_intents.add(intent_id)
    if seen_intents != set(range(len(_INTENTS))):
        raise WorkerError()
    return request


@contextmanager
def _verified_source(path: Path) -> Iterator[BinaryIO]:
    with _stable_regular_reader(
        path,
        maximum=_SOURCE_SIZE_BYTES,
        expected_size=_SOURCE_SIZE_BYTES,
    ) as (source, opened):
        digest = hashlib.sha256()
        total = 0
        while total < opened.st_size:
            chunk = source.read(min(1024 * 1024, opened.st_size - total))
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        if (
            total != _SOURCE_SIZE_BYTES
            or source.read(1)
            or digest.hexdigest() != _SOURCE_SHA256
        ):
            raise WorkerError()
        source.seek(0)
        yield source


def _feature_type(feature: object, expected: str) -> Mapping[str, object]:
    if not isinstance(feature, dict) or feature.get("_type") != expected:
        raise WorkerError()
    return feature


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
    expected_names = (
        "path",
        "audio",
        "transcription",
        "english_transcription",
        "intent_class",
        "lang_id",
    )
    if not isinstance(features, dict) or tuple(features) != expected_names:
        raise WorkerError()
    for name in ("path", "transcription", "english_transcription"):
        feature = _feature_type(features[name], "Value")
        if set(feature) != {"dtype", "_type"} or feature.get("dtype") != "string":
            raise WorkerError()
    audio = _feature_type(features["audio"], "Audio")
    if set(audio) != {"sampling_rate", "_type"} or audio.get("sampling_rate") != 8000:
        raise WorkerError()
    intent = _feature_type(features["intent_class"], "ClassLabel")
    language = _feature_type(features["lang_id"], "ClassLabel")
    if (
        set(intent) != {"names", "_type"}
        or tuple(intent.get("names") or ()) != _INTENTS
    ):
        raise WorkerError()
    if (
        set(language) != {"names", "_type"}
        or tuple(language.get("names") or ()) != _LANGUAGES
    ):
        raise WorkerError()


def _validate_arrow_schema(schema: object) -> None:
    import pyarrow as pa

    expected = pa.schema(
        [
            pa.field("path", pa.string()),
            pa.field(
                "audio",
                pa.struct(
                    [
                        pa.field("bytes", pa.binary()),
                        pa.field("path", pa.string()),
                    ]
                ),
            ),
            pa.field("transcription", pa.string()),
            pa.field("english_transcription", pa.string()),
            pa.field("intent_class", pa.int64()),
            pa.field("lang_id", pa.int64()),
        ]
    )
    if not hasattr(schema, "remove_metadata") or not schema.remove_metadata().equals(
        expected
    ):
        raise WorkerError()
    _validate_huggingface_metadata(schema.metadata)


def _validate_mulaw_wav(payload: bytes) -> int:
    if (
        len(payload) < 44
        or len(payload) > _MAX_AUDIO_BYTES
        or payload[:4] != b"RIFF"
        or payload[8:12] != b"WAVE"
    ):
        raise WorkerError()
    declared = struct.unpack_from("<I", payload, 4)[0]
    if declared + 8 != len(payload):
        raise WorkerError()
    offset = 12
    fmt: tuple[int, int, int, int] | None = None
    data_size = 0
    while offset + 8 <= len(payload):
        kind = payload[offset : offset + 4]
        size = struct.unpack_from("<I", payload, offset + 4)[0]
        start = offset + 8
        end = start + size
        if end > len(payload):
            raise WorkerError()
        if kind == b"fmt ":
            if fmt is not None or size < 16:
                raise WorkerError()
            tag, channels, sample_rate = struct.unpack_from("<HHI", payload, start)
            bits_per_sample = struct.unpack_from("<H", payload, start + 14)[0]
            fmt = (tag, channels, sample_rate, bits_per_sample)
        elif kind == b"data":
            if data_size or size <= 0:
                raise WorkerError()
            data_size = size
        offset = end + (size & 1)
    if fmt != (7, 1, 8000, 8) or data_size <= 0 or data_size > _MAX_AUDIO_DATA_BYTES:
        raise WorkerError()
    return data_size


def _rank(seed: str, source_sha256: str, intent_label: str, source_path: str) -> str:
    return hashlib.sha256(
        seed.encode("utf-8")
        + b"\0"
        + source_sha256.encode("ascii")
        + b"\0"
        + intent_label.encode("utf-8")
        + b"\0"
        + source_path.encode("utf-8")
    ).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if len(encoded) > 128 * 1024:
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
    except ImportError:  # pragma: no cover - POSIX is the reference runtime
        return

    limits = (
        ("RLIMIT_CORE", 0),
        ("RLIMIT_CPU", 55),
        ("RLIMIT_FSIZE", 40 * 1024 * 1024),
        ("RLIMIT_NOFILE", 64),
        ("RLIMIT_AS", 2 * 1024 * 1024 * 1024),
    )
    try:
        for name, desired in limits:
            limit = getattr(resource, name, None)
            if limit is None:
                continue
            _soft, hard = resource.getrlimit(limit)
            capped = desired if hard == resource.RLIM_INFINITY else min(desired, hard)
            resource.setrlimit(limit, (capped, hard))
    except (OSError, ValueError) as exc:
        raise WorkerError() from exc


def _read_verified_rows(pq, source_path: Path, columns: Sequence[str]) -> list[object]:
    with _verified_source(source_path) as source:
        parquet = pq.ParquetFile(source)
        if (
            parquet.metadata.num_rows != _SOURCE_ROWS
            or parquet.metadata.num_row_groups != _SOURCE_ROW_GROUPS
        ):
            raise WorkerError()
        _validate_arrow_schema(parquet.schema_arrow)
        rows = parquet.read(columns=list(columns), use_threads=False).to_pylist()
        del parquet
    if len(rows) != _SOURCE_ROWS:
        raise WorkerError()
    return rows


def _select(request: Mapping[str, object], output_dir: Path) -> Mapping[str, object]:
    try:
        import pyarrow
        import pyarrow.parquet as pq
    except Exception as exc:  # noqa: BLE001
        raise WorkerError() from exc
    if pyarrow.__version__ != _PYARROW_VERSION:
        raise WorkerError()

    source_path = Path(_text(request["source_path"], maximum=4096))
    if not source_path.is_absolute():
        raise WorkerError()

    columns = (
        "path",
        "audio",
        "transcription",
        "english_transcription",
        "intent_class",
        "lang_id",
    )
    rows = _read_verified_rows(pq, source_path, columns)
    seed = str(request["selection_seed"])
    candidates: dict[int, tuple[str, int, str, str]] = {}
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != set(columns):
            raise WorkerError()
        source_value = _text(row["path"], maximum=2048)
        transcription = _text(row["transcription"])
        _text(row["english_transcription"])
        intent_id = row["intent_class"]
        language_id = row["lang_id"]
        if (
            isinstance(intent_id, bool)
            or not isinstance(intent_id, int)
            or not 0 <= intent_id < len(_INTENTS)
            or language_id != _LANGUAGE_ID
        ):
            raise WorkerError()
        _source_audio_basename(source_value, _INTENTS[intent_id])
        rank = _rank(seed, _SOURCE_SHA256, _INTENTS[intent_id], source_value)
        candidate = (rank, row_index, source_value, transcription)
        current = candidates.get(intent_id)
        if (
            current is not None
            and candidate[0] == current[0]
            and candidate[2:] != current[2:]
        ):
            raise WorkerError()
        if current is None or candidate[:2] < current[:2]:
            candidates[intent_id] = candidate
    if set(candidates) != set(range(len(_INTENTS))):
        raise WorkerError()

    requested = {
        int(item["intent_id"]): item
        for item in request["cases"]
        if isinstance(item, dict)
    }
    result_cases: list[dict[str, object]] = []
    total_audio = 0
    total_audio_data = 0
    for intent_id in range(len(_INTENTS)):
        rank, row_index, source_value, transcription = candidates[intent_id]
        audio = rows[row_index]["audio"]
        if not isinstance(audio, dict) or set(audio) != {"bytes", "path"}:
            raise WorkerError()
        payload = audio["bytes"]
        if not isinstance(payload, bytes):
            raise WorkerError()
        if _embedded_audio_basename(audio["path"]) != _source_audio_basename(
            source_value, _INTENTS[intent_id]
        ):
            raise WorkerError()
        audio_data_size = _validate_mulaw_wav(payload)
        total_audio += len(payload)
        total_audio_data += audio_data_size
        if total_audio > _MAX_TOTAL_AUDIO_BYTES:
            raise WorkerError()
        if total_audio_data > _MAX_TOTAL_AUDIO_DATA_BYTES:
            raise WorkerError()
        fixture_id = str(requested[intent_id]["fixture_id"])
        filename = f"{fixture_id}.wav"
        output_path = output_dir / filename
        with output_path.open("xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        output_path.chmod(0o400)
        result_cases.append(
            {
                "fixture_id": fixture_id,
                "intent_id": intent_id,
                "intent_label": _INTENTS[intent_id],
                "language_id": _LANGUAGE_ID,
                "language": _LANGUAGES[_LANGUAGE_ID],
                "row_index": row_index,
                "source_path": source_value,
                "transcription": transcription,
                "transcription_sha256": hashlib.sha256(
                    transcription.encode("utf-8")
                ).hexdigest(),
                "audio_file": filename,
                "audio_sha256": hashlib.sha256(payload).hexdigest(),
                "audio_size_bytes": len(payload),
                "selection_rank_sha256": rank,
            }
        )
    with _verified_source(source_path):
        pass
    return {
        "schema_version": _PROTOCOL_VERSION,
        "source_sha256": _SOURCE_SHA256,
        "source_size_bytes": _SOURCE_SIZE_BYTES,
        "source_row_count": _SOURCE_ROWS,
        "source_row_groups": _SOURCE_ROW_GROUPS,
        "pyarrow_version": _PYARROW_VERSION,
        "cases": result_cases,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        os.umask(0o077)
        _set_resource_limits()
        args = _parse_args(argv)
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
        if set(result) != _RESULT_FIELDS or any(
            not isinstance(item, dict) or set(item) != _RESULT_CASE_FIELDS
            for item in result["cases"]
        ):
            raise WorkerError()
        _atomic_json(args.output_dir / "result.json", result)
        return 0
    except Exception:  # noqa: BLE001 - no source paths or transcripts on stderr
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
