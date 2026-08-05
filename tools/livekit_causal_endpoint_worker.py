#!/usr/bin/env python3
"""Isolated private materializer for causal LiveKit endpoint evaluation.

The worker runs only under the explicitly selected PyArrow 25 interpreter. It
reuses the immutable ADR-0134 schema/audio/label validator, reads only its four
admitted columns, and writes ordinal raw PCM plus a private sample-domain
manifest. The bounded embedded-audio struct includes a flat ``audio.path``
label, which is validated but never opened or published. IDs, words, messages,
and transcript values are neither loaded nor published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import struct
from types import ModuleType
from typing import Mapping, Sequence


PROTOCOL_VERSION = 1
MATERIALIZATION_KIND = "livekit-eot-causal-pcm-v1"
GRID_SAMPLES = 1_600
_SCAN_COLUMNS = ("audio", "language", "duration", "silence_spans")
_MAX_REQUEST_BYTES = 128 * 1024
_MAX_MANIFEST_BYTES = 2 * 1024 * 1024
_MAX_PCM_BYTES = 32 * 1024 * 1024
_REQUEST_FIELDS = {
    "schema_version",
    "source_descriptor",
    "source_revision",
    "source_size_bytes",
    "source_sha256",
    "causal_worker_descriptor",
    "inventory_worker_descriptor",
    "causal_worker_sha256",
    "inventory_worker_sha256",
    "output_leaf_identities",
}


class WorkerError(RuntimeError):
    """A detail-free materialization contract failure."""


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


def _strict_json(raw: bytes) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate")
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeError, ValueError, OverflowError):
        raise WorkerError() from None


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


def _directory_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        stat.S_IFMT(info.st_mode),
        stat.S_IMODE(info.st_mode),
        info.st_uid,
    )


def _read_bound_descriptor(descriptor: int, maximum: int) -> bytes:
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > maximum
            or stat.S_IMODE(before.st_mode) & 0o077
            or (hasattr(os, "geteuid") and before.st_uid != os.geteuid())
        ):
            raise WorkerError()
        chunks = bytearray()
        while len(chunks) < before.st_size:
            chunk = os.pread(
                descriptor,
                min(1024 * 1024, before.st_size - len(chunks)),
                len(chunks),
            )
            if not chunk:
                break
            chunks.extend(chunk)
        if len(chunks) != before.st_size or _identity(os.fstat(descriptor)) != _identity(before):
            raise WorkerError()
        return bytes(chunks)
    except WorkerError:
        raise
    except OSError:
        raise WorkerError() from None


def _load_inventory_contract(descriptor: int, expected_sha256: str):
    payload = _read_bound_descriptor(descriptor, 1024 * 1024)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise WorkerError()
    try:
        module = ModuleType("speaker_livekit_inventory_contract")
        module.__file__ = "<verified-fd:inventory-worker>"
        exec(compile(payload, module.__file__, "exec"), module.__dict__)  # noqa: S102
        return module
    except WorkerError:
        raise
    except Exception:
        raise WorkerError() from None


def _load_request_descriptor(descriptor: int) -> Mapping[str, object]:
    value = _strict_json(_read_bound_descriptor(descriptor, _MAX_REQUEST_BYTES))
    if not isinstance(value, dict) or set(value) != _REQUEST_FIELDS:
        raise WorkerError()
    source_descriptor = value.get("source_descriptor")
    schema_version = value.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != PROTOCOL_VERSION
        or isinstance(source_descriptor, bool)
        or not isinstance(source_descriptor, int)
        or source_descriptor < 3
    ):
        raise WorkerError()
    for name in (
        "causal_worker_descriptor",
        "inventory_worker_descriptor",
    ):
        worker_descriptor = value.get(name)
        if (
            isinstance(worker_descriptor, bool)
            or not isinstance(worker_descriptor, int)
            or worker_descriptor < 3
        ):
            raise WorkerError()
    for name in ("source_sha256", "causal_worker_sha256", "inventory_worker_sha256"):
        digest = value.get(name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise WorkerError()
    return value


def _output_leaf_identities(
    request: Mapping[str, object],
) -> Mapping[str, tuple[int, ...]]:
    value = request.get("output_leaf_identities")
    expected_names = {
        *(f"row-{ordinal:04d}.pcm" for ordinal in range(400)),
        "manifest.json",
    }
    if not isinstance(value, dict) or set(value) != expected_names:
        raise WorkerError()
    result: dict[str, tuple[int, ...]] = {}
    for name, raw_identity in value.items():
        if (
            not isinstance(raw_identity, list)
            or len(raw_identity) != 6
            or any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in raw_identity
            )
        ):
            raise WorkerError()
        result[name] = tuple(raw_identity)
    return result


def _wav_pcm16(payload: bytes) -> bytes:
    """Return data bytes after ADR-0134 has validated the full WAV contract."""

    offset = 12
    data: bytes | None = None
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
        if chunk_id == b"data":
            if data is not None:
                raise WorkerError()
            data = payload[start:end]
        offset = padded_end
    if data is None or not data or len(data) % 2:
        raise WorkerError()
    return data


def _write_private_file(
    directory_descriptor: int,
    name: str,
    payload: bytes,
    *,
    maximum: int,
    expected_identity: Sequence[int],
) -> None:
    if not payload or len(payload) > maximum:
        raise WorkerError()
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise WorkerError()
    if (
        not isinstance(expected_identity, (list, tuple))
        or len(expected_identity) != 6
        or any(isinstance(item, bool) or not isinstance(item, int) for item in expected_identity)
    ):
        raise WorkerError()
    expected = tuple(expected_identity)
    descriptor = -1
    try:
        before = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size != 0
            or stat.S_IMODE(before.st_mode) != 0o600
            or (hasattr(os, "geteuid") and before.st_uid != os.geteuid())
            or _stable_file_identity(before) != expected
        ):
            raise WorkerError()
        flags = os.O_RDWR
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
        opened = os.fstat(descriptor)
        if (
            _identity(opened) != _identity(before)
            or _stable_file_identity(opened) != expected
        ):
            raise WorkerError()
        offset = 0
        while offset < len(payload):
            written = os.pwrite(descriptor, payload[offset:], offset)
            if written <= 0:
                raise WorkerError()
            offset += written
        os.fsync(descriptor)
        after_fd = os.fstat(descriptor)
        current = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            _identity(after_fd) != _identity(current)
            or _stable_file_identity(after_fd) != expected
            or after_fd.st_nlink != 1
            or after_fd.st_size != len(payload)
            or _read_bound_descriptor(descriptor, maximum) != payload
        ):
            raise WorkerError()
    except WorkerError:
        raise
    except OSError:
        raise WorkerError() from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _stable_file_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        stat.S_IFMT(info.st_mode),
        stat.S_IMODE(info.st_mode),
        info.st_uid,
        info.st_nlink,
    )


def _materialize_row(
    row: Mapping[str, object],
    *,
    ordinal: int,
    output_descriptor: int,
    inventory_contract: object,
    output_leaf_identities: Mapping[str, tuple[int, ...]],
) -> tuple[Mapping[str, object], Mapping[str, object], bytes]:
    """Validate and materialize one already column-bounded causal row."""

    if not isinstance(row, dict) or set(row) != set(_SCAN_COLUMNS):
        raise WorkerError()
    summary = inventory_contract._validate_row(row)
    audio = row.get("audio")
    spans = row.get("silence_spans")
    if not isinstance(audio, dict) or not isinstance(spans, list):
        raise WorkerError()
    wav = audio.get("bytes")
    if not isinstance(wav, bytes):
        raise WorkerError()
    pcm = _wav_pcm16(wav)
    if len(pcm) // 2 != int(summary["audio_samples"]):
        raise WorkerError()
    sample_spans: list[list[int]] = []
    packed_spans = bytearray()
    for raw_span in spans:
        if not isinstance(raw_span, dict):
            raise WorkerError()
        start = inventory_contract._endpoint_sample(raw_span.get("start"))
        end = inventory_contract._endpoint_sample(raw_span.get("end"))
        sample_spans.append([start, end])
        packed_spans.extend(struct.pack(">QQ", start, end))

    filename = f"row-{ordinal:04d}.pcm"
    pcm_sha256 = hashlib.sha256(pcm).hexdigest()
    _write_private_file(
        output_descriptor,
        filename,
        pcm,
        maximum=_MAX_PCM_BYTES,
        expected_identity=output_leaf_identities[filename],
    )
    return (
        {
            "ordinal": ordinal,
            "pcm_filename": filename,
            "pcm_bytes": len(pcm),
            "pcm_samples": len(pcm) // 2,
            "pcm_sha256": pcm_sha256,
            "silence_spans": sample_spans,
        },
        summary,
        bytes(packed_spans),
    )


def _materialize(
    request: Mapping[str, object],
    output_descriptor: int,
    inventory_contract: object,
) -> Mapping[str, object]:
    try:
        import pyarrow
        import pyarrow.parquet as pq
    except Exception:
        raise WorkerError() from None
    if pyarrow.__version__ != inventory_contract.PYARROW_VERSION:
        raise WorkerError()
    if tuple(inventory_contract._SCAN_COLUMNS) != _SCAN_COLUMNS:
        raise WorkerError()
    if (
        request.get("source_revision") != inventory_contract.SOURCE_REVISION
        or request.get("source_size_bytes") != inventory_contract.SOURCE_SIZE_BYTES
        or request.get("source_sha256") != inventory_contract.SOURCE_SHA256
    ):
        raise WorkerError()

    rows_out: list[dict[str, object]] = []
    output_leaf_identities = _output_leaf_identities(request)
    pcm_set = hashlib.sha256()
    materialization = hashlib.sha256()
    audio_set = hashlib.sha256()
    silence_set = hashlib.sha256()
    pcm_bytes_total = 0
    pcm_samples_total = 0
    audio_bytes_total = 0
    silence_span_count = 0
    hold_label_count = 0
    off_grid_span_count = 0
    duration_exact_sample_count = 0
    duration_microsecond_quantized_count = 0
    final_gap_zero_sample_count = 0
    final_gap_one_sample_count = 0
    inventory = hashlib.sha256()
    descriptor = int(request["source_descriptor"])
    with inventory_contract._opened_parquet(pq, descriptor) as (parquet, row_groups):
        consumed = 0
        batches = parquet.iter_batches(
            batch_size=4,
            columns=list(_SCAN_COLUMNS),
            use_threads=False,
        )
        for batch in batches:
            materialized_rows = batch.to_pylist()
            if not isinstance(materialized_rows, list) or not materialized_rows:
                raise WorkerError()
            for row in materialized_rows:
                if consumed >= inventory_contract.SOURCE_ROWS:
                    raise WorkerError()
                ordinal = consumed
                row_record, summary, packed_spans = _materialize_row(
                    row,
                    ordinal=ordinal,
                    output_descriptor=output_descriptor,
                    inventory_contract=inventory_contract,
                    output_leaf_identities=output_leaf_identities,
                )
                rows_out.append(row_record)
                consumed += 1
                pcm_bytes_total += int(row_record["pcm_bytes"])
                pcm_samples_total += int(row_record["pcm_samples"])
                audio_bytes_total += int(summary["audio_bytes"])
                row_spans = row_record["silence_spans"]
                if not isinstance(row_spans, list):
                    raise WorkerError()
                silence_span_count += len(row_spans)
                hold_label_count += len(row_spans) - 1
                off_grid_span_count += sum(
                    1 for start, end in row_spans if (end - start) % GRID_SAMPLES
                )
                duration_quantized = summary["duration_quantized"]
                if duration_quantized is True:
                    duration_microsecond_quantized_count += 1
                elif duration_quantized is False:
                    duration_exact_sample_count += 1
                else:
                    raise WorkerError()
                final_gap_samples = int(summary["final_gap_samples"])
                if final_gap_samples == 0:
                    final_gap_zero_sample_count += 1
                elif final_gap_samples == 1:
                    final_gap_one_sample_count += 1
                else:
                    raise WorkerError()
                pcm_digest = bytes.fromhex(str(row_record["pcm_sha256"]))
                pcm_set.update(pcm_digest)
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
                        int(duration_quantized),
                        final_gap_samples,
                    )
                )
                materialization.update(pcm_digest)
                materialization.update(packed_spans)
                materialization.update(
                    struct.pack(
                        ">QQQ",
                        ordinal,
                        int(row_record["pcm_bytes"]),
                        len(row_spans),
                    )
                )

    if (
        consumed != inventory_contract.SOURCE_ROWS
        or row_groups != 2
        or audio_bytes_total != 161_326_412
        or pcm_samples_total != 80_654_406
        or pcm_bytes_total != pcm_samples_total * 2
        or silence_span_count != 1_250
        or hold_label_count != 850
        or duration_exact_sample_count != 394
        or duration_microsecond_quantized_count != 6
        or final_gap_zero_sample_count != 394
        or final_gap_one_sample_count != 6
        or audio_set.hexdigest()
        != "b15ca98e781f7347fe0439d750d240ef1cb5a3ed3936753b273cc6339df1a973"
        or silence_set.hexdigest()
        != "535c246fd73a13fcd5b19e9ce3da7e2324f184c6d43afdfeea97ecec103b5800"
        or inventory.hexdigest()
        != "693ce2c21acbde3f7b3751557262c87353e06fc293688bc2e46022bcbb99eb48"
    ):
        raise WorkerError()
    return {
        "schema_version": PROTOCOL_VERSION,
        "kind": MATERIALIZATION_KIND,
        "pyarrow_version": inventory_contract.PYARROW_VERSION,
        "source_revision": inventory_contract.SOURCE_REVISION,
        "source_size_bytes": inventory_contract.SOURCE_SIZE_BYTES,
        "source_sha256": inventory_contract.SOURCE_SHA256,
        "schema_contract_sha256": inventory_contract.SCHEMA_CONTRACT_SHA256,
        "inventory_report_sha256": (
            "4c6c53d6324741006fa34da7bc6ecb02069ca68e2d5efffd166661b8b2f610e5"
        ),
        "inventory_sha256": (
            "693ce2c21acbde3f7b3751557262c87353e06fc293688bc2e46022bcbb99eb48"
        ),
        "sample_rate_hz": inventory_contract.SAMPLE_RATE_HZ,
        "pcm_contract": "signed-pcm16-little-endian-mono-16000hz",
        "label_convention": inventory_contract.LABEL_CONVENTION,
        "grid_samples": GRID_SAMPLES,
        "causal_worker_sha256": request["causal_worker_sha256"],
        "inventory_worker_sha256": request["inventory_worker_sha256"],
        "row_count": consumed,
        "row_group_count": row_groups,
        "audio_bytes_total": audio_bytes_total,
        "pcm_bytes_total": pcm_bytes_total,
        "pcm_samples_total": pcm_samples_total,
        "silence_span_count": silence_span_count,
        "hold_label_count": hold_label_count,
        "eot_label_count": consumed,
        "off_grid_span_count": off_grid_span_count,
        "duration_exact_sample_count": duration_exact_sample_count,
        "duration_microsecond_quantized_count": (
            duration_microsecond_quantized_count
        ),
        "final_gap_zero_sample_count": final_gap_zero_sample_count,
        "final_gap_one_sample_count": final_gap_one_sample_count,
        "pcm_set_sha256": pcm_set.hexdigest(),
        "materialization_sha256": materialization.hexdigest(),
        "rows": rows_out,
    }


def _set_resource_limits() -> None:
    try:
        import resource
    except ImportError:  # pragma: no cover - Linux is the production target
        raise WorkerError() from None
    limits = (
        ("RLIMIT_CORE", 0),
        ("RLIMIT_CPU", 600),
        ("RLIMIT_FSIZE", 32 * 1024 * 1024),
        ("RLIMIT_NOFILE", 64),
        ("RLIMIT_AS", 2 * 1024 * 1024 * 1024),
    )
    try:
        for name, desired in limits:
            resource_id = getattr(resource, name)
            _soft, hard = resource.getrlimit(resource_id)
            capped = desired if hard == resource.RLIM_INFINITY else min(desired, hard)
            resource.setrlimit(resource_id, (capped, hard))
    except (AttributeError, OSError, ValueError):
        raise WorkerError() from None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--request-fd", type=int, required=True)
    parser.add_argument("--output-dir-fd", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        os.umask(0o077)
        _set_resource_limits()
        args = _parser().parse_args(argv)
        if args.request_fd < 3 or args.output_dir_fd < 3:
            raise WorkerError()
        request = _load_request_descriptor(args.request_fd)
        output_leaf_identities = _output_leaf_identities(request)
        causal_descriptor = int(request["causal_worker_descriptor"])
        inventory_descriptor = int(request["inventory_worker_descriptor"])
        if hashlib.sha256(
            _read_bound_descriptor(causal_descriptor, 1024 * 1024)
        ).hexdigest() != request["causal_worker_sha256"]:
            raise WorkerError()
        inventory_contract = _load_inventory_contract(
            inventory_descriptor,
            str(request["inventory_worker_sha256"]),
        )
        output_descriptor = args.output_dir_fd
        try:
            info = os.fstat(output_descriptor)
            if (
                not stat.S_ISDIR(info.st_mode)
                or stat.S_IMODE(info.st_mode) != 0o700
                or set(os.listdir(output_descriptor)) != set(output_leaf_identities)
                or (hasattr(os, "geteuid") and info.st_uid != os.geteuid())
            ):
                raise WorkerError()
            manifest = _materialize(request, output_descriptor, inventory_contract)
            payload = _canonical_json_bytes(manifest)
            _write_private_file(
                output_descriptor,
                "manifest.json",
                payload,
                maximum=_MAX_MANIFEST_BYTES,
                expected_identity=output_leaf_identities["manifest.json"],
            )
            if _directory_identity(os.fstat(output_descriptor)) != _directory_identity(info):
                raise WorkerError()
            if hashlib.sha256(
                _read_bound_descriptor(causal_descriptor, 1024 * 1024)
            ).hexdigest() != request["causal_worker_sha256"]:
                raise WorkerError()
            if hashlib.sha256(
                _read_bound_descriptor(inventory_descriptor, 1024 * 1024)
            ).hexdigest() != request["inventory_worker_sha256"]:
                raise WorkerError()
            os.fsync(output_descriptor)
        finally:
            os.close(output_descriptor)
        return 0
    except Exception:  # noqa: BLE001 - never expose private source detail
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
