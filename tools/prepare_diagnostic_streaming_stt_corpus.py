"""Publish labeled exact final-selection inputs as a private STT corpus.

The synchronized diagnostic bundle owns the authoritative f32le spool and
transcript-free receipts. A separate private label file supplies references and
binds both the bundle manifest and every selected slice digest. This tool never
infers turn audio from the continuous PCM16 tracks and never prints transcript
or path details.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Mapping, Sequence

import numpy as np

from core.diagnostic_bundle import FinalModelInputReceipt, validate_manifest
from core.wer import normalize
from tools.streaming_stt.bounded_io import (
    BoundedReadError,
    opened_directory_nofollow,
    read_regular_bounded,
)
from tools.streaming_stt.corpus import (
    CorpusProvenance,
    LoadedCorpus,
    verify_corpus_snapshot,
)
from tools.streaming_stt.corpus_writer import (
    CorpusWriteCase,
    CorpusWriterError,
    publish_private_corpus,
)
from tools.streaming_stt.private_diagnostic_receipt import (
    PrivateDiagnosticCaseSurface,
    PrivateDiagnosticReceiptError,
    create_private_diagnostic_receipt,
)
from tools.streaming_stt.protocol import MAX_CORPUS_BYTES, MAX_PCM_BYTES


_LABEL_SCHEMA_VERSION = 1
_MAX_LABEL_BYTES = 256 * 1024
_MAX_MANIFEST_BYTES = 1 << 20
_MAX_TIMELINE_BYTES = 64 << 20
_MAX_TIMELINE_RECORDS = 131_072
_MAX_CASES = 512
_MAX_REFERENCE_CHARS = 4096
_MAX_TAGS = 32
_MAX_TAG_CHARS = 64
_TIMELINE_LINE_MAX_BYTES = 1 << 16
_SAFE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ERROR = {
    "ok": False,
    "error": "diagnostic_streaming_corpus_prerequisites_unavailable",
}


class DiagnosticCorpusPreparationError(RuntimeError):
    """A detail-free invalid label, evidence, or destination failure."""


class _PrivateArgumentParser(argparse.ArgumentParser):
    """Convert argument failures into the tool's detail-free error result."""

    def error(self, message: str) -> None:
        del message
        raise DiagnosticCorpusPreparationError()


@dataclass(frozen=True, slots=True)
class _LabelCase:
    case_id: str
    input_index: int
    input_sha256: str
    expected_text: str = field(repr=False)
    tags: tuple[str, ...]


def _bad() -> object:
    raise DiagnosticCorpusPreparationError()


def _strict_json(raw: bytes) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise DiagnosticCorpusPreparationError()
            result[key] = value
        return result

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: _bad())
    except (
        UnicodeError,
        ValueError,
        OverflowError,
        DiagnosticCorpusPreparationError,
    ):
        raise DiagnosticCorpusPreparationError() from None


def _identity(
    metadata: os.stat_result,
) -> tuple[int, int, int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_nlink,
        getattr(metadata, "st_uid", -1),
    )


def _private_snapshot(path: Path | str, *, maximum_bytes: int) -> tuple[Path, bytes]:
    try:
        snapshot = read_regular_bounded(Path(path).expanduser(), maximum_bytes=maximum_bytes)
        metadata = snapshot.path.stat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
        ):
            raise DiagnosticCorpusPreparationError()
        with opened_directory_nofollow(
            snapshot.path.parent,
            require_private=True,
        ):
            pass
        return snapshot.path, snapshot.data
    except (
        OSError,
        BoundedReadError,
        DiagnosticCorpusPreparationError,
    ):
        raise DiagnosticCorpusPreparationError() from None


def _sha256(value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DiagnosticCorpusPreparationError()
    return value


def _safe_id(value: object) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise DiagnosticCorpusPreparationError()
    return value


def _labels(raw: bytes) -> tuple[str, tuple[_LabelCase, ...]]:
    payload = _strict_json(raw)
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "diagnostic_manifest_sha256",
        "cases",
    }:
        raise DiagnosticCorpusPreparationError()
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != _LABEL_SCHEMA_VERSION
    ):
        raise DiagnosticCorpusPreparationError()
    manifest_sha256 = _sha256(payload.get("diagnostic_manifest_sha256"))
    rows = payload.get("cases")
    if not isinstance(rows, list) or not rows or len(rows) > _MAX_CASES:
        raise DiagnosticCorpusPreparationError()
    result: list[_LabelCase] = []
    seen_ids: set[str] = set()
    seen_indexes: set[int] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "id",
            "input_index",
            "input_sha256",
            "expected_text",
            "tags",
        }:
            raise DiagnosticCorpusPreparationError()
        case_id = _safe_id(row.get("id"))
        input_index = row.get("input_index")
        expected_text = row.get("expected_text")
        raw_tags = row.get("tags")
        if (
            case_id in seen_ids
            or type(input_index) is not int
            or input_index < 0
            or input_index in seen_indexes
            or not isinstance(expected_text, str)
            or not expected_text.strip()
            or len(expected_text) > _MAX_REFERENCE_CHARS
            or not normalize(expected_text)
            or not isinstance(raw_tags, list)
            or len(raw_tags) > _MAX_TAGS
        ):
            raise DiagnosticCorpusPreparationError()
        tags: list[str] = []
        for tag in raw_tags:
            if (
                not isinstance(tag, str)
                or len(tag) > _MAX_TAG_CHARS
                or _SAFE_ID_RE.fullmatch(tag) is None
                or tag in tags
            ):
                raise DiagnosticCorpusPreparationError()
            tags.append(tag)
        seen_ids.add(case_id)
        seen_indexes.add(input_index)
        result.append(
            _LabelCase(
                case_id=case_id,
                input_index=input_index,
                input_sha256=_sha256(row.get("input_sha256")),
                expected_text=expected_text,
                tags=tuple(tags),
            )
        )
    return manifest_sha256, tuple(sorted(result, key=lambda item: item.input_index))


def _artifact_path(parent: Path, entry: object) -> tuple[Path, str, int]:
    if not isinstance(entry, dict):
        raise DiagnosticCorpusPreparationError()
    filename = entry.get("file")
    digest = entry.get("sha256")
    size = entry.get("bytes")
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
        or filename in {".", ".."}
        or not isinstance(digest, str)
        or _SHA256_RE.fullmatch(digest) is None
        or type(size) is not int
        or size < 0
    ):
        raise DiagnosticCorpusPreparationError()
    return parent / filename, digest, size


def _timeline_receipts(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> tuple[FinalModelInputReceipt, ...]:
    if expected_bytes <= 0 or expected_bytes > _MAX_TIMELINE_BYTES:
        raise DiagnosticCorpusPreparationError()
    descriptor = -1
    try:
        before = path.lstat()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (
            _identity(opened) != _identity(before)
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o600
            or (
                hasattr(os, "geteuid")
                and getattr(opened, "st_uid", -1) != os.geteuid()
            )
            or opened.st_size != expected_bytes
        ):
            raise DiagnosticCorpusPreparationError()
        digest = hashlib.sha256()
        total = 0
        receipts: list[FinalModelInputReceipt] = []
        records = 0
        with os.fdopen(descriptor, "rb", buffering=0) as handle:
            descriptor = -1
            while True:
                line = handle.readline(_TIMELINE_LINE_MAX_BYTES + 1)
                if not line:
                    break
                total += len(line)
                records += 1
                if (
                    total > expected_bytes
                    or records > _MAX_TIMELINE_RECORDS
                    or not line.endswith(b"\n")
                    or len(line) > _TIMELINE_LINE_MAX_BYTES
                ):
                    raise DiagnosticCorpusPreparationError()
                digest.update(line)
                record = _strict_json(line)
                if not isinstance(record, dict):
                    raise DiagnosticCorpusPreparationError()
                if record.get("kind") == "final_model_input":
                    try:
                        receipts.append(FinalModelInputReceipt.from_payload(record))
                    except (TypeError, ValueError):
                        raise DiagnosticCorpusPreparationError() from None
            after = os.fstat(handle.fileno())
        current = path.lstat()
        if (
            total != expected_bytes
            or digest.hexdigest() != expected_sha256
            or _identity(after) != _identity(opened)
            or _identity(current) != _identity(opened)
        ):
            raise DiagnosticCorpusPreparationError()
        return tuple(receipts)
    except (OSError, DiagnosticCorpusPreparationError):
        raise DiagnosticCorpusPreparationError() from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _selected_audio(
    spool_path: Path,
    *,
    expected_bytes: int,
    receipts: Mapping[int, FinalModelInputReceipt],
    labels: tuple[_LabelCase, ...],
) -> dict[int, bytes]:
    selected: list[tuple[_LabelCase, FinalModelInputReceipt]] = []
    total = 0
    for label in labels:
        receipt = receipts.get(label.input_index)
        if (
            receipt is None
            or receipt.sha256 != label.input_sha256
            or receipt.sample_rate_hz != 16_000
            or receipt.bytes <= 0
            or receipt.bytes > MAX_PCM_BYTES
        ):
            raise DiagnosticCorpusPreparationError()
        total += receipt.bytes
        if total > MAX_CORPUS_BYTES:
            raise DiagnosticCorpusPreparationError()
        selected.append((label, receipt))

    descriptor = -1
    try:
        before = spool_path.lstat()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(spool_path, flags)
        opened = os.fstat(descriptor)
        if (
            _identity(opened) != _identity(before)
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o600
            or (
                hasattr(os, "geteuid")
                and getattr(opened, "st_uid", -1) != os.geteuid()
            )
            or opened.st_size != expected_bytes
        ):
            raise DiagnosticCorpusPreparationError()
        result: dict[int, bytes] = {}
        for label, receipt in selected:
            os.lseek(descriptor, receipt.byte_start, os.SEEK_SET)
            chunks: list[bytes] = []
            remaining = receipt.bytes
            while remaining:
                chunk = os.read(descriptor, min(1 << 20, remaining))
                if not chunk:
                    raise DiagnosticCorpusPreparationError()
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            if (
                len(raw) != receipt.bytes
                or hashlib.sha256(raw).hexdigest() != receipt.sha256
                or not bool(np.all(np.isfinite(np.frombuffer(raw, dtype="<f4"))))
            ):
                raise DiagnosticCorpusPreparationError()
            result[label.input_index] = raw
        after = os.fstat(descriptor)
        current = spool_path.lstat()
        if _identity(after) != _identity(opened) or _identity(current) != _identity(
            opened
        ):
            raise DiagnosticCorpusPreparationError()
        return result
    except (OSError, ValueError, MemoryError, DiagnosticCorpusPreparationError):
        raise DiagnosticCorpusPreparationError() from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def prepare_diagnostic_corpus(
    *,
    diagnostic_manifest: Path | str,
    labels: Path | str,
    output_dir: Path | str,
) -> LoadedCorpus:
    """Validate one exact-input bundle and publish selected labeled slices."""

    manifest_path = Path(diagnostic_manifest).expanduser()
    if not validate_manifest(manifest_path):
        raise DiagnosticCorpusPreparationError()
    manifest_path, manifest_raw = _private_snapshot(
        manifest_path,
        maximum_bytes=_MAX_MANIFEST_BYTES,
    )
    manifest_digest = hashlib.sha256(manifest_raw).hexdigest()
    manifest = _strict_json(manifest_raw)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 2:
        raise DiagnosticCorpusPreparationError()

    labels_path, labels_raw = _private_snapshot(labels, maximum_bytes=_MAX_LABEL_BYTES)
    bound_manifest_digest, label_cases = _labels(labels_raw)
    if bound_manifest_digest != manifest_digest:
        raise DiagnosticCorpusPreparationError()
    labels_digest = hashlib.sha256(labels_raw).hexdigest()

    timeline_path, timeline_digest, timeline_bytes = _artifact_path(
        manifest_path.parent,
        manifest.get("timeline"),
    )
    spool_entry = manifest.get("final_model_input")
    spool_path, _spool_digest, spool_bytes = _artifact_path(
        manifest_path.parent,
        spool_entry,
    )
    if (
        not isinstance(spool_entry, dict)
        or spool_entry.get("encoding") != "f32le"
        or type(spool_entry.get("input_count")) is not int
    ):
        raise DiagnosticCorpusPreparationError()
    all_receipts = _timeline_receipts(
        timeline_path,
        expected_sha256=timeline_digest,
        expected_bytes=timeline_bytes,
    )
    if len(all_receipts) != spool_entry.get("input_count"):
        raise DiagnosticCorpusPreparationError()
    receipts = {receipt.input_index: receipt for receipt in all_receipts}
    if len(receipts) != len(all_receipts):
        raise DiagnosticCorpusPreparationError()
    audio = _selected_audio(
        spool_path,
        expected_bytes=spool_bytes,
        receipts=receipts,
        labels=label_cases,
    )

    selected_receipts = tuple(
        receipts[label.input_index] for label in label_cases
    )
    published_cases = tuple(
        CorpusWriteCase(
            case_id=label.case_id,
            audio_bytes=audio[label.input_index],
            reference=label.expected_text,
            tags=tuple(
                dict.fromkeys(
                    (
                        "private-diagnostic",
                        receipts[label.input_index].role.value,
                        *label.tags,
                    )
                )
            ),
        )
        for label in label_cases
    )
    try:
        receipt_payload = create_private_diagnostic_receipt(
            diagnostic_manifest_sha256=manifest_digest,
            labels_sha256=labels_digest,
            cases=tuple(
                PrivateDiagnosticCaseSurface(
                    case_id=case.case_id,
                    filename=f"{case.case_id}.f32le",
                    sha256=hashlib.sha256(case.audio_bytes).hexdigest(),
                    samples=len(case.audio_bytes) // 4,
                    expected_text=case.reference,
                    assertion=case.assertion,
                    commands=case.commands,
                    tags=case.tags,
                )
                for case in published_cases
            ),
            selected_inputs=selected_receipts,
        )
    except PrivateDiagnosticReceiptError:
        raise DiagnosticCorpusPreparationError() from None
    source_set_digest = hashlib.sha256(receipt_payload).hexdigest()
    provenance = CorpusProvenance(
        kind="private-diagnostic-v1",
        suite="final-model-input",
        manifest_sha256=manifest_digest,
        metadata_sha256=labels_digest,
        source_set_sha256=source_set_digest,
    )
    try:
        loaded = publish_private_corpus(
            cases=published_cases,
            provenance=provenance,
            output_dir=output_dir,
            purpose="exact private diagnostic final-selection input v1",
            sidecars={"preparation-receipt.json": receipt_payload},
        )
    except (
        KeyError,
        ValueError,
        CorpusWriterError,
        PrivateDiagnosticReceiptError,
    ):
        raise DiagnosticCorpusPreparationError() from None

    if (
        loaded.schema_version != 3
        or loaded.provenance != provenance
        or not validate_manifest(manifest_path)
    ):
        raise DiagnosticCorpusPreparationError()
    manifest_after_path, manifest_after = _private_snapshot(
        manifest_path,
        maximum_bytes=_MAX_MANIFEST_BYTES,
    )
    labels_after_path, labels_after = _private_snapshot(
        labels_path,
        maximum_bytes=_MAX_LABEL_BYTES,
    )
    if (
        manifest_after_path != manifest_path
        or labels_after_path != labels_path
        or hashlib.sha256(manifest_after).hexdigest() != manifest_digest
        or hashlib.sha256(labels_after).hexdigest() != labels_digest
    ):
        raise DiagnosticCorpusPreparationError()
    try:
        verify_corpus_snapshot(loaded)
    except Exception:  # noqa: BLE001 - preserve the detail-free API boundary
        raise DiagnosticCorpusPreparationError() from None
    return loaded


def _safe_result(corpus: LoadedCorpus) -> dict[str, object]:
    if corpus.schema_version != 3 or corpus.provenance is None:
        raise DiagnosticCorpusPreparationError()
    return {
        "ok": True,
        "corpus_sha256": corpus.digest,
        "cases": len(corpus.cases),
        "audio_bytes": corpus.audio_bytes,
        "purpose_sha256": hashlib.sha256(corpus.purpose.encode("utf-8")).hexdigest(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = _PrivateArgumentParser(
        description=(
            "Convert exact synchronized-diagnostic final inputs plus a private "
            "label file into one immutable private streaming-STT corpus."
        )
    )
    parser.add_argument("--diagnostic-manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        corpus = prepare_diagnostic_corpus(
            diagnostic_manifest=args.diagnostic_manifest,
            labels=args.labels,
            output_dir=args.output_dir,
        )
        result: Mapping[str, object] = _safe_result(corpus)
        code = 0
    except Exception:  # noqa: BLE001 - paths and transcript rows stay private
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
