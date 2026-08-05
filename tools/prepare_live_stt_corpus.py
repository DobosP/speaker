"""Bind an owner-written reference plan to every exact live STT input.

The synchronized diagnostic manifest and its transcript-free receipts define
the audio order.  The separate private reference plan defines truth.  This
tool joins those two inputs without consulting ASR observations, publishes the
existing private label schema, and delegates corpus creation to the existing
exact-input exporter.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
from typing import Mapping, Sequence

from core.wer import normalize
from tools.prepare_diagnostic_streaming_stt_corpus import (
    DiagnosticCorpusPreparationError,
    DiagnosticInputInventory,
    _MAX_CASES,
    _MAX_LABEL_BYTES,
    _MAX_REFERENCE_CHARS,
    _MAX_TAG_CHARS,
    _MAX_TAGS,
    _SAFE_ID_RE,
    _labels,
    _private_snapshot,
    _safe_result,
    _strict_json,
    load_diagnostic_input_inventory,
    prepare_diagnostic_corpus,
)
from tools.streaming_stt.bounded_io import BoundedReadError, opened_directory_nofollow
from tools.streaming_stt.corpus import LoadedCorpus
from tools.streaming_stt.protocol import MAX_CORPUS_BYTES, MAX_PCM_BYTES
from tools.tool_route_gate import EXPECTED_ROUTES


_REFERENCE_PLAN_SCHEMA_VERSION = 1
_MAX_REFERENCE_PLAN_BYTES = 256 * 1024
_MAX_OWNER_TAGS = _MAX_TAGS - 2
_EXPORTER_OWNED_TAGS = frozenset(
    {"private-diagnostic", "model_gate_segment", "selected_asr_segment"}
)
_EXPECTED_ROUTE_TAG_PREFIX = "expected-tool."
_EXPECTED_ROUTE_SET = frozenset(EXPECTED_ROUTES)
_SAFE_ERROR = {
    "ok": False,
    "error": "live_stt_corpus_prerequisites_unavailable",
}


class LiveSttCorpusPreparationError(RuntimeError):
    """A detail-free invalid plan, evidence, or destination failure."""


class _PrivateArgumentParser(argparse.ArgumentParser):
    """Keep private arguments out of parser diagnostics."""

    def error(self, message: str) -> None:
        del message
        raise LiveSttCorpusPreparationError()


@dataclass(frozen=True, slots=True)
class _ReferenceCase:
    case_id: str
    expected_text: str = field(repr=False)
    tags: tuple[str, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class _PublishedLabelWitness:
    path: Path = field(repr=False)
    sha256: str = field(repr=False)
    inode: tuple[int, int] = field(repr=False)


def _reference_plan(raw: bytes) -> tuple[_ReferenceCase, ...]:
    try:
        payload = _strict_json(raw)
    except DiagnosticCorpusPreparationError:
        raise LiveSttCorpusPreparationError() from None
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "cases",
    }:
        raise LiveSttCorpusPreparationError()
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != _REFERENCE_PLAN_SCHEMA_VERSION
    ):
        raise LiveSttCorpusPreparationError()
    rows = payload.get("cases")
    if not isinstance(rows, list) or not rows or len(rows) > _MAX_CASES:
        raise LiveSttCorpusPreparationError()

    result: list[_ReferenceCase] = []
    seen_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "id",
            "expected_text",
            "tags",
        }:
            raise LiveSttCorpusPreparationError()
        case_id = row.get("id")
        expected_text = row.get("expected_text")
        raw_tags = row.get("tags")
        if (
            not isinstance(case_id, str)
            or _SAFE_ID_RE.fullmatch(case_id) is None
            or case_id in seen_ids
            or not isinstance(expected_text, str)
            or not expected_text.strip()
            or len(expected_text) > _MAX_REFERENCE_CHARS
            or not normalize(expected_text)
            or not isinstance(raw_tags, list)
            or len(raw_tags) > _MAX_OWNER_TAGS
        ):
            raise LiveSttCorpusPreparationError()
        tags: list[str] = []
        for tag in raw_tags:
            if (
                not isinstance(tag, str)
                or _SAFE_ID_RE.fullmatch(tag) is None
                or len(tag) > _MAX_TAG_CHARS
                or tag in tags
                or tag in _EXPORTER_OWNED_TAGS
            ):
                raise LiveSttCorpusPreparationError()
            tags.append(tag)
        route_tags = tuple(
            tag for tag in tags if tag.startswith(_EXPECTED_ROUTE_TAG_PREFIX)
        )
        if len(route_tags) > 1 or (
            route_tags
            and route_tags[0][len(_EXPECTED_ROUTE_TAG_PREFIX) :]
            not in _EXPECTED_ROUTE_SET
        ):
            raise LiveSttCorpusPreparationError()
        seen_ids.add(case_id)
        result.append(
            _ReferenceCase(
                case_id=case_id,
                expected_text=expected_text,
                tags=tuple(tags),
            )
        )
    return tuple(result)


def _validate_inventory(inventory: DiagnosticInputInventory) -> None:
    receipts = inventory.receipts
    if not receipts or len(receipts) > _MAX_CASES:
        raise LiveSttCorpusPreparationError()
    total_bytes = 0
    for expected_index, receipt in enumerate(receipts):
        if (
            receipt.input_index != expected_index
            or receipt.sample_rate_hz != 16_000
            or receipt.bytes <= 0
            or receipt.bytes > MAX_PCM_BYTES
        ):
            raise LiveSttCorpusPreparationError()
        total_bytes += receipt.bytes
        if total_bytes > MAX_CORPUS_BYTES:
            raise LiveSttCorpusPreparationError()
    if inventory.spool_bytes != total_bytes:
        raise LiveSttCorpusPreparationError()


def _label_payload(
    plan: tuple[_ReferenceCase, ...],
    inventory: DiagnosticInputInventory,
) -> bytes:
    if len(plan) != len(inventory.receipts):
        raise LiveSttCorpusPreparationError()
    try:
        payload = (
            json.dumps(
                {
                    "schema_version": 1,
                    "diagnostic_manifest_sha256": inventory.manifest_sha256,
                    "cases": [
                        {
                            "id": case.case_id,
                            "input_index": receipt.input_index,
                            "input_sha256": receipt.sha256,
                            "expected_text": case.expected_text,
                            "tags": list(case.tags),
                        }
                        for case, receipt in zip(
                            plan,
                            inventory.receipts,
                            strict=True,
                        )
                    ],
                },
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
    except (MemoryError, TypeError, UnicodeError, ValueError, OverflowError):
        raise LiveSttCorpusPreparationError() from None
    if not payload or len(payload) > _MAX_LABEL_BYTES:
        raise LiveSttCorpusPreparationError()
    try:
        bound_manifest, parsed = _labels(payload)
    except DiagnosticCorpusPreparationError:
        raise LiveSttCorpusPreparationError() from None
    if (
        bound_manifest != inventory.manifest_sha256
        or len(parsed) != len(plan)
        or any(
            label.case_id != case.case_id
            or label.expected_text != case.expected_text
            or label.tags != case.tags
            or label.input_index != receipt.input_index
            or label.input_sha256 != receipt.sha256
            for label, case, receipt in zip(
                parsed,
                plan,
                inventory.receipts,
                strict=True,
            )
        )
    ):
        raise LiveSttCorpusPreparationError()
    return payload


def _destination(value: Path | str) -> Path:
    try:
        candidate = Path(os.path.abspath(Path(value).expanduser()))
    except (OSError, RuntimeError, TypeError, ValueError):
        raise LiveSttCorpusPreparationError() from None
    if not candidate.name or candidate.name in {".", ".."}:
        raise LiveSttCorpusPreparationError()
    return candidate


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _require_absent(path: Path, *, private_parent: bool) -> None:
    try:
        with opened_directory_nofollow(
            path.parent,
            require_private=private_parent,
        ) as (stable_parent, parent_fd):
            if stable_parent != path.parent:
                raise LiveSttCorpusPreparationError()
            try:
                os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                return
            raise LiveSttCorpusPreparationError()
    except (OSError, BoundedReadError, LiveSttCorpusPreparationError):
        raise LiveSttCorpusPreparationError() from None


def _preflight_destinations(
    labels_output: Path | str,
    output_dir: Path | str,
    *,
    protected_paths: tuple[Path, ...] = (),
) -> tuple[Path, Path]:
    labels_path = _destination(labels_output)
    corpus_path = _destination(output_dir)
    if (
        labels_path == corpus_path
        or _is_relative_to(labels_path, corpus_path)
        or _is_relative_to(corpus_path, labels_path)
    ):
        raise LiveSttCorpusPreparationError()
    for source in protected_paths:
        if any(
            destination == source
            or _is_relative_to(destination, source)
            or _is_relative_to(source, destination)
            for destination in (labels_path, corpus_path)
        ):
            raise LiveSttCorpusPreparationError()
    _require_absent(labels_path, private_parent=True)
    _require_absent(corpus_path, private_parent=True)
    return labels_path, corpus_path


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_size,
        metadata.st_uid,
    )


def _directory_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
    )


def _publish_private_labels(path: Path, payload: bytes) -> _PublishedLabelWitness:
    """Atomically link one fully written private inode into an absent path."""

    descriptor = -1
    parent_fd = -1
    temporary_name = f".live-stt-labels-{os.getpid()}-{secrets.token_hex(12)}.tmp"
    temporary_exists = False
    candidate_linked = False
    publication_complete = False
    opened: os.stat_result | None = None
    try:
        with opened_directory_nofollow(
            path.parent,
            require_private=True,
        ) as (stable_parent, opened_parent_fd):
            if stable_parent != path.parent:
                raise LiveSttCorpusPreparationError()
            parent_fd = os.dup(opened_parent_fd)
            parent_identity = _directory_identity(os.fstat(parent_fd))
            try:
                os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise LiveSttCorpusPreparationError()
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(temporary_name, flags, 0o600, dir_fd=parent_fd)
            temporary_exists = True

        os.fchmod(descriptor, 0o600)
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise LiveSttCorpusPreparationError()
            written += count
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size != len(payload)
            or stat.S_IMODE(opened.st_mode) != 0o600
            or (hasattr(os, "geteuid") and opened.st_uid != os.geteuid())
        ):
            raise LiveSttCorpusPreparationError()
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        remaining = len(payload)
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                raise LiveSttCorpusPreparationError()
            digest.update(chunk)
            remaining -= len(chunk)
        current_parent = path.parent.lstat()
        if (
            digest.digest() != hashlib.sha256(payload).digest()
            or _directory_identity(current_parent) != parent_identity
            or path.parent.resolve(strict=True) != path.parent
            or stat.S_IMODE(current_parent.st_mode) & 0o077
            or (
                hasattr(os, "geteuid")
                and current_parent.st_uid != os.geteuid()
            )
        ):
            raise LiveSttCorpusPreparationError()

        os.link(
            temporary_name,
            path.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        candidate_linked = True
        os.unlink(temporary_name, dir_fd=parent_fd)
        temporary_exists = False
        os.fsync(parent_fd)
        published = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        lexical = path.lstat()
        if (
            _file_identity(published) != _file_identity(opened)
            or _file_identity(lexical) != _file_identity(opened)
            or published.st_nlink != 1
            or lexical.st_nlink != 1
            or stat.S_IMODE(published.st_mode) != 0o600
            or stat.S_IMODE(lexical.st_mode) != 0o600
        ):
            raise LiveSttCorpusPreparationError()
        publication_complete = True
    except LiveSttCorpusPreparationError:
        raise
    except Exception:
        raise LiveSttCorpusPreparationError() from None
    finally:
        if temporary_exists and parent_fd >= 0:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except OSError:
                pass
        if (
            candidate_linked
            and not publication_complete
            and parent_fd >= 0
            and opened is not None
        ):
            try:
                candidate = os.stat(
                    path.name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                if (candidate.st_dev, candidate.st_ino) == (
                    opened.st_dev,
                    opened.st_ino,
                ):
                    os.unlink(path.name, dir_fd=parent_fd)
                    os.fsync(parent_fd)
            except OSError:
                pass
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if parent_fd >= 0:
            try:
                os.close(parent_fd)
            except OSError:
                pass

    try:
        published_path, published_raw = _private_snapshot(
            path,
            maximum_bytes=_MAX_LABEL_BYTES,
        )
    except DiagnosticCorpusPreparationError:
        raise LiveSttCorpusPreparationError() from None
    if published_path != path or published_raw != payload:
        raise LiveSttCorpusPreparationError()
    try:
        metadata = published_path.stat()
    except OSError:
        raise LiveSttCorpusPreparationError() from None
    return _PublishedLabelWitness(
        path=published_path,
        sha256=hashlib.sha256(published_raw).hexdigest(),
        inode=(metadata.st_dev, metadata.st_ino),
    )


def _recheck_published_labels(
    witness: _PublishedLabelWitness,
    expected_payload: bytes,
) -> None:
    try:
        current_path, current_raw = _private_snapshot(
            witness.path,
            maximum_bytes=_MAX_LABEL_BYTES,
        )
        metadata = current_path.stat()
    except (OSError, DiagnosticCorpusPreparationError):
        raise LiveSttCorpusPreparationError() from None
    if (
        current_path != witness.path
        or current_raw != expected_payload
        or hashlib.sha256(current_raw).hexdigest() != witness.sha256
        or (metadata.st_dev, metadata.st_ino) != witness.inode
    ):
        raise LiveSttCorpusPreparationError()


def _recheck_private_source(
    path: Path,
    *,
    maximum_bytes: int,
    expected_path: Path,
    expected_sha256: str,
) -> None:
    try:
        current_path, raw = _private_snapshot(path, maximum_bytes=maximum_bytes)
    except DiagnosticCorpusPreparationError:
        raise LiveSttCorpusPreparationError() from None
    if (
        current_path != expected_path
        or hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        raise LiveSttCorpusPreparationError()


def prepare_live_stt_corpus(
    *,
    diagnostic_manifest: Path | str,
    reference_plan: Path | str,
    labels_output: Path | str,
    output_dir: Path | str,
) -> LoadedCorpus:
    """Publish complete labels and a schema-v3 corpus for one live bundle."""

    try:
        labels_path, corpus_path = _preflight_destinations(
            labels_output,
            output_dir,
        )
        plan_path, plan_raw = _private_snapshot(
            reference_plan,
            maximum_bytes=_MAX_REFERENCE_PLAN_BYTES,
        )
        plan = _reference_plan(plan_raw)
        plan_digest = hashlib.sha256(plan_raw).hexdigest()
        inventory = load_diagnostic_input_inventory(diagnostic_manifest)
        _validate_inventory(inventory)
        labels_payload = _label_payload(plan, inventory)

        # Recheck both absent destinations and both private source bindings at
        # the last safe point before publishing the complete label inode.
        _preflight_destinations(
            labels_path,
            corpus_path,
            protected_paths=(
                plan_path,
                inventory.manifest_path,
                inventory.spool_path,
            ),
        )
        _recheck_private_source(
            plan_path,
            maximum_bytes=_MAX_REFERENCE_PLAN_BYTES,
            expected_path=plan_path,
            expected_sha256=plan_digest,
        )
        _recheck_private_source(
            inventory.manifest_path,
            maximum_bytes=1 << 20,
            expected_path=inventory.manifest_path,
            expected_sha256=inventory.manifest_sha256,
        )
        label_witness = _publish_private_labels(labels_path, labels_payload)

        # A changed plan cannot silently authorize corpus publication.  A
        # complete label may remain for reuse if this or delegated publication
        # fails; a partial label is never linked into place.
        _recheck_private_source(
            plan_path,
            maximum_bytes=_MAX_REFERENCE_PLAN_BYTES,
            expected_path=plan_path,
            expected_sha256=plan_digest,
        )
        _recheck_published_labels(label_witness, labels_payload)
        try:
            corpus = prepare_diagnostic_corpus(
                diagnostic_manifest=inventory.manifest_path,
                labels=labels_path,
                output_dir=corpus_path,
                expected_labels_sha256=label_witness.sha256,
                expected_labels_inode=label_witness.inode,
            )
        finally:
            _recheck_published_labels(label_witness, labels_payload)
        if (
            corpus.schema_version != 3
            or corpus.provenance is None
            or corpus.provenance.metadata_sha256 != label_witness.sha256
            or corpus.provenance.manifest_sha256 != inventory.manifest_sha256
        ):
            raise LiveSttCorpusPreparationError()
        _recheck_private_source(
            plan_path,
            maximum_bytes=_MAX_REFERENCE_PLAN_BYTES,
            expected_path=plan_path,
            expected_sha256=plan_digest,
        )
        return corpus
    except LiveSttCorpusPreparationError:
        raise
    except Exception:
        raise LiveSttCorpusPreparationError() from None


def _parser() -> argparse.ArgumentParser:
    parser = _PrivateArgumentParser(
        description=(
            "Bind an ordered private reference plan to every exact live "
            "final-input receipt and publish one private STT corpus."
        )
    )
    parser.add_argument("--diagnostic-manifest", type=Path, required=True)
    parser.add_argument("--reference-plan", type=Path, required=True)
    parser.add_argument("--labels-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        corpus = prepare_live_stt_corpus(
            diagnostic_manifest=args.diagnostic_manifest,
            reference_plan=args.reference_plan,
            labels_output=args.labels_output,
            output_dir=args.output_dir,
        )
        result: Mapping[str, object] = _safe_result(corpus)
        code = 0
    except Exception:  # noqa: BLE001 - private paths and references stay private
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


__all__ = [
    "LiveSttCorpusPreparationError",
    "main",
    "prepare_live_stt_corpus",
]
