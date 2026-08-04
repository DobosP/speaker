"""No-overwrite publication for private versioned streaming-STT corpora.

Callers validate source provenance and choose the cases.  This module owns the
small shared publication boundary: exact private permissions, exclusive file
creation, deterministic JSON, and a final read through the authoritative corpus
loader.  A failed publication is intentionally left in place as evidence.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Iterable, Iterator, Mapping

from core.wer import normalize

from .bounded_io import BoundedReadError, opened_directory_nofollow
from .corpus import CorpusError, CorpusProvenance, LoadedCorpus, load_corpus
from .protocol import MAX_CORPUS_BYTES, MAX_PCM_BYTES


_SAFE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}\Z")
_SAFE_SIDECAR_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_PUBLIC_PROVENANCE_KIND = "public-voice-v1"
_PRIVATE_DIAGNOSTIC_PROVENANCE_KIND = "private-diagnostic-v1"
_PUBLIC_COMMAND_NOISE_PROVENANCE_KIND = "public-command-noise-v1"
_SCHEMA_BY_PROVENANCE_KIND = {
    _PUBLIC_PROVENANCE_KIND: 2,
    _PRIVATE_DIAGNOSTIC_PROVENANCE_KIND: 3,
    _PUBLIC_COMMAND_NOISE_PROVENANCE_KIND: 4,
}
_PREPARATION_RECEIPT_FILENAME = "preparation-receipt.json"
_MAX_CASES = 512
_MAX_PURPOSE_CHARS = 512
_MAX_REFERENCE_CHARS = 4096
_MAX_TAGS = 32
_MAX_TAG_CHARS = 64
_MAX_CORPUS_MANIFEST_BYTES = 256 * 1024
_MAX_SIDECARS = 16
_MAX_SIDECAR_BYTES = 256 * 1024
_MAX_ALL_SIDECAR_BYTES = 1024 * 1024


class CorpusWriterError(RuntimeError):
    """A detail-free invalid corpus or failed private publication."""


@dataclass(frozen=True)
class CorpusWriteCase:
    """One caller-selected transcript case ready for corpus publication."""

    case_id: str
    audio_bytes: bytes = field(repr=False)
    reference: str = field(repr=False)
    tags: tuple[str, ...]
    commands: tuple[str, ...] = field(default=(), repr=False)
    assertion: str = "transcript"
    forbidden_commands: tuple[str, ...] = field(default=(), repr=False)


@dataclass(frozen=True)
class _PrivateOutput:
    path: Path
    parent_fd: int = field(repr=False)
    directory_fd: int = field(repr=False)
    parent_identity: tuple[int, int, int]
    directory_identity: tuple[int, int, int]


@dataclass(frozen=True)
class _WrittenPrivateFile:
    filename: str
    snapshot: tuple[int, int, int, int, int, int, int, int, int]
    size_bytes: int
    sha256: str


def _contains_tokens(
    tokens: tuple[str, ...],
    target: tuple[str, ...],
) -> bool:
    if not target or len(target) > len(tokens):
        return False
    return any(
        tokens[index : index + len(target)] == target
        for index in range(len(tokens) - len(target) + 1)
    )


def _identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
    )


def _file_snapshot(
    metadata: os.stat_result,
) -> tuple[int, int, int, int, int, int, int, int, int]:
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


def _verify_output_binding(output: _PrivateOutput) -> None:
    try:
        parent = os.fstat(output.parent_fd)
        opened = os.fstat(output.directory_fd)
        entry = os.stat(
            output.path.name,
            dir_fd=output.parent_fd,
            follow_symlinks=False,
        )
        lexical_parent = output.path.parent.lstat()
        lexical_entry = output.path.lstat()
        if (
            _identity(parent) != output.parent_identity
            or _identity(lexical_parent) != output.parent_identity
            or output.path.parent.resolve(strict=True) != output.path.parent
            or _identity(opened) != output.directory_identity
            or _identity(entry) != output.directory_identity
            or _identity(lexical_entry) != output.directory_identity
            or output.path.resolve(strict=True) != output.path
            or not stat.S_ISDIR(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o700
            or stat.S_IMODE(entry.st_mode) != 0o700
            or stat.S_IMODE(lexical_entry.st_mode) != 0o700
            or (
                hasattr(os, "geteuid")
                and (
                    opened.st_uid != os.geteuid()
                    or entry.st_uid != os.geteuid()
                    or lexical_entry.st_uid != os.geteuid()
                )
            )
        ):
            raise CorpusWriterError()
    except (OSError, RuntimeError, ValueError, OverflowError):
        raise CorpusWriterError() from None


@contextmanager
def _new_private_output(path: Path | str) -> Iterator[_PrivateOutput]:
    candidate = Path(os.path.abspath(Path(path).expanduser()))
    if not candidate.name or candidate.name in {".", ".."}:
        raise CorpusWriterError()
    directory_fd = -1
    try:
        with opened_directory_nofollow(candidate.parent) as (parent, parent_fd):
            if parent != candidate.parent:
                raise CorpusWriterError()
            parent_metadata = os.fstat(parent_fd)
            os.mkdir(candidate.name, mode=0o700, dir_fd=parent_fd)
            created = os.stat(
                candidate.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(created.st_mode):
                raise CorpusWriterError()
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_DIRECTORY", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            directory_fd = os.open(candidate.name, flags, dir_fd=parent_fd)
            os.fchmod(directory_fd, 0o700)
            metadata = os.fstat(directory_fd)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or _identity(metadata) != _identity(created)
                or stat.S_IMODE(metadata.st_mode) != 0o700
                or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
            ):
                raise CorpusWriterError()
            output = _PrivateOutput(
                path=candidate,
                parent_fd=parent_fd,
                directory_fd=directory_fd,
                parent_identity=_identity(parent_metadata),
                directory_identity=_identity(metadata),
            )
            os.fsync(directory_fd)
            os.fsync(parent_fd)
            _verify_output_binding(output)
            yield output
            _verify_output_binding(output)
            os.fsync(directory_fd)
            os.fsync(parent_fd)
    except (OSError, BoundedReadError, CorpusWriterError):
        raise CorpusWriterError() from None
    finally:
        if directory_fd >= 0:
            try:
                os.close(directory_fd)
            except OSError:
                pass


def _write_new_private(
    directory_fd: int,
    filename: str,
    payload: bytes,
) -> _WrittenPrivateFile:
    if (
        not isinstance(filename, str)
        or not filename
        or "/" in filename
        or filename in {".", ".."}
        or not isinstance(payload, bytes)
    ):
        raise CorpusWriterError()
    descriptor = -1
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(filename, flags, 0o600, dir_fd=directory_fd)
        os.fchmod(descriptor, 0o600)
        opened = os.fstat(descriptor)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise CorpusWriterError()
            view = view[written:]
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        current = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
        if (
            _identity(after) != _identity(opened)
            or _identity(current) != _identity(opened)
            or _file_snapshot(current) != _file_snapshot(after)
            or not stat.S_ISREG(after.st_mode)
            or after.st_nlink != 1
            or current.st_nlink != 1
            or after.st_size != len(payload)
            or stat.S_IMODE(after.st_mode) != 0o600
            or stat.S_IMODE(current.st_mode) != 0o600
            or (hasattr(os, "geteuid") and after.st_uid != os.geteuid())
            or (hasattr(os, "geteuid") and current.st_uid != os.geteuid())
        ):
            raise CorpusWriterError()
        os.fsync(directory_fd)
        return _WrittenPrivateFile(
            filename=filename,
            snapshot=_file_snapshot(after),
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
    except (OSError, ValueError, OverflowError, CorpusWriterError):
        raise CorpusWriterError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _verify_written_private_metadata(
    directory_fd: int,
    written: _WrittenPrivateFile,
) -> None:
    try:
        current = os.stat(
            written.filename,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if (
            _file_snapshot(current) != written.snapshot
            or not stat.S_ISREG(current.st_mode)
            or current.st_nlink != 1
            or current.st_size != written.size_bytes
            or stat.S_IMODE(current.st_mode) != 0o600
            or (hasattr(os, "geteuid") and current.st_uid != os.geteuid())
        ):
            raise CorpusWriterError()
    except (OSError, ValueError, OverflowError, CorpusWriterError):
        raise CorpusWriterError() from None


def _read_exact_written_private(
    directory_fd: int,
    written: _WrittenPrivateFile,
    expected: bytes,
    *,
    maximum_bytes: int,
) -> str:
    if (
        not isinstance(expected, bytes)
        or len(expected) != written.size_bytes
        or len(expected) > maximum_bytes
    ):
        raise CorpusWriterError()
    descriptor = -1
    try:
        before = os.stat(
            written.filename,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if _file_snapshot(before) != written.snapshot:
            raise CorpusWriterError()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(written.filename, flags, dir_fd=directory_fd)
        opened = os.fstat(descriptor)
        if _file_snapshot(opened) != written.snapshot:
            raise CorpusWriterError()
        chunks: list[bytes] = []
        consumed = 0
        while consumed <= written.size_bytes:
            chunk = os.read(
                descriptor,
                min(64 * 1024, written.size_bytes + 1 - consumed),
            )
            if not chunk:
                break
            chunks.append(chunk)
            consumed += len(chunk)
            if consumed > written.size_bytes:
                raise CorpusWriterError()
        after = os.fstat(descriptor)
        current = os.stat(
            written.filename,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        raw = b"".join(chunks)
        digest = hashlib.sha256(raw).hexdigest()
        if (
            consumed != written.size_bytes
            or raw != expected
            or digest != written.sha256
            or _file_snapshot(after) != written.snapshot
            or _file_snapshot(current) != written.snapshot
        ):
            raise CorpusWriterError()
        return digest
    except (OSError, ValueError, OverflowError, CorpusWriterError):
        raise CorpusWriterError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _preflight(
    cases: Iterable[CorpusWriteCase],
    provenance: CorpusProvenance,
    purpose: str,
    sidecars: Mapping[str, bytes] | None,
) -> tuple[tuple[CorpusWriteCase, ...], tuple[tuple[str, bytes], ...]]:
    if (
        not isinstance(purpose, str)
        or not purpose.strip()
        or len(purpose) > _MAX_PURPOSE_CHARS
        or type(provenance) is not CorpusProvenance
        or provenance.kind not in _SCHEMA_BY_PROVENANCE_KIND
        or not isinstance(provenance.suite, str)
        or _SAFE_ID_RE.fullmatch(provenance.suite) is None
        or not isinstance(provenance.manifest_sha256, str)
        or _SHA256_RE.fullmatch(provenance.manifest_sha256) is None
        or not isinstance(provenance.metadata_sha256, str)
        or _SHA256_RE.fullmatch(provenance.metadata_sha256) is None
        or not isinstance(provenance.source_set_sha256, str)
        or _SHA256_RE.fullmatch(provenance.source_set_sha256) is None
    ):
        raise CorpusWriterError()
    schema_version = _SCHEMA_BY_PROVENANCE_KIND[provenance.kind]
    prepared_rows: list[CorpusWriteCase] = []
    total_bytes = 0
    identifiers: list[str] = []
    identifier_set: set[str] = set()
    try:
        iterator = iter(cases)
        for index in range(_MAX_CASES + 1):
            try:
                case = next(iterator)
            except StopIteration:
                break
            if index >= _MAX_CASES:
                raise CorpusWriterError()
            commands_valid = (
                isinstance(case, CorpusWriteCase)
                and isinstance(case.commands, tuple)
                and len(case.commands) <= 16
                and all(
                    isinstance(command, str)
                    and command
                    and len(command) <= 128
                    and normalize(command)
                    for command in case.commands
                )
                and len(set(case.commands)) == len(case.commands)
            )
            forbidden_commands_valid = (
                isinstance(case, CorpusWriteCase)
                and isinstance(case.forbidden_commands, tuple)
                and len(case.forbidden_commands) <= 16
                and all(
                    isinstance(command, str)
                    and command
                    and len(command) <= 128
                    and normalize(command)
                    for command in case.forbidden_commands
                )
                and len(set(case.forbidden_commands))
                == len(case.forbidden_commands)
            )
            if (
                not isinstance(case, CorpusWriteCase)
                or not isinstance(case.case_id, str)
                or _SAFE_ID_RE.fullmatch(case.case_id) is None
                or case.case_id in identifier_set
                or not isinstance(case.audio_bytes, bytes)
                or not case.audio_bytes
                or len(case.audio_bytes) % 4
                or len(case.audio_bytes) > MAX_PCM_BYTES
                or not isinstance(case.reference, str)
                or len(case.reference) > _MAX_REFERENCE_CHARS
                or not isinstance(case.tags, tuple)
                or len(case.tags) > _MAX_TAGS
                or any(
                    not isinstance(tag, str)
                    or not tag
                    or len(tag) > _MAX_TAG_CHARS
                    or _SAFE_ID_RE.fullmatch(tag) is None
                    for tag in case.tags
                )
                or len(set(case.tags)) != len(case.tags)
                or not commands_valid
                or not forbidden_commands_valid
            ):
                raise CorpusWriterError()
            normalized_commands = tuple(tuple(normalize(value)) for value in case.commands)
            normalized_forbidden = tuple(
                tuple(normalize(value)) for value in case.forbidden_commands
            )
            reference_tokens = tuple(normalize(case.reference))
            if schema_version in {2, 3}:
                if (
                    case.assertion != "transcript"
                    or case.forbidden_commands
                    or not case.reference.strip()
                    or not normalize(case.reference)
                ):
                    raise CorpusWriterError()
            elif (
                case.assertion not in {"transcript", "speech_negative", "silence"}
                or len(set(normalized_commands)) != len(normalized_commands)
                or len(set(normalized_forbidden)) != len(normalized_forbidden)
                or set(normalized_commands).intersection(normalized_forbidden)
                or (
                    case.assertion == "transcript"
                    and (
                        not case.reference.strip()
                        or not reference_tokens
                        or not (case.commands or case.forbidden_commands)
                        or any(
                            not _contains_tokens(reference_tokens, target)
                            for target in normalized_commands
                        )
                        or any(
                            _contains_tokens(reference_tokens, target)
                            for target in normalized_forbidden
                        )
                    )
                )
                or (
                    case.assertion == "speech_negative"
                    and (
                        case.reference != ""
                        or case.commands
                        or not case.forbidden_commands
                    )
                )
                or (
                    case.assertion == "silence"
                    and (case.reference != "" or case.commands)
                )
            ):
                raise CorpusWriterError()
            total_bytes += len(case.audio_bytes)
            if total_bytes > MAX_CORPUS_BYTES:
                raise CorpusWriterError()
            identifiers.append(case.case_id)
            identifier_set.add(case.case_id)
            prepared_rows.append(case)
    except CorpusWriterError:
        raise
    except Exception:
        raise CorpusWriterError() from None
    prepared = tuple(prepared_rows)
    if not prepared:
        raise CorpusWriterError()
    if sidecars is None:
        prepared_sidecars: tuple[tuple[str, bytes], ...] = ()
    elif not isinstance(sidecars, Mapping):
        raise CorpusWriterError()
    else:
        reserved = {"corpus.json", *(f"{value}.f32le" for value in identifiers)}
        sidecar_rows: list[tuple[str, bytes]] = []
        sidecar_total = 0
        sidecar_names: set[str] = set()
        try:
            for index, item in enumerate(sidecars.items()):
                if index >= _MAX_SIDECARS:
                    raise CorpusWriterError()
                try:
                    filename, payload = item
                except (TypeError, ValueError):
                    raise CorpusWriterError() from None
                if (
                    not isinstance(filename, str)
                    or _SAFE_SIDECAR_RE.fullmatch(filename) is None
                    or filename in reserved
                    or filename in sidecar_names
                    or not isinstance(payload, bytes)
                    or not payload
                    or len(payload) > _MAX_SIDECAR_BYTES
                ):
                    raise CorpusWriterError()
                sidecar_total += len(payload)
                if sidecar_total > _MAX_ALL_SIDECAR_BYTES:
                    raise CorpusWriterError()
                sidecar_names.add(filename)
                sidecar_rows.append((filename, payload))
        except CorpusWriterError:
            raise
        except Exception:
            raise CorpusWriterError() from None
        prepared_sidecars = tuple(sorted(sidecar_rows))
    if schema_version in {3, 4}:
        preparation_receipt = dict(prepared_sidecars).get(
            _PREPARATION_RECEIPT_FILENAME
        )
        expected_digest = (
            provenance.source_set_sha256
            if schema_version == 3
            else provenance.metadata_sha256
        )
        if (
            preparation_receipt is None
            or hashlib.sha256(preparation_receipt).hexdigest()
            != expected_digest
        ):
            raise CorpusWriterError()
    return prepared, prepared_sidecars


def _serialize_manifest(
    cases: tuple[CorpusWriteCase, ...],
    provenance: CorpusProvenance,
    purpose: str,
) -> tuple[list[dict[str, object]], bytes]:
    try:
        schema_version = _SCHEMA_BY_PROVENANCE_KIND[provenance.kind]
    except (KeyError, TypeError):
        raise CorpusWriterError() from None
    rows: list[dict[str, object]] = []
    for case in cases:
        raw = case.audio_bytes
        row: dict[str, object] = {
            "id": case.case_id,
            "file": f"{case.case_id}.f32le",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "samples": len(raw) // 4,
            "expected_text": case.reference,
            "assertion": (
                case.assertion if schema_version == 4 else "transcript"
            ),
            "commands": list(case.commands),
            "tags": list(case.tags),
        }
        if schema_version == 4:
            row["forbidden_commands"] = list(case.forbidden_commands)
        rows.append(row)
    try:
        payload = (
            json.dumps(
                {
                    "schema_version": schema_version,
                    "purpose": purpose,
                    "provenance": {
                        "kind": provenance.kind,
                        "suite": provenance.suite,
                        "manifest_sha256": provenance.manifest_sha256,
                        "metadata_sha256": provenance.metadata_sha256,
                        "source_set_sha256": provenance.source_set_sha256,
                    },
                    "cases": rows,
                },
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
    except (MemoryError, TypeError, UnicodeError, ValueError, OverflowError):
        raise CorpusWriterError() from None
    if not payload or len(payload) > _MAX_CORPUS_MANIFEST_BYTES:
        raise CorpusWriterError()
    return rows, payload


def publish_private_corpus(
    *,
    cases: Iterable[CorpusWriteCase],
    provenance: CorpusProvenance,
    output_dir: Path | str,
    purpose: str,
    sidecars: Mapping[str, bytes] | None = None,
) -> LoadedCorpus:
    """Publish one immutable private corpus and validate the exact result.

    The destination must not exist.  Invalid source values are rejected before
    creation when possible.  Once publication begins, an error never removes or
    overwrites files, so interrupted output remains available for diagnosis.
    """

    prepared, prepared_sidecars = _preflight(
        cases,
        provenance,
        purpose,
        sidecars,
    )
    rows, payload = _serialize_manifest(prepared, provenance, purpose)
    manifest_digest = hashlib.sha256(payload).hexdigest()
    loaded: LoadedCorpus | None = None
    try:
        with _new_private_output(output_dir) as output:
            descriptor = output.directory_fd
            sidecar_receipts: list[tuple[_WrittenPrivateFile, bytes]] = []
            for filename, sidecar_payload in prepared_sidecars:
                sidecar_receipts.append(
                    (
                        _write_new_private(descriptor, filename, sidecar_payload),
                        sidecar_payload,
                    )
                )
            audio_receipts: list[_WrittenPrivateFile] = []
            for case, row in zip(prepared, rows, strict=True):
                audio_receipts.append(
                    _write_new_private(
                        descriptor,
                        str(row["file"]),
                        case.audio_bytes,
                    )
                )
            manifest_receipt = _write_new_private(descriptor, "corpus.json", payload)

            _verify_output_binding(output)
            for receipt, expected in sidecar_receipts:
                _read_exact_written_private(
                    descriptor,
                    receipt,
                    expected,
                    maximum_bytes=_MAX_SIDECAR_BYTES,
                )
            if (
                _read_exact_written_private(
                    descriptor,
                    manifest_receipt,
                    payload,
                    maximum_bytes=_MAX_CORPUS_MANIFEST_BYTES,
                )
                != manifest_digest
            ):
                raise CorpusWriterError()
            for receipt in audio_receipts:
                _verify_written_private_metadata(descriptor, receipt)

            loaded = load_corpus(output.path / "corpus.json")

            _verify_output_binding(output)
            for receipt, expected in sidecar_receipts:
                _read_exact_written_private(
                    descriptor,
                    receipt,
                    expected,
                    maximum_bytes=_MAX_SIDECAR_BYTES,
                )
            if (
                _read_exact_written_private(
                    descriptor,
                    manifest_receipt,
                    payload,
                    maximum_bytes=_MAX_CORPUS_MANIFEST_BYTES,
                )
                != manifest_digest
            ):
                raise CorpusWriterError()
            for receipt in audio_receipts:
                _verify_written_private_metadata(descriptor, receipt)
            if (
                loaded.path != output.path / "corpus.json"
                or loaded.digest != manifest_digest
                or loaded.provenance != provenance
            ):
                raise CorpusWriterError()
    except (OSError, BoundedReadError, CorpusError, CorpusWriterError):
        raise CorpusWriterError() from None
    if loaded is None:
        raise CorpusWriterError()
    return loaded


__all__ = [
    "CorpusWriteCase",
    "CorpusWriterError",
    "publish_private_corpus",
]
