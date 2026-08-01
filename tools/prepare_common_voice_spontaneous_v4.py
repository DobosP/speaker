"""Prepare a bounded Common Voice Spontaneous Speech v4 English STT slice.

This command never downloads a corpus, contacts MDC, runs a recognizer, or
opens an audio device.  It accepts one already acquired private archive, checks
the exact public release identity and a redacted MDC content receipt, parses
only the allowlisted main metadata, and publishes a deterministic private
schema-v2 streaming corpus plus a transcript-free preparation receipt.

Reported-audio metadata is deliberately never read.  The v4 archive retains
reported clips (including reports about PII), so only audio paths referenced by
the main ``ss-corpus-en.tsv`` may reach the decoder.
"""

from __future__ import annotations

from array import array
import argparse
import csv
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import sys
import tarfile
from typing import Callable, Iterator, Mapping, Sequence

from core.wer import normalize
from tools.public_voice_eval_matrix import (
    SelectionPolicy,
    dataset_by_id,
    selected_rows_sha256,
    selection_policy_sha256,
    select_subset_rows,
)
from tools.streaming_stt.bounded_io import (
    BoundedReadError,
    hash_regular_bounded,
    opened_directory_nofollow,
)
from tools.streaming_stt.corpus import CorpusProvenance, LoadedCorpus
from tools.streaming_stt.corpus_writer import (
    CorpusWriteCase,
    publish_private_corpus,
)
from tools.streaming_stt.protocol import MAX_CORPUS_BYTES, MAX_PCM_BYTES


DATASET_ID = "common_voice_spontaneous_4_en"
MDC_DATASET_ID = "cmqialpeo0077nr077xqdqo0j"
RELEASE_ID = "sps-corpus-4.0-2026-06-12"
RELEASE_DATE = "2026-06-12"
MDC_PUBLISHED_DATE = "2026-06-17"
LOCALE = "en"
ARCHIVE_FILENAME = "common-voice-spontaneous-speech-4-0-engl-c643378f.tar.gz"
ARCHIVE_SIZE_BYTES = 522_005_930
ARCHIVE_SHA256 = "3b03ada7676a5f440a797d896035137fd073d0683133c3e9a83963480d88abfe"
ARCHIVE_ROOT = f"{RELEASE_ID}-{LOCALE}"

STATS_GIT_COMMIT = "f99d8239d2796131b73ac99f92ee7cb4443bf3ba"
STATS_GIT_BLOB = "32954cc956a0dd2a2fcdec0f42048982e80ae167"
STATS_SHA256 = "6e770f26f125590417a46b02bd413589374971cf4cbba87da11135890f19e6a4"
SCHEMA_REGISTRY_COMMIT = "0fc0999776f4262b8acbe18e2ba2bd75fa5778e5"

REQUIRED_TERMS = frozenset({"CC0-1.0", "MDC-CONSUMER-TERMS-2026-05-06"})
EXPECTED_ROWS = 5_679
EXPECTED_DURATION_MS = 63_315_720
EXPECTED_SPLIT_COUNTS = {
    "train": 1_519,
    "dev": 409,
    "test": 377,
    "unassigned": 3_374,
}
EXPECTED_SPLIT_DURATION_MS = {
    "train": 17_247_600,
    "dev": 5_468_724,
    "test": 5_430_960,
    "unassigned": 35_168_436,
}
EXPECTED_SPLIT_SPEAKERS = {"train": 274, "dev": 33, "test": 79}
EXPECTED_TOTAL_SPEAKERS = 609

MAIN_TSV_FIELDS = (
    "client_id",
    "audio_id",
    "audio_file",
    "duration_ms",
    "prompt_id",
    "prompt",
    "transcription",
    "votes",
    "age",
    "gender",
    "accents",
    "variant",
    "language",
    "prompt_upvotes",
    "prompt_reports",
    "is_edited",
    "split",
    "char_per_sec",
    "quality_tags",
)

DURATION_BUCKETS = (
    ("d0_2_6s", 2_000, 6_000, False),
    ("d1_6_10s", 6_000, 10_000, False),
    ("d2_10_15s", 10_000, 15_000, False),
    ("d3_15_25s", 15_000, 25_000, True),
)
SELECTION_COUNT_PER_BUCKET = 8
SELECTION_COUNT = 32
MAX_DURATION_DELTA_MS = 500

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MAIN_MEMBER = f"{ARCHIVE_ROOT}/ss-corpus-en.tsv"
_REPORTED_MEMBER = f"{ARCHIVE_ROOT}/ss-reported-audios-en.tsv"
_QA_MEMBER = f"{ARCHIVE_ROOT}/ss-corpus-en.qa-summary.json"
_README_MEMBER = f"{ARCHIVE_ROOT}/README.md"
_AUDIO_PREFIX = f"{ARCHIVE_ROOT}/audios/"
_MAX_ARCHIVE_BYTES = 1024 * 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 12_000
_MAX_DECLARED_MEMBER_BYTES = 2 * 1024 * 1024 * 1024
_MAX_MAIN_TSV_BYTES = 32 * 1024 * 1024
_MAX_AUDIO_MEMBER_BYTES = 16 * 1024 * 1024
_MAX_TEXT_FIELD_CHARS = 32_768
_MAX_RECEIPT_BYTES = 256 * 1024
_MAX_RUNTIME_TREE_ENTRIES = 10_000
_MAX_RUNTIME_TREE_BYTES = 1024 * 1024 * 1024
_MAX_RUNTIME_TREE_DEPTH = 32
_SAFE_AUDIO_RE = re.compile(
    r"spontaneous-speech-en-[A-Za-z0-9][A-Za-z0-9._-]{0,159}\.mp3\Z"
)
_SAFE_TAG_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")
_UNSIGNED_INT_RE = re.compile(r"0|[1-9][0-9]*\Z")
_SIGNED_INT_RE = re.compile(r"-?(?:0|[1-9][0-9]*)\Z")
_BRACKET_ANNOTATION_RE = re.compile(r"\[[^\]\r\n]{1,80}\]")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ERROR = {
    "ok": False,
    "error": "common_voice_spontaneous_v4_prerequisites_unavailable",
}


class SpsPreparationError(RuntimeError):
    """A detail-free release, selection, decode, or publication failure."""


@dataclass(frozen=True, slots=True)
class MdcApiReceipt:
    """Non-secret content fields copied from one MDC download response."""

    filename: str
    size_bytes: int
    checksum_algorithm: str
    checksum_sha256: str


@dataclass(frozen=True, slots=True)
class DecodedAudio:
    raw_f32le: bytes = field(repr=False)
    samples: int


@dataclass(frozen=True, slots=True)
class TestDecoderInjection:
    """An unmistakably test-only decode seam with no caller-authored receipt."""

    decoder: Callable[[bytes], DecodedAudio] = field(repr=False)
    fixture_id: str


@dataclass(frozen=True, slots=True)
class SpsRow:
    client_id: str = field(repr=False)
    audio_id: str = field(repr=False)
    audio_file: str = field(repr=False)
    prompt_id: str = field(repr=False)
    duration_ms: int
    transcription: str = field(repr=False)
    split: str
    language: str
    quality_tags: tuple[str, ...]

    def selector_row(self, duration_bucket: str) -> dict[str, object]:
        return {
            "audio_id": self.audio_id,
            "audio_file": self.audio_file,
            "client_id": self.client_id,
            "duration_bucket": duration_bucket,
        }


@dataclass(frozen=True, slots=True)
class ParsedMetadata:
    rows: tuple[SpsRow, ...] = field(repr=False)
    split_speakers: Mapping[str, frozenset[str]] = field(repr=False)
    split_counts: Mapping[str, int]
    split_duration_ms: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class ArchiveLayout:
    members: Mapping[str, tarfile.TarInfo] = field(repr=False)
    main_tsv: tarfile.TarInfo = field(repr=False)
    reported_tsv: tarfile.TarInfo = field(repr=False)
    qa_summary: tarfile.TarInfo = field(repr=False)
    readme: tarfile.TarInfo = field(repr=False)


@dataclass(frozen=True, slots=True)
class PreparedSpsCorpus:
    corpus: LoadedCorpus
    receipt_sha256: str
    selection_recipe_sha256: str
    metadata_sha256: str


Decoder = Callable[[bytes], DecodedAudio]


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
    except (TypeError, ValueError, OverflowError):
        raise SpsPreparationError("receipt contract failed") from None


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


def _validate_api_receipt(receipt: MdcApiReceipt) -> None:
    if (
        receipt.filename != ARCHIVE_FILENAME
        or type(receipt.size_bytes) is not int
        or receipt.size_bytes != ARCHIVE_SIZE_BYTES
        or receipt.checksum_algorithm != "sha256"
        or _SHA256_RE.fullmatch(receipt.checksum_sha256) is None
        or receipt.checksum_sha256 != ARCHIVE_SHA256
    ):
        raise SpsPreparationError("MDC content receipt failed")


def _validate_catalog_contract() -> None:
    """Keep the executable release pins identical to the acquisition catalog."""

    try:
        dataset = dataset_by_id(DATASET_ID)
        archives = [
            artifact
            for artifact in dataset.artifacts
            if artifact.artifact_id == "archive"
        ]
        artifact = archives[0]
        checksum = artifact.checksum
        valid = (
            len(archives) == 1
            and dataset.upstream_id == f"mdc:{MDC_DATASET_ID}"
            and dataset.revision == RELEASE_ID
            and tuple(dataset.license.acceptance_ids) == tuple(sorted(REQUIRED_TERMS))
            and artifact.filename == ARCHIVE_FILENAME
            and artifact.size_bytes == ARCHIVE_SIZE_BYTES
            and checksum is not None
            and checksum.algorithm.value == "sha256"
            and checksum.digest == ARCHIVE_SHA256
            and checksum.source == "cv-dataset-stats"
        )
    except (AttributeError, IndexError, TypeError, ValueError):
        valid = False
    if not valid:
        raise SpsPreparationError("catalog contract failed")


def _git_marker_present(directory_fd: int) -> bool:
    """Recognize a Git worktree marker without following or logging its target."""

    marker_fd = -1
    try:
        try:
            marker = os.stat(".git", dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        if stat.S_ISREG(marker.st_mode):
            return marker.st_size > 0
        if stat.S_ISLNK(marker.st_mode) or not stat.S_ISDIR(marker.st_mode):
            return True
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        marker_fd = os.open(".git", flags, dir_fd=directory_fd)
        opened = os.fstat(marker_fd)
        if _identity(opened) != _identity(marker):
            raise BoundedReadError()
        try:
            os.stat("HEAD", dir_fd=marker_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True
    except BoundedReadError:
        raise
    except (OSError, OverflowError, ValueError):
        raise BoundedReadError() from None
    finally:
        if marker_fd >= 0:
            try:
                os.close(marker_fd)
            except OSError:
                pass


def _has_git_ancestor(directory: Path) -> bool:
    """Check each ancestor for a real worktree marker through no-follow FDs."""

    candidate = Path(os.path.abspath(directory))
    if not candidate.is_absolute():
        raise BoundedReadError()
    for ancestor in (candidate, *candidate.parents):
        with opened_directory_nofollow(ancestor) as (_stable, descriptor):
            if _git_marker_present(descriptor):
                return True
    return False


def _validated_output_path(path: Path | str) -> Path:
    supplied = Path(path).expanduser()
    if not supplied.is_absolute():
        raise SpsPreparationError("output prerequisite failed")
    candidate = Path(os.path.abspath(supplied))
    try:
        parent = candidate.parent.resolve(strict=True)
        inside_git = _has_git_ancestor(parent)
    except (OSError, RuntimeError, ValueError, BoundedReadError):
        raise SpsPreparationError("output prerequisite failed") from None
    if (
        candidate.name in {"", ".", ".."}
        or candidate == _REPO_ROOT
        or _REPO_ROOT in candidate.parents
        or inside_git
        or parent != candidate.parent
        or candidate.exists()
        or candidate.is_symlink()
    ):
        raise SpsPreparationError("output prerequisite failed")
    return candidate


def _hash_archive_descriptor(descriptor: int) -> str:
    """Hash exactly the pinned archive byte count through one open descriptor."""

    try:
        digest = hashlib.sha256()
        consumed = 0
        while consumed <= ARCHIVE_SIZE_BYTES:
            chunk = os.pread(
                descriptor,
                min(1024 * 1024, ARCHIVE_SIZE_BYTES + 1 - consumed),
                consumed,
            )
            if not chunk:
                break
            consumed += len(chunk)
            if consumed > ARCHIVE_SIZE_BYTES:
                raise SpsPreparationError("archive prerequisite failed")
            digest.update(chunk)
        if consumed != ARCHIVE_SIZE_BYTES:
            raise SpsPreparationError("archive prerequisite failed")
        return digest.hexdigest()
    except SpsPreparationError:
        raise
    except (OSError, OverflowError, ValueError):
        raise SpsPreparationError("archive prerequisite failed") from None


@contextmanager
def _verified_archive(path: Path | str) -> Iterator[tarfile.TarFile]:
    """Hash, parse, and recheck the archive through one open descriptor."""

    supplied = Path(path).expanduser()
    if not supplied.is_absolute():
        raise SpsPreparationError("archive prerequisite failed")
    candidate = Path(os.path.abspath(supplied))
    descriptor = -1
    handle: io.BufferedReader | None = None
    archive: tarfile.TarFile | None = None
    try:
        if (
            not candidate.is_absolute()
            or candidate == _REPO_ROOT
            or _REPO_ROOT in candidate.parents
            or _has_git_ancestor(candidate.parent)
        ):
            raise SpsPreparationError("archive prerequisite failed")
        before = candidate.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size != ARCHIVE_SIZE_BYTES
            or before.st_size > _MAX_ARCHIVE_BYTES
            or stat.S_IMODE(before.st_mode) & 0o077
            or (hasattr(os, "geteuid") and before.st_uid != os.geteuid())
            or candidate.resolve(strict=True) != candidate
        ):
            raise SpsPreparationError("archive prerequisite failed")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(candidate, flags)
        opened = os.fstat(descriptor)
        if _identity(opened) != _identity(before):
            raise SpsPreparationError("archive prerequisite failed")

        if _hash_archive_descriptor(descriptor) != ARCHIVE_SHA256:
            raise SpsPreparationError("archive prerequisite failed")
        os.lseek(descriptor, 0, os.SEEK_SET)
        handle = os.fdopen(descriptor, "rb", closefd=False)
        archive = tarfile.open(fileobj=handle, mode="r:gz")
        yield archive

        archive.close()
        archive = None
        if _hash_archive_descriptor(descriptor) != ARCHIVE_SHA256:
            raise SpsPreparationError("archive changed during preparation")
        handle.close()
        handle = None

        after = os.fstat(descriptor)
        current = candidate.lstat()
        if (
            _identity(after) != _identity(opened)
            or _identity(current) != _identity(opened)
            or candidate.resolve(strict=True) != candidate
        ):
            raise SpsPreparationError("archive changed during preparation")
    except SpsPreparationError:
        raise
    except (
        OSError,
        EOFError,
        tarfile.TarError,
        ValueError,
        OverflowError,
        BoundedReadError,
    ):
        raise SpsPreparationError("archive prerequisite failed") from None
    finally:
        if archive is not None:
            try:
                archive.close()
            except OSError:
                pass
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _safe_member_name(name: str) -> PurePosixPath:
    if not name or len(name) > 255 or "\\" in name or "\x00" in name:
        raise SpsPreparationError("archive layout failed")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or "/".join(path.parts) != name.rstrip("/")
    ):
        raise SpsPreparationError("archive layout failed")
    return path


def _scan_archive(archive: tarfile.TarFile) -> ArchiveLayout:
    members: dict[str, tarfile.TarInfo] = {}
    casefolded: set[str] = set()
    total_size = 0
    allowed_directories = {ARCHIVE_ROOT, f"{ARCHIVE_ROOT}/audios"}
    try:
        for count, member in enumerate(archive, start=1):
            if count > _MAX_ARCHIVE_MEMBERS:
                raise SpsPreparationError("archive layout failed")
            path = _safe_member_name(member.name)
            name = "/".join(path.parts)
            folded = name.casefold()
            if name in members or folded in casefolded:
                raise SpsPreparationError("archive layout failed")
            casefolded.add(folded)
            if member.isdir():
                if (
                    member.type != tarfile.DIRTYPE
                    or name not in allowed_directories
                    or member.size != 0
                ):
                    raise SpsPreparationError("archive layout failed")
                members[name] = member
                continue
            if (
                member.type not in {tarfile.REGTYPE, tarfile.AREGTYPE}
                or not member.isfile()
                or member.size <= 0
                or getattr(member, "sparse", None)
                or any("sparse" in str(key).casefold() for key in member.pax_headers)
            ):
                raise SpsPreparationError("archive layout failed")
            total_size += int(member.size)
            if total_size > _MAX_DECLARED_MEMBER_BYTES:
                raise SpsPreparationError("archive layout failed")
            if name in {_MAIN_MEMBER, _REPORTED_MEMBER, _QA_MEMBER, _README_MEMBER}:
                maximum = {
                    _MAIN_MEMBER: _MAX_MAIN_TSV_BYTES,
                    _REPORTED_MEMBER: 64 * 1024 * 1024,
                    _QA_MEMBER: 4 * 1024 * 1024,
                    _README_MEMBER: 1024 * 1024,
                }[name]
                if member.size > maximum:
                    raise SpsPreparationError("archive layout failed")
            elif name.startswith(_AUDIO_PREFIX):
                filename = name[len(_AUDIO_PREFIX) :]
                if (
                    "/" in filename
                    or _SAFE_AUDIO_RE.fullmatch(filename) is None
                    or member.size > _MAX_AUDIO_MEMBER_BYTES
                ):
                    raise SpsPreparationError("archive layout failed")
            else:
                raise SpsPreparationError("archive layout failed")
            members[name] = member
    except SpsPreparationError:
        raise
    except (OSError, tarfile.TarError, ValueError, OverflowError):
        raise SpsPreparationError("archive layout failed") from None

    try:
        return ArchiveLayout(
            members=members,
            main_tsv=members[_MAIN_MEMBER],
            reported_tsv=members[_REPORTED_MEMBER],
            qa_summary=members[_QA_MEMBER],
            readme=members[_README_MEMBER],
        )
    except KeyError:
        raise SpsPreparationError("archive layout failed") from None


def _read_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    *,
    maximum_bytes: int,
) -> bytes:
    try:
        extracted = archive.extractfile(member)
        if extracted is None:
            raise SpsPreparationError("archive member read failed")
        payload = extracted.read(maximum_bytes + 1)
        if len(payload) != member.size or len(payload) > maximum_bytes:
            raise SpsPreparationError("archive member read failed")
        return payload
    except SpsPreparationError:
        raise
    except (OSError, EOFError, tarfile.TarError, ValueError, OverflowError):
        raise SpsPreparationError("archive member read failed") from None


def _unsigned_int(value: str, *, maximum: int) -> int:
    if _UNSIGNED_INT_RE.fullmatch(value) is None:
        raise SpsPreparationError("metadata contract failed")
    parsed = int(value)
    if parsed <= 0 or parsed > maximum:
        raise SpsPreparationError("metadata contract failed")
    return parsed


def _validate_optional_integer(value: str) -> None:
    if not value:
        return
    if _SIGNED_INT_RE.fullmatch(value) is None or abs(int(value)) > 1_000_000_000:
        raise SpsPreparationError("metadata contract failed")


def _validate_optional_float(value: str) -> None:
    if not value:
        return
    try:
        parsed = float(value)
    except ValueError:
        raise SpsPreparationError("metadata contract failed") from None
    if not math.isfinite(parsed) or parsed < 0.0 or parsed > 100_000.0:
        raise SpsPreparationError("metadata contract failed")


def _quality_tags(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    values = tuple(value.split("|"))
    if (
        not values
        or len(values) > 16
        or len(values) != len(set(values))
        or any(_SAFE_TAG_RE.fullmatch(item) is None for item in values)
    ):
        raise SpsPreparationError("metadata contract failed")
    return values


def _parse_main_tsv(payload: bytes) -> ParsedMetadata:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeError:
        raise SpsPreparationError("metadata contract failed") from None
    if not text or "\x00" in text:
        raise SpsPreparationError("metadata contract failed")
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter="\t", strict=True)
    if tuple(reader.fieldnames or ()) != MAIN_TSV_FIELDS:
        raise SpsPreparationError("metadata contract failed")

    rows: list[SpsRow] = []
    seen_audio_ids: set[str] = set()
    seen_audio_files: set[str] = set()
    split_counts = {name: 0 for name in EXPECTED_SPLIT_COUNTS}
    split_duration_ms = {name: 0 for name in EXPECTED_SPLIT_COUNTS}
    split_speakers: dict[str, set[str]] = {
        "train": set(),
        "dev": set(),
        "test": set(),
        "unassigned": set(),
    }
    try:
        for raw in reader:
            if set(raw) != set(MAIN_TSV_FIELDS):
                raise SpsPreparationError("metadata contract failed")
            if any(
                not isinstance(value, str)
                or "\x00" in value
                or len(value) > _MAX_TEXT_FIELD_CHARS
                for value in raw.values()
            ):
                raise SpsPreparationError("metadata contract failed")
            client_id = raw["client_id"]
            audio_id = raw["audio_id"]
            audio_file = raw["audio_file"]
            prompt_id = raw["prompt_id"]
            transcription = raw["transcription"]
            language = raw["language"]
            split = raw["split"] or "unassigned"
            if (
                split not in split_counts
                or not client_id
                or len(client_id) > 512
                or not audio_id
                or len(audio_id) > 256
                or _SAFE_AUDIO_RE.fullmatch(audio_file) is None
                or len(prompt_id) > 256
                or audio_id in seen_audio_ids
                or audio_file in seen_audio_files
                or len(language) > 32
            ):
                raise SpsPreparationError("metadata contract failed")
            duration_ms = _unsigned_int(raw["duration_ms"], maximum=10 * 60 * 1000)
            for field_name in ("votes", "prompt_upvotes", "prompt_reports"):
                _validate_optional_integer(raw[field_name])
            _validate_optional_float(raw["char_per_sec"])
            if raw["is_edited"] not in {"", "0", "1"}:
                raise SpsPreparationError("metadata contract failed")
            tags = _quality_tags(raw["quality_tags"])
            seen_audio_ids.add(audio_id)
            seen_audio_files.add(audio_file)
            split_counts[split] += 1
            split_duration_ms[split] += duration_ms
            split_speakers[split].add(client_id)
            rows.append(
                SpsRow(
                    client_id=client_id,
                    audio_id=audio_id,
                    audio_file=audio_file,
                    prompt_id=prompt_id,
                    duration_ms=duration_ms,
                    transcription=transcription,
                    split=split,
                    language=language,
                    quality_tags=tags,
                )
            )
    except SpsPreparationError:
        raise
    except (csv.Error, KeyError, TypeError, ValueError, OverflowError):
        raise SpsPreparationError("metadata contract failed") from None

    if (
        len(rows) != EXPECTED_ROWS
        or sum(row.duration_ms for row in rows) != EXPECTED_DURATION_MS
        or split_counts != EXPECTED_SPLIT_COUNTS
        or split_duration_ms != EXPECTED_SPLIT_DURATION_MS
        or any(
            len(split_speakers[name]) != count
            for name, count in EXPECTED_SPLIT_SPEAKERS.items()
        )
        or len(set().union(*split_speakers.values())) != EXPECTED_TOTAL_SPEAKERS
    ):
        raise SpsPreparationError("metadata statistics failed")
    assigned = (split_speakers["train"], split_speakers["dev"], split_speakers["test"])
    if any(
        assigned[left] & assigned[right]
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        raise SpsPreparationError("metadata speaker split failed")
    return ParsedMetadata(
        rows=tuple(rows),
        split_speakers={key: frozenset(value) for key, value in split_speakers.items()},
        split_counts=dict(split_counts),
        split_duration_ms=dict(split_duration_ms),
    )


def duration_bucket(duration_ms: int) -> str | None:
    for name, lower, upper, inclusive_upper in DURATION_BUCKETS:
        if duration_ms >= lower and (
            duration_ms <= upper if inclusive_upper else duration_ms < upper
        ):
            return name
    return None


def _eligible_rows(metadata: ParsedMetadata) -> tuple[tuple[SpsRow, str], ...]:
    eligible: list[tuple[SpsRow, str]] = []
    for row in metadata.rows:
        bucket = duration_bucket(row.duration_ms)
        if (
            row.split != "test"
            or bucket is None
            or row.quality_tags
            or not row.transcription.strip()
            or len(row.transcription) > 4096
            or not normalize(row.transcription)
            or _BRACKET_ANNOTATION_RE.search(row.transcription) is not None
        ):
            continue
        eligible.append((row, bucket))
    return tuple(eligible)


def _selection_recipe(policy: SelectionPolicy) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "release_id": RELEASE_ID,
        "locale": LOCALE,
        "policy_sha256": selection_policy_sha256(policy),
        "split": "test",
        "archive_locale": "en",
        "language_column_filter": None,
        "require_empty_quality_tags": True,
        "exclude_bracket_annotations": True,
        "require_normalized_reference": True,
        "maximum_reference_chars": 4096,
        "duration_buckets_ms": [
            {
                "id": name,
                "minimum_inclusive": lower,
                "maximum": upper,
                "maximum_inclusive": inclusive,
                "quota": SELECTION_COUNT_PER_BUCKET,
            }
            for name, lower, upper, inclusive in DURATION_BUCKETS
        ],
        "distinct_selected_speakers": True,
        "forbidden_speaker_splits": ["train", "dev"],
        "literal_reference_preserved": True,
        "scoring_normalizer": "core.wer.normalize-v1",
    }


def _salted_digest(kind: str, value: str) -> str:
    return hashlib.sha256(
        RELEASE_ID.encode("ascii")
        + b"\0"
        + kind.encode("ascii")
        + b"\0"
        + value.encode("utf-8")
    ).hexdigest()


def _digest_string_set(kind: str, values: frozenset[str] | set[str]) -> str:
    payload = [sorted(_salted_digest(kind, value) for value in values)]
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _select_rows(
    metadata: ParsedMetadata,
) -> tuple[tuple[SpsRow, str], str, str, dict[str, object]]:
    dataset = dataset_by_id(DATASET_ID)
    policy = dataset.selection
    eligible = _eligible_rows(metadata)
    by_audio_id = {row.audio_id: (row, bucket) for row, bucket in eligible}
    selector_rows = [row.selector_row(bucket) for row, bucket in eligible]
    forbidden = frozenset(
        metadata.split_speakers["train"] | metadata.split_speakers["dev"]
    )
    try:
        chosen = select_subset_rows(
            policy,
            selector_rows,
            forbidden_speaker_ids=forbidden,
        )
    except Exception:
        raise SpsPreparationError("selection contract failed") from None
    try:
        selected = tuple(by_audio_id[str(item["audio_id"])] for item in chosen)
    except (KeyError, TypeError):
        raise SpsPreparationError("selection contract failed") from None
    counts = {name: 0 for name, *_rest in DURATION_BUCKETS}
    selected_speakers = set()
    for row, bucket in selected:
        counts[bucket] += 1
        selected_speakers.add(row.client_id)
    if (
        len(selected) != SELECTION_COUNT
        or len(selected_speakers) != SELECTION_COUNT
        or any(value != SELECTION_COUNT_PER_BUCKET for value in counts.values())
    ):
        raise SpsPreparationError("selection contract failed")

    recipe = _selection_recipe(policy)
    recipe_sha256 = hashlib.sha256(_canonical_json_bytes(recipe)).hexdigest()
    selected_sha256 = selected_rows_sha256(policy, chosen)
    eligible_binding = sorted(
        (
            {
                "identity_sha256": _salted_digest(
                    "row", f"{row.audio_id}\0{row.audio_file}\0{row.client_id}"
                ),
                "speaker_sha256": _salted_digest("speaker", row.client_id),
                "transcription_sha256": hashlib.sha256(
                    row.transcription.encode("utf-8")
                ).hexdigest(),
                "duration_ms": row.duration_ms,
                "duration_bucket": bucket,
            }
            for row, bucket in eligible
        ),
        key=lambda item: str(item["identity_sha256"]),
    )
    eligible_sha256 = hashlib.sha256(
        _canonical_json_bytes(eligible_binding)
    ).hexdigest()
    selection_summary = {
        "recipe": recipe,
        "recipe_sha256": recipe_sha256,
        "eligible_rows": len(eligible),
        "eligible_set_sha256": eligible_sha256,
        "selected_rows_sha256": selected_sha256,
        "selected_cases": len(selected),
        "selected_distinct_speakers": len(selected_speakers),
        "bucket_counts": counts,
    }
    return selected, recipe_sha256, selected_sha256, selection_summary


def _verify_decoded_audio(decoded: DecodedAudio, *, duration_ms: int) -> None:
    if (
        type(decoded.samples) is not int
        or decoded.samples <= 0
        or decoded.samples * 4 > MAX_PCM_BYTES
        or len(decoded.raw_f32le) != decoded.samples * 4
    ):
        raise SpsPreparationError("audio decode failed")
    values = array("f")
    try:
        values.frombytes(decoded.raw_f32le)
    except (BufferError, ValueError):
        raise SpsPreparationError("audio decode failed") from None
    if sys.byteorder != "little":
        values.byteswap()
    observed_ms = decoded.samples * 1000.0 / 16_000.0
    if (
        len(values) != decoded.samples
        or any(not math.isfinite(value) for value in values)
        or abs(observed_ms - duration_ms) > MAX_DURATION_DELTA_MS
    ):
        raise SpsPreparationError("audio decode failed")


def _decode_mp3_pyav(payload: bytes) -> DecodedAudio:
    """Decode one allowlisted MP3 strictly through PyAV and its resampler."""

    try:
        import av
        import numpy as np

        chunks: list[bytes] = []
        total_samples = 0
        converted_frames = 0

        def append_converted(converted: object) -> None:
            nonlocal total_samples, converted_frames
            converted_frames += 1
            sample_count = getattr(converted, "samples", None)
            sample_rate = getattr(converted, "sample_rate", None)
            audio_format = getattr(getattr(converted, "format", None), "name", None)
            layout = getattr(getattr(converted, "layout", None), "name", None)
            if (
                converted_frames > 20_000
                or type(sample_count) is not int
                or sample_count <= 0
                or sample_rate != 16_000
                or audio_format != "flt"
                or layout != "mono"
                or total_samples + sample_count > MAX_PCM_BYTES // 4
            ):
                raise SpsPreparationError("audio decode failed")
            values = converted.to_ndarray()
            if (
                not isinstance(values, np.ndarray)
                or values.dtype != np.dtype("float32")
                or values.shape != (1, sample_count)
                or not values.flags.c_contiguous
                or not np.isfinite(values).all()
            ):
                raise SpsPreparationError("audio decode failed")
            raw = values.reshape(-1).astype("<f4", copy=False).tobytes(order="C")
            if len(raw) != sample_count * 4:
                raise SpsPreparationError("audio decode failed")
            chunks.append(raw)
            total_samples += sample_count

        with av.open(io.BytesIO(payload), mode="r", format="mp3") as container:
            streams = [stream for stream in container.streams if stream.type == "audio"]
            if len(streams) != 1 or any(
                stream.type != "audio" for stream in container.streams
            ):
                raise SpsPreparationError("audio decode failed")
            stream = streams[0]
            stream.codec_context.thread_count = 1
            if stream.codec_context.name not in {"mp3", "mp3float"}:
                raise SpsPreparationError("audio decode failed")
            resampler = av.AudioResampler(format="flt", layout="mono", rate=16_000)
            for frame in container.decode(stream):
                for converted in resampler.resample(frame):
                    append_converted(converted)
            for converted in resampler.resample(None):
                append_converted(converted)
        if not chunks or total_samples <= 0:
            raise SpsPreparationError("audio decode failed")
        raw = b"".join(chunks)
        if len(raw) != total_samples * 4 or len(raw) > MAX_PCM_BYTES:
            raise SpsPreparationError("audio decode failed")
        return DecodedAudio(raw, total_samples)
    except SpsPreparationError:
        raise
    except Exception:
        raise SpsPreparationError("audio decode failed") from None


def _module_digest(path_value: object) -> str:
    if not isinstance(path_value, str):
        raise SpsPreparationError("decoder provenance failed")
    try:
        return hash_regular_bounded(
            Path(path_value).resolve(strict=True),
            maximum_bytes=128 * 1024 * 1024,
            expected_bytes=Path(path_value).resolve(strict=True).stat().st_size,
        ).sha256
    except (OSError, BoundedReadError):
        raise SpsPreparationError("decoder provenance failed") from None


def _runtime_tree_digest(root: Path) -> dict[str, object]:
    """Bind a package/bundled-library tree without recording its local path."""

    root_descriptor = -1
    try:
        stable_root = root.resolve(strict=True)
        if stable_root != root or not stable_root.is_dir():
            raise SpsPreparationError("decoder provenance failed")
        root_before = stable_root.lstat()
        directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        directory_flags |= getattr(os, "O_DIRECTORY", 0)
        directory_flags |= getattr(os, "O_NOFOLLOW", 0)
        file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        file_flags |= getattr(os, "O_NOFOLLOW", 0)
        root_descriptor = os.open(stable_root, directory_flags)
        root_opened = os.fstat(root_descriptor)
        if not stat.S_ISDIR(root_opened.st_mode) or _identity(root_opened) != _identity(
            root_before
        ):
            raise SpsPreparationError("decoder provenance failed")

        rows: list[dict[str, object]] = []
        total_bytes = 0
        entries_seen = 0

        def walk(directory_fd: int, parts: tuple[str, ...], depth: int) -> None:
            nonlocal entries_seen, total_bytes
            if depth > _MAX_RUNTIME_TREE_DEPTH:
                raise SpsPreparationError("decoder provenance failed")
            names: list[str] = []
            with os.scandir(directory_fd) as entries:
                for entry in entries:
                    entries_seen += 1
                    if entries_seen > _MAX_RUNTIME_TREE_ENTRIES:
                        raise SpsPreparationError("decoder provenance failed")
                    names.append(entry.name)
            names.sort()
            for name in names:
                metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                relative_parts = (*parts, name)
                if stat.S_ISLNK(metadata.st_mode):
                    raise SpsPreparationError("decoder provenance failed")
                if stat.S_ISDIR(metadata.st_mode):
                    child = os.open(name, directory_flags, dir_fd=directory_fd)
                    try:
                        opened = os.fstat(child)
                        if _identity(opened) != _identity(metadata):
                            raise SpsPreparationError("decoder provenance failed")
                        walk(child, relative_parts, depth + 1)
                        after = os.fstat(child)
                        current = os.stat(
                            name,
                            dir_fd=directory_fd,
                            follow_symlinks=False,
                        )
                        if _identity(after) != _identity(opened) or _identity(
                            current
                        ) != _identity(opened):
                            raise SpsPreparationError("decoder provenance failed")
                    finally:
                        os.close(child)
                    continue
                if not stat.S_ISREG(metadata.st_mode):
                    raise SpsPreparationError("decoder provenance failed")

                relative = PurePosixPath(*relative_parts).as_posix()
                if (
                    metadata.st_size == 0
                    or name.endswith(".pyc")
                    or "__pycache__" in relative_parts
                ):
                    continue
                if metadata.st_size < 0 or metadata.st_size > 512 * 1024 * 1024:
                    raise SpsPreparationError("decoder provenance failed")
                if total_bytes + metadata.st_size > _MAX_RUNTIME_TREE_BYTES:
                    raise SpsPreparationError("decoder provenance failed")

                descriptor = os.open(name, file_flags, dir_fd=directory_fd)
                try:
                    opened = os.fstat(descriptor)
                    if not stat.S_ISREG(opened.st_mode) or _identity(
                        opened
                    ) != _identity(metadata):
                        raise SpsPreparationError("decoder provenance failed")
                    digest = hashlib.sha256()
                    consumed = 0
                    while consumed <= metadata.st_size:
                        chunk = os.read(
                            descriptor,
                            min(1024 * 1024, metadata.st_size + 1 - consumed),
                        )
                        if not chunk:
                            break
                        consumed += len(chunk)
                        if consumed > metadata.st_size:
                            raise SpsPreparationError("decoder provenance failed")
                        digest.update(chunk)
                    after = os.fstat(descriptor)
                    current = os.stat(
                        name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                    if (
                        consumed != metadata.st_size
                        or _identity(after) != _identity(opened)
                        or _identity(current) != _identity(opened)
                    ):
                        raise SpsPreparationError("decoder provenance failed")
                finally:
                    os.close(descriptor)
                total_bytes += consumed
                rows.append(
                    {
                        "relative_name": relative,
                        "size_bytes": consumed,
                        "sha256": digest.hexdigest(),
                    }
                )

        walk(root_descriptor, (), 0)
        root_after = os.fstat(root_descriptor)
        root_current = stable_root.lstat()
        if (
            _identity(root_after) != _identity(root_opened)
            or _identity(root_current) != _identity(root_opened)
            or stable_root.resolve(strict=True) != stable_root
        ):
            raise SpsPreparationError("decoder provenance failed")
        if not rows:
            raise SpsPreparationError("decoder provenance failed")
        return {
            "tree_sha256": hashlib.sha256(_canonical_json_bytes(rows)).hexdigest(),
            "files": len(rows),
            "bytes": total_bytes,
        }
    except SpsPreparationError:
        raise
    except (OSError, RuntimeError, ValueError, BoundedReadError):
        raise SpsPreparationError("decoder provenance failed") from None
    finally:
        if root_descriptor >= 0:
            try:
                os.close(root_descriptor)
            except OSError:
                pass


def _pyav_decoder_provenance() -> dict[str, object]:
    try:
        import av
        import numpy as np

        pyav_root = Path(str(av.__file__)).resolve(strict=True).parent
        numpy_root = Path(str(np.__file__)).resolve(strict=True).parent
        pyav_libraries = pyav_root.parent / "av.libs"
        numpy_libraries = numpy_root.parent / "numpy.libs"
        if not pyav_libraries.is_dir() or not numpy_libraries.is_dir():
            raise SpsPreparationError("decoder provenance failed")
        return {
            "contract": "direct-pyav-mp3-mono-f32le-16000-v1",
            "pyav_version": str(av.__version__),
            "pyav_package": _runtime_tree_digest(pyav_root),
            "pyav_bundled_ffmpeg": _runtime_tree_digest(pyav_libraries),
            "numpy_version": str(np.__version__),
            "numpy_package": _runtime_tree_digest(numpy_root),
            "numpy_bundled_libraries": _runtime_tree_digest(numpy_libraries),
            "ffmpeg_library_versions": {
                str(name): [int(part) for part in version]
                for name, version in sorted(av.library_versions.items())
            },
            "python_implementation": sys.implementation.name,
            "python_version": platform.python_version(),
            "platform_system": platform.system(),
            "platform_machine": platform.machine(),
            "decoder_threads": 1,
        }
    except SpsPreparationError:
        raise
    except Exception:
        raise SpsPreparationError("decoder provenance failed") from None


def _preparer_code_provenance() -> dict[str, object]:
    files = (
        "core/wer.py",
        "tools/prepare_common_voice_spontaneous_v4.py",
        "tools/public_voice_eval_matrix.py",
        "tools/streaming_stt/bounded_io.py",
        "tools/streaming_stt/corpus.py",
        "tools/streaming_stt/corpus_writer.py",
        "tools/streaming_stt/protocol.py",
    )
    return {
        "contract": "common-voice-spontaneous-v4-private-preparer-v3",
        "files": {
            relative: _module_digest(str(_REPO_ROOT / relative)) for relative in files
        },
    }


def _test_decoder_binding(value: TestDecoderInjection) -> dict[str, object]:
    if (
        type(value) is not TestDecoderInjection
        or not callable(value.decoder)
        or _SAFE_TAG_RE.fullmatch(value.fixture_id) is None
    ):
        raise SpsPreparationError("decoder provenance failed")
    return {
        "contract": "test-injected-f32le-v1",
        "fixture_id": value.fixture_id,
        "production_evidence": False,
    }


def _case_id(index: int, row: SpsRow) -> tuple[str, str]:
    identity = _salted_digest(
        "row", f"{row.audio_id}\0{row.audio_file}\0{row.client_id}"
    )
    return f"cvss4-en-{index:02d}-{identity[:12]}", identity


def prepare_common_voice_spontaneous_v4(
    *,
    archive_path: Path | str,
    output_dir: Path | str,
    accepted_terms: frozenset[str],
    api_receipt: MdcApiReceipt,
    test_decoder_injection: TestDecoderInjection | None = None,
) -> PreparedSpsCorpus:
    """Verify one private SPS archive and publish the deterministic P0 slice."""

    if not REQUIRED_TERMS <= accepted_terms:
        raise SpsPreparationError("terms acceptance failed")
    if test_decoder_injection is None:
        _validate_catalog_contract()
    _validate_api_receipt(api_receipt)
    destination = _validated_output_path(output_dir)
    preparer_binding = _preparer_code_provenance()
    if test_decoder_injection is None:
        selected_decoder = _decode_mp3_pyav
        bound_decoder = _pyav_decoder_provenance()
    else:
        bound_decoder = _test_decoder_binding(test_decoder_injection)
        selected_decoder = test_decoder_injection.decoder

    with _verified_archive(archive_path) as archive:
        layout = _scan_archive(archive)
        main_tsv = _read_member(
            archive,
            layout.main_tsv,
            maximum_bytes=_MAX_MAIN_TSV_BYTES,
        )
        metadata_sha256 = hashlib.sha256(main_tsv).hexdigest()
        metadata = _parse_main_tsv(main_tsv)
        selected, recipe_sha256, _selected_sha256, selection = _select_rows(metadata)

        published_cases: list[CorpusWriteCase] = []
        receipt_cases: list[dict[str, object]] = []
        total_pcm_bytes = 0
        selected_by_member = {
            f"{_AUDIO_PREFIX}{row.audio_file}": (position, row, bucket)
            for position, (row, bucket) in enumerate(selected)
        }
        if len(selected_by_member) != len(selected):
            raise SpsPreparationError("selection contract failed")
        try:
            selected_members = [
                (layout.members[name], position, row, bucket)
                for name, (position, row, bucket) in selected_by_member.items()
            ]
        except KeyError:
            raise SpsPreparationError("selected audio is unavailable") from None
        decoded_by_position: dict[int, tuple[CorpusWriteCase, dict[str, object]]] = {}
        for member, position, row, bucket in sorted(
            selected_members, key=lambda item: int(item[0].offset_data)
        ):
            source = _read_member(
                archive,
                member,
                maximum_bytes=_MAX_AUDIO_MEMBER_BYTES,
            )
            try:
                decoded = selected_decoder(source)
            except SpsPreparationError:
                raise
            except Exception:
                raise SpsPreparationError("audio decode failed") from None
            _verify_decoded_audio(decoded, duration_ms=row.duration_ms)
            total_pcm_bytes += len(decoded.raw_f32le)
            if total_pcm_bytes > MAX_CORPUS_BYTES:
                raise SpsPreparationError("decoded corpus budget failed")
            case_id, identity_sha256 = _case_id(position, row)
            pcm_sha256 = hashlib.sha256(decoded.raw_f32le).hexdigest()
            published = CorpusWriteCase(
                case_id=case_id,
                audio_bytes=decoded.raw_f32le,
                reference=row.transcription,
                tags=("public-voice", "cvss4-en", "spontaneous", "clean", bucket),
            )
            receipt_case = {
                "case_id": case_id,
                "identity_sha256": identity_sha256,
                "speaker_sha256": _salted_digest("speaker", row.client_id),
                "prompt_sha256": _salted_digest("prompt", row.prompt_id),
                "duration_ms": row.duration_ms,
                "duration_bucket": bucket,
                "quality_tags": list(row.quality_tags),
                "transcription_sha256": hashlib.sha256(
                    row.transcription.encode("utf-8")
                ).hexdigest(),
                "source_mp3_sha256": hashlib.sha256(source).hexdigest(),
                "source_mp3_bytes": len(source),
                "pcm_sha256": pcm_sha256,
                "pcm_samples": decoded.samples,
            }
            decoded_by_position[position] = (published, receipt_case)
        if set(decoded_by_position) != set(range(SELECTION_COUNT)):
            raise SpsPreparationError("selected audio is unavailable")
        for position in range(SELECTION_COUNT):
            published, receipt_case = decoded_by_position[position]
            published_cases.append(published)
            receipt_cases.append(receipt_case)

    forbidden = metadata.split_speakers["train"] | metadata.split_speakers["dev"]
    receipt = {
        "schema_version": 1,
        "kind": "common-voice-spontaneous-v4-preparation",
        "dataset": {
            "dataset_id": DATASET_ID,
            "mdc_dataset_id": MDC_DATASET_ID,
            "release_id": RELEASE_ID,
            "release_date": RELEASE_DATE,
            "mdc_published_date": MDC_PUBLISHED_DATE,
            "locale": LOCALE,
            "archive_filename": ARCHIVE_FILENAME,
            "archive_size_bytes": ARCHIVE_SIZE_BYTES,
            "archive_sha256": ARCHIVE_SHA256,
            "stats_git_commit": STATS_GIT_COMMIT,
            "stats_git_blob": STATS_GIT_BLOB,
            "stats_sha256": STATS_SHA256,
            "schema_registry_commit": SCHEMA_REGISTRY_COMMIT,
        },
        "accepted_terms": sorted(REQUIRED_TERMS),
        "usage_constraints": [
            "evaluation_only",
            "private_cache_only",
            "no_speaker_reidentification",
            "no_rehosting",
        ],
        "mdc_content_receipt": {
            "filename": api_receipt.filename,
            "size_bytes": api_receipt.size_bytes,
            "checksum_algorithm": api_receipt.checksum_algorithm,
            "checksum_sha256": api_receipt.checksum_sha256,
        },
        "metadata": {
            "main_tsv_sha256": metadata_sha256,
            "rows": len(metadata.rows),
            "duration_ms": sum(row.duration_ms for row in metadata.rows),
            "split_counts": dict(metadata.split_counts),
            "split_duration_ms": dict(metadata.split_duration_ms),
            "train_speakers_sha256": _digest_string_set(
                "speaker", metadata.split_speakers["train"]
            ),
            "dev_speakers_sha256": _digest_string_set(
                "speaker", metadata.split_speakers["dev"]
            ),
            "test_speakers_sha256": _digest_string_set(
                "speaker", metadata.split_speakers["test"]
            ),
            "forbidden_speakers_sha256": _digest_string_set("speaker", forbidden),
            "reported_tsv_content_read": False,
        },
        "selection": selection,
        "preparer": {
            **preparer_binding,
        },
        "decoder": bound_decoder,
        "cases": receipt_cases,
        "total_pcm_bytes": total_pcm_bytes,
        "evidence_scope": {
            "pcm_preparation_complete": True,
            "stt_model_run": False,
            "timing_scope": "prepared-pcm-only",
            "capture": False,
            "vad": False,
            "aec": False,
            "endpoint_ground_truth": False,
            "agent_tools": False,
            "live_latency": False,
            "model_training_disjoint": False,
        },
    }
    receipt_payload = _canonical_json_bytes(receipt)
    if len(receipt_payload) > _MAX_RECEIPT_BYTES:
        raise SpsPreparationError("receipt contract failed")
    receipt_sha256 = hashlib.sha256(receipt_payload).hexdigest()
    provenance = CorpusProvenance(
        kind="public-voice-v1",
        suite="cvss4-en-spontaneous",
        manifest_sha256=recipe_sha256,
        metadata_sha256=metadata_sha256,
        source_set_sha256=receipt_sha256,
    )
    if hashlib.sha256(receipt_payload).hexdigest() != provenance.source_set_sha256:
        raise SpsPreparationError("receipt contract failed")
    try:
        corpus = publish_private_corpus(
            cases=tuple(published_cases),
            provenance=provenance,
            output_dir=destination,
            purpose="Common Voice Spontaneous Speech v4 English clean P0 slice",
            sidecars={"preparation-receipt.json": receipt_payload},
        )
    except Exception:
        raise SpsPreparationError("publication failed") from None
    if corpus.provenance != provenance:
        raise SpsPreparationError("publication failed")
    if _preparer_code_provenance() != preparer_binding or (
        test_decoder_injection is None and _pyav_decoder_provenance() != bound_decoder
    ):
        raise SpsPreparationError("decoder provenance changed")
    return PreparedSpsCorpus(
        corpus=corpus,
        receipt_sha256=receipt_sha256,
        selection_recipe_sha256=recipe_sha256,
        metadata_sha256=metadata_sha256,
    )


def _safe_result(result: PreparedSpsCorpus) -> dict[str, object]:
    corpus = result.corpus
    if corpus.provenance is None:
        raise SpsPreparationError("publication failed")
    return {
        "ok": True,
        "dataset_id": DATASET_ID,
        "release_id": RELEASE_ID,
        "corpus_sha256": corpus.digest,
        "cases": len(corpus.cases),
        "audio_bytes": corpus.audio_bytes,
        "receipt_sha256": result.receipt_sha256,
        "selection_recipe_sha256": result.selection_recipe_sha256,
        "metadata_sha256": result.metadata_sha256,
        "provenance": corpus.provenance.as_dict(),
    }


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise SpsPreparationError("argument contract failed")


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description=(
            "Prepare a verified private Common Voice Spontaneous Speech v4 "
            "English P0 streaming corpus; no download, model, network, or "
            "audio-device execution."
        )
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--accept-term",
        action="append",
        default=[],
        help=(
            "repeat for CC0-1.0 and MDC-CONSUMER-TERMS-2026-05-06 after "
            "accepting the dataset terms through MDC"
        ),
    )
    parser.add_argument("--mdc-api-filename", required=True)
    parser.add_argument("--mdc-api-size-bytes", type=int, required=True)
    parser.add_argument(
        "--mdc-api-checksum",
        required=True,
        help="redacted MDC response checksum in the exact form sha256:<digest>",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        algorithm, separator, digest = args.mdc_api_checksum.partition(":")
        if separator != ":":
            raise SpsPreparationError("MDC content receipt failed")
        result = prepare_common_voice_spontaneous_v4(
            archive_path=args.archive,
            output_dir=args.output_dir,
            accepted_terms=frozenset(args.accept_term),
            api_receipt=MdcApiReceipt(
                filename=args.mdc_api_filename,
                size_bytes=args.mdc_api_size_bytes,
                checksum_algorithm=algorithm,
                checksum_sha256=digest,
            ),
        )
        payload: Mapping[str, object] = _safe_result(result)
        code = 0
    except Exception:  # noqa: BLE001 - never expose paths, IDs, or transcripts
        payload = _SAFE_ERROR
        code = 2
    print(
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
