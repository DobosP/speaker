"""Materialize the EdAcc part of the private conversation/STT fixture.

Production owns descriptor-safe extraction of the exact EdAcc v1.0 archive
into a new private ``edacc_v1.0`` directory.  It verifies the published
archive MD5, freezes a local SHA-256, validates the complete archive and
source layout, recomputes the official test metadata, binds every used file
back to the archive, selects 24 scoreable turns, and publishes the official
key-matched filtered-STM references in a private schema-v2 streaming corpus.

No download, recognizer, or audio device is used.  Literal references,
speaker identifiers, linguistic-background values, source paths, and audio
remain outside the preparation receipt.

An explicitly synthetic ``TestSourceInjection`` may retain an existing source
tree for deterministic tests.  It never constitutes production evidence.
"""

from __future__ import annotations

from array import array
import argparse
from contextlib import contextmanager
import csv
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tarfile
import wave
from typing import Iterator, Mapping, Sequence
import zlib

from core.wer import normalize
from tools import public_conversation_fixture as fixture
from tools.streaming_stt import corpus_writer
from tools.public_conversation_fixture import (
    DEFAULT_LOCK,
    FIXTURE_ID,
    RECEIPT_KIND,
    decoder_closure_sha256,
    load_fixture_lock,
)
from tools.public_voice_eval_matrix import (
    Dataset,
    SELECTION_SEED,
    SelectionAlgorithm,
    dataset_by_id,
    selected_rows_sha256,
    selection_policy_sha256,
    select_subset_rows,
)
from tools.streaming_stt.bounded_io import (
    hash_regular_bounded,
    opened_directory_nofollow,
    read_regular_bounded,
)
from tools.streaming_stt.corpus import (
    CorpusProvenance,
    LoadedCorpus,
    load_corpus,
    verify_corpus_snapshot,
)
from tools.streaming_stt.corpus_writer import CorpusWriteCase
from tools.streaming_stt.protocol import MAX_CORPUS_BYTES, MAX_PCM_BYTES


DATASET_ID = "edacc_v1_test"
SOURCE_ID = "edacc-test"
ARCHIVE_FILENAME = "edacc_v1.0.tar.gz"
ARCHIVE_ROOT = "edacc_v1.0"
ARCHIVE_SIZE_BYTES = 5_916_732_170
ARCHIVE_MD5 = "146b4b8026b5d0ce9611667c708456b3"
DOI = "10.7488/ds/7914"
LICENSE_ID = "CC-BY-SA-4.0"
REQUIRED_TERMS = frozenset({LICENSE_ID})
DECODER_CONTRACT = "edacc-wav-pcm16-mono32k-box2-f32le16k-v1"
PREPARER_CONTRACT = "public-conversation-edacc-v1"
CORPUS_PURPOSE = "EdAcc v1.0 official test conversation fixture"
SOURCE_RECIPE_ALGORITHM = "hash_speaker_stratified_v1"
SOURCE_RECIPE_ELIGIBILITY = (
    "official_test_human_text",
    "raw_text_stm_agreement",
    "scoreable_filtered_stm",
    "no_actual_overlap",
    "no_ignore_segments",
    "no_alternative_references",
    "no_noise_only",
    "no_paralinguistic_only",
)

SOURCE_SAMPLE_RATE_HZ = 32_000
OUTPUT_SAMPLE_RATE_HZ = 16_000
SOURCE_CHANNELS = 1
SOURCE_SAMPLE_WIDTH_BYTES = 2
CASES_PER_STRATUM = 8
CASE_COUNT = 24
STRATA = ("clean", "disfluent", "overlap_adjacent")
MIN_DURATION_SECONDS = Decimal("0.50")
MAX_DURATION_SECONDS = Decimal("12.00")
OVERLAP_ADJACENCY_SECONDS = Decimal("0.50")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_METADATA_FILES = (
    "test/conv.list",
    "test/segments",
    "test/stm",
    "test/stm.filt",
    "test/text",
    "test/utt2spk",
    "linguistic_background.csv",
)
_METADATA_LIMITS = {
    "test/conv.list": 64 * 1024,
    "test/segments": 8 * 1024 * 1024,
    "test/stm": 16 * 1024 * 1024,
    "test/stm.filt": 16 * 1024 * 1024,
    "test/text": 16 * 1024 * 1024,
    "test/utt2spk": 8 * 1024 * 1024,
    "linguistic_background.csv": 2 * 1024 * 1024,
}
_ARCHIVE_DIRECTORIES = frozenset(
    {
        ARCHIVE_ROOT,
        f"{ARCHIVE_ROOT}/data",
        f"{ARCHIVE_ROOT}/dev",
        f"{ARCHIVE_ROOT}/test",
    }
)
_ARCHIVE_ROOT_FILE_LIMITS = {
    "evaluate.sh": 1024 * 1024,
    "glm": 2 * 1024 * 1024,
    "linguistic_background.csv": _METADATA_LIMITS[
        "linguistic_background.csv"
    ],
}
_ARCHIVE_SPLIT_FILE_LIMITS = {
    "company.ctm": 64 * 1024 * 1024,
    "conv.list": _METADATA_LIMITS["test/conv.list"],
    "segments": _METADATA_LIMITS["test/segments"],
    "stm": _METADATA_LIMITS["test/stm"],
    "stm.filt": _METADATA_LIMITS["test/stm.filt"],
    "text": _METADATA_LIMITS["test/text"],
    "utt2spk": _METADATA_LIMITS["test/utt2spk"],
}
_ARCHIVE_REQUIRED_FILES = frozenset(
    {
        *(f"{ARCHIVE_ROOT}/{name}" for name in _ARCHIVE_ROOT_FILE_LIMITS),
        *(
            f"{ARCHIVE_ROOT}/{split}/{name}"
            for split in ("dev", "test")
            for name in _ARCHIVE_SPLIT_FILE_LIMITS
        ),
    }
)
_MAX_ARCHIVE_BYTES = 8 * 1024 * 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 10_000
_MAX_AUDIO_BYTES = 2 * 1024 * 1024 * 1024
_MAX_ARCHIVE_AUDIO_FILES = 1_024
_MAX_DECLARED_ARCHIVE_BYTES = 16 * 1024 * 1024 * 1024
_MAX_ARCHIVE_METADATA_BYTES = 256 * 1024 * 1024
_MAX_TAR_TRAILING_ZERO_BYTES = 1024 * 1024
_MAX_TAR_STREAM_BYTES = (
    _MAX_DECLARED_ARCHIVE_BYTES
    + (_MAX_ARCHIVE_MEMBERS + 32) * 1024
    + _MAX_TAR_TRAILING_ZERO_BYTES
)
_MAX_TAR_NAME_BYTES = 255
_MAX_GZIP_EXPANSION_RATIO = 64
_GZIP_EXPANSION_FLOOR_BYTES = 64 * 1024 * 1024
_ARCHIVE_IO_CHUNK_BYTES = 1024 * 1024
_MAX_ROWS = 100_000
_MAX_LINE_CHARS = 32_768
_MAX_REFERENCE_CHARS = 4_096
_MAX_SOURCE_REFERENCE_CHARS = 32_000
_MAX_RECEIPT_BYTES = 256 * 1024
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}\Z")
_SAFE_FIXTURE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}\Z")
_SAFE_TAR_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}\Z")
_SAFE_ARCHIVE_AUDIO_RE = re.compile(r"EDACC-C[0-9]{2,4}\.wav\Z")
_MD5_RE = re.compile(r"[0-9a-f]{32}\Z")
_SPECIAL_TOKEN_RE = re.compile(r"<[^<>\r\n]{1,80}>")
_ALTERNATIVE_REFERENCE_RE = re.compile(r"[{}\[\]()]|(?:^|\s)/|/(?:\s|$)")
_SPEAKER_KEY_RE = re.compile(
    r"[A-Za-z]+-C0*(?P<conversation>[0-9]+)-(?P<side>[AB])\Z",
    re.IGNORECASE,
)
_PARTICIPANT_KEY_RE = re.compile(
    r"[A-Za-z]+-C0*(?P<conversation>[0-9]+)P(?P<participant>[12AB])\Z",
    re.IGNORECASE,
)
_PARTICIPANT_FIELDS = (
    "FINAL-Participant_ID",
    "Final-Participant_ID",
    "participant_id",
    "speaker_id",
)
_ACCENT_FIELDS = (
    "How would you describe your accent in English? (e.g. Italian, Glaswegian)",
    "raw_accent",
    "accent",
)
_FILLED_PAUSES = frozenset(
    {
        "ah",
        "eh",
        "er",
        "erm",
        "hm",
        "hmm",
        "mhm",
        "mm",
        "uh",
        "um",
    }
)
_PRIVATE_RECEIPT_PRIVACY = {
    "contains_audio": False,
    "contains_transcripts": False,
    "contains_local_paths": False,
    "contains_raw_identifiers": False,
}
_RECEIPT_EVIDENCE_SCOPE = {
    "development_only": True,
    "hard_wer_requires_unambiguous_nonoverlap": True,
    "ambiguous_overlap": "diagnostic_only",
    "capture": False,
    "vad": False,
    "aec": False,
    "agent_tools": False,
    "live_latency": False,
    "default_promotion": False,
    "pcm_preparation_complete": True,
    "stt_model_run": False,
}
_SAFE_ERROR = {
    "ok": False,
    "error": "edacc_conversation_fixture_prerequisites_unavailable",
}


class EdaccPreparationError(RuntimeError):
    """A detail-free acquisition, metadata, decode, or publication failure."""


@dataclass(frozen=True, slots=True)
class TestSourceInjection:
    """Explicitly non-production archive identity for synthetic source tests."""

    archive_size_bytes: int
    archive_md5: str
    fixture_id: str


@dataclass(frozen=True, slots=True)
class _FileBinding:
    path: Path = field(repr=False)
    size_bytes: int
    sha256: str
    identity: tuple[int, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class _ArchiveBinding:
    path: Path = field(repr=False)
    size_bytes: int
    md5: str
    sha256: str
    identity: tuple[int, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class _PrivatePaths:
    source_root: Path = field(repr=False)
    archive_path: Path = field(repr=False)
    output_dir: Path = field(repr=False)
    source_identity: tuple[int, ...] = field(repr=False)
    archive_identity: tuple[int, ...] = field(repr=False)
    output_parent_identity: tuple[int, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class _PrivateAcquisitionPaths:
    source_root: Path = field(repr=False)
    archive_path: Path = field(repr=False)
    output_dir: Path = field(repr=False)
    source_parent_identity: tuple[int, ...] = field(repr=False)
    archive_identity: tuple[int, ...] = field(repr=False)
    output_parent_identity: tuple[int, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class _RawTarMember:
    name: str
    is_directory: bool
    size_bytes: int


@dataclass(frozen=True, slots=True)
class _ArchiveMemberBinding:
    name: str
    is_directory: bool
    size_bytes: int
    sha256: str | None


@dataclass(frozen=True, slots=True)
class _ArchiveSnapshot:
    archive: _ArchiveBinding
    members: Mapping[str, _ArchiveMemberBinding] = field(repr=False)
    layout_sha256: str
    extraction_parent_identity: tuple[int, ...] | None = field(
        default=None,
        repr=False,
    )
    extracted_directory_identities: Mapping[str, tuple[int, ...]] | None = field(
        default=None,
        repr=False,
    )
    extracted_file_identities: Mapping[str, tuple[int, ...]] | None = field(
        default=None,
        repr=False,
    )


@dataclass(slots=True)
class _ExtractionTree:
    root: Path = field(repr=False)
    parent_descriptor: int = field(repr=False)
    directory_descriptors: Mapping[str, int] = field(repr=False)
    parent_identity: tuple[int, ...] = field(repr=False)
    directory_identities: Mapping[str, tuple[int, ...]] = field(repr=False)
    file_identities: dict[str, tuple[int, ...]] = field(
        default_factory=dict,
        repr=False,
    )


@dataclass(frozen=True, slots=True)
class _Segment:
    utterance_id: str = field(repr=False)
    recording_id: str = field(repr=False)
    speaker_id: str = field(repr=False)
    start: Decimal
    end: Decimal
    reference: str = field(repr=False)
    stm_reference: str = field(repr=False)
    score_reference: str = field(repr=False)
    accent: str | None = field(repr=False)
    actual_overlap: bool
    overlap_adjacent: bool
    disfluent: bool
    stratum: str

    def selector_row(self) -> dict[str, str]:
        return {
            "utterance_id": self.utterance_id,
            "speaker_id": self.speaker_id,
            "scoreability_stratum": self.stratum,
        }


@dataclass(frozen=True, slots=True)
class _ParsedSource:
    rows: tuple[_Segment, ...] = field(repr=False)
    eligible_rows: tuple[_Segment, ...] = field(repr=False)
    metadata_sha256: str
    metadata_bindings: Mapping[str, _FileBinding] = field(repr=False)


@dataclass(frozen=True, slots=True)
class _DecodedAudio:
    raw_f32le: bytes = field(repr=False)
    samples: int


@dataclass(frozen=True, slots=True)
class PreparedEdaccCorpus:
    corpus: LoadedCorpus
    receipt_sha256: str
    metadata_sha256: str
    selected_rows_sha256: str
    production_evidence: bool


def _file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        stat.S_IMODE(value.st_mode),
        value.st_uid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_nlink,
    )


def _directory_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        stat.S_IMODE(value.st_mode),
        value.st_uid,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _directory_entry_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        stat.S_IMODE(value.st_mode),
        value.st_uid,
    )


def _owned_by_process(value: os.stat_result) -> bool:
    return not hasattr(os, "geteuid") or value.st_uid == os.geteuid()


def _private_source_root(value: os.stat_result) -> bool:
    return (
        stat.S_ISDIR(value.st_mode)
        and stat.S_IMODE(value.st_mode) == 0o700
        and _owned_by_process(value)
    )


def _private_archive_file(value: os.stat_result) -> bool:
    return (
        stat.S_ISREG(value.st_mode)
        and stat.S_IMODE(value.st_mode) == 0o600
        and value.st_nlink == 1
        and _owned_by_process(value)
    )


def _git_marker_present(directory_fd: int) -> bool:
    marker_fd = -1
    try:
        try:
            marker = os.stat(".git", dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        if stat.S_ISREG(marker.st_mode):
            if marker.st_size > 0:
                return True
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            marker_fd = os.open(".git", flags, dir_fd=directory_fd)
            opened = os.fstat(marker_fd)
            if (
                _file_identity(opened) != _file_identity(marker)
                or not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or os.read(marker_fd, 1)
            ):
                raise EdaccPreparationError()
            after = os.fstat(marker_fd)
            current_marker = os.stat(
                ".git",
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            if (
                _file_identity(after) != _file_identity(opened)
                or _file_identity(current_marker) != _file_identity(marker)
            ):
                raise EdaccPreparationError()
            return False
        if stat.S_ISLNK(marker.st_mode) or not stat.S_ISDIR(marker.st_mode):
            return True
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        marker_fd = os.open(".git", flags, dir_fd=directory_fd)
        opened = os.fstat(marker_fd)
        if _directory_identity(opened) != _directory_identity(marker):
            raise EdaccPreparationError()
        try:
            os.stat("HEAD", dir_fd=marker_fd, follow_symlinks=False)
        except FileNotFoundError:
            has_head = False
        else:
            has_head = True
        current_marker = os.stat(
            ".git",
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if (
            _directory_identity(os.fstat(marker_fd))
            != _directory_identity(opened)
            or _directory_identity(current_marker) != _directory_identity(marker)
        ):
            raise EdaccPreparationError()
        return has_head
    except EdaccPreparationError:
        raise
    except Exception:
        raise EdaccPreparationError() from None
    finally:
        if marker_fd >= 0:
            try:
                os.close(marker_fd)
            except OSError:
                pass


def _has_git_ancestor(directory: Path) -> bool:
    try:
        for ancestor in (directory, *directory.parents):
            with opened_directory_nofollow(ancestor) as (stable, descriptor):
                opened = os.fstat(descriptor)
                if (
                    stable != ancestor
                    or _directory_identity(ancestor.lstat())
                    != _directory_identity(opened)
                ):
                    raise EdaccPreparationError()
                has_marker = _git_marker_present(descriptor)
                if (
                    _directory_identity(os.fstat(descriptor))
                    != _directory_identity(opened)
                    or _directory_identity(ancestor.lstat())
                    != _directory_identity(opened)
                ):
                    raise EdaccPreparationError()
                if has_marker:
                    return True
        return False
    except EdaccPreparationError:
        raise
    except Exception:
        raise EdaccPreparationError() from None


def _absolute_canonical(value: Path | str, *, must_exist: bool) -> Path:
    supplied = Path(value).expanduser()
    if not supplied.is_absolute():
        raise EdaccPreparationError()
    candidate = Path(os.path.abspath(supplied))
    if supplied != candidate:
        raise EdaccPreparationError()
    try:
        resolved = (
            candidate.resolve(strict=True)
            if must_exist
            else candidate.parent.resolve(strict=True) / candidate.name
        )
    except Exception:
        raise EdaccPreparationError() from None
    if resolved != candidate:
        raise EdaccPreparationError()
    return candidate


def _validate_private_paths(
    *,
    source_root: Path | str,
    archive_path: Path | str,
    output_dir: Path | str,
    production: bool,
) -> _PrivatePaths:
    try:
        root = _absolute_canonical(source_root, must_exist=True)
        archive = _absolute_canonical(archive_path, must_exist=True)
        destination = _absolute_canonical(output_dir, must_exist=False)
        root_metadata = root.lstat()
        archive_metadata = archive.lstat()
        if (
            root.name != ARCHIVE_ROOT
            or not _private_source_root(root_metadata)
            or not _private_archive_file(archive_metadata)
            or destination.name in {"", ".", ".."}
            or os.path.lexists(destination)
            or destination == root
            or root in destination.parents
            or archive == destination
        ):
            raise EdaccPreparationError()
        with opened_directory_nofollow(
            root,
            require_private=True,
        ) as (bound_root, descriptor):
            if bound_root != root or _directory_identity(
                os.fstat(descriptor)
            ) != _directory_identity(root_metadata):
                raise EdaccPreparationError()
        with opened_directory_nofollow(archive.parent):
            pass
        with opened_directory_nofollow(destination.parent) as (
            bound_parent,
            parent_descriptor,
        ):
            parent_metadata = os.fstat(parent_descriptor)
            if (
                bound_parent != destination.parent
                or _directory_entry_identity(destination.parent.lstat())
                != _directory_entry_identity(parent_metadata)
            ):
                raise EdaccPreparationError()
        if production and (
            _has_git_ancestor(root)
            or _has_git_ancestor(archive.parent)
            or _has_git_ancestor(destination.parent)
        ):
            raise EdaccPreparationError()
        return _PrivatePaths(
            source_root=root,
            archive_path=archive,
            output_dir=destination,
            source_identity=_directory_identity(root_metadata),
            archive_identity=_file_identity(archive_metadata),
            output_parent_identity=_directory_entry_identity(parent_metadata),
        )
    except EdaccPreparationError:
        raise
    except Exception:
        raise EdaccPreparationError() from None


def _validate_private_acquisition_paths(
    *,
    source_root: Path | str,
    archive_path: Path | str,
    output_dir: Path | str,
    production: bool,
) -> _PrivateAcquisitionPaths:
    """Bind an absent extraction root and its exact private parent."""

    try:
        root = _absolute_canonical(source_root, must_exist=False)
        archive = _absolute_canonical(archive_path, must_exist=True)
        destination = _absolute_canonical(output_dir, must_exist=False)
        archive_metadata = archive.lstat()
        if (
            root.name != ARCHIVE_ROOT
            or os.path.lexists(root)
            or not _private_archive_file(archive_metadata)
            or destination.name in {"", ".", ".."}
            or os.path.lexists(destination)
            or destination == root
            or root in destination.parents
            or archive in {root, destination}
        ):
            raise EdaccPreparationError()
        with opened_directory_nofollow(archive.parent):
            pass
        with opened_directory_nofollow(
            root.parent,
            require_private=True,
        ) as (bound_source_parent, source_parent_descriptor):
            source_parent = os.fstat(source_parent_descriptor)
            if (
                bound_source_parent != root.parent
                or not _private_source_root(source_parent)
                or _directory_entry_identity(root.parent.lstat())
                != _directory_entry_identity(source_parent)
            ):
                raise EdaccPreparationError()
        with opened_directory_nofollow(destination.parent) as (
            bound_output_parent,
            output_parent_descriptor,
        ):
            output_parent = os.fstat(output_parent_descriptor)
            if (
                bound_output_parent != destination.parent
                or _directory_entry_identity(destination.parent.lstat())
                != _directory_entry_identity(output_parent)
            ):
                raise EdaccPreparationError()
        if production and (
            _has_git_ancestor(root.parent)
            or _has_git_ancestor(archive.parent)
            or _has_git_ancestor(destination.parent)
        ):
            raise EdaccPreparationError()
        return _PrivateAcquisitionPaths(
            source_root=root,
            archive_path=archive,
            output_dir=destination,
            source_parent_identity=_directory_entry_identity(source_parent),
            archive_identity=_file_identity(archive_metadata),
            output_parent_identity=_directory_entry_identity(output_parent),
        )
    except EdaccPreparationError:
        raise
    except Exception:
        raise EdaccPreparationError() from None


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
            + b"\n"
        )
    except (MemoryError, TypeError, UnicodeError, ValueError, OverflowError):
        raise EdaccPreparationError() from None


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value).rstrip(b"\n")).hexdigest()


def _salted_digest(seed: str, namespace: str, value: str) -> str:
    return hashlib.sha256(
        seed.encode("utf-8")
        + b"\0"
        + namespace.encode("ascii")
        + b"\0"
        + value.encode("utf-8")
    ).hexdigest()


def _safe_source_id(value: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise EdaccPreparationError()
    return value


def _decimal(value: str) -> Decimal:
    try:
        result = Decimal(value)
    except (InvalidOperation, ValueError):
        raise EdaccPreparationError() from None
    if not result.is_finite() or result < 0:
        raise EdaccPreparationError()
    return result


def _decode_text(data: bytes) -> str:
    try:
        result = data.decode("utf-8-sig")
    except UnicodeError:
        raise EdaccPreparationError() from None
    if "\x00" in result:
        raise EdaccPreparationError()
    return result


def _lines(data: bytes) -> tuple[str, ...]:
    values = tuple(_decode_text(data).splitlines())
    if (
        not values
        or len(values) > _MAX_ROWS
        or any(not row or len(row) > _MAX_LINE_CHARS for row in values)
    ):
        raise EdaccPreparationError()
    return values


def _module_sha256(relative: str) -> str:
    try:
        root = _REPO_ROOT.resolve(strict=True)
        path = (root / relative).resolve(strict=True)
        if root not in path.parents:
            raise EdaccPreparationError()
        snapshot = read_regular_bounded(path, maximum_bytes=4 * 1024 * 1024)
        return hashlib.sha256(snapshot.data).hexdigest()
    except Exception:
        raise EdaccPreparationError() from None


def _preparer_code_provenance(files: Sequence[str]) -> dict[str, object]:
    if "tools/prepare_edacc_conversation_fixture.py" not in files:
        raise EdaccPreparationError()
    return {
        "contract": PREPARER_CONTRACT,
        "files": {relative: _module_sha256(relative) for relative in files},
    }


def _validate_catalog_contract() -> None:
    try:
        dataset = dataset_by_id(DATASET_ID)
        artifact = dataset.artifacts[0]
        if (
            dataset.revision != DOI
            or dataset.license.license_id != LICENSE_ID
            or dataset.license.acceptance_ids != (LICENSE_ID,)
            or len(dataset.artifacts) != 1
            or artifact.artifact_id != "archive"
            or artifact.filename != ARCHIVE_FILENAME
            or artifact.size_bytes != ARCHIVE_SIZE_BYTES
            or artifact.checksum is None
            or artifact.checksum.algorithm.value != "md5"
            or artifact.checksum.digest != ARCHIVE_MD5
            or dataset.selection.algorithm
            is not SelectionAlgorithm.HASH_SPEAKER_STRATIFIED
            or dataset.selection.split != "test"
            or dataset.selection.identity_fields != ("utterance_id",)
            or dataset.selection.strata_fields != ("scoreability_stratum",)
            or dataset.selection.seed != SELECTION_SEED
            or dataset.selection.expected_examples != CASE_COUNT
            or dataset.selection.speaker_field != "speaker_id"
            or not dataset.selection.speaker_disjoint
        ):
            raise EdaccPreparationError()
    except EdaccPreparationError:
        raise
    except Exception:
        raise EdaccPreparationError() from None


def _expected_source_recipe(dataset: Dataset) -> dict[str, object]:
    return {
        "algorithm": SOURCE_RECIPE_ALGORITHM,
        "matrix_policy_sha256": selection_policy_sha256(dataset.selection),
        "split": "test",
        "seed": SELECTION_SEED,
        "eligibility": list(SOURCE_RECIPE_ELIGIBILITY),
        "strata": [
            {"id": stratum, "cases": CASES_PER_STRATUM} for stratum in STRATA
        ],
        "speaker_requirement": "exactly_24_distinct",
        "scoreability": "hard_wer_nonoverlap_only",
    }


def _validate_fixture_source_contract(
    source: object,
    *,
    dataset: Dataset,
) -> None:
    try:
        valid = (
            source.dataset_id == DATASET_ID
            and source.expected_cases == CASE_COUNT
            and source.preparer_contract == PREPARER_CONTRACT
            and source.decoder_contract == DECODER_CONTRACT
            and dict(source.selection_recipe) == _expected_source_recipe(dataset)
        )
    except (AttributeError, TypeError, ValueError):
        valid = False
    if not valid:
        raise EdaccPreparationError()


class _CompressedArchiveStream:
    """Bounded single-member gzip reader that hashes every compressed byte."""

    def __init__(self, descriptor: int, *, expected_size: int) -> None:
        self._descriptor = descriptor
        self._expected_size = expected_size
        self._decompressor = zlib.decompressobj(wbits=31)
        self._md5 = hashlib.md5(usedforsecurity=False)
        self._sha256 = hashlib.sha256()
        self._compressed_bytes = 0
        self._decompressed_bytes = 0
        self._raw_eof = False
        self._finished = False

    @property
    def compressed_bytes(self) -> int:
        return self._compressed_bytes

    @property
    def decompressed_bytes(self) -> int:
        return self._decompressed_bytes

    @property
    def md5(self) -> str:
        if not self._finished:
            raise EdaccPreparationError()
        return self._md5.hexdigest()

    @property
    def sha256(self) -> str:
        if not self._finished:
            raise EdaccPreparationError()
        return self._sha256.hexdigest()

    def _read_compressed(self) -> bytes:
        if self._raw_eof:
            return b""
        remaining = self._expected_size + 1 - self._compressed_bytes
        if remaining <= 0:
            raise EdaccPreparationError()
        chunk = os.read(
            self._descriptor,
            min(_ARCHIVE_IO_CHUNK_BYTES, remaining),
        )
        if not chunk:
            self._raw_eof = True
            return b""
        self._compressed_bytes += len(chunk)
        if self._compressed_bytes > self._expected_size:
            raise EdaccPreparationError()
        self._md5.update(chunk)
        self._sha256.update(chunk)
        return chunk

    def _record_decompressed(self, count: int) -> None:
        self._decompressed_bytes += count
        ratio_limit = max(
            _GZIP_EXPANSION_FLOOR_BYTES,
            self._compressed_bytes * _MAX_GZIP_EXPANSION_RATIO,
        )
        if (
            self._decompressed_bytes > _MAX_TAR_STREAM_BYTES
            or self._decompressed_bytes > ratio_limit
        ):
            raise EdaccPreparationError()

    def read(self, size: int) -> bytes:
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise EdaccPreparationError()
        if self._finished:
            return b""
        result = bytearray()
        while len(result) < size and not self._finished:
            compressed = self._decompressor.unconsumed_tail
            if not compressed:
                compressed = self._read_compressed()
                if not compressed:
                    if not self._decompressor.eof:
                        raise EdaccPreparationError()
                    self._finished = True
                    break
            produced = self._decompressor.decompress(
                compressed,
                max_length=min(
                    _ARCHIVE_IO_CHUNK_BYTES,
                    size - len(result),
                ),
            )
            if self._decompressor.unused_data:
                raise EdaccPreparationError()
            if produced:
                self._record_decompressed(len(produced))
                result.extend(produced)
            if self._decompressor.eof:
                if self._decompressor.unused_data:
                    raise EdaccPreparationError()
                if self._read_compressed():
                    raise EdaccPreparationError()
                if self._compressed_bytes != self._expected_size:
                    raise EdaccPreparationError()
                self._finished = True
            elif not produced and not self._decompressor.unconsumed_tail:
                continue
        return bytes(result)


def _read_archive_exact(stream: _CompressedArchiveStream, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        chunk = stream.read(min(_ARCHIVE_IO_CHUNK_BYTES, size - len(result)))
        if not chunk:
            raise EdaccPreparationError()
        result.extend(chunk)
    return bytes(result)


def _tar_octal(
    field_value: bytes,
    *,
    maximum: int,
    allow_empty: bool = False,
) -> int:
    if (
        not field_value
        or field_value[0] & 0x80
        or re.fullmatch(rb" *[0-7]+[\x00 ]*", field_value) is None
    ):
        if allow_empty and not field_value.strip(b"\x00 "):
            return 0
        raise EdaccPreparationError()
    digits = field_value.strip(b"\x00 ")
    try:
        result = int(digits, 8)
    except ValueError:
        raise EdaccPreparationError() from None
    if result < 0 or result > maximum:
        raise EdaccPreparationError()
    return result


def _tar_text(field_value: bytes, *, allow_empty: bool) -> str:
    nul = field_value.find(b"\0")
    if nul >= 0:
        if any(field_value[nul + 1 :]):
            raise EdaccPreparationError()
        value = field_value[:nul]
    else:
        value = field_value
    if not value and not allow_empty:
        raise EdaccPreparationError()
    try:
        return value.decode("ascii", errors="strict")
    except UnicodeError:
        raise EdaccPreparationError() from None


def _safe_tar_member_name(raw_name: str, *, is_directory: bool) -> str:
    if not raw_name or "\\" in raw_name or ":" in raw_name:
        raise EdaccPreparationError()
    if is_directory and raw_name.endswith("/"):
        name = raw_name[:-1]
    else:
        name = raw_name
    try:
        encoded = name.encode("ascii", errors="strict")
    except UnicodeError:
        raise EdaccPreparationError() from None
    parts = name.split("/")
    if (
        not encoded
        or len(encoded) > _MAX_TAR_NAME_BYTES
        or raw_name.startswith("/")
        or (not is_directory and raw_name.endswith("/"))
        or any(
            part in {"", ".", ".."}
            or _SAFE_TAR_COMPONENT_RE.fullmatch(part) is None
            for part in parts
        )
        or "/".join(parts) != name
    ):
        raise EdaccPreparationError()
    return name


def _parse_tar_header(header: bytes) -> _RawTarMember:
    if len(header) != 512 or header == b"\0" * 512:
        raise EdaccPreparationError()
    if any(header[345:512]):
        raise EdaccPreparationError()
    mode = _tar_octal(header[100:108], maximum=0o7777)
    _tar_octal(header[108:116], maximum=(1 << 31) - 1)
    _tar_octal(header[116:124], maximum=(1 << 31) - 1)
    size = _tar_octal(header[124:136], maximum=_MAX_DECLARED_ARCHIVE_BYTES)
    _tar_octal(header[136:148], maximum=(1 << 63) - 1)
    checksum = _tar_octal(header[148:156], maximum=512 * 255)
    if mode & ~0o777 or checksum != (
        sum(header[:148]) + 8 * ord(" ") + sum(header[156:])
    ):
        raise EdaccPreparationError()
    magic_version = (header[257:263], header[263:265])
    if magic_version not in {
        (b"\0" * 6, b"\0" * 2),
        (b"ustar\0", b"00"),
        (b"ustar ", b" \0"),
    }:
        raise EdaccPreparationError()
    if _tar_octal(header[329:337], maximum=(1 << 31) - 1, allow_empty=True):
        raise EdaccPreparationError()
    if _tar_octal(header[337:345], maximum=(1 << 31) - 1, allow_empty=True):
        raise EdaccPreparationError()
    raw_name = _tar_text(header[:100], allow_empty=False)
    link_name = _tar_text(header[157:257], allow_empty=True)
    member_type = header[156:157]
    if member_type in {tarfile.REGTYPE, tarfile.AREGTYPE}:
        is_directory = False
        if size <= 0 or link_name:
            raise EdaccPreparationError()
    elif member_type == tarfile.DIRTYPE:
        is_directory = True
        if size != 0 or link_name:
            raise EdaccPreparationError()
    else:
        raise EdaccPreparationError()
    name = _safe_tar_member_name(raw_name, is_directory=is_directory)
    try:
        parsed = tarfile.TarInfo.frombuf(
            header,
            encoding="ascii",
            errors="strict",
        )
    except Exception:
        raise EdaccPreparationError() from None
    if (
        parsed.name.rstrip("/") != name
        or parsed.size != size
        or parsed.type != member_type
    ):
        raise EdaccPreparationError()
    return _RawTarMember(
        name=name,
        is_directory=is_directory,
        size_bytes=size,
    )


def _regular_archive_target(name: str) -> tuple[str, str, int, bool]:
    parts = name.split("/")
    if len(parts) == 2 and parts[0] == ARCHIVE_ROOT:
        maximum = _ARCHIVE_ROOT_FILE_LIMITS.get(parts[1])
        if maximum is not None:
            return ARCHIVE_ROOT, parts[1], maximum, False
    if (
        len(parts) == 3
        and parts[0] == ARCHIVE_ROOT
        and parts[1] in {"dev", "test"}
    ):
        maximum = _ARCHIVE_SPLIT_FILE_LIMITS.get(parts[2])
        if maximum is not None:
            return f"{ARCHIVE_ROOT}/{parts[1]}", parts[2], maximum, False
    if (
        len(parts) == 3
        and parts[:2] == [ARCHIVE_ROOT, "data"]
        and _SAFE_ARCHIVE_AUDIO_RE.fullmatch(parts[2]) is not None
    ):
        return f"{ARCHIVE_ROOT}/data", parts[2], _MAX_AUDIO_BYTES, True
    raise EdaccPreparationError()


def _verify_extraction_directories(tree: _ExtractionTree) -> None:
    try:
        parent = os.fstat(tree.parent_descriptor)
        lexical_parent = tree.root.parent.lstat()
        if (
            _directory_entry_identity(parent) != tree.parent_identity
            or _directory_entry_identity(lexical_parent) != tree.parent_identity
            or tree.root.parent.resolve(strict=True) != tree.root.parent
        ):
            raise EdaccPreparationError()
        root_descriptor = tree.directory_descriptors[ARCHIVE_ROOT]
        root_opened = os.fstat(root_descriptor)
        root_entry = os.stat(
            tree.root.name,
            dir_fd=tree.parent_descriptor,
            follow_symlinks=False,
        )
        root_lexical = tree.root.lstat()
        root_identity = tree.directory_identities[ARCHIVE_ROOT]
        if (
            _directory_entry_identity(root_opened) != root_identity
            or _directory_entry_identity(root_entry) != root_identity
            or _directory_entry_identity(root_lexical) != root_identity
            or not _private_source_root(root_opened)
            or not _private_source_root(root_entry)
            or not _private_source_root(root_lexical)
            or tree.root.resolve(strict=True) != tree.root
        ):
            raise EdaccPreparationError()
        for child in ("data", "dev", "test"):
            name = f"{ARCHIVE_ROOT}/{child}"
            descriptor = tree.directory_descriptors[name]
            opened = os.fstat(descriptor)
            entry = os.stat(
                child,
                dir_fd=root_descriptor,
                follow_symlinks=False,
            )
            lexical = (tree.root / child).lstat()
            identity = tree.directory_identities[name]
            if (
                _directory_entry_identity(opened) != identity
                or _directory_entry_identity(entry) != identity
                or _directory_entry_identity(lexical) != identity
                or not _private_source_root(opened)
                or not _private_source_root(entry)
                or not _private_source_root(lexical)
                or (tree.root / child).resolve(strict=True) != tree.root / child
            ):
                raise EdaccPreparationError()
    except EdaccPreparationError:
        raise
    except Exception:
        raise EdaccPreparationError() from None


@contextmanager
def _new_extraction_tree(
    root: Path,
    *,
    expected_parent_identity: tuple[int, ...],
) -> Iterator[_ExtractionTree]:
    descriptors: dict[str, int] = {}
    try:
        with opened_directory_nofollow(
            root.parent,
            require_private=True,
        ) as (bound_parent, parent_descriptor):
            parent = os.fstat(parent_descriptor)
            if (
                bound_parent != root.parent
                or _directory_entry_identity(parent) != expected_parent_identity
                or _directory_entry_identity(root.parent.lstat())
                != expected_parent_identity
                or os.path.lexists(root)
            ):
                raise EdaccPreparationError()
            os.mkdir(root.name, mode=0o700, dir_fd=parent_descriptor)
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_DIRECTORY", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            root_descriptor = os.open(root.name, flags, dir_fd=parent_descriptor)
            descriptors[ARCHIVE_ROOT] = root_descriptor
            os.fchmod(root_descriptor, 0o700)
            for child in ("data", "dev", "test"):
                os.mkdir(child, mode=0o700, dir_fd=root_descriptor)
                child_descriptor = os.open(
                    child,
                    flags,
                    dir_fd=root_descriptor,
                )
                os.fchmod(child_descriptor, 0o700)
                descriptors[f"{ARCHIVE_ROOT}/{child}"] = child_descriptor
            identities = {
                name: _directory_entry_identity(os.fstat(descriptor))
                for name, descriptor in descriptors.items()
            }
            tree = _ExtractionTree(
                root=root,
                parent_descriptor=parent_descriptor,
                directory_descriptors=dict(descriptors),
                parent_identity=expected_parent_identity,
                directory_identities=identities,
            )
            os.fsync(root_descriptor)
            os.fsync(parent_descriptor)
            _verify_extraction_directories(tree)
            yield tree
    except EdaccPreparationError:
        raise
    except Exception:
        raise EdaccPreparationError() from None
    finally:
        for descriptor in reversed(tuple(descriptors.values())):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    consumed = 0
    while consumed < len(view):
        written = os.write(descriptor, view[consumed:])
        if written <= 0:
            raise EdaccPreparationError()
        consumed += written


def _consume_archive_file(
    stream: _CompressedArchiveStream,
    member: _RawTarMember,
    *,
    tree: _ExtractionTree | None,
) -> str:
    parent_name, filename, maximum, _is_audio = _regular_archive_target(member.name)
    if member.size_bytes > maximum:
        raise EdaccPreparationError()
    descriptor = -1
    digest = hashlib.sha256()
    try:
        if tree is not None:
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(
                filename,
                flags,
                0o600,
                dir_fd=tree.directory_descriptors[parent_name],
            )
            os.fchmod(descriptor, 0o600)
        remaining = member.size_bytes
        while remaining:
            chunk = stream.read(min(_ARCHIVE_IO_CHUNK_BYTES, remaining))
            if not chunk:
                raise EdaccPreparationError()
            remaining -= len(chunk)
            digest.update(chunk)
            if descriptor >= 0:
                _write_all(descriptor, chunk)
        padding_size = (-member.size_bytes) % 512
        if padding_size and any(_read_archive_exact(stream, padding_size)):
            raise EdaccPreparationError()
        if descriptor >= 0:
            os.fsync(descriptor)
            opened = os.fstat(descriptor)
            if (
                not _private_output_file(opened)
                or opened.st_size != member.size_bytes
            ):
                raise EdaccPreparationError()
            os.lseek(descriptor, 0, os.SEEK_SET)
            persisted_digest = hashlib.sha256()
            persisted_remaining = member.size_bytes
            while persisted_remaining:
                persisted = os.read(
                    descriptor,
                    min(_ARCHIVE_IO_CHUNK_BYTES, persisted_remaining),
                )
                if not persisted:
                    raise EdaccPreparationError()
                persisted_remaining -= len(persisted)
                persisted_digest.update(persisted)
            if os.read(descriptor, 1):
                raise EdaccPreparationError()
            after_read = os.fstat(descriptor)
            entry = os.stat(
                filename,
                dir_fd=tree.directory_descriptors[parent_name],
                follow_symlinks=False,
            )
            if (
                _file_identity(after_read) != _file_identity(opened)
                or _file_identity(after_read) != _file_identity(entry)
                or not _private_output_file(after_read)
                or not _private_output_file(entry)
                or after_read.st_size != member.size_bytes
                or persisted_digest.hexdigest() != digest.hexdigest()
            ):
                raise EdaccPreparationError()
            tree.file_identities[member.name] = _file_identity(after_read)
        return digest.hexdigest()
    except EdaccPreparationError:
        raise
    except Exception:
        raise EdaccPreparationError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _finalize_extraction_tree(
    tree: _ExtractionTree,
    members: Mapping[str, _ArchiveMemberBinding],
) -> None:
    try:
        _verify_extraction_directories(tree)
        expected_root = {
            "data",
            "dev",
            "test",
            *_ARCHIVE_ROOT_FILE_LIMITS,
        }
        expected_split = set(_ARCHIVE_SPLIT_FILE_LIMITS)
        expected_audio = {
            name.rsplit("/", 1)[1]
            for name, binding in members.items()
            if not binding.is_directory and name.startswith(f"{ARCHIVE_ROOT}/data/")
        }
        root_descriptor = tree.directory_descriptors[ARCHIVE_ROOT]
        if set(os.listdir(root_descriptor)) != expected_root:
            raise EdaccPreparationError()
        for split in ("dev", "test"):
            if (
                set(os.listdir(tree.directory_descriptors[f"{ARCHIVE_ROOT}/{split}"]))
                != expected_split
            ):
                raise EdaccPreparationError()
        if (
            not expected_audio
            or set(os.listdir(tree.directory_descriptors[f"{ARCHIVE_ROOT}/data"]))
            != expected_audio
        ):
            raise EdaccPreparationError()
        expected_files = {name for name, value in members.items() if not value.is_directory}
        if set(tree.file_identities) != expected_files:
            raise EdaccPreparationError()
        for name in expected_files:
            parent_name, filename, _maximum, _is_audio = _regular_archive_target(name)
            current = os.stat(
                filename,
                dir_fd=tree.directory_descriptors[parent_name],
                follow_symlinks=False,
            )
            if (
                _file_identity(current) != tree.file_identities[name]
                or not _private_output_file(current)
                or current.st_size != members[name].size_bytes
            ):
                raise EdaccPreparationError()
        for descriptor in tree.directory_descriptors.values():
            os.fsync(descriptor)
        os.fsync(tree.parent_descriptor)
        _verify_extraction_directories(tree)
    except EdaccPreparationError:
        raise
    except Exception:
        raise EdaccPreparationError() from None


def _consume_archive(
    path: Path,
    *,
    expected_identity: tuple[int, ...],
    expected_size: int,
    expected_md5: str,
    extraction_root: Path | None = None,
    extraction_parent_identity: tuple[int, ...] | None = None,
) -> _ArchiveSnapshot:
    descriptor = -1
    tree_context = None
    try:
        before = path.lstat()
        if (
            _file_identity(before) != expected_identity
            or not _private_archive_file(before)
            or before.st_size != expected_size
            or expected_size <= 0
            or expected_size > _MAX_ARCHIVE_BYTES
            or _MD5_RE.fullmatch(expected_md5) is None
        ):
            raise EdaccPreparationError()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (
            _file_identity(opened) != expected_identity
            or not _private_archive_file(opened)
        ):
            raise EdaccPreparationError()
        if (extraction_root is None) != (extraction_parent_identity is None):
            raise EdaccPreparationError()
        if extraction_root is None:
            tree = None
        else:
            tree_context = _new_extraction_tree(
                extraction_root,
                expected_parent_identity=extraction_parent_identity,
            )
            tree = tree_context.__enter__()

        stream = _CompressedArchiveStream(descriptor, expected_size=expected_size)
        members: dict[str, _ArchiveMemberBinding] = {}
        folded_names: set[str] = set()
        declared_bytes = 0
        metadata_bytes = 0
        audio_files = 0
        zero_blocks = 0
        while zero_blocks < 2:
            header = _read_archive_exact(stream, 512)
            if header == b"\0" * 512:
                zero_blocks += 1
                continue
            if zero_blocks:
                raise EdaccPreparationError()
            member = _parse_tar_header(header)
            if len(members) >= _MAX_ARCHIVE_MEMBERS:
                raise EdaccPreparationError()
            folded = member.name.casefold()
            if member.name in members or folded in folded_names:
                raise EdaccPreparationError()
            folded_names.add(folded)
            if member.is_directory:
                if member.name not in _ARCHIVE_DIRECTORIES:
                    raise EdaccPreparationError()
                binding = _ArchiveMemberBinding(
                    name=member.name,
                    is_directory=True,
                    size_bytes=0,
                    sha256=None,
                )
            else:
                _parent, _filename, maximum, is_audio = _regular_archive_target(
                    member.name
                )
                if member.size_bytes > maximum:
                    raise EdaccPreparationError()
                declared_bytes += member.size_bytes
                if declared_bytes > _MAX_DECLARED_ARCHIVE_BYTES:
                    raise EdaccPreparationError()
                if is_audio:
                    audio_files += 1
                    if audio_files > _MAX_ARCHIVE_AUDIO_FILES:
                        raise EdaccPreparationError()
                else:
                    metadata_bytes += member.size_bytes
                    if metadata_bytes > _MAX_ARCHIVE_METADATA_BYTES:
                        raise EdaccPreparationError()
                binding = _ArchiveMemberBinding(
                    name=member.name,
                    is_directory=False,
                    size_bytes=member.size_bytes,
                    sha256=_consume_archive_file(stream, member, tree=tree),
                )
            members[member.name] = binding

        trailing = 0
        while True:
            chunk = stream.read(_ARCHIVE_IO_CHUNK_BYTES)
            if not chunk:
                break
            trailing += len(chunk)
            if trailing > _MAX_TAR_TRAILING_ZERO_BYTES or any(chunk):
                raise EdaccPreparationError()
        if trailing % 512:
            raise EdaccPreparationError()
        if (
            set(name for name, value in members.items() if value.is_directory)
            != _ARCHIVE_DIRECTORIES
            or not _ARCHIVE_REQUIRED_FILES <= set(members)
            or audio_files <= 0
            or stream.compressed_bytes != expected_size
            or stream.md5 != expected_md5
        ):
            raise EdaccPreparationError()
        after = os.fstat(descriptor)
        current = path.lstat()
        if (
            _file_identity(after) != expected_identity
            or _file_identity(current) != expected_identity
            or not _private_archive_file(after)
            or not _private_archive_file(current)
            or path.resolve(strict=True) != path
        ):
            raise EdaccPreparationError()
        if tree is not None:
            _finalize_extraction_tree(tree, members)
        layout_payload = [
            {
                "directory": value.is_directory,
                "name": value.name,
                "sha256": value.sha256,
                "size_bytes": value.size_bytes,
            }
            for value in sorted(members.values(), key=lambda item: item.name)
        ]
        extracted_directory_identities = (
            None
            if tree is None
            else {
                name: _directory_identity(os.fstat(descriptor))
                for name, descriptor in tree.directory_descriptors.items()
            }
        )
        return _ArchiveSnapshot(
            archive=_ArchiveBinding(
                path=path,
                size_bytes=expected_size,
                md5=stream.md5,
                sha256=stream.sha256,
                identity=expected_identity,
            ),
            members=dict(members),
            layout_sha256=_canonical_sha256(layout_payload),
            extraction_parent_identity=(
                None
                if tree is None
                else _directory_entry_identity(os.fstat(tree.parent_descriptor))
            ),
            extracted_directory_identities=extracted_directory_identities,
            extracted_file_identities=(
                None if tree is None else dict(tree.file_identities)
            ),
        )
    except EdaccPreparationError:
        raise
    except Exception:
        raise EdaccPreparationError() from None
    finally:
        if tree_context is not None:
            try:
                tree_context.__exit__(None, None, None)
            except Exception:
                pass
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _snapshot_metadata(root: Path) -> Mapping[str, _FileBinding]:
    result: dict[str, _FileBinding] = {}
    for relative in _METADATA_FILES:
        path = root / relative
        snapshot = read_regular_bounded(path, maximum_bytes=_METADATA_LIMITS[relative])
        metadata = path.lstat()
        result[relative] = _FileBinding(
            path=snapshot.path,
            size_bytes=len(snapshot.data),
            sha256=hashlib.sha256(snapshot.data).hexdigest(),
            identity=_file_identity(metadata),
        )
    return result


def _metadata_bytes(bindings: Mapping[str, _FileBinding], relative: str) -> bytes:
    binding = bindings[relative]
    snapshot = read_regular_bounded(
        binding.path,
        maximum_bytes=_METADATA_LIMITS[relative],
        expected_bytes=binding.size_bytes,
    )
    if (
        hashlib.sha256(snapshot.data).hexdigest() != binding.sha256
        or _file_identity(binding.path.lstat()) != binding.identity
    ):
        raise EdaccPreparationError()
    return snapshot.data


def _verify_bound_source_file(binding: _FileBinding) -> None:
    before = binding.path.lstat()
    if (
        _file_identity(before) != binding.identity
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size != binding.size_bytes
    ):
        raise EdaccPreparationError()
    resolved = binding.path.resolve(strict=True)
    after = binding.path.lstat()
    if (
        resolved != binding.path
        or _file_identity(after) != binding.identity
        or not stat.S_ISREG(after.st_mode)
        or after.st_nlink != 1
        or after.st_size != binding.size_bytes
    ):
        raise EdaccPreparationError()


def _verify_bound_archive(paths: _PrivatePaths) -> None:
    before = paths.archive_path.lstat()
    if (
        _file_identity(before) != paths.archive_identity
        or not _private_archive_file(before)
    ):
        raise EdaccPreparationError()
    resolved = paths.archive_path.resolve(strict=True)
    after = paths.archive_path.lstat()
    if (
        resolved != paths.archive_path
        or _file_identity(after) != paths.archive_identity
        or not _private_archive_file(after)
    ):
        raise EdaccPreparationError()


def _verify_bound_source_tree(
    paths: _PrivatePaths,
    *,
    metadata: Mapping[str, _FileBinding],
    audio: Mapping[str, _FileBinding],
    reread_metadata: bool,
) -> None:
    root = paths.source_root
    source_files = (*metadata.values(), *audio.values())
    with opened_directory_nofollow(
        root,
        require_private=True,
    ) as (bound_root, root_descriptor):
        if (
            bound_root != root
            or root.resolve(strict=True) != root
            or _directory_identity(os.fstat(root_descriptor))
            != paths.source_identity
            or _directory_identity(root.lstat()) != paths.source_identity
        ):
            raise EdaccPreparationError()
        with opened_directory_nofollow(root / "test"):
            pass
        with opened_directory_nofollow(root / "data"):
            pass
        for binding in source_files:
            _verify_bound_source_file(binding)
        if reread_metadata:
            for relative in metadata:
                _metadata_bytes(metadata, relative)
        for binding in source_files:
            _verify_bound_source_file(binding)
        if (
            _directory_identity(os.fstat(root_descriptor))
            != paths.source_identity
            or _directory_identity(root.lstat()) != paths.source_identity
            or root.resolve(strict=True) != root
        ):
            raise EdaccPreparationError()


def _verify_private_source_snapshot(
    paths: _PrivatePaths,
    *,
    archive_snapshot: _ArchiveSnapshot,
    metadata: Mapping[str, _FileBinding],
    audio: Mapping[str, _FileBinding],
    production: bool,
    output_created: bool,
) -> None:
    try:
        _verify_bound_source_tree(
            paths,
            metadata=metadata,
            audio=audio,
            reread_metadata=True,
        )
        _verify_bound_archive(paths)
        _validate_complete_archive_source_layout(
            paths.source_root,
            archive_snapshot,
        )

        parent = paths.output_dir.parent
        with opened_directory_nofollow(parent) as (
            bound_parent,
            parent_descriptor,
        ):
            if (
                bound_parent != parent
                or parent.resolve(strict=True) != parent
                or _directory_entry_identity(os.fstat(parent_descriptor))
                != paths.output_parent_identity
                or _directory_entry_identity(parent.lstat())
                != paths.output_parent_identity
            ):
                raise EdaccPreparationError()
            try:
                output_metadata = os.stat(
                    paths.output_dir.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                if output_created:
                    raise EdaccPreparationError() from None
            else:
                if (
                    not output_created
                    or not _private_source_root(output_metadata)
                    or _directory_entry_identity(paths.output_dir.lstat())
                    != _directory_entry_identity(output_metadata)
                ):
                    raise EdaccPreparationError()

        if production and (
            _has_git_ancestor(paths.source_root)
            or _has_git_ancestor(paths.archive_path.parent)
            or _has_git_ancestor(paths.output_dir.parent)
        ):
            raise EdaccPreparationError()
        _verify_bound_source_tree(
            paths,
            metadata=metadata,
            audio=audio,
            reread_metadata=False,
        )
        _verify_bound_archive(paths)
        _validate_complete_archive_source_layout(
            paths.source_root,
            archive_snapshot,
        )
    except EdaccPreparationError:
        raise
    except Exception:
        raise EdaccPreparationError() from None


def _parse_keyed_text(data: bytes) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in _lines(data):
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise EdaccPreparationError()
        utterance = _safe_source_id(fields[0])
        reference = fields[1].strip()
        if (
            utterance in result
            or not reference
            or len(reference) > _MAX_SOURCE_REFERENCE_CHARS
        ):
            raise EdaccPreparationError()
        result[utterance] = reference
    return result


def _parse_two_column(data: bytes) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in _lines(data):
        fields = line.split()
        if len(fields) != 2:
            raise EdaccPreparationError()
        key = _safe_source_id(fields[0])
        value = _safe_source_id(fields[1])
        if key in result:
            raise EdaccPreparationError()
        result[key] = value
    return result


def _parse_segments(data: bytes) -> dict[str, tuple[str, Decimal, Decimal]]:
    result: dict[str, tuple[str, Decimal, Decimal]] = {}
    for line in _lines(data):
        fields = line.split()
        if len(fields) != 4:
            raise EdaccPreparationError()
        utterance = _safe_source_id(fields[0])
        recording = _safe_source_id(fields[1])
        start = _decimal(fields[2])
        end = _decimal(fields[3])
        if utterance in result or end <= start:
            raise EdaccPreparationError()
        result[utterance] = (recording, start, end)
    return result


def _parse_stm(
    data: bytes,
) -> dict[tuple[str, str, Decimal, Decimal], str]:
    result: dict[tuple[str, str, Decimal, Decimal], str] = {}
    for line in _lines(data):
        fields = line.split(maxsplit=6)
        if len(fields) != 7:
            raise EdaccPreparationError()
        recording = _safe_source_id(fields[0])
        if fields[1] != "1":
            raise EdaccPreparationError()
        speaker = _safe_source_id(fields[2])
        start = _decimal(fields[3])
        end = _decimal(fields[4])
        label = fields[5]
        reference = fields[6].strip()
        key = (recording, speaker, start, end)
        if (
            end <= start
            or key in result
            or not reference
            or len(reference) > _MAX_SOURCE_REFERENCE_CHARS
            or not (label.startswith("<") and label.endswith(">"))
        ):
            raise EdaccPreparationError()
        result[key] = reference
    return result


def _parse_filtered_stm(
    data: bytes,
) -> dict[tuple[str, str, Decimal, Decimal], str]:
    result: dict[tuple[str, str, Decimal, Decimal], str] = {}
    for line in _lines(data):
        fields = line.split(maxsplit=6)
        if len(fields) not in {6, 7}:
            raise EdaccPreparationError()
        recording = _safe_source_id(fields[0])
        if fields[1] != "1":
            raise EdaccPreparationError()
        speaker = _safe_source_id(fields[2])
        start = _decimal(fields[3])
        end = _decimal(fields[4])
        label = fields[5]
        reference = "" if len(fields) == 6 else fields[6].strip()
        key = (recording, speaker, start, end)
        if (
            end <= start
            or key in result
            or len(reference) > _MAX_SOURCE_REFERENCE_CHARS
            or not (label.startswith("<") and label.endswith(">"))
        ):
            raise EdaccPreparationError()
        result[key] = reference
    return result


def _parse_conversations(data: bytes) -> frozenset[str]:
    values: list[str] = []
    for line in _lines(data):
        fields = line.split()
        if len(fields) != 1:
            raise EdaccPreparationError()
        values.append(_safe_source_id(fields[0]))
    if len(set(values)) != len(values):
        raise EdaccPreparationError()
    return frozenset(values)


def _read_archive_bound_layout_file(
    *,
    root: Path,
    archive: _ArchiveSnapshot,
    relative: str,
    maximum_bytes: int,
) -> bytes:
    try:
        member_name = f"{ARCHIVE_ROOT}/{relative}"
        member = archive.members.get(member_name)
        if member is None or member.is_directory or member.sha256 is None:
            raise EdaccPreparationError()
        path = root / relative
        before = path.lstat()
        extracted_identity = (
            None
            if archive.extracted_file_identities is None
            else archive.extracted_file_identities.get(member_name)
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size != member.size_bytes
            or member.size_bytes > maximum_bytes
            or path.resolve(strict=True) != path
            or (
                extracted_identity is not None
                and _file_identity(before) != extracted_identity
            )
        ):
            raise EdaccPreparationError()
        snapshot = read_regular_bounded(
            path,
            maximum_bytes=maximum_bytes,
            expected_bytes=member.size_bytes,
        )
        after = path.lstat()
        if (
            snapshot.path != path
            or _file_identity(after) != _file_identity(before)
            or (
                extracted_identity is not None
                and _file_identity(after) != extracted_identity
            )
            or hashlib.sha256(snapshot.data).hexdigest() != member.sha256
        ):
            raise EdaccPreparationError()
        return snapshot.data
    except EdaccPreparationError:
        raise
    except Exception:
        raise EdaccPreparationError() from None


def _validate_complete_archive_source_layout(
    root: Path,
    archive: _ArchiveSnapshot,
) -> None:
    """Bind the exact extracted inventory and its split/audio relationships."""

    expected_root = {"data", "dev", "test", *_ARCHIVE_ROOT_FILE_LIMITS}
    expected_split = set(_ARCHIVE_SPLIT_FILE_LIMITS)
    expected_audio = {
        name.rsplit("/", 1)[1]
        for name, member in archive.members.items()
        if not member.is_directory and name.startswith(f"{ARCHIVE_ROOT}/data/")
    }
    expected_members = {
        *_ARCHIVE_DIRECTORIES,
        *_ARCHIVE_REQUIRED_FILES,
        *(f"{ARCHIVE_ROOT}/data/{name}" for name in expected_audio),
    }
    try:
        if set(archive.members) != expected_members or not expected_audio:
            raise EdaccPreparationError()
        with opened_directory_nofollow(
            root,
            require_private=True,
        ) as (bound_root, root_descriptor):
            root_identity = _directory_identity(os.fstat(root_descriptor))
            extracted_root_identity = (
                None
                if archive.extracted_directory_identities is None
                else archive.extracted_directory_identities.get(ARCHIVE_ROOT)
            )
            if (
                bound_root != root
                or root.name != ARCHIVE_ROOT
                or root.resolve(strict=True) != root
                or _directory_identity(root.lstat()) != root_identity
                or (
                    extracted_root_identity is not None
                    and root_identity != extracted_root_identity
                )
                or set(os.listdir(root_descriptor)) != expected_root
            ):
                raise EdaccPreparationError()
            directory_bindings: dict[str, tuple[int, tuple[int, ...]]] = {}
            with (
                opened_directory_nofollow(root / "data") as (
                    _bound_data,
                    data_descriptor,
                ),
                opened_directory_nofollow(root / "dev") as (
                    _bound_dev,
                    dev_descriptor,
                ),
                opened_directory_nofollow(root / "test") as (
                    _bound_test,
                    test_descriptor,
                ),
            ):
                for name, descriptor in (
                    ("data", data_descriptor),
                    ("dev", dev_descriptor),
                    ("test", test_descriptor),
                ):
                    path = root / name
                    identity = _directory_identity(os.fstat(descriptor))
                    extracted_identity = (
                        None
                        if archive.extracted_directory_identities is None
                        else archive.extracted_directory_identities.get(
                            f"{ARCHIVE_ROOT}/{name}"
                        )
                    )
                    if (
                        path.resolve(strict=True) != path
                        or _directory_identity(path.lstat()) != identity
                        or (
                            extracted_identity is not None
                            and identity != extracted_identity
                        )
                    ):
                        raise EdaccPreparationError()
                    directory_bindings[name] = (descriptor, identity)
                if (
                    set(os.listdir(data_descriptor)) != expected_audio
                    or set(os.listdir(dev_descriptor)) != expected_split
                    or set(os.listdir(test_descriptor)) != expected_split
                ):
                    raise EdaccPreparationError()

                for name, maximum in _ARCHIVE_ROOT_FILE_LIMITS.items():
                    _read_archive_bound_layout_file(
                        root=root,
                        archive=archive,
                        relative=name,
                        maximum_bytes=maximum,
                    )

                recordings: dict[str, frozenset[str]] = {}
                for split in ("dev", "test"):
                    split_payloads = {
                        name: _read_archive_bound_layout_file(
                            root=root,
                            archive=archive,
                            relative=f"{split}/{name}",
                            maximum_bytes=maximum,
                        )
                        for name, maximum in _ARCHIVE_SPLIT_FILE_LIMITS.items()
                    }
                    segment_rows = _parse_segments(split_payloads["segments"])
                    split_recordings = frozenset(
                        recording for recording, _start, _end in segment_rows.values()
                    )
                    split_conversations = _parse_conversations(
                        split_payloads["conv.list"]
                    )
                    if (
                        not split_recordings
                        or not split_conversations
                        or not split_conversations <= split_recordings
                    ):
                        raise EdaccPreparationError()
                    recordings[split] = split_recordings

                if recordings["dev"] & recordings["test"]:
                    raise EdaccPreparationError()
                expected_recordings = {
                    name[: -len(".wav")] for name in expected_audio
                }
                if recordings["dev"] | recordings["test"] != expected_recordings:
                    raise EdaccPreparationError()
                for filename in expected_audio:
                    member_name = f"{ARCHIVE_ROOT}/data/{filename}"
                    path = root / "data" / filename
                    current = path.lstat()
                    member = archive.members[member_name]
                    extracted_identity = (
                        None
                        if archive.extracted_file_identities is None
                        else archive.extracted_file_identities.get(member_name)
                    )
                    if (
                        not stat.S_ISREG(current.st_mode)
                        or current.st_nlink != 1
                        or current.st_size != member.size_bytes
                        or path.resolve(strict=True) != path
                        or (
                            extracted_identity is not None
                            and (
                                _file_identity(current) != extracted_identity
                                or not _private_output_file(current)
                            )
                        )
                    ):
                        raise EdaccPreparationError()

                if (
                    set(os.listdir(root_descriptor)) != expected_root
                    or set(os.listdir(data_descriptor)) != expected_audio
                    or set(os.listdir(dev_descriptor)) != expected_split
                    or set(os.listdir(test_descriptor)) != expected_split
                    or _directory_identity(os.fstat(root_descriptor))
                    != root_identity
                    or _directory_identity(root.lstat()) != root_identity
                ):
                    raise EdaccPreparationError()
                for name, (descriptor, identity) in directory_bindings.items():
                    path = root / name
                    if (
                        _directory_identity(os.fstat(descriptor)) != identity
                        or _directory_identity(path.lstat()) != identity
                        or path.resolve(strict=True) != path
                    ):
                        raise EdaccPreparationError()
    except EdaccPreparationError:
        raise
    except Exception:
        raise EdaccPreparationError() from None


def _verify_archive_source_bindings(
    archive: _ArchiveSnapshot,
    *,
    metadata: Mapping[str, _FileBinding],
    audio: Mapping[str, _FileBinding],
) -> None:
    expected = {
        **{
            f"{ARCHIVE_ROOT}/{relative}": binding
            for relative, binding in metadata.items()
        },
        **{
            f"{ARCHIVE_ROOT}/data/{recording}.wav": binding
            for recording, binding in audio.items()
        },
    }
    try:
        for name, source in expected.items():
            member = archive.members.get(name)
            if (
                member is None
                or member.is_directory
                or member.sha256 is None
                or member.size_bytes != source.size_bytes
                or member.sha256 != source.sha256
            ):
                raise EdaccPreparationError()
    except EdaccPreparationError:
        raise
    except Exception:
        raise EdaccPreparationError() from None


def _speaker_key(value: str) -> tuple[int, str] | None:
    match = _SPEAKER_KEY_RE.fullmatch(value)
    if match is None:
        return None
    return int(match.group("conversation")), match.group("side").upper()


def _participant_key(value: str) -> tuple[int, str] | None:
    match = _PARTICIPANT_KEY_RE.fullmatch(value)
    if match is None:
        return None
    side = match.group("participant").upper()
    side = {"1": "A", "2": "B"}.get(side, side)
    return int(match.group("conversation")), side


def _header_field(fieldnames: Sequence[str], candidates: Sequence[str]) -> str:
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    raise EdaccPreparationError()


def _parse_accents(data: bytes) -> tuple[dict[str, str], dict[tuple[int, str], str]]:
    try:
        reader = csv.DictReader(
            io.StringIO(_decode_text(data), newline=""), strict=True
        )
        if reader.fieldnames is None or len(reader.fieldnames) > 128:
            raise EdaccPreparationError()
        participant_field = _header_field(reader.fieldnames, _PARTICIPANT_FIELDS)
        accent_field = _header_field(reader.fieldnames, _ACCENT_FIELDS)
        direct: dict[str, str] = {}
        keyed: dict[tuple[int, str], str] = {}
        participants: set[str] = set()
        rows = 0
        for row in reader:
            rows += 1
            if rows > 1_000 or None in row:
                raise EdaccPreparationError()
            participant = str(row.get(participant_field, "")).strip()
            accent = str(row.get(accent_field, "")).strip()
            if (
                not participant
                or len(participant) > 160
                or len(accent) > 512
                or any(ord(character) < 32 for character in accent)
                or participant in participants
            ):
                raise EdaccPreparationError()
            participants.add(participant)
            if not accent:
                continue
            direct[participant] = accent
            key = _participant_key(participant)
            if key is not None:
                if key in keyed:
                    raise EdaccPreparationError()
                keyed[key] = accent
        if not direct:
            raise EdaccPreparationError()
        return direct, keyed
    except EdaccPreparationError:
        raise
    except Exception:
        raise EdaccPreparationError() from None


def _accent_for_speaker(
    speaker: str,
    direct: Mapping[str, str],
    keyed: Mapping[tuple[int, str], str],
) -> str | None:
    if speaker in direct:
        return direct[speaker]
    key = _speaker_key(speaker)
    return None if key is None else keyed.get(key)


def _overlap_state(
    rows: Sequence[tuple[str, str, str, Decimal, Decimal]],
) -> tuple[frozenset[str], Mapping[str, tuple[tuple[Decimal, Decimal], ...]]]:
    by_recording: dict[str, list[tuple[str, str, str, Decimal, Decimal]]] = {}
    for row in rows:
        by_recording.setdefault(row[1], []).append(row)
    overlapping: set[str] = set()
    regions: dict[str, list[tuple[Decimal, Decimal]]] = {}
    for recording, values in by_recording.items():
        ordered = sorted(values, key=lambda item: (item[3], item[4], item[0]))
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                if right[3] >= left[4]:
                    break
                if right[2] != left[2]:
                    start = max(left[3], right[3])
                    end = min(left[4], right[4])
                    if end > start:
                        overlapping.update((left[0], right[0]))
                        regions.setdefault(recording, []).append((start, end))
    return frozenset(overlapping), {
        recording: tuple(sorted(set(values))) for recording, values in regions.items()
    }


def _is_overlap_adjacent(
    recording: str,
    start: Decimal,
    end: Decimal,
    regions: Mapping[str, Sequence[tuple[Decimal, Decimal]]],
) -> bool:
    for overlap_start, overlap_end in regions.get(recording, ()):
        if end <= overlap_start:
            gap = overlap_start - end
        elif start >= overlap_end:
            gap = start - overlap_end
        else:
            return False
        if gap <= OVERLAP_ADJACENCY_SECONDS:
            return True
    return False


def _is_disfluent(reference: str) -> bool:
    tokens = normalize(reference)
    if any(token in _FILLED_PAUSES for token in tokens):
        return True
    return any(left == right for left, right in zip(tokens, tokens[1:]))


def _scoreable_reference(
    text_reference: str,
    stm_reference: str,
    filtered_reference: str,
) -> bool:
    if stm_reference == "IGNORE_TIME_SEGMENT_IN_SCORING":
        return False
    if (
        len(text_reference) > _MAX_REFERENCE_CHARS
        or len(stm_reference) > _MAX_REFERENCE_CHARS
        or len(filtered_reference) > _MAX_REFERENCE_CHARS
        or _SPECIAL_TOKEN_RE.search(text_reference) is not None
        or _SPECIAL_TOKEN_RE.search(stm_reference) is not None
        or _SPECIAL_TOKEN_RE.search(filtered_reference) is not None
        or _ALTERNATIVE_REFERENCE_RE.search(text_reference) is not None
        or _ALTERNATIVE_REFERENCE_RE.search(stm_reference) is not None
        or _ALTERNATIVE_REFERENCE_RE.search(filtered_reference) is not None
    ):
        return False
    source_words = normalize(text_reference)
    score_words = normalize(filtered_reference)
    return (
        2 <= len(source_words) <= 100
        and source_words == normalize(stm_reference)
        and 2 <= len(score_words) <= 100
    )


def _parse_source(bindings: Mapping[str, _FileBinding]) -> _ParsedSource:
    conversations = _parse_conversations(_metadata_bytes(bindings, "test/conv.list"))
    segments = _parse_segments(_metadata_bytes(bindings, "test/segments"))
    speakers = _parse_two_column(_metadata_bytes(bindings, "test/utt2spk"))
    text = _parse_keyed_text(_metadata_bytes(bindings, "test/text"))
    stm = _parse_stm(_metadata_bytes(bindings, "test/stm"))
    filtered_stm = _parse_filtered_stm(
        _metadata_bytes(bindings, "test/stm.filt")
    )
    direct_accents, keyed_accents = _parse_accents(
        _metadata_bytes(bindings, "linguistic_background.csv")
    )
    if (
        set(segments) != set(speakers)
        or set(segments) != set(text)
        or len(stm) != len(segments)
        or set(filtered_stm) != set(stm)
        or {value[0] for value in segments.values()} != set(conversations)
    ):
        raise EdaccPreparationError()

    structural = tuple(
        (utterance, recording, speakers[utterance], start, end)
        for utterance, (recording, start, end) in segments.items()
    )
    overlapping, overlap_regions = _overlap_state(structural)
    rows: list[_Segment] = []
    eligible: list[_Segment] = []
    for utterance, recording, speaker, start, end in structural:
        try:
            stm_key = (recording, speaker, start, end)
            stm_reference = stm[stm_key]
            score_reference = filtered_stm[stm_key]
        except KeyError:
            raise EdaccPreparationError() from None
        reference = text[utterance]
        accent = _accent_for_speaker(speaker, direct_accents, keyed_accents)
        actual_overlap = utterance in overlapping
        overlap_adjacent = not actual_overlap and _is_overlap_adjacent(
            recording, start, end, overlap_regions
        )
        disfluent = _is_disfluent(reference)
        stratum = (
            "overlap_adjacent"
            if overlap_adjacent
            else "disfluent"
            if disfluent
            else "clean"
        )
        row = _Segment(
            utterance_id=utterance,
            recording_id=recording,
            speaker_id=speaker,
            start=start,
            end=end,
            reference=reference,
            stm_reference=stm_reference,
            score_reference=score_reference,
            accent=accent,
            actual_overlap=actual_overlap,
            overlap_adjacent=overlap_adjacent,
            disfluent=disfluent,
            stratum=stratum,
        )
        rows.append(row)
        duration = end - start
        if (
            not actual_overlap
            and accent is not None
            and MIN_DURATION_SECONDS <= duration <= MAX_DURATION_SECONDS
            and _scoreable_reference(
                reference,
                stm_reference,
                score_reference,
            )
        ):
            eligible.append(row)

    if len(eligible) < CASE_COUNT:
        raise EdaccPreparationError()
    metadata_evidence = {
        relative: {
            "bytes": binding.size_bytes,
            "sha256": binding.sha256,
        }
        for relative, binding in bindings.items()
    }
    return _ParsedSource(
        rows=tuple(rows),
        eligible_rows=tuple(eligible),
        metadata_sha256=_canonical_sha256(metadata_evidence),
        metadata_bindings=bindings,
    )


def _select_rows(parsed: _ParsedSource) -> tuple[tuple[_Segment, ...], str, str]:
    dataset = dataset_by_id(DATASET_ID)
    by_utterance = {row.utterance_id: row for row in parsed.eligible_rows}
    selector_rows = tuple(row.selector_row() for row in parsed.eligible_rows)
    try:
        selected_selector = select_subset_rows(
            dataset.selection,
            selector_rows,
            forbidden_speaker_ids=frozenset(),
        )
    except Exception:
        raise EdaccPreparationError() from None
    selected = tuple(
        by_utterance[str(row["utterance_id"])] for row in selected_selector
    )
    grouped = tuple(
        row for stratum in STRATA for row in selected if row.stratum == stratum
    )
    if (
        len(grouped) != CASE_COUNT
        or len({row.utterance_id for row in grouped}) != CASE_COUNT
        or len({row.speaker_id for row in grouped}) != CASE_COUNT
        or any(
            sum(row.stratum == stratum for row in grouped) != CASES_PER_STRATUM
            for stratum in STRATA
        )
    ):
        raise EdaccPreparationError()
    eligible_digest = _canonical_sha256(
        sorted(
            (
                row.utterance_id,
                row.speaker_id,
                row.stratum,
            )
            for row in parsed.eligible_rows
        )
    )
    selected_digest = selected_rows_sha256(
        dataset.selection,
        tuple(row.selector_row() for row in grouped),
    )
    return grouped, eligible_digest, selected_digest


def _bind_audio(path: Path) -> _FileBinding:
    try:
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > _MAX_AUDIO_BYTES
        ):
            raise EdaccPreparationError()
        digest = hash_regular_bounded(
            path,
            maximum_bytes=_MAX_AUDIO_BYTES,
            expected_bytes=before.st_size,
        )
        current = path.lstat()
        if _file_identity(current) != _file_identity(before):
            raise EdaccPreparationError()
        return _FileBinding(
            path=digest.path,
            size_bytes=digest.size_bytes,
            sha256=digest.sha256,
            identity=_file_identity(current),
        )
    except EdaccPreparationError:
        raise
    except Exception:
        raise EdaccPreparationError() from None


def _output_sample(value: Decimal) -> int:
    result = int(
        (value * OUTPUT_SAMPLE_RATE_HZ).to_integral_value(rounding=ROUND_HALF_UP)
    )
    if result < 0:
        raise EdaccPreparationError()
    return result


def _decode_segment(binding: _FileBinding, row: _Segment) -> _DecodedAudio:
    descriptor = -1
    try:
        if _file_identity(binding.path.lstat()) != binding.identity:
            raise EdaccPreparationError()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(binding.path, flags)
        opened = os.fstat(descriptor)
        if _file_identity(opened) != binding.identity:
            raise EdaccPreparationError()
        with os.fdopen(descriptor, "rb", buffering=0) as handle:
            descriptor = -1
            with wave.open(handle, "rb") as audio:
                if (
                    audio.getnchannels() != SOURCE_CHANNELS
                    or audio.getsampwidth() != SOURCE_SAMPLE_WIDTH_BYTES
                    or audio.getframerate() != SOURCE_SAMPLE_RATE_HZ
                    or audio.getcomptype() != "NONE"
                ):
                    raise EdaccPreparationError()
                start_sample = _output_sample(row.start)
                end_sample = _output_sample(row.end)
                samples = end_sample - start_sample
                source_start = start_sample * 2
                source_frames = samples * 2
                if (
                    samples <= 0
                    or samples * 4 > MAX_PCM_BYTES
                    or source_start < 0
                    or source_start + source_frames > audio.getnframes()
                ):
                    raise EdaccPreparationError()
                audio.setpos(source_start)
                raw = audio.readframes(source_frames)
                if len(raw) != source_frames * SOURCE_SAMPLE_WIDTH_BYTES:
                    raise EdaccPreparationError()
            after = os.fstat(handle.fileno())
        if (
            _file_identity(after) != binding.identity
            or _file_identity(binding.path.lstat()) != binding.identity
        ):
            raise EdaccPreparationError()

        source = array("h")
        source.frombytes(raw)
        if sys.byteorder != "little":
            source.byteswap()
        if len(source) != source_frames:
            raise EdaccPreparationError()
        output = array(
            "f",
            (
                (int(source[index]) + int(source[index + 1])) / 65_536.0
                for index in range(0, len(source), 2)
            ),
        )
        if output.itemsize != 4 or len(output) != samples:
            raise EdaccPreparationError()
        if any(not math.isfinite(value) for value in output):
            raise EdaccPreparationError()
        if sys.byteorder != "little":
            output.byteswap()
        payload = output.tobytes()
        if not payload or len(payload) != samples * 4:
            raise EdaccPreparationError()
        return _DecodedAudio(payload, samples)
    except EdaccPreparationError:
        raise
    except Exception:
        raise EdaccPreparationError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _test_contract(value: TestSourceInjection) -> tuple[int, str]:
    if (
        type(value) is not TestSourceInjection
        or type(value.archive_size_bytes) is not int
        or not 1 <= value.archive_size_bytes <= _MAX_ARCHIVE_BYTES
        or not isinstance(value.archive_md5, str)
        or _MD5_RE.fullmatch(value.archive_md5) is None
        or not isinstance(value.fixture_id, str)
        or _SAFE_FIXTURE_ID_RE.fullmatch(value.fixture_id) is None
    ):
        raise EdaccPreparationError()
    return value.archive_size_bytes, value.archive_md5


def _verify_bound_publication_directory(
    *,
    destination: Path,
    parent_descriptor: int,
    directory_descriptor: int,
    expected_parent_identity: tuple[int, ...],
    output_identity: tuple[int, ...],
) -> None:
    try:
        parent = os.fstat(parent_descriptor)
        opened = os.fstat(directory_descriptor)
        entry = os.stat(
            destination.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        lexical_parent = destination.parent.lstat()
        lexical_output = destination.lstat()
        if (
            _directory_entry_identity(parent) != expected_parent_identity
            or _directory_entry_identity(lexical_parent)
            != expected_parent_identity
            or destination.parent.resolve(strict=True) != destination.parent
            or _directory_entry_identity(opened) != output_identity
            or _directory_entry_identity(entry) != output_identity
            or _directory_entry_identity(lexical_output) != output_identity
            or destination.resolve(strict=True) != destination
            or not _private_source_root(opened)
            or not _private_source_root(entry)
            or not _private_source_root(lexical_output)
        ):
            raise EdaccPreparationError()
    except EdaccPreparationError:
        raise
    except Exception:
        raise EdaccPreparationError() from None


def _publish_private_corpus_bound(
    *,
    cases: Sequence[CorpusWriteCase],
    provenance: CorpusProvenance,
    output_dir: Path,
    purpose: str,
    sidecars: Mapping[str, bytes],
    expected_parent_identity: tuple[int, ...],
) -> LoadedCorpus:
    """Publish through the exact parent bound during EdAcc preflight."""

    directory_descriptor = -1
    try:
        prepared, prepared_sidecars = corpus_writer._preflight(
            cases,
            provenance,
            purpose,
            sidecars,
        )
        rows, manifest_payload = corpus_writer._serialize_manifest(
            prepared,
            provenance,
            purpose,
        )
        manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
        with opened_directory_nofollow(output_dir.parent) as (
            bound_parent,
            parent_descriptor,
        ):
            parent_metadata = os.fstat(parent_descriptor)
            if (
                bound_parent != output_dir.parent
                or _directory_entry_identity(parent_metadata)
                != expected_parent_identity
                or _directory_entry_identity(output_dir.parent.lstat())
                != expected_parent_identity
            ):
                raise EdaccPreparationError()
            os.mkdir(output_dir.name, mode=0o700, dir_fd=parent_descriptor)
            created = os.stat(
                output_dir.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(created.st_mode):
                raise EdaccPreparationError()
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_DIRECTORY", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            directory_descriptor = os.open(
                output_dir.name,
                flags,
                dir_fd=parent_descriptor,
            )
            os.fchmod(directory_descriptor, 0o700)
            output_metadata = os.fstat(directory_descriptor)
            output_identity = _directory_entry_identity(output_metadata)
            if (
                _directory_entry_identity(created) != output_identity
                or not _private_source_root(output_metadata)
            ):
                raise EdaccPreparationError()
            os.fsync(directory_descriptor)
            os.fsync(parent_descriptor)
            _verify_bound_publication_directory(
                destination=output_dir,
                parent_descriptor=parent_descriptor,
                directory_descriptor=directory_descriptor,
                expected_parent_identity=expected_parent_identity,
                output_identity=output_identity,
            )

            sidecar_receipts = [
                (
                    corpus_writer._write_new_private(
                        directory_descriptor,
                        filename,
                        payload,
                    ),
                    payload,
                )
                for filename, payload in prepared_sidecars
            ]
            audio_receipts = [
                corpus_writer._write_new_private(
                    directory_descriptor,
                    str(row["file"]),
                    case.audio_bytes,
                )
                for case, row in zip(prepared, rows, strict=True)
            ]
            manifest_receipt = corpus_writer._write_new_private(
                directory_descriptor,
                "corpus.json",
                manifest_payload,
            )
            _verify_bound_publication_directory(
                destination=output_dir,
                parent_descriptor=parent_descriptor,
                directory_descriptor=directory_descriptor,
                expected_parent_identity=expected_parent_identity,
                output_identity=output_identity,
            )
            for receipt, expected in sidecar_receipts:
                corpus_writer._read_exact_written_private(
                    directory_descriptor,
                    receipt,
                    expected,
                    maximum_bytes=corpus_writer._MAX_SIDECAR_BYTES,
                )
            if (
                corpus_writer._read_exact_written_private(
                    directory_descriptor,
                    manifest_receipt,
                    manifest_payload,
                    maximum_bytes=corpus_writer._MAX_CORPUS_MANIFEST_BYTES,
                )
                != manifest_sha256
            ):
                raise EdaccPreparationError()
            for receipt in audio_receipts:
                corpus_writer._verify_written_private_metadata(
                    directory_descriptor,
                    receipt,
                )

            loaded = load_corpus(output_dir / "corpus.json")
            _verify_bound_publication_directory(
                destination=output_dir,
                parent_descriptor=parent_descriptor,
                directory_descriptor=directory_descriptor,
                expected_parent_identity=expected_parent_identity,
                output_identity=output_identity,
            )
            for receipt, expected in sidecar_receipts:
                corpus_writer._read_exact_written_private(
                    directory_descriptor,
                    receipt,
                    expected,
                    maximum_bytes=corpus_writer._MAX_SIDECAR_BYTES,
                )
            if (
                corpus_writer._read_exact_written_private(
                    directory_descriptor,
                    manifest_receipt,
                    manifest_payload,
                    maximum_bytes=corpus_writer._MAX_CORPUS_MANIFEST_BYTES,
                )
                != manifest_sha256
            ):
                raise EdaccPreparationError()
            for receipt in audio_receipts:
                corpus_writer._verify_written_private_metadata(
                    directory_descriptor,
                    receipt,
                )
            if (
                loaded.path != output_dir / "corpus.json"
                or loaded.digest != manifest_sha256
                or loaded.provenance != provenance
            ):
                raise EdaccPreparationError()
            _verify_bound_publication_directory(
                destination=output_dir,
                parent_descriptor=parent_descriptor,
                directory_descriptor=directory_descriptor,
                expected_parent_identity=expected_parent_identity,
                output_identity=output_identity,
            )
            os.fsync(directory_descriptor)
            os.fsync(parent_descriptor)
            return loaded
    except EdaccPreparationError:
        raise
    except Exception:
        raise EdaccPreparationError() from None
    finally:
        if directory_descriptor >= 0:
            try:
                os.close(directory_descriptor)
            except OSError:
                pass


def _private_output_file(value: os.stat_result) -> bool:
    return (
        stat.S_ISREG(value.st_mode)
        and stat.S_IMODE(value.st_mode) == 0o600
        and value.st_nlink == 1
        and value.st_size > 0
        and _owned_by_process(value)
    )


def _published_case_surface(
    value: LoadedCorpus,
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            case.case_id,
            case.source_path.name,
            case.sha256,
            case.samples,
            case.expected_text,
            case.assertion,
            case.commands,
            case.forbidden_commands,
            case.tags,
        )
        for case in value.cases
    )


def _expected_case_surface(
    cases: Sequence[CorpusWriteCase],
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            case.case_id,
            f"{case.case_id}.f32le",
            hashlib.sha256(case.audio_bytes).hexdigest(),
            len(case.audio_bytes) // 4,
            case.reference,
            case.assertion,
            case.commands,
            case.forbidden_commands,
            case.tags,
        )
        for case in cases
    )


def _verify_output_parent(
    destination: Path,
    *,
    expected_parent_identity: tuple[int, ...],
) -> None:
    try:
        parent = destination.parent
        with opened_directory_nofollow(parent) as (
            bound_parent,
            parent_descriptor,
        ):
            parent_metadata = os.fstat(parent_descriptor)
            output_metadata = os.stat(
                destination.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                bound_parent != parent
                or parent.resolve(strict=True) != parent
                or _directory_entry_identity(parent_metadata)
                != expected_parent_identity
                or _directory_entry_identity(parent.lstat())
                != expected_parent_identity
                or not _private_source_root(output_metadata)
                or _directory_entry_identity(destination.lstat())
                != _directory_entry_identity(output_metadata)
            ):
                raise EdaccPreparationError()
    except EdaccPreparationError:
        raise
    except Exception:
        raise EdaccPreparationError() from None


def _published_output_snapshot(
    *,
    destination: Path,
    receipt_payload: bytes,
    provenance: CorpusProvenance,
    expected_cases: Sequence[CorpusWriteCase],
    expected_manifest_sha256: str,
    expected_parent_identity: tuple[int, ...],
) -> LoadedCorpus:
    try:
        _verify_output_parent(
            destination,
            expected_parent_identity=expected_parent_identity,
        )
        expected_names = {
            "corpus.json",
            "preparation-receipt.json",
            *(f"{case.case_id}.f32le" for case in expected_cases),
        }
        with opened_directory_nofollow(
            destination,
            require_private=True,
        ) as (bound_destination, directory_descriptor):
            directory_identity = _directory_identity(
                os.fstat(directory_descriptor)
            )
            file_identities: dict[str, tuple[int, ...]] = {}
            if (
                bound_destination != destination
                or destination.resolve(strict=True) != destination
                or _directory_identity(destination.lstat()) != directory_identity
                or set(os.listdir(directory_descriptor)) != expected_names
            ):
                raise EdaccPreparationError()
            for name in expected_names:
                metadata = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if not _private_output_file(metadata):
                    raise EdaccPreparationError()
                file_identities[name] = _file_identity(metadata)

            receipt = read_regular_bounded(
                destination / "preparation-receipt.json",
                maximum_bytes=_MAX_RECEIPT_BYTES,
                expected_bytes=len(receipt_payload),
            )
            if receipt.data != receipt_payload:
                raise EdaccPreparationError()
            loaded = load_corpus(destination / "corpus.json")
            if (
                loaded.path != destination / "corpus.json"
                or loaded.digest != expected_manifest_sha256
                or loaded.schema_version != 2
                or loaded.purpose != CORPUS_PURPOSE
                or loaded.provenance != provenance
                or len(loaded.cases) != CASE_COUNT
                or loaded.audio_bytes
                != sum(len(case.audio_bytes) for case in expected_cases)
                or _published_case_surface(loaded)
                != _expected_case_surface(expected_cases)
            ):
                raise EdaccPreparationError()
            verify_corpus_snapshot(loaded)
            if (
                _directory_identity(os.fstat(directory_descriptor))
                != directory_identity
                or _directory_identity(destination.lstat()) != directory_identity
                or set(os.listdir(directory_descriptor)) != expected_names
            ):
                raise EdaccPreparationError()
            for name in expected_names:
                current = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if (
                    not _private_output_file(current)
                    or _file_identity(current) != file_identities[name]
                ):
                    raise EdaccPreparationError()
            _verify_output_parent(
                destination,
                expected_parent_identity=expected_parent_identity,
            )
            return loaded
    except EdaccPreparationError:
        raise
    except Exception:
        raise EdaccPreparationError() from None


def _validate_published_output(
    *,
    published: LoadedCorpus,
    destination: Path,
    receipt_payload: bytes,
    receipt_sha256: str,
    provenance: CorpusProvenance,
    expected_cases: Sequence[CorpusWriteCase],
    lock: fixture.FixtureLock,
    production: bool,
    expected_parent_identity: tuple[int, ...],
) -> LoadedCorpus:
    first = _published_output_snapshot(
        destination=destination,
        receipt_payload=receipt_payload,
        provenance=provenance,
        expected_cases=expected_cases,
        expected_manifest_sha256=published.digest,
        expected_parent_identity=expected_parent_identity,
    )
    if (
        published.path != first.path
        or published.digest != first.digest
        or published.provenance != first.provenance
        or published.schema_version != first.schema_version
        or published.audio_bytes != first.audio_bytes
        or _published_case_surface(published) != _published_case_surface(first)
    ):
        raise EdaccPreparationError()
    if production:
        try:
            binding = fixture.validate_private_source(
                first.path,
                lock=lock,
                expected_source_id=SOURCE_ID,
            )
        except Exception:
            raise EdaccPreparationError() from None
        if (
            binding.receipt_sha256 != receipt_sha256
            or binding.corpus_manifest_sha256 != first.digest
            or binding.cases != CASE_COUNT
            or binding.pcm_bytes != first.audio_bytes
        ):
            raise EdaccPreparationError()
    return _published_output_snapshot(
        destination=destination,
        receipt_payload=receipt_payload,
        provenance=provenance,
        expected_cases=expected_cases,
        expected_manifest_sha256=first.digest,
        expected_parent_identity=expected_parent_identity,
    )


def prepare_edacc_conversation_fixture(
    *,
    source_root: Path | str,
    archive_path: Path | str,
    output_dir: Path | str,
    accepted_terms: frozenset[str],
    lock_path: Path | str = DEFAULT_LOCK,
    test_source_injection: TestSourceInjection | None = None,
) -> PreparedEdaccCorpus:
    """Verify EdAcc raw evidence and publish its deterministic 24-case slice."""

    try:
        _validate_catalog_contract()
        if (
            not isinstance(accepted_terms, frozenset)
            or not REQUIRED_TERMS <= accepted_terms
        ):
            raise EdaccPreparationError()
        production = test_source_injection is None
        dataset = dataset_by_id(DATASET_ID)
        supplied_root = Path(source_root).expanduser()
        existing_test_source = not production and os.path.lexists(supplied_root)
        if existing_test_source:
            acquisition = None
            paths = _validate_private_paths(
                source_root=source_root,
                archive_path=archive_path,
                output_dir=output_dir,
                production=False,
            )
            archive_candidate = paths.archive_path
        else:
            paths = None
            acquisition = _validate_private_acquisition_paths(
                source_root=source_root,
                archive_path=archive_path,
                output_dir=output_dir,
                production=production,
            )
            archive_candidate = acquisition.archive_path
        lock = load_fixture_lock(lock_path)
        source = lock.source_by_id(SOURCE_ID)
        _validate_fixture_source_contract(source, dataset=dataset)
        slots = lock.slots_for(SOURCE_ID)
        if tuple(slot.stratum for slot in slots) != tuple(
            stratum for stratum in STRATA for _ in range(CASES_PER_STRATUM)
        ):
            raise EdaccPreparationError()
        preparer = _preparer_code_provenance(source.preparer_files)

        if production:
            expected_size, expected_md5 = ARCHIVE_SIZE_BYTES, ARCHIVE_MD5
        else:
            expected_size, expected_md5 = _test_contract(test_source_injection)
        archive_snapshot = _consume_archive(
            archive_candidate,
            expected_identity=(
                paths.archive_identity
                if paths is not None
                else acquisition.archive_identity
            ),
            expected_size=expected_size,
            expected_md5=expected_md5,
            extraction_root=(
                None if acquisition is None else acquisition.source_root
            ),
            extraction_parent_identity=(
                None
                if acquisition is None
                else acquisition.source_parent_identity
            ),
        )
        if acquisition is not None:
            paths = _validate_private_paths(
                source_root=acquisition.source_root,
                archive_path=acquisition.archive_path,
                output_dir=acquisition.output_dir,
                production=production,
            )
            if (
                paths.archive_identity != acquisition.archive_identity
                or paths.output_parent_identity
                != acquisition.output_parent_identity
                or archive_snapshot.extraction_parent_identity
                != acquisition.source_parent_identity
                or archive_snapshot.extracted_directory_identities is None
                or paths.source_identity
                != archive_snapshot.extracted_directory_identities.get(
                    ARCHIVE_ROOT
                )
                or _directory_entry_identity(paths.source_root.parent.lstat())
                != acquisition.source_parent_identity
            ):
                raise EdaccPreparationError()
        if paths is None:
            raise EdaccPreparationError()
        root = paths.source_root
        destination = paths.output_dir
        archive = archive_snapshot.archive

        with opened_directory_nofollow(
            root,
            require_private=True,
        ) as (bound_root, root_fd):
            if (
                bound_root != root
                or root.name != ARCHIVE_ROOT
                or _directory_identity(os.fstat(root_fd))
                != paths.source_identity
            ):
                raise EdaccPreparationError()
            with opened_directory_nofollow(root / "test"):
                pass
            with opened_directory_nofollow(root / "data"):
                pass
            _validate_complete_archive_source_layout(root, archive_snapshot)
            metadata_bindings = _snapshot_metadata(root)
            parsed = _parse_source(metadata_bindings)
            selected, eligible_digest, selected_digest = _select_rows(parsed)
            recordings = tuple(sorted({row.recording_id for row in selected}))
            audio_bindings = {
                recording: _bind_audio(root / "data" / f"{recording}.wav")
                for recording in recordings
            }
            _verify_archive_source_bindings(
                archive_snapshot,
                metadata=metadata_bindings,
                audio=audio_bindings,
            )

            published_cases: list[CorpusWriteCase] = []
            receipt_cases: list[dict[str, object]] = []
            total_samples = 0
            for index, (row, slot) in enumerate(zip(selected, slots, strict=True)):
                if slot.stratum != row.stratum:
                    raise EdaccPreparationError()
                decoded = _decode_segment(audio_bindings[row.recording_id], row)
                total_samples += decoded.samples
                if total_samples * 4 > MAX_CORPUS_BYTES:
                    raise EdaccPreparationError()
                identity_sha256 = _salted_digest(
                    lock.selection_seed, "utterance", row.utterance_id
                )
                speaker_sha256 = _salted_digest(
                    lock.selection_seed, "speaker", row.speaker_id
                )
                accent_sha256 = _salted_digest(
                    lock.selection_seed,
                    "accent",
                    str(row.accent).casefold(),
                )
                case_id = f"edacc-{index:02d}-{identity_sha256[:12]}"
                tags = (
                    "public-voice",
                    "conversation96",
                    "edacc",
                    "accented",
                    row.stratum,
                    "single",
                )
                pcm_sha256 = hashlib.sha256(decoded.raw_f32le).hexdigest()
                published_cases.append(
                    CorpusWriteCase(
                        case_id=case_id,
                        audio_bytes=decoded.raw_f32le,
                        reference=row.score_reference,
                        tags=tags,
                    )
                )
                source_audio = audio_bindings[row.recording_id]
                receipt_cases.append(
                    {
                        "slot_id": slot.slot_id,
                        "case_id": case_id,
                        "identity_sha256": identity_sha256,
                        "speaker_sha256": speaker_sha256,
                        "reference_sha256": hashlib.sha256(
                            row.score_reference.encode("utf-8")
                        ).hexdigest(),
                        "source_audio_sha256": source_audio.sha256,
                        "source_audio_bytes": source_audio.size_bytes,
                        "pcm_sha256": pcm_sha256,
                        "pcm_samples": decoded.samples,
                        "tags": list(tags),
                        "attributes": {
                            "stratum": row.stratum,
                            "accent_sha256": accent_sha256,
                            "actual_overlap": False,
                            "reference_source": "official_filtered_stm",
                            "scoreability": "hard_wer",
                        },
                    }
                )

            if len({case["speaker_sha256"] for case in receipt_cases}) != CASE_COUNT:
                raise EdaccPreparationError()
            if (
                len(
                    {str(case["attributes"]["accent_sha256"]) for case in receipt_cases}
                )
                < 8
            ):
                raise EdaccPreparationError()
            if (
                _directory_identity(os.fstat(root_fd))
                != paths.source_identity
                or _directory_identity(root.lstat()) != paths.source_identity
            ):
                raise EdaccPreparationError()

        decoder = {
            "contract": DECODER_CONTRACT,
            "closure_sha256": decoder_closure_sha256(
                DECODER_CONTRACT,
                preparer["files"],
            ),
            "files": len(audio_bindings),
            "bytes": sum(binding.size_bytes for binding in audio_bindings.values()),
            "production_evidence": production,
        }
        receipt = {
            "schema_version": 1,
            "kind": RECEIPT_KIND,
            "fixture_id": FIXTURE_ID,
            "lock_recipe_sha256": lock.recipe_sha256,
            "source_id": SOURCE_ID,
            "source_recipe_sha256": source.selection_recipe_sha256,
            "dataset_contract_sha256": source.dataset_contract_sha256,
            "accepted_terms": sorted(REQUIRED_TERMS),
            "artifacts": [
                {
                    "artifact_id": "archive",
                    "upstream_algorithm": "md5",
                    "upstream_digest": expected_md5,
                    "size_bytes": expected_size,
                    "local_sha256": archive.sha256,
                }
            ],
            "metadata_sha256": parsed.metadata_sha256,
            "selection": {
                "policy_sha256": selection_policy_sha256(
                    dataset.selection
                ),
                "eligible_rows": len(parsed.eligible_rows),
                "eligible_set_sha256": eligible_digest,
                "selected_rows_sha256": selected_digest,
                "selected_cases": CASE_COUNT,
                "selected_case_set_sha256": _canonical_sha256(receipt_cases),
                "parent_receipt_sha256": None,
                "parent_corpus_manifest_sha256": None,
            },
            "preparer": preparer,
            "decoder": decoder,
            "cases": receipt_cases,
            "totals": {
                "cases": CASE_COUNT,
                "pcm_samples": total_samples,
                "pcm_bytes": total_samples * 4,
            },
            "privacy": _PRIVATE_RECEIPT_PRIVACY,
            "evidence_scope": _RECEIPT_EVIDENCE_SCOPE,
        }
        receipt_payload = _canonical_json_bytes(receipt)
        if len(receipt_payload) > _MAX_RECEIPT_BYTES:
            raise EdaccPreparationError()
        receipt_sha256 = hashlib.sha256(receipt_payload).hexdigest()
        provenance = CorpusProvenance(
            kind="public-voice-v1",
            suite=source.corpus_suite,
            manifest_sha256=source.selection_recipe_sha256,
            metadata_sha256=parsed.metadata_sha256,
            source_set_sha256=receipt_sha256,
        )
        _verify_private_source_snapshot(
            paths,
            archive_snapshot=archive_snapshot,
            metadata=metadata_bindings,
            audio=audio_bindings,
            production=production,
            output_created=False,
        )
        published_cases_tuple = tuple(published_cases)
        published = _publish_private_corpus_bound(
            cases=published_cases_tuple,
            provenance=provenance,
            output_dir=destination,
            purpose=CORPUS_PURPOSE,
            sidecars={"preparation-receipt.json": receipt_payload},
            expected_parent_identity=paths.output_parent_identity,
        )
        _verify_private_source_snapshot(
            paths,
            archive_snapshot=archive_snapshot,
            metadata=metadata_bindings,
            audio=audio_bindings,
            production=production,
            output_created=True,
        )
        corpus = _validate_published_output(
            published=published,
            destination=destination,
            receipt_payload=receipt_payload,
            receipt_sha256=receipt_sha256,
            provenance=provenance,
            expected_cases=published_cases_tuple,
            lock=lock,
            production=production,
            expected_parent_identity=paths.output_parent_identity,
        )
        if _preparer_code_provenance(source.preparer_files) != preparer:
            raise EdaccPreparationError()
        _verify_private_source_snapshot(
            paths,
            archive_snapshot=archive_snapshot,
            metadata=metadata_bindings,
            audio=audio_bindings,
            production=production,
            output_created=True,
        )
        corpus = _published_output_snapshot(
            destination=destination,
            receipt_payload=receipt_payload,
            provenance=provenance,
            expected_cases=published_cases_tuple,
            expected_manifest_sha256=corpus.digest,
            expected_parent_identity=paths.output_parent_identity,
        )
        return PreparedEdaccCorpus(
            corpus=corpus,
            receipt_sha256=receipt_sha256,
            metadata_sha256=parsed.metadata_sha256,
            selected_rows_sha256=selected_digest,
            production_evidence=production,
        )
    except EdaccPreparationError:
        raise
    except Exception:
        raise EdaccPreparationError() from None


def _safe_result(value: PreparedEdaccCorpus) -> dict[str, object]:
    return {
        "ok": True,
        "dataset_id": DATASET_ID,
        "fixture_id": FIXTURE_ID,
        "cases": len(value.corpus.cases),
        "audio_bytes": value.corpus.audio_bytes,
        "corpus_sha256": value.corpus.digest,
        "receipt_sha256": value.receipt_sha256,
        "metadata_sha256": value.metadata_sha256,
        "selected_rows_sha256": value.selected_rows_sha256,
        "production_evidence": value.production_evidence,
    }


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise EdaccPreparationError()


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description=(
            "Safely extract and verify an acquired EdAcc v1.0 archive, then "
            "publish the private 24-case conversation fixture source."
        )
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--accept-term", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        result = prepare_edacc_conversation_fixture(
            source_root=arguments.source_root,
            archive_path=arguments.archive,
            output_dir=arguments.output_dir,
            accepted_terms=frozenset(arguments.accept_term),
            lock_path=arguments.lock,
        )
        print(json.dumps(_safe_result(result), sort_keys=True, separators=(",", ":")))
        return 0
    except EdaccPreparationError:
        print(
            json.dumps(_SAFE_ERROR, sort_keys=True, separators=(",", ":")),
            file=sys.stderr,
        )
        return 2
    except SystemExit as exc:
        return int(exc.code or 0)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARCHIVE_FILENAME",
    "ARCHIVE_MD5",
    "ARCHIVE_SIZE_BYTES",
    "DECODER_CONTRACT",
    "DOI",
    "EdaccPreparationError",
    "LICENSE_ID",
    "PreparedEdaccCorpus",
    "REQUIRED_TERMS",
    "SOURCE_ID",
    "TestSourceInjection",
    "main",
    "prepare_edacc_conversation_fixture",
]
