"""Materialize the EdAcc part of the private conversation/STT fixture.

The command accepts both the exact EdAcc v1.0 archive and its directly
extracted ``edacc_v1.0`` directory.  It verifies the published archive MD5,
freezes a local SHA-256, recomputes the official test metadata, binds every
used extracted file back to the archive, selects 24 scoreable turns, and
publishes the official key-matched filtered-STM references in a private
schema-v2 streaming corpus.

No download, recognizer, or audio device is used.  Literal references,
speaker identifiers, linguistic-background values, source paths, and audio
remain outside the preparation receipt.
"""

from __future__ import annotations

from array import array
import argparse
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
from typing import Mapping, Sequence

from core.wer import normalize
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
from tools.streaming_stt.corpus import CorpusProvenance, LoadedCorpus
from tools.streaming_stt.corpus_writer import CorpusWriteCase, publish_private_corpus
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
_MAX_ARCHIVE_BYTES = 8 * 1024 * 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 10_000
_MAX_AUDIO_BYTES = 2 * 1024 * 1024 * 1024
_MAX_ROWS = 100_000
_MAX_LINE_CHARS = 32_768
_MAX_REFERENCE_CHARS = 4_096
_MAX_SOURCE_REFERENCE_CHARS = 32_000
_MAX_RECEIPT_BYTES = 256 * 1024
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}\Z")
_SAFE_FIXTURE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}\Z")
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
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _git_marker_present(directory_fd: int) -> bool:
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
        if _directory_identity(opened) != _directory_identity(marker):
            raise EdaccPreparationError()
        try:
            os.stat("HEAD", dir_fd=marker_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True
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
    for ancestor in (directory, *directory.parents):
        with opened_directory_nofollow(ancestor) as (_stable, descriptor):
            if _git_marker_present(descriptor):
                return True
    return False


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
) -> tuple[Path, Path, Path]:
    try:
        root = _absolute_canonical(source_root, must_exist=True)
        archive = _absolute_canonical(archive_path, must_exist=True)
        destination = _absolute_canonical(output_dir, must_exist=False)
        root_metadata = root.lstat()
        archive_metadata = archive.lstat()
        if (
            root.name != ARCHIVE_ROOT
            or not stat.S_ISDIR(root_metadata.st_mode)
            or stat.S_IMODE(root_metadata.st_mode) != 0o700
            or not stat.S_ISREG(archive_metadata.st_mode)
            or archive_metadata.st_nlink != 1
            or stat.S_IMODE(archive_metadata.st_mode) != 0o600
            or destination.name in {"", ".", ".."}
            or os.path.lexists(destination)
            or destination == root
            or root in destination.parents
            or archive == destination
        ):
            raise EdaccPreparationError()
        if hasattr(os, "geteuid") and (
            root_metadata.st_uid != os.geteuid()
            or archive_metadata.st_uid != os.geteuid()
        ):
            raise EdaccPreparationError()
        with opened_directory_nofollow(root) as (bound_root, descriptor):
            if bound_root != root or _directory_identity(
                os.fstat(descriptor)
            ) != _directory_identity(root_metadata):
                raise EdaccPreparationError()
        with opened_directory_nofollow(archive.parent):
            pass
        with opened_directory_nofollow(destination.parent):
            pass
        if production and (
            _has_git_ancestor(root)
            or _has_git_ancestor(archive.parent)
            or _has_git_ancestor(destination.parent)
        ):
            raise EdaccPreparationError()
        return root, archive, destination
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


def _hash_archive(
    path: Path | str,
    *,
    expected_size: int,
    expected_md5: str,
) -> _ArchiveBinding:
    candidate = Path(os.path.abspath(Path(path).expanduser()))
    descriptor = -1
    try:
        before = candidate.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size != expected_size
            or expected_size <= 0
            or expected_size > _MAX_ARCHIVE_BYTES
            or _MD5_RE.fullmatch(expected_md5) is None
        ):
            raise EdaccPreparationError()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(candidate, flags)
        opened = os.fstat(descriptor)
        if _file_identity(opened) != _file_identity(before):
            raise EdaccPreparationError()
        md5 = hashlib.md5(usedforsecurity=False)
        sha256 = hashlib.sha256()
        consumed = 0
        while consumed <= expected_size:
            chunk = os.read(descriptor, min(1024 * 1024, expected_size + 1 - consumed))
            if not chunk:
                break
            consumed += len(chunk)
            if consumed > expected_size:
                raise EdaccPreparationError()
            md5.update(chunk)
            sha256.update(chunk)
        after = os.fstat(descriptor)
        current = candidate.lstat()
        if (
            consumed != expected_size
            or md5.hexdigest() != expected_md5
            or _file_identity(after) != _file_identity(opened)
            or _file_identity(current) != _file_identity(opened)
            or candidate.resolve(strict=True) != candidate
        ):
            raise EdaccPreparationError()
        return _ArchiveBinding(
            path=candidate,
            size_bytes=consumed,
            md5=md5.hexdigest(),
            sha256=sha256.hexdigest(),
            identity=_file_identity(opened),
        )
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


def _verify_archive_members(
    archive: _ArchiveBinding,
    metadata: Mapping[str, _FileBinding],
    audio: Mapping[str, _FileBinding],
) -> None:
    expected: dict[str, _FileBinding] = {
        f"{ARCHIVE_ROOT}/{relative}": binding for relative, binding in metadata.items()
    }
    expected.update(
        {
            f"{ARCHIVE_ROOT}/data/{recording}.wav": binding
            for recording, binding in audio.items()
        }
    )
    found: set[str] = set()
    try:
        if _file_identity(archive.path.lstat()) != archive.identity:
            raise EdaccPreparationError()
        with archive.path.open("rb", buffering=0) as raw:
            if _file_identity(os.fstat(raw.fileno())) != archive.identity:
                raise EdaccPreparationError()
            with tarfile.open(fileobj=raw, mode="r|gz") as package:
                for index, member in enumerate(package):
                    if index >= _MAX_ARCHIVE_MEMBERS:
                        raise EdaccPreparationError()
                    binding = expected.get(member.name)
                    if binding is None:
                        continue
                    if (
                        member.name in found
                        or not member.isfile()
                        or member.size != binding.size_bytes
                        or member.size <= 0
                        or member.size > _MAX_AUDIO_BYTES
                    ):
                        raise EdaccPreparationError()
                    extracted = package.extractfile(member)
                    if extracted is None:
                        raise EdaccPreparationError()
                    digest = hashlib.sha256()
                    consumed = 0
                    while consumed <= member.size:
                        chunk = extracted.read(
                            min(1024 * 1024, member.size + 1 - consumed)
                        )
                        if not chunk:
                            break
                        consumed += len(chunk)
                        if consumed > member.size:
                            raise EdaccPreparationError()
                        digest.update(chunk)
                    if consumed != member.size or digest.hexdigest() != binding.sha256:
                        raise EdaccPreparationError()
                    found.add(member.name)
            if _file_identity(os.fstat(raw.fileno())) != archive.identity:
                raise EdaccPreparationError()
        if (
            found != set(expected)
            or _file_identity(archive.path.lstat()) != archive.identity
        ):
            raise EdaccPreparationError()
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
        root, archive_candidate, destination = _validate_private_paths(
            source_root=source_root,
            archive_path=archive_path,
            output_dir=output_dir,
            production=production,
        )
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
        archive = _hash_archive(
            archive_candidate,
            expected_size=expected_size,
            expected_md5=expected_md5,
        )

        with opened_directory_nofollow(root) as (bound_root, root_fd):
            root_identity = _directory_identity(os.fstat(root_fd))
            if bound_root != root or root.name != ARCHIVE_ROOT:
                raise EdaccPreparationError()
            with opened_directory_nofollow(root / "test"):
                pass
            with opened_directory_nofollow(root / "data"):
                pass
            metadata_bindings = _snapshot_metadata(root)
            parsed = _parse_source(metadata_bindings)
            selected, eligible_digest, selected_digest = _select_rows(parsed)
            recordings = tuple(sorted({row.recording_id for row in selected}))
            audio_bindings = {
                recording: _bind_audio(root / "data" / f"{recording}.wav")
                for recording in recordings
            }
            _verify_archive_members(archive, metadata_bindings, audio_bindings)

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
            if _directory_identity(os.fstat(root_fd)) != root_identity:
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
        corpus = publish_private_corpus(
            cases=tuple(published_cases),
            provenance=provenance,
            output_dir=destination,
            purpose="EdAcc v1.0 official test conversation fixture",
            sidecars={"preparation-receipt.json": receipt_payload},
        )
        if (
            corpus.provenance != provenance
            or _preparer_code_provenance(source.preparer_files) != preparer
            or _file_identity(archive.path.lstat()) != archive.identity
        ):
            raise EdaccPreparationError()
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
            "Verify an acquired EdAcc v1.0 archive plus its direct extraction "
            "and publish the private 24-case conversation fixture source."
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
