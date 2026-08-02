"""Prepare the bounded public short-command and noisy-confusion corpus.

This command never downloads data, runs a recognizer, or extracts an archive.
It accepts the exact private Google Speech Commands v0.02 test tarball and the
exact private English Consistent Confusion Corpus v1.2 zip, validates every
archive member, and publishes a new private schema-v4 replay corpus outside a
Git checkout.  The preparation receipt and CLI output contain aggregates and
hashes only; source paths, member names, speaker identifiers, and transcripts
are deliberately omitted.
"""

from __future__ import annotations

import argparse
from array import array
from contextlib import contextmanager
import csv
from dataclasses import dataclass, field
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import sys
import tarfile
from typing import BinaryIO, Iterator, Mapping, Sequence
import wave
import zipfile

from tools.public_voice_eval_matrix import (
    ChecksumAlgorithm,
    IntegrityKind,
    MatrixError,
    SelectionAlgorithm,
    dataset_by_id,
)
from tools.streaming_stt.bounded_io import (
    BoundedReadError,
    opened_directory_nofollow,
    read_regular_bounded,
)
from tools.streaming_stt.corpus import (
    CorpusError,
    CorpusProvenance,
    LoadedCorpus,
    verify_corpus_snapshot,
)
from tools.streaming_stt.corpus_writer import (
    CorpusWriteCase,
    CorpusWriterError,
    publish_private_corpus,
)
from tools.streaming_stt.protocol import MAX_CORPUS_BYTES, MAX_PCM_BYTES


SAMPLE_RATE_HZ = 16_000
COMMANDS = (
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
)
SELECTION_SEED = "speaker-public-command-noise-v1-2026-08-02"
LOCK_PATH = (
    Path(__file__).resolve().parent
    / "streaming_stt"
    / "public-command-noise-v1.lock.json"
)
EXPECTED_RECIPE_SHA256 = (
    "4d4e7a1cab6e498fdcda7a74cfab08a01e6bd8ac9c292a4ad940f4c09046433e"
)
REQUIRED_TERMS = frozenset({"CC-BY-4.0"})
ECCC_FIXED_IDS = (
    "4767",
    "5457",
    "6605",
    "7043",
    "8656",
    "11589",
    "16266",
    "17433",
    "19696",
    "20838",
    "20943",
    "22199",
    "22713",
    "25518",
    "26181",
    "28918",
    "29569",
    "32062",
    "32602",
    "33010",
    "34213",
    "34682",
    "38922",
    "39830",
    "40011",
    "41414",
    "42408",
)
PURPOSE = (
    "Development-only isolated command and noisy word-confusion evaluation; "
    "not capture, VAD/endpoint, AEC, barge-in, speaker identity, tools, live "
    "latency, training-disjoint selection, or runtime-default evidence."
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MAX_LOCK_BYTES = 64 * 1024
_MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
_MAX_GSC_MEMBERS = 6_000
_MAX_ECCC_MEMBERS = 4_000
_MAX_ECCC_CENTRAL_DIRECTORY_BYTES = 2 * 1024 * 1024
_MAX_ARCHIVE_EXPANDED_BYTES = 256 * 1024 * 1024
_MAX_AUDIO_MEMBER_BYTES = 4 * 1024 * 1024
_MAX_TEXT_MEMBER_BYTES = 2 * 1024 * 1024
_MAX_WAV_FRAMES = 10 * SAMPLE_RATE_HZ
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SPEECH_FILENAME_RE = re.compile(r"([0-9a-f]{8})_nohash_[0-9]+\.wav\Z")
_UNKNOWN_FILENAME_RE = re.compile(
    r"([a-z]+)_([0-9a-f]{8})_nohash_[0-9]+\.wav\.wav\Z"
)
_SILENCE_FILENAME_RE = re.compile(r"[0-9]+\.wav\Z")
_ECCC_WAV_RE = re.compile(r"EnglishCCC_v1\.2/confusionWavs/T_([0-9]+)\.wav\Z")
_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?\Z")
_SAFE_ERROR = {
    "ok": False,
    "error": "public_command_noise_corpus_prerequisites_unavailable",
}

_ECCC_HEADERS = (
    "ID",
    "Length",
    "Masker",
    "Onset",
    "SNR",
    "Speaker",
    "Target",
    "Raw",
    "Responses",
    "N-Listeners",
    "Counts",
    "Confusion",
    "Consistency",
    "Target-Arpabet",
    "Target-IPA",
    "Target-frequency",
    "Confusion-Arpabet",
    "Confusion-IPA",
    "Confusion-frequency",
    "Phoneme-distance",
)
_ECCC_ROOT = "EnglishCCC_v1.2"
_ECCC_CSV = f"{_ECCC_ROOT}/confusionCorpus_v1.2.csv"
_ECCC_README = f"{_ECCC_ROOT}/README.md"
_ECCC_LICENSE = f"{_ECCC_ROOT}/LICENSE"
_ECCC_FIXED_FILES = {
    _ECCC_CSV,
    _ECCC_README,
    _ECCC_LICENSE,
    f"{_ECCC_ROOT}/confusionWavs/ReadMe.txt",
    f"{_ECCC_ROOT}/maskerWavs/BAB4.txt",
    f"{_ECCC_ROOT}/maskerWavs/BAB4.wav",
    f"{_ECCC_ROOT}/maskerWavs/BMN3.wav",
    f"{_ECCC_ROOT}/maskerWavs/ReadMe.txt",
    f"{_ECCC_ROOT}/maskerWavs/SSN.wav",
}
_ECCC_DIRECTORIES = {
    f"{_ECCC_ROOT}/",
    f"{_ECCC_ROOT}/confusionWavs/",
    f"{_ECCC_ROOT}/maskerWavs/",
}


class CommandNoisePreparationError(RuntimeError):
    """A detail-free invalid source, recipe, transform, or publication."""


@dataclass(frozen=True, slots=True)
class SpeechCommandsArchive:
    filename: str
    size_bytes: int
    sha256: str
    member_count: int
    file_count: int
    uncompressed_bytes: int
    readme_size_bytes: int
    readme_sha256: str
    license_size_bytes: int
    license_sha256: str


@dataclass(frozen=True, slots=True)
class EcccArchive:
    filename: str
    size_bytes: int
    sha256: str
    member_count: int
    file_count: int
    uncompressed_bytes: int
    csv_rows: int
    csv_size_bytes: int
    csv_sha256: str
    readme_size_bytes: int
    readme_sha256: str
    license_size_bytes: int
    license_sha256: str


@dataclass(frozen=True, slots=True)
class TestCommandNoiseInjection:
    """Explicit synthetic source contract; the CLI cannot construct this."""

    speech_commands: SpeechCommandsArchive
    eccc: EcccArchive
    fixture_id: str


@dataclass(frozen=True, slots=True)
class PreparedCommandNoiseCorpus:
    corpus: LoadedCorpus
    receipt_sha256: str
    source_set_sha256: str
    recipe_lock_sha256: str
    production_sources: bool


@dataclass(frozen=True, slots=True)
class _RecipeLock:
    path: Path = field(repr=False)
    raw_sha256: str
    recipe_sha256: str
    raw: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class _PreparedCase:
    case_id: str
    audio_bytes: bytes = field(repr=False)
    reference: str = field(repr=False)
    assertion: str
    commands: tuple[str, ...] = field(repr=False)
    forbidden_commands: tuple[str, ...] = field(repr=False)
    tags: tuple[str, ...]
    selection_record: Mapping[str, object] = field(repr=False)


@dataclass(frozen=True, slots=True)
class _PreparedSource:
    source_id: str
    archive_sha256: str
    cases: tuple[_PreparedCase, ...] = field(repr=False)
    selection_sha256: str


@dataclass(frozen=True, slots=True)
class _GscCandidate:
    label: str
    speaker_id: str | None = field(repr=False)
    member_name: str = field(repr=False)
    info: tarfile.TarInfo = field(repr=False)


@dataclass(frozen=True, slots=True)
class _EcccRow:
    row_id: str = field(repr=False)
    length: int
    target: str = field(repr=False)
    confusion: str = field(repr=False)


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
    except (MemoryError, OverflowError, TypeError, UnicodeError, ValueError):
        raise CommandNoisePreparationError() from None


def _digest_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_json(raw: bytes) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise CommandNoisePreparationError()
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                CommandNoisePreparationError()
            ),
        )
    except CommandNoisePreparationError:
        raise
    except (OverflowError, UnicodeError, ValueError):
        raise CommandNoisePreparationError() from None


def _load_recipe_lock(path: Path | str = LOCK_PATH) -> _RecipeLock:
    try:
        snapshot = read_regular_bounded(Path(path), maximum_bytes=_MAX_LOCK_BYTES)
        value = _strict_json(snapshot.data)
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "kind",
            "fixture_id",
            "recipe_sha256",
            "recipe_digest_rule",
            "provenance",
            "required_terms",
            "selection_seed",
            "command_vocabulary",
            "sample_format",
            "totals",
            "sources",
            "privacy",
            "evidence_scope",
        }:
            raise CommandNoisePreparationError()
        declared = value.get("recipe_sha256")
        digest_value = dict(value)
        digest_value.pop("recipe_sha256", None)
        calculated = _digest_json(digest_value)
        if (
            value.get("schema_version") != 1
            or value.get("kind") != "public-command-noise-recipe-lock-v1"
            or value.get("fixture_id") != "public-command-noise-v1"
            or value.get("recipe_digest_rule")
            != "sha256-canonical-json-without-recipe_sha256-v1"
            or declared != EXPECTED_RECIPE_SHA256
            or calculated != declared
            or value.get("selection_seed") != SELECTION_SEED
            or value.get("command_vocabulary") != list(COMMANDS)
            or value.get("provenance")
            != {"kind": "public-command-noise-v1", "suite": "command-noise-v1"}
            or value.get("required_terms") != sorted(REQUIRED_TERMS)
            or value.get("sample_format")
            != {"encoding": "f32le", "sample_rate_hz": 16000, "channels": 1}
            or value.get("totals")
            != {
                "sources": 2,
                "cases": 57,
                "command_positive": 26,
                "command_negative": 26,
                "assertions": {
                    "transcript": 47,
                    "speech_negative": 5,
                    "silence": 5,
                },
            }
            or not isinstance(value.get("sources"), list)
            or len(value["sources"]) != 2
        ):
            raise CommandNoisePreparationError()
        return _RecipeLock(
            path=snapshot.path,
            raw_sha256=hashlib.sha256(snapshot.data).hexdigest(),
            recipe_sha256=str(declared),
            raw=snapshot.data,
        )
    except (BoundedReadError, CommandNoisePreparationError):
        raise CommandNoisePreparationError() from None


def _production_contracts(recipe: _RecipeLock) -> tuple[SpeechCommandsArchive, EcccArchive]:
    value = _strict_json(recipe.raw)
    if not isinstance(value, dict):
        raise CommandNoisePreparationError()
    sources = value.get("sources")
    if not isinstance(sources, list) or len(sources) != 2:
        raise CommandNoisePreparationError()
    gsc = sources[0]
    eccc = sources[1]
    try:
        if not isinstance(gsc, dict) or not isinstance(eccc, dict):
            raise CommandNoisePreparationError()
        gsc_readme = gsc["readme"]
        gsc_license = gsc["license_file"]
        eccc_csv = eccc["csv"]
        eccc_readme = eccc["readme"]
        eccc_license = eccc["license_file"]
        if not all(
            isinstance(value, dict)
            for value in (
                gsc_readme,
                gsc_license,
                eccc_csv,
                eccc_readme,
                eccc_license,
            )
        ):
            raise CommandNoisePreparationError()
        result = (
            SpeechCommandsArchive(
                filename=str(gsc["filename"]),
                size_bytes=int(gsc["size_bytes"]),
                sha256=str(gsc["sha256"]),
                member_count=int(gsc["members"]),
                file_count=int(gsc["files"]),
                uncompressed_bytes=int(gsc["uncompressed_bytes"]),
                readme_size_bytes=int(gsc_readme["size_bytes"]),
                readme_sha256=str(gsc_readme["sha256"]),
                license_size_bytes=int(gsc_license["size_bytes"]),
                license_sha256=str(gsc_license["sha256"]),
            ),
            EcccArchive(
                filename=str(eccc["filename"]),
                size_bytes=int(eccc["size_bytes"]),
                sha256=str(eccc["sha256"]),
                member_count=int(eccc["members"]),
                file_count=int(eccc["files"]),
                uncompressed_bytes=int(eccc["uncompressed_bytes"]),
                csv_rows=int(eccc_csv["rows"]),
                csv_size_bytes=int(eccc_csv["size_bytes"]),
                csv_sha256=str(eccc_csv["sha256"]),
                readme_size_bytes=int(eccc_readme["size_bytes"]),
                readme_sha256=str(eccc_readme["sha256"]),
                license_size_bytes=int(eccc_license["size_bytes"]),
                license_sha256=str(eccc_license["sha256"]),
            ),
        )
    except (KeyError, OverflowError, TypeError, ValueError):
        raise CommandNoisePreparationError() from None
    _validate_contracts(*result)
    return result


def _valid_digest(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _validate_contracts(gsc: SpeechCommandsArchive, eccc: EcccArchive) -> None:
    if (
        type(gsc) is not SpeechCommandsArchive
        or type(eccc) is not EcccArchive
        or not gsc.filename
        or "/" in gsc.filename
        or not eccc.filename
        or "/" in eccc.filename
        or not 0 < gsc.size_bytes <= _MAX_ARCHIVE_BYTES
        or not 0 < eccc.size_bytes <= _MAX_ARCHIVE_BYTES
        or not _valid_digest(gsc.sha256)
        or not _valid_digest(eccc.sha256)
        or not 1 <= gsc.member_count <= _MAX_GSC_MEMBERS
        or not 1 <= gsc.file_count <= gsc.member_count
        or not 1 <= gsc.uncompressed_bytes <= _MAX_ARCHIVE_EXPANDED_BYTES
        or not 1 <= eccc.member_count <= _MAX_ECCC_MEMBERS
        or not 1 <= eccc.file_count <= eccc.member_count
        or not 1 <= eccc.uncompressed_bytes <= _MAX_ARCHIVE_EXPANDED_BYTES
        or not 1 <= eccc.csv_rows <= _MAX_ECCC_MEMBERS
        or not 1 <= gsc.readme_size_bytes <= _MAX_TEXT_MEMBER_BYTES
        or not 1 <= gsc.license_size_bytes <= _MAX_TEXT_MEMBER_BYTES
        or not 1 <= eccc.csv_size_bytes <= _MAX_TEXT_MEMBER_BYTES
        or not 1 <= eccc.readme_size_bytes <= _MAX_TEXT_MEMBER_BYTES
        or not 1 <= eccc.license_size_bytes <= _MAX_TEXT_MEMBER_BYTES
        or not all(
            _valid_digest(value)
            for value in (
                gsc.readme_sha256,
                gsc.license_sha256,
                eccc.csv_sha256,
                eccc.readme_sha256,
                eccc.license_sha256,
            )
        )
    ):
        raise CommandNoisePreparationError()


def _validate_catalog_binding(
    gsc: SpeechCommandsArchive,
    eccc: EcccArchive,
) -> tuple[str, ...]:
    """Bind production pins and the ECCC selector to the typed catalog."""

    try:
        gsc_dataset = dataset_by_id("speech_commands_v002_test")
        eccc_dataset = dataset_by_id("english_consistent_confusion_v1_2")
        if (
            gsc_dataset.dataset_id != "speech_commands_v002_test"
            or eccc_dataset.dataset_id != "english_consistent_confusion_v1_2"
            or gsc_dataset.license.license_id != "CC-BY-4.0"
            or eccc_dataset.license.license_id != "CC-BY-4.0"
            or gsc_dataset.license.acceptance_ids != ("CC-BY-4.0",)
            or eccc_dataset.license.acceptance_ids != ("CC-BY-4.0",)
            or len(gsc_dataset.artifacts) != 1
            or len(eccc_dataset.artifacts) != 1
        ):
            raise CommandNoisePreparationError()
        gsc_artifact = gsc_dataset.artifacts[0]
        eccc_artifact = eccc_dataset.artifacts[0]
        if (
            gsc_artifact.artifact_id != "test-archive"
            or gsc_artifact.filename != gsc.filename
            or gsc_artifact.size_bytes != gsc.size_bytes
            or gsc_artifact.integrity is not IntegrityKind.PINNED
            or gsc_artifact.checksum is None
            or gsc_artifact.checksum.algorithm is not ChecksumAlgorithm.SHA256
            or gsc_artifact.checksum.digest != gsc.sha256
            or eccc_artifact.artifact_id != "archive"
            or eccc_artifact.filename != eccc.filename
            or eccc_artifact.size_bytes != eccc.size_bytes
            or eccc_artifact.integrity is not IntegrityKind.LOCAL_SHA256_RECEIPT
            or eccc_artifact.checksum is None
            or eccc_artifact.checksum.algorithm is not ChecksumAlgorithm.SHA256
            or eccc_artifact.checksum.digest != eccc.sha256
            or gsc_dataset.selection.algorithm
            is not SelectionAlgorithm.COMPLETE_SPLIT
            or gsc_dataset.selection.expected_examples != 4_890
            or eccc_dataset.selection.algorithm is not SelectionAlgorithm.FIXED_IDS
            or eccc_dataset.selection.identity_fields != ("id",)
            or eccc_dataset.selection.expected_examples != 27
            or eccc_dataset.selection.fixed_ids != ECCC_FIXED_IDS
        ):
            raise CommandNoisePreparationError()
        return eccc_dataset.selection.fixed_ids
    except CommandNoisePreparationError:
        raise
    except (AttributeError, KeyError, MatrixError, TypeError, ValueError):
        raise CommandNoisePreparationError() from None


def _validated_accepted_terms(values: object) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise CommandNoisePreparationError()
    try:
        terms = frozenset(values)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise CommandNoisePreparationError() from None
    if (
        not REQUIRED_TERMS.issubset(terms)
        or any(
            not isinstance(value, str)
            or not value
            or len(value) > 128
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+-]*", value) is None
            for value in terms
        )
    ):
        raise CommandNoisePreparationError()
    return tuple(sorted(terms))


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        stat.S_IMODE(metadata.st_mode),
        metadata.st_uid,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _hash_descriptor(descriptor: int, *, expected_bytes: int) -> str:
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        consumed = 0
        while consumed <= expected_bytes:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, expected_bytes + 1 - consumed),
            )
            if not chunk:
                break
            consumed += len(chunk)
            if consumed > expected_bytes:
                raise CommandNoisePreparationError()
            digest.update(chunk)
        if consumed != expected_bytes:
            raise CommandNoisePreparationError()
        result = digest.hexdigest()
        os.lseek(descriptor, 0, os.SEEK_SET)
        return result
    except CommandNoisePreparationError:
        raise
    except (OSError, OverflowError, ValueError):
        raise CommandNoisePreparationError() from None


@contextmanager
def _open_private_archive(
    path: Path | str,
    *,
    filename: str,
    size_bytes: int,
    sha256: str,
) -> Iterator[BinaryIO]:
    supplied = Path(path).expanduser()
    if not supplied.is_absolute():
        raise CommandNoisePreparationError()
    candidate = supplied
    descriptor = -1
    duplicate = -1
    try:
        before = candidate.lstat()
        if (
            candidate.name != filename
            or candidate.resolve(strict=True) != candidate
            or _has_git_ancestor(candidate.parent)
            or not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_nlink != 1
            or before.st_size != size_bytes
            or (hasattr(os, "geteuid") and before.st_uid != os.geteuid())
        ):
            raise CommandNoisePreparationError()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(candidate, flags)
        opened = os.fstat(descriptor)
        identity = _file_identity(opened)
        if identity != _file_identity(before):
            raise CommandNoisePreparationError()
        if _hash_descriptor(descriptor, expected_bytes=size_bytes) != sha256:
            raise CommandNoisePreparationError()
        duplicate = os.dup(descriptor)
        with os.fdopen(duplicate, "rb", buffering=0) as handle:
            duplicate = -1
            try:
                yield handle
            finally:
                if (
                    _file_identity(os.fstat(descriptor)) != identity
                    or _file_identity(candidate.lstat()) != identity
                    or candidate.resolve(strict=True) != candidate
                    or _hash_descriptor(descriptor, expected_bytes=size_bytes)
                    != sha256
                    or _file_identity(os.fstat(descriptor)) != identity
                    or _file_identity(candidate.lstat()) != identity
                ):
                    raise CommandNoisePreparationError()
    except CommandNoisePreparationError:
        raise
    except (OSError, RuntimeError, ValueError, OverflowError):
        raise CommandNoisePreparationError() from None
    finally:
        if duplicate >= 0:
            try:
                os.close(duplicate)
            except OSError:
                pass
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _safe_tar_name(name: str) -> str:
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise CommandNoisePreparationError()
    if name == ".":
        return ""
    if not name.startswith("./"):
        raise CommandNoisePreparationError()
    value = name[2:]
    pure = PurePosixPath(value)
    if (
        not value
        or value.startswith("/")
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != value
    ):
        raise CommandNoisePreparationError()
    return value


def _read_tar_member(archive: tarfile.TarFile, info: tarfile.TarInfo) -> bytes:
    if not info.isreg() or not 0 < info.size <= max(
        _MAX_AUDIO_MEMBER_BYTES, _MAX_TEXT_MEMBER_BYTES
    ):
        raise CommandNoisePreparationError()
    try:
        extracted = archive.extractfile(info)
        if extracted is None:
            raise CommandNoisePreparationError()
        with extracted:
            payload = extracted.read(info.size + 1)
            if len(payload) != info.size or extracted.read(1):
                raise CommandNoisePreparationError()
        return payload
    except CommandNoisePreparationError:
        raise
    except (EOFError, OSError, tarfile.TarError, ValueError):
        raise CommandNoisePreparationError() from None


def _rank(*values: str) -> str:
    return hashlib.sha256("\x00".join((SELECTION_SEED, *values)).encode()).hexdigest()


def _speaker_first(
    candidates: Sequence[_GscCandidate],
    *,
    label: str,
    count: int,
    used_speakers: set[str],
) -> list[_GscCandidate]:
    grouped: dict[str, list[_GscCandidate]] = {}
    for candidate in candidates:
        if candidate.label == label and candidate.speaker_id is not None:
            grouped.setdefault(candidate.speaker_id, []).append(candidate)
    result: list[_GscCandidate] = []
    for speaker in sorted(grouped, key=lambda value: (_rank(label, value), value)):
        if speaker in used_speakers:
            continue
        choices = sorted(
            grouped[speaker],
            key=lambda value: (
                _rank(label, speaker, value.member_name),
                value.member_name,
            ),
        )
        if not choices:
            continue
        result.append(choices[0])
        used_speakers.add(speaker)
        if len(result) == count:
            break
    if len(result) != count:
        raise CommandNoisePreparationError()
    return result


def _decode_pcm16_wav(payload: bytes, *, channels: int) -> tuple[bytes, int]:
    if (
        not isinstance(payload, bytes)
        or channels not in {1, 2}
        or not 44 <= len(payload) <= _MAX_AUDIO_MEMBER_BYTES
    ):
        raise CommandNoisePreparationError()
    try:
        with wave.open(io.BytesIO(payload), "rb") as reader:
            frames = reader.getnframes()
            if (
                reader.getnchannels() != channels
                or reader.getsampwidth() != 2
                or reader.getframerate() != SAMPLE_RATE_HZ
                or reader.getcomptype() != "NONE"
                or not 0 < frames <= _MAX_WAV_FRAMES
                or frames * 4 > MAX_PCM_BYTES
            ):
                raise CommandNoisePreparationError()
            raw = reader.readframes(frames + 1)
            if len(raw) != frames * channels * 2 or reader.readframes(1):
                raise CommandNoisePreparationError()
        source = array("h")
        source.frombytes(raw)
        if sys.byteorder != "little":
            source.byteswap()
        if len(source) != frames * channels:
            raise CommandNoisePreparationError()
        output = array("f")
        if channels == 1:
            output.extend(float(value) / 32768.0 for value in source)
        else:
            for index in range(0, len(source), 2):
                mixed = int(source[index]) + int(source[index + 1])
                if not -32768 <= mixed <= 32767:
                    raise CommandNoisePreparationError()
                output.append(float(mixed) / 32768.0)
        if sys.byteorder != "little":
            output.byteswap()
        result = output.tobytes()
        if (
            len(result) != frames * 4
            or not result
            or any(not math.isfinite(value) for value in output)
        ):
            raise CommandNoisePreparationError()
        return result, frames
    except CommandNoisePreparationError:
        raise
    except (EOFError, MemoryError, OverflowError, TypeError, ValueError, wave.Error):
        raise CommandNoisePreparationError() from None


def _prepare_gsc(
    path: Path | str,
    contract: SpeechCommandsArchive,
) -> _PreparedSource:
    labels = {*COMMANDS, "_unknown_", "_silence_"}
    with _open_private_archive(
        path,
        filename=contract.filename,
        size_bytes=contract.size_bytes,
        sha256=contract.sha256,
    ) as handle:
        try:
            with tarfile.open(fileobj=handle, mode="r:gz") as archive:
                names: set[str] = set()
                directories: set[str] = set()
                member_count = 0
                file_count = 0
                expanded = 0
                candidates: list[_GscCandidate] = []
                by_name: dict[str, tarfile.TarInfo] = {}
                for info in archive:
                    member_count += 1
                    if (
                        member_count > contract.member_count
                        or member_count > _MAX_GSC_MEMBERS
                    ):
                        raise CommandNoisePreparationError()
                    name = _safe_tar_name(info.name)
                    if name in names:
                        raise CommandNoisePreparationError()
                    names.add(name)
                    if info.isdir():
                        if info.size != 0:
                            raise CommandNoisePreparationError()
                        directories.add(name)
                        continue
                    if not info.isreg() or not 0 < info.size <= _MAX_AUDIO_MEMBER_BYTES:
                        raise CommandNoisePreparationError()
                    file_count += 1
                    expanded += info.size
                    if expanded > _MAX_ARCHIVE_EXPANDED_BYTES:
                        raise CommandNoisePreparationError()
                    by_name[name] = info
                    if name in {"README.md", "LICENSE"}:
                        continue
                    pure = PurePosixPath(name)
                    if len(pure.parts) != 2 or pure.parts[0] not in labels:
                        raise CommandNoisePreparationError()
                    label, filename = pure.parts
                    if label == "_silence_":
                        if _SILENCE_FILENAME_RE.fullmatch(filename) is None:
                            raise CommandNoisePreparationError()
                        speaker = None
                    elif label == "_unknown_":
                        match = _UNKNOWN_FILENAME_RE.fullmatch(filename)
                        if match is None or match.group(1) in COMMANDS:
                            raise CommandNoisePreparationError()
                        speaker = match.group(2)
                    else:
                        match = _SPEECH_FILENAME_RE.fullmatch(filename)
                        if match is None:
                            raise CommandNoisePreparationError()
                        speaker = match.group(1)
                    candidates.append(
                        _GscCandidate(
                            label=label,
                            speaker_id=speaker,
                            member_name=name,
                            info=info,
                        )
                    )
                if (
                    member_count != contract.member_count
                    or file_count != contract.file_count
                    or expanded != contract.uncompressed_bytes
                    or directories != {"", *labels}
                    or set(by_name) != {"README.md", "LICENSE"}.union(
                        candidate.member_name for candidate in candidates
                    )
                ):
                    raise CommandNoisePreparationError()
                for name, expected_size, expected_sha in (
                    (
                        "README.md",
                        contract.readme_size_bytes,
                        contract.readme_sha256,
                    ),
                    (
                        "LICENSE",
                        contract.license_size_bytes,
                        contract.license_sha256,
                    ),
                ):
                    info = by_name.get(name)
                    if info is None or info.size != expected_size:
                        raise CommandNoisePreparationError()
                    if hashlib.sha256(_read_tar_member(archive, info)).hexdigest() != expected_sha:
                        raise CommandNoisePreparationError()

                used_speakers: set[str] = set()
                selected: list[_GscCandidate] = []
                for command in COMMANDS:
                    selected.extend(
                        _speaker_first(
                            candidates,
                            label=command,
                            count=2,
                            used_speakers=used_speakers,
                        )
                    )
                selected.extend(
                    _speaker_first(
                        candidates,
                        label="_unknown_",
                        count=5,
                        used_speakers=used_speakers,
                    )
                )
                silence = sorted(
                    (candidate for candidate in candidates if candidate.label == "_silence_"),
                    key=lambda value: (
                        _rank("_silence_", value.member_name),
                        value.member_name,
                    ),
                )[:5]
                if len(used_speakers) != 25 or len(silence) != 5:
                    raise CommandNoisePreparationError()
                selected.extend(silence)

                prepared: list[_PreparedCase] = []
                for index, candidate in enumerate(selected, start=1):
                    payload = _read_tar_member(archive, candidate.info)
                    audio, frames = _decode_pcm16_wav(payload, channels=1)
                    if candidate.label in COMMANDS:
                        assertion = "transcript"
                        reference = candidate.label
                        commands = (candidate.label,)
                        forbidden = tuple(
                            value for value in COMMANDS if value != candidate.label
                        )
                        tags = ("public-command-noise", "gsc", "command-positive")
                    elif candidate.label == "_unknown_":
                        assertion = "speech_negative"
                        reference = ""
                        commands = ()
                        forbidden = COMMANDS
                        tags = (
                            "public-command-noise",
                            "gsc",
                            "command-negative",
                            "speech-negative",
                        )
                    elif candidate.label == "_silence_":
                        assertion = "silence"
                        reference = ""
                        commands = ()
                        forbidden = COMMANDS
                        tags = ("public-command-noise", "gsc", "silence")
                    else:
                        raise CommandNoisePreparationError()
                    prepared.append(
                        _PreparedCase(
                            case_id=f"gsc-case-{index:03d}",
                            audio_bytes=audio,
                            reference=reference,
                            assertion=assertion,
                            commands=commands,
                            forbidden_commands=forbidden,
                            tags=tags,
                            selection_record={
                                "member_sha256": hashlib.sha256(payload).hexdigest(),
                                "output_sha256": hashlib.sha256(audio).hexdigest(),
                                "samples": frames,
                                "selection_sha256": _digest_json(
                                    {
                                        "label": candidate.label,
                                        "member": candidate.member_name,
                                        "speaker": candidate.speaker_id,
                                    }
                                ),
                            },
                        )
                    )
        except CommandNoisePreparationError:
            raise
        except (EOFError, OSError, tarfile.TarError, ValueError):
            raise CommandNoisePreparationError() from None
    selection_sha256 = _digest_json(
        [dict(case.selection_record) for case in prepared]
    )
    return _PreparedSource(
        source_id="google-speech-commands-v002-test",
        archive_sha256=contract.sha256,
        cases=tuple(prepared),
        selection_sha256=selection_sha256,
    )


def _safe_zip_name(info: zipfile.ZipInfo) -> str:
    name = info.filename
    core = name[:-1] if isinstance(name, str) and name.endswith("/") else name
    if (
        not isinstance(name, str)
        or not name
        or "\x00" in name
        or "\\" in name
        or name.startswith("/")
        or not core
        or PurePosixPath(core).as_posix() != core
        or any(part in {"", ".", ".."} for part in PurePosixPath(core).parts)
    ):
        raise CommandNoisePreparationError()
    return name


def _preflight_zip_central_directory(
    handle: BinaryIO,
    *,
    archive_bytes: int,
    expected_members: int,
) -> None:
    """Bound ZIP metadata before ``ZipFile`` materializes its entry objects."""

    try:
        tail_bytes = min(archive_bytes, 65_535 + 22)
        handle.seek(archive_bytes - tail_bytes)
        tail = handle.read(tail_bytes + 1)
        if len(tail) != tail_bytes:
            raise CommandNoisePreparationError()
        offset = tail.rfind(b"PK\x05\x06")
        if offset < 0 or offset + 22 > len(tail):
            raise CommandNoisePreparationError()
        (
            signature,
            disk_number,
            central_disk,
            disk_entries,
            total_entries,
            central_bytes,
            central_offset,
            comment_bytes,
        ) = struct.unpack_from("<4s4H2LH", tail, offset)
        absolute_offset = archive_bytes - tail_bytes + offset
        if (
            signature != b"PK\x05\x06"
            or disk_number != 0
            or central_disk != 0
            or disk_entries != total_entries
            or total_entries != expected_members
            or not 1 <= total_entries <= _MAX_ECCC_MEMBERS
            or not 1 <= central_bytes <= _MAX_ECCC_CENTRAL_DIRECTORY_BYTES
            or central_offset + central_bytes != absolute_offset
            or comment_bytes != 0
            or absolute_offset + 22 != archive_bytes
        ):
            raise CommandNoisePreparationError()
        handle.seek(0)
    except CommandNoisePreparationError:
        raise
    except (OSError, OverflowError, struct.error, ValueError):
        raise CommandNoisePreparationError() from None


def _read_zip_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    maximum: int,
) -> bytes:
    if info.is_dir() or not 0 < info.file_size <= maximum:
        raise CommandNoisePreparationError()
    try:
        with archive.open(info, "r") as handle:
            payload = handle.read(info.file_size + 1)
            if len(payload) != info.file_size or handle.read(1):
                raise CommandNoisePreparationError()
        return payload
    except CommandNoisePreparationError:
        raise
    except (EOFError, OSError, RuntimeError, ValueError, zipfile.BadZipFile):
        raise CommandNoisePreparationError() from None


def _parse_int(value: str, *, minimum: int, maximum: int) -> int:
    if not value or not value.isascii() or not value.isdigit():
        raise CommandNoisePreparationError()
    result = int(value)
    if not minimum <= result <= maximum:
        raise CommandNoisePreparationError()
    return result


def _parse_eccc_csv(payload: bytes, *, expected_rows: int) -> tuple[_EcccRow, ...]:
    try:
        text = payload.decode("utf-8")
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        header = next(reader)
        if tuple(header) != _ECCC_HEADERS:
            raise CommandNoisePreparationError()
        rows: list[_EcccRow] = []
        identifiers: set[str] = set()
        for values in reader:
            if len(values) != len(_ECCC_HEADERS):
                raise CommandNoisePreparationError()
            row = dict(zip(_ECCC_HEADERS, values, strict=True))
            if any(
                not isinstance(value, str)
                or "\x00" in value
                or len(value) > 4096
                for value in values
            ):
                raise CommandNoisePreparationError()
            row_id = row["ID"]
            length = _parse_int(row["Length"], minimum=1, maximum=_MAX_WAV_FRAMES)
            _parse_int(row["Onset"], minimum=0, maximum=100_000_000)
            _parse_int(row["N-Listeners"], minimum=1, maximum=10_000)
            _parse_int(row["Consistency"], minimum=1, maximum=10_000)
            try:
                snr = float(row["SNR"])
            except ValueError:
                raise CommandNoisePreparationError() from None
            target = row["Target"]
            confusion = row["Confusion"]
            if (
                row_id in identifiers
                or not row_id.isdigit()
                or not math.isfinite(snr)
                or row["Masker"] not in {"SSN", "BMN3", "BAB4"}
                or re.fullmatch(r"s[1-9][0-9]*", row["Speaker"]) is None
                or _WORD_RE.fullmatch(target) is None
                or _WORD_RE.fullmatch(confusion) is None
                or not row["Counts"]
                or not all(value.isdigit() for value in row["Counts"].split(" "))
            ):
                raise CommandNoisePreparationError()
            identifiers.add(row_id)
            rows.append(
                _EcccRow(
                    row_id=row_id,
                    length=length,
                    target=target,
                    confusion=confusion,
                )
            )
        if len(rows) != expected_rows:
            raise CommandNoisePreparationError()
        return tuple(rows)
    except CommandNoisePreparationError:
        raise
    except (csv.Error, StopIteration, UnicodeError, ValueError):
        raise CommandNoisePreparationError() from None


def _prepare_eccc(
    path: Path | str,
    contract: EcccArchive,
    *,
    expected_fixed_ids: tuple[str, ...] | None,
) -> _PreparedSource:
    with _open_private_archive(
        path,
        filename=contract.filename,
        size_bytes=contract.size_bytes,
        sha256=contract.sha256,
    ) as handle:
        try:
            _preflight_zip_central_directory(
                handle,
                archive_bytes=contract.size_bytes,
                expected_members=contract.member_count,
            )
            with zipfile.ZipFile(handle, mode="r", allowZip64=False) as archive:
                infos = archive.infolist()
                if (
                    archive.comment
                    or len(infos) != contract.member_count
                    or len(infos) > _MAX_ECCC_MEMBERS
                ):
                    raise CommandNoisePreparationError()
                by_name: dict[str, zipfile.ZipInfo] = {}
                directories: set[str] = set()
                file_count = 0
                expanded = 0
                wav_ids: set[str] = set()
                for info in infos:
                    name = _safe_zip_name(info)
                    if name in by_name:
                        raise CommandNoisePreparationError()
                    by_name[name] = info
                    unix_mode = (info.external_attr >> 16) & 0xFFFF
                    file_type = stat.S_IFMT(unix_mode)
                    if info.flag_bits & 0x1 or info.header_offset < 0:
                        raise CommandNoisePreparationError()
                    if info.is_dir():
                        if info.file_size != 0 or file_type not in {0, stat.S_IFDIR}:
                            raise CommandNoisePreparationError()
                        directories.add(name)
                        continue
                    if (
                        file_type not in {0, stat.S_IFREG}
                        or info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                        or not 0 < info.file_size <= _MAX_AUDIO_MEMBER_BYTES
                    ):
                        raise CommandNoisePreparationError()
                    file_count += 1
                    expanded += info.file_size
                    if expanded > _MAX_ARCHIVE_EXPANDED_BYTES:
                        raise CommandNoisePreparationError()
                    match = _ECCC_WAV_RE.fullmatch(name)
                    if match is not None:
                        if match.group(1) in wav_ids:
                            raise CommandNoisePreparationError()
                        wav_ids.add(match.group(1))
                    elif name not in _ECCC_FIXED_FILES:
                        raise CommandNoisePreparationError()
                if (
                    directories != _ECCC_DIRECTORIES
                    or file_count != contract.file_count
                    or expanded != contract.uncompressed_bytes
                ):
                    raise CommandNoisePreparationError()
                for name, expected_size, expected_sha in (
                    (_ECCC_CSV, contract.csv_size_bytes, contract.csv_sha256),
                    (
                        _ECCC_README,
                        contract.readme_size_bytes,
                        contract.readme_sha256,
                    ),
                    (
                        _ECCC_LICENSE,
                        contract.license_size_bytes,
                        contract.license_sha256,
                    ),
                ):
                    info = by_name.get(name)
                    if info is None or info.file_size != expected_size:
                        raise CommandNoisePreparationError()
                    maximum = (
                        _MAX_TEXT_MEMBER_BYTES
                        if name != _ECCC_CSV
                        else _MAX_TEXT_MEMBER_BYTES
                    )
                    if (
                        hashlib.sha256(
                            _read_zip_member(archive, info, maximum=maximum)
                        ).hexdigest()
                        != expected_sha
                    ):
                        raise CommandNoisePreparationError()
                csv_info = by_name[_ECCC_CSV]
                csv_payload = _read_zip_member(
                    archive,
                    csv_info,
                    maximum=_MAX_TEXT_MEMBER_BYTES,
                )
                rows = _parse_eccc_csv(csv_payload, expected_rows=contract.csv_rows)
                if {row.row_id for row in rows} != wav_ids:
                    raise CommandNoisePreparationError()
                selected = tuple(
                    sorted(
                        (
                            row
                            for row in rows
                            if row.target in COMMANDS
                            or (row.target not in COMMANDS and row.confusion in COMMANDS)
                        ),
                        key=lambda row: int(row.row_id),
                    )
                )
                positive_count = sum(row.target in COMMANDS for row in selected)
                negative_count = len(selected) - positive_count
                if (
                    positive_count != 6
                    or negative_count != 21
                    or (
                        expected_fixed_ids is not None
                        and tuple(row.row_id for row in selected) != expected_fixed_ids
                    )
                ):
                    raise CommandNoisePreparationError()

                prepared: list[_PreparedCase] = []
                for index, row in enumerate(selected, start=1):
                    member_name = f"{_ECCC_ROOT}/confusionWavs/T_{row.row_id}.wav"
                    info = by_name.get(member_name)
                    if info is None:
                        raise CommandNoisePreparationError()
                    payload = _read_zip_member(
                        archive,
                        info,
                        maximum=_MAX_AUDIO_MEMBER_BYTES,
                    )
                    audio, frames = _decode_pcm16_wav(payload, channels=2)
                    if frames != row.length + 2 * (SAMPLE_RATE_HZ // 5):
                        raise CommandNoisePreparationError()
                    if row.target in COMMANDS:
                        reference = row.target
                        assertion = "transcript"
                        commands = (row.target,)
                        forbidden = tuple(value for value in COMMANDS if value != row.target)
                        kind = "command-positive"
                    elif row.confusion in COMMANDS:
                        reference = row.target
                        assertion = "transcript"
                        commands = ()
                        forbidden = (row.confusion,)
                        kind = "command-negative"
                    else:
                        raise CommandNoisePreparationError()
                    prepared.append(
                        _PreparedCase(
                            case_id=f"eccc-case-{index:03d}",
                            audio_bytes=audio,
                            reference=reference,
                            assertion=assertion,
                            commands=commands,
                            forbidden_commands=forbidden,
                            tags=("public-command-noise", "eccc", "noisy", kind),
                            selection_record={
                                "member_sha256": hashlib.sha256(payload).hexdigest(),
                                "output_sha256": hashlib.sha256(audio).hexdigest(),
                                "samples": frames,
                                "selection_sha256": _digest_json(
                                    {
                                        "member": member_name,
                                        "target": row.target,
                                        "confusion": row.confusion,
                                    }
                                ),
                            },
                        )
                    )
        except CommandNoisePreparationError:
            raise
        except (EOFError, OSError, RuntimeError, ValueError, zipfile.BadZipFile):
            raise CommandNoisePreparationError() from None
    selection_sha256 = _digest_json(
        [dict(case.selection_record) for case in prepared]
    )
    return _PreparedSource(
        source_id="english-consistent-confusion-v1.2",
        archive_sha256=contract.sha256,
        cases=tuple(prepared),
        selection_sha256=selection_sha256,
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
        if _file_identity(opened) != _file_identity(marker):
            raise CommandNoisePreparationError()
        try:
            os.stat("HEAD", dir_fd=marker_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True
    except CommandNoisePreparationError:
        raise
    except (OSError, OverflowError, ValueError):
        raise CommandNoisePreparationError() from None
    finally:
        if marker_fd >= 0:
            try:
                os.close(marker_fd)
            except OSError:
                pass


def _has_git_ancestor(path: Path) -> bool:
    candidate = Path(os.path.abspath(path))
    try:
        for ancestor in (candidate, *candidate.parents):
            with opened_directory_nofollow(ancestor) as (_stable, descriptor):
                if _git_marker_present(descriptor):
                    return True
        return False
    except (BoundedReadError, CommandNoisePreparationError):
        raise CommandNoisePreparationError() from None


def _validate_output_path(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() or not candidate.name or candidate.name in {".", ".."}:
        raise CommandNoisePreparationError()
    try:
        try:
            candidate.lstat()
        except FileNotFoundError:
            pass
        else:
            raise CommandNoisePreparationError()
        parent = candidate.parent.resolve(strict=True)
        if parent != candidate.parent or _has_git_ancestor(parent):
            raise CommandNoisePreparationError()
    except (OSError, RuntimeError, ValueError):
        raise CommandNoisePreparationError() from None
    return candidate


def _source_contract(
    recipe: _RecipeLock,
    injection: TestCommandNoiseInjection | None,
) -> tuple[SpeechCommandsArchive, EcccArchive, bool, tuple[str, ...] | None]:
    if injection is None:
        gsc, eccc = _production_contracts(recipe)
        fixed_ids = _validate_catalog_binding(gsc, eccc)
        return gsc, eccc, True, fixed_ids
    if (
        type(injection) is not TestCommandNoiseInjection
        or not isinstance(injection.fixture_id, str)
        or re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,63}", injection.fixture_id) is None
    ):
        raise CommandNoisePreparationError()
    _validate_contracts(injection.speech_commands, injection.eccc)
    return injection.speech_commands, injection.eccc, False, None


def prepare_public_command_noise_corpus(
    *,
    speech_commands_archive: Path | str,
    eccc_archive: Path | str,
    output_dir: Path | str,
    accepted_terms: object = frozenset(),
    test_injection: TestCommandNoiseInjection | None = None,
    recipe_lock_path: Path | str = LOCK_PATH,
) -> PreparedCommandNoiseCorpus:
    """Validate the two local archives and publish the exact schema-v4 corpus."""

    recipe = _load_recipe_lock(recipe_lock_path)
    accepted = _validated_accepted_terms(accepted_terms)
    (
        gsc_contract,
        eccc_contract,
        production_sources,
        catalog_fixed_ids,
    ) = _source_contract(recipe, test_injection)
    destination = _validate_output_path(output_dir)
    try:
        gsc = _prepare_gsc(speech_commands_archive, gsc_contract)
        eccc = _prepare_eccc(
            eccc_archive,
            eccc_contract,
            expected_fixed_ids=catalog_fixed_ids,
        )
        sources = (gsc, eccc)
        cases = tuple(case for source in sources for case in source.cases)
        assertion_counts = {
            name: sum(case.assertion == name for case in cases)
            for name in ("transcript", "speech_negative", "silence")
        }
        if (
            len(gsc.cases) != 30
            or len(eccc.cases) != 27
            or len(cases) != 57
            or assertion_counts
            != {"transcript": 47, "speech_negative": 5, "silence": 5}
            or sum(len(case.audio_bytes) for case in cases) > MAX_CORPUS_BYTES
        ):
            raise CommandNoisePreparationError()
        command_positive = sum(bool(case.commands) for case in cases)
        command_negative = sum(
            case.assertion != "silence" and not case.commands for case in cases
        )
        if command_positive != 26 or command_negative != 26:
            raise CommandNoisePreparationError()
        source_records = [
            {
                "source_id": source.source_id,
                "archive_sha256": source.archive_sha256,
                "cases": len(source.cases),
                "selection_sha256": source.selection_sha256,
            }
            for source in sources
        ]
        source_set_sha256 = _digest_json(
            {"schema_version": 1, "sources": source_records}
        )
        pcm_set_sha256 = _digest_json(
            [
                {
                    "sha256": hashlib.sha256(case.audio_bytes).hexdigest(),
                    "samples": len(case.audio_bytes) // 4,
                }
                for case in cases
            ]
        )
        receipt = {
            "schema_version": 1,
            "kind": "public-command-noise-preparation-receipt-v1",
            "recipe_sha256": recipe.recipe_sha256,
            "recipe_lock_sha256": recipe.raw_sha256,
            "production_sources": production_sources,
            "accepted_terms": list(accepted),
            "source_set_sha256": source_set_sha256,
            "counts": {
                "sources": 2,
                "cases": len(cases),
                "command_positive": command_positive,
                "command_negative": command_negative,
                "assertions": assertion_counts,
            },
            "audio": {
                "encoding": "f32le",
                "sample_rate_hz": SAMPLE_RATE_HZ,
                "channels": 1,
                "samples": sum(len(case.audio_bytes) // 4 for case in cases),
                "pcm_set_sha256": pcm_set_sha256,
            },
            "source_aggregates": [
                {
                    "source_id": source.source_id,
                    "cases": len(source.cases),
                    "selection_sha256": source.selection_sha256,
                }
                for source in sources
            ],
        }
        receipt_payload = _canonical_json_bytes(receipt)
        receipt_sha256 = hashlib.sha256(receipt_payload).hexdigest()
        current_lock = _load_recipe_lock(recipe.path)
        if current_lock.raw_sha256 != recipe.raw_sha256 or current_lock.raw != recipe.raw:
            raise CommandNoisePreparationError()
        provenance = CorpusProvenance(
            kind="public-command-noise-v1",
            suite="command-noise-v1",
            manifest_sha256=recipe.raw_sha256,
            metadata_sha256=receipt_sha256,
            source_set_sha256=source_set_sha256,
        )
        published = publish_private_corpus(
            cases=(
                CorpusWriteCase(
                    case_id=case.case_id,
                    audio_bytes=case.audio_bytes,
                    reference=case.reference,
                    tags=case.tags,
                    commands=case.commands,
                    assertion=case.assertion,
                    forbidden_commands=case.forbidden_commands,
                )
                for case in cases
            ),
            provenance=provenance,
            output_dir=destination,
            purpose=PURPOSE,
            sidecars={"preparation-receipt.json": receipt_payload},
        )
        verify_corpus_snapshot(published)
        if (
            published.schema_version != 4
            or published.provenance != provenance
            or len(published.cases) != 57
            or tuple(case.assertion for case in published.cases)
            != tuple(case.assertion for case in cases)
        ):
            raise CommandNoisePreparationError()
        return PreparedCommandNoiseCorpus(
            corpus=published,
            receipt_sha256=receipt_sha256,
            source_set_sha256=source_set_sha256,
            recipe_lock_sha256=recipe.raw_sha256,
            production_sources=production_sources,
        )
    except CommandNoisePreparationError:
        raise
    except (CorpusError, CorpusWriterError, OSError, RuntimeError, ValueError):
        raise CommandNoisePreparationError() from None


def _safe_result(result: PreparedCommandNoiseCorpus) -> dict[str, object]:
    return {
        "ok": True,
        "schema_version": result.corpus.schema_version,
        "cases": len(result.corpus.cases),
        "corpus_sha256": result.corpus.digest,
        "receipt_sha256": result.receipt_sha256,
        "source_set_sha256": result.source_set_sha256,
        "recipe_lock_sha256": result.recipe_lock_sha256,
        "production_sources": result.production_sources,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the exact local Speech Commands/ECCC command-noise corpus; "
            "this command performs no download and prints aggregates only."
        )
    )
    parser.add_argument("--speech-commands-archive", required=True, type=Path)
    parser.add_argument("--eccc-archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--accept-term",
        action="append",
        default=[],
        help="Explicitly accept a required public-corpus license identifier.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = prepare_public_command_noise_corpus(
            speech_commands_archive=args.speech_commands_archive,
            eccc_archive=args.eccc_archive,
            output_dir=args.output_dir,
            accepted_terms=frozenset(args.accept_term),
        )
    except CommandNoisePreparationError:
        print(json.dumps(_SAFE_ERROR, sort_keys=True, separators=(",", ":")))
        return 2
    print(json.dumps(_safe_result(result), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
