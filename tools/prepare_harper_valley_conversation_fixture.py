"""Materialize the HarperValleyBank quarter of conversation fixture v1.

The command is deliberately local-only: it never clones, fetches, or opens an
audio device.  It consumes a complete owner-private checkout of the exact
upstream Git commit, verifies the commit and its authoritative trees, and
publishes 24 caller-turn cases into a new private schema-v2 corpus.

Source assumptions are fail-closed and intentionally narrower than a general
HarperValley loader:

* the checkout is clean, non-sparse, SHA-1 Git at the pinned commit/tree;
* every session has matching metadata, transcript, caller WAV, and agent WAV;
* every task record in one session has the same authoritative ``task_type``;
* caller turns use channel 1 and ``start_ms``/``duration_ms`` against the
  corresponding caller-channel PCM WAV;
* only ``human_transcript`` becomes a reference.  Machine ``transcript``,
  ``dialog_acts``, and ``emotion`` values are structurally checked but never
  become references, tags, strata, or receipt values;
* references containing the caller's metadata name, a recognizable textual
  date/time, or only known noise/paralinguistic markers are ineligible, and
  neither absolute transcript timestamps nor raw identities are written to the
  receipt.

Errors and CLI output are detail-free so a path, transcript, name, or speaker
identifier cannot escape through normal diagnostics.  Failed no-overwrite
publication is retained by the shared corpus writer as evidence.
"""

from __future__ import annotations

from array import array
import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import unicodedata
import wave

from core.wer import normalize
from tools import public_conversation_fixture as fixture
from tools.public_voice_eval_matrix import (
    Dataset,
    MatrixError,
    SELECTION_SEED,
    SelectionAlgorithm,
    dataset_by_id,
    selected_rows_sha256,
    selection_policy_sha256,
    select_subset_rows,
)
from tools.streaming_stt.bounded_io import (
    BoundedReadError,
    opened_directory_nofollow,
    read_regular_bounded,
)
from tools.streaming_stt.corpus import CorpusProvenance, LoadedCorpus, load_corpus
from tools.streaming_stt.corpus_writer import CorpusWriteCase, publish_private_corpus
from tools.streaming_stt.protocol import MAX_CORPUS_BYTES, MAX_PCM_BYTES


DATASET_ID = "harper_valley_bank_v1"
SOURCE_ID = "harper-valley"
GIT_COMMIT = "0bd721e877c4a85d8c13ff837e68661ea6200a98"
GIT_ROOT_TREE = "c0ed22efd81bcce435d7e0374bf747af8672e343"
GIT_METADATA_TREE = "1dea39eab4f00bfa59d25aa7b5eaedb72ecac138"
GIT_TRANSCRIPT_TREE = "d257637c918af5fc3f405be795299d9cb70560c4"
GIT_CALLER_AUDIO_TREE = "0ac2188002623ea0466e1f16ccaf4e8c71368423"
GIT_AGENT_AUDIO_TREE = "c9d668983c96794ac91bef5cc1774af88b58c340"
EXPECTED_SESSIONS = 1_446
REQUIRED_TERMS = frozenset({"CC-BY-4.0"})
DECODER_CONTRACT = "python-stdlib-wave-pcm-linear-mono-f32le-16000-v1"
PREPARER_CONTRACT = "public-conversation-harper-v1"
PREPARER_FILES = (
    "core/wer.py",
    "tools/prepare_harper_valley_conversation_fixture.py",
    "tools/public_conversation_fixture.py",
    "tools/public_voice_eval_matrix.py",
    "tools/streaming_stt/bounded_io.py",
    "tools/streaming_stt/corpus.py",
    "tools/streaming_stt/corpus_writer.py",
    "tools/streaming_stt/protocol.py",
)
TASK_TYPES = (
    "check balance",
    "get branch hours",
    "order checks",
    "pay bill",
    "replace card",
    "reset password",
    "schedule appointment",
    "transfer money",
)
SOURCE_RECIPE_ALGORITHM = "hash_speaker_stratified_v1"
SOURCE_RECIPE_ELIGIBILITY = (
    "caller_channel",
    "human_transcript",
    "no_names",
    "no_timestamps",
    "no_nonlexical_marker_only",
    "no_machine_transcript",
    "no_dialog_act",
    "no_emotion_labels",
)
EXPECTED_METADATA_SHA256 = (
    "f3f1333be2ec6f35186e1bb791c39eb4eb2ddf32493ce5bdaad3cb8ff0e2a260"
)
EXPECTED_ELIGIBLE_ROWS = 8_000
EXPECTED_ELIGIBLE_SET_SHA256 = (
    "6b3774c65feb03ae1b65a3456dd23c7fec452fd60d19a98da136ece3db2cb59a"
)
EXPECTED_SELECTED_ROWS_SHA256 = (
    "b4bdc5ad4cfded0551b7feb0699dc626168cddf8c5dfd38b44dff5c44ad83820"
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GIT = Path("/usr/bin/git")
_SID_RE = re.compile(r"[0-9a-f]{16}\Z")
_GIT_OID_RE = re.compile(r"[0-9a-f]{40}\Z")
_SAFE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}\Z")
_MAX_GIT_OUTPUT = 4 * 1024 * 1024
_MAX_METADATA_BYTES = 1024 * 1024
_MAX_TRANSCRIPT_BYTES = 4 * 1024 * 1024
_MAX_WAV_BYTES = 512 * 1024 * 1024
_MAX_REFERENCE_CHARS = 4_096
_MAX_TURN_MS = 30_000
_MAX_CALL_MS = 60 * 60 * 1000
_MAX_JSON_DEPTH = 32
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
_METADATA_FIELDS = {
    "agent",
    "caller",
    "end_time_ms",
    "labels",
    "session",
    "sid",
    "start_time_ms",
    "tasks",
}
_PARTY_FIELDS = {
    "arrival_time_ms",
    "hangup_time_ms",
    "metadata",
    "responses",
    "speaker_id",
    "survey_response",
}
_TRANSCRIPT_FIELDS = {
    "channel_index",
    "dialog_acts",
    "duration_ms",
    "emotion",
    "human_transcript",
    "index",
    "offset_ms",
    "speaker_role",
    "start_ms",
    "start_timestamp_ms",
    "transcript",
    "word_durations_ms",
    "word_offsets_ms",
}
_CLOCK_RE = re.compile(
    r"(?ix)(?:\b(?:[01]?\d|2[0-3]):[0-5]\d(?:\s*[ap]\.?m\.?)?\b|"
    r"\b(?:1[0-2]|0?[1-9])\s*[ap]\.?m\.?\b|"
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    r"\s+(?:a\s*m|p\s*m)\b)"
)
_DATE_RE = re.compile(
    r"(?ix)(?:\b\d{4}-\d{1,2}-\d{1,2}\b|"
    r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b|"
    r"\b(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?\b)"
)
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_NONLEXICAL_MARKERS = (
    "baby",
    "ringing",
    "laughter",
    "kids",
    "music",
    "noise",
    "unintelligible",
    "dogs",
    "cough",
)
_NONLEXICAL_MARKER_RE = re.compile(
    r"(?i)\[(?:" + "|".join(re.escape(item) for item in _NONLEXICAL_MARKERS) + r")\]"
)
_SAFE_ERROR = {
    "ok": False,
    "error": "harper_valley_fixture_prerequisites_unavailable",
}


class HarperPreparationError(RuntimeError):
    """A detail-free source, selection, decode, or publication failure."""


@dataclass(frozen=True, slots=True)
class GitEntry:
    relative: str = field(repr=False)
    oid: str


@dataclass(frozen=True, slots=True)
class TrackedSnapshot:
    relative: str = field(repr=False)
    oid: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class Candidate:
    sid: str = field(repr=False)
    turn_index: int
    task_type: str = field(repr=False)
    speaker_id: str = field(repr=False)
    reference: str = field(repr=False)
    start_ms: int
    duration_ms: int
    audio_relative: str = field(repr=False)
    audio_oid: str

    def selector_row(self) -> dict[str, object]:
        return {
            "sid": self.sid,
            "index": self.turn_index,
            "task_type": self.task_type,
            "speaker_id": self.speaker_id,
        }


@dataclass(frozen=True, slots=True)
class SourceInventory:
    candidates: tuple[Candidate, ...] = field(repr=False)
    tracked: tuple[TrackedSnapshot, ...] = field(repr=False)
    metadata_sha256: str
    eligible_set_sha256: str
    relevant_bytes: int


@dataclass(frozen=True, slots=True)
class DecodedTurn:
    raw_f32le: bytes = field(repr=False)
    samples: int


@dataclass(frozen=True, slots=True)
class PreparedHarperCorpus:
    corpus: LoadedCorpus
    receipt_sha256: str
    metadata_sha256: str
    selected_rows_sha256: str


@dataclass(frozen=True, slots=True)
class TestSourceInjection:
    """Explicit non-production source identity for synthetic contract tests."""

    fixture_id: str
    commit: str
    root_tree: str
    metadata_tree: str
    transcript_tree: str
    caller_audio_tree: str
    agent_audio_tree: str
    expected_sessions: int


@dataclass(frozen=True, slots=True)
class _RepositoryContract:
    commit: str
    root_tree: str
    metadata_tree: str
    transcript_tree: str
    caller_audio_tree: str
    agent_audio_tree: str
    expected_sessions: int
    production_evidence: bool
    fixture_id: str | None


def _repository_contract(
    injection: TestSourceInjection | None,
) -> _RepositoryContract:
    if injection is None:
        return _RepositoryContract(
            commit=GIT_COMMIT,
            root_tree=GIT_ROOT_TREE,
            metadata_tree=GIT_METADATA_TREE,
            transcript_tree=GIT_TRANSCRIPT_TREE,
            caller_audio_tree=GIT_CALLER_AUDIO_TREE,
            agent_audio_tree=GIT_AGENT_AUDIO_TREE,
            expected_sessions=EXPECTED_SESSIONS,
            production_evidence=True,
            fixture_id=None,
        )
    if type(injection) is not TestSourceInjection:
        raise HarperPreparationError()
    values = (
        injection.commit,
        injection.root_tree,
        injection.metadata_tree,
        injection.transcript_tree,
        injection.caller_audio_tree,
        injection.agent_audio_tree,
    )
    if (
        _SAFE_ID_RE.fullmatch(injection.fixture_id) is None
        or any(_GIT_OID_RE.fullmatch(value) is None for value in values)
        or type(injection.expected_sessions) is not int
        or not 1 <= injection.expected_sessions <= 10_000
    ):
        raise HarperPreparationError()
    return _RepositoryContract(
        commit=injection.commit,
        root_tree=injection.root_tree,
        metadata_tree=injection.metadata_tree,
        transcript_tree=injection.transcript_tree,
        caller_audio_tree=injection.caller_audio_tree,
        agent_audio_tree=injection.agent_audio_tree,
        expected_sessions=injection.expected_sessions,
        production_evidence=False,
        fixture_id=injection.fixture_id,
    )


def _canonical_json_bytes(value: object, *, newline: bool = False) -> bytes:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (MemoryError, TypeError, UnicodeError, ValueError, OverflowError):
        raise HarperPreparationError() from None
    return raw + (b"\n" if newline else b"")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_json(raw: bytes) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise HarperPreparationError()
            result[key] = value
        return result

    def bad_constant(_value: str) -> object:
        raise HarperPreparationError()

    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=bad_constant)
    except (
        RecursionError,
        UnicodeError,
        ValueError,
        OverflowError,
        HarperPreparationError,
    ):
        raise HarperPreparationError() from None
    try:
        if _json_depth(value) > _MAX_JSON_DEPTH:
            raise HarperPreparationError()
    except RecursionError:
        raise HarperPreparationError() from None
    return value


def _json_depth(value: object) -> int:
    if isinstance(value, dict):
        return 1 + max((_json_depth(item) for item in value.values()), default=0)
    if isinstance(value, list):
        return 1 + max((_json_depth(item) for item in value), default=0)
    return 0


def _git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw).hexdigest()  # noqa: S324 - Git object identity


def _private_directory(path: Path) -> None:
    try:
        metadata = path.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) & 0o077
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
            or path.resolve(strict=True) != path
        ):
            raise HarperPreparationError()
    except (OSError, RuntimeError, ValueError, OverflowError):
        raise HarperPreparationError() from None


def _private_file(path: Path) -> None:
    try:
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) & 0o077
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
            or path.resolve(strict=True) != path
        ):
            raise HarperPreparationError()
    except (OSError, RuntimeError, ValueError, OverflowError):
        raise HarperPreparationError() from None


def _git(root: Path, *arguments: str) -> bytes:
    if not _GIT.is_file():
        raise HarperPreparationError()
    env = {
        "HOME": os.environ.get("HOME", "/nonexistent"),
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }
    try:
        completed = subprocess.run(
            (
                str(_GIT),
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.untrackedCache=false",
                "-c",
                "protocol.file.allow=never",
                "-C",
                str(root),
                *arguments,
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        raise HarperPreparationError() from None
    if (
        completed.returncode != 0
        or len(completed.stdout) > _MAX_GIT_OUTPUT
        or len(completed.stderr) > _MAX_GIT_OUTPUT
    ):
        raise HarperPreparationError()
    return completed.stdout


def _one_git_line(root: Path, *arguments: str) -> str:
    try:
        raw = _git(root, *arguments)
        value = raw.decode("utf-8", "strict")
    except UnicodeError:
        raise HarperPreparationError() from None
    if not value.endswith("\n") or "\n" in value[:-1] or "\0" in value:
        raise HarperPreparationError()
    return value[:-1]


def _validated_repository(path: Path | str) -> Path:
    supplied = Path(path).expanduser()
    if not supplied.is_absolute():
        raise HarperPreparationError()
    candidate = Path(os.path.abspath(supplied))
    try:
        with opened_directory_nofollow(candidate, require_private=True) as (stable, _fd):
            if stable != candidate:
                raise HarperPreparationError()
        _private_directory(candidate)
        top = Path(_one_git_line(candidate, "rev-parse", "--show-toplevel"))
        if top.resolve(strict=True) != candidate:
            raise HarperPreparationError()
    except (OSError, RuntimeError, ValueError, BoundedReadError):
        raise HarperPreparationError() from None
    return candidate


def _validate_catalog_contract(dataset: Dataset) -> None:
    try:
        artifact = dataset.artifacts[0]
        checksum = artifact.checksum
        valid = (
            dataset.dataset_id == DATASET_ID
            and dataset.revision == GIT_COMMIT
            and tuple(dataset.license.acceptance_ids) == tuple(sorted(REQUIRED_TERMS))
            and len(dataset.artifacts) == 1
            and artifact.artifact_id == "repository"
            and artifact.integrity.value == "git_commit"
            and checksum is not None
            and checksum.algorithm.value == "git_sha1"
            and checksum.digest == GIT_COMMIT
            and dataset.selection.expected_examples == 24
            and dataset.selection.algorithm
            is SelectionAlgorithm.HASH_SPEAKER_STRATIFIED
            and dataset.selection.split == "all"
            and dataset.selection.identity_fields == ("sid", "index")
            and dataset.selection.strata_fields == ("task_type",)
            and dataset.selection.seed == SELECTION_SEED
            and dataset.selection.speaker_disjoint is True
            and dataset.selection.speaker_field == "speaker_id"
        )
    except (AttributeError, IndexError, TypeError, ValueError):
        valid = False
    if not valid:
        raise HarperPreparationError()


def _expected_source_recipe(dataset: Dataset) -> dict[str, object]:
    return {
        "algorithm": SOURCE_RECIPE_ALGORITHM,
        "matrix_policy_sha256": selection_policy_sha256(dataset.selection),
        "split": "all",
        "seed": SELECTION_SEED,
        "eligibility": list(SOURCE_RECIPE_ELIGIBILITY),
        "strata": [
            {"id": f"task_{index:02d}", "cases": 3}
            for index in range(len(TASK_TYPES))
        ],
        "speaker_requirement": "exactly_24_distinct",
        "scoreability": "hard_wer_nonoverlap_only",
    }


def _validate_fixture_source_contract(
    source: fixture.FixtureSource,
    *,
    dataset: Dataset,
) -> None:
    if (
        source.dataset_id != DATASET_ID
        or source.expected_cases != 24
        or dict(source.selection_recipe) != _expected_source_recipe(dataset)
    ):
        raise HarperPreparationError()


def _validate_git_state(
    root: Path,
    contract: _RepositoryContract,
) -> dict[str, GitEntry]:
    expected_refs = {
        "HEAD": contract.commit,
        "HEAD^{tree}": contract.root_tree,
        "HEAD:data/metadata": contract.metadata_tree,
        "HEAD:data/transcript": contract.transcript_tree,
        "HEAD:data/audio/caller": contract.caller_audio_tree,
        "HEAD:data/audio/agent": contract.agent_audio_tree,
    }
    for reference, expected in expected_refs.items():
        value = _one_git_line(root, "rev-parse", "--verify", reference)
        if value != expected or _GIT_OID_RE.fullmatch(value) is None:
            raise HarperPreparationError()
    if _git(root, "replace", "-l") or _git(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    ):
        raise HarperPreparationError()

    raw = _git(
        root,
        "ls-files",
        "--stage",
        "-z",
        "--",
        "data/metadata",
        "data/transcript",
        "data/audio/caller",
        "data/audio/agent",
    )
    entries: dict[str, GitEntry] = {}
    try:
        for record in raw.split(b"\0"):
            if not record:
                continue
            header, encoded_path = record.split(b"\t", 1)
            mode, oid, stage = header.decode("ascii").split(" ")
            relative = encoded_path.decode("utf-8", "strict")
            parsed = PurePosixPath(relative)
            if (
                mode != "100644"
                or stage != "0"
                or _GIT_OID_RE.fullmatch(oid) is None
                or parsed.is_absolute()
                or "\\" in relative
                or any(part in {"", ".", ".."} for part in parsed.parts)
                or relative in entries
            ):
                raise HarperPreparationError()
            entries[relative] = GitEntry(relative=relative, oid=oid)
    except (UnicodeError, ValueError, HarperPreparationError):
        raise HarperPreparationError() from None

    groups = {
        "metadata": {
            path for path in entries if path.startswith("data/metadata/")
        },
        "transcript": {
            path for path in entries if path.startswith("data/transcript/")
        },
        "caller": {
            path for path in entries if path.startswith("data/audio/caller/")
        },
        "agent": {
            path for path in entries if path.startswith("data/audio/agent/")
        },
    }
    if any(len(paths) != contract.expected_sessions for paths in groups.values()):
        raise HarperPreparationError()

    def stems(paths: Iterable[str], *, suffix: str) -> set[str]:
        result: set[str] = set()
        for relative in paths:
            name = PurePosixPath(relative).name
            if not name.endswith(suffix):
                raise HarperPreparationError()
            stem = name[: -len(suffix)]
            if _SID_RE.fullmatch(stem) is None or stem in result:
                raise HarperPreparationError()
            result.add(stem)
        return result

    observed = (
        stems(groups["metadata"], suffix=".json"),
        stems(groups["transcript"], suffix=".json"),
        stems(groups["caller"], suffix=".wav"),
        stems(groups["agent"], suffix=".wav"),
    )
    if any(value != observed[0] for value in observed[1:]):
        raise HarperPreparationError()
    return entries


def _read_tracked(
    root: Path,
    entry: GitEntry,
    *,
    maximum_bytes: int,
) -> tuple[bytes, TrackedSnapshot]:
    path = root / entry.relative
    try:
        _private_file(path)
        snapshot = read_regular_bounded(path, maximum_bytes=maximum_bytes)
    except (BoundedReadError, OSError, RuntimeError, ValueError):
        raise HarperPreparationError() from None
    if _git_blob_sha1(snapshot.data) != entry.oid:
        raise HarperPreparationError()
    return snapshot.data, TrackedSnapshot(
        relative=entry.relative,
        oid=entry.oid,
        sha256=hashlib.sha256(snapshot.data).hexdigest(),
        size_bytes=len(snapshot.data),
    )


def _speaker_id(value: object) -> str:
    if isinstance(value, bool):
        raise HarperPreparationError()
    if type(value) is int and value >= 0:
        return f"int:{value}"
    if isinstance(value, str) and value and len(value) <= 128:
        return f"str:{value}"
    raise HarperPreparationError()


def _integer(value: object, *, minimum: int = 0, maximum: int = 2**63 - 1) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise HarperPreparationError()
    return value


def _name_tokens(value: object) -> tuple[str, ...]:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise HarperPreparationError()
    normalized = unicodedata.normalize("NFKC", value).casefold()
    tokens = tuple(_WORD_RE.findall(normalized))
    if not tokens:
        raise HarperPreparationError()
    return tokens


def _reference_is_safe(reference: str, names: tuple[str, ...]) -> bool:
    lexical_reference = _NONLEXICAL_MARKER_RE.sub(" ", reference)
    if (
        not reference
        or reference != reference.strip()
        or len(reference) > _MAX_REFERENCE_CHARS
        or any(character in reference for character in ("\0", "\r", "\n"))
        or not normalize(lexical_reference)
        or _CLOCK_RE.search(reference) is not None
        or _DATE_RE.search(reference) is not None
    ):
        return False
    words = set(_WORD_RE.findall(unicodedata.normalize("NFKC", reference).casefold()))
    return not any(name in words for name in names)


def _metadata_contract(root: object, *, sid: str) -> tuple[str, str, tuple[str, ...]]:
    if not isinstance(root, dict) or set(root) != _METADATA_FIELDS:
        raise HarperPreparationError()
    if root.get("sid") != sid:
        raise HarperPreparationError()
    _integer(root.get("start_time_ms"), maximum=2**63 - 1)
    _integer(root.get("end_time_ms"), maximum=2**63 - 1)
    if not isinstance(root.get("session"), str) or not root["session"]:
        raise HarperPreparationError()
    if not isinstance(root.get("labels"), dict):
        raise HarperPreparationError()
    parties: dict[str, dict[str, object]] = {}
    for role in ("agent", "caller"):
        party = root.get(role)
        if not isinstance(party, dict) or set(party) != _PARTY_FIELDS:
            raise HarperPreparationError()
        _integer(party.get("arrival_time_ms"), maximum=2**63 - 1)
        hangup = party.get("hangup_time_ms")
        if hangup is not None:
            _integer(hangup, maximum=2**63 - 1)
        survey = party.get("survey_response")
        if (
            not isinstance(party.get("metadata"), dict)
            or not isinstance(party.get("responses"), list)
            or (survey is not None and not isinstance(survey, dict))
        ):
            raise HarperPreparationError()
        parties[role] = party
    speaker = _speaker_id(parties["caller"].get("speaker_id"))
    caller_metadata = parties["caller"]["metadata"]
    if set(caller_metadata) != {"first and last name"}:
        raise HarperPreparationError()
    names = _name_tokens(caller_metadata["first and last name"])

    tasks = root.get("tasks")
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 32:
        raise HarperPreparationError()
    task_types: list[str] = []
    for task in tasks:
        if not isinstance(task, dict) or not task:
            raise HarperPreparationError()
        task_type = task.get("task_type")
        if not isinstance(task_type, str) or task_type not in TASK_TYPES:
            raise HarperPreparationError()
        task_types.append(task_type)
    if len(set(task_types)) != 1:
        raise HarperPreparationError()
    return task_types[0], speaker, names


def _transcript_candidates(
    root: object,
    *,
    sid: str,
    task_type: str,
    speaker_id: str,
    names: tuple[str, ...],
    audio_entry: GitEntry,
) -> tuple[Candidate, ...]:
    if not isinstance(root, list) or not root or len(root) > 10_000:
        raise HarperPreparationError()
    result: list[Candidate] = []
    indices: set[int] = set()
    for value in root:
        if not isinstance(value, dict) or set(value) != _TRANSCRIPT_FIELDS:
            raise HarperPreparationError()
        role = value.get("speaker_role")
        channel = value.get("channel_index")
        if role not in {"caller", "agent"} or channel not in {1, 2}:
            raise HarperPreparationError()
        index = _integer(value.get("index"), minimum=1, maximum=1_000_000)
        if index in indices:
            raise HarperPreparationError()
        indices.add(index)
        start_ms = _integer(value.get("start_ms"), maximum=_MAX_CALL_MS)
        _integer(value.get("offset_ms"), maximum=_MAX_CALL_MS)
        duration_ms = _integer(
            value.get("duration_ms"), minimum=1, maximum=_MAX_TURN_MS
        )
        _integer(value.get("start_timestamp_ms"), maximum=2**63 - 1)
        if start_ms + duration_ms > _MAX_CALL_MS:
            raise HarperPreparationError()
        human = value.get("human_transcript")
        machine = value.get("transcript")
        if (
            not isinstance(human, str)
            or not isinstance(machine, str)
            or not isinstance(value.get("dialog_acts"), list)
            or not isinstance(value.get("emotion"), dict)
            or not isinstance(value.get("word_durations_ms"), list)
            or not isinstance(value.get("word_offsets_ms"), list)
        ):
            raise HarperPreparationError()
        # Upstream's authoritative loader assigns the caller by channel 1;
        # ``speaker_role`` contains several hundred known disagreements and is
        # therefore structural metadata only, not a source label here.
        if channel != 1:
            continue
        if not _reference_is_safe(human, names):
            continue
        result.append(
            Candidate(
                sid=sid,
                turn_index=index,
                task_type=task_type,
                speaker_id=speaker_id,
                reference=human,
                start_ms=start_ms,
                duration_ms=duration_ms,
                audio_relative=audio_entry.relative,
                audio_oid=audio_entry.oid,
            )
        )
    return tuple(result)


def _eligible_digest(rows: Sequence[Mapping[str, object]]) -> str:
    values = sorted(
        (
            str(row["sid"]),
            int(row["index"]),
            str(row["task_type"]),
            str(row["speaker_id"]),
        )
        for row in rows
    )
    return _canonical_sha256(values)


def _scan_repository(
    root: Path,
    entries: Mapping[str, GitEntry],
    contract: _RepositoryContract,
) -> SourceInventory:
    candidates: list[Candidate] = []
    tracked: list[TrackedSnapshot] = []
    closure_rows: list[tuple[str, str, str, int]] = []
    metadata_paths = sorted(
        relative for relative in entries if relative.startswith("data/metadata/")
    )
    for metadata_relative in metadata_paths:
        sid = PurePosixPath(metadata_relative).stem
        transcript_relative = f"data/transcript/{sid}.json"
        audio_relative = f"data/audio/caller/{sid}.wav"
        try:
            metadata_entry = entries[metadata_relative]
            transcript_entry = entries[transcript_relative]
            audio_entry = entries[audio_relative]
        except KeyError:
            raise HarperPreparationError() from None
        metadata_raw, metadata_snapshot = _read_tracked(
            root, metadata_entry, maximum_bytes=_MAX_METADATA_BYTES
        )
        transcript_raw, transcript_snapshot = _read_tracked(
            root, transcript_entry, maximum_bytes=_MAX_TRANSCRIPT_BYTES
        )
        tracked.extend((metadata_snapshot, transcript_snapshot))
        closure_rows.extend(
            (
                (sid, "metadata", metadata_snapshot.sha256, metadata_snapshot.size_bytes),
                (
                    sid,
                    "transcript",
                    transcript_snapshot.sha256,
                    transcript_snapshot.size_bytes,
                ),
            )
        )
        task_type, speaker_id, names = _metadata_contract(
            _strict_json(metadata_raw), sid=sid
        )
        candidates.extend(
            _transcript_candidates(
                _strict_json(transcript_raw),
                sid=sid,
                task_type=task_type,
                speaker_id=speaker_id,
                names=names,
                audio_entry=audio_entry,
            )
        )
    if not candidates:
        raise HarperPreparationError()
    rows = tuple(candidate.selector_row() for candidate in candidates)
    return SourceInventory(
        candidates=tuple(candidates),
        tracked=tuple(tracked),
        metadata_sha256=_canonical_sha256(
            {
                "commit": contract.commit,
                "metadata_tree": contract.metadata_tree,
                "transcript_tree": contract.transcript_tree,
                "files": closure_rows,
            }
        ),
        eligible_set_sha256=_eligible_digest(rows),
        relevant_bytes=sum(item.size_bytes for item in tracked),
    )


def _select_candidates(
    inventory: SourceInventory,
    *,
    dataset: Dataset,
) -> tuple[tuple[Candidate, ...], str]:
    lookup = {
        (candidate.sid, candidate.turn_index): candidate
        for candidate in inventory.candidates
    }
    if len(lookup) != len(inventory.candidates):
        raise HarperPreparationError()
    rows = tuple(candidate.selector_row() for candidate in inventory.candidates)
    try:
        matrix_selected = select_subset_rows(
            dataset.selection,
            rows,
            forbidden_speaker_ids=frozenset(),
        )
        selected_sha = selected_rows_sha256(dataset.selection, matrix_selected)
    except MatrixError:
        raise HarperPreparationError() from None
    selected: list[Candidate] = []
    for task_type in TASK_TYPES:
        task_rows = [row for row in matrix_selected if row["task_type"] == task_type]
        if len(task_rows) != 3:
            raise HarperPreparationError()
        for row in task_rows:
            try:
                selected.append(lookup[(str(row["sid"]), int(row["index"]))])
            except (KeyError, TypeError, ValueError):
                raise HarperPreparationError() from None
    if (
        len(selected) != 24
        or len({item.speaker_id for item in selected}) != 24
        or Counter(item.task_type for item in selected)
        != {task_type: 3 for task_type in TASK_TYPES}
    ):
        raise HarperPreparationError()
    return tuple(selected), selected_sha


def _decode_pcm(raw: bytes, *, sample_width: int) -> tuple[list[int], int]:
    try:
        if sample_width == 1:
            return [value - 128 for value in raw], 128
        if sample_width == 2:
            values = array("h")
            values.frombytes(raw)
            if sys.byteorder != "little":
                values.byteswap()
            return list(values), 32_768
        if sample_width == 3:
            values: list[int] = []
            for offset in range(0, len(raw), 3):
                value = int.from_bytes(raw[offset : offset + 3], "little", signed=False)
                values.append(value - (1 << 24) if value & (1 << 23) else value)
            return values, 8_388_608
        if sample_width == 4:
            values = array("i")
            values.frombytes(raw)
            if values.itemsize != 4:
                raise HarperPreparationError()
            if sys.byteorder != "little":
                values.byteswap()
            return list(values), 2_147_483_648
    except (MemoryError, OverflowError, ValueError):
        raise HarperPreparationError() from None
    raise HarperPreparationError()


def _linear_resample_f32le(
    values: Sequence[int],
    *,
    scale: int,
    source_rate: int,
) -> DecodedTurn:
    if not values or scale <= 0 or not 8_000 <= source_rate <= 96_000:
        raise HarperPreparationError()
    output_samples = max(1, (len(values) * 16_000 + source_rate // 2) // source_rate)
    if output_samples * 4 > MAX_PCM_BYTES:
        raise HarperPreparationError()
    output = array("f")
    try:
        for index in range(output_samples):
            numerator = index * source_rate
            left, remainder = divmod(numerator, 16_000)
            if left >= len(values) - 1:
                interpolated = float(values[-1])
            else:
                interpolated = (
                    values[left] * (16_000 - remainder)
                    + values[left + 1] * remainder
                ) / 16_000.0
            output.append(float(interpolated / scale))
    except (MemoryError, OverflowError, ValueError):
        raise HarperPreparationError() from None
    if sys.byteorder != "little":
        output.byteswap()
    raw = output.tobytes()
    if (
        len(raw) != output_samples * 4
        or not any(value != 0 for value in values)
        or any(not math.isfinite(value) for value in output)
    ):
        raise HarperPreparationError()
    return DecodedTurn(raw_f32le=raw, samples=output_samples)


def _decode_turn(source: bytes, *, start_ms: int, duration_ms: int) -> DecodedTurn:
    try:
        with wave.open(io.BytesIO(source), "rb") as reader:
            channels = reader.getnchannels()
            sample_width = reader.getsampwidth()
            sample_rate = reader.getframerate()
            frames = reader.getnframes()
            if (
                channels != 1
                or sample_width not in {1, 2, 3, 4}
                or not 8_000 <= sample_rate <= 96_000
                or frames <= 0
                or reader.getcomptype() != "NONE"
            ):
                raise HarperPreparationError()
            start_frame = start_ms * sample_rate // 1_000
            end_frame = (start_ms + duration_ms) * sample_rate // 1_000
            if start_frame >= frames or end_frame <= start_frame or end_frame > frames:
                raise HarperPreparationError()
            reader.setpos(start_frame)
            frame_count = end_frame - start_frame
            raw = reader.readframes(frame_count)
            if len(raw) != frame_count * sample_width:
                raise HarperPreparationError()
        values, scale = _decode_pcm(raw, sample_width=sample_width)
        decoded = _linear_resample_f32le(
            values,
            scale=scale,
            source_rate=sample_rate,
        )
        observed_ms = decoded.samples * 1_000.0 / 16_000.0
        if abs(observed_ms - duration_ms) > 2.0:
            raise HarperPreparationError()
        return decoded
    except HarperPreparationError:
        raise
    except (EOFError, MemoryError, OverflowError, ValueError, wave.Error):
        raise HarperPreparationError() from None


def _salted_digest(lock_digest: str, domain: str, value: str) -> str:
    return hashlib.sha256(
        b"harper-valley-fixture-v1\0"
        + lock_digest.encode("ascii")
        + b"\0"
        + domain.encode("ascii")
        + b"\0"
        + value.encode("utf-8")
    ).hexdigest()


def _preparer_binding(source: fixture.FixtureSource) -> dict[str, object]:
    if source.preparer_contract != PREPARER_CONTRACT or source.preparer_files != PREPARER_FILES:
        raise HarperPreparationError()
    try:
        files = {
            relative: fixture._repo_file_sha256(relative)
            for relative in source.preparer_files
        }
    except Exception:
        raise HarperPreparationError() from None
    return {"contract": PREPARER_CONTRACT, "files": files}


def _decoder_binding(
    preparer: Mapping[str, object],
    *,
    contract: _RepositoryContract,
) -> dict[str, object]:
    files = preparer.get("files")
    if not isinstance(files, dict) or tuple(files) != PREPARER_FILES:
        raise HarperPreparationError()
    decoder_contract = DECODER_CONTRACT
    if not contract.production_evidence:
        if contract.fixture_id is None:
            raise HarperPreparationError()
        decoder_contract = f"test-source-{contract.fixture_id}-wave-f32le-v1"
        if _SAFE_ID_RE.fullmatch(decoder_contract) is None:
            raise HarperPreparationError()
    try:
        total_bytes = sum((_REPO_ROOT / relative).stat().st_size for relative in files)
        closure = fixture.decoder_closure_sha256(decoder_contract, files)
    except Exception:
        raise HarperPreparationError() from None
    if total_bytes <= 0:
        raise HarperPreparationError()
    return {
        "contract": decoder_contract,
        "closure_sha256": closure,
        "files": len(files),
        "bytes": total_bytes,
        "production_evidence": contract.production_evidence,
    }


def _has_git_ancestor(path: Path) -> bool:
    try:
        for ancestor in (path, *path.parents):
            marker = ancestor / ".git"
            try:
                metadata = marker.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISREG(metadata.st_mode):
                if metadata.st_size > 0:
                    return True
                continue
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                return True
            try:
                head = (marker / "HEAD").lstat()
            except FileNotFoundError:
                # Managed sandboxes may place an empty, read-only ``.git``
                # sentinel at /tmp.  It is not a repository marker.
                continue
            if stat.S_ISREG(head.st_mode) and head.st_size > 0:
                return True
            return True
    except (OSError, RuntimeError, ValueError):
        raise HarperPreparationError() from None
    return False


def _validated_output(path: Path | str, *, source_root: Path) -> Path:
    supplied = Path(path).expanduser()
    if not supplied.is_absolute():
        raise HarperPreparationError()
    candidate = Path(os.path.abspath(supplied))
    try:
        parent = candidate.parent.resolve(strict=True)
        if (
            candidate.name in {"", ".", ".."}
            or candidate.exists()
            or _has_git_ancestor(parent)
            or source_root == candidate
            or source_root in candidate.parents
        ):
            raise HarperPreparationError()
        with opened_directory_nofollow(parent) as (stable, _fd):
            if stable != parent:
                raise HarperPreparationError()
    except HarperPreparationError:
        raise
    except (OSError, RuntimeError, ValueError, BoundedReadError):
        raise HarperPreparationError() from None
    return candidate


def _verify_snapshots(root: Path, snapshots: Sequence[TrackedSnapshot]) -> None:
    for saved in snapshots:
        entry = GitEntry(relative=saved.relative, oid=saved.oid)
        raw, current = _read_tracked(
            root,
            entry,
            maximum_bytes=saved.size_bytes,
        )
        if (
            len(raw) != saved.size_bytes
            or current.sha256 != saved.sha256
            or current != saved
        ):
            raise HarperPreparationError()


def prepare_harper_valley_conversation_fixture(
    *,
    repository: Path | str,
    output_dir: Path | str,
    accepted_terms: frozenset[str],
    lock_path: Path | str = fixture.DEFAULT_LOCK,
    test_source_injection: TestSourceInjection | None = None,
) -> PreparedHarperCorpus:
    """Verify one exact checkout and publish its deterministic 24-case slice."""

    if not isinstance(accepted_terms, frozenset) or not REQUIRED_TERMS <= accepted_terms:
        raise HarperPreparationError()
    root = _validated_repository(repository)
    destination = _validated_output(output_dir, source_root=root)
    lock = fixture.load_fixture_lock(lock_path)
    source = lock.source_by_id(SOURCE_ID)
    dataset = dataset_by_id(DATASET_ID)
    repository_contract = _repository_contract(test_source_injection)
    if repository_contract.production_evidence:
        _validate_catalog_contract(dataset)
    _validate_fixture_source_contract(source, dataset=dataset)
    preparer = _preparer_binding(source)
    decoder = _decoder_binding(preparer, contract=repository_contract)

    entries = _validate_git_state(root, repository_contract)
    inventory = _scan_repository(root, entries, repository_contract)
    selected, selected_sha = _select_candidates(inventory, dataset=dataset)
    if repository_contract.production_evidence and (
        inventory.metadata_sha256 != EXPECTED_METADATA_SHA256
        or len(inventory.candidates) != EXPECTED_ELIGIBLE_ROWS
        or inventory.eligible_set_sha256 != EXPECTED_ELIGIBLE_SET_SHA256
        or selected_sha != EXPECTED_SELECTED_ROWS_SHA256
        or any(
            _NONLEXICAL_MARKER_RE.search(candidate.reference) is not None
            for candidate in selected
        )
    ):
        raise HarperPreparationError()
    slots = lock.slots_for(SOURCE_ID)

    write_cases: list[CorpusWriteCase] = []
    receipt_cases: list[dict[str, object]] = []
    audio_snapshots: dict[str, TrackedSnapshot] = {}
    total_samples = 0
    total_pcm_bytes = 0
    for ordinal, (candidate, slot) in enumerate(zip(selected, slots, strict=True)):
        task_index = ordinal // 3
        if (
            slot.ordinal != ordinal
            or slot.stratum != f"task_{task_index:02d}"
            or slot.channel != "caller"
            or candidate.task_type != TASK_TYPES[task_index]
        ):
            raise HarperPreparationError()
        entry = entries.get(candidate.audio_relative)
        if entry is None or entry.oid != candidate.audio_oid:
            raise HarperPreparationError()
        source_audio, audio_snapshot = _read_tracked(
            root, entry, maximum_bytes=_MAX_WAV_BYTES
        )
        previous = audio_snapshots.get(audio_snapshot.relative)
        if previous is not None and previous != audio_snapshot:
            raise HarperPreparationError()
        audio_snapshots[audio_snapshot.relative] = audio_snapshot
        decoded = _decode_turn(
            source_audio,
            start_ms=candidate.start_ms,
            duration_ms=candidate.duration_ms,
        )
        total_samples += decoded.samples
        total_pcm_bytes += len(decoded.raw_f32le)
        if total_pcm_bytes > MAX_CORPUS_BYTES:
            raise HarperPreparationError()
        identity = _salted_digest(
            lock.recipe_sha256,
            "row",
            f"{candidate.sid}\0{candidate.turn_index}",
        )
        case_id = f"harper-{ordinal:02d}-{identity[:12]}"
        if _SAFE_ID_RE.fullmatch(case_id) is None:
            raise HarperPreparationError()
        tags = tuple(dict.fromkeys((*source.required_tags, slot.stratum)))
        pcm_sha = hashlib.sha256(decoded.raw_f32le).hexdigest()
        write_cases.append(
            CorpusWriteCase(
                case_id=case_id,
                audio_bytes=decoded.raw_f32le,
                reference=candidate.reference,
                tags=tags,
            )
        )
        receipt_cases.append(
            {
                "slot_id": slot.slot_id,
                "case_id": case_id,
                "identity_sha256": identity,
                "speaker_sha256": _salted_digest(
                    lock.recipe_sha256, "speaker", candidate.speaker_id
                ),
                "reference_sha256": hashlib.sha256(
                    candidate.reference.encode("utf-8")
                ).hexdigest(),
                "source_audio_sha256": audio_snapshot.sha256,
                "source_audio_bytes": audio_snapshot.size_bytes,
                "pcm_sha256": pcm_sha,
                "pcm_samples": decoded.samples,
                "tags": list(tags),
                "attributes": {
                    "task_type_index": task_index,
                    "task_type_sha256": _salted_digest(
                        lock.recipe_sha256, "task", candidate.task_type
                    ),
                    "channel": "caller",
                    "pii_scan_passed": True,
                },
            }
        )

    selected_audio = tuple(sorted(audio_snapshots.values(), key=lambda item: item.relative))
    source_closure_sha = _canonical_sha256(
        {
            "commit": repository_contract.commit,
            "root_tree": repository_contract.root_tree,
            "metadata_sha256": inventory.metadata_sha256,
            "selected_audio": [
                (item.oid, item.sha256, item.size_bytes) for item in selected_audio
            ],
        }
    )
    relevant_bytes = inventory.relevant_bytes + sum(
        item.size_bytes for item in selected_audio
    )
    artifact = dataset.artifacts[0]
    checksum = artifact.checksum
    if checksum is None:
        raise HarperPreparationError()
    receipt = {
        "schema_version": 1,
        "kind": fixture.RECEIPT_KIND,
        "fixture_id": fixture.FIXTURE_ID,
        "lock_recipe_sha256": lock.recipe_sha256,
        "source_id": SOURCE_ID,
        "source_recipe_sha256": source.selection_recipe_sha256,
        "dataset_contract_sha256": source.dataset_contract_sha256,
        "accepted_terms": sorted(dataset.license.acceptance_ids),
        "artifacts": [
            {
                "artifact_id": artifact.artifact_id,
                "upstream_algorithm": checksum.algorithm.value,
                "upstream_digest": checksum.digest,
                "size_bytes": relevant_bytes,
                "local_sha256": source_closure_sha,
            }
        ],
        "metadata_sha256": inventory.metadata_sha256,
        "selection": {
            "policy_sha256": selection_policy_sha256(dataset.selection),
            "eligible_rows": len(inventory.candidates),
            "eligible_set_sha256": inventory.eligible_set_sha256,
            "selected_rows_sha256": selected_sha,
            "selected_cases": len(receipt_cases),
            "selected_case_set_sha256": fixture._canonical_sha256(receipt_cases),
            "parent_receipt_sha256": None,
            "parent_corpus_manifest_sha256": None,
        },
        "preparer": preparer,
        "decoder": decoder,
        "cases": receipt_cases,
        "totals": {
            "cases": len(receipt_cases),
            "pcm_samples": total_samples,
            "pcm_bytes": total_samples * 4,
        },
        "privacy": dict(_PRIVATE_RECEIPT_PRIVACY),
        "evidence_scope": dict(_RECEIPT_EVIDENCE_SCOPE),
    }
    receipt_raw = _canonical_json_bytes(receipt, newline=True)
    receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
    provenance = CorpusProvenance(
        kind="public-voice-v1",
        suite=source.corpus_suite,
        manifest_sha256=source.selection_recipe_sha256,
        metadata_sha256=inventory.metadata_sha256,
        source_set_sha256=receipt_sha,
    )
    try:
        corpus = publish_private_corpus(
            cases=write_cases,
            provenance=provenance,
            output_dir=destination,
            purpose="HarperValleyBank conversation fixture caller turns",
            sidecars={"preparation-receipt.json": receipt_raw},
        )
    except Exception:
        raise HarperPreparationError() from None

    _validate_git_state(root, repository_contract)
    _verify_snapshots(root, (*inventory.tracked, *selected_audio))
    if _preparer_binding(source) != preparer or _decoder_binding(
        preparer, contract=repository_contract
    ) != decoder:
        raise HarperPreparationError()
    if repository_contract.production_evidence:
        try:
            binding = fixture.validate_private_source(
                corpus.path,
                lock=lock,
                expected_source_id=SOURCE_ID,
            )
        except Exception:
            raise HarperPreparationError() from None
        if binding.corpus_manifest_sha256 != corpus.digest:
            raise HarperPreparationError()
    elif load_corpus(corpus.path).digest != corpus.digest:
        raise HarperPreparationError()
    return PreparedHarperCorpus(
        corpus=corpus,
        receipt_sha256=receipt_sha,
        metadata_sha256=inventory.metadata_sha256,
        selected_rows_sha256=selected_sha,
    )


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise HarperPreparationError()


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description=(
            "Prepare the pinned HarperValleyBank 24-case private conversation fixture; "
            "the command never downloads or prints private source values."
        )
    )
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=fixture.DEFAULT_LOCK)
    parser.add_argument("--accept-term", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        result = prepare_harper_valley_conversation_fixture(
            repository=args.repository,
            output_dir=args.output_dir,
            accepted_terms=frozenset(args.accept_term),
            lock_path=args.lock,
        )
        payload = {
            "ok": True,
            "source_id": SOURCE_ID,
            "cases": len(result.corpus.cases),
            "receipt_sha256": result.receipt_sha256,
            "corpus_manifest_sha256": result.corpus.digest,
        }
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return 0
    except Exception:
        print(json.dumps(_SAFE_ERROR, sort_keys=True, separators=(",", ":")))
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised through main
    raise SystemExit(main())


__all__ = [
    "DECODER_CONTRACT",
    "GIT_COMMIT",
    "GIT_ROOT_TREE",
    "HarperPreparationError",
    "PREPARER_FILES",
    "PreparedHarperCorpus",
    "TASK_TYPES",
    "TestSourceInjection",
    "main",
    "prepare_harper_valley_conversation_fixture",
]
