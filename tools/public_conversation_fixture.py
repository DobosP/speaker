"""Strict binding for the private 96-case public conversation/STT fixture.

The committed lock contains source recipes and abstract slots only.  Audio,
literal references, selected source identities, and local paths remain in four
separate owner-private schema-v2 streaming corpora.  Each corpus is bound to a
fixed sibling ``preparation-receipt.json`` and is validated independently so
the existing 32 MiB eager-PCM limit is never widened.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import argparse
import hashlib
from itertools import islice
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys

from tools.public_voice_eval_matrix import (
    MATRIX_VERSION,
    Artifact,
    Checksum,
    Dataset,
    SelectionAlgorithm,
    SelectionPolicy,
    dataset_by_id,
    selection_policy_sha256,
)
from tools.streaming_stt.bounded_io import opened_directory_nofollow, read_regular_bounded
from tools.streaming_stt.corpus import load_corpus, verify_corpus_snapshot
from tools.streaming_stt.protocol import MAX_CORPUS_BYTES


DEFAULT_LOCK = (
    Path(__file__).resolve().parent
    / "streaming_stt"
    / "public-conversation-96.lock.json"
)
FIXTURE_ID = "public-conversation-96-v1"
LOCK_KIND = "public-conversation-fixture-lock-v1"
RECEIPT_KIND = "public-conversation-source-receipt-v1"
SOURCE_IDS = (
    "harper-valley",
    "edacc-test",
    "ami-es2004a-close-far",
    "cvss4-en",
)
DATASET_IDS = (
    "harper_valley_bank_v1",
    "edacc_v1_test",
    "ami_es2004a_close_far_v1",
    "common_voice_spontaneous_4_en",
)
CASES_PER_SOURCE = 24
TOTAL_CASES = 96

_SAFE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,95}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_DRIVE_PATH_RE = re.compile(r"[A-Za-z]:[\\/]")
_MAX_LOCK_BYTES = 256 * 1024
_MAX_RECEIPT_BYTES = 256 * 1024
_MAX_DECODER_FILES = 100_000
_MAX_DECODER_BYTES = 128 * 1024 * 1024 * 1024
_MAX_SOURCE_AUDIO_BYTES = 8 * 1024 * 1024 * 1024
_MAX_PCM_SAMPLES = MAX_CORPUS_BYTES // 4

_LOCK_ROOT_FIELDS = {
    "schema_version",
    "kind",
    "fixture_id",
    "matrix_version",
    "selection_seed",
    "sample_format",
    "totals",
    "privacy",
    "evidence_scope",
    "sources",
    "slots",
    "recipe_sha256",
}
_LOCK_SOURCE_FIELDS = {
    "source_id",
    "dataset_id",
    "dataset_contract_sha256",
    "corpus_suite",
    "expected_cases",
    "selection_recipe",
    "selection_recipe_sha256",
    "receipt_kind",
    "preparer_contract",
    "preparer_files",
    "decoder_contract",
    "required_tags",
}
_LOCK_SLOT_FIELDS = {
    "slot_id",
    "source_id",
    "ordinal",
    "stratum",
    "channel",
    "pair_id",
    "scoreability",
}
_RECEIPT_FIELDS = {
    "schema_version",
    "kind",
    "fixture_id",
    "lock_recipe_sha256",
    "source_id",
    "source_recipe_sha256",
    "dataset_contract_sha256",
    "accepted_terms",
    "artifacts",
    "metadata_sha256",
    "selection",
    "preparer",
    "decoder",
    "cases",
    "totals",
    "privacy",
    "evidence_scope",
}
_ARTIFACT_FIELDS = {
    "artifact_id",
    "upstream_algorithm",
    "upstream_digest",
    "size_bytes",
    "local_sha256",
}
_SELECTION_FIELDS = {
    "policy_sha256",
    "eligible_rows",
    "eligible_set_sha256",
    "selected_rows_sha256",
    "selected_cases",
    "selected_case_set_sha256",
    "parent_receipt_sha256",
    "parent_corpus_manifest_sha256",
}
_PREPARER_FIELDS = {"contract", "files"}
_DECODER_FIELDS = {
    "contract",
    "closure_sha256",
    "files",
    "bytes",
    "production_evidence",
}
_RECEIPT_CASE_FIELDS = {
    "slot_id",
    "case_id",
    "identity_sha256",
    "speaker_sha256",
    "reference_sha256",
    "source_audio_sha256",
    "source_audio_bytes",
    "pcm_sha256",
    "pcm_samples",
    "tags",
    "attributes",
}
_PRIVATE_RECEIPT_PRIVACY = {
    "contains_audio": False,
    "contains_transcripts": False,
    "contains_local_paths": False,
    "contains_raw_identifiers": False,
}
_LOCK_PRIVACY = {
    "audio_committed": False,
    "transcripts_committed": False,
    "selected_identities_committed": False,
    "local_paths_committed": False,
    "private_receipts_required": True,
}
_LOCK_EVIDENCE_SCOPE = {
    "development_only": True,
    "hard_wer_requires_unambiguous_nonoverlap": True,
    "ambiguous_overlap": "diagnostic_only",
    "capture": False,
    "vad": False,
    "aec": False,
    "agent_tools": False,
    "live_latency": False,
    "default_promotion": False,
}
_RECEIPT_EVIDENCE_SCOPE = {
    **_LOCK_EVIDENCE_SCOPE,
    "pcm_preparation_complete": True,
    "stt_model_run": False,
}
_SOURCE_PREPARER_FILES = {
    "harper-valley": (
        "core/wer.py",
        "tools/prepare_harper_valley_conversation_fixture.py",
        "tools/public_conversation_fixture.py",
        "tools/public_voice_eval_matrix.py",
        "tools/streaming_stt/bounded_io.py",
        "tools/streaming_stt/corpus.py",
        "tools/streaming_stt/corpus_writer.py",
        "tools/streaming_stt/protocol.py",
    ),
    "edacc-test": (
        "core/wer.py",
        "tools/prepare_edacc_conversation_fixture.py",
        "tools/public_conversation_fixture.py",
        "tools/public_voice_eval_matrix.py",
        "tools/streaming_stt/bounded_io.py",
        "tools/streaming_stt/corpus.py",
        "tools/streaming_stt/corpus_writer.py",
        "tools/streaming_stt/protocol.py",
    ),
    "ami-es2004a-close-far": (
        "core/wer.py",
        "tools/prepare_ami_capture_replay.py",
        "tools/prepare_ami_conversation_fixture.py",
        "tools/public_conversation_fixture.py",
        "tools/public_voice_eval_matrix.py",
        "tools/streaming_stt/bounded_io.py",
        "tools/streaming_stt/corpus.py",
        "tools/streaming_stt/corpus_writer.py",
        "tools/streaming_stt/protocol.py",
    ),
    "cvss4-en": (
        "core/wer.py",
        "tools/prepare_common_voice_conversation_fixture.py",
        "tools/prepare_common_voice_spontaneous_v4.py",
        "tools/public_conversation_fixture.py",
        "tools/public_voice_eval_matrix.py",
        "tools/streaming_stt/bounded_io.py",
        "tools/streaming_stt/corpus.py",
        "tools/streaming_stt/corpus_writer.py",
        "tools/streaming_stt/protocol.py",
    ),
}
_SOURCE_PREPARER_CONTRACTS = {
    "harper-valley": "public-conversation-harper-v1",
    "edacc-test": "public-conversation-edacc-v1",
    "ami-es2004a-close-far": "public-conversation-ami-v1",
    "cvss4-en": "public-conversation-cv-projection-v1",
}
_SOURCE_DECODER_CONTRACTS = {
    "harper-valley": "python-stdlib-wave-pcm-linear-mono-f32le-16000-v1",
    "edacc-test": "edacc-wav-pcm16-mono32k-box2-f32le16k-v1",
    "ami-es2004a-close-far": "ami-riff-pcm16-mono16k-window-to-f32le-v1",
    "cvss4-en": "verified-cvss4-parent-f32le-copy-v1",
}
_DURATION_BOUNDS = {
    "d0_2_6s": (2_000, 6_000, False),
    "d1_6_10s": (6_000, 10_000, False),
    "d2_10_15s": (10_000, 15_000, False),
    "d3_15_25s": (15_000, 25_000, True),
}


class FixtureError(RuntimeError):
    """A detail-free lock, receipt, corpus, or snapshot validation failure."""


@dataclass(frozen=True)
class FixtureSlot:
    slot_id: str
    source_id: str
    ordinal: int
    stratum: str
    channel: str | None
    pair_id: str | None
    scoreability: str


@dataclass(frozen=True)
class FixtureSource:
    source_id: str
    dataset_id: str
    dataset_contract_sha256: str
    corpus_suite: str
    expected_cases: int
    selection_recipe_sha256: str
    receipt_kind: str
    preparer_contract: str
    preparer_files: tuple[str, ...]
    decoder_contract: str
    required_tags: tuple[str, ...]
    selection_recipe: Mapping[str, object] = field(repr=False)


@dataclass(frozen=True)
class FixtureLock:
    path: Path = field(repr=False)
    raw_sha256: str
    recipe_sha256: str
    selection_seed: str
    sources: tuple[FixtureSource, ...]
    slots: tuple[FixtureSlot, ...]

    def source_by_id(self, source_id: str) -> FixtureSource:
        try:
            return next(item for item in self.sources if item.source_id == source_id)
        except StopIteration:
            raise FixtureError() from None

    def slots_for(self, source_id: str) -> tuple[FixtureSlot, ...]:
        result = tuple(item for item in self.slots if item.source_id == source_id)
        if len(result) != CASES_PER_SOURCE:
            raise FixtureError()
        return result


@dataclass(frozen=True)
class FixtureSourceBinding:
    source_id: str
    corpus_path: Path = field(repr=False)
    receipt_path: Path = field(repr=False)
    receipt_sha256: str
    corpus_manifest_sha256: str
    cases: int
    pcm_bytes: int


@dataclass(frozen=True)
class FixtureSet:
    lock: FixtureLock
    sources: tuple[FixtureSourceBinding, ...]
    fixture_set_sha256: str
    cases: int
    pcm_bytes: int

    @property
    def corpus_paths(self) -> tuple[Path, ...]:
        return tuple(item.corpus_path for item in self.sources)


def _bad() -> object:
    raise FixtureError()


def _strict_json(raw: bytes) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise FixtureError()
            result[key] = value
        return result

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: _bad())
    except (UnicodeError, ValueError, OverflowError, FixtureError):
        raise FixtureError() from None


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (MemoryError, TypeError, UnicodeError, ValueError, OverflowError):
        raise FixtureError() from None


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _safe_id(value: object) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise FixtureError()
    return value


def _sha256(value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise FixtureError()
    return value


def _integer(
    value: object,
    *,
    minimum: int = 0,
    maximum: int = 2**63 - 1,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise FixtureError()
    return value


def _string_list(
    value: object,
    *,
    maximum_items: int,
    exact_items: int | None = None,
) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or len(value) > maximum_items
        or (exact_items is not None and len(value) != exact_items)
    ):
        raise FixtureError()
    result = tuple(_safe_id(item) for item in value)
    if len(set(result)) != len(result):
        raise FixtureError()
    return result


def _relative_file(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise FixtureError()
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise FixtureError()
    if not all(_SAFE_ID_RE.fullmatch(part) is not None for part in path.parts):
        raise FixtureError()
    return value


def _assert_path_free(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise FixtureError()
            _assert_path_free(item)
        return
    if isinstance(value, list):
        for item in value:
            _assert_path_free(item)
        return
    if isinstance(value, str) and (
        value.startswith(("/", "\\", "file://"))
        or _DRIVE_PATH_RE.match(value) is not None
        or "../" in value
        or "..\\" in value
    ):
        raise FixtureError()


def _enum_value(value: object) -> object:
    return value.value if isinstance(value, Enum) else value


def _checksum_payload(value: Checksum | None) -> object:
    if value is None:
        return None
    return {
        "algorithm": value.algorithm.value,
        "digest": value.digest,
        "source": value.source,
    }


def _artifact_payload(value: Artifact) -> dict[str, object]:
    return {
        "artifact_id": value.artifact_id,
        "locator": value.locator,
        "filename": value.filename,
        "size_bytes": value.size_bytes,
        "integrity": value.integrity.value,
        "checksum": _checksum_payload(value.checksum),
        "integrity_note": value.integrity_note,
    }


def _selection_payload(value: SelectionPolicy) -> dict[str, object]:
    return {
        "description": value.description,
        "algorithm": value.algorithm.value,
        "split": value.split,
        "identity_fields": list(value.identity_fields),
        "strata_fields": list(value.strata_fields),
        "seed": value.seed,
        "expected_examples": value.expected_examples,
        "speaker_disjoint": value.speaker_disjoint,
        "speaker_field": value.speaker_field,
        "fixed_ids": list(value.fixed_ids),
    }


def dataset_contract_sha256(dataset: Dataset) -> str:
    """Bind the complete selectable catalog contract for one source."""

    if type(dataset) is not Dataset:
        raise FixtureError()
    payload = {
        "dataset_id": dataset.dataset_id,
        "name": dataset.name,
        "upstream_id": dataset.upstream_id,
        "revision": dataset.revision,
        "canonical_url": dataset.canonical_url,
        "license": {
            "license_id": dataset.license.license_id,
            "license_url": dataset.license.license_url,
            "acceptance_ids": list(dataset.license.acceptance_ids),
            "redistribution": dataset.license.redistribution,
        },
        "artifacts": [_artifact_payload(item) for item in dataset.artifacts],
        "selection": _selection_payload(dataset.selection),
        "credential_env": dataset.credential_env,
        "evidence_note": dataset.evidence_note,
        "evaluation_only": dataset.evaluation_only,
        "usage_constraints": [_enum_value(item) for item in dataset.usage_constraints],
    }
    return _canonical_sha256(payload)


def _validate_source_recipe(value: object, *, dataset: Dataset) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise FixtureError()
    expected_fields = {
        "algorithm",
        "matrix_policy_sha256",
        "split",
        "seed",
        "eligibility",
        "strata",
        "speaker_requirement",
        "scoreability",
    }
    if set(value) != expected_fields:
        raise FixtureError()
    algorithm = _safe_id(value.get("algorithm"))
    if value.get("matrix_policy_sha256") != selection_policy_sha256(dataset.selection):
        raise FixtureError()
    split = _safe_id(value.get("split"))
    seed = _safe_id(value.get("seed"))
    if algorithm == "hash_speaker_stratified_v1" and (
        dataset.selection.algorithm
        is not SelectionAlgorithm.HASH_SPEAKER_STRATIFIED
        or split != dataset.selection.split
        or seed != dataset.selection.seed
    ):
        raise FixtureError()
    eligibility = _string_list(value.get("eligibility"), maximum_items=32)
    if not eligibility:
        raise FixtureError()
    strata = value.get("strata")
    if not isinstance(strata, list) or not strata:
        raise FixtureError()
    total = 0
    seen: set[str] = set()
    for item in strata:
        if not isinstance(item, dict) or set(item) != {"id", "cases"}:
            raise FixtureError()
        stratum = _safe_id(item.get("id"))
        if stratum in seen:
            raise FixtureError()
        seen.add(stratum)
        total += _integer(item.get("cases"), minimum=1, maximum=CASES_PER_SOURCE)
    if total != CASES_PER_SOURCE:
        raise FixtureError()
    _safe_id(value.get("speaker_requirement"))
    if value.get("scoreability") != "hard_wer_nonoverlap_only":
        raise FixtureError()
    return value


def _validate_slot_shape(
    value: object,
    *,
    source_id: str,
    ordinal: int,
) -> FixtureSlot:
    if not isinstance(value, dict) or set(value) != _LOCK_SLOT_FIELDS:
        raise FixtureError()
    slot_id = _safe_id(value.get("slot_id"))
    if value.get("source_id") != source_id:
        raise FixtureError()
    if _integer(value.get("ordinal"), maximum=CASES_PER_SOURCE - 1) != ordinal:
        raise FixtureError()
    stratum = _safe_id(value.get("stratum"))
    channel_value = value.get("channel")
    channel = None if channel_value is None else _safe_id(channel_value)
    pair_value = value.get("pair_id")
    pair_id = None if pair_value is None else _safe_id(pair_value)
    if value.get("scoreability") != "hard_wer":
        raise FixtureError()
    return FixtureSlot(
        slot_id=slot_id,
        source_id=source_id,
        ordinal=ordinal,
        stratum=stratum,
        channel=channel,
        pair_id=pair_id,
        scoreability="hard_wer",
    )


def _validate_committed_slots(slots: tuple[FixtureSlot, ...]) -> None:
    if len(slots) != TOTAL_CASES or len({item.slot_id for item in slots}) != TOTAL_CASES:
        raise FixtureError()
    by_source = {source_id: tuple(s for s in slots if s.source_id == source_id) for source_id in SOURCE_IDS}
    if any(len(items) != CASES_PER_SOURCE for items in by_source.values()):
        raise FixtureError()

    harper = by_source["harper-valley"]
    if (
        Counter(item.stratum for item in harper)
        != {f"task_{index:02d}": 3 for index in range(8)}
        or any(item.channel != "caller" or item.pair_id is not None for item in harper)
    ):
        raise FixtureError()

    edacc = by_source["edacc-test"]
    if Counter(item.stratum for item in edacc) != {
        "clean": 8,
        "disfluent": 8,
        "overlap_adjacent": 8,
    } or any(item.channel != "single" or item.pair_id is not None for item in edacc):
        raise FixtureError()

    ami = by_source["ami-es2004a-close-far"]
    pairs: dict[str, list[FixtureSlot]] = {}
    for item in ami:
        if item.pair_id is None or item.channel not in {"close", "far"}:
            raise FixtureError()
        pairs.setdefault(item.pair_id, []).append(item)
    if len(pairs) != 12:
        raise FixtureError()
    kinds: Counter[str] = Counter()
    for pair in pairs.values():
        if len(pair) != 2 or {item.channel for item in pair} != {"close", "far"}:
            raise FixtureError()
        if len({item.stratum for item in pair}) != 1:
            raise FixtureError()
        kinds[pair[0].stratum] += 1
    if kinds != {"isolated": 8, "turn_transition": 4}:
        raise FixtureError()

    cv = by_source["cvss4-en"]
    if Counter(item.stratum for item in cv) != {
        "d0_2_6s": 6,
        "d1_6_10s": 6,
        "d2_10_15s": 6,
        "d3_15_25s": 6,
    } or any(item.channel != "original" or item.pair_id is not None for item in cv):
        raise FixtureError()


def load_fixture_lock(path: Path | str = DEFAULT_LOCK) -> FixtureLock:
    """Load and validate the committed transcript/path-free recipe lock."""

    try:
        snapshot = read_regular_bounded(Path(path), maximum_bytes=_MAX_LOCK_BYTES)
        root = _strict_json(snapshot.data)
        if not isinstance(root, dict) or set(root) != _LOCK_ROOT_FIELDS:
            raise FixtureError()
        _assert_path_free(root)
        if (
            root.get("schema_version") != 1
            or root.get("kind") != LOCK_KIND
            or root.get("fixture_id") != FIXTURE_ID
            or root.get("matrix_version") != MATRIX_VERSION
            or root.get("sample_format")
            != {"encoding": "f32le", "sample_rate_hz": 16_000, "channels": 1}
            or root.get("totals")
            != {"sources": 4, "cases": TOTAL_CASES, "cases_per_source": CASES_PER_SOURCE}
            or root.get("privacy") != _LOCK_PRIVACY
            or root.get("evidence_scope") != _LOCK_EVIDENCE_SCOPE
        ):
            raise FixtureError()
        selection_seed = _safe_id(root.get("selection_seed"))
        recipe_sha256 = _sha256(root.get("recipe_sha256"))
        digest_payload = dict(root)
        del digest_payload["recipe_sha256"]
        if _canonical_sha256(digest_payload) != recipe_sha256:
            raise FixtureError()

        source_values = root.get("sources")
        if not isinstance(source_values, list) or len(source_values) != 4:
            raise FixtureError()
        sources: list[FixtureSource] = []
        for index, value in enumerate(source_values):
            if not isinstance(value, dict) or set(value) != _LOCK_SOURCE_FIELDS:
                raise FixtureError()
            source_id = _safe_id(value.get("source_id"))
            dataset_id = _safe_id(value.get("dataset_id"))
            if source_id != SOURCE_IDS[index] or dataset_id != DATASET_IDS[index]:
                raise FixtureError()
            dataset = dataset_by_id(dataset_id)
            contract_sha = _sha256(value.get("dataset_contract_sha256"))
            if dataset_contract_sha256(dataset) != contract_sha:
                raise FixtureError()
            corpus_suite = _safe_id(value.get("corpus_suite"))
            if value.get("expected_cases") != CASES_PER_SOURCE:
                raise FixtureError()
            recipe = _validate_source_recipe(value.get("selection_recipe"), dataset=dataset)
            recipe_digest = _sha256(value.get("selection_recipe_sha256"))
            if _canonical_sha256(recipe) != recipe_digest:
                raise FixtureError()
            if value.get("receipt_kind") != RECEIPT_KIND:
                raise FixtureError()
            preparer_contract = _safe_id(value.get("preparer_contract"))
            preparer_files_value = value.get("preparer_files")
            if not isinstance(preparer_files_value, list):
                raise FixtureError()
            preparer_files = tuple(_relative_file(item) for item in preparer_files_value)
            if (
                preparer_contract != _SOURCE_PREPARER_CONTRACTS[source_id]
                or preparer_files != _SOURCE_PREPARER_FILES[source_id]
            ):
                raise FixtureError()
            decoder_contract = _safe_id(value.get("decoder_contract"))
            if decoder_contract != _SOURCE_DECODER_CONTRACTS[source_id]:
                raise FixtureError()
            required_tags = _string_list(value.get("required_tags"), maximum_items=16)
            if not required_tags:
                raise FixtureError()
            sources.append(
                FixtureSource(
                    source_id=source_id,
                    dataset_id=dataset_id,
                    dataset_contract_sha256=contract_sha,
                    corpus_suite=corpus_suite,
                    expected_cases=CASES_PER_SOURCE,
                    selection_recipe_sha256=recipe_digest,
                    receipt_kind=RECEIPT_KIND,
                    preparer_contract=preparer_contract,
                    preparer_files=preparer_files,
                    decoder_contract=decoder_contract,
                    required_tags=required_tags,
                    selection_recipe=dict(recipe),
                )
            )

        slot_values = root.get("slots")
        if not isinstance(slot_values, list) or len(slot_values) != TOTAL_CASES:
            raise FixtureError()
        slots: list[FixtureSlot] = []
        position = 0
        for source_id in SOURCE_IDS:
            for ordinal in range(CASES_PER_SOURCE):
                slots.append(
                    _validate_slot_shape(
                        slot_values[position],
                        source_id=source_id,
                        ordinal=ordinal,
                    )
                )
                position += 1
        prepared_slots = tuple(slots)
        _validate_committed_slots(prepared_slots)
        return FixtureLock(
            path=snapshot.path,
            raw_sha256=hashlib.sha256(snapshot.data).hexdigest(),
            recipe_sha256=recipe_sha256,
            selection_seed=selection_seed,
            sources=tuple(sources),
            slots=prepared_slots,
        )
    except FixtureError:
        raise
    except Exception:
        raise FixtureError() from None


def _private_file(path: Path) -> None:
    try:
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
            or path.resolve(strict=True) != path
        ):
            raise FixtureError()
    except (OSError, RuntimeError, ValueError, OverflowError):
        raise FixtureError() from None


def _directory_identity(
    value: os.stat_result,
) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        stat.S_IMODE(value.st_mode),
        value.st_uid,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _repo_file_sha256(relative: str) -> str:
    try:
        root = Path(__file__).resolve().parents[1]
        path = (root / _relative_file(relative)).resolve(strict=True)
        if root not in path.parents:
            raise FixtureError()
        snapshot = read_regular_bounded(path, maximum_bytes=4 * 1024 * 1024)
        return hashlib.sha256(snapshot.data).hexdigest()
    except Exception:
        raise FixtureError() from None


def _validate_artifacts(value: object, *, dataset: Dataset) -> None:
    if not isinstance(value, list) or len(value) != len(dataset.artifacts):
        raise FixtureError()
    for receipt, artifact in zip(value, dataset.artifacts, strict=True):
        if not isinstance(receipt, dict) or set(receipt) != _ARTIFACT_FIELDS:
            raise FixtureError()
        if receipt.get("artifact_id") != artifact.artifact_id:
            raise FixtureError()
        local_sha = _sha256(receipt.get("local_sha256"))
        size_bytes = _integer(
            receipt.get("size_bytes"), minimum=1, maximum=128 * 1024 * 1024 * 1024
        )
        if artifact.size_bytes is not None and size_bytes != artifact.size_bytes:
            raise FixtureError()
        if artifact.checksum is None:
            if (
                receipt.get("upstream_algorithm") != "sha256"
                or receipt.get("upstream_digest") != local_sha
            ):
                raise FixtureError()
        elif (
            receipt.get("upstream_algorithm") != artifact.checksum.algorithm.value
            or receipt.get("upstream_digest") != artifact.checksum.digest
        ):
            raise FixtureError()
        if (
            artifact.checksum is not None
            and artifact.checksum.algorithm.value == "sha256"
            and local_sha != artifact.checksum.digest
        ):
            raise FixtureError()


def _validate_preparer(
    value: object,
    *,
    source: FixtureSource,
) -> Mapping[str, str]:
    if not isinstance(value, dict) or set(value) != _PREPARER_FIELDS:
        raise FixtureError()
    if value.get("contract") != source.preparer_contract:
        raise FixtureError()
    files = value.get("files")
    if not isinstance(files, dict) or tuple(files) != source.preparer_files:
        raise FixtureError()
    for relative in source.preparer_files:
        if _sha256(files.get(relative)) != _repo_file_sha256(relative):
            raise FixtureError()
    return files


def decoder_closure_sha256(
    contract: str,
    preparer_files: Mapping[str, str],
) -> str:
    """Bind a decoder contract to the exact repository preparation closure."""

    _safe_id(contract)
    if not isinstance(preparer_files, Mapping) or not preparer_files:
        raise FixtureError()
    files: dict[str, str] = {}
    for relative, digest in preparer_files.items():
        files[_relative_file(relative)] = _sha256(digest)
    return _canonical_sha256({"contract": contract, "preparer_files": files})


def _validate_decoder(
    value: object,
    *,
    source: FixtureSource,
    preparer_files: Mapping[str, str],
) -> None:
    if not isinstance(value, dict) or set(value) != _DECODER_FIELDS:
        raise FixtureError()
    contract = _safe_id(value.get("contract"))
    if contract != source.decoder_contract:
        raise FixtureError()
    if value.get("closure_sha256") != decoder_closure_sha256(
        contract,
        preparer_files,
    ):
        raise FixtureError()
    _integer(value.get("files"), minimum=1, maximum=_MAX_DECODER_FILES)
    _integer(value.get("bytes"), minimum=1, maximum=_MAX_DECODER_BYTES)
    if value.get("production_evidence") is not True:
        raise FixtureError()


def _case_attributes(source_id: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise FixtureError()
    expected = {
        "harper-valley": {
            "task_type_index",
            "task_type_sha256",
            "channel",
            "pii_scan_passed",
        },
        "edacc-test": {
            "stratum",
            "accent_sha256",
            "actual_overlap",
            "reference_source",
            "scoreability",
        },
        "ami-es2004a-close-far": {
            "window_index",
            "window_sha256",
            "window_kind",
            "channel",
            "start_sample",
            "end_sample",
            "actual_overlap",
            "scoreability",
        },
        "cvss4-en": {"duration_bucket", "duration_ms", "prompt_sha256"},
    }[source_id]
    if set(value) != expected:
        raise FixtureError()
    return value


def _validate_case_slot_attributes(
    source_id: str,
    *,
    slot: FixtureSlot,
    attributes: Mapping[str, object],
) -> None:
    if source_id == "harper-valley":
        if (
            attributes.get("task_type_index") != slot.ordinal // 3
            or attributes.get("channel") != slot.channel
        ):
            raise FixtureError()
        return
    if source_id == "edacc-test":
        if attributes.get("stratum") != slot.stratum:
            raise FixtureError()
        return
    if source_id == "ami-es2004a-close-far":
        if (
            attributes.get("window_index") != slot.ordinal // 2
            or attributes.get("window_kind") != slot.stratum
            or attributes.get("channel") != slot.channel
        ):
            raise FixtureError()
        return
    if source_id == "cvss4-en":
        if attributes.get("duration_bucket") != slot.stratum:
            raise FixtureError()
        return
    raise FixtureError()


def _validate_source_invariants(
    source_id: str,
    cases: Sequence[Mapping[str, object]],
) -> None:
    speakers = [item.get("speaker_sha256") for item in cases]
    attributes = [item["attributes"] for item in cases]
    if source_id == "harper-valley":
        if any(item is None for item in speakers) or len(set(speakers)) != 24:
            raise FixtureError()
        task_hashes: dict[int, str] = {}
        counts: Counter[int] = Counter()
        for case, value in zip(cases, attributes, strict=True):
            index = _integer(value.get("task_type_index"), maximum=7)
            digest = _sha256(value.get("task_type_sha256"))
            if value.get("channel") != "caller" or value.get("pii_scan_passed") is not True:
                raise FixtureError()
            if index in task_hashes and task_hashes[index] != digest:
                raise FixtureError()
            task_hashes[index] = digest
            counts[index] += 1
        if counts != {index: 3 for index in range(8)} or len(set(task_hashes.values())) != 8:
            raise FixtureError()
        return

    if source_id == "edacc-test":
        if any(item is None for item in speakers) or len(set(speakers)) != 24:
            raise FixtureError()
        counts: Counter[str] = Counter()
        accents: set[str] = set()
        for case, value in zip(cases, attributes, strict=True):
            stratum = value.get("stratum")
            if stratum not in {"clean", "disfluent", "overlap_adjacent"}:
                raise FixtureError()
            counts[str(stratum)] += 1
            accents.add(_sha256(value.get("accent_sha256")))
            if (
                value.get("actual_overlap") is not False
                or value.get("reference_source") != "official_filtered_stm"
                or value.get("scoreability") != "hard_wer"
            ):
                raise FixtureError()
        if counts != {"clean": 8, "disfluent": 8, "overlap_adjacent": 8} or len(accents) < 8:
            raise FixtureError()
        return

    if source_id == "ami-es2004a-close-far":
        pairs: dict[int, list[tuple[Mapping[str, object], Mapping[str, object]]]] = {}
        for case, value in zip(cases, attributes, strict=True):
            index = _integer(value.get("window_index"), maximum=11)
            _sha256(value.get("window_sha256"))
            if value.get("window_kind") not in {"isolated", "turn_transition"}:
                raise FixtureError()
            if value.get("channel") not in {"close", "far"}:
                raise FixtureError()
            start = _integer(value.get("start_sample"), maximum=2**47)
            end = _integer(value.get("end_sample"), minimum=1, maximum=2**47)
            if end <= start or value.get("actual_overlap") is not False or value.get("scoreability") != "hard_wer":
                raise FixtureError()
            if case.get("pcm_samples") != end - start:
                raise FixtureError()
            pairs.setdefault(index, []).append((case, value))
        if set(pairs) != set(range(12)):
            raise FixtureError()
        kinds: Counter[str] = Counter()
        for rows in pairs.values():
            if len(rows) != 2 or {row[1]["channel"] for row in rows} != {"close", "far"}:
                raise FixtureError()
            left_case, left = rows[0]
            right_case, right = rows[1]
            bound_fields = (
                "window_index",
                "window_sha256",
                "window_kind",
                "start_sample",
                "end_sample",
            )
            if any(left[field] != right[field] for field in bound_fields):
                raise FixtureError()
            if left_case["reference_sha256"] != right_case["reference_sha256"]:
                raise FixtureError()
            left_speaker = left_case.get("speaker_sha256")
            right_speaker = right_case.get("speaker_sha256")
            if left["window_kind"] == "isolated":
                if (
                    left_speaker is None
                    or right_speaker is None
                    or left_speaker != right_speaker
                ):
                    raise FixtureError()
            elif left_speaker is not None or right_speaker is not None:
                raise FixtureError()
            kinds[str(left["window_kind"])] += 1
        if kinds != {"isolated": 8, "turn_transition": 4}:
            raise FixtureError()
        return

    if source_id == "cvss4-en":
        if any(item is None for item in speakers) or len(set(speakers)) != 24:
            raise FixtureError()
        counts: Counter[str] = Counter()
        prompts: set[str] = set()
        for case, value in zip(cases, attributes, strict=True):
            bucket = value.get("duration_bucket")
            if bucket not in _DURATION_BOUNDS:
                raise FixtureError()
            duration = _integer(value.get("duration_ms"), minimum=1, maximum=25_000)
            lower, upper, inclusive = _DURATION_BOUNDS[str(bucket)]
            if duration < lower or duration > upper or (duration == upper and not inclusive):
                raise FixtureError()
            expected_samples = duration * 16
            if abs(int(case.get("pcm_samples", -1)) - expected_samples) > 8_000:
                raise FixtureError()
            counts[str(bucket)] += 1
            prompts.add(_sha256(value.get("prompt_sha256")))
        if counts != {bucket: 6 for bucket in _DURATION_BOUNDS} or len(prompts) != 24:
            raise FixtureError()
        return
    raise FixtureError()


def _receipt_source_id(value: object) -> str:
    if not isinstance(value, dict) or set(value) != _RECEIPT_FIELDS:
        raise FixtureError()
    return _safe_id(value.get("source_id"))


def validate_private_source(
    corpus_path: Path | str,
    *,
    lock: FixtureLock,
    expected_source_id: str | None = None,
) -> FixtureSourceBinding:
    """Validate one private source corpus and its fixed sibling receipt."""

    try:
        selected = Path(os.path.abspath(Path(corpus_path).expanduser()))
        if selected.name != "corpus.json":
            raise FixtureError()
        parent = selected.parent
        receipt_path = parent / "preparation-receipt.json"
        with opened_directory_nofollow(parent, require_private=True) as (
            bound_parent,
            directory_fd,
        ):
            directory_identity = _directory_identity(os.fstat(directory_fd))
            if bound_parent != parent:
                raise FixtureError()
            _private_file(selected)
            _private_file(receipt_path)
            receipt_snapshot = read_regular_bounded(
                receipt_path,
                maximum_bytes=_MAX_RECEIPT_BYTES,
            )
            root = _strict_json(receipt_snapshot.data)
            source_id = _receipt_source_id(root)
            _assert_path_free(root)
            if expected_source_id is not None and source_id != expected_source_id:
                raise FixtureError()
            source = lock.source_by_id(source_id)
            dataset = dataset_by_id(source.dataset_id)
            if (
                root.get("schema_version") != 1
                or root.get("kind") != RECEIPT_KIND
                or root.get("fixture_id") != FIXTURE_ID
                or root.get("lock_recipe_sha256") != lock.recipe_sha256
                or root.get("source_recipe_sha256") != source.selection_recipe_sha256
                or root.get("dataset_contract_sha256") != source.dataset_contract_sha256
                or root.get("accepted_terms") != sorted(dataset.license.acceptance_ids)
                or root.get("privacy") != _PRIVATE_RECEIPT_PRIVACY
                or root.get("evidence_scope") != _RECEIPT_EVIDENCE_SCOPE
            ):
                raise FixtureError()
            _validate_artifacts(root.get("artifacts"), dataset=dataset)
            metadata_sha256 = _sha256(root.get("metadata_sha256"))
            preparer_files = _validate_preparer(root.get("preparer"), source=source)
            _validate_decoder(
                root.get("decoder"),
                source=source,
                preparer_files=preparer_files,
            )

            case_values = root.get("cases")
            if not isinstance(case_values, list) or len(case_values) != CASES_PER_SOURCE:
                raise FixtureError()
            selection = root.get("selection")
            if not isinstance(selection, dict) or set(selection) != _SELECTION_FIELDS:
                raise FixtureError()
            if (
                selection.get("policy_sha256")
                != selection_policy_sha256(dataset.selection)
                or _integer(selection.get("eligible_rows"), minimum=CASES_PER_SOURCE)
                < CASES_PER_SOURCE
                or _sha256(selection.get("eligible_set_sha256")) is None
                or _sha256(selection.get("selected_rows_sha256")) is None
                or selection.get("selected_cases") != CASES_PER_SOURCE
                or selection.get("selected_case_set_sha256")
                != _canonical_sha256(case_values)
            ):
                raise FixtureError()
            parent_receipt = selection.get("parent_receipt_sha256")
            parent_corpus = selection.get("parent_corpus_manifest_sha256")
            if source_id == "cvss4-en":
                _sha256(parent_receipt)
                _sha256(parent_corpus)
            elif parent_receipt is not None or parent_corpus is not None:
                raise FixtureError()

            corpus = load_corpus(selected)
            if (
                corpus.schema_version != 2
                or len(corpus.cases) != CASES_PER_SOURCE
                or corpus.audio_bytes > MAX_CORPUS_BYTES
                or corpus.provenance is None
            ):
                raise FixtureError()
            for corpus_case in corpus.cases:
                _private_file(corpus_case.source_path)
            receipt_sha = hashlib.sha256(receipt_snapshot.data).hexdigest()
            expected_provenance = {
                "kind": "public-voice-v1",
                "suite": source.corpus_suite,
                "manifest_sha256": source.selection_recipe_sha256,
                "metadata_sha256": metadata_sha256,
                "source_set_sha256": receipt_sha,
            }
            if corpus.provenance.as_dict() != expected_provenance:
                raise FixtureError()

            slots = lock.slots_for(source_id)
            prepared_cases: list[Mapping[str, object]] = []
            identities: set[str] = set()
            total_samples = 0
            for receipt_case, corpus_case, slot in zip(
                case_values, corpus.cases, slots, strict=True
            ):
                if not isinstance(receipt_case, dict) or set(receipt_case) != _RECEIPT_CASE_FIELDS:
                    raise FixtureError()
                if receipt_case.get("slot_id") != slot.slot_id:
                    raise FixtureError()
                case_id = _safe_id(receipt_case.get("case_id"))
                identity = _sha256(receipt_case.get("identity_sha256"))
                if identity in identities:
                    raise FixtureError()
                identities.add(identity)
                speaker = receipt_case.get("speaker_sha256")
                if speaker is not None:
                    _sha256(speaker)
                reference_sha = _sha256(receipt_case.get("reference_sha256"))
                _sha256(receipt_case.get("source_audio_sha256"))
                _integer(
                    receipt_case.get("source_audio_bytes"),
                    minimum=1,
                    maximum=_MAX_SOURCE_AUDIO_BYTES,
                )
                pcm_sha = _sha256(receipt_case.get("pcm_sha256"))
                pcm_samples = _integer(
                    receipt_case.get("pcm_samples"),
                    minimum=1,
                    maximum=_MAX_PCM_SAMPLES,
                )
                tags = _string_list(receipt_case.get("tags"), maximum_items=32)
                if not set(source.required_tags) <= set(tags):
                    raise FixtureError()
                attributes = _case_attributes(source_id, receipt_case.get("attributes"))
                _validate_case_slot_attributes(
                    source_id,
                    slot=slot,
                    attributes=attributes,
                )
                if (
                    case_id != corpus_case.case_id
                    or pcm_sha != corpus_case.sha256
                    or pcm_samples != corpus_case.samples
                    or reference_sha
                    != hashlib.sha256(corpus_case.expected_text.encode("utf-8")).hexdigest()
                    or tags != corpus_case.tags
                    or corpus_case.assertion != "transcript"
                    or corpus_case.commands
                ):
                    raise FixtureError()
                prepared_cases.append({**receipt_case, "attributes": attributes})
                total_samples += pcm_samples

            _validate_source_invariants(source_id, prepared_cases)
            totals = root.get("totals")
            if totals != {
                "cases": CASES_PER_SOURCE,
                "pcm_samples": total_samples,
                "pcm_bytes": total_samples * 4,
            } or corpus.audio_bytes != total_samples * 4:
                raise FixtureError()
            if (
                _directory_identity(os.fstat(directory_fd)) != directory_identity
                or _directory_identity(parent.lstat()) != directory_identity
                or parent.resolve(strict=True) != parent
            ):
                raise FixtureError()
            verify_corpus_snapshot(corpus)
            return FixtureSourceBinding(
                source_id=source_id,
                corpus_path=corpus.path,
                receipt_path=receipt_snapshot.path,
                receipt_sha256=receipt_sha,
                corpus_manifest_sha256=corpus.digest,
                cases=CASES_PER_SOURCE,
                pcm_bytes=corpus.audio_bytes,
            )
    except FixtureError:
        raise
    except Exception:
        raise FixtureError() from None


def validate_fixture_set(
    corpus_paths: Iterable[Path | str],
    *,
    lock_path: Path | str = DEFAULT_LOCK,
) -> FixtureSet:
    """Validate exactly four source corpora and return them in lock order."""

    try:
        materialized = tuple(islice(iter(corpus_paths), len(SOURCE_IDS) + 1))
        if len(materialized) != len(SOURCE_IDS):
            raise FixtureError()
        lock = load_fixture_lock(lock_path)
        bindings = tuple(validate_private_source(path, lock=lock) for path in materialized)
        by_source = {item.source_id: item for item in bindings}
        if len(by_source) != len(SOURCE_IDS) or set(by_source) != set(SOURCE_IDS):
            raise FixtureError()
        ordered = tuple(by_source[source_id] for source_id in SOURCE_IDS)
        digest_rows = [
            {
                "source_id": item.source_id,
                "receipt_sha256": item.receipt_sha256,
                "corpus_manifest_sha256": item.corpus_manifest_sha256,
            }
            for item in ordered
        ]
        return FixtureSet(
            lock=lock,
            sources=ordered,
            fixture_set_sha256=_canonical_sha256(digest_rows),
            cases=sum(item.cases for item in ordered),
            pcm_bytes=sum(item.pcm_bytes for item in ordered),
        )
    except FixtureError:
        raise
    except Exception:
        raise FixtureError() from None


def verify_fixture_set_snapshot(value: FixtureSet) -> None:
    """Re-open every bound file and reject any post-validation change."""

    if type(value) is not FixtureSet:
        raise FixtureError()
    try:
        current_lock = load_fixture_lock(value.lock.path)
        if (
            current_lock.raw_sha256 != value.lock.raw_sha256
            or current_lock.recipe_sha256 != value.lock.recipe_sha256
        ):
            raise FixtureError()
        current = tuple(
            validate_private_source(
                item.corpus_path,
                lock=current_lock,
                expected_source_id=item.source_id,
            )
            for item in value.sources
        )
        for before, after in zip(value.sources, current, strict=True):
            if (
                before.receipt_sha256 != after.receipt_sha256
                or before.corpus_manifest_sha256 != after.corpus_manifest_sha256
                or before.cases != after.cases
                or before.pcm_bytes != after.pcm_bytes
            ):
                raise FixtureError()
        rows = [
            {
                "source_id": item.source_id,
                "receipt_sha256": item.receipt_sha256,
                "corpus_manifest_sha256": item.corpus_manifest_sha256,
            }
            for item in current
        ]
        if _canonical_sha256(rows) != value.fixture_set_sha256:
            raise FixtureError()
    except FixtureError:
        raise
    except Exception:
        raise FixtureError() from None


def _validate_suite_binding(
    report: object,
    *,
    fixture_set: FixtureSet,
) -> Mapping[str, object]:
    if not isinstance(report, Mapping) or set(report) != {
        "ok",
        "complete",
        "controller",
        "config",
        "runs",
    }:
        raise FixtureError()
    controller = report.get("controller")
    runs = report.get("runs")
    if (
        type(report.get("ok")) is not bool
        or report.get("complete") is not True
        or not isinstance(controller, Mapping)
        or not isinstance(runs, list)
        or controller.get("state") != "complete"
        or controller.get("planned_corpora") != len(fixture_set.sources)
        or controller.get("corpora") != len(fixture_set.sources)
        or controller.get("corpus_set_sha256")
        != _canonical_sha256(
            [item.corpus_manifest_sha256 for item in fixture_set.sources]
        )
    ):
        raise FixtureError()
    workers = _integer(controller.get("planned_worker_manifests"), minimum=1, maximum=8)
    if (
        controller.get("worker_manifests") != workers
        or controller.get("planned_runs") != workers * len(fixture_set.sources)
        or controller.get("runs") != len(runs)
        or len(runs) != workers * len(fixture_set.sources)
    ):
        raise FixtureError()
    seen: set[tuple[int, int]] = set()
    for item in runs:
        if not isinstance(item, Mapping) or set(item) != {"binding", "report"}:
            raise FixtureError()
        binding = item.get("binding")
        if not isinstance(binding, Mapping):
            raise FixtureError()
        worker_index = _integer(binding.get("worker_index"), maximum=workers - 1)
        corpus_index = _integer(
            binding.get("corpus_index"), maximum=len(fixture_set.sources) - 1
        )
        if (
            (worker_index, corpus_index) in seen
            or binding.get("corpus_manifest_sha256")
            != fixture_set.sources[corpus_index].corpus_manifest_sha256
        ):
            raise FixtureError()
        seen.add((worker_index, corpus_index))
    if seen != {
        (worker, corpus)
        for worker in range(workers)
        for corpus in range(len(fixture_set.sources))
    }:
        raise FixtureError()
    _assert_path_free(report)
    return report


def run_validated_suite(
    worker_manifest_paths: Iterable[Path | str],
    corpus_paths: Iterable[Path | str],
    *,
    scratch_parent: Path | str,
    lock_path: Path | str = DEFAULT_LOCK,
    repeats: int = 3,
    chunk_samples: int | None = None,
    pace: str | None = None,
    partial_interval_ms: int | None = None,
    tail_padding_samples: int | None = None,
    stratum_tags: Sequence[str] = (),
    corpus_stratum_tags: Mapping[Path | str, Sequence[str]] | None = None,
    checkpoint_path: Path | str | None = None,
) -> dict[str, object]:
    """Validate, run the exact four corpora sequentially, then revalidate."""

    try:
        from tools import streaming_stt_suite

        fixture_set = validate_fixture_set(corpus_paths, lock_path=lock_path)
        suite_report: object | None = None
        suite_failed = False
        try:
            suite_report = streaming_stt_suite.run_suite(
                worker_manifest_paths,
                fixture_set.corpus_paths,
                scratch_parent=scratch_parent,
                repeats=repeats,
                chunk_samples=chunk_samples,
                pace=pace,
                partial_interval_ms=partial_interval_ms,
                tail_padding_samples=tail_padding_samples,
                stratum_tags=stratum_tags,
                corpus_stratum_tags=corpus_stratum_tags,
                checkpoint_path=checkpoint_path,
            )
        except Exception:
            suite_failed = True
        verify_fixture_set_snapshot(fixture_set)
        if suite_failed or suite_report is None:
            raise FixtureError()
        bound_report = _validate_suite_binding(
            suite_report,
            fixture_set=fixture_set,
        )
        result = {
            "ok": bound_report["ok"],
            "fixture": {
                "fixture_id": FIXTURE_ID,
                "lock_recipe_sha256": fixture_set.lock.recipe_sha256,
                "fixture_set_sha256": fixture_set.fixture_set_sha256,
                "sources": len(fixture_set.sources),
                "cases": fixture_set.cases,
                "pcm_bytes": fixture_set.pcm_bytes,
                "validation": "before_and_after_suite",
            },
            "suite": bound_report,
        }
        _assert_path_free(result)
        return result
    except FixtureError:
        raise
    except Exception:
        raise FixtureError() from None


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise FixtureError()


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description="Validate four private public-conversation/STT corpora.",
    )
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--corpus", action="append", type=Path, default=[])
    parser.add_argument("--worker-manifest", action="append", type=Path, default=[])
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--chunk-samples", type=int)
    parser.add_argument("--pace", choices=("burst", "realtime"))
    parser.add_argument("--partial-interval-ms", type=int)
    parser.add_argument("--tail-padding-samples", type=int)
    parser.add_argument("--stratum-tag", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        if args.worker_manifest:
            if args.scratch_root is None or args.output is None:
                raise FixtureError()
            from tools import streaming_stt_suite

            suite_result = run_validated_suite(
                tuple(args.worker_manifest),
                tuple(args.corpus),
                scratch_parent=args.scratch_root,
                lock_path=args.lock,
                repeats=args.repeats,
                chunk_samples=args.chunk_samples,
                pace=args.pace,
                partial_interval_ms=args.partial_interval_ms,
                tail_padding_samples=args.tail_padding_samples,
                stratum_tags=tuple(args.stratum_tag),
                checkpoint_path=args.output,
            )
            streaming_stt_suite.write_report(args.output, suite_result)
            fixture_summary = suite_result["fixture"]
            if not isinstance(fixture_summary, Mapping):
                raise FixtureError()
            payload = {
                "ok": suite_result["ok"],
                "fixture_id": FIXTURE_ID,
                "recipe_sha256": fixture_summary["lock_recipe_sha256"],
                "fixture_set_sha256": fixture_summary["fixture_set_sha256"],
                "sources": fixture_summary["sources"],
                "cases": fixture_summary["cases"],
                "pcm_bytes": fixture_summary["pcm_bytes"],
                "suite_report_written": True,
            }
            print(_canonical_json_bytes(payload).decode("ascii"))
            return 0 if suite_result["ok"] is True else 1
        if (
            args.scratch_root is not None
            or args.output is not None
            or args.repeats != 3
            or args.chunk_samples is not None
            or args.pace is not None
            or args.partial_interval_ms is not None
            or args.tail_padding_samples is not None
            or args.stratum_tag
        ):
            raise FixtureError()
        result = validate_fixture_set(tuple(args.corpus), lock_path=args.lock)
        verify_fixture_set_snapshot(result)
        payload = {
            "ok": True,
            "fixture_id": FIXTURE_ID,
            "recipe_sha256": result.lock.recipe_sha256,
            "fixture_set_sha256": result.fixture_set_sha256,
            "sources": len(result.sources),
            "cases": result.cases,
            "pcm_bytes": result.pcm_bytes,
        }
        print(_canonical_json_bytes(payload).decode("ascii"))
        return 0
    except Exception:
        print('{"error":"fixture validation failed","ok":false}', file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CASES_PER_SOURCE",
    "DATASET_IDS",
    "DEFAULT_LOCK",
    "FIXTURE_ID",
    "FixtureError",
    "FixtureLock",
    "FixtureSet",
    "FixtureSlot",
    "FixtureSource",
    "FixtureSourceBinding",
    "LOCK_KIND",
    "RECEIPT_KIND",
    "SOURCE_IDS",
    "TOTAL_CASES",
    "dataset_contract_sha256",
    "decoder_closure_sha256",
    "load_fixture_lock",
    "main",
    "run_validated_suite",
    "validate_fixture_set",
    "validate_private_source",
    "verify_fixture_set_snapshot",
]
