"""Project the verified 32-case Common Voice SPS v4 slice into fixture slots.

The input is the private corpus produced by
``prepare_common_voice_spontaneous_v4`` plus the same acquired archive.  The
archive is re-opened through that preparer's bounded verifier, metadata and the
32-row selection are recomputed, every selected MP3 is rehashed, and the parent
receipt/corpus are cross-bound before 24 PCM cases are copied.  No transcript,
source identity, archive path, or PCM is written to the committed lock.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import stat
from typing import Callable, Mapping, Sequence

from tools import prepare_common_voice_spontaneous_v4 as parent
from tools import public_conversation_fixture as fixture
from tools.public_voice_eval_matrix import dataset_by_id, selection_policy_sha256
from tools.streaming_stt.bounded_io import (
    opened_directory_nofollow,
    read_regular_bounded,
)
from tools.streaming_stt.corpus import (
    CorpusProvenance,
    LoadedCorpus,
    load_corpus,
    verify_corpus_snapshot,
)
from tools.streaming_stt.corpus_writer import CorpusWriteCase, publish_private_corpus


SOURCE_ID = "cvss4-en"
PREPARER_CONTRACT = "public-conversation-cv-projection-v1"
DECODER_CONTRACT = "verified-cvss4-parent-f32le-copy-v1"
SELECTION_ALGORITHM = "cvss4_32_to_24_projection_v1"
SOURCE_RECIPE_ELIGIBILITY = (
    "existing_verified_32",
    "six_per_duration_bucket",
    "distinct_speakers",
    "unique_prompt_digests",
)
SOURCE_RECIPE_STRATA = (
    ("d0_2_6s", 6),
    ("d1_6_10s", 6),
    ("d2_10_15s", 6),
    ("d3_15_25s", 6),
)
_MAX_PARENT_RECEIPT_BYTES = 256 * 1024
_ROOT_FIELDS = {
    "schema_version",
    "kind",
    "dataset",
    "accepted_terms",
    "usage_constraints",
    "mdc_content_receipt",
    "metadata",
    "selection",
    "preparer",
    "decoder",
    "cases",
    "total_pcm_bytes",
    "evidence_scope",
}
_PARENT_CASE_FIELDS = {
    "case_id",
    "identity_sha256",
    "speaker_sha256",
    "prompt_sha256",
    "duration_ms",
    "duration_bucket",
    "quality_tags",
    "transcription_sha256",
    "source_mp3_sha256",
    "source_mp3_bytes",
    "pcm_sha256",
    "pcm_samples",
}
_PARENT_EVIDENCE_SCOPE = {
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
}
_USAGE_CONSTRAINTS = [
    "evaluation_only",
    "private_cache_only",
    "no_speaker_reidentification",
    "no_rehosting",
]


class CvProjectionError(RuntimeError):
    """A detail-free parent, projection, or publication failure."""


@dataclass(frozen=True, slots=True)
class TestProjectionInjection:
    """Permit an unmistakably non-production parent decoder in unit tests."""

    fixture_id: str
    decoder: Callable[[bytes], parent.DecodedAudio] = field(repr=False)


@dataclass(frozen=True, slots=True)
class _BoundParentCase:
    row: parent.SpsRow = field(repr=False)
    bucket: str
    corpus_case: object = field(repr=False)
    receipt_case: Mapping[str, object] = field(repr=False)


@dataclass(frozen=True, slots=True)
class _VerifiedParentProjection:
    corpus: LoadedCorpus = field(repr=False)
    receipt_sha256: str
    metadata_sha256: str
    eligible: tuple[_BoundParentCase, ...] = field(repr=False)
    selected: tuple[_BoundParentCase, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class PreparedCvProjection:
    corpus: LoadedCorpus
    receipt_sha256: str
    fixture_recipe_sha256: str
    parent_receipt_sha256: str
    parent_corpus_manifest_sha256: str
    production_evidence: bool


def _canonical_bytes(value: object) -> bytes:
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
    except (TypeError, UnicodeError, ValueError, OverflowError):
        raise CvProjectionError() from None


def _sha256(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CvProjectionError()
    return value


def _private_parent_file(path: Path) -> None:
    try:
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
            or path.resolve(strict=True) != path
        ):
            raise CvProjectionError()
    except (OSError, RuntimeError, ValueError, OverflowError):
        raise CvProjectionError() from None


def _parent_dataset_payload() -> dict[str, object]:
    return {
        "dataset_id": parent.DATASET_ID,
        "mdc_dataset_id": parent.MDC_DATASET_ID,
        "release_id": parent.RELEASE_ID,
        "release_date": parent.RELEASE_DATE,
        "mdc_published_date": parent.MDC_PUBLISHED_DATE,
        "locale": parent.LOCALE,
        "archive_filename": parent.ARCHIVE_FILENAME,
        "archive_size_bytes": parent.ARCHIVE_SIZE_BYTES,
        "archive_sha256": parent.ARCHIVE_SHA256,
        "stats_git_commit": parent.STATS_GIT_COMMIT,
        "stats_git_blob": parent.STATS_GIT_BLOB,
        "stats_sha256": parent.STATS_SHA256,
        "schema_registry_commit": parent.SCHEMA_REGISTRY_COMMIT,
    }


def _parent_metadata_payload(
    metadata: parent.ParsedMetadata,
    *,
    metadata_sha256: str,
) -> dict[str, object]:
    forbidden = metadata.split_speakers["train"] | metadata.split_speakers["dev"]
    return {
        "main_tsv_sha256": metadata_sha256,
        "rows": len(metadata.rows),
        "duration_ms": sum(row.duration_ms for row in metadata.rows),
        "split_counts": dict(metadata.split_counts),
        "split_duration_ms": dict(metadata.split_duration_ms),
        "train_speakers_sha256": parent._digest_string_set(
            "speaker", metadata.split_speakers["train"]
        ),
        "dev_speakers_sha256": parent._digest_string_set(
            "speaker", metadata.split_speakers["dev"]
        ),
        "test_speakers_sha256": parent._digest_string_set(
            "speaker", metadata.split_speakers["test"]
        ),
        "forbidden_speakers_sha256": parent._digest_string_set("speaker", forbidden),
        "reported_tsv_content_read": False,
    }


def _expected_parent_decoder(
    injection: TestProjectionInjection | None,
) -> tuple[
    Mapping[str, object],
    bool,
    Callable[[bytes], parent.DecodedAudio],
]:
    if injection is None:
        return parent._pyav_decoder_provenance(), True, parent._decode_mp3_pyav
    if (
        type(injection) is not TestProjectionInjection
        or not isinstance(injection.fixture_id, str)
        or parent._SAFE_TAG_RE.fullmatch(injection.fixture_id) is None
        or not callable(injection.decoder)
    ):
        raise CvProjectionError()
    return (
        {
            "contract": "test-injected-f32le-v1",
            "fixture_id": injection.fixture_id,
            "production_evidence": False,
        },
        False,
        injection.decoder,
    )


def _validate_parent_root(
    root: object,
    *,
    metadata: parent.ParsedMetadata,
    metadata_sha256: str,
    selection: Mapping[str, object],
    api_receipt: parent.MdcApiReceipt,
    expected_decoder: Mapping[str, object],
) -> Sequence[Mapping[str, object]]:
    if not isinstance(root, dict) or set(root) != _ROOT_FIELDS:
        raise CvProjectionError()
    expected_preparer = parent._preparer_code_provenance()
    if (
        root.get("schema_version") != 1
        or root.get("kind") != "common-voice-spontaneous-v4-preparation"
        or root.get("dataset") != _parent_dataset_payload()
        or root.get("accepted_terms") != sorted(parent.REQUIRED_TERMS)
        or root.get("usage_constraints") != _USAGE_CONSTRAINTS
        or root.get("mdc_content_receipt")
        != {
            "filename": api_receipt.filename,
            "size_bytes": api_receipt.size_bytes,
            "checksum_algorithm": api_receipt.checksum_algorithm,
            "checksum_sha256": api_receipt.checksum_sha256,
        }
        or root.get("metadata")
        != _parent_metadata_payload(metadata, metadata_sha256=metadata_sha256)
        or root.get("selection") != selection
        or root.get("preparer") != expected_preparer
        or root.get("decoder") != expected_decoder
        or root.get("evidence_scope") != _PARENT_EVIDENCE_SCOPE
    ):
        raise CvProjectionError()
    cases = root.get("cases")
    if not isinstance(cases, list) or len(cases) != parent.SELECTION_COUNT:
        raise CvProjectionError()
    if any(not isinstance(item, dict) or set(item) != _PARENT_CASE_FIELDS for item in cases):
        raise CvProjectionError()
    return cases


def _bind_parent_cases(
    *,
    archive: object,
    layout: parent.ArchiveLayout,
    selected: Sequence[tuple[parent.SpsRow, str]],
    parent_corpus: LoadedCorpus,
    receipt_cases: Sequence[Mapping[str, object]],
    decoder: Callable[[bytes], parent.DecodedAudio],
) -> tuple[_BoundParentCase, ...]:
    if (
        parent_corpus.schema_version != 2
        or len(parent_corpus.cases) != parent.SELECTION_COUNT
        or parent_corpus.provenance is None
    ):
        raise CvProjectionError()
    selected_members: list[tuple[object, int]] = []
    try:
        for position, (row, _bucket) in enumerate(selected):
            member = layout.members[f"{parent._AUDIO_PREFIX}{row.audio_file}"]
            selected_members.append((member, position))
    except CvProjectionError:
        raise
    except Exception:
        raise CvProjectionError() from None

    # Process archive members in physical order, but retain only the already
    # bounded parent PCM.  Peak incremental memory is one MP3 plus one decode,
    # rather than all 32 MP3s and a duplicate 32-case PCM corpus.
    result: dict[int, _BoundParentCase] = {}
    for member, position in sorted(
        selected_members,
        key=lambda value: int(value[0].offset_data),
    ):
        row, bucket = selected[position]
        corpus_case = parent_corpus.cases[position]
        receipt_case = receipt_cases[position]
        case_id, identity_sha256 = parent._case_id(position, row)
        try:
            source = parent._read_member(
                archive,
                member,
                maximum_bytes=parent._MAX_AUDIO_MEMBER_BYTES,
            )
            decoded = decoder(source)
            parent._verify_decoded_audio(decoded, duration_ms=row.duration_ms)
        except CvProjectionError:
            raise
        except Exception:
            raise CvProjectionError() from None
        expected = {
            "case_id": case_id,
            "identity_sha256": identity_sha256,
            "speaker_sha256": parent._salted_digest("speaker", row.client_id),
            "prompt_sha256": parent._salted_digest("prompt", row.prompt_id),
            "duration_ms": row.duration_ms,
            "duration_bucket": bucket,
            "quality_tags": list(row.quality_tags),
            "transcription_sha256": hashlib.sha256(
                row.transcription.encode("utf-8")
            ).hexdigest(),
            "source_mp3_sha256": hashlib.sha256(source).hexdigest(),
            "source_mp3_bytes": len(source),
            "pcm_sha256": hashlib.sha256(decoded.raw_f32le).hexdigest(),
            "pcm_samples": decoded.samples,
        }
        if (
            receipt_case != expected
            or corpus_case.case_id != case_id
            or corpus_case.expected_text != row.transcription
            or corpus_case.assertion != "transcript"
            or corpus_case.commands
            or corpus_case.tags
            != ("public-voice", "cvss4-en", "spontaneous", "clean", bucket)
            or corpus_case.audio_bytes != decoded.raw_f32le
            or corpus_case.samples != decoded.samples
            or corpus_case.sha256 != expected["pcm_sha256"]
        ):
            raise CvProjectionError()
        result[position] = _BoundParentCase(
            row=row,
            bucket=bucket,
            corpus_case=corpus_case,
            receipt_case=dict(receipt_case),
        )
        del source, decoded
    if len(result) != parent.SELECTION_COUNT:
        raise CvProjectionError()
    return tuple(result[position] for position in range(parent.SELECTION_COUNT))


def _rank(seed: str, value: _BoundParentCase) -> str:
    return hashlib.sha256(
        seed.encode("ascii")
        + b"\0"
        + value.bucket.encode("ascii")
        + b"\0"
        + str(value.receipt_case["identity_sha256"]).encode("ascii")
    ).hexdigest()


def _project_cases(
    values: Sequence[_BoundParentCase],
    *,
    seed: str,
) -> tuple[_BoundParentCase, ...]:
    bucket_options: list[
        tuple[tuple[tuple[_BoundParentCase, str, str], ...], ...]
    ] = []
    for bucket, _lower, _upper, _inclusive in parent.DURATION_BUCKETS:
        candidates = sorted(
            (item for item in values if item.bucket == bucket),
            key=lambda item: (
                _rank(seed, item),
                str(item.receipt_case["identity_sha256"]),
            ),
        )
        if len(candidates) != parent.SELECTION_COUNT_PER_BUCKET:
            raise CvProjectionError()
        annotated = tuple(
            (
                item,
                _sha256(item.receipt_case.get("speaker_sha256")),
                _sha256(item.receipt_case.get("prompt_sha256")),
            )
            for item in candidates
        )
        options: list[tuple[tuple[_BoundParentCase, str, str], ...]] = []
        for indexes in combinations(range(len(annotated)), 6):
            choice = tuple(annotated[index] for index in indexes)
            speakers = {item[1] for item in choice}
            prompts = {item[2] for item in choice}
            if len(speakers) == 6 and len(prompts) == 6:
                options.append(choice)
        if not options:
            raise CvProjectionError()
        bucket_options.append(tuple(options))

    def search(
        bucket_index: int,
        used_speakers: frozenset[str],
        used_prompts: frozenset[str],
    ) -> tuple[_BoundParentCase, ...] | None:
        if bucket_index == len(bucket_options):
            return ()
        for choice in bucket_options[bucket_index]:
            speakers = frozenset(item[1] for item in choice)
            prompts = frozenset(item[2] for item in choice)
            if used_speakers & speakers or used_prompts & prompts:
                continue
            remainder = search(
                bucket_index + 1,
                used_speakers | speakers,
                used_prompts | prompts,
            )
            if remainder is not None:
                return tuple(item[0] for item in choice) + remainder
        return None

    projected = search(0, frozenset(), frozenset())
    if projected is None:
        raise CvProjectionError()
    used_speakers = {
        _sha256(item.receipt_case.get("speaker_sha256")) for item in projected
    }
    used_prompts = {
        _sha256(item.receipt_case.get("prompt_sha256")) for item in projected
    }
    if (
        len(projected) != fixture.CASES_PER_SOURCE
        or len(used_speakers) != fixture.CASES_PER_SOURCE
        or len(used_prompts) != fixture.CASES_PER_SOURCE
        or Counter(item.bucket for item in projected)
        != {bucket: 6 for bucket, *_rest in parent.DURATION_BUCKETS}
    ):
        raise CvProjectionError()
    return projected


def _preparer_binding(source: fixture.FixtureSource) -> tuple[dict[str, str], int]:
    files = {
        relative: fixture._repo_file_sha256(relative)
        for relative in source.preparer_files
    }
    try:
        total = sum(
            (Path(__file__).resolve().parents[1] / relative).stat().st_size
            for relative in files
        )
    except (OSError, ValueError, OverflowError):
        raise CvProjectionError() from None
    if not files or total <= 0:
        raise CvProjectionError()
    return files, total


def _expected_source_recipe(*, seed: str) -> dict[str, object]:
    dataset = dataset_by_id(parent.DATASET_ID)
    return {
        "algorithm": SELECTION_ALGORITHM,
        "matrix_policy_sha256": selection_policy_sha256(dataset.selection),
        "split": "test",
        "seed": seed,
        "eligibility": list(SOURCE_RECIPE_ELIGIBILITY),
        "strata": [
            {"id": stratum, "cases": cases}
            for stratum, cases in SOURCE_RECIPE_STRATA
        ],
        "speaker_requirement": "exactly_24_distinct",
        "scoreability": "hard_wer_nonoverlap_only",
    }


def _validate_fixture_source_contract(
    source: fixture.FixtureSource,
    *,
    seed: str,
) -> None:
    if (
        source.dataset_id != parent.DATASET_ID
        or source.expected_cases != fixture.CASES_PER_SOURCE
        or source.preparer_contract != PREPARER_CONTRACT
        or source.decoder_contract != DECODER_CONTRACT
        or dict(source.selection_recipe) != _expected_source_recipe(seed=seed)
    ):
        raise CvProjectionError()


def _verify_and_project_parent(
    *,
    archive_path: Path | str,
    parent_corpus_path: Path | str,
    api_receipt: parent.MdcApiReceipt,
    expected_decoder: Mapping[str, object],
    decoder: Callable[[bytes], parent.DecodedAudio],
    seed: str,
) -> _VerifiedParentProjection:
    """Bind one private parent snapshot before any derived publication."""

    selected_parent_path = Path(
        os.path.abspath(Path(parent_corpus_path).expanduser())
    )
    if selected_parent_path.name != "corpus.json":
        raise CvProjectionError()
    parent_directory = selected_parent_path.parent
    parent_receipt_path = parent_directory / "preparation-receipt.json"

    try:
        with opened_directory_nofollow(
            parent_directory,
            require_private=True,
        ) as (stable_directory, directory_fd):
            directory_identity = fixture._directory_identity(os.fstat(directory_fd))
            if (
                stable_directory != parent_directory
                or stat.S_IMODE(os.fstat(directory_fd).st_mode) != 0o700
            ):
                raise CvProjectionError()
            _private_parent_file(selected_parent_path)
            _private_parent_file(parent_receipt_path)
            receipt_snapshot = read_regular_bounded(
                parent_receipt_path,
                maximum_bytes=_MAX_PARENT_RECEIPT_BYTES,
            )
            parent_receipt_sha256 = hashlib.sha256(receipt_snapshot.data).hexdigest()
            parent_root = fixture._strict_json(receipt_snapshot.data)
            parent_corpus = load_corpus(selected_parent_path)
            for corpus_case in parent_corpus.cases:
                if corpus_case.source_path.parent != parent_directory:
                    raise CvProjectionError()
                _private_parent_file(corpus_case.source_path)

            with parent._verified_archive(archive_path) as archive:
                layout = parent._scan_archive(archive)
                main_tsv = parent._read_member(
                    archive,
                    layout.main_tsv,
                    maximum_bytes=parent._MAX_MAIN_TSV_BYTES,
                )
                metadata_sha256 = hashlib.sha256(main_tsv).hexdigest()
                metadata = parent._parse_main_tsv(main_tsv)
                selected, recipe_sha256, _selected_sha256, selection = (
                    parent._select_rows(metadata)
                )
                receipt_cases = _validate_parent_root(
                    parent_root,
                    metadata=metadata,
                    metadata_sha256=metadata_sha256,
                    selection=selection,
                    api_receipt=api_receipt,
                    expected_decoder=expected_decoder,
                )
                if (
                    parent_corpus.provenance is None
                    or parent_corpus.provenance.as_dict()
                    != {
                        "kind": "public-voice-v1",
                        "suite": "cvss4-en-spontaneous",
                        "manifest_sha256": recipe_sha256,
                        "metadata_sha256": metadata_sha256,
                        "source_set_sha256": parent_receipt_sha256,
                    }
                    or parent_root.get("total_pcm_bytes") != parent_corpus.audio_bytes
                ):
                    raise CvProjectionError()
                eligible = _bind_parent_cases(
                    archive=archive,
                    layout=layout,
                    selected=selected,
                    parent_corpus=parent_corpus,
                    receipt_cases=receipt_cases,
                    decoder=decoder,
                )
                projected = _project_cases(eligible, seed=seed)

            # Recheck every path-bound input and the live directory descriptor
            # before returning the in-memory snapshot to publication code.
            verify_corpus_snapshot(parent_corpus)
            _private_parent_file(selected_parent_path)
            _private_parent_file(parent_receipt_path)
            for corpus_case in parent_corpus.cases:
                _private_parent_file(corpus_case.source_path)
            if (
                hashlib.sha256(
                    read_regular_bounded(
                        parent_receipt_path,
                        maximum_bytes=_MAX_PARENT_RECEIPT_BYTES,
                    ).data
                ).hexdigest()
                != parent_receipt_sha256
                or fixture._directory_identity(os.fstat(directory_fd))
                != directory_identity
                or fixture._directory_identity(parent_directory.lstat())
                != directory_identity
                or parent_directory.resolve(strict=True) != parent_directory
            ):
                raise CvProjectionError()
            return _VerifiedParentProjection(
                corpus=parent_corpus,
                receipt_sha256=parent_receipt_sha256,
                metadata_sha256=metadata_sha256,
                eligible=eligible,
                selected=projected,
            )
    except CvProjectionError:
        raise
    except Exception:
        raise CvProjectionError() from None


def prepare_common_voice_conversation_fixture(
    *,
    archive_path: Path | str,
    parent_corpus_path: Path | str,
    output_dir: Path | str,
    accepted_terms: frozenset[str],
    api_receipt: parent.MdcApiReceipt,
    lock_path: Path | str = fixture.DEFAULT_LOCK,
    test_projection_injection: TestProjectionInjection | None = None,
) -> PreparedCvProjection:
    """Recompute the 32-case source evidence and publish the locked 24 cases."""

    try:
        if not parent.REQUIRED_TERMS <= accepted_terms:
            raise CvProjectionError()
        if test_projection_injection is None:
            parent._validate_catalog_contract()
        parent._validate_api_receipt(api_receipt)
        destination = parent._validated_output_path(output_dir)
        lock = fixture.load_fixture_lock(lock_path)
        source_contract = lock.source_by_id(SOURCE_ID)
        _validate_fixture_source_contract(
            source_contract,
            seed=lock.selection_seed,
        )
        expected_decoder, production_evidence, selected_decoder = _expected_parent_decoder(
            test_projection_injection
        )
        verified_parent = _verify_and_project_parent(
            archive_path=archive_path,
            parent_corpus_path=parent_corpus_path,
            api_receipt=api_receipt,
            expected_decoder=expected_decoder,
            decoder=selected_decoder,
            seed=lock.selection_seed,
        )
        parent_corpus = verified_parent.corpus
        parent_receipt_sha256 = verified_parent.receipt_sha256
        metadata_sha256 = verified_parent.metadata_sha256
        bound_parent = verified_parent.eligible
        projected = verified_parent.selected

        slots = lock.slots_for(SOURCE_ID)
        write_cases: list[CorpusWriteCase] = []
        output_receipt_cases: list[dict[str, object]] = []
        total_samples = 0
        for slot, item in zip(slots, projected, strict=True):
            if slot.stratum != item.bucket:
                raise CvProjectionError()
            parent_case = item.corpus_case
            identity = _sha256(item.receipt_case.get("identity_sha256"))
            case_id = f"cvss4-en-{slot.ordinal:02d}-{identity[:12]}"
            tags = tuple(
                dict.fromkeys((*source_contract.required_tags, "clean", item.bucket))
            )
            write_cases.append(
                CorpusWriteCase(
                    case_id=case_id,
                    audio_bytes=parent_case.audio_bytes,
                    reference=parent_case.expected_text,
                    tags=tags,
                )
            )
            output_receipt_cases.append(
                {
                    "slot_id": slot.slot_id,
                    "case_id": case_id,
                    "identity_sha256": identity,
                    "speaker_sha256": _sha256(
                        item.receipt_case.get("speaker_sha256")
                    ),
                    "reference_sha256": _sha256(
                        item.receipt_case.get("transcription_sha256")
                    ),
                    "source_audio_sha256": _sha256(
                        item.receipt_case.get("source_mp3_sha256")
                    ),
                    "source_audio_bytes": item.receipt_case.get("source_mp3_bytes"),
                    "pcm_sha256": parent_case.sha256,
                    "pcm_samples": parent_case.samples,
                    "tags": list(tags),
                    "attributes": {
                        "duration_bucket": item.bucket,
                        "duration_ms": item.row.duration_ms,
                        "prompt_sha256": _sha256(
                            item.receipt_case.get("prompt_sha256")
                        ),
                    },
                }
            )
            total_samples += parent_case.samples

        parent_manifest_sha256 = parent_corpus.digest
        eligible_binding = {
            "parent_receipt_sha256": parent_receipt_sha256,
            "parent_corpus_manifest_sha256": parent_manifest_sha256,
            "cases": [
                {
                    "identity_sha256": item.receipt_case["identity_sha256"],
                    "speaker_sha256": item.receipt_case["speaker_sha256"],
                    "prompt_sha256": item.receipt_case["prompt_sha256"],
                    "reference_sha256": item.receipt_case["transcription_sha256"],
                    "source_audio_sha256": item.receipt_case["source_mp3_sha256"],
                    "pcm_sha256": item.receipt_case["pcm_sha256"],
                    "duration_bucket": item.bucket,
                }
                for item in bound_parent
            ],
        }
        preparer_files, preparer_bytes = _preparer_binding(source_contract)
        dataset = dataset_by_id(source_contract.dataset_id)
        artifact = dataset.artifacts[0]
        local_archive_sha256 = parent.ARCHIVE_SHA256
        receipt = {
            "schema_version": 1,
            "kind": fixture.RECEIPT_KIND,
            "fixture_id": fixture.FIXTURE_ID,
            "lock_recipe_sha256": lock.recipe_sha256,
            "source_id": SOURCE_ID,
            "source_recipe_sha256": source_contract.selection_recipe_sha256,
            "dataset_contract_sha256": source_contract.dataset_contract_sha256,
            "accepted_terms": sorted(dataset.license.acceptance_ids),
            "artifacts": [
                {
                    "artifact_id": artifact.artifact_id,
                    "upstream_algorithm": (
                        artifact.checksum.algorithm.value
                        if artifact.checksum is not None
                        else "sha256"
                    ),
                    "upstream_digest": (
                        artifact.checksum.digest
                        if artifact.checksum is not None
                        else local_archive_sha256
                    ),
                    "size_bytes": parent.ARCHIVE_SIZE_BYTES,
                    "local_sha256": local_archive_sha256,
                }
            ],
            "metadata_sha256": metadata_sha256,
            "selection": {
                "policy_sha256": selection_policy_sha256(dataset.selection),
                "eligible_rows": len(bound_parent),
                "eligible_set_sha256": fixture._canonical_sha256(eligible_binding),
                "selected_rows_sha256": fixture._canonical_sha256(
                    [item.receipt_case["identity_sha256"] for item in projected]
                ),
                "selected_cases": fixture.CASES_PER_SOURCE,
                "selected_case_set_sha256": fixture._canonical_sha256(
                    output_receipt_cases
                ),
                "parent_receipt_sha256": parent_receipt_sha256,
                "parent_corpus_manifest_sha256": parent_manifest_sha256,
            },
            "preparer": {
                "contract": source_contract.preparer_contract,
                "files": preparer_files,
            },
            "decoder": {
                "contract": DECODER_CONTRACT,
                "closure_sha256": fixture.decoder_closure_sha256(
                    DECODER_CONTRACT, preparer_files
                ),
                "files": len(preparer_files),
                "bytes": preparer_bytes,
                "production_evidence": production_evidence,
            },
            "cases": output_receipt_cases,
            "totals": {
                "cases": fixture.CASES_PER_SOURCE,
                "pcm_samples": total_samples,
                "pcm_bytes": total_samples * 4,
            },
            "privacy": dict(fixture._PRIVATE_RECEIPT_PRIVACY),
            "evidence_scope": dict(fixture._RECEIPT_EVIDENCE_SCOPE),
        }
        receipt_payload = _canonical_bytes(receipt)
        receipt_sha256 = hashlib.sha256(receipt_payload).hexdigest()
        provenance = CorpusProvenance(
            kind="public-voice-v1",
            suite=source_contract.corpus_suite,
            manifest_sha256=source_contract.selection_recipe_sha256,
            metadata_sha256=metadata_sha256,
            source_set_sha256=receipt_sha256,
        )
        published = publish_private_corpus(
            cases=tuple(write_cases),
            provenance=provenance,
            output_dir=destination,
            purpose="Common Voice SPS v4 24-case public-conversation fixture",
            sidecars={"preparation-receipt.json": receipt_payload},
        )
        if published.provenance != provenance:
            raise CvProjectionError()
        return PreparedCvProjection(
            corpus=published,
            receipt_sha256=receipt_sha256,
            fixture_recipe_sha256=lock.recipe_sha256,
            parent_receipt_sha256=parent_receipt_sha256,
            parent_corpus_manifest_sha256=parent_manifest_sha256,
            production_evidence=production_evidence,
        )
    except CvProjectionError:
        raise
    except Exception:
        raise CvProjectionError() from None


def _parser() -> object:
    parser = parent._SafeArgumentParser(
        description="Project a verified private CV SPS v4 corpus into 24 fixture cases."
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--parent-corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=fixture.DEFAULT_LOCK)
    parser.add_argument("--accept-term", action="append", default=[])
    parser.add_argument("--mdc-api-filename", required=True)
    parser.add_argument("--mdc-api-size-bytes", type=int, required=True)
    parser.add_argument("--mdc-api-checksum", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        algorithm, separator, digest = args.mdc_api_checksum.partition(":")
        if separator != ":":
            raise CvProjectionError()
        result = prepare_common_voice_conversation_fixture(
            archive_path=args.archive,
            parent_corpus_path=args.parent_corpus,
            output_dir=args.output_dir,
            accepted_terms=frozenset(args.accept_term),
            api_receipt=parent.MdcApiReceipt(
                filename=args.mdc_api_filename,
                size_bytes=args.mdc_api_size_bytes,
                checksum_algorithm=algorithm,
                checksum_sha256=digest,
            ),
            lock_path=args.lock,
        )
        payload = {
            "ok": True,
            "source_id": SOURCE_ID,
            "cases": len(result.corpus.cases),
            "corpus_sha256": result.corpus.digest,
            "receipt_sha256": result.receipt_sha256,
            "fixture_recipe_sha256": result.fixture_recipe_sha256,
            "production_evidence": result.production_evidence,
        }
        code = 0
    except Exception:
        payload = {"ok": False, "error": "cvss4 fixture preparation failed"}
        code = 2
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
