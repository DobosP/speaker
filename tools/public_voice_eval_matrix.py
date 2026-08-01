"""Typed, no-download catalog for public voice-agent evaluation data.

The catalog deliberately separates acoustic and semantic tasks.  It records
where an evaluation source comes from, how it is selected, which metric family
may consume it, and what must be verified before any private acquisition is
usable.  This module never downloads data and never stores credential values.

Raw corpora belong under ``$XDG_CACHE_HOME/speaker/corpora`` (or
``~/.cache/speaker/corpora``), outside the repository.  A caller must explicitly
accept every applicable license/terms identifier and must provide a SHA-256
receipt when the upstream service does not publish a stable file digest.
"""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse


MATRIX_VERSION = 3
SELECTION_SEED = "speaker-public-eval-v1-2026-08-01"
MDC_TOKEN_ENV = "MDC_TOKEN"

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAFE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,95}\Z")
_SAFE_FILENAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}\Z")
_SHA1_RE = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_MD5_RE = re.compile(r"[0-9a-f]{32}\Z")
_MAX_LOCAL_ARTIFACT_BYTES = 128 * 1024 * 1024 * 1024


class MatrixError(RuntimeError):
    """The matrix or an acquisition request failed closed."""


class TaskCategory(str, Enum):
    STT = "stt"
    STOP_KEYWORD = "stop_keyword"
    FAR_FIELD_DEVICE = "far_field_device"
    NOISE_REVERB = "noise_reverb"
    AEC_DOUBLE_TALK = "aec_double_talk"
    OVERLAP_TURN_TAKING = "overlap_turn_taking"
    SPOKEN_INTENT = "spoken_intent"
    TEXT_SEMANTIC_ROUTING = "text_semantic_routing"


class Metric(str, Enum):
    LITERAL_WER = "literal_wer"
    CANONICAL_WER = "canonical_wer"
    LITERAL_CER = "literal_cer"
    CANONICAL_CER = "canonical_cer"
    STOP_PRECISION = "stop_precision"
    STOP_RECALL = "stop_recall"
    STOP_CONFUSION_RATE = "stop_confusion_rate"
    FALSE_ALARMS_PER_HOUR = "false_alarms_per_hour"
    PAIRED_WER_DELTA = "paired_wer_delta"
    WER_DEGRADATION_GRID = "wer_degradation_grid"
    ERLE_DB = "erle_db"
    AECMOS = "aecmos"
    NEAR_END_ATTENUATION_DB = "near_end_attenuation_db"
    BARGE_MISS_RATE = "barge_miss_rate"
    BARGE_FALSE_RATE = "barge_false_rate"
    BARGE_CUT_LATENCY_MS = "barge_cut_latency_ms"
    OVERLAP_WER = "overlap_wer"
    ENDPOINT_EARLY_CUT_RATE = "endpoint_early_cut_rate"
    ENDPOINT_DELAY_MS = "endpoint_delay_ms"
    BACKCHANNEL_RETENTION = "backchannel_retention"
    INTENT_ACCURACY = "intent_accuracy"
    SLOT_F1 = "slot_f1"
    TOOL_SELECTION_ACCURACY = "tool_selection_accuracy"
    TOOL_RESULT_ACCURACY = "tool_result_accuracy"


class ChecksumAlgorithm(str, Enum):
    SHA256 = "sha256"
    MD5 = "md5"
    GIT_SHA1 = "git_sha1"


class IntegrityKind(str, Enum):
    PINNED = "pinned"
    MDC_API_SHA256 = "mdc_api_sha256"
    LOCAL_SHA256_RECEIPT = "local_sha256_receipt"
    GIT_COMMIT = "git_commit"
    GIT_COMMIT_AND_LFS_OIDS = "git_commit_and_lfs_oids"


class SelectionAlgorithm(str, Enum):
    COMPLETE_SPLIT = "complete_split"
    HASH_TOP_K = "sha256_top_k_v1"
    HASH_STRATIFIED = "sha256_stratified_round_robin_v1"
    HASH_SPEAKER_STRATIFIED = "sha256_speaker_stratified_matching_v1"
    FIXED_IDS = "fixed_ids"
    FIXED_TRANSFORM_GRID = "fixed_transform_grid"
    PAIRED_COMPLETE = "paired_complete"
    HUMAN_LABEL_MAP_REQUIRED = "blocked_until_human_label_map_v1"


class UsageConstraint(str, Enum):
    EVALUATION_ONLY = "evaluation_only"
    PRIVATE_CACHE_ONLY = "private_cache_only"
    NO_REIDENTIFICATION = "no_speaker_reidentification"
    NO_REHOSTING = "no_rehosting"
    NO_VOICE_CLONING = "no_voice_cloning_or_published_voice_clone"
    HUMAN_LABEL_MAP_REQUIRED = "human_or_author_filename_transcript_map_required"


@dataclass(frozen=True, slots=True)
class Checksum:
    algorithm: ChecksumAlgorithm
    digest: str
    source: str


@dataclass(frozen=True, slots=True)
class Artifact:
    artifact_id: str
    locator: str
    filename: str | None
    size_bytes: int | None
    integrity: IntegrityKind
    checksum: Checksum | None = None
    integrity_note: str = ""


@dataclass(frozen=True, slots=True)
class LicenseGate:
    license_id: str
    license_url: str
    acceptance_ids: tuple[str, ...]
    redistribution: str = "cache-only"


@dataclass(frozen=True, slots=True)
class SelectionPolicy:
    description: str
    algorithm: SelectionAlgorithm = SelectionAlgorithm.COMPLETE_SPLIT
    split: str = "test"
    identity_fields: tuple[str, ...] = ()
    strata_fields: tuple[str, ...] = ()
    seed: str | None = None
    expected_examples: int | None = None
    speaker_disjoint: bool = False
    speaker_field: str | None = None
    fixed_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Dataset:
    dataset_id: str
    name: str
    upstream_id: str
    revision: str
    canonical_url: str
    license: LicenseGate
    artifacts: tuple[Artifact, ...]
    selection: SelectionPolicy
    credential_env: str | None = None
    evidence_note: str = ""
    evaluation_only: bool = True
    usage_constraints: tuple[UsageConstraint, ...] = (
        UsageConstraint.EVALUATION_ONLY,
        UsageConstraint.PRIVATE_CACHE_ONLY,
    )


@dataclass(frozen=True, slots=True)
class EvaluationTrack:
    track_id: str
    category: TaskCategory
    dataset_ids: tuple[str, ...]
    metrics: tuple[Metric, ...]
    protocol: str


@dataclass(frozen=True, slots=True)
class ExcludedDataset:
    dataset_id: str
    name: str
    reason: str
    selectable: bool = False


@dataclass(frozen=True, slots=True)
class PlannedArtifact:
    dataset_id: str
    artifact_id: str
    locator: str
    destination: Path
    checksum: Checksum
    size_bytes: int | None
    credential_env: str | None


@dataclass(frozen=True, slots=True)
class AcquisitionPlan:
    root: Path
    dataset_ids: tuple[str, ...]
    artifacts: tuple[PlannedArtifact, ...]
    credential_env_names: tuple[str, ...]
    usage_constraints: tuple[tuple[str, tuple[UsageConstraint, ...]], ...]


CC_BY_4 = LicenseGate(
    "CC-BY-4.0",
    "https://creativecommons.org/licenses/by/4.0/legalcode",
    ("CC-BY-4.0",),
)
CC0_MDC = LicenseGate(
    "CC0-1.0+MDC-consumer-terms",
    "https://mozilladatacollective.com/terms/consumers",
    ("CC0-1.0", "MDC-CONSUMER-TERMS-2026-05-06"),
)
APACHE_2 = LicenseGate(
    "Apache-2.0",
    "https://www.apache.org/licenses/LICENSE-2.0",
    ("Apache-2.0",),
)
CC_BY_SA_3 = LicenseGate(
    "CC-BY-SA-3.0",
    "https://creativecommons.org/licenses/by-sa/3.0/legalcode",
    ("CC-BY-SA-3.0",),
)
AEC_MIXED = LicenseGate(
    (
        "LicenseRef-Microsoft-AEC-Challenge-Mixed:LibriVox-PD+VCTK-terms+"
        "AudioSet-CC-BY-4.0+Freesound-CC0-1.0+DEMAND-CC-BY-SA-3.0"
    ),
    "https://github.com/microsoft/AEC-Challenge/tree/6c633d0a9d2a143a0e364899b91b06f127315b18",
    (
        "LIBRIVOX-PUBLIC-DOMAIN",
        "EDINBURGH-VCTK-DATA-TERMS",
        "AUDIOSET-CC-BY-4.0",
        "FREESOUND-SELECTED-CC0-1.0",
        "DEMAND-CC-BY-SA-3.0",
    ),
)


def _sha256(digest: str, source: str = "upstream") -> Checksum:
    return Checksum(ChecksumAlgorithm.SHA256, digest, source)


def _md5(digest: str) -> Checksum:
    return Checksum(ChecksumAlgorithm.MD5, digest, "upstream")


def _git(digest: str) -> Checksum:
    return Checksum(ChecksumAlgorithm.GIT_SHA1, digest, "upstream-git")


def _mdc_dataset(
    *,
    dataset_id: str,
    name: str,
    mdc_id: str,
    filename: str | None,
    approximate_size: str,
    selection: SelectionPolicy,
    forbid_voice_cloning: bool = False,
    revision: str | None = None,
    size_bytes: int | None = None,
    checksum: Checksum | None = None,
) -> Dataset:
    usage_constraints = (
        UsageConstraint.EVALUATION_ONLY,
        UsageConstraint.PRIVATE_CACHE_ONLY,
        UsageConstraint.NO_REIDENTIFICATION,
        UsageConstraint.NO_REHOSTING,
        *((UsageConstraint.NO_VOICE_CLONING,) if forbid_voice_cloning else ()),
    )
    return Dataset(
        dataset_id=dataset_id,
        name=name,
        upstream_id=f"mdc:{mdc_id}",
        revision=mdc_id if revision is None else revision,
        canonical_url=f"https://mozilladatacollective.com/datasets/{mdc_id}",
        license=CC0_MDC,
        artifacts=(
            Artifact(
                "archive",
                f"mdc://dataset/{mdc_id}",
                filename,
                size_bytes,
                IntegrityKind.MDC_API_SHA256,
                checksum,
                integrity_note=(
                    "Require the MDC API's SHA-256 receipt after terms acceptance"
                    + (
                        " and exact agreement with the catalog pin"
                        if checksum is not None
                        else ""
                    )
                    + f"; catalog display size is {approximate_size}."
                ),
            ),
        ),
        selection=selection,
        credential_env=MDC_TOKEN_ENV,
        usage_constraints=usage_constraints,
    )


DATASETS: tuple[Dataset, ...] = (
    Dataset(
        "speech_commands_v002_test",
        "Google Speech Commands v0.02 test",
        "google/speech_commands:v0.02:test",
        "57ba463ab37e1e7845e0626539a6f6d0fcfbe64a",
        "https://huggingface.co/datasets/google/speech_commands",
        CC_BY_4,
        (
            Artifact(
                "test-archive",
                "https://s3.amazonaws.com/datasets.huggingface.co/SpeechCommands/v0.02/v0.02_test.tar.gz",
                "v0.02_test.tar.gz",
                None,
                IntegrityKind.PINNED,
                _sha256(
                    "af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58"
                ),
            ),
        ),
        SelectionPolicy("Complete upstream test split.", expected_examples=4_890),
        evidence_note=(
            "Parakeet training includes Speech Commands; use this as a STOP "
            "diagnostic, not independent candidate-selection evidence."
        ),
    ),
    Dataset(
        "rochester_smart_speaker_v1",
        "University of Rochester Smart Speaker Command Dataset v1",
        "figshare:26417548:v1",
        "10.60593/ur.d.26417548.v1",
        "https://rochester.figshare.com/articles/dataset/26417548",
        CC_BY_4,
        (
            Artifact(
                "archive",
                "https://ndownloader.figshare.com/files/48058378",
                "SmartSpeakerCommandDataset_RutowskiEtAl_2023.zip",
                470_415_384,
                IntegrityKind.PINNED,
                _sha256(
                    "5cb2f37ad1c4646cc870e653e80eb7e10028ad9d67608e985b8625d45024757d",
                    "local-freeze-after-upstream-md5-f0561310a234d2be445a5376f08e75c6",
                ),
                "Figshare file 48058378 publishes MD5 f0561310a234d2be445a5376f08e75c6.",
            ),
            Artifact(
                "manual",
                "https://ndownloader.figshare.com/files/48058369",
                "SmartSpeakerCommandDataset_Manual_2023.pdf",
                62_533,
                IntegrityKind.PINNED,
                _sha256(
                    "0697dc54ec2365ae534de4135aa4409e55cf4b826eeb782facdf78032d135475",
                    "local-freeze-after-upstream-md5-0299663feb01cad5c55f8483385525da",
                ),
                "Figshare file 48058369 publishes MD5 0299663feb01cad5c55f8483385525da.",
            ),
        ),
        SelectionPolicy(
            (
                "After a checksum-bound human/author filename-to-transcript map "
                "exists, hash-select 28 takes across seven speakers and four "
                "command domains. Never infer labels from numbering."
            ),
            algorithm=SelectionAlgorithm.HUMAN_LABEL_MAP_REQUIRED,
            identity_fields=("filename", "speaker", "command", "take"),
            strata_fields=("speaker", "command_domain"),
            seed=SELECTION_SEED,
            expected_examples=28,
        ),
        evidence_note=(
            "The deposit has an ordered prose transcription list but no "
            "authoritative machine-readable filename map. Acquisition may be "
            "verified now; WER publication must fail closed until the label-map "
            "constraint is satisfied."
        ),
        usage_constraints=(
            UsageConstraint.EVALUATION_ONLY,
            UsageConstraint.PRIVATE_CACHE_ONLY,
            UsageConstraint.HUMAN_LABEL_MAP_REQUIRED,
        ),
    ),
    _mdc_dataset(
        dataset_id="common_voice_26_ro",
        name="Common Voice Scripted Speech 26.0 Romanian",
        mdc_id="cmqinu08d00xenq07ha33e29y",
        filename="common-voice-scripted-speech-26-0-romani-32698751.tar.gz",
        approximate_size="1.08 GB",
        selection=SelectionPolicy(
            "Hash-stratified 250-row test subset by speaker, accent, and duration.",
            algorithm=SelectionAlgorithm.HASH_STRATIFIED,
            identity_fields=("path", "client_id"),
            strata_fields=("accent", "duration_bucket"),
            seed=SELECTION_SEED,
            expected_examples=250,
            speaker_disjoint=True,
            speaker_field="client_id",
        ),
    ),
    _mdc_dataset(
        dataset_id="common_voice_spontaneous_4_en",
        name="Common Voice Spontaneous Speech 4.0 English",
        mdc_id="cmqialpeo0077nr077xqdqo0j",
        filename="common-voice-spontaneous-speech-4-0-engl-c643378f.tar.gz",
        approximate_size="497.82 MB",
        revision="sps-corpus-4.0-2026-06-12",
        size_bytes=522_005_930,
        checksum=_sha256(
            "3b03ada7676a5f440a797d896035137fd073d0683133c3e9a83963480d88abfe",
            "cv-dataset-stats",
        ),
        selection=SelectionPolicy(
            (
                "Exactly 32 clean/unannotated validated test rows: require empty "
                "quality_tags, a non-empty transcription, one row per client_id, "
                "and duration filters [2,6),[6,10),[10,15),[15,25] seconds."
            ),
            algorithm=SelectionAlgorithm.HASH_SPEAKER_STRATIFIED,
            split="test",
            identity_fields=("audio_id", "audio_file", "client_id"),
            strata_fields=("duration_bucket",),
            seed="speaker-cvss4-en-v1-2026-08-01",
            expected_examples=32,
            speaker_disjoint=True,
            speaker_field="client_id",
        ),
    ),
    _mdc_dataset(
        dataset_id="common_voice_26_southern_american",
        name="Common Voice 26.0 Southern American English accent set",
        mdc_id="cmryvrc6i014vo90786kriix5",
        filename=None,
        approximate_size="32.34 MB",
        selection=SelectionPolicy(
            "Complete dataset-defined speaker-disjoint test split.",
            speaker_disjoint=True,
            speaker_field="client_id",
        ),
        forbid_voice_cloning=True,
    ),
    _mdc_dataset(
        dataset_id="common_voice_26_malaysian_english",
        name="Common Voice 26.0 Malaysian English accent set",
        mdc_id="cmrt71n6q001fmm07u64adwuv",
        filename=None,
        approximate_size="86.30 MB",
        selection=SelectionPolicy(
            "Complete dataset-defined speaker-disjoint test split.",
            speaker_disjoint=True,
            speaker_field="client_id",
        ),
    ),
    _mdc_dataset(
        dataset_id="common_voice_26_irish_english",
        name="Common Voice 26.0 Irish English accent set",
        mdc_id="cmrt71fck001bmm07cj9h312i",
        filename=None,
        approximate_size="330.28 MB",
        selection=SelectionPolicy(
            "Complete dataset-defined speaker-disjoint test split.",
            speaker_disjoint=True,
            speaker_field="client_id",
        ),
    ),
    Dataset(
        "speechocean762",
        "SpeechOcean762",
        "OpenSLR:SLR101",
        "SLR101-2021-02-02",
        "https://www.openslr.org/101/",
        CC_BY_4,
        (
            Artifact(
                "archive",
                "https://www.openslr.org/resources/101/speechocean762.tar.gz",
                "speechocean762.tar.gz",
                None,
                IntegrityKind.LOCAL_SHA256_RECEIPT,
                integrity_note="No upstream digest; freeze a locally computed SHA-256 before use.",
            ),
        ),
        SelectionPolicy(
            "Hash-stratified 250-case diagnostic from the official test set.",
            algorithm=SelectionAlgorithm.HASH_STRATIFIED,
            identity_fields=("utterance_id",),
            strata_fields=("speaker_id",),
            seed=SELECTION_SEED,
            expected_examples=250,
            speaker_disjoint=True,
            speaker_field="speaker_id",
        ),
    ),
    Dataset(
        "dr_vctk_small",
        "DR-VCTK Small",
        "doi:10.7488/ds/2316",
        "10.7488/ds/2316",
        "https://doi.org/10.7488/ds/2316",
        CC_BY_4,
        (
            Artifact(
                "archive",
                "https://doi.org/10.7488/ds/2316",
                None,
                None,
                IntegrityKind.PINNED,
                _md5("29e93debeb0e779986542229a81ff29b"),
                "Catalog size is 1.67 GB; verify the DOI-delivered archive by MD5.",
            ),
        ),
        SelectionPolicy(
            "Preserve upstream paired clean/device utterances and speakers.",
            algorithm=SelectionAlgorithm.PAIRED_COMPLETE,
        ),
    ),
    Dataset(
        "rirs_noises",
        "RIRS_NOISES",
        "OpenSLR:SLR28",
        "SLR28",
        "https://www.openslr.org/28/",
        APACHE_2,
        (
            Artifact(
                "archive",
                "https://www.openslr.org/resources/28/rirs_noises.zip",
                "rirs_noises.zip",
                None,
                IntegrityKind.PINNED,
                _md5("e6f48e257286e05de56413b4779d8ffb"),
                "Catalog size is 1.3 GB.",
            ),
        ),
        SelectionPolicy(
            "Fixed-seed room/noise/SNR degradation grid; never resample cases randomly.",
            algorithm=SelectionAlgorithm.FIXED_TRANSFORM_GRID,
            seed=SELECTION_SEED,
        ),
    ),
    Dataset(
        "demand_domestic_16k",
        "DEMAND domestic 16 kHz noise subset",
        "zenodo:1227121:DKITCHEN+DLIVING+DWASHING",
        "10.5281/zenodo.1227121",
        "https://zenodo.org/records/1227121",
        CC_BY_SA_3,
        (
            Artifact(
                "dkitchen",
                (
                    "https://zenodo.org/api/records/1227121/files/"
                    "DKITCHEN_16k.zip/content"
                ),
                "DKITCHEN_16k.zip",
                110_501_049,
                IntegrityKind.PINNED,
                _sha256(
                    "9d68e46d2709847c59580770e2f8aa160607ebff7475788174aa1080332cc845",
                    "local-freeze-after-upstream-md5-7ffbf52d7f4699f96927846103dc8788",
                ),
                "Zenodo publishes MD5 7ffbf52d7f4699f96927846103dc8788.",
            ),
            Artifact(
                "dliving",
                (
                    "https://zenodo.org/api/records/1227121/files/"
                    "DLIVING_16k.zip/content"
                ),
                "DLIVING_16k.zip",
                80_170_627,
                IntegrityKind.PINNED,
                _sha256(
                    "2b1726fe06e41551ce2397f2aaf3e4fb692c912d81914b04708df0bbb5252338",
                    "local-freeze-after-upstream-md5-46741384d9e434a0bd8b3ec1830b6052",
                ),
                "Zenodo publishes MD5 46741384d9e434a0bd8b3ec1830b6052.",
            ),
            Artifact(
                "dwashing",
                (
                    "https://zenodo.org/api/records/1227121/files/"
                    "DWASHING_16k.zip/content"
                ),
                "DWASHING_16k.zip",
                102_343_250,
                IntegrityKind.PINNED,
                _sha256(
                    "f687c0806b0e9731b71a9c01c6134a21330efc5fa4aa7ee97cf4df276ad1dae1",
                    "local-freeze-after-upstream-md5-7e5ee9437ce9409c5f9a779b6212a240",
                ),
                "Zenodo publishes MD5 7e5ee9437ce9409c5f9a779b6212a240.",
            ),
        ),
        SelectionPolicy(
            (
                "Use channel 01 with deterministic non-overlapping intervals "
                "and a fixed 20/10/0 dB SNR rotation over kitchen, living-room, "
                "and washing-machine conditions."
            ),
            algorithm=SelectionAlgorithm.FIXED_TRANSFORM_GRID,
            identity_fields=("environment", "channel", "sample_offset"),
            strata_fields=("environment", "snr_db"),
            seed=SELECTION_SEED,
        ),
        evidence_note=(
            "The deposit prose and original paper specify CC BY-SA 3.0 while "
            "current Zenodo structured metadata says CC BY 4.0; the catalog "
            "conservatively applies ShareAlike 3.0 and keeps derivatives private."
        ),
    ),
    Dataset(
        "microsoft_aec_challenge",
        "Microsoft AEC Challenge synthetic set",
        "github:microsoft/AEC-Challenge",
        "6c633d0a9d2a143a0e364899b91b06f127315b18",
        "https://github.com/microsoft/AEC-Challenge",
        AEC_MIXED,
        (
            Artifact(
                "repository",
                "https://github.com/microsoft/AEC-Challenge.git",
                None,
                None,
                IntegrityKind.GIT_COMMIT_AND_LFS_OIDS,
                _git("6c633d0a9d2a143a0e364899b91b06f127315b18"),
                (
                    "Verify every selected WAV against its embedded Git LFS "
                    "SHA-256 OID and bind its meta.csv source provenance to the "
                    "explicit LibriVox/VCTK/AudioSet/Freesound/DEMAND terms."
                ),
            ),
        ),
        SelectionPolicy(
            "Sort synthetic/meta.csv IDs by SHA-256(seed || ID), take 32, and bind all four LFS WAVs per ID.",
            algorithm=SelectionAlgorithm.HASH_TOP_K,
            identity_fields=("file_id",),
            seed=SELECTION_SEED,
            expected_examples=32,
        ),
    ),
    Dataset(
        "ami_eval_three",
        "AMI official evaluation meetings (three-channel subset)",
        "AMI:ES2004a+IS1009a+TS3003a",
        "manual-annotations-1.6.2",
        "https://groups.inf.ed.ac.uk/ami/corpus/",
        CC_BY_4,
        (
            Artifact(
                "annotations",
                "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip",
                "ami_public_manual_1.6.2.zip",
                22_887_865,
                IntegrityKind.PINNED,
                _sha256(
                    "b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d",
                    "local-freeze",
                ),
            ),
            Artifact(
                "es2004a-array1-01",
                "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2004a/audio/ES2004a.Array1-01.wav",
                "ES2004a.Array1-01.wav",
                33_579_394,
                IntegrityKind.PINNED,
                _sha256(
                    "6936edac5d0904fc5c4ab175546c5cc5366601fdc1b1e5183a6ea2c10f05d150",
                    "local-freeze",
                ),
            ),
            Artifact(
                "is1009a-array1-01",
                "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/IS1009a/audio/IS1009a.Array1-01.wav",
                "IS1009a.Array1-01.wav",
                26_849_383,
                IntegrityKind.PINNED,
                _sha256(
                    "23fd54fd6dbe23870f10317b256f7c648fe0fd587e56d2f940840309f1578b60",
                    "local-freeze",
                ),
            ),
            Artifact(
                "ts3003a-array1-01",
                "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/TS3003a/audio/TS3003a.Array1-01.wav",
                "TS3003a.Array1-01.wav",
                48_183_678,
                IntegrityKind.PINNED,
                _sha256(
                    "46f81fb403e40a98c1b694c842404c0db24f3a6c67397b247faf0d3f42cf9163",
                    "local-freeze",
                ),
            ),
        ),
        SelectionPolicy(
            "Only ES2004a, IS1009a, and TS3003a; explicitly exclude dropout-prone ES2004d.",
            algorithm=SelectionAlgorithm.FIXED_IDS,
            identity_fields=("meeting_id",),
            fixed_ids=("ES2004a", "IS1009a", "TS3003a"),
            expected_examples=3,
        ),
        evidence_note=(
            "Parakeet training includes AMI; this is overlap/turn diagnostic evidence, "
            "not independent candidate-selection evidence."
        ),
    ),
    Dataset(
        "minds14_english",
        "MInDS-14 English locales",
        "PolyAI/minds14:en-US+en-GB+en-AU",
        "40ce77cb32a384e4d50a568e1ec39ac804019d33",
        "https://huggingface.co/datasets/PolyAI/minds14",
        CC_BY_4,
        (
            Artifact(
                "en-us",
                "https://huggingface.co/datasets/PolyAI/minds14/resolve/40ce77cb32a384e4d50a568e1ec39ac804019d33/en-US/train-00000-of-00001.parquet",
                "en-US-train.parquet",
                34_196_221,
                IntegrityKind.PINNED,
                _sha256(
                    "37004471dc896ce20771b3fdda0ee8fb33ec7a030fb4e2fd047c561ec0a1ee30"
                ),
            ),
            Artifact(
                "en-gb",
                "https://huggingface.co/datasets/PolyAI/minds14/resolve/40ce77cb32a384e4d50a568e1ec39ac804019d33/en-GB/train-00000-of-00001.parquet",
                "en-GB-train.parquet",
                34_551_079,
                IntegrityKind.PINNED,
                _sha256(
                    "52c988dc66991be64109bb7043fd3acb8c00a7cfc1006670ee0b3906868a9b00"
                ),
            ),
            Artifact(
                "en-au",
                "https://huggingface.co/datasets/PolyAI/minds14/resolve/40ce77cb32a384e4d50a568e1ec39ac804019d33/en-AU/train-00000-of-00001.parquet",
                "en-AU-train.parquet",
                37_348_052,
                IntegrityKind.PINNED,
                _sha256(
                    "74476455327528992aa23a321ca3f1de3037c863a7d76b56e2a05df19cba4bab"
                ),
            ),
        ),
        SelectionPolicy(
            "Preserve the pinned en-US/en-GB/en-AU locale partitions and upstream labels.",
            seed=SELECTION_SEED,
        ),
    ),
    Dataset(
        "massive_1_1_text",
        "MASSIVE 1.1 text semantics",
        "AmazonScience/massive:en-US+ro-RO",
        "ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740",
        "https://huggingface.co/datasets/AmazonScience/massive",
        CC_BY_4,
        (
            Artifact(
                "archive",
                "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz",
                "amazon-massive-dataset-1.1.tar.gz",
                None,
                IntegrityKind.LOCAL_SHA256_RECEIPT,
                integrity_note=(
                    "The pinned Hugging Face loader revision names this exact "
                    "official 1.1 archive, but publishes no archive digest; "
                    "freeze a locally computed SHA-256 before use."
                ),
            ),
        ),
        SelectionPolicy("Complete upstream en-US and ro-RO test partitions."),
    ),
)


TRACKS: tuple[EvaluationTrack, ...] = (
    EvaluationTrack(
        "stt",
        TaskCategory.STT,
        (
            "common_voice_26_ro",
            "common_voice_spontaneous_4_en",
            "common_voice_26_southern_american",
            "common_voice_26_malaysian_english",
            "common_voice_26_irish_english",
            "speechocean762",
            "minds14_english",
            "rochester_smart_speaker_v1",
        ),
        (
            Metric.LITERAL_WER,
            Metric.CANONICAL_WER,
            Metric.LITERAL_CER,
            Metric.CANONICAL_CER,
        ),
        "Report literal and canonical normalization separately; never publish only the better score.",
    ),
    EvaluationTrack(
        "stop-keyword",
        TaskCategory.STOP_KEYWORD,
        ("speech_commands_v002_test",),
        (
            Metric.STOP_PRECISION,
            Metric.STOP_RECALL,
            Metric.STOP_CONFUSION_RATE,
            Metric.FALSE_ALARMS_PER_HOUR,
        ),
        "Grade STOP against other commands, unknown words, and silence; do not substitute WER.",
    ),
    EvaluationTrack(
        "far-field-device",
        TaskCategory.FAR_FIELD_DEVICE,
        ("dr_vctk_small", "ami_eval_three"),
        (Metric.LITERAL_WER, Metric.CANONICAL_WER, Metric.PAIRED_WER_DELTA),
        "Keep paired clean/device deltas separate from AMI far-field absolute WER.",
    ),
    EvaluationTrack(
        "noise-reverb",
        TaskCategory.NOISE_REVERB,
        ("rirs_noises", "speechocean762", "demand_domestic_16k"),
        (Metric.WER_DEGRADATION_GRID,),
        "Apply the fixed seed grid to pinned reference speech and report degradation from its clean decode.",
    ),
    EvaluationTrack(
        "aec-double-talk",
        TaskCategory.AEC_DOUBLE_TALK,
        ("microsoft_aec_challenge",),
        (
            Metric.ERLE_DB,
            Metric.AECMOS,
            Metric.NEAR_END_ATTENUATION_DB,
            Metric.BARGE_MISS_RATE,
            Metric.BARGE_FALSE_RATE,
            Metric.BARGE_CUT_LATENCY_MS,
        ),
        "Grade echo and barge behavior from paired signals; WER is intentionally forbidden.",
    ),
    EvaluationTrack(
        "overlap-turn-taking",
        TaskCategory.OVERLAP_TURN_TAKING,
        ("ami_eval_three",),
        (
            Metric.OVERLAP_WER,
            Metric.ENDPOINT_EARLY_CUT_RATE,
            Metric.ENDPOINT_DELAY_MS,
            Metric.BACKCHANNEL_RETENTION,
        ),
        "Use manual timing annotations and report overlap, endpoint, and backchannel results separately.",
    ),
    EvaluationTrack(
        "spoken-intent",
        TaskCategory.SPOKEN_INTENT,
        ("minds14_english",),
        (Metric.INTENT_ACCURACY,),
        "Score intent on audio independently of the STT track's WER/CER report.",
    ),
    EvaluationTrack(
        "text-semantic-routing",
        TaskCategory.TEXT_SEMANTIC_ROUTING,
        ("massive_1_1_text",),
        (
            Metric.INTENT_ACCURACY,
            Metric.SLOT_F1,
            Metric.TOOL_SELECTION_ACCURACY,
            Metric.TOOL_RESULT_ACCURACY,
        ),
        "Use text only; report semantic routing separately from speech recognition.",
    ),
)


EXCLUSIONS: tuple[ExcludedDataset, ...] = (
    ExcludedDataset(
        "smart_turn_v3_2_test",
        "Pipecat Smart Turn v3.2 test data",
        "No dataset license is published.",
    ),
    ExcludedDataset(
        "notsofar_1_open_subsets",
        "NOTSOFAR-1 open train/dev-1/eval subsets",
        "CC-BY-4.0 is eligible, but this slice lacks a pinned immutable artifact/checksum receipt.",
    ),
    ExcludedDataset(
        "notsofar_1_dev_set_2",
        "NOTSOFAR-1 Dev-set-2",
        "Dev-set-2 is challenge-only and is not covered by the open subset grant.",
    ),
    ExcludedDataset(
        "libricss", "LibriCSS", "Data license and stable artifact checksum are unclear."
    ),
    ExcludedDataset(
        "slurp_audio",
        "SLURP audio",
        "CC-BY-NC terms are incompatible with this selectable matrix.",
    ),
    ExcludedDataset("l2_arctic", "L2-ARCTIC", "CC-BY-NC plus a manual access gate."),
    ExcludedDataset("morovoc", "MoRoVoc", "No reference transcripts for STT grading."),
    ExcludedDataset(
        "openwakeword_collections",
        "openWakeWord training collections",
        "Collection terms are often non-commercial or unclear.",
    ),
    ExcludedDataset(
        "hey_snips",
        "Hey Snips",
        "Request/research-only access is not a reusable public grant.",
    ),
    ExcludedDataset(
        "fluent_speech_commands",
        "Fluent Speech Commands",
        (
            "The official research-only license forbids product testing, "
            "benchmarking, and development; a third-party metadata label "
            "cannot broaden that upstream grant."
        ),
    ),
    ExcludedDataset(
        "sonos_voice_control_dataset",
        "Sonos voice-control research dataset",
        (
            "The official grant is research/non-commercial and acquisition is "
            "form-gated, so it is not eligible for this reusable product matrix."
        ),
    ),
)


_ALLOWED_METRICS: Mapping[TaskCategory, frozenset[Metric]] = {
    TaskCategory.STT: frozenset(
        {
            Metric.LITERAL_WER,
            Metric.CANONICAL_WER,
            Metric.LITERAL_CER,
            Metric.CANONICAL_CER,
        }
    ),
    TaskCategory.STOP_KEYWORD: frozenset(
        {
            Metric.STOP_PRECISION,
            Metric.STOP_RECALL,
            Metric.STOP_CONFUSION_RATE,
            Metric.FALSE_ALARMS_PER_HOUR,
        }
    ),
    TaskCategory.FAR_FIELD_DEVICE: frozenset(
        {Metric.LITERAL_WER, Metric.CANONICAL_WER, Metric.PAIRED_WER_DELTA}
    ),
    TaskCategory.NOISE_REVERB: frozenset({Metric.WER_DEGRADATION_GRID}),
    TaskCategory.AEC_DOUBLE_TALK: frozenset(
        {
            Metric.ERLE_DB,
            Metric.AECMOS,
            Metric.NEAR_END_ATTENUATION_DB,
            Metric.BARGE_MISS_RATE,
            Metric.BARGE_FALSE_RATE,
            Metric.BARGE_CUT_LATENCY_MS,
        }
    ),
    TaskCategory.OVERLAP_TURN_TAKING: frozenset(
        {
            Metric.OVERLAP_WER,
            Metric.ENDPOINT_EARLY_CUT_RATE,
            Metric.ENDPOINT_DELAY_MS,
            Metric.BACKCHANNEL_RETENTION,
        }
    ),
    TaskCategory.SPOKEN_INTENT: frozenset({Metric.INTENT_ACCURACY}),
    TaskCategory.TEXT_SEMANTIC_ROUTING: frozenset(
        {
            Metric.INTENT_ACCURACY,
            Metric.SLOT_F1,
            Metric.TOOL_SELECTION_ACCURACY,
            Metric.TOOL_RESULT_ACCURACY,
        }
    ),
}


def dataset_by_id(dataset_id: str) -> Dataset:
    try:
        return next(dataset for dataset in DATASETS if dataset.dataset_id == dataset_id)
    except StopIteration as exc:
        if any(item.dataset_id == dataset_id for item in EXCLUSIONS):
            raise MatrixError(
                f"dataset {dataset_id!r} is explicitly non-selectable"
            ) from exc
        raise MatrixError(f"unknown dataset {dataset_id!r}") from exc


def default_corpus_root(environ: Mapping[str, str] | None = None) -> Path:
    values = os.environ if environ is None else environ
    configured = values.get("XDG_CACHE_HOME", "").strip()
    base = Path(configured).expanduser() if configured else Path.home() / ".cache"
    return base / "speaker" / "corpora"


def _validate_checksum(checksum: Checksum) -> None:
    patterns = {
        ChecksumAlgorithm.SHA256: _SHA256_RE,
        ChecksumAlgorithm.MD5: _MD5_RE,
        ChecksumAlgorithm.GIT_SHA1: _SHA1_RE,
    }
    if patterns[checksum.algorithm].fullmatch(checksum.digest) is None:
        raise MatrixError(f"invalid {checksum.algorithm.value} digest")
    if not checksum.source or any(char.isspace() for char in checksum.source):
        raise MatrixError("checksum source must be a non-empty token")


def _receipt_for(
    dataset: Dataset,
    artifact: Artifact,
    receipts: Mapping[str, Checksum],
) -> Checksum:
    key = f"{dataset.dataset_id}/{artifact.artifact_id}"
    if artifact.checksum is not None:
        _validate_checksum(artifact.checksum)
        if artifact.integrity is not IntegrityKind.MDC_API_SHA256:
            return artifact.checksum
    try:
        receipt = receipts[key]
    except KeyError as exc:
        raise MatrixError(
            f"{key} requires an explicit SHA-256 checksum receipt before acquisition"
        ) from exc
    _validate_checksum(receipt)
    if receipt.algorithm is not ChecksumAlgorithm.SHA256:
        raise MatrixError(f"{key} receipt must use sha256")
    required_source = {
        IntegrityKind.MDC_API_SHA256: "mdc-api",
        IntegrityKind.LOCAL_SHA256_RECEIPT: "local-freeze",
    }.get(artifact.integrity)
    if required_source is None or receipt.source != required_source:
        raise MatrixError(f"{key} receipt source must be {required_source!r}")
    if artifact.checksum is not None:
        if artifact.checksum.algorithm is not ChecksumAlgorithm.SHA256:
            raise MatrixError(f"{key} catalog pin must use sha256")
        if receipt.digest != artifact.checksum.digest:
            raise MatrixError(f"{key} MDC API receipt does not match the catalog pin")
        return Checksum(
            ChecksumAlgorithm.SHA256,
            artifact.checksum.digest,
            f"{artifact.checksum.source}+mdc-api",
        )
    return receipt


def _safe_root(root: Path) -> Path:
    if not root.is_absolute():
        raise MatrixError("corpus root must be absolute")
    resolved = root.expanduser().resolve()
    if resolved == _REPO_ROOT or _REPO_ROOT in resolved.parents:
        raise MatrixError("raw corpora must stay outside the Git worktree")
    return resolved


def _artifact_destination(root: Path, dataset: Dataset, artifact: Artifact) -> Path:
    filename = artifact.filename or f"{artifact.artifact_id}.source"
    if _SAFE_FILENAME_RE.fullmatch(filename) is None or Path(filename).name != filename:
        raise MatrixError(f"unsafe artifact filename for {dataset.dataset_id}")
    dataset_root = (root / dataset.dataset_id).resolve()
    destination = (dataset_root / filename).resolve()
    if destination.parent != dataset_root or root not in destination.parents:
        raise MatrixError(
            f"artifact destination escapes corpus root for {dataset.dataset_id}"
        )
    return destination


def _canonical_row_fields(
    row: Mapping[str, object],
    fields: tuple[str, ...],
) -> str:
    values: list[str | int] = []
    for field_name in fields:
        value = row.get(field_name)
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise MatrixError(
                f"selection field {field_name!r} must be string or integer"
            )
        values.append(value)
    return json.dumps(
        values,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _selection_rank(seed: str, identity: str) -> str:
    return hashlib.sha256(
        seed.encode("utf-8") + b"\0" + identity.encode("utf-8")
    ).hexdigest()


def select_subset_rows(
    policy: SelectionPolicy,
    rows: Iterable[Mapping[str, object]],
    *,
    forbidden_speaker_ids: frozenset[str] | None = None,
) -> tuple[Mapping[str, object], ...]:
    """Apply the catalog's exact deterministic subset algorithm.

    Complete splits, paired corpora, and transform grids are consumed by their
    own preparers. This helper owns every policy that reduces a row set.
    """

    if policy.algorithm is SelectionAlgorithm.HUMAN_LABEL_MAP_REQUIRED:
        raise MatrixError(
            "selection requires a checksum-bound human/author "
            "filename-to-transcript map"
        )
    if policy.algorithm not in {
        SelectionAlgorithm.HASH_TOP_K,
        SelectionAlgorithm.HASH_STRATIFIED,
        SelectionAlgorithm.HASH_SPEAKER_STRATIFIED,
        SelectionAlgorithm.FIXED_IDS,
    }:
        raise MatrixError("selection policy does not define a row subset")
    if not policy.identity_fields or policy.expected_examples is None:
        raise MatrixError("subset policy lacks identity/count fields")
    materialized: list[tuple[str, Mapping[str, object]]] = []
    identities: set[str] = set()
    speaker_stratified = policy.algorithm is SelectionAlgorithm.HASH_SPEAKER_STRATIFIED
    if speaker_stratified and policy.speaker_field is None:
        raise MatrixError("speaker-stratified selection lacks a speaker field")
    for row in rows:
        identity = _canonical_row_fields(row, policy.identity_fields)
        if identity in identities:
            raise MatrixError("selection row identity is not unique")
        if speaker_stratified:
            speaker = row.get(policy.speaker_field)
            if not isinstance(speaker, str) or not speaker.strip():
                raise MatrixError(
                    "speaker-stratified selection requires non-empty speaker ids"
                )
        identities.add(identity)
        materialized.append((identity, row))

    if policy.algorithm is SelectionAlgorithm.FIXED_IDS:
        fixed_order = {value: index for index, value in enumerate(policy.fixed_ids)}
        if len(fixed_order) != len(policy.fixed_ids):
            raise MatrixError("fixed selection ids are not unique")
        chosen = [
            (fixed_order.get(str(row.get(policy.identity_fields[0]))), identity, row)
            for identity, row in materialized
            if str(row.get(policy.identity_fields[0])) in fixed_order
        ]
        chosen.sort(key=lambda item: (item[0], item[1]))
        selected = [(identity, row) for _order, identity, row in chosen]
    else:
        if policy.seed is None:
            raise MatrixError("hash selection lacks a seed")
        if policy.algorithm is SelectionAlgorithm.HASH_TOP_K:
            ranked = [
                (_selection_rank(policy.seed, identity), identity, row)
                for identity, row in materialized
            ]
            ranked.sort(key=lambda item: (item[0], item[1]))
            selected = [(identity, row) for _rank, identity, row in ranked]
        elif speaker_stratified:
            if not policy.strata_fields or policy.speaker_field is None:
                raise MatrixError("speaker-stratified selection lacks strata fields")
            forbidden = (
                frozenset() if forbidden_speaker_ids is None else forbidden_speaker_ids
            )
            grouped: dict[
                str,
                dict[str, list[tuple[str, str, Mapping[str, object]]]],
            ] = {}
            for identity, row in materialized:
                stratum = _canonical_row_fields(row, policy.strata_fields)
                speaker = str(row[policy.speaker_field])
                if speaker in forbidden:
                    continue
                clip_rank = _selection_rank(
                    policy.seed,
                    f"clip\0{identity}",
                )
                grouped.setdefault(stratum, {}).setdefault(speaker, []).append(
                    (clip_rank, identity, row)
                )
            strata = sorted(grouped)
            if not strata:
                raise MatrixError("selection source has insufficient exact rows")
            base, extra = divmod(policy.expected_examples, len(strata))
            quotas = {
                stratum: base + (1 if index < extra else 0)
                for index, stratum in enumerate(strata)
            }
            candidates: dict[
                str,
                list[tuple[str, str, str, Mapping[str, object]]],
            ] = {}
            for stratum in strata:
                ranked_speakers = []
                for speaker, clips in grouped[stratum].items():
                    clips.sort(key=lambda item: (item[0], item[1]))
                    _clip_rank, identity, row = clips[0]
                    speaker_rank = _selection_rank(
                        policy.seed,
                        f"speaker\0{stratum}\0{speaker}",
                    )
                    ranked_speakers.append((speaker_rank, speaker, identity, row))
                ranked_speakers.sort(key=lambda item: (item[0], item[1], item[2]))
                candidates[stratum] = ranked_speakers

            owner: dict[str, str] = {}

            def augment(stratum: str, seen: set[str]) -> bool:
                for _rank, speaker, _identity, _row in candidates[stratum]:
                    if speaker in seen:
                        continue
                    seen.add(speaker)
                    previous = owner.get(speaker)
                    if previous is None or (
                        previous != stratum and augment(previous, seen)
                    ):
                        owner[speaker] = stratum
                        return True
                return False

            for stratum in strata:
                for _slot in range(quotas[stratum]):
                    if not augment(stratum, set()):
                        raise MatrixError(
                            "selection source has insufficient exact rows"
                        )

            assigned: dict[
                str,
                list[tuple[str, str, Mapping[str, object]]],
            ] = {stratum: [] for stratum in strata}
            for stratum in strata:
                for rank, speaker, identity, row in candidates[stratum]:
                    if owner.get(speaker) == stratum:
                        assigned[stratum].append((rank, identity, row))
                if len(assigned[stratum]) != quotas[stratum]:
                    raise MatrixError("selection source has insufficient exact rows")
            selected = []
            for slot in range(max(quotas.values(), default=0)):
                for stratum in strata:
                    if slot < len(assigned[stratum]):
                        _rank, identity, row = assigned[stratum][slot]
                        selected.append((identity, row))
        else:
            if not policy.strata_fields:
                raise MatrixError("stratified selection lacks strata fields")
            ranked = [
                (_selection_rank(policy.seed, identity), identity, row)
                for identity, row in materialized
            ]
            groups: dict[str, list[tuple[str, str, Mapping[str, object]]]] = {}
            for rank, identity, row in ranked:
                stratum = _canonical_row_fields(row, policy.strata_fields)
                groups.setdefault(stratum, []).append((rank, identity, row))
            for group in groups.values():
                group.sort(key=lambda item: (item[0], item[1]))
            selected = []
            offsets = {stratum: 0 for stratum in groups}
            while len(selected) < policy.expected_examples:
                progressed = False
                for stratum in sorted(groups):
                    group = groups[stratum]
                    while offsets[stratum] < len(group):
                        offset = offsets[stratum]
                        _rank, identity, row = group[offset]
                        offsets[stratum] = offset + 1
                        selected.append((identity, row))
                        progressed = True
                        break
                    if len(selected) == policy.expected_examples:
                        break
                if not progressed:
                    break
        selected = selected[: policy.expected_examples]

    if len(selected) != policy.expected_examples:
        raise MatrixError("selection source has insufficient exact rows")
    if policy.speaker_disjoint:
        if policy.speaker_field is None or (
            forbidden_speaker_ids is None and not speaker_stratified
        ):
            raise MatrixError(
                "speaker-disjoint selection requires a forbidden-speaker set"
            )
        forbidden = (
            frozenset() if forbidden_speaker_ids is None else forbidden_speaker_ids
        )
        observed_speakers: set[str] = set()
        for _identity, row in selected:
            speaker = row.get(policy.speaker_field)
            if (
                not isinstance(speaker, str)
                or not speaker.strip()
                or speaker in forbidden
                or (speaker_stratified and speaker in observed_speakers)
            ):
                raise MatrixError("selection violates speaker-disjoint contract")
            observed_speakers.add(speaker)
    return tuple(row for _identity, row in selected)


def selected_rows_sha256(
    policy: SelectionPolicy,
    rows: Iterable[Mapping[str, object]],
) -> str:
    """Bind the ordered canonical identities emitted by ``select_subset_rows``."""

    identities = [_canonical_row_fields(row, policy.identity_fields) for row in rows]
    payload = json.dumps(
        identities,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def selection_policy_sha256(policy: SelectionPolicy) -> str:
    """Bind every field in a selection recipe using canonical JSON."""

    payload = json.dumps(
        {
            "algorithm": policy.algorithm.value,
            "description": policy.description,
            "expected_examples": policy.expected_examples,
            "fixed_ids": list(policy.fixed_ids),
            "identity_fields": list(policy.identity_fields),
            "seed": policy.seed,
            "speaker_disjoint": policy.speaker_disjoint,
            "speaker_field": policy.speaker_field,
            "split": policy.split,
            "strata_fields": list(policy.strata_fields),
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_acquisition_plan(
    dataset_ids: tuple[str, ...],
    *,
    root: Path,
    accepted_terms: frozenset[str],
    checksum_receipts: Mapping[str, Checksum] | None = None,
    environ: Mapping[str, str] | None = None,
) -> AcquisitionPlan:
    """Build a deterministic acquisition plan without performing I/O or network.

    Credential *values* are checked for presence only.  They are never retained
    in the plan or interpolated into an exception.
    """

    if not dataset_ids:
        raise MatrixError("at least one dataset must be selected explicitly")
    if len(dataset_ids) != len(set(dataset_ids)):
        raise MatrixError("dataset selection contains duplicates")
    resolved_root = _safe_root(root)
    receipts = {} if checksum_receipts is None else checksum_receipts
    values = os.environ if environ is None else environ
    planned: list[PlannedArtifact] = []
    credential_names: set[str] = set()
    usage_constraints: list[tuple[str, tuple[UsageConstraint, ...]]] = []
    for dataset_id in sorted(dataset_ids):
        dataset = dataset_by_id(dataset_id)
        usage_constraints.append((dataset.dataset_id, dataset.usage_constraints))
        missing_terms = sorted(set(dataset.license.acceptance_ids) - accepted_terms)
        if missing_terms:
            raise MatrixError(
                f"{dataset_id} requires explicit acceptance of: {', '.join(missing_terms)}"
            )
        if dataset.credential_env:
            credential_names.add(dataset.credential_env)
            if not values.get(dataset.credential_env):
                raise MatrixError(
                    f"{dataset_id} requires credential environment variable "
                    f"{dataset.credential_env}"
                )
        for artifact in sorted(dataset.artifacts, key=lambda item: item.artifact_id):
            checksum = _receipt_for(dataset, artifact, receipts)
            destination = _artifact_destination(resolved_root, dataset, artifact)
            planned.append(
                PlannedArtifact(
                    dataset.dataset_id,
                    artifact.artifact_id,
                    artifact.locator,
                    destination,
                    checksum,
                    artifact.size_bytes,
                    dataset.credential_env,
                )
            )
    return AcquisitionPlan(
        resolved_root,
        tuple(sorted(dataset_ids)),
        tuple(planned),
        tuple(sorted(credential_names)),
        tuple(usage_constraints),
    )


def _file_identity(value: os.stat_result) -> tuple[int, ...]:
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


def _stable_file_digests(
    path: Path,
    *,
    expected_bytes: int | None,
) -> tuple[int, str, str]:
    descriptor = -1
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        named_before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(named_before.st_mode)
            or before.st_nlink != 1
            or _file_identity(before) != _file_identity(named_before)
            or before.st_size < 0
            or before.st_size > _MAX_LOCAL_ARTIFACT_BYTES
            or (expected_bytes is not None and before.st_size != expected_bytes)
        ):
            raise MatrixError()
        sha256 = hashlib.sha256()
        md5 = hashlib.md5()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            sha256.update(chunk)
            md5.update(chunk)
        after = os.fstat(descriptor)
        named_after = path.lstat()
        if _file_identity(before) != _file_identity(after) or _file_identity(
            after
        ) != _file_identity(named_after):
            raise MatrixError()
        return before.st_size, sha256.hexdigest(), md5.hexdigest()
    except (OSError, ValueError, MatrixError):
        raise MatrixError(f"artifact is unavailable or unstable: {path.name}") from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def verify_local_artifact(plan: PlannedArtifact, path: Path) -> Checksum:
    """Verify the exact planned file and return its strong local SHA-256 freeze."""

    if not path.is_absolute() or Path(os.path.abspath(path)) != plan.destination:
        raise MatrixError("artifact path must equal its absolute planned destination")
    algorithm = plan.checksum.algorithm
    if algorithm is ChecksumAlgorithm.GIT_SHA1:
        raise MatrixError(
            "Git revisions require a repository/LFS verifier, not file hashing"
        )
    _size, sha256, md5 = _stable_file_digests(
        path,
        expected_bytes=plan.size_bytes,
    )
    observed = sha256 if algorithm is ChecksumAlgorithm.SHA256 else md5
    if observed != plan.checksum.digest:
        raise MatrixError(f"artifact checksum mismatch: {path.name}")
    return Checksum(ChecksumAlgorithm.SHA256, sha256, "local-freeze")


def validate_matrix() -> None:
    dataset_ids = [dataset.dataset_id for dataset in DATASETS]
    exclusion_ids = [item.dataset_id for item in EXCLUSIONS]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise MatrixError("duplicate selectable dataset id")
    if len(exclusion_ids) != len(set(exclusion_ids)) or set(dataset_ids) & set(
        exclusion_ids
    ):
        raise MatrixError("invalid exclusion ids")
    for dataset in DATASETS:
        if _SAFE_ID_RE.fullmatch(dataset.dataset_id) is None or not dataset.revision:
            raise MatrixError("dataset id/revision is not pinned")
        if not dataset.evaluation_only:
            raise MatrixError(f"{dataset.dataset_id} is not evaluation-only")
        if (
            len(dataset.usage_constraints) != len(set(dataset.usage_constraints))
            or UsageConstraint.EVALUATION_ONLY not in dataset.usage_constraints
            or UsageConstraint.PRIVATE_CACHE_ONLY not in dataset.usage_constraints
        ):
            raise MatrixError(f"{dataset.dataset_id} has incomplete usage constraints")
        if not dataset.license.acceptance_ids:
            raise MatrixError(f"{dataset.dataset_id} has no license gate")
        if dataset.license.redistribution != "cache-only":
            raise MatrixError(f"{dataset.dataset_id} may not leave the private cache")
        if "NC" in dataset.license.license_id.upper():
            raise MatrixError(f"{dataset.dataset_id} has a non-commercial license")
        if dataset.credential_env == MDC_TOKEN_ENV and not {
            UsageConstraint.NO_REIDENTIFICATION,
            UsageConstraint.NO_REHOSTING,
        } <= set(dataset.usage_constraints):
            raise MatrixError(f"{dataset.dataset_id} lacks MDC privacy constraints")
        selection = dataset.selection
        selection_fields = (
            *selection.identity_fields,
            *selection.strata_fields,
            *((selection.speaker_field,) if selection.speaker_field else ()),
        )
        if (
            not selection.description
            or _SAFE_ID_RE.fullmatch(selection.split) is None
            or len(selection.identity_fields) != len(set(selection.identity_fields))
            or len(selection.strata_fields) != len(set(selection.strata_fields))
            or any(_SAFE_ID_RE.fullmatch(field) is None for field in selection_fields)
            or (
                selection.expected_examples is not None
                and (
                    type(selection.expected_examples) is not int
                    or selection.expected_examples <= 0
                )
            )
        ):
            raise MatrixError(f"{dataset.dataset_id} has an invalid selection policy")
        if selection.speaker_disjoint and selection.speaker_field is None:
            raise MatrixError(f"{dataset.dataset_id} lacks a speaker identity field")
        if (
            selection.algorithm is SelectionAlgorithm.HUMAN_LABEL_MAP_REQUIRED
        ) != (
            UsageConstraint.HUMAN_LABEL_MAP_REQUIRED in dataset.usage_constraints
        ):
            raise MatrixError(
                f"{dataset.dataset_id} has an inconsistent human label-map gate"
            )
        if selection.algorithm in {
            SelectionAlgorithm.HASH_TOP_K,
            SelectionAlgorithm.HASH_STRATIFIED,
            SelectionAlgorithm.HASH_SPEAKER_STRATIFIED,
        } and (
            selection.seed is None
            or not selection.identity_fields
            or selection.expected_examples is None
            or (
                selection.algorithm
                in {
                    SelectionAlgorithm.HASH_STRATIFIED,
                    SelectionAlgorithm.HASH_SPEAKER_STRATIFIED,
                }
                and not selection.strata_fields
            )
            or (
                selection.algorithm is SelectionAlgorithm.HASH_SPEAKER_STRATIFIED
                and selection.speaker_field is None
            )
        ):
            raise MatrixError(f"{dataset.dataset_id} has an incomplete hash selector")
        if selection.algorithm is SelectionAlgorithm.FIXED_IDS and (
            not selection.identity_fields
            or not selection.fixed_ids
            or selection.expected_examples != len(selection.fixed_ids)
            or len(selection.fixed_ids) != len(set(selection.fixed_ids))
        ):
            raise MatrixError(f"{dataset.dataset_id} has an incomplete fixed selector")
        artifact_ids = [artifact.artifact_id for artifact in dataset.artifacts]
        if not artifact_ids or len(artifact_ids) != len(set(artifact_ids)):
            raise MatrixError(f"{dataset.dataset_id} has invalid artifact ids")
        for artifact in dataset.artifacts:
            if _SAFE_ID_RE.fullmatch(artifact.artifact_id) is None:
                raise MatrixError("unsafe artifact id")
            if artifact.filename is not None and (
                _SAFE_FILENAME_RE.fullmatch(artifact.filename) is None
                or Path(artifact.filename).name != artifact.filename
            ):
                raise MatrixError(f"unsafe artifact filename for {dataset.dataset_id}")
            parsed = urlparse(artifact.locator)
            if (
                parsed.scheme not in {"https", "mdc"}
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
            ):
                raise MatrixError(f"unsafe artifact locator for {dataset.dataset_id}")
            if (
                artifact.integrity
                in {
                    IntegrityKind.PINNED,
                    IntegrityKind.GIT_COMMIT,
                    IntegrityKind.GIT_COMMIT_AND_LFS_OIDS,
                }
                and artifact.checksum is None
            ):
                raise MatrixError(f"{dataset.dataset_id} lacks a pinned checksum")
            if artifact.checksum is not None:
                _validate_checksum(artifact.checksum)
                if (
                    artifact.integrity is IntegrityKind.MDC_API_SHA256
                    and artifact.checksum.algorithm is not ChecksumAlgorithm.SHA256
                ):
                    raise MatrixError(
                        f"{dataset.dataset_id} MDC catalog pin is not SHA-256"
                    )
    categories = [track.category for track in TRACKS]
    if set(categories) != set(TaskCategory) or len(categories) != len(set(categories)):
        raise MatrixError("task categories must appear exactly once")
    if len({track.track_id for track in TRACKS}) != len(TRACKS):
        raise MatrixError("duplicate evaluation track id")
    for track in TRACKS:
        if (
            not track.metrics
            or not set(track.metrics) <= _ALLOWED_METRICS[track.category]
        ):
            raise MatrixError(f"metric family leak in {track.track_id}")
        if not track.dataset_ids or not set(track.dataset_ids) <= set(dataset_ids):
            raise MatrixError(f"unknown/non-selectable dataset in {track.track_id}")
    if not all(item.selectable is False for item in EXCLUSIONS):
        raise MatrixError("excluded datasets must be non-selectable")
    if (
        UsageConstraint.NO_VOICE_CLONING
        not in dataset_by_id("common_voice_26_southern_american").usage_constraints
    ):
        raise MatrixError(
            "Southern American corpus lacks its voice-cloning restriction"
        )


validate_matrix()


__all__ = [
    "AcquisitionPlan",
    "Artifact",
    "Checksum",
    "ChecksumAlgorithm",
    "DATASETS",
    "Dataset",
    "EXCLUSIONS",
    "EvaluationTrack",
    "IntegrityKind",
    "MATRIX_VERSION",
    "MDC_TOKEN_ENV",
    "MatrixError",
    "Metric",
    "PlannedArtifact",
    "SELECTION_SEED",
    "SelectionAlgorithm",
    "SelectionPolicy",
    "TRACKS",
    "TaskCategory",
    "UsageConstraint",
    "build_acquisition_plan",
    "dataset_by_id",
    "default_corpus_root",
    "selection_policy_sha256",
    "selected_rows_sha256",
    "select_subset_rows",
    "validate_matrix",
    "verify_local_artifact",
]
