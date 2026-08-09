"""Strict, no-download Anyreach artifact and benchmark-selection contracts.

The committed lock binds public source bytes and 24 public synthetic row IDs.
This module never decodes Parquet, imports a candidate/runtime library, loads a
model, or maps Anyreach action scores into the inactive interruption policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import Final, Mapping

from tools.streaming_stt.bounded_io import (
    BoundedReadError,
    FileSnapshot,
    opened_directory_nofollow,
    read_regular_bounded,
)


CANDIDATE_ID: Final = "anyreach-semantic-turn-taking-q8-synthetic24-v1"
LOCK_KIND: Final = "anyreach-semantic-turn-source-lock-v1"
MANIFEST_KIND: Final = "anyreach-semantic-turn-artifact-manifest-v1"
ACCEPTANCE_ID: Final = "ANYREACH-HF-APACHE-2.0-METADATA"
DEFAULT_LOCK: Final = (
    Path(__file__).resolve().parent / "anyreach-q8-synthetic24-v1.lock.json"
)
LOCK_COPY_FILENAME: Final = DEFAULT_LOCK.name
BENCHMARK_COPY_FILENAME: Final = "synthetic-00000-of-00001.parquet"
MANIFEST_FILENAME: Final = "anyreach-artifact-manifest.json"
LOCK_RAW_SHA256: Final = (
    "d463fc45176da94c6923b2a2fc980a1e8cf5a0c70009eeb554341c6aff18892e"
)
LOCK_RECIPE_SHA256: Final = (
    "1062831086aeea841cd9c31d98807f0395b89e57edec72dd8434247c512b2c77"
)
LOCK_SIZE_BYTES: Final = 9_068
SELECTED_IDS_SHA256: Final = (
    "0d7cbea2954b0f5c4daafe83735a47f048a12e9e38d793de144e3bc234bf112c"
)
_MAX_LOCK_BYTES: Final = 64 * 1024
_MAX_MANIFEST_BYTES: Final = 64 * 1024
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_SHA1_RE: Final = re.compile(r"[0-9a-f]{40}\Z")
_SAFE_ID_RE: Final = re.compile(r"[a-z0-9][a-z0-9_.-]{0,95}\Z")


class AnyreachContractError(RuntimeError):
    """A source lock, artifact, selection, or manifest failed closed."""


class ArtifactRole(str, Enum):
    """Whether a frozen model leaf is executed or retained as metadata."""

    EXECUTION = "execution"
    METADATA_WITNESS = "metadata-witness"


class AnyreachAction(str, Enum):
    """Publisher action namespace; this is not a policy disposition."""

    START_SPEAKING = "start_speaking"
    CONTINUE_LISTENING = "continue_listening"
    START_LISTENING = "start_listening"
    CONTINUE_SPEAKING = "continue_speaking"


@dataclass(frozen=True)
class SourceArtifact:
    name: str
    role: ArtifactRole
    upstream_path: str
    filename: str
    size_bytes: int
    sha256: str
    git_blob_sha1: str
    storage: str


@dataclass(frozen=True)
class BenchmarkArtifact:
    upstream_path: str
    filename: str
    size_bytes: int
    sha256: str
    git_blob_sha1: str
    storage: str


@dataclass(frozen=True)
class BenchmarkSlot:
    ordinal: int
    source_row: int
    row_id: str
    action_index: int
    action: AnyreachAction


@dataclass(frozen=True)
class AnyreachSourceLock:
    path: Path = field(repr=False)
    raw_sha256: str
    raw_size_bytes: int
    recipe_sha256: str
    candidate_id: str
    acceptance_id: str
    model_repository: str
    model_revision: str
    model_artifacts: tuple[SourceArtifact, ...]
    model_action_token_ids: tuple[tuple[AnyreachAction, int], ...]
    predict_token_id: int
    benchmark_repository: str
    benchmark_revision: str
    benchmark_artifact: BenchmarkArtifact
    benchmark_class_labels: tuple[AnyreachAction, ...]
    source_rows: int
    selected_ids_sha256: str
    slots: tuple[BenchmarkSlot, ...]
    privacy: Mapping[str, bool] = field(repr=False)
    evidence_scope: Mapping[str, bool] = field(repr=False)


@dataclass(frozen=True)
class BoundArtifact:
    name: str
    role: str
    path: Path = field(repr=False)
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class AnyreachArtifactManifest:
    path: Path = field(repr=False)
    digest: str
    candidate_id: str
    source_lock: BoundArtifact
    artifacts: tuple[BoundArtifact, ...]
    artifact_set_sha256: str
    selected_rows: int
    selected_ids_sha256: str
    evidence_scope: Mapping[str, bool] = field(repr=False)

    @property
    def artifact_by_name(self) -> Mapping[str, BoundArtifact]:
        return MappingProxyType({item.name: item for item in self.artifacts})


_MODEL_REPOSITORY = "anyreach-ai/semantic-turn-taking"
_MODEL_REVISION = "e3a37384c4b19f1e7d29568a61634756e487b75e"
_BENCHMARK_REPOSITORY = "anyreach-ai/semantic-turn-taking-benchmark"
_BENCHMARK_REVISION = "a22ba86c174c55c8de685d233771fa018ebc8c99"
_MODEL_ARTIFACTS = (
    SourceArtifact(
        "model-q8",
        ArtifactRole.EXECUTION,
        "onnx/model_q8.onnx",
        "model_q8.onnx",
        495_715_962,
        "1974c790d2aa557642b0906eafbb593af8ca48fc045e5b9d187d7fa95faf88fe",
        "800720b30dc6cd25abf66ca915daaf6e788ccf7b",
        "git-lfs",
    ),
    SourceArtifact(
        "model-tokenizer",
        ArtifactRole.EXECUTION,
        "onnx/tokenizer.json",
        "tokenizer.json",
        11_422_970,
        "cb8544e622fff1dd549fddc8273bd5fd327f9642604868434f594f19ac2a49cc",
        "dfa3ab4005e44b3abb3d66036a3fea2d79d4730d",
        "git-lfs",
    ),
    SourceArtifact(
        "model-config",
        ArtifactRole.METADATA_WITNESS,
        "onnx/config.json",
        "config.json",
        1_253,
        "530a891bea69916e691b6a46ab3f176410719eaa4de1df62c2ad9489a8652f23",
        "5efa531960708d276b5c123918ea7866a4568c43",
        "git",
    ),
    SourceArtifact(
        "model-tokenizer-config",
        ArtifactRole.METADATA_WITNESS,
        "onnx/tokenizer_config.json",
        "tokenizer_config.json",
        5_459,
        "b16f88942f0039b4eae09591e56bfa557e4c008c13e80cd11cabbe2f81c0ca74",
        "ce9d09aa7d7c58778382efc83db3d258248a1a3c",
        "git",
    ),
)
_BENCHMARK_ARTIFACT = BenchmarkArtifact(
    "data/synthetic-00000-of-00001.parquet",
    BENCHMARK_COPY_FILENAME,
    17_788,
    "c289c3146c903dd157884c423739b14cfd3e3133bf4b7daa361010883d7857d5",
    "fc2c8a45af7e8356d328b09d8662d1227b54832f",
    "git-lfs",
)
_MODEL_ACTION_TOKEN_IDS = (
    (AnyreachAction.CONTINUE_LISTENING, 151_665),
    (AnyreachAction.START_SPEAKING, 151_666),
    (AnyreachAction.START_LISTENING, 151_667),
    (AnyreachAction.CONTINUE_SPEAKING, 151_668),
)
_BENCHMARK_CLASS_LABELS = (
    AnyreachAction.START_SPEAKING,
    AnyreachAction.CONTINUE_LISTENING,
    AnyreachAction.START_LISTENING,
    AnyreachAction.CONTINUE_SPEAKING,
)
_PRIVACY = MappingProxyType(
    {
        "public_row_ids_committed": True,
        "messages_committed": False,
        "conversation_text_committed": False,
        "local_paths_committed": False,
        "provision_outputs_private": True,
        "safe_stdout_aggregate_only": True,
    }
)
_EVIDENCE_SCOPE = MappingProxyType(
    {
        "source_bytes_bound": True,
        "benchmark_selection_declared": True,
        "parquet_decoded": False,
        "model_executed": False,
        "tokenizer_executed": False,
        "onnx_contract_verified": False,
        "runtime_bound": False,
        "action_scores_observed": False,
        "policy_score_mapping": False,
        "assistant_playback": False,
        "audio_or_device": False,
        "latency": False,
        "production_evidence": False,
        "runtime_authority": False,
        "promotion_authority": False,
    }
)


def _expected_slots() -> tuple[BenchmarkSlot, ...]:
    return tuple(
        BenchmarkSlot(
            ordinal=index,
            source_row=36 + index,
            row_id=(
                f"manual_sli_{index + 1:02d}"
                if index < 12
                else f"manual_cs_{index - 11:02d}"
            ),
            action_index=2 if index < 12 else 3,
            action=(
                AnyreachAction.START_LISTENING
                if index < 12
                else AnyreachAction.CONTINUE_SPEAKING
            ),
        )
        for index in range(24)
    )


_SLOTS = _expected_slots()


def _bad() -> object:
    raise AnyreachContractError()


def _strict_json(raw: bytes) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise AnyreachContractError()
            result[key] = value
        return result

    try:
        return json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda _value: _bad(),
        )
    except AnyreachContractError:
        raise
    except (OverflowError, RecursionError, UnicodeError, ValueError):
        raise AnyreachContractError() from None


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (OverflowError, TypeError, ValueError):
        raise AnyreachContractError() from None


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256(value: object) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise AnyreachContractError()
    return value


def _sha1(value: object) -> str:
    if type(value) is not str or _SHA1_RE.fullmatch(value) is None:
        raise AnyreachContractError()
    return value


def _positive_int(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise AnyreachContractError()
    return value


def _same_typed_json(value: object, expected: object) -> bool:
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(value) == set(expected) and all(  # type: ignore[arg-type]
            _same_typed_json(value[key], expected[key])
            for key in expected  # type: ignore[index]
        )
    if isinstance(expected, list):
        return len(value) == len(expected) and all(  # type: ignore[arg-type]
            _same_typed_json(left, right)
            for left, right in zip(value, expected, strict=True)  # type: ignore[arg-type]
        )
    return value == expected


def _artifact_from_json(value: object) -> SourceArtifact:
    if not isinstance(value, dict) or set(value) != {
        "name",
        "role",
        "upstream_path",
        "filename",
        "size_bytes",
        "sha256",
        "git_blob_sha1",
        "storage",
    }:
        raise AnyreachContractError()
    try:
        role = ArtifactRole(value.get("role"))
    except (TypeError, ValueError):
        raise AnyreachContractError() from None
    name = value.get("name")
    upstream_path = value.get("upstream_path")
    filename = value.get("filename")
    storage = value.get("storage")
    if (
        type(name) is not str
        or _SAFE_ID_RE.fullmatch(name) is None
        or type(upstream_path) is not str
        or not upstream_path.startswith("onnx/")
        or type(filename) is not str
        or Path(upstream_path).name != filename
        or type(storage) is not str
        or storage not in {"git", "git-lfs"}
    ):
        raise AnyreachContractError()
    return SourceArtifact(
        name=name,
        role=role,
        upstream_path=upstream_path,
        filename=filename,
        size_bytes=_positive_int(value.get("size_bytes")),
        sha256=_sha256(value.get("sha256")),
        git_blob_sha1=_sha1(value.get("git_blob_sha1")),
        storage=storage,
    )


def _slot_from_json(value: object) -> BenchmarkSlot:
    if not isinstance(value, dict) or set(value) != {
        "ordinal",
        "source_row",
        "row_id",
        "action_index",
        "action",
    }:
        raise AnyreachContractError()
    ordinal = value.get("ordinal")
    source_row = value.get("source_row")
    action_index = value.get("action_index")
    row_id = value.get("row_id")
    if (
        type(ordinal) is not int
        or ordinal < 0
        or type(source_row) is not int
        or source_row < 0
        or type(action_index) is not int
        or action_index < 0
        or type(row_id) is not str
        or _SAFE_ID_RE.fullmatch(row_id) is None
    ):
        raise AnyreachContractError()
    try:
        action = AnyreachAction(value.get("action"))
    except (TypeError, ValueError):
        raise AnyreachContractError() from None
    return BenchmarkSlot(ordinal, source_row, row_id, action_index, action)


def load_source_lock(path: Path | str = DEFAULT_LOCK) -> AnyreachSourceLock:
    """Load the exact committed or byte-identical retained source lock."""

    try:
        snapshot = read_regular_bounded(
            Path(path),
            maximum_bytes=_MAX_LOCK_BYTES,
            expected_bytes=LOCK_SIZE_BYTES,
        )
        raw_sha256 = hashlib.sha256(snapshot.data).hexdigest()
        if raw_sha256 != LOCK_RAW_SHA256:
            raise AnyreachContractError()
        root = _strict_json(snapshot.data)
        if not isinstance(root, dict) or set(root) != {
            "schema_version",
            "kind",
            "candidate_id",
            "recipe_digest_rule",
            "recipe_sha256",
            "license",
            "model",
            "benchmark",
            "privacy",
            "evidence_scope",
        }:
            raise AnyreachContractError()
        digest_value = dict(root)
        declared_recipe = _sha256(digest_value.pop("recipe_sha256", None))
        if (
            root.get("schema_version") != 1
            or type(root.get("schema_version")) is not int
            or root.get("kind") != LOCK_KIND
            or root.get("candidate_id") != CANDIDATE_ID
            or root.get("recipe_digest_rule")
            != "sha256-canonical-json-without-recipe_sha256-v1"
            or declared_recipe != LOCK_RECIPE_SHA256
            or _canonical_sha256(digest_value) != declared_recipe
        ):
            raise AnyreachContractError()

        license_value = root.get("license")
        expected_license = {
            "acceptance_id": ACCEPTANCE_ID,
            "asserted_spdx": "Apache-2.0",
            "provenance": "publisher-hugging-face-metadata",
            "bundled_license_file": False,
            "bundled_notice_file": False,
            "private_materialization_only": True,
            "rehosting_authority": False,
            "redistribution_authority": False,
            "assertions": [
                {
                    "repository_id": _MODEL_REPOSITORY,
                    "revision": _MODEL_REVISION,
                    "metadata_url": (
                        "https://huggingface.co/anyreach-ai/semantic-turn-taking/"
                        f"tree/{_MODEL_REVISION}"
                    ),
                },
                {
                    "repository_id": _BENCHMARK_REPOSITORY,
                    "revision": _BENCHMARK_REVISION,
                    "metadata_url": (
                        "https://huggingface.co/datasets/anyreach-ai/"
                        "semantic-turn-taking-benchmark/tree/"
                        f"{_BENCHMARK_REVISION}"
                    ),
                },
            ],
        }
        if not _same_typed_json(license_value, expected_license):
            raise AnyreachContractError()

        model = root.get("model")
        if not isinstance(model, dict) or set(model) != {
            "repository_id",
            "revision",
            "canonical_url",
            "format",
            "quantization",
            "curated_root_policy",
            "artifacts",
            "published_interface",
        }:
            raise AnyreachContractError()
        raw_artifacts = model.get("artifacts")
        if not isinstance(raw_artifacts, list):
            raise AnyreachContractError()
        model_artifacts = tuple(_artifact_from_json(item) for item in raw_artifacts)
        interface = model.get("published_interface")
        expected_interface = {
            "prompt_protocol": "qwen-chatml-ending-predict-v1",
            "maximum_input_tokens": 1024,
            "truncation_side": "left",
            "predict_token_id": 151669,
            "action_token_ids": [
                {"action": action.value, "token_id": token_id}
                for action, token_id in _MODEL_ACTION_TOKEN_IDS
            ],
            "execution_adapter_bound": False,
        }
        if (
            model.get("repository_id") != _MODEL_REPOSITORY
            or model.get("revision") != _MODEL_REVISION
            or model.get("canonical_url")
            != "https://huggingface.co/anyreach-ai/semantic-turn-taking"
            or model.get("format") != "onnx"
            or model.get("quantization") != "dynamic-int8-q8"
            or model.get("curated_root_policy") != "exact-four-files-no-extras-v1"
            or model_artifacts != _MODEL_ARTIFACTS
            or not _same_typed_json(interface, expected_interface)
        ):
            raise AnyreachContractError()

        benchmark = root.get("benchmark")
        if not isinstance(benchmark, dict) or set(benchmark) != {
            "repository_id",
            "revision",
            "canonical_url",
            "artifact",
            "schema",
            "class_labels",
            "source_rows",
            "selection",
        }:
            raise AnyreachContractError()
        artifact_value = benchmark.get("artifact")
        if not isinstance(artifact_value, dict) or set(artifact_value) != {
            "upstream_path",
            "filename",
            "size_bytes",
            "sha256",
            "git_blob_sha1",
            "storage",
        }:
            raise AnyreachContractError()
        benchmark_artifact = BenchmarkArtifact(
            upstream_path=str(artifact_value.get("upstream_path")),
            filename=str(artifact_value.get("filename")),
            size_bytes=_positive_int(artifact_value.get("size_bytes")),
            sha256=_sha256(artifact_value.get("sha256")),
            git_blob_sha1=_sha1(artifact_value.get("git_blob_sha1")),
            storage=str(artifact_value.get("storage")),
        )
        expected_schema = [
            {"name": "id", "type": "string"},
            {"name": "source", "type": "string"},
            {
                "name": "messages",
                "type": "list<struct<role:string,content:string>>",
            },
            {"name": "conversation", "type": "string"},
            {"name": "action", "type": "class-label"},
            {"name": "original_label", "type": "string"},
            {"name": "num_turns", "type": "int32"},
        ]
        expected_labels = [
            {"index": 0, "action": "start_speaking", "rows": 24},
            {"index": 1, "action": "continue_listening", "rows": 12},
            {"index": 2, "action": "start_listening", "rows": 12},
            {"index": 3, "action": "continue_speaking", "rows": 12},
        ]
        selection = benchmark.get("selection")
        if not isinstance(selection, dict) or set(selection) != {
            "algorithm",
            "physical_row_index_base",
            "physical_row_start",
            "physical_row_end_exclusive",
            "selected_rows",
            "selected_id_digest_rule",
            "selected_ids_sha256",
            "required_source",
            "required_num_turns",
            "required_original_label",
            "slots",
        }:
            raise AnyreachContractError()
        raw_slots = selection.get("slots")
        if not isinstance(raw_slots, list):
            raise AnyreachContractError()
        slots = tuple(_slot_from_json(item) for item in raw_slots)
        id_digest = hashlib.sha256(
            "".join(f"{slot.row_id}\n" for slot in slots).encode("utf-8")
        ).hexdigest()
        if (
            benchmark.get("repository_id") != _BENCHMARK_REPOSITORY
            or benchmark.get("revision") != _BENCHMARK_REVISION
            or benchmark.get("canonical_url")
            != "https://huggingface.co/datasets/anyreach-ai/semantic-turn-taking-benchmark"
            or benchmark_artifact != _BENCHMARK_ARTIFACT
            or not _same_typed_json(benchmark.get("schema"), expected_schema)
            or not _same_typed_json(benchmark.get("class_labels"), expected_labels)
            or type(benchmark.get("source_rows")) is not int
            or benchmark.get("source_rows") != 60
            or selection.get("algorithm") != "physical-order-action-filter-v1"
            or type(selection.get("physical_row_index_base")) is not int
            or selection.get("physical_row_index_base") != 0
            or selection.get("physical_row_start") != 36
            or selection.get("physical_row_end_exclusive") != 60
            or selection.get("selected_rows") != 24
            or selection.get("selected_id_digest_rule")
            != "sha256-utf8-id-plus-lf-in-physical-order-v1"
            or selection.get("selected_ids_sha256") != SELECTED_IDS_SHA256
            or selection.get("required_source") != "synthetic"
            or selection.get("required_num_turns") != 3
            or selection.get("required_original_label") != "equals-action-label"
            or slots != _SLOTS
            or id_digest != SELECTED_IDS_SHA256
        ):
            raise AnyreachContractError()
        if not _same_typed_json(
            root.get("privacy"), dict(_PRIVACY)
        ) or not _same_typed_json(root.get("evidence_scope"), dict(_EVIDENCE_SCOPE)):
            raise AnyreachContractError()
        return AnyreachSourceLock(
            path=snapshot.path,
            raw_sha256=raw_sha256,
            raw_size_bytes=len(snapshot.data),
            recipe_sha256=declared_recipe,
            candidate_id=CANDIDATE_ID,
            acceptance_id=ACCEPTANCE_ID,
            model_repository=_MODEL_REPOSITORY,
            model_revision=_MODEL_REVISION,
            model_artifacts=model_artifacts,
            model_action_token_ids=_MODEL_ACTION_TOKEN_IDS,
            predict_token_id=151_669,
            benchmark_repository=_BENCHMARK_REPOSITORY,
            benchmark_revision=_BENCHMARK_REVISION,
            benchmark_artifact=benchmark_artifact,
            benchmark_class_labels=_BENCHMARK_CLASS_LABELS,
            source_rows=60,
            selected_ids_sha256=SELECTED_IDS_SHA256,
            slots=slots,
            privacy=_PRIVACY,
            evidence_scope=_EVIDENCE_SCOPE,
        )
    except AnyreachContractError:
        raise
    except (BoundedReadError, OSError, RuntimeError, TypeError, ValueError):
        raise AnyreachContractError() from None


def benchmark_slots(lock: AnyreachSourceLock) -> tuple[BenchmarkSlot, ...]:
    """Return the locked public IDs/actions without decoding conversation text."""

    if type(lock) is not AnyreachSourceLock or lock.slots != _SLOTS:
        raise AnyreachContractError()
    return lock.slots


def _absolute(path: Path | str) -> Path:
    try:
        supplied = Path(path).expanduser()
        candidate = Path(os.path.abspath(supplied))
    except (OSError, RuntimeError, TypeError, ValueError):
        raise AnyreachContractError() from None
    if not supplied.is_absolute() or candidate != supplied:
        raise AnyreachContractError()
    return candidate


def _has_git_ancestor(path: Path) -> bool:
    try:
        current = path.resolve(strict=True)
        while True:
            try:
                (current / ".git").lstat()
            except FileNotFoundError:
                pass
            else:
                return True
            if current.parent == current:
                return False
            current = current.parent
    except (OSError, RuntimeError, ValueError):
        raise AnyreachContractError() from None


def _directory_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        stat.S_IMODE(metadata.st_mode),
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _private_directory(
    path: Path | str,
) -> tuple[Path, tuple[str, ...], tuple[int, ...]]:
    candidate = _absolute(path)
    try:
        with opened_directory_nofollow(candidate, require_private=True) as (
            stable,
            descriptor,
        ):
            metadata = os.fstat(descriptor)
            names = tuple(sorted(os.listdir(descriptor)))
            current = candidate.lstat()
        if (
            stat.S_IMODE(metadata.st_mode) != 0o700
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
            or _directory_identity(current) != _directory_identity(metadata)
            or any(
                not isinstance(name, str)
                or not name
                or name in {".", ".."}
                or "/" in name
                or "\x00" in name
                for name in names
            )
        ):
            raise AnyreachContractError()
        return stable, names, _directory_identity(metadata)
    except AnyreachContractError:
        raise
    except (BoundedReadError, OSError, RuntimeError, ValueError):
        raise AnyreachContractError() from None


def _private_file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        stat.S_IMODE(metadata.st_mode),
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


@dataclass(frozen=True)
class _PrivateFileRead:
    path: Path = field(repr=False)
    sha256: str
    size_bytes: int
    identity: tuple[int, ...] = field(repr=False)
    data: bytes | None = field(default=None, repr=False)


def _read_private_regular(
    path: Path,
    *,
    maximum_bytes: int,
    expected_bytes: int | None = None,
    retain_bytes: bool,
) -> _PrivateFileRead:
    if (
        type(maximum_bytes) is not int
        or maximum_bytes <= 0
        or (
            expected_bytes is not None
            and (
                type(expected_bytes) is not int
                or expected_bytes <= 0
                or expected_bytes > maximum_bytes
            )
        )
    ):
        raise AnyreachContractError()
    candidate = _absolute(path)
    descriptor = -1
    try:
        with opened_directory_nofollow(
            candidate.parent,
            require_private=True,
        ) as (parent, parent_descriptor):
            if parent != candidate.parent:
                raise AnyreachContractError()
            parent_before = os.fstat(parent_descriptor)
            before = os.stat(
                candidate.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            selected_bytes = (
                before.st_size if expected_bytes is None else expected_bytes
            )
            if (
                not _private_regular_metadata(before, size_bytes=selected_bytes)
                or selected_bytes > maximum_bytes
            ):
                raise AnyreachContractError()
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(candidate.name, flags, dir_fd=parent_descriptor)
            opened = os.fstat(descriptor)
            if _private_file_identity(opened) != _private_file_identity(before):
                raise AnyreachContractError()
            digest = hashlib.sha256()
            chunks: list[bytes] | None = [] if retain_bytes else None
            consumed = 0
            while consumed <= selected_bytes:
                chunk = os.read(
                    descriptor,
                    min(1024 * 1024, selected_bytes + 1 - consumed),
                )
                if not chunk:
                    break
                consumed += len(chunk)
                if consumed > selected_bytes:
                    raise AnyreachContractError()
                digest.update(chunk)
                if chunks is not None:
                    chunks.append(chunk)
            after = os.fstat(descriptor)
            named = os.stat(
                candidate.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            parent_after = os.fstat(parent_descriptor)
            if (
                consumed != selected_bytes
                or _private_file_identity(after) != _private_file_identity(opened)
                or _private_file_identity(named) != _private_file_identity(opened)
                or _directory_identity(parent_after)
                != _directory_identity(parent_before)
            ):
                raise AnyreachContractError()
        resolved = candidate.resolve(strict=True)
        current = candidate.lstat()
        resolved_metadata = resolved.lstat()
        if (
            resolved != candidate
            or _private_file_identity(current) != _private_file_identity(opened)
            or _private_file_identity(resolved_metadata)
            != _private_file_identity(opened)
        ):
            raise AnyreachContractError()
        return _PrivateFileRead(
            path=resolved,
            sha256=digest.hexdigest(),
            size_bytes=consumed,
            identity=_private_file_identity(opened),
            data=b"".join(chunks) if chunks is not None else None,
        )
    except AnyreachContractError:
        raise
    except (BoundedReadError, OSError, OverflowError, RuntimeError, ValueError):
        raise AnyreachContractError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _private_regular_metadata(
    metadata: os.stat_result,
    *,
    size_bytes: int,
) -> bool:
    return (
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_nlink == 1
        and stat.S_IMODE(metadata.st_mode) == 0o600
        and metadata.st_size == size_bytes
        and (not hasattr(os, "geteuid") or metadata.st_uid == os.geteuid())
    )


def _verify_private_leaf_identities(
    path: Path,
    *,
    expected_names: tuple[str, ...],
    expected_identities: Mapping[str, tuple[int, ...]],
    expected_directory_identity: tuple[int, ...],
) -> None:
    """Rebind one private directory and its named leaves in one dirfd scan."""

    if tuple(sorted(expected_names)) != expected_names or set(
        expected_identities
    ) - set(expected_names):
        raise AnyreachContractError()
    candidate = _absolute(path)
    try:
        with opened_directory_nofollow(
            candidate,
            require_private=True,
        ) as (stable, descriptor):
            before = os.fstat(descriptor)
            names_before = tuple(sorted(os.listdir(descriptor)))
            identities = {
                name: _private_file_identity(
                    os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                )
                for name in expected_identities
            }
            names_after = tuple(sorted(os.listdir(descriptor)))
            after = os.fstat(descriptor)
            current = candidate.lstat()
        if (
            stable != candidate
            or names_before != expected_names
            or names_after != expected_names
            or identities != dict(expected_identities)
            or _directory_identity(before) != expected_directory_identity
            or _directory_identity(after) != expected_directory_identity
            or _directory_identity(current) != expected_directory_identity
        ):
            raise AnyreachContractError()
    except AnyreachContractError:
        raise
    except (BoundedReadError, OSError, RuntimeError, ValueError):
        raise AnyreachContractError() from None


def _bind_exact_file(
    path: Path,
    *,
    name: str,
    role: str,
    size_bytes: int,
    sha256: str,
) -> BoundArtifact:
    digest = _read_private_regular(
        path,
        maximum_bytes=size_bytes,
        expected_bytes=size_bytes,
        retain_bytes=False,
    )
    if digest.sha256 != sha256:
        raise AnyreachContractError()
    return BoundArtifact(name, role, digest.path, sha256, size_bytes)


@dataclass(frozen=True)
class _BoundModelRoot:
    root: Path = field(repr=False)
    root_names: tuple[str, ...]
    root_identity: tuple[int, ...] = field(repr=False)
    artifact_root: Path = field(repr=False)
    artifact_names: tuple[str, ...]
    artifact_root_identity: tuple[int, ...] = field(repr=False)
    leaf_identities: tuple[tuple[str, tuple[int, ...]], ...] = field(repr=False)
    artifacts: tuple[BoundArtifact, ...]


def _verify_bound_model_root(state: _BoundModelRoot) -> None:
    if type(state) is not _BoundModelRoot:
        raise AnyreachContractError()
    final_root, final_root_names, final_root_identity = _private_directory(state.root)
    if (
        final_root != state.root
        or final_root_names != state.root_names
        or final_root_identity != state.root_identity
    ):
        raise AnyreachContractError()
    _verify_private_leaf_identities(
        state.artifact_root,
        expected_names=state.artifact_names,
        expected_identities=dict(state.leaf_identities),
        expected_directory_identity=state.artifact_root_identity,
    )


def _bind_model_root_state(
    path: Path | str,
    lock: AnyreachSourceLock,
) -> _BoundModelRoot:
    if type(lock) is not AnyreachSourceLock:
        raise AnyreachContractError()
    root, root_names, root_identity = _private_directory(path)
    if _has_git_ancestor(root):
        raise AnyreachContractError()
    if root_names != ("onnx",):
        raise AnyreachContractError()
    artifact_root, names, artifact_root_identity = _private_directory(root / "onnx")
    expected_names = tuple(item.filename for item in lock.model_artifacts)
    if len(names) != len(expected_names) or set(names) != set(expected_names):
        raise AnyreachContractError()
    reads = tuple(
        _read_private_regular(
            root / item.upstream_path,
            maximum_bytes=item.size_bytes,
            expected_bytes=item.size_bytes,
            retain_bytes=False,
        )
        for item in lock.model_artifacts
    )
    if any(
        read.sha256 != item.sha256
        for read, item in zip(reads, lock.model_artifacts, strict=True)
    ):
        raise AnyreachContractError()
    bound = tuple(
        BoundArtifact(
            item.name,
            item.role.value,
            read.path,
            item.sha256,
            item.size_bytes,
        )
        for read, item in zip(reads, lock.model_artifacts, strict=True)
    )
    state = _BoundModelRoot(
        root=root,
        root_names=root_names,
        root_identity=root_identity,
        artifact_root=artifact_root,
        artifact_names=names,
        artifact_root_identity=artifact_root_identity,
        leaf_identities=tuple(
            (item.filename, read.identity)
            for item, read in zip(lock.model_artifacts, reads, strict=True)
        ),
        artifacts=bound,
    )
    _verify_bound_model_root(state)
    return state


def _bind_model_root(
    path: Path | str,
    lock: AnyreachSourceLock,
) -> tuple[BoundArtifact, ...]:
    return _bind_model_root_state(path, lock).artifacts


def _snapshot_benchmark_source(
    path: Path | str,
    lock: AnyreachSourceLock,
) -> FileSnapshot:
    if type(lock) is not AnyreachSourceLock:
        raise AnyreachContractError()
    candidate = _absolute(path)
    parent, names, parent_identity = _private_directory(candidate.parent)
    if parent != candidate.parent or _has_git_ancestor(parent):
        raise AnyreachContractError()
    expected = lock.benchmark_artifact
    private = _read_private_regular(
        candidate,
        maximum_bytes=expected.size_bytes,
        expected_bytes=expected.size_bytes,
        retain_bytes=True,
    )
    if private.sha256 != expected.sha256 or private.data is None:
        raise AnyreachContractError()
    _verify_private_leaf_identities(
        parent,
        expected_names=names,
        expected_identities={candidate.name: private.identity},
        expected_directory_identity=parent_identity,
    )
    return FileSnapshot(path=private.path, data=private.data)


def _declared_binding(value: object) -> tuple[str, str, Path, str, int]:
    if not isinstance(value, dict) or set(value) != {
        "name",
        "role",
        "path",
        "sha256",
        "size_bytes",
    }:
        raise AnyreachContractError()
    name = value.get("name")
    role = value.get("role")
    raw_path = value.get("path")
    if (
        type(name) is not str
        or _SAFE_ID_RE.fullmatch(name) is None
        or type(role) is not str
        or not role
        or type(raw_path) is not str
        or not raw_path
        or "\x00" in raw_path
    ):
        raise AnyreachContractError()
    return (
        name,
        role,
        _absolute(raw_path),
        _sha256(value.get("sha256")),
        _positive_int(value.get("size_bytes")),
    )


def artifact_set_sha256(
    source_lock: BoundArtifact,
    artifacts: tuple[BoundArtifact, ...],
) -> str:
    """Return a relocatable digest over names, roles, sizes, and bytes."""

    if type(source_lock) is not BoundArtifact or any(
        type(item) is not BoundArtifact for item in artifacts
    ):
        raise AnyreachContractError()
    rows = [
        {
            "name": item.name,
            "role": item.role,
            "sha256": item.sha256,
            "size_bytes": item.size_bytes,
        }
        for item in (source_lock, *artifacts)
    ]
    return _canonical_sha256(rows)


def load_artifact_manifest(path: Path | str) -> AnyreachArtifactManifest:
    """Load and close-time rehash one private, non-executable provision."""

    try:
        candidate = _absolute(path)
        output, output_names, output_identity = _private_directory(candidate.parent)
        if (
            candidate.parent != output
            or candidate.name != MANIFEST_FILENAME
            or _has_git_ancestor(output)
            or set(output_names)
            != {MANIFEST_FILENAME, LOCK_COPY_FILENAME, BENCHMARK_COPY_FILENAME}
        ):
            raise AnyreachContractError()
        manifest_read = _read_private_regular(
            candidate,
            maximum_bytes=_MAX_MANIFEST_BYTES,
            retain_bytes=True,
        )
        if manifest_read.data is None:
            raise AnyreachContractError()
        value = _strict_json(manifest_read.data)
        if manifest_read.data != _canonical_json(value) + b"\n":
            raise AnyreachContractError()
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "kind",
            "candidate_id",
            "source_lock",
            "artifacts",
            "benchmark_selection",
            "artifact_set_sha256",
            "manifest_self_sha256",
            "evidence_scope",
        }:
            raise AnyreachContractError()
        if (
            type(value.get("schema_version")) is not int
            or value.get("schema_version") != 1
            or value.get("kind") != MANIFEST_KIND
            or value.get("candidate_id") != CANDIDATE_ID
        ):
            raise AnyreachContractError()
        self_value = dict(value)
        declared_self = _sha256(self_value.pop("manifest_self_sha256", None))
        if _canonical_sha256(self_value) != declared_self:
            raise AnyreachContractError()
        source_declared = _declared_binding(value.get("source_lock"))
        if (
            source_declared[0] != "source-lock"
            or source_declared[1] != "source-lock"
            or source_declared[2] != output / LOCK_COPY_FILENAME
            or source_declared[3] != LOCK_RAW_SHA256
            or source_declared[4] != LOCK_SIZE_BYTES
        ):
            raise AnyreachContractError()
        source_read = _read_private_regular(
            source_declared[2],
            maximum_bytes=LOCK_SIZE_BYTES,
            expected_bytes=LOCK_SIZE_BYTES,
            retain_bytes=False,
        )
        if source_read.sha256 != LOCK_RAW_SHA256:
            raise AnyreachContractError()
        source_bound = BoundArtifact(
            "source-lock",
            "source-lock",
            source_read.path,
            LOCK_RAW_SHA256,
            LOCK_SIZE_BYTES,
        )
        lock = load_source_lock(source_bound.path)

        raw_artifacts = value.get("artifacts")
        if not isinstance(raw_artifacts, list) or len(raw_artifacts) != 5:
            raise AnyreachContractError()
        declared = tuple(_declared_binding(item) for item in raw_artifacts)
        expected_rows = tuple(
            (item.name, item.role.value, item.filename, item.sha256, item.size_bytes)
            for item in lock.model_artifacts
        ) + (
            (
                "benchmark-parquet",
                "benchmark-source",
                BENCHMARK_COPY_FILENAME,
                lock.benchmark_artifact.sha256,
                lock.benchmark_artifact.size_bytes,
            ),
        )
        if (
            tuple(
                (name, role, path_item.name, sha, size)
                for name, role, path_item, sha, size in declared
            )
            != expected_rows
        ):
            raise AnyreachContractError()
        model_artifact_parents = {item[2].parent for item in declared[:4]}
        if len(model_artifact_parents) != 1:
            raise AnyreachContractError()
        model_artifact_root = next(iter(model_artifact_parents))
        if model_artifact_root.name != "onnx":
            raise AnyreachContractError()
        model_state = _bind_model_root_state(model_artifact_root.parent, lock)
        model_bound = model_state.artifacts
        if (
            tuple(
                (item.name, item.role, item.path, item.sha256, item.size_bytes)
                for item in model_bound
            )
            != declared[:4]
        ):
            raise AnyreachContractError()
        benchmark_declared = declared[4]
        if benchmark_declared[2] != output / BENCHMARK_COPY_FILENAME:
            raise AnyreachContractError()
        benchmark_read = _read_private_regular(
            benchmark_declared[2],
            maximum_bytes=lock.benchmark_artifact.size_bytes,
            expected_bytes=lock.benchmark_artifact.size_bytes,
            retain_bytes=False,
        )
        if benchmark_read.sha256 != lock.benchmark_artifact.sha256:
            raise AnyreachContractError()
        benchmark_bound = BoundArtifact(
            "benchmark-parquet",
            "benchmark-source",
            benchmark_read.path,
            lock.benchmark_artifact.sha256,
            lock.benchmark_artifact.size_bytes,
        )
        artifacts = (*model_bound, benchmark_bound)
        selection = value.get("benchmark_selection")
        if not _same_typed_json(
            selection,
            {
                "source_rows": 60,
                "selected_rows": 24,
                "selected_ids_sha256": SELECTED_IDS_SHA256,
                "parquet_decoded": False,
            },
        ) or not _same_typed_json(value.get("evidence_scope"), dict(_EVIDENCE_SCOPE)):
            raise AnyreachContractError()
        calculated_set = artifact_set_sha256(source_bound, artifacts)
        declared_set = _sha256(value.get("artifact_set_sha256"))
        if calculated_set != declared_set:
            raise AnyreachContractError()
        _verify_private_leaf_identities(
            output,
            expected_names=output_names,
            expected_identities={
                MANIFEST_FILENAME: manifest_read.identity,
                LOCK_COPY_FILENAME: source_read.identity,
                BENCHMARK_COPY_FILENAME: benchmark_read.identity,
            },
            expected_directory_identity=output_identity,
        )
        _verify_bound_model_root(model_state)
        return AnyreachArtifactManifest(
            path=manifest_read.path,
            digest=manifest_read.sha256,
            candidate_id=CANDIDATE_ID,
            source_lock=source_bound,
            artifacts=artifacts,
            artifact_set_sha256=calculated_set,
            selected_rows=24,
            selected_ids_sha256=SELECTED_IDS_SHA256,
            evidence_scope=_EVIDENCE_SCOPE,
        )
    except AnyreachContractError:
        raise
    except (BoundedReadError, OSError, RuntimeError, TypeError, ValueError):
        raise AnyreachContractError() from None


__all__ = [
    "ACCEPTANCE_ID",
    "BENCHMARK_COPY_FILENAME",
    "CANDIDATE_ID",
    "DEFAULT_LOCK",
    "LOCK_COPY_FILENAME",
    "LOCK_KIND",
    "LOCK_RAW_SHA256",
    "LOCK_RECIPE_SHA256",
    "LOCK_SIZE_BYTES",
    "MANIFEST_FILENAME",
    "MANIFEST_KIND",
    "SELECTED_IDS_SHA256",
    "AnyreachAction",
    "AnyreachArtifactManifest",
    "AnyreachContractError",
    "AnyreachSourceLock",
    "ArtifactRole",
    "BenchmarkArtifact",
    "BenchmarkSlot",
    "BoundArtifact",
    "SourceArtifact",
    "artifact_set_sha256",
    "benchmark_slots",
    "load_artifact_manifest",
    "load_source_lock",
]
