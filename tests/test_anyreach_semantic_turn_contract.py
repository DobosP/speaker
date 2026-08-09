"""Exact public-source and private-manifest contracts for Anyreach."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path

import pytest

from tools.semantic_interruption import anyreach


def test_source_lock_has_exact_raw_and_self_digests() -> None:
    raw = anyreach.DEFAULT_LOCK.read_bytes()
    value = json.loads(raw)
    digest_value = dict(value)
    declared = digest_value.pop("recipe_sha256")

    assert len(raw) == anyreach.LOCK_SIZE_BYTES == 9_068
    assert hashlib.sha256(raw).hexdigest() == anyreach.LOCK_RAW_SHA256
    assert declared == anyreach.LOCK_RECIPE_SHA256
    assert anyreach._canonical_sha256(digest_value) == declared
    assert value["recipe_digest_rule"] == (
        "sha256-canonical-json-without-recipe_sha256-v1"
    )


def test_source_lock_loads_exact_model_closure_and_metadata_roles() -> None:
    lock = anyreach.load_source_lock()

    assert lock.candidate_id == anyreach.CANDIDATE_ID
    assert lock.model_repository == "anyreach-ai/semantic-turn-taking"
    assert lock.model_revision == "e3a37384c4b19f1e7d29568a61634756e487b75e"
    assert [item.upstream_path for item in lock.model_artifacts] == [
        "onnx/model_q8.onnx",
        "onnx/tokenizer.json",
        "onnx/config.json",
        "onnx/tokenizer_config.json",
    ]
    assert [item.role for item in lock.model_artifacts] == [
        anyreach.ArtifactRole.EXECUTION,
        anyreach.ArtifactRole.EXECUTION,
        anyreach.ArtifactRole.METADATA_WITNESS,
        anyreach.ArtifactRole.METADATA_WITNESS,
    ]
    assert [item.size_bytes for item in lock.model_artifacts] == [
        495_715_962,
        11_422_970,
        1_253,
        5_459,
    ]


def test_source_lock_uses_metadata_assertion_without_distribution_authority() -> None:
    value = json.loads(anyreach.DEFAULT_LOCK.read_bytes())
    license_value = value["license"]

    assert anyreach.load_source_lock().acceptance_id == (
        "ANYREACH-HF-APACHE-2.0-METADATA"
    )
    assert license_value == {
        "acceptance_id": anyreach.ACCEPTANCE_ID,
        "asserted_spdx": "Apache-2.0",
        "provenance": "publisher-hugging-face-metadata",
        "bundled_license_file": False,
        "bundled_notice_file": False,
        "private_materialization_only": True,
        "rehosting_authority": False,
        "redistribution_authority": False,
        "assertions": [
            {
                "repository_id": "anyreach-ai/semantic-turn-taking",
                "revision": "e3a37384c4b19f1e7d29568a61634756e487b75e",
                "metadata_url": (
                    "https://huggingface.co/anyreach-ai/semantic-turn-taking/"
                    "tree/e3a37384c4b19f1e7d29568a61634756e487b75e"
                ),
            },
            {
                "repository_id": "anyreach-ai/semantic-turn-taking-benchmark",
                "revision": "a22ba86c174c55c8de685d233771fa018ebc8c99",
                "metadata_url": (
                    "https://huggingface.co/datasets/anyreach-ai/"
                    "semantic-turn-taking-benchmark/tree/"
                    "a22ba86c174c55c8de685d233771fa018ebc8c99"
                ),
            },
        ],
    }


def test_benchmark_pin_schema_and_selection_are_exact() -> None:
    lock = anyreach.load_source_lock()
    artifact = lock.benchmark_artifact
    slots = anyreach.benchmark_slots(lock)

    assert lock.benchmark_repository == ("anyreach-ai/semantic-turn-taking-benchmark")
    assert lock.benchmark_revision == "a22ba86c174c55c8de685d233771fa018ebc8c99"
    assert artifact.upstream_path == "data/synthetic-00000-of-00001.parquet"
    assert artifact.size_bytes == 17_788
    assert artifact.sha256 == (
        "c289c3146c903dd157884c423739b14cfd3e3133bf4b7daa361010883d7857d5"
    )
    assert lock.source_rows == 60
    assert len(slots) == 24
    assert [slot.source_row for slot in slots] == list(range(36, 60))
    assert [slot.ordinal for slot in slots] == list(range(24))
    assert [slot.row_id for slot in slots] == [
        *(f"manual_sli_{index:02d}" for index in range(1, 13)),
        *(f"manual_cs_{index:02d}" for index in range(1, 13)),
    ]
    assert [slot.action_index for slot in slots] == [2] * 12 + [3] * 12
    assert [slot.action for slot in slots] == [
        anyreach.AnyreachAction.START_LISTENING
    ] * 12 + [anyreach.AnyreachAction.CONTINUE_SPEAKING] * 12


def test_selected_id_digest_has_one_explicit_domain() -> None:
    lock = anyreach.load_source_lock()
    raw = "".join(f"{slot.row_id}\n" for slot in lock.slots).encode("utf-8")

    assert hashlib.sha256(raw).hexdigest() == anyreach.SELECTED_IDS_SHA256
    assert lock.selected_ids_sha256 == anyreach.SELECTED_IDS_SHA256


def test_model_action_tokens_and_benchmark_indices_are_not_conflated() -> None:
    lock = anyreach.load_source_lock()

    assert lock.model_action_token_ids == (
        (anyreach.AnyreachAction.CONTINUE_LISTENING, 151_665),
        (anyreach.AnyreachAction.START_SPEAKING, 151_666),
        (anyreach.AnyreachAction.START_LISTENING, 151_667),
        (anyreach.AnyreachAction.CONTINUE_SPEAKING, 151_668),
    )
    assert lock.benchmark_class_labels == (
        anyreach.AnyreachAction.START_SPEAKING,
        anyreach.AnyreachAction.CONTINUE_LISTENING,
        anyreach.AnyreachAction.START_LISTENING,
        anyreach.AnyreachAction.CONTINUE_SPEAKING,
    )
    assert tuple(action for action, _token in lock.model_action_token_ids) != (
        lock.benchmark_class_labels
    )
    assert lock.predict_token_id == 151_669


def test_lock_commits_no_message_or_conversation_content() -> None:
    lock = anyreach.load_source_lock()

    assert dict(lock.privacy) == {
        "public_row_ids_committed": True,
        "messages_committed": False,
        "conversation_text_committed": False,
        "local_paths_committed": False,
        "provision_outputs_private": True,
        "safe_stdout_aggregate_only": True,
    }
    assert all(
        set(field.name for field in fields(type(slot)))
        == {
            "ordinal",
            "source_row",
            "row_id",
            "action_index",
            "action",
        }
        for slot in lock.slots
    )
    assert str(anyreach.DEFAULT_LOCK) not in repr(lock)


def test_lock_evidence_scope_is_closed_and_non_authoritative() -> None:
    scope = dict(anyreach.load_source_lock().evidence_scope)

    assert scope == {
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


def test_byte_identical_relocated_lock_loads(tmp_path: Path) -> None:
    retained = tmp_path / anyreach.LOCK_COPY_FILENAME
    retained.write_bytes(anyreach.DEFAULT_LOCK.read_bytes())

    loaded = anyreach.load_source_lock(retained)

    assert loaded.path == retained.resolve(strict=True)
    assert loaded.raw_sha256 == anyreach.LOCK_RAW_SHA256


def test_changed_lock_fails_even_when_json_remains_valid(tmp_path: Path) -> None:
    value = json.loads(anyreach.DEFAULT_LOCK.read_bytes())
    value["model"]["quantization"] = "other"
    changed = tmp_path / anyreach.LOCK_COPY_FILENAME
    changed.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(anyreach.AnyreachContractError):
        anyreach.load_source_lock(changed)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":2}',
        b'{"value":NaN}',
        b'{"value":Infinity}',
        b"\xff",
    ],
)
def test_strict_json_rejects_duplicates_constants_and_non_utf8(raw: bytes) -> None:
    with pytest.raises(anyreach.AnyreachContractError):
        anyreach._strict_json(raw)


def test_benchmark_slots_requires_the_exact_typed_lock() -> None:
    with pytest.raises(anyreach.AnyreachContractError):
        anyreach.benchmark_slots({})  # type: ignore[arg-type]


def test_contract_module_has_no_candidate_runtime_or_policy_imports() -> None:
    source = Path(anyreach.__file__).read_text(encoding="utf-8")
    import_lines = {
        line.strip()
        for line in source.splitlines()
        if line.startswith("import ") or line.startswith("from ")
    }

    for forbidden in (
        "onnx",
        "onnxruntime",
        "tokenizers",
        "transformers",
        "pyarrow",
        "huggingface_hub",
        "always_on_agent",
        "core",
    ):
        assert all(forbidden not in line for line in import_lines)


def test_public_dataclass_surfaces_hide_local_paths() -> None:
    assert {item.name for item in fields(anyreach.AnyreachSourceLock)} == {
        "path",
        "raw_sha256",
        "raw_size_bytes",
        "recipe_sha256",
        "candidate_id",
        "acceptance_id",
        "model_repository",
        "model_revision",
        "model_artifacts",
        "model_action_token_ids",
        "predict_token_id",
        "benchmark_repository",
        "benchmark_revision",
        "benchmark_artifact",
        "benchmark_class_labels",
        "source_rows",
        "selected_ids_sha256",
        "slots",
        "privacy",
        "evidence_scope",
    }
    assert {item.name for item in fields(anyreach.BoundArtifact)} == {
        "name",
        "role",
        "path",
        "sha256",
        "size_bytes",
    }
    assert fields(anyreach.AnyreachSourceLock)[0].repr is False
    assert fields(anyreach.BoundArtifact)[2].repr is False
