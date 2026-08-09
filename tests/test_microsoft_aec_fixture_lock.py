from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from tools import public_voice_eval_matrix as matrix


LOCK_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "audio_eval"
    / "microsoft_aec_32.lock.json"
)
SELF_DIGEST = "a1c6c65f8d40dd147bb7ba13fc0e389bfe33adecf411b3200031215ba65cf198"
POLICY_DIGEST = "abd006fe51ba980bdabc197a0ac40a47a5c5728b964badf26b43a5aa9e1e10bd"
SELECTED_ROWS_DIGEST = (
    "542d6d102cad3f47c7c5aad96bbf04c7796654ba13573068bc930232b38ba331"
)
CASE_IDS = (
    "3960",
    "5234",
    "6147",
    "9489",
    "2250",
    "1872",
    "565",
    "710",
    "4223",
    "8756",
    "620",
    "4673",
    "7082",
    "3946",
    "3835",
    "5685",
    "9956",
    "9864",
    "3970",
    "6444",
    "4049",
    "6411",
    "8566",
    "4427",
    "6622",
    "1557",
    "1477",
    "3163",
    "6537",
    "6078",
    "431",
    "5598",
)
ROLES = (
    "echo_signal",
    "farend_speech",
    "nearend_mic_signal",
    "nearend_speech",
)
METADATA_COLUMNS = (
    "nearend_speaker",
    "nearend_wav_path",
    "nearend_wav_path_noisy",
    "farend_speaker",
    "farend_wav_path",
    "farend_wav_path_noisy",
    "ser",
    "is_farend_nonlinear",
    "is_farend_noisy",
    "is_nearend_noisy",
    "split",
    "fileid",
    "nearend_scale",
)
LICENSE_ACCEPTANCES = (
    "LIBRIVOX-PUBLIC-DOMAIN",
    "EDINBURGH-VCTK-DATA-TERMS",
    "AUDIOSET-CC-BY-4.0",
    "FREESOUND-SELECTED-CC0-1.0",
    "DEMAND-CC-BY-SA-3.0",
)


def _load_lock() -> dict[str, object]:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            assert key not in value
            value[key] = item
        return value

    def reject_constant(value: str) -> object:
        raise AssertionError(f"non-finite JSON number: {value}")

    loaded = json.loads(
        LOCK_PATH.read_text(encoding="utf-8"),
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )
    assert isinstance(loaded, dict)
    return loaded


def _canonical_self_digest(lock: dict[str, object]) -> str:
    payload = json.loads(json.dumps(lock, ensure_ascii=False, allow_nan=False))
    self_digest = payload["self_digest"]
    assert isinstance(self_digest, dict)
    self_digest.pop("value")
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_microsoft_aec_lock_has_exact_root_shape_and_self_digest():
    lock = _load_lock()

    assert set(lock) == {
        "schema_version",
        "fixture_id",
        "self_digest",
        "source",
        "license",
        "usage_constraints",
        "selection",
        "layout",
        "metadata",
        "metric_evidence",
        "cases",
    }
    assert lock["schema_version"] == 1
    assert lock["fixture_id"] == "microsoft_aec_challenge_synthetic_hash32_v1"
    assert lock["self_digest"] == {
        "algorithm": "sha256",
        "canonicalization": (
            "UTF-8 JSON with sort_keys=true, separators=(',', ':'), "
            "ensure_ascii=false, allow_nan=false"
        ),
        "excluded_json_pointer": "/self_digest/value",
        "value": SELF_DIGEST,
    }
    assert _canonical_self_digest(lock) == SELF_DIGEST


def test_microsoft_aec_lock_matches_source_terms_and_owns_exact_selector():
    lock = _load_lock()
    dataset = matrix.dataset_by_id("microsoft_aec_challenge")

    assert lock["source"] == {
        "repository_url": dataset.canonical_url,
        "commit": dataset.revision,
        "root_tree_sha1": "0febaaae06af9ac70a7f21d16b9ec4719b8fbfe7",
    }
    assert dataset.artifacts[0].checksum is not None
    assert dataset.artifacts[0].checksum.digest == lock["source"]["commit"]
    assert tuple(lock["license"]["acceptance_ids"]) == LICENSE_ACCEPTANCES
    assert dataset.license.acceptance_ids == LICENSE_ACCEPTANCES
    assert lock["license"]["gate_id"] == dataset.license.license_id
    assert lock["license"]["redistribution"] == dataset.license.redistribution
    assert lock["usage_constraints"] == [
        item.value for item in dataset.usage_constraints
    ]

    selection = lock["selection"]
    assert selection == {
        "algorithm": "sha256_top_k_v1",
        "description": (
            "Rank synthetic/meta.csv rows by SHA-256(UTF8(seed) || 0x00 || "
            "compact canonical JSON([fileid])), sort by (rank, identity), take "
            "32, and bind all four LFS WAVs per selected row."
        ),
        "split": "all-metadata-rows",
        "identity_fields": ["fileid"],
        "strata_fields": [],
        "seed": "speaker-public-eval-v1-2026-08-01",
        "expected_examples": 32,
        "speaker_disjoint": False,
        "speaker_field": None,
        "fixed_ids": [],
        "policy_sha256": POLICY_DIGEST,
        "selected_rows_sha256": SELECTED_ROWS_DIGEST,
        "source_row_domain": "all_metadata_rows",
        "selected_upstream_split_counts": {"train": 31, "test": 1},
    }
    policy_payload = {
        key: selection[key]
        for key in (
            "algorithm",
            "description",
            "expected_examples",
            "fixed_ids",
            "identity_fields",
            "seed",
            "speaker_disjoint",
            "speaker_field",
            "split",
            "strata_fields",
        )
    }
    assert (
        hashlib.sha256(
            json.dumps(
                policy_payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        == POLICY_DIGEST
    )
    identities = [
        json.dumps([case_id], ensure_ascii=False, separators=(",", ":"))
        for case_id in CASE_IDS
    ]
    ranked = sorted(
        (
            hashlib.sha256(
                selection["seed"].encode("utf-8")
                + b"\0"
                + json.dumps(
                    [str(index)], ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest(),
            json.dumps([str(index)], ensure_ascii=False, separators=(",", ":")),
            str(index),
        )
        for index in range(10_000)
    )
    assert tuple(fileid for _rank, _identity, fileid in ranked[:32]) == CASE_IDS
    assert (
        hashlib.sha256(
            json.dumps(
                identities,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        == SELECTED_ROWS_DIGEST
    )


def test_microsoft_aec_lock_pins_exact_metadata_and_case_order():
    lock = _load_lock()
    metadata = lock["metadata"]
    cases = lock["cases"]

    assert metadata == {
        "relative_path": "datasets/synthetic/meta.csv",
        "git_blob_sha1": "1a02a5b41ef43420bd24b1b8386d1df0cba2e6d8",
        "sha256": "6b867aa68f510e4bafc9ffa05202930ffc1bb54bb0cffe66aa1ec3bcc606f3e5",
        "size_bytes": 3_243_688,
        "row_count": 10_000,
        "columns": list(METADATA_COLUMNS),
    }
    assert tuple(case["fileid"] for case in cases) == CASE_IDS
    assert tuple(case["ordinal"] for case in cases) == tuple(range(1, 33))
    assert Counter(case["metadata"]["split"] for case in cases) == Counter(
        {"train": 31, "test": 1}
    )
    for case in cases:
        row = case["metadata"]
        assert tuple(row) == METADATA_COLUMNS
        assert all(isinstance(value, str) for value in row.values())
        assert row["fileid"] == case["fileid"]
        assert float(row["nearend_scale"]) > 0.0


def test_microsoft_aec_lock_has_four_ordered_lfs_pins_and_exact_totals():
    lock = _load_lock()
    cases = lock["cases"]
    paths: list[str] = []
    audio_bytes = 0
    leaf_by_role = {
        "echo_signal": "echo_fileid_{fileid}.wav",
        "farend_speech": "farend_speech_fileid_{fileid}.wav",
        "nearend_mic_signal": "nearend_mic_fileid_{fileid}.wav",
        "nearend_speech": "nearend_speech_fileid_{fileid}.wav",
    }

    for case in cases:
        artifacts = case["artifacts"]
        assert len(artifacts) == 4
        assert tuple(artifact["role"] for artifact in artifacts) == ROLES
        for artifact in artifacts:
            assert set(artifact) == {
                "role",
                "relative_path",
                "git_pointer_blob_sha1",
                "lfs_oid_sha256",
                "size_bytes",
            }
            role = artifact["role"]
            expected = (
                PurePosixPath("datasets/synthetic")
                / role
                / leaf_by_role[role].format(fileid=case["fileid"])
            )
            path = PurePosixPath(artifact["relative_path"])
            assert path == expected
            assert not path.is_absolute()
            assert ".." not in path.parts
            assert re.fullmatch(r"[0-9a-f]{40}", artifact["git_pointer_blob_sha1"])
            assert re.fullmatch(r"[0-9a-f]{64}", artifact["lfs_oid_sha256"])
            assert artifact["size_bytes"] in {320_042, 320_044}
            paths.append(artifact["relative_path"])
            audio_bytes += artifact["size_bytes"]

    assert len(paths) == len(set(paths)) == 128
    assert lock["layout"] == {
        "case_count": 32,
        "signal_roles": list(ROLES),
        "artifact_count": 128,
        "audio_bytes": 40_965_456,
        "metadata_bytes": 3_243_688,
        "total_pinned_bytes": 44_209_144,
    }
    assert audio_bytes == lock["layout"]["audio_bytes"]
    assert audio_bytes + lock["metadata"]["size_bytes"] == 44_209_144


def test_microsoft_aec_lock_keeps_metric_and_evidence_limits_explicit():
    lock = _load_lock()
    evidence = lock["metric_evidence"]
    track = next(item for item in matrix.TRACKS if item.track_id == "aec-double-talk")

    assert evidence["track_id"] == track.track_id
    assert evidence["declared_metrics"] == [metric.value for metric in track.metrics]
    assert evidence["forbidden_metrics"] == [
        "wer",
        "cer",
        "intent_accuracy",
        "tool_selection_accuracy",
        "tool_result_accuracy",
    ]
    assert set(evidence["limitations"]) == {
        "target_scaling",
        "erle_db",
        "aecmos",
        "near_end_attenuation_db",
        "barge_metrics",
        "evidence_scope",
    }
    assert "nearend_scale" in evidence["limitations"]["target_scaling"]
    assert "no interval annotations" in evidence["limitations"]["erle_db"]
    assert "separately pinned compatible scorer" in evidence["limitations"]["aecmos"]
    assert "no command transcript" in evidence["limitations"]["barge_metrics"]
    assert "not live microphone" in evidence["limitations"]["evidence_scope"]
