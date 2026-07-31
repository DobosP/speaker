from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tools import public_voice_eval_matrix as matrix


ALL_TERMS = frozenset(
    {
        "CC-BY-4.0",
        "CC0-1.0",
        "MDC-DATA-TERMS",
        "Apache-2.0",
        "LIBRIVOX-PUBLIC-DOMAIN",
        "EDINBURGH-VCTK-DATA-TERMS",
        "AUDIOSET-CC-BY-4.0",
        "FREESOUND-SELECTED-CC0-1.0",
        "DEMAND-CC-BY-SA-3.0",
    }
)


def _artifact(dataset_id: str, artifact_id: str) -> matrix.Artifact:
    dataset = matrix.dataset_by_id(dataset_id)
    return next(item for item in dataset.artifacts if item.artifact_id == artifact_id)


def test_matrix_has_one_track_per_separate_task_metric_family():
    matrix.validate_matrix()

    assert {track.category for track in matrix.TRACKS} == set(matrix.TaskCategory)
    assert len(matrix.TRACKS) == len(matrix.TaskCategory)
    aec = next(
        track
        for track in matrix.TRACKS
        if track.category is matrix.TaskCategory.AEC_DOUBLE_TALK
    )
    assert matrix.Metric.ERLE_DB in aec.metrics
    assert matrix.Metric.BARGE_CUT_LATENCY_MS in aec.metrics
    assert not any("wer" in metric.value for metric in aec.metrics)

    spoken = next(
        track
        for track in matrix.TRACKS
        if track.category is matrix.TaskCategory.SPOKEN_INTENT
    )
    assert spoken.metrics == (matrix.Metric.INTENT_ACCURACY,)
    routing = next(
        track
        for track in matrix.TRACKS
        if track.category is matrix.TaskCategory.TEXT_SEMANTIC_ROUTING
    )
    assert matrix.Metric.TOOL_SELECTION_ACCURACY in routing.metrics
    assert matrix.Metric.LITERAL_WER not in routing.metrics


def test_primary_dataset_ids_revisions_checksums_and_licenses_are_pinned():
    speech = matrix.dataset_by_id("speech_commands_v002_test")
    assert speech.revision == "57ba463ab37e1e7845e0626539a6f6d0fcfbe64a"
    assert speech.selection.expected_examples == 4_890
    assert _artifact("speech_commands_v002_test", "test-archive").checksum == matrix.Checksum(
        matrix.ChecksumAlgorithm.SHA256,
        "af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58",
        "upstream",
    )

    assert matrix.dataset_by_id("common_voice_26_ro").upstream_id == (
        "mdc:cmqinu08d00xenq07ha33e29y"
    )
    assert matrix.dataset_by_id("common_voice_26_southern_american").revision == (
        "cmryvrc6i014vo90786kriix5"
    )
    assert matrix.dataset_by_id("common_voice_26_malaysian_english").revision == (
        "cmrt71n6q001fmm07u64adwuv"
    )
    assert matrix.dataset_by_id("common_voice_26_irish_english").revision == (
        "cmrt71fck001bmm07cj9h312i"
    )
    assert matrix.dataset_by_id("minds14_english").revision == (
        "40ce77cb32a384e4d50a568e1ec39ac804019d33"
    )
    assert matrix.dataset_by_id("massive_1_1_text").revision == (
        "ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740"
    )
    assert _artifact("dr_vctk_small", "archive").checksum.digest == (
        "29e93debeb0e779986542229a81ff29b"
    )
    assert _artifact("rirs_noises", "archive").checksum.digest == (
        "e6f48e257286e05de56413b4779d8ffb"
    )
    assert matrix.dataset_by_id("microsoft_aec_challenge").revision == (
        "6c633d0a9d2a143a0e364899b91b06f127315b18"
    )
    assert all(dataset.evaluation_only for dataset in matrix.DATASETS)
    assert all(dataset.license.redistribution == "cache-only" for dataset in matrix.DATASETS)
    assert all(dataset.license.acceptance_ids for dataset in matrix.DATASETS)
    assert all("NC" not in dataset.license.license_id.upper() for dataset in matrix.DATASETS)


def test_ami_local_freeze_is_exact_and_warns_about_training_overlap():
    ami = matrix.dataset_by_id("ami_eval_three")
    assert [artifact.artifact_id for artifact in ami.artifacts] == [
        "annotations",
        "es2004a-array1-01",
        "is1009a-array1-01",
        "ts3003a-array1-01",
    ]
    assert all(artifact.checksum.source == "local-freeze" for artifact in ami.artifacts)
    assert "ES2004d" in ami.selection.description
    assert "not independent" in ami.evidence_note
    assert "not independent" in matrix.dataset_by_id(
        "speech_commands_v002_test"
    ).evidence_note


def test_unlicensed_challenge_only_and_nc_sources_are_non_selectable():
    expected = {
        "smart_turn_v3_2_test",
        "notsofar_1_open_subsets",
        "notsofar_1_dev_set_2",
        "libricss",
        "slurp_audio",
        "l2_arctic",
        "morovoc",
        "openwakeword_collections",
        "hey_snips",
    }
    assert {item.dataset_id for item in matrix.EXCLUSIONS} == expected
    assert all(item.selectable is False for item in matrix.EXCLUSIONS)
    assert not expected & {dataset.dataset_id for dataset in matrix.DATASETS}
    with pytest.raises(matrix.MatrixError, match="explicitly non-selectable"):
        matrix.dataset_by_id("notsofar_1_dev_set_2")


def test_default_corpus_root_uses_xdg_and_plan_rejects_git_worktree():
    assert matrix.default_corpus_root({"XDG_CACHE_HOME": "/private/cache"}) == Path(
        "/private/cache/speaker/corpora"
    )
    with pytest.raises(matrix.MatrixError, match="outside the Git worktree"):
        matrix.build_acquisition_plan(
            ("speech_commands_v002_test",),
            root=Path(matrix.__file__).resolve().parents[1] / ".cache" / "corpora",
            accepted_terms=ALL_TERMS,
            environ={},
        )


def test_acquisition_is_explicit_license_gated_and_deterministic(tmp_path):
    root = tmp_path / "corpora"
    with pytest.raises(matrix.MatrixError, match="explicit acceptance"):
        matrix.build_acquisition_plan(
            ("speech_commands_v002_test",),
            root=root,
            accepted_terms=frozenset(),
            environ={},
        )

    first = matrix.build_acquisition_plan(
        ("minds14_english", "speech_commands_v002_test"),
        root=root,
        accepted_terms=ALL_TERMS,
        environ={},
    )
    second = matrix.build_acquisition_plan(
        ("speech_commands_v002_test", "minds14_english"),
        root=root,
        accepted_terms=ALL_TERMS,
        environ={},
    )
    assert first == second
    assert first.dataset_ids == ("minds14_english", "speech_commands_v002_test")
    assert not root.exists()
    assert all(artifact.destination.is_absolute() for artifact in first.artifacts)


def test_mdc_requires_api_sha256_and_never_retains_or_prints_token(tmp_path):
    secret = "mdc-secret-that-must-not-escape"
    receipt = matrix.Checksum(
        matrix.ChecksumAlgorithm.SHA256,
        "a" * 64,
        "mdc-api",
    )
    kwargs = {
        "root": tmp_path / "corpora",
        "accepted_terms": ALL_TERMS,
        "checksum_receipts": {"common_voice_26_ro/archive": receipt},
    }
    with pytest.raises(matrix.MatrixError) as missing:
        matrix.build_acquisition_plan(
            ("common_voice_26_ro",), environ={}, **kwargs
        )
    assert matrix.MDC_TOKEN_ENV in str(missing.value)
    assert secret not in str(missing.value)

    plan = matrix.build_acquisition_plan(
        ("common_voice_26_ro",),
        environ={matrix.MDC_TOKEN_ENV: secret},
        **kwargs,
    )
    assert plan.credential_env_names == (matrix.MDC_TOKEN_ENV,)
    constraints = dict(plan.usage_constraints)["common_voice_26_ro"]
    assert matrix.UsageConstraint.NO_REIDENTIFICATION in constraints
    assert matrix.UsageConstraint.NO_REHOSTING in constraints
    assert plan.artifacts[0].checksum == receipt
    assert secret not in repr(plan)
    assert secret not in repr(plan.artifacts[0])


@pytest.mark.parametrize(
    ("receipt", "message"),
    [
        (
            matrix.Checksum(matrix.ChecksumAlgorithm.MD5, "a" * 32, "mdc-api"),
            "must use sha256",
        ),
        (
            matrix.Checksum(matrix.ChecksumAlgorithm.SHA256, "a" * 64, "local-freeze"),
            "receipt source",
        ),
    ],
)
def test_mdc_receipt_fails_closed_on_wrong_algorithm_or_source(
    tmp_path, receipt, message
):
    with pytest.raises(matrix.MatrixError, match=message):
        matrix.build_acquisition_plan(
            ("common_voice_26_ro",),
            root=tmp_path / "corpora",
            accepted_terms=ALL_TERMS,
            checksum_receipts={"common_voice_26_ro/archive": receipt},
            environ={matrix.MDC_TOKEN_ENV: "private"},
        )


def test_no_upstream_digest_requires_local_sha256_freeze(tmp_path):
    with pytest.raises(matrix.MatrixError, match="explicit SHA-256"):
        matrix.build_acquisition_plan(
            ("speechocean762",),
            root=tmp_path / "corpora",
            accepted_terms=ALL_TERMS,
            environ={},
        )
    receipt = matrix.Checksum(
        matrix.ChecksumAlgorithm.SHA256,
        "b" * 64,
        "local-freeze",
    )
    plan = matrix.build_acquisition_plan(
        ("speechocean762",),
        root=tmp_path / "corpora",
        accepted_terms=ALL_TERMS,
        checksum_receipts={"speechocean762/archive": receipt},
        environ={},
    )
    assert plan.artifacts[0].checksum == receipt


def test_massive_binds_official_archive_and_requires_content_receipt(tmp_path):
    with pytest.raises(matrix.MatrixError, match="explicit SHA-256"):
        matrix.build_acquisition_plan(
            ("massive_1_1_text",),
            root=tmp_path / "corpora",
            accepted_terms=ALL_TERMS,
            environ={},
        )
    receipt = matrix.Checksum(
        matrix.ChecksumAlgorithm.SHA256,
        "c" * 64,
        "local-freeze",
    )
    plan = matrix.build_acquisition_plan(
        ("massive_1_1_text",),
        root=tmp_path / "corpora",
        accepted_terms=ALL_TERMS,
        checksum_receipts={"massive_1_1_text/archive": receipt},
        environ={},
    )
    assert plan.artifacts[0].locator.endswith("amazon-massive-dataset-1.1.tar.gz")
    assert plan.artifacts[0].checksum == receipt


def test_hash_stratified_selector_is_executable_order_independent_and_bound():
    policy = matrix.SelectionPolicy(
        "unit selector",
        algorithm=matrix.SelectionAlgorithm.HASH_STRATIFIED,
        identity_fields=("id",),
        strata_fields=("accent",),
        seed="unit-seed",
        expected_examples=4,
        speaker_disjoint=True,
        speaker_field="speaker",
    )
    rows = [
        {"id": f"row-{index}", "accent": f"a{index % 2}", "speaker": f"s{index}"}
        for index in range(8)
    ]
    first = matrix.select_subset_rows(
        policy,
        rows,
        forbidden_speaker_ids=frozenset(),
    )
    second = matrix.select_subset_rows(
        policy,
        reversed(rows),
        forbidden_speaker_ids=frozenset(),
    )
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert matrix.selected_rows_sha256(policy, first) == matrix.selected_rows_sha256(
        policy, second
    )
    with pytest.raises(matrix.MatrixError, match="speaker-disjoint"):
        matrix.select_subset_rows(
            policy,
            rows,
            forbidden_speaker_ids=frozenset({str(first[0]["speaker"])}),
        )


def test_southern_american_terms_prohibit_voice_clone_and_rehosting():
    constraints = set(
        matrix.dataset_by_id(
            "common_voice_26_southern_american"
        ).usage_constraints
    )
    assert matrix.UsageConstraint.NO_REIDENTIFICATION in constraints
    assert matrix.UsageConstraint.NO_REHOSTING in constraints
    assert matrix.UsageConstraint.NO_VOICE_CLONING in constraints


def test_local_artifact_verification_checks_size_hash_and_regular_file(tmp_path):
    payload = b"known public fixture"
    path = (tmp_path / "artifact.bin").resolve()
    path.write_bytes(payload)
    planned = matrix.PlannedArtifact(
        "unit",
        "archive",
        "https://example.invalid/archive",
        path,
        matrix.Checksum(
            matrix.ChecksumAlgorithm.SHA256,
            hashlib.sha256(payload).hexdigest(),
            "local-freeze",
        ),
        len(payload),
        None,
    )
    freeze = matrix.verify_local_artifact(planned, path)
    assert freeze == matrix.Checksum(
        matrix.ChecksumAlgorithm.SHA256,
        hashlib.sha256(payload).hexdigest(),
        "local-freeze",
    )

    path.write_bytes(payload + b"changed")
    with pytest.raises(matrix.MatrixError, match="unavailable or unstable"):
        matrix.verify_local_artifact(planned, path)

    target = (tmp_path / "target.bin").resolve()
    target.write_bytes(payload)
    link = tmp_path / "link.bin"
    link.symlink_to(target)
    linked_plan = matrix.PlannedArtifact(
        "unit",
        "archive",
        "https://example.invalid/archive",
        link,
        planned.checksum,
        len(payload),
        None,
    )
    with pytest.raises(matrix.MatrixError, match="unavailable or unstable"):
        matrix.verify_local_artifact(linked_plan, link)


def test_aec_contract_pins_commit_and_requires_embedded_lfs_sha256():
    artifact = _artifact("microsoft_aec_challenge", "repository")
    assert artifact.integrity is matrix.IntegrityKind.GIT_COMMIT_AND_LFS_OIDS
    assert artifact.checksum == matrix.Checksum(
        matrix.ChecksumAlgorithm.GIT_SHA1,
        "6c633d0a9d2a143a0e364899b91b06f127315b18",
        "upstream-git",
    )
    assert "every selected WAV" in artifact.integrity_note
    assert "meta.csv source provenance" in artifact.integrity_note
    assert set(matrix.dataset_by_id("microsoft_aec_challenge").license.acceptance_ids) == {
        "LIBRIVOX-PUBLIC-DOMAIN",
        "EDINBURGH-VCTK-DATA-TERMS",
        "AUDIOSET-CC-BY-4.0",
        "FREESOUND-SELECTED-CC0-1.0",
        "DEMAND-CC-BY-SA-3.0",
    }
    selection = matrix.dataset_by_id("microsoft_aec_challenge").selection
    assert selection.expected_examples == 32
    assert selection.seed == matrix.SELECTION_SEED


def test_artifact_destination_cannot_escape_private_root(tmp_path):
    dataset = matrix.dataset_by_id("speech_commands_v002_test")
    artifact = matrix.Artifact(
        "bad",
        "https://example.invalid/source",
        "../escape.tar.gz",
        None,
        matrix.IntegrityKind.PINNED,
        matrix.Checksum(matrix.ChecksumAlgorithm.SHA256, "a" * 64, "upstream"),
    )
    with pytest.raises(matrix.MatrixError, match="unsafe artifact filename"):
        matrix._artifact_destination(tmp_path.resolve(), dataset, artifact)
