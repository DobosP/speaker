from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from tools import public_voice_eval_matrix as matrix


ALL_TERMS = frozenset(
    {
        "CC-BY-4.0",
        "CC0-1.0",
        "MDC-CONSUMER-TERMS-2026-05-06",
        "Apache-2.0",
        "LIBRIVOX-PUBLIC-DOMAIN",
        "EDINBURGH-VCTK-DATA-TERMS",
        "AUDIOSET-CC-BY-4.0",
        "FREESOUND-SELECTED-CC0-1.0",
        "DEMAND-CC-BY-SA-3.0",
        "CC-BY-SA-3.0",
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
    assert matrix.MATRIX_VERSION == 3
    speech = matrix.dataset_by_id("speech_commands_v002_test")
    assert speech.revision == "57ba463ab37e1e7845e0626539a6f6d0fcfbe64a"
    assert speech.selection.expected_examples == 4_890
    assert _artifact(
        "speech_commands_v002_test", "test-archive"
    ).checksum == matrix.Checksum(
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
    assert all(
        dataset.license.redistribution == "cache-only" for dataset in matrix.DATASETS
    )
    assert all(dataset.license.acceptance_ids for dataset in matrix.DATASETS)
    assert all(
        "NC" not in dataset.license.license_id.upper() for dataset in matrix.DATASETS
    )


def test_rochester_and_demand_sources_are_exact_and_fail_closed():
    rochester = matrix.dataset_by_id("rochester_smart_speaker_v1")
    assert rochester.revision == "10.60593/ur.d.26417548.v1"
    assert rochester.license.license_id == "CC-BY-4.0"
    assert matrix.UsageConstraint.HUMAN_LABEL_MAP_REQUIRED in (
        rochester.usage_constraints
    )
    assert (
        rochester.selection.algorithm
        is matrix.SelectionAlgorithm.HUMAN_LABEL_MAP_REQUIRED
    )
    assert rochester.selection.expected_examples == 28
    assert "Never infer labels" in rochester.selection.description
    assert _artifact("rochester_smart_speaker_v1", "archive").size_bytes == (
        470_415_384
    )
    assert _artifact("rochester_smart_speaker_v1", "archive").checksum.digest == (
        "5cb2f37ad1c4646cc870e653e80eb7e10028ad9d67608e985b8625d45024757d"
    )
    assert _artifact("rochester_smart_speaker_v1", "manual").checksum.digest == (
        "0697dc54ec2365ae534de4135aa4409e55cf4b826eeb782facdf78032d135475"
    )

    consumed = False

    def inferred_rows():
        nonlocal consumed
        consumed = True
        yield {
            "filename": "unsafe-inferred.wav",
            "speaker": "speaker-1",
            "command": "unsafe inferred label",
            "take": 1,
            "command_domain": "unsafe",
        }

    with pytest.raises(matrix.MatrixError, match="human/author"):
        matrix.select_subset_rows(rochester.selection, inferred_rows())
    assert consumed is False

    demand = matrix.dataset_by_id("demand_domestic_16k")
    assert demand.revision == "10.5281/zenodo.1227121"
    assert demand.license.license_id == "CC-BY-SA-3.0"
    assert [item.size_bytes for item in demand.artifacts] == [
        110_501_049,
        80_170_627,
        102_343_250,
    ]
    assert [item.checksum.digest for item in demand.artifacts] == [
        "9d68e46d2709847c59580770e2f8aa160607ebff7475788174aa1080332cc845",
        "2b1726fe06e41551ce2397f2aaf3e4fb692c912d81914b04708df0bbb5252338",
        "f687c0806b0e9731b71a9c01c6134a21330efc5fa4aa7ee97cf4df276ad1dae1",
    ]
    noise = next(track for track in matrix.TRACKS if track.track_id == "noise-reverb")
    assert "demand_domestic_16k" in noise.dataset_ids


def test_common_voice_spontaneous_v4_pin_terms_and_track_are_exact():
    dataset = matrix.dataset_by_id("common_voice_spontaneous_4_en")
    assert dataset.upstream_id == "mdc:cmqialpeo0077nr077xqdqo0j"
    assert dataset.revision == "sps-corpus-4.0-2026-06-12"
    assert dataset.canonical_url == (
        "https://mozilladatacollective.com/datasets/cmqialpeo0077nr077xqdqo0j"
    )
    assert dataset.license.license_url == (
        "https://mozilladatacollective.com/terms/consumers"
    )
    assert dataset.license.acceptance_ids == (
        "CC0-1.0",
        "MDC-CONSUMER-TERMS-2026-05-06",
    )
    artifact = _artifact("common_voice_spontaneous_4_en", "archive")
    assert artifact.filename == (
        "common-voice-spontaneous-speech-4-0-engl-c643378f.tar.gz"
    )
    assert artifact.size_bytes == 522_005_930
    assert artifact.integrity is matrix.IntegrityKind.MDC_API_SHA256
    assert artifact.checksum == matrix.Checksum(
        matrix.ChecksumAlgorithm.SHA256,
        "3b03ada7676a5f440a797d896035137fd073d0683133c3e9a83963480d88abfe",
        "cv-dataset-stats",
    )
    assert set(dataset.usage_constraints) == {
        matrix.UsageConstraint.EVALUATION_ONLY,
        matrix.UsageConstraint.PRIVATE_CACHE_ONLY,
        matrix.UsageConstraint.NO_REIDENTIFICATION,
        matrix.UsageConstraint.NO_REHOSTING,
    }
    assert "clean/unannotated" in dataset.selection.description
    for duration_filter in ("[2,6)", "[6,10)", "[10,15)", "[15,25]"):
        assert duration_filter in dataset.selection.description
    stt = next(
        track for track in matrix.TRACKS if track.category is matrix.TaskCategory.STT
    )
    assert dataset.dataset_id in stt.dataset_ids
    assert all(
        item.canonical_url.startswith("https://mozilladatacollective.com/datasets/")
        for item in matrix.DATASETS
        if item.upstream_id.startswith("mdc:")
    )


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
    assert (
        "not independent"
        in matrix.dataset_by_id("speech_commands_v002_test").evidence_note
    )


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
        "fluent_speech_commands",
        "sonos_voice_control_dataset",
    }
    assert {item.dataset_id for item in matrix.EXCLUSIONS} == expected
    assert all(item.selectable is False for item in matrix.EXCLUSIONS)
    assert not expected & {dataset.dataset_id for dataset in matrix.DATASETS}
    with pytest.raises(matrix.MatrixError, match="explicitly non-selectable"):
        matrix.dataset_by_id("notsofar_1_dev_set_2")
    exclusions = {item.dataset_id: item.reason for item in matrix.EXCLUSIONS}
    assert "product testing" in exclusions["fluent_speech_commands"]
    assert "research/non-commercial" in exclusions["sonos_voice_control_dataset"]


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
        matrix.build_acquisition_plan(("common_voice_26_ro",), environ={}, **kwargs)
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


def test_pinned_mdc_archive_requires_matching_api_receipt(tmp_path):
    dataset_id = "common_voice_spontaneous_4_en"
    artifact = _artifact(dataset_id, "archive")
    assert artifact.checksum is not None
    secret = "mdc-secret-pin-must-not-escape"
    kwargs = {
        "root": tmp_path / "corpora",
        "accepted_terms": ALL_TERMS,
        "environ": {matrix.MDC_TOKEN_ENV: secret},
    }
    stale_terms = frozenset(
        (ALL_TERMS - {"MDC-CONSUMER-TERMS-2026-05-06"}) | {"MDC-DATA-TERMS"}
    )
    with pytest.raises(matrix.MatrixError, match="MDC-CONSUMER-TERMS-2026-05-06"):
        matrix.build_acquisition_plan(
            (dataset_id,),
            root=kwargs["root"],
            accepted_terms=stale_terms,
            environ=kwargs["environ"],
        )
    with pytest.raises(matrix.MatrixError, match="explicit SHA-256"):
        matrix.build_acquisition_plan((dataset_id,), **kwargs)

    matching = matrix.Checksum(
        matrix.ChecksumAlgorithm.SHA256,
        artifact.checksum.digest,
        "mdc-api",
    )
    plan = matrix.build_acquisition_plan(
        (dataset_id,),
        checksum_receipts={f"{dataset_id}/archive": matching},
        **kwargs,
    )
    assert plan.artifacts[0].checksum == matrix.Checksum(
        matrix.ChecksumAlgorithm.SHA256,
        artifact.checksum.digest,
        "cv-dataset-stats+mdc-api",
    )
    assert secret not in repr(plan)

    mismatch = replace(matching, digest="d" * 64)
    with pytest.raises(matrix.MatrixError, match="does not match the catalog pin"):
        matrix.build_acquisition_plan(
            (dataset_id,),
            checksum_receipts={f"{dataset_id}/archive": mismatch},
            **kwargs,
        )


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


def test_speaker_stratified_selector_is_balanced_unique_and_order_independent():
    policy = matrix.dataset_by_id("common_voice_spontaneous_4_en").selection
    strata = ("[2,6)", "[6,10)", "[10,15)", "[15,25]")
    rows = [
        {
            "audio_id": f"audio-{stratum_index}-{row_index}",
            "audio_file": f"clip-{stratum_index}-{row_index}.mp3",
            "client_id": f"speaker-{stratum_index}-{row_index}",
            "duration_bucket": stratum,
        }
        for stratum_index, stratum in enumerate(strata)
        for row_index in range(12)
    ]

    first = matrix.select_subset_rows(policy, rows)
    second = matrix.select_subset_rows(policy, reversed(rows))

    assert [row["audio_id"] for row in first] == [row["audio_id"] for row in second]
    assert len(first) == 32
    assert len({row["client_id"] for row in first}) == 32
    assert Counter(row["duration_bucket"] for row in first) == {
        stratum: 8 for stratum in strata
    }


def test_speaker_stratified_selector_rejects_empty_and_forbidden_speakers():
    policy = matrix.SelectionPolicy(
        "unit speaker selector",
        algorithm=matrix.SelectionAlgorithm.HASH_SPEAKER_STRATIFIED,
        identity_fields=("id",),
        strata_fields=("bucket",),
        seed="unit-speaker-seed",
        expected_examples=2,
        speaker_disjoint=True,
        speaker_field="speaker",
    )
    rows = [
        {"id": "one", "bucket": "short", "speaker": "speaker-one"},
        {"id": "two", "bucket": "long", "speaker": "speaker-two"},
    ]
    selected = matrix.select_subset_rows(policy, rows)
    with pytest.raises(matrix.MatrixError, match="insufficient exact rows"):
        matrix.select_subset_rows(
            policy,
            rows,
            forbidden_speaker_ids=frozenset({str(selected[0]["speaker"])}),
        )

    invalid_rows = [*rows, {"id": "empty", "bucket": "short", "speaker": ""}]
    with pytest.raises(matrix.MatrixError, match="non-empty speaker"):
        matrix.select_subset_rows(policy, invalid_rows)


def test_speaker_stratified_selector_requires_exact_unique_speaker_count():
    policy = matrix.SelectionPolicy(
        "unit speaker selector",
        algorithm=matrix.SelectionAlgorithm.HASH_SPEAKER_STRATIFIED,
        identity_fields=("id",),
        strata_fields=("bucket",),
        seed="unit-speaker-seed",
        expected_examples=3,
        speaker_field="speaker",
    )
    rows = [
        {"id": "one-a", "bucket": "short", "speaker": "one"},
        {"id": "one-b", "bucket": "long", "speaker": "one"},
        {"id": "two", "bucket": "short", "speaker": "two"},
    ]
    with pytest.raises(matrix.MatrixError, match="insufficient exact rows"):
        matrix.select_subset_rows(policy, rows)


def test_speaker_stratified_selector_ranks_speakers_before_clips(monkeypatch):
    policy = matrix.SelectionPolicy(
        "speaker-first selector",
        algorithm=matrix.SelectionAlgorithm.HASH_SPEAKER_STRATIFIED,
        identity_fields=("id",),
        strata_fields=("bucket",),
        seed="speaker-first-seed",
        expected_examples=1,
        speaker_field="speaker",
    )
    rows = [
        {"id": f"prolific-{index:02d}", "bucket": "one", "speaker": "prolific"}
        for index in range(20)
    ]
    rows.append({"id": "rare-only", "bucket": "one", "speaker": "rare"})

    def rank(_seed: str, identity: str) -> str:
        if identity.startswith("speaker\0"):
            return "0" if identity.endswith("\0rare") else "f"
        return identity

    monkeypatch.setattr(matrix, "_selection_rank", rank)
    selected = matrix.select_subset_rows(policy, rows)
    assert [row["speaker"] for row in selected] == ["rare"]


def test_speaker_stratified_matching_finds_feasible_cross_stratum_assignment(
    monkeypatch,
):
    policy = matrix.SelectionPolicy(
        "feasible quota matching",
        algorithm=matrix.SelectionAlgorithm.HASH_SPEAKER_STRATIFIED,
        identity_fields=("id",),
        strata_fields=("bucket",),
        seed="matching-seed",
        expected_examples=2,
        speaker_field="speaker",
    )
    rows = [
        {"id": "a-shared", "bucket": "a", "speaker": "shared"},
        {"id": "a-alternative", "bucket": "a", "speaker": "alternative"},
        {"id": "b-shared", "bucket": "b", "speaker": "shared"},
    ]

    def rank(_seed: str, identity: str) -> str:
        if identity.startswith("speaker\0"):
            return "0" if identity.endswith("\0shared") else "f"
        return identity

    monkeypatch.setattr(matrix, "_selection_rank", rank)
    selected = matrix.select_subset_rows(policy, rows)
    assert {(row["bucket"], row["speaker"]) for row in selected} == {
        ("a", "alternative"),
        ("b", "shared"),
    }


def test_selection_policy_digest_binds_full_recipe_only():
    base = matrix.SelectionPolicy(
        "base recipe",
        algorithm=matrix.SelectionAlgorithm.HASH_SPEAKER_STRATIFIED,
        split="test",
        identity_fields=("id",),
        strata_fields=("bucket",),
        seed="base-seed",
        expected_examples=2,
        speaker_disjoint=True,
        speaker_field="speaker",
        fixed_ids=("fixed",),
    )
    variants = (
        replace(base, description="changed recipe"),
        replace(base, algorithm=matrix.SelectionAlgorithm.HASH_STRATIFIED),
        replace(base, split="dev"),
        replace(base, identity_fields=("other_id",)),
        replace(base, strata_fields=("other_bucket",)),
        replace(base, seed="other-seed"),
        replace(base, expected_examples=3),
        replace(base, speaker_disjoint=False),
        replace(base, speaker_field="other_speaker"),
        replace(base, fixed_ids=("other",)),
    )
    digest = matrix.selection_policy_sha256(base)
    assert len(digest) == 64
    assert matrix.selection_policy_sha256(base) == digest
    assert all(matrix.selection_policy_sha256(policy) != digest for policy in variants)
    assert (
        len({digest, *(matrix.selection_policy_sha256(item) for item in variants)})
        == 11
    )

    rows = ({"id": "one"}, {"id": "two"})
    assert matrix.selected_rows_sha256(base, rows) == matrix.selected_rows_sha256(
        replace(base, description="changed recipe"), rows
    )


def test_southern_american_terms_prohibit_voice_clone_and_rehosting():
    constraints = set(
        matrix.dataset_by_id("common_voice_26_southern_american").usage_constraints
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
    assert set(
        matrix.dataset_by_id("microsoft_aec_challenge").license.acceptance_ids
    ) == {
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
