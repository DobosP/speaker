from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools import mobile_asr_evidence_eval as baseline
from tools import mobile_asr_two_pass_eval as evaluate
from tools.streaming_stt.protocol import (
    FinalEvent,
    MobileZipformerEndpointObservation,
    ResourceUsage,
)


def _private(path: Path) -> Path:
    path.mkdir(mode=0o700)
    path.chmod(0o700)
    return path


def _outputs() -> list[baseline._EndpointOutcome]:
    return [
        baseline._EndpointOutcome(
            (leaf.expected_text,),
            (True,),
            "endpoint",
            True,
        )
        for component in baseline._synthetic_inputs().components
        for leaf in component.leaves
    ]


def _candidate_texts(
    outputs: list[baseline._EndpointOutcome],
) -> tuple[str, ...]:
    return tuple(text for output in outputs for text in output.endpoint_texts)


def _run(tmp_path: Path, outputs=None, candidate_texts=None):
    selected = _outputs() if outputs is None else outputs
    texts = _candidate_texts(selected) if candidate_texts is None else candidate_texts
    parent = _private(tmp_path / "report-parent")
    return evaluate._run_synthetic_test_evaluation(
        tuple(selected), tuple(texts), parent / "report.json"
    )


def _one_leaf_inputs(raw: bytes) -> baseline._Inputs:
    leaf = baseline._Leaf(
        component_id="test",
        kind="corpus",
        sha256=hashlib.sha256(raw).hexdigest(),
        samples=len(raw) // 4,
        audio=raw,
        expected_text="",
    )
    return baseline._Inputs(
        components=(
            baseline._Component(
                component_id="test",
                metric_domain="test",
                logical_cases=1,
                leaves=(leaf,),
            ),
        ),
        production_inputs=False,
        packet_lock_sha256="0" * 64,
        packet_index_sha256="1" * 64,
        packet_receipt_sha256="2" * 64,
        pcm_bytes=len(raw),
        samples=len(raw) // 4,
    )


def _evaluation(
    observations: tuple[MobileZipformerEndpointObservation, ...],
) -> evaluate._BaselineEvaluation:
    return evaluate._BaselineEvaluation(
        outcome=baseline._EndpointOutcome(
            tuple(value.text for value in observations),
            tuple(value.source_complete for value in observations),
            "endpoint" if observations else "tail_exhausted",
            True,
        ),
        observations=observations,
    )


def _candidate_model_with_venv(tmp_path: Path):
    venv_root = tmp_path / "candidate-venv"
    python = venv_root / "bin" / "python"
    site_packages = venv_root / "lib" / "python3.12" / "site-packages"
    python.parent.mkdir(parents=True)
    site_packages.mkdir(parents=True)
    python.write_bytes(b"candidate python")

    provision = tmp_path / "candidate-provision"
    model = provision / "model"
    model.mkdir(parents=True)
    manifest_path = provision / "worker-manifest.json"
    worker = provision / "worker.py"
    artifact = model / "model.onnx"
    for path in (manifest_path, worker, artifact):
        path.write_bytes(b"bound")
    manifest = SimpleNamespace(
        python=SimpleNamespace(path=python),
        worker=SimpleNamespace(path=worker),
        artifacts=(SimpleNamespace(path=artifact),),
    )
    return (
        SimpleNamespace(manifest_path=manifest_path, manifest=manifest),
        venv_root,
        site_packages,
    )


def test_fixed_models_packet_and_accepted_baseline_bindings() -> None:
    assert evaluate.BASELINE_MODEL_ID == baseline.MODEL_ID
    assert evaluate.BASELINE_ADAPTER == baseline.ADAPTER
    assert evaluate.CANDIDATE_MODEL_ID == ("sherpa-onnx-whisper-base.en-mobile-int8-v1")
    assert evaluate.CANDIDATE_ADAPTER == "sherpa-onnx-mobile-whisper-final-v1"
    assert evaluate.ACCEPTED_BASELINE_REPORT_SHA256 == (
        "264f439652064ca78821abdcfed77d03a9a637fb82d6ed336b7a6112d083e796"
    )
    assert evaluate.ACCEPTED_BASELINE_AGGREGATE_SHA256 == (
        "e485da1a658bb010c0edf03707d875f667e80a9a44b8572099239e07b74aa188"
    )
    assert evaluate.EXPECTED_SOURCE_EVALUATIONS == 129
    assert evaluate.EXPECTED_ENDPOINT_EPOCHS == 198
    assert evaluate.MAXIMUM_DERIVED_EPOCH_PCM_BYTES == 64_067_500
    assert evaluate.CANDIDATE_WORKER_LIFETIME_PCM_LIMIT_BYTES == 65_011_712
    assert (
        evaluate.MAXIMUM_DERIVED_EPOCH_PCM_BYTES
        < evaluate.CANDIDATE_WORKER_LIFETIME_PCM_LIMIT_BYTES
    )


def test_candidate_config_binding_is_exact_and_uses_frozen_repo_name() -> None:
    from tools import provision_mobile_whisper as provision
    from tools.streaming_stt.manifest import (
        MOBILE_WHISPER_ARTIFACT_SET_SHA256,
        MOBILE_WHISPER_TOTAL_SIZE_BYTES,
        MobileWhisperConfig,
    )

    class ProvisionConfig:
        def __init__(self, value):
            self.value = value

        def as_dict(self):
            return dict(self.value)

    class ManifestConfig:
        def __init__(self, value):
            self.value = value

        def as_dict(self):
            return dict(self.value)

    revision = evaluate.CANDIDATE_SOURCE_REVISION
    exact = dict(evaluate._EXPECTED_CANDIDATE_CONFIG)
    assert exact["source_repo_id"] == "csukuangfj/sherpa-onnx-whisper-base.en"
    assert evaluate._bind_candidate_configs(
        ProvisionConfig(exact),
        ManifestConfig(exact),
        provision_type=ProvisionConfig,
        manifest_type=ManifestConfig,
        source_revision=revision,
    ) == evaluate._canonical_sha256(exact)
    changed = {**exact, "control_authority": True}
    with pytest.raises(evaluate.MobileAsrTwoPassEvalError):
        evaluate._bind_candidate_configs(
            ProvisionConfig(exact),
            ManifestConfig(changed),
            provision_type=ProvisionConfig,
            manifest_type=ManifestConfig,
            source_revision=revision,
        )
    with pytest.raises(evaluate.MobileAsrTwoPassEvalError):
        evaluate._bind_candidate_configs(
            ProvisionConfig({**exact, "source_revision": "a" * 40}),
            ManifestConfig({**exact, "source_revision": "a" * 40}),
            provision_type=ProvisionConfig,
            manifest_type=ManifestConfig,
            source_revision="a" * 40,
        )

    actual = MobileWhisperConfig()
    actual_value = actual.as_dict()
    assert provision.SOURCE_REVISION == evaluate.CANDIDATE_SOURCE_REVISION
    assert actual_value == evaluate._EXPECTED_CANDIDATE_CONFIG
    assert actual_value["tail_paddings"] == -1
    assert actual_value["maximum_tail_padding_samples"] == 0
    assert (
        evaluate._bind_candidate_configs(
            actual,
            actual,
            provision_type=provision.MobileWhisperConfig,
            manifest_type=MobileWhisperConfig,
            source_revision=provision.SOURCE_REVISION,
        )
        == evaluate.CANDIDATE_MOBILE_CONFIG_SHA256
    )
    assert evaluate.CANDIDATE_ARTIFACT_SET_SHA256 == MOBILE_WHISPER_ARTIFACT_SET_SHA256
    assert evaluate.CANDIDATE_MODEL_TOTAL_SIZE_BYTES == MOBILE_WHISPER_TOTAL_SIZE_BYTES


def test_candidate_venv_root_requires_exact_bin_python_shape(tmp_path: Path) -> None:
    candidate, venv_root, _site_packages = _candidate_model_with_venv(tmp_path)
    assert evaluate._candidate_venv_root(candidate.manifest) == venv_root

    malformed = venv_root / "bin" / "python3"
    malformed.write_bytes(b"wrong shape")
    candidate.manifest.python.path = malformed
    with pytest.raises(evaluate.MobileAsrTwoPassEvalError):
        evaluate._candidate_venv_root(candidate.manifest)


@pytest.mark.parametrize(
    ("entrypoint_name", "protected_destination"),
    [
        ("preflight_mobile_asr_two_pass_eval", "scratch"),
        ("preflight_mobile_asr_two_pass_eval", "output"),
        ("run_mobile_asr_two_pass_eval", "scratch"),
        ("run_mobile_asr_two_pass_eval", "output"),
    ],
)
def test_public_entrypoints_reject_destinations_inside_candidate_venv(
    tmp_path: Path,
    monkeypatch,
    entrypoint_name: str,
    protected_destination: str,
) -> None:
    candidate, _venv_root, site_packages = _candidate_model_with_venv(tmp_path)
    safe = tmp_path / "safe-private"
    safe.mkdir()
    scratch = (
        site_packages / "scratch"
        if protected_destination == "scratch"
        else safe / "scratch"
    )
    output = (
        site_packages / "report.json"
        if protected_destination == "output"
        else safe / "report.json"
    )

    inputs = SimpleNamespace()
    baseline_model = SimpleNamespace()
    monkeypatch.setattr(
        evaluate.baseline_api,
        "_load_production_inputs",
        lambda *_args, **_kwargs: inputs,
    )
    monkeypatch.setattr(
        evaluate.baseline_api,
        "_load_model_authority",
        lambda *_args, **_kwargs: baseline_model,
    )
    monkeypatch.setattr(
        evaluate,
        "_load_candidate_authority",
        lambda *_args, **_kwargs: candidate,
    )
    monkeypatch.setattr(
        evaluate.baseline_api,
        "_validate_destinations",
        lambda *_args, scratch_root, output_path, **_kwargs: (
            Path(scratch_root),
            Path(output_path),
            (1,),
            (2,),
        ),
    )

    entrypoint = getattr(evaluate, entrypoint_name)
    with pytest.raises(evaluate.MobileAsrTwoPassEvalError):
        entrypoint(
            tmp_path / "packet",
            packet_index_sha256="0" * 64,
            packet_receipt_sha256="1" * 64,
            baseline_worker_manifest=tmp_path / "baseline.json",
            candidate_worker_manifest=tmp_path / "candidate.json",
            scratch_root=scratch,
            output_path=output,
        )
    assert not scratch.exists()
    assert not output.exists()


def test_control_projection_matches_dart_normalization_quirks() -> None:
    cases = {
        " STOP!!! ": ("stop", "stop"),
        "cancel-that": ("none", "none"),
        "stop\ttalking": ("none", "none"),
        "don't": ("deny_lexeme", "deny_lexeme"),
        "YES": ("confirm_lexeme", "confirm_lexeme"),
        "research mode": ("mode_lexeme", "mode_lexeme:research"),
        "assistant   mode": ("mode_lexeme", "mode_lexeme:assistant"),
        "oprește": ("none", "none"),
        "ＳＴＯＰ": ("none", "none"),
        "": ("none", "none"),
    }
    for raw, expected in cases.items():
        assert evaluate._control_lexeme_projection(raw) == expected


def test_control_projection_sources_and_closure_are_exact() -> None:
    evaluate._verify_control_projection_sources()
    root = Path(evaluate.__file__).resolve().parents[1]
    for relative, expected in evaluate._CONTROL_SOURCE_SHA256.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
    assert "mobile/lib/contract.dart" in evaluate._EVALUATOR_FILES
    assert "mobile/lib/agent_decision.dart" in evaluate._EVALUATOR_FILES
    assert "tests/golden/commands.json" in evaluate._EVALUATOR_FILES
    assert (
        "tools/streaming_stt/mobile-whisper-base-en-v1.lock.json"
        in evaluate._EVALUATOR_FILES
    )


def test_epoch_payload_partitions_source_then_declared_zero_tail() -> None:
    raw = b"aaaabbbbccccdddd"
    leaf = _one_leaf_inputs(raw).components[0].leaves[0]
    assert (
        evaluate._epoch_payload(
            leaf, start_sample=0, end_sample=2, maximum_tail_samples=2
        )
        == b"aaaabbbb"
    )
    assert (
        evaluate._epoch_payload(
            leaf, start_sample=2, end_sample=6, maximum_tail_samples=2
        )
        == b"ccccdddd" + b"\x00" * 8
    )
    with pytest.raises(evaluate.MobileAsrTwoPassEvalError):
        evaluate._epoch_payload(
            leaf, start_sample=2, end_sample=7, maximum_tail_samples=2
        )


def test_epoch_files_preserve_boundaries_and_empty_text_still_gets_request(
    tmp_path: Path,
) -> None:
    inputs = _one_leaf_inputs(b"aaaabbbbccccdddd")
    observations = (
        MobileZipformerEndpointObservation("", 2, False),
        MobileZipformerEndpointObservation("final", 6, True),
    )
    evaluation = _evaluation(observations)
    root = tmp_path / "epochs"
    epochs = evaluate._write_epoch_files(root, inputs, (evaluation,))
    assert [(value.start_sample, value.end_sample) for value in epochs] == [
        (0, 2),
        (2, 6),
    ]
    assert len(epochs) == 2
    assert epochs[0].path.read_bytes() == b"aaaabbbb"
    assert epochs[1].path.read_bytes() == b"ccccdddd" + b"\x00" * 8
    evaluate._verify_epochs(root, inputs, (evaluation,), epochs)


def test_zero_endpoint_evaluation_creates_no_candidate_request(tmp_path: Path) -> None:
    inputs = _one_leaf_inputs(b"aaaa")
    evaluation = _evaluation(())
    epochs = evaluate._write_epoch_files(tmp_path / "epochs", inputs, (evaluation,))
    assert epochs == ()
    evaluate._verify_epochs(tmp_path / "epochs", inputs, (evaluation,), epochs)


def test_candidate_root_advances_baseline_scratch_snapshot(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    inputs = baseline._synthetic_inputs()
    _worker_root, paths, bundle, original = baseline._prepare_scratch(scratch, inputs)
    evaluations = tuple(
        _evaluation((MobileZipformerEndpointObservation("", 1, True),))
        for _component in inputs.components
        for _leaf in _component.leaves
    )
    epoch_root = scratch / "candidate-worker"
    epochs = evaluate._write_epoch_files(epoch_root, inputs, evaluations)

    with pytest.raises(baseline.MobileAsrEvidenceEvalError):
        baseline._verify_scratch(scratch, original, bundle, paths, inputs)

    advanced = evaluate._advance_scratch_snapshot_for_candidate(scratch, original)
    assert advanced[:5] == original[:5]
    assert advanced[5] == original[5] + 1
    baseline._verify_scratch(scratch, advanced, bundle, paths, inputs)
    evaluate._verify_two_pass_scratch_snapshot(scratch, advanced)
    evaluate._verify_epochs(epoch_root, inputs, evaluations, epochs)


def _advanced_scratch_layout(tmp_path: Path) -> tuple[Path, tuple[int, ...]]:
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    (scratch / "source-bundle").mkdir(mode=0o700)
    (scratch / "worker").mkdir(mode=0o700)
    original = evaluate._snapshot(scratch.lstat())
    (scratch / "candidate-worker").mkdir(mode=0o700)
    return (
        scratch,
        evaluate._advance_scratch_snapshot_for_candidate(scratch, original),
    )


def test_two_pass_scratch_snapshot_rejects_extra_child(tmp_path: Path) -> None:
    scratch, advanced = _advanced_scratch_layout(tmp_path)
    (scratch / "unexpected").write_bytes(b"unexpected")
    with pytest.raises(evaluate.MobileAsrTwoPassEvalError):
        evaluate._verify_two_pass_scratch_snapshot(scratch, advanced)


def test_two_pass_scratch_transition_rejects_extra_regular_child(
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    (scratch / "source-bundle").mkdir(mode=0o700)
    (scratch / "worker").mkdir(mode=0o700)
    original = evaluate._snapshot(scratch.lstat())
    (scratch / "candidate-worker").mkdir(mode=0o700)
    (scratch / "unexpected").write_bytes(b"unexpected")
    assert evaluate._snapshot(scratch.lstat())[5] == original[5] + 1
    with pytest.raises(evaluate.MobileAsrTwoPassEvalError):
        evaluate._advance_scratch_snapshot_for_candidate(scratch, original)


def test_two_pass_scratch_transition_rejects_extra_directory(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    (scratch / "source-bundle").mkdir(mode=0o700)
    (scratch / "worker").mkdir(mode=0o700)
    original = evaluate._snapshot(scratch.lstat())
    (scratch / "candidate-worker").mkdir(mode=0o700)
    (scratch / "unexpected").mkdir(mode=0o700)
    assert evaluate._snapshot(scratch.lstat())[5] == original[5] + 2
    with pytest.raises(evaluate.MobileAsrTwoPassEvalError):
        evaluate._advance_scratch_snapshot_for_candidate(scratch, original)


def test_two_pass_scratch_snapshot_rejects_root_replacement(tmp_path: Path) -> None:
    scratch, advanced = _advanced_scratch_layout(tmp_path)
    scratch.rename(tmp_path / "original")
    replacement = tmp_path / "replacement"
    replacement.mkdir(mode=0o700)
    for name in evaluate._TWO_PASS_SCRATCH_CHILDREN:
        (replacement / name).mkdir(mode=0o700)
    replacement.rename(scratch)
    with pytest.raises(evaluate.MobileAsrTwoPassEvalError):
        evaluate._verify_two_pass_scratch_snapshot(scratch, advanced)


def test_two_pass_scratch_snapshot_rejects_root_mode_tamper(tmp_path: Path) -> None:
    scratch, advanced = _advanced_scratch_layout(tmp_path)
    scratch.chmod(0o755)
    with pytest.raises(evaluate.MobileAsrTwoPassEvalError):
        evaluate._verify_two_pass_scratch_snapshot(scratch, advanced)


def test_two_pass_scratch_snapshot_rejects_child_mode_tamper(tmp_path: Path) -> None:
    scratch, advanced = _advanced_scratch_layout(tmp_path)
    (scratch / "candidate-worker").chmod(0o755)
    with pytest.raises(evaluate.MobileAsrTwoPassEvalError):
        evaluate._verify_two_pass_scratch_snapshot(scratch, advanced)


def test_two_pass_scratch_snapshot_rejects_symlink_child(tmp_path: Path) -> None:
    scratch, advanced = _advanced_scratch_layout(tmp_path)
    candidate = scratch / "candidate-worker"
    candidate.rename(tmp_path / "candidate-real")
    candidate.symlink_to(tmp_path / "candidate-real", target_is_directory=True)
    with pytest.raises(evaluate.MobileAsrTwoPassEvalError):
        evaluate._verify_two_pass_scratch_snapshot(scratch, advanced)


def test_candidate_outcomes_preserve_epoch_count_order_and_boundaries() -> None:
    first = _evaluation(
        (
            MobileZipformerEndpointObservation("", 1, False),
            MobileZipformerEndpointObservation("baseline", 2, True),
        )
    )
    second = _evaluation(())
    outcomes = evaluate._candidate_outcomes((first, second), ("new", ""))
    assert outcomes[0].endpoint_texts == ("new", "")
    assert outcomes[0].endpoint_source_complete == (False, True)
    assert outcomes[1].endpoint_texts == ()
    assert outcomes[1].endpoint_reason == "tail_exhausted"
    with pytest.raises(evaluate.MobileAsrTwoPassEvalError):
        evaluate._candidate_outcomes((first, second), ("one",))


def test_baseline_epoch_geometry_digest_binds_order_and_boundaries_only() -> None:
    first = _evaluation(
        (
            MobileZipformerEndpointObservation("private one", 1, False),
            MobileZipformerEndpointObservation("private two", 2, True),
        )
    )
    same_geometry_new_text = _evaluation(
        (
            MobileZipformerEndpointObservation("different", 1, False),
            MobileZipformerEndpointObservation("content", 2, True),
        )
    )
    changed_boundary = _evaluation(
        (
            MobileZipformerEndpointObservation("private one", 1, False),
            MobileZipformerEndpointObservation("private two", 3, True),
        )
    )
    changed_source_complete = _evaluation(
        (
            MobileZipformerEndpointObservation("private one", 1, True),
            MobileZipformerEndpointObservation("private two", 2, True),
        )
    )
    other_source = _evaluation((MobileZipformerEndpointObservation("other", 4, True),))
    assert evaluate._baseline_endpoint_epoch_geometry((first,)) == (
        evaluate._baseline_endpoint_epoch_geometry((same_geometry_new_text,))
    )
    assert evaluate._baseline_endpoint_epoch_geometry((first,))[0] == 2
    assert evaluate._baseline_endpoint_epoch_geometry((first,)) != (
        evaluate._baseline_endpoint_epoch_geometry((changed_boundary,))
    )
    assert evaluate._baseline_endpoint_epoch_geometry((first,)) != (
        evaluate._baseline_endpoint_epoch_geometry((changed_source_complete,))
    )
    assert evaluate._baseline_endpoint_epoch_geometry((first, other_source)) != (
        evaluate._baseline_endpoint_epoch_geometry((other_source, first))
    )


def test_candidate_terminal_is_final_only_complete_pcm() -> None:
    epoch = evaluate._Epoch(
        leaf_index=0,
        observation_index=0,
        start_sample=0,
        end_sample=160,
        samples=160,
        sha256="0" * 64,
        path=Path("/private/epoch.f32le"),
    )
    resources = ResourceUsage(rss_mb=1.0, threads=1, vram_mb=None)
    final = FinalEvent(
        request_id="candidate-000",
        seq=0,
        text="",
        samples_seen=160,
        elapsed_ms=1.0,
        finalization_ms=0.1,
        compute_ms=0.5,
        audio_seconds=0.01,
        chunks=1,
        deadline_misses=0,
        max_backlog_ms=0.0,
        resources=resources,
        model_padding_samples=0,
    )
    assert (
        evaluate._extract_candidate_text(
            SimpleNamespace(partials=(), final=final), epoch, "candidate-000"
        )
        == ""
    )
    with pytest.raises(evaluate.MobileAsrTwoPassEvalError):
        evaluate._extract_candidate_text(
            SimpleNamespace(partials=(object(),), final=final),
            epoch,
            "candidate-000",
        )
    # This protocol field covers evaluator/adapter alignment PCM, not opaque
    # model-internal behavior controlled by the separately bound tail_paddings.
    with pytest.raises(evaluate.MobileAsrTwoPassEvalError):
        evaluate._extract_candidate_text(
            SimpleNamespace(partials=(), final=replace(final, model_padding_samples=1)),
            epoch,
            "candidate-000",
        )


def test_all_epoch_control_lexeme_transitions_are_aggregate_only() -> None:
    inputs = baseline._synthetic_inputs()
    base = _outputs()
    candidate = _outputs()
    base[0] = baseline._EndpointOutcome(
        ("stop", "yes", "no", "passive mode", ""),
        (False, False, False, False, True),
        "endpoint",
        True,
    )
    candidate[0] = baseline._EndpointOutcome(
        ("hello", "confirm", "research mode", "assistant mode", "cancel"),
        (False, False, False, False, True),
        "endpoint",
        True,
    )
    result = evaluate._comparison(inputs, base, candidate)
    control = result["control_lexeme_projection"]
    assert control["endpoint_epochs"] == 133
    assert control["actual_agent_decision_authority"] is False
    assert control["pending_confirmation_state_applied"] is False
    assert control["mode_alias_disagreements"] == 1
    assert control["candidate_gained_control_lexeme_epochs"] == 1
    assert control["candidate_lost_control_lexeme_epochs"] == 1
    assert control["changed_nonempty_control_class_epochs"] == 1
    matrix = {
        row["baseline_class"]: row["candidate_counts"]
        for row in control["transition_matrix"]
    }
    assert matrix["stop"]["none"] == 1
    assert matrix["none"]["stop"] == 1
    assert matrix["deny_lexeme"]["mode_lexeme"] == 1


def test_synthetic_report_is_private_unpooled_and_has_zero_candidate_authority(
    tmp_path: Path,
) -> None:
    loaded = _run(tmp_path)
    report = loaded.value
    assert loaded.path.stat().st_mode & 0o777 == 0o600
    assert loaded.path.stat().st_nlink == 1
    assert report["production_inputs"] is False
    assert report["candidate"]["authority"] == evaluate._CANDIDATE_AUTHORITY
    assert report["bindings"]["current_rerun_endpoint_epoch_geometry_count"] == 129
    assert len(report["bindings"]["current_rerun_endpoint_epoch_geometry_sha256"]) == 64
    assert report["execution"]["candidate_epoch_geometry"] == {
        "baseline_endpoint_observations": 129,
        "candidate_app_admissions": 0,
        "candidate_requests": 129,
        "candidate_worker_lifetime_pcm_limit_bytes": 65_011_712,
        "derived_epoch_pcm_bytes_total": 516,
        "derived_epoch_samples_total": 129,
        "empty_baseline_epochs_create_requests": True,
        "maximum_derived_epoch_pcm_bytes_total": 64_067_500,
        "maximum_derived_epoch_samples_total": 16_016_875,
        "preserves_baseline_order_and_boundaries": True,
        "reported_adapter_or_evaluator_appended_pcm_padding_samples": 0,
        "sherpa_internal_tail_padding_observation": "unobserved-and-unreported",
        "source_evaluations": 129,
    }
    assert report["metric_aggregation"]["pooled_metric"] is None
    assert len(report["baseline"]["components"]) == 5
    assert len(report["candidate"]["components"]) == 5
    raw = loaded.path.read_text(encoding="ascii")
    for forbidden in (
        '"case_id"',
        '"endpoint_text"',
        '"hypothesis"',
        '"path"',
        '"reference"',
        '"request_id"',
        '"selected_text"',
        '"transcript"',
    ):
        assert forbidden not in raw


def test_candidate_component_accounting_is_per_component_and_cross_bound(
    tmp_path: Path,
) -> None:
    loaded = _run(tmp_path)
    baseline_rows = loaded.value["baseline"]["components"]
    candidate_rows = loaded.value["candidate"]["components"]
    for baseline_row, candidate_row in zip(baseline_rows, candidate_rows, strict=True):
        accounting = candidate_row["epoch_text_accounting"]
        endpoint = baseline_row["endpoint_integrity"]
        assert (
            accounting["inherited_endpoint_epochs"] == endpoint["committed_endpoints"]
        )
        assert accounting["source_evaluations"] == endpoint["evaluations"]
        assert (
            accounting["empty_candidate_text_epochs"]
            + accounting["nonempty_candidate_text_epochs"]
            == accounting["inherited_endpoint_epochs"]
        )


@pytest.mark.parametrize("raw_nonempty_side", ("baseline", "candidate"))
@pytest.mark.parametrize(
    "token_empty_text",
    ("...", " \t ", "\u00e9"),
    ids=("punctuation-only", "whitespace-only", "non-ascii-only"),
)
def test_token_empty_raw_nonempty_epoch_is_published_with_literal_empty_contract(
    tmp_path: Path,
    raw_nonempty_side: str,
    token_empty_text: str,
) -> None:
    outputs = _outputs()
    candidate = list(_candidate_texts(outputs))
    if raw_nonempty_side == "baseline":
        index = next(
            ordinal
            for ordinal, value in enumerate(candidate)
            if evaluate.normalize(value)
        )
        original = outputs[index]
        outputs[index] = baseline._EndpointOutcome(
            (token_empty_text,),
            original.endpoint_source_complete,
            original.endpoint_reason,
            original.source_complete,
        )
        candidate[index] = ""
    else:
        index = next(ordinal for ordinal, value in enumerate(candidate) if value == "")
        candidate[index] = token_empty_text

    loaded = _run(tmp_path, outputs, tuple(candidate))
    report = loaded.value
    comparison = report["comparison"]
    baseline_texts = _candidate_texts(outputs)
    accounting = [
        row["epoch_text_accounting"] for row in report["candidate"]["components"]
    ]

    assert token_empty_text != ""
    assert not evaluate.normalize(token_empty_text)
    assert comparison["baseline_empty_epochs"] == sum(
        value == "" for value in baseline_texts
    )
    assert comparison["candidate_empty_epochs"] == sum(
        value == "" for value in candidate
    )
    assert comparison["candidate_empty_on_nonempty_baseline_epochs"] == (
        raw_nonempty_side == "baseline"
    )
    assert comparison["candidate_nonempty_on_empty_baseline_epochs"] == (
        raw_nonempty_side == "candidate"
    )
    assert (
        comparison["exact_normalized_epoch_agreements"] == comparison["endpoint_epochs"]
    )
    assert (
        sum(row["empty_candidate_text_epochs"] for row in accounting)
        == comparison["candidate_empty_epochs"]
    )
    assert sum(row["nonempty_candidate_text_epochs"] for row in accounting) == (
        comparison["endpoint_epochs"] - comparison["candidate_empty_epochs"]
    )
    assert json.dumps(token_empty_text, ensure_ascii=True) not in loaded.path.read_text(
        encoding="ascii"
    )


def test_literal_empty_transitions_are_distinct_from_normalized_agreement(
    tmp_path: Path,
) -> None:
    outputs = _outputs()
    original = outputs[0]
    outputs[0] = baseline._EndpointOutcome(
        ("", ".", ".", "hello"),
        (False, False, False, True),
        original.endpoint_reason,
        original.source_complete,
    )
    candidate = list(_candidate_texts(outputs))
    candidate[:4] = (".", "", "...", ".")

    loaded = _run(tmp_path, outputs, tuple(candidate))
    comparison = loaded.value["comparison"]

    assert comparison["candidate_empty_on_nonempty_baseline_epochs"] == 1
    assert comparison["candidate_nonempty_on_empty_baseline_epochs"] == 1
    assert comparison["exact_normalized_epoch_agreements"] == (
        comparison["endpoint_epochs"] - 1
    )


def test_empty_baseline_epoch_can_gain_candidate_text_and_is_counted(
    tmp_path: Path,
) -> None:
    outputs = _outputs()
    outputs[0] = baseline._EndpointOutcome(("",), (True,), "endpoint", True)
    candidate = list(_candidate_texts(outputs))
    candidate[0] = "candidate revision"
    loaded = _run(tmp_path, outputs, tuple(candidate))
    comparison = loaded.value["comparison"]
    assert comparison["candidate_nonempty_on_empty_baseline_epochs"] == 1
    assert comparison["baseline_empty_epochs"] >= 1


def test_report_is_no_clobber_and_strict_loader_rejects_tamper(
    tmp_path: Path,
) -> None:
    loaded = _run(tmp_path)
    with pytest.raises(
        (evaluate.MobileAsrTwoPassEvalError, baseline.MobileAsrEvidenceEvalError)
    ):
        evaluate._run_synthetic_test_evaluation(
            tuple(_outputs()), _candidate_texts(_outputs()), loaded.path
        )
    raw = loaded.path.read_bytes()
    loaded.path.write_bytes(raw[:-1] + b" ")
    loaded.path.chmod(0o600)
    with pytest.raises(evaluate.MobileAsrTwoPassEvalError):
        evaluate._load_report_common(
            loaded.path,
            expected_sha256=loaded.digest,
            production_inputs=False,
        )


def test_two_pass_publication_rejects_linked_not_durable_report(
    tmp_path: Path, monkeypatch
) -> None:
    output_parent = _private(tmp_path / "report-parent")
    output = output_parent / "report.json"
    real_fsync = os.fsync

    def fail_directory_fsync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("synthetic directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(baseline.os, "fsync", fail_directory_fsync)
    outputs = _outputs()
    with pytest.raises(
        (evaluate.MobileAsrTwoPassEvalError, baseline.MobileAsrEvidenceEvalError)
    ):
        evaluate._run_synthetic_test_evaluation(
            tuple(outputs), _candidate_texts(outputs), output
        )
    assert output.exists()

    with pytest.raises(
        (evaluate.MobileAsrTwoPassEvalError, baseline.MobileAsrEvidenceEvalError)
    ):
        evaluate._run_synthetic_test_evaluation(
            tuple(outputs), _candidate_texts(outputs), output
        )


def test_two_pass_publication_recovers_only_after_durable_link(
    tmp_path: Path, monkeypatch
) -> None:
    real_link = os.link

    def link_then_raise(*args, **kwargs):
        real_link(*args, **kwargs)
        raise OSError("synthetic post-link interruption")

    monkeypatch.setattr(baseline.os, "link", link_then_raise)
    loaded = _run(tmp_path)
    assert loaded.path.exists()
    assert hashlib.sha256(loaded.path.read_bytes()).hexdigest() == loaded.digest


def test_loader_rejects_mutation_between_initial_and_closing_reads(
    tmp_path: Path, monkeypatch
) -> None:
    loaded = _run(tmp_path)
    real_read = evaluate.read_regular_bounded
    calls = 0

    def mutate_on_closing_read(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = real_read(*args, **kwargs)
        if calls == 2:
            return SimpleNamespace(path=result.path, data=result.data[:-1] + b" ")
        return result

    monkeypatch.setattr(evaluate, "read_regular_bounded", mutate_on_closing_read)
    with pytest.raises(evaluate.MobileAsrTwoPassEvalError):
        evaluate._load_report_common(
            loaded.path,
            expected_sha256=loaded.digest,
            production_inputs=False,
        )
    assert calls == 2


def test_rebound_authority_accounting_and_matrix_tamper_are_rejected(
    tmp_path: Path,
) -> None:
    loaded = _run(tmp_path)
    base = json.loads(loaded.path.read_text(encoding="ascii"))
    mutations = []

    authority = json.loads(json.dumps(base))
    authority["candidate"]["authority"]["control_authority"] = True
    mutations.append(authority)

    accounting = json.loads(json.dumps(base))
    accounting["candidate"]["components"][0]["epoch_text_accounting"][
        "inherited_endpoint_epochs"
    ] += 1
    mutations.append(accounting)

    matrix = json.loads(json.dumps(base))
    matrix["comparison"]["control_lexeme_projection"]["transition_matrix"][0][
        "candidate_counts"
    ]["none"] += 1
    mutations.append(matrix)

    empty_pair = json.loads(json.dumps(base))
    empty_pair["comparison"]["candidate_empty_epochs"] += 1
    mutations.append(empty_pair)

    baseline_empty_pair = json.loads(json.dumps(base))
    baseline_empty_pair["comparison"]["baseline_empty_epochs"] += 1
    mutations.append(baseline_empty_pair)

    candidate_accounting_pair = json.loads(json.dumps(base))
    candidate_accounting = candidate_accounting_pair["candidate"]["components"][1][
        "epoch_text_accounting"
    ]
    candidate_accounting["empty_candidate_text_epochs"] += 1
    candidate_accounting["nonempty_candidate_text_epochs"] -= 1
    mutations.append(candidate_accounting_pair)

    padding_scope = json.loads(json.dumps(base))
    padding_scope["execution"]["candidate_epoch_geometry"][
        "sherpa_internal_tail_padding_observation"
    ] = "observed-zero"
    mutations.append(padding_scope)

    target_hits = json.loads(json.dumps(base))
    target_metrics = target_hits["candidate"]["components"][0]["metrics"]
    target_delta = 1 if target_metrics["command_hits"] == 0 else -1
    target_metrics["command_hits"] += target_delta
    target_metrics["command_recall"] = (
        target_metrics["command_hits"] / target_metrics["command_attempts"]
    )
    mutations.append(target_hits)

    forbidden_hits = json.loads(json.dumps(base))
    forbidden_metrics = forbidden_hits["candidate"]["components"][0]["metrics"]
    forbidden_delta = (
        1
        if forbidden_metrics["forbidden_target_hits"]
        < forbidden_metrics["forbidden_target_attempts"]
        else -1
    )
    forbidden_metrics["forbidden_target_hits"] += forbidden_delta
    forbidden_metrics["forbidden_target_false_positive_rate"] = (
        forbidden_metrics["forbidden_target_hits"]
        / forbidden_metrics["forbidden_target_attempts"]
    )
    mutations.append(forbidden_hits)

    for value in mutations:
        unsigned = dict(value)
        unsigned.pop("binding_sha256")
        value["binding_sha256"] = evaluate._canonical_sha256(unsigned)
        with pytest.raises(evaluate.MobileAsrTwoPassEvalError):
            evaluate._validate_report(value, production_inputs=False)


def test_public_api_has_no_candidate_authority_or_runtime_overrides() -> None:
    assert set(inspect.signature(evaluate.run_mobile_asr_two_pass_eval).parameters) == {
        "packet_root",
        "packet_index_sha256",
        "packet_receipt_sha256",
        "baseline_worker_manifest",
        "candidate_worker_manifest",
        "scratch_root",
        "output_path",
    }
    actions = {action.dest for action in evaluate._parser()._actions}
    assert not actions.intersection(
        {
            "chunk_samples",
            "partial_interval_ms",
            "provider",
            "repeats",
            "synthetic",
            "tail_padding_samples",
            "threads",
        }
    )
    source = Path(evaluate.__file__).read_text(encoding="utf-8")
    assert "AgentSession" not in source
    baseline_close = source.index("baseline_worker.close()")
    accepted = source.index("_require_accepted_baseline(", baseline_close)
    old_scratch_verify = source.index("baseline_api._verify_scratch(", accepted)
    write_epochs = source.index("epochs = _write_epoch_files(", old_scratch_verify)
    advance_snapshot = source.index(
        "scratch_snapshot = _advance_scratch_snapshot_for_candidate(", write_epochs
    )
    new_scratch_verify = source.index("baseline_api._verify_scratch(", advance_snapshot)
    new_layout_verify = source.index(
        "_verify_two_pass_scratch_snapshot(scratch, scratch_snapshot)",
        new_scratch_verify,
    )
    verify_epochs = source.index("_verify_epochs(", new_layout_verify)
    candidate_construct = source.index(
        "candidate_worker = StreamingWorker(", verify_epochs
    )
    guard = source.index("def guard() -> None:", candidate_construct)
    guard_layout_verify = source.index(
        "_verify_two_pass_scratch_snapshot(scratch, scratch_snapshot)", guard
    )
    guard_scratch_verify = source.index(
        "baseline_api._verify_scratch(", guard_layout_verify
    )
    guard_epoch_verify = source.index("_verify_epochs(", guard_scratch_verify)
    publish = source.index("baseline_api._publish_report(", guard_epoch_verify)
    assert (
        baseline_close
        < accepted
        < old_scratch_verify
        < write_epochs
        < advance_snapshot
        < new_scratch_verify
        < new_layout_verify
        < verify_epochs
        < candidate_construct
        < guard
        < guard_layout_verify
        < guard_scratch_verify
        < guard_epoch_verify
        < publish
    )


def test_cli_stdout_failure_is_single_attempt_and_detail_free(monkeypatch) -> None:
    argv = [
        "--packet-root",
        "/private/packet",
        "--packet-index-sha256",
        "0" * 64,
        "--packet-receipt-sha256",
        "1" * 64,
        "--baseline-worker-manifest",
        "/private/baseline.json",
        "--candidate-worker-manifest",
        "/private/candidate.json",
        "--scratch-root",
        "/private/scratch",
        "--output",
        "/private/report.json",
    ]

    class BrokenBuffer:
        def __init__(self) -> None:
            self.attempts = 0

        def write(self, _raw: bytes) -> int:
            self.attempts += 1
            raise BrokenPipeError("private stdout detail")

        def flush(self) -> None:
            raise AssertionError("flush must not follow a failed write")

    broken = BrokenBuffer()

    monkeypatch.setattr(
        evaluate,
        "run_mobile_asr_two_pass_eval",
        lambda *_args, **_kwargs: SimpleNamespace(digest="2" * 64),
    )
    monkeypatch.setattr(
        evaluate,
        "sys",
        SimpleNamespace(stdout=SimpleNamespace(buffer=broken)),
    )
    assert evaluate.main(argv) == 2
    assert broken.attempts == 1

    def fail_run(*_args, **_kwargs):
        raise evaluate.MobileAsrTwoPassEvalError("private evaluator detail")

    monkeypatch.setattr(evaluate, "run_mobile_asr_two_pass_eval", fail_run)
    assert evaluate.main(argv) == 2
    assert broken.attempts == 2


def test_cli_failures_are_detail_free(tmp_path: Path, capsys) -> None:
    result = evaluate.main(
        [
            "--preflight",
            "--packet-root",
            str(tmp_path / "private-missing-packet"),
            "--packet-index-sha256",
            baseline.PACKET_INDEX_SHA256,
            "--packet-receipt-sha256",
            baseline.PACKET_RECEIPT_SHA256,
            "--baseline-worker-manifest",
            str(tmp_path / "private-baseline"),
            "--candidate-worker-manifest",
            str(tmp_path / "private-candidate"),
            "--scratch-root",
            str(tmp_path / "private-scratch"),
            "--output",
            str(tmp_path / "private-report"),
        ]
    )
    captured = capsys.readouterr()
    assert result == 2
    assert json.loads(captured.out) == evaluate._SAFE_ERROR
    assert captured.err == ""
    assert "private-missing-packet" not in captured.out

    assert evaluate.main(["--unknown", "/private/do-not-echo"]) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out) == evaluate._SAFE_ERROR
    assert "/private/do-not-echo" not in captured.out


def test_cli_closed_stdout_pipe_is_controlled_and_silent() -> None:
    read_descriptor, write_descriptor = os.pipe()
    os.close(read_descriptor)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "tools.mobile_asr_two_pass_eval",
                "--unknown",
                "/private/do-not-echo",
            ],
            cwd=Path(evaluate.__file__).resolve().parents[1],
            stdin=subprocess.DEVNULL,
            stdout=write_descriptor,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10,
        )
    finally:
        os.close(write_descriptor)
    assert completed.returncode == 2
    assert completed.stderr == b""
