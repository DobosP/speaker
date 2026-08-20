from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools import mobile_asr_evidence_eval as evaluate


def _private(path: Path) -> Path:
    path.mkdir(mode=0o700)
    path.chmod(0o700)
    return path


def _outputs() -> list[evaluate._EndpointOutcome]:
    rows: list[evaluate._EndpointOutcome] = []
    for component in evaluate._synthetic_inputs().components:
        for leaf in component.leaves:
            if leaf.kind == "mix":
                text = f"{leaf.reference_a} {leaf.reference_b}"
            else:
                text = leaf.expected_text
            if text:
                rows.append(
                    evaluate._EndpointOutcome(
                        (text,),
                        (True,),
                        "endpoint",
                        True,
                    )
                )
            else:
                rows.append(evaluate._EndpointOutcome((), (), "tail_exhausted", True))
    assert len(rows) == 129
    return rows


def _run(tmp_path: Path, rows=None):
    output_parent = _private(tmp_path / "reports")
    selected = _outputs() if rows is None else rows
    return evaluate._run_synthetic_test_evaluation(
        evaluate._SyntheticWorkerContract(tuple(selected)),
        output_parent / "report.json",
    )


def _component(report, name: str):
    return next(row for row in report.value["components"] if row["component"] == name)


def _artifact(name: str, raw: bytes):
    return evaluate.packet_api._Artifact(
        name=name,
        payload=raw,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _snapshot_component(base, root: Path, manifest_value, pcm):
    manifest_raw = evaluate._canonical_json(manifest_value, newline=True)
    receipt_raw = b'{"synthetic_test_only":true}\n'
    spec = replace(
        base,
        manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        receipt_sha256=hashlib.sha256(receipt_raw).hexdigest(),
        pcm_bytes=sum(len(raw) for _name, raw in pcm),
    )
    artifacts = (
        _artifact(spec.manifest_file, manifest_raw),
        _artifact(spec.receipt_file, receipt_raw),
        *(_artifact(name, raw) for name, raw in pcm),
    )
    return spec, evaluate.packet_api._ValidatedComponent(
        spec=spec,
        root=root,
        artifacts=artifacts,
        inventory=(),
    )


def _isolated_snapshot(root: Path):
    base = next(
        row
        for row in evaluate.packet_api.load_packet_lock().components
        if row.component_id == "primock-isolated"
    )
    pcm = tuple(
        (f"primock57-isolated-{index:02d}.f32le", bytes((index, 0, 0, 0)))
        for index in range(3)
    )
    cases = [
        {
            "assertion": "transcript",
            "commands": [],
            "expected_text": f"private-reference-{index}",
            "file": name,
            "id": f"primock57-isolated-{index:02d}",
            "samples": 1,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "tags": ["primock57", "isolated", "role-a"],
        }
        for index, (name, raw) in enumerate(pcm)
    ]
    return _snapshot_component(
        base,
        root,
        {
            "cases": cases,
            "provenance": {"synthetic_test_only": True},
            "purpose": "synthetic-test-only",
            "schema_version": 2,
        },
        pcm,
    )


def _overlap_snapshot(root: Path):
    base = next(
        row
        for row in evaluate.packet_api.load_packet_lock().components
        if row.component_id == "primock-overlap"
    )
    pcm = []
    cases = []
    reference_a = "private-role-a-reference"
    reference_b = "private-role-b-reference"
    for index in range(3):
        rows = {}
        for kind in ("role-a", "role-b", "mix"):
            name = f"primock57-overlap-{index:02d}-{kind}.f32le"
            raw = bytes((index + 1, len(kind), 0, 0))
            pcm.append((name, raw))
            rows[kind] = {
                "bytes": 4,
                "file": name,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        role_a = {
            **rows["role-a"],
            "activity_end_sample": 1,
            "activity_start_sample": 0,
            "interval_index": index + 1,
            "reference": reference_a,
            "reference_sha256": hashlib.sha256(reference_a.encode("utf-8")).hexdigest(),
        }
        role_b = {
            **rows["role-b"],
            "activity_end_sample": 1,
            "activity_start_sample": 0,
            "interval_index": index + 1,
            "reference": reference_b,
            "reference_sha256": hashlib.sha256(reference_b.encode("utf-8")).hexdigest(),
        }
        cases.append(
            {
                "case_id": f"primock57-overlap-{index:02d}",
                "envelope": {
                    "samples": 1,
                    "source_end_sample": index + 1,
                    "source_start_sample": index,
                },
                "mix": {
                    **rows["mix"],
                    "arithmetic": "float32-add-then-multiply-float32-0.5-v1",
                },
                "overlap": {
                    "relative_end_sample": 1,
                    "relative_start_sample": 0,
                    "samples": 1,
                },
                "role_a": role_a,
                "role_b": role_b,
            }
        )
    return _snapshot_component(
        base,
        root,
        {
            "cases": cases,
            "evidence_scope": {"synthetic_test_only": True},
            "fixture_id": "synthetic-test-only",
            "kind": "primock57-two-role-overlap-diagnostic-v1",
            "lock_recipe_sha256": "0" * 64,
            "production_evidence": True,
            "sample_format": {"synthetic_test_only": True},
            "schema_version": 1,
            "source_contract_sha256": "0" * 64,
        },
        tuple(pcm),
    )


def test_fixed_packet_v2_and_mobile_model_authority() -> None:
    assert evaluate.PACKET_SCHEMA_VERSION == 2
    assert evaluate.PACKET_ID == "mobile-asr-english-five-component-v2"
    assert (
        evaluate.PACKET_LOCK_SHA256
        == "c81a89a7a7df534ec0a23dc20c0a915c4aa5eedec6270a13ec08456a0683cc11"
    )
    assert (
        evaluate.PACKET_INDEX_SHA256
        == "232e4230d16b0014baff38e2daf4156aaef3c4e129541079248fac0b2f9e835c"
    )
    assert (
        evaluate.PACKET_RECEIPT_SHA256
        == "36eec947383069fb15bc998209ae4aaf76ea82f7031666b972fe98f8c7c5f51f"
    )
    assert evaluate.MODEL_TOTAL_SIZE_BYTES == 74_207_237
    assert (
        evaluate.MODEL_ARTIFACT_SET_SHA256
        == "eebca4036773b78642c846acdc966d1bab8f377a8b8ab632f9d3aac7ab38bcc5"
    )
    assert (
        evaluate.MOBILE_CONFIG_SHA256
        == "749fa50525a08fc612bc82fb4e05b4fbfa62f3e762b621acceaf05f9223fa883"
    )


def test_distinct_config_types_bind_only_when_closed_mappings_match() -> None:
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

    exact = dict(evaluate._EXPECTED_MOBILE_CONFIG)
    assert (
        evaluate._bind_mobile_configs(
            ProvisionConfig(exact),
            ManifestConfig(exact),
            provision_type=ProvisionConfig,
            manifest_type=ManifestConfig,
        )
        == evaluate.MOBILE_CONFIG_SHA256
    )
    mismatched = {**exact, "provider": "cuda"}
    with pytest.raises(evaluate.MobileAsrEvidenceEvalError):
        evaluate._bind_mobile_configs(
            ProvisionConfig(exact),
            ManifestConfig(mismatched),
            provision_type=ProvisionConfig,
            manifest_type=ManifestConfig,
        )

    from tools.provision_mobile_zipformer import (
        MobileZipformerConfig as ProvisionMobileZipformerConfig,
    )
    from tools.streaming_stt.manifest import (
        MobileZipformerConfig as WorkerMobileZipformerConfig,
    )

    assert (
        evaluate._bind_mobile_configs(
            ProvisionMobileZipformerConfig(**exact),
            WorkerMobileZipformerConfig(),
            provision_type=ProvisionMobileZipformerConfig,
            manifest_type=WorkerMobileZipformerConfig,
        )
        == evaluate.MOBILE_CONFIG_SHA256
    )


def test_public_api_has_no_synthetic_or_runtime_override() -> None:
    parameters = inspect.signature(evaluate.run_mobile_asr_evidence_eval).parameters
    assert set(parameters) == {
        "packet_root",
        "packet_index_sha256",
        "packet_receipt_sha256",
        "worker_manifest",
        "scratch_root",
        "output_path",
    }
    actions = {action.dest for action in evaluate._parser()._actions}
    assert not actions.intersection(
        {
            "provider",
            "threads",
            "pace",
            "chunk_samples",
            "tail_padding_samples",
            "partial_interval_ms",
            "repeats",
            "synthetic",
        }
    )
    source = Path(evaluate.__file__).read_text(encoding="utf-8")
    assert "load_primock57_" not in source


def test_exact_packet_snapshot_selector_requires_type_spec_root_and_order(
    tmp_path: Path,
) -> None:
    spec, component = _isolated_snapshot(tmp_path / "isolated")
    lock = evaluate.packet_api.PacketLock(
        raw_sha256=evaluate.PACKET_LOCK_SHA256,
        recipe_sha256="0" * 64,
        components=(spec,),
    )
    packet = evaluate.packet_api.LoadedMobileAsrEvidencePacket(
        schema_version=2,
        packet_id=evaluate.PACKET_ID,
        packet_lock_sha256=evaluate.PACKET_LOCK_SHA256,
        packet_index_sha256=evaluate.PACKET_INDEX_SHA256,
        packet_receipt_sha256=evaluate.PACKET_RECEIPT_SHA256,
        components=1,
        logical_cases=3,
        pcm_inputs=3,
        pcm_bytes=spec.pcm_bytes,
        samples=3,
        root=tmp_path,
        _validated_components=(component,),
        _tree_snapshot=(),
    )
    assert (
        evaluate._packet_component_snapshot(packet, lock, spec, component.root)
        is component
    )
    with pytest.raises(evaluate.MobileAsrEvidenceEvalError):
        evaluate._packet_component_snapshot(
            SimpleNamespace(_validated_components=(component,)),
            lock,
            spec,
            component.root,
        )
    with pytest.raises(evaluate.MobileAsrEvidenceEvalError):
        evaluate._packet_component_snapshot(
            packet, lock, spec, component.root / "changed"
        )
    changed = replace(spec, receipt_sha256="f" * 64)
    with pytest.raises(evaluate.MobileAsrEvidenceEvalError):
        evaluate._packet_component_snapshot(packet, lock, changed, component.root)
    other_spec = replace(
        spec,
        component_id="other",
        directory="other",
        manifest_file="other.json",
        receipt_file="other-receipt.json",
    )
    other_component = replace(
        component,
        spec=other_spec,
        root=tmp_path / "other",
    )
    reordered_lock = replace(lock, components=(other_spec, spec))
    reordered_packet = replace(
        packet,
        _validated_components=(component, other_component),
    )
    with pytest.raises(evaluate.MobileAsrEvidenceEvalError):
        evaluate._packet_component_snapshot(
            reordered_packet,
            reordered_lock,
            spec,
            component.root,
        )


def test_primock_snapshot_projection_ignores_current_runtime_and_preserves_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import prepare_primock57_conversation_fixture as primock

    def forbidden(*_args, **_kwargs):
        raise AssertionError("current PriMock runtime must not be used")

    for name in (
        "load_primock57_isolated_bundle",
        "verify_primock57_isolated_bundle",
        "load_primock57_overlap_bundle",
        "_preparer_closure",
    ):
        monkeypatch.setattr(primock, name, forbidden)
    monkeypatch.setattr(
        primock,
        "_PREPARER_FILES",
        (*primock._PREPARER_FILES[1:], "tools/future-runtime.py"),
    )
    monkeypatch.setattr(primock, "_SOURCE_ROLES", ("future-role",))
    monkeypatch.setattr(primock, "_SOURCE_PATHS", ("future-source",))

    isolated_spec, isolated_snapshot = _isolated_snapshot(tmp_path / "isolated")
    overlap_spec, overlap_snapshot = _overlap_snapshot(tmp_path / "overlap")
    isolated = evaluate._isolated_component_from_snapshot(
        isolated_spec, isolated_snapshot
    )
    overlap = evaluate._overlap_component_from_snapshot(overlap_spec, overlap_snapshot)

    assert isolated.metric_domain == "ordinary-wer-cer-v1"
    assert len(isolated.leaves) == 3
    assert [leaf.group_index for leaf in isolated.leaves] == [0, 1, 2]
    assert overlap.metric_domain == evaluate.OVERLAP_METRIC
    assert len(overlap.leaves) == 9
    assert [leaf.kind for leaf in overlap.leaves] == [
        "role_a",
        "role_b",
        "mix",
    ] * 3
    assert [leaf.group_index for leaf in overlap.leaves] == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert "private-reference" not in repr(isolated)
    assert "private-role" not in repr(overlap)


def test_production_input_loader_routes_primock_through_snapshot_and_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import prepare_primock57_conversation_fixture as primock

    packet_root = _private(tmp_path / "packet")
    components_root = _private(packet_root / "components")
    authority = evaluate.packet_api.load_packet_lock()
    roots = {}
    for spec in authority.components:
        roots[spec.component_id] = _private(components_root / spec.directory)
    isolated_spec, isolated = _isolated_snapshot(roots["primock-isolated"])
    overlap_spec, overlap = _overlap_snapshot(roots["primock-overlap"])
    specs = (*authority.components[:3], isolated_spec, overlap_spec)
    lock = replace(authority, components=specs)
    generic_snapshots = tuple(
        evaluate.packet_api._ValidatedComponent(
            spec=spec,
            root=roots[spec.component_id],
            artifacts=(),
            inventory=(),
        )
        for spec in specs[:3]
    )
    snapshots = (*generic_snapshots, isolated, overlap)
    packet = evaluate.packet_api.LoadedMobileAsrEvidencePacket(
        schema_version=evaluate.PACKET_SCHEMA_VERSION,
        packet_id=evaluate.PACKET_ID,
        packet_lock_sha256=evaluate.PACKET_LOCK_SHA256,
        packet_index_sha256=evaluate.PACKET_INDEX_SHA256,
        packet_receipt_sha256=evaluate.PACKET_RECEIPT_SHA256,
        components=5,
        logical_cases=123,
        pcm_inputs=129,
        pcm_bytes=isolated_spec.pcm_bytes + overlap_spec.pcm_bytes,
        samples=12,
        root=packet_root,
        _validated_components=snapshots,
        _tree_snapshot=(),
    )
    expected = {row[0]: row for row in evaluate._EXPECTED_COMPONENTS}

    def generic_component(spec, root):
        row = expected[spec.component_id]
        assert root == roots[spec.component_id]
        leaf = evaluate._Leaf(
            component_id=spec.component_id,
            kind="corpus",
            sha256="0" * 64,
            samples=0,
            audio=b"",
            expected_text="private-test-reference",
        )
        return evaluate._Component(
            component_id=spec.component_id,
            metric_domain=spec.metric_domain,
            logical_cases=spec.logical_cases,
            leaves=(leaf,) * row[3],
        )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("current PriMock runtime must not be used")

    monkeypatch.setattr(
        evaluate.packet_api, "load_mobile_asr_evidence_packet", lambda _root: packet
    )
    monkeypatch.setattr(evaluate.packet_api, "load_packet_lock", lambda: lock)
    monkeypatch.setattr(evaluate, "_corpus_component", generic_component)
    for name in (
        "load_primock57_isolated_bundle",
        "verify_primock57_isolated_bundle",
        "load_primock57_overlap_bundle",
        "_preparer_closure",
    ):
        monkeypatch.setattr(primock, name, forbidden)
    verified = []
    monkeypatch.setattr(
        evaluate.packet_api,
        "verify_mobile_asr_evidence_packet",
        lambda value: verified.append(value),
    )

    loaded = evaluate._load_production_inputs(
        packet_root,
        packet_index_sha256=evaluate.PACKET_INDEX_SHA256,
        packet_receipt_sha256=evaluate.PACKET_RECEIPT_SHA256,
    )
    assert verified == [packet]
    assert loaded.production_inputs is True
    assert (
        tuple(
            (row.component_id, row.metric_domain, row.logical_cases, len(row.leaves))
            for row in loaded.components
        )
        == evaluate._EXPECTED_COMPONENTS
    )
    assert sum(len(row.leaves) for row in loaded.components) == 129

    def changed_packet(_value):
        raise evaluate.packet_api.MobileAsrEvidencePacketError()

    monkeypatch.setattr(
        evaluate.packet_api, "verify_mobile_asr_evidence_packet", changed_packet
    )
    with pytest.raises(evaluate.MobileAsrEvidenceEvalError):
        evaluate._load_production_inputs(
            packet_root,
            packet_index_sha256=evaluate.PACKET_INDEX_SHA256,
            packet_receipt_sha256=evaluate.PACKET_RECEIPT_SHA256,
        )
    model_attempts = []
    monkeypatch.setattr(
        evaluate,
        "_load_model_authority",
        lambda value: model_attempts.append(value),
    )
    with pytest.raises(evaluate.MobileAsrEvidenceEvalError):
        evaluate.preflight_mobile_asr_evidence_eval(
            packet_root,
            packet_index_sha256=evaluate.PACKET_INDEX_SHA256,
            packet_receipt_sha256=evaluate.PACKET_RECEIPT_SHA256,
            worker_manifest=tmp_path / "must-not-open-model",
            scratch_root=tmp_path / "must-not-create-scratch",
            output_path=tmp_path / "must-not-create-report",
        )
    assert model_attempts == []
    assert not (tmp_path / "must-not-create-scratch").exists()
    assert not (tmp_path / "must-not-create-report").exists()


@pytest.mark.parametrize("builder", [_isolated_snapshot, _overlap_snapshot])
def test_primock_snapshot_rejects_manifest_receipt_and_leaf_mutations(
    tmp_path: Path,
    builder,
) -> None:
    spec, component = builder(tmp_path / "component")
    extractor = (
        evaluate._isolated_component_from_snapshot
        if spec.component_id == "primock-isolated"
        else evaluate._overlap_component_from_snapshot
    )
    extractor(spec, component)
    artifacts = list(component.artifacts)
    pcm_index = 2
    mutations = (
        artifacts[:pcm_index] + artifacts[pcm_index + 1 :],
        [*artifacts, _artifact("attacker-extra.f32le", b"\x00\x00\x00\x00")],
        [
            *artifacts[:pcm_index],
            replace(artifacts[pcm_index], payload=b"\xff\xff\xff\xff"),
            *artifacts[pcm_index + 1 :],
        ],
        [
            *artifacts[:pcm_index],
            _artifact(artifacts[pcm_index].name, b"\xff\xff\xff\xff"),
            *artifacts[pcm_index + 1 :],
        ],
        [
            replace(artifacts[0], payload=artifacts[0].payload + b" "),
            *artifacts[1:],
        ],
        [
            artifacts[0],
            replace(artifacts[1], payload=artifacts[1].payload + b" "),
            *artifacts[2:],
        ],
        [
            *artifacts[:pcm_index],
            _artifact("substituted.f32le", artifacts[pcm_index].payload),
            *artifacts[pcm_index + 1 :],
        ],
    )
    for mutated in mutations:
        with pytest.raises(evaluate.MobileAsrEvidenceEvalError):
            extractor(spec, replace(component, artifacts=tuple(mutated)))


def test_synthetic_seam_is_explicitly_nonproduction() -> None:
    with pytest.raises(evaluate.MobileAsrEvidenceEvalError):
        evaluate._SyntheticWorkerContract(tuple(_outputs()), production_inputs=True)
    assert "_run_synthetic_test_evaluation" not in evaluate.__all__


def test_synthetic_run_publishes_private_single_link_report(tmp_path: Path) -> None:
    loaded = _run(tmp_path)
    metadata = loaded.path.lstat()
    assert stat.S_IMODE(metadata.st_mode) == 0o600
    assert metadata.st_nlink == 1
    assert loaded.value["kind"] == evaluate.SYNTHETIC_REPORT_KIND
    assert loaded.value["production_inputs"] is False
    assert loaded.value["complete"] is True
    assert loaded.value["execution"]["evaluations"] == 129
    assert loaded.value["execution"]["logical_cases"] == 123
    assert loaded.value["evidence_scope"]["model_executed"] is False
    assert loaded.value["evidence_scope"]["mobile_model_identity_authority"] is False
    assert (
        loaded.value["evidence_scope"]["unpooled_component_result_authority"] is False
    )


def test_production_authority_is_narrowly_component_scoped() -> None:
    scope = evaluate._report_evidence_scope(
        production_inputs=True,
        model_executed=True,
    )
    assert scope["mobile_model_identity_authority"] is True
    assert scope["unpooled_component_result_authority"] is True
    assert scope["model_quality_authority"] is False
    assert scope["qualification_authority"] is False
    assert scope["promotion_authority"] is False
    assert scope["default_authority"] is False
    with pytest.raises(evaluate.MobileAsrEvidenceEvalError):
        evaluate._report_evidence_scope(
            production_inputs=True,
            model_executed=False,
        )


def test_zero_empty_and_multiple_endpoints_stay_distinct(tmp_path: Path) -> None:
    rows = _outputs()
    rows[0] = evaluate._EndpointOutcome(
        ("alpha", "beta"),
        (False, True),
        "endpoint",
        True,
    )
    rows[1] = evaluate._EndpointOutcome(("",), (True,), "endpoint", True)
    rows[2] = evaluate._EndpointOutcome((), (), "tail_exhausted", True)
    loaded = _run(tmp_path, rows)
    endpoint = _component(loaded, "command-noise")["endpoint_integrity"]
    assert endpoint["multiple_endpoint_evaluations"] == 1
    assert endpoint["empty_committed_endpoints"] == 1
    assert endpoint["zero_endpoint_evaluations"] >= 1
    assert endpoint["pre_source_complete_endpoints"] == 1
    assert endpoint["source_complete_endpoints"] >= 1


def test_tail_exhaustion_is_valid_non_authoritative_empty_hypothesis() -> None:
    from tools.streaming_stt.protocol import MobileZipformerFinalEvent

    leaf = evaluate._synthetic_inputs().components[0].leaves[0]
    final = MobileZipformerFinalEvent(
        request_id="mobile-000",
        seq=0,
        text="",
        samples_seen=leaf.samples + 48_000,
        elapsed_ms=1.0,
        finalization_ms=0.0,
        compute_ms=0.0,
        audio_seconds=leaf.samples / 16_000,
        chunks=31,
        deadline_misses=0,
        max_backlog_ms=0.0,
        resources=SimpleNamespace(),
        model_padding_samples=0,
        source_samples=leaf.samples,
        source_samples_consumed=leaf.samples,
        declared_tail_samples=48_000,
        tail_samples_consumed=48_000,
        endpoint_observations=(),
        endpoint_reason="tail_exhausted",
        native_endpoint=False,
        endpoint_sample=None,
        endpoint_latency_ms=None,
        authoritative=False,
    )
    outcome = evaluate._extract_endpoint_outcome(SimpleNamespace(final=final), leaf)
    assert outcome == evaluate._EndpointOutcome((), (), "tail_exhausted", True)


def test_command_target_cannot_be_synthesized_across_endpoint_boundary(
    tmp_path: Path,
) -> None:
    rows = _outputs()
    rows[0] = evaluate._EndpointOutcome(
        ("alpha", "beta"),
        (False, True),
        "endpoint",
        True,
    )
    loaded = _run(tmp_path, rows)
    metrics = _component(loaded, "command-noise")["metrics"]
    assert metrics["command_attempts"] == 1
    assert metrics["command_hits"] == 0
    assert metrics["transcript_accuracy"]["wer"] == 0.0


def test_zero_endpoint_scores_as_empty_hypothesis(tmp_path: Path) -> None:
    rows = _outputs()
    rows[0] = evaluate._EndpointOutcome((), (), "tail_exhausted", True)
    loaded = _run(tmp_path, rows)
    accuracy = _component(loaded, "command-noise")["metrics"]["transcript_accuracy"]
    assert accuracy["hypothesis_words"] == 0
    assert accuracy["deletions"] == 2
    assert accuracy["wer"] == 1.0


def test_demand_has_only_fixed_independent_strata(tmp_path: Path) -> None:
    loaded = _run(tmp_path)
    demand = _component(loaded, "demand-noise")
    assert demand["metric_domain"] == "stratified-wer-cer-v1"
    assert [row["stratum"] for row in demand["metrics"]["strata"]] == list(
        evaluate._DEMAND_STRATA
    )
    assert {row["accuracy"]["evaluations"] for row in demand["metrics"]["strata"]} == {
        14
    }


def test_notsofar_retains_paired_anonymous_channels(tmp_path: Path) -> None:
    loaded = _run(tmp_path)
    metrics = _component(loaded, "notsofar-far-field")["metrics"]
    assert [row["channel"] for row in metrics["channels"]] == ["a", "b"]
    assert [row["accuracy"]["evaluations"] for row in metrics["channels"]] == [
        9,
        9,
    ]
    assert metrics["paired_channel"] == {
        "exact_normalized_output_agreements": 9,
        "pairs": 9,
    }


def test_primock_isolated_is_ordinary_wer_cer(tmp_path: Path) -> None:
    loaded = _run(tmp_path)
    isolated = _component(loaded, "primock-isolated")
    assert isolated["metric_domain"] == "ordinary-wer-cer-v1"
    assert isolated["metrics"]["accuracy"]["evaluations"] == 3
    assert isolated["metrics"]["accuracy"]["wer"] == 0.0
    assert isolated["metrics"]["accuracy"]["cer"] == 0.0


def test_overlap_exposes_only_custom_mix_metric(tmp_path: Path) -> None:
    loaded = _run(tmp_path)
    overlap = _component(loaded, "primock-overlap")
    assert overlap["metric_domain"] == evaluate.OVERLAP_METRIC
    assert set(overlap["metrics"]) == {
        "custom_accuracy",
        "scored_mix_evaluations",
        "unscored_stem_executions",
    }
    assert overlap["metrics"]["scored_mix_evaluations"] == 3
    assert overlap["metrics"]["unscored_stem_executions"] == 6
    assert overlap["metrics"]["custom_accuracy"]["protocol"] == evaluate.OVERLAP_METRIC
    assert overlap["metrics"]["custom_accuracy"]["wer"] == 0.0


def test_report_has_no_pooled_wer_latency_or_rtf(tmp_path: Path) -> None:
    loaded = _run(tmp_path)
    assert loaded.value["metric_aggregation"] == {
        "component_domains_are_independent": True,
        "pooled_metric": None,
        "pooled_wer": False,
    }
    encoded = loaded.path.read_text(encoding="ascii")
    assert '"latency_ms"' not in encoded
    assert '"rtf"' not in encoded
    assert '"case_id"' not in encoded
    assert '"path"' not in encoded
    assert '"transcript"' not in encoded
    assert "juliet" not in encoded
    assert "foxtrot" not in encoded


def test_partial_interval_is_named_cadence_not_latency(tmp_path: Path) -> None:
    loaded = _run(tmp_path)
    stream = loaded.value["execution"]["stream"]
    assert stream["partial_report_cadence_ms"] == 100
    assert "partial_interval_ms" not in stream


def test_public_loader_rejects_synthetic_report(tmp_path: Path) -> None:
    loaded = _run(tmp_path)
    with pytest.raises(evaluate.MobileAsrEvidenceEvalError):
        evaluate.load_mobile_asr_evidence_eval_report(
            loaded.path,
            expected_sha256=loaded.digest,
            packet_index_sha256=loaded.packet_index_sha256,
            packet_receipt_sha256=loaded.packet_receipt_sha256,
            model_manifest_sha256=loaded.model_manifest_sha256,
        )


def test_report_is_no_clobber(tmp_path: Path) -> None:
    loaded = _run(tmp_path)
    with pytest.raises(evaluate.MobileAsrEvidenceEvalError):
        evaluate._run_synthetic_test_evaluation(
            evaluate._SyntheticWorkerContract(tuple(_outputs())),
            loaded.path,
        )
    assert hashlib.sha256(loaded.path.read_bytes()).hexdigest() == loaded.digest


def test_report_recovers_truthfully_if_link_succeeds_then_raises(
    tmp_path: Path, monkeypatch
) -> None:
    real_link = os.link

    def link_then_raise(*args, **kwargs):
        real_link(*args, **kwargs)
        raise OSError("synthetic post-link interruption")

    monkeypatch.setattr(evaluate.os, "link", link_then_raise)
    loaded = _run(tmp_path)
    assert loaded.path.exists()
    assert loaded.path.lstat().st_nlink == 1
    assert hashlib.sha256(loaded.path.read_bytes()).hexdigest() == loaded.digest


def test_publication_fsync_failure_stays_linked_but_not_durable(
    tmp_path: Path, monkeypatch
) -> None:
    parent = _private(tmp_path / "reports")
    output = parent / "report.json"
    state = evaluate._CommitState()
    real_fsync = os.fsync

    def fail_directory_fsync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("synthetic directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(evaluate.os, "fsync", fail_directory_fsync)
    with pytest.raises(evaluate.MobileAsrEvidenceEvalError):
        evaluate._publish_report(
            output,
            {"ok": True},
            guard=lambda: None,
            state=state,
        )
    assert output.exists()
    assert state.linked is True
    assert state.committed is False
    assert state.digest == ""
    with pytest.raises(evaluate.MobileAsrEvidenceEvalError):
        evaluate._publish_report(
            output,
            {"ok": True},
            guard=lambda: None,
            state=evaluate._CommitState(),
        )


def test_publication_rejects_parent_replacement_after_directory_fsync(
    tmp_path: Path, monkeypatch
) -> None:
    parent = _private(tmp_path / "reports")
    moved = tmp_path / "moved-reports"
    output = parent / "report.json"
    state = evaluate._CommitState()
    real_fsync = os.fsync
    replaced = False

    def replace_parent_after_fsync(descriptor: int) -> None:
        nonlocal replaced
        real_fsync(descriptor)
        if stat.S_ISDIR(os.fstat(descriptor).st_mode) and not replaced:
            replaced = True
            parent.rename(moved)
            parent.mkdir(mode=0o700)

    monkeypatch.setattr(evaluate.os, "fsync", replace_parent_after_fsync)
    with pytest.raises(evaluate.MobileAsrEvidenceEvalError):
        evaluate._publish_report(
            output,
            {"ok": True},
            guard=lambda: None,
            state=state,
        )
    assert replaced is True
    assert state.linked is True
    assert state.committed is False
    assert not output.exists()
    assert (moved / output.name).exists()


def test_post_link_signal_unmasks_only_after_durable_commit(
    tmp_path: Path, monkeypatch
) -> None:
    real_fsync = os.fsync
    real_pthread_sigmask = signal.pthread_sigmask
    events: list[str] = []

    def observed_fsync(descriptor: int) -> None:
        real_fsync(descriptor)
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            events.append("directory-fsync")

    def interrupt_after_unmask(how, mask):
        restored = real_pthread_sigmask(how, mask)
        if how == signal.SIG_SETMASK:
            events.append("signal-unmask")
            raise KeyboardInterrupt()
        return restored

    monkeypatch.setattr(evaluate.os, "fsync", observed_fsync)
    monkeypatch.setattr(evaluate.signal, "pthread_sigmask", interrupt_after_unmask)
    loaded = _run(tmp_path)
    assert loaded.path.exists()
    assert events == ["directory-fsync", "signal-unmask"]


def test_publication_blocks_exact_plain_int_signals_and_restores(
    tmp_path: Path, monkeypatch
) -> None:
    expected = (int(signal.SIGINT), int(signal.SIGHUP), int(signal.SIGTERM))
    assert expected
    assert evaluate._publication_signal_numbers() == expected
    assert all(type(value) is int for value in expected)

    real_pthread_sigmask = signal.pthread_sigmask
    calls: list[tuple[object, object, object]] = []

    def observed_pthread_sigmask(how, mask):
        returned = real_pthread_sigmask(how, mask)
        calls.append((how, mask, returned))
        return returned

    monkeypatch.setattr(
        evaluate.signal,
        "pthread_sigmask",
        observed_pthread_sigmask,
    )
    loaded = _run(tmp_path)
    assert loaded.path.exists()
    assert len(calls) == 2
    assert calls[0][0] == signal.SIG_BLOCK
    assert calls[0][1] == expected
    assert calls[1][0] == signal.SIG_SETMASK
    assert calls[1][1] == calls[0][2]


def test_report_loader_rejects_byte_tamper(tmp_path: Path) -> None:
    loaded = _run(tmp_path)
    raw = loaded.path.read_bytes()
    loaded.path.write_bytes(raw[:-1] + b" ")
    loaded.path.chmod(0o600)
    with pytest.raises(evaluate.MobileAsrEvidenceEvalError):
        evaluate._load_report_common(
            loaded.path,
            expected_sha256=loaded.digest,
            production_inputs=False,
        )


def test_report_loader_rejects_hardlink(tmp_path: Path) -> None:
    loaded = _run(tmp_path)
    os.link(loaded.path, loaded.path.with_name("alias.json"))
    with pytest.raises(evaluate.MobileAsrEvidenceEvalError):
        evaluate._load_report_common(
            loaded.path,
            expected_sha256=loaded.digest,
            production_inputs=False,
        )


def test_report_loader_rejects_mode_change(tmp_path: Path) -> None:
    loaded = _run(tmp_path)
    loaded.path.chmod(0o640)
    with pytest.raises(evaluate.MobileAsrEvidenceEvalError):
        evaluate._load_report_common(
            loaded.path,
            expected_sha256=loaded.digest,
            production_inputs=False,
        )


def test_strict_json_rejects_duplicate_keys() -> None:
    with pytest.raises(evaluate.MobileAsrEvidenceEvalError):
        evaluate._strict_json(b'{"ok":true,"ok":false}')


def test_endpoint_outcome_rejects_shape_and_non_bool() -> None:
    with pytest.raises(evaluate.MobileAsrEvidenceEvalError):
        evaluate._EndpointOutcome(("alpha",), (), "endpoint", True)
    with pytest.raises(evaluate.MobileAsrEvidenceEvalError):
        evaluate._EndpointOutcome(("alpha",), (1,), "endpoint", True)  # type: ignore[arg-type]


def test_cli_failure_is_exact_detail_free_json(tmp_path: Path, capsys) -> None:
    first = _private(tmp_path / "scratch-parent") / "new-scratch"
    second = _private(tmp_path / "output-parent") / "report.json"
    result = evaluate.main(
        [
            "--preflight",
            "--packet-root",
            str(tmp_path / "missing-packet"),
            "--packet-index-sha256",
            evaluate.PACKET_INDEX_SHA256,
            "--packet-receipt-sha256",
            evaluate.PACKET_RECEIPT_SHA256,
            "--worker-manifest",
            str(tmp_path / "missing-model"),
            "--scratch-root",
            str(first),
            "--output",
            str(second),
        ]
    )
    assert result == 2
    assert json.loads(capsys.readouterr().out) == evaluate._SAFE_ERROR
    assert not first.exists()
    assert not second.exists()


def test_cli_parse_failure_does_not_echo_private_argument(capsys) -> None:
    result = evaluate.main(["--unknown", "/private/do-not-echo"])
    captured = capsys.readouterr()
    assert result == 2
    assert json.loads(captured.out) == evaluate._SAFE_ERROR
    assert captured.err == ""
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
                "tools.mobile_asr_evidence_eval",
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


def test_synthetic_scratch_stages_and_close_revalidates(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    worker_root, paths, bundle, snapshot = evaluate._prepare_scratch(
        scratch, evaluate._synthetic_inputs()
    )
    assert worker_root == scratch / "worker"
    assert len(paths) == 129
    evaluate._verify_scratch(
        scratch,
        snapshot,
        bundle,
        paths,
        evaluate._synthetic_inputs(),
    )


def test_accuracy_uses_integer_micro_aggregation() -> None:
    result = evaluate._accuracy((("a b", "a"), ("c", "c d")))
    assert result["evaluations"] == 2
    assert result["deletions"] == 1
    assert result["insertions"] == 1
    assert result["word_errors"] == 2
    assert result["reference_words"] == 3
    assert result["wer"] == 2 / 3


def test_report_binding_detects_semantic_mutation(tmp_path: Path) -> None:
    loaded = _run(tmp_path)
    value = json.loads(loaded.path.read_text(encoding="ascii"))
    value["execution"]["evaluations"] = 128
    unsigned = dict(value)
    unsigned.pop("binding_sha256")
    value["binding_sha256"] = evaluate._canonical_sha256(unsigned)
    with pytest.raises(evaluate.MobileAsrEvidenceEvalError):
        evaluate._validate_report(value, production_inputs=False)


def test_report_rejects_rebound_metric_arithmetic_mutation(tmp_path: Path) -> None:
    loaded = _run(tmp_path)
    value = json.loads(loaded.path.read_text(encoding="ascii"))
    value["components"][1]["metrics"]["accuracy"]["word_errors"] = 1
    unsigned = dict(value)
    unsigned.pop("binding_sha256")
    value["binding_sha256"] = evaluate._canonical_sha256(unsigned)
    with pytest.raises(evaluate.MobileAsrEvidenceEvalError):
        evaluate._validate_report(value, production_inputs=False)


def test_report_rejects_rebound_component_endpoint_drift(tmp_path: Path) -> None:
    loaded = _run(tmp_path)
    value = json.loads(loaded.path.read_text(encoding="ascii"))
    endpoint = value["components"][0]["endpoint_integrity"]
    endpoint["single_endpoint_evaluations"] -= 1
    endpoint["zero_endpoint_evaluations"] += 1
    unsigned = dict(value)
    unsigned.pop("binding_sha256")
    value["binding_sha256"] = evaluate._canonical_sha256(unsigned)
    with pytest.raises(evaluate.MobileAsrEvidenceEvalError):
        evaluate._validate_report(value, production_inputs=False)


def test_source_declares_exact_one_pass_stream() -> None:
    assert evaluate._STREAM == evaluate.StreamConfig(
        chunk_samples=1_600,
        pace="burst",
        partial_interval_ms=100,
        tail_padding_samples=48_000,
    )
    assert evaluate.PROTOCOL_VERSION == 5
    assert sum(row[3] for row in evaluate._EXPECTED_COMPONENTS) == 129
    assert sum(row[2] for row in evaluate._EXPECTED_COMPONENTS) == 123
