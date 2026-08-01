from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from tests.streaming_stt_helpers import (
    WORKER_PATH,
    bound_file,
    resource,
    scripted_case,
    write_fixture,
)
from tools import streaming_stt_eval
from tools.streaming_stt.corpus import CorpusProvenance, load_corpus
from tools.streaming_stt.corpus_writer import CorpusWriteCase, publish_private_corpus
from tools.streaming_stt.metrics import RunRecord, aggregate_metrics
from tools.streaming_stt.manifest import (
    MOONSHINE_ADAPTER,
    MOONSHINE_ARTIFACT_NAMES,
    NEMOTRON_ADAPTER,
    NEMOTRON_ARTIFACT_NAMES,
    PARAKEET_REALTIME_EOU_ADAPTER,
    MoonshineConfig,
    NemotronConfig,
    ParakeetRealtimeEouConfig,
)
from tools.streaming_stt.protocol import (
    NATIVE_ENDPOINT_PROTOCOL_VERSION,
    PROTOCOL_VERSION,
    CaseTrace,
    FinalEvent,
    NativeFinalEvent,
    PartialEvent,
    ReadyEvent,
    ResourceUsage,
    StreamConfig,
)
from tools.streaming_stt.runtime_receipt import PARAKEET_RUNTIME_TREE_LIMITS


@pytest.fixture(autouse=True)
def _isolated_host_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(
        streaming_stt_eval,
        "_HOST_LOCK_PATH",
        tmp_path / "host-locks" / "streaming-stt.lock",
    )


def _benchmark_fixture(tmp_path: Path) -> tuple[Path, Path]:
    manifest, corpus, _ = write_fixture(
        tmp_path,
        [
            {
                "id": "sentinel-private-command",
                "values": [0.01] * 160,
                "expected_text": "stop now",
                "commands": ["stop"],
                "tags": ["command"],
            },
            {
                "id": "sentinel-private-silence",
                "values": [0.0] * 80,
                "expected_text": "",
                "assertion": "silence",
                "tags": ["silence"],
            },
        ],
        [
            scripted_case(
                partials=[
                    (80, "sto", 40.0, 1.0),
                    (160, "stop now", 80.0, 2.0),
                ],
                final="stop now",
                elapsed_ms=100.0,
                finalization_ms=20.0,
                compute_ms=4.0,
                deadline_misses=1,
                max_backlog_ms=3.0,
                resources=resource(rss_mb=90.0, threads=2, vram_mb=20.0),
            ),
            scripted_case(
                partials=[(80, "", 20.0, 0.5)],
                final="",
                elapsed_ms=30.0,
                finalization_ms=10.0,
                compute_ms=1.0,
                resources=resource(rss_mb=80.0, threads=1, vram_mb=10.0),
            ),
        ],
    )
    return manifest, corpus


def _parakeet_config() -> ParakeetRealtimeEouConfig:
    return ParakeetRealtimeEouConfig()


def _parakeet_ready(
    usage: ResourceUsage,
    *,
    protocol_version: int = NATIVE_ENDPOINT_PROTOCOL_VERSION,
) -> ReadyEvent:
    return ReadyEvent(
        model_id="parakeet-realtime-eou-120m-v1",
        manifest_sha256="a" * 64,
        source_bundle_sha256="b" * 64,
        adapter=PARAKEET_REALTIME_EOU_ADAPTER,
        model_load_ms=5.0,
        resources=usage,
        runtime={"python": "3.12.3", "platform": "linux"},
        protocol_version=protocol_version,
    )


def _native_final_event(
    *,
    usage: ResourceUsage,
    text: str,
    source: int,
    source_consumed: int,
    tail_consumed: int,
    reason: str,
    probability: float,
    authoritative: bool,
    declared_tail: int = 160,
) -> NativeFinalEvent:
    samples_seen = source_consumed + tail_consumed
    return NativeFinalEvent(
        request_id="x",
        seq=0,
        text=text,
        samples_seen=samples_seen,
        elapsed_ms=100.0,
        finalization_ms=2.0,
        compute_ms=4.0,
        audio_seconds=source / 16_000,
        chunks=1,
        deadline_misses=0,
        max_backlog_ms=0.0,
        resources=usage,
        model_padding_samples=0,
        source_samples=source,
        source_samples_consumed=source_consumed,
        declared_tail_samples=declared_tail,
        tail_samples_consumed=tail_consumed,
        endpoint_reason=reason,
        native_endpoint=True,
        endpoint_probability=probability,
        endpoint_sample=samples_seen,
        endpoint_latency_ms=(tail_consumed / 16.0 if authoritative else None),
        authoritative=authoritative,
    )


def test_end_to_end_report_is_aggregate_exact_bound_and_fake_labelled(tmp_path):
    manifest, corpus = _benchmark_fixture(tmp_path)
    scratch = tmp_path / "scratch-parent"

    report = streaming_stt_eval.run_benchmark(
        manifest,
        corpus,
        scratch_parent=scratch,
        repeats=3,
        stream=StreamConfig(64, "burst", 100, 16),
    )

    assert report["ok"] is True
    assert report["evidence"] == {
        "kind": "bounded_pcm_streaming_harness",
        "fake_adapter": True,
        "production_model": False,
        "real_candidate_model": False,
        "model_executed": False,
        "replay_pace": "burst",
        "accelerated_replay": True,
        "conversational_latency_valid": False,
        "latency_evidence": "accelerated_replay_only",
        "timing_scope": "adapter_after_pcm_load_to_result",
        "compute_rtf_scope": "candidate_calls_only",
        "partial_source": "candidate_snapshot",
        "final_source": "candidate_final",
        "model_padding_reported": True,
        "end_to_end_rtf": False,
        "pcm_snapshot_io_included": False,
        "controller_ipc_included": False,
        "capture_to_text_latency": False,
        "vad": False,
        "endpointing": False,
        "aec": False,
        "live_hardware": False,
        "adoption_authority": False,
    }
    assert report["worker"]["adapter"] == "fake-json-v1"
    assert "manifest_schema_version" not in report["worker"]
    assert "adapter_config" not in report["worker"]
    assert "runtime_receipt_sha256" not in report["worker"]
    assert len(report["worker"]["manifest_sha256"]) == 64
    assert len(report["worker"]["source_bundle_sha256"]) == 64
    assert report["worker"]["source_bundle_files"] > 1
    assert len(report["worker"]["artifact_set_sha256"]) == 64
    assert report["corpus"]["cases"] == 2
    assert "schema_version" not in report["corpus"]
    assert "provenance" not in report["corpus"]
    metrics = report["metrics"]
    assert "endpointing" not in metrics
    assert metrics["evaluations"] == 6
    assert metrics["coverage_complete"] is True
    assert metrics["accuracy"]["transcript"]["clips"] == 3
    assert metrics["accuracy"]["transcript"]["exact"] == 3
    assert metrics["accuracy"]["command_recall"] == 1.0
    assert metrics["accuracy"]["silence_nonempty_finals"] == 0
    assert metrics["streaming"]["deadline_misses"] == 3
    assert metrics["streaming"]["model_padding_total_samples"] == 0
    assert metrics["streaming"]["model_padding_max_samples"] == 0
    assert metrics["latency"]["first_nonempty_partial_p50_ms"] == 40.0
    assert metrics["latency"]["stable_partial_p95_ms"] == 80.0
    assert metrics["resources"]["max_reported_rss_mb"] == 90.0
    assert "peak_rss_mb" not in metrics["resources"]
    assert metrics["resources"]["rss_scope"] == (
        "maximum_of_ready_and_final_samples_not_process_peak"
    )
    assert metrics["resources"]["peak_vram_mb"] == 20.0
    assert metrics["determinism"]["cases_with_final_disagreement"] == 0
    assert report["evaluator"]["kind"] == "isolated_streaming_stt_fake_harness"
    assert report["evaluator"]["production_model"] is False
    assert report["evaluator"]["real_candidate_model"] is False
    assert report["evaluator"]["model_executed"] is False
    assert {
        "tools/streaming_stt/runtime_receipt.py",
        "tools/streaming_stt/adapters/moonshine.py",
    } <= set(report["evaluator"]["files"])
    assert {
        "tools/prepare_public_streaming_stt_corpus.py",
        "tools/provision_moonshine_candidate.py",
    }.isdisjoint(report["evaluator"]["files"])
    assert not any(scratch.glob(".streaming-stt-*"))

    encoded = json.dumps(report)
    for private in (
        "stop now",
        "sto",
        str(corpus),
        str(tmp_path / "audio-0.f32le"),
        "sentinel-private-command",
        "sentinel-private-silence",
    ):
        assert private not in encoded


def test_schema_v3_private_corpus_runs_through_aggregate_fake_benchmark(
    tmp_path: Path,
) -> None:
    private_reference = "find in my private vault"
    private_case_id = "private-diagnostic-case"
    manifest, _legacy_corpus, _digests = write_fixture(
        tmp_path,
        [
            {
                "id": private_case_id,
                "values": [0.01] * 160,
                "expected_text": private_reference,
                "tags": ["owner-voice", "quiet-room"],
            }
        ],
        [
            scripted_case(
                partials=[(80, "find in", 20.0, 1.0)],
                final=private_reference,
                elapsed_ms=40.0,
                finalization_ms=5.0,
                compute_ms=2.0,
            )
        ],
    )
    tmp_path.chmod(0o700)
    published = publish_private_corpus(
        cases=(
            CorpusWriteCase(
                case_id=private_case_id,
                audio_bytes=(tmp_path / "audio-0.f32le").read_bytes(),
                reference=private_reference,
                tags=("owner-voice", "quiet-room"),
            ),
        ),
        provenance=CorpusProvenance(
            kind="private-diagnostic-v1",
            suite="final-model-input",
            manifest_sha256="1" * 64,
            metadata_sha256="2" * 64,
            source_set_sha256="3" * 64,
        ),
        output_dir=tmp_path / "private-v3",
        purpose="exact private final model input benchmark",
    )
    assert published.schema_version == 3

    report = streaming_stt_eval.run_benchmark(
        manifest,
        published.path,
        scratch_parent=tmp_path / "scratch-parent",
        repeats=1,
        stream=StreamConfig(80, "burst", 100, 0),
    )

    assert report["ok"] is True
    assert report["corpus"]["schema_version"] == 3
    assert report["corpus"]["provenance"] == {
        "kind": "private-diagnostic-v1",
        "suite": "final-model-input",
        "manifest_sha256": "1" * 64,
        "metadata_sha256": "2" * 64,
        "source_set_sha256": "3" * 64,
    }
    assert report["metrics"]["evaluations"] == 1
    assert report["metrics"]["coverage_complete"] is True
    encoded = json.dumps(report)
    assert private_reference not in encoded
    assert private_case_id not in encoded
    assert str(published.path) not in encoded


def test_moonshine_worker_and_evidence_bindings_are_closed_and_non_adopting(
    tmp_path,
):
    artifacts = tuple(
        SimpleNamespace(
            name=name,
            sha256=f"{index + 1:064x}",
            size_bytes=index + 1,
            path=tmp_path / f"private-artifact-{index}",
        )
        for index, name in enumerate(MOONSHINE_ARTIFACT_NAMES)
    )
    manifest = SimpleNamespace(
        digest="a" * 64,
        schema_version=2,
        model_id="moonshine-small-streaming",
        adapter=MOONSHINE_ADAPTER,
        python=SimpleNamespace(sha256="b" * 64),
        worker=SimpleNamespace(sha256="c" * 64),
        artifacts=artifacts,
        adapter_config=MoonshineConfig(
            package_version="0.1.0",
            api_version=30000,
            model_arch="small-streaming",
            provider="cpu",
            language="en",
        ),
    )
    source_bundle = SimpleNamespace(
        tree_sha256="d" * 64,
        files=(object(), object()),
    )

    binding = streaming_stt_eval._worker_binding(
        manifest,
        source_bundle,
        runtime_receipt_sha256="e" * 64,
    )

    assert binding["manifest_schema_version"] == 2
    assert binding["adapter_config"] == {
        "package_version": "0.1.0",
        "api_version": 30000,
        "model_arch": "small-streaming",
        "provider": "cpu",
        "language": "en",
    }
    assert binding["runtime_receipt_sha256"] == "e" * 64
    assert str(tmp_path) not in json.dumps(binding)
    assert streaming_stt_eval._evidence_binding(
        MOONSHINE_ADAPTER,
        pace="realtime",
    ) == {
        "kind": "bounded_pcm_streaming_harness",
        "fake_adapter": False,
        "production_model": False,
        "real_candidate_model": True,
        "model_executed": True,
        "replay_pace": "realtime",
        "accelerated_replay": False,
        "conversational_latency_valid": False,
        "latency_evidence": "paced_replay_only",
        "timing_scope": "adapter_after_pcm_load_to_result",
        "compute_rtf_scope": "candidate_calls_only",
        "partial_source": "candidate_snapshot",
        "final_source": "candidate_final",
        "model_padding_reported": True,
        "end_to_end_rtf": False,
        "pcm_snapshot_io_included": False,
        "controller_ipc_included": False,
        "capture_to_text_latency": False,
        "vad": False,
        "endpointing": False,
        "aec": False,
        "live_hardware": False,
        "adoption_authority": False,
    }
    evaluator = streaming_stt_eval._evaluator_binding(MOONSHINE_ADAPTER)
    assert evaluator["production_model"] is False
    assert evaluator["real_candidate_model"] is True
    assert evaluator["model_executed"] is True
    assert {
        "tools/prepare_public_streaming_stt_corpus.py",
        "tools/provision_moonshine_candidate.py",
    }.isdisjoint(evaluator["files"])
    with pytest.raises(ValueError):
        streaming_stt_eval._evidence_binding("unknown-adapter", pace="burst")
    with pytest.raises(ValueError):
        streaming_stt_eval._evidence_binding(MOONSHINE_ADAPTER, pace="warp")


def test_nemotron_binding_labels_provisional_finals_padding_and_compute_scope(
    tmp_path,
):
    artifacts = tuple(
        SimpleNamespace(
            name=name,
            sha256=f"{index + 1:064x}",
            size_bytes=index + 1,
            path=tmp_path / f"private-nemotron-{index}",
        )
        for index, name in enumerate(NEMOTRON_ARTIFACT_NAMES)
    )
    config = NemotronConfig(language="ro-RO", lookahead_tokens=6)
    manifest = SimpleNamespace(
        digest="a" * 64,
        schema_version=3,
        model_id="nemotron-streaming-ro-la6",
        adapter=NEMOTRON_ADAPTER,
        python=SimpleNamespace(sha256="b" * 64),
        worker=SimpleNamespace(sha256="c" * 64),
        artifacts=artifacts,
        adapter_config=config,
    )
    source_bundle = SimpleNamespace(tree_sha256="d" * 64, files=(object(),))

    binding = streaming_stt_eval._worker_binding(
        manifest,
        source_bundle,
        runtime_receipt_sha256="e" * 64,
    )

    assert binding["manifest_schema_version"] == 3
    assert binding["adapter_config"] == config.as_dict()
    evidence = streaming_stt_eval._evidence_binding(
        NEMOTRON_ADAPTER,
        pace="realtime",
        adapter_config=config,
    )
    assert evidence["production_model"] is False
    assert evidence["real_candidate_model"] is True
    assert evidence["compute_rtf_scope"] == "generate_wall_minus_pacing"
    assert evidence["partial_source"] == "provisional_streamer"
    assert evidence["final_source"] == "authoritative_generate_sequences"
    assert evidence["model_padding_reported"] is True
    assert evidence["conversational_latency_valid"] is False
    assert evidence["latency_evidence"] == "paced_replay_only"
    assert evidence["capture_accrual_included"] is False
    assert (
        evidence["conversational_latency_scope"]
        == "diagnostic_replay_excludes_native_capture_accrual"
    )
    assert evidence["deadline_accounting"] == "post_consumption_per_native_chunk"
    assert evidence["native_stream_geometry"] == {
        "sample_rate_hz": 16_000,
        "hop_length_samples": 160,
        "n_fft_samples": 512,
        "win_length_samples": 400,
        "first_chunk_frames": 49,
        "chunk_frames": 56,
        "first_window_samples": 7_880,
        "window_samples": 9_360,
        "stride_samples": 8_960,
        "streaming_latency_ms": 560,
    }
    with pytest.raises(ValueError):
        streaming_stt_eval._evidence_binding(
            NEMOTRON_ADAPTER,
            pace="realtime",
        )


def test_parakeet_schema_v5_binding_and_native_endpoint_evidence_are_closed(
    tmp_path,
):
    config = _parakeet_config()
    artifacts = tuple(
        SimpleNamespace(
            name=name,
            sha256=f"{index + 1:064x}",
            size_bytes=index + 1,
            path=tmp_path / f"private-parakeet-{index}",
        )
        for index, name in enumerate(
            (
                "runtime-receipt",
                "runtime-wheel-lock",
                "venv-marker",
                "model-nemo",
            )
        )
    )
    manifest = SimpleNamespace(
        digest="a" * 64,
        schema_version=5,
        model_id="parakeet-realtime-eou-120m-v1",
        adapter=PARAKEET_REALTIME_EOU_ADAPTER,
        python=SimpleNamespace(sha256="b" * 64),
        worker=SimpleNamespace(sha256="c" * 64),
        artifacts=artifacts,
        adapter_config=config,
    )
    source_bundle = SimpleNamespace(tree_sha256="d" * 64, files=(object(),))

    binding = streaming_stt_eval._worker_binding(
        manifest,
        source_bundle,
        runtime_receipt_sha256="e" * 64,
    )
    evidence = streaming_stt_eval._evidence_binding(
        PARAKEET_REALTIME_EOU_ADAPTER,
        pace="realtime",
        adapter_config=config,
    )

    assert binding["manifest_schema_version"] == 5
    assert binding["adapter_config"] == config.as_dict()
    assert binding["runtime_receipt_sha256"] == "e" * 64
    assert evidence["real_candidate_model"] is True
    assert evidence["endpointing"] is True
    assert evidence["conversational_latency_valid"] is False
    assert evidence["latency_evidence"] == "paced_replay_only"
    assert evidence["final_source"] == (
        "candidate_native_eou_eob_or_declared_tail_exhaustion"
    )
    assert evidence["response_authority"] == "complete_source_native_eou_only"
    assert evidence["endpoint_ground_truth"] is False
    assert evidence["endpoint_sample_domain"] == (
        "model_input_including_terminal_zero_fill"
    )
    assert evidence["source_tail_accounting"] == (
        "worker_reported_supervisor_validated"
    )
    assert evidence["resource_control"] == {
        "kind": "systemd-user-scope-cgroup-v2",
        "proof_source": "supervisor_cgroup_files_before_ready",
        "receipt_field": "worker.cgroup_evidence",
        "hard_host_memory_limit": True,
        "hard_cpu_quota": True,
        "hard_tasks_limit": True,
        "hard_vram_limit": False,
    }
    assert evidence["native_stream_geometry"] == {
        "sample_rate_hz": 16_000,
        "chunk_samples": 1_280,
        "chunk_ms": 80.0,
        "maximum_tail_padding_samples": 48_000,
    }
    evaluator = streaming_stt_eval._evaluator_binding(
        PARAKEET_REALTIME_EOU_ADAPTER
    )
    assert evaluator["real_candidate_model"] is True
    assert "tools/streaming_stt/adapters/parakeet_realtime_eou.py" in evaluator[
        "files"
    ]


def _parakeet_cgroup_evidence() -> dict[str, object]:
    return {
        "kind": "systemd-user-scope-cgroup-v2",
        "memory_high_bytes": 6_442_450_944,
        "memory_max_bytes": 8_589_934_592,
        "memory_swap_max_bytes": 0,
        "cpu_quota_percent": 200,
        "tasks_max": 128,
        "oom_policy": "kill",
        "verified": True,
    }


def test_parakeet_worker_report_binds_exact_verified_cgroup_receipt():
    manifest = SimpleNamespace(
        adapter=PARAKEET_REALTIME_EOU_ADAPTER,
        schema_version=5,
    )
    binding = {"adapter": PARAKEET_REALTIME_EOU_ADAPTER}
    runtime = {"python": "3.12.3", "platform": "linux"}

    report = streaming_stt_eval._worker_report(
        manifest,
        binding,
        runtime,
        _parakeet_cgroup_evidence(),
    )

    assert report == {
        "adapter": PARAKEET_REALTIME_EOU_ADAPTER,
        "runtime": runtime,
        "cgroup_evidence": _parakeet_cgroup_evidence(),
    }


@pytest.mark.parametrize(
    "mutation",
    [
        {"memory_high_bytes": 6_442_450_945},
        {"memory_max_bytes": 8_589_934_593},
        {"memory_swap_max_bytes": 1},
        {"cpu_quota_percent": 201},
        {"tasks_max": 129},
        {"oom_policy": "continue"},
        {"verified": False},
        {"verified": 1},
        {"extra": True},
    ],
)
def test_parakeet_worker_report_rejects_missing_or_inexact_cgroup_receipt(
    mutation,
):
    manifest = SimpleNamespace(
        adapter=PARAKEET_REALTIME_EOU_ADAPTER,
        schema_version=5,
    )
    evidence = _parakeet_cgroup_evidence()
    evidence.update(mutation)

    with pytest.raises(ValueError):
        streaming_stt_eval._worker_report(
            manifest,
            {},
            {},
            evidence,
        )

    with pytest.raises(ValueError):
        streaming_stt_eval._worker_report(
            manifest,
            {},
            {},
            None,
        )


def test_legacy_worker_report_shape_ignores_cgroup_only_provenance():
    manifest = SimpleNamespace(adapter="fake-json-v1", schema_version=1)
    binding = {"adapter": "fake-json-v1"}
    runtime = {"python": "3.12.3", "platform": "linux"}

    report = streaming_stt_eval._worker_report(
        manifest,
        binding,
        runtime,
        _parakeet_cgroup_evidence(),
    )

    assert report == {
        "adapter": "fake-json-v1",
        "runtime": runtime,
    }
    assert "cgroup_evidence" not in report


def test_adapter_specific_default_streams_preserve_fake_and_moonshine():
    fake = streaming_stt_eval._default_stream(
        SimpleNamespace(adapter="fake-json-v1", adapter_config=None)
    )
    moonshine = streaming_stt_eval._default_stream(
        SimpleNamespace(adapter=MOONSHINE_ADAPTER, adapter_config=MoonshineConfig())
    )
    nemotron = streaming_stt_eval._default_stream(
        SimpleNamespace(adapter=NEMOTRON_ADAPTER, adapter_config=NemotronConfig())
    )

    assert fake == StreamConfig(1600, "burst", 200, 0)
    assert moonshine == StreamConfig(1600, "burst", 200, 0)
    assert nemotron == StreamConfig(5120, "burst", 320, 0)


def test_parakeet_default_stream_uses_exact_native_chunk_and_declared_tail():
    manifest = SimpleNamespace(
        adapter=PARAKEET_REALTIME_EOU_ADAPTER,
        adapter_config=_parakeet_config(),
    )

    assert streaming_stt_eval._default_stream(manifest) == StreamConfig(
        1_280, "burst", 80, 48_000
    )
    selected = streaming_stt_eval._selected_stream(
        manifest,
        None,
        chunk_samples=None,
        pace=None,
        partial_interval_ms=None,
        tail_padding_samples=None,
    )

    assert selected == StreamConfig(1_280, "burst", 80, 48_000)


def test_evaluator_stream_validation_preserves_v2_and_v3_tail_bounds():
    legacy = StreamConfig(1_600, "burst", 200, 16_000)
    native = StreamConfig(1_280, "burst", 80, 48_000)

    assert streaming_stt_eval._validated_stream(
        legacy,
        protocol_version=PROTOCOL_VERSION,
    ) == legacy
    assert streaming_stt_eval._validated_stream(
        native,
        protocol_version=NATIVE_ENDPOINT_PROTOCOL_VERSION,
    ) == native

    with pytest.raises(ValueError):
        streaming_stt_eval._validated_stream(
            replace(legacy, tail_padding_samples=16_001),
            protocol_version=PROTOCOL_VERSION,
        )
    with pytest.raises(ValueError):
        streaming_stt_eval._validated_stream(
            replace(native, tail_padding_samples=48_001),
            protocol_version=NATIVE_ENDPOINT_PROTOCOL_VERSION,
        )


def test_parakeet_non_native_chunk_is_rejected_before_receipt_or_worker(
    tmp_path,
    monkeypatch,
):
    manifest = SimpleNamespace(
        adapter=PARAKEET_REALTIME_EOU_ADAPTER,
        adapter_config=_parakeet_config(),
    )
    later_calls: list[str] = []
    monkeypatch.setattr(
        streaming_stt_eval,
        "load_worker_manifest",
        lambda _path: manifest,
    )
    monkeypatch.setattr(
        streaming_stt_eval,
        "_verified_runtime_receipt_digest",
        lambda _manifest: later_calls.append("receipt"),
    )

    with pytest.raises(ValueError):
        streaming_stt_eval._run_benchmark_locked(
            tmp_path / "manifest.json",
            tmp_path / "corpus.json",
            scratch_parent=tmp_path / "scratch",
            repeats=1,
            stream=StreamConfig(640, "burst", 80, 48_000),
            chunk_samples=None,
            pace=None,
            partial_interval_ms=None,
            tail_padding_samples=None,
        )

    assert later_calls == []


def test_stream_overrides_keep_nemotron_native_defaults_for_unspecified_fields():
    manifest = SimpleNamespace(
        adapter=NEMOTRON_ADAPTER,
        adapter_config=NemotronConfig(),
    )

    selected = streaming_stt_eval._selected_stream(
        manifest,
        None,
        chunk_samples=None,
        pace="realtime",
        partial_interval_ms=None,
        tail_padding_samples=160,
    )

    assert selected == StreamConfig(5120, "realtime", 320, 160)


def test_moonshine_runtime_receipt_load_performs_the_tree_verification_once(
    tmp_path,
    monkeypatch,
):
    artifact = SimpleNamespace(
        path=tmp_path / "runtime-receipt.json",
        sha256="a" * 64,
    )
    manifest = SimpleNamespace(
        adapter=MOONSHINE_ADAPTER,
        artifact_by_name={"runtime-receipt": artifact},
        python=SimpleNamespace(path=tmp_path / "venv" / "bin" / "python"),
    )
    receipt = SimpleNamespace(digest=artifact.sha256)
    calls: list[object] = []

    def load(path, *, expected_digest):
        calls.append(("load", path, expected_digest))
        return receipt

    def verify_location(value, python_path):
        calls.append(("location", value, python_path))

    monkeypatch.setattr(streaming_stt_eval, "load_runtime_tree_receipt", load)
    monkeypatch.setattr(
        streaming_stt_eval,
        "verify_venv_runtime_location",
        verify_location,
    )

    assert (
        streaming_stt_eval._verified_runtime_receipt_digest(manifest) == artifact.sha256
    )
    assert calls == [
        ("load", artifact.path, artifact.sha256),
        ("location", receipt, manifest.python.path),
    ]


def test_parakeet_runtime_receipt_uses_bounded_tree_and_exact_config(
    tmp_path,
    monkeypatch,
):
    config = _parakeet_config()
    receipt_artifact = SimpleNamespace(
        path=tmp_path / "runtime-receipt.json",
        sha256="a" * 64,
    )
    wheel_lock = SimpleNamespace(sha256=config.wheel_lock_sha256)
    manifest = SimpleNamespace(
        adapter=PARAKEET_REALTIME_EOU_ADAPTER,
        adapter_config=config,
        artifact_by_name={
            "runtime-receipt": receipt_artifact,
            "runtime-wheel-lock": wheel_lock,
        },
        python=SimpleNamespace(path=tmp_path / "venv" / "bin" / "python"),
    )
    receipt = SimpleNamespace(
        digest=receipt_artifact.sha256,
        content_digest=config.runtime_content_sha256,
        file_count=config.runtime_file_count,
        total_size_bytes=config.runtime_total_size_bytes,
        files=(SimpleNamespace(size_bytes=config.runtime_maximum_file_bytes),),
    )
    calls: list[tuple[object, ...]] = []

    def load(path, *, expected_digest, limits):
        calls.append(("load", path, expected_digest, limits))
        return receipt

    def verify_location(value, python_path):
        calls.append(("location", value, python_path))

    monkeypatch.setattr(streaming_stt_eval, "load_runtime_tree_receipt", load)
    monkeypatch.setattr(
        streaming_stt_eval,
        "verify_venv_runtime_location",
        verify_location,
    )

    assert streaming_stt_eval._verified_runtime_receipt_digest(manifest) == "a" * 64
    assert calls == [
        (
            "load",
            receipt_artifact.path,
            receipt_artifact.sha256,
            PARAKEET_RUNTIME_TREE_LIMITS,
        ),
        ("location", receipt, manifest.python.path),
    ]


def test_controller_checks_runtime_receipt_boundary_before_and_after_worker(
    tmp_path,
    monkeypatch,
):
    manifest, corpus = _benchmark_fixture(tmp_path)
    calls: list[str] = []
    real_verify = streaming_stt_eval._verified_runtime_receipt_digest

    def record(candidate):
        calls.append(candidate.adapter)
        return real_verify(candidate)

    monkeypatch.setattr(
        streaming_stt_eval,
        "_verified_runtime_receipt_digest",
        record,
    )

    report = streaming_stt_eval.run_benchmark(
        manifest,
        corpus,
        scratch_parent=tmp_path / "scratch",
        repeats=1,
    )

    assert report["ok"] is True
    assert calls == ["fake-json-v1", "fake-json-v1"]


def test_cli_description_is_manifest_adapter_generic():
    help_text = " ".join(streaming_stt_eval._parser().format_help().split())

    assert "manifest-selected adapter" in help_text
    assert "fake worker receipts only" not in help_text


def test_host_run_lock_rejects_a_second_controller(tmp_path):
    lock_path = tmp_path / "host.lock"
    first = streaming_stt_eval._BenchmarkRunLock.acquire(lock_path)
    try:
        with pytest.raises(ValueError):
            streaming_stt_eval._BenchmarkRunLock.acquire(lock_path)
    finally:
        first.close()

    second = streaming_stt_eval._BenchmarkRunLock.acquire(lock_path)
    second.close()


def test_host_run_lock_path_is_independent_of_ambient_temp_directories(
    tmp_path,
    monkeypatch,
):
    expected = streaming_stt_eval._trusted_host_lock_path()

    monkeypatch.setenv("TMPDIR", str(tmp_path / "ambient-tmp"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "ambient-runtime"))

    assert streaming_stt_eval._trusted_host_lock_path() == expected
    assert expected.is_absolute()
    assert str(tmp_path) not in str(expected)


def test_host_run_lock_rename_never_permits_a_second_lease(tmp_path):
    lock_path = tmp_path / "host.lock"
    moved_path = tmp_path / "moved.lock"
    first = streaming_stt_eval._BenchmarkRunLock.acquire(lock_path)
    lock_path.rename(moved_path)

    with pytest.raises(ValueError):
        streaming_stt_eval._BenchmarkRunLock.acquire(lock_path)
    with pytest.raises(ValueError):
        first.assert_bound()
    with pytest.raises(ValueError):
        first.close()

    replacement = streaming_stt_eval._BenchmarkRunLock.acquire(lock_path)
    replacement.close()


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_host_run_lock_never_chmods_an_external_link_target(tmp_path, link_kind):
    external = tmp_path / "external-lock-target"
    external.write_text("sentinel", encoding="utf-8")
    external.chmod(0o640)
    original_mode = stat.S_IMODE(external.stat().st_mode)
    lock_path = tmp_path / "host.lock"
    if link_kind == "symlink":
        lock_path.symlink_to(external)
    else:
        os.link(external, lock_path)

    with pytest.raises(ValueError):
        streaming_stt_eval._BenchmarkRunLock.acquire(lock_path)

    assert external.read_text(encoding="utf-8") == "sentinel"
    assert stat.S_IMODE(external.stat().st_mode) == original_mode


def test_private_cleanup_never_chmods_linked_external_targets(tmp_path):
    external_file = tmp_path / "external-file"
    external_file.write_text("sentinel", encoding="utf-8")
    external_file.chmod(0o640)
    external_directory = tmp_path / "external-directory"
    external_directory.mkdir(mode=0o750)
    external_directory.chmod(0o750)
    file_mode = stat.S_IMODE(external_file.stat().st_mode)
    directory_mode = stat.S_IMODE(external_directory.stat().st_mode)

    scratch = tmp_path / "private-scratch"
    scratch.mkdir(mode=0o700)
    os.link(external_file, scratch / "hardlink")
    (scratch / "directory-link").symlink_to(
        external_directory, target_is_directory=True
    )
    locked = scratch / "locked"
    locked.mkdir()
    (locked / "payload").write_text("private", encoding="utf-8")
    locked.chmod(0o000)

    metadata = scratch.lstat()
    streaming_stt_eval._remove_private_scratch(
        scratch,
        expected_identity=(metadata.st_dev, metadata.st_ino),
    )

    assert not scratch.exists()
    assert external_file.read_text(encoding="utf-8") == "sentinel"
    assert stat.S_IMODE(external_file.stat().st_mode) == file_mode
    assert stat.S_IMODE(external_directory.stat().st_mode) == directory_mode


def test_private_cleanup_rejects_a_replaced_scratch_root(tmp_path):
    scratch = tmp_path / "private-scratch"
    scratch.mkdir(mode=0o700)
    metadata = scratch.lstat()
    original_identity = (metadata.st_dev, metadata.st_ino)
    moved = tmp_path / "moved-original"
    scratch.rename(moved)
    scratch.mkdir(mode=0o700)
    sentinel = scratch / "replacement-sentinel"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError):
        streaming_stt_eval._remove_private_scratch(
            scratch,
            expected_identity=original_identity,
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert moved.is_dir()


def test_scratch_creation_rejects_a_symlinked_ancestor_without_writing_through_it(
    tmp_path,
):
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir(mode=0o700)
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError):
        streaming_stt_eval._prepare_scratch(linked_parent / "nested")

    assert not (real_parent / "nested").exists()


def test_invalid_stream_is_rejected_before_inputs_or_worker_are_touched(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        streaming_stt_eval,
        "load_worker_manifest",
        lambda _path: (_ for _ in ()).throw(AssertionError),
    )

    with pytest.raises(ValueError):
        streaming_stt_eval.run_benchmark(
            tmp_path / "missing-worker.json",
            tmp_path / "missing-corpus.json",
            scratch_parent=tmp_path / "scratch",
            stream=StreamConfig(0, "burst", 100, 0),
        )

    assert not (tmp_path / "scratch").exists()


def test_controller_rejects_identical_worker_bytes_from_an_unbound_source_tree(
    tmp_path,
):
    manifest_path, corpus_path = _benchmark_fixture(tmp_path)
    copied_worker = tmp_path / "copied-tree" / "tools" / "streaming_stt" / "worker.py"
    copied_worker.parent.mkdir(parents=True)
    copied_worker.write_bytes(WORKER_PATH.read_bytes())
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["worker"] = bound_file(copied_worker)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        streaming_stt_eval.run_benchmark(
            manifest_path,
            corpus_path,
            scratch_parent=tmp_path / "scratch",
            repeats=1,
        )


def test_private_bundle_is_cleaned_after_worker_start_failure(tmp_path, monkeypatch):
    manifest_path, corpus_path = _benchmark_fixture(tmp_path)
    scratch_parent = tmp_path / "scratch"
    monkeypatch.setattr(
        streaming_stt_eval.StreamingWorker,
        "start",
        lambda _self: (_ for _ in ()).throw(RuntimeError("forced start failure")),
    )

    with pytest.raises(RuntimeError, match="forced start failure"):
        streaming_stt_eval.run_benchmark(
            manifest_path,
            corpus_path,
            scratch_parent=scratch_parent,
            repeats=1,
        )

    assert not any(scratch_parent.glob(".streaming-stt-*"))


def test_repo_ancestor_retarget_cannot_execute_alternate_modules_and_fails_report(
    tmp_path,
    monkeypatch,
):
    manifest_path, corpus_path = _benchmark_fixture(tmp_path)
    source_link = tmp_path / "worker-source"
    source_link.symlink_to(streaming_stt_eval._REPO_ROOT, target_is_directory=True)
    alternate_root = tmp_path / "alternate"
    alternate_worker = alternate_root / "tools" / "streaming_stt" / "worker.py"
    alternate_worker.parent.mkdir(parents=True)
    alternate_worker.write_bytes(WORKER_PATH.read_bytes())
    marker = tmp_path / "alternate-module-executed"
    alternate_adapter = (
        alternate_root / "tools" / "streaming_stt" / "adapters" / "__init__.py"
    )
    alternate_adapter.parent.mkdir()
    alternate_adapter.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n",
        encoding="utf-8",
    )

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["worker"]["path"] = str(
        source_link / "tools" / "streaming_stt" / "worker.py"
    )
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    real_stage = streaming_stt_eval.stage_worker_source_bundle

    def stage_then_retarget(repository_root, destination):
        bundle = real_stage(repository_root, destination)
        source_link.unlink()
        source_link.symlink_to(alternate_root, target_is_directory=True)
        return bundle

    monkeypatch.setattr(
        streaming_stt_eval,
        "stage_worker_source_bundle",
        stage_then_retarget,
    )

    with pytest.raises(ValueError):
        streaming_stt_eval.run_benchmark(
            manifest_path,
            corpus_path,
            scratch_parent=tmp_path / "scratch",
            repeats=1,
        )

    assert not marker.exists()


def test_metric_reducer_reports_exact_native_endpoint_source_and_tail_accounting(
    tmp_path,
):
    _, corpus_path = _benchmark_fixture(tmp_path)
    corpus = load_corpus(corpus_path)
    usage = ResourceUsage(rss_mb=10.0, threads=1, vram_mb=20.0)
    ready = _parakeet_ready(usage)

    records = (
        RunRecord(
            0,
            0,
            CaseTrace(
                partials=(),
                final=_native_final_event(
                    usage=usage,
                    text="stop now",
                    source=160,
                    source_consumed=160,
                    tail_consumed=80,
                    reason="eou",
                    authoritative=True,
                    probability=0.9,
                ),
            ),
        ),
        RunRecord(
            0,
            1,
            CaseTrace(
                partials=(),
                final=_native_final_event(
                    usage=usage,
                    text="stop now",
                    source=160,
                    source_consumed=80,
                    tail_consumed=0,
                    reason="eou",
                    authoritative=False,
                    probability=0.8,
                ),
            ),
        ),
        RunRecord(
            1,
            0,
            CaseTrace(
                partials=(),
                final=_native_final_event(
                    usage=usage,
                    text="",
                    source=80,
                    source_consumed=80,
                    tail_consumed=0,
                    reason="eob",
                    authoritative=False,
                    probability=0.7,
                ),
            ),
        ),
        RunRecord(
            1,
            1,
            CaseTrace(
                partials=(),
                final=_native_final_event(
                    usage=usage,
                    text="",
                    source=80,
                    source_consumed=40,
                    tail_consumed=0,
                    reason="eob",
                    authoritative=False,
                    probability=0.6,
                ),
            ),
        ),
    )

    metrics = aggregate_metrics(corpus, records, ready, repeats=2)

    assert metrics["coverage_complete"] is True
    assert metrics["throughput"] == {
        "audio_total_sec": 0.03,
        "compute_total_ms": 16.0,
        "aggregate_rtf": 0.5333,
        "aggregate_rtf_scope": "declared_source_audio",
        "model_input_total_sec": 0.028,
        "model_input_aggregate_rtf": 0.5818,
    }
    assert metrics["endpointing"] == {
        "evaluations": 4,
        "native_endpoints": 4,
        "native_eou_events": 2,
        "native_eob_backchannels": 2,
        "authoritative_endpoints": 1,
        "early_source_endpoints": 2,
        "source_early_native_events": 2,
        "source_early_eou_events": 1,
        "source_early_eob_backchannels": 1,
        "tail_exhaustions": 0,
        "reasons": {"eou": 2, "eob": 2, "tail_exhausted": 0},
        "source_samples": {
            "offered_total": 480,
            "consumed_total": 360,
            "dropped_total": 120,
        },
        "declared_tail_samples": {
            "offered_total": 640,
            "consumed_total": 80,
            "unconsumed_total": 560,
        },
        "terminal_model_padding_samples_total": 0,
        "model_input_samples_consumed_total": 440,
        "endpoint_sample_p50": 80.0,
        "endpoint_sample_p95": 240.0,
        "authoritative_endpoint_latency_p50_ms": 5.0,
        "authoritative_endpoint_latency_p95_ms": 5.0,
        "authoritative_endpoint_latency_max_ms": 5.0,
        "endpoint_probability_p50": 0.7,
        "endpoint_probability_p95": 0.9,
    }


@pytest.mark.parametrize(
    "ready_version,mutation",
    [
        (PROTOCOL_VERSION, {}),
        (NATIVE_ENDPOINT_PROTOCOL_VERSION, {"protocol_version": PROTOCOL_VERSION}),
        (NATIVE_ENDPOINT_PROTOCOL_VERSION, {"endpoint_probability": float("nan")}),
        (NATIVE_ENDPOINT_PROTOCOL_VERSION, {"endpoint_probability": float("inf")}),
        (NATIVE_ENDPOINT_PROTOCOL_VERSION, {"endpoint_probability": -0.01}),
        (NATIVE_ENDPOINT_PROTOCOL_VERSION, {"endpoint_probability": 1.01}),
        (NATIVE_ENDPOINT_PROTOCOL_VERSION, {"endpoint_probability": True}),
        (NATIVE_ENDPOINT_PROTOCOL_VERSION, {"audio_seconds": 0.011}),
    ],
)
def test_metric_reducer_rejects_constructed_native_event_seam_mutations(
    tmp_path,
    ready_version,
    mutation,
):
    _, corpus_path = _benchmark_fixture(tmp_path)
    corpus = load_corpus(corpus_path)
    usage = ResourceUsage(rss_mb=10.0, threads=1, vram_mb=20.0)
    final = replace(
        _native_final_event(
            usage=usage,
            text="stop now",
            source=160,
            source_consumed=160,
            tail_consumed=80,
            reason="eou",
            authoritative=True,
            probability=0.9,
        ),
        **mutation,
    )

    with pytest.raises(ValueError):
        aggregate_metrics(
            corpus,
            (RunRecord(0, 0, CaseTrace(partials=(), final=final)),),
            _parakeet_ready(usage, protocol_version=ready_version),
            repeats=1,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        {"authoritative": True},
        {"endpoint_latency_ms": 0.0},
    ],
)
def test_metric_reducer_rejects_response_authority_or_latency_on_complete_eob(
    tmp_path,
    mutation,
):
    _, corpus_path = _benchmark_fixture(tmp_path)
    corpus = load_corpus(corpus_path)
    usage = ResourceUsage(rss_mb=10.0, threads=1, vram_mb=20.0)
    final = replace(
        _native_final_event(
            usage=usage,
            text="stop now",
            source=160,
            source_consumed=160,
            tail_consumed=0,
            reason="eob",
            authoritative=False,
            probability=0.9,
        ),
        **mutation,
    )

    with pytest.raises(ValueError):
        aggregate_metrics(
            corpus,
            (RunRecord(0, 0, CaseTrace(partials=(), final=final)),),
            _parakeet_ready(usage),
            repeats=1,
        )


def test_metric_reducer_counts_final_disagreement_retraction_and_missing_stability(
    tmp_path,
):
    _, corpus_path = _benchmark_fixture(tmp_path)
    corpus = load_corpus(corpus_path)
    usage = ResourceUsage(rss_mb=10.0, threads=1, vram_mb=None)
    ready = ReadyEvent(
        model_id="fake-stream-v1",
        manifest_sha256="a" * 64,
        source_bundle_sha256="b" * 64,
        adapter="fake-json-v1",
        model_load_ms=5.0,
        resources=usage,
        runtime={"python": "3.12", "platform": "linux"},
    )

    def trace(final_text: str) -> CaseTrace:
        return CaseTrace(
            partials=(
                PartialEvent("x", 0, "stop now please", 80, 20.0, 1.0),
                PartialEvent("x", 1, "stop", 160, 40.0, 1.0),
            ),
            final=FinalEvent(
                "x",
                2,
                final_text,
                160,
                60.0,
                20.0,
                2.0,
                0.01,
                1,
                0,
                0.0,
                usage,
            ),
        )

    metrics = aggregate_metrics(
        corpus,
        (
            RunRecord(0, 0, trace("stop")),
            RunRecord(0, 1, trace("cancel")),
        ),
        ready,
        repeats=2,
    )

    assert metrics["coverage_complete"] is False
    assert metrics["determinism"]["cases_with_final_disagreement"] == 1
    assert metrics["streaming"]["retracted_words"] > 0
    assert metrics["latency"]["stable_partial_missing"] == 1


def test_metric_reducer_counts_last_partial_to_final_rewrite(tmp_path):
    _, corpus_path = _benchmark_fixture(tmp_path)
    corpus = load_corpus(corpus_path)
    usage = ResourceUsage(rss_mb=10.0, threads=1, vram_mb=None)
    ready = ReadyEvent(
        model_id="fake-stream-v1",
        manifest_sha256="a" * 64,
        source_bundle_sha256="b" * 64,
        adapter="fake-json-v1",
        model_load_ms=5.0,
        resources=usage,
        runtime={"python": "3.12", "platform": "linux"},
    )
    trace = CaseTrace(
        partials=(PartialEvent("x", 0, "stop now please", 160, 40.0, 1.0),),
        final=FinalEvent(
            "x",
            1,
            "cancel",
            160,
            60.0,
            20.0,
            2.0,
            0.01,
            1,
            0,
            0.0,
            usage,
        ),
    )

    metrics = aggregate_metrics(
        corpus,
        (RunRecord(0, 0, trace),),
        ready,
        repeats=1,
    )

    assert metrics["streaming"]["churn_token_edits"] > 0
    assert metrics["streaming"]["retracted_words"] == 3


def test_silence_punctuation_hallucinations_are_counted_nonempty(tmp_path):
    _, corpus_path = _benchmark_fixture(tmp_path)
    corpus = load_corpus(corpus_path)
    usage = ResourceUsage(rss_mb=10.0, threads=1, vram_mb=None)
    ready = ReadyEvent(
        model_id="fake-stream-v1",
        manifest_sha256="a" * 64,
        source_bundle_sha256="b" * 64,
        adapter="fake-json-v1",
        model_load_ms=5.0,
        resources=usage,
        runtime={"python": "3.12", "platform": "linux"},
    )
    trace = CaseTrace(
        partials=(PartialEvent("x", 0, "!!!", 80, 20.0, 1.0),),
        final=FinalEvent(
            "x",
            1,
            "???",
            80,
            30.0,
            10.0,
            2.0,
            0.005,
            1,
            0,
            0.0,
            usage,
        ),
    )

    metrics = aggregate_metrics(
        corpus,
        (RunRecord(1, 0, trace),),
        ready,
        repeats=1,
    )

    assert metrics["accuracy"]["silence_nonempty_partials"] == 1
    assert metrics["accuracy"]["silence_nonempty_finals"] == 1


def test_metric_reducer_rejects_finite_inputs_whose_sum_overflows(tmp_path):
    _, corpus_path = _benchmark_fixture(tmp_path)
    corpus = load_corpus(corpus_path)
    usage = ResourceUsage(rss_mb=10.0, threads=1, vram_mb=None)
    ready = ReadyEvent(
        model_id="fake-stream-v1",
        manifest_sha256="a" * 64,
        source_bundle_sha256="b" * 64,
        adapter="fake-json-v1",
        model_load_ms=5.0,
        resources=usage,
        runtime={"python": "3.12", "platform": "linux"},
    )
    trace = CaseTrace(
        partials=(),
        final=FinalEvent(
            "x",
            0,
            "stop now",
            160,
            30.0,
            10.0,
            1e308,
            0.01,
            1,
            0,
            0.0,
            usage,
        ),
    )

    with pytest.raises(ValueError, match="non-finite aggregate"):
        aggregate_metrics(
            corpus,
            (
                RunRecord(0, 0, trace),
                RunRecord(0, 1, trace),
            ),
            ready,
            repeats=2,
        )


def test_cli_writes_mode_600_report_and_never_prints_private_rows(
    tmp_path,
    capsys,
):
    manifest, corpus = _benchmark_fixture(tmp_path)
    output = tmp_path / "report.json"

    assert (
        streaming_stt_eval.main(
            [
                "--worker-manifest",
                str(manifest),
                "--corpus",
                str(corpus),
                "--scratch-root",
                str(tmp_path / "scratch"),
                "--repeats",
                "1",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    stdout = capsys.readouterr().out
    assert json.loads(stdout) == json.loads(output.read_text(encoding="utf-8"))
    assert "stop now" not in stdout
    assert "audio-0" not in stdout
    assert os.stat(output).st_mode & 0o777 == 0o600


def test_cli_failure_is_detail_free(monkeypatch, capsys):
    monkeypatch.setattr(
        streaming_stt_eval,
        "run_benchmark",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("SENTINEL_PRIVATE_TRANSCRIPT_AND_PATH")
        ),
    )

    assert (
        streaming_stt_eval.main(
            ["--worker-manifest", "/missing", "--corpus", "/missing"]
        )
        == 2
    )
    output = capsys.readouterr().out
    assert "SENTINEL" not in output
    assert json.loads(output) == streaming_stt_eval._SAFE_ERROR


def test_cli_never_writes_or_prints_nonstandard_infinity(tmp_path, monkeypatch, capsys):
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        streaming_stt_eval,
        "run_benchmark",
        lambda *_args, **_kwargs: {"ok": True, "metric": float("inf")},
    )

    assert (
        streaming_stt_eval.main(
            [
                "--worker-manifest",
                "/unused",
                "--corpus",
                "/unused",
                "--output",
                str(output),
            ]
        )
        == 2
    )

    stdout = capsys.readouterr().out
    assert "Infinity" not in stdout
    assert json.loads(stdout) == streaming_stt_eval._SAFE_ERROR
    assert not output.exists()
