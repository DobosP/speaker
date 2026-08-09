from __future__ import annotations

from pathlib import Path
import os
import sys
from types import SimpleNamespace

import pytest

from tools.streaming_stt.manifest import KYUTAI_ADAPTER, KyutaiConfig, ManifestError
from tools.streaming_stt.protocol import PcmInput, StreamConfig, TranscribeRequest
from tools.streaming_stt.runtime_receipt import KYUTAI_RUNTIME_TREE_LIMITS
from tools.streaming_stt.source_bundle import WORKER_SOURCE_FILES
from tools.streaming_stt.supervisor import (
    WorkerError,
    _expected_final_evidence,
    _kyutai_bwrap_command,
    _kyutai_systemd_scope_command,
    _verify_kyutai_cgroup_evidence,
    _verify_kyutai_host_memory_available,
)
from tools.streaming_stt.worker import (
    _BOOTSTRAP_SOURCE_FILES,
    _verify_kyutai_cuda_preflight,
)
import tools.streaming_stt.worker as worker_module
import tools.streaming_stt_eval as eval_module
from tools.streaming_stt_eval import (
    _KYUTAI_CGROUP_EVIDENCE,
    _default_stream,
    _evaluator_binding,
    _evidence_binding,
    _mark_kyutai_finalization_metrics_not_applicable,
    _selected_stream,
    _validated_parakeet_cgroup_evidence,
    _worker_report,
)


def _manifest(*, config: KyutaiConfig | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        adapter=KYUTAI_ADAPTER,
        schema_version=9,
        adapter_config=KyutaiConfig() if config is None else config,
    )


def test_kyutai_source_and_runtime_closures_are_bounded():
    relative = "tools/streaming_stt/adapters/kyutai.py"
    assert relative in WORKER_SOURCE_FILES
    assert relative in _BOOTSTRAP_SOURCE_FILES
    assert KYUTAI_RUNTIME_TREE_LIMITS.maximum_files == 20_000
    assert KYUTAI_RUNTIME_TREE_LIMITS.maximum_aggregate_bytes == 6 * 1024**3
    config = KyutaiConfig()
    assert config.wheel_lock_sha256 == (
        "8d0e41563bb5e91500af42a912c5e6f825f16a67537ddc350556303e88609e25"
    )
    assert config.runtime_content_sha256 == (
        "9edc9c42b8c718d0e4b17d917c04c05acc0b5ecca4ec9504866ad34f1f62dbf0"
    )
    assert (
        config.runtime_file_count,
        config.runtime_total_size_bytes,
        config.runtime_maximum_file_bytes,
    ) == (16_547, 5_694_993_765, 984_633_129)
    evaluator = _evaluator_binding(KYUTAI_ADAPTER)
    assert evaluator["production_model"] is False
    assert evaluator["real_candidate_model"] is True
    assert relative in evaluator["files"]


def test_kyutai_runtime_receipt_is_content_and_lock_bound(monkeypatch):
    config = KyutaiConfig()
    runtime_artifact = SimpleNamespace(
        name="runtime-receipt",
        path=Path("/private/runtime-receipt.json"),
        sha256="c" * 64,
    )
    lock_artifact = SimpleNamespace(
        name="runtime-wheel-lock",
        path=Path("/private/kyutai-runtime.lock.json"),
        sha256=config.wheel_lock_sha256,
    )
    manifest = SimpleNamespace(
        adapter=KYUTAI_ADAPTER,
        schema_version=9,
        adapter_config=config,
        artifact_by_name={
            runtime_artifact.name: runtime_artifact,
            lock_artifact.name: lock_artifact,
        },
        python=SimpleNamespace(path=Path(os.path.abspath(sys.executable))),
    )
    receipt = SimpleNamespace(
        digest=runtime_artifact.sha256,
        content_digest=config.runtime_content_sha256,
        file_count=config.runtime_file_count,
        total_size_bytes=config.runtime_total_size_bytes,
        files=(SimpleNamespace(size_bytes=config.runtime_maximum_file_bytes),),
        root=Path("/private/venv/lib/python3.12/site-packages"),
    )
    monkeypatch.setattr(
        worker_module, "load_runtime_tree_receipt", lambda *_a, **_k: receipt
    )
    monkeypatch.setattr(worker_module, "verify_venv_runtime_location", lambda *_: None)
    monkeypatch.setattr(worker_module, "_verify_python312_venv_layout", lambda *_: None)
    monkeypatch.setattr(
        eval_module, "load_runtime_tree_receipt", lambda *_a, **_k: receipt
    )
    monkeypatch.setattr(eval_module, "verify_venv_runtime_location", lambda *_: None)

    assert worker_module._verify_runtime_receipt(manifest) is receipt
    assert eval_module._verified_runtime_receipt_digest(manifest) == receipt.digest

    manifest.artifact_by_name["runtime-wheel-lock"] = SimpleNamespace(sha256="d" * 64)
    with pytest.raises(ManifestError):
        worker_module._verify_runtime_receipt(manifest)
    with pytest.raises(ValueError):
        eval_module._verified_runtime_receipt_digest(manifest)


def test_kyutai_stream_contract_and_evidence_have_no_endpoint_authority():
    manifest = _manifest()
    expected = StreamConfig(
        chunk_samples=1_280,
        pace="burst",
        partial_interval_ms=160,
        tail_padding_samples=16_000,
    )
    assert _default_stream(manifest) == expected
    assert (
        _selected_stream(
            manifest,
            None,
            chunk_samples=None,
            pace=None,
            partial_interval_ms=None,
            tail_padding_samples=None,
        )
        == expected
    )
    with pytest.raises(ValueError):
        _selected_stream(
            manifest,
            None,
            chunk_samples=None,
            pace=None,
            partial_interval_ms=320,
            tail_padding_samples=None,
        )
    evidence = _evidence_binding(
        KYUTAI_ADAPTER,
        pace="burst",
        adapter_config=manifest.adapter_config,
    )
    assert evidence["production_model"] is False
    assert evidence["endpointing"] is False
    assert evidence["endpoint_evidence"] is False
    assert evidence["early_stop"] is False
    assert evidence["streaming_fidelity"] == (
        "native-moshi-lm-frame-step-after-noncausal-resample"
    )
    assert evidence["finalization_latency_scope"] == (
        "last_declared_tail_frame_candidate_step"
    )
    assert evidence["finalization_is_speech_end_latency"] is False
    assert evidence["finalization_metrics_applicable"] is False
    assert evidence["semantic_heads"] == {
        "executed": True,
        "shape_and_finiteness_validated": True,
        "policy": "diagnostic-finite-only",
        "count": 4,
        "dimension": 6,
        "diagnostic_only": True,
        "endpoint_authority": False,
        "early_stop": False,
    }
    assert evidence["resource_control"]["hard_vram_limit"] is False
    assert evidence["resource_control"]["cuda_graph"] is False
    assert evidence["resource_control"]["no_cuda_graph_env"] == "1"
    assert evidence["native_stream_geometry"]["resampling_mode"] == (
        "whole-buffer-noncausal"
    )
    assert evidence["native_stream_geometry"]["resampling_scope"] == (
        "complete-source-tail-alignment-once-noncausal"
    )
    assert evidence["native_stream_geometry"]["initial_frame_policy"] == (
        "duplicate-first-frame-prime"
    )
    assert evidence["native_stream_geometry"]["initial_frame_prime_steps"] == 1
    assert evidence["native_stream_geometry"]["sample_accounting_domain"] == (
        "input-16000hz-source-tail"
    )


def test_kyutai_finalization_placeholder_cannot_be_compared_as_latency():
    nested = {
        "latency": {
            "finalization_p50_ms": 1.0,
            "finalization_p95_ms": 2.0,
            "finalization_max_ms": 3.0,
        }
    }
    metrics = {
        "latency": {
            "finalization_p50_ms": 4.0,
            "finalization_p95_ms": 5.0,
            "finalization_max_ms": 6.0,
        },
        "strata": [{"tag": "far-field", "cases": 1, "metrics": nested}],
    }

    _mark_kyutai_finalization_metrics_not_applicable(metrics)

    assert metrics["latency"] == {
        "finalization_p50_ms": None,
        "finalization_p95_ms": None,
        "finalization_max_ms": None,
        "finalization_metrics_applicable": False,
    }
    assert nested["latency"] == metrics["latency"]


def test_kyutai_supervisor_derives_exact_chunks_and_wire_padding(tmp_path):
    request = TranscribeRequest(
        request_id="case-0000-r0",
        pcm=PcmInput(
            path=(tmp_path / "case.f32le").resolve(),
            sha256="0" * 64,
            samples=1_000,
        ),
        stream=_default_stream(_manifest()),
    )
    assert _expected_final_evidence(_manifest(), request) == (14, 920)
    invalid = TranscribeRequest(
        request_id=request.request_id,
        pcm=request.pcm,
        stream=StreamConfig(
            chunk_samples=1_280,
            pace="burst",
            partial_interval_ms=320,
            tail_padding_samples=16_000,
        ),
    )
    with pytest.raises(WorkerError, match="invalid_request"):
        _expected_final_evidence(_manifest(), invalid)


def test_kyutai_host_memory_preflight_is_exact_and_fail_closed(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal: 1 kB\nMemAvailable: 12582912 kB\n",
        encoding="ascii",
    )
    _verify_kyutai_host_memory_available(meminfo_path=meminfo)

    meminfo.write_text("MemAvailable: 12582911 kB\n", encoding="ascii")
    with pytest.raises(WorkerError, match="worker_prerequisite"):
        _verify_kyutai_host_memory_available(meminfo_path=meminfo)

    meminfo.write_text(
        "MemAvailable: 12582912 kB\nMemAvailable: 12582912 kB\n",
        encoding="ascii",
    )
    with pytest.raises(WorkerError, match="worker_prerequisite"):
        _verify_kyutai_host_memory_available(meminfo_path=meminfo)


def _write_kyutai_cgroup(root: Path) -> bytes:
    scope = root / "user.slice" / "run-speaker.scope"
    scope.mkdir(parents=True)
    (root / "cgroup.controllers").write_text("cpu io memory pids\n", encoding="ascii")
    values = {
        "memory.high": "5368709120\n",
        "memory.max": "6442450944\n",
        "memory.swap.max": "0\n",
        "memory.oom.group": "1\n",
        "cpu.max": "100000 100000\n",
        "pids.max": "128\n",
        "io.weight": "default 1\n",
    }
    for name, value in values.items():
        (scope / name).write_text(value, encoding="ascii")
    return b"0::/user.slice/run-speaker.scope\n"


def test_kyutai_cgroup_and_scope_bind_every_resource_limit(tmp_path):
    root = tmp_path / "cgroup"
    payload = _write_kyutai_cgroup(root)
    assert _verify_kyutai_cgroup_evidence(payload, cgroup_root=root) == (
        _KYUTAI_CGROUP_EVIDENCE
    )
    assert (
        _validated_parakeet_cgroup_evidence(
            _manifest(),
            _KYUTAI_CGROUP_EVIDENCE,
        )
        == _KYUTAI_CGROUP_EVIDENCE
    )
    command = _kyutai_systemd_scope_command(
        ["/usr/bin/bwrap", "--unshare-net", "--", "worker"],
        bwrap_descriptor=11,
    )
    properties = {
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--property"
    }
    assert properties == {
        "MemoryHigh=5368709120",
        "MemoryMax=6442450944",
        "MemorySwapMax=0",
        "CPUQuota=100%",
        "TasksMax=128",
        "IOWeight=1",
        "IOSchedulingClass=idle",
        "OOMPolicy=kill",
    }
    (root / "user.slice" / "run-speaker.scope" / "io.weight").write_text(
        "default 2\n", encoding="ascii"
    )
    with pytest.raises(WorkerError, match="worker_prerequisite"):
        _verify_kyutai_cgroup_evidence(payload, cgroup_root=root)


def test_kyutai_report_binds_complete_cgroup_phase_observations():
    observations = {
        "schema_version": 1,
        "source": "supervisor_cgroup_v2",
        "scope": "systemd_user_scope_all_descendants",
        "os_cache_control": "uncontrolled",
        "cache_state_claim": "none",
        "cache_eviction_performed": False,
        "wall_clock_origin": "supervisor_immediately_before_popen",
        "memory_current_scope": "point_sample_including_cgroup_charged_cache",
        "memory_peak_scope": "cumulative_since_scope_creation_not_phase_local",
        "cpu_stat_scope": "cumulative_since_scope_creation",
        "snapshot_atomicity": "sequential_cgroup_file_reads_not_atomic",
        "resident_scope": "pre_shutdown_after_requested_cases_no_idle_settle",
        "boundaries": [
            {
                "phase": phase,
                "boundary": boundary,
                "completed_cases": completed,
                "wall_elapsed_ms": float(index + 1),
                "memory_current_bytes": 100 + index,
                "memory_peak_bytes": 110 + index,
                "cpu_usage_usec": 30 + index,
                "cpu_user_usec": 20 + index,
                "cpu_system_usec": 10 + index,
            }
            for index, (phase, boundary, completed) in enumerate(
                (
                    ("startup", "validated_ready", 0),
                    ("first_use", "first_validated_terminal", 1),
                    ("resident", "pre_shutdown_after_suite", 3),
                )
            )
        ],
    }
    report = _worker_report(
        _manifest(),
        {"adapter": KYUTAI_ADAPTER},
        {"python": "3.12.3", "platform": "linux"},
        _KYUTAI_CGROUP_EVIDENCE,
        observations,
        expected_completed_cases=3,
    )
    assert report["cgroup_evidence"] == _KYUTAI_CGROUP_EVIDENCE
    assert report["resource_observations"] == observations


class _FakeCuda:
    def __init__(self, *, free_bytes: int = 9 * 1024**3) -> None:
        self.free_bytes = free_bytes
        self.fraction_calls: list[tuple[float, int]] = []

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def device_count() -> int:
        return 1

    def mem_get_info(self, device: int) -> tuple[int, int]:
        assert device == 0
        return self.free_bytes, 16 * 1024**3

    def set_per_process_memory_fraction(self, fraction: float, device: int) -> None:
        self.fraction_calls.append((fraction, device))


class _FakeTorch:
    __version__ = "2.7.1+cu126"
    version = SimpleNamespace(cuda="12.6")

    def __init__(self, *, free_bytes: int = 9 * 1024**3) -> None:
        self.cuda = _FakeCuda(free_bytes=free_bytes)
        self.thread_calls: list[tuple[str, int]] = []

    def set_num_threads(self, value: int) -> None:
        self.thread_calls.append(("intra", value))

    def set_num_interop_threads(self, value: int) -> None:
        self.thread_calls.append(("interop", value))


def test_kyutai_worker_preflight_requires_free_vram_before_load(monkeypatch):
    monkeypatch.setenv("NO_TORCH_COMPILE", "1")
    monkeypatch.setenv("NO_CUDA_GRAPH", "1")
    torch_api = _FakeTorch()
    _verify_kyutai_cuda_preflight(_manifest(), torch_api=torch_api)
    assert torch_api.thread_calls == []
    assert torch_api.cuda.fraction_calls == []

    with pytest.raises(ManifestError):
        _verify_kyutai_cuda_preflight(
            _manifest(),
            torch_api=_FakeTorch(free_bytes=8 * 1024**3 - 1),
        )
    monkeypatch.setenv("NO_CUDA_GRAPH", "0")
    with pytest.raises(ManifestError):
        _verify_kyutai_cuda_preflight(_manifest(), torch_api=_FakeTorch())


def _direct_model_manifest(tmp_path: Path) -> tuple[SimpleNamespace, SimpleNamespace]:
    model_root = tmp_path / "model"
    model_root.mkdir()
    model_names = {
        "model-config": "config.json",
        "model-weights": "model.safetensors",
        "model-mimi": "mimi-pytorch-e351c8d8@125.safetensors",
        "model-tokenizer": "tokenizer_en_fr_audio_8000.model",
    }
    artifacts = []
    for name, basename in model_names.items():
        path = model_root / basename
        path.touch()
        artifacts.append(SimpleNamespace(name=name, path=path))
    runtime_receipt = SimpleNamespace(
        name="runtime-receipt", path=tmp_path / "runtime-receipt.json"
    )
    runtime_lock = SimpleNamespace(
        name="runtime-wheel-lock", path=tmp_path / "runtime.lock.json"
    )
    artifacts.extend((runtime_receipt, runtime_lock))
    manifest = SimpleNamespace(
        artifacts=tuple(artifacts),
        artifact_by_name={item.name: item for item in artifacts},
        python=SimpleNamespace(path=(tmp_path / "venv/bin/python")),
        path=tmp_path / "manifest.json",
        worker=SimpleNamespace(path=tmp_path / "worker.py"),
    )
    source = SimpleNamespace(
        root=tmp_path / "source",
        worker_path=tmp_path / "source/tools/streaming_stt/worker.py",
        manifest_path=tmp_path / "source/source-bundle.json",
        tree_sha256="a" * 64,
    )
    return manifest, source


def test_kyutai_bwrap_seals_exact_direct_model_root(tmp_path):
    manifest, source = _direct_model_manifest(tmp_path)
    command = _kyutai_bwrap_command(
        manifest,
        tmp_path / "scratch",
        source,
        bundle_descriptor=7,
        worker_descriptor=8,
        python_descriptor=9,
        device_nodes=(),
        system_ro_paths=(),
        proc_driver_available=False,
    )
    assert "--unshare-net" in command
    model_root = str(tmp_path / "model")
    assert [
        command[index : index + 3]
        for index, value in enumerate(command[:-2])
        if value == "--ro-bind" and command[index + 1] == model_root
    ] == [["--ro-bind", model_root, model_root]]

    (tmp_path / "model" / "unexpected").touch()
    with pytest.raises(WorkerError, match="worker_prerequisite"):
        _kyutai_bwrap_command(
            manifest,
            tmp_path / "scratch",
            source,
            bundle_descriptor=7,
            worker_descriptor=8,
            python_descriptor=9,
            device_nodes=(),
            system_ro_paths=(),
            proc_driver_available=False,
        )
