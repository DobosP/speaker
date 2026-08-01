from __future__ import annotations

from array import array
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import streaming_stt_eval
from tools.streaming_stt.manifest import (
    FASTER_WHISPER_ENDPOINT_ADAPTER,
    FasterWhisperEndpointConfig,
)
from tools.streaming_stt.protocol import (
    ErrorEvent,
    FinalEvent,
    PartialEvent,
    PcmInput,
    ReadyEvent,
    ResourceUsage,
    StreamConfig,
    TranscribeRequest,
    encode_message,
    parse_response,
)
import tools.streaming_stt.worker as worker_module


_VERSIONS = {
    "faster-whisper": "1.2.1",
    "ctranslate2": "4.8.1",
    "numpy": "2.4.6",
    "nvidia-cublas-cu12": "12.9.2.10",
    "nvidia-cudnn-cu12": "9.24.0.43",
    "nvidia-cuda-nvrtc-cu12": "12.9.86",
}


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.advance(seconds)


class OversleepClock(FakeClock):
    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.advance(seconds + 0.005)


class FakeModel:
    def __init__(self, clock: FakeClock, *, decode_seconds: float = 0.05) -> None:
        self.clock = clock
        self.decode_seconds = decode_seconds
        self.calls: list[tuple[object, dict[str, object]]] = []

    def transcribe(self, samples, **kwargs):
        self.calls.append((samples.copy(), kwargs))
        self.clock.advance(self.decode_seconds)
        segments = (
            SimpleNamespace(text=" final"),
            SimpleNamespace(text=" answer "),
        )
        return (segment for segment in segments), SimpleNamespace(language="en")


class FakeFactory:
    def __init__(
        self,
        clock: FakeClock,
        model: FakeModel,
        *,
        load_seconds: float = 0.2,
    ) -> None:
        self.clock = clock
        self.model = model
        self.load_seconds = load_seconds
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        self.clock.advance(self.load_seconds)
        return self.model


def _api(clock: FakeClock, model: FakeModel):
    factory = FakeFactory(clock, model)
    return (
        SimpleNamespace(
            versions=dict(_VERSIONS),
            numpy=np,
            whisper_model=factory,
            supported_compute_types=lambda device, index: (
                {"float16"} if (device, index) == ("cuda", 0) else set()
            ),
        ),
        factory,
    )


def _request(
    tmp_path: Path,
    *,
    samples: int = 3200,
    tail: int = 800,
    chunk: int = 1000,
    pace: str = "burst",
) -> TranscribeRequest:
    raw = array("f", [0.25] * samples).tobytes()
    path = tmp_path / "case.f32le"
    path.write_bytes(raw)
    return TranscribeRequest(
        request_id="fw-endpoint-0",
        pcm=PcmInput(
            path=path.resolve(strict=True),
            sha256=hashlib.sha256(raw).hexdigest(),
            samples=samples,
        ),
        stream=StreamConfig(
            chunk_samples=chunk,
            pace=pace,
            partial_interval_ms=80,
            tail_padding_samples=tail,
        ),
    )


def test_adapter_buffers_complete_pcm_and_emits_one_final_only(tmp_path):
    from tools.streaming_stt.adapters.faster_whisper_endpoint import (
        FasterWhisperEndpointAdapter,
    )

    clock = FakeClock()
    model = FakeModel(clock)
    api, factory = _api(clock, model)
    adapter = FasterWhisperEndpointAdapter(
        tmp_path,
        config=FasterWhisperEndpointConfig(),
        api=api,
        clock=clock,
        sleeper=clock.sleep,
    )
    emitted: list[object] = []

    result = adapter.transcribe(_request(tmp_path), emit_partial=emitted.append)

    assert adapter.ready.model_load_ms == pytest.approx(200.0)
    assert emitted == []
    assert result.partials == ()
    assert result.final == "final answer"
    assert result.finalization_ms == pytest.approx(50.0)
    assert result.compute_ms == pytest.approx(50.0)
    assert result.elapsed_ms == pytest.approx(50.0)
    assert result.deadline_misses == 0
    assert result.max_backlog_ms == 0.0
    assert factory.calls == [
        (
            (str(tmp_path.resolve()),),
            {
                "device": "cuda",
                "device_index": 0,
                "compute_type": "float16",
                "cpu_threads": 1,
                "num_workers": 1,
                "local_files_only": True,
            },
        )
    ]
    decoded, options = model.calls[0]
    assert decoded.dtype == np.float32
    assert decoded.size == 3200  # Declared tail is paced, never decoded.
    assert options["language"] == "en"
    assert options["beam_size"] == 5
    assert options["vad_filter"] is False
    assert options["condition_on_previous_text"] is False
    assert options["word_timestamps"] is False


def test_realtime_replay_paces_chunks_without_inventing_partials(tmp_path):
    from tools.streaming_stt.adapters.faster_whisper_endpoint import (
        FasterWhisperEndpointAdapter,
    )

    clock = FakeClock()
    model = FakeModel(clock, decode_seconds=0.0)
    api, _factory = _api(clock, model)
    adapter = FasterWhisperEndpointAdapter(
        tmp_path,
        config=FasterWhisperEndpointConfig(),
        api=api,
        clock=clock,
        sleeper=clock.sleep,
    )

    result = adapter.transcribe(
        _request(
            tmp_path,
            samples=1600,
            tail=1600,
            chunk=800,
            pace="realtime",
        ),
        emit_partial=lambda *_args: pytest.fail("final-only adapter emitted partial"),
    )

    assert result.partials == ()
    assert sum(clock.sleeps) == pytest.approx(0.2)
    assert result.elapsed_ms == pytest.approx(200.0)
    assert result.deadline_misses == 0


def test_realtime_scheduler_overshoot_is_not_streaming_deadline_evidence(tmp_path):
    from tools.streaming_stt.adapters.faster_whisper_endpoint import (
        FasterWhisperEndpointAdapter,
    )

    clock = OversleepClock()
    model = FakeModel(clock, decode_seconds=0.0)
    api, _factory = _api(clock, model)
    adapter = FasterWhisperEndpointAdapter(
        tmp_path,
        config=FasterWhisperEndpointConfig(),
        api=api,
        clock=clock,
        sleeper=clock.sleep,
    )

    result = adapter.transcribe(
        _request(
            tmp_path,
            samples=1600,
            tail=1600,
            chunk=800,
            pace="realtime",
        ),
        emit_partial=lambda *_args: pytest.fail("final-only adapter emitted partial"),
    )

    assert result.elapsed_ms > 200.0
    assert result.deadline_misses == 0  # Required wire placeholder only.
    assert result.max_backlog_ms == 0.0  # Required wire placeholder only.


def test_adapter_rehashes_pcm_and_fails_detail_free_on_mutation(tmp_path):
    from tools.streaming_stt.adapters.faster_whisper_endpoint import (
        FasterWhisperEndpointAdapter,
        FasterWhisperEndpointAdapterError,
    )

    clock = FakeClock()
    model = FakeModel(clock)
    api, _factory = _api(clock, model)
    adapter = FasterWhisperEndpointAdapter(
        tmp_path,
        config=FasterWhisperEndpointConfig(),
        api=api,
        clock=clock,
    )
    request = _request(tmp_path)
    request.pcm.path.write_bytes(array("f", [0.5] * request.pcm.samples).tobytes())

    with pytest.raises(FasterWhisperEndpointAdapterError) as raised:
        adapter.transcribe(request, emit_partial=lambda *_args: None)

    assert str(raised.value) == ""
    assert model.calls == []


def test_cuda_wheel_resolution_is_bound_to_inserted_runtime_root(
    tmp_path,
    monkeypatch,
):
    from tools.streaming_stt.adapters import faster_whisper_endpoint as adapter_module

    runtime = tmp_path / "site-packages"
    package = runtime / "nvidia" / "cublas"
    library_directory = package / "lib"
    library_directory.mkdir(parents=True)
    monkeypatch.setattr(
        adapter_module,
        "find_spec",
        lambda _package: SimpleNamespace(submodule_search_locations=[str(package)]),
    )
    monkeypatch.setattr(adapter_module.sys, "path", ["source-bundle", str(runtime)])

    assert adapter_module._wheel_lib_directory("nvidia.cublas") == (
        library_directory.resolve(strict=True)
    )


def test_evaluator_provenance_labels_endpoint_final_only_not_streaming():
    config = FasterWhisperEndpointConfig()
    evidence = streaming_stt_eval._evidence_binding(
        FASTER_WHISPER_ENDPOINT_ADAPTER,
        pace="burst",
        adapter_config=config,
    )
    evaluator = streaming_stt_eval._evaluator_binding(FASTER_WHISPER_ENDPOINT_ADAPTER)

    assert evidence["streaming_recognizer"] is False
    assert evidence["partial_source"] == "none_final_only"
    assert evidence["external_endpoint_required"] is True
    assert evidence["endpointing"] is False
    assert evidence["adoption_authority"] is False
    assert evidence["streaming_deadline_metrics_applicable"] is False
    assert evidence["latency_evidence"] == "complete_pcm_to_candidate_final_only"
    assert evidence["decode_contract"]["cpu_threads"] == 1
    assert evaluator["real_candidate_model"] is True
    assert evaluator["model_executed"] is True
    assert (
        "tools/streaming_stt/adapters/faster_whisper_endpoint.py" in evaluator["files"]
    )


def test_evaluator_nulls_final_only_deadline_protocol_placeholders():
    nested_streaming = {
        "deadline_misses": 3,
        "max_backlog_p95_ms": 2.0,
        "max_backlog_ms": 4.0,
    }
    metrics: dict[str, object] = {
        "streaming": {
            "deadline_misses": 12,
            "max_backlog_p95_ms": 8.0,
            "max_backlog_ms": 9.0,
        },
        "strata": [
            {
                "tag": "noise-kitchen",
                "cases": 2,
                "metrics": {"streaming": nested_streaming},
            }
        ],
    }

    streaming_stt_eval._mark_endpoint_deadline_metrics_not_applicable(metrics)

    assert metrics["streaming"] == {
        "deadline_misses": None,
        "max_backlog_p95_ms": None,
        "max_backlog_ms": None,
        "deadline_metrics_applicable": False,
    }
    assert nested_streaming == {
        "deadline_misses": None,
        "max_backlog_p95_ms": None,
        "max_backlog_ms": None,
        "deadline_metrics_applicable": False,
    }


def test_schema_v6_bwrap_command_is_networkless_and_receipt_mounted(
    tmp_path,
    monkeypatch,
):
    from tools.streaming_stt import supervisor

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    source_root = tmp_path / "source"
    source_root.mkdir()
    worker_path = source_root / "worker.py"
    worker_path.write_bytes(b"worker")
    source_manifest = source_root / "source-bundle.json"
    source_manifest.write_bytes(b"source")
    venv = tmp_path / "venv"
    python = venv / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python")
    model_root = tmp_path / "model"
    model_root.mkdir()
    runtime_receipt = tmp_path / "runtime-receipt.json"
    runtime_receipt.write_bytes(b"runtime")
    model_receipt = tmp_path / "model-receipt.json"
    model_receipt.write_bytes(b"model")
    manifest_path = tmp_path / "worker-manifest.json"
    manifest_path.write_bytes(b"manifest")
    manifest = SimpleNamespace(
        path=manifest_path,
        worker=SimpleNamespace(path=worker_path),
        python=SimpleNamespace(path=python),
        artifact_by_name={
            "runtime-receipt": SimpleNamespace(
                path=runtime_receipt,
                sha256="a" * 64,
            ),
            "model-receipt": SimpleNamespace(
                path=model_receipt,
                sha256="b" * 64,
            ),
        },
    )
    source_bundle = SimpleNamespace(
        root=source_root,
        worker_path=worker_path,
        manifest_path=source_manifest,
        tree_sha256="c" * 64,
    )
    monkeypatch.setattr(
        supervisor,
        "load_runtime_tree_receipt",
        lambda *_args, **_kwargs: SimpleNamespace(root=tmp_path / "site-packages"),
    )
    monkeypatch.setattr(
        supervisor,
        "verify_venv_runtime_location",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        supervisor,
        "_verify_faster_whisper_runtime_layout",
        lambda *_args, **_kwargs: SimpleNamespace(root=model_root),
    )

    command = supervisor._faster_whisper_bwrap_command(
        manifest,
        scratch,
        source_bundle,
        bundle_descriptor=11,
        worker_descriptor=12,
        python_descriptor=13,
        device_nodes=(Path("/dev/nvidia0"),),
        system_ro_paths=(Path("/usr"),),
        proc_driver_available=False,
    )

    def has_mount(option: str, path: Path) -> bool:
        expected = [option, str(path), str(path)]
        return any(
            command[index : index + len(expected)] == expected
            for index in range(len(command) - len(expected) + 1)
        )

    assert command[0] == "/usr/bin/bwrap"
    assert "--unshare-net" in command
    assert command.count("--bind") == 1
    assert has_mount("--bind", scratch)
    assert has_mount("--ro-bind", venv)
    assert has_mount("--ro-bind", model_root)
    assert has_mount("--ro-bind", runtime_receipt)
    assert has_mount("--ro-bind", model_receipt)
    isolated = command.index("-I")
    assert command[isolated : isolated + 3] == ["-I", "-S", "-B"]
    assert not any("http" in value or "token" in value for value in command)


class StubEndpointAdapter:
    def __init__(self) -> None:
        self.close_calls = 0
        self.ready = SimpleNamespace(
            model_load_ms=5.0,
            resources=ResourceUsage(rss_mb=32.0, threads=1, vram_mb=None),
        )

    def transcribe(self, _request, *, emit_partial):
        del emit_partial
        return SimpleNamespace(
            partials=(),
            final="endpoint final",
            elapsed_ms=12.0,
            finalization_ms=12.0,
            compute_ms=12.0,
            deadline_misses=0,
            max_backlog_ms=0.0,
            resources=ResourceUsage(rss_mb=33.0, threads=1, vram_mb=None),
        )

    def close(self) -> None:
        self.close_calls += 1


class PartialEmittingEndpointAdapter(StubEndpointAdapter):
    def transcribe(self, _request, *, emit_partial):
        emit_partial(
            0,
            SimpleNamespace(
                text="forbidden partial",
                after_samples=800,
                elapsed_ms=5.0,
                decode_ms=1.0,
            ),
        )
        raise AssertionError("schema-v6 worker accepted an adapter partial")


class CandidateFailure(RuntimeError):
    pass


def _run_schema_v6_worker(
    tmp_path,
    monkeypatch,
    adapter,
):
    request = _request(tmp_path, samples=1600, tail=0, chunk=800)
    worker_path = Path(worker_module.__file__).resolve(strict=True)
    worker_receipt = SimpleNamespace(
        sha256="a" * 64,
        size_bytes=worker_path.stat().st_size,
    )
    python_receipt = SimpleNamespace(
        sha256="b" * 64,
        size_bytes=worker_path.stat().st_size,
    )
    manifest = SimpleNamespace(
        path=tmp_path / "worker-manifest.json",
        digest="c" * 64,
        schema_version=6,
        model_id="faster-whisper-worker-test",
        adapter=FASTER_WHISPER_ENDPOINT_ADAPTER,
        adapter_config=FasterWhisperEndpointConfig(),
        worker=worker_receipt,
        python=python_receipt,
    )
    source_bundle = SimpleNamespace(
        tree_sha256="d" * 64,
        worker_relative_path="tools/streaming_stt/worker.py",
        worker_path=worker_path,
        file_by_path={"tools/streaming_stt/worker.py": worker_receipt},
    )
    output: list[dict[str, object]] = []
    monkeypatch.setattr(worker_module, "_SOURCE_BUNDLE_SHA256", "d" * 64)
    monkeypatch.setattr(
        worker_module,
        "load_source_bundle",
        lambda *_args, **_kwargs: source_bundle,
    )
    monkeypatch.setattr(
        worker_module,
        "load_worker_manifest",
        lambda *_args, **_kwargs: manifest,
    )
    monkeypatch.setattr(
        worker_module,
        "hash_regular_bounded",
        lambda *_args, **_kwargs: SimpleNamespace(sha256=python_receipt.sha256),
    )
    monkeypatch.setattr(
        worker_module,
        "_verify_worker_state",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        worker_module,
        "_activate_candidate_runtime",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        worker_module,
        "_create_adapter",
        lambda *_args, **_kwargs: (adapter, CandidateFailure),
    )
    monkeypatch.setattr(worker_module, "_write", output.append)
    monkeypatch.setattr(
        worker_module.sys,
        "stdin",
        SimpleNamespace(buffer=io.BytesIO(encode_message(request.as_dict()))),
    )

    result = worker_module._run(
        manifest.path,
        tmp_path,
        source_bundle_manifest=Path("unused.json"),
        source_bundle_sha256=source_bundle.tree_sha256,
    )
    responses = [parse_response(encode_message(event)) for event in output]
    return request, result, responses


def test_schema_v6_worker_serializes_final_with_zero_partials(
    tmp_path,
    monkeypatch,
):
    adapter = StubEndpointAdapter()
    request, result, responses = _run_schema_v6_worker(
        tmp_path,
        monkeypatch,
        adapter,
    )

    assert result == 0
    assert isinstance(responses[0], ReadyEvent)
    final = next(event for event in responses if isinstance(event, FinalEvent))
    assert final.seq == 0
    assert final.text == "endpoint final"
    assert final.samples_seen == request.pcm.samples
    assert adapter.close_calls == 1


def test_schema_v6_worker_rejects_adapter_partial_before_serialization(
    tmp_path,
    monkeypatch,
):
    adapter = PartialEmittingEndpointAdapter()
    _request_value, result, responses = _run_schema_v6_worker(
        tmp_path,
        monkeypatch,
        adapter,
    )

    assert result == 2
    assert isinstance(responses[0], ReadyEvent)
    assert isinstance(responses[-1], ErrorEvent)
    assert responses[-1].code == "candidate_failure"
    assert responses[-1].fatal is True
    assert not any(isinstance(event, PartialEvent) for event in responses)
    assert not any(isinstance(event, FinalEvent) for event in responses)
    assert adapter.close_calls == 1


@pytest.mark.parametrize("event_kind", ["partial", "final-seq-one"])
def test_schema_v6_supervisor_rejects_partial_or_nonzero_final_sequence(
    tmp_path,
    monkeypatch,
    event_kind,
):
    from tools.streaming_stt import supervisor

    request = _request(tmp_path, samples=1600, tail=0, chunk=800)
    manifest = SimpleNamespace(
        schema_version=6,
        adapter=FASTER_WHISPER_ENDPOINT_ADAPTER,
        adapter_config=FasterWhisperEndpointConfig(),
        limits=SimpleNamespace(case_timeout_sec=2.0),
    )
    controlled = supervisor.StreamingWorker(
        manifest,
        tmp_path,
        SimpleNamespace(),
        monotonic=lambda: 0.0,
    )
    controlled.ready = SimpleNamespace()
    if event_kind == "partial":
        event = PartialEvent(
            request_id=request.request_id,
            seq=0,
            text="forbidden partial",
            samples_seen=800,
            elapsed_ms=5.0,
            decode_ms=1.0,
        )
    else:
        event = FinalEvent(
            request_id=request.request_id,
            seq=1,
            text="forbidden sequence",
            samples_seen=request.pcm.samples,
            elapsed_ms=10.0,
            finalization_ms=5.0,
            compute_ms=5.0,
            audio_seconds=request.pcm.samples / request.pcm.sample_rate,
            chunks=2,
            deadline_misses=0,
            max_backlog_ms=0.0,
            resources=ResourceUsage(rss_mb=1.0, threads=1, vram_mb=None),
            model_padding_samples=0,
        )
    monkeypatch.setattr(controlled, "_verify_pcm", lambda _request: None)
    monkeypatch.setattr(controlled, "_send", lambda _message: None)
    monkeypatch.setattr(controlled, "_next_event", lambda _timeout: event)

    with pytest.raises(supervisor.WorkerError, match="worker_protocol"):
        controlled.transcribe(request)
