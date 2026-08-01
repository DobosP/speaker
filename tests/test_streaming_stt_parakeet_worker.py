from __future__ import annotations

from array import array
from dataclasses import replace
import hashlib
import io
import os
from pathlib import Path
from types import SimpleNamespace

from tools.streaming_stt.manifest import (
    PARAKEET_REALTIME_EOU_ADAPTER,
    ParakeetRealtimeEouConfig,
)
from tools.streaming_stt.protocol import (
    ByeEvent,
    ErrorEvent,
    NATIVE_ENDPOINT_PROTOCOL_VERSION,
    NativeFinalEvent,
    PartialEvent,
    PcmInput,
    ReadyEvent,
    ResourceUsage,
    ShutdownEvent,
    ShutdownRequest,
    StreamConfig,
    TranscribeRequest,
    encode_message,
    parse_response,
)
import tools.streaming_stt.worker as worker_module
from tools.streaming_stt.runtime_receipt import PARAKEET_RUNTIME_TREE_LIMITS


class CandidateFailure(RuntimeError):
    pass


def _config() -> ParakeetRealtimeEouConfig:
    return ParakeetRealtimeEouConfig()


def _request(scratch: Path) -> TranscribeRequest:
    samples = array("f", [0.0, 0.1, -0.1, 0.0])
    raw = samples.tobytes()
    pcm = scratch / "worker-case.f32le"
    pcm.write_bytes(raw)
    pcm.chmod(0o600)
    return TranscribeRequest(
        request_id="parakeet-worker-0",
        pcm=PcmInput(
            path=pcm.resolve(strict=True),
            sha256=hashlib.sha256(raw).hexdigest(),
            samples=len(samples),
        ),
        stream=StreamConfig(
            chunk_samples=1_280,
            pace="burst",
            partial_interval_ms=80,
            tail_padding_samples=48_000,
        ),
        protocol_version=NATIVE_ENDPOINT_PROTOCOL_VERSION,
    )


class StubCandidate:
    def __init__(
        self,
        *,
        invalid_chunks: bool = False,
        endpoint_reason: str = "eou",
    ) -> None:
        self.invalid_chunks = invalid_chunks
        self.endpoint_reason = endpoint_reason
        self.close_calls = 0
        self.ready = SimpleNamespace(
            model_load_ms=12.5,
            resources=ResourceUsage(rss_mb=42.0, threads=1, vram_mb=512.0),
        )

    def transcribe(self, request, *, emit_partial):
        partial = SimpleNamespace(
            text="find",
            after_samples=request.pcm.samples,
            elapsed_ms=4.0,
            decode_ms=1.0,
        )
        emit_partial(17, partial)
        tail_seen = 1_280
        samples_seen = request.pcm.samples + tail_seen
        return SimpleNamespace(
            partials=(partial,),
            final="find in my vault",
            elapsed_ms=90.0,
            finalization_ms=2.0,
            compute_ms=6.0,
            deadline_misses=0,
            max_backlog_ms=0.0,
            resources=ResourceUsage(rss_mb=43.0, threads=1, vram_mb=520.0),
            source_samples=request.pcm.samples,
            source_samples_consumed=request.pcm.samples,
            declared_tail_samples=request.stream.tail_padding_samples,
            tail_samples_consumed=tail_seen,
            chunks_yielded=1 if self.invalid_chunks else 2,
            model_padding_samples=2 * 1_280 - samples_seen,
            endpoint_reason=self.endpoint_reason,
            native_endpoint=True,
            endpoint_probability=0.75,
            endpoint_sample=2 * 1_280,
            endpoint_latency_ms=(
                (tail_seen + 2 * 1_280 - samples_seen) / 16.0
                if self.endpoint_reason == "eou"
                else None
            ),
            authoritative=self.endpoint_reason == "eou",
        )

    def close(self) -> None:
        self.close_calls += 1


def _bind_run(monkeypatch, tmp_path: Path, adapter: StubCandidate):
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
        schema_version=5,
        model_id="parakeet-worker-test",
        adapter=PARAKEET_REALTIME_EOU_ADAPTER,
        adapter_config=_config(),
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

    monkeypatch.setattr(
        worker_module,
        "_SOURCE_BUNDLE_SHA256",
        source_bundle.tree_sha256,
    )
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
    return manifest, source_bundle, output


def _run_bound_worker(
    monkeypatch,
    manifest,
    source_bundle,
    scratch: Path,
    messages: bytes,
) -> int:
    monkeypatch.setattr(
        worker_module.sys,
        "stdin",
        SimpleNamespace(buffer=io.BytesIO(messages)),
    )
    return worker_module._run(
        manifest.path,
        scratch,
        source_bundle_manifest=Path("unused-source-bundle.json"),
        source_bundle_sha256=source_bundle.tree_sha256,
    )


def test_schema_v5_worker_emits_exact_native_terminal_and_v3_lifecycle(
    tmp_path,
    monkeypatch,
):
    request = _request(tmp_path)
    adapter = StubCandidate()
    manifest, bundle, output = _bind_run(monkeypatch, tmp_path, adapter)
    messages = encode_message(request.as_dict()) + encode_message(
        ShutdownRequest(
            "parakeet-shutdown",
            protocol_version=NATIVE_ENDPOINT_PROTOCOL_VERSION,
        ).as_dict()
    )

    result = _run_bound_worker(
        monkeypatch,
        manifest,
        bundle,
        tmp_path,
        messages,
    )
    responses = [parse_response(encode_message(event)) for event in output]

    assert result == 0
    assert type(responses[0]) is ReadyEvent
    assert responses[0].protocol_version == NATIVE_ENDPOINT_PROTOCOL_VERSION
    partial = next(event for event in responses if type(event) is PartialEvent)
    assert partial.seq == 0
    assert partial.protocol_version == NATIVE_ENDPOINT_PROTOCOL_VERSION
    final = next(event for event in responses if type(event) is NativeFinalEvent)
    assert final.seq == 1
    assert final.samples_seen == 1_284
    assert final.source_samples_consumed == 4
    assert final.tail_samples_consumed == 1_280
    assert final.declared_tail_samples == 48_000
    assert final.endpoint_reason == "eou"
    assert final.endpoint_sample == 2_560
    assert final.endpoint_latency_ms == 159.75
    assert final.authoritative is True
    assert type(responses[-2]) is ShutdownEvent
    assert type(responses[-1]) is ByeEvent
    assert responses[-2].protocol_version == NATIVE_ENDPOINT_PROTOCOL_VERSION
    assert responses[-1].protocol_version == NATIVE_ENDPOINT_PROTOCOL_VERSION
    assert adapter.close_calls == 1


def test_schema_v5_worker_emits_complete_eob_without_response_authority(
    tmp_path,
    monkeypatch,
):
    request = _request(tmp_path)
    adapter = StubCandidate(endpoint_reason="eob")
    manifest, bundle, output = _bind_run(monkeypatch, tmp_path, adapter)
    messages = encode_message(request.as_dict()) + encode_message(
        ShutdownRequest(
            "parakeet-shutdown",
            protocol_version=NATIVE_ENDPOINT_PROTOCOL_VERSION,
        ).as_dict()
    )

    result = _run_bound_worker(
        monkeypatch,
        manifest,
        bundle,
        tmp_path,
        messages,
    )
    responses = [parse_response(encode_message(event)) for event in output]
    final = next(event for event in responses if type(event) is NativeFinalEvent)

    assert result == 0
    assert final.endpoint_reason == "eob"
    assert final.native_endpoint is True
    assert final.source_samples_consumed == final.source_samples
    assert final.endpoint_latency_ms is None
    assert final.authoritative is False
    assert adapter.close_calls == 1


def test_schema_v5_worker_fails_closed_on_inconsistent_adapter_evidence(
    tmp_path,
    monkeypatch,
):
    request = _request(tmp_path)
    adapter = StubCandidate(invalid_chunks=True)
    manifest, bundle, output = _bind_run(monkeypatch, tmp_path, adapter)

    result = _run_bound_worker(
        monkeypatch,
        manifest,
        bundle,
        tmp_path,
        encode_message(request.as_dict()),
    )
    responses = [parse_response(encode_message(event)) for event in output]

    assert result == 2
    assert type(responses[-1]) is ErrorEvent
    assert responses[-1].protocol_version == NATIVE_ENDPOINT_PROTOCOL_VERSION
    assert responses[-1].fatal is True
    assert not any(type(event) is NativeFinalEvent for event in responses)
    assert adapter.close_calls == 1


def test_schema_v5_worker_rejects_legacy_request_before_candidate_input(
    tmp_path,
    monkeypatch,
):
    request = replace(_request(tmp_path), protocol_version=2)
    adapter = StubCandidate()
    manifest, bundle, output = _bind_run(monkeypatch, tmp_path, adapter)

    result = _run_bound_worker(
        monkeypatch,
        manifest,
        bundle,
        tmp_path,
        encode_message(request.as_dict()),
    )
    responses = [parse_response(encode_message(event)) for event in output]

    assert result == 2
    assert type(responses[-1]) is ErrorEvent
    assert responses[-1].protocol_version == NATIVE_ENDPOINT_PROTOCOL_VERSION
    assert responses[-1].code == "invalid_request"
    assert adapter.close_calls == 1


def test_worker_runtime_receipt_load_is_the_single_tree_scan(monkeypatch, tmp_path):
    config = _config()
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
        python=SimpleNamespace(path=Path(os.path.abspath(worker_module.sys.executable))),
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

    monkeypatch.setattr(worker_module, "load_runtime_tree_receipt", load)
    monkeypatch.setattr(
        worker_module,
        "verify_venv_runtime_location",
        lambda value, python: calls.append(("location", value, python)),
    )
    monkeypatch.setattr(
        worker_module,
        "_verify_python312_venv_layout",
        lambda candidate, value: calls.append(("layout", candidate, value)),
    )
    monkeypatch.setattr(
        worker_module,
        "verify_runtime_tree_receipt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("redundant runtime scan")
        ),
        raising=False,
    )

    assert worker_module._verify_runtime_receipt(manifest) is receipt
    assert calls == [
        (
            "load",
            receipt_artifact.path,
            receipt_artifact.sha256,
            PARAKEET_RUNTIME_TREE_LIMITS,
        ),
        ("location", receipt, manifest.python.path),
        ("layout", manifest, receipt),
    ]
