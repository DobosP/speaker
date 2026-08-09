from __future__ import annotations

from array import array
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.streaming_stt.manifest import KYUTAI_ADAPTER, KyutaiConfig
from tools.streaming_stt.protocol import (
    ErrorEvent,
    FinalEvent,
    PcmInput,
    ReadyEvent,
    ResourceUsage,
    StreamConfig,
    TranscribeRequest,
    encode_message,
    parse_response,
)
import tools.streaming_stt.worker as worker_module


class CandidateFailure(RuntimeError):
    pass


class StubCandidate:
    def __init__(self, **overrides: object) -> None:
        self.overrides = overrides
        self.close_calls = 0
        self.ready = SimpleNamespace(
            model_load_ms=12.5,
            resources=ResourceUsage(rss_mb=42.0, threads=1, vram_mb=2_400.0),
            stream_geometry=SimpleNamespace(
                input_sample_rate_hz=16_000,
                mimi_sample_rate_hz=24_000,
                input_chunk_samples=1_280,
                mimi_frame_samples=1_920,
                terminal_tail_samples=16_000,
                resampling_mode="whole-buffer-noncausal",
            ),
        )

    def transcribe(self, request, *, emit_partial):
        del emit_partial
        total = request.pcm.samples + request.stream.tail_padding_samples
        chunks = (
            total + request.stream.chunk_samples - 1
        ) // request.stream.chunk_samples
        values: dict[str, object] = {
            "partials": (),
            "final": "streaming final",
            "elapsed_ms": 20.0,
            "finalization_ms": 4.0,
            "compute_ms": 10.0,
            "deadline_misses": 0,
            "max_backlog_ms": 0.0,
            "model_padding_samples": (-total) % request.stream.chunk_samples,
            "chunks_yielded": chunks,
            "resources": ResourceUsage(
                rss_mb=43.0,
                threads=1,
                vram_mb=2_500.0,
            ),
            "source_samples": request.pcm.samples,
            "source_samples_consumed": request.pcm.samples,
            "declared_tail_samples": request.stream.tail_padding_samples,
            "tail_samples_consumed": request.stream.tail_padding_samples,
            "semantic_head_steps": chunks + 1,
        }
        values.update(self.overrides)
        return SimpleNamespace(**values)

    def close(self) -> None:
        self.close_calls += 1


def _request(scratch: Path) -> TranscribeRequest:
    samples = array("f", [0.0] * 1_000)
    raw = samples.tobytes()
    pcm = scratch / "worker-case.f32le"
    pcm.write_bytes(raw)
    pcm.chmod(0o600)
    return TranscribeRequest(
        request_id="kyutai-worker-0",
        pcm=PcmInput(
            path=pcm.resolve(strict=True),
            sha256=hashlib.sha256(raw).hexdigest(),
            samples=len(samples),
        ),
        stream=StreamConfig(
            chunk_samples=1_280,
            pace="burst",
            partial_interval_ms=160,
            tail_padding_samples=16_000,
        ),
    )


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
        schema_version=9,
        model_id="kyutai-worker-test",
        adapter=KYUTAI_ADAPTER,
        adapter_config=KyutaiConfig(),
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
        worker_module, "_verify_candidate_python_version", lambda *_: None
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
        "_verify_kyutai_cuda_preflight",
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
        SimpleNamespace(
            buffer=io.BytesIO(encode_message(_request(tmp_path).as_dict()))
        ),
    )
    return manifest, source_bundle, output


def _run_bound_worker(manifest, source_bundle, scratch: Path) -> int:
    return worker_module._run(
        manifest.path,
        scratch,
        source_bundle_manifest=Path("unused-source-bundle.json"),
        source_bundle_sha256=source_bundle.tree_sha256,
    )


def test_schema_v9_worker_binds_semantic_steps_and_complete_source_tail(
    tmp_path,
    monkeypatch,
):
    adapter = StubCandidate()
    manifest, bundle, output = _bind_run(monkeypatch, tmp_path, adapter)

    result = _run_bound_worker(manifest, bundle, tmp_path)
    responses = [parse_response(encode_message(event)) for event in output]

    assert result == 0
    assert isinstance(responses[0], ReadyEvent)
    final = next(event for event in responses if isinstance(event, FinalEvent))
    assert final.samples_seen == 17_000
    assert final.chunks == 14
    assert final.model_padding_samples == 920
    assert adapter.close_calls == 1


def test_schema_v9_worker_rejects_causal_or_unbound_resampling_geometry(
    tmp_path,
    monkeypatch,
):
    adapter = StubCandidate()
    adapter.ready.stream_geometry.resampling_mode = "online-causal"
    manifest, bundle, output = _bind_run(monkeypatch, tmp_path, adapter)

    with pytest.raises(worker_module.ManifestError):
        _run_bound_worker(manifest, bundle, tmp_path)

    assert output == []
    assert adapter.close_calls == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("chunks_yielded", 13),
        ("semantic_head_steps", 13),
        ("source_samples_consumed", 999),
        ("tail_samples_consumed", 15_999),
        ("model_padding_samples", 919),
    ],
)
def test_schema_v9_worker_rejects_incomplete_or_inexact_stream_evidence(
    tmp_path,
    monkeypatch,
    field,
    value,
):
    adapter = StubCandidate(**{field: value})
    manifest, bundle, output = _bind_run(monkeypatch, tmp_path, adapter)

    result = _run_bound_worker(manifest, bundle, tmp_path)
    responses = [parse_response(encode_message(event)) for event in output]

    assert result == 2
    assert isinstance(responses[-1], ErrorEvent)
    assert responses[-1].fatal is True
    assert not any(isinstance(event, FinalEvent) for event in responses)
    assert adapter.close_calls == 1
