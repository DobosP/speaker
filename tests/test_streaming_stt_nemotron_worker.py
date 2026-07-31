from __future__ import annotations

from array import array
import hashlib
import io
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.streaming_stt.corpus import load_corpus, verify_corpus_snapshot
from tools.streaming_stt.manifest import (
    ManifestError,
    NEMOTRON_ADAPTER,
    NemotronConfig,
    load_worker_manifest,
)
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
from tools.streaming_stt.source_bundle import stage_worker_source_bundle
from tools.streaming_stt.supervisor import StreamingWorker
import tools.streaming_stt.worker as worker_module


_REPO_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST_ENV = "SPEAKER_NEMOTRON_WORKER_MANIFEST"
_CORPUS_ENV = "SPEAKER_NEMOTRON_STREAMING_CORPUS"


class CandidateFailure(RuntimeError):
    pass


class StubCandidate:
    def __init__(self, *, chunks_yielded: object = 3) -> None:
        self.chunks_yielded = chunks_yielded
        self.close_calls = 0
        self.ready = SimpleNamespace(
            model_load_ms=12.5,
            resources=ResourceUsage(rss_mb=42.0, threads=1, vram_mb=512.0),
        )

    def transcribe(self, _request, *, emit_partial):
        del emit_partial
        return SimpleNamespace(
            partials=(),
            final="authoritative final",
            elapsed_ms=20.0,
            finalization_ms=4.0,
            compute_ms=10.0,
            deadline_misses=0,
            max_backlog_ms=0.0,
            model_padding_samples=4_384,
            chunks_yielded=self.chunks_yielded,
            resources=ResourceUsage(rss_mb=43.0, threads=1, vram_mb=520.0),
        )

    def close(self) -> None:
        self.close_calls += 1


def _request(scratch: Path) -> TranscribeRequest:
    samples = array("f", [0.0] * 10_000)
    raw = samples.tobytes()
    pcm = scratch / "worker-case.f32le"
    pcm.write_bytes(raw)
    pcm.chmod(0o600)
    return TranscribeRequest(
        request_id="nemotron-worker-0",
        pcm=PcmInput(
            path=pcm.resolve(strict=True),
            sha256=hashlib.sha256(raw).hexdigest(),
            samples=len(samples),
        ),
        stream=StreamConfig(
            chunk_samples=5_120,
            pace="burst",
            partial_interval_ms=320,
            tail_padding_samples=0,
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
        schema_version=3,
        model_id="nemotron-worker-test",
        adapter=NEMOTRON_ADAPTER,
        adapter_config=NemotronConfig(),
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


def test_schema_v3_worker_reports_adapter_native_chunk_count_not_ceil(
    tmp_path,
    monkeypatch,
):
    scratch = tmp_path
    adapter = StubCandidate(chunks_yielded=3)
    manifest, bundle, output = _bind_run(monkeypatch, tmp_path, adapter)

    result = _run_bound_worker(manifest, bundle, scratch)
    responses = [parse_response(encode_message(event)) for event in output]

    assert result == 0
    assert isinstance(responses[0], ReadyEvent)
    final = next(event for event in responses if isinstance(event, FinalEvent))
    assert final.chunks == 3
    assert final.chunks != 2  # ceil(10_000 / 5_120) loses native window evidence.
    assert final.model_padding_samples == 4_384
    assert adapter.close_calls == 1


@pytest.mark.parametrize("chunks_yielded", [None, True, 0, -1])
def test_schema_v3_worker_rejects_missing_or_invalid_native_chunk_evidence(
    tmp_path,
    monkeypatch,
    chunks_yielded,
):
    adapter = StubCandidate(chunks_yielded=chunks_yielded)
    manifest, bundle, output = _bind_run(monkeypatch, tmp_path, adapter)

    result = _run_bound_worker(manifest, bundle, tmp_path)
    responses = [parse_response(encode_message(event)) for event in output]

    assert result == 2
    assert isinstance(responses[-1], ErrorEvent)
    assert responses[-1].fatal is True
    assert not any(isinstance(event, FinalEvent) for event in responses)
    assert adapter.close_calls == 1


def test_schema_v3_worker_checks_actual_python_before_ready(monkeypatch):
    manifest = SimpleNamespace(
        adapter=NEMOTRON_ADAPTER,
        adapter_config=NemotronConfig(),
    )
    monkeypatch.setattr(
        worker_module.platform,
        "python_version",
        lambda: "3.12.4",
    )

    with pytest.raises(ManifestError):
        worker_module._verify_nemotron_python_version(manifest)


def test_non_nemotron_worker_python_contract_is_unchanged(monkeypatch):
    manifest = SimpleNamespace(
        adapter="fake-json-v1",
        adapter_config=None,
    )
    monkeypatch.setattr(
        worker_module.platform,
        "python_version",
        lambda: "9.9.9",
    )

    worker_module._verify_nemotron_python_version(manifest)


def _native_final_evidence(
    samples: int,
    config: NemotronConfig,
) -> tuple[int, int]:
    geometry = (
        config.native_first_chunk_frames,
        config.native_hop_length_samples,
        config.native_n_fft_samples,
        config.native_first_window_samples,
        config.native_window_samples,
        config.native_stride_samples,
    )
    if (
        type(samples) is not int
        or samples <= 0
        or any(type(value) is not int or value <= 0 for value in geometry)
    ):
        raise AssertionError
    (
        first_frames,
        hop_length,
        n_fft,
        first_window,
        window,
        stride,
    ) = geometry
    final_window_end = first_window
    chunks = 1
    if final_window_end < samples:
        final_window_end = first_frames * hop_length - n_fft // 2 + window
        chunks = 2
        while final_window_end < samples:
            final_window_end += stride
            chunks += 1
    return chunks, final_window_end - samples


@pytest.mark.real_model
@pytest.mark.timeout(300)
def test_opt_in_exact_nemotron_worker_smoke(tmp_path):
    """Run the first exact case through the schema-3 Bubblewrap worker."""

    manifest_value = os.environ.get(_MANIFEST_ENV)
    corpus_value = os.environ.get(_CORPUS_ENV)
    if not manifest_value or not corpus_value:
        pytest.skip(f"set {_MANIFEST_ENV} and {_CORPUS_ENV} to explicit exact assets")
    manifest_path = Path(manifest_value)
    corpus_path = Path(corpus_value)
    if not manifest_path.is_absolute() or not corpus_path.is_absolute():
        pytest.fail("real-model asset paths must be absolute")

    manifest = load_worker_manifest(manifest_path)
    corpus = load_corpus(corpus_path)
    verify_corpus_snapshot(corpus)
    assert manifest.schema_version == 3
    assert manifest.adapter == NEMOTRON_ADAPTER
    config = manifest.adapter_config
    assert isinstance(config, NemotronConfig)

    scratch = tmp_path / "nemotron-worker-smoke"
    scratch.mkdir(mode=0o700)
    source_bundle = stage_worker_source_bundle(
        _REPO_ROOT,
        scratch / "source-bundle",
    )
    case = corpus.cases[0]
    pcm = scratch / "case-0.f32le"
    pcm.write_bytes(case.audio_bytes)
    pcm.chmod(0o600)
    request = TranscribeRequest(
        request_id="nemotron-real-smoke-0",
        pcm=PcmInput(
            path=pcm.resolve(strict=True),
            sha256=case.sha256,
            samples=case.samples,
        ),
        stream=StreamConfig(
            chunk_samples=config.native_stride_samples,
            pace="burst",
            partial_interval_ms=config.streaming_latency_ms,
            tail_padding_samples=0,
        ),
    )
    worker = StreamingWorker(manifest, scratch, source_bundle)
    try:
        try:
            ready = worker.start()
            trace = worker.transcribe(request)
        finally:
            worker.close()
    finally:
        verify_corpus_snapshot(corpus)

    expected_chunks, expected_padding = _native_final_evidence(
        case.samples,
        config,
    )
    assert ready.adapter == NEMOTRON_ADAPTER
    assert ready.model_id == manifest.model_id
    assert ready.manifest_sha256 == manifest.digest
    assert ready.source_bundle_sha256 == source_bundle.tree_sha256
    assert ready.runtime == {
        "python": config.python_version,
        "platform": "linux",
    }
    assert [event.seq for event in trace.partials] == list(range(len(trace.partials)))
    assert all(0 <= event.samples_seen <= case.samples for event in trace.partials)
    assert trace.final.seq == len(trace.partials)
    assert isinstance(trace.final.text, str)
    assert trace.final.samples_seen == case.samples
    assert trace.final.audio_seconds == pytest.approx(case.samples / 16_000)
    assert trace.final.chunks == expected_chunks
    assert trace.final.model_padding_samples == expected_padding
