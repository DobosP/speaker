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
    MOONSHINE_ADAPTER,
    ManifestError,
    load_worker_manifest,
)
from tools.streaming_stt.protocol import (
    ByeEvent,
    ErrorEvent,
    FinalEvent,
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
from tools.streaming_stt.source_bundle import stage_worker_source_bundle
from tools.streaming_stt.supervisor import StreamingWorker
import tools.streaming_stt.worker as worker_module


_REPO_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST_ENV = "SPEAKER_MOONSHINE_WORKER_MANIFEST"
_CORPUS_ENV = "SPEAKER_MOONSHINE_STREAMING_CORPUS"


class CandidateFailure(RuntimeError):
    pass


class StubCandidate:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.close_calls = 0
        self.ready = SimpleNamespace(
            model_load_ms=12.5,
            resources=ResourceUsage(rss_mb=42.0, threads=1, vram_mb=None),
        )

    def transcribe(self, request, *, emit_partial):
        if self.fail:
            raise CandidateFailure
        first = SimpleNamespace(
            text="find",
            after_samples=2,
            elapsed_ms=4.0,
            decode_ms=1.0,
        )
        second = SimpleNamespace(
            text="find in my vault",
            after_samples=request.pcm.samples + request.stream.tail_padding_samples,
            elapsed_ms=8.0,
            decode_ms=1.5,
        )
        # Adapter sequence values are deliberately sparse. The worker owns the
        # wire sequence and must make it dense.
        emit_partial(17, first)
        emit_partial(91, second)
        return SimpleNamespace(
            partials=(first, second),
            final="find in my vault",
            elapsed_ms=10.0,
            finalization_ms=2.0,
            compute_ms=6.0,
            deadline_misses=0,
            max_backlog_ms=0.0,
            resources=ResourceUsage(rss_mb=43.0, threads=1, vram_mb=None),
        )

    def close(self) -> None:
        self.close_calls += 1


def _request(scratch: Path) -> TranscribeRequest:
    samples = array("f", [0.0, 0.1, -0.1, 0.0])
    raw = samples.tobytes()
    pcm = scratch / "worker-case.f32le"
    pcm.write_bytes(raw)
    pcm.chmod(0o600)
    return TranscribeRequest(
        request_id="moonshine-worker-0",
        pcm=PcmInput(
            path=pcm.resolve(strict=True),
            sha256=hashlib.sha256(raw).hexdigest(),
            samples=len(samples),
        ),
        stream=StreamConfig(
            chunk_samples=2,
            pace="burst",
            partial_interval_ms=100,
            tail_padding_samples=1,
        ),
    )


def _bind_run(
    monkeypatch,
    tmp_path: Path,
    adapter: StubCandidate,
    *,
    reject_verification: bool = False,
) -> tuple[SimpleNamespace, SimpleNamespace, list[str], list[dict[str, object]]]:
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
        schema_version=2,
        model_id="moonshine-worker-test",
        adapter=MOONSHINE_ADAPTER,
        adapter_config=SimpleNamespace(),
        worker=worker_receipt,
        python=python_receipt,
    )
    source_bundle = SimpleNamespace(
        tree_sha256="d" * 64,
        worker_relative_path="tools/streaming_stt/worker.py",
        worker_path=worker_path,
        file_by_path={"tools/streaming_stt/worker.py": worker_receipt},
    )
    events: list[str] = []
    output: list[dict[str, object]] = []

    def load_bundle(*_args, **_kwargs):
        events.append("source")
        return source_bundle

    def load_manifest(*_args, **_kwargs):
        events.append("manifest")
        return manifest

    def hash_python(*_args, **_kwargs):
        events.append("python")
        return SimpleNamespace(sha256=python_receipt.sha256)

    def verify_state(*_args, **_kwargs):
        events.append("verify")
        if reject_verification:
            raise ManifestError

    def activate_runtime(*_args, **_kwargs):
        events.append("activate")

    def create_adapter(*_args, **_kwargs):
        events.append("candidate")
        return adapter, CandidateFailure

    def close_adapter(selected):
        events.append("close")
        selected.close()

    monkeypatch.setattr(
        worker_module, "_SOURCE_BUNDLE_SHA256", source_bundle.tree_sha256
    )
    monkeypatch.setattr(worker_module, "load_source_bundle", load_bundle)
    monkeypatch.setattr(worker_module, "load_worker_manifest", load_manifest)
    monkeypatch.setattr(worker_module, "hash_regular_bounded", hash_python)
    monkeypatch.setattr(worker_module, "_verify_worker_state", verify_state)
    monkeypatch.setattr(worker_module, "_activate_candidate_runtime", activate_runtime)
    monkeypatch.setattr(worker_module, "_create_adapter", create_adapter)
    monkeypatch.setattr(worker_module, "_close_adapter", close_adapter)
    monkeypatch.setattr(worker_module, "_write", output.append)
    return manifest, source_bundle, events, output


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


def test_candidate_selection_never_runs_before_closed_state_verification(
    tmp_path,
    monkeypatch,
):
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    adapter = StubCandidate()
    manifest, bundle, events, output = _bind_run(
        monkeypatch,
        tmp_path,
        adapter,
        reject_verification=True,
    )

    with pytest.raises(ManifestError):
        _run_bound_worker(monkeypatch, manifest, bundle, scratch, b"")

    assert events == ["source", "manifest", "python", "verify"]
    assert "candidate" not in events
    assert output == []
    assert adapter.close_calls == 0


def test_schema_v2_worker_owns_dense_partial_final_protocol_and_shutdown_cleanup(
    tmp_path,
    monkeypatch,
):
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    request = _request(scratch)
    adapter = StubCandidate()
    manifest, bundle, events, output = _bind_run(
        monkeypatch,
        tmp_path,
        adapter,
    )
    messages = encode_message(request.as_dict()) + encode_message(
        ShutdownRequest("moonshine-shutdown").as_dict()
    )

    result = _run_bound_worker(
        monkeypatch,
        manifest,
        bundle,
        scratch,
        messages,
    )
    responses = [parse_response(encode_message(event)) for event in output]

    assert result == 0
    assert isinstance(responses[0], ReadyEvent)
    partials = [event for event in responses if isinstance(event, PartialEvent)]
    assert [event.seq for event in partials] == [0, 1]
    assert [event.text for event in partials] == ["find", "find in my vault"]
    final = next(event for event in responses if isinstance(event, FinalEvent))
    assert final.seq == len(partials) == 2
    assert final.samples_seen == 5
    assert final.chunks == 3
    assert isinstance(responses[-2], ShutdownEvent)
    assert isinstance(responses[-1], ByeEvent)
    assert events.index("verify") < events.index("activate") < events.index("candidate")
    assert events[-2:] == ["close", "verify"]
    assert adapter.close_calls == 1


def test_schema_v2_candidate_failure_is_fatal_and_closes_candidate(
    tmp_path,
    monkeypatch,
):
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    request = _request(scratch)
    adapter = StubCandidate(fail=True)
    manifest, bundle, events, output = _bind_run(
        monkeypatch,
        tmp_path,
        adapter,
    )

    result = _run_bound_worker(
        monkeypatch,
        manifest,
        bundle,
        scratch,
        encode_message(request.as_dict()),
    )
    responses = [parse_response(encode_message(event)) for event in output]

    assert result == 2
    assert isinstance(responses[0], ReadyEvent)
    assert len(responses) == 2
    error = responses[1]
    assert isinstance(error, ErrorEvent)
    assert error.request_id == request.request_id
    assert error.code == "candidate_failure"
    assert error.fatal is True
    assert not any(isinstance(event, FinalEvent) for event in responses)
    assert events[-1] == "close"
    assert adapter.close_calls == 1


@pytest.mark.real_model
def test_opt_in_exact_moonshine_worker_smoke(tmp_path):
    """Run one exact external case; never install, download, or open audio."""

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
    assert manifest.schema_version == 2
    assert manifest.adapter == MOONSHINE_ADAPTER

    scratch = tmp_path / "moonshine-worker-smoke"
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
        request_id="moonshine-real-smoke-0",
        pcm=PcmInput(
            path=pcm.resolve(strict=True),
            sha256=case.sha256,
            samples=case.samples,
        ),
        stream=StreamConfig(
            chunk_samples=1600,
            pace="burst",
            partial_interval_ms=500,
            tail_padding_samples=0,
        ),
    )
    worker = StreamingWorker(manifest, scratch, source_bundle)
    try:
        ready = worker.start()
        trace = worker.transcribe(request)
    finally:
        worker.close()
    verify_corpus_snapshot(corpus)

    assert ready.adapter == MOONSHINE_ADAPTER
    assert ready.manifest_sha256 == manifest.digest
    assert [event.seq for event in trace.partials] == list(range(len(trace.partials)))
    assert trace.final.seq == len(trace.partials)
    assert trace.final.samples_seen == case.samples
