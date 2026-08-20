from __future__ import annotations

from array import array
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.streaming_stt_helpers import stage_test_source_bundle, write_fixture
from tools.provision_mobile_whisper import load_mobile_whisper_provision
from tools.streaming_stt.bounded_io import FileDigest
import tools.streaming_stt.manifest as manifest_module
from tools.streaming_stt.manifest import (
    MOBILE_WHISPER_ADAPTER,
    MOBILE_WHISPER_ARTIFACT_SET_SHA256,
    MOBILE_WHISPER_ARTIFACT_SPECS,
    MOBILE_WHISPER_SOURCE_LOCK_RECIPE_SHA256,
    MOBILE_WHISPER_SOURCE_LOCK_SHA256,
    MOBILE_WHISPER_SOURCE_LOCK_SIZE_BYTES,
    MOBILE_WHISPER_TOTAL_SIZE_BYTES,
    MOBILE_ZIPFORMER_ADAPTER,
    SHERPA_ZIPFORMER_ADAPTER,
    BoundArtifact,
    BoundFile,
    ManifestError,
    MobileWhisperConfig,
    load_worker_manifest,
)
from tools.streaming_stt.protocol import (
    MAX_CORPUS_BYTES,
    PROTOCOL_VERSION,
    ErrorEvent,
    FinalEvent,
    PartialEvent,
    PcmInput,
    ProtocolError,
    ReadyEvent,
    ResourceUsage,
    StreamConfig,
    TranscribeRequest,
    encode_message,
    parse_response,
)
from tools.streaming_stt.supervisor import StreamingWorker, WorkerError
import tools.streaming_stt.supervisor as supervisor_module
import tools.streaming_stt.worker as worker_module


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float = 0.001) -> None:
        self.value += seconds


class _Numpy:
    __version__ = "2.4.6"

    def __init__(self) -> None:
        self.calls: list[tuple[list[float], str]] = []

    def asarray(self, values, *, dtype):
        copied = list(values)
        self.calls.append((copied, dtype))
        return copied


class _OfflineStream:
    def __init__(self, recognizer: _OfflineRecognizer) -> None:
        self.recognizer = recognizer
        self.accepted: list[float] | None = None

    def accept_waveform(self, sample_rate: int, samples) -> None:
        assert sample_rate == 16_000
        self.recognizer.operations.append("accept_waveform")
        self.accepted = list(samples)
        self.recognizer.clock.advance()

    @property
    def result(self) -> object:
        self.recognizer.operations.append("result")
        self.recognizer.clock.advance()
        return SimpleNamespace(text=self.recognizer.text)


class _OfflineRecognizer:
    def __init__(self, clock: _Clock, text: str) -> None:
        self.clock = clock
        self.text = text
        self.operations: list[str] = []
        self.streams: list[_OfflineStream] = []

    def create_stream(self) -> _OfflineStream:
        self.operations.append("create_stream")
        self.clock.advance()
        stream = _OfflineStream(self)
        self.streams.append(stream)
        return stream

    def decode_stream(self, stream: _OfflineStream) -> None:
        assert stream.accepted is not None
        self.operations.append("decode_stream")
        self.clock.advance()


class _OfflineFactory:
    def __init__(self, recognizer: _OfflineRecognizer) -> None:
        self.recognizer = recognizer
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> _OfflineRecognizer:
        self.calls.append(kwargs)
        self.recognizer.clock.advance(0.01)
        return self.recognizer


def _api(factory: _OfflineFactory) -> object:
    return SimpleNamespace(
        __version__="1.13.3",
        OfflineRecognizer=SimpleNamespace(from_whisper=factory),
    )


def _model_root(tmp_path: Path) -> Path:
    root = tmp_path / "model"
    root.mkdir()
    for _name, basename, _precision, _digest, _size in (
        MOBILE_WHISPER_ARTIFACT_SPECS
    ):
        (root / basename).write_bytes(b"mobile-whisper-test-placeholder")
    return root


def _request(
    tmp_path: Path,
    *,
    request_id: str = "mobile-whisper-case",
    samples: int = 1_600,
    tail_padding_samples: int = 0,
    chunk_samples: int = 1_600,
    pace: str = "burst",
) -> TranscribeRequest:
    raw = array("f", [0.125] * samples).tobytes()
    path = tmp_path / "mobile-whisper-case.f32le"
    path.write_bytes(raw)
    path.chmod(0o600)
    return TranscribeRequest(
        request_id=request_id,
        pcm=PcmInput(
            path=path.resolve(strict=True),
            sha256=hashlib.sha256(raw).hexdigest(),
            samples=samples,
        ),
        stream=StreamConfig(
            chunk_samples=chunk_samples,
            pace=pace,
            partial_interval_ms=100,
            tail_padding_samples=tail_padding_samples,
        ),
        protocol_version=PROTOCOL_VERSION,
    )


def _adapter(tmp_path: Path, *, text: str = ""):
    from tools.streaming_stt.adapters.zipformer import MobileWhisperOfflineAdapter

    clock = _Clock()
    recognizer = _OfflineRecognizer(clock, text)
    factory = _OfflineFactory(recognizer)
    numpy = _Numpy()
    adapter = MobileWhisperOfflineAdapter(
        _model_root(tmp_path),
        config=MobileWhisperConfig(),
        api=_api(factory),
        numpy_api=numpy,
        clock=clock,
    )
    return adapter, recognizer, factory, numpy


def test_adapter_runs_one_complete_epoch_decode_and_preserves_empty_final(
    tmp_path: Path,
) -> None:
    adapter, recognizer, factory, numpy = _adapter(tmp_path, text="   ")
    emitted: list[object] = []
    request = _request(tmp_path)

    case = adapter.transcribe(
        request,
        emit_partial=lambda _seq, partial: emitted.append(partial),
    )

    assert factory.calls == [
        {
            "encoder": str(adapter.model_root / "base.en-encoder.int8.onnx"),
            "decoder": str(adapter.model_root / "base.en-decoder.int8.onnx"),
            "tokens": str(adapter.model_root / "base.en-tokens.txt"),
            "language": "en",
            "task": "transcribe",
            "num_threads": 1,
            "decoding_method": "greedy_search",
            "debug": True,
            "provider": "cpu",
            "tail_paddings": -1,
            "enable_token_timestamps": False,
            "enable_segment_timestamps": False,
            "rule_fsts": "",
            "rule_fars": "",
            "hr_dict_dir": "",
            "hr_rule_fsts": "",
            "hr_lexicon": "",
        }
    ]
    assert recognizer.operations == [
        "create_stream",
        "accept_waveform",
        "decode_stream",
        "result",
    ]
    assert len(recognizer.streams) == 1
    assert recognizer.streams[0].accepted == [0.125] * request.pcm.samples
    assert numpy.calls == [([0.125] * request.pcm.samples, "float32")]
    assert emitted == []
    assert case.partials == ()
    assert case.final == ""
    # This is only adapter-appended PCM alignment padding, not sherpa's opaque
    # internal tail_paddings=-1 behavior.
    assert case.model_padding_samples == 0


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"tail_padding_samples": 1}, True),
        ({"chunk_samples": 800}, True),
        ({"pace": "realtime"}, True),
    ],
)
def test_adapter_rejects_non_epoch_geometry(
    tmp_path: Path,
    overrides: dict[str, object],
    error: bool,
) -> None:
    from tools.streaming_stt.adapters.zipformer import MobileWhisperAdapterError

    adapter, _recognizer, _factory, _numpy = _adapter(tmp_path)
    request = _request(tmp_path, **overrides)

    assert error is True
    with pytest.raises(MobileWhisperAdapterError):
        adapter.transcribe(request, emit_partial=lambda _seq, _partial: None)


def _source_payload(lock: dict[str, object]) -> dict[str, object]:
    return {
        "artifact_license_status": "unverified",
        "conversion_license_artifact_present": False,
        "conversion_origin_revision_status": "unverified",
        "exact_conversion_recipe_status": "unavailable",
        "repo_id": "csukuangfj/sherpa-onnx-whisper-base.en",
        "revision": "59eea950fc76df2453efb57e6c0fd334548e8ffe",
        "upstream_base_en_checkpoint_sha256": (
            "25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead"
        ),
        "upstream_code_license": "MIT",
        "upstream_evidence_revision": (
            "5f86d1d86363843179951550570367b37c5d6f78"
        ),
        "upstream_evidence_scope": "license-and-base.en-checkpoint-map-only",
        "upstream_license_sha256": (
            "b5d65a59060e68c4ff940e1eddfa6f94b2d68fdf58ed7f4dd57721c997e35e9d"
        ),
        "upstream_license_size_bytes": 1_063,
        "upstream_license_url": (
            "https://raw.githubusercontent.com/openai/whisper/"
            "5f86d1d86363843179951550570367b37c5d6f78/LICENSE"
        ),
        "upstream_model_map_sha256": (
            "79ecbed714783b6230931a3dba6d5da6d386e2d44f9fc27256e4fc72751e1f62"
        ),
        "upstream_model_map_size_bytes": 7_432,
        "upstream_model_map_url": (
            "https://raw.githubusercontent.com/openai/whisper/"
            "5f86d1d86363843179951550570367b37c5d6f78/whisper/__init__.py"
        ),
        "upstream_project": "openai/whisper",
        "lock": lock,
        "lock_recipe_sha256": MOBILE_WHISPER_SOURCE_LOCK_RECIPE_SHA256,
    }


def _schema_v11_payload(tmp_path: Path) -> dict[str, object]:
    python = {
        "path": str((tmp_path / "venv" / "bin" / "python").resolve()),
        "sha256": "a" * 64,
        "size_bytes": 123,
    }
    worker = {
        "path": str((tmp_path / "worker.py").resolve()),
        "sha256": "b" * 64,
        "size_bytes": 456,
    }
    lock = {
        "path": str((tmp_path / "source-lock.json").resolve()),
        "sha256": MOBILE_WHISPER_SOURCE_LOCK_SHA256,
        "size_bytes": MOBILE_WHISPER_SOURCE_LOCK_SIZE_BYTES,
    }
    model_root = tmp_path / "model"
    return {
        "schema_version": 11,
        "kind": "mobile-whisper-provision-v1",
        "model_id": "sherpa-onnx-whisper-base.en-mobile-int8-v1",
        "adapter": MOBILE_WHISPER_ADAPTER,
        "python": python,
        "worker": worker,
        "artifacts": [
            {
                "name": name,
                "path": str((model_root / basename).resolve()),
                "sha256": digest,
                "size_bytes": size_bytes,
            }
            for name, basename, _precision, digest, size_bytes in (
                MOBILE_WHISPER_ARTIFACT_SPECS
            )
        ],
        "limits": {"startup_timeout_sec": 120.0, "case_timeout_sec": 300.0},
        "mobile_config": MobileWhisperConfig().as_dict(),
        "source": _source_payload(lock),
        "runtime": {
            "python": dict(python),
            "worker": dict(worker),
            "distributions": {
                "sherpa-onnx": "1.13.3",
                "sherpa-onnx-core": "1.13.3",
                "numpy": "2.4.6",
            },
            "metadata_only_verified": True,
            "packages_imported": False,
            "model_loaded": False,
        },
        "artifact_set_sha256": MOBILE_WHISPER_ARTIFACT_SET_SHA256,
        "total_size_bytes": MOBILE_WHISPER_TOTAL_SIZE_BYTES,
        "evidence_scope": {
            "downloaded": False,
            "packages_imported": False,
            "model_loaded": False,
            "model_executed": False,
            "gpu_used": False,
            "audio_device_opened": False,
            "evaluation_result_authority": False,
            "mobile_device_evidence": False,
        },
    }


def _load_mocked_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
):
    manifest_path = tmp_path / "worker-manifest.json"
    manifest_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    source = payload["source"]
    assert isinstance(source, dict)
    artifacts = payload["artifacts"]
    assert isinstance(artifacts, list)
    rows = [payload["python"], payload["worker"], source["lock"], *artifacts]
    receipts = {
        Path(item["path"]): (item["sha256"], item["size_bytes"])
        for item in rows
        if isinstance(item, dict)
    }

    def fake_hash(
        path: Path,
        *,
        maximum_bytes: int,
        expected_bytes: int,
        allow_final_symlink: bool = False,
    ) -> FileDigest:
        del maximum_bytes, allow_final_symlink
        digest, size = receipts[Path(path)]
        assert expected_bytes == size
        return FileDigest(Path(path), digest, size)

    monkeypatch.setattr(manifest_module, "hash_regular_bounded", fake_hash)
    return load_worker_manifest(manifest_path)


def test_schema_v11_manifest_loads_exact_final_only_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _load_mocked_manifest(
        tmp_path,
        monkeypatch,
        _schema_v11_payload(tmp_path),
    )

    assert manifest.schema_version == 11
    assert manifest.adapter == MOBILE_WHISPER_ADAPTER
    assert manifest.adapter_config == MobileWhisperConfig()
    assert manifest.control_files == (
        BoundFile(
            path=(tmp_path / "source-lock.json").resolve(),
            sha256=MOBILE_WHISPER_SOURCE_LOCK_SHA256,
            size_bytes=MOBILE_WHISPER_SOURCE_LOCK_SIZE_BYTES,
        ),
    )


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("mobile_config", "max_active_paths", 4),
        ("source", "conversion_license_artifact_present", True),
        ("source", "upstream_evidence_revision", "0" * 40),
    ],
)
def test_schema_v11_manifest_rejects_config_or_provenance_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    field: str,
    replacement: object,
) -> None:
    payload = _schema_v11_payload(tmp_path)
    mapping = payload[section]
    assert isinstance(mapping, dict)
    mapping[field] = replacement

    with pytest.raises(ManifestError):
        _load_mocked_manifest(tmp_path, monkeypatch, payload)


def test_schema_v11_counts_every_request_and_other_ledgers_still_deduplicate() -> None:
    mobile = SimpleNamespace(schema_version=11, adapter=MOBILE_WHISPER_ADAPTER)
    schema10 = SimpleNamespace(schema_version=10, adapter=MOBILE_ZIPFORMER_ADAPTER)
    generic = SimpleNamespace(schema_version=2, adapter="another-adapter")
    ceiling = worker_module._worker_corpus_limit_bytes(mobile)

    assert ceiling == 62 * 1024 * 1024
    assert worker_module._worker_corpus_limit_bytes(schema10) == 38 * 1024 * 1024
    assert worker_module._worker_corpus_limit_bytes(generic) == MAX_CORPUS_BYTES
    assert worker_module._account_request_pcm(
        0,
        size_bytes=ceiling,
        maximum_bytes=ceiling,
    ) == ceiling
    with pytest.raises(ProtocolError):
        worker_module._account_request_pcm(
            ceiling,
            size_bytes=1,
            maximum_bytes=ceiling,
        )
    first = worker_module._account_request_pcm(
        0,
        size_bytes=4,
        maximum_bytes=ceiling,
    )
    assert worker_module._account_request_pcm(
        first,
        size_bytes=4,
        maximum_bytes=ceiling,
    ) == 8
    consumed_inputs: set[str] = set()
    digest = "a" * 64
    generic_first = worker_module._account_unique_pcm(
        consumed_inputs,
        0,
        pcm_sha256=digest,
        size_bytes=4,
        maximum_bytes=MAX_CORPUS_BYTES,
    )
    assert worker_module._account_unique_pcm(
        consumed_inputs,
        generic_first,
        pcm_sha256=digest,
        size_bytes=4,
        maximum_bytes=MAX_CORPUS_BYTES,
    ) == generic_first


class _WorkerCandidate:
    ready = SimpleNamespace(
        model_load_ms=1.0,
        resources=ResourceUsage(rss_mb=1.0, threads=1, vram_mb=None),
    )

    def __init__(self) -> None:
        self.transcribe_calls = 0
        self.close_calls = 0

    def transcribe(self, _request: object, *, emit_partial: object):
        from tools.streaming_stt.adapters.zipformer import MobileWhisperCase

        assert callable(emit_partial)
        self.transcribe_calls += 1
        return MobileWhisperCase(
            final="",
            elapsed_ms=1.0,
            finalization_ms=0.5,
            compute_ms=0.5,
            resources=ResourceUsage(rss_mb=1.0, threads=1, vram_mb=None),
            model_padding_samples=0,
        )

    def close(self) -> None:
        self.close_calls += 1


def _run_schema_v11_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requests: list[TranscribeRequest],
    *,
    ceiling: int,
):
    worker_path = Path(worker_module.__file__).resolve(strict=True)
    worker_receipt = SimpleNamespace(sha256="c" * 64, size_bytes=123)
    python_receipt = SimpleNamespace(sha256="d" * 64, size_bytes=456)
    manifest = SimpleNamespace(
        path=tmp_path / "worker-manifest.json",
        digest="e" * 64,
        schema_version=11,
        model_id="mobile-whisper-worker-test",
        adapter=MOBILE_WHISPER_ADAPTER,
        adapter_config=MobileWhisperConfig(),
        worker=worker_receipt,
        python=python_receipt,
    )
    source_bundle = SimpleNamespace(
        tree_sha256="f" * 64,
        worker_relative_path="tools/streaming_stt/worker.py",
        worker_path=worker_path,
        file_by_path={"tools/streaming_stt/worker.py": worker_receipt},
    )
    candidate = _WorkerCandidate()
    output: list[dict[str, object]] = []
    monkeypatch.setattr(worker_module, "_SOURCE_BUNDLE_SHA256", "f" * 64)
    monkeypatch.setattr(worker_module, "_MOBILE_WHISPER_MAX_CORPUS_BYTES", ceiling)
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
        lambda *_args, **_kwargs: (candidate, RuntimeError),
    )
    monkeypatch.setattr(worker_module, "_write", output.append)
    wire = b"".join(encode_message(request.as_dict()) for request in requests)
    monkeypatch.setattr(
        worker_module.sys,
        "stdin",
        SimpleNamespace(buffer=io.BytesIO(wire)),
    )

    result = worker_module._run(
        manifest.path,
        tmp_path,
        source_bundle_manifest=Path("unused-source-bundle.json"),
        source_bundle_sha256=source_bundle.tree_sha256,
    )
    return result, candidate, [
        parse_response(encode_message(value)) for value in output
    ]


def test_schema_v11_worker_emits_one_empty_final_then_counts_repeat_before_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _request(tmp_path, request_id="epoch-1", samples=1)
    second = replace(first, request_id="epoch-2")

    result, candidate, responses = _run_schema_v11_worker(
        tmp_path,
        monkeypatch,
        [first, second],
        ceiling=first.pcm.size_bytes,
    )

    assert result == 2
    assert candidate.transcribe_calls == 1
    assert candidate.close_calls == 1
    assert isinstance(responses[0], ReadyEvent)
    finals = [event for event in responses if type(event) is FinalEvent]
    assert len(finals) == 1
    assert finals[0].request_id == "epoch-1"
    assert finals[0].seq == 0
    assert finals[0].text == ""
    assert finals[0].samples_seen == 1
    assert finals[0].model_padding_samples == 0
    assert isinstance(responses[-1], ErrorEvent)
    assert responses[-1].request_id == "epoch-2"
    assert responses[-1].fatal is True


def _controlled_supervisor(
    tmp_path: Path,
    *,
    schema_version: int,
    adapter: str,
    config: object,
) -> StreamingWorker:
    controlled = StreamingWorker(
        SimpleNamespace(
            schema_version=schema_version,
            adapter=adapter,
            adapter_config=config,
            limits=SimpleNamespace(case_timeout_sec=2.0),
        ),
        tmp_path,
        SimpleNamespace(),
        monotonic=lambda: 0.0,
    )
    controlled.ready = SimpleNamespace()
    return controlled


def test_schema_v11_supervisor_accepts_one_empty_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    controlled = _controlled_supervisor(
        tmp_path,
        schema_version=11,
        adapter=MOBILE_WHISPER_ADAPTER,
        config=MobileWhisperConfig(),
    )
    event = FinalEvent(
        request_id=request.request_id,
        seq=0,
        text="",
        samples_seen=request.pcm.samples,
        elapsed_ms=1.0,
        finalization_ms=0.5,
        compute_ms=0.5,
        audio_seconds=request.pcm.samples / 16_000,
        chunks=1,
        deadline_misses=0,
        max_backlog_ms=0.0,
        resources=ResourceUsage(rss_mb=1.0, threads=1, vram_mb=None),
        model_padding_samples=0,
    )
    monkeypatch.setattr(controlled, "_verify_pcm", lambda _request: None)
    monkeypatch.setattr(controlled, "_send", lambda _message: None)
    monkeypatch.setattr(controlled, "_next_event", lambda _timeout: event)

    trace = controlled.transcribe(request)

    assert trace.partials == ()
    assert trace.final == event


@pytest.mark.parametrize(
    ("schema_version", "adapter", "config"),
    [
        (11, SHERPA_ZIPFORMER_ADAPTER, SimpleNamespace()),
        (10, MOBILE_WHISPER_ADAPTER, MobileWhisperConfig()),
    ],
)
def test_supervisor_rejects_schema11_adapter_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schema_version: int,
    adapter: str,
    config: object,
) -> None:
    controlled = _controlled_supervisor(
        tmp_path,
        schema_version=schema_version,
        adapter=adapter,
        config=config,
    )
    request = _request(tmp_path)
    monkeypatch.setattr(controlled, "_verify_pcm", lambda _request: None)

    with pytest.raises(WorkerError, match="worker_protocol"):
        controlled.transcribe(request)


@pytest.mark.parametrize("event_kind", ["partial", "padding"])
def test_schema_v11_supervisor_rejects_partial_or_nonzero_adapter_pcm_padding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    event_kind: str,
) -> None:
    request = _request(tmp_path)
    controlled = _controlled_supervisor(
        tmp_path,
        schema_version=11,
        adapter=MOBILE_WHISPER_ADAPTER,
        config=MobileWhisperConfig(),
    )
    if event_kind == "partial":
        event: object = PartialEvent(
            request_id=request.request_id,
            seq=0,
            text="forbidden",
            samples_seen=request.pcm.samples,
            elapsed_ms=0.5,
            decode_ms=0.5,
        )
    else:
        event = FinalEvent(
            request_id=request.request_id,
            seq=0,
            text="",
            samples_seen=request.pcm.samples,
            elapsed_ms=1.0,
            finalization_ms=0.5,
            compute_ms=0.5,
            audio_seconds=request.pcm.samples / 16_000,
            chunks=1,
            deadline_misses=0,
            max_backlog_ms=0.0,
            resources=ResourceUsage(rss_mb=1.0, threads=1, vram_mb=None),
            model_padding_samples=1,
        )
    monkeypatch.setattr(controlled, "_verify_pcm", lambda _request: None)
    monkeypatch.setattr(controlled, "_send", lambda _message: None)
    monkeypatch.setattr(controlled, "_next_event", lambda _timeout: event)

    with pytest.raises(WorkerError, match="worker_protocol"):
        controlled.transcribe(request)


def test_schema_v6_final_only_supervisor_regression_still_accepts_seq_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.streaming_stt.manifest import FASTER_WHISPER_ENDPOINT_ADAPTER

    request = _request(tmp_path, chunk_samples=800)
    controlled = _controlled_supervisor(
        tmp_path,
        schema_version=6,
        adapter=FASTER_WHISPER_ENDPOINT_ADAPTER,
        config=SimpleNamespace(),
    )
    event = FinalEvent(
        request_id=request.request_id,
        seq=0,
        text="legacy final-only",
        samples_seen=request.pcm.samples,
        elapsed_ms=1.0,
        finalization_ms=0.5,
        compute_ms=0.5,
        audio_seconds=request.pcm.samples / 16_000,
        chunks=2,
        deadline_misses=0,
        max_backlog_ms=0.0,
        resources=ResourceUsage(rss_mb=1.0, threads=1, vram_mb=None),
        model_padding_samples=0,
    )
    monkeypatch.setattr(controlled, "_verify_pcm", lambda _request: None)
    monkeypatch.setattr(controlled, "_send", lambda _message: None)
    monkeypatch.setattr(controlled, "_next_event", lambda _timeout: event)

    trace = controlled.transcribe(request)

    assert trace.final == event


def _set_schema11_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in worker_module._MOBILE_WHISPER_REQUIRED_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)


def test_schema11_runtime_activation_skips_pth_and_sitecustomize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    venv_root = tmp_path / "venv"
    python = venv_root / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python")
    runtime = venv_root / "lib" / "python3.12" / "site-packages"
    runtime.mkdir(parents=True)
    pth_marker = tmp_path / "pth-executed"
    site_marker = tmp_path / "sitecustomize-executed"
    (runtime / "adversarial.pth").write_text(
        f"import pathlib; pathlib.Path({str(pth_marker)!r}).write_text('bad')\n",
        encoding="utf-8",
    )
    (runtime / "sitecustomize.py").write_text(
        "import pathlib\n"
        f"pathlib.Path({str(site_marker)!r}).write_text('bad')\n",
        encoding="utf-8",
    )
    fake_sys = SimpleNamespace(
        flags=SimpleNamespace(
            no_site=1,
            no_user_site=1,
            ignore_environment=1,
            isolated=1,
            dont_write_bytecode=1,
        ),
        modules={},
        path=["verified-source-bundle"],
        version_info=SimpleNamespace(major=3, minor=12),
    )
    monkeypatch.setattr(worker_module, "sys", fake_sys)
    _set_schema11_environment(monkeypatch)
    manifest = SimpleNamespace(
        adapter=MOBILE_WHISPER_ADAPTER,
        schema_version=11,
        adapter_config=MobileWhisperConfig(),
        python=SimpleNamespace(path=python),
    )

    worker_module._activate_candidate_runtime(manifest, None)

    assert fake_sys.path == ["verified-source-bundle", str(runtime)]
    assert not pth_marker.exists()
    assert not site_marker.exists()


@pytest.mark.parametrize(
    "module_name",
    ["sherpa_onnx_core", "_sherpa_onnx", "sherpa_onnx.lib._sherpa_onnx"],
)
def test_schema11_runtime_activation_rejects_preloaded_native_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
) -> None:
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python")
    (tmp_path / "venv" / "lib" / "python3.12" / "site-packages").mkdir(
        parents=True
    )
    fake_sys = SimpleNamespace(
        flags=SimpleNamespace(
            no_site=1,
            no_user_site=1,
            ignore_environment=1,
            isolated=1,
            dont_write_bytecode=1,
        ),
        modules={module_name: object()},
        path=["verified-source-bundle"],
        version_info=SimpleNamespace(major=3, minor=12),
    )
    monkeypatch.setattr(worker_module, "sys", fake_sys)
    _set_schema11_environment(monkeypatch)
    manifest = SimpleNamespace(
        adapter=MOBILE_WHISPER_ADAPTER,
        schema_version=11,
        adapter_config=MobileWhisperConfig(),
        python=SimpleNamespace(path=python),
    )

    with pytest.raises(ManifestError):
        worker_module._activate_candidate_runtime(manifest, None)

    assert fake_sys.path == ["verified-source-bundle"]


@pytest.mark.parametrize(
    ("name", "replacement"),
    [
        ("OMP_NUM_THREADS", "2"),
        ("OPENBLAS_NUM_THREADS", "2"),
        ("MKL_NUM_THREADS", "2"),
        ("NUMEXPR_NUM_THREADS", "2"),
        ("HF_HUB_OFFLINE", "0"),
        ("TRANSFORMERS_OFFLINE", "0"),
        ("CUDA_VISIBLE_DEVICES", "0"),
        ("CUDA_DEVICE_ORDER", "PCI_BUS_ID"),
    ],
)
def test_schema11_runtime_activation_rejects_environment_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    replacement: str,
) -> None:
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python")
    (tmp_path / "venv" / "lib" / "python3.12" / "site-packages").mkdir(
        parents=True
    )
    fake_sys = SimpleNamespace(
        flags=SimpleNamespace(
            no_site=1,
            no_user_site=1,
            ignore_environment=1,
            isolated=1,
            dont_write_bytecode=1,
        ),
        modules={},
        path=["verified-source-bundle"],
        version_info=SimpleNamespace(major=3, minor=12),
    )
    monkeypatch.setattr(worker_module, "sys", fake_sys)
    _set_schema11_environment(monkeypatch)
    monkeypatch.setenv(name, replacement)
    manifest = SimpleNamespace(
        adapter=MOBILE_WHISPER_ADAPTER,
        schema_version=11,
        adapter_config=MobileWhisperConfig(),
        python=SimpleNamespace(path=python),
    )

    with pytest.raises(ManifestError):
        worker_module._activate_candidate_runtime(manifest, None)

    assert fake_sys.path == ["verified-source-bundle"]


def _bwrap_inputs(tmp_path: Path, adapter: str):
    authority_root = tmp_path / f"{adapter}-authority"
    model_root = authority_root / "model"
    source_root = tmp_path / f"{adapter}-source"
    provision_root = authority_root / "provision"
    venv_root = authority_root / "venv"
    manifest = SimpleNamespace(
        adapter=adapter,
        schema_version=(
            11
            if adapter == MOBILE_WHISPER_ADAPTER
            else 10
            if adapter == MOBILE_ZIPFORMER_ADAPTER
            else 4
        ),
        adapter_config=(
            MobileWhisperConfig()
            if adapter == MOBILE_WHISPER_ADAPTER
            else SimpleNamespace()
        ),
        artifacts=(
            SimpleNamespace(name="model-one", path=model_root / "model.onnx"),
        ),
        python=SimpleNamespace(path=venv_root / "bin" / "python"),
        path=provision_root / "worker-manifest.json",
        worker=SimpleNamespace(
            path=supervisor_module._CANONICAL_REPO_ROOT
            / "tools"
            / "streaming_stt"
            / "worker.py"
        ),
        control_files=(
            SimpleNamespace(path=provision_root / "source-lock.json"),
        ),
    )
    source_bundle = SimpleNamespace(
        root=source_root,
        worker_path=source_root / "tools/streaming_stt/worker.py",
        manifest_path=source_root / "source-bundle.json",
        tree_sha256="a" * 64,
    )
    return manifest, source_bundle


def _scratch_fence_inputs(tmp_path: Path):
    authority = tmp_path / "authority"
    runtime = authority / "runtime"
    provision = authority / "provision"
    model = authority / "model"
    control = tmp_path / "external-control" / "source-lock.json"
    manifest = SimpleNamespace(
        adapter=MOBILE_WHISPER_ADAPTER,
        schema_version=11,
        adapter_config=MobileWhisperConfig(),
        python=SimpleNamespace(path=runtime / "bin" / "python"),
        path=provision / "worker-manifest.json",
        worker=SimpleNamespace(
            path=supervisor_module._CANONICAL_REPO_ROOT
            / "tools"
            / "streaming_stt"
            / "worker.py"
        ),
        artifacts=(
            SimpleNamespace(name="model-encoder", path=model / "encoder.onnx"),
        ),
        control_files=(SimpleNamespace(path=control),),
    )
    source_bundle = SimpleNamespace(
        root=tmp_path / "staged-source-bundle",
        worker_path=tmp_path / "staged-source-bundle" / "worker.py",
        manifest_path=tmp_path / "staged-source-bundle" / "source-bundle.json",
        tree_sha256="a" * 64,
    )
    return manifest, source_bundle, {
        "runtime": runtime / "scratch",
        "provision": provision / "scratch",
        "model": model / "scratch",
        "control": control.parent,
        "repo": supervisor_module._CANONICAL_REPO_ROOT,
    }


@pytest.mark.parametrize("authority", ["runtime", "provision", "model", "control"])
def test_schema11_start_rejects_scratch_overlapping_read_only_authority(
    tmp_path: Path,
    authority: str,
) -> None:
    manifest, source_bundle, scratches = _scratch_fence_inputs(tmp_path)
    scratch = scratches[authority]
    scratch.mkdir(mode=0o700, parents=True)
    launched = False

    def forbidden_popen(*_args, **_kwargs):
        nonlocal launched
        launched = True
        raise AssertionError("scratch fencing must precede launch")

    worker = StreamingWorker(
        manifest,
        scratch,
        source_bundle,
        popen_factory=forbidden_popen,
    )

    with pytest.raises(WorkerError, match="worker_prerequisite"):
        worker.start()

    assert launched is False


@pytest.mark.parametrize(
    "authority", ["runtime", "provision", "model", "control", "repo"]
)
def test_schema11_bwrap_rejects_scratch_overlapping_read_only_authority(
    tmp_path: Path,
    authority: str,
) -> None:
    manifest, source_bundle, scratches = _scratch_fence_inputs(tmp_path)

    with pytest.raises(WorkerError, match="worker_prerequisite"):
        supervisor_module._zipformer_bwrap_command(
            manifest,
            scratches[authority],
            source_bundle,
            bundle_descriptor=10,
            worker_descriptor=11,
            python_descriptor=12,
            system_ro_paths=(),
        )


@pytest.mark.parametrize(
    ("adapter", "expected"),
    [
        (MOBILE_WHISPER_ADAPTER, ("-I", "-S", "-B")),
        (MOBILE_ZIPFORMER_ADAPTER, ("-I", "-B")),
        (SHERPA_ZIPFORMER_ADAPTER, ("-I", "-B")),
    ],
)
def test_outer_and_bwrap_interpreter_vectors_are_schema11_only(
    tmp_path: Path,
    adapter: str,
    expected: tuple[str, ...],
) -> None:
    manifest, source_bundle = _bwrap_inputs(tmp_path, adapter)

    assert supervisor_module._worker_interpreter_flags(manifest) == expected
    command = supervisor_module._zipformer_bwrap_command(
        manifest,
        tmp_path / "scratch",
        source_bundle,
        bundle_descriptor=10,
        worker_descriptor=11,
        python_descriptor=12,
        system_ro_paths=(),
    )
    python_index = command.index("/proc/self/fd/12")

    assert tuple(command[python_index + 1 : python_index + 1 + len(expected)]) == expected
    assert command[python_index + 1 + len(expected)] == "/proc/self/fd/11"


def test_supervisor_launches_schema11_networkless_with_no_site(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_root = tmp_path / "fixture-authority"
    fixture_root.mkdir()
    manifest_path, _corpus, _digests = write_fixture(
        fixture_root,
        [{"values": [0.0], "expected_text": ""}],
        [
            {
                "partials": [],
                "final": "",
                "elapsed_ms": 1.0,
                "finalization_ms": 0.0,
                "compute_ms": 0.0,
                "deadline_misses": 0,
                "max_backlog_ms": 0.0,
                "resources": {"rss_mb": 1.0, "threads": 1, "vram_mb": None},
                "hang_sec": 0.0,
            }
        ],
    )
    base = load_worker_manifest(manifest_path)
    model_root = _model_root(tmp_path)
    lock = tmp_path / "source-lock.json"
    lock.write_text("{}\n", encoding="utf-8")
    manifest = replace(
        base,
        schema_version=11,
        model_id="sherpa-onnx-whisper-base.en-mobile-int8-v1",
        adapter=MOBILE_WHISPER_ADAPTER,
        adapter_config=MobileWhisperConfig(),
        artifacts=tuple(
            BoundArtifact(
                path=model_root / basename,
                sha256=digest,
                size_bytes=size_bytes,
                name=name,
            )
            for name, basename, _precision, digest, size_bytes in (
                MOBILE_WHISPER_ARTIFACT_SPECS
            )
        ),
        control_files=(
            BoundFile(
                path=lock.resolve(strict=True),
                sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),
                size_bytes=lock.stat().st_size,
            ),
        ),
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    bundle = stage_test_source_bundle(scratch, manifest.worker.path)
    captured: dict[str, object] = {}

    def rejecting_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        raise OSError

    worker = StreamingWorker(manifest, scratch, bundle, popen_factory=rejecting_popen)
    monkeypatch.setattr(worker, "_verify_bound_files", lambda: None)

    with pytest.raises(WorkerError, match="worker_start"):
        worker.start()

    command = captured["command"]
    assert isinstance(command, list)
    separator = command.index("--")
    assert command[separator + 2 : separator + 5] == ["-I", "-S", "-B"]
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert "CUDA_DEVICE_ORDER" not in kwargs["env"]
    for name, value in worker_module._MOBILE_WHISPER_REQUIRED_ENV.items():
        assert kwargs["env"][name] == value


@pytest.mark.real_model
def test_real_model_one_private_generated_epoch_terminal_shape(tmp_path: Path) -> None:
    configured = os.environ.get("SPEAKER_MOBILE_WHISPER_PROVISION")
    if not configured:
        pytest.skip("set SPEAKER_MOBILE_WHISPER_PROVISION to an exact provision")
    provision = load_mobile_whisper_provision(Path(configured))
    manifest = load_worker_manifest(provision.path)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    raw = array("f", [0.0] * 16_000).tobytes()
    pcm = scratch / "private-generated-silence.f32le"
    pcm.write_bytes(raw)
    pcm.chmod(0o600)
    request = TranscribeRequest(
        request_id="real-mobile-whisper-smoke",
        pcm=PcmInput(
            path=pcm.resolve(strict=True),
            sha256=hashlib.sha256(raw).hexdigest(),
            samples=16_000,
        ),
        stream=StreamConfig(
            chunk_samples=1_600,
            pace="burst",
            partial_interval_ms=100,
            tail_padding_samples=0,
        ),
        protocol_version=PROTOCOL_VERSION,
    )
    bundle = stage_test_source_bundle(scratch, manifest.worker.path)

    with StreamingWorker(manifest, scratch, bundle) as worker:
        trace = worker.transcribe(request)

    assert trace.partials == ()
    assert type(trace.final) is FinalEvent
    assert trace.final.request_id == request.request_id
    assert trace.final.seq == 0
    assert isinstance(trace.final.text, str)
    assert trace.final.samples_seen == request.pcm.samples
    assert trace.final.model_padding_samples == 0
