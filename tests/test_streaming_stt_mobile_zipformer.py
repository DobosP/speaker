from __future__ import annotations

from array import array
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.streaming_stt_helpers import stage_test_source_bundle, write_fixture
from tools.streaming_stt.bounded_io import FileDigest
import tools.streaming_stt.manifest as manifest_module
from tools.streaming_stt.manifest import (
    MOBILE_ZIPFORMER_ADAPTER,
    MOBILE_ZIPFORMER_ARTIFACT_SET_SHA256,
    MOBILE_ZIPFORMER_ARTIFACT_SPECS,
    MOBILE_ZIPFORMER_SOURCE_LOCK_RECIPE_SHA256,
    MOBILE_ZIPFORMER_SOURCE_LOCK_SHA256,
    MOBILE_ZIPFORMER_SOURCE_LOCK_SIZE_BYTES,
    MOBILE_ZIPFORMER_TOTAL_SIZE_BYTES,
    BoundArtifact,
    BoundFile,
    ManifestError,
    MobileZipformerConfig,
    load_worker_manifest,
)
from tools.streaming_stt.protocol import (
    MAX_CORPUS_BYTES,
    MOBILE_ZIPFORMER_ENDPOINT_PROTOCOL_VERSION,
    ErrorEvent,
    MobileZipformerFinalEvent,
    PcmInput,
    ProtocolError,
    StreamConfig,
    TranscribeRequest,
    encode_message,
    parse_response,
)
from tools.streaming_stt.supervisor import StreamingWorker, WorkerError
import tools.streaming_stt.supervisor as supervisor_module
import tools.streaming_stt.worker as worker_module


_MOBILE_PACKET_PCM_BYTES = (
    3_714_904,
    26_230_128,
    2_309_120,
    284_668,
    6_760_680,
)


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float = 0.001) -> None:
        self.value += seconds

    def sleep(self, seconds: float) -> None:
        self.advance(seconds)


class _Numpy:
    __version__ = "2.4.6"

    def __init__(self) -> None:
        self.dtypes: list[str] = []

    def asarray(self, values, *, dtype):
        self.dtypes.append(dtype)
        return list(values)


class _MobileStream:
    def __init__(self, recognizer: _MobileRecognizer) -> None:
        self.recognizer = recognizer
        self.ready = False
        self.chunks: list[list[float]] = []

    def accept_waveform(self, sample_rate: int, samples) -> None:
        assert sample_rate == 16_000
        self.recognizer.operations.append("accept")
        self.chunks.append(list(samples))
        self.ready = True
        self.recognizer.clock.advance()

    def input_finished(self) -> None:
        raise AssertionError("mobile adapter must never call input_finished")


class _MobileRecognizer:
    def __init__(
        self,
        clock: _Clock,
        *,
        results: list[str],
        endpoints: list[bool],
        reset_fails: bool = False,
    ) -> None:
        assert len(results) == len(endpoints)
        self.clock = clock
        self.results = results
        self.endpoints = endpoints
        self.reset_fails = reset_fails
        self.operations: list[str] = []
        self.streams: list[_MobileStream] = []
        self.decode_index = -1
        self.reset_calls = 0

    def create_stream(self) -> _MobileStream:
        self.operations.append("create")
        self.clock.advance()
        stream = _MobileStream(self)
        self.streams.append(stream)
        return stream

    def is_ready(self, stream: _MobileStream) -> bool:
        return stream.ready

    def decode_stream(self, stream: _MobileStream) -> None:
        assert stream.ready
        self.operations.append("decode")
        self.clock.advance()
        self.decode_index += 1
        stream.ready = False

    def get_result(self, stream: _MobileStream) -> str:
        del stream
        self.operations.append("get_result")
        self.clock.advance()
        return self.results[self.decode_index]

    def is_endpoint(self, stream: _MobileStream) -> bool:
        del stream
        self.operations.append("is_endpoint")
        self.clock.advance()
        return self.endpoints[self.decode_index]

    def reset(self, stream: _MobileStream) -> None:
        del stream
        self.operations.append("reset")
        self.clock.advance()
        self.reset_calls += 1
        if self.reset_fails:
            raise RuntimeError


class _MobileFactory:
    def __init__(self, recognizer: _MobileRecognizer) -> None:
        self.recognizer = recognizer
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        self.recognizer.clock.advance(0.01)
        return self.recognizer


def _api(factory: _MobileFactory) -> object:
    return SimpleNamespace(
        __version__="1.13.3",
        OnlineRecognizer=SimpleNamespace(from_transducer=factory),
    )


def _model_root(tmp_path: Path) -> Path:
    root = tmp_path / "model"
    root.mkdir()
    for _name, basename, _precision, _digest, _size in MOBILE_ZIPFORMER_ARTIFACT_SPECS:
        (root / basename).write_bytes(b"mobile-test-placeholder")
    return root


def _request(
    tmp_path: Path,
    samples: int,
    *,
    tail_padding_samples: int,
) -> TranscribeRequest:
    raw = array("f", [0.25] * samples).tobytes()
    path = tmp_path / f"case-{samples}.f32le"
    path.write_bytes(raw)
    return TranscribeRequest(
        request_id="mobile-case-0",
        pcm=PcmInput(
            path=path.resolve(strict=True),
            sha256=hashlib.sha256(raw).hexdigest(),
            samples=samples,
        ),
        stream=StreamConfig(
            chunk_samples=1_600,
            pace="burst",
            partial_interval_ms=200,
            tail_padding_samples=tail_padding_samples,
        ),
        protocol_version=MOBILE_ZIPFORMER_ENDPOINT_PROTOCOL_VERSION,
    )


def _adapter(
    tmp_path: Path,
    *,
    results: list[str],
    endpoints: list[bool],
    reset_fails: bool = False,
):
    from tools.streaming_stt.adapters.zipformer import MobileZipformerAdapter

    clock = _Clock()
    recognizer = _MobileRecognizer(
        clock,
        results=results,
        endpoints=endpoints,
        reset_fails=reset_fails,
    )
    factory = _MobileFactory(recognizer)
    numpy = _Numpy()
    adapter = MobileZipformerAdapter(
        _model_root(tmp_path),
        config=MobileZipformerConfig(),
        api=_api(factory),
        numpy_api=numpy,
        clock=clock,
        sleeper=clock.sleep,
    )
    return adapter, recognizer, factory, numpy


def test_mobile_adapter_preserves_empty_and_multiple_reset_committed_endpoints(
    tmp_path,
):
    adapter, recognizer, factory, numpy = _adapter(
        tmp_path,
        results=["alpha", "", "omega"],
        endpoints=[True, True, True],
    )
    operations: list[str] = []

    def emit(_seq, partial) -> None:
        operations.append(f"emit:{partial.text}")
        recognizer.operations.append("emit")

    request = _request(tmp_path, 4_800, tail_padding_samples=48_000)
    case = adapter.transcribe(request, emit_partial=emit)

    assert factory.calls == [
        {
            "tokens": str(adapter.model_root / "tokens.txt"),
            "encoder": str(
                adapter.model_root
                / "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx"
            ),
            "decoder": str(
                adapter.model_root / "decoder-epoch-99-avg-1-chunk-16-left-128.onnx"
            ),
            "joiner": str(
                adapter.model_root / "joiner-epoch-99-avg-1-chunk-16-left-128.onnx"
            ),
            "num_threads": 1,
            "provider": "cpu",
            "sample_rate": 16_000,
            "feature_dim": 80,
            "enable_endpoint_detection": True,
            "decoding_method": "greedy_search",
            "max_active_paths": 4,
            "rule1_min_trailing_silence": 2.4,
            "rule2_min_trailing_silence": 0.8,
            "rule3_min_utterance_length": 20.0,
            "debug": True,
            "model_type": "zipformer2",
        }
    ]
    assert recognizer.operations == [
        "create",
        "accept",
        "decode",
        "get_result",
        "is_endpoint",
        "reset",
        "emit",
        "accept",
        "decode",
        "get_result",
        "is_endpoint",
        "reset",
        "emit",
        "accept",
        "decode",
        "get_result",
        "is_endpoint",
        "reset",
        "emit",
    ]
    assert operations == ["emit:alpha", "emit:", "emit:omega"]
    assert [item.text for item in case.endpoint_observations] == ["alpha", "", "omega"]
    assert [item.source_complete for item in case.endpoint_observations] == [
        False,
        False,
        True,
    ]
    assert case.final == "alpha omega"
    assert case.endpoint_reason == "endpoint"
    assert case.endpoint_sample == 4_800
    assert case.endpoint_latency_ms == 0.0
    assert case.tail_samples_consumed == 0
    assert case.chunks_yielded == 3
    assert recognizer.reset_calls == 3
    assert numpy.dtypes == ["float32"] * 3


def test_mobile_adapter_tail_exhaustion_never_promotes_last_partial(tmp_path):
    adapter, recognizer, _factory, _numpy = _adapter(
        tmp_path,
        results=["draft", "", "unterminated"],
        endpoints=[False, False, False],
    )
    emitted = []

    case = adapter.transcribe(
        _request(tmp_path, 800, tail_padding_samples=3_200),
        emit_partial=lambda seq, partial: emitted.append((seq, partial.text)),
    )

    assert emitted == [(0, "draft"), (1, ""), (2, "unterminated")]
    assert case.endpoint_observations == ()
    assert case.final == ""
    assert case.endpoint_reason == "tail_exhausted"
    assert case.native_endpoint is False
    assert case.authoritative is False
    assert case.endpoint_sample is None
    assert case.endpoint_latency_ms is None
    assert case.tail_samples_consumed == 3_200
    assert case.chunks_yielded == 3
    assert recognizer.reset_calls == 0


def test_mobile_adapter_continues_after_early_endpoint_until_post_source_endpoint(
    tmp_path,
):
    adapter, _recognizer, _factory, _numpy = _adapter(
        tmp_path,
        results=["alpha", "draft", "omega"],
        endpoints=[True, False, True],
    )

    case = adapter.transcribe(
        _request(tmp_path, 3_200, tail_padding_samples=48_000),
        emit_partial=lambda _seq, _partial: None,
    )

    assert [item.observed_sample for item in case.endpoint_observations] == [
        1_600,
        4_800,
    ]
    assert case.final == "alpha omega"
    assert case.tail_samples_consumed == 1_600
    assert case.endpoint_sample == 4_800
    assert case.endpoint_latency_ms == 100.0
    assert case.chunks_yielded == 3


def test_mobile_adapter_reset_failure_suppresses_endpoint_and_partial(tmp_path):
    from tools.streaming_stt.adapters.zipformer import MobileZipformerAdapterError

    adapter, recognizer, _factory, _numpy = _adapter(
        tmp_path,
        results=["would-be-final"],
        endpoints=[True],
        reset_fails=True,
    )
    emitted = []

    with pytest.raises(MobileZipformerAdapterError):
        adapter.transcribe(
            _request(tmp_path, 1_600, tail_padding_samples=0),
            emit_partial=lambda seq, partial: emitted.append((seq, partial.text)),
        )

    assert emitted == []
    assert recognizer.reset_calls == 1


def test_mobile_adapter_maps_bound_pcm_failure_to_mobile_error(tmp_path):
    from tools.streaming_stt.adapters.zipformer import MobileZipformerAdapterError

    adapter, _recognizer, _factory, _numpy = _adapter(
        tmp_path,
        results=["unused"],
        endpoints=[False],
    )
    request = _request(tmp_path, 1_600, tail_padding_samples=0)
    request = replace(request, pcm=replace(request.pcm, sha256="0" * 64))

    with pytest.raises(MobileZipformerAdapterError):
        adapter.transcribe(request, emit_partial=lambda _seq, _partial: None)


def test_mobile_runtime_candidate_failure_uses_mobile_error(monkeypatch):
    from importlib import metadata
    import tools.streaming_stt.adapters.zipformer as zipformer_module
    from tools.streaming_stt.adapters.zipformer import MobileZipformerAdapterError

    def missing(_distribution: str) -> str:
        raise metadata.PackageNotFoundError

    monkeypatch.setattr(zipformer_module.metadata, "version", missing)

    with pytest.raises(MobileZipformerAdapterError):
        zipformer_module._mobile_candidate_apis(MobileZipformerConfig())


def _schema_v10_payload(tmp_path: Path) -> dict[str, object]:
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
    model_root = tmp_path / "model"
    source_lock = {
        "path": str((tmp_path / "source-lock.json").resolve()),
        "sha256": MOBILE_ZIPFORMER_SOURCE_LOCK_SHA256,
        "size_bytes": MOBILE_ZIPFORMER_SOURCE_LOCK_SIZE_BYTES,
    }
    return {
        "schema_version": 10,
        "kind": "mobile-zipformer-provision-v1",
        "model_id": ("sherpa-onnx-streaming-zipformer-en-2023-06-26-mobile-hybrid-v1"),
        "adapter": MOBILE_ZIPFORMER_ADAPTER,
        "python": python,
        "worker": worker,
        "artifacts": [
            {
                "name": name,
                "path": str((model_root / basename).resolve()),
                "sha256": digest,
                "size_bytes": size,
            }
            for name, basename, _precision, digest, size in (
                MOBILE_ZIPFORMER_ARTIFACT_SPECS
            )
        ],
        "limits": {"startup_timeout_sec": 120.0, "case_timeout_sec": 300.0},
        "mobile_config": MobileZipformerConfig().as_dict(),
        "source": {
            "repo_id": MobileZipformerConfig().source_repo_id,
            "revision": MobileZipformerConfig().source_revision,
            "lock": source_lock,
            "lock_recipe_sha256": MOBILE_ZIPFORMER_SOURCE_LOCK_RECIPE_SHA256,
        },
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
        "artifact_set_sha256": MOBILE_ZIPFORMER_ARTIFACT_SET_SHA256,
        "total_size_bytes": MOBILE_ZIPFORMER_TOTAL_SIZE_BYTES,
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


def _load_mocked_manifest(tmp_path: Path, monkeypatch, payload):
    manifest_path = tmp_path / "worker-manifest.json"
    manifest_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    rows = [
        payload["python"],
        payload["worker"],
        payload["source"]["lock"],
        *payload["artifacts"],
    ]
    receipts = {
        Path(item["path"]): (item["sha256"], item["size_bytes"]) for item in rows
    }

    def fake_hash(path, *, maximum_bytes, expected_bytes, allow_final_symlink=False):
        del maximum_bytes, allow_final_symlink
        digest, size = receipts[Path(path)]
        assert expected_bytes == size
        return FileDigest(Path(path), digest, size)

    monkeypatch.setattr(manifest_module, "hash_regular_bounded", fake_hash)
    return load_worker_manifest(manifest_path)


def test_schema_v10_manifest_maps_exact_mobile_provision_contract(
    tmp_path,
    monkeypatch,
):
    manifest = _load_mocked_manifest(
        tmp_path,
        monkeypatch,
        _schema_v10_payload(tmp_path),
    )

    assert manifest.schema_version == 10
    assert manifest.adapter == MOBILE_ZIPFORMER_ADAPTER
    assert manifest.adapter_config == MobileZipformerConfig()
    assert manifest.control_files == (
        BoundFile(
            path=(tmp_path / "source-lock.json").resolve(),
            sha256=MOBILE_ZIPFORMER_SOURCE_LOCK_SHA256,
            size_bytes=MOBILE_ZIPFORMER_SOURCE_LOCK_SIZE_BYTES,
        ),
    )
    assert [
        (item.name, item.path.name, item.sha256, item.size_bytes)
        for item in manifest.artifacts
    ] == [
        (name, basename, digest, size)
        for name, basename, _precision, digest, size in (
            MOBILE_ZIPFORMER_ARTIFACT_SPECS
        )
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        "artifact",
        "config",
        "runtime",
        "source",
        "source_lock",
        "limits",
        "evidence",
        "total",
        "extra",
    ],
)
def test_schema_v10_manifest_rejects_any_mobile_contract_drift(
    tmp_path,
    monkeypatch,
    mutation,
):
    payload = _schema_v10_payload(tmp_path)
    if mutation == "artifact":
        payload["artifacts"][0]["sha256"] = "0" * 64
    elif mutation == "config":
        payload["mobile_config"]["debug"] = False
    elif mutation == "runtime":
        payload["runtime"]["distributions"]["sherpa-onnx-core"] = "1.13.2"
    elif mutation == "source":
        payload["source"]["lock_recipe_sha256"] = "0" * 64
    elif mutation == "source_lock":
        payload["source"]["lock"]["sha256"] = "0" * 64
    elif mutation == "limits":
        payload["limits"]["case_timeout_sec"] = 299.0
    elif mutation == "evidence":
        payload["evidence_scope"]["model_loaded"] = True
    elif mutation == "total":
        payload["total_size_bytes"] += 1
    else:
        payload["unexpected"] = False

    with pytest.raises(ManifestError):
        _load_mocked_manifest(tmp_path, monkeypatch, payload)


def _mobile_final_payload(
    *,
    observations: list[dict[str, object]],
    endpoint: bool,
) -> dict[str, object]:
    source_samples = 3_200
    tail = 0
    text = " ".join(str(item["text"]) for item in observations if item["text"])
    return {
        "v": MOBILE_ZIPFORMER_ENDPOINT_PROTOCOL_VERSION,
        "id": "mobile-case-0",
        "type": "mobile_zipformer_final",
        "seq": 2,
        "text": text,
        "samples_seen": source_samples + tail,
        "elapsed_ms": 10.0,
        "finalization_ms": 1.0,
        "compute_ms": 5.0,
        "audio_seconds": 0.2,
        "chunks": 2,
        "deadline_misses": 0,
        "max_backlog_ms": 0.0,
        "model_padding_samples": 0,
        "resources": {"rss_mb": 64.0, "threads": 1, "vram_mb": None},
        "source_samples": source_samples,
        "source_samples_consumed": source_samples,
        "declared_tail_samples": 0,
        "tail_samples_consumed": tail,
        "endpoint_observations": observations,
        "endpoint_reason": "endpoint" if endpoint else "tail_exhausted",
        "native_endpoint": endpoint,
        "endpoint_sample": source_samples if endpoint else None,
        "endpoint_latency_ms": 0.0 if endpoint else None,
        "authoritative": endpoint,
    }


@pytest.mark.parametrize(
    ("observations", "endpoint", "expected_texts"),
    [
        ([], False, []),
        (
            [{"text": "", "observed_sample": 1_600, "source_complete": False}],
            False,
            [""],
        ),
        (
            [{"text": "done", "observed_sample": 3_200, "source_complete": True}],
            True,
            ["done"],
        ),
        (
            [
                {"text": "first", "observed_sample": 1_600, "source_complete": False},
                {"text": "", "observed_sample": 3_200, "source_complete": True},
            ],
            True,
            ["first", ""],
        ),
    ],
)
def test_protocol_v5_preserves_zero_empty_one_and_multiple_endpoint_observations(
    observations,
    endpoint,
    expected_texts,
):
    event = parse_response(
        encode_message(
            _mobile_final_payload(
                observations=observations,
                endpoint=endpoint,
            )
        )
    )

    assert type(event) is MobileZipformerFinalEvent
    assert [item.text for item in event.endpoint_observations] == expected_texts


def test_protocol_v5_rejects_last_partial_promotion_on_tail_exhaustion():
    payload = _mobile_final_payload(observations=[], endpoint=False)
    payload["text"] = "unterminated partial"

    with pytest.raises(ProtocolError):
        parse_response(encode_message(payload))


def test_worker_payload_round_trips_mobile_case_and_supervisor_revalidates(tmp_path):
    adapter, _recognizer, _factory, _numpy = _adapter(
        tmp_path,
        results=["first", "last"],
        endpoints=[True, True],
    )
    request = _request(tmp_path, 3_200, tail_padding_samples=48_000)
    emitted = []
    case = adapter.transcribe(
        request,
        emit_partial=lambda seq, partial: emitted.append((seq, partial)),
    )
    payload = worker_module._mobile_zipformer_final_payload(
        request,
        case,
        MobileZipformerConfig(),
        partial_count=len(emitted),
    )
    event = parse_response(encode_message(payload))
    assert type(event) is MobileZipformerFinalEvent

    supervisor_module._validate_mobile_zipformer_final(
        SimpleNamespace(adapter_config=MobileZipformerConfig()),
        request,
        event,
        expected_seq=len(emitted),
        last_samples=emitted[-1][1].after_samples,
        last_elapsed=emitted[-1][1].elapsed_ms,
    )


def test_worker_dispatch_selects_separate_mobile_adapter(tmp_path, monkeypatch):
    import tools.streaming_stt.adapters.zipformer as zipformer_module

    root = _model_root(tmp_path)
    artifacts = tuple(
        SimpleNamespace(name=name, path=root / basename)
        for name, basename, _precision, _digest, _size in (
            MOBILE_ZIPFORMER_ARTIFACT_SPECS
        )
    )
    manifest = SimpleNamespace(
        schema_version=10,
        adapter=MOBILE_ZIPFORMER_ADAPTER,
        adapter_config=MobileZipformerConfig(),
        artifacts=artifacts,
    )
    sentinel = object()
    calls = []

    def create(model_root, *, config):
        calls.append((model_root, config))
        return sentinel

    class CandidateError(RuntimeError):
        pass

    monkeypatch.setattr(zipformer_module, "MobileZipformerAdapter", create)
    monkeypatch.setattr(
        zipformer_module,
        "MobileZipformerAdapterError",
        CandidateError,
    )

    adapter, error_type = worker_module._create_adapter(manifest)

    assert adapter is sentinel
    assert error_type is CandidateError
    assert calls == [(root.resolve(strict=True), MobileZipformerConfig())]
    assert (
        worker_module._wire_protocol_version(manifest)
        == MOBILE_ZIPFORMER_ENDPOINT_PROTOCOL_VERSION
    )
    assert (
        supervisor_module._wire_protocol_version(manifest)
        == MOBILE_ZIPFORMER_ENDPOINT_PROTOCOL_VERSION
    )


def test_worker_runtime_activation_admits_exact_mobile_schema(monkeypatch):
    fake_sys = SimpleNamespace(
        flags=SimpleNamespace(
            no_site=0,
            no_user_site=1,
            ignore_environment=1,
            isolated=1,
            dont_write_bytecode=1,
        ),
        modules={},
    )
    monkeypatch.setattr(worker_module, "sys", fake_sys)
    manifest = SimpleNamespace(
        adapter=MOBILE_ZIPFORMER_ADAPTER,
        schema_version=10,
        adapter_config=MobileZipformerConfig(),
    )

    worker_module._activate_candidate_runtime(manifest, None)

    manifest.schema_version = 4
    with pytest.raises(ManifestError):
        worker_module._activate_candidate_runtime(manifest, None)


def test_mobile_worker_corpus_ceiling_is_narrow_and_generic_stays_32_mib():
    generic = SimpleNamespace(schema_version=2, adapter="another-adapter")
    mobile = SimpleNamespace(
        schema_version=10,
        adapter=MOBILE_ZIPFORMER_ADAPTER,
    )
    wrong_adapter = SimpleNamespace(schema_version=10, adapter="another-adapter")
    wrong_schema = SimpleNamespace(
        schema_version=9,
        adapter=MOBILE_ZIPFORMER_ADAPTER,
    )

    assert MAX_CORPUS_BYTES == 32 * 1024 * 1024
    assert worker_module._worker_corpus_limit_bytes(generic) == MAX_CORPUS_BYTES
    assert worker_module._worker_corpus_limit_bytes(wrong_adapter) == MAX_CORPUS_BYTES
    assert worker_module._worker_corpus_limit_bytes(wrong_schema) == MAX_CORPUS_BYTES
    mobile_ceiling = worker_module._worker_corpus_limit_bytes(mobile)
    packet_bytes = sum(_MOBILE_PACKET_PCM_BYTES)
    assert packet_bytes == 39_299_500
    assert MAX_CORPUS_BYTES < packet_bytes <= mobile_ceiling
    assert mobile_ceiling == 38 * 1024 * 1024


def test_mobile_worker_ledger_accepts_packet_scale_deduplicates_and_closes():
    ceiling = worker_module._MOBILE_ZIPFORMER_MAX_CORPUS_BYTES
    consumed_inputs: set[str] = set()
    consumed_bytes = 0
    hashes: list[str] = []
    for index, size_bytes in enumerate(_MOBILE_PACKET_PCM_BYTES, 1):
        digest = f"{index:064x}"
        hashes.append(digest)
        consumed_bytes = worker_module._account_unique_pcm(
            consumed_inputs,
            consumed_bytes,
            pcm_sha256=digest,
            size_bytes=size_bytes,
            maximum_bytes=ceiling,
        )

    assert consumed_bytes == 39_299_500
    assert len(consumed_inputs) == len(_MOBILE_PACKET_PCM_BYTES)
    before = set(consumed_inputs)
    assert (
        worker_module._account_unique_pcm(
            consumed_inputs,
            consumed_bytes,
            pcm_sha256=hashes[-1],
            size_bytes=_MOBILE_PACKET_PCM_BYTES[-1],
            maximum_bytes=ceiling,
        )
        == consumed_bytes
    )
    assert consumed_inputs == before

    consumed_bytes = worker_module._account_unique_pcm(
        consumed_inputs,
        consumed_bytes,
        pcm_sha256="a" * 64,
        size_bytes=ceiling - consumed_bytes,
        maximum_bytes=ceiling,
    )
    assert consumed_bytes == ceiling
    before = set(consumed_inputs)
    with pytest.raises(ProtocolError):
        worker_module._account_unique_pcm(
            consumed_inputs,
            consumed_bytes,
            pcm_sha256="b" * 64,
            size_bytes=4,
            maximum_bytes=ceiling,
        )
    assert consumed_inputs == before


@pytest.mark.parametrize(
    ("maximum_bytes", "digest"),
    (
        (MAX_CORPUS_BYTES, "c" * 64),
        (worker_module._MOBILE_ZIPFORMER_MAX_CORPUS_BYTES, "d" * 64),
    ),
)
def test_worker_ledger_accepts_exact_ceiling_and_rejects_next_sample(
    maximum_bytes,
    digest,
):
    consumed_inputs: set[str] = set()
    assert (
        worker_module._account_unique_pcm(
            consumed_inputs,
            0,
            pcm_sha256=digest,
            size_bytes=maximum_bytes,
            maximum_bytes=maximum_bytes,
        )
        == maximum_bytes
    )
    before = set(consumed_inputs)
    with pytest.raises(ProtocolError):
        worker_module._account_unique_pcm(
            consumed_inputs,
            maximum_bytes,
            pcm_sha256="e" * 64,
            size_bytes=4,
            maximum_bytes=maximum_bytes,
        )
    assert consumed_inputs == before


def test_mobile_worker_over_budget_failure_is_fatal_and_detail_free(
    tmp_path,
    monkeypatch,
):
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    pcm = scratch / "input.f32le"
    raw = array("f", [0.0]).tobytes()
    pcm.write_bytes(raw)
    pcm.chmod(0o600)
    request = TranscribeRequest(
        request_id="mobile-budget",
        pcm=PcmInput(
            path=pcm.resolve(strict=True),
            sha256=hashlib.sha256(raw).hexdigest(),
            samples=1,
        ),
        stream=StreamConfig(
            chunk_samples=1_600,
            pace="burst",
            partial_interval_ms=100,
            tail_padding_samples=48_000,
        ),
        protocol_version=MOBILE_ZIPFORMER_ENDPOINT_PROTOCOL_VERSION,
    )
    worker_path = Path(worker_module.__file__).resolve(strict=True)
    worker_receipt = SimpleNamespace(sha256="c" * 64, size_bytes=123)
    python_receipt = SimpleNamespace(sha256="d" * 64, size_bytes=456)
    manifest = SimpleNamespace(
        path=tmp_path / "worker-manifest.json",
        digest="e" * 64,
        schema_version=10,
        model_id="mobile-budget-test",
        adapter=MOBILE_ZIPFORMER_ADAPTER,
        adapter_config=MobileZipformerConfig(),
        worker=worker_receipt,
        python=python_receipt,
    )
    source_bundle = SimpleNamespace(
        tree_sha256="f" * 64,
        worker_relative_path="tools/streaming_stt/worker.py",
        worker_path=worker_path,
        file_by_path={"tools/streaming_stt/worker.py": worker_receipt},
    )
    output: list[dict[str, object]] = []

    class Candidate:
        ready = SimpleNamespace(
            model_load_ms=1.0,
            resources=SimpleNamespace(rss_mb=1.0, threads=1, vram_mb=None),
        )

        def __init__(self) -> None:
            self.close_calls = 0
            self.transcribe_calls = 0

        def transcribe(self, *_args, **_kwargs):
            self.transcribe_calls += 1
            raise AssertionError("over-budget PCM must fail before inference")

        def close(self) -> None:
            self.close_calls += 1

    class CandidateFailure(RuntimeError):
        pass

    candidate = Candidate()
    monkeypatch.setattr(worker_module, "_SOURCE_BUNDLE_SHA256", "f" * 64)
    monkeypatch.setattr(
        worker_module,
        "_MOBILE_ZIPFORMER_MAX_CORPUS_BYTES",
        4,
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
        "_verify_candidate_python_version",
        lambda *_args, **_kwargs: None,
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
        lambda *_args, **_kwargs: (candidate, CandidateFailure),
    )
    monkeypatch.setattr(worker_module, "_load_pcm", lambda *_args: b"12345")
    monkeypatch.setattr(worker_module, "_write", output.append)
    monkeypatch.setattr(
        worker_module.sys,
        "stdin",
        SimpleNamespace(buffer=io.BytesIO(encode_message(request.as_dict()))),
    )

    result = worker_module._run(
        manifest.path,
        scratch,
        source_bundle_manifest=Path("unused-source-bundle.json"),
        source_bundle_sha256=source_bundle.tree_sha256,
    )
    responses = [parse_response(encode_message(value)) for value in output]

    assert result == 2
    assert candidate.transcribe_calls == 0
    assert candidate.close_calls == 1
    assert len(responses) == 2
    error = responses[-1]
    assert type(error) is ErrorEvent
    assert error.code == "candidate_failure"
    assert error.fatal is True
    assert output[-1] == {
        "v": MOBILE_ZIPFORMER_ENDPOINT_PROTOCOL_VERSION,
        "id": request.request_id,
        "type": "error",
        "code": "candidate_failure",
        "fatal": True,
    }


def test_supervisor_launches_mobile_worker_networkless_cpu_only_and_mounts_lock(
    tmp_path,
    monkeypatch,
):
    manifest_path, _corpus, _digests = write_fixture(
        tmp_path,
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
        schema_version=10,
        model_id="sherpa-onnx-streaming-zipformer-en-2023-06-26-mobile-hybrid-v1",
        adapter=MOBILE_ZIPFORMER_ADAPTER,
        adapter_config=MobileZipformerConfig(),
        artifacts=tuple(
            BoundArtifact(
                path=model_root / basename,
                sha256=digest,
                size_bytes=size,
                name=name,
            )
            for name, basename, _precision, digest, size in (
                MOBILE_ZIPFORMER_ARTIFACT_SPECS
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
    captured = {}

    def rejecting_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        raise OSError

    worker = StreamingWorker(
        manifest,
        scratch,
        bundle,
        popen_factory=rejecting_popen,
    )
    monkeypatch.setattr(worker, "_verify_bound_files", lambda: None)

    with pytest.raises(WorkerError, match="worker_start"):
        worker.start()

    command = captured["command"]
    kwargs = captured["kwargs"]
    assert command[0] == "/usr/bin/bwrap"
    assert command.count("--unshare-net") == 1
    ro_pairs = {
        (command[index + 1], command[index + 2])
        for index, value in enumerate(command)
        if value == "--ro-bind"
    }
    assert (str(lock.resolve(strict=True)), str(lock.resolve(strict=True))) in ro_pairs
    separator = command.index("--")
    assert command[separator + 2 : separator + 4] == ["-I", "-B"]
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert "CUDA_DEVICE_ORDER" not in kwargs["env"]
    assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
    assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
