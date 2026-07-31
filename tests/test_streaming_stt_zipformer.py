from __future__ import annotations

from array import array
from dataclasses import replace
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tests.streaming_stt_helpers import (
    scripted_case,
    stage_test_source_bundle,
    write_fixture,
)
from tools import provision_zipformer_baseline, streaming_stt_eval
from tools.streaming_stt.bounded_io import FileDigest
import tools.streaming_stt.manifest as manifest_module
from tools.streaming_stt.manifest import (
    BoundArtifact,
    SHERPA_ZIPFORMER_ADAPTER,
    SHERPA_ZIPFORMER_ARTIFACT_SPECS,
    ManifestError,
    SherpaZipformerConfig,
    load_worker_manifest,
)
from tools.streaming_stt.protocol import PcmInput, StreamConfig, TranscribeRequest
from tools.streaming_stt.supervisor import StreamingWorker, WorkerError
import tools.streaming_stt.worker as worker_module


_REPO_ROOT = Path(__file__).resolve().parents[1]


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
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


class _Stream:
    def __init__(self, recognizer) -> None:
        self.recognizer = recognizer
        self.chunks: list[list[float]] = []
        self.ready = False
        self.finished = False
        self.text = ""
        self.input_finished_calls = 0

    def accept_waveform(self, sample_rate: int, samples) -> None:
        assert sample_rate == 16_000
        self.chunks.append(list(samples))
        self.ready = True
        self.recognizer.clock.advance(0.001)

    def input_finished(self) -> None:
        self.input_finished_calls += 1
        self.finished = True
        self.ready = True
        self.recognizer.clock.advance(0.001)


class _Recognizer:
    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self.streams: list[_Stream] = []
        self.resets: list[_Stream] = []

    def create_stream(self) -> _Stream:
        self.clock.advance(0.001)
        stream = _Stream(self)
        self.streams.append(stream)
        return stream

    def is_ready(self, stream: _Stream) -> bool:
        return stream.ready

    def decode_stream(self, stream: _Stream) -> None:
        assert stream.ready
        self.clock.advance(0.002)
        stream.ready = False
        stream.text = (
            "final flushed"
            if stream.finished
            else ("first partial" if len(stream.chunks) == 1 else "second partial")
        )

    def get_result(self, stream: _Stream) -> str:
        self.clock.advance(0.001)
        return stream.text

    def reset(self, stream: _Stream) -> None:
        self.clock.advance(0.001)
        self.resets.append(stream)
        stream.ready = False
        stream.text = ""


class _Factory:
    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self.calls: list[dict[str, object]] = []
        self.recognizer = _Recognizer(clock)

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        self.clock.advance(0.010)
        return self.recognizer


def _model_root(tmp_path: Path) -> Path:
    root = tmp_path / "model"
    root.mkdir()
    for _name, basename, _digest, _size in SHERPA_ZIPFORMER_ARTIFACT_SPECS:
        (root / basename).write_bytes(b"local-test-placeholder")
    return root


def _request(
    tmp_path: Path,
    values: list[float],
    *,
    chunk_samples: int,
    tail_padding_samples: int,
) -> TranscribeRequest:
    raw = array("f", values).tobytes()
    path = tmp_path / "case.f32le"
    path.write_bytes(raw)
    return TranscribeRequest(
        request_id="zipformer-case-0",
        pcm=PcmInput(
            path=path.resolve(strict=True),
            sha256=hashlib.sha256(raw).hexdigest(),
            samples=len(values),
        ),
        stream=StreamConfig(
            chunk_samples=chunk_samples,
            pace="burst",
            partial_interval_ms=1,
            tail_padding_samples=tail_padding_samples,
        ),
    )


def _api(factory: _Factory, *, version: str = "1.13.3") -> object:
    return SimpleNamespace(
        __version__=version,
        OnlineRecognizer=SimpleNamespace(from_transducer=factory),
    )


def test_adapter_import_is_inert_and_does_not_import_native_runtime(monkeypatch):
    for name in tuple(sys.modules):
        if name in {"sherpa_onnx", "numpy"}:
            monkeypatch.delitem(sys.modules, name)

    original = importlib.import_module

    def guarded(name: str, *args, **kwargs):
        if name in {"sherpa_onnx", "numpy"}:
            raise AssertionError("native candidate imported at module import")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", guarded)
    module = original("tools.streaming_stt.adapters.zipformer")
    importlib.reload(module)


def test_adapter_binds_production_fp32_paths_and_one_thread_stream_contract(
    tmp_path,
):
    from tools.streaming_stt.adapters.zipformer import SherpaZipformerAdapter

    clock = _Clock()
    factory = _Factory(clock)
    numpy = _Numpy()
    adapter = SherpaZipformerAdapter(
        _model_root(tmp_path),
        config=SherpaZipformerConfig(),
        api=_api(factory),
        numpy_api=numpy,
        clock=clock,
        sleeper=clock.sleep,
    )
    emitted = []

    case = adapter.transcribe(
        _request(
            tmp_path,
            [1.0] * 17,
            chunk_samples=16,
            tail_padding_samples=15,
        ),
        emit_partial=lambda seq, partial: emitted.append((seq, partial)),
    )

    kwargs = factory.calls[0]
    assert kwargs == {
        "tokens": str((adapter.model_root / "tokens.txt")),
        "encoder": str(
            adapter.model_root
            / "encoder-epoch-99-avg-1-chunk-16-left-128.onnx"
        ),
        "decoder": str(
            adapter.model_root
            / "decoder-epoch-99-avg-1-chunk-16-left-128.onnx"
        ),
        "joiner": str(
            adapter.model_root
            / "joiner-epoch-99-avg-1-chunk-16-left-128.onnx"
        ),
        "num_threads": 1,
        "provider": "cpu",
        "sample_rate": 16_000,
        "feature_dim": 80,
        "enable_endpoint_detection": True,
        "decoding_method": "modified_beam_search",
        "max_active_paths": 4,
        "rule1_min_trailing_silence": 2.4,
        "rule2_min_trailing_silence": 0.8,
        "rule3_min_utterance_length": 20.0,
    }
    stream = factory.recognizer.streams[0]
    assert stream.chunks == [
        [1.0] * 16,
        [1.0] + [0.0] * 15,
    ]
    assert stream.input_finished_calls == 1
    assert factory.recognizer.resets == [stream]
    assert numpy.dtypes == ["float32", "float32"]
    assert [partial.text for _seq, partial in emitted] == [
        "first partial",
        "second partial",
    ]
    assert [seq for seq, _partial in emitted] == [0, 1]
    assert case.partials == tuple(partial for _seq, partial in emitted)
    assert case.final == "final flushed"
    assert case.compute_ms > 0.0


def test_adapter_rejects_package_version_mismatch_before_factory_call(tmp_path):
    from tools.streaming_stt.adapters.zipformer import (
        SherpaZipformerAdapter,
        SherpaZipformerAdapterError,
    )

    clock = _Clock()
    factory = _Factory(clock)
    with pytest.raises(SherpaZipformerAdapterError):
        SherpaZipformerAdapter(
            _model_root(tmp_path),
            config=SherpaZipformerConfig(),
            api=_api(factory, version="1.13.2"),
            numpy_api=_Numpy(),
            clock=clock,
            sleeper=clock.sleep,
        )
    assert factory.calls == []


def _schema_v4_payload(tmp_path: Path) -> dict[str, object]:
    model_root = tmp_path / "selected-model"
    return {
        "schema_version": 4,
        "model_id": "production-gigaspeech-zipformer-fp32-benchmark-1t",
        "adapter": SHERPA_ZIPFORMER_ADAPTER,
        "adapter_config": SherpaZipformerConfig().as_dict(),
        "python": {
            "path": str((tmp_path / "venv" / "bin" / "python").resolve()),
            "sha256": "a" * 64,
            "size_bytes": 123,
        },
        "worker": {
            "path": str((tmp_path / "worker.py").resolve()),
            "sha256": "b" * 64,
            "size_bytes": 456,
        },
        "artifacts": [
            {
                "name": name,
                "path": str((model_root / basename).resolve()),
                "sha256": digest,
                "size_bytes": size,
            }
            for name, basename, digest, size in SHERPA_ZIPFORMER_ARTIFACT_SPECS
        ],
        "limits": {
            "startup_timeout_sec": 120.0,
            "case_timeout_sec": 300.0,
        },
    }


def _load_mocked_manifest(tmp_path: Path, monkeypatch, payload):
    manifest_path = tmp_path / "worker-manifest.json"
    manifest_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    receipts = {
        Path(item["path"]): (item["sha256"], item["size_bytes"])
        for item in [payload["python"], payload["worker"], *payload["artifacts"]]
    }

    def fake_hash(path, *, maximum_bytes, expected_bytes, allow_final_symlink=False):
        del maximum_bytes, allow_final_symlink
        digest, size = receipts[Path(path)]
        assert expected_bytes == size
        return FileDigest(Path(path), digest, size)

    monkeypatch.setattr(manifest_module, "hash_regular_bounded", fake_hash)
    return load_worker_manifest(manifest_path)


def test_schema_v4_manifest_is_closed_to_exact_fp32_receipts(tmp_path, monkeypatch):
    payload = _schema_v4_payload(tmp_path)
    manifest = _load_mocked_manifest(tmp_path, monkeypatch, payload)

    assert manifest.schema_version == 4
    assert manifest.adapter == SHERPA_ZIPFORMER_ADAPTER
    assert manifest.adapter_config == SherpaZipformerConfig()
    assert [
        (item.name, item.path.name, item.sha256, item.size_bytes)
        for item in manifest.artifacts
    ] == list(SHERPA_ZIPFORMER_ARTIFACT_SPECS)


@pytest.mark.parametrize("mutation", ["digest", "int8-basename", "threads"])
def test_schema_v4_manifest_rejects_receipt_or_profile_drift(
    tmp_path,
    monkeypatch,
    mutation,
):
    payload = _schema_v4_payload(tmp_path)
    if mutation == "digest":
        payload["artifacts"][0]["sha256"] = "0" * 64
    elif mutation == "int8-basename":
        payload["artifacts"][0]["path"] = str(
            tmp_path
            / "selected-model"
            / "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx"
        )
    else:
        payload["adapter_config"]["num_threads"] = 4

    with pytest.raises(ManifestError):
        _load_mocked_manifest(tmp_path, monkeypatch, payload)


def test_comparison_profile_stays_in_parity_with_committed_production_config():
    config = SherpaZipformerConfig()
    committed = json.loads((_REPO_ROOT / "config.json").read_text(encoding="utf-8"))
    sherpa = committed["sherpa"]
    production_profile = committed["device_profiles"][
        config.production_device_profile
    ]["sherpa"]

    assert production_profile["asr_num_threads"] == config.production_num_threads == 4
    assert config.benchmark_profile == "resource-controlled-one-thread"
    assert config.num_threads == 1
    assert config.num_threads != config.production_num_threads
    assert sherpa["sample_rate"] == config.sample_rate
    assert sherpa["provider"] == config.provider
    assert sherpa["asr_decoding_method"] == config.decoding_method
    assert sherpa["asr_max_active_paths"] == config.max_active_paths
    assert sherpa["asr_rule1_min_trailing_silence"] == (
        config.rule1_min_trailing_silence
    )
    assert sherpa["asr_rule2_min_trailing_silence"] == (
        config.rule2_min_trailing_silence
    )
    assert sherpa["asr_rule3_min_utterance_length"] == (
        config.rule3_min_utterance_length
    )


def test_supervisor_launches_zipformer_with_site_but_offline_one_thread_env(
    tmp_path,
    monkeypatch,
):
    manifest_path, _corpus, _digests = write_fixture(
        tmp_path,
        [{"values": [0.0], "expected_text": ""}],
        [scripted_case(partials=[], final="")],
    )
    selected_model_root = _model_root(tmp_path)
    manifest = replace(
        load_worker_manifest(manifest_path),
        schema_version=4,
        adapter=SHERPA_ZIPFORMER_ADAPTER,
        adapter_config=SherpaZipformerConfig(),
        artifacts=tuple(
            BoundArtifact(
                path=selected_model_root / basename,
                sha256=digest,
                size_bytes=size,
                name=name,
            )
            for name, basename, digest, size in SHERPA_ZIPFORMER_ARTIFACT_SPECS
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
    assert (
        str(manifest.python.path.parent.parent),
        str(manifest.python.path.parent.parent),
    ) in ro_pairs
    assert (
        str(selected_model_root.resolve(strict=True)),
        str(selected_model_root.resolve(strict=True)),
    ) in ro_pairs
    assert (str(bundle.root), str(bundle.root)) in ro_pairs
    separator = command.index("--")
    assert command[separator + 2 : separator + 4] == ["-I", "-B"]
    assert "-S" not in command[separator + 1 :]
    environment = kwargs["env"]
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["OMP_NUM_THREADS"] == "1"
    assert environment["OPENBLAS_NUM_THREADS"] == "1"
    assert environment["MKL_NUM_THREADS"] == "1"
    assert kwargs["start_new_session"] is True
    assert kwargs["close_fds"] is True
    assert len(kwargs["pass_fds"]) == 4
    assert kwargs["executable"] == f"/proc/self/fd/{kwargs['pass_fds'][-1]}"


def test_worker_dispatch_selects_zipformer_only_after_manifest_selection(
    tmp_path,
    monkeypatch,
):
    import tools.streaming_stt.adapters.zipformer as zipformer_module

    root = _model_root(tmp_path)
    artifacts = tuple(
        SimpleNamespace(name=name, path=root / basename)
        for name, basename, _digest, _size in SHERPA_ZIPFORMER_ARTIFACT_SPECS
    )
    manifest = SimpleNamespace(
        adapter=SHERPA_ZIPFORMER_ADAPTER,
        adapter_config=SherpaZipformerConfig(),
        artifacts=artifacts,
    )
    calls = []
    sentinel = object()

    def create(model_root, *, config):
        calls.append((model_root, config))
        return sentinel

    class CandidateError(RuntimeError):
        pass

    monkeypatch.setattr(zipformer_module, "SherpaZipformerAdapter", create)
    monkeypatch.setattr(zipformer_module, "SherpaZipformerAdapterError", CandidateError)

    adapter, error_type = worker_module._create_adapter(manifest)

    assert adapter is sentinel
    assert error_type is CandidateError
    assert calls == [(root.resolve(strict=True), SherpaZipformerConfig())]


def test_evaluator_labels_production_model_but_not_runtime_equivalence(tmp_path):
    config = SherpaZipformerConfig()
    artifacts = tuple(
        SimpleNamespace(
            name=name,
            path=tmp_path / basename,
            sha256=digest,
            size_bytes=size,
        )
        for name, basename, digest, size in SHERPA_ZIPFORMER_ARTIFACT_SPECS
    )
    manifest = SimpleNamespace(
        digest="a" * 64,
        schema_version=4,
        model_id="production-gigaspeech-zipformer-fp32-benchmark-1t",
        adapter=SHERPA_ZIPFORMER_ADAPTER,
        python=SimpleNamespace(sha256="b" * 64),
        worker=SimpleNamespace(sha256="c" * 64),
        artifacts=artifacts,
        adapter_config=config,
    )
    source_bundle = SimpleNamespace(tree_sha256="d" * 64, files=(object(),))

    binding = streaming_stt_eval._worker_binding(
        manifest,
        source_bundle,
        runtime_receipt_sha256=None,
    )
    evidence = streaming_stt_eval._evidence_binding(
        SHERPA_ZIPFORMER_ADAPTER,
        pace="realtime",
        adapter_config=config,
    )

    assert binding["manifest_schema_version"] == 4
    assert binding["adapter_config"] == config.as_dict()
    assert "runtime_receipt_sha256" not in binding
    assert evidence["production_model"] is True
    assert evidence["real_candidate_model"] is False
    assert evidence["runtime_equivalent_to_production"] is False
    assert evidence["resource_control"] == {
        "benchmark_profile": "resource-controlled-one-thread",
        "benchmark_num_threads": 1,
        "production_device_profile": "desktop_gpu_4090",
        "production_num_threads": 4,
    }
    assert evidence["endpointing"] is False
    assert evidence["decode_contract"]["endpoint_rules"] == [2.4, 0.8, 20.0]
    assert streaming_stt_eval._default_stream(manifest) == StreamConfig(
        1600,
        "burst",
        200,
        0,
    )
    evaluator = streaming_stt_eval._evaluator_binding(SHERPA_ZIPFORMER_ADAPTER)
    assert evaluator["production_model"] is True
    assert evaluator["real_candidate_model"] is False
    assert evaluator["kind"] == "isolated_streaming_stt_production_harness"
    assert "tools/streaming_stt/adapters/zipformer.py" in evaluator["files"]


def test_version_probe_checks_installed_metadata_without_importing_model():
    provision_zipformer_baseline._verify_runtime_versions(  # noqa: SLF001
        Path(sys.executable),
        SherpaZipformerConfig(),
    )
    mismatch = SimpleNamespace(package_version="0.0.0", numpy_version="2.4.6")
    with pytest.raises(provision_zipformer_baseline.ProvisionError):
        provision_zipformer_baseline._verify_runtime_versions(  # noqa: SLF001
            Path(sys.executable),
            mismatch,
        )


@pytest.mark.real_model
def test_opt_in_provision_existing_production_model_without_download(tmp_path):
    model_root = os.environ.get("SPEAKER_ZIPFORMER_MODEL_ROOT")
    python = os.environ.get("SPEAKER_ZIPFORMER_PYTHON")
    if not model_root or not python:
        pytest.skip(
            "set SPEAKER_ZIPFORMER_MODEL_ROOT and SPEAKER_ZIPFORMER_PYTHON"
        )

    manifest = provision_zipformer_baseline.provision_baseline(
        python=Path(python),
        model_root=Path(model_root),
        output_dir=tmp_path / "zipformer-baseline",
    )

    assert manifest.schema_version == 4
    assert manifest.adapter == SHERPA_ZIPFORMER_ADAPTER
    assert manifest.adapter_config == SherpaZipformerConfig()
