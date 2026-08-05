from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import streaming_stt_eval
from tools.streaming_stt.ami_endpoint_proxy import (
    AMI_ENDPOINT_PROXY_KIND,
    AMI_ENDPOINT_PROXY_SCHEMA_VERSION,
    AmiEndpointProxyBinding,
    AmiEndpointProxyLabel,
    canonical_label_set_sha256,
    canonical_sidecar_sha256,
    validate_ami_endpoint_proxy_binding,
)
from tools.streaming_stt.corpus import (
    CorpusCase,
    CorpusProvenance,
    LoadedCorpus,
)
from tools.streaming_stt.manifest import (
    PARAKEET_CPP_ADAPTER,
    PARAKEET_REALTIME_EOU_ADAPTER,
    ParakeetCppConfig,
)
from tools.streaming_stt.protocol import StreamConfig


_CORPUS_SHA256 = "a" * 64
_RECEIPT_SHA256 = "b" * 64
_SAMPLES = 16_000
_PROXY_SAMPLE = 15_360
_STREAM = StreamConfig(
    chunk_samples=1_280,
    pace="burst",
    partial_interval_ms=80,
    tail_padding_samples=1_280,
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _surface(
    tmp_path: Path,
) -> tuple[LoadedCorpus, AmiEndpointProxyBinding]:
    cases: list[CorpusCase] = []
    labels: list[AmiEndpointProxyLabel] = []
    for window in range(12):
        window_kind = "isolated" if window < 8 else "turn_transition"
        for channel in ("close", "far"):
            case_id = f"ami-es2004a-{window:02d}-{channel}"
            pcm_sha256 = hashlib.sha256(case_id.encode("ascii")).hexdigest()
            cases.append(
                CorpusCase(
                    case_id=case_id,
                    source_path=tmp_path / f"private-{case_id}.f32le",
                    sha256=pcm_sha256,
                    samples=_SAMPLES,
                    expected_text=f"private reference {window} {channel}",
                    assertion="transcript",
                    commands=(),
                    forbidden_commands=(),
                    tags=(window_kind, channel),
                    audio_bytes=b"",
                )
            )
            labels.append(
                AmiEndpointProxyLabel(
                    case_id=case_id,
                    pcm_sha256=pcm_sha256,
                    samples=_SAMPLES,
                    forced_aligned_last_word_end_sample=_PROXY_SAMPLE,
                    window_kind=window_kind,
                    channel=channel,
                )
            )
    label_tuple = tuple(labels)
    label_set_sha256 = canonical_label_set_sha256(label_tuple)
    annotation_metadata_sha256 = "d" * 64
    sidecar_sha256 = canonical_sidecar_sha256(
        annotation_metadata_sha256=annotation_metadata_sha256,
        label_set_sha256=label_set_sha256,
        labels=label_tuple,
    )
    corpus = LoadedCorpus(
        path=tmp_path / "private-corpus.json",
        digest=_CORPUS_SHA256,
        schema_version=2,
        purpose="private AMI endpoint proxy fixture",
        provenance=CorpusProvenance(
            kind="public-voice-v1",
            suite="conversation96-ami",
            manifest_sha256="c" * 64,
            metadata_sha256=sidecar_sha256,
            source_set_sha256=_RECEIPT_SHA256,
        ),
        cases=tuple(cases),
        audio_bytes=0,
    )
    binding = AmiEndpointProxyBinding(
        corpus_manifest_sha256=_CORPUS_SHA256,
        receipt_sha256=_RECEIPT_SHA256,
        sidecar_sha256=sidecar_sha256,
        annotation_metadata_sha256=annotation_metadata_sha256,
        label_set_sha256=label_set_sha256,
        labels=label_tuple,
    )
    return corpus, binding


def _manifest(tmp_path: Path, *, adapter: str = PARAKEET_CPP_ADAPTER):
    return SimpleNamespace(
        path=tmp_path / "private-worker-manifest.json",
        digest="e" * 64,
        adapter=adapter,
        adapter_config=(
            ParakeetCppConfig() if adapter == PARAKEET_CPP_ADAPTER else None
        ),
        worker=SimpleNamespace(path=streaming_stt_eval._FIXED_WORKER),
    )


def _install_fake_controller(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    corpus: LoadedCorpus,
    manifest: object,
    events: list[str],
) -> None:
    scratch_index = 0

    def prepare_scratch(_parent: Path) -> Path:
        nonlocal scratch_index
        scratch_index += 1
        path = tmp_path / f"scratch-{scratch_index}"
        path.mkdir(mode=0o700)
        return path

    def write_snapshot(path: Path, _payload: bytes) -> None:
        path.touch(mode=0o600, exist_ok=False)

    def snapshot_hash(path: Path, **_kwargs: object) -> object:
        index = int(path.stem.removeprefix("case-"))
        return SimpleNamespace(sha256=corpus.cases[index].sha256)

    class FakeWorker:
        def __init__(self, *_args: object) -> None:
            events.append("worker_init")
            self.ready = SimpleNamespace(
                runtime={"python": "3.12", "platform": "test"}
            )
            self.stderr_summary = {}
            self.cgroup_evidence = None
            self.resource_observations = None
            self.fully_stopped = False

        def __enter__(self) -> FakeWorker:
            return self

        def __exit__(self, *_args: object) -> None:
            self.fully_stopped = True

        def transcribe(self, _request: object) -> object:
            events.append("transcribe")
            return SimpleNamespace()

    def aggregate_metrics(*_args: object, **_kwargs: object) -> dict[str, object]:
        events.append("aggregate_metrics")
        return {"coverage_complete": True, "legacy_metric": 7}

    monkeypatch.setattr(
        streaming_stt_eval,
        "load_worker_manifest",
        lambda _path: manifest,
    )
    monkeypatch.setattr(
        streaming_stt_eval,
        "_verified_runtime_receipt_digest",
        lambda _manifest: "f" * 64,
    )
    monkeypatch.setattr(
        streaming_stt_eval,
        "_verified_model_receipt_digest",
        lambda _manifest: None,
    )
    monkeypatch.setattr(streaming_stt_eval, "load_corpus", lambda _path: corpus)
    monkeypatch.setattr(
        streaming_stt_eval,
        "validate_stratum_tags",
        lambda _corpus, tags: tuple(tags),
    )
    monkeypatch.setattr(
        streaming_stt_eval,
        "_evaluator_binding",
        lambda adapter: {"kind": "test-evaluator", "adapter": adapter},
    )
    monkeypatch.setattr(streaming_stt_eval, "_prepare_scratch", prepare_scratch)
    monkeypatch.setattr(
        streaming_stt_eval,
        "stage_worker_source_bundle",
        lambda *_args: SimpleNamespace(),
    )
    monkeypatch.setattr(
        streaming_stt_eval,
        "_worker_binding",
        lambda *_args, **_kwargs: {
            "adapter": PARAKEET_CPP_ADAPTER,
            "manifest_sha256": "e" * 64,
        },
    )
    monkeypatch.setattr(streaming_stt_eval, "_write_pcm_snapshot", write_snapshot)
    monkeypatch.setattr(
        streaming_stt_eval,
        "hash_regular_bounded",
        snapshot_hash,
    )
    monkeypatch.setattr(streaming_stt_eval, "StreamingWorker", FakeWorker)
    monkeypatch.setattr(
        streaming_stt_eval,
        "verify_corpus_snapshot",
        lambda _corpus: None,
    )
    monkeypatch.setattr(
        streaming_stt_eval,
        "verify_source_bundle",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(streaming_stt_eval, "aggregate_metrics", aggregate_metrics)
    monkeypatch.setattr(
        streaming_stt_eval,
        "_worker_report",
        lambda *_args, **_kwargs: {
            "adapter": PARAKEET_CPP_ADAPTER,
            "runtime": {"python": "3.12", "platform": "test"},
        },
    )
    monkeypatch.setattr(
        streaming_stt_eval,
        "_remove_private_scratch",
        lambda *_args, **_kwargs: events.append("scratch_removed"),
    )


def _run_locked(
    tmp_path: Path,
    *,
    ami_endpoint_proxy: AmiEndpointProxyBinding | None = None,
    pass_proxy_argument: bool = True,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "scratch_parent": tmp_path / "scratch-parent",
        "repeats": 1,
        "stream": _STREAM,
        "chunk_samples": None,
        "pace": None,
        "partial_interval_ms": None,
        "tail_padding_samples": None,
    }
    if pass_proxy_argument:
        kwargs["ami_endpoint_proxy"] = ami_endpoint_proxy
    return streaming_stt_eval._run_benchmark_locked(
        tmp_path / "private-worker-manifest.json",
        tmp_path / "private-corpus.json",
        **kwargs,
    )


def test_absent_proxy_keeps_legacy_bindings_and_report_shape_byte_compatible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, _binding = _surface(tmp_path)
    legacy_evidence = streaming_stt_eval._evidence_binding(
        "fake-json-v1",
        pace="burst",
    )
    explicit_none_evidence = streaming_stt_eval._evidence_binding(
        "fake-json-v1",
        pace="burst",
        ami_endpoint_proxy=None,
    )
    legacy_corpus = streaming_stt_eval._corpus_binding(corpus)
    explicit_none_corpus = streaming_stt_eval._corpus_binding(
        corpus,
        ami_endpoint_proxy=None,
    )

    assert _canonical_bytes(legacy_evidence) == _canonical_bytes(
        explicit_none_evidence
    )
    assert _canonical_bytes(legacy_corpus) == _canonical_bytes(
        explicit_none_corpus
    )
    assert "ami_endpoint_proxy" not in legacy_evidence
    assert "ami_endpoint_proxy" not in legacy_corpus

    events: list[str] = []
    manifest = _manifest(tmp_path)
    _install_fake_controller(monkeypatch, tmp_path, corpus, manifest, events)
    monkeypatch.setattr(
        streaming_stt_eval,
        "validate_ami_endpoint_proxy_binding",
        lambda *_args: pytest.fail("absent proxy reached proxy validator"),
    )
    monkeypatch.setattr(
        streaming_stt_eval,
        "aggregate_ami_endpoint_proxy",
        lambda *_args, **_kwargs: pytest.fail(
            "absent proxy reached proxy reducer"
        ),
    )

    omitted = _run_locked(tmp_path, pass_proxy_argument=False)
    explicit_none = _run_locked(tmp_path, ami_endpoint_proxy=None)

    assert _canonical_bytes(omitted) == _canonical_bytes(explicit_none)
    assert set(omitted) == {
        "ok",
        "evidence",
        "worker",
        "corpus",
        "config",
        "metrics",
        "worker_stderr",
        "evaluator",
    }
    assert omitted["config"] == {
        **_STREAM.as_dict(),
        "repeats": 1,
        "contract_sha256": streaming_stt_eval._canonical_digest(
            {**_STREAM.as_dict(), "repeats": 1}
        ),
    }
    assert "ami_endpoint_proxy" not in _canonical_bytes(omitted).decode("ascii")


def test_proxy_bindings_are_exact_non_authoritative_and_aggregate_only(
    tmp_path: Path,
) -> None:
    corpus, binding = _surface(tmp_path)

    evidence = streaming_stt_eval._evidence_binding(
        PARAKEET_CPP_ADAPTER,
        pace="burst",
        adapter_config=ParakeetCppConfig(),
        ami_endpoint_proxy=binding,
    )
    corpus_binding = streaming_stt_eval._corpus_binding(
        corpus,
        ami_endpoint_proxy=binding,
    )

    assert evidence["ami_endpoint_proxy"] == {
        "schema_version": AMI_ENDPOINT_PROXY_SCHEMA_VERSION,
        "kind": AMI_ENDPOINT_PROXY_KIND,
        "sidecar_sha256": binding.sidecar_sha256,
        "label_set_sha256": binding.label_set_sha256,
        "scoreable_scope": "isolated_nonoverlap_only",
        "ground_truth": False,
        "aggregate_only": True,
        "quality_thresholds": "none",
        "capture_included": False,
        "vad_included": False,
        "device_included": False,
        "live_latency": False,
        "promotional": False,
    }
    assert corpus_binding["ami_endpoint_proxy"] == {
        "receipt_sha256": binding.receipt_sha256,
        "sidecar_sha256": binding.sidecar_sha256,
        "annotation_metadata_sha256": binding.annotation_metadata_sha256,
        "label_set_sha256": binding.label_set_sha256,
    }
    encoded = _canonical_bytes(
        {"evidence": evidence, "corpus": corpus_binding}
    ).decode("ascii")
    for label in binding.labels:
        assert label.case_id not in encoded
        assert label.pcm_sha256 not in encoded


@pytest.mark.parametrize(
    "adapter",
    (PARAKEET_CPP_ADAPTER, PARAKEET_REALTIME_EOU_ADAPTER),
)
def test_proxy_adapter_gate_allows_both_endpoint_adapters(
    adapter: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _corpus, binding = _surface(tmp_path)
    manifest = _manifest(tmp_path, adapter=adapter)

    class ReachedStreamSelection(Exception):
        pass

    monkeypatch.setattr(
        streaming_stt_eval,
        "load_worker_manifest",
        lambda _path: manifest,
    )
    monkeypatch.setattr(
        streaming_stt_eval,
        "_selected_stream",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ReachedStreamSelection
        ),
    )

    with pytest.raises(ReachedStreamSelection):
        _run_locked(tmp_path, ami_endpoint_proxy=binding)


def test_proxy_adapter_gate_rejects_other_adapters_before_stream_or_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _corpus, binding = _surface(tmp_path)
    manifest = _manifest(tmp_path, adapter="fake-json-v1")
    later_calls: list[str] = []
    monkeypatch.setattr(
        streaming_stt_eval,
        "load_worker_manifest",
        lambda _path: manifest,
    )
    monkeypatch.setattr(
        streaming_stt_eval,
        "_selected_stream",
        lambda *_args, **_kwargs: later_calls.append("stream"),
    )
    monkeypatch.setattr(
        streaming_stt_eval,
        "StreamingWorker",
        lambda *_args: later_calls.append("worker"),
    )

    with pytest.raises(ValueError):
        _run_locked(tmp_path, ami_endpoint_proxy=binding)

    assert later_calls == []


def test_invalid_proxy_is_rejected_before_scratch_or_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, binding = _surface(tmp_path)
    manifest = _manifest(tmp_path)
    invalid = replace(binding, labels=binding.labels[:-1])
    later_calls: list[str] = []
    monkeypatch.setattr(
        streaming_stt_eval,
        "load_worker_manifest",
        lambda _path: manifest,
    )
    monkeypatch.setattr(
        streaming_stt_eval,
        "_verified_runtime_receipt_digest",
        lambda _manifest: "f" * 64,
    )
    monkeypatch.setattr(
        streaming_stt_eval,
        "_verified_model_receipt_digest",
        lambda _manifest: None,
    )
    monkeypatch.setattr(streaming_stt_eval, "load_corpus", lambda _path: corpus)
    monkeypatch.setattr(
        streaming_stt_eval,
        "validate_stratum_tags",
        lambda _corpus, tags: tuple(tags),
    )
    monkeypatch.setattr(
        streaming_stt_eval,
        "_prepare_scratch",
        lambda _path: later_calls.append("scratch"),
    )
    monkeypatch.setattr(
        streaming_stt_eval,
        "StreamingWorker",
        lambda *_args: later_calls.append("worker"),
    )

    with pytest.raises(ValueError):
        _run_locked(tmp_path, ami_endpoint_proxy=invalid)

    assert later_calls == []


def test_proxy_is_revalidated_after_worker_and_bound_into_aggregate_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, binding = _surface(tmp_path)
    manifest = _manifest(tmp_path)
    events: list[str] = []
    _install_fake_controller(monkeypatch, tmp_path, corpus, manifest, events)
    real_validate = validate_ami_endpoint_proxy_binding

    def validate(
        selected_corpus: LoadedCorpus,
        selected_binding: AmiEndpointProxyBinding,
    ) -> tuple[AmiEndpointProxyLabel, ...]:
        events.append("validate")
        return real_validate(selected_corpus, selected_binding)

    proxy_metrics = {
        "schema_version": 1,
        "kind": "ami-endpoint-proxy-aggregate-test",
        "ground_truth": False,
        "scoreable_evaluations": 16,
    }

    def aggregate_proxy(*args: object, **_kwargs: object) -> dict[str, object]:
        events.append("aggregate_proxy")
        assert args[0] is corpus
        assert args[1] is binding
        assert len(args[2]) == len(corpus.cases)
        return proxy_metrics

    monkeypatch.setattr(
        streaming_stt_eval,
        "validate_ami_endpoint_proxy_binding",
        validate,
    )
    monkeypatch.setattr(
        streaming_stt_eval,
        "aggregate_ami_endpoint_proxy",
        aggregate_proxy,
    )

    report = _run_locked(tmp_path, ami_endpoint_proxy=binding)

    worker_init = events.index("worker_init")
    last_transcribe = len(events) - 1 - events[::-1].index("transcribe")
    aggregate_metrics = events.index("aggregate_metrics")
    aggregate_proxy = events.index("aggregate_proxy")
    validation_positions = [
        index for index, event in enumerate(events) if event == "validate"
    ]
    assert len(validation_positions) == 3
    assert validation_positions[0] < validation_positions[1] < worker_init
    assert (
        last_transcribe
        < aggregate_metrics
        < validation_positions[2]
        < aggregate_proxy
    )
    assert report["metrics"] == {
        "coverage_complete": True,
        "legacy_metric": 7,
        "ami_endpoint_proxy": proxy_metrics,
    }
    assert report["corpus"]["ami_endpoint_proxy"] == {
        "receipt_sha256": binding.receipt_sha256,
        "sidecar_sha256": binding.sidecar_sha256,
        "annotation_metadata_sha256": binding.annotation_metadata_sha256,
        "label_set_sha256": binding.label_set_sha256,
    }
    assert report["evidence"]["ami_endpoint_proxy"]["ground_truth"] is False
    config_without_contract = {
        **_STREAM.as_dict(),
        "repeats": 1,
        "ami_endpoint_proxy_label_set_sha256": binding.label_set_sha256,
        "ami_endpoint_proxy_sidecar_sha256": binding.sidecar_sha256,
    }
    assert report["config"] == {
        **config_without_contract,
        "contract_sha256": streaming_stt_eval._canonical_digest(
            config_without_contract
        ),
    }
    encoded = _canonical_bytes(report).decode("ascii")
    for label in binding.labels:
        assert label.case_id not in encoded
        assert label.pcm_sha256 not in encoded


def test_post_run_proxy_validation_failure_blocks_proxy_aggregation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, binding = _surface(tmp_path)
    manifest = _manifest(tmp_path)
    events: list[str] = []
    _install_fake_controller(monkeypatch, tmp_path, corpus, manifest, events)
    validation_calls = 0

    def validate(
        selected_corpus: LoadedCorpus,
        selected_binding: AmiEndpointProxyBinding,
    ) -> tuple[AmiEndpointProxyLabel, ...]:
        nonlocal validation_calls
        validation_calls += 1
        if validation_calls == 3:
            raise ValueError
        return validate_ami_endpoint_proxy_binding(
            selected_corpus,
            selected_binding,
        )

    monkeypatch.setattr(
        streaming_stt_eval,
        "validate_ami_endpoint_proxy_binding",
        validate,
    )
    monkeypatch.setattr(
        streaming_stt_eval,
        "aggregate_ami_endpoint_proxy",
        lambda *_args, **_kwargs: events.append("aggregate_proxy"),
    )

    with pytest.raises(ValueError):
        _run_locked(tmp_path, ami_endpoint_proxy=binding)

    assert validation_calls == 3
    assert events.count("transcribe") == len(corpus.cases)
    assert "aggregate_metrics" in events
    assert "aggregate_proxy" not in events
    assert events[-1] == "scratch_removed"
