from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace
from typing import Callable

import pytest

from tools import ami_endpoint_proxy_eval as evaluate
from tools import public_conversation_fixture as fixture
from tools.prepare_ami_conversation_fixture import (
    ENDPOINT_PROXY_CONTRACT,
    ENDPOINT_PROXY_LABELS_FILENAME,
    ENDPOINT_PROXY_LABELS_KIND,
    SOURCE_ID,
)
from tools.streaming_stt.ami_endpoint_proxy import (
    AMI_ENDPOINT_PROXY_KIND,
    AMI_ENDPOINT_PROXY_LABEL_SET_KIND,
    AMI_ENDPOINT_PROXY_SCHEMA_VERSION,
    AmiEndpointProxyBinding,
    AmiEndpointProxyLabel,
    canonical_label_set_sha256,
)
from tools.streaming_stt.corpus import CorpusProvenance, LoadedCorpus
from tools.streaming_stt.corpus_writer import (
    CorpusWriteCase,
    publish_private_corpus,
)
from tools.streaming_stt.manifest import PARAKEET_CPP_ADAPTER


_RECEIPT_SHA256 = "b" * 64
_ANNOTATION_SHA256 = "c" * 64
_SOURCE_MANIFEST_SHA256 = "d" * 64
_WORKER_MANIFEST_SHA256 = "e" * 64
_SAMPLES = 16
_PROXY_SAMPLE = 12
_AUDIO = bytes(_SAMPLES * 4)
_AUDIO_SHA256 = hashlib.sha256(_AUDIO).hexdigest()
_PRIVACY = {
    "contains_audio": False,
    "contains_transcripts": False,
    "contains_local_paths": False,
    "contains_raw_speaker_identifiers": False,
}
_EVIDENCE_SCOPE = {
    "after_pcm": True,
    "forced_alignment_proxy": True,
    "acoustic_ground_truth": False,
    "capture": False,
    "vad": False,
    "device": False,
    "live_latency": False,
    "default_promotion": False,
}


def _surface_rows() -> tuple[
    tuple[CorpusWriteCase, ...],
    tuple[AmiEndpointProxyLabel, ...],
]:
    cases: list[CorpusWriteCase] = []
    labels: list[AmiEndpointProxyLabel] = []
    for window in range(12):
        window_kind = "isolated" if window < 8 else "turn_transition"
        for channel in ("close", "far"):
            case_id = f"ami-es2004a-{window:02d}-{channel}"
            cases.append(
                CorpusWriteCase(
                    case_id=case_id,
                    audio_bytes=_AUDIO,
                    reference=f"confidential reference {window} {channel}",
                    tags=(window_kind, channel),
                )
            )
            labels.append(
                AmiEndpointProxyLabel(
                    case_id=case_id,
                    pcm_sha256=_AUDIO_SHA256,
                    samples=_SAMPLES,
                    forced_aligned_last_word_end_sample=_PROXY_SAMPLE,
                    window_kind=window_kind,
                    channel=channel,
                )
            )
    return tuple(cases), tuple(labels)


def _label_rows(
    labels: tuple[AmiEndpointProxyLabel, ...],
) -> list[dict[str, object]]:
    return [
        {
            "case_id": label.case_id,
            "pcm_sha256": label.pcm_sha256,
            "samples": label.samples,
            "forced_aligned_last_word_end_sample": (
                label.forced_aligned_last_word_end_sample
            ),
            "window_kind": label.window_kind,
            "channel": label.channel,
        }
        for label in labels
    ]


def _label_set_sha256(rows: list[dict[str, object]]) -> str:
    payload = json.dumps(
        {
            "kind": AMI_ENDPOINT_PROXY_LABEL_SET_KIND,
            "labels": rows,
        },
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _sidecar_root() -> dict[str, object]:
    _cases, labels = _surface_rows()
    rows = _label_rows(labels)
    assert _label_set_sha256(rows) == canonical_label_set_sha256(labels)
    return {
        "schema_version": 1,
        "kind": ENDPOINT_PROXY_LABELS_KIND,
        "fixture_id": fixture.FIXTURE_ID,
        "source_id": SOURCE_ID,
        "corpus_suite": "conversation96-ami",
        "sample_rate_hz": 16_000,
        "label_contract": ENDPOINT_PROXY_CONTRACT,
        "scoreable_scope": "isolated_nonoverlap_only",
        "annotation_metadata_sha256": _ANNOTATION_SHA256,
        "label_set_sha256": _label_set_sha256(rows),
        "labels": rows,
        "privacy": dict(_PRIVACY),
        "evidence_scope": dict(_EVIDENCE_SCOPE),
    }


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )


def _publish_surface(
    root: Path,
    *,
    mutate: Callable[[dict[str, object]], None] | None = None,
    sidecar_raw: bytes | None = None,
    metadata_sha256: str | None = None,
) -> LoadedCorpus:
    cases, _labels = _surface_rows()
    sidecar = _sidecar_root()
    if mutate is not None:
        mutate(sidecar)
    raw = _json_bytes(sidecar) if sidecar_raw is None else sidecar_raw
    return publish_private_corpus(
        cases=cases,
        provenance=CorpusProvenance(
            kind="public-voice-v1",
            suite="conversation96-ami",
            manifest_sha256=_SOURCE_MANIFEST_SHA256,
            metadata_sha256=(
                hashlib.sha256(raw).hexdigest()
                if metadata_sha256 is None
                else metadata_sha256
            ),
            source_set_sha256=_RECEIPT_SHA256,
        ),
        output_dir=root,
        purpose="synthetic private AMI endpoint proxy fixture",
        sidecars={ENDPOINT_PROXY_LABELS_FILENAME: raw},
    )


def _loaded_surface(root: Path) -> evaluate.LoadedAmiEndpointProxy:
    corpus = _publish_surface(root)
    return evaluate._load_sidecar(
        corpus,
        receipt_sha256=_RECEIPT_SHA256,
    )


def _proxy_evidence(binding: AmiEndpointProxyBinding) -> dict[str, object]:
    return {
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


def _accuracy(evaluations: int) -> dict[str, object]:
    return {
        "transcript": {
            "clips": evaluations,
            "nonempty": 0,
            "exact": 0,
            "wer": 0.0,
            "cer": 0.0,
            "word_errors": 0,
            "substitutions": 0,
            "insertions": 0,
            "deletions": 0,
            "ref_words": 0,
            "hyp_words": 0,
            "char_edits": 0,
            "ref_chars": 0,
            "hyp_chars": 0,
            "keyword_attempts": 0,
            "keyword_hits": 0,
        }
    }


def _endpointing(evaluations: int) -> dict[str, object]:
    samples = evaluations * _SAMPLES
    return {
        "evaluations": evaluations,
        "accepted_endpoints": 0,
        "tail_exhaustions": evaluations,
        "terminal_reasons": {
            "eou": 0,
            "tail_exhausted": evaluations,
        },
        "terminal_origins": {"feed": 0, "none": evaluations},
        "observations": {
            "total": 0,
            "eou": 0,
            "eob": 0,
            "origins": {"feed": 0, "finalize": 0},
            "source_early": {"total": 0, "eou": 0, "eob": 0},
            "post_source_feed": {"total": 0, "eou": 0, "eob": 0},
        },
        "source_samples": {
            "offered_total": samples,
            "consumed_total": samples,
            "dropped_total": 0,
        },
        "tail_samples": {
            "offered_total": 0,
            "consumed_total": 0,
            "unconsumed_total": 0,
        },
        "terminal_model_padding_samples_total": 0,
        "model_input_samples_consumed_total": samples,
        "accepted_endpoint_sample_p50": None,
        "accepted_endpoint_sample_p95": None,
        "observed_encoder_frame_p50": None,
        "observed_encoder_frame_p95": None,
        "observed_encoder_time_p50_seconds": None,
        "observed_encoder_time_p95_seconds": None,
        "accepted_endpoint_latency_p50_ms": None,
        "accepted_endpoint_latency_p95_ms": None,
        "accepted_endpoint_latency_max_ms": None,
    }


def _base_metrics(evaluations: int) -> dict[str, object]:
    return {
        "evaluations": evaluations,
        "coverage_complete": True,
        "accuracy": _accuracy(evaluations),
        "endpointing": _endpointing(evaluations),
    }


def _proxy_metrics(
    binding: AmiEndpointProxyBinding,
    *,
    repeats: int,
) -> dict[str, object]:
    binding_row = {
        "corpus_manifest_sha256": binding.corpus_manifest_sha256,
        "receipt_sha256": binding.receipt_sha256,
        "sidecar_sha256": binding.sidecar_sha256,
        "annotation_metadata_sha256": binding.annotation_metadata_sha256,
        "label_set_sha256": binding.label_set_sha256,
    }
    empty = {"p50": None, "p95": None, "max": None}
    scored = 16 * repeats
    return {
        "schema_version": AMI_ENDPOINT_PROXY_SCHEMA_VERSION,
        "kind": AMI_ENDPOINT_PROXY_KIND,
        "ground_truth": False,
        "binding": binding_row,
        "scope": "isolated_nonoverlap_diagnostic",
        "sample_domain": {
            "adapter": PARAKEET_CPP_ADAPTER,
            "sample_rate_hz": 16_000,
            "endpoint_field": "observed_sample",
            "definition": "source_pcm_plus_fed_tail_samples",
            "selection": "first_feed_eou",
        },
        "denominators": {
            "corpus_cases": 24,
            "isolated_cases": 16,
            "excluded_turn_transition_cases": 8,
            "repeats": repeats,
            "all_evaluations": 24 * repeats,
            "scored_evaluations": scored,
        },
        "endpoint_observations": {
            "feed_eou_observations": 0,
            "non_feed_eou_observations": 0,
            "evaluations_with_non_feed_eou": 0,
            "missing_eou_evaluations": scored,
            "missing_feed_eou_evaluations": scored,
            "non_feed_only_eou_evaluations": 0,
            "tail_exhaustion_evaluations": scored,
            "premature_feed_eou_evaluations": 0,
            "at_or_after_proxy_feed_eou_evaluations": 0,
        },
        "sample_deltas": {
            "signed": dict(empty),
            "early": dict(empty),
            "post_proxy": dict(empty),
            "absolute": dict(empty),
        },
        "milliseconds": {
            "signed": dict(empty),
            "early": dict(empty),
            "post_proxy": dict(empty),
            "absolute": dict(empty),
        },
    }


def _nested_report(
    binding: AmiEndpointProxyBinding,
    *,
    repeats: int = 3,
    chunk_samples: int = 1_280,
    pace: str = "burst",
    partial_interval_ms: int = 80,
    tail_padding_samples: int = 1_280,
) -> dict[str, object]:
    config_without_contract = {
        "chunk_samples": chunk_samples,
        "pace": pace,
        "partial_interval_ms": partial_interval_ms,
        "tail_padding_samples": tail_padding_samples,
        "repeats": repeats,
        "stratum_tags": ["close", "far", "isolated"],
        "ami_endpoint_proxy_label_set_sha256": binding.label_set_sha256,
        "ami_endpoint_proxy_sidecar_sha256": binding.sidecar_sha256,
    }
    metrics = _base_metrics(24 * repeats)
    metrics["strata"] = [
        {
            "tag": tag,
            "cases": cases,
            "metrics": _base_metrics(cases * repeats),
        }
        for tag, cases in (("close", 12), ("far", 12), ("isolated", 16))
    ]
    metrics["ami_endpoint_proxy"] = _proxy_metrics(binding, repeats=repeats)
    return {
        "ok": True,
        "evidence": {"ami_endpoint_proxy": _proxy_evidence(binding)},
        "worker": {
            "adapter": PARAKEET_CPP_ADAPTER,
            "manifest_sha256": _WORKER_MANIFEST_SHA256,
        },
        "corpus": {
            "manifest_sha256": binding.corpus_manifest_sha256,
            "ami_endpoint_proxy": {
                "receipt_sha256": binding.receipt_sha256,
                "sidecar_sha256": binding.sidecar_sha256,
                "annotation_metadata_sha256": (
                    binding.annotation_metadata_sha256
                ),
                "label_set_sha256": binding.label_set_sha256,
            },
        },
        "config": {
            **config_without_contract,
            "contract_sha256": evaluate._canonical_digest(
                config_without_contract
            ),
        },
        "metrics": metrics,
        "worker_stderr": {"bytes": 0, "sha256": "f" * 64},
        "evaluator": evaluate.streaming_stt_eval._evaluator_binding(
            PARAKEET_CPP_ADAPTER
        ),
    }


def _assert_detail_free(error: pytest.ExceptionInfo[BaseException]) -> None:
    assert error.value.args == ()
    assert str(error.value) == ""


@pytest.fixture(autouse=True)
def _fixed_worker_manifest(monkeypatch):
    monkeypatch.setattr(
        evaluate,
        "load_worker_manifest",
        lambda _path: SimpleNamespace(
            adapter=PARAKEET_CPP_ADAPTER,
            digest=_WORKER_MANIFEST_SHA256,
        ),
    )


def test_private_sidecar_loads_exact_surface_without_repr_disclosure(tmp_path):
    loaded = _loaded_surface(tmp_path / "private-corpus")

    assert loaded.binding.corpus_manifest_sha256 == loaded.corpus.digest
    assert loaded.binding.receipt_sha256 == _RECEIPT_SHA256
    assert loaded.binding.label_set_sha256 == canonical_label_set_sha256(
        loaded.binding.labels
    )
    assert loaded.sidecar_sha256 == loaded.corpus.provenance.metadata_sha256
    assert loaded.annotation_metadata_sha256 == _ANNOTATION_SHA256
    assert loaded.sidecar_path.name == ENDPOINT_PROXY_LABELS_FILENAME
    assert len(loaded.binding.labels) == 24

    rendered = repr(loaded)
    assert str(tmp_path) not in rendered
    assert "confidential reference" not in rendered
    assert loaded.binding.labels[0].case_id not in rendered


def test_public_loader_preserves_official_fixture_validation(
    tmp_path,
    monkeypatch,
):
    corpus = _publish_surface(tmp_path / "private-corpus")
    source = SimpleNamespace(
        corpus_path=corpus.path,
        corpus_manifest_sha256=corpus.digest,
        receipt_path=corpus.path.parent / "preparation-receipt.json",
        receipt_sha256=_RECEIPT_SHA256,
    )
    lock = object()
    calls: list[tuple[object, object, str]] = []
    monkeypatch.setattr(evaluate.fixture, "load_fixture_lock", lambda path: lock)

    def validate(path, *, lock, expected_source_id):
        calls.append((path, lock, expected_source_id))
        return source

    monkeypatch.setattr(evaluate.fixture, "validate_private_source", validate)
    monkeypatch.setattr(evaluate, "load_corpus", lambda path: corpus)

    loaded = evaluate.load_ami_endpoint_proxy(
        corpus.path,
        lock_path=tmp_path / "private-lock.json",
    )

    assert loaded.corpus is corpus
    assert calls == [(corpus.path, lock, SOURCE_ID)]


def _extra_root(root: dict[str, object]) -> None:
    root["unexpected"] = False


def _missing_row_field(root: dict[str, object]) -> None:
    labels = root["labels"]
    assert isinstance(labels, list)
    del labels[0]["channel"]


def _boolean_sample(root: dict[str, object]) -> None:
    labels = root["labels"]
    assert isinstance(labels, list)
    labels[0]["samples"] = True


def _wrong_contract(root: dict[str, object]) -> None:
    root["label_contract"] = "private-other-contract"


def _wrong_privacy(root: dict[str, object]) -> None:
    privacy = root["privacy"]
    assert isinstance(privacy, dict)
    privacy["contains_transcripts"] = True


def _boolean_schema(root: dict[str, object]) -> None:
    root["schema_version"] = True


def _numeric_privacy(root: dict[str, object]) -> None:
    privacy = root["privacy"]
    assert isinstance(privacy, dict)
    privacy["contains_audio"] = 0


def _numeric_evidence(root: dict[str, object]) -> None:
    evidence = root["evidence_scope"]
    assert isinstance(evidence, dict)
    evidence["acoustic_ground_truth"] = 0


def _reordered_bound_rows(root: dict[str, object]) -> None:
    labels = root["labels"]
    assert isinstance(labels, list)
    labels[0], labels[1] = labels[1], labels[0]
    root["label_set_sha256"] = _label_set_sha256(labels)


@pytest.mark.parametrize(
    "mutate",
    (
        _extra_root,
        _missing_row_field,
        _boolean_sample,
        _wrong_contract,
        _wrong_privacy,
        _boolean_schema,
        _numeric_privacy,
        _numeric_evidence,
        _reordered_bound_rows,
    ),
    ids=(
        "extra-root-field",
        "missing-row-field",
        "boolean-sample",
        "wrong-contract",
        "privacy-claim",
        "boolean-schema",
        "numeric-privacy",
        "numeric-evidence",
        "reordered-bound-rows",
    ),
)
def test_private_sidecar_rejects_semantic_tamper_detail_free(
    tmp_path,
    mutate,
):
    corpus = _publish_surface(tmp_path / mutate.__name__, mutate=mutate)

    with pytest.raises(evaluate.AmiEndpointProxyEvaluationError) as error:
        evaluate._load_sidecar(corpus, receipt_sha256=_RECEIPT_SHA256)

    _assert_detail_free(error)


def test_private_sidecar_rejects_duplicate_json_keys_detail_free(tmp_path):
    valid = _json_bytes(_sidecar_root())
    duplicate = b'{"schema_version":1,' + valid[1:]
    corpus = _publish_surface(
        tmp_path / "duplicate-key",
        sidecar_raw=duplicate,
    )

    with pytest.raises(evaluate.AmiEndpointProxyEvaluationError) as error:
        evaluate._load_sidecar(corpus, receipt_sha256=_RECEIPT_SHA256)

    _assert_detail_free(error)


def test_private_sidecar_rejects_unbound_byte_tamper_detail_free(tmp_path):
    corpus = _publish_surface(tmp_path / "byte-tamper")
    sidecar = corpus.path.parent / ENDPOINT_PROXY_LABELS_FILENAME
    sidecar.write_bytes(sidecar.read_bytes() + b" ")

    with pytest.raises(evaluate.AmiEndpointProxyEvaluationError) as error:
        evaluate._load_sidecar(corpus, receipt_sha256=_RECEIPT_SHA256)

    _assert_detail_free(error)


@pytest.mark.parametrize("mutation", ("mode", "hardlink", "symlink", "parent"))
def test_private_sidecar_rejects_unsafe_filesystem_state_detail_free(
    tmp_path,
    mutation,
):
    corpus = _publish_surface(tmp_path / mutation)
    sidecar = corpus.path.parent / ENDPOINT_PROXY_LABELS_FILENAME
    if mutation == "mode":
        sidecar.chmod(0o640)
    elif mutation == "hardlink":
        os.link(sidecar, corpus.path.parent / "second-label-name.json")
    elif mutation == "symlink":
        held = corpus.path.parent / "held-labels.json"
        sidecar.rename(held)
        sidecar.symlink_to(held.name)
    else:
        corpus.path.parent.chmod(0o750)

    with pytest.raises(evaluate.AmiEndpointProxyEvaluationError) as error:
        evaluate._load_sidecar(corpus, receipt_sha256=_RECEIPT_SHA256)

    _assert_detail_free(error)


def test_private_sidecar_rejects_mode_change_during_read_detail_free(
    tmp_path,
    monkeypatch,
):
    corpus = _publish_surface(tmp_path / "mode-race")
    sidecar = corpus.path.parent / ENDPOINT_PROXY_LABELS_FILENAME
    real_read = evaluate.os.read
    raced = False

    def race(descriptor, size):
        nonlocal raced
        if not raced:
            raced = True
            sidecar.chmod(0o640)
            corpus.path.parent.chmod(0o750)
            return real_read(descriptor, size)
        return real_read(descriptor, size)

    monkeypatch.setattr(evaluate.os, "read", race)
    with pytest.raises(evaluate.AmiEndpointProxyEvaluationError) as error:
        evaluate._load_sidecar(corpus, receipt_sha256=_RECEIPT_SHA256)

    assert raced is True
    _assert_detail_free(error)


def test_wrapper_revalidates_both_sides_and_binds_nested_report(
    tmp_path,
    monkeypatch,
):
    loaded = _loaded_surface(tmp_path / "private-corpus")
    load_calls: list[tuple[object, object]] = []
    runner_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def load(corpus_path, *, lock_path):
        load_calls.append((corpus_path, lock_path))
        return loaded

    def runner(*args, **kwargs):
        runner_calls.append((args, kwargs))
        return _nested_report(
            loaded.binding,
            repeats=2,
            chunk_samples=1_600,
            pace="realtime",
            partial_interval_ms=200,
            tail_padding_samples=800,
        )

    monkeypatch.setattr(evaluate, "load_ami_endpoint_proxy", load)
    monkeypatch.setattr(evaluate, "_PRODUCTION_RUNNER", runner)
    result = evaluate.run_ami_endpoint_proxy_evaluation(
        worker_manifest=tmp_path / "private-worker.json",
        corpus_path=loaded.corpus.path,
        scratch_parent=tmp_path / "private-scratch",
        lock_path=tmp_path / "private-lock.json",
        repeats=2,
        chunk_samples=1_600,
        pace="realtime",
        partial_interval_ms=200,
        tail_padding_samples=800,
    )

    assert len(load_calls) == 2
    assert len(runner_calls) == 1
    args, kwargs = runner_calls[0]
    assert args == (tmp_path / "private-worker.json", loaded.corpus.path)
    assert kwargs == {
        "scratch_parent": tmp_path / "private-scratch",
        "repeats": 2,
        "chunk_samples": 1_600,
        "pace": "realtime",
        "partial_interval_ms": 200,
        "tail_padding_samples": 800,
        "stratum_tags": ("isolated", "close", "far"),
        "ami_endpoint_proxy": loaded.binding,
    }
    assert result["ok"] is True
    assert result["execution_complete"] is True
    assert result["quality_decision"] == "not_evaluated"
    assert result["promotional"] is False
    assert result["binding"]["label_set_sha256"] == (
        loaded.binding.label_set_sha256
    )
    assert str(tmp_path) not in json.dumps(result, sort_keys=True)


def test_wrapper_runs_post_validation_when_runner_fails_detail_free(
    tmp_path,
    monkeypatch,
):
    loaded = _loaded_surface(tmp_path / "private-corpus")
    load_calls = 0

    def load(_corpus_path, *, lock_path):
        nonlocal load_calls
        del lock_path
        load_calls += 1
        return loaded

    def fail(*_args, **_kwargs):
        raise RuntimeError(f"private runner detail: {tmp_path}")

    monkeypatch.setattr(evaluate, "load_ami_endpoint_proxy", load)
    monkeypatch.setattr(evaluate, "_PRODUCTION_RUNNER", fail)
    with pytest.raises(evaluate.AmiEndpointProxyEvaluationError) as error:
        evaluate.run_ami_endpoint_proxy_evaluation(
            worker_manifest=tmp_path / "private-worker.json",
            corpus_path=loaded.corpus.path,
            scratch_parent=tmp_path / "private-scratch",
        )

    assert load_calls == 2
    _assert_detail_free(error)


@pytest.mark.parametrize("raised", (SystemExit("private"), KeyboardInterrupt()))
def test_wrapper_revalidates_after_base_exception_detail_free(
    tmp_path,
    monkeypatch,
    raised,
):
    loaded = _loaded_surface(tmp_path / type(raised).__name__)
    load_calls = 0

    def load(_corpus_path, *, lock_path):
        nonlocal load_calls
        del lock_path
        load_calls += 1
        return loaded

    def fail(*_args, **_kwargs):
        raise raised

    monkeypatch.setattr(evaluate, "load_ami_endpoint_proxy", load)
    monkeypatch.setattr(evaluate, "_PRODUCTION_RUNNER", fail)
    with pytest.raises(evaluate.AmiEndpointProxyEvaluationError) as error:
        evaluate.run_ami_endpoint_proxy_evaluation(
            worker_manifest=tmp_path / "private-worker.json",
            corpus_path=loaded.corpus.path,
            scratch_parent=tmp_path / "private-scratch",
        )

    assert load_calls == 2
    _assert_detail_free(error)


def test_wrapper_rejects_post_run_binding_change(tmp_path, monkeypatch):
    before = _loaded_surface(tmp_path / "private-corpus")
    after = replace(
        before,
        binding=replace(before.binding, receipt_sha256="9" * 64),
    )
    loaded = iter((before, after))
    monkeypatch.setattr(
        evaluate,
        "load_ami_endpoint_proxy",
        lambda *_args, **_kwargs: next(loaded),
    )
    monkeypatch.setattr(
        evaluate,
        "_PRODUCTION_RUNNER",
        lambda *_args, **_kwargs: _nested_report(before.binding),
    )

    with pytest.raises(evaluate.AmiEndpointProxyEvaluationError) as error:
        evaluate.run_ami_endpoint_proxy_evaluation(
            worker_manifest=tmp_path / "private-worker.json",
            corpus_path=before.corpus.path,
            scratch_parent=tmp_path / "private-scratch",
        )

    _assert_detail_free(error)


def test_wrapper_prevalidation_failure_prevents_runner(tmp_path, monkeypatch):
    runner_called = False

    def fail_load(*_args, **_kwargs):
        raise evaluate.AmiEndpointProxyEvaluationError()

    def runner(*_args, **_kwargs):
        nonlocal runner_called
        runner_called = True
        return {}

    monkeypatch.setattr(evaluate, "load_ami_endpoint_proxy", fail_load)
    monkeypatch.setattr(evaluate, "_PRODUCTION_RUNNER", runner)
    with pytest.raises(evaluate.AmiEndpointProxyEvaluationError) as error:
        evaluate.run_ami_endpoint_proxy_evaluation(
            worker_manifest=tmp_path / "private-worker.json",
            corpus_path=tmp_path / "private-corpus.json",
            scratch_parent=tmp_path / "private-scratch",
        )

    assert runner_called is False
    _assert_detail_free(error)


def _stale_corpus(report: dict[str, object]) -> None:
    report["corpus"]["manifest_sha256"] = "0" * 64


def _stale_metric_binding(report: dict[str, object]) -> None:
    report["metrics"]["ami_endpoint_proxy"]["binding"][
        "label_set_sha256"
    ] = "0" * 64


def _stale_evidence(report: dict[str, object]) -> None:
    report["evidence"]["ami_endpoint_proxy"]["ground_truth"] = True


def _forged_proxy_extra(report: dict[str, object]) -> None:
    report["metrics"]["ami_endpoint_proxy"]["promotional"] = True


def _forged_proxy_counter(report: dict[str, object]) -> None:
    report["metrics"]["ami_endpoint_proxy"]["endpoint_observations"][
        "missing_feed_eou_evaluations"
    ] = 0


def _forged_proxy_partition(report: dict[str, object]) -> None:
    report["metrics"]["ami_endpoint_proxy"]["endpoint_observations"][
        "missing_eou_evaluations"
    ] -= 1


def _fractional_proxy_samples(report: dict[str, object]) -> None:
    proxy = report["metrics"]["ami_endpoint_proxy"]
    observations = proxy["endpoint_observations"]
    observations["feed_eou_observations"] = 1
    observations["missing_eou_evaluations"] -= 1
    observations["missing_feed_eou_evaluations"] -= 1
    observations["premature_feed_eou_evaluations"] = 1
    for name, value in (("signed", -1.5), ("early", 1.5), ("absolute", 1.5)):
        proxy["sample_deltas"][name] = {
            "p50": value,
            "p95": value,
            "max": value,
        }
        milliseconds = round(value * 1000.0 / 16_000, 3)
        proxy["milliseconds"][name] = {
            "p50": milliseconds,
            "p95": milliseconds,
            "max": milliseconds,
        }


def _forged_non_feed_only_counter(report: dict[str, object]) -> None:
    observations = report["metrics"]["ami_endpoint_proxy"][
        "endpoint_observations"
    ]
    observations["missing_eou_evaluations"] -= 1
    observations["non_feed_only_eou_evaluations"] = 1


def _forged_non_feed_evaluation_counter(report: dict[str, object]) -> None:
    observations = report["metrics"]["ami_endpoint_proxy"][
        "endpoint_observations"
    ]
    observations["non_feed_eou_observations"] = 1
    observations["evaluations_with_non_feed_eou"] = 1


@pytest.mark.parametrize(
    "mutate",
    (
        _stale_corpus,
        _stale_metric_binding,
        _stale_evidence,
        _forged_proxy_extra,
        _forged_proxy_counter,
        _forged_proxy_partition,
        _fractional_proxy_samples,
        _forged_non_feed_only_counter,
        _forged_non_feed_evaluation_counter,
    ),
    ids=(
        "corpus",
        "metric-binding",
        "evidence",
        "proxy-extra",
        "counter",
        "partition",
        "fractional-samples",
        "non-feed-only-counter",
        "non-feed-evaluation-counter",
    ),
)
def test_wrapper_rejects_stale_or_forged_nested_report(
    tmp_path,
    monkeypatch,
    mutate,
):
    loaded = _loaded_surface(tmp_path / mutate.__name__)
    monkeypatch.setattr(
        evaluate,
        "load_ami_endpoint_proxy",
        lambda *_args, **_kwargs: loaded,
    )
    nested = deepcopy(_nested_report(loaded.binding))
    mutate(nested)
    monkeypatch.setattr(
        evaluate,
        "_PRODUCTION_RUNNER",
        lambda *_args, **_kwargs: nested,
    )

    with pytest.raises(evaluate.AmiEndpointProxyEvaluationError) as error:
        evaluate.run_ami_endpoint_proxy_evaluation(
            worker_manifest=tmp_path / "private-worker.json",
            corpus_path=loaded.corpus.path,
            scratch_parent=tmp_path / "private-scratch",
        )

    _assert_detail_free(error)


def test_wrapper_reconstructs_output_without_untrusted_nested_extras(
    tmp_path,
    monkeypatch,
):
    loaded = _loaded_surface(tmp_path / "private-corpus")
    monkeypatch.setattr(
        evaluate,
        "load_ami_endpoint_proxy",
        lambda *_args, **_kwargs: loaded,
    )
    nested = _nested_report(loaded.binding)
    nested["worker"]["verbatim_transcript"] = "confidential spoken words"
    nested["worker"]["relative_path"] = "private/source.f32le"
    nested["metrics"]["raw_labels"] = [
        {"case_id": loaded.binding.labels[0].case_id}
    ]
    nested["worker_stderr"]["speaker_id"] = "private-speaker"
    monkeypatch.setattr(
        evaluate,
        "_PRODUCTION_RUNNER",
        lambda *_args, **_kwargs: nested,
    )

    result = evaluate.run_ami_endpoint_proxy_evaluation(
        worker_manifest=tmp_path / "private-worker.json",
        corpus_path=loaded.corpus.path,
        scratch_parent=tmp_path / "private-scratch",
    )

    encoded = json.dumps(result, sort_keys=True)
    assert "confidential spoken words" not in encoded
    assert "private/source.f32le" not in encoded
    assert loaded.binding.labels[0].case_id not in encoded
    assert "private-speaker" not in encoded
    assert "raw_labels" not in encoded


def test_private_report_writer_is_mode_600_and_no_clobber(tmp_path):
    private_parent = tmp_path / "private-report-parent"
    private_parent.mkdir(mode=0o700)
    private_parent.chmod(0o700)
    output = private_parent / "report.json"
    value = {"ok": True, "report_sha256": "a" * 64}

    digest = evaluate._write_new_private(output, value)

    expected = _json_bytes(value)
    assert output.read_bytes() == expected
    assert digest == hashlib.sha256(expected).hexdigest()
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    with pytest.raises(evaluate.AmiEndpointProxyEvaluationError) as error:
        evaluate._write_new_private(output, {"ok": False})
    _assert_detail_free(error)


def test_private_report_writer_rejects_nonprivate_parent_detail_free(tmp_path):
    public_parent = tmp_path / "public-report-parent"
    public_parent.mkdir(mode=0o755)
    public_parent.chmod(0o755)

    with pytest.raises(evaluate.AmiEndpointProxyEvaluationError) as error:
        evaluate._write_new_private(public_parent / "report.json", {"ok": True})

    _assert_detail_free(error)
    assert not (public_parent / "report.json").exists()


def test_private_report_writer_rejects_mode_change_during_read_detail_free(
    tmp_path,
    monkeypatch,
):
    private_parent = tmp_path / "private-report-parent"
    private_parent.mkdir(mode=0o700)
    private_parent.chmod(0o700)
    output = private_parent / "report.json"
    real_read = evaluate.os.read
    raced = False

    def race(descriptor, size):
        nonlocal raced
        if not raced:
            raced = True
            output.chmod(0o640)
            private_parent.chmod(0o750)
            return real_read(descriptor, size)
        return real_read(descriptor, size)

    monkeypatch.setattr(evaluate.os, "read", race)
    with pytest.raises(evaluate.AmiEndpointProxyEvaluationError) as error:
        evaluate._write_new_private(output, {"ok": True})

    assert raced is True
    _assert_detail_free(error)


def test_cli_failure_is_detail_free(tmp_path, monkeypatch, capsys):
    private_detail = f"private corpus failed at {tmp_path / 'source.f32le'}"

    def fail(**_kwargs):
        raise RuntimeError(private_detail)

    monkeypatch.setattr(evaluate, "run_ami_endpoint_proxy_evaluation", fail)
    code = evaluate.main(
        [
            "--worker-manifest",
            str(tmp_path / "worker.json"),
            "--corpus",
            str(tmp_path / "corpus.json"),
            "--scratch-root",
            str(tmp_path / "scratch"),
            "--output",
            str(tmp_path / "report.json"),
        ]
    )

    captured = capsys.readouterr()
    assert code == 2
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "error": "ami_endpoint_proxy_evaluation_unavailable",
        "ok": False,
    }
    assert private_detail not in captured.out
    assert str(tmp_path) not in captured.out
