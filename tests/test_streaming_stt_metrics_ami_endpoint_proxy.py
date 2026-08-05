from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from tools import ami_endpoint_proxy_eval, streaming_stt_eval
from tools.streaming_stt.ami_endpoint_proxy import (
    AMI_ENDPOINT_PROXY_LABEL_SET_KIND,
    AmiEndpointProxyBinding,
    AmiEndpointProxyError,
    AmiEndpointProxyLabel,
    aggregate_ami_endpoint_proxy,
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
from tools.streaming_stt.metrics import RunRecord, aggregate_metrics
from tools.streaming_stt.protocol import (
    NATIVE_ENDPOINT_PROTOCOL_VERSION,
    PARAKEET_CPP_ENDPOINT_PROTOCOL_VERSION,
    CaseTrace,
    FinalEvent,
    NativeFinalEvent,
    ParakeetCppFinalEvent,
    ParakeetCppObservedEvent,
    ReadyEvent,
    ResourceUsage,
)


_SAMPLES = 16_000
_PROXY_SAMPLE = 15_360
_TAIL_SAMPLES = 1_280
_MANIFEST_SHA256 = "a" * 64
_RECEIPT_SHA256 = "b" * 64
_USAGE = ResourceUsage(rss_mb=10.0, threads=1, vram_mb=None)


def _surface(
    tmp_path: Path,
) -> tuple[LoadedCorpus, AmiEndpointProxyBinding]:
    cases: list[CorpusCase] = []
    labels: list[AmiEndpointProxyLabel] = []
    for window in range(12):
        kind = "isolated" if window < 8 else "turn_transition"
        for channel in ("close", "far"):
            case_id = f"ami-es2004a-{window:02d}-{channel}"
            pcm_sha256 = hashlib.sha256(case_id.encode("ascii")).hexdigest()
            cases.append(
                CorpusCase(
                    case_id=case_id,
                    source_path=tmp_path / f"private-{case_id}.f32le",
                    sha256=pcm_sha256,
                    samples=_SAMPLES,
                    expected_text=f"private transcript {window} {channel}",
                    assertion="transcript",
                    commands=(),
                    forbidden_commands=(),
                    tags=(kind, channel),
                    audio_bytes=b"",
                )
            )
            labels.append(
                AmiEndpointProxyLabel(
                    case_id=case_id,
                    pcm_sha256=pcm_sha256,
                    samples=_SAMPLES,
                    forced_aligned_last_word_end_sample=_PROXY_SAMPLE,
                    window_kind=kind,
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
        digest=_MANIFEST_SHA256,
        schema_version=2,
        purpose="private AMI fixture",
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
        corpus_manifest_sha256=_MANIFEST_SHA256,
        receipt_sha256=_RECEIPT_SHA256,
        sidecar_sha256=sidecar_sha256,
        annotation_metadata_sha256=annotation_metadata_sha256,
        label_set_sha256=label_set_sha256,
        labels=label_tuple,
    )
    return corpus, binding


def _ready(adapter: str) -> ReadyEvent:
    protocol_version = (
        NATIVE_ENDPOINT_PROTOCOL_VERSION
        if adapter == PARAKEET_REALTIME_EOU_ADAPTER
        else PARAKEET_CPP_ENDPOINT_PROTOCOL_VERSION
    )
    return ReadyEvent(
        model_id="private-model",
        manifest_sha256="e" * 64,
        source_bundle_sha256="f" * 64,
        adapter=adapter,
        model_load_ms=1.0,
        resources=_USAGE,
        runtime={"python": "3.12", "platform": "test"},
        protocol_version=protocol_version,
    )


def _native_final(
    *,
    endpoint_sample: int | None = None,
    reason: str = "tail_exhausted",
    model_padding_samples: int = 0,
) -> NativeFinalEvent:
    if endpoint_sample is None:
        source_consumed = _SAMPLES
        tail_consumed = _TAIL_SAMPLES
        samples_seen = _SAMPLES + _TAIL_SAMPLES
        native_endpoint = False
        probability = None
        authoritative = False
        latency = None
    else:
        samples_seen = endpoint_sample - model_padding_samples
        source_consumed = min(samples_seen, _SAMPLES)
        tail_consumed = max(0, samples_seen - _SAMPLES)
        native_endpoint = True
        probability = 0.75
        authoritative = reason == "eou" and source_consumed == _SAMPLES
        latency = tail_consumed / 16.0 if authoritative else None
    return NativeFinalEvent(
        request_id="private-request",
        seq=0,
        text="private hypothesis",
        samples_seen=samples_seen,
        elapsed_ms=10.0,
        finalization_ms=1.0,
        compute_ms=1.0,
        audio_seconds=1.0,
        chunks=1,
        deadline_misses=0,
        max_backlog_ms=0.0,
        resources=_USAGE,
        model_padding_samples=model_padding_samples,
        source_samples=_SAMPLES,
        source_samples_consumed=source_consumed,
        declared_tail_samples=_TAIL_SAMPLES,
        tail_samples_consumed=tail_consumed,
        endpoint_reason=reason,
        native_endpoint=native_endpoint,
        endpoint_probability=probability,
        endpoint_sample=endpoint_sample,
        endpoint_latency_ms=latency,
        authoritative=authoritative,
    )


def _observed(
    origin: str,
    sample: int,
    frame: int,
) -> ParakeetCppObservedEvent:
    return ParakeetCppObservedEvent(
        event_type="eou",
        event_origin=origin,
        observed_sample=sample,
        encoder_frame=frame,
        encoder_time_seconds=frame * 0.08,
    )


def _cpp_final(
    events: tuple[ParakeetCppObservedEvent, ...] = (),
) -> ParakeetCppFinalEvent:
    return ParakeetCppFinalEvent(
        request_id="private-request",
        seq=0,
        text="private hypothesis",
        samples_seen=_SAMPLES + _TAIL_SAMPLES,
        elapsed_ms=10.0,
        finalization_ms=1.0,
        compute_ms=1.0,
        audio_seconds=1.0,
        chunks=14,
        deadline_misses=0,
        max_backlog_ms=0.0,
        resources=_USAGE,
        model_padding_samples=0,
        source_samples_offered=_SAMPLES,
        source_samples_consumed=_SAMPLES,
        tail_samples_offered=_TAIL_SAMPLES,
        tail_samples_consumed=_TAIL_SAMPLES,
        observed_events=events,
        endpoint_reason="tail_exhausted",
        native_endpoint=False,
        encoder_frame=None,
        encoder_time_seconds=None,
        event_origin="none",
        endpoint_sample=None,
        endpoint_latency_ms=None,
        authoritative=False,
    )


def _records(
    corpus: LoadedCorpus,
    *,
    adapter: str,
    overrides: dict[int, NativeFinalEvent | ParakeetCppFinalEvent] | None = None,
    repeats: int = 1,
) -> tuple[RunRecord, ...]:
    fallback: NativeFinalEvent | ParakeetCppFinalEvent = (
        _native_final()
        if adapter == PARAKEET_REALTIME_EOU_ADAPTER
        else _cpp_final()
    )
    selected = overrides or {}
    return tuple(
        RunRecord(
            case_index,
            repeat,
            CaseTrace(partials=(), final=selected.get(case_index, fallback)),
        )
        for repeat in range(repeats)
        for case_index in range(len(corpus.cases))
    )


def test_label_set_digest_is_exact_ordered_and_domain_separated(tmp_path):
    corpus, binding = _surface(tmp_path)
    rows = [
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
        for label in binding.labels
    ]
    expected = hashlib.sha256(
        json.dumps(
            {"kind": AMI_ENDPOINT_PROXY_LABEL_SET_KIND, "labels": rows},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()

    assert binding.label_set_sha256 == expected
    assert validate_ami_endpoint_proxy_binding(corpus, binding) == binding.labels
    swapped = (
        binding.labels[1],
        binding.labels[0],
        *binding.labels[2:],
    )
    assert canonical_label_set_sha256(swapped) != expected
    with pytest.raises(AmiEndpointProxyError):
        validate_ami_endpoint_proxy_binding(
            corpus,
            replace(
                binding,
                labels=swapped,
                label_set_sha256=canonical_label_set_sha256(swapped),
            ),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        {"corpus_manifest_sha256": "0" * 64},
        {"receipt_sha256": "0" * 64},
        {"sidecar_sha256": "0" * 64},
        {"annotation_metadata_sha256": "0" * 64},
        {"label_set_sha256": "0" * 64},
    ],
)
def test_binding_rejects_digest_mismatch(tmp_path, mutation):
    corpus, binding = _surface(tmp_path)
    with pytest.raises(AmiEndpointProxyError):
        validate_ami_endpoint_proxy_binding(corpus, replace(binding, **mutation))

    assert corpus.provenance is not None
    wrong_suite = replace(
        corpus,
        provenance=replace(corpus.provenance, suite="other-suite"),
    )
    with pytest.raises(AmiEndpointProxyError):
        validate_ami_endpoint_proxy_binding(wrong_suite, binding)


def test_binding_rejects_pair_count_and_label_mutations(tmp_path):
    corpus, binding = _surface(tmp_path)
    pair_mismatch = list(binding.labels)
    pair_mismatch[1] = replace(
        pair_mismatch[1],
        forced_aligned_last_word_end_sample=_PROXY_SAMPLE + 1,
    )
    pair_mismatch_tuple = tuple(pair_mismatch)
    with pytest.raises(AmiEndpointProxyError):
        validate_ami_endpoint_proxy_binding(
            corpus,
            replace(
                binding,
                labels=pair_mismatch_tuple,
                label_set_sha256=canonical_label_set_sha256(pair_mismatch_tuple),
            ),
        )

    wrong_count = binding.labels[:-1]
    with pytest.raises(AmiEndpointProxyError):
        validate_ami_endpoint_proxy_binding(
            corpus,
            replace(
                binding,
                labels=wrong_count,
                label_set_sha256=canonical_label_set_sha256(wrong_count),
            ),
        )

    excessive_context = tuple(
        replace(
            label,
            forced_aligned_last_word_end_sample=label.samples - 1_601,
        )
        for label in binding.labels
    )
    with pytest.raises(AmiEndpointProxyError):
        canonical_label_set_sha256(excessive_context)

    coordinated_shift = tuple(
        replace(
            label,
            forced_aligned_last_word_end_sample=(
                label.forced_aligned_last_word_end_sample - 128
            ),
        )
        for label in binding.labels
    )
    shifted_label_set_sha256 = canonical_label_set_sha256(coordinated_shift)
    shifted_sidecar_sha256 = canonical_sidecar_sha256(
        annotation_metadata_sha256=binding.annotation_metadata_sha256,
        label_set_sha256=shifted_label_set_sha256,
        labels=coordinated_shift,
    )
    with pytest.raises(AmiEndpointProxyError):
        validate_ami_endpoint_proxy_binding(
            corpus,
            replace(
                binding,
                labels=coordinated_shift,
                label_set_sha256=shifted_label_set_sha256,
            ),
        )
    with pytest.raises(AmiEndpointProxyError):
        validate_ami_endpoint_proxy_binding(
            corpus,
            replace(
                binding,
                labels=coordinated_shift,
                label_set_sha256=shifted_label_set_sha256,
                sidecar_sha256=shifted_sidecar_sha256,
            ),
        )

    wrong_strata = list(binding.labels)
    wrong_cases = list(corpus.cases)
    for index in (0, 1):
        wrong_strata[index] = replace(
            wrong_strata[index],
            window_kind="turn_transition",
        )
        wrong_cases[index] = replace(
            wrong_cases[index],
            tags=("turn_transition", wrong_strata[index].channel),
        )
    wrong_strata_tuple = tuple(wrong_strata)
    with pytest.raises(AmiEndpointProxyError):
        validate_ami_endpoint_proxy_binding(
            replace(corpus, cases=tuple(wrong_cases)),
            replace(
                binding,
                labels=wrong_strata_tuple,
                label_set_sha256=canonical_label_set_sha256(
                    wrong_strata_tuple
                ),
            ),
        )

    for invalid in (
        replace(binding.labels[0], samples=False),
        replace(binding.labels[0], forced_aligned_last_word_end_sample=0),
        replace(
            binding.labels[0],
            forced_aligned_last_word_end_sample=_SAMPLES,
        ),
        replace(binding.labels[0], window_kind="overlap"),
        replace(binding.labels[0], channel="single"),
    ):
        with pytest.raises(AmiEndpointProxyError):
            canonical_label_set_sha256((invalid, *binding.labels[1:]))


def test_native_proxy_boundary_and_exact_denominators(tmp_path):
    corpus, binding = _surface(tmp_path)
    records = _records(
        corpus,
        adapter=PARAKEET_REALTIME_EOU_ADAPTER,
        overrides={
            0: _native_final(
                endpoint_sample=_PROXY_SAMPLE - 1,
                reason="eou",
            ),
            1: _native_final(
                endpoint_sample=_PROXY_SAMPLE,
                reason="eou",
            ),
            2: _native_final(
                endpoint_sample=_PROXY_SAMPLE + 1_280,
                reason="eou",
            ),
        },
    )

    report = aggregate_ami_endpoint_proxy(
        corpus,
        binding,
        records,
        _ready(PARAKEET_REALTIME_EOU_ADAPTER),
        repeats=1,
    )

    assert report["ground_truth"] is False
    assert report["denominators"] == {
        "corpus_cases": 24,
        "isolated_cases": 16,
        "excluded_turn_transition_cases": 8,
        "repeats": 1,
        "all_evaluations": 24,
        "scored_evaluations": 16,
    }
    assert report["endpoint_observations"] == {
        "feed_eou_observations": 3,
        "non_feed_eou_observations": 0,
        "evaluations_with_non_feed_eou": 0,
        "missing_eou_evaluations": 13,
        "missing_feed_eou_evaluations": 13,
        "non_feed_only_eou_evaluations": 0,
        "tail_exhaustion_evaluations": 13,
        "premature_feed_eou_evaluations": 1,
        "at_or_after_proxy_feed_eou_evaluations": 2,
    }
    assert report["sample_deltas"] == {
        "signed": {"p50": 0, "p95": 1_280, "max": 1_280},
        "early": {"p50": 1, "p95": 1, "max": 1},
        "post_proxy": {"p50": 0, "p95": 1_280, "max": 1_280},
        "absolute": {"p50": 1, "p95": 1_280, "max": 1_280},
    }
    assert report["milliseconds"] == {
        "signed": {"p50": 0.0, "p95": 80.0, "max": 80.0},
        "early": {"p50": 0.062, "p95": 0.062, "max": 0.062},
        "post_proxy": {"p50": 0.0, "p95": 80.0, "max": 80.0},
        "absolute": {"p50": 0.062, "p95": 80.0, "max": 80.0},
    }


def test_native_proxy_uses_endpoint_sample_including_model_padding(tmp_path):
    corpus, binding = _surface(tmp_path)
    report = aggregate_ami_endpoint_proxy(
        corpus,
        binding,
        _records(
            corpus,
            adapter=PARAKEET_REALTIME_EOU_ADAPTER,
            overrides={
                0: _native_final(
                    endpoint_sample=_PROXY_SAMPLE + 320,
                    reason="eou",
                    model_padding_samples=320,
                )
            },
        ),
        _ready(PARAKEET_REALTIME_EOU_ADAPTER),
        repeats=1,
    )

    assert report["sample_deltas"]["signed"] == {
        "p50": 320,
        "p95": 320,
        "max": 320,
    }
    assert report["milliseconds"]["signed"] == {
        "p50": 20.0,
        "p95": 20.0,
        "max": 20.0,
    }


def test_parakeet_cpp_counts_missing_tail_and_non_feed_eou(tmp_path):
    corpus, binding = _surface(tmp_path)
    feed_and_finalize = (
        _observed("feed", _PROXY_SAMPLE, 5),
        _observed("feed", _PROXY_SAMPLE + 1_280, 7),
        _observed("finalize", _SAMPLES + _TAIL_SAMPLES, 10),
    )
    finalize_only = (
        _observed("finalize", _SAMPLES + _TAIL_SAMPLES, 10),
    )
    report = aggregate_ami_endpoint_proxy(
        corpus,
        binding,
        _records(
            corpus,
            adapter=PARAKEET_CPP_ADAPTER,
            overrides={
                0: _cpp_final(feed_and_finalize),
                1: _cpp_final(finalize_only),
            },
        ),
        _ready(PARAKEET_CPP_ADAPTER),
        repeats=1,
    )

    assert report["sample_domain"] == {
        "adapter": PARAKEET_CPP_ADAPTER,
        "sample_rate_hz": 16_000,
        "endpoint_field": "observed_sample",
        "definition": "source_pcm_plus_fed_tail_samples",
        "selection": "first_feed_eou",
    }
    assert report["endpoint_observations"] == {
        "feed_eou_observations": 2,
        "non_feed_eou_observations": 2,
        "evaluations_with_non_feed_eou": 2,
        "missing_eou_evaluations": 14,
        "missing_feed_eou_evaluations": 15,
        "non_feed_only_eou_evaluations": 1,
        "tail_exhaustion_evaluations": 16,
        "premature_feed_eou_evaluations": 0,
        "at_or_after_proxy_feed_eou_evaluations": 1,
    }
    assert report["sample_deltas"]["signed"] == {
        "p50": 0,
        "p95": 0,
        "max": 0,
    }


def test_empty_endpoint_percentiles_are_explicitly_null(tmp_path):
    corpus, binding = _surface(tmp_path)
    report = aggregate_ami_endpoint_proxy(
        corpus,
        binding,
        _records(corpus, adapter=PARAKEET_REALTIME_EOU_ADAPTER),
        _ready(PARAKEET_REALTIME_EOU_ADAPTER),
        repeats=1,
    )

    empty = {"p50": None, "p95": None, "max": None}
    assert report["sample_deltas"] == {
        "signed": empty,
        "early": empty,
        "post_proxy": empty,
        "absolute": empty,
    }
    assert report["milliseconds"] == {
        "signed": empty,
        "early": empty,
        "post_proxy": empty,
        "absolute": empty,
    }


def test_wrapper_accepts_real_reducer_shapes_and_reconstructs_safe_snapshot(
    tmp_path,
):
    corpus, binding = _surface(tmp_path)
    ready = _ready(PARAKEET_CPP_ADAPTER)
    records = _records(corpus, adapter=PARAKEET_CPP_ADAPTER)
    metrics = aggregate_metrics(
        corpus,
        records,
        ready,
        repeats=1,
        stratum_tags=("isolated", "close", "far"),
    )
    metrics["ami_endpoint_proxy"] = aggregate_ami_endpoint_proxy(
        corpus,
        binding,
        records,
        ready,
        repeats=1,
    )
    config_without_contract = {
        "chunk_samples": 1_280,
        "pace": "burst",
        "partial_interval_ms": 80,
        "tail_padding_samples": 1_280,
        "repeats": 1,
        "stratum_tags": ["close", "far", "isolated"],
        "ami_endpoint_proxy_label_set_sha256": binding.label_set_sha256,
        "ami_endpoint_proxy_sidecar_sha256": binding.sidecar_sha256,
    }
    manifest_sha256 = "e" * 64
    nested = {
        "ok": True,
        "evidence": streaming_stt_eval._evidence_binding(
            PARAKEET_CPP_ADAPTER,
            pace="burst",
            adapter_config=ParakeetCppConfig(),
            stratum_tags=("close", "far", "isolated"),
            ami_endpoint_proxy=binding,
        ),
        "worker": {
            "adapter": PARAKEET_CPP_ADAPTER,
            "manifest_sha256": manifest_sha256,
        },
        "corpus": streaming_stt_eval._corpus_binding(
            corpus,
            ami_endpoint_proxy=binding,
        ),
        "config": {
            **config_without_contract,
            "contract_sha256": streaming_stt_eval._canonical_digest(
                config_without_contract
            ),
        },
        "metrics": metrics,
        "worker_stderr": {},
        "evaluator": streaming_stt_eval._evaluator_binding(
            PARAKEET_CPP_ADAPTER
        ),
    }

    safe = ami_endpoint_proxy_eval._validate_nested_report(
        nested,
        binding=binding,
        adapter=PARAKEET_CPP_ADAPTER,
        repeats=1,
        worker_manifest_sha256=manifest_sha256,
    )

    assert safe["ok"] is True
    assert safe["metrics"]["evaluations"] == 24
    assert [row["tag"] for row in safe["metrics"]["strata"]] == [
        "close",
        "far",
        "isolated",
    ]


def test_wrapper_accepts_real_native_source_endpoint_shape(tmp_path):
    corpus, _binding = _surface(tmp_path)
    ready = _ready(PARAKEET_REALTIME_EOU_ADAPTER)
    metrics = aggregate_metrics(
        corpus,
        _records(corpus, adapter=PARAKEET_REALTIME_EOU_ADAPTER),
        ready,
        repeats=1,
    )

    snapshot = ami_endpoint_proxy_eval._endpointing_snapshot(
        metrics["endpointing"],
        adapter=PARAKEET_REALTIME_EOU_ADAPTER,
        expected_evaluations=24,
    )

    assert snapshot["evaluations"] == 24
    assert snapshot["tail_exhaustions"] == 24


def test_proxy_rejects_record_coverage_order_adapter_and_final_types(tmp_path):
    corpus, binding = _surface(tmp_path)
    ready = _ready(PARAKEET_REALTIME_EOU_ADAPTER)
    records = _records(corpus, adapter=PARAKEET_REALTIME_EOU_ADAPTER)
    invalid_rows: tuple[object, ...] = (
        records[:-1],
        (records[1], records[0], *records[2:]),
        list(records[:-1]) + [object()],
    )
    for invalid in invalid_rows:
        with pytest.raises(AmiEndpointProxyError):
            aggregate_ami_endpoint_proxy(
                corpus,
                binding,
                invalid,  # type: ignore[arg-type]
                ready,
                repeats=1,
            )

    with pytest.raises(AmiEndpointProxyError):
        aggregate_ami_endpoint_proxy(
            corpus,
            binding,
            records,
            replace(ready, adapter="other-adapter"),
            repeats=1,
        )

    generic = FinalEvent(
        request_id="private-request",
        seq=0,
        text="private hypothesis",
        samples_seen=_SAMPLES,
        elapsed_ms=10.0,
        finalization_ms=1.0,
        compute_ms=1.0,
        audio_seconds=1.0,
        chunks=1,
        deadline_misses=0,
        max_backlog_ms=0.0,
        resources=_USAGE,
    )
    wrong_final = (
        replace(records[0], trace=CaseTrace(partials=(), final=generic)),
        *records[1:],
    )
    with pytest.raises(AmiEndpointProxyError):
        aggregate_ami_endpoint_proxy(
            corpus,
            binding,
            wrong_final,
            ready,
            repeats=1,
        )


def test_proxy_report_and_private_dataclass_repr_do_not_expose_rows(tmp_path):
    corpus, binding = _surface(tmp_path)
    report = aggregate_ami_endpoint_proxy(
        corpus,
        binding,
        _records(
            corpus,
            adapter=PARAKEET_REALTIME_EOU_ADAPTER,
            overrides={
                0: _native_final(
                    endpoint_sample=_PROXY_SAMPLE,
                    reason="eou",
                )
            },
        ),
        _ready(PARAKEET_REALTIME_EOU_ADAPTER),
        repeats=1,
    )
    serialized = json.dumps(report, sort_keys=True)

    assert repr(binding.labels[0]) == "AmiEndpointProxyLabel()"
    assert binding.labels[0].case_id not in repr(binding)
    assert "forced_aligned_last_word_end_sample" not in serialized
    assert "case_id" not in serialized
    assert "expected_text" not in serialized
    assert "private transcript" not in serialized
    assert str(tmp_path) not in serialized
    assert binding.labels[0].case_id not in serialized
    assert binding.labels[0].pcm_sha256 not in serialized
    assert str(_PROXY_SAMPLE) not in serialized
