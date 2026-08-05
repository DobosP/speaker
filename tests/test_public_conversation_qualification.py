from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_public_conversation_fixture import _publish_set
from tests.test_streaming_stt_suite import _aggregate_report
from tools import public_conversation_fixture as fixture
from tools import public_conversation_qualification as qualification
from tools import streaming_stt_eval
from tools import streaming_stt_suite
from tools.streaming_stt.corpus import load_corpus
from tools.streaming_stt.manifest import (
    FasterWhisperEndpointConfig,
    ParakeetCppConfig,
)

_FAKE_STRATA = (
    (tuple(f"task_{index:02d}" for index in range(8)), (3,) * 8),
    (("clean", "disfluent", "overlap_adjacent"), (8, 8, 8)),
    (("close", "far", "isolated", "turn_transition"), (12, 12, 16, 8)),
    (
        ("d0_2_6s", "d1_6_10s", "d2_10_15s", "d3_15_25s"),
        (6, 6, 6, 6),
    ),
)
_ENDPOINT_ADAPTERS = {
    "native": qualification.PARAKEET_REALTIME_EOU_ADAPTER,
    "cpp": qualification.PARAKEET_CPP_ADAPTER,
}


def _worker_digest(path: Path | str) -> str:
    selected = Path(path).resolve(strict=False)
    return hashlib.sha256(str(selected).encode("utf-8")).hexdigest()


@pytest.fixture(autouse=True)
def _fake_worker_manifest_loader(monkeypatch):
    monkeypatch.setattr(
        qualification,
        "load_worker_manifest",
        lambda path: SimpleNamespace(
            adapter="fake-json-v1",
            digest=_worker_digest(path),
        ),
    )


def _metric_snapshot(
    *,
    cases: int,
    repeats: int,
) -> dict[str, object]:
    evaluations = cases * repeats
    transcript = {
        "clips": evaluations,
        "nonempty": evaluations,
        "exact": evaluations,
        "wer": 0.0,
        "cer": 0.0,
        "word_errors": 0,
        "substitutions": 0,
        "insertions": 0,
        "deletions": 0,
        "ref_words": evaluations,
        "hyp_words": evaluations,
        "char_edits": 0,
        "ref_chars": evaluations,
        "hyp_chars": evaluations,
        "keyword_attempts": 0,
        "keyword_hits": 0,
    }
    return {
        "evaluations": evaluations,
        "coverage_complete": True,
        "accuracy": {
            "transcript": transcript,
            "command_attempts": 0,
            "command_hits": 0,
            "command_recall": None,
            "silence_nonempty_partials": 0,
            "silence_nonempty_finals": 0,
        },
        "latency": {
            "model_load_ms": 1.0,
            "first_nonempty_partial_p50_ms": 1.0,
            "first_nonempty_partial_p95_ms": 1.0,
            "stable_partial_p50_ms": 1.0,
            "stable_partial_p95_ms": 1.0,
            "stable_partial_missing": 0,
            "finalization_p50_ms": 1.0,
            "finalization_p95_ms": 1.0,
            "finalization_max_ms": 1.0,
        },
        "streaming": {
            "partial_events": evaluations,
            "churn_token_edits": 0,
            "retracted_words": 0,
            "deadline_misses": 0,
            "max_backlog_p95_ms": 0.0,
            "max_backlog_ms": 0.0,
            "model_padding_total_samples": 0,
            "model_padding_max_samples": 0,
        },
        "throughput": {
            "audio_total_sec": float(cases),
            "compute_total_ms": 0.0,
            "aggregate_rtf": 0.0,
        },
        "determinism": {
            "repeats": repeats,
            "cases_with_final_disagreement": 0,
        },
        "resources": {
            "max_reported_rss_mb": 1.0,
            "peak_threads": 1,
            "peak_vram_mb": None,
            "source": "worker_ready_and_final_resource_reports",
            "rss_scope": "maximum_of_ready_and_final_samples_not_process_peak",
        },
    }


def _native_endpointing(evaluations: int) -> dict[str, object]:
    source_samples = evaluations * 16_000
    return {
        "evaluations": evaluations,
        "native_endpoints": 0,
        "native_eou_events": 0,
        "native_eob_backchannels": 0,
        "authoritative_endpoints": 0,
        "early_source_endpoints": 0,
        "source_early_native_events": 0,
        "source_early_eou_events": 0,
        "source_early_eob_backchannels": 0,
        "tail_exhaustions": evaluations,
        "reasons": {"eou": 0, "eob": 0, "tail_exhausted": evaluations},
        "source_samples": {
            "offered_total": source_samples,
            "consumed_total": source_samples,
            "dropped_total": 0,
        },
        "declared_tail_samples": {
            "offered_total": 0,
            "consumed_total": 0,
            "unconsumed_total": 0,
        },
        "terminal_model_padding_samples_total": 0,
        "model_input_samples_consumed_total": source_samples,
        "endpoint_sample_p50": None,
        "endpoint_sample_p95": None,
        "authoritative_endpoint_latency_p50_ms": None,
        "authoritative_endpoint_latency_p95_ms": None,
        "authoritative_endpoint_latency_max_ms": None,
        "endpoint_probability_p50": None,
        "endpoint_probability_p95": None,
    }


def _cpp_endpointing(evaluations: int) -> dict[str, object]:
    source_samples = evaluations * 16_000
    return {
        "evaluations": evaluations,
        "accepted_endpoints": 0,
        "tail_exhaustions": evaluations,
        "terminal_reasons": {"eou": 0, "tail_exhausted": evaluations},
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
            "offered_total": source_samples,
            "consumed_total": source_samples,
            "dropped_total": 0,
        },
        "tail_samples": {
            "offered_total": 0,
            "consumed_total": 0,
            "unconsumed_total": 0,
        },
        "terminal_model_padding_samples_total": 0,
        "model_input_samples_consumed_total": source_samples,
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


def _endpoint_metric_snapshot(
    kind: str,
    *,
    cases: int,
    repeats: int,
) -> dict[str, object]:
    metrics = _metric_snapshot(cases=cases, repeats=repeats)
    evaluations = cases * repeats
    metrics["throughput"].update(
        {
            "aggregate_rtf_scope": "declared_source_audio",
            "model_input_total_sec": float(evaluations),
            "model_input_aggregate_rtf": 0.0,
        }
    )
    metrics["endpointing"] = (
        _native_endpointing(evaluations)
        if kind == "native"
        else _cpp_endpointing(evaluations)
    )
    return metrics


def _complete_metrics(
    strata: tuple[tuple[str, ...], tuple[int, ...]],
    *,
    adapter: str = "fake-json-v1",
    repeats: int,
) -> dict[str, object]:
    tags, counts = strata
    kind = next(
        (kind for kind, selected in _ENDPOINT_ADAPTERS.items() if selected == adapter),
        None,
    )
    factory = (
        (lambda cases: _endpoint_metric_snapshot(kind, cases=cases, repeats=repeats))
        if kind is not None
        else (lambda cases: _metric_snapshot(cases=cases, repeats=repeats))
    )
    result = factory(24)
    result["strata"] = [
        {
            "tag": tag,
            "cases": count,
            "metrics": factory(count),
        }
        for tag, count in zip(tags, counts, strict=True)
    ]
    return result


@pytest.mark.parametrize("kind", ("native", "cpp"))
def test_metrics_snapshot_accepts_production_endpoint_shapes(kind):
    metrics = _endpoint_metric_snapshot(kind, cases=24, repeats=3)
    metrics["strata"] = [
        {
            "tag": "clean",
            "cases": 8,
            "metrics": _endpoint_metric_snapshot(kind, cases=8, repeats=3),
        }
    ]

    assert qualification._metrics_snapshot(
        metrics,
        adapter=_ENDPOINT_ADAPTERS[kind],
        cases=24,
        repeats=3,
        expected_strata=(("clean", 8),),
    ) == metrics


def _nonzero_endpoint_metric_snapshot(
    kind: str,
) -> tuple[dict[str, object], int, int]:
    cases, repeats = ((2, 2) if kind == "native" else (1, 2))
    metrics = _metric_snapshot(cases=cases, repeats=repeats)
    if kind == "native":
        metrics["throughput"] = {
            "audio_total_sec": 0.03,
            "compute_total_ms": 16.0,
            "aggregate_rtf": 0.5333,
            "aggregate_rtf_scope": "declared_source_audio",
            "model_input_total_sec": 0.028,
            "model_input_aggregate_rtf": 0.5818,
        }
        endpointing = _native_endpointing(4)
        endpointing.update(
            {
                "native_endpoints": 4,
                "native_eou_events": 2,
                "native_eob_backchannels": 2,
                "authoritative_endpoints": 1,
                "early_source_endpoints": 2,
                "source_early_native_events": 2,
                "source_early_eou_events": 1,
                "source_early_eob_backchannels": 1,
                "tail_exhaustions": 0,
                "reasons": {"eou": 2, "eob": 2, "tail_exhausted": 0},
                "source_samples": {
                    "offered_total": 480,
                    "consumed_total": 360,
                    "dropped_total": 120,
                },
                "declared_tail_samples": {
                    "offered_total": 640,
                    "consumed_total": 80,
                    "unconsumed_total": 560,
                },
                "model_input_samples_consumed_total": 440,
                "endpoint_sample_p50": 80.0,
                "endpoint_sample_p95": 240.0,
                "authoritative_endpoint_latency_p50_ms": 5.0,
                "authoritative_endpoint_latency_p95_ms": 5.0,
                "authoritative_endpoint_latency_max_ms": 5.0,
                "endpoint_probability_p50": 0.7,
                "endpoint_probability_p95": 0.9,
            }
        )
    else:
        metrics["throughput"] = {
            "audio_total_sec": 2.0,
            "compute_total_ms": 8.0,
            "aggregate_rtf": 0.004,
            "aggregate_rtf_scope": "declared_source_audio",
            "model_input_total_sec": 2.12,
            "model_input_aggregate_rtf": 0.0038,
        }
        endpointing = _cpp_endpointing(2)
        endpointing.update(
            {
                "accepted_endpoints": 1,
                "tail_exhaustions": 1,
                "terminal_reasons": {"eou": 1, "tail_exhausted": 1},
                "terminal_origins": {"feed": 1, "none": 1},
                "observations": {
                    "total": 10,
                    "eou": 4,
                    "eob": 6,
                    "origins": {"feed": 8, "finalize": 2},
                    "source_early": {"total": 4, "eou": 1, "eob": 3},
                    "post_source_feed": {"total": 4, "eou": 2, "eob": 2},
                },
                "source_samples": {
                    "offered_total": 32_000,
                    "consumed_total": 32_000,
                    "dropped_total": 0,
                },
                "tail_samples": {
                    "offered_total": 2_560,
                    "consumed_total": 1_920,
                    "unconsumed_total": 640,
                },
                "model_input_samples_consumed_total": 33_920,
                "accepted_endpoint_sample_p50": 16_640.0,
                "accepted_endpoint_sample_p95": 16_640.0,
                "observed_encoder_frame_p50": 10.0,
                "observed_encoder_frame_p95": 14.0,
                "observed_encoder_time_p50_seconds": 0.8,
                "observed_encoder_time_p95_seconds": 1.12,
                "accepted_endpoint_latency_p50_ms": 40.0,
                "accepted_endpoint_latency_p95_ms": 40.0,
                "accepted_endpoint_latency_max_ms": 40.0,
            }
        )
    metrics["endpointing"] = endpointing
    return metrics, cases, repeats


@pytest.mark.parametrize("kind", ("native", "cpp"))
def test_metrics_snapshot_accepts_nonzero_production_endpoint_shapes(kind):
    metrics, cases, repeats = _nonzero_endpoint_metric_snapshot(kind)

    assert qualification._metrics_snapshot(
        metrics,
        adapter=_ENDPOINT_ADAPTERS[kind],
        cases=cases,
        repeats=repeats,
        expected_strata=None,
    ) == metrics


@pytest.mark.parametrize("kind", ("native", "cpp"))
def test_metrics_snapshot_rejects_inconsistent_endpoint_counters(kind):
    metrics = _endpoint_metric_snapshot(kind, cases=24, repeats=3)
    field = "native_endpoints" if kind == "native" else "accepted_endpoints"
    metrics["endpointing"][field] = 1

    with pytest.raises(qualification.QualificationError):
        qualification._metrics_snapshot(
            metrics,
            adapter=_ENDPOINT_ADAPTERS[kind],
            cases=24,
            repeats=3,
            expected_strata=None,
        )


@pytest.mark.parametrize(
    ("adapter", "metrics"),
    (
        (
            qualification.PARAKEET_REALTIME_EOU_ADAPTER,
            _endpoint_metric_snapshot("cpp", cases=24, repeats=3),
        ),
        (
            qualification.PARAKEET_CPP_ADAPTER,
            _endpoint_metric_snapshot("native", cases=24, repeats=3),
        ),
        (
            qualification.PARAKEET_CPP_ADAPTER,
            _metric_snapshot(cases=24, repeats=3),
        ),
        (
            "fake-json-v1",
            _endpoint_metric_snapshot("native", cases=24, repeats=3),
        ),
    ),
)
def test_metrics_snapshot_rejects_adapter_endpoint_shape_mismatch(adapter, metrics):
    with pytest.raises(qualification.QualificationError):
        qualification._metrics_snapshot(
            metrics,
            adapter=adapter,
            cases=24,
            repeats=3,
            expected_strata=None,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "native_early_eou",
        "native_model_input",
        "native_probability_range",
        "native_partial_summary",
        "cpp_source_drop",
        "cpp_padding",
        "cpp_model_input",
        "cpp_observation_partition",
        "cpp_subtype_partition",
        "cpp_partial_summary",
    ),
)
def test_metrics_snapshot_rejects_reducer_impossible_endpoint_aggregates(mutation):
    kind = "native" if mutation.startswith("native") else "cpp"
    metrics, cases, repeats = _nonzero_endpoint_metric_snapshot(kind)
    endpointing = metrics["endpointing"]
    assert isinstance(endpointing, dict)
    if mutation == "native_early_eou":
        endpointing["source_early_native_events"] = 3
        endpointing["early_source_endpoints"] = 3
        endpointing["source_early_eou_events"] = 2
    elif mutation == "native_model_input":
        endpointing["model_input_samples_consumed_total"] = 441
    elif mutation == "native_probability_range":
        endpointing["endpoint_probability_p50"] = 2.0
        endpointing["endpoint_probability_p95"] = 2.0
    elif mutation == "native_partial_summary":
        endpointing["endpoint_sample_p50"] = None
    elif mutation == "cpp_source_drop":
        endpointing["source_samples"]["offered_total"] = 32_001
        endpointing["source_samples"]["dropped_total"] = 1
    elif mutation == "cpp_padding":
        endpointing["terminal_model_padding_samples_total"] = 1
        endpointing["model_input_samples_consumed_total"] = 33_921
    elif mutation == "cpp_model_input":
        endpointing["model_input_samples_consumed_total"] = 33_921
    elif mutation == "cpp_observation_partition":
        endpointing["observations"]["source_early"] = {
            "total": 5,
            "eou": 2,
            "eob": 3,
        }
    elif mutation == "cpp_subtype_partition":
        endpointing["observations"]["source_early"] = {
            "total": 4,
            "eou": 4,
            "eob": 0,
        }
        endpointing["observations"]["post_source_feed"] = {
            "total": 4,
            "eou": 4,
            "eob": 0,
        }
    else:
        endpointing["accepted_endpoint_sample_p50"] = None

    with pytest.raises(qualification.QualificationError):
        qualification._metrics_snapshot(
            metrics,
            adapter=_ENDPOINT_ADAPTERS[kind],
            cases=cases,
            repeats=repeats,
            expected_strata=None,
        )


def test_metrics_snapshot_accepts_source_complete_native_eob():
    metrics = _endpoint_metric_snapshot("native", cases=1, repeats=1)
    endpointing = metrics["endpointing"]
    endpointing.update(
        {
            "native_endpoints": 1,
            "native_eob_backchannels": 1,
            "tail_exhaustions": 0,
            "reasons": {"eou": 0, "eob": 1, "tail_exhausted": 0},
            "endpoint_sample_p50": 16_000.0,
            "endpoint_sample_p95": 16_000.0,
            "endpoint_probability_p50": 0.5,
            "endpoint_probability_p95": 0.5,
        }
    )

    assert qualification._metrics_snapshot(
        metrics,
        adapter=qualification.PARAKEET_REALTIME_EOU_ADAPTER,
        cases=1,
        repeats=1,
        expected_strata=None,
    ) == metrics


def test_metrics_snapshot_accepts_cpp_observations_without_accepted_endpoint():
    metrics, cases, repeats = _nonzero_endpoint_metric_snapshot("cpp")
    endpointing = metrics["endpointing"]
    endpointing.update(
        {
            "accepted_endpoints": 0,
            "tail_exhaustions": 2,
            "terminal_reasons": {"eou": 0, "tail_exhausted": 2},
            "terminal_origins": {"feed": 0, "none": 2},
            "accepted_endpoint_sample_p50": None,
            "accepted_endpoint_sample_p95": None,
            "accepted_endpoint_latency_p50_ms": None,
            "accepted_endpoint_latency_p95_ms": None,
            "accepted_endpoint_latency_max_ms": None,
        }
    )

    assert qualification._metrics_snapshot(
        metrics,
        adapter=qualification.PARAKEET_CPP_ADAPTER,
        cases=cases,
        repeats=repeats,
        expected_strata=None,
    ) == metrics


def _write_checkpoint(kwargs: dict[str, object], result: dict[str, object]) -> None:
    checkpoint = Path(kwargs["checkpoint_path"])
    checkpoint.write_bytes(
        qualification._canonical_json_bytes(result["suite"]) + b"\n"
    )
    checkpoint.chmod(0o600)


def _fake_result(
    workers: tuple[Path, ...],
    corpora: tuple[Path, ...],
) -> dict[str, object]:
    fixture_set = fixture.validate_fixture_set(corpora)
    digests = [load_corpus(path).digest for path in corpora]
    manifests = [qualification.load_worker_manifest(path) for path in workers]
    worker_digests = [manifest.digest for manifest in manifests]
    config_rows = [
        {
            "corpus_index": index,
            "tags": list(tags),
            "selection_sha256": qualification._canonical_sha256(list(tags)),
        }
        for index, (tags, _counts) in enumerate(_FAKE_STRATA)
    ]
    strata_config = {
        "mode": "per_corpus",
        "corpora": config_rows,
        "selection_sha256": qualification._canonical_sha256(config_rows),
    }
    config: dict[str, object] = {
        "repeats": 3,
        "strata": strata_config,
        "overrides": {
            "chunk_samples": None,
            "pace": None,
            "partial_interval_ms": None,
            "tail_padding_samples": None,
        },
    }
    config["contract_sha256"] = qualification._canonical_sha256(config)
    evaluator_digest: str | None = None
    runs: list[dict[str, object]] = []
    for worker_index in range(2):
        manifest = manifests[worker_index]
        adapter = manifest.adapter
        selected_stream = streaming_stt_eval._selected_stream(
            manifest,
            None,
            chunk_samples=None,
            pace=None,
            partial_interval_ms=None,
            tail_padding_samples=None,
        )
        for corpus_index, digest in enumerate(digests):
            tags = _FAKE_STRATA[corpus_index][0]
            report = _aggregate_report(
                workers[worker_index],
                corpora[corpus_index],
                worker_digest=worker_digests[worker_index],
                corpus_digest=digest,
                stratum_tags=tags,
            )
            report["metrics"] = _complete_metrics(
                _FAKE_STRATA[corpus_index],
                adapter=adapter,
                repeats=3,
            )
            report["evidence"] = streaming_stt_eval._evidence_binding(
                adapter,
                pace=selected_stream.pace,
                adapter_config=getattr(manifest, "adapter_config", None),
                stratum_tags=tags,
            )
            report_config: dict[str, object] = {
                **selected_stream.as_dict(),
                "repeats": 3,
                "stratum_tags": list(tags),
            }
            report_config["contract_sha256"] = qualification._canonical_sha256(
                report_config
            )
            report["config"] = report_config
            report["evaluator"] = streaming_stt_eval._evaluator_binding(adapter)
            current_evaluator = streaming_stt_suite._evaluator_implementation_sha256(
                report["evaluator"]
            )
            if evaluator_digest is None:
                evaluator_digest = current_evaluator
            else:
                assert evaluator_digest == current_evaluator
            binding: dict[str, object] = {
                "worker_index": worker_index,
                "corpus_index": corpus_index,
                "worker_manifest_sha256": worker_digests[worker_index],
                "corpus_manifest_sha256": digest,
                "stratum_selection_sha256": qualification._canonical_sha256(
                    list(_FAKE_STRATA[corpus_index][0])
                ),
                "evaluator_implementation_sha256": evaluator_digest,
                "report_sha256": qualification._canonical_sha256(report),
            }
            binding["binding_sha256"] = qualification._canonical_sha256(binding)
            runs.append({"binding": binding, "report": report})
    assert evaluator_digest is not None
    suite = streaming_stt_suite._suite_report(
        state="complete",
        planned_workers=2,
        planned_corpora=4,
        worker_digests=worker_digests,
        corpus_digests=digests,
        evaluator_implementation_sha256=evaluator_digest,
        controller_sha256=streaming_stt_suite._controller_source_sha256(),
        config=config,
        runs=runs,
    )
    return {
        "ok": True,
        "fixture": {
            "fixture_id": fixture.FIXTURE_ID,
            "lock_recipe_sha256": fixture_set.lock.recipe_sha256,
            "fixture_set_sha256": fixture_set.fixture_set_sha256,
            "sources": len(fixture_set.sources),
            "cases": fixture_set.cases,
            "pcm_bytes": fixture_set.pcm_bytes,
            "validation": "before_and_after_suite",
        },
        "suite": suite,
    }


def _rebind_fake_run(result: dict[str, object], run_index: int = 0) -> None:
    suite = result["suite"]
    assert isinstance(suite, dict)
    runs = suite["runs"]
    assert isinstance(runs, list)
    run = runs[run_index]
    report = run["report"]
    binding = run["binding"]
    binding.pop("binding_sha256")
    binding["report_sha256"] = qualification._canonical_sha256(report)
    binding["binding_sha256"] = qualification._canonical_sha256(binding)
    matrix = qualification._canonical_sha256([item["binding"] for item in runs])
    suite["controller"]["matrix_sha256"] = matrix
    suite["controller"]["completed_matrix_sha256"] = matrix


def test_run_supplies_exact_canonical_strata_and_nonpromotional_result(
    tmp_path,
    monkeypatch,
):
    _lock, paths = _publish_set(tmp_path)
    observed: dict[str, object] = {}

    def run_suite(workers, corpora, **kwargs):
        observed["workers"] = tuple(workers)
        observed["corpora"] = tuple(corpora)
        observed["stratum_tags"] = kwargs["stratum_tags"]
        observed["corpus_stratum_tags"] = kwargs["corpus_stratum_tags"]
        observed["scratch"] = kwargs["scratch_parent"]
        observed["checkpoint"] = kwargs["checkpoint_path"]
        result = _fake_result(observed["workers"], observed["corpora"])
        _write_checkpoint(kwargs, result)
        return result

    monkeypatch.setattr(qualification, "_PRODUCTION_RUNNER", run_suite)

    result = qualification.run_qualification(
        baseline_worker_manifest=tmp_path / "baseline.json",
        candidate_worker_manifest=tmp_path / "candidate.json",
        corpus_paths=reversed(paths),
        scratch_parent=tmp_path / "scratch",
        output_path=tmp_path / "report.json",
    )

    assert tuple(path.parent.name for path in observed["corpora"]) == fixture.SOURCE_IDS
    assert observed["stratum_tags"] == ()
    assert observed["checkpoint"] != tmp_path / "report.json"
    assert Path(observed["checkpoint"]).parent == observed["scratch"]
    assert Path(observed["checkpoint"]).name.startswith(".canonical-strata-inner-")
    selected = observed["corpus_stratum_tags"]
    assert isinstance(selected, dict)
    assert list(selected.values()) == [
        tuple(f"task_{index:02d}" for index in range(8)),
        ("clean", "disfluent", "overlap_adjacent"),
        ("close", "far", "isolated", "turn_transition"),
        ("d0_2_6s", "d1_6_10s", "d2_10_15s", "d3_15_25s"),
    ]
    assert result["ok"] is True
    assert result["execution_semantics"] == "coverage_only"
    assert result["quality_decision"] == "not_evaluated"
    assert result["development_only"] is True
    assert result["promotional"] is False
    assert result["worker_roles"] == [
        {
            "role": "baseline",
            "worker_index": 0,
            "manifest_sha256": _worker_digest(tmp_path / "baseline.json"),
        },
        {
            "role": "candidate",
            "worker_index": 1,
            "manifest_sha256": _worker_digest(tmp_path / "candidate.json"),
        },
    ]
    encoded = json.dumps(result, sort_keys=True)
    assert str(tmp_path) not in encoded
    assert "expected_text" not in encoded


def test_production_entry_has_no_public_runner_injection() -> None:
    assert "run_validated_suite_fn" not in inspect.signature(
        qualification.run_qualification
    ).parameters


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("evaluations", 999),
        ("verbatim_transcript", "PRIVATE_TRANSCRIPT_CANARY"),
        ("relative_path", "private/source.f32le"),
        ("speaker_identity", "PRIVATE_SPEAKER_CANARY"),
    ),
)
def test_coordinated_forged_metrics_fail_closed(
    field,
    value,
    tmp_path,
    monkeypatch,
):
    _lock, paths = _publish_set(tmp_path)

    def runner(workers, corpora, **kwargs):
        result = _fake_result(tuple(workers), tuple(corpora))
        result["suite"]["runs"][0]["report"]["metrics"][field] = value
        _rebind_fake_run(result)
        _write_checkpoint(kwargs, result)
        return result

    monkeypatch.setattr(qualification, "_PRODUCTION_RUNNER", runner)
    with pytest.raises(qualification.QualificationError):
        qualification.run_qualification(
            baseline_worker_manifest=tmp_path / "baseline.json",
            candidate_worker_manifest=tmp_path / "candidate.json",
            corpus_paths=paths,
            scratch_parent=tmp_path / "scratch",
            output_path=tmp_path / "report.json",
        )


@pytest.mark.parametrize(
    "field",
    ("adoption_authority", "capture_to_text_latency", "live_hardware"),
)
def test_source_safety_evidence_cannot_be_rewritten_as_safe(field, tmp_path, monkeypatch):
    _lock, paths = _publish_set(tmp_path)

    def runner(workers, corpora, **kwargs):
        result = _fake_result(tuple(workers), tuple(corpora))
        result["suite"]["runs"][0]["report"]["evidence"][field] = True
        _rebind_fake_run(result)
        _write_checkpoint(kwargs, result)
        return result

    monkeypatch.setattr(qualification, "_PRODUCTION_RUNNER", runner)
    with pytest.raises(qualification.QualificationError):
        qualification.run_qualification(
            baseline_worker_manifest=tmp_path / "baseline.json",
            candidate_worker_manifest=tmp_path / "candidate.json",
            corpus_paths=paths,
            scratch_parent=tmp_path / "scratch",
            output_path=tmp_path / "report.json",
        )


def test_source_run_config_requires_exact_resolved_production_fields(
    tmp_path,
    monkeypatch,
):
    _lock, paths = _publish_set(tmp_path)

    def runner(workers, corpora, **kwargs):
        result = _fake_result(tuple(workers), tuple(corpora))
        config = result["suite"]["runs"][0]["report"]["config"]
        config.pop("chunk_samples")
        config.pop("contract_sha256")
        config["contract_sha256"] = qualification._canonical_sha256(config)
        _rebind_fake_run(result)
        _write_checkpoint(kwargs, result)
        return result

    monkeypatch.setattr(qualification, "_PRODUCTION_RUNNER", runner)
    with pytest.raises(qualification.QualificationError):
        qualification.run_qualification(
            baseline_worker_manifest=tmp_path / "baseline.json",
            candidate_worker_manifest=tmp_path / "candidate.json",
            corpus_paths=paths,
            scratch_parent=tmp_path / "scratch",
            output_path=tmp_path / "report.json",
        )


def test_result_reconstructs_safe_aggregate_without_nested_extras(
    tmp_path,
    monkeypatch,
):
    _lock, paths = _publish_set(tmp_path)
    canaries = (
        "PRIVATE_TRANSCRIPT_CANARY",
        "private/source.f32le",
        "PRIVATE_SPEAKER_CANARY",
    )

    def runner(workers, corpora, **kwargs):
        result = _fake_result(tuple(workers), tuple(corpora))
        report = result["suite"]["runs"][0]["report"]
        report["worker"]["verbatim_transcript"] = canaries[0]
        report["evidence"]["relative_source"] = canaries[1]
        report["worker_stderr"]["speaker_identity"] = canaries[2]
        _rebind_fake_run(result)
        _write_checkpoint(kwargs, result)
        return result

    monkeypatch.setattr(qualification, "_PRODUCTION_RUNNER", runner)
    result = qualification.run_qualification(
        baseline_worker_manifest=tmp_path / "baseline.json",
        candidate_worker_manifest=tmp_path / "candidate.json",
        corpus_paths=paths,
        scratch_parent=tmp_path / "scratch",
        output_path=tmp_path / "report.json",
    )

    encoded = json.dumps(result, sort_keys=True)
    assert all(canary not in encoded for canary in canaries)
    assert qualification.validate_qualification_report(
        json.loads(qualification._canonical_json_bytes(result))
    ) == result
    assert result["schema_version"] == 2
    assert result["suite"]["controller"]["kind"] == (
        "public-conversation-safe-aggregate-suite-v1"
    )
    assert "source_controller_sha256" in result["suite"]["controller"]
    assert "controller_sha256" not in result["suite"]["controller"]
    safe_report = result["suite"]["runs"][0]["report"]
    assert set(safe_report) == {
        "ok",
        "evidence",
        "worker",
        "corpus",
        "config",
        "metrics",
        "evaluator",
        "source_report_sha256",
    }
    assert set(result["suite"]["runs"][0]["binding"]) == {
        "worker_index",
        "corpus_index",
        "worker_manifest_sha256",
        "corpus_manifest_sha256",
            "stratum_selection_sha256",
            "evaluator_implementation_sha256",
            "worker_adapter",
            "stream_config_sha256",
            "source_report_sha256",
            "source_binding_sha256",
            "report_sha256",
        "binding_sha256",
    }


@pytest.mark.parametrize(
    "mutation",
    ("raw_suite_kind", "source_binding", "metric_denominator"),
)
def test_retained_validator_rejects_coordinated_safe_view_tamper(
    mutation,
    tmp_path,
    monkeypatch,
):
    _lock, paths = _publish_set(tmp_path)

    def runner(workers, corpora, **kwargs):
        result = _fake_result(tuple(workers), tuple(corpora))
        _write_checkpoint(kwargs, result)
        return result

    monkeypatch.setattr(qualification, "_PRODUCTION_RUNNER", runner)
    result = qualification.run_qualification(
        baseline_worker_manifest=tmp_path / "baseline.json",
        candidate_worker_manifest=tmp_path / "candidate.json",
        corpus_paths=paths,
        scratch_parent=tmp_path / "scratch",
        output_path=tmp_path / "report.json",
    )
    controller = result["suite"]["controller"]
    if mutation == "raw_suite_kind":
        controller["kind"] = "bounded_sequential_streaming_stt_suite"
    else:
        run = result["suite"]["runs"][0]
        binding = run["binding"]
        if mutation == "source_binding":
            binding["source_binding_sha256"] = "0" * 64
        else:
            run["report"]["metrics"]["evaluations"] = 999
            binding["report_sha256"] = qualification._canonical_sha256(
                run["report"]
            )
        binding.pop("binding_sha256")
        binding["binding_sha256"] = qualification._canonical_sha256(binding)
        matrix = qualification._canonical_sha256(
            [item["binding"] for item in result["suite"]["runs"]]
        )
        controller["matrix_sha256"] = matrix
        controller["completed_matrix_sha256"] = matrix

    with pytest.raises(qualification.QualificationError):
        qualification.validate_qualification_report(result)


@pytest.mark.parametrize("mutation", ("adapter", "resolved_config"))
def test_retained_worker_contract_rejects_coordinated_drift(
    mutation,
    tmp_path,
    monkeypatch,
):
    _lock, paths = _publish_set(tmp_path)

    def runner(workers, corpora, **kwargs):
        result = _fake_result(tuple(workers), tuple(corpora))
        _write_checkpoint(kwargs, result)
        return result

    monkeypatch.setattr(qualification, "_PRODUCTION_RUNNER", runner)
    result = qualification.run_qualification(
        baseline_worker_manifest=tmp_path / "baseline.json",
        candidate_worker_manifest=tmp_path / "candidate.json",
        corpus_paths=paths,
        scratch_parent=tmp_path / "scratch",
        output_path=tmp_path / "report.json",
    )
    changed = result["suite"]["runs"][:4] if mutation == "adapter" else [
        result["suite"]["runs"][0]
    ]
    for run in changed:
        report = run["report"]
        binding = run["binding"]
        if mutation == "adapter":
            report["worker"]["adapter"] = streaming_stt_eval.MOONSHINE_ADAPTER
            binding["worker_adapter"] = streaming_stt_eval.MOONSHINE_ADAPTER
        else:
            report["config"]["chunk_samples"] = 1601
            report["config"].pop("contract_sha256")
            report["config"]["contract_sha256"] = (
                qualification._canonical_sha256(report["config"])
            )
        binding["report_sha256"] = qualification._canonical_sha256(report)
        binding.pop("binding_sha256")
        binding["binding_sha256"] = qualification._canonical_sha256(binding)
    matrix = qualification._canonical_sha256(
        [item["binding"] for item in result["suite"]["runs"]]
    )
    result["suite"]["controller"]["matrix_sha256"] = matrix
    result["suite"]["controller"]["completed_matrix_sha256"] = matrix

    with pytest.raises(qualification.QualificationError):
        qualification.validate_qualification_report(result)


def test_complete_internally_consistent_red_result_stays_red(
    tmp_path,
    monkeypatch,
):
    _lock, paths = _publish_set(tmp_path)

    def run_suite(workers, corpora, **kwargs):
        result = _fake_result(tuple(workers), tuple(corpora))
        result["ok"] = False
        suite = result["suite"]
        suite["ok"] = False
        suite["runs"][0]["report"]["ok"] = False
        suite["runs"][0]["report"]["metrics"]["coverage_complete"] = False
        _rebind_fake_run(result)
        _write_checkpoint(kwargs, result)
        return result

    monkeypatch.setattr(qualification, "_PRODUCTION_RUNNER", run_suite)

    result = qualification.run_qualification(
        baseline_worker_manifest=tmp_path / "baseline.json",
        candidate_worker_manifest=tmp_path / "candidate.json",
        corpus_paths=paths,
        scratch_parent=tmp_path / "scratch",
        output_path=tmp_path / "report.json",
    )

    assert result["ok"] is False
    assert result["execution_complete"] is True
    assert result["quality_decision"] == "not_evaluated"


def test_default_fixture_and_suite_controllers_run_exact_two_by_four(
    tmp_path,
    monkeypatch,
):
    _lock, paths = _publish_set(tmp_path)
    calls: list[tuple[int, int]] = []
    path_index = {path.resolve(): index for index, path in enumerate(paths)}

    def benchmark(worker, corpus, **kwargs):
        corpus_index = path_index[Path(corpus).resolve()]
        worker_index = 0 if Path(worker).name == "baseline.json" else 1
        calls.append((worker_index, corpus_index))
        tags = tuple(kwargs["stratum_tags"])
        report = _aggregate_report(
            Path(worker),
            Path(corpus),
            worker_digest=_worker_digest(worker),
            corpus_digest=load_corpus(corpus).digest,
            stratum_tags=tags,
        )
        report["metrics"] = _complete_metrics(
            _FAKE_STRATA[corpus_index],
            repeats=1,
        )
        report["evaluator"] = streaming_stt_eval._evaluator_binding(
            "fake-json-v1"
        )
        report["evidence"] = streaming_stt_eval._evidence_binding(
            "fake-json-v1",
            pace="burst",
            stratum_tags=tags,
        )
        config: dict[str, object] = {
            "chunk_samples": 1600,
            "pace": "burst",
            "partial_interval_ms": 200,
            "tail_padding_samples": 0,
            "repeats": 1,
            "stratum_tags": list(tags),
        }
        config["contract_sha256"] = qualification._canonical_sha256(config)
        report["config"] = config
        return report

    monkeypatch.setattr(streaming_stt_eval, "run_benchmark", benchmark)
    result = qualification.run_qualification(
        baseline_worker_manifest=tmp_path / "baseline.json",
        candidate_worker_manifest=tmp_path / "candidate.json",
        corpus_paths=reversed(paths),
        scratch_parent=tmp_path / "scratch",
        output_path=tmp_path / "report.json",
        repeats=1,
    )

    assert calls == [
        (worker_index, corpus_index)
        for worker_index in range(2)
        for corpus_index in range(4)
    ]
    assert result["ok"] is True
    assert result["suite"]["controller"]["runs"] == 8


def test_mixed_faster_whisper_and_cpp_two_by_four_binds_endpoint_shape(
    tmp_path,
    monkeypatch,
):
    _lock, paths = _publish_set(tmp_path)

    def manifest(path):
        selected = Path(path)
        candidate = selected.name == "candidate.json"
        return SimpleNamespace(
            adapter=(
                qualification.PARAKEET_CPP_ADAPTER
                if candidate
                else streaming_stt_eval.FASTER_WHISPER_ENDPOINT_ADAPTER
            ),
            adapter_config=(
                ParakeetCppConfig()
                if candidate
                else FasterWhisperEndpointConfig()
            ),
            digest=_worker_digest(selected),
        )

    def runner(workers, corpora, **kwargs):
        result = _fake_result(tuple(workers), tuple(corpora))
        _write_checkpoint(kwargs, result)
        return result

    monkeypatch.setattr(qualification, "load_worker_manifest", manifest)
    monkeypatch.setattr(qualification, "_PRODUCTION_RUNNER", runner)
    result = qualification.run_qualification(
        baseline_worker_manifest=tmp_path / "baseline.json",
        candidate_worker_manifest=tmp_path / "candidate.json",
        corpus_paths=paths,
        scratch_parent=tmp_path / "scratch",
        output_path=tmp_path / "report.json",
    )

    assert result["suite"]["controller"]["kind"] == qualification._SAFE_SUITE_KIND
    for run in result["suite"]["runs"]:
        report = run["report"]
        candidate = run["binding"]["worker_index"] == 1
        assert report["worker"]["adapter"] == (
            qualification.PARAKEET_CPP_ADAPTER
            if candidate
            else streaming_stt_eval.FASTER_WHISPER_ENDPOINT_ADAPTER
        )
        assert ("endpointing" in report["metrics"]) is candidate
        assert (
            "aggregate_rtf_scope" in report["metrics"]["throughput"]
        ) is candidate
    assert qualification.validate_qualification_report(result) == result


@pytest.mark.parametrize(
    "mutation",
    ("missing", "unknown", "wrong_count", "reordered"),
)
def test_lock_selection_strata_fail_closed(mutation, tmp_path):
    _lock, paths = _publish_set(tmp_path)
    fixture_set = fixture.validate_fixture_set(paths)
    source = fixture_set.lock.sources[0]
    rows = list(source.selection_recipe["strata"])
    if mutation == "missing":
        rows = rows[:-1]
    elif mutation == "unknown":
        rows[-1] = {"id": "unknown", "cases": rows[-1]["cases"]}
    elif mutation == "wrong_count":
        rows[-1] = {"id": rows[-1]["id"], "cases": rows[-1]["cases"] + 1}
    else:
        rows = list(reversed(rows))
    changed_source = replace(
        source,
        selection_recipe={**source.selection_recipe, "strata": rows},
    )
    changed_lock = replace(
        fixture_set.lock,
        sources=(changed_source, *fixture_set.lock.sources[1:]),
    )

    with pytest.raises(qualification.QualificationError):
        qualification.canonical_stratum_plan(
            replace(fixture_set, lock=changed_lock)
        )


def test_reordered_slots_fail_closed(tmp_path):
    _lock, paths = _publish_set(tmp_path)
    fixture_set = fixture.validate_fixture_set(paths)
    slots = list(fixture_set.lock.slots)
    slots[0], slots[1] = slots[1], slots[0]

    with pytest.raises(qualification.QualificationError):
        qualification.canonical_stratum_plan(
            replace(fixture_set, lock=replace(fixture_set.lock, slots=tuple(slots)))
        )


def test_changed_slot_distribution_fails_closed(tmp_path):
    _lock, paths = _publish_set(tmp_path)
    fixture_set = fixture.validate_fixture_set(paths)
    slots = list(fixture_set.lock.slots)
    slots[2] = replace(slots[2], stratum="task_01")

    with pytest.raises(qualification.QualificationError):
        qualification.canonical_stratum_plan(
            replace(fixture_set, lock=replace(fixture_set.lock, slots=tuple(slots)))
        )


def test_missing_case_stratum_fails_before_runner(tmp_path, monkeypatch):
    _lock, paths = _publish_set(tmp_path)
    real_load = qualification.load_corpus
    called = False

    def load_without_first_task(path):
        corpus = real_load(path)
        if corpus.path == paths[0]:
            case = corpus.cases[0]
            changed = replace(
                case,
                tags=tuple(tag for tag in case.tags if tag != "task_00"),
            )
            return replace(corpus, cases=(changed, *corpus.cases[1:]))
        return corpus

    def runner(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(qualification, "load_corpus", load_without_first_task)
    monkeypatch.setattr(qualification, "_PRODUCTION_RUNNER", runner)
    with pytest.raises(qualification.QualificationError):
        qualification.run_qualification(
            baseline_worker_manifest=tmp_path / "baseline.json",
            candidate_worker_manifest=tmp_path / "candidate.json",
            corpus_paths=paths,
            scratch_parent=tmp_path / "scratch",
            output_path=tmp_path / "report.json",
        )
    assert called is False


def test_extra_sibling_stratum_fails_before_runner(tmp_path, monkeypatch):
    _lock, paths = _publish_set(tmp_path)
    real_load = qualification.load_corpus
    called = False

    def load_with_sibling_task(path):
        corpus = real_load(path)
        if corpus.path == paths[0]:
            case = corpus.cases[0]
            changed = replace(case, tags=(*case.tags, "task_01"))
            return replace(corpus, cases=(changed, *corpus.cases[1:]))
        return corpus

    def runner(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(qualification, "load_corpus", load_with_sibling_task)
    monkeypatch.setattr(qualification, "_PRODUCTION_RUNNER", runner)
    with pytest.raises(qualification.QualificationError):
        qualification.run_qualification(
            baseline_worker_manifest=tmp_path / "baseline.json",
            candidate_worker_manifest=tmp_path / "candidate.json",
            corpus_paths=paths,
            scratch_parent=tmp_path / "scratch",
            output_path=tmp_path / "report.json",
        )
    assert called is False


def test_same_worker_role_path_fails_before_runner(tmp_path, monkeypatch):
    _lock, paths = _publish_set(tmp_path)
    called = False

    def runner(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError

    worker = tmp_path / "same.json"
    monkeypatch.setattr(qualification, "_PRODUCTION_RUNNER", runner)
    with pytest.raises(qualification.QualificationError):
        qualification.run_qualification(
            baseline_worker_manifest=worker,
            candidate_worker_manifest=worker,
            corpus_paths=paths,
            scratch_parent=tmp_path / "scratch",
            output_path=tmp_path / "report.json",
        )
    assert called is False


def test_worker_role_symlink_alias_fails_before_runner(tmp_path, monkeypatch):
    _lock, paths = _publish_set(tmp_path)
    worker = tmp_path / "worker.json"
    worker.write_text("{}", encoding="utf-8")
    alias = tmp_path / "worker-alias.json"
    alias.symlink_to(worker)
    called = False

    def runner(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(qualification, "_PRODUCTION_RUNNER", runner)
    with pytest.raises(qualification.QualificationError):
        qualification.run_qualification(
            baseline_worker_manifest=worker,
            candidate_worker_manifest=alias,
            corpus_paths=paths,
            scratch_parent=tmp_path / "scratch",
            output_path=tmp_path / "report.json",
        )
    assert called is False


def test_same_worker_manifest_digest_fails_before_runner(tmp_path, monkeypatch):
    _lock, paths = _publish_set(tmp_path)
    monkeypatch.setattr(
        qualification,
        "load_worker_manifest",
        lambda _path: SimpleNamespace(
            adapter="fake-json-v1",
            digest="3" * 64,
        ),
    )
    called = False

    def runner(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(qualification, "_PRODUCTION_RUNNER", runner)
    with pytest.raises(qualification.QualificationError):
        qualification.run_qualification(
            baseline_worker_manifest=tmp_path / "baseline.json",
            candidate_worker_manifest=tmp_path / "candidate.json",
            corpus_paths=paths,
            scratch_parent=tmp_path / "scratch",
            output_path=tmp_path / "report.json",
        )
    assert called is False


def test_implementation_change_during_run_fails_closed(tmp_path, monkeypatch):
    _lock, paths = _publish_set(tmp_path)
    digests = iter(("1" * 64, "2" * 64))
    monkeypatch.setattr(
        qualification,
        "_implementation_sha256",
        lambda: next(digests),
    )

    output = tmp_path / "report.json"

    def runner(workers, corpora, **kwargs):
        inner = Path(kwargs["checkpoint_path"])
        inner.write_text('{"complete":true,"ok":true}\n', encoding="utf-8")
        inner.chmod(0o600)
        return _fake_result(tuple(workers), tuple(corpora))

    monkeypatch.setattr(qualification, "_PRODUCTION_RUNNER", runner)
    with pytest.raises(qualification.QualificationError):
        qualification.run_qualification(
            baseline_worker_manifest=tmp_path / "baseline.json",
            candidate_worker_manifest=tmp_path / "candidate.json",
            corpus_paths=paths,
            scratch_parent=tmp_path / "scratch",
            output_path=output,
        )
    assert output.exists() is False
    inner = tmp_path / "scratch" / ".canonical-strata-inner-suite-checkpoint.json"
    assert json.loads(inner.read_text(encoding="utf-8"))["ok"] is True


def test_checkpoint_snapshot_accepts_bounded_short_reads(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    scratch.chmod(0o700)
    checkpoint = scratch / qualification._INNER_CHECKPOINT_NAME
    payload = {"ok": True, "rows": list(range(256))}
    checkpoint.write_bytes(qualification._canonical_json_bytes(payload) + b"\n")
    checkpoint.chmod(0o600)
    identity = qualification._entry_identity(scratch.lstat())
    real_read = qualification.os.read

    def short_read(descriptor, maximum):
        return real_read(descriptor, min(maximum, 7))

    monkeypatch.setattr(qualification.os, "read", short_read)
    assert qualification._checkpoint_snapshot(
        scratch=scratch,
        checkpoint=checkpoint,
        scratch_identity=identity,
        required=True,
    ) == payload


@pytest.mark.parametrize("target", ("scratch", "checkpoint"))
def test_private_evidence_is_rechecked_after_report_reconstruction(
    target,
    tmp_path,
    monkeypatch,
):
    _lock, paths = _publish_set(tmp_path)
    real_validate = qualification.validate_qualification_report

    def runner(workers, corpora, **kwargs):
        result = _fake_result(tuple(workers), tuple(corpora))
        _write_checkpoint(kwargs, result)
        return result

    def validate_then_mutate(value):
        result = real_validate(value)
        selected = (
            tmp_path / "scratch"
            if target == "scratch"
            else tmp_path / "scratch" / qualification._INNER_CHECKPOINT_NAME
        )
        selected.chmod(0o750 if target == "scratch" else 0o640)
        return result

    monkeypatch.setattr(qualification, "_PRODUCTION_RUNNER", runner)
    monkeypatch.setattr(
        qualification,
        "validate_qualification_report",
        validate_then_mutate,
    )
    with pytest.raises(qualification.QualificationError):
        qualification.run_qualification(
            baseline_worker_manifest=tmp_path / "baseline.json",
            candidate_worker_manifest=tmp_path / "candidate.json",
            corpus_paths=paths,
            scratch_parent=tmp_path / "scratch",
            output_path=tmp_path / "report.json",
        )


@pytest.mark.parametrize("target", ("scratch", "checkpoint"))
def test_private_scratch_and_checkpoint_are_rechecked_after_runner(
    target,
    tmp_path,
    monkeypatch,
):
    _lock, paths = _publish_set(tmp_path)

    def runner(workers, corpora, **kwargs):
        result = _fake_result(tuple(workers), tuple(corpora))
        _write_checkpoint(kwargs, result)
        selected = (
            Path(kwargs["scratch_parent"])
            if target == "scratch"
            else Path(kwargs["checkpoint_path"])
        )
        selected.chmod(0o750 if target == "scratch" else 0o640)
        return result

    monkeypatch.setattr(qualification, "_PRODUCTION_RUNNER", runner)
    with pytest.raises(qualification.QualificationError):
        qualification.run_qualification(
            baseline_worker_manifest=tmp_path / "baseline.json",
            candidate_worker_manifest=tmp_path / "candidate.json",
            corpus_paths=paths,
            scratch_parent=tmp_path / "scratch",
            output_path=tmp_path / "report.json",
        )


@pytest.mark.parametrize("failure", (SystemExit(7), KeyboardInterrupt()))
def test_runner_baseexception_still_revalidates_private_scratch(
    failure,
    tmp_path,
    monkeypatch,
):
    _lock, paths = _publish_set(tmp_path)
    real_snapshot = qualification._checkpoint_snapshot
    observed = 0

    def snapshot(**kwargs):
        nonlocal observed
        observed += 1
        return real_snapshot(**kwargs)

    def runner(_workers, _corpora, **kwargs):
        checkpoint = Path(kwargs["checkpoint_path"])
        checkpoint.write_text('{"ok":false}\n', encoding="ascii")
        checkpoint.chmod(0o600)
        Path(kwargs["scratch_parent"]).chmod(0o750)
        raise failure

    monkeypatch.setattr(qualification, "_checkpoint_snapshot", snapshot)
    monkeypatch.setattr(qualification, "_PRODUCTION_RUNNER", runner)
    with pytest.raises(qualification.QualificationError) as error:
        qualification.run_qualification(
            baseline_worker_manifest=tmp_path / "baseline.json",
            candidate_worker_manifest=tmp_path / "candidate.json",
            corpus_paths=paths,
            scratch_parent=tmp_path / "scratch",
            output_path=tmp_path / "report.json",
        )

    assert observed == 1
    assert str(error.value) == ""


def test_existing_output_fails_before_runner(tmp_path, monkeypatch):
    _lock, paths = _publish_set(tmp_path)
    output = tmp_path / "report.json"
    output.write_text('{"ok":true}\n', encoding="utf-8")
    output.chmod(0o600)
    called = False

    def runner(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(qualification, "_PRODUCTION_RUNNER", runner)
    with pytest.raises(qualification.QualificationError):
        qualification.run_qualification(
            baseline_worker_manifest=tmp_path / "baseline.json",
            candidate_worker_manifest=tmp_path / "candidate.json",
            corpus_paths=paths,
            scratch_parent=tmp_path / "scratch",
            output_path=output,
        )
    assert called is False


def test_bound_corpus_cannot_be_output(tmp_path):
    _lock, paths = _publish_set(tmp_path)
    with pytest.raises(qualification.QualificationError):
        qualification.run_qualification(
            baseline_worker_manifest=tmp_path / "baseline.json",
            candidate_worker_manifest=tmp_path / "candidate.json",
            corpus_paths=paths,
            scratch_parent=tmp_path / "scratch",
            output_path=paths[0],
        )


def test_scratch_inside_corpus_evidence_fails_without_creation(tmp_path):
    _lock, paths = _publish_set(tmp_path)
    scratch = paths[0].parent / "qualification-scratch"
    with pytest.raises(qualification.QualificationError):
        qualification.run_qualification(
            baseline_worker_manifest=tmp_path / "baseline.json",
            candidate_worker_manifest=tmp_path / "candidate.json",
            corpus_paths=paths,
            scratch_parent=scratch,
            output_path=tmp_path / "report.json",
        )
    assert scratch.exists() is False


@pytest.mark.parametrize("nested", (False, True))
def test_scratch_cannot_replace_or_descend_from_future_output(tmp_path, nested):
    _lock, paths = _publish_set(tmp_path)
    output = tmp_path / "future-report"
    scratch = output / "scratch" if nested else output
    with pytest.raises(qualification.QualificationError):
        qualification.run_qualification(
            baseline_worker_manifest=tmp_path / "baseline.json",
            candidate_worker_manifest=tmp_path / "candidate.json",
            corpus_paths=paths,
            scratch_parent=scratch,
            output_path=output,
        )
    assert output.exists() is False


def test_paths_under_any_git_worktree_fail_before_runner(tmp_path, monkeypatch):
    _lock, paths = _publish_set(tmp_path)
    other_checkout = tmp_path / "other-checkout"
    other_checkout.mkdir(mode=0o700)
    (other_checkout / ".git").write_text("gitdir: elsewhere\n", encoding="ascii")
    private = other_checkout / "private"
    private.mkdir(mode=0o700)
    called = False

    def runner(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(qualification, "_PRODUCTION_RUNNER", runner)
    with pytest.raises(qualification.QualificationError):
        qualification.run_qualification(
            baseline_worker_manifest=tmp_path / "baseline.json",
            candidate_worker_manifest=tmp_path / "candidate.json",
            corpus_paths=paths,
            scratch_parent=tmp_path / "scratch",
            output_path=private / "report.json",
        )
    assert called is False
    assert (tmp_path / "scratch").exists() is False


@pytest.mark.parametrize("placement", ("scratch_runtime", "output_provision"))
def test_worker_evidence_trees_cannot_receive_outputs(
    placement,
    tmp_path,
    monkeypatch,
):
    _lock, paths = _publish_set(tmp_path)
    provision = tmp_path / "worker-provision"
    provision.mkdir(mode=0o700)
    runtime = tmp_path / "worker-runtime"
    runtime.mkdir(mode=0o700)
    receipt = provision / "runtime-receipt.json"
    receipt.write_text(
        json.dumps({"schema_version": 1, "root": str(runtime), "files": []}),
        encoding="ascii",
    )
    receipt.chmod(0o600)
    receipt_bytes = receipt.read_bytes()

    def manifest(path):
        selected = Path(path)
        return SimpleNamespace(
            adapter="fake-json-v1",
            digest=_worker_digest(selected),
            path=provision / f"{selected.stem}-manifest.json",
            worker=SimpleNamespace(path=provision / "worker.py"),
            python=SimpleNamespace(path=provision / "venv" / "bin" / "python"),
            artifacts=(
                SimpleNamespace(
                    name="runtime-receipt",
                    path=receipt,
                    sha256=hashlib.sha256(receipt_bytes).hexdigest(),
                    size_bytes=len(receipt_bytes),
                ),
            ),
        )

    (provision / "venv").mkdir(mode=0o700)
    monkeypatch.setattr(qualification, "load_worker_manifest", manifest)
    scratch = (
        runtime / "qualification-scratch"
        if placement == "scratch_runtime"
        else tmp_path / "scratch"
    )
    output = (
        tmp_path / "report.json"
        if placement == "scratch_runtime"
        else provision / "report.json"
    )
    with pytest.raises(qualification.QualificationError):
        qualification.run_qualification(
            baseline_worker_manifest=tmp_path / "baseline.json",
            candidate_worker_manifest=tmp_path / "candidate.json",
            corpus_paths=paths,
            scratch_parent=scratch,
            output_path=output,
        )
    assert tuple(runtime.iterdir()) == ()
    assert scratch.exists() is False
    assert output.exists() is False


def test_final_report_is_private_and_no_clobber(tmp_path):
    output = tmp_path / "report.json"
    payload = {"ok": True, "development_only": True, "promotional": False}
    digest = qualification.write_new_report(output, payload)

    assert output.stat().st_mode & 0o777 == 0o600
    assert json.loads(output.read_text(encoding="ascii")) == payload
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
    with pytest.raises(qualification.QualificationError):
        qualification.write_new_report(output, payload)


def test_final_report_readback_accepts_bounded_short_reads(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    payload = {"ok": True, "rows": list(range(256))}
    real_read = qualification.os.read

    def short_read(descriptor, maximum):
        return real_read(descriptor, min(maximum, 7))

    monkeypatch.setattr(qualification.os, "read", short_read)
    digest = qualification.write_new_report(output, payload)

    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()


@pytest.mark.parametrize("failure", (SystemExit(7), KeyboardInterrupt()))
def test_final_report_baseexception_after_link_leaves_no_artifact(
    failure,
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "report.json"
    real_link = qualification.os.link

    def link_then_raise(*args, **kwargs):
        real_link(*args, **kwargs)
        raise failure

    monkeypatch.setattr(qualification.os, "link", link_then_raise)
    with pytest.raises(qualification.QualificationError):
        qualification.write_new_report(output, {"ok": True})

    assert output.exists() is False
    assert not any(
        entry.name.startswith(qualification._REPORT_TEMP_PREFIX)
        for entry in tmp_path.iterdir()
    )


@pytest.mark.parametrize("failure", ("partial_write", "file_fsync"))
def test_final_report_publication_failures_leave_no_artifact(
    failure,
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "report.json"
    payload = {"ok": True, "development_only": True, "promotional": False}
    if failure == "partial_write":
        real_write = qualification.os.write
        writes = 0

        def fail_after_partial(descriptor, data):
            nonlocal writes
            writes += 1
            if writes == 1:
                partial = max(1, len(data) // 2)
                return real_write(descriptor, data[:partial])
            raise OSError("injected write failure")

        monkeypatch.setattr(qualification.os, "write", fail_after_partial)
    else:

        def fail_fsync(_descriptor):
            raise OSError("injected fsync failure")

        monkeypatch.setattr(qualification.os, "fsync", fail_fsync)

    with pytest.raises(qualification.QualificationError):
        qualification.write_new_report(output, payload)

    assert output.exists() is False
    assert not any(
        entry.name.startswith(qualification._REPORT_TEMP_PREFIX)
        for entry in tmp_path.iterdir()
    )


def test_first_temporary_fstat_failure_is_cleanup_safe(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    real_open = qualification.os.open
    real_fstat = qualification.os.fstat
    temporary_descriptor: int | None = None
    injected = False

    def record_open(path, *args, **kwargs):
        nonlocal temporary_descriptor
        descriptor = real_open(path, *args, **kwargs)
        if isinstance(path, str) and path.startswith(qualification._REPORT_TEMP_PREFIX):
            temporary_descriptor = descriptor
        return descriptor

    def fail_first_temporary_fstat(descriptor):
        nonlocal injected
        if descriptor == temporary_descriptor and not injected:
            injected = True
            raise OSError("injected first temp fstat failure")
        return real_fstat(descriptor)

    monkeypatch.setattr(qualification.os, "open", record_open)
    monkeypatch.setattr(qualification.os, "fstat", fail_first_temporary_fstat)
    with pytest.raises(qualification.QualificationError):
        qualification.write_new_report(
            output,
            {"ok": True, "development_only": True, "promotional": False},
        )

    assert injected is True
    assert output.exists() is False
    assert not any(
        entry.name.startswith(qualification._REPORT_TEMP_PREFIX)
        for entry in tmp_path.iterdir()
    )


@pytest.mark.parametrize(
    "failure",
    (
        "reordered",
        "worker_digest",
        "corpus_digest",
        "stratum_config",
        "stratum_count",
        "stratum_evaluations",
        "stratum_coverage",
        "fixture",
        "top_ok",
        "suite_ok",
        "run_ok",
        "path_leak",
    ),
)
def test_returned_matrix_and_privacy_are_revalidated(
    failure,
    tmp_path,
    monkeypatch,
):
    _lock, paths = _publish_set(tmp_path)

    def runner(workers, corpora, **kwargs):
        result = _fake_result(tuple(workers), tuple(corpora))
        suite = result["suite"]
        assert isinstance(suite, dict)
        if failure == "reordered":
            suite["runs"] = list(reversed(suite["runs"]))
        elif failure == "worker_digest":
            suite["runs"][0]["report"]["worker"]["manifest_sha256"] = "0" * 64
            suite["runs"][0]["binding"]["worker_manifest_sha256"] = "0" * 64
            _rebind_fake_run(result)
        elif failure == "corpus_digest":
            suite["runs"][0]["report"]["corpus"]["manifest_sha256"] = "0" * 64
            suite["runs"][0]["binding"]["corpus_manifest_sha256"] = "0" * 64
            _rebind_fake_run(result)
        elif failure == "stratum_config":
            suite["config"]["strata"]["corpora"][0]["tags"] = ["task_00"]
        elif failure == "stratum_count":
            suite["runs"][0]["report"]["metrics"]["strata"][0]["cases"] = 4
            _rebind_fake_run(result)
        elif failure == "stratum_evaluations":
            suite["runs"][0]["report"]["metrics"]["strata"][0]["metrics"][
                "evaluations"
            ] = 999
            _rebind_fake_run(result)
        elif failure == "stratum_coverage":
            suite["runs"][0]["report"]["metrics"]["strata"][0]["metrics"][
                "coverage_complete"
            ] = False
            _rebind_fake_run(result)
        elif failure == "fixture":
            result["fixture"]["cases"] = 95
        elif failure == "top_ok":
            result["ok"] = False
        elif failure == "suite_ok":
            suite["ok"] = False
        elif failure == "run_ok":
            suite["runs"][0]["report"]["ok"] = False
            _rebind_fake_run(result)
        else:
            suite["runs"][0]["report"]["metrics"]["private_path"] = str(
                tmp_path / "canary"
            )
            _rebind_fake_run(result)
        _write_checkpoint(kwargs, result)
        return result

    monkeypatch.setattr(qualification, "_PRODUCTION_RUNNER", runner)
    with pytest.raises(qualification.QualificationError):
        qualification.run_qualification(
            baseline_worker_manifest=tmp_path / "baseline.json",
            candidate_worker_manifest=tmp_path / "candidate.json",
            corpus_paths=paths,
            scratch_parent=tmp_path / "scratch",
            output_path=tmp_path / "report.json",
        )


def test_cli_success_prints_exact_private_report_digest(
    tmp_path,
    monkeypatch,
    capsys,
):
    output = tmp_path / "report.json"
    report = {
        "ok": True,
        "execution_complete": True,
        "quality_decision": "not_evaluated",
        "development_only": True,
        "promotional": False,
    }
    monkeypatch.setattr(
        qualification,
        "run_qualification",
        lambda **_kwargs: report,
    )

    code = qualification.main(
        [
            "--baseline-worker-manifest",
            str(tmp_path / "baseline.json"),
            "--candidate-worker-manifest",
            str(tmp_path / "candidate.json"),
            "--corpus",
            str(tmp_path / "corpus.json"),
            "--scratch-root",
            str(tmp_path / "scratch"),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert code == 0
    assert captured.err == ""
    assert output.stat().st_mode & 0o777 == 0o600
    assert summary["report_sha256"] == hashlib.sha256(
        output.read_bytes()
    ).hexdigest()
    assert summary["report_written"] is True


def test_cli_requires_exact_two_roles_and_hides_paths(tmp_path, capsys):
    canary = tmp_path / "private-baseline.json"
    assert (
        qualification.main(
            [
                "--baseline-worker-manifest",
                str(canary),
                "--corpus",
                str(tmp_path / "corpus.json"),
                "--scratch-root",
                str(tmp_path / "scratch"),
                "--output",
                str(tmp_path / "report.json"),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "error": "public conversation qualification failed",
        "ok": False,
    }
    assert str(canary) not in captured.err
