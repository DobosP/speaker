from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json

import pytest

from tools.conversation_eval.flow_schema import (
    FLOW_PROFILE,
    FLOW_REQUIRED_RUNS,
    FLOW_SCENARIO_IDS,
    FLOW_SCENARIO_SET_VERSION,
    FLOW_SCHEMA_VERSION,
    FlowCheck,
    FlowScenarioResult,
    build_flow_report,
)
from tools.conversation_eval.schema import TraceEvent


def _result(
    scenario_id: str,
    run_index: int,
    *,
    duration_ms: float = 2.0,
    elapsed_ms: float = 1.0,
) -> FlowScenarioResult:
    return FlowScenarioResult(
        scenario_id=scenario_id,
        description=f"flow evidence for {scenario_id}",
        run_index=run_index,
        passed=True,
        duration_ms=duration_ms,
        checks=(FlowCheck("causal_order", True, ("start", "end"), ("start", "end")),),
        trace=(
            TraceEvent(
                sequence=1,
                elapsed_ms=elapsed_ms,
                kind="flow.complete",
                turn_id=run_index,
                payload={
                    "scenario_id": scenario_id,
                    "checks_ok": True,
                    "error": "",
                    "fixture_transcript_classification": "plumbing_only",
                    "stages": ("start", "end"),
                },
            ),
        ),
    )


def _complete_results() -> tuple[FlowScenarioResult, ...]:
    return tuple(
        _result(scenario_id, run_index)
        for scenario_id in FLOW_SCENARIO_IDS
        for run_index in range(1, FLOW_REQUIRED_RUNS + 1)
    )


def _replace_first(
    results: tuple[FlowScenarioResult, ...],
    replacement: FlowScenarioResult,
) -> tuple[FlowScenarioResult, ...]:
    return (replacement, *results[1:])


def test_flow_contract_constants_are_exact() -> None:
    assert FLOW_SCHEMA_VERSION == 1
    assert FLOW_SCENARIO_SET_VERSION == 1
    assert FLOW_REQUIRED_RUNS == 3
    assert FLOW_PROFILE == "headless_scripted_runtime"
    assert FLOW_SCENARIO_IDS == (
        "turn_incremental",
        "barge_recovery",
        "slow_tool_complete_followup",
        "stop_restart_fence",
    )


def test_flow_dataclasses_are_immutable_and_as_dict_is_json_safe() -> None:
    result = _result(FLOW_SCENARIO_IDS[0], 1)

    with pytest.raises(FrozenInstanceError):
        result.passed = False  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.checks[0].ok = False  # type: ignore[misc]

    serialized = result.as_dict()
    assert serialized["checks"][0]["expected"] == ["start", "end"]
    assert serialized["trace"][0]["payload"]["stages"] == ["start", "end"]
    json.dumps(serialized, allow_nan=False, sort_keys=True)


def test_complete_flow_report_passes_with_exact_canonical_coverage() -> None:
    results = _complete_results()
    report = build_flow_report(tuple(reversed(results)), requested_runs=3)

    assert report["schema_version"] == 1
    assert report["scenario_set_version"] == 1
    assert report["profile"] == "headless_scripted_runtime"
    assert "not acoustic quality evidence" in report["scope"]
    assert report["coverage"] == {
        "required_scenario_ids": list(FLOW_SCENARIO_IDS),
        "required_run_indices": [1, 2, 3],
        "expected_result_count": 12,
        "observed_result_count": 12,
        "observed": {
            scenario_id: [1, 2, 3] for scenario_id in FLOW_SCENARIO_IDS
        },
        "unknown_scenario_ids": [],
        "invalid_run_indices": [],
        "duplicate_scenario_runs": [],
        "missing_scenario_runs": [],
        "requested_runs_exact": True,
        "exact": True,
    }
    assert report["gate"]["passed"] is True
    assert report["passed"] is True
    assert report["summary"] == {
        "scenarios": 4,
        "runs_per_scenario": 3,
        "passed_results": 12,
        "total_results": 12,
        "pass_at_1": True,
        "pass_power_3": True,
    }
    assert [
        (row["scenario_id"], row["run_index"]) for row in report["results"]
    ] == [
        (scenario_id, run_index)
        for scenario_id in FLOW_SCENARIO_IDS
        for run_index in (1, 2, 3)
    ]
    assert report == build_flow_report(results, requested_runs=3)
    json.dumps(report, allow_nan=False, sort_keys=True)


def test_evidence_matrix_never_promotes_fixture_text_to_audio_evidence() -> None:
    report = build_flow_report(_complete_results(), requested_runs=3)
    matrix = report["evidence_matrix"]

    assert report["fixture_transcripts"] == {
        "classification": "plumbing_only",
        "stt_quality_evidence": False,
    }
    assert matrix["headless_runtime_plumbing"]["status"] == "COVERED"
    assert {
        matrix[area]["status"]
        for area in {
            "typed_turn_endpoint_policy",
            "scripted_streaming_playback",
            "interruption_recovery",
            "read_tool_lifecycle",
            "synthetic_action_tool_lifecycle",
            "fresh_session_fencing",
            "causal_headless_timing",
        }
    } == {"COVERED"}
    assert matrix["fixture_transcripts"]["status"] == "plumbing_only"
    assert {
        area: row["status"]
        for area, row in matrix.items()
        if area
        in {
            "microphone_native_reader",
            "vad_capture",
            "actual_stt_wer",
            "real_tts_audio_pacing",
            "aec_room_open_speaker",
            "physical_audibility",
            "live_perceived_latency",
            "external_effects",
        }
    } == {
        "microphone_native_reader": "NOT_COVERED",
        "vad_capture": "NOT_COVERED",
        "actual_stt_wer": "NOT_COVERED",
        "real_tts_audio_pacing": "NOT_COVERED",
        "aec_room_open_speaker": "NOT_COVERED",
        "physical_audibility": "NOT_COVERED",
        "live_perceived_latency": "NOT_COVERED",
        "external_effects": "NOT_COVERED",
    }


@pytest.mark.parametrize(
    "poison",
    ("missing", "duplicate", "unknown", "wrong_index", "wrong_requested_runs"),
)
def test_exact_coverage_rejects_every_shape_poison(poison: str) -> None:
    results = _complete_results()
    requested_runs = 3
    if poison == "missing":
        results = results[:-1]
    elif poison == "duplicate":
        results = (*results, results[0])
    elif poison == "unknown":
        results = _replace_first(
            results,
            replace(results[0], scenario_id="unknown_flow"),
        )
    elif poison == "wrong_index":
        results = _replace_first(results, replace(results[0], run_index=4))
    elif poison == "wrong_requested_runs":
        requested_runs = 2

    report = build_flow_report(results, requested_runs=requested_runs)

    assert report["coverage"]["exact"] is False
    assert report["gate"]["exact_coverage"] is False
    assert report["passed"] is False


@pytest.mark.parametrize("poison", ("result", "check", "error", "no_checks", "no_trace"))
def test_gate_rejects_failed_or_absent_result_evidence(poison: str) -> None:
    results = _complete_results()
    first = results[0]
    if poison == "result":
        replacement = replace(first, passed=False)
    elif poison == "check":
        replacement = replace(
            first,
            checks=(FlowCheck("causal_order", False, True, False),),
        )
    elif poison == "error":
        replacement = replace(first, error="synthetic failure")
    elif poison == "no_checks":
        replacement = replace(first, checks=())
    else:
        replacement = replace(first, trace=())

    report = build_flow_report(
        _replace_first(results, replacement),
        requested_runs=3,
    )

    assert report["coverage"]["exact"] is True
    assert report["gate"]["passed"] is False
    assert report["passed"] is False


@pytest.mark.parametrize("duration_ms", (-0.001, float("nan"), float("inf")))
def test_gate_rejects_negative_or_nonfinite_duration_and_stays_json_safe(
    duration_ms: float,
) -> None:
    results = _complete_results()
    poisoned = replace(results[0], duration_ms=duration_ms)
    report = build_flow_report(
        _replace_first(results, poisoned),
        requested_runs=3,
    )

    assert report["gate"]["durations_finite_nonnegative"] is False
    assert report["passed"] is False
    json.dumps(report, allow_nan=False, sort_keys=True)


@pytest.mark.parametrize("elapsed_ms", (-0.001, float("nan"), float("inf")))
def test_gate_rejects_negative_or_nonfinite_trace_timestamp_and_stays_json_safe(
    elapsed_ms: float,
) -> None:
    results = _complete_results()
    poisoned = _result(results[0].scenario_id, results[0].run_index, elapsed_ms=elapsed_ms)
    report = build_flow_report(
        _replace_first(results, poisoned),
        requested_runs=3,
    )

    assert report["gate"]["timestamps_finite_nonnegative"] is False
    assert report["passed"] is False
    json.dumps(report, allow_nan=False, sort_keys=True)


@pytest.mark.parametrize("poison", ("duplicate_sequence", "gap", "time_reversal", "after_duration"))
def test_gate_rejects_noncausal_trace_shape(poison: str) -> None:
    results = _complete_results()
    first = results[0]
    if poison == "duplicate_sequence":
        extra = replace(first.trace[0], elapsed_ms=1.5)
    elif poison == "gap":
        extra = replace(first.trace[0], sequence=3, elapsed_ms=1.5)
    elif poison == "time_reversal":
        extra = replace(first.trace[0], sequence=2, elapsed_ms=0.5)
    else:
        extra = replace(first.trace[0], sequence=2, elapsed_ms=3.0)
    poisoned = replace(first, trace=(*first.trace, extra))

    report = build_flow_report(
        _replace_first(results, poisoned),
        requested_runs=3,
    )

    assert report["gate"]["passed"] is False
    assert report["passed"] is False


@pytest.mark.parametrize(
    "poison",
    ("run_index", "scenario_id", "checks_ok", "error", "duplicate_terminal"),
)
def test_gate_rejects_relabelled_or_unbound_result_trace(poison: str) -> None:
    results = _complete_results()
    first = results[0]
    terminal = first.trace[0]
    if poison == "run_index":
        replacement = replace(first, run_index=2)
    elif poison == "scenario_id":
        replacement = replace(
            first,
            trace=(
                replace(
                    terminal,
                    payload={**terminal.payload, "scenario_id": "barge_recovery"},
                ),
            ),
        )
    elif poison == "checks_ok":
        replacement = replace(
            first,
            trace=(
                replace(
                    terminal,
                    payload={**terminal.payload, "checks_ok": False},
                ),
            ),
        )
    elif poison == "error":
        replacement = replace(
            first,
            trace=(
                replace(
                    terminal,
                    payload={**terminal.payload, "error": "different"},
                ),
            ),
        )
    else:
        replacement = replace(
            first,
            duration_ms=3.0,
            trace=(terminal, replace(terminal, sequence=2, elapsed_ms=2.0)),
        )

    report = build_flow_report(
        _replace_first(results, replacement),
        requested_runs=3,
    )

    assert report["gate"]["trace_result_binding"] is False
    assert report["passed"] is False
