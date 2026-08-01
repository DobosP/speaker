from __future__ import annotations

import math

import pytest

import tools.conversation_eval.flow as conversation_flow
from tools.conversation_eval.flow import run_flow_scenario, run_flow_suite
from tools.conversation_eval.flow_schema import (
    FLOW_SCENARIO_IDS,
    FlowCheck,
    FlowScenarioResult,
    build_flow_report,
)
from tools.conversation_eval.runner import safe_config
from tools.conversation_eval.schema import TraceEvent


@pytest.fixture(scope="module")
def evaluation_config() -> dict:
    return safe_config()


@pytest.fixture(scope="module")
def flow_results(evaluation_config: dict) -> dict[str, FlowScenarioResult]:
    return {
        scenario_id: run_flow_scenario(
            scenario_id,
            config=evaluation_config,
            run_index=1,
        )
        for scenario_id in FLOW_SCENARIO_IDS
    }


@pytest.fixture(scope="module")
def flow_report_results(
    evaluation_config: dict,
    flow_results: dict[str, FlowScenarioResult],
) -> tuple[FlowScenarioResult, ...]:
    return tuple(
        flow_results[scenario_id]
        if run_index == 1
        else run_flow_scenario(
            scenario_id,
            config=evaluation_config,
            run_index=run_index,
        )
        for run_index in (1, 2, 3)
        for scenario_id in FLOW_SCENARIO_IDS
    )


def _checks(result: FlowScenarioResult) -> dict[str, FlowCheck]:
    return {check.name: check for check in result.checks}


def _assert_green(result: FlowScenarioResult) -> None:
    failed = {
        check.name: {"expected": check.expected, "actual": check.actual}
        for check in result.checks
        if not check.ok
    }
    assert result.passed, {"error": result.error, "failed": failed}
    assert result.error == ""
    assert result.checks
    assert result.trace
    assert result.trace[-1].kind == "flow.complete"
    assert result.trace[-1].turn_id == result.run_index
    assert result.trace[-1].payload["scenario_id"] == result.scenario_id
    assert result.trace[-1].payload["checks_ok"] is result.passed
    assert result.trace[-1].payload["error"] == result.error
    assert [event.sequence for event in result.trace] == list(
        range(1, len(result.trace) + 1)
    )
    elapsed = [event.elapsed_ms for event in result.trace]
    assert all(math.isfinite(value) and value >= 0.0 for value in elapsed)
    assert elapsed == sorted(elapsed)
    assert math.isfinite(result.duration_ms) and result.duration_ms >= 0.0
    assert elapsed[-1] <= result.duration_ms + 0.001


def _assert_checks_pass(
    result: FlowScenarioResult,
    names: tuple[str, ...],
) -> None:
    checks = _checks(result)
    missing = set(names) - set(checks)
    assert not missing
    assert all(checks[name].ok for name in names)


def test_headless_flow_config_disables_external_effect_surfaces(
    evaluation_config: dict,
) -> None:
    config = conversation_flow._headless_config(
        evaluation_config,
        stream_tts=True,
    )

    assert config["local_only"] is True
    assert config["web_search"]["enabled"] is False
    assert config["web_search"]["base_url"] == ""
    assert config["obsidian"]["enabled"] is False
    assert config["reminders"]["enabled"] is False
    assert config["trusted_apps"] == {"enabled": False, "apps": {}}
    assert config["watch"]["enabled"] is False
    assert config["watch"]["grants"] == []
    assert config["gui_actions"]["enabled"] is False
    assert config["screen_capture"]["enabled"] is False
    assert config["memory"]["backend"] == "inmemory"
    assert config["memory"]["persist_assistant"] is False
    assert config["llm"]["cloud"]["enabled"] is False
    assert config["agent_brain"]["offline"] is True


def test_turn_incremental_uses_typed_revisions_and_receipt_owned_streaming(
    flow_results: dict[str, FlowScenarioResult],
) -> None:
    result = flow_results["turn_incremental"]

    _assert_green(result)
    _assert_checks_pass(
        result,
        (
            "endpoint_hold_created_no_work",
            "first_fragment_before_model_continuation",
            "incremental_fragment_count",
            "exact_final_transcript_text",
            "exact_assistant_answer_query",
            "stale_lower_revision_fenced",
            "receipt_backed_memory",
            "task_identity_complete_and_consistent",
            "actor_children_drained",
            "causal_metrics_finite",
            "playback_fragment_ownership_one_to_one",
        ),
    )


def test_barge_recovery_proves_control_plane_scope_without_acoustic_claim(
    flow_results: dict[str, FlowScenarioResult],
) -> None:
    result = flow_results["barge_recovery"]

    _assert_green(result)
    _assert_checks_pass(
        result,
        (
            "v4.redirect_task_terminals",
            "v4.redirect_playback_outcomes",
            "v4.cancelled_task_tts_after_cut",
            "v4_required_check_names_present",
            "physical_acoustic_barge_in_proven",
            "barge_evidence_scope",
        ),
    )
    checks = _checks(result)
    assert checks["physical_acoustic_barge_in_proven"].actual is False
    assert checks["barge_evidence_scope"].actual == "scripted_control_plane_only"


def test_slow_tool_finishes_before_terminal_and_supports_followup(
    flow_results: dict[str, FlowScenarioResult],
) -> None:
    result = flow_results["slow_tool_complete_followup"]

    _assert_green(result)
    _assert_checks_pass(
        result,
        (
            "latency_ack_before_tool_wait",
            "no_premature_task_terminal",
            "tool_finished_before_task_terminal",
            "followup_receipt_uses_prior_context",
            "capability_invocations_paired",
            "actor_children_drained",
        ),
    )


def test_stop_restart_fences_late_provider_from_fresh_session(
    flow_results: dict[str, FlowScenarioResult],
) -> None:
    result = flow_results["stop_restart_fence"]

    _assert_green(result)
    _assert_checks_pass(
        result,
        (
            "session_a_confirmation_prompt_receipted",
            "session_a_pending_confirmation_exact",
            "session_a_provider_not_started_before_confirmation",
            "session_a_active_tool_while_blocked",
            "session_a_staged_task_identity_exact",
            "session_a_tool_ids_nonempty",
            "synthetic_command_external_effect_performed",
            "session_a_command_provider_called_once",
            "session_a_stop_bounded_before_provider_release",
            "fresh_actor_session_identity",
            "session_b_completed_with_receipt",
            "old_engine_callback_fenced",
            "provider_release_did_not_mutate_session_b",
            "provider_release_produced_no_old_output",
            "session_a_tool_lifecycle_identity_paired",
            "session_a_active_tools_zero_after_release",
            "session_a_children_zero",
            "session_b_children_zero",
            "session_a_stopped",
        ),
    )


def test_runner_converts_scenario_exception_to_nonempty_red_result(
    monkeypatch: pytest.MonkeyPatch,
    evaluation_config: dict,
) -> None:
    def fail(_config, _trace):
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(conversation_flow, "_run_turn_incremental", fail)

    result = run_flow_scenario(
        "turn_incremental",
        config=evaluation_config,
        run_index=7,
    )

    assert result.passed is False
    assert result.error == "RuntimeError: fixture failure"
    assert result.trace
    assert [event.kind for event in result.trace] == [
        "flow.fixture",
        "flow.complete",
    ]
    assert _checks(result)["runner_error"].ok is False


def test_actual_results_form_a_valid_report_under_strict_trace_timing(
    flow_report_results: tuple[FlowScenarioResult, ...],
) -> None:
    report = build_flow_report(flow_report_results, requested_runs=3)

    assert report["passed"] is True
    assert report["coverage"]["observed_result_count"] == 12
    assert report["gate"]["trace_sequences_contiguous"] is True
    assert report["gate"]["trace_elapsed_monotonic"] is True
    assert report["gate"]["trace_within_duration"] is True
    assert report["gate"]["trace_result_binding"] is True


def test_flow_suite_dispatches_every_run_in_stable_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []

    def fake_run(
        scenario_id: str,
        *,
        config,
        run_index: int,
    ) -> FlowScenarioResult:
        del config
        calls.append((scenario_id, run_index))
        if scenario_id == "barge_recovery" and run_index == 2:
            raise RuntimeError("one isolated run failed")
        return FlowScenarioResult(
            scenario_id=scenario_id,
            description="fixture",
            run_index=run_index,
            passed=True,
            duration_ms=0.0,
            checks=(FlowCheck("fixture", True, True, True),),
            trace=(TraceEvent(1, 0.0, "flow.complete"),),
        )

    monkeypatch.setattr(conversation_flow, "run_flow_scenario", fake_run)

    results = run_flow_suite(config={})

    expected = [
        (scenario_id, run_index)
        for run_index in (1, 2, 3)
        for scenario_id in FLOW_SCENARIO_IDS
    ]
    assert len(results) == 12
    assert calls == expected
    assert [(result.scenario_id, result.run_index) for result in results] == expected
    isolated_failure = next(
        result
        for result in results
        if result.scenario_id == "barge_recovery" and result.run_index == 2
    )
    assert isolated_failure.passed is False
    assert isolated_failure.error == "RuntimeError: one isolated run failed"
    assert isolated_failure.trace
    assert isolated_failure.trace[-1].kind == "flow.complete"
    assert results[-1].scenario_id == FLOW_SCENARIO_IDS[-1]
    assert results[-1].run_index == 3


@pytest.mark.parametrize("run_index", [0, -1, True, 1.0])
def test_flow_scenario_rejects_invalid_run_index(run_index) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        run_flow_scenario("turn_incremental", config={}, run_index=run_index)


def test_flow_scenario_rejects_unknown_id() -> None:
    with pytest.raises(ValueError, match="unknown flow scenario"):
        run_flow_scenario("unknown", config={}, run_index=1)


@pytest.mark.parametrize("runs", [0, -1, True, 1.0])
def test_flow_suite_rejects_invalid_runs(runs) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        run_flow_suite(config={}, runs=runs)
