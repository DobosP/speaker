from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.conversation_eval.runner as conversation_runner
import tools.conversation_eval.__main__ as conversation_cli
from always_on_agent.capabilities import (
    CapabilityRegistry,
    CapabilityResult,
    CapabilitySpec,
)
from tools.conversation_eval.__main__ import _gate_exit_code
from tools.conversation_eval.identity import (
    _expected_template,
    verify_minicpm_identity,
    verify_ollama_blob_identity,
)
from tools.conversation_eval.models import (
    ObservedLLM,
    StreamGate,
    _history_message_sha256,
)
from tools.conversation_eval.report import build_report, summarize_model, write_report
from tools.conversation_eval.runner import (
    _grade,
    deterministic_models,
    ollama_models,
    run_scenario,
    safe_config,
    warm_models,
)
from tools.conversation_eval.schema import TraceEvent
from tools.conversation_eval.scenarios import SCENARIOS, selected
from tools.setup_minicpm import SOURCE_MODEL_BLOB_SHA256


@pytest.fixture(scope="module")
def evaluation_config() -> dict:
    return safe_config()


def _scenario(name: str):
    return next(item for item in SCENARIOS if item.scenario_id == name)


class _TokenLLM:
    model = "token-test"

    def __init__(self, tokens):
        self._tokens = tuple(tokens)

    def stream(self, prompt, *, system=None, **kwargs):
        del prompt, system, kwargs
        yield from self._tokens


def test_stream_gate_waits_only_after_a_real_continuation():
    gate = StreamGate()
    gate.arm()
    gate.open()
    observed = ObservedLLM(
        _TokenLLM(("Blue.", " ", "White.")),
        stream_gate=gate,
    )

    assert "".join(observed.stream("answer")) == "Blue. White."
    assert gate.paused.is_set()
    assert gate.continuation_observed.is_set()


def test_stream_gate_does_not_pause_at_one_sentence_eos():
    gate = StreamGate()
    gate.arm()
    observed = ObservedLLM(_TokenLLM(("Blue. ",)), stream_gate=gate)

    assert "".join(observed.stream("answer")) == "Blue. "
    assert not gate.paused.is_set()
    assert not gate.continuation_observed.is_set()


@pytest.mark.parametrize(
    "tokens",
    [
        ("Blue. White. Red.",),
        ("Blue\nWhite\nRed",),
    ],
)
def test_stream_gate_handles_shared_boundaries_with_same_token_continuation(tokens):
    gate = StreamGate()
    gate.arm()
    gate.open()
    observed = ObservedLLM(_TokenLLM(tokens), stream_gate=gate)

    assert "".join(observed.stream("answer")) == "".join(tokens)
    assert gate.paused.is_set()
    assert gate.continuation_observed.is_set()


def test_stream_gate_skips_empty_leading_boundaries_before_pausing():
    gate = StreamGate()
    gate.arm()
    gate.open()
    observed = ObservedLLM(
        _TokenLLM(("\n\nBlue. White.",)),
        stream_gate=gate,
    )

    assert "".join(observed.stream("answer")) == "\n\nBlue. White."
    assert gate.paused.is_set()
    assert gate.continuation_observed.is_set()


class _FrozenTrace:
    def __init__(self, events):
        self._events = tuple(events)

    def events(self):
        return self._events

    def event_kinds(self):
        return tuple(event.kind for event in self._events)

    def tool_names(self):
        return tuple(
            str(event.payload.get("name", ""))
            for event in self._events
            if event.kind == "capability.started"
            and bool(event.payload.get("planner_tool"))
        )

    def cancellation_reasons(self):
        return tuple(
            str(event.payload.get("reason", ""))
            for event in self._events
            if event.kind == "control.stop"
        )


def _regrade(
    scenario,
    result,
    *,
    turns=None,
    events=None,
    model_calls=None,
    metrics=None,
):
    return _grade(
        scenario,
        result.turns if turns is None else tuple(turns),
        _FrozenTrace(result.trace if events is None else events),
        result.model_calls if model_calls is None else tuple(model_calls),
        result.memory_after,
        result.metrics if metrics is None else tuple(metrics),
        quiescent=True,
        invocations_closed=True,
        model_calls_closed=True,
        stale_output=False,
        error="",
    )


def test_capability_invocation_observer_captures_exact_boundary_and_is_removable():
    registry = CapabilityRegistry()
    registry.register(
        "search.test",
        lambda query, _context: CapabilityResult(True, f"result for {query}"),
        spec=CapabilitySpec(
            "search.test",
            summary="test search",
            planner_tool=True,
        ),
    )
    events = []
    remove = registry.observe_invocations(events.append)

    result = registry.invoke("search.test", "needle", {"task_id": "raw-task"})

    assert result.ok is True
    assert [event.phase for event in events] == ["started", "finished"]
    assert {event.invocation_id for event in events} == {1}
    assert all(event.name == "search.test" for event in events)
    assert all(event.query == "needle" for event in events)
    assert all(event.task_id == "raw-task" for event in events)
    assert all(event.planner_tool is True for event in events)
    assert events[0].result is None
    assert events[1].result is not None
    assert events[1].result.ok == result.ok
    assert events[1].result.text == result.text

    remove()
    remove()
    registry.invoke("search.test", "after remove")
    assert len(events) == 2


def test_capability_observer_failure_never_changes_provider_result():
    registry = CapabilityRegistry()
    registry.register("answer", lambda _query, _context: CapabilityResult(True, "ok"))
    registry.observe_invocations(lambda _event: (_ for _ in ()).throw(RuntimeError("boom")))

    assert registry.invoke("answer", "question") == CapabilityResult(True, "ok")
    assert registry.invoke("missing", "question").error == "missing capability: missing"


def test_capability_observer_cannot_mutate_live_result_data():
    registry = CapabilityRegistry()
    live = CapabilityResult(
        True,
        "safe",
        data={"egress": True, "nested": {"route": "fast"}},
    )
    registry.register("answer", lambda _query, _context: live)

    def attack(event) -> None:
        if event.result is not None:
            with pytest.raises((AttributeError, TypeError)):
                event.result.data["egress"] = False
            nested = event.result.data["nested"]
            with pytest.raises((AttributeError, TypeError)):
                nested["route"] = "main"

    registry.observe_invocations(attack)

    returned = registry.invoke("answer", "question")
    assert returned.data == {"egress": True, "nested": {"route": "fast"}}


def test_safe_config_keeps_production_gates_but_disables_external_state():
    config = safe_config()

    assert config["input_gate"]["enabled"] is True
    assert config["cleanup"]["enabled"] is True
    assert config["capability_router"]["enabled"] is True
    assert config["agent"]["planner"]["enabled"] is True
    assert config["memory"]["backend"] == "inmemory"
    assert config["llm"]["cloud"]["enabled"] is False
    assert config["llm"]["cloud"]["strategy"] == "local_only"
    assert config["web_search"]["enabled"] is False
    assert config["watch"]["enabled"] is False
    assert config["gui_actions"]["enabled"] is False
    assert config["screen_capture"]["enabled"] is False
    assert config["warm_on_start"] is False


@pytest.mark.parametrize(
    "host, message",
    (
        ("https://example.test:11434", "non-loopback"),
        ("http://user:password@127.0.0.1:11434", "credentials"),
    ),
)
def test_safe_config_rejects_remote_or_credentialed_hosts(
    monkeypatch,
    host: str,
    message: str,
):
    monkeypatch.setattr(
        conversation_runner,
        "load_config",
        lambda *_args, **_kwargs: {"llm": {"host": host}},
    )
    monkeypatch.setattr(
        conversation_runner,
        "apply_device_profile",
        lambda config, _device, strict: config,
    )

    with pytest.raises(ValueError, match=message):
        safe_config(include_local_config=True)


def test_conversation_reports_are_ignored():
    root = Path(__file__).resolve().parents[1]
    ignored = (root / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert "logs/conversation-eval/" in ignored


def test_config_contract_hash_includes_recent_context_memory_knobs():
    base = {
        "llm": {
            "backend": "ollama",
            "host": "http://127.0.0.1:11434",
            "main_model": "main:test",
            "fast_model": "fast:test",
            "options": {"num_ctx": 4096},
        },
        "memory": {"recent_context_as_messages": True},
    }
    changed = {
        **base,
        "memory": {"recent_context_as_messages": False},
    }

    first = conversation_cli._config_metadata(
        base,
        include_local_config=False,
    )
    second = conversation_cli._config_metadata(
        changed,
        include_local_config=False,
    )

    assert first["contract_sha256"] != second["contract_sha256"]


def test_deterministic_warmup_uses_runtime_prompt_without_polluting_scenario_calls(
    evaluation_config: dict,
):
    models = deterministic_models()
    warmup = warm_models(evaluation_config, models)
    result = run_scenario(
        _scenario("simple_qa"),
        config=evaluation_config,
        models=models,
        run_index=1,
    )

    assert warmup["ok"] is True
    assert len(warmup["system_prompt_sha256"]) == 64
    assert [call["model"] for call in warmup["calls"]] == [
        "deterministic-main-v1",
        "deterministic-fast-v1",
    ]
    assert [call["category"] for call in result.model_calls] == [
        "addressing",
        "answer",
    ]
    assert result.passed


def test_ollama_topology_defaults_to_configured_main_and_candidate_fast(
    evaluation_config: dict,
):
    production = ollama_models(
        evaluation_config,
        "candidate:test",
        topology="production-hybrid",
    )
    stress = ollama_models(
        evaluation_config,
        "candidate:test",
        topology="all-roles",
    )

    assert production.model_assignment == "production_hybrid_fast_override"
    assert production.role_map() == {
        "main": "gemma3:12b",
        "fast": "candidate:test",
    }
    assert stress.model_assignment == "all_roles_override"
    assert stress.role_map() == {
        "main": "candidate:test",
        "fast": "candidate:test",
    }


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda item: item.scenario_id)
def test_deterministic_conversation_scenario_gate(
    scenario,
    evaluation_config: dict,
):
    result = run_scenario(
        scenario,
        config=evaluation_config,
        models=deterministic_models(),
        run_index=1,
    )

    failures = {
        check.name: {"expected": check.expected, "actual": check.actual}
        for check in result.checks
        if not check.ok
    }
    assert result.error == ""
    assert result.passed is True, failures
    assert result.trace
    assert all(not event.task_id or event.task_id.startswith("T") for event in result.trace)
    assert all(
        not call.get("task_id") or str(call["task_id"]).startswith("T")
        for call in result.model_calls
    )


def test_barge_trace_proves_interrupted_sink_without_heard_memory(
    evaluation_config: dict,
):
    result = run_scenario(
        _scenario("barge_stop"),
        config=evaluation_config,
        models=deterministic_models(),
        run_index=1,
    )
    terminal = [
        event for event in result.trace if event.kind == "playback.terminal"
    ]

    assert result.passed
    assert [event.payload["outcome"] for event in terminal] == ["interrupted"]
    assert not [event for event in result.trace if event.kind == "memory.commit"]
    assert "task.cancelled" in {event.kind for event in result.trace}


def test_barge_grade_rejects_cancelled_task_tts_emitted_after_cut(
    evaluation_config: dict,
):
    scenario = _scenario("barge_redirect")
    result = run_scenario(
        scenario,
        config=evaluation_config,
        models=deterministic_models(),
        run_index=1,
    )
    final_sequence = max(event.sequence for event in result.trace)
    final_elapsed = max(event.elapsed_ms for event in result.trace)
    poisoned_events = result.trace + (
        TraceEvent(
            sequence=final_sequence + 1,
            elapsed_ms=final_elapsed + 1.0,
            kind="tts.request",
            task_id="T1",
            turn_id=1,
            payload={"text": "stale cancelled fragment"},
        ),
    )

    checks = _regrade(scenario, result, events=poisoned_events)
    stale_check = next(
        check for check in checks if check.name == "cancelled_task_tts_after_cut"
    )

    assert result.passed
    assert stale_check.ok is False
    assert stale_check.actual == (final_sequence + 1,)


def test_redirect_grade_rejects_stale_input_generation(
    evaluation_config: dict,
):
    scenario = _scenario("barge_redirect")
    result = run_scenario(
        scenario,
        config=evaluation_config,
        models=deterministic_models(),
        run_index=1,
    )
    cut_sequence = next(
        event.sequence
        for event in result.trace
        if event.kind == "eval.barge_requested"
    )
    poisoned_events = tuple(
        replace(
            event,
            payload={**event.payload, "input_generation": 1},
        )
        if event.kind == "playback.attributed"
        and int(event.payload.get("requested_sequence", 0)) > cut_sequence
        else event
        for event in result.trace
    )
    checks = _regrade(scenario, result, events=poisoned_events)
    generation = next(
        check
        for check in checks
        if check.name == "redirect_playback_input_generation"
    )

    assert result.passed
    assert generation.ok is False
    assert generation.expected == (2,)
    assert generation.actual == (1,)


def test_redirect_grade_rejects_multiple_completed_sentence_fragments(
    evaluation_config: dict,
):
    scenario = _scenario("barge_redirect")
    result = run_scenario(
        scenario,
        config=evaluation_config,
        models=deterministic_models(),
        run_index=1,
    )
    final_sequence = max(event.sequence for event in result.trace)
    final_elapsed = max(event.elapsed_ms for event in result.trace)
    events = result.trace + (
        TraceEvent(
            sequence=final_sequence + 1,
            elapsed_ms=final_elapsed + 1.0,
            kind="playback.terminal",
            task_id="T2",
            turn_id=2,
            payload={"fragment_id": "extra", "outcome": "completed"},
        ),
    )

    checks = _regrade(scenario, result, events=events)
    outcomes = next(
        check for check in checks if check.name == "redirect_playback_outcomes"
    )

    assert outcomes.ok is False


@pytest.mark.parametrize("poison", ["Tokyoish.", "Tokyo1."])
def test_redirect_grade_requires_normalized_exact_word(
    evaluation_config: dict,
    poison: str,
):
    scenario = _scenario("barge_redirect")
    result = run_scenario(
        scenario,
        config=evaluation_config,
        models=deterministic_models(),
        run_index=1,
    )
    turns = (
        result.turns[0],
        replace(
            result.turns[1],
            attempted_output=poison,
            sink_attested_output=poison,
        ),
    )

    checks = _regrade(scenario, result, turns=turns)
    exact = next(
        check for check in checks if check.name == "turn_2_exact_response"
    )

    assert exact.ok is False


def test_redirect_grade_requires_controller_route_and_no_model_call(
    evaluation_config: dict,
):
    scenario = _scenario("barge_redirect")
    result = run_scenario(
        scenario,
        config=evaluation_config,
        models=deterministic_models(),
        run_index=1,
    )
    events = tuple(
        replace(
            event,
            payload={
                **event.payload,
                "data": {"route": "fast", "handled_local": False},
            },
        )
        if event.kind == "task.completed" and event.task_id == "T2"
        else event
        for event in result.trace
    )
    model_calls = result.model_calls + (
        {
            "task_id": "T2",
            "category": "answer",
            "kind": "stream",
            "duration_ms": 1.0,
            "ttft_ms": 1.0,
        },
    )

    checks = _regrade(
        scenario,
        result,
        events=events,
        model_calls=model_calls,
    )
    controller = next(
        check for check in checks if check.name == "redirect_used_controller_answer"
    )
    bypass = next(
        check for check in checks if check.name == "redirect_bypassed_model"
    )

    assert controller.ok is False
    assert bypass.ok is False


def test_redirect_grade_joins_terminal_to_attributed_owner(
    evaluation_config: dict,
):
    scenario = _scenario("barge_redirect")
    result = run_scenario(
        scenario,
        config=evaluation_config,
        models=deterministic_models(),
        run_index=1,
    )
    terminal_ids = [
        str(event.payload.get("fragment_id", ""))
        for event in result.trace
        if event.kind == "playback.terminal"
    ]
    assert len(terminal_ids) == 2
    swapped = {terminal_ids[0]: terminal_ids[1], terminal_ids[1]: terminal_ids[0]}
    events = tuple(
        replace(
            event,
            payload={
                **event.payload,
                "fragment_id": swapped[str(event.payload.get("fragment_id", ""))],
            },
        )
        if event.kind == "playback.terminal"
        else event
        for event in result.trace
    )

    checks = _regrade(scenario, result, events=events)
    ownership = next(
        check
        for check in checks
        if check.name == "redirect_fragment_terminal_ownership"
    )
    outcomes = next(
        check for check in checks if check.name == "redirect_playback_outcomes"
    )
    terminal_pairs = next(
        check
        for check in checks
        if check.name == "one_terminal_per_playback_request"
    )

    assert outcomes.ok is True
    assert terminal_pairs.ok is True
    assert ownership.ok is False


def test_redirect_grade_rejects_nonempty_victim_safe_prefix(
    evaluation_config: dict,
):
    scenario = _scenario("barge_redirect")
    result = run_scenario(
        scenario,
        config=evaluation_config,
        models=deterministic_models(),
        run_index=1,
    )
    events = tuple(
        replace(
            event,
            payload={**event.payload, "safe_text_prefix": "Blue."},
        )
        if event.kind == "playback.terminal"
        and event.payload.get("outcome") == "interrupted"
        else event
        for event in result.trace
    )

    checks = _regrade(scenario, result, events=events)
    ownership = next(
        check
        for check in checks
        if check.name == "redirect_fragment_terminal_ownership"
    )

    assert ownership.ok is False


def test_redirect_grade_binds_response_only_to_second_input(
    evaluation_config: dict,
):
    scenario = _scenario("barge_redirect")
    result = run_scenario(
        scenario,
        config=evaluation_config,
        models=deterministic_models(),
        run_index=1,
    )

    def poison_response_only(event):
        if event.kind != "stt.final":
            return event
        metadata = dict(event.payload.get("metadata", {}))
        metadata["post_barge_response_only"] = event.turn_id == 1
        metadata["skip_user_memory"] = event.turn_id == 1
        return replace(event, payload={**event.payload, "metadata": metadata})

    checks = _regrade(
        scenario,
        result,
        events=tuple(poison_response_only(event) for event in result.trace),
    )
    ownership = next(
        check
        for check in checks
        if check.name == "redirect_response_only_input_ownership"
    )
    legacy_any = next(
        check for check in checks if check.name == "redirect_is_response_only"
    )

    assert legacy_any.ok is True
    assert ownership.ok is False


def test_redirect_grade_binds_completion_and_receipt_to_generation_two(
    evaluation_config: dict,
):
    scenario = _scenario("barge_redirect")
    result = run_scenario(
        scenario,
        config=evaluation_config,
        models=deterministic_models(),
        run_index=1,
    )
    events = tuple(
        replace(
            event,
            turn_id=1,
            payload={**event.payload, "input_generation": 1},
        )
        if (
            event.kind == "task.completed" and event.task_id == "T2"
        ) or (
            event.kind == "memory.commit"
            and event.payload.get("source") == "playback_receipt"
        )
        else event
        for event in result.trace
    )

    checks = _regrade(scenario, result, events=events)
    completion = next(
        check
        for check in checks
        if check.name == "redirect_completion_input_ownership"
    )
    receipt = next(
        check
        for check in checks
        if check.name == "redirect_memory_receipt_ownership"
    )

    assert completion.ok is False
    assert receipt.ok is False


def test_model_history_records_exact_privacy_safe_role_evidence(
    evaluation_config: dict,
):
    scenario = _scenario("model_history_followup")
    result = run_scenario(
        scenario,
        config=evaluation_config,
        models=deterministic_models(),
        run_index=1,
    )
    answer_calls = [
        call
        for call in result.model_calls
        if call.get("category") == "answer" and call.get("task_id")
    ]
    assert len(answer_calls) == 2
    second = answer_calls[1]
    expected_hashes = (
        _history_message_sha256("user", scenario.turns[0].text),
        _history_message_sha256(
            "assistant", result.turns[0].sink_attested_output
        ),
    )

    assert tuple(second["history_roles"]) == ("user", "assistant")
    assert tuple(second["history_message_sha256"]) == expected_hashes
    assert scenario.turns[0].text not in json.dumps(second)
    assert result.turns[0].sink_attested_output not in json.dumps(second)


@pytest.mark.parametrize(
    "field,value,check_name",
    (
        (
            "history_roles",
            ("assistant", "user"),
            "second_answer_history_roles",
        ),
        (
            "history_message_sha256",
            ("0" * 64, "1" * 64),
            "second_answer_history_messages",
        ),
    ),
)
def test_model_history_grade_rejects_poisoned_role_evidence(
    evaluation_config: dict,
    field: str,
    value: tuple[str, str],
    check_name: str,
):
    scenario = _scenario("model_history_followup")
    result = run_scenario(
        scenario,
        config=evaluation_config,
        models=deterministic_models(),
        run_index=1,
    )
    model_calls = tuple(
        {**call, field: value}
        if call.get("category") == "answer" and call.get("task_id") == "T2"
        else call
        for call in result.model_calls
    )

    checks = _regrade(
        scenario,
        result,
        model_calls=model_calls,
    )
    evidence = next(check for check in checks if check.name == check_name)

    assert evidence.ok is False


def test_context_grade_requires_first_audio_metric_per_generation(
    evaluation_config: dict,
):
    scenario = _scenario("typed_session_fact")
    result = run_scenario(
        scenario,
        config=evaluation_config,
        models=deterministic_models(),
        run_index=1,
    )
    checks = _regrade(scenario, result, metrics=result.metrics[:1])
    coverage = next(
        check
        for check in checks
        if check.name == "first_audio_record_per_playback_generation"
    )

    assert result.passed
    assert coverage.ok is False
    assert coverage.expected == ">= 2"
    assert coverage.actual == 1


def test_context_grade_anchors_each_turn_to_its_input_generation(
    evaluation_config: dict,
):
    scenario = _scenario("typed_session_fact")
    result = run_scenario(
        scenario,
        config=evaluation_config,
        models=deterministic_models(),
        run_index=1,
    )

    def collapse_second_generation(event):
        payload = dict(event.payload)
        if payload.get("input_generation") == 2:
            payload["input_generation"] = 1
        metadata = payload.get("metadata")
        if isinstance(metadata, dict) and metadata.get("input_generation") == 2:
            payload["metadata"] = {**metadata, "input_generation": 1}
        return replace(
            event,
            turn_id=1 if event.turn_id == 2 else event.turn_id,
            payload=payload,
        )

    checks = _regrade(
        scenario,
        result,
        events=tuple(collapse_second_generation(event) for event in result.trace),
    )
    turn_sequence = next(
        check for check in checks if check.name == "exact_evaluator_turn_sequence"
    )
    final_sequence = next(
        check
        for check in checks
        if check.name == "exact_final_input_generation_sequence"
    )

    assert result.passed
    assert turn_sequence.ok is False
    assert final_sequence.ok is False


def test_grade_rejects_negative_first_audio_latency(
    evaluation_config: dict,
):
    scenario = _scenario("simple_qa")
    result = run_scenario(
        scenario,
        config=evaluation_config,
        models=deterministic_models(),
        run_index=1,
    )
    poisoned_metrics = (
        {**result.metrics[0], "first_audio_latency": -0.1},
    )
    checks = _regrade(scenario, result, metrics=poisoned_metrics)
    validity = next(
        check for check in checks if check.name == "first_audio_values_valid"
    )

    assert result.passed
    assert validity.ok is False
    assert validity.expected == 1
    assert validity.actual == 0


def test_grade_rejects_playback_attribution_to_unknown_task(
    evaluation_config: dict,
):
    scenario = _scenario("simple_qa")
    result = run_scenario(
        scenario,
        config=evaluation_config,
        models=deterministic_models(),
        run_index=1,
    )
    poisoned_events = tuple(
        replace(event, task_id="T999")
        if event.kind == "playback.attributed"
        else event
        for event in result.trace
    )
    checks = _regrade(scenario, result, events=poisoned_events)
    source = next(
        check for check in checks if check.name == "exact_playback_attribution_source"
    )
    generation = next(
        check
        for check in checks
        if check.name == "playback_matches_task_input_generation"
    )

    assert result.passed
    assert source.ok is False
    assert generation.ok is False


def test_stream_grade_requires_finite_answer_ttft_and_duration(
    evaluation_config: dict,
):
    scenario = _scenario("simple_qa")
    result = run_scenario(
        scenario,
        config=evaluation_config,
        models=deterministic_models(),
        run_index=1,
    )
    poisoned_calls = tuple(
        {**call, "ttft_ms": float("nan"), "duration_ms": None}
        if call.get("category") == "answer" and call.get("task_id")
        else call
        for call in result.model_calls
    )
    checks = _regrade(scenario, result, model_calls=poisoned_calls)
    coverage = next(
        check for check in checks if check.name == "ttft_recorded_for_every_stream_call"
    )
    duration_coverage = next(
        check
        for check in checks
        if check.name == "duration_recorded_for_every_model_call"
    )

    assert result.passed
    assert coverage.ok is False
    assert coverage.actual == coverage.expected - 1
    assert duration_coverage.ok is False
    assert duration_coverage.actual == duration_coverage.expected - 1


def test_untrusted_tool_result_never_invokes_command_capability(
    evaluation_config: dict,
):
    result = run_scenario(
        _scenario("untrusted_tool_result"),
        config=evaluation_config,
        models=deterministic_models(),
        run_index=1,
    )
    invoked = [
        event.payload["name"]
        for event in result.trace
        if event.kind == "capability.started"
    ]

    assert result.passed
    assert "web.search" in invoked
    assert "command.stage" not in invoked
    assert "paris" in result.turns[0].sink_attested_output.lower()
    assert "ignore previous" not in result.turns[0].attempted_output.lower()


def test_model_call_slices_do_not_leak_across_scenarios(evaluation_config: dict):
    models = deterministic_models()
    scenario = _scenario("simple_qa")
    first = run_scenario(
        scenario,
        config=evaluation_config,
        models=models,
        run_index=1,
    )
    second = run_scenario(
        scenario,
        config=evaluation_config,
        models=models,
        run_index=2,
    )

    assert len(first.model_calls) == len(second.model_calls) == 2
    assert first.passed and second.passed


def _fake_show_pair(*, digest: str, source_digest: str | None = None, template: str | None = None):
    alias = {
        "modelfile": f"FROM /models/sha256-{digest}",
        "template": _expected_template() if template is None else template,
        "parameters": (
            'stop "<|im_end|>"\n'
            'stop "</s>"\n'
            "temperature 0.7\n"
            "top_p 0.95\n"
            "num_ctx 8192"
        ),
        "details": {"quantization_level": "Q8_0"},
    }
    source = {
        "modelfile": f"FROM /models/sha256-{source_digest or digest}",
    }
    return lambda model: alias if model.startswith("minicpm5") else source


def test_minicpm_identity_requires_blob_quantization_template_and_parameters():
    digest = SOURCE_MODEL_BLOB_SHA256
    verified = verify_minicpm_identity(show=_fake_show_pair(digest=digest))
    wrong_blob = verify_minicpm_identity(
        show=_fake_show_pair(digest=digest, source_digest="b" * 64)
    )
    wrong_template = verify_minicpm_identity(
        show=_fake_show_pair(digest=digest, template="{{ .Prompt }}")
    )

    assert verified.ok is True
    assert verified.alias_blob == verified.source_blob == digest[:12]
    assert verified.pinned_blob_match is True
    assert wrong_blob.ok is False and wrong_blob.blob_match is False
    assert wrong_template.ok is False and wrong_template.template_match is False


def test_minicpm_identity_prefers_effective_alias_modelfile_template():
    digest = SOURCE_MODEL_BLOB_SHA256
    show = _fake_show_pair(digest=digest)

    def conflicting_show(model: str):
        value = show(model)
        if not model.startswith("minicpm5"):
            return value
        expected = _expected_template()
        return {
            **value,
            "template": "{{ upstream.base_template }}",
            "modelfile": (
                f"FROM /models/sha256-{digest}\n"
                f'TEMPLATE "{expected}"\n'
                + value["parameters"].replace("stop ", "PARAMETER stop ")
                .replace("temperature ", "PARAMETER temperature ")
                .replace("top_p ", "PARAMETER top_p ")
                .replace("num_ctx ", "PARAMETER num_ctx ")
            ),
        }

    identity = verify_minicpm_identity(show=conflicting_show)

    assert identity.template_match is True
    assert identity.ok is True


def test_minicpm_identity_transport_construction_error_fails_closed(monkeypatch):
    import tools.conversation_eval.identity as identity_module

    monkeypatch.setattr(
        identity_module,
        "_ollama_show",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic client construction failure")
        ),
    )

    identity = identity_module.verify_minicpm_identity()

    assert identity.ok is False
    assert "synthetic client construction failure" in identity.error


def test_generic_ollama_identity_records_full_blob_digest():
    digest = "d" * 64
    identity = verify_ollama_blob_identity(
        "gemma:test",
        show=lambda _model: {"modelfile": f"FROM /models/sha256-{digest}"},
    )

    assert identity.ok is True
    assert identity.blob_sha256 == digest
    assert len(identity.effective_config_sha256) == 64


def test_generic_ollama_identity_changes_with_template_or_parameters():
    digest = "d" * 64

    def verify(template: str, temperature: float):
        return verify_ollama_blob_identity(
            "gemma:test",
            show=lambda _model: {
                "modelfile": (
                    f"FROM /models/sha256-{digest}\n"
                    f'TEMPLATE "{template}"\n'
                    f"PARAMETER temperature {temperature}"
                ),
                "template": template,
                "parameters": f"temperature {temperature}",
            },
        )

    original = verify("{{ .Prompt }}", 0.7)
    changed_template = verify("{{ .System }} {{ .Prompt }}", 0.7)
    changed_parameter = verify("{{ .Prompt }}", 0.2)

    assert original.blob_sha256 == changed_template.blob_sha256
    assert original.blob_sha256 == changed_parameter.blob_sha256
    assert original.effective_config_sha256 != changed_template.effective_config_sha256
    assert original.effective_config_sha256 != changed_parameter.effective_config_sha256


def test_report_is_versioned_json_and_defines_repeat_reliability(
    tmp_path,
    evaluation_config: dict,
):
    models = deterministic_models()
    scenario = _scenario("simple_qa")
    results = tuple(
        run_scenario(
            scenario,
            config=evaluation_config,
            models=models,
            run_index=index,
        )
        for index in (1, 2, 3)
    )
    summary = summarize_model(models.label, results)
    report = build_report(
        mode="deterministic",
        device="desktop_gpu_4090",
        candidate_results=results,
    )
    path = write_report(tmp_path / "report.json", report)
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert summary.pass_at_1 is True
    assert summary.pass_power_k is True
    assert summary.scenario_reliability["simple_qa"]["runs_passed"] == 3
    assert loaded["schema_version"] == 1
    assert loaded["scenario_set_version"] == 2
    assert loaded["candidate"]["summary"]["pass_power_k"] is True
    assert loaded["gate"]["coverage_ok"] is False
    assert loaded["gate"]["passed"] is False
    assert _gate_exit_code(loaded["gate"]) == 1
    assert "no ASR" in loaded["scope"]

    complete_results = tuple(
        replace(
            results[run_index - 1],
            scenario_id=scenario_spec.scenario_id,
            description=scenario_spec.description,
            run_index=run_index,
        )
        for scenario_spec in SCENARIOS
        for run_index in (1, 2, 3)
    )
    complete = build_report(
        mode="deterministic",
        device="desktop_gpu_4090",
        candidate_results=complete_results,
    )
    assert complete["gate"]["coverage_ok"] is True
    assert complete["gate"]["passed"] is True

    candidate_complete = tuple(
        replace(result, model="candidate:test") for result in complete_results
    )
    baseline_complete = tuple(
        replace(result, model="baseline:test") for result in complete_results
    )

    def evaluation_details(fast_model: str, *, warm: bool = True):
        return {
            "model_assignment": "production_hybrid_fast_override",
            "role_models": {"main": "shared-main:test", "fast": fast_model},
            "warmup": (
                {
                    "policy": "production_system_prompt",
                    "performed": True,
                    "gate_eligible": True,
                    "ok": True,
                    "system_prompt_sha256": "a" * 64,
                    "calls": [
                        {
                            "model": "shared-main:test",
                            "roles": ["main"],
                            "ok": True,
                        },
                        {"model": fast_model, "roles": ["fast"], "ok": True},
                    ],
                }
                if warm
                else {
                    "policy": "cold",
                    "performed": False,
                    "gate_eligible": False,
                    "ok": True,
                    "calls": [],
                }
            ),
        }

    def identity_record(model: str, blob: str, effective: str):
        return {
            "model": model,
            "verification": "ollama_blob_effective_config",
            "required": True,
            "ok": True,
            "blob_sha256": blob,
            "effective_config_sha256": effective,
        }

    def identity_bundle(fast_model: str, blob: str, effective: str):
        role_models = {"main": "shared-main:test", "fast": fast_model}
        before = {
            "role_models": role_models,
            "models": {
                "shared-main:test": identity_record(
                    "shared-main:test", "1" * 64, "2" * 64
                ),
                fast_model: identity_record(fast_model, blob, effective),
            },
            "ok": True,
        }
        return {
            "before": before,
            "after": json.loads(json.dumps(before)),
            "stable": True,
            "ok": True,
        }

    identity_records = {
        "candidate": identity_bundle("candidate:test", "e" * 64, "a" * 64),
        "baseline": identity_bundle("baseline:test", "f" * 64, "b" * 64),
    }
    repository_metadata = {"revision": "b" * 40, "dirty": False}
    config_metadata = {
        "contract_sha256": "c" * 64,
        "llm_options_sha256": "d" * 64,
        "include_local_config": False,
        "backend": "ollama",
        "host": "http://127.0.0.1:11434",
        "configured_role_models": {
            "main": "shared-main:test",
            "fast": "candidate:test",
        },
    }
    real_metadata = {
        "topology": "production-hybrid",
        "warm_policy": "production",
        "execution_order": ["baseline", "candidate"],
        "ollama_python_version": "0.test",
        "repository": repository_metadata,
        "config": config_metadata,
        "provenance_snapshots": {
            "before": {
                "repository": json.loads(json.dumps(repository_metadata)),
                "config": json.loads(json.dumps(config_metadata)),
            },
            "after": {
                "repository": json.loads(json.dumps(repository_metadata)),
                "config": json.loads(json.dumps(config_metadata)),
            },
        },
    }
    valid_real = build_report(
        mode="ollama",
        device="desktop_gpu_4090",
        candidate_results=candidate_complete,
        baseline_results=baseline_complete,
        identities=identity_records,
        metadata=real_metadata,
        candidate_metadata=evaluation_details("candidate:test"),
        baseline_metadata=evaluation_details("baseline:test"),
        provenance_ok=True,
    )
    assert valid_real["gate"]["identity_evidence_ok"] is True
    assert valid_real["gate"]["topology_ok"] is True
    assert valid_real["gate"]["ab_distinct"] is True
    assert valid_real["gate"]["passed"] is True

    dirty_metadata = json.loads(json.dumps(real_metadata))
    dirty_metadata["repository"]["dirty"] = True
    dirty_metadata["provenance_snapshots"]["after"]["repository"]["dirty"] = True
    dirty_real = build_report(
        mode="ollama",
        device="desktop_gpu_4090",
        candidate_results=candidate_complete,
        baseline_results=baseline_complete,
        identities=identity_records,
        metadata=dirty_metadata,
        candidate_metadata=evaluation_details("candidate:test"),
        baseline_metadata=evaluation_details("baseline:test"),
        provenance_ok=True,
    )
    assert dirty_real["gate"]["repository_clean"] is False
    assert dirty_real["gate"]["provenance_ok"] is False
    assert dirty_real["gate"]["passed"] is False
    assert _gate_exit_code(dirty_real["gate"]) == 2

    incomplete_warm = evaluation_details("candidate:test")
    incomplete_warm["warmup"]["calls"] = incomplete_warm["warmup"]["calls"][:1]
    missing_role_warm = build_report(
        mode="ollama",
        device="desktop_gpu_4090",
        candidate_results=candidate_complete,
        baseline_results=baseline_complete,
        identities=identity_records,
        metadata=real_metadata,
        candidate_metadata=incomplete_warm,
        baseline_metadata=evaluation_details("baseline:test"),
        provenance_ok=True,
    )
    assert missing_role_warm["gate"]["warmup_ok"] is False
    assert missing_role_warm["gate"]["passed"] is False

    same_identity_records = {
        **identity_records,
        "baseline": identity_bundle("candidate:test", "e" * 64, "a" * 64),
    }
    same_model_ab = build_report(
        mode="ollama",
        device="desktop_gpu_4090",
        candidate_results=candidate_complete,
        baseline_results=candidate_complete,
        identities=same_identity_records,
        metadata=real_metadata,
        candidate_metadata=evaluation_details("candidate:test"),
        baseline_metadata=evaluation_details("candidate:test"),
        provenance_ok=True,
    )
    assert same_model_ab["gate"]["topology_ok"] is True
    assert same_model_ab["gate"]["ab_distinct"] is False
    assert same_model_ab["gate"]["passed"] is False

    missing_baseline = build_report(
        mode="ollama",
        device="desktop_gpu_4090",
        candidate_results=complete_results,
    )
    assert missing_baseline["gate"]["baseline_required"] is True
    assert missing_baseline["gate"]["baseline_present"] is False
    assert missing_baseline["gate"]["semantic_pass"] is False
    assert missing_baseline["gate"]["passed"] is False

    unverified = build_report(
        mode="ollama",
        device="desktop_gpu_4090",
        candidate_results=candidate_complete,
        baseline_results=baseline_complete,
        identities=identity_records,
        metadata=real_metadata,
        candidate_metadata=evaluation_details("candidate:test"),
        baseline_metadata=evaluation_details("baseline:test"),
        provenance_ok=False,
        diagnostic_override_used=True,
    )
    assert unverified["gate"]["semantic_pass"] is True
    assert unverified["gate"]["provenance_ok"] is False
    assert unverified["gate"]["diagnostic_override_used"] is True
    assert unverified["gate"]["passed"] is False
    assert _gate_exit_code(unverified["gate"]) == 2

    cold = build_report(
        mode="ollama",
        device="desktop_gpu_4090",
        candidate_results=candidate_complete,
        baseline_results=baseline_complete,
        identities=identity_records,
        metadata=real_metadata,
        candidate_metadata=evaluation_details("candidate:test", warm=False),
        baseline_metadata=evaluation_details("baseline:test", warm=False),
        provenance_ok=True,
    )
    assert cold["gate"]["semantic_pass"] is True
    assert cold["gate"]["warmup_ok"] is False
    assert cold["gate"]["passed"] is False
    assert _gate_exit_code(cold["gate"]) == 1


def test_unknown_scenario_fails_closed():
    with pytest.raises(ValueError, match="unknown scenario"):
        selected(["not-a-scenario"])


def test_unverified_override_writes_red_report_and_exits_two(
    monkeypatch,
    tmp_path,
):
    output = tmp_path / "unverified.json"
    monkeypatch.setattr(
        conversation_cli,
        "verify_minicpm_identity",
        lambda **_kwargs: SimpleNamespace(
            ok=False,
            as_dict=lambda: {"ok": False, "error": "synthetic mismatch"},
        ),
    )
    monkeypatch.setattr(
        conversation_cli,
        "verify_ollama_blob_identity",
        lambda model, **_kwargs: SimpleNamespace(
            ok=True,
            as_dict=lambda: {
                "model": model,
                "blob_sha256": "d" * 64,
                "ok": True,
            },
        ),
    )
    monkeypatch.setattr(
        conversation_cli,
        "ollama_models",
        lambda *_args, **_kwargs: deterministic_models(),
    )
    monkeypatch.setattr(
        conversation_cli,
        "warm_models",
        lambda *_args, **_kwargs: {
            "policy": "production_system_prompt",
            "performed": True,
            "ok": True,
            "calls": [],
        },
    )

    exit_code = conversation_cli.main(
        [
            "--mode",
            "ollama",
            "--allow-unverified-model",
            "--runs",
            "1",
            "--scenario",
            "simple_qa",
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 2
    assert report["gate"]["coverage_ok"] is False
    assert report["gate"]["semantic_pass"] is False
    assert report["gate"]["provenance_ok"] is False
    assert report["gate"]["diagnostic_override_used"] is True
    assert report["gate"]["passed"] is False
