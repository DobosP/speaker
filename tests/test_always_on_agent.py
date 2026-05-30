from __future__ import annotations

import time

from always_on_agent.continuation import (
    CONTINUE,
    NEW,
    ContinuationConfig,
    ScriptedContinuationClassifier,
)
from always_on_agent.diagnostics import summarize
from always_on_agent.events import AgentEvent, EventKind, Mode
from always_on_agent.bridge import TranscriptBridge
from always_on_agent.models import IntentDecision, IntentKind
from always_on_agent.replay import replay_records
from always_on_agent.runtime import AlwaysOnAgentRuntime
from always_on_agent.speech_analyzer import LiveSpeechAnalyzer, ModePolicy
from always_on_agent.supervisor import AgentSupervisor


def _drain_until_idle(supervisor: AgentSupervisor, timeout: float = 1.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        supervisor.drain()
        if not supervisor.state.active_tasks and not supervisor.state.queued_tasks:
            supervisor.drain()
            return
        time.sleep(0.01)
    supervisor.drain()


def test_mode_switch_then_assistant_task_emits_tts_request():
    supervisor = AgentSupervisor()

    supervisor.publish(AgentEvent.final("assistant mode"))
    supervisor.publish(AgentEvent.final("tell me a short joke"))
    _drain_until_idle(supervisor)

    assert supervisor.state.mode == Mode.ASSISTANT
    assert "Mode: assistant" in supervisor.state.spoken_outputs
    assert "I can help with: tell me a short joke." in supervisor.state.spoken_outputs
    assert any(e.kind == EventKind.TTS_REQUEST for e in supervisor.state.event_log)


def test_passive_mode_ignores_unprefixed_final_text():
    supervisor = AgentSupervisor()

    supervisor.publish(AgentEvent.final("this should only be logged"))
    _drain_until_idle(supervisor)

    assert supervisor.state.mode == Mode.PASSIVE
    assert supervisor.state.transcript_log == ["this should only be logged"]
    assert supervisor.state.spoken_outputs == []


def test_search_prefix_runs_even_from_passive_mode():
    supervisor = AgentSupervisor()

    supervisor.publish(AgentEvent.final("search local speech to text"))
    _drain_until_idle(supervisor)

    assert supervisor.state.mode == Mode.PASSIVE
    assert supervisor.state.spoken_outputs
    assert "Moonshine" in supervisor.state.spoken_outputs[0]


def test_partial_stop_has_priority_and_cancels_research():
    supervisor = AgentSupervisor()

    supervisor.publish(AgentEvent.final("research open source voice pipelines"))
    supervisor.drain()
    assert supervisor.state.active_tasks

    supervisor.publish(AgentEvent.partial("stop"))
    _drain_until_idle(supervisor)

    assert supervisor.state.active_tasks == {}
    assert "[cancelled]" in supervisor.state.spoken_outputs
    assert any(e.kind == EventKind.TASK_CANCELLED for e in supervisor.state.event_log)


def test_research_task_reports_progress_and_completion():
    supervisor = AgentSupervisor()

    supervisor.publish(AgentEvent.final("research async audio processing"))
    _drain_until_idle(supervisor)

    progress_steps = [
        e.payload["step"]
        for e in supervisor.state.event_log
        if e.kind == EventKind.TASK_PROGRESS
    ]
    assert progress_steps == ["scope", "search", "synthesize"]
    assert supervisor.state.spoken_outputs
    assert supervisor.state.spoken_outputs[0].startswith("Research summary for async audio processing")


def test_romanian_research_prefix_is_classified():
    supervisor = AgentSupervisor()

    supervisor.publish(AgentEvent.final("cerceteaza moonshine pentru voice assistant"))
    _drain_until_idle(supervisor)

    assert supervisor.state.spoken_outputs
    assert "Moonshine" in supervisor.state.spoken_outputs[0]
    assert supervisor.state.observations[-1].language == "ro"


def test_wake_word_in_passive_mode_starts_assistant_task():
    supervisor = AgentSupervisor()

    supervisor.publish(AgentEvent.final("assistant help me plan the audio pipeline"))
    _drain_until_idle(supervisor)

    assert supervisor.state.mode == Mode.PASSIVE
    assert any("audio pipeline" in output for output in supervisor.state.spoken_outputs)


def test_dictation_mode_stores_transcript_without_speaking():
    supervisor = AgentSupervisor()

    supervisor.publish(AgentEvent.final("dictation mode"))
    supervisor.publish(AgentEvent.final("this is a clean note"))
    _drain_until_idle(supervisor)

    assert supervisor.state.mode == Mode.DICTATION
    assert supervisor.state.transcript_log[-1] == "this is a clean note"
    assert supervisor.state.spoken_outputs == ["Mode: dictation"]


def test_replay_records_and_diagnostics_summary():
    supervisor = replay_records(
        [
            {"kind": "stt.final", "text": "assistant mode"},
            {"kind": "stt.final", "text": "search pipecat livekit"},
        ]
    )

    summary = summarize(supervisor)
    assert summary["mode"] == "assistant"
    assert summary["tts_requests"] >= 1
    assert summary["transcripts"] == 2
    assert summary["decisions"][0]["kind"] == "mode_switch"


def test_runtime_facade_ingests_live_transcripts_and_snapshots_state():
    runtime = AlwaysOnAgentRuntime()

    runtime.ingest_final("assistant mode")
    runtime.ingest_partial("search moonshine")
    runtime.ingest_final("search moonshine")
    assert runtime.wait_idle()

    snapshot = runtime.snapshot()
    assert snapshot.mode == "assistant"
    assert snapshot.last_partial == "search moonshine"
    assert snapshot.transcripts == ("assistant mode", "search moonshine")
    assert snapshot.outputs
    assert runtime.diagnostics()["tts_requests"] >= 1


def test_transcript_bridge_matches_existing_callback_shape():
    bridge = TranscriptBridge()

    bridge.on_final_text("assistant mode")
    bridge.on_final_text("research livekit agents")
    assert bridge.runtime.wait_idle()

    snapshot = bridge.runtime.snapshot()
    assert snapshot.mode == "assistant"
    assert any("LiveKit" in output for output in snapshot.outputs)


def test_research_task_records_plan_metadata():
    supervisor = AgentSupervisor()

    supervisor.publish(AgentEvent.final("research wyoming voice services"))
    _drain_until_idle(supervisor)

    completed = [
        event
        for event in supervisor.state.event_log
        if event.kind == EventKind.TASK_COMPLETED
    ]
    assert completed
    data = completed[-1].payload["data"]
    assert "plan" in data
    task_started = [
        event
        for event in supervisor.state.event_log
        if event.kind == EventKind.TASK_STARTED
    ][-1]
    assert task_started.payload["capability"] == "research.local"


def test_command_requires_voice_confirmation_before_task_start():
    runtime = AlwaysOnAgentRuntime()

    runtime.ingest_final("command mode")
    runtime.ingest_final("open browser")
    snapshot = runtime.snapshot()

    assert snapshot.mode == "command"
    assert snapshot.active_tasks == ()
    assert len(snapshot.pending_confirmations) == 1
    assert "Confirm command: open browser" in snapshot.outputs

    runtime.ingest_final("confirm")
    assert runtime.wait_idle()

    snapshot = runtime.snapshot()
    assert snapshot.pending_confirmations == ()
    assert any("Command staged for confirmation: open browser" in output for output in snapshot.outputs)


def test_command_confirmation_can_be_denied():
    runtime = AlwaysOnAgentRuntime()

    runtime.ingest_final("command mode")
    runtime.ingest_final("open browser")
    runtime.ingest_final("no")
    assert runtime.wait_idle()

    snapshot = runtime.snapshot()
    assert snapshot.pending_confirmations == ()
    assert not any("Command staged" in output for output in snapshot.outputs)
    assert "Command cancelled." in snapshot.outputs


def test_research_tasks_queue_when_parallel_limit_is_reached():
    analyzer = LiveSpeechAnalyzer(ModePolicy(research_parallel_tasks=1))
    supervisor = AgentSupervisor(analyzer=analyzer)

    supervisor.publish(AgentEvent.final("research moonshine"))
    supervisor.drain()
    supervisor.publish(AgentEvent.final("research livekit"))
    supervisor.drain()

    assert len(supervisor.state.active_tasks) == 1
    assert len(supervisor.state.queued_tasks) == 1
    assert any(output.startswith("Queued research") for output in supervisor.state.spoken_outputs)

    _drain_until_idle(supervisor)

    assert supervisor.state.active_tasks == {}
    assert supervisor.state.queued_tasks == []
    research_outputs = [
        output for output in supervisor.state.spoken_outputs if output.startswith("Research summary")
    ]
    assert len(research_outputs) == 2


# --- ADD-ON / continuation (follow-up extends the in-flight turn) ---------------


def _continuation_supervisor(classifier, *, enabled=True):
    return AgentSupervisor(
        continuation_config=ContinuationConfig(enabled=enabled),
        continuation=classifier,
    )


def _seat_assistant_victim(supervisor, text="whats the weather", *, started_speaking=False):
    """Seat an in-flight ASSISTANT task in active_tasks WITHOUT a worker thread.

    Mirrors what _start_task records (epoch stamp + active_tasks entry) but skips
    spawning a thread, so the 'victim' stays put as a deterministic in-flight turn
    while the test drives a follow-up -- no thread-timing race on the merge window.
    """
    decision = IntentDecision(IntentKind.ASSISTANT, 0.9, text, "assistant_mode", mode=Mode.ASSISTANT)
    task = supervisor.tasks.create_task(decision)
    task.started_speaking = started_speaking
    with supervisor._cancel_lock:  # noqa: SLF001 - test seats an in-flight task
        task.speech_epoch = supervisor.speech_epoch
        supervisor.state.active_tasks[task.task_id] = task
    return task


def _addon(text):
    return IntentDecision(IntentKind.ASSISTANT, 0.8, text, "assistant_mode", mode=Mode.ASSISTANT)


def test_continuation_merges_before_audio_into_single_turn():
    classifier = ScriptedContinuationClassifier({"and also the forecast": CONTINUE})
    supervisor = _continuation_supervisor(classifier)
    supervisor.state.mode = Mode.ASSISTANT
    victim = _seat_assistant_victim(supervisor, "whats the weather", started_speaking=False)

    supervisor._execute_decision(_addon("and also the forecast"))

    # The not-yet-spoken victim is superseded; exactly one merged turn remains.
    assert victim.cancel_event.is_set()
    assert victim.task_id not in supervisor.state.active_tasks
    assert len(supervisor.state.active_tasks) == 1
    merged = next(iter(supervisor.state.active_tasks.values()))
    assert merged.task_id != victim.task_id
    assert "whats the weather" in merged.input_text
    assert "and also the forecast" in merged.input_text
    assert merged.metadata.get("continuation_of") == victim.task_id
    assert merged.metadata.get("skip_user_memory") is True
    # Epoch advanced so any sentence the victim races out is dropped by the gate.
    assert supervisor.speech_epoch == 1
    assert classifier.calls == [("and also the forecast", "whats the weather")]

    supervisor.cancel_all()
    _drain_until_idle(supervisor)


def test_continuation_queues_behind_speaking_turn():
    classifier = ScriptedContinuationClassifier({"and also the forecast": CONTINUE})
    supervisor = _continuation_supervisor(classifier)
    supervisor.state.mode = Mode.ASSISTANT
    victim = _seat_assistant_victim(supervisor, "whats the weather", started_speaking=True)

    supervisor._execute_decision(_addon("and also the forecast"))

    # The speaking victim is left alone; the continuation waits strictly behind it.
    assert not victim.cancel_event.is_set()
    assert victim.task_id in supervisor.state.active_tasks
    assert supervisor.speech_epoch == 0  # no epoch bump: the victim's audio stands
    assert len(supervisor.state.queued_tasks) == 1
    cont = supervisor.state.queued_tasks[0]
    assert cont.metadata.get("continue_after") == victim.task_id
    assert "whats the weather" in cont.input_text
    assert "and also the forecast" in cont.input_text

    # The continuation only starts once the victim completes (drains the queue).
    supervisor.publish(
        AgentEvent(EventKind.TASK_COMPLETED, {"task_id": victim.task_id, "text": "sunny", "speak": True})
    )
    supervisor.drain()
    assert supervisor.state.queued_tasks == []

    supervisor.cancel_all()
    _drain_until_idle(supervisor)


def test_followup_classified_new_starts_independent_task():
    classifier = ScriptedContinuationClassifier(default=NEW)
    supervisor = _continuation_supervisor(classifier)
    supervisor.state.mode = Mode.ASSISTANT
    victim = _seat_assistant_victim(supervisor, "whats the weather")

    supervisor._execute_decision(_addon("what is the capital of france"))

    # NEW -> falls through to a normal independent task; the victim is untouched.
    assert not victim.cancel_event.is_set()
    assert victim.task_id in supervisor.state.active_tasks
    assert len(supervisor.state.active_tasks) == 2
    assert classifier.calls == [("what is the capital of france", "whats the weather")]

    supervisor.cancel_all()
    _drain_until_idle(supervisor)


def test_continuation_not_consulted_for_control_or_non_assistant_finals():
    # default CONTINUE: if the gate were (wrongly) consulted it would try to merge.
    classifier = ScriptedContinuationClassifier(default=CONTINUE)
    supervisor = _continuation_supervisor(classifier)
    supervisor.state.mode = Mode.ASSISTANT

    # A real STOP forks to cancel_all BEFORE the continuation gate.
    _seat_assistant_victim(supervisor, "whats the weather")
    supervisor.publish(AgentEvent.final("stop"))
    supervisor.drain()
    # A RESEARCH-prefixed final is a non-ASSISTANT decision -> gate is a no-op.
    supervisor.publish(AgentEvent.final("research edge tpus"))
    supervisor.drain()

    assert classifier.calls == []  # never consulted for STOP or non-ASSISTANT intents

    supervisor.cancel_all()
    _drain_until_idle(supervisor)


def test_continuation_disabled_by_default_spawns_independent_task():
    # A supervisor built without continuation config behaves exactly as before:
    # a follow-up is a competing task, not a merge (byte-identical-when-off guard).
    supervisor = AgentSupervisor()
    supervisor.state.mode = Mode.ASSISTANT
    victim = _seat_assistant_victim(supervisor, "whats the weather")

    supervisor._execute_decision(_addon("and also the forecast"))

    assert not victim.cancel_event.is_set()
    assert len(supervisor.state.active_tasks) == 2
    assert supervisor._continuation is None

    supervisor.cancel_all()
    _drain_until_idle(supervisor)


def test_chained_addons_fold_into_one_queued_continuation():
    # Two add-ons arriving behind the SAME speaking turn must not become two
    # queued continuations that then start in parallel -- they fold into one.
    classifier = ScriptedContinuationClassifier(
        {"and also the forecast": CONTINUE, "and the humidity": CONTINUE}
    )
    supervisor = _continuation_supervisor(classifier)
    supervisor.state.mode = Mode.ASSISTANT
    _seat_assistant_victim(supervisor, "whats the weather", started_speaking=True)

    supervisor._execute_decision(_addon("and also the forecast"))
    supervisor._execute_decision(_addon("and the humidity"))

    assert len(supervisor.state.queued_tasks) == 1  # folded, not two competitors
    cont = supervisor.state.queued_tasks[0]
    assert cont.metadata["continuation_addons"] == ["and also the forecast", "and the humidity"]
    assert "and also the forecast" in cont.input_text
    assert "and the humidity" in cont.input_text

    supervisor.cancel_all()
    _drain_until_idle(supervisor)


def test_superseded_completion_is_dropped():
    # A task that completed at an older epoch (its turn was superseded by a
    # continuation merge / barge-in) must not be spoken or remembered.
    supervisor = AgentSupervisor()
    supervisor.state.mode = Mode.ASSISTANT
    supervisor.speech_epoch = 1

    supervisor.publish(
        AgentEvent(EventKind.TASK_COMPLETED, {"task_id": "stale", "text": "stale answer", "speak": True, "epoch": 0})
    )
    supervisor.drain()
    assert "stale answer" not in supervisor.state.spoken_outputs
    assert all("stale answer" != item.text for item in supervisor.memory.all())

    # A completion stamped with the current epoch is still delivered.
    supervisor.publish(
        AgentEvent(EventKind.TASK_COMPLETED, {"task_id": "fresh", "text": "fresh answer", "speak": True, "epoch": 1})
    )
    supervisor.drain()
    assert "fresh answer" in supervisor.state.spoken_outputs


def test_looks_like_continuation_reflects_inflight_turn():
    classifier = ScriptedContinuationClassifier({"make it shorter": CONTINUE, "what time is it": NEW})
    supervisor = _continuation_supervisor(classifier)
    supervisor.state.mode = Mode.ASSISTANT

    # No in-flight turn -> never a continuation.
    assert supervisor.looks_like_continuation("make it shorter") is False

    _seat_assistant_victim(supervisor, "tell me a story")
    assert supervisor.looks_like_continuation("make it shorter") is True
    assert supervisor.looks_like_continuation("what time is it") is False

    supervisor.cancel_all()
    _drain_until_idle(supervisor)


def test_merge_records_origin_and_addon_in_memory():
    # The before-audio merge cancels the victim (which here never ran its worker),
    # so the supervisor must record BOTH the original ask and the add-on so memory
    # doesn't silently lose the prior turn.
    classifier = ScriptedContinuationClassifier({"and also the forecast": CONTINUE})
    supervisor = _continuation_supervisor(classifier)
    supervisor.state.mode = Mode.ASSISTANT
    _seat_assistant_victim(supervisor, "whats the weather", started_speaking=False)

    supervisor._execute_decision(_addon("and also the forecast"))

    texts = [item.text for item in supervisor.memory.all()]
    assert "whats the weather" in texts
    assert "and also the forecast" in texts

    supervisor.cancel_all()
    _drain_until_idle(supervisor)
