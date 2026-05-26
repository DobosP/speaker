from __future__ import annotations

import time

from always_on_agent.diagnostics import summarize
from always_on_agent.events import AgentEvent, EventKind, Mode
from always_on_agent.bridge import TranscriptBridge
from always_on_agent.replay import replay_records
from always_on_agent.runtime import AlwaysOnAgentRuntime
from always_on_agent.models import IntentKind
from always_on_agent.speech_analyzer import LiveSpeechAnalyzer, ModePolicy
from always_on_agent.supervisor import AgentSupervisor


def _decide(text: str, mode: Mode = Mode.PASSIVE, *, is_final: bool = True):
    """Run the analyzer's observe->decide directly (unit-level, no supervisor)."""
    analyzer = LiveSpeechAnalyzer()
    return analyzer.decide(analyzer.observe(text, is_final=is_final), mode)


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


# ── Direct decide() unit tests (the rest of this file goes through the supervisor) ──
def test_decide_exact_control_stop():
    assert _decide("stop").kind == IntentKind.STOP


def test_decide_bilingual_confirm_and_deny():
    assert _decide("da").kind == IntentKind.CONFIRM
    assert _decide("nu").kind == IntentKind.DENY


def test_decide_command_prefix_requires_confirmation():
    decision = _decide("open browser", Mode.COMMAND)
    assert decision.kind == IntentKind.COMMAND
    assert decision.requires_confirmation is True


def test_decide_partial_non_control_is_ignored():
    # A non-control partial must not start a task before the transcript finalizes.
    assert _decide("assistant help me plan", is_final=False).kind == IntentKind.IGNORE
