"""Tests for proactive ("always-listening") follow-ups.

Pure cadence logic is tested directly; the timer is bypassed (the supervisor's
_emit_followup is invoked directly) so behavior is deterministic without sleeps.
A large delay_sec keeps the real timer from firing mid-test.
"""

from __future__ import annotations

from always_on_agent.followups import FollowupConfig, FollowupState
from always_on_agent.supervisor import AgentSupervisor

from core.engines.scripted import ScriptedEngine
from core.llm import EchoLLM
from core.runtime import VoiceRuntime


def test_followup_state_cycles_markers_and_caps():
    state = FollowupState(markers=("a", "b"), max_followups=3)
    assert state.can_continue()
    assert [state.next_marker() for _ in range(3)] == ["a", "b", "a"]
    assert not state.can_continue()
    state.reset()
    assert state.can_continue()
    assert state.count == 0


def test_followup_config_from_dict():
    cfg = FollowupConfig.from_dict(
        {"enabled": True, "delay_sec": 1.5, "max_followups": 2, "markers": ["x"]}
    )
    assert cfg.enabled is True
    assert cfg.delay_sec == 1.5
    assert cfg.max_followups == 2
    assert cfg.markers == ("x",)


def test_reset_followups_clears_timer_and_count():
    sup = AgentSupervisor(
        followup_config=FollowupConfig(enabled=True, delay_sec=100, max_followups=2)
    )
    sup._followup_state.count = 1
    sup._schedule_followup()
    assert sup._followup_timer is not None
    sup._reset_followups()
    assert sup._followup_timer is None
    assert sup._followup_state.count == 0
    sup.shutdown()


def _runtime_with_followups(max_followups=2):
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(),
        followup_config=FollowupConfig(
            enabled=True, delay_sec=100, max_followups=max_followups
        ),
    )
    runtime.start(run_bus=False)
    return runtime, engine


def test_followup_speaks_marker_after_an_answer():
    runtime, engine = _runtime_with_followups()

    engine.final("hello")
    assert runtime.wait_idle()
    assert engine.spoken == ["You said: hello"]
    # The assistant spoke, so a follow-up timer should now be armed.
    assert runtime.supervisor._followup_timer is not None

    # Simulate the silence timer firing.
    runtime.supervisor._emit_followup()
    assert runtime.wait_idle()
    assert engine.spoken[-1] == "You said: [silent]"

    runtime.stop()


def test_followups_stop_at_max():
    runtime, engine = _runtime_with_followups(max_followups=2)

    engine.final("hello")
    assert runtime.wait_idle()

    markers_spoken = []
    for _ in range(4):  # ask for more than the cap
        runtime.supervisor._emit_followup()
        runtime.wait_idle()
        markers_spoken.append(engine.spoken[-1])

    # Only two distinct follow-up markers fire; further ticks are no-ops.
    assert markers_spoken[0] == "You said: [silent]"
    assert markers_spoken[1] == "You said: [no response]"
    assert markers_spoken[2] == "You said: [no response]"  # capped: no new speech
    runtime.stop()


def test_user_speech_cancels_pending_followup():
    runtime, engine = _runtime_with_followups()

    engine.final("hello")
    assert runtime.wait_idle()
    assert runtime.supervisor._followup_timer is not None

    # User speaks again before the timer fires: cadence resets.
    engine.final("are you there")
    assert runtime.wait_idle()
    assert runtime.supervisor._followup_state.count == 0
    runtime.stop()
