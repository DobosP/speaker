"""Tests for the never-stuck controller (ask #4: kill hung work, don't hang).

- Per-mode wall-clock task deadlines + a reap that removes a hung task from
  active_tasks so the controller is never left waiting on a capability that
  won't return.
- The watchdog's tick drives that reap (heal, not just diagnose).
- B3: an escalated ReAct turn stamps LLM_FIRST_TOKEN so it isn't false-flagged
  "llm stuck".
"""
from __future__ import annotations

import threading
import time

from always_on_agent.capabilities import create_default_capabilities
from always_on_agent.events import EventKind, Mode
from always_on_agent.models import IntentDecision, IntentKind
from always_on_agent.react import ReactPlanner
from always_on_agent.supervisor import AgentSupervisor

from core.engines.scripted import ScriptedEngine
from core.metrics import MetricsRecorder
from core.runtime import VoiceRuntime
from core.watchdog import StuckWatchdog


def _seat(sup, *, mode=Mode.ASSISTANT, overdue=True):
    kind = IntentKind.RESEARCH if mode == Mode.RESEARCH else IntentKind.ASSISTANT
    decision = IntentDecision(kind, 0.9, "x", "test", mode=mode)
    task = sup.tasks.create_task(decision)
    with sup._cancel_lock:  # noqa: SLF001 - test seats an in-flight task
        task.deadline_at = (time.monotonic() - 1.0) if overdue else (time.monotonic() + 100.0)
        sup.state.active_tasks[task.task_id] = task
    return task


# --- the reap ----------------------------------------------------------------


def test_reap_cancels_and_removes_overdue_task():
    sup = AgentSupervisor()
    task = _seat(sup, overdue=True)
    assert sup.reap_overdue_tasks() == 1
    assert task.task_id not in sup.state.active_tasks
    assert task.cancel_event.is_set()
    assert f"task {task.task_id} timed out" in sup.state.failures
    sup.drain()


def test_reaped_speakable_turn_speaks_an_apology():
    from always_on_agent.supervisor import _TIMEOUT_APOLOGY

    sup = AgentSupervisor()
    _seat(sup, overdue=True)  # ASSISTANT -> speak=True
    sup.reap_overdue_tasks()
    assert _TIMEOUT_APOLOGY in sup.state.spoken_outputs
    sup.drain()
    assert any(
        e.kind == EventKind.TTS_REQUEST and e.payload.get("text") == _TIMEOUT_APOLOGY
        for e in sup.state.event_log
    )


def test_reaped_silent_turn_gets_no_apology():
    from always_on_agent.supervisor import _TIMEOUT_APOLOGY

    sup = AgentSupervisor()
    decision = IntentDecision(IntentKind.DICTATION, 0.9, "notes", "test", mode=Mode.DICTATION, speak=False)
    task = sup.tasks.create_task(decision)
    with sup._cancel_lock:
        task.deadline_at = time.monotonic() - 1.0
        sup.state.active_tasks[task.task_id] = task
    sup.reap_overdue_tasks()
    assert _TIMEOUT_APOLOGY not in sup.state.spoken_outputs
    sup.drain()


def test_non_mapping_task_timeouts_is_ignored_not_fatal():
    sup = AgentSupervisor(task_timeouts="not a dict")  # must not raise
    assert sup._timeout_for(Mode.ASSISTANT) == 25.0
    AgentSupervisor(task_timeouts=["a", "b"])  # also fine


def test_event_bus_survives_a_raising_handler():
    # A handler bug must not kill the single bus thread (which would make the
    # whole assistant go dead). The reap/continuation race could raise here.
    from always_on_agent.event_bus import EventBus
    from always_on_agent.events import AgentEvent

    bus = EventBus()
    seen = []

    def _raise(_event):
        raise RuntimeError("boom")

    bus.subscribe(_raise)
    bus.subscribe(lambda e: seen.append(e.kind))
    bus.publish(AgentEvent(EventKind.STT_FINAL, {"text": "hi"}))
    bus.drain()
    assert seen, "a raising handler suppressed the next handler / killed the bus"
    # The bus is still alive for the next event.
    bus.publish(AgentEvent(EventKind.STT_FINAL, {"text": "again"}))
    bus.drain()
    assert len(seen) == 2


def test_react_renews_its_deadline_for_the_agent_budget():
    renewed = []

    class _LLM:
        def generate(self, prompt, *, system=None):
            return "FINAL: done"

        def stream(self, prompt, *, system=None):
            yield "FINAL: done"

    planner = ReactPlanner(_LLM(), create_default_capabilities())
    planner.run("do a thing", {"renew_deadline": lambda secs: renewed.append(secs)})
    assert renewed and renewed[0] >= 60.0


def test_renew_deadline_only_extends_an_existing_one():
    from always_on_agent.tasks import AgentTask, TaskRuntime

    task = AgentTask(mode=Mode.ASSISTANT, input_text="x")
    task.deadline_at = 0.0
    TaskRuntime._renew_deadline(task, 100.0)
    assert task.deadline_at == 0.0  # a disabled deadline stays disabled
    task.deadline_at = time.monotonic() + 5.0
    old = task.deadline_at
    TaskRuntime._renew_deadline(task, 100.0)
    assert task.deadline_at > old  # an existing one is pushed out


def test_reap_leaves_a_live_task_alone():
    sup = AgentSupervisor()
    task = _seat(sup, overdue=False)
    assert sup.reap_overdue_tasks() == 0
    assert task.task_id in sup.state.active_tasks
    assert not task.cancel_event.is_set()
    sup.cancel_all()
    sup.drain()


def test_reap_republishes_cancellation_for_the_bus_thread_to_drain():
    sup = AgentSupervisor()
    _seat(sup, overdue=True)
    sup.reap_overdue_tasks()
    sup.drain()  # the bus thread handles the republished TASK_CANCELLED
    assert any(
        e.kind == EventKind.TASK_CANCELLED and e.payload.get("reaped")
        for e in sup.state.event_log
    )


def test_start_task_stamps_a_per_mode_deadline():
    sup = AgentSupervisor(task_timeouts={"assistant": 12.0})
    sup.state.mode = Mode.ASSISTANT
    decision = IntentDecision(IntentKind.ASSISTANT, 0.9, "hi", "test", mode=Mode.ASSISTANT)
    task = sup.tasks.create_task(decision)
    before = time.monotonic()
    sup._start_task(task)
    assert task.deadline_at >= before + 11.0  # ~12s out
    sup.cancel_all()
    sup.drain()


def test_timeout_zero_disables_the_deadline():
    sup = AgentSupervisor(task_timeouts={"assistant": 0.0})
    sup.state.mode = Mode.ASSISTANT
    decision = IntentDecision(IntentKind.ASSISTANT, 0.9, "hi", "test", mode=Mode.ASSISTANT)
    task = sup.tasks.create_task(decision)
    sup._start_task(task)
    assert task.deadline_at == 0.0  # never reaped
    assert sup.reap_overdue_tasks() == 0
    sup.cancel_all()
    sup.drain()


# --- watchdog drives the reap ------------------------------------------------


def test_watchdog_tick_invokes_on_tick():
    calls = []
    wd = StuckWatchdog(MetricsRecorder(), on_tick=lambda: calls.append(1))
    wd.tick()
    assert calls == [1]


def test_watchdog_on_tick_error_does_not_kill_the_loop():
    def _boom():
        raise RuntimeError("reap failed")

    wd = StuckWatchdog(MetricsRecorder(), on_tick=_boom)
    wd.tick()  # must not raise


# --- B3: escalated turns stamp LLM_FIRST_TOKEN -------------------------------


def test_react_fires_first_token_hook_on_first_token():
    fired = []

    class _LLM:
        def generate(self, prompt, *, system=None):
            return "FINAL: done"

        def stream(self, prompt, *, system=None):
            yield "FINAL: "
            yield "done"

    planner = ReactPlanner(
        _LLM(), create_default_capabilities(), first_token_hook=lambda: fired.append(1)
    )
    planner.run("do a thing", {})
    assert fired, "the ReAct planner did not stamp its first token"


# --- end-to-end: a hung turn is reaped, the controller returns to idle -------


def test_runtime_reaps_a_hung_turn_so_the_controller_recovers():
    class _HangLLM:
        def __init__(self):
            self.release = threading.Event()

        def generate(self, prompt, *, system=None, images=None):
            self.release.wait(timeout=5.0)
            return "late"

        def stream(self, prompt, *, system=None, images=None):
            self.release.wait(timeout=5.0)  # blocks before any token
            yield "late"

    llm = _HangLLM()
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine, llm, start_mode=Mode.ASSISTANT, stream_tts=True,
        task_timeouts={"assistant": 0.1},
    )
    runtime.start(run_bus=False)
    try:
        engine.final("tell me something")
        # Drain so the task starts on its worker thread and then hangs.
        deadline = time.time() + 2.0
        while time.time() < deadline and not runtime.supervisor.state.active_tasks:
            runtime.bus.drain()
            time.sleep(0.01)
        assert runtime.supervisor.state.active_tasks, "the hung task never started"

        time.sleep(0.15)  # let the 0.1s deadline lapse
        runtime._watchdog.tick()  # -> on_tick -> reap (publishes TASK_CANCELLED)
        runtime.bus.drain()  # process the cancellation
        assert runtime.supervisor.state.active_tasks == {}, "hung task was not reaped"
    finally:
        llm.release.set()
        runtime.stop()
