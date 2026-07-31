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

from always_on_agent.capabilities import (
    CapabilityResult,
    CapabilitySpec,
    create_default_capabilities,
)
from always_on_agent.events import EventKind, Mode
from always_on_agent.models import IntentDecision, IntentKind
from always_on_agent.react import ReactPlanner
from always_on_agent.session_actor import SessionActor
from always_on_agent.supervisor import AgentSupervisor
from always_on_agent.tasks import TaskRuntime, TaskState

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


def test_reap_cancels_mutation_waiting_for_provider_capacity():
    sup = AgentSupervisor()
    sup.tasks = TaskRuntime(
        sup.publish,
        sup.capabilities,
        max_active_tasks=1,
        session_actor=sup.session_actor,
    )
    provider_called = threading.Event()

    def mutate(_query, _context):
        provider_called.set()
        return CapabilityResult(True, "must not execute")

    sup.capabilities.register(
        "timeout.mutate",
        mutate,
        spec=CapabilitySpec(
            "timeout.mutate",
            "timeout mutation",
            side_effecting=True,
        ),
    )
    task = _seat(sup, overdue=True)
    slot = sup.tasks._invocation_slots  # noqa: SLF001 - deterministic capacity
    assert slot.acquire(blocking=False)
    invocation_done = threading.Event()

    def invoke() -> None:
        try:
            sup.tasks._invoke(task, "timeout.mutate")  # noqa: SLF001
        except BaseException:
            pass
        finally:
            invocation_done.set()

    invocation = threading.Thread(target=invoke, daemon=True)
    invocation.start()
    deadline = time.monotonic() + 1.0
    while (
        sup.session_actor.active_tool_count == 0
        and time.monotonic() < deadline
    ):
        time.sleep(0.005)
    assert sup.session_actor.active_tool_count == 1

    assert sup.reap_overdue_tasks() == 1
    # Release the slot into the cancellation race. The waiting mutation must
    # re-check its actor-owned Event before any provider can be claimed.
    slot.release()
    assert invocation_done.wait(timeout=1.0)
    invocation.join(timeout=1.0)

    assert not provider_called.is_set()
    assert sup.session_actor.active_tool_count == 0
    sup.drain()
    sup.shutdown()


def test_reap_fences_tool_start_before_claiming_task_terminal(monkeypatch):
    sup = AgentSupervisor()
    task = _seat(sup, overdue=True)
    assert task.identity is not None
    tool = sup.session_actor.admit_tool(
        task_id=task.task_id,
        identity=task.identity,
        tool_call_id="timeout-gap-tool",
        idempotency_key="timeout-gap-effect",
    )
    actor_fence_complete = threading.Event()
    claim_attempt_done = threading.Event()
    claim_results: list[bool] = []
    original_fence = sup.tasks.fence_task_tool_calls

    def held_fence(task_id, identity):
        result = original_fence(task_id, identity)
        actor_fence_complete.set()
        assert claim_attempt_done.wait(timeout=1.0)
        return result

    monkeypatch.setattr(
        sup.tasks,
        "fence_task_tool_calls",
        held_fence,
    )

    def race_provider_start() -> None:
        assert actor_fence_complete.wait(timeout=1.0)
        claim_results.append(
            sup.session_actor.claim_tool_start(
                tool.tool_call_id,
                tool.identity,
            )
        )
        claim_attempt_done.set()

    claimant = threading.Thread(target=race_provider_start, daemon=True)
    claimant.start()

    assert sup.reap_overdue_tasks() == 1
    claimant.join(timeout=1.0)

    assert claim_results == [False]
    assert tool.cancel_event.is_set()
    sup.session_actor.finish_tool(tool.tool_call_id, tool.identity)
    sup.drain()
    sup.shutdown()


def test_completed_task_wins_just_before_timeout_reap(monkeypatch):
    from always_on_agent.supervisor import _TIMEOUT_APOLOGY

    sup = AgentSupervisor(defer_output_until_tts_admission=True)
    task = _seat(sup, overdue=True)
    task.output_text = "Completion already won."
    task.metadata["result_data"] = {}
    completion_claimed = threading.Event()
    release_completion = threading.Event()
    original_claim_terminal = task.claim_terminal

    def held_completion_claim(desired):
        result = original_claim_terminal(desired)
        if desired == TaskState.COMPLETED:
            completion_claimed.set()
            assert release_completion.wait(timeout=1.0)
        return result

    monkeypatch.setattr(task, "claim_terminal", held_completion_claim)
    completion = threading.Thread(
        target=sup.tasks._publish_completed,  # noqa: SLF001
        args=(task,),
        daemon=True,
    )
    completion.start()
    assert completion_claimed.wait(timeout=1.0)

    assert sup.reap_overdue_tasks() == 0
    assert sup.state.active_tasks.get(task.task_id) is task
    assert sup.session_actor.identity_for_task(task.task_id) == task.identity
    assert _TIMEOUT_APOLOGY not in sup.state.spoken_outputs

    release_completion.set()
    completion.join(timeout=1.0)
    assert not completion.is_alive()
    sup.drain()
    tts = next(
        event
        for event in sup.state.event_log
        if (
            event.kind == EventKind.TTS_REQUEST
            and event.payload.get("task_id") == task.task_id
        )
    )
    admission = sup.admit_playback(
        tts.payload,
        task_id=task.task_id,
        auxiliary_tts=False,
        aux_tts_id="",
    )
    assert admission is not None
    sup.finish_playback(admission.playback_admission_id)
    assert _TIMEOUT_APOLOGY not in sup.state.spoken_outputs
    sup.shutdown()


def test_reap_fences_task_while_worker_is_about_to_admit_tool(monkeypatch):
    sup = AgentSupervisor()
    provider_called = threading.Event()

    def mutate(_query, _context):
        provider_called.set()
        return CapabilityResult(True, "must not execute")

    sup.capabilities.register(
        "timeout.pre_admit",
        mutate,
        spec=CapabilitySpec(
            "timeout.pre_admit",
            "pre-admission timeout mutation",
            side_effecting=True,
        ),
    )
    task = _seat(sup, overdue=True)
    actor = sup.session_actor
    admit_entered = threading.Event()
    release_admit = threading.Event()
    original_admit = actor.admit_tool

    def held_admit(**kwargs):
        admit_entered.set()
        assert release_admit.wait(timeout=1.0)
        return original_admit(**kwargs)

    monkeypatch.setattr(actor, "admit_tool", held_admit)
    invocation_done = threading.Event()
    outcome: list[CapabilityResult] = []

    def invoke() -> None:
        try:
            outcome.append(
                sup.tasks._invoke(task, "timeout.pre_admit")  # noqa: SLF001
            )
        finally:
            invocation_done.set()

    invocation = threading.Thread(target=invoke, daemon=True)
    invocation.start()
    assert admit_entered.wait(timeout=1.0)

    assert sup.reap_overdue_tasks() == 1
    release_admit.set()
    assert invocation_done.wait(timeout=1.0)
    invocation.join(timeout=1.0)

    assert not provider_called.is_set()
    assert actor.active_tool_count == 0
    assert len(outcome) == 1 and not outcome[0].ok
    sup.drain()
    sup.shutdown()


def test_reaped_speakable_turn_speaks_an_apology():
    from always_on_agent.supervisor import _TIMEOUT_APOLOGY

    sup = AgentSupervisor()
    task = _seat(sup, overdue=True)  # ASSISTANT -> speak=True
    sup.reap_overdue_tasks()
    assert _TIMEOUT_APOLOGY in sup.state.spoken_outputs
    sup.drain()
    tts = next(
        e
        for e in sup.state.event_log
        if (
            e.kind == EventKind.TTS_REQUEST
            and e.payload.get("text") == _TIMEOUT_APOLOGY
        )
    )
    admission = sup.admit_playback(
        tts.payload,
        task_id=task.task_id,
        auxiliary_tts=True,
        aux_tts_id=str(tts.payload["aux_tts_id"]),
    )
    assert admission is not None
    assert admission.identity != task.identity
    sup.note_aux_tts_admitted(str(tts.payload["aux_tts_id"]))
    sup.finish_playback(admission.playback_admission_id)
    sup.shutdown()


def test_timeout_capacity_omits_apology_but_keeps_cancellation_lifecycle():
    from always_on_agent.supervisor import _TIMEOUT_APOLOGY

    sup = AgentSupervisor()
    actor = SessionActor(
        session_id="timeout-capacity",
        max_remembered_tasks=1,
    )
    sup.session_actor = actor
    sup.tasks = TaskRuntime(
        sup.publish,
        sup.capabilities,
        session_actor=actor,
    )
    task = _seat(sup, overdue=True)
    assert task.identity is not None
    assert actor.admit_task(task.task_id, task.identity)
    worker_saw_cancel = threading.Event()
    release_worker = threading.Event()
    worker_drained = threading.Event()

    def drain_worker_admission() -> None:
        assert task.cancel_event.wait(timeout=1.0)
        worker_saw_cancel.set()
        assert release_worker.wait(timeout=1.0)
        sup.tasks.retire_task_admission(task)
        worker_drained.set()

    worker = threading.Thread(target=drain_worker_admission, daemon=True)
    worker.start()

    assert sup.reap_overdue_tasks() == 1
    assert worker_saw_cancel.wait(timeout=1.0)
    assert actor.snapshot().remembered_tasks == 1
    assert _TIMEOUT_APOLOGY not in sup.state.spoken_outputs
    sup.drain()
    terminal = [
        event
        for event in sup.state.event_log
        if (
            event.kind == EventKind.TASK_CANCELLED
            and event.payload.get("task_id") == task.task_id
        )
    ]
    assert len(terminal) == 1 and terminal[0].payload.get("reaped") is True
    assert not any(
        event.kind == EventKind.TTS_REQUEST
        and event.payload.get("text") == _TIMEOUT_APOLOGY
        for event in sup.state.event_log
    )

    release_worker.set()
    assert worker_drained.wait(timeout=1.0)
    worker.join(timeout=1.0)
    assert actor.snapshot().remembered_tasks == 0
    replacement = sup.tasks.create_task(
        IntentDecision(
            IntentKind.ASSISTANT,
            1.0,
            "replacement",
            "test",
            mode=Mode.ASSISTANT,
        )
    )
    assert replacement.identity is not None
    sup.shutdown()


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


def test_react_prefers_task_scoped_first_token_hook():
    default_fired = []
    scoped_fired = []

    class _LLM:
        def generate(self, prompt, *, system=None):
            return "FINAL: done"

        def stream(self, prompt, *, system=None):
            yield "FINAL: done"

    planner = ReactPlanner(
        _LLM(),
        create_default_capabilities(),
        first_token_hook=lambda: default_fired.append(1),
    )
    planner.run("do a thing", {"first_token_hook": lambda: scoped_fired.append(1)})

    assert scoped_fired
    assert not default_fired


# --- end-to-end: a hung turn is reaped, the controller returns to idle -------


def test_runtime_reaps_a_hung_turn_so_the_controller_recovers():
    class _HangLLM:
        def __init__(self):
            self.release = threading.Event()
            self.started = threading.Event()
            self.finished = threading.Event()

        def generate(self, prompt, *, system=None, images=None):
            self.release.wait(timeout=5.0)
            return "late"

        def stream(self, prompt, *, system=None, images=None):
            self.started.set()
            try:
                self.release.wait(timeout=5.0)  # blocks before any token
                yield "late"
            finally:
                self.finished.set()

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
        assert llm.started.wait(timeout=2.0), "the provider never reached its gate"
        task_id, task = next(iter(runtime.supervisor.state.active_tasks.items()))
        task_runtime = runtime.supervisor.tasks
        with task_runtime._threads_lock:  # noqa: SLF001 - capture the real worker
            coordinator = task_runtime._threads[task_id]  # noqa: SLF001

        # Expire deterministically instead of sleeping past the configured
        # deadline.  The watchdog still drives the production reap path.
        task.deadline_at = time.monotonic() - 1.0
        runtime._watchdog.tick()  # -> on_tick -> reap (publishes TASK_CANCELLED)
        assert runtime.supervisor.state.active_tasks == {}, "hung task was not reaped"

        # Join the captured object, not merely the registry count: _reap removes
        # the registry entry just before the coordinator actually returns.
        coordinator.join(timeout=2.0)
        assert not coordinator.is_alive(), "timed-out coordinator did not exit"
        assert task_runtime.active_count == 0
        assert not llm.release.is_set()
        assert not llm.finished.is_set(), "provider completion caused coordinator exit"

        # Drain only after the coordinator has exited, then inspect the complete
        # lifecycle.  The watchdog owns the authoritative reaped cancellation;
        # the coordinator must not publish a second terminal event while exiting.
        runtime.bus.drain()
        terminal_events = [
            event
            for event in runtime.supervisor.state.event_log
            if event.payload.get("task_id") == task_id
            and event.kind in {
                EventKind.TASK_CANCELLED,
                EventKind.TASK_COMPLETED,
                EventKind.TASK_FAILED,
            }
        ]
        assert [event.kind for event in terminal_events] == [EventKind.TASK_CANCELLED]
        assert terminal_events[0].payload.get("reaped") is True
    finally:
        llm.release.set()
        llm.finished.wait(timeout=2.0)
        runtime.stop()
