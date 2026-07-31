"""Headless ownership tests for the bounded post-recognition session actor."""

from __future__ import annotations

from threading import Event, Thread
import time

import pytest

import always_on_agent.tasks as tasks_module
from always_on_agent.capabilities import (
    CapabilityRegistry,
    CapabilityResult,
    CapabilitySpec,
)
from always_on_agent.acoustic import AcousticLineage, AcousticSpan
from always_on_agent.events import AgentEvent, EventKind, Mode
from always_on_agent.models import IntentDecision, IntentKind
from always_on_agent.session_actor import (
    AdmissionRejected,
    SessionActor,
    TaskIdentity,
)
from always_on_agent.tasks import AgentTask, TaskRuntime
from always_on_agent.supervisor import AgentSupervisor
from core.playback_history import PlaybackContext, PlaybackHistory


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _assistant_decision(text: str = "hello") -> IntentDecision:
    return IntentDecision(
        IntentKind.ASSISTANT,
        1.0,
        text,
        "test",
        mode=Mode.ASSISTANT,
    )


def test_revision_identity_reaches_task_tts_and_playback_admission():
    actor = SessionActor(session_id="session-test")
    turn = actor.open_turn(revision=7)
    published = []
    registry = CapabilityRegistry()

    def answer(_query, context):
        context["emit_speech"]("A streamed sentence.")
        return CapabilityResult(
            True,
            "A streamed sentence.",
            data={"streamed": True},
        )

    registry.register(
        "assistant.answer",
        answer,
        spec=CapabilitySpec("assistant.answer", "answer"),
    )
    runtime = TaskRuntime(
        published.append,
        registry,
        stream_tts=True,
        session_actor=actor,
    )
    task = runtime.create_task(_assistant_decision(), turn_identity=turn)

    assert task.identity == TaskIdentity(
        session_id="session-test",
        turn_id=turn.turn_id,
        revision=7,
        cancel_generation=0,
        binding_id=1,
    )
    assert runtime.start(task) is True
    assert _wait_until(
        lambda: any(event.kind == EventKind.TASK_COMPLETED for event in published)
    )

    started = next(event for event in published if event.kind == EventKind.TASK_STARTED)
    tts = next(event for event in published if event.kind == EventKind.TTS_REQUEST)
    completed = next(
        event for event in published if event.kind == EventKind.TASK_COMPLETED
    )
    for event in (started, tts, completed):
        assert TaskIdentity.from_payload(event.payload) == task.identity

    admission = actor.admit_playback(
        task_id=task.task_id,
        identity=TaskIdentity.from_payload(tts.payload),
    )
    assert admission is not None
    history = PlaybackHistory()
    fragment = history.register(
        task_id=task.task_id,
        epoch=task.speech_epoch,
        input_generation=1,
        followup_generation=0,
        text="A streamed sentence.",
        remember=True,
        is_followup=False,
        streaming=False,
        playback_admission_id=admission.playback_admission_id,
        **admission.identity.to_payload(),
    )
    assert history.fragment_context(fragment) == PlaybackContext(
        task_id=task.task_id,
        epoch=task.speech_epoch,
        input_generation=1,
        remember=True,
        session_id="session-test",
        turn_id=turn.turn_id,
        revision=7,
        cancel_generation=0,
        binding_id=1,
    )
    assert (
        history.fragment_playback_admission(fragment)
        == admission.playback_admission_id
    )
    actor.finish_playback(admission.playback_admission_id)
    assert actor.snapshot().active_playbacks == 0


def test_supervisor_binds_acoustic_revision_before_task_and_tts():
    supervisor = AgentSupervisor()
    supervisor.state.mode = Mode.ASSISTANT
    lineage = AcousticLineage.single(
        AcousticSpan(
            stream_id="capture-stream",
            utterance_id="utterance-5",
            capture_generation=5,
        )
    )
    try:
        supervisor.handle_event(
            # The endpoint supplies acoustic lineage/revision; it does not
            # allocate session or cancellation identity itself.
            AgentEvent.final(
                "hello assistant",
                acoustic=lineage,
                revision=9,
            )
        )
        assert _wait_until(
            lambda: (
                supervisor.drain() >= 0
                and any(
                    event.kind == EventKind.TTS_REQUEST
                    for event in supervisor.state.event_log
                )
            )
        )
        task_started = next(
            event
            for event in supervisor.state.event_log
            if event.kind == EventKind.TASK_STARTED
        )
        tts = next(
            event
            for event in supervisor.state.event_log
            if event.kind == EventKind.TTS_REQUEST
        )
        started_identity = TaskIdentity.from_payload(task_started.payload)
        assert started_identity.revision == 9
        assert started_identity.session_id == supervisor.session_actor.session_id
        assert TaskIdentity.from_payload(tts.payload) == started_identity
    finally:
        supervisor.shutdown()


def test_task_playback_and_tool_admission_are_bounded():
    actor = SessionActor(
        session_id="bounded",
        max_active_tasks=1,
        max_active_tools=1,
        max_active_playbacks=1,
    )
    first_turn = actor.open_turn(revision=1)
    second_turn = actor.open_turn(revision=2)
    first = actor.bind_task("first", first_turn)
    second = actor.bind_task("second", second_turn)

    assert actor.admit_task("first", first) is True
    assert actor.admit_task("second", second) is False
    actor.finish_task("first", first)
    assert actor.admit_task("second", second) is True

    playback = actor.admit_playback(task_id="second", identity=second)
    assert playback is not None
    assert actor.admit_playback(task_id="second", identity=second) is None
    actor.finish_playback(playback.playback_admission_id)
    assert actor.admit_playback(task_id="second", identity=second) is not None

    tool = actor.admit_tool(
        task_id="second",
        identity=second,
        tool_call_id="tool-1",
        idempotency_key="same-effect",
    )
    with pytest.raises(AdmissionRejected, match="idempotency key"):
        actor.admit_tool(
            task_id="second",
            identity=second,
            tool_call_id="tool-2",
            idempotency_key="same-effect",
        )
    assert actor.cancel_speech() == 1
    assert tool.cancel_event.is_set() is True
    assert actor.claim_tool_start("tool-1", tool.identity) is False
    assert actor.cancel_tool("tool-1", tool.identity) is True
    assert tool.cancel_event.is_set() is True
    assert actor.admit_playback(task_id="second", identity=second) is None


def test_turn_handle_cancel_fences_effects_and_reports_exact_children():
    actor = SessionActor(
        session_id="turn-handle",
        max_active_tasks=2,
        max_active_providers=2,
        max_active_tools=2,
        max_active_playbacks=2,
        max_remembered_tasks=2,
    )
    causal_turn = actor.open_turn(revision=1)
    handle = actor.bind_turn("owned", causal_turn)
    sibling = actor.bind_turn("sibling", causal_turn)

    assert actor.admit_task(handle.task_id, handle.identity)
    provider = actor.admit_provider(
        task_id=handle.task_id,
        identity=handle.identity,
    )
    assert provider is not None
    tool = actor.admit_tool(
        task_id=handle.task_id,
        identity=handle.identity,
        tool_call_id="held-tool",
        idempotency_key="held-effect",
    )
    playback = actor.admit_playback(
        task_id=handle.task_id,
        identity=handle.identity,
    )
    assert playback is not None

    result = handle.cancel_and_drain(0.0)

    assert result.effects_fenced is True
    assert result.drained is False
    assert result.timed_out is True
    assert result.external_owners == 0
    assert result.active_tasks == 1
    assert result.active_providers == 1
    assert result.active_tools == 1
    assert result.active_playbacks == 1
    assert result.remaining == 4
    assert tool.cancel_event.is_set() is True

    # Closing this exact handle cannot affect a sibling turn, while every new
    # effect on the cancelled scope fails closed.
    assert actor.task_effects_allowed(sibling.task_id, sibling.identity)
    assert not actor.admit_task(handle.task_id, handle.identity)
    assert (
        actor.admit_provider(
            task_id=handle.task_id,
            identity=handle.identity,
        )
        is None
    )
    assert (
        actor.admit_playback(
            task_id=handle.task_id,
            identity=handle.identity,
        )
        is None
    )
    with pytest.raises(AdmissionRejected):
        actor.admit_tool(
            task_id=handle.task_id,
            identity=handle.identity,
            tool_call_id="late-tool",
            idempotency_key="late-effect",
        )

    actor.finish_task(handle.task_id, handle.identity)
    actor.finish_provider(provider.provider_admission_id)
    actor.finish_tool(tool.tool_call_id, tool.identity)
    actor.finish_playback(playback.playback_admission_id)

    drained = handle.wait_drained(0.1)
    assert drained.drained is True
    assert drained.timed_out is False
    assert drained.remaining == 0
    assert handle.cancel() is False

    # The actor-issued binding nonce makes a same-ID replacement exact and
    # leaves the old handle monotonic.
    replacement = actor.bind_turn(handle.task_id, causal_turn)
    assert replacement.identity != handle.identity
    assert replacement.identity.binding_id > handle.identity.binding_id
    assert actor.admit_task(replacement.task_id, replacement.identity)
    replacement_provider = actor.admit_provider(
        task_id=replacement.task_id,
        identity=replacement.identity,
    )
    assert replacement_provider is not None
    assert actor.finish_task(replacement.task_id, handle.identity) is False
    assert actor.release_task_binding(
        replacement.task_id,
        handle.identity,
    ) is False
    assert actor.task_effects_allowed(
        replacement.task_id,
        replacement.identity,
    )
    assert handle.wait_drained(0.0).drained is True
    assert handle.wait_drained(0.0).active_providers == 0
    replacement.cancel()
    actor.finish_task(replacement.task_id, replacement.identity)
    actor.finish_provider(replacement_provider.provider_admission_id)
    assert replacement.wait_drained(0.1).drained is True


def test_agent_task_cancel_alone_closes_late_mutation_admission():
    actor = SessionActor(session_id="task-cancel-tool-fence")
    registry = CapabilityRegistry()
    executions = 0

    def mutate(_query, _context):
        nonlocal executions
        executions += 1
        return CapabilityResult(True, "must not execute")

    registry.register(
        "command.stage",
        mutate,
        spec=CapabilitySpec(
            "command.stage",
            "mutation",
            side_effecting=True,
        ),
    )
    runtime = TaskRuntime(
        lambda _event: None,
        registry,
        session_actor=actor,
    )
    task = runtime.create_task(
        IntentDecision(
            IntentKind.COMMAND,
            1.0,
            "cancel this scope",
            "test",
            mode=Mode.COMMAND,
        )
    )
    handle = task.turn_handle
    assert handle is not None

    task.cancel()
    result = runtime._invoke(task, "command.stage")  # noqa: SLF001

    assert result.ok is False
    assert "bound task" in result.error
    assert executions == 0
    assert handle.wait_drained(0.0).drained is True


def test_cancel_and_claim_terminal_without_callback_fences_unstarted_tool():
    actor = SessionActor(session_id="terminal-cancel-tool-fence")
    runtime = TaskRuntime(
        lambda _event: None,
        CapabilityRegistry(),
        session_actor=actor,
    )
    task = runtime.create_task(_assistant_decision("cancel terminally"))
    assert task.identity is not None
    handle = task.turn_handle
    assert handle is not None
    tool = actor.admit_tool(
        task_id=task.task_id,
        identity=task.identity,
        tool_call_id="terminal-tool",
        idempotency_key="terminal-effect",
    )

    assert task.cancel_and_claim_terminal() is True
    assert tool.cancel_event.is_set() is True
    assert actor.claim_tool_start(tool.tool_call_id, tool.identity) is False
    pending = handle.wait_drained(0.0)
    assert pending.drained is False
    assert pending.active_tools == 1

    assert actor.finish_tool(tool.tool_call_id, tool.identity) is True
    assert handle.wait_drained(0.1).drained is True


@pytest.mark.parametrize(
    "timeout",
    [-1.0, float("nan"), float("inf"), 1e100],
)
def test_cancel_and_drain_rejects_invalid_timeout_before_fencing(timeout):
    actor = SessionActor(session_id="invalid-drain-timeout")
    handle = actor.bind_turn("still-open", actor.open_turn(revision=1))

    with pytest.raises(ValueError, match="finite non-negative"):
        handle.cancel_and_drain(timeout)

    assert handle.cancelled is False
    assert actor.task_effects_allowed(handle.task_id, handle.identity)
    pending = handle.wait_drained(0.0)
    assert pending.effects_fenced is False
    assert pending.external_owners == 1

    assert actor.release_task_binding(handle.task_id, handle.identity)
    assert handle.wait_drained(0.1).drained is True


def test_playback_requires_exact_actor_bound_task_identity():
    actor = SessionActor(session_id="playback-auth")
    bound_turn = actor.open_turn(revision=3)
    identity = actor.bind_task("real-task", bound_turn)
    unbound_turn = actor.open_turn(revision=4)
    fabricated = TaskIdentity(
        session_id=actor.session_id,
        turn_id=unbound_turn.turn_id,
        revision=unbound_turn.revision,
        cancel_generation=actor.cancel_generation,
        binding_id=identity.binding_id,
    )

    # Current session/generation alone cannot authenticate playback. The exact
    # actor-owned task binding must match both the task ID and turn revision.
    assert (
        actor.admit_playback(task_id="real-task", identity=fabricated) is None
    )
    assert (
        actor.admit_playback(task_id="invented-task", identity=identity) is None
    )
    assert actor.admit_playback(task_id="real-task", identity=identity) is not None


@pytest.mark.parametrize("failure_point", ("construct", "start"))
def test_supervisor_rolls_back_synchronous_task_thread_failure(
    monkeypatch,
    failure_point,
):
    supervisor = AgentSupervisor()
    task = supervisor.tasks.create_task(_assistant_decision("cannot start"))
    task.metadata["latency_policy"] = "ack_then_think"
    handle = task.turn_handle
    assert handle is not None

    class _FailingThread:
        def __init__(self, *args, **kwargs):
            if failure_point == "construct":
                raise RuntimeError("thread construction failed")

        def start(self):
            raise RuntimeError("thread start failed")

    monkeypatch.setattr(tasks_module, "Thread", _FailingThread)

    with pytest.raises(RuntimeError, match="thread"):
        supervisor._start_task(task)  # noqa: SLF001

    assert task.task_id not in supervisor.state.active_tasks
    assert task.task_id not in supervisor.state.pending_audio_tasks
    assert not any(
        pending[4] == task.task_id
        for pending in supervisor.state.pending_aux_tts.values()
    )
    assert supervisor.tasks.active_count == 0
    actor_snapshot = supervisor.session_actor.snapshot()
    assert actor_snapshot.active_tasks == 0
    assert handle.cancelled is True
    assert handle.wait_drained(0.1).drained is True
    supervisor.shutdown()


def test_queued_start_failure_preserves_unvisited_sibling(monkeypatch):
    supervisor = AgentSupervisor()
    first = supervisor.tasks.create_task(_assistant_decision("first"))
    second = supervisor.tasks.create_task(_assistant_decision("second"))
    first_handle = first.turn_handle
    second_handle = second.turn_handle
    assert first_handle is not None
    assert second_handle is not None
    supervisor.state.queued_tasks.extend((first, second))

    class _FailingThread:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("thread construction failed")

    monkeypatch.setattr(tasks_module, "Thread", _FailingThread)

    with pytest.raises(RuntimeError, match="thread construction failed"):
        supervisor._start_queued_tasks()  # noqa: SLF001

    assert first.cancel_event.is_set()
    assert first_handle.wait_drained(0.1).drained is True
    assert supervisor.state.queued_tasks == [second]
    assert not second.cancel_event.is_set()
    assert supervisor.session_actor.identity_for_task(
        second.task_id
    ) == second.identity
    assert supervisor.session_actor.snapshot().remembered_tasks == 1

    supervisor.cancel_all()
    assert second_handle.wait_drained(0.1).drained is True
    assert supervisor.session_actor.snapshot().remembered_tasks == 0
    supervisor.shutdown()


def test_default_supervisor_completion_does_not_retain_output_binding():
    supervisor = AgentSupervisor()
    task = supervisor.tasks.create_task(_assistant_decision("text-only"))
    assert task.identity is not None
    assert supervisor.tasks.admit_task(task)
    supervisor.state.active_tasks[task.task_id] = task

    supervisor.publish(
        AgentEvent(
            EventKind.TASK_COMPLETED,
            {
                "task_id": task.task_id,
                "text": "text-only answer",
                "speak": True,
                "data": {},
                "epoch": supervisor.speech_epoch,
                **task.identity.to_payload(),
            },
        )
    )
    supervisor.drain()
    supervisor.tasks.retire_task_admission(task)

    assert supervisor.pending_output_count == 0
    assert supervisor.session_actor.snapshot().remembered_tasks == 0
    supervisor.shutdown()


def test_cancelled_published_completion_has_no_outward_effects():
    supervisor = AgentSupervisor()
    task = supervisor.tasks.create_task(_assistant_decision("terminal race"))
    assert task.identity is not None
    handle = task.turn_handle
    assert handle is not None
    supervisor.state.active_tasks[task.task_id] = task
    supervisor.publish(
        AgentEvent(
            EventKind.TASK_COMPLETED,
            {
                "task_id": task.task_id,
                "text": "must stay fenced",
                "speak": True,
                "followup": False,
                "data": {},
                "epoch": supervisor.speech_epoch,
                **task.identity.to_payload(),
            },
            priority=60,
        )
    )

    task.cancel()
    supervisor.drain()

    assert task.task_id not in supervisor.state.active_tasks
    assert "must stay fenced" not in supervisor.state.spoken_outputs
    assert all(
        item.text != "must stay fenced"
        for item in supervisor.memory.all()
    )
    assert not any(
        event.kind == EventKind.TTS_REQUEST
        and event.payload.get("text") == "must stay fenced"
        for event in supervisor.state.event_log
    )
    assert handle.wait_drained(0.1).drained is True
    supervisor.shutdown()


def test_cancel_after_completion_before_tts_admission_retires_exact_output():
    supervisor = AgentSupervisor(defer_output_until_tts_admission=True)
    task = supervisor.tasks.create_task(_assistant_decision("stage output"))
    assert task.identity is not None
    handle = task.turn_handle
    assert handle is not None
    supervisor.state.active_tasks[task.task_id] = task
    completion = AgentEvent(
        EventKind.TASK_COMPLETED,
        {
            "task_id": task.task_id,
            "text": "never admit this",
            "speak": True,
            "followup": False,
            "data": {},
            "epoch": supervisor.speech_epoch,
            **task.identity.to_payload(),
        },
        priority=60,
    )
    supervisor._handle_task_completed(completion)  # noqa: SLF001
    assert task.task_id in supervisor.state.pending_audio_tasks
    assert supervisor.pending_output_count == 1

    task.cancel()

    assert not supervisor.tts_request_allowed(
        task.task_id,
        supervisor.speech_epoch,
        completion.payload,
    )
    supervisor.note_tts_rejected(task.task_id, completion.payload)
    assert task.task_id not in supervisor.state.pending_audio_tasks
    assert supervisor.pending_output_count == 0
    assert handle.wait_drained(0.1).drained is True
    supervisor.shutdown()


@pytest.mark.parametrize(
    "kind",
    (EventKind.TASK_CANCELLED, EventKind.TASK_FAILED),
)
def test_stale_terminal_identity_cannot_pop_replacement_task(kind):
    supervisor = AgentSupervisor()
    task = supervisor.tasks.create_task(_assistant_decision("current scope"))
    assert task.identity is not None
    supervisor.state.active_tasks[task.task_id] = task
    forged = {
        **task.identity.to_payload(),
        "binding_id": task.identity.binding_id + 1,
    }
    supervisor.publish(
        AgentEvent(
            kind,
            {
                "task_id": task.task_id,
                "error": "forged failure",
                **forged,
            },
            priority=20,
        )
    )

    supervisor.drain()

    assert supervisor.state.active_tasks.get(task.task_id) is task
    assert "forged failure" not in supervisor.state.failures
    task.cancel()
    supervisor.state.active_tasks.pop(task.task_id, None)
    supervisor.shutdown()


@pytest.mark.parametrize(
    ("kind", "replacement_owner"),
    (
        (EventKind.TASK_CANCELLED, "active_tasks"),
        (EventKind.TASK_FAILED, "pending_audio_tasks"),
    ),
)
def test_tombstoned_old_terminal_is_cleanup_only_after_same_id_rebind(
    monkeypatch,
    kind,
    replacement_owner,
):
    supervisor = AgentSupervisor()
    old = supervisor.tasks.create_task(_assistant_decision("old scope"))
    assert old.identity is not None
    old_identity = old.identity
    old_handle = old.turn_handle
    assert old_handle is not None
    with supervisor._cancel_lock:  # noqa: SLF001
        supervisor._retire_task_tts_locked(  # noqa: SLF001
            old.task_id,
            old_identity,
        )
    old.cancel()
    assert old_handle.wait_drained(0.1).drained

    replacement = AgentTask(
        mode=Mode.ASSISTANT,
        input_text="replacement scope",
        task_id=old.task_id,
    )
    replacement.attach_turn_handle(
        supervisor.session_actor.bind_turn(
            replacement.task_id,
            supervisor.session_actor.open_turn(revision=1),
        )
    )
    assert replacement.identity is not None
    assert replacement.identity.binding_id != old_identity.binding_id
    getattr(supervisor.state, replacement_owner)[
        replacement.task_id
    ] = replacement

    queued = supervisor.tasks.create_task(_assistant_decision("queued sibling"))
    supervisor.state.queued_tasks.append(queued)
    start_attempts: list[str] = []
    monkeypatch.setattr(
        supervisor,
        "_start_task",
        lambda task, **_kwargs: start_attempts.append(task.task_id) or True,
    )
    prior_failures = tuple(supervisor.state.failures)
    prior_outputs = tuple(supervisor.state.spoken_outputs)
    event = AgentEvent(
        kind,
        {
            "task_id": old.task_id,
            "error": "stale old failure",
            "speak": True,
            "epoch": supervisor.speech_epoch,
            **old_identity.to_payload(),
        },
        priority=20,
    )

    supervisor.handle_event(event)

    assert (
        getattr(supervisor.state, replacement_owner).get(replacement.task_id)
        is replacement
    )
    assert supervisor.state.queued_tasks == [queued]
    assert start_attempts == []
    assert tuple(supervisor.state.failures) == prior_failures
    assert tuple(supervisor.state.spoken_outputs) == prior_outputs
    accepted, resolved, _success_current = (
        supervisor.consume_terminal_event_acceptance(event)
    )
    assert accepted is True
    assert resolved == old_identity

    supervisor.cancel_all()
    supervisor.shutdown()


def test_playback_pins_only_its_exact_same_turn_task_binding():
    actor = SessionActor(
        session_id="same-turn-playback",
        max_remembered_tasks=2,
    )
    turn = actor.open_turn(revision=3)
    first = actor.bind_task("first", turn)
    second = actor.bind_task("second", turn)
    assert first != second
    playback = actor.admit_playback(task_id="second", identity=second)
    assert playback is not None

    assert actor.release_task_binding("first", first)
    replacement = actor.bind_task("replacement", turn)
    assert replacement != first

    assert actor.release_task_binding("second", second)
    assert actor.release_task_binding("replacement", replacement)
    actor.finish_playback(playback.playback_admission_id)
    assert actor.snapshot().remembered_tasks == 0


def test_mutating_tool_provider_start_is_claimed_exactly_once():
    actor = SessionActor(session_id="tool-claim")
    identity = actor.bind_task("task", actor.open_turn(revision=1))
    tool = actor.admit_tool(
        task_id="task",
        identity=identity,
        tool_call_id="tool-once",
        idempotency_key="effect-once",
    )

    assert actor.claim_tool_start(tool.tool_call_id, tool.identity) is True
    assert actor.claim_tool_start(tool.tool_call_id, tool.identity) is False


def test_speech_cancel_and_mutating_provider_start_are_lock_ordered():
    actor = SessionActor(session_id="tool-start-race")
    before_identity = actor.bind_task(
        "before-cancel",
        actor.open_turn(revision=1),
    )
    before = actor.admit_tool(
        task_id="before-cancel",
        identity=before_identity,
        tool_call_id="before",
        idempotency_key="before-effect",
    )

    # Cancellation wins: an admitted-but-unstarted mutation cannot launch.
    actor.cancel_speech()
    assert before.cancel_event.is_set() is True
    assert actor.claim_tool_start(before.tool_call_id, before.identity) is False
    actor.finish_tool(before.tool_call_id, before.identity)

    after_identity = actor.bind_task(
        "after-claim",
        actor.open_turn(revision=2),
    )
    after = actor.admit_tool(
        task_id="after-claim",
        identity=after_identity,
        tool_call_id="after",
        idempotency_key="after-effect",
    )

    # Provider start wins: speech interruption suppresses output but does not
    # cancel the already-started mutation. Explicit tool cancellation still can.
    assert actor.claim_tool_start(after.tool_call_id, after.identity) is True
    actor.cancel_speech()
    assert after.cancel_event.is_set() is False
    assert actor.cancel_tool(after.tool_call_id, after.identity) is True
    assert after.cancel_event.is_set() is True


def test_runtime_mutation_start_claim_wins_handle_cancel_race():
    actor = SessionActor(session_id="runtime-tool-start-race")
    registry = CapabilityRegistry()
    claim_won = Event()
    release_claim = Event()
    provider_called = Event()
    original_claim = actor.claim_tool_start

    def gated_claim(
        tool_call_id: str,
        identity: TaskIdentity,
    ) -> bool:
        claimed = original_claim(tool_call_id, identity)
        assert claimed is True
        claim_won.set()
        assert release_claim.wait(timeout=2.0)
        return claimed

    actor.claim_tool_start = gated_claim  # type: ignore[method-assign]

    def mutate(_query, _context):
        provider_called.set()
        return CapabilityResult(True, "executed once")

    registry.register(
        "command.stage",
        mutate,
        spec=CapabilitySpec(
            "command.stage",
            "mutation",
            side_effecting=True,
        ),
    )
    runtime = TaskRuntime(
        lambda _event: None,
        registry,
        session_actor=actor,
    )
    task = runtime.create_task(
        IntentDecision(
            IntentKind.COMMAND,
            1.0,
            "perform the effect",
            "test",
            mode=Mode.COMMAND,
        )
    )
    handle = task.turn_handle
    assert handle is not None
    outcome: dict[str, object] = {}

    def invoke() -> None:
        try:
            outcome["result"] = runtime._invoke(  # noqa: SLF001
                task,
                "command.stage",
            )
        except BaseException as exc:
            outcome["error"] = exc

    thread = Thread(target=invoke, daemon=True)
    try:
        thread.start()
        assert claim_won.wait(timeout=1.0)

        # The tool start claim already linearized under the actor lock. Closing
        # speech now fences its output but must not silently cancel the effect.
        assert handle.cancel() is True
        release_claim.set()
        thread.join(timeout=2.0)

        assert not thread.is_alive()
        assert provider_called.is_set()
        result = outcome.get("result")
        assert isinstance(result, CapabilityResult) and result.ok is True
        assert handle.wait_drained(1.0).drained is True
    finally:
        release_claim.set()
        if thread.ident is not None:
            thread.join(timeout=2.0)


def test_mutation_waiting_for_provider_capacity_cannot_start_after_barge():
    actor = SessionActor(session_id="semaphore-barge")
    registry = CapabilityRegistry()
    blocker_started = Event()
    blocker_release = Event()
    mutation_started = Event()

    def blocking_read(_query, _context):
        blocker_started.set()
        assert blocker_release.wait(timeout=2.0)
        return CapabilityResult(True, "read complete")

    def mutate(_query, _context):
        mutation_started.set()
        return CapabilityResult(True, "must not execute")

    registry.register(
        "read.block",
        blocking_read,
        spec=CapabilitySpec("read.block", "blocking read"),
    )
    registry.register(
        "command.stage",
        mutate,
        spec=CapabilitySpec(
            "command.stage",
            "mutation",
            side_effecting=True,
        ),
    )
    runtime = TaskRuntime(
        lambda _event: None,
        registry,
        max_active_tasks=1,
        session_actor=actor,
    )
    read_task = runtime.create_task(_assistant_decision("hold the slot"))
    mutation_task = runtime.create_task(
        IntentDecision(
            IntentKind.COMMAND,
            1.0,
            "mutate once",
            "test",
            mode=Mode.COMMAND,
        )
    )
    mutation_done = Event()
    mutation_outcome: dict[str, object] = {}
    read_outcome: dict[str, object] = {}

    def invoke_read() -> None:
        try:
            read_outcome["result"] = runtime._invoke(  # noqa: SLF001
                read_task,
                "read.block",
            )
        except BaseException as exc:
            read_outcome["error"] = exc

    read_thread = Thread(target=invoke_read, daemon=True)

    def invoke_mutation() -> None:
        try:
            mutation_outcome["result"] = runtime._invoke(  # noqa: SLF001
                mutation_task,
                "command.stage",
            )
        except BaseException as exc:
            mutation_outcome["error"] = exc
        finally:
            mutation_done.set()

    mutation_thread = Thread(target=invoke_mutation, daemon=True)
    read_thread.start()
    assert blocker_started.wait(timeout=1.0)
    mutation_thread.start()
    assert _wait_until(lambda: actor.active_tool_count == 1)

    actor.cancel_speech()
    mutation_task.cancel()
    assert mutation_done.wait(timeout=1.0)
    assert mutation_started.is_set() is False
    assert type(mutation_outcome.get("error")).__name__ == "_TaskCancelled"
    assert actor.active_tool_count == 0

    blocker_release.set()
    read_thread.join(timeout=1.0)
    mutation_thread.join(timeout=1.0)
    assert not read_thread.is_alive() and not mutation_thread.is_alive()
    assert isinstance(read_outcome.get("result"), CapabilityResult)


def test_supervisor_rejects_partial_malformed_and_forged_playback_identity():
    supervisor = AgentSupervisor()
    task = supervisor.tasks.create_task(_assistant_decision())
    supervisor.state.active_tasks[task.task_id] = task
    assert task.identity is not None
    valid = task.identity.to_payload()

    malformed_payloads = (
        {"session_id": task.identity.session_id},
        {**valid, "revision": "not-an-integer"},
        {**valid, "revision": task.identity.revision + 1},
    )
    for payload in malformed_payloads:
        assert (
            supervisor.admit_playback(
                payload,
                task_id=task.task_id,
                auxiliary_tts=False,
                aux_tts_id="",
            )
            is None
        )

    legacy = supervisor.admit_playback(
        {},
        task_id=task.task_id,
        auxiliary_tts=False,
        aux_tts_id="",
    )
    assert legacy is not None
    supervisor.finish_playback(legacy.playback_admission_id)

    aux_id = supervisor.register_aux_tts(task.task_id)
    assert (
        supervisor.admit_playback(
            {"turn_id": task.identity.turn_id},
            task_id=task.task_id,
            auxiliary_tts=True,
            aux_tts_id=aux_id,
        )
        is None
    )
    auxiliary = supervisor.admit_playback(
        valid,
        task_id=task.task_id,
        auxiliary_tts=True,
        aux_tts_id=aux_id,
    )
    assert auxiliary is not None
    supervisor.finish_playback(auxiliary.playback_admission_id)
    supervisor.note_aux_tts_admitted(aux_id)
    supervisor.shutdown()


def test_legacy_synthetic_playback_binding_reclaims_after_sink_terminal():
    supervisor = AgentSupervisor()
    actor = SessionActor(
        session_id="legacy-playback",
        max_remembered_tasks=1,
    )
    supervisor.session_actor = actor
    supervisor.tasks = TaskRuntime(
        supervisor.publish,
        supervisor.capabilities,
        session_actor=actor,
    )

    admission = supervisor.admit_playback(
        {},
        task_id="",
        auxiliary_tts=False,
        aux_tts_id="",
    )
    assert admission is not None
    assert actor.snapshot().remembered_tasks == 1
    supervisor.finish_playback(admission.playback_admission_id)
    assert actor.snapshot().remembered_tasks == 0
    assert actor.bind_task("next", actor.open_turn(revision=1))
    supervisor.shutdown()


def test_auxiliary_tts_rebinds_when_source_binding_is_retired_but_pinned():
    supervisor = AgentSupervisor()
    actor = SessionActor(session_id="retired-aux")
    supervisor.session_actor = actor
    supervisor.tasks = TaskRuntime(
        supervisor.publish,
        supervisor.capabilities,
        session_actor=actor,
    )
    task = supervisor.tasks.create_task(_assistant_decision("source"))
    assert task.identity is not None
    source_playback = actor.admit_playback(
        task_id=task.task_id,
        identity=task.identity,
    )
    assert source_playback is not None
    assert actor.release_task_binding(task.task_id, task.identity)
    assert actor.identity_for_task(task.task_id) is None

    aux_id = supervisor.register_aux_tts(task.task_id)
    payload = supervisor.auxiliary_tts_identity_payload(aux_id)
    assert payload
    auxiliary = supervisor.admit_playback(
        payload,
        task_id=task.task_id,
        auxiliary_tts=True,
        aux_tts_id=aux_id,
    )
    assert auxiliary is not None
    assert auxiliary.identity != task.identity

    supervisor.note_aux_tts_admitted(aux_id)
    supervisor.finish_playback(auxiliary.playback_admission_id)
    supervisor.finish_playback(source_playback.playback_admission_id)
    assert actor.snapshot().remembered_tasks == 0
    supervisor.shutdown()


def test_task_binding_capacity_requires_explicit_external_release():
    actor = SessionActor(
        session_id="binding-capacity",
        max_remembered_tasks=2,
    )
    queued = actor.bind_task("queued", actor.open_turn(revision=1))
    confirmation = actor.bind_task("confirmation", actor.open_turn(revision=2))

    # Neither binding is actor-active, but both remain externally owned.
    with pytest.raises(AdmissionRejected, match="identity table is at capacity"):
        actor.bind_task("pending-output", actor.open_turn(revision=3))
    assert actor.identity_for_task("queued") == queued
    assert actor.identity_for_task("confirmation") == confirmation

    assert actor.release_task_binding("queued", queued) is True
    pending_output = actor.bind_task(
        "pending-output",
        actor.open_turn(revision=3),
    )
    with pytest.raises(AdmissionRejected, match="identity table is at capacity"):
        actor.bind_task("new-turn", actor.open_turn(revision=4))
    assert actor.identity_for_task("pending-output") == pending_output

    assert actor.release_task_binding("confirmation", confirmation) is True
    assert actor.release_task_binding("pending-output", pending_output) is True
    assert actor.bind_task("new-turn", actor.open_turn(revision=4))


@pytest.mark.parametrize(
    "owner,release",
    (
        (
            "queued",
            lambda supervisor, task: supervisor.cancel_all(),
        ),
        (
            "confirmation",
            lambda supervisor, task: supervisor._deny_all(),  # noqa: SLF001
        ),
        (
            "pending-output",
            lambda supervisor, task: supervisor.note_tts_rejected(
                task.task_id,
                task.identity.to_payload(),
            ),
        ),
    ),
)
def test_supervisor_releases_binding_only_at_external_terminal_boundary(
    owner,
    release,
):
    supervisor = AgentSupervisor()
    actor = SessionActor(
        session_id=f"supervisor-{owner}",
        max_remembered_tasks=1,
    )
    supervisor.session_actor = actor
    supervisor.tasks = TaskRuntime(
        supervisor.publish,
        supervisor.capabilities,
        session_actor=actor,
    )
    task = supervisor.tasks.create_task(_assistant_decision(owner))
    if owner == "queued":
        supervisor.state.queued_tasks.append(task)
    elif owner == "confirmation":
        supervisor.state.pending_confirmations[task.task_id] = task
    else:
        supervisor.state.pending_audio_tasks[task.task_id] = task
        supervisor._pending_output_bindings[task.task_id] = (  # noqa: SLF001
            task.identity
        )

    with pytest.raises(AdmissionRejected, match="identity table is at capacity"):
        supervisor.tasks.create_task(_assistant_decision("must wait"))
    assert actor.identity_for_task(task.task_id) == task.identity

    release(supervisor, task)
    replacement = supervisor.tasks.create_task(_assistant_decision("released"))
    assert replacement.identity is not None
    supervisor.shutdown()


def test_interrupted_speech_does_not_cancel_or_reexecute_mutating_tool():
    actor = SessionActor(session_id="tool-session")
    turn = actor.open_turn(revision=4)
    registry = CapabilityRegistry()
    provider_started = Event()
    provider_release = Event()
    provider_finished = Event()
    seen_context: dict[str, object] = {}
    executions = 0

    def mutate(_query, context):
        nonlocal executions
        executions += 1
        seen_context.update(context)
        provider_started.set()
        assert provider_release.wait(timeout=2.0)
        provider_finished.set()
        return CapabilityResult(True, "mutation completed")

    registry.register(
        "command.stage",
        mutate,
        spec=CapabilitySpec(
            "command.stage",
            "mutate",
            side_effecting=True,
        ),
    )
    published = []
    runtime = TaskRuntime(
        published.append,
        registry,
        session_actor=actor,
    )
    decision = IntentDecision(
        IntentKind.COMMAND,
        1.0,
        "perform once",
        "test",
        mode=Mode.COMMAND,
    )
    task = runtime.create_task(decision, turn_identity=turn)
    handle = task.turn_handle
    assert handle is not None
    assert runtime.start(task) is True
    assert provider_started.wait(timeout=2.0)

    # Barge-in invalidates output/task work. The provider-started mutation owns
    # a different cancellation Event and therefore completes exactly once.
    actor.cancel_speech()
    task.cancel()
    assert seen_context["cancel_event"].is_set() is False
    pending = handle.wait_drained(0.0)
    assert pending.effects_fenced is True
    assert pending.drained is False
    assert pending.active_providers == 1
    assert pending.active_tools == 1
    provider_release.set()
    assert provider_finished.wait(timeout=2.0)
    assert _wait_until(
        lambda: any(event.kind == EventKind.TASK_CANCELLED for event in published)
    )
    assert executions == 1
    assert seen_context["tool_call_id"]
    assert seen_context["idempotency_key"]
    assert handle.wait_drained(1.0).drained is True

    # Reconstructing the same turn/step after interruption derives the same
    # generic idempotency key. The actor rejects it before provider execution.
    retry = runtime.create_task(decision, turn_identity=turn)
    retry_result = runtime._invoke(retry, "command.stage")  # noqa: SLF001
    assert retry_result.ok is False
    assert "idempotency key already admitted" in retry_result.error
    assert executions == 1
