"""Headless ownership tests for the bounded post-recognition session actor."""

from __future__ import annotations

from threading import Event, Thread
import time

import pytest

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
from always_on_agent.tasks import TaskRuntime
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
    actor.finish_task("first")
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
    assert actor.claim_tool_start("tool-1") is False
    assert actor.cancel_tool("tool-1") is True
    assert tool.cancel_event.is_set() is True
    assert actor.admit_playback(task_id="second", identity=second) is None


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


def test_playback_pins_only_its_exact_same_turn_task_binding():
    actor = SessionActor(
        session_id="same-turn-playback",
        max_remembered_tasks=2,
    )
    turn = actor.open_turn(revision=3)
    first = actor.bind_task("first", turn)
    second = actor.bind_task("second", turn)
    assert first == second
    playback = actor.admit_playback(task_id="second", identity=second)
    assert playback is not None

    assert actor.release_task_binding("first")
    replacement = actor.bind_task("replacement", turn)
    assert replacement == first

    assert actor.release_task_binding("second")
    assert actor.release_task_binding("replacement")
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

    assert actor.claim_tool_start(tool.tool_call_id) is True
    assert actor.claim_tool_start(tool.tool_call_id) is False


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
    assert actor.claim_tool_start(before.tool_call_id) is False
    actor.finish_tool(before.tool_call_id)

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
    assert actor.claim_tool_start(after.tool_call_id) is True
    actor.cancel_speech()
    assert after.cancel_event.is_set() is False
    assert actor.cancel_tool(after.tool_call_id) is True
    assert after.cancel_event.is_set() is True


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

    read_thread = Thread(
        target=lambda: runtime._invoke(read_task, "read.block"),  # noqa: SLF001
        daemon=True,
    )

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
    assert actor.release_task_binding(task.task_id)
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

    assert actor.release_task_binding("queued") is True
    pending_output = actor.bind_task(
        "pending-output",
        actor.open_turn(revision=3),
    )
    with pytest.raises(AdmissionRejected, match="identity table is at capacity"):
        actor.bind_task("new-turn", actor.open_turn(revision=4))
    assert actor.identity_for_task("pending-output") == pending_output

    assert actor.release_task_binding("confirmation") is True
    assert actor.release_task_binding("pending-output") is True
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
            lambda supervisor, task: supervisor.note_tts_rejected(task.task_id),
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
        supervisor._pending_output_bindings.add(task.task_id)  # noqa: SLF001

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
    assert runtime.start(task) is True
    assert provider_started.wait(timeout=2.0)

    # Barge-in invalidates output/task work. The provider-started mutation owns
    # a different cancellation Event and therefore completes exactly once.
    actor.cancel_speech()
    task.cancel()
    assert seen_context["cancel_event"].is_set() is False
    provider_release.set()
    assert provider_finished.wait(timeout=2.0)
    assert _wait_until(
        lambda: any(event.kind == EventKind.TASK_CANCELLED for event in published)
    )
    assert executions == 1
    assert seen_context["tool_call_id"]
    assert seen_context["idempotency_key"]

    # Reconstructing the same turn/step after interruption derives the same
    # generic idempotency key. The actor rejects it before provider execution.
    retry = runtime.create_task(decision, turn_identity=turn)
    retry_result = runtime._invoke(retry, "command.stage")  # noqa: SLF001
    assert retry_result.ok is False
    assert "idempotency key already admitted" in retry_result.error
    assert executions == 1
