"""Deterministic bounded-mailbox, ordering, lifecycle, and shutdown contracts."""

from __future__ import annotations

import logging
from threading import Event, Thread
import time

import pytest

from always_on_agent.acoustic import (
    AcousticLineage,
    AcousticSource,
    AcousticSpan,
)
from always_on_agent.event_bus import (
    EventBus,
    MailboxLane,
    classified_event_kinds,
    event_mailbox_policy,
)
from always_on_agent.events import AgentEvent, EventKind
from always_on_agent.models import IntentDecision, IntentKind
from always_on_agent.supervisor import AgentSupervisor
from core.engines.scripted import ScriptedEngine
from core.playback_history import PlaybackCommit
from core.runtime import VoiceRuntime


def _event(text: str = "hi", kind: EventKind = EventKind.STT_FINAL) -> AgentEvent:
    return AgentEvent(kind, {"text": text})


def _lineage(utterance_id: str) -> AcousticLineage:
    return AcousticLineage.single(
        AcousticSpan(
            stream_id="event-bus-test",
            utterance_id=utterance_id,
            source=AcousticSource.SCRIPTED,
        )
    )


def _partial(
    utterance_id: str,
    revision: int,
    text: str | None = None,
) -> AgentEvent:
    return AgentEvent.partial(
        text or f"{utterance_id}-{revision}",
        acoustic=_lineage(utterance_id),
        revision=revision,
    )


def _wait_for_idle(bus: EventBus, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if bus.idle():
            return True
        Event().wait(0.005)
    return bus.idle()


def test_every_event_kind_has_an_explicit_mailbox_policy():
    critical = {
        EventKind.STT_FINAL,
        EventKind.STT_ABORTED,
        EventKind.CONTROL_STOP,
        EventKind.CONTROL_MODE,
        EventKind.CONTROL_CONFIRM,
        EventKind.CONTROL_DENY,
        EventKind.TASK_STARTED,
        EventKind.TASK_COMPLETED,
        EventKind.TASK_CANCELLED,
        EventKind.TASK_FAILED,
        EventKind.TTS_REQUEST,
        EventKind.TTS_STREAM_END,
        EventKind.MEMORY_COMMIT,
        EventKind.MAILBOX_FAULT,
    }
    data = {
        EventKind.STT_PARTIAL,
        EventKind.SPEECH_OBSERVATION,
        EventKind.INTENT_DECISION,
        EventKind.TASK_PROGRESS,
        EventKind.FOLLOWUP_TICK,
        EventKind.CAPTURE_STATE,
    }
    assert classified_event_kinds() == frozenset(EventKind)
    assert critical | data == set(EventKind)
    assert critical.isdisjoint(data)
    for kind in critical:
        assert event_mailbox_policy(AgentEvent(kind)).lane == MailboxLane.CONTROL
    for kind in data:
        assert event_mailbox_policy(AgentEvent(kind)).lane == MailboxLane.DATA

    assert (
        event_mailbox_policy(
            AgentEvent(EventKind.CAPTURE_STATE, {"state": "fatal"})
        ).lane
        == MailboxLane.CONTROL
    )
    assert (
        event_mailbox_policy(
            AgentEvent(EventKind.CAPTURE_STATE, {"state": "recovering"})
        ).lane
        == MailboxLane.DATA
    )
    stop = event_mailbox_policy(_partial("stop-turn", 1, "stop"))
    mode = event_mailbox_policy(_partial("mode-turn", 1, "assistant mode"))
    yes = event_mailbox_policy(_partial("yes-turn", 1, "yes"))
    assert stop.lane == mode.lane == MailboxLane.CONTROL
    assert stop.coalesce_key is not None and mode.coalesce_key is not None
    assert stop.preemptive and mode.preemptive
    assert yes.lane == MailboxLane.DATA and yes.coalesce_key is not None
    legacy_stop = event_mailbox_policy(AgentEvent.partial("stop"))
    assert legacy_stop.lane == MailboxLane.CONTROL
    assert legacy_stop.coalesce_key is None
    assert legacy_stop.reject_unscoped_control

    for kind in critical - {
        EventKind.TTS_REQUEST,
        EventKind.TTS_STREAM_END,
    }:
        assert event_mailbox_policy(AgentEvent(kind)).preemptive
    for kind in {EventKind.TTS_REQUEST, EventKind.TTS_STREAM_END}:
        output = event_mailbox_policy(AgentEvent(kind))
        assert output.ordered_output and not output.preemptive


@pytest.mark.parametrize(
    ("max_events", "control_reserve"),
    [(0, 1), (1, 1), (4, 0), (4, 4), (4, 5), (True, 1), (4, True)],
)
def test_mailbox_capacity_requires_a_nonempty_control_reserve(
    max_events,
    control_reserve,
):
    with pytest.raises(ValueError):
        EventBus(
            max_events=max_events,
            control_reserve=control_reserve,
        )


@pytest.mark.parametrize("max_output_bypass", [0, -1, True, 1.5])
def test_output_bypass_bound_requires_a_positive_integer(max_output_bypass):
    with pytest.raises(ValueError, match="max_output_bypass"):
        EventBus(max_output_bypass=max_output_bypass)


def test_stop_discards_pending_events_by_default():
    # A TTS_REQUEST racing shutdown must remain undispatched; its handler could
    # otherwise start speaking after teardown began.
    bus = EventBus()
    spoken: list[AgentEvent] = []
    bus.subscribe(spoken.append)
    bus.publish(_event("late reply", EventKind.TTS_REQUEST))
    bus.stop()
    assert spoken == []
    assert not bus.idle()
    assert bus.stats().pending == 1


def test_stop_drain_true_dispatches_leftovers():
    bus = EventBus()
    seen: list[str] = []
    bus.subscribe(lambda event: seen.append(event.payload["text"]))
    for index in range(3):
        bus.publish(_event(f"e{index}"))
    bus.stop(drain=True)
    assert seen == ["e0", "e1", "e2"]
    assert bus.idle()


def test_stop_drain_survives_raising_handler():
    bus = EventBus()
    seen: list[str] = []

    def _raise(_event):
        raise RuntimeError("boom")

    bus.subscribe(_raise)
    bus.subscribe(lambda event: seen.append(event.payload["text"]))
    bus.publish(_event("a"))
    bus.publish(_event("b"))
    bus.stop(drain=True)
    assert seen == ["a", "b"]
    assert bus.stats().handled == 2


def test_partial_storm_is_physically_bounded_and_delivers_latest():
    bus = EventBus(max_events=8, control_reserve=2)
    seen: list[AgentEvent] = []
    bus.subscribe(seen.append)

    for revision in range(10_000):
        bus.publish(_partial("turn-a", revision))

    snapshot = bus.stats()
    assert snapshot.pending == 1
    assert snapshot.pending_data == 1
    assert snapshot.outstanding == 1
    assert snapshot.peak_pending == 1
    assert snapshot.offered == 10_000
    assert snapshot.admitted == 10_000
    assert snapshot.coalesced == 9_999

    assert bus.drain() == 1
    assert seen[0].payload["revision"] == 9_999
    assert bus.idle()


def test_partial_coalescing_is_scoped_to_complete_acoustic_turn():
    bus = EventBus(max_events=8, control_reserve=2)
    seen: list[str] = []
    bus.subscribe(lambda event: seen.append(event.payload["text"]))

    bus.publish(_partial("turn-a", 1, "a1"))
    bus.publish(_partial("turn-b", 1, "b1"))
    bus.publish(_partial("turn-a", 2, "a2"))
    bus.drain()

    # The new A revision keeps its real publication order behind B.
    assert seen == ["b1", "a2"]


def test_older_partial_revision_cannot_replace_newer_pending_revision():
    bus = EventBus(max_events=8, control_reserve=2)
    seen: list[AgentEvent] = []
    bus.subscribe(seen.append)

    bus.publish(_partial("turn-a", 4))
    bus.publish(_partial("turn-a", 3))
    assert bus.stats().rejected_stale == 1
    assert bus.stats().outstanding == 1

    bus.drain()
    assert [event.payload["revision"] for event in seen] == [4]
    assert bus.idle()


@pytest.mark.parametrize("terminal_kind", ["final", "abort"])
def test_terminal_retires_only_its_matching_pending_partial(terminal_kind):
    bus = EventBus(max_events=8, control_reserve=2)
    seen: list[AgentEvent] = []
    bus.subscribe(seen.append)
    lineage_a = _lineage("turn-a")

    bus.publish(_partial("turn-a", 1, "a partial"))
    bus.publish(_partial("turn-b", 1, "b partial"))
    terminal = (
        AgentEvent.final("a final", acoustic=lineage_a, revision=2)
        if terminal_kind == "final"
        else AgentEvent.transcript_aborted(
            reason="test",
            acoustic=lineage_a,
            revision=2,
        )
    )
    bus.publish(terminal)

    snapshot = bus.stats()
    assert snapshot.retired_partials == 1
    assert snapshot.pending == 2
    bus.drain()
    assert seen[0] is terminal
    assert [event.payload.get("text") for event in seen[1:]] == ["b partial"]
    assert bus.idle()


@pytest.mark.parametrize("terminal_kind", ["final", "abort"])
@pytest.mark.parametrize("terminal_revision", [3, 4])
def test_non_newer_terminal_cannot_replace_queued_partial(
    terminal_kind,
    terminal_revision,
):
    bus = EventBus(max_events=8, control_reserve=2)
    seen: list[AgentEvent] = []
    bus.subscribe(seen.append)
    lineage = _lineage("turn-a")
    partial = AgentEvent.partial("newer", acoustic=lineage, revision=4)
    terminal = (
        AgentEvent.final(
            "stale",
            acoustic=lineage,
            revision=terminal_revision,
        )
        if terminal_kind == "final"
        else AgentEvent.transcript_aborted(
            reason="stale",
            acoustic=lineage,
            revision=terminal_revision,
        )
    )

    assert bus.publish(partial)
    assert not bus.publish(terminal)

    snapshot = bus.stats()
    assert snapshot.rejected_stale == 1
    assert snapshot.retired_partials == 0
    assert snapshot.pending == snapshot.outstanding == 1
    bus.drain()
    assert seen == [partial]
    assert bus.idle()


def test_composite_terminal_retires_each_covered_partial_lineage():
    bus = EventBus(max_events=8, control_reserve=2)
    seen: list[AgentEvent] = []
    bus.subscribe(seen.append)
    lineage_a = _lineage("turn-a")
    lineage_b = _lineage("turn-b")
    lineage_c = _lineage("turn-c")
    combined = AcousticLineage.combine((lineage_a, lineage_b))
    final = AgentEvent.final(
        "tell me about stocks",
        acoustic=combined,
        revision=3,
    )
    unrelated = AgentEvent.partial(
        "unrelated",
        acoustic=lineage_c,
        revision=1,
    )

    bus.publish(
        AgentEvent.partial("stop", acoustic=lineage_b, revision=1)
    )
    bus.publish(
        AgentEvent.partial("tell me about", acoustic=lineage_a, revision=1)
    )
    bus.publish(unrelated)
    assert bus.publish(final)

    snapshot = bus.stats()
    assert snapshot.retired_partials == 2
    assert snapshot.pending == snapshot.outstanding == 2
    bus.drain()
    assert seen == [final, unrelated]
    assert bus.idle()


def test_stale_composite_terminal_retires_no_constituent_partials():
    bus = EventBus(max_events=8, control_reserve=2)
    seen: list[AgentEvent] = []
    bus.subscribe(seen.append)
    lineage_a = _lineage("turn-a")
    lineage_b = _lineage("turn-b")
    partial_a = AgentEvent.partial("first", acoustic=lineage_a, revision=1)
    partial_b = AgentEvent.partial("second", acoustic=lineage_b, revision=4)
    terminal = AgentEvent.final(
        "stale merge",
        acoustic=AcousticLineage.combine((lineage_a, lineage_b)),
        revision=4,
    )

    bus.publish(partial_a)
    bus.publish(partial_b)
    assert not bus.publish(terminal)

    snapshot = bus.stats()
    assert snapshot.rejected_stale == 1
    assert snapshot.retired_partials == 0
    assert snapshot.pending == snapshot.outstanding == 2
    bus.drain()
    assert seen == [partial_a, partial_b]


def test_unique_data_saturation_evicts_oldest_least_urgent_event():
    bus = EventBus(max_events=5, control_reserve=2)
    seen: list[str] = []
    bus.subscribe(lambda event: seen.append(event.payload["text"]))

    for name in ("a", "b", "c", "d"):
        bus.publish(_partial(f"turn-{name}", 1, name))

    snapshot = bus.stats()
    assert snapshot.pending == 3
    assert snapshot.pending_data == 3
    assert snapshot.evicted_data == 1
    bus.drain()
    assert seen == ["b", "c", "d"]
    assert bus.idle()


def test_data_saturation_rejects_less_urgent_arrival():
    bus = EventBus(max_events=5, control_reserve=2)
    for index in range(3):
        bus.publish(
            AgentEvent(
                EventKind.INTENT_DECISION,
                {"task_id": f"task-{index}"},
                priority=40,
            )
        )

    bus.publish(_partial("late-partial", 1))
    snapshot = bus.stats()
    assert snapshot.pending_data == 3
    assert snapshot.rejected_data == 1
    assert snapshot.outstanding == 3
    bus.drain()
    assert bus.idle()


def test_data_saturation_cannot_consume_reserved_control_capacity():
    bus = EventBus(max_events=8, control_reserve=4)
    seen: list[AgentEvent] = []
    bus.subscribe(seen.append)
    for index in range(4):
        bus.publish(_partial(f"data-{index}", 1))

    stop = AgentEvent.stop()
    final = AgentEvent.final("final")
    cancelled = AgentEvent(
        EventKind.TASK_CANCELLED,
        {"task_id": "task"},
        priority=20,
    )
    tts = AgentEvent(
        EventKind.TTS_REQUEST,
        {"task_id": "task", "text": "answer"},
    )
    for event in (stop, final, cancelled, tts):
        bus.publish(event)

    snapshot = bus.stats()
    assert snapshot.pending == snapshot.max_events == 8
    assert snapshot.pending_data == 4
    assert snapshot.pending_control == 4
    bus.drain()

    assert seen[0] is stop
    assert seen[1] is cancelled
    assert seen[2] is final
    assert seen[-1] is tts
    assert all(seen.count(event) == 1 for event in (stop, final, cancelled, tts))


def test_critical_event_evicts_data_after_using_the_reserve():
    bus = EventBus(max_events=4, control_reserve=1)
    seen: list[AgentEvent] = []
    bus.subscribe(seen.append)
    for index in range(3):
        bus.publish(_partial(f"data-{index}", 1))
    stop = AgentEvent.stop()
    stream_end = AgentEvent(EventKind.TTS_STREAM_END, priority=110)

    bus.publish(stop)
    bus.publish(stream_end)

    snapshot = bus.stats()
    assert snapshot.pending == snapshot.max_events == 4
    assert snapshot.pending_data == 2
    assert snapshot.pending_control == 2
    assert snapshot.evicted_data == 1
    assert snapshot.mailbox_faults == 0
    bus.drain()
    assert seen[0] is stop
    assert seen[-1] is stream_end
    assert stop in seen and stream_end in seen


def test_all_critical_saturation_becomes_one_durable_fault(caplog):
    bus = EventBus(max_events=3, control_reserve=1)
    seen: list[AgentEvent] = []
    bus.subscribe(seen.append)
    for index in range(3):
        assert bus.publish(AgentEvent.final(f"final-{index}"))

    with caplog.at_level(logging.CRITICAL, logger="speaker.event_bus"):
        assert not bus.publish(AgentEvent.stop())

    snapshot = bus.stats()
    assert snapshot.pending == 1
    assert snapshot.mailbox_faults == 1
    assert snapshot.fault_discarded == 3
    assert snapshot.outstanding == 1
    assert snapshot.faulted and bus.faulted
    assert len(
        [record for record in caplog.records if "mailbox fault" in record.message]
    ) == 1
    assert not bus.publish(AgentEvent.final("after fault"))
    assert bus.stats().rejected_after_fault == 1
    assert bus.drain() == 1
    assert [event.kind for event in seen] == [EventKind.MAILBOX_FAULT]
    assert bus.idle()


def test_reentrant_all_critical_fault_does_not_abort_handler_cleanup():
    bus = EventBus(max_events=3, control_reserve=1)
    seen: list[EventKind] = []
    handler_completed = Event()

    def handler(event: AgentEvent) -> None:
        seen.append(event.kind)
        if event.kind == EventKind.TASK_STARTED:
            assert bus.publish(AgentEvent.final("fills released slot"))
            assert not bus.publish(AgentEvent.stop())
            handler_completed.set()

    bus.subscribe(handler)
    bus.publish(AgentEvent(EventKind.TASK_STARTED, priority=40))
    bus.publish(AgentEvent.final("queued one"))
    bus.publish(AgentEvent.final("queued two"))

    assert bus.drain() == 2
    assert handler_completed.is_set()
    assert seen == [EventKind.TASK_STARTED, EventKind.MAILBOX_FAULT]
    assert bus.stats().fault_discarded == 3
    assert bus.idle()


def test_mailbox_fault_directly_retires_supervisor_task_ownership():
    bus = EventBus(max_events=3, control_reserve=1)
    supervisor = AgentSupervisor(bus=bus)
    task = supervisor.tasks.create_task(
        IntentDecision(
            IntentKind.ASSISTANT,
            1.0,
            "test turn",
            "mailbox fault test",
        )
    )
    with supervisor._cancel_lock:
        supervisor.state.active_tasks[task.task_id] = task
    for index in range(3):
        supervisor.publish(AgentEvent.final(f"queued-{index}"))

    accepted = supervisor.publish(
        AgentEvent(
            EventKind.TASK_COMPLETED,
            {"task_id": task.task_id},
            priority=60,
        )
    )

    assert not accepted and bus.faulted
    assert supervisor.drain() == 1
    assert supervisor.state.active_tasks == {}
    assert task.cancel_event.is_set()
    assert any("mailbox fault" in failure for failure in supervisor.state.failures)
    assert bus.idle()


def test_runtime_mailbox_fault_stops_and_rejects_playback_commit_cleanly():
    runtime = VoiceRuntime(ScriptedEngine())
    runtime.start(run_bus=False)
    try:
        commit = PlaybackCommit(
            role="assistant",
            text="not retained after terminal mailbox fault",
            is_followup=False,
            schedule_followup=False,
            epoch=0,
            input_generation=0,
            followup_generation=0,
        )
        with runtime._playback_effect_changed:
            runtime._publish_playback_commits((commit, commit))
            assert runtime._pending_playback_memory_commits == 2

        for index in range(EventBus._MAX_EVENTS - 2):
            assert runtime.bus.publish(AgentEvent.final(f"queued-{index}"))
        assert not runtime.bus.publish(AgentEvent.stop())
        assert runtime.bus.faulted

        with runtime._playback_effect_changed:
            runtime._publish_playback_commits((commit,))
            assert runtime._pending_playback_memory_commits == 2

        assert runtime.bus.drain() == 1
        deadline = time.monotonic() + 2.0
        while not runtime._stop_complete and time.monotonic() < deadline:
            Event().wait(0.005)
        assert runtime._stop_complete
        assert runtime._stopping
        assert runtime._pending_playback_memory_commits == 0
        assert any(
            "mailbox fault" in failure
            for failure in runtime.supervisor.state.failures
        )
    finally:
        runtime.stop()


def test_preemptive_control_then_priority_fifo_order_crosses_lanes():
    bus = EventBus(max_events=8, control_reserve=2)
    seen: list[AgentEvent] = []
    bus.subscribe(seen.append)
    stream_end = AgentEvent(EventKind.TTS_STREAM_END, priority=110)
    progress = AgentEvent(EventKind.TASK_PROGRESS, priority=70)
    started = AgentEvent(EventKind.TASK_STARTED, priority=40)
    final = AgentEvent.final("final")
    equal_data = AgentEvent(EventKind.TASK_PROGRESS, priority=75)
    equal_control = AgentEvent(EventKind.TASK_FAILED, priority=75)

    for event in (
        stream_end,
        progress,
        started,
        final,
        equal_data,
        equal_control,
    ):
        bus.publish(event)
    bus.drain()

    assert seen == [
        started,
        final,
        equal_control,
        progress,
        equal_data,
        stream_end,
    ]


def test_ordered_output_is_serviced_after_bounded_data_bypass():
    bus = EventBus(
        max_events=16,
        control_reserve=4,
        max_output_bypass=4,
    )
    seen: list[AgentEvent] = []
    bus.subscribe(seen.append)
    stream_end = AgentEvent(EventKind.TTS_STREAM_END, priority=110)
    bus.publish(stream_end)

    for index in range(12):
        bus.publish(_partial(f"turn-{index}", 1))
        bus.drain_once()
        if stream_end in seen:
            break

    assert seen.index(stream_end) == 4
    assert bus.stats().forced_output_dispatches == 1
    bus.drain()
    assert bus.idle()


def test_output_promotion_preserves_fragments_before_stream_terminal():
    bus = EventBus(
        max_events=16,
        control_reserve=4,
        max_output_bypass=2,
    )
    seen: list[AgentEvent] = []
    bus.subscribe(seen.append)
    request_a = AgentEvent(EventKind.TTS_REQUEST, {"text": "a"}, priority=100)
    request_b = AgentEvent(EventKind.TTS_REQUEST, {"text": "b"}, priority=100)
    stream_end = AgentEvent(EventKind.TTS_STREAM_END, priority=110)
    for event in (request_a, request_b, stream_end):
        bus.publish(event)

    for index in range(5):
        bus.publish(_partial(f"storm-{index}", 1))
        bus.drain_once()

    assert seen[-3:] == [request_a, request_b, stream_end]
    assert bus.stats().forced_output_dispatches == 3
    bus.drain()
    assert bus.idle()


def test_preemptive_lifecycle_work_overtakes_aged_ordered_output():
    bus = EventBus(
        max_events=16,
        control_reserve=4,
        max_output_bypass=2,
    )
    seen: list[AgentEvent] = []
    bus.subscribe(seen.append)
    stream_end = AgentEvent(EventKind.TTS_STREAM_END, priority=110)
    bus.publish(stream_end)
    for index in range(2):
        bus.publish(_partial(f"warmup-{index}", 1))
        bus.drain_once()

    failed = AgentEvent(EventKind.TASK_FAILED, priority=25)
    memory = AgentEvent(EventKind.MEMORY_COMMIT, priority=30)
    final = AgentEvent.final("new user turn")
    completed = AgentEvent(EventKind.TASK_COMPLETED, priority=60)
    data = AgentEvent(EventKind.TASK_PROGRESS, priority=20)
    for event in (completed, data, final, memory, failed):
        bus.publish(event)

    for _ in range(6):
        bus.drain_once()

    assert seen[-6:] == [
        failed,
        memory,
        final,
        completed,
        stream_end,
        data,
    ]
    assert bus.stats().forced_output_dispatches == 1


@pytest.mark.parametrize("text", ["stop", "assistant mode"])
def test_unscoped_exact_partial_is_rejected_without_using_capacity(text):
    bus = EventBus(max_events=3, control_reserve=1)

    for _ in range(4):
        assert not bus.publish(AgentEvent.partial(text))

    snapshot = bus.stats()
    assert snapshot.rejected_unscoped_control == 4
    assert snapshot.pending == snapshot.outstanding == 0
    assert not snapshot.faulted


def test_identityless_noncontrol_partial_final_and_direct_stop_remain_supported():
    bus = EventBus(max_events=4, control_reserve=2)
    seen: list[AgentEvent] = []
    bus.subscribe(seen.append)
    partial = AgentEvent.partial("hello")
    final = AgentEvent.final("hello there")
    stop = AgentEvent.stop()

    assert bus.publish(partial)
    assert bus.publish(final)
    assert bus.publish(stop)
    bus.drain()

    assert seen == [stop, final, partial]


def test_identityless_correction_cannot_leave_stale_mode_partial():
    bus = EventBus(max_events=4, control_reserve=2)
    seen: list[AgentEvent] = []
    bus.subscribe(seen.append)
    final = AgentEvent.final("assistant modem")

    assert not bus.publish(AgentEvent.partial("assistant mode"))
    assert bus.publish(final)
    bus.drain()

    assert seen == [final]


@pytest.mark.parametrize(
    "payload",
    [
        {"text": "stop", "revision": 1, "acoustic": {"bad": "shape"}},
        {
            "text": "stop",
            "revision": True,
            "acoustic": _lineage("malformed-bool").to_payload(),
        },
        {
            "text": "stop",
            "revision": -1,
            "acoustic": _lineage("malformed-negative").to_payload(),
        },
    ],
)
def test_malformed_typed_exact_partial_fails_closed(payload):
    bus = EventBus(max_events=4, control_reserve=2)
    event = AgentEvent(EventKind.STT_PARTIAL, payload, priority=90)

    assert not bus.publish(event)
    assert bus.stats().rejected_unscoped_control == 1
    assert bus.idle()


def test_newer_typed_correction_supersedes_queued_exact_control_partial():
    bus = EventBus(max_events=4, control_reserve=2)
    seen: list[str] = []
    bus.subscribe(lambda event: seen.append(str(event.payload.get("text", ""))))
    lineage = _lineage("control-turn")

    bus.publish(AgentEvent.partial("stop", acoustic=lineage, revision=1))
    bus.publish(AgentEvent.partial("stock", acoustic=lineage, revision=2))

    snapshot = bus.stats()
    assert snapshot.pending_data == 1
    assert snapshot.pending_control == 0
    assert snapshot.coalesced == 1
    bus.drain()
    assert seen == ["stock"]


def test_control_to_data_correction_respects_data_capacity():
    bus = EventBus(max_events=3, control_reserve=1)
    seen: list[str] = []
    bus.subscribe(lambda event: seen.append(str(event.payload.get("text", ""))))
    lineage = _lineage("capacity-correction")
    bus.publish(_partial("turn-a", 1, "a"))
    bus.publish(_partial("turn-b", 1, "b"))
    bus.publish(AgentEvent.partial("stop", acoustic=lineage, revision=1))

    assert bus.publish(
        AgentEvent.partial("stock", acoustic=lineage, revision=2)
    )

    snapshot = bus.stats()
    assert snapshot.pending == snapshot.pending_data == 2
    assert snapshot.pending_control == 0
    assert snapshot.evicted_data == 1
    assert snapshot.coalesced == 1
    bus.drain()
    assert seen == ["b", "stock"]


def test_unadmittable_correction_still_retires_stale_control_partial():
    bus = EventBus(max_events=3, control_reserve=1)
    seen: list[EventKind] = []
    bus.subscribe(lambda event: seen.append(event.kind))
    for index in range(2):
        bus.publish(
            AgentEvent(
                EventKind.INTENT_DECISION,
                {"index": index},
                priority=40,
            )
        )
    lineage = _lineage("protected-correction")
    bus.publish(AgentEvent.partial("stop", acoustic=lineage, revision=1))

    assert not bus.publish(
        AgentEvent.partial("stock", acoustic=lineage, revision=2)
    )

    snapshot = bus.stats()
    assert snapshot.pending == snapshot.pending_data == 2
    assert snapshot.pending_control == 0
    assert snapshot.rejected_data == 1
    bus.drain()
    assert seen == [EventKind.INTENT_DECISION, EventKind.INTENT_DECISION]


@pytest.mark.parametrize("control_text", ["stop", "assistant mode"])
def test_correcting_final_retires_queued_exact_control_partial(control_text):
    bus = EventBus(max_events=4, control_reserve=2)
    seen: list[AgentEvent] = []
    bus.subscribe(seen.append)
    lineage = _lineage("corrected-control")
    final = AgentEvent.final("stock", acoustic=lineage, revision=2)

    bus.publish(AgentEvent.partial(control_text, acoustic=lineage, revision=1))
    bus.publish(final)

    snapshot = bus.stats()
    assert snapshot.retired_partials == 1
    assert snapshot.pending == 1
    bus.drain()
    assert seen == [final]


def test_replacement_uses_incoming_sequence_and_cannot_leapfrog_stop():
    bus = EventBus(max_events=8, control_reserve=2)
    seen: list[str] = []
    bus.subscribe(lambda event: seen.append(str(event.payload.get("text", ""))))

    bus.publish(_partial("turn-a", 1, "a1"))
    bus.publish(_partial("turn-stop", 1, "stop"))
    bus.publish(_partial("turn-a", 2, "a2"))
    bus.drain()

    assert seen == ["stop", "a2"]


def test_reentrant_control_preempts_remaining_lower_priority_data():
    bus = EventBus(max_events=8, control_reserve=2)
    seen: list[EventKind] = []
    published = False

    def handler(event: AgentEvent) -> None:
        nonlocal published
        seen.append(event.kind)
        if not published:
            published = True
            bus.publish(AgentEvent.stop())

    bus.subscribe(handler)
    bus.publish(AgentEvent(EventKind.TASK_STARTED, priority=40))
    bus.publish(AgentEvent(EventKind.TASK_PROGRESS, priority=70))
    bus.drain()

    assert seen == [
        EventKind.TASK_STARTED,
        EventKind.CONTROL_STOP,
        EventKind.TASK_PROGRESS,
    ]


def test_drain_once_dispatches_exactly_one_across_lanes():
    bus = EventBus(max_events=8, control_reserve=2)
    seen: list[EventKind] = []
    bus.subscribe(lambda event: seen.append(event.kind))
    bus.publish(AgentEvent(EventKind.TASK_PROGRESS, priority=70))
    bus.publish(AgentEvent.stop())

    assert bus.drain_once()
    assert seen == [EventKind.CONTROL_STOP]
    assert bus.outstanding_count == 1
    assert bus.drain_once()
    assert seen == [EventKind.CONTROL_STOP, EventKind.TASK_PROGRESS]
    assert not bus.drain_once()
    assert bus.idle()


def test_idle_counts_event_currently_executing_in_handler():
    bus = EventBus(max_events=8, control_reserve=2)
    entered = Event()
    release = Event()
    handler_timed_out = Event()

    def handler(_event: AgentEvent) -> None:
        entered.set()
        if not release.wait(2.0):
            handler_timed_out.set()

    bus.subscribe(handler)
    bus.publish(AgentEvent.final("hello"))
    bus.start()
    try:
        assert entered.wait(2.0)
        assert bus.stats().pending == 0
        assert bus.outstanding_count == 1
        assert not bus.idle()
        release.set()
        assert _wait_for_idle(bus)
        assert not handler_timed_out.is_set()
    finally:
        release.set()
        bus.stop()


def test_publish_does_not_wait_for_blocked_handler():
    bus = EventBus(max_events=8, control_reserve=2)
    entered = Event()
    release = Event()
    producer_done = Event()

    def handler(_event: AgentEvent) -> None:
        entered.set()
        release.wait(2.0)

    def produce() -> None:
        for revision in range(2_000):
            bus.publish(_partial("storm", revision))
        producer_done.set()

    bus.subscribe(handler)
    bus.publish(AgentEvent.final("blocks"))
    bus.start()
    producer = Thread(
        target=produce,
        daemon=True,
    )
    producer_started = False
    try:
        assert entered.wait(2.0)
        producer.start()
        producer_started = True
        assert producer_done.wait(2.0)
        producer.join(timeout=1.0)
        assert not producer.is_alive()
        assert bus.stats().pending == 1
    finally:
        release.set()
        if producer_started:
            producer.join(timeout=2.0)
        _wait_for_idle(bus)
        bus.stop()


def test_subscribers_run_in_registration_order():
    bus = EventBus()
    seen: list[str] = []
    bus.subscribe(lambda _event: seen.append("supervisor"))
    bus.subscribe(lambda _event: seen.append("runtime"))
    bus.subscribe(lambda _event: seen.append("trace"))
    bus.publish(AgentEvent.final("hello"))
    bus.drain()
    assert seen == ["supervisor", "runtime", "trace"]


def test_data_pressure_warning_is_private_once_and_rearms(caplog):
    bus = EventBus(max_events=4, control_reserve=1)
    with caplog.at_level(logging.WARNING, logger="speaker.event_bus"):
        for index in range(8):
            bus.publish(_partial(f"first-{index}", 1, f"private-{index}"))
    warnings = [
        record for record in caplog.records if "data pressure" in record.message
    ]
    assert len(warnings) == 1
    assert all("private-" not in record.message for record in warnings)

    bus.drain()
    with caplog.at_level(logging.WARNING, logger="speaker.event_bus"):
        for index in range(4):
            bus.publish(_partial(f"second-{index}", 1, f"secret-{index}"))
    warnings = [
        record for record in caplog.records if "data pressure" in record.message
    ]
    assert len(warnings) == 2
    assert all("secret-" not in record.message for record in warnings)
    bus.drain()
    assert bus.idle()
