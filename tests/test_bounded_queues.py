"""Backlog: bounded queued-tasks admission + bounded runlog queue under storms."""

from __future__ import annotations

import logging
import queue as queue_mod

from always_on_agent.models import IntentDecision, IntentKind
from always_on_agent.supervisor import AgentSupervisor
from core.runlog import _ThreadQueueHandler


# --- supervisor queued_tasks cap -------------------------------------------------


def _task(sup, text):
    return sup.tasks.create_task(
        IntentDecision(kind=IntentKind.ASSISTANT, confidence=1.0, text=text, reason="t")
    )


def test_queue_overflow_drops_oldest_and_caps_length():
    sup = AgentSupervisor(max_queued_tasks=3)
    tasks = [_task(sup, f"turn {i}") for i in range(5)]
    for task in tasks:
        sup._queue_task(task)

    assert len(sup.state.queued_tasks) == 3
    queued_ids = {t.task_id for t in sup.state.queued_tasks}
    # Oldest two dropped, newest three kept (newest-input-wins).
    assert tasks[0].task_id not in queued_ids and tasks[1].task_id not in queued_ids
    assert {t.task_id for t in tasks[2:]} == queued_ids
    # Dropped tasks are CANCELLED so a concurrent drain can't resurrect them.
    assert tasks[0].cancel_event.is_set() and tasks[1].cancel_event.is_set()
    assert any("dropped (queue full)" in f for f in sup.state.failures)


def test_queue_overflow_announces_once_per_storm():
    sup = AgentSupervisor(max_queued_tasks=2)
    for i in range(6):
        sup._queue_task(_task(sup, f"turn {i}"))
    notices = [s for s in sup.state.spoken_outputs if "at capacity" in s]
    assert len(notices) == 1


def test_queue_overflow_prefers_dropping_cold_task_over_continuation():
    sup = AgentSupervisor(max_queued_tasks=2)
    continuation = _task(sup, "and also")
    continuation.metadata["continuation_of"] = "parent-task"
    cold = _task(sup, "old cold request")
    sup._queue_task(continuation)
    sup._queue_task(cold)
    sup._queue_task(_task(sup, "newest"))

    queued_ids = {t.task_id for t in sup.state.queued_tasks}
    assert continuation.task_id in queued_ids  # continuation survives
    assert cold.task_id not in queued_ids  # the cold task was the victim


def test_cancel_all_still_clears_a_full_queue():
    sup = AgentSupervisor(max_queued_tasks=4)
    for i in range(4):
        sup._queue_task(_task(sup, f"turn {i}"))
    sup.cancel_all()
    assert sup.state.queued_tasks == []


# --- runlog bounded queue --------------------------------------------------------


def _record(level, msg):
    return logging.LogRecord("speaker.test", level, __file__, 0, msg, None, None)


def test_debug_storm_is_dropped_counted_and_coalesced():
    q: "queue_mod.Queue" = queue_mod.Queue(maxsize=2)
    handler = _ThreadQueueHandler(q)

    handler.enqueue(_record(logging.DEBUG, "a"))
    handler.enqueue(_record(logging.DEBUG, "b"))
    for i in range(10):  # queue full: all dropped + counted
        handler.enqueue(_record(logging.DEBUG, f"storm {i}"))
    assert q.qsize() == 2

    q.get_nowait()  # listener catches up a little
    q.get_nowait()
    handler.enqueue(_record(logging.DEBUG, "after storm"))

    drained = []
    while not q.empty():
        drained.append(q.get_nowait())
    messages = [r.getMessage() for r in drained]
    assert "after storm" in messages
    assert any("dropped 10 record(s)" in m for m in messages)


def test_warning_survives_backpressure_when_listener_recovers():
    q: "queue_mod.Queue" = queue_mod.Queue(maxsize=1)
    handler = _ThreadQueueHandler(q)
    handler._WARN_PUT_TIMEOUT_SEC = 0.05  # keep the test fast

    handler.enqueue(_record(logging.DEBUG, "fills the queue"))
    import threading

    def drain_soon():
        q.get()  # frees the slot while the WARNING put is blocking

    t = threading.Thread(target=drain_soon, daemon=True)
    t.start()
    handler.enqueue(_record(logging.WARNING, "must survive"))
    t.join(timeout=2)

    drained = [q.get_nowait().getMessage() for _ in range(q.qsize())]
    assert "must survive" in drained


def test_saturated_warning_is_counted_not_lost_silently():
    q: "queue_mod.Queue" = queue_mod.Queue(maxsize=2)
    handler = _ThreadQueueHandler(q)
    handler._WARN_PUT_TIMEOUT_SEC = 0.01

    handler.enqueue(_record(logging.DEBUG, "fill 1"))
    handler.enqueue(_record(logging.DEBUG, "fill 2"))
    handler.enqueue(_record(logging.WARNING, "storm warning"))  # times out -> counted
    assert handler._dropped == 1

    q.get_nowait()  # listener recovers
    q.get_nowait()
    handler.enqueue(_record(logging.DEBUG, "post storm"))
    messages = [q.get_nowait().getMessage() for _ in range(q.qsize())]
    assert "post storm" in messages
    assert any("dropped 1 record(s)" in m for m in messages)
