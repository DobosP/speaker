"""EventBus shutdown + backlog contracts (2026-07-06 recon follow-up).

The bus thread exits between ``get`` timeouts, so an event published just
before ``stop()`` stays queued. ``stop()`` DISCARDS those by default -- the
queue may hold action-producing events (a ``TTS_REQUEST`` whose handler starts
speaking), and executing them after the runtime decided to stop is worse than
dropping them (codex-review 2026-07-06 caught exactly that on
``VoiceRuntime.stop``, core/runtime.py bus.stop()). These pin: (1) default
stop() discards -- a queued TTS_REQUEST must NOT dispatch; (2) ``drain=True``
dispatches leftovers through the same exception-guarded path as live dispatch,
for pure/log-only buses; (3) the unbounded queue WARNs once per high-water
crossing instead of growing silently (publish must never block, so a soft mark
is the observability compromise).
"""
from __future__ import annotations

import logging

from always_on_agent.event_bus import EventBus
from always_on_agent.events import AgentEvent, EventKind


def _event(text: str = "hi", kind: EventKind = EventKind.STT_FINAL) -> AgentEvent:
    return AgentEvent(kind, {"text": text})


def test_stop_discards_pending_events_by_default():
    # The VoiceRuntime.stop() contract: a TTS_REQUEST racing shutdown must be
    # dropped, never dispatched (its handler would call engine.speak).
    bus = EventBus()
    spoken: list[AgentEvent] = []
    bus.subscribe(spoken.append)
    bus.publish(_event("late reply", EventKind.TTS_REQUEST))
    bus.stop()
    assert spoken == []        # discarded, not executed at shutdown
    assert not bus.idle()      # the loss is visible, not silent


def test_stop_drain_true_dispatches_leftovers():
    bus = EventBus()
    seen: list[str] = []
    bus.subscribe(lambda e: seen.append(e.payload["text"]))
    # No dispatch thread running (never started) -- the worst case: everything
    # published is still queued when stop() is called.
    for i in range(3):
        bus.publish(_event(f"e{i}"))
    bus.stop(drain=True)
    assert seen == ["e0", "e1", "e2"]
    assert bus.idle()


def test_stop_drain_survives_raising_handler():
    # The opt-in drain uses the same exception guard as live dispatch: one bad
    # handler drops its event, the rest of the queue still drains.
    bus = EventBus()
    seen: list[str] = []

    def _raise(_event):
        raise RuntimeError("boom")

    bus.subscribe(_raise)
    bus.subscribe(lambda e: seen.append(e.payload["text"]))
    bus.publish(_event("a"))
    bus.publish(_event("b"))
    bus.stop(drain=True)
    assert seen == ["a", "b"]


def test_publish_warns_once_at_high_water(caplog):
    bus = EventBus()
    with caplog.at_level(logging.WARNING, logger="speaker.event_bus"):
        for _ in range(EventBus._HIGH_WATER + 5):
            bus.publish(_event())
    warnings = [r for r in caplog.records if "backlog" in r.getMessage()]
    assert len(warnings) == 1  # once per crossing, not per publish
    # Draining below half the mark re-arms the warning for the next crossing.
    bus.drain()
    with caplog.at_level(logging.WARNING, logger="speaker.event_bus"):
        for _ in range(EventBus._HIGH_WATER):
            bus.publish(_event())
    warnings = [r for r in caplog.records if "backlog" in r.getMessage()]
    assert len(warnings) == 2
