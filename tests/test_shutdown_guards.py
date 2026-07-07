"""Backlog: clean shutdown for the follow-up timer and the watch poller.

Two late-lifecycle races pinned here: (1) a follow-up Timer callback already in
flight when the supervisor shuts down must not publish into a dead bus, and
nothing may re-arm after shutdown; (2) WatchManager.shutdown() must wake the
poller out of its cadence wait and JOIN it before clearing state, so a late tick
can't race a half-torn-down manager.
"""

from __future__ import annotations

import json
import threading
import time

from always_on_agent.supervisor import AgentSupervisor
from core.watch import GrantStore, WatchManager


# --- supervisor follow-up timer ------------------------------------------------


def test_followup_tick_after_shutdown_publishes_nothing():
    sup = AgentSupervisor()
    published: list = []
    sup.publish = published.append  # type: ignore[method-assign]

    sup.shutdown()
    sup._tick_followup()  # a Timer callback that lost the race with shutdown()

    assert published == []


def test_schedule_followup_after_shutdown_arms_no_timer():
    sup = AgentSupervisor()
    sup.shutdown()
    sup._schedule_followup()
    assert sup._followup_timer is None


def test_followup_still_works_before_shutdown():
    sup = AgentSupervisor()
    published: list = []
    sup.publish = published.append  # type: ignore[method-assign]

    sup._tick_followup()

    assert len(published) == 1
    sup.shutdown()


# --- watch poller ---------------------------------------------------------------


def _store(tmp_path) -> GrantStore:
    path = tmp_path / "config.local.json"
    path.write_text(json.dumps({}), encoding="utf-8")
    return GrantStore(local_path=str(path))


class _CountingSource:
    def observe(self, grant):  # pragma: no cover - never reached (no active watches)
        raise AssertionError("no watch is armed in these tests")


def test_shutdown_joins_poller_and_stops_ticks(tmp_path):
    ticks: list[float] = []
    mgr = WatchManager(
        _store(tmp_path),
        source=_CountingSource(),
        publish=lambda e: None,
        current_epoch=lambda: 1,
        max_active=1,
        min_poll_sec=30.0,  # long cadence: only an interruptible wait can exit fast
        autostart=True,
    )
    mgr.tick = lambda: ticks.append(time.monotonic())  # type: ignore[method-assign]
    mgr._maybe_start_poller()
    poller = mgr._poller
    assert poller is not None and poller.is_alive()

    start = time.monotonic()
    mgr.shutdown()
    elapsed = time.monotonic() - start

    assert elapsed < 5.0  # woke out of the 30 s cadence, didn't sleep it out
    assert mgr._poller is None
    poller.join(timeout=2.0)
    assert not poller.is_alive()
    ticks_at_shutdown = list(ticks)
    time.sleep(0.1)
    assert ticks == ticks_at_shutdown  # no tick after shutdown returned


def test_shutdown_without_poller_is_safe(tmp_path):
    mgr = WatchManager(
        _store(tmp_path),
        source=_CountingSource(),
        publish=lambda e: None,
        current_epoch=lambda: 1,
        autostart=False,
    )
    mgr.shutdown()  # no poller ever started
    assert mgr._poller is None


def test_poller_not_restarted_after_shutdown(tmp_path):
    mgr = WatchManager(
        _store(tmp_path),
        source=_CountingSource(),
        publish=lambda e: None,
        current_epoch=lambda: 1,
        autostart=True,
    )
    mgr.shutdown()
    mgr._maybe_start_poller()
    assert mgr._poller is None  # _shutdown latch holds


def test_shutdown_from_poller_thread_does_not_self_join(tmp_path):
    # A tick handler calling shutdown() runs ON the poller thread — joining
    # yourself deadlocks; the guard must skip the join and still tear down.
    mgr = WatchManager(
        _store(tmp_path),
        source=_CountingSource(),
        publish=lambda e: None,
        current_epoch=lambda: 1,
        min_poll_sec=1.0,
        autostart=True,
    )
    done = threading.Event()

    def tick_then_shutdown():
        mgr.shutdown()
        done.set()

    mgr.tick = tick_then_shutdown  # type: ignore[method-assign]
    mgr._maybe_start_poller()
    assert done.wait(timeout=10.0)
    assert mgr._poller is None
