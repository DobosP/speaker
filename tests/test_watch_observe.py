"""Tier-0 tests for the watch/monitor observe loop (``core/watch.py``).

Self-contained: defines its own tiny fakes (source / evaluator / clock) and never
touches a real display, ``mss``/Xlib/tesseract, or a model. Every ``GrantStore`` is
pointed at a tmp ``config.local.json`` so the real machine-local config is never
written.

Covers the security invariants the watch loop must hold:
- INV-2  live-recheck: the grant is re-fetched from the store on every tick; a
         revoked grant stops the watch and is never observed again.
- INV-3  ephemerality: the captured ``Observation`` (and any image-derived text) is
         never stored anywhere reachable after ``tick`` -- only the spoken alert text
         survives.
- INV-5  bounded: a hard cap on active watches + a poll-interval floor.
- INV-6  system-origin: the fired alert is epoch-stamped, tagged ``Origin.SCREEN``,
         and NOT tagged owner_verified (watched screen text can shape a heads-up but
         can never originate an action).
"""
from __future__ import annotations

import gc

import pytest

from always_on_agent.events import AgentEvent, EventKind
from always_on_agent.origin import Origin
from core.watch import ActiveWatch, GrantStore, WatchGrant, WatchManager
from core.watch_source import Observation, WindowRect


# --- fakes -----------------------------------------------------------------

class FakeSource:
    """A WatchSource that returns a fixed Observation and counts observe() calls."""

    def __init__(self, obs):
        self._obs = obs
        self.calls = 0

    def observe(self, app):
        self.calls += 1
        return self._obs


class FakeEvaluator:
    """Returns scripted (met, evidence) verdicts, one per evaluate() call. When the
    script runs out it keeps returning the last verdict (so 'stays True' is easy)."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = 0

    def evaluate(self, obs, condition):
        self.calls += 1
        idx = min(self.calls - 1, len(self._script) - 1)
        return self._script[idx]


class FakeClock:
    """A controllable monotonic clock: now() reads it, advance() moves it forward."""

    def __init__(self, start=0.0):
        self.t = float(start)

    def now(self):
        return self.t

    def advance(self, dt):
        self.t += float(dt)


def _collector():
    """A publish() sink that records every AgentEvent it receives."""
    events: list[AgentEvent] = []
    return events, events.append


def _grant(grant_id="g1", *, min_poll_sec=5.0, wm_class="org.app.Chat"):
    return WatchGrant(
        id=grant_id,
        label=f"label-{grant_id}",
        app={"wm_class": wm_class, "backend": "x11"},
        min_poll_sec=min_poll_sec,
    )


def _make_manager(tmp_path, *, grants, source, evaluator=None, epoch=7,
                  max_active=2, min_poll_sec=5.0, clock=None):
    """Build a GrantStore (rooted at a tmp local path) + a manager with autostart
    OFF so NO poller thread runs -- the test drives ``tick`` directly."""
    clock = clock or FakeClock()
    store = GrantStore(
        [g.to_dict() for g in grants],
        local_path=str(tmp_path / "config.local.json"),
    )
    events, publish = _collector()
    mgr = WatchManager(
        store,
        source=source,
        publish=publish,
        current_epoch=lambda: epoch,
        evaluator=evaluator,
        max_active=max_active,
        min_poll_sec=min_poll_sec,
        autostart=False,
        clock=clock.now,
    )
    return mgr, store, events, clock


def _tts_events(events):
    return [e for e in events if e.kind is EventKind.TTS_REQUEST]


# --- condition fires on the false->true transition (one_shot) --------------

def test_one_shot_fires_once_on_transition(tmp_path):
    obs = Observation(text="ignored", words=(), rect=None)
    src = FakeSource(obs)
    # tick 1: not met; tick 2: met (the edge -> fire); tick 3: still met (silent).
    ev = FakeEvaluator([(False, ""), (True, "ORDER SHIPPED"), (True, "ORDER SHIPPED")])
    clock = FakeClock()
    mgr, store, events, clock = _make_manager(
        tmp_path, grants=[_grant()], source=src, evaluator=ev, clock=clock,
    )
    wid = mgr.start_watch("g1", "shipped", interval_sec=5.0, one_shot=True)

    # tick 1 -- due immediately (next_due == clock at start), not met.
    mgr.tick()
    assert _tts_events(events) == []
    assert any(w.watch_id == wid for w in mgr.active())

    # tick 2 -- advance past the interval; condition flips False->True -> ONE alert.
    clock.advance(10.0)
    mgr.tick()
    fired = _tts_events(events)
    assert len(fired) == 1
    # one_shot -> the watch is removed from active() immediately after firing.
    assert all(w.watch_id != wid for w in mgr.active())

    # tick 3 -- still 'met' per the script, but the watch is gone: no observe, no alert.
    calls_before = src.calls
    clock.advance(10.0)
    mgr.tick()
    assert _tts_events(events) == fired  # still exactly one
    assert src.calls == calls_before    # stopped watch is never observed again


# --- repeat watch: fires on every false->true EDGE, silent while staying true --

def test_repeat_fires_on_each_rising_edge(tmp_path):
    obs = Observation(text="ignored", words=())
    src = FakeSource(obs)
    # F -> T(fire) -> T(silent) -> F(reset) -> T(fire again)
    ev = FakeEvaluator([
        (False, ""),
        (True, "edge-1"),
        (True, "edge-1"),
        (False, ""),
        (True, "edge-2"),
    ])
    clock = FakeClock()
    mgr, store, events, clock = _make_manager(
        tmp_path, grants=[_grant()], source=src, evaluator=ev, clock=clock,
    )
    wid = mgr.start_watch("g1", "ping", interval_sec=5.0, one_shot=False)

    def step():
        clock.advance(10.0)  # always past next_due so the watch is due
        mgr.tick()

    mgr.tick()            # tick 1: False -- no fire
    assert _tts_events(events) == []
    step()                # tick 2: True (rising edge) -> fire #1
    assert len(_tts_events(events)) == 1
    step()                # tick 3: still True -> silent
    assert len(_tts_events(events)) == 1
    step()                # tick 4: back to False -> silent, arms the next edge
    assert len(_tts_events(events)) == 1
    step()                # tick 5: True again (rising edge) -> fire #2
    assert len(_tts_events(events)) == 2

    # repeat watch is NOT one_shot -> still active after firing.
    assert any(w.watch_id == wid for w in mgr.active())


# --- alert payload: epoch + SCREEN origin, NOT owner_verified ---------------

def test_alert_payload_epoch_origin_not_owner_verified(tmp_path):
    obs = Observation(text="ignored")
    src = FakeSource(obs)
    ev = FakeEvaluator([(False, ""), (True, "match")])
    clock = FakeClock()
    mgr, store, events, clock = _make_manager(
        tmp_path, grants=[_grant()], source=src, evaluator=ev, epoch=42, clock=clock,
    )
    mgr.start_watch("g1", "shipped", interval_sec=5.0, one_shot=True)
    mgr.tick()
    clock.advance(10.0)
    mgr.tick()

    fired = _tts_events(events)
    assert len(fired) == 1
    payload = fired[0].payload
    assert payload["epoch"] == 42                       # carries current_epoch()
    assert payload["origin"] == Origin.SCREEN.value     # system/screen origin
    assert payload["origin"] != Origin.LIVE_AUDIO.value
    # A heads-up is DATA, never an owner-trusted action lineage.
    assert payload.get("owner_verified") is None
    assert "owner_verified" not in payload
    assert payload["text"]  # something is spoken


# --- ephemerality: the Observation / image text is never retained -----------

def test_observation_not_retained_anywhere(tmp_path):
    # A sentinel string we can hunt for through the manager's reachable object graph.
    secret = "EPHEMERAL_SENTINEL_9c3f"
    rect = WindowRect(left=0, top=0, width=10, height=10, wm_class="org.app.Chat")
    obs = Observation(text=secret, words=({"text": secret},), rect=rect)
    src = FakeSource(obs)
    # Never matches -> nothing of the captured text reaches the spoken alert.
    ev = FakeEvaluator([(False, "")])
    clock = FakeClock()
    mgr, store, events, clock = _make_manager(
        tmp_path, grants=[_grant()], source=src, evaluator=ev, clock=clock,
    )
    wid = mgr.start_watch("g1", "never", interval_sec=5.0, one_shot=True)
    mgr.tick()
    assert src.calls == 1  # it really did observe

    # The Observation object must not be held by the manager OR the ActiveWatch.
    watch = mgr.active()[0]
    for holder in (mgr.__dict__, watch.__dict__):
        for value in holder.values():
            assert value is not obs

    # And no reachable string-bearing attribute echoes the captured text.
    def _no_secret(container):
        for value in container.values():
            assert not (isinstance(value, str) and secret in value)

    _no_secret(mgr.__dict__)
    _no_secret(watch.__dict__)
    # Nothing was spoken, so the secret never left the tick at all.
    assert _tts_events(events) == []

    # The only objects that survive referencing `obs` should be our own test locals
    # (src._obs, obs). The manager must not be among obs's referrers.
    gc.collect()
    referrers = gc.get_referrers(obs)
    assert mgr not in referrers
    assert mgr.__dict__ not in referrers
    assert watch not in referrers


def test_matched_text_redacted_but_no_observation_object_kept(tmp_path):
    # When it DOES match, the alert text survives -- but the Observation object and
    # its raw OCR words still must not be retained by the manager.
    rect = WindowRect(left=0, top=0, width=10, height=10, wm_class="org.app.Chat")
    obs = Observation(text="ORDER 123 shipped", words=({"text": "ORDER"},), rect=rect)
    src = FakeSource(obs)
    ev = FakeEvaluator([(True, "ORDER 123 shipped")])
    clock = FakeClock()
    mgr, store, events, clock = _make_manager(
        tmp_path, grants=[_grant()], source=src, evaluator=ev, clock=clock,
    )
    mgr.start_watch("g1", "shipped", interval_sec=5.0, one_shot=True)
    mgr.tick()

    assert len(_tts_events(events)) == 1
    gc.collect()
    referrers = gc.get_referrers(obs)
    assert mgr not in referrers and mgr.__dict__ not in referrers
    # active() is now empty (one_shot fired) -- no ActiveWatch holds the obs either.
    assert mgr.active() == ()


# --- stop_watch(watch_id): the watch is no longer observed ------------------

def test_stop_watch_by_id_stops_observation(tmp_path):
    obs = Observation(text="ignored")
    src = FakeSource(obs)
    ev = FakeEvaluator([(False, "")])  # never fires; keeps it active
    clock = FakeClock()
    mgr, store, events, clock = _make_manager(
        tmp_path, grants=[_grant()], source=src, evaluator=ev, clock=clock,
    )
    wid = mgr.start_watch("g1", "x", interval_sec=5.0, one_shot=False)

    mgr.tick()
    assert src.calls == 1

    stopped = mgr.stop_watch(wid)
    assert stopped == 1
    assert all(w.watch_id != wid for w in mgr.active())

    calls_before = src.calls
    clock.advance(10.0)
    mgr.tick()
    assert src.calls == calls_before  # the stopped watch is not observed
    assert wid not in {w.watch_id for w in mgr.active()}


# --- max_active enforced ----------------------------------------------------

def test_max_active_enforced(tmp_path):
    obs = Observation(text="ignored")
    src = FakeSource(obs)
    grants = [_grant("g1"), _grant("g2"), _grant("g3")]
    mgr, store, events, clock = _make_manager(
        tmp_path, grants=grants, source=src, max_active=2,
    )
    mgr.start_watch("g1", "a", interval_sec=5.0, one_shot=False)
    mgr.start_watch("g2", "b", interval_sec=5.0, one_shot=False)
    with pytest.raises(RuntimeError):
        mgr.start_watch("g3", "c", interval_sec=5.0, one_shot=False)
    assert len(mgr.active()) == 2


# --- interval clamped UP to the floor ---------------------------------------

def test_interval_clamped_to_floor(tmp_path):
    obs = Observation(text="ignored")
    src = FakeSource(obs)
    # Both the manager floor (5.0) and the grant floor (8.0) exceed the request (1.0).
    mgr, store, events, clock = _make_manager(
        tmp_path, grants=[_grant(min_poll_sec=8.0)], source=src, min_poll_sec=5.0,
    )
    wid = mgr.start_watch("g1", "x", interval_sec=1.0, one_shot=False)
    watch = next(w for w in mgr.active() if w.watch_id == wid)
    # max(min_poll_sec=5.0, grant.min_poll_sec=8.0, interval_sec=1.0) == 8.0
    assert watch.interval_sec == 8.0


def test_interval_floor_is_manager_min_when_grant_lower(tmp_path):
    obs = Observation(text="ignored")
    src = FakeSource(obs)
    # grant floor (2.0) and request (1.0) both below the manager floor (5.0).
    mgr, store, events, clock = _make_manager(
        tmp_path, grants=[_grant(min_poll_sec=2.0)], source=src, min_poll_sec=5.0,
    )
    wid = mgr.start_watch("g1", "x", interval_sec=1.0, one_shot=False)
    watch = next(w for w in mgr.active() if w.watch_id == wid)
    assert watch.interval_sec == 5.0


# --- revoke mid-loop: live grant recheck stops the watch (INV-2) ------------

def test_revoke_mid_loop_stops_without_observe(tmp_path):
    obs = Observation(text="ignored")
    src = FakeSource(obs)
    ev = FakeEvaluator([(False, "")])
    clock = FakeClock()
    mgr, store, events, clock = _make_manager(
        tmp_path, grants=[_grant()], source=src, evaluator=ev, clock=clock,
    )
    wid = mgr.start_watch("g1", "x", interval_sec=5.0, one_shot=False)

    mgr.tick()
    assert src.calls == 1
    assert any(w.watch_id == wid for w in mgr.active())

    # Revoke the grant between ticks. The live store.get() recheck must catch it.
    assert store.remove("g1") is True

    calls_before = src.calls
    clock.advance(10.0)
    mgr.tick()
    # The revoked watch must NOT be observed (no source.observe call) ...
    assert src.calls == calls_before
    # ... and it must be stopped (removed from active()).
    assert all(w.watch_id != wid for w in mgr.active())
    assert mgr.active() == ()
