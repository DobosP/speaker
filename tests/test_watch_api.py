"""Tier-0 tests for the remaining watch/monitor public-API surface
(``core/watch.py``) that the invariant-focused suites
(``test_watch_{source,observe,grant_gate,egress}.py``) don't pin down directly.

This file is the "behavioral contract" layer: the pure ``TextMatchEvaluator``
grammar, ``GrantStore`` CRUD + atomic machine-local persistence + dataclass
round-trips, ``WatchManager.stop_watch(None)`` (stop-all), and the read-only
``watch.list`` / ``watch.stop`` capability handlers plus the planner-isolation
spec invariant (none of the four watch capabilities is ever a planner tool, so
an LLM can never arm/list/egress one).

Self-contained + Tier-0: defines its own tiny fakes, never touches a real
display / mss / Xlib / tesseract / model, and ALWAYS points GrantStore at a tmp
``config.local.json`` so the real machine-local config is never written.
"""
from __future__ import annotations

import json

import pytest

from always_on_agent.capabilities import CapabilityRegistry
from always_on_agent.origin import Origin
from core.watch import (
    GrantStore,
    TextMatchEvaluator,
    WatchGrant,
    WatchManager,
    attach_watch_capability,
)
from core.watch_source import Observation


# --- tiny, self-contained fakes --------------------------------------------

class FakeSource:
    """A deterministic WatchSource: returns a fixed Observation (or None) and
    counts observe() calls. No real display."""

    def __init__(self, obs=None):
        self._obs = obs
        self.calls = 0

    def observe(self, app):
        self.calls += 1
        return self._obs


class FakeEvaluator:
    """Never matches -- keeps a started watch alive across ticks for stop tests."""

    def evaluate(self, obs, condition):
        return False, ""


def _make_manager(store, *, source=None, max_active=2, min_poll_sec=5.0):
    """A WatchManager with NO poller thread (autostart=False) so tests drive it
    purely through the explicit API."""
    published: list = []
    mgr = WatchManager(
        store,
        source=source if source is not None else FakeSource(None),
        publish=published.append,
        current_epoch=lambda: 7,
        evaluator=FakeEvaluator(),
        max_active=max_active,
        min_poll_sec=min_poll_sec,
        autostart=False,
    )
    return mgr, published


def _registry_with_watch(store, *, source=None):
    mgr, published = _make_manager(store, source=source)
    registry = CapabilityRegistry()
    attach_watch_capability(registry, {"enabled": True}, manager=mgr)
    return registry, mgr, published


def _owner_ctx(**extra):
    ctx = {"owner_verified": True, "origin": Origin.LIVE_AUDIO.value}
    ctx.update(extra)
    return ctx


def _grant_dict(grant_id="g1", *, wm_class="org.app.Chat", min_poll_sec=5.0):
    return {
        "id": grant_id,
        "label": f"label-{grant_id}",
        "app": {"wm_class": wm_class, "backend": "x11"},
        "min_poll_sec": min_poll_sec,
    }


# === TextMatchEvaluator grammar (pure, local) ===============================

def test_evaluator_substring_is_case_insensitive():
    ev = TextMatchEvaluator()
    met, evidence = ev.evaluate(Observation(text="Order #4 SHIPPED today"), "shipped")
    assert met is True
    assert "SHIPPED" in evidence  # evidence is the raw (un-lowered) matched window


def test_evaluator_regex_is_case_insensitive_and_anchored_by_slashes():
    ev = TextMatchEvaluator()
    met, evidence = ev.evaluate(Observation(text="status: shiPPed now"), "/ship+ed/")
    assert met is True
    assert "shiPPed" in evidence


def test_evaluator_no_match_returns_false_empty_evidence():
    ev = TextMatchEvaluator()
    assert ev.evaluate(Observation(text="nothing relevant here"), "shipped") == (False, "")


def test_evaluator_empty_text_or_condition_never_matches():
    ev = TextMatchEvaluator()
    assert ev.evaluate(Observation(text=""), "anything") == (False, "")
    assert ev.evaluate(Observation(text="abc"), "") == (False, "")
    assert ev.evaluate(Observation(text="abc"), "   ") == (False, "")  # whitespace-only


def test_evaluator_invalid_regex_fails_closed_not_raises():
    """A malformed regex condition must NOT crash the poller -- it fails closed
    to a benign no-match."""
    ev = TextMatchEvaluator()
    assert ev.evaluate(Observation(text="abc[def"), "/[/") == (False, "")


def test_evaluator_lone_slash_is_substring_not_regex():
    """A single '/' is too short to be the ``/regex/`` form (needs len>=2), so it
    is treated as a plain substring search -- matches literal '/' in the text."""
    ev = TextMatchEvaluator()
    met, evidence = ev.evaluate(Observation(text="path a/b here"), "/")
    assert met is True
    assert "/" in evidence


def test_evaluator_snippet_is_capped_and_contains_match():
    """Evidence is windowed around the match and capped (so a huge OCR dump can't
    blow up the alert), but always contains the matched token."""
    ev = TextMatchEvaluator()
    text = "A" * 300 + "NEEDLE" + "B" * 300
    met, evidence = ev.evaluate(Observation(text=text), "NEEDLE")
    assert met is True
    assert "NEEDLE" in evidence
    assert len(evidence) <= 160  # _EVIDENCE_CAP


# === WatchGrant dataclass round-trip + defaults ============================

def test_grant_from_dict_round_trips_and_applies_defaults():
    g = WatchGrant.from_dict({"id": "x1", "app": {"wm_class": "foo"}})
    # label falls back to id; granted_by defaults to owner_verified; floor default 5.0.
    assert g.id == "x1"
    assert g.label == "x1"
    assert g.granted_by == "owner_verified"
    assert g.min_poll_sec == 5.0
    # round-trips through to_dict/from_dict identically.
    again = WatchGrant.from_dict(g.to_dict())
    assert again == g


def test_grant_from_dict_coerces_nonsensical_poll_floors():
    """A nonsensical poll floor is never allowed to go negative: a falsy 0 (or a
    missing key) falls back to the 5.0 default, and a negative value is floored to
    0.0 (the WatchManager's own min_poll_sec is the real lower bound at start time,
    so the grant floor can be 0 without enabling a tight loop)."""
    # 0 is falsy -> the ``or 5.0`` fallback kicks in -> default.
    assert WatchGrant.from_dict({"id": "a", "app": {}, "min_poll_sec": 0}).min_poll_sec == 5.0
    # missing -> default.
    assert WatchGrant.from_dict({"id": "a", "app": {}}).min_poll_sec == 5.0
    # negative is truthy -> kept, but clamped up to 0.0 (never below zero).
    assert WatchGrant.from_dict({"id": "a", "app": {}, "min_poll_sec": -3}).min_poll_sec == 0.0


# === GrantStore CRUD + atomic machine-local persistence ====================

def test_store_default_deny_starts_empty(tmp_path):
    store = GrantStore(local_path=str(tmp_path / "config.local.json"))
    assert store.list() == ()
    assert store.get("anything") is None


def test_store_ignores_seed_entries_without_id(tmp_path):
    """A seed grant with no id is dropped (can't address a nameless grant)."""
    store = GrantStore(
        [{"app": {"wm_class": "x"}}, _grant_dict("real")],
        local_path=str(tmp_path / "config.local.json"),
    )
    assert [g.id for g in store.list()] == ["real"]


def test_store_add_persists_and_enables_and_is_atomic(tmp_path):
    """add() writes the grant + watch.enabled=true to config.local.json via an
    atomic temp-then-rename (no leftover .tmp), and get() finds it in memory."""
    local = tmp_path / "config.local.json"
    store = GrantStore(local_path=str(local))

    store.add(WatchGrant.from_dict(_grant_dict("g1")))

    assert store.get("g1") is not None
    written = json.loads(local.read_text(encoding="utf-8"))
    assert written["watch"]["enabled"] is True
    assert [g["id"] for g in written["watch"]["grants"]] == ["g1"]
    # Atomic write left no temp file behind.
    assert not (tmp_path / "config.local.json.tmp").exists()


def test_store_add_replaces_same_id(tmp_path):
    """Re-adding the same id updates in place (no duplicate allowlist entries)."""
    store = GrantStore(local_path=str(tmp_path / "config.local.json"))
    store.add(WatchGrant.from_dict(_grant_dict("g1", wm_class="old")))
    store.add(WatchGrant.from_dict(_grant_dict("g1", wm_class="new")))
    grants = store.list()
    assert len(grants) == 1
    assert grants[0].app["wm_class"] == "new"


def test_store_remove_returns_bool_and_persists(tmp_path):
    local = tmp_path / "config.local.json"
    store = GrantStore([_grant_dict("g1"), _grant_dict("g2")], local_path=str(local))

    assert store.remove("g1") is True
    assert store.remove("g1") is False  # already gone
    assert [g.id for g in store.list()] == ["g2"]

    written = json.loads(local.read_text(encoding="utf-8"))
    assert [g["id"] for g in written["watch"]["grants"]] == ["g2"]


def test_store_preserves_unrelated_local_keys_on_write(tmp_path):
    """Persisting a grant must not clobber other machine-local config keys."""
    local = tmp_path / "config.local.json"
    local.write_text(
        json.dumps({"device": "thinkpad", "watch": {"some_flag": True}}),
        encoding="utf-8",
    )
    store = GrantStore(local_path=str(local))
    store.add(WatchGrant.from_dict(_grant_dict("g1")))

    written = json.loads(local.read_text(encoding="utf-8"))
    assert written["device"] == "thinkpad"          # unrelated top-level key kept
    assert written["watch"]["some_flag"] is True     # unrelated watch.* key kept
    assert written["watch"]["enabled"] is True       # and the grant turned watching on


def test_store_backs_up_and_aborts_on_corrupt_local_file(tmp_path):
    """A corrupt config.local.json must NOT be silently overwritten with {} -- that
    would destroy every other machine-local key. The grant aborts loudly and the
    corrupt content is preserved as a .corrupt backup (security review, LOW fix)."""
    import pytest

    local = tmp_path / "config.local.json"
    local.write_text("{ this is not json", encoding="utf-8")
    store = GrantStore(local_path=str(local))

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        store.add(WatchGrant.from_dict(_grant_dict("g1")))

    assert (tmp_path / "config.local.json.corrupt").exists()


def test_store_never_creates_committed_config_json(tmp_path):
    """Persistence touches ONLY the local path it was handed -- never a sibling
    committed config.json."""
    local = tmp_path / "config.local.json"
    committed = tmp_path / "config.json"
    store = GrantStore(local_path=str(local))
    store.add(WatchGrant.from_dict(_grant_dict("g1")))
    assert local.exists()
    assert not committed.exists()


# === WatchManager.stop_watch(None) -- stop ALL =============================

def test_stop_all_returns_count_and_clears_active(tmp_path):
    store = GrantStore([_grant_dict("g1"), _grant_dict("g2")],
                       local_path=str(tmp_path / "config.local.json"))
    mgr, _ = _make_manager(store, source=FakeSource(Observation(text="x")))
    mgr.start_watch("g1", "a", interval_sec=5.0, one_shot=False)
    mgr.start_watch("g2", "b", interval_sec=5.0, one_shot=False)
    assert len(mgr.active()) == 2

    n = mgr.stop_watch(None)  # None => stop everything

    assert n == 2
    assert mgr.active() == ()


def test_stop_unknown_watch_id_returns_zero(tmp_path):
    store = GrantStore([_grant_dict("g1")], local_path=str(tmp_path / "config.local.json"))
    mgr, _ = _make_manager(store, source=FakeSource(Observation(text="x")))
    mgr.start_watch("g1", "a", interval_sec=5.0, one_shot=False)
    assert mgr.stop_watch("no-such-watch") == 0
    assert len(mgr.active()) == 1  # the real watch is untouched


def test_start_watch_unknown_grant_raises_value_error(tmp_path):
    store = GrantStore(local_path=str(tmp_path / "config.local.json"))
    mgr, _ = _make_manager(store)
    with pytest.raises(ValueError):
        mgr.start_watch("ghost", "x", interval_sec=5.0)


# === capability handlers: watch.list (read-only) + watch.stop ==============

def test_planner_can_never_reach_any_watch_capability(tmp_path):
    """The load-bearing isolation invariant: all four watch capabilities are
    registered planner_tool=False AND user_facing=False, so registry.planner_tools()
    excludes every one -- an LLM/ReAct planner can never arm, list, or egress a
    watcher."""
    store = GrantStore(local_path=str(tmp_path / "config.local.json"))
    registry, _mgr, _pub = _registry_with_watch(store)

    planner = set(registry.planner_tools())
    for cap in ("watch.grant", "watch.start", "watch.stop", "watch.list"):
        spec = registry.spec(cap)
        assert spec.planner_tool is False
        assert spec.user_facing is False
        assert cap not in planner
    assert not any(n.startswith("watch.") for n in planner)


def test_watch_list_is_read_only_needs_no_owner_verify(tmp_path):
    """watch.list works with NO owner context (read-only: no action, no laundering
    risk) and reports both granted apps and active watches."""
    store = GrantStore([_grant_dict("g1")], local_path=str(tmp_path / "config.local.json"))
    registry, mgr, _pub = _registry_with_watch(store, source=FakeSource(Observation(text="x")))
    mgr.start_watch("g1", "cond", interval_sec=5.0, one_shot=False)

    res = registry.invoke("watch.list", "what are you watching", {})  # no owner ctx

    assert res.ok is True
    assert res.data == {"grants": 1, "active": 1}
    # It is NOT gated -- the read-only handler never returns the owner-block payload.
    assert res.data.get("blocked") != "owner_verification"
    assert "label-g1" in res.text  # names the granted app


def test_watch_list_empty_when_nothing_granted_or_active(tmp_path):
    store = GrantStore(local_path=str(tmp_path / "config.local.json"))
    registry, _mgr, _pub = _registry_with_watch(store)
    res = registry.invoke("watch.list", "watching?", {})
    assert res.ok is True
    assert res.data == {"grants": 0, "active": 0}


def test_watch_stop_blocked_without_owner_verification(tmp_path):
    """watch.stop is side-effecting -> it must route through the owner-verify
    chokepoint; an un-verified turn is refused (and nothing is stopped)."""
    store = GrantStore([_grant_dict("g1")], local_path=str(tmp_path / "config.local.json"))
    registry, mgr, _pub = _registry_with_watch(store, source=FakeSource(Observation(text="x")))
    mgr.start_watch("g1", "cond", interval_sec=5.0, one_shot=False)

    res = registry.invoke("watch.stop", "stop watching", {})  # no owner ctx

    assert res.data["blocked"] == "owner_verification"
    assert res.data.get("executed") is False
    assert len(mgr.active()) == 1  # the watch is still running


def test_watch_stop_owner_verified_stops_all_when_no_id(tmp_path):
    """An owner-verified watch.stop with no watch_id stops everything."""
    store = GrantStore([_grant_dict("g1"), _grant_dict("g2")],
                       local_path=str(tmp_path / "config.local.json"))
    registry, mgr, _pub = _registry_with_watch(store, source=FakeSource(Observation(text="x")))
    mgr.start_watch("g1", "a", interval_sec=5.0, one_shot=False)
    mgr.start_watch("g2", "b", interval_sec=5.0, one_shot=False)

    res = registry.invoke("watch.stop", "stop everything", _owner_ctx())

    assert res.data.get("executed") is True
    assert res.data.get("stopped") == 2
    assert mgr.active() == ()


def test_watch_stop_owner_verified_stops_single_by_id(tmp_path):
    store = GrantStore([_grant_dict("g1"), _grant_dict("g2")],
                       local_path=str(tmp_path / "config.local.json"))
    registry, mgr, _pub = _registry_with_watch(store, source=FakeSource(Observation(text="x")))
    wid1 = mgr.start_watch("g1", "a", interval_sec=5.0, one_shot=False)
    mgr.start_watch("g2", "b", interval_sec=5.0, one_shot=False)

    res = registry.invoke("watch.stop", "stop that one", _owner_ctx(watch_id=wid1))

    assert res.data.get("stopped") == 1
    remaining = {w.watch_id for w in mgr.active()}
    assert wid1 not in remaining
    assert len(remaining) == 1


def test_watch_stop_owner_verified_reports_zero_when_idle(tmp_path):
    store = GrantStore(local_path=str(tmp_path / "config.local.json"))
    registry, _mgr, _pub = _registry_with_watch(store)
    res = registry.invoke("watch.stop", "stop", _owner_ctx())
    assert res.data.get("executed") is True
    assert res.data.get("stopped") == 0


# === capability handlers: watch.grant / watch.start argument validation =====

def test_watch_grant_rejects_missing_wm_class_even_when_owner_verified(tmp_path):
    """Owner-verified but the grant spec lacks app.wm_class -> rejected (a grant
    with no window identity would match nothing / everything; fail-closed)."""
    store = GrantStore(local_path=str(tmp_path / "config.local.json"))
    registry, _mgr, _pub = _registry_with_watch(store)

    res = registry.invoke(
        "watch.grant", "watch it",
        _owner_ctx(grant={"id": "x", "label": "X", "app": {}}),
    )

    assert res.ok is False
    assert store.list() == ()  # nothing entered the allowlist


def test_watch_start_rejects_missing_condition(tmp_path):
    """Owner-verified, granted app, but no condition to watch for -> refused."""
    store = GrantStore([_grant_dict("g1")], local_path=str(tmp_path / "config.local.json"))
    registry, mgr, _pub = _registry_with_watch(store)

    res = registry.invoke(
        "watch.start", "",  # empty query, no condition in context
        _owner_ctx(grant_id="g1", condition=""),
    )

    assert res.ok is False
    assert mgr.active() == ()


def test_watch_start_owner_verified_arms_and_clamps_interval(tmp_path):
    """End-to-end through the capability: an owner-verified start on a granted app
    arms a watch and clamps the requested interval UP to the grant's floor."""
    store = GrantStore([_grant_dict("g1", min_poll_sec=12.0)],
                       local_path=str(tmp_path / "config.local.json"))
    registry, mgr, _pub = _registry_with_watch(store)

    res = registry.invoke(
        "watch.start", "alert on error",
        _owner_ctx(grant_id="g1", condition="error", interval_sec=1.0, one_shot=True),
    )

    assert res.data.get("executed") is True
    active = mgr.active()
    assert len(active) == 1
    assert active[0].interval_sec == 12.0  # clamped up to the grant floor


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
