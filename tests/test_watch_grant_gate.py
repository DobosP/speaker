"""Tier-0 tests for the watch/monitor owner-verification gate + default-deny
persistence (INV-1, INV-7).

Self-contained: defines its own tiny fakes, never touches a real screen /
mss / Xlib / tesseract / model, and always points GrantStore at a tmp
``config.local.json`` so the real machine-local config is never modified.

INV-1 (owner-verification): ``watch.grant`` / ``watch.start`` / ``watch.stop``
route through the always_on_agent.origin chokepoint -- a grant can only be
created from an owner-verified LIVE_AUDIO turn, and ``owner_verified`` must be
the literal boolean ``True`` (a truthy ``1`` does NOT pass).

INV-7 (default-deny + machine-local persistence): nothing is watchable until
the owner grants; an owner-verified grant persists ONLY to the machine-local
``config.local.json`` (sets ``watch.enabled``, preserves unrelated keys) and
never touches the committed ``config.json``; ``watch.start`` for an un-granted
app is refused (``not_granted``).
"""
from __future__ import annotations

import json

import pytest

from always_on_agent.capabilities import CapabilityRegistry
from always_on_agent.origin import Origin

from core.watch import GrantStore, WatchManager, attach_watch_capability


# --- tiny fakes ------------------------------------------------------------

class FakeSource:
    """A deterministic WatchSource: returns a fixed Observation (or None) and
    counts observe() calls. No real display."""

    def __init__(self, obs=None):
        self._obs = obs
        self.calls = 0

    def observe(self, app):
        self.calls += 1
        return self._obs


def _make_manager(store, *, source=None):
    """A WatchManager wired with no poller thread (autostart=False) so tests
    drive it purely through the explicit API -- no background timing."""
    published: list = []
    epoch = {"n": 7}
    mgr = WatchManager(
        store,
        source=source if source is not None else FakeSource(None),
        publish=published.append,
        current_epoch=lambda: epoch["n"],
        autostart=False,
    )
    return mgr, published


def _registry_with_watch(store, *, enabled=True, source=None):
    """Build a fresh registry + manager and attach the watch capability."""
    mgr, published = _make_manager(store, source=source)
    registry = CapabilityRegistry()
    attach_watch_capability(registry, {"enabled": enabled}, manager=mgr)
    return registry, mgr, published


def _owner_ctx(**extra):
    """A context that PASSES the owner-verify chokepoint: literal True +
    LIVE_AUDIO origin."""
    ctx = {"owner_verified": True, "origin": Origin.LIVE_AUDIO.value}
    ctx.update(extra)
    return ctx


def _grant_spec():
    return {"id": "term", "label": "Terminal", "app": {"wm_class": "gnome-terminal"}}


# --- INV-1: owner-verification chokepoint ----------------------------------

def test_grant_blocked_when_not_owner_verified(tmp_path):
    """No owner_verified flag (default-false) + unknown origin -> refused at the
    chokepoint; the store stays empty (nothing was granted)."""
    store = GrantStore(local_path=str(tmp_path / "config.local.json"))
    registry, mgr, _ = _registry_with_watch(store)

    # No owner_verified, no recognized origin: fail-closed.
    res = registry.invoke("watch.grant", "watch the terminal", {"grant": _grant_spec()})

    assert res.data["blocked"] == "owner_verification"
    assert res.data.get("executed") is False
    assert store.list() == ()  # default-deny: nothing entered the allowlist


def test_grant_blocked_when_origin_untrusted(tmp_path):
    """owner_verified=True but the turn arrived on the SCREEN (OCR) channel --
    an attacker-controllable origin. The action-trusted channel is LIVE_AUDIO
    only, so the grant is refused and the store stays empty (no trust
    laundering from screen text into a grant)."""
    store = GrantStore(local_path=str(tmp_path / "config.local.json"))
    registry, mgr, _ = _registry_with_watch(store)

    res = registry.invoke(
        "watch.grant",
        "watch the terminal",
        {"owner_verified": True, "origin": Origin.SCREEN.value, "grant": _grant_spec()},
    )

    assert res.data["blocked"] == "owner_verification"
    assert res.data.get("executed") is False
    assert store.list() == ()


def test_grant_blocked_when_owner_verified_is_truthy_not_true(tmp_path):
    """STRICT is-True: a truthy sentinel (``1``) is NOT the literal boolean
    ``True`` -- the chokepoint must reject it. This guards against an actuator
    handing the gate ``owner_verified=1`` and accidentally laundering trust."""
    store = GrantStore(local_path=str(tmp_path / "config.local.json"))
    registry, mgr, _ = _registry_with_watch(store)

    res = registry.invoke(
        "watch.grant",
        "watch the terminal",
        {"owner_verified": 1, "origin": Origin.LIVE_AUDIO.value, "grant": _grant_spec()},
    )

    assert res.data["blocked"] == "owner_verification"
    assert res.data.get("executed") is False
    assert store.list() == ()


def test_start_owner_verified_but_grant_not_in_store(tmp_path):
    """Owner-verified (passes INV-1) but the named grant was never added ->
    default-deny refusal (``not_granted``); no watch is armed."""
    store = GrantStore(local_path=str(tmp_path / "config.local.json"))
    registry, mgr, _ = _registry_with_watch(store)

    res = registry.invoke(
        "watch.start",
        "alert me on error",
        _owner_ctx(grant_id="ghost", condition="error", interval_sec=30),
    )

    assert res.data.get("executed") is False
    assert res.data["blocked"] == "not_granted"
    assert mgr.active() == ()  # nothing armed


# --- INV-7: default-off registration ---------------------------------------

def test_disabled_config_registers_no_watch_capabilities(tmp_path):
    """When ``watch_cfg['enabled']`` is False, attach registers NONE of
    watch.grant/start/stop/list -- the feature is wholly absent (default-OFF),
    so neither the owner nor (crucially) the model can reach it."""
    store = GrantStore(local_path=str(tmp_path / "config.local.json"))
    mgr, _ = _make_manager(store)
    registry = CapabilityRegistry()

    attach_watch_capability(registry, {"enabled": False}, manager=mgr)

    names = set(registry.names())
    for cap in ("watch.grant", "watch.start", "watch.stop", "watch.list"):
        assert cap not in names
    assert not any(n.startswith("watch.") for n in names)


# --- INV-7: machine-local persistence (never config.json) ------------------

def test_owner_verified_grant_persists_to_tmp_local_only(tmp_path):
    """An owner-verified grant writes ONLY to the tmp config.local.json: the
    file gains watch.grants with the id + watch.enabled=true, pre-existing
    unrelated keys survive, and the committed config.json is never created or
    touched."""
    local_path = tmp_path / "config.local.json"
    committed_path = tmp_path / "config.json"

    # A pre-existing machine-local file with an unrelated key + an unrelated
    # key nested under "watch" -- both must survive the grant write.
    local_path.write_text(
        json.dumps({"some_other_key": {"keep": 1}, "watch": {"unrelated": "x"}}),
        encoding="utf-8",
    )
    committed_before = json.dumps({"watch": {"enabled": False, "grants": []}})
    committed_path.write_text(committed_before, encoding="utf-8")

    store = GrantStore(local_path=str(local_path))
    registry, mgr, _ = _registry_with_watch(store)

    res = registry.invoke("watch.grant", "watch the terminal", _owner_ctx(grant=_grant_spec()))

    # The grant succeeded and landed in the in-memory allowlist.
    assert res.data.get("executed") is True
    assert res.data.get("grant_id") == "term"
    assert [g.id for g in store.list()] == ["term"]

    # The machine-local file now carries the grant + watch.enabled, and the
    # unrelated keys are preserved.
    written = json.loads(local_path.read_text(encoding="utf-8"))
    assert written["watch"]["enabled"] is True
    grant_ids = [g["id"] for g in written["watch"]["grants"]]
    assert grant_ids == ["term"]
    assert written["some_other_key"] == {"keep": 1}
    assert written["watch"]["unrelated"] == "x"  # other watch.* keys preserved

    # The committed config.json was never modified.
    assert committed_path.read_text(encoding="utf-8") == committed_before


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
