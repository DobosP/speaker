"""Guardrail for device-adapt-1: auto-profile-selection must stay cloud-safe.

Wiring the hardware probe into startup (``core/app.py`` resolves ``device:
"auto"`` -> ``tools.recommend_profile``) is only safe while NO ``device_profile``
can flip the runtime to a less-private posture. The architecture-audit security
review (docs/architecture_audit_2026-06-16.md) made that a hard precondition;
this test pins it in CI:

  * no profile enables the cloud tier, PRC presets, the actuator, watch/monitor,
    or web egress, and none disables ``local_only``;
  * every profile ``recommend()`` can return actually EXISTS (else auto-detect
    would crash on ``apply_device_profile(strict=True)``);
  * the resolve / strict-apply contract (cross-platform-8) behaves: ``"auto"``
    probes a real profile, an unknown ``--device`` fails fast, and the
    non-strict path keeps its historical silent no-op.

Tier 0: reads the COMMITTED config.json template + pure config transforms. No
audio, no models, no network (the probe is stdlib hardware counters only).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.config import apply_device_profile, deep_merge, resolve_device
from tools.recommend_profile import HostInfo, recommend

_REPO = Path(__file__).resolve().parents[1]


def _config_template() -> dict:
    """The committed config.json (NO machine-local overlay): what actually ships."""
    return json.loads((_REPO / "config.json").read_text())


def _profiles() -> dict:
    return _config_template().get("device_profiles", {})


def _merged(name: str) -> dict:
    cfg = _config_template()
    return deep_merge(cfg, cfg["device_profiles"][name])


@pytest.mark.parametrize("name", list(_profiles()))
def test_profile_never_enables_cloud_or_relaxes_owner_gates(name):
    """The cloud/owner-gate invariant, per profile (the device-adapt-1 guardrail)."""
    merged = _merged(name)

    # §9.7: raw-audio-stays-local posture must not be relaxed by a profile.
    assert merged.get("local_only", True) is not False, (
        f"{name} disables local_only"
    )
    # The single-cloud back-compat tier stays OFF and PRC presets opt-in.
    cloud = (merged.get("llm", {}) or {}).get("cloud", {}) or {}
    assert cloud.get("enabled", False) is not True, f"{name} enables llm.cloud"
    assert cloud.get("allow_prc", False) is not True, f"{name} enables PRC presets"
    # Actuator + observe-and-act + egress gates stay closed (owner-gated default-deny).
    assert (merged.get("gui_actions", {}) or {}).get("enabled", False) is not True, (
        f"{name} enables gui_actions (actuator)"
    )
    assert (merged.get("watch", {}) or {}).get("enabled", False) is not True, (
        f"{name} enables watch/monitor"
    )
    assert (merged.get("web_search", {}) or {}).get("enabled", False) is not True, (
        f"{name} enables web_search egress"
    )
    # The voice-identity gate and the owner-verified brain gate stay closed: a
    # profile retunes models/threads, it must never widen WHO the assistant obeys.
    assert (merged.get("sherpa", {}) or {}).get("speaker_gate_input", True) is not False, (
        f"{name} disables speaker_gate_input (the voice-identity gate)"
    )
    assert (merged.get("agent_brain", {}) or {}).get("require_owner_verified", True) is not False, (
        f"{name} disables agent_brain.require_owner_verified"
    )


def test_profiles_do_not_touch_cloud_routing_blocks():
    """Profiles retune llm models/threads only; they must not switch on the
    sensitivity-routed cloud_providers/cloud_chains (those stay base-defined)."""
    for name, prof in _profiles().items():
        llm = prof.get("llm", {}) or {}
        assert "cloud_providers" not in llm, f"{name} overrides cloud_providers"
        assert "cloud_chains" not in llm, f"{name} overrides cloud_chains"


# Representative hosts covering every branch of recommend()'s decision tree.
_REPRESENTATIVE_HOSTS = [
    HostInfo(cores=16, ram_gb=64, gpu_kind="nvidia", gpu_mem_gb=24),
    HostInfo(cores=8, ram_gb=32, gpu_kind="nvidia", gpu_mem_gb=12),
    HostInfo(cores=8, ram_gb=16, gpu_kind="nvidia", gpu_mem_gb=8),
    HostInfo(cores=10, ram_gb=16, gpu_kind="apple", gpu_mem_gb=16),
    HostInfo(cores=8, ram_gb=8, gpu_kind=None, gpu_mem_gb=0.0, mobile=True),
    HostInfo(cores=8, ram_gb=6, gpu_kind=None, gpu_mem_gb=0.0, mobile=True),
    HostInfo(cores=8, ram_gb=16, gpu_kind=None, gpu_mem_gb=0.0),
    HostInfo(cores=4, ram_gb=4, gpu_kind=None, gpu_mem_gb=0.0),
    HostInfo(cores=2, ram_gb=2, gpu_kind=None, gpu_mem_gb=0.0),
]


def test_every_recommended_profile_exists():
    """Auto-detection can only ever pick a profile that exists in config.json."""
    profiles = _profiles()
    for host in _REPRESENTATIVE_HOSTS:
        name, _ = recommend(host)
        assert name in profiles, f"recommend() -> unknown profile {name!r} for {host}"


def test_resolve_device_concrete_is_passthrough():
    assert resolve_device(_config_template(), "phone") == ("phone", None)


def test_resolve_device_auto_probes_a_valid_profile():
    name, rationale = resolve_device(_config_template(), "auto")
    assert name in _profiles()
    assert rationale, "auto-detection must return a rationale for the log"


def test_apply_strict_fails_fast_on_unknown_device():
    with pytest.raises(ValueError):
        apply_device_profile(_config_template(), "no-such-profile", strict=True)


def test_apply_nonstrict_preserves_silent_noop():
    """Back-compat: existing callers that pass a non-profile key (e.g. 'watch')
    still get the input unchanged when not strict."""
    cfg = _config_template()
    assert apply_device_profile(cfg, "no-such-profile") is cfg


def test_apply_auto_resolves_and_applies_a_real_profile():
    cfg = _config_template()
    merged = apply_device_profile(cfg, "auto")
    assert merged is not cfg, "auto must resolve+apply a profile, not no-op"
