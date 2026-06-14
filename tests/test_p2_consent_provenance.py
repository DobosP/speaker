"""P2: cloud consent (no silent activation), egress receipt (HedgeLLM winning
source), and ACT containment (state-changing tools stay off the planner). Tier-0.
"""
from __future__ import annotations

import json
import os
from typing import Iterator

from core.llm import HedgeLLM, LLMClient


class _FakeLLM(LLMClient):
    def __init__(self, tokens):
        self.tokens = list(tokens)

    def generate(self, prompt, *, system=None, images=None) -> str:
        return "".join(self.tokens)

    def stream(self, prompt, *, system=None, images=None) -> Iterator[str]:
        for t in self.tokens:
            yield t


# --- egress receipt: HedgeLLM.last_source -----------------------------------

def test_last_source_none_before_any_call():
    assert HedgeLLM(local=_FakeLLM(["x"]), cloud=[]).last_source is None


def test_last_source_local_when_no_cloud():
    h = HedgeLLM(local=_FakeLLM(["hi"]), cloud=[])
    assert "".join(h.stream("q")) == "hi"
    assert h.last_source == "local"


def test_last_source_cloud_when_cloud_wins():
    # local empty (no tokens) so the cloud member serves the turn.
    h = HedgeLLM(local=_FakeLLM([]), cloud=[_FakeLLM(["cloud answer"])], hedge_delay_ms=0)
    out = "".join(h.stream("q"))
    assert out == "cloud answer"
    assert h.last_source == "cloud_0"


# --- consent: no profile silently auto-enables cloud -------------------------

def _config():
    return json.load(open(os.path.join(os.path.dirname(__file__), "..", "config.json")))


def test_base_config_cloud_disabled():
    cfg = _config()
    assert cfg["llm"].get("cloud", {}).get("enabled", False) is False


def test_no_device_profile_auto_enables_cloud():
    # SECURITY: cloud must be enabled DELIBERATELY, never shipped enabled=true on a
    # profile (which would activate on mere API-key presence).
    cfg = _config()
    on = []
    for name, prof in cfg.get("device_profiles", {}).items():
        cloud = (prof.get("llm") or {}).get("cloud")
        if isinstance(cloud, dict) and cloud.get("enabled"):
            on.append(name)
    assert not on, f"device profile(s) ship cloud enabled (silent activation): {on}"


# --- ACT containment: state-changing tools stay off the planner -------------

def test_planner_default_tools_are_all_read_only():
    from always_on_agent.capabilities import create_default_capabilities
    from always_on_agent.react import DEFAULT_TOOLS

    reg = create_default_capabilities()
    for name in DEFAULT_TOOLS:
        spec = reg.spec(name)
        # A planner gather-tool must never be side-effecting (ACT routes only to
        # read-only tools in v1; a state-changing tool here would let an escalated
        # turn -- shaped by untrusted observations -- trigger an action).
        assert spec is not None, f"planner tool {name} has no spec"
        assert spec.side_effecting is False, f"planner DEFAULT_TOOL {name} is side-effecting"


def test_no_side_effecting_capability_is_a_planner_tool():
    # Structural quarantine: a side-effecting capability must never be exposed to
    # the ReAct planner (planner_tool=True), so untrusted planner text can't select
    # an action tool.
    from always_on_agent.capabilities import create_default_capabilities

    reg = create_default_capabilities()
    offenders = [s.name for s in reg.manifest() if s.side_effecting and s.planner_tool]
    assert not offenders, f"side-effecting capabilities exposed to the planner: {offenders}"
