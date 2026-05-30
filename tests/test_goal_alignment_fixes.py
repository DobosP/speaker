"""Regression tests locking the 2026-05 goal-alignment fixes.

Each fix closes a finding from docs/perf_audit_2026-05_goal_alignment.md. These
tests pin the *committed* behavior so a silent revert (the exact class of bug
that caused several findings -- e.g. the router threshold living only in an
uncommitted working tree) is caught by CI.

- B1   : tier router escalates (config threshold 0.3 + generation markers)
- lat-1: true SPEECH_END backdating makes endpoint_latency visible
- lat-2: startup pre-warm (non-blocking, local-only, default off, fail-safe)
- dec-4: input_gate enabled on the active desktop profile
- dec-3: ReAct planner reachable in the shipped config
"""
from __future__ import annotations

import json
import pathlib
import threading
import time

import pytest

from core.config import apply_device_profile
from core.capabilities import DEFAULT_SYSTEM, _answers_locally
from core.engines.scripted import ScriptedEngine
from core.llm import HedgeLLM
from core.metrics import (
    ASR_FINAL,
    LLM_FIRST_TOKEN,
    MetricsRecorder,
    SPEECH_END,
    TTS_FIRST_AUDIO,
)
from core.routing import FAST, MAIN, build_router
from core.runtime import VoiceRuntime

# Read the COMMITTED config.json directly (not load_config, which would merge a
# machine-local config.local.json) so these pins reflect what ships / lands in CI.
_REPO = pathlib.Path(__file__).resolve().parents[1]
CONFIG = json.loads((_REPO / "config.json").read_text())


def _router_for(device: str):
    return build_router(apply_device_profile(CONFIG, device))


def _wait_until(pred, timeout: float = 2.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


class RecordingLLM:
    """Minimal LLMClient that records every prompt it is asked to answer."""

    def __init__(self, reply: str = "ok"):
        self._reply = reply
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def generate(self, prompt, *, system=None, images=None):
        with self._lock:
            self.calls.append(prompt)
        return self._reply

    def stream(self, prompt, *, system=None, images=None):
        with self._lock:
            self.calls.append(prompt)
        yield self._reply

    @property
    def call_count(self) -> int:
        with self._lock:
            return len(self.calls)


class RaisingLLM:
    """LLMClient whose generate() always fails -- to prove warm-up is fail-safe."""

    def __init__(self):
        self.attempts = 0
        self._lock = threading.Lock()

    def generate(self, prompt, *, system=None, images=None):
        with self._lock:
            self.attempts += 1
        raise RuntimeError("backend down")

    def stream(self, prompt, *, system=None, images=None):
        with self._lock:
            self.attempts += 1
        raise RuntimeError("backend down")
        yield ""  # pragma: no cover - unreachable, makes this a generator


# --------------------------------------------------------------------------- #
# B1 -- the tier router escalates instead of answering everything on the fast model
# --------------------------------------------------------------------------- #
def test_shipped_desktop_router_threshold_is_03():
    """The committed desktop profile escalates reasoning/long-form turns.

    Pins the config value: a revert to 0.5 (the old fast-only default that made
    125/125 field turns stay on the small model) would fail here.
    """
    r = _router_for("desktop")
    assert r.threshold == 0.3
    ctx = {"intent_kind": "assistant"}
    assert r.choose("explain how a transmission works", ctx) == MAIN
    assert r.choose("tell me a story", ctx) == MAIN
    assert r.choose("what time is it", ctx) == FAST


def test_phone_profile_does_not_over_escalate():
    """Phone's CPU-bound 4b 'main' tier must not catch idiom/borderline turns.

    The base threshold is 0.3, but phone overrides to 0.55 so 'give me a ...' /
    'tell me about it' (generation-marker idioms, score ~0.5) stay on the fast
    1b tier instead of offloading onto the slow CPU 4b.
    """
    r = _router_for("phone")
    assert r.threshold == 0.55
    ctx = {"intent_kind": "assistant"}
    assert r.choose("give me a minute", ctx) == FAST
    assert r.choose("tell me about it", ctx) == FAST


def test_default_system_prompt_self_generates():
    """The abstaining prompt tells the model to GENERATE long-form, not deflect."""
    assert "generate it yourself" in DEFAULT_SYSTEM


# --------------------------------------------------------------------------- #
# lat-1 -- true SPEECH_END so endpoint_latency stops reading ~0
# --------------------------------------------------------------------------- #
def test_metrics_speech_end_backdating_makes_endpoint_visible():
    clock = {"t": 100.0}
    rec = MetricsRecorder(clock=lambda: clock["t"])
    # SPEECH_END stamped at the true silence onset (an earlier perf_counter)...
    rec.mark(SPEECH_END, at=100.0)
    clock["t"] = 100.8  # ...the endpointer fires ~0.8s later
    rec.mark(ASR_FINAL)
    clock["t"] = 101.0
    rec.mark(LLM_FIRST_TOKEN)
    clock["t"] = 101.2
    rec.mark(TTS_FIRST_AUDIO)
    rec.close_turn()
    r = rec.records()[0]
    assert r.endpoint_latency == pytest.approx(0.8)
    # first_audio is now measured from SPEECH_END, so it INCLUDES the endpoint.
    assert r.first_audio_latency == pytest.approx(1.2)


def test_metrics_speech_end_at_none_stamps_now():
    """at=None preserves the old behavior (stamp the current clock)."""
    clock = {"t": 50.0}
    rec = MetricsRecorder(clock=lambda: clock["t"])
    rec.mark(SPEECH_END, at=None)
    rec.close_turn()
    assert rec.records()[0].stamps[SPEECH_END] == 50.0


# --------------------------------------------------------------------------- #
# lat-2 -- startup pre-warm: non-blocking, local-only, default off, fail-safe
# --------------------------------------------------------------------------- #
def test_warm_on_start_warms_each_local_model_once():
    main, fast = RecordingLLM(), RecordingLLM()
    rt = VoiceRuntime(ScriptedEngine(), main, fast_llm=fast, warm_on_start=True)
    rt.start(run_bus=False)
    try:
        assert _wait_until(lambda: main.call_count >= 1 and fast.call_count >= 1)
        time.sleep(0.1)  # allow any erroneous extra call to land
        assert main.calls == ["hi"]
        assert fast.calls == ["hi"]
    finally:
        rt.stop()


def test_warm_collapsed_fast_main_warmed_once():
    shared = RecordingLLM()
    rt = VoiceRuntime(ScriptedEngine(), shared, fast_llm=shared, warm_on_start=True)
    rt.start(run_bus=False)
    try:
        assert _wait_until(lambda: shared.call_count >= 1)
        time.sleep(0.1)
        assert shared.calls == ["hi"]  # identity-deduped -> exactly once
    finally:
        rt.stop()


def test_warm_default_off_spins_no_warm():
    main, fast = RecordingLLM(), RecordingLLM()
    rt = VoiceRuntime(ScriptedEngine(), main, fast_llm=fast)  # default: no warm
    rt.start(run_bus=False)
    try:
        time.sleep(0.15)
        assert main.call_count == 0 and fast.call_count == 0
    finally:
        rt.stop()


def test_warm_failure_is_swallowed():
    bad = RaisingLLM()
    rt = VoiceRuntime(ScriptedEngine(), bad, warm_on_start=True)
    rt.start(run_bus=False)  # must not raise even though warm-up throws
    try:
        assert _wait_until(lambda: bad.attempts >= 1)
        assert rt.mode is not None  # pipeline still alive
    finally:
        rt.stop()


def test_warm_skips_cloud_leg_but_warms_local_leg():
    """§9.7: a cloud-backed main must NOT be warm-called as a whole (no egress
    until invoked) -- but its purely-LOCAL leg IS warmed, so a cloud-hybrid's
    local tier isn't left cold on turn 1."""
    local_fast = RecordingLLM()
    cloud = RecordingLLM()
    hedge_local = RecordingLLM()
    hedged_main = HedgeLLM(local=hedge_local, cloud=cloud)
    assert not _answers_locally(hedged_main)  # predicate sanity
    rt = VoiceRuntime(ScriptedEngine(), hedged_main, fast_llm=local_fast, warm_on_start=True)
    rt.start(run_bus=False)
    try:
        assert _wait_until(lambda: local_fast.call_count >= 1 and hedge_local.call_count >= 1)
        time.sleep(0.05)
        assert cloud.call_count == 0              # cloud never touched at warm (no egress)
        assert hedged_main not in rt._warm_models  # the wrapper itself isn't warmed...
        assert local_fast in rt._warm_models       # ...but the local fast tier is...
        assert hedge_local in rt._warm_models      # ...and so is the hedge's local leg
    finally:
        rt.stop()


def test_warm_on_start_kwarg_accepted():
    """The exact construction shape tools/stress.py scn_real uses (used to TypeError)."""
    rt = VoiceRuntime(ScriptedEngine(), RecordingLLM(), fast_llm=RecordingLLM(), warm_on_start=True)
    rt.stop()


# --------------------------------------------------------------------------- #
# dec-4 / dec-3 -- the gates are actually ON in the shipped config
# --------------------------------------------------------------------------- #
def test_input_gate_enabled_on_active_desktop_profile():
    merged = apply_device_profile(CONFIG, "desktop")
    assert merged.get("input_gate", {}).get("enabled") is True


def test_react_planner_enabled_in_shipped_config():
    assert CONFIG.get("agent", {}).get("planner", {}).get("enabled") is True


def test_continuation_enabled_in_shipped_config():
    # ADD-ON / continuation ships ON: a follow-up merges into the in-flight turn
    # instead of racing a competing cold task. from_dict defaults off, so this
    # pin guards against the block being dropped / silently disabled.
    from always_on_agent.continuation import ContinuationConfig

    block = CONFIG.get("continuation")
    assert isinstance(block, dict) and block.get("enabled") is True
    assert ContinuationConfig.from_dict(block).enabled is True
