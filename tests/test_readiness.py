"""Tests for startup readiness / pre-warm (ask #3: loaded & ready to fire).

The runtime pre-warms the answering models (with the REAL system prompt so the
cacheable prefix is filled), the input gate + cleaner, and the engine (a real
``warm()`` -- previously dead code), then raises ``warm_ready``. These pin the
WIRING (no models needed); the actual cold-vs-warm latency is a bench concern.
"""
from __future__ import annotations

import threading

from always_on_agent.events import Mode

from core.addressing import ACT, ScriptedAddressingClassifier
from core.cleanup import ScriptedTranscriptCleaner
from core.engines.scripted import ScriptedEngine
from core.llm import EchoLLM
from core.runtime import VoiceRuntime


class _WarmRecordingLLM:
    """Records the system prompt each generate()/stream() was warmed/called with."""

    def __init__(self):
        self._lock = threading.Lock()
        self.generate_systems: list = []

    def generate(self, prompt, *, system=None, images=None):
        with self._lock:
            self.generate_systems.append(system)
        return "ok"

    def stream(self, prompt, *, system=None, images=None):
        yield "ok"


def test_audioengine_warm_is_optional_noop():
    # Base engines (and the scripted test engine) get a no-op warm().
    ScriptedEngine().warm()  # must not raise


def test_sherpa_warm_is_noop_without_a_tts_model():
    # The real engine's warm() is guarded: no TTS built -> no-op, no crash.
    from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine

    SherpaOnnxEngine(SherpaConfig()).warm()


def test_warm_ready_set_immediately_when_warm_disabled():
    runtime = VoiceRuntime(ScriptedEngine(), EchoLLM(), warm_on_start=False)
    runtime.start(run_bus=False)
    try:
        assert runtime.warm_ready.is_set()
    finally:
        runtime.stop()


def test_warm_uses_the_real_system_prompt():
    llm = _WarmRecordingLLM()
    runtime = VoiceRuntime(
        ScriptedEngine(), llm, start_mode=Mode.ASSISTANT, warm_on_start=True
    )
    runtime.start(run_bus=False)
    try:
        assert runtime.warm_ready.wait(timeout=3.0)
        # The pre-warm prefilled the model with the capability-aware system
        # prompt, not a bare "hi" with no system.
        assert any(
            s and "on-device voice assistant" in s for s in llm.generate_systems
        ), llm.generate_systems
    finally:
        runtime.stop()


def test_engine_warm_is_invoked_during_prewarm():
    class _WarmEngine(ScriptedEngine):
        def __init__(self):
            super().__init__()
            self.warmed = threading.Event()

        def warm(self):
            self.warmed.set()

    engine = _WarmEngine()
    runtime = VoiceRuntime(engine, EchoLLM(), warm_on_start=True)
    runtime.start(run_bus=False)
    try:
        assert runtime.warm_ready.wait(timeout=3.0)
        assert engine.warmed.is_set()
    finally:
        runtime.stop()


def test_input_gate_and_cleaner_are_warmed():
    addressing = ScriptedAddressingClassifier(default=ACT)
    cleaner = ScriptedTranscriptCleaner()
    runtime = VoiceRuntime(
        ScriptedEngine(), EchoLLM(), warm_on_start=True,
        addressing=addressing, cleaner=cleaner,
    )
    runtime.start(run_bus=False)
    try:
        assert runtime.warm_ready.wait(timeout=3.0)
        assert ("hi", ()) in addressing.calls
        assert ("hi", ()) in cleaner.calls
    finally:
        runtime.stop()


def test_warm_ready_set_even_if_a_warm_step_raises():
    class _BoomLLM:
        def generate(self, prompt, *, system=None, images=None):
            raise RuntimeError("model not pulled")

        def stream(self, prompt, *, system=None, images=None):
            yield "x"

    runtime = VoiceRuntime(ScriptedEngine(), _BoomLLM(), warm_on_start=True)
    runtime.start(run_bus=False)
    try:
        # A failing warm step must still flip readiness -- "finished", not "perfect".
        assert runtime.warm_ready.wait(timeout=3.0)
    finally:
        runtime.stop()


def test_cloud_hybrid_warms_only_the_local_leg():
    # A cloud-backed HedgeLLM is not warmed whole (would egress before a real
    # turn), but its purely-local leg IS warmed so the local tier isn't cold.
    from core.llm import HedgeLLM

    class _Rec:
        def __init__(self):
            self.calls: list = []

        def generate(self, prompt, *, system=None, images=None):
            self.calls.append(system)
            return "ok"

        def stream(self, prompt, *, system=None, images=None):
            yield "ok"

    local, cloud = _Rec(), _Rec()
    hedge = HedgeLLM(local=local, cloud=cloud)
    runtime = VoiceRuntime(ScriptedEngine(), hedge, warm_on_start=True)
    runtime.start(run_bus=False)
    try:
        assert runtime.warm_ready.wait(timeout=3.0)
        assert local.calls, "local leg of the cloud-hybrid was not warmed"
        assert cloud.calls == [], "cloud leg must not be warmed before a real turn"
    finally:
        runtime.stop()


def test_llamacpp_serializes_inference_on_the_shared_context():
    # The single llama.cpp context can't run two inferences at once; the lock
    # serializes the warm pass and a concurrent live call.
    import time

    from core.llm import LlamaCppLLM

    state = {"active": 0, "max": 0}
    guard = threading.Lock()

    class _FakeClient:
        def create_chat_completion(self, messages, stream=False, **kw):
            with guard:
                state["active"] += 1
                state["max"] = max(state["max"], state["active"])
            time.sleep(0.02)
            with guard:
                state["active"] -= 1
            if stream:
                return iter([{"choices": [{"delta": {"content": "x"}}]}])
            return {"choices": [{"message": {"content": "ok"}}]}

    llm = LlamaCppLLM("fake.gguf", client=_FakeClient())
    threads = [threading.Thread(target=lambda: llm.generate("hi")) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert state["max"] == 1, f"inference was not serialized (max concurrency {state['max']})"
