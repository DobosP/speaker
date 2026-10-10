"""Foreground input owns inference priority over cancellable idle startup work."""
from __future__ import annotations

import threading

import pytest

from always_on_agent.models import CLOUD_EGRESS_SCOPE_CONTEXT_KEY, CloudEgressScope
from core.engines.scripted import ScriptedEngine
from core.llm import EchoLLM, LLMCallCancelled, OpenAICompatLLM, capability_context
from core.runtime import VoiceRuntime


class CooperativeModel:
    def __init__(self):
        self.entered = threading.Event()
        self.cancelled = threading.Event()
        self.real_requested = threading.Event()
        self.real_entered = threading.Event()
        self.closed = threading.Event()
        self.release = threading.Event()
        self.lock = threading.Lock()
        self.cancel_context = None
    def generate(self, *args, **kwargs):
        raise AssertionError("warm must use cancellable streaming")
    def stream(self, prompt, **kwargs):
        if prompt != "hi":
            self.real_requested.set()
            with self.lock:
                self.real_entered.set()
                yield "A public synthetic answer."
            return
        with self.lock:
            self.cancel_context = capability_context.get()["cancel_event"]
            self.entered.set()
            try:
                while not self.release.wait(0.001):
                    if self.cancel_context.is_set():
                        self.cancelled.set()
                        raise LLMCallCancelled("synthetic warm retired")
                yield "Warm completed."
            finally:
                self.closed.set()


class Gate:
    def __init__(self):
        self.calls = []
    def classify(self, text, *, recent=()):
        self.calls.append(text)
        return "INGEST"


def test_accepted_final_retires_warm_and_foreground_uses_same_model_after_cleanup():
    model = CooperativeModel()
    runtime = VoiceRuntime(ScriptedEngine(), model, warm_on_start=True)
    runtime.start(run_bus=True)
    try:
        assert model.entered.wait(1)
        runtime._on_final("What color is a robin egg?")
        assert model.cancelled.wait(1)
        assert model.closed.wait(1)
        assert model.real_entered.wait(1)
        assert runtime.warm_ready.wait(1)
    finally:
        model.release.set()
        runtime.stop()


def test_stop_cancels_warm_and_never_starts_successor_helper():
    model, gate = CooperativeModel(), Gate()
    runtime = VoiceRuntime(ScriptedEngine(), model, addressing=gate, warm_on_start=True)
    runtime.start(run_bus=False)
    try:
        assert model.entered.wait(1)
        runtime.stop()
        assert model.cancelled.wait(1) and model.closed.wait(1)
        assert runtime.warm_ready.wait(1)
        assert gate.calls == []
    finally:
        model.release.set()
        runtime.stop()


def test_uncancellable_legacy_call_cannot_start_helper_after_stop_or_fake_readiness():
    entered, release = threading.Event(), threading.Event()
    class Legacy:
        def generate(self, prompt, *, system):
            entered.set()
            assert release.wait(2)
            return "done"
    gate = Gate()
    runtime = VoiceRuntime(ScriptedEngine(), Legacy(), addressing=gate, warm_on_start=True)
    runtime.start(run_bus=False)
    try:
        assert entered.wait(1)
        runtime.stop()
        assert not runtime.warm_ready.is_set()  # cancellation is not true source terminal
        release.set()
        assert runtime.warm_ready.wait(1)
        assert gate.calls == []
    finally:
        release.set()
        runtime.stop()


def test_input_during_media_warm_skips_later_models_and_helpers():
    entered, release = threading.Event(), threading.Event()
    class Engine(ScriptedEngine):
        def warm(self):
            entered.set()
            assert release.wait(2)
    model, gate = CooperativeModel(), Gate()
    runtime = VoiceRuntime(Engine(), model, addressing=gate, warm_on_start=True)
    runtime.start(run_bus=False)
    try:
        assert entered.wait(1)
        runtime._new_input_generation()
        release.set()
        assert runtime.warm_ready.wait(1)
        assert not model.entered.is_set() and gate.calls == []
    finally:
        release.set()
        runtime.stop()


def test_final_rejected_at_punctuation_floor_does_not_retire_idle_warm():
    model = CooperativeModel()
    runtime = VoiceRuntime(ScriptedEngine(), model, warm_on_start=True)
    runtime.start(run_bus=False)
    try:
        assert model.entered.wait(1)
        runtime._on_final("...")
        assert not runtime._warm_cancel.is_set()
    finally:
        model.release.set()
        runtime.stop()


def test_accepted_partial_generation_retires_warm_without_waiting_on_source():
    model = CooperativeModel()
    runtime = VoiceRuntime(ScriptedEngine(), model, warm_on_start=True)
    runtime.start(run_bus=False)
    try:
        assert model.entered.wait(1)
        runtime._on_partial("What color is a robin egg")
        assert runtime._warm_cancel.is_set()
        assert model.cancelled.wait(1)
    finally:
        model.release.set()
        runtime.stop()


def test_pre_start_accepted_input_prevents_speculative_work():
    calls = []
    class Engine(ScriptedEngine):
        def warm(self):
            calls.append("media")
    runtime = VoiceRuntime(Engine(), CooperativeModel())
    runtime._new_input_generation()
    runtime._warm()
    assert calls == [] and runtime.warm_ready.is_set()


def test_entered_legacy_typeerror_is_never_retried():
    calls = []
    class Legacy:
        def generate(self, prompt, *, system=None):
            calls.append(system)
            raise TypeError("synthetic entered failure")
    runtime = VoiceRuntime(ScriptedEngine(), Legacy())
    runtime._warm()
    assert len(calls) == 1 and calls[0] == runtime._system_prompt
    assert runtime.warm_ready.is_set()


def test_warm_scope_narrows_egress_and_preserves_inherited_cancellation():
    external = threading.Event()
    external.set()
    inherited = {"cancel_event": external, CLOUD_EGRESS_SCOPE_CONTEXT_KEY: CloudEgressScope.LOCAL_ONLY,
                 "marker": "public synthetic context"}
    token = capability_context.set(inherited)
    model = CooperativeModel()
    try:
        runtime = VoiceRuntime(ScriptedEngine(), model)
        runtime._warm()
        assert not model.entered.is_set()
        assert capability_context.get() is inherited
    finally:
        capability_context.reset(token)


def test_helpers_are_local_only_and_retirement_between_them_skips_cleaner():
    class Addressing:
        def classify(self, text, *, recent=()):
            context = capability_context.get()
            assert context[CLOUD_EGRESS_SCOPE_CONTEXT_KEY] is CloudEgressScope.LOCAL_ONLY
            runtime._new_input_generation()
            return "INGEST"
    class Cleaner:
        def clean(self, text, *, recent=()):
            pytest.fail("retired plan cannot start the next helper")
    runtime = VoiceRuntime(ScriptedEngine(), EchoLLM(), addressing=Addressing(), cleaner=Cleaner())
    runtime._warm()
    assert runtime.warm_ready.is_set()


def test_direct_cloud_model_and_declared_cloud_helper_are_not_warmed(monkeypatch):
    cloud = OpenAICompatLLM(model="public-model", api_key="public-fake-key", base_url="https://example.invalid")
    monkeypatch.setattr(cloud, "generate", lambda *a, **k: pytest.fail("zero cloud transport"))
    monkeypatch.setattr(cloud, "stream", lambda *a, **k: pytest.fail("zero cloud transport"))
    class Helper:
        _llm = cloud
        def classify(self, *a, **k):
            pytest.fail("cloud-owned helper cannot warm")
        clean = classify
    runtime = VoiceRuntime(ScriptedEngine(), cloud, addressing=Helper(), cleaner=Helper())
    runtime._warm()
    assert runtime._warm_models == [] and runtime.warm_ready.is_set()


def test_ollama_warm_cancels_before_first_token_and_same_client_remains_usable():
    from core.llm import OllamaLLM, collect_llm_text
    from tests.test_ollama_async_cancel import FakeAsyncChunks, FakeAsyncClientFactory, FakeSyncClient
    warm = FakeAsyncChunks(block_after_chunks=True)
    real = FakeAsyncChunks("A public answer.")
    factory = FakeAsyncClientFactory(warm, real)
    sync = FakeSyncClient()
    options = {"num_ctx": 2048, "num_predict": 256, "num_thread": 2}
    model = OllamaLLM("public-local-model", client=sync, async_client_factory=factory,
                      options=options, keep_alive="60s", think=False)
    runtime = VoiceRuntime(ScriptedEngine(), model, warm_on_start=True)
    runtime.start(run_bus=False)
    try:
        assert warm.waiting.wait(1)
        runtime._new_input_generation()
        assert warm.task_cancelled.wait(1)
        assert runtime.warm_ready.wait(1)
        assert warm.closed.is_set() and factory.clients[0].closed.is_set()
        assert collect_llm_text(model, "A public question.") == "A public answer."
        assert sync.calls == []  # no uncancellable generate() warm transport
        assert len(factory.clients) == 2
        assert all(client.calls[0]["options"] == options for client in factory.clients)
        assert all(client.calls[0]["keep_alive"] == "60s" for client in factory.clients)
    finally:
        runtime.stop()


def test_retirement_between_local_models_skips_unused_main_without_disabling_it():
    class Fast:
        def stream(self, *a, **k):
            runtime._new_input_generation()
            yield "finished concurrently with input"
    class Main:
        def __init__(self):
            self.calls = 0
        def generate(self, *a, **k):
            self.calls += 1
            return "Public native-free main answer"
    main = Main()
    runtime = VoiceRuntime(ScriptedEngine(), main, fast_llm=Fast())
    runtime._warm()
    assert main.calls == 0
    assert main.generate("Public on-demand main request") == "Public native-free main answer"
    assert main.calls == 1


def test_warm_context_snapshot_never_invokes_foreign_mapping_hooks():
    class Hostile(dict):
        def __iter__(self):
            pytest.fail("foreign mapping must not be iterated")
        def keys(self):
            pytest.fail("foreign mapping must not be inspected")
    class Model:
        def stream(self, *args, **kwargs):
            assert capability_context.get()[CLOUD_EGRESS_SCOPE_CONTEXT_KEY] is CloudEgressScope.LOCAL_ONLY
            yield "public warm result"
    original = Hostile()
    token = capability_context.set(original)
    try:
        runtime = VoiceRuntime(ScriptedEngine(), Model())
        runtime._warm()
        assert runtime.warm_ready.is_set()
        assert capability_context.get() is original
    finally:
        capability_context.reset(token)
