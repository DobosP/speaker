"""Tests that capability_context (a ContextVar) is set + reset correctly
under concurrency, exceptions, and back-to-back calls.

ContextVar bugs are subtle: a leaked value persists across unrelated turns
or threads and silently mis-routes the next request. These tests pin the
guarantees the SensitivityRouterLLM relies on:

1. The ContextVar reads the value set most recently in the current frame.
2. Concurrent threads each see their own value (no cross-contamination).
3. An exception inside a capability still resets the ContextVar.
4. Default value is an empty mapping (so a missing setter doesn't crash
   ChainSelector with a NoneType).
"""
from __future__ import annotations

import threading
import time

import pytest

from always_on_agent.events import Mode
from always_on_agent.models import IntentKind

from core.capabilities import attach_llm_capabilities
from core.engines.scripted import ScriptedEngine
from core.llm import EchoLLM, SensitivityRouterLLM, capability_context
from core.routing import ChainSelector
from core.runtime import VoiceRuntime


# --- direct ContextVar semantics ------------------------------------------


def test_default_is_empty_mapping():
    """Before any setter runs, capability_context.get() must return an
    empty mapping (not None) -- ChainSelector reads .get('sensitivity', ...)
    on whatever this returns."""
    val = capability_context.get()
    assert val == {}
    assert hasattr(val, "get")  # mapping protocol


def test_set_and_reset_restores_previous_value():
    """Standard ContextVar lifecycle: set returns a token, reset restores."""
    original = capability_context.get()
    token = capability_context.set({"sensitivity": "code"})
    try:
        assert capability_context.get() == {"sensitivity": "code"}
    finally:
        capability_context.reset(token)
    assert capability_context.get() == original


def test_nested_set_unwinds_lifo():
    """Two nested sets unwind in LIFO order via paired reset tokens."""
    t1 = capability_context.set({"sensitivity": "private"})
    try:
        t2 = capability_context.set({"sensitivity": "public"})
        try:
            assert capability_context.get()["sensitivity"] == "public"
        finally:
            capability_context.reset(t2)
        assert capability_context.get()["sensitivity"] == "private"
    finally:
        capability_context.reset(t1)


# --- threaded isolation ---------------------------------------------------


def test_concurrent_threads_do_not_share_context():
    """Three threads set different sensitivities at the same time; each
    must observe its OWN value, not another thread's. Critical for the
    brain's multi-task supervisor -- two cancellable tasks may overlap."""
    barrier = threading.Barrier(3)
    observed: dict[str, list] = {"a": [], "b": [], "c": []}

    def worker(label: str, value: str):
        barrier.wait()
        # Each thread inherits a COPY of the parent context (ContextVar
        # semantics). Setting only affects this thread's copy.
        capability_context.set({"sensitivity": value})
        # Yield to other threads to maximize chance of cross-contamination.
        time.sleep(0.01)
        observed[label].append(capability_context.get().get("sensitivity"))

    threads = [
        threading.Thread(target=worker, args=("a", "private")),
        threading.Thread(target=worker, args=("b", "code")),
        threading.Thread(target=worker, args=("c", "public")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert observed["a"] == ["private"]
    assert observed["b"] == ["code"]
    assert observed["c"] == ["public"]


def test_router_picks_chain_per_thread():
    """End-to-end: each thread's SensitivityRouterLLM.stream call sees
    only its own thread's context, not the others'."""
    chains = {
        "private": EchoLLM(reply="P"),
        "code": EchoLLM(reply="C"),
        "public": EchoLLM(reply="U"),
    }
    selector = ChainSelector(
        {"private": "private", "code": "code", "public": "public"},
        default_chain="private",
    )
    router_llm = SensitivityRouterLLM(chains, selector=selector, default_chain="private")

    results: dict[str, str] = {}
    barrier = threading.Barrier(3)

    def run(label: str, sens: str):
        barrier.wait()
        capability_context.set({"sensitivity": sens})
        time.sleep(0.01)
        results[label] = "".join(router_llm.stream("hi"))

    threads = [
        threading.Thread(target=run, args=("a", "private")),
        threading.Thread(target=run, args=("b", "code")),
        threading.Thread(target=run, args=("c", "public")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results == {"a": "P", "b": "C", "c": "U"}


# --- exception-safety -----------------------------------------------------


class _ExplodingLLM:
    def generate(self, prompt, *, system=None, images=None) -> str:
        raise RuntimeError("kaboom")

    def stream(self, prompt, *, system=None, images=None):
        raise RuntimeError("kaboom")
        yield ""  # unreachable but keeps it a generator


def test_capability_resets_context_even_when_llm_raises():
    """If the backing LLM raises mid-call, capability_context must still
    be reset so the next turn doesn't inherit the failed turn's tag."""
    # Use a SensitivityRouterLLM pointed at an exploding backend so the
    # capability layer's try/finally is exercised.
    chains = {"private": _ExplodingLLM()}
    selector = ChainSelector({"private": "private"}, default_chain="private")
    router_llm = SensitivityRouterLLM(chains, selector=selector, default_chain="private")

    engine = ScriptedEngine()
    runtime = VoiceRuntime(engine, router_llm, start_mode=Mode.ASSISTANT)
    runtime.start(run_bus=False)
    try:
        engine.final("read me my notes")  # private -> exploding LLM
        runtime.wait_idle(timeout=5.0)
        # No assertion about success/failure here -- just isolation.
        assert capability_context.get() == {}, (
            "context leaked after the LLM raised inside the capability"
        )
    finally:
        runtime.stop()


# --- intent flow ----------------------------------------------------------


def test_intent_kind_flows_from_task_to_capability_context():
    """The task layer forwards task.intent.value into context; the
    capability layer then enriches that with sensitivity. The test
    captures the actual context the LLM was called with."""
    # Capture whatever context capability_context.get() yields when the
    # LLM is invoked, via a recording backend.
    seen_ctx: dict[str, object] = {}

    class _CaptureLLM:
        def generate(self, prompt, *, system=None, images=None) -> str:
            seen_ctx.update(capability_context.get())
            return ""

        def stream(self, prompt, *, system=None, images=None):
            seen_ctx.update(capability_context.get())
            yield ""

    chains = {"private": _CaptureLLM()}
    selector = ChainSelector({"private": "private"}, default_chain="private")
    router_llm = SensitivityRouterLLM(chains, selector=selector, default_chain="private")

    engine = ScriptedEngine()
    runtime = VoiceRuntime(engine, router_llm, start_mode=Mode.ASSISTANT)
    runtime.start(run_bus=False)
    try:
        engine.final("what is the answer to life")  # generic factual
        assert runtime.wait_idle()
        # The brain set intent_kind based on the analyzer's decision. We
        # only assert that *something* came through -- the specific value
        # depends on the analyzer's classification, which is its own test.
        assert "intent_kind" in seen_ctx
        assert "sensitivity" in seen_ctx
        assert "mode" in seen_ctx
    finally:
        runtime.stop()


def test_attach_llm_capabilities_assistant_enriches_context_directly():
    """Unit-level: call assistant() directly with a bare context dict and
    verify it adds intent_kind + sensitivity before invoking the model.
    Doesn't go through the brain -- catches enrich logic regressions."""
    from always_on_agent.capabilities import CapabilityRegistry

    seen: dict = {}

    class _Spy:
        def generate(self, prompt, *, system=None, images=None) -> str:
            seen.update(capability_context.get())
            return "ok"

        def stream(self, prompt, *, system=None, images=None):
            seen.update(capability_context.get())
            yield "ok"

    registry = CapabilityRegistry()
    attach_llm_capabilities(registry, _Spy())

    result = registry.invoke(
        "assistant.answer",
        "What is the capital of France?",
        {"mode": Mode.ASSISTANT.value},
    )
    assert result.ok
    # Both signals were added before the LLM was called.
    assert seen.get("intent_kind") == IntentKind.ASSISTANT.value
    # Public-style opener with no personal markers -> public sensitivity.
    assert seen.get("sensitivity") == "public"


def test_attach_llm_capabilities_preserves_explicit_intent_kind():
    """If the caller already set intent_kind, the enricher must NOT
    overwrite it (the brain's task layer is the source of truth)."""
    from always_on_agent.capabilities import CapabilityRegistry

    seen: dict = {}

    class _Spy:
        def generate(self, prompt, *, system=None, images=None) -> str:
            seen.update(capability_context.get())
            return "ok"

        def stream(self, prompt, *, system=None, images=None):
            seen.update(capability_context.get())
            yield "ok"

    registry = CapabilityRegistry()
    attach_llm_capabilities(registry, _Spy())

    registry.invoke(
        "assistant.answer",
        "open the door",
        {"mode": Mode.COMMAND.value, "intent_kind": IntentKind.COMMAND.value},
    )
    assert seen["intent_kind"] == IntentKind.COMMAND.value
    # COMMAND intent forces private regardless of text.
    assert seen["sensitivity"] == "private"
