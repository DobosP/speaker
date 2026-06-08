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


def test_escalated_turn_publishes_context_and_resets_after():
    """An ESCALATED (ReAct planner) turn must classify sensitivity and publish
    a non-empty capability_context to the invoked planner capability, AND the
    ContextVar must be reset to its default after the call returns -- no leak
    into the next, unrelated turn (the §9.7 cross-turn risk, backlog P2 (d)).

    Before the fix the escalation early-return ran BEFORE _enrich_context and
    BEFORE capability_context.set(), so the planner's nested LLM calls saw an
    empty context and fell back to the default cloud chain."""
    from always_on_agent.capabilities import CapabilityRegistry

    seen: dict = {}

    def _planner(query: str, context: dict) -> "CapabilityResult":  # type: ignore[name-defined]
        from always_on_agent.capabilities import CapabilityResult

        # Capture what the planner-layer sees published in the ContextVar.
        seen.update(capability_context.get())
        return CapabilityResult(True, "planned")

    class _Spy:
        def generate(self, prompt, *, system=None, images=None) -> str:
            return "ok"

        def stream(self, prompt, *, system=None, images=None):
            yield "ok"

    registry = CapabilityRegistry()
    # Register the planner under the default agent capability name so the
    # escalation branch can find + invoke it.
    registry.register("agent.react", _planner)
    attach_llm_capabilities(registry, _Spy(), escalate=lambda q, ctx: True)

    # Sanity: clean ContextVar going in.
    assert capability_context.get() == {}

    result = registry.invoke(
        "assistant.answer",
        "research the latest news and compare options",
        {"mode": Mode.ASSISTANT.value},
    )
    assert result.ok
    assert result.text == "planned"  # the planner ran, not the one-shot reply

    # The planner saw a non-empty, sensitivity-bearing context.
    assert seen.get("sensitivity"), "escalated turn did not publish sensitivity"
    assert seen.get("intent_kind") == IntentKind.ASSISTANT.value

    # And the ContextVar is back to its default afterward -- no cross-turn leak.
    assert capability_context.get() == {}, (
        "capability_context leaked after the escalated planner returned"
    )


def test_escalated_turn_resets_context_on_failure_path():
    """The reset must run via finally on EVERY exit path. A planner that raises
    is turned into a failed CapabilityResult by registry.invoke (it swallows the
    exception), so the escalated branch returns normally -- but the finally must
    still have restored the ContextVar so a failed escalated turn can't leak its
    tag into the next turn."""
    from always_on_agent.capabilities import CapabilityRegistry

    def _planner(query: str, context: dict):
        # During this call the ContextVar must be set (published to the planner);
        # raising here exercises the finally-reset on the failure path.
        assert capability_context.get(), "context not published to the planner"
        raise RuntimeError("planner boom")

    class _Spy:
        def generate(self, prompt, *, system=None, images=None) -> str:
            return "ok"

        def stream(self, prompt, *, system=None, images=None):
            yield "ok"

    registry = CapabilityRegistry()
    registry.register("agent.react", _planner)
    attach_llm_capabilities(registry, _Spy(), escalate=lambda q, ctx: True)

    assert capability_context.get() == {}
    result = registry.invoke(
        "assistant.answer",
        "compare the options",
        {"mode": Mode.ASSISTANT.value},
    )
    # registry.invoke swallowed the planner's RuntimeError into a failed result.
    assert not result.ok
    assert "planner boom" in result.error
    assert capability_context.get() == {}, (
        "context leaked after the escalated planner raised"
    )


def test_escalated_turn_receives_recent_conversation_context():
    """An ESCALATED (ReAct planner) turn must see the recent-conversation block so
    "explain that step by step" keeps the thread -- before the fix the recent block
    was built only on the one-shot path (after the escalation early-return), so the
    planner got a contextless query."""
    from always_on_agent.capabilities import CapabilityRegistry, CapabilityResult
    from always_on_agent.memory import SessionMemory

    seen: dict = {}

    def _planner(query: str, context: dict) -> CapabilityResult:
        seen.update(context)
        return CapabilityResult(True, "planned")

    class _Spy:
        def generate(self, prompt, *, system=None, images=None) -> str:
            return "ok"

        def stream(self, prompt, *, system=None, images=None):
            yield "ok"

    memory = SessionMemory()
    memory.add("what is the capital of france", tags=("user",))
    memory.add("Paris.", tags=("assistant_output",))

    registry = CapabilityRegistry()
    registry.register("agent.react", _planner)
    attach_llm_capabilities(
        registry, _Spy(), escalate=lambda q, ctx: True, memory=memory
    )

    result = registry.invoke(
        "assistant.answer",
        "explain that in more detail",
        {"mode": Mode.ASSISTANT.value},
    )
    assert result.ok and result.text == "planned"
    block = seen.get("recent_conversation", "")
    assert block, "escalated turn did not receive the recent-conversation context"
    assert "Recent conversation" in block
    assert "capital of france" in block and "Paris" in block


def test_escalated_turn_floats_sensitivity_over_recent_turns():
    """§9.7: when a prior turn in the recent block is private, the escalated turn's
    published sensitivity must float to private BEFORE the planner's nested LLM
    calls run (the recent turns now ride the planner prompt, so they must not pull
    a public chain)."""
    from always_on_agent.capabilities import CapabilityRegistry, CapabilityResult
    from always_on_agent.memory import SessionMemory
    from core.sensitivity import PRIVATE

    seen: dict = {}

    def _planner(query: str, context: dict) -> CapabilityResult:
        seen.update(capability_context.get())
        return CapabilityResult(True, "planned")

    class _Spy:
        def generate(self, prompt, *, system=None, images=None) -> str:
            return "ok"

        def stream(self, prompt, *, system=None, images=None):
            yield "ok"

    memory = SessionMemory()
    memory.add("my social security number is 123-45-6789", tags=("user",))
    memory.add("Got it.", tags=("assistant_output",))

    registry = CapabilityRegistry()
    registry.register("agent.react", _planner)
    attach_llm_capabilities(
        registry, _Spy(), escalate=lambda q, ctx: True, memory=memory
    )

    registry.invoke(
        "assistant.answer", "explain that", {"mode": Mode.ASSISTANT.value}
    )
    assert seen.get("sensitivity") == PRIVATE, (
        "escalated turn did not float sensitivity over a private prior turn"
    )


def test_one_shot_path_still_composes_recent_conversation_into_system():
    """Regression: refactoring the recent block to build BEFORE the escalation
    branch must NOT change the one-shot path -- the block still lands in the
    answering model's system prompt."""
    from always_on_agent.capabilities import CapabilityRegistry
    from always_on_agent.memory import SessionMemory

    seen_systems: list = []

    class _Spy:
        def generate(self, prompt, *, system=None, images=None) -> str:
            return "ok"

        def stream(self, prompt, *, system=None, images=None):
            seen_systems.append(system or "")
            yield "ok"

    memory = SessionMemory()
    memory.add("what is the capital of france", tags=("user",))
    memory.add("Paris.", tags=("assistant_output",))

    registry = CapabilityRegistry()
    # escalate=None -> always the one-shot path.
    attach_llm_capabilities(registry, _Spy(), memory=memory)
    result = registry.invoke(
        "assistant.answer", "what is its population", {"mode": Mode.ASSISTANT.value}
    )
    assert result.ok
    assert seen_systems, "the answering model was never streamed"
    assert "Recent conversation" in seen_systems[-1]
    assert "Paris" in seen_systems[-1]


def test_research_synth_publishes_context_and_resets_after():
    """research.local streams through the same SensitivityRouterLLM, so it must
    also classify + publish sensitivity and reset the ContextVar afterward."""
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

    assert capability_context.get() == {}
    result = registry.invoke(
        "research.local",
        "what is the weather in Paris",
        {"mode": Mode.RESEARCH.value},
    )
    assert result.ok
    assert seen.get("sensitivity"), "research.local did not publish sensitivity"
    assert capability_context.get() == {}, (
        "capability_context leaked after research.local returned"
    )


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
