"""Tests for the ReAct planner capability and smart-mode escalation.

A scripted LLM drives the plan/execute loop deterministically, so these need no
real model and no audio.
"""

from __future__ import annotations

from threading import Event
from typing import Iterator

from always_on_agent.capabilities import (
    CapabilityRegistry,
    CapabilityResult,
    create_default_capabilities,
)
from always_on_agent.react import (
    FINAL_SYSTEM,
    PlannerConfig,
    ReactPlanner,
    attach_react_capability,
    should_escalate,
)

from core.capabilities import attach_llm_capabilities


class ScriptLLM:
    """Scripted LLM for the planner loop.

    The planner now drives BOTH plan steps and final synthesis through
    ``stream()`` (a real LLM yields the same content from ``stream`` and
    ``generate``). Route by the system prompt: the synthesis step
    (``FINAL_SYSTEM``) returns the fixed final; every other call (plan steps,
    which use ``PLANNER_SYSTEM``) pops the next queued plan reply. ``generate()``
    mirrors the same queue for any caller still using it.
    """

    def __init__(self, plan_replies: list[str], final: str = "final answer"):
        self._plan = list(plan_replies)
        self._final = final
        self.plan_prompts: list[str] = []

    def generate(self, prompt: str, *, system=None) -> str:
        self.plan_prompts.append(prompt)
        return self._plan.pop(0) if self._plan else "FINAL: fallback"

    def stream(self, prompt: str, *, system=None) -> Iterator[str]:
        if system == FINAL_SYSTEM:
            yield self._final
            return
        self.plan_prompts.append(prompt)
        yield self._plan.pop(0) if self._plan else "FINAL: fallback"


def test_planner_calls_tool_then_finalizes():
    registry = create_default_capabilities()
    llm = ScriptLLM(["TOOL search.local: pipecat", "FINAL: it is a voice framework"])
    planner = ReactPlanner(llm, registry, tools=("search.local",))

    result = planner.run("what is pipecat", {})
    assert result.ok
    assert result.text == "it is a voice framework"
    assert result.data["steps"] == ["search.local"]


def test_planner_recovers_from_failing_tool():
    registry = CapabilityRegistry()

    def flaky(query, context):
        return CapabilityResult(False, "", error="boom")

    registry.register("flaky.tool", flaky)
    llm = ScriptLLM(["TOOL flaky.tool: x", "FINAL: recovered without it"])
    planner = ReactPlanner(llm, registry, tools=("flaky.tool",))

    result = planner.run("do the thing", {})
    assert result.ok
    assert result.text == "recovered without it"
    assert result.data["steps"] == ["flaky.tool"]


def test_planner_handles_unavailable_tool():
    registry = create_default_capabilities()
    llm = ScriptLLM(["TOOL bogus.tool: x", "FINAL: answered anyway"])
    planner = ReactPlanner(llm, registry, tools=("search.local",))

    result = planner.run("q", {})
    assert result.text == "answered anyway"
    assert result.data["steps"] == []


def test_planner_respects_step_budget():
    registry = create_default_capabilities()
    # Always asks for another tool -> never finalizes on its own.
    llm = ScriptLLM(["TOOL search.local: a"] * 10, final="synthesized")
    planner = ReactPlanner(llm, registry, max_steps=2, tools=("search.local",))

    result = planner.run("q", {})
    assert result.data["exhausted"] is True
    assert len(result.data["steps"]) == 2
    assert result.text == "synthesized"


def test_planner_cancels_before_first_step():
    registry = create_default_capabilities()
    llm = ScriptLLM(["FINAL: never reached"])
    planner = ReactPlanner(llm, registry)

    cancel = Event()
    cancel.set()
    result = planner.run("q", {"cancel_event": cancel})
    assert result.data.get("cancelled") is True


def test_should_escalate_distinguishes_gathering_from_chitchat():
    assert should_escalate("search for the latest local TTS options") is True
    assert should_escalate("what time is it") is False


def test_planner_config_from_dict():
    cfg = PlannerConfig.from_dict({"enabled": True, "max_steps": 7, "tools": ["a", "b"]})
    assert cfg.enabled is True
    assert cfg.max_steps == 7
    assert cfg.tools == ("a", "b")


def test_smart_mode_escalates_assistant_to_planner():
    registry = create_default_capabilities()
    llm = ScriptLLM(["TOOL search.local: tts", "FINAL: here are the options"])
    attach_llm_capabilities(registry, llm, escalate=should_escalate)
    attach_react_capability(registry, llm, config=PlannerConfig(enabled=True))

    result = registry.invoke("assistant.answer", "search for local tts options")
    assert result.data.get("agent") is True
    assert result.text == "here are the options"


def test_simple_query_does_not_escalate():
    registry = create_default_capabilities()
    llm = ScriptLLM(["FINAL: should not be used"], final="should not be used")
    attach_llm_capabilities(registry, llm, escalate=should_escalate)
    attach_react_capability(registry, llm, config=PlannerConfig(enabled=True))

    result = registry.invoke("assistant.answer", "hello there")
    assert result.data.get("agent") is None
    assert result.data.get("route") in ("fast", "main")
