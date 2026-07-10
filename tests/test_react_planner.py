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
    DEFAULT_TOOLS,
    FINAL_SYSTEM,
    PlannerConfig,
    ReactPlanner,
    _parse_step,
    attach_react_capability,
    should_escalate,
)

from core.capabilities import attach_llm_capabilities
from core.websearch import WebSearchConfig, attach_web_search_capability


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
        self.stream_calls = 0

    def generate(self, prompt: str, *, system=None) -> str:
        self.plan_prompts.append(prompt)
        return self._plan.pop(0) if self._plan else "FINAL: fallback"

    def stream(self, prompt: str, *, system=None) -> Iterator[str]:
        self.stream_calls += 1
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


def test_tool_cancellation_cannot_launch_post_budget_final_model_call():
    cancel = Event()
    registry = CapabilityRegistry()

    def cancelling_tool(query, context):
        cancel.set()
        return CapabilityResult(True, "late tool result")

    registry.register("cancel.tool", cancelling_tool)
    llm = ScriptLLM(["TOOL cancel.tool: x"], final="must not run")
    planner = ReactPlanner(llm, registry, max_steps=1, tools=("cancel.tool",))

    result = planner.run("q", {"cancel_event": cancel})

    assert result.data.get("cancelled") is True
    assert llm.stream_calls == 1


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


# --- web.search is a default planner tool (P3 step 5) ----------------------


def test_web_search_is_in_default_tools():
    """web.search leads the default gather tools so the planner can reach real
    web research (with corpus fallback) without per-config opt-in."""
    assert "web.search" in DEFAULT_TOOLS
    # search.local stays available as the offline fallback tool.
    assert "search.local" in DEFAULT_TOOLS


def test_web_search_appears_in_planner_catalog():
    """The catalog the model sees (built from _TOOL_DESCRIPTIONS) lists
    web.search with a description so the planner knows it can call it."""
    registry = create_default_capabilities()
    planner = ReactPlanner(registry=registry, llm=ScriptLLM([]))
    catalog = planner._catalog()
    assert "web.search" in catalog
    # It's described, not the generic "a local capability" fallback.
    assert "- web.search:" in catalog
    assert "search the web" in catalog


def test_planner_can_call_web_search_tool():
    """End to end: the planner invokes the registered web.search capability and
    folds its result into the final answer."""
    registry = create_default_capabilities()
    attach_web_search_capability(registry, WebSearchConfig(enabled=False))
    llm = ScriptLLM(["TOOL web.search: pipecat", "FINAL: pipecat is a voice framework"])
    planner = ReactPlanner(llm, registry, tools=("web.search",))

    result = planner.run("what is pipecat", {})
    assert result.ok
    assert result.data["steps"] == ["web.search"]
    assert result.text == "pipecat is a voice framework"


def test_planner_threads_recent_conversation_into_plan_and_final_prompts():
    """The runtime publishes a recent-conversation block under
    ``context['recent_conversation']``; the planner must weave it into BOTH the
    plan prompt (so it can resolve "explain that") AND the final-answer prompt (so
    the spoken reply keeps the thread). Absent -> prompts unchanged (other tests)."""
    from always_on_agent.react import FINAL_SYSTEM

    class _CapturingLLM:
        def __init__(self) -> None:
            self.prompts: list[str] = []
            self._plan = ["FINAL"]  # FINAL with NO arg -> forces _final()

        def generate(self, prompt: str, *, system=None) -> str:
            self.prompts.append(prompt)
            return "answer"

        def stream(self, prompt: str, *, system=None) -> Iterator[str]:
            self.prompts.append(prompt)
            if system == FINAL_SYSTEM:
                yield "final answer"
                return
            yield self._plan.pop(0) if self._plan else "FINAL"

    registry = create_default_capabilities()
    llm = _CapturingLLM()
    planner = ReactPlanner(llm, registry)
    recent = (
        "=== Recent conversation (most recent last) ===\n"
        "User: what is the capital of france\nYou: Paris."
    )
    result = planner.run("explain that", {"recent_conversation": recent})
    assert result.ok
    # Both the plan prompt and the final prompt carried the conversation block.
    assert len(llm.prompts) >= 2
    assert all("Recent conversation" in p and "Paris" in p for p in llm.prompts)


def test_planner_omits_recent_block_when_absent():
    """No recent_conversation in context -> the prompts are byte-identical to the
    pre-enhancement shape (no stray blank header)."""
    registry = create_default_capabilities()
    llm = ScriptLLM(["FINAL: done"])
    planner = ReactPlanner(llm, registry)
    planner.run("just answer", {})
    assert llm.plan_prompts
    assert "Recent conversation" not in llm.plan_prompts[0]
    assert llm.plan_prompts[0].startswith("User request:")


def test_recent_conversation_reaches_the_real_planner_end_to_end():
    """Contract pin: a turn that ESCALATES through the real assistant() -> the real
    ReactPlanner must carry the recent-conversation block into the planner's
    prompts. Guards the constant<->literal coupling between
    core.capabilities.RECENT_CONVERSATION_KEY and the "recent_conversation" literal
    in always_on_agent.react that the unit tests pin only one side of -- a rename of
    one without the other would pass both unit tests yet break production."""
    from always_on_agent.memory import SessionMemory
    from always_on_agent.react import FINAL_SYSTEM, attach_react_capability
    from core.capabilities import RECENT_CONVERSATION_KEY

    # The two modules must agree on the literal key, end to end.
    assert RECENT_CONVERSATION_KEY == "recent_conversation"

    class _CapturingLLM:
        def __init__(self) -> None:
            self.prompts: list[str] = []
            self._plan = ["FINAL"]

        def generate(self, prompt: str, *, system=None, images=None) -> str:
            self.prompts.append(prompt)
            return "answer"

        def stream(self, prompt: str, *, system=None, images=None) -> Iterator[str]:
            self.prompts.append(prompt)
            if system == FINAL_SYSTEM:
                yield "final answer"
                return
            yield self._plan.pop(0) if self._plan else "FINAL"

    memory = SessionMemory()
    memory.add("what is the capital of france", tags=("user",))
    memory.add("Paris.", tags=("assistant_output",))

    llm = _CapturingLLM()
    registry = CapabilityRegistry()
    attach_react_capability(registry, llm)  # the REAL planner under "agent.react"
    attach_llm_capabilities(registry, llm, escalate=lambda q, ctx: True, memory=memory)

    result = registry.invoke(
        "assistant.answer", "explain that in detail", {"mode": "assistant"}
    )
    assert result.ok
    assert llm.prompts, "the real planner never called the LLM"
    assert any("Recent conversation" in p and "Paris" in p for p in llm.prompts), (
        "recent-conversation block did not reach the real planner end-to-end"
    )


# --- P3: robust tool-call parsing + bounded re-prompt ------------------------

def test_parse_step_scans_past_preamble_and_markdown():
    # A small model that rambles before the directive, or wraps it in a bullet,
    # still parses (a strict colon-required scan over every line).
    assert _parse_step("Let me check that.\nTOOL search.local: pgvector") == ("search.local", "pgvector")
    assert _parse_step("- TOOL web.search: cats") == ("web.search", "cats")
    assert _parse_step("`FINAL: forty two`") == ("FINAL", "forty two")
    assert _parse_step("FINAL: The capital is Paris.") == ("FINAL", "The capital is Paris.")
    # genuinely no directive -> None (wrap up)
    action, _ = _parse_step("I think the capital is Paris.")
    assert action is None


def test_parse_step_does_not_misread_prose_as_directive():
    # Review regression (P3): a benign line that merely STARTS with "Tool"/"Final"
    # (no colon after the keyword/name) must NOT be parsed as a directive -- whether
    # it follows a preamble line OR is the model's ONLY line (the bare-prose form the
    # earlier line-1-lenient parser still misread into a spurious web.search egress).
    for prose in (
        "I will help.\nTool choice depends on your needs.",
        "Sure.\nTool web.search is great for code.",
        "Here is my reasoning.\nFinal answer: 42.",
        "Let me summarize.\nFinal note: I think the answer is yes.",
        "The best tool for the job is a hammer.",
        # bare single-line prose (no benign first line in front of it)
        "Tool web.search is great for code.",
        "Tool choice depends on your needs.",
        "Final answer is probably forty two.",
        "Finally, I should mention the weather.",
    ):
        assert _parse_step(prose)[0] is None, prose


def test_parse_step_skips_directives_inside_code_fences():
    # Review regression (P3): an EXAMPLE directive the model quotes inside a fenced
    # code block must NOT fire a real tool call / outbound egress.
    fenced_tool = "Here is the format:\n```\nTOOL web.search: example query\n```\nFINAL: done"
    assert _parse_step(fenced_tool) == ("FINAL", "done")
    fenced_only = "```\nTOOL search.local: not a real call\n```"
    assert _parse_step(fenced_only)[0] is None
    # a real directive after the fence still parses
    assert _parse_step("```\nsome example\n```\nTOOL web.search: cats") == ("web.search", "cats")


def test_parse_step_preserves_payload_punctuation():
    # Review regression (P3): interior * ` # > must NOT be stripped from the arg.
    assert _parse_step("TOOL web.search: C# tutorials") == ("web.search", "C# tutorials")
    assert _parse_step("looking...\nTOOL web.search: C# tips & #hashtags") == (
        "web.search", "C# tips & #hashtags")


def test_planner_reprompts_once_on_unparseable_step():
    # First plan reply is junk (no directive); the planner re-prompts ONCE with the
    # strict format reminder, the model then emits a valid tool call.
    registry = create_default_capabilities()
    llm = ScriptLLM(["um, let me think about this...", "TOOL search.local: pipecat",
                     "FINAL: it is a voice framework"])
    planner = ReactPlanner(llm, registry, tools=("search.local",))
    result = planner.run("what is pipecat", {})
    assert result.ok
    assert result.data["steps"] == ["search.local"]
    # the re-prompt appended the format reminder to a plan prompt
    assert any("EXACTLY one line" in p for p in llm.plan_prompts)


def test_planner_reprompt_is_bounded_to_once():
    # Two junk replies in a row: after the single re-prompt is spent, it gives up
    # and synthesizes a FINAL rather than looping.
    registry = create_default_capabilities()
    llm = ScriptLLM(["rambling one", "rambling two"], final="synthesized answer")
    planner = ReactPlanner(llm, registry, tools=("search.local",))
    result = planner.run("q", {})
    assert result.ok
    assert result.text == "synthesized answer"


def test_build_router_llm_defaults_to_fast_when_unset():
    from core.llm_factory import build_router_llm

    sentinel = object()
    assert build_router_llm({"llm": {}}, sentinel) is sentinel


def test_build_router_llm_builds_dedicated_local_when_set():
    from core.llm_factory import build_router_llm
    from core.llm import HedgeLLM, OllamaLLM, SensitivityRouterLLM

    r = build_router_llm({"llm": {"router_model": "xlam2:3b", "backend": "ollama"}}, object())
    assert isinstance(r, OllamaLLM) and r.model == "xlam2:3b"
    # §9.7: the router sees the raw query, so it must NEVER be cloud-wrapped.
    assert not isinstance(r, (HedgeLLM, SensitivityRouterLLM))
    # non-ollama backend falls back to the fast tier
    sentinel = object()
    assert build_router_llm({"llm": {"router_model": "x", "backend": "llamacpp"}}, sentinel) is sentinel
