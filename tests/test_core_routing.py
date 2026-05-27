"""Tests for per-query model-tier routing (the fast/main split).

No audio, no models: a recording fake LLM lets us assert which tier the router
picked and that the assistant capability ran the matching model.
"""

from __future__ import annotations

from typing import Iterator

from always_on_agent.capabilities import CapabilityRegistry

from core.capabilities import attach_llm_capabilities
from core.routing import FAST, MAIN, HeuristicRouter, build_router


class RecordingLLM:
    def __init__(self, tag: str):
        self.tag = tag
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, system=None, images=None) -> str:
        self.prompts.append(prompt)
        return f"[{self.tag}] {prompt}"

    def stream(self, prompt: str, *, system=None, images=None) -> Iterator[str]:
        self.prompts.append(prompt)
        yield f"[{self.tag}] {prompt}"


def test_heuristic_keeps_short_literal_query_on_fast():
    router = HeuristicRouter()
    assert router.choose("what time is it", {}) == FAST


def test_heuristic_escalates_reasoning_query_to_main():
    router = HeuristicRouter()
    query = "explain why barge-in cancels the task and compare it to debouncing"
    assert router.choose(query, {}) == MAIN


def test_heuristic_research_mode_forces_main():
    router = HeuristicRouter()
    assert router.choose("options", {"mode": "research"}) == MAIN


def test_heuristic_dictation_mode_stays_fast():
    router = HeuristicRouter()
    long_dictation = " ".join(["word"] * 30)
    assert router.choose(long_dictation, {"mode": "dictation"}) == FAST


def test_build_router_defaults_to_heuristic():
    router = build_router({})
    assert isinstance(router, HeuristicRouter)
    assert router.threshold == 0.5


def test_build_router_reads_threshold():
    router = build_router({"llm": {"router": {"threshold": 0.8}}})
    assert isinstance(router, HeuristicRouter)
    assert router.threshold == 0.8


def test_assistant_routes_simple_query_to_fast_model():
    main = RecordingLLM("main")
    fast = RecordingLLM("fast")
    registry = attach_llm_capabilities(CapabilityRegistry(), main, fast_llm=fast)

    result = registry.invoke("assistant.answer", "what time is it")
    assert result.text.startswith("[fast]")
    assert result.data["route"] == FAST
    assert fast.prompts and not main.prompts


def test_assistant_routes_complex_query_to_main_model():
    main = RecordingLLM("main")
    fast = RecordingLLM("fast")
    registry = attach_llm_capabilities(CapabilityRegistry(), main, fast_llm=fast)

    query = "explain how the event bus works and why barge-in cancels tasks"
    result = registry.invoke("assistant.answer", query)
    assert result.text.startswith("[main]")
    assert result.data["route"] == MAIN
    assert main.prompts and not fast.prompts
