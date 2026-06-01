"""Pure-logic tests for the unified capability router (core/capability_router.py).

No audio, no model: a fake fast-LLM exercises the disambiguation path. Covers the
heuristic floor (control/simple/research/act + tier + confidence), the LLM-assist
wrap (only fires on low confidence, memoized, fail-safe), the Router/escalate
adapters the runtime wires, and config-driven construction."""
from __future__ import annotations

import pytest

from core.capability_router import (
    ACT,
    CONTROL,
    RESEARCH,
    SIMPLE,
    CapabilityTierRouter,
    HeuristicCapabilityRouter,
    LLMCapabilityRouter,
    build_capability_router,
    escalate_predicate,
)
from core.routing import FAST, MAIN

# A long, marker-free utterance: the heuristic routes it SIMPLE with LOW
# confidence (>=12 words, no gather/act markers) -- the exact case the LLM
# disambiguator is meant to reconsider.
AMBIGUOUS = "i was wondering what you personally think about the future of artificial intelligence"


class FakeLLM:
    """Minimal LLMClient.generate stub with a call counter + scripted reply."""

    def __init__(self, reply: str = "SIMPLE", *, raises: bool = False) -> None:
        self.reply = reply
        self.raises = raises
        self.calls = 0

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        self.calls += 1
        if self.raises:
            raise RuntimeError("llm down")
        return self.reply


# --- heuristic floor ---------------------------------------------------------


@pytest.mark.parametrize(
    "text, want",
    [
        ("stop", CONTROL),
        ("cancel", CONTROL),
        ("what time is it", SIMPLE),
        ("what is the capital of france", SIMPLE),
        ("compare the economies of japan and germany step by step", RESEARCH),
        ("search for the latest news on mars", RESEARCH),
        ("set a timer for five minutes", ACT),
        ("turn off the living room lights", ACT),
        ("remind me to call mom at six", ACT),
    ],
)
def test_heuristic_actions(text, want):
    assert HeuristicCapabilityRouter().route(text, {}).action == want


def test_configured_command_phrase_is_control():
    r = HeuristicCapabilityRouter(command_phrases=["assistant mode", "research mode"])
    assert r.route("assistant mode", {}).action == CONTROL


def test_research_and_act_escalate_but_simple_control_do_not():
    r = HeuristicCapabilityRouter()
    assert r.route("compare a and b step by step", {}).escalates is True
    assert r.route("set a timer for ten minutes", {}).escalates is True
    assert r.route("what time is it", {}).escalates is False
    assert r.route("stop", {}).escalates is False


def test_short_simple_is_high_confidence_long_is_low():
    r = HeuristicCapabilityRouter()
    short = r.route("what time is it", {})
    assert short.action == SIMPLE and short.confidence >= 0.65
    longish = r.route(AMBIGUOUS, {})
    assert longish.action == SIMPLE and longish.confidence < 0.65


def test_control_tier_is_fast_research_tier_is_main():
    r = HeuristicCapabilityRouter()
    assert r.route("stop", {}).tier == FAST
    assert r.route("compare a and b step by step", {}).tier == MAIN
    assert r.route("set a timer for ten minutes", {}).tier == MAIN


def test_empty_text_is_safe_simple():
    d = HeuristicCapabilityRouter().route("", {})
    assert d.action == SIMPLE  # no crash, no spurious control/escalate
    assert d.escalates is False


# --- LLM-assist wrap ---------------------------------------------------------


def test_llm_consulted_only_on_low_confidence():
    llm = FakeLLM(reply="RESEARCH")
    r = LLMCapabilityRouter(HeuristicCapabilityRouter(), llm, confidence_threshold=0.65)
    # High-confidence SIMPLE -> never calls the model.
    assert r.route("what time is it", {}).action == SIMPLE
    assert llm.calls == 0
    # Low-confidence SIMPLE -> the model reclassifies it RESEARCH.
    out = r.route(AMBIGUOUS, {})
    assert out.action == RESEARCH and out.source == "llm" and out.tier == MAIN
    assert llm.calls == 1


def test_llm_not_consulted_for_control():
    llm = FakeLLM(reply="RESEARCH")
    r = LLMCapabilityRouter(HeuristicCapabilityRouter(), llm, confidence_threshold=0.65)
    assert r.route("stop", {}).action == CONTROL
    assert llm.calls == 0


def test_llm_action_is_memoized_per_utterance():
    llm = FakeLLM(reply="RESEARCH")
    r = LLMCapabilityRouter(HeuristicCapabilityRouter(), llm, confidence_threshold=0.65)
    # The runtime consults the router multiple times per turn (escalate + tier);
    # the LLM must run at most once for the same utterance.
    for _ in range(3):
        assert r.route(AMBIGUOUS, {}).action == RESEARCH
    assert llm.calls == 1


def test_llm_unparseable_reply_falls_back_to_heuristic():
    llm = FakeLLM(reply="I think this needs research")  # first word not a label
    r = LLMCapabilityRouter(HeuristicCapabilityRouter(), llm, confidence_threshold=0.65)
    assert r.route(AMBIGUOUS, {}).action == SIMPLE  # heuristic stands
    # None is cached too -> no repeated calls.
    r.route(AMBIGUOUS, {})
    assert llm.calls == 1


def test_llm_exception_falls_back_to_heuristic():
    llm = FakeLLM(raises=True)
    r = LLMCapabilityRouter(HeuristicCapabilityRouter(), llm, confidence_threshold=0.65)
    assert r.route(AMBIGUOUS, {}).action == SIMPLE


# --- runtime adapters --------------------------------------------------------


def test_tier_router_adapter_reflects_decision():
    r = HeuristicCapabilityRouter()
    tier = CapabilityTierRouter(r)
    assert tier.choose("compare a and b step by step", {}) == MAIN
    assert tier.choose("what time is it", {}) == FAST
    assert tier.score("compare a and b step by step", {}) == 1.0
    assert tier.score("what time is it", {}) == 0.0


def test_escalate_predicate_matches_research_and_act():
    esc = escalate_predicate(HeuristicCapabilityRouter())
    assert esc("compare a and b step by step", {}) is True
    assert esc("set a timer for ten minutes", {}) is True
    assert esc("what time is it", {}) is False
    assert esc("hello there", None) is False  # tolerates None context


# --- config construction -----------------------------------------------------


def test_build_returns_none_when_disabled_or_absent():
    assert build_capability_router({}) is None
    assert build_capability_router({"capability_router": {"enabled": False}}) is None


def test_build_heuristic_when_enabled_without_llm_assist():
    r = build_capability_router({"capability_router": {"enabled": True, "llm_assist": False}})
    assert isinstance(r, HeuristicCapabilityRouter)


def test_build_llm_assist_only_when_fast_llm_present():
    cfg = {"capability_router": {"enabled": True, "llm_assist": True}}
    # No fast LLM -> degrade to the heuristic (never crash on a weak device).
    assert isinstance(build_capability_router(cfg), HeuristicCapabilityRouter)
    # With a fast LLM -> the disambiguating wrap.
    r = build_capability_router(cfg, fast_llm=FakeLLM())
    assert isinstance(r, LLMCapabilityRouter)
