"""Tests for the intent-aware extensions to core.routing.

Covers:
- HeuristicRouter.score reads context["intent_kind"] and adjusts the tier.
- ChainSelector picks chain names by context["sensitivity"].
- build_chain_selector consumes config.llm.cloud_routing correctly.
"""
from __future__ import annotations

from core.routing import (
    ChainSelector,
    HeuristicRouter,
    build_chain_selector,
)


# --- HeuristicRouter intent_kind signals -----------------------------------


def test_research_intent_pushes_to_main():
    router = HeuristicRouter()
    short_factual = "what is climate change"
    # Short factual question: in plain assistant mode -> below threshold.
    base = router.score(short_factual, {"mode": "assistant"})
    # Same short question tagged as RESEARCH intent -> escalates.
    with_intent = router.score(short_factual, {"intent_kind": "research", "mode": "assistant"})
    assert with_intent > base
    assert router.choose(short_factual, {"intent_kind": "research"}) == "main"


def test_search_intent_pushes_to_main():
    router = HeuristicRouter()
    assert router.choose("look up something", {"intent_kind": "search"}) == "main"


def test_command_intent_pushes_to_fast():
    router = HeuristicRouter()
    # A wordy "how does this compare to ... explain step by step" would
    # normally trigger MAIN; tagging it COMMAND drops it to FAST.
    wordy = "compare and explain step by step why the cat sat on the mat"
    base = router.score(wordy, {})
    with_cmd = router.score(wordy, {"intent_kind": "command"})
    assert with_cmd < base


def test_dictation_intent_keeps_fast():
    router = HeuristicRouter(threshold=0.5)
    assert router.choose("hello there", {"intent_kind": "dictation"}) == "fast"


def test_assistant_intent_is_neutral():
    router = HeuristicRouter()
    base = router.score("how does this work", {"mode": "assistant"})
    with_assist = router.score(
        "how does this work", {"mode": "assistant", "intent_kind": "assistant"}
    )
    assert with_assist == base  # ASSISTANT doesn't change the score


def test_unknown_intent_kind_does_not_crash():
    router = HeuristicRouter()
    # Random strings in intent_kind shouldn't raise -- they just don't match
    # _MAIN_INTENTS or _FAST_INTENTS, leaving the score untouched.
    score = router.score("hi", {"intent_kind": "klingon"})
    assert 0.0 <= score <= 1.0


def test_score_stays_clamped_to_0_1_even_with_strong_intent_boost():
    router = HeuristicRouter()
    # A long, complex query plus RESEARCH intent could push the unclamped
    # sum past 1.0 -- the clamp must hold.
    long_complex = (
        "why does this happen and how does it compare and explain step by step "
        "in detail with pros and cons and the trade-off and analyze the implications"
    )
    score = router.score(long_complex, {"intent_kind": "research", "mode": "research"})
    assert 0.0 <= score <= 1.0


# --- ChainSelector ---------------------------------------------------------


def test_chain_selector_maps_sensitivity_to_chain():
    s = ChainSelector(
        {"private": "private", "code": "code", "public": "public"},
        default_chain="private",
    )
    assert s.choose_chain({"sensitivity": "private"}) == "private"
    assert s.choose_chain({"sensitivity": "code"}) == "code"
    assert s.choose_chain({"sensitivity": "public"}) == "public"


def test_chain_selector_returns_default_on_unknown_sensitivity():
    s = ChainSelector({"private": "private"}, default_chain="private")
    assert s.choose_chain({"sensitivity": "alien"}) == "private"
    assert s.choose_chain({}) == "private"
    assert s.choose_chain({"sensitivity": ""}) == "private"


def test_chain_selector_accepts_non_string_default():
    """Defensive: even a malformed context shouldn't blow up."""
    s = ChainSelector({"x": "x"}, default_chain="x")
    assert s.choose_chain(None) == "x"  # type: ignore[arg-type]
    assert s.choose_chain({"sensitivity": None}) == "x"


# --- build_chain_selector --------------------------------------------------


def test_build_chain_selector_reads_config_block():
    cfg = {
        "llm": {
            "cloud_routing": {
                "default_chain": "code",
                "sensitivity_to_chain": {"code": "code", "public": "public"},
            }
        }
    }
    s = build_chain_selector(cfg)
    assert s.default_chain == "code"
    assert s.choose_chain({"sensitivity": "public"}) == "public"
    assert s.choose_chain({"sensitivity": "unknown"}) == "code"


def test_build_chain_selector_with_missing_config_uses_defaults():
    """No config block -> default_chain="private", empty mapping."""
    s = build_chain_selector(None)
    assert s.default_chain == "private"
    assert s.choose_chain({"sensitivity": "private"}) == "private"  # falls through


def test_build_chain_selector_with_empty_routing_uses_defaults():
    s = build_chain_selector({"llm": {}})
    assert s.default_chain == "private"
