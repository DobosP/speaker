"""Tests for per-query model-tier routing (the fast/main split).

No audio, no models: a recording fake LLM lets us assert which tier the router
picked and that the assistant capability ran the matching model.
"""

from __future__ import annotations

from typing import Iterator

from always_on_agent.capabilities import CapabilityRegistry

from core.capabilities import attach_llm_capabilities
from core.metrics import ASR_FINAL, LLM_FIRST_TOKEN, MetricsRecorder
from core.routing import (
    FAST,
    MAIN,
    HeuristicRouter,
    build_router,
    live_nudge,
    order_presets_by_cost,
)

# A query whose static heuristic score lands just below the 0.5 threshold
# (two complexity markers: "explain" + "difference between" -> 0.36). It is the
# borderline case the live signal is meant to tip: FAST statically, MAIN once a
# slow-local nudge is applied. Pinned here so the routing tests below stay
# meaningful even if the marker weights are retuned (assert the static side too).
_BORDERLINE_QUERY = "explain the difference between them"


class FakeClock:
    """Deterministic monotonic clock advanced by hand (matches test_metrics)."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


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


# --- Live headroom signal (smart-routing-2) --------------------------------
#
# The contract: a *good* slow-local signal nudges a borderline turn toward main,
# while a missing/garbage signal leaves the static decision untouched. The nudge
# is additive-only + clamped so it can never starve the local tier.


def test_borderline_query_is_fast_without_live_signal():
    # Anchors the rest of the section: this query is FAST under the static
    # heuristic alone, so any escalation below is attributable to the signal.
    router = HeuristicRouter()
    assert router.choose(_BORDERLINE_QUERY, {}) == FAST


def test_slow_local_signal_nudges_borderline_query_to_main():
    router = HeuristicRouter()
    live = {"live": {"ttft_ms": 3000}}  # local first-token ~3s: slow tier
    assert router.choose(_BORDERLINE_QUERY, live) == MAIN


def test_high_system_load_signal_nudges_borderline_query_to_main():
    router = HeuristicRouter()
    live = {"live": {"load": 0.99}}  # device pegged: offload to main/cloud
    assert router.choose(_BORDERLINE_QUERY, live) == MAIN


def test_snappy_local_signal_leaves_borderline_on_fast():
    # A *good* signal that says local is fast must not escalate (no nudge).
    router = HeuristicRouter()
    live = {"live": {"ttft_ms": 400}}  # below the nudge floor
    assert router.choose(_BORDERLINE_QUERY, live) == FAST


def test_missing_live_signal_leaves_static_decision_unchanged():
    router = HeuristicRouter()
    # No 'live' key at all, and an empty/None mapping: all behave identically
    # to the static decision (cannot starve local).
    assert router.choose(_BORDERLINE_QUERY, {}) == FAST
    assert router.choose(_BORDERLINE_QUERY, {"live": {}}) == FAST
    assert router.choose(_BORDERLINE_QUERY, {"live": None}) == FAST


def test_garbage_live_signal_cannot_change_decision():
    router = HeuristicRouter()
    for bad in (
        {"live": "not-a-mapping"},
        {"live": {"ttft_ms": "oops"}},
        {"live": {"ttft_ms": float("nan")}},
        {"live": {"ttft_ms": float("inf")}},
        {"live": {"ttft_ms": -5.0}},
        {"live": {"load": True}},  # bool is rejected, not coerced to 1.0
        {"live": 12345},
        {"live": ["ttft_ms", 9000]},
    ):
        assert router.choose(_BORDERLINE_QUERY, bad) == FAST, bad


def test_slow_signal_cannot_flip_a_clearly_fast_query():
    # The clamp guarantees a single hot dimension cannot drag a clearly-fast
    # query (score 0.0) over the threshold -- the local tier is never starved.
    router = HeuristicRouter()
    extreme = {"live": {"ttft_ms": 999999, "load": 1.0}}
    assert router.choose("what time is it", extreme) == FAST


def test_live_nudge_is_additive_only_and_clamped():
    # Direct unit checks on the nudge: never negative, never exceeds the cap,
    # zero for anything uninterpretable.
    assert live_nudge(None) == 0.0
    assert live_nudge({}) == 0.0
    assert live_nudge("garbage") == 0.0
    assert live_nudge({"ttft_ms": 400}) == 0.0  # below floor
    assert 0.0 < live_nudge({"ttft_ms": 1600}) < 0.25  # mid-ramp
    assert live_nudge({"ttft_ms": 99999}) == 0.25  # saturated at cap
    # Two hot dimensions combine by max, not sum -- still capped.
    assert live_nudge({"ttft_ms": 99999, "load": 1.0}) == 0.25


def test_live_signal_never_lowers_the_static_score():
    # Property: for any query, the score with a live signal is >= the static
    # score. The nudge floors the local-tier behavior; it can only escalate.
    router = HeuristicRouter()
    for q in (
        "what time is it",
        _BORDERLINE_QUERY,
        "explain why barge-in cancels the task and compare it to debouncing",
    ):
        static = router.score(q, {})
        for live in ({"ttft_ms": 3000}, {"load": 0.99}, {"ttft_ms": "bad"}):
            assert router.score(q, {"live": live}) >= static


# --- Observed TTFT updates the EWMA (smart-routing-2 plumbing) --------------


def test_recorder_ttft_ewma_starts_unknown_and_updates():
    clock = FakeClock()
    rec = MetricsRecorder(clock=clock)
    assert rec.recent_ttft_ms() is None  # no measurable turn yet

    rec.mark(ASR_FINAL)
    clock.t = 0.5
    rec.mark(LLM_FIRST_TOKEN)  # 500ms first-token
    assert rec.recent_ttft_ms() == 500.0  # seeds the EWMA

    # Second turn at 1500ms pulls the EWMA up (toward, not all the way to, it).
    rec.mark(ASR_FINAL)  # repeated start banks the open turn / opens a new one
    clock.t = 1.0
    rec.mark(ASR_FINAL)
    clock.t = 2.5
    rec.mark(LLM_FIRST_TOKEN)  # 1500ms first-token
    ewma = rec.recent_ttft_ms()
    assert ewma is not None and 500.0 < ewma < 1500.0


def test_recorder_ttft_ewma_resets():
    clock = FakeClock()
    rec = MetricsRecorder(clock=clock)
    rec.mark(ASR_FINAL)
    clock.t = 0.3
    rec.mark(LLM_FIRST_TOKEN)
    assert rec.recent_ttft_ms() is not None
    rec.reset()
    assert rec.recent_ttft_ms() is None


# --- Headroom-aware routing through attach_llm_capabilities -----------------


def test_live_routing_off_by_default_keeps_static_decision():
    main = RecordingLLM("main")
    fast = RecordingLLM("fast")
    rec = MetricsRecorder()
    rec._observe_ttft_ms(3000.0)  # a very slow local sample is on record
    registry = attach_llm_capabilities(
        CapabilityRegistry(), main, fast_llm=fast, recorder=rec,
    )
    # live_routing defaults off -> the slow sample is ignored, borderline -> fast.
    result = registry.invoke("assistant.answer", _BORDERLINE_QUERY)
    assert result.data["route"] == FAST
    assert fast.prompts and not main.prompts


def test_live_routing_on_with_slow_sample_escalates_to_main():
    main = RecordingLLM("main")
    fast = RecordingLLM("fast")
    rec = MetricsRecorder()
    rec._observe_ttft_ms(3000.0)  # slow local tier on record
    registry = attach_llm_capabilities(
        CapabilityRegistry(), main, fast_llm=fast, recorder=rec, live_routing=True,
    )
    result = registry.invoke("assistant.answer", _BORDERLINE_QUERY)
    assert result.data["route"] == MAIN
    assert main.prompts and not fast.prompts


def test_live_routing_on_without_sample_does_not_starve_local():
    # Live routing enabled but no TTFT sample yet (cold start): the borderline
    # query must still go to fast -- a missing signal can never starve local.
    main = RecordingLLM("main")
    fast = RecordingLLM("fast")
    rec = MetricsRecorder()  # recent_ttft_ms() is None
    registry = attach_llm_capabilities(
        CapabilityRegistry(), main, fast_llm=fast, recorder=rec, live_routing=True,
    )
    result = registry.invoke("assistant.answer", _BORDERLINE_QUERY)
    assert result.data["route"] == FAST
    assert fast.prompts and not main.prompts


# --- Optional cloud-chain cost ordering (smart-routing-5) -------------------

_PROVIDERS = {
    "slow_cheap": {"_pricing_usd_per_mtok": {"in": 0.10, "out": 0.20, "ttft_ms": 500}},
    "fast_pricey": {"_pricing_usd_per_mtok": {"in": 0.60, "out": 1.20, "ttft_ms": 80}},
    "mid": {"_pricing_usd_per_mtok": {"in": 0.50, "out": 1.00, "ttft_ms": 100}},
}


def test_order_presets_by_ttft_then_cost():
    ordered = order_presets_by_cost(
        ["slow_cheap", "fast_pricey", "mid"], _PROVIDERS
    )
    # Lowest ttft_ms first; the helper does not reshuffle on cost when ttft
    # already separates them. Same multiset, just reordered.
    assert ordered == ["fast_pricey", "mid", "slow_cheap"]
    assert sorted(ordered) == sorted(["slow_cheap", "fast_pricey", "mid"])


def test_order_presets_ties_break_by_input_cost_stably():
    providers = {
        "a": {"_pricing_usd_per_mtok": {"in": 0.50, "out": 1.0, "ttft_ms": 100}},
        "b": {"_pricing_usd_per_mtok": {"in": 0.20, "out": 2.0, "ttft_ms": 100}},
        "c": {"_pricing_usd_per_mtok": {"in": 0.20, "out": 2.0, "ttft_ms": 100}},
    }
    # Same ttft: cheaper input wins (b/c over a); b and c tie fully so the
    # original order (b before c) is preserved (stable sort = stable failover).
    assert order_presets_by_cost(["a", "b", "c"], providers) == ["b", "c", "a"]


def test_order_presets_unannotated_sink_to_end_keeping_order():
    providers = {"fast_pricey": _PROVIDERS["fast_pricey"]}  # only one annotated
    ordered = order_presets_by_cost(
        ["unknown_one", "fast_pricey", "unknown_two"], providers
    )
    assert ordered == ["fast_pricey", "unknown_one", "unknown_two"]


def test_order_presets_is_failsafe_on_bad_input():
    # Garbage / missing inputs never break (or empty) the chain -- they return
    # the original order so failover semantics are preserved.
    assert order_presets_by_cost(["a", "b"], None) == ["a", "b"]
    assert order_presets_by_cost(["a", "b"], {}) == ["a", "b"]
    assert order_presets_by_cost("not-a-list", _PROVIDERS) == []
    assert order_presets_by_cost(["only"], _PROVIDERS) == ["only"]
    # A preset whose metadata is malformed sorts last but is never dropped.
    providers = {"good": _PROVIDERS["mid"], "bad": {"_pricing_usd_per_mtok": "nope"}}
    assert order_presets_by_cost(["bad", "good"], providers) == ["good", "bad"]


def test_order_presets_all_unannotated_preserves_chain_order():
    # No metadata anywhere: the configured failover order is the floor.
    assert order_presets_by_cost(["x", "y", "z"], {"x": {}, "y": {}, "z": {}}) == [
        "x",
        "y",
        "z",
    ]
