"""Tests for per-query model-tier routing (the fast/main split).

No audio, no models: a recording fake LLM lets us assert which tier the router
picked and that the assistant capability ran the matching model.
"""

from __future__ import annotations

from typing import Iterator

from always_on_agent.capabilities import CapabilityRegistry

from core.capabilities import attach_llm_capabilities
from core.llm import HedgeLLM
from core.llm_factory import _wrap_cloud
from core.metrics import ASR_FINAL, LLM_FIRST_TOKEN, MetricsRecorder, mark_first_token
from core.routing import (
    FAST,
    HEDGE_DELAY_FLOOR_MS,
    MAIN,
    HeuristicRouter,
    build_router,
    dynamic_hedge_delay_ms,
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


def test_order_presets_cn_sorts_after_us_regardless_of_speed():
    # host_rank is the OUTERMOST key: cost/latency optimizes WITHIN a jurisdiction
    # tier, never across it -- a faster, cheaper CN preset still races LAST so it
    # can't float ahead of a US preset the user ordered first (§9.7 / PRC opt-in).
    providers = {
        "us_slow": {"_pricing_usd_per_mtok": {"in": 0.60, "out": 1.20, "ttft_ms": 500, "host": "US"}},
        "cn_fast": {"_pricing_usd_per_mtok": {"in": 0.10, "out": 0.20, "ttft_ms": 80, "host": "CN"}},
    }
    assert order_presets_by_cost(["cn_fast", "us_slow"], providers) == ["us_slow", "cn_fast"]


def test_order_presets_us_no_ttft_still_outranks_cn():
    # The shipped-public-chain bug: a US aggregator with cost but NO ttft_ms
    # (OpenRouter) must NOT sink below a CN provider. host_rank lifts every
    # US/unknown preset above all CN ones; within US, the ttft-annotated cerebras
    # races first and the no-ttft openrouter follows -- but both beat CN deepseek.
    providers = {
        # host can be top-level (openrouter) OR inside the pricing block (others).
        "openrouter": {"host": "US", "_pricing_usd_per_mtok": {"in": 0.15, "out": 0.60, "host": "US"}},
        "deepseek": {"_pricing_usd_per_mtok": {"in": 0.14, "out": 0.28, "ttft_ms": 400, "host": "CN"}},
        "cerebras": {"_pricing_usd_per_mtok": {"in": 0.60, "out": 1.20, "ttft_ms": 80, "host": "US"}},
    }
    assert order_presets_by_cost(
        ["openrouter", "deepseek", "cerebras"], providers
    ) == ["cerebras", "openrouter", "deepseek"]


# --- cost_order flag wiring in _wrap_cloud (smart-routing-5) -----------------
#
# The flag-gated reorder lands in the multi-provider HedgeLLM chain. Models are
# distinct per preset so the resolved HedgeLLM.clouds order is verifiable; API
# keys are set so every preset resolves. The flag default (off) must preserve
# the configured failover order byte-for-byte.

# ttft_ms decreasing in CONFIGURED order so a cost reorder visibly reverses it.
_COST_PROVIDERS = {
    "p_slow": {
        "model": "model-slow", "api_key_env": "COST_KEY",
        "_pricing_usd_per_mtok": {"in": 0.10, "out": 0.20, "ttft_ms": 500, "host": "US"},
    },
    "p_mid": {
        "model": "model-mid", "api_key_env": "COST_KEY",
        "_pricing_usd_per_mtok": {"in": 0.50, "out": 1.00, "ttft_ms": 200, "host": "US"},
    },
    "p_fast": {
        "model": "model-fast", "api_key_env": "COST_KEY",
        "_pricing_usd_per_mtok": {"in": 0.60, "out": 1.20, "ttft_ms": 80, "host": "US"},
    },
}


def _cost_llm_cfg(*, cost_order: bool) -> dict:
    return {
        "cloud": {"enabled": True, "strategy": "hedge", "cost_order": cost_order},
        "cloud_providers": dict(_COST_PROVIDERS),
        "cloud_chains": {"private": ["p_slow", "p_mid", "p_fast"]},
        "cloud_routing": {"default_chain": "private"},
    }


def _chain_models(wrapped) -> list[str]:
    chain = wrapped.chains["private"]
    assert isinstance(chain, HedgeLLM)
    return [c.model for c in chain.clouds]


def test_cost_order_off_preserves_configured_chain_order(monkeypatch):
    monkeypatch.setenv("COST_KEY", "k")
    wrapped = _wrap_cloud(EchoLLMStub(), _cost_llm_cfg(cost_order=False))
    # Default (flag off): the configured failover order is unchanged.
    assert _chain_models(wrapped) == ["model-slow", "model-mid", "model-fast"]


def test_cost_order_on_reorders_chain_by_ttft(monkeypatch):
    monkeypatch.setenv("COST_KEY", "k")
    wrapped = _wrap_cloud(EchoLLMStub(), _cost_llm_cfg(cost_order=True))
    # Flag on: cheapest/fastest (lowest ttft_ms) races first; same multiset.
    assert _chain_models(wrapped) == ["model-fast", "model-mid", "model-slow"]


class EchoLLMStub:
    """Minimal local LLM the wrap-cloud chains race against (never invoked)."""

    def generate(self, prompt, *, system=None, images=None) -> str:
        return prompt

    def stream(self, prompt, *, system=None, images=None) -> Iterator[str]:
        yield prompt


# --- EWMA folds only when the LOCAL tier answered (P4 low) ------------------
#
# mark_first_token forwards a fold_local_ttft flag to the recorder so a cloud
# hedge winner is still recorded but kept out of the LOCAL TTFT EWMA.


def test_mark_first_token_folds_local_sample_by_default():
    clock = FakeClock()
    rec = MetricsRecorder(clock=clock)
    rec.mark(ASR_FINAL)
    clock.t = 0.4
    list(mark_first_token(iter(["tok"]), rec))  # default fold_local_ttft=True
    assert rec.recent_ttft_ms() == 400.0


def test_mark_first_token_skips_ewma_when_not_local():
    clock = FakeClock()
    rec = MetricsRecorder(clock=clock)
    rec.mark(ASR_FINAL)
    clock.t = 0.4
    # A cloud hedge won: stamp the turn but do NOT fold into the local EWMA.
    list(mark_first_token(iter(["tok"]), rec, fold_local_ttft=False))
    assert rec.recent_ttft_ms() is None  # local headroom signal not mislabeled
    # The stamp itself is still recorded (recording is unaffected by the gate).
    [record] = rec.records()
    assert record.final_to_first_token == 0.4


def test_assistant_cloud_main_tier_does_not_fold_local_ewma():
    # A cloud-racing HedgeLLM on the main tier answers a borderline-escalated
    # turn; its first-token latency must NOT pollute the LOCAL TTFT EWMA.
    cloud = RecordingLLM("cloud")
    main = HedgeLLM(local=RecordingLLM("local"), cloud=cloud, hedge_delay_ms=0)
    fast = RecordingLLM("fast")
    rec = MetricsRecorder()
    rec.mark(ASR_FINAL)  # a real open turn -> the fold WOULD fire if not gated
    registry = attach_llm_capabilities(
        CapabilityRegistry(), main, fast_llm=fast, recorder=rec,
    )
    # Force the main tier so the cloud-wrapped model answers.
    registry.invoke(
        "assistant.answer", "explain why", {"mode": "research"}
    )
    assert rec.recent_ttft_ms() is None


def test_assistant_local_fast_tier_folds_local_ewma():
    # The fast tier is always purely local, so its first-token latency DOES
    # fold into the EWMA (the headroom signal we actually want).
    main = RecordingLLM("main")
    fast = RecordingLLM("fast")
    rec = MetricsRecorder()
    rec.mark(ASR_FINAL)  # open a turn so the first-token fold has an anchor
    registry = attach_llm_capabilities(
        CapabilityRegistry(), main, fast_llm=fast, recorder=rec,
    )
    registry.invoke("assistant.answer", "what time is it")  # -> fast tier
    assert rec.recent_ttft_ms() is not None


# --- SystemMonitor load snapshot into the live signal (P4 follow-up) --------
#
# load_snapshot feeds context['live']['load']; gated behind live_routing and
# clamped/no-op in live_nudge so a missing reading never starves local.


def test_load_nudge_clamped_in_live_nudge():
    # High load saturates at the cap; load is clamped to [0,1] (a >1 reading
    # cannot push the nudge past the cap), and a missing/garbage load is zero.
    assert live_nudge({"load": 0.99}) == 0.25  # saturated
    assert live_nudge({"load": 5.0}) == 0.25   # clamped, not amplified
    assert live_nudge({"load": 0.5}) == 0.0    # below the load floor
    assert live_nudge({"load": "hot"}) == 0.0  # garbage -> no nudge


def test_load_snapshot_off_by_default_keeps_static_decision():
    main = RecordingLLM("main")
    fast = RecordingLLM("fast")
    # A pegged-load snapshot is available, but live_routing defaults off, so the
    # borderline query stays on fast (behaviour byte-identical to no signal).
    registry = attach_llm_capabilities(
        CapabilityRegistry(), main, fast_llm=fast, load_snapshot=lambda: 0.99,
    )
    result = registry.invoke("assistant.answer", _BORDERLINE_QUERY)
    assert result.data["route"] == FAST
    assert fast.prompts and not main.prompts


def test_load_snapshot_on_with_high_load_escalates_to_main():
    main = RecordingLLM("main")
    fast = RecordingLLM("fast")
    registry = attach_llm_capabilities(
        CapabilityRegistry(), main, fast_llm=fast,
        load_snapshot=lambda: 0.99, live_routing=True,
    )
    result = registry.invoke("assistant.answer", _BORDERLINE_QUERY)
    assert result.data["route"] == MAIN
    assert main.prompts and not fast.prompts


def test_load_snapshot_none_reading_does_not_starve_local():
    # live_routing on but the snapshot returns None (no telemetry): no nudge,
    # borderline stays fast -- a missing reading can never starve local.
    main = RecordingLLM("main")
    fast = RecordingLLM("fast")
    registry = attach_llm_capabilities(
        CapabilityRegistry(), main, fast_llm=fast,
        load_snapshot=lambda: None, live_routing=True,
    )
    result = registry.invoke("assistant.answer", _BORDERLINE_QUERY)
    assert result.data["route"] == FAST


def test_load_snapshot_raising_is_swallowed():
    # A snapshot that raises must never break the turn (best-effort signal).
    main = RecordingLLM("main")
    fast = RecordingLLM("fast")

    def boom() -> float:
        raise RuntimeError("psutil exploded")

    registry = attach_llm_capabilities(
        CapabilityRegistry(), main, fast_llm=fast,
        load_snapshot=boom, live_routing=True,
    )
    result = registry.invoke("assistant.answer", _BORDERLINE_QUERY)
    assert result.data["route"] == FAST  # static decision stands


def test_system_monitor_load_fraction_reads_last_sample():
    from core.sysinfo import SystemMonitor

    gpu_lo = {"util_percent": 5.0, "mem_used_mb": 100.0, "mem_total_mb": 24000.0}
    gpu_hi = {"util_percent": 30.0, "mem_used_mb": 200.0, "mem_total_mb": 24000.0}
    seq = [
        {"t": 1, "cpu_percent": 10.0, "gpu": [gpu_lo]},
        {"t": 2, "cpu_percent": 90.0, "gpu": [gpu_hi]},
    ]
    it = iter(seq)
    mon = SystemMonitor(interval=10_000, sampler=lambda: next(it))
    assert mon.load_fraction() is None  # no sample yet
    mon.start()  # baseline = seq[0] -> max(10,5)/100
    assert mon.load_fraction() == 0.10
    mon.stop()  # final = seq[1] -> max(90,30)/100
    assert mon.load_fraction() == 0.90


def test_system_monitor_load_fraction_none_without_telemetry():
    from core.sysinfo import SystemMonitor

    mon = SystemMonitor(interval=10_000, sampler=lambda: {"t": 1, "gpu": None})
    mon.start()
    assert mon.load_fraction() is None  # no cpu/gpu numbers -> no signal


# --- Dynamic hedge delay override + floor (Task 4) --------------------------
#
# A slow/loaded local tier shortens the per-call hedge delay so the cloud race
# starts sooner; clamped to a floor so it can never reach zero/negative, and a
# no-op (None) when off / signal absent / good.


def test_dynamic_hedge_delay_none_without_signal():
    # No live signal, empty mapping, or garbage -> keep the static delay.
    assert dynamic_hedge_delay_ms(None, 150) is None
    assert dynamic_hedge_delay_ms({}, 150) is None
    assert dynamic_hedge_delay_ms("nope", 150) is None
    assert dynamic_hedge_delay_ms({"ttft_ms": "bad"}, 150) is None


def test_dynamic_hedge_delay_none_for_snappy_local():
    # A good (snappy) signal must not shorten the delay -> no override.
    assert dynamic_hedge_delay_ms({"ttft_ms": 400}, 150) is None
    assert dynamic_hedge_delay_ms({"load": 0.5}, 150) is None


def test_dynamic_hedge_delay_shortens_when_local_slow():
    # A slow local tier pulls the cloud race forward: a strictly smaller delay.
    shortened = dynamic_hedge_delay_ms({"ttft_ms": 1600}, 150)
    assert shortened is not None
    assert HEDGE_DELAY_FLOOR_MS <= shortened < 150


def test_dynamic_hedge_delay_clamped_to_floor():
    # A pegged signal saturates the fraction; the result is clamped to the
    # floor -- never zero/negative (which would be a full race every turn).
    floored = dynamic_hedge_delay_ms({"ttft_ms": 99999, "load": 1.0}, 150)
    assert floored == HEDGE_DELAY_FLOOR_MS
    assert floored > 0


def test_dynamic_hedge_delay_no_override_when_base_at_floor():
    # If the static delay is already at/below the floor there's nothing to
    # shorten, so the static delay stands (we never lengthen it).
    assert dynamic_hedge_delay_ms({"ttft_ms": 99999}, HEDGE_DELAY_FLOOR_MS) is None
    assert dynamic_hedge_delay_ms({"ttft_ms": 99999}, 0) is None


class HedgeDelaySpy(HedgeLLM):
    """HedgeLLM that records the resolved per-call hedge_delay_ms it streamed
    with, so the capability wiring of the dynamic override is observable."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.seen_hedge_delay_ms: object = "unset"

    def stream(self, prompt, *, system=None, images=None, hedge_delay_ms=None):
        self.seen_hedge_delay_ms = hedge_delay_ms
        return super().stream(prompt, system=system, images=images, hedge_delay_ms=hedge_delay_ms)


def test_assistant_passes_dynamic_hedge_override_when_slow(monkeypatch):
    # live_routing on + a slow local sample -> the assistant forwards a
    # shortened (floored, positive) hedge_delay_ms into the cloud-racing model.
    cloud = RecordingLLM("cloud")
    model = HedgeDelaySpy(
        local=RecordingLLM("local"), cloud=cloud, hedge_delay_ms=150,
    )
    rec = MetricsRecorder()
    rec._observe_ttft_ms(3000.0)  # slow local tier on record
    registry = attach_llm_capabilities(
        CapabilityRegistry(), model, recorder=rec, live_routing=True,
    )
    registry.invoke("assistant.answer", "explain why", {"mode": "research"})
    assert isinstance(model.seen_hedge_delay_ms, int)
    assert HEDGE_DELAY_FLOOR_MS <= model.seen_hedge_delay_ms < 150


def test_assistant_no_hedge_override_when_live_routing_off():
    # Default (off): the per-call override is never passed, so the constructor
    # static delay stands (seen value stays the contract default None).
    cloud = RecordingLLM("cloud")
    model = HedgeDelaySpy(
        local=RecordingLLM("local"), cloud=cloud, hedge_delay_ms=150,
    )
    rec = MetricsRecorder()
    rec._observe_ttft_ms(3000.0)
    registry = attach_llm_capabilities(
        CapabilityRegistry(), model, recorder=rec,  # live_routing off
    )
    registry.invoke("assistant.answer", "explain why", {"mode": "research"})
    assert model.seen_hedge_delay_ms is None
