"""Spec-simulation harness: model-fit, latency math, budgets, HTML rendering."""
from __future__ import annotations

from tools.specsim.report import render
from tools.specsim.simulate import (
    BARGE_IN_BUDGET,
    FIRST_AUDIO_BUDGET,
    SCENARIOS,
    Scenario,
    classify,
    simulate_turn,
)
from tools.specsim.specs import CATALOG, MachineSpec


def _spec(**kw) -> MachineSpec:
    base = dict(
        name="t",
        platform="Linux",
        accelerator="x",
        cores=8,
        ram_gb=16,
        model_budget_gb=8.0,
        fast_model="minicpm5-1b:q8",
        main_model="gemma3:4b",
        stt_partial_interval_sec=0.1,
        stt_endpoint_delay_sec=0.5,
        llm_ttft_sec=0.2,
        llm_per_token_sec=0.01,
        tts_ttfa_sec=0.1,
        tts_realtime_factor=0.3,
        barge_in_stop_sec=0.3,
    )
    base.update(kw)
    return MachineSpec(**base)


# --- model fit ---------------------------------------------------------------


def test_fit_status_good_tight_fail():
    assert _spec(model_budget_gb=2.0).fit_status("minicpm5-1b:q8") == "good"
    assert _spec(model_budget_gb=10.0).fit_status("gemma3:4b") == "good"  # 3.3 of 10
    assert _spec(model_budget_gb=9.0).fit_status("gemma3:12b") == "tight"  # 8.1 of 9 (>80%)
    assert _spec(model_budget_gb=5.0).fit_status("gemma3:12b") == "fail"  # 8.1 > 5


def test_largest_fitting_model_tracks_budget():
    assert _spec(model_budget_gb=16).largest_fitting_model() == "gemma3:12b"
    assert _spec(model_budget_gb=2.0).largest_fitting_model() == "minicpm5-1b:q8"
    assert _spec(model_budget_gb=0.5).largest_fitting_model() is None


def test_tokens_per_sec_derived_from_per_token():
    spec = _spec(llm_per_token_sec=0.01)
    assert spec.fast_tokens_per_sec == 100.0
    assert spec.tokens_per_sec == 100.0  # compatibility alias


def test_configured_model_is_fast_path_compatibility_alias():
    spec = _spec(fast_model="minicpm5-1b:q4", main_model="gemma3:12b")
    assert spec.configured_model == "minicpm5-1b:q4"
    assert spec.role_models == (
        ("fast", "minicpm5-1b:q4"),
        ("main", "gemma3:12b"),
    )


def test_configured_roles_fit_checks_fast_and_main():
    assert _spec(
        model_budget_gb=10.0,
        fast_model="minicpm5-1b:q8",
        main_model="gemma3:12b",
    ).configured_roles_fit()
    assert not _spec(
        model_budget_gb=5.0,
        fast_model="minicpm5-1b:q8",
        main_model="gemma3:12b",
    ).configured_roles_fit()


def test_configured_roles_fit_counts_distinct_weights_and_dedupes_shared_model():
    hybrid = _spec(
        model_budget_gb=8.5,
        fast_model="minicpm5-1b:q8",
        main_model="gemma3:12b",
    )
    shared = _spec(
        model_budget_gb=0.7,
        fast_model="minicpm5-1b:q4",
        main_model="minicpm5-1b:q4",
    )

    assert hybrid.fits(hybrid.fast_model) and hybrid.fits(hybrid.main_model)
    assert hybrid.configured_footprint_gb == 9.25
    assert not hybrid.configured_roles_fit()
    assert shared.configured_footprint_gb == 0.688
    assert shared.configured_roles_fit()


# --- latency model -----------------------------------------------------------


def test_first_audio_latency_is_sum_of_post_speech_stages():
    spec = _spec()
    sc = Scenario("s", "", user_words=5, reply_words=10)
    r = simulate_turn(spec, sc)
    reply_tokens = round(10 * 1.3)  # 13
    expected = (
        spec.stt_endpoint_delay_sec
        + spec.llm_ttft_sec
        + reply_tokens * spec.llm_per_token_sec
        + spec.tts_ttfa_sec
    )
    assert abs(r.first_audio_latency - expected) < 1e-6
    assert r.barge_in_stop is None


def test_slower_device_has_higher_latency():
    fast = _spec(llm_per_token_sec=0.01, llm_ttft_sec=0.15)
    slow = _spec(llm_per_token_sec=0.20, llm_ttft_sec=2.0)
    sc = SCENARIOS[1]  # research (long reply amplifies per-token cost)
    assert simulate_turn(slow, sc).first_audio_latency > simulate_turn(fast, sc).first_audio_latency


def test_barge_in_scenario_reports_stop_latency():
    r = simulate_turn(_spec(barge_in_stop_sec=0.42), SCENARIOS[2])
    assert r.barge_in_stop == 0.42
    # an interrupted reply finishes sooner than it would have fully played
    assert r.response_complete < simulate_turn(_spec(), SCENARIOS[1]).response_complete


def test_segments_are_contiguous():
    r = simulate_turn(_spec(), SCENARIOS[0])
    for prev, nxt in zip(r.segments, r.segments[1:]):
        assert abs(prev.end - nxt.start) < 1e-9
    assert abs(r.segments[-1].end - r.total) < 1e-9


def test_classify_thresholds():
    assert classify(1.0, FIRST_AUDIO_BUDGET) == "good"
    assert classify(2.0, FIRST_AUDIO_BUDGET) == "ok"
    assert classify(3.0, FIRST_AUDIO_BUDGET) == "fail"
    assert classify(0.3, BARGE_IN_BUDGET) == "good"


# --- catalog + rendering -----------------------------------------------------


def test_catalog_is_a_descending_gradient():
    quick = SCENARIOS[0]
    latencies = [simulate_turn(s, quick).first_audio_latency for s in CATALOG]
    # the 4090 laptop (first) must be the snappiest; web (last) the slowest
    assert latencies[0] == min(latencies)
    assert latencies[-1] == max(latencies)


def test_every_catalog_model_role_actually_fits():
    for spec in CATALOG:
        assert spec.configured_roles_fit(), spec.name
        for role, model in spec.role_models:
            assert spec.fits(model), f"{spec.name}: {role}={model}"


def test_catalog_uses_hybrid_desktop_and_shared_phone_roles():
    desktops = CATALOG[:3]
    phones = CATALOG[3:5]
    assert all(spec.fast_model == "minicpm5-1b:q8" for spec in desktops)
    assert all(spec.fast_model != spec.main_model for spec in desktops)
    assert all(
        spec.fast_model == spec.main_model == "minicpm5-1b:q4"
        for spec in phones
    )


def test_render_produces_self_contained_html():
    out = render(CATALOG, SCENARIOS)
    assert out.startswith("<!doctype html>")
    assert "<svg" in out and "</svg>" in out
    assert "http://" not in out and "https://" not in out  # no external assets
    for spec in CATALOG:
        assert spec.name in out
    assert "Responsiveness matrix" in out and "Model fit per device" in out
    assert "Fast / ordinary model" in out and "Main / complex model" in out
    assert "Estimated fast-path speed" in out
    assert "main-tier latency claims" in out
    assert "minicpm5-1b:q4 (shared)" in out
