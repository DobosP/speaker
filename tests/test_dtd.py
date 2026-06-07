"""Device-adaptive fused double-talk detector (core/engines/_dtd.py).

Pins the properties that make it work on ANY open speaker without a fixed margin:
the per-feature control chart self-calibrates to the device's echo, a steady echo
never fires (no self-interrupt), a single-feature spike (nonlinear-echo coherence)
can't fire alone, and a real talk-over -- all three features elevated together --
fires after the confirm-frame hysteresis at a NORMAL elevation (not a shout),
because the z-score bar is mean + k*sigma of THIS device's echo, not a fixed dB.
Pure floats, no audio.
"""
from __future__ import annotations

import pytest

from core.engines._dtd import AdaptiveDTD, Chart


# --- Chart ------------------------------------------------------------------


def test_chart_warmup_seeds_running_mean():
    c = Chart(warmup=3, provisional=0.5)
    assert c.warming
    for x in (0.10, 0.20, 0.30):
        c.seed(x)
    assert not c.warming
    assert c.mean == pytest.approx(0.20)  # running mean of the warm-up blocks


def test_chart_z_is_zero_at_mean_and_positive_above():
    c = Chart(warmup=1)
    c.seed(0.1)
    for _ in range(50):
        c.update_echo(0.1)  # steady echo
    assert c.z(0.1) == pytest.approx(0.0, abs=1e-6)
    assert c.z(0.4) > 0.0


def test_chart_sigma_floored_relative_to_mean_no_explosion():
    # A perfectly steady feature has zero measured variance; the relative sigma
    # floor keeps z finite/bounded instead of exploding to infinity.
    c = Chart(warmup=1, rel_floor=0.15)
    c.seed(0.2)
    for _ in range(50):
        c.update_echo(0.2)
    assert c.sigma == pytest.approx(0.15 * 0.2, rel=0.1)
    assert c.z(0.2 * 1.15) == pytest.approx(1.0, abs=0.2)  # ~+1 sigma, finite


def test_chart_variance_is_upward_only():
    c = Chart(warmup=1)
    c.seed(0.1)
    base = c.sigma
    for _ in range(20):
        c.update_echo(0.02)  # downward excursions must not inflate variance
    # mean drifted down but sigma stayed near the relative floor (no upward spread)
    assert c.sigma <= base * 1.5 + 1e-6


# --- AdaptiveDTD ------------------------------------------------------------


def _warm(dtd, raw=0.10, resid=0.02, coh=0.8, n=5):
    for _ in range(n):
        assert dtd.decide(raw, resid, coh) is False  # warming -> never fires


def test_warmup_seeds_and_never_fires():
    dtd = AdaptiveDTD(warmup_frames=4, confirm_frames=2, k=5.0)
    for _ in range(4):
        assert dtd.decide(0.5, 0.4, 0.99) is False  # even a loud "talk-over" can't fire mid-warmup


def test_steady_echo_never_self_interrupts():
    dtd = AdaptiveDTD(warmup_frames=3, confirm_frames=2, k=5.0)
    _warm(dtd, n=3)
    for _ in range(40):
        assert dtd.decide(0.10, 0.02, 0.8) is False  # steady echo -> D~0 -> no fire


def test_single_feature_coherence_spike_does_not_fire():
    # The nonlinear-echo failure mode: coherence incoherence spikes on echo while
    # raw + residual stay at the echo floor. With coh weighted 0.5 the lone z
    # cannot reach K, so it does NOT self-interrupt (the bug that killed approach 1).
    dtd = AdaptiveDTD(warmup_frames=3, confirm_frames=2, k=5.0, weights=(1.0, 1.0, 0.5))
    _warm(dtd, n=3)
    for _ in range(10):
        dtd.decide(0.10, 0.02, 0.8)
    res = [dtd.decide(0.10, 0.02, 0.99) for _ in range(5)]  # only coherence elevated
    assert not any(res)


def test_real_talkover_fires_at_normal_elevation_after_confirm():
    # A real talk-over lifts ALL THREE features together. The summed z-scores cross
    # K at a NORMAL elevation (raw 0.1->0.30 is only ~+9.5 dB, no shout), and it
    # fires after confirm_frames consecutive blocks.
    dtd = AdaptiveDTD(warmup_frames=3, confirm_frames=2, k=5.0, weights=(1.0, 1.0, 0.5))
    _warm(dtd, n=3)
    for _ in range(10):
        dtd.decide(0.10, 0.02, 0.8)  # establish the echo baseline
    v = [dtd.decide(0.30, 0.12, 0.95) for _ in range(3)]
    assert v[0] is False and v[1] is True  # confirm_frames=2 -> fires on the 2nd


def test_candidate_run_freezes_charts():
    # While a barge is being confirmed the charts must NOT learn the talk-over (it
    # can't be allowed to raise its own bar).
    dtd = AdaptiveDTD(warmup_frames=3, confirm_frames=5, k=5.0)
    _warm(dtd, n=3)
    for _ in range(10):
        dtd.decide(0.10, 0.02, 0.8)
    mean_before = dtd._raw.mean
    for _ in range(3):
        dtd.decide(0.40, 0.20, 0.97)  # candidate frames (D>K) -> frozen
    assert dtd._raw.mean == pytest.approx(mean_before)  # unchanged


def test_reset_rearms_warmup():
    dtd = AdaptiveDTD(warmup_frames=2, confirm_frames=2)
    _warm(dtd, n=2)
    dtd.decide(0.1, 0.02, 0.8)  # post-warmup
    dtd.reset()
    assert dtd.decide(0.1, 0.02, 0.8) is False and dtd._raw.warming  # warming again
