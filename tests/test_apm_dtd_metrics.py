from __future__ import annotations

from dataclasses import asdict
import math

import numpy as np
import pytest

from tools.audio_eval.metrics import (
    DB_MAX,
    DB_MIN,
    ORACLE_FRAME_SAMPLES,
    absolute_cosine_similarity,
    capped_db_ratio,
    echo_attenuation_db,
    interference_reduction_db,
    nearest_rank_percentile,
    oracle_activity_mask,
    percentile_summary,
    projection_gain,
    si_sdr_db,
    signal_power,
    signal_rms,
    strict_mono_float,
)


def _mono(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _constant_frames(amplitudes: list[float]) -> np.ndarray:
    return np.concatenate(
        [
            np.full(ORACLE_FRAME_SAMPLES, amplitude, dtype=np.float32)
            for amplitude in amplitudes
        ]
    )


def test_power_and_rms_accept_only_strict_finite_mono_float_pcm():
    samples = _mono([3.0, 4.0, -3.0, -4.0])

    assert signal_power(samples) == pytest.approx(12.5)
    assert signal_rms(samples) == pytest.approx(math.sqrt(12.5))
    assert signal_power(np.zeros(8, dtype=np.float64)) == 0.0
    assert signal_rms(np.zeros(8, dtype=np.float32)) == 0.0
    assert strict_mono_float(samples).dtype == np.float64

    invalid = (
        [1.0, 2.0],
        np.asarray([1, 2], dtype=np.int16),
        np.asarray([], dtype=np.float32),
        np.asarray([[1.0], [2.0]], dtype=np.float32),
        _mono([1.0, float("nan")]),
        _mono([1.0, float("inf")]),
    )
    for value in invalid:
        with pytest.raises(ValueError):
            signal_power(value)


def test_power_avoids_an_overflowing_peak_square_when_mean_is_finite():
    sparse_extreme = np.zeros(101, dtype=np.float64)
    sparse_extreme[0] = 1e155

    assert signal_rms(sparse_extreme) == pytest.approx(1e155 / math.sqrt(101))
    assert signal_power(sparse_extreme) == pytest.approx(1e308 * (100 / 101))


def test_capped_db_ratio_handles_exact_removal_and_extreme_ratios():
    assert capped_db_ratio(1.0, 1.0) == 0.0
    assert capped_db_ratio(1.0, 0.0) == DB_MAX
    assert capped_db_ratio(0.0, 1.0) == DB_MIN
    assert capped_db_ratio(1e300, 1e-300) == DB_MAX
    assert capped_db_ratio(1e-300, 1e300) == DB_MIN

    for numerator, denominator in (
        (0.0, 0.0),
        (-1.0, 1.0),
        (1.0, float("nan")),
        (True, 1.0),
    ):
        with pytest.raises(ValueError):
            capped_db_ratio(numerator, denominator)


def test_projection_cosine_and_si_sdr_cover_identity_and_sign_inversion():
    target = _mono([0.25, -0.5, 0.75, -1.0])

    assert projection_gain(target, target) == pytest.approx(1.0)
    assert projection_gain(-2.0 * target, target) == pytest.approx(-2.0)
    assert absolute_cosine_similarity(target, target) == pytest.approx(1.0)
    assert absolute_cosine_similarity(-target, target) == pytest.approx(1.0)
    assert si_sdr_db(target, target) == DB_MAX
    assert si_sdr_db(-target, target) == DB_MAX


def test_projection_family_rejects_undefined_zero_and_bad_geometry():
    target = _mono([1.0, -1.0, 0.5, -0.5])
    zeros = np.zeros_like(target)

    assert projection_gain(zeros, target) == 0.0
    with pytest.raises(ValueError):
        projection_gain(target, zeros)
    with pytest.raises(ValueError):
        absolute_cosine_similarity(zeros, target)
    with pytest.raises(ValueError):
        si_sdr_db(zeros, target)
    with pytest.raises(ValueError):
        si_sdr_db(target[:-1], target)
    with pytest.raises(ValueError):
        projection_gain(_mono([1.0, float("nan")]), _mono([1.0, 1.0]))


def test_projection_avoids_an_overflowing_scale_ratio_when_gain_is_finite():
    peak = 1e308
    estimate = np.asarray(
        [peak, -np.nextafter(peak, 0.0)],
        dtype=np.float64,
    )
    target = np.asarray([0.1, 0.1], dtype=np.float64)

    gain = projection_gain(estimate, target)

    assert math.isfinite(gain)
    assert 1e292 < gain < 1e294


def test_si_sdr_has_a_known_zero_db_orthogonal_decomposition():
    target = _mono([1.0, -1.0, 1.0, -1.0])
    orthogonal = _mono([1.0, 1.0, -1.0, -1.0])
    estimate = target + orthogonal

    assert projection_gain(estimate, target) == pytest.approx(1.0)
    assert si_sdr_db(estimate, target) == pytest.approx(0.0)


def test_echo_attenuation_excludes_exact_convergence_samples():
    echo = _mono([100.0, -100.0, 1.0, -1.0, 2.0, -2.0])
    residual = _mono([100.0, -100.0, 0.1, -0.1, 0.2, -0.2])

    assert echo_attenuation_db(
        echo,
        residual,
        convergence_samples=2,
    ) == pytest.approx(20.0)
    assert echo_attenuation_db(echo, echo, convergence_samples=2) == pytest.approx(0.0)
    assert (
        echo_attenuation_db(
            echo,
            np.zeros_like(echo),
            convergence_samples=2,
        )
        == DB_MAX
    )


def test_echo_attenuation_rejects_impossible_scoring_windows():
    echo = _mono([1.0, -1.0, 1.0, -1.0])

    for convergence_samples in (-1, True, len(echo), len(echo) + 1):
        with pytest.raises(ValueError):
            echo_attenuation_db(
                echo,
                np.zeros_like(echo),
                convergence_samples=convergence_samples,
            )
    with pytest.raises(ValueError):
        echo_attenuation_db(
            np.zeros_like(echo),
            np.zeros_like(echo),
            convergence_samples=0,
        )
    with pytest.raises(ValueError):
        echo_attenuation_db(echo, echo[:-1], convergence_samples=0)


def test_interference_reduction_uses_the_same_exact_scaled_near_target():
    target = _mono([1.0, -1.0, 2.0, -2.0])
    interference = _mono([0.5, 0.5, -0.5, -0.5])
    mixture = target + interference
    processed = target + 0.1 * interference

    assert interference_reduction_db(mixture, mixture, target) == pytest.approx(0.0)
    assert interference_reduction_db(mixture, processed, target) == pytest.approx(20.0)
    assert interference_reduction_db(mixture, target, target) == DB_MAX

    # A projection inside this metric would erase the target scaling error and
    # produce 20 dB.  The exact-target residual correctly reports less.
    scaled_processed = 0.5 * target + 0.1 * interference
    assert interference_reduction_db(mixture, scaled_processed, target) < 10.0

    prefixed_target = np.concatenate((_mono([100.0]), target))
    prefixed_mixture = np.concatenate((_mono([-100.0]), mixture))
    prefixed_processed = np.concatenate((_mono([-100.0]), processed))
    assert interference_reduction_db(
        prefixed_mixture,
        prefixed_processed,
        prefixed_target,
        convergence_samples=1,
    ) == pytest.approx(20.0)


def test_interference_reduction_rejects_absent_targets_or_interference():
    target = _mono([1.0, -1.0, 2.0, -2.0])
    zeros = np.zeros_like(target)

    with pytest.raises(ValueError):
        interference_reduction_db(target, target, target)
    with pytest.raises(ValueError):
        interference_reduction_db(target, target, zeros)
    assert interference_reduction_db(target, zeros, target) == DB_MIN
    with pytest.raises(ValueError):
        interference_reduction_db(target, target[:-1], target)


def test_nearest_rank_percentiles_and_aggregate_summary_are_deterministic():
    values = tuple(float(value) for value in range(20, 0, -1))

    assert nearest_rank_percentile(values, 0.0) == 1.0
    assert nearest_rank_percentile(values, 0.50) == 10.0
    assert nearest_rank_percentile(values, 0.95) == 19.0
    assert nearest_rank_percentile(values, 1.0) == 20.0
    assert percentile_summary(values).as_dict() == {
        "count": 20,
        "min": 1.0,
        "p50": 10.0,
        "p95": 19.0,
        "max": 20.0,
    }


@pytest.mark.parametrize(
    ("values", "quantile"),
    (
        ((), 0.5),
        ((1.0, float("nan")), 0.5),
        ((1.0, True), 0.5),
        ((1.0,), -0.1),
        ((1.0,), 1.1),
        ((1.0,), float("inf")),
    ),
)
def test_percentiles_reject_empty_nonfinite_or_impossible_inputs(values, quantile):
    with pytest.raises(ValueError):
        nearest_rank_percentile(values, quantile)


def test_activity_oracle_uses_fixed_100ms_geometry_and_relative_threshold():
    samples = _constant_frames([0.0] * 5 + [0.04] * 5 + [0.5] * 11)

    oracle = oracle_activity_mask(samples, min_active_frames=11)

    assert oracle.frame_samples == 1_600
    assert oracle.frame_count == 21
    assert oracle.p95_rms == pytest.approx(0.5)
    assert oracle.threshold_rms == pytest.approx(0.05)
    assert oracle.active_mask == (False,) * 10 + (True,) * 11
    assert oracle.active_frames == 11
    assert oracle.active_fraction == pytest.approx(11 / 21)
    assert "active_mask" not in oracle.as_dict()
    assert all("mask" not in key for key in asdict(oracle))


def test_activity_oracle_applies_absolute_floor_and_rejects_weak_tracks():
    floor_active = _constant_frames([2e-5] * 10)
    oracle = oracle_activity_mask(floor_active)

    assert oracle.p95_rms == pytest.approx(2e-5)
    assert oracle.threshold_rms == 1e-5
    assert oracle.active_frames == 10

    with pytest.raises(ValueError):
        oracle_activity_mask(np.zeros(ORACLE_FRAME_SAMPLES * 10, dtype=np.float32))
    with pytest.raises(ValueError):
        oracle_activity_mask(floor_active, min_active_frames=9)
    with pytest.raises(ValueError):
        oracle_activity_mask(floor_active, min_active_frames=True)
    with pytest.raises(ValueError):
        oracle_activity_mask(floor_active, min_active_frames=11)
    with pytest.raises(ValueError):
        oracle_activity_mask(
            np.zeros(ORACLE_FRAME_SAMPLES * 10 + 1, dtype=np.float32),
        )
    nonfinite = floor_active.copy()
    nonfinite[0] = np.nan
    with pytest.raises(ValueError):
        oracle_activity_mask(nonfinite)
