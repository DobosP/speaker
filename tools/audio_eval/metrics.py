"""Pure, aggregate-safe metrics for deterministic AEC/DTD replay.

The helpers in this module deliberately accept only non-empty, finite,
one-dimensional NumPy floating-point arrays.  That keeps accidental channel
mixing, integer PCM, and NaN/Inf observations from silently entering a public
evaluation report.  No helper accepts or returns source paths, case IDs, or
transcript text.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass
import math
from numbers import Real
from typing import Final, Sequence

import numpy as np
from numpy.typing import NDArray


DB_MIN: Final = -120.0
DB_MAX: Final = 120.0
ORACLE_FRAME_SAMPLES: Final = 1_600
ORACLE_MIN_ACTIVE_FRAMES: Final = 10
ORACLE_ABSOLUTE_RMS_FLOOR: Final = 1e-5
ORACLE_RELATIVE_RMS_FACTOR: Final = 0.1

FloatSamples = NDArray[np.float64]


def strict_mono_float(samples: object, *, name: str = "samples") -> FloatSamples:
    """Return a float64 view/copy of strict finite mono floating-point PCM.

    Lists, scalars, integer PCM, empty arrays, multichannel arrays, and arrays
    containing non-finite values are rejected rather than coerced.
    """

    if (
        not isinstance(samples, np.ndarray)
        or samples.ndim != 1
        or samples.size == 0
        or samples.dtype.kind != "f"
    ):
        raise ValueError(f"invalid {name}: expected non-empty mono float array")
    result = np.asarray(samples, dtype=np.float64)
    if not bool(np.all(np.isfinite(result))):
        raise ValueError(f"invalid {name}: samples must be finite")
    return result


def _scaled_mean_square(samples: FloatSamples) -> tuple[float, float]:
    """Return ``(peak_scale, mean_square_of_scaled_samples)`` safely."""

    scale = float(np.max(np.abs(samples)))
    if scale == 0.0:
        return 0.0, 0.0
    normalized = samples / scale
    mean_square = float(np.dot(normalized, normalized) / normalized.size)
    if not math.isfinite(mean_square) or mean_square <= 0.0:
        raise ValueError("audio energy is not representable")
    return scale, mean_square


def signal_power(samples: object) -> float:
    """Return arithmetic mean-square power for strict mono float samples."""

    value = strict_mono_float(samples)
    scale, normalized_power = _scaled_mean_square(value)
    if scale == 0.0:
        return 0.0
    # Squaring ``scale`` first can overflow even when sparse input makes the
    # final mean square representable.  The RMS intermediate is never larger
    # than the finite peak and is therefore the safe order of operations.
    rms = scale * math.sqrt(normalized_power)
    result = rms * rms
    if not math.isfinite(result) or result == 0.0:
        raise ValueError("audio power is not representable")
    return result


def signal_rms(samples: object) -> float:
    """Return root-mean-square amplitude for strict mono float samples."""

    value = strict_mono_float(samples)
    scale, normalized_power = _scaled_mean_square(value)
    if scale == 0.0:
        return 0.0
    result = scale * math.sqrt(normalized_power)
    if not math.isfinite(result) or result == 0.0:
        raise ValueError("audio RMS is not representable")
    return result


def _nonnegative_finite_scalar(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"invalid {name}")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"invalid {name}")
    return result


def capped_db_ratio(numerator_power: object, denominator_power: object) -> float:
    """Return a power ratio in dB, deterministically capped to [-120, 120].

    A positive value over an exact zero denominator represents perfect removal
    and returns +120 dB.  An exact zero numerator over positive energy returns
    -120 dB.  The undefined 0/0 case is rejected.
    """

    numerator = _nonnegative_finite_scalar(
        numerator_power,
        name="dB numerator",
    )
    denominator = _nonnegative_finite_scalar(
        denominator_power,
        name="dB denominator",
    )
    if numerator == 0.0 and denominator == 0.0:
        raise ValueError("undefined zero-over-zero dB ratio")
    if numerator == 0.0:
        return DB_MIN
    if denominator == 0.0:
        return DB_MAX
    value = 10.0 * (math.log10(numerator) - math.log10(denominator))
    if not math.isfinite(value):
        raise ValueError("dB ratio is not finite")
    return min(DB_MAX, max(DB_MIN, value))


def _aligned_pair(
    left: object,
    right: object,
    *,
    left_name: str,
    right_name: str,
) -> tuple[FloatSamples, FloatSamples]:
    left_value = strict_mono_float(left, name=left_name)
    right_value = strict_mono_float(right, name=right_name)
    if left_value.shape != right_value.shape:
        raise ValueError("audio signals must have identical sample geometry")
    return left_value, right_value


def _finite_product_ratio(
    numerator_factors: Sequence[float],
    denominator_factors: Sequence[float],
    *,
    name: str,
) -> float:
    """Combine finite factors without overflowing an intermediate ratio."""

    sign = 1.0
    mantissa = 1.0
    exponent = 0
    for value in numerator_factors:
        if not math.isfinite(value):
            raise ValueError(f"{name} is not finite")
        if value == 0.0:
            return 0.0
        if value < 0.0:
            sign = -sign
        factor_mantissa, factor_exponent = math.frexp(abs(value))
        mantissa *= factor_mantissa
        exponent += factor_exponent
        mantissa, adjustment = math.frexp(mantissa)
        exponent += adjustment
    for value in denominator_factors:
        if not math.isfinite(value) or value == 0.0:
            raise ValueError(f"{name} denominator is invalid")
        if value < 0.0:
            sign = -sign
        factor_mantissa, factor_exponent = math.frexp(abs(value))
        mantissa /= factor_mantissa
        exponent -= factor_exponent
        mantissa, adjustment = math.frexp(mantissa)
        exponent += adjustment
    try:
        result = math.ldexp(sign * mantissa, exponent)
    except OverflowError:
        raise ValueError(f"{name} is not representable") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} is not representable")
    return result


def projection_gain(estimate: object, target: object) -> float:
    """Return the signed least-squares gain projecting ``estimate`` on target."""

    estimate_value, target_value = _aligned_pair(
        estimate,
        target,
        left_name="estimate",
        right_name="target",
    )
    target_scale = float(np.max(np.abs(target_value)))
    if target_scale == 0.0:
        raise ValueError("projection target has zero energy")
    estimate_scale = float(np.max(np.abs(estimate_value)))
    if estimate_scale == 0.0:
        return 0.0

    normalized_target = target_value / target_scale
    normalized_estimate = estimate_value / estimate_scale
    target_energy = float(np.dot(normalized_target, normalized_target))
    cross_energy = float(np.dot(normalized_estimate, normalized_target))
    return _finite_product_ratio(
        (estimate_scale, cross_energy),
        (target_scale, target_energy),
        name="projection gain",
    )


def absolute_cosine_similarity(estimate: object, target: object) -> float:
    """Return absolute cosine similarity, treating sign inversion as aligned."""

    estimate_value, target_value = _aligned_pair(
        estimate,
        target,
        left_name="estimate",
        right_name="target",
    )
    estimate_scale = float(np.max(np.abs(estimate_value)))
    target_scale = float(np.max(np.abs(target_value)))
    if estimate_scale == 0.0 or target_scale == 0.0:
        raise ValueError("cosine similarity requires two nonzero signals")
    normalized_estimate = estimate_value / estimate_scale
    normalized_target = target_value / target_scale
    numerator = abs(float(np.dot(normalized_estimate, normalized_target)))
    denominator = math.sqrt(
        float(np.dot(normalized_estimate, normalized_estimate))
        * float(np.dot(normalized_target, normalized_target))
    )
    if denominator == 0.0 or not math.isfinite(denominator):
        raise ValueError("cosine similarity is not representable")
    result = numerator / denominator
    if not math.isfinite(result):
        raise ValueError("cosine similarity is not finite")
    # Floating-point dot products can overshoot the mathematical bound by a
    # handful of ulps for long, identical vectors.
    return min(1.0, max(0.0, result))


def si_sdr_db(estimate: object, target: object) -> float:
    """Return scale-invariant SDR in dB using a signed target projection."""

    estimate_value, target_value = _aligned_pair(
        estimate,
        target,
        left_name="estimate",
        right_name="target",
    )
    if signal_power(estimate_value) == 0.0:
        raise ValueError("SI-SDR estimate has zero energy")
    gain = projection_gain(estimate_value, target_value)
    projected = gain * target_value
    residual = estimate_value - projected
    return capped_db_ratio(signal_power(projected), signal_power(residual))


def _post_convergence(
    *signals: tuple[object, str],
    convergence_samples: int,
) -> tuple[FloatSamples, ...]:
    if type(convergence_samples) is not int or convergence_samples < 0:
        raise ValueError("convergence_samples must be a nonnegative integer")
    validated = tuple(strict_mono_float(value, name=name) for value, name in signals)
    sample_count = validated[0].size
    if any(value.size != sample_count for value in validated[1:]):
        raise ValueError("audio signals must have identical sample geometry")
    if convergence_samples >= sample_count:
        raise ValueError("convergence exclusion leaves no scored samples")
    return tuple(value[convergence_samples:] for value in validated)


def echo_attenuation_db(
    echo_signal: object,
    residual_echo: object,
    *,
    convergence_samples: int,
) -> float:
    """Return post-convergence echo attenuation (ERLE-like power ratio)."""

    echo_tail, residual_tail = _post_convergence(
        (echo_signal, "echo signal"),
        (residual_echo, "residual echo"),
        convergence_samples=convergence_samples,
    )
    echo_power = signal_power(echo_tail)
    if echo_power == 0.0:
        raise ValueError("post-convergence echo signal has zero energy")
    return capped_db_ratio(echo_power, signal_power(residual_tail))


def interference_reduction_db(
    mixture: object,
    processed: object,
    scaled_near_target: object,
    *,
    convergence_samples: int = 0,
) -> float:
    """Return interference reduction against one exact scaled near target.

    The numerator is ``sum((mixture - target) ** 2)`` and the denominator is
    ``sum((processed - target) ** 2)`` over the identical post-convergence
    samples.  Target projection is intentionally a separate metric.
    """

    mixture_tail, processed_tail, target_tail = _post_convergence(
        (mixture, "mixture"),
        (processed, "processed signal"),
        (scaled_near_target, "scaled near target"),
        convergence_samples=convergence_samples,
    )
    if signal_power(target_tail) == 0.0:
        raise ValueError("scaled near target has zero energy")
    input_interference_power = signal_power(mixture_tail - target_tail)
    output_interference_power = signal_power(processed_tail - target_tail)
    return capped_db_ratio(input_interference_power, output_interference_power)


def _finite_values(values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("percentile values must be a finite sequence")
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("percentile values must be finite numbers")
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError("percentile values must be finite numbers")
        result.append(converted)
    if not result:
        raise ValueError("percentile values cannot be empty")
    return tuple(result)


def nearest_rank_percentile(values: Sequence[float], quantile: float) -> float:
    """Return a deterministic nearest-rank percentile for finite values."""

    finite = sorted(_finite_values(values))
    if isinstance(quantile, bool) or not isinstance(quantile, Real):
        raise ValueError("quantile must be finite and within [0, 1]")
    converted_quantile = float(quantile)
    if not math.isfinite(converted_quantile) or not 0.0 <= converted_quantile <= 1.0:
        raise ValueError("quantile must be finite and within [0, 1]")
    index = max(0, math.ceil(converted_quantile * len(finite)) - 1)
    return finite[index]


@dataclass(frozen=True)
class PercentileSummary:
    """Fixed aggregate-only distribution summary."""

    count: int
    minimum: float
    p50: float
    p95: float
    maximum: float

    def __post_init__(self) -> None:
        if type(self.count) is not int or self.count <= 0:
            raise ValueError("invalid percentile count")
        ordered = tuple(
            _nonnegative_or_signed_finite(value, name="percentile summary")
            for value in (self.minimum, self.p50, self.p95, self.maximum)
        )
        if ordered != tuple(sorted(ordered)):
            raise ValueError("percentile summary is not ordered")

    def as_dict(self) -> dict[str, int | float]:
        return {
            "count": self.count,
            "min": self.minimum,
            "p50": self.p50,
            "p95": self.p95,
            "max": self.maximum,
        }


def _nonnegative_or_signed_finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"invalid {name}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"invalid {name}")
    return result


def percentile_summary(values: Sequence[float]) -> PercentileSummary:
    """Summarize finite values with deterministic nearest-rank p50 and p95."""

    finite = _finite_values(values)
    return PercentileSummary(
        count=len(finite),
        minimum=min(finite),
        p50=nearest_rank_percentile(finite, 0.50),
        p95=nearest_rank_percentile(finite, 0.95),
        maximum=max(finite),
    )


@dataclass(frozen=True)
class ActivityOracle:
    """One transcript/path-free 100 ms energy activity oracle."""

    frame_samples: int
    frame_count: int
    active_frames: int
    minimum_active_frames: int
    p95_rms: float
    threshold_rms: float
    _active_mask_input: InitVar[tuple[bool, ...]]

    def __post_init__(self, _active_mask_input: tuple[bool, ...]) -> None:
        if (
            self.frame_samples != ORACLE_FRAME_SAMPLES
            or type(self.frame_count) is not int
            or type(self.active_frames) is not int
            or type(self.minimum_active_frames) is not int
            or self.minimum_active_frames < ORACLE_MIN_ACTIVE_FRAMES
            or self.frame_count != len(_active_mask_input)
            or self.frame_count < self.minimum_active_frames
            or any(type(value) is not bool for value in _active_mask_input)
            or self.active_frames != sum(_active_mask_input)
            or self.active_frames < self.minimum_active_frames
        ):
            raise ValueError("invalid activity oracle geometry")
        p95 = _nonnegative_finite_scalar(self.p95_rms, name="activity p95 RMS")
        threshold = _nonnegative_finite_scalar(
            self.threshold_rms,
            name="activity threshold RMS",
        )
        if threshold != max(
            ORACLE_ABSOLUTE_RMS_FLOOR,
            ORACLE_RELATIVE_RMS_FACTOR * p95,
        ):
            raise ValueError("invalid activity oracle threshold")
        object.__setattr__(self, "_active_mask", _active_mask_input)

    @property
    def active_mask(self) -> tuple[bool, ...]:
        """Return the frame-level oracle without exposing it in aggregates."""

        return self._active_mask

    @property
    def active_fraction(self) -> float:
        return self.active_frames / self.frame_count

    def as_dict(self) -> dict[str, int | float]:
        """Return aggregate counts only; frame-level timing stays internal."""

        return {
            "frame_samples": self.frame_samples,
            "frame_count": self.frame_count,
            "active_frames": self.active_frames,
            "minimum_active_frames": self.minimum_active_frames,
            "active_fraction": self.active_fraction,
            "p95_rms": self.p95_rms,
            "threshold_rms": self.threshold_rms,
        }


def oracle_activity_mask(
    samples: object,
    *,
    min_active_frames: int = ORACLE_MIN_ACTIVE_FRAMES,
) -> ActivityOracle:
    """Build the fixed 16 kHz/100 ms energy oracle for one source track.

    Geometry must be an exact multiple of 1,600 samples.  Activity uses
    ``RMS >= max(1e-5, 0.1 * nearest-rank-p95(frame RMS))``.  A track with
    fewer than the requested active frames is not an evaluable oracle and is
    rejected; callers may raise the requirement but may not lower it below 10.
    """

    value = strict_mono_float(samples)
    if (
        type(min_active_frames) is not int
        or min_active_frames < ORACLE_MIN_ACTIVE_FRAMES
    ):
        raise ValueError("min_active_frames must be an integer of at least 10")
    if value.size % ORACLE_FRAME_SAMPLES != 0:
        raise ValueError("activity oracle requires exact 1,600-sample geometry")
    frame_count = value.size // ORACLE_FRAME_SAMPLES
    if frame_count < min_active_frames:
        raise ValueError("activity oracle has too few complete frames")

    framed = value.reshape(frame_count, ORACLE_FRAME_SAMPLES)
    frame_rms = tuple(signal_rms(frame) for frame in framed)
    p95_rms = nearest_rank_percentile(frame_rms, 0.95)
    threshold_rms = max(
        ORACLE_ABSOLUTE_RMS_FLOOR,
        ORACLE_RELATIVE_RMS_FACTOR * p95_rms,
    )
    active_mask = tuple(rms >= threshold_rms for rms in frame_rms)
    active_frames = sum(active_mask)
    if active_frames < min_active_frames:
        raise ValueError("activity oracle has too few active frames")
    return ActivityOracle(
        frame_samples=ORACLE_FRAME_SAMPLES,
        frame_count=frame_count,
        active_frames=active_frames,
        minimum_active_frames=min_active_frames,
        p95_rms=p95_rms,
        threshold_rms=threshold_rms,
        _active_mask_input=active_mask,
    )
