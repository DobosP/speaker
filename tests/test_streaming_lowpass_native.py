"""Synthetic equivalence/ownership tests for the optional compiled biquad."""

from __future__ import annotations

import sys

import numpy as np
import pytest

import core.audio_frontend as frontend
from core.audio_frontend import StreamingLowpass


def _scalar_filter(filter, samples):
    x = np.asarray(samples, dtype="float32").reshape(-1)
    output = np.empty_like(x)
    z1, z2 = filter._z1, filter._z2
    for index, sample in enumerate(x):
        sample = float(sample)
        value = filter._b0 * sample + z1
        z1 = filter._b1 * sample - filter._a1 * value + z2
        z2 = filter._b2 * sample - filter._a2 * value
        output[index] = value
    filter._z1, filter._z2 = z1, z2
    return output


@pytest.mark.parametrize(
    "sample_rate,cutoff,q",
    [
        (16000, 1000.0, 0.5),
        (24000, 7000.0, np.sqrt(0.5)),
        (48000, 12000.0, 1.0),
        (24000, 11760.0, 0.7),
    ],
)
def test_native_filter_matches_scalar_output_and_state_across_partitions(
    sample_rate, cutoff, q
):
    pytest.importorskip("scipy.signal")
    inputs = np.random.default_rng(12).uniform(-0.98, 0.98, 4096).astype("float32")
    native, scalar = (
        StreamingLowpass(sample_rate, cutoff, q=q),
        StreamingLowpass(sample_rate, cutoff, q=q),
    )
    blocks = [
        inputs[:1],
        inputs[1:2],
        inputs[2:50],
        inputs[50:1600],
        inputs[1600:1601],
        inputs[1601:],
    ]
    for block in blocks:
        actual = native.process(block)
        expected = _scalar_filter(scalar, block)
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == np.float32
        assert native._z1 == scalar._z1 and native._z2 == scalar._z2
    whole = StreamingLowpass(sample_rate, cutoff, q=q).process(inputs)
    partitioned = StreamingLowpass(sample_rate, cutoff, q=q)
    np.testing.assert_array_equal(
        np.concatenate([partitioned.process(part) for part in blocks]), whole
    )


def test_300_native_chunks_match_the_measured_scalar_prototype_exactly():
    pytest.importorskip("scipy.signal")
    inputs = (np.random.default_rng(7).standard_normal(2400) * 0.1).astype("float32")
    inputs.flags.writeable = False
    native, scalar = StreamingLowpass(24000, 7000.0), StreamingLowpass(24000, 7000.0)
    snapshot = inputs.copy()
    for _ in range(300):
        np.testing.assert_array_equal(
            native.process(inputs), _scalar_filter(scalar, inputs)
        )
    assert native._z1 == scalar._z1 and native._z2 == scalar._z2
    np.testing.assert_array_equal(inputs, snapshot)


def test_native_reset_and_reconfigure_keep_existing_state_contract():
    pytest.importorskip("scipy.signal")
    inputs = np.ones(32, dtype="float32") * 0.2
    filt = StreamingLowpass(24000, 7000.0)
    first = filt.process(inputs)
    assert filt._z1 != 0
    filt.reset()
    assert filt._z1 == filt._z2 == 0
    np.testing.assert_array_equal(filt.process(inputs), first)
    filt.configure(16000, 1000.0)
    assert filt._z1 == filt._z2 == 0
    np.testing.assert_array_equal(
        filt.process(inputs), StreamingLowpass(16000, 1000.0).process(inputs)
    )


def test_empty_and_disabled_paths_never_import_or_call_native_kernel(monkeypatch):
    monkeypatch.setattr(
        frontend,
        "_has_scipy",
        lambda: (_ for _ in ()).throw(AssertionError("no optional probe")),
    )
    inputs = np.ones(4, dtype="float32")
    for sample_rate, cutoff in (
        (0, 1000.0),
        (24000, 0.0),
        (24000, 12000.0),
        (24000, float("nan")),
    ):
        assert StreamingLowpass(sample_rate, cutoff).process(inputs) is inputs
    active = StreamingLowpass(24000, 7000.0)
    active._z1, active._z2 = 0.1, -0.2
    empty = active.process(np.zeros(0, dtype="float32"))
    assert empty.size == 0 and empty.dtype == np.float32
    assert (active._z1, active._z2) == (0.1, -0.2)


def test_no_scipy_and_failed_kernel_fall_back_without_losing_prior_state(monkeypatch):
    signal = pytest.importorskip("scipy.signal")
    inputs = np.arange(24, dtype="float32") / 24
    expected = StreamingLowpass(24000, 7000.0)
    expected._z1, expected._z2 = 0.2, -0.1
    baseline = _scalar_filter(expected, inputs)
    for kind in ("absent", "failed", "bad-state"):
        filt = StreamingLowpass(24000, 7000.0)
        filt._z1, filt._z2 = 0.2, -0.1
        with monkeypatch.context() as patch:
            patch.setattr(frontend, "_has_scipy", lambda: kind != "absent")

            def failure(*_args, **kwargs):
                kwargs["zi"][:] = 99  # Optional kernel cannot mutate owner state.
                if kind == "bad-state":
                    return np.ones_like(inputs), np.array(
                        [1.0, "invalid"], dtype=object
                    )
                raise RuntimeError("synthetic kernel error")

            patch.setattr(signal, "sosfilt", failure)
            np.testing.assert_array_equal(filt.process(inputs), baseline)
        assert filt._z1 == expected._z1 and filt._z2 == expected._z2


def test_missing_optional_signal_import_keeps_scalar_result(monkeypatch):
    inputs = np.linspace(-0.5, 0.5, 40).astype("float32")
    scalar = StreamingLowpass(24000, 7000.0)
    expected = _scalar_filter(scalar, inputs)
    monkeypatch.setattr(frontend, "_has_scipy", lambda: True)
    monkeypatch.setitem(sys.modules, "scipy.signal", None)
    filt = StreamingLowpass(24000, 7000.0)
    np.testing.assert_array_equal(filt.process(inputs), expected)
    assert filt._z1 == scalar._z1 and filt._z2 == scalar._z2


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_input_propagation_matches_legacy_and_reset_recovers(bad):
    pytest.importorskip("scipy.signal")
    inputs = np.array([0.1, 0.2, bad, -0.3, 0.4], dtype="float32")
    native, scalar = StreamingLowpass(24000, 7000.0), StreamingLowpass(24000, 7000.0)
    with np.errstate(invalid="ignore"):
        expected = _scalar_filter(scalar, inputs)
        actual = native.process(inputs)
    np.testing.assert_array_equal(actual, expected)
    assert np.isnan(native._z1) == np.isnan(scalar._z1)
    assert np.isnan(native._z2) == np.isnan(scalar._z2)
    native.reset()
    np.testing.assert_array_equal(
        native.process(np.ones(8, dtype="float32")),
        StreamingLowpass(24000, 7000.0).process(np.ones(8, dtype="float32")),
    )


def test_failed_optional_dependency_probe_preserves_scalar_fallback(monkeypatch):
    inputs = np.linspace(-0.5, 0.5, 40).astype("float32")
    reference = StreamingLowpass(24000, 7000.0)
    expected = _scalar_filter(reference, inputs)

    def failure():
        raise ValueError("synthetic broken optional module spec")

    monkeypatch.setattr(frontend, "_has_scipy", failure)
    filt = StreamingLowpass(24000, 7000.0)
    np.testing.assert_array_equal(filt.process(inputs), expected)
    assert (filt._z1, filt._z2) == (reference._z1, reference._z2)
