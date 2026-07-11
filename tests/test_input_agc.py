"""Input AGC: normalize a clean-but-quiet mic toward a target level so the user
can run a LOW (never-clipping) OS gain and still be heard. Pure DSP, no models."""
from __future__ import annotations

import numpy as np
from pytest import approx

from core.audio_frontend import InputAGC, apply_gain_soft_limit


def _sine(rms: float, n: int = 1600, f: int = 200, sr: int = 16000):
    t = np.arange(n) / sr
    x = np.sin(2 * np.pi * f * t).astype("float32")
    return (x * (rms / float(np.sqrt(np.mean(x ** 2))))).astype("float32")


def _rms(x) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype="float64") ** 2)))


def test_boosts_a_quiet_signal_toward_target():
    agc = InputAGC(target_rms=0.12, max_gain=12.0)
    blk = _sine(0.03)                       # desired gain = 0.12/0.03 = 4.0
    out = blk
    for _ in range(100):
        out = agc.process(blk)
    assert agc.gain == approx(4.0, abs=0.2)
    assert _rms(out) == approx(0.12, abs=0.02)   # recognizer now hears a healthy level


def test_holds_gain_on_silence_never_pumps_hiss():
    agc = InputAGC(noise_floor_rms=0.004)
    for _ in range(50):
        out = agc.process(_sine(0.001))     # below the noise floor
    assert agc.gain == approx(1.0)
    assert _rms(out) == approx(0.001, abs=1e-4)  # untouched


def test_is_boost_only_does_not_attenuate_a_loud_signal():
    # Software can't un-clip, so the AGC never attenuates -- a too-loud signal is
    # the user's cue to lower the OS gain, not something the AGC silently masks.
    agc = InputAGC(target_rms=0.12)
    loud = _sine(0.3)
    for _ in range(30):
        out = agc.process(loud)
    assert agc.gain == approx(1.0, abs=0.01)
    assert _rms(out) == approx(0.3, abs=0.02)


def test_gain_rises_slowly_to_avoid_pumping():
    agc = InputAGC(target_rms=0.12, rise=0.08)
    agc.process(_sine(0.03))                # desired 4.0, one block
    assert agc.gain == approx(1.0 + 0.08 * (4.0 - 1.0), abs=0.05)   # ~1.24, not a jump to 4


def test_stale_max_gain_cannot_overdrive_a_normal_current_block():
    """A long quiet run may leave high state; the next word stays at target."""
    agc = InputAGC(target_rms=0.12, max_gain=12.0, fall=0.4)
    agc.gain = 12.0
    rng = np.random.default_rng(1)
    block = rng.standard_normal(1600).astype("float32")
    block *= 0.061 / _rms(block)            # current desired gain ~= 1.97

    out = agc.process(block)

    desired = 0.12 / 0.061
    old_smoothed_gain = 12.0 + 0.4 * (desired - 12.0)
    old_output = apply_gain_soft_limit(block, old_smoothed_gain)
    # Counterfactual pins the live 0.4633-RMS regime to the old controller.
    assert _rms(old_output) == approx(0.4653, abs=0.002)
    # Preserve the smoothed state/release contract for following blocks...
    assert agc.gain == approx(old_smoothed_gain, abs=1e-6)
    assert agc.gain > desired
    # ...but stale state is never the boost applied to this louder block.
    assert _rms(out) == approx(0.12, abs=0.005)


def test_stale_boost_still_never_attenuates_a_hot_current_block():
    agc = InputAGC(target_rms=0.12, max_gain=12.0)
    agc.gain = 12.0
    loud = _sine(0.3)

    out = agc.process(loud)

    assert _rms(out) == approx(0.3, abs=0.02)


def test_max_gain_caps_the_boost():
    agc = InputAGC(target_rms=0.12, max_gain=5.0)
    blk = _sine(0.005)                      # uncapped would want 24x
    for _ in range(200):
        agc.process(blk)
    assert agc.gain == approx(5.0, abs=0.1)
