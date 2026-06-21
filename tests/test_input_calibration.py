"""Startup ambient calibration (core.audio_frontend.compute_input_calibration).

The device-generic operating-point step: measure THIS mic's quiet floor at startup
and set the AGC noise gate just above it -- no per-machine hand tuning. Pure logic,
no audio device.
"""
import numpy as np

from core.audio_frontend import compute_input_calibration, InputAGC


def _block(rms, n=1600, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype("float32")
    x *= rms / float(np.sqrt(np.mean(x ** 2)))
    return x


def test_floor_tracks_quiet_level_with_headroom():
    # A quiet room ~0.01 RMS -> floor ~ headroom * 0.01, clamped into range.
    blocks = [_block(0.01, seed=i) for i in range(15)]
    cal = compute_input_calibration(blocks, headroom=3.0)
    assert cal["n_blocks"] == 15
    assert abs(cal["ambient_rms"] - 0.01) < 0.003
    assert 0.02 < cal["noise_floor_rms"] < 0.04          # ~3x ambient
    assert cal["clipping_fraction"] == 0.0


def test_low_percentile_is_robust_to_a_word_during_calibration():
    # Mostly quiet, but a few loud (speech) blocks slipped in -> the floor must
    # follow the QUIET level, not be dragged up by the loud blocks.
    blocks = [_block(0.008, seed=i) for i in range(12)] + [_block(0.3, seed=99 + i) for i in range(3)]
    cal = compute_input_calibration(blocks, headroom=3.0)
    assert cal["noise_floor_rms"] < 0.05                 # not pulled toward 0.3


def test_floor_is_clamped():
    # Near-silent -> clamped to min_floor; very loud floor -> clamped to max_floor.
    assert compute_input_calibration([_block(1e-5)], min_floor=0.004)["noise_floor_rms"] == 0.004
    loud = compute_input_calibration([_block(0.5, seed=i) for i in range(5)], max_floor=0.08)
    assert loud["noise_floor_rms"] == 0.08


def test_clipping_fraction_detected():
    railed = np.ones(1600, dtype="float32")              # fully railed
    cal = compute_input_calibration([railed])
    assert cal["clipping_fraction"] > 0.9
    assert cal["peak"] >= 1.0


def test_empty_input_is_safe():
    cal = compute_input_calibration([])
    assert cal["n_blocks"] == 0
    assert cal["noise_floor_rms"] == 0.004
    assert cal["clipping_fraction"] == 0.0


def test_calibration_feeds_the_agc_floor():
    # End-to-end: the measured floor is the value the engine assigns to InputAGC.
    blocks = [_block(0.02, seed=i) for i in range(10)]
    cal = compute_input_calibration(blocks)
    agc = InputAGC(noise_floor_rms=0.004)
    agc.noise_floor_rms = cal["noise_floor_rms"]
    assert agc.noise_floor_rms == cal["noise_floor_rms"]
    assert agc.noise_floor_rms > 0.004                   # adapted up from the default
