"""Unit tests for the TTS de-clicker (core.audio_frontend.declick).

The on-device VITS voice emits deterministic sample-level impulse spikes on some
text -> audible clicks. ``declick`` repairs the isolated impulses while leaving
clean speech untouched. These tests pin both properties (effective + safe) with
no audio/model deps.
"""
import numpy as np

from core.audio_frontend import declick


def _clicks(x, thr=0.3):
    return int((np.abs(np.diff(x)) > thr).sum())


def test_removes_isolated_impulses():
    sr = 22050
    t = np.arange(sr) / sr
    clean = (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32")
    glitchy = clean.copy()
    # inject single-sample impulses (jump away from neighbours and back)
    for i in (1000, 5000, 9000, 13000, 17000):
        glitchy[i] = 0.95 if glitchy[i] < 0 else -0.95
    assert _clicks(glitchy) > _clicks(clean)
    fixed = declick(glitchy)
    # impulses repaired -> click count back down near the clean baseline
    assert _clicks(fixed) <= _clicks(clean) + 1
    # and the underlying tone is preserved
    assert np.corrcoef(clean, fixed)[0, 1] > 0.999


def test_noop_on_clean_speech():
    sr = 22050
    t = np.arange(sr) / sr
    # a chirp + harmonics stands in for clean speech (fast but CONTINUOUS motion)
    clean = (0.4 * np.sin(2 * np.pi * (200 + 600 * t) * t)
             + 0.2 * np.sin(2 * np.pi * 1500 * t)).astype("float32")
    out = declick(clean)
    # essentially unchanged: a genuine fast transition tracks the median, so it
    # is not mistaken for an impulse.
    assert np.corrcoef(clean, out)[0, 1] > 0.9999
    assert float(np.max(np.abs(out - clean))) < 0.05


def test_threshold_zero_is_passthrough():
    x = np.array([0.0, 0.9, -0.9, 0.1, 0.0], dtype="float32")
    out = declick(x, threshold=0.0)
    assert np.array_equal(np.asarray(out, dtype="float32"), x)


def test_short_input_is_safe():
    for x in (np.zeros(0, "float32"), np.array([0.5], "float32"), np.array([0.1, -0.2], "float32")):
        out = np.asarray(declick(x), dtype="float32")
        assert out.shape == x.shape


def test_repairs_short_run():
    x = np.zeros(2000, dtype="float32")
    x[500:503] = [0.9, -0.9, 0.9]   # a 3-sample crackle burst
    out = declick(x)
    assert _clicks(out) < _clicks(x)
    assert float(np.max(np.abs(out[500:503]))) < 0.3


def test_higher_threshold_preserves_fricative_energy():
    # Dense consonant / fricative energy is fast but BAND-LIMITED (below ~8 kHz),
    # so neighbouring samples are correlated and the 3-point median tracks it --
    # unlike a true VITS spike (one sample jumping to 0.5-0.95). The engine default
    # (0.22) must leave such energy untouched while still repairing a real spike on
    # top of it. (A higher-amplitude fricative than the noop test, to stress it.)
    sr = 22050
    t = np.arange(sr) / sr
    fric = (0.35 * np.sin(2 * np.pi * (300 + 700 * t) * t)
            + 0.25 * np.sin(2 * np.pi * 1800 * t)).astype("float32")
    spiked = fric.copy()
    spiked[11000] = 0.95 if spiked[11000] < 0 else -0.95   # a real impulse on top
    out = declick(spiked, threshold=0.22)
    # The spike is repaired ...
    assert abs(float(out[11000]) - float(fric[11000])) < 0.3
    # ... but the surrounding fricative is preserved (no smear).
    keep = np.r_[0:10990, 11010:sr]
    assert np.corrcoef(fric[keep], out[keep])[0, 1] > 0.999
    assert float(np.max(np.abs(out[keep] - fric[keep]))) < 0.05

