"""Tests for the opt-in TTS OUTPUT LEVELER (perceptual loudness target +
look-ahead true-peak soft-knee limiter) -- a pure-numpy WebRTC AGC2 port.

Covers the algorithm (loudness target, slew, true-peak ceiling, length/NaN/
silence guarantees) AND the load-bearing DEFAULTS-SAFE invariant: with the
config bool OFF the engine's whole-clip path is byte-identical to the legacy
normalize_rms path. Pure numpy / stdlib -- no audio device, no models."""
from __future__ import annotations

import numpy as np
import pytest

from core.audio_frontend import (
    _rms_dbfs_voiced,
    _running_min_forward,
    _true_peak_gain_envelope,
    _upsampled_abs_per_sample,
    normalize_rms,
    output_leveler,
)


def _rms(x):
    x = np.asarray(x, dtype="float64").reshape(-1)
    return float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0


def _dbfs(x):
    r = _rms(x)
    return 20.0 * np.log10(r) if r > 0 else -np.inf


def _tone(rms_level, n=16000, sr=16000, freq=220.0):
    t = np.arange(n) / sr
    x = np.sin(2 * np.pi * freq * t).astype("float32")
    cur = float(np.sqrt(np.mean(x.astype("float64") ** 2)))
    if cur <= 0.0:  # n too small to hold a sine cycle (e.g. n=1 -> all-zero) -> DC
        return np.full(n, rms_level, dtype="float32")
    x *= rms_level / cur
    return x.astype("float32")


# --- _rms_dbfs_voiced -------------------------------------------------------


def test_voiced_rms_ignores_leading_trailing_silence():
    # A loud tone padded with silence: the voiced estimate must reflect the TONE
    # level, not the diluted whole-clip RMS (the perceptual point).
    tone = _tone(0.2, n=8000)
    clip = np.concatenate([np.zeros(8000, "float32"), tone, np.zeros(8000, "float32")])
    voiced_dbfs = _rms_dbfs_voiced(clip)
    whole_dbfs = _dbfs(clip)
    assert voiced_dbfs == pytest.approx(_dbfs(tone), abs=0.5)
    assert voiced_dbfs > whole_dbfs + 2.0  # silence would have dragged it down


def test_voiced_rms_none_on_silence():
    assert _rms_dbfs_voiced(np.zeros(4000, "float32")) is None
    assert _rms_dbfs_voiced(np.zeros(0, "float32")) is None
    assert _rms_dbfs_voiced(np.full(4000, 1e-7, "float32")) is None


# --- _upsampled_abs_per_sample (inter-sample / true-peak measurement) --------


def test_upsample_catches_intersample_peak():
    # A signal whose true peak sits BETWEEN samples: a per-sample |x| underreports
    # it; the oversampled measurement must report MORE than the raw sample peak.
    sr = 16000
    x = _tone(0.3, n=2000, sr=sr, freq=3900.0)  # near Nyquist -> big inter-sample peaks
    raw_peak = float(np.max(np.abs(x)))
    tp = float(np.max(_upsampled_abs_per_sample(x, 4)))
    assert tp >= raw_peak - 1e-6
    assert tp > raw_peak * 1.001  # genuinely found inter-sample energy
    assert _upsampled_abs_per_sample(x, 4).shape[0] == x.shape[0]  # length preserved


def test_upsample_oversample_one_is_abs():
    x = np.array([0.1, -0.5, 0.3], dtype="float32")
    np.testing.assert_allclose(_upsampled_abs_per_sample(x, 1), np.abs(x.astype("float64")))


# --- _running_min_forward (look-ahead) --------------------------------------


def test_running_min_forward_pulls_dip_earlier():
    g = np.array([1.0, 1.0, 1.0, 0.5, 1.0, 1.0], dtype="float64")
    out = _running_min_forward(g, 3)  # window of 3 samples forward
    # The 0.5 at index 3 is pulled back to indices 1,2,3 (the look-ahead).
    assert out[1] == pytest.approx(0.5)
    assert out[3] == pytest.approx(0.5)
    assert out.shape == g.shape


def test_running_min_forward_noop_window_one():
    g = np.array([1.0, 0.5, 1.0], dtype="float64")
    np.testing.assert_array_equal(_running_min_forward(g, 1), g)


# --- _true_peak_gain_envelope -----------------------------------------------


def test_true_peak_envelope_holds_peak_below_ceiling():
    sr = 16000
    x = _tone(0.9, n=4000, sr=sr, freq=300.0)
    ceiling = 10.0 ** (-1.0 / 20.0)  # -1 dBTP
    env = _true_peak_gain_envelope(x, ceiling, sr)
    y = x * env
    tp = float(np.max(_upsampled_abs_per_sample(y, 4)))
    assert tp <= ceiling + 1e-3  # inter-sample peak held under the ceiling
    assert env.shape == x.shape


def test_true_peak_envelope_unity_when_below_ceiling():
    sr = 16000
    x = _tone(0.1, n=2000, sr=sr)  # well below any sane ceiling
    ceiling = 10.0 ** (-1.0 / 20.0)
    env = _true_peak_gain_envelope(x, ceiling, sr)
    np.testing.assert_allclose(env, 1.0, atol=1e-6)


# --- output_leveler: loudness target ----------------------------------------


def test_leveler_brings_quiet_and_loud_to_same_target():
    sr = 16000
    target = -20.0
    quiet, _ = output_leveler(_tone(0.02), target_dbfs=target, true_peak_dbtp=-1.0, sr=sr,
                              max_step_db=100.0)  # unbounded step -> reach target in one clip
    loud, _ = output_leveler(_tone(0.30), target_dbfs=target, true_peak_dbtp=-1.0, sr=sr,
                             max_step_db=100.0)
    # Both land near the perceptual target (within boost/cut bounds + limiter pull).
    assert _dbfs(quiet) == pytest.approx(target, abs=1.5)
    assert _dbfs(loud) == pytest.approx(target, abs=2.0)


def test_leveler_respects_max_boost_on_quiet_clip():
    # A quiet-but-voiced clip wanting a big boost must be clamped to max_boost
    # (not amplified the full distance to target). Input is above the voiced floor
    # (-50 dBFS) so it is treated as speech, not silence.
    sr = 16000
    quiet = _tone(0.01)  # ~-40 dBFS, target -20 -> wants +20 dB -> clamp to +18
    out, applied = output_leveler(quiet, target_dbfs=-20.0, true_peak_dbtp=-1.0, sr=sr,
                                  max_boost_db=18.0, max_step_db=100.0)
    assert applied == pytest.approx(18.0, abs=1e-6)  # clamped to max_boost


def test_leveler_respects_max_cut_on_hot_clip():
    sr = 16000
    hot = _tone(0.5)  # ~-6 dBFS, target -20 -> wants -14 dB, clamp at -12
    out, applied = output_leveler(hot, target_dbfs=-20.0, true_peak_dbtp=-1.0, sr=sr,
                                  max_cut_db=12.0, max_step_db=100.0)
    assert applied == pytest.approx(-12.0, abs=1e-6)


def test_leveler_slews_gain_between_sentences():
    # The inter-sentence slew: a big desired gain change is reached over several
    # sentences (max_step_db per call), not in one jump -- AGC2's slow attack.
    sr = 16000
    quiet = _tone(0.01)  # wants a large boost
    prev = 0.0
    gains = []
    for _ in range(5):
        _, prev = output_leveler(quiet, target_dbfs=-20.0, true_peak_dbtp=-1.0, sr=sr,
                                 prev_gain_db=prev, max_step_db=3.0, max_boost_db=18.0)
        gains.append(prev)
    # Monotonic ramp, each step bounded by max_step_db, converging toward the cap.
    for a, b in zip(gains, gains[1:]):
        assert 0 < (b - a) <= 3.0 + 1e-6
    assert gains[-1] > gains[0]


# --- output_leveler: true-peak guarantee ------------------------------------


def test_leveler_never_clips_full_scale():
    sr = 16000
    # A hot clip that the loudness stage would push even hotter: the limiter must
    # still keep the inter-sample peak below the dBTP ceiling and never hit 1.0.
    hot = _tone(0.4)
    out, _ = output_leveler(hot, target_dbfs=-6.0, true_peak_dbtp=-1.0, sr=sr,
                            max_boost_db=18.0, max_step_db=100.0)
    ceiling = 10.0 ** (-1.0 / 20.0)
    assert float(np.max(np.abs(out))) < 1.0
    tp = float(np.max(_upsampled_abs_per_sample(out, 4)))
    assert tp <= ceiling + 2e-3  # true (inter-sample) peak under the ceiling


def test_leveler_with_impulse_does_not_clip():
    sr = 16000
    x = _tone(0.2, n=4000, sr=sr).copy()
    x[2000] = 0.99  # a leftover transient (declick would normally catch it)
    out, _ = output_leveler(x, target_dbfs=-12.0, true_peak_dbtp=-1.0, sr=sr,
                            max_step_db=100.0)
    assert float(np.max(np.abs(out))) < 1.0


# --- output_leveler: guarantees (length, NaN, silence) ----------------------


def test_leveler_preserves_length():
    sr = 16000
    for n in (1, 100, 4001, 16000):
        x = _tone(0.1, n=n, sr=sr)
        out, _ = output_leveler(x, target_dbfs=-20.0, true_peak_dbtp=-1.0, sr=sr)
        assert out.shape[0] == n
        assert out.dtype == np.float32


def test_leveler_silence_passthrough_holds_gain():
    sr = 16000
    out, applied = output_leveler(np.zeros(4000, "float32"), target_dbfs=-20.0,
                                  true_peak_dbtp=-1.0, sr=sr, prev_gain_db=4.0)
    assert float(np.max(np.abs(out))) == 0.0
    assert applied == pytest.approx(4.0)  # loudness gain unchanged on silence


def test_leveler_empty_input():
    out, applied = output_leveler(np.zeros(0, "float32"), target_dbfs=-20.0,
                                  true_peak_dbtp=-1.0, sr=16000, prev_gain_db=2.5)
    assert out.shape[0] == 0
    assert applied == pytest.approx(2.5)


def test_leveler_no_nan_on_pathological_input():
    sr = 16000
    x = np.array([0.0, 1e-9, -1e-9, 1.5, -2.0], dtype="float32")  # incl. over-full-scale
    out, _ = output_leveler(x, target_dbfs=-20.0, true_peak_dbtp=-1.0, sr=sr)
    assert np.all(np.isfinite(out))
    assert float(np.max(np.abs(out))) <= 1.0


# --- DEFAULTS-SAFE invariant (engine-level OFF == legacy) -------------------


def test_engine_off_path_is_byte_identical_to_legacy_normalize_rms():
    """The load-bearing repo invariant: with tts_output_leveler False the
    whole-clip path runs normalize_rms->declick exactly as before -- byte for
    byte. Drives the real engine's _synthesize with a fake TTS so the test has
    no audio/model deps. (The leveler must be UNREACHABLE when the bool is off.)"""
    from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine

    sr = 16000
    rng = np.random.default_rng(3)
    raw = (rng.standard_normal(4000) * 0.3).astype("float32")
    raw[1500] = 0.95  # an impulse spike, so declick has work to do too

    class _Tts:
        sample_rate = sr

        def generate(self, text, sid=0, speed=1.0):
            return type("A", (), {"samples": raw.copy(), "sample_rate": sr})()

    # Legacy reference: normalize_rms then declick, exactly the OFF code path.
    from core.audio_frontend import declick

    target_rms = 0.12
    ref = np.asarray(normalize_rms(raw.copy(), target_rms), dtype="float32").reshape(-1)
    ref = np.asarray(declick(ref, threshold=0.22), dtype="float32").reshape(-1)

    eng = SherpaOnnxEngine(
        SherpaConfig(tts_target_rms=target_rms, tts_declick=True,
                     tts_declick_threshold=0.22, tts_output_leveler=False)
    )
    eng._tts = _Tts()
    eng._tts_can_stream = False  # force the whole-clip path
    written: list = []
    eng._synthesize("x", written.append)
    got = np.concatenate(written)
    np.testing.assert_array_equal(got, ref)  # byte-identical to the legacy path


def test_engine_leveler_on_replaces_normalize_rms_and_caps_peak():
    """With the bool ON the leveler OWNS loudness+peak: normalize_rms is skipped
    (its linear-RMS target would fight the perceptual one) and the output never
    clips. Proves the wiring + composition order in _synthesize."""
    from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine

    sr = 16000
    loud = _tone(0.4, n=4000, sr=sr).copy()
    loud[2000] = 0.97  # a spike declick should repair before the limiter sees it

    class _Tts:
        sample_rate = sr

        def generate(self, text, sid=0, speed=1.0):
            return type("A", (), {"samples": loud.copy(), "sample_rate": sr})()

    eng = SherpaOnnxEngine(
        SherpaConfig(
            tts_target_rms=0.12,           # set, but must be IGNORED when leveler is on
            tts_declick=True, tts_declick_threshold=0.22,
            tts_output_leveler=True,
            tts_loudness_target_dbfs=-18.0,
            tts_true_peak_dbtp=-1.0,
        )
    )
    eng._tts = _Tts()
    eng._tts_can_stream = True  # leveler must FORCE the whole-clip path anyway
    written: list = []
    eng._synthesize("x", written.append)
    out = np.concatenate(written)
    assert float(np.max(np.abs(out))) < 1.0          # true-peak limiter held it
    # The leveler (not normalize_rms) ran: a loud (~-8 dBFS) FIRST sentence is
    # SLEWED toward the -18 dBFS target by one max_step_db (=3.0 default) -- a CUT
    # -- so the loudness gain is exactly -3 dB and the output got quieter, moving
    # toward target. normalize_rms would instead have driven RMS straight to 0.12
    # (-18.4 dBFS) in one shot, so this proves the legacy stage was skipped (no
    # double-normalize). Loudness converges to target over subsequent sentences
    # (covered by test_engine_leveler_slews_across_a_multi_sentence_reply).
    assert eng._tts_level_gain_db == pytest.approx(-3.0, abs=1e-6)
    assert _dbfs(out) < _dbfs(loud) - 1.0            # moved toward the quieter target


def test_engine_leveler_slews_across_a_multi_sentence_reply():
    # The engine carries _tts_level_gain_db across sentences, so a quiet reply
    # ramps up over several sentences instead of jumping.
    from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine

    sr = 16000
    quiet = _tone(0.01, n=4000, sr=sr)

    class _Tts:
        sample_rate = sr

        def generate(self, text, sid=0, speed=1.0):
            return type("A", (), {"samples": quiet.copy(), "sample_rate": sr})()

    eng = SherpaOnnxEngine(
        SherpaConfig(tts_output_leveler=True, tts_loudness_target_dbfs=-20.0,
                     tts_true_peak_dbtp=-1.0, tts_declick=False)
    )
    eng._tts = _Tts()
    eng._tts_can_stream = False
    gains = []
    for _ in range(4):
        eng._synthesize("x", (lambda *a: None))
        gains.append(eng._tts_level_gain_db)
    for a, b in zip(gains, gains[1:]):
        assert b > a            # ramps up sentence to sentence
        assert (b - a) <= 3.0 + 1e-6  # bounded by the per-sentence slew step
