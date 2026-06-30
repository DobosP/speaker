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
    audio_quality_metrics,
    lowpass_soft,
    normalize_rms,
    output_leveler,
)


def _sine(freq, sr=24000, dur=0.25, amp=0.3):
    t = np.arange(int(sr * dur)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype("float32")


# --- HF roll-off (lowpass_soft) for cheap open speakers --------------------

def test_lowpass_off_is_passthrough():
    x = _sine(9000)
    assert np.array_equal(lowpass_soft(x, 24000, 0.0), x)       # cutoff<=0
    assert np.array_equal(lowpass_soft(x, 24000, 20000.0), x)   # >= Nyquist


def test_lowpass_attenuates_above_cutoff_keeps_below():
    sr = 24000
    hi = _sine(10000, sr)   # well above a 7k cutoff -> should be crushed
    lo = _sine(1000, sr)    # well below -> should survive
    hi_out = lowpass_soft(hi, sr, 7000.0, width_hz=1500.0)
    lo_out = lowpass_soft(lo, sr, 7000.0, width_hz=1500.0)
    assert _rms(hi_out) < 0.1 * _rms(hi)    # HF removed
    assert _rms(lo_out) > 0.9 * _rms(lo)    # LF preserved


def test_lowpass_length_preserving_and_safe():
    x = _sine(3000)
    assert lowpass_soft(x, 24000, 7000.0).shape == x.shape
    assert lowpass_soft(np.zeros(4, dtype="float32"), 24000, 7000.0).size == 4  # too short -> passthrough
    assert lowpass_soft(np.zeros(0, dtype="float32"), 24000, 7000.0).size == 0


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
    # prev_gain_db=None (default) SEEDS to target on this first call -> reaches it.
    quiet, _ = output_leveler(_tone(0.02), target_dbfs=target, true_peak_dbtp=-1.0, sr=sr)
    loud, _ = output_leveler(_tone(0.30), target_dbfs=target, true_peak_dbtp=-1.0, sr=sr)
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
                                  max_boost_db=18.0)  # first call seeds to clamped desired
    assert applied == pytest.approx(18.0, abs=1e-6)  # clamped to max_boost


def test_leveler_respects_max_cut_on_hot_clip():
    sr = 16000
    hot = _tone(0.5)  # ~-6 dBFS, target -20 -> wants -14 dB, clamp at -12
    out, applied = output_leveler(hot, target_dbfs=-20.0, true_peak_dbtp=-1.0, sr=sr,
                                  max_cut_db=12.0)  # first call seeds to clamped desired
    assert applied == pytest.approx(-12.0, abs=1e-6)


def test_leveler_slew_is_time_aware_after_seeding():
    # The FIRST call seeds (prev=None); subsequent calls SLEW toward a new desired
    # by at most rate*duration per call -- so the step scales with sentence LENGTH
    # (time-aware, like AGC2), not a fixed per-call amount.
    sr = 16000
    n = sr  # 1.0 s clips -> step cap == rate*1.0 == rate
    rate = 4.0  # dB/s
    quiet = _tone(0.01, n=n, sr=sr)   # -40 dBFS, target -20 -> desired +18 (clamped)
    _, g0 = output_leveler(quiet, target_dbfs=-20.0, true_peak_dbtp=-1.0, sr=sr,
                           max_boost_db=18.0, max_slew_db_per_s=rate)
    assert g0 == pytest.approx(18.0, abs=1e-6)        # seeded straight to clamped target
    # A LOUD 1 s clip flips the desired gain negative; it slews DOWN by rate*1.0 s.
    loud = _tone(0.3, n=n, sr=sr)     # -10.5 dBFS, target -20 -> desired ~ -9.5
    _, g1 = output_leveler(loud, target_dbfs=-20.0, true_peak_dbtp=-1.0, sr=sr,
                           prev_gain_db=g0, max_slew_db_per_s=rate)
    assert (g0 - g1) == pytest.approx(rate, abs=1e-6)         # stepped by rate * 1.0 s
    # A SHORTER clip at the same level moves PROPORTIONALLY less (the time-aware fix).
    short = _tone(0.3, n=n // 4, sr=sr)  # 0.25 s -> step cap rate*0.25 = 1.0 dB
    _, g2 = output_leveler(short, target_dbfs=-20.0, true_peak_dbtp=-1.0, sr=sr,
                           prev_gain_db=g0, max_slew_db_per_s=rate)
    assert (g0 - g2) == pytest.approx(rate * 0.25, abs=1e-6)  # quarter the time -> quarter step


# --- output_leveler: true-peak guarantee ------------------------------------


def test_leveler_never_clips_full_scale():
    sr = 16000
    # A hot clip that the loudness stage would push even hotter: the limiter must
    # still keep the inter-sample peak below the dBTP ceiling and never hit 1.0.
    hot = _tone(0.4)
    out, _ = output_leveler(hot, target_dbfs=-6.0, true_peak_dbtp=-1.0, sr=sr,
                            max_boost_db=18.0)  # first call seeds; loudness pushes it hotter
    ceiling = 10.0 ** (-1.0 / 20.0)
    assert float(np.max(np.abs(out))) < 1.0
    tp = float(np.max(_upsampled_abs_per_sample(out, 4)))
    assert tp <= ceiling + 2e-3  # true (inter-sample) peak under the ceiling


def test_leveler_with_impulse_does_not_clip():
    sr = 16000
    x = _tone(0.2, n=4000, sr=sr).copy()
    x[2000] = 0.99  # a leftover transient (declick would normally catch it)
    out, _ = output_leveler(x, target_dbfs=-12.0, true_peak_dbtp=-1.0, sr=sr)
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
    # The leveler (not normalize_rms) OWNS loudness: the FIRST sentence of the
    # session SEEDS straight to the -18 dBFS target, so a loud (~-8 dBFS) clip is
    # brought right onto target in this one call. (normalize_rms would instead drive
    # LINEAR RMS to 0.12; the discriminating proof is the gain state + that the
    # whole-clip leveler path ran and capped the peak -- no double-normalize.)
    assert eng._tts_level_gain_db is not None and eng._tts_level_gain_db < -1.0  # a real cut
    assert abs(_dbfs(out) - (-18.0)) < 1.5           # seeded onto the loudness target


def test_engine_leveler_consistent_loudness_across_sentences():
    # The owner's GOAL: every sentence of a reply lands at the SAME loudness. The
    # first sentence SEEDS to target; same-level sentences then HOLD that gain (no
    # per-sentence ramp, no reply-length dependence).
    from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine

    sr = 16000
    clip = _tone(0.05, n=4000, sr=sr)  # -26 dBFS, a reachable +8 dB to the -18 target

    class _Tts:
        sample_rate = sr

        def generate(self, text, sid=0, speed=1.0):
            return type("A", (), {"samples": clip.copy(), "sample_rate": sr})()

    eng = SherpaOnnxEngine(
        SherpaConfig(tts_output_leveler=True, tts_loudness_target_dbfs=-18.0,
                     tts_true_peak_dbtp=-1.0, tts_declick=False)
    )
    eng._tts = _Tts()
    eng._tts_can_stream = False
    gains = []
    for _ in range(4):
        eng._synthesize("x", (lambda *a: None))
        gains.append(eng._tts_level_gain_db)
    assert gains[0] == pytest.approx(8.0, abs=0.6)    # first sentence seeded to target
    for g in gains[1:]:
        assert g == pytest.approx(gains[0], abs=0.5)  # held -> consistent loudness


def test_leveler_default_config_on_target_realistic_vits():
    # The SHIPPING default (time-aware slew 6 dB/s, target -18) on realistic VITS-
    # level clips at the real 22050 Hz rate: every sentence lands ON target -- the
    # owner's consistent-loudness goal, proven for the actual config (not a 10-dB-off
    # synthetic). max_step is not overridden, so the default slew is exercised.
    sr = 22050
    levels = [0.108, 0.130, 0.115, 0.122]  # measured VITS RMS range (config.json comment)
    prev = None
    outs = []
    for rms in levels:
        clip = _tone(rms, n=int(sr * 1.2), sr=sr)  # ~1.2 s sentences
        y, prev = output_leveler(clip, target_dbfs=-18.0, true_peak_dbtp=-1.0, sr=sr,
                                 prev_gain_db=prev)
        outs.append(_dbfs(y))
    for d in outs:
        assert abs(d - (-18.0)) < 1.5  # consistent, on the loudness target every sentence


@pytest.mark.parametrize("sr", [16000, 22050])
def test_true_peak_holds_below_ceiling_at_production_rate(sr):
    # The shipped voice is 22050 Hz, but the rest of the suite runs at 16000; the
    # lookahead window + release coefficient both scale with sr, so cover both.
    x = _tone(0.9, n=sr // 4, sr=sr, freq=300.0)
    ceiling = 10.0 ** (-1.0 / 20.0)
    out, _ = output_leveler(x, target_dbfs=-6.0, true_peak_dbtp=-1.0, sr=sr, max_boost_db=18.0)
    tp = float(np.max(_upsampled_abs_per_sample(out, 8)))
    assert tp <= ceiling + 2e-3
    assert float(np.max(np.abs(out))) < 1.0


def test_leveler_without_scipy_fallback_is_safe(monkeypatch):
    # When scipy is absent, _upsampled_abs_per_sample degrades to linear-interp
    # (which cannot SEE inter-sample peaks), so the limiter becomes a sample-peak
    # limiter. It must still preserve length and never clip full scale.
    import core.audio_frontend as af

    monkeypatch.setattr(af, "_has_scipy", lambda: False)
    sr = 16000
    x = _tone(0.5, n=2000, sr=sr, freq=300.0)
    out, _ = af.output_leveler(x, target_dbfs=-12.0, true_peak_dbtp=-1.0, sr=sr)
    assert out.shape[0] == x.shape[0]
    assert float(np.max(np.abs(out))) <= 1.0


# --- audio_quality_metrics (per-utterance quality telemetry) ---------------
# Runtime instrumentation added for the 2026-06-30 "robotic / white-noise"
# investigation: the only forensic tap on a run bundle before this was
# .ref.wav, a naive-linear-resampled 16 kHz AEC tap (see audio_frontend's
# docstring) that should not be trusted to judge fine digital-domain quality.
# These tests pin the metric on signals with known, opposite character so a
# future change can't silently break the "noise-like vs tonal" discrimination.

def test_audio_quality_metrics_pure_tone_reads_tonal_and_clean():
    sr = 24000
    x = _sine(220, sr=sr, dur=0.5, amp=0.3)  # 220 Hz * 0.5 s = exact 110 cycles
    m = audio_quality_metrics(x, sr)
    assert abs(m["rms"] - 0.3 / (2 ** 0.5)) < 1e-3
    assert m["peak"] == pytest.approx(0.3, abs=1e-4)
    assert m["clip_pct"] == 0.0
    assert abs(m["dc_offset"]) < 1e-6
    assert m["hf_ratio"] == 0.0          # well below the 4 kHz default cutoff
    assert m["spectral_flatness"] < 0.01  # tonal, not noise-like


def test_audio_quality_metrics_white_noise_reads_flat_and_broadband():
    sr = 24000
    rng = np.random.default_rng(0)
    x = (0.2 * rng.standard_normal(int(sr * 0.5))).astype("float32")
    m = audio_quality_metrics(x, sr)
    assert m["hf_ratio"] > 0.3            # noise has real energy above 4 kHz
    assert m["spectral_flatness"] > 0.3   # the "sounds like static" signature


def test_audio_quality_metrics_flags_clipping_and_dc_bias():
    sr = 24000
    x = _sine(220, sr=sr, dur=0.5, amp=0.3)
    biased_clipped = np.clip(x * 4.0 + 0.05, -1.0, 1.0).astype("float32")
    m = audio_quality_metrics(biased_clipped, sr)
    assert m["clip_pct"] > 20.0
    assert m["dc_offset"] > 0.01


def test_audio_quality_metrics_hf_only_tone_is_all_high_frequency():
    sr = 24000
    x = _sine(9000, sr=sr, dur=0.5, amp=0.3)  # above the 4 kHz default cutoff
    m = audio_quality_metrics(x, sr)
    assert m["hf_ratio"] == 1.0


def test_audio_quality_metrics_empty_clip_is_all_none():
    m = audio_quality_metrics(np.zeros(0, dtype="float32"), 24000)
    assert m == {
        "rms": None, "peak": None, "clip_pct": None, "dc_offset": None,
        "hf_ratio": None, "spectral_flatness": None, "n_samples": 0,
    }
