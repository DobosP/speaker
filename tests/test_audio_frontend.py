"""Tests for the capture-path audio front-end (anti-alias resampler + soft gain)
and the WER utility. Pure numpy / stdlib -- no audio device, no models."""
from __future__ import annotations

import numpy as np
import pytest

from core.audio_frontend import (
    AudioResampler,
    StreamingLowpass,
    apply_gain_soft_limit,
    normalize_rms,
    rms_of,
)
from core.wer import word_error_rate


# --- AudioResampler ---------------------------------------------------------


def test_resampler_identity_when_rates_equal():
    r = AudioResampler(16000, 16000)
    x = np.array([0.1, -0.2, 0.3], dtype="float32")
    assert r.kind == "identity"
    assert np.array_equal(r.process(x), x)


def test_resampler_downsamples_roughly_by_ratio():
    r = AudioResampler(48000, 16000)
    # Feed 1s of 48k audio in 0.1s blocks; total output ~= 16000 samples (/3).
    total = 0
    for _ in range(10):
        total += len(r.process(np.zeros(4800, dtype="float32")))
    assert 15000 <= total <= 16200  # warm-up delay makes it approximate


def test_resampler_anti_aliases_a_tone_above_nyquist():
    # A 9 kHz tone is ABOVE the 8 kHz Nyquist of 16 kHz. A proper anti-alias
    # resampler attenuates it; naive linear interpolation folds it back into the
    # 0-8 kHz band. Assert the resampler's output has far less in-band energy
    # than a naive linear decimation of the same tone.
    sr = 48000
    t = np.arange(sr) / sr
    tone = np.sin(2 * np.pi * 9000 * t).astype("float32")
    r = AudioResampler(sr, 16000)
    out = r.process(tone, last=True)
    energy = float(np.mean(out * out))
    # naive linear baseline (what the old path did)
    n_out = int(round(len(tone) * 16000 / sr))
    idx = np.linspace(0, len(tone) - 1, n_out)
    linear = np.interp(idx, np.arange(len(tone)), tone).astype("float32")
    linear_energy = float(np.mean(linear * linear))
    if r.kind in ("soxr", "scipy"):
        assert energy < 0.2 * linear_energy  # anti-alias kills the aliased tone
    # (linear fallback can't anti-alias; skip the assertion there)


# --- soft-knee gain ---------------------------------------------------------


def test_soft_gain_unity_is_noop():
    x = [0.1, -0.2]
    assert apply_gain_soft_limit(x, 1.0) is x


def test_soft_gain_below_knee_is_linear():
    x = np.array([0.05, -0.05], dtype="float32")  # *2 = 0.1, below the 0.8 knee
    assert np.allclose(apply_gain_soft_limit(x, 2.0), x * 2.0)


def test_soft_gain_no_flat_top():
    # Two inputs that both exceed full-scale after gain. np.clip would tie both
    # at 1.0 (flat top); the soft limiter keeps them below 1.0 and distinct.
    a = apply_gain_soft_limit(np.array([0.5], dtype="float32"), 2.0)[0]  # 1.0
    b = apply_gain_soft_limit(np.array([0.6], dtype="float32"), 2.0)[0]  # 1.2
    assert a < 1.0 and b < 1.0 and a < b


# --- normalize_rms (per-sentence TTS loudness) ------------------------------


def test_normalize_rms_off_is_identity():
    x = np.array([0.1, -0.2], dtype="float32")
    assert normalize_rms(x, 0.0) is x  # target<=0 -> no-op, same object


def test_normalize_rms_equalizes_quiet_and_loud_to_one_level():
    # The core fix: an offline VITS model emits a different amplitude per sentence;
    # normalizing to one target RMS couples a STABLE echo level into the mic.
    rng = np.random.RandomState(0)
    quiet = (rng.randn(16000) * 0.02).astype("float32")  # a quiet sentence
    loud = (rng.randn(16000) * 0.30).astype("float32")   # a loud sentence
    q = normalize_rms(quiet, 0.12)
    ld = normalize_rms(loud, 0.12)
    assert np.sqrt(np.mean(q.astype("float64") ** 2)) == pytest.approx(0.12, rel=0.05)
    # The loud clip saturates a little on the peaks (soft knee), so allow a small
    # downward tolerance -- the point is both land near the SAME level, not far apart.
    assert np.sqrt(np.mean(ld.astype("float64") ** 2)) == pytest.approx(0.12, rel=0.12)


def test_normalize_rms_does_not_hard_clip_a_loud_clip():
    loud = (np.random.RandomState(1).randn(16000) * 0.5).astype("float32")
    y = normalize_rms(loud, 0.15)
    assert np.abs(y).max() < 1.0  # soft-knee limiter, never full-scale


def test_normalize_rms_caps_gain_on_near_silence():
    # A near-silent clip must not be amplified into noise: max_gain caps it.
    almost_silent = np.full(1600, 1e-4, dtype="float32")
    y = normalize_rms(almost_silent, 0.12, max_gain=20.0)
    assert np.abs(y).max() <= 1e-4 * 20.0 + 1e-6


def test_normalize_rms_silence_is_safe():
    y = normalize_rms(np.zeros(1600, dtype="float32"), 0.12)
    assert float(np.abs(y).max()) == 0.0  # no divide-by-zero blow-up


# --- streaming TTS low-pass -------------------------------------------------


def _tone_mag(samples, sr: int, freq: float) -> float:
    x = np.asarray(samples, dtype="float64").reshape(-1)
    t = np.arange(x.size, dtype="float64") / float(sr)
    s = np.sin(2.0 * np.pi * float(freq) * t)
    c = np.cos(2.0 * np.pi * float(freq) * t)
    return float((2.0 / x.size) * np.hypot(np.dot(x, s), np.dot(x, c)))


def test_rms_of_empty_and_tone():
    assert rms_of(np.zeros(0, dtype="float32")) == 0.0
    x = np.array([1.0, -1.0, 1.0, -1.0], dtype="float32")
    assert rms_of(x) == pytest.approx(1.0)


def test_streaming_lowpass_attenuates_high_frequency_energy_across_chunks():
    sr = 16000
    t = np.arange(sr, dtype="float64") / sr
    low = 0.30 * np.sin(2.0 * np.pi * 300.0 * t)
    high = 0.30 * np.sin(2.0 * np.pi * 6000.0 * t)
    x = (low + high).astype("float32")

    filt = StreamingLowpass(sr, 1000.0)
    y = np.concatenate([filt.process(chunk) for chunk in np.array_split(x, 17)])

    assert _tone_mag(y, sr, 6000.0) < 0.08 * _tone_mag(x, sr, 6000.0)
    assert _tone_mag(y, sr, 300.0) > 0.85 * _tone_mag(x, sr, 300.0)


def test_streaming_lowpass_chunking_matches_single_pass():
    sr = 16000
    rng = np.random.default_rng(4)
    x = rng.standard_normal(4096).astype("float32") * 0.05
    whole = StreamingLowpass(sr, 1800.0).process(x)

    filt = StreamingLowpass(sr, 1800.0)
    chunked = np.concatenate([filt.process(chunk) for chunk in np.array_split(x, 13)])

    np.testing.assert_allclose(chunked, whole, rtol=1e-6, atol=1e-7)


def test_streaming_lowpass_disabled_and_above_nyquist_are_unchanged():
    x = np.random.default_rng(5).standard_normal(128).astype("float32")
    assert StreamingLowpass(16000, 0.0).process(x) is x
    assert StreamingLowpass(16000, -200.0).process(x) is x
    assert StreamingLowpass(16000, 8000.0).process(x) is x
    assert StreamingLowpass(0, 1000.0).process(x) is x


# --- WER --------------------------------------------------------------------


def test_wer_identical_is_zero():
    assert word_error_rate("the cat sat", "the cat sat").wer == 0.0


def test_wer_counts_ops():
    r = word_error_rate("the cat sat on the mat", "the dog sat on mat")
    # 'cat'->'dog' = 1 sub, 'the' before 'mat' deleted = 1 del. ref=6 words.
    assert r.substitutions == 1
    assert r.deletions == 1
    assert r.insertions == 0
    assert r.wer == round(2 / 6, 4) or abs(r.wer - 2 / 6) < 1e-9


def test_wer_normalizes_case_and_punctuation():
    assert word_error_rate("Hello, world!", "hello world").wer == 0.0


def test_wer_empty_reference():
    assert word_error_rate("", "").wer == 0.0
    assert word_error_rate("", "spurious words").wer == 1.0


def test_resampler_accepts_quality_kwarg():
    # The capture downsampler takes a soxr quality kernel; VHQ/HQ/LQ must all build
    # (default HQ is byte-identical) and still resample roughly by the ratio.
    for q in ("VHQ", "HQ", "LQ"):
        r = AudioResampler(48000, 16000, quality=q)
        assert r.kind in ("soxr", "scipy", "linear")
        out = r.process(np.zeros(4800, dtype="float32"))
        assert 0 < out.size <= 1700          # ~1600 samples out (+/- FIR warmup)


def test_normalize_rms_collapses_sentence_to_sentence_variance():
    """The "fluid TTS" property: a sequence of sentences at WILDLY different raw
    levels (what the VITS voice emits) must come out at a STEADY level, so playback
    isn't choppy. Pins the spread-collapse, not just per-clip level (which the
    test above already covers)."""
    rng = np.random.default_rng(7)
    target = 0.12
    levels = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40]
    raw, out = [], []
    for i, lvl in enumerate(levels):
        x = rng.standard_normal(2000).astype("float32")
        x *= lvl / float(np.sqrt(np.mean(x ** 2)))      # exact raw RMS = lvl
        raw.append(lvl)
        y = np.asarray(normalize_rms(x, target), dtype="float32")
        out.append(float(np.sqrt(np.mean(y.astype("float64") ** 2))))
    raw_spread = max(raw) - min(raw)                    # 0.38
    out_spread = max(out) - min(out)
    assert out_spread < raw_spread / 8                  # variance collapsed (no /0)
    for r in out:
        assert abs(r - target) < 0.15 * target          # each within 15% of target
