"""Unit tests for tools/audio_mix.py — the combine-two-files-into-one mixer.

Pure numpy math; no audio device, no models. Runs in the logic suite / CI.
"""

import numpy as np
import pytest

from tools.audio_mix import mix, _rms


def _tone(freq, dur, sr, amp=0.5):
    t = np.arange(int(dur * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype("float32")


def test_simultaneous_sum_same_rate():
    a = _tone(220, 1.0, 16000, amp=0.3)
    b = _tone(440, 1.0, 16000, amp=0.3)
    out, sr = mix([(a, 16000), (b, 16000)], normalize=False)
    assert sr == 16000
    assert out.shape[0] == a.shape[0]
    # No offset, no gain, no clipping -> exact sample-wise sum.
    np.testing.assert_allclose(out, a + b, atol=1e-5)


def test_offset_lengthens_timeline():
    a = _tone(220, 1.0, 16000)
    b = _tone(440, 1.0, 16000)
    out, sr = mix([(a, 16000), (b, 16000)], offsets_sec=[0.0, 0.5], normalize=False)
    # second clip starts 0.5s (8000 samples) in -> total = 1.5s.
    assert out.shape[0] == int(1.5 * 16000)
    # first half-second is clip A only.
    np.testing.assert_allclose(out[:8000], a[:8000], atol=1e-5)


def test_gain_scaling():
    a = _tone(220, 0.5, 16000, amp=0.4)
    out, _ = mix([(a, 16000)], gains=[0.5], normalize=False)
    np.testing.assert_allclose(out, a * 0.5, atol=1e-6)


def test_snr_places_clip2_below_clip1():
    sig = _tone(300, 1.0, 16000, amp=0.5)
    intr = _tone(700, 1.0, 16000, amp=0.5)
    snr_db = 6.0
    out, _ = mix([(sig, 16000), (intr, 16000)], snr_db=snr_db, normalize=False)
    # Recover the scaled intruder and confirm its level vs the signal.
    scaled_intr = out - sig
    achieved = 20.0 * np.log10(_rms(sig) / _rms(scaled_intr))
    assert achieved == pytest.approx(snr_db, abs=0.3)


def test_normalize_prevents_clipping():
    a = _tone(220, 0.5, 16000, amp=0.9)
    b = _tone(440, 0.5, 16000, amp=0.9)  # sum can reach ~1.8
    out, _ = mix([(a, 16000), (b, 16000)], normalize=True)
    assert float(np.max(np.abs(out))) <= 1.0 + 1e-6
    raw, _ = mix([(a, 16000), (b, 16000)], normalize=False)
    assert float(np.max(np.abs(raw))) > 1.0  # would have clipped without normalize


def test_resamples_to_common_rate():
    a = _tone(200, 0.5, 8000)  # 8 kHz
    b = _tone(200, 0.5, 16000)  # 16 kHz
    out, sr = mix([(a, 8000), (b, 16000)])
    assert sr == 16000  # highest input rate
    assert out.shape[0] == int(0.5 * 16000)


def test_empty_clips():
    out, sr = mix([])
    assert out.shape[0] == 0


def test_length_mismatch_raises():
    a = _tone(220, 0.2, 16000)
    with pytest.raises(ValueError):
        mix([(a, 16000)], gains=[1.0, 1.0])  # 2 gains, 1 clip
