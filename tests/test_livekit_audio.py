"""Tests for the LiveKit worker's pure audio helpers."""
import numpy as np

from remote.livekit_agent import (
    float32_to_pcm_int16,
    pcm_int16_to_float32,
    resample_linear,
)


def test_pcm_int16_to_float32_range():
    out = pcm_int16_to_float32(np.array([0, 32767, -32768], dtype=np.int16))
    assert abs(out[0]) < 1e-6
    assert 0.99 < out[1] <= 1.0
    assert -1.0001 <= out[2] <= -0.99


def test_float32_to_pcm_int16_clips_and_types():
    out = float32_to_pcm_int16(np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32))
    assert out.dtype == np.int16
    assert out[1] == 32767 and out[2] == -32767 and out[3] == 32767


def test_resample_same_rate_is_identity():
    x = np.arange(10, dtype=np.float32)
    assert np.allclose(resample_linear(x, 16000, 16000), x)


def test_resample_empty_stays_empty():
    assert resample_linear(np.zeros(0, dtype=np.float32), 16000, 48000).size == 0


def test_resample_changes_length_proportionally():
    x = np.sin(np.linspace(0, 3.14, 100)).astype(np.float32)
    assert abs(len(resample_linear(x, 16000, 48000)) - 300) <= 2
    assert abs(len(resample_linear(x, 48000, 16000)) - 33) <= 2
