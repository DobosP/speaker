"""Tests for the PURE helpers in tools.voice_audition.

The actual synthesis loop (run_audition / _build_engine_or_die) needs a real
Kokoro/VITS model on disk, so it self-skips elsewhere (real_model tier, run
via ``python -m tools.voice_audition`` directly). These tests cover the
logic that decides WHICH voices to audition and how results are presented --
no audio device, no models.
"""
from __future__ import annotations

import numpy as np
import pytest

from tools.voice_audition import format_table, select_voices, spectral_centroid


def test_select_voices_includes_default_plus_all_configured():
    voices = {"warm": 16, "soft": 3, "deep": 9}
    rows = select_voices(voices, default_sid=0)

    assert rows[0] == ("default", 0, None)
    assert ("warm", 16, {"voice": "warm"}) in rows
    assert ("soft", 3, {"voice": "soft"}) in rows
    assert ("deep", 9, {"voice": "deep"}) in rows
    assert len(rows) == 4


def test_select_voices_narrows_to_requested_names():
    voices = {"warm": 16, "soft": 3, "deep": 9}
    rows = select_voices(voices, default_sid=0, requested=["soft"])

    names = [r[0] for r in rows]
    assert names == ["default", "soft"]


def test_select_voices_empty_config_is_just_default():
    rows = select_voices({}, default_sid=0)
    assert rows == [("default", 0, None)]


def test_select_voices_skips_bad_sid_entries():
    voices = {"warm": "not-a-number"}
    rows = select_voices(voices, default_sid=0)
    assert rows == [("default", 0, None)]


def test_spectral_centroid_low_tone_is_low_high_tone_is_high():
    sr = 24000
    t = np.arange(int(sr * 0.5)) / sr
    low = (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32")
    high = (0.3 * np.sin(2 * np.pi * 8000 * t)).astype("float32")

    c_low = spectral_centroid(low, sr)
    c_high = spectral_centroid(high, sr)

    assert c_low == pytest.approx(220.0, abs=5.0)
    assert c_high == pytest.approx(8000.0, abs=5.0)
    assert c_high > c_low


def test_spectral_centroid_empty_is_none():
    assert spectral_centroid(np.zeros(0, dtype="float32"), 24000) is None


def test_format_table_lists_every_row_and_its_wav():
    rows = [
        {
            "name": "default", "sid": 0, "rms": 0.07, "peak": 0.5,
            "clip_pct": 0.0, "dc_offset": 0.0, "hf_ratio": 0.01,
            "spectral_flatness": 0.02, "centroid_hz": 1800.0,
            "wav": "/tmp/default_sid0.wav",
        },
        {
            "name": "soft", "sid": 3, "rms": 0.06, "peak": 0.4,
            "clip_pct": 0.0, "dc_offset": 0.0, "hf_ratio": 0.02,
            "spectral_flatness": 0.03, "centroid_hz": 1700.0,
            "wav": "/tmp/soft_sid3.wav",
        },
    ]
    out = format_table(rows)
    assert "default" in out and "soft" in out
    assert "/tmp/default_sid0.wav" in out
    assert "/tmp/soft_sid3.wav" in out


def test_format_table_empty():
    assert format_table([]) == "(no voices auditioned)"
