"""Tests for the session recorder (core.recorder.WavRecorder).

Proves the WAV it writes is exactly what the replay engine's loader consumes,
so a recorded session can be replayed and frozen into a regression test.
"""
from __future__ import annotations

import wave

import numpy as np

from core.engines.file_replay import load_waveform
from core.recorder import WavRecorder


def test_wavrecorder_writes_16bit_mono_pcm(tmp_path):
    path = tmp_path / "rec.wav"
    rec = WavRecorder(str(path), sample_rate=16000)
    block = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype="float32")
    rec.write(block)
    rec.write(block)
    assert rec.frames == 10
    assert abs(rec.seconds - 10 / 16000) < 1e-9
    rec.close()

    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000
        assert wf.getnframes() == 10
        data = np.frombuffer(wf.readframes(10), dtype="<i2")
    assert data[0] == 0
    assert data[3] == 32767  # +1.0 -> max
    assert data[4] == -32767  # -1.0 -> -max


def test_wavrecorder_clips_out_of_range(tmp_path):
    path = tmp_path / "clip.wav"
    rec = WavRecorder(str(path))
    rec.write(np.array([2.0, -2.0], dtype="float32"))
    rec.close()
    with wave.open(str(path), "rb") as wf:
        data = np.frombuffer(wf.readframes(2), dtype="<i2")
    assert data[0] == 32767 and data[1] == -32767


def test_recording_is_replay_loadable(tmp_path):
    path = tmp_path / "session.wav"
    rec = WavRecorder(str(path), sample_rate=16000)
    rec.write(np.linspace(-1.0, 1.0, 1600, dtype="float32"))  # 0.1 s
    rec.close()

    samples, sample_rate = load_waveform(str(path))
    assert sample_rate == 16000
    assert len(samples) == 1600
    assert samples.dtype == np.float32


def test_close_is_idempotent(tmp_path):
    rec = WavRecorder(str(tmp_path / "x.wav"))
    rec.close()
    rec.close()  # no error
    rec.write(np.array([0.1], dtype="float32"))  # ignored after close
    assert rec.frames == 0
