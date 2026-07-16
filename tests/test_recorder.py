"""Tests for the session recorder (core.recorder.WavRecorder).

Proves the WAV it writes is exactly what the replay engine's loader consumes,
so a recorded session can be replayed and frozen into a regression test.
"""
from __future__ import annotations

import os
import stat
import wave

import numpy as np

from core.engines.file_replay import load_waveform
from core.recorder import WavRecorder, sidecar_wav_path


def test_sidecar_path_is_stable_with_or_without_wav_suffix():
    assert sidecar_wav_path("run.wav", "pre-dsp") == "run.pre-dsp.wav"
    assert sidecar_wav_path("run.WAV", "ref") == "run.ref.wav"
    assert sidecar_wav_path("run", "pre-dsp") == "run.pre-dsp.wav"


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


def test_recording_file_is_private_even_under_permissive_umask(tmp_path):
    path = tmp_path / "private.wav"
    previous = os.umask(0)
    try:
        rec = WavRecorder(str(path), sample_rate=16000)
        rec.write(np.zeros(16, dtype="float32"))
        rec.close()
    finally:
        os.umask(previous)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


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


def test_file_is_valid_wav_mid_recording_without_close(tmp_path):
    # Kill-safety (2026-07-06, run-20260706-231226): a SIGTERM/SIGKILL'd run
    # never reaches close(), so the on-disk file must already be a valid WAV
    # after the periodic flush -- audio evidence must survive any exit.
    import time as _time

    path = tmp_path / "killed.wav"
    rec = WavRecorder(str(path), sample_rate=16000, flush_sec=0.1)
    rec.write(np.linspace(-1.0, 1.0, 1600, dtype="float32"))  # 0.1 s
    deadline = _time.monotonic() + 3.0
    frames_seen = 0
    while _time.monotonic() < deadline:
        _time.sleep(0.05)
        try:  # read WITHOUT closing the recorder (simulates a killed process)
            with wave.open(str(path), "rb") as wf:
                frames_seen = wf.getnframes()
            if frames_seen >= 1600:
                break
        except (wave.Error, EOFError):
            continue  # header not flushed yet -- keep polling
    assert frames_seen == 1600, "flushed WAV must be readable before close()"
    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000
    rec.close()  # normal close still finalizes cleanly on top


def test_writes_after_flush_are_kept_by_close(tmp_path):
    # A flush mid-stream must not corrupt the append position: frames written
    # after a header patch land after the existing payload, not over it.
    import time as _time

    path = tmp_path / "appended.wav"
    rec = WavRecorder(str(path), sample_rate=16000, flush_sec=0.1)
    first = np.full(800, 0.25, dtype="float32")
    rec.write(first)
    _time.sleep(0.4)  # let the writer flush + patch the header
    second = np.full(800, -0.25, dtype="float32")
    rec.write(second)
    rec.close()
    with wave.open(str(path), "rb") as wf:
        assert wf.getnframes() == 1600
        data = np.frombuffer(wf.readframes(1600), dtype="<i2")
    assert data[0] > 0 and data[-1] < 0  # both halves present, in order


def test_sigterm_handler_raises_keyboard_interrupt():
    # The app installs SIGTERM -> KeyboardInterrupt so a `kill` reuses the
    # exact Ctrl-C teardown (runtime.stop() -> recorder close -> summary).
    import pytest

    from core.app import _sigterm_to_keyboard_interrupt

    with pytest.raises(KeyboardInterrupt):
        _sigterm_to_keyboard_interrupt(15, None)
