"""
Tests for ``--record`` session capture (SessionRecorder).

Regression targets:
- Chunked TTS calls ``on_tts_start`` once per sentence; each chunk must become
  its own saved turn, not overwrite the previous turn in memory.
- ``on_interrupt`` must not be wrapped twice (would duplicate barge-in events).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.session_recorder import SessionRecorder, TtsTurn, TARGET_SR


class _FakeRecorder:
    """Minimal stand-in for AudioRecorder (no sounddevice)."""

    device_sample_rate = 22_050
    _noise_floor = 0.02

    def __init__(self):
        self.on_interrupt = None


def test_on_tts_start_twice_flushes_previous_turn():
    """Mimics AudioPlayer._speak_chunked: two sentences → two recorded turns."""
    r = _FakeRecorder()
    sr = SessionRecorder(recorder=r, profile="unit_test")
    sr.start()

    a1 = np.linspace(0, 0.1, 800, dtype=np.float32)
    a2 = np.linspace(0, 0.2, 400, dtype=np.float32)

    sr.on_tts_start(a1, r.device_sample_rate, "first")
    sr.on_mic_chunk(np.zeros(64, dtype=np.float32), 0.01, 0.05)

    sr.on_tts_start(a2, r.device_sample_rate, "second")
    sr.on_mic_chunk(np.ones(64, dtype=np.float32) * 0.03, 0.02, 0.12)

    sr.on_tts_end()

    assert len(sr._turns) == 2, "expected two flushed turns for two on_tts_start + final end"
    assert sr._turns[0].index == 0
    assert sr._turns[1].index == 1
    assert 0 in sr._turn_tts and 1 in sr._turn_tts
    assert len(sr._turn_tts[0]) == 800 and len(sr._turn_tts[1]) == 400
    assert 0 in sr._turn_mic and 1 in sr._turn_mic


def test_on_tts_end_only_single_chunk():
    r = _FakeRecorder()
    sr = SessionRecorder(recorder=r, profile="unit_test")
    sr.start()
    sr.on_tts_start(np.ones(100, dtype=np.float32) * 0.05, r.device_sample_rate)
    sr.on_tts_end()
    assert len(sr._turns) == 1


def test_attach_recorder_hooks_idempotent():
    """Second _attach_recorder_hooks must not stack wrappers on on_interrupt."""
    r = _FakeRecorder()
    orig_calls: list[str] = []

    def user_interrupt(info=None):
        orig_calls.append("user")

    r.on_interrupt = user_interrupt

    sr = SessionRecorder(recorder=r, profile="unit_test")
    sr.start()
    sr._attach_recorder_hooks()
    sr._attach_recorder_hooks()

    sr._current_turn = TtsTurn(index=0, start_sec=0.0, text="x")
    r.on_interrupt({"rms": 0.1, "threshold": 0.01, "voiced": True, "echo_sim": 0.0})

    assert orig_calls.count("user") == 1
    assert len(sr._current_turn.barge_in_events) == 1


def test_save_writes_turn_npy_and_json(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "utils.session_recorder.RECORDINGS_DIR",
        tmp_path,
    )
    r = _FakeRecorder()
    sr = SessionRecorder(recorder=r, profile="unit_test")
    sr.start()
    sr.on_tts_start(np.ones(220, dtype=np.float32) * 0.1, r.device_sample_rate, "hi")
    sr.on_tts_end()
    sr.stop()

    out = sr.save()
    assert out == sr.session_dir
    turn0 = sr.session_dir / "turn_000"
    assert (turn0 / "tts_16k.npy").exists()
    assert (turn0 / "mic_16k.npy").exists() is False  # no mic chunks
    assert (turn0 / "turn.json").exists()
    with open(turn0 / "turn.json") as f:
        meta = json.load(f)
    assert meta["index"] == 0
    assert meta["text"] == "hi"

    tts = np.load(turn0 / "tts_16k.npy")
    assert tts.dtype == np.float32
    assert len(tts) > 0


def test_save_without_attachment_uses_target_sr_for_metadata(monkeypatch, tmp_path):
    """save() must not assume recorder is non-None (lazy attach edge case)."""
    monkeypatch.setattr("utils.session_recorder.RECORDINGS_DIR", tmp_path)
    sr = SessionRecorder(recorder=None, profile="lazy")
    sr.start()
    sr.stop()
    # No turns — still writes metadata
    sr.save()
    with open(sr.session_dir / "metadata.json") as f:
        meta = json.load(f)
    assert meta["num_tts_turns"] == 0
    assert meta["device_sample_rate"] == TARGET_SR
