"""Tests for the live stuck-state watchdog (core.watchdog).

Pure: drives the watchdog with a controlled clock so deadlines elapse without
real sleeping. Verifies each heuristic fires at the right moment, only once
per turn, and recovers between events.
"""
from __future__ import annotations

import logging

import pytest

from core.metrics import ASR_FINAL, LLM_FIRST_TOKEN, TTS_FIRST_AUDIO, MetricsRecorder
from core.watchdog import StuckWatchdog


@pytest.fixture
def fake_clock():
    """A controllable clock shared between MetricsRecorder and StuckWatchdog
    so stamps and 'now' use the same monotonic reference."""
    t = [0.0]
    return t, lambda: t[0]


def _make(fake_clock):
    t, now = fake_clock
    recorder = MetricsRecorder(clock=now)
    wd = StuckWatchdog(recorder, clock=now)
    return t, recorder, wd


def test_warns_when_llm_never_produces_first_token(fake_clock, caplog):
    t, rec, wd = _make(fake_clock)
    wd.LLM_FIRST_TOKEN_DEADLINE_SEC = 0.5
    rec.mark(ASR_FINAL)  # stamps at t=0.0
    t[0] = 0.4
    with caplog.at_level(logging.WARNING, logger="speaker.watchdog"):
        wd.tick()
    assert "llm stuck" not in caplog.text  # still inside the deadline
    t[0] = 0.6
    with caplog.at_level(logging.WARNING, logger="speaker.watchdog"):
        wd.tick()
    assert "llm stuck: turn 0 had asr_final but no llm_first_token" in caplog.text


def test_does_not_warn_once_llm_first_token_arrives(fake_clock, caplog):
    t, rec, wd = _make(fake_clock)
    wd.LLM_FIRST_TOKEN_DEADLINE_SEC = 0.5
    rec.mark(ASR_FINAL)
    t[0] = 0.3
    rec.mark(LLM_FIRST_TOKEN)  # first token arrived
    t[0] = 10.0  # well past the deadline
    with caplog.at_level(logging.WARNING, logger="speaker.watchdog"):
        wd.tick()
    assert "llm stuck" not in caplog.text


def test_warns_when_tts_never_produces_audio_after_first_token(fake_clock, caplog):
    t, rec, wd = _make(fake_clock)
    wd.TTS_FIRST_AUDIO_DEADLINE_SEC = 0.5
    rec.mark(ASR_FINAL)
    t[0] = 0.1
    rec.mark(LLM_FIRST_TOKEN)
    t[0] = 0.7  # > 0.5s after first token, no audio yet
    with caplog.at_level(logging.WARNING, logger="speaker.watchdog"):
        wd.tick()
    assert "tts stuck: turn 0 had llm_first_token but no tts_first_audio" in caplog.text


def test_does_not_warn_when_tts_audio_arrives(fake_clock, caplog):
    t, rec, wd = _make(fake_clock)
    wd.TTS_FIRST_AUDIO_DEADLINE_SEC = 0.5
    rec.mark(ASR_FINAL)
    t[0] = 0.1
    rec.mark(LLM_FIRST_TOKEN)
    t[0] = 0.2
    rec.mark(TTS_FIRST_AUDIO)
    t[0] = 10.0
    with caplog.at_level(logging.WARNING, logger="speaker.watchdog"):
        wd.tick()
    assert "tts stuck" not in caplog.text


def test_warns_only_once_per_turn(fake_clock, caplog):
    """Repeated ticks on a stalled turn must not flood the log."""
    t, rec, wd = _make(fake_clock)
    wd.LLM_FIRST_TOKEN_DEADLINE_SEC = 0.5
    rec.mark(ASR_FINAL)
    t[0] = 0.6
    with caplog.at_level(logging.WARNING, logger="speaker.watchdog"):
        wd.tick()
        wd.tick()
        wd.tick()
    assert caplog.text.count("llm stuck: turn 0") == 1


def test_capture_silence_warns_only_after_heartbeat_seen(fake_clock, caplog):
    """Engines that never emit heartbeats (console/replay) must not spuriously
    trip the silence check."""
    t, rec, wd = _make(fake_clock)
    wd.CAPTURE_SILENT_DEADLINE_SEC = 0.5
    t[0] = 100.0  # plenty of time has passed since process start
    with caplog.at_level(logging.WARNING, logger="speaker.watchdog"):
        wd.tick()
    assert "capture silent" not in caplog.text


def test_capture_silence_warns_when_heartbeat_stops(fake_clock, caplog):
    t, rec, wd = _make(fake_clock)
    wd.CAPTURE_SILENT_DEADLINE_SEC = 0.5
    wd.note_heartbeat()  # at t=0.0
    t[0] = 0.4
    with caplog.at_level(logging.WARNING, logger="speaker.watchdog"):
        wd.tick()
    assert "capture silent" not in caplog.text
    t[0] = 0.6
    with caplog.at_level(logging.WARNING, logger="speaker.watchdog"):
        wd.tick()
    assert "capture silent: no heartbeat for" in caplog.text


def test_capture_silence_rearms_after_heartbeat_resumes(fake_clock, caplog):
    t, rec, wd = _make(fake_clock)
    wd.CAPTURE_SILENT_DEADLINE_SEC = 0.5
    wd.note_heartbeat()  # t=0
    t[0] = 0.6
    with caplog.at_level(logging.WARNING, logger="speaker.watchdog"):
        wd.tick()
    first_count = caplog.text.count("capture silent")
    assert first_count == 1
    # Engine recovers and reports a heartbeat again.
    wd.note_heartbeat()
    t[0] = 0.7
    with caplog.at_level(logging.WARNING, logger="speaker.watchdog"):
        wd.tick()
    assert caplog.text.count("capture silent") == 1  # still just the first
    # Then silent again, second incident triggers a second warning.
    t[0] = 1.4
    with caplog.at_level(logging.WARNING, logger="speaker.watchdog"):
        wd.tick()
    assert caplog.text.count("capture silent") == 2


def test_barge_in_storm_warns_when_threshold_exceeded(fake_clock, caplog):
    t, rec, wd = _make(fake_clock)
    wd.BARGE_IN_STORM_WINDOW_SEC = 1.0
    wd.BARGE_IN_STORM_THRESHOLD = 3
    t[0] = 0.0
    wd.note_barge_in()
    t[0] = 0.3
    wd.note_barge_in()
    with caplog.at_level(logging.WARNING, logger="speaker.watchdog"):
        wd.tick()
    assert "barge-in storm" not in caplog.text  # only 2 in the window
    t[0] = 0.5
    wd.note_barge_in()  # 3rd within the 1s window
    with caplog.at_level(logging.WARNING, logger="speaker.watchdog"):
        wd.tick()
    assert "barge-in storm: 3 detections" in caplog.text


def test_barge_in_storm_drops_stale_entries(fake_clock, caplog):
    """Spaced-out barge-ins shouldn't be a storm."""
    t, rec, wd = _make(fake_clock)
    wd.BARGE_IN_STORM_WINDOW_SEC = 0.5
    wd.BARGE_IN_STORM_THRESHOLD = 3
    for i in range(5):
        t[0] = i * 1.0  # each barge-in is 1s apart, window is 0.5s
        wd.note_barge_in()
        with caplog.at_level(logging.WARNING, logger="speaker.watchdog"):
            wd.tick()
    assert "barge-in storm" not in caplog.text


def test_thread_lifecycle_starts_and_stops_cleanly():
    """Daemon thread starts, runs at least one tick, and joins on stop."""
    rec = MetricsRecorder()
    wd = StuckWatchdog(rec, interval_sec=0.01)
    wd.start()
    wd.start()  # idempotent
    wd.stop()
    wd.stop()  # idempotent
