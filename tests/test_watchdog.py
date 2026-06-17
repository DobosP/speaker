"""Tests for the live stuck-state watchdog (core.watchdog).

Pure: drives the watchdog with a controlled clock so deadlines elapse without
real sleeping. Verifies each heuristic fires at the right moment, only once
per turn, and recovers between events.
"""
from __future__ import annotations

import logging

import pytest

from core.metrics import (
    ASR_FINAL,
    BARGE_IN,
    BARGE_IN_STOP,
    HANDLED_LOCAL,
    LLM_FIRST_TOKEN,
    SUPERSEDED,
    TTS_FIRST_AUDIO,
    MetricsRecorder,
)
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


def test_handled_local_turn_is_not_flagged_stuck(fake_clock, caplog):
    """A turn resolved by the no-LLM intent fast-path stamps HANDLED_LOCAL; it
    has asr_final but never reaches llm_first_token -- not a stall (rc-5)."""
    t, rec, wd = _make(fake_clock)
    wd.LLM_FIRST_TOKEN_DEADLINE_SEC = 0.5
    rec.mark(ASR_FINAL)
    rec.mark(HANDLED_LOCAL)
    t[0] = 10.0  # well past the deadline
    with caplog.at_level(logging.WARNING, logger="speaker.watchdog"):
        wd.tick()
    assert "llm stuck" not in caplog.text
    assert "tts stuck" not in caplog.text


def test_superseded_turn_is_not_flagged_stuck(fake_clock, caplog):
    """A turn preempted by a newer final (newest-input-wins) is cancelled
    pre-answer: it has asr_final but never reaches llm_first_token. The runtime
    stamps SUPERSEDED on it (via mark_superseded_turn) so the watchdog skips it
    instead of mis-reading the gap as a stalled LLM (rc-5). Mirrors the real
    sequence: new final's ASR_FINAL banks the old turn, then it's marked."""
    t, rec, wd = _make(fake_clock)
    wd.LLM_FIRST_TOKEN_DEADLINE_SEC = 0.5
    wd.TTS_FIRST_AUDIO_DEADLINE_SEC = 0.5
    rec.mark(ASR_FINAL)          # turn 0 (will be superseded)
    t[0] = 0.2
    rec.mark(ASR_FINAL)          # turn 1's final arrives -> banks turn 0
    rec.mark_superseded_turn()   # stamp the just-banked turn 0
    rec.mark(LLM_FIRST_TOKEN)    # turn 1 proceeds normally...
    rec.mark(TTS_FIRST_AUDIO)    # ...to completion
    t[0] = 10.0                  # well past both deadlines
    with caplog.at_level(logging.WARNING, logger="speaker.watchdog"):
        wd.tick()
    assert "stuck" not in caplog.text  # neither the superseded nor the done turn flags


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


def test_barged_in_turn_is_not_flagged_tts_stuck(fake_clock, caplog):
    """A turn the user barged into (BARGE_IN stamped) may have llm_first_token
    but never tts_first_audio -- that is an interrupt, not a stall. The live run
    surfaced this exact false positive on a cancelled turn."""
    t, rec, wd = _make(fake_clock)
    wd.TTS_FIRST_AUDIO_DEADLINE_SEC = 0.5
    rec.mark(ASR_FINAL)
    t[0] = 0.1
    rec.mark(LLM_FIRST_TOKEN)
    rec.mark(BARGE_IN)  # user talked over the reply before it produced audio
    t[0] = 10.0  # well past the deadline
    with caplog.at_level(logging.WARNING, logger="speaker.watchdog"):
        wd.tick()
    assert "tts stuck" not in caplog.text


def test_stop_command_abort_turn_is_not_flagged_stuck(fake_clock, caplog):
    """A "stop" command that aborted playback stamps BARGE_IN_STOP; same rule --
    neither the tts-stuck nor llm-stuck check should fire for it."""
    t, rec, wd = _make(fake_clock)
    wd.TTS_FIRST_AUDIO_DEADLINE_SEC = 0.5
    wd.LLM_FIRST_TOKEN_DEADLINE_SEC = 0.5
    rec.mark(ASR_FINAL)
    rec.mark(LLM_FIRST_TOKEN)
    rec.mark(BARGE_IN_STOP)
    t[0] = 10.0
    with caplog.at_level(logging.WARNING, logger="speaker.watchdog"):
        wd.tick()
    assert "tts stuck" not in caplog.text
    assert "llm stuck" not in caplog.text


def test_barge_in_before_first_token_is_not_flagged_llm_stuck(fake_clock, caplog):
    """Cancelled before the LLM produced a token: BARGE_IN present, no
    llm_first_token -- an interrupt, not an llm stall."""
    t, rec, wd = _make(fake_clock)
    wd.LLM_FIRST_TOKEN_DEADLINE_SEC = 0.5
    rec.mark(ASR_FINAL)
    rec.mark(BARGE_IN)
    t[0] = 10.0
    with caplog.at_level(logging.WARNING, logger="speaker.watchdog"):
        wd.tick()
    assert "llm stuck" not in caplog.text


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


# --- capture-stream lifecycle (PR-3 PortAudio recovery integration) -------


def test_recovering_state_suppresses_false_stalled_warning(fake_clock, caplog):
    """While the engine is mid-recovery (its capture loop is sleeping
    through PortAudio backoffs), the heartbeat will legitimately go
    silent for several seconds. The watchdog must NOT fire the
    'audio thread stalled' warning during that window -- that was
    the noisy mis-attribution audited in PR-3."""
    t, rec, wd = _make(fake_clock)
    wd.CAPTURE_SILENT_DEADLINE_SEC = 0.5
    wd.note_heartbeat()  # at t=0.0
    # Engine reports it's recovering -- heartbeat will stop coming.
    wd.note_capture_state("recovering", "PortAudio -9985")
    t[0] = 1.0  # deadline exceeded by 2x
    with caplog.at_level(logging.WARNING, logger="speaker.watchdog"):
        wd.tick()
    # No false-positive: state was recovering, so the silence is expected.
    assert "capture silent" not in caplog.text


def test_silent_warning_resumes_after_state_returns_to_open(fake_clock, caplog):
    """Once the engine reports it recovered (state=open), the watchdog
    re-arms. If the heartbeat then goes truly silent again, the warning
    fires as normal."""
    t, rec, wd = _make(fake_clock)
    wd.CAPTURE_SILENT_DEADLINE_SEC = 0.5
    wd.note_heartbeat()  # t=0.0
    # Brief recovery; engine reports it's back.
    wd.note_capture_state("recovering", "test")
    t[0] = 0.6
    wd.note_capture_state("open", "recovered")
    wd.note_heartbeat()  # fresh heartbeat at t=0.6
    t[0] = 1.3  # now silent for 0.7s -- past the 0.5s deadline
    with caplog.at_level(logging.WARNING, logger="speaker.watchdog"):
        wd.tick()
    assert "capture silent" in caplog.text


def test_fatal_state_logs_capture_lost_error(fake_clock, caplog):
    """When the engine reports FATAL (recovery exhausted), the watchdog
    emits an error-level 'capture lost' message so the run bundle
    records that capture truly stopped (not just stalled)."""
    t, rec, wd = _make(fake_clock)
    with caplog.at_level(logging.ERROR, logger="speaker.watchdog"):
        wd.note_capture_state("fatal", "recovery exhausted")
    assert "capture lost" in caplog.text


def test_open_after_recovering_clears_warning_token(fake_clock, caplog):
    """The first warning per silent-incident is one-shot via
    ``_heartbeat_warned``. Returning to OPEN must reset that token so a
    subsequent stall (after some healthy time) warns again -- otherwise
    a single warned-then-recovered turn would mute the next legitimate
    stall."""
    t, rec, wd = _make(fake_clock)
    wd.CAPTURE_SILENT_DEADLINE_SEC = 0.5

    # First stall + warning.
    wd.note_heartbeat()
    t[0] = 0.6
    with caplog.at_level(logging.WARNING, logger="speaker.watchdog"):
        wd.tick()
    assert caplog.text.count("capture silent") == 1

    # Engine recovers. note_capture_state("open") clears the warning
    # token so the next stall re-fires.
    wd.note_capture_state("open", "recovered")
    wd.note_heartbeat()
    t[0] = 1.0
    # No stall yet (just 0.4s after heartbeat).
    with caplog.at_level(logging.WARNING, logger="speaker.watchdog"):
        wd.tick()
    assert caplog.text.count("capture silent") == 1  # still just the first

    # Now stall again.
    t[0] = 1.7
    with caplog.at_level(logging.WARNING, logger="speaker.watchdog"):
        wd.tick()
    assert caplog.text.count("capture silent") == 2  # warned again


def test_recovering_state_does_not_block_other_checks(fake_clock, caplog):
    """A capture-state warning shouldn't gate the LLM-stuck heuristic --
    those are independent. If the user is in recovering+LLM-stuck both,
    we still want the LLM warning."""
    t, rec, wd = _make(fake_clock)
    wd.LLM_FIRST_TOKEN_DEADLINE_SEC = 0.5
    wd.note_capture_state("recovering", "stuff")
    rec.mark(ASR_FINAL)  # t=0
    t[0] = 0.6
    with caplog.at_level(logging.WARNING, logger="speaker.watchdog"):
        wd.tick()
    assert "llm stuck" in caplog.text
