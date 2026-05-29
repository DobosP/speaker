"""Self-interruption suppression (finding realtime-concurrency-5).

With a fail-open (unenrolled) speaker gate + VAD-only barge-in + no AEC, the
assistant's own TTS can self-interrupt. These tests cover the threshold/gate
logic added to keep that from happening while still letting a genuine user
barge-in through. Everything here is pure: injected embeddings / playback
levels, a fake clock for the watchdog -- no sherpa-onnx, no models, no audio
device.
"""
from __future__ import annotations

import logging

import pytest

from core.engines.speaker_gate import (
    SpeakerGate,
    passes_output_margin,
    rms,
)
from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine
from core.metrics import MetricsRecorder
from core.watchdog import StuckWatchdog

USER = [1.0, 0.0, 0.0, 0.0]
OTHER = [0.0, 1.0, 0.0, 0.0]


# --- level helpers ---------------------------------------------------------


def test_rms_of_empty_is_zero():
    assert rms([]) == 0.0


def test_rms_of_constant_block():
    assert rms([0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)


def test_output_margin_fails_open_when_nothing_playing():
    # playback_level == 0 -> no echo risk -> never suppress a real interrupt.
    assert passes_output_margin(0.001, 0.0, margin_db=6.0) is True


def test_output_margin_rejects_quiet_speech_over_loud_playback():
    # Speech at the same level as playback is plausibly the assistant's own
    # echo; it must not clear a +6 dB margin.
    assert passes_output_margin(0.10, 0.10, margin_db=6.0) is False


def test_output_margin_accepts_speech_well_above_playback():
    # ~+12 dB over playback -> a real user talking over the assistant.
    assert passes_output_margin(0.40, 0.10, margin_db=6.0) is True


def test_output_margin_silent_speech_never_passes_when_playing():
    assert passes_output_margin(0.0, 0.10, margin_db=6.0) is False


# --- SpeakerGate conservative fallback (unenrolled) ------------------------


def test_unenrolled_gate_legacy_fail_open_without_margin():
    # No margin requested -> exact legacy behaviour (always True).
    gate = SpeakerGate(embed_fn=lambda s, sr: None)
    assert gate.accept([0.5, 0.5], 16000) is True
    assert gate.accept([0.5, 0.5], 16000, playback_level=0.5, output_margin_db=0.0) is True


def test_unenrolled_gate_suppresses_echo_with_margin():
    gate = SpeakerGate(embed_fn=lambda s, sr: None)
    assert not gate.is_enrolled
    # Detected speech == playback level => below the +6 dB margin => suppressed.
    quiet = [0.1, 0.1, 0.1, 0.1]
    assert gate.accept(quiet, 16000, playback_level=0.1, output_margin_db=6.0) is False


def test_unenrolled_gate_allows_real_barge_in_with_margin():
    gate = SpeakerGate(embed_fn=lambda s, sr: None)
    loud = [0.4, 0.4, 0.4, 0.4]  # ~+12 dB over playback
    assert gate.accept(loud, 16000, playback_level=0.1, output_margin_db=6.0) is True


def test_unenrolled_gate_fails_open_when_idle_even_with_margin():
    # The margin only bites while the assistant is actually playing audio.
    gate = SpeakerGate(embed_fn=lambda s, sr: None)
    assert gate.accept([0.001], 16000, playback_level=0.0, output_margin_db=6.0) is True


def test_enrolled_gate_ignores_margin_and_uses_identity():
    # When enrolled, speaker-ID does the rejection; the margin args are inert.
    gate = SpeakerGate(threshold=0.5, embed_fn=lambda s, sr: USER)
    gate.enroll_embedding(USER)
    assert gate.accept([0.0], 16000, playback_level=0.9, output_margin_db=6.0) is True
    gate_other = SpeakerGate(threshold=0.5, embed_fn=lambda s, sr: OTHER)
    gate_other.enroll_embedding(USER)
    assert gate_other.accept([0.0], 16000, playback_level=0.0, output_margin_db=6.0) is False


# --- SherpaOnnxEngine._looks_like_user output-activity gating ---------------


def _engine(**cfg) -> SherpaOnnxEngine:
    return SherpaOnnxEngine(SherpaConfig(**cfg))


def test_looks_like_user_no_gate_suppresses_echo_during_playback():
    eng = _engine(barge_in_output_margin_db=6.0)
    eng._playback_level = 0.2  # assistant is playing
    assert eng._looks_like_user([0.2, 0.2, 0.2]) is False  # at playback level


def test_looks_like_user_no_gate_allows_loud_user_during_playback():
    eng = _engine(barge_in_output_margin_db=6.0)
    eng._playback_level = 0.1
    assert eng._looks_like_user([0.5, 0.5, 0.5]) is True


def test_looks_like_user_no_gate_fails_open_when_idle():
    eng = _engine(barge_in_output_margin_db=6.0)
    eng._playback_level = 0.0  # nothing playing
    assert eng._looks_like_user([0.001]) is True


def test_looks_like_user_margin_zero_is_pure_fail_open():
    eng = _engine(barge_in_output_margin_db=0.0)
    eng._playback_level = 0.5
    assert eng._looks_like_user([0.5, 0.5]) is True  # guard disabled


def test_looks_like_user_enrolled_gate_takes_precedence():
    eng = _engine(barge_in_output_margin_db=6.0)
    gate = SpeakerGate(threshold=0.5, embed_fn=lambda s, sr: USER)
    gate.enroll_embedding(USER)
    eng._speaker_gate = gate
    eng._playback_level = 0.9  # would suppress under the unenrolled path
    assert eng._looks_like_user([0.0]) is True  # identity match wins


# --- playback-level EWMA ----------------------------------------------------


def test_note_playback_level_tracks_and_decays():
    eng = _engine()
    assert eng._playback_level == 0.0
    eng._note_playback_level([0.4, 0.4, 0.4, 0.4])
    after_loud = eng._playback_level
    assert after_loud > 0.0
    # Feeding silence releases the level back toward zero (slow release).
    for _ in range(20):
        eng._note_playback_level([0.0, 0.0, 0.0, 0.0])
    assert eng._playback_level < after_loud
    assert eng._playback_level == pytest.approx(0.0, abs=1e-2)


# --- watchdog-storm debounce hook ------------------------------------------


def test_note_barge_in_storm_arms_debounce_window():
    eng = _engine(barge_in_suppress_sec=0.5)
    assert eng._barge_in_suppressed_until == 0.0
    eng.note_barge_in_storm()
    assert eng._barge_in_suppressed_until > 0.0


def test_watchdog_on_storm_hook_fires_once_per_storm():
    t = [0.0]
    rec = MetricsRecorder(clock=lambda: t[0])
    calls = {"n": 0}
    wd = StuckWatchdog(
        rec, clock=lambda: t[0], on_storm=lambda: calls.__setitem__("n", calls["n"] + 1)
    )
    wd.BARGE_IN_STORM_WINDOW_SEC = 1.0
    wd.BARGE_IN_STORM_THRESHOLD = 3
    for ts in (0.0, 0.3, 0.5):
        t[0] = ts
        wd.note_barge_in()
    wd.tick()  # 3 within the window -> storm
    assert calls["n"] == 1
    assert wd.in_storm is True
    # Ticking again inside the suppression window does not re-fire.
    wd.tick()
    assert calls["n"] == 1


def test_watchdog_on_storm_hook_failure_does_not_break_tick(caplog):
    t = [0.0]
    rec = MetricsRecorder(clock=lambda: t[0])

    def boom():
        raise RuntimeError("hook blew up")

    wd = StuckWatchdog(rec, clock=lambda: t[0], on_storm=boom)
    wd.BARGE_IN_STORM_WINDOW_SEC = 1.0
    wd.BARGE_IN_STORM_THRESHOLD = 2
    t[0] = 0.0
    wd.note_barge_in()
    t[0] = 0.2
    wd.note_barge_in()
    with caplog.at_level(logging.WARNING, logger="speaker.watchdog"):
        wd.tick()  # must not raise despite the hook failing
    assert "barge-in storm" in caplog.text


def test_watchdog_no_storm_no_hook():
    t = [0.0]
    rec = MetricsRecorder(clock=lambda: t[0])
    calls = {"n": 0}
    wd = StuckWatchdog(
        rec, clock=lambda: t[0], on_storm=lambda: calls.__setitem__("n", calls["n"] + 1)
    )
    wd.BARGE_IN_STORM_THRESHOLD = 3
    wd.note_barge_in()
    wd.note_barge_in()  # only 2 -> below threshold
    wd.tick()
    assert calls["n"] == 0
    assert wd.in_storm is False
