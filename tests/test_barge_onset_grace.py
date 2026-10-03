"""L3 playback-onset barge grace (open-speaker self-interrupt fix).

At playback onset the echo-coherence reference ring is still filling, so its
Welch estimate is unstable and reads the assistant's OWN TTS echo as a barge
(live run-20260617-231807: every reply self-cancelled 0.04-0.24s after speaking
started). The grace suppresses barge-in for ``barge_in_playback_onset_grace_sec``
after the reply's TRUE first audio, at the single fire chokepoint, WITHOUT losing
a real talk-over past the window. Headless -- no audio device, VAD + looks_like_user
forced True so that, absent the grace, the block WOULD fire.
"""
from __future__ import annotations

import time

import numpy as np

from core.engines._aec import PlaybackFIFO
from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine
from core.media_session import CaptureStamp
from core.metrics import TTS_FIRST_AUDIO

_BLK = np.ones(1600, dtype="float32")


class _Vad:
    def is_speech_detected(self) -> bool:
        return True


def _engine(grace: float) -> SherpaOnnxEngine:
    eng = SherpaOnnxEngine(SherpaConfig.from_dict({"barge_in_playback_onset_grace_sec": grace}))
    eng._vad = _Vad()
    eng._looks_like_user = lambda *a, **k: True   # absent the grace, this WOULD fire
    eng._barge_in_fired_this_run = False
    return eng


def test_suppressed_within_the_onset_window():
    eng = _engine(0.30)
    eng._playback_onset_at = time.monotonic()          # reply just started speaking
    assert eng._barge_in_fire_eligible(_BLK, _BLK) is False


def test_fires_after_the_window_real_talkover_preserved():
    # The hard requirement: a real talk-over past the grace still cuts.
    eng = _engine(0.30)
    eng._playback_onset_at = time.monotonic() - 0.5    # well past the 0.30s window
    assert eng._barge_in_fire_eligible(_BLK, _BLK) is True


def test_grace_zero_is_byte_identical_passthrough():
    eng = _engine(0.0)
    eng._playback_onset_at = time.monotonic()
    assert eng._barge_in_fire_eligible(_BLK, _BLK) is True   # disabled -> no suppression


def test_inert_with_no_onset_stamp():
    eng = _engine(0.30)
    eng._playback_onset_at = 0.0                        # nothing playing -> grace inert
    assert eng._barge_in_fire_eligible(_BLK, _BLK) is True


def test_latch_still_wins_over_grace():
    # The one-barge-per-run latch short-circuits before the grace, unchanged.
    eng = _engine(0.30)
    eng._barge_in_fired_this_run = True
    eng._playback_onset_at = time.monotonic() - 0.5
    assert eng._barge_in_fire_eligible(_BLK, _BLK) is False


def test_window_covers_the_synth_leadin():
    # The live self-interrupts fire 0.04-0.24s after synth-start (before audio).
    # Anchored at synth-start, the 0.40s default window covers that whole span.
    eng = _engine(0.40)
    eng._playback_onset_at = time.monotonic() - 0.24   # mid synth lead-in
    assert eng._barge_in_fire_eligible(_BLK, _BLK) is False


# Drive the same physical-output callback and reader context used live, with
# synthetic PCM and a deterministic clock; no microphone/model is involved.
def _pull_audio(eng, *, real=True):
    if eng._fifo is None:
        eng._fifo = PlaybackFIFO(8)
    if real:
        eng._fifo.write(
            np.full(4, 0.25, dtype="float32"), lambda: False,
            playback_generation=eng._playback_generation,
        )
    out = np.zeros((4, 1), dtype="float32")
    eng._audio_cb(out, 4, None, None)
    return out


def _capture_stamp(at):
    return CaptureStamp(
        sample_rate_hz=16000,
        sample_count=1600,
        captured_started_at=at - 0.1,
        captured_at=at,
        capture_epoch=1,
        source_generation=0,
        source_device=None,
        source_sample_start=0,
        source_sample_end=1600,
    )


def test_slow_synthesis_keeps_full_audible_onset_grace(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    eng = _engine(0.4)
    assert eng._begin_playback_run_if_current(eng._speak_gen) is False

    # Whole-clip synthesis or device opening consumed far more than the grace.
    clock[0] = 20.0
    np.testing.assert_array_equal(_pull_audio(eng)[:, 0], np.full(4, 0.25))
    clock[0] = 20.1
    assert eng._barge_in_fire_eligible(_BLK, _BLK) is False
    clock[0] = 20.401
    assert eng._barge_in_fire_eligible(_BLK, _BLK) is True


def test_empty_output_keeps_pre_audio_grace_and_stop(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    eng = _engine(0.4)
    assert eng._begin_playback_run_if_current(eng._speak_gen) is False
    clock[0] = 10.1
    eng._first_audio_pending = True  # playback worker arms the first fragment
    _pull_audio(eng, real=False)
    assert eng._playback_audible_onset is None
    assert eng._barge_in_fire_eligible(_BLK, _BLK) is False
    assert eng._barge_watch_active() is False

    # A pre-audio STOP still revokes the generation and releases listening.
    before = eng._speak_gen
    eng.stop_speaking()
    assert eng._speak_gen == before + 1
    assert eng.is_speaking is False


def test_fragment_first_audio_and_dry_gaps_do_not_restart_reply_grace(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    eng = _engine(0.4)
    metrics = []
    eng._cb.on_metric = metrics.append
    assert eng._begin_playback_run_if_current(eng._speak_gen) is False
    eng._first_audio_pending = True
    clock[0] = 20.0
    _pull_audio(eng)

    clock[0] = 21.0
    _pull_audio(eng, real=False)
    assert eng._begin_playback_run_if_current(eng._speak_gen) is True
    eng._first_audio_pending = True  # ordinary next fragment's metric only
    _pull_audio(eng)
    assert metrics == [TTS_FIRST_AUDIO, TTS_FIRST_AUDIO]
    assert eng._playback_onset_for_generation(eng._playback_generation) == 20.0
    assert eng._barge_in_fire_eligible(_BLK, _BLK) is True


def test_reader_context_freezes_audible_onset_for_its_generation(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    eng = _engine(0.4)
    eng._begin_playback_run_if_current(eng._speak_gen)
    clock[0] = 20.0
    _pull_audio(eng)
    context = eng._snapshot_capture_context(_capture_stamp(20.1))
    assert context.playback_onset_at == 20.0

    eng.stop_speaking()
    clock[0] = 30.0
    eng._begin_playback_run_if_current(eng._speak_gen)
    successor = eng._snapshot_capture_context(_capture_stamp(30.1))
    assert successor.playback_generation != context.playback_generation
    assert successor.playback_onset_at == 30.0  # pre-audio fallback, not old audio
    assert context.playback_onset_at == 20.0
    assert eng._barge_in_fire_eligible(
        _BLK, _BLK, now=20.1, playback_onset_at=context.playback_onset_at,
    ) is False


def test_callback_overlapping_stop_and_new_run_cannot_stamp_successor(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    eng = _engine(0.4)
    eng._begin_playback_run_if_current(eng._speak_gen)
    predecessor = eng._playback_generation
    fifo = PlaybackFIFO(8)
    eng._fifo = fifo
    fifo.write(
        np.full(4, 0.25, dtype="float32"), lambda: False,
        playback_generation=predecessor,
    )
    read = fifo.read_into

    def read_across_replacement(out, **kwargs):
        count = read(out, **kwargs)
        eng.stop_speaking()
        clock[0] = 30.0
        eng._begin_playback_run_if_current(eng._speak_gen)
        clock[0] = 31.0  # retired callback resumes after successor admission
        return count

    monkeypatch.setattr(fifo, "read_into", read_across_replacement)
    _pull_audio(eng, real=False)
    assert eng._playback_audible_onset == (predecessor, 31.0)
    assert eng._playback_onset_for_generation(eng._playback_generation) == 30.0
    monkeypatch.setattr(fifo, "read_into", read)
    clock[0] = 32.0
    _pull_audio(eng)
    assert eng._playback_onset_for_generation(eng._playback_generation) == 32.0
    assert eng._barge_in_fire_eligible(_BLK, _BLK, now=32.1) is False


def test_callback_replacement_before_fifo_read_binds_successor_pcm(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    eng = _engine(0.4)
    eng._begin_playback_run_if_current(eng._speak_gen)
    fifo = PlaybackFIFO(8)
    eng._fifo = fifo
    read = fifo.read_into

    def replace_before_read(out, **kwargs):
        eng.stop_speaking()
        clock[0] = 30.0
        eng._begin_playback_run_if_current(eng._speak_gen)
        fifo.write(
            np.full(4, 0.5, dtype="float32"), lambda: False,
            playback_generation=eng._playback_generation,
        )
        clock[0] = 40.0  # successor synthesis grace already elapsed
        return read(out, **kwargs)

    monkeypatch.setattr(fifo, "read_into", replace_before_read)
    out = _pull_audio(eng, real=False)
    np.testing.assert_array_equal(out[:, 0], np.full(4, 0.5))
    assert eng._playback_onset_for_generation(eng._playback_generation) == 40.0
    assert eng._barge_in_fire_eligible(_BLK, _BLK, now=40.1) is False
    assert eng._barge_in_fire_eligible(_BLK, _BLK, now=40.5) is True


def test_successor_onset_survives_predecessor_fade_in_same_callback(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    eng = _engine(0.4)
    eng._play_sr = 16000
    eng._begin_playback_run_if_current(eng._speak_gen)
    fifo = PlaybackFIFO(16)
    eng._fifo = fifo
    head, successor = object(), object()
    assert fifo.open_tag(head)
    fifo.write(
        np.full(8, 0.25, dtype="float32"), lambda: False,
        tag=head, playback_generation=eng._playback_generation,
    )
    clock[0] = 20.0
    _pull_audio(eng, real=False)  # establish head playback before interruption
    assert fifo.interrupt_tags("interrupted", 2) == (head,)
    eng._speaking.clear()  # the same stop/new-run transition as the engine
    clock[0] = 30.0
    eng._begin_playback_run_if_current(eng._speak_gen)
    assert fifo.open_tag(successor)
    fifo.write(
        np.full(4, 0.5, dtype="float32"), lambda: False,
        tag=successor, playback_generation=eng._playback_generation,
    )
    clock[0] = 40.0
    out = np.zeros((6, 1), dtype="float32")
    eng._audio_cb(out, 6, None, None)
    np.testing.assert_array_equal(out[2:, 0], np.full(4, 0.5))
    # Successor starts after two retained fade samples, not at the old run's
    # onset or at callback sample zero. Its full audible grace is preserved.
    assert eng._playback_onset_for_generation(eng._playback_generation) == 40.0 + 2 / 16000
    assert eng._barge_in_fire_eligible(_BLK, _BLK, now=40.1) is False
    assert eng._barge_in_fire_eligible(_BLK, _BLK, now=40.5) is True
