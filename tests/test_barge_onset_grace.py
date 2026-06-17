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

from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine

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


class _Fifo:
    def __init__(self, n: int):
        self.n = n

    def read_into(self, view) -> int:
        view[: self.n] = 0.1
        view[self.n :] = 0.0
        return self.n


def test_multi_sentence_reply_does_not_reopen_the_grace():
    # _first_audio_pending re-arms per sentence; the onset stamp must move ONLY on
    # the reply's first audio (reset happens on silent->speaking), so a later
    # sentence cannot re-open the grace mid-reply.
    eng = SherpaOnnxEngine(SherpaConfig())
    eng._speaking.set()
    eng._fifo = _Fifo(64)
    out = np.zeros((64, 1), dtype="float32")

    eng._first_audio_pending = True
    eng._audio_cb(out, 64, None, None)
    first = eng._playback_onset_at
    assert first > 0.0

    time.sleep(0.01)
    eng._first_audio_pending = True                    # next sentence re-arms it
    eng._audio_cb(out, 64, None, None)
    assert eng._playback_onset_at == first             # stamp did NOT move
