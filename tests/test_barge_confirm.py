"""Duck-then-confirm word-gated barge-in (2026-07-02, R12 slice).

The live dichotomy on an open nonlinear speaker (run-20260702-220207): with the
coherence veto ON every correct DTD fire is vetoed (no barge); with it OFF the
DTD also fires on the assistant's own imperfectly-cancelled echo
(self-interrupt). The word gate makes the acoustic trigger REVERSIBLE: duck
playback, require real transcribed words within a confirm window, hard-fire only
then -- else restore. These pin the state machine with fakes: no audio device,
no models, no threads.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from core.engine import EngineCallbacks
from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine


class _FakeStream:
    def __init__(self):
        self.fed_blocks = 0

    def accept_waveform(self, sr, samples):
        self.fed_blocks += 1


class _FakeRecognizer:
    """Streaming-recognizer stand-in: yields scripted partials in order."""

    def __init__(self, partials):
        self._partials = list(partials)
        self._i = -1
        self.resets = 0

    def is_ready(self, stream):
        return False  # decode loop is a no-op for the fake

    def decode_stream(self, stream):  # pragma: no cover - never reached
        pass

    def get_result(self, stream):
        # Advance one scripted partial per call, then hold the last.
        if self._i < len(self._partials) - 1:
            self._i += 1
        return self._partials[self._i] if self._partials else ""

    def reset(self, stream):
        self.resets += 1


class _Rec:
    def __init__(self):
        self.barges = 0
        self.metrics: list[str] = []

    def callbacks(self) -> EngineCallbacks:
        return EngineCallbacks(
            on_barge_in=self._on_barge, on_metric=self._on_metric
        )

    def _on_barge(self):
        self.barges += 1

    def _on_metric(self, name, **kw):
        self.metrics.append(name)


def _engine(rec: _Rec | None = None, **cfg) -> SherpaOnnxEngine:
    eng = SherpaOnnxEngine(SherpaConfig(barge_confirm_enabled=True, **cfg))
    if rec is not None:
        eng._cb = rec.callbacks()
    return eng


_BLOCK = np.zeros(1600, dtype="float32")  # one 0.1s block at 16kHz


# --- config -------------------------------------------------------------------


def test_confirm_disabled_by_default():
    c = SherpaConfig()
    assert c.barge_confirm_enabled is False  # shipped default: legacy hard-fire
    assert 0.0 < c.barge_confirm_duck_gain < 1.0


def test_config_from_dict_roundtrip():
    c = SherpaConfig.from_dict(
        {"barge_confirm_enabled": True, "barge_confirm_min_words": 3,
         "barge_confirm_window_sec": 2.0}
    )
    assert c.barge_confirm_enabled is True
    assert c.barge_confirm_min_words == 3
    assert c.barge_confirm_window_sec == 2.0


# --- begin: duck + window ------------------------------------------------------


def test_begin_ducks_and_opens_window():
    rec = _Rec()
    eng = _engine(rec)
    r, s = _FakeRecognizer(["old partial"]), _FakeStream()
    now = time.monotonic()
    eng._begin_barge_confirm(r, s, now)
    assert eng._barge_confirm_active()
    assert eng._duck_gain == eng.config.barge_confirm_duck_gain  # ducked, not cut
    assert eng._confirm_base_text == "old partial"  # only NEW words will count
    assert "barge_in_duck" in rec.metrics
    assert rec.barges == 0  # nothing fired yet


# --- confirm: real words fire the barge ----------------------------------------


def test_new_words_confirm_and_fire():
    rec = _Rec()
    eng = _engine(rec)
    # Partial grows one word per block: "hey" (1 word, below min) -> "hey wait"
    # (2 words -> confirms).
    r, s = _FakeRecognizer(["", "hey", "hey wait"]), _FakeStream()
    now = time.monotonic()
    eng._begin_barge_confirm(r, s, now)  # consumes partial "" as base
    assert eng._barge_confirm_step(r, s, _BLOCK, now + 0.1) is False  # 1 word: not yet
    assert rec.barges == 0
    assert eng._duck_gain < 1.0  # still ducked mid-window
    assert eng._barge_confirm_step(r, s, _BLOCK, now + 0.2) is True   # 2 words: fire
    assert rec.barges == 1
    assert "barge_in_confirmed" in rec.metrics
    assert eng._duck_gain == 1.0                 # volume restored on the way out
    assert not eng._barge_confirm_active()
    assert eng._barge_in_fired_this_run is True  # latch burned only on a REAL fire


def test_stop_command_confirms_alone():
    rec = _Rec()
    eng = _engine(rec, barge_confirm_min_words=2)
    r, s = _FakeRecognizer(["", "stop"]), _FakeStream()
    now = time.monotonic()
    eng._begin_barge_confirm(r, s, now)
    assert eng._barge_confirm_step(r, s, _BLOCK, now + 0.1) is True  # 1 word, but a stop command
    assert rec.barges == 1


def test_stop_during_barge_callback_cannot_publish_handoff_into_next_epoch():
    eng = _engine(barge_confirm_min_words=2)
    metrics: list[str] = []

    def _invalidate_capture_epoch():
        with eng._capture_effect_condition:
            eng._capture_epoch += 1
            eng._capture_stopping.set()

    eng._cb = EngineCallbacks(
        on_barge_in=_invalidate_capture_epoch,
        on_metric=lambda name, **_kwargs: metrics.append(name),
    )
    eng._capture_callback_context.epoch = eng._capture_epoch
    r, s = _FakeRecognizer(["", "stop"]), _FakeStream()
    now = time.monotonic()

    try:
        eng._begin_barge_confirm(r, s, now)
        assert not eng._barge_confirm_step(r, s, _BLOCK, now + 0.1)
    finally:
        del eng._capture_callback_context.epoch

    assert "barge_in_confirmed" in metrics
    assert not eng._confirm_handoff_stream_live


def test_stop_winning_after_barge_callback_entry_atomically_blocks_handoff():
    eng = _engine(barge_confirm_min_words=2)
    callback_entered = threading.Event()
    release_callback = threading.Event()
    result: list[bool] = []

    def _blocking_barge_callback():
        callback_entered.set()
        assert release_callback.wait(timeout=2.0)

    eng._cb = EngineCallbacks(on_barge_in=_blocking_barge_callback)
    capture_epoch = eng._capture_epoch
    r, s = _FakeRecognizer(["", "stop"]), _FakeStream()
    now = time.monotonic()

    def _confirm():
        eng._capture_callback_context.epoch = capture_epoch
        try:
            eng._begin_barge_confirm(r, s, now)
            result.append(
                eng._barge_confirm_step(r, s, _BLOCK, now + 0.1)
            )
        finally:
            del eng._capture_callback_context.epoch

    worker = threading.Thread(target=_confirm)
    worker.start()
    assert callback_entered.wait(timeout=1.0)
    with eng._capture_effect_condition:
        eng._capture_epoch += 1
        eng._capture_stopping.set()
        eng._confirm_handoff_stream_live = False
    release_callback.set()
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert result == [False]
    assert not eng._confirm_handoff_stream_live


def test_cancel_that_command_confirms_below_word_floor():
    rec = _Rec()
    eng = _engine(rec, barge_confirm_min_words=4)
    r, s = _FakeRecognizer(["", "cancel that"]), _FakeStream()
    now = time.monotonic()
    eng._begin_barge_confirm(r, s, now)
    assert eng._barge_confirm_step(r, s, _BLOCK, now + 0.1) is True
    assert rec.barges == 1


# --- reject: echo / silence restores volume ------------------------------------


def test_own_echo_does_not_confirm_and_window_expires():
    rec = _Rec()
    eng = _engine(rec)
    eng._recent_spoken.append("The dragon flew over the misty mountains.")
    # The recognizer "hears" the assistant's own ducked echo.
    r = _FakeRecognizer(["", "the dragon flew", "the dragon flew over"])
    s = _FakeStream()
    now = time.monotonic()
    eng._begin_barge_confirm(r, s, now)
    assert eng._barge_confirm_step(r, s, _BLOCK, now + 0.1) is False  # echo filtered
    # Window expires -> restore + reset stream + retry-suppress armed.
    assert eng._barge_confirm_step(r, s, _BLOCK, now + 99.0) is False
    assert rec.barges == 0
    assert "barge_in_unconfirmed" in rec.metrics
    assert eng._duck_gain == 1.0
    assert not eng._barge_confirm_active()
    assert r.resets == 1                          # echo purged from the stream
    assert eng._barge_in_suppressed_until > now   # can't immediately re-duck
    assert eng._barge_in_fired_this_run is False  # latch NOT burned by a false trigger


def test_silence_expires_without_firing():
    rec = _Rec()
    eng = _engine(rec)
    r, s = _FakeRecognizer([""]), _FakeStream()
    now = time.monotonic()
    eng._begin_barge_confirm(r, s, now)
    assert eng._barge_confirm_step(r, s, _BLOCK, now + 99.0) is False
    assert rec.barges == 0
    assert eng._duck_gain == 1.0


# --- restore paths --------------------------------------------------------------


def test_stop_speaking_closes_window_and_restores_duck():
    eng = _engine(_Rec())
    r, s = _FakeRecognizer([""]), _FakeStream()
    eng._begin_barge_confirm(r, s, time.monotonic())
    assert eng._duck_gain < 1.0
    eng.stop_speaking()
    assert eng._duck_gain == 1.0
    assert not eng._barge_confirm_active()


def test_engine_stop_clears_pending_confirm_handoff_before_reuse():
    eng = _engine(_Rec())
    now = time.monotonic()
    assert eng._publish_confirm_handoff_if_current(
        [_BLOCK],
        [_BLOCK],
        speech_at=now,
        speech_end_at=now,
    )
    assert eng._confirm_handoff_pending is not None

    eng.stop()

    assert not eng._confirm_handoff_stream_live
    assert eng._confirm_handoff_pending is None


# --- the duck is applied by the audio callback ----------------------------------


def test_audio_cb_applies_duck_gain():
    from core.engines._aec import PlaybackFIFO

    eng = _engine(_Rec())
    eng._play_sr = 16000
    eng._playback_level = 0.0
    eng._echo_coherence = None
    eng._first_audio_pending = False
    eng._fifo = PlaybackFIFO(16000)
    eng._fifo.write(np.ones(256, dtype="float32"), should_abort=lambda: False)
    eng._duck_gain = 0.25
    outdata = np.zeros((256, 1), dtype="float32")
    eng._audio_cb(outdata, 256, None, None)
    assert np.allclose(outdata[:, 0], 0.25)  # ducked in place

    # And the far-end/coherence tees see the DUCKED signal (gain applied before
    # the tees), keeping the AEC reference true to what leaves the speaker.
    eng._fifo.write(np.ones(256, dtype="float32"), should_abort=lambda: False)
    eng._duck_gain = 1.0
    outdata2 = np.zeros((256, 1), dtype="float32")
    eng._audio_cb(outdata2, 256, None, None)
    assert np.allclose(outdata2[:, 0], 1.0)  # restored -> unity passthrough


# --- echo filter ----------------------------------------------------------------


def test_reads_like_own_speech_overlap_and_novelty():
    eng = _engine(_Rec())
    eng._recent_spoken.append("Once upon a time there was a dragon named Ember.")
    assert eng._reads_like_own_speech("upon a time there was") is True
    assert eng._reads_like_own_speech("what are you talking about") is False
    assert eng._reads_like_own_speech("") is True  # inaudible -> not user evidence


def test_now_playing_covers_ring_eviction():
    # Live bug (run-20260702-223217): a long reply enqueues more sentences than
    # the ring holds, evicting the CURRENTLY PLAYING one -- its echo then passed
    # the filter and false-confirmed. _now_playing is the authoritative reference.
    eng = _engine(_Rec())
    for i in range(64):  # flood the ring far past maxlen
        eng._recent_spoken.append(f"filler sentence number {i} with unique words")
    eng._now_playing = "He would never abandon the outbreak in my village."
    assert eng._reads_like_own_speech("he would") is True
    assert eng._reads_like_own_speech("outbreak my") is True
    assert eng._reads_like_own_speech("please stop talking now") is False


# --- unconfirmed windows teach the DTD charts ------------------------------------


class _FakeDTD:
    def __init__(self):
        self.echo_obs: list[tuple] = []

    def observe_echo(self, raw, resid, incoh):
        self.echo_obs.append((raw, resid, incoh))


def test_unconfirmed_window_teaches_dtd_charts():
    rec = _Rec()
    eng = _engine(rec)
    eng._dtd = _FakeDTD()
    eng._echo_coherence = None
    r, s = _FakeRecognizer([""]), _FakeStream()
    now = time.monotonic()
    eng._begin_barge_confirm(r, s, now)
    eng._barge_confirm_step(r, s, _BLOCK, now + 0.1)      # banks one echo obs
    eng._barge_confirm_step(r, s, _BLOCK, now + 0.2)      # banks another
    assert eng._barge_confirm_step(r, s, _BLOCK, now + 99.0) is False  # expiry
    # All banked blocks (incl. the expiry block) were fed to the charts, so the
    # echo level is LEARNED and the trigger flood decays.
    assert len(eng._dtd.echo_obs) == 3
    assert eng._confirm_echo_obs == []  # cleared with the window


def test_confirmed_window_discards_echo_obs():
    rec = _Rec()
    eng = _engine(rec)
    eng._dtd = _FakeDTD()
    eng._echo_coherence = None
    r, s = _FakeRecognizer(["", "hey wait"]), _FakeStream()
    now = time.monotonic()
    eng._begin_barge_confirm(r, s, now)
    assert eng._barge_confirm_step(r, s, _BLOCK, now + 0.1) is True  # user confirmed
    # User speech contaminated the window -> nothing taught to the echo charts.
    assert eng._dtd.echo_obs == []
    assert eng._confirm_echo_obs == []
