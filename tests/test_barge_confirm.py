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
from types import SimpleNamespace

import numpy as np
import pytest

from core.engine import EngineCallbacks
from core.engines.sherpa import (
    SherpaConfig,
    SherpaOnnxEngine,
    _KwsControlAuthority,
    _KwsFeedRoute,
)


class _FakeStream:
    def __init__(self):
        self.fed_blocks = 0
        self.blocks: list[np.ndarray] = []

    def accept_waveform(self, sr, samples):
        self.fed_blocks += 1
        self.blocks.append(np.asarray(samples, dtype="float32").copy())


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


class _BoundedWorkRecognizer(_FakeRecognizer):
    """Recognizer whose one feed needs an exact or unbounded decode count."""

    def __init__(self, *, ready_decodes: int | None, result: str = "") -> None:
        super().__init__([result])
        self.ready_decodes = ready_decodes
        self.ready_calls = 0
        self.decode_calls = 0
        self.result_calls = 0

    def is_ready(self, stream):
        self.ready_calls += 1
        return self.ready_decodes is None or self.decode_calls < self.ready_decodes

    def decode_stream(self, stream):
        self.decode_calls += 1

    def get_result(self, stream):
        self.result_calls += 1
        return super().get_result(stream)


class _FaultingConfirmStream(_FakeStream):
    def __init__(self, failure_phase: str) -> None:
        super().__init__()
        self.failure_phase = failure_phase

    def accept_waveform(self, sr, samples):
        super().accept_waveform(sr, samples)
        if self.failure_phase == "accept":
            raise RuntimeError("injected confirm accept failure")


class _FaultingConfirmRecognizer(_BoundedWorkRecognizer):
    def __init__(self, failure_phase: str) -> None:
        super().__init__(
            ready_decodes=1 if failure_phase == "decode" else 0,
        )
        self.failure_phase = failure_phase

    def is_ready(self, stream):
        if self.failure_phase == "readiness":
            raise RuntimeError("injected confirm readiness failure")
        return super().is_ready(stream)

    def decode_stream(self, stream):
        super().decode_stream(stream)
        if self.failure_phase == "decode":
            raise RuntimeError("injected confirm decode failure")

    def get_result(self, stream):
        result = super().get_result(stream)
        if self.failure_phase == "result" and self.result_calls > 1:
            raise RuntimeError("injected confirm result failure")
        return result


class _Rec:
    def __init__(self):
        self.barges = 0
        self.metrics: list[str] = []

    def callbacks(self) -> EngineCallbacks:
        return EngineCallbacks(on_barge_in=self._on_barge, on_metric=self._on_metric)

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
        {
            "barge_confirm_enabled": True,
            "barge_confirm_min_words": 3,
            "barge_confirm_window_sec": 2.0,
        }
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
    assert eng._barge_confirm_step(r, s, _BLOCK, now + 0.2) is True  # 2 words: fire
    assert rec.barges == 1
    assert "barge_in_confirmed" in rec.metrics
    assert eng._duck_gain == 1.0  # volume restored on the way out
    assert not eng._barge_confirm_active()
    assert eng._barge_in_fired_this_run is True  # latch burned only on a REAL fire


def test_confirm_forwards_current_capture_diagnostic_authority():
    eng = _engine()
    recognizer, stream = _FakeRecognizer(["", "hey wait"]), _FakeStream()
    span = object()
    observed = {}

    def emit_barge(**kwargs):
        observed.update(kwargs)
        return True

    eng._emit_barge_in_callback = emit_barge
    now = time.monotonic()
    eng._begin_barge_confirm(recognizer, stream, now)
    assert eng._barge_confirm_step(
        recognizer,
        stream,
        _BLOCK,
        now + 0.1,
        capture_epoch=19,
        diagnostic_span=span,
    )
    assert observed["detected_at"] == now + 0.1
    assert observed["capture_epoch"] == 19
    assert observed["diagnostic_span"] is span


def test_stop_command_confirms_alone():
    rec = _Rec()
    eng = _engine(rec, barge_confirm_min_words=2)
    r, s = _FakeRecognizer(["", "stop"]), _FakeStream()
    now = time.monotonic()
    eng._begin_barge_confirm(r, s, now)
    assert (
        eng._barge_confirm_step(r, s, _BLOCK, now + 0.1) is True
    )  # 1 word, but a stop command
    assert rec.barges == 1


def test_confirm_decode_source_is_raw_only_when_canceller_masks_near_end():
    processed = np.full(1600, 0.25, dtype="float32")
    raw = np.full(1600, 0.75, dtype="float32")

    for resid_blind, expected in ((False, processed), (True, raw)):
        eng = _engine()
        eng._resid_blind = resid_blind
        recognizer = _FakeRecognizer([""])
        stream = _FakeStream()
        now = time.monotonic()
        eng._begin_barge_confirm(recognizer, stream, now)

        assert not eng._barge_confirm_step(
            recognizer,
            stream,
            processed,
            now + 0.1,
            mic_raw=raw,
        )
        assert len(stream.blocks) == 1
        assert np.array_equal(stream.blocks[0], expected)
        assert len(eng._confirm_primary_pcm) == 1
        assert len(eng._confirm_alternate_pcm) == 1
        assert np.array_equal(eng._confirm_primary_pcm[0], processed)
        assert np.array_equal(eng._confirm_alternate_pcm[0], expected)
        eng._end_barge_confirm()


def test_confirm_window_from_old_playback_generation_cannot_cut_new_run():
    rec = _Rec()
    eng = _engine(rec, barge_confirm_min_words=2)
    r, s = _FakeRecognizer(["", "stop"]), _FakeStream()
    now = time.monotonic()
    eng._speak_gen = 4
    eng._playback_generation = 1
    eng._begin_barge_confirm(
        r,
        s,
        now,
        speak_generation=4,
        playback_generation=1,
    )
    # A new reply can make the current block's own context guard pass. The
    # confirmation window still belongs to generation 1 and must fail closed.
    eng._playback_generation = 2

    assert not eng._barge_confirm_step(
        r,
        s,
        _BLOCK,
        now + 0.1,
        control_effect_guard=lambda: True,
    )
    assert rec.barges == 0
    assert r.resets == 1
    assert not eng._barge_confirm_active()
    assert eng._duck_gain == 1.0
    assert not eng._confirm_handoff_stream_live


def test_generation_change_at_old_confirm_seam_cannot_callback_or_cut():
    eng = _engine(barge_confirm_min_words=2)
    callbacks: list[str] = []
    metrics: list[str] = []
    stop_calls: list[str] = []

    def _stale_barge_callback():
        callbacks.append("barge")
        stop_calls.append("stop")
        eng.stop_speaking()

    eng._cb = EngineCallbacks(
        on_barge_in=_stale_barge_callback,
        on_metric=lambda name, **_kwargs: metrics.append(name),
    )
    eng._speak_gen = 4
    eng._playback_generation = 1
    eng._speaking.set()
    r, s = _FakeRecognizer(["", "stop"]), _FakeStream()
    now = time.monotonic()
    eng._begin_barge_confirm(
        r,
        s,
        now,
        speak_generation=4,
        playback_generation=1,
    )
    guard_calls = 0

    def _generation_changes_after_old_true_check() -> bool:
        nonlocal guard_calls
        guard_calls += 1
        if guard_calls == 2:
            # The old confirmed path checked the window generation immediately
            # before this guard. Changing generation here reproduced the exact
            # compare-then-callback seam: both booleans returned true, but the
            # callback then targeted a newer playback run.
            with eng._gen_lock:
                eng._playback_generation += 1
        return True

    assert not eng._barge_confirm_step(
        r,
        s,
        _BLOCK,
        now + 0.1,
        control_effect_guard=_generation_changes_after_old_true_check,
    )

    assert guard_calls == 2
    assert eng._playback_generation == 2
    assert eng._speak_gen == 4
    assert callbacks == []
    assert stop_calls == []
    assert "barge_in_confirmed" not in metrics
    assert eng._confirmed_barge_claim is None
    assert r.resets == 1


def _current_kws_authority(eng: SherpaOnnxEngine) -> _KwsControlAuthority:
    eng._stream_in = SimpleNamespace(generation=0, actual_device=None)
    eng._capture_authority_source_generation = 0
    eng._capture_authority_source_device = None
    speaking = eng._speaking.is_set()
    if speaking:
        route_owner = SimpleNamespace(reset=lambda: None)
        eng._aec = route_owner
        route = _KwsFeedRoute.IN_APP_AEC
    else:
        route_owner = None
        route = _KwsFeedRoute.IDLE
    return _KwsControlAuthority(
        capture_epoch=eng._capture_epoch,
        capture_generation=0,
        capture_sequence=1,
        source_generation=0,
        source_device=None,
        source_sample_rate_hz=eng.config.sample_rate,
        source_sample_count=1_600,
        source_sample_start=0,
        source_sample_end=1_600,
        source_domain=eng._capture_resolution,
        kws_sample_rate_hz=eng.config.sample_rate,
        authority_source_generation=0,
        authority_source_device=None,
        os_echo_route_verified=eng._os_echo_route_verified,
        route=route,
        route_owner=route_owner,
        speaker_authority_available=False,
        speaker_authority=None,
        speaking=speaking,
        speak_generation=eng._speak_gen,
        playback_generation=eng._playback_generation,
    )


def test_kws_claim_blocks_confirm_claim_for_same_playback_generation() -> None:
    eng = _engine()
    eng._speak_gen = 4
    eng._playback_generation = 7
    eng._speaking.set()
    eng._confirm_speak_generation = 4
    eng._confirm_playback_generation = 7
    kws_claim = eng._claim_playback_kws_if_current(_current_kws_authority(eng))
    assert kws_claim is not None

    assert eng._claim_barge_confirm_if_current(now=10.0) is None

    assert eng._speak_gen == 4
    assert not eng._stop_speaking.is_set()
    assert eng._confirmed_barge_claim is None
    eng._release_playback_kws_claim(kws_claim)


def test_confirm_claim_blocks_kws_claim_even_with_post_claim_current_authority() -> (
    None
):
    eng = _engine()
    eng._speak_gen = 4
    eng._playback_generation = 7
    eng._speaking.set()
    eng._confirm_speak_generation = 4
    eng._confirm_playback_generation = 7
    confirm_claim = eng._claim_barge_confirm_if_current(now=10.0)
    assert confirm_claim is not None
    # Build the KWS authority after confirm advanced the stop generation. It is
    # otherwise current, so rejection specifically proves mutual exclusion.
    authority = _current_kws_authority(eng)

    assert eng._claim_playback_kws_if_current(authority) is None

    assert eng._playback_kws_claim is None
    assert eng._confirmed_barge_claim is confirm_claim
    eng._release_barge_confirm_claim(confirm_claim)


def test_current_generation_claim_allows_reentrant_callback_stop():
    eng = _engine(barge_confirm_min_words=2)
    effects: list[tuple[int, int, int]] = []
    metrics: list[str] = []

    def _current_barge_callback():
        claim = eng._confirmed_barge_claim
        assert claim is not None
        effects.append(
            (
                claim.origin_speak_generation,
                claim.origin_playback_generation,
                claim.claimed_speak_generation,
            )
        )
        # Production's runtime callback synchronously re-enters the engine here.
        # This must not deadlock on _gen_lock or advance the claim generation a
        # second time.
        eng.stop_speaking()

    eng._cb = EngineCallbacks(
        on_barge_in=_current_barge_callback,
        on_metric=lambda name, **_kwargs: metrics.append(name),
    )
    eng._speak_gen = 4
    eng._playback_generation = 7
    eng._speaking.set()
    r, s = _FakeRecognizer(["", "stop"]), _FakeStream()
    now = time.monotonic()
    eng._begin_barge_confirm(
        r,
        s,
        now,
        speak_generation=4,
        playback_generation=7,
    )

    assert eng._barge_confirm_step(r, s, _BLOCK, now + 0.1)

    assert effects == [(4, 7, 5)]
    assert eng._speak_gen == 5
    assert eng._playback_generation == 7
    assert eng._confirmed_barge_claim is None
    assert "barge_in_confirmed" in metrics
    assert eng._confirm_handoff_stream_live
    assert not eng._speaking.is_set()


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
            result.append(eng._barge_confirm_step(r, s, _BLOCK, now + 0.1))
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
    assert r.resets == 1  # echo purged from the stream
    assert eng._barge_in_suppressed_until > now  # can't immediately re-duck
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
    eng._barge_confirm_step(r, s, _BLOCK, now + 0.1)  # banks one echo obs
    eng._barge_confirm_step(r, s, _BLOCK, now + 0.2)  # banks another
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


# --- finite native-work and retained-PCM bounds -------------------------------


def _assert_confirm_window_fully_closed(eng: SherpaOnnxEngine) -> None:
    assert not eng._barge_confirm_active()
    assert eng._duck_gain == 1.0
    assert eng._confirm_base_text == ""
    assert eng._confirm_speak_generation is None
    assert eng._confirm_playback_generation is None
    assert not eng._confirm_handoff_stream_live
    assert eng._confirm_handoff_pending is None
    assert eng._confirm_primary_pcm == []
    assert eng._confirm_alternate_pcm == []
    assert eng._confirm_primary_samples == 0
    assert eng._confirm_alternate_samples == 0
    assert eng._confirm_invocations == 0
    assert eng._confirm_feed_sample_limit == 0
    assert eng._confirm_sample_limit == 0
    assert eng._confirm_invocation_limit == 0
    assert eng._confirm_last_at is None
    assert eng._confirm_audio_started_at is None
    assert eng._confirm_audio_ended_at is None
    assert eng._confirm_echo_obs == []


@pytest.mark.parametrize(
    "now",
    [
        -1.0,
        float("nan"),
        float("inf"),
        -float("inf"),
        True,
        pytest.param(10**10_000, id="giant-int"),
    ],
)
def test_confirm_begin_rejects_invalid_time_before_duck(now):
    rec = _Rec()
    eng = _engine(rec)
    recognizer = _BoundedWorkRecognizer(ready_decodes=0)
    stream = _FakeStream()

    eng._begin_barge_confirm(recognizer, stream, now)

    _assert_confirm_window_fully_closed(eng)
    assert recognizer.result_calls == 0
    assert rec.metrics == []
    assert rec.barges == 0


def test_confirm_begin_accepts_zero_timestamp():
    rec = _Rec()
    eng = _engine(rec)
    recognizer = _BoundedWorkRecognizer(ready_decodes=0)

    eng._begin_barge_confirm(recognizer, _FakeStream(), 0.0)

    assert eng._barge_confirm_active()
    assert eng._confirm_last_at == 0.0
    assert recognizer.result_calls == 1
    assert rec.metrics == ["barge_in_duck"]
    eng._end_barge_confirm()


@pytest.mark.parametrize("window", [-1.0, 0.0, 0.01, 0.1])
def test_confirm_begin_clamps_finite_short_window_to_point_one_second(window):
    eng = _engine(barge_confirm_window_sec=window)
    recognizer = _FakeRecognizer([""])
    stream = _FakeStream()

    eng._begin_barge_confirm(recognizer, stream, 10.0)

    assert eng._confirm_until == pytest.approx(10.1)
    assert eng._confirm_invocation_limit == 2
    eng._end_barge_confirm()


@pytest.mark.parametrize(
    "config_override",
    [
        {"barge_confirm_window_sec": float("nan")},
        {"barge_confirm_window_sec": float("inf")},
        {"block_sec": 0.0},
        {"block_sec": float("nan")},
        {"block_sec": float.fromhex("0x0.0000000000001p-1022")},
        {"barge_confirm_window_sec": True},
        {"block_sec": True},
        {"sample_rate": True},
        {"media_pcm_queue_ms": True},
    ],
)
def test_confirm_begin_rejects_invalid_or_unrepresentable_derived_bounds(
    config_override,
):
    rec = _Rec()
    config = {"media_pcm_queue_ms": 0.0, **config_override}
    eng = _engine(rec, **config)
    recognizer = _BoundedWorkRecognizer(ready_decodes=0)

    eng._begin_barge_confirm(recognizer, _FakeStream(), 10.0)

    _assert_confirm_window_fully_closed(eng)
    assert recognizer.result_calls == 0
    assert rec.metrics == []


@pytest.mark.parametrize(
    "step_at",
    [
        10.0,
        9.9,
        float("nan"),
        float("inf"),
        -float("inf"),
        pytest.param(10**10_000, id="giant-int"),
    ],
)
def test_confirm_step_rejects_invalid_or_nonadvancing_time_before_native_work(
    step_at,
):
    rec = _Rec()
    eng = _engine(rec)
    eng._dtd = _FakeDTD()
    eng._echo_coherence = None
    recognizer = _BoundedWorkRecognizer(ready_decodes=0)
    stream = _FakeStream()
    eng._begin_barge_confirm(recognizer, stream, 10.0)

    assert not eng._barge_confirm_step(recognizer, stream, _BLOCK, step_at)

    _assert_confirm_window_fully_closed(eng)
    assert stream.fed_blocks == 0
    assert recognizer.ready_calls == 0
    assert recognizer.decode_calls == 0
    assert recognizer.result_calls == 1  # begin snapshot only
    assert recognizer.resets == 1
    assert eng._dtd.echo_obs == []
    assert rec.metrics == ["barge_in_duck"]
    assert rec.barges == 0


def test_confirm_step_rejects_bool_after_zero_begin_before_native_work():
    rec = _Rec()
    eng = _engine(rec)
    eng._dtd = _FakeDTD()
    eng._echo_coherence = None
    recognizer = _BoundedWorkRecognizer(ready_decodes=0)
    stream = _FakeStream()
    eng._begin_barge_confirm(recognizer, stream, 0.0)

    assert not eng._barge_confirm_step(recognizer, stream, _BLOCK, True)

    _assert_confirm_window_fully_closed(eng)
    assert stream.fed_blocks == 0
    assert recognizer.ready_calls == 0
    assert recognizer.decode_calls == 0
    assert recognizer.result_calls == 1
    assert recognizer.resets == 1
    assert eng._dtd.echo_obs == []
    assert rec.metrics == ["barge_in_duck"]
    assert rec.barges == 0


@pytest.mark.parametrize(
    "queue_ms, primary_samples, alternate_samples",
    [
        (300.0, 31, 1),
        (300.0, 1, 31),
        (300.0, 0, 1),
        (300.0, 1, 0),
        (0.0, 11, 1),
    ],
)
def test_confirm_rejects_empty_or_per_feed_oversize_domain_before_side_effects(
    queue_ms,
    primary_samples,
    alternate_samples,
):
    rec = _Rec()
    eng = _engine(
        rec,
        sample_rate=100,
        block_sec=0.1,
        media_pcm_queue_ms=queue_ms,
        barge_confirm_window_sec=0.5,
    )
    eng._resid_blind = True
    eng._dtd = _FakeDTD()
    eng._echo_coherence = None
    recognizer = _BoundedWorkRecognizer(ready_decodes=0)
    stream = _FakeStream()
    eng._begin_barge_confirm(recognizer, stream, 10.0)

    assert not eng._barge_confirm_step(
        recognizer,
        stream,
        np.zeros(primary_samples, dtype="float32"),
        10.1,
        mic_raw=np.zeros(alternate_samples, dtype="float32"),
    )

    _assert_confirm_window_fully_closed(eng)
    assert stream.fed_blocks == 0
    assert recognizer.ready_calls == 0
    assert recognizer.decode_calls == 0
    assert recognizer.result_calls == 1
    assert recognizer.resets == 1
    assert eng._dtd.echo_obs == []
    assert rec.metrics == ["barge_in_duck"]
    assert rec.barges == 0


@pytest.mark.parametrize(
    "primary_samples, alternate_samples",
    [(30, 1), (1, 30)],
)
def test_confirm_accepts_each_domain_at_the_exact_per_feed_cap(
    primary_samples,
    alternate_samples,
):
    eng = _engine(
        sample_rate=100,
        block_sec=0.1,
        media_pcm_queue_ms=300.0,
        barge_confirm_window_sec=0.5,
    )
    eng._resid_blind = True
    recognizer = _BoundedWorkRecognizer(ready_decodes=0)
    stream = _FakeStream()
    eng._begin_barge_confirm(recognizer, stream, 10.0)

    assert not eng._barge_confirm_step(
        recognizer,
        stream,
        np.zeros(primary_samples, dtype="float32"),
        10.1,
        mic_raw=np.zeros(alternate_samples, dtype="float32"),
    )

    assert eng._barge_confirm_active()
    assert eng._confirm_feed_sample_limit == 30
    assert eng._confirm_primary_samples == primary_samples
    assert eng._confirm_alternate_samples == alternate_samples
    assert stream.fed_blocks == 1
    eng._end_barge_confirm()


@pytest.mark.parametrize("overflow_domain", ["primary", "alternate"])
def test_confirm_prospective_cumulative_overflow_aborts_before_third_feed(
    overflow_domain,
):
    rec = _Rec()
    eng = _engine(
        rec,
        sample_rate=100,
        block_sec=0.1,
        media_pcm_queue_ms=300.0,
        barge_confirm_window_sec=0.5,
    )
    eng._resid_blind = True
    recognizer = _BoundedWorkRecognizer(ready_decodes=0)
    stream = _FakeStream()
    eng._begin_barge_confirm(recognizer, stream, 10.0)
    primary_n, alternate_n = (30, 1) if overflow_domain == "primary" else (1, 30)

    for step_at in (10.1, 10.2):
        assert not eng._barge_confirm_step(
            recognizer,
            stream,
            np.zeros(primary_n, dtype="float32"),
            step_at,
            mic_raw=np.zeros(alternate_n, dtype="float32"),
        )
    retained = (
        eng._confirm_primary_samples
        if overflow_domain == "primary"
        else eng._confirm_alternate_samples
    )
    assert retained == 60
    assert retained <= 80  # ceil(100 * .5s) + one 30-sample feed

    assert not eng._barge_confirm_step(
        recognizer,
        stream,
        np.zeros(primary_n, dtype="float32"),
        10.3,
        mic_raw=np.zeros(alternate_n, dtype="float32"),
    )

    _assert_confirm_window_fully_closed(eng)
    assert stream.fed_blocks == 2
    assert recognizer.ready_calls == 2
    assert recognizer.result_calls == 3  # begin plus two accepted feeds
    assert rec.metrics == ["barge_in_duck"]
    assert rec.barges == 0


def test_confirm_zero_window_has_two_invocation_minimum_and_no_third_feed():
    rec = _Rec()
    eng = _engine(
        rec,
        sample_rate=100,
        block_sec=1.0,
        media_pcm_queue_ms=0.0,
        barge_confirm_window_sec=0.0,
    )
    recognizer = _BoundedWorkRecognizer(ready_decodes=0)
    stream = _FakeStream()
    eng._begin_barge_confirm(recognizer, stream, 10.0)
    assert eng._confirm_invocation_limit == 2

    assert not eng._barge_confirm_step(recognizer, stream, np.zeros(1), 10.01)
    assert not eng._barge_confirm_step(recognizer, stream, np.zeros(1), 10.02)
    assert eng._barge_confirm_active()
    assert not eng._barge_confirm_step(recognizer, stream, np.zeros(1), 10.03)

    _assert_confirm_window_fully_closed(eng)
    assert stream.fed_blocks == 2
    assert recognizer.result_calls == 3
    assert rec.metrics == ["barge_in_duck"]
    assert rec.barges == 0


def test_confirm_exactly_sixty_four_decodes_runs_final_probe_and_result_path():
    eng = _engine()
    eng._dtd = _FakeDTD()
    eng._echo_coherence = None
    recognizer = _BoundedWorkRecognizer(ready_decodes=64)
    stream = _FakeStream()
    eng._begin_barge_confirm(recognizer, stream, 10.0)

    assert not eng._barge_confirm_step(recognizer, stream, _BLOCK, 10.1)

    assert recognizer.ready_calls == 65
    assert recognizer.decode_calls == 64
    assert recognizer.result_calls == 2  # begin snapshot plus accepted feed
    assert stream.fed_blocks == 1
    assert len(eng._confirm_primary_pcm) == 1
    assert len(eng._confirm_alternate_pcm) == 1
    assert len(eng._confirm_echo_obs) == 1
    assert eng._dtd.echo_obs == []
    eng._end_barge_confirm()


def test_confirm_decode_exhaustion_never_reads_result_or_labels_window():
    rec = _Rec()
    eng = _engine(rec)
    eng._dtd = _FakeDTD()
    eng._echo_coherence = None
    recognizer = _BoundedWorkRecognizer(ready_decodes=None, result="stop")
    stream = _FakeStream()
    eng._begin_barge_confirm(recognizer, stream, 10.0)
    eng._confirm_handoff_stream_live = True
    eng._confirm_handoff_pending = object()

    assert not eng._barge_confirm_step(recognizer, stream, _BLOCK, 10.1)

    _assert_confirm_window_fully_closed(eng)
    assert recognizer.ready_calls == 65
    assert recognizer.decode_calls == 64
    assert recognizer.result_calls == 1  # begin only; no post-exhaustion read
    assert recognizer.resets == 1
    assert eng._dtd.echo_obs == []
    assert rec.metrics == ["barge_in_duck"]
    assert rec.barges == 0


@pytest.mark.parametrize("failure_phase", ["accept", "readiness", "decode", "result"])
def test_legacy_confirm_native_fault_closes_resets_and_abstains(failure_phase):
    rec = _Rec()
    eng = _engine(rec)
    eng._dtd = _FakeDTD()
    eng._echo_coherence = None
    recognizer = _FaultingConfirmRecognizer(failure_phase)
    stream = _FaultingConfirmStream(failure_phase)
    eng._begin_barge_confirm(recognizer, stream, 10.0)
    eng._confirm_handoff_stream_live = True
    eng._confirm_handoff_pending = object()

    assert not eng._barge_confirm_step(recognizer, stream, _BLOCK, 10.1)

    _assert_confirm_window_fully_closed(eng)
    assert recognizer.resets == 1
    assert eng._dtd.echo_obs == []
    assert rec.metrics == ["barge_in_duck"]
    assert rec.barges == 0


def test_confirm_callback_exception_is_not_reclassified_as_native_failure():
    class _CallbackFailure(RuntimeError):
        pass

    eng = _engine()

    def fail_callback():
        raise _CallbackFailure("barge callback failed")

    eng._cb = EngineCallbacks(on_barge_in=fail_callback)
    recognizer = _FakeRecognizer(["", "stop"])
    stream = _FakeStream()
    eng._begin_barge_confirm(recognizer, stream, 10.0)

    with pytest.raises(_CallbackFailure, match="barge callback failed"):
        eng._barge_confirm_step(recognizer, stream, _BLOCK, 10.1)

    assert recognizer.resets == 0
    assert eng._confirmed_barge_claim is None


def test_engine_stop_clears_active_confirm_window_and_all_retained_state():
    eng = _engine(_Rec())
    recognizer = _FakeRecognizer([""])
    stream = _FakeStream()
    eng._begin_barge_confirm(recognizer, stream, 10.0)
    assert not eng._barge_confirm_step(recognizer, stream, _BLOCK, 10.1)
    eng._confirm_handoff_stream_live = True
    eng._confirm_handoff_pending = object()
    assert eng._barge_confirm_active()
    assert eng._confirm_primary_pcm

    eng.stop()

    _assert_confirm_window_fully_closed(eng)
