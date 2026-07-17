from __future__ import annotations

import threading
import types

import numpy as np
import pytest

from core.engine import EngineCallbacks
from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine


class _Binder:
    def __init__(self):
        self.capture_bound = False
        self.playback_bound = False
        self.contract = types.SimpleNamespace(provenance="autotest-virtual-delay:test")
        self.fail_duplex = False

    @property
    def fully_bound(self):
        return self.capture_bound and self.playback_bound

    def bind_capture(self, *, timeout_sec):
        assert timeout_sec > 0
        self.capture_bound = True

    def bind_playback(self, *, timeout_sec):
        assert timeout_sec > 0
        self.playback_bound = True

    def verify(self, *, require_playback):
        if require_playback and (self.fail_duplex or not self.fully_bound):
            return False, "playback wrong"
        return self.capture_bound, "exact virtual route"


def _engine(binder):
    return SherpaOnnxEngine(
        SherpaConfig(
            barge_in_enabled=True,
            barge_word_cut_enabled=True,
            aec_enabled=False,
        ),
        virtual_audio_binder=binder,
    )


def test_virtual_word_cut_authority_is_two_phase():
    binder = _Binder()
    engine = _engine(binder)

    engine._bind_virtual_capture_route()
    assert binder.capture_bound is True
    assert engine._word_cut_route_verified is False
    assert engine._os_echo_route_verified is False

    engine._prepare_virtual_playback_route()
    assert engine._word_cut_route_verified is False
    assert engine._os_echo_route_verified is False
    # Capture may have consumed the reply reset while playback was still
    # unbound. Successful authorization must re-arm this same reply.
    engine._barge_sustain_reset_pending = False
    engine._authorize_virtual_playback_route()
    assert binder.fully_bound is True
    assert engine._word_cut_route_verified is True
    assert engine._os_echo_route_verified is True
    assert engine._barge_sustain_reset_pending is True


def test_loss_started_before_duplex_grant_cannot_reauthorize_route():
    binder = _Binder()
    binder.capture_bound = True
    binder.playback_bound = True
    engine = _engine(binder)
    engine._virtual_route_failure_in_progress = True

    with pytest.raises(RuntimeError, match="revocation already started"):
        engine._authorize_virtual_playback_route()

    assert engine._word_cut_route_verified is False
    assert engine._os_echo_route_verified is False
    assert engine._barge_sustain_reset_pending is False


def test_rearmed_duplex_grant_initializes_word_cut_in_same_reply():
    class _Stream:
        def __init__(self):
            self.blocks = []

        def accept_waveform(self, _sample_rate, samples):
            self.blocks.append(np.asarray(samples).copy())

    class _Recognizer:
        def __init__(self):
            self.streams = []

        def create_stream(self, **_kwargs):
            stream = _Stream()
            self.streams.append(stream)
            return stream

        def is_ready(self, _stream):
            return False

        def get_result(self, _stream):
            return ""

        def reset(self, _stream):
            pass

        def is_endpoint(self, _stream):
            return False

    class _Vad:
        def accept_waveform(self, _samples):
            pass

        def is_speech_detected(self):
            return True

    class _OneBlockInput:
        generation = 0

        def __init__(self, engine):
            self.engine = engine

        def read(self, frames):
            self.engine._running.clear()
            return np.zeros(frames, dtype="float32"), False

    binder = _Binder()
    engine = _engine(binder)
    engine._recognizer = _Recognizer()
    engine._vad = _Vad()
    engine._capture_sr = engine.config.sample_rate
    engine._stream_in = _OneBlockInput(engine)
    engine._speaking.set()
    # Model the race: the reply reset was already consumed with capture-only
    # authority, so no detector stream exists yet.
    engine._barge_sustain_reset_pending = False
    engine._bind_virtual_capture_route()
    engine._prepare_virtual_playback_route()
    engine._authorize_virtual_playback_route()

    engine._running.set()
    engine._capture_loop()

    assert len(engine._recognizer.streams) == 2
    assert len(engine._recognizer.streams[0].blocks) == 0
    assert len(engine._recognizer.streams[1].blocks) == 1


def test_private_kws_rechecks_route_at_callback_seam():
    class _KwsStream:
        def accept_waveform(self, _sample_rate, _samples):
            pass

    class _Kws:
        def __init__(self, engine):
            self.engine = engine
            self.resets = 0

        def is_ready(self, _stream):
            return False

        def get_result(self, _stream):
            # Revocation begins after the outer eligibility read and decode but
            # before the callback seam protected by the route lock.
            self.engine._virtual_route_failure_in_progress = True
            self.engine._speaking.clear()
            return "stop"

        def reset_stream(self, _stream):
            self.resets += 1

    binder = _Binder()
    engine = _engine(binder)
    commands = []
    engine._kws_stream = _KwsStream()
    engine._kws = _Kws(engine)
    engine._cb = EngineCallbacks(on_command=commands.append)
    engine._os_echo_route_verified = True

    engine._poll_keywords(
        np.zeros(1600, dtype="float32"),
        guard_private_route=True,
        require_os_echo_route=True,
    )

    assert engine._kws.resets == 1
    assert commands == []


class _FixedKws:
    """Minimal spotter that reports one fixed keyword hit per poll."""

    def __init__(self, keyword):
        self.keyword = keyword
        self.resets = 0

    def is_ready(self, _stream):
        return False

    def get_result(self, _stream):
        return self.keyword

    def reset_stream(self, _stream):
        self.resets += 1


class _NullKwsStream:
    def accept_waveform(self, _sample_rate, _samples):
        pass


def _kws_engine(keyword):
    engine = SherpaOnnxEngine(SherpaConfig())
    engine._kws = _FixedKws(keyword)
    engine._kws_stream = _NullKwsStream()
    commands: list[str] = []
    engine._cb = EngineCallbacks(on_command=commands.append)
    return engine, commands


def test_kws_own_echo_suppressed_when_now_playing_contains_keyword():
    # The assistant's own TTS ("I'll stop now") leaks back through imperfect AEC
    # and the spotter fires on it -- the own-echo guard must drop the hit so it
    # cannot self-interrupt.
    engine, commands = _kws_engine("stop")
    engine._speaking.set()
    engine._now_playing = "Okay, I will stop now."

    engine._poll_keywords(np.zeros(1600, dtype="float32"))

    assert commands == []
    assert engine._kws_own_echo_suppressions == 1
    assert engine._kws.resets == 1  # stream is still reset after the hit


def test_kws_hit_passes_when_not_speaking_even_if_now_playing_matches():
    # Not speaking -> the guard is inert; a genuine user "stop" must reach the
    # command callback even if the last-played line contained the word.
    engine, commands = _kws_engine("stop")
    engine._now_playing = "Okay, I will stop now."

    engine._poll_keywords(np.zeros(1600, dtype="float32"))

    assert commands == ["stop"]
    assert engine._kws_own_echo_suppressions == 0


def test_kws_hit_passes_when_now_playing_does_not_contain_keyword():
    # Speaking, but the playing sentence does not contain the spotted word ->
    # this is the user talking over the assistant, not its own echo. Pass it.
    engine, commands = _kws_engine("stop")
    engine._speaking.set()
    engine._now_playing = "The weather is nice today."

    engine._poll_keywords(np.zeros(1600, dtype="float32"))

    assert commands == ["stop"]
    assert engine._kws_own_echo_suppressions == 0


def test_failed_virtual_playback_proof_never_grants_authority():
    binder = _Binder()
    binder.capture_bound = True
    binder.fail_duplex = True
    engine = _engine(binder)
    engine._word_cut_route_verified = True
    engine._os_echo_route_verified = True

    with pytest.raises(RuntimeError, match="duplex route"):
        engine._prepare_virtual_playback_route()

    assert engine._word_cut_route_verified is False
    assert engine._os_echo_route_verified is False


def test_production_engine_has_no_virtual_route_authority():
    engine = SherpaOnnxEngine(SherpaConfig())

    engine._bind_virtual_capture_route()
    engine._prepare_virtual_playback_route()
    engine._authorize_virtual_playback_route()

    assert engine._virtual_audio_binder is None
    assert engine._word_cut_route_verified is False


def test_route_loss_hard_flushes_and_closes_before_publishing_failure():
    engine = _engine(_Binder())
    engine._running.set()
    engine._word_cut_route_verified = True
    engine._os_echo_route_verified = True

    class _FIFO:
        def __init__(self):
            self.calls = []

        def interrupt_tags(self, status, fade_samples=0):
            self.calls.append(("interrupt", status, fade_samples))
            return ()

        def flush(self, fade_samples=0):
            self.calls.append(("flush", fade_samples))

    class _Output:
        def __init__(self):
            self.calls = []

        def abort(self):
            assert engine.virtual_route_failure == ""
            self.calls.append("abort")

        def close(self):
            assert engine.virtual_route_failure == ""
            self.calls.append("close")

    fifo = _FIFO()
    output = _Output()
    engine._fifo = fifo
    engine._out_stream = output

    engine._fail_virtual_route("target drift")

    assert [call[0] for call in fifo.calls] == ["interrupt", "flush"]
    assert all(call[-1] == 0 for call in fifo.calls)
    assert output.calls == ["abort", "close"]
    assert engine._fifo is None and engine._out_stream is None
    assert engine._running.is_set() is False
    assert engine._word_cut_route_verified is False
    assert engine._os_echo_route_verified is False
    assert engine.virtual_route_failure == "target drift"


def test_concurrent_route_failure_cannot_publish_before_blocked_abort_finishes():
    engine = _engine(_Binder())
    engine._running.set()
    abort_entered = threading.Event()
    release_abort = threading.Event()

    class _Output:
        def __init__(self):
            self.abort_calls = 0

        def abort(self):
            self.abort_calls += 1
            abort_entered.set()
            assert release_abort.wait(2.0)

        def close(self):
            pass

    output = _Output()
    engine._out_stream = output
    first = threading.Thread(
        target=engine._fail_virtual_route,
        args=("first failure",),
    )
    first.start()
    assert abort_entered.wait(1.0)

    engine._fail_virtual_route("second failure")
    assert engine.virtual_route_failure == ""
    assert output.abort_calls == 1

    release_abort.set()
    first.join(timeout=1.0)
    assert first.is_alive() is False
    assert engine.virtual_route_failure == "first failure"
    assert output.abort_calls == 1


def test_unexpected_private_capture_exit_publishes_fatal_but_stop_does_not():
    engine = _engine(_Binder())
    engine._running.set()

    assert engine._fail_virtual_capture_if_unexpected("capture failed") is True
    assert engine.virtual_route_failure == "capture failed"
    assert engine._running.is_set() is False

    stopping = _engine(_Binder())
    stopping._running.set()
    stopping._capture_stopping.set()
    assert stopping._fail_virtual_capture_if_unexpected("normal stop") is False
    assert stopping.virtual_route_failure == ""


def test_route_loss_fences_an_inflight_capture_block_before_callbacks():
    read_entered = threading.Event()
    release_read = threading.Event()
    commands = []

    class _Stream:
        def accept_waveform(self, _sample_rate, _samples):
            pass

    class _Recognizer:
        def create_stream(self, **_kwargs):
            return _Stream()

        def is_ready(self, _stream):
            return False

        def get_result(self, _stream):
            return ""

        def reset(self, _stream):
            pass

        def is_endpoint(self, _stream):
            return False

    class _BlockingInput:
        generation = 0

        def read(self, frames):
            read_entered.set()
            assert release_read.wait(2.0)
            return np.ones(frames, dtype="float32"), False

    engine = _engine(_Binder())
    engine._recognizer = _Recognizer()
    engine._capture_sr = engine.config.sample_rate
    engine._stream_in = _BlockingInput()
    engine._poll_keywords = lambda _samples, **_kwargs: commands.append("stop")
    engine._running.set()
    worker = threading.Thread(target=engine._capture_loop)
    worker.start()
    assert read_entered.wait(1.0)

    engine._fail_virtual_route("proof lost during read")
    release_read.set()
    worker.join(timeout=1.0)

    assert worker.is_alive() is False
    assert commands == []
    assert engine._capture_stopping.is_set()
    assert engine.virtual_route_failure == "proof lost during read"


def test_private_capture_without_recognizer_fails_parent_instead_of_hanging():
    engine = _engine(_Binder())
    engine._recognizer = None
    engine._running.set()

    engine._capture_loop()

    assert engine._running.is_set() is False
    assert engine.virtual_route_failure == "virtual capture loop stopped unexpectedly"


def test_restart_refuses_retained_virtual_route_monitor_before_audio_open():
    engine = _engine(_Binder())
    engine._virtual_route_thread = types.SimpleNamespace(is_alive=lambda: True)

    with pytest.raises(RuntimeError, match="virtual-route monitor"):
        engine.start(EngineCallbacks())
