"""Desktop output failure isolation with fake synthesis and manual playback."""
import threading

import pytest

from core.engine import EngineCallbacks, OutputState, PlaybackOutcome, TrackedSpeech
from core.engines import sherpa as sherpa_module
from tests import test_sherpa_playback as playback


class FailingTts(playback._StreamingTts):
    def __init__(self):
        super().__init__()
        self.submitted = threading.Event()
        self.release = threading.Event()

    def generate(self, text, sid=0, speed=1.0, callback=None):
        self.calls += 1
        assert callback is not None
        callback(playback.np.array([0.1, 0.2], dtype="float32"), 1.0)
        self.submitted.set()
        assert self.release.wait(timeout=3)
        raise TypeError("public fake synthesis failure")


def _wait_worker_exit(engine):
    worker = engine._play_thread
    worker.join(timeout=2)
    assert not worker.is_alive()


def test_output_failure_keeps_capture_event_and_controls_and_fails_every_receipt(monkeypatch):
    tts = FailingTts()
    engine = playback._engine(tts)
    states = []
    engine._cb = EngineCallbacks(on_output_state=states.append)
    holder = playback._start_playback_harness(monkeypatch, engine, default_sr=22050)
    probe = playback._ReceiptProbe()
    legacy_done = threading.Event()
    epoch = engine._capture_epoch
    engine.speak_tracked(TrackedSpeech("active", "public active"), on_terminal=probe.on_terminal)
    try:
        assert tts.submitted.wait(timeout=1)
        engine.speak_tracked(TrackedSpeech("queued", "public queued"), on_terminal=probe.on_terminal)
        engine.speak("public legacy queued", on_done=legacy_done.set)
        tts.release.set()
        assert playback._wait_until(lambda: engine.output_state is OutputState.RECOVERABLE)
        _wait_worker_exit(engine)
        engine.speak_tracked(TrackedSpeech("new", "public new"), on_terminal=probe.on_terminal)
        assert playback._wait_until(lambda: len(probe.snapshot()[1]) == 3)
        receipts = [item[0] for item in probe.snapshot()[1]]
        assert {r.fragment_id for r in receipts} == {"active", "queued", "new"}
        assert all(r.outcome is PlaybackOutcome.FAILED and r.safe_text_prefix == "" for r in receipts)
        assert legacy_done.wait(timeout=1)
        assert engine._running.is_set()
        assert engine._capture_epoch == epoch
        assert not engine.is_speaking
        assert states == ["unavailable", "recoverable"]
        assert engine._play_q.empty()
        assert tts.calls == 1
        original = holder["stream"]
        assert original.stop_calls == original.close_calls == 1
        prior_generation = engine._speak_gen
        engine.stop_speaking()
        assert engine._speak_gen == prior_generation + 1
        assert engine._running.is_set()

        engine._tts = playback._StreamingTts()
        engine._tts_lock.acquire()
        try:
            assert not engine.recover_output()  # no wait for model ownership
        finally:
            engine._tts_lock.release()
        assert engine.recover_output()
        assert engine.output_state is OutputState.READY
        engine.speak_tracked(TrackedSpeech("fresh", "public fresh"), on_terminal=probe.on_terminal)
        assert playback._wait_until(lambda: len(holder["streams"]) == 2 and playback._fifo_count_is(engine, 3))
        holder["stream"].pull(3)
        assert playback._wait_until(lambda: len(probe.snapshot()[1]) == 4)
        assert probe.snapshot()[1][-1][0].outcome is PlaybackOutcome.COMPLETED
        assert original.stop_calls == original.close_calls == 1
    finally:
        tts.release.set()
        engine.stop()


def test_uncertain_close_poison_is_output_only_and_retains_acoustic_quarantine(monkeypatch):
    close_entered = threading.Event()
    release_close = threading.Event()
    original_stream = playback._ManualOutputStream
    class BlockedClose(original_stream):
        def close(self):
            self.close_calls += 1
            close_entered.set()
            assert release_close.wait(timeout=4)
            self.closed = True
            self.active = False
    monkeypatch.setattr(playback, "_ManualOutputStream", BlockedClose)
    tts = FailingTts()
    engine = playback._engine(tts)
    holder = playback._start_playback_harness(monkeypatch, engine, default_sr=22050)
    probe = playback._ReceiptProbe()
    engine.speak_tracked(TrackedSpeech("fault", "public fault"), on_terminal=probe.on_terminal)
    try:
        assert tts.submitted.wait(timeout=1)
        tts.release.set()
        assert close_entered.wait(timeout=1)
        assert playback._wait_until(lambda: engine.output_state is OutputState.POISONED, timeout=2)
        _wait_worker_exit(engine)
        assert engine._running.is_set()
        assert engine._output_quarantine and engine.is_speaking
        generation = engine._speak_gen
        engine.stop_speaking()
        assert engine._speak_gen == generation + 1
        assert engine.is_speaking  # STOP cannot pretend native silence was proven
        engine.speak_tracked(TrackedSpeech("blocked", "public blocked"), on_terminal=probe.on_terminal)
        assert playback._wait_until(lambda: len(probe.snapshot()[1]) == 2)
        assert all(r[0].outcome is PlaybackOutcome.FAILED for r in probe.snapshot()[1])
        assert not engine.recover_output()
        assert engine._output_cleanup.stream is holder["stream"]
        engine.stop()
        assert holder["stream"].stop_calls == holder["stream"].close_calls == 1
        release_close.set()
        engine._output_cleanup._thread.join(timeout=1)
        assert not engine._output_cleanup.snapshot().closed
        assert not engine.recover_output()
        with pytest.raises(RuntimeError, match="output cleanup remains uncertain"):
            engine.start(EngineCallbacks())
    finally:
        tts.release.set()
        release_close.set()
        engine.stop()


def test_constructor_failure_poison_does_not_kill_capture_or_wait_for_audio(monkeypatch):
    engine = playback._engine(playback._StreamingTts())
    holder = playback._start_playback_harness(monkeypatch, engine)
    def fail_constructor(*_args, **_kwargs):
        raise RuntimeError("public constructor failure")
    monkeypatch.setattr(playback.sys.modules["sounddevice"], "OutputStream", fail_constructor)
    probe = playback._ReceiptProbe()
    engine.speak_tracked(TrackedSpeech("open", "public open"), on_terminal=probe.on_terminal)
    try:
        assert playback._wait_until(lambda: engine.output_state is OutputState.POISONED)
        _wait_worker_exit(engine)
        assert engine._running.is_set()
        assert not engine.is_speaking  # no native PCM was ever admitted
        assert playback._wait_until(lambda: len(probe.snapshot()[1]) == 1)
        assert probe.snapshot()[1][0][0].outcome is PlaybackOutcome.FAILED
        assert not engine.recover_output()
        assert holder["streams"] == []
    finally:
        engine.stop()


def test_failed_output_state_observer_cannot_stop_capture(monkeypatch):
    engine = playback._engine(FailingTts())
    def observer(_state):
        raise RuntimeError("public observer failure")
    engine._cb = EngineCallbacks(on_output_state=observer)
    playback._start_playback_harness(monkeypatch, engine)
    engine.speak("public phrase")
    try:
        assert engine._tts.submitted.wait(timeout=1)
        engine._tts.release.set()
        assert playback._wait_until(lambda: engine.output_state is OutputState.RECOVERABLE)
        assert engine._running.is_set()
    finally:
        engine._tts.release.set()
        engine.stop()


def test_ambiguous_recovery_worker_launch_poisons_only_output(monkeypatch):
    tts = FailingTts()
    engine = playback._engine(tts)
    states = []
    engine._cb = EngineCallbacks(on_output_state=states.append)
    holder = playback._start_playback_harness(monkeypatch, engine)
    engine.speak("public phrase")
    assert tts.submitted.wait(timeout=1)
    tts.release.set()
    assert playback._wait_until(lambda: engine.output_state is OutputState.RECOVERABLE)
    _wait_worker_exit(engine)
    real_thread = threading.Thread
    def ambiguous_thread(**kwargs):
        worker = real_thread(**kwargs)
        real_start = worker.start
        def start_then_throw():
            real_start()
            raise RuntimeError("public launch failure")
        worker.start = start_then_throw
        return worker
    monkeypatch.setattr(sherpa_module.threading, "Thread", ambiguous_thread)
    try:
        assert not engine.recover_output()
        assert engine.output_state is OutputState.POISONED
        assert states[-1] == "poisoned"
        assert engine._running.is_set()
        assert not engine.recover_output()
        assert len(holder["streams"]) == 1
    finally:
        engine.stop()


def test_stop_and_failure_join_one_exact_native_cleanup_owner(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    original_stream = playback._ManualOutputStream
    class PausedClose(original_stream):
        def close(self):
            self.close_calls += 1
            entered.set()
            assert release.wait(timeout=3)
            self.closed = True
            self.active = False
    monkeypatch.setattr(playback, "_ManualOutputStream", PausedClose)
    tts = FailingTts()
    engine = playback._engine(tts)
    holder = playback._start_playback_harness(monkeypatch, engine)
    probe = playback._ReceiptProbe()
    engine.speak_tracked(TrackedSpeech("race", "public race"), on_terminal=probe.on_terminal)
    stopper = None
    try:
        assert tts.submitted.wait(timeout=1)
        tts.release.set()
        assert entered.wait(timeout=1)
        stopper = threading.Thread(target=engine.stop)
        stopper.start()
        release.set()
        stopper.join(timeout=2)
        assert not stopper.is_alive()
        _wait_worker_exit(engine)
        assert holder["stream"].stop_calls == holder["stream"].close_calls == 1
        assert len(probe.snapshot()[1]) == 1
        assert probe.snapshot()[1][0][0].outcome is PlaybackOutcome.FAILED
    finally:
        release.set()
        tts.release.set()
        if stopper is not None:
            stopper.join(timeout=2)
        engine.stop()


def _production_capture_engine(monkeypatch, tts, aec=None):
    from core import enroll
    from tests import test_sherpa_duplex_runtime as duplex

    accepted = []
    class CountingRecognizer(duplex._EnergyStopRecognizer):
        def create_stream(self, **_kwargs):
            stream = super().create_stream()
            original_accept = stream.accept_waveform
            def count_accept(sample_rate, samples):
                accepted.append(len(samples))
                original_accept(sample_rate, samples)
            stream.accept_waveform = count_accept
            return stream
    recognizer = CountingRecognizer()
    for name in (
        "build_final_recognizer", "build_final_verifier", "build_denoiser",
        "build_keyword_spotter", "build_punctuation",
    ):
        monkeypatch.setattr(sherpa_module, name, lambda _config: None)
    monkeypatch.setattr(sherpa_module, "build_aec", lambda *_args, **_kwargs: aec)
    monkeypatch.setattr(sherpa_module, "build_recognizer", lambda _config: recognizer)
    monkeypatch.setattr(sherpa_module, "build_vad", lambda _config: duplex._EnergyVad())
    monkeypatch.setattr(sherpa_module, "build_tts", lambda _config: tts)
    monkeypatch.setattr(enroll, "verify_required_os_echo_route", lambda _config: "headless-verified-route")
    holder = {}
    duplex._fake_sounddevice(monkeypatch, holder, threading.Event())
    config = sherpa_module.SherpaConfig(
        sample_rate=16000, block_sec=0.1,
        input_device="headless-echo-source", output_device="headless-output",
        aec_enabled=aec is not None, coherence_barge_in_enabled=False, dtd_enabled=False,
        input_calibrate=False, tts_dc_block=False,
    )
    engine = sherpa_module.SherpaOnnxEngine(config)
    return engine, accepted


def test_production_capture_worker_keeps_feeding_asr_after_clean_output_failure(monkeypatch):
    from tests import test_sherpa_duplex_runtime as duplex
    tts = FailingTts()
    engine, accepted = _production_capture_engine(monkeypatch, tts)
    try:
        engine.start(EngineCallbacks())
        assert duplex._wait_until(lambda: len(accepted) > 0)
        engine.speak("public output failure")
        assert tts.submitted.wait(timeout=1)
        tts.release.set()
        assert duplex._wait_until(lambda: engine.output_state is OutputState.RECOVERABLE)
        count = len(accepted)
        assert duplex._wait_until(lambda: len(accepted) > count + 1)
        assert engine._capture_thread.is_alive()
        assert engine._running.is_set()
        assert not engine.is_speaking
    finally:
        tts.release.set()
        engine.stop()


def test_recovery_refuses_busy_admission_lock_without_waiting():
    import time
    engine = playback._engine(playback._StreamingTts())
    engine._output_state = OutputState.RECOVERABLE
    engine._running.set()
    held = threading.Event()
    release = threading.Event()
    def hold_lock():
        with engine._receipt_lock:
            held.set()
            assert release.wait(timeout=2)
    holder = threading.Thread(target=hold_lock)
    holder.start()
    try:
        assert held.wait(timeout=1)
        started = time.monotonic()
        assert not engine.recover_output()
        assert time.monotonic() - started < 0.2
        assert engine._play_thread is None
        assert engine.output_state is OutputState.RECOVERABLE
    finally:
        release.set()
        holder.join(timeout=1)
        engine.stop()


@pytest.mark.parametrize("error_type", [SystemExit, KeyboardInterrupt])
def test_base_exception_worker_exit_cannot_leave_ready_or_orphan_new_receipts(monkeypatch, error_type):
    class FatalTts(playback._StreamingTts):
        def generate(self, text, sid=0, speed=1.0, callback=None):
            self.calls += 1
            callback(playback.np.array([0.1, 0.2], dtype="float32"), 1.0)
            raise error_type()
    tts = FatalTts()
    engine = playback._engine(tts)
    playback._start_playback_harness(monkeypatch, engine)
    probe = playback._ReceiptProbe()
    engine.speak_tracked(TrackedSpeech("fatal", "public fatal"), on_terminal=probe.on_terminal)
    try:
        assert playback._wait_until(lambda: engine.output_state is OutputState.RECOVERABLE)
        _wait_worker_exit(engine)
        engine.speak_tracked(TrackedSpeech("new", "public new"), on_terminal=probe.on_terminal)
        assert playback._wait_until(lambda: len(probe.snapshot()[1]) == 2)
        assert all(item[0].outcome is PlaybackOutcome.FAILED for item in probe.snapshot()[1])
        assert engine._running.is_set()
        assert engine._play_q.empty()
        assert tts.calls == 1
    finally:
        engine.stop()


def test_sounddevice_import_failure_is_output_only_and_fails_new_request(monkeypatch):
    import builtins
    original_import = builtins.__import__
    def fail_output_import(name, *args, **kwargs):
        if name == "sounddevice":
            raise ImportError()
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fail_output_import)
    engine = playback._engine(playback._StreamingTts())
    engine._running.set()
    engine._start_receipt_dispatcher()
    engine._play_thread = threading.Thread(target=engine._playback_loop, daemon=True)
    engine._play_thread.start()
    probe = playback._ReceiptProbe()
    try:
        _wait_worker_exit(engine)
        assert engine.output_state is OutputState.RECOVERABLE
        engine.speak_tracked(TrackedSpeech("missing", "public missing"), on_terminal=probe.on_terminal)
        assert playback._wait_until(lambda: len(probe.snapshot()[1]) == 1)
        assert probe.snapshot()[1][0][0].outcome is PlaybackOutcome.FAILED
        assert engine._running.is_set()
        assert engine._play_q.empty()
        assert engine._tts.calls == 0
    finally:
        engine.stop()


def test_unexpected_sentinel_exit_is_not_advertised_as_ready(monkeypatch):
    engine = playback._engine(playback._StreamingTts())
    playback._start_playback_harness(monkeypatch, engine)
    try:
        engine._play_q.put_nowait((None, None, 0, None, None))
        _wait_worker_exit(engine)
        assert engine._running.is_set()
        assert engine.output_state is OutputState.RECOVERABLE
        assert not engine.is_speaking
    finally:
        engine.stop()


def test_idle_release_blocked_stop_retains_quarantine_after_shared_detach(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    original_stream = playback._ManualOutputStream
    class BlockedStop(original_stream):
        def stop(self):
            self.stop_calls += 1
            entered.set()
            assert release.wait(timeout=4)
            self.active = False
    monkeypatch.setattr(playback, "_ManualOutputStream", BlockedStop)
    engine = playback._engine(playback._StreamingTts())
    engine.config.release_output_when_idle = True
    holder = playback._start_playback_harness(monkeypatch, engine, default_sr=22050)
    probe = playback._ReceiptProbe()
    engine.speak_tracked(TrackedSpeech("idle", "public idle"), on_terminal=probe.on_terminal)
    try:
        assert playback._wait_until(lambda: playback._fifo_count_is(engine, 3))
        holder["stream"].pull(3)
        assert entered.wait(timeout=1)
        assert engine._out_stream is None
        assert engine._output_quarantine and engine.is_speaking
        assert playback._wait_until(lambda: engine.output_state is OutputState.POISONED, timeout=2)
        assert engine._running.is_set()
        assert engine._output_quarantine and engine.is_speaking
        assert holder["stream"].active
        assert not engine.recover_output()
        assert playback._wait_until(lambda: len(probe.snapshot()[1]) == 1)
        assert probe.snapshot()[1][0][0].outcome is PlaybackOutcome.COMPLETED
    finally:
        release.set()
        engine.stop()
    assert holder["stream"].stop_calls == holder["stream"].close_calls == 1


def test_failure_dsp_reset_waits_for_capture_owner_safe_boundary(monkeypatch):
    from tests import test_sherpa_duplex_runtime as duplex
    class BlockingAec:
        always_on = True

        def __init__(self):
            self.active = threading.Event()
            self.release = threading.Event()
            self.resets = []
        def process_16k(self, samples, _far):
            self.active.set()
            try:
                assert self.release.wait(timeout=4)
                return samples
            finally:
                self.active.clear()
        def reset(self):
            assert not self.active.is_set(), "reset raced capture DSP"
            self.resets.append(threading.get_ident())
    aec = BlockingAec()
    tts = FailingTts()
    engine, _accepted = _production_capture_engine(monkeypatch, tts, aec)
    try:
        engine.start(EngineCallbacks())
        assert aec.active.wait(timeout=1)
        prior_resets = len(aec.resets)
        engine.speak("public DSP barrier")
        assert tts.submitted.wait(timeout=1)
        tts.release.set()
        assert duplex._wait_until(lambda: engine._output_reset_request is not None)
        assert engine._output_cleanup.snapshot().closed
        assert engine.output_state is OutputState.UNAVAILABLE
        assert engine._output_quarantine and engine.is_speaking
        assert len(aec.resets) == prior_resets
        assert not engine.recover_output()
        aec.release.set()
        assert duplex._wait_until(lambda: engine.output_state is OutputState.RECOVERABLE)
        assert len(aec.resets) == prior_resets + 1
        assert aec.resets[-1] == engine._capture_thread.ident
        assert engine._output_reset_request is None
        assert engine._output_reset_ack is not None
        assert not engine.is_speaking
        assert engine._running.is_set()
    finally:
        aec.release.set()
        tts.release.set()
        engine.stop()
