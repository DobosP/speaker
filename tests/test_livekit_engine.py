"""LiveKitEngine logic tests with fakes -- no livekit server / models / audio.

The async room/audio path needs a live LiveKit server and so is out of scope
here; these cover the thread-safe stepping logic (streaming ASR, barge-in) and
the no-op speak path, plus import-safety without the optional deps.
"""
import asyncio
import sys
import threading
import types

import numpy as np
import pytest

import core.engines.livekit as livekit_module
from always_on_agent.acoustic import AcousticSource, EndpointReason
from core.engine import (
    EngineCallbacks,
    PlaybackOutcome,
    TrackedSpeech,
    TranscriptAbortReason,
)
from core.engines.livekit import STT_SR, LiveKitEngine
from core.engines.sherpa import SherpaConfig


class _FakeStream:
    def __init__(self):
        self.fed = []

    def accept_waveform(self, sr, mono):
        self.fed.append((sr, len(mono)))


class _FakeRecognizer:
    """Emits one partial then an endpoint final on the first feed."""

    def __init__(self, result="hello world", endpoint=True):
        self._result = result
        self._endpoint = endpoint
        self._ready = True

    def create_stream(self):
        return _FakeStream()

    def is_ready(self, stream):
        if self._ready:
            self._ready = False
            return True
        return False

    def decode_stream(self, stream):
        pass

    def get_result(self, stream):
        return self._result

    def is_endpoint(self, stream):
        return self._endpoint

    def reset(self, stream):
        self._ready = True


class _RetractingRecognizer(_FakeRecognizer):
    def __init__(self):
        super().__init__()
        self._result_calls = 0

    def get_result(self, stream):
        self._result_calls += 1
        return "assistant mode" if self._result_calls == 1 else ""


class _FakeVad:
    def __init__(self, speech=True):
        self._speech = speech
        self.fed = 0

    def accept_waveform(self, mono):
        self.fed += 1

    def is_speech_detected(self):
        return self._speech


class _FakeAudioFrame:
    def __init__(
        self,
        *,
        data,
        sample_rate,
        num_channels,
        samples_per_channel,
    ):
        self.data = data
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.samples_per_channel = samples_per_channel


class _FakeAudio:
    sample_rate = 48000
    samples = np.full(960, 0.1, dtype=np.float32)


class _TwoFrameAudio:
    sample_rate = 48000
    samples = np.full(1920, 0.1, dtype=np.float32)


class _FakeHandle:
    def __init__(self):
        self.disposed = False

    def dispose(self):
        self.disposed = True


class _GatedTTS:
    def __init__(self, *, audio=None):
        self.calls = []
        self.audio = audio or _FakeAudio()
        self._entered = {}
        self._returned = {}
        self._blocked = {}
        self._lock = threading.Lock()

    def block(self, text):
        with self._lock:
            entered = self._entered.setdefault(text, threading.Event())
            release = self._blocked.setdefault(text, threading.Event())
        return entered, release

    def returned(self, text):
        with self._lock:
            return self._returned.setdefault(text, threading.Event())

    def generate(self, text, *, sid, speed):
        with self._lock:
            self.calls.append(text)
            entered = self._entered.setdefault(text, threading.Event())
            returned = self._returned.setdefault(text, threading.Event())
            release = self._blocked.get(text)
        entered.set()
        if release is not None and not release.wait(2.0):
            raise TimeoutError(f"test did not release TTS for {text}")
        returned.set()
        return self.audio


class _FakeAudioSource:
    def __init__(
        self,
        *,
        block_clear=False,
        fail_capture=False,
        fail_capture_at=None,
        fail_clear=False,
    ):
        self.log = []
        self.fail_capture = fail_capture
        self.fail_capture_at = fail_capture_at
        self.fail_clear = fail_clear
        self.before_capture_return = None
        self._ffi_handle = _FakeHandle()
        self.capture_started = [threading.Event() for _ in range(8)]
        self.wait_started = [threading.Event() for _ in range(8)]
        self.wait_gates = [None] * 8
        self.clear_entered = threading.Event()
        self.clear_release = threading.Event()
        if not block_clear:
            self.clear_release.set()
        self.clear_count = 0
        self._lock = threading.Lock()

    async def capture_frame(self, frame):
        with self._lock:
            index = sum(item == "capture" for item in self.log)
            self.log.append("capture")
        self.capture_started[index].set()
        if self.fail_capture or index == self.fail_capture_at:
            raise RuntimeError("capture failed")
        if self.before_capture_return is not None:
            self.before_capture_return()

    async def wait_for_playout(self):
        with self._lock:
            index = sum(item == "wait" for item in self.log)
            gate = asyncio.Event()
            self.wait_gates[index] = gate
            self.log.append("wait")
        self.wait_started[index].set()
        await gate.wait()

    def clear_queue(self):
        with self._lock:
            self.log.append("clear")
            self.clear_count += 1
        self.clear_entered.set()
        if self.fail_clear:
            raise RuntimeError("clear failed")
        if not self.clear_release.wait(2.0):
            raise TimeoutError("test did not release source clear")
        for gate in tuple(self.wait_gates):
            if gate is not None:
                gate.set()

    def release_playout(self, loop, index):
        gate = self.wait_gates[index]
        assert gate is not None
        loop.call_soon_threadsafe(gate.set)

    async def aclose(self):
        with self._lock:
            self.log.append("close")
        self._ffi_handle.dispose()


class _LiveKitHarness:
    def __init__(self, engine, *, source=None, tts=None):
        self.engine = engine
        self.source = source or _FakeAudioSource()
        self.tts = tts or _GatedTTS()
        self.loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._stopping = threading.Event()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        assert self._ready.wait(1.0)

        async def install():
            engine._playback_slots = asyncio.Semaphore(2)
            engine._synth_order_lock = asyncio.Lock()
            engine._speak_lock = asyncio.Lock()

        asyncio.run_coroutine_threadsafe(install(), self.loop).result(1.0)
        engine._loop = self.loop
        engine._source = self.source
        engine._tts = self.tts
        engine._running.set()
        engine._output_ready.set()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        # This sandbox's selector does not reliably wake on the cross-thread
        # self-pipe. A short loop timer keeps public run_coroutine_threadsafe()
        # admission deterministic without sleeps in the tests themselves.
        self.loop.call_soon(self._heartbeat)
        self.loop.call_soon(self._ready.set)
        self.loop.run_forever()

    def _heartbeat(self):
        if not self._stopping.is_set():
            self.loop.call_later(0.005, self._heartbeat)

    def flush(self):
        asyncio.run_coroutine_threadsafe(asyncio.sleep(0), self.loop).result(1.0)

    def close(self):
        self.engine._running.clear()
        self.engine.stop_speaking()
        self.flush()
        self._stopping.set()
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(1.0)
        assert not self.thread.is_alive()
        self.loop.close()


@pytest.fixture(autouse=True)
def _fake_livekit_module(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "livekit",
        types.SimpleNamespace(
            rtc=types.SimpleNamespace(AudioFrame=_FakeAudioFrame)
        ),
    )


def _engine():
    return LiveKitEngine(SherpaConfig(), url="ws://localhost:7880", token="t")


def test_asr_loop_uses_shared_hotword_stream_helper(monkeypatch):
    config = SherpaConfig(asr_hotwords="VAULT")
    eng = LiveKitEngine(config, url="ws://localhost:7880", token="t")
    recognizer = object()
    sentinel = object()
    calls = []
    eng._recognizer = recognizer

    def create(rec, cfg):
        calls.append((rec, cfg))
        return sentinel

    monkeypatch.setattr(livekit_module, "create_recognizer_stream", create)
    eng._asr_loop()

    assert calls == [(recognizer, config)]
    assert eng._asr_startup.is_set()
    assert eng._asr_stream is None  # retired on the owning worker


def test_start_fails_closed_before_livekit_loop_without_hotword_stream_support(
    monkeypatch,
):
    class _NoHotwordRecognizer:
        plain_calls = 0

        def create_stream(self):
            self.plain_calls += 1
            return _FakeStream()

    recognizer = _NoHotwordRecognizer()
    monkeypatch.setattr(
        livekit_module, "build_recognizer", lambda _config: recognizer
    )
    monkeypatch.setattr(livekit_module, "build_vad", lambda _config: None)
    monkeypatch.setattr(livekit_module, "build_tts", lambda _config: None)
    eng = LiveKitEngine(
        SherpaConfig(asr_hotwords="VAULT"),
        url="ws://localhost:7880",
        token="t",
    )

    with pytest.raises(RuntimeError, match="ASR stream startup failed"):
        eng.start(EngineCallbacks())

    assert recognizer.plain_calls == 0
    assert not eng._running.is_set()
    assert eng._asr_stream is None
    assert eng._asr_thread is None
    assert eng._loop_thread is None


def test_blocked_asr_startup_retains_worker_until_native_call_returns(
    monkeypatch,
):
    entered = threading.Event()
    release = threading.Event()

    class _BlockingRecognizer:
        def create_stream(self):
            entered.set()
            assert release.wait(2.0)
            return _FakeStream()

    monkeypatch.setattr(
        livekit_module,
        "build_recognizer",
        lambda _config: _BlockingRecognizer(),
    )
    monkeypatch.setattr(livekit_module, "build_vad", lambda _config: None)
    monkeypatch.setattr(livekit_module, "build_tts", lambda _config: None)
    monkeypatch.setattr(livekit_module, "ASR_START_TIMEOUT_SEC", 0.01)
    monkeypatch.setattr(livekit_module, "ASR_START_JOIN_TIMEOUT_SEC", 0.01)
    eng = _engine()

    with pytest.raises(RuntimeError, match="native worker retained"):
        eng.start(EngineCallbacks())

    assert entered.is_set()
    assert eng._asr_thread is not None
    assert eng._asr_thread.is_alive()
    assert eng._loop_thread is None
    assert not eng._running.is_set()
    release.set()
    eng._asr_thread.join(1.0)
    assert not eng._asr_thread.is_alive()
    assert eng._asr_stream is None
    eng.stop()
    assert eng._asr_thread is None


def test_feed_asr_legacy_callbacks_remain_fallback():
    eng = _engine()
    partials, finals = [], []
    eng._cb = EngineCallbacks(on_partial=partials.append, on_final=finals.append)
    eng._recognizer = _FakeRecognizer()
    eng._asr_stream = eng._recognizer.create_stream()

    eng._feed_asr(np.zeros(160, dtype=np.float32))

    assert partials == ["hello world"]
    assert finals == ["hello world"]
    assert eng._asr_stream.fed == [(STT_SR, 160)]


def test_feed_asr_prefers_scoped_typed_partial_and_final():
    eng = _engine()
    partials, finals = [], []
    eng._cb = EngineCallbacks(
        on_partial_result=partials.append,
        on_final_result=finals.append,
    )
    eng._recognizer = _FakeRecognizer()
    eng._asr_stream = eng._recognizer.create_stream()

    eng._feed_asr(np.zeros(160, dtype=np.float32))

    assert [result.text for result in partials] == ["hello world"]
    assert [result.text for result in finals] == ["hello world"]
    assert partials[0].revision == 0
    assert finals[0].revision == 1
    assert partials[0].acoustic is not None
    assert finals[0].acoustic is not None
    assert partials[0].acoustic.spans[0].key == finals[0].acoustic.spans[0].key
    assert partials[0].acoustic.spans[0].source == AcousticSource.REMOTE_TRACK
    assert (
        finals[0].acoustic.spans[0].endpoint_reason
        == EndpointReason.ASR
    )


def test_empty_endpoint_emits_typed_abort_for_scoped_partial():
    eng = _engine()
    partials, aborts = [], []
    eng._cb = EngineCallbacks(
        on_partial_result=partials.append,
        on_transcript_abort=aborts.append,
    )
    eng._recognizer = _RetractingRecognizer()
    eng._asr_stream = eng._recognizer.create_stream()

    eng._feed_asr(np.zeros(160, dtype=np.float32))

    assert [result.text for result in partials] == ["assistant mode"]
    assert len(aborts) == 1
    assert aborts[0].reason == TranscriptAbortReason.EMPTY_FINAL
    assert aborts[0].revision == 1
    assert (
        partials[0].acoustic.spans[0].key
        == aborts[0].acoustic.spans[0].key
    )


def test_feed_asr_noop_without_recognizer():
    eng = _engine()
    finals = []
    eng._cb = EngineCallbacks(on_final=finals.append)
    eng._feed_asr(np.zeros(160, dtype=np.float32))  # no recognizer/stream
    assert finals == []


def test_watch_barge_in_fires_after_min_speech():
    cfg = SherpaConfig(barge_in_min_speech_sec=0.2)
    eng = LiveKitEngine(cfg, url="ws://localhost:7880", token="t")
    barges = []
    eng._cb = EngineCallbacks(on_barge_in=lambda: barges.append(1))
    eng._vad = _FakeVad(speech=True)

    frame = np.zeros(STT_SR // 10, dtype=np.float32)  # 0.1s
    eng._watch_barge_in(frame)
    assert barges == []  # 0.1s < 0.2s threshold
    eng._watch_barge_in(frame)
    assert barges == [1]  # 0.2s reached


def test_watch_barge_in_silence_resets():
    eng = _engine()
    barges = []
    eng._cb = EngineCallbacks(on_barge_in=lambda: barges.append(1))
    eng._vad = _FakeVad(speech=False)
    eng._watch_barge_in(np.zeros(STT_SR, dtype=np.float32))  # 1s but no speech
    assert barges == []
    assert eng._voiced_run == 0.0


def test_speak_without_tts_or_loop_calls_on_done():
    eng = _engine()
    done = []
    eng.speak("hi", on_done=lambda: done.append(1))
    assert done == [1]
    assert eng.is_speaking is False


def test_tracked_rejection_is_exact_and_legacy_override_stays_legacy():
    eng = _engine()
    receipts = []
    eng.speak_tracked(
        TrackedSpeech("rejected-id", "not running"),
        on_terminal=receipts.append,
    )
    assert [receipt.outcome for receipt in receipts] == [PlaybackOutcome.DROPPED]

    class LegacyOnlyLiveKit(LiveKitEngine):
        def speak(self, text, on_done=None):
            if on_done is not None:
                on_done()

    legacy = LegacyOnlyLiveKit(
        SherpaConfig(),
        url="ws://localhost:7880",
        token="t",
    )
    assert not legacy.playback_capabilities.tracked_terminal


def test_construction_does_not_require_optional_deps():
    # Building the engine must not import livekit / sherpa_onnx (models are only
    # built in start()); construction should be cheap and dependency-free.
    eng = _engine()
    assert eng._recognizer is None and eng._tts is None and eng._loop is None


def test_session_initialization_failure_clears_running_state():
    eng = _engine()
    eng._running.set()
    eng._output_ready.set()

    with pytest.raises(AttributeError):
        asyncio.run(eng._session())

    assert not eng._running.is_set()
    assert not eng._output_ready.is_set()
    assert eng._source is None
    assert eng._room is None


def test_repeated_stop_cannot_cancel_session_teardown_twice():
    eng = _engine()
    harness = _LiveKitHarness(eng)
    cleanup_started = threading.Event()
    cleanup_finished = threading.Event()
    holder = {}

    async def install_session():
        cleanup_gate = asyncio.Event()
        holder["cleanup_gate"] = cleanup_gate

        async def fake_session():
            try:
                await asyncio.Event().wait()
            finally:
                eng._session_teardown.set()
                cleanup_started.set()
                await cleanup_gate.wait()
                cleanup_finished.set()

        task = asyncio.create_task(fake_session())
        eng._session_task = task
        holder["task"] = task

    try:
        asyncio.run_coroutine_threadsafe(
            install_session(),
            harness.loop,
        ).result(1.0)
        eng.stop()
        assert cleanup_started.wait(1.0)

        eng.stop()
        assert not cleanup_finished.is_set()
        assert not holder["task"].done()

        harness.loop.call_soon_threadsafe(holder["cleanup_gate"].set)
        assert cleanup_finished.wait(1.0)
        harness.flush()
        assert holder["task"].done()
        eng._session_task = None
    finally:
        harness.loop.call_soon_threadsafe(holder["cleanup_gate"].set)
        harness.close()


def test_tracked_terminal_waits_for_source_playout():
    eng = _engine()
    source = _FakeAudioSource()
    harness = _LiveKitHarness(eng, source=source)
    started, receipts = [], []
    terminal = threading.Event()
    try:
        caps = eng.playback_capabilities
        assert caps.tracked_terminal and caps.exact_started
        assert not caps.sample_counts and not caps.speech_style_hints

        eng.speak_tracked(
            TrackedSpeech("fragment-1", "complete answer"),
            on_started=started.append,
            on_terminal=lambda receipt: (receipts.append(receipt), terminal.set()),
        )

        assert source.capture_started[0].wait(1.0)
        assert source.wait_started[0].wait(1.0)
        assert started == ["fragment-1"]
        assert receipts == []

        source.release_playout(harness.loop, 0)
        assert terminal.wait(1.0)
        assert len(receipts) == 1
        receipt = receipts[0]
        assert receipt.fragment_id == "fragment-1"
        assert receipt.outcome is PlaybackOutcome.COMPLETED
        assert receipt.safe_text_prefix == "complete answer"
        assert receipt.played_samples is None
        assert receipt.total_samples is None
        assert receipt.output_sample_rate is None

        source.release_playout(harness.loop, 0)
        eng.stop_speaking()
        assert source.clear_entered.wait(1.0)
        harness.flush()
        assert len(receipts) == 1
        assert started == ["fragment-1"]
    finally:
        harness.close()


def test_tracked_waits_for_connected_published_output():
    eng = _engine()
    tts = _GatedTTS()
    source = _FakeAudioSource()
    harness = _LiveKitHarness(eng, source=source, tts=tts)
    receipts = []
    terminal = threading.Event()
    try:
        eng._output_ready.clear()
        eng.speak_tracked(
            TrackedSpeech("waiting-id", "wait for transport"),
            on_terminal=lambda receipt: (receipts.append(receipt), terminal.set()),
        )
        harness.flush()
        assert tts.calls == []
        assert receipts == []
        assert not source.capture_started[0].is_set()

        eng._output_ready.set()
        assert source.capture_started[0].wait(1.0)
        assert source.wait_started[0].wait(1.0)
        source.release_playout(harness.loop, 0)
        assert terminal.wait(1.0)
        assert receipts[0].outcome is PlaybackOutcome.COMPLETED
    finally:
        harness.close()


def test_disposed_livekit_source_fails_without_false_start():
    eng = _engine()
    tts = _GatedTTS()
    source = _FakeAudioSource()
    source._ffi_handle.disposed = True
    harness = _LiveKitHarness(eng, source=source, tts=tts)
    starts, receipts = [], []
    terminal = threading.Event()
    try:
        eng.speak_tracked(
            TrackedSpeech("disposed-id", "cannot route"),
            on_started=starts.append,
            on_terminal=lambda receipt: (receipts.append(receipt), terminal.set()),
        )
        assert terminal.wait(1.0)
        assert tts.calls == []
        assert starts == []
        assert receipts[0].outcome is PlaybackOutcome.FAILED
        assert receipts[0].safe_text_prefix == ""
        assert not source.capture_started[0].is_set()
    finally:
        harness.close()


def test_source_disposed_during_capture_never_emits_exact_start():
    eng = _engine()
    source = _FakeAudioSource()
    harness = _LiveKitHarness(eng, source=source)
    starts, receipts = [], []
    terminal = threading.Event()
    try:
        source.before_capture_return = lambda: setattr(
            source._ffi_handle,
            "disposed",
            True,
        )
        eng.speak_tracked(
            TrackedSpeech("dispose-race-id", "capture race"),
            on_started=starts.append,
            on_terminal=lambda receipt: (receipts.append(receipt), terminal.set()),
        )
        assert terminal.wait(1.0)
        assert starts == []
        assert receipts[0].outcome is PlaybackOutcome.FAILED
        assert receipts[0].safe_text_prefix == ""
        assert source.log == ["capture", "clear"]
    finally:
        harness.close()


def test_one_successor_pregenerates_while_pipeline_memory_stays_bounded():
    eng = _engine()
    tts = _GatedTTS()
    second_entered, second_release = tts.block("second")
    second_returned = tts.returned("second")
    third_returned = tts.returned("third")
    source = _FakeAudioSource()
    harness = _LiveKitHarness(eng, source=source, tts=tts)
    starts, receipts = [], []
    first_terminal = threading.Event()
    second_terminal = threading.Event()
    third_terminal = threading.Event()
    try:
        eng.speak_tracked(
            TrackedSpeech("first-id", "first"),
            on_started=starts.append,
            on_terminal=lambda receipt: (
                receipts.append(receipt),
                first_terminal.set(),
            ),
        )
        assert source.capture_started[0].wait(1.0)
        assert source.wait_started[0].wait(1.0)

        eng.speak_tracked(
            TrackedSpeech("second-id", "second"),
            on_started=starts.append,
            on_terminal=lambda receipt: (
                receipts.append(receipt),
                second_terminal.set(),
            ),
        )
        assert second_entered.wait(1.0)
        second_release.set()
        assert second_returned.wait(1.0)
        assert not source.capture_started[1].is_set()
        assert starts == ["first-id"]

        eng.speak_tracked(
            TrackedSpeech("third-id", "third"),
            on_started=starts.append,
            on_terminal=lambda receipt: (
                receipts.append(receipt),
                third_terminal.set(),
            ),
        )
        harness.flush()
        assert tts.calls == ["first", "second"]
        assert not third_returned.is_set()

        source.release_playout(harness.loop, 0)
        assert first_terminal.wait(1.0)
        assert source.capture_started[1].wait(1.0)
        assert source.wait_started[1].wait(1.0)
        assert third_returned.wait(1.0)
        assert not source.capture_started[2].is_set()
        source.release_playout(harness.loop, 1)
        assert second_terminal.wait(1.0)
        assert source.capture_started[2].wait(1.0)
        assert source.wait_started[2].wait(1.0)
        source.release_playout(harness.loop, 2)
        assert third_terminal.wait(1.0)
        assert [receipt.outcome for receipt in receipts] == [
            PlaybackOutcome.COMPLETED,
            PlaybackOutcome.COMPLETED,
            PlaybackOutcome.COMPLETED,
        ]
        assert starts == ["first-id", "second-id", "third-id"]
    finally:
        second_release.set()
        harness.close()


def test_stop_during_native_synthesis_retains_active_and_drops_queued():
    eng = _engine()
    tts = _GatedTTS()
    first_entered, first_release = tts.block("first")
    source = _FakeAudioSource()
    harness = _LiveKitHarness(eng, source=source, tts=tts)
    starts = []
    receipts = {}
    first_terminal = threading.Event()
    second_terminal = threading.Event()

    def terminal_for(name, event):
        def record(receipt):
            receipts.setdefault(name, []).append(receipt)
            event.set()

        return record

    try:
        eng.speak_tracked(
            TrackedSpeech("first-id", "first"),
            on_started=starts.append,
            on_terminal=terminal_for("first", first_terminal),
        )
        assert first_entered.wait(1.0)
        eng.speak_tracked(
            TrackedSpeech("second-id", "second"),
            on_started=starts.append,
            on_terminal=terminal_for("second", second_terminal),
        )

        eng.stop_speaking()
        assert second_terminal.wait(1.0)
        assert not first_terminal.is_set()
        assert receipts["second"][0].outcome is PlaybackOutcome.DROPPED
        assert receipts["second"][0].safe_text_prefix == ""
        assert tts.calls == ["first"]
        assert starts == []
        assert not source.capture_started[0].is_set()

        first_release.set()
        assert first_terminal.wait(1.0)
        harness.flush()
        assert receipts["first"][0].outcome is PlaybackOutcome.DROPPED
        assert receipts["first"][0].safe_text_prefix == ""
        assert tts.calls == ["first"]
        assert starts == []
        assert not source.capture_started[0].is_set()

        eng.stop_speaking()
        harness.flush()
        assert len(receipts["first"]) == 1
        assert len(receipts["second"]) == 1
    finally:
        first_release.set()
        harness.close()


def test_stop_clear_precedes_fresh_generation_playback():
    eng = _engine()
    source = _FakeAudioSource(block_clear=True)
    harness = _LiveKitHarness(eng, source=source)
    starts = []
    receipts = {}
    clear_seen_by_queued_callback = []
    terminal_events = {
        name: threading.Event() for name in ("old-active", "old-queued", "fresh")
    }

    def terminal_for(name):
        def record(receipt):
            if name == "old-queued":
                clear_seen_by_queued_callback.append(
                    source.clear_entered.wait(1.0)
                )
            receipts.setdefault(name, []).append(receipt)
            terminal_events[name].set()

        return record

    try:
        eng.speak_tracked(
            TrackedSpeech("old-active-id", "old active"),
            on_started=starts.append,
            on_terminal=terminal_for("old-active"),
        )
        assert source.capture_started[0].wait(1.0)
        assert source.wait_started[0].wait(1.0)
        eng.speak_tracked(
            TrackedSpeech("old-queued-id", "old queued"),
            on_started=starts.append,
            on_terminal=terminal_for("old-queued"),
        )

        eng.stop_speaking()
        assert terminal_events["old-queued"].wait(1.0)
        assert clear_seen_by_queued_callback == [True]
        assert source.clear_entered.wait(1.0)
        eng.speak_tracked(
            TrackedSpeech("fresh-id", "fresh"),
            on_started=starts.append,
            on_terminal=terminal_for("fresh"),
        )
        # clear_queue is synchronously blocking the owning loop, so a fresh
        # frame cannot be admitted until the old-generation barrier releases.
        assert not source.capture_started[1].is_set()
        assert starts == ["old-active-id"]

        source.clear_release.set()
        assert terminal_events["old-active"].wait(1.0)
        assert source.capture_started[1].wait(1.0)
        assert source.wait_started[1].wait(1.0)
        source.release_playout(harness.loop, 1)
        assert terminal_events["fresh"].wait(1.0)

        assert receipts["old-active"][0].outcome is PlaybackOutcome.INTERRUPTED
        assert receipts["old-queued"][0].outcome is PlaybackOutcome.DROPPED
        assert receipts["fresh"][0].outcome is PlaybackOutcome.COMPLETED
        assert receipts["old-active"][0].safe_text_prefix == ""
        assert receipts["old-queued"][0].safe_text_prefix == ""
        assert receipts["fresh"][0].safe_text_prefix == "fresh"
        assert starts == ["old-active-id", "fresh-id"]
        assert all(len(items) == 1 for items in receipts.values())
        assert source.log.index("capture") < source.log.index("clear")
        assert source.log.index("clear") < len(source.log) - 1 - source.log[::-1].index(
            "capture"
        )
        assert source.clear_count == 1
    finally:
        source.clear_release.set()
        harness.close()


def test_stop_racing_capture_clears_source_before_interrupt_receipt():
    eng = _engine()
    source = _FakeAudioSource()
    harness = _LiveKitHarness(eng, source=source)
    starts, receipts = [], []
    terminal = threading.Event()
    clear_counts_at_terminal = []

    def record(receipt):
        receipts.append(receipt)
        clear_counts_at_terminal.append(source.clear_count)
        terminal.set()

    try:
        # This hook executes on the LiveKit loop immediately before a successful
        # capture returns. It deterministically reproduces STOP winning the
        # generation race while its scheduled clear has not run yet.
        source.before_capture_return = eng.stop_speaking
        eng.speak_tracked(
            TrackedSpeech("race-id", "race"),
            on_started=starts.append,
            on_terminal=record,
        )
        assert terminal.wait(1.0)
        assert starts == ["race-id"]
        assert len(receipts) == 1
        assert receipts[0].outcome is PlaybackOutcome.INTERRUPTED
        assert receipts[0].safe_text_prefix == ""
        assert clear_counts_at_terminal == [1]
        assert source.log == ["capture", "clear"]
    finally:
        harness.close()


def test_failure_after_first_frame_clears_before_failed_receipt():
    eng = _engine()
    source = _FakeAudioSource(fail_capture_at=1)
    tts = _GatedTTS(audio=_TwoFrameAudio())
    harness = _LiveKitHarness(eng, source=source, tts=tts)
    starts, receipts = [], []
    terminal = threading.Event()
    clear_counts_at_terminal = []

    def record(receipt):
        receipts.append(receipt)
        clear_counts_at_terminal.append(source.clear_count)
        terminal.set()

    try:
        eng.speak_tracked(
            TrackedSpeech("partial-failure-id", "two frames"),
            on_started=starts.append,
            on_terminal=record,
        )
        assert terminal.wait(1.0)
        assert starts == ["partial-failure-id"]
        assert len(receipts) == 1
        assert receipts[0].outcome is PlaybackOutcome.FAILED
        assert receipts[0].safe_text_prefix == ""
        assert clear_counts_at_terminal == [1]
        assert source.log == ["capture", "capture", "clear"]
    finally:
        harness.close()


def test_clear_failure_disposes_source_before_terminal_receipt():
    eng = _engine()
    source = _FakeAudioSource(fail_clear=True)
    harness = _LiveKitHarness(eng, source=source)
    starts, receipts = [], []
    terminal = threading.Event()
    disposed_at_terminal = []

    def record(receipt):
        receipts.append(receipt)
        disposed_at_terminal.append(source._ffi_handle.disposed)
        terminal.set()

    try:
        eng.speak_tracked(
            TrackedSpeech("clear-failure-id", "buffered"),
            on_started=starts.append,
            on_terminal=record,
        )
        assert source.capture_started[0].wait(1.0)
        assert source.wait_started[0].wait(1.0)
        eng.stop_speaking()
        assert terminal.wait(1.0)
        assert starts == ["clear-failure-id"]
        assert receipts[0].outcome is PlaybackOutcome.FAILED
        assert receipts[0].safe_text_prefix == ""
        assert disposed_at_terminal == [True]
        assert source.log.index("clear") < source.log.index("close")
        assert source.log[-1] == "close"
        assert eng._source is None
    finally:
        harness.close()


def test_terminal_claim_precedes_reentrant_raising_callback():
    eng = _engine()
    source = _FakeAudioSource(fail_capture=True)
    harness = _LiveKitHarness(eng, source=source)
    failed_receipts, fresh_receipts = [], []
    failed = threading.Event()
    fresh = threading.Event()
    starts = []

    def raising_terminal(receipt):
        failed_receipts.append(receipt)
        eng.stop_speaking()
        failed.set()
        raise RuntimeError("consumer failed")

    try:
        eng.speak_tracked(
            TrackedSpeech("failed-id", "will fail"),
            on_started=starts.append,
            on_terminal=raising_terminal,
        )
        assert failed.wait(1.0)
        assert source.clear_entered.wait(1.0)
        assert len(failed_receipts) == 1
        assert failed_receipts[0].outcome is PlaybackOutcome.FAILED
        assert failed_receipts[0].safe_text_prefix == ""
        assert starts == []

        source.fail_capture = False
        eng.speak_tracked(
            TrackedSpeech("fresh-id", "recovers"),
            on_started=starts.append,
            on_terminal=lambda receipt: (
                fresh_receipts.append(receipt),
                fresh.set(),
            ),
        )
        assert source.capture_started[1].wait(1.0)
        assert source.wait_started[0].wait(1.0)
        source.release_playout(harness.loop, 0)
        assert fresh.wait(1.0)
        assert len(failed_receipts) == 1
        assert len(fresh_receipts) == 1
        assert fresh_receipts[0].outcome is PlaybackOutcome.COMPLETED
        assert fresh_receipts[0].safe_text_prefix == "recovers"
        assert starts == ["fresh-id"]
    finally:
        harness.close()
