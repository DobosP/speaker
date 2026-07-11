"""Playback-sink + streaming-synthesis logic for the local sherpa engine.

No audio hardware, no sherpa-onnx, no models: a fake TTS is injected and the
pure synthesis/queue logic is exercised directly. The threaded OutputStream
path (``_playback_loop``) needs a real device and is out of scope here, so the
testable pieces (`_synthesize`, queue/flush) are factored out of the thread.
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from core.audio_frontend import InputAGC, apply_gain_soft_limit
from core.engines._aec import FarEndRing, PlaybackFIFO
from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine
from core.engine import EngineCallbacks, PlaybackOutcome, PlaybackReceipt, TrackedSpeech
from core.metrics import TTS_FIRST_AUDIO


class _GenAudio:
    def __init__(self, samples, sample_rate=22050):
        self.samples = samples
        self.sample_rate = sample_rate


class _StreamingTts:
    """Calls the streaming callback with two chunks, like sherpa-onnx."""

    sample_rate = 22050

    def __init__(self):
        self.calls = 0

    def generate(self, text, sid=0, speed=1.0, callback=None):
        self.calls += 1
        chunks = [np.array([0.1, 0.2], dtype="float32"), np.array([0.3], dtype="float32")]
        if callback is not None:
            for c in chunks:
                if callback(c, 1.0) == 0:
                    break
            return _GenAudio(np.concatenate(chunks))
        return _GenAudio(np.concatenate(chunks))


class _NonStreamingTts:
    """A build whose generate() has no ``callback`` param (raises TypeError)."""

    sample_rate = 16000

    def generate(self, text, sid=0, speed=1.0):
        return _GenAudio(np.arange(4000, dtype="float32"), sample_rate=16000)


def _engine(tts) -> SherpaOnnxEngine:
    # tts_dc_block off: these tests assert streaming MECHANICS (chunk counts +
    # raw pass-through values). The DC-blocking high-pass is an orthogonal output
    # stage whose own chunk/sentence continuity is covered in test_audio_frontend.
    eng = SherpaOnnxEngine(SherpaConfig(tts_dc_block=False))
    eng._tts = tts
    return eng


def _wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return predicate()


class _ManualOutputStream:
    """Device-free PortAudio sink whose callback advances only on ``pull``."""

    def __init__(
        self,
        *args,
        samplerate=16000,
        channels=1,
        callback=None,
        **kwargs,
    ):
        del args, kwargs
        self.samplerate = int(samplerate)
        self.channels = int(channels)
        self.callback = callback
        self.active = False
        self.closed = False
        self.pull_thread_ids: list[int] = []

    def start(self):
        self.active = True

    def pull(self, frames):
        out = np.zeros((int(frames), self.channels), dtype="float32")
        self.pull_thread_ids.append(threading.get_ident())
        if self.callback is not None:
            self.callback(out, int(frames), None, None)
        return out

    def stop(self):
        self.active = False

    def close(self):
        self.active = False
        self.closed = True


def _start_playback_harness(
    monkeypatch,
    engine,
    *,
    default_sr=16000,
    supported_rates=None,
):
    holder = {"streams": []}

    def output_stream(*args, **kwargs):
        stream = _ManualOutputStream(*args, **kwargs)
        holder["stream"] = stream
        holder["streams"].append(stream)
        return stream

    def check_output_settings(*, samplerate, **_kwargs):
        if supported_rates is not None and int(samplerate) not in supported_rates:
            raise RuntimeError("scripted unsupported output rate")

    fake_sd = SimpleNamespace(
        OutputStream=output_stream,
        query_devices=lambda *_args, **_kwargs: {
            "name": "manual-output",
            "default_samplerate": default_sr,
        },
        check_output_settings=check_output_settings,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    engine._playback_stopping.clear()
    engine._running.set()
    engine._start_receipt_dispatcher()
    engine._play_thread = threading.Thread(
        target=engine._playback_loop,
        name="test-sherpa-playback",
        daemon=True,
    )
    engine._play_thread.start()
    return holder


class _ReceiptProbe:
    def __init__(self):
        self.lock = threading.Lock()
        self.started: list[tuple[str, int]] = []
        self.receipts: list[tuple[PlaybackReceipt, int]] = []

    def on_started(self, fragment_id):
        with self.lock:
            self.started.append((fragment_id, threading.get_ident()))

    def on_terminal(self, receipt):
        with self.lock:
            self.receipts.append((receipt, threading.get_ident()))

    def snapshot(self):
        with self.lock:
            return list(self.started), list(self.receipts)


class _FixedTts:
    sample_rate = 16000

    def __init__(self, count):
        self.samples = np.linspace(0.05, 0.25, int(count), dtype="float32")

    def generate(self, text, sid=0, speed=1.0):
        del text, sid, speed
        return _GenAudio(self.samples.copy(), sample_rate=self.sample_rate)


class _WedgedStreamingTts:
    sample_rate = 16000

    def __init__(self, count=200):
        self.samples = np.ones(int(count), dtype="float32") * 0.1
        self.entered = threading.Event()
        self.release = threading.Event()

    def generate(self, text, sid=0, speed=1.0, callback=None):
        del text, sid, speed
        assert callback is not None
        callback(self.samples.copy(), 1.0)
        self.entered.set()
        self.release.wait(timeout=3.0)  # deliberately ignores cancellation
        return _GenAudio(self.samples.copy(), sample_rate=self.sample_rate)


def test_sherpa_tracked_receipt_waits_for_exact_sink_drain(monkeypatch):
    engine = _engine(_StreamingTts())
    probe = _ReceiptProbe()
    holder = _start_playback_harness(monkeypatch, engine, default_sr=22050)
    engine.speak_tracked(
        TrackedSpeech("fragment-1", "hello there"),
        on_started=probe.on_started,
        on_terminal=probe.on_terminal,
    )
    try:
        assert _wait_until(lambda: "stream" in holder and engine._fifo.count() == 3)
        assert probe.snapshot() == ([], [])

        holder["stream"].pull(2)
        assert _wait_until(lambda: len(probe.snapshot()[0]) == 1)
        assert probe.snapshot()[1] == []

        holder["stream"].pull(4)
        assert _wait_until(lambda: len(probe.snapshot()[1]) == 1)
        started, terminal = probe.snapshot()
        receipt, terminal_thread = terminal[0]
        assert started[0][0] == "fragment-1"
        assert receipt == PlaybackReceipt(
            fragment_id="fragment-1",
            outcome=PlaybackOutcome.COMPLETED,
            safe_text_prefix="hello there",
            played_samples=3,
            total_samples=3,
            output_sample_rate=22050,
        )
        assert started[0][1] == terminal_thread
        assert terminal_thread not in holder["stream"].pull_thread_ids
    finally:
        engine.stop()


def test_sherpa_tracked_barge_receipt_waits_for_owned_fade(monkeypatch):
    engine = _engine(_FixedTts(200))
    probe = _ReceiptProbe()
    holder = _start_playback_harness(monkeypatch, engine)
    engine.speak_tracked(
        TrackedSpeech("fragment-cut", "a longer fragment"),
        on_started=probe.on_started,
        on_terminal=probe.on_terminal,
    )
    try:
        assert _wait_until(lambda: "stream" in holder and engine._fifo.count() == 200)
        holder["stream"].pull(50)
        assert _wait_until(lambda: len(probe.snapshot()[0]) == 1)

        engine.stop_speaking()
        assert engine._fifo.count() == 64  # 4 ms at the exact 16 kHz sink rate
        assert probe.snapshot()[1] == []

        holder["stream"].pull(64)
        assert _wait_until(lambda: len(probe.snapshot()[1]) == 1)
        [entry] = probe.snapshot()[1]
        receipt, _thread_id = entry
        assert receipt.outcome is PlaybackOutcome.INTERRUPTED
        assert receipt.safe_text_prefix == ""
        assert receipt.played_samples == 114
        assert receipt.total_samples == 200
        assert receipt.output_sample_rate == 16000
    finally:
        engine.stop()


def test_sherpa_advertises_receipts_without_bypassing_speak_only_subclasses():
    capabilities = SherpaOnnxEngine(SherpaConfig()).playback_capabilities
    assert capabilities.tracked_terminal
    assert capabilities.exact_started
    assert capabilities.sample_counts

    class _SpeakOnlySherpa(SherpaOnnxEngine):
        def speak(self, text, on_done=None):
            super().speak(text, on_done)

    assert not _SpeakOnlySherpa(SherpaConfig()).playback_capabilities.tracked_terminal


def test_sherpa_missing_tts_fails_tracked_request_without_start():
    engine = SherpaOnnxEngine(SherpaConfig())
    probe = _ReceiptProbe()

    engine.speak_tracked(
        TrackedSpeech("missing-tts", "cannot synthesize"),
        on_started=probe.on_started,
        on_terminal=probe.on_terminal,
    )

    started, terminal = probe.snapshot()
    assert started == []
    assert len(terminal) == 1
    assert terminal[0][0].outcome is PlaybackOutcome.FAILED
    assert terminal[0][0].played_samples is None
    assert terminal[0][0].safe_text_prefix == ""


def test_sherpa_tracked_request_before_start_is_dropped_immediately():
    engine = _engine(_StreamingTts())
    probe = _ReceiptProbe()

    engine.speak_tracked(
        TrackedSpeech("pre-start", "too early"),
        on_started=probe.on_started,
        on_terminal=probe.on_terminal,
    )

    started, terminal = probe.snapshot()
    assert started == []
    assert len(terminal) == 1
    assert terminal[0][0].outcome is PlaybackOutcome.DROPPED
    assert engine._play_q.empty()


def test_sherpa_queue_eviction_and_stop_drop_every_unclaimed_request():
    engine = _engine(_StreamingTts())
    engine._play_q.maxsize = 1
    engine._running.set()
    engine._start_receipt_dispatcher()
    probe = _ReceiptProbe()
    for fragment_id in ("oldest", "newest"):
        engine.speak_tracked(
            TrackedSpeech(fragment_id, fragment_id),
            on_started=probe.on_started,
            on_terminal=probe.on_terminal,
        )

    assert _wait_until(
        lambda: [entry[0].fragment_id for entry in probe.snapshot()[1]]
        == ["oldest"]
    )
    engine.stop()

    started, terminal = probe.snapshot()
    assert started == []
    assert [entry[0].fragment_id for entry in terminal] == ["oldest", "newest"]
    assert all(entry[0].outcome is PlaybackOutcome.DROPPED for entry in terminal)


def test_sherpa_claimed_pre_fifo_stop_is_interrupted_not_orphaned():
    engine = _engine(_StreamingTts())
    engine._running.set()
    engine._start_receipt_dispatcher()
    probe = _ReceiptProbe()
    engine.speak_tracked(
        TrackedSpeech("claimed", "claimed before device open"),
        on_started=probe.on_started,
        on_terminal=probe.on_terminal,
    )
    item = engine._play_q.get_nowait()
    ticket = item[4]
    with engine._receipt_lock:
        assert engine._claim_utterance(item[2]) == item[2]
        ticket.claimed = True

    try:
        engine.stop_speaking()
        assert _wait_until(lambda: len(probe.snapshot()[1]) == 1)
        started, terminal = probe.snapshot()
        assert started == []
        assert terminal[0][0].outcome is PlaybackOutcome.INTERRUPTED
    finally:
        engine.stop()


def test_sherpa_shutdown_dispatches_partial_receipt_before_return(monkeypatch):
    engine = _engine(_FixedTts(200))
    probe = _ReceiptProbe()
    holder = _start_playback_harness(monkeypatch, engine)
    engine.speak_tracked(
        TrackedSpeech("shutdown-partial", "partial at shutdown"),
        on_started=probe.on_started,
        on_terminal=probe.on_terminal,
    )
    assert _wait_until(lambda: "stream" in holder and engine._fifo.count() == 200)
    holder["stream"].pull(20)
    assert _wait_until(lambda: len(probe.snapshot()[0]) == 1)

    engine.stop()

    started, terminal = probe.snapshot()
    assert len(started) == 1
    assert len(terminal) == 1
    receipt = terminal[0][0]
    assert receipt.outcome is PlaybackOutcome.INTERRUPTED
    assert receipt.played_samples == 20
    assert receipt.total_samples == 200
    assert receipt.safe_text_prefix == ""


def test_sherpa_output_open_failure_terminalizes_current_request(monkeypatch):
    def fail_output(*_args, **_kwargs):
        raise RuntimeError("scripted output-open failure")

    fake_sd = SimpleNamespace(
        OutputStream=fail_output,
        query_devices=lambda *_args, **_kwargs: {"default_samplerate": 16000},
        check_output_settings=lambda **_kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    engine = _engine(_FixedTts(20))
    probe = _ReceiptProbe()
    engine._running.set()
    engine._start_receipt_dispatcher()
    engine._play_thread = threading.Thread(target=engine._playback_loop, daemon=True)
    engine._play_thread.start()
    engine.speak_tracked(
        TrackedSpeech("open-failure", "will fail"),
        on_started=probe.on_started,
        on_terminal=probe.on_terminal,
    )
    try:
        assert _wait_until(lambda: len(probe.snapshot()[1]) == 1)
        started, terminal = probe.snapshot()
        assert started == []
        assert terminal[0][0].outcome is PlaybackOutcome.FAILED
        assert terminal[0][0].safe_text_prefix == ""
    finally:
        engine.stop()


def test_sherpa_receipt_callback_failure_does_not_kill_later_callbacks(monkeypatch):
    engine = _engine(_StreamingTts())
    holder = _start_playback_harness(monkeypatch, engine, default_sr=22050)
    events = []
    lock = threading.Lock()

    def started(fragment_id):
        with lock:
            events.append(("started", fragment_id))

    def first_terminal(receipt):
        with lock:
            events.append(("terminal", receipt.fragment_id))
        raise RuntimeError("intentional callback failure")

    def second_terminal(receipt):
        with lock:
            events.append(("terminal", receipt.fragment_id))

    engine.speak_tracked(
        TrackedSpeech("first", "first"),
        on_started=started,
        on_terminal=first_terminal,
    )
    engine.speak_tracked(
        TrackedSpeech("second", "second"),
        on_started=started,
        on_terminal=second_terminal,
    )
    try:
        assert _wait_until(lambda: "stream" in holder and engine._fifo.count() == 6)
        holder["stream"].pull(6)
        assert _wait_until(
            lambda: events.count(("terminal", "second")) == 1
        )
        assert events == [
            ("started", "first"),
            ("terminal", "first"),
            ("started", "second"),
            ("terminal", "second"),
        ]
    finally:
        engine.stop()


def test_sherpa_tracked_completion_attests_sanitized_markup(monkeypatch):
    engine = SherpaOnnxEngine(
        SherpaConfig(
            tts_markup=True,
            tts_speaker_voices={"warm": 0},
            tts_dc_block=False,
        )
    )
    engine._tts = _StreamingTts()
    probe = _ReceiptProbe()
    holder = _start_playback_harness(monkeypatch, engine, default_sr=22050)
    engine.speak_tracked(
        TrackedSpeech("markup", "[voice:warm] Hello there."),
        on_started=probe.on_started,
        on_terminal=probe.on_terminal,
    )
    try:
        assert _wait_until(lambda: "stream" in holder and engine._fifo.count() == 3)
        holder["stream"].pull(3)
        assert _wait_until(lambda: len(probe.snapshot()[1]) == 1)
        receipt = probe.snapshot()[1][0][0]
        assert receipt.outcome is PlaybackOutcome.COMPLETED
        assert receipt.safe_text_prefix == "Hello there."
    finally:
        engine.stop()


def test_sherpa_tracked_counts_use_resampled_output_domain(monkeypatch):
    engine = _engine(_FixedTts(10))
    probe = _ReceiptProbe()
    holder = _start_playback_harness(
        monkeypatch,
        engine,
        default_sr=48000,
        supported_rates={48000},
    )
    engine.speak_tracked(
        TrackedSpeech("resampled", "resampled output"),
        on_started=probe.on_started,
        on_terminal=probe.on_terminal,
    )
    try:
        assert _wait_until(lambda: "stream" in holder and engine._fifo.count() == 30)
        holder["stream"].pull(30)
        assert _wait_until(lambda: len(probe.snapshot()[1]) == 1)
        receipt = probe.snapshot()[1][0][0]
        assert receipt.outcome is PlaybackOutcome.COMPLETED
        assert receipt.played_samples == receipt.total_samples == 30
        assert receipt.output_sample_rate == 48000
    finally:
        engine.stop()


def test_sherpa_tracked_receipts_survive_idle_fifo_reopen(monkeypatch):
    engine = SherpaOnnxEngine(
        SherpaConfig(
            release_output_when_idle=True,
            tts_dc_block=False,
        )
    )
    engine._tts = _StreamingTts()
    probe = _ReceiptProbe()
    holder = _start_playback_harness(monkeypatch, engine, default_sr=22050)
    try:
        for index in (1, 2):
            fragment_id = f"reopen-{index}"
            engine.speak_tracked(
                TrackedSpeech(fragment_id, fragment_id),
                on_started=probe.on_started,
                on_terminal=probe.on_terminal,
            )
            assert _wait_until(
                lambda: len(holder["streams"]) == index
                and engine._fifo is not None
                and engine._fifo.count() == 3
            )
            holder["streams"][index - 1].pull(3)
            assert _wait_until(lambda: len(probe.snapshot()[1]) == index)
            assert _wait_until(lambda: engine._fifo is None)

        assert [entry[0].fragment_id for entry in probe.snapshot()[1]] == [
            "reopen-1",
            "reopen-2",
        ]
        assert all(
            entry[0].outcome is PlaybackOutcome.COMPLETED
            for entry in probe.snapshot()[1]
        )
    finally:
        engine.stop()


def test_sherpa_wedged_synth_cannot_orphan_stalled_barge_fade(monkeypatch):
    tts = _WedgedStreamingTts(200)
    engine = _engine(tts)
    probe = _ReceiptProbe()
    holder = _start_playback_harness(monkeypatch, engine)
    engine.speak_tracked(
        TrackedSpeech("wedged-cut", "wedged after first chunk"),
        on_started=probe.on_started,
        on_terminal=probe.on_terminal,
    )
    try:
        assert tts.entered.wait(timeout=1.0)
        assert _wait_until(lambda: engine._fifo.count() == 200)
        holder["stream"].pull(50)
        assert _wait_until(lambda: len(probe.snapshot()[0]) == 1)

        engine.stop_speaking()

        # Do not pull the retained fade: its independent lease timer must expire
        # even though the playback worker remains wedged inside native TTS.
        assert _wait_until(lambda: len(probe.snapshot()[1]) == 1, timeout=0.5)
        assert engine._play_thread.is_alive()
        receipt = probe.snapshot()[1][0][0]
        assert receipt.outcome is PlaybackOutcome.INTERRUPTED
        assert receipt.played_samples == 50
        assert receipt.total_samples == 200
    finally:
        tts.release.set()
        engine.stop()


def test_sherpa_capture_liveness_loss_fails_fifo_bound_receipt(monkeypatch):
    engine = _engine(_FixedTts(200))
    probe = _ReceiptProbe()
    holder = _start_playback_harness(monkeypatch, engine)
    engine.speak_tracked(
        TrackedSpeech("capture-fatal", "capture failed during playback"),
        on_started=probe.on_started,
        on_terminal=probe.on_terminal,
    )
    try:
        assert _wait_until(lambda: "stream" in holder and engine._fifo.count() == 200)

        # Capture fatal paths share this liveness flag with playback. The worker
        # must fail—not abandon—the bound FIFO ownership on its normal loop exit.
        engine._running.clear()
        engine._play_thread.join(timeout=1.0)

        assert not engine._play_thread.is_alive()
        assert _wait_until(lambda: len(probe.snapshot()[1]) == 1)
        started, terminal = probe.snapshot()
        assert started == []
        receipt = terminal[0][0]
        assert receipt.outcome is PlaybackOutcome.FAILED
        assert receipt.played_samples == 0
        assert receipt.total_samples == 200
    finally:
        engine.stop()


def test_sherpa_drain_timeout_fails_exact_fifo_accounting(monkeypatch):
    engine = _engine(_FixedTts(200))
    probe = _ReceiptProbe()
    holder = _start_playback_harness(monkeypatch, engine)
    engine.speak_tracked(
        TrackedSpeech("drain-timeout", "stalled output callback"),
        on_started=probe.on_started,
        on_terminal=probe.on_terminal,
    )
    try:
        assert _wait_until(lambda: "stream" in holder and engine._fifo.count() == 200)
        assert engine._wait_for_playback_drain(timeout_sec=0.0) is False
        assert _wait_until(lambda: len(probe.snapshot()[1]) == 1)
        started, terminal = probe.snapshot()
        assert started == []
        receipt = terminal[0][0]
        assert receipt.outcome is PlaybackOutcome.FAILED
        assert receipt.played_samples == 0
        assert receipt.total_samples == 200
        assert receipt.safe_text_prefix == ""
    finally:
        engine.stop()


def test_sherpa_terminal_callback_can_reenter_stop_and_drain_later_receipts(
    monkeypatch,
):
    engine = _engine(_StreamingTts())
    holder = _start_playback_harness(monkeypatch, engine, default_sr=22050)
    second_terminal = threading.Event()
    stop_returned = threading.Event()
    second_was_done_at_return = []

    def first_done(_receipt):
        engine.stop()
        second_was_done_at_return.append(second_terminal.is_set())
        stop_returned.set()

    engine.speak_tracked(
        TrackedSpeech("reentrant-first", "first"),
        on_terminal=first_done,
    )
    engine.speak_tracked(
        TrackedSpeech("reentrant-second", "second"),
        on_terminal=lambda _receipt: second_terminal.set(),
    )

    assert _wait_until(lambda: "stream" in holder and engine._fifo.count() == 6)
    holder["stream"].pull(6)
    assert stop_returned.wait(timeout=2.0)
    assert second_terminal.is_set()
    assert second_was_done_at_return == [True]


def test_synthesize_streams_chunks_as_produced():
    eng = _engine(_StreamingTts())
    written: list = []
    eng._synthesize("hello there", written.append)
    # Two callback chunks arrived separately (streamed), not one consolidated blob.
    assert len(written) == 2
    assert np.allclose(np.concatenate(written), [0.1, 0.2, 0.3])
    assert eng._tts_can_stream is True


def test_synthesize_stops_early_when_interrupted():
    eng = _engine(_StreamingTts())
    written: list = []
    eng._stop_speaking.set()  # barge-in before synthesis
    eng._synthesize("hello there", written.append)
    # First chunk is handed over, then the callback returns 0 and synthesis stops.
    assert len(written) == 1


def test_synthesize_falls_back_to_chunked_waveform_without_callback():
    eng = _engine(_NonStreamingTts())
    written: list = []
    eng._synthesize("hi", written.append)
    assert eng._tts_can_stream is False  # detected and remembered
    # 4000 samples at 16 kHz => 0.1s (1600-sample) chunks => 3 writes (1600,1600,800).
    assert [len(w) for w in written] == [1600, 1600, 800]
    assert np.concatenate(written).shape[0] == 4000


def test_synthesize_normalizes_then_declicks_when_target_rms_set():
    """With tts_target_rms>0 the non-streaming path runs normalize_rms THEN declick
    on the whole sentence -- the shipped fluidity path, currently untested at the
    engine level (no other test sets tts_target_rms, so they all take the streaming
    branch). A LOUD clip with an impulse spike must come out near the target level
    with the spike repaired -- proving both stages ran in _synthesize."""
    sr = 16000
    t = np.arange(4000) / sr
    loud = (0.4 * np.sqrt(2) * np.sin(2 * np.pi * 220 * t)).astype("float32")  # RMS ~0.4
    loud[2000] = 0.95                                   # an impulse spike on top

    class _LoudTts:
        sample_rate = sr

        def generate(self, text, sid=0, speed=1.0):
            return _GenAudio(loud.copy(), sample_rate=sr)

    eng = SherpaOnnxEngine(
        SherpaConfig(tts_target_rms=0.12, tts_declick=True, tts_declick_threshold=0.22)
    )
    eng._tts = _LoudTts()
    written: list = []
    eng._synthesize("x", written.append)
    out = np.concatenate(written)
    out_rms = float(np.sqrt(np.mean(out.astype("float64") ** 2)))
    assert abs(out_rms - 0.12) < 0.02                  # normalize_rms ran (level steady)
    assert abs(float(out[2000])) < 0.22                # declick repaired the post-gain spike


def test_speak_enqueues_and_stop_speaking_flushes():
    eng = _engine(_StreamingTts())
    eng.speak("one")
    eng.speak("two")
    assert eng._play_q.qsize() == 2
    eng.stop_speaking()
    assert eng._stop_speaking.is_set()
    assert eng._play_q.empty()


def test_speak_without_tts_calls_on_done_immediately():
    eng = SherpaOnnxEngine(SherpaConfig())  # no _tts
    done = []
    eng.speak("hi", on_done=lambda: done.append(1))
    assert done == [1]
    assert eng._play_q.empty()


def test_speak_logs_tts_sanitize_marker(caplog):
    eng = SherpaOnnxEngine(SherpaConfig(tts_markup=True, tts_speaker_voices={"warm": 7}))
    eng._tts = _StreamingTts()
    caplog.set_level(logging.INFO, logger="speaker.sherpa")

    eng.speak("[voice:warm] Hello there.")

    assert "tts sanitize:" in caplog.text
    assert '"raw": "[voice:warm] Hello there."' in caplog.text
    assert '"spoken": "Hello there."' in caplog.text
    assert '"voice": "warm"' in caplog.text


def test_synthesize_logs_resolved_quality_params(caplog):
    eng = SherpaOnnxEngine(
        SherpaConfig(
            tts_target_rms=0.12,
            tts_output_lowpass_hz=7000.0,
            tts_speaker_id=5,
            tts_speed=1.1,
            tts_declick=False,
        )
    )
    eng._tts = _StreamingTts()
    caplog.set_level(logging.INFO, logger="speaker.sherpa")

    eng._synthesize("hello there", [].append)

    assert "tts resolved:" in caplog.text
    assert '"sid": 5' in caplog.text
    assert '"speed": 1.1' in caplog.text
    assert '"lowpass_hz": 7000.0' in caplog.text
    assert '"target_rms": 0.12' in caplog.text


def test_synthesize_logs_audio_quality_whole_clip(caplog):
    """The whole-clip path (forced here by target_rms>0 on the first/uncarried
    sentence) must log a full quality snapshot -- incl. the spectral fields --
    computed on the EXACT final samples about to reach the FIFO. This is the
    trustworthy alternative to the lossy 16 kHz .ref.wav AEC tap (see
    audio_quality_metrics' docstring)."""
    eng = SherpaOnnxEngine(
        SherpaConfig(tts_target_rms=0.12, tts_output_lowpass_hz=7000.0, tts_declick=False)
    )
    eng._tts = _StreamingTts()  # generate() called w/o callback here -> whole-clip
    caplog.set_level(logging.INFO, logger="speaker.sherpa")

    written: list = []
    eng._synthesize("hello there", written.append)

    assert "tts audio quality:" in caplog.text
    assert '"mode": "whole_clip"' in caplog.text
    assert '"n_samples": 3' in caplog.text          # [0.1, 0.2, 0.3] concatenated
    assert '"hf_ratio"' in caplog.text              # spectral fields computed (not null)
    assert '"spectral_flatness"' in caplog.text


def test_synthesize_logs_audio_quality_streaming(caplog):
    """The streaming path logs the cheap scalar subset (rms/peak/clip/dc) from
    running per-chunk accumulators, with the spectral fields explicitly null
    (a per-chunk FFT would need Welch-style aggregation, and buffering the
    whole clip just to measure it would defeat the point of streaming)."""
    eng = SherpaOnnxEngine(SherpaConfig(tts_target_rms=0.0, tts_declick=False))
    eng._tts = _StreamingTts()
    caplog.set_level(logging.INFO, logger="speaker.sherpa")

    written: list = []
    eng._synthesize("hello there", written.append)

    assert "tts audio quality:" in caplog.text
    assert '"mode": "streaming"' in caplog.text
    assert '"hf_ratio": null' in caplog.text
    assert '"spectral_flatness": null' in caplog.text
    assert '"n_samples": 3' in caplog.text          # two chunks, 2 + 1 samples


def test_enqueue_drops_oldest_under_backpressure():
    eng = _engine(_StreamingTts())
    eng._play_q.maxsize = 2
    eng.speak("a")
    eng.speak("b")
    eng.speak("c")  # full -> oldest ("a") dropped, "c" enqueued
    queued = [eng._play_q.get_nowait()[0] for _ in range(eng._play_q.qsize())]
    assert queued == ["b", "c"]


# --- rc-3: per-utterance generation counter (stale-utterance suppression) ---
# A barge-in / stop bumps the generation; a sentence enqueued BEFORE the bump is
# stale and the playback worker skips it instead of clearing _stop_speaking (the
# wipe race) and speaking it. A sentence enqueued AFTER the bump is current and
# plays -- so a fresh reply after a barge is never muted.


def test_speak_gen_bumps_on_each_stop():
    eng = _engine(_StreamingTts())
    g0 = eng._speak_gen
    eng.stop_speaking()
    assert eng._speak_gen == g0 + 1
    eng.stop_speaking()
    assert eng._speak_gen == g0 + 2


def test_enqueue_stamps_current_generation_and_barge_makes_it_stale():
    eng = _engine(_StreamingTts())
    eng.speak("two")
    item = eng._play_q.get_nowait()  # take it out as the worker would, pre-barge
    assert item[0] == "two"
    assert item[2] == eng._speak_gen  # stamped with the generation at enqueue
    eng.stop_speaking()  # barge: bumps the generation (and drains the queue)
    # The sentence we hold was enqueued before the barge -> now stale, so the
    # worker's `item_gen != self._speak_gen` check skips it.
    assert item[2] != eng._speak_gen


def test_new_reply_after_barge_is_current_generation():
    """Mute-bug guard: a sentence enqueued AFTER a barge carries the new
    generation, so the worker plays it (it is NOT treated as stale)."""
    eng = _engine(_StreamingTts())
    eng.speak("old reply")        # generation G
    eng.stop_speaking()           # barge: bump to G+1, drain the queue
    assert eng._play_q.empty()    # the old reply was drained
    eng.speak("brand new reply")  # enqueued at G+1
    item = eng._play_q.get_nowait()
    assert item[0] == "brand new reply"
    assert item[2] == eng._speak_gen  # current generation -> the worker plays it


def test_claim_utterance_skips_stale_without_clearing_stop():
    """The load-bearing rc-3 guard: a stale dequeued sentence is rejected (None)
    and _stop_speaking is NOT wiped, so a pending barge survives."""
    eng = _engine(_StreamingTts())
    eng.speak("x")        # enqueued at generation G
    eng.stop_speaking()   # barge: bumps to G+1, sets _stop_speaking
    assert eng._stop_speaking.is_set()
    assert eng._claim_utterance(0) is None     # the old-generation sentence is stale
    assert eng._stop_speaking.is_set()         # and the barge flag was NOT wiped


def test_claim_utterance_accepts_current_and_clears_stop():
    eng = _engine(_StreamingTts())
    eng._stop_speaking.set()  # a prior barge left the flag set
    g = eng._speak_gen
    assert eng._claim_utterance(g) == g        # current generation -> claimed
    assert not eng._stop_speaking.is_set()     # cleared for this fresh utterance


def test_synthesize_streaming_aborts_on_generation_mismatch():
    eng = _engine(_StreamingTts())
    eng._speak_gen = 7
    written: list = []
    eng._synthesize("hi", written.append, gen=7)  # current gen -> full synthesis
    assert len(written) == 2
    written.clear()
    eng._speak_gen = 8  # a barge bumped the generation past this utterance's gen
    eng._synthesize("hi", written.append, gen=7)
    # First chunk is handed over, then on_chunk returns 0 and synthesis stops --
    # same shape as the _stop_speaking interrupt path.
    assert len(written) == 1


def test_synthesize_nonstreaming_aborts_before_write_on_generation_mismatch():
    eng = _engine(_NonStreamingTts())
    eng._speak_gen = 8
    written: list = []
    eng._synthesize("hi", written.append, gen=7)  # stale -> loop breaks first
    assert written == []
    # A matching generation synthesizes normally.
    eng._speak_gen = 7
    eng._synthesize("hi", written.append, gen=7)
    assert written  # chunks were written


# --- barge-in + shutdown now go through the playback FIFO ---
# The callback-OutputStream rewrite means barge-in no longer abort()s the stream;
# it FLUSHES the playback FIFO (the next PortAudio callback emits silence) while
# the stream stays open. Shutdown stop()/close()s the stream (clean device
# handback; PortAudio joins the callback) after flushing the FIFO. So the fakes
# below model stop()/close() (not abort), and stop_speaking's cut now also
# requires a FIFO + a speaking stream.


class _FakeOutStream:
    def __init__(self):
        self.stopped = 0
        self.closed = 0

    def stop(self):
        self.stopped += 1

    def close(self):
        self.closed += 1


class _FakeFIFO:
    def __init__(self):
        self.flushed = 0
        self.last_fade = None

    def flush(self, fade_samples=0):
        self.flushed += 1
        self.last_fade = fade_samples

    def interrupt_tags(self, _status, fade_samples=0):
        self.flush(fade_samples)

    def count(self):
        return 0


class _StuckFIFO:
    def __init__(self, count=1600):
        self._count = int(count)
        self.flushed = 0

    def flush(self, fade_samples=0):
        self.flushed += 1
        self._count = 0

    def interrupt_tags(self, _status, fade_samples=0):
        self.flush(fade_samples)

    def count(self):
        return self._count


def test_playback_drain_timeout_flushes_before_reopening_asr(caplog):
    # Root-cause guard for full-sentence echo finals: if the output callback stalls,
    # _speaking must not clear while queued assistant audio remains available to
    # play later with ASR open. The timeout path drops that stale tail and emits a
    # metric so the hardware stall is visible in the run bundle.
    eng = _engine(_StreamingTts())
    metrics: list = []
    eng._cb.on_metric = metrics.append
    eng._fifo = _StuckFIFO(count=800)
    eng._running.set()
    caplog.set_level(logging.WARNING, logger="speaker.sherpa")

    drained = eng._wait_for_playback_drain(timeout_sec=0.0)

    assert drained is False
    assert eng._fifo.flushed == 1
    assert eng._fifo.count() == 0
    assert "playback FIFO did not drain before idle deadline" in caplog.text
    assert metrics == ["playback_drain_timeout"]


def test_playback_drain_timeout_expires_stalled_barge_fade_without_metric():
    # A deliberate stop normally leaves a short de-click fade for the callback.
    # If that callback is stalled, the worker must expire the fade after its
    # bounded grace so a receipt cannot remain pending forever. This is still a
    # user interruption, not a healthy-session drain-timeout diagnostic.
    eng = _engine(_StreamingTts())
    metrics: list = []
    eng._cb.on_metric = metrics.append
    eng._fifo = _StuckFIFO(count=800)
    eng._running.set()
    eng._stop_speaking.set()

    drained = eng._wait_for_playback_drain(timeout_sec=0.0)

    assert drained is False
    assert eng._fifo.flushed == 1
    assert metrics == []


def test_stop_speaking_flushes_live_fifo_and_stamps_metric():
    # Barge-in must drop queued audio (flush the FIFO so the next callback emits
    # silence), not just stop feeding it -- and stamp the true audible-stop
    # instant at that moment. Requires a live stream + FIFO + a speaking run.
    eng = _engine(_StreamingTts())
    metrics: list = []
    eng._cb.on_metric = metrics.append
    eng._out_stream = _FakeOutStream()
    eng._fifo = _FakeFIFO()
    eng._speaking.set()
    eng.stop_speaking()
    assert eng._fifo.flushed == 1
    assert "barge_in_stop" in metrics


def test_stop_speaking_without_live_stream_is_silent_noop():
    # Nothing playing -> no flush, no spurious barge_in_stop metric.
    eng = _engine(_StreamingTts())
    metrics: list = []
    eng._cb.on_metric = metrics.append
    eng.stop_speaking()  # _out_stream/_fifo are None, not speaking
    assert "barge_in_stop" not in metrics


def test_stop_speaking_clears_speaking_and_relatches_on_the_cut():
    # RC-2 regression guard: stop_speaking() must AUTHORITATIVELY end the speaking
    # state itself, not wait on the playback worker's epilogue. If the native
    # tts.generate() wedges, the worker never clears _speaking and the capture
    # loop stays deaf to ASR for the rest of the session. So the cut path must
    # clear _speaking + re-arm the one-per-run barge latch on its own. (Here we
    # never run the worker at all -- exactly the wedged case.)
    eng = _engine(_StreamingTts())
    eng._out_stream = _FakeOutStream()
    eng._fifo = _FakeFIFO()
    eng._speaking.set()                   # worker set it; worker will NOT clear it (wedged)
    eng._barge_in_fired_this_run = True   # a barge already fired this run
    eng.stop_speaking()
    assert eng._fifo.flushed == 1
    assert eng.is_speaking is False              # ASR re-enables without the worker returning
    assert eng._barge_in_fired_this_run is False  # barge-in re-armed for the next interrupt


def test_barge_watch_is_not_armed_before_first_audio_plays():
    # A reply is marked _speaking before TTS has produced its first audible block.
    # During that synth lead-in there is no playback reference yet, so VAD noise
    # must not accumulate as a rejected/detected "during playback" barge episode.
    eng = _engine(_StreamingTts())
    eng._speaking.set()
    eng._first_audio_pending = True
    eng._playback_level = 0.0

    assert eng._barge_watch_active() is False


def test_barge_watch_stays_armed_between_queued_sentences_while_tail_is_audible():
    # For adjacent queued sentences the next utterance can set _first_audio_pending
    # while the previous sentence's FIFO tail is still audibly playing. Keep the
    # watch armed then so real talk-over in the inter-sentence gap is not missed.
    eng = _engine(_StreamingTts())
    eng._speaking.set()
    eng._first_audio_pending = True
    eng._playback_level = 0.02
    eng._last_playback_at = time.monotonic()

    assert eng._barge_watch_active() is True


def test_barge_watch_treats_stale_tail_level_as_ref_empty():
    # _playback_level is an EWMA and can remain nonzero after the FIFO has drained.
    # Do not let that stale value arm ref-dependent barge logic during a long
    # queued-sentence synth lead-in.
    eng = _engine(_StreamingTts())
    eng._speaking.set()
    eng._first_audio_pending = True
    eng._playback_level = 0.02
    eng._last_playback_at = time.monotonic() - 1.0

    assert eng._barge_watch_active() is False


# --- clean shutdown: stop() must tear the live stream down so a wedged play ---
# thread can exit. On a dead device the play thread can block in FIFO.write();
# the queue sentinel can't wake it, so stop() would hang on the join. stop() now
# flushes the FIFO (releasing a blocked producer) and stop()/close()s the live
# stream first (a clean handback; PortAudio joins the callback) so the daemon
# play thread exits.


def test_stop_flushes_and_closes_live_stream_before_join():
    # With a live stream + FIFO set, stop() flushes the FIFO and stop()/close()s
    # the stream (NOT abort) but emits NO barge_in_stop metric -- this is
    # teardown, not an interruption.
    eng = _engine(_StreamingTts())
    metrics: list = []
    eng._cb.on_metric = metrics.append
    eng._out_stream = _FakeOutStream()
    eng._fifo = _FakeFIFO()
    eng._running.set()
    eng.stop()  # no capture/play threads started -> joins are no-ops
    assert eng._out_stream.stopped == 1 and eng._out_stream.closed == 1
    assert eng._fifo.flushed == 1
    assert "barge_in_stop" not in metrics  # teardown, not a barge-in
    assert not eng._running.is_set()
    assert eng._stop_speaking.is_set()


def test_stop_joins_capture_before_closing_input_without_read_overlap():
    """A cooperative block returns before PortAudio is physically closed."""
    events = []
    read_started = threading.Event()
    release_read = threading.Event()

    class _Input:
        def __init__(self):
            self.read_active = False
            self.close_overlapped = False

        def read(self):
            self.read_active = True
            events.append("read-start")
            read_started.set()
            assert release_read.wait(timeout=2.0)
            self.read_active = False
            events.append("read-exit")

        def request_close(self):
            events.append("input-request-close")
            release_read.set()

        def close(self):
            self.close_overlapped = self.read_active
            events.append("input-close")

    stream = _Input()
    eng = _engine(_StreamingTts())
    eng.config.block_sec = 0.02
    eng._stream_in = stream
    eng._running.set()
    eng._capture_thread = threading.Thread(target=stream.read, daemon=True)
    eng._capture_thread.start()
    assert read_started.wait(timeout=1.0)

    eng.stop()

    assert events == [
        "read-start",
        "input-request-close",
        "read-exit",
        "input-close",
    ]
    assert stream.close_overlapped is False
    assert not eng._capture_thread.is_alive()
    assert eng._stream_in is None


def test_stop_aborts_stuck_capture_then_closes_without_read_overlap():
    """Abort unblocks the owner; physical close follows its bounded join."""
    read_started = threading.Event()
    forced = threading.Event()

    class _Input:
        def __init__(self):
            self.read_active = False
            self.abort_overlapped = False
            self.close_overlapped = False
            self.close_calls = 0

        def read(self):
            self.read_active = True
            read_started.set()
            assert forced.wait(timeout=2.0)
            self.read_active = False

        def request_close(self):
            pass  # emulate a native host read that ignores logical shutdown

        def close(self):
            self.close_overlapped = self.read_active
            self.close_calls += 1
            return True

        def abort_read(self, *, timeout):
            assert timeout > 0.0
            self.abort_overlapped = self.read_active
            forced.set()
            return True

    stream = _Input()
    eng = _engine(_StreamingTts())
    eng.config.block_sec = 0.01
    eng._stream_in = stream
    eng._running.set()
    eng._capture_thread = threading.Thread(target=stream.read, daemon=True)
    eng._capture_thread.start()
    assert read_started.wait(timeout=1.0)

    started = time.monotonic()
    eng.stop()
    elapsed = time.monotonic() - started

    assert stream.abort_overlapped is True
    assert stream.close_overlapped is False
    assert stream.close_calls == 1
    assert elapsed < 0.75
    assert not eng._capture_thread.is_alive()
    assert eng._stream_in is None


def test_stop_retains_truly_stuck_input_without_unbounded_close():
    """If abort cannot quiesce, bounded stop retains rather than close-racing."""
    class _Input:
        def __init__(self):
            self.close_calls = 0
            self.abort_calls = 0

        def request_close(self):
            pass

        def abort_read(self, *, timeout):
            assert timeout > 0.0
            self.abort_calls += 1
            return False

        def close(self):
            self.close_calls += 1
            raise AssertionError("must not close while capture remains active")

    class _StuckThread:
        def __init__(self):
            self.join_timeouts = []

        def join(self, timeout):
            self.join_timeouts.append(timeout)

        def is_alive(self):
            return True

    stream = _Input()
    thread = _StuckThread()
    eng = _engine(_StreamingTts())
    eng.config.block_sec = 0.01
    eng._stream_in = stream
    eng._capture_thread = thread
    eng._running.set()

    started = time.monotonic()
    eng.stop()
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    assert stream.abort_calls == 1
    assert stream.close_calls == 0
    assert thread.join_timeouts == [pytest.approx(0.06), 1.0]
    assert eng._stream_in is stream


@pytest.mark.parametrize(
    ("old_thread_alive", "message"),
    [
        (True, "previous capture thread is still alive"),
        (False, "previous input stream was retained"),
    ],
)
def test_retained_capture_ownership_fails_restart_before_clearing_fences(
    old_thread_alive, message
):
    """A later start cannot resurrect the retained native capture owner."""
    class _RetainedInput:
        pass

    class _StuckThread:
        def is_alive(self):
            return old_thread_alive

    retained = _RetainedInput()
    old_thread = _StuckThread()
    eng = _engine(_StreamingTts())
    eng._input_agc = InputAGC()
    eng._stream_in = retained
    eng._capture_thread = old_thread
    eng._capture_stopping.set()
    eng._playback_stopping.set()
    eng._input_agc.gain = 9.0

    with pytest.raises(RuntimeError, match=message):
        eng.start(EngineCallbacks())

    assert eng._capture_stopping.is_set()
    assert eng._playback_stopping.is_set()
    assert not eng._running.is_set()
    assert eng._stream_in is retained
    assert eng._capture_thread is old_thread
    assert eng._input_agc.gain == 9.0


def test_start_resets_input_agc_only_after_prior_owner_guards(monkeypatch):
    eng = SherpaOnnxEngine(SherpaConfig(tts_dc_block=False, input_agc=True))
    eng._input_agc.gain = 9.0
    eng._input_agc.process(np.full(1600, 0.001, dtype="float32"))
    assert eng._input_agc.last_input_rms > 0.0

    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace())

    def stop_after_reset():
        raise RuntimeError("scripted build stop")

    monkeypatch.setattr(eng, "_build", stop_after_reset)

    with pytest.raises(RuntimeError, match="scripted build stop"):
        eng.start(EngineCallbacks())

    assert eng._input_agc.gain == 1.0
    assert eng._input_agc.last_input_rms == 0.0
    assert eng._input_agc.last_applied_gain == 1.0
    assert eng._input_agc.last_above_floor is False


def test_retained_capture_resource_hold_blocks_restart_after_effect_finishes():
    """Cleanup must collect retained output/recorders before a new run starts."""
    eng = _engine(_StreamingTts())
    eng._capture_resource_hold.set()
    eng._capture_stopping.set()
    eng._playback_stopping.set()

    with pytest.raises(RuntimeError, match="capture resources remain retained"):
        eng.start(EngineCallbacks())

    assert eng._capture_resource_hold.is_set()
    assert eng._capture_stopping.is_set()
    assert eng._playback_stopping.is_set()


@pytest.mark.parametrize(
    ("worker_attr", "message"),
    [
        ("_play_thread", "previous playback worker is still alive"),
        ("_final_thread", "previous final worker is still alive"),
        ("_receipt_thread", "previous receipt worker is still alive"),
    ],
)
def test_live_shared_worker_fails_restart_before_shared_events_are_reset(
    worker_attr, message
):
    """A slow prior worker cannot be resurrected by the next run's Event.set()."""
    started = threading.Event()
    release = threading.Event()

    def _slow_worker():
        started.set()
        assert release.wait(timeout=2.0)

    worker = threading.Thread(target=_slow_worker, daemon=True)
    worker.start()
    assert started.wait(timeout=1.0)
    eng = _engine(_StreamingTts())
    setattr(eng, worker_attr, worker)
    eng._capture_stopping.set()
    eng._playback_stopping.set()
    prior_play_q = eng._play_q
    prior_final_q = eng._final_q

    try:
        with pytest.raises(RuntimeError, match=message):
            eng.start(EngineCallbacks())

        assert eng._capture_stopping.is_set()
        assert eng._playback_stopping.is_set()
        assert not eng._running.is_set()
        assert not eng._receipt_running.is_set()
        assert eng._play_q is prior_play_q
        assert eng._final_q is prior_final_q
        assert getattr(eng, worker_attr) is worker
    finally:
        release.set()
        worker.join(timeout=1.0)


def test_timed_out_receipt_dispatcher_reference_is_retained_for_restart_guard():
    """A wedged receipt callback remains discoverable after bounded stop."""
    class _StuckReceiptThread:
        def __init__(self):
            self.join_timeouts = []

        def join(self, timeout):
            self.join_timeouts.append(timeout)

        def is_alive(self):
            return True

    worker = _StuckReceiptThread()
    eng = _engine(_StreamingTts())
    eng._receipt_thread = worker
    eng._receipt_running.set()
    eng._capture_stopping.set()
    eng._playback_stopping.set()

    eng._stop_receipt_dispatcher()

    assert worker.join_timeouts == [0.5]
    assert eng._receipt_thread is worker
    assert not eng._receipt_running.is_set()
    with pytest.raises(RuntimeError, match="previous receipt worker is still alive"):
        eng.start(EngineCallbacks())
    assert eng._capture_stopping.is_set()
    assert eng._playback_stopping.is_set()


def test_stop_retains_input_until_timed_out_abort_call_itself_finishes():
    """Capture may join first; an in-flight native abort still fences close."""
    from core.engines._recovering_input import OpenAttempt, _RecoveringInputStream

    read_started = threading.Event()
    abort_started = threading.Event()
    allow_abort = threading.Event()
    abort_done = threading.Event()
    release_read = threading.Event()

    class _NativeInput:
        def __init__(self):
            self.read_active = False
            self.abort_active = False
            self.close_calls = 0
            self.close_overlapped = False

        def start(self):
            pass

        def read(self, frames):
            self.read_active = True
            read_started.set()
            assert release_read.wait(timeout=2.0)
            self.read_active = False
            return np.zeros((frames, 1), dtype="float32"), False

        def abort(self):
            self.abort_active = True
            abort_started.set()
            assert allow_abort.wait(timeout=2.0)
            self.abort_active = False
            abort_done.set()

        def stop(self):
            pass

        def close(self):
            self.close_overlapped = self.read_active or self.abort_active
            self.close_calls += 1

    native = _NativeInput()
    wrapper = _RecoveringInputStream(
        [OpenAttempt(device=None, samplerate=16000)],
        opener=lambda _device, _samplerate: native,
        block_seconds=0.01,
    )
    wrapper.open()
    capture = threading.Thread(target=lambda: wrapper.read(160), daemon=True)
    capture.start()
    assert read_started.wait(timeout=1.0)

    def _release_read_during_abort():
        assert abort_started.wait(timeout=1.0)
        release_read.set()

    threading.Thread(target=_release_read_during_abort, daemon=True).start()
    eng = _engine(_StreamingTts())
    eng.config.block_sec = 0.01
    eng._stream_in = wrapper
    eng._capture_thread = capture
    eng._running.set()

    started = time.monotonic()
    eng.stop()
    elapsed = time.monotonic() - started

    assert elapsed < 0.75
    assert not capture.is_alive()
    assert native.close_calls == 0
    assert eng._stream_in is wrapper

    allow_abort.set()
    assert abort_done.wait(timeout=1.0)
    assert wrapper.close(wait_timeout=0.0)
    assert native.close_calls == 1
    assert native.close_overlapped is False


@pytest.mark.parametrize("blocked_method", ["stop", "close"])
def test_engine_stop_is_bounded_and_retains_stuck_native_teardown(
    blocked_method, monkeypatch
):
    """A wedged native stop/close helper cannot hang engine stop or detach."""
    from core.engines._recovering_input import OpenAttempt, _RecoveringInputStream
    from core.engines import sherpa as sherpa_module

    monkeypatch.setattr(sherpa_module, "_INPUT_TEARDOWN_TIMEOUT_SEC", 0.02)

    teardown_entered = threading.Event()
    release_teardown = threading.Event()
    teardown_done = threading.Event()

    class _NativeInput:
        def __init__(self):
            self.stop_calls = 0
            self.close_calls = 0
            self.read_active = False
            self.abort_active = False
            self.overlap = False

        def start(self):
            pass

        def stop(self):
            self.stop_calls += 1
            self.overlap |= self.read_active or self.abort_active
            if blocked_method == "stop":
                teardown_entered.set()
                assert release_teardown.wait(timeout=2.0)

        def close(self):
            self.close_calls += 1
            self.overlap |= self.read_active or self.abort_active
            if blocked_method == "close":
                teardown_entered.set()
                assert release_teardown.wait(timeout=2.0)
            teardown_done.set()

    native = _NativeInput()
    wrapper = _RecoveringInputStream(
        [OpenAttempt(device=None, samplerate=16000)],
        opener=lambda _device, _samplerate: native,
        block_seconds=0.02,
    )
    wrapper.open()
    eng = _engine(_StreamingTts())
    eng.config.block_sec = 0.02
    eng._stream_in = wrapper
    eng._running.set()

    started = time.monotonic()
    eng.stop()
    elapsed = time.monotonic() - started

    assert teardown_entered.is_set()
    assert elapsed < 0.5
    assert eng._stream_in is wrapper
    assert wrapper._stream is native  # noqa: SLF001 - retained until helper succeeds
    assert native.overlap is False

    release_teardown.set()
    assert teardown_done.wait(timeout=1.0)
    eng.stop()  # collect the successfully completed retained wrapper
    assert eng._stream_in is None
    assert native.stop_calls == 1
    assert native.close_calls == 1


def test_stop_without_live_stream_is_noop():
    # No live stream -> stop() must not raise and must still tear down cleanly.
    eng = _engine(_StreamingTts())
    eng._running.set()
    eng.stop()  # _out_stream is None
    assert not eng._running.is_set()


def test_stop_returns_promptly_when_producer_is_blocked():
    # Simulate the dead-device hang: a play thread blocked in FIFO.write() until
    # the abort predicate flips. stop() sets _stop_speaking + clears _running
    # (the should_abort predicate) and flushes the FIFO (notify_all), both of
    # which release the blocked producer, then joins within its bounded timeout
    # instead of hanging forever.
    import threading

    eng = _engine(_StreamingTts())
    fifo = PlaybackFIFO(4)  # tiny so the producer blocks immediately when full
    eng._fifo = fifo
    eng._out_stream = _FakeOutStream()
    eng._running.set()
    eng._stop_speaking.clear()

    def fake_play():
        # Block in FIFO.write() like the real producer; the should_abort
        # predicate (set by stop()) releases it and it returns cleanly.
        fifo.write(
            np.ones(100, dtype="float32"),  # >> capacity -> blocks until aborted
            should_abort=lambda: eng._stop_speaking.is_set() or not eng._running.is_set(),
        )

    t = threading.Thread(target=fake_play, daemon=True)
    eng._play_thread = t
    t.start()

    done = threading.Event()

    def call_stop():
        eng.stop()
        done.set()

    threading.Thread(target=call_stop, daemon=True).start()
    # stop() bounds each join at 1.0s; with the abort predicate unblocking the
    # producer it should finish well under that.
    assert done.wait(timeout=3.0), "stop() hung instead of returning promptly"
    assert not t.is_alive()
    # stop() flushes the FIFO but never nulls it (the worker loop owns that, on
    # idle-release/teardown), so the same instance survives and was emptied.
    assert eng._fifo is fifo
    assert eng._fifo.count() == 0


# --- _audio_cb: the core of the callback rewrite (drain + tee + first-audio) ---
# _audio_cb is the SINGLE place the far-end AEC ref / coherence ref / level EWMA
# are teed and TTS_FIRST_AUDIO is stamped -- all off TRUE playback, which is the
# whole point of the rewrite. It is pure Python (no device, no models), so we
# drive it directly. Setting _play_sr == config.sample_rate keeps the far-ring
# tee resample-free so we can assert exact samples; _echo_coherence stays None
# (the callback guards on it) to keep the test scipy-free.


def _outbuf(frames):
    # -1.0 sentinel so any cell the callback fails to write is caught.
    return np.full((frames, 1), -1.0, dtype="float32")


def test_audio_cb_drains_fifo_zero_fills_underrun_and_tees_only_real_samples():
    eng = _engine(_StreamingTts())
    eng._play_sr = eng.config.sample_rate  # no resample -> far ring gets exact samples
    eng._far_ref = FarEndRing()
    eng._echo_coherence = None
    eng._fifo = PlaybackFIFO(8)
    eng._fifo.write(np.array([0.5, 0.25], dtype="float32"), lambda: False)  # float32-exact
    metrics: list = []
    eng._cb.on_metric = metrics.append
    eng._first_audio_pending = True

    out = _outbuf(5)
    eng._audio_cb(out, 5, None, None)

    # The 2 queued samples play; the 3-sample underrun tail is silent zero-fill.
    np.testing.assert_array_equal(out[:, 0], [0.5, 0.25, 0.0, 0.0, 0.0])
    # ONLY the 2 real samples are teed into the far ring -- NOT the zero tail.
    # (A regression to teeing `view`/`view[:frames]` would push silence into the
    # AEC far-end and stay green without this assertion.)
    assert eng._far_ref._written == 2
    np.testing.assert_array_equal(eng._far_ref.read(2, 0), [0.5, 0.25])
    # TTS_FIRST_AUDIO stamped exactly once, at the first real-audio block.
    assert metrics == [TTS_FIRST_AUDIO]
    assert eng._first_audio_pending is False
    assert eng._last_playback_at > 0.0


def test_audio_cb_does_not_stamp_first_audio_on_an_underrun_only_block():
    eng = _engine(_StreamingTts())
    eng._play_sr = eng.config.sample_rate
    eng._far_ref = FarEndRing()
    eng._fifo = PlaybackFIFO(8)  # empty -> pure underrun, n == 0
    metrics: list = []
    eng._cb.on_metric = metrics.append
    eng._first_audio_pending = True  # armed...

    out = _outbuf(4)
    eng._audio_cb(out, 4, None, None)

    np.testing.assert_array_equal(out[:, 0], [0.0, 0.0, 0.0, 0.0])  # all silence
    assert eng._far_ref._written == 0  # nothing teed
    assert metrics == []  # ...no real audio played -> no stamp
    assert eng._first_audio_pending is True  # stays armed for the first real block
    assert eng._last_playback_at == 0.0


def test_audio_cb_with_no_fifo_emits_silence_and_does_not_tee_or_stamp():
    eng = _engine(_StreamingTts())
    eng._play_sr = eng.config.sample_rate
    eng._far_ref = FarEndRing()
    eng._fifo = None  # before the first utterance / after teardown
    metrics: list = []
    eng._cb.on_metric = metrics.append
    eng._first_audio_pending = True

    out = _outbuf(3)
    eng._audio_cb(out, 3, None, None)

    np.testing.assert_array_equal(out[:, 0], [0.0, 0.0, 0.0])
    assert eng._far_ref._written == 0
    assert metrics == []
    assert eng._first_audio_pending is True
    assert eng._last_playback_at == 0.0


# --- Streaming normalize_rms + low-pass path --------------------------------


class _TrackingTts:
    """Streaming TTS stub that records whether _synthesize used callback=."""

    sample_rate = 16000

    def __init__(self, samples):
        self.samples = np.asarray(samples, dtype="float32").reshape(-1)
        self.callback_used: list[bool] = []

    def generate(self, text, sid=0, speed=1.0, callback=None):
        self.callback_used.append(callback is not None)
        samples = self.samples.copy()
        if callback is not None:
            for chunk in np.array_split(samples, 4):
                if callback(chunk, 1.0) == 0:
                    break
        return _GenAudio(samples, sample_rate=self.sample_rate)


def _sine(sr=16000, dur_s=0.25, rms=0.4, freq=300.0):
    t = np.arange(int(sr * dur_s), dtype="float64") / sr
    return (rms * np.sqrt(2.0) * np.sin(2.0 * np.pi * freq * t)).astype("float32")


def _tone_mag(samples, sr: int, freq: float) -> float:
    x = np.asarray(samples, dtype="float64").reshape(-1)
    t = np.arange(x.size, dtype="float64") / float(sr)
    s = np.sin(2.0 * np.pi * float(freq) * t)
    c = np.cos(2.0 * np.pi * float(freq) * t)
    return float((2.0 / x.size) * np.hypot(np.dot(x, s), np.dot(x, c)))


def test_target_rms_wholeclip_on_first_sentence_sets_carry():
    """First sentence with tts_target_rms>0: whole-clip path runs (callback not
    used) and _tts_normalize_gain is set so the next sentence can stream."""
    samp = _sine()
    tts = _TrackingTts(samp)
    eng = SherpaOnnxEngine(SherpaConfig(tts_target_rms=0.12, tts_declick=False))
    eng._tts = tts
    assert eng._tts_normalize_gain is None  # not yet set
    written: list = []
    eng._synthesize("hello", written.append)
    assert tts.callback_used == [False]         # whole-clip path (no callback)
    assert eng._tts_normalize_gain is not None  # carry established
    assert eng._tts_normalize_gain > 0.0


def test_target_rms_streaming_on_second_sentence_applies_carry():
    """Second sentence with tts_target_rms>0: streaming callback is used, the
    carried gain is applied, and _tts_normalize_gain is updated afterward."""
    samp = _sine(rms=0.4)
    tts = _TrackingTts(samp)
    eng = SherpaOnnxEngine(SherpaConfig(tts_target_rms=0.12, tts_declick=False))
    eng._tts = tts
    # Synthesize first sentence (whole-clip) to establish the carry.
    eng._synthesize("first", [].append)
    assert eng._tts_normalize_gain is not None
    gain_after_first = eng._tts_normalize_gain
    # Second sentence: streaming path should be taken.
    written2: list = []
    eng._synthesize("second", written2.append)
    assert tts.callback_used == [False, True]  # streaming path taken
    # Carry is updated from the measured raw RMS.
    assert eng._tts_normalize_gain is not None
    # Output level should be near target_rms (within ±20% tolerance,
    # acknowledging feed-forward may be slightly off if the TTS level varies).
    out = np.concatenate(written2)
    out_rms = float(np.sqrt(np.mean(out.astype("float64") ** 2)))
    assert abs(out_rms - 0.12) < 0.06, f"streaming RMS {out_rms:.3f} far from 0.12"
    _ = gain_after_first  # used for context; the carry updated after second


def test_target_rms_zero_always_takes_streaming_regardless_of_carry():
    """With tts_target_rms=0 the streaming path is taken on the first sentence
    (no carry needed, no normalization). This pins the pure-streaming invariant."""
    samp = _sine()
    tts = _TrackingTts(samp)
    eng = SherpaOnnxEngine(SherpaConfig(tts_target_rms=0.0, tts_declick=False))
    eng._tts = tts
    assert eng._tts_normalize_gain is None  # irrelevant on this path
    eng._synthesize("hi", [].append)
    assert tts.callback_used == [True]      # streaming path regardless
    assert eng._tts_normalize_gain is None  # still None (not updated by this path)


def test_lowpass_enabled_streams_from_first_sentence_when_target_rms_off():
    sr = 16000
    t = np.arange(sr // 2, dtype="float64") / sr
    raw = (
        0.15 * np.sin(2.0 * np.pi * 300.0 * t)
        + 0.30 * np.sin(2.0 * np.pi * 6000.0 * t)
    ).astype("float32")
    tts = _TrackingTts(raw)
    eng = SherpaOnnxEngine(
        SherpaConfig(
            tts_target_rms=0.0,
            tts_output_lowpass_hz=1000.0,
            tts_declick=False,
        )
    )
    eng._tts = tts

    written: list = []
    eng._synthesize("lowpass now streams", written.append)

    assert tts.callback_used == [True]
    assert len(written) == 4
    out = np.concatenate(written)
    assert _tone_mag(out, sr, 6000.0) < 0.08 * _tone_mag(raw, sr, 6000.0)


def test_streaming_lowpass_output_is_peak_bounded_after_filter_overshoot():
    sr = 24000
    t = np.arange(sr, dtype="float64") / sr
    raw = (
        0.40 * np.sin(2.0 * np.pi * 300.0 * t)
        + 0.35 * np.sin(2.0 * np.pi * 7000.0 * t)
    ).astype("float32")
    hot = np.asarray(apply_gain_soft_limit(raw, 4.0), dtype="float32")
    tts = _TrackingTts(hot)
    tts.sample_rate = sr
    eng = SherpaOnnxEngine(
        SherpaConfig(
            tts_target_rms=0.0,
            tts_output_lowpass_hz=7000.0,
            tts_declick=False,
        )
    )
    eng._tts = tts

    written: list = []
    eng._synthesize("lowpass overshoot is bounded", written.append)

    assert tts.callback_used == [True]
    out = np.concatenate(written)
    assert np.all(np.isfinite(out))
    assert float(np.max(np.abs(out))) <= 1.0


def test_target_rms_and_lowpass_stream_on_second_sentence_after_carry_seeded():
    samp = _sine(rms=0.4, freq=300.0)
    tts = _TrackingTts(samp)
    eng = SherpaOnnxEngine(
        SherpaConfig(
            tts_target_rms=0.12,
            tts_output_lowpass_hz=2000.0,
            tts_declick=False,
        )
    )
    eng._tts = tts

    first: list = []
    eng._synthesize("first", first.append)
    assert tts.callback_used == [False]
    assert eng._tts_normalize_gain is not None

    second: list = []
    eng._synthesize("second", second.append)

    assert tts.callback_used == [False, True]
    assert len(second) == 4
    out = np.concatenate(second)
    out_rms = float(np.sqrt(np.mean(out.astype("float64") ** 2)))
    assert out_rms == pytest.approx(0.12, rel=0.15)


def test_output_leveler_still_forces_whole_clip_with_lowpass_enabled():
    tts = _TrackingTts(_sine(rms=0.2, freq=300.0))
    eng = SherpaOnnxEngine(
        SherpaConfig(
            tts_target_rms=0.0,
            tts_output_leveler=True,
            tts_output_lowpass_hz=2000.0,
            tts_declick=False,
        )
    )
    eng._tts = tts

    written: list = []
    eng._synthesize("leveler remains whole clip", written.append)

    assert tts.callback_used == [False]
    assert written


def test_next_fifo_sec_self_sizes_from_underruns():
    """fix 5b control law: grow on a starved reply, slow-decay toward the seed when
    clean, bounded by the UX-latency ceiling. Pure function -- no audio loop."""
    from core.engines.sherpa import _FIFO_SEC_MAX, SherpaOnnxEngine

    f = SherpaOnnxEngine._next_fifo_sec
    seed = 1.0
    # starved (ur>2) grows multiplicatively
    assert f(1.0, 18, seed) == pytest.approx(1.5)
    assert f(1.5, 5, seed) == pytest.approx(2.25)
    # benign 0-2 underruns do NOT grow (end-of-utterance straddle)
    assert f(1.5, 2, seed) == pytest.approx(1.5 * 0.9)   # clean-ish -> decays
    assert f(1.0, 2, seed) == pytest.approx(1.0)          # at seed, stays
    # clean reply slow-decays toward the seed but never below it
    assert f(2.25, 0, seed) == pytest.approx(2.25 * 0.9)
    assert f(1.05, 0, seed) == pytest.approx(1.0)          # clamps to the seed floor
    assert f(1.0, 0, seed) == pytest.approx(1.0)
    # bounded by the ceiling
    assert f(_FIFO_SEC_MAX, 50, seed) == pytest.approx(_FIFO_SEC_MAX)
    assert f(3.5, 50, seed) == pytest.approx(_FIFO_SEC_MAX)  # 3.5*1.5=5.25 -> clamped
