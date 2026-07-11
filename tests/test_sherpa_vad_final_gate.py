"""Capture-loop regressions for the load-bearing normal-final VAD gate."""
from __future__ import annotations

import threading
import time

import numpy as np

from core.engine import EngineCallbacks
from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine


class _Input:
    def __init__(self, engine, blocks=1):
        self.engine = engine
        self.left = blocks

    def read(self, n):
        self.left -= 1
        if self.left <= 0:
            self.engine._running.clear()
        return np.zeros(n, dtype="float32"), False


class _Stream:
    def accept_waveform(self, sample_rate, samples):
        pass


class _HallucinatingRecognizer:
    """The exact live idle shape: stable raw ``AND`` + an endpoint."""
    def create_stream(self):
        return _Stream()

    def is_ready(self, stream):
        return False

    def decode_stream(self, stream):  # pragma: no cover - never ready
        pass

    def get_result(self, stream):
        return "AND"

    def is_endpoint(self, stream):
        return True

    def reset(self, stream):
        pass


class _Vad:
    def __init__(self, speech: bool):
        self.speech = speech
        self.accepted = 0

    def accept_waveform(self, samples):
        self.accepted += 1

    def is_speech_detected(self):
        return self.speech


def _run(*, vad_speech: bool):
    engine = SherpaOnnxEngine(SherpaConfig(endpoint_enabled=False))
    engine._recognizer = _HallucinatingRecognizer()
    engine._vad = _Vad(vad_speech)
    engine._stream_in = _Input(engine)
    finals: list[str] = []
    metrics: list[str] = []
    engine._cb = EngineCallbacks(
        on_final=finals.append,
        on_metric=lambda name, **kwargs: metrics.append(name),
    )
    engine._running.set()
    thread = threading.Thread(target=engine._capture_loop)
    thread.start()
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    return engine, finals, metrics


def test_idle_and_hallucination_is_dropped_when_vad_saw_no_speech():
    engine, finals, metrics = _run(vad_speech=False)
    assert finals == []
    assert metrics.count("vad_rejected_final") == 1
    assert engine._vad.accepted == 1


def test_same_recognizer_final_is_admitted_after_vad_speech():
    engine, finals, metrics = _run(vad_speech=True)
    assert finals == ["And"]
    assert "vad_rejected_final" not in metrics
    assert engine._vad.accepted == 1


def test_stop_drops_block_that_returns_after_capture_shutdown_signal():
    """An abort-unblocked stale block cannot re-enter ASR during teardown."""
    read_started = threading.Event()
    release_read = threading.Event()

    class _BlockingInput:
        generation = 0

        def read(self, n):
            read_started.set()
            assert release_read.wait(timeout=2.0)
            return np.ones(n, dtype="float32"), False

        def request_close(self):
            release_read.set()

        def close(self):
            return True

    engine = SherpaOnnxEngine(SherpaConfig(endpoint_enabled=False, block_sec=0.02))
    engine._recognizer = _HallucinatingRecognizer()
    engine._vad = _Vad(True)
    engine._stream_in = _BlockingInput()
    engine._cb = EngineCallbacks()
    engine._running.set()
    engine._capture_thread = threading.Thread(target=engine._capture_loop, daemon=True)
    engine._capture_thread.start()
    assert read_started.wait(timeout=1.0)

    engine.stop()

    assert not engine._capture_thread.is_alive()
    assert engine._vad.accepted == 0


def test_stop_after_initial_read_check_fences_post_dsp_side_effects():
    """Shutdown during DSP drops the block before recorder/KWS/ASR callbacks."""
    dsp_entered = threading.Event()
    release_dsp = threading.Event()

    class _Input:
        generation = 0

        def read(self, n):
            return np.ones(n, dtype="float32"), False

        def request_close(self):
            pass

        def close(self):
            return True

    class _Denoiser:
        def process_16k(self, samples):
            dsp_entered.set()
            assert release_dsp.wait(timeout=2.0)
            return samples

        def reset(self):
            pass

    class _Recorder:
        seconds = 0.1
        path = "headless.wav"

        def __init__(self):
            self.writes = 0
            self.closes = 0

        def write(self, _samples):
            self.writes += 1

        def close(self):
            self.closes += 1

    events = []
    recorder = _Recorder()
    engine = SherpaOnnxEngine(SherpaConfig(endpoint_enabled=False, block_sec=0.02))
    engine._recognizer = _HallucinatingRecognizer()
    engine._vad = _Vad(True)
    engine._denoiser = _Denoiser()
    engine._stream_in = _Input()
    engine._recorder = recorder
    engine._poll_keywords = lambda _samples: events.append("kws")
    engine._cb = EngineCallbacks(
        on_partial=lambda _text: events.append("partial"),
        on_final=lambda _text: events.append("final"),
        on_command=lambda _text: events.append("command"),
        on_heartbeat=lambda: events.append("heartbeat"),
        on_barge_in=lambda: events.append("barge"),
    )
    engine._running.set()
    engine._capture_thread = threading.Thread(target=engine._capture_loop, daemon=True)
    engine._capture_thread.start()
    assert dsp_entered.wait(timeout=1.0)

    stopper = threading.Thread(target=engine.stop, daemon=True)
    stopper.start()
    assert engine._capture_stopping.wait(timeout=1.0)
    release_dsp.set()
    stopper.join(timeout=2.0)

    assert not stopper.is_alive()
    assert not engine._capture_thread.is_alive()
    assert recorder.writes == 0
    assert recorder.closes == 1
    assert events == []


def test_stop_retains_resources_while_admitted_capture_effect_is_stuck(monkeypatch):
    """A block admitted before stop keeps every resource it may still touch."""
    from core.engines import sherpa as sherpa_module

    monkeypatch.setattr(sherpa_module, "_CAPTURE_FORCE_JOIN_TIMEOUT_SEC", 0.02)
    write_entered = threading.Event()
    release_write = threading.Event()

    class _Input:
        generation = 0

        def __init__(self):
            self.close_calls = 0

        def read(self, n):
            return np.ones(n, dtype="float32"), False

        def request_close(self):
            pass

        def abort_read(self, *, timeout):
            return False

        def close(self):
            self.close_calls += 1
            return True

    class _Recorder:
        seconds = 0.1
        path = "headless.wav"

        def __init__(self):
            self.close_calls = 0

        def write(self, _samples):
            write_entered.set()
            assert release_write.wait(timeout=2.0)

        def close(self):
            self.close_calls += 1

    class _Output:
        def __init__(self):
            self.stop_calls = 0
            self.close_calls = 0

        def stop(self):
            self.stop_calls += 1

        def close(self):
            self.close_calls += 1

    stream = _Input()
    recorder = _Recorder()
    output = _Output()
    engine = SherpaOnnxEngine(SherpaConfig(endpoint_enabled=False, block_sec=0.01))
    engine._recognizer = _HallucinatingRecognizer()
    engine._vad = _Vad(True)
    engine._stream_in = stream
    engine._recorder = recorder
    engine._out_stream = output
    engine._cb = EngineCallbacks()
    engine._running.set()
    engine._capture_thread = threading.Thread(target=engine._capture_loop, daemon=True)
    engine._capture_thread.start()
    assert write_entered.wait(timeout=1.0)

    started = time.monotonic()
    engine.stop()
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    assert engine._capture_resource_hold.is_set()
    assert engine._stream_in is stream
    assert stream.close_calls == 0
    assert recorder.close_calls == 0
    assert output.stop_calls == 0 and output.close_calls == 0

    release_write.set()
    engine._capture_thread.join(timeout=1.0)
    assert not engine._capture_thread.is_alive()
    engine.stop()
    assert not engine._capture_resource_hold.is_set()
    assert engine._stream_in is None
    assert stream.close_calls == 1
    assert recorder.close_calls == 1
    assert output.stop_calls == 1 and output.close_calls == 1
    assert engine._out_stream is None


def test_active_vad_rule3_endpoint_finalizes_and_resets_bounded_segment():
    from core.endpointing import ScriptedTurnCompletionDetector

    class _Rule3Recognizer(_HallucinatingRecognizer):
        def __init__(self):
            self.resets = 0

        def get_result(self, stream):
            return "and then"

        def reset(self, stream):
            self.resets += 1

    detector = ScriptedTurnCompletionDetector({"and then": 0.05})
    engine = SherpaOnnxEngine(
        SherpaConfig(endpoint_enabled=True, endpoint_max_silence_sec=1.6),
        turn_detector=detector,
    )
    recognizer = _Rule3Recognizer()
    engine._recognizer = recognizer
    engine._vad = _Vad(True)  # continuous speech: no semantic-silence clock
    engine._stream_in = _Input(engine)
    finals: list[str] = []
    engine._cb = EngineCallbacks(on_final=finals.append)

    engine._running.set()
    thread = threading.Thread(target=engine._capture_loop)
    thread.start()
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert finals == ["And then"]
    assert recognizer.resets == 1
    assert detector.calls == []


def test_capture_reopen_preserves_recovered_block_after_rebinding_domain():
    from core.engines.speaker_gate import SpeakerGate

    class _RecoveringInput:
        def __init__(self, engine):
            self.engine = engine
            self.generation = 1
            self.actual_samplerate = 16000
            self.actual_device = "preferred"
            self.read_sizes = []
            self.calls = 0

        def read(self, n):
            self.read_sizes.append(n)
            self.calls += 1
            if self.calls == 1:
                return np.full(n, 0.11, dtype="float32"), False
            if self.calls == 2:
                self.generation = 2
                self.actual_samplerate = 48000
                self.actual_device = None
                # Mirror _RecoveringInputStream: the read was initiated with the
                # old frame count, but its internal retry returns one correctly
                # timed 100 ms block at the recovered rate.
                return np.full(4800, 0.22, dtype="float32"), False
            self.engine._running.clear()
            return np.full(n, 0.33, dtype="float32"), False

    class _Recognizer:
        def __init__(self):
            self.accepted = []
            self.resets = 0

        def create_stream(self):
            return _Stream()

        def is_ready(self, stream):
            return False

        def decode_stream(self, stream):  # pragma: no cover
            pass

        def get_result(self, stream):
            return "hello" if self.accepted else ""

        def is_endpoint(self, stream):
            return self.resets > 0 and bool(self.accepted)

        def reset(self, stream):
            self.resets += 1
            self.accepted.clear()

    class _SpeechVad(_Vad):
        def __init__(self):
            super().__init__(True)
            self.resets = 0

        def reset(self):
            self.resets += 1

    recognizer = _Recognizer()

    def accept(_sr, samples):
        recognizer.accepted.append(np.asarray(samples).copy())

    engine = SherpaOnnxEngine(SherpaConfig(endpoint_enabled=False))
    stream = _Stream()
    stream.accept_waveform = accept
    recognizer.create_stream = lambda: stream
    engine._recognizer = recognizer
    engine._vad = _SpeechVad()
    engine._stream_in = _RecoveringInput(engine)
    engine._capture_sr = 16000
    gate = SpeakerGate(threshold=0.5, embed_fn=lambda _s, _sr: [1.0])
    gate.enroll_embedding([1.0])
    engine._speaker_gate = gate
    resolved = []
    engine._resolve_capture_domain = (
        lambda _sd, selector, **_kwargs: resolved.append(selector) or False
    )
    finals = []
    engine._finalize_and_dispatch = (
        lambda seg, raw, speech_end, asr_seg=None, speech_sec=None: finals.append(
            np.asarray(seg).copy()
        )
    )
    engine._cb = EngineCallbacks()

    engine._running.set()
    thread = threading.Thread(target=engine._capture_loop)
    thread.start()
    thread.join(timeout=5.0)
    assert not thread.is_alive()

    assert engine._stream_in.read_sizes == [1600, 1600, 4800]
    assert engine._capture_sr == 48000
    assert engine._resampler is not None and engine._resampler.src_sr == 48000
    assert recognizer.resets >= 1
    assert resolved == [None]
    assert not gate.is_enrolled
    assert finals and not np.any(np.isclose(finals[0], 0.11))
    assert np.any(np.isclose(finals[0], 0.22, atol=1e-3))
