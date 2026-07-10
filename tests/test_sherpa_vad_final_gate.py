"""Capture-loop regressions for the load-bearing normal-final VAD gate."""
from __future__ import annotations

import threading

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
