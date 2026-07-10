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

