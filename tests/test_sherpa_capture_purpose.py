from __future__ import annotations

import logging
import sys
from threading import Event, Thread
from types import SimpleNamespace
import time

import numpy as np
import pytest

import core.engines._recovering_input as recovering_input
import core.engines.sherpa as sherpa
from core.engine import EngineCallbacks
from core.engines.sherpa import (
    SherpaConfig,
    SherpaExecutionPurpose,
    SherpaOnnxEngine,
)
from core.realtime_media_stage import RealtimeMediaStage


def _patch_model_builders(monkeypatch, calls: list[str]) -> None:
    monkeypatch.setattr(sherpa, "build_recognizer", lambda _cfg: None)
    monkeypatch.setattr(sherpa, "build_final_recognizer", lambda _cfg: None)
    monkeypatch.setattr(sherpa, "build_final_verifier", lambda _cfg: None)
    monkeypatch.setattr(sherpa, "build_vad", lambda _cfg: None)
    monkeypatch.setattr(sherpa, "build_denoiser", lambda _cfg: None)
    monkeypatch.setattr(sherpa, "build_aec", lambda _cfg, **_kwargs: None)
    monkeypatch.setattr(sherpa, "build_punctuation", lambda _cfg: None)

    def build_tts(_cfg):
        calls.append("tts")
        return object()

    class Kws:
        def create_stream(self):
            return object()

    def build_kws(_cfg):
        calls.append("kws")
        return Kws()

    def build_gate(*_args, **_kwargs):
        calls.append("speaker")
        return object()

    monkeypatch.setattr(sherpa, "build_tts", build_tts)
    monkeypatch.setattr(sherpa, "build_keyword_spotter", build_kws)
    monkeypatch.setattr(sherpa, "sherpa_speaker_gate", build_gate)
    monkeypatch.setattr(
        sherpa,
        "resolve_speaker_identity_activation",
        lambda **_kwargs: SimpleNamespace(
            active=True,
            word_cut_requires_speaker=False,
        ),
    )


def _config() -> SherpaConfig:
    return SherpaConfig(
        endpoint_enabled=False,
        input_calibrate=False,
        final_speech_evidence_enabled=False,
        aec_enabled=False,
        coherence_barge_in_enabled=False,
        barge_in_enabled=False,
        speaker_embedding_model="configured-speaker-model",
        media_pcm_queue_ms=0.0,
    )


def test_capture_build_omits_tts_kws_speaker_while_default_builds_them(
    monkeypatch,
) -> None:
    calls: list[str] = []
    _patch_model_builders(monkeypatch, calls)

    capture = SherpaOnnxEngine(
        _config(),
        purpose=SherpaExecutionPurpose.STT_CAPTURE_ONLY,
    )
    capture._build()

    assert calls == []
    assert capture._tts is None
    assert capture._kws is None
    assert capture._speaker_gate is None
    assert not capture.playback_capabilities.tracked_terminal
    with pytest.raises(RuntimeError, match="playback is unavailable"):
        capture.speak("must not play")

    normal = SherpaOnnxEngine(_config())
    normal._build()

    assert calls == ["tts", "kws", "speaker"]
    assert normal.execution_purpose is SherpaExecutionPurpose.FULL_ASSISTANT
    assert normal._tts is not None
    assert normal._kws is not None
    assert normal._speaker_gate is not None


def test_legacy_uninitialized_engine_defaults_to_full_assistant_purpose() -> None:
    engine = SherpaOnnxEngine.__new__(SherpaOnnxEngine)

    assert engine.execution_purpose is SherpaExecutionPurpose.FULL_ASSISTANT


def test_capture_stt_warm_exercises_only_final_dependencies_before_attestation() -> (
    None
):
    events: list[str] = []

    class Stream:
        result = SimpleNamespace(text="private silence hallucination")

        def accept_waveform(self, sample_rate, samples) -> None:
            assert sample_rate == 16_000
            assert np.count_nonzero(samples) == 0
            events.append("offline-audio")

    class Offline:
        def create_stream(self):
            events.append("offline-create")
            return Stream()

        def decode_stream(self, _stream) -> None:
            events.append("offline-decode")

    class Punctuation:
        def add_punctuation(self, text):
            assert text == "ok"
            events.append("punctuation")

    class Verifier:
        def warm(self) -> None:
            events.append("verifier")

    engine = SherpaOnnxEngine(
        _config(),
        purpose=SherpaExecutionPurpose.STT_CAPTURE_ONLY,
    )
    engine._final_recognizer = Offline()
    engine._punct = Punctuation()
    engine._final_verifier = Verifier()
    engine._tts = object()

    engine._warm_capture_stt_dependencies()
    engine.warm_stt()

    assert events == [
        "offline-create",
        "offline-audio",
        "offline-decode",
        "punctuation",
        "verifier",
    ]
    assert engine._capture_stt_warmed


def test_capture_start_queries_no_output_and_launches_no_playback_owner(
    monkeypatch,
) -> None:
    calls: list[str] = []
    _patch_model_builders(monkeypatch, calls)
    close_requested = Event()
    device_queries: list[str] = []
    lifecycle: list[str] = []

    class Input:
        actual_samplerate = 16_000
        actual_device = "capture-input"
        generation = 1

        def __init__(self, *_args, **_kwargs) -> None:
            lifecycle.append("mic-construct")

        def open(self) -> None:
            lifecycle.append("mic-open")

        def read(self, frames):
            close_requested.wait(timeout=2.0)
            return np.zeros(frames, dtype="float32"), False

        def request_close(self) -> None:
            close_requested.set()

        def abort_read(self, *, timeout=None) -> bool:
            close_requested.set()
            return True

        def close(self, *, teardown_timeout=None) -> bool:
            close_requested.set()
            return True

    def query_devices(_device=None, *, kind=None):
        device_queries.append(kind)
        if kind == "output":
            raise AssertionError("capture start queried output")
        return {"name": "capture-input", "default_samplerate": 16_000}

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(
            query_devices=query_devices,
            check_input_settings=lambda **_kwargs: None,
        ),
    )
    monkeypatch.setattr(recovering_input, "_RecoveringInputStream", Input)
    engine = SherpaOnnxEngine(
        _config(),
        purpose=SherpaExecutionPurpose.STT_CAPTURE_ONLY,
    )
    original_warm = engine._warm_capture_stt_dependencies

    def traced_warm() -> None:
        assert not engine._running.is_set()
        lifecycle.append("warm")
        original_warm()

    monkeypatch.setattr(engine, "_warm_capture_stt_dependencies", traced_warm)
    monkeypatch.setattr(engine, "_resolve_capture_domain", lambda *_a, **_k: True)
    monkeypatch.setattr(engine, "_rebind_speech_evidence_domain", lambda: None)

    engine.start(EngineCallbacks())
    try:
        assert device_queries and set(device_queries) == {"input"}
        assert lifecycle[:3] == ["warm", "mic-construct", "mic-open"]
        assert calls == []
        assert engine._capture_stt_warmed
        assert engine._play_thread is None
        assert engine._receipt_thread is None
        assert not engine._receipt_running.is_set()
    finally:
        engine.stop()


@pytest.mark.parametrize("above_floor", [True, False])
def test_capture_final_and_drop_logs_never_contain_recognized_text(
    caplog,
    above_floor: bool,
) -> None:
    canary = "PRIVATE RECOGNIZED TEXT CANARY"
    engine = SherpaOnnxEngine(
        _config(),
        purpose=SherpaExecutionPurpose.STT_CAPTURE_ONLY,
    )
    engine._final_recognizer = None
    engine._final_verifier = None
    engine._final_above_floor = lambda _samples: above_floor
    engine._cb = EngineCallbacks()

    with caplog.at_level(logging.DEBUG):
        engine._finalize_and_dispatch(
            np.full(3200, 0.2, dtype="float32"),
            canary,
            time.perf_counter(),
        )

    assert canary not in caplog.text


def test_capture_final_dependency_and_callback_exceptions_hide_private_text(
    caplog,
) -> None:
    canary = "PRIVATE EXCEPTION TEXT CANARY"

    class Punctuation:
        def add_punctuation(self, _text):
            raise RuntimeError(canary)

    class Offline:
        def create_stream(self):
            raise RuntimeError(canary)

    class Verifier:
        def transcribe(self, _samples, _sample_rate):
            raise RuntimeError(canary)

    def callback(_result) -> None:
        raise RuntimeError(canary)

    engine = SherpaOnnxEngine(
        _config(),
        purpose=SherpaExecutionPurpose.STT_CAPTURE_ONLY,
    )
    engine._punct = Punctuation()
    engine._final_recognizer = Offline()
    engine._final_verifier = Verifier()
    engine._final_above_floor = lambda _samples: True
    engine._cb = EngineCallbacks(on_final_result=callback)

    with caplog.at_level(logging.DEBUG):
        engine._finalize_and_dispatch(
            np.full(32_000, 0.2, dtype="float32"),
            "ordinary recognized words",
            time.perf_counter(),
            speech_sec=2.0,
        )

    assert canary not in caplog.text
    assert "punctuation model failed" not in caplog.text
    assert "second-pass recognizer failed" not in caplog.text
    assert "final verifier failed" not in caplog.text
    assert "final transcript callback failed" in caplog.text


def test_capture_async_final_worker_exception_hides_private_text(caplog) -> None:
    canary = "PRIVATE WORKER EXCEPTION CANARY"
    engine = SherpaOnnxEngine(
        _config(),
        purpose=SherpaExecutionPurpose.STT_CAPTURE_ONLY,
    )
    stage = RealtimeMediaStage(max_queued=2)
    engine._final_stage = stage

    def fail_final(*_args, **_kwargs) -> None:
        raise RuntimeError(canary)

    engine._finalize_and_dispatch = fail_final
    engine._running.set()
    worker = Thread(target=engine._final_worker, args=(stage,), daemon=True)
    with caplog.at_level(logging.DEBUG):
        worker.start()
        engine._enqueue_final(
            np.ones(1600, dtype="float32"),
            "private queued text",
            time.perf_counter(),
        )
        deadline = time.monotonic() + 2.0
        while stage.snapshot().finished < 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        stage.close()
        engine._running.clear()
        worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert stage.snapshot().finished == 1
    assert "final ASR processing failed" in caplog.text
    assert canary not in caplog.text
