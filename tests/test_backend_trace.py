"""Tests for backend call tracing and STT serialization counters."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import backend_trace
from utils.llm import LocalLLM
from utils.stt import transcribe_audio


def test_reset_and_partial_counter():
    backend_trace.reset()
    backend_trace.configure(enabled=False, diagnostics_log_path=None)
    backend_trace.record_stt_partial("tiny", 100)
    backend_trace.record_stt_partial("tiny", 200)
    assert backend_trace.snapshot()["stt_partial"] == 2
    backend_trace.reset()
    assert backend_trace.snapshot()["stt_partial"] == 0


def test_transcribe_audio_whisper_counts_final_only(monkeypatch):
    backend_trace.reset()
    backend_trace.configure(enabled=False, diagnostics_log_path=None)

    class FakeModel:
        backend = "faster-whisper"

        def transcribe(self, audio):
            return "ok"

    monkeypatch.setattr(
        "utils.stt.get_stt_model",
        lambda model_id: FakeModel(),
    )
    audio = np.zeros(1600, dtype=np.float32)
    out = transcribe_audio(audio, model_id="base", model_type="whisper")
    assert out == "ok"
    snap = backend_trace.snapshot()
    assert snap["stt_final"] == 1
    assert snap["stt_partial"] == 0


def test_transcribe_audio_whispercpp_final_no_partial_counter(monkeypatch):
    backend_trace.reset()
    backend_trace.configure(enabled=False, diagnostics_log_path=None)

    class FakeStreaming:
        model_id = "tiny"

        def transcribe_final(self, audio):
            return "done"

    monkeypatch.setattr(
        "utils.stt.get_streaming_stt",
        lambda model_id, n_threads=4: FakeStreaming(),
    )
    audio = np.zeros(3200, dtype=np.float32)
    out = transcribe_audio(
        audio, model_id="tiny", model_type="whispercpp", n_threads=4
    )
    assert out == "done"
    snap = backend_trace.snapshot()
    assert snap["stt_final"] == 1
    assert snap["stt_partial"] == 0


def test_local_llm_streaming_records_one_stream_chat(monkeypatch):
    backend_trace.reset()
    backend_trace.configure(enabled=False, diagnostics_log_path=None)

    monkeypatch.setattr("ollama.list", lambda: {})

    def fake_chat(**kwargs):
        if kwargs.get("stream"):
            return iter(
                [
                    {"message": {"content": "Say "}},
                    {"message": {"content": "hello."}},
                ]
            )
        return {"message": {"content": "batch"}}

    monkeypatch.setattr("ollama.chat", fake_chat)

    llm = LocalLLM(model="test-model", generation_profile="balanced")
    parts = list(llm.get_streaming_response("ping"))
    assert len(parts) >= 1
    snap = backend_trace.snapshot()
    assert snap["ollama_chat_stream"] == 1
    assert snap["ollama_chat_batch"] == 0


def test_local_llm_batch_records_one_batch_chat(monkeypatch):
    backend_trace.reset()
    backend_trace.configure(enabled=False, diagnostics_log_path=None)

    monkeypatch.setattr("ollama.list", lambda: {})
    monkeypatch.setattr(
        "ollama.chat",
        lambda **kwargs: {"message": {"content": "Reply."}},
    )

    llm = LocalLLM(model="test-model", generation_profile="balanced")
    text = llm.get_response("hello")
    assert "Reply" in text or text
    snap = backend_trace.snapshot()
    assert snap["ollama_chat_batch"] == 1
