"""
Schema checks for pipeline JSONL using real session mic payloads (shape only).

Ensures recorded-session dimensions fit the structured logging fields used on
the utterance path without spinning up full STT/LLM.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from utils import pipeline_log

REPO_ROOT = Path(__file__).resolve().parents[1]
SESSION_MIC = (
    REPO_ROOT
    / "recordings"
    / "session_20260406_165119"
    / "turn_000"
    / "mic_16k.npy"
)


def teardown_module() -> None:
    pipeline_log.configure(enabled=False, path=None)


def _assert_pipeline_emit(obj: dict) -> None:
    assert obj["subsystem"] == "pipeline"
    assert isinstance(obj["session_id"], str)
    assert isinstance(obj["timestamp"], (int, float))


@pytest.mark.audio
def test_session_mic_load_and_pipeline_schema(tmp_path: Path):
    if not SESSION_MIC.is_file():
        pytest.skip("Checkout recordings/ or run with session artifacts present")

    mic = np.load(str(SESSION_MIC))
    n = int(len(mic))

    log_path = tmp_path / "pipeline.jsonl"
    pipeline_log.configure(enabled=True, path=str(log_path), session_id="schema_ut")

    with pipeline_log.span(
        "utterance",
        "final_stt",
        model="stub",
        model_type="stub",
        audio_samples=n,
    ):
        pass
    pipeline_log.emit(
        "pipeline_route",
        action="llm",
        reason="test_path",
        transcript_chars=12,
        capability=None,
    )
    pipeline_log.emit("cancel_output")
    pipeline_log.emit(
        "barge_in",
        should_stop=False,
        voiced=True,
        rms=0.01,
    )
    pipeline_log.emit(
        "tts_speak",
        text_chars=42,
        defer_assistant_speaking=False,
        chunked=True,
    )

    pipeline_log.flush_sync()
    pipeline_log.configure(enabled=False, path=None)

    lines = [json.loads(L) for L in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) >= 5

    span_ev = next(L for L in lines if L["event"] == "pipeline_span")
    assert span_ev["component"] == "utterance"
    assert span_ev["span"] == "final_stt"
    assert span_ev["audio_samples"] == n
    _assert_pipeline_emit(span_ev)

    route_ev = next(L for L in lines if L["event"] == "pipeline_route")
    assert route_ev["action"] == "llm"
    _assert_pipeline_emit(route_ev)

    for ev in lines:
        _assert_pipeline_emit(ev)
