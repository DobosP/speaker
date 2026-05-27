"""Tests for the run-logging summary (core.runlog).

Pure: drives the logger + summary handler directly, no app run needed.
"""
from __future__ import annotations

import json
import logging

import pytest

from core.runlog import setup_logging


@pytest.fixture(autouse=True)
def _restore_speaker_logger():
    """setup_logging() reconfigures the shared 'speaker' logger; snapshot and
    restore it so these tests don't clobber the conftest test-log handler."""
    lg = logging.getLogger("speaker")
    saved = (list(lg.handlers), lg.level, lg.propagate)
    try:
        yield
    finally:
        lg.handlers[:] = saved[0]
        lg.setLevel(saved[1])
        lg.propagate = saved[2]


def test_setup_logging_writes_files_and_aggregates(tmp_path):
    runlog = setup_logging(debug=True, log_dir=str(tmp_path), run_id="unit", console=False)
    runlog.summary.note(engine="sherpa", llm="ollama", device="desktop")

    log = logging.getLogger("speaker.llm.ollama")
    # An LLM request line with the structured extra the summary harvests.
    log.info(
        "ollama gemma3:4b done",
        extra={
            "llm_request": {
                "model": "gemma3:4b",
                "duration_sec": 0.42,
                "out_chars": 31,
                "cancelled": False,
            }
        },
    )
    # Transcript entries (user + assistant) flow through the same async path.
    logging.getLogger("speaker.runtime").info(
        "final", extra={"transcript": {"role": "user", "text": "hello there"}}
    )
    logging.getLogger("speaker.runtime").info(
        "assistant", extra={"transcript": {"role": "assistant", "text": "hi!"}}
    )
    logging.getLogger("speaker.tasks").error("task boom: ConnectionError")

    runlog.finalize(metrics_records=[{"first_audio_latency": 0.5}])

    # The .txt log exists and captured our lines (listener.stop() flushed it).
    text = (tmp_path / "run-unit.txt").read_text(encoding="utf-8")
    assert "ollama gemma3:4b done" in text
    assert "task boom" in text

    data = json.loads((tmp_path / "run-unit.summary.json").read_text(encoding="utf-8"))
    assert data["meta"]["engine"] == "sherpa"
    assert data["counts"]["llm_requests"] == 1
    assert data["counts"]["errors"] == 1
    assert data["counts"]["turns"] == 1
    assert data["llm"]["requests"][0]["model"] == "gemma3:4b"
    assert data["llm"]["total_time_sec"] == 0.42
    assert any("boom" in e["message"] for e in data["errors"])
    # Transcript captured in order, with relative stage timing.
    roles = [t["role"] for t in data["transcript"]]
    assert roles == ["user", "assistant"]
    assert data["transcript"][0]["text"] == "hello there"
    assert all("at_sec" in t for t in data["transcript"])
    assert data["counts"]["transcript_entries"] == 2


def test_finalize_is_idempotent(tmp_path):
    runlog = setup_logging(debug=False, log_dir=str(tmp_path), run_id="idem", console=False)
    runlog.finalize()
    runlog.finalize()  # second call is a no-op, not an error
    assert (tmp_path / "run-idem.summary.json").exists()


def test_summary_flags_all_cancelled_llm(tmp_path):
    runlog = setup_logging(debug=False, log_dir=str(tmp_path), run_id="cancel", console=False)
    runlog.summary.note(engine="sherpa")
    logging.getLogger("speaker.llm.ollama").info(
        "cut off", extra={"llm_request": {"model": "m", "duration_sec": 0.1, "cancelled": True}}
    )
    runlog.finalize()
    data = json.loads((tmp_path / "run-cancel.summary.json").read_text(encoding="utf-8"))
    assert any("cancelled" in hint for hint in data["stuck_hints"])


def test_summary_flags_no_llm_request_for_voice_engine(tmp_path):
    runlog = setup_logging(debug=False, log_dir=str(tmp_path), run_id="nollm", console=False)
    runlog.summary.note(engine="sherpa")
    runlog.finalize()
    data = json.loads((tmp_path / "run-nollm.summary.json").read_text(encoding="utf-8"))
    assert any("no LLM request" in hint for hint in data["stuck_hints"])
