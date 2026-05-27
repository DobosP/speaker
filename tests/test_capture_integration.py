"""Integration tests for the capture stack end to end.

Proves the pieces work *together*: setup_logging + SystemMonitor + the real
VoiceRuntime (brain -> task -> capability -> TTS) -> transcript + per-turn
metrics + system block land in run-<id>.summary.json, and the full app.main
wiring produces the same bundle. No audio hardware, no Ollama, no models.
"""
from __future__ import annotations

import io
import json
import logging
import sys

import pytest

import core.app as app
from core.engines.scripted import ScriptedEngine
from core.llm import EchoLLM
from core.runlog import setup_logging
from core.runtime import VoiceRuntime
from core.sysinfo import SystemMonitor


@pytest.fixture(autouse=True)
def _restore_speaker_logger():
    """setup_logging()/app.main() reconfigure the shared 'speaker' logger; snap
    and restore so these tests don't clobber the conftest test-log handler."""
    lg = logging.getLogger("speaker")
    saved = (list(lg.handlers), lg.level, lg.propagate)
    try:
        yield
    finally:
        lg.handlers[:] = saved[0]
        lg.setLevel(saved[1])
        lg.propagate = saved[2]


def test_runtime_turn_populates_bundle(tmp_path):
    runlog = setup_logging(log_dir=str(tmp_path), run_id="rt", console=False)
    runlog.summary.note(engine="console", llm="echo")
    monitor = SystemMonitor(runlog.summary, interval=10_000)
    monitor.start()

    engine = ScriptedEngine()
    runtime = VoiceRuntime(engine, EchoLLM(), stream_tts=True)
    runtime.start(run_bus=False)
    engine.final("what is the capital of france")
    runtime.wait_idle()
    runtime.stop()

    monitor.stop()
    runlog.finalize([r.as_dict() for r in runtime.metrics.records()])

    data = json.loads((tmp_path / "run-rt.summary.json").read_text(encoding="utf-8"))
    roles = [t["role"] for t in data["transcript"]]
    assert "user" in roles and "assistant" in roles
    assert data["transcript"][0]["text"] == "what is the capital of france"
    assert all("at_sec" in t for t in data["transcript"])
    assert data["counts"]["turns"] >= 1
    assert "system" in data and "baseline" in data["system"]
    # The .txt trace shows the brain's decision + task lifecycle.
    text = (tmp_path / "run-rt.txt").read_text(encoding="utf-8")
    assert "final -> brain" in text
    assert "task" in text and "started" in text


def test_app_main_console_writes_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEAKER_RUN_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "stdin", io.StringIO("hello assistant\nstop\n"))

    rc = app.main(["--engine", "console", "--llm", "echo"])
    assert rc == 0

    summaries = list(tmp_path.glob("run-*.summary.json"))
    assert len(summaries) == 1
    data = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert data["meta"]["engine"] == "console"
    assert data["meta"]["llm"] == "echo"
    assert any(t["role"] == "user" and t["text"] == "hello assistant" for t in data["transcript"])
    assert "system" in data
    # The matching .txt log is in the same bundle.
    assert list(tmp_path.glob("run-*.txt"))


def test_app_main_record_ignored_on_console_engine(tmp_path, monkeypatch):
    # ScriptedEngine has no recorder -> --record should warn, not crash.
    monkeypatch.setenv("SPEAKER_RUN_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))

    rc = app.main(["--engine", "console", "--llm", "echo", "--record"])
    assert rc == 0

    data = json.loads(next(tmp_path.glob("run-*.summary.json")).read_text(encoding="utf-8"))
    assert data["counts"]["warnings"] >= 1
    assert any("record ignored" in e["message"].lower() for e in data["errors"])
    assert "recording" not in data["meta"]
