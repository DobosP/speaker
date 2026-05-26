"""Unit tests for ``utils.pipeline_log`` (bounded queue, JSON shape)."""

from __future__ import annotations

import json
from pathlib import Path

from utils import pipeline_log


def teardown_module() -> None:
    pipeline_log.configure(enabled=False, path=None)


def test_emit_noop_when_disabled():
    pipeline_log.configure(enabled=False, path=None)
    pipeline_log.emit("should_not_crash", x=1)


def test_emit_and_span_write_jsonl(tmp_path: Path):
    log_path = tmp_path / "p.jsonl"
    pipeline_log.configure(enabled=True, path=str(log_path), session_id="ut")
    pipeline_log.emit("probe", k="v")
    with pipeline_log.span("c", "s", extra_field=3):
        pass
    pipeline_log.flush_sync()
    pipeline_log.configure(enabled=False, path=None)

    text = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(text) >= 2
    ev0 = json.loads(text[0])
    assert ev0["event"] == "probe"
    assert ev0["subsystem"] == "pipeline"
    assert ev0["session_id"] == "ut"
    assert "timestamp" in ev0
    assert ev0["k"] == "v"

    span_ev = json.loads(text[-1])
    assert span_ev["event"] == "pipeline_span"
    assert span_ev["component"] == "c"
    assert span_ev["span"] == "s"
    assert "duration_ms" in span_ev
    assert span_ev["extra_field"] == 3


def test_configure_shutdown_is_repeatable(tmp_path: Path):
    log_path = tmp_path / "t.jsonl"
    pipeline_log.configure(enabled=True, path=str(log_path))
    pipeline_log.configure(enabled=False, path=None)
    pipeline_log.configure(enabled=True, path=str(log_path))
    pipeline_log.emit("after_reopen")
    pipeline_log.flush_sync()
    pipeline_log.configure(enabled=False, path=None)
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["event"] == "after_reopen"
