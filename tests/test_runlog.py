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


def test_async_formatting_preserves_args_and_traceback(tmp_path):
    # With the deferred-formatting queue handler, args + exc_info reach the
    # listener thread intact: the message interpolates and the traceback is
    # captured as a structured field (not just embedded in the text).
    runlog = setup_logging(debug=False, log_dir=str(tmp_path), run_id="exc", console=False)
    log = logging.getLogger("speaker.tasks")
    log.info("answered in %.2fs (%d chars)", 0.5, 42)  # %-args formatted off-thread
    try:
        raise ValueError("boom")
    except ValueError:
        log.exception("task blew up")
    runlog.finalize()

    text = (tmp_path / "run-exc.txt").read_text(encoding="utf-8")
    assert "answered in 0.50s (42 chars)" in text
    assert "Traceback (most recent call last)" in text  # traceback rendered to file

    data = json.loads((tmp_path / "run-exc.summary.json").read_text(encoding="utf-8"))
    err = [e for e in data["errors"] if "blew up" in e["message"]]
    assert err and err[0]["exc"] and "ValueError: boom" in err[0]["exc"]


def test_prune_old_runs_keeps_newest(tmp_path):
    from core.runlog import prune_old_runs

    # Six runs, each a .txt + .summary.json (+ a .wav on one).
    for i in range(6):
        stem = f"run-2026010{i}-000000"
        (tmp_path / f"{stem}.txt").write_text("x", encoding="utf-8")
        (tmp_path / f"{stem}.summary.json").write_text("{}", encoding="utf-8")
    (tmp_path / "run-20260105-000000.wav").write_bytes(b"RIFF")

    removed = prune_old_runs(str(tmp_path), keep=2)
    assert removed == 4
    remaining = sorted({p.name.split(".", 1)[0] for p in tmp_path.glob("run-*.*")})
    assert remaining == ["run-20260104-000000", "run-20260105-000000"]
    # The whole bundle for a kept run survives (incl. its .wav).
    assert (tmp_path / "run-20260105-000000.wav").exists()


def test_prune_keep_zero_is_noop(tmp_path):
    from core.runlog import prune_old_runs

    (tmp_path / "run-x.txt").write_text("x", encoding="utf-8")
    assert prune_old_runs(str(tmp_path), keep=0) == 0
    assert (tmp_path / "run-x.txt").exists()


def test_prune_never_deletes_protected_tracked_bundles(tmp_path):
    """Committed (git-tracked) bundles are a curated corpus and must survive the
    per-startup prune regardless of age -- only ephemeral runs count toward keep.
    This is what protects the barge-in replay WAVs the owner keeps for dev."""
    from core.runlog import prune_old_runs

    # 4 ephemeral (untracked) runs ...
    for i in range(4):
        (tmp_path / f"run-2026020{i}-000000.txt").write_text("x", encoding="utf-8")
    # ... and 2 OLDER "tracked" bundles (with WAVs) that must never be pruned.
    protected = {"run-20260101-000000", "run-20260102-000000"}
    for stem in protected:
        (tmp_path / f"{stem}.txt").write_text("keep", encoding="utf-8")
        (tmp_path / f"{stem}.wav").write_bytes(b"RIFF")

    removed = prune_old_runs(str(tmp_path), keep=2, protected=protected)

    # Only the 4 ephemeral runs count toward keep=2 -> 2 oldest ephemeral pruned.
    assert removed == 2
    # Both protected bundles survive (incl. their WAVs) despite being the oldest.
    assert (tmp_path / "run-20260101-000000.wav").exists()
    assert (tmp_path / "run-20260102-000000.wav").exists()
    remaining = {p.name.split(".", 1)[0] for p in tmp_path.glob("run-*.*")}
    assert "run-20260203-000000" in remaining  # newest ephemeral kept
    assert "run-20260200-000000" not in remaining  # oldest ephemeral pruned


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


def test_summary_does_not_flag_echo_transcript_as_missing_llm(tmp_path):
    runlog = setup_logging(
        debug=False, log_dir=str(tmp_path), run_id="echo", console=False
    )
    runlog.summary.note(engine="sherpa", llm="echo")
    logging.getLogger("speaker.runtime").info(
        "final", extra={"transcript": {"role": "user", "text": "hello"}}
    )
    logging.getLogger("speaker.runtime").info(
        "assistant",
        extra={"transcript": {"role": "assistant", "text": "you said hello"}},
    )
    runlog.finalize()
    data = json.loads(
        (tmp_path / "run-echo.summary.json").read_text(encoding="utf-8")
    )

    assert not any("no LLM request" in hint for hint in data["stuck_hints"])


@pytest.mark.parametrize(
    "wd_message, expected_hint_substring",
    [
        ("llm stuck: turn 0 had asr_final but no llm_first_token after 12.0s", "LLM stalled mid-turn"),
        ("tts stuck: turn 0 had llm_first_token but no tts_first_audio after 7.0s", "TTS stalled mid-turn"),
        ("capture silent: no heartbeat for 8.0s (audio thread crashed?)", "capture thread went silent"),
        ("barge-in storm: 4 detections in the last 1.5s (gate flapping)", "barge-in gate flapping"),
    ],
)
def test_watchdog_warnings_promote_to_named_stuck_hints(tmp_path, wd_message, expected_hint_substring):
    runlog = setup_logging(debug=False, log_dir=str(tmp_path), run_id="wd", console=False)
    runlog.summary.note(engine="sherpa")
    logging.getLogger("speaker.watchdog").warning(wd_message)
    runlog.finalize()
    data = json.loads((tmp_path / "run-wd.summary.json").read_text(encoding="utf-8"))
    assert any(expected_hint_substring in hint for hint in data["stuck_hints"]), (
        f"expected {expected_hint_substring!r} in stuck_hints, got {data['stuck_hints']}"
    )
