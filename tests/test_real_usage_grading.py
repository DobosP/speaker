"""Unit tests for the NON-audio half of the real-usage harness: the PASS/FAIL
grading, the report rendering, and the shutdown-timeout wrapper.

No audio, no models, no sounddevice -- these run in CI exactly like the
tools/live_session logic tests, while the real-audio run stays on-machine-only.
The three live failures the harness catches each get a grading test here, plus a
direct test of the shutdown-timeout wrapper against a fake stop() that HANGS
(the regression test for the shutdown hang, without needing a wedged ALSA write).
"""
from __future__ import annotations

import threading
import time

import pytest

from tools.real_usage import report as R
from tools.real_usage.runner import run_stop_with_timeout


def _clean_result(**overrides):
    """A result dict that should grade PASS; override fields per test."""
    base = {
        "fixture": "run-clean.wav",
        "asr_finals": ["are you listening to me"],
        "spoken": ["Mode: assistant", "Yes, I'm listening to you."],
        "first_audio_latencies": [1.8],
        "barge_in_count": 0,
        "playback_errors": [],
        "playback_loop_dead": False,
        "shutdown_ok": True,
        "shutdown_seconds": 0.4,
        "shutdown_timeout": 8.0,
        "error": None,
    }
    base.update(overrides)
    return base


# --- grade_fixture: the happy path ---

def test_clean_result_passes():
    g = R.grade_fixture(_clean_result())
    assert g["verdict"] == "PASS"
    assert g["passed"] is True
    assert g["reasons"] == []
    assert g["spoken_answer"] == "Yes, I'm listening to you."
    assert g["checks"]["transcript_sane"] is True
    assert g["checks"]["response_present"] is True
    assert g["checks"]["playback_clean"] is True
    assert g["checks"]["shutdown_clean"] is True
    assert g["checks"]["barge_in_storm"] is False


# --- failure #1: SHUTDOWN HANG ---

def test_shutdown_over_timeout_fails():
    g = R.grade_fixture(_clean_result(shutdown_ok=False, shutdown_seconds=8.0, shutdown_timeout=8.0))
    assert g["passed"] is False
    assert g["checks"]["shutdown_clean"] is False
    assert any("SHUTDOWN HANG" in r for r in g["reasons"])


def test_shutdown_not_ok_even_under_time_fails():
    # stop() returned False (e.g. join timed out) even though elapsed < timeout.
    g = R.grade_fixture(_clean_result(shutdown_ok=False, shutdown_seconds=1.2, shutdown_timeout=8.0))
    assert g["passed"] is False
    assert g["checks"]["shutdown_clean"] is False


def test_shutdown_at_timeout_boundary_fails():
    g = R.grade_fixture(_clean_result(shutdown_ok=True, shutdown_seconds=8.0, shutdown_timeout=8.0))
    # ok=True but seconds >= timeout still counts as a hang (the join returned
    # 'alive' exactly at the boundary).
    assert g["checks"]["shutdown_clean"] is False
    assert g["passed"] is False


# --- failure #2: BARGE-IN STORM ---

def test_barge_in_storm_fails():
    g = R.grade_fixture(_clean_result(barge_in_count=12))
    assert g["passed"] is False
    assert g["checks"]["barge_in_storm"] is True
    assert any("BARGE-IN STORM" in r and "12" in r for r in g["reasons"])


def test_single_barge_in_is_not_a_storm():
    g = R.grade_fixture(_clean_result(barge_in_count=1))
    assert g["checks"]["barge_in_storm"] is False
    assert g["passed"] is True


def test_barge_in_threshold_is_configurable():
    g = R.grade_fixture(_clean_result(barge_in_count=3), barge_in_storm_threshold=5)
    assert g["checks"]["barge_in_storm"] is False
    assert g["passed"] is True
    g2 = R.grade_fixture(_clean_result(barge_in_count=3), barge_in_storm_threshold=2)
    assert g2["checks"]["barge_in_storm"] is True
    assert g2["passed"] is False


# --- broken OUTPUT path ---

def test_playback_errors_fail():
    g = R.grade_fixture(_clean_result(playback_errors=["ALSA lib pcm.c: write error", "underrun occurred"]))
    assert g["passed"] is False
    assert g["checks"]["playback_clean"] is False
    assert any("PLAYBACK ERROR" in r for r in g["reasons"])


def test_playback_loop_dead_fails():
    g = R.grade_fixture(_clean_result(playback_loop_dead=True))
    assert g["passed"] is False
    assert g["checks"]["playback_clean"] is False
    assert any("PLAYBACK LOOP DEAD" in r for r in g["reasons"])


# --- went silent / went deaf ---

def test_no_spoken_response_fails():
    g = R.grade_fixture(_clean_result(spoken=["Mode: assistant", "Queued reminder"]))
    # Only control lines spoken -> no real answer.
    assert g["passed"] is False
    assert g["checks"]["response_present"] is False
    assert any("WENT SILENT" in r for r in g["reasons"])


def test_empty_spoken_fails_silent():
    g = R.grade_fixture(_clean_result(spoken=[]))
    assert g["passed"] is False
    assert g["checks"]["response_present"] is False


def test_empty_asr_finals_flags_deaf_but_is_not_a_hard_fail():
    # STT garble is the thing under test -- an empty transcript is a visibility
    # flag, not (on its own) a hard fail. With a spoken answer + clean shutdown
    # the run still PASSes overall, but transcript_sane is False and a reason is
    # surfaced.
    g = R.grade_fixture(_clean_result(asr_finals=[]))
    assert g["checks"]["transcript_sane"] is False
    assert any("WENT DEAF" in r for r in g["reasons"])
    assert g["passed"] is True  # not gated on transcript


def test_harness_error_fails():
    g = R.grade_fixture(_clean_result(error="RuntimeError: the engine never opened an input stream"))
    assert g["passed"] is False
    assert any("HARNESS ERROR" in r for r in g["reasons"])


# --- latency is report-only ---

def test_high_latency_is_soft_flag_not_a_fail():
    g = R.grade_fixture(_clean_result(first_audio_latencies=[9.9]), first_audio_budget_sec=3.0)
    assert g["latency_over_budget"] is True
    assert g["first_audio_latency_max"] == 9.9
    assert g["passed"] is True  # latency never hard-fails


# --- run-level grading + report rendering ---

def test_grade_run_counts_and_all_pass():
    results = [_clean_result(fixture="a.wav"), _clean_result(fixture="b.wav")]
    run = R.grade_run(results)
    assert run["n_total"] == 2
    assert run["n_pass"] == 2
    assert run["n_fail"] == 0
    assert run["all_pass"] is True


def test_grade_run_mixed():
    results = [
        _clean_result(fixture="ok.wav"),
        _clean_result(fixture="hang.wav", shutdown_ok=False, shutdown_seconds=8.0),
        _clean_result(fixture="storm.wav", barge_in_count=9),
    ]
    run = R.grade_run(results)
    assert run["n_total"] == 3
    assert run["n_pass"] == 1
    assert run["n_fail"] == 2
    assert run["all_pass"] is False


def test_render_markdown_has_row_per_fixture_and_shows_transcript():
    results = [
        _clean_result(fixture="ok.wav", asr_finals=["hello there friend"]),
        _clean_result(fixture="hang.wav", shutdown_ok=False, shutdown_seconds=8.0),
    ]
    run = R.grade_run(results)
    md = R.render_markdown(run, run_id="20260530-000000")
    assert "ok.wav" in md
    assert "hang.wav" in md
    assert "hello there friend" in md      # STT garble is visible
    assert "SHUTDOWN HANG" in md           # the failure mode is named
    assert "1/2 PASS" in md
    # One table data row per fixture (plus header + separator).
    table_rows = [ln for ln in md.splitlines() if ln.startswith("| ") and "---" not in ln]
    assert any("ok.wav" in r for r in table_rows)
    assert any("hang.wav" in r for r in table_rows)


def test_write_reports_emits_md_and_json(tmp_path):
    run = R.grade_run([_clean_result()])
    paths = R.write_reports(run, tmp_path, run_id="20260530-111111")
    assert paths["markdown"].exists()
    assert paths["json"].exists()
    import json

    data = json.loads(paths["json"].read_text())
    assert data["run_id"] == "20260530-111111"
    assert data["n_total"] == 1


# --- the shutdown-timeout WRAPPER (regression test for the hang) ---

def test_stop_wrapper_returns_ok_for_fast_stop():
    flag = {"called": False}

    def fast_stop():
        flag["called"] = True

    ok, secs = run_stop_with_timeout(fast_stop, timeout=2.0)
    assert ok is True
    assert flag["called"] is True
    assert secs < 1.0


def test_stop_wrapper_flags_a_hanging_stop():
    # A fake stop() that blocks forever (stands in for the ALSA out.write() that
    # never returns). The wrapper MUST time out and report ok=False -- this is
    # the regression test for the shutdown hang without needing a real wedged
    # audio device.
    release = threading.Event()

    def hanging_stop():
        release.wait(timeout=30.0)  # never released within the test window

    start = time.perf_counter()
    ok, secs = run_stop_with_timeout(hanging_stop, timeout=0.3)
    elapsed = time.perf_counter() - start
    assert ok is False                  # FLAGGED as not-completed
    assert secs >= 0.3
    assert elapsed < 5.0                # the wrapper itself returned promptly
    release.set()  # let the daemon worker unwind


def test_stop_wrapper_reports_completion_even_if_stop_raises():
    def raising_stop():
        raise RuntimeError("teardown blew up")

    ok, secs = run_stop_with_timeout(raising_stop, timeout=2.0)
    # The worker finished (the exception is swallowed/logged), so it's NOT a hang.
    assert ok is True


def test_grade_uses_hung_stop_wrapper_result_end_to_end():
    # Tie the wrapper to the grader: a hung stop -> ok=False -> shutdown FAIL.
    release = threading.Event()

    def hanging_stop():
        release.wait(timeout=30.0)

    ok, secs = run_stop_with_timeout(hanging_stop, timeout=0.2)
    result = _clean_result(shutdown_ok=ok, shutdown_seconds=secs, shutdown_timeout=0.2)
    g = R.grade_fixture(result)
    assert g["passed"] is False
    assert g["checks"]["shutdown_clean"] is False
    release.set()


# --- CLI plumbing (pure: no audio/models) ---

def test_resolve_wavs_specific_files(tmp_path):
    import tools.real_usage.__main__ as M

    a = tmp_path / "run-1.wav"
    b = tmp_path / "run-2.wav"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    args = type("A", (), {"wav": [str(a), str(b)], "recordings": None})()
    out = M._resolve_wavs(args)
    assert [p.name for p in out] == ["run-1.wav", "run-2.wav"]


def test_resolve_wavs_directory_prefers_run_glob(tmp_path):
    import tools.real_usage.__main__ as M

    (tmp_path / "run-1.wav").write_bytes(b"x")
    (tmp_path / "other.wav").write_bytes(b"x")  # ignored when run-* exist
    args = type("A", (), {"wav": None, "recordings": [str(tmp_path)]})()
    out = M._resolve_wavs(args)
    assert [p.name for p in out] == ["run-1.wav"]


def test_resolve_wavs_dedups_and_drops_missing(tmp_path):
    import tools.real_usage.__main__ as M

    a = tmp_path / "run-1.wav"
    a.write_bytes(b"x")
    missing = tmp_path / "gone.wav"
    args = type("A", (), {"wav": [str(a), str(a), str(missing)], "recordings": None})()
    out = M._resolve_wavs(args)
    assert [p.name for p in out] == ["run-1.wav"]  # deduped, missing dropped


def test_subprocess_timeout_synthesizes_shutdown_hang(tmp_path, monkeypatch):
    import subprocess

    import tools.real_usage.__main__ as M

    args = type("A", (), {
        "llm": "ollama", "model": None, "fast_model": None, "device": None,
        "output_device": None, "open_mic": False, "response_timeout": 1.0,
        "start_timeout": 20.0, "shutdown_timeout": 8.0,
    })()

    def _raise_timeout(cmd, timeout=None, check=False):
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(M.subprocess, "run", _raise_timeout)
    wav = tmp_path / "run-x.wav"
    wav.write_bytes(b"x")
    res = M._run_one_subprocess(wav, args, timeout=5.0)
    assert res["shutdown_ok"] is False
    assert res["shutdown_seconds"] == 5.0
    g = R.grade_fixture(res)
    assert g["verdict"] == "FAIL"
    assert g["checks"]["shutdown_clean"] is False


# --- EMPTY: a digitally-silent capture is EXCLUDED, not failed ---

def test_is_silent_input_thresholds():
    assert R.is_silent_input(0.0, 0.0) is True
    assert R.is_silent_input(1e-6, 1e-5) is True
    assert R.is_silent_input(0.2, 1.0) is False
    # rms below but peak above (a single click) -> NOT silent
    assert R.is_silent_input(1e-6, 0.5) is False
    # unknown level can't be claimed empty
    assert R.is_silent_input(None, None) is False
    assert R.is_silent_input(0.0, None) is False


def test_empty_result_grades_empty_not_fail():
    g = R.grade_fixture(R.empty_result("run-silent.wav", input_rms=0.0, input_peak=0.0))
    assert g["verdict"] == "EMPTY"
    assert g["empty"] is True
    assert g["passed"] is False
    assert any("EMPTY RECORDING" in r for r in g["reasons"])
    # the vacuous went-deaf / went-silent reasons are replaced by the empty note
    assert not any("WENT DEAF" in r or "WENT SILENT" in r for r in g["reasons"])


def test_silent_input_on_a_full_result_is_empty():
    # Even a result that otherwise looks like a failure is EMPTY when the input was
    # digitally silent -- the pipeline never had audio to work with.
    g = R.grade_fixture(_clean_result(spoken=[], asr_finals=[], input_rms=0.0, input_peak=0.0))
    assert g["verdict"] == "EMPTY"
    assert g["empty"] is True


def test_loud_input_is_graded_normally():
    g = R.grade_fixture(_clean_result(input_rms=0.15, input_peak=1.0))
    assert g["empty"] is False
    assert g["verdict"] == "PASS"


def test_grade_run_excludes_empties_from_pass_fail():
    results = [
        _clean_result(fixture="ok.wav"),
        _clean_result(fixture="fail.wav", spoken=[]),                 # a real FAIL
        R.empty_result("silent.wav", input_rms=0.0, input_peak=0.0),  # EMPTY
    ]
    run = R.grade_run(results)
    assert run["n_total"] == 3
    assert run["n_empty"] == 1
    assert run["n_pass"] == 1
    assert run["n_fail"] == 1          # the empty is NOT counted as a fail
    assert run["all_pass"] is False


def test_all_pass_true_with_empties_present():
    results = [_clean_result(fixture="ok.wav"), R.empty_result("silent.wav")]
    run = R.grade_run(results)
    assert run["n_empty"] == 1
    assert run["n_pass"] == 1
    assert run["n_fail"] == 0
    assert run["all_pass"] is True     # one pass + one excluded empty == all (non-empty) pass


def test_all_empty_run_passes_vacuously():
    # A batch where EVERY recording is an excluded empty has NO failures, so the
    # run passes vacuously (exit 0) -- it must not report failure just because
    # nothing was validated.
    run = R.grade_run([R.empty_result("a.wav"), R.empty_result("b.wav")])
    assert run["n_empty"] == 2
    assert run["n_pass"] == 0
    assert run["n_fail"] == 0
    assert run["all_pass"] is True


def test_render_markdown_marks_empty_and_separates_counts():
    results = [
        _clean_result(fixture="ok.wav"),
        R.empty_result("silent.wav", input_rms=0.0, input_peak=0.0),
    ]
    run = R.grade_run(results)
    md = R.render_markdown(run, run_id="20260603-000000")
    assert "EMPTY" in md
    assert "silent.wav" in md
    assert "1 EMPTY" in md             # the header breakdown
    assert "1/2 PASS" in md


def test_print_summary_marks_empty(capsys):
    run = R.grade_run([_clean_result(fixture="ok.wav"),
                       R.empty_result("silent.wav", input_rms=0.0, input_peak=0.0)])
    R.print_summary(run)
    out = capsys.readouterr().out
    assert "[EMPTY] silent.wav" in out
    assert "1 EMPTY" in out


# --- inventory / run-history overview (pure render) ---

def test_inventory_render_flags_empty_and_counts():
    from tools.real_usage import inventory as INV

    rows = [
        {"run_id": "run-20260602-231913", "when": "2026-06-02 23:19:13", "duration_sec": 45.2,
         "turns": 6, "llm_requests": 11, "errors": 0, "stuck_hints": 0, "has_wav": True,
         "wav_rms": 0.19, "wav_peak": 1.0, "silent": False, "snippet": "the / yeah"},
        {"run_id": "run-20260603-101952", "when": "2026-06-03 10:19:52", "duration_sec": 17.1,
         "turns": 5, "llm_requests": 4, "errors": 0, "stuck_hints": 1, "has_wav": True,
         "wav_rms": 0.0, "wav_peak": 0.0, "silent": True, "snippet": ""},
        {"run_id": "run-20260602-203131", "when": "2026-06-02 20:31:31", "duration_sec": 3.0,
         "turns": 0, "llm_requests": 0, "errors": 0, "stuck_hints": 0, "has_wav": False,
         "snippet": ""},
    ]
    md = INV.render_inventory_markdown(rows)
    assert "3 runs" in md
    assert "run-20260603-101952" in md
    assert "EMPTY" in md
    assert "no-wav" in md
    assert "prune" in md.lower()


def test_inventory_render_sanitizes_parse_error():
    from tools.real_usage import inventory as INV

    rows = [{"run_id": "run-bad", "parse_error": "Expecting value | line 1\ncolumn 2"}]
    md = INV.render_inventory_markdown(rows)
    data_rows = [ln for ln in md.splitlines() if ln.startswith("| run-bad")]
    assert len(data_rows) == 1
    # 10 columns -> exactly 11 pipe delimiters; an unsanitised '|' would add one.
    assert data_rows[0].count("|") == 11
    assert "PARSE ERR" in md


def test_inventory_empty_wavs_helper():
    from tools.real_usage import inventory as INV

    rows = [
        {"run_id": "a", "has_wav": True, "silent": True},
        {"run_id": "b", "has_wav": True, "silent": False},
        {"run_id": "c", "has_wav": False, "silent": False},
    ]
    assert INV.empty_wavs(rows) == ["a"]


def test_inventory_fmt_when():
    from tools.real_usage import inventory as INV

    assert INV._fmt_when("run-20260602-231913") == "2026-06-02 23:19:13"
    assert INV._fmt_when("weird") == "weird"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
