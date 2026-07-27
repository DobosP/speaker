"""Crash visibility in the staged test harness (tools/testing).

Regression tests for the 2026-07 blindness incident: sherpa-onnx called C++
``exit(-1)`` on a TTS config mismatch, pytest died with rc 255 and ZERO
recorded outcomes, and the harness reported PASS for 10 days because its
verdict was built purely from counted outcomes. These pin the three closed
mechanisms:

1. ``ReportParser`` turns (non-zero rc + zero counted failures) and (rc 0 +
   zero outcomes) into explicit ``crash_evidence`` + a synthetic failure.
2. ``summary._verdict`` reports FAIL on that evidence (and never PASS on a
   red exit code), while the allowed-to-fail / preflight-skip lanes keep
   their designed non-red verdicts.
3. ``find_interrupted_runs`` flags an orphaned ``tests-*.started.json``
   marker (written eagerly by tests/conftest.py, removed only after the
   summary lands) as evidence of a hard interpreter death.

Plus one e2e proof that a real ``os._exit(255)`` inside a pytest subprocess
comes out of ``PytestRunner`` as a FAIL with the returncode in the report.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.testing.reports import ReportParser
from tools.testing.summary import LLMSummary, _verdict, find_interrupted_runs


def _parse(tmp_path: Path, *, returncode: int, stdout: str = "", junit: str | None = None):
    stdout_path = tmp_path / "stdout.txt"
    stdout_path.write_text(stdout, encoding="utf-8")
    junit_path = tmp_path / "junit.xml"
    if junit is not None:
        junit_path.write_text(junit, encoding="utf-8")
    return ReportParser().parse(
        stage="unit",
        returncode=returncode,
        duration_sec=1.0,
        stdout_path=stdout_path,
        junit_path=junit_path,
    )


GREEN_JUNIT = (
    '<testsuite tests="2" failures="0" errors="0" skipped="0">'
    '<testcase classname="tests.test_x" name="test_a" time="0.1"/>'
    '<testcase classname="tests.test_x" name="test_b" time="0.1"/>'
    "</testsuite>"
)


# --- 1+2: parser evidence + verdict ----------------------------------------


def test_native_crash_rc_and_no_outcomes_is_fail(tmp_path):
    # The incident shape: pytest killed mid-run (native exit 255) -> no junit,
    # no "N passed" line, nothing counted. Must be a loud FAIL, not PASS.
    parsed = _parse(
        tmp_path,
        returncode=255,
        stdout="tests/test_replay_recorded.py::test_replay ",
    )
    assert parsed.summary["crash_evidence"]
    assert "255" in str(parsed.summary["crash_evidence"])
    assert parsed.failures and parsed.failures[0]["type"] == "crash"
    verdict = _verdict(parsed.summary)
    assert verdict.startswith("FAIL")
    assert "255" in verdict


def test_zero_outcomes_with_rc_zero_is_fail(tmp_path):
    # An "all green" run that never ran anything proves nothing.
    parsed = _parse(tmp_path, returncode=0, stdout="no tests ran in 0.01s\n")
    assert parsed.summary["crash_evidence"]
    assert _verdict(parsed.summary).startswith("FAIL")


def test_green_run_stays_pass(tmp_path):
    parsed = _parse(tmp_path, returncode=0, junit=GREEN_JUNIT)
    assert "crash_evidence" not in parsed.summary
    assert _verdict(parsed.summary) == "PASS"


def test_counted_failures_still_fail_and_carry_no_crash_evidence(tmp_path):
    junit = (
        '<testsuite tests="1" failures="1" errors="0" skipped="0">'
        '<testcase classname="tests.test_x" name="test_a" time="0.1">'
        '<failure message="boom">boom</failure></testcase></testsuite>'
    )
    parsed = _parse(tmp_path, returncode=1, junit=junit)
    assert "crash_evidence" not in parsed.summary  # ordinary red, not a crash
    assert _verdict(parsed.summary) == "FAIL"


def test_allowed_to_fail_lane_keeps_its_design_on_crash(tmp_path):
    parsed = _parse(tmp_path, returncode=255, stdout="")
    parsed.summary["allowed_to_fail"] = True
    assert _verdict(parsed.summary) == "PASS (allowed)"


def test_preflight_skip_is_not_red():
    summary = {
        "stage": "live",
        "returncode": 0,
        "preflight_failed": True,
        "allowed_to_fail": True,
    }
    assert _verdict(summary) == "SKIPPED (preflight)"


def test_red_returncode_never_passes_even_without_parser_evidence():
    # Defense in depth: even a summary that skipped the parser entirely must
    # not turn a non-zero exit code into PASS.
    assert _verdict({"stage": "unit", "returncode": 2}).startswith("FAIL")


# --- 2: rendered reports ----------------------------------------------------


def test_stage_report_shouts_native_crash(tmp_path):
    parsed = _parse(tmp_path, returncode=255, stdout="last line before death")
    parsed.summary["interrupted_test_logs"] = [
        {"run_id": "20260727-000000", "marker": "logs/tests/tests-20260727-000000.started.json"}
    ]
    out = tmp_path / "llm-summary.md"
    LLMSummary().write_stage(path=out, summary=parsed.summary, failures=parsed.failures)
    text = out.read_text(encoding="utf-8")
    assert "NATIVE CRASH SUSPECTED" in text
    assert "returncode 255" in text
    assert "never finished" in text  # the orphaned started-marker evidence


def test_run_index_goes_fail_on_crashed_stage(tmp_path):
    crashed = _parse(tmp_path, returncode=255, stdout="").summary
    green = {"stage": "core", "returncode": 0, "passed": 5, "total": 5, "duration_sec": 1.0}
    out = tmp_path / "run.md"
    LLMSummary().write_run_index(path=out, run_summaries=[green, crashed])
    text = out.read_text(encoding="utf-8")
    assert "**Overall: FAIL**" in text
    assert "NATIVE CRASH SUSPECTED" in text


# --- 3: started-marker / missing-summary detection ---------------------------


def _write_marker(log_dir: Path, run_id: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    marker = log_dir / f"tests-{run_id}.started.json"
    marker.write_text(json.dumps({"run_id": run_id, "pid": 12345}), encoding="utf-8")
    return marker


def test_orphaned_started_marker_is_flagged(tmp_path):
    _write_marker(tmp_path, "20260727-101010")
    orphans = find_interrupted_runs(tmp_path)
    assert [o["run_id"] for o in orphans] == ["20260727-101010"]
    assert orphans[0]["pid"] == 12345  # marker contents ride along


def test_finished_run_is_not_flagged(tmp_path):
    _write_marker(tmp_path, "20260727-101010")
    (tmp_path / "tests-20260727-101010.summary.json").write_text("{}", encoding="utf-8")
    assert find_interrupted_runs(tmp_path) == []


def test_since_filter_ignores_older_orphans(tmp_path):
    import os

    marker = _write_marker(tmp_path, "20260101-000000")
    old = 1_600_000_000
    os.utime(marker, (old, old))
    assert find_interrupted_runs(tmp_path, since=old + 10) == []
    assert find_interrupted_runs(tmp_path, since=None)


def test_missing_log_dir_is_empty_not_error(tmp_path):
    assert find_interrupted_runs(tmp_path / "nope") == []


# --- e2e: a real native death through PytestRunner ---------------------------


@pytest.mark.e2e
def test_runner_reports_fail_on_hard_interpreter_death(tmp_path):
    """Subprocess the real pytest on a test whose fixture os._exit(255)s --
    the exact blindness shape: rc 255, no junit, no outcome lines. The stage
    summary must carry crash evidence and a FAIL verdict, not a silent PASS."""
    from tools.testing.artifacts import ArtifactStore
    from tools.testing.runner import PytestRunner
    from tools.testing.stages import TestStage

    crash_file = tmp_path / "test_fake_native_crash.py"
    crash_file.write_text(
        "import os\n"
        "import pytest\n"
        "\n"
        "@pytest.fixture\n"
        "def native_death():\n"
        "    os._exit(255)  # simulates sherpa-onnx's C++ exit(-1)\n"
        "\n"
        "def test_never_reports(native_death):\n"
        "    pass\n",
        encoding="utf-8",
    )
    stage = TestStage(
        name="fakecrash",
        purpose="harness regression: native death must be visible",
        paths=(str(crash_file),),
        extra_args=("-q", "-p", "no:cacheprovider"),
    )
    runner = PytestRunner(ArtifactStore(base_dir=tmp_path / "reports"))
    summary = runner.run_stage(stage)

    assert int(summary["returncode"]) == 255
    assert summary["crash_evidence"]
    assert _verdict(summary).startswith("FAIL")
    report = Path(str(summary["stdout_path"])).parent / "llm-summary.md"
    assert "NATIVE CRASH SUSPECTED" in report.read_text(encoding="utf-8")
