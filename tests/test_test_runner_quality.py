"""Tests for the staged test runner and reporting infrastructure."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.testing.duplicates import DuplicateScanner
from scripts.testing.reports import ReportParser
from scripts.testing.stages import StageRegistry

pytestmark = [pytest.mark.smoke, pytest.mark.dev]


def test_stage_registry_has_expected_quality_gates():
    registry = StageRegistry.default()

    assert {"smoke", "dev", "full", "discovery", "backend", "all"}.issubset(
        set(registry.names())
    )
    assert registry.get("full").markers is not None
    assert "not discovery" in registry.get("full").markers
    assert not registry.get("full").allow_failures
    assert registry.get("discovery").allow_failures
    assert "tests/test_real_usage_open_data.py" not in registry.get("discovery").paths


def test_stage_pytest_args_are_predictable():
    stage = StageRegistry.default().get("dev")
    args = stage.pytest_args(maxfail=3)

    assert "tests/test_conversation_simulation.py" in args
    assert "-m" in args
    assert "dev" in args
    assert "--maxfail=3" in args


def test_report_parser_extracts_junit_failures(tmp_path: Path):
    stdout_path = tmp_path / "stdout.txt"
    junit_path = tmp_path / "junit.xml"
    stdout_path.write_text("1 failed, 1 passed in 0.01s", encoding="utf-8")
    junit_path.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuite name="pytest" tests="2" failures="1" errors="0" skipped="0">
  <testcase classname="tests.test_audio" name="test_ok" time="0.1" />
  <testcase classname="tests.test_audio" name="test_bad" time="0.2">
    <failure message="boom">traceback text</failure>
  </testcase>
</testsuite>
""",
        encoding="utf-8",
    )

    parsed = ReportParser().parse(
        stage="unit",
        returncode=1,
        duration_sec=0.3,
        stdout_path=stdout_path,
        junit_path=junit_path,
    )

    assert parsed.summary["passed"] == 1
    assert parsed.summary["failed"] == 1
    assert parsed.summary["failure_count"] == 1
    assert parsed.summary["failure_categories"] == {"unknown": 1}
    assert parsed.summary["failure_solvability"] == {"solvable": 1}
    assert parsed.summary["quality_metrics"]["solvable_failure_count"] == 1
    assert parsed.failures[0]["nodeid"] == "tests.test_audio.test_bad"
    assert parsed.failures[0]["area"] == "audio"
    assert "category" in parsed.failures[0]
    assert "cause" in parsed.failures[0]


def test_report_parser_classifies_ungated_background_as_unsolvable(tmp_path: Path):
    stdout_path = tmp_path / "stdout.txt"
    junit_path = tmp_path / "junit.xml"
    stdout_path.write_text("1 failed in 0.01s", encoding="utf-8")
    junit_path.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuite name="pytest" tests="1" failures="1" errors="0" skipped="0">
  <testcase classname="tests.test_real_usage_open_data" name="test_real_usage_callback_expectations[background_jackson_0_distant_speech]" time="0.1">
    <failure message="background speech failed">ungated_audio_has_no_intent_signal</failure>
  </testcase>
</testsuite>
""",
        encoding="utf-8",
    )

    parsed = ReportParser().parse(
        stage="full",
        returncode=1,
        duration_sec=0.1,
        stdout_path=stdout_path,
        junit_path=junit_path,
    )

    failure = parsed.failures[0]
    assert failure["area"] == "real_usage"
    assert failure["category"] == "unsolvable_without_intent_gate"
    assert failure["solvable"] is False
    assert "wakeword" in failure["suggested_fix"]
    assert parsed.summary["failure_categories"] == {"unsolvable_without_intent_gate": 1}
    assert parsed.summary["failure_solvability"] == {"unsolvable": 1}
    assert parsed.summary["quality_metrics"]["false_accept_failures"] == 1
    assert parsed.summary["quality_metrics"]["unsolvable_failure_count"] == 1


def test_duplicate_scanner_reports_audio_corpora():
    report = DuplicateScanner().scan()

    assert "audio_corpora" in report
    corpora = report["audio_corpora"]
    if "failure_discovery" in corpora:
        assert corpora["failure_discovery"]["duplicate_audio_hash_groups"] == []
    if "real_usage_full" in corpora:
        real_usage = corpora["real_usage_full"]
        assert real_usage["case_count"] >= 220
        assert real_usage["duplicate_names"] == {}
        assert real_usage["duplicate_audio_hash_groups"] == []
    if "virtual_real_world" in corpora:
        virtual = corpora["virtual_real_world"]
        assert virtual["case_count"] >= 100
        assert virtual["duplicate_names"] == {}
        assert virtual["duplicate_audio_hash_groups"] == []


def test_runner_reports_are_json_serializable():
    report = DuplicateScanner().scan()
    encoded = json.dumps(report)

    assert "test_functions" in encoded
