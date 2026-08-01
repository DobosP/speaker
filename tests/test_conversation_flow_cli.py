from __future__ import annotations

import json

import pytest

import tools.conversation_eval.__main__ as conversation_cli


def test_flow_gate_composes_fail_closed_without_mutating_v4_semantics():
    report = {
        "scenario_set_version": 4,
        "gate": {
            "semantic_pass": True,
            "coverage_ok": True,
            "passed": True,
        },
    }
    flow = {"passed": False, "gate": {"passed": False}}

    conversation_cli._attach_flow_acceptance(report, flow)

    assert report["scenario_set_version"] == 4
    assert report["gate"]["semantic_pass"] is True
    assert report["gate"]["coverage_ok"] is True
    assert report["gate"]["v4_gate_pass"] is True
    assert report["gate"]["flow_acceptance_required"] is True
    assert report["gate"]["flow_acceptance_pass"] is False
    assert report["gate"]["passed"] is False
    assert report["flow_acceptance"] is flow


def test_flow_gate_cannot_rescue_red_v4_gate():
    report = {"gate": {"semantic_pass": False, "passed": False}}
    flow = {"passed": True, "gate": {"passed": True}}

    conversation_cli._attach_flow_acceptance(report, flow)

    assert report["gate"]["v4_gate_pass"] is False
    assert report["gate"]["flow_acceptance_pass"] is True
    assert report["gate"]["passed"] is False


@pytest.mark.parametrize(
    "flow",
    (
        {"passed": True, "gate": {"passed": False}},
        {"passed": False, "gate": {"passed": True}},
    ),
)
def test_flow_gate_rejects_redundant_pass_mismatch(flow):
    report = {"gate": {"passed": True}}

    conversation_cli._attach_flow_acceptance(report, flow)

    assert report["gate"]["flow_acceptance_pass"] is False
    assert report["gate"]["passed"] is False


def test_flow_gate_rejects_missing_nested_gate():
    with pytest.raises(ValueError, match="flow report is missing its gate"):
        conversation_cli._attach_flow_acceptance(
            {"gate": {"passed": True}},
            {"passed": True},
        )


def test_flag_off_keeps_v4_report_and_console_flow_free(tmp_path, capsys):
    output = tmp_path / "v4-only.json"

    exit_code = conversation_cli.main(
        [
            "--runs",
            "1",
            "--scenario",
            "simple_qa",
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    console = capsys.readouterr().out

    assert exit_code == 1  # A partial v4 diagnostic is intentionally not gate-green.
    assert report["scenario_set_version"] == 4
    assert "flow_acceptance" not in report
    assert "flow_acceptance_required" not in report["gate"]
    assert "flow_acceptance_pass" not in report["gate"]
    assert "v4_gate_pass" not in report["gate"]
    assert "flow:" not in console
    assert "flow scope:" not in console


@pytest.mark.parametrize(
    "argv, expected",
    (
        (("--flow-acceptance", "--runs", "1"), "requires exactly 3 runs"),
        (
            ("--flow-acceptance", "--scenario", "simple_qa"),
            "requires the complete v4 scenario set",
        ),
        (
            ("--flow-acceptance", "--mode", "ollama"),
            "requires --mode deterministic",
        ),
        (
            ("--flow-acceptance", "--include-local-config"),
            "refuses --include-local-config",
        ),
    ),
)
def test_flow_cli_rejects_partial_evidence(argv, expected, capsys):
    with pytest.raises(SystemExit) as raised:
        conversation_cli.main(list(argv))

    assert raised.value.code == 2
    assert expected in capsys.readouterr().err
