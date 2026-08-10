from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from types import SimpleNamespace

import pytest

from tools.interrupt_suite import (
    STRATEGIES,
    DiagnosticOutcome,
    DiagnosticStatus,
    classify_outcome,
    coupled,
    main,
    run_cell,
)


def _summary_row(
    label: str,
    *,
    self_int: int = 0,
    is_coupled: bool = True,
    error: object = None,
) -> dict[str, object]:
    return {
        "label": label,
        "self_int": self_int,
        "coupled": is_coupled,
        "error": error,
    }


def _cell(
    *,
    self_int: object = 0,
    vad: object = 1,
    peak: object = 0.2,
    label: object = "child-controlled-label",
    error: object = None,
) -> dict[str, object]:
    return {
        "label": label,
        "self_interruptions": self_int,
        "vad_flagged_during_play": vad,
        "peak_playback_level": peak,
        "error": error,
    }


def _capture_publisher():
    published: list[tuple[str, dict[str, object]]] = []

    def publish(path: str, report: dict[str, object]) -> None:
        published.append((path, deepcopy(report)))

    return published, publish


def test_strategy_matrix_uses_backend_neutral_configured_aec_label():
    labels = [strategy["label"] for strategy in STRATEGIES]
    assert labels[-1] == "configured-aec"
    assert all("dtln" not in label.lower() for label in labels)
    assert STRATEGIES[-1]["aec"] == "on"


def test_classify_candidate_lists_every_unique_label_lexically_and_keeps_partial():
    outcome = classify_outcome(
        [
            _summary_row("zeta"),
            _summary_row("alpha"),
            _summary_row("alpha"),
            _summary_row("nonzero", self_int=2),
            {"label": "failed", "self_int": None, "error": "timeout"},
        ]
    )

    assert outcome.status is DiagnosticStatus.QUIET_CONTROL_CANDIDATE
    assert outcome.exit_code == 0
    assert outcome.candidate_labels == ("alpha", "zeta")
    assert outcome.valid_coupled_cells == 4
    assert outcome.uncoupled_zero_cells == 0
    assert outcome.error_cells == 1
    assert outcome.matrix_partial is True
    assert outcome.live_validation_required is True


def test_classify_no_safe_candidate_requires_valid_coupled_evidence():
    outcome = classify_outcome(
        [
            _summary_row("one", self_int=1),
            _summary_row("two", self_int=4),
            _summary_row("uncoupled", is_coupled=False),
        ]
    )

    assert outcome.status is DiagnosticStatus.NO_SAFE_CANDIDATE
    assert outcome.exit_code == 1
    assert outcome.candidate_labels == ()
    assert outcome.valid_coupled_cells == 2
    assert outcome.uncoupled_zero_cells == 1
    assert outcome.error_cells == 0
    assert outcome.matrix_partial is False


def test_classify_valid_uncoupled_zero_is_inconclusive_but_not_an_error():
    outcome = classify_outcome([_summary_row("quiet", is_coupled=False)])

    assert outcome.status is DiagnosticStatus.INCONCLUSIVE
    assert outcome.reason == "no-valid-coupled-cell"
    assert outcome.exit_code == 2
    assert outcome.valid_coupled_cells == 0
    assert outcome.uncoupled_zero_cells == 1
    assert outcome.error_cells == 0
    assert outcome.matrix_partial is False


class _DictSubclass(dict):
    pass


class _StrSubclass(str):
    pass


@pytest.mark.parametrize(
    "bad_row",
    [
        None,
        _DictSubclass(label="bad", self_int=0, coupled=True, error=None),
        {"label": "explicit", "self_int": 0, "coupled": True, "error": "boom"},
        {"label": _StrSubclass("label"), "self_int": 0, "coupled": True, "error": None},
        {"label": "", "self_int": 0, "coupled": True, "error": None},
        {"label": "bool", "self_int": False, "coupled": True, "error": None},
        {"label": "negative", "self_int": -1, "coupled": True, "error": None},
        {"label": "coerced", "self_int": 0, "coupled": 1, "error": None},
    ],
)
def test_classify_counts_each_explicit_or_structurally_malformed_row(bad_row):
    outcome = classify_outcome([_summary_row("valid"), bad_row])

    assert outcome.status is DiagnosticStatus.QUIET_CONTROL_CANDIDATE
    assert outcome.candidate_labels == ("valid",)
    assert outcome.error_cells == 1
    assert outcome.matrix_partial is True


def _candidate_outcome(**overrides: object) -> DiagnosticOutcome:
    values: dict[str, object] = {
        "status": DiagnosticStatus.QUIET_CONTROL_CANDIDATE,
        "reason": "coupled-zero-self-interruption-observed",
        "candidate_labels": ("candidate",),
        "valid_coupled_cells": 1,
        "uncoupled_zero_cells": 0,
        "error_cells": 0,
        "matrix_partial": False,
        "live_validation_required": True,
    }
    values.update(overrides)
    return DiagnosticOutcome(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "overrides",
    [
        {"status": "quiet-control-candidate"},
        {"reason": "wrong"},
        {"candidate_labels": ["candidate"]},
        {"candidate_labels": ("z", "a"), "valid_coupled_cells": 2},
        {"candidate_labels": ("a", "a"), "valid_coupled_cells": 2},
        {"valid_coupled_cells": True},
        {"error_cells": 1, "matrix_partial": False},
        {"live_validation_required": False},
        {"candidate_labels": (), "valid_coupled_cells": 1},
    ],
)
def test_outcome_rejects_incoherent_or_coercible_candidate_state(overrides):
    with pytest.raises((TypeError, ValueError)):
        _candidate_outcome(**overrides)


@pytest.mark.parametrize(
    "values",
    [
        {
            "status": DiagnosticStatus.INCONCLUSIVE,
            "reason": "report-publication-failed",
            "candidate_labels": (),
            "valid_coupled_cells": 1,
            "uncoupled_zero_cells": 0,
            "error_cells": 0,
            "matrix_partial": False,
        },
        {
            "status": DiagnosticStatus.INCONCLUSIVE,
            "reason": "invalid-mic-selection",
            "candidate_labels": (),
            "valid_coupled_cells": 0,
            "uncoupled_zero_cells": 1,
            "error_cells": 1,
            "matrix_partial": True,
        },
        {
            "status": DiagnosticStatus.INCONCLUSIVE,
            "reason": "invalid-mic-selection",
            "candidate_labels": (),
            "valid_coupled_cells": 0,
            "uncoupled_zero_cells": 0,
            "error_cells": 0,
            "matrix_partial": False,
        },
    ],
)
def test_outcome_rejects_inconclusive_states_without_required_error_metadata(values):
    with pytest.raises(ValueError):
        DiagnosticOutcome(**values)  # type: ignore[arg-type]


def test_outcome_serialization_and_exit_code_revalidate_class_bound_state():
    outcome = _candidate_outcome()
    object.__setattr__(outcome, "_validate", lambda: None)
    object.__setattr__(outcome, "status", "quiet-control-candidate")

    with pytest.raises(TypeError):
        outcome.as_dict()
    with pytest.raises(TypeError):
        _ = outcome.exit_code


@pytest.mark.parametrize(
    ("vad", "peak", "expected"),
    [
        (1, 0.1, True),
        (0, 0.1, False),
        (1, 0, False),
        (1, 10**500, True),
        (True, 0.1, None),
        (1, True, None),
        (1, float("nan"), None),
        (1, float("inf"), None),
        (-1, 0.1, None),
        (1, -0.1, None),
        (None, 0.1, None),
    ],
)
def test_coupling_is_exact_finite_and_never_coerces(vad, peak, expected):
    assert coupled({"vad_flagged": vad, "peak_play": peak}) is expected


def test_run_cell_uses_exact_echo_probe_arguments_and_accepts_object_json():
    calls: list[tuple[list[str], dict[str, object]]] = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout='noise before json {"self_interruptions": 0}',
            stderr="",
        )

    result = run_cell(
        "Mic Device",
        STRATEGIES[-1],
        4,
        12.5,
        runner=runner,
        python_executable="/exact/python",
        output_device="Speaker Device",
    )

    assert result == {"self_interruptions": 0}
    command, kwargs = calls[0]
    assert command[:3] == ["/exact/python", "-m", "tools.echo_probe"]
    assert command[command.index("--label") + 1] == "Mic Device/configured-aec"
    assert command[command.index("--input-device") + 1] == "Mic Device"
    assert command[command.index("--output-device") + 1] == "Speaker Device"
    assert command[command.index("--aec") + 1] == "on"
    assert command[command.index("--sentences") + 1] == "4"
    assert kwargs == {"capture_output": True, "text": True, "timeout": 12.5}


@pytest.mark.parametrize(
    ("process", "expected_error"),
    [
        (
            SimpleNamespace(
                returncode=1, stdout='{"self_interruptions": 0}', stderr=""
            ),
            "subprocess-failed",
        ),
        (
            SimpleNamespace(
                returncode=True, stdout='{"self_interruptions": 0}', stderr=""
            ),
            "subprocess-failed",
        ),
        (SimpleNamespace(returncode=0, stdout=None, stderr=""), "invalid-stdout"),
        (SimpleNamespace(returncode=0, stdout="no object", stderr="secret"), "no-json"),
        (SimpleNamespace(returncode=0, stdout="{broken", stderr=""), "invalid-json"),
        (SimpleNamespace(returncode=0, stdout="[1, 2]", stderr=""), "no-json"),
        (
            SimpleNamespace(returncode=0, stdout='{"value": NaN}', stderr=""),
            "invalid-json",
        ),
    ],
)
def test_run_cell_rejects_failed_or_malformed_child_without_output_tails(
    process, expected_error
):
    result = run_cell(
        "Mic",
        STRATEGIES[0],
        1,
        2.0,
        runner=lambda *_args, **_kwargs: process,
    )

    assert result == {"label": "Mic/coherence-confirm1", "error": expected_error}
    assert all("tail" not in key for key in result)


@pytest.mark.parametrize(
    "fault", [subprocess.TimeoutExpired("probe", 1), RuntimeError("boom")]
)
def test_run_cell_converts_runner_faults_to_detail_free_errors(fault):
    def runner(*_args, **_kwargs):
        raise fault

    result = run_cell("Mic", STRATEGIES[0], 1, 2.0, runner=runner)
    expected = (
        "timeout"
        if isinstance(fault, subprocess.TimeoutExpired)
        else "subprocess-failed"
    )
    assert result == {"label": "Mic/coherence-confirm1", "error": expected}


def test_main_publishes_closed_candidate_outcome_and_exact_live_gate(capsys):
    published, publisher = _capture_publisher()

    def cell_probe(_mic, strategy, _sentences, _timeout):
        zero = strategy["label"] in {"coherence-confirm2", "configured-aec"}
        return _cell(self_int=0 if zero else 3)

    rc = main(
        ["--mics", "alc285", "--out", "ignored.json"],
        mic_probe=lambda _device: (True, 0.01),
        cell_probe=cell_probe,
        report_publisher=publisher,
    )

    assert rc == 0
    assert len(published) == 1
    path, report = published[0]
    assert path == "ignored.json"
    assert report["diagnostic_outcome"] == {
        "status": "quiet-control-candidate",
        "reason": "coupled-zero-self-interruption-observed",
        "candidate_labels": [
            "ALC285 Analog/coherence-confirm2",
            "ALC285 Analog/configured-aec",
        ],
        "valid_coupled_cells": 5,
        "uncoupled_zero_cells": 0,
        "error_cells": 0,
        "matrix_partial": False,
        "live_validation_required": True,
    }
    assert all(raw["label"].startswith("ALC285 Analog/") for raw in report["raw"])
    output = capsys.readouterr().out
    assert "================ DIAGNOSTIC OUTCOME ================" in output
    assert "./live.sh" in output
    assert "python -m tools.live_audio_ab logs/runs/run-<id>.txt" in output
    forbidden = ("headphone", "dtln-aec", "works", "recommended")
    assert all(word not in output.lower() for word in forbidden)


def test_main_candidate_survives_other_mic_error_as_explicit_partial():
    published, publisher = _capture_publisher()

    def mic_probe(device):
        return (False, 0.0) if device == "plughw:2,0" else (True, 0.01)

    rc = main(
        ["--mics", "alc285,at2020", "--out", "ignored.json"],
        mic_probe=mic_probe,
        cell_probe=lambda *_args: _cell(self_int=0),
        report_publisher=publisher,
    )

    assert rc == 0
    outcome = published[0][1]["diagnostic_outcome"]
    assert outcome["status"] == "quiet-control-candidate"
    assert outcome["error_cells"] == 1
    assert outcome["matrix_partial"] is True


def test_main_returns_one_when_valid_coupled_cells_all_self_interrupt():
    published, publisher = _capture_publisher()
    rc = main(
        ["--mics", "alc285", "--out", "ignored.json"],
        mic_probe=lambda _device: (True, 0.01),
        cell_probe=lambda *_args: _cell(self_int=2),
        report_publisher=publisher,
    )

    assert rc == 1
    outcome = published[0][1]["diagnostic_outcome"]
    assert outcome["status"] == "no-safe-candidate"
    assert outcome["valid_coupled_cells"] == len(STRATEGIES)
    assert outcome["candidate_labels"] == []


def test_main_valid_uncoupled_zero_is_clean_but_inconclusive():
    published, publisher = _capture_publisher()
    rc = main(
        ["--mics", "alc285", "--out", "ignored.json"],
        mic_probe=lambda _device: (True, 0.01),
        cell_probe=lambda *_args: _cell(self_int=0, vad=0),
        report_publisher=publisher,
    )

    assert rc == 2
    outcome = published[0][1]["diagnostic_outcome"]
    assert outcome["status"] == "inconclusive"
    assert outcome["uncoupled_zero_cells"] == len(STRATEGIES)
    assert outcome["error_cells"] == 0
    assert outcome["matrix_partial"] is False


@pytest.mark.parametrize("selection", ["", "unknown", "alc285,unknown"])
def test_main_invalid_mic_selection_is_preflight_inconclusive(selection):
    published, publisher = _capture_publisher()
    calls: list[str] = []

    rc = main(
        ["--mics", selection, "--out", "ignored.json"],
        mic_probe=lambda device: calls.append(device),  # type: ignore[arg-type,return-value]
        cell_probe=lambda *_args: pytest.fail("cell probe must not run"),
        report_publisher=publisher,
    )

    assert rc == 2
    assert calls == []
    outcome = published[0][1]["diagnostic_outcome"]
    assert outcome["reason"] == "invalid-mic-selection"
    assert outcome["candidate_labels"] == []
    assert outcome["error_cells"] == 1
    assert outcome["matrix_partial"] is True


@pytest.mark.parametrize(
    "liveness",
    [
        (False, 0.0),
        None,
        (True, float("nan")),
        (True, -0.1),
        (True, 10**500),
        (1, 0.1),
    ],
)
def test_main_dead_or_malformed_liveness_is_inconclusive(liveness):
    published, publisher = _capture_publisher()
    rc = main(
        ["--mics", "alc285", "--out", "ignored.json"],
        mic_probe=lambda _device: liveness,
        cell_probe=lambda *_args: pytest.fail("cell probe must not run"),
        report_publisher=publisher,
    )

    assert rc == 2
    outcome = published[0][1]["diagnostic_outcome"]
    assert outcome["status"] == "inconclusive"
    assert outcome["valid_coupled_cells"] == 0
    assert outcome["error_cells"] == 1
    assert outcome["matrix_partial"] is True


def test_main_malformed_cells_fail_closed_and_raw_label_is_canonical():
    published, publisher = _capture_publisher()
    rc = main(
        ["--mics", "alc285", "--out", "ignored.json"],
        mic_probe=lambda _device: (True, 0.01),
        cell_probe=lambda *_args: _cell(
            self_int=0,
            peak=float("nan"),
            label="forged/label",
        ),
        report_publisher=publisher,
    )

    assert rc == 2
    report = published[0][1]
    assert report["diagnostic_outcome"]["error_cells"] == len(STRATEGIES)
    assert all(raw["label"].startswith("ALC285 Analog/") for raw in report["raw"])


def test_main_publication_failure_clears_candidate_authority_and_returns_two(capsys):
    attempted: list[dict[str, object]] = []

    def fail_publish(_path, report):
        attempted.append(deepcopy(report))
        raise OSError("private path")

    rc = main(
        ["--mics", "alc285", "--out", "ignored.json"],
        mic_probe=lambda _device: (True, 0.01),
        cell_probe=lambda *_args: _cell(self_int=0),
        report_publisher=fail_publish,
    )

    assert rc == 2
    assert attempted[0]["diagnostic_outcome"]["status"] == "quiet-control-candidate"
    output = capsys.readouterr().out
    assert "reason: report-publication-failed" in output
    assert "quiet-control candidates" not in output
    assert "matrix: partial (1 invalid or failed cell(s))" in output


def test_main_writes_json_report_with_matching_outcome(tmp_path):
    report_path = tmp_path / "interrupt-suite.json"
    rc = main(
        ["--mics", "alc285", "--out", str(report_path)],
        mic_probe=lambda _device: (True, 0.01),
        cell_probe=lambda *_args: _cell(self_int=1),
    )

    assert rc == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["diagnostic_outcome"]["status"] == "no-safe-candidate"
    assert report["diagnostic_outcome"]["live_validation_required"] is True
    assert len(report["rows"]) == len(STRATEGIES)
    assert len(report["raw"]) == len(STRATEGIES)


def test_main_failed_json_publication_leaves_no_partial_report(tmp_path, capsys):
    report_path = tmp_path / "interrupt-suite.json"
    rc = main(
        ["--mics", "alc285", "--out", str(report_path)],
        mic_probe=lambda _device: (True, 0.01),
        cell_probe=lambda *_args: _cell(self_int=0, peak=float("nan")),
    )

    assert rc == 2
    assert not report_path.exists()
    assert list(tmp_path.iterdir()) == []
    output = capsys.readouterr().out
    assert "reason: report-publication-failed" in output
    assert "matrix: partial (6 invalid or failed cell(s))" in output
