"""Pure false-green tests for the autonomous voice/barge scorecards."""
from __future__ import annotations

import pytest

from tools.autotest.score import score_transcripts
from tools.autotest.verdicts import (
    DIAGNOSTIC_PASS,
    FAIL,
    INCOMPLETE,
    NOT_COVERED,
    PASS,
    aggregate_reports,
    evaluate_barge_stress,
    evaluate_voice,
)


def _voice(**overrides):
    values = {
        "mode": "delay",
        "engine_ready": True,
        "summary_present": True,
        "monitor_rms": 0.05,
        "round_trip_clips": 3,
        "assistant_spoke": True,
        "assistant_audio_turns": 5,
        "expected_audio_turns": 5,
        "expected_wer_n": 3,
        "wer_n": 3,
        "mean_wer": 0.20,
        "error_count": 0,
        "stuck_hints": (),
        "self_interrupt_pass": True,
        "barge_pass": True,
        "barge_latency_s": 0.35,
    }
    values.update(overrides)
    return evaluate_voice(**values)


def _checks(verdict):
    return {check.name: check.status for check in verdict.checks}


def test_echo_capable_voice_run_requires_every_axis_and_can_fully_pass():
    verdict = _voice()

    assert verdict.outcome == PASS
    assert verdict.passed is True
    assert verdict.complete is True
    assert verdict.failures == ()
    assert verdict.not_covered == ()


@pytest.mark.parametrize(
    ("overrides", "failed_check"),
    (
        ({"engine_ready": False}, "engine_ready"),
        ({"summary_present": False}, "run_bundle"),
        ({"monitor_rms": 0.01}, "audio_flow"),
        ({"monitor_rms": float("nan")}, "audio_flow"),
        ({"round_trip_clips": 0}, "round_trip"),
        ({"assistant_spoke": False}, "round_trip"),
        ({"assistant_audio_turns": 4}, "assistant_audio"),
        ({"expected_wer_n": 0, "wer_n": 0}, "wer_coverage"),
        ({"wer_n": 2}, "wer_coverage"),
        ({"mean_wer": 0.501}, "wer_quality"),
        ({"mean_wer": float("nan")}, "wer_quality"),
        ({"error_count": 1}, "runtime_errors"),
        ({"stuck_hints": ("LLM stalled",)}, "stuck_hints"),
        ({"self_interrupt_pass": False}, "self_interrupt"),
        ({"self_interrupt_pass": None}, "self_interrupt"),
        ({"barge_pass": False}, "barge_in"),
        ({"barge_pass": None}, "barge_in"),
        ({"barge_latency_s": None}, "barge_latency"),
        ({"barge_latency_s": -0.01}, "barge_latency"),
        ({"barge_latency_s": 1.001}, "barge_latency"),
    ),
)
def test_voice_required_failure_can_never_be_programmatic_pass(
    overrides, failed_check
):
    verdict = _voice(**overrides)

    assert verdict.outcome == FAIL
    assert verdict.passed is False
    assert verdict.checks_ok is False
    assert failed_check in verdict.failures


def test_wer_boundary_is_explicit_and_inclusive():
    assert _voice(mean_wer=0.50).passed is True
    assert _voice(mean_wer=0.500001).passed is False


def test_wer_coverage_counts_real_hypotheses_not_synthetic_missing_pairs():
    score = score_transcripts(
        ["first labelled clip", "second labelled clip"],
        ["first labelled clip"],
    )

    assert score.n == 1
    assert score.pairs[-1] == ("second labelled clip", "", 1.0)
    verdict = _voice(expected_wer_n=2, wer_n=score.n, mean_wer=score.mean_wer)
    assert verdict.passed is False
    assert "wer_coverage" in verdict.failures


def test_cable_is_successful_diagnostic_but_never_a_full_barge_pass():
    verdict = _voice(
        mode="cable",
        self_interrupt_pass=None,
        barge_pass=None,
        barge_latency_s=None,
    )

    assert verdict.outcome == DIAGNOSTIC_PASS
    assert verdict.passed is False
    assert verdict.checks_ok is True
    assert verdict.complete is False
    assert verdict.not_covered == ("self_interrupt", "barge_in", "barge_latency")
    assert _checks(verdict)["self_interrupt"] == NOT_COVERED
    assert _checks(verdict)["barge_in"] == NOT_COVERED


def test_cable_still_fails_when_its_stt_contract_is_red():
    verdict = _voice(
        mode="cable",
        mean_wer=0.9,
        self_interrupt_pass=None,
        barge_pass=None,
        barge_latency_s=None,
    )

    assert verdict.outcome == FAIL
    assert verdict.passed is False
    assert "wer_quality" in verdict.failures


def _stress_trials():
    return (
        {"kind": "self_interrupt", "started": True, "fired": 0},
        {"kind": "self_interrupt", "started": True, "fired": 0},
        {"kind": "talk_over", "started": True, "fired": 1, "latency_s": 0.35},
        {"kind": "talk_over", "started": True, "fired": 1, "latency_s": 0.48},
    )


def test_barge_stress_requires_artifacts_started_replies_and_every_cut():
    verdict = evaluate_barge_stress(
        trials=_stress_trials(),
        summary_present=True,
        wav_present=True,
        assistant_audio_turns=4,
        error_count=0,
        stuck_hints=(),
    )

    assert verdict.outcome == PASS
    assert verdict.passed is True


@pytest.mark.parametrize(
    ("trials", "summary", "wav", "error", "failed_check"),
    (
        (
            (
                {"kind": "self_interrupt", "started": False, "fired": 0},
                {"kind": "talk_over", "started": True, "fired": 1},
            ),
            True,
            True,
            "",
            "self_interrupt",
        ),
        (
            (
                {"kind": "self_interrupt", "started": True, "fired": 1},
                {"kind": "talk_over", "started": True, "fired": 1},
            ),
            True,
            True,
            "",
            "self_interrupt",
        ),
        (
            (
                {"kind": "self_interrupt", "started": True, "fired": 0},
                {"kind": "talk_over", "started": False, "fired": 0},
            ),
            True,
            True,
            "",
            "barge_in",
        ),
        (
            (
                {"kind": "self_interrupt", "started": True, "fired": 0},
                {"kind": "talk_over", "started": True, "fired": 0},
            ),
            True,
            True,
            "",
            "barge_in",
        ),
        (_stress_trials(), False, True, "", "run_bundle"),
        (_stress_trials(), True, False, "", "recorded_audio"),
        (_stress_trials(), True, True, "engine crashed", "runner_error"),
    ),
)
def test_barge_stress_failure_can_never_be_programmatic_pass(
    trials, summary, wav, error, failed_check
):
    verdict = evaluate_barge_stress(
        trials=trials,
        summary_present=summary,
        wav_present=wav,
        assistant_audio_turns=len(trials),
        error_count=0,
        stuck_hints=(),
        error=error,
    )

    assert verdict.outcome == FAIL
    assert verdict.passed is False
    assert failed_check in verdict.failures


def test_zero_barge_trials_are_not_covered_instead_of_green():
    verdict = evaluate_barge_stress(
        trials=({"kind": "self_interrupt", "started": True, "fired": 0},),
        summary_present=True,
        wav_present=True,
        assistant_audio_turns=1,
        error_count=0,
        stuck_hints=(),
    )

    assert verdict.outcome == DIAGNOSTIC_PASS
    assert verdict.passed is False
    assert verdict.complete is False
    assert verdict.not_covered == ("barge_in", "barge_latency")


@pytest.mark.parametrize("latency", (None, -0.01, 1.001, float("nan")))
def test_barge_stress_rejects_missing_or_out_of_bound_cut_latency(latency):
    trials = list(_stress_trials())
    trials[-1] = {**trials[-1], "latency_s": latency}

    verdict = evaluate_barge_stress(
        trials=trials,
        summary_present=True,
        wav_present=True,
        assistant_audio_turns=len(trials),
        error_count=0,
        stuck_hints=(),
    )

    assert verdict.passed is False
    assert "barge_latency" in verdict.failures


@pytest.mark.parametrize(
    ("overrides", "failed_check"),
    (
        ({"assistant_audio_turns": 3}, "assistant_audio"),
        ({"error_count": 1}, "runtime_errors"),
        ({"stuck_hints": ("tts stalled",)}, "stuck_hints"),
    ),
)
def test_barge_stress_rejects_incomplete_runtime_evidence(overrides, failed_check):
    values = {
        "trials": _stress_trials(),
        "summary_present": True,
        "wav_present": True,
        "assistant_audio_turns": 4,
        "error_count": 0,
        "stuck_hints": (),
    }
    values.update(overrides)

    verdict = evaluate_barge_stress(**values)

    assert verdict.passed is False
    assert failed_check in verdict.failures


def test_single_cable_diagnostic_is_explicit_but_can_exit_successfully():
    overall = aggregate_reports(
        (
            {
                "tier": "voice",
                "ok": False,
                "complete": False,
                "outcome": DIAGNOSTIC_PASS,
            },
        ),
        require_complete=False,
    )

    assert overall.outcome == DIAGNOSTIC_PASS
    assert overall.passed is False
    assert overall.exit_code == 0
    assert overall.incomplete_tiers == ("voice",)


def test_all_cannot_pass_when_cable_leaves_barge_uncovered():
    reports = (
        {"tier": "memory", "ok": True},
        {
            "tier": "voice",
            "ok": False,
            "complete": False,
            "outcome": DIAGNOSTIC_PASS,
        },
        {"tier": "suite", "ok": True},
    )

    overall = aggregate_reports(reports, require_complete=True)

    assert overall.outcome == INCOMPLETE
    assert overall.passed is False
    assert overall.exit_code == 2
    assert overall.incomplete_tiers == ("voice",)


def test_hard_failure_wins_over_incomplete_diagnostic():
    overall = aggregate_reports(
        (
            {"tier": "voice", "outcome": DIAGNOSTIC_PASS, "complete": False},
            {"tier": "suite", "ok": False},
        ),
        require_complete=True,
    )

    assert overall.outcome == FAIL
    assert overall.exit_code == 1
    assert overall.failed_tiers == ("suite",)
