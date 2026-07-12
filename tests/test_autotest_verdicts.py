"""Pure false-green tests for the autonomous voice/barge scorecards."""
from __future__ import annotations

import json

import pytest

from tools.autotest.__main__ import _analyze_bundle
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
from tools.autotest.voice_loop import (
    PromptPlaybackBinding,
    RuntimeMarkerLedger,
    parse_runtime_marker,
    score_prompt_bindings,
    summarize_prompt_bindings,
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


def test_runtime_marker_parser_recovers_text_and_causal_identity():
    final = parse_runtime_marker(
        '12:00 INFO speaker.runtime | final -> brain: "Paul\'s (test)" '
        '(mode=assistant input_generation=7)'
    )
    started = parse_runtime_marker(
        "12:00 DEBUG speaker.runtime | playback receipt started: "
        "fragment=playback-3 task=T9 input_generation=7"
    )
    terminal = parse_runtime_marker(
        "12:01 DEBUG speaker.runtime | playback quiescent: tracked assistant "
        "reply terminal task=T9 input_generation=7 outcome=completed"
    )
    barge = parse_runtime_marker(
        "12:02 INFO speaker.sherpa | barge-in detected"
    )

    assert final is not None
    assert (final.kind, final.text, final.input_generation) == (
        "final",
        "Paul's (test)",
        7,
    )
    assert started is not None
    assert (started.kind, started.task_id, started.fragment_id) == (
        "playback_started",
        "T9",
        "playback-3",
    )
    assert terminal is not None
    assert (
        terminal.kind,
        terminal.task_id,
        terminal.input_generation,
        terminal.outcome,
    ) == (
        "playback_quiescent",
        "T9",
        7,
        "completed",
    )
    assert barge is not None
    assert (barge.kind, barge.input_generation) == ("barge_in", None)
    assert parse_runtime_marker(
        "final -> brain: 'missing generation' (mode=assistant)"
    ) is None
    assert parse_runtime_marker(
        "playback receipt started: fragment=x task= input_generation=7"
    ) is None
    assert parse_runtime_marker(
        "playback quiescent: tracked assistant reply terminal "
        "task=T9 input_generation=7"
    ) is None
    # Do not double-count the explanatory line which precedes the legacy
    # standalone marker in the confirmed-barge path.
    assert parse_runtime_marker(
        "12:02 INFO speaker.sherpa | barge-in confirmed by speech: "
        "'barge-in detected'"
    ) is None


def test_prompt_binding_requires_ordered_exact_task_and_generation():
    ledger = RuntimeMarkerLedger()
    # A terminal before this injection cannot satisfy it.
    ledger.observe(
        "playback quiescent: tracked assistant reply terminal "
        "task=T7 input_generation=7 outcome=completed"
    )
    cursor = ledger.cursor()
    ledger.observe(
        'final -> brain: "Paul\'s test" '
        "(mode=assistant input_generation=7)"
    )
    ledger.observe(
        "playback receipt started: fragment=other task=T0 input_generation=8"
    )
    ledger.observe(
        "playback receipt started: fragment=target task=T7 input_generation=7"
    )
    ledger.observe(
        "playback quiescent: tracked assistant reply terminal "
        "task=T8 input_generation=7 outcome=completed"
    )
    ledger.observe(
        "playback quiescent: tracked assistant reply terminal "
        "task=T7 input_generation=7 outcome=completed"
    )

    binding = ledger.wait_prompt_binding(
        "Paul's test",
        role="round_trip",
        after_sequence=cursor,
        final_timeout=0,
        start_timeout=0,
        terminal_timeout=0,
    )

    assert binding.passed is True
    assert binding.word_error_rate == 0.0
    assert binding.task_id == "T7"
    assert (
        binding.final_sequence
        < binding.playback_started_sequence
        < binding.playback_quiescent_sequence
    )


def test_late_terminal_cannot_upgrade_a_timed_out_binding():
    ledger = RuntimeMarkerLedger()
    cursor = ledger.cursor()
    ledger.observe(
        "final -> brain: 'hello there' "
        "(mode=assistant input_generation=2)"
    )
    ledger.observe(
        "playback receipt started: fragment=p2 task=T2 input_generation=2"
    )

    binding = ledger.wait_prompt_binding(
        "hello there",
        role="round_trip",
        after_sequence=cursor,
        final_timeout=0,
        start_timeout=0,
        terminal_timeout=0,
    )
    assert binding.audio_started is True
    assert binding.passed is False

    ledger.observe(
        "playback quiescent: tracked assistant reply terminal "
        "task=T2 input_generation=2 outcome=completed"
    )

    # Bindings are immutable bounded observations.  No end-of-run refresh may
    # turn a response that missed its terminal deadline into a pass.
    assert binding.playback_quiescent_sequence is None
    assert binding.passed is False


def test_terminal_before_specific_barge_marker_cannot_satisfy_cut():
    ledger = RuntimeMarkerLedger()
    ledger.observe("12:02 INFO speaker.sherpa | barge-in detected")
    cursor = ledger.cursor()
    ledger.observe(
        "final -> brain: 'tell me a long story' "
        "(mode=assistant input_generation=12)"
    )
    ledger.observe(
        "playback receipt started: fragment=p12 task=T12 input_generation=12"
    )
    ledger.observe(
        "playback quiescent: tracked assistant reply terminal "
        "task=T12 input_generation=12 outcome=interrupted"
    )
    barge = ledger.observe(
        "12:03 INFO speaker.sherpa | barge-in detected"
    )
    assert barge is not None
    selected_barge = ledger.wait_barge(after_sequence=cursor, timeout=0)
    assert selected_barge is not None
    assert selected_barge.sequence == barge.sequence

    binding = ledger.wait_prompt_binding(
        "tell me a long story",
        role="speak",
        after_sequence=cursor,
        final_timeout=0,
        start_timeout=0,
        terminal_timeout=None,
        required_terminal_outcome="interrupted",
    )
    rejected = ledger.wait_binding_terminal(
        binding,
        timeout=0,
        after_sequence=selected_barge.sequence,
    )

    assert rejected.audio_started is True
    assert rejected.audio_quiescent is False
    assert rejected.passed is False

    ledger.observe(
        "playback quiescent: tracked assistant reply terminal "
        "task=T99 input_generation=12 outcome=interrupted"
    )
    ledger.observe(
        "playback quiescent: tracked assistant reply terminal "
        "task=T12 input_generation=12 outcome=interrupted"
    )
    completed = ledger.wait_binding_terminal(
        binding,
        timeout=0,
        after_sequence=selected_barge.sequence,
    )

    assert completed.audio_quiescent is True
    assert completed.playback_quiescent_sequence > selected_barge.sequence


@pytest.mark.parametrize(
    "outcome",
    ("failed", "dropped", "interrupted", "unknown"),
)
def test_ordinary_prompt_requires_completed_terminal_outcome(outcome):
    ledger = RuntimeMarkerLedger()
    cursor = ledger.cursor()
    ledger.observe(
        "final -> brain: 'ordinary question' "
        "(mode=assistant input_generation=21)"
    )
    ledger.observe(
        "playback receipt started: fragment=p21 task=T21 input_generation=21"
    )
    ledger.observe(
        "playback quiescent: tracked assistant reply terminal "
        f"task=T21 input_generation=21 outcome={outcome}"
    )

    binding = ledger.wait_prompt_binding(
        "ordinary question",
        role="round_trip",
        after_sequence=cursor,
        final_timeout=0,
        start_timeout=0,
        terminal_timeout=0,
    )

    assert binding.terminal_outcome == outcome
    assert binding.required_terminal_outcome == "completed"
    assert binding.audio_quiescent is False
    assert binding.passed is False


def test_barge_prompt_rejects_completed_terminal_after_cut():
    ledger = RuntimeMarkerLedger()
    cursor = ledger.cursor()
    ledger.observe(
        "final -> brain: 'keep talking' "
        "(mode=assistant input_generation=22)"
    )
    ledger.observe(
        "playback receipt started: fragment=p22 task=T22 input_generation=22"
    )
    barge = ledger.observe("12:04 INFO speaker.sherpa | barge-in detected")
    assert barge is not None
    ledger.observe(
        "playback quiescent: tracked assistant reply terminal "
        "task=T22 input_generation=22 outcome=completed"
    )

    binding = ledger.wait_prompt_binding(
        "keep talking",
        role="speak",
        after_sequence=cursor,
        final_timeout=0,
        start_timeout=0,
        terminal_timeout=None,
        required_terminal_outcome="interrupted",
    )
    binding = ledger.wait_binding_terminal(
        binding,
        timeout=0,
        after_sequence=barge.sequence,
    )

    assert binding.playback_quiescent_sequence > barge.sequence
    assert binding.terminal_outcome == "completed"
    assert binding.audio_quiescent is False
    assert binding.passed is False


def test_command_is_recognition_graded_but_excluded_from_audio_count():
    audio = PromptPlaybackBinding(
        role="round_trip",
        reference="what is the capital of france",
        recognized_text="what is the capital of france",
        input_generation=3,
        word_error_rate=0.0,
        final_sequence=1,
        task_id="T3",
        playback_started_sequence=2,
        playback_quiescent_sequence=3,
        terminal_outcome="completed",
    )
    # A mapped command may be silent or acknowledge.  Either way, playback is
    # excluded; its exact final still has to match within the WER bound.
    command = PromptPlaybackBinding(
        role="command",
        reference="stop",
        recognized_text="stop",
        input_generation=4,
        word_error_rate=0.0,
        final_sequence=4,
        task_id="T4",
        playback_started_sequence=5,
        playback_quiescent_sequence=6,
    )
    # Recorded manifests may omit text.  The first post-cursor final binds the
    # generation while WER coverage correctly excludes the unlabelled clip.
    unlabelled = PromptPlaybackBinding(
        role="round_trip",
        reference="",
        recognized_text="some recognized request",
        input_generation=5,
        word_error_rate=None,
        final_sequence=7,
        task_id="T5",
        playback_started_sequence=8,
        playback_quiescent_sequence=9,
        terminal_outcome="completed",
    )

    evidence = summarize_prompt_bindings([audio, command, unlabelled])
    score = score_prompt_bindings([audio, command, unlabelled])

    assert command.passed is True
    assert unlabelled.passed is True
    assert PromptPlaybackBinding(
        role="command",
        reference="stop",
        recognized_text="speaking",
        input_generation=9,
        word_error_rate=0.500001,
        final_sequence=10,
    ).passed is False
    assert score.n == 2
    assert [pair[0] for pair in score.pairs] == [
        "what is the capital of france",
        "stop",
    ]
    assert evidence == {
        "expected_prompts": 3,
        "recognized_prompts": 3,
        "expected_labelled_prompts": 2,
        "recognized_labelled_prompts": 2,
        "expected_audio_prompts": 2,
        "causal_audio_prompts": 2,
        "expected_commands": 1,
        "recognized_commands": 1,
        "passed": True,
    }


def test_analyze_bundle_counts_only_finite_receipt_latencies(tmp_path):
    bundle = {
        "transcript": [
            {"role": "user", "text": "hello"},
            {"role": "assistant", "text": "hi"},
        ],
        "turns": [
            {"first_audio_latency": 0.0, "barge_in_latency": 0.2},
            {"first_audio_latency": 1.25, "barge_in_latency": None},
            {"first_audio_latency": float("nan"), "barge_in_latency": float("nan")},
            {"first_audio_latency": float("inf"), "barge_in_latency": -1.0},
            {"first_audio_latency": -0.1, "barge_in_latency": True},
            {"first_audio_latency": True, "barge_in_latency": None},
        ],
        "stuck_hints": ["one hint"],
        "errors": [{"message": "first"}, {"message": "second"}],
        "counts": {"errors": 1, "warnings": 2.0},
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")

    parsed = _analyze_bundle(str(path))

    assert parsed["first_audio_turns"] == 2
    assert parsed["n_barge_in_turns"] == 1
    assert parsed["error_count"] == 2
    assert parsed["warnings"] == 2


def test_analyze_bundle_rejects_nonfinite_counts(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "transcript": [],
                "turns": [],
                "stuck_hints": [],
                "errors": [],
                "counts": {"errors": float("nan"), "warnings": 0},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="finite integer"):
        _analyze_bundle(str(path))


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
        {"kind": "self_interrupt", "started": True, "terminal": True, "fired": 0},
        {"kind": "self_interrupt", "started": True, "terminal": True, "fired": 0},
        {
            "kind": "talk_over", "started": True, "terminal": True,
            "fired": 1, "latency_s": 0.35,
            "injection_ok": True, "causal_cut": True,
        },
        {
            "kind": "talk_over", "started": True, "terminal": True,
            "fired": 1, "latency_s": 0.48,
            "injection_ok": True, "causal_cut": True,
        },
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
        trials=(
            {
                "kind": "self_interrupt",
                "started": True,
                "terminal": True,
                "fired": 0,
            },
        ),
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


@pytest.mark.parametrize(
    ("field", "value", "failed_check"),
    (
        ("terminal", False, "barge_in"),
        ("injection_ok", False, "barge_in"),
        ("causal_cut", False, "barge_in"),
    ),
)
def test_barge_stress_requires_terminal_and_causal_successful_injection(
    field, value, failed_check
):
    trials = [dict(trial) for trial in _stress_trials()]
    trials[-1][field] = value

    verdict = evaluate_barge_stress(
        trials=trials,
        summary_present=True,
        wav_present=True,
        assistant_audio_turns=len(trials),
        error_count=0,
        stuck_hints=(),
    )

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
