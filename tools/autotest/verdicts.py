"""Pure verdict policy for the autonomous voice and barge harnesses.

The runners collect evidence; this module is the single place that decides
whether that evidence is a complete pass, a failed gate, or a successful but
incomplete diagnostic.  Keeping the policy free of audio/model imports makes
the false-green boundary cheap to test.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

PASS = "pass"
FAIL = "fail"
NOT_COVERED = "not_covered"
DIAGNOSTIC_PASS = "diagnostic_pass"
INCOMPLETE = "incomplete"

VOICE_MIN_MONITOR_RMS = 0.01
VOICE_MAX_MEAN_WER = 0.50
VOICE_MAX_BARGE_LATENCY_SEC = 1.0
VOICE_MAX_SYNTHETIC_DELAY_COMMAND_LATENCY_SEC = 1.4


def voice_barge_latency_limit(
    *,
    mode: str,
    clip_source: str,
    clip_role: str,
) -> float:
    """Return the fail-closed limit for one harness-owned acoustic profile.

    Only the physical-device-free, synthesized, exact-command probe carries the
    streaming recognizer's two-word/block-quantized budget. Recorded-owner,
    physical speaker, generic, and stress paths retain the 1.0-second ceiling.
    """

    if mode == "delay" and clip_source == "synth" and clip_role == "command":
        return VOICE_MAX_SYNTHETIC_DELAY_COMMAND_LATENCY_SEC
    return VOICE_MAX_BARGE_LATENCY_SEC


def finite_nonnegative(value: object) -> bool:
    """True only for a real finite latency in the metrics clock domain."""

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str = ""


@dataclass(frozen=True)
class HarnessVerdict:
    """One harness tier's outcome.

    ``passed`` is deliberately stricter than ``checks_ok``: a diagnostic whose
    requested topology cannot cover barge-in may have no failed checks, but it
    is never exposed programmatically as a full pass.
    """

    passed: bool
    checks_ok: bool
    complete: bool
    outcome: str
    checks: tuple[Check, ...]
    failures: tuple[str, ...]
    not_covered: tuple[str, ...]


@dataclass(frozen=True)
class OverallVerdict:
    passed: bool
    outcome: str
    exit_code: int
    failed_tiers: tuple[str, ...]
    incomplete_tiers: tuple[str, ...]


def _finish(checks: Sequence[Check]) -> HarnessVerdict:
    frozen = tuple(checks)
    failures = tuple(check.name for check in frozen if check.status == FAIL)
    missing = tuple(check.name for check in frozen if check.status == NOT_COVERED)
    checks_ok = not failures
    complete = not missing
    outcome = FAIL if failures else PASS if complete else DIAGNOSTIC_PASS
    return HarnessVerdict(
        passed=outcome == PASS,
        checks_ok=checks_ok,
        complete=complete,
        outcome=outcome,
        checks=frozen,
        failures=failures,
        not_covered=missing,
    )


def evaluate_voice(
    *,
    mode: str,
    engine_ready: bool,
    summary_present: bool,
    monitor_rms: float,
    round_trip_clips: int,
    assistant_spoke: bool,
    assistant_audio_turns: int,
    expected_audio_turns: int,
    expected_wer_n: int,
    wer_n: int,
    mean_wer: float,
    error_count: int,
    stuck_hints: Sequence[object],
    self_interrupt_pass: bool | None,
    barge_pass: bool | None,
    barge_latency_s: float | None,
    virtual_route_evidence: Mapping[str, object] | None = None,
    max_mean_wer: float = VOICE_MAX_MEAN_WER,
    max_barge_latency_s: float = VOICE_MAX_BARGE_LATENCY_SEC,
) -> HarnessVerdict:
    """Grade one autonomous voice run without touching an audio device.

    Cable is an intentionally echo-free STT/round-trip diagnostic, so its
    self-interrupt and talk-over checks are ``not_covered``.  Delay and speaker
    modes claim echo coverage and therefore fail closed unless both checks are
    explicitly true.
    """

    checks: list[Check] = [
        Check("engine_ready", PASS if engine_ready else FAIL),
        Check("run_bundle", PASS if summary_present else FAIL),
        Check(
            "audio_flow",
            PASS
            if math.isfinite(monitor_rms) and monitor_rms > VOICE_MIN_MONITOR_RMS
            else FAIL,
            f"monitor_rms={monitor_rms!r}; required > {VOICE_MIN_MONITOR_RMS}",
        ),
        Check(
            "round_trip",
            PASS if round_trip_clips > 0 and assistant_spoke else FAIL,
            f"clips={round_trip_clips}; assistant_spoke={assistant_spoke}",
        ),
        Check(
            "assistant_audio",
            PASS
            if expected_audio_turns > 0 and assistant_audio_turns >= expected_audio_turns
            else FAIL,
            f"first_audio_turns={assistant_audio_turns}; expected>={expected_audio_turns}",
        ),
    ]

    wer_coverage = expected_wer_n > 0 and wer_n == expected_wer_n
    checks.extend(
        (
            Check(
                "wer_coverage",
                PASS if wer_coverage else FAIL,
                f"scored={wer_n}; expected={expected_wer_n}",
            ),
            Check(
                "wer_quality",
                PASS
                if (
                    wer_coverage
                    and math.isfinite(mean_wer)
                    and 0.0 <= mean_wer <= max_mean_wer
                )
                else FAIL,
                f"mean_wer={mean_wer!r}; required <= {max_mean_wer}",
            ),
            Check(
                "runtime_errors",
                PASS if error_count == 0 else FAIL,
                f"error_count={error_count}",
            ),
            Check(
                "stuck_hints",
                PASS if not stuck_hints else FAIL,
                f"count={len(stuck_hints)}",
            ),
        )
    )

    if mode == "cable":
        checks.extend(
            (
                Check("self_interrupt", NOT_COVERED, "echo-free cable"),
                Check("barge_in", NOT_COVERED, "echo-free cable"),
                Check("barge_latency", NOT_COVERED, "echo-free cable"),
            )
        )
    elif mode in ("delay", "speaker"):
        if mode == "delay":
            route = virtual_route_evidence or {}
            checks.extend(
                Check(
                    f"virtual_route_{axis}",
                    PASS if route.get(axis) is True else FAIL,
                    f"explicit_proof={route.get(axis)!r}",
                )
                for axis in (
                    "topology", "capture", "duplex", "correlated",
                    "child_exit", "cleanup",
                )
            )
        checks.extend(
            (
                Check(
                    "self_interrupt",
                    PASS if self_interrupt_pass is True else FAIL,
                    f"explicit_pass={self_interrupt_pass!r}",
                ),
                Check(
                    "barge_in",
                    PASS if barge_pass is True else FAIL,
                    f"explicit_pass={barge_pass!r}",
                ),
                Check(
                    "barge_latency",
                    PASS
                    if (
                        barge_pass is True
                        and finite_nonnegative(barge_latency_s)
                        and float(barge_latency_s) <= max_barge_latency_s
                    )
                    else FAIL,
                    f"latency_s={barge_latency_s!r}; required 0..{max_barge_latency_s}",
                ),
            )
        )
    else:
        checks.extend(
            (
                Check("self_interrupt", FAIL, f"unknown mode={mode!r}"),
                Check("barge_in", FAIL, f"unknown mode={mode!r}"),
                Check("barge_latency", FAIL, f"unknown mode={mode!r}"),
            )
        )
    return _finish(checks)


def evaluate_barge_stress(
    *,
    trials: Sequence[Mapping[str, object]],
    summary_present: bool,
    wav_present: bool,
    assistant_audio_turns: int,
    error_count: int,
    stuck_hints: Sequence[object],
    error: str = "",
    max_barge_latency_s: float = VOICE_MAX_BARGE_LATENCY_SEC,
) -> HarnessVerdict:
    """Grade the repeated over-the-air stress run.

    A self-interrupt trial is meaningful only if the assistant actually began
    speaking.  A talk-over trial passes only if every intended reply started and
    every injected overlap caused a cut.  Empty trial classes are diagnostics,
    never full passes.
    """

    self_trials = [trial for trial in trials if trial.get("kind") == "self_interrupt"]
    talk_trials = [trial for trial in trials if trial.get("kind") == "talk_over"]
    checks: list[Check] = [
        Check("runner_error", PASS if not error else FAIL, error),
        Check("run_bundle", PASS if summary_present else FAIL),
        Check("recorded_audio", PASS if wav_present else FAIL),
        Check(
            "assistant_audio",
            PASS if trials and assistant_audio_turns >= len(trials) else FAIL,
            f"first_audio_turns={assistant_audio_turns}; expected>={len(trials)}",
        ),
        Check(
            "runtime_errors",
            PASS if error_count == 0 else FAIL,
            f"error_count={error_count}",
        ),
        Check(
            "stuck_hints",
            PASS if not stuck_hints else FAIL,
            f"count={len(stuck_hints)}",
        ),
    ]
    if not self_trials:
        checks.append(Check("self_interrupt", NOT_COVERED, "zero trials"))
    else:
        self_ok = all(
            trial.get("started") is True
            and trial.get("terminal") is True
            and int(trial.get("fired", 0) or 0) == 0
            for trial in self_trials
        )
        checks.append(
            Check(
                "self_interrupt",
                PASS if self_ok else FAIL,
                f"passed={sum(bool(t.get('started')) and bool(t.get('terminal')) and int(t.get('fired', 0) or 0) == 0 for t in self_trials)}/{len(self_trials)}",
            )
        )
    if not talk_trials:
        checks.append(Check("barge_in", NOT_COVERED, "zero trials"))
        checks.append(Check("barge_latency", NOT_COVERED, "zero trials"))
    else:
        talk_ok = all(
            trial.get("started") is True
            and trial.get("terminal") is True
            and int(trial.get("fired", 0) or 0) >= 1
            and trial.get("injection_ok") is True
            and trial.get("causal_cut") is True
            for trial in talk_trials
        )
        checks.append(
            Check(
                "barge_in",
                PASS if talk_ok else FAIL,
                f"passed={sum(bool(t.get('started')) and bool(t.get('terminal')) and int(t.get('fired', 0) or 0) >= 1 and t.get('injection_ok') is True and t.get('causal_cut') is True for t in talk_trials)}/{len(talk_trials)}",
            )
        )
        latency_ok = all(
            trial.get("started") is True
            and trial.get("terminal") is True
            and int(trial.get("fired", 0) or 0) >= 1
            and finite_nonnegative(trial.get("latency_s"))
            and 0.0 <= float(trial["latency_s"]) <= max_barge_latency_s
            for trial in talk_trials
        )
        checks.append(
            Check(
                "barge_latency",
                PASS if latency_ok else FAIL,
                f"required every cut in 0..{max_barge_latency_s}s",
            )
        )
    return _finish(checks)


def aggregate_reports(
    reports: Sequence[Mapping[str, object]],
    *,
    require_complete: bool,
    advisory_tiers: frozenset[str] = frozenset({"replay"}),
) -> OverallVerdict:
    """Combine tier reports without turning diagnostics into a false PASS."""

    failed: list[str] = []
    incomplete: list[str] = []
    for report in reports:
        tier = str(report.get("tier") or "unknown")
        outcome = report.get("outcome")
        if tier in advisory_tiers and report.get("ok") is not True:
            incomplete.append(tier)
        elif outcome == FAIL or (outcome is None and report.get("ok") is False):
            failed.append(tier)
        elif outcome == DIAGNOSTIC_PASS or report.get("complete") is False:
            incomplete.append(tier)

    if failed:
        outcome = FAIL
        exit_code = 1
    elif incomplete and require_complete:
        outcome = INCOMPLETE
        exit_code = 2
    elif incomplete:
        outcome = DIAGNOSTIC_PASS
        exit_code = 0
    else:
        outcome = PASS
        exit_code = 0
    return OverallVerdict(
        passed=outcome == PASS,
        outcome=outcome,
        exit_code=exit_code,
        failed_tiers=tuple(failed),
        incomplete_tiers=tuple(incomplete),
    )
