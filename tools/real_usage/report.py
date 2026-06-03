"""PASS/FAIL grading + report emission for the real-usage harness.

This module is PURE LOGIC -- it takes plain result dicts (one per recording) and
grades them, then renders Markdown + JSON. No audio, no models, no sounddevice;
that keeps the half that decides PASS/FAIL unit-testable in CI while the actual
audio run stays on-machine-only (exactly the tools/live_session split).

A "result" dict (produced by runner.RealUsageRun, or hand-built in tests) has:

    {
      "fixture": "run-20260530-181513.wav",
      "asr_finals": ["are you listening to me", ...],   # STT visibility
      "spoken": ["Mode: assistant", "Yes, I'm listening."],  # what engine.speak()'d
      "first_audio_latencies": [1.83, 2.10],            # per-turn, seconds
      "barge_in_count": 0,
      "playback_errors": [],            # ALSA/PortAudio log lines seen this run
      "playback_loop_dead": false,      # engine._running cleared == loop crashed
      "shutdown_ok": true,
      "shutdown_seconds": 0.4,
      "shutdown_timeout": 8.0,
      "error": null,                    # harness-level exception text, if any
    }

The three live failures this grades for:
  1. SHUTDOWN HANG    -> shutdown_clean (shutdown_ok and < timeout)
  2. BARGE-IN STORM   -> barge_in_storm (count > threshold in one recording)
  3. (broken output)  -> playback_clean (no ALSA errors, loop not dead)
plus STT quality (asr_finals shown, empty == went-deaf visibility flag) and the
basic "did it answer at all" check (response_present).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

# A genuine single recording has at most one real barge-in (a second speaker over
# the TTS). More than this in ONE recording with no real second speaker == the
# assistant interrupting itself off its own playback (the self-storm heuristic).
DEFAULT_BARGE_IN_STORM_THRESHOLD = 1

# Control / framing lines the engine speaks that are NOT a real spoken answer.
# Mirrors tools/live_session.driver._is_answer so a run whose only "speech" was a
# mode announcement is correctly graded as silent.
_CONTROL_PREFIXES = (
    "Mode:", "Queued ", "[cancelled]", "Confirm command:", "Nothing to ",
    "Command cancelled.", "I can help with:",
)

# Soft budget for first-audio latency (report-only nudge, never a hard fail).
DEFAULT_FIRST_AUDIO_BUDGET_SEC = 3.0

# Inputs whose RMS *and* peak are both below these are DIGITALLY SILENT -- an empty
# capture (dead/muted mic), not a pipeline failure. They are graded EMPTY and
# EXCLUDED from pass/fail so a silent-mic recording can't masquerade as a "went
# deaf" app bug. Mirrors the engine's own ~silent threshold (avg_rms < 1e-4).
SILENT_INPUT_RMS = 1e-4
SILENT_INPUT_PEAK = 1e-3


def _is_answer(text: str) -> bool:
    text = (text or "").strip()
    return bool(text) and not text.startswith(_CONTROL_PREFIXES)


def _ms(v: Any) -> Optional[float]:
    return round(v * 1000.0, 1) if isinstance(v, (int, float)) else None


def _fmt_level(v: Any) -> str:
    return f"{float(v):.5f}" if isinstance(v, (int, float)) else "?"


def is_silent_input(input_rms: Any, input_peak: Any) -> bool:
    """True iff the injected waveform is digitally silent (both RMS and peak below
    the silent thresholds). Pure -- decided from two floats so it is unit-testable
    without numpy/audio. Unknown (None) level == NOT silent: we can't claim a
    recording is empty without having measured it."""
    if input_rms is None or input_peak is None:
        return False
    try:
        return float(input_rms) < SILENT_INPUT_RMS and float(input_peak) < SILENT_INPUT_PEAK
    except (TypeError, ValueError):
        return False


def empty_result(fixture: str, *, input_rms: float = 0.0, input_peak: float = 0.0) -> dict:
    """A result dict for a recording short-circuited as digitally silent BEFORE the
    pipeline ran (the CLI skips building the runtime for empties -- no model load,
    nothing spoken aloud). grade_fixture maps it to the EMPTY verdict."""
    return {
        "fixture": fixture,
        "asr_finals": [],
        "spoken": [],
        "first_audio_latencies": [],
        "barge_in_count": 0,
        "playback_errors": [],
        "playback_loop_dead": False,
        "shutdown_ok": True,
        "shutdown_seconds": 0.0,
        "shutdown_timeout": 0.0,
        "error": None,
        "input_rms": input_rms,
        "input_peak": input_peak,
        "skipped_silent": True,
    }


def grade_fixture(
    result: dict,
    *,
    barge_in_storm_threshold: int = DEFAULT_BARGE_IN_STORM_THRESHOLD,
    first_audio_budget_sec: float = DEFAULT_FIRST_AUDIO_BUDGET_SEC,
) -> dict:
    """Grade ONE recording's result dict. Returns a grade dict with per-check
    booleans, human reasons, and an overall verdict.

    Overall PASS requires: a real spoken response, a clean playback path, a clean
    (bounded) shutdown, and NO barge-in storm. The transcript is always surfaced
    (STT garble is the thing under test, not a hard fail) and latency is
    report-only.
    """
    asr_finals = [str(t) for t in (result.get("asr_finals") or []) if str(t).strip()]
    spoken = list(result.get("spoken") or [])
    answers = [s for s in spoken if _is_answer(s)]
    barge_in_count = int(result.get("barge_in_count") or 0)
    playback_errors = list(result.get("playback_errors") or [])
    playback_loop_dead = bool(result.get("playback_loop_dead"))
    shutdown_ok = bool(result.get("shutdown_ok"))
    shutdown_seconds = result.get("shutdown_seconds")
    shutdown_timeout = float(result.get("shutdown_timeout") or 0.0)
    harness_error = result.get("error")
    input_rms = result.get("input_rms")
    input_peak = result.get("input_peak")
    # A digitally-silent capture (dead/muted mic) is EMPTY, not a failure: the
    # pipeline never had input to work with, so the went-deaf/went-silent checks
    # below are vacuous. We still compute them, then override the verdict.
    empty = bool(result.get("skipped_silent")) or is_silent_input(input_rms, input_peak)

    reasons: list[str] = []
    checks: dict[str, bool] = {}

    # --- STT quality (visibility flag, not a hard fail) ---
    transcript_sane = bool(asr_finals)
    checks["transcript_sane"] = transcript_sane
    if not transcript_sane:
        reasons.append("WENT DEAF: no ASR finals (recognizer produced nothing)")

    # --- did it answer at all? ---
    response_present = bool(answers)
    checks["response_present"] = response_present
    if not response_present:
        reasons.append("WENT SILENT: no non-control spoken response")

    # --- broken-output / ALSA path ---
    playback_clean = not playback_errors and not playback_loop_dead
    checks["playback_clean"] = playback_clean
    if playback_errors:
        reasons.append(f"PLAYBACK ERROR(S): {len(playback_errors)} ALSA/PortAudio log line(s): "
                       f"{playback_errors[0]!r}")
    if playback_loop_dead:
        reasons.append("PLAYBACK LOOP DEAD: engine._running cleared (playback loop crashed)")

    # --- SHUTDOWN HANG (failure #1) ---
    over_budget = (
        shutdown_seconds is not None
        and shutdown_timeout > 0
        and float(shutdown_seconds) >= shutdown_timeout
    )
    shutdown_clean = shutdown_ok and not over_budget
    checks["shutdown_clean"] = shutdown_clean
    if not shutdown_clean:
        secs = f"{float(shutdown_seconds):.2f}s" if shutdown_seconds is not None else "?"
        reasons.append(
            f"SHUTDOWN HANG: stop() did not return cleanly ({secs} >= "
            f"{shutdown_timeout:.0f}s timeout); play-thread join blocked in ALSA "
            f"out.write (sherpa.py stop()/_playback_loop)"
        )

    # --- BARGE-IN STORM (failure #2) ---
    barge_in_storm = barge_in_count > barge_in_storm_threshold
    checks["barge_in_storm"] = barge_in_storm
    if barge_in_storm:
        reasons.append(
            f"BARGE-IN STORM: {barge_in_count} barge-in fires in one recording "
            f"(self-storm heuristic threshold={barge_in_storm_threshold}); the mic "
            f"is re-triggering on the assistant's own TTS"
        )

    # --- harness-level failure (engine never started, build failed, ...) ---
    if harness_error:
        reasons.append(f"HARNESS ERROR: {harness_error}")

    # --- latency (report-only soft flag) ---
    latencies = [float(x) for x in (result.get("first_audio_latencies") or [])
                 if isinstance(x, (int, float))]
    latency_over_budget = bool(latencies) and max(latencies) > first_audio_budget_sec

    verdict_pass = (
        response_present
        and playback_clean
        and shutdown_clean
        and not barge_in_storm
        and not harness_error
    )

    if empty:
        # Neither PASS nor FAIL -- replace the vacuous went-deaf/silent reasons
        # with the empty note so a silent-mic capture can't read as an app bug.
        verdict = "EMPTY"
        verdict_pass = False
        reasons = [
            f"EMPTY RECORDING: input is digitally silent "
            f"(rms={_fmt_level(input_rms)}, peak={_fmt_level(input_peak)}); the "
            f"capture itself is empty (dead/muted mic), NOT a pipeline failure -- "
            f"excluded from pass/fail"
        ]
    else:
        verdict = "PASS" if verdict_pass else "FAIL"

    return {
        "fixture": result.get("fixture"),
        "verdict": verdict,
        "passed": verdict_pass,
        "empty": empty,
        "input_rms": input_rms,
        "input_peak": input_peak,
        "checks": checks,
        "reasons": reasons,
        "asr_finals": asr_finals,
        "spoken_answer": " ".join(answers) if answers else "",
        "spoken_all": spoken,
        "barge_in_count": barge_in_count,
        "playback_errors": playback_errors,
        "playback_loop_dead": playback_loop_dead,
        "shutdown_ok": shutdown_ok,
        "shutdown_seconds": shutdown_seconds,
        "shutdown_timeout": shutdown_timeout,
        "first_audio_latencies": latencies,
        "first_audio_latency_max": max(latencies) if latencies else None,
        "latency_over_budget": latency_over_budget,
    }


def grade_run(
    results: list[dict],
    **kwargs,
) -> dict:
    """Grade every recording's result. Returns the run-level summary
    (per-fixture grades + counts + overall pass).

    EMPTY (digitally-silent) recordings are counted separately and EXCLUDED from
    pass/fail -- ``all_pass`` is over the non-empty set, so a dead-mic capture
    neither fails the run nor counts as a pass."""
    grades = [grade_fixture(r, **kwargs) for r in results]
    graded = [g for g in grades if not g.get("empty")]
    n_pass = sum(1 for g in graded if g["passed"])
    n_empty = len(grades) - len(graded)
    return {
        "grades": grades,
        "n_total": len(grades),
        "n_pass": n_pass,
        "n_fail": len(graded) - n_pass,
        "n_empty": n_empty,
        # Over the NON-empty set: a run with no real failures passes -- even if
        # every recording was an excluded empty (vacuously true, n_pass==0==len).
        # So an all-silent batch exits 0 (no failures), not 1.
        "all_pass": n_pass == len(graded),
    }


def _truncate(text: str, n: int) -> str:
    text = (text or "").replace("|", "/").replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 3] + "..."


def render_markdown(run: dict, *, run_id: str) -> str:
    """Render the grade summary as Markdown. One row per recording with the ASR
    finals (so STT garble is visible), the spoken response, latency, barge-ins,
    playback + shutdown status, and the verdict."""
    n_empty = run.get("n_empty", 0)
    empty_note = f", {n_empty} EMPTY (silent capture, excluded)" if n_empty else ""
    lines: list[str] = []
    lines.append(f"# Real-usage validation: {run_id}\n")
    lines.append(
        f"**{run['n_pass']}/{run['n_total']} PASS** "
        f"({run.get('n_fail', 0)} FAIL{empty_note}; "
        f"{'ALL PASS' if run['all_pass'] else 'FAILURES PRESENT'})\n"
    )
    lines.append("\n## Results\n")
    lines.append("| fixture | asr_finals (heard) | spoken_response | first_audio | barge_ins | playback | shutdown | VERDICT |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for g in run["grades"]:
        if g.get("empty"):
            heard = f"(silent capture, rms={_fmt_level(g.get('input_rms'))})"
            spoken = "(skipped -- empty)"
        else:
            heard = _truncate(" | ".join(g["asr_finals"]) or "(none)", 70)
            spoken = _truncate(g["spoken_answer"] or "(silent)", 60)
        fa = g["first_audio_latency_max"]
        fa_s = f"{fa:.2f}s" if isinstance(fa, (int, float)) else "-"
        if g["latency_over_budget"]:
            fa_s += " (slow)"
        playback = "ok" if g["checks"].get("playback_clean") else "ERROR"
        if g["shutdown_seconds"] is not None:
            sd = f"{float(g['shutdown_seconds']):.2f}s"
        else:
            sd = "?"
        if not g["checks"].get("shutdown_clean"):
            sd += " HANG"
        lines.append(
            f"| {_truncate(str(g['fixture']), 28)} | {heard} | {spoken} | {fa_s} | "
            f"{g['barge_in_count']} | {playback} | {sd} | **{g['verdict']}** |"
        )

    # Per-fixture detail with the failure reasons.
    lines.append("\n## Detail\n")
    for g in run["grades"]:
        lines.append(f"### {g['fixture']} -- {g['verdict']}\n")
        if g.get("empty"):
            lines.append(
                f"- **EMPTY:** digitally-silent capture "
                f"(rms={_fmt_level(g.get('input_rms'))}, "
                f"peak={_fmt_level(g.get('input_peak'))}) -- the recording is empty "
                f"(dead/muted mic), so it is excluded from pass/fail (not a pipeline failure)."
            )
            lines.append("")
            continue
        lines.append(f"- **heard (ASR finals):** {g['asr_finals'] or '(none -- went deaf)'}")
        lines.append(f"- **spoken response:** {g['spoken_answer'] or '(silent)'}")
        if g["spoken_all"] and g["spoken_all"] != [g["spoken_answer"]]:
            lines.append(f"- **all speak() calls:** {g['spoken_all']}")
        if g["first_audio_latencies"]:
            lines.append(f"- **first-audio latency (s):** {g['first_audio_latencies']}")
        lines.append(f"- **barge-in fires:** {g['barge_in_count']}")
        lines.append(f"- **shutdown:** ok={g['shutdown_ok']} "
                     f"seconds={g['shutdown_seconds']} timeout={g['shutdown_timeout']}")
        if g["playback_errors"]:
            lines.append(f"- **playback errors:** {g['playback_errors']}")
        if g["reasons"]:
            lines.append("- **reasons:**")
            for r in g["reasons"]:
                lines.append(f"    - {'FAIL' if g['verdict'] == 'FAIL' else 'note'}: {r}")
        lines.append("")
    return "\n".join(lines) + "\n"


def write_reports(run: dict, out_dir: Path, *, run_id: str) -> dict:
    """Write report.md + report.json under ``out_dir`` and return the paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "report.md"
    json_path = out_dir / "report.json"
    md_path.write_text(render_markdown(run, run_id=run_id), encoding="utf-8")
    json_path.write_text(json.dumps({"run_id": run_id, **run}, indent=2), encoding="utf-8")
    return {"markdown": md_path, "json": json_path}


def print_summary(run: dict) -> None:
    """Print a compact PASS/FAIL/EMPTY table to stdout (mirrors live_session grading)."""
    n_empty = run.get("n_empty", 0)
    extra = f", {n_empty} EMPTY" if n_empty else ""
    print(f"\n=== real-usage: {run['n_pass']}/{run['n_total']} PASS "
          f"({run.get('n_fail', 0)} FAIL{extra}) ===")
    for g in run["grades"]:
        mark = "EMPTY" if g.get("empty") else ("PASS" if g["passed"] else "FAIL")
        print(f"  [{mark}] {g['fixture']}")
        if g.get("empty") or not g["passed"]:
            for r in g["reasons"]:
                print(f"           - {r}")
