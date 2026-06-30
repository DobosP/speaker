"""Live open-speaker A/B validation report.

Parses one or two run bundles and prints a concise pass/fail comparison table
covering: first-audio timing, TTS underruns, barge-in / self-interrupt events,
DTD/coherence stats, and pre-first-audio barge noise (the _barge_watch_active
gate metric).

Typical workflow
----------------
1. Run the assistant:
       ./session.sh --debug --record
   (or: python -m core --engine sherpa)
2. Reproduce the test scenario (let the assistant speak, then interrupt it).
3. Exit and run this tool:
       python -m tools.live_audio_ab                        # latest run
       python -m tools.live_audio_ab logs/runs/run-A.txt logs/runs/run-B.txt
4. Read the verdict: PASS / WARN / FAIL per run, plus a side-by-side diff if
   two runs are given (useful for before-vs-after A/B comparisons).

Exit codes
----------
0 = all runs PASS
1 = at least one run FAIL
2 = at least one run WARN (and none FAIL)

Pass/fail criteria (tunable via --first-audio-warn-ms etc.)
-------------------------------------------------------------
* self_interrupt  : 0 suspects → PASS
* underruns       : ≤3 total → PASS; 4-10 → WARN; >10 → FAIL
* first_audio_ms  : avg ≤ 300 ms → PASS; 300-600 ms → WARN; > 600 ms → FAIL
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# We import from the tools package; run as `python -m tools.live_audio_ab`
# so sys.path already has the repo root.
from tools.diagnose_run import (
    ParsedRun,
    _FIRST_AUDIO_FAIL_MS,
    _FIRST_AUDIO_WARN_MS,
    _UNDERRUN_FAIL,
    _UNDERRUN_WARN,
    format_report,
    parse_log,
    pass_fail_verdict,
    self_interrupt_summary,
)


# ---------------------------------------------------------------------------
# Helper: auto-discover latest run log
# ---------------------------------------------------------------------------

def _find_latest_logs(n: int = 2) -> list[Path]:
    """Return up to `n` most-recently-modified .txt run logs."""
    run_dir = Path("logs/runs")
    if not run_dir.exists():
        return []
    logs = sorted(
        run_dir.glob("run-*.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return logs[:n]


# ---------------------------------------------------------------------------
# Compact table for a single run
# ---------------------------------------------------------------------------

_ICONS = {"PASS": "✓", "WARN": "~", "FAIL": "✗", "UNKNOWN": "?"}
_COLORS = {
    "PASS": "\033[32m",   # green
    "WARN": "\033[33m",   # yellow
    "FAIL": "\033[31m",   # red
    "UNKNOWN": "\033[90m",
    "RESET": "\033[0m",
}


def _color(verdict: str, text: str, use_color: bool) -> str:
    if not use_color:
        return text
    c = _COLORS.get(verdict, "")
    return f"{c}{text}{_COLORS['RESET']}"


def _row(label: str, verdict: str, detail: str, use_color: bool) -> str:
    icon = _ICONS.get(verdict, "?")
    tag = f"[{icon}] {verdict:<5}"
    colored = _color(verdict, tag, use_color)
    return f"  {colored}  {label:<30} {detail}"


def _ab_table(run: ParsedRun, *, use_color: bool, label: str = "") -> str:
    pf = pass_fail_verdict(run)
    si = self_interrupt_summary(run)
    lines: list[str] = []

    header = f"Run: {run.run_id}"
    if label:
        header = f"{label} — {header}"
    lines.append(header)

    # Overall verdict
    ov = pf["overall"]
    ov_colored = _color(ov, f"OVERALL: {ov}", use_color)
    lines.append(f"  {ov_colored}")
    lines.append("")

    # Self-interrupt
    si_detail = (
        f"suspects={pf['self_interrupt_suspects']} "
        f"detected={si['detected_total']} "
        f"rejected_while_speaking={pf['rejected_while_speaking']}"
    )
    lines.append(_row("self-interrupt", pf["self_interrupt"], si_detail, use_color))

    # Underruns
    ur_detail = f"total={pf['underruns_total']}  (warn>{_UNDERRUN_WARN} fail>{_UNDERRUN_FAIL})"
    lines.append(_row("playback underruns", pf["underrun_verdict"], ur_detail, use_color))

    # First-audio latency
    fa_ms = (
        f"{pf['first_audio_avg_ms']:.0f} ms avg over {pf['first_audio_count']} sentences"
        if pf["first_audio_avg_ms"] is not None
        else "n/a (no playback-open events logged)"
    )
    fa_bounds = f"(warn>{_FIRST_AUDIO_WARN_MS:.0f} ms fail>{_FIRST_AUDIO_FAIL_MS:.0f} ms)"
    lines.append(_row("first-audio latency", pf["first_audio_verdict"], f"{fa_ms}  {fa_bounds}", use_color))

    # DTD summary while speaking
    all_dtd_while_speaking = [
        f for s in run.sentences for f in s.dtd_frames
    ]
    if all_dtd_while_speaking:
        n = len(all_dtd_while_speaking)
        fired = sum(1 for f in all_dtd_while_speaking if f.fired)
        gated = sum(1 for f in all_dtd_while_speaking if f.gated)
        avg_incoh = sum(f.incoh for f in all_dtd_while_speaking) / n
        dtd_detail = (
            f"frames={n} fired={fired} gated={gated} avg_incoh={avg_incoh:.2f}"
        )
        lines.append(_row("DTD frames (speaking)", "UNKNOWN", dtd_detail, False))

    # Heartbeat totals
    if run.heartbeats:
        last = run.heartbeats[-1]
        hb_detail = (
            f"underruns={last.underruns}  "
            f"total_blocks={last.blocks}  "
            f"total_finals={sum(hb.finals for hb in run.heartbeats)}"
        )
        lines.append(_row("heartbeat totals", "UNKNOWN", hb_detail, False))

    # Pre-first-audio barge noise
    if pf["pre_first_audio_noise"] > 0:
        noise_detail = (
            f"{pf['pre_first_audio_noise']} barge events before 'playback opened' "
            f"-- suppressed by _barge_watch_active gate"
        )
        lines.append(_row("pre-first-audio noise", "WARN", noise_detail, use_color))
    else:
        lines.append(_row("pre-first-audio noise", "PASS", "0 events (gate working)", use_color))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Side-by-side diff (two runs)
# ---------------------------------------------------------------------------

def _metric_rows(run: ParsedRun) -> dict:
    pf = pass_fail_verdict(run)
    si = self_interrupt_summary(run)
    return {
        "suspects": si["suspect_count"],
        "detected": si["detected_total"],
        "rejected_while_speaking": pf["rejected_while_speaking"],
        "underruns": pf["underruns_total"],
        "first_audio_avg_ms": pf["first_audio_avg_ms"],
        "pre_first_audio_noise": pf["pre_first_audio_noise"],
        "self_interrupt": pf["self_interrupt"],
        "underrun_verdict": pf["underrun_verdict"],
        "first_audio_verdict": pf["first_audio_verdict"],
        "overall": pf["overall"],
    }


def _diff_table(run_a: ParsedRun, run_b: ParsedRun, *, use_color: bool, label_a: str = "A", label_b: str = "B") -> str:
    m_a = _metric_rows(run_a)
    m_b = _metric_rows(run_b)
    col_w = 22
    lines: list[str] = [
        f"{'Metric':<30} {label_a:<{col_w}} {label_b:<{col_w}}",
        "-" * (30 + col_w * 2 + 2),
    ]

    def _row_diff(label: str, key: str, verdict_key: str | None = None) -> str:
        va = str(m_a[key]) if m_a[key] is not None else "n/a"
        vb = str(m_b[key]) if m_b[key] is not None else "n/a"
        diff = ""
        try:
            delta = float(m_b[key]) - float(m_a[key])  # type: ignore[arg-type]
            diff = f" ({'+' if delta > 0 else ''}{delta:.1f})"
        except (TypeError, ValueError):
            pass
        vb_str = f"{vb}{diff}"
        if verdict_key:
            verdict_b = m_b[verdict_key]
            vb_str = _color(verdict_b, vb_str, use_color)
        return f"{label:<30} {va:<{col_w}} {vb_str}"

    lines.append(_row_diff("self-interrupt suspects", "suspects", "self_interrupt"))
    lines.append(_row_diff("detected barges", "detected"))
    lines.append(_row_diff("rejected while speaking", "rejected_while_speaking"))
    lines.append(_row_diff("underruns total", "underruns", "underrun_verdict"))
    lines.append(_row_diff("first-audio avg ms", "first_audio_avg_ms", "first_audio_verdict"))
    lines.append(_row_diff("pre-first-audio noise", "pre_first_audio_noise"))
    lines.append("")
    ov_a = _color(m_a["overall"], f"OVERALL: {m_a['overall']}", use_color)
    ov_b = _color(m_b["overall"], f"OVERALL: {m_b['overall']}", use_color)
    lines.append(f"{'Overall verdict':<30} {ov_a:<{col_w+10}} {ov_b}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Live open-speaker A/B validation report. "
            "Parses one or two run .txt logs and prints a pass/fail table."
        )
    )
    parser.add_argument(
        "logs", nargs="*",
        help=(
            "Path(s) to run-<id>.txt log file(s). "
            "If omitted, auto-discovers the two most recent runs in logs/runs/."
        ),
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Print the full diagnose_run report for each run (default: compact table only)",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable ANSI color output",
    )
    parser.add_argument(
        "--label-a", default="A (before)",
        help="Label for the first run (default: 'A (before)')",
    )
    parser.add_argument(
        "--label-b", default="B (after)",
        help="Label for the second run (default: 'B (after)')",
    )
    args = parser.parse_args(argv)

    use_color = not args.no_color and sys.stdout.isatty()

    log_paths: list[Path] = []
    if args.logs:
        for p in args.logs:
            pp = Path(p)
            if not pp.exists():
                print(f"ERROR: log not found: {p}", file=sys.stderr)
                return 1
            log_paths.append(pp)
    else:
        log_paths = _find_latest_logs(2)
        if not log_paths:
            print("ERROR: no run logs found in logs/runs/; pass a path explicitly", file=sys.stderr)
            return 1
        print(f"Auto-discovered: {', '.join(str(p) for p in log_paths)}\n")

    runs = [parse_log(str(p)) for p in log_paths]

    if args.full:
        for run, lp in zip(runs, log_paths):
            print(f"\n{'='*60}")
            print(f"Full report: {lp}")
            print('='*60)
            print(format_report(run))

    if len(runs) == 1:
        print(_ab_table(runs[0], use_color=use_color, label=args.label_a))
    else:
        print(f"=== Individual Run Reports ===\n")
        print(_ab_table(runs[0], use_color=use_color, label=args.label_a))
        print()
        print(_ab_table(runs[1], use_color=use_color, label=args.label_b))
        print(f"\n=== Side-by-Side Comparison ===\n")
        print(_diff_table(runs[0], runs[1], use_color=use_color, label_a=args.label_a, label_b=args.label_b))

    print("\n--- How to run a live A/B ---")
    print("  # Launch the assistant:")
    print("  ./session.sh --debug --record")
    print("  # (or: python -m core --engine sherpa)")
    print("  # Let it speak a long reply, then interrupt it.")
    print("  # Exit (Ctrl-C), then re-run this tool:")
    print("  python -m tools.live_audio_ab")
    print("  # For full diagnosis:")
    print("  python -m tools.diagnose_run logs/runs/run-<id>.txt")

    # Exit code
    verdicts = [pass_fail_verdict(r)["overall"] for r in runs]
    if any(v == "FAIL" for v in verdicts):
        return 1
    if any(v == "WARN" for v in verdicts):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
