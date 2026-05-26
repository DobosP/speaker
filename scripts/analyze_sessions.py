#!/usr/bin/env python3
"""
Analyze recorded sessions and print an aggregated report.

Usage::

    python scripts/analyze_sessions.py [--session SESSION_ID] [--json]

This script reads all session recordings from ``recordings/`` and prints:

- Per-session stats: turns, barge-ins, false-positive candidates, noise floor
- Cross-session trends: barge-in rate over time, false-positive rate
- Actionable recommendations when thresholds look problematic

Designed to be run by agents or humans after accumulating several sessions.
The ``--json`` flag emits machine-readable output suitable for CI dashboards.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
RECORDINGS_DIR = ROOT / "recordings"


def _fmt_ts(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


def _load_session(session_dir: Path) -> dict | None:
    meta_path = session_dir / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _load_turns(session_dir: Path, meta: dict) -> list[dict]:
    turns = []
    for turn_meta in meta.get("turns", []):
        idx = turn_meta["index"]
        turn_path = session_dir / f"turn_{idx:03d}" / "turn.json"
        if turn_path.exists():
            try:
                with open(turn_path) as f:
                    turns.append(json.load(f))
            except Exception:
                turns.append(turn_meta)
        else:
            turns.append(turn_meta)
    return turns


def analyze_session(session_dir: Path) -> dict:
    """Return a structured analysis dict for one session."""
    meta = _load_session(session_dir)
    if meta is None:
        return {"error": "no metadata.json"}

    turns = _load_turns(session_dir, meta)
    total_turns = len(turns)
    bi_turns = [t for t in turns if t.get("barge_in_fired")]
    no_bi_turns = [t for t in turns if not t.get("barge_in_fired")]

    # False-positive candidates: barge-in with high echo_sim or unvoiced
    fp_candidates = []
    for t in bi_turns:
        for evt in t.get("barge_in_events", []):
            if evt.get("echo_sim", 0) > 0.20 or not evt.get("voiced"):
                fp_candidates.append({
                    "turn": t["index"],
                    "rms": evt.get("rms", 0),
                    "echo_sim": evt.get("echo_sim", 0),
                    "voiced": evt.get("voiced"),
                })
                break

    return {
        "session_id": meta["session_id"],
        "start_time": meta["start_time"],
        "duration_sec": meta.get("duration_sec", 0),
        "noise_floor": meta.get("noise_floor"),
        "profile": meta.get("profile", "unknown"),
        "total_turns": total_turns,
        "barge_in_count": len(bi_turns),
        "no_barge_in_count": len(no_bi_turns),
        "false_positive_candidates": len(fp_candidates),
        "fp_rate": round(len(fp_candidates) / max(total_turns, 1), 4),
        "fp_events": fp_candidates,
        "turn_details": [
            {
                "index": t["index"],
                "barge_in_fired": t.get("barge_in_fired", False),
                "mic_rms_mean": t.get("mic_rms_mean", 0),
                "echo_sim_mean": t.get("echo_sim_mean", 0),
                "barge_in_events": t.get("barge_in_events", []),
            }
            for t in turns
        ],
    }


def print_session_report(analysis: dict, verbose: bool = False):
    sid = analysis["session_id"]
    ts = _fmt_ts(analysis["start_time"])
    dur = analysis["duration_sec"]
    nf = analysis.get("noise_floor")
    nf_str = f"{nf:.4f}" if nf is not None else "unknown"

    fp = analysis["false_positive_candidates"]
    total = analysis["total_turns"]
    bi = analysis["barge_in_count"]
    fp_rate = analysis["fp_rate"]

    health = "✓ OK"
    if fp_rate > 0.50:
        health = "✗ HIGH FP RATE"
    elif fp_rate > 0.20:
        health = "⚠ moderate FP rate"

    print(f"\n{'─'*60}")
    print(f"Session : {sid}")
    print(f"Time    : {ts}  ({dur:.0f}s)")
    print(f"Profile : {analysis['profile']}   noise_floor={nf_str}")
    print(f"Turns   : {total} total  |  {bi} barge-in  |  {fp} FP candidates  → {fp_rate:.1%}  {health}")

    if verbose and analysis["fp_events"]:
        print(f"\n  False-positive candidates:")
        for evt in analysis["fp_events"]:
            print(
                f"    turn_{evt['turn']:03d}: RMS={evt['rms']:.4f}  "
                f"echo_sim={evt['echo_sim']:.2f}  voiced={evt['voiced']}"
            )

    if verbose and analysis["turn_details"]:
        print(f"\n  Turn breakdown:")
        for t in analysis["turn_details"]:
            bi_mark = "⚡" if t["barge_in_fired"] else "  "
            print(
                f"  {bi_mark} turn_{t['index']:03d}: "
                f"rms_mean={t['mic_rms_mean']:.4f}  "
                f"echo_sim_mean={t['echo_sim_mean']:.3f}"
            )


def print_aggregate_report(sessions: list[dict]):
    if not sessions:
        print("No sessions to aggregate.")
        return

    total_turns = sum(s["total_turns"] for s in sessions)
    total_bi = sum(s["barge_in_count"] for s in sessions)
    total_fp = sum(s["false_positive_candidates"] for s in sessions)
    fp_rate = total_fp / max(total_turns, 1)

    print(f"\n{'═'*60}")
    print("AGGREGATE REPORT")
    print(f"{'═'*60}")
    print(f"Sessions       : {len(sessions)}")
    print(f"Total turns    : {total_turns}")
    print(f"Total barge-ins: {total_bi}")
    print(f"FP candidates  : {total_fp} ({fp_rate:.1%} of turns)")

    if fp_rate > 0.50:
        print(
            "\n⚠ HIGH FALSE-POSITIVE RATE. Recommendations:"
            "\n  1. Check noise_floor calibration — if it's very low, the energy gate is too sensitive."
            "\n  2. Increase barge_in_min_delay_after_ref_sec (current default 0.35 s)."
            "\n  3. Run: python scripts/analyze_sessions.py --verbose to see per-turn details."
        )
    elif fp_rate > 0.10:
        print(
            "\n⚠ Moderate FP rate. Consider:"
            "\n  1. Annotate candidate turns in recordings/*/turn_*/turn.json"
            "\n  2. Re-run: python scripts/generate_session_tests.py --all"
        )
    else:
        print("\n✓ False-positive rate within acceptable limits.")

    # Trend (last 5 sessions)
    recent = sessions[-5:]
    if len(recent) >= 2:
        fp_rates = [s["fp_rate"] for s in recent]
        trend = fp_rates[-1] - fp_rates[0]
        direction = "↑ increasing" if trend > 0.05 else ("↓ decreasing" if trend < -0.05 else "→ stable")
        print(f"\nFP trend (last {len(recent)} sessions): {direction}  ({fp_rates[0]:.1%} → {fp_rates[-1]:.1%})")

    print(f"\nTo generate regression tests: python scripts/generate_session_tests.py --all")
    print(f"{'═'*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--session", help="Analyze a specific session ID only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-turn breakdown")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args()

    if args.session:
        session_dir = RECORDINGS_DIR / args.session
        if not session_dir.exists():
            print(f"Session not found: {session_dir}", file=sys.stderr)
            sys.exit(1)
        analysis = analyze_session(session_dir)
        if args.json:
            print(json.dumps(analysis, indent=2))
        else:
            print_session_report(analysis, verbose=args.verbose)
        return

    session_dirs = sorted(RECORDINGS_DIR.glob("session_*/"))
    if not session_dirs:
        print("No sessions recorded yet.  Run:  python main.py --record")
        sys.exit(0)

    analyses = []
    for sdir in session_dirs:
        a = analyze_session(sdir)
        if "error" not in a:
            analyses.append(a)
            if not args.json:
                print_session_report(a, verbose=args.verbose)

    if args.json:
        print(json.dumps({"sessions": analyses, "count": len(analyses)}, indent=2))
    else:
        print_aggregate_report(analyses)


if __name__ == "__main__":
    main()
