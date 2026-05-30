"""Writes the run artifacts: an attributed timeline (JSON), a latency report, and
a human-readable Markdown summary for grading.

The timeline is the heart of "differentiate the speakers": every event carries
``speaker`` (user vs assistant), the exact ``audio`` file that produced/played
it, timestamps, and -- for assistant turns -- the latency breakdown. For a user
turn, ``asr_final`` is what the assistant actually heard (the transcript that the
user's audio generated), closing the loop from played audio -> recognition ->
answer.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path


def write_timeline(events: list[dict], out_dir: Path) -> Path:
    path = Path(out_dir) / "timeline.json"
    path.write_text(json.dumps(events, indent=2))
    return path


def _latencies(events: list[dict]) -> list[dict]:
    rows = []
    for e in events:
        if e.get("speaker") != "assistant":
            continue
        lat = e.get("latency") or {}
        rows.append({
            "turn": e.get("idx"),
            "first_audio_ms": _ms(lat.get("first_audio_latency")),
            "endpoint_ms": _ms(lat.get("endpoint_latency")),
            "final_to_token_ms": _ms(lat.get("final_to_first_token")),
            "token_to_audio_ms": _ms(lat.get("first_token_to_audio")),
            "text": (e.get("text") or "")[:60],
        })
    return rows


def _ms(v):
    return round(v * 1000.0, 1) if isinstance(v, (int, float)) else None


def write_latency_report(events: list[dict], out_dir: Path) -> Path:
    rows = _latencies(events)
    floor = [r["first_audio_ms"] for r in rows if r["first_audio_ms"] is not None]
    agg = {}
    if floor:
        agg = {
            "first_audio_ms_median": round(statistics.median(floor), 1),
            "first_audio_ms_min": min(floor),
            "first_audio_ms_max": max(floor),
            "n": len(floor),
        }
    report = {"per_turn": rows, "aggregate_first_audio": agg}
    path = Path(out_dir) / "latency.json"
    path.write_text(json.dumps(report, indent=2))
    return path


def write_summary(scenario, events: list[dict], out_dir: Path, *, voice: dict) -> Path:
    """A Markdown summary a human reads to grade the run."""
    lines: list[str] = []
    lines.append(f"# Live validation: {scenario.name}\n")
    lines.append(f"**Capability:** {scenario.capability}\n")
    lines.append(f"**Goal:** {scenario.goal}\n")
    lines.append(f"**Synthetic-user voice:** speaker_id={voice.get('speaker_id')} speed={voice.get('speed')}\n")

    lines.append("\n## Conversation (attributed)\n")
    for e in events:
        who = "🗣️  USER " if e["speaker"] == "user" else "🤖 ASSISTANT"
        t = f"{e.get('t_start', 0):6.2f}s"
        lines.append(f"- `{t}` **{who}** — {e.get('text','')!r}")
        if e.get("audio"):
            lines.append(f"    - audio: `{e['audio']}`")
        if e["speaker"] == "user" and e.get("asr_final") is not None:
            lines.append(f"    - heard as (asr_final): `{e['asr_final']}`")
        if e["speaker"] == "assistant" and e.get("latency"):
            la = e["latency"]
            lines.append(
                f"    - latency: first_audio={_ms(la.get('first_audio_latency'))}ms "
                f"(endpoint={_ms(la.get('endpoint_latency'))} | "
                f"final→token={_ms(la.get('final_to_first_token'))} | "
                f"token→audio={_ms(la.get('first_token_to_audio'))})"
            )

    rows = _latencies(events)
    lines.append("\n## Latency (ms)\n")
    lines.append("| turn | first_audio | endpoint | final→token | token→audio |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['turn']} | {r['first_audio_ms']} | {r['endpoint_ms']} | "
            f"{r['final_to_token_ms']} | {r['token_to_audio_ms']} |"
        )

    lines.append("\n## How to grade (from the scenario design)\n")
    lines.append(f"**Expected:** {scenario.expected_behavior}\n")
    if scenario.pass_signals:
        lines.append("\n**Pass signals:**")
        for s in scenario.pass_signals:
            lines.append(f"- ✅ {s}")
    if scenario.failure_modes:
        lines.append("\n**Failure modes:**")
        for s in scenario.failure_modes:
            lines.append(f"- ❌ {s}")

    path = Path(out_dir) / "summary.md"
    path.write_text("\n".join(lines) + "\n")
    return path
