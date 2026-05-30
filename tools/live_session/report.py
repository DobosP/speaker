"""Writes the run artifacts: an attributed timeline (JSON), a latency report, an
honest over-the-air STT grade, and a human-readable Markdown summary for grading.

The timeline is the heart of "differentiate the speakers": every event carries
``speaker`` (user vs assistant), the exact ``audio`` file that produced/played
it, timestamps, and -- for assistant turns -- the latency breakdown. For a user
turn, ``asr_final`` is what the assistant actually heard (the transcript that the
user's audio generated), closing the loop from played audio -> recognition ->
answer.

Two pieces of the acoustic full-duplex story live here as PURE, stdlib-only,
unit-testable functions (no audio/model deps):

* :func:`stt_score` grades what the mic heard against the scripted line, so the
  over-the-air STT quality is a number in [0, 1], not a vibe a human eyeballs.
* :func:`summarize_capture` turns the driver's continuous-capture observations
  (recording duration vs wall-clock, the capture-silent latch, partials produced
  while the user audio played) into an explicit full_duplex PASS/FAIL verdict.
"""
from __future__ import annotations

import json
import re
import statistics
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

# A heard turn counts as "correct" at or above this blended-similarity score.
# It's a judgment call; grade.json always carries the raw score so a human can
# recalibrate without touching code.
HEARD_OK_THRESHOLD = 0.6
# The continuous-capture verdict treats the recording-vs-wallclock comparison as
# a tolerance band (device clock skew / resampling drift), not exact equality.
CAPTURE_COVERAGE_MIN = 0.85


# --- over-the-air STT grading (pure) -----------------------------------------


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace -- so 'Paris.' and
    'paris' compare equal and only the words matter."""
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _word_f1(a: str, b: str) -> float:
    """Token-overlap F1 between two normalized strings (order-insensitive, so a
    re-ordered-but-correct transcript still scores well)."""
    wa, wb = a.split(), b.split()
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    from collections import Counter

    ca, cb = Counter(wa), Counter(wb)
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(wb)
    recall = overlap / len(wa)
    return 2 * precision * recall / (precision + recall)


def stt_score(scripted: str, heard: Optional[str]) -> float:
    """Blend a difflib character-sequence ratio with a word-overlap F1 between the
    scripted line (truth) and what the mic heard (``asr_final``). Returns [0, 1]:
    1.0 for an exact match (after normalization), ~0 for total garble, monotonic
    in between. ``None``/empty heard -> 0.0 (the mic produced nothing)."""
    s = _normalize(scripted)
    h = _normalize(heard or "")
    if not s and not h:
        return 1.0
    if not h or not s:
        return 0.0
    seq = SequenceMatcher(None, s, h).ratio()
    f1 = _word_f1(s, h)
    return round(0.5 * seq + 0.5 * f1, 4)


def _grade_rows(events: list[dict]) -> list[dict]:
    """One row per user turn: scripted vs heard, the STT score, the heard-ok
    boolean, and (paired in order) the answer's first-audio latency."""
    rows: list[dict] = []
    assistant_lat = [
        _ms((e.get("latency") or {}).get("first_audio_latency"))
        for e in events if e.get("speaker") == "assistant"
    ]
    ai = 0
    for e in events:
        if e.get("speaker") != "user":
            continue
        scripted = e.get("text") or ""
        heard = e.get("asr_final")
        score = stt_score(scripted, heard)
        first_audio = assistant_lat[ai] if ai < len(assistant_lat) else None
        ai += 1
        rows.append({
            "turn": e.get("idx"),
            "scripted": scripted,
            "heard": heard,
            "stt_score": score,
            "heard_ok": score >= HEARD_OK_THRESHOLD,
            "first_audio_ms": first_audio,
            "capture": e.get("capture"),
        })
    return rows


def _grade_aggregate(rows: list[dict]) -> dict:
    scores = [r["stt_score"] for r in rows]
    n = len(scores)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "n_correct": sum(1 for r in rows if r["heard_ok"]),
        "stt_score_median": round(statistics.median(scores), 4),
        "stt_score_min": round(min(scores), 4),
        "stt_score_mean": round(statistics.fmean(scores), 4),
    }


# --- continuous-capture / full-duplex verdict (pure) -------------------------


def summarize_capture(
    *,
    rec_seconds: float,
    wall_seconds: float,
    partials_during_user_total: int,
    capture_silent_warned: bool,
) -> dict:
    """Grade the full-duplex / always-on-capture story from the driver's
    observations. PASS requires all of:

    * the recorder covered ~the whole wall-clock session (it writes EVERY capture
      block, even during the assistant's own playback, so its duration is the
      ground truth that capture never paused);
    * no >5 s "capture silent" gap was ever latched (the audio thread stayed
      alive on its heartbeat cadence);
    * at least one partial appeared WHILE a user line was playing (the recognizer
      transcribed live, in parallel with output -> genuinely full-duplex).

    Coverage is a tolerance band, not exact equality, to avoid false FAILs from
    device clock skew / resampling drift on an otherwise healthy run."""
    coverage = (rec_seconds / wall_seconds) if wall_seconds > 0 else 0.0
    covered = coverage >= CAPTURE_COVERAGE_MIN
    transcribed_during_user = partials_during_user_total > 0
    ok = bool(covered and not capture_silent_warned and transcribed_during_user)
    return {
        "full_duplex": "ok" if ok else "FAIL",
        "ok": ok,
        "recording_seconds": round(float(rec_seconds), 2),
        "wall_seconds": round(float(wall_seconds), 2),
        "coverage": round(float(coverage), 3),
        "covered_whole_session": covered,
        "capture_silent_warned": bool(capture_silent_warned),
        "partials_during_user_total": int(partials_during_user_total),
        "transcribed_during_user": transcribed_during_user,
    }


# --- artifact writers --------------------------------------------------------


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


def write_grade(events: list[dict], out_dir: Path, *, capture: Optional[dict] = None) -> dict:
    """Emit grade.json -- the honest over-the-air STT grade (per turn + aggregate)
    plus the session full-duplex verdict. Returns the dict written."""
    rows = _grade_rows(events)
    report = {
        "per_turn": rows,
        "aggregate": _grade_aggregate(rows),
        "full_duplex": capture or {},
    }
    path = Path(out_dir) / "grade.json"
    path.write_text(json.dumps(report, indent=2))
    return report


def write_summary(
    scenario, events: list[dict], out_dir: Path, *, voice: dict,
    capture: Optional[dict] = None,
) -> Path:
    """A Markdown summary a human reads to grade the run."""
    lines: list[str] = []
    lines.append(f"# Live validation: {scenario.name}\n")
    lines.append(f"**Capability:** {scenario.capability}\n")
    lines.append(f"**Goal:** {scenario.goal}\n")
    lines.append(
        f"**Synthetic-user voice:** speaker_id={voice.get('speaker_id')} "
        f"speed={voice.get('speed')} volume={voice.get('volume')}\n"
    )

    lines.append("\n## Conversation (attributed)\n")
    for e in events:
        who = "USER " if e["speaker"] == "user" else "ASSISTANT"
        t = f"{e.get('t_start', 0):6.2f}s"
        lines.append(f"- `{t}` **{who}** - {e.get('text','')!r}")
        if e.get("audio"):
            lines.append(f"    - audio: `{e['audio']}`")
        if e["speaker"] == "user" and e.get("asr_final") is not None:
            lines.append(f"    - heard as (asr_final): `{e['asr_final']}`")
        if e["speaker"] == "assistant" and e.get("latency"):
            la = e["latency"]
            lines.append(
                f"    - latency: first_audio={_ms(la.get('first_audio_latency'))}ms "
                f"(endpoint={_ms(la.get('endpoint_latency'))} | "
                f"final->token={_ms(la.get('final_to_first_token'))} | "
                f"token->audio={_ms(la.get('first_token_to_audio'))})"
            )

    rows = _latencies(events)
    lines.append("\n## Latency (ms)\n")
    lines.append("| turn | first_audio | endpoint | final->token | token->audio |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['turn']} | {r['first_audio_ms']} | {r['endpoint_ms']} | "
            f"{r['final_to_token_ms']} | {r['token_to_audio_ms']} |"
        )

    # Honest over-the-air STT grade.
    grade_rows = _grade_rows(events)
    agg = _grade_aggregate(grade_rows)
    lines.append("\n## STT accuracy (over-the-air)\n")
    if agg.get("n"):
        lines.append(
            f"**Headline:** median STT score {agg['stt_score_median']} "
            f"(min {agg['stt_score_min']}); heard correctly "
            f"{agg['n_correct']}/{agg['n']} turns (>= {HEARD_OK_THRESHOLD}).\n"
        )
    lines.append("| turn | scripted | heard | score | ok |")
    lines.append("|---|---|---|---|---|")
    for r in grade_rows:
        heard = (r["heard"] or "").replace("|", "/")
        scripted = (r["scripted"] or "").replace("|", "/")
        ok = "yes" if r["heard_ok"] else "NO"
        lines.append(
            f"| {r['turn']} | {scripted[:40]!r} | {heard[:40]!r} | "
            f"{r['stt_score']} | {ok} |"
        )

    # Full-duplex / continuous-capture verdict.
    lines.append("\n## Full-duplex / continuous capture\n")
    if capture:
        verdict = capture.get("full_duplex", "n/a")
        lines.append(f"**full_duplex: {verdict}**\n")
        lines.append(
            f"- recording covered {capture.get('recording_seconds')}s of "
            f"{capture.get('wall_seconds')}s wall-clock "
            f"(coverage {capture.get('coverage')}, "
            f"covered_whole_session={capture.get('covered_whole_session')})"
        )
        lines.append(
            f"- capture-silent gap (>5s) seen: {capture.get('capture_silent_warned')} "
            f"(absence == the always-on proof)"
        )
        lines.append(
            f"- partials produced WHILE the user spoke: "
            f"{capture.get('partials_during_user_total')} "
            f"(transcribed_during_user={capture.get('transcribed_during_user')})"
        )
        lines.append(
            "- the recorder writes every capture block (even during the assistant's "
            "own playback), so its duration proves capture ran in+out at once."
        )
    else:
        lines.append("_no capture verdict recorded for this run._")

    lines.append("\n## How to grade (from the scenario design)\n")
    lines.append(f"**Expected:** {scenario.expected_behavior}\n")
    if scenario.pass_signals:
        lines.append("\n**Pass signals:**")
        for s in scenario.pass_signals:
            lines.append(f"- [pass] {s}")
    if scenario.failure_modes:
        lines.append("\n**Failure modes:**")
        for s in scenario.failure_modes:
            lines.append(f"- [fail] {s}")

    path = Path(out_dir) / "summary.md"
    path.write_text("\n".join(lines) + "\n")
    return path
