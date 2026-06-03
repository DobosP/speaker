"""Run-history inventory: scan the committed ``logs/runs/`` bundles and render a
single OVERVIEW of every run -- duration, turns, LLM requests, errors, stuck
hints, and (where a WAV was recorded) the captured audio level, with empty /
digitally-silent captures flagged as prune candidates.

The scan half reads the per-run ``*.summary.json`` and measures any sibling WAV
(numpy, via :func:`tools.real_usage.runner.load_and_measure`). The render half is
PURE (plain dicts -> Markdown), so it is unit-testable in CI without numpy/audio,
exactly like :mod:`tools.real_usage.report`.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any, Optional

from . import report


def _fmt_when(run_id: str) -> str:
    """'run-20260602-231913' -> '2026-06-02 23:19:13' (best-effort; pass through
    anything that doesn't match the run-YYYYMMDD-HHMMSS convention)."""
    stem = run_id[4:] if run_id.startswith("run-") else run_id
    parts = stem.split("-")
    if len(parts) == 2 and len(parts[0]) == 8 and len(parts[1]) == 6:
        d, t = parts
        return f"{d[0:4]}-{d[4:6]}-{d[6:8]} {t[0:2]}:{t[2:4]}:{t[4:6]}"
    return run_id


def _snippet(transcript: Any, n: int = 3, width: int = 60) -> str:
    """A short join of the first few USER utterances (what was actually said).
    Sanitises ``|`` / newlines so a cell can't break the Markdown table."""
    if not isinstance(transcript, list):
        return ""
    texts = [
        str(e.get("text", "")).strip().replace("|", "/").replace("\n", " ")
        for e in transcript
        if isinstance(e, dict) and e.get("role") == "user" and str(e.get("text", "")).strip()
    ]
    joined = " / ".join(texts[:n])
    return joined if len(joined) <= width else joined[: width - 3] + "..."


def scan_runs(runs_dir: str = "logs/runs", *, measure: bool = True) -> list[dict]:
    """Read every ``run-*.summary.json`` under ``runs_dir`` into a row dict; where
    a sibling ``.wav`` exists and ``measure`` is set, attach its level + a
    ``silent`` flag. Sorted by run id (chronological). File/measure errors on one
    run never abort the scan."""
    rows: list[dict] = []
    for sp in sorted(glob.glob(str(Path(runs_dir) / "run-*.summary.json"))):
        try:
            data = json.loads(Path(sp).read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - a corrupt summary shouldn't kill the scan
            rows.append({"run_id": Path(sp).stem.replace(".summary", ""),
                         "parse_error": str(exc)})
            continue
        counts = data.get("counts") or {}
        meta = data.get("meta") or {}
        llm = data.get("llm") or {}
        errors = data.get("errors") or []
        # Derive the bundle base ('run-YYYYMMDD-HHMMSS') from the summary FILENAME,
        # not the JSON 'run_id' field -- the latter omits the 'run-' prefix, so a
        # WAV lookup built from it would always miss.
        base = Path(sp).name[: -len(".summary.json")]
        run_id = base
        wav_path = Path(sp).with_name(base + ".wav")
        has_wav = wav_path.exists()
        row: dict = {
            "run_id": run_id,
            "when": _fmt_when(run_id),
            "duration_sec": data.get("duration_sec"),
            "turns": counts.get("turns"),
            "transcript_entries": counts.get("transcript_entries"),
            "llm_requests": counts.get("llm_requests"),
            "errors": counts.get("errors", len(errors)),
            "warnings": counts.get("warnings"),
            "stuck_hints": len(data.get("stuck_hints") or []),
            "engine": meta.get("engine"),
            "llm": meta.get("llm"),
            "mode": meta.get("mode"),
            "llm_avg_sec": llm.get("avg_time_sec"),
            "snippet": _snippet(data.get("transcript")),
            "first_error": (errors[0].get("message") if errors and isinstance(errors[0], dict) else None),
            "has_wav": has_wav,
            "wav_rms": None,
            "wav_peak": None,
            "wav_dur": None,
            "silent": False,
        }
        if has_wav and measure:
            try:
                from .runner import load_and_measure

                lvl = load_and_measure(wav_path)
                row["wav_rms"] = lvl["rms"]
                row["wav_peak"] = lvl["peak"]
                row["wav_dur"] = lvl["duration_sec"]
                row["silent"] = report.is_silent_input(lvl["rms"], lvl["peak"])
            except Exception as exc:  # noqa: BLE001
                row["measure_error"] = str(exc)
        rows.append(row)
    return rows


def empty_wavs(rows: list[dict]) -> list[str]:
    """Run ids whose recorded WAV is digitally silent (prune candidates)."""
    return [r["run_id"] for r in rows if r.get("has_wav") and r.get("silent")]


def _cell(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def render_inventory_markdown(rows: list[dict], *, title: str = "Run history overview") -> str:
    """Render the scanned rows as a Markdown overview. PURE -- no IO, no numpy."""
    n = len(rows)
    n_wav = sum(1 for r in rows if r.get("has_wav"))
    silent = [r for r in rows if r.get("has_wav") and r.get("silent")]
    with_err = [r for r in rows if (r.get("errors") or 0)]

    out: list[str] = []
    out.append(f"# {title}\n")
    out.append(
        f"**{n} runs** -- {n_wav} with a recorded WAV, "
        f"{len(silent)} digitally-silent/empty, {len(with_err)} with errors.\n"
    )
    if silent:
        names = ", ".join(f"`{r['run_id']}`" for r in silent)
        out.append(
            f"\n> **Empty captures (prune candidates):** {names} -- recorded WAV is "
            f"digitally silent (dead/muted mic); no usable audio.\n"
        )

    out.append("\n## Runs\n")
    out.append("| run | when | dur(s) | turns | llm req | errors | stuck | audio (rms/peak) | status | said |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("parse_error"):
            err = str(r["parse_error"]).replace("|", "/").replace("\n", " ")
            out.append(f"| {r.get('run_id')} | - | - | - | - | - | - | - | **PARSE ERR** | {err} |")
            continue
        if not r.get("has_wav"):
            status, audio = "no-wav", "-"
        elif r.get("silent"):
            status, audio = "**EMPTY**", f"{_cell(r.get('wav_rms'))}/{_cell(r.get('wav_peak'))}"
        else:
            status, audio = "audio", f"{_cell(r.get('wav_rms'))}/{_cell(r.get('wav_peak'))}"
        out.append(
            f"| {r.get('run_id')} | {r.get('when')} | {_cell(r.get('duration_sec'))} | "
            f"{_cell(r.get('turns'))} | {_cell(r.get('llm_requests'))} | {_cell(r.get('errors'))} | "
            f"{_cell(r.get('stuck_hints'))} | {audio} | {status} | {r.get('snippet') or '-'} |"
        )
    out.append("")
    return "\n".join(out) + "\n"


def write_inventory(rows: list[dict], out_path: Path, *, title: str = "Run history overview") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_inventory_markdown(rows, title=title), encoding="utf-8")
    return out_path
