from __future__ import annotations

import html
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from tools.specsim.simulate import (
    BARGE_IN_BUDGET,
    FIRST_AUDIO_BUDGET,
    SCENARIOS,
    classify,
    simulate_turn,
)
from tools.specsim.specs import CATALOG

from .runner import TurnSample

_METRICS = (
    ("first_audio_latency", "First audio (speech end -> assistant speaks)"),
    ("endpoint_latency", "Endpoint (speech end -> ASR final)"),
    ("final_to_first_token", "ASR final -> first LLM token"),
    ("first_token_to_audio", "First token -> first audio"),
    ("barge_in_latency", "Barge-in (interrupt -> stop)"),
)

# Which specsim spec to calibrate a given bench profile against.
_PROFILE_TO_SPEC = {
    "phone": "Android phone (12 GB)",
    "desktop": "RTX 4090 Laptop",
}

_STATUS_COLORS = {"good": "#1e7e34", "ok": "#b8860b", "fail": "#c0392b", "n/a": "#777"}


def _values(samples: list[TurnSample], attr: str) -> list[float]:
    out = []
    for s in samples:
        v = getattr(s.record, attr)
        if v is not None:
            out.append(float(v))
    return out


def _percentile(values: list[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def summarize(samples: list[TurnSample]) -> dict[str, object]:
    """Per-metric count/median/p90/min/max plus a simple correctness tally."""
    stats: dict[str, object] = {}
    for attr, _label in _METRICS:
        vals = _values(samples, attr)
        stats[attr] = {
            "count": len(vals),
            "median": round(statistics.median(vals), 4) if vals else None,
            "p90": round(_percentile(vals, 0.9), 4) if vals else None,
            "min": round(min(vals), 4) if vals else None,
            "max": round(max(vals), 4) if vals else None,
        }
    responded = sum(1 for s in samples if s.responded)
    stats["turns"] = len(samples)
    stats["responded"] = responded
    return stats


def _modelled(spec_name: str) -> dict[str, Optional[float]]:
    spec = next((s for s in CATALOG if s.name == spec_name), None)
    quick = next((sc for sc in SCENARIOS if sc.name == "quick"), SCENARIOS[0])
    barge = next((sc for sc in SCENARIOS if sc.barge_in), None)
    if spec is None:
        return {"first_audio_latency": None, "barge_in_latency": None}
    fa = simulate_turn(spec, quick).first_audio_latency
    bi = simulate_turn(spec, barge).barge_in_stop if barge else None
    return {"first_audio_latency": fa, "barge_in_latency": bi}


def calibration(stats: dict[str, object], profile: str) -> dict[str, object]:
    """Measured median vs the specsim model, classified against the budgets."""
    spec_name = _PROFILE_TO_SPEC.get(profile, "Android phone (12 GB)")
    modelled = _modelled(spec_name)
    rows = []
    for attr, budget in (
        ("first_audio_latency", FIRST_AUDIO_BUDGET),
        ("barge_in_latency", BARGE_IN_BUDGET),
    ):
        measured = stats[attr]["median"] if isinstance(stats.get(attr), dict) else None  # type: ignore[index]
        rows.append(
            {
                "metric": attr,
                "measured_median": measured,
                "modelled": modelled.get(attr),
                "budget": list(budget),
                "status": classify(measured, budget) if measured is not None else "n/a",
            }
        )
    return {"spec": spec_name, "rows": rows}


# --- HTML rendering (self-contained, in the spirit of tools/specsim/report) ---
def _esc(text: object) -> str:
    return html.escape(str(text))


def _fmt(value: Optional[float]) -> str:
    return f"{value:.3f}s" if value is not None else "—"


def render_html(profile: str, samples: list[TurnSample], stats: dict[str, object]) -> str:
    cal = calibration(stats, profile)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    cal_rows = "".join(
        f"<tr><td>{_esc(r['metric'])}</td>"
        f"<td>{_fmt(r['measured_median'])}</td>"
        f"<td>{_fmt(r['modelled'])}</td>"
        f"<td>&le;{r['budget'][0]} / &le;{r['budget'][1]}</td>"
        f"<td><span class='chip' style='background:{_STATUS_COLORS.get(str(r['status']), '#777')}'>"
        f"{_esc(r['status'])}</span></td></tr>"
        for r in cal["rows"]
    )

    stat_rows = "".join(
        f"<tr><td>{_esc(label)}</td>"
        f"<td>{(stats[attr] or {}).get('count', 0)}</td>"
        f"<td>{_fmt((stats[attr] or {}).get('median'))}</td>"
        f"<td>{_fmt((stats[attr] or {}).get('p90'))}</td>"
        f"<td>{_fmt((stats[attr] or {}).get('min'))}</td>"
        f"<td>{_fmt((stats[attr] or {}).get('max'))}</td></tr>"
        for attr, label in _METRICS
    )

    sample_rows = "".join(
        f"<tr><td>{_esc(s.name)}</td><td>{_esc(s.expectation)}</td>"
        f"<td>{_esc(s.transcript)}</td>"
        f"<td>{'yes' if s.responded else 'no'}</td>"
        f"<td>{_fmt(s.record.first_audio_latency)}</td>"
        f"<td>{_fmt(s.record.barge_in_latency)}</td></tr>"
        for s in samples
    )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Speaker perf — {_esc(profile)}</title>
<style>
 body{{font:14px/1.5 system-ui,sans-serif;margin:2rem;color:#1a1a1a}}
 h1{{font-size:1.4rem}} h2{{font-size:1.1rem;margin-top:1.6rem}}
 table{{border-collapse:collapse;margin:.6rem 0;min-width:520px}}
 th,td{{border:1px solid #ddd;padding:.35rem .6rem;text-align:left}}
 th{{background:#f4f4f4}}
 .chip{{color:#fff;border-radius:4px;padding:.1rem .5rem;font-size:.8rem}}
 .muted{{color:#666}}
</style></head><body>
<h1>Speaker real-model latency — profile <code>{_esc(profile)}</code></h1>
<p class="muted">{len(samples)} turn(s), {stats.get('responded', 0)} answered.
 Generated {generated}. Measured on this run's CPU — compare trends, and
 calibrate against the on-device model below rather than reading as phone-absolute.</p>

<h2>Calibration vs specsim budget ({_esc(cal['spec'])})</h2>
<table><tr><th>metric</th><th>measured median</th><th>modelled</th>
 <th>budget good/ok</th><th>status</th></tr>{cal_rows}</table>

<h2>Measured latency distribution</h2>
<table><tr><th>stage</th><th>n</th><th>median</th><th>p90</th><th>min</th><th>max</th></tr>
 {stat_rows}</table>

<h2>Per-turn</h2>
<table><tr><th>fixture</th><th>expectation</th><th>transcript</th>
 <th>answered</th><th>first audio</th><th>barge-in</th></tr>{sample_rows}</table>
</body></html>"""


def write_reports(out_dir: Path, profile: str, samples: list[TurnSample]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = summarize(samples)
    cal = calibration(stats, profile)
    (out_dir / "index.html").write_text(render_html(profile, samples, stats), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "profile": profile,
                "stats": stats,
                "calibration": cal,
                "turns": [s.as_dict() for s in samples],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_dir / "index.html"


def markdown_summary(profile: str, samples: list[TurnSample]) -> str:
    """Compact summary for a CI job step ($GITHUB_STEP_SUMMARY)."""
    stats = summarize(samples)
    cal = calibration(stats, profile)
    lines = [
        f"## Speaker perf — `{profile}`",
        "",
        f"{stats['turns']} turn(s), {stats['responded']} answered.",
        "",
        "| metric | measured median | modelled | budget | status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in cal["rows"]:
        lines.append(
            f"| {r['metric']} | {_fmt(r['measured_median'])} | {_fmt(r['modelled'])} "
            f"| ≤{r['budget'][0]} / ≤{r['budget'][1]} | {r['status']} |"
        )
    return "\n".join(lines) + "\n"
