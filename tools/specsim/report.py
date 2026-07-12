"""Render the spec-simulation results as a single self-contained HTML file."""
from __future__ import annotations

import html
from datetime import datetime

from .simulate import (
    BARGE_IN_BUDGET,
    FIRST_AUDIO_BUDGET,
    SCENARIOS,
    Scenario,
    TurnResult,
    classify,
    simulate_turn,
)
from .specs import MODEL_FOOTPRINTS_GB, MachineSpec

_KIND_COLORS = {
    "speech": "#4f7cff",
    "endpoint": "#9aa0a6",
    "ttft": "#f5a623",
    "gen": "#f7c948",
    "ttfa": "#b07cf0",
    "play": "#34c759",
}
_STATUS_COLORS = {
    "good": "#1e7e34",
    "ok": "#b8860b",
    "tight": "#b8860b",
    "fail": "#c0392b",
}
_BAR_WIDTH = 820  # px the longest segment-row is scaled to fill
_BAR_HEIGHT = 26


def _esc(text: object) -> str:
    return html.escape(str(text))


def _chip(text: str, status: str) -> str:
    color = _STATUS_COLORS.get(status, "#555")
    return f'<span class="chip" style="background:{color}">{_esc(text)}</span>'


def _timeline_svg(result: TurnResult) -> str:
    total = result.total or 1.0
    pps = _BAR_WIDTH / total
    parts = [f'<svg width="{_BAR_WIDTH}" height="{_BAR_HEIGHT}" class="tl">']
    for seg in result.segments:
        x = seg.start * pps
        w = max(0.5, seg.duration * pps)
        color = _KIND_COLORS.get(seg.kind, "#ccc")
        title = f"{seg.label}: {seg.duration:.2f}s"
        parts.append(
            f'<rect x="{x:.1f}" y="0" width="{w:.1f}" height="{_BAR_HEIGHT}" '
            f'fill="{color}"><title>{_esc(title)}</title></rect>'
        )
    # Marker at first assistant audio (speech_end + first_audio_latency).
    fa_abs = result.segments[0].duration + result.first_audio_latency
    mx = fa_abs * pps
    parts.append(
        f'<line x1="{mx:.1f}" y1="-2" x2="{mx:.1f}" y2="{_BAR_HEIGHT + 2}" '
        f'stroke="#111" stroke-width="2" stroke-dasharray="3,2"><title>first audio</title></line>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _legend() -> str:
    items = "".join(
        f'<span class="lg"><i style="background:{color}"></i>{_esc(kind)}</span>'
        for kind, color in _KIND_COLORS.items()
    )
    return (
        f'<div class="legend">{items}'
        '<span class="lg"><i class="marker"></i>first audio</span></div>'
    )


def _model_fit_table(specs: tuple[MachineSpec, ...]) -> str:
    rows = []
    for spec in specs:
        fast_status = spec.fit_status(spec.fast_model)
        fast_footprint = MODEL_FOOTPRINTS_GB.get(spec.fast_model, 0.0)
        main_status = spec.fit_status(spec.main_model)
        main_footprint = MODEL_FOOTPRINTS_GB.get(spec.main_model, 0.0)
        combined_status = "good" if spec.configured_roles_fit() else "fail"
        shared = " (shared)" if spec.shares_model_across_roles else ""
        largest = spec.largest_fitting_model() or "(none)"
        rows.append(
            "<tr>"
            f"<td>{_esc(spec.name)}</td>"
            f"<td>{_esc(spec.platform)}</td>"
            f"<td>{spec.cores}</td>"
            f"<td>{spec.ram_gb:g} GB</td>"
            f"<td>{spec.model_budget_gb:g} GB</td>"
            f"<td>{_esc(spec.fast_model)} ({fast_footprint:g} GB) "
            f"{_chip(fast_status, fast_status)}</td>"
            f"<td>{_esc(spec.main_model)}{shared} ({main_footprint:g} GB) "
            f"{_chip(main_status, main_status)}</td>"
            f"<td>{spec.configured_footprint_gb:g} GB "
            f"{_chip(combined_status, combined_status)}</td>"
            f"<td>{_esc(spec.fast_tokens_per_sec)} tok/s</td>"
            f"<td>{_esc(largest)}</td>"
            "</tr>"
        )
    return (
        "<h2>Model fit per device</h2>"
        '<table><thead><tr>'
        "<th>Device</th><th>Platform</th><th>Cores</th><th>RAM</th>"
        "<th>Model budget</th><th>Fast / ordinary model</th>"
        "<th>Main / complex model</th><th>Combined role weights</th>"
        "<th>Estimated fast-path speed</th>"
        "<th>Largest that fits</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _matrix(specs: tuple[MachineSpec, ...], scenarios: tuple[Scenario, ...]) -> str:
    speak = [s for s in scenarios if not s.barge_in]
    barge = [s for s in scenarios if s.barge_in]
    head = "".join(f"<th>{_esc(s.name)}<br>first audio</th>" for s in speak)
    head += "".join(f"<th>{_esc(s.name)}<br>stop latency</th>" for s in barge)
    rows = []
    for spec in specs:
        cells = []
        for sc in speak:
            r = simulate_turn(spec, sc)
            status = classify(r.first_audio_latency, FIRST_AUDIO_BUDGET)
            cells.append(
                f'<td style="background:{_STATUS_COLORS[status]}">'
                f"{r.first_audio_latency:.2f}s</td>"
            )
        for sc in barge:
            r = simulate_turn(spec, sc)
            status = classify(r.barge_in_stop or 0.0, BARGE_IN_BUDGET)
            cells.append(
                f'<td style="background:{_STATUS_COLORS[status]}">'
                f"{r.barge_in_stop:.2f}s</td>"
            )
        rows.append(f"<tr><td class=\"name\">{_esc(spec.name)}</td>{''.join(cells)}</tr>")
    return (
        "<h2>Responsiveness matrix</h2>"
        f"<p class=\"note\">first audio budget: &le;{FIRST_AUDIO_BUDGET[0]}s good, "
        f"&le;{FIRST_AUDIO_BUDGET[1]}s ok &middot; barge-in stop: "
        f"&le;{BARGE_IN_BUDGET[0]}s good, &le;{BARGE_IN_BUDGET[1]}s ok</p>"
        f'<table class="matrix"><thead><tr><th>Device</th>{head}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _spec_cards(specs: tuple[MachineSpec, ...], scenarios: tuple[Scenario, ...]) -> str:
    cards = []
    for spec in specs:
        rows = []
        for sc in scenarios:
            r = simulate_turn(spec, sc)
            extra = (
                f"stop {r.barge_in_stop:.2f}s"
                if r.barge_in_stop is not None
                else f"first audio {r.first_audio_latency:.2f}s &middot; "
                f"done {r.response_complete:.2f}s"
            )
            rows.append(
                '<div class="row">'
                f'<div class="rowlabel">{_esc(sc.name)}'
                f'<small>{_esc(sc.description)}</small></div>'
                f"{_timeline_svg(r)}"
                f'<div class="metric">{extra} <small>(total {r.total:.1f}s)</small></div>'
                "</div>"
            )
        cards.append(
            '<div class="card">'
            f"<h3>{_esc(spec.name)}</h3>"
            f'<div class="hw">{_esc(spec.accelerator)} &middot; {spec.cores} cores '
            f"&middot; {spec.ram_gb:g} GB RAM &middot; fast {_esc(spec.fast_model)} "
            f"/ main {_esc(spec.main_model)} &middot; estimated fast path "
            f"@ {_esc(spec.fast_tokens_per_sec)} tok/s</div>"
            + "".join(rows)
            + "</div>"
        )
    return "<h2>Per-device turn timelines</h2>" + "".join(cards)


_STYLE = """
body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#1b1b1b;background:#fafafa}
h1{margin:0 0 4px}h2{margin:28px 0 8px;border-bottom:2px solid #eee;padding-bottom:4px}
.sub{color:#666;margin:0 0 8px}.note{color:#666;font-size:12px;margin:2px 0 10px}
table{border-collapse:collapse;width:100%;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.06)}
th,td{border:1px solid #e3e3e3;padding:6px 9px;text-align:left}
th{background:#f4f4f6;font-weight:600}
.matrix td{text-align:center;color:#fff;font-weight:600}.matrix td.name{color:#1b1b1b;background:#fff;text-align:left}
.chip{color:#fff;border-radius:9px;padding:1px 7px;font-size:11px;margin-left:4px}
.card{background:#fff;border:1px solid #e3e3e3;border-radius:8px;padding:12px 14px;margin:10px 0;box-shadow:0 1px 2px rgba(0,0,0,.06)}
.card h3{margin:0 0 2px}.hw{color:#666;font-size:12px;margin-bottom:8px}
.row{display:flex;align-items:center;gap:10px;margin:4px 0;flex-wrap:wrap}
.rowlabel{width:90px;font-weight:600;font-size:12px}.rowlabel small{display:block;font-weight:400;color:#888}
.metric{font-size:12px;color:#333}.metric small{color:#999}
svg.tl{border:1px solid #eee;border-radius:3px;background:#fcfcfc;overflow:visible}
.legend{margin:10px 0;display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:#444}
.legend .lg{display:flex;align-items:center;gap:4px}
.legend i{width:13px;height:13px;border-radius:2px;display:inline-block}
.legend i.marker{width:2px;height:14px;background:#111;border-radius:0}
"""


def render(specs: tuple[MachineSpec, ...], scenarios: tuple[Scenario, ...] = SCENARIOS) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Speaker — on-device spec simulation</title>"
        f"<style>{_STYLE}</style></head><body>"
        "<h1>On-device capability simulation</h1>"
        f"<p class='sub'>Generated {ts}. Modelled latencies for an "
        "ASR&rarr;LLM&rarr;TTS turn across target machine specs &mdash; "
        "fast/ordinary-path estimates for comparison, not measurements or "
        "main-tier latency claims.</p>"
        + _legend()
        + _model_fit_table(specs)
        + _matrix(specs, scenarios)
        + _spec_cards(specs, scenarios)
        + "</body></html>"
    )
