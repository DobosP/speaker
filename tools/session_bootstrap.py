"""Session bootstrap -- reconstruct prior-session state into a one-page briefing.

Run at the START of every session (see CLAUDE.md "Session bootstrap"):

    python -m tools.session_bootstrap

Pure-local, stdlib-only, <1s, no models/audio/network. The user works across
multiple machines and per-user Claude memory does NOT travel between them, so
prior state lives in the repo. This tool reads it, in priority order:

  1. .agents/status.json            -> machine profile + last test verdict
  2. docs/session_*.md (newest)     -> headline, branch, first 3 next-steps
  3. logs/runs/*.summary.json (3)   -> per-run health (stuck_hints / errors / latency)
  4. .agents/backlog.md             -> OPEN P0 items only

...then prints a "Recommended working strategy" block. It is ADVISORY: it sets
direction, it never changes config or git state. Every source is wrapped so a
missing/corrupt file degrades to a "(not found)" line rather than crashing.

At session END, refresh docs/session_<YYYY-MM-DD>_<slug>.md and .agents/status.json
so the NEXT session's bootstrap has fresh state to read.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

SESSION_DOC_GLOB = "docs/session_*.md"
RUN_SUMMARY_GLOB = "logs/runs/*.summary.json"
SLOW_TURN_SEC = 3.0          # final_to_first_token above this flags a run WARN
N_RUNS = 3                   # how many recent runs to summarize


def repo_root() -> Path:
    """Repo root = parent of tools/ (this file lives in tools/)."""
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_status(root: Path) -> Optional[dict]:
    return _load_json(root / ".agents" / "status.json")


def newest_session_doc(root: Path) -> Optional[Path]:
    # The session_<YYYY-MM-DD>_*.md convention makes lexical sort == chronological.
    paths = sorted(root.glob(SESSION_DOC_GLOB))
    return paths[-1] if paths else None


def parse_session_doc(path: Path) -> dict:
    """Pull headline (first H1), branch, and the first 3 'Next steps' items."""
    out = {"name": path.name, "headline": "", "branch": "", "next_steps": []}
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return out
    lines = text.splitlines()
    for ln in lines:
        if ln.startswith("# "):
            out["headline"] = ln[2:].strip()
            break
    # branch: prefer "...branch **`feat/x`**", else first `feat/...`/`main` token.
    m = re.search(r"branch\s*\*\*`([^`]+)`\*\*", text, re.IGNORECASE)
    if not m:
        m = re.search(r"`(feat/[\w./-]+|main)`", text)
    if m:
        out["branch"] = m.group(1)
    collecting = False
    for ln in lines:
        if ln.startswith("#"):
            if collecting:
                break  # next heading ends the Next-steps block
            if re.search(r"next steps", ln, re.IGNORECASE):
                collecting = True
            continue
        if collecting:
            s = ln.strip()
            if s[:1] in ("-", "*") or (s[:1].isdigit() and "." in s[:3]):
                out["next_steps"].append(s)
                if len(out["next_steps"]) >= 3:
                    break
    return out


def recent_run_summaries(root: Path, n: int = N_RUNS) -> list[Path]:
    # run-YYYYMMDD-HHMMSS -> lexical sort == chronological; newest first.
    paths = sorted(root.glob(RUN_SUMMARY_GLOB))
    return list(reversed(paths[-n:]))


def summarize_run(path: Path) -> dict:
    d = _load_json(path) or {}
    run_id = d.get("run_id") or path.name
    counts = d.get("counts") or {}
    errors = d.get("errors") or []
    n_err = counts.get("errors", len(errors))
    stuck = d.get("stuck_hints") or []
    turns = d.get("turns") or []
    n_turns = counts.get("turns", len(turns))
    dur = d.get("duration_sec")
    slow = 0.0
    for t in turns:
        v = t.get("final_to_first_token") if isinstance(t, dict) else None
        if isinstance(v, (int, float)) and v > slow:
            slow = float(v)
    warn = bool(stuck) or (isinstance(n_err, int) and n_err > 0) or slow > SLOW_TURN_SEC
    parts = []
    if isinstance(dur, (int, float)):
        parts.append(f"{dur:.0f}s")
    parts.append(f"{n_turns} turns")
    parts.append(f"{n_err} errors")
    if stuck:
        parts.append(f"stuck:{len(stuck)}")
    if slow > SLOW_TURN_SEC:
        parts.append(f"slow_ttft {slow:.1f}s")
    tag = "WARN" if warn else "OK"
    return {"run_id": run_id, "warn": warn, "line": f"[{tag}] {run_id} ({', '.join(parts)})"}


def open_p0(root: Path) -> list[str]:
    """Open ('- [ ]') items under any '## P0' heading in .agents/backlog.md."""
    try:
        text = (root / ".agents" / "backlog.md").read_text(encoding="utf-8")
    except Exception:
        return []
    items, in_p0 = [], False
    for ln in text.splitlines():
        if ln.startswith("## "):
            in_p0 = "p0" in ln.lower()
            continue
        if in_p0 and ln.strip().startswith("- [ ]"):
            items.append(ln.strip()[5:].strip())
    return items


def _recommend(status, doc_info, run_dicts, p0) -> list[str]:
    recs = []
    warn_runs = [r["run_id"] for r in run_dicts if r["warn"]]
    verdict = (status or {}).get("last_verdict") or {}
    if verdict.get("green") is False:
        recs.append("Tests are RED in the last verdict -- fix the suite before new work.")
    if warn_runs:
        recs.append(
            f"A recent run needs a look ({warn_runs[0]}) -- see docs/debugging.md "
            "(summary.json stuck_hints / errors) before starting new work."
        )
    if doc_info and doc_info.get("next_steps"):
        first = doc_info["next_steps"][0].lstrip("-*0123456789. ").strip()
        recs.append(f"Continue from prior session's next-step: {first}")
    elif p0:
        recs.append(f"No prior next-steps; take the top open P0 item: {p0[0]}")
    else:
        recs.append("Clean slate -- no pending next-steps or open P0 items.")
    recs.append("Confirm the baseline with `python -m pytest tests -q` before changing code.")
    return recs


def build_briefing(root: Path) -> str:
    """Assemble the one-page briefing markdown from all prior-session sources."""
    out: list[str] = ["# Session bootstrap briefing", ""]

    status = read_status(root)
    out.append("## Machine & last verdict")
    if status:
        m = status.get("machine") or {}
        out.append(
            f"- machine: {m.get('cpu', '?')}, {m.get('ram_gib', '?')} GiB, "
            f"{m.get('gpu', '?')} (profile: {m.get('target_profile', '?')})"
        )
        v = status.get("last_verdict") or {}
        flag = "GREEN" if v.get("green") else ("RED" if v.get("green") is False else "?")
        out.append(f"- last tests: [{flag}] {v.get('tests', '(unknown)')}")
        if status.get("next"):
            out.append(f"- status.next: {status['next']}")
    else:
        out.append("- .agents/status.json (not found)")
    out.append("")

    out.append("## Prior session")
    doc = newest_session_doc(root)
    doc_info = parse_session_doc(doc) if doc else None
    if doc_info:
        out.append(f"- {doc_info['name']}")
        if doc_info["headline"]:
            out.append(f"- headline: {doc_info['headline']}")
        if doc_info["branch"]:
            out.append(f"- branch: {doc_info['branch']}")
        if doc_info["next_steps"]:
            out.append("- next steps:")
            out.extend(f"  {s}" for s in doc_info["next_steps"])
    else:
        out.append("- docs/session_*.md (not found)")
    out.append("")

    out.append("## Recent runs (newest first)")
    run_dicts = [summarize_run(p) for p in recent_run_summaries(root)]
    if run_dicts:
        out.extend(f"- {r['line']}" for r in run_dicts)
    else:
        out.append("- logs/runs/*.summary.json (none found)")
    out.append("")

    out.append("## Open P0 backlog")
    p0 = open_p0(root)
    out.extend(f"- {s}" for s in p0) if p0 else out.append("- (none open)")
    out.append("")

    out.append("## Recommended working strategy")
    out.extend(f"- {r}" for r in _recommend(status, doc_info, run_dicts, p0))
    out.append("")
    return "\n".join(out)


def main(argv: Optional[list[str]] = None) -> int:
    root = repo_root()
    briefing = build_briefing(root)
    # The briefing echoes unicode (arrows, en-dashes) from the source docs; a
    # cp1252 Windows console would otherwise crash on print. Make stdout tolerant.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        print(briefing)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        sys.stdout.write(briefing.encode(enc, "replace").decode(enc, "replace") + "\n")
    try:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = root / "logs" / f"session_{ts}_bootstrap.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(briefing, encoding="utf-8")
        print(f"\n(written to {out.relative_to(root)} -- gitignored)")
    except Exception:
        pass  # writing the copy is best-effort; the printed briefing is the product
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
