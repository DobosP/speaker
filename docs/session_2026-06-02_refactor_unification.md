# Session 2026-06-02 — repo unification: docs consolidation + session-bootstrap helper + lean-code pass

Cross-machine handoff (ran on the **Windows 11** box; continuation may be elsewhere).
Written in-repo because per-user Claude memory does NOT travel between machines.
Goal of this session (user's words): one unified architecture doc, CLAUDE.md
helpers that use previous-session info to set the working strategy, and a unified
lean codebase that drops duplicate/dead code **without losing functionality**.

## Branch / commit map

Work was done on **`feat/aec-dtln`** (the branch that already descends from the
router + AEC + DTLN merges). Intended landing: merge `feat/aec-dtln → main`, push
`main`, delete the merged branches.

| Branch | State at session start | Action |
|--------|------------------------|--------|
| `feat/aec-dtln` (this) | ahead of `main` by 1 docs commit | session work committed here, then merged to `main` |
| `feat/ondevice-aec` | fully merged into `main` | deletable |
| `feat/unified-capability-router` | fully merged into `main` | deletable |
| local `main` | 6 commits ahead of `origin/main` (unpushed router+AEC+DTLN) | push to sync |

## What landed

**Repo hygiene**
- Relocated the accidentally-nested **`social_media_activities_app/`** (a separate
  Django project with its own `.git`, already pushed to its own remote) OUT to the
  sibling `../social_media_activities_app`. Removed the `UsersPaul` junk file
  (a stray GitHub-API 404 body). Added `.gitignore` guards for both.

**Lean-code pass** (the audit found the codebase already lean — suspected "dupes"
are intentional layering, see unified doc §2/§4/§8):
- Removed the ONLY genuinely dead module: `always_on_agent/snapshots.py` (0 importers).
- Renamed the private `core/agent.py` `AgentEvent` → **`AgentBrainEvent`** to stop it
  colliding with the PUBLIC `always_on_agent/events.py::AgentEvent` platform contract.
- Cross-ref docstring in `core/routing.py` ↔ `core/capability_router.py` (composition,
  not duplication). Fixed `always_on_agent/README.md` (dropped the deleted `adapters.py`).
- **No** experimental tiers cut — DTLN-aec + Smart Turn v3 kept gated/default-off
  (user decision; zero cost when off). Full gate inventory in unified doc §9.

**Docs consolidation**
- New **`docs/unified_architecture.md`** — single current-truth, §0–§13, authored by a
  fan-out workflow (one agent per section reading real sources). Absorbs ~14 dated docs.
- 20 stale docs banner-marked "superseded → unified_architecture.md" (kept in place,
  not moved, to preserve code/test/doc references). The two poles + CLAUDE.md link to it.

**Session-bootstrap helper (the "use previous-session info" ask)**
- `tools/session_bootstrap.py` (stdlib, <1s) → reads `.agents/status.json`, newest
  `docs/session_*.md`, 3 newest `logs/runs/*.summary.json`, open P0 backlog → prints a
  one-page briefing + recommended working strategy. Test: `tests/test_session_bootstrap.py`.
- New CLAUDE.md **"Session bootstrap (run first every session)"** section documents it.

Full suite at session end: **1205 passed, 10 skipped** (`.venv\Scripts\python.exe -m pytest tests`).

## Environment note (won't transfer)
- The git-bash `python` here is **system pyenv 3.10.11** and LACKS onnxruntime/sherpa_onnx
  → bare `python -m pytest tests` mass-errors at collection. Use **`.venv\Scripts\python.exe`**
  for the full suite. (Caught + documented this session; status.json records it.)
- `core/runlog.py` prunes to `SPEAKER_KEEP_RUNS` (default 20) on each run finalize; a test
  hitting the real `logs/runs/` can prune committed bundles. Set `SPEAKER_KEEP_RUNS=9999`
  before a full run, or `git restore logs/runs/` after. (Latent test-isolation bug — candidate fix.)

## Next steps (pick up here)
1. **Finish landing** if not already: `.venv\Scripts\python.exe -m pytest tests -q` green →
   merge `feat/aec-dtln → main`, `git push origin main`, delete `feat/ondevice-aec` +
   `feat/unified-capability-router` (+ `feat/aec-dtln` after merge).
2. **Resume P1 voice/audio** (`.agents/backlog.md`): enable + validate AEC on real hardware
   (mic), extend `tools/echo_probe.py` to print ERLE, validate Smart Turn v3 before enabling.
3. **Optional cleanup:** fix the `core/runlog.py` test-isolation pruning; consider physically
   moving the banner'd `docs/*_2026-05.md` into `docs/archive/` once the ~10 code/test comment
   pointers to them are updated.
4. **Cross-platform P1:** mobile convergence onto the `AgentEvent` contract; SQLite memory
   backend for mobile (unified doc §6/§10/§12).
