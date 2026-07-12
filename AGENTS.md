# Agent Instructions — speaker

## Project summary
`speaker` is Paul's local voice/audio stack. Audio pipeline changes need deterministic headless tests plus manual live validation when hardware behavior matters.

## Fleet context
- Role: local-first always-on voice assistant (personal R&D; repo stays on the personal account).
- Upstream: none · Downstream: none (standalone; always-on audio never leaves the machine).
- Fleet map + parallel-agent protocol: `~/work/AGENTS.md` (agent-ops ADR-0025).

## Parallel work (mandatory)
- This shared checkout stays on `main`, clean — never switch branches or commit task work here.
- One task = one branch (`<type>/<slug>`) = one worktree under `~/work/_worktrees/speaker/`:
  `python3 ~/work/agent-ops/scripts/create_task_worktree.py --repo ~/work/speaker --branch <type>/<slug> --task "..." --write`
- Never create worktrees under `/tmp` (existing `/tmp/speaker-*` worktrees are legacy — migrate at
  next idle; backups: `refs/backups/2026-07-12/*`). Workers never push; the orchestrating session
  lands green work on `main` (ADR-0014) and backs up unlanded branches to origin. Deletion is
  human-confirmed only.

## Read first
1. `CLAUDE.md` if present.
2. `STATUS.md` for durable status.
3. `docs/agent-map.md` and `docs/agent-testing.md`.
4. Task-specific engine/test files.

## Token discipline
- Do not load large audio logs, WAVs, screenshots, or run artifacts by default.
- Summarize acoustic/test outputs instead of pasting large logs.
- Keep context anchored to the engine/test files touched.

## Safety
- Never read or print secret values.
- Do not delete logs unless Paul explicitly asks.
- Do not claim live hardware validation unless it actually ran.
- Direct merge + push to `main` is allowed once the test gate is green (owner
  decision 2026-07-07, development phase). Never land a red suite.

## Commands
- APM/DTD regression: `/home/dobo/work/speaker/.venv/bin/python -m pytest tests/test_apm_double_talk.py -q`
- Whitespace: `git diff --check`

## Dispatch
- One audio behavior change per branch/worktree.
- Worker briefs must separate headless verification from required live A/B validation.

## Code conventions
- Keep control-plane logic in `always_on_agent/`, typed and testable — never resurrect a monolith entrypoint.
- Prefer replay/transcript tests over tests that require live audio devices.

## Docs discipline (mandatory)

- `STATUS.md` is this repo's single source of current truth. On any doc conflict: STATUS.md > newest-dated ADR in `docs/adr/` > everything else. An undated doc is history, not instructions.
- Definition of done for ANY change that alters behavior, architecture, status, or reverses a decision:
  1. Update `STATUS.md` (facts + `Last verified: YYYY-MM-DD`).
  2. Decision made or reverted → add `docs/adr/NNNN-<slug>.md` (next number; template = docs/adr/0000-template.md) and flip the superseded ADR's `Status:` to `superseded-by ADR-NNNN`. Same commit as the change.
- ADRs are append-only: never edit one after landing — supersede it instead.
- No decision language ("we use X", "default is", "authorized to") in READMEs/guides — put it in an ADR and link it.
- Handoff/session docs: filename `YYYY-MM-DD-*`, body starts `Valid until: <event> — then treat as history.` Never obey an expired handoff.
- Keep this file under ~60 lines; STATUS.md under ~100; deep content in docs/.
