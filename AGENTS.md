# Agent Instructions — speaker

## Project summary
`speaker` is Paul's local voice/audio stack. Audio pipeline changes need deterministic headless tests plus manual live validation when hardware behavior matters.

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
- Do not push or merge unless Paul explicitly asks.

## Commands
- APM/DTD regression: `/home/dobo/work/speaker/.venv/bin/python -m pytest tests/test_apm_double_talk.py -q`
- Whitespace: `git diff --check`

## Dispatch
- One audio behavior change per branch/worktree.
- Worker briefs must separate headless verification from required live A/B validation.
