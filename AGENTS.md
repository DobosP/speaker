# Agent Instructions — speaker

## Project summary
- Purpose: local-first, always-listening voice assistant (ASR → LLM → TTS) with barge-in and a mode-based
  control plane. `core/` (`VoiceRuntime` on sherpa-onnx) is the reference desktop runtime, `mobile/` the
  on-device Flutter app, `remote/` + `web/` the optional host + thin-client path. Audio pipeline changes need
  deterministic headless tests plus manual live validation when hardware behavior matters.
- Runtime: Python `.venv` created by `./install.sh` (`PYTHON` env selects the interpreter, install.sh:21; the
  Linux ROG box runs CPython 3.12); Flutter for `mobile/`. Current truth: `STATUS.md`.
- Hard invariants (cited, not restated): local/cloud boundary `docs/target_architecture.md` §9.7 + ADR-0097 —
  raw audio never leaves the device except an explicit machine-local trusted-LAN grant; only post-ASR text,
  screen captures and files given to the assistant may reach the opt-in cloud thinking tier. Open-speaker
  barge-in with NO headphones is an owner decision (ADR-0008, ADR-0013). Never hard-set `aec_ref_delay_ms`
  to 260 ms — calibrate with `tools/echo_probe.py` or `aec_auto_delay` (ADR-0005).
- Shape: one portable core + thin per-platform shells sharing the `always_on_agent` `AgentEvent`/`Mode`
  contract (ADR-0001, session topology ADR-0097; the `main.py` monolith is deleted, ADR-0002). Read
  `docs/target_architecture.md` §9 before structural changes.

## Fleet context
- Role: standalone personal R&D — local-first voice assistant (off the company books). Canonical role/status/next: vault note `dobo-brain/paul-brain/projects/speaker.md`
  (fleet view: vault `projects/index.md` + `NOW.md`; agent-ops ADR-0032). Upstream: none · Downstream: none.
- Fleet map + parallel-agent protocol: `~/work/AGENTS.md` (agent-ops ADR-0025/0026). Global session, git,
  scratch and secrets rules: `~/.claude/CLAUDE.md` (agent-ops ADR-0027/0028/0037/0063/0074/0077) — cited here, not restated.
- Delegation: roles and rungs per the `agent-routing` skill (the ladder in `fleet-tiers.sh`); Codex is opt-in (agent-ops ADR-0065).

## Parallel work (mandatory)
- This shared checkout stays on `main`, clean — clean includes untracked (agent-ops ADR-0063): `git status --porcelain`
  is empty when you finish; a stray file blocks the next task-worktree/Ctrl-N session here. A stray file you
  did not write gets reported, not deleted.
- One task = one branch (`<type>/<slug>`) = one worktree `~/work/_worktrees/speaker/<slug>`, never under `/tmp`:
  `python3 ~/work/agent-ops/scripts/create_task_worktree.py --repo ~/work/speaker --branch <type>/<slug> --task "..." --write`
  Scratch and one-off scripts: `~/work/_temp/<slug>/`, run against this repo by path (agent-ops ADR-0028).
- Workers never push. The orchestrating session lands green work on `main` (agent-ops ADR-0014) and finishes the landing
  in the same session (agent-ops ADR-0037): delete the verified-merged branch (local + origin), its worktree, `_temp/<slug>/`.
  Unmerged work is deleted only per item, human-confirmed. A branch reaches origin only on land or by
  `ops publish speaker <branch>` — `ops sync` never creates a remote ref (agent-ops ADR-0077).

## Read first
1. `STATUS.md` — current truth.
2. `docs/agent-map.md`, then `docs/agent-testing.md`.
3. `docs/unified_architecture.md` for architecture; `docs/target_architecture.md` §9 before structural changes.
4. Task-specific engine/test files.

## Commands
| Purpose | Command | Expect |
|---|---|---|
| Session briefing (advisory, stdlib, <1 s) | `python -m tools.session_bootstrap` | advisory one-page briefing; its `.agents/status.json`/`docs/session_*.md` inputs are frozen since 2026-07 — `STATUS.md` wins |
| Logic gate (CI, Tier 0) | `~/work/speaker/.venv/bin/python -m pytest tests -q` | green (`docs/testing.md`); CI runs the same with `--ignore=tests/test_livekit_audio.py --ignore=tests/test_livekit_engine.py` (`.github/workflows/tests.yml:32-34`) |
| Staged runner | `python tools/run_tests.py list` / `unit` / `real_model` / `live` | stages per `docs/testing.md` |
| APM/DTD regression | `~/work/speaker/.venv/bin/python -m pytest tests/test_apm_double_talk.py -q` | `6 passed` |
| Whitespace | `git diff --check` | no output |
| Console run, no audio/models | `python -m core --session console --llm echo` | brain exercised by typing |
| Linux physical session (sole physical entry, ADR-0075) | `./live.sh` | doctor `READY` gate then mic; private bundle under `logs/live/` |
| Readiness | `python -m tools.doctor` (full `READY` only when the route is prepared; `--defer-ollama` is base-only) | each failing line includes its fix |

Use `~/work/speaker/.venv/bin/python` so the command works from any worktree.

## Safety
- Never read or print secret values; names live in `CREDENTIALS.md` (registry agent-ops ADR-0027).
- Do not delete logs unless Paul explicitly asks. Committed run bundles must be PII-free per §9.7: no raw voice
  WAVs or verbatim-PII transcripts — scrub or omit before `git add` (`docs/debugging.md`).
- Never claim live hardware validation unless it actually ran; headless tests do not prove live behavior.
- Token discipline: do not load audio logs, WAVs, screenshots or run artifacts by default; summarize acoustic
  and test output instead of pasting it.
- One audio behavior change per branch/worktree; worker briefs separate headless verification from required live A/B.
- Code: control-plane logic stays in `always_on_agent/`, typed and testable — never resurrect a monolith
  entrypoint; prefer replay/transcript tests over live devices.

## Docs discipline (mandatory)
- `STATUS.md` is this repo's single source of current truth. On conflict: `STATUS.md` > newest-dated ADR in
  `docs/adr/` > everything else. An undated doc is history, not instructions.
- Definition of done for any change of behavior, architecture, status, or decision — same commit: update
  `STATUS.md` (facts + `Last verified: YYYY-MM-DD`); a decision made or reverted gets `docs/adr/NNNN-<slug>.md`
  (next free number in `docs/adr/`, template `docs/adr/0000-template.md`; flip the superseded ADR's `Status:`).
- ADRs are append-only. No decision language ("we use X", "default is") in READMEs/guides — link the ADR.
- Continuation lives in the tab handoff and the vault task note (agent-ops ADR-0078), never a new dated handoff. Dated
  records open with `Valid until: <event> — then treat as history.`
- Budgets: this file ≤ 80 lines, `CLAUDE.md` ≤ 12 non-blank, `STATUS.md` ≤ 120, `docs/agent-map.md` ≤ 60,
  `docs/agent-testing.md` ≤ 80. History overflows to `WORKLOG.md`. Convention: `agent-ops/docs/29-doc-governance.md`.
