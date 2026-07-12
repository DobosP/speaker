# CLAUDE.md

Guidance for Claude Code working in this repo. Read this first every session.

Docs discipline: see AGENTS.md — STATUS.md + ADR update is part of definition of done.
Fleet context + parallel-agent protocol (ADR-0025): top stanzas of `AGENTS.md`; map: `~/work/AGENTS.md`.

## What this project is

A local-first, real-time voice assistant (`ASR → LLM → TTS`) with barge-in and a
mode-based control plane, targeting always-listening use across
Linux/Windows/macOS/Android/iOS. The desktop Python runtime `core/`
(`VoiceRuntime` on sherpa-onnx) is the reference; `mobile/` is the on-device
Flutter app; `remote/` + `web/` is the host + thin-client path (LiveKit/WebRTC).
Shape (decided): **one portable core + thin per-platform shells** sharing the
`always_on_agent` `AgentEvent`/`Mode` contract (`docs/adr/0001`); the legacy
`main.py` monolith is deleted (`docs/adr/0002`). Read
`docs/target_architecture.md` §9 before structural changes.

**Local/cloud boundary (`docs/target_architecture.md` §9.7 — hard invariant):**
STT, TTS, VAD, speaker-ID, the always-on capture loop, and the fast/answering
LLM tier stay **on-device — raw audio never leaves the device**. The thinking
tier (main planner / research / multimodal summarize / web search) may use
cloud; only post-ASR text + screen captures + files given to the assistant may
cross over, and only when invoked. This boundary supersedes the earlier blanket
"no cloud STT/LLM/TTS by default" stance; the always-on loop is still fully local.

> **HARD REQUIREMENT — open-speaker barge-in, NO headphones (owner decision
> 2026-06-05; D-A, `docs/adr/0008`).** Barge-in MUST work on the bare laptop
> speaker; headphones must never be assumed or suggested as the fix. The current
> implementation, evidence, and remaining live A/B gate are in `STATUS.md` and
> `docs/adr/0013`; the older AdaptiveDTD result is history, not current status.
> **Never hard-set `aec_ref_delay_ms` to 260 ms** — calibrate
> per machine with `tools/echo_probe.py` or use `aec_auto_delay`
> (`docs/adr/0005`).

## Quickstart

```
python -m tools.session_bootstrap            # session start: rebuild context (<1s, no deps)
python -m core --engine console --llm echo   # run without audio/models
python -m core --engine sherpa               # on-device audio
python -m pytest tests -q                    # logic suite (the CI gate)
python tools/run_tests.py fast               # staged runner, Tier 0
python -m tools.doctor                       # preflight when a run won't start
```

## Hard policies

- **Git (fleet standard 2026-07-07, `docs/adr/0014`; see AGENTS.md).** `main` is
  the integration branch; do feature work on a short-lived branch. When the work
  is complete and the logic suite is green, **direct merge + push to `main` is
  allowed** (owner decision 2026-07-07, partially superseding ADR-0007's PR-only
  landing). Don't delete branches that aren't yours or fully merged.
- **On the Windows box**, the guard hook (`.claude/hooks/guard.ps1`) still blocks
  touching the work git/SSH identity; the main-push block was removed 2026-07-07
  (ADR-0014). `docs/windows_landing_workflow.md` describes the old PR flow for
  when the guard returns at release hardening.
- **Secrets live in [`CREDENTIALS.md`](CREDENTIALS.md)** — single source of
  truth for every credential. **Golden rule:** read from the env at runtime;
  **never** hard-code, echo, or commit a token — reference it only as `$VAR`.
- **Committed run bundles must be PII-free per §9.7** (no raw voice WAVs /
  verbatim-PII transcripts) — scrub or omit before `git add`; see `docs/debugging.md`.

## Docs map

- `STATUS.md` — current truth (top of precedence); `.agents/backlog.md` — live work queue.
- `docs/dev_guide.md` — session bootstrap/end protocol, repo layout, run matrix, tooling (bench/specsim/cloudchat), CI workflows, environment notes.
- `docs/unified_architecture.md` — current-truth architecture overview (start here); `docs/target_architecture.md` — north-star + structural decisions (§9).
- `docs/adr/` — dated decisions (append-only); `docs/archive/` — superseded history (incl. the old as-built `architecture.md`).
- `docs/testing.md` — test tiers + staged runner; `docs/debugging.md` — run logs, telemetry, stuck runs.
- `docs/audio_pipeline.md` — audio chain (AEC/APM/DTD); `docs/deployment_profiles.md` — device profiles + the `llm` config block.
- `docs/windows_landing_workflow.md` — PR landing from Windows; `MEMORY.md` — memory subsystem; `SETUP.md` — install.
- `docs/PROJECT_KICKOFF.md` — running list of product decisions; check it for current intent.

## When unsure

Ask clarifying questions before large changes.
