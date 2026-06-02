# Ultracode Working Scope — Secure Real-Time Assistant Overhaul

> ⚠️ **Historical record (point-in-time).** Superseded by [`docs/unified_architecture.md`](unified_architecture.md) and the current [`.agents/backlog.md`](../.agents/backlog.md). Kept for history. (2026-06-02 consolidation.)

> **Single source of truth for the multi-agent overhaul.** Every agent and every
> session working on this effort reads this FIRST and stays within the current
> phase's scope. This is a *coordination layer* over the existing authority docs —
> it does **not** override them: `docs/target_architecture.md` (§9, §9.7, §10),
> `docs/PROJECT_KICKOFF.md`, `docs/archive/code_review_2026-05.md`, and `CLAUDE.md`.
>
> Branch: `claude/ultracode-overhaul`. Status of each item lives in the session
> task list (Task tool) and is mirrored in the **Phase pipeline** below.

## Mission

Evolve the existing `speaker` codebase into a **secure, open-source, real-time
voice assistant with layered memory and smart routing** that runs well across
machine types and uses **external LLM providers for research / hard tasks**.
**Evolve, do not rewrite** — build on the working modules (`core/runtime.py`,
`core/routing.py`, `utils/memory*.py`, `tools/cloudchat.py`, the
`always_on_agent` brain). No monolith, no parallel forks of working code.

## The six goals (definition of done)

1. **Secure & OSS-ready** — no secret leakage; tokens read from env only (never
   hard-coded/echoed/committed); safe `remote/` network surface (auth, CORS,
   injection, SSRF); deps/supply-chain sane; license + headers present.
2. **Layered memory** — distinct tiers (working / episodic / semantic / rolling
   summaries / user profile) with retention, eviction, and privacy controls;
   embeddings + recall; cross-session continuity; portable (Postgres desktop →
   SQLite mobile).
3. **Smart routing** — route each turn to the right tier: instant KWS command
   fast-path → local **fast** LLM → local **main**/multimodal → **external cloud**
   for research/hard tasks; cost/latency/headroom-aware; graceful fallback.
4. **Real-time quality** — low-latency `ASR→LLM→TTS`, barge-in, robust
   concurrency (priority event bus, supervisor, cancellable threaded tasks),
   correct cancellation + error handling.
5. **Cross-platform** — one portable core + thin per-platform shells across
   Linux/Windows/macOS/Android(/iOS); `device_profiles`; the
   `always_on_agent` `AgentEvent`/`Mode` contract shared by every shell; mobile
   Dart convergence onto that contract.
6. **External providers** — clean **multi-provider** abstraction for the cloud
   "thinking tier"; key handling; streaming + **hard cancellation** (cost control).

## Non-negotiable constraints (apply in every phase)

- **§9.7 data boundary** — raw audio + PII **never leave the device**. Only
  post-ASR text / screenshots / files explicitly given to the assistant may cross
  to cloud, and only when invoked. The always-on capture loop stays fully local.
- **Secrets** — env-only; never hard-code, echo, or commit a token (see
  `CREDENTIALS.md` golden rule). Reference only as `$VAR`.
- **Don't discard working code** — prefer evolving existing modules; justify any
  removal against `docs/target_architecture.md`.
- **Keep CI green** — `python -m pytest tests` (audio/model-dep tests excluded)
  must stay passing. Add tests for new behavior.
- **Git** — all work on `claude/ultracode-overhaul`; commits local; push **feature
  branches only, never `main`** (enforced by `.claude/hooks/guard.ps1`).

## Roadmap (authoritative detail in `docs/archive/review_ultracode.md`)

Review phase is **done** — 33 findings, **0 refuted** under adversarial verification.
Implementation phases, each its own verified workflow with human sign-off between:

| Phase | Objective | Status |
|------|-----------|--------|
| Review | Read-only map + dimensional audit + adversarial verify → `docs/archive/review_ultracode.md` | ✅ done |
| **P0** | Security & OSS-readiness hardening (auth `/token`, kill secret leaks/dev-key defaults, `LICENSE`+`NOTICE`, harden PII gate) | **this cycle** |
| **P1** | Real-time correctness — deterministic barge-in + cancellation | **this cycle** |
| P2 | Layered-memory seam: wire the tested Postgres engine behind a Memory Protocol; plumb recall into the prompt | pending |
| P3 | Real web research (SearXNG) + cost-controlled cloud streaming (OpenRouter) | pending |
| P4 | Routing headroom-awareness + config deep-merge + composition hygiene | pending |
| P5 | Cross-platform: Dart brain onto the shared `AgentEvent`/`Mode` contract | pending |

## Locked decisions (2026-05-29)

1. **Edit scope this cycle:** **P0 + P1** (security/OSS + real-time correctness). Memory (P2) deferred to its own measured cycle (largest design surface).
2. **Cloud provider:** add **OpenRouter** behind the existing `OpenAICompatLLM` (one key, many models); **US-hosted chains are the default**, PRC-hosted presets (DeepSeek/Moonshot) gated behind explicit config opt-in. *(P3)*
3. **Web search:** **self-hosted SearXNG** behind the pluggable `web.search` CapabilityProvider; every query routed through `classify_sensitivity`. *(P3)*
4. **Memory backend:** **SQLite + sqlite-vec on mobile, Postgres + pgvector on desktop**, both behind the one Memory Protocol; desktop defaults to in-RAM unless `DATABASE_URL` is set. *(P2)*
5. **`setup_database.py`:** reduce to a thin role-create/verify wrapper that defers to `tools/migrate apply`. *(P0/P2)*
6. **Speaker-ID:** keep fail-open default + add P1 output-activity barge-in suppression + a first-run enroll nudge. *(P1)*

## Working agreement for agents

- Read **this file** and the cited authority docs before changing anything.
- **Cite `file:line`** for every claim and every edit. Stay within the current
  phase's scope; surface out-of-scope findings as notes, don't act on them.
- Use **structured outputs** when asked; **verify before asserting**; never print
  secret values.
- Update the **task list** status as items complete (in_progress → completed).
- One concern per change; keep diffs reviewable; run the relevant tests.
