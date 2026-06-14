# Session 2026-06-14 — Gap-analysis P0/P1 hardening (code-actionable slice)

**Headline:** Landed the **code-actionable** subset of the 2026-06-10 gap-analysis
roadmap (`docs/review_2026-06-10_gap_analysis.md`) — every P0/P1 item that needs
no owner key, no git-history rewrite, and no live mic/screen. Branch
`feat/gap-analysis-p0-p1-hardening` → `main`. Full logic suite **1640 passed,
13 skipped** (baseline was 1614). Built/verified with three ultracode workflows:
an 11-agent scout (verified each item's real state), a 17-agent adversarial
review (6 dimensions → verify each finding; 9 confirmed, all fixed), and a focused
rc-4 concurrency re-verifier.

## Branch / commit map

- `0ab1ca3` security + lm-1 (P0 code parts)
- `f24b455` rc-3 / rc-4 / rc-5 / rc-6/aq-7 + sr-2 (P1 real-time correctness)
- `417650e` review fixes (9 adversarial-review findings)
- `<merge>` follow-up: `int|None` annotation fix in `_start_task`

## What landed

**P0 security / docs / migration:**
- `.github/workflows/gitleaks.yml` + `.gitleaks.toml` — blocking secret-scan gate,
  **incremental** (scans the PR/push diff, not full history): the known historical
  `.env` leak awaits the owner's rotate + `filter-repo`; a full-history audit is the
  separate pre-release step. Allowlists the public Android `debug.keystore`.
- `SECURITY.md` — disclosure channel, §9.7 boundary + env-only golden rule as stated
  invariants, accurate token-server note (deny-by-default bearer auth exists; the
  single shared token doesn't scope identity/room).
- `tools/migrate.py` — redact the DSN in `status` (was printed raw). Self-contained
  redaction (does **not** import `setup_database`, which `sys.exit(1)`s at import
  without psycopg — a `SystemExit`/`BaseException` that would have RED-ed CI and
  crashed `migrate status`); also masks `?password=` query-param form.
- `remote/token_server.py` — `/token` and `/chat` 500s no longer return `str(exc)`
  (could leak `LIVEKIT_API_SECRET` / backend host); log server-side, return generic
  detail. **Also fixed a latent bug:** under PEP 563 the locally-imported `Request`
  annotation was unresolvable → every `/chat` call 422'd; hoisted `Request` into
  module globals.
- §9.7 doc reconcile — committed run bundles must be PII-free (`docs/debugging.md`
  callout, CLAUDE.md run-logs bullet, `.gitignore` whitelist comment).
- `migrations/002_constraints.sql` — invalid `ADD CONSTRAINT IF NOT EXISTS` →
  DO-block `duplicate_object` (broke `migrate apply`). `migrations/004_hnsw_indexes.sql`
  — HNSW on the unconstrained `vector` column failed (`column does not have
  dimensions`); cast the partial-index expr to `vector(384)`.
  `.github/workflows/migrations.yml` — pgvector service runs `migrate apply` +
  idempotency re-apply. **Verified end-to-end (apply/re-apply/rollback) against
  `pgvector/pgvector:pg16` in docker.**

**P1 real-time correctness** (rc-1/rc-2/lm-3 were already done):
- **rc-5** — `HANDLED_LOCAL` metric on the no-LLM intent fast-path; watchdog skips
  those turns (kills the false `llm stuck`/`tts stuck` hint). (The INGEST/KWS/merged
  paths were already watchdog-safe.)
- **rc-6/aq-7** — bounded the high-churn `SupervisorState` histories with
  `deque(maxlen)`; `transcript_log` stays a list (sliced by the drivers); STT_PARTIAL
  excluded from `event_log`.
- **rc-4** — `cancel_all` cancels queued tasks; `_start_queued_tasks` snapshots+swaps
  under `_cancel_lock` and **starts inline** (so capacity counts update — fixes the
  over-admission the review caught) with a `start_epoch` re-check (drops a pass a
  barge-in superseded mid-drain — closes the residual resurrection window);
  `_start_task` drops pre-cancelled / epoch-stale tasks.
- **sr-2** — failure apology spoken on `TASK_FAILED` (payload now carries
  speak/followup/epoch like TASK_COMPLETED); cross-tier fast↔main retry in
  `capabilities.assistant()` (only a distinct other tier, never after audio emitted —
  no double-speak). Addresses the "mute on tier failure" UX.
- **rc-3** — per-utterance **generation counter** in `core/engines/sherpa.py`. Each
  queued sentence carries the generation at enqueue; a barge/stop bumps it. The worker
  `_claim_utterance()` helper skips a stale dequeued sentence **without clearing**
  `_stop_speaking` (the wipe race), and in-flight synthesis aborts on a generation
  mismatch. A new reply after a barge carries the new generation → never muted.
  (The scout's proposed "clear-only-if-not-stale" had a latent mute-bug; the
  generation-on-enqueue design avoids it.)

## Environment on i9-13980HX (Linux, desktop_gpu_4090)

- `.venv` Python 3.12.11; full suite `~71s`.
- Docker available — used `pgvector/pgvector:pg16` to verify the migration chain.
- Working tree still carries the pre-existing `logs/runs/` churn (deleted + untracked
  bundles, incl. a `.wav`); **deliberately not committed** (PII per §9.7).

## Next steps (pick up here)

1. **OWNER, still open (cannot be done by the agent):** rotate the leaked Gemini key
   (public history since `d32db9f`); decide D1 history purge; speaker-ID enrollment
   (`python -m core --enroll`). The gitleaks gate is now incremental, so it won't be
   held red by the historical leak — but the full-history purge is still owed.
2. **Watch the new CI gates green on the PR/push:** `Secret scan` (gitleaks) and
   `Migrations` (pgvector). Both verified locally; confirm on GitHub.
3. **gap-analysis P2+ (next phases):** P2 layered-memory continuity (cross-session
   summary head, SqliteVecMemory, dead knobs, lm-2/4/5/6/7/9) — gated on the lm-3
   §9.7 fix which is already in; P3 routing quality axis + cost accounting; P4
   installer (scipy/soxr), profile validation, BargeInDetector extraction, mobile
   convergence; P5 remote `/token` per-principal auth + docs sweep.
4. **Live (needs mic):** the rc-3/rc-4/sr-2 changes are deterministically tested but
   touch the real-time path — re-validate barge-in + a tier-failure fallback on a
   live `--engine sherpa` run.
5. **Carried:** flip `recall_enabled` after `tools.bench` shows a favorable token
   delta; the 2026-06-10 live follow-ups in `.agents/backlog.md`.
