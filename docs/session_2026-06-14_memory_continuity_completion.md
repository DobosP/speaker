# Session 2026-06-14 (pt5) — Memory continuity-completion slice

**Headline:** Landed the **continuity-completion** slice of the gap-analysis P2
memory roadmap — four default-OFF, additive items that finish the cross-session
memory story and clean up dead config. Branch
`feat/memory-continuity-completion` → `main`. Full logic suite **1681 passed,
9 skipped** (real-model tier excluded, load-flaky per prior notes; baseline was
1677). Built via a 5-agent P2-state scout + a 15-agent adversarial review (no
blockers/majors; all confirmed minor/nit findings addressed).

## Branch / commit map

- `520930b` feat: lm-5 + lm-2 Wire 3 + Recall-B + lm-8 prune (+ review fixes folded in)
- `<this>` docs(session) handoff + status
- `<merge>` → main

## What landed (all default-OFF, opening-turn byte-identical)

### lm-5 — persist + recall assistant finals (Postgres tier)
RAM (`SessionMemory`) and SQLite already store + recall `assistant_output`
items; only the Postgres tier was dark. New default-OFF knob
`memory.persist_assistant` folds `'assistant'` into `MemoryManager.persist_roles`.
- `utils/memory.py` `add_message`: when enabled, the assistant final is persisted
  **off the bus thread** (`_schedule_background`) via
  `_save_message_to_db(source='assistant_final')` (embedding computed off-thread
  too). `_last_assistant_text` still set as before.
- `search_memory`: role filter relaxed from `role = 'user'` to a **parameterized**
  `role = ANY(%s)` admitting `'assistant'` only when the knob is on (disabled ⇒
  no write **and** no read ⇒ recall byte-identical).
- Plumbed `core/app._build_memory` → `MemoryManagerAdapter` →
  `create_memory_manager` → `MemoryManager`. **No episodes table** (would fork the
  schema + break parity); reuses the `messages` table with `role='assistant'`.

### lm-2 Wire 3 — one-shot "Last session" recap
Wire 1 (`_seed_summary_head`) + Wire 2 (cross-session `_load_recent_messages`)
landed in pt4. Wire 3:
- `_seed_summary_head` now snapshots a **separate** `_last_session_head` (immune
  to later rolling-summary writes at `_create_summary`/`apply_retention`).
- `last_session_summary()` on `MemoryManager` + the `Memory` protocol exposes it
  (`''` for RAM/SQLite / continuity off / no prior summary).
- `core/capabilities.py`: a **process-start latch** (`memory_state` in
  `attach_llm_capabilities`) injects `=== Last Session ===\n<head>` on the **first
  one-shot answer turn** only, ahead of recall, **compressed** to the recall token
  budget, and **sensitivity-floated** (§9.7). The latch burns **only once the
  recap is actually built**, so a transient turn-1 error stays retryable and an
  empty head never burns it (keeps default-OFF byte-identical).
- Injection point chosen: a **separate runtime-prepended string** (not inside
  `get_context_for_llm`'s budget), so the budgeted recall block stays
  byte-identical and the recap can be one-shot. Postgres-only by construction.

### Recall-B — decouple profile injection from `recall_enabled`
Previously `core/capabilities.py:358` skipped the **entire** memory context (incl.
durable profile facts) when `recall_enabled` was off.
- New `MemoryManager.get_profile_context()` renders the profile block alone
  (`''` unless `profile_enabled` + facts); `profile_block()` verb added to the
  `Memory` protocol (RAM/SQLite return `''`).
- One-shot path: recall ON → `context_for_llm(query)` (already includes the
  profile sub-pass); recall OFF → `profile_block()` (profile only). **No
  double-inject.** The lm-3 sensitivity float now runs over the **combined**
  (last-session + recall/profile) block, and the `VISION_LABEL`→PRIVATE guard
  rides the combined block too.

### lm-8 — prune dead writer-config + consumption test
Deleted 10 dead `memory.*` writer-config keys from `config.json`
(`save_interval_sec`, `min_confidence`, `llm_cleanup`, `llm_gate`,
`cleanup_model`, `max_buffer_items`, `min_chars`, `dedupe_similarity`,
`persist_user_only`, `save_control_phrases`) — **verified zero live readers**
(`_build_memory` never builds a `MemoryWriterConfig`; `from_mapping` has no
callers, so the writer always runs on defaults). Owner chose **prune** over wire
(activating LLM-based memory cleanup is an unrequested behavior change).
- New `test_every_memory_config_key_is_consumed`: allow-list **+ real-reader
  check** (each present key must appear as a quoted literal in `core/`), so a key
  that loses its reader also fails. Extended the dead-knob regression guard to
  cover the pruned keys.

## Review findings addressed (15-agent adversarial pass; no blockers/majors)
- **§9.7 coverage gaps** (minor, code was correct): added capability-level tests
  pinning (a) `VISION_LABEL`→PRIVATE through the combined block, (b) the
  recall-ON combined float over a **private last-session head**, (c) the
  escalation path neither injecting the recap/profile nor burning the latch.
- **Latch ordering** (minor): moved the latch flip into `if head:` so a transient
  turn-1 failure is retryable (+ regression test).
- **Consumption test strength** (nit): added the real-reader assertion.
- **Background write after `close()`** (nit): `_save_message_to_db` now early-
  returns when `self._pool is None` (also hardens the pre-existing summary/profile
  jobs).
- Refuted/non-issues: `runtime_checkable` already pins the new verbs on all
  backends; the non-atomic latch check-then-set is a cosmetic race already
  prevented by newest-input-wins; get_profile_context budget matches the live trim.

## Tests
- `tests/test_memory_continuity_completion.py` (capability-level, no DB) — Wire 3
  latch/empty/sensitivity/compress, Recall-B inject/empty/no-double/sensitivity,
  combined §9.7 floats, escalation exclusion, latch retry-on-raise.
- `tests/test_memory_lm5_pg.py` (Postgres tier, fake-pool, self-skips w/o psycopg)
  — persist_roles folding, assistant-final INSERT (`assistant_final`), `role =
  ANY` admit/exclude, `last_session_summary` snapshot immunity, `get_profile_context`.
- `tests/test_memory_app_wiring.py` — `persist_assistant` forwarding +
  pruned-keys guard + consumption test.

## Environment on i9-13980HX (Linux)
- `.venv` Python 3.12.11; full logic suite ~55s. No new runtime deps.
- PG-tier tests use the fake-pool harness (psycopg installed). **Real-pgvector
  end-to-end verification of lm-5 is recommended before flipping
  `persist_assistant` on** (the new SQL is `role = ANY` + an assistant INSERT;
  standard, but prior continuity work verified against Docker pgvector). It's
  gated behind the default-OFF opt-in regardless.

## Next steps (rest of gap-analysis P2 — still scouted)
1. **Recall quality — timestamps on recalled lines:** parity-safe design only
   (3 backends stamp different wall-clocks for the same logical add). Add a
   default-OFF `show_age` flag on `RecallBudget` + thread a render-time `now`
   through `build_block`→`render`→`_render_line`; render **coarse relative-age
   buckets** ONLY when the flag is on; keep ts unrendered by default (all parity
   tests stay green); age tests single-backend (never cross-backend equality).
2. **Retention / deletion / provenance (lm-6 + D8):** `apply_retention` exists but
   runs only at `VoiceRuntime.stop()` close-time — add a **daily timer** on the
   `MemoryWriter` thread; a **`forget that` intent** (`IntentKind.FORGET` +
   delete verb routing to `clear_observations`/`clear_session`/targeted delete);
   **migration 005** `tags`/`channel` columns (persist the adapter's tags) +
   make `ingested`/ambient RAM-only + recall down-ranks ambient (lm-6); record
   **Decision D8** in `target_architecture.md` (TTLs-now, at-rest encryption
   deferred to mobile) + fill `PROJECT_KICKOFF.md`.
3. **lm-9/aq-4 factor-policy-out refactor (large, parity-sensitive):** extract a
   shared `always_on_agent/memory_policy.py` (worthiness/clean, token-estimate,
   summary-trigger, topic-extract, profile-fact regex, ttl-cutoff). NOTE:
   factoring worthiness/clean into the keyword backends **changes what RAM/SQLite
   ingest** (they currently keep junk Postgres drops) — guard with the
   byte-identical parity tests; do it as its own slice, one cluster per commit.

Then P3 (routing quality axis + cost accounting), P4 (installer scipy/soxr,
BargeInDetector extraction, mobile convergence), P5 (remote `/token` per-principal
auth). **Owner still owes:** rotate the leaked Gemini key, D1 history purge,
speaker-ID enroll; flip `recall_enabled` / `cross_session_continuity` /
`persist_assistant` after `tools.bench` + real-PG validation.
