# Memory Layers Enhancement Plan — Hermes-inspired, on-device, off the hot path

> **Decision record (2026-07-02):** the approve/defer decision is pinned in
> [`docs/adr/0009`](adr/0009-memory-layers-approved-deferred.md) — this plan is
> the design reference, not an active work order.
>
> Design output of the `memory-layers-hermes-design` workflow (2026-06-21): 2 web
> research + 2 code-mapping agents → 3 competing designs → 3-lens judge panel →
> synthesis. Status: **PLAN APPROVED, IMPLEMENTATION DEFERRED** (owner chose "stop
> at the plan", 2026-06-21). Every item ships default-OFF, off the real-time hot
> path, on-device (§9.7).
>
> **OWNER DECISION (2026-06-21):** when implementation starts, ship the
> **deterministic (no-LLM) baseline FIRST** — the frequency+regex consolidation
> (2a) and keyword-cluster reflection (2b) with NO model — and add the optional
> small-LLM (`gemma3:1b`) extractor/reflector only later, once the baseline is
> validated. This makes Phases 1–2 zero-extra-LLM-cost to start (resolves open
> question #2). The suggested first slice below is unchanged and remains the entry
> point.

## What "Hermes-inspired" means here

The owner's "HERMES architecture" reference resolves (**medium confidence**) to
**Nous Research's Hermes Agent** (2026-02-26) and specifically its **multi-layer
persistent memory system** — *not* the Hermes 3/4 language models, which are
function-calling/reasoning LLMs with no memory design. The load-bearing idea is
**"architecture determines access, not agent judgment"**: each memory tier has a
fixed location and a fixed moment it is read, so what enters the prompt is decided
*structurally*. Hermes' four layers (curated prompt/workspace files, SQLite-FTS5
session archive, procedural skills, pluggable providers with prefetch) and the
community **Memory OS** 6-layer extension (structured facts with trust scoring,
LLM "fabric" cross-session extraction, decay scanner, cosine>0.92 dedup-merge) are
the direct references. We graft these onto the well-established agent-memory canon
the repo already partly implements: **CoALA's** working/episodic/semantic/procedural
taxonomy, **Mem0's** off-hot-path ADD/UPDATE/DELETE/NOOP consolidation,
**Generative-Agents** reflection + recency·importance·relevance scoring,
**Zep/Graphiti's** "invalidate, don't delete" supersede, and **FadeMem/ACT-R**
salience decay. Honest caveat: because "Hermes" is ambiguous in the literature, we
treat it as *"the layered-cognitive-memory pattern family"* and pick patterns on
engineering merit, not brand.

## Current memory layers

| Layer | Store | What's good | The gap |
|---|---|---|---|
| **L0 Working** (RAM window) | `SessionMemory._items` / `MemoryManagerAdapter._ring` / SQLite recent rows (cap 200) | Free eviction, summarize-then-evict on overflow, read every turn, no model call | None material |
| **L1 Episodic** | Postgres `messages` (pgvector/HNSW) · SQLite `items` | Debounced off-thread writes, junk/echo/dup hygiene, per-embedder partial HNSW | RAM/SQLite recall is **keyword-overlap only** (no embedder installed) |
| **L2 Rolling summary** | Postgres `summaries` + `_summary_head` | Off-bus-thread LLM fold, token-trigger | **Postgres-only**; mechanical text-fold, not synthesis |
| **L3 Semantic profile** | Postgres `user_profile` (key→value) | Durable, never-TTL'd, upsert-by-key (already a supersede) | **4 hardcoded regex patterns only**, Postgres-only, default-OFF; no episode→fact distillation |
| **L4 Procedural** | `rule::` rows / `'procedural'` tag / `_procedural` list | Separate trusted channel, never decayed, cap 64, kept out of recall | No supersede when a new rule contradicts an old one |
| **L5 Vision/ambient** | `messages` role=`observation` via `core/visual_memory.py` | Rate-limited, dhash-gated, local-caption-only, PII-redacted, PRIVATE-floated | None — the reference shape for any new producer |
| **Recall selector** | `always_on_agent/recall.py` `build_block` | Backend-neutral, token-budgeted, adaptive-cutoff elbow, byte-identical across backends, ~40 Tier-0 tests | Recency/importance scoring **shipped disabled** (weights 0) |

**Three audited high-impact gaps:** (1) **no consolidation** of recurring episodes
into durable semantic facts; (2) **no reflection/insight layer** (CoALA's clearest
missing piece); (3) **no graceful forgetting** — only a close-time age-TTL *cliff*,
salience/decay disabled. Plus a **cross-backend asymmetry**: profile/summary/
continuity are Postgres-only, so the SQLite/mobile brain is materially weaker, and
there is **no Dart memory module at all** yet.

## Target layer model

Keep the existing CoALA realization; **add two derived layers + one cross-cutting
salience signal**, all in vocabulary the code already has (`Candidate.kind`, the
`tags` channel on `add()`, `_KIND_IMPORTANCE`). Nothing requires a new store.

```
WORKING    L0  RAM ring                          — unchanged
EPISODIC   L1  messages / items                  — unchanged
SUMMARY    L2  summaries / _summary_head          — EXTEND to SQLite
SEMANTIC   L3  {regex user_profile facts}
               ∪ {NEW consolidated 'fact' rows}   — ADD (Mem0 4-op, supersede stale)
PROCEDURAL L4  rules                               — EXTEND with supersede
REFLECTIVE L6  NEW 'insight' rows                  — ADD (Gen-Agents reflection)
VISION     L5  observations                        — unchanged (reference shape)

cross-cutting SALIENCE:    access_count/last_accessed → recall._apply_signals (reuse blend)
cross-cutting FORGETTING:  graceful decay + cosine/SequenceMatcher>0.92 merge → fold into prune()
```

**Store mapping (no new tables required):** `'fact'`/`'insight'` are two new
`Candidate.kind` values + two `_KIND_IMPORTANCE` entries (`fact≈0.95`, `insight≈0.8`)
+ two `elif` branches in `candidate_for_item` / the SQLite cosine path. The rest of
the selector (`build_block`/`pack`/`collapse`/`adaptive_cutoff`) is kind-agnostic and
unchanged — the new layers flow through the token-budgeted, deduped, byte-identical
pipeline for free (~15 lines). On **SQLite/mobile** facts/insights are tagged rows in
the existing `items` table; on **Postgres** insights reuse `messages` with
`role='insight'` (the trick vision uses with `role='observation'`) and facts reuse
the `user_profile` upsert path. Zero new tables for L6; one idempotent additive
migration for salience columns. **Because the new layers are tagged `add()` items
through the shared `recall.py`/`memory.py` seam, they land on Postgres-desktop AND
SQLite-mobile at once — closing the asymmetry as a free side effect.**

## The plan, phased

Every item is **default-OFF behind a config flag** (byte-identical opening turn),
runs **only on `_schedule_background`/`MemoryWriter` or close-time `prune()`** (never
the bus/audio thread), and uses **only the local fast LLM on post-ASR text** (§9.7).
`tools.bench` gates any default flip.

### Phase 1 — Turn on what's already built + the recall-seam extension
- **1a. Add `'fact'`+`'insight'` recall kinds.** `recall.py` `_KIND_IMPORTANCE` +
  `_render_line` labels; `memory.py:candidate_for_item` + `sqlite_memory.py:_cosine_candidates`
  tag→kind branches. *Mem0 structured facts + insight rows as first-class recall.*
  **S.** Inert until a producer writes rows → existing ~40 recall tests still pass.
- **1b. Flip on the dormant recency+importance scoring (bench-gated).** Config-only
  (`recall_recency_weight`/`recall_importance_weight` → small non-zero) **after
  `tools.bench` confirms no TTFT regression.** *Generative-Agents scoring.* **S.**

### Phase 2 — The two missing producers (the real gap)
- **2a. `consolidation.py` — episode→FACT (Mem0 4-op).** New stdlib-first,
  backend-neutral module (like `recall.py`, ports to Dart). `extract_fact_candidates`
  with a deterministic frequency+regex baseline (no model) + optional injected
  `gemma3:1b` extractor; `ConsolidationRunner.run()` pulls the recent window, compares
  vs the top-few similar facts, writes via `Memory.add(text, tags=('fact', key))` →
  both backends. SUPERSEDE marks `superseded_at` (one nullable column); recall adds
  `AND superseded_at IS NULL`. *Mem0 + Zep "invalidate, don't delete".* **L.**
- **2b. `reflection.py` — idle/close INSIGHT synthesis.** A `Reflector` accumulates a
  session importance sum; at threshold or close/idle runs ONE fast-LLM pass emitting a
  few insight lines with evidence pointers; persisted as `role='insight'` / `tags=
  ('insight',)`. Deterministic keyword-cluster fallback for CI/no-LLM. *Generative-Agents
  reflection.* **L.** Batched ~2–3×/day, never per-turn.
- **2c. Procedural supersede.** Reuse 2a's supersede in `procedural.py`. **S.**

### Phase 3 — Graceful forgetting + salience
- **3a. Salience signal.** `migrations/005_salience_access.sql` adds `importance`,
  `access_count`, `last_accessed` (idempotent; SQLite mirror). Bump on recall, batched
  on the writer thread. `salience_weight` (default 0) folds `log1p(access_count)` into
  the existing `_apply_signals` blend. *ACT-R/FadeMem.* **S.**
- **3b. Graceful decay + dedup-merge in `prune()`.** Drop low-salience-expired first,
  protect high-salience/importance (age-cliff → slope); merge near-dup fact/insight/
  episodic rows above the existing `dedupe_similarity=0.92`, keep the higher-salience
  one. *Memory OS decay scanner + cosine merge.* **M.** Close-time only; default-OFF.

### Phase 4 — Wire-up, parity, mobile/SQLite tier
- **4a. Lifecycle wiring + flags.** Thread the new flags through `core/app.py:_build_memory`;
  in `core/runtime.py:stop()` call `consolidate()` → `reflect()` → `decay()` (no-op-guarded)
  before `prune()`/`close()`. `config.json` keys, default-OFF. **M.**
- **4b. Bring SQLite to full taxonomy.** Implement `SqliteVecMemory.profile_block()`
  (render `'fact'` rows; today `''`) + `last_session_summary()` (newest `summary`/`insight`
  as a frozen one-shot head). *SQLite-first + Hermes frozen-snapshot.* **M.** Becomes the
  faithful Dart reference.

## What we deliberately DON'T do
- **No graph/temporal-KG store** (Zep/Graphiti ~2.6s p95, A-MEM per-note LLM links) —
  too heavy for a phone/the hot path. Take only the cheap nugget: **supersede, don't
  delete** (`superseded_at`).
- **No cloud memory / cloud STT-LLM-TTS / SaaS vector** — §9.7: raw audio + verbatim
  memory never leave the device. Hermes' "External Providers" layer is out of scope.
- **No MemGPT per-turn LLM self-paging or hot-path prefetch** — every page is an LLM
  round-trip; all paging/consolidation/reflection is off-turn only.
- **No new ANN index for SQLite** — brute-force pure-Python cosine over the small recent
  pool ports cleanly to Dart; native ANN waits for a genuinely large corpus.
- **No rewrite** of the recall selector / write pipeline / visual producer — extend via
  `kind`/`tags`, not replacement.
- **No default behavior change without `tools.bench`.**

## Risks & open questions for the owner
1. **`tools.bench` has no recall-QUALITY corpus** (latency only). Want a small PII-free
   multi-session recall fixture before flipping recency/importance/fact/insight on? (Gate
   for 1b + any default flip.)
2. **Local fast-LLM cost for consolidation/reflection** — **RESOLVED (owner, 2026-06-21):
   ship the deterministic no-LLM baseline first; the optional small-LLM extractor is a
   later opt-in.** (When the LLM path is added, run it idle/close-only on `phone`,
   post-flush allowed on `desktop`.)
3. **Soft-delete (`superseded_at`) vs dev-retention.** OK to **hard-evict** superseded+
   low-salience past TTL on mobile (bounded growth) while desktop keeps longer?
4. **Reflection PII surface.** Insights are LLM-synthesized free text about the owner —
   §9.7 PRIVATE-floated + cleaned, but *new* derived content. Confirm excluded from any
   committed `logs/runs` bundle, same as transcripts.
5. **Dart port is aspirational.** Phases 1–4 make `SqliteVecMemory` the complete *Python*
   reference; the actual `mobile/lib` Dart memory is a separate larger workstream.
6. **Hermes identification is medium confidence.** If "HERMES" meant something else, the
   framing shifts — the adopted patterns remain correct regardless.

## Suggested first slice (the one PR to start with)
**"Memory: add `fact`/`insight` recall kinds + reflection scaffold (default-OFF,
headless-testable)"** — Phase **1a + the `reflection.py` pure functions of 2b**, NO
runtime wiring yet. Entirely Tier-0, no DB, no model, no audio.
- `recall.py`: `fact=0.95`/`insight=0.8` + `Fact:`/`Insight:` labels.
- `memory.py:candidate_for_item` + `sqlite_memory.py:_cosine_candidates`: tag→kind.
- `reflection.py`: pure `reflect(episodes, prior_insights) -> list[str]` (keyword-cluster
  fallback + optional injected LLM; no producer wiring).
- Tests: extend `test_recall_selector.py` (new-kind ranking + salience blend shape), new
  `test_reflection.py` (fake LLM → deterministic insights + importance trigger), a
  **cross-backend byte-identity** test for `fact`/`insight` rendering.
- `config.json`: flags default-OFF (`_comment` discipline). `MEMORY.md`: target model.

Why this slice: lands the load-bearing **recall-seam extension** (everything rides it),
is fully **headless/CI-green with zero hot-path / DB / model dependency**, keeps the
opening turn **byte-identical**, and de-risks the architecture before any producer writes
a row.

**Key files:** `always_on_agent/recall.py` (`_KIND_IMPORTANCE`, `_apply_signals`,
`build_block`, weights default-0), `always_on_agent/memory.py` (`Memory` protocol,
`candidate_for_item`), `always_on_agent/sqlite_memory.py` (`profile_block`/
`last_session_summary` return `''`, `prune`), `utils/memory.py` (`_schedule_background`,
`_check_and_summarize`, `apply_retention`, `add_observation`, `_PROFILE_PATTERNS`,
`update_user_profile`, `search_memory`), `core/visual_memory.py`, `core/app.py:_build_memory`,
`core/runtime.py:stop()`.
