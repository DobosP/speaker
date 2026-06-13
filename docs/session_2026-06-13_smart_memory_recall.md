# Session 2026-06-13 — Smart, efficient memory recall (stop context blow-up)

**Headline:** Reworked the long-term **memory recall** layer so injected memory is
bounded, non-redundant, and query-compressed instead of blunt-truncated — the
owner ask was *"enhance my memory layer, use postgres but make it smart and
efficient to not blow up the context."*

**Branch:** `feat/smart-memory-recall-budget` → merged to `main`.
**Commits:** `1f0a43c` (feat) + `c23f5cc` (adversarial-review fixes).
**Verdict:** `.venv/bin/python -m pytest tests` → **1528 passed / 13 skipped / 0 failed**.
(Use `.venv/bin/python` — the system anaconda python lacks `psycopg`, which
`importorskip`-skips the memory-contract + producer suites. The green gate needs them.)

## What changed (branch → main map)

| File | Change |
|---|---|
| `always_on_agent/recall.py` **(NEW)** | The shared, **stdlib-only, zero-utils-dependency** recall selector. `Candidate`, `RecallBudget`, `estimate_tokens`, `adaptive_cutoff`, `collapse`, `pack`, `compress`, `build_block`, `trim_block_to_tokens`. Imports nothing from `utils` → one-way dep, no cycle, SQLite/Dart-mobile portable. |
| `always_on_agent/memory.py` | `SessionMemory` (RAM) → `_candidates()` + `build_block`; deleted `_RECALL_MIN_OVERLAP`/`_RECALL_MAX_ITEMS`/`[:150]`; canonical `User:`/`Assistant:` labels; current-utterance self-exclusion; `budget` kwarg. `MemoryManagerAdapter` forwards `recall_budget`. |
| `utils/memory.py` | `get_context_for_llm` → `build_block` (PG cosine rows → `Candidate`, summary `span` for subsumption); **profile = separate budgeted sub-pass sharing ONE `max_tokens`** with recall; `_to_epoch`, `_finite`, `_recall_pool`, `recall_budget` kwarg. **`_estimate_tokens` (summarize trigger) deliberately untouched.** |
| `core/capabilities.py` | `RecallConfig`: `max_tokens` (+ deprecated `max_chars` alias → `//4`); injection site uses whole-line `trim_block_to_tokens` instead of `[:max_chars]`. |
| `core/conversation.py` | `RecentContextConfig.reserve_tokens` (default **0 = off**, additive token cap on the recent block). |
| `core/app.py` | `_build_recall_budget` wires the budget into **both** backends; honors the legacy `recall_max_chars//4` fallback so the injection cap and memory budget never diverge. |
| `config.json` | Added `recall_max_tokens` (150), `chars_per_token`, `recall_cutoff_k`, `recall_dedup_ratio`, `recall_recent_reserve_tokens`, `recent_context_per_turn_chars`; **removed** dead `recall_min_similarity`/`recall_max_items`; kept `recall_max_chars` as a documented legacy alias. |
| `tests/test_recall_selector.py` **(NEW)** | ~40 Tier-0 unit/property tests (no DB/models), incl. a 2000-case token-bound property over tiny budgets + long words. |
| `tests/test_memory_contract.py` | Rewrote the char-cap test → token-budget bound; added RAM/PG parity + multi-item determinism + canonical-label tests. |

## Design (what "smart & efficient" means here)

- **One token budget** replaces three stacked char caps (per-item 150 ×2 + total 600).
- **Data-derived adaptive cutoff** replaces fixed `cosine>0.6` / `top-3` / `overlap≥2`:
  a **scale-free dominance rule** — keep down to the largest gap only when it is
  ≥ 2× the next-largest gap. FP-robust (even gradients never cut) and fires at
  n≥3 (a mean+std z-score is mathematically inert at exactly 3 candidates — caught
  in review). Honors the owner's **no-fixed-magic-numbers / device-adaptive** directive.
- **Redundancy collapse**: a summary subsumes the messages inside its `start/end`
  span (no schema change); near-duplicates collapse at the write-side ratio (0.92).
  Profile rows are exact-dedup only (never fuzzy-merged).
- **Whole-sentence, query-aware, label-aware compression** for an overflow item —
  never a mid-word cut; reserves the render label cost so the *rendered line* fits.
- **RAM ↔ Postgres parity**: both gather candidates their own way and hand them to
  the same `build_block` → byte-identical block for identical candidates.
- **Profile** is a separate budgeted sub-pass (flat profile scores never ranked
  against cosine in one list) **sharing one `max_tokens`** with recall, with a
  reserved floor so episodic recall can't evict durable identity facts.

## How it was built (2 ultracode workflows)

1. **Design** (16 agents): map → 3 independent designs → 6 judges → synthesis →
   completeness critic. Picked the "token-budget-first adaptive recall" base.
2. **Adversarial review** (28 agents, 5 dimensions, per-finding verification):
   20 confirmed findings. **All real ones fixed**, incl. the headline correctness
   bug (the token bound *could* be exceeded via label-undercount + an `or not
   selected` escape), the FP-fragile/n=3-dead cutoff, the combined-block + cap-
   divergence overflow, and the default ceiling (220→150).

## Next steps (pick up here)

1. **Measure, then enable.** Run `python -m tools.bench` to confirm the injected-
   token delta is favorable, then flip `memory.recall_enabled: true` (one-line).
   Recall is **default-OFF** today → no live behavior change yet.
2. **Optional follow-up** `migrations/005_salience_access.sql` (additive,
   idempotent): `importance`/`access_count` + `pg_trgm`/`tsvector` GIN indexes and
   a hybrid lexical leg behind an opt-in `recall_hybrid` flag (default off).
   Targets the documented proper-noun/STT-garble → confabulation cascade. Core
   win ships without it.
3. **ReAct/escalated path**: recall is injected only on `assistant.answer` today.
   Consider injecting the recall block on the escalated/ReAct planner path too.
4. **Per-device budget**: `recall_max_tokens` is config + `device_profiles`-
   mergeable; the `phone`/`phone_lite` profiles may want a tighter value (no
   `memory` override block exists yet — mind the shallow-merge semantics).
