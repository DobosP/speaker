# Session 2026-06-14 (pt2) — Persistent SqliteVecMemory backend (gap-analysis P2 start)

**Headline:** Implemented the **persistent SQLite memory backend** (gap-analysis
Decision D6) — the third `Memory` backend behind the one protocol, alongside
in-RAM `SessionMemory` and the Postgres `MemoryManagerAdapter`. It gives
cross-session memory continuity on a desktop **without Postgres** (today that
path is RAM-only, lost on restart) and is the reference for the Dart/mobile
SQLite tier. Branch `feat/sqlite-memory-backend` → `main`. Full suite **1660
passed, 13 skipped** (baseline 1640). Built via a 6-agent P2-state scout + a
15-agent adversarial review (8 findings, all fixed).

## Branch / commit map

- `4e352c3` feat: SqliteVecMemory backend + `candidate_for_item` extraction + wiring + tests
- `974ba78` review fixes (8 adversarial-review findings)
- `<merge>` handoff + status

## What landed

- **`always_on_agent/sqlite_memory.py`** (new): stdlib-`sqlite3`-only, persistent.
  Same six-verb protocol (`add/search/all/context_for_llm/prune/close`). Recall
  goes through the SAME shared `candidate_for_item` → `build_block` as
  `SessionMemory`, so it is **byte-identical for identical stored data at any
  size** (the recall pool tracks `max_items`; `SessionMemory` evicts beyond
  `max_items`, so both consider the same window). Keyword overlap by default; an
  optional injected `embedder` stores float BLOBs and ranks by a **pure-Python
  cosine** over the recent pool (native `sqlite-vec` ANN is a future accel —
  neither `sqlite_vec` nor an embedder is installed). TTL `prune()` +
  working-window `all()`; thread-safe (one connection + lock).
- **`always_on_agent/memory.py`**: extracted `candidate_for_item()` as the single
  source of truth for keyword candidate construction so RAM and SQLite are
  byte-identical *by construction* (`SessionMemory._candidates` now calls it —
  behavior-preserving; existing recall tests green). **No** MemoryManager
  "factor-policy-out" refactor (deferred lm-9/aq-4 follow-up).
- **`core/app.py` `_build_memory`**: new `backend == "sqlite"` branch (`sqlite_path`,
  default `~/.speaker/memory.db`), degrades to in-RAM on any error. Default backend
  unchanged (`auto`), so this is **opt-in**. `config.json` comment documents it.
- **Privacy (§9.7):** the persisted store holds private post-ASR text (and, with
  visual memory on, screen OCR/caption rows) — `memory.db` is chmod'd `0o600` and
  its dir `0o700` so a co-tenant on a multi-user host can't read it off disk.
- **Tests:** `tests/test_sqlite_memory.py` (stdlib-only, CI-safe) — conformance,
  tag fidelity, **persistence across reopen**, byte-identical parity with
  `SessionMemory` (incl. **beyond the working window**), pure-Python cosine
  **isolated from keyword overlap** (synonym-axis embedder), embedder-failure
  graceful-degrade (add-time + query-time), idempotent double-close, concurrent
  add/recall thread-safety, TTL prune. `test_memory_contract.py` `_backends()` is
  now a **factory list** (fresh instance per test) covering all three backends.

## Review findings fixed (8)

`_RECALL_POOL=64` parity break (recall scanned 64 rows vs `max_items` 200) →
pool tracks `max_items`; world-readable `memory.db` → `0o600`/`0o700`;
non-idempotent `close()` → `_closed` guard; tautological cosine test → synonym
axis + control; plus added embedder-exception / >window-parity / concurrency
coverage.

## Environment on i9-13980HX (Linux)

- `.venv` Python 3.12.11; full suite ~75s. No new runtime deps (stdlib sqlite3).
- The SQLite backend is opt-in (`memory.backend = "sqlite"`); the shipped default
  is still `auto` (Postgres when `$DATABASE_URL`, else in-RAM).

## Next steps (rest of gap-analysis P2 — all scouted this session)

1. **lm-2 continuity:** seed `_summary_head` from the newest `summaries` row;
   cross-session fallback for `_load_recent_messages`; one-shot "Last session"
   block at startup (Wire 3 needs an injection-point decision: inside
   `get_context_for_llm`'s budget vs a separate runtime-prepended string).
2. **Recall quality:** add timestamps to recalled lines (`recall._render_line`);
   **decouple profile injection from `recall_enabled`** so durable profile facts
   inject even with episodic recall off — MUST keep the lm-3 sensitivity float on
   the profile-only path.
3. **Dead-knob hygiene (lm-4/5/6/8):** DELETE `memory_persist_assistant` +
   `meeting_persist` (unconsumed); add a "every memory config key is consumed"
   regression test. (`recall_min_similarity`/`recall_max_items` already removed.)
4. **Retention/deletion/provenance:** daily `apply_retention` schedule on the
   writer thread; a "forget that" intent (delete primitives exist); migration 005
   `tags`/`channel` + make ambient/`ingested` RAM-only + recall down-ranks ambient
   (lm-6); record Decision D8 in `target_architecture.md` (gap-analysis already
   recommends the default: TTLs now, at-rest encryption deferred to mobile).
5. **lm-7 window parity** (forward `working_window` to the adapter ring;
   ambient-vs-conversation caps) + **lm-5 episodic** (persist assistant finals via
   the `memory_persist_assistant` knob OR an `episodes` table — and/or fork,
   default-OFF, lm-3 precondition).
6. **Deferred from this slice:** the lm-9/aq-4 "factor-policy-out" refactor
   (extract MemoryManager's summary/profile/TTL/worthiness into shared pure
   functions so SQLite and Postgres share one policy module).

Then P3 (routing quality axis + cost accounting), P4 (installer scipy/soxr,
BargeInDetector extraction, mobile convergence), P5 (remote `/token` per-principal
auth + docs sweep). Owner still owes: rotate the leaked Gemini key, D1 history
purge, speaker-ID enroll; flip `recall_enabled` after `tools.bench`.
