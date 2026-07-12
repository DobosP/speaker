# Smart Postgres memory (speaker)

Long-term memory stores **only substantive user speech**, not assistant TTS echoes, STT noise, or control phrases. Writes are **debounced** and optionally **cleaned with a small local LLM** (Ollama).

## What gets saved

| Saved | Skipped |
|-------|---------|
| Final user STT after routing to LLM | Assistant replies (default) |
| Cleaned, deduped utterances | Partials (`user_partial`) |
| | Junk markers (`[blank_audio]`, etc.) |
| | Stop/quit/filler unless configured |
| | Near-duplicate lines in a session |
| | Text similar to last assistant output (echo) |

Short-term **in-session** history still includes assistant turns for the current LLM context; Postgres + vector search use **user** rows only.

## Scheduling

- Buffer fills on each final user transcript.
- Auto-flush after the writer's bounded debounce interval; a full buffer flushes early.
- Flush on shutdown (`MemoryManager.close()`).
- On the Postgres/Ollama path, cleanup uses the already-constructed fast client
  (including its host/options/lifecycle) in JSON mode. Non-Ollama paths keep the
  deterministic filters without making an undeclared Ollama call
  ([ADR-0057](docs/adr/0057-reuse-fast-client-for-memory-ingest.md)).

## Schema (`messages`)

| Column | Purpose |
|--------|---------|
| `content` | Cleaned text used for embeddings / recall |
| `raw_text` | Original STT |
| `cleaned_text` | Same as `content` after cleanup |
| `source` | `user_final` (partials not persisted) |
| `confidence` | STT confidence when available (default 1.0) |
| `saved_at` | Persist timestamp |
| `embedding` | pgvector `vector` (unconstrained dim) |
| `embedding_dim` | Integer dim; CHECK enforces match with `vector_dims(embedding)` |
| `embedder_id` | e.g. `all-MiniLM-L6-v2`; partial HNSW indexes filter on this |

**Index policy (post-May-2026 hardening, PR-2):**

- One **HNSW partial index** per `(embedder_id, embedding_dim)` pair --
  e.g. `idx_messages_emb_minilm_l6 WHERE embedder_id = 'all-MiniLM-L6-v2'`.
  HNSW (not IVFFlat) because HNSW has no "training" step that requires
  a populated table -- it builds correctly on a fresh empty DB, which was
  the audited IVFFlat bug.
- Partial-per-embedder so swapping models (e.g. adding `bge-small-en` for
  768-d) is non-destructive -- old 384-d rows keep their index, new 768-d
  rows get a new partial index, queries never mix dimensions.
- Cosine ops (`vector_cosine_ops`); per-query `SET LOCAL hnsw.ef_search = N`
  is tunable (defaults to 40, pgvector default).

**Connection management:** A thread-safe `psycopg_pool.ConnectionPool`
(psycopg3, `min_size=2`, `max_size=5`) replaces the previous single shared
`psycopg2.connection`. Every DB call site uses
`with self._pool.connection() as conn: with conn.cursor() as cur: ...`
so the background writer thread + the request thread + the summary thread
get their own short-lived connections. This is what `mem0` does for its
PGVector backend (issue #3332 documents the same bug we had and the same fix).

## Schema migrations (`migrations/`, run via `tools/migrate.py`)

Yoyo-migrations manages schema state. Files: `001_init.sql` (tables +
indexes), `002_constraints.sql` (CHECK constraints), `003_backfill_dim_embedder.sql`
(legacy-DB upgrade hook -- backfills 384-d rows with `embedder_id`),
`004_hnsw_indexes.sql` (drops old IVFFlat, adds partial HNSW per embedder).
Each has a matching `*.rollback.sql`.

Usage:

```sh
# Default DATABASE_URL = postgresql:///voice_assistant
python tools/migrate.py status              # show applied / pending
python tools/migrate.py apply               # apply all pending
python tools/migrate.py apply --dry-run     # preview without writing
python tools/migrate.py rollback --count 1  # rollback the most recent
```

For the demo / dev path (`python utils/memory.py`), `MemoryManager`
ensures the same schema idempotently on first connect -- production
deploys should still use the migration tool so applied versions are
tracked in `_yoyo_migration`.

## Configuration

The live `memory` block contains backend, recall/context, persistence, and
retention controls. Writer-internal cleanup/buffer knobs are intentionally not a
second live config surface; see ADR-0057.

| Key | Shipped value | Meaning |
|-----|---------------|---------|
| `memory.backend` | `auto` | in-memory unless `DATABASE_URL` selects Postgres; `sqlite` is explicit |
| `memory.recall_enabled` | `false` | inject bounded semantic recall |
| `memory.recent_context_enabled` | `true` | inject bounded same-session turns |
| `memory.embeddings` | `false` | enable pgvector embeddings on Postgres |
| `memory.profile_enabled` | `false` | enable durable profile extraction on Postgres |
| `memory.cross_session_continuity` | `false` | seed Postgres context from prior sessions |
| `memory.persist_assistant` | `false` | persist assistant finals as episodic rows |
| `memory.episodic_ttl_days` | `90` | summarize then evict old messages |
| `memory.summary_ttl_days` | `365` | remove old summaries |

Environment:

- `DATABASE_URL` — Postgres DSN

## Example `config.json`

```json
{
  "memory": {
    "backend": "auto",
    "recall_enabled": false,
    "recent_context_enabled": true,
    "embeddings": false,
    "profile_enabled": false,
    "cross_session_continuity": false,
    "persist_assistant": false,
    "episodic_ttl_days": 90,
    "summary_ttl_days": 365
  }
}
```

## Run

```bash
# Postgres (see SETUP.md / env.example)
export DATABASE_URL=postgresql:///voice_assistant

# Fast path: no embeddings, debounced user-only saves
python -m core
```

There are no `--no-memory`/`--db-url` flags (the legacy `main.py` was deleted
2026-05-26 — `docs/adr/0002`). Memory is toggled via the `memory` block in
`config.json` and the `DATABASE_URL` env var: with `DATABASE_URL` unset, the
runtime uses the in-RAM `SessionMemory` (nothing persists).

## Tests

```bash
pytest tests/test_memory_writer.py -q
```

No live Postgres or Ollama required for unit tests.
