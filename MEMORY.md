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
- Auto-flush after `memory.save_interval_sec` (default **240s**).
- Force flush when the buffer hits `memory.max_buffer_items` (default **32**).
- Flush on shutdown (`MemoryManager.close()`).

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

Add a `memory` block to `config.json` (see below). CLI flags override file defaults:

| Key / flag | Default | Meaning |
|------------|---------|---------|
| `memory.save_interval_sec` / `--memory-flush-interval` | 180–240 | Debounce seconds |
| `memory.min_confidence` | 0.55 | Drop low-confidence STT |
| `memory.llm_cleanup` | true | Fix typos via Ollama JSON |
| `memory.llm_gate` | true | Skip non-substantive lines |
| `memory.cleanup_model` | `gemma3:1b` | Ollama model for cleanup |
| `memory.max_buffer_items` | 32 | Max buffer before forced flush |
| `memory.embeddings` | off | pgvector semantic search |

Environment:

- `DATABASE_URL` — Postgres DSN
- `MEMORY_CLEANUP_MODEL` — override cleanup Ollama model

## Example `config.json`

```json
{
  "memory": {
    "save_interval_sec": 240,
    "min_confidence": 0.55,
    "llm_cleanup": true,
    "llm_gate": true,
    "cleanup_model": "gemma3:1b",
    "max_buffer_items": 32,
    "min_chars": 3,
    "dedupe_similarity": 0.92,
    "persist_user_only": true,
    "save_control_phrases": false
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
