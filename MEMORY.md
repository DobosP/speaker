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
- Auto-flush after `memory.save_interval_sec` (default **240s**, overridable via `memory_flush_interval_sec` in config/CLI).
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

Run `python setup_database.py` on existing DBs; `MemoryManager` also migrates columns on connect.

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
| `memory_smart_save` / `--memory-no-smart-save` | on | Enable buffered writer |
| `memory_enable_embeddings` | off | pgvector semantic search |
| `memory_persist_assistant` | off | Also save assistant to DB |
| `memory_llm_clean` / `--memory-no-llm-clean` | on | Use main LLM cleaner hook |

Environment:

- `DATABASE_URL` — Postgres DSN
- `MEMORY_CLEANUP_MODEL` — override cleanup Ollama model

## Example `config.json`

```json
{
  "memory_smart_save": true,
  "memory_flush_interval_sec": 240,
  "memory_enable_embeddings": false,
  "memory_llm_clean": true,
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
python main.py

# Disable memory entirely
python main.py --no-memory
```

## Tests

```bash
pytest tests/test_memory_writer.py -q
```

No live Postgres or Ollama required for unit tests.
