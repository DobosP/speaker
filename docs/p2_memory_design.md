# P2 — Layered Memory: Design (gated by adversarial review)

_Branch `claude/ultracode-overhaul`. Authority: `docs/review_ultracode.md` (§P2, layered-memory-1/2/3), `docs/ultracode_scope.md` (Locked Decisions 4 & 5). Evolve, do not rewrite._

> **Status: design APPROVED with binding corrections.** Two adversarial critics
> returned `approve=false` on the first design; the corrections below are part of
> the spec and MUST be implemented. They fix one functional dead-end (recall would
> retrieve nothing) and several regressions.

## Binding revisions from adversarial review (implement these)

- **R1 (CRITICAL) — ingest answered queries.** The normal answered turn never writes the user's question to memory, so recall has nothing to retrieve. Add an explicit `memory.add(query, tags=("user",))` in the `assistant` closure (`core/capabilities.py`) *before* the recall prepend, so each answered question enters the index. The contract test must assert the **adapter** backend recalls a turn-1 fact at turn N (not just SessionMemory).
- **R2 (HIGH) — summary off the bus thread.** `_check_and_summarize` runs synchronously inside `add_message`, which the supervisor calls on the single bus thread (`_handle_task_completed`). An LLM summarizer there would stall TTS/barge-in (a P1 regression). Schedule the rolling summary on the existing `MemoryWriter` background thread / enqueue a job — never inline. Test: `add_message`/`queue_user_utterance` returns without invoking the summarizer on the caller's thread.
- **R3 (HIGH) — adapter `all()` tag fidelity.** `MemoryManager.recent_messages` keeps only `role` and junk-filters, so the adapter would drop the `('ingested',)` tag `test_addressing` asserts. The adapter keeps its **own small in-RAM ring buffer** of the raw `(text, tags)` handed to `add()` and returns those from `all()`, so both backends behave identically. Contract test asserts tag preservation on BOTH backends.
- **R4 (HIGH) — `setup_database.py` psycopg2→psycopg3.** `psycopg2` was removed from `requirements.txt` (psycopg3 is in). Port `create_database()`/`verify_setup()` to `import psycopg`, or drop verify to `python -m tools.migrate status`. Import-smoke must pass with only declared deps. Wrap the `tools.migrate.main([...])` call in try/except with a friendly "install yoyo-migrations" message.
- **R5 (MEDIUM) — no `research_synth` recall.** Inject recall ONLY in the `assistant` closure this cycle (research recall is unmeasured scope-creep; revisit in P3).
- **R6 (MEDIUM) — wire `close()`/`prune()`.** Add a guarded `memory.close()` in `VoiceRuntime.stop()`; run `prune()` at close-time this cycle (no vague "periodic hook").
- **R7 (MEDIUM) — `meeting.note` RAM-only by default.** Route `("meeting", …)` to a non-persisted add unless `memory.meeting_persist` is set (§9.7 privacy).
- **R8 (MEDIUM) — profile producer.** Default-off, confidence floor, **Postgres-only**; SessionMemory has no profile section. Contract test asserts profile recall only on the adapter-with-DB (marked, excluded from the logic suite).
- **R9 — defensive search tags.** `tags = tuple(t for t in (d.get("type"), d.get("role")) if t)` (summary rows have no `role`).
- **R10 (LOW) — flatten the config block.** Single-level keys (`memory.recall_enabled`, `memory.recall_min_similarity`, `memory.recall_max_items`, `memory.recall_max_chars`, `memory.profile_enabled`, `memory.meeting_persist`, retention TTLs) so a future device-profile override survives the shallow merge (deep-merge is P4).
- **R11 (LOW) — SessionMemory relevance parity.** `SessionMemory.context_for_llm` needs a minimum keyword-overlap threshold + item cap so its injection volume ≈ the Postgres top-3; the bench must measure the **in-RAM default** TTFT too, not only Postgres.
- **R12 (LOW) — redact connect failures.** `_build_memory` must not echo a psycopg connect-error string (may contain the DSN); log a `_redact_db_url`'d message.

## Goal

Make the running assistant remember. Today `core/runtime.py:88` hardcodes a volatile in-RAM `SessionMemory`, recall is never read into the answer prompt (`utils/memory.py get_context_for_llm` has no caller), and the tested Postgres/pgvector `MemoryManager` is fully built but unwired. P2 introduces **one** `Memory` Protocol seam in `always_on_agent`, wires the existing `MemoryManager` behind it (reuse wholesale), plumbs gated recall into the answer prompt, and adds tiers/retention + a rolling summary + a profile producer.

## 1. The one Memory Protocol (`always_on_agent/memory.py`)

Backend-neutral verbs only — no `db_url`, embeddings, `session_id`, pool, `role/content` dicts, pgvector, or SQL.

```python
@runtime_checkable
class Memory(Protocol):
    def add(self, text: str, tags: tuple[str, ...] = ()) -> None: ...        # ingest (tag = neutral channel)
    def search(self, query: str, limit: int = 5) -> Sequence[MemoryItem]: ... # recall -> neutral items
    def all(self) -> Sequence[MemoryItem]: ...                               # recent window (tag-faithful)
    def context_for_llm(self, query: str) -> str: ...                        # ready-to-prepend block or ''
    def prune(self) -> int: ...                                              # retention/eviction
    def close(self) -> None: ...                                             # flush + release
```

`add/search/all` are the brain's existing surface (`runtime.py:183,187,197`, `supervisor.py:332`, `capabilities.py:54,126`, `tests/test_addressing.py:147,150,172`). `context_for_llm/prune/close` are additive. Type the three injection seams (`supervisor.py:42`, `capabilities.py:42`, new `VoiceRuntime` param) against `Memory`.

## 2. SessionMemory = trivial default + working-window cap

Already has `add/search/all`. Add: `context_for_llm` (keyword recall with a **min-overlap threshold + cap** per R11; `''` on no hit = relevance gate, zero embedding cost), `prune()->0`, `close()->None`, and a `max_items` (default 200) working-window truncation in `add()`. Default desktop store when `DATABASE_URL` is unset.

## 3. Postgres adapter (thin, reuse — `MemoryManagerAdapter`)

Lazily imports `utils.memory` so `always_on_agent` stays Postgres-free. Maps:

| Protocol | MemoryManager (file:line) |
|---|---|
| `add` (tag-routed; **+ in-RAM tagged ring buffer for `all()`**, R3) | `queue_user_utterance` (`447`) / `add_message` (`418`) — honors `persist_roles=('user',)`; meeting tags RAM-only (R7) |
| `search` | `search_memory` (`708`), wrap `List[Dict]` -> `MemoryItem` with **defensive tags** (R9) |
| `context_for_llm` | `get_context_for_llm` (`789`) verbatim (similarity>0.6 + profile) |
| `all` | returns the adapter's own ring buffer of raw `(text, tags)` (R3) — NOT `recent_messages` |
| `prune` | new `apply_retention` (close-time, R6) |
| `close` | `close` (`892`) |

Relies on MemoryManager's graceful degradation (`utils/memory.py:50-63,294-333,359-378`); safe to construct without a live DB (`tests/test_memory_pool.py:144`). All Postgres-isms stay inside the adapter.

## 4. Recall injection (the headline fix) — `assistant` closure only (R5)

In `core/capabilities.py`, after `_enrich_context` (`141`), **ingest the query (R1)**, then prepend recall before `model.stream(...)` at `:160`:

```python
if memory is not None:
    memory.add(query, tags=("user",))            # R1: so recall has something to find
system_for_call = system
if memory is not None and recall_cfg.enabled:
    recall = memory.context_for_llm(query)        # '' when irrelevant/unavailable
    if recall:
        system_for_call = recall[: recall_cfg.max_chars] + "\n\n" + system
tokens = mark_first_token(model.stream(query, system=system_for_call), recorder)
```

- `memory` is a new keyword-only param on `attach_llm_capabilities` (`:76`), captured as a closure like `llm`/`router`/`recorder`, forwarded from `core/runtime.py:94-97` (currently missing). `None` default keeps existing test call sites valid.
- **Two-layer gate:** config flag `memory.recall_enabled` (default **false**, short-circuits before embedding) + relevance (the existing `similarity>0.6` self-gate; `''` => no prompt change). Never mutate `query` (keeps router/sensitivity inputs clean).
- **No latency/quality regression:** prepend only enlarges INPUT — `_stream_and_speak`/`_collect` drain OUTPUT, so streaming/barge-in/sentence-splitting are unaffected. Bound TTFT with top-3 + `max_chars`. **Measure with `tools.bench` (both in-RAM AND Postgres backends, R11)** before enabling.

## 5. Summary + profile producers (`utils/memory.py`) — both OFF the hot path

- Replace the keyword `_create_summary` with a **fast-tier LLM rolling summary** via an injected `summarizer` callable (keyword fallback when absent), **scheduled on the `MemoryWriter` background thread — never synchronously inside `add_message`/`_check_and_summarize`** (R2).
- Wire the uncalled `update_user_profile`: a gated, **default-off, confidence-floored** ingest-time extractor (deterministic regex first; optional fast-LLM pass on the writer thread). **Postgres-only** (R8); profile rows surface in recall via `get_context_for_llm`.

## 6. Tiers + retention/privacy

Working (RAM cap), Episodic (`messages`), Semantic (pgvector index), Summary (`summaries`, LLM rolling), Profile (`user_profile`, durable). New `MemoryManager.apply_retention` does age-TTL DELETEs (summarize-then-evict for episodic; long TTL for summaries; none for profile) guarded by `_db_available`, invoked from `prune()` at close-time (R6). Privacy: §9.7 boundary preserved (text only), `persist_user_only`, meeting RAM-only by default (R7), junk filters, `clear_session` for forget. Never log `DATABASE_URL` (R12).

## 7. Config (`config.json` `memory` block — FLAT, R10)

`memory.backend` (`auto|inmemory|postgres`), `memory.recall_enabled:false`, `memory.recall_min_similarity:0.6`, `memory.recall_max_items:3`, `memory.recall_max_chars`, `memory.embeddings:false`, `memory.max_recent:20`, `memory.profile_enabled:false`, `memory.meeting_persist:false`, retention TTL keys. Single-level so a device-profile override survives the shallow merge.

## 8. Backend selection (`core/app.py`, Locked Decision 4)

`_build_memory(config)`: if `backend=='postgres'` or (`auto`/unset and `os.environ.get('DATABASE_URL')`) -> build adapter in try/except (ImportError/connect) falling back to `SessionMemory`, **redacting any connect-error (R12)**; else `SessionMemory`. **Desktop defaults to in-RAM unless `DATABASE_URL` is set.** Pass via the new `VoiceRuntime(memory=...)`.

## 9. SQLite scope this cycle — DEFER (both critics endorsed)

**Defer the Python SQLite+sqlite-vec backend.** No Python consumer needs it now (mobile is the Dart app, P5; the `phone` profile is a desktop sim), migrations are Postgres-only (`tools/migrate.py:17-19`), and it is net-new. The Protocol makes a future `SqliteVecMemory` a same-interface drop-in. Ship Protocol + Postgres adapter + in-RAM default only.

## 10. Migration consolidation (Locked Decision 5)

`python -m tools.migrate apply` is canonical (correct unconstrained-vector + per-row dim + partial HNSW; `004` DROPs old IVFFlat; migrations are idempotent/additive so they reconcile legacy DBs). Reduce `setup_database.py` to `create_database()` + `verify_setup()` **ported to psycopg3 (R4)**; delete the schema SQL constants + `setup_tables`; `main()` shells to `tools.migrate.main(['apply', ...])` (try/except friendly). Update `setup.sh` + `SETUP.md` to the single schema path in lockstep. OS install + role/db create stay in `setup.sh`.

## 11. Ordered steps (file ownership)

1. Protocol + SessionMemory hooks/cap/threshold — `always_on_agent/memory.py`
2. `MemoryManagerAdapter` (+ ring buffer, defensive tags) — `always_on_agent/memory.py` (after 1)
3. Type seams — `always_on_agent/{supervisor,capabilities}.py` (parallel with 4)
4. `attach_llm_capabilities` memory= + **query ingest (R1)** + gated prepend (assistant only) — `core/capabilities.py`
5. `VoiceRuntime(memory=)` + forward + `memory.close()` in stop() (R6) — `core/runtime.py`
6. `_build_memory` + flat config block — `core/app.py`, `config.json` (after 2,5)
7. Rolling summary (off-thread, R2) + profile producer (R8) — `utils/memory.py` (after the summarizer-plumbing decision in 2/6)
8. `apply_retention`/prune — `utils/memory.py` (after 7)
9. setup_database.py thin wrapper (psycopg3, R4)
10. setup.sh + SETUP.md (after 9)
11. Contract + recall regression tests (both backends, tag fidelity, off-thread summary, default-off neutrality) — `tests/test_memory_contract.py` (after 1,2,4,5,6)

## 12. Tests

Contract over BOTH backends (`isinstance(x, Memory)`, **tag-faithful `all()`**); **"fact in turn 1 recalled in turn N" on the adapter (R1)**; default-off latency-neutrality (system byte-identical); empty-recall = no-change; `max_chars` cap; backend selection + no-secret-print (R12); preserved `test_addressing`; `attach_llm_capabilities` back-compat; **`add_message` returns without inline summarizer (R2)**; summary/profile units (marked Postgres); `tools.bench` TTFT gate (both backends); migrate + setup_database import smoke with only declared deps (R4).

## 13. Top regression risks

Prompt-quality + TTFT (default-off, capped, bench both backends); Postgres-ism leak (str/MemoryItem Protocol + lazy import); construct-without-DB (degradation, don't build adapter without `DATABASE_URL`); bus-thread stall from summary (R2 off-thread); tag loss (R3 ring buffer); psycopg2 breakage (R4 port); secret leakage (R12 redact); shallow-merge (R10 flat).
