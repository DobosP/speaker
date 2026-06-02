# P3 Design — Real Web Research (SearXNG) + Cost-Controlled Cloud Streaming

> ⚠️ **Superseded — durable content merged into [`docs/unified_architecture.md`](unified_architecture.md).** Kept for revision history; do not treat as current. (2026-06-02 consolidation.)

_Branch `claude/ultracode-overhaul`. Gates the P3 implementation. Authority: `docs/archive/review_ultracode.md` §P3 (smart-routing-1/3/4, security-5), `docs/archive/ultracode_scope.md` Locked Decisions 2 & 3, `docs/target_architecture.md` §9.7. Evolve existing modules; do not rewrite._

> **Status: design APPROVED with binding corrections.** Two adversarial critics
> returned `approve=false` on the first design (two HIGH). The corrections below
> are part of the spec. One item (BR3) is a product/security DECISION for the user.

## Binding revisions from adversarial review

- **BR1 (HIGH) — pre-first-token socket/billing.** The HTTP hard-close is deterministic only **after** tokens flow: a losing cloud worker still blocked in the first-token read never observes the stop event, so it holds the socket + billing until the client timeout (**default 30s, never overridden**). Fix: plumb a short `llm.cloud.timeout_s` (low single digits, e.g. 5) through `_build_cloud_client` **and** the single-cloud back-compat path into `OpenAICompatLLM(timeout=)`, so a pre-first-token loser is reaped fast. Correct the narrative (close is not instant pre-TTFT). Test: a fake stream blocking in `__next__` asserts close within the timeout bound.
- **BR2 (HIGH) — unguarded enum coercion in the gate.** `Mode(context["mode"])` / `IntentKind(context["intent_kind"])` raise on an out-of-vocab string → `registry.invoke` turns it into `ok=False` → aborts the RESEARCH plan (`tasks.py:229-231`), or worse bypasses the gate. Fix: wrap the coercion in try/except (mirror `_enrich_context` `core/capabilities.py:142-149`) defaulting to `None`, OR have `may_leave_device` accept raw strings and convert fail-safe. The gate must still run (fail-safe to PRIVATE) on a bad value. Test: bogus `context["mode"]` ⇒ `ok=True` corpus fallback, still gated.
- **BR3 (DECISION — LOCKED: Option B, "block PII only").** The search-surface egress predicate is `may_leave_device(query) = not _is_personal(query)` (also blocking `Mode.MEETING` and `COMMAND`/`DICTATION`/`MEETING_NOTE` intents). This hard-blocks any PII/personal/possessive query but lets plain non-PII lookups ("weather in Berlin") reach the **self-hosted** SearXNG (egress to user-controlled infra; SEARCH/RESEARCH intent signals the user wants external lookup). Pin the rule with tests: a PII/possessive query ⇒ `False`; a plain public query ⇒ `True`; a CODE query carrying a credential ⇒ `False` (PII precedence, BR5).
- **BR4 (low) — max_tokens target.** Inject the per-turn ceiling into `merged` **before** the profile-cap block (`core/llm.py:507-512`), not into `kwargs`, so `min()` composition keeps the profile cap authoritative.
- **BR5 (low) — CODE-with-secret precedence.** `may_leave_device` allows CODE (CODE != PRIVATE). Add a golden test that a CODE query containing a credential phrase ("debug this, the api key is sk-…") returns `False` (PRIVATE wins via `_is_personal` precedence at `sensitivity.py:182` before `_CODE_MARKERS`). Cross-reference comment between `may_leave_device` and `routing.py` so the two sensitivity consumers don't drift.
- **BR6 (low) — pre-first-token close edge.** Keep the `sdk_stream` binding + `try/finally` INSIDE the generator body so a `close()` before the first token is a no-op (no `AttributeError`). Test `gen.close()` with zero tokens consumed.
- **BR7 (low) — web.search wedge.** A blocking `httpx`/`urllib` GET cannot poll `cancel_event`; only the timeout protects. Use a bounded connect+read `timeout_s` (~4s) framed as the wedge bound. Test: a backend blocking past `timeout_s` returns the corpus fallback (`ok=True`, latency bounded).
- **BR8 (low) — PRC opt-in silent fallback.** When a CN preset is dropped for lack of `allow_prc`, emit an INFO log (distinct from the missing-API-key drop) so a user who forgot `allow_prc` isn't silently degraded to local. Add to step 6.

---

## Goals (3 & 6)
1. Real web search via a pluggable `web.search` CapabilityProvider backed by self-hosted SearXNG, for RESEARCH/SEARCH plans + ReAct `DEFAULT_TOOLS`. Corpus stays as offline fallback.
2. §9.7 boundary: every web-search query routed through the sensitivity gate; only permitted text egresses (predicate per BR3).
3. OpenRouter behind `OpenAICompatLLM`; US chains default, PRC presets explicit opt-in (Decision 2).
4. HTTP-level hard-close of cloud streams on cancel + a short cloud timeout (BR1) + optional per-turn `max_tokens` ceiling.
5. `tools/cloudchat` consumes `OpenAICompatLLM` so quirk-handling + hard-close live in one place.

## 1. web.search provider (`core/websearch.py`, new)
- Lives in `core/` (not `always_on_agent/`, which is core-free per `always_on_agent/react.py:9-12`) so it can call `core.sensitivity` directly.
- `WebSearchConfig.from_dict` (mirror `core/capabilities.py:45-51`), `SearxngBackend` (lazy `httpx` import à la `core/llm.py:482-491`) behind a `Backend` Protocol (pluggable per Decision 3), `attach_web_search_capability(registry, config, *, classify, backend, fallback_capability="search.local")`.
- Provider order: **gate → SearXNG → corpus fallback**. Never raises (a non-ok step aborts the plan, `always_on_agent/tasks.py:229-231`). Result mirrors corpus `search()` shape (`always_on_agent/capabilities.py:75-80`), `citations=` source URLs, `data["egress"]`/`data["sensitivity"]` audit stamps.
- Register at `core/runtime.py:96` (after `create_default_capabilities`, with `attach_llm_capabilities`). Plumb config from `core/app.py:657` → new `VoiceRuntime` kwarg.
- Plans: swap `search.local`→`web.search` in `always_on_agent/planner.py:43,55`. ReAct: add to `DEFAULT_TOOLS` (`always_on_agent/react.py:34`) + `_TOOL_DESCRIPTIONS` (`:36-41`) + `config.json:240-244`.

## 2. Sensitivity egress gate (§9.7)
- New `core.sensitivity.may_leave_device(query, *, mode, intent_kind)` (after `:196`, export at `:199`). Predicate per **BR3**; shares `_is_personal` (`:136-152`) and fail-safe-to-PRIVATE. **Guarded enum coercion (BR2).**
- Called FIRST inside the provider on its own `query` arg (NOT trusting `context['sensitivity']`, which is set only on the assistant path `core/capabilities.py:150-153` and may be absent for gather/ReAct steps). `mode`/`intent_kind` from `always_on_agent/tasks.py:267,274-275`.
- PRIVATE ⇒ SearXNG never called; corpus result with `egress=False`.

## 3. OpenRouter + US-default / PRC opt-in
- Add `openrouter_*` presets (`base_url https://openrouter.ai/api/v1`, `OPENROUTER_API_KEY`, `profile "openai_compat"`, host `US`) to `config.json` `cloud_providers`. Secrets env-only. Ship `openai/gpt-oss-120b` + `meta-llama/llama-3.3-70b-instruct` (US); lead the default `public` chain with gpt-oss-120b.
- `_build_cloud_client` (`core/app.py:122`) gains `allow_prc`; drops `host=="CN"` presets unless set (reuse `:140-141`), INFO-logging the drop (BR8). `_wrap_cloud` reads `llm.cloud.allow_prc` (default false).

## 4. Hard-close + timeout + cost ceiling
- **Gap after P1:** `OpenAICompatLLM.stream` (`core/llm.py:533-561`) never `.close()`s the SDK `Stream`; close at GC. **Fix:** bind `sdk_stream`, iterate in `try`, `sdk_stream.close()` in `finally` (port `tools/cloudchat.py:181-185`); binding inside the generator body (BR6).
- **BR1 timeout:** `llm.cloud.timeout_s` → `OpenAICompatLLM(timeout=)` at both construction sites.
- **max_tokens:** add to `__init__`; inject into `merged` before the profile cap via `min()` (BR4); plumb `llm.cloud.max_tokens` (default `None`).

## 5. cloudchat refactor (smart-routing-4)
Replace `OpenAICloudClient` + `_OpenAIStream` (`tools/cloudchat.py:170-225`) with a `CompatCloudClient` wrapping `OpenAICompatLLM`; keep `CloudClient`/`CloudStream` Protocols so `tests/test_cloudchat.py` fakes still inject. SSE + hard-close now live only in `core/llm.py`.

## 6. Implementation order & parallel islands
1. `core/sensitivity.py` (island) →
2. `core/llm.py` hard-close + timeout + max_tokens (island; read-only for 6/7) →
3. `core/websearch.py` (new; needs 1) →
4. `core/runtime.py` + `core/app.py` register/plumb (needs 3) →
5. `always_on_agent/planner.py` + `react.py` + `config.json` tools (island; needs 3) →
6. `core/app.py` OpenRouter/PRC + `config.json` (needs 2) →
7. `tools/cloudchat.py` (island; reads 2) →
8. docs + sanity tests.

## 7. Tests (must add)
- PRIVATE query never reaches a tripwire backend (corpus, `egress=False`).
- PUBLIC/eligible query hits a fake SearXNG; citations = urls; corpus-compatible shape.
- SearXNG-unreachable / blocks-past-timeout ⇒ corpus fallback `ok=True` (BR7).
- Bogus `context["mode"]` ⇒ corpus fallback, still gated (BR2).
- Losing/cancelled cloud worker ⇒ SDK stream `.close()` asserted; pre-first-token `close()` is a no-op (BR6); blocked-`__next__` reaped within `timeout_s` (BR1).
- max_tokens `min()` composition with profile cap (BR4); CODE-with-credential ⇒ no egress (BR5).
- PRC opt-in gate drops CN presets unless `allow_prc`.
- Full suite + `python tools/run_tests.py imports`.

## 8. Decisions (recommendations)
- **Egress predicate (BR3): USER decision** — recommend `not _is_personal(query)` for the search surface (PII-safe, feature-functional, self-hosted egress).
- SearXNG: keyless `base_url` in a new `web_search` config block, `enabled=false` default.
- max_tokens: default `None`, set ~512 in cloud-enabled voice profiles.
- SearXNG unreachable: silent corpus fallback with `data["source"]` stamp.
- OpenRouter models: `openai/gpt-oss-120b` + `meta-llama/llama-3.3-70b-instruct`, US.
