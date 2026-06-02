# Speaker — Lead Architect Synthesis & Roadmap

> ⚠️ **Superseded — durable content merged into [`docs/unified_architecture.md`](unified_architecture.md).** Kept for revision history; do not treat as current. (2026-06-02 consolidation.)

_Target: a secure, OSS-ready, real-time voice assistant with layered memory and smart routing that runs well across machine types and uses external LLM providers for the hard/thinking tier._

_Date: 2026-05-29. Built on `docs/target_architecture.md` (§9, §9.7, §10) and `docs/code_review_2026-05.md`. This report gates the implementation phase._

## 1. Overall Assessment

This is an unusually well-engineered real-time voice substrate, roughly **60-65%** of the way to the stated target. What is built is mostly built well:

- **Solid:** the audio/STT/TTS engine seam (`core/engines`), the control-plane brain (`always_on_agent`: priority event bus, supervisor, cancellable threaded tasks, deterministic analyzer), the multi-provider cloud abstraction (`core/llm.py` + `core/routing.py` + sensitivity gating), and the observability belt (metrics/runlog/watchdog/recorder). The s9.7 audio boundary (raw audio stays local) holds **by construction**.

The shortfall is concentrated in three areas that block the headline goals:

1. **Layered memory is the largest miss.** The Postgres/pgvector smart-memory stack (`utils/memory.py`) is fully built, audited, and tested — but **completely unwired**. The runtime hardcodes a volatile in-RAM `SessionMemory`, and **recall is never read into the answer prompt on any real turn**. The running assistant cannot remember anything.
2. **Smart routing + external research has no real web egress.** Research mode gathers from a **5-entry hardcoded corpus**; only final synthesis hits a real LLM. Routing is also not runtime cost/latency/headroom-aware (static per-profile config only).
3. **Security/OSS-readiness has concrete, fixable defects:** unauthenticated `/token` on a `0.0.0.0` bind, a DB password echoed to stdout, well-known LiveKit dev keys as live defaults, and a MIT-claimed repo with no `LICENSE`.

Real-time quality is strong but has two correctness bugs that bite where it matters (stale sentence after barge-in; non-cancellable escalated turns). Cross-platform is adequate-by-design but thin: only ~4 pure functions are truly shared with the Dart shell.

**None of this requires discarding working code.** Every fix evolves an existing, well-factored module. The single highest-leverage move is a **Memory Protocol seam** (P2), which simultaneously unblocks Goal 2, wires the existing Postgres work, and makes SQLite-on-mobile a same-interface implementation rather than a rewrite.

## 2. Per-Goal State

| Goal | State | One-line |
|---|---|---|
| 1. Secure & OSS-ready | **weak** | Good runtime token hygiene + s9.7 holds, but unauthenticated remote surface, leaked DB password, no LICENSE, brittle PII gate. |
| 2. Layered memory | **missing** | Only a volatile in-RAM tier exists and it is never recalled; the real Postgres engine is unwired. |
| 3. Smart routing | **adequate** | Tier + chain routing real and tested; not runtime headroom/cost-aware. |
| 4. Real-time quality | **adequate** | Strong concurrency design; two cancellation/barge-in correctness bugs. |
| 5. Cross-platform | **weak** | One core + shells in intent; only ~4 fns shared with Dart, mobile already drifted; shallow config merge. |
| 6. External providers | **adequate** | Clean multi-provider abstraction; no real web research; soft (billing-leaking) cancellation. |

### Goal 1 — Secure & OSS-ready (weak)
First-party runtime token hygiene is genuinely good and the audio boundary holds. The OSS-ready and remote-surface bars are not met.
- **High:** `/token` mints a JWT for any identity/room, no auth, `0.0.0.0` bind (`remote/token_server.py:123,166`).
- **High:** `setup_database.py:350,364,366` echoes the full `DATABASE_URL` incl. password.
- **High:** no `LICENSE` despite README "MIT"; no attribution for Vocalis/Leon/RouteLLM/sherpa.
- **Medium:** well-known LiveKit `devkey/secret` defaults (`docker-compose.yml:47,61`); brittle sensitivity gate routes "my coworker John's salary" to a CN-hosted chain (`core/sensitivity.py:59-68`); unpinned/no-SRI web SDK (`web/index.html:8`); unverified mobile model download (`mobile/lib/model_store.dart:71`).

### Goal 2 — Layered memory (missing)
The running assistant has exactly one volatile in-RAM tier, rebuilt empty each start, and never read into answers.
- **Critical:** Postgres `MemoryManager` unwired; runtime hardcodes `SessionMemory()` with no `memory=` param (`core/runtime.py:88`).
- **Critical:** recall not plumbed into the answer path; `assistant.answer` uses query+system only; `get_context_for_llm` has no caller (`core/capabilities.py:131,216`).
- **High:** summary tier is keyword-frequency placeholder; `update_user_profile` uncalled; no episodic/semantic split; no retention/eviction/TTL; no SQLite mobile backend.

### Goal 3 — Smart routing (adequate)
KWS fast-path + `HeuristicRouter` tier split + sensitivity `ChainSelector` + `HedgeLLM` failover are real and tested.
- **Medium:** not runtime headroom-aware; `sysinfo`/measured TTFT never feed the router or hedge timing (`core/routing.py:96`).
- **Low:** per-provider `_pricing/ttft_ms` metadata is documentation-only.

### Goal 4 — Real-time quality (adequate)
Well-designed bus/supervisor/cancellable-tasks/watchdog; happy-path latency is real.
- **High:** barge-in/STOP does not deterministically preempt in-flight/queued `TTS_REQUEST` — stale sentence after interrupt on the streaming path; no threaded-bus barge-in test (`core/runtime.py:212,269`).
- **High:** ReAct planner + `HedgeLLM.generate` use blocking `generate()`, defeating cancellation during escalated turns (`always_on_agent/react.py:187`, `core/llm.py:606`).
- **Medium:** Hedge final-drain `q.get()` no timeout (stalled winner wedges turn) (`core/llm.py:744`); `_threads` never reaped + only RESEARCH capped (`always_on_agent/tasks.py:93`); fail-open speaker gate + VAD-only barge-in + no AEC = TTS-echo self-interruption (`core/engines/speaker_gate.py:55`).

### Goal 5 — Cross-platform (weak)
One Python core, swappable engines, device_profiles, remote thin-client, golden-pinned contract — but realized mobile sharing is ~4 pure fns.
- **Medium:** no Dart `AgentEvent`/`Mode`/supervisor; mobile stop check diverges (`mobile/lib/assistant.dart:300`).
- **High:** shallow per-section config merge can strand nested keys / silently disable cloud (`core/app.py:57,48`).

### Goal 6 — External providers (adequate)
`OpenAICompatLLM` + `ProviderProfile` + per-chain `HedgeLLM` + `SensitivityRouterLLM`, env-only keys, working streaming + failover.
- **High:** no web/search egress; research + ReAct tools run on a 5-entry corpus (`always_on_agent/capabilities.py:42`, `core/capabilities.py:195`).
- **Medium:** lost/cancelled workers stop only at next token, stream never `.close()`'d, no per-turn token cap (`core/llm.py:615,739`).

## 3. Findings by Severity

| Severity | Count | IDs |
|---|---|---|
| Critical | 2 | layered-memory-1, layered-memory-2 |
| High | 11 | security-1, security-2, security-3, layered-memory-3, layered-memory-4, layered-memory-5, smart-routing-1, realtime-concurrency-1, realtime-concurrency-2, cross-platform-2, architecture-quality-1, architecture-quality-2 |
| Medium | 11 | security-4, security-5, security-6, security-7, layered-memory-6, smart-routing-2, smart-routing-3, realtime-concurrency-3, realtime-concurrency-4, realtime-concurrency-5, cross-platform-1, architecture-quality-3, architecture-quality-4 |
| Low / Uncertain | rest | security-8, smart-routing-4, smart-routing-5, realtime-concurrency-6, architecture-quality-5, architecture-quality-6 |

_(architecture-quality-1/2/3 overlap layered-memory-1 and the security/license findings; counted where most load-bearing.)_

## 4. Prioritized Roadmap

Sequenced so security/correctness land first and later phases build on earlier seams. Phases prefer evolving existing modules.

1. **P0 — Security & OSS-readiness hardening** _(ship-blocker; depends on: none)_
   - Auth on `/token` (env bearer dependency); default `127.0.0.1`, require `SPEAKER_REMOTE_BIND_ALL=1` for `0.0.0.0`; public-bind warning.
   - Stop echoing DB password (`setup_database.py`); add deprecation banner.
   - Remove working secret defaults (`LIVEKIT_*` empty + fail-fast); drop `--dev` from non-dev compose; sentinel `.env.example`.
   - Add `LICENSE` (MIT) + `NOTICE`/`THIRD_PARTY_NOTICES.md` (Vocalis/RouteLLM/vLLM-SR Apache-2.0, Leon MIT, Gemma).
   - Pin+SRI or vendor the LiveKit web SDK; add CSP.
   - `/chat` body-size + rate limit.
   - Harden sensitivity gate: possessive+noun ⇒ PRIVATE; PII golden tests; gate PRC chains behind opt-in.
   - _Files:_ `remote/token_server.py`, `setup_database.py`, `docker-compose.yml`, `docker/.env.example`, `LICENSE`(new), `NOTICE`(new), `web/index.html`, `core/sensitivity.py`, `tests/test_sensitivity.py`. _Risk:_ Low.

2. **P1 — Real-time correctness (cancellation + barge-in)** _(depends on: P0 base)_
   - Set cancel_event before `stop_speaking()` returns; drop `TTS_REQUEST` for non-active task_ids or stamp a speech-epoch; add threaded-bus + streaming-TTS barge-in test.
   - Convert ReAct plan/step `generate()` to cancel-aware `stream()`; thread cancel_event into `registry.invoke`.
   - Bounded idle timeout on Hedge final-drain `q.get()`; socket/read timeouts; join/bound worker threads.
   - Output-activity barge-in suppression + unenrolled-conservative fallback; watchdog-storm debounce.
   - Reap `_threads`; global active-task cap overflowing into queued_tasks.
   - _Files:_ `core/runtime.py`, `always_on_agent/react.py`, `always_on_agent/tasks.py`, `always_on_agent/supervisor.py`, `core/llm.py`, `core/engines/sherpa.py`, `core/watchdog.py`, `tests/test_core_runtime.py`. _Risk:_ Medium (deadlock/over-drop — covered by new threaded-bus test + sandbox harness).

3. **P2 — Layered memory seam + wiring** _(highest leverage; depends on: P1)_
   - Define a `Memory` Protocol in `always_on_agent`: `ingest/recall/context_for_llm` + retention hooks.
   - `SessionMemory` = trivial impl; thin adapter onto `MemoryManager`.
   - Add `memory=` to `VoiceRuntime`; select in `core/app.py` from a `memory` config block (in-RAM default; Postgres when `DATABASE_URL`).
   - Prepend recalled context in `attach_llm_capabilities` before `model.stream()`, gated by relevance.
   - Replace `_create_summary` keyword path with fast-tier LLM rolling summary; add profile-extraction calling `update_user_profile`.
   - SessionMemory cap/window; episodic/semantic split + age-prune + per-tier retention/privacy.
   - Make `tools/migrate apply` the single documented setup path; quarantine `setup_database.py`.
   - Contract test: supervisor loop vs both backends; "fact in turn 1 recalled in turn N".
   - _Files:_ `always_on_agent/memory.py`, `utils/memory.py`, `core/runtime.py`, `core/app.py`, `core/capabilities.py`, `setup.sh`, `SETUP.md`, `tests/test_memory_contract.py`(new). _Risk:_ Medium-high (prompt-quality regression — gate + measure with bench).

4. **P3 — External research path + cost-controlled cloud streaming** _(depends on: P0 gate, P1 cancel; benefits from P2)_
   - Add `web.search` CapabilityProvider (pluggable backend) for RESEARCH/SEARCH + ReAct DEFAULT_TOOLS; route query text through `classify_sensitivity`; keep corpus as offline fallback.
   - Make `OpenAICompatLLM.stream` closable/context-managed; close in `HedgeLLM._worker` finally/on-stop; optional per-turn `max_tokens` ceiling.
   - Refactor `tools/cloudchat` to consume `OpenAICompatLLM` so hard-close lives in one place.
   - _Files:_ `core/capabilities.py`, `always_on_agent/react.py`, `core/llm.py`, `core/sensitivity.py`, `tools/cloudchat.py`, `config.json`. _Risk:_ Medium (new egress must respect s9.7).

5. **P4 — Routing headroom-awareness + composition hygiene** _(depends on: P3, P0)_
   - Feed sysinfo load + TTFT EWMA into `HeuristicRouter.score`; dynamically scale `hedge_delay`.
   - Optional cost/ttft-aware chain ordering; observed-TTFT adaptation.
   - Recursive deep-merge for device_profiles + `config.local.json`; per-profile merge tests.
   - Extract `core/llm_factory.build_llms`; promote `core/config.py` so `remote/worker` stops importing app internals.
   - _Files:_ `core/routing.py`, `core/capabilities.py`, `core/app.py`, `core/llm_factory.py`(new), `core/config.py`(new), `remote/worker.py`, `tests/test_device_profiles.py`. _Risk:_ Low-medium.

6. **P5 — Cross-platform convergence (Dart brain onto contract)** _(depends on: P2, P1)_
   - Mobile uses shared `isStopCommand`.
   - Golden fixtures for `observe/decide` + minimal Mode machine; port thin Dart `AgentEvent`/`Mode` + analyzer behind them.
   - Mobile model checksum/signature verification.
   - Delete/archive `always_on_agent/adapters.py` stub.
   - _Files:_ `mobile/lib/assistant.dart`, `mobile/lib/contract.dart`, `always_on_agent/speech_analyzer.py`, `tests/golden/`, `mobile/lib/model_store.dart`, `always_on_agent/adapters.py`. _Risk:_ Medium (fixtures authored Python-side first as the gate).

## 5. Decisions Needed Before Implementation

1. **Cloud provider(s)** — _Options:_ keep OpenAI-compat presets / add OpenAI+Anthropic+Gemini / add OpenRouter / US-only-default + opt-in non-US. **Rec:** add OpenRouter behind existing abstraction + US-only default with PRC chains opt-in (resolves security-5).
2. **Web search backend** — _Options:_ Tavily / Brave / SerpAPI / self-hosted SearXNG. **Rec:** Tavily first behind the pluggable provider; SearXNG later for full self-host. Query must route through `classify_sensitivity`.
3. **Mobile + desktop memory backend** — _Options:_ SQLite-mobile + Postgres-desktop behind one Protocol / SQLite everywhere / Postgres everywhere. **Rec:** SQLite+sqlite-vec mobile, Postgres+pgvector desktop, one Protocol (matches docs decision 6); in-RAM default unless `DATABASE_URL`.
4. **Edit scope this cycle** — _Options:_ P0 only / P0+P1 / P0+P1+P2. **Rec:** P0+P1 (safe to expose AND correct); isolate P2 for measured rollout.
5. **`setup_database.py` disposition** — _Options:_ delete / thin wrapper over `tools/migrate` / keep+banner. **Rec:** thin role-create/verify wrapper deferring to `tools/migrate apply`; update `setup.sh`+`SETUP.md`.
6. **Speaker-ID enrollment posture** — _Options:_ keep fail-open + P1 suppression / require enrollment in speaker mode / push-to-talk until enrolled. **Rec:** keep fail-open default + land P1 suppression + first-run enroll nudge when no headset (target_architecture decision 9 makes speaker-ID non-optional for v1 laptop-mic case).

## 6. Top Risks

1. Memory — the headline differentiator — is non-functional at runtime; the built Postgres work is unreachable until P2.
2. Two real-time cancellation bugs (stale post-barge-in sentence; non-interruptible escalated turns) degrade the most-distinguishing feature.
3. The remote path is unsafe to expose (unauthenticated `/token` on `0.0.0.0` + dev keys) — P0 before any non-localhost demo.
4. OSS-readiness blocked by absent LICENSE + unmet Apache-2.0 attribution (cheap, easy to forget).
5. "Cloud for research" is hollow — gathers from 5 hardcoded sentences.
6. Cost-control gap: cancelled Hedge cloud workers keep billing; stalled winner can wedge a turn.
7. Cross-platform drift compounds — only ~4 functions shared with the Dart shell today.
8. Shallow config merge can silently disable the cloud tier / strand stale keys, undermining "runs on many machines."
