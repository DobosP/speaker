# Speaker â€” Architecture Gap Report & Implementation Roadmap

**Date:** 2026-06-10 Â· **Scope:** full repo (`core/`, `always_on_agent/`, `utils/memory*`, `remote/`+`web/`, `mobile/`, `tools/`, `docs/`, config plane) Â· **Basis:** subsystem map (10 subsystems) + 45 adversarially-verified findings
**Target:** a secure, open-source, real-time voice assistant with layered memory and smart routing, running well on many machine types, using external LLM providers for research/hard tasks.

---

## 1. Overall assessment

The codebase is architecturally far better than a typical prototype. The `AudioEngine` seam (`core/engine.py`), the `AgentEvent`/`Mode` shellâ†”core contract (`always_on_agent/events.py`), the sensitivity-routed cloud chains (`core/sensitivity.py` â†’ `core/routing.py` â†’ `core/llm.py`), the epoch-gated cancellation model, and the async observability stack are real, tested, and well documented. The architecture docs (`docs/target_architecture.md` Â§9/Â§9.7/Â§10) already specify most of what's missing â€” the gap is execution, not design.

But the project is currently failing its two most absolute stated requirements:

1. **The repo is PUBLIC today with committed secrets and biometric PII.** A tracked `.env` carries two real-format Google/Gemini API keys in pushed history (commit `d32db9f`), and 52 tracked files under `logs/runs/` include the owner's raw 16 kHz voice WAVs and verbatim transcripts â€” `.gitignore:44-46` deliberately commits them. This violates Goal 1, CREDENTIALS.md's own golden rule, and Â§9.7's "raw audio never leaves the device" in the published artifact itself.
2. **Several headline capabilities exist as machinery, not behavior.** Layered memory has zero cross-session continuity in any default config; the canonical Postgres migration path is broken by invalid SQL; smart routing has no quality axis (cloud is only a first-token latency hedge), no spoken fallback, and a dead KWS tier; cross-platform is one Python core plus a deliberately divergent Dart shell sharing only two pure-function families, on an installer that silently degrades the flagship barge-in.

The good news: almost every gap has its seam already built (Memory protocol, `ChainSelector`, `RecallConfig`, `EngineCallbacks`, golden fixtures). The roadmap is overwhelmingly **wiring and policy extraction, not rewrites** â€” consistent with the instruction to evolve `core/routing.py`, `utils/memory*.py`, and friends rather than discard working code.

---

## 2. Findings summary by severity (45 verified findings)

| Severity | Count | IDs |
|---|---|---|
| **Critical** | 1 | security-1 (tracked `.env` with real Google keys in public history) |
| **High** | 15 | layered-memory-1/2/3/4 Â· smart-routing-1/2 Â· realtime-concurrency-1/2 Â· cross-platform-1/2/3 Â· architecture-quality-1/2/3/4 |
| **Medium** | 19 | security-2/3/5 Â· layered-memory-5/6/7/8/9 Â· smart-routing-3/4/5 Â· realtime-concurrency-3/4/5/6 Â· cross-platform-4/5/6 Â· architecture-quality-5 |
| **Low / uncertain** | 10 | security-4 (corrected low) Â· security-6 Â· layered-memory-10 Â· smart-routing-6 Â· realtime-concurrency-7/8/9 Â· cross-platform-7 Â· architecture-quality-6 (corrected low) Â· architecture-quality-7 |

Highest-leverage single observations:

- **security-1 (critical):** `.env` tracked since `d32db9f`; two 39-char `AIzaSyâ€¦` Gemini keys (one on an *active* assignment); repo confirmed public. Rotate + untrack + history purge + CI secret scan.
- **layered-memory-3 (high, Â§9.7):** the recall block at `core/capabilities.py:341-348` skips the sensitivity float that recent-turns and images get â€” recalled PII can ride the "public" (cheapest, potentially PRC) chain. **Must land before memory is ever enabled.**
- **realtime-concurrency-2 (high):** up to three blocking LLM calls (addressing gate, cleaner, LLM-assisted router) run on the audio capture thread per ASR final in the shipped desktop/4090 profiles (`sherpa.py:1597` â†’ `runtime.py:422-494`), starving mic reads, KWS polling, and the heartbeat â€” violating `core/engine.py:17-50`'s own contract.
- **realtime-concurrency-1 (high):** the replay regression harness (`app.py:209-216`) drains the bus from two threads â€” uncaught `queue.Empty` crashes plus violation of the supervisor's bus-thread-only invariant â€” poisoning the measurement loop the open barge-in P1 depends on.
- **smart-routing-1/2 (high):** cloud is only a first-token race (no RESEARCH escalation, `ChainSelector` keys only on sensitivity), and tier failure is mute (Ollama down on the default profile = silent assistant; timeouts apologize, errors don't).
- **cross-platform-1 (high):** `tools/install.py` omits scipy/soxr; `requirements.txt` is the deleted pre-refactor stack â€” every fresh machine runs degraded barge-in and a NOT-READY doctor.

---

## 3. Per-goal scorecard

| # | Goal | State | One-line verdict |
|---|---|---|---|
| 1 | Secure & OSS-ready | **Weak** | Good code-level discipline (env-only keys, redaction helper, hardened compose, MIT LICENSE+NOTICE) defeated by the published artifact: committed keys + voice PII on a public repo; no secret-scan CI; token-server identity/500-detail gaps. |
| 2 | Layered memory | **Weak** | Excellent protocol seam + hardened Postgres engine, but every tier is dark/dead at defaults: no continuity, broken migration 002, dead knobs (`recall_min_similarity`, `meeting_persist`, `persist_roles`â€¦), recall Â§9.7 bypass, nominal episodic/profile tiers, zero mobile memory, no SQLite. |
| 3 | Smart routing | **Weak** | Tier router + sensitivity/jurisdiction chains are real and tested; but no quality/difficulty axis, mute tier failure, dead KWS instant tier, zero cost accounting, substring-marker misfires, dormant headroom signal, route recomputed 3Ã— with divergent contexts. |
| 4 | Real-time quality | **Adequate** | The strongest pillar (epoch gating, deterministic cancel ordering, reaping, async logging) with verified dents: LLM-on-capture-thread, replay-harness race, two barge-in TOCTOUs, queued-task resurrection, watchdog false positives, unbounded supervisor histories; barge-in policy welded into `sherpa.py`. |
| 5 | Cross-platform | **Weak** | Desktop core honors the architecture; everything at the edges fails: installer deps, unprovisionable phone profiles, silent `--device` typos, mobile sharing only `contract.dart` (analyzer fixture has no Dart runner; shipped stop-matching drift), remote path without barge-in parity, POSIX-only command-safety regexes. |
| 6 | External providers | **Adequate** | `OpenAICompatLLM` + `ProviderProfile` + `HedgeLLM` + hard stream close is a genuinely good multi-provider base with env-only keys and jurisdiction gating; missing the operational half: no per-call logging, no spend estimate/budget, one global strategy, no frontier-provider presets. |

---

## 4. Prioritized roadmap

> Sequencing rationale: security/PII first (active public exposure), then real-time correctness + the one Â§9.7 hole that gates memory enablement, then memory, then routing/providers (which consume memory context and the P1 fallback work), then cross-platform (which needs the deterministic replay harness), then the OSS-readiness sweep. Every phase evolves existing modules; the only "extraction" (BargeInDetector) is an additive move.

1. ### P0 â€” Security & privacy emergency *(repo is public today)*
   - **Rotate/revoke** the two Google/Gemini keys in `.env` (the `OPENAI_API_KEY` value is a placeholder). `git rm --cached .env`; verify nothing else secret is tracked; ship `env.example` only.
   - **Untrack the PII run bundles** (6 `.wav` + transcript-bearing bundles under `logs/runs/`); flip `.gitignore:44-46` to ignore-by-default; move regression fixtures to synthetic audio / a private store (Decision D2).
   - **History rewrite** with `git filter-repo`/BFG purging `.env` + PII bundles; tag the pre-rewrite SHA privately; force-push in one coordinated window (use the documented Windows landing workflow â€” this legitimately needs the privileged path).
   - **CI secret-scan gate** (gitleaks) in `.github/workflows/`; add `SECURITY.md`.
   - **DSN hygiene:** redact in `tools/migrate.py:63` (reuse `_redact_db_url` from `setup_database.py:40`); prefer `PGPASSWORD` over `--password` (`setup_database.py:228/:259`).
   - **token_server:** replace `detail=str(exc)` at `:267`/`:293` with generic messages + server-side logging.
   - Reconcile `docs/debugging.md` + CLAUDE.md's commit-the-bundles convention with Â§9.7 (committed bundles must be PII-free).
   - *Risk:* history rewrite invalidates clones/PR refs across multiple dev machines â€” coordinate, mirror privately first.

2. ### P1 â€” Real-time correctness + the Â§9.7 recall boundary fix *(depends: P0 sequencing)*
   - **rc-1:** `wait_idle` poll-only when the bus thread runs (mirror `tests/sandbox/scenario.py:83-92`); catch `queue.Empty` in `EventBus.drain_once`; fix `tests/replay_voice_driver.py`. â†’ makes the replay harness deterministic (gates everything barge-in).
   - **rc-2:** `_on_final` does only mark+enqueue on the engine thread; addressingâ†’cleanerâ†’routerâ†’publish moves to a single-worker thread with a one-deep supersede queue.
   - **rc-3:** post-`speak()` re-check of `tts_request_allowed` + per-utterance generation counter replacing the shared `_stop_speaking.clear()` (`sherpa.py:1776-1781`).
   - **rc-4:** `cancel_all` cancels queued tasks; `_start_queued_tasks` swap under `_cancel_lock`; `_start_task` drops pre-cancelled tasks.
   - **sr-2:** spoken apology on `TASK_FAILED` (mirror `_TIMEOUT_APOLOGY`); cross-tier fastâ†”main retry in `core/capabilities.assistant()`.
   - **lm-3 (Â§9.7):** float `classify_sensitivity` over the recall block â€” or force PRIVATE on any non-empty recall â€” at `core/capabilities.py:341-348`; routing test beside `test_capability_context_isolation.py`. **Hard precondition for P2.**
   - **rc-5:** `handled_local` metric stamp on no-LLM turns; watchdog skips them (kills fake `llm stuck` hints).
   - **rc-6/aq-7:** `deque(maxlen)` on `SupervisorState` histories; drop `STT_PARTIAL` from `event_log`.
   - **lm-1:** rewrite `migrations/002_constraints.sql` with the DO-block `duplicate_object` pattern (proven at `utils/memory.py:129-148`); add a postgres+pgvector CI service job running `tools/migrate.py apply`.
   - *Risk:* touches the actively-tuned barge-in path â€” land rc-1 first, validate against replay WAVs + one live session.

3. ### P2 â€” Layered memory: make the tiers real and portable *(depends: P1)*
   - **Continuity (lm-2):** seed `_summary_head` from the newest `summaries` row; cross-session fallback for `_load_recent_messages`; one-shot "Last session" block at startup. Three small wires against existing tables.
   - **Wire or delete dead knobs (lm-4/5/6/8):** `recall_min_similarity`/`recall_max_items` through `RecallConfig`; `meeting_persist` through `_build_memory`â†’adapterâ†’manager; `persist_roles`/`memory_persist_assistant`. Add a test that every `config.json` memory key is consumed.
   - **Provenance (lm-6):** migration 005 adds `tags`/`channel` to `messages`; `ingested` becomes RAM-only unless opted in; recall excludes/down-ranks ambient rows.
   - **Retention + deletion:** schedule `apply_retention` daily on the writer thread (keep the `stop()` pass); add a "forget that" intent; answer `PROJECT_KICKOFF.md:98-99` (Decision D8) and record it in `target_architecture.md`.
   - **Window parity (lm-7):** forward `working_window` into the adapter ring; separate ambient vs conversation caps; flood-parity test in `test_memory_contract.py`.
   - **Episodic (lm-5):** persist assistant finals (`role='assistant'`, turn pairing) and/or an `episodes` row per completed task from `_handle_task_completed`.
   - **Recall quality:** ~300-char per-item budget + timestamps; decouple profile injection from `recall_enabled`.
   - **Portability (lm-9/aq-4):** factor summary/profile/TTL **policy** out of `utils/memory.py` (template: `memory_writer.should_persist`), leaving `MemoryManager` as the Postgres I/O adapter; implement `SqliteVecMemory` behind the Memory protocol (Decision D6) â€” serves desktop-without-Postgres now, mobile later; parity-run `test_memory_contract.py`.
   - **Bless the recipe:** one "memory on" config per profile, bench TTFT with `tools/bench`, enable recall on desktop (Decision D5); rewrite `MEMORY.md` (lm-10).
   - *Risk:* migration on live DBs; recall changes every prompt â€” lm-3 from P1 is the safety precondition.

4. ### P3 â€” Smart routing + external thinking tier *(depends: P1, P2)*
   - **Quality axis (sr-1):** key `ChainSelector` on (sensitivity, intent_kind/route-action); per-chain `strategy`/`ttft_deadline`/`max_tokens` in `llm.cloud_chains` (a cloud-first `research` chain); optionally a per-turn prefer-cloud hint following the pinned `hedge_delay_ms` override pattern. All additive; local-first default preserved.
   - **Provider presets (Decision D3):** add chosen frontier providers (Anthropic/OpenAI via `ProviderProfile`, OpenRouter as aggregator) to `llm.cloud_providers`, env-only keys; extend `tools/llm_sanity.py` probes.
   - **Cost accounting first (sr-3):** structured `llm_request` logging in `OpenAICompatLLM.stream` + hedge-winner line in `HedgeLLM`; per-run estimated spend in `RunSummary` from `_pricing_usd_per_mtok`; optional `llm.cloud.max_usd` budget â†’ local-only on exhaustion (Decision D4); raise default `hedge_delay_ms` above the measured ~270 ms warm local TTFT.
   - **KWS (sr-4):** `setup_models.py --kws` + keywords file generated from the `commands` block; `tools/doctor` reports KWS presence.
   - **Marker hygiene (sr-5):** word-boundary `_compile_markers` applied to `react._ESCALATE_MARKERS` and `capability_router._ACT_MARKERS` (copy helper into `always_on_agent` to preserve the import inversion); regression tests for `findings`/`display brightness`.
   - **One decision per turn (aq-5):** compute `RouteDecision` once in `_on_final` with full context; carry it in the event payload/task metadata; tier router, escalate predicate, and summary logger read the carried decision.
   - **Enable `live_routing` (sr-6)** on cpu_laptop/phone_lite/macbook after a `tools.live_session` validation run.
   - *Risk:* chain-semantics changes affect billing â€” cost logging lands first inside the phase.

5. ### P4 â€” Cross-platform: install, profiles, engine parity, mobile convergence *(depends: P1)*
   - **Installer (cp-1):** scipy+soxr into `RUNTIME_DEPS`; test asserting `RUNTIME_DEPS âŠ‡ doctor.REQUIRED_IMPORTS`; rewrite `requirements.txt` (or pyproject extras) â€” it currently lists the deleted legacy stack and omits sherpa-onnx.
   - **Low-end profiles (cp-2):** `setup_models.py --gguf` (reuse `tools/bench/models.py` manifest) â†’ `config.local.json`; `llama-cpp-python` as a documented extra; `build_llms` fail-fast on a missing GGUF (mirror `_require_asr_models`).
   - **Profile validation (cp-5):** unknown `--device`/`config.device` exits with valid profile names (also in `remote/worker.py`).
   - **BargeInDetector extraction (aq-1, Decision D7):** move `_looks_like_user`'s cascade + floors + latch + refractory into a pure-policy class under `core/engines/`; engine keeps all PortAudio/threading; replay-WAV regressions pin the P1 tuning; LiveKitEngine consumes the same detector; the class becomes the Dart-port spec.
   - **Engine conformance (aq-2, cp-6, rc-9):** document metric/heartbeat obligations in `core/engine.py`; parametrized conformance test (scripted/file_replay/livekit-fake-room); LiveKitEngine stamps `TTS_FIRST_AUDIO`/`BARGE_IN_STOP`, gains latch+refractory+`note_barge_in_storm`; pin `echoCancellation` constraints in `web/app.js`.
   - **Mobile convergence (cp-3, aq-3, lm-9):** (1) extract plain-Dart `AssistantController` from the widget State; (2) port `events.py` â†’ `events.dart`; (3) Dart runner for `speech_analyzer_contract.json` (fixture + Python gate already exist); (4) route stop/endpoint through the ported analyzer, retire or contract-pin `_looksLikeStop`; (5) bounded recent-turns prompt block (mobile working tier). `_turn`â†’speech-epoch unification follows. **Do not discard the working Dart loop.**
   - **Windows command safety (cp-4):** per-platform agent_brain allow/deny lists with a cross-OS denylist floor (Remove-Item -Recurse, del/rd /s, format, reg delete, Stop-Computer, iwr|iex) + per-platform tests.
   - *Risk:* the extraction touches the most actively-tuned code â€” gate behind replay regressions (enabled by rc-1) + a live validation. Mobile needs on-device validation; iOS stays out of scope this phase.

6. ### P5 â€” Remote hardening + docs/consistency sweep (pre-OSS checklist) *(depends: P0, P4)*
   - **Per-principal `/token` auth (security-3):** bind minted identity/room to the authenticated principal (per-user tokens or a tokenâ†’(identity, rooms) allowlist); CREDENTIALS.md:149 already records the gap.
   - **web/app.js:** stop accepting `?token=` in the URL (history/proxy leak); explicit set-token UI, header-only.
   - **Mobile model integrity (security-6):** compile-time SHA-256 of the Gemma `.task`, verified over `.part` before the atomic rename.
   - **Docs pass (cp-7 + docs-subsystem):** fix `deployment_profiles.md` merge semantics + stale `local_only` guardrail per Â§9.7; reconcile `--llm` choices (add `llamacpp` to argparse â€” the factory already supports it); fix `docker/.env.example` compose path; mark `AlwaysOnAgentRuntime`/`bridge.py` as test-support; delete stale `.pyc` files.
   - **OSS checklist:** CONTRIBUTING.md; LICENSE/NOTICE header policy confirmed (MIT + NOTICE already present); final gitleaks + PII sweep over full history before announcing.
   - *Risk:* low; the `/token` auth change breaks existing clients â€” migrate the web client in the same commit.

---

## 5. Decisions needed before implementation

| # | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | History remediation depth | full filter-repo rewrite / rotate-only / go-private-then-rewrite | **Full rewrite**, optionally fronted by going private during the work â€” rotation can't fix biometric voice WAVs. Mirror the pre-rewrite state privately. |
| D2 | Replay fixture policy post-purge | synthetic-only / private asset store / per-file review | **Synthetic-only public + private store** for the real-voice fixtures that pin the P1 tuning. |
| D3 | Frontier providers for the thinking tier | keep current set / +Anthropic+OpenAI native / OpenRouter-only / +Google too | **Anthropic + OpenAI as native `ProviderProfile` presets**, OpenRouter as aggregator fallback; defer Google; keep `allow_prc` default-false. |
| D4 | Cloud cost control | logging only / hard budget cap / escalation-only cloud | **Budget cap (`llm.cloud.max_usd`) on top of full instrumentation** + raise default `hedge_delay_ms` above warm local TTFT. |
| D5 | Memory defaults | all off (today) / on for desktop profiles after bench / on wherever DATABASE_URL | **On for desktop/4090 after a `tools/bench` TTFT validation** â€” gated on the lm-3 fix. |
| D6 | No-Postgres/mobile memory backend | Python `SqliteVecMemory` now / Dart-first later / Postgres-only | **Python SqliteVecMemory now** behind the existing protocol; it also fixes desktop-without-Postgres and becomes the Dart reference. |
| D7 | Edit scope on tuned barge-in code | surgical only / extract in P4 behind replay tests / extract immediately | **Extract in P4 behind replay-WAV regression tests**; P1 does only the surgical race fixes. |
| D8 | Transcript retention & encryption (blank at `PROJECT_KICKOFF.md:98-99`) | TTLs only / TTLs + at-rest encryption / summaries-only persistence | **TTLs now (already half-built), record the decision in `target_architecture.md`**; at-rest encryption as the follow-up for the SQLite/mobile tier. |

---

## 6. Top risks

1. **Active public exposure** â€” real Google keys + the owner's voice biometrics are published right now; every day before P0 is ongoing leakage, and rewrite cost grows with forks/clones.
2. **Â§9.7 regression on memory enablement** â€” enabling recall (P2) without the lm-3 fix (P1) ships recalled PII to the cheapest cloud chain. Phase order is load-bearing.
3. **The barge-in P1 is tuned on a broken measurement loop** â€” replay-harness race (rc-1) + LLM-on-capture-thread audio loss (rc-2) mean existing tuning conclusions may need re-validation after P1.
4. **Silent fresh-install degradation** â€” missing scipy/soxr reproduces *different* barge-in behavior per dev machine, on a project explicitly worked from multiple boxes.
5. **Pythonâ†”Dart drift compounds per commit** â€” shipped behavioral divergence with only 2 of 3 golden fixture sets pinned; the repo's own 2026-05 review called this the main structural risk.
6. **Silent total failure UX** â€” Ollama down = mute assistant + watchdog false positives training operators to ignore real signals.
7. **Invisible, structurally-high cloud spend** â€” hedging bills nearly every cloud-enabled turn with zero logging or budget; instrument before expanding cloud (P3 ordering).
8. **History-rewrite operational risk** â€” force-push across multiple machines + the Windows guard-hook workflow; tag and mirror before rewriting.

---

## 7. Where the docs already cover it

- Local/cloud boundary, hybrid topology, SQLite-on-mobile, speaker-ID gating: **resolved** in `docs/target_architecture.md` Â§9 (esp. Â§9.6â€“9.9) â€” the roadmap implements, it does not re-decide.
- Sensitivity-routed chains + provider quirks: specified in Â§10.4â€“10.7 and largely built; P3 extends (quality axis, per-chain strategy) rather than replaces.
- Pythonâ†”Dart drift risk + the golden-contract gate: named in `docs/archive/code_review_2026-05.md` Â§6; the analyzer fixture and Python test exist â€” only the Dart consumer is missing (P4).
- `/token` any-identity gap: already acknowledged at `CREDENTIALS.md:149` (P5 closes it).
- Known stale docs to fix, not re-litigate: `deployment_profiles.md:72-74` `local_only` guardrail (superseded by Â§9.7), `MEMORY.md` (deleted `main.py`, wrong defaults, phantom flags), CLAUDE.md `--llm llamacpp` claim.

*This report gates the implementation phase. Execute P0 immediately; P1â€“P5 in order, with the decision table resolved before their respective phases begin.*