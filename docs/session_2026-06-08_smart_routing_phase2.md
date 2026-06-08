# Session 2026-06-08c — Smart-routing phase 2: audit + hardening

**Branch:** `feat/smart-routing-phase2-hardening` → merged to `main` @`f7bf559`.
Logic suite **1398 passed, 14 skipped, 0 failed**. Same box (Windows `.venv`).

## How: a 6-dimension fan-out audit, then implement the verified top items
Phase 2 began with a multi-agent **Workflow** (`smart-routing-audit`): 6 parallel
analysts (one per dimension) → **adversarial verify** every finding (refute-by-
default, re-read the code) → synthesis. 35 agents, **28 findings, 19 confirmed,
9 refuted**. The verification earned its keep — it dropped plausible-but-wrong
"boundary leak" claims that were actually §9.7-authorized (screen captures may
ride a US cloud per resolved-decision #7) or post-ASR-text egress (the boundary
protects *raw audio*, not post-ASR text), and several misreads of the deep-merge
/ hedge wiring.

## Verdict: smart routing is largely HEALTHY and fail-safe
No P0. No active §9.7 boundary violation. Every risky path is double-bounded and
fails toward the more-private / more-conservative choice. The dominant through-
line is **dormant intelligence**: `live_routing`, `cost_order`, and
`capability_router` are all built, wired, and unit-tested but **off in most
shipped profiles** — and the *most-capable* shipped device (desktop_gpu_4090) had
strictly *less* routing intelligence than the weaker base `desktop`.

## Landed (3 verified, headless-confirmable, boundary-safe fixes)

### 1. PII fail-safe for lowercased ASR — `core/sensitivity.py`
The name+money PII rule (`_NAME_AND_MONEY`) is case-**sensitive** because it keys
off a capitalized proper name (`_NAME = [A-Z][a-z]+`). But production sherpa ASR
emits **lowercase**, so `"what is john salary"` failed the name match, `"salary"`
isn't in `_PII_CATEGORY`, and `_PUBLIC_MARKERS` (`^what is`) sent it to **PUBLIC**
— i.e. third-party financial PII could ride a public (possibly PRC-hosted) chain
when cloud is enabled (the one fail-**unsafe** path). Fix: a new case-insensitive
`_COMPENSATION` rule (salary/wage/income/paycheck/pay stub/net worth/bonus) forces
**PRIVATE** regardless of casing or an adjacent name. Fails toward PRIVATE (the
§9.7 safe direction); a rare over-match ("average teacher salary") only keeps the
turn local. Test: `test_lowercased_compensation_pii_fails_safe_private`.

### 2. Host-aware cost ordering — `core/routing.py` `_preset_cost_key`
`cost_order` sorted **latency-first** and was **host-blind**: a CN provider could
float ahead of a US one the user ordered first, and a cost-annotated US aggregator
with no `ttft_ms` (OpenRouter) sank **below** CN. On the shipped public chain
`[openrouter(US,no-ttft), deepseek(CN,400), cerebras(US,80)]` this produced
`[cerebras, deepseek(CN!), openrouter]`. Fix: add `host_rank` as the **outermost**
sort key (CN → after all US/unknown), so cost/latency optimizes **within** a
jurisdiction tier, never across it → `[cerebras, openrouter, deepseek(CN-last)]`.
The existing ttft-primary contract (and all its tests) is preserved — the existing
test providers are same-host. Latent until `cost_order` is enabled, fixed now so
enablement lands on correct order. Tests: `test_order_presets_cn_sorts_after_us_*`.

### 3. `capability_router` ON for `desktop_gpu_4090` — `config.json`
The shipped device (pinned in `config.local.json`) inherited the base default
`enabled=false` while base `desktop` turns it on — so the most-capable profile had
the least routing intelligence. Mirrored desktop's block: action classification
(CONTROL/SIMPLE/RESEARCH/ACT) drives tier + planner-escalate from one module, with
fast-LLM disambiguation only on **low-confidence long turns** (≥12 words, conf 0.5).
Non-breaking, read-only (no state-changing ACT tools exist; `act_confirm` is a
no-op today). Verified: base default still off; only `desktop` + `desktop_gpu_4090`
have it on; cloud profiles untouched. **Latency note:** `llm_assist=true` adds one
fast-tier (gemma3:4b ~1.8s here) call on long ambiguous turns; set `llm_assist=false`
in `config.local.json` if that latency is unwelcome.

## Deferred (the audit's #4 + P2 — next steps, with the path now unblocked)

### #4 Activate the dormant cost/latency levers on the CLOUD profiles + measure
Set `live_routing:true` (llm block) + `cloud.cost_order:true` on `cpu_laptop` /
`phone_lite` (optionally `macbook_m_series`) where local is slow and cloud is on;
keep desktop/4090 **off** (local 12b is fast). The cost_order fix above unblocks
this. **Pair with measured evidence** — extend `tools/bench` (or a replay smoke)
to emit the chosen chain order and assert cost_order lowers TTFT and the live nudge
shortens the hedge under a high-load snapshot (the first real proof these levers
help). **Needs cloud API keys to validate**, so do it on a cloud-enabled device.

### P2 polish (verified, low-impact) — full list in `.agents/backlog.md`
Daemon-thread-leak WARN on hung workers; WINNER_SELECT_BUDGET_FLOOR 30s → ~10s;
word-boundary-aware tier markers (+ compose/draft verbs); centralize
`capability_context` so escalated ReAct/research.local turns publish sensitivity;
tier-aware `load_fraction` + faster monitor cadence when live_routing on;
LearnedRouter build-path test; doc-truth fixes.

## Next steps (pick up here)
1. **Audit #4** — enable live_routing + cost_order on the cloud profiles, with a
   bench/replay smoke for measured evidence (needs cloud keys).
2. **Phase-1 carryover (owner #2/#3):** prosody endpointing is *activated* on the
   Windows `config.local.json` — needs the owner's live tuning. SenseVoice
   agreement-guard (#3) still open (needs real recorded clips).
3. **P2 routing polish** as appetite allows.
