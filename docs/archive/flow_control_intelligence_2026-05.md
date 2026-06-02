# Flow-control intelligence — the decision & escalation layer

> ⚠️ **Superseded — durable content merged into [`docs/unified_architecture.md`](../unified_architecture.md).** Kept for revision history; do not treat as current. (2026-06-02 consolidation.)

This documents the "smart decision layer" of the voice assistant: how a turn is
classified, which model tier answers it, and when it escalates to multi-step
reasoning. It is the reference for the `llm.router`, `device_profiles.*`,
`agent.planner`, and `input_gate` config blocks. Companion audit:
[`perf_audit_2026-05_goal_alignment.md`](perf_audit_2026-05_goal_alignment.md).

## The ladder (cheapest → most capable)

A user final flows through four gates, each able to short-circuit the next:

1. **Should we act at all?** — `core/addressing.py` (`input_gate`).
   ACT / INGEST / UNSURE classification keeps ambient speech, read-aloud text,
   and disfluency fragments from triggering a reply. `enabled=false` in the base
   config; **on for the active `desktop` profile** (the fast tier has GPU
   headroom). `unsure_acts=true` answers on ambiguity (legacy behaviour);
   `false` is conservative (silently ingest). Needs `llm.fast_model`.

2. **Is it a control phrase?** — `core/intents.py` + the KWS command fast-path.
   "stop" / "confirm" / mode switches act with no LLM in the loop.

3. **Which model tier?** — `core/routing.py` (`HeuristicRouter`).
   A dependency-light, phone-safe lexical scorer maps the query to `[0,1]`;
   `score >= threshold` → the **main** model, else the **fast** model. Signals:
   mode, intent kind, length buckets, `_COMPLEXITY_MARKERS`
   (why/how/explain/compare/…), `_GENERATION_MARKERS` (story/poem/"tell me a"/…
   — one hit adds +0.5), and a double-question bump. An optional, additive-only
   live nudge (rolling local TTFT + load) can tip a *borderline* turn toward
   main when the local tier is slow (never subtracts — see `live_routing`).

4. **Does it need multi-step gathering?** — `always_on_agent/react.py`
   (`agent.planner`). ASSISTANT-mode queries that hit an `_ESCALATE_MARKER`
   (search / look up / find / compare / "and then" / step by step / …) run a
   bounded ReAct plan→execute loop over the configured tools instead of a
   one-shot reply. Gated, so ordinary short turns stay one-shot.

## What changed (2026-05, goal-alignment fixes)

The audit found the layer was running in a **fast-only** configuration:
`125/125` field "answering" turns went to the small model, the main tier had
zero activations, and the ReAct planner was unreachable. Three changes fix it:

| Change | Where | Why |
|---|---|---|
| `router.threshold` **0.5 → 0.3** (base) | `config.json` `llm.router` | At 0.5, short spoken reasoning queries (`"explain how X works"` ≈ 0.36) never cleared the bar, so the main model was never used. 0.3 lets reasoning + long-form turns escalate. Per-profile overrides still win. |
| `_GENERATION_MARKERS` (+0.5, one hit) | `core/routing.py` | "tell me a story / write a poem / walk me through" are unambiguous generation asks the fast tier deflects or answers shallowly — route them to the big model. Kept distinct from `_COMPLEXITY_MARKERS` so the calibrated borderline-router tests are undisturbed. |
| `agent.planner.enabled` **false → true** | `config.json` `agent` | Makes the ReAct ladder (gate 4) reachable so gather/synthesize queries stop degrading to one-shot. |

### Tier choice is local-only by default
Both tiers are **on-device** (e.g. `gemma3:12b` main + `gemma3:4b` fast on the
desktop GPU). Escalation to "main" therefore trades a little latency for quality
with **no cloud egress** — the cloud chains (`llm.cloud_chains`) remain disabled
unless explicitly enabled, and even then are gated by `core/sensitivity.py`
(§9.7). Lowering the threshold cannot leak data; it only picks the bigger local
model more often.

### Calibration notes
- The router is **lexical, not semantic** — a deliberate phone-safe choice. It
  mis-scores a small class of short reasoning questions (single complexity
  marker ≈ 0.18 → fast), which is arguably correct for a voice assistant. The
  learned path (`LearnedRouter`, RouteLLM-style BERT) is opt-in
  (`router.backend = "learned"`, desktop-only) when a meaning signal is wanted.
- Per-profile thresholds let constrained devices opt out: a phone whose "main"
  model is itself slow can raise its `device_profiles.phone.llm.router.threshold`
  back up so it doesn't escalate into a sluggish tier.
- To verify escalation is live, set `SPEAKER_DEBUG_ROUTING=1` and watch the
  `[route] main|fast …` lines, or record a session and confirm a real
  `gemma3:12b` request appears in `run-<id>.summary.json`.
