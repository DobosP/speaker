# Session 2026-06-14 (pt6) — Audit-driven hardening: memory, image, prompt-injection

**Headline:** Audited the app's **memory management, image/visual memory, and
prompt-injection protection** against the open-source state of the art (Mem0,
Letta/MemGPT, Zep/Graphiti, Generative-Agents, Microsoft spotlighting, OWASP
LLM01, Presidio, screenpipe/Pensieve), then landed the best-fit features as **four
sequential, adversarially-reviewed slices** on `main`. All four are **default-OFF /
additive** (opening-turn prompt byte-identical). Full logic suite **1736 passed,
10 skipped**.

## How it was built

1. **6-agent research+audit workflow** — 3 web-research agents (the OSS landscape
   per domain) + 3 internal read-only audit agents (the app's real state + gaps).
2. Owner picked all four improvement areas (Security / Memory quality / Visual /
   Procedural).
3. Four slices, each: implement → tests → full gate → a **focused adversarial-
   review workflow** (review dimensions → verify each finding) → fix confirmed
   findings → land. **No blocker findings across any slice.**

## What landed (commit map)

| Slice | Commit (feat + merge) |
|---|---|
| 1 Security hardening | `b73928a` → `dd9dc11` |
| 2 Visual memory quality | `4abbfee` → `35970fa` |
| 3 Memory quality | `aecd733` → `e12abac` |
| 4 Procedural memory | `a265cc5` → `f243932` |

### Slice 1 — prompt-injection spotlighting + OCR PII redaction
The standout gap (zero coverage; all 3 research streams ranked it #1). NEW
`always_on_agent/untrusted.py` (stdlib): `wrap_untrusted()` spotlights untrusted
content in a per-process unguessable fence + a never-obey directive (strips forged
fences; flags embedded injection); `detect_injection()` (obfuscation-resistant);
`redact_pii()` (provider keys/Bearer/env-prefixed secrets/Luhn cards/SSN/email/
phone). Wired at **four** untrusted-content sites: recalled+last-session+vision
blocks (capabilities), egress web results in the ReAct planner, egress steps in
the RESEARCH plan path (`research_synth`), and OCR **+ caption** redaction before
a `vision` memory is persisted. Command/tool fast-path quarantine verified
structurally. 25 tests; 16 review findings fixed.

### Slice 2 — perceptual-hash near-dup gate + structured caption
`_dhash()` (Pillow, LANCZOS) replaces exact-SHA1 in `observe()` so a cursor
blink / clock tick / 1px scroll no longer burns a caption + a memory row; SHA1
fallback when Pillow absent; throttle pre-check avoids the decode for a frame the
interval would discard; `_last_phash` tracked unconditionally (no stale-mis-gate).
Structured entity-rich caption prompt. 2 review findings fixed.

### Slice 3 — multi-signal recall scoring (recency + importance)
`RecallBudget` gains `recency_weight` / `recency_half_life_days` /
`importance_weight` (all default 0 = OFF, byte-identical). `_apply_signals()`
scales the native relevance score by a **convex [0,1]** blend of exponential
recency-decay + a kind-importance prior (profile>summary>message>vision), in the
**shared** recall layer so all backends benefit; a **negative cosine** score is
left unchanged (the review-caught major: multiplying a negative by <1 ranked it
up). 1 major fixed. **Deferred (DB-coupled):** hybrid lexical+semantic retrieval
(RRF / migration 005 pg_trgm) and supersede-on-contradiction (bi-temporal
migration + LLM consolidation; profile facts already overwrite).

### Slice 4 — procedural memory (user-taught behavior rules)
NEW `always_on_agent/procedural.py`: `extract_rule()` (high-precision, requires a
teaching frame; rejects bare always/never, statements, content requests, non-name
"call me" idioms) + `render_rules()`. New `procedural_rules()` protocol verb on all
3 backends; RAM keeps rules in a list exempt from the working-window cap; SQLite
scans whole-table by tag, excludes from cosine recall, exempt from age-TTL;
Postgres stores under a reserved `rule::` profile key (never TTL'd, excluded from
the profile block). Capture runs on the live query before escalation, guarded by
`detect_injection()`; the block is injected TRUSTED (not spotlighted) and floats
§9.7 sensitivity. 15 review findings fixed.

## Defaults / how to turn things on

All default-OFF. To enable in `config.json` `memory` (or a profile):
- prompt-injection spotlighting + OCR redaction: **ON by default when their
  feature is active** (only activate when untrusted content is present); env
  escape hatches `SPEAKER_DISABLE_SPOTLIGHT=1` / `SPEAKER_DISABLE_REDACT=1`.
- visual near-dup: `memorize_phash_threshold` (default 4; only when
  `screen_capture.memorize` on).
- recall scoring: `recall_recency_weight` / `recall_recency_half_life_days` /
  `recall_importance_weight` (default 0 = off). **Single-backend** when recency on
  (backends stamp different wall-clocks).
- procedural memory: `procedural_enabled` (default false).

## Environment on i9-13980HX (Linux)
- `.venv` Python 3.12.11; full logic gate ~55s. No new runtime deps (Pillow is the
  existing optional OCR dep; the perceptual-hash test self-skips without it).
- PG-tier tests use the fake-pool harness. **Recommended before flipping any
  Postgres-tier flag on:** a real-pgvector validation pass (the new SQL —
  `role = ANY`, `rule::`/`NOT LIKE` profile filters — is standard but unverified
  against a live DB this session).

## Next steps (deferred, all scouted)
1. **Memory hybrid retrieval (RRF)** — fuse keyword/FTS with vector on the
   Postgres path (migration 005 pg_trgm) so exact tokens (names, error strings,
   proper nouns) recall reliably.
2. **Supersede-on-contradiction** — a bi-temporal migration + an off-hot-path
   ("sleep-time") LLM consolidation pass (Mem0 ADD/UPDATE/DELETE) so a new fact
   retires a contradicted one.
3. **CLIP/SigLIP visual recall** — image-embedding retrieval (text→image /
   image→image) reusing the float-BLOB store.
4. **Optional local injection classifier** (Llama Prompt Guard 2 22M) as a
   default-OFF defense-in-depth layer over untrusted blocks.
5. Minor: when BOTH `profile_enabled` + `procedural_enabled`, a "call me X" /
   "I prefer X" directive can land as both a profile fact and a rule (redundant,
   not harmful) — suppress one if it matters.
6. **Owner:** measure each flag (`tools.bench`) before enabling; rotate the leaked
   Gemini key + D1 history purge + speaker-ID enroll (carried).
