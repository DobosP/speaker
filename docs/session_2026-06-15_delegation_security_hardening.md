# Session 2026-06-15 — Delegation security hardening (P0→P3) + audit

**Headline:** Following the delegation security audit (is this local agent safe +
smart enough to delegate to apps / cloud LLMs?), landed the full **P0→P3**
remediation program as separate adversarially-reviewed commits on `main`. All
default-OFF / default-byte-identical. Full logic suite **1775 passed, 12 skipped**.

> ⚠️ **Environment incident (recovered):** mid-session the working directory
> `/home/dobo/work/speaker` was deleted by something external (not by me; disk was
> 46% full, not a space issue). Committed work was safe on `origin/main`; I
> re-cloned, rebuilt the venv (bootstrapped pip via get-pip.py — `python3-venv`
> isn't installed; restored git identity + SSH known_hosts), and redid the one
> uncommitted slice (P0) from context. Only that staged P0 work was ever at risk;
> nothing committed was lost. The logic suite runs without the heavy ASR/torch
> stack (core modules degrade gracefully), so ~12 tests self-skip vs the full env.

## The audit (assessment, no code change)
Verdict, grounded in code:
- **Cloud delegation (thinking): reasonably safe** — conservative-by-default (cloud
  OFF in base config + desktop profiles; raw audio/STT/TTS never leave §9.7; a
  fail-safe-to-PRIVATE classifier + most-sensitive-wins float). Residual: regex-only
  classifier, no outbound PII scrub, 3 profiles auto-enabled cloud on key-presence.
- **App delegation: bimodal** — the read-only computer-use foundation is safe; the
  opt-in Open Interpreter brain had a **verified auto-RCE** (`python -c` auto-ran
  unconfirmed) and the new action chokepoint gated nothing on that path.
- **Smart enough?** Yes as a **router/answerer** (heuristic-first, small LLM only
  disambiguates low-confidence with a one-word reply + fallback); **no as an
  autonomous multi-step actuator** (BFCL agentic ~17–29% for 7–14B) — which is the
  division of labor the design already encodes.
- **Prior art:** Apple Intelligence+PCC, DeepMind CaMeL, goose, Home Assistant,
  RouteLLM/FrugalGPT — the app matches the good patterns.

## What landed (commit map, all pushed to origin/main)

| Commit | Slice |
|---|---|
| `2ee7987` | **P0** — close OI auto-RCE + wire owner-verified action chokepoint |
| `cd08f15` | **P1a** — outbound-cloud PII redactor + no CN in default chains |
| `8ece346` | **P1b** — owner-verified action plumbing + bound/gated confirm |
| `4583e10` | **P1 review fixes** — close trust-laundering paths + single-cloud redaction hole |
| `b1dae48` | **P2** — cloud consent (no silent activation) + egress receipt + ACT containment |
| `c83db7e` | **P3** — robust tool-call parsing + optional router model |
| `020cd15` | **P2/P3 review fixes** — parser false-directives + payload corruption + receipt/docs |

Every slice was adversarially reviewed (multi-dimension → verify each finding);
all confirmed findings fixed. Notable review catches: P1's partial-"yes" trust
laundering (blocker) and P3's parser misreading prose like "Tool web.search is
great" as a directive + `str.translate` corrupting payloads ("C# tutorials").

### P0 — OI auto-RCE + action chokepoint
Curated `agent_brain.allowlist` to no-effect status commands only (removed
`python -c`/cat/head/tail/echo); a built-in `_NEVER_AUTO_SAFE` set downgrades any
code-exec/chaining/redirect/newline to needs-confirm even under a broad allowlist;
`require_owner_verified` (default ON) makes `command.stage` refuse unless the turn
is owner-verified live audio (fail-closed → `--agent` actuation is INERT until
speaker-ID is wired). `--agent` is opt-in/default-OFF regardless.

### P1a — cloud PII redaction + jurisdiction
`_redact_messages_for_egress` scrubs high-confidence PII from **outbound cloud**
prompts only (a §9.7 net independent of the regex classifier; local models never
touched); removed the CN provider from the default public chain (now US-only even
with `allow_prc`).

### P1b — owner-verified plumbing + confirm
Threads a turn's speaker-ID trust (`owner_verified`/`origin`, fail-closed defaults)
event → supervisor → task → capability context, so the P0 chokepoint becomes
operative on enrollment. `_confirm_next` reads back the specific action and requires
an owner-verified "yes" to approve an owner-verified staged action.
*Review (1 blocker + 4 major, all fixed):* partial-"yes" trust laundering (analyzer
now requires `is_final` for CONFIRM/DENY), followup + FOLD-continuation trust
laundering (fail-close / demote), and the single-cloud back-compat path silently
skipping the PII scrub.

### P2 — consent + provenance + containment
Flipped the 3 cloud-enabled profiles to `enabled:false` (no silent activation on
key-presence) + a WARN when cloud activates; `HedgeLLM.last_source` egress receipt;
ACT-containment regression tests (planner tools stay read-only).

### P3 — orchestration ceiling (capability)
`_parse_step` scans all lines + strips markdown + a bounded 1× strict-format
re-prompt (lightweight grammar-constrained-decoding stand-in); optional
`llm.router_model` for a function-calling-tuned LOCAL router model (default unset →
fast tier). **Deferred (owner-validated, need real models):** swap in xLAM/Qwen +
measure; a FrugalGPT-style quality/verifier gate on HedgeLLM.

## Owner to-dos (the security story is load-bearing on these)
1. **Enroll speaker-ID** (`python -m core --enroll`) — until then `--agent` /
   computer-use actuation stays fail-closed/inert (the intended posture).
2. Wire the runtime to pass `owner_verified=<SpeakerGate verdict>` to
   `AgentEvent.final` for action finals (needs the turn's audio at the final seam +
   live validation; documented in P1b).
3. Cloud stays default-OFF; enable deliberately per machine when wanted.
4. Carried: rotate the leaked Gemini key, D1 history purge.

## Environment notes for next session
- `.venv` rebuilt with a MINIMAL set (pytest, numpy, pytest-timeout, psycopg[binary],
  Pillow) — enough for the full logic gate (`-m "not real_model and not recorded and
  not live_output"`). The heavy ASR/TTS/torch stack is NOT installed; install
  `requirements.txt` for the real_model/live tiers. `python3-venv` missing on the
  box (used get-pip.py); bare `python` not on PATH — use `.venv/bin/python`.
- Git identity was reset by the re-clone; set repo-locally to Dobos Paul /
  pauldobos6@gmail.com.
