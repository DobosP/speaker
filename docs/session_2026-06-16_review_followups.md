# Session 2026-06-16 — Post-crash recovery + P0→P3 review follow-ups

**Headline:** Verified the previous session's work survived the working-dir-wipe
crash intact (nothing lost), then ran an 18-agent adversarial review of the landed
**P0→P3 delegation-security** series and fixed the confirmed defects. The program is
**safe-as-shipped** (cloud OFF by default, actuation INERT until speaker-ID
enrollment, no exploitable bypass); one **MAJOR** correctness fail-open and five
minor hardening/coherence gaps are now closed. Full logic suite **1781 passed, 12
skipped** (+3 vs the 1778 baseline). Branch: `main`.

## Part 1 — crash recovery: nothing was lost
The earlier session's working dir was wiped mid-session; this box was **re-cloned**
(`reflog HEAD@{8}: clone`). Checked exhaustively:
- No outstanding feature branches (local or remote — only `main`).
- No stashes, no extra worktrees, no dangling/unreachable commits (`git fsck`).
- `local main == origin/main == 6064c2f` at session start; the full P0→P3 series
  (`2ee7987..6064c2f`, 7 commits) was already on `origin/main`.

So "all feature branches merged into main origin" was already satisfied — the only
working-tree noise was `logs/runs/` churn (left untouched).

## Part 2 — adversarial review of the P0→P3 changes
Reviewed range `7fe1341..020cd15` (P0 auto-RCE closure, P1a cloud-PII egress, P1b
owner-verified trust plumbing, P2 consent/provenance, P3 tool-call parsing, plus a
cross-cutting coherence pass). Each finding was adversarially verified against the
code before it counted. **Verdict: MOSTLY — coherent, default-safe, no exploitable
bypass; one MAJOR defect + several overstated claims.** All fixes landed in commit
`2815f65` (merge `74f0254`).

### Fixes landed
| ID | Sev | Fix |
|---|---|---|
| **P3-1** | **MAJOR** | `react.py` line-1 used the lenient colon-optional regex, so a bare single-line `"Tool web.search is great for code."` still parsed as a directive and fired a **spurious outbound web.search** — the exact string commit `020cd15` claimed to fix. Its regression test passed **vacuously** (it prefixed every prose case with a benign first line). Fixed: collapsed to a single **strict colon-required** scan over every line (the colon is the format `PLANNER_SYSTEM` tells the model to emit), dropped the dead lenient regexes, added bare single-line prose cases. |
| **P3-3** | minor | Parser now **skips fenced ``` code blocks**, so an *example* directive the model quotes can't fire a real tool/egress. |
| **XC1** | minor | The §9.7 cloud-egress PII scrub shared the `SPEAKER_DISABLE_REDACT` kill-switch with the durable-record redactor — disabling local-record scrubbing silently disabled cloud-egress scrubbing. `redact_pii()` gained `force=True`; egress uses it (mandatory net). |
| **XC2** | minor | The single-cloud back-compat path activated cloud egress with **no WARN** (multi-chain path warns). Mirrored the WARN — never silent. |
| **XC3** | minor | Trust keys (`owner_verified`/`origin`) were stamped **before** `context.update(extra_context)`, so a future caller could override the fail-closed verdict. Stamp them **after** — the trust seam stays authoritative. |
| **P0-1 / P0-2** | minor | `_NEVER_AUTO_SAFE` missed **direct** dangerous Python stdlib calls (`shutil.`/`os.remove`/`open()`/`urllib`/`requests`), `find -exec`, network CLIs, and the `--output` file-write flag (`git log --output=` → was auto-SAFE). Extended the set (zero effect on the shipped read-only allowlist) and **corrected the overstated "closes the bypass even under a broad misconfigured allowlist" claim** to its real scope and limits. |

### Deferred (noted, NOT defects)
- **P2-3** — `HedgeLLM.last_source` egress receipt is **correct-but-dormant**: no
  production code reads it and it isn't propagated through `SensitivityRouterLLM`
  (the multi-chain cloud path). If pursued, wire it to one log site **or** drop the
  "a caller can surface it" claim. Left as-is (latent, no false provenance).

## Owner to-dos (carried, unchanged)
1. **Enroll speaker-ID** (`python -m core --enroll`) — until then `--agent` /
   computer-use actuation stays fail-closed/inert (intended posture).
2. Wire `owner_verified=<SpeakerGate verdict>` into `AgentEvent.final` for action
   finals at the runtime seam (`runtime.py` omits it by design today). When you do,
   also enable the confirm-binding path (`supervisor.py`) or it stays inert.
3. Cloud stays default-OFF; enable deliberately per machine.
4. Rotate the leaked Gemini key; D1 history purge.

## Environment notes
- `.venv` MINIMAL (pytest, numpy, pytest-timeout, psycopg[binary], Pillow) — enough
  for the full logic gate. Heavy ASR/TTS/torch NOT installed (~12 self-skips). Bare
  `python` not on PATH — use **`.venv/bin/python`**.
</content>
</invoke>
