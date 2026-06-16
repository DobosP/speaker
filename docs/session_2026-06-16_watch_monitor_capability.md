# Session 2026-06-16 (pt2) — Watch/monitor capability + redactor PII fix

**Headline:** Built the north-star **watch/monitor for events** capability (a feature
the audit flagged as NOT built) as a security-first, default-deny **v1**, designed
and hardened entirely via multi-agent fan-out. Also fixed a **pre-existing PII leak**
in the shared redactor that the watch egress tests surfaced. All merged to `main`
(merge `7bc8aa5`, commit `8896cbc`). Full logic suite **1852 passed, 12 skipped**.

## What it is
The assistant can watch ONE **owner-granted** application window for a described
event and speak a single heads-up — **observe + speak only, never actuates**. The
owner grants a specific app, then asks the assistant to watch it for a condition; a
single shared poller observes only that window, checks the condition locally, and
fires one alert on the false→true transition.

- `core/watch.py` — `WatchGrant` / `GrantStore` (machine-local persistence) /
  `TextMatchEvaluator` (pure, local) / `WatchManager` (one shared poller) +
  `attach_watch_capability` registering `watch.grant` / `watch.start` / `watch.stop`
  (owner-verified, side-effecting) and `watch.list` (read-only).
- `core/watch_source.py` — the scoped `WindowResolver`/`WatchSource` capture seam
  (mirrors the AudioEngine seam), X11-only in v1, fake-able for Tier-0.
- Wired **default-OFF** in `core/runtime.py` + `core/app.py`; `config.json` ships
  `watch.enabled=false`, `grants:[]`; documented in `docs/unified_architecture.md` §8.

## How it was built (fan-out)
1. **Design** (`wf_731b4eda`): 2 recon agents mapped the exact seams + 3 design lenses
   (security/consent, architecture, capture/testability) → a lead-engineer synthesis
   into one file-level plan with the security invariants pinned to enforcement points.
2. **Implement**: I wrote the security-critical spine; 4 independent agents wrote one
   Tier-0 test file each (per invariant) + a doc agent (`wf_522cb76f`). 66 tests.
3. **Adversarial review** (`wf_6c2ef3d7`): 5 attacker agents tried to break each
   invariant. The core (default-deny, owner-verification, planner-isolation,
   anti-laundering, ephemerality) **held**; 4 real findings were fixed (below).

## Security model (enforced + tested)
- **Default-deny** — nothing watchable; a disabled config registers zero capabilities.
- **Owner-verified** — grant/start/stop go through `always_on_agent.origin`
  (`owner_verified is True` AND `LIVE_AUDIO`); `watch.list` is read-only/ungated.
- **LLM can't arm it** — all four caps are `planner_tool=False` + `user_facing=False`;
  an injected `TOOL watch.start` line is rejected (action not in planner tools).
- **Scoped + ephemeral** — one granted window, never full-screen; frames discarded in
  the poll tick (logger/memory/`set_current_frame` untouched).
- **Local + redacted** — pure local text match (no LLM/network); any text in a spoken
  alert is `redact_pii(force=True)` + spotlight-fenced.
- **Bounded** — single shared poller, `max_active` cap, min poll interval (clamped up).
- **Machine-local** — grants persist ONLY to gitignored `config.local.json`.

## Adversarial review findings (all fixed this commit)
| Sev | Finding | Fix |
|---|---|---|
| HIGH | `_fire` interpolated the owner `label`/`condition` **verbatim** (only OCR evidence was redacted) into the spoken alert → the git-committed `summary.json`. | Redact label + condition with `redact_pii(force=True)` too. |
| MED | `_X11Resolver` matched `wm_class` as a **substring** → a window classed `Signal-Phisher` satisfied a grant for `signal`. | `wm_class_matches`: EXACT case-insensitive identity. |
| MED | Owner `/regex/` condition + `title_pattern` ran **unbounded** on the shared poller (ReDoS, ~17s measured). | `safe_search`: reject nested-quantifier patterns + bound the haystack. |
| LOW | `GrantStore._persist` reset to `{}` on an unreadable local file → destroyed other machine-local keys. | Back up to `.corrupt` + abort the grant. |

## Bonus: pre-existing redactor PII leak (fixed)
The watch egress tests surfaced a real bug in the **shared** redactor
(`always_on_agent.untrusted`): `_CARD_RE` greedily spanned a card into an abutting
digit run (e.g. `card SSN` on one OCR row); the over-long match failed Luhn, so the
**raw card passed through** — affecting durable records, cloud egress, AND watch
alerts. `_redact_card_span` now recovers the Luhn-valid card window inside an
over-greedy match (group-aligned; favours recall for a §9.7 net). Regression-tested.

## Deferred hardening (noted, NOT blockers)
- Push owner-verify into `WatchManager`/`GrantStore` primitives (defense-in-depth;
  today the only callers are the gated capabilities).
- Pin a started watch to its resolved `win_id` for its lifetime (re-resolve by id).
- Hard-fail `_reconcile_capabilities` if a planner tool names a `planner_tool=False`/
  `side_effecting` capability.
- Wayland / macOS / Windows resolver backends (X11-only today).
- NL → grant/condition **intent extraction** — v1 reads structured args from the
  capability `context`; a dialog/intent layer to populate them from speech is next.

## Owner to-dos (carried)
- Enroll speaker-ID (`python -m core --enroll`) — gates `watch.grant`/`watch.start`
  like every action; until then they fail-closed/inert by design.
- Cloud default-OFF; rotate the leaked Gemini key + D1 history purge.

## Environment
`.venv` MINIMAL; bare `python` not on PATH — use **`.venv/bin/python`**. Heavy
ASR/torch not installed (~12 self-skips).
</content>
