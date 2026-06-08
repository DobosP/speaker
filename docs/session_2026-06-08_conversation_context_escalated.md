# Session 2026-06-08g — Conversation thread on escalated/ReAct turns

**Branch → main:** `feat/conversation-context-to-escalated-path` → `main`
(@6b39e2e). Suite green (**1422 passed**, 14 skipped). Adversarial-reviewed: SHIP.

## The request
Owner: *"Enhance the memory of my application — the last sentences and the context
sent for the agent to know its capabilities and what is prior he spoke, with the
good version of the speech to text."*

## What was already there (so this was an enhancement, not a build)
The whole pipeline the owner described already exists and uses the **good** final:
- **Recent turns / "what was prior spoken"** → `core/conversation.py` builds a
  bounded `=== Recent conversation ===` block (last 6 turns / 800 chars) each turn,
  from the working-window memory.
- **Capabilities** → `core/persona.build_system_prompt` enumerates the agent's live
  registered skills (`render_skills` off the capability manifest).
- **Good STT** → the post-processed / agreement-guarded final (from the
  2026-06-08f cascade fix) is what gets stored as the `("user",)` turn — at **zero
  extra latency**. The optional LLM `cleanup` pass (one fast-tier call/turn) stays
  **off** by owner choice.

Owner answers (AskUserQuestion): keep cleanup **off**, keep depth **6/800**, keep
persona **anonymous**. So the enhancement was not a config toggle.

## The real gap fixed
The recent-conversation block was assembled **only in the one-shot answer path** of
`core/capabilities.assistant()` — *after* the escalation early-return. So an
**escalated** turn (the ReAct planner: "explain that step by step", "tell me more
about it") reached the planner with a **contextless query** and could not resolve
"that" / "it" / "the second one".

**Fix:** build the recent block **once, up front, before the escalation branch**,
publish it into `context["recent_conversation"]` (`RECENT_CONVERSATION_KEY`), and
have `always_on_agent/react.py` prepend it to **both** the plan prompt (resolve
references) and the final-answer prompt (keep the spoken thread). One-shot path is
**byte-identical** (same block, same float, same collect-before-ingest order, same
continuation suppression).

### §9.7 (preserved + tightened)
The build floats the turn's sensitivity to the most-private over **every included
prior turn** *before* `capability_context.set()` in the escalation branch — so a
private earlier turn (e.g. an SSN) can't pull the planner's nested LLM calls onto a
public/cloud chain now that the prior turns ride the planner prompt.

## Tests (+6; suite 1422)
`tests/test_capability_context_isolation.py`: escalated turn receives the block;
floats private sensitivity over a private prior turn; one-shot still composes it
into the system prompt. `tests/test_react_planner.py`: planner threads it into both
prompts; absent → prompts unchanged; **end-to-end contract pin** wiring the real
`attach_react_capability` + `attach_llm_capabilities` (guards the
constant↔literal coupling a future rename could silently break).

## Open follow-ups
- (P2, pre-existing, flagged in review) `web.search` gates egress on the raw tool
  `arg` via `may_leave_device(arg)` independent of the floated
  `context["sensitivity"]`. Widening the planner's view to prior turns gives it
  more material it *could* phrase into a search arg; `_is_personal` re-classifies
  that exact arg so literal PII still fails closed, but consider also gating the
  search arg on the turn's floated sensitivity. Orthogonal to this change.
- If the owner meant longer-term cross-session recall (not this-conversation
  short-term), that's `memory.context_for_llm` / `recall_enabled` (default off) —
  a separate lever.
