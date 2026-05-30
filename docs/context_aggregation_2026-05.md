# Context aggregation: the model's short-term memory (2026-05)

Goal (user): get the correct input the model needs, aggregated from the previous
conversation. `assistant()` built the prompt from the **current utterance alone**,
so the answering model had no idea what was just said — "what's its population?"
after "what's the capital of France?" had no referent, and "make it shorter" /
"the second one" couldn't resolve. The only memory wired was *semantic recall*
(`recall_enabled`, default off), which surfaces snippets from **past sessions**,
not the immediate turns.

## What changed

- **`core/conversation.py` (new)** — `build_recent_context(memory, config)`
  assembles a compact `=== Recent conversation (most recent last) ===` block from
  the last few user/assistant turns in the working-window memory
  (`memory.all()`, filtered to the `user` / `assistant_output` tags; ambient,
  ingested, and meeting items excluded). Bounded by `max_turns` (6), a per-turn
  char cap, and a hard `max_chars` (800, dropping oldest turns until it fits).
  Best-effort: any error → `""`.
- **`core/capabilities.py` `assistant()`** — collects the prior turns **before**
  ingesting the current query, then composes `system_for_call = [recall] + system
  + [recent turns]`. **Empty memory → exactly the base system** (first turn /
  fresh session unchanged). The model resolves the references itself — subsuming a
  separate coreference-resolution pass.
  - **Ordering (KV-cache):** the stable system prompt comes **first** (it's what
    the pre-warm warms, so the cacheable prefix is reused turn to turn) and the
    volatile recent block is appended **after**. Recall (default-off) keeps its
    historical position ahead of system so its contract is unchanged.
  - **§9.7 privacy float:** a private prior turn in the recent block must not ride
    a *public* current query to a public/cloud chain. So the prompt's sensitivity
    is floated to the **most-private** over the current turn AND every included
    prior turn (`core/sensitivity.most_sensitive`). Consequence: once anything
    private is in the recent window, following turns route conservatively
    (local/private) until it ages out — the right §9.7 trade-off.
  - **Excluded:** ambient/ingested/meeting items, and placeholder/abstain replies
    ("Sorry, I don't have an answer…", the timeout apology) — they carry no
    conversational content. **Continuation turns** suppress the block entirely
    (their merge/continue prompt already embeds the prior context).
- **Config** (`RecentContextConfig`, the flat `memory` block) threaded through
  `runtime` / `app`. **Default ON** — short-term context is the headline fix and
  is cheap (no embeddings, unlike `recall_*`).

## Notes

- Distinct from `recall_*`: recall is semantic, past-session, default-off; this is
  chronological, this-conversation, default-on.
- The pre-warm prefills `build_system_prompt` (identity + skills), not this
  per-turn block, so the recent-context prefix is assembled fresh each turn.
- A future refinement is proper multi-turn chat *messages* (role-tagged) through
  the `LLMClient` protocol; the block-in-system-prompt form here gets the value
  with a much smaller surface area.

## Tests

`tests/test_conversation.py` — builder (turn order, tag filtering, empty /
disabled / none, `max_turns`, `max_chars` drop-oldest, per-turn truncation,
`from_dict`) and the `assistant()` injection end-to-end (turn 2 sees turn 1; base
system stays at the end; disabled → unchanged).
