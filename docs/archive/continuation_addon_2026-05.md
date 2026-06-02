# ADD-ON / continuation (2026-05)

> ⚠️ **Superseded — durable content merged into [`docs/unified_architecture.md`](unified_architecture.md).** Kept for revision history; do not treat as current. (2026-06-02 consolidation.)

When the user **adds on** to a request that is still being answered — "...and also
tomorrow", "oh wait, make it shorter", "what about Mars too" — the always-on loop
used to spawn a **second, competing task** that raced the first: two LLM
generations, two TTS streams, two overlapping spoken answers. This feature
detects the add-on and folds it into the in-flight turn instead, so the user
hears **one coherent response**.

## Where it lives

- `always_on_agent/continuation.py` — `HeuristicContinuationClassifier`
  (deterministic, local, no LLM) behind a `ContinuationClassifier` `Protocol`
  (so an optional local-LLM upgrade can drop in later), plus
  `ScriptedContinuationClassifier` (test fake) and `ContinuationConfig`.
- `always_on_agent/supervisor.py` — `_maybe_continue` is the routing brain,
  hooked into `_execute_decision` **after** the deterministic
  STOP/CONFIRM/DENY/MODE_SWITCH forks and only for `IntentKind.ASSISTANT`, so a
  real control phrase is never misread as a continuation.
- `always_on_agent/tasks.py` — `AgentTask.started_speaking` (set on the first
  streamed sentence) is the before/after-audio discriminator; `TASK_COMPLETED`
  now carries the task's start `epoch`.
- `core/runtime.py` — the input gate consults `supervisor.looks_like_continuation`
  so a short add-on isn't dropped as "ambient" while a turn is in flight.
- `core/capabilities.py` — `assistant()` skips the query ingest for a
  continuation's synthetic prompt (the raw add-on is recorded by the supervisor).
- Config: the `continuation` block in `config.json` (**enabled by default**).

## Behaviour

A follow-up is only treated as a continuation when **exactly one** assistant
turn is in flight and the classifier says `CONTINUE`. Then:

- **Before the turn has spoken** (`started_speaking == False`): cancel the
  not-yet-heard turn and answer **one merged prompt** rebuilt from the original
  ask + every add-on so far. `_cancel_one` bumps the (single-victim) speech epoch
  so any sentence the cancelled turn races out is dropped by the TTS staleness
  gate. One stream, one coherent reply.
- **After the turn is speaking** (`started_speaking == True`): don't cancel or
  re-speak it. Answer the add-on as a **context-carrying continuation queued
  strictly behind** the speaking turn (no race, no overlap); it starts when the
  turn completes. Multiple rapid add-ons **fold** into that one pending
  continuation rather than queueing several that would start in parallel.

Continuations track their `continuation_origin` + raw `continuation_addons` in
task metadata, so a chain of add-ons always rebuilds the prompt from
(origin + all add-ons) — never a nested template.

## Detection (deterministic, EN + RO)

`HeuristicContinuationClassifier.classify(addon, prev)`:

- **Strong markers** (multi-word / unambiguous add-on phrases: "and also", "what
  about", "make it", "as well", RO "de asemenea", "in schimb", ...): always
  `CONTINUE`, any length.
- **Weak markers** (bare conjunctions that also begin fresh questions: "and",
  "but", "or", "wait", RO "si", "sau", ...): `CONTINUE` only when the utterance
  is short (`<= addon_max_words`) **or** the cue is immediately followed by a
  modifier — so a long fresh question that merely opens with "and" is **not**
  swallowed.
- **Trailing** ("the forecast too") and **short modifier-led fragments** ("in
  spanish", "make it shorter"): bounded to short utterances.

`normalize_text` folds diacritics to ASCII, so accented Romanian input matches
the de-diacritic'd tables (also repairs the existing RO stop/mode phrases). A
miss degrades to today's behaviour (a clean second task), never worse.

## Scope / known limits (v1)

- A **correction after audio has started** ("make it shorter" while the assistant
  is already talking) cannot un-speak what was already voiced — there is no
  mid-stream token-injection / shared-KV-cache append on `LLMClient.stream`. The
  add-on is answered as a follow-up, not a re-render.
- Detection is heuristic; marker-less continuations the cue sets miss fall back
  to a separate task (no worse than before this feature).
- An optional LLM continuation classifier is a future upgrade via the
  `ContinuationClassifier` `Protocol`; it would stay off the §9.7 local path by
  default.

## Tests

`tests/test_continuation.py` (classifier: strong/weak bounds, trailing, RO
diacritics, empty-list override), `tests/test_always_on_agent.py` (supervisor
lifecycle: before-audio merge, after-audio queue-behind, chained fold,
superseded-completion drop, `looks_like_continuation`, prev/addon memory,
control-intent no-consult, disabled-by-default), `tests/test_core_runtime.py`
(end-to-end merge + input-gate override over the real worker/stream),
`tests/test_goal_alignment_fixes.py` (config pinned ON).
