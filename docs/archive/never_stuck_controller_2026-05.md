# Never-stuck controller (2026-05)

> ⚠️ **Superseded — durable content merged into [`docs/unified_architecture.md`](unified_architecture.md).** Kept for revision history; do not treat as current. (2026-06-02 consolidation.)

Goal (user): *the controller must be really smart about which processes to kill
and must not get stuck waiting for some answer/output.* The control plane could
cancel on barge-in/stop and (now) merge add-ons, but it had **no wall-clock
deadline anywhere** — a capability that blocks uninterruptibly inside a step (a
hung `generate`, a network read with no timeout) sat "active" forever, and the
watchdog only *diagnosed* a stall, it never *healed* it.

## What changed

- **Per-mode task deadlines + reap** (`always_on_agent/tasks.py` +
  `supervisor.py`). Each `AgentTask` gets a `deadline_at` stamped at
  `_start_task` from a per-mode budget (`DEFAULT_TASK_TIMEOUTS`: assistant 25 s,
  search 30 s, research 120 s, … — generous, so only a genuinely hung task is
  hit; config-overridable via the `task_timeouts` block; `0` disables a mode).
  `reap_overdue_tasks()` cancels and removes any active task past its deadline
  from `active_tasks`, so the supervisor stops treating it as live — the
  controller moves on even though the daemon worker may still be blocked (it
  exits when its own I/O finally returns, or leaks harmlessly as a daemon). A
  reaped turn that would have spoken says a short apology ("Sorry, that took too
  long — let's try again.") instead of leaving dead air.

  A long-running capability can push its own deadline out via a `renew_deadline`
  hook in the task context: the **ReAct planner** (which runs under the short
  ASSISTANT budget but legitimately makes several LLM + tool calls) renews to a
  multi-step agent budget on entry, so a real agent turn isn't killed mid-plan.

- **The watchdog heals, not just diagnoses** (`core/watchdog.py` +
  `core/runtime.py`). The watchdog gained an `on_tick` hook; the runtime wires it
  to `reap_overdue_tasks()`, so a hung task is killed on the watchdog's existing
  1 s cadence.

- **B3 fix — escalated turns aren't false-flagged "stuck"**
  (`always_on_agent/react.py`). An escalated (ReAct) turn does its LLM work
  inside the planner, which previously never stamped `LLM_FIRST_TOKEN`, so the
  watchdog read it as a stuck turn. The planner now fires a `first_token_hook`
  on its first streamed token; the runtime wires it to
  `metrics.mark(LLM_FIRST_TOKEN)` (idempotent), clearing the false positive.

## Concurrency

`reap_overdue_tasks` runs on the **watchdog thread**. It mutates `active_tasks`
under `_cancel_lock` (exactly like `cancel_all` from the audio thread) and
`cancel()` only sets an `Event`. `queued_tasks` is only ever touched on the bus
thread, so the reap **republishes a `TASK_CANCELLED`** per reaped task and lets
the bus thread drain the queue, rather than draining off-thread.

A reaped task whose worker later unblocks does **not** speak or remember its late
answer: `_run_plan` checks `cancel_event` after the step's `_invoke` returns and
drops the result (`_publish_cancelled`), and the streaming path stops at the next
token boundary. The reap deliberately does **not** bump the global speech epoch
(that would strand a concurrent sibling's TTS); a hung task has no queued audio
anyway.

Because the reap pops from `active_tasks` on the watchdog thread, the bus thread's
check-then-act reads of `active_tasks` (`_maybe_continue`, `_should_queue`) now
**snapshot once** — a reap popping the sole task between a `len()` and a
`next(iter())` would otherwise raise and, since `EventBus` ran handlers without a
guard, **kill the bus thread** (the whole assistant going dead). `EventBus` now
also wraps each handler in `try/except` (both the threaded and synchronous drain
paths), so any future handler bug degrades to a dropped event, never a dead bus.

## Scope / follow-ups

- A reaped turn currently ends in silence (logged + counted in `failures`); a
  spoken "sorry, that timed out" is a possible follow-up.
- "A new request supersedes the prior in-flight turn" (kill-and-replace for a
  genuinely new question, complementing continuation's merge) is a deliberate
  follow-up, not in this change.

## Tests

`tests/test_never_stuck.py` — reap cancels+removes an overdue task / leaves a
live one / republishes the cancellation; per-mode deadline stamping; `0`
disables; watchdog `on_tick` invoked + error-isolated; ReAct first-token hook
fires; and an **end-to-end** test where a hung LLM turn is reaped and the
controller returns to idle.
