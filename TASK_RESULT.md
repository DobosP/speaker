Valid until: this branch lands or is superseded — then treat as history.

# Task result — task-scoped structured cancellation

## Summary

Added an actor-issued `TurnHandle` for every executable task binding. Its exact
five-field identity now follows coordinator, provider, mutating tool, TTS,
stream, and playback work. Cancellation fences new effects immediately;
caller-bounded drain reports any provider or sink ownership that is still
exiting instead of declaring the turn idle early.

Provider leases survive a canceled coordinator until the real provider thread
returns. Exact `binding_id` checks prevent stale terminal/TTS/playback cleanup
from touching a newer task that reused the same task ID. Terminal acceptance
also carries the supervisor-resolved identity to runtime subscribers; a
completion canceled before dispatch performs interrupted cleanup without
successful completion effects.

Synchronous task/provider thread-construction failures roll back their actor
and semaphore ownership. A queued-start failure preserves unvisited siblings,
and both public/core idle checks include actor task, provider, tool, playback,
pending-output, and pending-audio ownership. The text-only facade explicitly
terminalizes TTS events because it has no physical audio sink.

## Files changed

- Ownership/control: `always_on_agent/session_actor.py`, `tasks.py`,
  `supervisor.py`, and the public runtime facade.
- Output lifecycle: `core/runtime.py`, `playback_history.py`, and
  `tts_markup.py`.
- Tests: actor/task/provider races, preprocessing cancellation, queue rollback,
  terminal authentication, playback receipts, and idle/shutdown accounting.
- Durable docs: `STATUS.md` and ADR-0094.

## Verification

- Structured lifecycle matrix: 270 passed.
- Full `-m "not real_model"` repository gate: 6,439 passed, 14 skipped,
  22 deselected, and 9 pre-existing warnings.
- APM/double-talk, targeted Ruff, and `git diff --check`: passed.
- Independent audits found and closed terminal-identity, stale-completion,
  queued-sibling, text-facade, and bare-supervisor lifetime gaps.

## Risks and manual review

All evidence is deterministic and headless. No model, recording, live audio
device, STT-quality comparison, physical barge-in, or measured conversational
latency was exercised. A `TurnHandle` starts after task creation; capture,
VAD/STT/AEC, physical playback, and pre-task preprocessing remain outside it.
Started external side effects cannot be rolled back. Phone/remote ownership
parity remains future work.

## Merge recommendation

Green for fast-forward landing on `main`. Do not claim live hardware
validation.
