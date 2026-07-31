Valid until: this branch lands or is superseded — then treat as history.

# Task result — bounded post-ASR event mailbox

## Summary

Replaced the unbounded post-ASR `PriorityQueue` with a finite 512-event
mailbox. Data admission is capped below a 64-slot control reserve; typed
partial snapshots coalesce by complete acoustic lineage and strictly increasing
revision. Composite finals/aborts atomically retire every covered partial.

Non-output lifecycle/control work is preemptive. Ordered
`TTS_REQUEST`/`TTS_STREAM_END` traffic retains fragment-before-terminal order
and gains service after at most eight data dispatches, preventing cleanup
starvation. An all-critical overflow installs one durable `MAILBOX_FAULT`,
rejects later publications, cuts playback, reconciles discarded playback
commits, retires supervisor ownership, and stops the runtime off the bus thread.

Unscoped exact STOP/mode partials fail closed instead of acting after an
uncorrelated correction. The public text facade, replay path, and LiveKit engine
now provide scoped monotonic acoustic identity; LiveKit also publishes a typed
abort when an endpoint retracts to empty.

## Files changed

- Mailbox/schema/runtime: `always_on_agent/event_bus.py`, `events.py`,
  `supervisor.py`, `runtime.py`, `replay.py`, `core/runtime.py`, and
  `core/engines/livekit.py`.
- Tests: mailbox, supervisor facade, playback/preprocessing, and LiveKit
  regressions.
- Durable docs: `STATUS.md`, ADR-0093, agent map/testing, and unified
  architecture.

## Verification

- Mailbox: 57 passed.
- Mailbox/adjacent focused gate: 221 passed.
- Scoped LiveKit engine gate: 8 passed.
- Broader lineage/barge/playback/shutdown/follow-up/LiveKit gate: 107 passed.
- APM/double-talk gate: 6 passed.
- Full `-m "not real_model"` repository gate: 6,411 passed, 13 skipped,
  22 deselected, and 9 pre-existing warnings in 111.70 seconds.
- Targeted Ruff and `git diff --check`: passed.
- Three independent read-only blocker audits: clear after fixes.

## Risks and manual review

All evidence is deterministic and headless. No live audio device, physical
barge-in, real LiveKit room, or user recording was exercised. The eight-event
output service bound and scoped remote transcript identity are covered by
fakes/replay, not measured conversational latency. Phone and cross-participant
ownership parity remain future work.

## Merge recommendation

Green for fast-forward landing on `main`. Do not claim live hardware
validation.
