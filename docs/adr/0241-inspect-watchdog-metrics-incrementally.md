# ADR-0241: Inspect watchdog metrics incrementally

Date: 2026-10-10
Status: accepted

## Decision

Keep complete per-turn metric history and the existing stall predicates, warning
messages, deadlines, heartbeat/storm checks and maintenance callback. Replace
repeated full-history watchdog scans with a per-observer monotonic turn cursor,
unresolved-token set and current-turn snapshot. Reuse the ordered completed
record list for binary lookup, without a second permanent history index.

`TurnRecord.watchdog_anchors()` owns the shared existing LLM/TTS stall predicates.
New settled history is filtered before allocating stamp copies. Pending records
are refreshed under the recorder lock, including retrospective supersede/merge
markers; current records are always refreshed for later phase transitions. The
watchdog retains a closed unresolved record until each eligible phase warns or
is suppressed. Warned closed records are then retired from observer state.

Use monotonic turn tokens for warning deduplication. Recorder reset increments
an observer epoch and never reuses tokens, so a previous session's turn index
cannot suppress a new warning. Each observer owns its cursor and warning state;
no observer consumes another's evidence. Public `records()`, exports, EWMAs,
route decisions and task-reaping policy retain their existing semantics.

## Context / why

On baseline `1bed9e21df29c7885a88757b07e154e2ade972a3`, each one-second watchdog tick
copies and traverses every metric record since startup. The warning set also
retains keys for every stalled turn. That makes recurring monitoring work grow
with the length of an always-listening session, even when every old turn settled.
The root defect is repeated historical work, not model throughput or Python's
ability to execute native inference; a Rust rewrite would not remove that cost.

A first implementation copied all new historical stamps during late observer
attachment. The benchmark exposed a 20.5 MB initial allocation at 50,000 turns.
The accepted version shares the stall predicate and filters settled records before
copying; initial allocation is measured alongside steady-state allocation.

## Evidence and consequences

`python -m tools.bench_watchdog_history --baseline-ref 1bed9e2` executes the actual
pinned baseline and working source modules with public synthetic healthy metrics.
Paired order is reversed on the second repetition. The current compact receipt is
`docs/evidence/watchdog-history-2026-10-10.json` and binds both source hashes.

- At 1,000 settled turns, median repeated tick time was 349–366 us before and
  7.63–7.77 us after. Incremental traced peak: 8,232 → 624 bytes.
- At 50,000 settled turns, median repeated tick time was 20.37–22.06 ms before
  and 6.60–6.89 us after. Incremental traced peak: 400,232 → 624 bytes; initial
  discovery peak: 400,232 → 808 bytes.
- Complete metric history and each maintenance callback are retained. Initial
  discovery still examines newly seen records; unresolved turns remain until
  their warning obligations settle. This is not an absolute bound on history or
  on a burst of genuinely unresolved turns.

These are host-load-sensitive synthetic monitoring timings and Python allocation
measurements, not total RSS, inference throughput, end-to-end speech latency or
physical audio evidence. The recorder's API only adds retrospective suppression
markers to closed turns; direct external mutation of returned historical record
objects is outside the incremental observer contract.

The first source gate passed 113 tests covering existing watchdog/metrics/runtime
behavior, long history, late warnings on closed turns, independent phases, reset,
retrospective cancellation, separate observers, detached snapshots and exact
history preservation. Combined integration and physical limits are in STATUS.
No microphone, doctor, model download, native inference or live test ran.
