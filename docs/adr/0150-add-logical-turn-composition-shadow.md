# ADR-0150: Add opt-in logical-turn composition shadow replay

Date: 2026-08-06
Status: accepted

## Decision

Add a serialized, fail-contained `LogicalTurnCompositionShadow` that feeds the
pure ADR-0149 coordinator exactly one terminal result from each one-based
Sherpa native stream epoch. Record whether each epoch was held for a successor
reset or was terminal, admit explicit empty successor epochs, and compare the
coordinator commit in memory with the existing composed raw Sherpa final. Make
that comparison immediately after legacy composition and before speech
admission, offline final selection, dispatcher publication, supervisor
continuation, or task creation.

Expose the observer only through the synchronous capture-replay/test call to
the private `_capture_loop`; do not add configuration or bind it to the
production decode-owner path. `tools.capture_replay_eval --logical-turn-shadow`
creates one fresh observer per case/repeat and changes that opt-in aggregate
report to schema 2 by adding a strict shadow section. With the flag absent, keep
the established schema-1 key shape and execution path. Bind the coordinator,
shadow observer, and stop-command classifier into source provenance only when
the flag is active.

Reduce transcript comparison immediately to closed counters. Snapshots and
reports may contain no text, text hashes or lengths, acoustic identities,
paths, exception strings, or event rows. Report exact raw-composition parity
only when health counters are clear, at least one paired multi-epoch commit was
observed, and there are no composition mismatches or extra/missing legacy
terminals. Treat `coverage.complete` only as proof that all requested observer
runs closed; it is not a per-run terminal denominator.

Map only closed endpoint evidence to native-final, semantic-complete,
silence-limit, or hard-limit counters. Replay callback aborts and exact stop
commands may terminalize an active shadow turn, but explicitly disclaim abort,
async-final, selected-final, dispatcher, supervisor, runtime stop/shutdown,
production-configuration, runtime-authority, and live parity. Observer failure
must never change existing capture callbacks or finals.

## Context / why

ADR-0149 deliberately left the new lifecycle kernel inactive and required
deterministic shadow/parity evidence before any authority transfer. The prior
EdAcc result repaired only one of five split rows, while several later layers
can still reject, rewrite, or reinterpret a raw composed final. Comparing at a
later callback would conflate native composition with admission, model
selection, async dispatch, and supervisor behavior; invoking a synthetic
observer directly would merely compare the coordinator with its own inputs.

The synchronous replay seam executes the real Sherpa endpoint and legacy
composition logic without an audio device and keeps callbacks on the same
thread. It can therefore test the narrow first boundary without introducing a
second runtime owner. A production shadow, global abort hook, default report
schema change, and immediate lifecycle cutover were rejected because replay
does not establish lineage-safe async abort pairing, device timing, or owner
conversation behavior.

## Consequences

Headless replay can now expose split native epochs, empty reset epochs, typed
boundaries, lifecycle closure, and exact raw-composition mismatches without
persisting transcripts. Strict counter algebra, exact JSON scalar types,
bounded state, hostile-string rejection, close races, initialization cleanup,
default-path isolation, and post-composition rejection ordering are covered.

This adds diagnostic visibility, not evidence that the new coordinator should
own live turns. It does not improve STT, endpoint latency, barge-in, tool use,
or conversation by itself, and supplies no model, GPU, microphone, acoustic,
device, latency, or owner validation. The production decode owner, public
entry point, models, tools, thresholds, persisted configuration, and shipped
defaults remain unchanged. A later decision must add lineage-correlated parity
for later finality layers and obtain matched owner `./live.sh` evidence before
any runtime authority can move.
