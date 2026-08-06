# ADR-0149: Establish a fenced logical-turn lifecycle kernel

Date: 2026-08-06
Status: accepted

## Decision

Add a pure, inactive `always_on_agent.logical_turn.LogicalTurnCoordinator` as
the future single authority for the user turn between recognizer evidence and
the existing post-commit session actor. Native ASR epochs and finals are
evidence, never commits. One serialized coordinator owns one active logical
turn and emits explicit STARTED, INFERENCE_READY, COMMITTED, or ABORTED
transitions; only its commit result may release transcript text downstream.

Fence every mutation with a strictly increasing per-turn evidence sequence.
Bind commit to the exact readiness token and boundary reason that made the turn
ready, reject implicit finals after readiness and revisions from older native
epochs, cap native epochs/finals/retained text, and keep snapshots, transitions,
exceptions, and representations transcript-free. The coordinator owns no
thread, clock, I/O, model, or callback; its caller must serialize access and
provide explicit silence, semantic, hard-limit, control, and shutdown evidence.

Keep this first slice disconnected from runtime/configuration. Sherpa
endpointing, semantic reset, `FinalDispatcher`/`FinalCoalescer`, supervisor
continuation, entry points, models, thresholds, tools, and shipped defaults
remain unchanged. A later ADR must first add deterministic shadow/parity replay,
then separately authorize any runtime authority transfer.

## Context / why

The exact EdAcc endpoint work showed that semantic HOLD alone repaired zero of
five split rows and native reset-and-accumulate repaired only one, while p95
remained 1.7 seconds. Code inspection found three layers able to reinterpret a
turn boundary: Sherpa/native semantic handling, post-final coalescing, and
in-flight supervisor continuation. More STT/model tuning cannot establish an
exactly-once conversational commit while those authorities overlap.

Current open-source voice stacks converge on a central turn lifecycle with
replaceable evidence strategies. Pipecat separates user-turn start, inference
trigger, and stop strategies in its
[turn management](https://docs.pipecat.ai/api-reference/server/utilities/turn-management/filter-incomplete-turns);
LiveKit centralizes VAD, STT endpointing, semantic detection, interruption, and
commit in its [turn handling](https://docs.livekit.io/agents/logic/turns/).
Importing either framework would discard speaker's existing privacy, authority,
AEC, replay, and trusted-tool contracts, so adopt the lifecycle pattern rather
than a runtime dependency.

An immediate live cutover was rejected: the new lifecycle had no parity replay,
adapter event contract, shutdown proof, or owner hardware A/B. A model-only
experiment was also rejected as the first next step because it could improve
word accuracy without fixing duplicate or premature commits.

## Consequences

The project now has a small testable target for exactly one terminal per logical
turn, multi-epoch accumulation, stale-evidence rejection, and future
speculative-inference fencing. It is safe to exercise in shadow replay because
it has no effects and releases no text except through an explicit commit.

This change alone does not improve STT, endpoint latency, split-turn rate,
barge-in, tool behavior, or live conversation. It supplies no device, model,
GPU, latency, or owner evidence and cannot retire any existing finality layer.
The next slice must bind typed engine evidence to it in shadow mode, compare
event sequences and aggregate timing against legacy behavior, and prove abort,
late-final, resumed-speech, stop, and shutdown parity before an opt-in authority
path is considered.
