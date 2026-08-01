# ADR-0112: Add a headless conversation-flow acceptance gate

Date: 2026-08-01
Status: accepted

## Decision

Keep the 14-scenario conversation-eval v4 contract unchanged and add an opt-in,
deterministic flow-v1 extension through the same diagnostic entry point:
`python -m tools.conversation_eval --flow-acceptance`. Require the complete v4
set plus exactly three repetitions of four flow journeys: incremental typed
turns, scripted interruption recovery, delayed read-tool completion/follow-up,
and confirmed synthetic action stop/restart fencing. Build lifecycle journeys
with the production runtime assembler and `VoiceSession`; disable every external
integration and use only scripted text, a tracked sink, deterministic models,
and in-memory providers. Preserve nonempty task-runtime tool-call and
idempotency identities in raw traces, bind every result to one terminal trace
record, and fail closed on partial, duplicate, relabelled, noncausal, or invalid
evidence. Refuse real-model mode and local config for this combined headless
gate; keep model A/B as the separate v4 adoption contract.

## Context / why

The v4 evaluator covers post-ASR conversational semantics, tool trajectories,
stream interruption, receipts, cancellation, and quiescence, but it can be green
without exercising provisional-to-final turn handling, delayed tool ordering,
fresh-session lifecycle fencing, or confirmation-gated action ownership. The
previous live failures also showed that scripted transcripts must never be
presented as proof of microphone capture or recognition quality. A second app
entry point would fragment runtime composition, while changing v4 would break
the existing model-comparison contract. The extension therefore composes below
the current diagnostic CLI and leaves `python -m core --session` as the one
application entry point.

## Consequences

CI and local diagnostics can require an exact 4x3 production-control-plane flow
contract and receive a JSON report that explicitly labels fixture transcripts
as plumbing-only. The gate proves causal headless ordering, tracked scripted
playback, read-tool completion, confirmation/action identities, bounded stop,
and cross-session isolation. It does not cover native microphone/VAD wiring,
actual STT/WER, real TTS/audio pacing, AEC/room/open-speaker behavior, physical
audibility, external device effects, or live/perceived latency; those require
hash-bound public/private recordings and later isolated model/device A/B. The
synthetic action provider performs no mutation, so reminder and trusted-app
effects remain covered only by their focused tests until live validation.
