# ADR-0096: Introduce a publisher-bound LiveKit Agents adapter

Date: 2026-07-31
Status: accepted

## Decision

Add an optional, dependency-safe `LiveKitAgentsEngine` compatibility adapter
for exactly one explicitly bound remote publisher. It accepts only a narrow
Speaker-owned wrapper contract; raw `AgentSession`, `RoomIO`, and mutable
`AgentOutput` objects may not escape that wrapper. The wrapper must attest a
publisher-scoped, recording-off, local-inference configuration with no LLM,
Agent/session tools, MCP, or preemptive generation. Commit a whole user turn
only from `Agent.on_user_turn_completed`; never treat each streaming-STT final
segment as a conversation turn. Marshal session operations onto the job event loop,
retain generation-fenced exact-one playback receipts, and carry PII-free remote
acoustic lineage into the existing runtime. The wrapper atomically returns each
speech handle with an opaque output-route lease that becomes invalid after any
sink/tail replacement, enable transition, detach, or close; only a clean handle
and unchanged lease may attest the full text. An interruption-request failure
fences new work and requests session close without releasing playback ownership.
The SDK `close` event is authoritative only after its speech drain and output
detach, and resolves any impossible leftover lease with no safe text.

Treat participant identity as a transport binding only. Remote finals use the
explicit `remote_audio` label, which the action-origin gate deliberately coerces
to unknown/untrusted, and remain owner-unverified until a separate authenticated
provenance and speaker-policy decision is implemented. Keep the existing
LiveKit engine and entrypoint as the rollback path until the new composition is
wired and passes live A/B.

## Context / why

The current `LiveKitEngine` is a second hand-written session stack: it owns
room subscription, resampling, one shared recognizer/turn tracker, endpointing,
barge detection, synthesis, and output. Multiple participant tracks can feed
that shared state, while the web `/chat` path and Flutter app also run separate
answer loops. Replacing the trusted Python controller with framework tools
would lose stronger `SessionActor`, `TurnHandle`, confirmation, idempotency,
and playback-history guarantees. The safer seam is therefore a publisher-
scoped media adapter behind the established `AudioEngine` surface, with the
existing control plane unchanged.

The pinned LiveKit Agents 1.6.7 framework supports this split: an Agent may
observe completed user turns while `llm=None` and `tools=[]`, and `session.say`
returns an interruptible playback handle. A clean handle terminal can attest a
complete fragment only while the exact audio route remains continuously valid.
The public API has no route epoch: `replace_audio_tail()` can swap a downstream
sink without changing `output.audio`, and disable/re-enable is invisible to a
point-in-time check. Therefore the future concrete wrapper must own/version all
output mutation and expose atomic admission leases. The public
`playback_started` event also has no speech ID, so the adapter does not claim an
exact per-fragment start. Its adaptive interruption detector uses remote
inference and is excluded; proprietary/default inference models are also
excluded. SDK sink completion is still not proof that a remote human heard the
audio.

## Consequences

The new seam is import-safe without LiveKit Agents and is deterministically
tested with a fake trusted wrapper, job loop, transcript/state/close events,
speech handles, output leases, restart barriers, and failure races. It does not
yet provide that concrete wrapper or construct an `AgentSession`, so it is not
a selectable production engine. It does not change the shipped engine, improve
STT, reduce latency, alter tools, or claim live hardware validity. A later
branch must pin/install the optional SDK, implement the non-escaping wrapper
and audited local VAD/STT/TTS, bind `ctx.job.participant.identity` using
publisher-scoped jobs, pass `record=False`, keep all framework tool/LLM surfaces
empty, and preserve the one configured Speaker runtime. Another explicit
decision is required before changing the raw-audio residency boundary or
granting remote audio action authority. Client audible-playout acknowledgement
and bare-speaker live A/B remain promotion gates; the legacy adapter stays
available until then.
