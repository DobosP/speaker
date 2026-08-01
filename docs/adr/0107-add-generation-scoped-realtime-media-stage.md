# ADR-0107: Add a generation-scoped real-time media stage

Date: 2026-08-01
Status: accepted

## Decision

Use a typed, process-local `RealtimeMediaStage` as the route-neutral handoff for
bounded media work. Carry the existing `capture_epoch` and
`capture_generation` together in an immutable `CaptureScope`; do not create or
collapse them into a new global generation. Keep acoustic lineage/revision on
the stage payload and keep post-ASR `SessionActor` task cancellation unchanged.
Never serialize PCM or put it on the post-ASR `AgentEvent` bus.

Apply the first production use at Sherpa's optional async endpoint-final seam.
Copy endpoint PCM into stage-owned, read-only float32 arrays; bind the worker to
one exact single-run stage; and fence work both before expensive processing and
at callback admission. Keep a physical eight-item waiting bound. On saturation,
retire the oldest queued item with `BACKPRESSURE` and admit the newest without
waiting. Never fall back to heavy inline finalization. On close, atomically
cancel in-flight leases, release queued ownership, wake consumers, and reject
late offers. Record shutdown-terminal diagnostics when recording is enabled,
but preserve the existing capture-epoch rule that no external transcript
callback crosses engine stop. Execute no transcript callback while holding the
stage lock, and return retired items intact so the caller terminalizes before
recycling their payloads.

Keep recognizer choices, setup, `./live.sh`, `python -m core --session`, barge-in
authority, enrollment policy, and the unified tool plane unchanged.

## Context / why

`CaptureMediaSession` already prevents a slow processor from blocking the
native audio read, but Sherpa's downstream capture processor still owns DSP,
VAD, streaming recognition, endpointing, and controller work synchronously.
Slow processing therefore becomes bounded PCM retirement and degraded STT
rather than a blocked PortAudio read. The existing async-final queue is the
smallest production-used seam where lifecycle and cancellation can be proven
before extracting online recognition.

That queue had two unsafe overload/lifecycle behaviors. A race after saturation
could run the expensive finalizer inline on the capture processor, and shutdown
released queued PCM only when a diagnostic bundle existed. A raw sentinel queue
also did not expose capture generation, in-flight cancellation, exact ownership
transfer, or a restart-safe single-run identity.

Capture continuity, acoustic observation, runtime input arrival, and actor task
cancellation are different domains. Treating the numerically newest scope as
universally authoritative would incorrectly cancel an already endpointed final
that is still valid across a later capture continuity change.

## Consequences

Overload and shutdown are deterministic, bounded, transcript-free in
representations, and testable without models or devices. Endpoint handoff adds
one explicit PCM copy in exchange for immutable worker ownership. An in-flight
native decode remains cooperative rather than forcibly killable, but its late
effects are fenced by the stage cancellation event and existing capture epoch.

The stage establishes plumbing, not a latency, STT-quality, or adoption result.
The next media change must first give one `SherpaStreamingDecodeSession`
exclusive ownership of every online recognizer stream operation, then move that
complete owner behind the stage. Moving only the obvious `decode_stream` call
would introduce concurrent native recognizer access. Live A/B and GPU-backed
model comparisons remain separate evidence gates.
