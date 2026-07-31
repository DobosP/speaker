# ADR-0094: Add task-scoped structured cancellation

Date: 2026-07-31
Status: accepted

## Decision

Make `SessionActor` issue one `TurnHandle` for every executable task binding.
Carry its immutable five-field identity, including a monotonic `binding_id`,
through task, provider, mutating-tool, TTS, and playback boundaries. Register
each child before it can start, fence new children immediately on cancellation,
retain exact ownership until every registered child exits, and expose only a
caller-bounded drain result that truthfully lists remaining owners. Authenticate
terminal events in the supervisor and pass the resolved identity plus
outward-current status to later runtime subscribers.

## Context / why

ADR-0086 bounded the number of post-recognition tasks and tools, but task
cancellation was still spread across mutable dictionaries, thread registries,
epochs, and task IDs. A coordinator could disappear while its provider thread
continued, completion could race cancellation before bus dispatch, failed
thread creation could lose queued siblings, and a stale terminal event could
clean up a newer task that reused the same ID. Idle checks consequently could
report success while a provider, tool, or playback handoff still existed.

A session-wide cancellation token is too coarse because continuations and
parallel tasks can share one acoustic turn. Unbounded joins are also unsuitable
for the capture/control plane: native providers may not cooperate promptly.
The task binding is therefore the smallest useful cancellation and drain scope,
while its actor is the single serialization point for child admission.

## Consequences

Cancellation is an admission fence, not a claim that native work has already
stopped. A detached provider remains counted until its real thread exits, and a
bounded drain can time out with an honest owner count. Started external side
effects are not rolled back; explicit tool cancellation and shutdown still
signal them according to their provider contract. An acknowledgement that
already reached an output sink before a synchronous start failure also cannot
be undone.

Exact binding identity prevents old task, tool, TTS, stream, and playback
cleanup from touching a same-ID replacement. Legacy identity-less in-process
terminal events remain supported only when the supervisor can resolve their
current binding. A completion canceled before dispatch performs interrupted
cleanup without successful-completion metrics, memory, cadence, or continuity.

Pre-task preprocessing, capture, VAD, STT, AEC, physical sink behavior, and the
later `MEMORY_COMMIT` bus dispatch are outside a `TurnHandle`. Mailbox priority
also does not promise that progress events precede cancellation. Remote/mobile
ownership parity remains roadmap work.

Verification is deterministic and headless. It proves bounded ownership,
identity, failure rollback, and cancellation races; it does not validate STT
quality, live conversational latency, open-speaker barge-in, or hardware.
