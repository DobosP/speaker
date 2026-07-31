# ADR-0086: Bound post-recognition session ownership

Date: 2026-07-31
Status: accepted

## Decision

Give each Python `AgentSupervisor` one device-neutral `SessionActor` in
`always_on_agent/`. Keep it synchronous and lock-protected: its transitions do
no audio, model, provider, or playback work. After a final wins ingress, the
actor allocates a session-local turn and binds the accepted acoustic revision.
Every executable `AgentTask` then carries an immutable `TaskIdentity` containing
`session_id`, `turn_id`, `revision`, and the current speech-cancellation
generation. Carry that identity on task lifecycle and production TTS events and
into receipt-backed playback context.

Bound actor admission independently for active tasks, mutating tools, playback
handoffs, remembered task identities, and session idempotency keys. Treat every
capability whose `CapabilitySpec.side_effecting` is true as a mutating tool.
Remembered task bindings remain externally owned until the controller
explicitly releases them; capacity reclamation may never infer terminality from
the absence of actor-active work. Reject a new binding when no explicitly
released entry is reclaimable. Queue-drain epoch failure explicitly retires
both already-deferred and unvisited snapshot tasks.

Before its provider starts, allocate a generic `tool_call_id` and stable
turn/step `idempotency_key`, pass both through capability context and invocation
observation, allow its provider-start guard to be claimed exactly once, and
reject a repeated key before provider execution. Existing durable tool backends
may retain their own longer-lived idempotency records; the actor's generic
record is deliberately bounded and session-local.

Make speech and tool cancellation separate ownership domains. A barge, stop,
or superseding turn advances the actor's speech generation, invalidating stale
TTS and new playback admission. Under the same actor lock, it cancels mutating
admissions that have not claimed provider start. If provider start wins first,
speech interruption cancels only the surrounding task/output; only
`cancel_tool_call(tool_call_id)` explicitly cancels that started tool. Retain
its idempotency record after speech cancellation and provider completion so
reconstructing the same turn/step cannot execute the mutation twice. A
watchdog timeout atomically wins the task terminal and retires its actor binding
under task-lock then actor-lock, rejecting future tool admission and cancelling
unstarted calls; a provider or completed task that won first keeps its existing
policy. Its spoken apology is optional: identity allocation/publication failure
never suppresses the authoritative cancellation lifecycle or queue drain.
Terminal session shutdown fences every new admission/start claim, sets every
tool Event, and boundedly drains cooperative calls before capability-owned
watch, reminder, or memory dependencies close. Serialize the full runtime stop
so concurrent callers cannot close those backends ahead of the drain.

Reserve one actor playback admission immediately before the output-sink
handoff, requiring the four identity fields to match the actor's exact
`task_id` binding; matching session and cancellation generation alone is not
authority. If any identity field is present, all four must parse and exactly
match; preserve malformed/partial lifecycle identity into TTS so the legacy
adapter applies only when all four fields were truly absent. One ownership
guard retains admission through preparation and legacy `speak()`, or transfers
it durably to receipt history. Track its exact task owner, not equality of
same-turn identities. Validate receipt outcome, safe-prefix, and sample field
types before fragment mutation; staging exceptions and malformed receipts
force-terminalize ledger entries and release actor capacity. Production
auxiliary speech follows the same rule and retains its actor binding; legacy
direct in-process TTS is adapted to a synthetic actor-owned identity.

Keep capture, VAD, endpointing, recognition, and acoustic revision rejection
outside the actor as thin producers. Keep the existing event bus and audio
queues unchanged in this slice. This completes the post-recognition ownership
step anticipated by ADR-0084; it does not supersede typed ingress.

## Context / why

Typed acoustic lineage previously stopped at `STT_FINAL`. Tasks used a task ID
plus several unrelated generations, TTS used a speech epoch, playback used a
task/epoch pair, and mutating providers inherited the same cancellation event
as disposable speech generation. A barge could therefore make an admitted
mutation's result disappear while a retry lacked one generic identity proving
that the action had already run.

Increasing timeouts or relying only on a reminder/app-specific database key
does not solve this ownership problem. It leaves other mutating capabilities
unprotected and cannot distinguish “stop talking” from “cancel this action.”
Moving recognition into the owner would solve a different problem while
blocking device capture loops on control-plane work.

## Consequences

- Deterministic tests can follow one acoustic revision through task, TTS, and
  playback admission without opening an audio device.
- Active tasks, mutating calls, and output handoffs have explicit ceilings; an
  uncooperative provider-started tool remains covered by both the tool ceiling
  and existing provider bulkhead until it actually exits.
- Speech interruption cannot launch or replay a stale mutation. It may suppress
  conversational output from a provider-started mutation that legitimately
  completes, so tool state—not spoken text—remains authoritative.
- Generic duplicate admission currently reports a controlled task failure
  rather than caching arbitrary provider result data. Persistent tools can
  still return richer idempotent replay state from their own stores.
- The compatibility `speech_epoch` mirrors the actor generation while older
  tests and adapters migrate to explicit `TaskIdentity`.
- Session idempotency is bounded, in-memory, and reset on process restart.
  Durable exactly-once behavior still belongs in a mutating tool's backend.
- Shutdown waits only a short bound. A cooperative provider observes its tool
  Event before watch, reminder, or memory dependency close; a native call that
  ignores cancellation is logged honestly and cannot be force-killed by Python.
  Concurrent stop callers wait for that full lifecycle. The trusted-app manager
  owns no separate close lifecycle.
- The central event bus is still unbounded, and Dart/mobile/remote brains do
  not yet implement this actor contract.
- Headless tests do not validate microphone capture, STT accuracy, echo,
  natural interruption timing, multiple voices, enrollment, sink drivers, or
  physical tool cancellation. A fresh `./live.sh` A/B remains required.
