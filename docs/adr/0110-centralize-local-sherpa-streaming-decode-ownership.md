# ADR-0110: Centralize local Sherpa streaming-decode ownership

Date: 2026-08-01
Status: accepted

## Decision

Give each `SherpaOnnxEngine` capture run one synchronous
`SherpaStreamingDecodeSession`. Transfer the primary online recognizer into the
session immediately before capture-thread launch, bind the session once to the
exact capture-processor thread, and route creation, waveform admission,
readiness, decode, result, endpoint, reset, retirement, and close through opaque
session-created stream handles. Reject foreign-thread, re-entrant, stale,
forged, cross-session, and post-close operations before they reach native code.

Retain native streams after an opaque handle is garbage-collected until the
owner's next operation or close. A weak-reference callback may mark metadata,
but only the bound owner may remove the native reference. Close all surviving
streams before the recognizer on that same owner thread. Permit an unbound
session to be closed by the startup caller only when capture never claimed it,
and block restart while a prior decode owner remains live. Keep every unclosed
session in a process-private strong live-owner registry and unregister it only
after owner-thread close; a forgotten close must leak native state fail-closed
instead of letting garbage collection finalize it on an arbitrary thread.

Keep the complete session synchronous on the capture processor in this slice.
Do not enqueue individual native calls or streaming PCM in
`RealtimeMediaStage`, and do not change recognizer models, endpoint settings,
barge-in authority, enrollment, voice-agent capabilities/tool authority, entry
points, or audio defaults. Bind the session source into capture-replay
provenance and attest that replay used the single-owner synchronous path.

This decision covers the primary online recognizer in `SherpaOnnxEngine`.
Offline final recognition and keyword spotting have separate owners. File
replay and the rollback-only legacy LiveKit engine remain independent
single-threaded loops and must adopt their own complete session before either
is split across threads. The isolated benchmark adapter remains out of runtime
scope.

## Context / why

The local engine already happened to call its online recognizer from the
capture processor, but ownership was only a convention: the raw recognizer was
stored on the engine, native streams escaped into several helpers, transient
word-cut streams were released by ordinary reference drops, and restart relied
on implicit extension-object destruction. That made a partial decode offload
look safe even though reset, replay, result, endpoint, confirmation, and
word-cut calls would still share the same mutable native recognizer.

ADR-0107 requires exclusive ownership before moving online recognition behind
a media stage. A preserved prototype established the useful opaque-handle and
thread-affinity shape, but its weak-reference cleanup could destroy a native
stream on whichever thread collected or inspected the handle. The accepted
implementation keeps weak cleanup metadata-only and proves foreign collection
cannot release a native object.

The generic media stage's drop-oldest policy is also insufficient for a
contiguous streaming decoder: retaining later frames from the same scope after
one lost frame would silently splice audio. `CaptureMailbox` instead rotates
generation and flushes the stale queue. That whole-scope loss contract belongs
to the later asynchronous-owner slice, not this synchronous boundary.

## Consequences

Every primary local online-recognizer operation now has an executable exact-
thread assertion, native calls cannot overlap or re-enter, stream provenance is
session-bound, and native destruction is owner-thread deterministic. Existing
normal listening, VAD rebase, endpoint, confirmation, word-cut handoff,
hotword fallback, capture-gap reset, and callback ordering remain unchanged and
are exercised headlessly through the owner.

The boundary adds small lock and handle-dispatch overhead to each native call;
it does not itself reduce capture latency or improve WER. Capture replay proves
control/order behavior but not native-reader, device, AEC, or live acoustic
behavior. No live or model-backed result follows from this decision.

An unclosed session now remains resident for the process lifetime. This is an
intentional bounded-lifecycle failure signal, not automatic recovery: normal
capture, evaluator, and proven pre-launch-construction failures close
explicitly. An unobserved `Thread.start()` escape or missed close instead
retains native memory and prevents unsafe destructor-thread reuse.

Shutdown wakes only a live playback worker and drains any unconsumed private
wake item after the bounded join. A capture-launch rollback or idempotent stop
therefore cannot leave a sentinel that terminates a later playback run.

The next asynchronous extraction must move the complete session as ordered
transactions, carry exact capture scope plus stream incarnation, own immutable
PCM, rotate generation and retire the whole old scope on loss, fence callbacks,
and retain a timed-out owner so restart cannot alias native state. It must not
reuse `RealtimeMediaStage` drop-oldest semantics unchanged.
