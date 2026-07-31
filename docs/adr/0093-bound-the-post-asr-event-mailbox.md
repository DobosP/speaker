# ADR-0093: Bound the post-ASR event mailbox

Date: 2026-07-31
Status: accepted

## Decision

Replace the Python control plane's unbounded `PriorityQueue` with a finite
512-event mailbox. Reserve 64 slots from data admission for control,
action-producing, and lifecycle-terminal events; critical events may use all
otherwise-empty slots and, at total saturation, evict data but never another
critical event. Keep publication nonblocking. Derive admission and dispatch
class from an exhaustive `EventKind` policy, not from numeric priority alone.
Dispatch non-output control/lifecycle work before data, preserving
`(numeric priority, publication sequence)` inside that preemptive class. Then
use global priority for data and ordered output, except that the oldest
`TTS_REQUEST`/`TTS_STREAM_END` is promoted over data after eight dispatches.
Publication FIFO within that output class preserves every fragment before its
stream terminal while preventing a partial storm from stranding closure.

Classify `STT_FINAL`, `STT_ABORTED`, every `CONTROL_*`, task starts and
terminals, `TTS_REQUEST`, `TTS_STREAM_END`, and `MEMORY_COMMIT` as critical. A
fatal `CAPTURE_STATE` is also critical. Treat observations, decisions, task
progress, follow-up ticks, and nonfatal capture state as bounded data. Treat an
exact STOP/mode partial as preemptive only when it has valid acoustic identity;
reject an unscoped or malformed actionable partial. Text-only callers must wait
for a final or publish explicit STOP/mode control. Partial confirm/deny remains
non-actionable data. Every scoped `STT_PARTIAL` is a supersedable snapshot keyed
by the complete ordered acoustic-span identity and a nonnegative revision.
Accept only a strictly newer revision and give the replacement its real incoming
publication sequence. A matching newer partial, final, or abort retires the
older snapshot, including an undelivered exact-control hypothesis, while
unrelated acoustic turns remain. A terminal that is not strictly newer than its
queued snapshot is rejected at this last serialized boundary. A merged terminal
retires snapshots for every constituent span it covers; partial-to-partial
replacement still requires the complete ordered lineage.

At data pressure, retain more urgent data and deterministically evict the
oldest among the least urgent eligible events; reject a less-urgent data
arrival when every retained data event is more urgent. Emit only aggregate,
transcript-free pressure, unscoped-control rejection, fairness, and fault
counters. If the mailbox contains only critical events and is full, atomically
discard queued work, latch the bus against new publication, and install one
highest-priority typed `MAILBOX_FAULT` carrying aggregate discarded-kind
counts. The supervisor directly retires actor/task ownership; the runtime cuts
output and starts full teardown off the bus thread. Expose admission results
and aggregate outstanding work so receipt bookkeeping and tests do not depend
on a queue implementation.

## Context / why

The prior bus was intentionally nonblocking near capture and playback, but its
512-event high-water mark only logged a warning while storage remained
unbounded. A stalled subscriber or producer storm could therefore grow memory
without limit after the audio capture path itself had already been bounded.
This also hid the difference between provisional transcript state and events
whose loss can strand task, playback, memory, or authority ownership.

The open-source audit did not find a bounded core mailbox to copy. Pipecat's
typed urgent/data/control/uninterruptible frame taxonomy and LiveKit's explicit
channel-full behavior were useful concepts, but the audited STT/voice queues
were unbounded. Pipecat's immediate system-frame handling was also unsuitable:
speaker subscribers are a single ordered state machine, and synchronous
control dispatch on an audio or worker publisher could re-enter supervisor
locks. Speaker additionally relies on priority across event classes:
`TASK_COMPLETED` must overtake queued `TTS_REQUEST` fragments, while
`TTS_STREAM_END` must remain after them. Separate FIFO dispatch lanes were
therefore rejected. Strict global priority was also insufficient: a
deterministic one-partial-per-dispatch replay left priority-110 stream closure
pending forever without filling the mailbox. Targeted output aging solves that
case without letting old fragments overtake a final, task terminal, memory
commit, fatal capture event, or direct control.

Blanket latest-partial coalescing by event kind was rejected. `LiveSpeechAnalyzer`
permits exact STOP or mode phrases to act on partials, but a later typed
correction from the same acoustic turn must supersede an undelivered hypothesis;
otherwise priority 50 could dispatch the correcting final before a stale
priority-90 STOP. Identity-less legacy/remote partials cannot be safely merged
across participants, so allowing them to act provisionally was rejected. The
text facade and replay now mint single-source opaque lineage, and LiveKit
prefers typed callbacks backed by a remote-track turn counter. A legacy
text-only callback fallback must wait for a final or use explicit barge/control.
The existing exact-control classifier, immutable acoustic span keys, and
monotonic revisions provide the smallest safe seam.

A finite, nonblocking mailbox cannot guarantee lossless admission for an
unlimited flood of distinct critical events. Upstream task, playback, and
provider admission remain the normal bound. The last-resort mailbox fault
deliberately stops the inconsistent runtime instead of letting a swallowed
publisher exception strand task, playback, or memory ownership. It requires
restart and investigation; it is not a normal shedding or recovery mechanism.

## Consequences

Pending event storage is physically bounded, and a data storm cannot consume
critical reserve. Newer acoustic partial snapshots replace older ones without
heap tombstones, stale revisions cannot roll state backward, and a terminal
cannot dispatch ahead of its own obsolete partial and leave that partial to run
afterward. Preemptive work is not delayed by aged output; ordered output cannot
be starved by discardable data. Handler order, exception isolation, exact-one
`drain_once`, recursive publication, in-flight idle accounting, threaded
dispatch, and shutdown's default no-drain behavior remain unchanged.

Under pressure, diagnostic/data events may be absent from traces; counters make
that visible without logging transcript text. An all-critical fault reports the
discarded event kinds, rejects later playback commits without leaking their
pending count, reconciles accepted commits discarded by the fault, and performs
terminal cleanup rather than retrying effects. The mailbox bounds event count,
not payload byte size, so providers must retain their existing bounded result
and playback contracts.

This is the first runtime architecture increment after ADR-0092. It does not
add structured turn cancellation, change mobile/remote ownership, improve STT
accuracy, or claim faster physical STOP. Physical barge-in still performs its
low-latency cancellation and speaker cut before publishing the bus
acknowledgement. The next isolated change is an actor-backed `TurnHandle`
cancel-and-drain boundary, followed by shared local/remote/mobile ownership.
