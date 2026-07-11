# ADR-0029: FileReplay null-sink playback receipts

Date: 2026-07-11
Status: accepted

## Decision
Make `FileReplayEngine` receipt-capable at its deterministic in-memory null sink.
Treat a successful positive-rate, non-empty numeric one-dimensional OfflineTts
result as one atomic exact start-and-completion boundary, but advertise no
sample counts: synthesis-domain samples are not acoustic output. Allow only one
generation at a time and drop concurrent tracked or legacy work instead of
queueing it past a cut. Fence admission with a monotonic playback generation
advanced by barge-in, shutdown, and restart. Claim an active ticket as
interrupted immediately on a cut, and prevent a late native return from
duplicating or upgrading it. Capture the admitted session's model and callbacks;
release the model lock and make completion immutable before invoking external
completion callbacks. Report full text only on completion and an empty prefix
for every failure, drop, or
interruption. Preserve legacy `on_done` as a compatibility callback, never as a
receipt.

## Context / why
Recorded-waveform and real-model sandboxes use FileReplay to exercise the same
recognizer and OfflineTts objects without audio hardware. The runtime could not
previously use those runs to validate sink-backed memory, resume, or follow-up
behavior because FileReplay's legacy `on_done` ran from an abort-inclusive
`finally`. It also had no sound-card callback from which to claim acoustic
playout. Fabricating output sample counts from the synthesized clip would confuse
model output with device admission, while adapting the legacy callback would
turn interruption and synthesis failure into false completion evidence.

## Consequences
Old-recording and real-model replay runs now exercise the same opt-in receipt
path as production without claiming live-speaker evidence. Successful null-sink
turns commit full assistant text; invalid, failed, stale, concurrent, cut, and
shutdown turns commit none. Nonblocking model admission and callback isolation
make barge-in terminalization bounded even when native synthesis is still
unwinding or callbacks reenter the engine. Deterministic race tests cover
admission, completion-versus-cut, repeated stops, callback failures, and restart
isolation. FileReplay still cannot validate audible output, echo cancellation,
fade quality, device recovery, or owner-mic talk-over; those remain manual
Sherpa/live-session gates.
