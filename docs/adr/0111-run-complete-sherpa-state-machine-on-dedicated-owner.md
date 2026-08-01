# ADR-0111: Run the complete Sherpa state machine on a dedicated owner

Date: 2026-08-01
Status: accepted

## Decision

Use the existing `CaptureMediaSession`/`CaptureMailbox` as the sole bounded
reader-to-processor PCM stage for the default local Sherpa path. Bind one
`SherpaStreamingDecodeOwner` thread to the run-scoped synchronous session and
run the complete ordered capture state machine on that thread: DSP, VAD,
primary streaming ASR, playback confirmation, word-cut ASR and replay,
endpoint decisions, and capture-loop callbacks. Do not add a third PCM queue.

Keep raw-capture scope, decoder continuity, and role-specific stream
incarnation distinct. A capture gap cancels all queued PCM plus an unclaimed
in-flight block, flushes the old scope, fences callbacks, retires both primary
and word-cut streams, rotates `CaptureScope`, and creates a new primary stream.
A decoder-native failure preserves capture scope but rotates decoder
continuity, retires every live role, resets decode front ends, and creates a
new primary stream. Publish each new identity before potentially blocking
native retirement.

Make mailbox PCM and reader-time reference snapshots bytes-backed and
irreversibly read-only. Give each block an effect lease: a gap may cancel an
unclaimed block, while an already-claimed exact-thread effect may finish and
must release before teardown. Permit one exact in-flight consumer and reject a
second take, foreign finish, stale finish, duplicate live role, reused capture
scope, foreign native call, or callback carrying an old decode identity.

Treat all post-input worker launches as one rollback transaction. Start the
native reader only after the complete processor publishes readiness. On
ambiguous launch or cleanup, retain the possible owner and refuse restart;
never join an unpublished Python thread. Publish a non-running `CLOSING` state
before native destruction, perform pre-start native close outside the owner
metadata lock, and let callback-driven `stop()` defer input, output, and
recorder cleanup until its effect lease unwinds. Startup rollback always marks
diagnostic recording incomplete, including when an owner launches late.

Keep `media_pcm_queue_ms=0` as direct synchronous compatibility only; it is not
evidence for the default asynchronous reader/owner boundary. Capture replay
continues to reuse one synchronous `SherpaStreamingDecodeSession`. Its source
digest binds the dedicated-owner implementation, but replay rejects and does
not execute a production `SherpaStreamingDecodeOwner`.

Do not change STT/TTS models, endpoint policy, enrollment, tool authority,
voice-agent capabilities, public entry points, or device defaults in this
decision.

## Context / why

ADR-0110 made native stream ownership exact but intentionally left the
complete loop on whichever capture processor called it. Moving only
`accept_waveform()` or decode into another worker would split one mutable turn
across DSP/VAD, result, endpoint, reset, playback-confirm, and word-cut owners.
Adding another generic drop-oldest media stage would also splice later frames
across a missing interval and increase latency.

The capture mailbox already supplied the correct bounded boundary, but its old
contract covered queued frames only. A processor can hold one popped block
while the reader overruns, so loss must linearize against that block's external
effects. Startup also opened several workers after input without one rollback
owner, and callback re-entry could retain a legacy recorder after input had
otherwise quiesced.

## Consequences

The primary online recognizer and its primary/word-cut streams now have one
exact native thread for their entire run, while the reader can continue filling
one bounded mailbox. Loss and native errors cannot silently preserve one role,
one stale callback identity, or later PCM from an old scope. Startup, shutdown,
and restart failures remain visible and fail closed.

The default path retains its bounded reader/processor boundary and gives the
complete processor an explicit owner lifecycle. It may drop a whole old scope
under overload; it does not promise lower latency or better WER.
Headless tests use fake inputs and recognizers. They do not validate a native
audio device, microphone, GPU/model, AEC, room echo, physical barge-in, or live
conversation.

Replay remains useful shared-loop and provenance evidence only. It bypasses the
production owner thread, native reader, mailbox overload, and effect leases;
binding the owner source into a digest is not execution evidence. Fresh private
recordings, disjoint public command/multi-voice data, model-backed evaluation,
and physical live A/B remain separate gates.
