# ADR-0027: Sink-attested playback history

Date: 2026-07-11
Status: accepted

## Decision
Add an opt-in terminal playback-receipt contract at the audio-engine boundary.
For an engine that advertises this capability, register every admitted TTS request
as an opaque fragment before playback; commit assistant memory, resume text, and
follow-up cadence only from the engine's terminal, sink-attested safe text.
Preserve fragment registration order across delayed callbacks, retain genuinely
heard text after interruption, and suppress cadence when the receipt's speech
epoch was cancelled. Treat first-sink-sample onset separately as evidence for
cut/resume and echo filtering. Sequence an intervening user turn between older
and later assistant groups. Hold the next conversation read behind unresolved
prior receipts for up to one second; on timeout, skip conversation memory for
that turn. Keep the legacy admission-time path unchanged for engines that do not
advertise receipts. Do not reinterpret legacy `on_done` as playout evidence and
do not derive words from sample percentages.

## Context / why
The runtime previously called queued text "actually spoken." Non-stream replies
entered memory as soon as `engine.speak()` accepted them; streamed replies joined
every admitted sentence when model production ended. A cut could therefore make
resume cite an unheard future sentence, make the echo guard match text that never
started, and arm a proactive follow-up before the output device drained.

Existing completion callbacks cannot repair this: Sherpa fires `on_done` after
synthesis but before FIFO drain, FileReplay fires it from an abort-inclusive
`finally`, and LiveKit completion does not prove remote playout. TTS word timing
is also unavailable, so mapping a played-sample ratio onto text would fabricate a
word boundary. Utterance fragments give a conservative boundary: earlier fully
drained fragments remain useful, while an unaligned partial fragment contributes
no text unless its engine can attest a safe prefix.

## Consequences
`ScriptedEngine` is the first receipt-capable adapter and deterministically tests
completion, interruption, replacement, callback races, ordered memory, shutdown,
resume, echo, and follow-up behavior. FileReplay, Sherpa, and LiveKit remain on
the legacy path until each can satisfy exactly-one terminal receipt on every
completion/drop/failure/shutdown path. Sherpa requires sample ownership inside
its playback FIFO plus off-audio-thread callback dispatch; that is a separate
audio behavior branch. A receipt-capable shutdown must terminalize playback and
drain its memory commits before closing the bus or memory backend. Its
`wait_idle()` includes terminal receipts and their pending memory commits; a caller
intentionally injecting a cut into held playback must request the explicit
brain-only wait.
