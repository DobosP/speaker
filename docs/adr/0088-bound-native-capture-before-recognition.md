# ADR-0088: Bound native capture before recognition

Date: 2026-07-31
Status: accepted

## Decision
Put live Sherpa capture behind a capture-only `MediaSession`: a dedicated native
reader copies each block into a mailbox capped at 300 ms and eight frames, while
the established capture processor consumes it. The producer never waits for
capacity. Saturation atomically retires queued stale PCM, keeps the newest
admissible block, rotates capture generation, and coalesces one typed gap before
that block.

At read return, bind the owned PCM to immutable capture/playout context:
monotonic start/end time, source sample coordinate, source device/generation,
capture-domain authority generation, speaking/barge-watch state, `_speak_gen`,
playback-run generation, route/speaker authority, measured AEC delay, and owned
zero-delay/delayed/coherence reference windows. Both reference windows come from
one atomic playback-ring head. The output callback advances one cumulative
rational playback-reference sample clock through played audio and zero-filled
silence; only a native output-domain open/restart resets its phase. The processor
uses capture time for VAD/endpoint decisions and the captured reference for
AEC/coherence/recording; processing time is diagnostics only. A later playback
or source generation fences stale popped PCM and its confirmation window from
command, barge, route, and speaker authority. Confirmed playback is validated
and irreversibly claimed in one generation-lock linearization; while its
callback and reentrant stop run without that lock, the claim prevents admission
of a later playback run and is released exactly when those effects finish.

Device overflow and source recovery use the same discontinuity ordering with
distinct reasons. Recovery lifecycle events cross a separate bounded priority
control lane: the native reader only enqueues, lower-severity arrivals cannot
evict retained recovery/fatal state, and the processor emits epoch-fenced
callbacks and metrics. Before post-gap PCM, the processor aborts the active
acoustic turn, resets KWS and streaming continuity, and rotates lineage. The
one-shot guard is consumed only by the first nonempty block that survives source,
playback, and epoch fences; that block cannot publish a control or update ambient
floors. A same-domain gap/reopen preserves ambient/playback and speech-evidence
floors plus the measured AEC delay and accepted-lag history; a true source
handoff resets and relearns them after re-resolving its capture domain.
`media_pcm_queue_ms=0` preserves inline scheduling, but native overflow remains
a mandatory gap/reset. Keep enrollment and barge-in authority outside this
transport: open-speaker barge-in remains enrollment-free, and speaker evidence
remains an optional multi-voice policy.

## Context / why
The native read and every capture-stage operation previously shared one thread.
Slow APM, VAD, recognition, endpoint, speaker, or callback work could therefore
stop reads long enough for the device to overrun. Worse, Sherpa discarded the
native `overflowed` flag, so missing PCM was silently joined into one recognizer
turn. Changing STT models cannot repair missing or cross-gap audio, and the
post-recognition `SessionActor` runs too late to bound this path. An unbounded
queue would avoid immediate loss only by turning processor delay into stale
conversation latency and memory growth.

## Consequences
Live capture pays one contiguous float32 copy per native block and one dedicated
reader thread, plus bounded per-block reference/context snapshots. Processor
latency is capped instead of accumulated; overload is observable through typed
lineage and cannot yield a final or keyword spanning missing PCM. Slow lifecycle
callbacks can delay the processor and cause an explicit backpressure gap, but
cannot stall the native reader. Same-domain gaps preserve calibration, route
authority, enrollment, learned floors, measured-delay history, and lexical barge
policy while resetting only invalid cross-gap windows. Real source recovery
continues to re-resolve its capture domain. Startup calibration remains inline.
Audible playback and STT/TTS model selection are unchanged; the reference tee now
records silence as timeline progress without per-callback rounding drift.
Final-event dispatch and the remote LiveKit path remain outside this slice and
can adopt the transport contract independently.
