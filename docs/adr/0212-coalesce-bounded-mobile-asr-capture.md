# ADR-0212: Coalesce bounded mobile capture through short ASR stalls

Date: 2026-10-04
Status: accepted

## Decision

Keep at most four sent, unacknowledged AsrWorkerAudio messages. When these credits are busy,
accept small PCM16 capture chunks into one pending batch up to 32,768 bytes; cap sent-plus-
pending meaningful PCM at the existing four-full-chunk budget of 131,072 bytes. Reserve its
sequence when the batch first receives audio and flush it only after an exact outstanding ACK.
Copy incoming PCM and preserve every original capture boundary in bounded Uint16 byte offsets.
Validate all offsets before worker input, observe decode/endpoint/reset at each original
boundary, and share the existing 2,048-step decode bound across the whole transport batch.

Revoke callbacks and discard pending PCM/offset buffers on exact session end or worker failure.
Foreign, duplicate and stale ACKs remain inert. Retain existing fail-closed behavior when the
byte/batch bound, native decoder bound or transport uncertainty is exceeded; never silently
drop samples, relabel a discontinuity as continuous audio, or relax successor cleanup.

## Context / why

The Assistant adapter treats any rejected feed as capture failure. The old four-message cap
therefore stopped always-on listening after four tiny recorder frames, regardless of how
little actual audio they carried. A brief worker/GC/decode stall could make the bot look stuck.
A byte-bounded pending batch tolerates that burst without increasing concurrent native work.

Combining PCM without its original boundaries would change recognition: the worker checks
native endpoint after each accepted capture chunk, and a batch spanning silence plus renewed
speech could hide an endpoint. Boundary metadata preserves those observation points. The
cumulative decode-step bound prevents many tiny subchunks from multiplying permitted work.

## Consequences

At 16 kHz mono PCM16, pending audio covers at most 1.024 seconds and the meaningful overall
payload cap is 4.096 seconds. The app's pending fixed PCM and offset arrays consume up to
64 KiB; transport copies, array backing retention, model/native feature state and ordinary
Dart objects are additional memory, so 131,072 bytes is not an application-RSS claim.

Accepted PCM remains ordered and caller-buffer-independent. Detach-before-send and
reserve-before-send permit reentrant ACK/revocation without sequence reuse or successor input.
The repaired transport absorbs temporary bursts; it does not improve model WER or allow an
ASR engine slower than real time to catch up indefinitely.

Independent focused ASR tests pass 52 cases and scoped Dart analysis is clean. The combined
full mobile suite, including ADR-0211, passes 252 tests. Nine new regressions cover lossless
batching, bounds, stale ACK/end, send failure/reentrancy, endpoint equivalence, hostile metadata,
cumulative decode work and revocation during endpoint output. No phone/plugin/model/microphone,
RTF, recognition quality, battery or thermal validation ran. ADR-0203's exact lifecycle scope
and the existing language/model defaults are unchanged.
