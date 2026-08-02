# ADR-0116: Add and reject the stock Moonshine external endpoint

Date: 2026-08-02
Status: accepted

## Decision

Admit one opt-in schema-v7 benchmark profile for the exact stock Moonshine
Voice 0.1.0 runtime, and reject it for live adoption. The profile accepts only
externally presegmented complete PCM, configures the native VAD threshold to
positive zero so Silero inference is bypassed, binds 1,280-sample chunks and a
500 ms partial cadence, forbids caller tail padding, and zero-pads the
authoritative batch input to the 2,560-sample VAD/model least-common multiple.
Free the online stream through the exact native API and require its zero return
code before using one complete-input batch decode as the final. Keep schema v2,
runtime defaults, setup, `core/`, and `./live.sh` unchanged.

Do not offer a stock internal-VAD schema-v7 comparison. A future controlled
internal/external pair requires a receipt-bound native build that resets
Silero recurrent state per stream and preserves each VAD segment's remainder
at the 1,280-sample streaming-model boundary, or an explicitly isolated
fresh-process design.

## Context / why

The first implementation treated whole-input 2,560-sample alignment as enough
for both profiles and polled partials every 80 ms. Exact v0.1.0 source review
showed three invalid assumptions. Internal VAD emits 512-aligned segments while
the streaming frontend consumes only complete 1,280-sample chunks, so a segment
can lose up to 1,279 samples. Silero is one process-global recurrent singleton
whose state is not reset by `VoiceActivityDetector::start()`. Moonshine also
re-decodes the accumulated segment at every forced update, making 80 ms polling
pathologically expensive. The stock Python `Stream.close()` additionally
discards the native free return code. The public disable-VAD flag described by
an older API comment is absent and its flags argument is ignored; threshold
`+0.0` is the exact stock path that treats every hop as voice without running
Silero.

The repaired A-B-A real smoke completed in 8.70 seconds and repeated A's final
and partial sequences exactly, but the middle known-speech clip returned an
empty final. On the 14-case MInDS-14 development slice, the external profile
had WER/CER 0.8750/0.8373, only 10 nonempty finals, candidate-call RTF 0.9950,
finalization p50/p95 608/4,907 ms, seven missing stable partials, and 1,635 MiB
maximum reported RSS. A fresh legacy control with the same 1,280-sample/500 ms
stream geometry had WER/CER 0.6685/0.6342, 14 nonempty finals, RTF 0.2971,
finalization p50/p95 0.918/1.496 ms, no missing stable partials, and 1,095 MiB
RSS. The comparison is descriptive rather than causal because the finalization
and VAD paths differ.

## Consequences

The benchmark can now reproduce the only stock v0.1.0 VAD-bypass path without
silently accepting an 80 ms decode-amplification request or an unchecked native
stream release. Receipt, worker, supervisor, adapter, and evaluator independently
enforce the external-only geometry. Real reports remain aggregate-only and the
committed evidence contains no transcript, audio path, or identity.

The profile performs a full second decode, is much less accurate and roughly
3.35 times as compute-heavy as the matched legacy control, and is not a live
recognizer. No paced or repeated cell is warranted after the burst rejection.
The result begins after externally bounded PCM exists and does not validate the
Speaker endpoint, capture, VAD quality, AEC, barge-in, enrollment, commands,
tools, devices, or live conversation. Mini Speech Commands and ECCC remain the
next public short-command/noise data work; owner recordings remain a later live
gate.
