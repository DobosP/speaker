# ADR-0108: Bind exact final-model inputs to private STT replay

Date: 2026-08-01
Status: accepted

## Decision

Extend ADR-0100's synchronized diagnostic bundle to schema v2 and implement the
private-evidence bridge required by ADR-0092. Retain the four continuous PCM16
capture/playback tracks, and add a separate mode-0600 f32le spool containing
each exact endpoint-owned segment selected as input to final transcript
selection. Before any final decoder, verifier, override, or selection decision,
synchronously validate, canonicalize, copy, and hash the finite one-dimensional
float32 segment. Enqueue its immutable bytes and one closed receipt on the
bundle's single writer. Bind every receipt to its packed sample and byte ranges,
SHA-256 digest, stream and utterance identity, capture epoch and generation,
revision, sample rate, and whether the segment came from the model-gate or
selected-ASR path.

Bound that synchronous evidence work and its asynchronous disk queue to 8 MiB
per input, 32 MiB pending input bytes, 256 MiB per session, and 4,096 receipts.
Bound validation to a 64 MiB timeline, 131,072 records, and 64 KiB per line.
Never block waiting for queue capacity. Any invalid input, limit, admission
loss, write error, or receipt/lifecycle mismatch fails the diagnostic bundle
closed while the voice path continues unchanged.

Keep diagnostic schema-v1 bundles read-only: strict validation remains
available, but they are ineligible for exact-input export because they contain
no lossless final-input slices. Current validation also rejects foreign-owned
or multiply linked schema-v1 artifacts; a published manifest may retain its
private publication-temporary hardlink after a cleanup failure. Export only
complete diagnostic schema-v2
bundles through a separate mode-0600 label file bound to both the manifest and
each selected input digest. Publish those bit-for-bit slices as private
streaming-corpus schema v3 with provenance `private-diagnostic-v1`; retain
public voice corpora as schema v2 with `public-voice-v1`, and reject provenance
or version cross-use. Generic session replay must reject synchronized
diagnostic directories because their parallel WAVs are stage views, not
independent utterance fixtures.

## Context / why

ADR-0100 made continuous capture, ASR, and playback stages causally comparable,
but its selected-ASR WAV is a quantized continuous tap. Finalization can choose
a different endpoint-owned segment after pre-roll, word-cut, reset, handoff, or
trimming, so reconstructing model inputs from that WAV would invent turn
boundaries and alter samples. It therefore could not provide the exact
after-endpoint evidence needed to compare final STT candidates under ADR-0092.

Putting transcripts into the diagnostic timeline would unnecessarily enlarge
the privacy surface. Keeping labels separate lets an owner review and bind only
chosen inputs without making every recorded turn a benchmark case. Reusing
public corpus schema v2 would erase the material distinction between licensed,
disjoint public evidence and private, potentially non-disjoint owner evidence.

## Consequences

One complete schema-v2 bundle can prove which lossless float32 bytes were
selected for final transcript processing and can reproduce those bytes in the
existing STT benchmark harness. Manifest validation checks the outer artifacts,
every slice digest/range, acoustic identity, and lifecycle order. The export refuses
overwrite, keeps transcripts out of the diagnostic timeline and command output,
and does not infer turns from continuous WAV tracks.

The feature adds bounded synchronous CPU and memory work on each endpoint-final
path: finite-value scanning, canonicalization/copying, and hashing are not
nonblocking merely because disk I/O uses a writer thread. It changes no model,
transcript selection, entry point, tool authority, enrollment policy, or
default. The headless contract proves data binding only; it makes no claim
about a model, GPU, WER, latency, AEC, barge-in, device behavior, natural
conversation, or live audio. Evidence begins after endpoint segment selection
and still lacks physical raw-microphone PCM, native-reader operations, exact
online-recognizer accept/reset operations, and audible-playout proof. Those
gaps still require disjoint public/private comparisons and fresh owner-run
`./live.sh` validation.
