# ADR-0090: Add exact Moonshine streaming evidence and reject adoption

Date: 2026-07-31
Status: accepted

## Decision

Land Moonshine Voice 0.1.0 Tiny and Small as isolated, benchmark-only English
CPU adapters for ADR-0089, and reject both as final-recognizer or live-runtime
choices. Require a disposable private virtual environment with system site
packages disabled, the official release wheel at SHA-256
`0f833deb43bad5dcfb4cfd3257b6df83ef9abd3f27be3199622fe41932e8d916`,
exact model-file receipts, and a receipt for every installed runtime file.
Launch the worker with `-I -S -B`; verify its source, manifest, interpreter,
runtime tree, installed Moonshine-owned wheel contents, and model artifacts
before inserting the verified site-packages directory directly into
`sys.path`. Never process candidate `.pth` files. Keep candidate packages and
models outside the production virtual environment, preserve `./live.sh` as the
single application entry point, and make no ASR default change.

## Context / why

Moonshine is small, native-streaming, and suitable for testing the low-resource
edge of the architecture, but the first implementation accidentally measured
scheduled 200 ms updates through Moonshine's 500 ms native cache and counted
finalization snapshots as online partials. Those results understated compute
cost and overstated partial quality. The repaired adapter forces every
scheduled update, reserves force/stop snapshots for final assembly, and labels
compute as candidate-call wall time rather than end-to-end RTF.

The exact public MInDS-14 schema-v2 run rejects adoption. Small at a 500 ms
accelerated replay interval produced WER 0.6884, CER 0.6416, candidate-call RTF
0.5752, and one final disagreement across three repeats. One full 200 ms
real-time pass produced WER 0.6957, first-nonempty-partial p50 2.016 seconds,
candidate-call RTF 1.1864, 1,000 missed deadlines, and maximum backlog 22.006
seconds. Tiny at 500 ms accelerated replay produced WER 0.8478 and RTF 0.2370.
Accelerated replay latency is explicitly non-conversational; the real-time
measurement is valid for this adapter-after-PCM-load scope but does not include
capture, PCM file I/O, controller IPC, VAD, endpointing, or AEC.

## Consequences

The reusable result is an exact real-candidate boundary: schema-v2 public
provenance, no-startup-code worker launch, direct persistent-transcriber/fresh-
stream execution, deterministic worker-boundary coverage, and an opt-in
one-case real-model smoke. Moonshine remains available only for repeatable
comparison and cannot be selected by setup or normal runtime code. Nemotron
3.5 ASR Streaming 0.6B is the next isolated candidate; recorded endpoint and
physical `./live.sh` A/B evidence remain mandatory before any recognizer
adoption.

Third-party dependency files are integrity-bound by the runtime-tree receipt
but are not individually traced to their upstream wheels. The manifest and
receipt directory remains a trusted local control-plane boundary, without a
signature. Offline environment flags are not a network sandbox, and resources
remain worker-reported. `-S` prevents `.pth` execution even within that residual
boundary.
