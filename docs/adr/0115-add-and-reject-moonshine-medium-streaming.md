# ADR-0115: Add and reject Moonshine Medium Streaming

Date: 2026-08-02
Status: accepted

## Decision

Extend the isolated Moonshine Voice 0.1.0 CPU benchmark with the official
English `medium-streaming` architecture and its exact seven-file SHA-256/size
receipt. Keep the existing persistent-transcriber/fresh-stream adapter,
aggregate evaluator, and schema-v2 worker boundary. Reject this configuration
as a live primary recognizer and retain it only as an opt-in benchmark
candidate. Do not add it to setup, `core/`, configuration defaults, or
`./live.sh`.

The candidate is the `MEDIUM_STREAMING=5` architecture from upstream release
commit `cc1695646a560f2eec7f7c058f3c4d580f039e4b`. The exact Linux wheel remains
56,369,045 bytes at SHA-256
`0f833deb43bad5dcfb4cfd3257b6df83ef9abd3f27be3199622fe41932e8d916`.
The seven model files total 303,329,727 bytes and match the pinned upstream
catalog's size/CRC32C entries. Their individual SHA-256 receipts and the
transcript-free run aggregates are bound in
`docs/evidence/2026-08-02-moonshine-medium-streaming.json`.

## Context / why

ADR-0090 rejected Moonshine Tiny and Small, but the same pinned 0.1.0 release
also ships a 245M-parameter Medium streaming architecture through the same
native interface. It therefore needs an exact comparison rather than a new
engine or a production dependency.

On the 14-case public MInDS development slice, the three-repeat 500 ms burst
cell was deterministic at WER 0.6630, CER 0.6300, and candidate-call RTF
0.3201, with no final disagreement and maximum reported RSS 1,091.645 MiB.
This improves the historical Small burst result of WER 0.6884 and RTF 0.5752.

The one-pass 200 ms paced cell remained unsuitable for natural conversation:
WER was 0.6739, candidate-call RTF 0.6508, first non-empty partial p50 was
1.969 seconds, stable-partial p50/p95 was 6.694/28.621 seconds, 660 deadlines
were missed, and maximum backlog reached 7.174 seconds. Those figures improve
Small's WER 0.6957, RTF 1.1864, 1,000 misses, and 22.006-second backlog, but
they do not establish a usable live stream.

The official native runtime is CPU/CoreML/NNAPI-oriented and does not expose a
CUDA provider. A separate Transformers/CUDA route is not fully efficient
streaming and is not equivalent evidence. The current comparable Tiny/Small/
Medium harness retains Moonshine's default internal VAD. Upstream recommends
disabling that VAD for already segmented accuracy corpora, so an
endpoint-owned/VAD-disabled cell must be evaluated as a distinct future
configuration rather than silently rewriting these results.

## Consequences

The exact isolated harness can now provision, load, and compare Medium without
changing the application. A one-case real worker smoke and the aggregate
burst/paced cells completed with empty worker stderr. The public slice had no
command or keyword attempts and supplies no capture, device-reader, endpoint,
AEC, barge-in, identity, tool, or live evidence. It is not held-out owner
speech and cannot authorize adoption.

Before reconsidering Moonshine, compare an explicitly receipt-bound
VAD-disabled/external-endpoint configuration and add short-command plus noisy
known-text coverage. Any surviving final-recognizer candidate still requires
fresh private command/negative recordings and physical open-speaker A/B.
