Valid until: this branch lands or is superseded — then treat as history.

# Task result — exact Moonshine streaming candidate

## Outcome

Added exact, disposable Moonshine Voice 0.1.0 Tiny and Small CPU candidates to
the isolated streaming-STT harness. Both are rejected for adoption; no runtime
default, setup option, production dependency, or application entry point
changed.

The worker now launches under `-I -S -B`, verifies its private source bundle,
manifest, interpreter, disabled-system-site-packages marker, runtime tree,
official Moonshine-owned wheel contents, and exact model files before directly
inserting the verified runtime path. Candidate `.pth` startup code is never
processed. The disposable runtime and models remain outside the production
virtual environment.

The adapter uses one persistent native transcriber and a fresh stream per case.
Every scheduled hypothesis forces a native update; final force/stop snapshots
only assemble the final and cannot masquerade as online partials. Reports label
accelerated replay as non-conversational and identify candidate-call compute
time separately from end-to-end RTF.

Public MInDS-14 conversion now emits a strict schema-v2 corpus with structured
suite, manifest, metadata, and source-set provenance. Reports distinguish a
real executed candidate from a production model and do not include transcript
rows, private paths, or case identifiers.

## Exact evidence

Inputs:

- `moonshine-voice==0.1.0` official Linux wheel, 56,369,045 bytes, SHA-256
  `0f833deb43bad5dcfb4cfd3257b6df83ef9abd3f27be3199622fe41932e8d916`;
- disposable Python 3.12 environment receipt: 1,257 files, 178,422,589 bytes,
  receipt SHA-256
  `f742d49cdfa3770387a59b8a1744c1c9bd9d9b1bd66c3d9de96906e327eaed85`;
- schema-v2 14-case MInDS-14 corpus SHA-256
  `a9c6859289d28b3bcc79665d890c9934f1f7b17dd7f3fe34b83a8d69a61ed3fe`.

Results:

- Small, 500 ms accelerated replay, three repeats: WER 0.6884, CER 0.6416,
  candidate-call RTF 0.5752, and one case with final disagreement.
- Small, 200 ms real-time replay, one full pass: WER 0.6957, first nonempty
  partial p50 2.016 s, candidate-call RTF 1.1864, 1,000 deadline misses, and
  maximum backlog 22.006 s.
- Tiny, 500 ms accelerated replay, one pass: WER 0.8478, CER 0.7204, and
  candidate-call RTF 0.2370.

The real-time scope begins after PCM loading inside the adapter and excludes
capture, snapshot I/O, controller IPC, VAD, endpointing, and AEC. The
accelerated runs provide accuracy and candidate-call throughput only. None is
live, held-out owner, or adoption evidence.

## Verification

- Combined deterministic Moonshine/public/streaming gate: 249 passed,
  1 real-model test deselected.
- Receipt/manifest/supervisor repair gate: 98 passed.
- Independent security/evidence/worker repair gates: 167 passed with one
  opt-in smoke skipped; 30 passed; and 35 passed with one deselected.
- Exact one-case Small worker smoke: 1 passed.
- Full `-m "not real_model"` repository gate: 6,105 passed, 13 skipped,
  21 deselected, and 9 pre-existing warnings in 113.79 seconds.
- Exact APM/double-talk gate: 6 passed.
- Exact official runtime/wheel validation: passed across all 1,257 receipt
  files.
- Scoped Ruff check/format and `git diff --check`: passed before final
  repository gates.

## Limits and next

Moonshine is English-only here and remains benchmark-only. It does not meet the
accuracy, determinism, or 200 ms streaming latency requirements. Third-party
dependencies are integrity-bound by the runtime receipt but are not
individually traced to upstream wheels; the manifest/receipt directory is a
trusted local control-plane boundary without a signature. Offline flags do not
enforce network isolation, and resource values remain worker-reported.

The next isolated candidate is NVIDIA Nemotron 3.5 ASR Streaming 0.6B on the
available RTX 4090. Any eventual recognizer selection still requires recorded
endpoint tests and a fresh physical `./live.sh` A/B.
