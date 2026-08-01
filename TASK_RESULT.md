Valid until: this branch lands or is superseded — then treat as history.

# Task result — exact Parakeet Realtime EOU NeMo reference

## Outcome

Added a receipt-bound NVIDIA Parakeet Realtime EOU 120M v1 implementation to
the isolated streaming evaluator and rejected it for application adoption. It
does not alter `.venv`, setup, configuration, STT defaults, `./live.sh`, or the
single `python -m core --session` application entry point.

Protocol v3 and manifest schema v5 preserve native EOU/EOB, transcript-delta,
source/tail, endpoint, resource, and cgroup evidence. Only complete-source EOU
can authorize a response; EOB, source-early endpoints, and tail exhaustion
cannot. The exact Python/NeMo/Torch/CUDA/NumPy runtime and model are selected by
hash and run with networkless Bubblewrap, authenticated read-only source/input
mounts, private scratch, and a verified 2-CPU/8-GiB/zero-swap worker scope.

## Evidence

The first burst report (`dd597e70...`) is preserved but accuracy-invalid: NeMo
returns transcript deltas, while the first adapter treated them as snapshots.
The fixed adapter accumulates deltas, preserves word/subword boundaries and
marker-only terminal text, and bounds the cumulative hypothesis.

The corrected burst report (`0bf07200...`) recorded WER 0.6576, CER 0.6161,
4/14 exact, model-input RTF 0.1910, 17.745 s load, 1,906.1 MiB sampled RSS,
and 924 MiB reported PyTorch VRAM. The paced report (`4fdfffa5...`) preserved
accuracy but recorded model-input RTF 0.2200, three deadline misses, and
189.411 ms maximum backlog. Both are after-PCM replay, not conversational or
capture-to-text latency.

Ten of 14 native EOU events preceded declared source end in both runs, leaving
only four response-authoritative endpoints. Burst left 198,848 samples
(9.097%) unconsumed; pacing consumed four more chunks and left 193,728 samples
(8.863%) unconsumed. This pace-sensitive endpoint position and the small
one-repeat corpus prevent endpoint or adoption claims.

The identical one-thread Zipformer control had WER 0.7283, CER 0.6259, 2/14
exact, RTF 0.089, 1.841 s load, and 378.8 MiB sampled RSS. Parakeet made 13
fewer word errors, but its model-input RTF was 2.146 times higher, load 9.641
times longer, sampled RSS 5.032 times larger, and its frozen runtime occupies
6.43 GB. It is therefore useful only as an isolated native-EOU reference.

Private telemetry and reports are mode 0600. The telemetry schema and audit
receipts preserve two impossible initial 593.51 W driver samples, mark them
invalid against the 150 W device cap, and exclude them without clamping. No raw
audio or transcripts are committed.

## Verification

- Streaming-STT gate: 567 passed, 3 model-only deselected.
- Full non-model logic gate: 6,787 passed, 13 skipped, 23 deselected, with nine
  pre-existing warnings.
- Required APM/double-talk gate: 6 passed.
- Changed-file Ruff and `git diff --check`: passed.
- Exact RTX 4090 burst and paced runs: completed with verified worker cgroups,
  empty worker stderr, and code/model/runtime/corpus hashes bound in each report.

The first full-suite attempt used `SPEAKER_TEST_LOG=0` and invalidated one test
whose contract is to inspect transcript log records. That test passed under the
normal logging environment, and the complete normal-environment rerun is the
green gate reported above.

## Remaining limits and next work

There is no live microphone, VAD, AEC, room, owner, multiple-speaker, semantic
turn-ground-truth, or physical barge-in evidence. Trust residuals include local
unsigned receipts, same-UID path swap limits, synchronous process-wide fd
suppression, isolated-importer assumptions, sampled RSS, and no hard VRAM
isolation.

Next, build a separately receipt-bound `parakeet.cpp` candidate and prepare a
small deterministic Common Voice Spontaneous Speech v4 slice. Only after those
headless comparisons should the best portable candidate enter capture/AEC,
multi-device, owner-recording, and fresh bare-speaker live A/B gates.
