Valid until: this branch lands or `main` advances beyond `0613e62` — then treat
as history.

# Task result — exact private final-input evidence

## Outcome

Diagnostic schema v2 retains four continuous PCM16 tracks and adds a separate
lossless f32le spool for each exact endpoint-owned segment offered to final
transcript selection. Typed receipts bind acoustic epoch/generation, turn and
revision identity, packed ranges, role, rate, and digest. Copy/scan/hash work is
synchronously bounded; disk writes share the asynchronous diagnostic writer.
Any loss invalidates evidence but does not alter transcript selection.

A private mode-0600 label file can export selected slices bit-for-bit as
streaming-corpus schema v3 (`private-diagnostic-v1`). Public corpus schema v2
remains `public-voice-v1`; version/provenance cross-use is rejected. Generic
session replay refuses diagnostic directories. Schema-v1 diagnostic bundles
remain validation-only.

## Verification

- Exact-evidence focused gate: 199 passed, 1 skipped.
- Public-corpus/supervisor adjacency: 186 passed.
- App/session adjacency: 83 passed.
- Required APM/DTD: 6 passed.
- Complete non-model split: 7,117 passed as 7,087 low-priority repository
  checks plus 30 isolated logging/thread/permission-sensitive checks; 14
  skipped, 23 model-only deselected, 9 pre-existing warnings.
- Adversarial coverage includes byte/record caps, inode swaps, manifest
  mutation, lifecycle order and real queue/start serialization, CLI privacy,
  schema-v3 benchmark execution, and diagnostic failure isolation.
- Python compilation and `git diff --check`: passed.

The first combined full run exposed only harness constraints: disabled logging
suppressed a logging assertion, the worktree checkout gave one private lock
0664 mode, and the sandbox cannot wake cross-thread asyncio loops. The exact 30
affected cases passed outside that sandbox under intended permissions.

## Limits / next

No owner corpus, live audio, candidate model, GPU, WER, latency, AEC, barge-in,
device, or default claim ran. Next acquire licensed public command/noise strata,
export future owner labels separately, and run sequential GPU-isolated STT
comparisons before changing streaming ownership or model selection.

## Merge recommendation

Land this independently reviewed evidence foundation. It changes no model,
entry point, tool authority, enrollment/barge-in policy, or runtime default.
