Valid until: main advances beyond this task's landing commit — then treat as
history.

# Task result — stock Moonshine external endpoint

## Outcome

Added a receipt-bound, opt-in schema-v7 benchmark for the only safe stock
Moonshine Voice 0.1.0 VAD-bypass path: externally presegmented complete PCM,
exact threshold `+0.0`, 1,280-sample chunks, a 500 ms partial cadence, zero
caller tail, and 2,560-sample authoritative batch alignment. The adapter
verifies the exact native stream-free ABI and zero return code before one
complete-input batch final. Worker, supervisor, adapter, manifest, and
evaluator independently reject geometry drift.

The profile is rejected for live use by ADR-0116. Legacy schema v2 is
unchanged, and no production dependency, setup selector, `core/` path,
configuration default, or `./live.sh` behavior changed. The transcript-free
receipt is `docs/evidence/2026-08-02-moonshine-external-endpoint.json`.

## Corrected native design

The initial internal/external pair was discarded after exact v0.1.0 source
review. Stock internal VAD uses a process-global recurrent Silero instance that
is not reset per stream, and its 512-aligned segments can lose up to 1,279
samples at the streaming model's 1,280-sample boundary. Whole-input padding
does not repair those per-segment losses. Forced updates also decode the whole
growing segment again, so the initial 80 ms cadence timed out on a 16.2-second
clip.

Schema v7 therefore admits external-presegmented threshold-zero input only.
Threshold zero bypasses Silero and creates one whole-input segment; padding to
2,560 makes that segment exact for both the 512- and 1,280-sample stages. A
direct native free is necessary because stock Python `Stream.close()` ignores
the C return code. The A-B-A native smoke repeated A's final and partial
sequences exactly after B.

## Real candidate evidence

- Exact external manifest SHA-256: `3e49e9c02899266a03a301973b6b0a4ec0dd61cf68559ea7ee101cdfc91882ab`.
- The 14-case, 136.615-second MInDS-14 burst had WER/CER 0.8750/0.8373,
  candidate-call RTF 0.9950, and only 10 nonempty known-speech finals.
- Finalization p50/p95 was 608.292/4,906.783 ms; seven cases had no stable
  partial; maximum reported RSS was 1,634.965 MiB.
- A fresh legacy control with identical 1,280-sample/500 ms geometry had
  WER/CER 0.6685/0.6342, RTF 0.2971, all 14 finals nonempty, finalization
  p50/p95 0.918/1.496 ms, no missing stable partial, and 1,094.664 MiB RSS.
- Both reports bind the same artifact set, runtime receipt, worker, source
  bundle, corpus, config geometry, and empty stderr. They are aggregate-only;
  no transcript rows, audio paths, or identities are committed.

The external profile regressed WER by 0.2065, consumed 3.349 times the
candidate-call compute, added 540.301 MiB reported RSS, and introduced four
empty known-speech finals. Paced and repeated cells were stopped after this
decisive burst rejection.

## Verification

- Integrated focused manifest/provision/adapter/worker/supervisor/evaluator
  gate: 326 passed, 2 real-model tests deselected.
- Exact external A-B-A worker smoke: 1 passed in 8.70 seconds.
- Independent final code/native audit: no correctness, isolation, or privacy
  blockers; candidate rejection and documentation were required.
- Split-condition non-real gate: 7,489 low-priority broad passes plus all 31
  condition-sensitive reruns, for 7,520 passing checks; 14 skipped, 24
  model-only deselected, and 11 warnings. The reruns used normal logging,
  repository-lock mode 600, and isolated async-loop timing.
- APM/DTD regression: 6 passed.
- Ruff, `git diff --check`, evidence JSON parsing, and committed/private receipt
  hash-and-size binding checks passed at the final landing gate.

## Limits and next work

This is CPU-only after-PCM benchmark evidence. It does not execute Speaker's
endpoint, microphone capture, VAD quality validation, AEC, barge-in,
enrollment, commands, agent tools, device hardware, or live conversation. The
public corpus has zero command attempts and is not an adoption gate.

Do not revive stock internal VAD as a causal control. A future pair needs a
receipt-bound native patch that resets Silero per stream and pads every VAD
segment, or a deliberately fresh process per case. The next data work is the
bounded mini Speech Commands and ECCC command/noise slice, followed later by
Paul's fresh exact recordings and live open-speaker validation.
