Valid until: main advances beyond this task's landing commit — then treat as
history.

# Task result — Moonshine Medium Streaming

## Outcome

Extended the existing isolated Moonshine Voice 0.1.0 benchmark to the official
English Medium Streaming architecture. The schema-v2 manifest now closes over
the exact seven-file model receipt, provisioning admits only the three pinned
Tiny/Small/Medium names, and the adapter maps Medium to upstream enum value 5.
No production dependency, setup selector, `core/` path, configuration default,
or `./live.sh` behavior changed.

ADR-0115 rejects Medium as a live primary recognizer and retains it only as an
opt-in comparison candidate. The transcript-free evidence receipt is
`docs/evidence/2026-08-02-moonshine-medium-streaming.json`.

## Real candidate evidence

- Exact Moonshine wheel: 56,369,045 bytes, SHA-256
  `0f833deb43bad5dcfb4cfd3257b6df83ef9abd3f27be3199622fe41932e8d916`.
- Official Medium model: seven files, 303,329,727 bytes; every file matched
  the pinned upstream size/CRC32C catalog and has an exact SHA-256 receipt.
- One exact worker smoke passed without transcript output or an audio device.
- Three-repeat 500 ms burst: WER 0.6630, CER 0.6300, candidate-call RTF
  0.3201, zero final disagreements, and maximum reported RSS 1,091.645 MiB.
- One-pass 200 ms paced: WER 0.6739, RTF 0.6508, first non-empty partial p50
  1.969 seconds, stable-partial p50/p95 6.694/28.621 seconds, 660 deadline
  misses, and 7.174 seconds maximum backlog.
- Every real worker run had empty stderr. Reports were aggregate-only and
  retained privately; committed evidence binds their SHA-256 and byte size.

Medium improves historical Small accuracy and compute/backlog figures, but its
partial latency and paced backlog are not suitable for natural conversation.
The public MInDS development slice had zero command/keyword attempts.

## Verification

- Focused Moonshine/manifest/provision/runtime/supervisor/evaluator gate:
  237 passed, 1 real-model test deselected.
- Full non-real gate: 7,426 passed, 13 skipped, 23 model-only deselected, and
  11 warnings; the two condition-specific failures passed on exact rerun with
  logging enabled and the repository lock restored to required mode 600.
- Exact real-model worker smoke: 1 passed.
- Mandatory APM/DTD regression: 6 passed.
- Ruff passed on all changed Python and test files.
- Evidence JSON parsing and `git diff --check` passed.

## Limits and next work

This benchmark worker is forced to CPU-only and has sanitized single-thread
settings, bounded IPC/timeouts, and process-group teardown, but no Bubblewrap
network namespace or independently verified hard RAM/CPU limit. The comparison
starts after PCM exists and supplies no microphone, capture-loop, VAD/endpoint
ownership, AEC, barge-in, identity, tool, device, or live evidence.

The comparable Tiny/Small/Medium adapter retains Moonshine's internal default
VAD. Upstream recommends disabling it on pre-segmented clips, so the next
bounded configuration should give endpoint ownership back to Speaker and test
VAD-disabled Moonshine explicitly. Add mini Speech Commands and ECCC slices for
short-command and hard-noise coverage before further candidate decisions.
