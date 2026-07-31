Valid until: this branch lands or is superseded — then treat as history.

# Task result — isolated streaming-STT harness

## Outcome

Added a generic controller for future long-lived streaming-STT candidates
without installing or importing a candidate runtime. The only adapter in this
change is a deterministic fake keyed by exact PCM hashes. This proves the
isolation, provenance, bounded protocol, replay, privacy, metrics, and teardown
seams; it is not model-quality evidence.

The controller validates hash/size-bound worker and corpus manifests, copies
validated f32le PCM into private scratch, and stable-reads the fixed worker
import closure into a private single-link source bundle. It launches that
bundle via an inherited directory descriptor with the declared interpreter
under `-I -B`. Ready and final messages bind the verified bundle digest.

Bounded JSONL, regular-file reads, and original-process-group teardown reject
oversized output, symlinks, hardlinks, replacement/mutation races, sparse
oversized inputs, and post-load growth. A pwd-derived host path plus parent/file
advisory locks serializes cooperating controllers independently of `TMPDIR`.

Aggregate-only reports bind the evaluator evidence set, corpus, interpreter,
source bundle, worker, and artifacts. They contain accuracy, command recall,
partial stability, latency, backlog, determinism, and worker-reported resource
metrics without transcript rows, paths, or private case identifiers.

## Files

- `tools/streaming_stt_eval.py` — aggregate controller and CLI.
- `tools/streaming_stt/` — bounded I/O, manifests, corpus/protocol, source
  bundle, supervisor, metrics, worker, and deterministic fake adapter.
- `tests/streaming_stt_helpers.py`, `tests/test_streaming_stt_*.py` —
  deterministic protocol/privacy/metric and adversarial process regressions.
- `docs/adr/0089-isolate-streaming-stt-benchmarks.md` — isolation decision and
  explicit non-adoption boundary.
- `STATUS.md`, `docs/agent-map.md`, `docs/agent-testing.md` — current truth,
  routing, limitations, and focused command.

## Verification

Independent pre-rebase review:

```text
tests/test_streaming_stt_manifest.py tests/test_streaming_stt_protocol.py
tests/test_streaming_stt_supervisor.py tests/test_streaming_stt_eval.py
tests/test_streaming_stt_environment.py

79 passed
```

The same files plus existing recorded/public STT evaluator regressions passed
181 tests. A broader evaluator gate passed 209 tests. Pycompile, scoped Ruff
check/format, `git diff --check`, and the STATUS-size guard passed.

The first post-rebase full gate exposed a real import-smoke failure: the worker
required launch arguments while merely being imported. The repair leaves
normal imports inert but still verifies the inherited private source tree
before project imports whenever the worker executes.

Independent follow-up review also found that drainer construction/start
failures occurred outside startup cleanup. Both drainer starts now share the
process cleanup guard; failures reap the process group, stop any started
drainer, close inherited descriptors and safe pipe ends, and return a
detail-free `worker_start` error. Output pipes with live drainers are skipped
until those threads stop, preventing a buffered-close deadlock.

Final post-rebase receipts:

- five streaming files: 83 passed;
- exact import/startup/live-drainer repair gate: 8 passed independently;
- streaming, recorded/public, worker-import, and APM combined: 273 passed;
- full `-m "not real_model"` gate: 6016 passed, 13 skipped, 20 deselected,
  9 warnings in 110.90 seconds;
- pycompile, scoped Ruff check/format, `git diff --check`, and the 100-line
  STATUS limit: passed.

## Limits and next evidence

- `fake-json-v1` cannot rank STT quality.
- VAD, endpointing, AEC, playback, multiple speakers, and natural conversation
  remain outside this harness.
- Offline environment flags are not a network sandbox. RAM/VRAM values are
  schema-bounded and finite but self-reported, not OS-enforced.
- The Linux/POSIX implementation uses `/proc/self/fd` and advisory `flock`.
  Original-process-group teardown cannot fence a descendant that calls
  `setsid()` and closes inherited pipes.
- Each real adapter still needs a disposable exact environment, pinned artifact
  receipt, public/private corpus runs, recorded endpoint acceptance, and live
  A/B before any default changes.

No candidate package/model, GPU job, audio device, or live-hardware test was
used by this change.
