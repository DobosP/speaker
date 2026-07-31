# ADR-0089: Isolate streaming STT benchmarks before candidate trials

Date: 2026-07-31
Status: accepted

## Decision

Require every streaming-STT candidate benchmark to run as one sequential child
process from an exact hash- and size-bound worker manifest. Stable-read the
fixed worker import closure into a run-private, single-link source bundle,
launch its worker through an inherited bundle-directory descriptor under the
declared interpreter's `-I -B`, and require pre-import plus ready/final bundle
digest verification. Use one fixed host-state lease independent of ambient
temporary directories, private no-symlink scratch, bounded JSONL and file
snapshots, and teardown of the original worker process group. Bind the selected
evaluator evidence set, corpus, interpreter, source bundle, worker, and artifacts
in aggregate-only output. Land only the deterministic `fake-json-v1` adapter in
this change; do not install, execute, select, or add a runtime dependency for
any real candidate.

## Context / why

The recorded endpoint evaluator in ADR-0078/0080 does not measure the partial
hypotheses, churn, backlog, finalization lag, or persistent resource behavior of
a long-lived streaming recognizer. Candidate projects also carry incompatible
native and model dependencies. Importing those candidates into the controller
would let dependency state, ambient credentials, crashes, or unbounded output
contaminate both the comparison and the lean shipped runtime. A fake protocol
seam is therefore required before scarce model/GPU trials can produce trusted
evidence. It is not STT-quality evidence by itself.

## Consequences

Future Moonshine, Nemotron, or other adapters must be provisioned in disposable
exact environments and supply artifact receipts before using this controller.
Candidate trials can compare aggregate transcript accuracy, command recall,
partial stability, latency, backlog, determinism, and worker-reported resources
without exposing transcript rows. The harness does not cover VAD, endpointing,
AEC, microphones, playback, multi-speaker behavior, or natural conversation,
and it has no adoption authority. SenseVoice and the opt-in Parakeet/Small final
pair remain unchanged until separate recorded and live acceptance gates pass.

This is a Linux/POSIX isolation contract: source execution uses an inherited
`/proc/self/fd` directory and the host lease uses advisory `flock`. Teardown can
fence only the original process group; a descendant that calls `setsid()` and
closes inherited pipes can escape it. The minimal environment removes ambient
secrets and sets library offline flags, but it does not enforce a network
sandbox. There is no OS-level RAM or VRAM limit. Timing and resource values are
schema-bounded and finite but self-reported by the worker, so future real-model
evidence needs independent corroboration. Partial and final `samples_seen`
values share one monotonic domain containing input PCM plus declared tail
padding; the final value must equal that total.
