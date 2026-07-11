# ADR-0032: Bound llama.cpp generation and batch thread pools

Date: 2026-07-11
Status: accepted

## Decision
Resolve a typed llama.cpp CPU thread pair at the `LlamaCppLLM` provider boundary
before model load, and always pass both counts to the pinned binding. Use the
tightest valid CPU count visible to the calling thread (`process_cpu_count` and
`sched_getaffinity`, then the portable host-count fallback). Automatic single-
token generation leaves arithmetic worker-count headroom of two where possible
and is capped at four. Automatic multi-token/prompt work may rise to eight only
while retaining at least generation-sized arithmetic headroom; this keeps
topologies of eight CPUs or fewer paired and resolves this 32-logical-CPU
host to 4 generation / 8 batch threads. An explicit generation value wins and
an omitted batch value follows it; an explicit batch value wins independently.
Reject booleans, non-integers, zero, and negative values before native load.
Warn when an explicit pair exceeds the bounded voice-headroom ceiling. Keep
normal OS scheduling with no process-wide affinity pin or CPU reservation.
Apply the resolver to runtime, remote text serving, and real benchmarks; keep
the 2/2 sanity probe as a topology-independent correctness gate. Benchmarks
sharing one GGUF for main/fast roles must also share one context, as production
does.

## Context / why
[`llama-cpp-python` 0.3.33](https://github.com/abetlen/llama-cpp-python/blob/e894f0d6010be8de14400359c10c87c16ddb3829/llama_cpp/llama.py#L304-L317)
bypasses llama.cpp's hybrid-aware CLI selection and defaults generation to half
of all logical CPUs and batch work to all logical CPUs. The old speaker factory
replaced only generation with `cpu_count - 2`, so this i9-13980HX's 8 SMT
P-cores plus 16 E-cores became a llama.cpp 30/32 pair;
the benchmark bypass became a different 16/32 pair. Both counted heterogeneous
logical processors as homogeneous workers, and batch/prompt evaluation could
still occupy every CPU. The old rationale also incorrectly called STT/TTS
sequential with inference: sentence streaming can synthesize/play TTS while
later model tokens are still generated, while capture/VAD/barge work remains
live throughout.

The [pinned API](https://github.com/ggml-org/llama.cpp/blob/78d2f524682d9fee790a6460c93d018dafeb5229/include/llama.h#L969-L978)
defines generation as single-token decode and batch threads as prompt/multi-token
decode. A controlled actual-Q4 grid (162-token prompt, 64 output pieces, four
runs per pair, load excluded) measured the old 30/32 pair at a
4.651 s median TTFT with a 1.916--4.750 s range and 6.135 s median total. An
interleaved confirmation measured 4/8 at 0.333 s TTFT and 1.092 s total. 8/16
was faster, but isolated throughput is not evidence that live capture, TTS, and
barge-in remain healthy. A synthetic 10 ms control loop found 4/8 wake jitter
at p99 (0.220 versus 0.226 ms; maxima 0.248 versus 0.233 ms), with 5.912 versus
4.074 ms median native cancel exits and healthy reuse. This supports 4/8 only
where larger-host arithmetic headroom exists. Hard P-core affinity was rejected:
the high-level binding has no supported per-pool affinity seam, process-wide
pinning would also constrain audio, and mobile schedulers need thermal/load
freedom.

## Consequences
No Python llama.cpp construction path can silently restore the binding's 16/32
or all-CPU batch default, and invalid zero-like config fails before loading a
GGUF. Phone-size CPU sets auto-resolve no higher than 4/4; larger hosts can use
4/8 for faster prompt evaluation while its worker count stays at least four below
detected availability. Explicit pairs remain available for measured hardware-
specific tuning, with a warning when they exceed the headroom ceiling.
The post-change production-auto Q4 gate reported native 4/8, answered all three
quality probes at 61.9--114.3 ms TTFT with natural stops and no reasoning fallback,
exited native cancellation in 7.3 ms, and recovered the same healthy context.
The real benchmark now exercises the same template, KV-cache, thread policy,
and shared-context identity as production. The host grid and scheduler probe are
headless evidence only: actual phone/low-power thermal behavior, callback xruns,
owner-mic talk-over cuts, false cuts, and reply-tail continuity still require
live validation. A container CPU quota that does not also narrow affinity is not
inferred; such deployments must set an explicit measured pair.
