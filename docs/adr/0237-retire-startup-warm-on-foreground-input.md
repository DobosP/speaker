# ADR-0237: Retire speculative startup warm when foreground work arrives

Date: 2026-10-10
Status: accepted

## Decision

Give startup warm one runtime-owned cancellation event. Retire it monotonically
when accepted input allocates a generation or runtime shutdown begins. Signal
only on the capture/control path; never wait there for provider or media cleanup.
Before each media/model/helper stage, refuse a retired plan. Use the existing
cancellation-aware local stream collector for providers that expose streaming;
keep generate-only legacy compatibility through pre-entry signature adaptation,
with one invocation and no retry after an entered TypeError.

Keep media first, the selected fast/all model residency plan, all foreground
capabilities, model options/context/output limits and decision deadlines unchanged.
Warm readiness means the plan finished or retired, not successful native prefill.
Model/helper warm uses the existing hook-free context snapshot, preserves inherited
cancellation, narrows scope to LOCAL_ONLY and restores the caller's context.
Explicit built-in direct cloud models and declared cloud helper owners cannot
warm. Do not change the general telemetry locality predicate into a privacy gate.

## Context / why

A synchronization-only reproducer on 70345ae established three defects: warm
had no cancellation carrier, a real foreground stream waited behind the warm
model lock, and shutdown returned while a later classifier warm still started.
The warm path called synchronous generate even though Ollama's stream bridge
already supports cancellation before token one; llama.cpp already has native
abort checks and one context lock. A speculative task must yield ownership when
actual work arrives instead of adding model work to the first-turn queue.

[Handy's released desktop application](https://github.com/cjpais/Handy/releases/tag/v0.9.8)
is a useful lifecycle reference. Its inspected
[model manager](https://github.com/cjpais/Handy/blob/main/src-tauri/src/managers/transcription.rs)
owns background model loading with a mutex/condition and drop guard, distinguishes
requesting unload from actual worker exit, and skips idle unloading while recording.
This informs explicit ownership and completion semantics here; it is not evidence
that this exact source shipped in that release or a transferable latency result.

Earlier CPU three-second classifier warm and twenty-second direct-prefill probes
remain unqualified, and the host was heavily loaded. This change repairs a
reproducible scheduling/lifecycle defect; it does not reinterpret those measurements,
raise deadlines or establish quiet-host CPU performance.

## Consequences

- Accepted partial/final generations retire warm; punctuation/noise rejected before
  generation allocation do not discard useful idle warm work. First real work may
  pay a cold load if it arrives early, with no queued speculative successor stage.
- Cooperative Ollama transport and supported llama.cpp abort retain their existing
  cleanup semantics. Tests cover the actual Ollama bridge through fake SDKs,
  response/client close, subsequent same-client use and unchanged options/residency.
- A blocking foreign generator or engine.warm can outlive retirement. Readiness
  stays unfinished until that call returns; no subsequent stage is admitted then.
  Cancellation is not proof of native kernel preemption or server/model cleanup.
- Unknown injected providers/helpers keep compatibility and are responsible for
  their locality and cancellation contracts. LOCAL_ONLY context is not a network
  sandbox. No cloud call, native model run or audio device is part of this receipt.
- No tiny warm output-cap API is added: the existing short-decision schema is not
  a general warm budget, and changing provider/cache behavior needs separate evidence.
  Other desktop platforms and eventual owner physical validation remain open.
