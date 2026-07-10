# ADR-0021: Bound synchronous provider invocations behind cancellable tasks

Date: 2026-07-10
Status: accepted

## Decision

Run every synchronous capability invocation in a daemon provider thread behind
a `TaskRuntime` bulkhead capped at `max_active_tasks`. Keep the registered task
thread as a coordinator that polls its `cancel_event` and provider completion.
On barge-in, supersede, stop, or watchdog timeout, retire that coordinator and
publish cancellation without waiting for a blocked provider. A provider that
does not cooperate retains its bulkhead slot until it actually returns, so a
cancellation storm cannot create unbounded abandoned calls. A coordinator
cancelled while waiting for a slot must never start its provider later.

Treat cancellation as higher priority than simultaneous completion, retain the
speech-epoch and emitter gates for stale output, and prohibit a late error from
a cancelled answering tier from starting a cross-tier retry. Atomically reserve
one terminal lifecycle event per task, require nested ReAct/model/tool starts to
claim admission against cancellation, and bind delayed first-token metrics to a
monotonic token captured from the originating ASR turn.

## Context / why

The token consumers previously checked cancellation only after `next()`
returned. An Ollama, cloud, llama.cpp, ReAct, or other synchronous capability
blocked before its first result therefore held the registered task worker even
after the supervisor removed the task. Six such turns exhausted the global task
budget and left later prompts queued. Watchdog reaping repaired supervisor state
but did not repair `TaskRuntime.active_count`.

Moving only the LLM token loop would leave blocking `generate()` and other
capabilities with the same fault. Adding a cancellation argument to `LLMClient`
would also claim a guarantee that arbitrary synchronous/native providers cannot
meet. In the installed synchronous Ollama client, closing a generator while its
first `next()` is executing is unsafe and closing the shared client does not
reliably wake that read. Process isolation could hard-kill a native stall but is
not a safe drop-in for shared in-process model contexts and side-effecting
capabilities. The task-boundary coordinator is the smallest provider-agnostic
control-plane guarantee; transport-native abort remains an additive follow-up.

## Consequences

- Barge-in and deadline reaping check coordinator cancellation on a 10 ms polling
  cadence, even if the provider has not produced token one; OS scheduling can add
  latency, so 10 ms is not a hard response-time promise.
- Live uncooperative provider calls are bounded by `max_active_tasks`; once all
  slots are occupied, new coordinators wait cancellably instead of spawning more.
- Late tokens cannot reach TTS or stamp a replacement turn's TTFT/watchdog
  metrics, and late provider/tool returns cannot start follow-on work after cancel.
- Python still cannot kill arbitrary native work. A permanently wedged provider
  can retain one bounded slot forever; exhausting every slot prevents useful
  inference until one returns. Async Ollama task cancellation and process-isolated
  llama.cpp are future hard-abort options, not guarantees of this decision.
- A side-effecting provider already admitted may finish after its task is
  cancelled. Existing origin/owner gates remain mandatory; waiting coordinators
  are cancelled before they can admit a late side effect.
- Deterministic headless tests cover pre-token barge-in, over-cap storms, timeout
  reaping, late stale tokens, cross-tier suppression, and recovery. No live audio
  or human validation is implied.
