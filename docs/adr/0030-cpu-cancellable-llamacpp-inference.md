# ADR-0030: CPU-cancellable llama.cpp inference

Date: 2026-07-11
Status: accepted

## Decision
Pin the on-device backend to `llama-cpp-python==0.3.33` and fail readiness
closed unless its audited abort/memory symbols are present and
`llm.n_gpu_layers == 0`. Serialize each shared `LlamaCppLLM` context behind a
cancellable admission lock, retain one native abort callback for the context,
and point it at only the lock owner's task cancellation event. Cancellation
wins completion/error races. Before releasing an interrupted context, clear its
native memory and reset the Python client; if either cleanup step fails, poison
the context and require a runtime restart instead of admitting more inference.
Keep the ADR-0021 task bulkhead as the outer bound.

## Context / why
ADR-0021 could retire a barged-in coordinator, but an in-process llama.cpp call
kept its provider slot and the sole shared phone context until native inference
returned. Repeated canceled waiters could therefore consume the bulkhead and
later begin stale work. High-level token stopping cannot interrupt chat prompt
evaluation before token one. The audited v0.3.33 C surface exposes
[`llama_set_abort_callback`](https://github.com/abetlen/llama-cpp-python/blob/v0.3.33/llama_cpp/llama_cpp.py#L3181-L3195),
but it is CPU-only; an aborted decode can leave processed ubatches, so recovery
also needs `llama_get_memory`/`llama_memory_clear` plus `Llama.reset()` while the
context lock is still held.

A persistent spawned model worker was considered as the stronger kill boundary.
It would also contain model-load deadlocks, crashes, and OOM, but adds mobile
process/IPC semantics and a full GGUF reload after every normal barge. The
smaller native boundary was accepted only after the pinned CPU binding and the
actual MiniCPM5-1B Q4 model passed the committed headless pre-token stream gate: 22.4 ms
from signal to exit, zero stale pieces, and a healthy completion on the same
context. Process isolation remains the fallback if that recovery gate regresses
or hard containment becomes necessary.

## Consequences
Default local-only `phone`/`phone_lite` inference now releases native work and
provider capacity on barge-in, task timeout, or shutdown after model
construction. Canceled lock waiters never become owners; ABI drift and GPU
offload refuse native sherpa startup and fail at first inference in engines that
skip preflight. Desktop Ollama behavior is unchanged. Deterministic
tests cover cancellation races, repeated barges beyond the task cap, timeout,
shutdown, cleanup poisoning, and healthy reuse; real-model CI exercises native
abort plus same-context recovery.

Model construction/startup warm remains cooperative-only because no context
exists on which to register the callback. A native deadlock may ignore the
callback. Cloud-enabled Hedge uses a separate loser-stop event that is not yet
bridged into this callback; shipped phone profiles keep cloud disabled. These
limits stay behind ADR-0021's bulkhead and must not be described as hard-killed
work or live phone/audio validation.
