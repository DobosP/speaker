# ADR-0022: Cancel Ollama streams through per-request async tasks

Date: 2026-07-10
Status: accepted

## Decision

Run production `OllamaLLM.stream()` calls through the official
`ollama.AsyncClient`, with one short-lived client and event-loop thread owned by
each stream. Bridge that request back to the synchronous `LLMClient` iterator
through a bounded queue. Snapshot the task's cancellation event when the stream
is created, cancel the async request task before or between tokens, discard any
queued output after cancellation, and close the response plus client before the
provider invocation is considered finished.

Detach producer ownership from the public iterator so explicit close, Hedge
loser retirement, task cancellation, and garbage collection of an abandoned
iterator reach the same cancellation state. Keep `cancel()` nonblocking for the
explicitly marked cross-thread Hedge seam; make the owning consumer's `close()`
wait for cooperative cleanup, propagate wrapper/ReAct closes to the provider,
and join marked Hedge workers through cleanup before their outer provider slot
returns. An incidental `cancel()` method is not sufficient. Keep injected synchronous
clients on the compatibility path, and keep non-streaming `generate()` calls on
the existing synchronous client. Require `ollama>=0.6.2` for this contract.

## Context / why

ADR-0021 made the task coordinator cancellable and bounded abandoned provider
calls, but it could not stop the provider computation itself. The installed
synchronous Ollama generator cannot safely be closed from another thread while
its first `next()` is running, and closing its shared client did not reliably
wake that read. That left the primary MiniCPM/Ollama voice path consuming a
bulkhead slot until a token or socket timeout arrived after barge-in.

Ollama's async client exposes the required ownership boundary: cancellation of
the task awaiting `chat()` or its async iterator unwinds the HTTP request. A
per-request loop/client was chosen over a process-global loop and shared async
transport so stream shutdown cannot cancel a sibling or leave a persistent
event-loop lifecycle for runtime shutdown to coordinate. Process isolation is
unnecessary for this HTTP boundary and would add substantially more state and
startup cost; it remains relevant to in-process llama.cpp.

Hedge also had two adjacent races: a fallback source retired at its TTFT
deadline could enqueue a late token and become the winner, and its generator
captured `capability_context` only when first iterated. Retired-source messages
are now ignored and the turn context is bound when `stream()` is called.

## Consequences

- Cooperative Ollama SDK requests now unwind before the provider slot is freed;
  deterministic tests cover cancellation while awaiting `chat()`, before token
  one, after token one, during client construction, under queued backpressure,
  across concurrent streams, and through the voice-runtime barge path.
- Every streaming request pays for a short-lived thread and async client. This
  avoids shared-transport cancellation races at the cost of small setup overhead.
- Dropping a sparsely consumed iterator cancels its detached producer; a bounded
  queue also self-cancels a prolific producer whose consumer stops draining.
- Cleanup exceptions, including `CancelledError`, cannot hide the original
  provider failure or skip later client-close/done signaling.
- Hedge keeps its bounded join for unknown synchronous workers, but waits for an
  advertised Ollama/safe worker's cooperative cleanup before returning the outer
  provider invocation.
- This is SDK-native cooperative request cancellation, not a transport hard-kill
  or a general ability to kill arbitrary Python/native work. A task suppressing
  `CancelledError`, hanging `aclose()`/`close()`, synchronous `generate()`,
  injected sync client, cloud SDK, or llama.cpp call can still retain its slot
  and relies on its timeout plus the ADR-0021 bulkhead/process boundary.
- Headless lifecycle coverage and a real local Ollama request/cancel/recovery
  probe do not constitute human-speech, microphone, or talk-over validation.
