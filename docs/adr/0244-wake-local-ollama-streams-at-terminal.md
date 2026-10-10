# ADR-0244: Wake local Ollama stream consumers at terminal

Date: 2026-10-10
Status: accepted

## Decision

Publish a private nonblocking queue terminal notification after the local
Ollama request has finished its response/client cleanup attempts and its private
asyncio loop has returned. Drain a known-complete queue without another timed
wait. Recheck the queue after observing completion when a timed empty read races
the final token/error. Preserve the 64-item bound, ordered output, original
provider exceptions, cancellation precedence and request-owned client lifecycle.

## Context / why

The async bridge already bounded its queue and could cancel before the first
token. However, natural completion set a separate Event without notifying a
consumer blocked in Queue.get. Even when cleanup was already complete, the final
next() waited the full 50 ms fallback poll. A public fake-local probe reproduced
50.168–50.239 ms across five cases on pristine 1bed9e2. Callers needing natural
exhaustion to finish a classifier or emit a short unpunctuated tail incurred that
wait after the provider had finished. No native model or network is needed to
demonstrate this scheduling defect.

One terminal notification is sent after private loop return. If all 64 queue
slots contain data, notification is skipped without waiting for space or dropping
data; the consumer later drains the queue using the done flag. A timed-empty
observation is retried after done, so data/error publication immediately before
completion cannot be mistaken for exhaustion. A queued error is terminal too:
the consumer waits for exact producer completion before exposing it, and a
cancellation during that settlement revokes the error/output as before.

## Consequences

Successful output still streams before cleanup completes. Only terminal
notification/error settlement requires completion. Explicit cancellation remains
nonblocking; owning close still waits for the entered producer. A foreign cleanup
or loop shutdown that never returns can still hold that ownership indefinitely;
there is no hard timeout, thread kill, native preemption or uncertain-slot release.
Cleanup-error handling is unchanged, so completion does not assert that a closer
which raised succeeded. Injected sync compatibility, client construction,
providers, models, output/context caps, residency and locality policy are unchanged.

The [compact paired synthetic receipt](../evidence/ollama-terminal-wakeup-2026-10-10.json)
binds the pinned baseline, both bridge/file hashes, fake SDK fixture and benchmark
source. Five adjacent alternating cases per variant measured terminal next()
after known cleanup: baseline p50 **50.1314 ms** (50.0995–50.2107), current
**0.0090 ms** (0.0042–0.0099). Only the two pinned repository bridge classes are
recreated with the same current public fixture/owner; this is not a historical
environment reconstruction. It is a fixed queue-wait removal, not a model
throughput, whole-turn, playback, phone or physical latency result. CPU isolation
was not established and no real transport request occurred.

The final focused lifecycle gate passed **28 tests in 1.23 s**. The adjacent
source-owner, hedge, decision, fake native-abort, startup warm, cancellation and
locality gate passed **302 tests in 7.24 s**. Independent bridge review returned
GO. Tests cover empty/sparse/full queues, notification while waiting, last-token
and error races, loop errors, blocked cleanup and cancellation with a full queue.
No daemon, socket, cloud, native model, private recording, microphone, doctor or
device ran. Linux/Windows/macOS hardware and live gates remain open.
