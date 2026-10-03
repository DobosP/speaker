# ADR-0210: Bind barge-in onset grace to rendered playback samples

Date: 2026-10-04
Status: accepted

## Decision

Measure the existing playback-onset echo grace from the first output samples of each reply,
with the synthesis-admission timestamp retained as the pre-audio fallback. Carry the engine's
playback generation with FIFO sample spans, preserving it through flush/fade and resampler tail.
Read consumed generation and first offset under the FIFO's existing copy lock and publish one
immutable generation/timestamp onset fact from the output callback. A capture snapshot may use
only facts matching its playback generation. Do not restart reply grace for later fragments or
application dry gaps, and retain the existing STOP, authority and interruption policies.

## Context / why

The owner reports repeated self-interruption and slow/stuck turns. A deterministic reproduction
on the previous implementation admits simulated echo 100 ms after playback begins despite
400 ms of configured onset grace: slow synthesis consumed that grace before any audio played.
The coherence/reference buffer is least mature precisely at that first playback block.

A timestamp based only on the per-fragment first-audio metric would incorrectly restart grace
between sentences. A generation sampled before FIFO read is also insufficient: stop/new reply
can replace PCM before the copy lock is acquired. Sample-span ownership covers that race and
a callback containing both retained predecessor fade and successor samples. The callback's
render timestamp does not prove acoustic DAC/speaker latency; physical route validation remains
separate. No new blocking callback lock, model, threshold or enrollment permission is added.

## Consequences

Slow synthesis retains a full rendered-onset protection window. Exact controls and synthesis
lead-in grace remain available; legitimate later talk-over still works after grace expiry.
Eight new deterministic regressions cover slow/empty output, STOP, fragments/dry gaps, capture
snapshots, before/after-read replacement and retained fade mixed with successor PCM.

The affected headless gate passes 479 tests with one inherited missing-DTLN-model skip,
including the required six APM/double-talk cases. Scoped Ruff and whitespace checks are clean.
Tests use synthetic PCM and do not establish effective live AEC, owner recognition or actual
speaker latency. The owner bare-speaker ./live.sh A/B remains required, including delayed first
synthesis, exact STOP, ordinary talk-over and route restoration. ADR-0209's startup/enrollment
and model-template restrictions remain in force.
