# ADR-0059: Bound memory-writer shutdown

Date: 2026-07-12
Status: accepted

## Decision

Refine ADR-0057 so normal timer and full-buffer flushes may use the configured
fast Ollama client for structured cleanup and gating. Track drained candidates
until persistence commits. At shutdown, revoke any in-flight cleanup ownership,
re-run deterministic admission, and persist eligible raw text without starting
new cleanup/gating inference.

## Context / why

The writer can buffer up to 32 utterances and the reused fast client may have a
60-second request timeout. Running cleanup once per pending item from `close()`
could therefore hold process teardown for many minutes. A timer can also drain
items, block in cleanup, and resume only after the manager has closed its pool.
Dropping those candidates was rejected because shutdown must remain a
persistence boundary, and a shorter hidden timeout would diverge from the
actual configured client.

## Consequences

Process teardown has no cleanup-model timeout multiplier and retains eligible
buffered or in-flight content. Resumed obsolete cleaners cannot touch the closed
persistence backend. Normal flushes keep optional structured cleanup/gating;
items first persisted at shutdown receive deterministic filters only. Backend
persistence may still perform its configured embedding work.
