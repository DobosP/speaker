# ADR-0057: Reuse the fast client for Postgres memory ingest

Date: 2026-07-12
Status: accepted

## Decision

When the Postgres memory backend receives an actual fast `OllamaLLM`, pass a
cleanup adapter over that same client into `MemoryWriter`. Use the client's model,
host, headers, timeout, options, keep-alive, thinking setting, and Ollama JSON
mode for the off-thread cleanup/substantive-content gate. Do not infer provenance
from a config model string. When the fast client is absent or non-Ollama, disable
the LLM cleanup/gate and retain deterministic junk, confidence, echo,
control-phrase, and dedupe filters. Set MiniCPM5-1B Q8 as the direct
`MemoryWriterConfig` fallback alias; do not reintroduce the pruned live writer
knobs into `config.json`.

## Context / why

`core.app` previously constructed `MemoryManagerAdapter` without a writer config,
so the Postgres-only background path fell through to `llama3.2:3b`. That hidden
third model was not provisioned by the shipped MiniCPM-fast/Gemma-main topology,
could consume additional VRAM, and failed open without surfacing in in-memory,
SQLite, or conversation-evaluation gates. Replacing only the model string was
also wrong: the old cleanup implementation used the module-global Ollama client,
losing a configured host, transport headers, timeout, options, and lifecycle.
Config-string backend detection was rejected because minimal config and actual
factory behavior can disagree.

## Consequences

Postgres ingest no longer silently loads or contacts a third model and custom
Ollama endpoints remain consistent with foreground inference. Cleanup remains
off the bus thread and fails open to the raw transcript if structured output or
transport fails. Non-Ollama profiles lose the optional fuzzy LLM gate rather than
making an undeclared localhost call. This decision does not prove general
semantic recall or change the bounded exact self-fact controller in ADR-0054.
