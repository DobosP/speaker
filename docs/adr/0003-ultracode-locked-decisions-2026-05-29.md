# ADR-0003: Ultracode locked decisions — OpenRouter US-default, SearXNG, SQLite-mobile/Postgres-desktop

Date: 2026-05-29
Status: accepted
Supersedes: none (first cloud-provider / web-search / memory-backend decisions)
Superseded-by: none

## Decision
Lock three platform choices from the ultracode overhaul
(`docs/archive/ultracode_scope.md`, "Locked decisions 2026-05-29"):
1. **Cloud LLM provider:** OpenRouter behind the existing `OpenAICompatLLM`
   (one key, many models); **US-hosted chains are the default**; PRC-hosted
   presets (DeepSeek/Moonshot) are gated behind an explicit config opt-in
   (`allow_prc`, plus never in a default chain).
2. **Web search:** self-hosted **SearXNG** behind the pluggable `web.search`
   CapabilityProvider; every query routes through sensitivity classification
   (the §9.7 egress gate) before any network call.
3. **Memory backend:** **SQLite + sqlite-vec on mobile, Postgres + pgvector on
   desktop**, both behind the one Memory Protocol; desktop defaults to in-RAM
   unless `DATABASE_URL` is set.

## Context / why
The overhaul added a cloud thinking tier and real research capability without
breaking the §9.7 boundary (ADR-0001). OpenRouter gives one key across models
with US hosting by default — a PRC endpoint must be a deliberate double opt-in
so a mis-classified personal turn cannot cross by default. SearXNG is keyless
and self-hosted, so search egress stays under the owner's control. One memory
protocol over two engines keeps mobile shippable (no Postgres on-device) while
desktop gets pgvector recall. Why not a single hosted vector DB: raw
transcripts are personal data; they stay in owner-controlled stores.

## Consequences
- `config.json` today ships exactly this shape: `openrouter_*` presets host=US,
  `allow_prc: false`, CN presets never in `cloud_chains` defaults;
  `web_search.enabled=false` with a SearXNG `base_url` (real instance is
  owner-gated infra); `memory.backend: "auto"` (Postgres only on
  `DATABASE_URL`, SQLite path for on-device persistence).
- Changing provider/host policy or the memory split needs a new ADR, not a
  config drive-by.
