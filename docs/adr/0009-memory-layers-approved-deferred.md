# ADR-0009: Memory-layers enhancement plan approved, implementation deferred

Date: 2026-06-21
Status: accepted (plan approved; implementation deferred, owner-gated)
Supersedes: none
Superseded-by: none

## Decision
The Hermes-inspired layered-memory enhancement plan
(`docs/memory_layers_enhancement_plan.md`) is **APPROVED as a plan and
DEFERRED as implementation** — the owner chose "stop at the plan"
(2026-06-21). When implementation starts, ship the **deterministic (no-LLM)
baseline FIRST** — frequency+regex consolidation and keyword-cluster
reflection with no model — and add the optional small-LLM (`gemma3:1b`)
extractor/reflector only after the baseline is validated. Every item ships
default-OFF, off the real-time hot path, on-device (§9.7).

## Context / why
An audit found three high-impact gaps in the existing CoALA-shaped memory
stack: no episode→fact consolidation, no reflection layer, and no graceful
forgetting (only an age-TTL cliff) — plus a Postgres/SQLite feature asymmetry.
The design workflow produced a phased plan, but memory is the largest design
surface in the repo and recall quality must be **measured** before more
machinery lands (recall_enabled is still default-OFF pending `tools.bench`
numbers). Deferring keeps the runtime byte-identical; approving the plan
prevents redesign churn. Why deterministic-first: zero extra LLM cost on the
hot path and a falsifiable baseline before any model-driven extraction.

## Consequences
- Agents must NOT start implementing memory layers on their own initiative —
  implementation is an explicit owner go.
- When it starts: deterministic baseline → measure → only then the small-LLM
  tier; entry point is the plan's "first slice".
- The plan doc stays living as the design reference; this ADR records the
  approve/defer decision so the plan is not mistaken for an active work order.
