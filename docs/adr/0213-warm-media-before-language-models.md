# ADR-0213: Warm local media before language-model generation

Date: 2026-10-04
Status: accepted

## Decision

Run the existing best-effort engine warm-up first in VoiceRuntime's existing single startup
worker, followed by local fast/main models, addressing and cleaner in their existing order.
Preserve local-only model filtering, identity deduplication, legacy generate TypeError fallback,
per-stage failure isolation and finally-signaled warm_ready. Add no concurrent warm thread,
model, inference, media callback work, cloud request or new readiness authority.

## Context / why

The prior startup sequence completed several full language-model generations before warming
TTS. On the desktop streaming RMS path, that warm step also seeds the first fragment's gain;
waiting behind LLM output leaves early replies on the cold/whole-clip fallback precisely when
the owner expects a snappy response. Native media preparation should not queue behind a full
text response. Reordering the existing worker removes that scheduling dependency without
increasing peak native inference concurrency during warm-up.

## Consequences

A slow LLM warm stage no longer prevents the engine's earlier warm step from completing.
The overall warm_ready event still means all selected warm steps finished best-effort, not
that every stage succeeded. Engine warm failure remains nonfatal and later warm stages run.
There is no new hard deadline or shutdown guarantee for an entered native warm call.

The readiness/runtime alignment gate passes 70 tests. New deterministic cases cover media
completion before a held LLM, single-worker order, independent stage failures, legacy fallback
success/failure and eventual readiness. Scoped Ruff and whitespace checks are clean.
No native model timing, microphone, physical speaker or phone measurement follows. Normal
engine warm-up still does not exercise its primary final SenseVoice recognizer; that separate
seam and sustained total CPU budgeting remain open. ADR-0214 records the performance plan.
