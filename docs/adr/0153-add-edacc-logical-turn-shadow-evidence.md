# ADR-0153: Add opt-in EdAcc logical-turn shadow evidence

Date: 2026-08-06
Status: accepted

## Decision

Extend the exact EdAcc endpoint-integrity evaluator with an explicit
`--logical-turn-shadow` mode. In every acoustic, Smart Turn HOLD-only, and
Smart Turn reset-and-accumulate cell, create one fresh
`LogicalTurnCompositionShadow` per retained row, pass it through the existing
synchronous production-capture seam, close it after the row, and retain only
its transcript-free snapshot long enough to build one aggregate per cell.

Keep the unflagged evaluator byte-shape compatible at schema 2 and kind
`edacc-endpoint-integrity-v2`, without importing, constructing, or passing a
shadow observer. Use schema 3 and kind `edacc-endpoint-integrity-v3` only for
the flagged report. Add a closed protocol marker and limitation in that mode,
and bind `logical_turn.py`, `logical_turn_shadow.py`, and `core/contract.py`
into its source digest. The guarded parent must infer the retained mode from
the strict schema/kind pair and reject disagreement with its requested flag.

Require all 24 observers in each cell to close with no active state, health
error, composition mismatch, or extra/missing legacy terminal. Controls may
report `exact=false` only because they have no multi-epoch evidence, but their
paired raw commits must still cover at least every selected final so a
disconnected or sparse observer cannot publish vacuous clean evidence. Require
the reset-and-accumulate cell to contain a positive multi-epoch commit,
`evidence_sufficient=true`, and exact raw-composition parity. Preserve the
shadow's existing disclaimers for selected-final, async-final, dispatcher,
supervisor, runtime, authority, and live parity.

## Context / why

ADR-0150 deliberately stopped at an opt-in raw-composition observer and
required fresh replay evidence before extending the lifecycle kernel into
later finality layers. The retained 24-row EdAcc corpus is the available real
speech replay that already exposes the relevant defect: acoustic and
HOLD-only cells split five rows, while native reset-and-accumulate repairs only
one. The generic retained AMI capture-replay input is no longer present, and
final-only STT benchmarks cannot prove native epoch composition or exactly-once
logical turns.

Adding a new finalizer or runtime owner before exercising this existing seam
would combine raw composition with later selection and async behavior, making
a mismatch harder to localize. Running the observer only in the reset cell
would also omit matched controls. A global schema bump, production config
flag, transcript-bearing report, per-row identifiers, and any immediate
authority transfer were rejected.

## Consequences

The retained EdAcc replay can now test the ADR-0150 observer against the same
real capture-loop split-turn rows in three matched cells. Strict report
validation covers default isolation, conditional source provenance,
worker/report mode agreement, privacy, capability boundaries, lifecycle
closure, non-vacuous control wiring, and positive reset-cell multi-epoch
evidence.

This is diagnostic infrastructure, not a speech-quality or product behavior
change. It does not improve WER, endpoint latency, conversation, barge-in, or
tool use, and it supplies no selected-final, async, device, microphone,
authority, or live evidence. The coordinator remains inactive; production
entry points, models, thresholds, configuration, defaults, and owner policy
remain unchanged. Later work may extend the shadow through selected-final and
async terminal delivery only after this raw seam is clean, while matched owner
`./live.sh` evidence remains required before any authority move.
