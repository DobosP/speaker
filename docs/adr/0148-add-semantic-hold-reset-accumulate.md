# ADR-0148: Add opt-in semantic HOLD reset-and-accumulate

Date: 2026-08-06
Status: accepted

## Decision

Add `endpoint_reset_on_semantic_hold` as a strict experimental Sherpa option
whose default is false. When it is true, require semantic endpointing and a
working VAD. If the installed native recognizer reports an endpoint and the
typed endpoint policy returns `SEMANTIC_HOLD`, retain the stripped native
hypothesis in private `ASRSegment` state and reset only that native stream on
its existing exact-thread owner. Keep the VAD-backed `ASRSegment`, complete
owned PCM, capture scope, acoustic-turn lineage, and already published partial
revision alive as one logical turn.

Compose retained native fragments and the current native result in order with
one ASCII space and no word deduplication. Use that logical text for subsequent
partials, semantic scoring, and the one terminal final. Pass the complete
logical-turn PCM to offline final selection once. Add no new repr, diagnostic,
or persistence surface for retained text; the existing private debug transcript
logging may observe the composed logical hypothesis under its existing access
controls. Erase retained state on every normal segment reset,
capture/generation rotation, gap, KWS terminal, stale handoff, abandoned
episode, decode failure, final, and shutdown path.

Restore the endpoint bounds whose native clocks restart: while held and
VAD-quiet, commit when the VAD-owned total trailing silence reaches the
configured semantic maximum; while held and VAD-active, enforce rule 3 from
the logical turn's first-speech clock. Never carry the pre-reset native
endpoint into resumed active speech. Leave the established path byte-for-byte
logical-text compatible when the option is false, and fail startup rather than
run this option without VAD authority.

Extend transcript-free endpoint replay with an explicit v2 algorithm. Record
the current native endpoint, effective endpoint, held-state bit, logical hard
boundary, logical elapsed time, and a one-based `ENDPOINT_HELD_RESET` ordinal,
but never text. Validation must pair every native `SEMANTIC_HOLD` with an
immediate successful reset and a later commit or typed abort on the same
identity. A KWS command that consumes a held turn records only
`FINAL_ABORTED(reason=command)` to close that evidence lifecycle.

Replace the EdAcc endpoint-integrity report with schema v2 and three sequential
CPU cells: acoustic control, the unchanged Smart Turn HOLD-only diagnostic,
and Smart Turn reset-and-accumulate. Force early commits off in both semantic
cells, publish all three aggregate-safe pairwise final-count matrices, attest
the reset option as false/false/true, and reject a report where the reset cell
never exercises a HOLD. Keep the diagnostic aggregate-only, no-clobber,
private, and non-promotional. Do not add an entry point or change a runtime,
model, detector, threshold, tool, or voice-agent default.

## Context / why

ADR-0147's exact one-thread EdAcc diagnostic proved that Smart Turn could
recognize incomplete hypotheses and issue 34 HOLD decisions, but native
HOLD-only behavior repaired none of the five multi-final rows. Sherpa's native
endpoint remained latched after the semantic policy declined to commit, while
the installed recognizer contract expects a reset after an endpoint. Extending
a fixed silence threshold avoided splits only at 2.4 seconds and therefore
traded turn integrity for conversational delay.

Resetting the native decoder while preserving the existing VAD/acoustic owner
addresses that compatibility boundary directly. Post-hoc transcript or agent
turn joining would erase causal endpoint and tool-turn boundaries, and replaying
accumulated PCM into a fresh native stream would immediately recreate the
latched endpoint. Neither is admitted.

## Consequences

Pure state, deterministic capture replay, full diagnostic-manifest validation,
capture-gap clearing, multiple reset ordinals, KWS termination, async whole-PCM
ownership, rule-3, total-silence, and unchanged-default behavior can be proved
headlessly. The exact three-cell retained-corpus replay remains required before
this branch can land.

Even a green retained replay authorizes only continued opt-in testing. It does
not establish microphone STT quality, far-field timing, room/AEC behavior,
command safety, physical audibility, or natural conversation. Paul must still
run matched bare-speaker control/candidate sessions through the single
`./live.sh` entry and review private recordings before any default can change.
