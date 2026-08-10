# ADR-0176: Make interrupt suite an open-speaker diagnostic

Date: 2026-08-10
Status: accepted

## Decision

Treat `tools.interrupt_suite` as a physical quiet-control diagnostic that
implements ADR-0008's bare-speaker requirement without claiming to complete its
live barge-in gate. Publish one `diagnostic_outcome` object with `status`,
`reason`, `candidate_labels`, `valid_coupled_cells`, `uncoupled_zero_cells`,
`error_cells`, `matrix_partial`, and `live_validation_required`, and make the
process exit code match its status:

- `0` / `quiet-control-candidate` means at least one valid echo-coupled cell
  produced zero self-interruptions and is only a candidate for the live
  talk-over gate. List every unique such label in deterministic lexical order;
  do not select or recommend a winner.
- `1` / `no-safe-candidate` means at least one valid echo-coupled cell ran, but
  none produced zero self-interruptions.
- `2` / `inconclusive` means no valid echo-coupled cell ran. Empty or unknown
  microphone selections and report-publication failure also force this result.

Before publication, count only malformed rows and rows carrying an explicit
error in `error_cells`. Valid uncoupled rows, including zero-self-interruption
rows, do not increment that count. A terminal publication failure adds one
error so its outcome remains partial. `matrix_partial` is true exactly when
`error_cells > 0`; do not infer it from an expected-cell mismatch or an
uncoupled row. Errors in other cells do not override valid coupled evidence, so
a partial run can return either `0` or `1`.
`live_validation_required` remains true for every outcome.

Name the AEC matrix cell `configured-aec`. The cell requests the configured
in-application AEC path; it must not attribute that path to DTLN or promise any
particular active backend.

Keep successful quiet control separate from bare-speaker acceptance. The live
gate remains a real assistant session launched through `./live.sh`, with a human
talking over speaker playback on the bare laptop and an explicit
`python -m tools.live_audio_ab logs/runs/run-<id>.txt` check of that exact retained
run log. Do not use auto-discovery as the evidence binding.

This implements ADR-0008 and supersedes no decision. It changes diagnostic
classification and reporting only; it does not change runtime behavior,
configuration, defaults, AEC selection, DTD, interruption authority, or live
acceptance criteria.

## Context / why

The suite exercises echo-probe cells while the operator remains quiet. Its
useful positive observation is therefore narrow: playback reached the
microphone and a cell avoided self-interruption. The previous command returned
zero even when no safe cell existed, and its fallback suggested headphones,
contradicting ADR-0008's product requirement. It also labelled an AEC request as
`dtln-aec`, although DTLN is not an authoritative description of the configured
canceller path.

A zero-self-interruption count without observed echo coupling is vacuous but is
still a valid uncoupled row, not an error. Malformed rows and explicit cell
errors remain visible, while valid coupled cells can still support the narrow
candidate or no-safe classification. Empty or unknown microphone preflight and
the absence of any valid coupled cell make automation abstain rather than
overstate the diagnostic; no inferred expected-cell mismatch is part of the
classifier.

The suite never supplies the other half of barge-in evidence: a real owner voice
must be admitted during simultaneous assistant playback and cut the intended
reply without a self-cut. That requires the production physical launcher and
the exact run-log analyzer, not the suite's quiet-control matrix.

## Consequences

- Scripts can distinguish a candidate, a negative result from valid coupled
  evidence, and a retry required because no valid coupled evidence exists.
- Every successfully published JSON report and its process exit code carry the
  same closed outcome; a lexically ordered list of unique candidates is nonempty
  only for the candidate state, and no candidate is promoted above the others.
- Publication failure is `inconclusive` with reason
  `report-publication-failed` and exits `2`. It clears `candidate_labels`,
  increments `error_cells`, and makes `matrix_partial` true while descriptive
  valid/uncoupled counts may remain; it cannot leave candidate authority or a
  success code detached from the requested report.
- The report remains diagnostic evidence. Exit `0` does not prove human
  talk-over, barge latency, current-room acceptance, or the complete open-speaker
  requirement.
- The suite still opens physical devices when a person explicitly runs it, but
  the headless verification for this change uses fakes and opens none.
- The fake-only focused gate passes 63 tests.
- The adjacent echo/self-interrupt/verdict/run-analysis gate passes 183 tests.
- The standard APM/DTD gate passes 6 tests; scoped Ruff, format, and whitespace
  checks are green.
- No physical suite, model, network, microphone, audio device, GPU, bare-speaker
  session, `./live.sh`, or `tools.live_audio_ab` run occurred for this change.
  No runtime/config/default/AEC-backend/live result follows.
