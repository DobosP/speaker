# ADR-0168: Define an inactive semantic-interruption policy

Date: 2026-08-09
Status: accepted

## Decision

Add a pure, device-neutral `semantic-interruption-policy-v1` contract under
`always_on_agent`. Represent detector availability, candidate facts, a positive
never-reused current-candidate token, the detector result's candidate token,
takeover score, the three dispositions `take_floor`, `non_interrupting`, and
`abstain`, and a closed decision basis with frozen typed values. Require exact
booleans and integers, a finite float score in `[0, 1]` only for scored
observations, and thresholds with at least one representable float strictly
between them. Scores at or below the configured non-interrupting ceiling
classify as `non_interrupting`; scores at or above the take-floor floor classify
as `take_floor`; the dead band and every missing, failed, token-mismatched, or
non-candidate observation abstain. Never reusing a token makes a delayed result
stale even if a later candidate resembles an earlier one.

Keep this first protocol inactive. It owns no detector, transcript, PCM,
speaker identity, callback, model, thread, clock, telemetry, or effect. Its
opaque token is a freshness binding only, not acoustic evidence or authority.
Do not wire it into an engine, runtime, config, evaluator, resume path, or
default. A `non_interrupting` result is evidence only and cannot keep or restore
playback. Exact STOP, existing barge confirmation, cancel-before-stop ordering,
post-barge response admission, logical-turn ownership, owner verification,
tool/control authority, and memory behavior remain unchanged.

## Context / why

The production barge callback proves a confirmed acoustic interruption and is
already an irreversible cancellation trigger; it does not prove conversational
intent. Endpoint `semantic` decisions answer turn completeness, not whether an
overlapping utterance takes the floor. Reusing either name would conflate two
different safety boundaries.

The exact AMI natural-turn model diagnostic completed its fixed source replay,
but its manual dialogue acts are descriptive and its recognition error is not a
semantic interruption prediction. Public-data research found stronger labelled
sets, but HumDial lacks corpus-scoped terms, English Duplex lacks an accessible
permissive exact release, and IAT/GAP is noncommercial mono human-human speech
without assistant playback. None can authorize a production cut or resume.

A closed pure score contract is still useful now: it fixes the meaning and
failure algebra needed by a future local detector and makes threshold behavior
exhaustively testable without pretending that a model or authority exists. It
also prevents a future implementation from silently treating detector errors,
missing scores, stale playback, or the uncertainty band as permission to act.
The bounded value is deliberately named a score, not a probability. Published
categorical detector states are not calibrated takeover probabilities; any
future version-pinned adapter must define and validate its score mapping rather
than silently mapping an argmax label to `0.0` or `1.0`.

## Consequences

- Scripted/headless tests can validate every score-state, threshold boundary,
  representable dead-band condition, malformed value, stale/ABA candidate
  condition, complete disposition/basis algebra, and privacy surface without a
  model, audio device, corpus, or callback.
- No application behavior improves merely because this contract exists. The
  next admissible slice is shadow-only use of a version-pinned local detector,
  off the audio callback, with aggregate decision telemetry and no effects.
- Post-cut semantic recovery or a reversible pre-cut playback hold needs a new
  ADR, exact detector/runtime closure, assistant-playback/far-reference test
  evidence, stale-generation and cancellation races, and owner bare-speaker
  live A/B. Failure, timeout, uncertainty, and unsupported engines must retain
  the existing committed-barge behavior; exact STOP remains stronger.
- A later semantic result may govern conversational speech continuity only. It
  must never mint identity, direct-instruction status, confirm/mode/tool
  authority, memory writes, or cancellation of an already-started mutation.

## Verification

- The focused pure-policy command in `docs/agent-testing.md` passes 95 tests for
  the complete threshold/state/malformed-input matrix, token freshness/ABA,
  exact output algebra, deterministic side-effect-free behavior, and
  transcript-free fields.
- The exact adjacent command recorded beside it passes 422 tests with three
  existing skips across STOP, acoustic lineage, post-cut admission,
  logical-turn ownership, endpointing, owner/tool authority, and response-only
  resume. The standard APM/DTD regression passes six tests.
- Scoped Ruff check/format and whitespace are green. Two independent read-only
  audits found no code, algebra, wiring, privacy, or documentation blocker.

No model, corpus, GPU, microphone, audio device, runtime, or live test was run
or required for this inactive contract.
