# ADR-0137: Make speaker-ID runtime allocation truly optional

Date: 2026-08-05
Status: accepted

## Decision

Treat a configured speaker-embedding model as an available capability, not by
itself as an active live-session feature. Allocate the live Sherpa speaker gate
only when at least one configured enrollment reference currently exists on
disk, or when the playback word-cut path that runs without in-app AEC
explicitly selects `barge_word_cut_require_speaker=true`. Resolve those two
conditions through one typed, injectable helper shared by live allocation and
readiness. Re-evaluate the condition on every engine build and clear any prior
gate and warm state before conditional allocation.

Apply strict readiness only under the same activation rule. An active identity
path must have its configured model on disk. The explicit multi-voice word-cut
path must additionally obtain a compatible enrollment and complete the
existing bounded warm-up before capture can run. A configured model, a missing
enrollment path, or a missing model on an otherwise inactive identity path is
advisory and must not allocate the model or block enrollment-free lexical
barge-in. A word-cut setting made unreachable by enabled in-app AEC remains
advisory unless an enrollment reference independently activates identity.

Preserve ADR-0072's behavior and ADR-0042's ambiguity guard. An existing
enrollment reference still activates the gate for normal-final
`speaker_gate_input`, owner verification, and enrolled authority over a
STOP-class control that resembles current or recent own TTS. Compatibility is
still checked against the resolved capture front end after the microphone
opens; an incompatible reference cannot mint authority. First-time enrollment
also remains independent: `python -m core --enroll` constructs the embedding
gate directly and does not rely on live-engine conditional allocation.

## Context / why

Fresh setup records the optional speaker model path even when the owner has not
enrolled and the default word-cut filter is identity-free. The live engine
therefore constructed a lazy speaker gate on every session merely because the
model was configured. Doctor also failed a missing configured model before it
considered whether any behavior could consume identity. That implementation
contradicted ADR-0072's decision that active word-cut can start without a
speaker model or enrollment when its optional multi-voice filter is off, and
it imposed avoidable startup and memory pressure on weaker devices.

Using `speaker_gate_input=true` alone as activation would preserve the same
problem: without an enrollment reference the gate deliberately fails open and
cannot filter a final or grant owner authority. Treating every configured
enrollment pathname as activation would likewise let a stale missing file
resurrect the resource and readiness coupling. File presence is therefore the
earliest useful activation signal; capture-domain and model compatibility
remain the later authority checks they were before this change.

The live engine may be built more than once. Only guarding new allocation
would retain an old gate if an enrollment file disappeared between builds, so
the lifecycle reset is part of the rule rather than an optimization detail.

## Consequences

- Default and freshly configured enrollment-free sessions do not allocate the
  speaker gate, including when the optional model path is present or stale.
- Existing enrollment continues to allocate the gate even when the generic
  word-cut filter is off, preserving normal-final and ambiguous-STOP identity
  semantics. Missing, incompatible, cold, busy, rejected, or errored authority
  still cannot approve an ambiguous STOP or mint owner verification.
- Explicit multi-voice word-cut on its actual no-in-app-AEC path remains
  fail-closed for model presence, compatible enrollment, and warm-up. Enabling
  in-app AEC keeps that word-cut identity setting inactive and advisory.
- Readiness can distinguish an inactive optional artifact from an active
  dependency without importing the large Sherpa engine, and headless tests can
  inject file presence into the exact production predicate.
- No entry point, enrollment format, threshold, barge-in lexical floor, action
  authority, model default, or live-validation status changes. Bare-speaker
  owner A/B remains required.
