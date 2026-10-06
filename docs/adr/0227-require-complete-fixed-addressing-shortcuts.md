# ADR-0227: Require complete fixed addressing shortcuts

Date: 2026-10-06
Status: proposed — WIP transport only; candidate qualification pending on Linux

## Proposed decision

Refine the imperative shortcut retained by ADR-0226: accept only the complete
fixed `repeat your previous answer exactly` request, with case/whitespace
normalization and one optional terminator, using fullmatch. Send search/research
subjects, conversation-memory clauses, requested speech payloads, additional
request details and quoted or narrated counterparts to the learned addressing
gate with the complete original utterance.

Honor its ACT/INGEST/UNSURE result and the existing caller uncertainty policy.
Keep complete question shortcuts and complete-token reply parsing unchanged.

## Context / why

The start-only imperative regex admitted synthetic narration such as
`Repeat your previous answer exactly, she read aloud.`,
`Remember for this conversation is the phrase printed in the guide.` and
`Please research shows the result was negative.` without calling the classifier.
Nine conservative runtime controls reproduce the historical bypass: every case
fails because the learned gate was not called. This does not establish the
cause of the owner's recorded room-noise activation.

An end anchor on an unrestricted payload can still swallow narration. Preserve
the full turn for learned classification instead of truncating it or adding a
suffix blacklist. Genuine payload requests remain supported when the learned
verdict is ACT.

## Consequences

- Seventy new synthetic cases cover complete repeat requests, legitimate open
  requests, quoted/narrated turns and conservative runtime admission. Together
  with existing addressing and ambient controls, the planned gate has 136 cases, NOT RUN.
- Open requests now depend on classifier quality and may suffer false negatives.
  Ambient speech identical to the fixed command remains lexically ambiguous.
  Exact ACT mistakes and provider-copied replies remain separate failures.
- This is a text admission repair. Model selection, privacy, enrollment and
  injected STOP authority remain unchanged. Microphone, acoustic quality,
  physical STOP/talk-over and live acceptance remain deferred to owner resume.
- Actual validation commands and results are recorded in `WORKLOG.md`.
