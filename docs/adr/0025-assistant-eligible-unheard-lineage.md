# ADR-0025: Unheard continuation lineage requires assistant eligibility

Date: 2026-07-11
Status: accepted

## Decision

Retain a committed but supervisor-unparsed final as an unheard continuation
origin only when the runtime is already in ASSISTANT mode, the exact shipped
deterministic analyzer is installed, and it can still resolve the text as a
conversational ASSISTANT turn. Exclude built-in stop, confirmation/denial,
mode-switch, search, research, dictation, and command forms before publishing
that lineage. Keep this preview pure and bounded; do not call a model, injected
classifier, memory backend, or task analyzer from the runtime terminal section.

## Context / why

The runtime snapshots an assistant final immediately before publishing it to the
event bus so a quick add-on can preserve context across the publish-to-task gap.
At that point the supervisor has not parsed explicit intent. The old guard used
only runtime mode and optional capability-route action, so a runtime without the
capability router could queue `research quantum computing`, then let `make it
shorter` steal it as assistant continuation lineage. The research turn was
never admitted; one synthetic assistant prompt containing both texts was spoken.

Parsing through the supervisor early was rejected because that analyzer call
does not belong inside runtime terminal ownership and an injected analyzer may
block or carry state. Dropping all publish-gap lineage was also rejected because
ordinary quick add-ons would again lose their original question. A pure preview
of the shipped analyzer's decisive non-assistant forms preserves the useful
handoff without guessing across modes.

## Consequences

- Bus-backlogged explicit search/research/dictation/command turns cannot become
  assistant continuation origins; newest-input generation fencing handles a
  later turn normally.
- Ordinary assistant finals still retain lineage across the publish-to-task gap.
- Pending bare confirmations/denials and control/mode phrases are ineligible.
- Custom analyzers fail closed, and passive wake-activated gap lineage remains a
  separate behavior decision rather than inheriting assumptions from the default.
- New built-in explicit intent syntax must update both the analyzer and this
  pure eligibility preview, with a regression test pinning their agreement.
