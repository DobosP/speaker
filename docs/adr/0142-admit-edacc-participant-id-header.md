# ADR-0142: Admit EdAcc's exact participant-ID header

Date: 2026-08-05
Status: accepted

Supersedes: ADR-0141

## Decision

Retain ADR-0141's complete archive grammar, bounded metadata, raw-header
validation, descriptor-safe extraction, exact segment-to-WAV authority,
conversation-family projection, split-disjointness checks, and source/output
rebinding. Add exactly `PARTICIPANT_ID` to the fixed participant-column aliases
accepted by the EdAcc linguistic-background CSV parser.

Keep header selection exact. Do not trim whitespace, change case, perform
Unicode normalization, or use prefix, substring, or fuzzy matching on column
names. Retain the existing exact accent-column aliases and all participant,
accent, duplicate, row-shape, row-count, field-length, CSV, and text parsing
rules unchanged.

## Context / why

The ADR-0141 non-production exact-source preflight reused the preserved complete
private source tree. It passed the full pinned archive, extracted layout,
segment-to-WAV, conversation-family, and split validation before reaching
`linguistic_background.csv`. The pinned 28-column publisher file names its
participant column exactly `PARTICIPANT_ID`, which was absent from the fixed
alias tuple. The preparer therefore failed closed before audio decode or corpus
publication, and the new output parent remained empty.

This is a source-interface spelling mismatch, not evidence that header
normalization is safe. Adding only the observed exact publisher spelling keeps
malformed or drifted headers visible while allowing the pinned source to
continue through the existing parser.

## Consequences

- Tests must accept the exact uppercase publisher field while rejecting
  case-varied, whitespace-padded, and otherwise normalized imitations. Existing
  aliases and all downstream accent, duplicate, and row validation remain
  covered.
- Preserve the complete preflight source and every earlier failed extraction or
  output tree. Do not reuse, overwrite, or delete them automatically; every
  production retry uses new absent private destinations.
- The preflight establishes only that the exact source reached this CSV-header
  boundary. It does not establish successful audio decoding, corpus
  publication, source quality, or recognition quality.
- No corpus, model, GPU, microphone, capture, VAD, endpoint, AEC, barge-in,
  device, tool, latency, live-conversation, or runtime-default claim follows.
