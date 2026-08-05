# ADR-0141: Separate EdAcc recordings from conversation families

Date: 2026-08-05
Status: accepted

Supersedes: ADR-0140

## Decision

Retain ADR-0140's complete archive-member grammar, bounded root README,
UID/GID-only base-256 compatibility, raw-header validation, descriptor-safe
extraction, and source/output rebinding. Retain exact segment-derived recording
identifiers as the sole authority for WAV membership: the disjoint dev/test
recording sets must still combine to exactly the complete WAV-stem set.

Interpret each split's `conv.list` as logical conversation families, not as a
second list of recording stems. Validate both segment recording identifiers
and `conv.list` identifiers against the same exact plain-or-one-participant
grammar, then project each identifier to its conversation family by removing
only one terminal `_P[0-9]`; a plain `EDACC-C[0-9]{2,4}` identifier projects to
itself. Require the projected recording and `conv.list` family sets to be equal
within each split. Require dev/test recording sets to remain disjoint and
dev/test conversation family sets to remain disjoint. Multiple participant
recordings may therefore belong to one listed conversation, but `conv.list`
can neither authorize a WAV member nor repair a missing or inconsistent
segment mapping.

Reject every other normalization: multi-digit, repeated, empty, or non-numeric
`_P` suffixes; suffixes in any position other than the end; and conversation
identifiers outside the existing fixed grammar still fail closed. Do not infer
a family from arbitrary punctuation or prefix matching.

## Context / why

The ADR-0140 no-extraction scan completed over all 5,916,732,170 bytes of the
unchanged publisher archive. It validated publisher MD5
`146b4b8026b5d0ce9611667c708456b3`, local SHA-256
`428e5b5d678ee2dba9ec7362878324a2e5fac1f13ac7b7266cacbcade52bb61b`,
layout SHA-256
`a4deda10b3246d8795304a4cb11365f3401e1acbb11fe6210ace61369b54e01b`,
and the complete 98-member/76-WAV inventory without extracting payloads.

A subsequent fresh extraction fully materialized a 94-file,
8,220,494,044-byte private source tree, then failed closed before corpus
publication because two development conversation families exposed ADR-0140's
incorrect namespace assumption: `dev/conv.list` names conversations while
exact segment mappings and WAV stems name participant recordings. The
publisher README describes `conv.list` as the conversation list and the
segment metadata as the audio-file mapping. Requiring both views to contain
identical recording stems therefore rejects a valid publisher relationship.

Allowing unconstrained prefix matching or ignoring `conv.list` would hide
split drift. The one-terminal-single-digit projection is instead the narrow
relationship already admitted by the archive member grammar, while exact
segments remain the authority for every extracted WAV.

## Consequences

- Synthetic and production-layout tests must cover plain and participant
  recording stems, many-to-one family projection, per-split equality, and
  recording/family split-crossing rejection without weakening exact WAV
  coverage.
- Preserve the fully materialized failed source tree and both earlier failed
  trees as evidence. Do not reuse, overwrite, or delete them automatically;
  any retry uses new absent private destinations.
- The complete raw scan now establishes the pinned archive size, publisher
  MD5, local SHA-256, layout digest, and member counts. It does not establish a
  prepared corpus or recognition quality.
- No corpus, model, GPU, microphone, capture, VAD, endpoint, AEC, barge-in,
  device, tool, latency, live-conversation, or runtime-default claim follows.
