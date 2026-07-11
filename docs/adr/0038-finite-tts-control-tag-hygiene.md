# ADR-0038: Sanitize only finite TTS control-tag namespaces

Date: 2026-07-11
Status: accepted

## Decision

When expressive TTS markup is enabled, consume an unsupported leading bracket
only when its complete body is one bounded, atom-shaped `namespace:value` or
`namespace=value` pair and its namespace is `tag`, `narrator`, `role`,
`speaker`, `style`, `tone`, or a configured voice/emotion name. Strip that
control text without producing a directive, so it cannot establish or switch a
reply voice. Apply the shared sanitizer at runtime admission, Sherpa playback,
Sherpa's final synthesis guard, and FileReplay playback/receipt generation.
Detect later-sentence sanitation by visible-text change rather than by the
presence of a valid directive.

Preserve free-form brackets and unknown namespaces byte-for-byte, including
`[citation needed]`, `[citation:needed]`, and `[chapter:one]`. Keep all behavior
behind the existing `tts_markup` opt-in.

## Context / why

Live run `20260711-144211` showed MiniCPM/Gemma fragments beginning
`[tag:story]`, `[tag:narrator]`, and `[narrator:deep]`. The existing parser did
not recognize those as valid `voice`/`emotion`/`rate` directives, so runtime
history and Sherpa synthesis retained and spoke the bracket text. Voice
continuity correctly carried the prior voice, which made these malformed tags
especially likely to survive on later streamed fragments.

Stripping every leading bracket or every unknown `x:y` pair was rejected: it
would silently delete listener-visible citations, notation, or literal examples.
Interpreting malformed aliases was also rejected because a model typo must not
invent a voice change. A finite namespace plus atom grammar covers the observed
control leak while keeping the semantic boundary explicit and testable.

## Consequences

- The observed unsupported tags cannot reach TTS, playback receipts, resume
  state, or assistant conversation history.
- A malformed control tag inherits any already-established reply voice but
  cannot mutate it; unstyled replies retain the configured default.
- New model-control namespaces require an explicit allowlist addition and tests.
  Unknown bracket forms remain audible by design rather than risking deletion
  of genuine listener text.
- Pure parser, runtime, Sherpa, and FileReplay tests prove the headless contract.
  This change does not claim a new microphone, speaker, barge-in, or ear-quality
  validation.
