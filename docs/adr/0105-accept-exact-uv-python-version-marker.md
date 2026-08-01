# ADR-0105: Accept the exact uv Python version marker

Date: 2026-08-01
Status: accepted

## Decision

Accept exactly one recognized interpreter-version entry in a Faster-Whisper
candidate's complete `pyvenv.cfg`: either stdlib/virtualenv's
`version = 3.12.3` or uv's `version_info = 3.12.3`. Reject both forms together,
duplicate recognized entries, wrong values, and malformed recognized lines
before creating the provision output. Continue binding the complete unchanged
marker into the schema-v6 manifest.

## Context / why

ADR-0102 requires an isolated Python 3.12.3 runtime, but its first provisioner
recognized only the `version` spelling used by its synthetic fixture. The
materialized local runtime was created by uv 0.11.23 and correctly records
`version_info = 3.12.3`, so the otherwise compatible runtime failed before a
receipt could be created. Rewriting the local marker would manufacture a
non-native environment description, while accepting arbitrary aliases or
multiple declarations would make the interpreter-version claim ambiguous.

## Consequences

Exact uv-created and stdlib/virtualenv-created Python 3.12.3 environments can
reach the existing receipt and runtime checks without weakening version
strictness. The complete marker, runtime tree, interpreter target, and model
tree remain content-bound, owner-private where required, and subject to the
same no-overwrite and zero-output-mutation contracts. This compatibility change
runs no candidate, corpus, GPU, audio device, or live session and does not alter
runtime defaults, setup, `./live.sh`, or `python -m core --session`.
