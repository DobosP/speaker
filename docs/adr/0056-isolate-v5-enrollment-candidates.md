# ADR-0056: Isolate v5 enrollment candidates

Date: 2026-07-12
Status: accepted

## Decision

Prepare each v5 enrollment as a unique feature-worktree candidate before opening
the microphone. Verify the feature `config.local.json` link and its explicitly
named primary target, verify that target names the expected non-empty historical
enrollment, publish a SHA-verified mode-600 adjacent backup without replacing an
existing path, reserve a mode-600 feature-local candidate, then atomically replace
the feature link with a regular config already wired to that candidate. Make
that config carry the worktree/candidate/backup/source inode identities and
frontend-v5 marker; the printed enrollment command must require it. Snapshot
the config, candidate, backup, source, and every directory component before
capture and revalidate immediately before atomic publish. Make enrollment/config
JSON persistence atomic and mode 600, refuse symlink path components, and run
those guards before recorder construction. Refuse any marker-free non-empty
target unless `--replace-enrollment` explicitly authorizes replacement. Keep the historical
reference unchanged until the isolated candidate passes the full live gate;
promotion is a later explicit operation, not part of enrollment preparation.

## Context / why

The feature checkout's ignored local config was a symlink into the primary
checkout. `run_enrollment` followed it and the configured absolute enrollment
path, so another attempt could overwrite both the known v4 biometric reference
and primary local config before v5 had any live evidence. Merely copying the
config first was insufficient: a crash after publishing an unwired copy would
leave a regular config that still named the historical file, and the next
enrollment would accept and overwrite it. A timestamped backup alone was also
insufficient because ordinary open/write can clobber a pre-existing backup or
follow a substituted link.

## Consequences

The preparation command is device-free and never prints config or embedding
contents. Ambiguous targets, existing backup/candidate names, symlinked candidate
ancestors, invalid JSON, or changed files refuse. Once the final config publish
lands, ordinary `python -m core ... --enroll` can write only the isolated
candidate. A preparation failure may leave a verified backup or empty reserved
candidate, but it leaves the feature config symlink guarded and cannot expose the
historical enrollment to a retry. The candidate remains biometric/private local
state and is not committed. `--require-prepared-enrollment` makes a wrong-checkout
invocation refuse before microphone capture.
