# ADR-0066: Promote accepted v5 enrollment without replacing v4

Date: 2026-07-13
Status: accepted

## Decision

Complete ADR-0056 with a separate, device-free promotion transaction that runs
only after explicit operator acceptance of the complete live gate. Preparation
schema v2 binds the isolated config to the exact worktree, primary config,
candidate reservation, historical source, and backup. For each bound file it
records device/inode, size, mtime, ctime, mode, owner/group, link count, and
SHA-256 content identity. Promotion obtains bytes and metadata through one
no-follow file descriptor, revalidates the full lineage, and requires both
configs plus the source, backup, populated candidate, and accepted copy to be
path- and inode-disjoint.

Accept only a candidate that atomically replaced the empty private reservation,
uses the prepared model and integer sample rate, matches the historical
same-model embedding dimension, contains a finite unit embedding from at least
three passes, and carries exact front-end-v5 non-raw provenance.

Derive the accepted filename exactly from the candidate ID and place it beside,
but never over, the historical enrollment. Exclusively publish an independent
mode-600 byte copy, or adopt an existing file only when it is a current-user,
single-link, independent, exact byte match. Strictly sync temporary files,
accepted/config files as applicable, and their directories; a failed required
sync never reports a normal success.

Hold a stable current-user mode-600 advisory lock from primary-config validation
through its atomic replacement. This serializes cooperating promotion commands;
it cannot exclude a process that ignores the lock. Revalidate protected state
immediately before replacement, but document the non-cooperating check/replace
race instead of claiming a stronger filesystem guarantee.

Treat primary-config replacement plus its directory sync as the successful
commit point. Use these CLI results:

- exit 0: the accepted copy is durable and the primary pointer committed;
- exit 2: the invocation refused before making an accepted/config commit;
- exit 3: an exact independent accepted copy is confirmed durable while the
  original primary config is confirmed unchanged and inactive; an identical
  retry may adopt the orphan;
- exit 4: publication, cleanup, config replacement, or required durability is
  ambiguous, so inspect the supplied paths before retrying.

After exit 0, perform no fallible state read that could turn a committed result
into a false refusal.

## Context / why

ADR-0056 intentionally stopped before activation, but its schema-v1 marker did
not bind all protected content and metadata needed for a safe later promotion.
Path checks and partial inode snapshots do not detect same-size mutations,
hard-link changes, or path ABA races. A split snapshot/read can also pair bytes
from one inode with metadata from another.

Overwriting `enrollment.json` would destroy the strongest rollback reference,
while rewriting config before accepted bytes are durable could activate a
partial or absent file. Enrollment and config are separate filesystem objects,
so one cross-file atomic operation is unavailable. Publishing a verified copy
first leaves v4 active on a pre-config interruption; explicit staged and
ambiguous outcomes keep that unavoidable boundary truthful.

## Consequences

- Promotion opens no microphone, speaker, model service, or network connection.
  It prints supplied file paths for operator review, but no config contents,
  embedding values, digests, or other file contents.
- Historical v4, its verified backup, and the isolated candidate remain intact
  as rollback/evidence files. Only the primary enrollment pointer changes.
- A stale schema, altered lineage, aliases, symlinks, unsafe ownership/mode/link
  count, mismatched model/rate/dimension, raw provenance, weak pass count, or
  malformed embedding fails closed.
- Deterministic tests cover preparation/promotion lineage, path ABA and content
  races, all six inode aliases, cooperative locking, strict fsync failures,
  exact orphan adoption, exit 3 confirmation, exit 4 ambiguity, and the
  no-fallible-read-after-commit boundary.
- Headless verification does not accept biometric state. A fresh v5 enrollment
  and the complete bare-speaker live gate remain pending before this command may
  modify the real primary config or accepted-enrollment path.
