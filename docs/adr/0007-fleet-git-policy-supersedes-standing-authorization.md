# ADR-0007: Fleet git policy (2026-06-24) — supersedes the 2026-05-30 standing authorization

Date: 2026-06-24
Status: accepted

## Decision
Adopt the fleet-wide git operating model (AGENTS.md, added 2026-06-24;
source: agent-ops docs/08): commit locally when the work is complete and the
logic suite is green; **never push, merge to `main`, or delete branches
without Paul's explicit ask**. One task = one branch = one worktree. This
REVOKES the 2026-05-30 "standing session workflow (durably authorized)"
grant that let sessions merge to `main`, push, and delete branches
autonomously.

## Context / why
The old CLAUDE.md block explicitly said it "supersedes the older
only-when-asked stance", so it survived 25+ days after the fleet no-push
policy replaced it — agents kept obeying the newest-sounding text. Recording
the reversal as a dated decision (per the 2026-07-02 doc-governance rules) is
what prevents another silent flip: any future re-grant of git autonomy must be
a new ADR superseding this one, not a prose edit. Why not per-repo autonomy:
cross-repo agents inherit whichever repo's rule they read last; the fleet
standard makes the answer uniform everywhere.

## Consequences
- A finished session ends with a green local branch and a summary, not a
  merge; Paul integrates (or explicitly asks for a push/PR).
- On the Windows box, landing happens via PR per
  `docs/windows_landing_workflow.md` (note: the local guard hook does not yet
  technically block `main` pushes — policy, not enforcement).
- Prior session notes describing autonomous merge/push flows are historical;
  never obey them.
