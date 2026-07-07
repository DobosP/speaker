# ADR-0014: Direct main landing allowed during development

Date: 2026-07-07
Status: accepted (owner decision, Paul)

## Decision

Direct merge + push to `main` is allowed from any of Paul's machines — including
this Windows box — during the current development phase, provided the logic
suite is green (`python -m pytest tests -q`). This partially supersedes
**ADR-0007** and the PR-only Windows rule of 2026-07-02: feature branches remain
the norm for in-progress work, but landing no longer requires a PR or an
explicit per-push ask.

The guard hook (`.claude/hooks/guard.ps1`) keeps its work-identity/SSH
protections; only the `git push … main/master` deny rule was removed.

## Rationale

The fleet-wide 2026-07-07 merge-audit sweep showed the PR detour adds friction
without review value while Paul is the only reviewer and drives sessions
directly. Fleet policy (global CLAUDE.md, same date) now allows direct main
landing across all personal repos.

## Revisit

Reinstate the push guard + PR flow when the project reaches release hardening
or gains a second contributor.
