# ADR-0236: Keep one rolling owned experiment setup

Date: 2026-10-10
Status: accepted

## Decision

Apply Paul's latest explicit retention instruction: check disk before substantial
runs, retain one reproducible verified setup and compact results, and prune owned
superseded transient copies after verifying the successor. Preserve original
baselines, canonical release/model keepers, source branches, unmerged source
packets and other owners' work. Unmerged branch deletion still requires specific
human authorization. Do not treat a historical inventory as proof its bytes remain.

## Context / why

Prior integration copied every task's generated log tree before removing merged
worktrees. Accumulating complete snapshots, synthetic fixtures and caches for each
iteration consumes storage without improving reproducibility. Git retains source;
a compact command/result/source manifest plus canonical baseline assets supports
continuation. The current human instruction supersedes older keep-every-capture
practice only within the owned transient scope, not the protected enrollment,
recordings, original model baselines or unmerged mobile work.

## Consequences

Before the second desktop pass the filesystem reported 169 GiB available; free
space also changes with other owners' work. The previous full-gate result hash
was verified against its retained 12192-pass receipt. All 80 duplicated synthetic
identity arrays in the owned archive matched canonical kept arrays by SHA-256.
Only `logs/runs/desktop-core-integration-20261010/worktree-logs` and `worker-logs`
were removed: **657 files, 49,690,305 logical bytes**. Neither tree remains available.
The ignored `retention-pruned-20261010.json` records counts, inventory digests,
source/result binding and preserved keepers; prior test-result logs remain.

The new pass keeps current source in Git, a committed AEC benchmark/compact result,
a reproducible public-text scheduling probe and one ignored combined test receipt.
Owned generated synthetic fixture trees are removed after their result is recorded;
failed run summaries can remain compact. No raw owner audio is committed or copied
into public evidence. Worktree cleanup after verified landing follows the existing
merged-ancestry checks. Future work must verify current keeper locations, not infer
that deleted historical archives can still be opened from older work-log prose.
