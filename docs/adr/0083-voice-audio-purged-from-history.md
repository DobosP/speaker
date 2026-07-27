# ADR-0083: Raw voice audio purged from git history; audio can no longer be whitelisted in

Date: 2026-07-28
Status: accepted

## Decision
All raw audio ever committed to this repository — 15 `logs/runs/run-*.wav` session recordings
and 5 `temp/2_min_sample.mp3_chunk_*.wav` files, 20 blobs total — was removed from **every ref
and every commit** with `git-filter-repo --invert-paths`, and the rewritten history was
force-pushed to `origin` (github.com:DobosP/speaker) and `nas`. Paul authorized the history
rewrite explicitly on 2026-07-28 ("you have the right to filter and rewrite repo and history
decisions"), which supersedes ADR-0008's deferral of the purge to a pre-release gate and
overrides the fleet's standing never-rewrite-published-history rule for this repository and
purpose.

`.gitignore` loses the `!logs/runs/*.wav` / `!logs/runs/*.mp3` replay whitelist permanently.
The rule that remains is one line: raw personal voice never enters git. Replay fixtures must be
PII-free, converted to `.npy` under `tests/fixture_audio/` (the form the replay tests actually
consume — they skip when missing, so clones without fixtures degrade gracefully), and reviewed
before an explicit `git add -f`.

## What was preserved, and where
- Every purged audio file was extracted from history **before** the rewrite into the owner-only
  archive `~/voice-archive-speaker/` (mode 700) on this machine.
- A full pre-purge bundle of all local refs (`speaker-pre-purge-15c87f0.bundle`) and a second
  bundle of the sixteen `refs/backups/2026-07-12/*` refs that existed only on origin
  (`speaker-origin-backup-refs.bundle`, fetched and verified before deletion) sit in the same
  archive. Nothing was destroyed without an offline copy.
- The 15 session WAVs were restored into the working tree at `logs/runs/` as **untracked,
  ignored** local test material — replay debugging keeps working on this machine.

## Mechanics and verification
- Rewrite ran in a fresh clone: 684 commits before, 684 after; `git log --all` over
  `*.wav/mp3/flac/ogg/m4a` returns zero paths; tags `android-latest`/`gemma-model` rewritten and
  pushed; HEAD content diff vs the old tree = exactly the 9 then-tracked WAVs, nothing else.
- Six local branches + worktrees and eight `nas` branches were deleted **after** each passed
  `git merge-base --is-ancestor` against main individually (fail-closed, per the 2026-07-06
  incident lesson); their content ships inside the pre-purge bundle regardless.
- The sixteen origin `refs/backups/*` refs and the stale dependabot branch were deleted from
  origin after bundling (deleting the dependabot ref closes its PR; dependabot will regenerate
  against the new history).
- NAS bare repo: `reflog expire --expire=now` + `gc --prune=now` ran on the NAS itself; pack is
  81 MiB of live objects only.

## The residual GitHub exposure — owner follow-up
GitHub keeps `refs/pull/1..23` alive server-side and clients cannot delete them; unreachable
objects also survive until GitHub's own gc. **The purge is complete for every surface we
control, but a determined party with old PR URLs may still reach pre-rewrite commits on
github.com.** Closing that requires one of: a GitHub Support request to run gc / drop PR refs,
flipping the repo private (no API token was available on this host), or delete-and-recreate the
repo. Until one of those happens, treat the GitHub copy as *reduced*, not *eliminated*,
exposure. This is the single remaining action item, and it is the owner's.

## Consequences
- Anyone with an old clone (the Windows box, NAS-derived checkouts) must `git fetch` +
  `git reset --hard origin/main` — old and new histories share no hashes from the first purged
  commit forward. Same-content verification is `git diff` against the old tip, which shows only
  WAV deletions.
- `STATUS.md`'s "raw audio never leaves the device" architecture claim (§9.7) is now true of
  the repository as well as the runtime.
- Revisit when: the owner executes one of the three GitHub-side closures above (note it here),
  or a replay fixture is needed in-repo (the `.npy` + review path, never raw WAV).
