# ADR-0120: Ignore inert Git sentinels for AMI fixture output

Date: 2026-08-02
Status: accepted

## Decision

Make the production AMI conversation materializer distinguish an actual Git
worktree marker from an inert sandbox sentinel through bounded, no-follow
directory descriptors. Reject a non-empty `.git` file, a `.git` directory with
`HEAD`, a symlink, or any abnormal marker; treat a `.git` directory without
`HEAD` as non-repository metadata, and rebind the marker identity after the
`HEAD` probe so an in-probe replacement fails closed. Keep the existing
absolute, canonical, new-destination, repository-containment, and
production-only Git checks. Keep this change local to the AMI materializer so
it changes only the AMI preparer closure.

## Context / why

The fleet requires private run data under its ignored `_temp` hierarchy, but a
managed ancestor has an inert `.git` directory containing no `HEAD`. The AMI
materializer used `lexists`, so verified official inputs failed before
publication even though the destination was not in a repository. Harper
Valley, EdAcc, and Common Voice already recognize real Git markers rather than
rejecting every entry named `.git`. Moving the helper into shared bounded-I/O
code would unnecessarily change all four receipt-bound preparer closures.

## Consequences

The exact AMI v1.6.2 annotations and synchronized ES2004a close/far recordings
now produce and independently validate a private 24-case, 3,802,880-byte corpus
covering 59.42 seconds. Its corpus and preparation-receipt SHA-256 values are
`5a8ba97f71c5c6c69013def3682c62bc6f4f902fc51d24a2a51ea92f46333c7f` and
`c6ca84004344c275498c787b760df0d3dfdad65876c24e84816bd8d74ba33810`.
The output stays owner-private and uncommitted. This is source/materialization
evidence only: the other three locked corpora, the four-source model run,
capture, endpoint, device, live, tool, and runtime-default gates remain open.
