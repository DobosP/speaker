# ADR-0162: Add a receipt-bound PriMock57 overlap evaluator

Date: 2026-08-09
Status: accepted

## Decision

Keep the PriMock57 fixture bytes, its production v4 evidence, public matrix v5,
the canonical 96-case qualification, and every runtime default unchanged. Add a
separate self-digested `primock57-overlap-evaluator-v1` lock and one dedicated
aggregate-only evaluator for the exact production PriMock57 overlap manifest
and receipt. The lock admits only the three existing two-role cases and fixes
three sequential inputs per case in role-A stem, role-B stem, then mix order,
with three repeats and one worker at a time.

Score each stem against its one private reference with the repository's frozen
word normalization and edit accounting. Score a mix with
`two-utterance-min-order-wer-v1`: compare its single hypothesis with the two
possible whole-utterance reference concatenations, retain the smaller integer
edit cost, and choose role A then role B on a tie. Sum substitutions,
insertions, deletions, and reference words across cases before deriving WER.
This is a bounded order-invariant diagnostic for a single-output recognizer.
It is not ORC-WER, tcpWER, tcORC, diarization, speaker attribution,
word-timing, or an official PriMock metric.

Do not rewrite the retained custom manifest's creation-time
`overlap_metric=not-implemented` field or modify the receipt-bound preparer.
The new evaluator lock supplies the later method and binds the exact v4
manifest SHA-256
`e69c85356a77cd87b0c7f4c100614dccbf1484e16532a5a2239a768162e443b4`
and receipt SHA-256
`69ec7df35591a2b1a7203109c58077540540c32deec99dc933a5ddef0840f507`.
Changing the preparer merely to retain new evaluator state would invalidate the
existing receipts, so the evaluator owns an independent fd-relative,
no-follow snapshot of the exact receipt, manifest, and nine PCM leaves. It
reopens the strict bundle loader and compares the snapshot before execution,
after worker shutdown, and immediately before returning or publishing.

Reuse the existing manifest, runtime/model receipt, source-bundle, host lock,
worker supervisor, protocol, cgroup evidence, and private scratch teardown
seams. Require every input's complete source to reach a valid final; native
early termination is not silently scored. Publish only a canonical private
mode-0600, single-link, no-clobber aggregate report. Reject transcript text,
reference fragments, case rows, role labels, local paths, non-finite values,
and self-declared privacy claims at the output boundary. The report marks
endpoint and latency evidence, qualification, and promotion authority false.

## Context / why

ADR-0161 intentionally stopped after publishing three immutable stem/mix
diagnostics because the generic schema-v2 evaluator has one reference per case
and cannot represent two simultaneous private references. The landed
NOTSOFAR slice contains zero overlap; the AMI endpoint proxy admits only
isolated windows; barge and logical-turn tests cover scripted or successive
single streams rather than simultaneous natural speech. PriMock57 is therefore
the smallest already-admitted source that can expose single-output STT loss on
two concurrent roles without another download or live/model run.

MeetEval tcpWER requires speaker-attributed hypothesis streams, and its
time-constrained families require hypothesis timing the current worker
protocol does not publish. Calling this result ORC-WER, tcpWER, or tcORC would
overstate the evidence. A future multi-channel or timestamped protocol may add
a separately pinned standard metric, but it must not reinterpret this report.

The Microsoft AEC Challenge 32-by-four subset remains the next orthogonal
open-speaker task. It can exercise production APM/DTD and barge behavior, but
its synthetic audiobook double-talk does not replace this natural-role STT
diagnostic.

## Consequences

- Any current streaming candidate can be measured on the three PriMock mixes
  without exposing private references or weakening ordinary-WER denominators.
- The tiny diagnostic can reveal omissions and order sensitivity, but three
  synthetic 0.5 mixes cannot justify an STT selection, threshold, endpoint,
  identity, tool-authority, entry-point, or runtime-default change.
- The existing production bundles remain byte-valid because their preparer and
  manifest are unchanged. The evaluator lock and wrapper form a new evidence
  boundary rather than mutating historical evidence.
- A real model/GPU run remains a separate resource-gated action. Bare-speaker
  barge-in and natural live conversation still require owner hardware A/B.

This ADR refines ADR-0098, ADR-0109, ADR-0129, ADR-0159, ADR-0160, and
ADR-0161.

## Verification

- Evaluator-lock recipe SHA-256:
  `0697956f0671fb77e101f40b67f4a46bfb5a6da9cd0962d2cde4eec4577f261e`;
  its 986 committed bytes have SHA-256
  `b8fdc497548b8d51fd5381513427cfffb715f1b6757481410900e45c337d2748`.
- The exact retained v4 bundle passes strict loader and evaluator preflight with
  three cases and nine PCM inputs; the final wrapper-closure SHA-256 is
  `7b75ee404f1b1c45a7bb18a5af27e58f505c92e2694c115406eb14fab64bcd14`.
  No model, GPU, microphone, device, quality, latency, endpoint, or
  live-conversation run occurred.
- The exact fixture, metric, and evaluator headless gate passes 163 tests; the
  adjacent evaluator, supervisor, corpus-writer, isolated-wrapper, and APM/DTD
  gate passes 250 tests. Static and independent-review results are recorded in
  `STATUS.md` from the final landed bytes.
