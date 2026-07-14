Valid until: a newer physical-barge ADR supersedes ADR-0071 — then treat as history.

# V5 live barge closeout

## Verdict

The 2026-07-14 physical bare-speaker gate failed. Voice selection remained
stable, but soft-speech admission, pause handling, prompt interruption, and
exact Stop were unreliable. Disabling enrollment reproduced the Stop failure,
so the next direction starts before speaker identity. The isolated v5 candidate
was never promoted.

This verdict does not roll back the deterministic infrastructure already on
main. It separates green headless/virtual capability from red physical behavior.

## Landed foundation

Main implementation commit `31654a2` and its ancestors include:

- pinned MiniCPM v5 provisioning/readiness identity and fail-closed fresh setup;
- memory-evidence provenance and controller-owned history/correction behavior;
- isolated v5 enrollment preparation plus descriptor-bound promotion that
  cannot replace historical v4;
- current-signal AGC, calibrated speech evidence, bounded word-cut decoding,
  capture recovery, session voice lock, and stale-output cancellation;
- a private virtual echo route with exact stream provenance and teardown proofs;
- deterministic synthetic command rendering, calibrated fallback confined to
  that private route, and capture-onset timing.

The later commits on main only updated credentials and fleet-context pointers;
they did not remove this implementation.

## Verification before the live gate

| Gate | Result | Boundary |
|---|---:|---|
| Full headless suite | 4049 passed, 31 skipped, 9 warnings | No physical audio |
| APM/DTD regression | 6 passed | Deterministic |
| Strict recorded-owner replay | 9 passed | Fake streams; archived input |
| Word-cut focus | 146 passed | Deterministic |
| TTS construction focus | 28 passed | No audible grade |
| Private delay run `041032` | PASS; one cut at 0.509 s | Virtual route |
| Private delay run `041156` | PASS; one cut at 0.818 s | Virtual route |
| Production-hybrid conversation A/B | 42/42 on both model pairings | Local-loopback models |
| Semantic-memory probe | PASS; PRIVATE main-only recall | Local-loopback models |

Both delay runs had zero self-cuts and all six topology, capture, duplex,
correlation, child-exit, and cleanup proof classes. These results validate the
harness and control path, not room acoustics or enrolled-owner barge-in.

## Enrollment and live evidence

| Phase | Objective result | Interpretation |
|---|---|---|
| `173321` | Aborted because the required echo-cancel module was absent | Safe fail-closed; no candidate write |
| `174105` | Interrupted after calibration; no candidate write | Incomplete, not evidence |
| `174212` | 3/3 clips, about 12 s each, dim 512, similarity min 0.60 / mean 0.67 | Isolated v5 candidate captured; never promoted |
| Doctor | Prepared desktop profile reported `READY` | Startup prerequisites present for that session |
| `192151`, enrollment on | Stable `sid=0`; normal France question passed | Voice lock and an ordinary turn worked |
| `192151`, enrollment on | Soft Yes dropped; one-second pause split; delayed/garbled cuts; exact Stop failed | Physical dialogue/barge gate red |
| `193713`, enrollment off | No Stop final, word-cut trace, handoff, or cut | Failure reproduced before identity |

In `192151`, the first real cut was roughly 27.8 seconds after the long reply
started, after repeated owner speech. The two eventual handoffs were garbled.
The reported sub-millisecond cancellation receipts measure detection to cancel,
not when the operator began speaking, so they are not a successful latency grade.

In `193713`, a visible near-end RMS burst still had zero VAD fraction and did
not begin the energy fallback. Both live runs later reported a steady input far
quieter than their initial ambient calibration. That is a leading diagnostic
hypothesis—route settling or a signal-domain mismatch—not a proven root cause.

Operator grades were explicit: the voice stayed the same; the solar explanation
did not stop correctly; the override was attempted more than once; exact Stop
failed with enrollment both enabled and disabled.

## Cleanup boundary

The owner authorized deletion after this summary lands. Cleanup covers the
rejected isolated candidate, its private prepared config, goal-session test/live
logs and worktree caches, the redundant pre-v5 preparation backup, obsolete
STATUS-only stash, merged task worktrees/branches, and dead legacy `/tmp`
worktree registrations. It does not cover the active historical enrollment,
primary local config, model files, unrelated branches/worktrees, or unrelated
main-run logs.

## Remaining work for the new direction

1. Reproduce exact Stop with enrollment disabled and bounded per-stage markers
   for physical input, OS echo-cancel output, calibration, AGC/GTCRN, VAD,
   energy admission, and decoder feed. Persist raw audio only with fresh owner
   approval.
2. Test the route-settling hypothesis. Calibration must observe the same stable
   signal domain used during playback, or safely reacquire when that domain
   changes.
3. Make playback-time VAD/energy admission observable and add replay tests from
   the smallest approved diagnostic capture. Do not tune speaker thresholds
   while this earlier stage is starved.
4. Resolve the open-speaker warning that the learned echo-floor gate is inert;
   prove whether OS echo cancellation makes that intentional or supply a valid
   learned floor.
5. Evaluate a bounded local exact-Stop detector only after the earlier signal
   path is measurable; require deterministic replay and one prompt physical cut
   with no self-cut before broadening it.
6. Repair one-second-pause continuation so a short prefix and continuation do
   not become separate answered turns.
7. Only after exact Stop and one multiword override pass from clean main should
   a fresh v5 candidate and the wider acceptance sequence be attempted.

The next session should begin from ADR-0071 and `STATUS.md`, not from the deleted
candidate or run directories.
