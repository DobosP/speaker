# Task Result — 2026-07-14 v5 live barge closeout

Valid until: ADR-0071 or its documented evidence changes — then treat as history.

Branch: `docs/close-v5-live-ab`

Status: green documentation/landing gate; physical bare-speaker release gate red.

## Outcome

- All implementation produced during the MiniCPM v5, memory, enrollment,
  voice-continuity, and deterministic barge goal is present on main at
  implementation commit `31654a2` and its ancestors.
- The headless, local-model, semantic-memory, recorded-owner, private virtual
  echo-route, and deterministic delay gates remain green.
- Live enrollment completed and the selected voice remained stable, but the
  physical A/B failed soft speech, a one-second pause, override handoff, and
  exact Stop. The v5 candidate was rejected and never promoted.
- Exact Stop also failed with enrollment disabled in memory. ADR-0071 therefore
  moves the next investigation before speaker identity, into playback-time
  capture, echo-cancel, calibration, VAD/energy, denoise, and decoder admission.
- The current failure evidence and next tasks are durable in `STATUS.md`,
  `.agents/backlog.md`, ADR-0071, and the dated live closeout.

## Main-integration audit

The goal's active worktree branches were compared with main before cleanup.
Six branch tips are direct main ancestors. Four non-ancestor task tips were
canonicalized during integration rather than omitted:

- MiniCPM provisioning identity → `f900f9f`;
- fail-closed fresh installation → `4bdd9d1`;
- memory evidence provenance → `71238b7`;
- v5 enrollment promotion → `d18276f`.

Their implementation and test patches are equal; only integrated STATUS,
TASK_RESULT, ADR numbering/cross-references, and already-landed API adaptation
differ. No unique task implementation remains outside main.

## Verification

On the closeout branch after rebasing onto current main:

```text
/home/dobo/work/speaker/.venv/bin/python -m pytest tests -q
4049 passed, 31 skipped, 9 warnings in 78.11s

/home/dobo/work/speaker/.venv/bin/python -m pytest tests/test_apm_double_talk.py -q
6 passed in 0.94s

/home/dobo/work/speaker/.venv/bin/python -m pytest tests/test_gh_admin.py -q
10 passed in 0.07s

git diff --check
PASS (no output)

test $(wc -l < STATUS.md) -le 100
PASS
```

The first full run exposed four stale `GIT_HUB_TOKEN` test fixtures after main's
new fleet credential setup had moved the helper to `GIT_HUB_ACCESS_TOKEN`. The
minimal test-only repair landed separately as `31b5872`; the second full run
above is green. Both execute paths were mocked; no network or real secret was
used.

Previously completed gates retained as implementation evidence:

- strict archived recorded-owner replay: 9 passed with fake streams;
- word-cut focus: 146 passed; TTS construction focus: 28 passed;
- production-hybrid MiniCPM/Gemma and Gemma/Gemma: 42/42 each;
- semantic memory: PASS with PRIVATE main-only recall;
- private delay runs `041032` and `041156`: one causal cut each at 0.509 and
  0.818 seconds from calibrated capture onset, zero self-cuts, all route and
  cleanup proofs green.

No microphone, speakers, model server, network, enrollment, or primary local
config were used or modified by the closeout tests.

## Live gate evidence

- Enrollment `174212`: three roughly 12-second clips, dimension 512, pass
  similarity minimum 0.60 and mean 0.67; isolated candidate only.
- Enrollment-on `192151`: all TTS resolutions stayed `sid=0`; one normal France
  question passed. Soft Yes was dropped, a pause split the turn, the first cut
  was very late and garbled, the override was repeated, and exact Stop failed.
- Enrollment-off `193713`: exact Stop produced no word-cut trace, Stop final,
  handoff, or cut. Playback-time VAD remained zero despite a near-end burst.
- The scalar evidence points before identity but does not yet distinguish route
  settling, signal-domain/calibration drift, VAD/energy starvation, echo-floor
  handling, denoise, or decoder admission.

## Files changed by the closeout branch

- `STATUS.md` — current physical verdict and next gate.
- `.agents/backlog.md` — prioritized new-session TODOs ahead of historical notes.
- `docs/adr/0071-reject-current-v5-candidate-and-refocus-physical-barge-admission.md`
  — rejection, preservation, cleanup, and direction decision.
- `docs/2026-07-14-v5-live-barge-closeout.md` — bounded live/headless evidence.
- `TASK_RESULT.md` — integration audit, exact tests, risks, and handoff.

## Cleanup authorization and limits

After this documentation is on main, the owner's explicit cleanup authorization
covers only this goal's rejected candidate, prepared private config, test/live
logs and caches, redundant pre-v5 backup, obsolete STATUS stash, merged task
worktrees/branches, and dead legacy `/tmp` worktree registrations. It does not
cover the active historical v4 enrollment, primary config, models, unrelated
branches/worktrees, or unrelated main-run logs.

## Risks and next session

- Cleanup intentionally removes the detailed live artifacts and rejected
  biometric candidate; the dated closeout retains bounded scalar evidence only.
- Virtual and recorded gates do not prove current-room acoustics or audible cut
  latency. Do not promote v5 or claim physical barge acceptance.
- Begin the new session with enrollment disabled and bounded per-stage markers.
  Exact Stop must cut promptly with no self-cut before a multiword override,
  pause repair, new enrollment, or wider acceptance A/B.

## Merge recommendation

The current repository suite and documentation gates are green. Land and push
the closeout, then perform only the audited, explicitly authorized cleanup.
