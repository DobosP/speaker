# Session 2026-06-02 — landed the unification refactor on `main` (reconciled with the origin/main voice batch)

Cross-machine handoff (ran on the **Windows 11** box; continuation may be elsewhere).
Written in-repo because per-user Claude memory does NOT travel between machines.
Goal of this session (user's words): "continue where the previous session was, I
closed terminal by mistake" + "I fixed the guard hook." The prior session's
next-step #1 was **finish landing** the unification refactor onto `main`.

## What I found (the situation had changed)

The landing was no longer a simple `feat/aec-dtln → main` merge:

- **`origin/main` had advanced ~30 commits** from another machine since the prior
  session — it was at `57cb6d6` *"Merge feat/aec-dtln: on-device AEC + capability
  router"* plus a large **voice/audio batch** (see below). It was no longer the old
  `049e947` base the prior handoff assumed.
- **Only 2 commits were still unlanded** on `feat/aec-dtln`: `398471c` (docs
  unification + `session_bootstrap`) and `8d62ca8` (Windows landing doc). The
  earlier router/AEC/DTLN work had already been merged to `origin/main` elsewhere.
- So `origin/main` and `feat/aec-dtln` had **diverged**: main had the voice batch
  but not the unification; the branch had the unification but not the voice batch.
- **The guard hook was fixed** (`.claude/hooks/guard.ps1` no longer blocks `main`
  pushes — it still blocks the work SSH key / `.ssh/config` / `.gitconfig` /
  `git config --global`). `GIT_HUB_TOKEN` is unset here and `gh` is not installed,
  so the PR-via-API route was unavailable anyway — direct push was the path.

## Branch / commit map

| Ref | Before | Action |
|-----|--------|--------|
| `origin/main` | `57cb6d6` (voice batch, no unification) | merged `feat/aec-dtln` in → `d215a31`, pushed |
| `feat/aec-dtln` | `8d62ca8` (unification, no voice batch) | merged into `main`, then **deleted** (local + remote) |
| local `main` | `3290fd1` (unpushed dry-run merge, missing `8d62ca8`) | reset to `origin/main`, re-merged cleanly |

## What landed

- **`d215a31` Merge `feat/aec-dtln` → `main`**: brings `398471c` (unification:
  `docs/unified_architecture.md`, `tools/session_bootstrap.py` + test, CLAUDE.md
  Session-bootstrap section, `always_on_agent/snapshots.py` deletion, `core/agent.py`
  `AgentEvent`→`AgentBrainEvent` rename) and `8d62ca8` (`docs/windows_landing_workflow.md`).
- **Merge was clean and validated:** `git merge-tree` reported no conflicts; the
  rename/snapshot-deletion were verified safe against origin/main's new code (every
  `AgentEvent` ref on main points to the PUBLIC `always_on_agent/events.py` contract,
  not the renamed private core one; nothing imports the deleted `snapshots.py`).
- **Full suite green on the merged tree: 1283 passed, 13 skipped, 0 failed**
  (`.venv\Scripts\python.exe -m pytest tests -q`, with `SPEAKER_KEEP_RUNS=9999`).
  +78 over the prior 1205 = origin/main's new test files. 2 pre-existing numpy
  divide-by-zero RuntimeWarnings in `core/endpointing.py` (mel filter), not failures.
- Reconciled `.agents/status.json` + `.agents/backlog.md`, wrote this handoff.

### origin/main voice batch now on `main` (landed from the other machine)
SenseVoice two-pass final ASR (now DEFAULT, English-pinned); Smart Turn v3 prosody
turn-completion detector + `tools/turn_detect_check` validation tool + adaptive
confidence-tiered endpoint floor; multi-signal barge-in (loudness fallback +
scale-invariant reference-coherence + self-calibrating EWMA margin); enrollment
hardening (AT2020 capture-rate pin, loudness rescue, VAD-trimmed); `tools/echo_probe.py`;
`live_session` per-capability latency + denoise A/B + response-quality grading.

## Environment on the Windows box
- Use **`.venv\Scripts\python.exe`** for the full suite (git-bash `python` is system
  3.10, lacks `onnxruntime`/`sherpa_onnx` → mass collection errors).
- The **Bash tool runs bash, not PowerShell** — `$env:VAR=` / backslash paths fail
  there. For env-prefixed runs in bash: `SPEAKER_KEEP_RUNS=9999 .venv/Scripts/python.exe ...`.
- Guard now allows `git push origin main` (personal SSH key `git@github-personal`).

## Next steps (pick up here)
1. ✅ **DONE — `docs/unified_architecture.md` + `docs/architecture.md` refreshed**
   to current truth for the voice batch: SenseVoice two-pass ASR (default-on),
   Smart Turn v3 `ProsodyTurnCompletionDetector` real-voice scoring (corrected the
   stale `smart_turn_enabled`/`smart_turn_model` keys + class name + the removed
   `test_smart_turn_endpoint.py` reference), scale-invariant coherence barge-in
   (primary) + self-calibrating EWMA margin, enrollment hardening (capture-rate pin,
   VAD-trim, loudness rescue), adaptive confidence-tiered endpoint floor. Authored via
   a per-subsystem research fan-out (6 Explore agents), every claim verified against
   code/config, suite green 1283. (This session.)
2. **Resume P1 voice/audio hardware validation** (`.agents/backlog.md`): AEC ERLE on
   the real mic; Smart Turn v3 on-hardware A/B (detector + `turn_detect_check` tool
   are now on `main` — what's left is the recording-based A/B).
3. **Optional:** fix the `core/runlog.py` test-isolation pruning; consider physically
   moving the banner'd `docs/*_2026-05.md` into `docs/archive/`.
4. **Cross-platform P1:** mobile convergence onto the `AgentEvent` contract; SQLite
   memory backend for mobile (unified doc §6/§10/§12).
