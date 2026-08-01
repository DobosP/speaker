Valid until: this branch lands or ADR-0107 is superseded — then treat as history.

# Task result — generation-scoped real-time media stage

## Outcome

Replaced Sherpa's raw async-final queue with the first route-neutral typed media
stage. It carries the existing capture epoch and capture generation, owns
immutable endpoint PCM, has a physical queue bound and per-item cancellation,
and binds one worker to one run. Saturation now retires the oldest queued final
with a typed backpressure outcome; it can no longer move expensive finalization
back onto the capture processor. Shutdown always releases queued work, wakes the
worker, and rejects late offers independently of diagnostic recording.
A final callback that calls `stop()` now defers retained audio cleanup until its
capture-effect lease unwinds, after which restart guards can pass normally.

No recognizer, TTS, entry point, setup option, barge-in/enrollment policy, tool
authority, or runtime default changed.

## Files changed

- Route-neutral stage: `core/realtime_media_stage.py`.
- First production integration: `core/engines/sherpa.py`.
- Deterministic stage, final-worker, lifecycle, and adjacent regressions under
  `tests/`.
- Decision/current truth/testing: ADR-0107, `STATUS.md`, and
  `docs/agent-testing.md`.

## Verification

All test commands used one-thread math, idle I/O priority, and process niceness.

- Isolated reusable stage: 17 passed.
- Final focused stage/worker gate: 45 passed, one real-model test deselected.
- Expanded affected-area gate: 381 passed, one real-model test deselected.
- Downstream runtime/session cancellation: 153 passed.
- Required APM/DTD: 6 passed.
- Complete non-real logic gate: 7,080 passed as 7,052 repository tests plus
  28 scheduling-sensitive LiveKit engine tests run in isolation; 13 skipped,
  23 model-only deselected, and nine pre-existing warnings.
- Scoped Ruff lint passed for every changed Python file; format checks passed
  for the new stage and expanded focused-test files. Baseline-formatted
  integration files were left untouched outside semantic hunks.
- `git diff --check`: passed.

A first combined attempt was not accepted as evidence: log suppression removed
one test's subject, the sandbox denied run-log creation, a worktree checkout had
not preserved one owner-only data-file mode, and LiveKit's one-second loop
fixture starved under the accumulated low-priority run. All affected cases then
passed under their intended writable-log/owner-only conditions; the complete
gate was rerun green with only that 28-test scheduling-sensitive file isolated.

## Limits and next change

This is deterministic fake/replay and lifecycle evidence only. No microphone,
speaker, live route, model weights, GPU decode, WER, natural-conversation, or
latency validation ran. A native decoder already executing at shutdown cannot
be forcibly killed, but its stage event and capture epoch fence late effects.

Next, introduce one synchronous `SherpaStreamingDecodeSession` that exclusively
owns every online recognizer and stream operation. Only after that ownership is
proven should the complete decoder move behind this stage and be compared on
recorded/public replay. Partial extraction is unsafe because VAD rebase, barge
confirmation, word-cut, and normal capture currently share native recognizer
state.

## Merge recommendation

Recommend fast-forwarding this green, independently reviewed branch. Do not
promote a recognizer or runtime default from this foundation.
