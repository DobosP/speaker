Valid until: main advances beyond this task's landing commit — then treat as
history.

# Task result — central local Sherpa streaming-decode ownership

## Outcome

Implemented one run-scoped `SherpaStreamingDecodeSession` as the exclusive
owner of the local engine's primary online Sherpa recognizer and native
streams. The capture processor binds the session to its exact `Thread`; only
opaque handles cross the boundary. Foreign-thread, re-entrant, forged, stale,
cross-session, and post-close access fails before native code, and native
streams are retired on the owner before the recognizer.

The production capture loop, normal recovery/rebase paths, barge confirmation,
word-cut streams, and capture-replay evaluator now use that owner. Multi-case
replay keeps one owner for the evaluator batch, creates fresh streams per case,
closes once, and attests the completed lifecycle instead of hard-coding a
provenance label.

Startup rollback covers capture-thread construction, pre-transfer start
failure, and interruption after the capture thread actually acquired ownership.
It fences capture and playback before callbacks, drains/stops raced playback
receipts, joins or retains capture owners, and releases native/input/finalizer/
recorder state only after quiescence. A delayed `speak_tracked()` cannot
resurrect the receipt dispatcher after rollback.

Shutdown now wakes only a live playback worker and drains an unconsumed private
wake item after the bounded join. Failed capture startup, stop-before-start,
and repeated stop therefore cannot terminate a later playback run.

## Files changed

- Added `core/engines/_sherpa_streaming_decode.py` and its focused lifecycle,
  race, ownership, launch-rollback, and engine-transfer tests.
- Updated `core/engines/sherpa.py` to transfer and route the primary online
  recognizer through the session, make pre-launch rollback fail closed, and
  leave playback restart state free of stale shutdown sentinels.
- Updated `tools/capture_replay_eval.py` plus replay tests for one owner across
  a multi-case batch and lifecycle-derived execution attestation.
- Updated adjacent ASR, word-cut, media-session, and virtual-audio tests to
  exercise opaque handles and verify owner closure.
- Added ADR-0110 and updated `STATUS.md` plus `docs/agent-testing.md`.

## Verification

- Focused ownership/capture/barge/replay/duplex gate: 295 passed.
- Complete non-model split: 7,204 passed as 7,174 low-priority repository
  checks plus 30 isolated logging/path/thread-sensitive checks; 14 skipped,
  23 model-only deselected, 9 pre-existing warnings.
- Required APM/double-talk regression: 6 passed.
- Independent lifecycle, integration, and test-gap reviews closed native
  destructor, launch-transfer, receipt-resurrection, and replay-batch races.
- `git diff --check`: clean.

## Limits / next

No model, endpoint, enrollment, voice-agent capability/tool authority,
entry-point, or audio default changed.
No microphone, device, native-reader, AEC, model-backed, or live conversational
test ran. The later async extraction must transfer the complete decode owner,
immutable PCM, exact scope/incarnation, and ordered whole-scope loss rotation;
generic drop-oldest media-stage semantics are insufficient.

FileReplay, rollback-only legacy LiveKit, offline final ASR, KWS, and isolated
benchmark adapters remain separate, explicitly out-of-scope owners. Broader
transactional rollback after the capture thread launches successfully but a
later playback/final worker launch fails remains existing lifecycle debt.

## Merge recommendation

Land this review-hardened synchronous ownership boundary. It is the prerequisite
for a later complete asynchronous decode coordinator, not evidence to promote a
model or claim improved live latency/STT quality.
