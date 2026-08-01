Valid until: main advances beyond this task's landing commit — then treat as
history.

# Task result — complete asynchronous Sherpa decode owner

## Outcome

The default local Sherpa path now has one bounded reader-to-owner PCM boundary:
the existing `CaptureMediaSession`/`CaptureMailbox`. A dedicated
`SherpaStreamingDecodeOwner` runs the complete ordered DSP, VAD, primary ASR,
playback-confirm, word-cut, endpoint, and capture-loop callback state machine.
No additional PCM queue was added.

Capture scope, decoder continuity, and stream incarnation are separate.
Mailbox loss flushes the old scope and cancels an unclaimed in-flight block;
an already-claimed effect may finish. Capture gaps and native errors retire
both primary and word-cut roles before replacement. PCM and reader-time
references are bytes-backed, irreversibly read-only, and excluded from reprs.

Startup is one rollback transaction after input opens. The complete processor
publishes readiness before the reader starts; tail-worker and owner launch
ambiguity is joined or retained without joining unpublished Python threads.
Native teardown publishes `CLOSING`, pre-start close does not hold the owner
lock across destructors, callback-driven full stop defers exact resources until
its lease unwinds, and both legacy WAV and synchronized diagnostic recording
receive the correct clean/incomplete disposition.

Capture replay remains a synchronous shared-loop evaluator. Its source digest
binds the production owner implementation, but its attestation rejects any
production owner and labels execution `single-session-synchronous-replay`.

## Main files

- Added `core/engines/_sherpa_streaming_decode_owner.py` and focused owner tests.
- Hardened `core/media_session.py` mailbox ownership, immutable snapshots,
  in-flight effect leases, whole-scope loss, and reader launch ambiguity.
- Updated `core/engines/sherpa.py` for the complete owner, exact role/identity
  rotation, native-error recovery, startup rollback, and re-entrant cleanup.
- Updated hotword, media, startup, replay provenance, playback, and adjacent
  regression tests.
- Added ADR-0111 and updated `STATUS.md` plus `docs/agent-testing.md`.

## Verification

- Full non-real-model split: 7,241 passed (7,210 low-priority broad plus
  31 isolated sandbox/logging/path/thread-sensitive checks), 14 skipped,
  23 model-only deselected, and 9 pre-existing warnings.
- Documented complete-owner/adjacent gate: 325 passed; dedicated owner file:
  10 passed, including terminal/registry stress; required APM/DTD: 6 passed.
- One unchanged bounded-I/O test was isolated after its process-global
  `os.read` hook was consumed by an unrelated reader; it passed alone and a
  reviewer reproduced the pre-snapshot interference deterministically.
- Working-tree and staged `git diff --check` are clean; the exact 15-file task
  set is staged with no unstaged changes.

## Limits / next

No STT/TTS model, endpoint policy, enrollment rule, tool authority, capability,
entry point, or device default changed. No physical microphone/audio-device,
GPU/model-backed, acoustic AEC/room, WER, latency, or live-conversation test
ran.

Fake-device tests prove owner-before-reader ordering, not native hardware.
Replay bypasses the production owner thread, native reader, mailbox overload,
and effect leases; its source digest is not execution evidence. The inline
`media_pcm_queue_ms=0` path is compatibility only.

Next, add an end-to-end conversation acceptance gate and acquire fresh private
plus permissively licensed disjoint command/multi-voice data. Run isolated GPU
model cells without changing runtime defaults, then perform physical live A/B.
