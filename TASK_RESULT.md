Valid until: this branch lands or ADR-0100 is superseded — then treat as history.

# Task result — synchronized private voice diagnostics

## Outcome

Added an opt-in diagnostic mode to the existing `python -m core --session`
runtime. No tool-specific or diagnostic entry point was added. Enabling both
Sherpa recording references now produces four private tracks with independent
cursors: model pre-gain, model gate, selected ASR input, and the unmodified
reader-time playback-reference snapshot.

The transcript-free typed timeline covers capture coordinates, VAD/ASR stages,
every endpoint decision, final selection/abort, barge-in, and causal playback
admission/onset/terminal receipt. Endpoint decisions replay from the exact
configuration and adaptive pause snapshot. Reply playback binds back to the
acoustic input generation; auxiliary, latency-ack, and reminder speech remain
distinct.

Publication is single-writer, bounded, private mode 0600, hash-bound, atomic,
and no-clobber. A caller can observe `finalizing`; every terminal error remains
sticky. Loss, malformed evidence, lifecycle mismatch, missing receipt, replay
mismatch, or unclean shutdown yields incomplete or publication-failed evidence
without taking ownership of voice behavior.

## Verification

- Focused diagnostic/endpoint/recording/runtime gate: 253 passed, 3 skipped.
- Diagnostic publication module: 38 passed, including filesystem-race and
  delayed-failure injection.
- Required APM/double-talk gate: 6 passed.
- Complete non-model logic gate: 6,865 passed, 13 skipped, 23 deselected, with
  nine pre-existing warnings.
- Scoped Ruff and `git diff --check`: passed.

All verification was deterministic and headless. No microphone, speaker, live
audio route, real STT/TTS model, or GPU was used.

## Limits and next work

A complete bundle proves internal alignment and causal bookkeeping, not physical
raw-microphone truth, audible playout, transcript accuracy, natural conversation,
room AEC, owner identity, or live barge-in. The next evidence step is a fresh
owner-run `./live.sh` vault/open-speaker A/B, followed by native-reader/gap and
synchronized AEC replay plus disjoint owner/noise/multi-voice data.
