# ADR-0100: Bind private voice diagnostics to causal runtime evidence

Date: 2026-08-01
Status: accepted

## Decision

When both Sherpa recording references are enabled, make the existing session
recording an opt-in synchronized diagnostic bundle rather than adding another
application entry point. Atomically admit four private PCM16 tracks—model
pre-gain input, model-gate input, selected ASR input, and the unmodified
reader-time playback-reference snapshot—on independent sample cursors, bound to
one capture coordinate. Record only closed, typed, transcript-free observations
for VAD, partial/final ASR, every endpoint evaluation and commit, final selection
or abort, barge-in, and admitted/started/terminal playback. Persist the exact
endpoint configuration and bounded adaptive-pause state, then replay decisions
in order during validation. Bind reply playback to its acoustic input generation
and effective terminal receipt; distinguish reply, auxiliary, latency-ack, and
reminder playback.

Publish a manifest only after the single writer closes, syncs, hashes, and
validates every artifact. Write a private same-directory temporary and commit it
atomically without replacing an existing path. Any admission loss, malformed
coordinate or observation, lifecycle mismatch, missing receipt, unclean close,
I/O failure, or replay mismatch makes the bundle incomplete or publication
failed. Finalization stays bounded and first-close-wins; a timed-out caller sees
`finalizing` while the writer retains ownership to a terminal state.

## Context / why

The July vault failure retained only post-GTCRN microphone audio. It could show
that speech reached one late stage, but could not locate whether words changed at
capture/resampling, gain/DSP, ASR selection, endpointing, final rewrite, runtime
admission, or playback. Ordinary logs are best-effort and text-oriented, while
stateful resampling and APM can give adjacent stages different sample counts.
Treating their files as one shared cursor would manufacture alignment.

A single processed-mic WAV cannot answer that causal question. Free-form event
metadata or transcript hashes would enlarge the privacy surface without proving
stage order. TTS admission cannot substitute for sink onset or a terminal
receipt, and an endpoint commit cannot be trusted without replaying the exact
adaptive state that produced it. Writing a final manifest directly also permits
a late filesystem error to contradict a visible, apparently complete artifact.

## Consequences

The normal `python -m core --session` and `./live.sh` paths remain the only
application and physical entry points. Diagnostics are additive and cannot own
capture, endpoint, task, or playback behavior; failures make evidence unusable
without breaking the voice path. Complete bundles are mode 0600, transcript-free,
strictly shaped, hash-bound, and internally replayable. `complete` means the
manifest is atomically visible and valid at publication time; swallowed
post-commit directory-sync errors do not promise survival across a filesystem or
power failure.

The bundle consumes additional private disk space and can remain `finalizing`
briefly after bounded shutdown. It still does not contain physical raw-microphone
samples, prove speaker audibility, grade transcript accuracy, or validate a room,
device, AEC route, natural conversation, or live barge-in. Those claims require a
fresh owner-run `./live.sh` A/B and, later, explicit audible-playout acknowledgement
for remote sessions.
