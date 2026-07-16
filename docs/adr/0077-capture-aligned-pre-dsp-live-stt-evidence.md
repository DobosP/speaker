# ADR-0077: Capture aligned pre-DSP live STT evidence

Date: 2026-07-16
Status: accepted

## Decision

Make the normal `./live.sh` session automatically retain three private,
sample-aligned tracks: the recognizer-rate microphone before application gain
and DSP, the processed microphone track, and the playback reference. Keep the
pre-DSP sidecar an explicit opt-in for the portable low-level `python -m core`
entry point. Open and close all tracks as one recorder lifecycle, use private
mode-600 files, record their locations only in the private run summary, and
preserve alignment through capture reopen, frame-size differences, shutdown,
and recorder rollback.

Do not change recognizer, VAD, language, input gain, denoise, endpointing, or
two-path agreement policy from the current live evidence. Compare the aligned
pre- and post-application-DSP tracks in the next physical run before selecting
a tuning change. This refines ADR-0075's automatic evidence bundle and does not
supersede it or ADR-0072's still-red physical barge verdict.

## Context / why

The 2026-07-16 vault-phrase run had six admitted, unclipped user windows and no
capture-reopen, decode, or finalizer failure. Playback residual separation was
about 50 dB, but both SenseVoice final paths remained poor and the domain word
`vault` was missed in every window. The retained microphone WAV was already
post-GTCRN. It therefore could not distinguish speech damaged by the application
frontend from an accent, recognizer, or domain-vocabulary failure.

Relaxing the agreement guard, raising gain, changing VAD, or disabling denoise
from that ambiguous evidence could admit more wrong speech without addressing
the cause. A control-path `diagnose_run` PASS also does not grade transcription
accuracy. The missing comparison signal must be collected before tuning.

## Consequences

- The next normal live test remains one command and automatically yields the
  aligned comparison needed to localize the STT failure.
- The pre-DSP track is after the selected host capture path and sample-rate
  conversion; it is not an unprocessed physical microphone or proof that host
  echo cancellation is correct.
- Private live sessions consume one additional mono WAV of local storage. All
  audio and verbatim run evidence remain ignored and must not be uploaded,
  committed, pushed, or pasted.
- Headless tests prove lifecycle, alignment, metadata, rollback, and shutdown.
  They do not prove improved STT, microphone behavior, or exact Stop barge-in.
