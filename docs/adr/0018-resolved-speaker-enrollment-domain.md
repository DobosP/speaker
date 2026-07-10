# ADR-0018: Resolve the speaker-enrollment domain after capture opens

Date: 2026-07-10
Status: accepted

## Decision

Load a persisted speaker reference only after the live input stream has opened
and optional input calibration has completed. Version-2 enrollment provenance
identifies the actual capture route, resolved capture and model rates, resampler
implementation, applied OS voice-processing mode, active capture processors,
and a 3 dB bucket of the calibrated InputAGC floor. Production enrollment uses
the same ordered capture-attempt ladder, verifies required PipeWire echo-cancel
routing, calibrates before voice clips, and aborts if the route changes between
passes. Enrollment and live identity checks both embed the same shared
energy-voiced slice. Unverified injected recordings are labeled as synthetic raw
audio and cannot match a production capture domain. Each production clip must
also contain sustained pre-AGC, model-band voice energy above the calibrated
capture floor; model embeddings and cross-pass similarity cannot admit silence
by themselves.
On capture recovery, retain the calibrated floor only when the resolved physical
domain is unchanged. A changed domain learns a new floor online from VAD-quiet,
pre-AGC blocks while speaker and playback-time word authority remain disabled.

## Context / why

ADR-0015 aligned configured processors but compared provenance during model
construction, before PortAudio had resolved the real device, rate, resampler, OS
voice mode, or calibrated AGC floor. On Linux, repointing the PipeWire default
between a raw mic and an echo-cancel source does not change a generic PortAudio
selector, so two acoustically different inputs could share a fingerprint. It
also trimmed enrollment clips while live speaker-ID embedded endpoint pre-roll
and tail, even though silence shifts the embedding. Hashing config intent was
therefore insufficient. Hashing the exact ambient floor was rejected because
normal sensor drift would disable identity gating on every boot; a 3 dB bucket
detects a material operating-point change without treating jitter as a new
front end.

## Consequences

- A stale model, route, resampler, processor chain, or materially different AGC
  operating point leaves the speaker gate unenrolled and fail-open, with a
  re-enrollment warning; it cannot silently reject the owner.
- Linux word-cut/voice-communications enrollment refuses to save a reference
  unless the active PipeWire source and sink route is verifiable.
- Existing version-1 references require `python -m core --enroll` on the intended
  route after this lands.
- A self-consistent speaker embedding of silence or carried-AGC noise is rejected
  before persistence, and changed-device recovery cannot reuse a stale AGC floor.
- Headless tests pin route/rate/resampler/calibration fingerprints, deferred
  loading, raw pre-gain admission, recovery recalibration, injected-recorder
  isolation, and byte-identical voiced envelopes.
  A real microphone re-enrollment and owner-acceptance check remain required.
