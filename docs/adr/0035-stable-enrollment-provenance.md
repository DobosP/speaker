# ADR-0035: Keep ambient calibration out of enrollment identity

Date: 2026-07-11
Status: accepted

## Decision

Persist version-3 speaker-enrollment provenance over the stable, model-visible
capture structure: the exact PortAudio selector/device identity, capture and model
rates, resampler and quality, verified OS voice-processing route, block size, AGC
algorithm settings, active APM configuration, and denoiser artifact. Keep measured
ambient and the derived InputAGC noise floor as per-start runtime state, not capture
identity. Continue using measured pre-gain ambient plus the ADR-0034 sustain and
absolute guards for enrollment admission, while runtime AGC uses its calibrated,
clamped floor.

Accept an existing version-2 enrollment only when its hash is one of the bounded
legacy hashes for the exact active structural chain, varying solely the old 3 dB
floor value. Do not carry that compatibility rule into a future provenance version.
Keep device selectors strict, and print non-default enrollment selectors in the
generated launch command so the user can reopen the same attested route.

On capture recovery, preserve calibration only for the same resolved physical
domain. A changed domain recalibrates while speaker and word-cut authority remain
disabled, as in ADR-0018.

## Context / why

Version 2 hashed the volatile calibrated floor in both its capture and AGC
descriptors. A fresh three-clip enrollment used the PipeWire echo-cancel chain and
an approximately `0.004` floor. Later starts on the identical chain measured
`0.0169` and `0.0404`; their human-readable front-end summaries were identical,
but both hashes mismatched and startup failed. Re-enrolling would only chase room
noise across buckets, contradicting device-generic auto-calibration.

A second live mismatch was legitimate: enrollment used the explicit `pipewire`
selector while the generated command reopened `default`. Both happened to resolve
through the same current PipeWire nodes, but selector identity can map differently
after device/default changes and remains attested. Launching with the exact
enrolled selector succeeded, including migration of the version-2 floor hash.

## Consequences

- Normal ambient changes no longer invalidate a reusable speaker reference or
  disable required word-cut authority.
- Route, selector, rate, resampler, OS processing, AGC algorithm, APM, and denoiser
  changes still fail compatibility; version-2 migration cannot mask them.
- Existing version-2 owners on an otherwise identical chain do not need another
  36-second recording. New saves use version 3.
- Enrollment admission remains tied to measured raw ambient and continues to
  reject mute, a steady bed, near-zero numerical noise, and short transients.
- This supersedes ADR-0018 and ADR-0034 while retaining their resolved-domain,
  voiced-envelope, recovery, and measured-admission rules.
