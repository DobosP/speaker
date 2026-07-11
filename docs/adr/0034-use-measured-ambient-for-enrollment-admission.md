# ADR-0034: Use measured ambient for enrollment admission

Date: 2026-07-11
Status: accepted

## Decision
Keep the calibration run's measured pre-gain ambient separate from the expanded,
clamped InputAGC noise floor. Admit enrollment clips against that measured ambient
plus 6 dB, with an absolute `1e-4` floor and the existing 0.35 s sustain rule. If
live calibration did not run, retain the configured InputAGC floor as the
compatibility fallback. Continue to use the expanded InputAGC floor for runtime
gain control and capture-provenance bucketing.

## Context / why
`compute_input_calibration` already adds headroom to measured ambient and clamps
the InputAGC floor to at least `0.004`. Enrollment then reused that operating
floor and added another 6 dB, requiring pre-gain speech above `0.008`. On the ROG
PipeWire echo-cancel route, quiet ambient measured about `0.00018` and valid owner
speech about `0.003`, so standard enrollment falsely rejected a usable device.
Lowering the runtime AGC floor or recording raw ambient in provenance would change
normal capture behavior and invalidate otherwise compatible enrollment; removing
the absolute/sustain guards would admit muted devices and short transients.

With the admission points separated, the owner completed standard matched-device
enrollment from three 12 s clips (512 dimensions, pass-to-reference similarity
minimum `0.58`, mean `0.78`) on the production PipeWire route.

## Consequences
Quiet but sustained speech can enroll while muted input, a steady calibrated bed,
near-zero numerical noise, and a 0.2 s transient remain rejected. Runtime AGC and
provenance behavior do not change. This decision does not compensate normal
runtime ASR for an unusually low OS microphone sensitivity and does not validate
open-speaker barge-in; those remain separate behaviors with separate live gates.
