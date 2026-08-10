# ADR-0175: Report runtime AEC delay provenance in echo probe

Date: 2026-08-10
Status: accepted

## Decision

Keep `strategy.aec_ref_delay_ms` in `tools.echo_probe` as the configured seed and
add an `aec_reference_delay` diagnostic block. Report configured AEC intent,
actual AEC construction, auto-delay intent/activity, exact seed and operating
samples, derived milliseconds, explicit estimate acceptance, and a closed state.
Read the runtime calibrator only after `engine.stop()` returned normally and both
the running flag and capture-resource hold prove the capture writer quiesced.
Otherwise report the runtime snapshot as unavailable.

Expose one frozen scalar `AecDelaySnapshot` from `AecDelayCalibrator`. It contains
only sample rate, effective seed samples, current operating samples, and an
explicit acceptance boolean. Never inspect the calibrator's rolling PCM arrays or
infer acceptance from a changed numeric delay: an accepted estimate may equal the
seed. The probe may describe the final operating snapshot, but must not attribute
its aggregate ERLE to that final value because a run can span the seed and several
accepted delays.

This extends ADR-0012's echo-probe evidence and ADR-0174's rebuild derivation. It
does not supersede either decision and does not change AEC construction,
auto-delay authority, configuration, defaults, DSP, barge-in, or tool authority.

## Context / why

`AecDelayCalibrator` already owns the live operating reference delay, while the
probe previously emitted only the configured `aec_ref_delay_ms`. With auto-delay
enabled that field is a seed, not necessarily the value used after an accepted
measurement. Reporting only the seed beside aggregate ERLE hid whether AEC was
inactive, failed open, still awaiting evidence, or operating on a measured value.

Reading `_acquired`, `_operating`, or rolling buffers directly would couple the
probe to mutable private state and could race the capture thread. Bounded shutdown
can also return while a capture owner is retained. A one-call scalar snapshot plus
the existing retained-resource signal gives the diagnostic a narrow fail-closed
boundary without adding a hot-path lock.

A static “best delay” suggestion or config rewrite is rejected. Runtime
cross-correlation already owns this choice when auto-delay is enabled, and one
probe run cannot establish a durable per-device optimum or safe live behavior.

## Consequences

- The additive states distinguish AEC disabled, configured but unavailable,
  fixed configured delay, auto-delay awaiting a measurement, accepted
  measurement, missing calibrator, and unavailable post-stop snapshot.
- Existing consumers remain compatible: the legacy strategy field is unchanged
  and `tools.interrupt_suite` ignores the additive object.
- Exact samples retain precision; milliseconds are derived from the snapshot's
  sample rate and rounded only for presentation.
- Headless tests cover explicit acceptance equal to the seed, reset semantics,
  non-16-kHz conversion, fail-open construction, retained shutdown, inconsistent
  snapshots, impossible pending/fixed states, JSON-scalar privacy, and unchanged
  summary consumption.
- Focused delay/AEC/denoise gate: 89 passed, 1 absent-DTLN-model case skipped.
- Adjacent capture/calibration gate: 134 passed, 1 absent-DTLN-model case skipped.
- Standard APM/DTD regression: 6 passed.
- No model, network, microphone, audio device, open-speaker, GPU, physical-delay,
  ERLE-quality, latency, or live path ran. Live AEC validation remains required.
