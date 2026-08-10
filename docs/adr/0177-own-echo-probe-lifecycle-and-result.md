# ADR-0177: Own echo-probe lifecycle and result

Date: 2026-08-10
Status: accepted

## Decision

Make `tools.echo_probe` the lifecycle owner as soon as it invokes
`engine.start()`. Every such invocation owns exactly one bounded `engine.stop()`
attempt, whether start returns or raises. Keep post-start instrumentation, the
capture warm-up, the sentence body, and the final tail inside that ownership
boundary.

Preserve `KeyboardInterrupt` after the stop attempt so operator interruption is
not converted into an ordinary diagnostic result. Treat `SystemExit` raised
after start invocation or by stop as an ordinary probe failure with exit code
`1`, including `SystemExit(0)`. Ordinary start/body/tail failures and a stop
failure emit only an error result and cannot build the normal report or read the
runtime AEC-delay snapshot. If the run already has a primary failure, retain it
as primary and record a stop failure as secondary rather than masking it.

Use one JSON emitter for ordinary success and error results. It must serialize
strict JSON, reject non-finite or otherwise invalid payloads before writing, and
fail closed to one bounded error object with exit code `1`. Keep the successful
probe fields consumed by `tools.interrupt_suite` compatible.

A normal return from bounded `stop()` remains distinct from native-owner
quiescence. If AEC is active and `_running` or `_capture_resource_hold` still
shows retained ownership, preserve ADR-0175's exit-`0` diagnostic with
`aec_reference_delay.measurement_state=snapshot_unavailable`; do not read the
calibrator and do not claim that native cleanup completed. Disabled or
fail-open AEC keeps ADR-0175's `aec_disabled` or `aec_unavailable` state.

This extends ADR-0175's post-stop snapshot boundary and ADR-0176's interrupt-suite
child contract. It supersedes neither decision and changes no runtime engine,
configuration, AEC backend, interruption authority, default, or live gate.

## Context / why

The probe previously caught an ordinary start exception before it acquired a
cleanup owner, left its one-second warm-up outside the body `finally`, allowed
body and interruption failures to escape without a JSON result, and swallowed a
stop exception before publishing a normal exit-`0` report. A start call can
acquire capture or playback ownership before it raises, so relying only on the
engine's internal startup rollback can strand work on early or asynchronous
failure. Sherpa stop is deliberately idempotent before workers and after an
ambiguous launch, making one caller-owned attempt the safe closed boundary.

Retrying stop automatically is rejected: the probe owns one bounded attempt and
must report its failure without hiding the earlier cause or extending teardown
indefinitely. Treating every normally returned stop as proof of resource release
is also rejected because Sherpa intentionally retains a possibly live native
owner rather than racing its close; ADR-0175 already defines the unavailable
snapshot for that state.

The prior JSON call also accepted non-standard `NaN`/infinite values. A child
could therefore exit zero with output that the hardened interrupt suite rejects.
Serializing completely before the single emission closes that report/exit mismatch.

## Consequences

- Start failure, post-start setup, warm-up, body, tail, ordinary stop failure,
  `SystemExit`, and double-fault precedence are deterministic and fake-testable.
- A failed lifecycle cannot publish self-interruption, coupling, coherence, ERLE,
  or runtime-delay evidence as a successful cell.
- `KeyboardInterrupt` still reaches the caller after cleanup; interrupt-suite
  subprocess handling continues to treat its nonzero child exit as a failed cell.
- A secondary stop failure remains observable without replacing the primary
  failure, while a stop-only failure is the primary error.
- The fake-only focused gate passes `176 passed, 1 skipped`; the adjacent
  capture/AEC/teardown gate passes `214 passed, 1 skipped`. Both skips are the
  same absent DTLN ONNX; the standard APM/DTD gate passes `6 passed`.
- No model, network, GPU, microphone, audio device, physical suite, bare-speaker
  session, runtime/default/backend change, or live validation follows.
