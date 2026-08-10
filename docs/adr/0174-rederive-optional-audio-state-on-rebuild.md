# ADR-0174: Re-derive optional audio state on every Sherpa build

Date: 2026-08-10
Status: accepted

## Decision

After rejecting any rebuild whose prior streaming decode owner/session is still
live, neutralize every optional audio field that is populated only by a
successful current build:

- coherence detector and adaptive DTD;
- operating AEC reference delay and its auto-delay calibrator;
- APM always-on and noise-suppression ownership flags;
- masking-canceller residual-source selection; and
- the relaxed-noise-suppression ASR AEC frontend.

Then construct the current run's recognizers, detectors, AEC, delay estimator,
and secondary frontend normally. Only objects and capability flags returned by
those current builders may become active. A disabled option, unavailable
detector, failed-open primary AEC, disabled auto-delay, or disabled relaxed-ASR
tap therefore remains inert instead of inheriting the preceding run.

Keep the live-owner guards before this reset so a rejected rebuild leaves these
optional audio fields unchanged. Keep the primary AEC, playback reference,
delayed-reference alias, and playback-reference resampler on their existing
unconditional successful-build replacement path. Do not change AEC
configuration, defaults, healthy capture math, barge authority, source
selection, or model selection.

## Context / why

The engine initialized these optional fields once, but `_build()` assigned them
only inside positive construction branches. A stopped engine can be built
again. If the first build produced an always-on, noise-suppressing APM and the
next primary AEC failed open, the new run retained the old ownership and
masking flags, relaxed ASR frontend, delay calibrator, operating delay, and
possibly the previous coherence/DTD objects.

That stale state made the fail-open contract untrue. It could skip a newly
available denoiser with no current AEC, keep an obsolete alternate ASR
processor, feed raw-mic residual policy from a prior masking backend, record an
old delay in reader contexts, or let an old DTD continue deciding barge
classification. Disabling auto-delay or the relaxed tap on a later build had
the same retention problem.

Adding only a positive APM test would not close the lifecycle gap. The required
contract is current-build provenance: every capability combination is derived
from the current primary AEC, and positive-to-disabled/fail-open transitions
must retire every conditional object. ADR-0012 remains the governing
self-calibrating audio decision and is not superseded.

The capture-loop comment that said a newly measured delay drove the current
read was also stale. Reader-time context already captures that block's
zero-delay and operating-delay references atomically; a newly accepted estimate
correctly affects the next reader snapshot.

## Consequences

- A current AEC or detector fail-open now produces a genuinely inert current
  path even after a previously successful run.
- APM ownership is the exact conjunction of current `always_on` and
  `suppresses_noise`; residual blindness additionally follows the current
  `suppresses_nearend` capability.
- The relaxed ASR AEC and auto-delay calibrator exist only when enabled and
  eligible in the current build.
- Full rebuilds do not preserve coherence/DTD/calibrator state. Successful
  rebuilds already replaced these objects; the behavioral change is limited to
  branches that previously retained stale objects.
- Headless tests establish lifecycle and derivation logic only. They provide no
  AEC quality, model, latency, microphone, audio-device, open-speaker, GPU, or
  live validation.

## Verification

- Focused APM/ownership/rebuild gate: 46 passed, 1 missing-DTLN-model case
  skipped.
- Adjacent AEC calibration, reader-context, media-session, and decode-session
  gate: 133 passed, 1 missing-DTLN-model case skipped.
- Standard APM/DTD regression: 6 passed.
- Changed-region Ruff/format and whitespace checks are green; pre-existing
  whole-file Sherpa lint/formatter debt is unchanged from `main`.
- No model, network, microphone, audio device, GPU, or live path ran.
