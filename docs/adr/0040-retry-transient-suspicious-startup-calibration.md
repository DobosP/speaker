# ADR-0040: Retry transient-suspicious startup calibration once

Date: 2026-07-11
Status: accepted

## Decision

Treat a complete startup input-calibration window as transient-suspicious when
its raw peak is at least `0.10` and at least 20 times its low-percentile ambient
RMS. Discard that window and measure exactly one fresh complete window within
the existing four-window read cap. Apply only a stable replacement. If the
replacement is suspicious or incomplete, retain the configured InputAGC noise
floor and report both the retry and unstable outcome through logs and metrics.

Keep stable first windows on the existing one-pass estimator and read count.
Do not change `compute_input_calibration`, enrollment calibration, or online
changed-route recalibration; this is a stream-settling guard at native startup.

## Context / why

Live run `20260711-144211` measured `ambient_rms=0.0049`, derived a `0.0148`
InputAGC floor, and also observed a `0.818` peak in the first 15 blocks read
immediately after opening the PipeWire stream. Three short utterances were then
garbled before a sustained command reached useful recognition. A high floor can
hold the boost-only InputAGC at unity for low-level blocks, but this run does not
prove the ambient estimate was inflated: calibration PCM was not retained, and
later heartbeat levels are measured after GTCRN while calibration is measured
before it. The 167-times crest proves only that the first window was not a clean
stationary observation.

Clamping the observed floor downward would mis-handle legitimately noisy
devices. Replacing the shared block-RMS estimator with sample trimming would
also change enrollment and recovery semantics without corresponding evidence.
A second raw measurement resolves the uncertainty directly: a legitimate raw
ambient repeats, while a stream-open transient settles. The absolute peak guard
avoids retrying for harmless low-level ADC ticks, and stationary loud noise has
a low crest ratio and remains one-pass.

## Consequences

- Stable quiet and stationary-noisy devices retain their existing calibration
  result, startup duration, enrollment behavior, and recovery behavior.
- A suspicious first window adds at most one calibration window; all reads stay
  under the existing cap, and no third attempt is possible.
- Repeated suspicion or an incomplete retry prefers the configured floor over a
  suspect measurement and is observable as `input_calibration_unstable`.
- Headless tests prove retry, single-pass neutrality, bounded exhaustion, and
  recovery compatibility. The ROG microphone still needs a live A/B before this
  can be claimed to fix low-sensitivity recognition.
