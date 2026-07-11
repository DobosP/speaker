# ADR-0047: Require current signal before applying InputAGC boost

Date: 2026-07-11
Status: accepted

## Decision

Apply InputAGC boost to a capture block only when that block's raw RMS exceeds
the calibrated `noise_floor_rms`. Keep the smoothed gain state unchanged during
below-floor blocks, but emit those blocks at unity. For an above-floor block,
retain ADR-0044's `min(smoothed_gain, current_desired_gain)` cap and boost-only
contract. Reset the state and its diagnostics when capture recovery resets or
recalibrates the front end, and at each new capture session after prior-owner
guards prove that the retired capture epoch is gone.

Bump enrollment front-end provenance to version 5 and fingerprint the new
`boost_only_current_signal_cap_v2` algorithm. Reject every v2, v3, or v4
enrollment whose model-visible path used InputAGC; preserve exact fingerprint
aliases for v2/v3/v4 chains without InputAGC because their PCM is unchanged.
Name the active stage `input-agc-current-signal`. This decision supersedes
ADR-0044 while retaining its current-block cap and ADR-0035's stable capture
identity fields.

## Context / why

Live run `20260711-200747` calibrated ambient RMS `0.0015` and an AGC floor of
`0.0045`, yet later admitted fresh garbled ordinary finals while its configured
final floor was inert. The run retained no raw WAV, so it cannot prove which
front-end stage caused any particular final. The controller defect is still
deterministic: after gain reaches `12`, a current `0.001`-RMS block below a
`0.004` floor was emitted at `0.012` RMS because below-floor hold applied the
stale gain as well as retaining it. Repeated `0.0046`-RMS blocks can drive the
default controller above `11x`; the next ambient block then receives that boost
indefinitely even though it is expressly classified as below-floor noise.

Decaying gain during silence still leaves a false-speech window, and faster
decay trades that window for audible level pumping at pauses. A hangover also
deliberately preserves the unsafe window. Lowering global maximum gain or using
a fixed higher floor would tune around this laptop and harm quiet microphones.
The current raw block is the narrow causal boundary: retained history may guide
later speech, but it cannot manufacture evidence that speech exists now.

## Consequences

- Below-floor ambient cannot inherit stale boost; adversarial pump-to-silence
  and alternating threshold blocks remain raw-level and cannot be amplified by
  AGC history.
- Quiet speech still receives adaptive boost whenever it clears the calibrated
  device floor. Gain continuity and the current-block target cap resume on that
  first above-floor block; hot input remains visible and is never attenuated.
- Existing InputAGC enrollment is intentionally incompatible and owner-backed
  word-cut authority remains fail-open until v5 enrollment is recorded. Non-AGC
  owners migrate without recording.
- The existing floor is measured in model-band calibration PCM while AGC runs
  before capture-rate resampling. That pre-existing cross-band limitation needs
  a separate calibrated-evidence design before claiming arbitrary-device
  robustness; it does not justify retaining known stale boost below the floor.
- Headless DSP, provenance, enrollment, and recovery tests cover the contract.
  Silent open-speaker control plus quiet speech, STOP, and four-word talk-over
  remain mandatory before live stability or barge-in can be called green.
