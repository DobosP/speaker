# ADR-0044: Cap stale InputAGC boost on the current block

Date: 2026-07-11
Status: superseded-by ADR-0047

## Decision

Keep InputAGC boost-only and keep its existing smoothed gain state, slow rise,
fast fall, noise-floor hold, target, and maximum. For every above-floor block,
apply no more than that block's freshly derived desired boost: the effective gain
is `min(smoothed_gain, desired_gain)`, where `desired_gain >= 1`. Historical state
may therefore release smoothly without overdriving a newly louder word. Do not
attenuate below unity, tune a device-specific maximum, or hide hot ADC input.

Bump enrollment front-end provenance to version 4 and fingerprint the capped AGC
algorithm explicitly. The human-readable stage is `input-agc-capped`. Reject v2
and v3 enrollments whose model-visible chain used the old InputAGC algorithm, so
speaker authority fails open until a fresh enrollment is recorded. Preserve
exact-fingerprint v2/v3 aliases only for chains without InputAGC, whose audio is
unchanged. Retain ADR-0035's stable route/rate/resampler/OS-processing/front-end
fields and continue excluding the volatile measured ambient/AGC floor from
identity. This supersedes ADR-0035 while restating those retained rules.

## Context / why

Live run `20260711-173340` rejected a startup window with ambient RMS `0.0015`,
peak `0.820`, and crest `536.8x`; its sole replacement was also not accepted, so
startup retained the configured `0.0040` floor. During the final counting reply,
159 of 160 word-cut blocks stayed quiet, while one 100 ms post-front-end block
reached RMS `0.4633`. VAD accepted none of them; the word-cut recognizer was fed
zero blocks and made zero cuts. The run saved no raw pre-AGC PCM or applied-gain
telemetry, so it does not by itself prove that the spike was owner speech.

The controller nevertheless reproduces that value deterministically. With the
existing defaults, stale gain `12`, and an ordinary `0.061`-RMS block, desired
gain is `1.967`. The `0.4` fall leaves smoothed gain `7.987`, and the old path
emits approximately `0.465` RMS after its soft limiter. A spectral/temporal VAD
need not classify that single over-limited burst as speech. The new cap emits the
same block at the `0.12` target while preserving the `7.987` smoothed state.

Changing `fall` to `1`, lowering `max_gain`, or relying on a higher calibrated
floor would tune around this device and would not protect level changes after
legitimate quiet speech. A true bidirectional AGC is unnecessary: the defect is
stale excess boost, not a need to attenuate already-hot input.

## Consequences

- A stale high integrator cannot amplify an above-floor block past its current
  RMS target; gradual upward adaptation and below-floor hold remain unchanged.
- The algorithm changes enrollment PCM. Existing v2/v3 InputAGC enrollments are
  intentionally incompatible and the owner must re-enroll before speaker-backed
  word-cut authority returns. Unchanged non-AGC chains migrate without recording.
- Pure DSP and provenance tests cover the exact 12x-to-0.061 transition, hot-input
  visibility, old-AGC rejection, and non-AGC aliases. A fresh enrollment and bare-
  speaker A/B are still required to prove VAD/ASR survival and barge-in live.
