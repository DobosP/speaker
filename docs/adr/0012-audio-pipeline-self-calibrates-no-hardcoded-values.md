# ADR-0012: Audio pipeline self-calibrates at runtime — no machine-specific hard-coded values

Date: 2026-07-04
Status: accepted

## Decision

The real-time audio pipeline calibrates itself on-device at runtime; it MUST NOT
depend on machine-specific hard-coded operating values. Any literal in the audio
path is either a **seed** (used only until a measurement replaces it) or a
**physics/UX/linguistic bound** (documented as not-per-machine) — never a tuned
operating point. Five fixes land under this rule (owner directive, 2026-07-04):

1. **AEC reference delay** — measured on-device by normalized cross-correlation
   (`AecDelayCalibrator`, `core/engines/_aec.py`), correlation-gated + median-
   tracked. `aec_ref_delay_ms` is demoted to a seed; the measured value drives
   cancellation. Replaces the coherence-median feedback that never converged.
2. **ASR feed** — under `_apm_owns_ns`, a second APM tap with ML noise-suppression
   OFF feeds the recognizer (auto-activated from the detected APM ownership).
3. **Cleaner** — anti-fabrication gate from dimensionless ratios of the raw's own
   tokens (`core/asr_text.py`, `core/cleanup.py`).
4. **Endpointing** — the commit floor is learned per session from the speaker's
   own pause distribution (`SessionPauseModel`); the old silence numbers become
   bounds.
5. **TTS output** — a parameter-free DC blocker (universal 20 Hz corner scaled by
   the output rate); the underrun prebuffer (fix 5b) is DEFERRED (real-time
   callback, needs live validation).

## Context / why

The 2026-07-04 real-usage forensics (7 recordings; report + backlog "REAL-USAGE
FORENSICS") showed the STT/barge bottleneck was the AEC/APM **pipeline**, not the
mic or the ASR model: `aec_ref_delay_ms` was hard-set to 40 ms on every session
while the true echo delay was 106–220 ms (corr ~0.15), so the canceller looked in
the wrong place, the always-on NS over-corrected, and the recognizer + barge gate
read a mangled signal. An A/B replay recovered a full narration the live path had
shredded. A per-machine config value cannot fix a fleet that "works on any device"
— the calibration must be measured on-device. `AecDelayCalibrator` was validated
on `run-20260702-004345`: seeded (wrong) 40 ms, it self-corrected to ~110–133 ms
once echo appeared, matching diagnose_run's ~147 ms.

**Why not tune `aec_ref_delay_ms` per machine (echo_probe once, commit the value)?**
That is exactly the pattern that failed: it drifts per device/route, was never
re-run, and violates ADR-0005 ("never hard-set"). Measured-at-runtime removes the
human step entirely. **Why not a second full APM everywhere for the ASR feed?**
It doubles APM CPU; it runs ONLY under `_apm_owns_ns` (the open_speaker profile),
and the ASR path is byte-identical (aliases the NS-on samples) otherwise.

## Consequences

- Zero per-machine audio tuning: `config.local.json`'s `aec_ref_delay_ms=40`
  override was removed; the calibrator measures it. Supersedes the operative half
  of ADR-0005 (calibrate-per-machine) — the delay is now measured, not calibrated
  by hand; ADR-0005's "never hard-set 260 ms" invariant still holds.
- Each intent flag is off-safe: features are byte-identical when disabled, and the
  standalone units (`AecDelayCalibrator`, `DCBlocker`, `SessionPauseModel`, the
  cleaner predicates) are unit-tested independently. Full logic suite green.
- **Must be revisited with a live mic on the open_speaker (apm) profile:** fix 2's
  STT recovery + no self-interrupt regression, and the barge-cut verdict — the
  autotest loopback can't judge these (`.agents/backlog.md`). Fix 5b (underrun
  prebuffer) is deferred to a session that can validate the real-time callback.
- The DC blocker, adaptive endpoint floor, and relaxed-NS tap change model-visible
  audio/text, so re-run the real-usage forensics (replay + diagnose_run) after a
  live session to confirm the garble/fragmentation actually drop on real audio.
