# ADR-0004: AdaptiveDTD device-adaptive barge-in; no fixed magic thresholds

Date: 2026-06-08
Status: accepted

## Decision
Open-speaker barge-in fires on `AdaptiveDTD` (`core/engines/_dtd.py`): a fused
z-score double-talk detector whose three features (raw energy / post-AEC
residual / coherence) are each self-calibrated upward z-scores from their OWN
echo-only control chart, fired on the weighted sum D > dimensionless K, with
per-frame fire (`dtd_confirm_frames=1`) + a leaky integrator in the capture
loop. NO fixed absolute thresholds anywhere in the barge path — every bar is
relative to a LEARNED per-device floor.

## Context / why
The owner's hard requirement (D-A, ADR-0008) is barge-in on the bare laptop
speaker, no headphones. Every fixed-margin generation before this failed:
speaker-gated barge (2026-06-01, `docs/archive/barge_in_speaker_gated_2026-06-01.md`),
coherence-as-primary (2026-06-02, `docs/archive/barge_in_coherence_2026-06-02.md`),
and the DTLN open-speaker stack (2026-06-03, `docs/archive/open_speaker_barge_in.md`)
each either self-interrupted, rejected normal talk-over, or needed a shout —
this ADR supersedes all three as the operative barge-in design. The decisive
fix was the FIRING LOGIC, not the physics: a real talk-over scored D=90–130
but flickered, so the old 3-consecutive-frames rule discarded it. Live
validated 2026-06-08 on the bare ALC285 (owner: "barge feels good now") and
re-confirmed 2026-06-10.

## Consequences
- Never reintroduce fixed magic-number barge thresholds; tune weights/K via
  `tools/echo_probe.py` per-frame D logs instead.
- Input AGC stays off (it breaks the coherence trigger's assumptions).
- Open P1 (2026-06-21): under `open_speaker` the DTD reads the
  APM-NS-suppressed residual, weakening z_resid — fix needs a live-mic A/B
  (see `.agents/backlog.md`); revisit weights when that lands.
