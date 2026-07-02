# ADR-0006: WebRTC APM is the production open-speaker AEC; NLMS is base-default only

Date: 2026-06-21
Status: accepted

## Decision
Use the WebRTC AudioProcessingModule (`aec_backend='apm'`: AEC3 + residual
echo suppression + NS + AGC2 + HPF) as the production echo canceller for
open-speaker operation, via the committed `open_speaker` device profile
(`apm_always_on=true`). Keep the dependency-free NumPy NLMS/FDAF only as the
base default for headset / near-field setups.

## Context / why
Live A/B on 2026-06-17 ruled NLMS out for open-speaker: a linear filter
measures ~0 dB ERLE against the nonlinear open laptop speaker — it cannot
model the speaker's distortion, so echo leaks straight into ASR. The APM
tolerates that nonlinearity (Teams-grade OS-style processing, fully
on-device, §9.7-clean) and landed as the production path on 2026-06-21. Why
not DTLN: the deep tier shipped no models (pending tflite→ONNX conversion,
fails open to no-AEC) and its NS suppresses the near-end user — the failure
that historically made the gate reject every barge. Why keep NLMS at all:
zero dependencies, real 10–20 dB ERLE on linear (headset/near-field) paths.

## Consequences
- Never present NLMS as an open-speaker option; profile selection carries the
  decision (`open_speaker` ⇒ APM).
- Open P1: the DTD barge gate reads the APM-NS-suppressed residual under
  `open_speaker` (2026-06-21 review headline; corroborated tp_rate=0.50) —
  needs a non-NS DTD tap or re-weighting, live-mic A/B required (backlog).
- With `apm_always_on`, the APM owns noise suppression (GTCRN denoiser is
  skipped); add the NS=true double-talk regression before touching weights.
