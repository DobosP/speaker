# ADR-0010: Kokoro TTS adopted as the desktop voice

Date: 2026-06-22 (re-verified 2026-06-30)
Status: accepted
Supersedes: the `en_US-libritts_r-medium` VITS default voice
Superseded-by: none

## Decision
Adopt **Kokoro** (`kokoro-int8-multi-lang-v1_1`, 103 voices, keyed on
`tts_voices` in `build_tts`) as the desktop TTS voice, replacing the muffled
libritts VITS. Expressivity is **voice-choice + rate-as-affect** via the
opt-in `tts_markup` directive tags (sherpa-onnx Kokoro exposes no latent style
vector). Bright-voice buzz on cheap open speakers is tamed by the opt-in
`tts_output_lowpass_hz` roll-off (~7 kHz on the owner's box, owner live A/B
2026-06-23). Missing Kokoro files fall back gracefully to VITS (landed
2026-07-02, `5161b0d`). Weak profiles (`phone`/`phone_lite`) stay on the cheap
streaming Piper/VITS path (profile gating = open backlog item).

## Context / why
Owner ask (2026-06-22): the TTS is "blurry, robotic, interrupted, not clear" —
wants a better voice + emotion + diversity, still cheap on-device. Diagnosis
proved the muffled timbre was intrinsic to the libritts VITS model, not
pipeline damage; live Kokoro synth measured far brighter/clearer (centroid
1763–2880 Hz vs ~800 Hz). Why not an emotion-capable cloud voice: TTS is
inside the always-on loop and must stay on-device (§9.7, ADR-0001).
Lesson recorded: judge TTS by the owner's ear on the target speaker, not by a
brightness metric.

## Consequences
- Known defect (2026-06-30, reproducible): the v1.1-zh int8 package's
  `tokens.txt` has no token for IPA `ɚ` — it silently drops unstressed "-er"
  vowels (water/teacher/computer). If voice-set finalization doesn't resolve
  "unclear", evaluate `kokoro-en-v0_19` (fp32, English) — see
  `docs/voice_upgrade_plan.md` P3.
- Voice-set finalization is **owner-gated by ear** (`tools/voice_audition`).
- `tts_target_rms`/leveler/low-pass force whole-clip synth (higher first-audio
  latency under Kokoro RTF~0.6); the streaming-compatible leveler is the
  planned fix, gated on a live A/B against the barge-in echo floor.
