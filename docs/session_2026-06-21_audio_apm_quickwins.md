# Session 2026-06-21 — audio pipeline: WebRTC APM + Teams-parity capture + fluid TTS

**Headline:** Closed the "why does Teams sound better than my app on the same
laptop" gap. Added a production WebRTC AudioProcessingModule echo-cancel backend,
wired the OS voice-comm capture path, made TTS output fluid by default, and ported
the desktop synth de-click/normalize to mobile. All landed behind safe defaults
(a clean clone is byte-identical except `tts_target_rms`); the heavy real-time
paths are opt-in and **still need live mic validation**.

**Branch → commit:** `feat/audio-apm-and-quickwins` → merged to `main`.
**Environment:** `dobo` Linux laptop, `.venv/bin/python`. Installed `soxr 1.1.0`
and `livekit 1.1.10` into the venv this session.

## What landed

| Area | Change | Files |
| --- | --- | --- |
| **WebRTC APM backend** | `aec_backend="apm"` → AEC3+RES+NS+AGC2+HPF via `livekit.rtc.AudioProcessingModule`. Measured **41–53 dB ERLE** in-vitro (NLMS managed ~0 on a nonlinear speaker). Drops into the existing `EchoCanceller` seam via `process(near, far)`; fails open when `livekit` absent. | `core/engines/_apm.py` (new), `core/engines/_aec.py::build_aec` |
| **Always-on APM** | `apm_always_on` runs the APM on every block (idle path too) for NS/AGC/HPF — desktop analog of the OS voice-comm path; skips the redundant GTCRN denoiser. ~93% idle passthrough. | `core/engines/sherpa.py` capture loop |
| **OS voice-comm capture** | `capture_voice_comm` → WASAPI Communications on Windows; PipeWire `module-echo-cancel` doc'd for Linux. | `sherpa.py::_open` |
| **AEC auto-delay** | `aec_auto_delay` (default on) feeds `echo_coherence.measured_delay_samples()` back into the far-end read delay (~1×/s) — the previously-unwired closed loop. | `sherpa.py` speaking branch |
| **Fluid TTS** | `tts_target_rms=0.12` now **default** (per-sentence loudness normalization → fixes "not fluid"). `barge_fade_ms=4` raised-cosine fade de-clicks barge cuts. `tts_declick_threshold=0.22` (was hardcoded 0.18) stops fricative smear ("robotic"). | `config.json`, `sherpa.py`, `_aec.py::PlaybackFIFO.flush` |
| **Reproducible config** | committed `open_speaker` device profile (`--device open_speaker`) turns on the full APM — no more gitignored-only config. | `config.json` |
| **Mobile parity** | ported `declick` + `normalize_rms` into the TTS worker isolate. | `mobile/lib/tts_isolate.dart` |
| **Doctor** | checks `soxr`, `livekit` (when a profile selects apm), and the PipeWire EC source (when `capture_voice_comm`). | `tools/doctor.py` |
| **Tests** | `test_apm.py` (echo-cancel + idle passthrough + the EchoCanceller-seam regression guard, self-skip without livekit), `test_barge_fade.py`, declick fricative test. | `tests/` |

**Tests:** `python -m pytest tests -q` → **2013 passed, 24 skipped**. Imports smoke 152/152.

## Why Teams is better (the answer)

Teams captures from the OS *voice-communication* path; the desktop core opened the
**raw mic** and ran no steady-state cleanup while idle. Fix = OS voice-comm path
(PipeWire/WASAPI) and/or the in-app WebRTC APM, + the GTCRN denoiser. Full writeup:
`docs/audio_pipeline.md`. Background diagnosis: memory `audio-quality-roadmap-2026-06-21`.

## Next steps (pick up here) — owner/hardware actions I could not do headless

1. **Live-validate the APM at the mic** on the open laptop speaker:
   `python -m core --engine sherpa --device open_speaker --record`, then
   `python -m tools.autotest ... barge_stress`. Confirm fp_rate→0 and cut latency
   drops vs the DTLN REVIEW baseline (`logs/barge_stress.out`). AEC3's ~40 dB ERLE
   should let the barge fire on a cleaner residual sooner.
2. **Re-enroll the speaker embedding** after enabling any capture cleanup
   (denoise/APM shifts the embedding): `python -m core --enroll`.
3. **GTCRN denoiser** (non-APM users): `python -m tools.setup_models --denoise-model`
   then `denoise_enabled=true` + `denoise_model=<path>`.
4. **Linux OS voice-comm path:** `pactl load-module module-echo-cancel aec_method=webrtc`
   and point `input_device` at the EC source (alternative to the in-app APM).
5. **`flutter analyze`** the mobile change on the mobile box (no Flutter SDK here),
   then listen for clicks gone + even loudness; consider the iOS `AVAudioSession
   .voiceChat` branch next.
6. **Consider `apm_gain_control=true`** once AGC is validated not to fight the
   barge detector live (it keys off `mic_raw`, so it should be safe).
7. If `tts_target_rms=0.12`'s +0.1 s first-audio is unwanted, A/B against `0.0`.
