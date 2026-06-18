# Session 2026-06-18 — open-speaker self-interrupt FIXED + DTLN AEC enabled

**Headline:** The central P1 — the open-speaker self-interrupt (assistant cancelling
its own replies on its TTS echo) — is **fixed at the root**, diagnosed and validated
by **replaying the owner's recorded audio headlessly** (no live mic loop). Then the
**DTLN deep echo-canceller** (which was implemented but disabled) was provisioned,
proven the right backend over the owner's recording, and **enabled**. Baseline 1907
→ **1983 tests**, all green. Commits this session: `259ec21..ee7f2eb`.

## The two arcs

### A. Roadmap waves 1–4 (earlier today) — all landed on main
- **asr-tts-2** async second-pass ASR worker + **asr-tts-1** per-tier ASR policy.
- **llm-inference-3** on-device output cap + **rc-5** superseded-turn watchdog fix +
  device-profile invariant hardening.
- **control-plane-3** EWMA-scaled adaptive watchdog deadlines.
- **llm-inference-9** KV-cache K-quant + **control-plane-2** load-elastic admission.
- Each adversarially reviewed (see `docs/session_2026-06-17_async_second_pass_asr.md`).

### B. Live barge-in + AEC saga (the main story)
Live runs on the bare laptop speaker exposed that the assistant **self-cancelled
every reply**. The fix took several layers, each validated:

1. **clip/underrun diagnostics** (`d5dd659`) — surfaced that the mic was clipping
   36–65% (hot OS gain) and that `_audio_cb` silently zero-fills FIFO underruns.
2. **agreement-guard hardening** (`65bc852`) — a 2nd pass that collapses to a bare
   letter (`'POOH'`→`'Okay.'`) never overrides real words, even on a long clip.
3. **input AGC** (`8a3cb08`) — built, but its time-varying gain **broke** the
   coherence barge detector → reverted (`input_agc=false`). AGC is INCOMPATIBLE with
   open-speaker barge-in; left in the codebase, off.
4. **playback-onset grace** (`0356c00` → `c467542`) — suppress barge for 0.40s from
   **synth-start** (the self-interrupts fire 0.04–0.24s after "speaking:", during the
   synth lead-in). Helped, but longer replies still cut post-grace.
5. **THE ROOT CAUSE** (`5562d22`): replaying the owner's recording through the real
   `EchoCoherenceDetector` showed `decide()` returned **None 24× / True 0×** during
   playback — and `_looks_like_user` falls through to the **loud-mic level gate** on a
   None, which fires on the echo. The detector's own docstring promised False (not
   None) while the ref ring builds; it wasn't. Fixed: `decide()` returns **False
   (echo-only) not None while the ring is building** (ref.size>0 but <min). Validated
   on the recording: 24 None → 0, zero self-interrupts, real talk-overs still fire.

## The "iterate without the mic" tooling (the breakthrough enabler)
- **`record_playback_reference`** (`83ef27d`, `d8e22ea`): a `--record` run now also
  writes `run-<id>.ref.wav` — the played far-end, frame-aligned with the mic. When
  AEC is on it records the **FarEndRing** reference (true-playback-aligned, the exact
  far the canceller reads), so even `aec_ref_delay_ms` is calibratable headlessly.
- **`tools/replay_barge.py`**: feeds mic→`decide()`, ref→`note_playback()`, replays
  the EXACT barge decision headlessly (this is how the root cause was found).
- **`tools/aec_probe.py`**: runs the canceller over mic+ref, measures ERLE, sweeps the
  reference delay.

## DTLN AEC — was never removed; now enabled
Both tiers live in `core/engines/_aec.py` (NumPy FDAF `nlms` + `_DTLNEchoCanceller`).
Off because NLMS diverged live. Measured over the owner's recording:
- **NLMS = ~0 dB at every delay** — the laptop-speaker echo is ~88% INCOHERENT
  (nonlinear); a linear filter can't touch it. (Why it diverged/failed.)
- **DTLN = +6.2 dB ERLE, ZERO divergence**, and **real-time-safe** (mean 18.6ms / p95
  21.9ms per 100ms block, RTF 0.22 on the i9).
- **Provisioned** (into `.venv`): `pip install onnxruntime tf2onnx tensorflow-cpu` +
  `python -m tools.setup_models --aec-model --aec-model-size 512` (fetched breizhn
  tflite → `pretrained_models/sherpa/aec/dtln_aec_stage{1,2}.onnx`, wired `aec_model`).
- **ENABLED** in `config.local.json`: `aec_enabled=true`, `aec_backend='dtln'`.
  `aec_ref_delay_ms` left at 260 (placeholder — the replay's 540ms used the OLD
  coherence-queue reference; the new FarEndRing reference makes the next recording's
  best-delay the LIVE value).

## State of the open-speaker experience
| Issue | Status |
|---|---|
| Self-interrupt ("stops responding") | **FIXED** (onset grace + coherence None→False; replay-validated) |
| White-noise/glitch | **FIXED** (was the self-interrupt's start-cut bursts) |
| STT echo bleed | **DTLN now cancels it** in the post-AEC ASR feed (+6 dB) |
| STT garble on LOUD speech | residual — the owner's mic clips at the ADC (hot OS gain); only the OS-level fix or the clip-warning helps (no software un-clips) |

## Next steps (pick up here)
1. **One live `./session.sh --llm echo` run** — confirms: self-interrupt gone on EVERY
   turn, DTLN doesn't misbehave live, STT cleaner. It also re-records a mic+**FarEndRing**
   reference bundle.
2. **Calibrate `aec_ref_delay_ms` headlessly** from that new bundle:
   `python -m tools.aec_probe logs/runs/run-<new>.wav --backend dtln --device desktop --max-delay-ms 800`
   → set `aec_ref_delay_ms` to the best-ERLE delay (now the live value, via FarEndRing).
3. If DTLN ERLE is still modest live, the 256 model is lighter; or investigate the
   FarEndRing alignment further. The clip warning will flag if the hot mic recurs.

## Environment note (i9-13980HX `.venv`)
This session ADDED to the `.venv`: **onnxruntime 1.27, tf2onnx 1.17, tensorflow-cpu
2.21** (DTLN runtime + conversion). Anything touching models/audio still uses
`.venv/bin/python`. DTLN ONNX models are under `pretrained_models/sherpa/aec/`.
