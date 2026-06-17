# Session 2026-06-17 (pt2) — env provisioned, speaker-ID enrolled, live open-speaker barge-in A/B

**Headline:** Provisioned the on-device runtime, enrolled speaker-ID, and ran the
first **live open-speaker barge-in A/B** on the bare laptop speaker — converting
the field-unvalidated P1 into **measured** data. No code merged this turn (latest
code is `audio-bargein-1`, merge `53b6be7`); this was environment + live
validation + analysis. The findings directly steer the next dev work.

## Environment (i9-13980HX, .venv ACTIVATED in the owner's shell)
- `.venv` now has the full on-device runtime: `scipy huggingface_hub sherpa-onnx
  sounddevice soundfile ollama` (+ deps). **Anything touching models/audio/LLM
  must use `.venv/bin/python`** — the owner's interactive `python3` *is* the venv
  (activated), but a fresh shell's `python3` is the bare `/usr/bin/python3`.
- `tools.setup_models` downloaded sherpa ASR/VAD/TTS + the CAM++ **speaker-ID** +
  **SenseVoice** second-pass models (~600 MB, `pretrained_models/sherpa/`), wired
  into `config.local.json`. Ollama is up with `gemma3:12b`/`gemma3:4b`.
  `python -m tools.doctor` → **READY**.
- Owner **enrolled** speaker-ID (`enrollment.json`, dim=512, pass-to-ref 0.77–0.80).

## Live A/B results (`--llm echo`, open speaker, no headphones)
| Run | Config | Barge | Self-interrupt | STT / notes |
|-----|--------|-------|----------------|-------------|
| `run-20260617-102023` | AEC off, SenseVoice off | **5 cut / 3 missed** | **0** ✅ | streaming-only (mediocre). **Best baseline.** Coherence-only path, echo baseline **0.70**, misses sub-0.3 s talk-overs. |
| `run-20260617-102808` | AEC **on** (NLMS @260 ms) | 1 cut / 5 missed | 0 | **NLMS diverged** (`AEC diverged … reset+passthrough`) → corrupted mic → worse STT. **NLMS-AEC ruled out.** |
| `run-20260617-103630` | AEC off, SenseVoice **on** | 3 cut / 1 missed | **several (regression)** | STT **much better** (`raw 'WHOLE HOLLO'` → `'Helen, how are you.'`) but **self-interrupts + white-noise output**. |

**Interpretation:**
- Open-speaker barge-in fundamentally **works** on the coherence-only path: it cuts
  deliberate talk-overs and (in the clean run) **never self-interrupts** — the hard
  half of the requirement held. The gap is sensitivity to brief (<0.3 s) talk-overs
  because the echo-coherence baseline is high (0.70; the reference is poorly aligned
  — measured `delay=74 ms` vs this box's real ~260 ms).
- **NLMS AEC at 260 ms diverges** and degrades everything → not the fix.
- **SenseVoice** is the STT win but its **synchronous** second pass (on the capture
  thread, audit `asr-tts-2`) contends with the real-time playback callback →
  white-noise artifacts + disrupted barge timing (the self-interrupts). This is the
  most actionable finding.

## Prepared state for the next run
`config.local.json` (machine-local, gitignored) set to the **empirically-clean
0-self-interrupt baseline**:
- `sherpa.aec_enabled = false`
- `sherpa.asr_final_backend = ""`  (SenseVoice downloaded but OFF until async)
- `sherpa.speaker_gate_input = false`  (owner's live voice scored <0.50 vs
  enrollment, so the gate rejected the owner; off so the assistant responds)

Run it with: `./session.sh --llm echo`.

## Next steps (pick up here)
1. **`asr-tts-2` — run the SenseVoice second pass ASYNCHRONOUSLY** off the capture
   thread (dispatch the streaming final immediately, upgrade in place). **Now
   data-justified** as the cause of the artifacts + self-interrupts in `103630`,
   and the way to keep the big STT win without breaking real-time. Headless-
   implementable; then re-enable SenseVoice and live-validate.
2. **Open-speaker echo:** pursue the **DTLN deep-AEC tier** (`audio-bargein-7`),
   NOT the linear NLMS that diverged. Needs the converted ONNX + live validation.
3. **Short-talk-over sensitivity (coherence path):** a controlled
   `coherence_margin_delta`/`coherence_sigma_k`/`confirm_frames` reduction —
   **live-only validation** (the Phase-0 scorecard covers the AEC-on DTD path, not
   the coherence-only path), watching for self-interrupt regression.
4. **Owner:** re-enroll speaker-ID in the normal speaking position/distance, or
   lower `sherpa.speaker_threshold` toward ~0.4, then flip `speaker_gate_input`
   back to `true`.

The run bundles are under `logs/runs/run-20260617-10*` for replay/analysis (kept
per decision D-B; the prune now protects committed bundles but these are untracked
— analyze them before many more `python -m core` runs age them out of the newest-20).
