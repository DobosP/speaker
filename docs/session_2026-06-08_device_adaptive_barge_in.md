# Session 2026-06-07/08 — Device-adaptive barge-in (WORKS on the open laptop speaker) + latency/persona, Gemma 4 on Linux

**Branch:** `fix/barge-raw-mic-and-latency` → merged to `main`. Prior same-session
work already on `main`: Gemma 4 adoption + the barge-in audit/coherence-primary
(`5cd7f60`, handoff `docs/session_2026-06-07_barge_in_coherence_primary.md`).
Logic suite **1369 passed, 14 skipped, 0 failed**.

**Machine:** i9-13980HX / 30 GiB / RTX 4090 Laptop 16 GB, Linux. Tested live on the
**built-in ALC285 laptop mic + speaker** (no premium/USB mic) per the owner directive.

## Owner directive (firm)
The assistant must work on **any device** — robust open-speaker barge-in on the
**bare laptop mic+speaker, no premium mic**, and the code must **auto-adapt** to
whatever machine + room it runs on. No fixed magic-number thresholds. (The earlier
"hardware limit → headphones / use the AT2020" conclusions are rejected.)

## Headline: open-speaker barge-in now works (live-validated)
Owner confirmed: **"barge feels good now"** — a **normal-volume** talk-over
interrupts reliably on the open laptop speaker, no shout, no self-interrupt.

### The journey (why it took several iterations)
Fixed-margin detectors all failed live because a hardcoded threshold can't fit a
loud, variable, nonlinear open speaker:
1. Coherence-alone (raw mic) → **self-interrupts** (nonlinear echo incoherence
   p50~0.88 overlaps a real voice ~1.0).
2. Coherence AND post-AEC residual energy (+10 dB) → **rejects normal talk-over**
   (DTLN suppresses the user's voice during double-talk).
3. Coherence AND raw-mic energy (+10 dB fixed) → **still needs a shout** (the echo
   is loud; +10 dB on top of a loud echo = a shout).

**Fix — `core/engines/_dtd.py` `AdaptiveDTD` (self-calibrating, no fixed margin):**
fuse three features (raw-mic energy, post-AEC residual energy, coherence
incoherent-fraction), each scored as an **upward z-score from its OWN echo-only
EWMA control chart** (the spread is learned from THIS device's echo), and fire when
the weighted **sum** D exceeds a dimensionless, device-independent K. Built when
coherence+AEC are on; the legacy coherence/level path remains for AEC-off
(headphones). Resets per speaking run; warm-up seeds the charts.

**The decisive live-data insight:** the separation was never the problem — a real
talk-over scored **D=90–130** while echo sat at **D≈0**. The bug was the *firing
logic*: the in-detector `confirm_frames=3` "consecutive" requirement discarded the
huge-D frames because a talk-over's D flickers frame-to-frame (breath/pauses).
Tuned from the per-frame logs:
- `confirm_frames` 3 → **1** (DTD reports per-frame; the capture-loop **leaky
  integrator** — `voiced_run *= 0.5` on a miss instead of reset, `barge_in_min_speech_sec`
  0.8 → 0.3 — does the temporal confirmation, tolerating the flicker).
- weights (1,1,0.5) → **(0.2, 1.0, 0.0)**: `z_resid` is the true discriminator (the
  user's voice isn't in the played reference, so AEC can't cancel it → it lands in
  the residual); `z_raw` is a shout-nudge only (it fires on loud *echo* transients);
  `z_coh` dropped (erratic/backwards on a nonlinear speaker).
- `dtd_chart_rel_floor` 0.15 → **0.4** (a small DTLN echo *leak* no longer prints a
  huge z and trips K — precision).

All knobs are `SherpaConfig` defaults, so every device gets the tuned behaviour.
`tools/echo_probe.py` now logs per-frame `(D, z_raw, z_resid, z_coh)` + the
echo-only-D-vs-K headroom, so K is **measurable/calibratable on any machine**.

### Also landed this session
- **Latency fixed:** fast tier was the 12B model; set `fast_model=gemma3:4b` → short
  replies dropped from **8–12 s to ~0.4 s** (live-confirmed). `main_model=gemma4:12b`.
- **Stop the clarification spiral:** rewrote the ASR system prompt
  (`core/persona.py _ASR`) to answer directly and clarify only as a last resort.
- **Fragmentation:** `endpoint_high_confidence_floor 0.6 → 0.0` (stop the early-commit
  that split one sentence into several finals).
- **Compute-stop on barge:** `_collect`/`_stream_and_speak` close the LLM token
  stream on cancel so the model stops generating at the barge point (not at GC).
- **Gemma 4 adopted on Linux:** Ollama 0.30.6, `gemma4:12b` (machine-local config).
- Addressing **input gate disabled** for live testing (`config.local.json`
  `device_profiles.desktop.input_gate.enabled=false`) so the assistant replies
  directly; **re-enable for normal always-on use** (so it ignores overheard speech).

### Machine-local config (`config.local.json`, gitignored)
ALC285 mic+speaker (AT2020 was unplugged); `gemma4:12b` main + `gemma3:4b` fast;
`coherence_barge_in_enabled=true`, `aec dtln ref_delay=0` (do NOT set 260ms — the
FIFO rewrite re-aligned the far-ref; a fresh echo_probe sweep peaked at 0);
`barge_in_min_speech_sec=0.3`; `asr_final_backend=""` (SenseVoice 2nd-pass reverted —
it hallucinated short clips; the agreement-guard below is the proper fix);
`input_gate.enabled=false` (testing only).

## What to improve next (owner's words, 2026-06-08)
1. **Stream the TTS for long answers — don't wait for the whole LLM answer.** A long
   story feels like it waits for the full generation before speaking. The runtime
   *has* sentence-streaming (`core/capabilities.py _stream_and_speak`), so
   investigate why long answers feel un-streamed on the sherpa path: likely the
   main-tier (gemma4:12b) **first-token latency** for a story, and/or confirm the
   sentence-by-sentence emit reaches the TTS playback incrementally end-to-end.
   Goal: first audio after sentence 1, not after the whole story.
2. **Smarter endpointing / turn-taking — don't barge in on the user's pauses.** When
   the user speaks with small mid-thought pauses, the assistant starts replying too
   early. It should use context to tell "still talking" from "done," and only
   respond with **high confidence** that the turn is complete. The lever exists: the
   **Smart Turn v3 prosody endpoint detector** is on disk
   (`pretrained_models/sherpa/smart_turn/`) but `endpoint_detector='lexical'`;
   switch to `'prosody'` (needs `onnxruntime`+`transformers`) and live-tune the
   confidence floor. See `core/endpointing.py` (the adaptive confidence-tiered floor).

## Other open follow-ups (from the design workflows, not yet done)
- **SenseVoice agreement-guard** (proper STT fix): accept the 2nd pass only when it
  agrees with / clearly improves the streaming final (kills the short-clip
  hallucination). New `core/asr_text.py` token-agreement helper + `_final_transcribe`.
- **TTS-output ducking while `_speaking`** (a few dB) — in reserve if any barge
  margin is needed on a louder/worse device; far-ref is teed after `write()` so the
  AEC reference stays consistent.
- **Auto-K / first-run `--calibrate`**: a headless echo-only sweep that seeds the
  charts + picks K per device (the full "adapts to any machine" generalization).
- Re-confirm everything on the AT2020 USB mic once plugged back in.

## Next steps (pick up here)
1. **Streaming TTS latency** for long answers (improvement #1 above) — highest owner-
   perceived value now that barge works.
2. **Prosody endpointing** to stop pause-triggered early replies (improvement #2).
3. Re-enable `input_gate` for always-on use; SenseVoice agreement-guard for STT.
