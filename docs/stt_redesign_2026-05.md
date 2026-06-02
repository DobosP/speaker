# Redesigned on-device STT pipeline — "really good STT quality"

> ⚠️ **Superseded — durable content merged into [`docs/unified_architecture.md`](unified_architecture.md).** Kept for revision history; do not treat as current. (2026-06-02 consolidation.)

**Goal (user):** conversational-English **word accuracy** is PRIMARY; enroll-user reliability is secondary. Local-first, STT on CPU, GPU reserved for the LLM. Fixture for before/after: `/home/dobo/work/speaker/logs/runs/run-20260529-212103.wav` (real user speech, mono 16 kHz, ~340 s, some TTS echo).

**Headline finding:** the biggest WER leaks are in the **input chain**, not the model. I verified the three dominant ones empirically on this machine (see "Evidence" inline). The model is also old (zipformer1), but a model swap fed aliased, clipped audio still underperforms — so the input chain is fixed first.

## What I verified (load-bearing measurements)

- **Aliasing (resample):** a 9 kHz tone (above the 8 kHz post-16k Nyquist) leaks **48,739,095** units of energy into the 0–8 kHz speech band with the current `np.interp` path, vs **58,593** with `scipy.signal.resample_poly` (~830×) and **1,112** with `soxr` VHQ (~44,000× less). `core/engines/sherpa.py:188-201` (`_resample_linear`), called every 0.1 s block at `:556-557`.
- **Per-block seam:** resampling a 1 kHz tone whole-stream vs in 0.1 s blocks differs by **0.068 RMS** — a ~10 Hz buzz/phase glitch that should not exist. Caused by `np.linspace` re-anchoring per block (`:200`).
- **Clipping:** `np.clip(samples * input_gain, -1, 1)` at `core/engines/sherpa.py:559` with `input_gain=8.0` (config.local.json): on a 300 Hz loud phoneme, **9.79% THD** and **43.75%** of samples pinned. On the actual recorded fixture: **9.7%** of voiced 100 ms frames hit the ceiling, **0.282%** of all samples pinned at ±1.0, median voiced RMS 0.052 with clipped peaks. Identical clip in `core/enroll.py:202`.
- **Endpoint fragmentation:** sweeping `rule2` on the fixture through the real recognizer: **0.8 → 39 finals / 12 short**, 1.2 → 31/9, **1.6 → 24/4 short**, avg words/final **7.5 → 12.0**. The fixture transcript shows the damage: "Then", "And gon out", "Up eels", "Higgins thin off sob" — and the LLM replies "I don't understand what you mean by up eels."
- **No confidence gate:** `recognizer.ys_probs` / `tokens` / `timestamps` are all present in installed sherpa 1.13.2 but never called.
- **Model:** active encoder ONNX metadata = `model_type=zipformer, version=1, encoder_dims=384×5` → zipformer1 (older arch).
- **Harness:** `core/wer.py` does **not** exist; `tools/bench/models.py:22` still pins the OLD LibriSpeech `en-2023-06-26`, not the live `en-2023-06-21`.
- **Libs:** `soxr 0.5.0`, `scipy 1.13.1`, `sherpa_onnx 1.13.2` all importable. Offline factories present: `from_nemo_ctc` (Parakeet), `from_whisper`, `from_moonshine_v2`, `from_nemo_canary`.

## Where the 16k replay does and does NOT measure a change

`FileReplayEngine` (core/engines/file_replay.py) feeds the 16k WAV **directly** to `build_recognizer`'s stream — it does **not** call `_resample_linear` or apply `input_gain`. Therefore:

- **Exercised by 16k replay:** WER harness, rule2 endpoint, beam width / blank_penalty / shallow-fusion, the confidence gate, the model swap, punctuation, the offline-engine A/B, the speaker gate.
- **NOT exercised by 16k replay (must use a 44.1k/48k source):** the resampler fix (rank 2), the 48k capture rate (rank 3), and the AGC/clip fix (rank 4) — the fixture is already aliased+clipped because it was captured through the broken path, so replaying it can neither re-alias nor un-clip. Build the source by `soxr`-upsampling the 16k fixture to 44.1k/48k (or record fresh), then A/B `np.interp` vs `soxr` HQ down to 16k through the WER harness.

## Stage-by-stage redesigned pipeline (capture → resample → gain → VAD → decode → endpoint → confidence → postproc), ranked by WER impact

1. **WER harness FIRST** — `core/wer.py` + a hand-transcribed reference sidecar; score in `tools/bench/runner.py:run_real`. Everything below is gated on a number, not anecdote. *(M, high)*
2. **Anti-alias resampler** — replace `_resample_linear` with a **stateful** `soxr.ResampleStream(capture_sr,16000,1,'float32','HQ')` (one per stream, flush on stop/reset); same change in `core/enroll.py:200`. Fixes aliasing AND the per-block seam. Add scipy+soxr to requirements. *(M, high — invisible to 16k replay)*
3. **Capture at 48000 (/3 exact)** — prepend 48000 (then 32000, 96000) ahead of device-native in the open-attempt ladder (`sherpa.py:379-390`); recovering wrapper re-walks it. Must be paired with rank 2 (naive `[::3]` is worse). *(S, medium)*
4. **Per-utterance target-RMS AGC, applied BEFORE resample** — shared `normalize_loudness(target_rms≈0.06)`: one-pole speech-gated AGC + soft-knee limiter, no hard clip; used by `sherpa.py:559` and `enroll.py:202`. Clamp default `input_gain` down; fix help/docstrings; emit `clip_fraction`/`agc_gain`. Recommend raising the OS mic level (raw avg_rms ~0.004). *(M, high — partly invisible to 16k replay)*
5. **Raise rule2 to 1.2–1.6 + per-profile + measurable endpoint latency** — and move the `SPEECH_END` stamp to endpoint-pending (`sherpa.py:628`, not `:639`). *(S, high)*
6. **Semantic end-of-turn** layered on Silero VAD (Smart Turn v3 ~8 MB) — commit on (silence>rule2 AND semantically complete) OR timeout. Removes the content-blind tradeoff. *(L, medium)*
7. **Confidence gate** — call `ys_probs`/`tokens` at the endpoint before `reset`; carry mean/min log-prob on the callback; reject/clarify when `meanLP < ~-0.9 AND words <= ~4` in `runtime._on_final`. Calibrate on the fixture; raw probs are poorly calibrated, prefer length+conf conjunction (and consider entropy/N-best). *(M, high — reduces garbage-to-LLM)*
8. **Decoder levers** — `max_active_paths` 4→8 (13× RTF headroom), plumb+A/B `blank_penalty` and **shallow-fusion LM** (`lm`/`lm_scale`/`lm_shallow_fusion`, all present in 1.13.2), and replace hardcoded `feature_dim=80` with `asr_feature_dim`. *(M, medium)*
9. **zipformer2 conversational model** — download a current k2-fsa multi-dataset/GigaSpeech streaming zipformer2 (the live one is zipformer1); repoint `config.local.json`; fix `tools/bench/models.py` to the live model. Same `OnlineRecognizer` contract → no engine code change. *(S, medium)*
10. **Punctuation + casing de-churn** — wire CT-Transformer punct (`build_punctuation` already supports it); skip partial casing when the model emits mixed case. Gives the EOU/confidence gates boundary signal. *(S, low direct WER)*
11. **VAD-segmented OfflineRecognizer "quality mode"** — `core/engines/offline_vad.py` running **Parakeet-TDT-0.6b-v2 int8** (~6% avg / 9.74% GigaSpeech WER, CPU RTF ~0.05–0.33), emitting `on_final` only; keep streaming as default + KWS for "stop". Eliminates rule2 fragmentation for free; costs live partials + sub-second decode latency. Streaming-preserving alternative: `sherpa-onnx-nemotron-speech-streaming-en-0.6b` int8 (8.20% WER, 0.56 s delay, loads via `OnlineRecognizer` unchanged). Only after ranks 1–9 are measured. *(L, high)*

## Sequencing rationale (biggest accuracy gain first)

Harness (1) is the prerequisite. Then the **input chain (2→4)** because it is provably broken — correct features benefit every downstream model and decoder; this is the largest, most certain gain and is decoupled from the model choice. Then **endpoint+confidence (5→7)** stop fragment-driven garbage turns and bad turns reaching the LLM. Then cheap **same-contract decoder/model wins (8→10)**. Finally the **offline "quality mode" (11)** for the conversational-WER ceiling, A/B'd once the gap left after the cheaper wins is known. Measure the resample/gain fixes on a synthesized 44.1k/48k source — not the 16k replay.

## Secondary: speaker-enroll reliability

The audit's gain-mismatch angle is **refuted** (enroll and live share one `input_gain` config value, so they are already level-matched). The **real, confirmed** defect is gate-specific: `_should_act_on_final` (`sherpa.py:908-921`) gates whole finals against a fixed cosine `threshold=0.5`, and short utterances embed poorly, so valid turns are dropped ("Dear me", "What's the"). Fix: (a) require a **minimum voiced length** (≥0.8–1.0 s) before the gate may REJECT, else fail-open; (b) **auto-tune threshold** from the enrollment spread (`mean − k·std`; enroll logged min 0.79/mean 0.86 so 0.5 is loose but live short clips undershoot); (c) **accumulate the embedding over the whole utterance** instead of the capped tail; (d) route enroll through the shared resampler+AGC; (e) given the WER-priority goal, consider defaulting `speaker_gate_input` **off** (gate only barge-in) until short-clip reliability is fixed — never delete a correctly-recognized turn.

## Key file pointers

- `core/engines/sherpa.py` — `_resample_linear` :188-201; capture loop :525-669; resample :556-557; gain clip :559; endpoint/final :628-646; `_should_act_on_final` :908-921; open-attempt ladder :379-390; SPEECH_END stamp :639.
- `core/engines/_sherpa_models.py` — `build_recognizer` :16-51 (feature_dim literal :37; kwargs filtered by `_supported`), `build_vad` :93, `build_punctuation` :72.
- `core/enroll.py` — `record_once` resample :200, clip :202.
- `core/engines/file_replay.py` — replay decode path :98-137 (no resample/gain).
- `core/runtime.py:199` — `_on_final` (confidence-gate hook).
- `core/app.py:324-329` — `--input-gain` help.
- `tools/bench/runner.py:42-62,109-150` — fixtures + `run_real` (WER hook).
- `tools/bench/models.py:22-23` — stale model manifest.
- `config.local.json` — `input_gain=8.0`, zipformer1 model paths; `config.json` sherpa block — `asr_max_active_paths=4`, `asr_rule2_min_trailing_silence=0.8`, empty `punct_model`, `speaker_threshold=0.5`.
- `core/wer.py` — **to be created.**
