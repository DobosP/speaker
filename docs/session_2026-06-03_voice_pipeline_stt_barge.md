# Session 2026-06-03 — voice pipeline: STT fixed (SenseVoice), barge-in still flaky, deep analysis

> **Status (2026-07-02):** immutable dated record — the DTLN stack + residual-floor gate and the config tunings below were superseded by `AdaptiveDTD` (`docs/adr/0004`) + WebRTC APM (`open_speaker` profile, `docs/adr/0006`; DTLN ships no models); do not apply the "CRITICAL for the other machine" config from this doc.

**Branch:** `main` (all work landed + pushed to origin). **Tests:** 1322 passed, 13
skipped (green). Worked on the **Windows desktop** (i9-13980HX, RTX 4090 Laptop).

This was a long live-iteration session driving the on-device voice app
(`python -m core --engine sherpa`) on **open laptop speakers + the Realtek
Microphone Array** (no headphones). The big win is **STT quality** (a real fix);
**barge-in on the open speaker is still at the hardware edge** and is the main
thing to continue. The user will pick this up from **other machines**.

## Branch → commit map (landed on `main` this session, oldest→newest)

| commit | what |
|---|---|
| `64aeced` | real_usage harness: **EMPTY-capture grading** (silent WAVs excluded, not "went deaf"), `--start-timeout`, `--inventory` → `logs/runs/OVERVIEW.md` |
| `9398e73` | **open-speaker barge-in**: DTLN/NLMS AEC + auto-calibrated residual-floor gate + AEC stabilization (leak/mu/divergence-guard). See `docs/open_speaker_barge_in.md` |
| `509f9b7` | **addressing prompt reframe** — fast `gemma3:4b` was dropping clear questions as ambient; now question/request/command → ACT |
| `2c6cf2e` | **DTLN onnxruntime thread bound** (`aec_num_threads=1`, no spin-wait) — fixed CPU 90-100% pegging that made the mic go deaf + turns freeze |
| `b358645` | **ASR 2nd-pass loud-warning** + **CONFIRM-gate** ("yes" no longer a dead-end; only a control reply when a confirmation is pending) |

## What actually works now vs what doesn't

- ✅ **STT quality — FIXED.** Root cause (found via whisper ground-truth + a
  20-agent analysis workflow): the offline **SenseVoice** second pass was
  *silently off* because its model was never downloaded — every final was the raw
  streaming-zipformer output ("Are you there" → `Ario der`). After
  `python -m tools.setup_models --sense-voice`, finals are clean + punctuated
  ("Are you there.", "What are your capabilities.", "...how long should I cook an
  egg to get it medium."). **The model is machine-local (gitignored) — each
  machine must run that download.** The loader now logs a WARNING if the backend
  is set but the model is missing (so it's never silently off again).
- ✅ **Addressing / "yes" confirm / DTLN CPU** — all fixed (see commits).
- ⚠️ **Barge-in (talk-over) — STILL FLAKY.** Open *nonlinear* laptop speaker is the
  documented hardware limit. DTLN cancels the echo well for ASR (residual ~0.0006)
  but **suppresses the user's voice during double-talk**, so the residual-floor
  barge gate often can't see a real talk-over and rejects it (fires on loud
  barges, rejects quiet ones). Iterated margins/min-speech/gain all session; it
  is inconsistent. The deep analysis says the real levers are FIX #5 below.
- ⚠️ **Endpointing** — it cut the user off mid-thought. Stop-gapped with
  `endpoint_min_silence_sec=1.1` + `endpoint_high_confidence_floor=0` (waits ~1.1s).
  Proper fix is FIX #6 below (prosody / Smart Turn — model already on disk).

## CRITICAL for the other machine

1. **`config.local.json` is gitignored AND was corrupted to 0 bytes mid-session
   when the disk filled** — I reconstructed it. Each machine has its own (absolute
   model paths differ). The **working tuning** (machine-independent) is:
   ```json
   "asr_final_model": "<abs>/pretrained_models/sherpa/sense_voice/model.int8.onnx",
   "asr_final_tokens": "<abs>/pretrained_models/sherpa/sense_voice/tokens.txt",
   "input_gain": 1.5,
   "endpoint_min_silence_sec": 1.1, "endpoint_high_confidence_floor": 0,
   "aec_enabled": true, "aec_backend": "dtln",
   "aec_model": "<abs>/pretrained_models/sherpa/aec",
   "aec_ref_delay_ms": 19, "barge_in_residual_margin_db": 6.0,
   "barge_in_min_speech_sec": 0.4
   ```
   (`asr_final_backend="sense_voice"` is already the committed default in
   `config.json` — only the model + paths are machine-local.)
2. **Disk was FULL (C: at 0 bytes / 100%).** This corrupted the config and degraded
   logging/recording. Free space before a long live session. I freed ~1.6 GB of
   caches (pip + a whisper model) this session.
3. **Models to fetch per machine** (gitignored `pretrained_models/`):
   `python -m tools.setup_models --sense-voice` (offline ASR, ~240 MB) and the
   DTLN AEC is already present from a prior session (`--aec-model` if not).
4. **`faster-whisper` is installed** in the venv (for `tools/transcribe_run.py`
   ground-truth) — its model cache was deleted to free disk; re-downloads on use.

## Next steps (pick up here)

The deep-analysis fix list (`logs/` workflow output) — **#1-#4 are DONE**; the
remaining high-value items, in priority order:

1. **FIX #5 — DTLN alignment + near-end protection (the barge fix).** `aec_ref_delay_ms`
   is pinned to **19 ms but the real acoustic delay is ~260 ms** (the coherence
   `delay=` logs show 258-337 ms; the 19 ms was a spurious early reading I pinned).
   Calibrate it with `tools/echo_probe.py`, and **gate DTLN to run only while
   `self._speaking` is set** (`sherpa.py` ~line 1161, the `far_rms>1e-4` engage
   gate has no speaking guard) so it stops distorting clean near-end speech. This
   is the most likely path to reliable talk-over barge-in. (Alternatively: the
   barge gate could trigger on the **coherence detector** — which sees the user's
   unexplained energy — instead of the DTLN-suppressed residual level.)
2. **FIX #6 — prosody/Smart-Turn endpointer.** `endpoint_detector="prosody"` reads
   trailing-off intonation (knows mid-thought vs done) far better than the blunt
   1.1 s silence stopgap. The **Smart Turn model is already on disk**
   (`pretrained_models/sherpa/smart_turn/`); wire the path + flip the detector.
3. **Hallucination** on the fast tier — route more to `gemma3:12b` or tune the prompt.
4. **A `"stop"` KWS fast-path** would be a robust belt-and-suspenders interrupt that
   sidesteps the echo entirely (KWS isn't built today: `models: kws=False`).

**Files most in play:** `core/engines/_aec.py`, `core/engines/sherpa.py` (capture
loop + barge gate `_looks_like_user` / `_update_playback_floor`),
`core/endpointing.py`, `core/engines/_sherpa_models.py`,
`always_on_agent/speech_analyzer.py` + `supervisor.py`, `config.local.json`
(machine-local). Reference: `docs/open_speaker_barge_in.md`.
