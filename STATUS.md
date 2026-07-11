# Status — speaker

Single source of current truth. On conflict: this file > newest accepted ADR in `docs/adr/`
> everything else. Dated session/handoff documents are history.

Last verified: 2026-07-11 on Linux ROG, `feat/file-replay-playback-receipts`; full
headless: 2786 passed, 24 skipped, 9 existing warnings; real-model: 5 passed,
12 skipped; APM/DTD: 6 passed; whitespace passed. Prior host doctor was READY outside
the sandbox on actual EC routes/models/Ollama. No human-speech A/B ran.

## Runtime

- Local-first always-on assistant: `core/VoiceRuntime` + sherpa-onnx; launch with
  `python -m core --engine sherpa`. Raw audio never leaves the device (ADR-0001).
- Current host resolves `desktop_gpu_4090`; MiniCPM5-1B Q8 is the local text tier and
  gemma3:12b remains the complex/vision main tier (ADR-0020). Warm MiniCPM TTFT was
  0.12–0.14 s at 1.1 GB VRAM and 4/4 text probes passed; ASR has async SenseVoice finals.
- ACT routing now requires command-shaped markers; informational action-word
  questions stay on MiniCPM and ambiguous terms use it to disambiguate (ADR-0024).
- Current host capture/output use PipeWire `echo-cancel-source`/`echo-cancel-sink`.
  GTCRN denoise is active. Word-cut is the open-speaker barge path; in-app AEC/APM
  are off (ADR-0013). EC nodes/Ollama are session-only, not persistent services.
- The host-local InputAGC flag remains on. Prior ear grading found pumping, while
  low-level word-cut evidence favored gain; do not flip it without a live A/B.

## Voice reliability now implemented

- VAD owns live ASR segments and the acoustic endpoint clock. Idle PCM is capped
  to 0.8 s pre-roll; complete speech is retained through the rule-3/endpoint bound.
  Finals need observed speech before SenseVoice/identity/LLM; no-VAD keeps bounded
  pre-partial audio (ADR-0017). The 0.900 s owner clip yields `CASTLE DEATH`
  streaming and `Cancel that.` in SenseVoice; selection requires owned speech
  timing, while all unlisted short rewrites remain fail-closed (ADR-0026).
- Word-cut uses isolated recognition and bounded candidate PCM; a confirmed cut
  replays it exactly once into normal ASR/finalization. A 300 ms ring preserves
  onset before delayed VAD. A 1–3-word reply tail needs fresh post-playback VAD
  and a bounded deadline; own/empty/stale/silent tails drop. Four novel words or
  a canonical stop (including `cancel that`) cut immediately (ADR-0013/0026).
- Capture recovery rebinds the actual rate/resampler, resets rate- and
  echo-dependent state, preserves the first correctly timed recovered block, and
  revalidates the actual fallback route. Same-domain recovery preserves its
  calibrated AGC floor; a changed domain relearns from VAD-quiet pre-AGC blocks.
  Speaker and word-cut authority stay cleared until compatibility succeeds.
- Enrollment provenance v2 is checked only after the live stream opens and input
  calibration completes. It covers actual route, rates, resampler, OS processing,
  active front end, and calibrated AGC range. Enrollment/live speaker embeddings
  use the same voiced slice, while sustained raw pre-AGC evidence rejects silent
  or carried-gain garbage references; mismatches leave the gate fail-open
  (ADR-0018, superseding ADR-0015).
- Native startup, doctor, and live-session preflight share one resolved-profile
  readiness contract. Selected ASR/TTS/VAD/denoise/KWS/punctuation/AEC artifacts
  and active PipeWire routes fail closed instead of silently degrading (ADR-0016).
  Windows voice-communications/word-cut is unavailable until a constructible,
  verified capture API replaces the unsupported setting (ADR-0019).
- Synchronous capabilities run behind bounded cancellable task coordinators;
  barge/timeout blocks stale output and caps abandoned providers at six (ADR-0021).
  Production Ollama streams additionally cancel an owned async request. A real,
  no-mic MiniCPM cancel ended in 135.9 ms with zero pieces and healthy follow-up
  (ADR-0022); sync generate, arbitrary providers, and llama.cpp remain bounded.
- Addressing, cleanup, and both routing layers now run behind a separate bounded
  cancellable final-preprocessing lease before any AgentTask exists (ADR-0023/0025).
  Partial/final/direct input fences every unheard output; only assistant-eligible
  add-ons retain lineage across gates, bus backlog, completion, and playback admission.
  Queued stream/aux audio, controls, confirmations, follow-ups, memory writes, and
  shutdown have generation/epoch ownership and cannot resurrect stale work. A
  real no-mic MiniCPM gate cancel took 157.9 ms with zero old pieces and a healthy
  ACT/follow-up.
- ScriptedEngine, Sherpa, and FileReplay use terminal sink receipts. Sherpa owns
  FIFO spans/off-audio-thread callbacks/bounded cut fade; FileReplay atomically
  attests numeric null-sink clips; sample ratios never imply words (ADR-0027/0028/0029).

## Live evidence and limits

- Pre-fix run `20260710-084939` emitted raw/final `AND` every ~2 s; post-fix
  runs `093305`/`100432` had no `AND` storm in 20 s. No audio was recorded.
- Capture reopen, fallback, and changed-domain recalibration have deterministic
  headless coverage only; no live device unplug/switch validation was run.
- One real word-cut occurred on 2026-07-07. Device-free concurrent I/O now covers
  real capture/playback workers, FIFO ownership/receipts, cut fade, and stale-provider
  fencing; tail/PCM and acoustic echo/owner-mic behavior remain headless-only.
- Current v1/legacy speaker enrollment is intentionally rejected by v2 provenance;
  local `speaker_gate_input` remains off until live re-enrollment succeeds.
- Still required with the owner at the mic: (1) `python -m core --enroll` on the active
  EC route; (2) quiet/casual phrase plus mid-thought-pause A/B; (3) bare-speaker
  talk-over cut/false-cut/tail continuity batch. Do not claim validated until run.

## Standard verification
```bash
/home/dobo/work/speaker/.venv/bin/python -m pytest tests -q
/home/dobo/work/speaker/.venv/bin/python -m pytest tests/test_apm_double_talk.py -q
git diff --check
```

Real models: `python tools/run_tests.py real_model`; host: `python -m tools.doctor`.

## Operating policy
- Queue: `.agents/backlog.md`; architecture: `docs/unified_architecture.md` and `docs/audio_pipeline.md`; decisions: append-only `docs/adr/`.
- Direct merge/push to `main` is authorized during development only after every
  required gate is green (ADR-0014). Never land a red suite.
- Do not delete logs/expose secrets/claim unrun hardware validation; public-history PII cleanup stays owner-deferred with no history rewrite (ADR-0008).
