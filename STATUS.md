# Status — speaker

Single source of current truth. On conflict: this file > newest accepted ADR in
`docs/adr/` > everything else. Dated session/handoff documents are history.

Last verified: 2026-07-10 on Linux ROG, `fix/cancellable-final-preprocessing`;
full headless: 2636 passed, 24 skipped, 9 existing warnings; real-model: 5 passed,
12 skipped; APM/DTD: 6 passed; whitespace passed; host doctor READY outside the
sandbox on the actual EC route/models/Ollama. No human-speech A/B ran.

## Runtime

- Local-first always-on assistant: `core/VoiceRuntime` + sherpa-onnx; launch with
  `python -m core --engine sherpa`. Raw audio never leaves the device (ADR-0001).
- Current host resolves `desktop_gpu_4090`; MiniCPM5-1B Q8 is the local text tier
  and gemma3:12b remains the complex/vision main tier (ADR-0020). Warm MiniCPM
  TTFT measured 0.10–0.11 s at 1.1 GB VRAM; ASR has async SenseVoice finals.
- Current host capture is routed through PipeWire `echo-cancel-source` and output
  through `echo-cancel-sink`. GTCRN denoise is active. Word-cut is the active
  open-speaker barge path; in-app AEC/APM are off (ADR-0013). EC nodes and
  Ollama are manually active for this session, not persistent boot services.
- The host-local InputAGC flag remains on. Prior ear grading found pumping, while
  low-level word-cut evidence favored gain; do not flip it without a live A/B.

## Voice reliability now implemented

- VAD owns live ASR segments and the acoustic endpoint clock. Idle PCM is capped
  to 0.8 s pre-roll; complete speech is retained through the configured rule-3
  and endpoint bound. A configured VAD must observe speech before a final reaches
  SenseVoice, identity, addressing, or the LLM; the no-VAD fallback stays bounded
  but retains audio before a delayed first partial (ADR-0017).
- Word-cut uses an isolated playback recognizer and bounded candidate-only PCM.
  A confirmed cut replays/splices that PCM exactly once into normal ASR and the
  finalizer. A 300 ms onset ring preserves speech before delayed VAD activation.
  A 1–3-word reply tail needs a fresh post-playback VAD epoch and is bounded by a
  deadline; own/empty/stale/silent tails are dropped. The four-word/`stop`
  authority remains unchanged (ADR-0013).
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
  cancellable final-preprocessing lease before any AgentTask exists (ADR-0023).
  Partial/final/direct input fences every unheard output; assistant add-ons retain
  lineage across gates, bus backlog, completion, and actual playback admission.
  Queued stream/aux audio, controls, confirmations, follow-ups, memory writes, and
  shutdown have generation/epoch ownership and cannot resurrect stale work. A
  real no-mic MiniCPM gate cancel took 157.9 ms with zero old pieces and a healthy
  ACT/follow-up.

## Live evidence and limits

- Pre-fix run `20260710-084939` opened the real EC-routed mic/models and emitted
  raw/final `AND` about every two seconds; post-fix runs `093305`/`100432` had no
  final/`AND` storm in the latest 20 s of near-silence. No audio was recorded.
- Capture reopen, fallback, and changed-domain recalibration have deterministic
  headless coverage only; no live device unplug/switch validation was run.
- A real word-cut occurred once on this rig on 2026-07-07, but the merged tail and
  PCM-ownership changes have only deterministic headless validation.
- Current v1/legacy speaker enrollment is intentionally rejected by v2 provenance;
  local `speaker_gate_input` remains off until live re-enrollment succeeds.
- Still required with the owner at the mic: (1) `python -m core --enroll` on the
  active EC route and owner-acceptance check; (2) quiet/casual phrase accuracy plus
  mid-thought pause A/B; (3) bare-speaker sustained talk-over batch measuring cut
  rate, silent-control false cuts/tails, and reply-tail word continuity. Do not
  claim these live-validated until they actually run.

## Standard verification
```bash
/home/dobo/work/speaker/.venv/bin/python -m pytest tests -q
/home/dobo/work/speaker/.venv/bin/python -m pytest tests/test_apm_double_talk.py -q
git diff --check
```

Real models: `python tools/run_tests.py real_model`; host: `python -m tools.doctor`.

## Operating policy
- Queue: `.agents/backlog.md`; architecture: `docs/unified_architecture.md` and
  `docs/audio_pipeline.md`; decisions: append-only `docs/adr/`.
- Direct merge/push to `main` is authorized during development only after every
  required gate is green (ADR-0014). Never land a red suite.
- Do not delete logs/expose secrets/claim unrun hardware validation; public-history PII cleanup stays owner-deferred with no history rewrite (ADR-0008).
