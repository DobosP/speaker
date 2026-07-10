# Status — speaker

Single source of current truth. On conflict: this file > newest accepted ADR in
`docs/adr/` > everything else. Dated session/handoff documents are history.

Last verified: 2026-07-10 on Linux ROG, branch
`feat/minicpm5-answering-tier`. Full headless suite: 2527 passed, 24 skipped,
9 existing warnings. Real-model/replay tier: 5 passed, 12 skipped. Required
APM/DTD gate: 6 passed; whitespace gate passed. Host doctor reported READY with
the actual EC audio route, selected models, and local Ollama. No human-speech
A/B was run.

## Runtime

- Local-first always-on assistant: `core/VoiceRuntime` + sherpa-onnx; launch with
  `python -m core --engine sherpa`. Raw audio never leaves the device (ADR-0001).
- Current host resolves `desktop_gpu_4090`; MiniCPM5-1B Q8 is the local
  text/answering tier and gemma3:12b remains the complex/vision main tier
  (ADR-0020). The resolved console path passed simple/long-form routing; the
  model probe measured warm MiniCPM TTFT at 0.10–0.11 s and 1.1 GB VRAM.
  Streaming ASR is followed by asynchronous SenseVoice finals.
- Current host capture is routed through PipeWire `echo-cancel-source` and output
  through `echo-cancel-sink`. GTCRN denoise is active. Word-cut is the active
  open-speaker barge path; in-app AEC/APM are off on this host (ADR-0013).
- The host-local InputAGC flag remains on. Prior ear grading found pumping, while
  low-level word-cut evidence favored gain; do not flip it without a live A/B.
- PipeWire EC nodes and the manually started Ollama daemon are active for this
  session, not yet configured as persistent boot services.

## Voice reliability now implemented

- VAD owns live ASR segments and the acoustic endpoint clock. Idle PCM is capped
  to 0.8 s pre-roll; complete speech is retained through the configured rule-3
  and endpoint bound. A configured VAD must observe speech before a final reaches
  SenseVoice, identity, addressing, or the LLM (ADR-0017).
- The no-VAD fallback remains bounded but keeps audio before a delayed first
  partial and lets SenseVoice use owned-PCM duration.
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

## Live evidence and limits

- Pre-fix run `20260710-084939` opened the real EC-routed mic and all models but,
  with nobody speaking, emitted raw/final `AND` about every two seconds.
- Post-fix runs `20260710-093305` and `20260710-100432` used the real route/models
  for near-silence and emitted no ASR final or `AND` storm. The latest exercised
  the integrated route/runtime build for 20 s. No audio was recorded.
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

Real models: `python tools/run_tests.py real_model`. Host preflight:
`python -m tools.doctor`.

## Operating policy

- Current queue: `.agents/backlog.md`; architecture: `docs/unified_architecture.md`
  and `docs/audio_pipeline.md`; decisions: append-only `docs/adr/`.
- Direct merge/push to `main` is authorized during development only after every
  required gate is green (ADR-0014). Never land a red suite.
- Do not delete logs, expose secrets, or claim hardware validation that did not
  run. Public-history/PII cleanup remains owner-deferred at the release gate
  (ADR-0008); do not force-push or run history rewriting now.
