# Status — speaker

Single source of truth: this file > newest accepted ADR > everything else; dated handoffs are history.

Last verified: 2026-07-11 on Linux ROG, `fix/tts-unsupported-tag-hygiene`;
focused: 135 passed, 1 skipped. Main full: 3100 passed, 30 skipped; live barge is red.

## Runtime

- Local-first always-on assistant: `core/VoiceRuntime` + sherpa-onnx; launch with
  `python -m core --engine sherpa`. Raw audio never leaves the device (ADR-0001).
- Current host resolves `desktop_gpu_4090`; MiniCPM5-1B Q8 is the local text tier,
  gemma3:12b is complex/vision main, and warm MiniCPM TTFT was 0.12–0.14 s.
  Phone Q4 uses native XML for registered read-only steps; desktop Gemma/Ollama
  retains textual ReAct (ADR-0020/0033).
- ACT routing now requires command-shaped markers; informational action-word
  questions stay on MiniCPM and ambiguous terms use it to disambiguate (ADR-0024).
- Current host capture/output use PipeWire `echo-cancel-source`/`echo-cancel-sink`.
  GTCRN denoise is active. Speaker-authorized audio-first word-cut is the
  open-speaker barge path; in-app AEC/APM are off (ADR-0036). EC nodes/Ollama are
  session-only, not persistent services.
- The host-local InputAGC flag remains on. Prior ear grading found pumping, while
  low-level word-cut evidence favored gain; do not flip it without a live A/B.

## Voice reliability now implemented

- VAD owns live ASR segments/acoustic endpoint time. Idle PCM is capped to 0.8 s
  pre-roll; complete speech is retained through rule-3/endpoint. Finals need
  observed speech; no-VAD keeps bounded pre-partial audio (ADR-0017). The 0.900 s
  owner clip's exact SenseVoice repair is allowed; other rewrites fail closed (ADR-0026).
- Word-cut uses isolated recognition and bounded PCM. Production cuts on a warmed,
  compatible enrolled-speaker score after 0.35 s of voiced audio even with zero
  ASR words; an ambiguous score starts a fresh identity window. Accepted PCM is
  replayed/spliced once, and an empty stream may reach offline ASR. Canonical stop
  stays immediate except standalone-own-echo ambiguity (ADR-0026/0036).
- Capture recovery rebinds actual rate/resampler, resets dependent state, preserves
  the first correctly timed block, and revalidates fallback. Same-domain recovery
  preserves its calibrated AGC floor; changed domains relearn from VAD-quiet
  pre-AGC blocks. Speaker/word-cut authority stays cleared until compatible.
- Enrollment provenance v3 is checked after live open/calibration and covers the
  stable route, rates, resampler, OS processing, gain algorithm, and front end;
  volatile measured AGC floor is excluded. Same-chain v2 records migrate through
  bounded runtime aliases, and enrollment prints exact nondefault launch selectors.
  Voiced slicing/measured-ambient admission remain shared (ADR-0035).
- Native startup, doctor, and live-session preflight share one resolved-profile
  readiness contract. Selected ASR/TTS/VAD/denoise/KWS/punctuation/AEC artifacts
  and active PipeWire routes fail closed instead of silently degrading (ADR-0016).
  Windows voice-communications/word-cut is unavailable until a constructible,
  verified capture API replaces the unsupported setting (ADR-0019).
- Synchronous capabilities use bounded cancellable task coordinators; barge/timeout
  blocks stale output and caps abandoned providers at six (ADR-0021). Exact CPU
  llama.cpp cancellation clears native memory before shared-context release; a real
  Q4 pre-token cancel took 22.4 ms with healthy reuse (ADR-0030). Model load/warm,
  arbitrary providers, and cloud Hedge losers remain bulkhead-only.
- Addressing, cleanup, and both routing layers use a separate bounded cancellable
  final-preprocessing lease before any AgentTask (ADR-0023/0025). Input fences every
  unheard output; only assistant-eligible add-ons retain lineage across gates and
  playback. Audio/controls, confirmations, follow-ups, memory, and shutdown have
  generation/epoch ownership; a real gate cancel took 157.9 ms with zero old pieces.
- Engine finals separate admission from `UNKNOWN`/`VERIFIED`/`REJECTED` identity.
  Only a finite enrolled final-gate match marks live audio owner-verified; disabled,
  unavailable, error, loudness-rescue, mixed, and 0.30-only barge paths cannot.
- ScriptedEngine, Sherpa, and FileReplay use terminal sink receipts; sample ratios
  never imply words. Streamed TTS snapshots task+epoch voice per reply; later voice
  switches and emotion/rate stay fragment-local. Finite unsupported control tags
  strip without changing style; other brackets stay visible (ADR-0027/28/29/37/38).

## Live evidence and limits

- Matched-device enrollment completed from three 12 s clips on the active PipeWire
  EC route: 512 dimensions, similarity 0.58 minimum/0.78 mean. Runtime `114725`
  accepted its fingerprint and reported the enrolled speaker-ID gate.
- At the owner's 13% OS source setting, run `114725` heard speech but SenseVoice
  damaged or lost commands; runtime calibration did not make normal use reliable.
  Run `115512` at moderate gain instead false-cut on own TTS, INGESTed the real
  override, and hit PortAudio -9999/allocator corruption. Run `130601` reproduced
  zero/fragmented playback ASR despite owner-energy windows (ADR-0036).
- Main `75b1717` run `144211` improved over `130601`: streamed playback retained
  `sid=18`, and the owner's “STOP” produced one cut with no self-storm. The remaining
  override was garbled then INGESTed; its spoken unsupported model tags are now
  headless-fixed (ADR-0038). Live status remains red despite the successful cut.
- Capture recovery/recalibration is headless-only; no live device unplug/switch validation ran.
- Real Q4 MiniCPM passed no-think/pre-TTS filtering, bounded 4/8, native
  cancellation/reuse, and two deterministic phone-lite XML local-tool round trips
  (ADR-0031/0032/0033). Phone thermals remain unvalidated; live barge is red.
- Still required at the mic: low-sensitivity use, self-echo rejection, override
  response, mid-thought pause, and reply-tail continuity. Do not claim barge validated.

## Standard verification
`/home/dobo/work/speaker/.venv/bin/python -m pytest tests -q`;
`/home/dobo/work/speaker/.venv/bin/python -m pytest tests/test_apm_double_talk.py -q`;
`git diff --check`.

Real models: `python tools/run_tests.py real_model`; MiniCPM auto-pair: `python -m tools.llm_sanity --production-threads`; native tools: `python -m tools.minicpm_tool_sanity`; host: `python -m tools.doctor`.

## Operating policy
- Queue: `.agents/backlog.md`; architecture: `docs/unified_architecture.md` and `docs/audio_pipeline.md`; decisions: append-only `docs/adr/`.
- Direct merge/push to `main` is authorized only after every gate is green (ADR-0014). Never land a red suite.
- Do not delete logs/expose secrets/claim unrun hardware validation; public-history PII cleanup stays owner-deferred with no history rewrite (ADR-0008).
