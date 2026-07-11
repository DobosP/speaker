# Status — speaker

Single source of truth: this file > newest accepted ADR > everything else; dated handoffs are history.

Last verified: 2026-07-11 on Linux ROG, `fix/enrollment-measured-ambient`; full
headless: 3043 passed, 24 skipped, 9 existing warnings; real-model: 5 passed,
12 skipped; APM/DTD: 6 passed; compilation/whitespace passed. Matched-device live
enrollment passed; normal-use low-gain and open-speaker barge A/B exposed failures.

## Runtime

- Local-first always-on assistant: `core/VoiceRuntime` + sherpa-onnx; launch with
  `python -m core --engine sherpa`. Raw audio never leaves the device (ADR-0001).
- Current host resolves `desktop_gpu_4090`; MiniCPM5-1B Q8 is the local text tier,
  gemma3:12b is complex/vision main, and warm MiniCPM TTFT was 0.12–0.14 s.
  Phone/phone_lite Q4 use native XML only for registered read-only planner steps;
  desktop Gemma/Ollama retains textual ReAct (ADR-0020/0033).
- ACT routing now requires command-shaped markers; informational action-word
  questions stay on MiniCPM and ambiguous terms use it to disambiguate (ADR-0024).
- Current host capture/output use PipeWire `echo-cancel-source`/`echo-cancel-sink`.
  GTCRN denoise is active. Word-cut is the open-speaker barge path; in-app AEC/APM
  are off (ADR-0013). EC nodes/Ollama are session-only, not persistent services.
- The host-local InputAGC flag remains on. Prior ear grading found pumping, while
  low-level word-cut evidence favored gain; do not flip it without a live A/B.

## Voice reliability now implemented

- VAD owns live ASR segments/acoustic endpoint time. Idle PCM is capped to 0.8 s
  pre-roll; complete speech is retained through the rule-3/endpoint bound. Finals
  need observed speech before SenseVoice/identity/LLM; no-VAD keeps bounded
  pre-partial audio (ADR-0017). The 0.900 s owner clip yields `CASTLE DEATH`
  streaming and `Cancel that.` in SenseVoice; other short rewrites fail closed (ADR-0026).
- Word-cut uses isolated recognition and bounded candidate PCM; a confirmed cut
  replays it exactly once into normal ASR/finalization. A 300 ms ring preserves
  onset before delayed VAD. A 1–3-word reply tail needs fresh post-playback VAD
  and a bounded deadline; own/empty/stale/silent tails drop. Four novel words or
  a canonical stop (including `cancel that`) cut immediately (ADR-0013/0026).
- Capture recovery rebinds actual rate/resampler, resets dependent state, preserves
  the first correctly timed block, and revalidates fallback. Same-domain recovery
  preserves its calibrated AGC floor; changed domains relearn from VAD-quiet
  pre-AGC blocks. Speaker/word-cut authority stays cleared until compatible.
- Enrollment provenance v2 is checked after live open/input calibration and covers
  actual route, rates, resampler, OS processing, front end, and calibrated AGC.
  Enrollment/live embeddings use the same voiced slice; sustained raw pre-AGC
  evidence now uses measured ambient, not the expanded/clamped runtime AGC floor;
  silence/transient guards remain. Mismatch is fail-open (ADR-0018/0034).
- Native startup, doctor, and live-session preflight share one resolved-profile
  readiness contract. Selected ASR/TTS/VAD/denoise/KWS/punctuation/AEC artifacts
  and active PipeWire routes fail closed instead of silently degrading (ADR-0016).
  Windows voice-communications/word-cut is unavailable until a constructible,
  verified capture API replaces the unsupported setting (ADR-0019).
- Synchronous capabilities use bounded cancellable task coordinators; barge/timeout
  blocks stale output and caps abandoned providers at six (ADR-0021). Ollama cancels
  its owned async request. Exact v0.3.33 CPU llama.cpp now captures task cancellation
  after construction, clears native memory, and resets before releasing its shared
  context. Actual MiniCPM5 Q4 pre-token stream cancel took 22.4 ms with zero pieces
  and healthy same-context reuse (ADR-0030). Model load/warm, native deadlock,
  arbitrary providers, and cloud-enabled Hedge losers remain bulkhead-only.
- Addressing, cleanup, and both routing layers use a separate bounded cancellable
  final-preprocessing lease before any AgentTask (ADR-0023/0025). Partial/final/direct
  input fences every unheard output; only assistant-eligible add-ons retain lineage
  across gates, bus backlog, completion, and playback. Queued audio/controls,
  confirmations, follow-ups, memory writes, and shutdown have generation/epoch
  ownership. A real no-mic gate cancel took 157.9 ms with zero old pieces and a
  healthy ACT/follow-up.
- ScriptedEngine, Sherpa, and FileReplay use terminal sink receipts. Sherpa owns
  FIFO spans/off-audio-thread callbacks/bounded cut fade; FileReplay atomically
  attests numeric null-sink clips; sample ratios never imply words (ADR-0027/0028/0029).

## Live evidence and limits

- Matched-device enrollment completed from three 12 s clips on the active PipeWire
  EC route: 512 dimensions, similarity 0.58 minimum/0.78 mean. Runtime `114725`
  accepted its fingerprint and reported the enrolled speaker-ID gate.
- At the owner's 13% OS source setting, run `114725` heard speech but SenseVoice
  damaged or lost commands; runtime calibration did not make normal use reliable.
  In controlled run `115512`, moderate hardware gain removed that immediate limit,
  but word-cut falsely cut on the assistant's own “rings of Saturn” and then the
  real Roman-architecture override was INGESTed instead of preserved. Shutdown
  also hit PortAudio -9999 followed by allocator corruption. These are red evidence.
- Capture recovery/recalibration is headless-only; no live device unplug/switch validation ran.
- Real Q4 MiniCPM passed no-think/pre-TTS filtering, bounded 4/8, native
  cancellation/reuse, and two deterministic phone-lite XML local-tool round trips
  (ADR-0031/0032/0033). Phone thermals remain unvalidated; today's live
  microphone/speaker barge batch is red.
- Still required with the owner at the mic: fix/verify low-sensitivity normal use,
  self-echo rejection, override preservation, mid-thought pause, bare `stop`, and
  reply-tail continuity. Do not claim barge validated until the full batch passes.

## Standard verification
`/home/dobo/work/speaker/.venv/bin/python -m pytest tests -q`;
`/home/dobo/work/speaker/.venv/bin/python -m pytest tests/test_apm_double_talk.py -q`;
`git diff --check`.

Real models: `python tools/run_tests.py real_model`; MiniCPM auto-pair: `python -m tools.llm_sanity --production-threads`; native tools: `python -m tools.minicpm_tool_sanity`; host: `python -m tools.doctor`.

## Operating policy
- Queue: `.agents/backlog.md`; architecture: `docs/unified_architecture.md` and `docs/audio_pipeline.md`; decisions: append-only `docs/adr/`.
- Direct merge/push to `main` is authorized only after every gate is green (ADR-0014). Never land a red suite.
- Do not delete logs/expose secrets/claim unrun hardware validation; public-history PII cleanup stays owner-deferred with no history rewrite (ADR-0008).
