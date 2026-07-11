# Status — speaker

Single source of truth: this file > newest accepted ADR > everything else; dated handoffs are history.

Last verified: 2026-07-11 on Linux ROG, stacked local `main`; full: 3205 passed,
24 skipped, 9 warnings; focused: 357 passed/1 skipped; APM/DTD: 6 passed; compile/whitespace passed. Live A/B is red.

## Runtime

- Local-first always-on assistant: `core/VoiceRuntime` + sherpa-onnx; launch with
  `python -m core --engine sherpa`. Raw audio never leaves the device (ADR-0001).
- Current host resolves `desktop_gpu_4090`; MiniCPM5-1B Q8 is local text,
  gemma3:12b is complex/vision, and warm MiniCPM TTFT was 0.12–0.14 s. Phone Q4
  uses native XML tools; desktop Gemma/Ollama retains textual ReAct (ADR-0020/0033).
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
  replayed/spliced once, and an empty stream may reach offline ASR. Exact STOP
  plus the attested `OF HE STOP` repair may cut; TTS containing STOP still
  requires short-window speaker authority (ADR-0026/0036/0042).
- Capture recovery rebinds rate/resampler, preserves the first correctly timed
  block, and treats host `-9999` as REOPEN. Shutdown epoch-fences effects before
  bounded abort/teardown; active owners retain resources and block restart.
  Same-domain recovery preserves its AGC floor; changed domains relearn from
  VAD-quiet pre-AGC blocks. Authority stays cleared until compatible (ADR-0043).
- Startup calibration retries once for a high-peak, 20x-ambient crest; stable raw windows stay one-pass and retry failure keeps config (ADR-0040).
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
- Addressing, cleanup, and routing use a bounded cancellable preprocessing lease.
  A post-barge live final may bypass `INGEST` only for a direct answer; identity,
  private context, tools, and continuation lineage stay stripped, and synthetic
  resumes share that envelope (ADR-0023/0025/0039). Input effects remain fenced.
- Engine finals separate admission from `UNKNOWN`/`VERIFIED`/`REJECTED` identity.
  Only a finite enrolled final-gate match marks live audio owner-verified; disabled,
  unavailable, error, loudness-rescue, mixed, and 0.30-only barge paths cannot.
- ScriptedEngine, Sherpa, and FileReplay use terminal sink receipts; sample ratios
  never imply words. The shipped TTS speaker ID is session-locked; voice tags
  sanitize without switching it, while emotion/rate stay fragment-local. Finite
  unsupported control tags strip; other brackets stay visible (ADR-0027/28/29/38/41).

## Live evidence and limits

- Matched-device enrollment on the PipeWire EC route: 512 dimensions, 0.58
  minimum/0.78 mean similarity; runtime `114725` accepted its fingerprint.
- At 13% OS gain `114725` damaged/lost commands; `115512` false-cut, INGESTed the
  override, then hit -9999/corruption; `130601` had fragmented ASR (ADR-0036).
- Main `75b1717` run `144211` kept `sid=18` and cut once, but INGESTed the override;
  tag/response admission fixes remained headless-only (ADR-0038/0039).
- Latest main `285d74e` run `154451` retried a 0.982 startup crest successfully
  (replacement peak 0.013/floor 0.0094), but switched `sid=0` to `sid=16`. Owner
  “STOP” reached word-cut text while scores 0.16–0.23 stayed below 0.30; cuts=0,
  then PortAudio -9999/allocator corruption recurred. Voice lock, exact Stop
  repair, and lifecycle fix are headless-only (ADR-0041/42/43); live stays red.
- Native close overlap is an inference; no live unplug/device-switch validation ran.
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
