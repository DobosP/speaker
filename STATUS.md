# Status — speaker

Single source of truth: this file > newest accepted ADR > everything else; dated handoffs are history.

Last verified: 2026-07-12 on Linux ROG; full: 3563 passed/19 skipped/9 warnings; strict recorded/APM: 9/6 passed.
ADR-0051 exact behavioral revision `6db50a9`: deterministic 42/42; warm production-hybrid MiniCPM Q8/Gemma each 42/42 (`133003`).
ADR-0054 exact live self-scalar: real topology 4/4, warm 1.6–2.2 s, PRIVATE/control-owned; general recall stays fenced/model-routed.
Inject 3x: 6/6 cuts/0 self-cuts; recorded two-block owner overlap: 2/2 at 0.414/0.611 s; live remains red/unlandable (ADR-0052/53).

## Runtime

- Local-first always-on assistant: `core/VoiceRuntime` + sherpa-onnx; launch with
  `python -m core --engine sherpa`. Raw audio never leaves the device (ADR-0001).
- Current host resolves `desktop_gpu_4090`; MiniCPM5-1B Q8 is local text,
  gemma3:12b is complex/vision, and warm MiniCPM TTFT was 0.12–0.14 s. Phone Q4
  uses native XML tools; desktop Gemma/Ollama retains textual ReAct (ADR-0020/0033).
- Anchored high-confidence requests take deterministic ACT/search/research paths;
  ambiguous room speech remains on MiniCPM's learned addressing (ADR-0024/0051).
- Current host capture/output use PipeWire `echo-cancel-source`/`echo-cancel-sink`.
  GTCRN denoise is active. Generic word-cut requires four novel words plus
  speaker authority; in-app AEC/APM are off (ADR-0045). EC nodes/Ollama are
  session-only, not persistent services.
- The host-local InputAGC remains boost-only. Boost is applied only when the
  current raw block clears its calibrated floor; below-floor PCM stays unity
  while state is retained. The v5 behavior remains live-unvalidated (ADR-0047).

## Voice reliability now implemented

- VAD owns live ASR segments/acoustic time; first onset rebases bounded pre-roll.
  Pre-VAD text cannot publish/finalize and no-text episodes expire. Ordinary
  partials/finals require periodic dynamic 80/80 ms or steady ≥120 ms calibrated
  pre-gain patterns; unavailable and bounded handoffs abstain/bypass (ADR-0046/48).
- Word-cut uses isolated recognition and bounded PCM. Production generic cuts
  need four novel words plus warmed compatible speaker authority; local short
  floors cannot reopen promotion. Canonical `stop speaking`, attested short
  repairs, and the exact SenseVoice `DON'T PLAY SPEAK` repair at owned 1.4–2.0 s
  stay bounded exceptions; empty streaming finals fail closed (ADR-0026/42/53).
- Capture recovery rebinds rate/resampler, preserves the first correctly timed
  block, and treats host `-9999` as REOPEN. Shutdown epoch-fences effects before
  bounded abort/teardown; active owners retain resources and block restart.
  Same-domain recovery preserves calibration/evidence; changed domains relearn
  outside complete speech epochs. Failed retry leaves authority cleared (ADR-0043/48).
- Startup retries transient/clipped/VAD windows once; invalid profiles abstain; recovery also retries profile instability (ADR-0048).
- Enrollment provenance v5 fingerprints current-signal-only AGC after live open;
  v2/v3/v4 InputAGC records fail open and require re-enrollment, while exact non-
  AGC aliases remain. Stable route/rates/resampler/OS processing stay covered;
  volatile ambient stays excluded and measured voice admission stays shared (ADR-0047).
- Native startup, doctor, and live-session preflight share one resolved-profile
  readiness contract. Selected ASR/TTS/VAD/denoise/KWS/punctuation/AEC artifacts
  and active PipeWire routes fail closed instead of silently degrading (ADR-0016).
  Windows voice-communications/word-cut is unavailable until a constructible,
  verified capture API replaces the unsupported setting (ADR-0019).
- Synchronous capabilities use bounded cancellable coordinators; failed plan
  tools cannot retry, and failed web may make one fenced local fallback.
  Barge/timeout blocks stale output and caps abandoned providers at six; CPU
  llama.cpp cancellation clears memory before context release (22.4 ms Q4
  pre-token with healthy reuse) (ADR-0021/30/51).
- Cancellable preprocessing lets clean finals bypass model cleanup; bounded exact-
  word/repeat/session-fact and post-barge exact-word replies are controller-owned.
  General history uses role messages under a 320-token cap; response-only turns
  keep identity, tools/actions, and durable-memory authority stripped (ADR-0023/39/51).
- Engine finals separate admission from `UNKNOWN`/`VERIFIED`/`REJECTED` identity.
  Only a finite enrolled final-gate match marks live audio owner-verified; disabled,
  unavailable, error, loudness-rescue, mixed, and 0.30-only barge paths cannot.
- Terminal sink receipts—not sample ratios—govern spoken history; admitted replies
  stamp `TTS_REQUESTED`, preserving first-audio watchdog coverage without a model token.
  TTS speaker ID is session-locked; voice tags cannot switch it, while finite
  unsupported control tags strip and other brackets remain visible (ADR-0027/28/29/38/41/51).

## Live evidence and limits

- Historical v4 enrollment on the PipeWire EC route: 512 dimensions, 0.58
  minimum/0.78 mean similarity; runtime `114725` accepted its fingerprint.
- At 13% OS gain `114725` damaged/lost commands; `115512` false-cut, INGESTed the
  override, then hit -9999/corruption; `130601` had fragmented ASR (ADR-0036).
- Main `75b1717` run `144211` kept `sid=18` and cut once, but INGESTed the override;
  tag/response admission fixes remained headless-only (ADR-0038/0039).
- Main `6d8e9c2` run `170840` stayed `sid=0` and exited cleanly once, but owner
  talk-over decoded only `AH`, scored 0.19–0.23, and made zero cuts. Ear grade,
  lifecycle causality, and unplug/switch remain unvalidated (ADR-0041/42/43).
- `173340` exposed stale AGC overshoot; `193818` made two silent zero-word cuts
  at 0.31/0.32 and spawned a raw-empty `I.` reply. The four-word guard prevents
  that path. `200747` made zero active cuts while stale/fresh garble entered; v5
  closes below-floor pumping and calibrated pattern evidence guards ordinary turns,
  but one echo-like tail handed off after playback. Live remains red (ADR-0045/47/48).
- Real Q4 MiniCPM passed no-think/pre-TTS filtering, bounded 4/8, native cancellation,
  reuse, and two phone-lite XML tool round trips (ADR-0031/32/33). Phone thermals remain unvalidated.
- Still required at the mic after fresh v5 enrollment: quiet `YES`, low sensitivity,
  self-echo, override, mid-thought pause, reply-tail continuity, and STOP. Do not claim barge validated.

## Standard verification
`/home/dobo/work/speaker/.venv/bin/python -m pytest tests -q`; APM: `python -m pytest tests/test_apm_double_talk.py -q`;
`/home/dobo/work/speaker/.venv/bin/python -m tools.conversation_eval --runs 3`; `git diff --check`.

Real models: `python tools/run_tests.py real_model`; MiniCPM auto-pair: `python -m tools.llm_sanity --production-threads`; native tools: `python -m tools.minicpm_tool_sanity`; host: `python -m tools.doctor`.

## Operating policy
- Queue: `.agents/backlog.md`; architecture: `docs/unified_architecture.md` and `docs/audio_pipeline.md`; decisions: append-only `docs/adr/`.
- Direct merge/push to `main` is authorized only after every gate is green (ADR-0014). Never land a red suite.
- Do not delete logs/expose secrets/claim unrun hardware validation; public-history PII cleanup stays owner-deferred with no history rewrite (ADR-0008).
