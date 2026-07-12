# Status — speaker

Single source of truth: this file > newest accepted ADR > everything else; dated handoffs are history.

Last verified: 2026-07-12 on Linux ROG; combined full: 3728 passed/31 skipped/9 warnings in 76.08s; focused identity/fresh/combined: 517/121/549 passed; strict archived fake-stream replay (no hardware): 9 passed in 53.66s; APM/DTD: 6 passed in 0.69s; deterministic conversation: 42/42, all 14 scenarios 3/3, semantics/coverage/A-B/provenance/warmup true.
ADR-0051 exact behavioral revision `6db50a9`: deterministic 42/42; warm production-hybrid MiniCPM Q8/Gemma each 42/42 (`133003`).
ADR-0054 exact self-scalar: real topology 4/4, warm 1.6–2.2 s, PRIVATE/control-owned; ADR-0060 fences restart recall and promotes only strong subjects. ADR-0055–63 headless code is green.
Inject 3x: 6/6 cuts/0 self-cuts; recorded two-block overlap is 2/2 with deterministic full-window setup; live remains red/unlandable (ADR-0052/61).

## Runtime

- Local-first always-on assistant: `core/VoiceRuntime` + sherpa-onnx; launch with
  `python -m core --engine sherpa`. Raw audio never leaves the device (ADR-0001).
- Desktop MiniCPM Q8 is local text; readiness pins its alias, full blob,
  quantization, template, and parameters (ADR-0020/0062). Gemma3 is complex/
  vision; phone Q4 uses native XML tools and no inferred desktop digest (ADR-0033).
- Anchored high-confidence requests take deterministic ACT/search/research paths;
  ambiguous room speech remains on MiniCPM's learned addressing (ADR-0024/0051).
- Smart-save reuses fast Ollama; other tiers load no third model (ADR-0057/59).
  SQLite restart rows stay outside native history; strong one-line recall routes
  to fenced Gemma, but final real semantic evidence is pending (ADR-0060).
- Current host capture/output use PipeWire `echo-cancel-source`/`echo-cancel-sink`.
  GTCRN denoise is active. Generic word-cut requires four novel words plus
  speaker authority; in-app AEC/APM are off (ADR-0045). EC nodes/Ollama are
  session-only, not persistent services.
- Host InputAGC is boost-only: only a current block above its calibrated floor
  is boosted; below-floor PCM stays unity. V5 remains live-unvalidated (ADR-0047).

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
  volatile ambient stays excluded and measured voice admission stays shared.
  Prep backs up v4 without clobber, reserves a feature candidate, and final-
  publishes an inode-bound config; non-empty overwrite needs explicit opt-in
  and wrong-checkout prepared enrollment refuses before capture (ADR-0047/0056).
- Native startup, doctor, and live-session preflight share one resolved-profile
  contract; selected artifacts/routes fail closed (ADR-0016). Fresh install includes
  SciPy/soxr plus SenseVoice/GTCRN/Kokoro/speaker-ID, atomically publishes only a
  complete selected config, and propagates failures. Stage one can say only `BASE
  READY` with Ollama deferred; skipped models are incomplete (ADR-0063). Windows voice-communications/word-cut remains unavailable pending a verified API (ADR-0019).
- Autonomous voice/stress verdicts require labelled WER, remembered sink onset,
  scenario-correct terminal outcome, zero errors/stuck/self-cuts, and causal cuts.
  Cable is explicitly incomplete and cannot make `all` green (ADR-0055/58).
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
- Terminal sink receipts govern spoken history; admitted replies stamp
  `TTS_REQUESTED`. TTS speaker ID is session-locked; voice tags cannot switch it,
  while finite unsupported control tags strip (ADR-0027/28/29/38/41/51).

## Live evidence and limits

- Historical v4 on PipeWire EC: 512 dimensions, 0.58 minimum/0.78 mean; `114725`
  accepted it. It remains unmodified; isolated v5 prep/enrollment has not run.
- Runs `114725`/`115512`/`130601`/`144211` exposed 13%-gain loss, false cuts,
  INGEST of an override, corruption, fragmentation, and wrong `sid` (ADR-0036/38/39).
- Main `6d8e9c2` run `170840` kept `sid=0` and exited once but decoded the owner
  talk-over as `AH` at 0.19–0.23 and made zero cuts. Later `173340`/`193818`/
  `200747` exposed AGC overshoot, silent cuts, garble, and a tail handoff; v5
  guards are headless-only and live remains red (ADR-0041/42/43/45/47/48).
- Real Q4 MiniCPM passed bounded 4/8, cancellation/reuse, and two phone-lite XML tool round trips; phone thermals remain unvalidated (ADR-0031/32/33).
- Still required at the mic after fresh v5 enrollment: quiet `YES`, low sensitivity,
  self-echo, override, mid-thought pause, reply-tail continuity, and STOP; use
  `docs/2026-07-12-v5-bare-speaker-acceptance.md`. Do not claim barge validated.
- Current comparable-project audit: `docs/2026-07-12-comparable-voice-agent-parity.md`;
  architecture is comparable, user experience is not yet proven.

## Standard verification
Full: `...python -m pytest tests -q`; strict recorded: `SPEAKER_REQUIRE_RECORDED=1 ...pytest tests/replay_recorded_voice_test.py -q`; APM: `...pytest tests/test_apm_double_talk.py -q`; conversation: `...python -m tools.conversation_eval --runs 3`; then whitespace, production-hybrid Ollama A/B, sanity/tool/doctor.

## Operating policy
- Queue: `.agents/backlog.md`; architecture: `docs/unified_architecture.md` and `docs/audio_pipeline.md`; decisions: append-only `docs/adr/`.
- Direct merge/push to `main` is authorized only after every gate is green (ADR-0014). Never land a red suite.
- Do not delete logs/expose secrets/claim unrun hardware validation; public-history PII cleanup stays owner-deferred with no history rewrite (ADR-0008).
