# Status — speaker

Single source of truth: this file > newest accepted ADR > everything else; dated handoffs are history.

Last verified: 2026-07-16 on Linux ROG: full suite 5303 passed/31 skipped/9 warnings; launcher/capture/doctor 197; APM 6.
Clean production-hybrid v4 A/B: MiniCPM/Gemma 42/42 and Gemma/Gemma 42/42; semantic-memory PASS with PRIVATE main-only recall; MiniCPM Q8 identity verified. ADR-0067/68 repair the history and correction regressions.
ADR-0054/0060 memory gates and ADR-0055–70 headless/virtual gates are green; silent delay `041032`/`041156`: 2/2 PASS, 0 self-cuts, capture-to-cut 0.509/0.818 s, all route/cleanup proofs.
Physical runs `192151`/`193713` failed with enrollment on and off. V5 is rejected/unpromoted; word-cut enrollment is now optional, but exact Stop remains physically red (ADR-0072).

## Runtime

- Local-first `core/VoiceRuntime` + sherpa-onnx; normal Linux physical entry is
  `./live.sh`, while low-level core assumes prepared audio (ADR-0001/0075).
- Desktop MiniCPM Q8 is local text; readiness pins its alias, full blob,
  quantization, template, and parameters (ADR-0020/0062). Gemma3 is complex/
  vision; phone Q4 uses native XML tools and no inferred desktop digest (ADR-0033).
- Anchored high-confidence requests take deterministic ACT/search/research paths;
  ambiguous room speech stays learned; explicit recent-thread referents use main only with desktop history, while phone stays fast (ADR-0024/0051/0067).
- Smart-save uses fast Ollama; SQLite recall routes strong subjects to fenced
  PRIVATE Gemma first (ADR-0057/60/65). Setup can add bounded PRIVATE vault
  search, durable reminders, and exact trusted apps to the same chatbot; low-risk
  mutations need unchanged direct speech plus direct confirmation and stay out of model planners (ADR-0074/76).
- `./live.sh` owns the product entry, host lock, conditional loopback Ollama,
  reversible Linux EC, doctor, and private mic+reference bundle (ADR-0075).
  GTCRN is active; four-word generic cuts are identity-optional; own-TTS-ambiguous STOP is not (ADR-0042/72).
- Host InputAGC is boost-only above its calibrated floor; below-floor PCM stays unity. V5 is live-red (ADR-0047/72).

## Voice reliability now implemented

- VAD owns live ASR segments/acoustic time; first onset rebases bounded pre-roll.
  Pre-VAD text cannot publish/finalize and no-text episodes expire. Ordinary
  partials/finals require periodic dynamic 80/80 ms or steady ≥120 ms calibrated
  pre-gain patterns; unavailable and bounded handoffs abstain/bypass (ADR-0046/48).
- Word-cut uses isolated recognition and bounded PCM. Production generic cuts
  need four novel non-own words; optional multi-voice mode also requires warmed
  compatible speaker authority. Local short floors cannot reopen promotion.
  Canonical/attested/SenseVoice short repairs stay bounded; own-TTS-ambiguous
  STOP needs compatible speaker authority; empty finals fail closed (ADR-0026/42/53/72).
- Capture recovery rebinds rate/resampler, preserves the first correctly timed
  block, and treats host `-9999` as REOPEN. Shutdown epoch-fences effects before
  bounded abort/teardown; active owners retain resources and block restart.
  Same-domain recovery preserves calibration/evidence; changed domains relearn
  outside complete speech epochs. Failed retry leaves authority cleared (ADR-0043/48).
- Startup retries transient/clipped/VAD windows once; invalid profiles abstain; recovery also retries profile instability (ADR-0048).
- Enrollment v5 fingerprints current-signal-only AGC after live open; old
  InputAGC records require re-enrollment, exact non-AGC aliases remain, and volatile
  ambient stays excluded. Prep protects v4/wrong checkouts; explicit promotion
  adopts one unique mode-600 accepted copy with rollback evidence (ADR-0047/56/66).
- Native startup, doctor, and live-session preflight share one resolved-profile
  contract; selected artifacts/routes fail closed (ADR-0016). Fresh install includes
  SciPy/soxr plus SenseVoice/GTCRN/Kokoro/speaker-ID, atomically publishes only a
  complete selected config, and propagates failures. Stage one can say only `BASE
  READY` with Ollama deferred; skipped models are incomplete (ADR-0063). Windows voice-communications/word-cut remains unavailable pending a verified API (ADR-0019).
- Autonomous voice/stress verdicts require labelled WER, remembered sink onset,
  scenario-correct terminals, zero errors/stuck/self-cuts, and causal cuts. Private
  synthetic delay has a 1.4 s capture clock; other paths stay 1.0 s (ADR-0055/58/70).
- Synchronous capabilities use bounded cancellable coordinators; failed plan tools
  do not retry, while failed web gets one fenced local fallback. Barge/timeout
  blocks stale output and caps abandoned providers at six; CPU llama.cpp cancels
  pre-token in 22.4 ms Q4 with healthy reuse (ADR-0021/30/51).
- Cancellable preprocessing lets clean finals and one anchored question repair
  bypass model cleanup; broader signalled corrections remain model-owned. Bounded
  replies and exact-word/repeat/session facts are controller-owned; role history
  is capped, while response-only authority stays stripped (ADR-0023/39/51/68).
- Engine finals separate admission from `UNKNOWN`/`VERIFIED`/`REJECTED` identity.
  Only a finite enrolled final-gate match marks live audio owner-verified; direct-
  live low-risk tool authority never does. Disabled, unavailable, error,
  loudness-rescue, mixed, and 0.30-only barge paths cannot mint identity.
- Terminal sink receipts govern spoken history; admitted replies stamp
  `TTS_REQUESTED`. TTS speaker ID is session-locked; voice tags cannot switch it,
  while finite unsupported control tags strip (ADR-0027/28/29/38/41/51).

## Live evidence and limits

- Historical v4 remains active and unmodified. Enrollment `174212` captured an
  isolated three-pass v5 candidate (dim 512; similarity min 0.60/mean 0.67), but
  the candidate failed its live gate and was never promoted (ADR-0072).
- Enrollment-on `192151` held `sid=0` and answered one normal question, but
  dropped soft Yes, split a one-second pause, cut only late with garbled handoff,
  and did not stop promptly. The owner repeated the override; physical gate red.
- Enrollment-off `193713` also produced no exact-Stop final, word-cut trace, or
  cut. Playback-time VAD stayed zero despite a near-end burst and the energy
  fallback never started, placing the blocker before speaker identity. Route/
  calibration settling is a hypothesis, not yet a root-cause proof.
- Real Q4 MiniCPM passed bounded 4/8, cancellation/reuse, and two phone-lite XML tool round trips; phone thermals remain unvalidated (ADR-0031/32/33).
- Next: diagnose physical capture→EC→calibration→VAD/energy→decoder with the
  optional word-cut identity filter off; require prompt exact Stop before a new
  candidate or wider acceptance. Do not claim physical barge validated.

## Standard verification
Full: `...python -m pytest tests -q`; strict recorded: `SPEAKER_REQUIRE_RECORDED=1 ...pytest tests/replay_recorded_voice_test.py -q`; APM: `...pytest tests/test_apm_double_talk.py -q`; conversation: `...python -m tools.conversation_eval --runs 3`; then whitespace, production-hybrid Ollama A/B, sanity/tool/doctor.

## Operating policy
- Queue: `.agents/backlog.md`; architecture: `docs/unified_architecture.md` and `docs/audio_pipeline.md`; decisions: append-only `docs/adr/`.
- Direct merge/push to `main` is authorized only after every gate is green (ADR-0014). Never land a red suite.
- Do not delete logs/expose secrets/claim unrun hardware validation; public-history PII cleanup stays owner-deferred with no history rewrite (ADR-0008).
