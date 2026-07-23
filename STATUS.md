# Status — speaker

Single source of truth: this file > newest accepted ADR > everything else; dated handoffs are history.

Last verified: 2026-07-17 on Linux ROG: full suite 5568 passed/16 skipped/9 warnings; Parakeet/Small aggregate WER 0.00/6 exact; strict recorded 9; APM 6; pinned artifact check, NeMo decode probe, and selected installer dry-run green.
Windows box 2026-07-17 (ADR-0081 Phase-0): 5296 passed/150 failed/70 skipped — the 150 are pre-existing Linux-lane surfaces, byte-identical to the clean-upstream baseline run (zero branch-attributable); doctor READY; playback/persona/runlog changes covered by 282 green targeted tests. Roadmap: docs/2026-07-17-performance-roadmap.md.
Clean production-hybrid v4 A/B: MiniCPM/Gemma 42/42 and Gemma/Gemma 42/42; semantic-memory PASS with PRIVATE main-only recall; MiniCPM Q8 identity verified. ADR-0067/68 repair the history and correction regressions.
ADR-0054/0060 memory gates and ADR-0055–70 headless/virtual gates are green; silent delay `041032`/`041156`: 2/2 PASS, 0 self-cuts, capture-to-cut 0.509/0.818 s, all route/cleanup proofs.
Physical runs `192151`/`193713` failed with enrollment on and off. V5 is rejected/unpromoted; word-cut enrollment is now optional, but exact Stop remains physically red (ADR-0072).

## Runtime

- Local-first `core/VoiceRuntime` + sherpa-onnx; normal Linux physical entry and setup success hint are `./live.sh`, while low-level core assumes prepared audio (ADR-0001/0075).
- Desktop MiniCPM Q8 is local text; readiness pins alias/blob/quantization/template/parameters (ADR-0020/0062). Gemma3 is complex/vision; phone Q4 uses native XML tools and no inferred desktop digest (ADR-0033).
- Anchored high-confidence requests take deterministic ACT/search/research paths; ambiguous room speech stays learned; explicit recent-thread referents use main only with desktop history, while phone stays fast (ADR-0024/0051/0067).
- Smart-save uses fast Ollama; SQLite recall routes strong subjects to fenced PRIVATE Gemma first (ADR-0057/60/65). Setup can add bounded PRIVATE vault search, durable reminders, and exact trusted apps; mutations need unchanged direct speech plus confirmation and stay out of planners (ADR-0074/76).
- `./live.sh` owns entry, host lock, conditional loopback Ollama, reversible Linux EC, doctor, and aligned private pre-DSP/processed-mic/playback-reference evidence (ADR-0075/77). GTCRN is active; four-word generic cuts are identity-optional; own-TTS-ambiguous STOP is not (ADR-0042/72).
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
- Optional checksum-pinned Parakeet Unified English CPU plus Faster-Whisper
  Small CUDA uses exact acoustic quorum. Two decoded-empty models may veto only
  a one-word non-control stream; attested controls need their exact pair/timing
  and both model votes. Generic changes lose owner/action authority. The Linux
  machine-local pair is selected; committed SenseVoice stays unchanged. Setup pins
  archive/four files/sherpa-onnx 1.13.3; readiness and replay require real decodes (ADR-0078/80).
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

- Windows box (ADR-0081): playback now resamples via soxr with sentence-end
  flush (receipts stay output-domain exact), dry gaps and dead input are
  first-class bundle metrics, run summaries carry p50/p95 stage latency, and
  the desktop profile unstarves recent-context (320→1536). Machine-local:
  gemma4:12b + pinned MiniCPM Q8 tiers restored (both verified resident,
  9.3 GB GPU), Kokoro 24 kHz + 7 kHz lowpass active. Known upstream latent:
  ADR-0068 repeat-guard clock race on coarse clocks (backlog; Linux
  unaffected in practice).
- Current plan: `docs/2026-07-17-stabilization-plan.md` — the unified
  barge/STT/maintenance sequencing (everything funnels through the Windows
  mic-level fix). Branches unified 2026-07-17: `main` is the only branch;
  `fix/live-stt-quality` (371e58b) and `feat/stt-consensus-v2` (08834e8)
  deleted after verifying their content was already landed (741e12b /
  4b9af8e).
- Windows barge route (ADR-0082): the native IAudioClient2 communications
  capture landed with the OS-effects verification contract — route verified
  ONLY when the effects framework reports AEC active on the live stream
  ("wasapi-pending" is gone; readiness/enroll/engine all probe). Measured on
  the box: AEC+NS+beamforming ON (build 26200, 2ch f32 48 kHz); doctor fully
  READY on the ADR-0013 word-cut recipe for the first time; two headless
  echo-probe runs: 4 sentences spoken, 0 self-interruptions, 0 VAD flags
  during playback (mic-level caveat applies — probe not conclusive alone).
  KWS stop-word floor wired machine-local (phone-based zipformer, own-echo
  guard, stop/wait/hold-on → stop; trigger thresholds conservative 0.25/0.30
  after phantom playback cuts at 0.15); word-cut funnel now logs the speaker-
  similarity distribution. Deferred-barge buffering parked in backlog with
  contract analysis. OPEN P1: owner-observed deterministic tail cut near
  reply end during probe runs (backlog ★ item; 1-4 self-cuts per probe on
  the noise-floor mic, attribution table recorded — undiagnosed, mic fix
  first). STILL REQUIRED before claiming barge works: OS mic-level
  fix, re-enrollment in the new wasapi-communications-aec capture domain, and
  the owner-run ADR-0013 physical gate (talk-over batch, bare stop, silent
  control) with the Kokoro voice.
- Historical v4 stays active. Enrollment `174212` captured an isolated v5
  candidate, but it failed its live gate and was never promoted (ADR-0072).
- Physical runs `192151`/`193713` failed with enrollment on and off; neither
  produced prompt exact Stop, and the second placed its blocker before identity.
  Route/calibration settling remains a hypothesis, not a root-cause proof.
- The 2026-07-16 vault run admitted six windows at -23.4 to -33.6 dBFS RMS,
  with no clipping, capture reopen, decode, or
  finalizer failure. Playback residual separation was ~50 dB, yet both
  SenseVoice final paths were poor and `vault` was recognized 0/6 times. Its
  retained mic WAV was post-GTCRN, so frontend versus recognizer/accent/domain
  failure remains unproven (ADR-0077).
- Private six-clip replay moved from original WER 0.12 to Small consensus 0.04.
  Endpoint-safe Parakeet Unified plus Small reached WER 0.00/6 exact, one win/no
  losses, deterministic direct repeats, protected STOP, and strict recorded 9/9.
  Parakeet v3 was direct-exact but tied 0.04 after endpointing; Small.en/Base
  regressed STOP. Development-only; defaults and physical evidence are unchanged (ADR-0078/80).
- Real Q4 MiniCPM passed bounded 4/8, cancellation/reuse, and two phone-lite XML tool round trips; phone thermals remain unvalidated (ADR-0031/32/33).
- Next: build a disjoint held-out set spanning vault terms, controls/near-controls,
  numerals, negation, silence/noise/echo, bystanders and multiple voices; then run
  a new `./live.sh` physical A/B. Prompt exact Stop remains physically required.

## Standard verification
Full: `...python -m pytest tests -q`; STT aggregate: `...python -m tools.recorded_stt_eval`; strict recorded: `SPEAKER_REQUIRE_RECORDED=1 ...pytest tests/replay_recorded_voice_test.py -q`; APM: `...pytest tests/test_apm_double_talk.py -q`; conversation: `...python -m tools.conversation_eval --runs 3`; then whitespace, production-hybrid Ollama A/B, sanity/tool/doctor.

## Operating policy
- Queue: `.agents/backlog.md`; architecture: `docs/unified_architecture.md` and `docs/audio_pipeline.md`; decisions: append-only `docs/adr/`.
- Direct merge/push to `main` is authorized only after every gate is green (ADR-0014). Never land a red suite.
- Do not delete logs/expose secrets/claim unrun hardware validation; public-history PII cleanup stays owner-deferred with no history rewrite (ADR-0008).
