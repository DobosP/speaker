# Status — speaker
Single source of current truth: this file > newest accepted ADR in docs/adr/ > everything else (AGENTS.md, Docs discipline).
Last verified: 2026-10-10 (headless admission/resume/decision/reference/TTS gates; owner live stopped; prior platform receipts retained).
Runtime/evidence facts are as of 2026-08-21 (ADR-0209) unless dated; frozen receipts are verbatim in `WORKLOG.md`.
## Current state — runtime
- GitHub Actions are manual-only (ADR-0223). Dormant web HTTP/auth/JWT/static/limits/probes are Go, with explicit bounded Python text IPC;
  Python audio/inference/actions remain; Go-only image has no Python/models. Tokens default-deny; pre-body refusals close promptly and HTTP/IPC reject unpaired Unicode escapes (ADR-0224).
- `python -m core --session` is the one public core entry: `VoiceSession` owns one injected `VoiceRuntime`, `build_runtime` is the sole
  tool/authority plane, and repeat-previous and continuation/resume lineage are fenced (ADR-0123/0154). Audio is device-only by default;
  trusted-LAN is unselectable until owner live A/B and legacy remote is rollback only (ADR-0096/0097/0164); web qualification: `docs/go_serving_boundary.md`.
- `./live.sh` is the single Linux physical entry (host lock, reversible echo route, conditional Ollama, doctor gate, private evidence); the
  `sense-voice`/`parakeet-faster-whisper` final-STT profiles fail closed instead of degrading, `--no-speaker-enrollment` is a session-only
  identity downgrade, and the production whole-turn replay failed closed at row 9 (ADR-0075/0077/0144/0146/0152).
- `./live.sh --guided-stt-capture` publishes an immutable 16-case close/far plan and an effect-free `stt_capture_only` core; short of 16/16
  plus the manifest is red and no mic run has followed (ADR-0157). `tools.guided_stt_pair_attestor` is the only paired qualifier; exit zero
  means attestation completed, not a win (ADR-0158).
- Desktop MiniCPM Q8 is the local text tier, Gemma3 handles complex/vision, and phone Q4 uses native XML tools with thermal unvalidated
  (ADR-0020). Mobile: ADR-0201–0204 own generation, reply subscription, ASR/listening and the app-root `AgentSession`; the mobile-hybrid
  Zipformer is the sole endpoint/control authority (ADR-0207) and ADR-0208 admits only offline desktop-CPU evidence (ADR-0186/0205/0206).
  Python-plane convergence, native cleanup, a disjoint owner holdout and phone validation stay open; ADR-0200 is unlanded provenance.
- ADR-0209 retains launcher-owned Go templating, streaming RMS and required enrolled word-cut authority on 4090. The original reference
  remains incompatible; the fresh isolated candidate passed live frontend compatibility on 2026-10-05, with acceptance/promotion pending.
- ADR-0225/0226 retain template/label boundaries; ADR-0227 gates admission before replacement while retaining canonical commands/vault requests and bounded vocatives; ADR-0228 bounds local decisions.
  Explicit Qwen+spoken factory canaries pass (ADR-0232); GPU mixed turns pass; CPU startup/semantic failures and rejected compact prompt remain open (ADR-0235).
- ADR-0210 binds desktop barge onset grace to the first rendered samples of the same playback generation; retained fades, slow synthesis
  and stop/start races cannot lend a timestamp to a successor. Pre-audio control and later talk-over remain available; live A/B is pending.
- ADR-0211/0212 retain one mobile TTS lookahead and four ASR credits plus bounded pending PCM, original capture boundaries and session-end revocation.
  Exact playback cleanup releases WAVs; uncertain cleanup retains them. ADR-0220 scopes active assets and ADR-0221 freezes startup thread requests.
  Phone first-audio, sustained RTF/WER, thermal and acoustic behavior remain unvalidated.
- ADR-0215 repairs prospective AMI/Microsoft import closures and private repository-lock test staging; original receipts stay historical.
- ADR-0213/0219 warm media first and optionally only the fast local tier; main/vision remain callable with per-role Ollama residency. ADR-0214
  offline acceptance plan in `docs/local_voice_performance.md`. Phone English-only selection and physical quality/thermal gates stay open.
- ADR-0216/0217 bind English candidate evidence (37 scripted/6 pinned mic clips, separate) and explicit preflighted Kitten; native defaults
  stay unchanged. `docs/english_model_comparison.md` owns measurements/licenses; factory generation passed, with no live/phone promotion.
- ADR-0218 adds current/responsive/compact startup modes with bound streaming/voice assets and unchanged capability/privacy/quality controls;
  ADR-0219 residency, ADR-0220 mobile asset scope, ADR-0221 startup pools and ADR-0222 native lowpass are documented in `docs/performance_modes.md`.
- Bounded PRIVATE vault search, reminders and trusted apps are opt-in; mutations need unchanged direct speech plus confirmation,
  `web.search` admits only `current_turn_only`, and retained context forces monotonic `local_only`. Residuals: cleaner byte channel,
  `last_source` race, blocking-backend cancellation, raw-gate false negatives (ADR-0003/0060/0073/0074/0076/0187/0189).
- ADR-0191 bounds the shipped SearXNG reader: `httpx==0.28.1`, one context-managed identity raw stream, clamped frozen
  `WebSearchConfig` budgets (`max_results` 1-8), a cooperative monotonic total deadline plus `Event` checkpoints, and detail-free refusal
  codes for bad content type/encoding/length, oversized bodies, non-strict UTF-8 and non-exact JSON/result shapes. Web search stays
  disabled by default and injected custom backends keep the one-argument `search(query)` protocol, inheriting no transport bound.
- ADR-0192 makes `CapabilityRegistry` return one fixed `capability_provider_failed` for any unexpected provider exception or
  non-`CapabilityResult` return, never binding, stringifying, typing or logging the thrown object, so no tool/backend detail reaches an
  observer, `TASK_FAILED` or the next ReAct prompt. Provider-authored errors pass through byte-for-byte and `BaseException` still
  propagates after one fixed finished receipt; residual: `error` is now coarse, so callers cannot distinguish unexpected failure kinds.
- Factory-built Hedges publish one owner per source worker and BUSY never waits or spawns, but a hung owner can hold a billable cloud
  request per key (ADR-0021/0030/0189/0190); model-setup archive writes go through one bounded no-follow extractor (ADR-0165/0167). Only
  case, ordinary surrounding spaces and one trailing terminator keep final text trusted; any other rewrite loses direct-live and owner
  action authority (ADR-0133), and cleanup sees only the newest four user utterances (ADR-0173). Conversation pairs, semantic memory, owner
  replay and synthetic-delay gates are stable (ADR-0051/0065/0067/0068/0070/0080); `run_tests.py cloud` carries the Hedge owner matrix
  (ADR-0190).
## Current state — voice reliability
- ADR-0185 bounds the KWS speaker-inference lifecycle (one unfinished task process-wide, a code-owned 50 ms safety deadline, capture-owned
  reduction, busy/cold/unavailable abstain); hostile config bytes, native bounds, other wheel builds and the owner bare-speaker A/B stay
  pending (ADR-0042/0072/0137/0152/0171/0172/0183–0185).
- Open-speaker barge-in must work without enrollment; four-novel-word cuts are identity-optional, while multi-voice and own-TTS-ambiguous
  STOP need compatible speaker authority (ADR-0008/0072/0137/0152). `semantic-interruption-policy-v1` and the Anyreach contracts are inert
  (ADR-0168–0170); playback-time KWS effects need the pinned token binding (ADR-0171/0172/0182).
- Sherpa native reads use a 300 ms/eight-frame capture-only MediaSession (ADR-0088), rebuilds re-derive the eight optional AEC/coherence/DTD
  roles (ADR-0174), and async endpoint finalization crosses a bounded route-neutral media stage (ADR-0107). Echo-probe acceptance binds to
  `./live.sh` plus `python -m tools.live_audio_ab logs/runs/run-<id>.txt` (ADR-0175/0177–0181); the interrupt suite is a diagnostic with
  `live_validation_required` always true (ADR-0176).
- The default Sherpa decode owner retains DSP/VAD/ASR/confirm/word-cut/endpoint; ADR-0222 uses native SOS lowpass with scalar fallback.
  ADR-0230 bounds reference-ring copies; ADR-0233/0234 prevent TTS replay and isolate output failure; ADR-0231 adds offline LocalVQE comparison without AEC adoption.
  Replay is headless evidence (ADR-0110/0111); acoustic identity stays immutable (ADR-0084/0086); VAD/recovery retain segment/rate evidence
  (ADR-0043/0046/0048). Only a finite enrolled final match mints owner trust (ADR-0027/0041/0051).
- The opt-in Linux final pair is checksum-pinned Parakeet Unified English plus a Faster-Whisper Small verifier with SenseVoice defaults
  unchanged (ADR-0078/0080/0144/0188); streaming hotwords need the pinned English Zipformer BPE family (ADR-0114). Smart Turn v3.2 is opt-in
  and lexical stays the endpoint default; EdAcc diagnostics are partial evidence with the owner A/B pending, the logical-turn kernel is
  inactive and schema 3–5 shadow modes are aggregate-only (ADR-0135/0136/0147–0150/0153/0155/0156).
- Public matrix v5 has 8 tracks, 19 sources and 11 exclusions over four private 24-case corpora and green is coverage-only
  (ADR-0098/0101/0109/0113/0120/0122/0127/0129–0132/0142/0143/0151); endpoint catalog v1 pins the LiveKit EoT shard endpoint-only
  (ADR-0134/0136) and the command/noise lock binds 57 schema-v4 cases whose atomic pair rejected promotion (ADR-0117/0118/0145).
- The conversation flow-v1 gate composes 4x3 deterministic journeys with no mic, TTS or live audio (ADR-0112); capabilities use per-task
  `TurnHandle`s, tools do not retry and failed web gets one local fallback (ADR-0021/0030/0051/0086/0094). Terminal receipts govern spoken
  history and diagnostic schema v2 binds four private PCM16 tracks plus the f32le final-input spool
  (ADR-0028/0029/0038/0100/0108/0124/0126/0138).
## Live evidence and limits
- Exact physical STOP is red: `192151`/`193713` failed with enrollment on and off and route settling is unproven (ADR-0072). The 2026-07-16
  vault run recognized `vault` 0/6 and kept only post-GTCRN mic audio, so the failure seam is unknown (ADR-0077).
- Private replay WER 0.00 is non-disjoint, the BPE candidate is not promotable and the public-v3 Small control stays rejected (ADR-0087);
  Common Voice SPS v4 is unrun and the first VoxPopuli preparation failed closed (ADR-0101/0106). Development-only after-PCM comparisons (no
  endpoint, live or adoption validity) favour Faster-Whisper Small over Zipformer on EdAcc, AMI and NOTSOFAR; every WER and digest is in the
  cited ADRs and `WORKLOG.md` (ADR-0102/0109/0143/0151/0159/0161/0162/0194/0196).
- Moonshine, Nemotron, Parakeet NeMo, parakeet.cpp and Kyutai are rejected or benchmark-only with no capture, VAD, endpoint, AEC, fixture,
  live or promotion validity; on the 57-case command/noise gate SenseVoice stays the live control
  (ADR-0090/0091/0099/0115–0119/0121/0122/0128/0145/0160). Legacy mic-only AMI is .852 and the natural-turn diagnostic is descriptive only
  (ADR-0092/0166).
- Windows communications capture verifies OS AEC/NS/beamforming with owner talk-over/STOP open (ADR-0081/0082) and diagnostic replay fails
  closed on malformed heartbeats (ADR-0083). The exact LiveKit 1.1.14/Agents 1.6.8/API 1.2.0/protocol 1.1.21 closure is pinned; this host has APM 1.1.10 on 2026-10-10, the Microsoft AEC slice is locked behind five term acceptances, and the exact-input tool-route gate has no owner-labelled run
  (ADR-0108/0124–0126/0163/0164).
## Open gates
- Owner bare-speaker / live A/B pending for barge-in, KWS, Smart Turn and the 4090 route (ADR-0185/0209).
- Trusted-LAN and Microsoft AEC gates remain open (ADR-0163/0164); repeat 4090 route enrollment/A/B before use (ADR-0209); ADR ledger open.
## Next
- Desktop-first plan: `docs/local_voice_performance.md`; admission/decision fixes are headless-verified; answer quality and live A/B remain open.
  Guided close/far STT, disjoint commands/multi-voice, native-reader/gap, bare-speaker/live latency evidence stay open (ADR-0157/0158).
- Paul stopped live testing on 2026-10-05; no microphone/doctor/live retry until owner resume. Preserve the candidate, backup and private evidence.
- Accept the five Microsoft AEC terms, materialize the fixture, provision exact LiveKit 1.1.14 alone and run the bounded replay; preserve
  the remote closure and run its self-hosted owner live A/B (ADR-0163/0164). Kyutai stays quarantined until its bounded resource gate;
  PriMock overlap and AMI natural-turn stay diagnostic-only (ADR-0160/0161/0162/0166).
- Mobile ASR: preserve every ADR-0207/0208 artifact; next evidence is a private owner-recorded holdout disjoint between tuning and verdict,
  then phone CPU/RSS/thermal/mic validation. Zipformer stays the English owner and Romanian is deferred (ADR-0203/0205–0208).
## Verification record (2026-10-05; commands/history in WORKLOG.md)
- Python CI-style full gate: `11701 passed, 40 skipped, 9 warnings` in 417.14 s; CI's two LiveKit files excluded, with hermetic synthetic fixtures.
- Mobile integration: `275 passed`, compiled Compact/asset gate `23 passed`, full analysis clean; current cached Sherpa 1.13.3/Gemma 0.16.5.
- Go HTTP/race/vet/native HTTP/image and fake pipe pass; Python scope `166 passed, 1 skipped`, adapter `73 passed`; installed SDK opt-in refused (missing closure).
- Owner live: fresh candidate passes 4090 Responsive compatibility; room noise still triggers instruction-copying. Stopped; route/server restored.
- Windows CPython 3.10.11: console echo exit 0 and `pip check` pass; doctor/mic deferred, no Windows READY or hardware claim.
  Sherpa 1.13.2 differs from the 1.13.3 pin; optimized assets are incomplete and enrollment persistence uses unavailable POSIX APIs.
- Headless: admission/factory 385+126, repaired requests/flow 264, resume 112; decisions 326+125; reference 437/2 skips; TTS 657/2 skips; cancellation 50; profile/identity 511+31, selection 365+17, setup 30, final model scope 210; LocalVQE 216/2 skips and 5+5 native cells; output/capture recovery 940/2 skips.
  Full CUDA FP16 doctor passed; candidate compatibility passes without promotion; live gates remain open. python-dotenv absent; pip check passes.
## Doc map
- Full index: `README.md` §Documentation. Contract `AGENTS.md`; routes `docs/agent-map.md`; gates `docs/agent-testing.md` and
  `docs/testing.md`; runbooks `docs/evaluation_runbooks.md`; architecture `docs/unified_architecture.md` and
  `docs/target_architecture.md` §9; decisions `docs/adr/` (append-only); history and receipts `WORKLOG.md`; work queue
  `.agents/backlog.md` (read by `tools/session_bootstrap.py`; the Next above wins).
