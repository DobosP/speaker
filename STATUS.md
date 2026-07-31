# Status — speaker

Single source of truth: this file > newest accepted ADR > everything else; dated handoffs are history.

Last verified: 2026-07-31 on Linux ROG.

- Repository logic gate (`-m "not real_model"`): 5801 passed, 13 skipped,
  20 deselected, 9 pre-existing warnings. Focused gates: typed-ingress/Sherpa/APM
  354 passed, 2 skipped; bounded session ownership 321 passed.
- Previous stable evidence remains: production-hybrid MiniCPM/Gemma and
  Gemma/Gemma 42/42, semantic-memory PASS, strict recorded owner replay 9/9,
  and two clean synthetic-delay runs (ADR-0051/0065/0067/0068/0070/0080).

## Runtime

- The Linux Python runtime is the reference. Its post-recognition `SessionActor`
  owns bounded admission, explicit binding release, identity, and serialized teardown; phone/remote shells lack it.
  Speech/fast answers stay local; only invoked post-ASR text/files/screens may
  reach a cloud tier (ADR-0001/0086).
- `./live.sh` is the single Linux physical entry. It owns the host lock,
  reversible echo-control route, conditional Ollama, doctor, and aligned
  private pre-DSP/processed-mic/playback-reference evidence (ADR-0075/0077).
- Desktop MiniCPM Q8 is the local text tier; Gemma3 is complex/vision. Phone Q4
  uses native XML tools; phone thermal behavior is unvalidated (ADR-0020/0033/0062).
- Setup may enable bounded PRIVATE vault search, durable reminders, and exact
  trusted apps. Mutations require unchanged direct speech plus confirmation and
  remain outside planners (ADR-0073/0074/0076).

## Voice reliability now implemented

- Open-speaker barge-in must work without enrollment. Generic four-novel-word
  cuts are identity-optional; optional multi-voice mode and own-TTS-ambiguous
  STOP require compatible speaker authority (ADR-0008/0042/0072).
- Sherpa carries immutable acoustic identity through partial, final, barge,
  command, and abort paths; FileReplay does so for deterministic partial,
  final, and abort replay. Typed stale/duplicate ingress is rejected before
  effects; command PCM cannot also become a normal ASR final; typed post-barge
  grants require matching acoustic keys; abort rollback affects only its
  matching partial. An accepted revision now continues through task, production
  TTS, and receipt-backed playback identity (ADR-0084/0086).
- VAD owns live ASR segments and acoustic time. Pre-VAD text cannot publish;
  calibrated speech evidence gates ordinary partials/finals, while unavailable
  bounded handoffs abstain or bypass as specified (ADR-0046/0048).
- Capture recovery rebinds rate/resampling and preserves the first timed block.
  Shutdown epoch-fences effects; same-domain recovery preserves evidence and a
  changed domain relearns outside complete speech epochs (ADR-0043/0048).
- Owner verification is distinct from admission. Only a finite enrolled final
  match can mint owner trust; advisory, mixed, rescue, and generic rewrite paths
  cannot grant device-action authority (ADR-0027/0041/0051).
- The opt-in Linux final pair remains checksum-pinned Parakeet Unified English
  plus Faster-Whisper Small. Exact acoustic quorum may rewrite text; bounded
  double-empty veto and protected controls fail closed. Committed SenseVoice
  defaults remain unchanged (ADR-0078/0080).
- Synchronous capabilities use bounded cancellable coordinators. Mutating
  capabilities additionally receive actor-owned `tool_call_id` and bounded
  session idempotency. Speech and winning task timeouts fence unstarted providers; timeout apologies are best-effort; provider-started
  calls ignore barge but see explicit/shutdown cancellation; stale output is fenced. Tools do not retry
  after failure; failed web gets one fenced local fallback (ADR-0021/0030/0051/0086).
- Terminal receipts govern spoken history with fail-closed identity, exact-owner capacity, and field validation; TTS identity is session-locked (ADR-0028/0029/0038/0086).

## Live evidence and limits

- Exact physical STOP is still red. Runs `192151`/`193713` failed with
  enrollment on and off; v5 was rejected and never promoted. Route/calibration
  settling is a hypothesis, not a proven cause (ADR-0072).
- The 2026-07-16 vault run admitted six unclipped windows with no capture,
  decode, finalizer, or echo-separation failure, yet recognized `vault` 0/6.
  Only post-GTCRN mic audio was retained, so frontend versus recognizer/domain
  failure remains unresolved (ADR-0077).
- Private replay selected Parakeet/Small at WER 0.00 over six development
  clips; it is not a disjoint live-room acceptance set. Public-v2 exact-bound
  tooling is ready, but no v2 model smoke has run (ADR-0078/0080/0085).
- Windows IAudioClient2 communications capture now verifies OS AEC. NS and
  beamforming were measured active on that box. Doctor is READY, but the mic
  fix, new-domain re-enrollment, and owner physical talk-over/bare STOP/silent
  control batch remain required (ADR-0081/0082).
- Diagnostic replay now fails closed on unparseable legacy heartbeats and silent
  playback references. Raw personal voice was purged from controlled Git refs;
  GitHub server-side PR refs remain owner follow-up (ADR-0083).

## Next

- Port the session actor contract and golden lifecycle cases to remote/mobile
  brains; bound the central event mailbox separately. Add a controller route
  for targeted tool-call cancellation before claiming cross-device parity.
- Add timestamped public frame replay and AEC assertions; then build a disjoint
  owner set spanning vault terms, controls/near-controls, numerals, negation,
  noise/echo, bystanders, and multiple voices. Run fresh `./live.sh` A/B before
  claiming stable bare-speaker barge-in, STT, or interaction latency.

## Standard verification

Logic: `...pytest -m "not real_model" -q`; APM: `...pytest tests/test_apm_double_talk.py -q`.
See `docs/agent-testing.md` for full, recorded, model, live, and whitespace gates.

## Operating policy

- Queue: `.agents/backlog.md`; architecture: `docs/unified_architecture.md`; decisions: append-only `docs/adr/`; landing requires a green gate (ADR-0014).
- Never expose secrets, commit raw personal audio, delete logs, or claim unrun live-hardware validation.
