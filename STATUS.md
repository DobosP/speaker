# Status — speaker

Single source of truth: this file > newest accepted ADR > everything else; dated handoffs are history.

Last verified: 2026-07-31 on Linux ROG.

- Repository logic gate (`-m "not real_model"`): 6016 passed, 13 skipped,
  20 deselected, 9 pre-existing warnings. Focused gates: typed-ingress/Sherpa/APM
  354 passed, 2 skipped; session ownership 321; public-v3 81 file/145 combined.
- Bounded capture review: 201 passed, 1 skipped; adjacent 305; APM/DTD 6.
- Streaming harness: 83 focused; repaired combined/import/APM gate 273;
  independent startup/import repair gate 8.
- Previous stable evidence remains: production-hybrid MiniCPM/Gemma and
  Gemma/Gemma 42/42, semantic-memory PASS, strict recorded owner replay 9/9,
  and two clean synthetic-delay runs (ADR-0051/0065/0067/0068/0070/0080).

## Runtime

- The Linux Python runtime is the reference. Its post-recognition `SessionActor`
  owns bounded admission, binding release, identity, and teardown; phone/remote
  shells lack it. Only invoked post-ASR data may reach cloud (ADR-0001/0086).
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
- Sherpa native reads use a 300 ms/eight-frame capture-only MediaSession.
  Overload retires stale PCM/lineage; atomic reader context and one rational
  playback clock prevent time/reference drift. Priority recovery preserves fatal
  state; same-domain gaps preserve learned evidence, recreate KWS, atomically
  claim confirmation, and guard first-block effects (ADR-0088).
- Sherpa carries immutable acoustic identity through partial, final, barge,
  command, and abort paths; FileReplay does so for deterministic replay. Typed
  stale/duplicate ingress is rejected before effects; command PCM cannot also
  finalize; accepted revisions continue through task, TTS, and playback (ADR-0084/0086).
- VAD owns live ASR segments and acoustic time. Pre-VAD text cannot publish;
  calibrated speech evidence gates ordinary partials/finals, while unavailable
  bounded handoffs abstain or bypass as specified (ADR-0046/0048).
- Capture recovery rebinds rate/resampling and preserves the first timed block.
  Same-domain recovery preserves evidence; changed domains relearn outside
  complete speech epochs (ADR-0043/0048).
- Owner verification is distinct from admission. Only a finite enrolled final
  match can mint owner trust; advisory, mixed, rescue, and generic rewrite paths
  cannot grant device-action authority (ADR-0027/0041/0051).
- The opt-in Linux final pair remains checksum-pinned Parakeet Unified English
  plus Faster-Whisper Small. Exact acoustic quorum may rewrite text; protected
  controls fail closed. SenseVoice defaults remain unchanged (ADR-0078/0080).
- Capabilities use bounded cancellable coordinators and actor-owned identities.
  Timeouts fence unstarted providers; started mutations see explicit/shutdown
  cancellation. Tools do not retry; failed web gets one local fallback (ADR-0021/0030/0051/0086).
- Terminal receipts govern spoken history with fail-closed identity, exact-owner
  capacity, and validated fields; TTS identity is session-locked (ADR-0028/0029/0038/0086).

## Live evidence and limits

- Exact physical STOP is still red. Runs `192151`/`193713` failed with
  enrollment on and off; v5 was rejected. Route settling is unproven (ADR-0072).
- The 2026-07-16 vault run admitted six unclipped windows without capture,
  decode, finalizer, or echo-separation failure, yet recognized `vault` 0/6.
  Only post-GTCRN mic audio was retained, so the failure seam is unknown (ADR-0077).
- Private replay WER 0.00 is non-disjoint. Public-v3 trusts its local PyArrow
  environment; its code-bound Small control WER 0.6685 remains rejected. These
  are development—not streaming, held-out, live, or adoption—results (ADR-0087).
- The fake-only streaming harness FD-launches a private hash-bound bundle and
  emits aggregate metrics. It has no network/RAM/VRAM enforcement or `setsid`
  escape fence; no real candidate or default changed (ADR-0089).
- Windows communications capture verifies OS AEC; NS and beamforming were
  measured active, but owner physical talk-over/STOP tests remain (ADR-0081/0082).
- Diagnostic replay fails closed on malformed heartbeats and silent playback
  references. GitHub server-side raw-audio PR refs remain owner follow-up (ADR-0083).

## Next

- Port session ownership to remote/mobile; bound the central mailbox and add
  targeted tool cancellation before claiming parity.
- Provision exact disposable streaming candidates and compare them on public-v3.
  Then add timestamped frame/AEC replay and a disjoint owner set spanning vault
  terms, controls, numerals, negation, noise/echo, bystanders, and multiple voices.
  Run fresh `./live.sh` A/B before claiming stable results (ADR-0089).

## Standard verification

Logic: `...pytest -m "not real_model" -q`; APM: `...pytest tests/test_apm_double_talk.py -q`.
See `docs/agent-testing.md` for full, recorded, model, live, and whitespace gates.

## Operating policy

- Queue: `.agents/backlog.md`; architecture: `docs/unified_architecture.md`; decisions: append-only `docs/adr/`; land only green (ADR-0014). Never expose secrets, commit raw personal audio, delete logs, or claim unrun live validation.
