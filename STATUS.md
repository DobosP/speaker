# Status — speaker

Single source of truth: this file > newest accepted ADR > everything else; dated handoffs are history.

Last verified: 2026-08-01 on Linux ROG.

- Repository logic baseline before ADR-0104 source-audit fixes: 6951 passed, 13 skipped, 23 model-only deselected, 9 pre-existing warnings; synchronized-diagnostic focused 253/3 skipped; current APM/DTD 6.
- Streaming family: 653 passed, 3 model-only deselected; structured lifecycle 270; capture-replay 71; Moonshine/Nemotron/Parakeet NeMo stay rejected (ADR-0090/0091/0099).
- Post-ASR mailbox/adjacent 221; LiveKit Agents seam/manual-session 73, adjacent baseline 224; unified-session 55; playback/actor/runtime 150; public-v3 81 file/145 combined.
- Stable evidence: both conversation pairs 42/42, semantic-memory PASS, owner replay 9/9, two synthetic-delay passes (ADR-0051/0065/0067/0068/0070/0080).

## Runtime

- `python -m core --session` is the one public core entry; `VoiceSession` owns one injected `VoiceRuntime`, and `build_runtime` remains the sole tool/authority plane.
- Audio defaults to device-only. The exact-pair manual-RoomIO trusted-LAN wrapper
  passes fake-SDK isolation but stays unselectable until live A/B; legacy remote is rollback only (ADR-0096/0097/0104).
- `./live.sh` is the single Linux physical entry. It owns the host lock,
  reversible echo-control route, conditional Ollama, doctor, and aligned
  private pre-DSP/processed-mic/playback-reference evidence (ADR-0075/0077).
- Desktop MiniCPM Q8 is the local text tier; Gemma3 is complex/vision. Phone Q4 uses native XML tools; phone thermal behavior is unvalidated (ADR-0020/0033/0062).
- Setup may enable bounded PRIVATE vault search, reminders, and exact trusted apps.
  Mutations require unchanged direct speech plus confirmation and stay outside planners (ADR-0073/0074/0076).

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
- Public evaluation has 8 tracks/13 sources/9 exclusions. Common Voice SPS v4 has a twice-hashed private 32-speaker P0 preparer; VoxPopuli has a synthetic-green explicit-local 24-speaker `en_ro` preparer; Zipformer is a one-thread control, Parakeet EOU remains rejected, and receipt-bound Faster-Whisper is final-only (ADR-0098/0099/0101/0102/0103).
- Capabilities use actor-issued per-task `TurnHandle`s and five-field bindings.
  Task/provider/tool/playback ownership registers before start; cancel fences
  new children and bounded drain reports providers still exiting (ADR-0094).
- Same-ID reuse cannot consume stale terminal/TTS/playback cleanup. Tools do not
  retry; failed web gets one local fallback (ADR-0021/0030/0051/0086/0094).
- Terminal receipts govern spoken history with fail-closed identity and exact
  ownership. Opt-in recording atomically binds four private tracks, exact endpoint
  replay, and causal playback; loss is incomplete and proves no audibility/live quality (ADR-0028/0029/0038/0086/0100).

## Live evidence and limits

- Exact physical STOP is red: `192151`/`193713` failed with enrollment on/off; v5 was rejected and route settling is unproven (ADR-0072).
- The 2026-07-16 vault run admitted six unclipped windows without capture,
  decode, finalizer, or echo-separation failure, yet recognized `vault` 0/6.
  Only post-GTCRN mic audio was retained, so the failure seam is unknown (ADR-0077).
- Private replay WER 0.00 is non-disjoint. Public-v3 trusts local PyArrow; its code-bound Small
  control WER 0.6685 stays rejected. These are development—not streaming, held-out, live, or adoption—results (ADR-0087).
- Common Voice SPS v4 real-archive/model compatibility is unrun. VoxPopuli real-Parquet preparation is also unrun; its formal L2-accent slice is not command, domestic, capture/AEC, tool, live, or training-disjoint evidence (ADR-0101/0103).
- The schema-v6 Faster-Whisper endpoint comparator enforces isolated receipt roots, one exact stdlib/uv Python 3.12.3 marker entry, and final-only worker/supervisor boundaries; fake coverage only—no candidate, corpus, GPU, WER, latency, streaming, or runtime/default claim has run (ADR-0102/0105).
- Exact Parakeet burst/paced WER is .6576/.6576 and model-input RTF .1910/.2200;
  paced replay missed three deadlines with 189 ms backlog. Ten of 14 EOU events
  were source-early, so the 6.43 GB NeMo runtime is rejected after-PCM evidence only (ADR-0099).
- Exact Moonshine 0.1.0 is benchmark-only. Small WER is .6884 burst/.6957 real-time;
  at 200 ms: RTF 1.1864, first partial p50 2.02 s, 1,000 misses, 22.01 s backlog. Tiny WER is .8478 (ADR-0090).
- Nemotron burst WER/CER/RTF: LA0 .7065/.6606/.4331; LA3 .7065/.6592/.1395;
  LA6 .6957/.6551/.0859; LA13 .6902/.6467/.0556. LA6 real-time was
  .6957 WER/.0992 RTF, first-partial p50 1.666 s, one miss, 10.439 ms backlog.
  Every mode is rejected; this excludes capture/VAD/endpoint/AEC/live validity
  and leaves `.venv`, defaults, and `./live.sh` unchanged (ADR-0091).
- Corrected mic-only AMI diagnostic has poor recognition: overall-including-
  linearized-overlap/single/linearized-overlap/transition WER .852/.632/1.000/.933.
  Silence stayed empty; six verifier decodes had no errors. First-partial/final
  p50 .184/1.113 s; peak RSS 2,033 MB. Not native-reader/AEC/live evidence (ADR-0092).
- Windows communications capture verifies OS AEC/NS/beamforming; owner physical talk-over/STOP tests remain (ADR-0081/0082).
- Diagnostic replay fails closed on malformed heartbeats/silent playback references; GitHub raw-audio PR refs remain owner follow-up (ADR-0083).
- The LiveKit Agents wrapper has no SDK, network, audio-device, or live A/B evidence; its headless fake-SDK result cannot promote trusted-LAN (ADR-0104).

## Next

- Run fresh `./live.sh` vault/open-speaker A/B with the synchronized bundle; add native-reader/gap, AEC replay, and disjoint owner/noise/multi-voice data (ADR-0092/0100).
- Keep `parakeet.cpp`, Common Voice/Faster-Whisper model runs, and publisher/mobile/remote work isolated until their capture, ownership, device, and live gates pass.

## Standard verification

Logic: `...pytest -m "not real_model" -q`; APM: `...pytest tests/test_apm_double_talk.py -q`; see `docs/agent-testing.md`.

## Operating policy

- Queue: `.agents/backlog.md`; architecture: `docs/unified_architecture.md`; decisions: append-only `docs/adr/`; land only green (ADR-0014). Never expose secrets, commit raw personal audio, delete logs, or claim unrun live validation.
