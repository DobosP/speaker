# Status — speaker

Single source of truth: this file > newest accepted ADR > everything else; dated handoffs are history.

Last verified: 2026-08-05 on Linux ROG.

- Last broad non-real baseline at `dfb5163`: 7,897 low-priority host-side passes plus the bounded-read race isolated; 14 skipped, 24 model-only deselected, 11 warnings. Current receipt-v2 gates: exact-private 362 + 1 skipped; adjacent consumers 448 + 1 skipped plus the race isolated; dry tool-route 1,287; safe canonical conversation 214, public-fixture lock 214, and full public-conversation 252; phase-resource 210; AMI proxy focused 206; broad streaming 1,015 + 1 skipped + 5 deselected; APM/DTD 6. Earlier repeat/memory 105, adjacent 304 + 3 skipped, conversation 213, candidate/shared 464 + 1 isolated, and public command/noise 246 remain green.
- Earlier landed component baselines: streaming family 653 passed, 3 model-only deselected; structured lifecycle 270; capture-replay 71. Moonshine Tiny/Small/Medium, Nemotron, and Parakeet NeMo stay rejected (ADR-0090/0091/0099/0115).
- Post-ASR mailbox/adjacent 221; LiveKit Agents seam/manual-session 73, adjacent baseline 224; unified-session 55; playback/actor/runtime 150; public-v3 81 file/145 combined.
- Stable evidence: both conversation pairs 42/42, semantic-memory PASS, owner replay 9/9, two synthetic-delay passes (ADR-0051/0065/0067/0068/0070/0080).

## Runtime

- `python -m core --session` is the one public core entry; `VoiceSession` owns one injected `VoiceRuntime`, and `build_runtime` remains the sole tool/authority plane. Repeat-previous admits built-in memory items only after a process-local ordinal fence; foreign backends require a finite, strictly newer wall time (ADR-0123).
- Audio defaults to device-only. The exact-pair manual-RoomIO trusted-LAN wrapper passes fake-SDK isolation but stays unselectable until live A/B; legacy remote is rollback only (ADR-0096/0097/0104).
- `./live.sh` is the single Linux physical entry. It owns the host lock, reversible echo-control route, conditional Ollama, doctor, and aligned private pre-DSP/processed-mic/playback-reference evidence (ADR-0075/0077).
- Desktop MiniCPM Q8 is the local text tier; Gemma3 is complex/vision. Phone Q4 uses native XML tools; phone thermal behavior is unvalidated (ADR-0020/0033/0062).
- Setup may enable bounded PRIVATE vault search, reminders, and exact trusted apps. Mutations require unchanged direct speech plus confirmation and stay outside planners (ADR-0073/0074/0076).

## Voice reliability now implemented

- Open-speaker barge-in must work without enrollment. Generic four-novel-word cuts are identity-optional; optional multi-voice mode and own-TTS-ambiguous STOP require compatible speaker authority (ADR-0008/0042/0072).
- Sherpa native reads use a 300 ms/eight-frame capture-only MediaSession. Overload retires stale PCM/lineage; atomic reader context and one rational playback clock prevent time/reference drift. Priority recovery preserves fatal state; same-domain gaps preserve learned evidence, recreate KWS, atomically claim confirmation, and guard first-block effects (ADR-0088).
- Async endpoint finalization crosses a route-neutral, process-local bounded media stage with explicit capture epoch/generation, immutable owned PCM, and per-item cancellation. Overload never decodes inline; shutdown releases queued work without depending on diagnostics, and callback-owned stop defers retained audio cleanup until its capture-effect lease unwinds. Identities, entry points, tools, and model defaults stay unchanged (ADR-0107).
- The default bounded Sherpa path now runs its complete DSP/VAD/primary-ASR/playback-confirm/word-cut/endpoint/capture-loop callback state machine on one dedicated exact-thread decode owner behind the existing capture mailbox. Immutable PCM/reference snapshots, cancelable effect leases, whole-scope loss rotation, whole-role native-error recovery, owner-before-reader readiness, rollback-safe tail workers, and deferred re-entrant cleanup are headlessly covered. The inline zero-queue path remains compatibility only (ADR-0110/0111).
- Capture replay uses one synchronous decode session and binds the dedicated-owner source in its digest, but explicitly rejects and does not execute the production owner thread. It is not native-reader, mailbox-overload, effect-lease, device, WER, latency, or live evidence (ADR-0111).
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
- The opt-in Linux final pair remains checksum-pinned Parakeet Unified English plus Faster-Whisper Small. Protected controls fail closed; SenseVoice defaults remain unchanged (ADR-0078/0080).
- Streaming hotwords now require explicit model context. Isolated opt-in setup pins and atomically selects a complete, self-contained English Zipformer ASR/BPE family plus digest/case policy but adds no phrases or capabilities; active local/replay/LiveKit paths fail closed and replay binds the consumed vocabulary bytes (ADR-0114).
- Public matrix v5 has 8 tracks/19 sources/11 exclusions. One self-digested, transcript/path/identity-free lock binds four separate private schema-v2 corpora at exactly 24 cases each: HarperValleyBank, EdAcc test, paired AMI ES2004a close/far, and a 6x4 Common Voice SPS v4 projection. Source-specific materializers recompute checkout/archive/audio evidence; the CV parent receipt/corpus and every generic receipt/reference/PCM are cross-bound. The two-role canonical-strata entry now reconstructs a schema-v2 fixed aggregate-only report, binds exact adapters/resolved streams and source/safe digest domains, rejects public runner injection, and exposes strict retained-report validation plus the exact published-byte digest; green remains coverage-only/non-promotional. The AMI materializer additionally publishes a private sidecar-bound, aggregate-only forced-aligned last-word endpoint proxy for its 16 isolated windows and only the Parakeet Realtime EOU/parakeet.cpp adapters; it is not acoustic ground truth and defines no threshold. Raw-source injections stay non-production. The retained exact AMI output (3,802,880 PCM bytes) needs fresh no-overwrite rematerialization for the sidecar; exact Harper validates at 24 cases/2,956,800 bytes, while EdAcc, Common Voice, and the four-source model run remain open (ADR-0098/0101/0109/0113/0120/0122/0127/0129/0130).
- The exact public command/noise lock and no-download preparer bind 57 private schema-v4 cases from official Speech Commands v0.02 and ECCC v1.2. Typed targets distinguish recall, precision, false positives, speech negatives, and silence without committing private rows or paths. An aggregate-only ASR-only FileReplay route now measures the configured final-selection code with exact corpus/config/model/source/runtime/provider and atomic-report binding, but without live VAD-owned timing or recovery authorization (ADR-0117/0118).
- The isolated harness now has a receipt-bound CPU-only parakeet.cpp v0.5.0 candidate: schema v8 binds its closed source/build/model/ELF set and Speaker bridge, protocol v4 preserves typed EOU/EOB observations and first-EOU text/terminal policy, and Bubblewrap plus a verified one-CPU, 2/3-GiB, zero-swap scope exposes no network or NVIDIA nodes. Applicable scoped workers additionally expose ordered startup, first-use, and resident whole-scope cgroup memory/CPU observations; cache state is uncontrolled, and these are neither cold/warm nor acceptance evidence. It has no runtime, tool, command, or adoption authority (ADR-0119/0128).
- The opt-in conversation flow-v1 gate composes unchanged v4 semantics with exact 4x3 deterministic journeys for typed incremental turns, scripted interruption recovery, delayed read tools/follow-up, and confirmed synthetic-action stop/restart fencing. It uses production runtime/session ownership but explicitly excludes microphone/VAD wiring, STT/WER, real TTS/audio, AEC/room, physical audibility, external effects, and live latency (ADR-0112).
- Capabilities use actor-issued per-task `TurnHandle`s and five-field bindings.
  Task/provider/tool/playback ownership registers before start; cancel fences
  new children and bounded drain reports providers still exiting (ADR-0094).
- Same-ID reuse cannot consume stale terminal/TTS/playback cleanup. Tools do not
  retry; failed web gets one local fallback (ADR-0021/0030/0051/0086/0094).
- Terminal receipts govern spoken history with fail-closed identity and exact
  ownership. Diagnostic schema v2 binds four continuous private PCM16 tracks,
  a separate lossless f32le final-input spool, endpoint replay, and causal
  playback; private labels export corpus schema v3 with mandatory receipt v2, which cross-binds the raw-label digest and a domain-separated exact ordered case-surface digest before aggregate final-selector/tool-route replay and close-time rechecks, while public corpora stay v2 (ADR-0028/0029/0038/0086/0100/0108/0124/0126).

## Live evidence and limits

- Exact physical STOP is red: `192151`/`193713` failed with enrollment on/off; v5 was rejected and route settling is unproven (ADR-0072).
- The 2026-07-16 vault run admitted six unclipped windows without capture,
  decode, finalizer, or echo-separation failure, yet recognized `vault` 0/6.
  Only post-GTCRN mic audio was retained, so the failure seam is unknown (ADR-0077).
- Private replay WER 0.00 is non-disjoint. The retained aggregate-only BPE candidate tied all six clips (streaming .20, selected 0.00) but had zero keyword attempts and no separate command/exit receipt, so it is not promotable or live evidence. Public-v3 trusts local PyArrow; its code-bound Small
  control WER 0.6685 stays rejected. These are development—not streaming, held-out, live, or adoption—results (ADR-0087).
- Common Voice SPS v4 real-archive/model compatibility is unrun. VoxPopuli's first real preparation failed closed on its superseded PCM16 assumption; aggregate inspection found the exact float32 container and a feasible 24-case selection, but no real corpus/model run exists and the slice is not command, domestic, capture/AEC, tool, live, or training-disjoint evidence (ADR-0101/0106).
- Strict sequential clean/noisy WER is .6685/.6757 for final-only Faster-Whisper Small, .6848/.6830 for Turbo, and .7283/.9076 for CPU Zipformer. Small wins this 14-source development slice but has no endpoint, live, or adoption validity (ADR-0102/0109).
- Parakeet clean/noisy WER is .6576/.5429; paced clean first/stable partial p50 is 1.692/5.612 s with two misses/120 ms backlog. Native EOU ended 85/126 noisy evaluations early and one source disagreed across repeats, so it remains rejected after-PCM evidence (ADR-0099/0109).
- Exact Moonshine 0.1.0 is benchmark-only. Medium's matched legacy burst WER/CER/RTF is .6685/.6342/.2971. Its stock external-presegmented threshold-zero profile is now reproducible but rejected: WER/CER/RTF .8750/.8373/.9950, 10/14 nonempty finals, finalization p50/p95 .608/4.907 s, seven missing stable partials, and 1,635 MiB reported RSS. A stock internal/external causal pair is invalid because internal VAD leaks recurrent state and can drop segment tails. No command, Speaker endpoint, or live evidence exists (ADR-0090/0115/0116).
- On the isolated 57-case command/noise gate, Faster-Whisper Small reached 18/20 clean commands, 1/6 noisy ECCC commands, and 11/21 ECCC false activations; one-second-context Zipformer reached 16/20, 1/6, and 0/21. These remain distinct development comparators, not endpoint/latency evidence (ADR-0117).
- Same-run raw/FileReplay-selected views moved overall recall 17/26 to 19/26 and clean recall 16/20 to 18/20, but noisy ECCC recall stayed 1/6 while ECCC false activations rose 1/21 to 5/21. Parakeet Realtime EOU reached 21/26 overall and 19/20 clean, but only 2/6 ECCC with 8/21 false activations. Both are rejected after-PCM results; no live validation or default change occurred (ADR-0118).
- On the same 57-case command/noise corpus, the selected parakeet.cpp tail-8,000 cell completed 171 burst evaluations over three repeats with zero final disagreement: WER/CER .5745/.5424, 63/141 exact matches, .8077 command recall, .2581 negative final-case FPR, 24 accepted first-EOU endpoints, 147 tail exhaustions, .4826 source-audio RTF, .3315 model-input RTF, and 306.027 MiB maximum reported RSS. Matched paced replay preserved exact accuracy with .5592/.3841 RTF, first-partial p50/p95 772.961/1,104.004 ms, six deadline misses, 30.43 ms maximum backlog, and 306.387 MiB maximum reported RSS. This remains after-PCM development evidence, not capture/VAD/AEC/device/live/tool/default authority (ADR-0119).
- On the locked AMI source alone, the same candidate completed 72 burst evaluations with zero disagreement and WER/CER .4684/.4113, but close/far WER split .2278/.7089 and only 3/12 far cases produced nonempty finals per repeat. Paced accuracy matched exactly; source/model-input RTF were .4367/.3677, first-partial p50/p95 622.543/2,369.120 ms, ten stable partials were missing, two deadlines missed, and maximum reported RSS was 308.105 MiB. NVIDIA discloses AMI in training/evaluation without exact membership, so disjointness is unproved; this single-source direct run is not four-source fixture, live, or promotion evidence (ADR-0121).
- On the locked Harper caller source alone, the same candidate completed 72 burst evaluations with zero disagreement and WER/CER .1562/.0677; paced accuracy matched exactly at 15/24 exact and 23/24 nonempty finals. Sixteen cases accepted a post-source EOU and eight exhausted the 500 ms tail; paced source/model-input RTF were .4736/.3886, first-partial p50/p95 615.498/788.595 ms, three stable partials were missing, four deadlines missed, and maximum reported RSS was 304.012 MiB. Harper is absent from NVIDIA's disclosed dataset list, which does not prove disjointness; this clean single-channel direct run is not four-source, live, or promotion evidence (ADR-0122).
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
- The opt-in exact-input tool-route gate uses receipt-bound schema-v3 tags and one selected terminal final to score the production deterministic analyzer/planner/device matcher without invocation; its explicit inert profile is digest-bound and reports aggregates only. No owner-labelled run, provider/tool execution, device, model, or live authority exists (ADR-0108/0124/0125/0126).

## Next

- Acquire and validate the remaining EdAcc and Common Voice private conversation/STT corpora, then run the exact canonical-strata baseline/candidate wrapper and retain its v2 safe report plus printed digest; AMI and Harper source preparation are complete. Use the public command/noise gate for hard-noise work; require a patched reset/tail-safe Moonshine build before any causal pair. Never change runtime defaults from development evidence (ADR-0112/0113/0115/0116/0117/0120/0122/0127/0130).
- Capture fresh exact target and negative recordings for the configured VAULT/OBSIDIAN/name phrases plus `./live.sh` open-speaker evidence, label before export, then run final-selector/tool-route and live BPE A/B; add disjoint command/multi-voice strata, native-reader/gap, and AEC replay (ADR-0092/0100/0108/0109/0114/0125/0126).
- Keep parakeet.cpp isolated; prioritize Common Voice SPS v4 after owner term/token acceptance, then rematerialize and run the AMI forced-alignment proxy, obtain acoustically labelled endpoint truth, add multi-host build receipts, and perform live capture validation. Do not change defaults until far-field, command safety, ownership, device, and live gates pass. Continue publisher/mobile/remote work separately (ADR-0099/0118/0119/0121/0122/0129).

## Standard verification

Logic: `...pytest -m "not real_model" -q`; APM: `...pytest tests/test_apm_double_talk.py -q`; see `docs/agent-testing.md`.

## Operating policy

- Queue: `.agents/backlog.md`; architecture: `docs/unified_architecture.md`; decisions: append-only `docs/adr/`; land only green (ADR-0014). Never expose secrets, commit raw personal audio, delete logs, or claim unrun live validation.
