# Agent Map — speaker

## What this repo owns
- Local speaker/voice/audio runtime and engine behavior.
- Headless regression tests for APM, double-talk detection, barge-in, and related audio routing.

## Entry points
| Area | Path | Notes |
|---|---|---|
| Engines | `core/engines/` | Audio engine implementations. |
| Tests | `tests/` | Headless audio and fixture tests. |
| Logs/artifacts | `logs/` | Local generated artifacts; do not commit. |
| Status | `STATUS.md` | Durable project status for agents. |

## Read first
1. `AGENTS.md`
2. `CLAUDE.md` if present
3. `STATUS.md`
4. `docs/agent-testing.md`
5. Relevant engine/test files

## Common task routes
| Task type | Start here | Verify with |
|---|---|---|
| APM/DTD/barge-in bug | `core/engines/`, `tests/test_apm_double_talk.py` | targeted pytest |
| Native capture latency/gaps | `core/media_session.py`, `core/engines/sherpa.py`, ADR-0088 | reader-time playback/source context + control-lane tests, then APM gate |
| Post-ASR mailbox/order | `always_on_agent/event_bus.py`, ADR-0093 | `tests/test_event_bus.py` plus supervisor/playback/preprocessing race tests |
| Session/task/tool/TTS ownership | `always_on_agent/session_actor.py`, `always_on_agent/tasks.py`, `always_on_agent/supervisor.py`, ADR-0086 | `tests/test_session_actor.py` plus focused cancellation/playback tests |
| Accepted continuation → resume lineage | `always_on_agent/supervisor.py`, `core/runtime.py`, `core/resume.py`, ADR-0154 | successful/failed start, non-assistant rejection, stale epoch/generation, query-only tracker, and held-playback merge → barge → `continue` tests; then APM/DTD |
| Trusted LiveKit publisher media | `core/engines/livekit_agents_session.py`, `core/engines/livekit_agents.py`, ADR-0096/0164 | exact four-version requirements/runtime checks plus fake-SDK manual-RoomIO publisher/route/disposal tests first; installed SDK and self-hosted live A/B are separate promotion evidence |
| Remote LiveKit token mint | `remote/token_server.py`, `requirements-remote.txt`, ADR-0164 | fake the reviewed API chain and require exact TTL assignment or no token; installed API and endpoint/service behavior remain separate gates |
| Fixture/test update | `tests/barge_fixtures.py`, matching tests | targeted pytest |
| Recording-driven STT/verifier | `core/config.py` final-STT profiles, `tools/recorded_stt_eval.py`, `tools/local_gpu_stt_eval.py`, `tools/stt_consensus_v2_eval.py`, `core/asr_verifier.py`, `core/engines/_faster_whisper.py` | explicit complete-profile aggregate endpoint-path A/B plus focused resolver/verifier/adapter tests. A one-corpus replay is exploratory and cannot qualify the physical pair (ADR-0078/0080/0144/0158) |
| Synchronized live evidence/private final-input corpus | `core/diagnostic_bundle.py`, `core/engines/sherpa.py`, `tools/prepare_diagnostic_streaming_stt_corpus.py`, `docs/voice_evidence.md`, ADR-0100/0108 | exact-input bundle/export contract tests first; then owner labels, disjoint public comparison, and separate live A/B |
| Paired guided physical final-STT evidence | `core/guided_stt_plan.py`, `core/stt_capture.py`, `tools/guided_stt_capture.py`, `tools/guided_stt_pair_attestor.py`, `docs/voice_evidence.md`, ADR-0157/0158 | run the fixed SenseVoice and Parakeet/Faster-Whisper `./live.sh --guided-stt-capture` entries with unchanged capture setup, retain both bundles, then run the fixed device-free paired attestor and obtain owner review. Ordinary `./live.sh`, `--llm echo`, backend-only `--asr-final`, low-level core capture, and one-bundle replay do not qualify labelled command evidence |
| Timestamped capture-loop replay | `tools/capture_replay/`, `tools/prepare_ami_capture_replay.py`, `tools/capture_replay_eval.py`, ADR-0092 | strict contracts, pinned aggregate diagnostic, then native-reader/AEC, disjoint owner, and live gates |
| EdAcc endpoint-integrity diagnostic | `tools/edacc_endpoint_integrity_eval.py`, `tools/capture_replay_eval.py`, `tools/capture_replay/async_delivery.py`, `core/endpointing.py`, `core/engines/_semantic_hold.py`, `always_on_agent/logical_turn_shadow.py`, ADR-0147/0148/0153/0155/0156 | exact retained corpus/profile/model binding; acoustic, HOLD-only Smart Turn, and opt-in native reset-and-accumulate through the production capture seam; optional raw composition and exact inline terminal lineage; separately flagged post-capture drain through the real final worker with accepted-item/callback identity and capture-side prequeue abort separation; aggregate privacy/no-clobber, source-closure, stage, and replay-v2 lifecycle tests. Simultaneous capture/finalizer concurrency, inline text parity, production stop/shutdown, dispatcher, supervisor, runtime, authority, device, latency, quality, and live evidence remain separate gates |
| Isolated streaming/final-STT benchmark | `tools/streaming_stt_eval.py`, `tools/prepare_public_streaming_stt_corpus.py`, `tools/provision_moonshine_candidate.py`, `tools/prepare_nemotron_runtime.py`, `tools/provision_nemotron_candidate.py`, `tools/provision_faster_whisper_endpoint_candidate.py`, `tools/streaming_stt/`, ADR-0089/0090/0091/0102 | deterministic streaming/final-only plus preparation/provision tests; candidate-specific opt-in exact smoke separately |
| Licensed stratified STT comparison | `tools/prepare_demand_noise_streaming_stt_corpus.py`, `tools/streaming_stt_suite.py`, `tools/public_voice_eval_matrix.py`, ADR-0109 | verify checksum/license/output and aggregate-stratum contracts headlessly; run receipt-bound workers strictly one at a time and keep archives, audio, and reports outside Git |
| Bounded disjoint far-field STT fixture | `tools/prepare_notsofar1_far_field_fixture.py`, `tools/streaming_stt/notsofar1-mtg32006-v1.lock.json`, ADR-0159 | verify the exact five-file NOTSOFAR eval-small source and produce nine isolated windows paired across two devices; overlap, identity, capture, and live behavior remain separate gates |
| PriMock57 two-role overlap diagnostic | `tools/prepare_primock57_conversation_fixture.py`, `tools/primock57_overlap_eval.py`, `tools/streaming_stt/overlap_metrics.py`, `tools/streaming_stt/primock57-overlap-evaluator-v1.lock.json`, ADR-0161/0162 | exact v4 bundle, locked three-case/nine-input private snapshot, one sequential worker, and aggregate stem WER plus the custom two-utterance minimum-order diagnostic; it is neither ORC-WER/tcpWER/tcORC nor endpoint, latency, identity, device, live, or promotion evidence |
| Microsoft AEC APM/DTD component replay | `tools/prepare_microsoft_aec_fixture.py`, `tools/audio_eval/microsoft_aec_32.lock.json`, `tools/audio_eval/metrics.py`, `tools/microsoft_aec_apm_dtd_eval.py`, ADR-0163 | verify the exact 32-case/128-WAV lock, five explicit term acceptances, private fixture/receipt, production APM/runtime closure, oracle-gated detector accounting, aggregate privacy, and terminal report publication headlessly; official materialization requires all five acceptances, production replay additionally requires exact `livekit==1.1.14`, and both remain synthetic component evidence rather than room, natural-turn, endpoint, device, live, or promotion evidence |
| Public short-command/noise gate | `tools/prepare_public_command_noise_corpus.py`, `tools/streaming_stt/public-command-noise-v1.lock.json`, `tools/streaming_stt_eval.py`, `tools/public_voice_eval_matrix.py`, ADR-0117 | verify archive/license/selection/schema-v4/command-metric contracts headlessly; prepare 57 private cases outside Git, then run exact workers sequentially and keep model results separate from capture/endpoint/live evidence |
| Configured final-command FileReplay gate | `tools/production_final_stt_eval.py`, `tools/streaming_stt/final_metrics.py`, `core/engines/file_replay.py`, ADR-0118/0145 | verify exact corpus/config/model/source/runtime/provider binding and aggregate-only decision metrics headlessly; opt-in schema-v3 pair mode resolves both atomic profiles from one base, runs SenseVoice then Parakeet/Faster-Whisper on the same corpus, and makes a streaming-control mismatch explicitly inconclusive; keep after-PCM results separate from owner recordings and live bare-speaker validation |
| Public conversation/STT fixture and qualification | `tools/public_conversation_fixture.py`, `tools/public_conversation_qualification.py`, the four `tools/prepare_*_conversation_fixture.py` materializers, `tools/streaming_stt/public-conversation-96.lock.json`, `tools/public_voice_eval_matrix.py`, ADR-0113/0127/0130/0131/0132 | verify raw-source/lock/receipt/tamper contracts first; EdAcc production requires an absent private extraction root and its first pinned-archive run is inventory validation; prepare four private 24-case corpora outside Git, use qualification v2 for the fixed two-role canonical-strata safe report, retain its printed file digest, and keep model quality plus live A/B as separate gates |
| Fixed EdAcc-only 2x1 STT comparison | `tools/edacc_stt_comparison.py`, `tools/streaming_stt_suite.py`, the retained production EdAcc corpus/receipt, ADR-0143 | verify exact corpus/receipt pins, fixed 24-case 8/8/8 marginal, `production-gigaspeech-zipformer-fp32-benchmark-1t` then `faster-whisper-small-local` with their exact adapters, stream geometry, sequential delegation, pre/post binding, aggregate-only reconstruction, strict retained validation, and no-clobber publication headlessly; run current-checkout receipt-bound workers one at a time and keep the result separate from canonical 2x4, latency, runtime, and live evidence |
| AMI forced-alignment endpoint proxy | `tools/prepare_ami_conversation_fixture.py`, `tools/ami_endpoint_proxy_eval.py`, `tools/streaming_stt/ami_endpoint_proxy.py`, ADR-0129 | verify sidecar/provenance/receipt, ordered labels, isolated-only aggregate reduction, supported adapter semantics, pre/post source checks, and privacy headlessly; model, acoustic-truth, and live validation remain separate |
| Causal LiveKit endpoint diagnostic | `tools/livekit_causal_endpoint_eval.py`, its two isolated workers, `core/endpointing.py`, ADR-0136 | verify shared Sherpa/helper parity, causal label/grid/censoring math, descriptor ownership, aggregate privacy, and failure cleanup headlessly; run the exact private source/model separately and keep owner-recording plus live A/B gates distinct |
| Exact Parakeet NeMo reference | `tools/prepare_parakeet_runtime.py`, `tools/provision_parakeet_realtime_eou_candidate.py`, `tools/streaming_stt/adapters/parakeet_realtime_eou.py`, `tools/streaming_stt_eval.py`, ADR-0099 | deterministic no-model contracts, then an explicit receipt-bound GPU run; benchmark only, never an app entry point |
| Exact parakeet.cpp CPU candidate | `tools/provision_parakeet_cpp_candidate.py`, `tools/streaming_stt/adapters/parakeet_cpp.py`, `tools/streaming_stt/native/parakeet_cpp_bridge.cpp`, `tools/streaming_stt_eval.py`, ADR-0119 | deterministic source/build/model/ELF/bridge/manifest/protocol/worker/supervisor/metric contracts first, then an explicit receipt-bound CPU run; benchmark only, never an app, tool, command, or live entry point |
| Answering-model adoption | `tools/conversation_eval/`, ADR-0051/0067/0068 | deterministic trace, then production-hybrid real-model A/B; label all-role stress explicitly |
| Autonomous voice verdict | `tools/autotest/verdicts.py`, `tools/autotest/README.md` | pure verdict tests, then selected cable/delay/speaker runner; cable is incomplete |
| Silent delay route | `tools/autotest/acoustics.py`, `tools/autotest/audio.py`, `tools/autotest/voice_loop.py`, `core/readiness.py`, ADR-0069/0070 | deterministic contract/topology/stream-binding/command-timing tests, then two fresh exact delay runs; never substitute host EC or physical evidence |
| Speaker enrollment | `tools/prepare_enrollment.py`, `tools/promote_enrollment.py`, `core/enroll.py`, ADR-0056/0066 | tempdir prep/promotion tests first; microphone only after explicit readiness |
| Live audio validation | docs/status + manual instructions | state what was and was not run |
| Docs/status | `STATUS.md`, `README.md` | `git diff --check` |

## Do not load by default
- `logs/**`
- WAV/audio captures
- Generated screenshots/images
- `.env` or credential files

## Known pitfalls
- Headless tests do not prove live microphone/speaker behavior; state manual validation needs.
- A native terminal marker does not itself authorize a reply; only complete-source EOU may do so (ADR-0099).
- For schema-v8 parakeet.cpp, only the first observed EOU can qualify, and only from `feed` after the complete externally bounded source. EOB, finalize EOU, later EOU, and encoder timestamps remain telemetry; the first EOU document freezes visible text (ADR-0119).
- The Parakeet worker's verified scope requests niceness 15; wrapping it with `nice -n 19` prevents scope creation. Use 15 and keep candidate runs sequential (ADR-0109).
- Do not commit local audio artifacts or logs.
