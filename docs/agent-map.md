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
| Trusted LiveKit publisher media | `core/engines/livekit_agents_session.py`, `core/engines/livekit_agents.py`, ADR-0096/0104 | fake-SDK manual-RoomIO publisher/route/disposal tests first; pinned SDK and self-hosted live A/B are separate promotion evidence |
| Fixture/test update | `tests/barge_fixtures.py`, matching tests | targeted pytest |
| Recording-driven STT/verifier | `tools/recorded_stt_eval.py`, `tools/local_gpu_stt_eval.py`, `tools/stt_consensus_v2_eval.py`, `core/asr_verifier.py`, `core/engines/_faster_whisper.py` | aggregate endpoint-path A/B, focused verifier/adapter tests, then manual live A/B (ADR-0078/0080) |
| Synchronized live evidence/private final-input corpus | `core/diagnostic_bundle.py`, `core/engines/sherpa.py`, `tools/prepare_diagnostic_streaming_stt_corpus.py`, `docs/voice_evidence.md`, ADR-0100/0108 | exact-input bundle/export contract tests first; then owner labels, disjoint public comparison, and separate live A/B |
| Timestamped capture-loop replay | `tools/capture_replay/`, `tools/prepare_ami_capture_replay.py`, `tools/capture_replay_eval.py`, ADR-0092 | strict contracts, pinned aggregate diagnostic, then native-reader/AEC, disjoint owner, and live gates |
| Isolated streaming/final-STT benchmark | `tools/streaming_stt_eval.py`, `tools/prepare_public_streaming_stt_corpus.py`, `tools/provision_moonshine_candidate.py`, `tools/prepare_nemotron_runtime.py`, `tools/provision_nemotron_candidate.py`, `tools/provision_faster_whisper_endpoint_candidate.py`, `tools/streaming_stt/`, ADR-0089/0090/0091/0102 | deterministic streaming/final-only plus preparation/provision tests; candidate-specific opt-in exact smoke separately |
| Exact Parakeet NeMo reference | `tools/prepare_parakeet_runtime.py`, `tools/provision_parakeet_realtime_eou_candidate.py`, `tools/streaming_stt/adapters/parakeet_realtime_eou.py`, `tools/streaming_stt_eval.py`, ADR-0099 | deterministic no-model contracts, then an explicit receipt-bound GPU run; benchmark only, never an app entry point |
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
- Do not commit local audio artifacts or logs.
