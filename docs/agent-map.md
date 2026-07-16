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
| Fixture/test update | `tests/barge_fixtures.py`, matching tests | targeted pytest |
| Recording-driven STT/verifier | `tools/recorded_stt_eval.py`, `tools/local_gpu_stt_eval.py`, `core/asr_verifier.py`, `core/engines/_faster_whisper.py` | aggregate recording A/B, focused verifier/adapter tests, then manual live A/B (ADR-0078/0079) |
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
- Do not commit local audio artifacts or logs.
