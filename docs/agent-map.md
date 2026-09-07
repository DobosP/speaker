# Agent Map — speaker

## What this repo owns
- Local speaker/voice/audio runtime and engine behavior.
- Headless regression tests for APM, double-talk detection, barge-in, and related audio routing.

## Entry points
| Area | Path | Notes |
|---|---|---|
| Runtime | `core/` | `VoiceRuntime`; `core/engine.py` is the `AudioEngine` seam. |
| Engines | `core/engines/` | `sherpa` production, `scripted` tests, `livekit` remote. |
| Control plane | `always_on_agent/` | The brain; owns the `AgentEvent`/`Mode` contract. |
| Mobile | `mobile/` | Flutter on-device shell. |
| Host + thin client | `remote/`, `web/` | Optional LiveKit/WebRTC path. |
| Tooling | `tools/` | `run_tests.py`, `doctor.py`, `live_launcher.py` (behind `./live.sh`), `setup_models.py`, evaluation tooling. |
| Tests | `tests/` | Headless; no audio hardware, models, or Ollama. |
| Artifacts | `logs/` | Generated; never commit. |
| Truth | `STATUS.md`, `docs/adr/` | Current state; decisions (append-only). |

## Common task routes
| Task | Start here | Verify with |
|---|---|---|
| Audio/APM/DTD/echo/interrupt | `core/engines/`, `tools/echo_probe.py`, `tools/interrupt_suite.py` | `tests/test_echo_probe.py test_interrupt_suite.py test_sherpa_playback.py test_apm.py test_apm_double_talk.py test_denoise.py`; physical acceptance only `./live.sh` + `python -m tools.live_audio_ab logs/runs/run-<id>.txt`; receipts: ADR-0174/0175/0176/0177/0181 |
| KWS / speaker authority / duck-confirm | `core/kws_contract.py`, `core/engines/sherpa.py`, `core/engines/_kws_speaker_inference_owner.py`, `core/engines/speaker_gate.py` | `tests/test_kws_contract.py test_kws_speaker_inference_owner.py test_virtual_audio_engine.py test_barge_word_cut.py test_barge_confirm.py test_sherpa_media_session.py test_speaker_gate.py`; receipts: ADR-0171/0172/0182/0183/0184/0185; owner bare-speaker A/B pending |
| Model setup, native capture, desktop GPU live turns | `tools/setup_models.py`, `tools/setup_minicpm.py`, `core/engines/_sherpa_models.py`, `core/media_session.py`, `tools/live_launcher.py` | setup/doctor/TTS tests + the APM gate; receipts: ADR-0088/0165/0167/0209; the next 4090 start fails closed until enrollment is repeated on the active route |
| Post-ASR mailbox / session ownership | `always_on_agent/event_bus.py`, `session_actor.py`, `tasks.py`, `supervisor.py` | `tests/test_event_bus.py test_session_actor.py`; receipts: ADR-0086/0093/0094 |
| Web-search / retained-context egress | `always_on_agent/capabilities.py`, `react.py`, `models.py`, `core/websearch.py`, `core/sensitivity.py`, `core/capabilities.py` | `tests/test_capability_context_isolation.py test_websearch.py test_react_planner.py test_llm_egress_policy.py test_cloud_pii_egress.py test_capability_exception_sanitization.py`; receipts: ADR-0003/0060/0187/0189/0191/0192 |
| Hedge lifecycle | `core/_hedge_source_owner.py`, `core/llm.py`, `core/llm_factory.py` | `tests/test_hedge_source_owner.py test_hedge_chain*.py test_multi_provider_llm.py`, `python tools/run_tests.py cloud`; receipts: ADR-0021/0030/0190 |
| Mobile Flutter owners | `mobile/lib/agent_session.dart`, `assistant*.dart`, `llm_generation_owner.dart`, `asr_isolate.dart`, `tts_*owner.dart` | `cd mobile && flutter analyze && flutter test`, `tests/test_golden_contract.py`, `.github/workflows/mobile-tests.yml`; receipts: ADR-0186/0201/0202/0203/0204; no plugin, device, or live claim |
| Mobile ASR evidence | `tools/provision_mobile_zipformer.py`, `provision_mobile_whisper.py`, `mobile_asr_evidence_eval.py`, `mobile_asr_two_pass_eval.py`, `prepare_mobile_asr_evidence_packet.py`, `tools/streaming_stt/mobile-*.lock.json` | the matching `tests/test_*.py`; receipts: ADR-0205/0206/0207/0208; offline desktop-CPU evidence only |
| Continuation/resume + cleaner | `always_on_agent/supervisor.py`, `core/runtime.py`, `core/resume.py`, `core/cleanup.py` | `tests/test_cleanup.py` plus the resume tests; receipts: ADR-0154/0173 |
| LiveKit trusted-LAN + token | `core/engines/livekit_agents*.py`, `remote/token_server.py` | `tests/test_livekit_agents_engine.py test_token_server.py`; receipts: ADR-0096/0097/0164 |
| Final-STT profiles, recorded/guided evidence | `core/config.py`, `core/asr_verifier.py`, `core/engines/_faster_whisper.py`, `core/guided_stt_plan.py`, `core/stt_capture.py`, `tools/recorded_stt_eval.py`, `tools/guided_stt_capture.py`, `tools/guided_stt_pair_attestor.py` | `docs/voice_evidence.md`; receipts: ADR-0078/0080/0144/0157/0158/0188 |
| Live evidence bundle | `core/diagnostic_bundle.py`, `tools/prepare_diagnostic_streaming_stt_corpus.py` | `docs/voice_evidence.md`; receipts: ADR-0100/0108 |
| Capture-loop replay / EdAcc endpoint integrity | `tools/capture_replay/`, `tools/capture_replay_eval.py`, `tools/edacc_endpoint_integrity_eval.py`, `core/endpointing.py`, `core/engines/_semantic_hold.py` | receipts: ADR-0092/0147/0148/0149/0150/0153/0155/0156 |
| Public corpora, STT candidates, benchmarks | `tools/streaming_stt/`, `tools/streaming_stt_eval.py`, `tools/public_voice_eval_matrix.py`, `tools/prepare_*_fixture.py`, `tools/provision_*_candidate.py` | `docs/public_voice_evaluation_matrix.md`, `docs/public_voice_regression.md`, `docs/evaluation_runbooks.md`; receipts: ADR-0089/0090/0091/0099/0102/0109/0113/0117/0118/0119/0127–0136/0143/0145/0159–0163/0166/0194/0196; benchmark-only, never app entry points |
| Semantic interruption / Anyreach | `always_on_agent/interruption_policy.py`, `tools/semantic_interruption/` | `tests/test_interruption_policy.py`; receipts: ADR-0168/0169/0170; inert |
| Answering-model adoption | `tools/conversation_eval/` | receipts: ADR-0051/0067/0068 |
| Autonomous voice / silent delay | `tools/autotest/`, `core/readiness.py` | `tools/autotest/README.md`; receipts: ADR-0055/0058/0069/0070 |
| Speaker enrollment | `tools/prepare_enrollment.py`, `tools/promote_enrollment.py`, `core/enroll.py` | receipts: ADR-0056/0066 |
| ASR biasing / hotwords | `docs/asr_biasing.md`, `core/engines/sherpa.py` | receipts: ADR-0114 |
| Memory | `utils/memory*.py`, `MEMORY.md` | `python tools/run_tests.py memory` |
| Live validation | `./live.sh`, `docs/voice_evidence.md` | state exactly what was and was not run |
| Fixture, docs, or status update | `tests/barge_fixtures.py`, `STATUS.md`, `WORKLOG.md`, `docs/adr/` | targeted pytest; `git diff --check`; `python3 ~/work/agent-ops/scripts/check_docs.py .` |

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
- `--engine` is the legacy alias of `--session` (`core/app.py:1159-1164`); write `--session`.
- Never hard-set `aec_ref_delay_ms` to 260 ms (ADR-0005).
