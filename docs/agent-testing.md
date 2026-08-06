# Agent Testing Guide — speaker

## Environment
- Runtime: Python audio stack.
- Preferred interpreter on this box: `/home/dobo/work/speaker/.venv/bin/python`.
- Live hardware validation is separate from headless pytest verification.

## Commands
| Scope | Command | Expected success |
|---|---|---|
| Live-launcher lifecycle (headless) | `/home/dobo/work/speaker/.venv/bin/python -m pytest tests/test_live_launcher.py tests/test_capture_integration.py tests/test_setup_doctor.py -q` | setup/reuse/failure/cleanup/private-path/doctor contracts pass without opening audio |
| APM/DTD regression | `/home/dobo/work/speaker/.venv/bin/python -m pytest tests/test_apm_double_talk.py -q` | `6 passed` on current setup |
| Real-time media-stage lifecycle | `SPEAKER_TEST_LOG=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_realtime_media_stage.py tests/test_asr_final_async.py tests/test_sherpa_media_session.py tests/test_sherpa_vad_final_gate.py tests/test_barge_word_cut.py tests/test_sherpa_playback.py tests/test_acoustic_lineage.py tests/test_media_session.py -m "not real_model" -q` | bounded nonblocking handoff, exact scope/cancellation, immutable owned final PCM, overload/shutdown/restart fences, and adjacent capture/acoustic behavior pass without models or audio devices (ADR-0107) |
| Complete local Sherpa decode owner | `SPEAKER_TEST_LOG=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_sherpa_streaming_decode_owner.py tests/test_sherpa_streaming_decode_session.py tests/test_media_session.py tests/test_sherpa_media_session.py tests/test_asr_postprocess.py tests/test_barge_confirm.py tests/test_barge_word_cut.py tests/test_sherpa_vad_final_gate.py tests/test_virtual_audio_engine.py tests/test_sherpa_duplex_runtime.py -q` | complete exact-thread DSP/VAD/primary+word-cut state machine, immutable mailbox ownership, effect-lease races, scope/continuity/incarnation rotation, native-fault recovery, startup rollback, re-entrant stop/recorder cleanup, restart, and fail-closed active-hotword incompatibility pass with fakes; no model or audio-device evidence (ADR-0110/0111/0114) |
| Trusted LiveKit publisher session | `/home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_livekit_agents_engine.py tests/test_livekit_agents_session.py -q` | `73 passed`: exact `livekit==1.1.14`/Agents 1.6.7 validation, manual RoomIO without implicit control/record/preconnect paths, pre-ready inbound-event suppression, console/start/ready/close races, all-reason publisher-disconnect fencing, route leases, and retryable disposal with a fake SDK; no installed SDK, network, audio device, or live validity (ADR-0104) |
| Bounded post-ASR mailbox | `/home/dobo/work/speaker/.venv/bin/python -B -m pytest tests/test_event_bus.py tests/test_always_on_agent.py tests/test_playback_receipts.py tests/test_final_preprocessing_cancel.py tests/test_never_stuck.py -q` | bounded/coalesced data, preemptive lifecycle service, aged ordered-output service, supervisor/playback races, and shutdown contracts pass headlessly (ADR-0093) |
| Conversation trace (device-free) | `/home/dobo/work/speaker/.venv/bin/python -m tools.conversation_eval --runs 3` | all 14 v4 scenarios pass 3/3; aggregate 42/42, `pass^3=True`, exact declared answer routes, and controller-owned bounded correction (ADR-0051/0067/0068) |
| Headless conversation flow | `SPEAKER_TEST_LOG=0 /home/dobo/work/speaker/.venv/bin/python -B -m tools.conversation_eval --flow-acceptance` | unchanged v4 42/42 plus flow-v1 12/12 and `pass^3=True`; typed turn, scripted playback/interruption, delayed read tool, confirmed synthetic action, and restart fencing pass through production control-plane ownership. The report explicitly excludes microphone/VAD wiring, STT/WER, real TTS/audio, AEC/room, external effects, and live latency (ADR-0112) |
| MiniCPM Q8 production-hybrid A/B | `/home/dobo/work/speaker/.venv/bin/python -m tools.conversation_eval --mode ollama --candidate-model minicpm5-1b:q8 --baseline-model gemma3:12b --runs 3` | clean revision; candidate and baseline each 42/42; verified identities, warm budgets, exact answer routes, and no regression (ADR-0051/0067/0068) |
| MiniCPM all-role stress | add `--topology all-roles` to the prior command | ADR-0051 replacement-stress diagnostic; never an adoption result |
| Cross-session semantic memory | `/home/dobo/work/speaker/.venv/bin/python -m tools.autotest memory` | SQLite reopen; clean native history; balanced fenced injection; PRIVATE first/only `route=main`; clause-grounded canary; clean/stable revision, contract, and full shipped-role identities (Echo is incomplete) (ADR-0065) |
| Autonomous cable diagnostic | `/home/dobo/work/speaker/.venv/bin/python -m tools.autotest voice --acoustics cable` | pipeline checks green but report says `diagnostic_pass`, `ok=false`, and barge/self/latency `not_covered`; not a landing result (ADR-0055/0058) |
| Autonomous silent voice gate | `/home/dobo/work/speaker/.venv/bin/python -m tools.autotest all --acoustics delay` | exact run-owned virtual EC topology and capture/playback bindings; labelled causal finals/terminals; mean WER ≤0.50; zero errors/stuck/self-cuts; synthesized exact-command cut from engine capture onset in 0–1.4 s; all six route/exit/cleanup proofs; aggregate PASS (ADR-0058/0069/0070) |
| Injected Sherpa barge replay | `/home/dobo/work/speaker/.venv/bin/python -m tools.live_session --scenario barge_in_interrupt_stop --repeat 3 --inject --barge-in --llm echo --no-assistant-audio` | exit 0 only when every repetition is full-duplex `ok`, intended FIFO cuts 2/2, zero self-interrupts (ADR-0064) |
| Recorded owner-voice landing gate | `SPEAKER_REQUIRE_RECORDED=1 /home/dobo/work/speaker/.venv/bin/python -m pytest tests/replay_recorded_voice_test.py -q` | reference host: exactly 9 passed/0 skipped (six utterances, one same-session multi-turn, two causal fake-stream owner talk-overs); missing private clips/models fail (ADR-0053) |
| Recording-driven STT accuracy | `/home/dobo/work/speaker/.venv/bin/python -m tools.recorded_stt_eval` | aggregate-only streaming/offline/selected WER+CER over every hash-pinned labelled clip; each selected offline recognizer/verifier must complete a decode with zero error outcomes; no runtime, TTS, tools, network, or audio device (ADR-0078/0080) |
| BPE streaming-hotword context | `SPEAKER_TEST_LOG=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_sherpa_models.py tests/test_setup_doctor.py tests/test_recorded_stt_eval.py tests/test_capture_replay_eval.py tests/test_install.py tests/test_asr_postprocess.py tests/test_file_replay_engine.py tests/test_livekit_engine.py -q` | constructor/stream fail-closed behavior across local, replay, and LiveKit; exact uppercase grammar; immutable complete-family setup; pre-download exporter check; readiness; conditional install; and exact active/inert replay provenance pass without a model, network, GPU, audio device, transcript output, or live claim (ADR-0114) |
| Exact private final-input evidence | `SPEAKER_TEST_LOG=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_diagnostic_bundle.py tests/test_reference_recording.py tests/test_sherpa_diagnostic_observations.py tests/test_asr_final_async.py tests/test_private_diagnostic_receipt.py tests/test_prepare_diagnostic_streaming_stt_corpus.py tests/test_recorded_stt_eval.py tests/test_streaming_stt_corpus_writer.py tests/test_streaming_stt_eval.py tests/test_app_replay_safety.py -q` | diagnostic schema-v1 read-only validation, schema-v2 exact f32le binding/lifecycle/tamper/backpressure, private schema-v3 bit-for-bit export and exact configured-selector replay, mandatory receipt-v2 label/case binding and v1 rejection, bounded regular-manifest dispatch, role/source accounting, post-copy/close-time mutation and report/input-alias rejection in every shared consumer, public schema-v2 separation, aggregate-only CLI privacy, and synchronized-directory replay refusal pass headlessly; no owner corpus, model, GPU, audio device, WER, latency, or live claim (ADR-0108/0124/0126) |
| Dry recorded tool-route gate | `SPEAKER_TEST_LOG=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_tool_route_gate.py tests/test_recorded_stt_eval.py tests/test_public_voice_fixtures.py tests/test_obsidian.py tests/test_trusted_apps.py tests/test_device_tools_runtime.py -q` | receipt-bound schema-v3 expected-tag preflight, one-final boundary, closed vault/web/reminder/app/negative routing, fixed-clock empty-state matching, no-invoke/no-launch sentinels, aggregate/profile privacy, and candidate promotion safety pass with production analyzer/planner/match contracts only. No model, provider, capability invocation, vault read, reminder mutation, app launch, runtime, device, or live authority follows (ADR-0125/0126) |
| Capture-loop replay contracts | `/home/dobo/work/speaker/.venv/bin/python -B -m pytest tests/test_capture_replay_corpus.py tests/test_capture_replay.py tests/test_capture_replay_metrics.py tests/test_capture_replay_eval.py tests/test_prepare_ami_capture_replay.py -q` | strict corpus, paced synchronous Sherpa session seam, fixed aggregate conditions/privacy, evaluator, deterministic AMI preparation, and explicit production-owner rejection pass without a model or audio device; the digest binds but replay does not execute the dedicated owner (ADR-0092/0111) |
| EdAcc endpoint-integrity contracts | `SPEAKER_TEST_LOG=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 nice -n 15 ionice -c 3 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_edacc_endpoint_integrity_eval.py tests/test_capture_replay_eval.py -q` | `103 passed`: exact retained-corpus/profile/Smart-Turn admission; distinct acoustic, HOLD-only, and opt-in reset-and-accumulate capture-loop cells; false/false/true reset attestation; fixed logical rule-3 binding; all pairwise aggregate transitions; HOLD exercise; fresh row policy; privacy; guarded execution; retained-byte verification; rechecks; and no-clobber publication pass with fakes. No model, audio device, runtime default, or live claim follows (ADR-0147/0148) |
| Isolated streaming-STT harness | `SPEAKER_TEST_LOG=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_prepare_public_streaming_stt_corpus.py tests/test_prepare_nemotron_runtime.py tests/test_prepare_parakeet_runtime.py tests/test_provision_moonshine_candidate.py tests/test_provision_nemotron_candidate.py tests/test_provision_parakeet_realtime_eou_candidate.py tests/test_provision_parakeet_cpp_candidate.py tests/test_parakeet_cpp_worker_supervisor.py tests/test_streaming_stt_*.py -m "not real_model" --deselect tests/test_streaming_stt_manifest.py::test_bound_file_hash_rejects_in_place_mutation_during_streaming_read -q` | `1,015 passed, 1 skipped, 5 deselected`: fake, Moonshine Tiny/Small/Medium, Nemotron, Parakeet NeMo, parakeet.cpp, phase-separated scoped-resource, and AMI endpoint-proxy private-source/exact-runtime or native-artifact, bounded protocol, sandbox, provenance, aggregate/privacy, and teardown contracts pass without a model or audio device; run the bounded-read race separately (ADR-0089/0090/0091/0099/0115/0119/0128/0129) |
| Exact parakeet.cpp CPU candidate contracts | `SPEAKER_TEST_LOG=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_provision_parakeet_cpp_candidate.py tests/test_parakeet_cpp_worker_supervisor.py tests/test_streaming_stt_parakeet_cpp.py tests/test_streaming_stt_metrics_parakeet_cpp.py tests/test_streaming_stt_manifest.py tests/test_streaming_stt_protocol.py tests/test_streaming_stt_supervisor.py tests/test_streaming_stt_eval.py -m "not real_model" --deselect tests/test_streaming_stt_manifest.py::test_bound_file_hash_rejects_in_place_mutation_during_streaming_read -q` | `464 passed, 1 deselected`: schema-v8 closed receipts and no-load provision, exact bridge/source/ELF/ABI binding, protocol-v4 first-EOU/text-freeze policy, no-network/no-NVIDIA sandbox, verified hard scope, aggregate metrics, evaluator binding, privacy, and teardown pass with fakes; no real model, device, live, tool, or adoption claim (ADR-0119) |
| Phase-separated cgroup observations | `SPEAKER_TEST_LOG=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_streaming_stt_cgroup_observation.py tests/test_streaming_stt_supervisor.py tests/test_parakeet_cpp_worker_supervisor.py tests/test_streaming_stt_eval.py -q` | `210 passed`: pinned fake-cgroup/fake-clock parsing, bounded multi-line reads, exact startup/first-use/resident lifecycle hooks and failure cleanup, membership/path/counter failure, evaluator binding, privacy, and unchanged legacy report shape pass without a model, GPU, audio device, cache eviction, or cold/warm claim (ADR-0128) |
| Faster-Whisper final-only comparator | `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_provision_faster_whisper_endpoint_candidate.py tests/test_streaming_stt_faster_whisper_endpoint.py -q` | separate runtime/model receipts, no-download provision, final-only adapter, evaluator provenance, PCM/tree tamper rejection, and schema-v6 worker serialization pass without importing/loading a candidate, GPU, network, or audio device (ADR-0102) |
| Public matrix + production-model STT control | `SPEAKER_TEST_LOG=0 /home/dobo/work/speaker/.venv/bin/python -B -m pytest tests/test_public_voice_eval_matrix.py tests/test_streaming_stt_zipformer.py -m "not real_model" -q` | no-download license/receipt/selection/privacy contracts and exact fp32 Zipformer Bubblewrap control pass without a corpus, model load, network, or audio device (ADR-0098) |
| LiveKit endpoint-source admission | `SPEAKER_TEST_LOG=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_public_endpoint_eval_catalog.py tests/test_livekit_eot_parquet_worker.py tests/test_inventory_livekit_eot_english.py -q` | exact endpoint-only catalog and immutable legacy-closure hashes; explicit private source/license/output contracts; isolated PyArrow schema, embedded-WAV, HOLD/EOT, aggregate/privacy, no-clobber, exact four-file execution closure with pre/post-publication rebinding, and recognized-Git-worktree boundary behavior, including inert empty sentinels plus same-inode/disagreement/growth/replacement races, pass with synthetic fixtures. The separately run exact private shard inventory is aggregate-only; no WER, model, GPU, network, audio device, runtime, or default claim follows (ADR-0134) |
| Causal LiveKit endpoint diagnostic | `SPEAKER_TEST_LOG=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 nice -n 15 ionice -c 3 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_endpointing.py tests/test_livekit_causal_endpoint_eval.py -m "not real_model and not slow and not backend" -q` | `141 passed, 5 deselected`: shared production-observation parity, causal HOLD/EOT grid and censoring math, immutable prefix/no-partial behavior, strict protocols including conservative cross-summary and percentile-order tamper rejection, retained-descriptor scratch cleanup, bounded worker failure, aggregate privacy, descriptor-bound execution closure/no-clobber publication, current >512 KiB Sherpa closure admission, unchanged public 512 KiB rejection, unallowlisted-path rejection, and lexical venv-symlink preservation pass with fakes. The deselected exact-source/model tests stay separate so ambient private-path variables cannot change this headless claim; no exact source/model, Sherpa/VAD/STT, audio device, threshold, default, or live claim follows (ADR-0136) |
| Licensed stratified STT evidence | `SPEAKER_TEST_LOG=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_prepare_demand_noise_streaming_stt_corpus.py tests/test_public_voice_eval_matrix.py tests/test_streaming_stt_corpus_writer.py tests/test_streaming_stt_eval.py tests/test_streaming_stt_faster_whisper_endpoint.py tests/test_streaming_stt_suite.py -q` | exact DEMAND/license/runtime receipts, outside-Git publication, Rochester label fail-closed behavior, bounded strata, N/A zero-attempt command recall, evaluator binding, sequential cells, failure checkpoints, and aggregate privacy pass without a model, network, GPU, or audio device (ADR-0109) |
| Public short-command/noise gate | `SPEAKER_TEST_LOG=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_prepare_public_command_noise_corpus.py tests/test_public_voice_eval_matrix.py tests/test_streaming_stt_corpus_writer.py tests/test_streaming_stt_manifest.py tests/test_streaming_stt_eval.py -m "not real_model" --deselect tests/test_streaming_stt_manifest.py::test_bound_file_hash_rejects_in_place_mutation_during_streaming_read -q` | `246 passed, 1 deselected`: exact official-archive/catalog/license/lock binding, bounded direct archive reads, deterministic 57-case selection, schema-v4 assertion and receipt integrity, final/partial positive and forbidden-target metrics, aggregate privacy, and no-clobber outside-Git publication pass without downloading, loading a model, using a GPU, network, or audio device (ADR-0117/0119) |
| Configured final-command FileReplay gate | `SPEAKER_TEST_LOG=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_production_final_stt_eval.py tests/test_streaming_stt_final_metrics.py tests/test_recorded_stt_eval.py tests/test_file_replay_engine.py -q` | decision-boundary-preserving aggregate metrics, exact private corpus/config/model/source/runtime/provider binding, schema-v2 compatibility, atomic schema-v3 complete-profile pairing, profile-aware verifier attestation, streaming-control validity, close-time privacy/mutation checks, model-output suppression, and failure-clean no-clobber publication pass with fakes; selected output explicitly lacks live VAD-owned duration/recovery (ADR-0118/0145) |
| Atomic profile command/noise A/B | `/home/dobo/work/speaker/.venv/bin/python -B -m tools.production_final_stt_eval --corpus /private/command-noise/corpus.json --config /home/dobo/work/speaker/config.json --local-config /home/dobo/work/speaker/config.local.json --device desktop_gpu_4090 --baseline-final-stt-profile sense-voice --candidate-final-stt-profile parakeet-faster-whisper --repeats 1 --stratum-tag command-negative --stratum-tag command-positive --stratum-tag eccc --stratum-tag gsc --stratum-tag noisy --stratum-tag silence --stratum-tag speech-negative --report /private/new-profile-pair-report.json` | fixed ordered same-PCM replay prebinds both model closures before execution and publishes aggregate-only schema v3. The retained exact 57x2 run had matching streaming controls; candidate gained two clean hits but added four noisy false activations with no noisy recall gain, so it remains non-promotional. Sequential timing is not comparable and no capture, endpoint, audio-device, AEC, barge-in, tool, or live claim follows (ADR-0145) |
| Bounded-read race isolation | `SPEAKER_TEST_LOG=0 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_streaming_stt_manifest.py::test_bound_file_hash_rejects_in_place_mutation_during_streaming_read -q` | `1 passed`; keep this pre-existing timestamp-sensitive mutation check isolated from the broad streaming manifest gate and report both conditions |
| Public conversation/STT fixture lock | `SPEAKER_TEST_LOG=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_public_conversation_fixture.py tests/test_prepare_harper_valley_conversation_fixture.py tests/test_prepare_edacc_conversation_fixture.py tests/test_prepare_ami_conversation_fixture.py tests/test_prepare_common_voice_conversation_fixture.py tests/test_public_voice_eval_matrix.py tests/test_prepare_common_voice_spontaneous_v4.py tests/test_streaming_stt_corpus_writer.py tests/test_streaming_stt_suite.py -q` | `325 passed`: matrix-v5 pins/licenses, raw-source contracts, self-digested 24x4 lock, explicit slots, private receipt/corpus/reference/PCM cross-binding, and EdAcc production-owned raw-header scan with bounded base-256 only for ignored UID/GID, a required bounded root README, exact plain/single-digit-participant WAV grammar, segment-to-WAV equality, strict terminal-suffix conversation-family projection, recording/family split disjointness, exact fixed CSV-header aliases including `PARTICIPANT_ID` without case/whitespace normalization, retained accent/duplicate/row parsing, descriptor extraction, persisted-byte, malformed-archive, race, and retry rejection. Source balance/pairing, CV parent re-decode/private-input binding, integrated sequential-suite validation, and coordinated tamper rejection also pass. Materializer outputs in this gate are synthetic/non-production; production-shaped validator fixtures are consistency tests only. Separately, the retained exact-source run published a `production_evidence=true` 24-case EdAcc corpus (4,790,400 PCM bytes) with corpus SHA-256 `4da392c39a0b6bd18057f63c96b4f67c0dfdaf4c14e1d2d76176cdbee0920772` and receipt SHA-256 `3c516b6fbb8ce139498cd2e01d83a4faf825e58bf2b9abcaa52896bdab4c853f`; preserve it alongside every failed and non-production tree. The command itself downloads nothing and provides no model, WER, GPU, audio-device, live, or runtime-default evidence (ADR-0113/0117/0131/0132/0142) |
| Public conversation canonical strata | `SPEAKER_TEST_LOG=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_public_conversation_qualification.py tests/test_public_conversation_fixture.py tests/test_streaming_stt_suite.py tests/test_streaming_stt_eval.py -q` | `214 passed`: exact lock/slot-derived marginal strata and fixed baseline/candidate 2x4 order; qualification-v2 reconstruction of the fixed aggregate-only safe suite; exact adapter/resolved-stream, source/safe digest-domain, denominator, endpoint-invariant, and retained-roundtrip validation; private bounded checkpoint reads, repeated close-time controller/fixture/worker/scratch checks, failure-clean no-clobber publication, exact file digest, and explicit coverage-only/non-promotional semantics pass with generated fixtures and fake workers. No corpus download, model, GPU, network, audio device, quality verdict, or live evidence (ADR-0127/0130) |
| Fixed EdAcc-only 2x1 comparison | `env SPEAKER_TEST_LOG=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 nice -n 15 ionice -c 3 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_edacc_stt_comparison.py -q` | `47 passed`: exact production corpus/receipt pins, 24-case 8/8/8 closure, schema-4 `production-gigaspeech-zipformer-fp32-benchmark-1t` then schema-6 English `faster-whisper-small-local`, fixed adapters and three-repeat/1,600-sample/burst/200 ms/zero-tail settings, strict sequential delegation, source/worker/code rechecks, private checkpoint/scratch separation, aggregate-only safe reconstruction, retained validation, and atomic no-clobber publication pass with generated fixtures and fake workers. No private corpus, model, GPU, network, audio device, quality decision, latency, or live evidence (ADR-0143) |
| AMI forced-alignment endpoint proxy | `SPEAKER_TEST_LOG=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_prepare_ami_conversation_fixture.py tests/test_streaming_stt_metrics_ami_endpoint_proxy.py tests/test_ami_endpoint_proxy_eval.py tests/test_streaming_stt_eval_ami_endpoint_proxy.py tests/test_streaming_stt_eval.py -q` | exact private sidecar/provenance/receipt/label-set binding, isolated-only aggregate reducer, supported native/first-feed EOU sample semantics, AMI-only pre/post validation, no-clobber publication, legacy report compatibility, and privacy pass with generated fixtures and fake workers; this is a non-authoritative forced-alignment proxy with no corpus download, model, GPU, network, audio device, threshold, promotion, or live evidence (ADR-0129) |
| Common Voice spontaneous P0 preparation | `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_prepare_common_voice_spontaneous_v4.py tests/test_streaming_stt_corpus_writer.py tests/test_public_voice_eval_matrix.py -q` | exact release/API receipt, twice-hashed private archive, safe tar/TSV, speaker-first quota matching, bounded publication/provenance, and synthetic decode pass; the two real-PyAV checks run only when the pinned optional `requirements-evaluation.txt` environment is installed. No corpus download, model, network, GPU, or audio device (ADR-0101) |
| VoxPopuli Romanian-accented English P0 preparation | `SPEAKER_TEST_LOG=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_prepare_voxpopuli_ro_accent.py tests/test_streaming_stt_corpus_writer.py -q` | exact two-shard/revision/license pins, explicit-local-only descriptor transport, exact float32-WAV/fact/empty-row contracts, finite identity conversion, deterministic 24-speaker four-band assignment, aggregate privacy, and hardened schema-v2 publication pass synthetically. The first real attempt failed closed; no real corpus was published and no model, network, GPU, or audio device runs here (ADR-0106) |
| Exact Moonshine worker smoke | set absolute `SPEAKER_MOONSHINE_WORKER_MANIFEST` and `SPEAKER_MOONSHINE_STREAMING_CORPUS`, then run `/home/dobo/work/speaker/.venv/bin/python -m pytest tests/test_streaming_stt_moonshine_worker.py -m real_model -q` | one exact pre-provisioned case runs with no installer/downloader or audio device; the worker lacks a network namespace and supplies no adoption claim (ADR-0090/0115) |
| Exact Nemotron worker smoke | prepare and provision new private paths as shown below, then run the candidate-specific `real_model` command | one exact pre-provisioned case passes through no-network Bubblewrap and the bound inputs still pass close-time rehash; no audio device or adoption claim (ADR-0091) |
| Opt-in selected final STT setup | `./install.sh --skip-system --final-asr parakeet-unified-en --final-verifier faster-whisper-small` | verify/stage the four-file Parakeet package and pinned Linux/NVIDIA verifier, publish atomically, and pass doctor; normal sessions still use `./live.sh` (ADR-0080) |
| Atomic final-STT recording A/B | `/home/dobo/work/speaker/.venv/bin/python -m tools.recorded_stt_eval --manifest /private/corpus/corpus.json --baseline-final-stt-profile sense-voice --candidate-final-stt-profile parakeet-faster-whisper --keyword vault --output /private/report.json` | same exact PCM passes through two closed complete production-selector profiles with profile/config/active-artifact digests and aggregate-only verifier outcome counts; the paired flags reject ambient self-comparison and field-level mixing. No transcript rows, capture, endpoint, audio device, latency, or default authority (ADR-0078/0080/0144) |
| Atomic final-STT physical A/B | `./live.sh --run-label owner-stt-sense-control --llm echo --final-stt-profile sense-voice`, then the same owner plan through `./live.sh --run-label owner-stt-parakeet-fws --llm echo --final-stt-profile parakeet-faster-whisper` | one session-scoped complete profile reaches both readiness and the recorded core child; private summary binds its safe name/digest. Physical owner review, equal plan execution, and separate bundle comparison are required; this row claims no unrun live result (ADR-0100/0138/0144) |
| Isolated enrollment prep | `/home/dobo/work/speaker/.venv/bin/python -m tools.prepare_enrollment --help` then supply the four explicit absolute paths and a unique `enrollment.v5-<id>.json` | device-free; verified no-clobber backup, empty feature candidate, regular mode-600 config with prepared marker; use its exact printed next command (ADR-0056) |
| Accepted v5 promotion | `/home/dobo/work/speaker/.venv/bin/python -m tools.promote_enrollment --help` then supply the exact worktree, primary config, prepared candidate/source/backup, candidate-derived adjacent accepted path, and `--accept-live-gate` | device-free and only after manual acceptance; exit 0 = active, 2 = refused, 3 = confirmed staged/inactive, 4 = ambiguous (ADR-0066) |
| Whitespace | `git diff --check` | no output |

## Exact EdAcc two-model development comparison

Run ADR-0143 only after provisioning a fresh Zipformer baseline manifest and a
fresh English Faster-Whisper Small endpoint manifest with exact model ID
`faster-whisper-small-local` from the same landed checkout that provides the
wrapper. The retained production corpus path below
is fixed; its sibling `preparation-receipt.json` is discovered and checked by
the wrapper. Substitute only the two provision paths and choose new, absent
scratch/report destinations for each attempt:

```bash
export EDACC_CORPUS=/var/tmp/speaker-edacc-private-20260805/corpus-retry-adr0142/edacc-test-v1/corpus.json
export EDACC_ZIPFORMER_MANIFEST=/absolute/private/current-checkout-zipformer/worker-manifest.json
export EDACC_FASTER_WHISPER_MANIFEST=/absolute/private/current-checkout-faster-whisper-small/worker-manifest.json
export EDACC_RUN_ID=replace-with-one-unique-run-id
export EDACC_2X1_SCRATCH=/var/tmp/speaker-edacc-private-20260805/new-edacc-2x1-scratch-${EDACC_RUN_ID}
export EDACC_2X1_REPORT=/var/tmp/speaker-edacc-private-20260805/new-edacc-2x1-report-${EDACC_RUN_ID}.json

SPEAKER_TEST_LOG=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
nice -n 15 ionice -c 3 /home/dobo/work/speaker/.venv/bin/python -B \
  -m tools.edacc_stt_comparison \
  --corpus "$EDACC_CORPUS" \
  --baseline-worker-manifest "$EDACC_ZIPFORMER_MANIFEST" \
  --candidate-worker-manifest "$EDACC_FASTER_WHISPER_MANIFEST" \
  --scratch-root "$EDACC_2X1_SCRATCH" \
  --output "$EDACC_2X1_REPORT"
```

The retained exact run from main/evaluation checkout `96c36fe` completed both
coverage-only cells: 72 evaluations per cell across three repeats, with zero
final disagreements. Its aggregate report is 20,956 bytes, mode 0600, has one
link, and has SHA-256
`7edcd8a6e2c8422799a94f5b852314517170959167d2821c510a73cff56ed0ba`.
Preserve it and all evidence to which it is bound.

The Zipformer manifest SHA-256 was
`ae9db644c22cac5d24107253707729fc4b76d5b4e1c8ce9080bb55c4c3113516`.
That cell reported WER/CER `.6583`/`.4884`, 0 exact matches, 57 nonempty
finals, clean/disfluent/overlap-adjacent WER `.6250`/`.7627`/`.6053`, 376.141
MB RSS, and one thread. The Faster-Whisper Small manifest SHA-256 was
`594128b55e675363357d446d2e3d3724c474b7e935e537b675926ffd8ddc14da`;
its runtime-receipt SHA-256 was
`85dfc56712f9db4997836329c4fdb2b1e56a265ac9a54a4d5859f74e2ea309a2`
and its model-receipt SHA-256 was
`28d63ead1b37e6241dbca196155dfc988ea61220e7484ad64ca60b15e9f7c132`.
That cell reported WER/CER `.3719`/`.2464`, 15 exact matches, 72 nonempty
finals, stratum WER `.3281`/`.5085`/`.3026`, 1,052.668 MB RSS, and six threads.
The observed WER difference is 28.64 percentage points, about 43.5% relative
to Zipformer, on this fixed EdAcc slice.

Reported peak VRAM was null for both cells, and the external monitor did not
sample the active interval; make no peak-VRAM claim. RTF and timing are
non-comparable and are not live latency. This small accuracy slice remains
development-only and non-promotional, with no quality verdict or default
change.

Do not add geometry or pacing flags: the wrapper fixes three repeats,
1,600-sample chunks, burst pacing, a 200 ms partial interval, zero tail
padding, and baseline-then-candidate execution. Both destination parents must
be owner-private and outside Git, the corpus/receipt tree, and either worker's
provision/runtime/model tree; scratch and output must not already exist. Keep
the report and printed exact published-byte SHA-256. A successful command
means the fixed accuracy slice completed; it is not latency, endpoint,
resource-acceptance, training-disjointness, adoption, runtime, or live evidence.

## Exact causal LiveKit endpoint diagnostic

ADR-0136 limits this command to a private aggregate diagnostic. Use the exact
ADR-0134 inventory report and source, the isolated PyArrow 25 environment, the
ADR-0135 model, canonical `config.json`, and new absent scratch/report paths:

```bash
export LIVEKIT_EOT_PARQUET=/absolute/private/validation-00000-of-00001.parquet
export LIVEKIT_EOT_INVENTORY=/absolute/private/livekit-eot-inventory.json
export PYARROW25_PYTHON=/absolute/private/pyarrow25-venv/bin/python
export SMART_TURN_MODEL=/home/dobo/work/speaker/pretrained_models/sherpa/turn/smart-turn-v3.2-cpu.onnx
export CAUSAL_SCRATCH=/absolute/private/new-livekit-causal-scratch
export CAUSAL_REPORT=/absolute/private/new-livekit-causal-report.json

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 \
  /home/dobo/work/speaker/.venv/bin/python -B \
  -m tools.livekit_causal_endpoint_eval \
  --source-parquet "$LIVEKIT_EOT_PARQUET" \
  --inventory-report "$LIVEKIT_EOT_INVENTORY" \
  --parquet-python "$PYARROW25_PYTHON" \
  --model "$SMART_TURN_MODEL" \
  --config /home/dobo/work/speaker/config.json \
  --scratch-root "$CAUSAL_SCRATCH" \
  --output "$CAUSAL_REPORT" \
  --accept-license CC-BY-4.0 \
  --accept-partial-assumption
```

Both destination paths must be absolute, private, outside recognized Git
worktrees, and absent before the run. The partial flag acknowledges the fixed
opaque nonempty same-epoch-partial assumption; it does not use publisher text.
The report is aggregate-only counterfactual evidence. Keep its printed SHA-256,
and do not interpret it as production runtime, inference latency, capture/VAD,
STT, device, or live-conversation evidence.

The hardened 2026-08-05 exact run passed the separate pinned-source and
proc-fd CPU-model gates at one test each, then published a mode-0600,
single-link, 13,966-byte aggregate report with SHA-256
`ee07bfbe0f1b16a4fafee44c4f4901a3799adcdf7bffa46ad8df775ce833e9e2`;
its scratch path was absent after cleanup. The current execution closure and
source bundle match the report. The pre-landing config SHA-256
`435a7dbda7aaad151e4b7d77ae0fba7f3ea544b75c9d2df543ef415740c1c811`
was byte-identical to `main`; after landing, keep the canonical main config
path shown above.

On the full 850-HOLD/400-EOT counterfactual, the opaque-partial candidate cut
57 HOLD labels across 32 rows, versus no-partial acoustic fallback 93 across
49. Candidate EOT committed 350, right-censored 50, and had conservative
p50/p95 700/1,600 ms; its observed commits had p50/p95 700/900 ms. Fallback
committed all 400 at 800 ms. On publisher labels wholly before rule 3, the
candidate cut 46/646 HOLD labels and committed 300/332 EOT labels, with 32
censored and conservative p50/p95 700/1,600 ms. Runtime versions were ONNX
Runtime 1.27.0, NumPy 2.4.6, and PyArrow 25.0.0 on CPUExecutionProvider.
These are diagnostic-only tradeoff results: do not change a threshold, model,
or default, and do not infer STT, device, latency, or live quality from them.

## Exact Moonshine Medium reference setup

Start with a disposable private Python 3.12 environment whose
`include-system-site-packages` marker is false, the exact Moonshine Voice 0.1.0
Linux wheel, the seven already-downloaded official Medium Streaming files, and
a prepared schema-v2 corpus. Do not import Moonshine before provisioning:
generated `__pycache__` files correctly make the exact wheel/tree check fail.
Every destination below must be a new absolute private path:

```bash
export MOONSHINE_RUNTIME=/absolute/private/moonshine-runtime
export MOONSHINE_WHEEL=/absolute/private/moonshine_voice-0.1.0-py3-none-manylinux_2_34_x86_64.whl
export MOONSHINE_MODEL=/absolute/private/medium-streaming-en
export MOONSHINE_PROVISION=/absolute/private/new-moonshine-medium-candidate
export STREAMING_CORPUS=/absolute/private/prepared-public-corpus/corpus.json
```

Bind the existing runtime, exact wheel, and model without installing,
downloading, importing the candidate, or opening an audio device:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 \
  /home/dobo/work/speaker/.venv/bin/python -B \
  -m tools.provision_moonshine_candidate \
  --venv-root "$MOONSHINE_RUNTIME" --wheel "$MOONSHINE_WHEEL" \
  --model-root "$MOONSHINE_MODEL" --model-arch medium-streaming \
  --output-dir "$MOONSHINE_PROVISION"
```

Run the one-case receipt smoke, then use new private scratch/report paths for
the aggregate burst and paced cells:

```bash
SPEAKER_MOONSHINE_WORKER_MANIFEST="$MOONSHINE_PROVISION/worker-manifest.json" \
SPEAKER_MOONSHINE_STREAMING_CORPUS="$STREAMING_CORPUS" \
  /home/dobo/work/speaker/.venv/bin/python -B -m pytest \
  tests/test_streaming_stt_moonshine_worker.py -m real_model -q

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 \
  /home/dobo/work/speaker/.venv/bin/python -B -m tools.streaming_stt_eval \
  --worker-manifest "$MOONSHINE_PROVISION/worker-manifest.json" \
  --corpus "$STREAMING_CORPUS" --repeats 3 --chunk-samples 1600 \
  --partial-interval-ms 500 --tail-padding-samples 0 --pace burst \
  --scratch-root /absolute/private/new-moonshine-burst-scratch \
  --output /absolute/private/new-moonshine-burst-report.json

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 \
  /home/dobo/work/speaker/.venv/bin/python -B -m tools.streaming_stt_eval \
  --worker-manifest "$MOONSHINE_PROVISION/worker-manifest.json" \
  --corpus "$STREAMING_CORPUS" --repeats 1 --chunk-samples 1600 \
  --partial-interval-ms 200 --tail-padding-samples 0 --pace realtime \
  --scratch-root /absolute/private/new-moonshine-paced-scratch \
  --output /absolute/private/new-moonshine-paced-report.json
```

The native 0.1.0 worker is CPU-only. It sanitizes thread settings and runs
sequentially, but has no Bubblewrap network namespace or independently verified
hard RAM/CPU limit. These aggregate runs begin after PCM is available and do
not validate capture, VAD/endpoint ownership, AEC, commands, live latency, or
adoption (ADR-0115).

### Stock Moonshine external-boundary diagnostic

The separate schema-v7 diagnostic recorded by ADR-0116 uses a new private
destination and the same exact runtime, wheel, model, and public corpus. Bind
the external-presegmented profile, then run its candidate-specific A-B-A smoke:

```bash
export MOONSHINE_EXTERNAL_PROVISION=/absolute/private/new-moonshine-external-candidate

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 \
  /home/dobo/work/speaker/.venv/bin/python -B \
  -m tools.provision_moonshine_candidate \
  --venv-root "$MOONSHINE_RUNTIME" --wheel "$MOONSHINE_WHEEL" \
  --model-root "$MOONSHINE_MODEL" --model-arch medium-streaming \
  --segmentation-mode external-presegmented \
  --output-dir "$MOONSHINE_EXTERNAL_PROVISION"

SPEAKER_MOONSHINE_SCHEMA7_WORKER_MANIFEST="$MOONSHINE_EXTERNAL_PROVISION/worker-manifest.json" \
SPEAKER_MOONSHINE_STREAMING_CORPUS="$STREAMING_CORPUS" \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 \
  /home/dobo/work/speaker/.venv/bin/python -B -m pytest \
  -p no:cacheprovider tests/test_streaming_stt_moonshine_worker.py \
  -m real_model -k schema_v7 -q
```

Schema v7 rejects stream overrides: chunk size is 1,280 samples, partial
cadence is 500 ms, and caller tail padding is zero. Run one aggregate burst
cell with new private mode-700 scratch and report paths; omitting the geometry
flags uses those receipt-bound values:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 \
  /home/dobo/work/speaker/.venv/bin/python -B -m tools.streaming_stt_eval \
  --worker-manifest "$MOONSHINE_EXTERNAL_PROVISION/worker-manifest.json" \
  --corpus "$STREAMING_CORPUS" --repeats 1 --pace burst \
  --scratch-root /absolute/private/new-moonshine-external-scratch \
  --output /absolute/private/new-moonshine-external-report.json
```

This is complete-PCM, aggregate-only development evidence. It does not run or
validate Speaker's endpoint, internal VAD quality, capture, AEC, barge-in,
commands, tools, a device, or live conversation (ADR-0116).

## Timestamped capture-loop replay

Obtain the AMI manual annotations and `ES2004a.Array1-01.wav` from the
[official AMI corpus](https://groups.inf.ed.ac.uk/ami/download/) under its
published terms. The tested v1.6.2 sources are:

- annotations: 22,887,865 bytes, SHA-256
  `b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d`
- audio: 33,579,394 bytes, SHA-256
  `6936edac5d0904fc5c4ab175546c5cc5366601fdc1b1e5183a6ea2c10f05d150`

Prepare a new private output directory; the command refuses overwrite and does
not download, load a model, open an audio device, or print transcript text:

```bash
/home/dobo/work/speaker/.venv/bin/python -m tools.prepare_ami_capture_replay \
  --annotations-zip /absolute/ami_public_manual_1.6.2.zip \
  --annotations-sha256 b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d \
  --annotations-bytes 22887865 \
  --audio-wav /absolute/ES2004a.Array1-01.wav \
  --audio-sha256 6936edac5d0904fc5c4ab175546c5cc5366601fdc1b1e5183a6ea2c10f05d150 \
  --audio-bytes 33579394 \
  --output-dir /new/private/ami-capture-replay
```

Run the prepared PCM through the configured Sherpa capture-loop seam into a new
aggregate-only report. Add `nice`/`taskset` locally when protecting other
workloads; `--asr-threads 1` bounds Sherpa's configured ASR threads, not final
verifier memory or total process CPU:

```bash
/home/dobo/work/speaker/.venv/bin/python -m tools.capture_replay_eval \
  --corpus /new/private/ami-capture-replay/capture-replay.json \
  --config /home/dobo/work/speaker/config.json \
  --local-config /home/dobo/work/speaker/config.local.json \
  --asr-threads 1 \
  --repeats 1 \
  --watchdog-seconds 300 \
  --report /new/private/production-capture-report.json
```

`execution_complete=true` means the diagnostic ran; its fixed
`quality_verdict=diagnostic_only` is never an accuracy pass. The public slice
grades far-field transcript/silence and timing lineage only. Human-overlap WER
has ambiguous word ordering, and the turn case does not grade speaker
boundaries. The evaluator also bypasses the native device reader and real
capture mailbox. Its endpoint values use engine-owned timestamps and do not
grade ground-truth VAD onset/offset accuracy.
It has no playback-reference track, target commands, owner voice, or physical
device, so it cannot grade AEC, barge-in, identity, authority, or live quality.
Do not pass `--provider cuda` unless the installed ONNX Runtime exposes
`CUDAExecutionProvider`; the evaluator refuses Sherpa's silent CPU fallback.

## Exact EdAcc endpoint-integrity diagnostic

Run from a clean checkout or task worktree after validating that the exact
retained schema-v2 EdAcc corpus and the pinned Smart Turn v3.2 CPU artifact are
present. Every path must be absolute, the report parent must already be an
owner-only directory outside Git, and the report leaf must not exist:

```bash
SPEAKER_TEST_LOG=0 PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
nice -n 15 ionice -c 3 \
/home/dobo/work/speaker/.venv/bin/python -B -m tools.edacc_endpoint_integrity_eval \
  --corpus /private/edacc-test-v1/corpus.json \
  --config /home/dobo/work/speaker/config.json \
  --local-config /home/dobo/work/speaker/config.local.json \
  --smart-turn-model /private/models/smart-turn-v3.2-cpu.onnx \
  --report /new/private/edacc-endpoint-integrity.json \
  --watchdog-seconds 3600
```

The three CPU cells replay sequentially and may take several minutes because
the capture clock is paced. A success receipt exposes only the report digest,
row count, per-semantic-cell HOLD-row counts, and reset-cell repaired-multi-row
counts against both controls. The report remains an offline counterfactual: it
cannot authorize a threshold, runtime default, device claim, or
live-conversation conclusion (ADR-0147/0148).

The retained clean-revision `e538595` run completed all 24 rows. Control and
HOLD-only candidate both produced 31 finals with 19 single-final and five
multi-final rows; 34 candidate HOLD ticks across five rows repaired zero. The
10,343-byte report SHA-256 is
`9561b6f11ddddeef4cba04caf777f93d0eb9a128769686e2bcecc9b1ef6e04d6`.
That report is the preserved schema-v1 two-cell result. Schema v2 adds the
separate reset-and-accumulate cell and must publish to a new no-overwrite path;
until its exact run completes, it carries no retained-corpus result.

## Exact Parakeet Realtime EOU reference setup

Start with an already-downloaded private 183-wheel closure, the exact local
`.nemo` model, and a prepared schema-v2 streaming corpus. Every output below
must be a new absolute private path:

```bash
export PARAKEET_WHEELHOUSE=/absolute/private/parakeet-wheelhouse
export PARAKEET_MODEL=/absolute/private/parakeet-model
export PARAKEET_LOCK=/absolute/private/new-parakeet-runtime-wheels.lock.json
export PARAKEET_RUNTIME=/absolute/private/new-parakeet-runtime
export PARAKEET_RECEIPT=/absolute/private/new-parakeet-runtime-receipt.json
export PARAKEET_PROVISION=/absolute/private/new-parakeet-candidate
export STREAMING_CORPUS=/absolute/private/prepared-public-corpus/corpus.json
export PARAKEET_BURST_SCRATCH=/absolute/private/new-parakeet-burst-scratch
export PARAKEET_REALTIME_SCRATCH=/absolute/private/new-parakeet-realtime-scratch
export PARAKEET_BURST_REPORT=/absolute/private/new-parakeet-burst-report.json
export PARAKEET_REALTIME_REPORT=/absolute/private/new-parakeet-realtime-report.json
```

Copy the reviewed lock, extract the runtime without executing installer code,
and bind the runtime/model into a new schema-v5 manifest:

```bash
install -m 600 \
  "$PWD/tools/streaming_stt/parakeet-realtime-eou-runtime-wheels.lock.json" \
  "$PARAKEET_LOCK"
/home/dobo/work/speaker/.venv/bin/python -B -m tools.prepare_parakeet_runtime \
  --wheel-lock "$PARAKEET_LOCK" \
  --wheelhouse "$PARAKEET_WHEELHOUSE" \
  --system-python /usr/bin/python3.12 \
  --python-version 3.12.3 \
  --output-root "$PARAKEET_RUNTIME" \
  --receipt "$PARAKEET_RECEIPT"
/home/dobo/work/speaker/.venv/bin/python -B \
  -m tools.provision_parakeet_realtime_eou_candidate \
  --venv-root "$PARAKEET_RUNTIME" \
  --model-root "$PARAKEET_MODEL" \
  --wheelhouse "$PARAKEET_WHEELHOUSE" \
  --output-dir "$PARAKEET_PROVISION"
```

Run one aggregate-only burst and one paced replay. The supervisor independently
places the model worker in its verified 2-CPU/8-GiB/zero-swap scope; keep the
controller one-threaded and low priority:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 nice -n 15 \
  /home/dobo/work/speaker/.venv/bin/python -B -m tools.streaming_stt_eval \
  --worker-manifest "$PARAKEET_PROVISION/worker-manifest.json" \
  --corpus "$STREAMING_CORPUS" --repeats 1 --chunk-samples 1280 \
  --partial-interval-ms 80 --tail-padding-samples 48000 --pace burst \
  --scratch-root "$PARAKEET_BURST_SCRATCH" --output "$PARAKEET_BURST_REPORT"

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 nice -n 15 \
  /home/dobo/work/speaker/.venv/bin/python -B -m tools.streaming_stt_eval \
  --worker-manifest "$PARAKEET_PROVISION/worker-manifest.json" \
  --corpus "$STREAMING_CORPUS" --repeats 1 --chunk-samples 1280 \
  --partial-interval-ms 80 --tail-padding-samples 48000 --pace realtime \
  --scratch-root "$PARAKEET_REALTIME_SCRATCH" \
  --output "$PARAKEET_REALTIME_REPORT"
```

Preparation, provisioning, and evaluation refuse overwrite. They download
nothing and open no audio device. Preserve reports privately. These runs begin
after PCM is available and cannot validate capture, VAD, AEC, turn ground
truth, conversational latency, live audio, or a default change (ADR-0099).

## Exact parakeet.cpp CPU benchmark setup

Start with two existing private mode-0700 roots. The native root must contain
only `source-receipt.json`, `build-receipt.json`, `libparakeet.so`, and
`libspeaker_parakeet_bridge.so`; the model root must contain only
`model-receipt.json` and `realtime_eou_120m-v1-f16.gguf`. Files are
owner-private, single-link regular files matching the schema-v8 receipts. This
route verifies the supplied build; it does not clone, patch, compile, download,
import, execute, or load the candidate during provisioning. Every destination
must be a new absolute private path:

```bash
export PARAKEET_CPP_NATIVE=/absolute/private/parakeet-cpp-native
export PARAKEET_CPP_MODEL=/absolute/private/parakeet-cpp-model
export PARAKEET_CPP_PROVISION=/absolute/private/new-parakeet-cpp-candidate
export STREAMING_CORPUS=/absolute/private/prepared-public-corpus/corpus.json
export PARAKEET_CPP_BURST_SCRATCH=/absolute/private/new-parakeet-cpp-burst-scratch
export PARAKEET_CPP_BURST_REPORT=/absolute/private/new-parakeet-cpp-burst-report.json
export PARAKEET_CPP_PACED_SCRATCH=/absolute/private/new-parakeet-cpp-paced-scratch
export PARAKEET_CPP_PACED_REPORT=/absolute/private/new-parakeet-cpp-paced-report.json
```

```bash
/home/dobo/work/speaker/.venv/bin/python -B \
  -m tools.provision_parakeet_cpp_candidate \
  --python /usr/bin/python3.12 \
  --native-root "$PARAKEET_CPP_NATIVE" \
  --model-root "$PARAKEET_CPP_MODEL" \
  --output-dir "$PARAKEET_CPP_PROVISION"
```

Run the selected strict-fidelity burst cell, then one matched paced repeat.
Keep the controller at nice 15: nice 19 prevents the worker's verified systemd
scope from being created. The explicit tags are part of the report contract:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 \
  /home/dobo/work/speaker/.venv/bin/python -B -m tools.streaming_stt_eval \
  --worker-manifest "$PARAKEET_CPP_PROVISION/worker-manifest.json" \
  --corpus "$STREAMING_CORPUS" --repeats 3 --chunk-samples 1280 \
  --partial-interval-ms 80 --tail-padding-samples 8000 --pace burst \
  --stratum-tag command-negative --stratum-tag command-positive \
  --stratum-tag eccc --stratum-tag gsc --stratum-tag noisy \
  --stratum-tag public-command-noise --stratum-tag silence \
  --stratum-tag speech-negative --scratch-root "$PARAKEET_CPP_BURST_SCRATCH" \
  --output "$PARAKEET_CPP_BURST_REPORT"

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 \
  /home/dobo/work/speaker/.venv/bin/python -B -m tools.streaming_stt_eval \
  --worker-manifest "$PARAKEET_CPP_PROVISION/worker-manifest.json" \
  --corpus "$STREAMING_CORPUS" --repeats 1 --chunk-samples 1280 \
  --partial-interval-ms 80 --tail-padding-samples 8000 --pace realtime \
  --stratum-tag command-negative --stratum-tag command-positive \
  --stratum-tag eccc --stratum-tag gsc --stratum-tag noisy \
  --stratum-tag public-command-noise --stratum-tag silence \
  --stratum-tag speech-negative --scratch-root "$PARAKEET_CPP_PACED_SCRATCH" \
  --output "$PARAKEET_CPP_PACED_REPORT"
```

Protocol v4 accepts only the first `feed` EOU observed after complete source
consumption. It appends text deltas through that EOU document, then freezes
visible text; EOB, finalize EOU, and later events are telemetry. A source-early
first EOU forces complete tail exhaustion. Bubblewrap exposes no network or
NVIDIA nodes, and the supervisor verifies one CPU, 2 GiB memory high, 3 GiB
memory max, zero swap, 64 tasks, and OOM kill before Ready. Preserve aggregate
reports privately. These runs begin after PCM and do not validate capture,
VAD/endpoint ground truth, AEC, barge-in, identity, tools, a device, live
conversation, or a default change (ADR-0119).

## Exact Faster-Whisper final-only benchmark setup

Start with an existing Python 3.12 virtual environment containing the pinned
Faster-Whisper/CTranslate2/CUDA-wheel closure, an already-downloaded local
CTranslate2 model directory, and a prepared schema-v2 streaming corpus. The
runtime `site-packages` root and model root must be owner-private, and every
file must be a materialized single-link regular file; symlink-backed Hugging
Face snapshots and package-manager hard links require separate materialized
copies. The environment must expose only `python3.12` as its `lib/pythonX.Y`
runtime, its interpreter must resolve to `python3.12`, and `pyvenv.cfg` must
contain exactly one recognized interpreter-version entry: either
`version = 3.12.3` or uv's `version_info = 3.12.3`. Both keys, duplicates,
wrong values, and malformed recognized entries are rejected, while the full
marker remains receipt-bound. Use a new absolute private output that neither
contains nor is contained by the virtual environment, `site-packages`, or
model root; the runtime and model roots also cannot overlap. Provisioning
hashes both trees but does not import candidate packages, load a model,
download, or open an audio device. Receipt strictness is intentional and is
not relaxed for local caches:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 \
  /home/dobo/work/speaker/.venv/bin/python -B \
  -m tools.provision_faster_whisper_endpoint_candidate \
  --venv-root /absolute/existing-faster-whisper-venv \
  --model-root /absolute/local-ctranslate2-model \
  --model-id faster-whisper-small-local \
  --language en \
  --output-dir /absolute/private/new-faster-whisper-receipt
```

Run the receipt through the aggregate evaluator. Use `--pace realtime` only
when wall-clock endpoint-to-final replay is needed; burst results are explicitly
accelerated. The adapter emits no partials and does not perform endpoint
detection:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 \
  /home/dobo/work/speaker/.venv/bin/python -B -m tools.streaming_stt_eval \
  --worker-manifest /absolute/private/new-faster-whisper-receipt/worker-manifest.json \
  --corpus /absolute/private/prepared-public-corpus/corpus.json \
  --repeats 1 --chunk-samples 1600 --partial-interval-ms 200 \
  --tail-padding-samples 0 --pace burst \
  --scratch-root /absolute/private/new-faster-whisper-scratch \
  --output /absolute/private/new-faster-whisper-report.json
```

The Bubblewrap worker has no network and mounts the receipt-bound runtime/model
trees read-only. It has one-thread settings but no independently verified hard
RAM or VRAM limit. Preserve reports privately. Results begin after controller
PCM snapshotting and cannot validate capture, VAD, endpoint detection, AEC,
conversation turn-taking, room echo, owner identity, live audio, or a default
change. Streaming deadline/backlog metrics are explicitly null/not applicable;
only complete-PCM-to-final decode latency is meaningful (ADR-0102).

## Exact Nemotron benchmark setup

Start with an already-downloaded private wheelhouse, the official six-file
model directory, and the prepared public corpus. Set only absolute paths, and
choose output paths that do not already exist:

```bash
export NEMOTRON_WHEELHOUSE=/absolute/private/nemotron-wheelhouse
export NEMOTRON_MODEL=/absolute/private/nemotron-model
export NEMOTRON_LOCK=/absolute/private/new-nemotron-runtime-wheels.lock.json
export NEMOTRON_RUNTIME=/absolute/private/new-nemotron-runtime
export NEMOTRON_RECEIPT=/absolute/private/new-nemotron-runtime-receipt.json
export NEMOTRON_PROVISION=/absolute/private/new-nemotron-la6
export STREAMING_CORPUS=/absolute/private/prepared-public-corpus/corpus.json
```

Prepare the exact locked Python 3.12 runtime without importing candidate code:

```bash
install -m 600 \
  "$PWD/tools/streaming_stt/nemotron_runtime_wheels.lock.json" \
  "$NEMOTRON_LOCK"
/home/dobo/work/speaker/.venv/bin/python -B -m tools.prepare_nemotron_runtime \
  --wheel-lock "$NEMOTRON_LOCK" \
  --wheelhouse "$NEMOTRON_WHEELHOUSE" \
  --system-python /usr/bin/python3.12 \
  --output-root "$NEMOTRON_RUNTIME" \
  --receipt "$NEMOTRON_RECEIPT"
```

Bind that runtime and model to one LA6 worker manifest:

```bash
/home/dobo/work/speaker/.venv/bin/python -B -m tools.provision_nemotron_candidate \
  --venv-root "$NEMOTRON_RUNTIME" \
  --model-root "$NEMOTRON_MODEL" \
  --wheelhouse "$NEMOTRON_WHEELHOUSE" \
  --output-dir "$NEMOTRON_PROVISION" \
  --lookahead-tokens 6 \
  --language en-US
```

Run the exact one-case GPU smoke through the isolated worker:

```bash
SPEAKER_NEMOTRON_WORKER_MANIFEST="$NEMOTRON_PROVISION/worker-manifest.json" \
SPEAKER_NEMOTRON_STREAMING_CORPUS="$STREAMING_CORPUS" \
/home/dobo/work/speaker/.venv/bin/python -B -m pytest \
  tests/test_streaming_stt_nemotron_worker.py -m real_model -q
```

The prepare and provision commands emit aggregate JSON receipts only. The
smoke opens no audio device and does not validate capture, VAD, endpointing,
AEC, room echo, perceived latency, or physical barge-in.

## Before commit
1. Run `git diff --check`.
2. Run targeted pytest for changed audio/engine code.
3. Document live A/B validation still required when hardware behavior is affected.
4. Do not commit generated `logs/**` artifacts.

The conversation trace opens no audio device and cannot validate ASR, echo
cancellation, TTS sound, or bare-speaker barge-in. Keep that result separate
from the Sherpa duplex regression and manual live evidence.

The `./live.sh` tests replace PipeWire, Ollama, doctor, and normal core execution
with fakes. Minimal signal-mask cases spawn an inert real subprocess but open no
audio device. They validate resource ownership and evidence setup only; a real
run is still required for current-room audio and barge-in.

The injected Sherpa replay also opens no physical audio device. Per ADR-0064,
its clean echo-impossible profile removes echo/level/word-confirm discrimination
and denoising, uses one eligible 100 ms VAD block, and validates capture
continuity, real ASR/VAD/TTS workers, and interrupt control flow. `--denoise`
is a separate stress diagnostic, not the landing command. Production retains
its physical front end and two-block policy; this does not validate physical
echo, owner enrollment, acoustic stop latency, or the live PipeWire word-cut
route. Any machine-owned injected grade failure or scenario-coverage shortfall
makes the command nonzero.
Inject timeline user times are enqueue metadata, not consumption/overlap evidence;
assistant latency and answers bind through explicit `response_to_user_idx` links.

The recorded owner talk-over replaces input/output streams and device queries,
capability checks, uses the shared inject profile, and binds actual owner-sample
consumption to one metrics token after onset grace plus a floor-only no-cut
control with a complete post-pacing floor-sample window. Its exact corpus/pairs
and one-final-per-clip contract prevent data-driven skips or trailing turns from
shrinking coverage. Its sustained clips override the synthetic one-block minimum
and run production's two-block temporal policy. The concurrent base setup uses a
400 ms floor lead and a 2.4 s acoustic tail so its full pinned window commits;
the separate per-clip replay owns production endpoint grading (ADR-0061).
Owner-to-FIFO stop must stay
within the verifier-owned 1.0 s ceiling. Without
`SPEAKER_REQUIRE_RECORDED=1`, clean clones
self-skip missing private clips/models; that diagnostic command is not a landing
gate. It covers historical waveforms, not the current room, speaker output, v5
enrollment, or live word-cut.

Per ADR-0051/0067/0068, production-warm real-model runs prewarm each distinct model with
the runtime system prompt. `--warm-policy cold` is a labelled red diagnostic.
Generated reports stay local under ignored `logs/conversation-eval/`; an
unverified MiniCPM override still exits 2 and can never make the gate green.
Changing `--runs` or selecting `--scenario` produces a coverage-red diagnostic;
the adoption gate is exactly all fourteen v4 scenarios repeated three times.
A real-model report is provenance-red when the revision/config changes, local
config is included, or any effective model role lacks stable identity evidence.

Per ADR-0065, the autonomous memory probe reopens SQLite and creates a fresh
capability registry in one interpreter; it does not claim a fresh OS process.
A green real-model result requires
`recall_available=true`, `recall_injected=true`, `recall_fenced=true`,
`recent_history_clean=true`, PRIVATE sensitivity, first/only `route=main`,
`controller=false`, and one affirmative clause binding the canary value to its
subject. It also requires one clean stable revision, a stable effective probe
contract, an ambient-credential-isolated loopback transport, and stable full
blob plus effective-config identities for the configured MiniCPM/Gemma roles.
The persisted digest binds the evidence; it does not reconstruct undisclosed
inputs. Echo performs no Git/config/model inspection and reports incomplete. Voice and barge
harnesses retain distinct main/fast arguments; all-role MiniCPM is diagnostic.

Per ADR-0058, a `speaking:` marker or quiet log interval is not sink evidence.
Voice/stress reports match each selected non-barge labelled prompt to a new final
and require remembered same-input-generation playback onset plus the scenario's
aggregate terminal outcome. Auxiliary acknowledgements cannot satisfy a reply;
talk-over requires same-task/generation `interrupted` after its barge marker.
They count only finite nonnegative first-audio values from the finalized bundle
and fail on missing labels, failed injection, late cuts, runtime errors, or stuck
hints. The asynchronous injected-onset clock is causal for the harness but is
not a physical human-onset measurement.

For delay, require PASS on topology, capture, duplex, digest correlation, child
exit, and cleanup. Preserve any retained graph/files and the log on failure; do
not load a host EC module or change desktop defaults to make the retry green.
The synthetic delay command uses deterministic VITS `quiet`, a calibrated
three-block admission fallback, and detector-only quiet padding. Grade its exact
capture-onset clock at ≤1.4 s and show source-onset latency only as a diagnostic.
Recorded-owner, generic, stress, and physical `speaker` paths retain ≤1.0 s.
Require two fresh delay passes before calling the gate stable. See ADR-0069/0070;
keep recorded-owner and physical acceptance separate.

Per ADR-0056, run preparation before any v5 capture. Its final config publish is
already wired to the reserved candidate and its printed command includes
`--require-prepared-enrollment`; a wrong-checkout launch then refuses before the
microphone. Marker-free enrollment refuses a non-empty reference unless the
operator explicitly supplies `--replace-enrollment`.

Per ADR-0066, preparation schema v2 binds full metadata and SHA-256 lineage for
the primary config, reservation, backup, and historical source. Promotion is a
separate no-audio operation after the complete manual live gate passes. The
accepted basename is the prepared candidate basename plus `-accepted`; it stays
adjacent to historical v4 but never replaces it. Exit 3 proves an exact private
orphan plus the unchanged inactive primary pointer, so the identical command may
adopt it. Exit 4 is ambiguous and requires inspection before retry. The stable
advisory lock serializes cooperating promoters only. Never interpret preparation,
staging, or an exit-3 result as live acceptance.

## Known blockers
- If `.venv` is missing, recreate/use a project venv and record the exact command.
- Do not bypass a red delay-route contract by loading an unrelated host EC route
  or changing system audio defaults; preserve the failed evidence for diagnosis.
- Do not claim live audio validation unless it was actually performed.
