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
| Local Sherpa streaming-decode owner | `SPEAKER_TEST_LOG=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_sherpa_streaming_decode_session.py tests/test_asr_postprocess.py tests/test_barge_confirm.py tests/test_barge_word_cut.py tests/test_sherpa_media_session.py tests/test_sherpa_vad_final_gate.py tests/test_virtual_audio_engine.py tests/test_capture_replay.py tests/test_capture_replay_eval.py tests/test_sherpa_duplex_runtime.py -q` | exact-thread owner, opaque stream lifecycle, owner-only native retirement, hotword compatibility, capture gap/reset ordering, word-cut/final handoff, replay provenance, and adjacent callbacks pass without models or audio devices (ADR-0110) |
| Trusted LiveKit publisher session | `/home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_livekit_agents_engine.py tests/test_livekit_agents_session.py -q` | `73 passed`: exact `livekit==1.1.14`/Agents 1.6.7 validation, manual RoomIO without implicit control/record/preconnect paths, pre-ready inbound-event suppression, console/start/ready/close races, all-reason publisher-disconnect fencing, route leases, and retryable disposal with a fake SDK; no installed SDK, network, audio device, or live validity (ADR-0104) |
| Bounded post-ASR mailbox | `/home/dobo/work/speaker/.venv/bin/python -B -m pytest tests/test_event_bus.py tests/test_always_on_agent.py tests/test_playback_receipts.py tests/test_final_preprocessing_cancel.py tests/test_never_stuck.py -q` | bounded/coalesced data, preemptive lifecycle service, aged ordered-output service, supervisor/playback races, and shutdown contracts pass headlessly (ADR-0093) |
| Conversation trace (device-free) | `/home/dobo/work/speaker/.venv/bin/python -m tools.conversation_eval --runs 3` | all 14 v4 scenarios pass 3/3; aggregate 42/42, `pass^3=True`, exact declared answer routes, and controller-owned bounded correction (ADR-0051/0067/0068) |
| MiniCPM Q8 production-hybrid A/B | `/home/dobo/work/speaker/.venv/bin/python -m tools.conversation_eval --mode ollama --candidate-model minicpm5-1b:q8 --baseline-model gemma3:12b --runs 3` | clean revision; candidate and baseline each 42/42; verified identities, warm budgets, exact answer routes, and no regression (ADR-0051/0067/0068) |
| MiniCPM all-role stress | add `--topology all-roles` to the prior command | ADR-0051 replacement-stress diagnostic; never an adoption result |
| Cross-session semantic memory | `/home/dobo/work/speaker/.venv/bin/python -m tools.autotest memory` | SQLite reopen; clean native history; balanced fenced injection; PRIVATE first/only `route=main`; clause-grounded canary; clean/stable revision, contract, and full shipped-role identities (Echo is incomplete) (ADR-0065) |
| Autonomous cable diagnostic | `/home/dobo/work/speaker/.venv/bin/python -m tools.autotest voice --acoustics cable` | pipeline checks green but report says `diagnostic_pass`, `ok=false`, and barge/self/latency `not_covered`; not a landing result (ADR-0055/0058) |
| Autonomous silent voice gate | `/home/dobo/work/speaker/.venv/bin/python -m tools.autotest all --acoustics delay` | exact run-owned virtual EC topology and capture/playback bindings; labelled causal finals/terminals; mean WER ≤0.50; zero errors/stuck/self-cuts; synthesized exact-command cut from engine capture onset in 0–1.4 s; all six route/exit/cleanup proofs; aggregate PASS (ADR-0058/0069/0070) |
| Injected Sherpa barge replay | `/home/dobo/work/speaker/.venv/bin/python -m tools.live_session --scenario barge_in_interrupt_stop --repeat 3 --inject --barge-in --llm echo --no-assistant-audio` | exit 0 only when every repetition is full-duplex `ok`, intended FIFO cuts 2/2, zero self-interrupts (ADR-0064) |
| Recorded owner-voice landing gate | `SPEAKER_REQUIRE_RECORDED=1 /home/dobo/work/speaker/.venv/bin/python -m pytest tests/replay_recorded_voice_test.py -q` | reference host: exactly 9 passed/0 skipped (six utterances, one same-session multi-turn, two causal fake-stream owner talk-overs); missing private clips/models fail (ADR-0053) |
| Recording-driven STT accuracy | `/home/dobo/work/speaker/.venv/bin/python -m tools.recorded_stt_eval` | aggregate-only streaming/offline/selected WER+CER over every hash-pinned labelled clip; each selected offline recognizer/verifier must complete a decode with zero error outcomes; no runtime, TTS, tools, network, or audio device (ADR-0078/0080) |
| Exact private final-input evidence | `SPEAKER_TEST_LOG=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_diagnostic_bundle.py tests/test_reference_recording.py tests/test_sherpa_diagnostic_observations.py tests/test_asr_final_async.py tests/test_prepare_diagnostic_streaming_stt_corpus.py tests/test_streaming_stt_corpus_writer.py tests/test_streaming_stt_eval.py tests/test_app_replay_safety.py -q` | diagnostic schema-v1 read-only validation, schema-v2 exact f32le binding/lifecycle/tamper/backpressure, private schema-v3 bit-for-bit export, public schema-v2 separation, aggregate-only CLI privacy, and synchronized-directory replay refusal pass headlessly; no owner corpus, model, GPU, audio device, WER, latency, or live claim (ADR-0108) |
| Capture-loop replay contracts | `/home/dobo/work/speaker/.venv/bin/python -B -m pytest tests/test_capture_replay_corpus.py tests/test_capture_replay.py tests/test_capture_replay_metrics.py tests/test_capture_replay_eval.py tests/test_prepare_ami_capture_replay.py -q` | strict corpus, paced Sherpa seam, fixed aggregate conditions/privacy, evaluator, and deterministic AMI preparation pass without a model or audio device (ADR-0092) |
| Isolated streaming-STT harness | `/home/dobo/work/speaker/.venv/bin/python -m pytest tests/test_prepare_public_streaming_stt_corpus.py tests/test_prepare_nemotron_runtime.py tests/test_prepare_parakeet_runtime.py tests/test_provision_moonshine_candidate.py tests/test_provision_nemotron_candidate.py tests/test_provision_parakeet_realtime_eou_candidate.py tests/test_streaming_stt_*.py -m "not real_model" -q` | fake, Moonshine, Nemotron, and Parakeet private-source/exact-runtime, bounded protocol, sandbox, provenance, aggregate/privacy, and teardown contracts pass (ADR-0089/0090/0091/0099) |
| Faster-Whisper final-only comparator | `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_provision_faster_whisper_endpoint_candidate.py tests/test_streaming_stt_faster_whisper_endpoint.py -q` | separate runtime/model receipts, no-download provision, final-only adapter, evaluator provenance, PCM/tree tamper rejection, and schema-v6 worker serialization pass without importing/loading a candidate, GPU, network, or audio device (ADR-0102) |
| Public matrix + production-model STT control | `SPEAKER_TEST_LOG=0 /home/dobo/work/speaker/.venv/bin/python -B -m pytest tests/test_public_voice_eval_matrix.py tests/test_streaming_stt_zipformer.py -m "not real_model" -q` | no-download license/receipt/selection/privacy contracts and exact fp32 Zipformer Bubblewrap control pass without a corpus, model load, network, or audio device (ADR-0098) |
| Licensed stratified STT evidence | `SPEAKER_TEST_LOG=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_prepare_demand_noise_streaming_stt_corpus.py tests/test_public_voice_eval_matrix.py tests/test_streaming_stt_corpus_writer.py tests/test_streaming_stt_eval.py tests/test_streaming_stt_faster_whisper_endpoint.py tests/test_streaming_stt_suite.py -q` | exact DEMAND/license/runtime receipts, outside-Git publication, Rochester label fail-closed behavior, bounded strata, N/A zero-attempt command recall, evaluator binding, sequential cells, failure checkpoints, and aggregate privacy pass without a model, network, GPU, or audio device (ADR-0109) |
| Common Voice spontaneous P0 preparation | `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_prepare_common_voice_spontaneous_v4.py tests/test_streaming_stt_corpus_writer.py tests/test_public_voice_eval_matrix.py -q` | exact release/API receipt, twice-hashed private archive, safe tar/TSV, speaker-first quota matching, bounded publication/provenance, and synthetic decode pass; the two real-PyAV checks run only when the pinned optional `requirements-evaluation.txt` environment is installed. No corpus download, model, network, GPU, or audio device (ADR-0101) |
| VoxPopuli Romanian-accented English P0 preparation | `SPEAKER_TEST_LOG=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_prepare_voxpopuli_ro_accent.py tests/test_streaming_stt_corpus_writer.py -q` | exact two-shard/revision/license pins, explicit-local-only descriptor transport, exact float32-WAV/fact/empty-row contracts, finite identity conversion, deterministic 24-speaker four-band assignment, aggregate privacy, and hardened schema-v2 publication pass synthetically. The first real attempt failed closed; no real corpus was published and no model, network, GPU, or audio device runs here (ADR-0106) |
| Exact Moonshine worker smoke | set absolute `SPEAKER_MOONSHINE_WORKER_MANIFEST` and `SPEAKER_MOONSHINE_STREAMING_CORPUS`, then run `/home/dobo/work/speaker/.venv/bin/python -m pytest tests/test_streaming_stt_moonshine_worker.py -m real_model -q` | one exact pre-provisioned case runs; no install, download, network, audio device, or adoption claim (ADR-0090) |
| Exact Nemotron worker smoke | prepare and provision new private paths as shown below, then run the candidate-specific `real_model` command | one exact pre-provisioned case passes through no-network Bubblewrap and the bound inputs still pass close-time rehash; no audio device or adoption claim (ADR-0091) |
| Opt-in selected final STT setup | `./install.sh --skip-system --final-asr parakeet-unified-en --final-verifier faster-whisper-small` | verify/stage the four-file Parakeet package and pinned Linux/NVIDIA verifier, publish atomically, and pass doctor; normal sessions still use `./live.sh` (ADR-0080) |
| Verifier recording A/B | `/home/dobo/work/speaker/.venv/bin/python -m tools.recorded_stt_eval --set asr_final_verifier_backend=faster_whisper --set asr_final_verifier_model=/home/dobo/work/speaker/pretrained_models/sherpa/faster_whisper_small-536b0662742c --keyword vault` | aggregate-only baseline/candidate comparison, verifier outcome counts, and artifact-bound provenance; no transcript rows or audio device (ADR-0078/0080) |
| Isolated enrollment prep | `/home/dobo/work/speaker/.venv/bin/python -m tools.prepare_enrollment --help` then supply the four explicit absolute paths and a unique `enrollment.v5-<id>.json` | device-free; verified no-clobber backup, empty feature candidate, regular mode-600 config with prepared marker; use its exact printed next command (ADR-0056) |
| Accepted v5 promotion | `/home/dobo/work/speaker/.venv/bin/python -m tools.promote_enrollment --help` then supply the exact worktree, primary config, prepared candidate/source/backup, candidate-derived adjacent accepted path, and `--accept-live-gate` | device-free and only after manual acceptance; exit 0 = active, 2 = refused, 3 = confirmed staged/inactive, 4 = ambiguous (ADR-0066) |
| Whitespace | `git diff --check` | no output |

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
