# Evaluation and benchmark runbooks — speaker

> **Read this when** you need the exact protected or isolated STT-candidate and
> endpoint diagnostics, the receipt-bound corpus preparation procedures, or the
> semantics of the conversation, autotest and enrollment harnesses. This is not a
> gate list — quick gates and the before-commit checklist live in
> [`agent-testing.md`](agent-testing.md). Results are records in the cited ADRs and `WORKLOG.md`.

- [Protected AMI natural-turn model diagnostic](#protected-ami-natural-turn-model-diagnostic)
- [Exact EdAcc two-model development comparison](#exact-edacc-two-model-development-comparison)
- [Exact causal LiveKit endpoint diagnostic](#exact-causal-livekit-endpoint-diagnostic)
- [Exact Moonshine Medium reference setup](#exact-moonshine-medium-reference-setup)
  - [Stock Moonshine external-boundary diagnostic](#stock-moonshine-external-boundary-diagnostic)
- [Timestamped capture-loop replay](#timestamped-capture-loop-replay)
- [Exact EdAcc endpoint-integrity diagnostic](#exact-edacc-endpoint-integrity-diagnostic)
- [Exact Parakeet Realtime EOU reference setup](#exact-parakeet-realtime-eou-reference-setup)
- [Exact parakeet.cpp CPU benchmark setup](#exact-parakeetcpp-cpu-benchmark-setup)
- [Exact Faster-Whisper final-only benchmark setup](#exact-faster-whisper-final-only-benchmark-setup)
- [Exact Nemotron benchmark setup](#exact-nemotron-benchmark-setup)
- [Harness semantics](#harness-semantics-moved-from-agent-testingmd-before-commit)

## Protected AMI natural-turn model diagnostic

Run this only from a normal approved host login with a working user systemd
manager. Choose an existing exact fixture and a new absent private run root.
The preflight fails rather than lowering the 12-GiB RAM/12-GiB GPU/no-compute
guards, and the transient unit asks systemd to cap the evaluator at one CPU,
4 GiB RAM, 64 tasks, and 35 minutes:

```bash
set -euo pipefail
umask 077

REPO=~/work/speaker
FIXTURE=/absolute/private/ami-natural-turn/fixture-v1
RUN_ROOT=/absolute/private/ami-natural-turn/new-run
REPORT=$RUN_ROOT/report.json

MEM_KIB=$(awk '$1=="MemAvailable:" {print $2}' /proc/meminfo)
GPU_MIB=$(nvidia-smi -i 0 --query-gpu=memory.free \
  --format=csv,noheader,nounits | tr -d '[:space:]')
GPU_PIDS=$(nvidia-smi --query-compute-apps=pid \
  --format=csv,noheader,nounits | tr -d '[:space:]')
[ "$MEM_KIB" -ge 12582912 ]
[ "$GPU_MIB" -ge 12288 ]
[ -z "$GPU_PIDS" ]

install -d -m 0700 "$RUN_ROOT"
[ "$(stat -c '%u:%a' "$RUN_ROOT")" = "$(id -u):700" ]
test ! -e "$REPORT"
test ! -L "$REPORT"

systemd-run --user --wait --pipe --collect --quiet \
  --unit=speaker-ami-natural-turn-gpu-v1 \
  --working-directory="$REPO" --nice=19 \
  -p CPUAccounting=yes -p CPUQuota=100% -p CPUWeight=1 \
  -p MemoryAccounting=yes -p MemoryHigh=3G -p MemoryMax=4G \
  -p MemorySwapMax=0 -p TasksMax=64 -p KillMode=control-group \
  -p OOMPolicy=stop -p RuntimeMaxSec=35min -p TimeoutStopSec=10s \
  -p IOSchedulingClass=idle -p UMask=0077 \
  -E SPEAKER_TEST_LOG=0 -E PYTHONDONTWRITEBYTECODE=1 \
  -E OMP_NUM_THREADS=1 -E OPENBLAS_NUM_THREADS=1 \
  -E MKL_NUM_THREADS=1 -E NUMEXPR_NUM_THREADS=1 \
  -E OMP_WAIT_POLICY=PASSIVE -E MALLOC_ARENA_MAX=2 \
  -E HF_HUB_OFFLINE=1 -E TRANSFORMERS_OFFLINE=1 \
  -E CUDA_VISIBLE_DEVICES=0 -E CUDA_MODULE_LOADING=LAZY \
  "$REPO/.venv/bin/python" -B \
  -m tools.ami_natural_turn_capture_replay_eval \
  --fixture-dir "$FIXTURE" \
  --config "$REPO/config.json" \
  --local-config "$REPO/config.local.json" \
  --device desktop_gpu_4090 --provider cpu --asr-threads 1 \
  --watchdog-seconds 1800 --report "$REPORT"
```

The evaluator's retained report attests its own closed aggregate schema and
model execution, not the host cgroup. Record the launch command outcome
separately, and do not infer GPU usage from this evaluator while its aggregate
`peak_vram_mb` remains null.

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
nice -n 15 ionice -c 3 ~/work/speaker/.venv/bin/python -B \
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
export SMART_TURN_MODEL=~/work/speaker/pretrained_models/sherpa/turn/smart-turn-v3.2-cpu.onnx
export CAUSAL_SCRATCH=/absolute/private/new-livekit-causal-scratch
export CAUSAL_REPORT=/absolute/private/new-livekit-causal-report.json

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 \
  ~/work/speaker/.venv/bin/python -B \
  -m tools.livekit_causal_endpoint_eval \
  --source-parquet "$LIVEKIT_EOT_PARQUET" \
  --inventory-report "$LIVEKIT_EOT_INVENTORY" \
  --parquet-python "$PYARROW25_PYTHON" \
  --model "$SMART_TURN_MODEL" \
  --config ~/work/speaker/config.json \
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
  ~/work/speaker/.venv/bin/python -B \
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
  ~/work/speaker/.venv/bin/python -B -m pytest \
  tests/test_streaming_stt_moonshine_worker.py -m real_model -q

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 \
  ~/work/speaker/.venv/bin/python -B -m tools.streaming_stt_eval \
  --worker-manifest "$MOONSHINE_PROVISION/worker-manifest.json" \
  --corpus "$STREAMING_CORPUS" --repeats 3 --chunk-samples 1600 \
  --partial-interval-ms 500 --tail-padding-samples 0 --pace burst \
  --scratch-root /absolute/private/new-moonshine-burst-scratch \
  --output /absolute/private/new-moonshine-burst-report.json

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 \
  ~/work/speaker/.venv/bin/python -B -m tools.streaming_stt_eval \
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
  ~/work/speaker/.venv/bin/python -B \
  -m tools.provision_moonshine_candidate \
  --venv-root "$MOONSHINE_RUNTIME" --wheel "$MOONSHINE_WHEEL" \
  --model-root "$MOONSHINE_MODEL" --model-arch medium-streaming \
  --segmentation-mode external-presegmented \
  --output-dir "$MOONSHINE_EXTERNAL_PROVISION"

SPEAKER_MOONSHINE_SCHEMA7_WORKER_MANIFEST="$MOONSHINE_EXTERNAL_PROVISION/worker-manifest.json" \
SPEAKER_MOONSHINE_STREAMING_CORPUS="$STREAMING_CORPUS" \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 \
  ~/work/speaker/.venv/bin/python -B -m pytest \
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
  ~/work/speaker/.venv/bin/python -B -m tools.streaming_stt_eval \
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
~/work/speaker/.venv/bin/python -m tools.prepare_ami_capture_replay \
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
~/work/speaker/.venv/bin/python -m tools.capture_replay_eval \
  --corpus /new/private/ami-capture-replay/capture-replay.json \
  --config ~/work/speaker/config.json \
  --local-config ~/work/speaker/config.local.json \
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
~/work/speaker/.venv/bin/python -B -m tools.edacc_endpoint_integrity_eval \
  --corpus /private/edacc-test-v1/corpus.json \
  --config ~/work/speaker/config.json \
  --local-config ~/work/speaker/config.local.json \
  --smart-turn-model /private/models/smart-turn-v3.2-cpu.onnx \
  --logical-turn-shadow \
  --logical-turn-terminal-lineage \
  --logical-turn-async-terminal-delivery \
  --report /new/private/edacc-endpoint-integrity.json \
  --watchdog-seconds 7200
```

The three CPU cells replay sequentially and may take several minutes because
the capture clock is paced. Omit all three logical-turn flags to reproduce the
unchanged schema-2 report. Raw shadow alone selects schema 3; adding
`--logical-turn-terminal-lineage` requires that raw flag, selects schema 4, and
adds strict aggregate-only inline selected-final/typed-abort correlation to
every cell. Adding `--logical-turn-async-terminal-delivery` requires both prior
flags and selects schema/kind v5. Capture still finishes first; one persistent
evaluator thread then drains each fresh stage through the real final worker,
with exact accepted-item/callback correlation and separately counted prequeue
aborts. A success receipt additionally binds the exact schema and three mode
booleans before the guarded parent reopens the retained report. This remains an
offline counterfactual: it cannot establish simultaneous capture/finalizer
concurrency, selected-text parity, overflow, production stop/shutdown,
dispatcher, supervisor, runtime/default, device, latency, authority, quality,
or live-conversation behavior (ADR-0147/0148/0153/0155/0156).

The retained clean-revision `e538595` run completed all 24 rows. Control and
HOLD-only candidate both produced 31 finals with 19 single-final and five
multi-final rows; 34 candidate HOLD ticks across five rows repaired zero. The
10,343-byte report SHA-256 is
`9561b6f11ddddeef4cba04caf777f93d0eb9a128769686e2bcecc9b1ef6e04d6`.
That report is the preserved schema-v1 two-cell result. Schema v2 adds the
separate reset-and-accumulate cell. Its exact clean-source `63e25e8` run also
completed all 24 rows in all three cells. Reset changed the common 19
single/five multi row shape to 20/four, repairing one multi-final row against
each control with no single-to-multi regression. It produced six HOLD ticks
across five rows, WER/CER 0.4874/0.3237, and endpoint-delay p50/p95 900/1,700
ms; acoustic was 0.5779/0.3919 and 800/1,200 ms, while HOLD-only was
0.5729/0.3896 and 800/1,700 ms. All cells retained four exact rows. Preserve
the mode-0600, single-link, 14,995-byte schema-v2 report SHA-256
`98ea171dfcd9ad553604992a6e4bfb450c8340af5be57dbcb89c24cce142f5f3`.
This partial repair does not promote the option; owner bare-speaker A/B remains
required.

The opt-in schema-3 run at clean source `92c2456` closed all 24 observers in
each cell with zero health errors, mismatches, or extra/missing raw terminals.
Acoustic and HOLD-only each matched 35 raw commits and correctly remained
inconclusive without multi-epoch evidence. Reset matched 32 raw commits,
including five exact multi-epoch commits from six held native finals and five
empty successor epochs. Downstream callbacks recorded 31/31/29 selected finals
and, separately, 4/4/3 typed input-rejection aborts. Cell-level totals equal
35/35/32 raw commits, but no per-terminal selected-final or abort lineage is
established. Peak process RSS was 864.781 MiB. Preserve the mode-0600,
single-link, 18,719-byte report SHA-256
`bcacaece6b1620520d3131054d3faef511fa5ecfe5d4a2b789a9dca5a72c6b8d`.
No runtime, default, latency, device, or live claim follows (ADR-0153).

The opt-in schema-4 run at clean source `dac50c9` bound all 35/35/32 raw
commits to 31/31/29 inline selected finals plus 4/4/3 typed input-rejection
aborts. All three cells were exact, every raw commit was acoustically bound,
and binding, terminal, late-observation, and internal-error counters were zero.
The guarded pre/post binding covered 19 source files. Preserve the mode-0600,
single-link, 23,350-byte report SHA-256
`1c03d64295fe9443d057c854b034fbc73723420641088f918f242a79cb1f5230`.
It proves transcript-free inline terminal identity and typed outcome only;
selected-text, async worker, dispatcher, supervisor, production authority,
device, latency, and live parity remain false (ADR-0155).

The opt-in schema-5 run at clean source `ab4f4eb` retained the same exact
35/35/32 raw and inline terminal lineage. Its post-capture worker accepted,
started, finished, and selected all 31/31/29 final items; all 24 rows per cell
observed a clean empty-stage drain on the persistent evaluator thread distinct
from capture. The 4/4/3 input-rejection aborts remained capture-side prequeue
outcomes, no selected final ran inline, and every async health, release,
cancellation, overflow, and unfinished counter was zero. The guarded source
closure covered 21 files. Preserve the mode-0600, single-link, 29,952-byte
report SHA-256
`b3c749fcbdc867c179fbac77717552147de65240c1e815e6fcaef6d12b9a0852`.
Peak process RSS was 1,269.109 MiB. This proves only deterministic accepted
handoff to post-capture dedicated-worker callback delivery; simultaneous live
capture/finalizer concurrency, selected-text parity, stop/shutdown, overflow,
dispatcher, supervisor, runtime/defaults, device, latency, authority, and STT
quality remain unproven (ADR-0156).

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
~/work/speaker/.venv/bin/python -B -m tools.prepare_parakeet_runtime \
  --wheel-lock "$PARAKEET_LOCK" \
  --wheelhouse "$PARAKEET_WHEELHOUSE" \
  --system-python /usr/bin/python3.12 \
  --python-version 3.12.3 \
  --output-root "$PARAKEET_RUNTIME" \
  --receipt "$PARAKEET_RECEIPT"
~/work/speaker/.venv/bin/python -B \
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
  ~/work/speaker/.venv/bin/python -B -m tools.streaming_stt_eval \
  --worker-manifest "$PARAKEET_PROVISION/worker-manifest.json" \
  --corpus "$STREAMING_CORPUS" --repeats 1 --chunk-samples 1280 \
  --partial-interval-ms 80 --tail-padding-samples 48000 --pace burst \
  --scratch-root "$PARAKEET_BURST_SCRATCH" --output "$PARAKEET_BURST_REPORT"

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 nice -n 15 \
  ~/work/speaker/.venv/bin/python -B -m tools.streaming_stt_eval \
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
~/work/speaker/.venv/bin/python -B \
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
  ~/work/speaker/.venv/bin/python -B -m tools.streaming_stt_eval \
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
  ~/work/speaker/.venv/bin/python -B -m tools.streaming_stt_eval \
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
  ~/work/speaker/.venv/bin/python -B \
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
  ~/work/speaker/.venv/bin/python -B -m tools.streaming_stt_eval \
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

The exact 2026-08-06 CUDA/FP16 run at `a46942c` used the freshly materialized
24-case AMI close/far corpus for three repeats and completed 72/72 evaluations
with tags `ami`, `close`, `far`, `isolated`, and `turn_transition` and zero
disagreement. Overall/close/far WER was `.3797`/`.2025`/`.5570`;
complete-PCM finalization p50/p95 was `28.562`/`80.935` ms and the final-only
adapter emitted no partials. Preserve the private 16,062-byte mode-0600,
single-link report with SHA-256
`06e51d89536e19d7c2e232d7c11e9857c3bb95eab535cec79b2361ac535dfc47`.
ADR-0151 records the evidence boundary and next gate.

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
~/work/speaker/.venv/bin/python -B -m tools.prepare_nemotron_runtime \
  --wheel-lock "$NEMOTRON_LOCK" \
  --wheelhouse "$NEMOTRON_WHEELHOUSE" \
  --system-python /usr/bin/python3.12 \
  --output-root "$NEMOTRON_RUNTIME" \
  --receipt "$NEMOTRON_RECEIPT"
```

Bind that runtime and model to one LA6 worker manifest:

```bash
~/work/speaker/.venv/bin/python -B -m tools.provision_nemotron_candidate \
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
~/work/speaker/.venv/bin/python -B -m pytest \
  tests/test_streaming_stt_nemotron_worker.py -m real_model -q
```

The prepare and provision commands emit aggregate JSON receipts only. The
smoke opens no audio device and does not validate capture, VAD, endpointing,
AEC, room echo, perceived latency, or physical barge-in.

## Harness semantics (moved from agent-testing.md "Before commit")

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
