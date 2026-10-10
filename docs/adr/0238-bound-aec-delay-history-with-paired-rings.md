# ADR-0238: Bound AEC delay history with paired rolling buffers

Date: 2026-10-10
Status: accepted

## Decision

Store AecDelayCalibrator's paired microphone/reference history in fixed float32
rings. Copy only the newest retained samples on observe, with wrap and oversized
inputs bounded by the configured window. Materialize independent chronological
snapshots only when the existing estimator or explicit inspection needs them.
Reuse/clear storage on reset. Preserve the complete estimator, clamp, recalc,
energy clock, acceptance, median, seed/operating-delay and continuity contracts.
Keep the existing Python/NumPy implementation and dependencies.

## Context / why

The previous concatenate-and-tail implementation copied both entire rolling
windows on every capture block, including silence. A tail view also retained the
whole concatenation base, so a large incoming block could retain much more
storage than the configured history length. The normal 16-kHz/100-ms/full-window
case copied 204,800 bytes for rolling history per observation; fixed rings copy
12,800 bytes. Input capture bounds still apply independently; no new native input
admission, dropped data, sample-rate conversion or echo-delay default is added.

The primary [whisper.cpp audio_async implementation](https://github.com/ggml-org/whisper.cpp/blob/master/examples/common-sdl.cpp)
uses fixed-duration storage, wrap copies and chronological window materialization
on read. Its [release record](https://github.com/ggml-org/whisper.cpp/releases/tag/v1.9.5)
publishes v1.9.5 on 2026-10-06; the buffering source pattern was reviewed on
2026-10-10, not imported or executed. [sounddevice's callback guidance](https://python-sounddevice.readthedocs.io/en/0.5.3/api/streams.html)
explains why allocation and unpredictable work deserve special care around audio
deadlines. This calibrator runs on the capture processor rather than PortAudio's
output callback; the change adds no hard real-time claim. [Pipecat's output transport](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/transports/base_output.py)
and [RealtimeTTS's stream player](https://github.com/KoljaB/RealtimeTTS/blob/master/RealtimeTTS/stream_player.py)
also make buffering, sample counts and boundary handling explicit. Existing
Speaker FIFO/receipt/cleanup ownership remains unchanged.

## Consequences

The [compact synthetic receipt](../evidence/aec-delay-window-ring-2026-10-10.json)
binds baseline Git commit 70345ae and both class hashes. Thirty-six exact byte-window
cases cover tiny windows, fill/wrap, empty/mismatched and oversized chunks. The
unchanged estimator/recalc/clamp/snapshot AST and a four-second synthetic echo
stream retain the exact 384-sample result, snapshot and median/reset outcomes.
The new tests additionally cover source/snapshot ownership, nonfinite/strided
bytes, reset reuse, energy timing and bounded oversized-feed retention.

Adjacent alternating, unpaced pairs of 100-ms PCM blocks measured baseline p50
21.965–27.806 us and p95 30.282–32.721 us; ring p50 15.022–19.273 us and p95
20.334–22.206 us (about 1.45x for this maintenance call). Absolute timings varied
with host load, so these are paired synthetic kernel measurements. Four BLAS/OpenMP
thread environment requests were one; CPU isolation or physical throughput was
not established. Original prototype/baseline metrics remain separate, with only
one verified current benchmark setup and compact current result retained.

After warm-up, incremental traced loop peak fell 308,072 to 7,704 bytes. Including
storage allocations, idle traced peak was 309,312 to 201,056 bytes and the energetic
recalculation cell was 496,180 to 483,156 bytes. Steady full-window storage was
204,800 to 192,000 bytes. These are Python-traced/sample-storage measures, not
process RSS. Fixed buffers reserve 192,000 bytes at construction even while empty;
the former implementation grew to that history over its first 1.5 seconds. Empty
snapshots expose no uninitialized samples, and resets clear retained buffers.

The current mathematical estimator still dominates its recalculation cells; this
change does not accelerate its correlation sweep or native ASR/TTS/LLM inference.
It applies only when AEC and automatic delay calibration are active. No voice,
model, context/output cap, speaker authority, barge threshold, delay bound,
configured seed, measured-delay policy or platform entrypoint changes. There is
no Rust/native ABI or library/runtime upgrade.

Focused capture/playback/DSP/calibration regression: **467 passed, 1 optional-model
skip** in 10.11 seconds. Scoped Ruff and whitespace/docs checks pass. Two existing
reset fixtures now seed history through observe rather than assigning private
window arrays; their original reset/continuity assertions are retained. No native
model, microphone, device, private recording, doctor, model download or live run
occurred. Windows/macOS drivers, acoustic barge-in and physical latency remain
owner-authorized qualification gates; the protected live candidate is untouched.

Reproduce with one compact fresh output path:

```sh
SPEAKER_TEST_LOG=0 SPEAKER_NO_LOCAL_CONFIG=1 SPEAKER_LIVE=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 ionice -c 3 nice -n 19 /home/dobo/work/speaker/.venv/bin/python -B -m tools.bench_aec_delay_windows --output <fresh-json>
```
