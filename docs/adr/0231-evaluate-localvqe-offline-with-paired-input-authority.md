# ADR-0231: Evaluate LocalVQE offline with explicit paired-input authority

Date: 2026-10-10
Status: accepted

## Decision

Accept two offline evaluation tools and their bounded, separately sourced
evidence. Keep the pinned LocalVQE echo-only 203K candidate and 2.7K filter-only
control outside the runtime. Compare them with the shipped APM wrapper with NS
off/on and the default FDAF/NLMS implementation, with explicit configurations
and provenance. Change no capture route, DSP default, calibration, enrollment,
admission, speaker authority, tool authority, or physical qualification.

## Context / why

The reference-ring repair in ADR-0230 removes a measured defect and unnecessary
work. A further native rewrite needs evidence of useful acoustic behavior and
resource cost. LocalVQE supplies a small released C++/GGML implementation with
a C API, an explicit CPU-thread setting, reset, and 256-sample streaming hops;
its echo-only scope avoids assuming that noise suppression preserves speech.
The public source/model/build identities and licenses are recorded in
`tools/audio_eval/localvqe-v14.lock.json`. Both weights also appear in the
publisher's built-in model hash allowlist. The unhashed-model bypass stays off.

Primary sources: [pinned API](https://github.com/localai-org/LocalVQE/blob/f53063c9eb2a85f96479867d1dd911dc3bf6319b/ggml/localvqe_api.h),
[model allowlist](https://github.com/localai-org/LocalVQE/blob/f53063c9eb2a85f96479867d1dd911dc3bf6319b/ggml/model_hash.cpp),
[Apache-2.0 source license](https://github.com/localai-org/LocalVQE/blob/f53063c9eb2a85f96479867d1dd911dc3bf6319b/LICENSE),
and [pinned weight repository](https://huggingface.co/LocalAI-io/LocalVQE/tree/29ca38495cba9d6393a92a4dd890f28dd81f758d).
The Git submodule is pinned, and the publisher's tracked GRU patch is applied
by its CMake configuration. The Linux CPU-only shared-library build uses
GNU C++ 13.3.0, isolated CMake 3.31.6, OpenMP off, two build jobs and a 300-second
wall limit, with monitored 768 MiB/50,000-file scratch caps. The public native
bundle is 23 flat library/alias files totaling 18,531,296 bytes; no source or
binary is installed into the shared Python environment or committed to Git.

**Actual APM baseline: LiveKit 1.1.10.** That is the distribution present on this
host, whose actual Python/native files were hashed before/after each cell. It
differs from the 1.1.14 closure pin in STATUS/ADR-0163. These results do not
qualify that pinned closure. The environment and dependency pins are unchanged.

## Tool and input contracts

`tools/localvqe_offline_eval.py` generates five deterministic, 8-second harmonic
and modulated-signal cases: far-only linear echo, nonlinear echo, imposed delay
drift, near-only, and double talk after a fixed far-only prefix. These are not
recorded speech. Every variant receives the same 16 kHz float32 inputs, fresh
state per case, and 1,280-sample callbacks. Fixed sample geometry makes terminal
padding unnecessary. Scoring uses the last four seconds, without asserting
that every model has converged. Near projection/SI-SDR are zero-lag and
uncompensated: processing delay, filter phase and out-of-distribution harmonic
behavior affect them. They are not speech-preservation or ASR scores. Exact
muting has an explicit flag and undefined SI-SDR, rather than a false quality win.

`tools/localvqe_recorded_eval.py` requires a caller-supplied exact manifest hash,
complete schema-2 bundle validation and all artifact hashes. It then verifies
equal pre-gain/reference ranges for every original capture frame, one continuous
source with no gaps/jumps, bounded PCM16 mono 16 kHz geometry, and a total
sample count divisible by every compared framer. It preserves original capture
calls; LocalVQE alone carries fewer than 256 pending samples between calls.
It adds no reference alignment, interpolation, terminal padding or wave copies.
Unequal packed track lengths cannot be silently flattened into a valid pair.

The protected owner bundle used here has 1,416 consecutive 1,600-sample frames,
2,265,600 paired samples (141.6 seconds), one source and no gaps. The bundle is
**AFTER-HOST**: its pre-gain tap includes application resampling and may already
include host echo processing. Native host PCM, physical raw microphone audio
and physical playback-audibility evidence are absent. The original producer
records its zero-delay reader reference snapshot; this replay does not
reconstruct its calibrator or reproduce a calibrated/physical AEC route. The
gate/ASR tracks are shorter and are not substituted for the pre-gain track.
The owner described the session as room noise; no isolated echo/near-end labels
exist. Only execution, input/output energy and numerical stability are reported.
No transcript, text log, ASR, VAD, endpoint, speaker model, device or tool ran.

Both tools isolate each variant in a separate process, suppress native OS output,
reject unexpected/nonfinite/non-scalar result fields, revalidate code/model/native
assets and input hashes, and retain failed cells. LocalVQE requests one native
CPU thread; Linux caller/all-thread masks are sampled after loads and each
processing callback. Sampling is not continuous or cgroup isolation. Python
socket hooks deny network; an OS network sandbox is not attested. APM has no
explicit thread-pool or native-close API in this wrapper, and its destruction is
not attested; process exit bounds the evaluation. Linux workers apply CPU/AS/core
limits and wall deadlines (45 seconds synthetic; 120 seconds recorded). Other
platforms report unavailable resource/affinity probes instead of pretending
enforcement. DLL/dylib loading has fake controls, but only Linux binaries were
built and measured. No Windows/macOS hardware, phone, thermal or live proof follows.

## Observed receipts

All five synthetic cells completed. On this Intel Core i9-13980HX with CPython
3.12.3/NumPy 2.4.6, low-priority scheduling and one-CPU affinity, processing RTF
ranges across the five generated cases were:

| Variant | Synthetic processing RTF range | Whole-child peak RSS |
| --- | ---: | ---: |
| LocalVQE 203K | .352–.404 | 61.9 MiB |
| LocalVQE 2.7K | .070–.091 | 60.9 MiB |
| APM, NS off | .0168–.0204 | 69.4 MiB |
| APM, NS on | .0164–.0229 | 69.1 MiB |
| FDAF/NLMS default | .0075–.0106 | 59.1 MiB |

The 203K model reduced generated far-only signal power strongly; that alone
does not qualify preservation. The 2.7K control passed the generated near-only
samples unchanged but produced −.31 dB far-only attenuation on the imposed drift
case. APM NS-on reduced harmonic near-only energy/projection, with uncompensated
phase/delay confounding samplewise metrics. No quality winner is selected.

All five recorded cells also completed, with exact full paired coverage,
no production guard fallback and no output sample exceeding magnitude 1:

| Variant | Processing wall / RTF | Whole-child peak RSS | Output/input power change |
| --- | ---: | ---: | ---: |
| LocalVQE 203K | 48.59 s / .3432 | 131.8 MiB | −.167 dB |
| LocalVQE 2.7K | 11.47 s / .0810 | 124.0 MiB | +.036 dB |
| APM, NS off | 2.27 s / .0161 | 131.3 MiB | −.425 dB |
| APM, NS on | 2.65 s / .0187 | 133.0 MiB | −2.474 dB |
| FDAF/NLMS default | .642 s / .00453 | 122.9 MiB | .000 dB |

Reference RMS exceeded the fixed numerical activity floor in 34.18% of original
frames. Output/input power change is not ERLE, echo attribution or a near-speech
score; quieter output can be worse. Whole-child peaks include Python, decoded
inputs, output buffers, libraries and metrics; they are neither model-only nor
whole-assistant memory. Model construction, file/hash checks and thread-mask
sampling are excluded from processing RTF. Each cell is one unpaced offline
pass, not a tail-latency distribution or sustained live/thermal measurement.
LocalVQE/NLMS showed one sampled process thread; APM showed four, all inside the
single-CPU mask. Brief unsampled activity remains unproved.

The first synthetic launch refused before child/model execution because an
installed runtime Python file was legitimately empty. Only runtime Python-file
hashing was changed to admit empty files; model/native assets remain nonempty.
That refusal is retained, not relabeled as a model failure. The synthetic
receipt/source remained unchanged while a separate recorded tool was added.

Exact aggregate SHA-256 receipts (machine-local, no raw arrays/text):

- Prerequisite refusal: `8c74920da203f8fd5a3e10ff4ed67c26304704b6d13217db167c12825f683294`.
- Synthetic v2: `f6bf0631992a6d5a36c62dc72f54a3780498f7ea30d2794de5aaa8c7aa381977`.
- Recorded AFTER-HOST v1: `22f9c2c36c01273aab5bd71a17bae4bfb29ff029511c3862cec5d0c89e8ab04b`.

Reproduction uses the pinned source/submodule/patch and small weights in the
public catalog, a CPU-only `localvqe_shared` build, and an explicit private asset
receipt listing every flat native file's byte length/SHA-256. Keep public assets
under an ignored candidate directory and scratch outside Git. The tools never
download or build implicitly:

```text
python -m tools.localvqe_offline_eval --assets ASSET_RECEIPT --scratch PRIVATE_SCRATCH --output NEW_SYNTHETIC_REPORT
python -m tools.localvqe_recorded_eval --assets ASSET_RECEIPT --manifest PRIVATE_BUNDLE --expected-manifest-sha256 EXACT_HASH --scratch PRIVATE_SCRATCH --output NEW_RECORDED_REPORT
```

## Verification

Generated-fixture/fake-runtime controls: 66 passed in 6.82 seconds. They cover
strict asset/JSON/file binding, portable DLL handling, CPU-thread configuration,
constructor cleanup, mutation/schema refusal, scalar-only output, process timeout,
source-change refusal, fresh model state, frame pairing/continuity, buffered native
hop geometry, exact sample coverage and no mistaken muting win. No owner recording
or model executes in these tests. Adjacent diagnostic-bundle/APM/DTD-metric/AEC
regressions passed 150 tests with 2 optional-model skips in 7.25 seconds. Full Ruff
and format checks pass for all four new Python files. Documentation checks cover
38 files with no dead links, stale terms, retired verbs or orphans; whitespace
passes and STATUS remains 120 lines. Full integration remains the orchestrator's gate.

## Consequences

The result is a reproducible local candidate/evidence seam, not an installed
acoustic replacement. This host's APM is cheaper on these inputs; LocalVQE's
published hardware timings cannot be generalized to this host. The available
recording does not establish a benefit from replacing the active host route,
and echo-only processing does not establish a room-noise admission fix.
No language rewrite or runtime adoption is justified here. The owner's live stop
and every physical/near-talk/STOP/enrollment acceptance gate remain open. The
protected candidate, backup and private evidence are unchanged.
