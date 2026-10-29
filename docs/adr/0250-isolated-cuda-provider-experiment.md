# ADR-0250: Compare an isolated CUDA provider in recorded response experiments

Date: 2026-10-28
Status: accepted

## Decision

Add only `vits-cuda` to the ADR-0246 development evaluator. Against its verified
VITS control, change exactly `sherpa.provider` from the exact string `cpu` to
`cuda`. Use the same typed literal resource map, whole-configuration restoration
check and existing selection-SHA binding as ADR-0249. Reject missing, non-string,
normalized-but-different or already-CUDA control values.

Preserve streaming/final/TTS model files, recognizer decoding and hotwords,
Faster Whisper verifier, native thread requests, LLM models/options/context and
reply limits, hold/cancellation rules, expression/DSP, capabilities, privacy and
authority. No production/default/local-config change, new model download,
provider fallback or adoption verdict follows from the source variant.

## Context / why

Recorded CPU resource ablations completed as descriptive development evidence,
but the host was heavily contended by unrelated work. More CPU threads did not
establish an overall latency win. An explicitly isolated GPU-enabled build can
test device placement without attributing model, decoder or authority changes
to acceleration. The configured Faster Whisper verifier already uses CUDA;
the new Sherpa provider request applies to both streaming/offline ASR and TTS.
It is not a TTS-only or final-ASR-only intervention.

The separately owned runtime preparation uses a pinned Sherpa 1.13.3 CUDA12 /
cuDNN9 wheel in one rolling small environment, with borrowed canonical Python
dependencies and only bounded pinned ABI additions. This source decision does
not attest that preparation's completeness or library identity. Its exact wheel,
interpreter, native dependencies and license/provenance tuple must be recorded
and verified externally before/after each native cell. Canonical virtualenv,
configuration, model weights and other owners' processes remain untouched.

## Consequences

- Run both a fresh CPU `vits` control and `vits-cuda` with the **same isolated
  interpreter, wheel/dependencies and newly committed source**. A prior CPU-only
  wheel result cannot isolate the provider effect. Refuse missing/changed ABI
  prerequisites instead of falling back, replacing system libraries or silently
  using the canonical environment.
- The existing report schema and full ASR/runtime/model/settings bindings remain
  intact. Provider changes deliberately change the full ASR identity; existing
  `compare_paired` continues to refuse them. Retain an external compact exact
  resolved-config witness showing only `sherpa.provider` changed, together with
  the identical interpreter/wheel/native-library tuple and all other bindings.
- A configured provider is a request, not proof that every operator ran on GPU.
  Record bounded actual execution/provider evidence and aggregate GPU/resource
  observations separately. Any unsupported/partial-placement behavior and CPU
  work remain explicit; no GPU latency, memory or power claim follows from
  source selection or shared library loading alone.
- Preserve the declared warm-disabled two-repeat Mic6 protocol, 120-second case
  and 1800-second cell limits, normal process priority, original input hashes,
  reply lengths, quality/final/outcome denominators and first/subsequent cohorts.
  Each condition owns its exact server reset and cleanup. Stop at a classified
  failure; process termination is not proof of server/native cleanup.
- Keep no microphone, speaker, doctor, cloud or external tool effects. Complete
  PCM is replayed unpaced to the actual runtime/synthesis/DSP null sink. Capture,
  AEC, real-time endpoints, DAC/acoustics and live/other-platform claims remain
  outside this experiment. No held-out or model/default-promotion authority is
  created by repeating the development corpus.
- Keep one rolling verified runtime/setup and compact receipts; do not make
  per-iteration environment/model snapshots. Preserve source branches, original
  baselines, canonical release keepers, prior failed/interrupted evidence and
  other owners' work. No artifact tree was pruned for this source change.

## Verification

Native-free public-fake gate: 72 tests passed in 4.69 seconds. New cases prove
exact one-field full-config differences, strict typed CPU control, unchanged
models/options/guards/environment, selection binding and continued strict ASR
pair refusal. Existing three thread variants remain covered unchanged. Scoped
Ruff and whitespace checks pass. No model constructor, CUDA context, native
inference, audio device, doctor, download, merge or default change ran in this
source receipt. The ABI qualification and serialized native comparison remain
separate pending gates.

## Completed isolated provider observations (2026-10-28)

The CPU VITS control and CUDA candidate both ran source
`8a350accd4751e3c3cd0b84bd95e32e0fb418a33` with the same isolated interpreter,
Sherpa 1.13.3 CUDA12/cuDNN9 wheel, bundled ONNX Runtime 1.24.4 and pinned
native-library environment. External before/after manifests verified that
tuple; full resolved-config restoration showed only `sherpa.provider` changed.
The original cells and compact comparison remain under
`logs/runs/english-recorded-quality-latency-20261028/gpu-provider-pass/`. Both
completed 12/12 responses with post-execution input/source/model checks, no
task/TTS-fragment failures and unchanged selected-ASR quality: 3 legacy errors
per 25 reference words in each repeat, 4 English-normalized errors per repeat.

| Same isolated runtime | Subsequent first-PCM p50 / p95 (s), n=11 | Separate first cold response (s) | Whole-cell worker CPU (s) | Peak worker RSS (decimal GB) | Total provider characters / reply fragments |
|---|---:|---:|---:|---:|---:|
| CPU provider | 4.211 / 6.153 | 17.341 | 356.8 | 2.535 | 6623 / 77 |
| CUDA provider | 2.975 / 4.857 | 12.145 | 175.7 | 2.818 | 7146 / 74 |

Selected-final medians were 1.762 → 0.968 seconds and TTS-admission-to-PCM
medians 0.849 → 0.488 seconds; these separate distributions are not an additive
median model. Synthesized output totalled 331.0 → 354.0 seconds, so worker CPU
differences also reflect different unseeded output. The CUDA cell observed
lower latency/worker CPU and higher worker RSS; it does not establish an
isolated causal speedup, lower total-system resource cost or an adoption win.
Host CPU snapshots were 88.7–97.7% during unrelated contending work. The earlier
approximately 1.98-second VITS campaign is not comparable to this pair.

Provider libraries were loaded and public native GPU-memory evidence was
recorded, but continuous peak GPU memory and per-node kernel placement were
not measured. Worker RSS/CPU exclude the Ollama server and GPU memory; no
claim of total-system peak memory, power, capture/DAC/live latency or platform
readiness follows. No default, canonical environment or authority policy was
promoted.

The runtime owner reports approximately 908 MB additional physical storage for
one rolling isolated environment; immutable canonical NVIDIA/dependency files
are also borrowed through hardlinks/path references, rather than duplicated
per iteration. Recorded before/after hashes leave canonical file bytes
unchanged. After verification, that owner pruned approximately 497 MiB of
owned compressed download archives and retained compact provenance/manifests.
Those manifests are historical evidence of the pruned archives; they do not
claim the deleted archive bytes remain available. The verified runtime, source
branches, original baselines and compact results remain retained.
