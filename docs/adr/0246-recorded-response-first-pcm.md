# ADR-0246: Measure recorded turns through local answer synthesis

Date: 2026-10-10
Status: accepted

## Decision

Use `tools.recording_response_benchmark` as an explicit development evaluator
for complete recorded PCM → selected final ASR → the production local runtime
and LLM factory → first finite nonzero answer PCM after production TTS DSP.
Open no capture/playback device and run no doctor, cloud provider, downloads or
external tool effects. Keep runtime/default/adoption decisions unchanged.

The exact machine-resolved baseline is `current`. The `vits` comparison copies
only the verified installed responsive profile's TTS fields; `vits-fast` also
selects the existing fast first-fragment policy. Both retain baseline streaming
ASR, final recognizer/verifier, LLM models, sampling/context/output limits,
addressing, cleaner, routing and authority settings. This deliberately avoids
attributing full responsive's simultaneous ASR/thread/residency changes to TTS.

## Context / why

Independent ASR/LLM/TTS medians do not measure one real turn. Existing production
FileReplay uses real models and runtime assembly but timestamps whole-clip TTS
completion, rebuilds audio models per row, forces a final-STT profile and cannot
observe production synthesis/DSP callback readiness. Its legacy download path
is inappropriate for an existing-recording latency comparison.

The actual resolved machine baseline on 2026-10-10 differs from generic default
descriptions: cpu_laptop, MiniCPM Q8/Gemma3:4b, 2048 context/256 reply cap,
FP32 streaming ASR with four threads, Parakeet Unified plus the CUDA Faster
Whisper verifier, Kokoro INT8/two threads, streaming text and normal fragments.
Whole-clip output leveling remains enabled. The missing configured Gemma3:4b was
provisioned separately by the orchestrator; a previously runnable full baseline
is not established by that provisioning.

Ollama 0.30.6 advertises completion/tools/thinking for the pinned MiniCPM alias;
its old canonical exact-capability adoption check remains false although
weight/template/parameter checks pass. Gemma3:4b now exposes two local FROM
blobs and vision; the older single-FROM evaluator identity refuses it. The new
benchmark-only binder verifies the local manifest SHA and every config/layer
blob's bytes/size/SHA, requires completion and local regular non-symlink blobs,
and binds every effective raw show field. Only parameter-map/capability ordering
and the non-behavioral modification timestamp are canonicalized; exact templates,
unknown settings and ordered arrays remain bound. Ambiguous duplicate non-stop
parameters are refused. No adoption, doctor or existing identity verifier is
relaxed, and advertised tool support grants no action authority.

## Consequences

- Feed each original recording faster than real time through FileReplay's
  streaming/final/verifier path, with its declared 0.6-second synthetic flush.
  Record wall-clock complete-PCM-available → selected final/first answer PCM/
  idle boundaries. Source-relative endpoint delay is separate. It is neither
  human speech-end nor real-time endpoint, player/FIFO drain, DAC or acoustic
  latency. VAD, capture, AEC, speaker identity and live interruption are outside
  this evaluator.
- Use the real threaded runtime bus, and production Sherpa `_synthesize` with
  its configured DSP to a null sink. Exact normal-reply metric bindings exclude
  acknowledgments/auxiliary output. PCM acceptance and generation revocation
  share the receipt lock; callbacks run after releasing it. Entered native work
  is never retried or treated as preempted merely because cancellation signaled.
- Load native audio models once per isolated cell. Each case receives fresh
  runtime/session/router/ASR-stream state; model caches and normal cross-reply
  TTS scalar/filter state remain resident. Configured warm finishes once before
  timing; later per-case warm is explicitly skipped. `--startup cold` means
  warm-plan disabled, not OS/page-cache-cold or daemon-unloaded: preflight hashes
  read model files, and daemon residency is sampled separately.
- Retain first/subsequent rows separately. Preserve response/no-answer, local
  control, failure/cancellation, fragment/text-chunk/character and PCM counts;
  production does not expose native tokenizer counts. Silence alone is not a
  failed intent. Existing caller-scripted owner and legacy mic corpora remain
  separate, development/non-disjoint and unverified as fresh owner holdouts.
  Labels only score ASR after inference and never influence routing/prompting.
- Compare the same hash-bound responding-case intersection, with response
  gains/losses and denominators. Multi-final, failed or changed-input cells
  cannot qualify a paired comparison. Never pool the two corpora or sum
  component medians. Keep legacy and pinned English-normalized scores distinct
  (ADR-0247).
- Bind clean source revision, an explicit selected source seam, original
  manifest/WAVs, effective config/profile, audio assets, runtime distributions,
  scoring metadata and local model identities before/after. This is not full
  import closure, bytecode, OS-network isolation or daemon-option attestation.
  Python clients reject ambient proxies/redirects and allow only the owned
  loopback endpoint; native output/logging is suppressed. Reports contain only
  typed counters/timings, fixed public metadata and hashes, without private
  text, audio, clip names or paths.
- A per-case worker watchdog and a hard parent cell timeout retain only validated
  numerical partial receipts. Reap the exact worker process group on abnormal
  parent exits. The orchestrator separately owns daemon retirement; terminating
  a client process does not prove server inference/GPU cleanup. Linux process
  RSS/CPU excludes the Ollama server; residency/VRAM/context samples are not
  continuous peaks or layer-offload/utilization measurements. No Windows/macOS
  hardware or live qualification follows.

## Verification

Native-free focused gate: 46 passed, scoped Ruff and whitespace clean. Public
fake ASR/TTS exercise actual runtime and synthesis/DSP, including expression,
no entered-call retry, post-stop rejection, atomic null-sink acceptance,
callback/lifecycle ownership, reused models/fresh sessions, transport privacy,
multimodal layer corruption and parameter-order invariance, strict numerical
receipts, paired cohorts and exact worker timeout/interrupt cleanup.

No native recording-response job, microphone, speaker or doctor ran for this
implementation receipt. Actual native cells require the orchestrator's serialized
window after clean-source preflight; results/limitations belong in the compact
current receipt and WORKLOG. Preserve original recordings and prior baselines.

### Cleanup addendum (2026-10-10)

The case watchdog's exact process exit runs in `finally`, so a failed private
receipt/resource write cannot extend the case to the longer parent timeout.
A process-group-gone race during escalation still reaches the mandatory reap.
The final native-free focused gate is 47 passed in 1.26 s; schema, timing, model
and runtime settings are unchanged. Native response measurements remain pending.

### Explicit canonical model-cache root (2026-10-10)

This host's model-cache ancestor is a legitimate symlink. An optional explicit
`--ollama-model-root` grants only its canonical, non-symlink directory target.
Every daemon FROM must resolve to that root's blobs directory with the same
content-addressed basename; individual blob symlinks and escapes still refuse.
Only canonical manifest/blob paths are read, the root is hash-bound, and FROM
resolution is rechecked before/after binding and the full measured run. No
model/cache/config/adoption verifier is modified and no implicit resolution
fallback is permitted. Fake tests cover default refusal, explicit success,
escaping FROM, symlink leaves and corrupted blob bytes; 47 tests pass.

The installed 0.30.6 manifest adds bounded optional `format`/`runner` strings
and layer `from` provenance strings. The benchmark accepts only those observed
metadata extensions, still verifies the exact raw manifest SHA and every local
blob byte, and never uses provenance strings as paths or remote locations.
The sanitized richer-schema fixture retains all containment/corruption checks.

Daemon-version metadata uses the bounded raw local `/api/version` endpoint,
which remains available with the installed SDK 0.6.2 lacking `Client.version`.
This does not change any inference request or model/client option.
