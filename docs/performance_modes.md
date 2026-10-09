# Voice performance modes

Valid until: a model/runtime/profile or fresh physical-device result changes — then treat as history.

Verified: 2026-10-04. Decisions: [ADR-0218](adr/0218-explicit-capability-preserving-performance-modes.md),
[residency](adr/0219-local-tier-residency-policy.md), [mobile assets](adr/0220-stage-only-active-mobile-model-assets.md),
[mobile budgets](adr/0221-freeze-mobile-startup-thread-budgets.md) and
[native DSP](adr/0222-use-existing-native-streaming-lowpass-kernel.md).
Current implementation/gates: [STATUS](../STATUS.md).

## Desktop selection

```sh
./live.sh --performance responsive
./live.sh --performance compact
./live.sh --performance current
```

| Mode | Streaming ASR | Speech output | Startup / residency |
|---|---|---|---|
| current | Existing selection | Existing voice | Existing settings |
| responsive | Bound English INT8 Zipformer | Installed LibriTTS VITS | Fast tier warm; large main on demand |
| compact | Same INT8 tuple | New Kitten Nano 0.8 INT8 | Same on-demand main policy |

`--performance` also works with `python -m core` and `python -m tools.doctor`.
Models in the ignored local cache must exist with the bound hashes; malformed or
missing assets produce an unavailable mode before microphone capture. Current
requires no candidate cache. No mode setting is written to `config.local.json`.
Use `performance_mode` in that local file for a deliberate persistent selection;
CLI selection wins. Switch between physical sessions, rather than during playback.

The configured final recognizer/verifier, privacy/cloud policy, tools, memory,
vision/research routing, addressing, cleanup, echo processing, speaker authority,
endpoints, expressive directives and context/output budgets survive selection.
Voices differ across model families; the default physical voice is speaker 0.
Active hotwords require the pinned English BPE vocabulary/casing context and
modified beam search; incompatible custom context is refused rather than disabled.

The existing Parakeet/Faster-Whisper final profile requires CUDA. A CPU session can
explicitly choose the existing required SenseVoice final profile:

```sh
./live.sh --performance responsive --final-stt-profile sense-voice
```

That separate final-profile selection changes its recognizer/verifier contract;
it is not performed implicitly by a resource mode. The doctor and runtime receive
the identical selection. The intended physical route must pass normal readiness.

With Ollama, optimized modes warm media and fast/local helpers on one worker,
keep the fast tier resident, and request 60-second main retention after use.
Research/images can therefore pay a cold load when first invoked. Existing daemon
models pinned by other/previous requests are not automatically reclaimed; per-role
retention applies on subsequent requests. Shared main/fast weights retain the fast
policy. GGUF and Flutter Gemma retain their existing context/lifetime contracts.

## Small desktop voice model

[ADR-0232](adr/0232-qualified-small-desktop-voice-profile.md) defines the optional
Qwen2.5-1.5B Q4_K_M profile and its measured spoken prompt. It composes with the
resource modes above; the configured larger model remains available for complex
requests and vision. The public asset is approximately 1.12 GB on disk.
[ADR-0235](adr/0235-retain-legacy-addressing-after-compact-probe.md) records the
rejected compact addressing prompt and the retained production classifier.

With an already running local Ollama daemon, provision the pinned asset/alias:

```sh
python -m tools.setup_voice_model --profile qwen2.5-1.5b
```

Select it using `--voice-model qwen2.5-1.5b` with core, doctor or the launcher.
`--voice-model current` uses the original configured selection for that process.
The same selection reaches readiness and runtime, and a conflicting custom
`--fast-model` refuses before host setup. To persist an explicit selection, the
config key is `voice_model_profile`; provisioning alone does not change it.

The owner stopped physical tests on October 5. The next Linux trial, after
explicit resume and preparation of the retained compatible enrollment lane, is:

```sh
./live.sh --device desktop_gpu_4090 --performance responsive --voice-model qwen2.5-1.5b
```

This command still requires full route/model/enrollment readiness. The four
GPU mixed-context cases passed, with median first text 447 ms and maximum
decision time 629 ms. These are component measurements after the fast-answer/classifier warm calls; main
model and media startup were not exercised, and this is not audible latency. CPU tests had 11–14-second answer outliers and
six false ACT labels among 16 standalone negative cases, so CPU performance
remains unqualified. The separate conversation-admission gate rejects those
16 idle negative fixtures before classification; that is not a general semantic
guarantee or a cure for startup delays.

These measurements came from an occupied Linux workstation; a subsequent
console smoke observed 94–96% total CPU load. No isolation from unrelated
workload or universal CPU speed claim is made.

Windows and macOS native evidence remains separate from these Linux measurements;
[STATUS](../STATUS.md) lists current acceptance limits.

## Measured evidence and limits

The [English comparison](english_model_comparison.md) contains three-repeat corpus
results and primary-source model/license evidence. INT8 streaming reduced isolated
ASR memory by about 54%, with equal larger-set word errors but one extra error on
six older mic clips. VITS had the shortest measured first PCM; Kitten had the
smallest isolated TTS process. These development results do not qualify a new
accuracy default or every noise/command case.

A later CPU probe held streaming ASR and TTS resident together: steady PSS was
805.7 MiB for current, 379.9 MiB for responsive and 355.8 MiB for compact. Each
process requested two threads per engine within a two-CPU caller mask and generated
three copies of one public phrase after synthetic silence. Final recognizer,
verifier, LLM, DSP, capture and playback were absent. This is a component pair,
not whole-assistant/phone RAM, sustained thermal behavior or actual audibility.
The [aggregate source/model-bound receipt](evidence/performance-modes-media-2026-10-04.json)
contains no audio, private text or machine paths. Original diagnostics remain under
ignored logs; commands and verification are recorded in WORKLOG.

The streaming low-pass implementation uses an existing compiled SOS kernel with
the original fallback. Synthetic p50 was 101 us versus 871 us (8.6x faster), with
exact tested PCM/state equality and about 30 KiB more transient memory per 100-ms
chunk. Cold import was excluded. Native model inference already uses Sherpa/ONNX
and Ollama/llama.cpp. A Rust PCM owner would need a measured callback/GIL bottleneck
and the architecture criteria in [ADR-0214](adr/0214-local-voice-performance-architecture.md);
this task avoids an unnecessary new build/ABI dependency.

## Flutter phone selection and packaging

```sh
cd mobile
flutter run --dart-define=SPEAKER_PERFORMANCE=compact
flutter build apk --dart-define=SPEAKER_PERFORMANCE=responsive
```

Phone current/responsive request ASR1/TTS2; compact requests ASR1/TTS1. These are
startup-only per-engine pools; responsive is the existing phone budget, not a
measured speed upgrade. Models, context, endpoint and ownership fences remain
unchanged. Python `--device phone` does not configure Flutter. Kitten is a desktop
selection here; the phone retains its own INT8 Zipformer, Amy-low and Gemma runtime.

Prepare default speech assets with the two existing tools:

```sh
./tool/download-models.sh
python3 tool/generate-asset-list.py
```

The generated bundle selects the exact four ASR files plus available rights/readme
information. Unused precisions/examples and the unused Whisper tuple are excluded;
cached source files remain intact. TTS stages only Amy-low and nested espeak data.
Explicit future Whisper use requires `--with-whisper` on both preparation commands.
Its omitted locked tuple is 160,626,066 bytes uncompressed; no actual APK size or
phone RSS/latency result is claimed. Existing phone answering/lifecycle capabilities
are preserved; a thread selector does not add the desktop tool plane to Flutter.

Physical owner A/B, phone mic/echo/thermal/battery and fully offline whole-session
acceptance still require the intended hardware. No such run occurred here.
