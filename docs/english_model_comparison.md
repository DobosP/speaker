# English model research and local recording benchmarks

Valid until: a model/runtime/profile changes or fresh disjoint device evidence supersedes this comparison — then treat as history.

Verified: 2026-10-04. Decision and authority scope: [ADR-0216](adr/0216-english-model-candidates-and-recording-benchmarks.md).
Current runtime selection: [STATUS](../STATUS.md).

The practical result is a targeted resource trial, not a wholesale latest-model upgrade.
The INT8 version of the current Zipformer cuts process memory and CPU cost substantially.
SenseVoice is the fastest measured final recognizer; Parakeet improves the scripted-set WER
at about a gigabyte of process memory. The installed VITS voice starts PCM much sooner than
Kokoro or the newer TTS candidates. MiniCPM5-2B doubles process memory and is slower than the
current 1B in this CPU canary. New candidates are installed and benchmark tools are executable;
existing live defaults and original recordings are preserved.

## Inputs and measurement conditions

The owner set has 37 provided script references, 101.1 seconds of mono 16 kHz PCM16 audio.
The five-item owner_subset duplicates that set and was not counted as independent evidence.
The recording tool can save real microphone takes or simulated TTS using the same manifest
shape, so these labels do not attest human provenance or a fresh held-out set. Separately,
six existing microphone clips total 12.9 seconds; all original WAV SHA-256 values match their
committed label manifest. They are older development clips, not new held-out/live evidence.
The 31 session turn records have no usable reference text and were not invented into labels.

The models processed local files only. Native stdout/stderr and decoded/reference text were
suppressed; reports contain aggregate counters, checksums, latency and memory only. Each
model ran sequentially in an isolated CPU process, with three repetitions, explicit settings,
timeouts and immutable pre/post source/model binding. Final ASR runs sample every native
thread mask; this is sampled coverage, not continuous/cgroup network or CPU isolation.
CPU-only Linux desktop measurements do not establish phone, GPU, thermal, battery or audible
latency results. Python network hooks and offline/local-file flags are explicit; OS network
sandboxing is not attested.

Four unchanged offline owner rows come from the initial six-complete-cell report; corrected
streaming/Moonshine rows come from a separate four-cell report on the same 37 clips. Their
source reports are independently hash-bound in the aggregate summary. The initial unpadded
Zipformer rows are retained diagnostics and excluded from this comparison. The corrected
Python-example flush appends 10,560 float32 zero samples before input_finished for both
Zipformer precisions. Other model inputs have no caller padding. Original WAVs stay unchanged;
RTF uses original audio seconds. No 37/6 score is pooled.

## Speech recognition results

| Model | Owner37 WER | Mic6 WER | Owner p50 / p95 decode ms | Owner process peak MiB | Owner commands exact |
|---|---:|---:|---:|---:|---:|
| Zipformer FP32 | 19.81% | 20.00% | 883 / 1460 | 386.1 | 3/5 |
| Zipformer INT8 | 19.81% | 24.00% | 520 / 856 | 177.9 | 3/5 |
| SenseVoice INT8 | 14.15% | 0.00% | 107 / 146 | 438.4 | 5/5 |
| Parakeet Unified English INT8 | 8.02% | 0.00% | 484 / 800 | 1168.8 | 5/5 |
| Parakeet TDT v3 INT8 | 7.55% | 0.00% | 534 / 848 | 1148.8 | 5/5 |
| Faster-Whisper Small CPU INT8 | 10.85% | 0.00% | 3474 / 4307 | 748.4 | 5/5 |
| Moonshine Tiny Streaming 0.1.5 / ORT single-thread | 25.00% | 40.00% | 298 / 508 | 241.4 | 2/5 |
| Moonshine Small Streaming 0.1.5 / ORT single-thread | 16.51% | 28.00% | 618 / 908 | 497.4 | 3/5 |

Lower WER is better. Decode p50/p95 measures complete-utterance native work, excluding live
capture, endpoint wait, DSP, authority, answering and playback. Owner accuracy is from the
first repetition; timing covers all calls except each model's first. Peak RSS belongs to the
whole isolated process, not just model weights. The initial report's thread-scope label
incorrectly mentioned post-call samples; those original rows have only before/after-load
samples. The corrected report fixes that label and adds actual native-mask sampling.

INT8 and FP32 have the same owner word-error count 42/212; INT8 has one extra word error on
Mic6 (6/25 versus 5/25). That is a memory/speed tradeoff, not universal quality improvement.
Numerical ITN, disfluencies and scripted references affect WER; these scores do not measure
intent/slot success. Five positive command clips cannot establish false-positive rate.

Moonshine's stock 0.1.5 runtime failed the intended resource budget: a diagnostic observed 117
threads, 115 outside the requested mask, across 32 CPUs. The supported
MOONSHINE_ORT_SINGLE_THREAD=1 variant used two process threads within two CPUs and completed
cleanly. Both new Moonshine rows explicitly bind this variant. Their quality does not beat
the existing final recognizers here; prior 0.1.0 rejection remains separate unchanged history.

## Speech synthesis results

All 111 generations per model use the same 37 provided reference phrases. PCM stays in memory;
no capture or playback device is used. Speaker 0, speed 1, CPU threads 2 and no output DSP are
bound; Supertonic uses eight generation steps. First nonzero PCM is a native callback metric,
not actual audibility. These distributions include the separately reported cold-first call.

| Model | p50 / p95 first nonzero PCM ms | p50 synth RTF | Peak process MiB |
|---|---:|---:|---:|
| Kokoro INT8 v 1.1 (current) | 2325 / 3377 | 1.401 | 470.4 |
| VITS/Piper libritts_r medium (installed) | 134 / 215 | 0.087 | 253.1 |
| Kitten Nano 0.8 INT8 (new) | 693 / 961 | 0.311 | 193.3 |
| Supertonic 3 INT8 (new) | 1950 / 2339 | 0.849 | 296.9 |

All four returned finite, nonzero waveforms. Kitten had four samples at or above full scale
out of 5,548,840 samples; the other three had none. This is waveform sanity, not an audibility,
clipping or naturalness certification. For these short phrases, callbacks
usually arrived only at full synthesis return, so a callback API alone did not provide useful
within-phrase streaming. VITS is the strongest latency option; Kitten is the smallest process
and about 3.4 times faster than Kokoro for first PCM, but slower than VITS. Voice quality and
speaker preference need listening. The current Kokoro aliases have identical model, voices
and token hashes, so counting them twice would not be a new-model comparison.

## Local text-model comparison

Both GGUFs ran with llama-cpp-python 0.3.33, GPU layers 0, context 1024, maximum 64 reply tokens,
greedy decoding and thinking requested off through each embedded template. Eight reference
prompts span the provided script groups; four public canaries give 36 generations per model.
Prompt tokenization respects the formatter's existing special tokens. Source hashes precede
model load, so load timings are not disk-cold measurements. Throughput retokenizes visible
completion text and includes prefill+generation, rather than claiming raw decoder token speed.

| Model | p50 / p95 first visible text ms | p50 completion ms | Peak process MiB | Retokenized completion tokens/s |
|---|---:|---:|---:|---:|
| minicpm5-1b-q8 | 377 / 540 | 1328 | 1285.1 | 11.47 |
| minicpm5-2b-q4km | 895 / 1230 | 2717 | 2601.5 | 6.60 |

The original canaries asked open questions while scoring a narrower output format; those
counts are not used as intelligence rankings. A separate explicit-format public-only run
passed 9/12 strict cases for each model (three repetitions each); both missed the arithmetic
format check, while city, spelling and literal READY passed. These four constraints do not
qualify tool calling, memory, factual dialogue or a model default. The newer 2B has no
latency/resource advantage in this test and is retained as a candidate rather than replacing 1B.

## VAD comparison

The installed Silero export is v 4-era (2023-09-18 source); the new canonical 6.2 weights come
from the checksum-pinned 6.2.3 wheel. That wrapper release is mostly dependency packaging.
Both actually load and execute in installed Sherpa 1.13.3. Python exposes no neg_threshold
setter, so the benchmark explicitly binds the unmodifiable native default and refuses a
nondefault request. Common settings:512 new samples, threshold 0.5, minimum speech 0.25 s,
minimum silence 0.3 s and maximum speech 20 s; reset per clip and flush without additional audio.

| Model | p50 / p95 API callback ms | Callback RTF | Process peak MiB | Detected segments across repeats |
|---|---:|---:|---:|---:|
| silero-v4-installed | 0.173 / 0.270 | 0.0073 | 75.6 | 123 |
| silero-v6.2-wheel | 0.144 / 0.237 | 0.0059 | 79.9 | 114 |

These are API callback/segmentation descriptors, not VAD precision, onset or endpoint scores.
V 6 may fill its 576-sample internal window on the first callback without inference; flush does
not guarantee posterior coverage of residual audio. Counts include three repetitions; fewer
segments are not automatically better. Missing frame and identity labels also prevent speaker,
KWS, natural-turn and live barge-in qualification from this corpus.

## Primary-source research and installed candidates

| Family | New/current source evidence | Action and limits |
|---|---|---|
| Zipformer | [English export](https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26), revision `672fbf1b30579d6585301139bb363f42a0ad4a24`; INT8 tuple 74,207,237 bytes, Apache-2.0. | Installed and measured. It is a precision variant, not new training. Old Kroko mirror licensing/export withdrawal is unresolved, so excluded. |
| Moonshine | [SDK 0.1.5](https://pypi.org/project/moonshine-voice/0.1.5/), 2026-08-24; [August export revision](https://huggingface.co/moonshine-ai/moonshine-voice-assets/commit/0bf2f2e5aff22e6fbba4300b00a4e00bbc4f8aae); Tiny 45,233,659 and Small 142,300,974 bytes, MIT. | Isolated runtime installed; explicit single-thread variant measured. Stock oversubscription rejected. |
| Parakeet | [Unified English](https://huggingface.co/nvidia/parakeet-unified-en-0.6b), NVIDIA Open Model License; [TDT v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), CC-BY-4.0. [Exact existing v3 export](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/commit/2bda32ec70b097a55adaa07d9a7173915b43cc78) is 670,478,772 bytes. | Existing weights compared separately; v3 is not newly downloaded. Realtime EOU 120M prior gate remains rejected. |
| Whisper/Qwen | [Distil-Large-v3.5 CT2](https://huggingface.co/distil-whisper/distil-large-v3.5-ct2), MIT, roughly 1.52 GB; [Qwen3-ASR 0.6B native export](https://k2-fsa.github.io/sherpa/onnx/qwen3-asr/pretrained.html), about 0.94 GB plus tokenizer, Apache-2.0 upstream. | Larger desktop references. Not installed in this low-resource batch; Small CPU already exceeds realtime on these clips. |
| Kitten | [Nano 0.8 INT8](https://huggingface.co/KittenML/kitten-tts-nano-0.8-int8), Apache-2.0; [Sherpa archive](https://github.com/k2-fsa/sherpa-onnx/releases/expanded_assets/tts-models), 2026-05-12, 45,652,547 extracted bytes. | Installed and measured; explicit runtime backend in [ADR-0217](adr/0217-explicit-kitten-tts-backend.md). Smallest measured TTS process. |
| Supertonic/Pocket | [Supertonic 3](https://github.com/supertone-oss-archive/supertonic), archived 2026-09-09, OpenRAIL-M weights/MIT code. [Pocket original weights](https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/blob/d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3/README.md) say CC-BY-4.0; newest release adds contact gating. | Supertonic installed/measured as restricted open weights, not an unrestricted open-source recommendation. Pocket skipped; no contacts or gated consent accepted. |
| MiniCPM/Gemma | [MiniCPM5-2B Q4_K_M](https://huggingface.co/openbmb/MiniCPM5-2B-GGUF/blob/2079a22f3beaa4e306449978533478fe0522f4b3/MiniCPM5-2B-Q4_K_M.gguf), 2026-09-07, 1,561,318,368 bytes, Apache-2.0. [Gemma 4 E2B](https://developers.google.com/edge/litert-lm/models/gemma-4) has official LiteRT support, approximately 2.59 GB bundle. | MiniCPM installed/measured; 1B retained. Gemma 4 requires a separate phone/runtime integration and thermal test. |
| Silero | [Release history](https://github.com/snakers4/silero-vad/releases), [6.2.3 wheel](https://pypi.org/project/silero-vad/6.2.3/), MIT; canonical ONNX 2,327,524 bytes. | Installed and executed as descriptive candidate; no accuracy/default promotion. |
| Denoising | [GTCRN](https://github.com/Xiaobin-Rong/gtcrn) current checkpoint family unchanged; [UL-UNAS](https://github.com/Xiaobin-Rong/ul-unas/tree/main/ulunas_onnx) successor 2026-02 with streaming ONNX, MIT. [DPDFNet](https://huggingface.co/Ceva-IP/DPDFNet) Apache-2.0 but higher compute. | New cache/STFT adapters and raw/clean paired input are needed. No double-denoising or quality claim from unlabeled processed captures. |
| Speaker/KWS/turn | [3D-Speaker](https://github.com/modelscope/3D-Speaker) current CAM++7.2M; ERes2NetV2 is 17.8M. [Smart Turn 3.2](https://huggingface.co/pipecat-ai/smart-turn-v3) current, BSD-2-Clause. [KWS export](https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html) already 2025-12-20. | No verified smaller new speaker checkpoint; per-weight rights remain separate from toolkit license. KWS weight-license clarification unresolved. No new authority/default selected. |

The native-thread failure follows [ORT's documented defaults](https://onnxruntime.ai/docs/performance/tune-performance/threading.html): unset counts can create affinitized per-session pools with spinning. Caller affinity alone is not proof of a hard all-thread CPU budget. This is a resource-policy issue, not a rejection of the newer weights from a generic worker error.

## Reproduce and try the installed options

Commands and complete artifact hashes are in WORKLOG.md and the aggregate evidence summary.
Actual recordings and detailed local receipts remain under ignored recordings/log directories.
The new tools are tools.english_asr_benchmark, tools.english_tts_benchmark,
tools.english_llm_benchmark and tools.english_vad_benchmark. Their help names the bound
manifest/model/config/scratch/output arguments. Models stay in the ignored
pretrained_models/sherpa/benchmarks/english-2026-10-04 cache; the shared production venv was not
changed. Moonshine owns a separate runtime. Candidate install availability is not live adoption.

The installed Kitten overlay is
`pretrained_models/sherpa/benchmarks/english-2026-10-04/kitten-runtime-overlay.json`.
It selects the new backend explicitly, locks speaker 0, uses two TTS threads and CPU.
Its shared `provider` also applies to ASR. It does not overwrite `config.local.json`.
The production factory has actually generated the public phrase sample at
`logs/runs/english-model-benchmarks-20261004/kitten-public-sample.wav` (24 kHz,
6.566 seconds); no playback was performed. A construction-only check from the repo root is:

```python
import json
from pathlib import Path
from core.engines.sherpa import SherpaConfig
from core.engines._sherpa_models import build_tts

overlay = Path("pretrained_models/sherpa/benchmarks/english-2026-10-04/kitten-runtime-overlay.json")
settings = json.loads(overlay.read_text())["sherpa"]
tts = build_tts(SherpaConfig.from_dict(settings))
assert tts is not None
```

Run that check with the existing `.venv/bin/python`. Production configuration layering
and physical sessions remain governed by [ADR-0217](adr/0217-explicit-kitten-tts-backend.md)
and the existing `./live.sh` doctor/owner route gates.

The low-resource trial priority is INT8 streaming plus a fast final recognizer and VITS; Kitten
is a smaller-memory voice option with an explicit backend. Keep current 1B for the CPU fast
text tier. Validate command/noise negatives, a disjoint owner holdout and the intended physical
route before any default change. Do not add isolated process peaks to predict simultaneous
app RSS: library sharing, mmap, KV, DSP and concurrent arenas need a whole-session measurement.

No raw recording, hypothesis, private label or individual clip identity is committed or uploaded
by these benchmark workers. No actual hardware listening/playback, phone thermal test, repaired
bare-speaker A/B or fully offline physical-device acceptance ran in this environment.
