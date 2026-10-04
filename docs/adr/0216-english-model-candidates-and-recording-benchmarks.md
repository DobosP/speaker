# ADR-0216: Bind English model trials to private recording and resource evidence

Date: 2026-10-04
Status: accepted

## Decision

Prepare and execute explicit English-only, local-file, CPU model candidates in isolated
benchmark processes. Keep original audio, labels, configuration, model locks and older
receipts unchanged. Bind model/source/runtime bytes before and after each complete run and
publish only aggregate quality/resource counters. Separate the37 scripted-reference clips,
101.1seconds, from the6 hash-pinned older microphone clips,12.9seconds. Do not pool their
scores or attest human, held-out, live, phone, endpoint or tool authority from this replay.

Install checksum-bound INT8 Zipformer, Moonshine0.1.5 Tiny/Small, Kitten Nano0.8 INT8,
Supertonic3 INT8, MiniCPM5-2B Q4_K_M and canonical Silero6.2 weights in the ignored model
cache. Keep Moonshine's dependencies in its own pinned runtime. Current production venv,
active config and model defaults remain unchanged. The optional Kitten backend is owned by
ADR-0217, with its own factory/readiness tests and live validation requirement.

Reject stock Moonshine threading as a few-resource configuration: its ORT default pools
escaped the calling thread's mask and exhausted CPU limits. Bind the supported explicit
MOONSHINE_ORT_SINGLE_THREAD=1 variant, sample all worker-thread masks and refuse known CPU
budget violations. Do not claim continuous/cgroup enforcement from samples or OS egress
isolation from Python socket hooks. Use the official0.66second flush for the two Zipformer
benchmark precisions and record padded input separately from untouched source audio.

## Context / why

The owner requested in-depth current open-source model research, application and benchmarks
on retained recordings, then explicitly selected English only. [The comparison](../english_model_comparison.md)
records primary-source release/license evidence and actual native model results. Larger/newer
models are not automatically better for this resource budget. VITS/Piper has the shortest
measured first PCM; Kitten has the lowest measured TTS process memory. INT8 Zipformer preserves
owner word-error count while reducing CPU/RSS, but adds one error on the six-clip set.
SenseVoice is the fastest final recognizer; Parakeet has lower scripted-set WER with a larger
process. The new2B text model doubles process memory and reply time in this limited CPU canary.

The source format cannot distinguish recorded owner speech from simulated TTS; the five-clip
subset duplicates the37 and the31 session records have no reference labels. No reference was
invented from recognition output. Initial failures exposed adapter-contract/protocol issues:
streaming flush padding, oversubscribed ORT defaults, unavailable Python VAD setter and overly
permissive nested aggregate validation. Original failed/diagnostic receipts were retained,
then corrected configurations were measured separately. WER's normalization and numerical ITN
remain limitations; positive command cases do not establish false-trigger rate.

## Consequences

Five repeatable benchmark tools, a public source/license catalog, an aggregate evidence
summary and private detailed receipts support concrete trials without capture/playback,
cloud calls, assistant actions or transcript publication. The [aggregate summary](../evidence/english-model-comparison-2026-10-04.json)
cross-binds contributing reports; a derived eight-model owner table is not described as a
single atomic run and older unpadded streaming rows are excluded from it.

No default is promoted. Apply any intended physical profile only after fresh disjoint quality,
command/noise negatives, owner bare-speaker A/B and phone CPU/RSS/thermal testing. Isolated
process peaks cannot be summed into simultaneous assistant RSS. Silero6.2 API cost and segment
counts provide no VAD accuracy/endpoint verdict without frame labels. Pocket gating and
Kroko's historical weight-license ambiguity prevented adoption; Supertonic is explicitly
restricted OpenRAIL-M open weights, with archived upstream, rather than unrestricted open source.
