# ADR-0092: Require capture-loop replay before STT promotion

Date: 2026-07-31
Status: accepted

## Decision

Require every future STT or final-transcript promotion to pass a hash-bound,
timestamped replay through the configured Sherpa
capture-loop seam before live A/B validation. Use capture-replay schema v1 with
100 ms f32le frames, immutable source identity, speech/speaker/word intervals,
and aggregate-only accuracy, acoustic-lineage, latency, streaming, paced-wall,
and resource metrics. Emit `execution_complete` plus
`quality_verdict=diagnostic_only`; never describe command success or schema
coverage as an accuracy pass. Prepare public AMI ES2004a Array1-01 locally from
caller-pinned inputs; never download data in the preparer or commit audio or
transcripts. During public replay, omit output-only and speaker-identity models,
run final recognition inline for deterministic event accounting, and otherwise
retain calibration, AGC, denoise, VAD, streaming/final ASR, punctuation, and
endpoint settings. Leave production defaults and `./live.sh` unchanged until
this diagnostic, disjoint owner recordings, and a fresh live bare-speaker A/B
support a change.

For this decision, replay “pass” means `execution_complete=true`, complete
declared case/provenance coverage, unchanged input/source snapshots, and no
evaluator, verifier, contract, or resource-bound error. It deliberately defines
no accuracy threshold. Accuracy adoption remains a separately predeclared,
disjoint-owner baseline/candidate A/B followed by the live gate.

## Context / why

The isolated Moonshine and Nemotron evaluations measured recognizers after PCM
was already available. They could not expose capture cadence, calibration,
speech-evidence, VAD, endpoint, or final-selection failures, so low inference
RTF was not evidence of a responsive or accurate agent.

The first corrected capture-loop replay used four deterministic, mic-only AMI
cases from one meeting and channel (single speaker, four-person human overlap,
two-person turn transition, and room silence), 59.4 seconds total, one repeat,
manifest SHA-256
`6b7b486d01384870ac64c183b99794625d519e0b4490a7bc651615466dcf1433`.
With the current CPU Sherpa path, configured NeMo-transducer final, and
CUDA-only Faster-Whisper verifier, it retained complete timing lineage,
attested six completed verifier decodes with zero errors, and emitted no text
for silence. Accuracy was red: overall WER/CER 0.8519/0.7441, with WER 0.6316
for single-speaker, linearized WER 1.0000 for order-ambiguous four-person
overlap, and WER 0.9333 for the turn-transition case; 68 of 92 word errors were
deletions. Across seven timed utterances, first-partial and speech-end-to-final
p50 values were 184 ms and 1,113 ms, with 61 partial churn edits, 28
retractions, one input rejection, and 2,033 MB peak process RSS. The 1.0065
wall/audio ratio confirms causal pacing, not inference headroom. This
development result does not reject a production model by itself, but it proves
isolated model scores cannot authorize a default change. Report SHA-256:
`929f265bf147e768f98e33269171108b6e0bc77518a832f896f161af0cadaa6d`;
23-file evaluator source digest:
`f594b3cff8830dc8f9b4e11c00fbfa21411d12b25f640b9a08a1ca473f75dd4e`.

AMI is a far-field meeting corpus. Its overlapping voices are multiple near-end
speakers on one microphone channel, not a synchronized playback-reference AEC
test; linearizing simultaneous words also makes overlap WER order-ambiguous.
The turn case does not grade final count or speaker-boundary correctness. The
slice has no commands, enrolled owner, multilingual speech, device matrix,
current-room echo, or physical output. It therefore cannot establish command
recall, owner authority, double-talk barge-in, perceived TTS latency, thermal
fitness, or live hardware behavior.

## Consequences

The replay schema, AMI preparer, and evaluator become a deterministic post-read
diagnostic seam. The CLI runs model work in a killable watchdog child and
validates only a bounded receipt. Reports refuse overwrite, bind the corpus/effective
configuration/model metadata plus a 23-file evaluator source subset, and
publish only closed aggregate conditions; private audio, arbitrary tags, case
names, and decoded text stay outside Git. Zero command attempts have no recall
value.

Model artifact metadata and the named 23-file source subset are not
dependency-complete content receipts, and public AMI remains a small
development slice. The evaluator installs deterministic real room-tone
calibration directly, invokes `SherpaOnnxEngine._capture_loop`, and bypasses the
native reader, real `CaptureMediaSession` queue, overflow/gap recovery, and
asynchronous final worker. Exact candidate provisioning retains its stronger
model/dependency content hashes. Its endpoint values are engine-internal
latencies, not ground-truth onset/offset accuracy; VAD, endpoint, and denoise
promotions need that missing grading. The deterministic postroll is a hard
splice to repeated real room tone, so it may perturb endpoint timing despite
removing the earlier digital-zero shortcut. STT promotion still requires
disjoint owner recordings with target command families plus a fresh
`./live.sh` bare-speaker A/B. A later corpus revision must add
native-reader/gap cases and synchronized mic/far-reference tracks before it can
grade capture recovery, AEC, or physical double-talk.

This evidence seam is orthogonal to the runtime roadmap and does not reorder
it: first bound the post-ASR mailbox, then add actor-backed `TurnHandle`
structured cancellation, and only then route remote/mobile through the same
ownership contract.
