# Public voice evaluation matrix

Architecture and evidence policy are recorded in
[ADR-0098](adr/0098-separate-public-voice-evidence-and-bind-production-baseline.md)
and [ADR-0113](adr/0113-bind-public-conversation-stt-fixture.md). The AMI-only
endpoint proxy is recorded in
[ADR-0129](adr/0129-add-ami-forced-alignment-endpoint-proxy.md). The bounded
short-command/noise fixture is recorded in
[ADR-0117](adr/0117-add-public-short-command-noise-gate.md). Matrix v5 has eight
metric-isolated tracks, 19 selectable sources, and 11 explicit exclusions.
`tools.public_voice_eval_matrix` is a typed catalog and acquisition planner; it
does not download, extract, or run any corpus.
The separate fixed EdAcc-only development comparison is recorded in
[ADR-0143](adr/0143-add-fixed-edacc-stt-comparison.md); it does not change the
matrix or the canonical four-source qualification.

The bounded NOTSOFAR-1 far-field fixture is likewise separate from matrix v5
and its 96-case lock. [ADR-0159](adr/0159-add-bounded-notsofar-far-field-fixture.md)
pins one eval-small meeting at an immutable source revision: nine isolated
single-speaker windows are replayed through two synchronized far-field devices
for 18 hard-WER cases. The selected fixture contains no overlap case and does
not test diarization, enrollment, owner identity, capture, or live behavior.

The first retained development replay completed all 18 cases. Faster-Whisper
Small reported WER/CER `.1538`/`.1030` and 13 exact clips; the one-thread
Zipformer control reported `.5096`/`.3467` and zero exact clips. Their
aggregate-only report SHA-256 values are
`df7fadd455f9de38639ea99fd69ec057d25dc7a22700a11385dee3fe1fb7920d` and
`46d8437310732336605443f38aec9a68fb4474235af36413d6cc18e5f22bf5c1`.
Both are accelerated after-PCM comparisons, not capture, endpoint, live, or
conversational-latency evidence; reported peak VRAM is null.

[ADR-0194](adr/0194-add-notsofar1-natural-overlap-bundle.md) records a second,
still separate NOTSOFAR-1 evidence boundary without changing matrix v5 or the
isolated 18-case fixture. It reuses the exact retained CC-BY-4.0 MTG_32006
five-file source and freezes exact-Decimal natural-overlap selection: at least
300 ms between two valid different-speaker segments whose unknown-free,
known-marker-stripped normalized references exactly match word timing; an
at-most-eight-second envelope with no third intersecting annotation; and an
ordinal-free salted SHA-256 rank that requires at least seven candidates and
takes seven. Seven selected rows contain ten known removable marker
occurrences, seven rows are raw-tag-free, and no selected row contains an
unknown or unrecognized marker; the contract does not require all raw source
text to be tag-free.

That contract is a private custom two-reference bundle rather than a schema-v2
ordinary-WER corpus. The approved exact-source v5 preparation published seven
grouped windows with two original synchronized far-field clips each: 14 leaves
and exactly 2,416,640 f32 bytes. Its manifest and terminal-receipt SHA-256 are
`898e291c713154e3cbd78ba9c38fca6fed08284edace98ac615ebdd37265da7b` and
`29b5d0f1f11f2ac56dee92a7d5ab6ef02a18c4d695a0b9b981b67dde3a24e2ba`;
strict root and manifest reopen passed. The aggregate terminal receipt excludes
reference text/fragments, per-window/case rows, raw speaker/device IDs,
source-relative paths, private-output/local absolute paths, ranks, and
ordinals; it retains only six fixed repository-relative preparer paths and the
fixed manifest filename. The manifest remains private because it necessarily
binds both references. The lock's opaque private-selection commitment is an
integrity binding, not a secrecy claim: the source is public and a determined
party can recompute the selected data. Lock, manifest, and receipt bind exact
`overlap_metric=not-implemented`. Exact built-in schema validation and hostile-
hook tests close full manifest/receipt/source authority before PCM opens.
Focused/adjacent fake gates pass 70/155 tests; type/security, fresh
materialization, and independent final audits are GO. ADR-0194 itself adds no
evaluator and supplies no ordinary WER, ORC-WER, tcpWER, tcORC,
diarization, endpoint, latency, quality, qualification, promotion, default,
device, or live result. Its creation-time `overlap_metric=not-implemented`
field remains an immutable fact.

[ADR-0196](adr/0196-add-receipt-bound-notsofar1-natural-overlap-evaluator.md)
adds a separate evaluator lock bound to those settled manifest/receipt bytes;
it does not rewrite the private bundle. One worker and one repeat submit all 14
channels sequentially in window-major/channel-A-then-B order. Each single
hypothesis is scored privately against both window references with the custom
`two-utterance-min-order-wer-v1` diagnostic. A closed aggregate report may
contain only `aggregate_accuracy`, anonymous `channel_accuracy`, bounded
`paired_device` agreement/difference counts, safe
worker/resource receipts, and exact execution bindings. It forbids references,
hypotheses, window/case rows, roles, raw device IDs, local paths, winner/delta
language, and repeat-stability claims.

The evaluator passes 111 focused and 440 adjacent fake/static tests. The
separately authorized retained Faster-Whisper Small CUDA/FP16 diagnostic then
completed all 14 source-complete burst evaluations across seven windows, two
anonymous synchronized channels, and one repeat. Its custom
`two-utterance-min-order-wer-v1` micro-aggregate is .4231: 77 errors over 182
reference-word opportunities, comprising 70 deletions, 7 substitutions, and
zero insertions. Anonymous channel A/B are .4505/.3956. Both channels reported
the same selected reference order in 7/7 windows, but their normalized
hypotheses agreed exactly in only 1/7; deterministic tie handling means this is
not utterance-order or speaker-attribution evidence.

Preserve the mode-0600/single-link 7,946-byte aggregate report SHA-256
`18ec9f583190752bd683f07341e8205d9b511e6823081b5173405a8fe0df45bd`.
Its 14/14 source-complete count establishes only that every fixed input reached
one typed final with exact source accounting. It does not establish corpus or
turn coverage, repeat stability, ordinary WER, ORC-WER, tcpWER, tcORC,
diarization, endpoint behavior, latency, device identity, held-out quality,
qualification, promotion, default, or live behavior. The report contains no
cgroup/resource observations; the collected launch unit and zero-compute-PID
check are separate operational cleanup evidence. This result remains separate
from matrix v5.

| Track | Public inputs | Report family |
|---|---|---|
| STT/accent | Common Voice 26, Common Voice SPS 4 English, SpeechOcean762, MInDS-14, HarperValleyBank, EdAcc test | literal/canonical WER and CER |
| STOP | Speech Commands v0.02 test | precision, recall, confusions, false alarms |
| Far field/device | DR-VCTK Small, AMI array evidence, paired AMI ES2004a close/far | absolute and paired WER delta |
| Noise/reverb | RIRS_NOISES plus pinned speech; English Consistent Confusion Corpus v1.2 | fixed-grid WER degradation; ECCC noisy literal WER and documented confusions |
| AEC/double-talk | Microsoft AEC synthetic pairs | ERLE/AECMOS, near-end loss, barge errors/latency |
| Overlap/turns | AMI manual transcript with forced-aligned word timing | overlap WER, early cuts, delay, backchannels |
| Spoken intent | MInDS-14 | intent accuracy, separate from WER |
| Text routing | MASSIVE 1.1 | intent/slot/tool routing on text only |

Raw material is planned under `$XDG_CACHE_HOME/speaker/corpora` (or
`~/.cache/speaker/corpora`) and stays outside Git. `build_acquisition_plan`
requires explicit term identifiers, MDC credential presence without retaining
its value, and an API/local SHA-256 receipt wherever upstream lacks one. Its
returned usage constraints include cache-only/evaluation-only handling and any
corpus-specific privacy limits. Reduced row sets use `select_subset_rows`; the
prepared manifest can bind their order with `selected_rows_sha256`.

## Public conversation/STT fixture

The committed
`tools/streaming_stt/public-conversation-96.lock.json` contains four exact
source recipes and 96 abstract slots. Its canonical digest excludes only its
own `recipe_sha256` field. It contains no audio, literal reference, selected
source identity, or local path. The root seed contributes to the receipt-hash
domain; each source recipe separately records the seed its executable selector
actually uses.

Matrix v5 redigests only this lock's root `matrix_version` and
`recipe_sha256`; its four source recipes and 96 slots are unchanged. Existing
private corpora prepared against the earlier root digest must be regenerated or
will fail closed at receipt validation.

| Source | Exact acquisition binding | Fixture selection |
|---|---|---|
| HarperValleyBank | Git commit `0bd721e877c4a85d8c13ff837e68661ea6200a98`, CC-BY-4.0 | three lexical `human_transcript` caller turns for each of eight authoritative task types; 24 distinct callers; metadata names, recognizable dates/clocks, and known marker-only noise/paralinguistic references excluded; machine transcript/dialog-act/emotion output forbidden as labels |
| EdAcc v1.0 test | current corrected DOI `10.7488/ds/7914`; `edacc_v1.0.tar.gz`; 5,916,732,170 bytes; publisher MD5 `146b4b8026b5d0ce9611667c708456b3`; verified local SHA-256 `428e5b5d678ee2dba9ec7362878324a2e5fac1f13ac7b7266cacbcade52bb61b`; verified layout SHA-256 `a4deda10b3246d8795304a4cb11365f3401e1acbb11fe6210ace61369b54e01b`; CC-BY-SA-4.0 | eight clean, eight disfluent, eight overlap-adjacent official human segments; raw text/STM agreement binds and classifies the source, while the key-matched official filtered STM is the hard-WER reference; actual overlap plus empty/alternative filtered references excluded; accent strata diagnostic only |
| AMI ES2004a | annotations: 22,887,865 bytes/SHA-256 `b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d`; Array1-01: 33,579,394 bytes/SHA-256 `6936edac5d0904fc5c4ab175546c5cc5366601fdc1b1e5183a6ea2c10f05d150`; local SHA-256 receipt required for unpinned Mix-Headset; CC-BY-4.0 | eight isolated and four non-overlapping transition intervals, each emitted once close and once far with the identical interval/reference; actual overlap diagnostic only |
| Common Voice SPS 4 English | existing 522,005,930-byte MDC archive/SHA-256 `3b03ada7676a5f440a797d896035137fd073d0683133c3e9a83963480d88abfe`; CC0 plus MDC consumer terms | deterministic six-case projection from each of the unchanged 32-case preparer's four duration bands; 24 distinct speaker and prompt digests |

Each source is published as its own schema-v2 corpus with exactly 24 cases.
Keeping four corpora avoids widening the loader's 32 MiB eager-PCM bound. A
mode-0600 fixed sibling `preparation-receipt.json` binds the catalog dataset,
accepted terms, artifact content receipts, selection/eligible sets,
preparer/decoder closure, opaque identity/speaker/reference hashes, and exact
PCM hashes. The four source materializers recompute that evidence from the
actual checkout/archive/audio rather than accepting caller-authored hashes:

- `tools.prepare_harper_valley_conversation_fixture` verifies the exact clean
  Git commit/tree and tracked metadata/audio blobs.
- `tools.prepare_edacc_conversation_fixture` owns strict raw-header extraction
  of the 5.9 GB production archive into a new private tree, repeatedly rebinds
  the exact source/archive/publication objects, key-matches raw and filtered
  official test STM, and records `official_filtered_stm` on every hard-WER
  receipt case. It admits bounded base-256 only for ignored UID/GID metadata;
  the exact root includes a bounded README, segment metadata remains the exact
  WAV-stem authority, and only a terminal single-digit participant suffix is
  removed when comparing recording stems with strict `conv.list` conversation
  families. Recording and conversation-family sets stay disjoint across
  splits. The pinned linguistic-background participant field is accepted only
  as exact `PARTICIPANT_ID`, with no case or whitespace normalization; existing
  accent, duplicate, and row parsing remains unchanged. Caller-provided trees
  remain synthetic-only
  ([ADR-0142](adr/0142-admit-edacc-participant-id-header.md)).
- `tools.prepare_ami_conversation_fixture` verifies the pinned annotations and
  far WAV plus the caller-supplied local SHA-256 freeze for Mix-Headset. Its
  private `speech-end-labels.json` sibling and evaluation limits are specified
  by [ADR-0129](adr/0129-add-ami-forced-alignment-endpoint-proxy.md).
- `tools.prepare_common_voice_conversation_fixture` re-verifies the pinned SPS
  archive, recomputes its 32 parent rows and MP3 hashes, and binds the parent
  preparation receipt and corpus manifest before projecting 24 cases.

Use each module's `--help` for its explicit source and license arguments. Their
unit-test source injections always emit `production_evidence=false` and cannot
pass the production fixture validator.

The unchanged EdAcc archive completed the hardened no-extraction scan over all
5,916,732,170 bytes with the MD5 and SHA-256 values above: layout SHA-256
`a4deda10b3246d8795304a4cb11365f3401e1acbb11fe6210ace61369b54e01b`,
98 members, and 76 WAVs. Preserve both the subsequent complete extraction that
failed closed on the superseded conversation-family assumption and the later
exact-source attempt that failed before decode/publication on the then-missing
`PARTICIPANT_ID` alias.

After ADR-0142, a new non-production exact-source preflight and a fresh
production materialization matched metadata digest
`e8d06327a595c6e7882bfddd4d9747be7242599fda7a9a09124ba41eb426bc45` and
selection digest
`d8b7e3e8e9dc4f43168ff96d700961261aea028e9d063d730851e53bbd25cd3b`.
Their corpus digests deliberately differ because only the production run emits
`production_evidence=true`. That run retained a new exact archive extraction
of 94 files/8,220,494,044 bytes and published 26 files/4,821,792 bytes: 24
cases containing 4,790,400 PCM bytes plus the corpus and receipt. Every retained
directory was mode 0700; every retained file was mode 0600 with one link. The
production corpus SHA-256 is
`4da392c39a0b6bd18057f63c96b4f67c0dfdaf4c14e1d2d76176cdbee0920772` and
the receipt SHA-256 is
`3c516b6fbb8ce139498cd2e01d83a4faf825e58bf2b9abcaa52896bdab4c853f`.
Preserve all failed, preflight, source, and production evidence. This proves
bounded corpus construction only; it supplies no model, WER, live, or
runtime-default claim.

### Fixed EdAcc-only 2x1 accuracy slice

`tools.edacc_stt_comparison` is the receipt-bound wrapper for the retained
private model run. It accepts only the production corpus and sibling receipt
whose SHA-256 values are
`4da392c39a0b6bd18057f63c96b4f67c0dfdaf4c14e1d2d76176cdbee0920772` and
`3c516b6fbb8ce139498cd2e01d83a4faf825e58bf2b9abcaa52896bdab4c853f`.
It requires exactly 24 cases and reconstructs the locked EdAcc marginal as
eight clean, eight disfluent, and eight overlap-adjacent cases before starting
a worker.

The two fixed cells are the schema-4
`production-gigaspeech-zipformer-fp32-benchmark-1t` control through
`sherpa-onnx-gigaspeech-zipformer-stream-v1`, followed by the schema-6 English
Faster-Whisper Small comparator `faster-whisper-small-local` through the
final-only adapter `faster-whisper-endpoint-v1`. Both use three repeats,
1,600-sample chunks, burst pacing, a 200 ms partial interval, and zero
tail-padding samples. The wrapper delegates them strictly sequentially and
reconstructs a fixed aggregate-only report after checking the corpus, receipt,
workers, code, scratch, and checkpoint before and after execution. Its retained
result is always `quality_decision=not_evaluated`, development-only, and
non-promotional.

The exact run from checkout `96c36fe` completed both coverage-only cells: 72
evaluations per cell over three repeats, with zero final disagreements. Its
20,956-byte mode-0600/single-link aggregate report has SHA-256
`7edcd8a6e2c8422799a94f5b852314517170959167d2821c510a73cff56ed0ba`.
Preserve that report and every bound corpus, worker, runtime, and model receipt.

The Zipformer cell used manifest SHA-256
`ae9db644c22cac5d24107253707729fc4b76d5b4e1c8ce9080bb55c4c3113516`.
It produced WER/CER `.6583`/`.4884`, 0 exact matches, and 57 nonempty finals;
clean/disfluent/overlap-adjacent WER was `.6250`/`.7627`/`.6053`. Its reported
RSS was 376.141 MB with one thread. The Faster-Whisper Small cell used manifest
SHA-256 `594128b55e675363357d446d2e3d3724c474b7e935e537b675926ffd8ddc14da`,
runtime-receipt SHA-256
`85dfc56712f9db4997836329c4fdb2b1e56a265ac9a54a4d5859f74e2ea309a2`,
and model-receipt SHA-256
`28d63ead1b37e6241dbca196155dfc988ea61220e7484ad64ca60b15e9f7c132`.
It produced WER/CER `.3719`/`.2464`, 15 exact matches, and 72 nonempty finals;
the same stratum WERs were `.3281`/`.5085`/`.3026`. Its reported RSS was
1,052.668 MB with six threads. That is a 28.64-percentage-point, about 43.5%
relative WER improvement over Zipformer on this fixed slice.

Both cells reported peak VRAM as null. The external monitor did not sample the
active interval, so this run establishes no peak-VRAM value. RTF and timing are
non-comparable between these cells and are not live latency. The result remains
a small fixed EdAcc accuracy slice: development-only, non-promotional, and
insufficient to change any model or runtime default.

This is an accuracy slice, not a latency comparison: the harness begins after
PCM is available, burst timing is not device timing, and Faster-Whisper is a
complete-PCM final-only adapter. Worker receipts bind the local trees consumed
by the run but do not independently establish upstream model origin or
training-data disjointness. The fixed 2x1 wrapper is additive; it is not a
single-source mode for the canonical 2x4 qualification below and cannot change
an application model or runtime default.

Receipts cross-bind one same-owner snapshot; they are not signatures. The
validator catches drift and inconsistent layer changes, but an actor able to
rewrite the corpus, receipt, and all hashes can reauthor that private record.
Only a raw-source materializer run against the pinned inputs establishes the
preparation claim. Production-shaped grammar fixtures in validator unit tests
are contract tests, never source evidence.

Validation-only reloads all four sources twice and prints aggregate values:

```bash
SPEAKER_TEST_LOG=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 \
  /home/dobo/work/speaker/.venv/bin/python -B \
  -m tools.public_conversation_fixture \
  --corpus /absolute/private/harper/corpus.json \
  --corpus /absolute/private/edacc/corpus.json \
  --corpus /absolute/private/ami-close-far/corpus.json \
  --corpus /absolute/private/cvss4-projection/corpus.json
```

For the exact four-source two-model comparison recorded in
[ADR-0127](adr/0127-require-public-conversation-canonical-strata.md) and hardened
by [ADR-0130](adr/0130-reconstruct-public-conversation-safe-reports.md), use the
additive qualification entry point. It fixes baseline/candidate role order,
derives the canonical source-specific marginal plan from the lock, checks exact
case membership before model work, delegates to the existing sequential suite,
then reconstructs and validates a qualification-v2 fixed aggregate-only 2x4
result. The retained contract binds each worker's adapter and resolved stream
configuration. AMI is reported as separate window-kind and close/far marginals
rather than four joint recipe labels:

```bash
SPEAKER_TEST_LOG=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 \
  /home/dobo/work/speaker/.venv/bin/python -B \
  -m tools.public_conversation_qualification \
  --corpus /absolute/private/harper/corpus.json \
  --corpus /absolute/private/edacc/corpus.json \
  --corpus /absolute/private/ami-close-far/corpus.json \
  --corpus /absolute/private/cvss4-projection/corpus.json \
  --baseline-worker-manifest /absolute/private/baseline/worker-manifest.json \
  --candidate-worker-manifest /absolute/private/candidate/worker-manifest.json \
  --scratch-root /absolute/private/new-candidate-scratch \
  --output /absolute/private/new-conversation96-report.json \
  --repeats 1 --pace burst
```

Preserve private aggregate reports and the `report_sha256` printed by the CLI;
the digest covers the exact canonical report bytes including the trailing
newline. Serialized reports can be checked with
`validate_qualification_report`. A green result means complete execution and
stratum coverage only; the wrapper emits `quality_decision=not_evaluated` and
`promotional=false`. This remains after-PCM development evidence, not native
capture/VAD/AEC, agent/tool, training-disjoint, device-latency, or live evidence.

Both the scratch root and output parent must be owner-private mode-0700 paths.
They must be outside every Git worktree and bound corpus or worker
provision/runtime/model tree. Use a fresh scratch root and a previously absent
output filename. The generic raw suite checkpoint stays under scratch. The
public report is reconstructed from exact aggregate fields, with raw worker text,
stderr, paths, transcripts, and extra evidence omitted. `--output` is staged,
file-synced, read back, and atomically published without clobber only after the
outer fixture, worker adapter/configuration, controller-source, denominator,
stratum, endpoint, privacy, and repeated close-time checks pass.

### AMI forced-alignment endpoint proxy

AMI's word timestamps are automatically forced-aligned, not human-labelled
acoustic speech ends. The private sidecar records that limitation, binds its
ordered label-set digest through the existing corpus provenance, and leaves the
generic schema-v2 corpus and four-source lock unchanged. Preparations made
before ADR-0129 require rematerialization from the pinned AMI inputs into a new
private output directory; the materializer does not overwrite an existing one.

The separate wrapper accepts only the Parakeet Realtime EOU and parakeet.cpp
adapters. It reports aggregate proxy deltas for the 16 isolated cases and
excludes all eight turn-transition cases. A run uses a fresh private scratch
root and previously absent private output path:

```bash
SPEAKER_TEST_LOG=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 \
  /home/dobo/work/speaker/.venv/bin/python -B \
  -m tools.ami_endpoint_proxy_eval \
  --worker-manifest /absolute/private/candidate/worker-manifest.json \
  --corpus /absolute/private/rematerialized-ami/corpus.json \
  --scratch-root /absolute/private/new-ami-endpoint-scratch \
  --output /absolute/private/new-ami-endpoint-report.json \
  --repeats 3 --pace burst
```

The result remains aggregate-only after-PCM development evidence. It defines no
quality threshold and is not acoustic ground truth, capture/VAD/device timing,
training-disjoint evidence, a model promotion, or live validation.

## Public short-command/noise fixture

The committed `tools/streaming_stt/public-command-noise-v1.lock.json` binds the
exact official Google Speech Commands v0.02 test archive, English Consistent
Confusion Corpus v1.2 archive, transform, selection, license, counts, and output
format. The preparer reads both archives directly without extracting them and
publishes a new private schema-v4 corpus outside Git. Inputs must be canonical,
owner-private, mode-0600 regular files with one link; the output must not exist
and must have no Git ancestor:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 \
  /home/dobo/work/speaker/.venv/bin/python -B \
  -m tools.prepare_public_command_noise_corpus \
  --speech-commands-archive /absolute/private/speech_commands_test_set_v0.02.tar.gz \
  --eccc-archive /absolute/private/EnglishCCC_v1.2.zip \
  --output-dir /absolute/private/new-public-command-noise-v1 \
  --accept-term CC-BY-4.0
```

Success prints only aggregate counts and hashes. The new mode-0700 directory
contains 57 mode-0600 f32le clips, `corpus.json`, and
`preparation-receipt.json`: 26 command positives, 26 non-silence command
negatives, five silence cases, and 47 literal transcript assertions. The
receipt contains no local paths, archive members, source speakers, selected
ECCC rows, or per-case transcript text. The private corpus necessarily contains
literal references for scoring.

Run each receipt-bound model sequentially and request all six aggregate strata.
Use a fresh scratch and output path for every cell:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 \
  /home/dobo/work/speaker/.venv/bin/python -B \
  -m tools.streaming_stt_eval \
  --worker-manifest /absolute/private/candidate/worker-manifest.json \
  --corpus /absolute/private/new-public-command-noise-v1/corpus.json \
  --scratch-root /absolute/private/new-candidate-command-scratch \
  --output /absolute/private/new-candidate-command-report.json \
  --repeats 1 --chunk-samples 1600 --partial-interval-ms 200 \
  --tail-padding-samples 0 --pace burst \
  --stratum-tag command-positive --stratum-tag command-negative \
  --stratum-tag gsc --stratum-tag eccc \
  --stratum-tag speech-negative --stratum-tag silence
```

For the schema-v4 Zipformer control only, a separate
`--tail-padding-samples 16000` cell supplies the one second of trailing zero
context needed by short isolated clips. It is a flush/context diagnostic: the
adapter does not call the native endpoint API, and tail compute is divided by
source-only duration. Do not interpret its endpoint or RTF values as a live
comparison. ECCC has no paired clean decode, so report its literal noisy WER
and documented command-confusion metrics rather than a clean/noisy delta.

## Common Voice spontaneous English P0

[ADR-0101](adr/0101-add-bounded-common-voice-spontaneous-stt-slice.md)
records the source, privacy, selection, and evidence limits. Obtain the English
archive only after accepting its MDC terms. Preserve the non-secret filename,
size, and algorithm-labelled checksum fields from the API response; never save
its bearer key, download token, or presigned URL in a receipt.

The preparer is Linux/POSIX-only. Install the isolated decode dependencies from
`requirements-evaluation.txt`; lean CI intentionally skips the two real-PyAV
contract tests when they are absent. The input must be an absolute canonical
path with no symlink component, the archive must be owned by the current user
and mode 0600 (`chmod 600 /absolute/private/archive.tar.gz`), and neither its
ancestry nor the new output path's ancestry may contain a recognized Git
worktree marker (a non-empty `.git` file or a `.git` directory containing
`HEAD`).

It verifies the exact archive twice through one descriptor, does not read
reported-audio comments, and does not download, run an STT model, contact a
network, use a GPU, or open an audio device:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 \
  /home/dobo/work/speaker/.venv/bin/python -B \
  -m tools.prepare_common_voice_spontaneous_v4 \
  --archive /absolute/private/common-voice-spontaneous-speech-4-0-engl-c643378f.tar.gz \
  --output-dir /absolute/private/new-cvss4-en-p0 \
  --accept-term CC0-1.0 \
  --accept-term MDC-CONSUMER-TERMS-2026-05-06 \
  --mdc-api-filename common-voice-spontaneous-speech-4-0-engl-c643378f.tar.gz \
  --mdc-api-size-bytes 522005930 \
  --mdc-api-checksum sha256:3b03ada7676a5f440a797d896035137fd073d0683133c3e9a83963480d88abfe
```

Success prints only counts and hashes. The unchanged output contains `corpus.json`, 32
f32le clips, and `preparation-receipt.json`, all mode 0600 under a mode-0700
directory. The receipt contains no transcript text, raw speaker/source
identifiers, local paths, credential material, or reported comments; the private
`corpus.json` necessarily retains literal reference text for WER. A failed run
may retain partial private evidence for diagnosis; only exit zero is readiness.
ADR-0113 adds a salted prompt digest to each private case receipt so the later
24-case fixture projection can prove prompt diversity without changing the
accepted 32-case selector.

## Zipformer comparison control

The provision command hashes only an already-installed model and writes a new
private receipt. It imports no ASR runtime, loads no model, and downloads
nothing:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m tools.provision_zipformer_baseline \
  --python /home/dobo/work/speaker/.venv/bin/python \
  --model-root /home/dobo/work/speaker/pretrained_models/sherpa/asr \
  --output-dir /absolute/private/new-zipformer-baseline
```

After a corpus-specific preparer has produced a schema-v2 streaming corpus,
run the aggregate evaluator at one thread and either burst or real-time pace:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
nice -n 15 /home/dobo/work/speaker/.venv/bin/python -B \
  -m tools.streaming_stt_eval \
  --worker-manifest /absolute/private/new-zipformer-baseline/worker-manifest.json \
  --corpus /absolute/private/prepared-public-corpus/corpus.json \
  --pace realtime \
  --repeats 1 \
  --output /absolute/private/new-zipformer-report.json
```

The worker uses Bubblewrap with no network, read-only model/environment/source
mounts, and private scratch. The result is an after-PCM production-model
comparison with a deliberate one-thread override. It is not the four-thread
desktop runtime, capture/VAD/AEC evidence, a recognizer change, or live audio
validation.
