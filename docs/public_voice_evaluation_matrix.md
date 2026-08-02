# Public voice evaluation matrix

Architecture and evidence policy are recorded in
[ADR-0098](adr/0098-separate-public-voice-evidence-and-bind-production-baseline.md)
and [ADR-0113](adr/0113-bind-public-conversation-stt-fixture.md). The bounded
short-command/noise fixture is recorded in
[ADR-0117](adr/0117-add-public-short-command-noise-gate.md). Matrix v5 has eight
metric-isolated tracks, 19 selectable sources, and 11 explicit exclusions.
`tools.public_voice_eval_matrix` is a typed catalog and acquisition planner; it
does not download, extract, or run any corpus.

| Track | Public inputs | Report family |
|---|---|---|
| STT/accent | Common Voice 26, Common Voice SPS 4 English, SpeechOcean762, MInDS-14, HarperValleyBank, EdAcc test | literal/canonical WER and CER |
| STOP | Speech Commands v0.02 test | precision, recall, confusions, false alarms |
| Far field/device | DR-VCTK Small, AMI array evidence, paired AMI ES2004a close/far | absolute and paired WER delta |
| Noise/reverb | RIRS_NOISES plus pinned speech; English Consistent Confusion Corpus v1.2 | fixed-grid WER degradation; ECCC noisy literal WER and documented confusions |
| AEC/double-talk | Microsoft AEC synthetic pairs | ERLE/AECMOS, near-end loss, barge errors/latency |
| Overlap/turns | AMI manual timing | overlap WER, early cuts, delay, backchannels |
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
| EdAcc v1.0 test | current corrected DOI `10.7488/ds/7914`; `edacc_v1.0.tar.gz`; 5,916,732,170 bytes; publisher MD5 `146b4b8026b5d0ce9611667c708456b3`; CC-BY-SA-4.0 | eight clean, eight disfluent, eight overlap-adjacent official human segments; raw text/STM agreement binds and classifies the source, while the key-matched official filtered STM is the hard-WER reference; actual overlap plus empty/alternative filtered references excluded; accent strata diagnostic only |
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
- `tools.prepare_edacc_conversation_fixture` verifies both the 5.9 GB archive
  and its direct private extraction, key-matches raw and filtered official test
  STM, and records `official_filtered_stm` on every hard-WER receipt case.
- `tools.prepare_ami_conversation_fixture` verifies the pinned annotations and
  far WAV plus the caller-supplied local SHA-256 freeze for Mix-Headset.
- `tools.prepare_common_voice_conversation_fixture` re-verifies the pinned SPS
  archive, recomputes its 32 parent rows and MP3 hashes, and binds the parent
  preparation receipt and corpus manifest before projecting 24 cases.

Use each module's `--help` for its explicit source and license arguments. Their
unit-test source injections always emit `production_evidence=false` and cannot
pass the production fixture validator.

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

For a model run, use the same entry point with one or more receipt-bound worker
manifests. It validates first, passes the corpora to the existing suite in lock
order, runs its cross-product strictly one cell at a time, revalidates every
receipt/manifest/PCM afterward even when the suite fails, and only then wraps
the private aggregate report:

```bash
SPEAKER_TEST_LOG=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 \
  /home/dobo/work/speaker/.venv/bin/python -B \
  -m tools.public_conversation_fixture \
  --corpus /absolute/private/harper/corpus.json \
  --corpus /absolute/private/edacc/corpus.json \
  --corpus /absolute/private/ami-close-far/corpus.json \
  --corpus /absolute/private/cvss4-projection/corpus.json \
  --worker-manifest /absolute/private/candidate/worker-manifest.json \
  --scratch-root /absolute/private/new-candidate-scratch \
  --output /absolute/private/new-conversation96-report.json \
  --repeats 1 --pace burst
```

Preserve private aggregate reports; this is after-PCM development evidence,
not native capture/VAD/AEC, agent/tool, training-disjoint, device-latency, or
live evidence.

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
