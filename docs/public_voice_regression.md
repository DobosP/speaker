# Public voice regression fixtures

`tools.public_voice_fixtures` prepares deterministic development fixtures from
the hash-pinned sources accepted by
[ADR-0085](adr/0085-harden-public-voice-regression-provenance.md). Source
archives and generated audio stay under `.cache/public_voice/` and outside Git.

This is development/regression data, never a final held-out set. It cannot
establish owner-voice accuracy, current-room or microphone quality, the
physical echo route, bare-speaker barge-in, or live interaction latency.

## Inspect and prepare

Show the accepted, populated suites:

```bash
python -m tools.public_voice_fixtures list
```

Network access is explicit and limited to one suite:

```bash
python -m tools.public_voice_fixtures prepare \
  --suite commands \
  --accept-download
```

Already-downloaded AMI files can be supplied without network access. Byte count
and SHA-256 must match the manifest:

```bash
python -m tools.public_voice_fixtures prepare \
  --suite conversation \
  --source ami-es2002a-mix-headset=/absolute/path/ES2002a.Mix-Headset.wav \
  --source ami-manual-annotations-162=/absolute/path/ami_public_manual_1.6.2.zip \
  --offline
```

The accepted v2 sources are Google Speech Commands v0.01's test mirror and AMI
meeting ES2002a plus AMI manual annotations 1.6.2. Each record has an exact
HTTPS URL, byte size, SHA-256, bounded SPDX license identifier, license link,
version, attribution, citation, voice-data class, and cache-only policy.
Redirected downloads, unresolved licenses, duplicate or hidden filenames, and
unknown schema fields fail closed.

Extraction reads private, hash-verified source snapshots under the selected
cache filesystem, never the override/cache path after verification. Snapshot
creation tries a copy-on-write reflink first and falls back to a bounded
on-disk file copy. It is cleaned after the whole suite finishes. Plan for
temporary free space up to the selected sources' total logical size when the
filesystem cannot reflink.

Generated suites contain 16 kHz mono float32 `.npy` clips and `metadata.json`.
The metadata binds the manifest hash, complete source records, every generated
case hash, deterministic selection evidence, expected assertion, and explicit
negative evidence scope.

## MInDS-14 intent-ASR slice

[ADR-0087](adr/0087-add-isolated-minds14-intent-asr-corpus.md) records the
separate public-v3 `intent-asr` preparation path. Its default remains opt-in:
the unqualified command above still loads public-v2.

The v3 manifest pins MInDS-14 en-US revision
`40ce77cb32a384e4d50a568e1ec39ac804019d33`, a 34,196,221-byte Parquet object,
its SHA-256, CC BY 4.0 terms, and the corrected paper DOI. The source file has
to be supplied explicitly because the common downloader rejects redirects.
`--accept-download` does not bypass this requirement.

Parquet support stays outside the production `.venv`. Prepare a trusted local
isolated virtual environment at an absolute path and install exactly PyArrow
25.0.0 there, then pass its interpreter explicitly:

```bash
python3 -m venv /absolute/path/to/minds-parquet-env
/absolute/path/to/minds-parquet-env/bin/python -m pip install \
  'pyarrow==25.0.0'

python -m tools.public_voice_fixtures \
  --manifest tests/fixtures/public_voice_manifest.v3.json \
  prepare \
  --suite intent-asr \
  --source minds14-en-us-train=/absolute/path/to/train-00000-of-00001.parquet \
  --parquet-python /absolute/path/to/minds-parquet-env/bin/python \
  --offline
```

Preparation selects one deterministic source transcript and embedded WAV per
intent label. The repository worker is stable-read and executed from an exact
private mode-0400 snapshot. It validates the schema and returns only private
bounded files; the parent verifies them independently before producing the
usual `.npy` clips. Before running it, the parent resolves the lexical
environment root, rejects production-`.venv` aliases, reads the one effective
`pyvenv.cfg`, and requires exactly one false
`include-system-site-packages` key. A bounded sanitized `-I -B` probe then
attests the effective prefix, disabled user site, search paths, and exact
in-environment PyArrow 25.0.0 import; the parent rechecks the root, executable,
and marker after reaping the probe's original private process group. `-I` and
group cleanup are not an OS process/filesystem/network sandbox and cannot
contain a descendant that creates a new session.

Evaluate the prepared slice through the same whole-clip decoder smoke:

```bash
python -m tools.public_stt_eval \
  --manifest tests/fixtures/public_voice_manifest.v3.json \
  --metadata .cache/public_voice/v3/intent-asr/metadata.json \
  --model-path /absolute/path/to/local/faster-whisper-model \
  --label candidate
```

This slice is not held out or speaker-disjoint, and intent labels are
provenance rather than routing assertions. A decoder result therefore says
nothing by itself about streaming latency, agent actions, owner selection, or
live conversation quality.

After the bounded-drain repair passed independent review, pinned preparation
produced 14 generated and 14 evaluator-eligible cases, 136.615 seconds of
audio, and 8,745,168 bytes of `.npy` files. The mode-600 metadata at
`/tmp/speaker-public-v3-20260731-0628/v3/intent-asr/metadata.json` has SHA-256
`95dd18f9d2abdd63afd6dbaaa447189ef4bcc52c1cc126adbd95bbadc904cf8f`.
The isolated worker read its request and source through stable descriptors and
gave PyArrow that already-verified source handle rather than reopening a path.

The code-bound Faster-Whisper Small CUDA control decoded 14/14 with no errors
and all transcripts nonempty. It produced 5 exact transcripts, WER 0.6685, CER
0.6412, and 123 word errors: 2 deletions, 20 substitutions, and 101 insertions.
Whole-clip compute evidence was 3,007.977 ms model load, 1,656.870 ms total
decode, aggregate RTF 0.0121, p50 65.337 ms, and p95/maximum 448.069 ms. The
mode-600 report at
`/tmp/speaker-public-v3-20260731-0628/faster-whisper-small-report.json` has
SHA-256
`619dfb5b421317c1e1e13cf9266ac77d97aa050144669a5a283a6e0cef92efa8`.
Its recorded hashes for `tools/public_stt_eval.py` and the material imported
`tools/public_voice_fixtures.py` logic are respectively
`bfd2611e57a261710b47b618fda520b001fdaa4a0b1638a0c011b09b63a71157`
and `0829139ec9d5b53b2be5b079638799a6148008382f5f7c42c3bad3fab2e657fa`.
The old `d23492ab...` report remains invalid because it binds pre-repair code.
Small's poor result rejects it as an adoption candidate. These numbers remain
development whole-clip evidence, not streaming/interaction latency, held-out,
live-hardware, or model-adoption evidence.

## VoxPopuli Romanian-accented English slice

[ADR-0106](adr/0106-correct-voxpopuli-embedded-float-wav-contract.md) records
the separate 24-speaker `en_ro` preparation path. It accepts only two explicit
local files from the official `en_accented` test split at revision
`42f01879c780b4a2e90ec0b4f616c2ece526e4f1`; the tool contains no downloader
or network client. Review the dataset's CC0 declaration and the linked European
Parliament source legal notice before preparation.

Use a separate environment containing exactly PyArrow 25.0.0. Keep the source
files owner-only and choose a new private output directory. The source object
names, sizes, and SHA-256 values are:

- `test-00000-of-00002.parquet`: 2,471,743,754 bytes,
  `202d71bde2680f30aa7ae1050a98cdf85972458f94e6493db2aa7b8dd8ff46ec`;
- `test-00001-of-00002.parquet`: 2,474,983,297 bytes,
  `7d61b4c7a2e5c565a382543236442a7a8dd081c985e52dd1084905e4becc3901`.

```bash
python3 -m venv /absolute/private/voxpopuli-parquet-env
/absolute/private/voxpopuli-parquet-env/bin/python -m pip install \
  'pyarrow==25.0.0'

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 15 \
  /home/dobo/work/speaker/.venv/bin/python -B \
  -m tools.prepare_voxpopuli_ro_accent \
  --source-shard-0 /absolute/private/test-00000-of-00002.parquet \
  --source-shard-1 /absolute/private/test-00001-of-00002.parquet \
  --parquet-python /absolute/private/voxpopuli-parquet-env/bin/python \
  --output-dir /absolute/private/new-voxpopuli-en-ro-corpus \
  --accept-term CC0-1.0 \
  --accept-term EUROPEAN-PARLIAMENT-LEGAL-NOTICE
```

The aggregate CLI response contains counts and digests only. The private
schema-v2 corpus carries literal references for scoring; its preparation
receipt carries only domain-separated identity hashes, transcript hashes, source/PCM
hashes, selection evidence, and official provenance. The worker inherits the
already-open source descriptors and never follows Parquet audio path values.

Embedded audio must have the exact observed RIFF/WAVE order `fmt `(18 bytes),
`fact` (4 bytes), then `data`; the format is mono 16 kHz little-endian IEEE
float32 and `fact` must equal the data sample count. Exact empty containers are
valid source rows but cannot be selected. The parent repeats these checks and
publishes selected finite float32 samples bit-for-bit. It applies no amplitude
bound, clipping, or normalization because IEEE float WAV does not require the
nominal `[-1, 1]` range.

The four duration bands each contribute six distinct speakers. This is a
formal spontaneous Romanian-L2-English diagnostic after PCM is available. It
does not cover domestic conversation, commands, capture, endpointing, echo,
tools, or live behavior, and model-training overlap may be unknown. The first
real preparation failed closed on the superseded PCM16 assumption. A later
aggregate-only inspection established the exact container contract and a
feasible 24-case selection, but did not publish a corpus. Run a new private
preparation before any model comparison.

## Whole-clip decoder smoke

The evaluator requires Linux and the production Faster-Whisper CUDA contract:
NVIDIA CUDA device 0, float16, one worker, local model files, and the fixed
decode arguments recorded in its report.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m tools.public_stt_eval \
  --manifest tests/fixtures/public_voice_manifest.v2.json \
  --metadata .cache/public_voice/v2/conversation/metadata.json \
  --model-path /absolute/path/to/local/faster-whisper-model \
  --snapshot-root /absolute/path/on-a-disk-with-free-space \
  --label candidate
```

`--snapshot-root` is optional and defaults to the model directory's parent.
The evaluator materializes the complete logical model tree there using
reflink-first/file-copy fallback, verifies its fingerprint, makes the private
files read-only, and keeps them until recognizer deletion and garbage
collection finish. The snapshot is then removed. The fallback therefore needs
temporary free disk up to the model's logical byte size; model files are never
copied into RAM or tmpfs by this mechanism.

`--output /private/path/report.json` atomically writes a mode-600 report. The
aggregate payload includes:

- exact manifest and metadata hashes;
- aggregate source-set and complete case-set hashes;
- an exact recursive model-directory fingerprint;
- decoder implementation hash, dependency versions, and fixed configuration;
- evaluator schema version plus exact hashes for `tools/public_stt_eval.py`,
  `tools/public_voice_fixtures.py`, `tools/recorded_stt_eval.py`, and
  `core/wer.py`;
- decoded counts for `transcript` and `empty_reference`, plus counts for every
  excluded assertion.

All case files, including excluded event-only or unknown-word cases, are read
into bounded hash-verified byte snapshots before model load and rechecked after
decoding. The evaluator rejects more than 8 MiB per case or 32 MiB for the
whole retained corpus. Eligible PCM is decoded only from those bytes. The
`nonempty_transcripts` and `exact_empty` counts use raw `text.strip()`, so
punctuation-only output is nonempty. WER, CER, and their edit counts retain the
project's normalized-text semantics. None is an agent-action or
false-activation measure.

The CLI accepts only the production recognizer. Tests may explicitly opt into
an injected recognizer; those reports bind the callable source, are labelled
non-production, set `evaluator.production_report` false, and contain no
production CUDA or decode-configuration claim.

Fixture paths, names, references, and hypotheses are absent from the report.
Decode timing is whole-clip compute time, not stable-partial,
speech-end-to-final, response, streaming, or conversational latency. No model
result from the earlier draft corpus is v2 evidence.

## Coverage

- `commands`: ten deterministic Speech Commands core words, one
  deterministic auxiliary word, and one pinned silence selection. Speech
  examples use distinct speakers.
- `conversation`: one transcript-labelled AMI window, one four-speaker
  overlap/backchannel event, and one laughter event. Laughter remains
  event-only even when its annotation window also contains words. Annotation
  members must identify distinct AMI speakers, and transcript proof is a
  contiguous normalized token sequence rather than a string substring.
- Product text contracts: current runtime semantics for `stop`/`cancel`, three
  vault phrasings, a reminder, a timer, and two negated near-controls.
  `wait`, `hold on`, and bystander-role handling are explicitly future
  taxonomy and claim no runtime support.

Text contracts are not audio, STT, speaker-selection, or tool-execution
evidence. In particular, a negated phrase being semantically non-STOP does not
prove that ordinary acoustic barge-in will leave playback uninterrupted.

## Licensing and future inputs

The historical FSDD-derived arrays have a fixture-local notice with the FSDD
creator attribution, project and license links, changes, and CC BY-SA 4.0
adapted-material terms. Their exact upstream revision remains honestly
unknown; no release was inferred.

CMU/Festvox speech and Microsoft AEC Challenge material are not accepted v2
sources. They remain possible future inputs only after exact release bytes,
hashes, and applicable terms are resolved. English/Romanian FLEURS has the
same prerequisite. Smart Speaker Commands contains phrase-level commands, but
its archive lacks a reliable machine-readable filename-to-phrase map; retakes
and reordered files make inferred numbering unsafe as ASR ground truth.
Circular STT labelling is not an acceptable substitute. SLURP and Audio2Tool
audio are CC BY-NC or mixed-term schema inspiration, not accepted product data.

## Evidence still required

- Timestamped replay through VAD, endpointing, streaming STT, session
  revisions, and the actual action boundary.
- Licensed AEC pairs with far-end-only, near-end-only, double-talk, and
  alignment assertions.
- Licensed or consented product-phrase audio with target/bystander roles.
- A disjoint owner set spanning controls, near-controls, vault terms,
  numerals, negation, noise/echo, bystanders, and multiple voices.
- Fresh `./live.sh` owner A/B in the current room and route.
