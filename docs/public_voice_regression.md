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
  `tools/recorded_stt_eval.py`, and `core/wer.py`;
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
hashes, and applicable terms are resolved. Smart Speaker Commands and
English/Romanian FLEURS have the same prerequisite. SLURP and Audio2Tool audio
are CC BY-NC or mixed-term schema inspiration, not accepted product data.

## Evidence still required

- Timestamped replay through VAD, endpointing, streaming STT, session
  revisions, and the actual action boundary.
- Licensed AEC pairs with far-end-only, near-end-only, double-talk, and
  alignment assertions.
- Licensed or consented product-phrase audio with target/bystander roles.
- A disjoint owner set spanning controls, near-controls, vault terms,
  numerals, negation, noise/echo, bystanders, and multiple voices.
- Fresh `./live.sh` owner A/B in the current room and route.
