Valid until: this branch lands or is superseded — then treat as history.

# Task result — isolated MInDS-14 intent-ASR corpus

## Outcome

Added a separate public-v3 `intent-asr` development suite without changing the
public-v2 default manifest, exported suite tuple, source semantics, prepared
metadata, eligibility, or metric behavior. The shared evaluator report
provenance now also binds `tools/public_voice_fixtures.py`.

The v3 manifest pins the MInDS-14 en-US training Parquet at revision
`40ce77cb32a384e4d50a568e1ec39ac804019d33`, size 34,196,221 bytes, SHA-256
`37004471dc896ce20771b3fdda0ee8fb33ec7a030fb4e2fd047c561ec0a1ee30`,
CC BY 4.0, and the paper DOI `10.18653/v1/2021.emnlp-main.591`. It selects one
deterministic transcript-labelled recording for each of 14 intents.

Parquet support remains out of the production `.venv`. Preparation requires an
absolute separate virtual-environment interpreter with exactly PyArrow 25.0.0
and an explicit hash-verified source override. The selected interpreter and
environment are trusted local inputs. Redirected download behavior was not
relaxed.

The interpreter gate derives and resolves the lexical environment root, rejects
production-`.venv` aliases, and accepts a shared base `/usr` executable only
when the environment itself remains distinct. It stable-reads the one effective
`pyvenv.cfg`, requires exactly one false `include-system-site-packages` key,
then runs a sanitized bounded `-I -B` probe. Strict probe evidence binds
`sys.prefix`, `sys.base_prefix`, user-site state, every search path, exact
PyArrow version, and an in-environment PyArrow file. Root, executable, and
marker identity are rechecked after the probe, and its original private process
group is reaped. `-I` and group cleanup are not an OS process/filesystem/network
sandbox and cannot contain a descendant that creates a new session.

The regular repository worker is stable-read into a private mode-0400 snapshot
and only that snapshot is invoked with `-I -B`, sanitized environment, no
inherited standard streams, bounded file IPC, POSIX resource limits, and a
process-group timeout. It performs one unthreaded Parquet read, validates the
exact Arrow/Hugging Face schema and pinned row coverage, and never opens either
path metadata field. Request and source reads are descriptor-stable; PyArrow
receives the already hash-verified source handle instead of reopening its path.
The parent and evaluator independently bind source, worker, protocol, PyArrow,
intent, language, transcript, rank, audio, duration, and private-file
provenance.

## Final code-bound preparation evidence

After the bounded-drain repair passed independent review, the orchestrating
session ran the exact pinned source through the isolated PyArrow 25.0.0 path:

- 14 generated cases and 14 evaluator-eligible transcript cases;
- 136.615 seconds of generated 16 kHz mono float32 audio;
- 8,745,168 total bytes across generated `.npy` files;
- prepared metadata accepted by `tools.public_stt_eval._load_corpus`;
- metadata SHA-256
  `95dd18f9d2abdd63afd6dbaaa447189ef4bcc52c1cc126adbd95bbadc904cf8f`.

The mode-600 metadata is at
`/tmp/speaker-public-v3-20260731-0628/v3/intent-asr/metadata.json`. It is
byte-identical to the pre-repair preparation because the repair changed
interpreter validation, not fixture selection or audio. This corpus is not
speaker-disjoint or held out.

## Final GPU whole-clip development control

The same session evaluated that fresh metadata with the existing local
Faster-Whisper Small model under the production CUDA float16 decoder contract:

- `ok=true`, 14/14 decoded, zero errors, and all transcripts nonempty;
- 5 exact transcripts, exact rate 0.3571;
- WER 0.6685 and CER 0.6412;
- 123 word errors: 2 deletions, 20 substitutions, and 101 insertions;
- 3,007.977 ms model load and 1,656.870 ms aggregate decode;
- aggregate RTF 0.0121, p50 65.337 ms, and p95/maximum 448.069 ms;
- mode-600 report
  `/tmp/speaker-public-v3-20260731-0628/faster-whisper-small-report.json`;
- report SHA-256
  `619dfb5b421317c1e1e13cf9266ac77d97aa050144669a5a283a6e0cef92efa8`;
- recorded `tools/public_stt_eval.py` SHA-256
  `bfd2611e57a261710b47b618fda520b001fdaa4a0b1638a0c011b09b63a71157`
  and `tools/public_voice_fixtures.py` SHA-256
  `0829139ec9d5b53b2be5b079638799a6148008382f5f7c42c3bad3fab2e657fa`,
  matching the final files;
- model tree SHA-256
  `c0d8567d470d5e51e59e51c7f1d496986b2bbecc7d2b05d277a5d26194c31bbb`
  across 486,212,372 logical bytes.

The prior `d23492ab...` report remains invalid because it binds pre-repair
fixture code. The final control's accuracy is poor, so Faster-Whisper Small is
rejected as an adoption candidate. Its timing is aggregate whole-clip decoder
compute, not streaming or interaction latency. No intent-routing, agent-action,
owner-voice, held-out, live-hardware, or model-adoption claim is made.

## Files changed

- `tests/fixtures/public_voice_manifest.v3.json`
- `tools/public_voice_parquet_worker.py`
- `tools/public_voice_fixtures.py`
- `tools/public_stt_eval.py`
- `tests/test_public_voice_v3.py`
- `tests/test_public_stt_eval.py`
- `docs/adr/0087-add-isolated-minds14-intent-asr-corpus.md`
- `docs/public_voice_regression.md`
- `STATUS.md`
- `TASK_RESULT.md`

## Verification

Focused deterministic gate after final code hardening:

```text
SPEAKER_TEST_LOG=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
TOKENIZERS_PARALLELISM=false taskset -c 31 nice -n 19 \
/home/dobo/work/speaker/.venv/bin/python -m pytest -p no:cacheprovider \
  tests/test_public_voice_v3.py tests/test_public_voice_fixtures.py \
  tests/test_public_stt_eval.py -q

145 passed in 7.99s
```

The same one-CPU environment passed `tests/test_public_voice_v3.py` alone:
81 passed in 1.66s.

Static and whitespace checks:

```text
/home/dobo/.local/bin/ruff check --no-cache \
  tools/public_voice_fixtures.py tools/public_voice_parquet_worker.py \
  tools/public_stt_eval.py tests/test_public_voice_v3.py \
  tests/test_public_stt_eval.py
/home/dobo/.local/bin/ruff format --check --no-cache \
  tools/public_voice_fixtures.py tools/public_voice_parquet_worker.py \
  tools/public_stt_eval.py tests/test_public_voice_v3.py \
  tests/test_public_stt_eval.py
git diff --check
```

Clean full deterministic landing gate with normal durable test logging:

```text
5883 passed, 13 skipped, 20 deselected, 9 warnings in 99.71s
```

An earlier invocation incorrectly set `SPEAKER_TEST_LOG=0` and ran inside a
filesystem boundary that prevented the CLI child from creating its run log.
Those two environment-caused failures passed together after correction; they
are not counted as landing evidence.

The next evidence step is a controlled candidate comparison on this same
bound corpus, followed separately by streaming replay, disjoint owner
acceptance, and live conversation checks. No commit or push has been made.
