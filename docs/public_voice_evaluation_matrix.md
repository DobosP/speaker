# Public voice evaluation matrix

Architecture and evidence policy are recorded in
[ADR-0098](adr/0098-separate-public-voice-evidence-and-bind-production-baseline.md).
`tools.public_voice_eval_matrix` is a typed catalog and acquisition planner; it
does not download, extract, or run any corpus.

| Track | Public inputs | Report family |
|---|---|---|
| STT/accent | Common Voice 26, Common Voice SPS 4 English, SpeechOcean762, MInDS-14 | literal/canonical WER and CER |
| STOP | Speech Commands v0.02 test | precision, recall, confusions, false alarms |
| Far field/device | DR-VCTK Small, AMI | absolute and paired WER delta |
| Noise/reverb | RIRS_NOISES plus pinned speech | fixed-grid WER degradation |
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

Success prints only counts and hashes. The output contains `corpus.json`, 32
f32le clips, and `preparation-receipt.json`, all mode 0600 under a mode-0700
directory. The receipt contains no transcript text, raw speaker/source
identifiers, local paths, credential material, or reported comments; the private
`corpus.json` necessarily retains literal reference text for WER. A failed run
may retain partial private evidence for diagnosis; only exit zero is readiness.

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
