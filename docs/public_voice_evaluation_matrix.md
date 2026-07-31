# Public voice evaluation matrix

Architecture and evidence policy are recorded in
[ADR-0098](adr/0098-separate-public-voice-evidence-and-bind-production-baseline.md).
`tools.public_voice_eval_matrix` is a typed catalog and acquisition planner; it
does not download, extract, or run any corpus.

| Track | Public inputs | Report family |
|---|---|---|
| STT/accent | Common Voice 26, SpeechOcean762, MInDS-14 | literal/canonical WER and CER |
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
