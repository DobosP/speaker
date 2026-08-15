# ADR-0207: Bind and run the exact mobile Zipformer endpoint evidence path

Date: 2026-08-15
Status: accepted
Refines: ADR-0098, ADR-0109, ADR-0117, ADR-0159, ADR-0161, ADR-0162,
ADR-0203, ADR-0205, ADR-0206
Supersedes: none

## Decision

Accept one offline, config-faithful desktop CPU evidence path for the shipped
English mobile Zipformer asset tuple. The path has three independently
fail-closed stages:

1. a hash-only provisioner for the exact four model artifacts;
2. a schema-10 worker manifest and protocol-v5 native-endpoint decoder; and
3. a fixed five-domain evaluator over ADR-0206's exact packet-v2.

This is evidence for the exact asset/configuration tuple when executed by the
bound Linux Python runtime on CPU. It does not establish Flutter FFI, Android
native-library/app-integration, phone, microphone, thermal, or live evidence.

The model identity is
`sherpa-onnx-streaming-zipformer-en-2023-06-26-mobile-hybrid-v1`, with adapter
`sherpa-onnx-mobile-zipformer-endpoint-v1`. Its 2,335-byte committed source lock
has raw SHA-256
`3b928f709a1c50f426291968bedff67811f7eed0e1c48e0773df475558520efc`,
canonical recipe SHA-256
`736d516ceada4bc8864ca0b5de9f03167a415cb93ad66252ab99351028a15c49`,
and binds upstream revision `672fbf1b30579d6585301139bb363f42a0ad4a24`.
It fixes four ordered artifacts, artifact-set SHA-256
`eebca4036773b78642c846acdc966d1bab8f377a8b8ab632f9d3aac7ab38bcc5`,
and exactly 74,207,237 bytes:

| Artifact | Precision | Bytes | SHA-256 |
|---|---|---:|---|
| `encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx` | int8 | 71,083,163 | `563fde436d16cf7607cf408cd6b30909819d03162652ef389c2450ced3f45ac1` |
| `decoder-epoch-99-avg-1-chunk-16-left-128.onnx` | fp32 | 2,092,621 | `7bf787f90b194b307e5a4ad6a34fadb4e748304c35f78a8d66358a05b13ee6ef` |
| `joiner-epoch-99-avg-1-chunk-16-left-128.onnx` | fp32 | 1,026,405 | `210591f72b3c56b8364f85f345dca240bc2b4c00632848f4aa923630d5639d3b` |
| `tokens.txt` | text | 5,048 | `49e3c2646595fd907228b3c6787069658f67b17377c60aeb8619c4551b2316fb` |

The exact 21-field mobile configuration has canonical SHA-256
`749fa50525a08fc612bc82fb4e05b4fbfa62f3e762b621acceaf05f9223fa883`.
It fixes sherpa-onnx and sherpa-onnx-core 1.13.3, NumPy 2.4.6, English,
16 kHz, 80 features, one CPU thread, debug enabled, greedy search, four active
paths, Zipformer2, endpoint detection enabled, endpoint rules
`2.4`/`0.8`/`20.0` seconds, native 1,600-sample chunks, and at most 48,000
zero-tail samples.

The provisioner accepts only those model bytes, the fixed worker source, and
the exact Python distribution metadata. Its isolated no-site metadata probe
rejects startup hooks and does not import NumPy or sherpa-onnx. It does not load
ONNX, decode audio, use a GPU, or open a device. Publication copies the admitted
files into a new owner-private outside-Git root, writes the schema-10
`mobile-zipformer-provision-v1` manifest, and atomically links a single terminal
receipt under a blocked HUP/INT/TERM commit window. Before that receipt the
reserved root is incomplete; recovery uses a new absent path. The strict loader
reopens and rehashes the lock, runtime, worker, manifest, receipt, and model set.

Extend the existing streaming worker without changing older manifest schemas
or protocols. Protocol v5 records each successfully reset-committed native
endpoint as its text, observed sample, and source-complete flag. Each native
up-to-1,600-sample source or tail chunk is accepted, drained, read for its current
result, and checked for endpoint status. A true endpoint is recorded and the
stream is reset; decoding may then continue through the remaining source PCM.
After source exhaustion the adapter may feed no more than the declared 48,000
zero samples. The terminal text is only the ordered nonempty endpoint texts.
A run with zero endpoint observations stays empty. Tail exhaustion may retain earlier
committed endpoint texts, but never manufactures text from the last partial or
another non-endpoint result. Never call `input_finished`.

Use a worker-lifetime cumulative unique-PCM-SHA ceiling. Retain the generic
ceiling of 32 MiB for every older or non-mobile worker. Only an exact schema-10
`sherpa-onnx-mobile-zipformer-endpoint-v1` worker receives a fixed 38 MiB
ceiling. That admits the exact 39,299,500-byte packet while remaining below a
generic broadening; repeated identical PCM SHA-256 values are not counted twice.
Crossing the selected ceiling is fatal and detail-free.

Fix the evaluator to packet-v2 lock/index/receipt
`c81a89a7…cc11`/`232e4230…835c`/`36eec947…f51f`. Historical packet-v2 remains
loadable across the later preparer closure only through packet-local validation
of its fixed historical receipt/source-lock bytes. Public PriMock loaders remain
current-closure-bound, and this compatibility does not promise arbitrary future
runtime independence. The evaluator derives its PriMock leaves only from the
packet loader's immutable validated component snapshots, not by reopening those
components through current public loaders.

Run exactly one sequential repeat over all 129 packet leaves in fixed component
order, with 1,600-sample chunks, burst feed, at most 48,000 zero-tail samples,
protocol 5, and 100 ms partial-report cadence. The cadence is only a reporting
control and the tail is only bounded feed accounting; neither has latency or RTF
authority. Reopen and rehash the exact packet, model manifest/receipt, model set,
worker source bundle, evaluator closure, scratch, and all staged PCM before
publication. The report may be linked only after worker close, fully-stopped
proof, host-lock close, and final input/source/scratch verification.

Keep the five result cells separate and in fixed order:

| Component | Metric domain | Evaluations |
|---|---|---:|
| GSC/ECCC command-noise | `command-assertion-v1` | 57 |
| MInDS/DEMAND domestic noise | `stratified-wer-cer-v1` | 42 |
| NOTSOFAR far-field | `paired-channel-wer-cer-v1` | 18 |
| PriMock isolated | `ordinary-wer-cer-v1` | 3 |
| PriMock grouped overlap | `two-utterance-min-order-wer-v1` | 9 leaves / 3 scored mixes |

Publish only one bounded, aggregate-only, no-clobber report. It may contain
component aggregates, endpoint-integrity counts, exact closure hashes, and
bounded worker-stderr metadata. It must contain no hypothesis, reference,
transcript, case ID, local path, speaker/role, or per-leaf row. Pooled WER and
every cross-domain aggregate accuracy score remain absent.

Accept the strictly reloaded production report with SHA-256
`264f439652064ca78821abdcfed77d03a9a637fb82d6ed336b7a6112d083e796`.
It is 9,792 bytes, mode 0600, single-link, complete, and binds the exact packet,
model provision, configuration, worker source, and evaluator closure. Its
endpoint-integrity counts are:

- 129/129 source-complete evaluations and 129/129 endpoint-terminated
  evaluations, with zero tail-exhausted and zero zero-endpoint evaluations;
- 89 single-endpoint and 40 multiple-endpoint evaluations;
- 198 reset-committed endpoints: 139 nonempty, 59 empty, and 69 observed before
  complete source consumption; and
- 110 evaluations with a nonempty joined hypothesis and 19 with an empty one.

Accept the following five unpooled result cells without deriving a winner,
cross-domain score, or quality verdict:

| Domain | Exact aggregate result |
|---|---|
| Command/noise | command recall 18/26 (`.6923`); forbidden targets 0/355; monitored negatives with a hit 0/31; 47-row transcript subset WER `.6809`, CER `.6610`, exact 19/47 |
| DEMAND | WER `.9149`, CER `.8076`, exact 2/42; environment WER `.8315` kitchen / `.9348` living / `.9783` washing; SNR WER `.9511` at 0 dB / `.8696` at 10 dB / `.9239` at 20 dB |
| NOTSOFAR | WER `.4423`, CER `.2638`; anonymous channel A/B WER `.5000`/`.3846`; exact normalized channel-output agreement 4/9 |
| PriMock isolated | WER `.6154`, CER `.3684`, exact 0/3 |
| PriMock overlap | custom two-utterance minimum-order WER 44/111 (`.3964`) on three mixes; six stems executed but unscored |

The three DEMAND environment rows partition the 42 evaluations once, and the
three SNR rows independently partition them once. Do not average all six rows.
The overlap metric is not ordinary WER, ORC-WER, tcpWER, or tcORC.

The successful production report alone sets
`mobile_model_identity_authority`, `model_executed`, `production_packet`, and
`unpooled_component_result_authority` true. It leaves default, device,
held-out, latency, live, microphone, model-quality, promotion, qualification,
and training-disjoint authority false. These fields authorize only the exact
desktop-CPU model execution and the five unpooled cells above. They do not
authorize Android/native equivalence, a model-quality verdict, a default
change, promotion, qualification, latency/RTF, phone/device behavior, or live
behavior.

## Context / why

ADR-0206 made the exact five-component packet available but deliberately gave
it neither model-identity nor evaluation-result authority. The retained desktop
fp32 Zipformer and CUDA comparators are not the mobile asset tuple. Likewise,
whole-clip or `input_finished` decoding would erase the endpoint behavior used
by the app and could turn a partial into a false final.

The narrow direct-runtime proxy uses the same asset basenames and the bound
desktop 1.13.3 counterpart configuration while preserving native endpoint
decisions. It can execute deterministically and privately on the available
desktop CPU. It cannot stand in for Flutter FFI, Android scheduling, a phone
microphone, or thermal behavior, so those boundaries remain explicit.

The packet components answer different questions. Command assertion,
ordinary/stratified recognition, paired-channel recognition, and minimum-order
overlap are not one exchangeable metric family. A fixed five-cell report makes
the observations reviewable without inventing a packet WER or promotion score.

The first packet-scale execution exposed the existing generic 32 MiB
worker-lifetime input ceiling. That ceiling was deterministic and correctly
failed closed, but it could never consume ADR-0206's 39,299,500-byte packet.
A one-case diagnostic completed cleanly. A schema-and-adapter-specific 38 MiB
unique-SHA ceiling is the smallest whole-MiB ceiling that admits this exact packet
without broadening any established worker.

## Consequences

- The exact mobile asset/configuration tuple now has one immutable, aggregate,
  config-faithful desktop CPU proxy result. It remains non-promotional and has
  no pooled metric or quality/default authority.
- The provisioner admits and copies exact bytes without importing the target
  runtime or executing ONNX. The later evaluator imports and executes the exact
  provision only inside its isolated worker.
- The worker represents zero, one, or multiple real reset-committed endpoints
  with exact source/tail accounting. Tail exhaustion adds no uncommitted text;
  it may stay empty or retain only earlier committed endpoints.
- Packet-v2's historical PriMock bytes remain evaluable without weakening
  public current-closure loaders or changing ADR-0201 through ADR-0206.
- The English mobile application default remains unchanged. Romanian still
  requires its own model/export, language/tokenizer/Unicode contract,
  compatible evidence, owner recordings, and physical-phone CPU/RSS/thermal/
  live decision.
- Physical validation must use the actual Flutter/native path on a phone. The
  retained desktop CPU result is not a substitute.

## Verification

- Independently audited implementation artifacts are: provisioner
  `aee89de7f4b1ab0fa2d6d2d92ea5959ff81b21507f76ffbf3e064eecae29d416`,
  protocol/runner
  `0d6c7840557c49235099c3bc4ea91899d1b4cbbaa68180b23064331939542c5d`,
  corrected evaluator
  `f4be380cfaefba9349019b270b9954b694b49d46709e16269e1b7253d23b8895`,
  packet historical-closure compatibility
  `82909c6c65faf3875a9e5075247c9c4f92b7fa7a78ca2374e36df4e72904266d`,
  evaluator packet-snapshot compatibility
  `114b79510256263484e162ca40ead463dd157fb5894251d20aa3d7d2f7419050`,
  and mobile-only worker ceiling
  `8a2bf68e9f23d9069ca9769741a1e15f5c15570360c73014dc8b751979f6b8b0`.
- The final integrated nine-test-file gate, including mobile runner/evaluator,
  packet/provision, shared manifest/protocol/supervisor, legacy Zipformer, and
  APM/DTD, passed `444` with one environment skip and two known timeout-plugin
  warnings in `25.15s`. Its 25%-CPU/512-MiB-hard/zero-swap unit consumed 6.041
  CPU-seconds, peaked at 76.1 MiB, and used zero swap. The exact mobile worker
  correction passed 30/30; combined affected gates passed 126/126; adjacent
  coverage passed 341 plus one environment skip. Scoped Ruff checked all 13
  changed Python files clean with `--no-cache`. Ruff format-check found 10/13
  clean; `adapters/zipformer.py`, `manifest.py`, and `protocol.py` retain the
  same three independently verified pre-existing base formatting debts and were
  not rewritten. `git diff --check` is clean.
- The fresh retained `provision-v2` output is still manifest schema 10/kind
  `mobile-zipformer-provision-v1`; `v2` names the fresh attempt, not a new schema.
  Its manifest SHA-256 is
  `d628380b0b81e4ccf29f48daadfc1ac72767bb0b178755c0d0ef09b59bbb1641`,
  terminal-receipt SHA-256 is
  `aa053ed681d85ade3b5e15b2ec8d9a261b9b66f5fcd2f7a2395c8cf02d4f04b4`,
  artifact-set SHA-256 is `eebca403…bcc5`, and the exact model total is
  74,207,237 bytes.
- Validation-only preflight rederived packet lock/index/receipt, all
  5 components/123 logical groups/129 leaves/39,299,500 bytes, model provision,
  and evaluator closure
  `6d4ac61c5700083d3b8b4914174dc51e49fe4100c6aad733c966b4671242bdc7`
  without starting a worker or model and without creating scratch or a report.
- The pre-correction packet-scale run failed closed at the generic 32 MiB
  cumulative ceiling and published no report. Its evidence was preserved. The
  fresh one-case diagnostic completed cleanly. The correction kept generic
  32 MiB unchanged and set only exact schema-10 mobile workers to 38 MiB; final
  worker/test SHA-256 values are
  `78aed87f96f7eaf4d9f9038e7119ffb0ee945c015bcf3fbec96dee3fb01380b2` /
  `53a6f8e01e243b151cab7ef18a232723c2d1279d5d81d25043413495b7a71c6c`.
- The successful 129/129 CPU-only run used nice 19, idle I/O, one numeric
  thread, a 25% CPU quota/weight 1, zero swap, and a 45-minute outer runtime
  maximum. It completed in about 202 seconds wall with 50.485 CPU-seconds,
  362.8 MiB peak memory, zero swap, no restart, and no OOM. The transient unit
  is inactive/dead and collected with no remaining process.
- Publication followed worker close and fully-stopped proof, host-lock close,
  scratch/source/all-129-input revalidation, atomic link, strict report reload,
  and close verification. The retained report is 9,792 bytes, mode 0600,
  single-link, SHA-256 `264f4396…e796`; retained scratch remains private mode
  0700 and was intentionally not deleted.
- Independent aggregate audit rederived every endpoint identity, five-domain
  metric, binding, authority flag, privacy rule, mode/link count, and report
  hash through the strict loader and close verifier without rerunning the model.
- No network, GPU, CUDA, Flutter plugin, APK/emulator, microphone, audio device,
  physical phone, native Android path, thermal measurement, or live validation
  ran. The report supplies no pooled metric, quality/default/promotion/
  qualification decision, held-out/training-disjoint status, or latency/RTF
  authority.
- Freeze the seven durable docs and two ignored task records only after
  STATUS100, accepted-ADR stability, link/marker/whitespace, native-patch replay,
  ordered inventory/hash, and independent wording gates pass.
