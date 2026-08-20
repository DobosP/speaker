# ADR-0208: Accept the MobileWhisper endpoint-epoch comparison as evidence only

Date: 2026-08-20
Status: accepted
Refines: ADR-0207
Supersedes: none

## Decision

Accept one exact, offline, evidence-only comparison between ADR-0207's
mobile-hybrid Zipformer baseline and a sherpa-onnx Whisper base.en INT8
candidate. Accept the comparator contract, its successful retained schema-v1
aggregate report, and the mixed component observations below. Do not select a
winner or authorize runtime wiring, a model/default change, or promotion.

ADR-0207 remains the sole owner of the 198 reset-committed endpoint epochs,
their order and boundaries, and every application/control decision. The
candidate receives exactly one final-only, complete-PCM request with zero
evaluator- or adapter-appended PCM padding for each inherited endpoint epoch,
including all 59 baseline epochs
whose literal adapter-final text is empty. It receives zero requests for a
source leaf with no endpoint. It cannot create, delete, merge, split, or move an
epoch. Its 198 outputs are reduced only into five separate aggregate metric
domains and comparison counters; candidate application admissions remain zero.

Bind candidate model
`sherpa-onnx-whisper-base.en-mobile-int8-v1`, adapter
`sherpa-onnx-mobile-whisper-final-v1`, and source revision
`59eea950fc76df2453efb57e6c0fd334548e8ffe`. The 3,314-byte committed lock has
SHA-256 `0fac7fee4e4b73f29b23130e1ee449501c4324211bae9e5da888b10483a3e496`
and canonical recipe SHA-256
`aeb7b5bbba9c558120bf68ac7a9d7dca671e75115c2f42d8897bfa31153b6087`.
Its three-file artifact set has SHA-256
`3a3e907d65aced6f76a40afaabee86c12e7d281de3397bbfed33ad90f227af34`
and exactly 160,626,066 bytes:

| Artifact | Precision | Bytes | SHA-256 |
|---|---|---:|---|
| `base.en-encoder.int8.onnx` | int8 | 29,120,534 | `ef6b936f4c9b1d90a3b68634b60c4ed8576b26172b33c2535ec0e933c9edb823` |
| `base.en-decoder.int8.onnx` | int8 | 130,669,978 | `f7162ad6db2dbef16cfaeaa7f945b9d7dd9c1b8d472f6aca82f2273d185e4d41` |
| `base.en-tokens.txt` | text | 835,554 | `306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930` |

The exact candidate configuration has SHA-256
`f3555a3ffaaa2569bb12cb71c39f31883d27d759b35ec85024ca057fbc5e34f1`.
It fixes sherpa-onnx and sherpa-onnx-core 1.13.3, NumPy 2.4.6, CPU provider,
one inference thread, English transcription, 16 kHz/80 features, greedy search,
no token or segment timestamps, `tail_paddings=-1`, and externally owned mobile-
Zipformer endpoint epochs. The evaluator and adapter append zero PCM padding;
Sherpa's internal padding is unobserved and unreported. The schema-11 worker
uses the established final-only protocol and a candidate-only 62 MiB cumulative
decoded-request PCM ceiling: every request byte counts again even when PCM is
repeated. The generic 32 MiB and exact schema-10 Zipformer 38 MiB lifetime
unique-PCM ceilings remain unchanged. The candidate's 49,072,300 derived PCM
bytes are below that narrow schema-11 ceiling.

Provision only exact admitted bytes into a new private outside-Git root through
the hash-only, no-clobber provisioner. It imports or loads neither sherpa-onnx,
NumPy, nor the model. Accept candidate manifest/terminal-receipt SHA-256
`4195ecd3ddb5068a98620fed37232db8958f240d5ec421e78b7686fa3580ff8b` /
`8a52ee00b655bf4df5e9f34517461db4dabd060c677687e9e6f3869d5c89818b`.
The comparator separately reopens the accepted Zipformer provision at
manifest/receipt SHA-256
`3ba365a057209dce4794aa9f5bff00efbd7b8b5da03c5662b48254164fab3315` /
`7f39b1a693063955eb45d6217247832b602304ae6cb7754f955a99cf00de5248`,
reruns all 129 packet leaves, and requires an exact match to ADR-0207 before
candidate execution.

Publish only one bounded, aggregate-only, no-clobber report after both workers
close and every packet/model/source/scratch/output binding revalidates. The
accepted report has SHA-256
`0ae9424df94e284f7724818c2ba84e9ef4fcefcd99d9afec75e9efb14bd4e625`,
is 19,574 bytes, mode 0600, and single-link. Its binding SHA-256 is
`9933e6415a793afd5e8b4bda4c97f45c04189e68a87cd70f813d667898631de0`;
the 198-epoch geometry SHA-256 is
`9c4d32639a0c391785817aaf882ef8ff1776b5a5bbb1ce5520e935989f9e405a`;
the evaluator closure SHA-256 is
`b4822fc474aa5abe2d8f37b53f95b6edcbcb9cc310bd6ab3f53be2ec317a6da0`;
and the worker-source bundle SHA-256 is
`dbee283623da1d61d98a087c08e972867bb823a420452762a1b7b6e81b412f57`.
It contains no reference, hypothesis, transcript, case ID, local path, PCM,
speaker/role,
per-epoch row, raw worker stderr, or pooled accuracy metric.

Accept these observations as mixed, unpooled evidence, not a winner:

| Domain | Zipformer baseline | MobileWhisper candidate |
|---|---:|---:|
| Command targets | 18/26 | 18/26; 2 gained and 2 lost |
| Forbidden command targets | 0/355 | 7/355 |
| Command transcript-subset WER | `.6809` | `2.3404` |
| DEMAND WER | `.9149` | `.7409` |
| NOTSOFAR WER | `.4423` | `.4038` |
| NOTSOFAR CER | `.2638` | `.3216` |
| NOTSOFAR paired exact agreement | 4/9 | 1/9 |
| PriMock isolated WER | `.6154` | `.5385` |
| PriMock overlap custom minimum-order WER | `.3964` | `.1892` |

All 59 literal-empty baseline endpoint epochs became literal-nonempty candidate
outputs; the candidate has zero literal-empty epochs. Exact normalized text
agrees on only 39/198 epochs. Literal adapter-final emptiness is intentionally
separate from normalized agreement, WER, command matching, and control
projection. The control comparison has exact class agreement on 194/198 epochs,
with one candidate-gained and three candidate-lost control lexemes, while
`actual_agent_decision_authority` remains false. These counters grant the
candidate no STOP, confirm, deny, mode, tool, endpoint, semantic-turn, or
`AgentSession` authority.

The accepted execution ran in transient unit
`speaker-mobile-whisper-endpoint-comparator-20260815-v3`, invocation
`c5e2cf9a3ae64b49965b19a3756a8c09`. It used 155.771 CPU-seconds over
623.803 seconds wall, one configured inference thread, at most six cgroup
tasks at peak, and an authoritative direct live cgroup peak of 994,983,936
bytes, with zero swap and zero memory events; the unit was collected with no
survivors. Its 508-byte mode-0600/single-link log has
SHA-256 `9d7595b8791dca411c82cc979d704cfa734645c1165866dcfb3efbcfca288736`.
The systemd footer displayed `Memory peak: 2.1M`; it under-observed the active
cgroup and is not a resource measurement. No latency or RTF authority follows
from this sequential
desktop CPU run.

Upstream OpenAI Whisper code and the pinned base.en checkpoint-map evidence are
MIT-scoped, but the retained converted sherpa-onnx artifact has no conversion
license artifact, unverified artifact-license status, unverified conversion
origin revision, and no exact conversion recipe. The lock therefore records
license/provenance evidence, not redistribution clearance. Runtime identity is
bound to exact Python distribution metadata and the provision, not an audited
installed-tree closure. Sherpa's internal padding and native resource/thread
behavior remain opaque; the configured one inference thread is not a proof of
all native/OS threads.

## Context / why

ADR-0207 established exact mobile-Zipformer model execution and five unpooled
cells, while deliberately retaining its English mobile runtime/default and all
endpoint/control authority. It did not answer how the exact mobile-sized
Whisper export behaves when given those same committed epochs. A comparator
that inherits boundaries instead of running Whisper's own segmentation answers
that narrow question without changing turn geometry.

The completed official-source screen chose base.en only as the bounded first
comparator, not as a promotion or a model-family ranking. The pinned
[20M Zipformer tree](https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/tree/d42f2d9f7ca24806fb667456a18a9f1b60f70d16)
(`csukuangfj/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17` at
`d42f2d9f7ca24806fb667456a18a9f1b60f70d16`) supplied footprint-only,
LibriSpeech-only research and no evaluation evidence; this record supplies no
noisy-speech upgrade evidence for it. The pinned, much larger
[Nemotron 560 ms INT8 sherpa export](https://huggingface.co/csukuangfj2/sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25/tree/52056fdc070914a48dcd68b31b44d6a6f5b85902)
(`csukuangfj2/sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25`
at `52056fdc070914a48dcd68b31b44d6a6f5b85902`) was not the bounded first
mobile step; its upstream licensing is separate, and neither candidate is
promoted here.

For better paired data, SLURP's real close/far layout is design inspiration
only. The official [Zenodo record](https://zenodo.org/records/4274930) lists
`slurp_real.tar.gz` as 3.9 GB with MD5
`9efc0f058ced47bf5131c7cb2cade513`; the annotations are pinned at
[revision `8eb1654`](https://github.com/pswietojanski/slurp/tree/8eb16545762be97ace75334109d73824217311f1).
Its audio is CC BY-NC 4.0 while text/annotations are CC BY 4.0. Therefore do not
use its audio for product/commercial evaluation without confirmed compatible
terms. No SLURP download, acquisition, corpus preparation, run, evaluation, or
held-out claim occurred. Reproduce the paired close/far design with consented
owner recordings instead; the disjoint private owner holdout remains the actual
next evidence gate.

Two production attempts failed closed and published no report:

- v1, unit `speaker-mobile-whisper-endpoint-comparator-20260815-v1`, invocation
  `e6017fe0852a43ef92eb1e495580aa0a`, exited 2 after 9m49.294s and 147.297
  CPU-seconds. Its direct sampled cgroup peak was 860,692,480 bytes with zero
  swap; it then rejected a stale pre-candidate scratch witness. A later
  post-run audit found the unit collected, its cgroup absent, and zero exact-
  unit `/proc` survivors. Its 391-byte mode-0600/
  single-link log has SHA-256
  `d160ac95bb9c129cd4aeb2751d044cf64174317f173085dc9ee3e2b7b49b62de`.
  The systemd footer displayed the literal `Memory peak: 826.7M`; its ambiguous
  unit is not compared with the direct sampled byte peak. Exact three-child
  scratch accounting fixed the witness; retained scratch/log evidence was not
  deleted.
- v2, unit `speaker-mobile-whisper-endpoint-comparator-20260815-v2`, invocation
  `344d53f62e2b4d76b8c66cf0e7d170a2`, exited 2 after 8m45.778s and 131.415
  CPU-seconds. It reached an 877,019,136-byte direct live peak, zero swap and
  memory events, and six tasks; it was collected with no survivors. Its
  391-byte mode-0600/single-link log has SHA-256
  `ea49d3d419de9ce2950e903161a13e4f39160028698779d9481d92faf4d77844`.
  The systemd footer displayed `Memory peak: 663.2M` and under-observed the
  direct peak. Static and synthetic diagnosis found that comparison emptiness
  used normalized tokens while both
  baseline endpoint integrity and candidate component accounting use literal
  adapter-final text. The v3 fix changed only comparison empty counts and
  transitions to exact `== ''` / `!= ''`; normalized agreement, WER, command
  matching, and control projection remain normalized.

Neither failed run has acceptance or report authority. No private transcript,
hypothesis, or PCM content was inspected to diagnose either failure.

## Consequences

- The exact comparator implementation and retained v3 report are accepted as a
  reproducible, evidence-only desktop CPU slice. ADR-0207's baseline result and
  endpoint/control ownership remain unchanged.
- Lower WER in four acoustic cells coexists with worse command-subset WER,
  seven forbidden command hits, two lost command targets, lower far-field pair
  agreement, higher NOTSOFAR CER, and pervasive text on baseline-empty epochs.
  No scalar or qualitative winner may be derived from these domains.
- The report sets no default, device, held-out, latency, live, microphone,
  model-quality, promotion, qualification, or training-disjoint authority.
  Candidate endpoint, application, agent-session, semantic-turn, control, and
  tool authority remain false.
- No mobile/runtime source, Flutter plugin, Android native library, app wiring,
  model download/default, phone CPU/RSS/thermal behavior, microphone, audio
  device, or live session changed or ran. The comparator/model execution used
  no application network, GPU, or CUDA path; the official-source screen above
  is research, not runtime/model execution.
- The 20M Zipformer and 560 ms Nemotron references remain research controls,
  not evaluated or promoted mobile alternatives. SLURP remains a license-gated
  data-design reference, not an acquired or accepted product corpus.
- Before any later adoption decision, keep an owner-recorded holdout disjoint
  from tuning and run the actual Flutter/native path on a physical phone for
  CPU, RSS, thermal, microphone, lifecycle, and live behavior. Romanian remains
  a separate model/export/language/tokenizer/Unicode/evidence decision.

## Verification

- The superseding 12-path v3 native package has patch SHA-256
  `f8b10443923f071df858f080d6400e366201940a28373cef8e684e94c93a5d3a`
  (358,438 bytes, 12 sections), ordered manifest SHA-256
  `fa8ade3c35a618d76c897b8f539d4ccf54e0cc5a6c9da207b5ad103e4970259e`
  (1,862 bytes, 12 paths/829,207 result bytes), and gate-receipt SHA-256
  `79ff88eabd389dd64f723902d8dbc562ac1004188b45778e08da5096b4be9f9e`
  (8,528 bytes). A fresh plain shadow from exact base `31ada83` applied it
  natively as six added/six modified paths; every replay/task hash and size
  matched and `git diff --check` was clean.
- Fresh replay gates passed 85 focused; 150 composed plus one opt-in skip; 354
  adjacent plus one opt-in skip; 6 APM/DTD; and 5 targeted publication-failure
  tests with 80 deselected and two inherited timeout-configuration warnings.
  Ruff passed all 11 Python paths. The two changed evaluator/test files are
  format-clean; seven provision/runtime paths unchanged by the v3 literal-empty
  fix keep inherited formatter debt. No package gate read private PCM/
  transcript content or ran a
  model or network path.
- Punctuation-only, whitespace-only, and non-ASCII-only regressions cover both
  raw-nonempty sides, a four-pair raw-versus-normalized matrix, and baseline/
  candidate cross-binding tamper. An independent 49-pair synthetic publish/load
  matrix and two independent full-package/schema/receipt audits are GO.
- Strict retained-report audit rederived all aggregate metrics, authority
  flags, bindings, privacy exclusions, report hash, mode, and link count. It
  does not create phone/live/default authority.
