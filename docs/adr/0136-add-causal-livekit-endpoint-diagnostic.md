# ADR-0136: Add a causal LiveKit endpoint diagnostic

Date: 2026-08-05
Status: accepted

## Decision

Add an explicit, private, aggregate-only causal endpoint diagnostic for the
exact LiveKit English EoT shard admitted by ADR-0134 and the exact Smart Turn
v3.2 CPU artifact corrected by ADR-0135. Keep lexical endpointing as the
runtime default and keep the diagnostic outside every app, tool-authority,
capture, and live session entry point. Require the pinned 400-row Parquet
source, its exact aggregate inventory receipt, the pinned 8,679,182-byte Smart Turn model at
revision `f766f81d3cfdf7737ac64aad813d91bbfd56bf93`, the canonical production
`config.json`, a separately validated PyArrow 25 virtual environment, new
absolute private scratch and report paths outside recognized Git worktrees,
CC-BY-4.0 acknowledgement, and explicit acceptance of the partial-availability
assumption.

Move the live Sherpa detector-admission branch into the typed,
aggregate-safe `observe_turn_completion` seam in `core/endpointing.py`, and
compose it with the existing policy through `evaluate_turn_completion`.
Preserve the branch order and live acoustic fallback exactly: a hard boundary,
missing policy/detector, empty partial, early guard, insufficient audio window,
or detector exception does not call or does not trust semantic scoring as
appropriate. Keep detector exceptions typed in the observation; Sherpa logs
and falls back acoustically, while the diagnostic fails rather than counting a
model error as acoustic evidence.

Materialize only `audio`, `language`, `duration`, and `silence_spans` in the
isolated PyArrow worker. Never load publisher IDs, words, or messages, and never
treat them as STT truth. Convert the already validated embedded WAV bytes to
private ordinal PCM16 files and retain the exact sample-domain labels. Score
each publisher silence causally from its start on the official 1,600-sample
(100 ms) grid. Exclude a HOLD resume boundary, so only a commit strictly before
the HOLD length is an early cut. For EOT, include the exact publisher-labelled
silence end when it is off-grid, stop there, and never score the six rows' one
unlabelled residual audio sample. Do not synthesize a zero tail.

For an EOT with no commit by its labelled end, report right censoring with the
labelled silence duration as the lower bound. Also report a conservative
all-row delay summary that substitutes the production 25,600-sample maximum
wait for censored rows. Publish the observed commit count, censored count,
observed publisher-labelled delay, censored lower-bound delay, and conservative
all-row delay. Preserve the full 850-HOLD/400-EOT semantic counterfactual under
`semantic_counterfactual_full_source`. Separately report
`publisher_labels_fully_before_rule3`, containing only whole labels whose end
is strictly before rule 3; this is a conservative label subset, not a
runtime-reachable or tick-level rule-3 simulation. Publish exact aggregate
counts for labels crossing rule 3 and labels starting at or beyond it because a
crossing label can contain earlier reachable decision ticks.

Evaluate two profiles. The candidate profile supplies one fixed opaque
nonempty sentinel at every scored span, representing a same-epoch streaming
partial without retaining or inventing text. The no-partial profile supplies
an empty partial, loads no PCM, performs no conversion, and never invokes the
detector. Allocate an immutable owning audio-prefix copy only at a tick that is
actually model-scored; a pending audio-window tick must not copy PCM or expose
future samples. Start each labelled span from a fresh policy seeded only with
prior publisher-HOLD pauses whose oracle outcome was resume. Treat later spans
after any predicted cut as independent counterfactual benchmark branches.

Run materialization and model evaluation as separate single-threaded workers
through verified executable, code, source, model, request, result, and
directory descriptors. Preserve each virtual environment's lexical `argv[0]`
while binding its root, executable, and `pyvenv.cfg` identity before and after
execution. Use a credential-minimal diagnostic worker environment with CUDA
hidden. Workers request finite RLIMIT soft ceilings for CPU, address space,
file size, and descriptors, plus a supervisor wall timeout; applied RLIMIT
values are not returned, and inherited hard limits may reduce the requests.
Record that no network namespace or cgroup receipt exists. Require 512 spare
parent file descriptors before scratch creation. Create and retain descriptors
for every run-owned directory and leaf before a worker may fill it, including all 400 PCM
files and the manifest; reject every unknown, replaced, or unowned scratch
entry and remove only descriptor-bound inodes. Publish the final mode-0600
report without overwrite by hard-linking the retained temporary inode through
Linux procfs, never through its mutable temporary name, and use the retained
inode identity for validation and rollback.

Emit aggregate counts, metric summaries, immutable source/model/config/code
bindings, bounded-execution receipts, and limitations only. Never emit paths,
rows, IDs, PCM, transcripts, messages, words, per-span decisions, or model
exceptions. Fix `quality_verdict` to `diagnostic_only`, `evidence_class` to
`real-source-offline-endpoint-counterfactual`, and
`production_runtime_evidence` to false. Here, `offline` in the evidence-class
label means prerecorded counterfactual evaluation, not enforced network
isolation. Do not infer Sherpa VAD, recognizer, rule-3, capture, recovery,
inference latency, microphone, AEC, barge-in, enrollment, owner voice, tools,
LLM, TTS, device, or live-conversation evidence.

## Context / why

ADR-0135 invalidated the prior Smart Turn short-turn scores and required a
corrected causal replay before any threshold or default could be reconsidered.
ADR-0134 admitted a suitable endpoint-only source, but intentionally stopped at
aggregate source validation. Feeding complete rows to the model would expose
future speech, adding artificial silence would change the publisher boundary,
using words/messages would manufacture unsupported STT labels, and scoring the
raw audio tail would turn six quantization residuals into unequal test data.

The production endpoint policy is stateful. Reusing predicted branch state
after an early cut would make later source labels depend on an event that never
occurred in the publisher conversation; ignoring prior resumed pauses would
omit the adaptive floor actually available at that point. Fresh per-label
counterfactual branches seeded from oracle-resumed prior HOLDs make that
assumption explicit. Likewise, labels crossing the long-utterance rule cannot
be represented honestly by a whole-label runtime subset, because some of their
ticks precede the rule. The strict fully-before subset and the separately named
full counterfactual prevent either view from being mistaken for production
reachability.

Private materialization is substantially larger and more failure-prone than
the final aggregate report. A worker can time out or fail after writing only a
prefix of the 400 rows. Parent-preowned leaves and retained descriptors allow
deterministic cleanup without trusting a post-failure directory walk, while
descriptor-bound publication prevents a same-name replacement from ever being
briefly published or deleted during rollback.

## Consequences

Headless tests can now prove causal grid/boundary/censoring math, immutable
prefix ownership, no-partial detector abstention, shared Sherpa/helper parity,
strict aggregate schemas with cross-summary count/sum/min/max, no-censor
equality, and percentile-order validation, private descriptor transport,
bounded failure cleanup, no-clobber publication, and replacement-race handling without loading
the exact dataset or model and without opening an audio device.

After that validator hardening, the exact diagnostic completed on 2026-08-05.
Its mode-0600, single-link, 13,966-byte aggregate report has SHA-256
`ee07bfbe0f1b16a4fafee44c4f4901a3799adcdf7bffa46ad8df775ce833e9e2`;
scratch cleanup completed. The full-source candidate cut 57/850 HOLD labels
across 32 rows, versus 93/850 across 49 rows for the no-partial acoustic
fallback. Candidate EOT committed 350/400 by each publisher-labelled silence
end and right-censored 50. Its conservative all-row delay was 600--1,600 ms
with p50 700 ms and p95 1,600 ms; the 350 observed commits had p50 700 ms and
p95 900 ms. The acoustic fallback committed all 400 EOT labels at 800 ms.

On publisher labels wholly before rule 3, the candidate cut 46/646 HOLD labels
and committed 300/332 EOT labels, with 32 right-censored and conservative p50
700 ms/p95 1,600 ms. The execution used CPUExecutionProvider with ONNX Runtime
1.27.0, NumPy 2.4.6, and PyArrow 25.0.0. Its pre-landing config SHA-256 is
`435a7dbda7aaad151e4b7d77ae0fba7f3ea544b75c9d2df543ef415740c1c811`,
byte-identical to `main`; the landed command continues to use the canonical
main config path.

This is a conditional diagnostic tradeoff, not a quality verdict: fewer HOLD
cuts coexist with publisher-boundary censoring and maximum-wait substitutions.
The fixed opaque-partial counterfactual and no-partial fallback are not a live
A/B. There is no threshold or model promotion and no runtime-default change.
Fresh owner recordings and bare-speaker live A/B remain mandatory before any
endpoint behavior or default is changed.
