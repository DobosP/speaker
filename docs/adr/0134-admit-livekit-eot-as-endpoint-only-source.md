# ADR-0134: Admit LiveKit EoT as an endpoint-only source

Date: 2026-08-05
Status: accepted

## Decision

Admit the exact LiveKit EoT Benchmark English validation Parquet shard under a
new endpoint-source catalog v1. Pin dataset revision
`ca9d98a9686b920a2d8c9eb984224ba9be74e4dd`, shard
`data/en/validation-00000-of-00001.parquet`, 162,406,142 bytes, SHA-256
`e475d435c8e693912243e8a810bf7e34a59e01e208551ce5b362c9c778b0fdc4`,
400 English rows, and CC-BY-4.0 attribution. Scope its evaluation track to
endpoint early-cut rate and endpoint delay only. Explicitly exclude WER, CER,
overlap-WER, backchannel, and canonical-conversation use.

Keep public voice matrix v5, its fixture validator, and the self-digested
four-source conversation lock byte-identical. Require an explicit absolute
private local shard, explicit license acknowledgement, an isolated PyArrow
25.0.0 interpreter, and a new owner-private report path outside every
recognized Git worktree. Treat a stable, single-link empty `.git` file and a
stable `.git` directory without `HEAD` as inert sentinels; reject non-empty
linked-worktree files, directories with `HEAD`, symlinks, special nodes, and
any sentinel growth or replacement. Probe absence twice and use full directory
metadata during each stable no-follow Git-boundary probe, while separately
binding the long-lived publication parent by immutable directory-entry
identity. Recheck the boundary immediately before publication. Transport
the already-open exact source descriptor into a single-threaded bounded worker;
verify file identity and complete SHA-256 before and after it runs. Validate the
exact schema and all 400 embedded RIFF/WAVE containers as format-1 signed
PCM16, mono, 16 kHz, 2-byte block alignment, and 32,000-byte/s rate without
opening their flat path labels. Scan only audio, language, duration, and
silence spans; validate IDs, words, and messages through schema metadata
without loading their columns.

Treat every ordered silence before the last as publisher-convention HOLD and
the last as EOT. Convert each original finite, non-boolean publisher timestamp
independently with `Decimal(str(value)) * 16000`; every span endpoint must be an
exact integral sample. Require endpoints within the audio, strict sample-domain
ordering/non-overlap, span lengths from 1,600 through 80,000 samples, a final
span of at least 3,200 samples, and a final audio gap of exactly zero or one
sample. Quantize duration metadata with `ROUND_HALF_EVEN`; its nearest sample
must equal the WAV sample count and its absolute residual must be at most 0.008
sample, the exact six-decimal publisher microsecond-rounding envelope rather
than a floating epsilon. Bind the observed duration exact/quantized profile
(394/6) and final-gap zero/one profile (394/6) into the schema, inventory digest,
worker result, and parent validation.

Emit fixed public contract/attribution fields plus aggregate counts and digests
only. Never emit source paths, row IDs, conversation messages, word timings,
transcripts, or audio. Refuse overwrite; publish the report atomically with mode
0600. Bind every report to a fixed execution-closure map containing only the
relative names and stable SHA-256s of the parent inventory, isolated worker,
fixture helper, and bounded-I/O helper. Snapshot all four before execution,
re-read them immediately before and after publication, and require the
production worker digest to equal its closure entry. The endpoint catalog stays
outside this execution closure and is bound separately by its contract tests.
Test injection remains a typed synthetic-only seam and can never claim
production evidence.

## Context / why

The agent still needs causal, conversational endpoint evidence. Existing clean
read-speech data does not exercise mid-turn HOLD versus true EOT, while the
LiveKit release provides a bounded English split with publisher silence spans
and embedded audio. Its word timings are not established as authoritative STT
references, so adding it to WER would manufacture an unsupported quality gate.
The dataset is CC-BY-4.0; Apache-2.0 applies to the separate benchmark code, not
to these data bytes.

The legacy public matrix file participates in all four retained conversation
source closures. Even an additive dataset entry changes its digest and
invalidates those receipts and the canonical 96-case lock. A separate typed
catalog preserves that evidence boundary while making the source and metric
scope reviewable. The inventory stops before selection/materialization or model
execution because those require their own causal timing and report contracts.

## Consequences

The exact shard can now be admitted and audited without network access, raw
content publication, PyArrow in the production runtime, or collateral changes
to STT evidence. A real isolated inventory verified 400 rows, two row groups,
1,250 silence spans (850 HOLD and 400 EOT), 161,326,412 embedded WAV bytes, and
80,654,406 samples. Duration metadata is exact-sample for 394 rows and
microsecond-quantized for six; final silence ends at the audio end for 394 rows
and one sample before it for six. Its final closure-bound private aggregate
report has SHA-256
`4c6c53d6324741006fa34da7bc6ecb02069ca68e2d5efffd166661b8b2f610e5`.

A later branch must define a deterministic private causal materializer and an
endpoint evaluator before this source can grade a candidate. This decision
does not download data, enable a runtime path, run an STT/LLM/TTS model, measure
latency or accuracy, alter enrollment/barge-in/tools, or change any default.
