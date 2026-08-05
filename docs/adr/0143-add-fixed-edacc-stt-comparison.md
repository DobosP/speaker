# ADR-0143: Add a fixed EdAcc-only STT comparison

Date: 2026-08-05
Status: accepted

Refines: ADR-0098, ADR-0102, ADR-0127, ADR-0130, ADR-0142

## Decision

Add a separate receipt-bound EdAcc-only 2x1 development wrapper. Do not add a
mode to, relax, or replace the canonical four-source 2x4 qualification entry
point. Require the retained production EdAcc corpus SHA-256
`4da392c39a0b6bd18057f63c96b4f67c0dfdaf4c14e1d2d76176cdbee0920772`
and sibling preparation-receipt SHA-256
`3c516b6fbb8ce139498cd2e01d83a4faf825e58bf2b9abcaa52896bdab4c853f`.
Before starting a worker, require `production_evidence=true`, exactly 24 cases,
and the locked EdAcc marginal of eight `clean`, eight `disfluent`, and eight
`overlap_adjacent` cases.

Fix worker order to the schema-4 model
`production-gigaspeech-zipformer-fp32-benchmark-1t` through adapter
`sherpa-onnx-gigaspeech-zipformer-stream-v1`, followed by the schema-6 English
Faster-Whisper Small model `faster-whisper-small-local` through the final-only
adapter `faster-whisper-endpoint-v1`. Fix both cells to three repeats,
1,600-sample chunks, burst pacing, a 200 ms partial interval, and zero
tail-padding samples.
Delegate the two cells sequentially to the existing validated suite; do not
parallelize model workers or expose caller overrides for the corpus, roles,
order, strata, geometry, or pacing contract.

Bind and recheck the corpus, receipt, both worker manifests, resolved adapters
and stream configurations, wrapper source, evaluator/suite source, private
scratch, and intermediate checkpoint before and after delegation and before
publication. Reconstruct a fixed aggregate-only 2x1 report rather than
republishing the generic checkpoint. Exclude transcript text, paths, stderr,
worker text, and unbounded evidence. Validate retained reports strictly and
publish canonical JSON plus one newline through an identity-safe,
failure-atomic, no-clobber write. A complete report remains
`quality_decision=not_evaluated`, `development_only=true`, and
`promotional=false`.

## Context / why

The production EdAcc materializer now provides one exact 24-case real-source
corpus, while the other three inputs required by the canonical four-source
qualification are not all ready for a current model run. The generic suite can
execute a caller-selected 2x1 matrix, but that would not prove that this exact
production corpus, fixed EdAcc strata, fixed workers, and fixed stream geometry
were used. Changing the canonical 2x4 wrapper to admit a single-source mode
would weaken the evidence boundary that makes its four-source result
interpretable.

This narrow wrapper answers only the immediate recognition-accuracy question:
how the production-model Zipformer control and the final-only Faster-Whisper
Small comparator score on the same retained EdAcc slice. Its burst-mode timing
begins after PCM is available. Faster-Whisper consumes complete PCM and emits
no streaming partials, so neither cross-model timing nor this wrapper's
deadline fields represent capture, endpoint, or conversational latency.

## Consequences

- Headless tests can establish the fixed corpus/receipt pins, 24-case and
  8/8/8 closure, exact worker roles and order, immutable stream settings,
  sequential delegation, repeated pre/post binding, aggregate privacy, strict
  retained validation, and failure-clean no-clobber publication without
  loading a model.
- A private model run can compare literal/canonical WER and CER over the exact
  EdAcc accuracy slice. It cannot qualify latency, endpoint behavior, resource
  acceptance, natural conversation, capture/VAD/AEC, barge-in, commands,
  tools, owner voice, device behavior, or a runtime default.
- The worker manifests and their content receipts bind the local runtime and
  model trees consumed by the run. This wrapper does not independently prove
  their upstream repository identity, training-data disjointness, or
  equivalence to a different local tree; those remain provenance limits.
- Provision both workers from the current checkout, run them strictly one at a
  time, retain the aggregate report and its printed published-byte SHA-256,
  and keep every private corpus, receipt, manifest, scratch tree, and report
  outside Git. Promotion still requires the unchanged four-source and owner
  live gates.
