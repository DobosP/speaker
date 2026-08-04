# ADR-0126: Bind private diagnostic case labels

Date: 2026-08-05
Status: accepted

Refines: ADR-0108, ADR-0124, ADR-0125

## Decision

Require selection-receipt schema v2 for every private diagnostic corpus-v3
publication and consumer. The closed receipt carries its source label-file
SHA-256, the diagnostic-manifest SHA-256, the existing selected-input receipts,
and a domain-separated `case_binding_sha256`; it carries no reference text,
case ID, filename, or tag. Compute that binding from canonical JSON over the
strictly increasing input indexes and the exact ordered emitted case surface:
input digest, ID, filename, PCM digest and sample count, reference text,
assertion, commands, and tags. Preserve strings and list order verbatim.

Make the shared corpus loader reconstruct and verify that binding after loading
schema-v3 cases, and make its snapshot verifier repeat the check. Keep the
recorded evaluator's stronger diagnostic-manifest, PCM, 16 kHz, role-tag,
private-metadata, and close-time checks. Reject receipt v1 in every schema-v3
consumer, and include the shared receipt implementation in the isolated
streaming evaluator's source digest. Migration is a no-overwrite export from
the original diagnostic bundle and original mode-0600 label file; never infer
a v2 receipt from an old published corpus.

## Context / why

Receipt v1 bound the selected exact PCM receipts and roles but not the published
case ID, reference text, non-role tags, or their order. Editing only
`corpus.json` could therefore change WER or an `expected-tool.*` route label
without invalidating the source-set receipt. The raw label-file digest stored
in provenance was an opaque value after publication and did not close that
edge. Accepting v1 for non-route evaluation would retain the same hole for WER,
so a compatibility mode is not evidence-safe.

## Consequences

Ordinary or inconsistent post-publication edits now fail in the shared loader
before model setup and again at snapshot verification before report
publication. The exporter, generic streaming benchmark, model workers, and
recorded final-selector/tool-route evaluator share the same closed binding
algorithm. Public schema-v2/v4 corpora, legacy WAV replay, runtime models,
tools, entry points, and capability authority are unchanged.

The receipt remains transcript- and tag-free, but old private corpus-v3 output
is historical and unusable unless it can be re-exported from its original
sources. This self-contained unkeyed hash graph does not authenticate against a
party deliberately rewriting the corpus, receipt, and every digest
consistently; that would require an externally retained corpus digest or a
signature. Headless tamper tests provide no model, GPU, microphone, device,
latency, tool-execution, or live evidence.
