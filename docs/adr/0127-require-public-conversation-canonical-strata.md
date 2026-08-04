# ADR-0127: Require public-conversation canonical strata

Date: 2026-08-05
Status: accepted

Refines: ADR-0109, ADR-0113

## Decision

Add a separate development-only public-conversation qualification entry point.
Require exactly one explicitly labelled baseline worker and one candidate worker
with distinct validated manifest digests, validate the existing four private
fixture corpora, and derive the one accepted
per-corpus stratum plan from the locked source recipes and ordered slots. AMI
uses two explicit marginal dimensions: window kind and close/far channel. Before
starting either worker, require every case to carry exactly its canonical slot
membership and no sibling canonical tag. Run the unchanged validated suite as a
strict baseline-then-candidate, four-corpus cross-product with no caller stratum
override, bind and recheck the additive controller's own source digest, and
revalidate the returned fixture summary, stream/stratum config, corpus/worker/
report/binding/set/matrix digests, case counts, exact 2x4 order, success-state
consistency, and aggregate privacy. Retain the inner suite checkpoint only
under the private scratch root. Keep scratch and output outside every Git
worktree, bound corpus tree, and bound worker provision/runtime/model tree.
Stage and file-sync the outer report privately, then publish it to a separate,
previously absent mode-0600 path with atomic no-clobber semantics.

Label a green result as complete coverage only, with no quality decision and no
promotion authority. Keep model-quality thresholds, resource acceptance,
speech-end truth, runtime defaults, tools, and live behavior outside this entry
point.

## Context / why

The generic suite already supports per-corpus tag maps, but the exact
public-conversation command exposed only global tags. No global tag list can
represent Harper task types, EdAcc speech conditions, AMI window/channel
dimensions, and Common Voice duration bands at once. A run could therefore
finish all 96 cases while silently omitting the breakdowns needed to interpret
the result. Its existing `ok` field means coverage completed, not that a model
met an accuracy or latency threshold.

Changing the existing fixture validator or any source materializer merely to
wire this caller would also change the preparer closure bound into retained AMI
and Harper receipts. An additive controller uses the existing library seam
without invalidating those artifacts.

## Consequences

The future four-source comparison has stable, source-specific aggregate strata
and fixed baseline/candidate role order. Malformed lock layouts, missing or
cross-contaminated case tags, and conflicting private paths fail before model
execution. A failed outer integrity or publication check cannot leave a raw or
outer green report at the advertised output path. The outer report remains
path- and transcript-free and explicitly non-promotional. Existing fixture
receipts, corpora, suite semantics, model workers, runtime selection, and app
entry points are unchanged.

This does not choose a model. EdAcc and Common Voice acquisition still require
owner term acceptance, the complete comparison remains unrun, and a separate
predeclared policy must define accuracy and resource thresholds before any run
can become qualification evidence. Headless tests use generated private fixture
grammar only and provide no model, GPU, microphone, device, endpoint, or live
evidence.
