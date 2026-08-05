# ADR-0129: Add a non-authoritative AMI endpoint proxy

Date: 2026-08-05
Status: accepted

Refines: ADR-0113, ADR-0119, ADR-0121, ADR-0127

## Decision

Add an AMI-only, development-only endpoint diagnostic derived from the forced-
aligned end of each selected window's last word. Require the AMI materializer
to publish a fixed owner-private `speech-end-labels.json` sibling whose exact
ordered rows bind case ID, PCM digest, sample count, relative proxy sample,
window kind, and close/far channel. Bind the sidecar's canonical-byte digest
through the existing schema-v2 provenance and receipt `metadata_sha256` field,
retain the prior annotation-metadata digest inside the sidecar, and keep the
generic corpus schema, fixture validator, four-source lock, and other source
materializers unchanged.

Load the sidecar only through a separate AMI evaluation entry point, validate
the corpus, receipt, sidecar, ordered label-set digest, and source closure before
model work, pass a typed in-memory binding to the existing isolated evaluator,
and revalidate after inference. The typed binding independently reconstructs
the exact canonical sidecar digest from its annotation digest and label rows;
the outer wrapper accepts only the built-in evaluator, validates its worker,
config, evaluator closure, denominators, and metric relations, and reconstructs
a fixed aggregate-only snapshot instead of passing nested reports through.
Support only Parakeet Realtime EOU's native
terminal EOU sample and parakeet.cpp's first feed-origin EOU observation. Score
only the 16 isolated, non-overlapping close/far cases and explicitly exclude the
eight turn-transition cases. Preserve the generic source-end endpoint metrics;
report the additional proxy deltas and missing/non-feed/tail observations only
as aggregate, path-free data.

Call this a forced-alignment endpoint proxy, never acoustic speech-end ground
truth. Define no quality threshold, model choice, promotion, runtime/default,
capture, VAD, device, tool, or live authority from this diagnostic.

## Context / why

The existing streaming metrics compare endpoint observations with the end of
the prepared PCM window. AMI windows intentionally retain post-speech context,
so that boundary answers whether an adapter ended before all supplied PCM, not
how its EOU relates to the last transcribed word. A separately bound label
surface permits the latter diagnostic without changing the established metric.

AMI's orthographic transcripts were manually produced, but its official
[corpus overview](https://groups.inf.ed.ac.uk/ami/corpus/overview.shtml) and
[transcription documentation](https://groups.inf.ed.ac.uk/ami/corpus/transcription.shtml)
state that word- and phoneme-level timings were generated automatically by
forced alignment. The official
[signal documentation](https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml)
also describes the distributed 16 kHz audio and the energy-based segmentation
used to aid transcription. A last-word timestamp is therefore a useful,
repeatable reference in the same sample domain, but it is not a human-labelled
acoustic offset and cannot justify sub-frame truth, live latency, or endpoint
acceptance thresholds.

Adding generic optional endpoint fields to schema v2 or changing the shared
fixture lock would widen every public-conversation consumer and invalidate
unrelated retained artifacts, including Harper. Reusing `metadata_sha256` for
the exact AMI sidecar keeps the existing provenance grammar fail-closed while
the nested annotation digest preserves the prior source binding. Scoring
turn-transition windows would also conflate two speakers and an intentionally
selected turn boundary with one speech end; those rows remain bound for exact
fixture pairing but outside the reducer's denominator.

## Consequences

New AMI preparations contain the mode-0600 sidecar and bind it through both the
receipt and corpus provenance. AMI corpora created by the earlier materializer
lack that binding and cannot enter this diagnostic; they must be rematerialized
from the pinned sources into a fresh private output directory because the
publisher is no-overwrite. Existing Harper, EdAcc, and Common Voice corpus
formats, receipts, locks, and validation behavior remain unchanged.

The AMI-only wrapper can compare supported adapters with aggregate signed,
early, post-proxy, and absolute sample/millisecond summaries while retaining
explicit missing-EOU and excluded-transition denominators. It exposes no raw
label rows, transcript, local path, audio, or speaker identity. Existing direct
AMI model results remain historical source-end evidence until rerun through the
new wrapper; no model run, GPU run, acoustic annotation, microphone, or live
validation is established by headless contract tests.

A future acoustically annotated, capture-aligned corpus and a predeclared
acceptance policy are still required for authoritative endpoint qualification.
