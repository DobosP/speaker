# ADR-0101: Add a bounded Common Voice spontaneous STT slice

Date: 2026-08-01
Status: accepted

## Decision

Add Common Voice Spontaneous Speech 4.0 English as an evaluation-only source
and prepare one deterministic private P0 slice without downloading from the
repository tool. Pin the full 2026-06-12 release, MDC dataset ID, archive name,
522,005,930-byte size, public-statistics SHA-256, current MDC consumer terms,
and a matching algorithm-labelled MDC API content receipt. Treat 2026-06-12 as
the corpus release date and 2026-06-17 separately as the MDC card publication
date. Independently hash the local mode-private archive through the same
descriptor both before and after parsing.

Parse the exact 5,679-row main TSV and its published split/duration/speaker
statistics, but never read the reported-audio TSV body or decode unallowlisted
audio. Reject traversal, links, sparse/special/duplicate members, unexpected
roots, mutable inputs, malformed fields, and train/dev/test speaker overlap.
Select exactly 32 clean, unannotated test clips from 32 speakers: eight each in
the [2,6), [6,10), [10,15), and [15,25] second bands. Rank speakers separately
from their clips and use deterministic quota matching so prolific contributors
gain no extra selection chances and cross-band speakers cannot make a feasible
assignment fail greedily. Bind the structured
filter, selector, eligible set, selected order, forbidden speaker sets, literal
transcript hashes, compressed inputs, decoder closure, and decoded PCM hashes
in a transcript/path-free private receipt. Carry its digest in the existing
schema-v2 streaming-corpus provenance.

Decode selected MP3s directly with one-thread PyAV into finite mono 16 kHz
little-endian float32. The optional evaluation environment pins PyAV and NumPy
separately from runtime/lean CI. Bound every converted frame and the final
corpus before allocation/publication. Hash the PyAV, bundled FFmpeg, NumPy, and
all local preparer code used for the decode before work, then require the same
closure after publication. Custom decoders are available only through a typed,
unmistakably test-only seam. Keep configuration, setup, `./live.sh`, and all
runtime STT defaults unchanged.

## Context / why

The prior 14-case MInDS slice is short scripted banking speech and did not
exercise natural open-ended turns. Common Voice SPS v4 supplies current CC0
spontaneous speech, but the English archive also retains reported audio and
free-text reports, including material reported for PII. The current MDC SDK
uses the API checksum for resume bookkeeping but does not independently verify
the completed local file before publication, so SDK completion is insufficient
integrity evidence.

The official English test split is 377 clips and 5,430,960 ms. Materializing it
as the current eager f32le corpus would require about 348 MB of PCM and compete
with owner-critical RAM. The four-band slice has a declared worst case of
28,672,000 bytes, below the existing 32 MiB corpus limit, while retaining
speaker and turn-length diversity. It is explicitly a clean P0 development
slice, not the complete official test result. Quality-tagged and annotated
rows remain future strata rather than being silently scored or discarded from
a claimed official aggregate.

Sources: the pinned [SPS v4 format](https://github.com/common-voice/cv-dataset/blob/f99d8239d2796131b73ac99f92ee7cb4443bf3ba/datasets/spontaneous-speech/README.md),
[English release statistics](https://github.com/common-voice/cv-dataset/blob/f99d8239d2796131b73ac99f92ee7cb4443bf3ba/datasets/spontaneous-speech/sps-corpus-4.0-2026-06-12.json),
[MDC English dataset card](https://mozilladatacollective.com/datasets/cmqialpeo0077nr077xqdqo0j),
[MDC consumer terms](https://mozilladatacollective.com/terms/consumers), and
the pinned [loader schema](https://github.com/Mozilla-Data-Collective/dataset-schema-registry/blob/0fc0999776f4262b8acbe18e2ba2bd75fa5778e5/registry/cmqialpeo0077nr077xqdqo0j/schema.yaml).

## Consequences

Candidate recognizers can now share a reproducible natural-speech PCM slice
without retaining raw corpus data in Git or exposing transcripts, raw
identities, paths, credentials, download tokens, or reported comments in
receipts/stdout.
The Linux/POSIX preparer requires canonical no-symlink paths, an owner-only
mode-0600 archive, a new output path, and no recognized Git worktree marker in
either path's ancestry. Preparation is slower because it hashes the 498 MiB
archive twice and the bounded decoder closure twice, and it deliberately fails
when a balanced 32-speaker assignment is impossible. Publication is
fail-closed/no-overwrite;
failed partial evidence is retained and is not an atomic-readiness signal.
Real-archive compatibility remains unclaimed until the owner acquires the
archive under MDC terms and runs the preparer.

This is after-PCM English STT development evidence only. It has no Romanian
coverage, capture/VAD/AEC evidence, word timing, overlap labels, semantic EOU
ground truth, tool/agent grading, live latency, owner voice, or guaranteed
model-training disjointness. Promotion still requires complete/quality-tagged
public strata, disjoint private owner recordings, multi-device resource tests,
synchronized capture/AEC evidence, and a fresh bare-speaker live A/B under
ADR-0092 and ADR-0100.
