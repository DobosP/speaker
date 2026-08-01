# ADR-0113: Bind a private 96-case public conversation/STT fixture

Date: 2026-08-02
Status: accepted

## Decision

Extend the typed public-data catalog to matrix v4 with HarperValleyBank at Git
commit `0bd721e877c4a85d8c13ff837e68661ea6200a98`, the corrected EdAcc v1.0
deposit at DOI `10.7488/ds/7914`, and synchronized AMI ES2004a close/far audio;
add CC-BY-SA-4.0 as an explicit license gate. Commit one canonical,
self-digested logical lock with 96 abstract slots, but keep audio, literal
references, selected identities, local paths, and per-source receipts outside
Git. Require four separate owner-private streaming-corpus schema-v2 corpora,
exactly 24 cases per source, and validate each fixed sibling receipt against the
catalog contract, selection recipe, accepted terms, artifact receipts,
preparer/decoder closure, ordered case hashes, reference hashes, and every PCM
sample before and after a run. Evaluate the four corpora sequentially through
the existing suite; do not concatenate them or widen its 32 MiB eager-PCM cap.
The public fixture entry point owns the validation/run/revalidation envelope;
a separately executed validator followed by an unbound suite command is not
fixture evidence.

Select HarperValleyBank as three lexical human caller turns from each of eight
authoritative task types with 24 distinct callers; exclude metadata names,
recognizable dates/clocks, and known marker-only noise/paralinguistic
references, and never use its machine transcript, dialog-act, or emotion output
as a reference or label. Select EdAcc from official test
human references as eight clean, eight disfluent, and eight overlap-adjacent
segments with distinct speakers. Use raw human text/STM agreement to bind the
source and classify acoustic disfluency, but score the key-matched official
filtered STM only; exclude actual ambiguous overlap, STM ignore segments,
empty or alternative filtered references, and noise/paralinguistic-only
references from hard WER, and keep accent subgroups diagnostic. Select twelve
AMI windows (eight isolated and four non-overlapping transitions) and emit each identical
sample interval/reference once from close audio and once from far audio; actual
overlap stays diagnostic-only. Preserve the accepted 32-case Common Voice
Spontaneous Speech v4 preparer and derive the fixture as six cases from each of
its four duration bands with distinct speaker and prompt digests. Source
materializers must recompute evidence from the exact Git checkout or
archive/audio inputs. The Common Voice projection must bind its parent receipt
and corpus manifest and rehash the archive metadata and selected MP3s. AMI
Mix-Headset remains bound by an explicit local SHA-256 receipt; until a shared
upstream/global pin is accepted, its close-channel identity is locally
reproducible rather than cross-machine predeclared.

## Context / why

The previous public comparisons mixed many useful sources but did not provide
one small, reproducible conversation-heavy contract for testing new streaming
and final STT configurations. A single 96-case corpus would exceed the current
loader's deliberate eager-audio budget and make one source's preparation
failure invalidate every other source. Committing audio or literal references
would violate privacy/licensing boundaries, while committing source IDs would
make selection receipts re-identifying. Aggregate counts alone would not catch
changed selections, decoder closures, transcript substitutions, reordered
pairs, or PCM drift. Four receipt-bound corpora plus one identity-free logical
lock preserve deterministic comparison without broadening runtime authority.

## Consequences

Headless tests can now prove the 24x4 recipe, catalog pins, license gates,
privacy schema, source balance, close/far pairing, exact receipt/provenance
cross-binding, and single-layer tamper detection without downloading data or
loading a model. The
Common Voice preparer additionally records salted prompt digests without
changing its accepted 32-case selection. Real source acquisition/preparation
and model runs remain private and opt-in; reports are after-PCM development
evidence and are not proof of training disjointness, native capture, VAD,
endpoint quality, AEC, barge-in, speaker authority, agent/tool behavior, device
latency, or live conversation quality. Runtime defaults cannot change from this
contract alone. Promotion still requires the private exact-input and fresh live
A/B gates retained by ADR-0092, ADR-0108, and ADR-0112.

The private receipts are same-owner consistency records, not signed attestations:
a coordinated actor who can rewrite a corpus, receipt, and all bound hashes can
reauthor the record. Raw-source materializer injections exist only for
deterministic headless tests; they must mark decoder/source evidence
non-production, and the production fixture validator must reject their output.
Production-shaped validator grammar fixtures may exercise acceptance and
tamper paths, but are never source or preparation evidence.
