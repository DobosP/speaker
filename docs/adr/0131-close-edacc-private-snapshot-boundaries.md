# ADR-0131: Close EdAcc private snapshot boundaries

Date: 2026-08-05
Status: accepted

Refines: ADR-0113, ADR-0120

## Decision

Bind the EdAcc source root, archive, publication parent, and published corpus
to the exact owner-private filesystem objects accepted at preflight. Require
the source root to remain an owner-owned mode-0700 directory and the archive to
remain an owner-owned, single-link mode-0600 regular file. Rebind every used
metadata and audio file before and after source reads, then rebind the complete
set and archive again after publication and validation. Treat device, inode,
type, mode, owner, link count, size, and mutation timestamps as identity where
applicable. Keep the existing archive payload hash and member checks; the new
rebinds are metadata-only and do not add another 5.9 GB read.

Probe every production-path Git ancestor through no-follow directory
descriptors. Rebind the lexical ancestor around the probe, reject abnormal
markers, treat a stable empty regular `.git` file or directory without `HEAD`
as inert, and fail closed if either marker grows or changes during inspection.

Publish through the exact destination-parent descriptor retained from
preflight. Create and fsync a new mode-0700 output through that descriptor,
write mode-0600 files with the established corpus-writer serialization and
readback helpers, then repeatedly verify the parent, output directory, exact
file set, file modes and identities, receipt bytes, schema-v2 manifest,
provenance, case surface, PCM, and canonical production fixture binding. Keep
failed private publication directories as evidence and never admit them merely
because they exist. Keep this parent-bound publisher local to the EdAcc
materializer; leave the shared corpus writer byte-identical so retained Harper,
AMI, and Common Voice preparer receipts preserve their code hashes.

Do not call the real EdAcc acquisition complete. The current command accepts a
caller-provided direct extraction and checks only the archive members it uses;
it does not own extraction or validate the complete archive layout. Require a
separate descriptor-bound full-layout scan and safe extractor before unpacking
the real archive. That follow-up must reject traversal, links, sparse or special
entries, duplicates and case-fold collisions, unexpected members, and
unbounded total expansion.

## Context / why

Adversarial generated fixtures showed that the prior materializer could accept
source/archive mode changes after preflight, destination-parent replacement,
published-file mode or content drift, and mutations of metadata, selected WAV,
or archive objects during late path resolution. An empty regular `.git` marker
could also become a worktree marker inside its probe without detection. Those
races allowed the evidence object validated at one stage to differ from the
object consumed or returned at another.

Moving the publication-parent contract into the shared corpus writer would
change a file whose exact SHA-256 is embedded in all four source preparer
receipts. That would invalidate retained Harper and AMI evidence for an
EdAcc-only closure. An EdAcc-local wrapper can retain the shared serializer and
bounded file primitives while avoiding that unrelated provenance rollover.

The official archive is large, and CPU/RAM are reserved for other work. The
existing one-pass archive hash, streaming tar comparison, and one hash per
selected source WAV already bind content. Repeated exact metadata identities
close the later read windows without another archive or audio hash pass.

The archive comparison verifies required payloads but ignores the remaining
layout, while extraction occurs outside this command. Consequently it cannot
yet prove that unpacking is traversal-, link-, sparse-, collision-, or
expansion-safe even when preparation itself later fails closed.

## Consequences

Generated EdAcc preparations now fail closed on the reproduced source, Git,
parent, and output races while preserving the public function signature, CLI
safe-result fields, corpus schema, receipt schema, selection, and decoder
contract. Synthetic source injections remain explicitly non-production.

The EdAcc preparer code digest changes, so any earlier EdAcc preparation would
need fresh no-overwrite materialization. No real EdAcc corpus was previously
claimed. The shared writer and the other three source preparers are unchanged,
so their retained code closures are not rolled over.

No real archive was extracted or materialized, and no model, GPU, network,
microphone, audio device, WER, latency, tool, or runtime-default evidence was
produced. Real EdAcc acquisition remains blocked until the full archive layout
and extraction gate above is implemented and independently reviewed.
