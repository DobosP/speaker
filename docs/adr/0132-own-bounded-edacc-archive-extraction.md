# ADR-0132: Own bounded EdAcc archive extraction

Date: 2026-08-05
Status: accepted

Refines: ADR-0113, ADR-0131

## Decision

Make the EdAcc materializer own production extraction from the pinned archive
into an absent `edacc_v1.0` directory under an owner-private mode-0700 parent.
Read the gzip stream once through the retained archive descriptor, hash every
compressed byte, and parse fixed 512-byte tar headers before reading member
payloads. Accept only strict ASCII names, exact regular files and directories,
the fixed root/dev/test/data layout, bounded metadata, and one-level
`EDACC-C[0-9]{2,4}.wav` audio members. Reject extension headers, links, sparse
or special entries, base-256 fields, invalid checksums or padding, duplicates
and case-fold collisions, unexpected or missing members, excessive members or
expansion, concatenated gzip streams, and nonzero or excessive tar tails.

Create the fixed directory tree through retained no-follow descriptors. Create
each file with exclusive no-follow semantics, force mode 0600, hash while
writing, fsync it, reread and hash its persisted descriptor, and retain its
stable identity. Before continuing, fsync and rebind
the parent, root, subdirectories, every extracted file, and the archive;
enumerate the exact tree; bind all fixed metadata to archive member hashes;
and require disjoint dev/test recording sets whose union exactly equals the
WAV stems. Keep a failed private partial extraction as non-authoritative
evidence and refuse same-path reuse. Permit a caller-provided existing tree
only behind `TestSourceInjection`, which remains non-production.

## Context / why

ADR-0131 closed the source, archive, Git, and publication races but deliberately
left real acquisition blocked: the command trusted caller extraction and its
archive comparison ignored unused members. Standard `tarfile` extraction is
not an adequate trust boundary here. Python may consume PAX or GNU extension
bodies before a caller sees the resolved member, and extraction filters do not
by themselves bound expansion, duplicates, case collisions, or concurrent
filesystem changes.

A strict raw-header reader rejects extension types before their declared bodies
can allocate or write. A one-pass private extraction avoids a second roughly
six-gigabyte compressed read while still withholding all authority until EOF,
gzip integrity, publisher MD5, local SHA-256, exact layout, fsync, and identity
checks pass. A failed run consumes a fresh private path but cannot publish a
corpus or escape the fixed descriptor map.

The publisher README establishes the broad layout, while the current source
parser and independent Lhotse recipe establish `segments`. The repository has
not observed the exact pinned archive inventory, and the README does not list
the currently required `stm.filt`. Therefore generated archives validate the
mechanism but cannot establish real-archive compatibility.

## Consequences

Production can no longer bypass extraction with a prebuilt directory, and the
archive is consumed once to produce its MD5, local SHA-256, complete member
manifest, extracted content hashes, and private tree. Existing synthetic tests
may continue injecting a direct source tree, but that route returns
`production_evidence=false`.

The EdAcc preparer code digest changes again, so future EdAcc materialization
must use this exact closure. The shared corpus writer and other public-voice
preparers remain byte-identical.

No pinned real archive was downloaded, scanned, or extracted. Its exact layout
and disk requirement remain unverified, and no corpus, model, GPU, network,
microphone, audio device, WER, latency, tool, or runtime-default evidence
follows from the generated headless tests. The first deliberate real run must
be treated as inventory validation and fail closed rather than widening this
allowlist automatically.
