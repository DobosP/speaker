# ADR-0139: Accept bounded EdAcc base-256 owner fields

Date: 2026-08-05
Status: superseded-by ADR-0140

Supersedes: ADR-0132

## Decision

Retain ADR-0132's production-owned, descriptor-safe, bounded extraction and
all of its path, type, layout, expansion, persistence, and publication rules.
Refine only tar numeric decoding: permit the standard positive base-256 form
in the eight-byte UID and GID header fields, require a `0x80` marker followed
by a big-endian value from `8^7` through `2^31 - 1`, and discard the parsed
ownership value after structural validation. Values below `8^7` remain octal
because they fit the ordinary seven-digit field. Never apply archive ownership
to an extracted object or include it in selection or evidence authority.

Continue to require octal encoding for mode, size, modification time,
checksum, and device-major/minor fields. Reject negative `0xff` base-256,
other marker forms, overflow, malformed octal, extension headers, and every
previously forbidden member or layout. Keep every extracted directory and file
owned by the running user at forced mode `0700` or `0600` respectively.

## Context / why

The first deliberate run against the exact-size pinned EdAcc v1.0 download
failed before consuming a member payload. Its first otherwise ordinary USTAR
directory header encodes UID with the standard positive base-256 bytes
`80 00 00 00 00 22 89 10`; the GID remains octal. The generated archives used
to build ADR-0132 encoded both fields as octal, so they could prove rejection
mechanics but not compatibility with the publisher's real tar writer.

UID and GID are non-authoritative here: the extractor never preserves them and
creates every output through retained descriptors as the current owner. A
broad base-256 allowance would be unnecessary and dangerous because size and
other authority-bearing fields affect allocation, stream position, and layout.
An explicit owner-field-only grammar admits the observed publisher encoding
without weakening those boundaries or changing the member allowlist.

## Consequences

- The exact pinned archive may be retried only at a fresh absent extraction
  path; the first empty failed extraction tree remains non-authoritative
  evidence and is not reused or deleted.
- Deterministic tests must cover the observed UID, positive boundaries, and
  rejection of negative, alternate-marker, overflow, and authority-field
  base-256 forms.
- A later successful complete scan must still validate publisher MD5, compute
  local SHA-256, match the fixed dev/test/data layout, publish exactly 24
  cases, and rebind every source and output object. This decision alone is not
  real-source completion.
- No corpus quality, model, GPU, microphone, capture, VAD, endpoint, AEC,
  barge-in, device, tool, latency, live-conversation, or runtime-default claim
  follows.
