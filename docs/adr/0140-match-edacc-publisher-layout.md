# ADR-0140: Match the bounded EdAcc publisher layout

Date: 2026-08-05
Status: accepted

Supersedes: ADR-0139

## Decision

Retain ADR-0139's UID/GID-only base-256 rule and every other ADR-0132 raw
header, descriptor, path, type, size, expansion, persistence, and publication
boundary. Match the pinned publisher layout with two explicit additions:
require one root `README.txt` regular file bounded to 64 KiB, and admit audio
members named `EDACC-C[0-9]{2,4}.wav` or
`EDACC-C[0-9]{2,4}_P[0-9].wav` in the existing one-level `data` directory.
Reject multi-digit, missing, repeated, or non-numeric participant suffixes and
every other member name.

Require both the segment-derived recording sets and `conv.list` conversation
sets to be disjoint across dev/test, and require each combined set to equal the
complete WAV-stem set exactly. A syntactically valid participant-suffixed WAV
therefore has no authority unless both fixed metadata views name that exact
recording. Extract the README as owner-owned mode `0600` and bind it into the
complete archive layout, but never parse it as selection metadata or place its
contents in the preparation receipt.

## Context / why

After ADR-0139 admitted the exact first header, the second fresh production
attempt extracted the complete test metadata and 17 plain-stem WAVs, then
failed closed on the first publisher member named `EDACC-C23_P1.wav`. A
read-only, low-priority path inventory of the pinned gzip reported exactly 98
members: four fixed directories, 76 WAVs, the expected root/dev/test files,
and no other shape except `README.txt`. Of the WAVs, 46 use plain stems and 30
use a single-digit `_P<digit>` form; no multi-digit or other participant form
exists. The README is 3,517 bytes.

Those names are ordinary single path components and remain cross-checked
against the official split metadata. Permitting arbitrary suffixes, nested
paths, optional extra files, or a general README family would weaken the exact
layout without evidence. The grammar and 64 KiB bound instead cover only the
observed publisher contract with substantial headroom for the pinned README.

The path inventory used the host tar lister only as non-authoritative diagnosis
and did not inspect transcript or audio contents. It does not establish the
publisher MD5, local SHA-256, raw-header validity of every member, or a complete
production corpus. Those remain requirements of the hardened scanner.

## Consequences

- Generated archives and exact extraction tests must include the root README
  and exercise a metadata-matched single-digit participant recording; missing,
  oversized, malformed-suffix, and set-mismatch variants fail closed.
- Before another extraction, run the hardened raw scanner across the entire
  unchanged archive without an extraction destination. Only a complete
  publisher-MD5/layout success permits a third fresh extraction path.
- Preserve both earlier failed extraction trees as non-authoritative evidence;
  never reuse or delete them automatically.
- No complete source, corpus quality, model, GPU, microphone, capture, VAD,
  endpoint, AEC, barge-in, device, tool, latency, live-conversation, or
  runtime-default claim follows.
