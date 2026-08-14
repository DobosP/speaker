# ADR-0206: Accept and materialize the mobile ASR evidence packet v2

Date: 2026-08-15
Status: accepted
Refines: ADR-0109, ADR-0117, ADR-0159, ADR-0161, ADR-0162, ADR-0205
Supersedes: none

## Decision

Accept packet lock v2 as the only production authority for the fixed English
five-component mobile ASR evidence packet. Preserve the byte-exact v1 lock as
historical evidence, but make the production loader reject it. V2 changes no
component count, order, PCM byte, sample format, license, metric domain,
publication rule, privacy boundary, or evidence-authority flag. It changes the
packet schema/kind/identity and binds the current reviewed DEMAND preparation:

- DEMAND corpus SHA-256
  `48797d031f24f19694f3c6ba81f6eca53c32f2286751612bc5661779396954a8`;
- DEMAND preparation-receipt SHA-256
  `24a7005ea6b327cc89f633bcd9d9edf20a6bb17c7afe0d60c49f37753e1e2f02`;
- stable DEMAND case-set, purpose, and source-set SHA-256 values
  `0fed46b8a8f5198bc39ddf8f5a72a45e9e89ba32fb6730a046c7fc4683cd0d3f`,
  `0f80a7a145870fa84ff38f96675233855fd868dfaea4729c8a3fb199c4d3662d`,
  and `4efcbefa674ea5d14ee584f77146f3160d6ece68c47eb1d43e361d5a1732cc60`.

The new receipt is authority only because it is reviewed and committed in the
exact v2 lock. It is not learned from a CLI argument, filesystem discovery, or
runtime success. The irreproducible historical DEMAND receipt remains rejected.
Any later DEMAND rematerialization still requires another reviewed lock.

Keep the v1 lock unchanged at SHA-256
`268377ba9ef648647112210cd2c812cf9ea8e9783f366676dee1c4fe1dc07820`.
The v2 lock is 8,481 bytes, has canonical recipe SHA-256
`3bd6600b914876caeea3c58e82b00f0dfe0c8b8951edc0f4affb28b152e48c9d`,
and raw SHA-256
`c81a89a7a7df534ec0a23dc20c0a915c4aa5eedec6270a13ec08456a0683cc11`.
The canonical production CLI, packet index, terminal receipt, preflight, help,
and publication result now identify schema/kind v2. Synthetic filesystem tests
retain their explicitly nonproduction test-only v1 identity.

Accept the resulting private packet materialization. Exact production preflight
admitted all five components, and a new absent outside-Git root was populated
under the existing no-clobber 0700/0600/single-link contract. The terminal
publication and an independent strict reload agree on:

| Field | Exact value |
|---|---:|
| Components | 5 |
| Logical cases | 123 |
| PCM leaves | 129 |
| Samples | 9,824,875 |
| f32le bytes | 39,299,500 |
| Packet-index SHA-256 | `232e4230d16b0014baff38e2daf4156aaef3c4e129541079248fac0b2f9e835c` |
| Packet-receipt SHA-256 | `36eec947383069fb15bc998209ae4aaf76ea82f7031666b972fe98f8c7c5f51f` |

The packet remains a self-contained private exact copy, not a committed data
artifact. Its five evaluation domains remain separate. It has no packet-level
pooled WER, RTF, or latency, and preparation grants neither model identity nor
evaluation-result authority. The packet index and terminal receipt are now
eligible inputs to the next evaluator only after that evaluator pins both exact
digests. No hypothesis, reference, transcript, local path, source member,
speaker/role, or device identity enters committed docs or command output.

## Context / why

ADR-0205 deliberately froze a fail-closed contract while the exact DEMAND and
PriMock production inputs were absent. The inputs were subsequently recovered
without weakening the locks: PriMock57 consultation-01 was restored from its
pinned upstream revision and reproduced both historical prepared receipts;
the three exact DEMAND 16 kHz archives were reacquired with their pinned
SHA-256 and published MD5 values, then the 42-case derivative was rematerialized
under the current code/runtime closure. Its case/source/purpose sets stayed
stable, but its receipt and corpus digest changed as expected because the
receipt binds current runtime and code state.

Accepting the new receipt dynamically would defeat ADR-0205's authority model.
Pretending the historical receipt could be reproduced would be false: its
code/runtime closure no longer exists byte-for-byte. A reviewed v2 lock is the
smallest honest change. It retains the entire packet structure while making the
new current receipt explicit and auditable.

The exact mobile hybrid Zipformer assets were also downloaded and hash-verified
outside Git, but were not imported, loaded, or decoded here. Packet preparation
does not establish that model's runtime identity. Provisioning, endpoint-
faithful decoding, five-domain evaluation, and any quality conclusion belong to
ADR-0207 or later.

## Consequences

- The production packet is now available as a strict private artifact with an
  exact packet-index/receipt pair. V1 remains immutable history and is not
  accepted by the production API.
- The fixed envelope remains 123 logical cases, 129 mono-16-kHz f32le leaves,
  9,824,875 samples, 39,299,500 bytes, and 614.0546875 aggregate stored-leaf
  seconds. Stored-leaf duration still counts overlap stems and mixes and is not
  speech hours, latency, or an RTF denominator.
- GSC/ECCC command assertions, DEMAND stratified WER/CER, NOTSOFAR paired-
  channel WER/CER, PriMock isolated ordinary WER/CER, and PriMock custom
  two-utterance minimum-order overlap remain independent. No pooled metric or
  cross-domain promotion score exists.
- The data and packet remain private and are not redistributed. The recovered
  archives/source, prepared component roots, packet root, and later reports stay
  outside Git with their existing license and privacy constraints.
- The accepted packet is source/evidence availability, not held-out or
  training-disjoint authority. It proves no model quality, endpoint behavior,
  latency, thermal behavior, microphone/device behavior, Romanian support,
  qualification, promotion, default, or live result.
- ADR-0207 may pin the exact packet index and receipt alongside the exact mobile
  hybrid model provision receipt, then run one bounded endpoint-faithful cell
  per component. It must still avoid pooled WER and must not manufacture a
  final from the last partial or an `input_finished` flush.

## Verification

- Focused cache-free packet tests pass `20/20`; the new regression proves the
  exact v1 lock remains byte-stable and rejected while v2 alone is canonical.
- Scoped Ruff check is clean and Ruff format reports both Python files already
  formatted.
- Exact production preflight returned only the aggregate v2 success receipt
  with all five components and the fixed totals.
- Publication ran in a transient user-systemd unit at nice 19, idle I/O,
  25% CPU quota, 512 MiB memory maximum, zero swap, 64 tasks, and ten-minute
  runtime maximum. It returned the exact packet-index and packet-receipt hashes
  above.
- A separate identically bounded process reopened the completed root through
  `load_mobile_asr_evidence_packet`, ran
  `verify_mobile_asr_evidence_packet`, and reproduced the same aggregate counts
  and hashes.
- No model import/load/decode, GPU operation, Flutter plugin, microphone,
  audio-device, physical-phone, endpoint/latency measurement, quality verdict,
  promotion, default change, or live validation ran. Network activity in the
  surrounding data-restoration step was limited to the pinned PriMock and
  DEMAND source files; packet preflight/publication/reload were offline.
