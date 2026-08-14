# ADR-0194: Add a strict NOTSOFAR-1 natural-overlap bundle

Date: 2026-08-13
Status: accepted
Refines: ADR-0098, ADR-0109, ADR-0113, ADR-0159, ADR-0162, ADR-0166
Supersedes: none

## Decision

Keep public matrix v5, its 96-case lock, the existing 18-case isolated
NOTSOFAR-1 far-field corpus, and every runtime/model default unchanged. Add one
offline preparer and strict retained-bundle loader for a separate custom
two-reference natural-overlap fixture:

- module `tools.prepare_notsofar1_natural_overlap_fixture`;
- lock `tools/streaming_stt/notsofar1-mtg32006-natural-overlap-v1.lock.json`;
- fixture ID `notsofar1-mtg32006-natural-overlap-v1`;
- private manifest `notsofar1-natural-overlap-v1.json`, kind
  `notsofar1-natural-overlap-paired-device-v1`; and
- terminal `preparation-receipt.json`, kind
  `notsofar1-natural-overlap-preparation-receipt-v1`.

Reuse ADR-0159's exact CC-BY-4.0
`240629.1_eval_small_with_GT/MTG/MTG_32006` source at immutable revision
`bb97bae9458f477a85dfd6e8aee92e35d5887dd3`. Admit exactly its existing five
pins and no caller substitute:

| Relative file | Bytes | SHA-256 |
|---|---:|---|
| `gt_transcription.json` | 224,525 | `323cb14d9bb129d67645128bc5a65559c830b54982541bc577de86f199ce955b` |
| `gt_meeting_metadata.json` | 469 | `43cf2473716aa5dc958f16dfef82a5ea431a6ce5800cd695114de61b219187d0` |
| `devices.json` | 1,692 | `6877215356b488462bf3bcc16fea76983a42d89dd2221c4dbff86aca3dea580f` |
| `sc_meetup_0/ch0.wav` | 12,255,808 | `02f49d286fbb3ada1cb5ec49e076a6d498d140124d11b420cfcee94b7ecc5ec6` |
| `sc_rockfall_2/ch0.wav` | 12,255,808 | `b37c3d34462ca56006d67bd0aba434fc8e7df95534d971ac66ecb46a7b908805` |

The exact source total is 24,738,302 bytes under a 24 MiB aggregate ceiling.
Bound each metadata file to 1 MiB, each WAV to 16 MiB, the lock, manifest, and
receipt to 256 KiB apiece, each derived artifact to 8 MiB, and total derived
audio to its exact production total below. The preparer performs no network,
subprocess, thread, model, GPU, or audio-device work.

Parse source times as exact `Decimal` values. A candidate contains exactly two
strict reference-valid segments from different raw speakers. Each reference is
unknown-marker-free after the ADR-0159 parser removes only its known markers,
and its normalized tokens exactly equal the segment's word timing. Across the
frozen 14 selected rows, seven contain ten known removable marker occurrences,
seven are raw-tag-free, and none contains an unknown or unrecognized marker;
do not require every raw row to be tag-free. Each segment lasts from `0.600000`
through `8.000000` seconds inclusive; their intersection is at least
`0.300000` seconds; their union envelope is at most `8.000000` seconds; and no
third annotated segment intersects that envelope. Outward floor/ceiling sample
quantization admits at most 128,001 samples for an otherwise valid interval;
an exact-Decimal `8.000001`-second envelope is ineligible. Do not infer
candidates from annotation ordinals or the meeting hashtag.

Rank every candidate with ordinal-free SHA-256. NUL-join the exact seed
`speaker-notsofar1-natural-overlap-v1-2026-08-12`, literal `pair` element, then
each member's raw speaker, exact start, exact end, and strict reference. Order
the members by speaker, start, end, then reference.
Require at least seven eligible candidates, take the seven lexicographically
smallest hashes, and require their envelopes to be mutually non-overlapping;
fail closed rather than repairing a selected collision by ordinal or fallback
rank. Do not retain ranks or source ordinals in the terminal receipt.

For each selected envelope, slice the same interval from both original
synchronized far-field WAVs. Do not create stems or a synthetic mix. Publish a
custom private bundle with seven grouped windows, two separate private
references per window, and two anonymous original-device f32le clips per
window: exactly 14 leaves, 302,080 samples per device, 604,160 samples and
2,416,640 f32 bytes total. The production selection also binds exactly 122,880
samples of source overlap. This is not schema-v2 streaming corpus data: a
single-reference generic loader or ordinary-WER runner must reject it.

Expose only these typed entry points:

- `prepare_notsofar1_natural_overlap_fixture(*, source_dir, output_dir,
  accepted_license, lock_path=DEFAULT_LOCK, test_source_injection=None)`;
- `load_notsofar1_natural_overlap_bundle(root_or_manifest)` returning exact
  `LoadedNotsofarNaturalOverlapBundle`; and
- `verify_notsofar1_natural_overlap_bundle(bundle)` for exact-type reopening
  and identity/digest comparison.

The production path accepts only the exact source and default committed lock;
explicit source injection is synthetic-test-only and cannot claim production
evidence. Both load and verify fail closed on a missing terminal receipt,
non-private object, wrong kind/schema/totals, malformed or unbounded JSON,
receipt/manifest/leaf or source/lock/preparer-closure disagreement, mutation,
aliasing, replacement, extra/missing leaves, a non-exact production lock, or
any Git marker in any bundle ancestor. Root and manifest inputs are equivalent:
a regular file named as the fixed manifest is treated as the manifest, while a
directory with that same basename remains a bundle root, preventing basename-
only identity collisions.

Every parsed lock, manifest, receipt, source, preparer, artifact, and nested
window row must use the exact built-in container, string, integer, Boolean, and
key types required by its closed schema before equality, digesting, or Boolean
reduction. Reject bool-as-int, wrong containers, subclasses/proxies,
non-string or colliding keys, and coercion/equality/truthiness hooks without
calling those hooks. Loading is two-phase: validate complete manifest and
terminal-receipt metadata plus source-contract authority before opening any
PCM leaf. Verification first validates the stored loaded-window schema, then
reopens and compares the bundle.

Require the output to be an absolute, absent path outside Git, the source tree,
and recognized worktrees. Keep its root and directories current-owner mode
0700, and every regular leaf current-owner mode 0600 with one link. Write all
artifacts and the manifest, then stage the receipt through an anonymous
`O_TMPFILE`. Reopen and recheck the source, lock, preparer closure, output
artifacts and manifest, output bindings, and staged receipt before linking
`preparation-receipt.json`. That terminal link is the sole commit point. Never
overwrite a destination or delete a failed diagnostic tree.

The terminal receipt is aggregate-only. It excludes reference text or
fragments, per-window/case rows, raw speaker/device IDs, source-relative or
local absolute paths, selection ranks, and annotation ordinals. The private
manifest necessarily binds the two references and anonymous channel leaves,
so the whole bundle stays outside Git and is not a safe aggregate report.

ADR-0194 adds no implemented evaluator or metric. Set ordinary WER, ORC-WER, tcpWER,
tcORC, diarization/identity, endpoint/latency, qualification, promotion,
runtime/default, device, and live authority false; bind exact
`overlap_metric=not-implemented` in the lock, manifest, and terminal receipt.
A later ADR may add a bounded aggregate evaluator only after
the exact production manifest and terminal-receipt digests are settled and
pinned. It must not reinterpret this preparer or use an ordinary single-
reference WER path.

## Context / why

ADR-0159 deliberately selected nine isolated single-speaker windows, so its
good far-field development results say nothing about simultaneous speech. The
PriMock57 overlap bundle is useful for two-role loss, but its three diagnostic
mixtures are synthetic combinations rather than two synchronized original
microphones observing one natural overlap. ADR-0166's AMI natural-turn replay
is capture-path evidence over a smaller manually labelled fixture and does not
replace this source/device shape.

The already retained MTG_32006 files contain a bounded natural-overlap slice,
so the missing fixture can be prepared without another download or a large
corpus. Keeping the two references private and distinct avoids inventing one
ordinary-WER denominator for concurrent speech. Keeping both original device
captures preserves the useful paired-room dimension without manufacturing a
mix or claiming diarization.

Locking selection before decoding prevents metadata order or parser iteration
from becoming hidden fixture authority. Exact decimal comparisons avoid float
threshold drift at the 300 ms boundary, while the ordinal-free salted hash
makes the chosen seven depend on semantic source fields rather than publisher
row position.

The preparer and evaluator are separate decisions because evaluator closure
must bind the bytes that production preparation actually publishes. Writing a
worker wrapper before those private digests settle would either accept a
moving bundle or force a historical receipt rewrite.

## Consequences

- A later exact evaluator can consume 14 original-device inputs and two
  references per natural-overlap window without exposing those references in
  its eventual aggregate report.
- Preparation is bounded to about 2.42 MB of derived audio and reuses the
  retained 24,738,302-byte source. It needs CPU only for bounded parsing,
  hashing, WAV decode, slicing, and publication; it needs no GPU or network.
- The custom bundle cannot be passed to generic schema-v2 tooling. Until a
  later ADR pins and implements its evaluator, it yields no WER, candidate
  comparison, quality result, or promotion evidence.
- The approved exact-source materialization produced and strictly reopened one
  retained private v5 bundle. This proves only its preparation contract, not a
  corpus/model result, physical device behavior, or live conversation.
- Public matrix v5, the canonical qualification, the original isolated
  NOTSOFAR fixture/results, and every recognizer/endpoint/default remain
  byte-valid and unchanged.

## Verification

The low-priority, thread-limited offline focused gate passed 70 tests in 2.65
seconds (2.80 seconds wall; 67,092 KiB maximum RSS). The exact adjacent command
in `docs/agent-testing.md` passed 155 tests in 3.74 seconds (3.94 seconds wall;
131,464 KiB maximum RSS).
They cover deterministic selection boundaries and rank order, exact totals,
strict load/verify, privacy, no-clobber publication, race/tamper rejection,
resource caps, the inherited isolated fixture, and APM/DTD.
Both commands require `TMPDIR=/var/tmp` plus a new absent explicit basetemp.
The managed environment's `/tmp/.git` ancestor marker makes pytest's default
temporary roots intentionally fail the preparer's outside-Git guard; the
unqualified default-temp command is not a green receipt.

The approved one-time production action read the already retained five-file
source and published one fresh absent owner-private output root without
downloading or using network, cloud, model, GPU, audio device, microphone,
subprocess, or paid service. Materialization completed in 0.26 seconds with
124,700 KiB maximum RSS; combined strict root/manifest reopen completed in
0.16 seconds with 38,652 KiB maximum RSS. The owner-private tree has 16 regular
files and 2,427,570 bytes total: root/directories mode 0700, every file mode
0600 with one link,
seven windows, 14 PCM leaves, and 2,416,640 audio bytes. All 14 PCM digests
equal the committed lock pins.

Frozen SHA-256 evidence is:

- manifest `898e291c713154e3cbd78ba9c38fca6fed08284edace98ac615ebdd37265da7b`;
- terminal receipt `29b5d0f1f11f2ac56dee92a7d5ab6ef02a18c4d695a0b9b981b67dde3a24e2ba`;
- source contract `c1bb8c6c10e2abff7bb4f7a57948dbb34a12c2d36384b66b304d26f1103b57cc`;
- preparer closure `65501f79bfa805b8f1636099836a883dba4ba4cf98d7d25352d70b18cf58411a`;
- lock recipe `ec6bf59b21e7fe59d7ea432fd4a6ac17e965fb3289e2523121576f71636150a5`;
- private-selection commitment
  `fec0323b7e677e373cfd7b8332258a930581ef87707f5a211a509967985b9beb`.

The private-selection commitment is an opaque integrity binding, not secrecy:
NOTSOFAR-1 is public, so a determined party can recompute the selected source
data. Receipt/privacy inspection found no references, per-reference hashes,
raw speaker/device identifiers, source-relative or local absolute paths,
selection ranks, or annotation ordinals in the aggregate receipt. The public
lock necessarily contains the five declared source-relative pin paths listed
above, including two upstream device-named WAV paths, but contains no private
output/local absolute paths, references, per-reference hashes, raw speaker
identifiers, ranks, or ordinals.

Frozen preparer/test/lock SHA-256 is respectively
`a4302d2094a7c4e84ae1a533d45e8e3cc1a787ec2c2381fe48ae9b62f2f3ce3d`,
`09fd8b056c5b842068f130922eaa0b57e2ac6804846741d470d795e39c1375ff`,
and `2847a01e65022674ca937093843a4bfeb95e8ceff04c33ffee4d77cbd74aa156`.
CPython parsed both Python files; Ruff 0.15.20 lint and format checks pass; the
module/test AST SHA-256 values are
`122d0a9909fdfaee84b21f43bfd2cf030f9b49f12d2a284fa1104fabf0f250b1` /
`101b9494a1bbe5f8368bec4f8696ffc51e70ebad788834fcebd187c94bd41166`;
canonical lock recomputation matches the recipe and all 14 pins are nonzero;
lock, manifest, and receipt parse under their three JSON checks; `git diff
--check` and the exact-100-line STATUS gate pass. The exact intended
inventory is the five modified durable docs plus this ADR, the preparer, lock,
and test: nine regular non-executable paths, five modified and four untracked.

Independent exact-type/security, materialization, and final audits are GO.
ADR-0194 ran no evaluator, recognizer, quality, promotion,
runtime/default, or live path.
