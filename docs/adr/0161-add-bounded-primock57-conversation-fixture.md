# ADR-0161: Add a bounded PriMock57 conversation fixture

Date: 2026-08-09
Status: accepted

## Decision

Keep public matrix v5, its canonical 96-case qualification, and the separate
NOTSOFAR far-field fixture unchanged. Add one offline, benchmark-only
PriMock57 fixture pinned to the CC-BY-4.0 source revision
`cd2ac707ad03cb4d2531f4ec6b90c659bf4357c5`. Admit only consultation 01's two
role-channel WAVs, two TextGrid annotations, and license: 29,358,062 exact
source bytes under a 30 MiB cap.

One explicit-license preparer invocation publishes two independent private
outputs. The first is an ordinary schema-v2 corpus with exactly three clean
full-annotation intervals whose role has zero temporal intersection with the
other role. Only those three cases may contribute ordinary WER/CER. The second
is a custom `primock57-two-role-overlap-diagnostic-v1.json` bundle with exactly
three fixed, tag-free source-timing overlap pairs. For each pair, preserve the
two complete annotated utterance spans at their original relative timing as
masked float32 stems and publish the deterministic float32 average of those
stems as a separate synthetic mix. Convert signed PCM16 by dividing by exactly
32,768, then form each mix with float32 addition followed by multiplication by
the float32 value 0.5.

Do not load the overlap bundle as an ordinary streaming-STT corpus and do not
compute or claim overlap WER yet. Its references are retained only for a later
dedicated, locked multi-talker evaluator. Mark ordinary WER, qualification,
promotion, and original-device-mix authority false. Preserve schema-v2's
writer publication order for the isolated corpus, whose final manifest is its
terminal commit. Make the custom overlap receipt its last no-clobber commit.
Aggregate receipts and command output exclude transcripts, source paths, and
raw role labels.

Treat neither a generic schema-v2 load nor a raw `streaming_stt_eval` result as
PriMock evidence. `tools.primock57_isolated_eval` is the only admitted isolated
model-evaluation entry: it requires the exact production fixture, receipt,
selection, private file identities, and corpus binding before the worker, then
reopens and compares them after the worker and again before returning a report.
Its aggregate-only top-level binding includes the corpus, receipt, source
contract, raw evaluator report, and wrapper execution-closure digests. The
generic evaluator's corpus admission and execution behavior remain unchanged
for other corpus families; its evidence closure intentionally gains the eager
device-config dependency.

Production reopening is anchored to the committed lock, not to self-consistent
mutable receipts. Both loaders require the exact five source rows and complete
product-specific selection digest. The isolated loader additionally requires
the three locked reference hashes, roles, sample counts, and immutable output
PCM hashes. The overlap loader recomputes both roles' interval indices,
absolute start/end samples, absolute and relative overlap boundaries,
reference hashes, and deterministic stem/mix bytes, then compares all six stem
and three mix hashes with the lock. Test injection must use a `synthetic-` ID
and a distinct test-only lock digest.

Hold the private source root and both source subdirectories open from admission
through the terminal commit, and recheck their full identity, mode, owner, and
path bindings. Both output loaders retain and recheck full private-directory
snapshots, not only inode identity. Before the signal-masked overlap link,
revalidate the already-published isolated bundle, exact overlap directory
entry set, every artifact, manifest, execution closure, source binding, and the
anonymous staged receipt's bytes and inode snapshot. The publisher fsyncs all
artifact directory entries first and treats any outcome after a confirmed link
as committed success.

Bind every eager local import in the preparer receipt. Preserve the existing
`core` and `tools.streaming_stt` package-level API through lazy compatibility
exports so importing narrow WER/corpus helpers does not initialize the audio
runtime, control plane, model manifest, or supervisor. The generic evaluator
closure additionally binds the eagerly imported device-config module. The
isolated evaluation wrapper ignores self-declared privacy flags, accepts only
closed aggregate surfaces, and recursively rejects embedded local paths and
any two-or-more-word verbatim reference fragment before adding its binding.

## Context / why

The retained public and NOTSOFAR slices improve clean, spontaneous, and
far-field coverage, but the selected NOTSOFAR cases deliberately contain no
overlap. [PriMock57](https://github.com/babylonhealth/primock57) provides a
small public mock-consultation recording with
separate synchronized role channels and manually aligned utterance intervals,
so it can add real conversational turn geometry without acquiring another
multi-gigabyte dataset or changing the voice runtime. The corpus design and
motivation are described in the
[ACL 2022 paper](https://aclanthology.org/2022.acl-short.65/); only the exact
repository revision and files pinned here are admitted.

The longest raw overlap pairs are not scoreable: at least one role contains an
unknown-speech marker, and removing that marker would silently delete spoken
words from the reference. The fixed overlap selection therefore uses only
tag-free pairs with separate references. The resulting 0.5-plus-0.5 signal is
clipping-safe and reproducible, but it is not an original microphone/device
mix and cannot establish far-field, diarization, speaker identity, endpoint,
latency, or live-conversation behavior.

## Consequences

- Streaming candidates can later be stress-tested on three receipt-bound
  two-role timing alignments without weakening the ordinary-WER denominator.
- A dedicated evaluator with an explicitly pinned overlap-valid metric such as
  tcORC/tcpWER is required before any accuracy statement about the custom
  bundle. Until then it is an acoustic/decode diagnostic only.
- Real source audio, references, stems, and mixes remain private and outside
  Git. Synthetic test injection is always marked non-production.
- The generic schema-v2 loader remains useful for storage interoperability but
  is insufficient evidence authority for this fixture; only the PriMock wrapper
  can bind a model report to the retained receipt before and after execution.
- Preparation, a future model replay, or a green overlap diagnostic cannot
  change STT, endpoint, tool, identity, setup, entry-point, or live defaults.
  Owner guided capture, bare-speaker barge-in, and natural live conversation
  remain separate gates.

This ADR refines ADR-0098, ADR-0109, ADR-0113, ADR-0117, ADR-0159, and
ADR-0160.

## Verification

- Lock recipe SHA-256:
  `2bc14c4114959fe1323d843ed6bb1b17cf81aa46f620b6805a98dd380a1e17ef`;
  raw lock SHA-256:
  `058da5caf533a771b276f81323c5f011d35ac406fa4f91c51cec5a0ed25c2031`.
- The fresh exact-source preparation published three production isolated cases
  with manifest/receipt SHA-256
  `769f0bea80da796d4213b5008b67f4bf80e4eea5a6e43a9b06472b0b548ace88` /
  `ac4f98cb650b7f8aa6096e3dd17d82b066a62e41ae2adbcd80ba652b5119d2b6`,
  and three production overlap diagnostics with manifest/receipt SHA-256
  `e69c85356a77cd87b0c7f4c100614dccbf1484e16532a5a2239a768162e443b4` /
  `69ec7df35591a2b1a7203109c58077540540c32deec99dc933a5ddef0840f507`.
  All 12 isolated/stem/mix PCM hashes matched their immutable lock pins; every
  output directory remained mode 0700/current-owner, and every file remained
  mode 0600/current-owner/single-link.
- The receipt-bound wrapper accepted the real isolated preflight with case-set
  SHA-256
  `398e6110cd7c716ce4753d54a8248c0624970330d8e47e52cbced53173447567`
  and wrapper-closure SHA-256
  `b935e67ca23dcf5a9bcebdfccb3a468952bce297bf960fd9d354a3a69ff90718`.
  Canonical isolated/overlap source-contract SHA-256s are `6c098e29…139b` and
  `506a1e24…6646` respectively; neither is mislabeled as a receipt digest.
- Headless gates are 112 preparer/wrapper/corpus-writer passes, 111
  streaming-evaluator/wrapper passes, and 6 APM/DTD passes, with scoped Ruff
  and formatting green. The broad managed-sandbox diagnostic reached 9,212 passed,
  15 skipped, and 65 deselected; its five ordinary failures passed together in
  an exact isolated rerun, and all 28 LiveKit cases passed outside the managed
  sandbox whose cross-thread asyncio wakeup is unavailable. No model, GPU,
  microphone, audio-device, quality, latency, or live-conversation run occurred.
