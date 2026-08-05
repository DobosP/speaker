# ADR-0138: Bind ordered live STT references without inferred truth

Date: 2026-08-05
Status: accepted

Refines: ADR-0108, ADR-0126

## Decision

Add a device-free `tools.prepare_live_stt_corpus` workflow for a completed
private `./live.sh` diagnostic bundle. Accept one owner-authored, mode-0600
reference-plan schema whose ordered cases contain only an ID, expected text,
and tags. Pair exactly one plan case with every validated schema-v2
`final_model_input` receipt in contiguous `input_index` order. Reject an empty
set, a count mismatch, more than 512 cases, non-16-kHz inputs, an input above
8 MiB, or a selection above 32 MiB. Do not select a subset or derive a
reference or turn boundary from ASR output, transcript logs, continuous WAVs,
or receipt identities.

Generate the existing private label schema v1 by adding the exact diagnostic
manifest digest plus each receipt index and input digest. Reserve the
`private-diagnostic` and receipt-role tags for the exporter, leave at most 30
owner tags, and validate any `expected-tool.*` annotation against the existing
closed route vocabulary. Publish the canonical label bytes atomically as a
new owner-owned, single-link, mode-0600 file in a private no-symlink parent.
Keep its inode and digest as a witness, pass the expected label digest into the
existing schema-v3 exporter, and recheck the witness after delegation. Labels
and corpus destinations must be absent, non-aliasing, and non-containing.

Treat the label and corpus as two independently durable no-overwrite
publications. Once a complete label is visible, retain it if delegated corpus
publication fails; the owner can inspect the failure and retry the existing
diagnostic exporter with that label and a new corpus destination. Preserve the
corpus writer's existing policy of retaining writer-owned failed output for
diagnosis. CLI success and failure remain aggregate-only and detail-free.

## Context / why

ADR-0108 deliberately kept transcript truth outside the diagnostic bundle.
Its safe exporter consequently required the owner to inspect receipt indexes,
copy hashes, compute the manifest digest, and hand-author label schema v1.
That manual cryptographic binding made a fresh recorded STT comparison easy to
mislabel and too cumbersome for routine live testing, even when the owner had
already prepared the exact phrases to speak.

Binding by count and order is the narrow automation available without adding
transcripts to the recording or trusting the model being measured. Pairing a
subset would make a missed turn or accidental extra turn ambiguous. Reading
recognized text to repair the pairing would make the candidate its own ground
truth. The binder therefore fails closed on any count disagreement and leaves
semantic adherence to the owner who follows the prepared script.

The generated label path is an authority boundary because ADR-0126 binds its
text and tags into the corpus receipt. A post-write replacement race could
otherwise publish different valid references. Digest-pinning the delegated
snapshot and retaining the created inode closes that accidental local race;
the existing unkeyed hash graph still does not defend against a party that
deliberately rewrites every bound artifact consistently.

## Consequences

- One private ordered script plus one completed `./live.sh` bundle is enough to
  create an exact final-input corpus without manual receipt or hash editing.
- A count match proves only a one-to-one ordered binding. It does not prove that
  the owner spoke each phrase correctly, that a missed turn and an extra turn
  did not cancel out, or that the plan predated recording; review and a careful
  paced capture remain owner responsibilities.
- A retained generated label is complete and reusable, not a partial success.
  A failed corpus directory is diagnostic evidence under the existing writer
  policy and is never reused or overwritten.
- The workflow changes no runtime entry point, recorder, STT model, selector,
  transcript, tool authority, enrollment, or default. Its synthetic headless
  tests provide no microphone, capture/VAD, AEC, barge-in, device, WER, model,
  GPU, latency, natural-conversation, tool-execution, or live-quality evidence.
