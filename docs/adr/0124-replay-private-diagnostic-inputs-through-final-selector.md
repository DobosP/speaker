# ADR-0124: Replay private diagnostic inputs through the final selector

Date: 2026-08-04
Status: accepted

Refines: ADR-0108

## Decision

Let `tools.recorded_stt_eval` consume the canonical schema-v3 output of the
exact-input diagnostic exporter in addition to its unchanged legacy WAV
manifest. Reserve every manifest containing `schema_version` for strict
versioned dispatch: accept only schema 3 with provenance
`private-diagnostic-v1` / `final-model-input`, and never reinterpret an invalid
or unsupported version as legacy input. Take the dispatch snapshot only from
a single-link regular manifest bounded to 8 MiB; reject final-component links,
nonregular inputs, empty inputs, and files above that legacy-compatible ceiling
before JSON decoding. Require an owner-private resolved
mode-0700 corpus directory and owner-owned, single-link, mode-0600 manifest,
selection receipt, and PCM files. Accept transcript cases with no command
annotations, the `private-diagnostic` marker, and exactly one receipt-matching
`model_gate_segment` or `selected_asr_segment` role tag.

Decode each validated f32le case into a new owned native-float32 array at
16 kHz with `speech_sec=None`. Preserve case tags and one private text slot per
terminal FileReplay decision, including empty slots, before existing accuracy
math joins the terminal strings. Bind schema-v3 `preparation-receipt.json` to
provenance `source_set_sha256` in the shared corpus writer, loader, and snapshot
verifier. In the recorded evaluator, additionally parse the closed selection
receipt and cross-bind its wrapper, diagnostic-manifest digest, strictly
increasing input indexes, PCM hash/sample count, 16 kHz rate, and role to the
loaded cases. Reverify the manifest, receipt, PCM, permissions, and semantic
receipt after copying and after evaluation immediately before any report.
Local-GPU and consensus-v2 consumers that share this loader retain and perform
the same final snapshot verification. Immediately before that final check and
publication, reject an output that names, resolves to, or is the same file as
any loaded manifest, selection receipt, PCM, or legacy WAV input.

Count every FileReplay decision in the closed production source vocabulary
`none`, `streaming`, `offline`, `verifier_consensus`, and
`established_override`. Add an explicit source-attestation flag and a separate
accounting-complete result to `EvaluationTotals`; keep both defaulted off so
generic selector and consensus accuracy callers retain their established
meaning. Recorded baseline success and candidate promotion require an attested
source sum exactly equal to the terminal-decision count.

## Context / why

ADR-0108 could export the exact endpoint-owned final-selection samples, but
the recording evaluator still understood only labelled WAV clips. Owners
therefore could not run those synchronized recordings through the configured
streaming/offline/verifier selection path without an unsafe manual conversion.
Treating a malformed schema-v3 manifest as the legacy shape would create a
format-downgrade seam, while reconstructing samples from the continuous ASR
WAV would use a quantized stage tap rather than the bytes actually offered to
final selection.

The generic schema-v3 loader previously retained the source-set digest but did
not require or rehash its selection receipt. PCM hashes alone could not attest
which diagnostic receipts, roles, and sample ranges the exporter selected.
The evaluator also joined terminal decisions before accounting for their
routes, so a future recognized-text-to-tool gate would have lost empty and
multi-terminal boundaries. Finally, `EvaluationTotals.complete` is shared by
experimental selectors whose source IDs are not production final routes;
changing that property would either break them or invite fabricated source
labels.

## Consequences

Fresh owner-labelled exact-input corpora can now enter the existing configured
final-selector A/B command without altering samples or creating a second app
entry point. Legacy corpus bytes, fields, and digest remain compatible inside
the finite regular-file dispatch contract. Reports stay aggregate-only and
distinguish accuracy coverage from source-route
attestation; malformed source keys, counts, receipt rows, paths, permissions,
mutations, oversized/nonregular manifests, and report/input aliases fail
through the existing detail-free error boundary without clobbering inputs.
Tags and terminal boundaries remain private in memory and are available for a later
dry-run recognized-text-to-tool route gate without invoking a capability.

Shared schema-v3 corpus publication now requires the canonical receipt, and
all consumers using the private loader must retain its verification witness.
This adds bounded receipt parsing and repeated file hashing around evaluation.
It does not change STT models, final-selection policy, runtime defaults, entry
points, tool authority, enrollment, or barge-in. Headless tests use synthetic
PCM and fakes only; they provide no owner corpus result, model/GPU run, capture,
VAD/endpoint, AEC, microphone, device, latency, natural-conversation, tool, or
live evidence.
