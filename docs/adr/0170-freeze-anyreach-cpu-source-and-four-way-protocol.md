# ADR-0170: Freeze the Anyreach CPU source and four-way shadow protocol

Date: 2026-08-10
Status: accepted

## Decision

Add the metadata-only `anyreach-cpu-py312-linux-x86_64-source-v1`
package-source lock for a future Anyreach worker. Pin the resolver-complete
CPython 3.12/Linux x86_64 CPU selection rooted at `onnxruntime==1.28.0`,
`tokenizers==0.23.1`, and `numpy==2.4.6`: exactly 20 official PyPI wheels and
46,744,727 published bytes. Keep this source lock distinct from a downloaded
and measured wheelhouse, installed-tree receipt, execution receipt, and model
compatibility claim. The selected wheel tags target glibc 2.28 or newer and
`CPUExecutionProvider`; no GPU package is admitted. Publisher metadata is
package-source evidence only. No wheel license/NOTICE inventory exists, so the
lock grants no rehosting or redistribution authority.

Add a strict bounded `anyreach-four-way-worker-protocol-v1` JSONL contract for
one future isolated worker. A score request contains exactly three private
`user`/`assistant`/`user` messages and its locked benchmark ordinal. A ready
event binds the future artifact, artifact-set, worker-manifest, and runtime
receipt digests plus the CPU provider. A score event carries exactly four
finite raw logits in the model-token order `continue_listening=151665`,
`start_speaking=151666`, `start_listening=151667`, then
`continue_speaking=151668`. Recompute one numerically stable softmax over all
four logits and preserve exact top ties instead of selecting by position. Do
not transmit a pair-renormalized score or map an argmax into ADR-0168's
takeover-score contract.

Add a pure aggregate reducer for the exact ADR-0169 24-slot declaration. It
requires one ordered result per slot and emits only a 2-by-4 confusion matrix
plus correct, cross-selected-wrong, out-of-slice, and tie counts. Predictions
of `start_speaking` or `continue_listening` on these selected rows are
out-of-slice, not silently folded into either selected class. It retains no
message, public row ID, per-row logit, local path, or policy decision.

Keep the slice synthetic and inactive. Do not add an executable candidate
adapter, worker manifest loader, process supervisor, Parquet decoder, model or
tokenizer import, runtime installer, evaluator/report CLI, application wiring,
threshold, effect, or default. A real worker requires a separately reviewed
private wheelhouse and exact installed-runtime receipt, the already gated model
artifact manifest, local tokenizer/prompt parity, exact graph inspection, a
versioned adapter/worker manifest, and one networkless CPU smoke.

## Context / why

ADR-0169 froze exact model and benchmark byte declarations without acquiring
or executing them. Official package metadata now permits a deterministic CPU
package selection, and the publisher's immutable model card is sufficient to
freeze the four action-token identities and high-level prompt intent. Neither
source proves that the locked ONNX graph parses under the selected runtime or
that its exact I/O, operators, cache geometry, tokenizer behavior, ChatML
serialization, special tokens, and 1,024-token left truncation match the card.
Treating package pins as an installed runtime or committing an executable
adapter before those checks would overstate the evidence.

The worker result algebra is independently valuable. It prevents a later
implementation from making the selected benchmark artificially easier by
normalizing only `start_listening` and `continue_speaking`, conflating the
benchmark ClassLabel order with model token order, hiding `start_speaking` or
`continue_listening` predictions, breaking exact ties by array position, or
feeding categorical output into the inactive interruption policy as a
calibrated score. Canonical bounded messages and aggregate-only reduction make
those invariants headlessly testable now.

## Consequences

- A future private runtime build can start from one exact package-source
  selection, but must still download, hash, archive-inspect, license-inventory,
  install, receipt, and reverify all 20 wheels before any runtime claim.
- The current repository environment is not evidence: it is mutable, contains
  extra distributions, and matches only half of the selected package versions.
- Generated tests can exercise every request/response, malformed JSON,
  fixed-order logit, stable-softmax, tie, locked-order, confusion, privacy, and
  no-policy-mapping branch without candidate bytes or package imports.
- A later real evaluator must first decode and validate the exact Parquet rows,
  then preserve four-way scoring. Static text replay remains shadow evidence;
  assistant-playback/far-reference evidence and owner bare-speaker live A/B are
  still mandatory before any post-cut recovery or reversible pre-cut hold.
- Nothing in this contract can suppress exact STOP, commit or release a barge,
  continue/resume playback, mint identity or tool authority, write memory, or
  change a production/default path.

## Verification

- The three focused files pass 107 tests for the 8,185-byte/20-wheel source
  lock, strict protocol, stable four-way math, explicit ties, exact 24-slot
  aggregate reducer, privacy, and forbidden runtime/policy imports with
  generated values only.
- The exact affected command recorded in `docs/agent-testing.md` passes 255
  tests: those 107 plus the unchanged 53-test ADR-0169 source/provision gate and
  95-test ADR-0168 policy gate. The standard APM/DTD gate passes six. Scoped
  Ruff check/format and whitespace checks are green; independent read-only
  audits found no blocker.

No candidate wheel, model, tokenizer, or Parquet object was downloaded,
installed, decoded, imported, parsed, loaded, or executed. No subprocess,
network-capable worker, inference, GPU, audio-device, effect, or live path ran.
