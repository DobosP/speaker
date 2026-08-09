# ADR-0169: Add an Anyreach shadow artifact contract

Date: 2026-08-09
Status: accepted

## Decision

Add the no-download, no-execution
`anyreach-semantic-turn-taking-q8-synthetic24-v1` source and
artifact-provision contract for the Anyreach Semantic Turn-Taking Q8 candidate.
Pin model repository revision
`e3a37384c4b19f1e7d29568a61634756e487b75e` and exactly four files under its
`onnx/` paths: the Q8 graph and tokenizer JSON as future execution artifacts,
plus model and tokenizer configuration JSON as metadata witnesses. Their
combined locked size is 507,145,644 bytes. Pin benchmark repository revision
`a22ba86c174c55c8de685d233771fa018ebc8c99`, its 17,788-byte synthetic
Parquet object, and a declared source-order 24-row slice at zero-based physical rows
36 through 59 consisting of twelve `start_listening` slots followed by twelve
`continue_speaking` slots. These are lock declarations, not locally decoded
Parquet evidence: rows 36 through 47 declare `manual_sli_01..12` / ClassLabel 2
and rows 48 through 59 declare `manual_cs_01..12` / ClassLabel 3; their ordered
ID-plus-LF digest is
`0d7cbea2954b0f5c4daafe83735a47f048a12e9e38d793de144e3bc234bf112c`.

Keep the model action-token mapping (`continue_listening=151665`,
`start_speaking=151666`, `start_listening=151667`,
`continue_speaking=151668`, `predict=151669`) distinct from the benchmark
ClassLabel mapping (`start_speaking=0`, `continue_listening=1`,
`start_listening=2`, `continue_speaking=3`). Bind the publisher-declared manual
ChatML terminator and a code-owned 1,024-token left-truncation rule as future
interface metadata, with `execution_adapter_bound=false`; do not implement
either yet. A future
evaluator must retain the publisher's four-way action comparison: it must not
renormalize only the two selected labels, reuse the separate end-of-utterance
mapping, or map an argmax label to a takeover score.

Permit a local provisioner to verify only pre-existing exact regular files,
their private root, identity, size, and SHA-256; reject extra or substituted
paths; copy only the exact small benchmark object and committed source lock into
a new owner-private outside-Git directory; and publish a private artifact
manifest last with no-clobber semantics. The manifest is not a worker or
runtime receipt. Its safe stdout is aggregate and path-free. Provisioning
requires the explicit acceptance
`ANYREACH-HF-APACHE-2.0-METADATA`, which records only the publisher's
Apache-2.0 metadata assertion; neither source repository contains a LICENSE or
NOTICE file, so the contract grants no redistribution authority.

## Context / why

ADR-0168 intentionally defined only a detector-agnostic, inactive score-policy
contract. Anyreach is the smallest current source-lockable CPU-oriented
candidate with actions corresponding to stop/listen and continue-speaking, but
its public benchmark is text-only, same-author, and mostly synthetic. The
publisher-reported `start_listening` result is weak, and the exact selected
24-row slice has no published dedicated metric. Published four-action outputs
are relative scores rather than calibrated takeover probabilities.

The model and benchmark revisions are public and immutable, and official
metadata exposes exact LFS identities. That is enough to freeze a source
contract without transferring the large objects. It is not enough to claim
that the Q8 graph is standalone, that its inputs or outputs match the model
card, that a tokenizer/runtime combination is reproducible, or that the
declared Parquet rows have been locally decoded. This slice adds or validates
no pinned inference implementation, runtime, or evaluator. Committing a worker
manifest or semantic adapter now would overstate the evidence.

## Consequences

- Headless tests can validate lock self-consistency, exact artifact and label
  mappings, the declared 24-slot order, strict JSON, private-root and file
  identity requirements, bounded opaque-byte hashing, no-clobber publication,
  and aggregate-only stdout using tiny injected bytes. The declared row
  indices, IDs, and labels remain lock metadata rather than locally verified
  Parquet contents.
- The provision path imports no Hugging Face, Transformers, tokenizers, ONNX,
  ONNX Runtime, Torch, Parquet, or model code; performs no network access or
  subprocess launch; and decodes or executes nothing. A successful manifest
  means only that supplied bytes matched the committed artifact contract.
- `selection_declared` may be true while `parquet_decoded`, `model_executed`,
  `tokenizer_executed`, `onnx_contract_verified`, and `production_evidence`
  remain false. The artifact contract provides no score, calibration, quality,
  resource, latency, acoustic, playback, VAD, AEC, endpoint, device, live,
  promotion, config/default, STOP, identity, memory, or tool authority.
- A separate ADR and exact runtime/worker receipt are required before one
  isolated CPU smoke may inspect the graph, reproduce tokenization and prompt
  bytes, validate the complete ONNX I/O contract, and perform a four-way
  prediction. `AutoTokenizer` is outside this four-file contract and would
  require a larger versioned closure. A later scorer must treat
  `start_speaking` or `continue_listening` argmax on the selected rows as wrong
  or out-of-slice, never pair-renormalize the selected labels, and never map an
  argmax to a takeover score. Any shadow adapter still needs aggregate
  assistant-playback evidence; effects additionally require owner bare-speaker
  live A/B.

## Verification

- The exact focused command recorded in `docs/agent-testing.md` passes 148
  tests: 53 new source-lock/provision cases plus the unchanged 95-test ADR-0168
  policy gate. It validates declared slots, private artifact publication,
  forbidden execution surfaces, and terminal failure recovery with generated
  bytes only. The managed host requires a unique `/var/tmp` base because its
  `/tmp` deliberately has a Git ancestor.
- The final 9,068-byte lock has raw SHA-256
  `d463fc45176da94c6923b2a2fc980a1e8cf5a0c70009eeb554341c6aff18892e`
  and canonical recipe SHA-256
  `1062831086aeea841cd9c31d98807f0395b89e57edec72dd8434247c512b2c77`.
- The standard APM/DTD regression passes six tests. Scoped Ruff check/format
  and whitespace checks are green. Independent official-source,
  model-contract, documentation, and settled filesystem-security reviews found
  no blocker within this artifact-only scope.

No locked candidate artifact was retained or provisioned locally; no Parquet
decoder, tokenizer/model runtime, inference, GPU, audio-device, effect, or live
path ran.
