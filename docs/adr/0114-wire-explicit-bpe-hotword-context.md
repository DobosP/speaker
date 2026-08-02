# ADR-0114: Wire explicit BPE context for streaming hotwords

Date: 2026-08-02
Status: accepted

## Decision

Keep streaming hotwords inactive by default. Every active hotword configuration
must name its sherpa-onnx modeling unit; `bpe` and `cjkchar+bpe` also require a
validated two-column BPE vocabulary. The pinned English context uses case policy
`upper_ascii_words`, whose per-line grammar is `[A-Z]+(?: [A-Z]+)*`. A custom
BPE context may leave the policy empty instead of inheriting that model-specific
restriction.

Make `tools.setup_models --bpe-hotwords` an isolated setup transaction. It may
not be combined with ordinary model selectors. It selects a complete English
Zipformer ASR family—tokens, encoder, decoder, and joiner—plus `bpe.model` from
`csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26` at commit
`672fbf1b30579d6585301139bb363f42a0ad4a24`. The source BPE SHA-256 is
`c53433de083c4a6ad12d034550ef22de68cec62c4f58932a7b6b8b2f1e743fa5`;
the exported vocabulary SHA-256 is
`f191a4935f668fa8cd8e607bcd378404f948321cd3134a5ea13d324ba921673d`.
The selected high/fast ASR filenames and SHA-256 values are pinned beside those
coordinates in `tools/setup_models.py`.

Before a new family download, require SentencePiece 0.2.1. Normalize downloaded
cache links into regular local files, verify every ASR/BPE digest, fsync a unique
sibling staging directory, and rename it without replacement. Its immutable
address binds the repositories, revision, filenames, ASR hashes, BPE source
identity/hash, and exported-vocabulary hash. `--force` validates and reuses an
existing address; it never mutates it.

One compare-before-replace config publication selects the four ASR paths and
sets `asr_modeling_unit`, `asr_bpe_vocab`, `asr_bpe_vocab_sha256`, and
`asr_hotwords_case_policy`. Refuse publication if the machine-local config
changed during setup. Do not add or rewrite phrases and do not touch VAD, TTS,
final ASR, speaker enrollment, Obsidian, reminders, or app grants. Ordinary
model setup preserves a complete selected BPE family and rejects an incomplete
one instead of silently rewiring its ASR quartet.

Readiness and runtime both parse the bounded vocabulary, verify its configured
digest, and reject malformed phrases/context. Active hotword stream creation is
fail-closed: a Sherpa build that rejects per-stream hotwords is not retried with
a plain stream. Apply that contract to local device, file replay, and LiveKit;
LiveKit waits for worker startup before entering its transport loop. Keep empty
hotwords on the plain, byte-compatible stream path. Bind consumed vocabulary
bytes into recording and capture-replay provenance.

The lean installer adds SentencePiece only for explicit `--bpe-hotwords`; the
full development requirements retain the exact pin. Keep the single normal app
entry and optional capability-provider architecture unchanged.

## Context / why

The 2026-07-16 physical vault run recognized `vault` in 0/6 admitted unclipped
windows. ADR-0078 correctly left plain-text hotwords unpromoted because the
English Zipformer is BPE-based. The runtime supplied a score and per-stream
phrases but omitted sherpa-onnx's required `modeling_unit=bpe` and `bpe_vocab`,
so it could not encode the intended English context. A SentencePiece inspection
showed that lowercase `vault` crosses the unknown token while uppercase `VAULT`
uses valid pieces.

Committing inferred phrases, crawling Obsidian content, changing the final
selector, or creating a vault-specific app entry were rejected. Those would
broaden private-data access or unrelated transcript behavior without target
recordings. A model swap remains separate from this context repair.

## Evidence and limits

- The focused device-free gate in `docs/agent-testing.md` passed 366 tests with
  one optional-model skip. It covers
  local/replay/LiveKit failure, setup immutability, readiness, installer, and
  provenance with fakes; it does not construct the real model.
- The retained aggregate-only six-clip report is
  [`2026-08-02-bpe-hotword-recorded-ab.json`](../evidence/2026-08-02-bpe-hotword-recorded-ab.json)
  (SHA-256
  `c793d912bf1349f93267c1b0d181dfbf76c7a65a8a369a9dd398041b34ed1a8c`).
  Baseline and candidate both had streaming WER 0.20/CER 0.1724, selected WER
  0.00, zero keyword attempts, zero wins/losses, and six ties. These are the
  same owner's non-disjoint recordings, and no separate command/exit receipt
  was retained; the result is not a promotion or live-validity claim.
- An earlier ad-hoc real-model construction had no durable command/exit receipt
  and is not accepted evidence. No physical microphone, endpoint, AEC,
  barge-in, speaker-identity, or conversational-quality validation ran here.

## Consequences

- With no phrases, shipped runtime behavior stays on the plain stream path.
- A configured phrase cannot silently lose its requested bias on an older or
  incompatible Sherpa runtime; startup fails with an actionable error.
- Candidate phrase lists and any non-default score overrides remain
  machine-local. Promotion still requires fresh, exact target and negative
  recordings followed by recorded and physical live A/B validation.
