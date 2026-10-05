# ADR-0226: Require one complete addressing decision

Date: 2026-10-05
Status: accepted

## Decision

Validate the whole addressing-model reply as one complete normalized decision
token. Preserve ACT, INGEST and UNSURE, the existing single-token ACTION/ACTIVE
aliases and the existing surrounding punctuation/case cleanup. Explanations,
lists, competing labels and reasoning content must yield UNSURE, even if their
first word is ACT. Never search within reasoning or quoted prose for a decision
label; a complete quoted token retains the existing normalization behavior.

Keep deterministic explicit-request shortcuts, model prompting/selection and
the caller's UNSURE policy unchanged. Shipped desktop profiles use the existing
conservative policy, so a malformed learned verdict grants no reply admission.
This does not make an exact but semantically incorrect ACT trustworthy.

## Context / why

The owner reported random room noise preceding unsolicited speech. Native
ASR/addressing/reply quality remains unresolved. A separate public-text probe
also exposed a concrete parsing bug: model recitations beginning with
`ACT, INGEST, or UNSURE...` or `ACT if it is...` were admitted because the parser
selected the first word and ignored the remainder, despite the one-word output
contract.

The 24-case public probe at seed 0, temperature 0, 4096 context and 64 output
tokens produced no exact labels for the current or two shorter prompts. The
existing parser admitted all 12 negative current-prompt cases. Shorter prompts
lost positive admissions, so no prompt replacement was adopted. This is one
bounded diagnostic configuration, not a universal model-quality verdict.

## Consequences

- Addressing tests pass 47; adjacent runtime, speaker-authority, final/pre-token
  cancellation, Ollama async cancellation and cleanup tests pass 227.
- New runtime regressions observe zero answering calls, capability dispatches
  and speech on malformed ACT-leading outputs under conservative policy.
  Clean ACT still dispatches and speaks. Existing explicit shortcuts remain.
- Loading only the historical e3e419b parser into those regressions produces
  all four expected malformed-response failures; the clean ACT control passes.
  The implementation therefore repairs a reproduced admission boundary defect.
- Output labels can still be wrong. A separate 12-case streaming JSON-enum
  diagnostic gives MiniCPM valid format on 12/12 but correct semantics on only
  6/12; every negative was ACT. Cached Gemma3 matches 12/12 in that small sample.
  No provider constraint, model/default change or live qualification follows
  from this comparison; those require their own evidence and review.
- The compatible isolated enrollment remains unpromoted. Ambient-noise
  admission, instruction-copying, answer quality, owner STOP/talk-over and
  performance acceptance remain open; private audio/evidence stay local.
