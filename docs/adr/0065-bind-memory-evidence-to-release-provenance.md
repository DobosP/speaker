# ADR-0065: Bind real memory evidence to release provenance

Date: 2026-07-13
Status: accepted

## Decision

Refine ADR-0060's autonomous-probe evidence contract without changing or
superseding its persistent-memory and routing decision. A green real-model
memory result requires the recalling turn to select the main tier first and
exclusively, retain PRIVATE sensitivity, contain the fact inside one balanced
untrusted-data fence, leave native chat history empty, avoid controller
authority, and ground the answer in one affirmative clause that also names the
non-personal canary subject. Contradictions, corrections, alternatives, and
subject/value claims split across clauses abstain.

Bind that result to one clean, stable 40-hex Git revision and to stable full
model identities before and after the probe. Repository state is captured
before the first effective-config read; the effective release contract is
recomputed before the final repository snapshot. The fast role must be the
configured, pinned MiniCPM5-1B Q8 alias and the main role the exact configured
Gemma model. Recompute each record's required flag, verification kind, exact
model/role assignment, immutable blob, MiniCPM pinned source/blob/Q8/template/
parameter fields, and effective configuration through the shared conversation
provenance validator. Use one explicit ambient-credential-isolated loopback
Ollama transport for inference and inspection.

Persist the sanitized host, stable role/model identity bundle, repository
snapshots, and a SHA-256 binding the probe settings and input texts. The digest
binds the result to that captured contract; it does not make undisclosed input
texts or local configuration reconstructable.

Describe the deterministic boundary precisely: the probe closes and reopens
the SQLite backend and constructs a fresh capability registry in one Python
interpreter. It does not claim a fresh OS process. Echo performs no Git,
configuration, or model inspection and remains an incomplete diagnostic even
when all plumbing checks pass.

## Context / why

ADR-0060 correctly fenced durable SQLite recall and promoted strong subject
matches, but its evidence could still turn green after a fast-tier failure fell
back to main, from keyword and subject text in contradictory clauses, without
checking PRIVATE sensitivity, or from a dirty revision and mutable model
aliases. It also called same-interpreter backend recreation a fresh process.
Those gaps made a real semantic answer useful diagnostic data, not release
evidence.

Duplicating repository and model inspection in the autonomous CLI was rejected
because the conversation and memory gates could drift. Their privacy-safe
repository, role, blob, and effective-configuration snapshots now share one
module while model-specific verification remains in the common identity API.

## Consequences

- Fallback, wrong-sensitivity, contradictory answer, dirty-revision, release-
  contract drift, role drift, model drift, unsafe host, and missing or weak
  identity cases fail even when the answer contains the canary.
- Reports bind a result to sanitized release evidence without claiming that a
  digest alone reproduces local inputs or an Ollama response.
- Echo remains cheap and device-free but cannot establish semantic correctness
  or release provenance.
- The real clean-revision MiniCPM/Gemma memory run remains a separately
  authorized post-commit gate; deterministic tests cannot substitute for it or
  for live audio validation.
