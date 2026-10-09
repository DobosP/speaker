# ADR-0228: Bound local model decisions independently of answers

Date: 2026-10-09
Status: accepted

## Decision

Use `collect_llm_decision` for small local label decisions. Bound each request to
at most 16 output tokens, 256 collected characters and a three-second monotonic
cancellation deadline. Accept only a complete selected raw or JSON-string label,
with the prior bounded single-token punctuation/case normalization. Invalid,
oversized, or expired output returns no decision; external cancellation retains
`LLMCallCancelled` and never becomes a fallback classification.

Copy scoped output settings into Ollama requests (temperature zero, no thinking,
JSON string-enum format) and llama.cpp requests (temperature zero, native label
GBNF). Preserve any smaller positive output cap. Keep model, context, residency,
answer options, history and native tool-chat contracts unchanged. Carry the typed
immutable request through the existing factory source-owner context snapshot.
Narrow egress to LOCAL_ONLY; direct OpenAI-compatible clients are refused because
this helper cannot attest their locality/cancellation contract. Factory cloud
wrappers may use only their existing local leg.

Wire capability-router disambiguation to the helper; unavailable decisions retain
the heuristic action. Addressing integration belongs to the conversational
admission change. This decision does not select a new model or prove a label's
semantic correctness, reply quality, microphone behavior or phone performance.

## Context / why

ADR-0226 repairs whole-label validation, but a wrong exact ACT remains wrong.
The same answer client previously supplied its full output budget to short label
requests, and classification waited for the complete stream. A public constrained
format diagnostic already separated format correctness from semantic accuracy;
format constraints alone cannot cure ambient activation or instruction recitation.

## Consequences

- Answer requests retain their exact original options after a decision, including
  context, token cap, thinking preference and keep-alive. Independent requests do
  not mutate shared client settings.
- Native-free tests cover output bounds, malformed labels, grammar/enum requests,
  original answer budgets, wrapped source-owner propagation, local-only routing,
  context restoration, deadline cancellation before first token, and external
  cancellation winning simultaneous completion.
- The deadline reuses the existing provider cancellation seam. It does not
  preempt an injected blocking iterator or a native backend that ignores abort;
  no new detached timeout worker is created. Ollama cancellation is asynchronous;
  native cleanup/return and physical latency remain distinct obligations.
- Public-only persona/answer-quality diagnosis is separate pending work. The
  original enrollment, isolated candidate and private recordings are unchanged;
  no microphone, doctor, live test, cloud call or model download is authorized by
  this implementation receipt.
