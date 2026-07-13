# ADR-0068: Make anchored self-correction controller-owned

Date: 2026-07-13
Status: accepted

Refines: ADR-0051

## Decision

Apply the existing bounded repair for exactly `What is the <subject> of OLD, I
mean NEW?`, where `OLD` and `NEW` are single tokens, before invoking transcript
cleanup. Publish the repaired question directly and make no cleanup-model call
for that grammar. Keep clean transcripts on their existing deterministic bypass
and leave every broader signalled correction with the configured fast cleanup
model. Do not expand the grammar or infer a general correction syntax.

Advance the fourteen-scenario conversation gate to scenario-set v4. The bounded
correction scenario must publish exactly `What is the capital of Japan?`, pass
that exact query to `assistant.answer`, and contain no cleanup-model call. Retain
the scenario-set v3 route, history, topology, identity, warm-up, provenance,
coverage, and three-repetition contracts unchanged.

## Context / why

The first clean-revision production-hybrid v3 A/B passed Gemma 42/42 but passed
MiniCPM only 40/42. Both failures were the correction scenario: MiniCPM's cleanup
call succeeded, yet the runtime published the unchanged `France, I mean Japan`
question and the answer omitted Tokyo. The cleanup reply is intentionally not
persisted, but a device-free stub reproduced the only normal cross-layer path:
a short `Japan` rewrite satisfied the cleaner's target-word check, then the
runtime's anti-overreach guard rejected it and restored the raw utterance.

Forcing the deterministic result only after a model call was rejected because a
provider error or cancellation could still leak the raw correction, and every
valid match would retain roughly 220 ms of unnecessary sampled work. Weakening
the anti-overreach guard was rejected because it protects against observed
phantom turns. A broad regex cleaner was rejected because ambiguous repairs
remain semantic model work; the accepted grammar is the already-bounded ADR-0051
contract.

## Consequences

This exact question correction no longer depends on MiniCPM wording, provider
availability, or cleanup latency. Its repaired text still loses acoustic action
authority like every model- or controller-rewritten final. Broader timers,
messages, memory statements, and multi-token repairs continue through the fast
cleanup model.

Pre-commit device-free evidence is 96 focused tests passed, the full suite at
3883 passed/31 skipped/9 warnings, and deterministic v4 42/42. A clean-revision
production-hybrid MiniCPM/Gemma 42/42 + 42/42 A/B remains required. This change
opens no audio device and does not validate physical bare-speaker barge-in.
