# ADR-0054: Bound exact recalled self-facts without promoting memory turns

Date: 2026-07-12
Status: superseded-by ADR-0060

## Decision

Extend ADR-0051's controller-owned session facts with one separate, bounded
same-session self-scalar contract. When episodic recall is enabled, accept only
a live, successfully stored `user` utterance matching anchored
`my <subject> is <value>` grammar and answer only the exact
`what is my <subject>?` form. Keep at most sixteen natural self-facts, require
the exact backing memory item to remain present, reject multiline, compound,
prompt-injection, and recognized PII/credential values, force the answer
PRIVATE, and bypass models, planners, tools, and actions. Never build trusted
controller state by parsing recalled, profile, visual, or last-session text.

Keep general recalled/profile/last-session context on the existing router and
inside its untrusted-data fence; a rendered memory block alone must not promote
a turn to the main tier. Run autonomous voice/barge tests with distinct shipped
main and fast model arguments. In the memory probe, grade retrieval availability
separately from model-prompt injection and label a deterministic answer as
`control`.

## Context / why

The real MiniCPM5-1B Q8 path retrieved `my favorite color is teal` correctly but
answered the exact preference question with a privacy/persona disclaimer in five
of five trials. It still answered ordinary dog-name and meeting recall three of
three each; Gemma answered the preference probe, so this was a narrow preference
wording failure rather than broken recall plumbing. Prompt reordering and stronger
instructions did not fix it without weakening the untrusted-memory boundary.

Routing every nonempty memory block to Gemma was rejected. The RAM selector can
render the favorite-color item for unrelated questions through stopword overlap;
durable profile content can appear on every turn; and a last-session recap appears
on the first eligible turn. Blanket promotion would therefore erode ADR-0020's
MiniCPM fast-tier adoption, add main-model latency/VRAM pressure, and could send
remembered text through an explicitly enabled cloud-wrapped main chain. The narrow
controller follows the existing bounded-control precedent without claiming a
general semantic-memory or cross-session solution.

The autonomous voice harness also previously passed one model name to both roles.
ADR-0051 permits all-role replacement only as a diagnostic, so that topology could
not validate the shipped MiniCPM-fast/Gemma-main deployment.

## Consequences

- Exact live self-scalars are deterministic and private; disabled recall,
  nonexact questions, stale/evicted backing state, memory errors, suspicious
  content, and preloaded untrusted memory all fall through unchanged.
- Explicit `remember for this conversation` facts retain their own sixteen-entry
  cache and cannot be evicted by natural self-facts.
- General episodic/profile/cross-session recall remains a fenced model task and
  is not declared fixed by this decision.
- The strengthened real memory probe passed four of four with MiniCPM handling
  ordinary turns and the controller returning the exact scalar; warm five-turn
  duration was 1.6–2.2 seconds after an 8.3-second cold run.
- Autonomous voice and barge stress now preserve distinct main/fast roles, and
  specsim reports both configured roles while labelling latency as a fast-path
  estimate. Bare-speaker v5 enrollment and live barge-in remain required.
