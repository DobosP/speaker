# ADR-0067: Route explicit recent-thread referents to the main tier

Date: 2026-07-13
Status: accepted

Refines: ADR-0051

## Decision

Extend ADR-0051's bounded lexical context-follow-up signal with exactly `you
just named`, `you just mentioned`, and `you just said`. Apply the existing 0.30
nudge only when `recent_conversation` is a nonblank string. Preserve profile
thresholds: the shipped desktop 0.30 threshold routes an otherwise-cheap
matching turn to main, while the phone 0.55 threshold remains fast; absent,
blank, or malformed recent context contributes no nudge. Do not add broader
`you named`, `you said`, bare-verb, or history-presence rules.

Advance the fourteen-scenario conversation gate to scenario-set v3. Replace its
fast-only boolean with an exact per-scenario answer-route tuple. Require both
successful task-completion routes and direct model-call roles to match that
tuple. The history scenario must use `(fast, main)` while retaining its exact
two-message `(user, assistant)` role sequence and privacy-safe per-message
content hashes. Retain every other ADR-0051 safety, provenance, topology,
warm-up, coverage, and three-repetition contract.

## Context / why

Correct role-structured history did not make MiniCPM reliable for the explicit
semantic reference. Repeated real probes retained the exact two-message history
but answered only 18/20 correctly on fast/fast. An interleaved local diagnostic
was 18/20 on fast/fast and 20/20 on fast/main, with about 0.29 seconds additional
mean two-turn latency. These probes identify the repair; they are not clean-
revision adoption evidence.

Prompt augmentation was rejected because an actual interleaved comparison
regressed from 30/30 unchanged to 29/30 guided and reproduced the same identity/
location hallucination. A controller regex answer was also rejected: resolving
the referent remains general semantic answering. Routing every history-bearing
turn was rejected because it would tax unrelated desktop asks and the shared-
tier phone profile without evidence.

## Consequences

MiniCPM answers the first factual turn and Gemma resolves only the evidenced
explicit recent-thread referent on desktop. No-context, malformed-context,
near-match, unrelated, and phone behavior remain on the fast path unless an
independent existing signal legitimately escalates them.

Route assertions prevent a semantically lucky answer on the wrong tier from
passing, while the existing role/hash assertions continue to prove history
ownership without recording raw history. Pre-commit device-free evidence is
162 focused tests passed and deterministic v3 42/42 with `(fast, main)` on all
three history repetitions. A clean-revision full production-hybrid 42/42 +
42/42 A/B remains required. This decision opens no audio device and does not
validate physical bare-speaker barge-in.
