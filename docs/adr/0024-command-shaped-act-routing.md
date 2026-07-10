# ADR-0024: ACT routing requires a command-shaped request

Date: 2026-07-11
Status: accepted

## Decision

Classify deterministic ACT markers only when they lead the request after a
bounded courtesy/request prefix. Recognize separable device phrasal verbs such
as "turn the lights off" without allowing their words to match elsewhere in an
informational question. Treat a leading ambiguous action term such as `alarm`,
`open`, `play`, or `brightness` as low-confidence ACT so MiniCPM can disambiguate it;
when MiniCPM demotes it to SIMPLE, recompute the ordinary answer tier instead of
retaining the provisional ACT decision's main tier.

## Context / why

The capability router used raw substring membership and assigned every hit 0.70
confidence, above the shipped 0.65 MiniCPM-assist threshold. Ordinary questions
such as "What is open source software?", "What is alarm fatigue?", and "How
does email encryption work?" therefore became ACT turns. On the 4090 profile
the first example spoke the slow-turn acknowledgement, made three Gemma/planner
calls, and made no MiniCPM answer call.

Replacing substring matching with only word boundaries was rejected because
`alarm`, `brightness`, and `play` are still whole words in those questions. A
blanket informational-question allowlist was also rejected because it would be
brittle and would not preserve real request shapes such as "could you turn the
volume down" or the separable form "turn the lights off". Leading request
grammar supplies the needed distinction while remaining deterministic and
phone-safe.

## Consequences

- Informational action-word questions remain SIMPLE and keep their normal
  fast/main tier and latency policy; the common short cases stay on MiniCPM.
- Clear imperatives, courtesy-prefixed requests, and separable device controls
  remain confident ACT turns.
- Ambiguous action-leading fragments pay at most one memoized MiniCPM classifier
  call on profiles with LLM assist; profiles without it retain conservative ACT.
- Indirect action wording that does not lead with a supported request shape may
  fall back to SIMPLE. Add a bounded grammar case when real usage demonstrates
  one; do not restore arbitrary substring matching.
