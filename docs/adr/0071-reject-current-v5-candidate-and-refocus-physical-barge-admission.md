# ADR-0071: Reject current v5 candidate and refocus physical barge admission

Date: 2026-07-14
Status: accepted

## Decision

Mark the 2026-07-14 physical bare-speaker gate red and reject its isolated v5
enrollment candidate. Do not promote or activate that candidate. Once the scalar
evidence and operator verdict are durable, delete the rejected candidate, its
prepared worktree config and run artifacts, and the redundant preparation
backup. Preserve the active historical v4 enrollment and primary config.

Keep the landed enrollment, promotion, session-voice-lock, calibrated admission,
virtual echo-route, and deterministic delay capabilities. The enrollment-off
diagnostic was an in-memory experiment, not a production default: do not remove
speaker identity or enable the private calibrated fallback in production based
on it.

Start the next physical barge effort before identity verification. With
enrollment disabled in memory, instrument the playback-time path from physical
capture through OS echo cancellation, settled calibration, VAD, energy admission,
denoise, and the isolated word-cut decoder. Prompt exact `STOP` is the first live
acceptance gate. Do not prepare another enrollment candidate or resume the wider
dialogue A/B until that gate cuts promptly and causally without a self-cut.

## Context / why

The deterministic foundation was green: 4049 tests passed with 31 skipped, the
APM/DTD regression passed 6/6, and two private virtual-delay runs cut once each
at 0.509 and 0.818 seconds from calibrated capture onset with all route and
cleanup proofs.

The isolated live enrollment also completed correctly: three roughly 12-second
passes produced a 512-dimensional v5 candidate with pass similarity minimum
0.60 and mean 0.67, and the prepared profile reached doctor `READY`. That proved
the capture and provenance workflow, not usable barge behavior.

In enrollment-on run `20260714-192151`, every spoken reply retained `sid=0` and
a normal France question received the correct Paris answer. The soft exact Yes
was dropped, a one-second pause split one request into two turns, the first
actual cut arrived about 27.8 seconds after reply playback began, override text
was garbled, and the operator had to repeat the override. Exact Stop did not
stop the reply promptly. Sub-millisecond detection-to-cancel receipts on the two
late cuts do not establish speech-onset-to-cut latency.

In enrollment-off run `20260714-193713`, exact Stop again produced no word-cut
trace, cut handoff, barge event, or Stop final. A near-end energy burst was
visible while VAD remained zero. Across both runs, the steady playback-time
input was tens of times quieter than the calibration interval and the calibrated
energy fallback never started. This does not prove which upstream stage is
wrong, but it places the blocking failure before speaker-identity acceptance.
Removing enrollment therefore does not solve barge-in.

## Consequences

- Physical bare-speaker barge-in remains unvalidated and unlandable despite the
  green headless, real-model, semantic-memory, and private virtual gates.
- Voice continuity gained useful live evidence: all resolutions in both runs
  stayed on `sid=0`. That isolated success does not make the overall live gate
  green.
- ADR-0066 remains the promotion contract for a future explicitly accepted
  candidate. ADR-0069/0070 remain valid device-free evidence; none substitutes
  for physical behavior.
- Cleanup intentionally makes the rejected biometric candidate and detailed
  session artifacts unavailable. The dated closeout preserves only the bounded
  scalar evidence and operator verdict needed to avoid repeating this direction.
- The next session must distinguish route/calibration settling, playback-time
  VAD/energy starvation, residual-echo floor handling, and decoder admission
  before changing identity or dialogue policy.
