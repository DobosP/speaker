# ADR-0055: Fail autonomous voice verdicts closed

Date: 2026-07-12
Status: superseded-by ADR-0058

## Decision

Grade autonomous voice and repeated barge runs through one pure verdict policy.
A voice pass requires a ready engine, a parseable run bundle, non-silent capture,
an STT/LLM/TTS round trip, the expected count of true first-audio turns, one
non-blank hypothesis per labelled clip, mean WER at most 0.50, and no runtime
errors or stuck hints. Echo-capable delay/speaker runs additionally require a
reply that does not cut itself and a second reply that is cut by the injected
talk-over. Play that talk-over asynchronously, measure from its expected speech
onset, and accept only a finite causal cut in 0–1.0 seconds. Apply the same
artifact, first-audio, error, self-cut, every-talk-over, and latency requirements
to repeated barge stress. Label echo-free cable runs `diagnostic_pass` with
self-interrupt, barge, and cut latency `not_covered`; expose `ok=false`, and make
the aggregate `all` command exit 2 while any required coverage is incomplete.

## Context / why

The prior harness treated bundle existence, monitor RMS, and one `speaking:` log
as enough. It could therefore report PASS with missing labelled transcripts,
100% WER, watchdog/runtime failures, a reply that never delivered audio, failed
self-interrupt/barge scenarios, or cable mode where echo and barge do not exist.
Repeated stress returned `ok=true` unconditionally and excluded replies that
never started from its true-positive denominator. Its talk-over timer also began
after blocking `paplay` had finished, so printed latency was not a cut-time
measurement. Keeping these outputs as advisory prose was rejected because shell
automation and later agents consumed their booleans as landing evidence.

## Consequences

Cable remains a useful silent STT diagnostic, but cannot satisfy mandatory
barge coverage or make `tools.autotest all` green. Delay mode can provide
device-free echo-path evidence; speaker mode remains audible and requires an
explicit sound opt-in. Expected injected onset is still not physical human onset,
so only a real bare-speaker run can close the live acoustic/ear-grade gate.
Verdict policy is device-free and unit-tested; runners collect evidence without
duplicating acceptance rules.
