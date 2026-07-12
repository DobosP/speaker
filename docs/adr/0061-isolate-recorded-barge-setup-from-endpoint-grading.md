# ADR-0061: Isolate recorded barge setup from endpoint grading

Date: 2026-07-12
Status: accepted

## Decision

Refine ADR-0053's concurrent recorded-owner barge driver without changing its
two-block interruption policy or production runtime. Prepend the same 400 ms
deterministic noise-floor settling lead used by the primary injection harness.
For the base utterance that exists only to start a long assistant reply, disable
semantic early endpointing and set both Sherpa acoustic trailing-silence rules to
2.4 seconds, so the complete hash-pinned window drains before one final commits.

Continue grading every labelled base clip separately through `FileReplayEngine`
with the selected production ASR/final policy. In the concurrent test, retain
the labelled-base overlap assertion, exact task/metrics token, true sink onset,
armed floor-only no-cut control, production two-block owner sustain, causal
sample receipts, 1.0-second owner-to-FIFO ceiling, and terminal worker teardown.

## Context / why

The promoted `utterance-01` window is stable when decoded as a complete clip,
but includes an internal pause. In real-time injection, the semantic/rule-2
endpointer can commit an earlier decoder startup hallucination before reaching
the labelled `Tell me a long story` tail. The strict gate had passed, then failed
twice with unrelated finals such as `I should think...`; the recorded cut itself
still landed causally in about 0.52 seconds. A 400 ms settling lead alone did not
prevent the early endpoint.

Weakening the ASR assertion or substituting manifest text was rejected because
either would manufacture setup evidence. Re-extracting a different same-run base
was not available: the only other same-session clip is the STOP control, which
cannot start playback. Endpoint latency is not the behavior this concurrent
scenario measures, and the six production-policy per-clip tests already own that
axis, so separating the two test responsibilities removes timing flakiness
without hiding an audio or interruption failure.

## Consequences

- The previously failing owner-overlap case passed four consecutive isolated
  runs; the complete strict recorded gate then passed 9/9 in 80.34 seconds.
- The concurrent setup takes longer because it waits for a 2.4-second acoustic
  tail. It still requires one real ASR final containing the labelled request;
  blank, wrong, duplicated, or manifest-substituted text remains red.
- Production endpointing, injected command timing, owner sustain, barge policy,
  model/audio artifacts, and private waveforms are unchanged.
- This remains deterministic no-device evidence only; current-room bare-speaker
  barge-in and fresh v5 enrollment are still mandatory.
