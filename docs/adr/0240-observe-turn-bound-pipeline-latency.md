# ADR-0240: Observe turn-bound pipeline latency separately from legacy metrics

Date: 2026-10-10
Status: accepted

## Decision

Provide a separate `latency_stage_breakdown` run-summary field with five observed
intervals: ASR final to answering-model request, request to first token, token to
first speakable text, text to tracked TTS admission, and admission to the runtime's
observation of an exact started receipt. Keep `TurnRecord.as_dict()`, legacy
first-audio stamps, local TTFT/TTS EWMAs, routing and watchdog inputs unchanged.

The new `LLM_REQUESTED`, `TTS_TEXT_READY`, `TTS_ADMITTED` and
`TTS_RENDER_START_OBSERVED` boundaries require an explicit current positive metric
turn token. Missing, stale, invalid or out-of-order boundaries stay unknown. The
producer owns model-request and first-text observations; runtime output owns
admission and onset-receipt observations. `MetricsRecorder.stage_breakdowns()`
takes a locked scalar snapshot, and `RunLog.finalize(stage_breakdown_records=...)`
exports only named finite non-negative interval values and their aggregates.
Producer and CLI finalization call sites are integrated in the companion work;
the isolated instrumentation gate supplies deterministic producer observations.

## Context / why

Existing ASR-final-to-token and token-to-audio numbers combine classifier/cleanup,
model dispatch, sentence aggregation, synthesis, queueing and output. A long first
sentence can therefore look like a slow voice model. Pipecat 1.12.0's
[observed latency contributions](https://github.com/pipecat-ai/pipecat/blob/1559a684b1ee9771b36454b72418d7364b518e7f/src/pipecat/observers/user_bot_latency_observer.py#L367)
illustrate separating settings-controlled waits from service time without
requiring another model or replacing the control plane.

The existing engine-global `TTS_FIRST_AUDIO` and potentially auxiliary
`TTS_REQUESTED` are unsuitable ownership evidence for the new intervals. A
runtime-only map binds at most 64 pending normal fragment IDs to captured metric
turn/task/binding identities. It is installed before `speak_tracked`, including
synchronous callbacks, and requires tracked terminal plus exact-start capability.
Auxiliary output, latency acknowledgements, legacy/nonexact sinks and missing
tokens do not populate the new output stages. Entries retire on start, terminal,
failed handoff, registration rollback, global interruption, exact task terminal
and shutdown; foreign IDs/tasks do not consume another fragment's observation.
This map grants no output, speaker or action authority.

## Consequences

The final interval is **admission to receipt-dispatch observation**, not a DAC
timestamp, first PCM, isolated synthesis time, pure GPU time or queue measurement.
All five intervals can be populated only when their corresponding boundaries
were observed for the same turn. Absent values remain null, with no inferred
zeros or legacy callback fallback. Raw text, audio, absolute timestamps, task IDs
and metric tokens are absent from the additive summary field. No per-token work,
new model, native pool, backend cap, route or cancellation policy is introduced.

Headless qualification: 79 focused tests passed in 2.43 s; the adjacent
playback/history, streaming/resume, runtime, routing and watchdog final gate passed
392 tests in 9.69 s. The formatted new receipt module separately passed 20 tests
in 2.45 s. These fake-clock and controlled-engine checks establish ownership and
serialization behavior, not actual audio latency. Linux/Windows/macOS physical
validation remains open; owner live testing is stopped. No native inference,
recording, microphone, doctor, network or device execution ran for this change.

### Producer integration

Assistant/research request and first-speakable-text marks use only the token
captured in dispatch metadata; the older first-token fallback is unchanged.
The app exports the separate stage snapshot during finalization. Missing dispatch
provenance stays unknown. Producer regressions also found a pre-existing
nonstream cancellation seam: a wrapper can stop without yielding another token.
The collector now rechecks cancellation at exhaustion and closes the provider,
so its partial text is not reported as a completed answer or marked ready.
