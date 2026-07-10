# ADR-0016: Shared selected-profile runtime readiness fails closed

Date: 2026-07-10
Status: accepted

## Decision

Resolve and apply the requested device profile once, then use the production
`core/readiness.py` contract over that same merged configuration for doctor,
live-session, and normal native-voice readiness.
Before a normal `--engine sherpa` launch, fail with actionable diagnostics when
load-bearing model, LLM, audio-device, or selected audio-front-end prerequisites
are broken. An active word-cut path requires an existing VAD model; on Linux it
also requires a loaded `module-echo-cancel` whose source and sink are the active
capture and playback routes. Require LiveKit APM only when AEC is enabled and
its selected backend is APM. Keep console, replay, LiveKit transport, and
one-shot enrollment outside this native-audio gate, and never require Ollama for
`--llm echo`. The live harness's explicit `--inject` mode keeps model/front-end
checks but intentionally skips physical-device and OS-route checks because it
replaces both streams; acoustic/default `--check` does not get that exception.

## Context / why

The same machine could report contradictory readiness: `tools.doctor` correctly
failed on missing audio devices and the ADR-0013 OS echo-cancel dependency while
`tools.live_session --check` reported ready because it only imported
`sounddevice` and checked non-empty model strings. Normal sherpa startup did not
consume either verdict, so a word-cut configuration could launch on a raw mic
and speaker even though its text authority cannot distinguish near-end speech in
that route. Doctor also applied `--device` to Ollama model selection but
independently re-resolved the configured device for audio-front-end checks.

Checking only that the echo-cancel module is loaded is insufficient: if capture
still uses the raw source or playback bypasses the echo-cancel sink, the module
has no usable near-end/far-end pair. Treating all `aec_backend="apm"` strings as
active was also wrong because a backend setting is inert while
`aec_enabled=false`.

## Consequences

Readiness failures now stop the normal native voice CLI before models or device
streams are constructed, while diagnostic and headless paths remain usable.
PipeWire/Pulse inspection is dependency-injected and pure at the decision seam,
so module-loaded-but-misrouted and device-profile override cases are deterministic
tests. An active Linux word-cut configuration needs `pactl` state that proves
both routes; an unverifiable route fails rather than silently degrading.

This establishes configuration and route readiness only. It does not prove
acoustic quality or cut rate, and it changes no sherpa barge state machine. The
first manual gate after landing is an escalated/hardware run of doctor followed
by bare-speaker talk-over, bare `stop`, and silent-control behavior on the
reported routed defaults; none of that live validation ran in this change.
