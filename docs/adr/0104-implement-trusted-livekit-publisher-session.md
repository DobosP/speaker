# ADR-0104: Implement the trusted LiveKit publisher session

Date: 2026-08-01
Status: superseded-by ADR-0164

## Decision

Implement ADR-0096's non-escaping publisher session in
`core.engines.livekit_agents_session` against exactly `livekit==1.1.14` and
`livekit-agents==1.6.7`. Load them lazily, validate both versions and the public
`RoomIO` and `AgentsConsole` surfaces, and keep the raw SDK session, agent,
RoomIO, output, and speech handles behind Speaker's wrapper. Accept only
already-built, audited local STT/VAD/TTS objects. Keep framework LLM,
Agent/session tools, MCP, recording, remote turn inference, aligned transcript
transforms, and preemptive generation disabled.

Compose the media route manually. Retain the exact disabled `AgentsConsole`
singleton, reject either its `enabled` or `record` state before construction or
start mutation, and recheck that same object immediately before and after SDK
session start and before admission. Construct and own
`RoomIO(session, room, options)` with exact publisher identity, microphone
publication, and `pre_connect_audio=False`. Require an observed exact
`CONN_CONNECTED`, then perform this order:

1. `RoomIO.start()`;
2. `AgentSession.start(agent, record=False, capture_run=False)`, with no `room`
   or `room_options` argument;
3. a bounded `RoomIO.wait_for_ready()`;
4. proof that session input/output are the exact enabled manual RoomIO objects.

Until all four steps succeed, suppress every non-close SDK event and completed
turn at the wrapper boundary. A failed or closing start never reopens inbound
event admission.

Never activate the SDK's implicit `RoomSessionTransport`, `SessionHost`,
`RecorderIO`, or pre-connect-audio path. A completed-turn committer must be
bound before the first media mutation. Partial event binding rolls back, and
new callbacks, committers, starts, or engine restarts are rejected once close
begins or the wrapper has closed.

Own output route generations and fail closed on SDK mutation exceptions or
partial mutations. Track replace/unpublish and every enable/detach transition
invalidate older leases. `AgentOutput.replace_audio_tail()` is deliberately
hard-rejected without calling the SDK: in the pinned source, direct manual
RoomIO has no `_AudioSinkProxy`, so that method falls back to replacing
`output.audio` and would displace the trusted participant-scoped publisher.
Speech admission requires connected state, manual readiness, exact attached
RoomIO output identity, and exact enabled state. After media mutation, every
exact-publisher disconnect reason is terminal; a same-identity reconnect cannot
reopen the session or release buffered input. Transcript, user/agent state,
false-interruption, and completed-turn callbacks also remain inside the wrapper
until successful start, manual RoomIO readiness, and connected-state admission;
the SDK close callback remains available to drive cleanup at every stage.

Treat the SDK close event as necessary but not sufficient disposal proof. One
shared, retryable async finalizer waits for in-flight starts, obtains SDK close
proof when `AgentSession.start()` returned successfully, detaches manual I/O,
awaits `RoomIO.aclose()`, and only then resolves retained handles, publishes
outward close, unbinds callbacks, and marks the wrapper closed. Rejected-start
cleanup may dispose manual RoomIO before waiting for SDK close. A timeout or
cleanup exception never reports a false closed state; later `aclose()` retries
the same unfinished proof. Only a close before any start mutation may terminate
without RoomIO cleanup.

Keep `VoiceSessionPlan.is_selectable` false for `trusted-lan` and preserve its
application hard failure. Promotion requires a separate decision after the
exact optional SDK is installed and exercised through the real self-hosted
route and the publisher/device live A/B passes.

## Context / why

ADR-0096 intentionally landed only the adapter protocol. An audit of the exact
1.6.7 source found that passing `room` into `AgentSession.start()` creates both
RoomIO and a primary room session host, including an unscoped remote session
control transport. Console mode can override attached audio, create another
session host, and enable recorder I/O despite a local `record=False`. Manual
RoomIO is the public composition that provides publisher-scoped media without
those control paths.

The same source detaches session I/O before its `close` event, but an externally
constructed RoomIO is not an SDK-owned resource and must be closed by its
caller. Therefore SDK close alone cannot release Speaker's uncertain playback
ownership. Likewise, `wait_for_ready()` and exact current output identity are
needed before media admission; construction success alone proves neither
publication readiness nor route ownership.

The pinned `AgentSession.start()` mutates internal state and awaits activity
setup before setting its private `_started` flag. If that await is cancelled or
raises, `shutdown()` schedules a close coroutine that returns immediately and
emits no close event. Speaker can still fence admission, detach both public
media heads, and close its manual RoomIO; it cannot prove disposal of any other
partially-created private AgentSession state. Outward wrapper close in this
rejected-start state attests only manual media-route disposal. No speech is
admitted before successful start and ready proof.

The console API has no public synchronization primitive. Retaining and
checking the exact singleton on both sides of session start closes observable
changes; malicious same-process mutation in the remaining instruction-level
window is a residual risk outside the trusted composition boundary. Any
mutation observed after start fails closed before speech admission.

Pinned RoomIO closes its AgentSession automatically for only three participant
disconnect reasons. Other reasons, including duplicate identity, reset its
participant future and permit a later same-identity binding. Speaker therefore
owns an exact-identity disconnect fence after the first media mutation, before
the SDK can commit a buffered turn or rebind that logical identity.

Publisher identity remains transport scoping, never owner verification or
action authority. Model-name strings and SDK defaults may select hosted
inference, so this wrapper rejects them. Headless fake-SDK tests cannot prove
package-install compatibility, network behavior, local model quality, AEC,
latency, device playback, or remote human audibility.

## Consequences

The optional module is import-safe in the base environment and the remote
dependency file pins the audited RTC/Agents pair. Deterministic tests cover
version/surface rejection, console races, partial callback binding, exact
manual composition and call order, start/ready/close races, retryable teardown,
pre-ready inbound callback suppression, publisher/source/track changes, route
mutations, all-reason publisher disconnect and non-reopening reconnect,
duplicate turn IDs, callback-before-output-admission, interruption, playout
failure, and sticky close.

No dependency was installed and no network, model, microphone, speaker, or
live hardware validation ran. `trusted-lan`, runtime defaults, action authority,
and the legacy rollback path remain unchanged. The audited component ID is a
configuration fact, not cryptographic proof; production assembly and live A/B
remain required before a separate promotion decision.
