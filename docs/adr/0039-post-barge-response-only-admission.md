# ADR-0039: Admit one post-barge final for response only

Date: 2026-07-11
Status: accepted

## Decision

After the engine confirms an acoustic barge-in, arm one monotonic eight-second
grant bound to the supervisor's new input epoch. Inspect the grant only after
`FinalDispatcher` has produced its terminal held/merged final, and consume it
only when that lease wins terminal ownership. A newer barge, input epoch, or
token can never be consumed by an older preprocessing lease.

Only a typed `live_audio` final may use the grant, and only when the addressing
gate would otherwise return `INGEST` (or conservative `UNSURE`). Admit that
final solely as a conversational `assistant.answer`: force `owner_verified=false`
and `origin=unknown`, bypass runtime local intents and capability/latency
routing, force the supervisor decision to `ASSISTANT`, select the direct fast
answer tier, and disable ReAct/tool escalation. Do not store the final as user
or procedural memory; it may read bounded prior conversation to resolve an
override such as “instead.” That bounded recent working conversation is the only
extra context allowed: do not read episodic recall, profile facts, last-session
summaries, or trusted procedural rules, and do not request or attach ambient or
explicit visual frames. An `ACT` verdict proceeds normally and consumes the
one-shot observation without inheriting any extra trust.

Exclude response-only finals and tasks from published-unheard, partial, arrival,
queued, and in-flight continuation lineage. A later add-on is a fresh turn; it
may never fold the untrusted command-shaped response-only text into a synthetic
prompt whose metadata has lost the restriction.

Every synthetic `ResumeTracker` prompt uses the same response-only envelope,
not only one created while a grant is armed: the tracker retains query, spoken
tail, and cut state, but no speaker, origin, action, or sensitivity provenance.
Keep that ephemeral lineage so a reply can be cut and resumed repeatedly. An
observed foreign/replaced grant or a non-`ASSISTANT` mode drops the resume and
clears retry lineage. Mode/control boundaries invalidate an armed grant, and
the supervisor rechecks mode before accepting an already-queued response final.

Expire a grant with no final, including exactly at its deadline. Invalid,
negative, zero, NaN, or infinite configured windows disable the grant.
Empty/punctuation-only, explicitly speaker-rejected, and recognized self-echo
finals invalidate it; a foreign-origin first final consumes it without a bypass.
If a token is invalidated or replaced while its addressing lease runs, resolve
and drop the stale lease without answering or writing ambient memory.

## Context / why

Live run `20260711-144211` proved the new word-cut path could stop playback, but
lost the actual conversational override. The cut fired at `14:43:49.730`; ASR
then held and merged two fragments until `14:43:55.527` as “Segwam though Carry
me about through roman arcticture instead.” The small-model addressing gate
classified that garbled but contextually obvious override as `INGEST`, so the
assistant stayed silent after successfully barging.

Blindly turning the next final into `ACT` was rejected: a self-echo, bystander,
late room utterance, or command-shaped transcript could gain control/action
authority from timing alone. Consuming at first ASR callback was also rejected
because the turn merger can retire that callback and emit a later combined
final. Retaining superseded garble as ambient memory was rejected because the
interruption already established that this is a response attempt, not durable
room context.

## Consequences

- The exact six-second held/merged shape from run `144211` receives one answer
  even when addressing says `INGEST`; a second final follows normal policy.
- Acoustic barge provenance never becomes speaker identity. Command, control,
  confirmation, local-intent, capability-router, ReAct, tool, and persistent
  user-memory paths remain structurally unreachable for the exception.
- The eight-second window must remain long enough for the configured six-second
  merge bound plus endpoint/dispatch slack. Raising it increases exposure to a
  later room final; lowering it can lose a long spoken override.
- Resuming an interrupted research/tool/visual turn becomes a safe fast-tier
  conversational continuation; the tracker cannot attest authority to restart
  tools or expose durable private context. Repeated cut/resume remains supported.
- Deterministic tests cover the recorded merged transcript, command-shaped
  input, action-classified consumption, foreign/expired/empty/self-echo input,
  token replacement, capability-context trust, and invalidation during a
  blocked addressing lease. Private-context spies, an in-flight add-on, mode
  races, foreign resume, and repeated escalation-shaped resumes prove durable
  context and lineage cannot launder the exception. Live validation is required.
