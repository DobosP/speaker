# ADR-0154: Preserve accepted continuation lineage when resuming

Date: 2026-08-06
Status: accepted

## Decision

Keep the raw accepted final as the initial `ResumeTracker` query fallback, then
replace only that ephemeral query text when an accepted continuation task
actually starts. Append an optional
`on_continuation_started(input_epoch, input_generation, composed_text)`
supervisor callback. Invoke it centrally only for a task carrying a nonempty
`continuation_of`, only after `TaskRuntime.start()` succeeds, and with the
task's exact current `input_text`. This covers immediate merges and later
queued or folded continuations without notifying for a queued, rejected,
cancelled, stale, or failed start. Contain callback exceptions so resume
bookkeeping cannot fail a task.

The runtime callback must acquire the terminal-effect lock and accept only an
exact current input epoch and generation across runtime arrival plus supervisor
arrival/commit state. It then calls a narrow, locked `ResumeTracker` query
replacement that does not reset playback, cut, or receipt evidence. The raw
final already opened and reset the tracker turn before the supervisor analyzed
the event; serialized bus delivery prevents continuation TTS admission before
the successful-start callback runs.

Keep synthetic resume publication unverified and response-only:
`owner_verified=false`, `origin=unknown`, `post_barge_response_only=true`, and
`skip_user_memory=true`. The callback emits no event, writes no memory, changes
no task metadata, and grants no routing, tool, confirmation, device, vault, or
owner authority.

## Context / why

The shipped continuation path correctly composed an original request such as
"explain the moon phases" with an accepted add-on such as "make it shorter".
If that composed reply started, was interrupted, and the user then said
"continue", the resume prompt still contained only the raw add-on. The runtime
recorded `final_text` before publishing the final, while the exact composed
prompt existed only later on the accepted assistant task.

Replacing the raw tracker query with `continuation_merged_text` during final
preprocessing was rejected. Materialization precedes supervisor intent
analysis, and a reserved final may resolve to research or another
non-assistant task whose synthetic continuation metadata is deliberately
discarded. Removing the raw fallback could then retain an older reply's resume
lineage. Reusing `on_continuation_admitted` was also rejected because that
callback clears reservation state for non-assistant resolution. Notifying at
queue or fold time would bind work that may never start and could miss the
latest folded text; notifying before worker start would bind a task whose
start may fail.

## Consequences

After an accepted composed continuation begins speaking, a later barge and
resume phrase now sends the same composed query back to the answering model,
not only the last raw add-on. Immediate, queued, and folded task starts share
one supervisor seam, while stale epoch/generation callbacks and non-assistant
reserved finals cannot replace newer resume state. The query-only update also
preserves any current playback receipt or cut evidence instead of reopening
the tracker turn.

Deterministic tests cover successful-versus-failed start ordering, callback
fault containment, non-assistant metadata removal, stale epoch/generation
rejection, query-only tracker behavior, and the full composed reply → audible
start → barge → `continue` sequence. No model, STT, TTS engine, microphone,
entry point, configuration default, durable memory, or authority policy
changes. Owner physical validation remains separate.
