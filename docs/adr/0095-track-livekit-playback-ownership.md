# ADR-0095: Track LiveKit playback ownership

Date: 2026-07-31
Status: accepted

## Decision

Make LiveKit output an opt-in tracked playback sink. Admit every legacy or
tracked fragment into one ordered, generation-fenced queue; a stop atomically
fences the active fragment and every older queued fragment, while a later
admission belongs to the next generation. Emit at most one exact start callback
after the first successful LiveKit `capture_frame` and exactly one terminal
receipt for every tracked admission. A completed receipt attests the full text;
an interrupted or failed fragment attests no text, and a fragment fenced before
its first frame is dropped. Retain ownership while native TTS is still
unwinding, and require `wait_for_playout` plus a current generation before
completion. Do not advertise sample counts: LiveKit does not expose how much of
a cleared queue was rendered rather than discarded. Wait for a connected,
published output track before synthesis, clear the source before any started
non-completion receipt, and bound pre-generation to the active sentence plus
one prepared successor so exact receipts do not add a synthesis-sized gap.

## Context / why

Remote input already enters the shared Python runtime and its actor-issued
`TurnHandle`, but `LiveKitEngine` advertised no playback receipts. The runtime
therefore committed memory and released playback ownership at handoff instead
of at sink terminal. LiveKit also used one shared stop event that each queued
speech coroutine cleared after acquiring its lock; a pre-stop queued sentence
could consequently play after STOP. Replacing the existing actor/control plane
with a third-party framework would discard stronger task and side-effect
guarantees without closing this output seam.

## Consequences

Remote playback now participates in the same cancellation, heard-history, and
idle/drain accounting as desktop playback, and stale queued speech cannot
survive a stop. STOP queues or performs the LiveKit source clear before
consumer callbacks, and transport loss fences the same generation. A
non-cancellable native synthesis may keep drain truthfully pending until it
returns; if that native call wedges, bounded `stop()` reports degraded teardown
and retains the loop/ownership instead of fabricating a terminal receipt. A
failed source clear is transport-fatal and disposes the source before receipts.
Source-queue drain proves only local SDK playout, not audible rendering by a
remote device, so client acknowledgements and live WebRTC validation remain
follow-up work. This decision does not yet scope remote input per participant,
isolate native TTS in a killable process, add typed input-gap events, converge
Dart, or change STT/AEC.
