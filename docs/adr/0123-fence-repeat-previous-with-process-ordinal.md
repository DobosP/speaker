# ADR-0123: Fence repeat-previous with a process-local ordinal

Date: 2026-08-04
Status: accepted

Refines: ADR-0051

## Decision

Mint a stable creation ordinal plus unpersisted token and PID witnesses only
when one of the three built-in memory backends appends a `MemoryItem` to its
current-process `all()` window. Keep all stamps outside the dataclass fields,
so ordinary construction, representation, equality, `asdict()`, and built-in
durable storage cannot mint process-ordinal authority. When the assistant
capability is attached, reserve an ordinal fence. The bounded repeat-previous
controller may reuse an assistant output only when its ordinal is a strict
integer newer than that fence and both witnesses belong to this loaded process.
For foreign or plainly reconstructed items without an ordinal, preserve
compatibility only through a finite wall-clock timestamp strictly newer than
session start. A present but malformed or unauthenticated ordinal fails closed
and never falls back to wall time.

Keep this ordering metadata process-local. Built-in `all()` implementations
must return their stable current-process objects; reconstructed durable search
results receive no ordinal authority merely because they can be recalled. A
pickled/restored stamped item cannot authenticate its copied ordinal with the
process token, and a forked child rejects the parent PID witness. Both therefore
fail closed rather than downgrade to timestamp comparison. Do not persist or
compare the ordinal as user memory and do not change the repeat grammar, model
fallback, entry points, tools, or runtime defaults.

## Context / why

The existing repeat controller accepted `timestamp >= session_started_at`.
On Windows, `time.time()` can return the same coarse value for an answer stored
before capability attachment and for the session-start fence. That made a
stale answer look current. Changing the comparison to `>` would reject the
stale tie but would also reject a valid answer created later in the same clock
tick, so wall time alone cannot express the required ordering.

A monotonic-clock stamp was rejected because independently sampled monotonic
values can still be unavailable on foreign records and do not identify an
item's position relative to capability attachment as directly as one shared
sequence. Persisting an ordinal was rejected because its only authority is
within one process, while durable or reconstructed items belong behind the
cross-session trust boundary. The strict finite timestamp fallback retains a
conservative compatibility seam for non-Speaker memory implementations without
letting malformed local metadata downgrade to the weaker check.

## Consequences

Pre-session and post-session answers are now distinguished deterministically
even when wall time is frozen. Repeat-previous still falls through to the model
when no eligible answer exists. The three built-in memory backends preserve
stable, unique, authenticated ordinals in their current-process `all()`
windows; their search and persistence behavior is unchanged. Concurrent
append order need not equal ordinal allocation order and is not part of the
freshness contract.

Headless verification covers both frozen-clock sides of the fence, equal and
strictly newer foreign timestamps, non-finite and raising timestamps, malformed
or unauthenticated ordinals, reconstructed and restored items, a changed PID
witness, and all three built-in memory backends. This is controller and memory
ordering evidence only: it opens no audio device, runs no model, and does not
validate STT, TTS, barge-in, tools, or a physical Windows session.

Low-priority Linux landing evidence is 105 focused conversation/memory/SQLite
checks, 304 adjacent checks with three skips, 7,821 broad non-real checks with
14 skips and 24 model-only deselections, the timestamp-sensitive bounded-read
race isolated at 1/1, the APM/DTD gate at 6/6, and a bounded child-process probe
that rejected its inherited parent item. The exact private wheel-lock fixture
was mode 600 for its broad validation.
