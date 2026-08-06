# ADR-0155: Add inline selected-terminal lineage evidence

Date: 2026-08-06
Status: accepted

## Decision

Extend the opt-in logical-turn composition observer with a separately reported,
transcript-free terminal-lineage mode. A successful raw-composition comparison
mints one bounded opaque token. Immediately after `AcousticTurnTracker.close()`,
bind that token in memory to the exact ordered `AcousticSpan.key` tuple and
typed revision, before speech-evidence admission, final selection, or any typed
abort. At the replay collector, resolve a selected `FinalTranscript` or
`TranscriptAbort` against that identity, reduce the result and closed abort
reason to counters, and erase all retained acoustic identities on close.

Expose the mode only through `--logical-turn-terminal-lineage`, require
`--logical-turn-shadow`, and keep existing unflagged and raw-only report shapes
and evaluator behavior unchanged. Capture replay remains schema 1 by default
and schema 2 for raw shadow evidence; the new combined mode is schema 3. Its
worker must validate exact section presence before publication, expand the
success receipt with schema/raw/terminal scalars, and require the guarded parent
to match that tuple to its requested flags. EdAcc remains schema/kind v2 by
default and v3 for raw shadow evidence; the new combined mode is strict
schema/kind v4. Its guarded parent must reject disagreement between both
requested flags and the retained schema. Preserve the existing schema-2 and
schema-3 source-file sets, then add `always_on_agent/acoustic.py` and
`core/engines/_acoustic_turn.py` only to the v4 closure. Its 19-file binding
therefore covers stable acoustic-key and revision minting as well as the raw
observer and capture handoff.

Require every v4 cell to reconcile exact per-terminal lineage, not FIFO or
aggregate coincidence:

```
raw paired commits
  == claimed commits
  == acoustically bound commits
  == matched selected finals + matched typed aborts
  == selected finals + outer typed-abort total
```

Require closed per-reason abort marginals to match the outer typed callback
report, require at least one matched `input_rejected` abort across the exact
three-cell run, and reject missing, foreign, duplicate, conflicting, late,
unbound, sparse, or capability-forged evidence. A selected final may contain
different model-selected words; this mode proves terminal identity and typed
outcome, not selected-text equality.

Limit this evidence to deterministic inline finalization. Keep async final
selection, final dispatcher, supervisor, runtime stop/shutdown, production
configuration, authority, and live capabilities false. The replay harness
forces `asr_final_async=false`; queued worker delivery needs separate lifecycle
ownership that drains the worker before closing observers or replacing row
callbacks.

## Context / why

ADR-0153 proved raw native-epoch composition on the retained EdAcc replay, but
its controls recorded 35 raw commits against 31 selected finals and four typed
input-rejection aborts, while reset-and-accumulate recorded 32 against 29 and
three. Equal cell totals did not prove which raw commit produced which terminal
callback. Reusing `LogicalTurnCompositionShadow.abort()` could not close this
gap: a raw commit has already cleared its coordinator, so the later typed abort
is correctly counted there only as an idle observation.

Matching by callback order was rejected because a later capture-side rejection
can overtake an earlier queued final. Comparing whole acoustic objects was also
rejected because callback delivery adds an emission timestamp. Persisting keys,
revisions, per-row records, transcripts, text hashes, or lengths was rejected
as unnecessary private evidence. Enabling the existing async worker in replay
was rejected because replay exhaustion clears its run condition and closes the
row observer before queued work is guaranteed to drain.

## Consequences

The inline capture seam can now distinguish a real one-to-one selected-final or
typed-abort handoff from balanced aggregate forgery. Stable acoustic identity
and revision survive final-model rewrites and callback timestamp decoration,
while reports contain only closed counters, reason marginals, capability
limits, sufficiency, exactness, and health.

The observer remains diagnostic and fail-contained. Its bounded maps retain at
most 256 pending/completed commit identities and 64 spans per lineage; overflow
invalidates evidence without changing capture behavior. Default construction,
production entry points, models, thresholds, tools, speaker policy, task
authority, and live behavior are unchanged. The capture success-receipt shape
adds only three transcript/path/identity-free scalars; strict consumers must
accept that expanded contract. The clean `dac50c9` exact EdAcc v4 replay bound
all 35/35/32 raw commits across its three cells to 31/31/29 selected finals plus
4/4/3 typed input-rejection aborts, with exact lineage and zero health errors.
Its aggregate-only report is mode 0600, single-link, 23,350 bytes, and SHA-256
`1c03d64295fe9443d057c854b034fbc73723420641088f918f242a79cb1f5230`.
A later branch must build an explicitly drained deterministic async-worker
harness before making any async terminal claim.
