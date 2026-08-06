# ADR-0156: Add post-capture async terminal-delivery evidence

Date: 2026-08-07
Status: accepted

## Decision

Add a separately opt-in, transcript-free diagnostic that executes endpointed
final work through the real `SherpaOnnxEngine._final_worker` after each replay
producer has naturally quiesced. Expose it through
`--logical-turn-async-terminal-delivery`, require both
`--logical-turn-shadow` and `--logical-turn-terminal-lineage`, and keep every
production caller, configuration, authority rule, threshold, model, and entry
point unchanged.

Give `RealtimeMediaStage` a route-neutral `wait_idle()` observation barrier.
It succeeds only when queued and in-flight ownership are both empty, and
`finish()` notifies it after consumer work and its synchronous callback have
returned. Preserve close as cancel-and-retire; the production stop path must
not drain through this diagnostic. Let the capture loop defer logical-turn
observer closure only through a strict default-true private argument. Default
capture and the native decode-owner path retain their existing closure.

For each flagged replay row, install a fresh stage and keep its callbacks and
observers open while capture runs synchronously without a final worker. After
natural source exhaustion, submit that stage to one persistent evaluator
thread per built engine. That thread invokes the real final worker, correlates
each exact ordered `AcousticSpan.key` tuple plus revision to the selected-final
or worker-generated typed-abort callback, and calls `finish()` afterward. The
producer then observes an empty stage, closes it only to wake the worker, waits
for worker completion, and closes the observers before callbacks can be
replaced. Failure, timeout, overflow, cancellation, nonempty close, callback
error, wrong thread, missing/duplicate/conflicting terminal, or unsafe teardown
publishes no report. Acoustic keys, stage sequences, thread identifiers,
transcripts, text hashes/lengths, paths, and exception strings are erased and
never serialized.

Require the following exact algebra per row and in aggregate:

```
raw commits = capture-side prequeue typed aborts + stage accepted
stage accepted = stage finished
               = worker selected finals + worker typed aborts
raw commits = worker selected finals
            + worker typed aborts
            + capture-side prequeue typed aborts
```

Require offered/accepted equality; empty pre-drain in-flight ownership; empty
post-drain queue and in-flight ownership; zero closed rejection, overflow,
scope retirement/cancellation, close retirement/cancellation, foreign/late
callbacks, lifecycle errors, and observer errors; and positive worker-selected
coverage. Match every closed abort marginal to the existing terminal-lineage
and outer callback sections. Classify current EdAcc `input_rejected` outcomes
as capture-side prequeue aborts: they did not traverse the worker. The new
subreport proves accepted-handoff to dedicated-worker callback lineage only;
raw-to-terminal correlation is established by the required cross-section
validator, not by the standalone subreport.

Preserve capture replay schemas 1–3 and add schema 4 only for this combined
mode. Preserve EdAcc schemas/kinds v2–v4 and add strict v5. Add
`core/realtime_media_stage.py` and the private async-delivery helper only to the
new conditional source closures (32 files for capture schema 4 and 21 for
EdAcc v5). Bind the third mode boolean through child receipts, guarded-parent
mode checks, retained-report reopening, and pre/post source checks.

Keep these capabilities explicitly false: simultaneous capture/finalizer
concurrency, async-versus-inline selected-text parity, overflow behavior,
production stop/shutdown/cancellation parity, final dispatcher, supervisor,
runtime/task delivery, defaults, authority, device/microphone, latency, live
conversation, and quality improvement. ADR-0155's
`async_final_selection_parity=false` remains correct.

## Context / why

ADR-0155 proved exact raw-commit identity through inline selected finals and
typed aborts, but replay deliberately forced async finalization off. Merely
enabling it was racy: replay exhaustion clears the production run event, both
capture-loop layers closed the observer, and row teardown replaced callbacks
before queued final work was guaranteed to finish. Calling `engine.stop()` was
also invalid evidence because it increments the capture epoch, cancels the
stage, retires queued work, and intentionally fences callbacks.

Unit tests for the final worker already proved FIFO/off-thread behavior with
fakes, but could not prove the model-backed capture → stage → real final worker
→ typed callback route on the retained corpus. A deterministic post-capture
drain supplies that missing ownership evidence without pretending to measure
the production concurrency or shutdown lifecycle.

## Consequences

The diagnostic can now fail closed on aggregate-balanced and identity-balanced
forgeries while retaining bounded memory (at most 256 terminal identities and
64 spans per lineage). A persistent evaluator thread avoids moving one native
final recognizer across successive row threads. The generic stage barrier adds
no admission fence: callers must independently prove producer quiescence.

Before the exact model replay, the branch passed 295 focused stage/engine/
capture/EdAcc tests with one skip, 522 adjacent Sherpa/ownership/APM tests, and
318 EdAcc/logical-turn contract tests. Scoped Ruff and whitespace checks were
green.

The exact schema-v5 replay at clean source `ab4f4eb` completed all 24 rows in
all three cells. Each cell observed 24 clean drains and 24 worker sessions on
the persistent evaluator thread distinct from capture. Acoustic and HOLD-only
each accepted and finished 31 items as 31 worker selected finals; reset and
accumulate accepted and finished 29 as 29 worker selected finals. No accepted
item produced a worker abort, no selected final ran inline, and the separately
matched capture-side prequeue typed aborts remained 4/4/3. All async health,
stage-release, cancellation, overflow, and unfinished counters were zero. The
existing raw and inline lineage sections remained exact at 35/35/32 raw
commits, 31/31/29 selected finals, and 4/4/3 typed aborts; reset retained five
multi-epoch commits. The guarded parent reopened schema 5 with all three flags
enabled and a 21-file source closure at the same revision.

Preserve the mode-0600, single-link, 29,952-byte aggregate report with SHA-256
`b3c749fcbdc867c179fbac77717552147de65240c1e815e6fcaef6d12b9a0852`.
Peak process RSS was 1,269.109 MiB. The result is diagnostic-only and makes no
claim about live capture/finalizer concurrency, selected text, stop/shutdown,
dispatcher, supervisor, runtime/defaults, device, latency, authority, or STT
quality.

After the replay, the focused gate remained 295 passed with one skip, the full
EdAcc/logical-turn contract remained 318 passed, and APM/double-talk remained
6 passed. Independent implementation and evidence/privacy reviews reran 88
passed with one skip and 228 passed respectively; both returned GO with no
blocking findings.
