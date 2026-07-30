# ADR-0084: Type the acoustic ingress lifecycle

Date: 2026-07-31
Status: accepted

## Decision

Represent recognizer observations in the control plane with an immutable
`AcousticSpan`: source, opaque stream/utterance identifiers, capture epoch and
generation, optional turn index, sample facts, acoustic clocks, endpoint reason,
and revision. Carry that lineage through acoustic signals, partials, finals,
mapped commands, aborts, runtime ingress events, turn merging, and asynchronous
Sherpa final work. Reject stale and duplicate typed revisions before runtime
side effects.

Make the recognizer source own exactly one terminal outcome once a typed turn
has emitted a partial: final, mapped command, or abort. Recovery, bounded-final
queue overflow, rejection, empty-final, and decode failure are explicit aborts
rather than silent disappearance. A mapped KWS command closes the source turn
and retires ordinary recognizer state in the same capture iteration, so its PCM
cannot later create a second ASR final. A typed barge-in carries a monotonic
revision and passes the same high-water gate before it can cancel work or stop
playback.

Bind typed post-barge response permission to the triggering acoustic key. An
unrelated or untyped final cannot inherit a typed grant, and a matching aborted
replacement partial restores only the still-valid final it displaced. Keep
legacy untyped callbacks compatible at ingress, including their untyped grant
behavior, but do not infer typed correlation from them. Serialize
mapped-command publication with other terminal effects so racing callbacks
cannot publish the same acoustic turn twice.

Use deterministic typed lineage in FileReplay and retain immutable closed
lineage across Sherpa's bounded final queue. On shutdown, retire source
ownership locally even when callback fencing correctly prevents publication
into a tearing-down runtime.

This is an ingress prerequisite, not the single session actor. The recognizer
loop remains synchronous, the central event bus remains unbounded, and task,
tool, TTS, and playback ownership still use legacy generations. Implement the
bounded single-owner session boundary as a separate behavior change. This
refines ADR-0023, ADR-0039, and ADR-0046 without superseding them.

## Context / why

Text plus runtime generation was not a sufficient identity for concurrent
speech. A late final could consume a newer barge grant, a failed replacement
partial could permanently erase an earlier valid final, and asynchronous
Sherpa failures could lose or double a terminal callback after mutable engine
state had moved on. These are ordering bugs even when transcript text happens
to match, so string comparison or larger timeout windows cannot repair them.

Command and barge paths had an additional source-ownership defect. A command
could be published while leaving the acoustic tracker active, letting the same
PCM later produce a normal final. Barge-in had no revision, so a duplicate or
late callback could cancel a newer turn before any stale-event check ran.

## Consequences

- Runtime and replay tests can assert monotonic revisions, acoustic
  correlation, matching rollback, and exactly-one-terminal behavior without
  opening an audio device.
- `stop()` still fences external callbacks first; source ownership is retired
  locally rather than publishing shutdown events into a tearing-down runtime.
- Typed post-barge grants fail closed across utterances. Legacy untyped engine
  integrations retain their legacy behavior until migrated.
- Typed lineage currently ends at runtime ingress; it does not yet make tools
  or playback revision-aware and does not add PCM backpressure.
- Headless replay does not validate the live microphone, speaker echo route,
  natural turn timing, multiple voices, enrollment, or owner authority.
- The next structural change is a bounded session actor that owns revisions,
  cancellation, tool calls, TTS, and playback while treating the endpoint as a
  thin producer.
