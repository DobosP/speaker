# ADR-0058: Bind autonomous playback evidence to its input

Date: 2026-07-12
Status: accepted

## Decision

Strengthen ADR-0055 by matching every selected non-barge prompt with a label to
a newly logged final transcript at WER at most 0.50 and carrying that final's
input generation through tracked playback onset and terminal markers. Grade
only remembered assistant groups, never auxiliary acknowledgements. Require an
aggregate `completed` terminal for ordinary/self-interrupt replies. For
talk-over, require the same task/generation's aggregate `interrupted` terminal
after the sequenced barge marker, successful injection playback, and a cut while
the injector is active or within its explicit 150 ms acoustic tail. Commands
require their matching final but are excluded from playback grading. Count only
finite nonnegative finalized-bundle latencies.

## Context / why

Global playback counters and a quiet `speaking:` interval were insufficient.
An unrelated ambient turn or auxiliary acknowledgement could satisfy a prompt's
temporal window, and a long multi-sentence reply could be mistaken for idle
after 2.5 seconds without a new log line. Presence-only first-audio values and
failed/dropped terminal receipts also admitted invalid evidence, while an early
natural terminal or injector failure could leave a coincidental cut looking
causal. Merely tightening timeouts was rejected because it would not establish
turn identity or terminal outcome.

## Consequences

Autonomous delay/speaker verdicts now fail unless recognition, remembered sink
onset, and the required aggregate ledger outcome belong to the same input
generation, with onset and terminal also sharing a task. A talk-over terminal
must follow its concrete barge marker. The verdict policy and its unit seam
remain device-free; delay mode is silent, while speaker mode deliberately uses
physical hardware. Expected injected onset is still not physical human onset,
so bare-speaker live validation remains required.
