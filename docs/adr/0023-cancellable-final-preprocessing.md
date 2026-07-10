# ADR-0023: Cancel and bound final preprocessing before task admission

Date: 2026-07-10
Status: accepted

## Decision

Run every potentially blocking final-preprocessing step—addressing, transcript
cleanup, capability routing, and custom tier/latency routing—off the engine
callback through `FinalDispatcher`. Give each dispatch a cancellation lease and
terminal commit point. A newer final, barge-in, stop, or shutdown retires an
uncommitted lease; run synchronous providers behind a six-slot bulkhead so an
unknown cancellation-ignoring call retains its slot until it returns but cannot
create unbounded work. Collect LLM gate output through `LLMClient.stream()` so
the native Ollama cancellation seam from ADR-0022 is available before token one.

Keep committed terminal effects ordered, reject callbacks after shutdown, and
preserve continuation prefixes only inside the configured total hold window.
On arrival of a new final, fence/cancel older active task output before the new
final enters its gates. The shipped deterministic continuation classifier is the
only audio-thread-safe semantic check: an add-on before answer audio atomically
reserves its lineage, advances the speech epoch, and silences the unheard victim;
an add-on after answer audio reserves the same lineage without canceling the
speaking parent. Materialize either reservation only after the winning lease
commits, so blocked gates cannot erase context, chained add-ons stay one turn,
and canceled leases cannot write memory or merge metrics. Preserve pending
confirmations.

Treat "before first audio" as the engine playback-admission boundary, not TTS
event publication. Keep a completed task with queued-but-unheard audio available
as a cancellation/continuation victim; commit its assistant output to memory only
after playback admission. Carry dispatcher coalescing provenance so fragmented
ASR finals replace, rather than duplicate, the reserved add-on. Intersect owner
verification and origin across the complete synthetic lineage, and bind merge
metrics to the victim's turn token.
Separate input invalidation from the speech epoch: an input epoch invalidates
dequeued-but-not-admitted finals on external control, while a monotonic committed
final generation prevents an older bus event from starting after a newer final.
Capture the input epoch when a final is submitted, not when its provider starts,
and revalidate both identities at task/queue/confirmation admission. Keep a
committed-but-unregistered assistant final as unheard lineage until the bus
resolves it, closing the publish-to-task continuation gap. The first substantive
partial silences every unheard task mode; it reserves conversational lineage only
for one unambiguous assistant victim, and recognized own-TTS echo partials never
enter that fence.

Serialize analyzer control validation with its state effect. `VoiceRuntime` owns
STOP's dispatcher retirement, supervisor cancellation, and physical playback cut
inside one terminal section; mapped KWS controls commit their generation before
older bus work can register. Route local-intent and watch speech through the same
cancellable auxiliary-TTS admission instead of calling the engine directly.

Give every auxiliary emission a unique generation/epoch identity, retire queued
sentences of failed/cancelled tasks by task ID, and keep completed streaming tasks
owned until an explicit stream-end marker drains after their queued sentences.
Only playback-admitted streaming fragments enter assistant memory. An irreversible
ambient-memory write must win lease and input terminal ownership before starting;
provider/memory I/O remains outside the acoustic lock after that commit. Preserve
the recognizer callback timestamp when `ASR_FINAL` commits so final-to-token
latency still includes preprocessing.

## Context / why

ADR-0021 made `AgentTask` execution cancellable and ADR-0022 made production
Ollama streams cooperatively cancellable, but both started too late. Before an
`AgentTask` existed, the single final-dispatch worker could spend up to the
provider timeout in addressing, cleanup, or routing. A new utterance and a real
barge-in could cancel tasks and playback yet remain stuck behind that old gate.
Late results could then publish stale tasks, and a final already on the event bus
could be admitted after stop because task cancellation had nothing registered to
find.

A thread hard-kill was rejected: Python cannot safely terminate arbitrary
Python/native provider work, and doing so can corrupt shared model or memory
state. Unbounded replacement threads were also rejected because repeated speech
would turn a hung provider into resource exhaustion. Reusing the task speech
epoch for input admission was rejected because continuation merges legitimately
advance the speech epoch and would discard later ordered add-ons or confirmations.

## Consequences

- Engine callbacks do only validation, generation bookkeeping, and a bounded
  queue handoff; gate/router provider waits no longer occupy the audio thread.
  A slow new gate cannot leave the previous answer audible while it waits.
- A pre-audio continuation silences the old answer at arrival and later starts
  one merged task. An after-audio continuation lets the parent finish and later
  starts or queues one context-carrying task, even if preprocessing outlives the
  parent. Chained/coalesced fragments keep one lineage; only real origin/add-on
  utterances enter user memory, and untrusted provenance cannot be upgraded.
- A substantive final generation fences an older lease even after that lease
  claimed terminal ownership but before it registered a task. Punctuation and
  recognized self-echo do not allocate a generation or retire valid work.
- Newest-input, command-stop, barge-in, and shutdown have deterministic headless
  coverage before token one, after terminal claim, across bus backlog, through
  continuation/confirmation, partial onset, direct KWS, streamed/auxiliary TTS,
  follow-up scheduling, and the final check-to-playback race.
- A real no-mic MiniCPM/Ollama addressing cancellation retired in 157.9 ms with
  zero old pieces; the healthy follow-up returned `ACT` and spoke only the new
  query. This is not human-speech, microphone, or acoustic barge validation.
- Cancellation-aware Ollama calls release their provider slot after native async
  cleanup. Arbitrary synchronous/native calls can still survive as daemon work
  and consume one of six preprocessing slots until their own timeout/return.
- Ambient ingestion commits terminal ownership before its irreversible memory
  write. A slow committed backend can therefore delay the next dispatcher turn,
  but a canceled lease can never persist stale ambient memory.
- Cancellable shutdown drops uncommitted held/final work instead of starting a
  new LLM request, rejects late task/aux/follow-up publication, and leaves no
  active task references. Legacy non-cancellable `FinalDispatcher` users retain
  their flush-on-stop behavior; sequential restart remains supported while a
  concurrent restart is rejected until stop returns.
