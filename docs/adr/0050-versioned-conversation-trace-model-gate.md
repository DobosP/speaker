# ADR-0050: Gate model adoption with versioned conversation traces

Date: 2026-07-12
Status: superseded-by ADR-0051

## Decision
Require a versioned, device-free conversation-trace gate before adopting or
expanding an answering model. Run fixed synthetic scenarios through the
production `VoiceRuntime` assembly with `ScriptedEngine`, record ordered turn,
task, capability, tool, playback, cancellation, memory, model-role, and latency
facts, and grade deterministic concepts plus structural invariants. Compare a
candidate with a baseline sequentially for three fresh-runtime repetitions;
report both first-run `pass@1` and all-runs `pass^3`, and fail the gate unless
both model suites and every candidate repetition pass. The default real-model
topology keeps the configured main model and replaces only the fast role;
forcing one model into both roles is a separately labelled `all-roles` stress
mode. Candidate and baseline effective role maps must differ. Prewarm every
distinct evaluated role model with the actual runtime system prompt before
production-warm measurements. Observe nested tool
arguments/results only through the opt-in `CapabilityRegistry` observer;
normal runtime logs remain unchanged. For `minicpm5-1b:q8`, additionally
require the local alias to resolve to the pinned official Q8 source blob and
the committed template/parameter contract. A cold-policy run is diagnostic and
cannot make the adoption gate green.

Gate coverage is exact: all twelve scenarios at run indices one through three.
Scenario selection or a different repetition count remains useful diagnostics,
but cannot return a green adoption verdict.

## Context / why
The existing real-model checks establish that a model loads, emits text, can be
cancelled, and can complete narrow native-tool probes. The latency benchmark
counts whether a turn answered but does not enforce answer correctness. Neither
surface exercises the configured voice system prompt, addressing gate, cleanup,
capability router, recent context, task lifecycle, tracked playback, or
post-barge response policy as one conversation. Thousands of deterministic
runtime tests still left model replacement vulnerable to semantic regressions,
tool-trajectory drift, prompt-injection obedience, and one-run flukes.

The alternative of grading only free-form answer bytes is too brittle across
valid phrasings. An advisory LLM judge is also unsuitable as the acceptance
authority because it is nondeterministic and can hide control-plane failures.
The gate therefore uses concept/forbidden checks for prose and exact invariants
for lifecycle, tool, cancellation, playback, and memory structure.

## Consequences
The committed v1 suite has twelve scenarios: ambient INGEST, transcript
self-correction, direct QA, concise instruction following, recent context,
local search, bounded research, failed-tool recovery, mid-tool barge, pure
fast-tier barge stop, fast-tier post-barge redirect, and an untrusted tool
result. The barge fixtures require and attest a withheld continuation after the
first sentence, rather than cancelling an already exhausted response. It fails if
first tracked sink start exceeds 2.5 seconds, any model TTFT or complete call
exceeds 2.5 seconds, playback does not stop within 500 ms of a scripted barge,
or the cancelled task/invocation does not close within one second. Playback
fragments are attributed to their exact TTS request, task, and input generation
at the sink boundary, and turn generations are anchored to the evaluator input,
so a stale cancelled-task fragment cannot hide inside redirect text.

Reports use schema/scenario-set versions and fixed synthetic text, live under
ignored `logs/conversation-eval/`, and include timing-independent structural
trace signatures plus git/config/client/topology/warmup metadata for A/B diffs.
The registry observer is inert when unsubscribed. It records the synthetic
capability query/result for this gate, but never exposes the provider context
dictionary, raw LLM prompts/history, or writes normal logs. An identity override
may continue a diagnostic run and write its report, but provenance remains red
and the process exits 2.

This gate proves conversation semantics without opening ASR, microphone,
speaker, echo-cancellation, or TTS hardware. It does not replace the Sherpa
duplex worker test, recorded-audio replay, or required live bare-speaker A/B.
MiniCPM identity verification also requires the official source tag to remain
locally available; an explicit unverified override is diagnostic only and does
not make provenance green.
