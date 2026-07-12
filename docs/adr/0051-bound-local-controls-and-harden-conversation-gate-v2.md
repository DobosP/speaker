# ADR-0051: Bound local controls and harden conversation gate v2

Date: 2026-07-12
Status: accepted

Supersedes: ADR-0050

## Decision

Retain ADR-0050's device-free, three-repetition production-hybrid A/B design,
but make scenario-set v2 exactly fourteen scenarios and strengthen its
provenance contract. Require identical clean repository and complete effective-
config evidence before and after evaluation. Verify every distinct effective
Ollama role before and after its suite, recording full blob and effective-alias
digests; MiniCPM additionally retains its pinned official Q8 blob, template, and
parameter checks. Require candidate and baseline fast roles to resolve to
different blobs, bind the shared main role to configured main, and keep
`all-roles` diagnostic-red.

Move only narrow, anchored control contracts out of generative-model discretion:
high-confidence ACT/search/research forms, exact one-word replies, exact
repetition of the latest committed current-session assistant answer, a bounded
sixteen-entry session-fact grammar, and the exact-word form inside the existing
post-barge response-only envelope. Let clean transcripts bypass model cleanup
and retain only bounded signalled-correction repair. Configure shipped recent
context as role-structured history under one 320-token cap, dropping oldest
whole turns, while preserving the legacy programmatic fallback for minimal
embedders. Within one ReAct plan, never reinvoke a failed provider and allow the
first failed `web.search` at most one cancellation-fenced `search.local` attempt
with the identical query. Stamp every admitted spoken reply with
`TTS_REQUESTED`, so controller-owned replies bypass the missing-model warning
but remain subject to first-audio stall detection.

## Context / why

The v1 smoke and repeated traces exposed variance in contracts that should not
depend on model wording: addressing labels, exact response length, correction
retention, failed-tool retry, session-fact recall, post-barge redirect text, and
stream-boundary handling. Prompt tuning alone was rejected because it leaves
admission, memory, recovery, and cancellation behavior sampling-dependent.
Replacing general answering or ambiguous room-speech judgment with regexes was
also rejected; those remain semantic model tasks.

Role history improved real follow-ups, but the prior turn/character bounds did
not provide one global context allowance on the 1536-token profile. The shipped
320-token reserve leaves at least 300 estimated tokens for current input and
chat framing after the runtime system prompt, recall allowance, and output cap.
Controller-owned answers also revealed an observability gap: `HANDLED_LOCAL`
correctly suppressed a nonexistent model-token warning, but also hid a spoken
reply whose admitted TTS request never produced audio.

Dirty-tree reports, mutable-tag checks sampled only before a run, alias-name A/B
comparison, and weight-only baseline identity were rejected as adoption
evidence. They did not prove which revision, effective config, role mapping,
weights, template, or parameters produced the result. Redirect outcome sets and
history counts were likewise insufficient without exact fragment ownership and
privacy-safe per-message role/content hashes.

## Consequences

Scenario-set v2 contains fourteen scenarios and requires 42/42 scenario-runs per
suite: ambient ingestion, bounded correction, ordinary QA, concise output, typed
session facts, model role-history follow-up, exact-word/repeat control, local
search, research, failed-tool recovery, mid-tool barge, barge stop, post-barge
redirect, and untrusted tool output. Redirect grading joins request, attribution,
terminal receipt, task, and input generation; history grading reconstructs the
exact expected two-message role/hash sequence without recording raw history.

Exact controls fail through safely when their grammar or required current-
session state does not match. Session facts are capped and invalidated when
their backing memory disappears. Post-barge response-only input remains owner-
unverified and cannot gain tool, action, private-recall, continuation, or durable
user-memory authority. A failed web provider can add one local invocation but
cannot loop. Structured history drops oldest complete turns to remain within
320 estimated tokens.

Pre-commit evidence on Linux ROG: full logic suite 3416 passed/30 skipped/9
warnings; APM 6 passed; focused duplex/barge 16 passed; deterministic v2 42/42.
A full dirty-tree semantic rehearsal passed 42/42 for MiniCPM Q8 and 42/42 for
Gemma; its adoption verdict remained provenance-red by design. A clean-revision
full A/B is still required before this branch can be considered model-gate green.

The trace opens no ASR, microphone, speaker, echo-cancellation, or real TTS
device. Live bare-speaker barge-in remains red, so this branch is not landable.
