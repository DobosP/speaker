# ADR-0227: Bind conversational admission before input replacement

Date: 2026-10-09
Status: accepted

## Decision

Add a typed English conversation-candidate gate in `always_on_agent` before a live
assistant-mode partial/final can fence or replace existing work. The existing enabled
input gate opts into this seam; `input_gate.conversation_admission.enabled=false` is
an explicit rollback. Questions, requests, configured assistant-name prefixes and
existing controls/continuations remain available without requiring a wake word for
every turn. Typed input and other modes retain their current contracts.

Only a current, remembered, completed and actually rendered reply ending in a question
opens an eight-second answer window (configurable 0–30 seconds). A positive partial
cue can be carried to its exact acoustic utterance's final for at most 30 seconds;
carried fragments cannot renew it, override hard refusals or transfer it to another
utterance. Abort, capture recovery, STOP/mode change and shutdown invalidate relevant
state. Restoring an already-admitted unheard input transfers only its exact identity.

Ambient observations use a bounded DATA mailbox event, with ordinary `ingested` memory
tags and no authority. Under overload their context may be dropped; they cannot consume
protected control capacity, create a task, retire a valid answer or resolve its metrics.
No database, learned continuation or inference runs in this early capture-side gate.

Addressing uses ADR-0228's separate local decision budget. A missing, malformed, timed-out
or failed decision refuses admission independently of `unsure_acts`; a genuine exact
semantic UNSURE retains the configured ambiguity policy. Missing fast models cannot
silently remove an enabled gate. No owner/direct-live/tool permission is added.

## Context / why

The October 5 live record contains a compatible capture route followed by ambient
activation and instruction-copying. ADR-0226 repairs malformed labels but cannot make
an exact wrong ACT safe. It also ran after arrival-time cancellation. The new candidate
boundary rejects unsupported idle fragments before either expensive classification or
input replacement; the model still decides whether admitted candidate speech is directed.

A blanket wake-word requirement would remove implicit requests. The candidate grammar
therefore retains common questions, commands, greetings and elliptical requests, plus
bounded answers to heard questions. This conservative English heuristic is not a general
semantic proof: uncommon unmarked noun-phrase requests may require an explicit request or
the assistant's name. A coherent hallucinated question can still reach the semantic gate.
It does not certify ASR, model intelligence, acoustic echo suppression or live usability.

## Consequences

Headless regressions cover wrong-ACT ambient rejection, positive requests, typed/mode
compatibility, no cancellation of playing output, overflow without a mailbox fault,
current/auxiliary/stale playback, cue lifetime and identity transfer. Independent review
found and removed a CONTROL-lane ambient flood, missing question subjects, a stale-window
race, unavailable-model fail-open and capture-thread classifier/manager calls.

The targeted gate passes 385 tests; final integration reruns are recorded in
WORKLOG. No microphone/doctor/phone run, enrollment promotion, acoustic default change,
model download, raw audio egress or native quality claim follows. Physical validation
remains required under ADR-0209; later answer-quality work has its own evidence.

### Explicit console assembly

The console factory explicitly disables the ambient classifier for typed input. An enabled
speech gate without a local classifier still refuses; this preserves the no-model console
smoke route without confusing missing classification with semantic uncertainty. Typed input
keeps its existing unknown/unverified instruction provenance and gains no action authority.

### 2026-10-10 addendum — questions in resumed replies

Bind the current synthetic-resume generation to conversational admission only after
terminal ownership and input-generation commit succeed. The response-only resume path
previously published its reply without that identity, so a completed resumed question
could not open its bounded answer window and a short answer was ingested as ambient.
Admission still cannot stand in for rendered speech: the held resumed question leaves
the window closed until its exact completed playback receipt. The resumed event retains
unknown origin, no owner verification, response-only scope and skipped user memory;
no retained query gains direct-live or tool authority. A genuine enabled ResumeConfig
regression fails before the fix and passes afterward; the admission/resume/post-barge/
continuation headless gate passes 112 tests in 4.09 seconds. No native model or live run.
