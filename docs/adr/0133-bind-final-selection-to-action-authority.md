# ADR-0133: Bind final selection to action authority

Date: 2026-08-05
Status: accepted

Refines: ADR-0076, ADR-0079, ADR-0080, ADR-0084

## Decision

Compare the final text selected for dispatch with the streaming ASR final at
the last engine boundary before typed publication. Use a deliberately narrow
rendering key that mirrors the current exact reminder/app grammars: ignore case,
surrounding ordinary ASCII spaces, and at most one trailing ASCII sentence
terminator. Preserve internal spacing and every other punctuation, Unicode,
and non-ordinary-whitespace difference. If that key changes, clear
`OwnerVerification.VERIFIED`, replace the `live_audio` origin with `unknown`,
and emit one transcript-free
`asr_final_selection_untrusted` metric. Continue delivering the selected text
for ordinary conversation.

Keep the stricter existing verifier rule: a `verifier_changed` decision loses
authority even if its two rendering keys are equivalent and continues to
emit the established `asr_verifier_changed_final_untrusted` metric. Emit only
one demotion metric for a final. Apply the rendering-key comparison whether the
selector is the built-in offline recognizer, punctuation stage, authorized
empty-stream recovery, diagnostic path, or an embedder-provided
`_final_transcribe` override. Do not require a retained
`FinalTranscriptDecision` for this safety property.

## Context / why

ADR-0076 requires reminder and trusted-app mutations to originate from an
unchanged direct live-audio instruction. The runtime correctly clears this
authority after its later transcript cleaner changes tokens, and the engine
already cleared it for verifier-backed changes. The earlier final-selection
stage did not enforce the same invariant.

In the normal no-verifier, no-diagnostic path, `_final_transcribe` returned only
the selected string and discarded its `FinalTranscriptDecision`. A SenseVoice
offline correction or custom override could therefore change the streaming
words while the result still left capture as verified `live_audio`. Diagnostic
mode retained the decision but demoted only `verifier_changed`, not an ordinary
offline rewrite. Runtime then derived `direct_user_instruction` from the live
origin, allowing rewritten content to cross a capability authority boundary.

Forcing every selector through the diagnostic decision object would change the
established extension seam and make recording affect behavior. A generic text
normalizer is also unsafe here: it can fold `opén`, `op'en`, or full-width
letters into `open`, even though the raw mutation grammar rejects the first
rendering and accepts the second. Comparing the two already-owned strings with
the narrow rendering key closes all paths without changing selection,
diagnostics, model invocation, or callback count.

## Consequences

Offline and custom corrections remain available to improve conversational
answers, including explicitly authorized recovery from an empty streaming
final, but a token-changing result cannot stage or confirm reminders, open a
trusted application, or acquire owner-required action authority. Equivalent
case/surrounding-space/trailing-terminator rendering changes retain the existing
direct-live behavior. Read-only tools and ordinary dialogue remain available
under their existing policies.

Model selection, final-ASR defaults, enrollment, barge-in, entry points,
diagnostic selected-source reporting, and verifier consensus are unchanged.
Headless tests cover default, custom, verifier, diagnostic, rendering-equivalent,
and empty-stream paths. No model, GPU, network, microphone, audio device, live
tool execution, WER, or latency evidence follows from this change.
