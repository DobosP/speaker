# ADR-0173: Restrict cleaner context to prior user utterances

Date: 2026-08-10
Status: accepted

## Decision

Build transcript-cleaner recent context only from the newest four canonical
conversational-user items in the runtime memory window. A canonical item has an
exact built-in one-element tag tuple containing the exact built-in string
`"user"` and an exact non-empty built-in string text. Reject assistant output,
ambient/ingested text, meeting notes, summaries, vision, procedural memory,
mixed or future channels, container/string subclasses, and malformed fields
before they can enter the cleanup prompt.

Treat recent context as optional. If the memory snapshot or an item accessor
fails, invoke the cleaner with empty context so deterministic repair and the
normal raw-text fallback still run. Preserve the existing cleaner cancellation
semantics: only `LLMCallCancelled` raised by the cleaner call itself retires the
preprocessing lease.

Do not change the answering model's recent-conversation path. It continues to
receive bounded role-structured user and assistant history through
`collect_recent_turns`. Keep `rewrite_is_overreach`, agreement checks, own-speech
drop, and action-authority stripping as defense in depth after cleanup.

## Context / why

A live failure showed the fast-tier cleaner rewriting a one-word noise fragment
into the assistant's immediately preceding sentence. The runtime supplied the
last four generic memory items to the cleanup model, so `assistant_output` was
present in the prompt and available to copy. The later own-speech and overreach
guards bounded the effect, but they acted only after invented text existed.

Assistant replies are useful evidence for the answering model, which resolves
follow-ups and references across both roles. They are not legitimate correction
material for a transcript cleaner whose job is limited to reconstructing what
the user just said. Reusing the general conversation collector would preserve
the unsafe assistant role and would also apply unrelated topic-reset/truncation
policy. A denylist of known non-user channels was rejected because any future
channel would silently become cleanup context.

The exact tag tuple is trusted in-process channel metadata, not speaker
identity, owner verification, or hostile-producer provenance. Code that can
fabricate memory items can also fabricate this tag; this decision does not turn
memory metadata into action authority.

## Consequences

- The cleanup model may use up to four prior user utterances, in chronological
  order, and its own configured prompt cap remains authoritative afterward.
- Assistant sentences and every noncanonical memory channel are unavailable to
  the generative cleanup pass, closing the observed copy path at its source.
- A memory-context failure degrades to context-free cleanup rather than dropping
  the turn or suppressing deterministic repair.
- Answering-model role history, recall, routing, prompts, models, thresholds,
  defaults, transcript logging, and action-authority rules are unchanged.
- Headless tests do not establish cleaner-model quality, microphone/STT
  accuracy, audio-device behavior, latency, or live conversational behavior.

## Verification

- Focused cleaner plus unchanged answering-history gate: 108 passed.
- Affected preprocessing/runtime/authority/watchdog gate: 273 passed.
- APM/DTD regression: 6 passed.
- Scoped Ruff (with the unchanged whole-file `ACT`-import baseline),
  changed-region format, and whitespace checks are green; pre-existing
  whole-file formatter debt is unchanged from `main`, and independent code and
  landing audits found no blocker.
- No cleaner model, microphone, audio device, GPU, network, or live path ran.
