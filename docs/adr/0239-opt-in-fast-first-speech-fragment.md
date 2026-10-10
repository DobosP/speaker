# ADR-0239: Opt-in fast first speech fragment

Date: 2026-10-10
Status: accepted

## Decision

Offer `tts.speech_latency: normal|fast` and the matching `--speech-latency` option
through core, doctor and the Linux launcher. Keep `normal` as the default and
preserve the shared desktop/mobile sentence contract. Apply `fast` only to
streamed assistant and research answers: permit at most one extra first-fragment
boundary, then retain sentence delivery. It does not select or download models,
change inference threads, enable streaming, or change capability authority.

The bounded policy searches up to 1024 characters for a neutral clause after
32 characters, or a completed word boundary after 28 words. Configuration permits
8–160 minimum characters and 8–80 words. A safe continuation character is required;
no timer, new worker, repeated synthesis or speculative native generation is added.
Encountered markup, quotations and grouping abstain. Numeric punctuation and
word boundaries adjacent to digits are retained. Full response text remains with
the capability. Capture-only paths reject this assistant option before setup.

## Context / why

[RealtimeTTS 0.8.10](https://github.com/KoljaB/RealtimeTTS/blob/50abd79cfb6033fe2781abc7c97291fe70dcc3ea/RealtimeTTS/text_to_stream.py#L520)
provides fast first fragments and buffered-audio-aware sentence scheduling. Our
installed sentence-generating native TTS cannot produce earlier samples merely
because the transport is named streaming. Supplying an earlier safe fragment is
an incremental option that preserves the existing engine and cancellation owner.
The full RealtimeTTS player/queue lifecycle is not adopted: ours already fences
uncertain native cleanup and exact playback receipts.

Do not split every clause by default. Extra fragments can increase native call
overhead and alter prosody. Leading emotion/voice directives must retain their
scope, and a continuation must not promote an interior directive into a new
fragment-leading command. No model, native audio or live quality claim follows
from public-text scheduling. The owner explicitly postponed live testing.

## Consequences

Cancellation is checked before each fragment and after the first-text observer.
Consumer failures and cancellation close the entered token provider. Provider
failure after one early fragment cannot retry the answer on another model tier.
Normal successful exhaustion keeps its existing cleanup contract. The optional
first-text observer is evidence only and cannot grant task/playback authority.

`python -m tools.benchmark_speech_chunking` is the current reproducible public-text
probe. In its fixed word-arrival schedule, the long clause becomes speakable at
step 10 instead of 25, and an unpunctuated reply at 29 instead of 56; short,
expression-tagged and numeric examples retain the original boundary. Each faster
case incurs one extra fragment. These are synthetic arrival steps, not LLM tokens,
wall-clock latency, synthesis performance, perceived quality or audible onset.

Headless regressions cover every two-part split of protected examples, unchanged
normal splitting, partial delivery before provider completion, full text,
cancellation, callback errors, no duplicate fallback, runtime streaming selection,
CLI parity and capture isolation. Later physical A/B must judge prosody and actual
first audio before any proposal to change the default. Rust is not required for
this bounded policy; native inference remains in the existing runtimes.
