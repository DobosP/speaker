# ADR-0031: Keep MiniCPM reasoning outside voice output

Date: 2026-07-11
Status: accepted

## Decision
For in-process llama.cpp voice turns, render a reasoning-capable GGUF's embedded
`chat_template.default` with an explicit Boolean `enable_thinking`; default it to
false. Reject an implicit mode, a reasoning template without that control, or a
custom handler/format that bypasses the audited template. When deliberate
thinking is enabled, track the closer actually prefilled by the installed
handler. Independently filter split, nested, case/whitespace-deformed, malformed,
or truncated `<think>` and `<|thought_*>` blocks before the first stream yield,
TTS, or returned assistant text; fail the provider call if markup is unclosed or
no safe answer remains. Keep the desktop Ollama path unchanged. The real-model
gate must require direct known answers, native `finish_reason=stop`, no filter
fallback, and healthy same-context reuse after native cancellation.

## Context / why
MiniCPM5's supported template defaults to an automatic thinking mode. Its
thinking branch prefills `<think>`, so direct llama.cpp generation begins with
reasoning text but without a generated opener; the old phone stream would pass
that text toward TTS or exhaust its voice cap before a final answer. Merely
stripping generated tags cannot recover an answer that was never generated, so
generation-time no-think control is required. Conversely, relying only on the
template would let model/template drift expose reasoning, so the output seam
also needs a chunk-safe fail-closed filter. `llama-cpp-python` 0.3.33's direct
chat-completion signature has no chat-template-kwargs argument; its bundled
server applies such kwargs by wrapping the resolved Jinja handler after model
construction, so the pinned in-process path mirrors that audited mechanism.

## Consequences
The voice path produces the final answer directly instead of spending its cap
on a silent/spoken chain of thought. The 2026-07-11 Q4 functional gate, fixed at
2 generation / 2 batch threads for topology-independent correctness, returned
Paris, a complete joke, and four with 130.2--255.9 ms TTFT, natural `stop`
finishes, zero fallback blocks, a 6.5 ms cancellation exit, and healthy reuse.
That fixed test topology is not production thread tuning; generation/batch
auto-tuning remains a separate P1 decision. The private handler seam and marker
contract must be re-audited before changing the exact llama-cpp-python pin or
the MiniCPM template. No live phone, microphone, speaker, or acoustic A/B was
performed.
