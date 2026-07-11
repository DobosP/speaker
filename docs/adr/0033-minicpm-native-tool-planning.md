# ADR-0033: Use MiniCPM5 native XML for phone read-only tool planning

Date: 2026-07-11
Status: accepted

## Decision

Opt the in-process MiniCPM5 main client in `phone` and `phone_lite` into a typed
native XML step backend while leaving desktop Gemma/Ollama and unconfigured
providers on textual ReAct. Offer at most the four configured and registered
capabilities marked `planner_tool` and not `side_effecting`, each with one
required string `query`; bound its model-facing guidance to 96 characters.
Buffer at most 256 output tokens, preserve MiniCPM's special tool tokens only
inside that completion, require a natural stop, and strictly parse exactly one
complete `function`/`param` object before execution.

Keep cancellation, admission claims, deadlines, step limits, invocation,
untrusted-result handling, and final-output validation in the ReAct controller.
Malformed or protocol-bearing output may receive one bounded reprompt but must
never execute a partial or unknown call or reach TTS. Bound phone prompts,
recent context, schemas, and canonical history for the shipped 1536-token
context and retain only the latest exchange. Neutralize XML structure in
replayed calls/results and preserve a balanced compact spotlight envelope for
untrusted egress data.

## Context / why

MiniCPM5 did not reliably follow the repository's textual `TOOL`/`FINAL`
contract. Its [official template](https://huggingface.co/openbmb/MiniCPM5-1B/blob/4e9de7a0778dc1c362e983e6858f0e77542cbdca/chat_template.jinja) instead emits
`<function name="..."><param name="query">...</param></function>`, and its
function/parameter delimiters are tokenizer control tokens. The pinned
[`llama-cpp-python` generic Jinja path](https://github.com/abetlen/llama-cpp-python/blob/e894f0d6010be8de14400359c10c87c16ddb3829/llama_cpp/llama_chat_format.py#L610-L755) detokenizes those tokens with
`special=False`, exposing malformed attribute fragments even when the model
generated the correct token IDs. A narrowly scoped `special=True`
reconstruction is therefore required; changing ordinary detokenization
globally would leak other control tokens.

Streaming or permissively extracting XML was rejected because an incomplete,
mixed-prose, duplicated, or attacker-shaped call must not become executable.
Side-effecting tools were also excluded until separately designed confirmation
and replay semantics exist. This implements the separate native adapter
anticipated by ADR-0020; it does not change that ADR's desktop, vision,
Flutter, or model-selection decisions, so ADR-0020 remains accepted.

## Consequences

Phone-class MiniCPM can perform bounded read-only planning without imitating
Gemma's textual protocol, while desktop behavior is unchanged. Long plans lose
native history older than the latest exchange, only four tools can be exposed
per step, and strict validation can prefer a safe apology over ambiguous
output. Changing the MiniCPM template, llama-cpp-python pin, special-token IDs,
or phone context requires re-auditing this private compatibility seam.

The accepted actual-Q4 headless gate rendered the worst bounded prompt at a
conservative, nonce-stable 1254/1279 input tokens. Two consecutive runs
cancelled native inference in 2.2 ms with healthy same-context reuse and each
completed exactly one harmless `search.local` to correct final-answer round
trip in 2.19--2.27 s. This does not validate a phone device, microphone,
speaker, thermal behavior, or acoustic barge-in.
