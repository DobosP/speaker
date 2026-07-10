# ADR-0020: MiniCPM5-1B is the local text and answering tier

Date: 2026-07-10
Status: accepted

## Decision

Use OpenBMB MiniCPM5-1B as the Python voice runtime's local fast/text tier.
Desktop Ollama profiles use the official Q8 GGUF through the pinned local alias
`minicpm5-1b:q8`, provisioned by `python -m tools.setup_minicpm` with the
committed ChatML Modelfile. Phone-class llama.cpp profiles use the official Q4
GGUF and share one model context across the logical main/fast roles. Keep the
existing Gemma 3 main model on Ollama for complex, long-form, planner, and image
turns. Keep the Flutter `flutter_gemma` model until a separately validated
MiniCPM-capable mobile runtime exists.

## Context / why

The owner requested MiniCPM5-1B. The final RL+OPD checkpoint is Apache-2.0,
public, standard Llama architecture, and has official Ollama/llama.cpp GGUFs.
On this host the Q8 artifact used about 1.1 GB VRAM and passed four basic text
probes with roughly 0.33–0.53 s warm TTFT. Its small footprint suits the
always-on local text path and leaves compute headroom for capture and barge-in.

A wholesale main-model replacement was rejected. MiniCPM5-1B is text-only, so
it cannot accept the screen/image inputs that the runtime sends to the main
tier. It also did not follow the repository's strict textual ReAct
`TOOL name: input` contract reliably in direct probes; its advertised native
XML tool calling would require a separate adapter and parser. Retaining Gemma
for those roles prevents a capability regression while ordinary voice answers,
addressing, cleanup, and routing move to MiniCPM.

A direct Hugging Face import into Ollama generated a different stop/template
set and produced empty or inappropriate answers under the real voice prompt.
The supported identity therefore uses OpenBMB's explicit ChatML template and
Ollama no-think mode; the direct `hf.co/...` tag is only a source artifact, not
the configured runtime model.

## Consequences

- Normal local text turns, input gating, cleanup, and routing assistance use
  MiniCPM5-1B; complex, long-form, planner, and visual turns still use Gemma.
- `tools.model_probe` treats text and required-vision fitness separately so a
  text-only model is explicit and a main-model image failure cannot pass.
- Identical llama.cpp main/fast paths share one object, avoiding duplicate
  weights and KV cache that would compete with real-time audio work.
- Existing headless audio/barge behavior is unchanged, but the new model's
  response timing and sentence cadence still require a bare-speaker live A/B.
- MiniCPM's native tool-call format and a non-Gemma Flutter runtime remain
  separate future changes with their own compatibility and validation gates.
