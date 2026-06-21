# ASR contextual biasing — fixing mis-heard names & jargon

The recognizer occasionally mis-transcribes a specific name, brand, or piece of
jargon ("Iric" for "Eric"). There are **two** biasing surfaces, and they apply to
**different passes** — picking the wrong one is a no-op.

## The two passes (why it matters)

The engine runs a **two-pass** ASR (`docs/unified_architecture.md`):

1. **Streaming transducer** — low-latency partials + the endpoint, and the FINAL
   for very short clips / when no second pass is configured.
2. **SenseVoice second pass** (`asr_final_backend="sense_voice"`, the default when
   the model is present) — re-transcribes the whole endpointed utterance and
   **overrides** the streaming text for any normal-length turn. This is the text
   the LLM sees.

So: a fix that biases only the *streaming* pass does **not** change the FINAL for a
normal turn — the second pass overwrites it.

## Which knob to use

| You want to bias… | Use | Config field |
| --- | --- | --- |
| The **final** transcript (the LLM-facing text) — the common case | SenseVoice **homophone replacement** + rule FSTs | `asr_final_hr_dict_dir`, `asr_final_hr_lexicon`, `asr_final_hr_rule_fsts`, `asr_final_rule_fsts` |
| Only the **streaming** partials / short-clip finals | transducer **hotwords** | `asr_hotwords` (newline list) + `asr_hotwords_score`; needs `asr_decoding_method="modified_beam_search"` |

`asr_hotwords` is the intuitive one and the one to **avoid** for fixing names in
normal turns — it's overridden by the second pass.

## Authoring SenseVoice homophone replacement

`asr_final_hr_*` wire sherpa-onnx's homophone replacer (the same machinery the
upstream `homophone-replacer` example uses). Point them at:

- `asr_final_hr_dict_dir` — a directory of the homophone/replacement dictionary.
- `asr_final_hr_lexicon` — a lexicon mapping tokens to pronunciations.
- `asr_final_hr_rule_fsts` / `asr_final_rule_fsts` — compiled replacement FSTs
  (e.g. force "iric" → "Eric").

All default empty (byte-identical). An older sherpa-onnx build that predates these
params drops them safely (`_supported` filters unknown kwargs), so setting them
never breaks capture. Author the dict/FSTs per the sherpa-onnx docs, drop them in
`config.local.json`'s `sherpa` block, and re-run — the change lands on the FINAL
transcript with no latency cost.

> Tip: the cheapest fix is often a single rule FST forcing the one or two tokens
> the model reliably gets wrong, rather than a full dictionary.
