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

## Preparing the English BPE streaming context

The current English Zipformer needs model-specific context before it can encode
streaming phrases. On a fresh or lean installation, prepare the checksum-pinned
complete ASR/BPE family without adding a phrase:

```bash
python tools/install.py --bpe-hotwords
```

This installs the required `sentencepiece==0.2.1`, runs the ordinary base setup,
then selects the pinned context as a separate transaction. On Linux,
`./install.sh --bpe-hotwords` is a convenience wrapper for that cross-platform
installer. In an existing full development environment that already has the
exact exporter, the targeted form is:

```bash
python -m tools.setup_models --bpe-hotwords
```

The targeted command checks the exporter before downloading a new family. If
`config.local.json` already contains `asr_hotwords`, first make every phrase
match the uppercase grammar and keep `asr_decoding_method` set to
`modified_beam_search`; setup rejects an invalid existing list before download.

The isolated transaction selects its pinned tokens/encoder/decoder/joiner and
writes `asr_modeling_unit="bpe"`, `asr_bpe_vocab`, its expected SHA-256, and the
`upper_ascii_words` case policy to `config.local.json`. It leaves every other
model/capability field and `asr_hotwords` unchanged. It does not inspect a vault.
Each phrase must contain only uppercase ASCII words separated by single spaces;
use machine-local candidates such as:

```json
{
  "sherpa": {
    "asr_hotwords": "VAULT\nOBSIDIAN\nPAUL BRAIN\nDOBO BRAIN"
  }
}
```

The existing command router already handles variants such as “search in my
vault”, “go in my vault”, and “find in my vault”; these hotwords reinforce the
domain terms at the acoustic decoder instead of creating a vault-specific entry
point. Gate any phrase/score change with labelled target and negative recordings:

```bash
python -m tools.recorded_stt_eval \
  --keyword vault --keyword obsidian \
  --set 'asr_hotwords="VAULT\nOBSIDIAN\nPAUL BRAIN\nDOBO BRAIN"'
```

A tie cannot promote a candidate. Zero target attempts provide no target-word
evidence even when aggregate metrics improve; this retained run tied and reports
`promotable=false`. See
[ADR-0114](adr/0114-wire-explicit-bpe-hotword-context.md) and
[ADR-0078](adr/0078-gate-stt-changes-on-private-recording-ab.md).

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
