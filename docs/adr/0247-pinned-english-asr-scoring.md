# ADR-0247: Preserve legacy ASR scores beside pinned English normalization

Date: 2026-10-10
Status: accepted

## Decision

Add a separately named English-normalized, aggregate-only evaluation view in
`tools/english_asr_scoring.py`. Keep the existing English benchmark's ASCII
metric byte-for-byte and retain both views in prospective comparisons. Apply
normalization only to private reference/hypothesis pairs after decoding; never
change PCM, model inputs, prompts, hotwords, runtime trust or consensus. Earlier
scores and receipts remain historical and are not rewritten.

The normalizer is an evaluation-only stdlib port of MIT OpenAI Whisper
**v20250625**, revision `31243bad24cc746f07d4c8bfdd2d974872cb1803`.
`english.py` keeps the upstream algorithm with one local window-helper import
and a provenance comment. The Unicode helper/table are exact upstream extracts;
the original tiny `basic.py` source is retained for mechanical comparison.
The spelling JSON and license are unchanged. The three-word window adapter
implements only the full windows actually used by EnglishNumberNormalizer;
it refuses unrelated padding/step/general-window APIs. No PyTorch, Transformers,
regex, more-itertools or production dependency installation is introduced.

## Context / why

The existing scorer lowercases and extracts `[a-z0-9']+`, so spoken number words
and an inverse-text-normalized digit representation can disagree despite the
same content. A new metric must not erase that historical lexical signal.
Neither an established Whisper normalizer package nor its two small optional
dependencies was installed in the canonical environment. A narrowly scoped
port preserves established English rules without adding inference dependencies
or corpus-specific substitutions. The [pinned upstream source and data](https://github.com/openai/whisper/tree/31243bad24cc746f07d4c8bfdd2d974872cb1803/whisper/normalizers)
and [MIT license](https://github.com/openai/whisper/blob/31243bad24cc746f07d4c8bfdd2d974872cb1803/LICENSE)
define its provenance.

## Consequences

`score_english_normalized_pairs` returns the same nine numeric aggregate keys
as the legacy view, using whitespace-separated normalized tokens. CER joins
those tokens without spaces, preserving numeric symbols. Both denominators are
`max(1, total_reference_units)`; normalization can empty a reference, change its
word count and yield insertion ratios above one. `normalization_disagreements`
returns only per-clip count comparisons, never input or normalized text. A lower
normalized error count is a formatting-sensitive difference, not model quality
improvement. References/hypotheses remain private and no decoder is called.

`scoring_metadata()` reports fixed public normalizer/version/upstream hashes,
actual spelling-data hash, Unicode data version and a bounded nine-file local
source/data digest. Callers compare metadata before/after scoring and bind it
to their prospective report. It attests source/data snapshots, not executed
bytecode, import state or later filesystem changes. Bounds are 256 pairs,
1,024 reference characters, 4,096 hypothesis characters and 16,384 normalized
characters per string; invalid input or failures expose one detail-free error.
The normalizer's only supported input is English evaluation text.

The upstream rules expand contractions/titles, normalize numbers/currencies,
remove selected fillers and bracketed/parenthesized content, strip diacritics
and map British spellings. They can hide semantic distinctions: possessive
`'s` can become `is`, annotations/fillers disappear and curly apostrophes do not
behave like ASCII contractions. This is no intent, numeric-slot, command,
speaker, authority or conversation-success metric. No corpus-specific fixes
are added. Owner37 provenance remains unverified; Mic6 remains a separate
small development corpus. Neither becomes a disjoint holdout through scoring.

## Verification

The public-text scorer and unchanged English ASR benchmark gate passed
**117 tests in 0.80 s** with the standard headless single-thread environment.
Tests mechanically reverse the English import/comment changes to the pinned
upstream SHA256, compare the Unicode AST extracts, verify unchanged data/license
bytes, check independent full-window parity and representative number/currency/
contraction cases, preserve exact legacy counts, and exercise bounded private
errors, source replacement and numeric-only summaries. Runtime/authority
sources do not import the evaluation helper. Scoped lint and whitespace pass.
No native model, private recording, network inference, microphone, doctor,
device or live validation ran. Prospective harness integration and fresh
per-model aggregate disagreements are separate qualifications.
