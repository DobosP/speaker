# Pinned English scoring normalizer

Evaluation-only port of OpenAI Whisper **v20250625**, commit
`31243bad24cc746f07d4c8bfdd2d974872cb1803`, under the included MIT license.
Original source digests and byte lengths are in `upstream.json`.

- `english.py` keeps the exact upstream algorithm; one import uses the local
  `_compat.windowed` adapter. The added provenance comment and import change can
  be reversed byte-for-byte to the pinned upstream SHA256.
- `basic.py` extracts the exact upstream `ADDITIONAL_DIACRITICS` table and
  `remove_symbols_and_diacritics` function. `upstream_basic.py.txt` preserves the
  complete tiny source for mechanical parity checks. Unused grapheme splitting
  and its `regex` import are excluded.
- `_compat.py` implements only complete windows of three for the normalizer's
  `[None] + nonempty_words + [None]` input. It intentionally refuses the broader
  more-itertools API. Representative public-text tests and independent window
  tests cover this restricted substitution.
- `english.json` and `LICENSE` are unchanged upstream bytes. The spelling map
  is upstream data, with no owner/corpus/model-specific entries.

No Whisper model, PyTorch, Transformers, regex or more-itertools installation is
required. The runtime and authority plane do not import this evaluation package.
The separate scorer reports local source/data hashes and Unicode data version;
source identity does not attest executed bytecode/import state.

The rules lower case, remove bracketed/parenthesized material and selected
fillers, expand contractions/titles, normalize numbers/currencies, remove
diacritics and map some British spellings. They are heuristics: possessive `'s`
can become `is`, meaningful fillers/annotations disappear, and curly apostrophes
are not handled like ASCII contractions. They do not prove semantic, intent,
numeric-slot, speaker or action equivalence. Keep the legacy score beside the
English-normalized view; the latter's reference denominator may differ.

Primary source: [pinned OpenAI Whisper normalizers](https://github.com/openai/whisper/tree/31243bad24cc746f07d4c8bfdd2d974872cb1803/whisper/normalizers).
Repository decision: [ADR-0247](../../../docs/adr/0247-pinned-english-asr-scoring.md).
