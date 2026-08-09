# ADR-0167: Preserve complete model selections by family

Date: 2026-08-09
Status: accepted

## Decision

Make `tools.setup_models.wire_sherpa_paths` the single configuration boundary
for model-family preservation and replacement. A complete, runtime-consistent,
on-disk selection survives every incidental setup default. Replace it only when
the caller names that exact family and supplies one complete successful target;
a raw CLI request whose fetch failed grants no replacement authority.

Treat streaming ASR, TTS, offline final ASR, final verifier, KWS, VAD, speaker
embedding, punctuation, denoise, prosody, and DTLN AEC as separate families.
Stage a copy of the Sherpa configuration, validate every touched target before
changing it, clear that family's entire replaceable key universe, write all of
its target paths and selector mode together, and assign the staged object once.
An invalid or incomplete existing family may be repaired by a complete fallback
even without replacement authority; a partial incoming family fails without a
configuration mutation.

TTS validity follows the same lightweight ONNX metadata classifier used at
runtime: recognizable VITS/Piper bytes cannot be paired with Kokoro voices and
recognizable Kokoro bytes cannot be selected as VITS. Inconclusive custom ONNX
metadata remains allowed. Faster-Whisper requires all four snapshot leaves,
KWS requires all five paths, final-ASR requirements follow its exact backend,
and DTLN requires both stages. Configured comma-separated lexicons/FSTs are
validated item by item when they are load-bearing.

An explicit `--accuracy` may replace streaming ASR; successful optional model
flags may replace only their own families; an explicit speaker-model URL flag
or environment override may replace the speaker family; `--require-selected`
authorizes the base ASR/VAD/TTS/speaker selection required by the installer.
The isolated BPE setup keeps its stricter complete-context rule and explicitly
replaces only the streaming-ASR family plus its pinned vocabulary metadata.

## Context / why

The old `preserve_existing_kokoro_selection` special case examined only
`tts_model` and `tts_voices`, then deleted three Piper defaults from a mutable
resolved mapping. It could preserve an already partial family, did not protect
custom ASR or speaker selections, and left family policy scattered outside the
function that actually writes every path.

More seriously, `--kokoro` disabled that special preservation before the
Kokoro fetch was known to have succeeded. A best-effort fetch failure therefore
left resolved Piper model/tokens/data beside the old Kokoro voices/lexicon and
published the exact family hybrid that runtime preflight was designed to reject.
SenseVoice over NeMo likewise left stale decoder/joiner paths. Atomic JSON
publication protected file bytes, but it could still publish a semantically
mixed in-memory object.

Treating every path independently or interpreting `--force` as blanket
replacement authority would still downgrade unrelated custom selections.
Keeping more one-off preservation helpers would repeat the same partial-family
mistake. Family-aware staging at the one writer is the smallest general rule.

## Consequences

- Ordinary setup fills absent or invalid families while preserving every
  complete custom family. A successful explicit family request replaces that
  family coherently; a failed best-effort request preserves a valid old family
  and may fall back only when no valid old selection exists.
- Final-ASR and verifier backend strings are published in the same staged
  operation as their files. Replacing NeMo with SenseVoice removes stale
  decoder/joiner members; repairing invalid Kokoro to Piper removes stale
  voices/lexicon members.
- The model downloader may still fetch defaults that are not selected. This
  decision changes only configuration admission and does not delete artifacts,
  alter a runtime default, select a model, or authorize a live session.
- Existing BPE context, config compare-and-swap publication, archive extraction,
  model checksums, installer failure policy, and normal voice entry points remain
  unchanged.

## Verification

Headless tests must cover complete-family preservation, explicit replacement,
invalid-family repair, partial-target rollback, TTS metadata mismatch, failed
best-effort Kokoro, final-ASR selector/member cleanup, exact verifier/KWS/AEC
membership, non-object configuration rejection, BPE isolation, and atomic config
publication. Run the affected setup/doctor/TTS suites and the required APM/DTD
gate at low priority; record settled counts in `STATUS.md`. No download, model,
GPU, audio-device, service, or live run is required or claimed.
