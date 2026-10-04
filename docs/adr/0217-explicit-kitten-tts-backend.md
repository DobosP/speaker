# ADR-0217: Add an explicit, preflighted Kitten speech backend

Date: 2026-10-04
Status: accepted

## Decision

Add `sherpa.tts_backend: "kitten"` as an explicit optional desktop TTS selection.
Keep the empty selector's existing VITS/Kokoro inference. Validate readable model,
voices, tokens and English espeak assets, conclusive bounded ONNX metadata,
style dimensions, voice-table size, speaker ID and native API compatibility before
constructing the Kitten native object. Refuse uncertainty and unsupported selectors.
Setup and readiness share the asset gate; unrelated setup preserves a complete
Kitten family and refuses silent replacement of an invalid explicit family.

## Context / why

The English recording comparison in [ADR-0216](0216-english-model-candidates-and-recording-benchmarks.md)
found Kitten Nano 0.8 INT8 had the smallest isolated TTS process peak, 193.3 MiB,
and median first nonzero PCM of 693 ms. Existing VITS was faster at 134 ms;
Kokoro was slower at 2325 ms. These CPU synthesis measurements justify a concrete
optional trial, not a new default or a phone/audibility claim. Kitten uses a distinct
Sherpa model configuration: the legacy nonempty-voices heuristic selects Kokoro
and cannot safely identify it. Known malformed family/voice metadata can make
native loaders terminate the process instead of raising a Python exception.

The Nano 0.8 INT8 assets are installed separately in the ignored benchmark cache.
An ignored `kitten-runtime-overlay.json` binds those assets, speaker 0, speed 1,
two TTS threads and CPU. The provider field is shared with ASR; this overlay is a
CPU experiment, not a TTS-only device-provider override. Active local configuration
and the shared production virtual environment are unchanged.

## Consequences

The new family can be built through the production `build_tts` factory, rather
than only the benchmark harness. A real factory construction and public-phrase
waveform generation succeeded under installed Sherpa 1.13.3; the local sample is
finite mono 24 kHz audio, 6.566 seconds long. It was not played to an audio device.
The combined CI-style Python gate passed 11543 tests with 42 skips. Metadata and
asset checks cover known constructor abort conditions, not every graph or native
failure; arbitrary corrupted graphs remain outside this preflight guarantee.

No backend default is promoted and no mobile implementation follows from this
Python integration. Subjective voice preference, actual first audibility, owner
bare-speaker A/B and phone CPU/RSS/thermal validation remain required for adoption.
The complete measurements, waveform-peak caveat, primary sources and local trial
instructions are in [the comparison](../english_model_comparison.md).
