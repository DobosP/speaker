# Smart endpoint: semantic turn-completion (2026-05)

Goal (user): be smart about *when* to speak. The endpoint decision — "has the
user finished their turn?" — was a single fixed acoustic timer: commit a final
after `asr_rule2_min_trailing_silence` (0.8 s) of silence, identical on every
device and **blind to what was said**. That cuts off a slow speaker mid-thought
and adds a fixed ~0.8 s tail to a turn that obviously ended (the #1 real latency
gap per the audit).

## What changed

- **`core/endpointing.py` (new)** — a pluggable `TurnCompletionDetector`
  (`completion_score(text, samples, sample_rate) -> 0..1`) + an
  `AdaptiveEndpointPolicy`:
  - **SHORTEN** — when the partial clearly reads as a complete turn, commit early
    (down to `endpoint_min_silence_sec`, ~0.2 s) — the latency win.
  - **EXTEND (bounded)** — when it ends mid-phrase ("…and", "…the", "um"), hold
    past the acoustic timer up to `endpoint_max_silence_sec` (1.6 s), so a pause
    isn't mistaken for the end. `rule3` (20 s) remains the ultimate backstop.
  - else the unchanged acoustic decision stands.
- **`core/engines/sherpa.py`** — `SherpaConfig` gains `endpoint_*` fields
  (default disabled); the engine builds the detector + policy when enabled; the
  capture loop gates the existing final-emit on a pure, unit-tested
  `_decide_endpoint(...)` instead of `recognizer.is_endpoint` directly. Disabled
  (the default) → `_decide_endpoint` returns the acoustic decision exactly →
  **byte-identical** behaviour.

## The shipped detector + the upgrade path

v1 is `LexicalTurnCompletionDetector` — cheap, deterministic, no model, no audio
work on the capture thread. It scores from the **last word** of the partial:
a turn ending on a word that ~never ends a sentence (conjunction / article /
filler, EN + RO) is mid-thought; otherwise it's likely complete.

The set is **deliberately conservative**, on this asymmetry:

> A false-**SHORTEN** (commit too early) is **recoverable** — the user's
> continued words become a new final that the ADD-ON / continuation layer merges
> back in. A false-**EXTEND** (hold a finished turn) is **not** recoverable — it
> only adds latency.

So the detector errs toward "complete": stranding-prone prepositions
("what are you waiting **for**"), terminal pronouns ("what time is **it**"), and
auxiliaries ("here it **is**") are **excluded** from the incomplete set.

The `TurnCompletionDetector` protocol also takes the recent **audio** and
declares `needs_audio`, so a prosodic model (Pipecat **Smart Turn v3**, ~8 MB
ONNX) drops in through the same seam — the higher-accuracy upgrade — without
re-architecting. The capture loop assembles the audio buffer **only** when a
detector sets `needs_audio` (the lexical default pays nothing). The audio model
also resolves the lexical limits below, since prosody (pitch fall, pause shape)
distinguishes "the capital of [done]" from "the capital of … [thinking] France"
that a last-word check cannot.

## Known limits of the lexical v1 (why it's EXPERIMENTAL / default-OFF)

These come out of the adversarial review and are the on-device-validation items:

- **Decoder lookahead vs. early commit.** The streaming decoder emits a word only
  after ~0.3–0.6 s of following frames. Committing earlier than that and
  `reset()`-ing clips the last word *irrecoverably* (it was spoken but not yet
  decoded — the continuation layer can't recover it). So `min_silence_sec`
  **must exceed the model's lookahead** — defaulted to **0.5 s** (latency win
  0.8 → 0.5 s); validate per model on device.
- **False-SHORTEN on a stranded preposition.** "what is the capital **of** …
  France" scores complete (preposition-stranding makes "of" unreliable). If the
  user pauses past `min_silence` before "France", the question is committed
  short. Rare at 0.5 s, but real; the Smart Turn audio model fixes it.

## Validation caveat

No ASR models or audio in CI, so the **capture-loop integration and the real
latency win are validated on device**, not by the logic suite. Default OFF so
nothing ships changed until measured (record turns ending on prepositions and
diff SHORTEN finals against the full-acoustic finals); the tests here pin the
pure decision logic. Enable via `sherpa.endpoint_enabled`.

## Tests

`tests/test_endpointing.py` — lexical scoring (complete / mid-phrase / short /
empty / RO), policy shorten/extend/backstop/fallback, `EndpointConfig.from_sherpa`,
and the engine's `_decide_endpoint` (disabled passthrough, shorten, bounded
extend→backstop, empty-partial skip, detector-error fallback, config parse).
