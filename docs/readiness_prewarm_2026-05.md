# Ready-to-fire: startup pre-warm (2026-05)

> ⚠️ **Superseded — durable content merged into [`docs/unified_architecture.md`](unified_architecture.md).** Kept for revision history; do not treat as current. (2026-06-02 consolidation.)

Goal (user): *code + models already loaded and ready to fire, for the best
possible latency when the interaction begins.* The pieces existed in spirit
(`warm_on_start`) but had real holes: **`engine.warm()` was dead code** — the
runtime called `getattr(self.engine, "warm", None)` but no engine implemented
it, so STT/TTS/VAD never paid their cold cost before turn 1; and the LLM warm
used a bare `"hi"` with no system prompt, leaving the cacheable system prefix
cold.

## What changed

- **Real `engine.warm()`** (`core/engine.py` base no-op + `core/engines/sherpa.py`
  + `core/engines/livekit.py`). The sherpa engine warms the models that stay cold
  until the first reply / first final: the **TTS** (synthesize a throwaway `"ok"`
  and discard it — never enqueued for playback, so nothing is heard), the
  **punctuation restorer**, and the **speaker-ID embedder** (both run only on a
  final). The LiveKit (remote) engine warms its TTS the same way. All best-effort.

  **Why not ASR/VAD/KWS:** the recognizer and keyword spotter are fed every 100 ms
  by the capture loop from the moment capture starts, so they JIT on the first
  blocks of *ambient* audio (self-warm); the VAD is tiny and only used during
  playback.

  **Concurrency:** a `_tts_lock` makes the warm synthesis and a live synthesis
  mutually exclusive (sherpa's `OfflineTts` / the LiveKit synth aren't safe for
  concurrent `generate`); it's uncontended on the hot path (the playback thread
  is otherwise the sole synthesizer). `LlamaCppLLM` likewise gained a lock so the
  warm pass and a concurrent first turn share the single llama context safely.

- **LLM warm uses the real system prompt** (`core/runtime.py`). The runtime keeps
  the capability-aware `system_prompt` it built and warms each local model with
  `generate("hi", system=system_prompt)`, so the (now longer) system prefix is
  prefilled into the KV-cache instead of being filled on turn 1's first token.
  A **cloud-hybrid** (`HedgeLLM`) is not warmed whole (that could egress before a
  real turn), but its purely-**local leg** is, so the local tier isn't left cold.

- **Gate + cleaner warmed.** The addressing classifier and transcript cleaner run
  on the fast tier with their own system prefixes, so `_warm()` exercises each
  once (`classify("hi")` / `clean("hi")`) — their first live call isn't cold.

- **Readiness signal.** `runtime.warm_ready` (a `threading.Event`) is raised when
  the background warm finishes — in a `finally`, so readiness means "warm-up
  finished", even if a step failed. Set immediately when `warm_on_start` is off.

## Scope / follow-ups

- Cold-vs-warm **turn-1 latency** is verified by `tools.bench` / real-model runs,
  not the logic suite (no models in CI). The tests here pin the *wiring*.
- A warm `generate()` has no hard timeout; a hung backend leaves `warm_ready`
  unset (waiters use timeouts; the warm thread is a daemon). Bounding it belongs
  with the **never-stuck controller** work (per-task deadlines).
- Deeper ASR warm (decoder/joiner on synthetic speech) and the punctuation /
  speaker-ID models are not explicitly warmed; the capture loop covers the ASR
  encoder and the others are small.

## Tests

`tests/test_readiness.py` — base/sherpa `warm()` no-op guards, `warm_ready`
immediate-when-off / set-on-finish-even-on-failure, warm uses the real system
prompt, `engine.warm()` invoked, gate + cleaner warmed.
