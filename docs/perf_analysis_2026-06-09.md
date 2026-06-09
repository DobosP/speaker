# Voice-pipeline performance analysis — 2026-06-09

Profiled the live `ASR → LLM → TTS` pipeline by replaying the **owner's real
recorded voice** through the real pipeline (the recorded-voice replay harness) and
cross-reading the live run bundles + code. Produced via a 6-dimension fan-out
(profile → synthesize → critique). **This is an analysis/direction document — no
code was changed.**

## Headline

Perceived latency today goes overwhelmingly to **two places**:

1. **The live trailing-silence endpoint wait (~1.16s mean, ≈53% of perceived
   first-audio)** — the single largest, most reducible slice. And right now it is
   throttled **worse than its own validated setting**: `config.local.json` forces
   `endpoint_high_confidence_floor = 0.0`, disabling the validated 0.6 shortening.
2. **STT word quality** on the streaming zipformer (not latency): "STOP"→"JOB",
   "long"→"WRONG", "are"→"OUR" — real errors on the owner's voice, with the
   SenseVoice 2nd-pass currently OFF.

**LLM latency is healthy and holds** — TTFT ~0.5–0.8s on both tiers (the
`think=false` fix); routing is correct (short→`gemma3:4b`, story→`gemma4:12b`);
streaming TTS emits the first sentence early (a 9.2s story still speaks at ~0.5s).

## Measured perceived-latency budget (per turn)

| stage | fast tier | main tier | source |
|---|---|---|---|
| **endpoint silence wait** | ~0.6–0.7s | ~0.7–1.73s | **MEASURED live** (run bundles; mean 1.16s — *inflated* by floor=0.0) |
| ASR final (zipformer) | ~50–100ms | ~50–100ms | MEASURED (RTF 0.076) |
| routing | <5ms | <5ms | estimated (in-proc) |
| LLM TTFT | ~0.51–0.53s | ~0.58–0.78s | **MEASURED** warm (this session) |
| TTS first-audio | ~0.2–0.4s | ~0.3–0.5s | **ESTIMATED — no trustworthy clock** (see gaps) |
| **perceived total → first audio** | **~1.4–2.1s** | **~1.6–3.0s** | endpoint ≈ 40–55% of it |

> The **replay harness cannot measure fine stage timing**: it stamps
> `endpoint_latency≈0` and produced *negative* `first_token_to_audio` (synchronous,
> no real-time playback clock). Trust replay only for **wall time, STT accuracy,
> and LLM-tier**. The endpoint number above comes from the **live run bundles**.

## Ranked directions (impact / effort)

1. **[endpointing, trivial] Re-enable `endpoint_high_confidence_floor=0.6`** (~−80–100ms
   first-audio on well-formed turns). ⚠️ **Not a blind revert:** it was set to `0.0`
   *on purpose* to stop an early-commit splitting "tell me a long story about |
   a lighthouse keeper". **Live-validate that split** before keeping it.
2. **[asr-stt, small] Turn the guarded SenseVoice 2nd-pass back on**
   (`asr_final_backend='sense_voice'`, `asr_final_min_sec=1.0` + the landed
   `agreement_guard`). Fixes the long-utterance word errors. ⚠️ It was reverted for
   short-clip hallucination — the length guard + agreement_guard now exist to handle
   that, but re-A/B the same 6 clips + a short-clip hallucination check.
3. **[measurement, medium] Build a real-time-paced replay engine** (playback thread +
   FIFO; stamp `TTS_FIRST_AUDIO` when samples leave the FIFO). Unblocks *all* future
   latency tuning — today every per-stage number except wall + STT accuracy is
   untrustworthy in replay.
4. **[endpointing, small+livecampaign] Deploy the Smart Turn v3 prosody detector** and
   lower the floor toward ~0.4s (potential −150–250ms first-audio). ⚠️ Model is
   human-audio-only → can't be validated synthetically; needs a live A/B.
5. **[system, small] Relieve RAM/VRAM pressure** (RAM 97% / 31.2 GB, VRAM 11.3/16 GB)
   via lazy fast-tier eviction (`keep_alive`), *if* the two "tts stuck >5s" stalls
   are allocation-pressure (unconfirmed — see gaps).
6. **[tts, small] Token/word-count flush fallback** in `drain_complete_sentences` so
   short fast-tier intents emit before a sentence terminator. ⚠️ risks awkward splits.

## Quick wins (config-only)
- `endpoint_high_confidence_floor` 0.0 → 0.6 (rec #1) — *after* re-validating the split.
- `asr_final_backend` "" → "sense_voice" (rec #2) — model on disk, guards coded.
- If the "tts stuck" warnings are slow-turn false positives, raise
  `TTS_FIRST_AUDIO_DEADLINE_SEC` 5.0 → 10.0 (`core/watchdog.py`).
- **Leave routing alone** — `live_routing=false`, `router.threshold=0.3` are healthy;
  lowering the threshold would cost quality for no latency win.

## Measurement gaps (what to measure next)
- **TTS first-audio** is the biggest blind spot (replay artifact). 
- **The endpoint wait** is the largest perceived slice yet absent from replay.
- **"tts stuck" root cause** unconfirmed (resource vs slow-turn vs FIFO contention).
- **STT-fix A/B** not yet run live.

### Most defensible next action
**One fresh LIVE `--engine sherpa` run on the owner's voice with `floor=0.6` +
SenseVoice on** kills three birds: it gives the faithful endpoint + TTS stage
numbers replay can't, validates the rec-#1 split, and A/Bs the rec-#2 STT fix — all
in a single, real measurement. The recorded-voice harness can then re-pin whatever
that run establishes.

## Adversarial critique notes
- Rec #1 is sound but **was disabled deliberately** (the split) — treat as
  live-validate, not a free revert. Corrected in the ranking above.
- The endpoint mean of 1.16s is from runs with `floor=0.0` (inflated); the real
  post-fix number must be re-measured.
- TTS-stage claims and the "tts stuck" causation rest on **unmeasured** data — do
  not act on them before the paced/live measurement (rec #3 / the next action).
- LLM/routing was correctly judged **healthy** (no busywork there); `think=false`
  and streaming-TTS were verified to hold, not re-proposed.
