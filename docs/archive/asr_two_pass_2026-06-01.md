> **SUPERSEDED (2026-07-02):** refined by the 2026-06-17 **async** second pass —
> the streaming final now dispatches immediately and SenseVoice upgrades it in
> place behind a fail-closed agreement guard; see
> [`docs/session_2026-06-17_async_second_pass_asr.md`](../session_2026-06-17_async_second_pass_asr.md)
> and `docs/unified_architecture.md` §7. Two-pass itself remains the shipped
> default (`asr_final_backend='sense_voice'`, config.json).

# Two-pass ASR: SenseVoice final recognizer (2026-06-01)

Fixing the real STT wall: the streaming GigaSpeech zipformer garbles **run-on /
casual** speech (it decodes incrementally and never sees the whole utterance). A
live session transcribed as `HEY IRIC LISTENING TO ME` / `THIS TURNED OFF THE
KITCHEN LIGHT`. Diagnosis + a real-voice bakeoff drove a two-pass upgrade.

## Diagnosis

- The recorded audio was clean (rms ~0.10, <0.1% clipping) — not a mic problem.
- On **clear, separated** reads the zipformer is fine (`THE CAPITAL OF FRANCE IS
  PARIS` perfect). It only garbles **continuous** speech (the live session was 71%
  voiced, 2.9 s runs).
- Whisper (gold standard) turned the word-salad into coherent English — so a
  stronger model that sees the whole utterance is the lever.

## Bakeoff (the user's real voice)

| model | run-on ("are you listening to me") | clear reads | 2nd-pass latency | verdict |
|---|---|---|---|---|
| streaming zipformer | `HEY IRIC LISTENING TO ME` ✗ | rough, ALL-CAPS, no punctuation | — (streaming) | the wall |
| **SenseVoice** | `Hey, are you listening to me.` ✓ | clean + punctuated, correct | **~55 ms** (2 threads) | **winner** |
| whisper-base | ✓ | good (one error) | ~300 ms | slower |
| moonshine | ✓ | **empty on half the clips** ✗ | ~90 ms | unreliable |

SenseVoice: accurate, robust to run-on, fast, reliable, sherpa-onnx native, and it
bundles **punctuation + casing + ITN** (so the separate casing/punct post-process
is skipped on the second-pass final).

## Architecture — two-pass

The streaming transducer keeps giving low-latency **partials + the endpoint**;
when `asr_final_backend` is set, the **endpointed utterance is re-transcribed** by
the offline model for the FINAL text that reaches the LLM.

- `core/engines/_sherpa_models.build_final_recognizer` (`sense_voice` | `whisper`;
  **fail-open** — a bad config keeps the streaming final, never breaks capture).
- `core/engines/sherpa._final_transcribe` — second pass with a min-length gate +
  empty/error fallback to the streaming final; wired at the endpoint.
- `tools/setup_models.py --sense-voice` downloads + wires it (~230 MB tar.bz2 →
  `pretrained_models/sherpa/sense_voice/`).

## Shipped default (2026-06-01)

`config.json` defaults `asr_final_backend=sense_voice` pointing at the standard
setup_models path — **default-ON when the model is present, byte-identical
streaming-only when absent**. Live-confirmed by the user ("felt nicer"); the
clean, punctuated finals are visible in the run transcript (`Are you ready.` →
`Yes, I am.`). Cost: ~55 ms/utterance.

## What it does NOT fix (measured, honest)

- **Mid-thought cutoff** ("Tell me" *[pause]* → cut off). Prosody can't catch it
  (the paused fragment scores **0.96 = complete**: a short phrase + falling pitch
  + silence is acoustically a finished turn), and the continuation layer only
  merges **cue-word** add-ons ("and also…"), not a sentence completion. The only
  lever is endpoint **patience** (a latency trade) — left snappy by user choice.
- **Barge-in** stays off on open speakers (self-interrupts + a PortAudio crash
  without AEC; the speaker-gated no-AEC path is the open follow-up).
