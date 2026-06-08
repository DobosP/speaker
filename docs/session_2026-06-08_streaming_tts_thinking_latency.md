# Session 2026-06-08b — Streaming-TTS latency: disable reasoning-model "thinking" (9.5s → 2.4s to first audio)

**Branch:** `fix/llm-think-off-tts-latency` → merged to `main` @`b2cddc4`.
Logic suite **1395 passed, 14 skipped, 0 failed** (Windows `.venv` python, ~55s).
**Machine:** i9-13980HX / 30 GiB / RTX 4090 Laptop 16 GB, **Windows** (same box,
Windows side; prior barge-in work was the Linux side). Ollama 0.30.x up with
`gemma4:12b` (main), `gemma3:4b` (fast), `gemma4:e4b`, `gemma3:1b/12b`.

## Owner ask (improvement #1 from the prior handoff)
> "Stream the TTS for long answers — don't wait for the whole LLM answer. A long
> story feels like it waits for the full generation before speaking. Goal: first
> audio after sentence 1, not after the whole story."

## Headline: it was NOT the streaming chain — it was reasoning-model "thinking"
The whole sentence-streaming path is already correct and fully incremental:

```
Ollama token  -> OllamaLLM.stream (yields each content piece)
              -> _stream_and_speak (core/capabilities.py): drain_complete_sentences, emit() per sentence
              -> emit -> TTS_REQUEST AgentEvent (always_on_agent/tasks.py _make_emitter)
              -> EventBus background thread -> VoiceRuntime._on_event -> engine.speak(sentence)
              -> sherpa _play_q -> playback worker -> _synthesize (streaming) -> PlaybackFIFO -> audio
```

`stream_tts=true` is the production default (`config.json tts.streaming=true`,
survives the profile merge; both `app.py` and `remote.worker` enable it). The bus
runs a background dispatch thread in the live path, so each `TTS_REQUEST` reaches
playback as it is emitted. **None of that was broken.**

### Root cause (measured, not guessed)
The main tier `gemma4:12b` is a **reasoning model**. It streams a silent
chain-of-thought into a **separate `thinking` field** *before* any `content`
token, and `OllamaLLM.stream` only yields `content` — so for a **story** (which
needs no reasoning) the user heard nothing while the model "thought".

Proof — raw Ollama `/api/chat` with `num_predict=80`: `content=''`, `thinking`
held the model planning the story ("Atmospheric, perhaps slightly melancholic…
*Setting:* A remote island…"), `done_reason: length`. The whole budget went to
thinking; zero spoken output.

Measured TTFT through the **exact streaming path** (real `OllamaLLM.stream` +
`drain_complete_sentences` + the real persona system prompt), warmed, 100% GPU:

| tier | model | content first-token | first spoken sentence | full answer |
|---|---|---|---|---|
| main | gemma4:12b (thinking on) | **9.1 s** | **9.5 s** | 11.2 s |
| fast | gemma3:4b (no reasoning) | 1.7 s | 1.8 s | 2.0 s |

A "story"/"tell me a" query routes to **main** via `_GENERATION_MARKERS`
(`core/routing.py`), so it always paid the 9 s thinking tax. Streaming TTS only
saved ~1.7 s (the model generated the whole story in ~2 s once it *started*), but
the 9.5 s of dead air dominated → "feels like it waits for the whole answer".

## The fix
`think=False` collapses the latency (Ollama `think` is a **top-level chat arg**,
not an `options` key):

- `core/llm.py` `OllamaLLM`: new `think: Optional[bool] = None` ctor param;
  `_chat_kwargs` adds `think` **only when not None** → non-reasoning models and
  older Ollama builds are byte-identical (verified: `gemma3:4b` accepts
  `think=False` as a graceful no-op).
- `core/llm_factory.py` `build_llms`: reads `llm.think`, **defaults `False`**
  (an explicit False, since `None` would defer to gemma4's own default = on), and
  forwards it to **both** tiers. The remote voice worker uses `build_llms` too,
  so it inherits the fix.
- `config.json`: `llm.think=false` + a `_think_comment`. Survives the
  device-profile merge (sibling of the per-profile `main_model`/`options`).

**After — measured through the production factory** (`build_llms` over the merged
`desktop_gpu_4090` config), warmed:

| | content first-token | first spoken sentence | full answer |
|---|---|---|---|
| gemma4:12b, **think=false** | **1.8 s** | **2.4 s** | 6.1 s |

→ **~7 s cut to time-to-first-audio** for a story (9.5 s → 2.4 s), and streaming
TTS now genuinely helps (first audio 2.4 s vs full answer 6.1 s). Story quality
without thinking was excellent (coherent lighthouse-keeper-and-whale story).

### Tests (all green)
- `tests/test_core_multimodal.py`: `OllamaLLM` passes `think` when set (False/True),
  omits it when `None` (generate + stream paths).
- `tests/test_device_profiles.py`: `build_llms` defaults `think` off and forwards
  it to both tiers; opt-in (`llm.think=true`) honoured; shipped config keeps
  `think` off through the device-profile merge.

### Tradeoff / how to opt back in
Thinking is off by default on the voice path because its multi-second latency is
unacceptable for a real-time turn. For a deliberate, non-real-time research tier
that wants reasoning, set `llm.think=true` (a per-tier think config would be the
follow-up if research-quality ever needs it without slowing the fast path).

## What's next (owner improvements #2/#3 — both NEED LIVE HUMAN validation)

These are **perceptual** changes (turn-taking / STT accuracy). The project's
repeated lesson (the whole barge-in saga) is that perceptual audio changes must
be live-validated; #1 was uniquely headless-measurable, these are not.

### #2 Prosody endpointing — ✅ ACTIVATED on the Windows box; needs the owner's live tuning
Stop replying on the user's mid-thought pauses.

**Owner clarification (important):** Smart Turn v3 does **not** replace SenseVoice
STT. They are different layers — SenseVoice decides *what words* were said
(transcription); Smart Turn v3 decides *when the turn is finished* (endpointing).
Enabling prosody is **not in the STT path**, so it cannot lower transcription
accuracy; if anything it *raises* effective accuracy by holding through pauses
instead of truncating the utterance. (That's why this was the safe one to start.)

The infrastructure was already built (`core/endpointing.py`
`ProsodyTurnCompletionDetector` + Smart Turn v3, `AdaptiveEndpointPolicy`; wired
in `core/engines/sherpa.py` `_build_turn_detector` / `_decide_endpoint` with
graceful fallback). **Verified ready + activated on this box:** onnxruntime
1.23.2, `pretrained_models/sherpa/smart_turn/smart-turn-v3.1-cpu.onnx` present,
the detector loads + runs (real finished human clip → **0.9134**, thin audio →
0.5), and the runtime builds `ProsodyTurnCompletionDetector` from the merged
config.

**Now set in `config.local.json`** (gitignored, per-machine):
- `sherpa.endpoint_detector = "prosody"`
- `sherpa.endpoint_prosody_model = "<abs path to smart-turn-v3.1-cpu.onnx>"`
- `sherpa.endpoint_incomplete_threshold = 0.5` (raised from 0.3 so mid-thought
  pauses — which score low — trigger the bounded HOLD; complete turns score ≥0.74
  so they're unaffected).

**Live-tune** (`python -m core --engine sherpa`, talk with mid-thought pauses):
if it still replies on a pause, raise `endpoint_incomplete_threshold` toward 0.6;
if it holds too long on finished turns, lower it; to cut post-speech latency,
lower `endpoint_min_silence_sec` (1.1 → ~0.5) once happy.
`endpoint_prosody_min_silence` (0.15) gates when the audio model runs.

**Caveat:** the model is human-audio only (flat ~0.97 on TTS), so the
synthetic-user / inject harness CANNOT validate turn-taking — validate on real
speech. **On other machines it's still OFF** (config.local.json is per-machine —
flip the same keys there).

### #3 SenseVoice agreement-guard (STT) — needs real recorded clips to tune
SenseVoice 2nd-pass is currently **disabled** (`config.local.json asr_final_backend=""`)
because it hallucinated **short** clips ('SO'→'I then.', 'WHAT'→'I.'). The proper
fix is a new `core/asr_text.py` token-agreement helper gating
`_final_transcribe` (`core/engines/sherpa.py:1098`).

**Subtlety found this session:** a naïve token-overlap guard would WRONGLY reject
the *legitimate* case the 2nd pass exists for — correcting a *garbled* streaming
final ('Ario der'→'are you there', which has **low** overlap) — while the
hallucinations to reject are the **short-clip** cases. So the discriminator must
key on **utterance length / short-clip risk**, not pure overlap. Tuning it
correctly needs real recorded clips (both hallucinations and good corrections);
it's perceptual → validate live.

### #3b Re-enable `input_gate`
`config.local.json input_gate.enabled=false` (set for direct-reply testing).
Re-enable (`true`) for normal always-on use so the assistant ignores overheard
speech.

## Next steps (pick up here)
1. **Prosody endpointing** (improvement #2): **already activated** on the Windows
   box's `config.local.json`. Just run `python -m core --engine sherpa` and
   live-tune `endpoint_incomplete_threshold` / `endpoint_min_silence_sec` by
   talking with mid-thought pauses. (Activate on other machines by flipping the
   same keys.)
2. **SenseVoice agreement-guard** (improvement #3): build `core/asr_text.py`
   length-aware guard; tune against real recorded clips.
3. Re-enable `input_gate` for always-on; (housekeeping) tame the `logs/runs`
   git churn (`SPEAKER_KEEP_RUNS` / trim committed bundles).
