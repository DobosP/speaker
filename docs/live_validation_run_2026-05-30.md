# Live validation run — 2026-05-30

First real on-hardware run of the live-validation harness (`tools/live_session`),
executed by Claude on the dev machine against the **real** models
(`gemma3:4b` on Ollama for both tiers; sherpa streaming-zipformer ASR + LibriTTS
TTS + silero VAD). All 7 scenarios ran end-to-end. This is the analysis.

## TL;DR

- **The system works end-to-end on real models.** STT → input/addressing gate →
  context aggregation → LLM → streaming TTS produced **correct, in-order,
  attributed answers** across all scenarios, with **bounded latency and no
  hangs**.
- **The latency bottleneck is endpointing (silence detection), not the brain.**
  First-audio is ~1.5–1.8 s, of which **~1.1–1.3 s is the trailing-silence wait**
  before the ASR finalizes. The LLM is a rock-solid **~200 ms** time-to-first-token
  (pre-warm is paying off) and TTS first-chunk is ~100–300 ms. → **The smart
  endpoint (currently default-OFF) is the single highest-value latency win.**
- **Capabilities validated live:** context aggregation (`"its population"` → Paris),
  anti-confabulation (declines web search when it's off), never-stuck (heavy
  research-style turns answered without a stall or timeout apology).
- **Running it caught 8 real bugs** (all fixed this session — see below).

## How it was run — and why `--inject`

Over-the-air (synthetic user plays through the speakers; the assistant hears it
through the mic) was tried first and **does not work reliably on this machine**:

- The built-in laptop **speaker→mic acoustic path garbles STT** ("What's the" →
  "Was"/"While"/"Once", "a week" dropped) and, with the input gate off, the mic
  re-hears the assistant's own TTS → **feedback loop**.
- Output and input were on **different codecs** (HDMI vs analog ALC285) so the mic
  couldn't even hear the synthetic user until playback was forced to the analog
  device.

So the reliable path on this hardware is **direct injection** (`--inject`): the
synthetic-user audio is fed **straight into the recognizer** (mic never opened),
and the assistant's TTS goes to a **null sink** (we read latency from metrics and
re-synthesize the assistant track for the artifact). This tests the **real**
STT→LLM→TTS pipeline and the brain on **clean** audio, with no acoustic
degradation, no feedback, and no flaky-device crashes. Over-the-air remains the
"is the literal mic/speaker wired up" smoke test — it needs better audio hardware
(external mic/speaker, or AEC) to be a capability test.

## Latency (the headline)

Per-turn first-audio (SPEECH_END → first assistant audio) and its breakdown, ms:

| scenario            | first_audio (median) | endpoint (silence) | LLM TTFT | TTS first-chunk |
|---------------------|---------------------:|-------------------:|---------:|----------------:|
| baseline            | 1522 | 1142–1276 | 186–217 | 109–115 |
| context_aggregation | 1620 | 1135–1280 | 186–200 | 114–228 |
| addon_continuation  | 1704 | 1138–1265 | 184–197 | 304–320 |
| self_awareness      | 1732 |  832–1350 | 187–204 | 251–696 |
| smart_endpoint      | 1570 | 1256–1271 | 197–222 |  82–112 |
| barge_in            | 1618 | 1243–1279 | 202–211 | 126–173 |
| never_stuck         | 1788 | 1157–1492 | 192–204 |  83–810 |

**Reading it:** first-audio ≈ **endpoint + LLM-TTFT + TTS-first-chunk**. The
endpoint term is **~75 % of the budget** and is the sherpa recognizer's
trailing-silence rule — independent of the LLM. The LLM tier is consistently
~200 ms (warm start working as designed). TTS first-chunk is usually ~100–300 ms,
with outliers to ~600–810 ms on long first sentences (heavy turns).

**Actionable:** the brain/LLM is **not** the latency problem. Cutting the
~1.2 s silence wait (smart/semantic endpoint, unit #1, today default-OFF) is the
win; after that, TTS first-chunk on long sentences is the secondary lever.

## Capability validation (per scenario)

- **baseline** — ✅ PASS. "Paris", "seven days." Clean STT, correct ACT, correct
  answers, clean attribution.
- **context_aggregation** — ✅ PASS (standout). `"And what's its population?"` →
  *"The population of **Paris** is approximately 2.1 million"* — **"its" resolved to
  Paris from the prior turn.** Short-term context aggregation works on real models.
  (Turn 3 `"which of those two cities is older"` had a shaky premise — only one
  city had been named — and the model gracefully invented a comparison; minor.)
- **self_awareness** — ◑ PARTIAL. Correctly **declined** the unavailable web skill
  (*"I don't have the ability to access the internet"*) instead of confabulating —
  the anti-confabulation rule holds. But the "what can you do" answer was **generic**
  ("chat, answer questions, make up a story") rather than enumerating the actual
  capability catalog (research/notes). The persona's skill enumeration isn't fully
  surfacing in the spoken answer — follow-up.
- **never_stuck** — ✅ PASS. Three heavy research-style turns (1929 crash → 3 causes;
  compare Japan/Germany economies) **each answered promptly, no stall, no timeout
  apology, no stuck task.** first-audio stayed bounded (≤2.3 s even on heavy turns).
  One *wrong* answer: `"what time of day works best for a short walk"` → *"It's
  4:52 PM"* (answered the clock, not the question) — a model/tool misfire, not a
  stuck-controller failure.
- **addon_continuation** — ◑ PARTIAL. The two close-spaced lines
  ("…in Paris today?" + "And also tomorrow.") **merged into one ASR utterance** and
  were answered once — the merge behavior, but driven by ASR endpointing rather than
  the continuation classifier. Injection timing makes after-audio continuation hard
  to exercise deterministically; revisit with explicit inter-line gaps.
- **smart_endpoint** — ✅ ran on the **default** endpoint (feature is default-OFF);
  endpoint ≈1.26 s confirms the acoustic timer is what the smart endpoint must beat.
- **barge_in** — ⚠️ NOT exercised by inject (expected): barge-in needs live audio
  *during* playback, and the mic is bypassed in inject mode (also `barge_in_enabled`
  is off until AEC). Needs the over-the-air path with real AEC hardware.

## STT on the synthetic voice

Even on clean injected audio the zipformer makes consistent **minor** errors on the
LibriTTS synthetic voice: "like"→"lake", "capital"→"apital", "Research"→"Researched",
"Tell me"→"Come in", "online"→"on line". The LLM absorbed most of them (still
answered correctly). This is ASR accuracy on a synthetic speaker, not a pipeline
bug — but it means the synthetic-user voice is an imperfect proxy for a human, and
the first word is the most fragile (mitigated, not eliminated, by a noise-floor
lead-in before each utterance).

## Bugs found + fixed by running it

The act of running the harness on real hardware surfaced 8 real defects, all fixed:

1. **Synthetic-user playback rate** — played at the TTS native 22050 Hz to a device
   that only opens at 48000 → `PortAudioError`. Resample playback to the device rate
   with fallbacks.
2. **Device index as string** — `--output-device 4` was passed as `"4"` and
   PortAudio read it as a device *name*. Normalize via the engine's `_norm_device`.
3. **Driver premature-return** — the "turn started" latch fired on a transcript
   appearing (which also happens for an INGEST'd turn, *before* a task exists), so
   the capture could advance before the assistant picked up. Latch on real
   engagement (task/speech).
4. **`'And'` hallucination on digital silence** — injecting perfect zeros makes the
   streaming zipformer hallucinate filler words every endpoint cycle. Feed a
   realistic low-level **noise floor** between utterances (a real mic never delivers
   pure zeros).
5. **`is_speaking`-stuck → 45 s gaps (the big one)** — the idle detector treated
   "has spoken since this line" as a *level*; since `spoken_count` only grows, it
   kept the activity timer fresh forever and the idle-settle never elapsed → every
   turn ran to the response-timeout. Fixed to an **edge** (new sentence only).
6. **First-word garbling on injected audio** — no pre-speech pause made the
   recognizer mis-decode the onset. Prepend a short noise-floor lead-in.
7. **Assistant TTS playback blocks on this box** — ALSA `out.write()` stalls for
   tens of seconds on a short clip. In inject mode route TTS to a **null sink**
   (metrics still fire).
8. **Flaky mic open crashed multi-scenario runs** — rapid per-scenario ALSA mic
   open/close is unstable here. Inject mode now **patches `sd.InputStream`** so the
   real mic is **never opened** — fixing both the crash and the cross-scenario
   "Device unavailable".

## Recommendations

1. **Validate + enable the smart endpoint.** It targets the dominant ~1.2 s cost.
   This run is the calibration baseline (acoustic endpoint ≈1.14 s floor).
2. **Surface the capability catalog in the spoken self-description** (self_awareness
   was generic). The decline path already works; the enumerate path needs the
   persona skills threaded into the answer.
3. **Barge-in + true over-the-air need real audio hardware** (external mic/speaker
   + AEC). Inject mode can't test interruption-during-playback.
4. **The synthetic voice is an imperfect ASR proxy.** For STT-accuracy claims, use
   human-recorded fixtures; the synthetic voice is fine for latency + brain logic.

Artifacts: `logs/live/20260530-134755/<scenario>/{summary.md,timeline.json,latency.json,user/,assistant/}`.
Reproduce: `python -m tools.live_session --all --model gemma3:4b --fast-model gemma3:4b --inject`.
