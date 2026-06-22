# Voice upgrade: natural + emotional + diverse, still cheap

**Status:** partially IMPLEMENTED (see status box) · **Date:** 2026-06-22 · **Scope:** desktop `core/` runtime (sherpa-onnx engine); mobile follows once validated
**Owner ask (verbatim):** the TTS is *"blurry, robotic, interrupted, not clear — the main issue"*; wants a **better voice + emotion + diversity, still cheap on-device.**

This doc is built entirely from one run's diagnosis (`run-20260622-093456` + a full read of the synthesis/playout chain). All file/line references were re-verified against the working tree.

---

## Implementation status — 2026-06-22 (what landed)

A fan-out investigation (4 read-only agents + synthesis) re-verified this diagnosis against the tree, and a **direct synthesis probe settled the open question**: Kokoro is **not** broken (the `run-20260622-100328` "empty `.ref.wav`" was an *idle* session — `finals=0`, so the assistant never spoke). Live Kokoro synth measures **centroid 1763–2880 Hz / HF-ratio 0.16–0.27** across voices vs the old libritts **~800 Hz / ~0.0026** — objectively far brighter/clearer. 103 voices, all real; the speed knob works (0.9→2.84 s, 1.15→2.27 s).

**Shipped (branch `feat/kokoro-tts-backend`):**
- **Problem A (robotic/blurry) — Kokoro model swap.** Already wired (`build_tts` keys Kokoro on `tts_voices`; commit `2b928b8`) + verified bright/clear above.
- **Emotion + voice diversity (the headline ask).** New `core/tts_markup.py` (pure `parse_tts_markup` + `resolve_tts_params` + `build_markup_guidance`). Opt-in `SherpaConfig.tts_markup`: the LLM prefixes a sentence with `[emotion:.. voice:.. rate:..]`; `speak()` strips it and maps to a **per-utterance (speaker_id, speed)** — voice-choice for timbre diversity, rate-as-affect for emotion (the only expressivity sherpa-onnx Kokoro exposes — no latent style vector). `runtime.py` teaches the answering model the grammar + its configured voices/emotions when the flag is on. **Default OFF → byte-identical.** Commit `e503538`; 26 tests.
- **Problem B (interrupted) — partial, safe mitigation.** `playback_fifo_sec` raised 1.0→1.5 on capable profiles (desktop/4090/macbook) so the whole-clip producer has run-ahead headroom under CPU contention; phone/weak keep 1.0. Commit `c60d11b`. NOTE: the box's earlier "interrupted" was dominated by **environment artifacts** (a dead-quiet mic left by `barge_stress`, + heavy external CPU load) per prior diagnosis — not assistant logic.

**Deliberately deferred (the real first-audio-latency fix):** a **streaming-compatible output leveler** (per-chunk feed-forward gain seeded from the carried `_tts_level_gain_db`, AGC2 is inherently streaming) to reclaim low first-audio latency *without* losing the stable open-speaker echo floor the whole-clip leveling provides. NOT landed because it must be **validated live at the mic against the barge-in P1** (echo-floor stability) on a quiet box — which the current CPU load + idle-mic sessions don't allow. This is the next step, gated on a clean live A/B.

**This machine (`config.local.json`, gitignored):** Kokoro + `tts_markup:true` + a provisional named-voice set `{warm:16, bright:28, deep:9, narrator:26, soft:3, lively:18}` + emotion→speed map; audition `logs/kokoro_voice_audition.wav` (104 s) for the owner to pick/rename voices by ear.

---

## 1. Diagnosis recap — there are TWO independent problems

The owner hears one bad voice, but it is two unrelated faults. Fixing only one will not satisfy the ask.

### Problem A — "robotic / blurry / not clear" = the model itself
The muffled, robotic timbre is **intrinsic to `en_US-libritts_r-medium`**, a basic 22050 Hz VITS. It is **not** pipeline damage. Verified by reading the whole output chain:

- `declick` is a documented no-op on clean speech (median-based isolated-impulse repair only).
- `output_leveler` / `normalize_rms` are scalar-gain + soft-knee true-peak limiter — **no spectral filtering**, length-preserving.
- **No output resampling** happens: the run log shows the device opened at **22050 Hz = the TTS native rate**, so the lossy `_resample_linear` is bypassed.

Conclusion: the muffled spectrum *is the model's own voice*. **The fix is to swap the model**, not to tune the front-end. (Confirmed: `build_tts` at `core/engines/_sherpa_models.py:213-216` only ever fills `tts_config.model.vits.*`; `config.local.json:8-10` points it at the single libritts VITS — the repo has exactly one voice today.)

### Problem B — "interrupted / choppy" = playout underrun churn (the leveler forces whole-clip synth)
The choppiness is **PlaybackFIFO underrun churn**, caused by the live config forcing the **non-streaming (whole-clip) synth path**, with no adaptive buffer and no real-time priority on the audio callback. Chain of evidence:

1. **The live config forces whole-clip synth.** `config.local.json` (deep-merged *over* `config.json`, local wins — `core/config.py`) sets `"tts_output_leveler": true` (`config.local.json:23`). In `_synthesize()` the **streaming callback** branch is taken **only** when `target_rms <= 0.0 and not leveler_on and self._tts_can_stream` (`core/engines/sherpa.py:2729`). With the leveler on, that branch is skipped and execution falls through to a **blocking whole-sentence** `audio = tts.generate(text, sid, speed)` (`:2750`) → declick (`:2761`) → `output_leveler` over the whole clip (`:2766`) → a `0.1s` chunk loop (`:2789-2793`) that only *then* feeds the FIFO. **No audio reaches PlaybackFIFO until the entire sentence is synthesized + leveled** — which both raises first-audio latency (the watchdog fired: *"tts stuck: turn 2 had llm_first_token but no tts_first_audio after 3.7s"*) and removes the intra-sentence pipelining that keeps the FIFO fed.

2. **No real-time priority on the audio path.** `sd.OutputStream(... latency="low", callback=self._audio_cb)` opens with **no blocksize and no RT scheduling** (`core/engines/sherpa.py:2575-2578`; a grep for `SCHED_FIFO`/`setpriority`/`nice` across the engine finds nothing). Under external CPU contention (the run had tesseract + a scraper, loadavg ~7) the whole-clip producer (on `tts_num_threads=2`) lands behind real-time and/or the callback is preempted → the FIFO drains mid-block → `_audio_cb.read_into()` zero-fills the underrun tail (`:226`) → audible gaps. This is **already counted**: `_audio_cb` increments `self._underrun_blocks` when `0 < n < frames and _speaking` (`:2419-2420`), surfaced as the `playback_underrun` metric + a WARNING at `:2638-2644`.

3. **The FIFO depth is fixed.** `SherpaConfig.playback_fifo_sec` defaults to `1.0` (`:676`) and is overridden by **no** device profile → a static 1.0s buffer that cannot grow under load. The `0.1s` chunking is benign (the FIFO de-couples chunk size from playout); the real serialization is the **whole-clip producer**.

> Both problems are real and separable. **Model swap fixes "robotic/blurry."** **Playout fix (starting with un-forcing whole-clip synth) fixes "interrupted/choppy."**

---

## 2. Recommended voice — Kokoro-82M (int8) via sherpa-onnx, Piper-high as the safe fallback

### Primary recommendation: **Kokoro `kokoro-int8-multi-lang-v1_1`**
| Axis | Kokoro-82M (int8 v1_1) | Why it wins for this ask |
|---|---|---|
| **Naturalness** | StyleTTS2-based; topped the TTS Spaces Arena despite only 82M params; "remarkably natural English speech" | Clear, audible jump in clarity/prosody over libritts VITS — directly answers "robotic/blurry." Output is **24000 Hz** (vs 22050) — slightly higher fidelity; the engine already reads `audio.sample_rate` (`sherpa.py:2752`) so the FIFO/output path adapts automatically. |
| **Voice diversity** | **103 voices** (EN+ZH), selected via the existing `sid` integer — no API change | Today the repo has **exactly one** voice. This is dozens-of-voices vs one — the headline win for the "diversity" ask. |
| **Emotion** | **No** explicit emotion/style API in sherpa-onnx (only `sid` + `speed`/`length_scale`) | Partial: "emotion" comes from picking an expressive voice + adjusting rate (see §3). True per-utterance affect control is **not** available — flagged as a caveat. |
| **sherpa-onnx fit** | **First-class.** `OfflineTtsKokoroModelConfig` is a sibling of `vits`/`matcha`. **Confirmed live in this repo:** `.venv/bin/python` reports `sherpa_onnx 1.13.3` with `hasattr(sherpa_onnx,'OfflineTtsKokoroModelConfig') == True`. No new runtime — same `OfflineTts` object, same `generate(text, sid=, speed=, callback=)`. | Drop-in via the existing seam. |
| **CPU cost** | int8 package keeps RTF well under 1.0 on an x86 laptop; streaming callback bounds perceived latency regardless | **Caution:** the only published CPU RTF is Raspberry Pi 4 — **RTF > 1** for the fp32 packages (3.19 / 2.77). Use **int8**, not fp32, and **bench on the actual laptop** before committing (no x86 RTF published). |
| **License** | **Apache-2.0** (upstream model card) | Permissive, commercial-friendly, fine for this local-first single-user app. (`espeak-ng-data` phonemizer is GPL, but that is bundled-data invoked the *same way as today's VITS* at `config.local.json:10` — **no new license posture**.) |

**Exact package:** `kokoro-int8-multi-lang-v1_1.tar.bz2` from [k2-fsa/sherpa-onnx releases (tts-models)](https://github.com/k2-fsa/sherpa-onnx/releases) — see the [Kokoro pretrained-models page](https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/kokoro.html). The archive bundles `model.onnx`, `voices.bin`, `tokens.txt`, `espeak-ng-data/`, `lexicon-*.txt`, and a `LICENSE`. 103 voices, 24 kHz. **English-only lighter fallback:** `kokoro-en-v0_19` (model.onnx 330 MB + voices.bin 5.5 MB, 11 EN voices, sid 0-10). Note fp32 packages are **310-330 MB on disk** — much larger than today's ~75 MB libritts VITS; this is the reason to prefer int8 v1_1 for the on-device/mobile constraint.

### Safe fallback (zero new family): **Piper-high VITS** — `en_US-lessac-high` / `en_US-libritts-high` / `en_US-ryan-high`
If Kokoro int8 turns out CPU-tight on a weak tier, the **lowest-risk** upgrade is a higher-quality voice *in the family the repo already runs*. Piper "high" tier (22.05 kHz, ~2× params) is smoother than medium ("sustained vowels smoother, VITS shimmer reduced"; lessac-high cited as smoothest), and stays comfortably real-time on a laptop — Piper-medium is ~0.36 RTF on 4 weak ARM cores; high is ~1.5-2× that, still < 1.0 on a laptop (**measure on the phone tier before shipping high there**). It is a **zero-code** swap (just repoint `tts_model`/`tts_tokens`/`tts_data_dir`). Ceiling is "clean/clear/consistent," **not** "expressive" — it does not satisfy diversity (one speaker) or emotion, so it is the *fallback*, not the recommendation. **Avoid VCTK/LJSpeech** non-Piper VITS on weak tiers (RTF ~6 @1 thread / 2.2 @4 on RPi4 — ~8× Piper-medium) despite VCTK's 109 speakers.

### The swap — exact files/keys to change
There is **one swap point with two flavors**:

**(1) Zero-code (Piper-high fallback):** repoint `config.local.json` `tts_model`/`tts_tokens`/`tts_data_dir` at the extracted Piper-high package. Done. Multi-speaker diversity, where the model has it, is already wired via `tts_speaker_id → generate(sid=...)`.

**(2) New family (Kokoro) — two small, isolated edits, no new dependency:**

- **`core/engines/sherpa.py` `SherpaConfig`** (near `tts_model`/`tts_tokens`/`tts_data_dir` at `:360-363`): add
  ```python
  tts_voices: str = ""    # path to voices.bin -> presence selects Kokoro
  tts_lexicon: str = ""   # needed for the multi-lang packages
  ```
- **`core/engines/_sherpa_models.py` `build_tts`** (`:206-219`): branch on `tts_voices`:
  ```python
  tts_config = sherpa_onnx.OfflineTtsConfig()
  if c.tts_voices:  # Kokoro
      k = tts_config.model.kokoro
      k.model    = c.tts_model
      k.voices   = c.tts_voices
      k.tokens   = c.tts_tokens
      k.data_dir = c.tts_data_dir          # espeak-ng-data
      if c.tts_lexicon:                    # multi-lang packages
          k.lexicon = c.tts_lexicon
  else:  # existing VITS path (UNCHANGED)
      tts_config.model.vits.model  = c.tts_model
      tts_config.model.vits.tokens = c.tts_tokens
      if c.tts_data_dir:
          tts_config.model.vits.data_dir = c.tts_data_dir
  tts_config.model.num_threads = c.resolved_tts_threads
  tts_config.model.provider    = c.provider
  return sherpa_onnx.OfflineTts(tts_config)
  ```
- **`config.local.json`**: point `tts_model`/`tts_tokens`/`tts_data_dir` at the extracted Kokoro package and add `tts_voices` (→ `voices.bin`) + `tts_lexicon`.
- **`tools/setup_models.py`**: add a Kokoro fetch entry. The file already fetches sherpa `.tar.bz2` release assets via `urllib` and extracts members (`fetch_speaker_model`/`extract_member`, `:117`/`:140`); add a `KOKORO_MODEL_URL` constant pointing at `kokoro-int8-multi-lang-v1_1.tar.bz2`, extract `model.onnx` + `voices.bin` + `tokens.txt` + `espeak-ng-data/` + `lexicon-*.txt`, and `wire_sherpa_paths` (`:201`) the four config keys.

**Everything downstream is unchanged** — synthesis still calls `tts.generate(text, sid=c.tts_speaker_id, speed=c.tts_speed, callback=...)` (`sherpa.py:2745`), voice selection is just `tts_speaker_id`, and sample rate auto-adapts (`audio.sample_rate`, `sherpa.py:2752`). The VITS seam stays intact as the fallback.

---

## 3. Emotion + diversity plan — cheap, on-device, no extra model

The reality of sherpa-onnx TTS: the only levers are **`sid` (voice)** and **`speed`/`length_scale` (pace)**. There is **no per-utterance affect vector** exposed. So we synthesize "emotion + diversity" from those two cheap knobs.

### Diversity (free, already wired)
Multiple voices come **for free** with Kokoro: `tts_speaker_id` selects among 11 / 53 / 103 voices via the existing `generate(sid=...)` call. Wire a small **named-voice → sid map** in config (e.g. `"voices": {"default": 9, "warm_female": 0, "narrator": 2}`) so the runtime/persona can pick a voice by name instead of a raw integer. Cost: a dict lookup. This alone is "dozens of voices vs one."

### Emotion (cheap approximation — the only realistic on-device path)
Since there is no affect API, approximate emotion with **an LLM-emitted per-utterance style tag** that the engine maps to the two available knobs:

- The fast LLM already produces the reply text. Have it prefix (or emit alongside) a tiny tag, e.g. `[voice=warm_female rate=1.05 emph=calm]`, drawn from a **small closed vocabulary**.
- A new mapping step (cleanest seam: just before `_synthesize` sets `sid`/`speed` at `sherpa.py:2717-2718`) parses the tag and maps it to **`tts_speaker_id` (which voice) + `tts_speed` (pace)**. Strip the tag from the spoken text.
- "Emotion" is then expressed by **(a) choosing an expressive voice** appropriate to the mood and **(b) nudging rate** (e.g. faster for excited, slower for calm). This is genuinely cheap: no second model, no extra synth pass.

**What is realistic vs too expensive on-device:**
- **Realistic / cheap:** voice-by-name diversity (free), rate-as-affect, LLM style-tag → (sid, speed) mapping. Per-sentence voice/rate switching is fine because each sentence is its own `generate()` call.
- **Too expensive / not available:** true latent style-vector control (Kokoro-82M *has* a style vector internally, but **sherpa-onnx does not surface it** — would need a custom ONNX export + a new inference path, out of scope and not cheap). Prosody-transfer / reference-audio emotion, and a separate emotion-classifier model, are all over the on-device budget.

> **Caveat to surface to the owner:** if *true affect control* (not voice+rate) is a hard requirement, **no sherpa-onnx TTS family delivers it today** — Kokoro included. The style-tag-→-(voice,speed) approach is the honest cheap ceiling.

---

## 4. Interruption fix — stop forcing whole-clip synth, then make playout robust under load

Fix in priority order; (a) alone removes the watchdog stall and most choppiness.

**(a) PRIMARY — stop forcing the whole-clip path.** The leveler needs the whole clip only for its loudness measure. Either:
- **Quick:** set `"tts_output_leveler": false` in `config.local.json:23`, so `_synthesize` takes the **streaming-callback branch** (`sherpa.py:2729`) — the FIFO is fed as audio is produced, restoring pipelined first-audio + intra-sentence feeding. Loudness falls back to the scalar `normalize_rms` path the streaming branch already allows. *This is a one-line config change and the single biggest anti-choppiness lever.*
- **Durable:** make the leveler **streaming-compatible** (per-chunk gain with carried state, like the existing `self._tts_level_gain_db` slew at `:2766-2773`) so we keep loudness *and* streaming. Then the leveler no longer forces the blocking path.

**(b) Adaptive playout buffer.** `playback_fifo_sec` is a static `1.0` (`sherpa.py:676`) overridden by no profile. Grow it / add pre-roll when the **existing `playback_underrun` metric** (`:2419-2420`, surfaced `:2638-2644`) rises — a contended box trades a little latency for gap-free output. The metric already exists; this is wiring a feedback loop, not new instrumentation.

**(c) Real-time scheduling + pinned blocksize on the output callback.** The `OutputStream` opens at default priority with no blocksize (`:2575-2578`). Add `SCHED_FIFO` (or `setpriority`) on the PortAudio callback thread and a pinned `blocksize`, so it is not preempted at loadavg ~7. Guard for permission failure (RT may be denied without rtprio limits) and degrade gracefully.

**(d) Weak/contended tier tuning.** Dial `tts_num_threads` up, or pre-roll more of the reply before starting playout, on a weak/contended profile.

---

## 5. Prioritized plan — biggest perceived jump first

| # | Step | Effort | Exact seam | Cheap? |
|---|---|---|---|---|
| **1** | **Un-force whole-clip synth** — flip `tts_output_leveler` off (validate loudness via `normalize_rms`) | **~1 line + a listen** | `config.local.json:23`; branch `sherpa.py:2729` | Free — removes the stall + most choppiness immediately |
| **2** | **Swap the voice → Kokoro int8 v1_1** (biggest perceived jump; fixes "robotic/blurry") | **S** (2 edits + fetch entry + config + bench) | `_sherpa_models.py:206-219`, `SherpaConfig` `:360`, `config.local.json:8-10`, `tools/setup_models.py` | Cheap *if int8* + benched < 1.0 RTF; fp32 is NOT cheap (avoid) |
| **3** | **Voice diversity** — named-voice → `sid` map | **XS** | config dict + lookup before `sherpa.py:2717` | Free (dict lookup) |
| **4** | **Emotion approximation** — LLM style-tag → (sid, speed) mapper | **S** | parse/map just before `sherpa.py:2717-2718`; strip tag from spoken text | Cheap (no extra model) |
| **5** | **Streaming-compatible leveler** (keep loudness *and* streaming) | **M** | per-chunk leveler reusing `self._tts_level_gain_db` slew (`:2766`) | Cheap CPU; moderate code |
| **6** | **Adaptive FIFO + pre-roll** under load | **M** | grow `playback_fifo_sec` (`:676`) on `playback_underrun` rise | Cheap |
| **7** | **RT priority + pinned blocksize** on the audio callback | **M** (perms-guarded) | `sherpa.py:2575-2578` | Cheap |

**What stays cheap:** steps 1, 3, 4 (config + tiny logic), and Kokoro **int8** at runtime. **What would be too much:** fp32 Kokoro packages (310-330 MB, RTF > 1 on a Pi); true latent-affect control (no sherpa surface); any second model for emotion or prosody transfer. Order rationale: step 1 is nearly free and kills the stutter; step 2 is the **biggest perceived quality jump** for the owner's main complaint; 3-4 add diversity+emotion cheaply; 5-7 harden playout on old/contended hardware.

> **Side-look (only if CPU headroom is tight after benching):** **Matcha-TTS** (sherpa-onnx, very low RTF) and **KittenTTS** (a community CPU benchmark found it faster/smaller than Kokoro). Neither matches Kokoro's voice count + naturalness, so they are contingency, not plan-of-record.

---

## 6. Validation — A/B with the existing record/replay rig

1. **Offline synth-and-listen (candidate voices, before wiring).** Synthesize a fixed sentence set across candidate `sid`s with each package (`tts.generate(text, sid, speed)`) and listen — pick the 2-4 voices to expose, and confirm the "blurry/robotic" complaint is gone on Kokoro vs the libritts baseline. Compare 22050 Hz libritts vs 24000 Hz Kokoro on the **same** sentences.
2. **Benchmark RTF on the actual laptop** with `python -m tools.bench` **before committing** — Kokoro int8 must land RTF well under 1.0 (no x86 figure is published upstream; the only k2-fsa numbers are RPi4 fp32 with RTF > 1). Also bench the **phone** profile if Kokoro is destined for mobile.
3. **A/B the interruption fix with a record bundle.** Run `./session.sh --record` under deliberate CPU contention (reproduce loadavg ~7) on (i) leveler-on/whole-clip baseline and (ii) leveler-off/streaming. The committable bundle (`logs/runs/run-<id>.{txt,summary.json,wav}`) carries the **`playback_underrun` metric** and the watchdog `stuck_hints` — compare underrun counts and `speech_end → tts_first_audio` between the two. Target: zero "tts stuck" watchdog fires, underrun blocks → 0 even under contention.
4. **Lock it as a regression.** The recorded WAV replays headless via `python -m core --engine replay --replay-dir logs/runs` (no sound card) — keep a baseline bundle so a future change that re-introduces whole-clip serialization or the muffled voice is caught.

**Done = ** the owner hears a clearly more natural voice (Kokoro), can be given several distinct voices + mood-driven rate/voice changes, and playback no longer stutters under load even on old hardware.