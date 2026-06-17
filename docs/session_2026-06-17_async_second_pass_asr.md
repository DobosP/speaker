# Session 2026-06-17 (pt3) — async second-pass ASR (asr-tts-2) + per-tier ASR policy (asr-tts-1)

**Headline:** Landed the data-justified next-step from the live barge-in A/B: the
offline **second-pass ASR (SenseVoice) now runs on a dedicated worker thread**,
off the real-time capture loop (`asr-tts-2`). This removes the measured cause of
the white-noise TTS output + false-barge self-interrupts in `run-20260617-103630`
(the synchronous decode stalling the capture thread) **while keeping the big STT
win**. Also shipped the sibling per-tier ASR policy (`asr-tts-1`): the weakest
profile (`phone_lite`) now uses streaming-only finals + greedy decode. Headless,
fully unit-tested + a real-model smoke; **logic suite green**.

## What landed

### asr-tts-2 — async offline second pass (`core/engines/sherpa.py`)
The problem (measured, prior session): `_final_transcribe` (the ~150 ms SenseVoice
offline decode) ran **inline on the capture loop** — the one thread that also
reads the mic, updates the echo reference, and services barge-in. That stall
starved the TTS playback fill (white-noise output) and time-misaligned the
mic/echo-reference rings on resume → false-barge self-interrupts (`run-103630`).

The fix:
- New config `SherpaConfig.asr_final_async` (**default `True`**). Engages only
  when a second-pass recognizer is actually built; otherwise the path is inline
  and byte-identical to before.
- Extracted **`_finalize_and_dispatch(seg, raw_final, speech_end_ts)`** — the
  three capture-thread-hostile steps (offline decode + L1 echo-floor gate +
  speaker-ID CAM++ gate) and the single dispatch. Runs **either** inline (no 2nd
  pass / async off) **or** on the worker.
- **`_final_worker`** — single-consumer drain of a bounded (`maxsize=8`) queue, so
  finals dispatch in **capture order** even though the decode is slow.
- **`_enqueue_final`** — never blocks the capture loop. On overflow (worker wedged;
  normally the queue sits near-empty) it **drops the OLDEST** queued utterance
  (the `_play_q` idiom), preserving capture-order dispatch. **This ordering matters:**
  the runtime supersede is *newest-ARRIVAL-wins*, so a stale final arriving after
  a newer one would wrongly cancel the newer turn — drop-oldest keeps arrival order
  == capture order.
- Lifecycle: `start()` spawns the worker (daemon) when async+recognizer; `stop()`
  sentinels + joins it (1.0 s), same pattern as capture/playback.

**Why not "stream the streaming-final first, upgrade in place"** (the roadmap's
original parenthetical): verified the runtime is **one-final-per-utterance** and
its supersede is a *cancel*, not an upgrade. And the live data
(`'WHOLE HOLLO' → 'Helen, how are you.'`) shows the streaming final is exactly the
text we must NOT send the LLM — dispatching it first would speak garbage, then
supersede it. So we run the 2nd pass off-thread and dispatch the single upgraded
final. Same final latency as the old synchronous path, minus the real-time
disruption.

### asr-tts-1 — per-tier ASR policy (`config.json`)
Confirmed gap: **no** device profile overrode the ASR policy — every tier
(incl. `phone_lite`) inherited the desktop `modified_beam_search` + `sense_voice`.
`phone_lite.sherpa` now sets `asr_final_backend=""` (streaming-only) +
`asr_decoding_method="greedy_search"`. Desktop/4090 keep beam + SenseVoice
(base defaults, asserted by test). Scoped to `phone_lite` per the roadmap's
explicit guidance; `phone`/`cpu_laptop` deferred to the Phase-0 low-spec latency
measurement (not yet done).

## Tests (Tier-0 unless noted)
- `tests/test_asr_final_async.py` (NEW): config default; `_finalize_and_dispatch`
  upgrade + backdated SPEECH_END + both drop paths (floor / speaker); worker
  ordering off-thread; worker survives a per-turn exception; `_enqueue_final`
  drop-oldest preserves capture order; no-drop when room; clean exit on
  `_running.clear()`. Plus a **`real_model`** smoke: the REAL SenseVoice model
  decodes the committed `recorded_ocean_utterance.npy` fixture **on the worker
  thread**, asserting exactly one terminal outcome off-thread (ran + passed here;
  self-skips in CI).
- `tests/test_device_profiles.py` (+2): `phone_lite` → streaming-only/greedy;
  desktop/4090 keep beam+SenseVoice.

## Validation
- Logic suite: GREEN — **1923 passed, 25 skipped** (`.venv/bin/python -m pytest tests -q`;
  +16 over the pt2 baseline of 1907).
- **Adversarial multi-lens review** (concurrency / runtime-contract / tests, each
  finding independently verified) → **SHIP-WITH-NITS**: "no correctness or
  concurrency defects; the design (one-final-per-utterance, capture-order dispatch,
  drop-oldest, daemon teardown) is verified sound." Addressed the two cheap
  follow-ups it surfaced: the **async gate** is now an extracted, unit-tested
  helper (`_maybe_setup_async_final`, all three worker-vs-inline cases), and the
  **overflow drop is now instrumented** (`second_pass_queue_overflow_dropped_final`
  metric) so a wedged worker isn't blind in the run bundle. Documented the
  intentional shutdown drop. Deferred nits: full start()/stop() lifecycle test
  (needs a real device, matches the existing capture/play coverage bar) and
  snapshotting the EWMA floor at enqueue (advisory, no torn read).
- **NOT yet live-validated** (needs the mic): the async path only runs under
  `--engine sherpa`; `--engine replay` uses `FileReplayEngine` (streaming-only).

## Next steps (pick up here)
1. **LIVE-VALIDATE asr-tts-2 (owner, the mic).** On `config.local.json` (machine-
   local) re-enable SenseVoice — set `sherpa.asr_final_backend="sense_voice"`
   (it's currently `""`, the prior clean baseline) and leave `asr_final_async`
   at its default `true`. Run `./session.sh --llm echo`, talk-over the assistant
   on the **open laptop speaker**, and confirm vs `run-103630`: the white-noise
   output + self-interrupts are GONE while the STT win (`SenseVoice` corrected
   finals) remains. Watch the run bundle for the `second-pass final ASR runs ASYNC`
   log line + `echo_floor_rejected_final`/`speaker_rejected_final` metrics.
2. **Open-speaker echo:** the **DTLN deep-AEC tier** (`audio-bargein-7`) — needs
   the converted ONNX + live validation (NLMS@260ms already ruled out).
3. **Short-talk-over sensitivity (coherence path):** a controlled
   `coherence_margin_delta`/`confirm_frames` reduction — live-only.
4. **Owner:** re-enroll speaker-ID in the normal speaking position, or lower
   `sherpa.speaker_threshold` ~0.4, then flip `speaker_gate_input` back on.
5. **asr-tts-1 follow-on:** once the Phase-0 low-spec latency scorecard exists,
   decide the ASR policy for `phone`/`cpu_laptop` (greedy? keep SenseVoice?).

## Environment (i9-13980HX, `.venv` ACTIVATED in the owner's shell)
Anything touching models/audio/LLM must use `.venv/bin/python` (a fresh shell's
`python3` is bare). Models present (sherpa ASR/VAD/TTS + CAM++ speaker-ID +
SenseVoice, ~600 MB); Ollama up; `doctor=READY`. `python` (no `3`) is NOT on PATH
in a fresh shell — use `.venv/bin/python`.
