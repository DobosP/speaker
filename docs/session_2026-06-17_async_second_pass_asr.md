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

## Wave 2 — additional headless items (no live test needed)

Scouted the Phase-1/3 roadmap before touching code; several items were already
done (device-adapt-1 / cross-platform-8 auto-profile + fail-fast are fully wired
in `core/app.py` + `remote/worker.py`; the cloud/gate invariant test already
exists; the `config.local.json` write-allowlist is moot — no risky writer). The
genuinely-open, confirmed, headless items landed:

- **llm-inference-3 (on-device output cap):** `LlamaCppLLM` fed its options
  straight to llama.cpp's `create_chat_completion`, but that API's output cap is
  `max_tokens` — Ollama's `num_predict` was silently ignored and `num_ctx` isn't a
  request param, so on-device generation ran to the context limit on the weakest
  CPU. New `_normalize_llamacpp_options` translates `num_predict`→`max_tokens`
  (explicit `max_tokens` wins) and drops `num_ctx`/`keep_alive`. `phone`/`phone_lite`
  now set an auditable `options.num_predict` (384/256). Desktop Ollama path
  unchanged. Tests: `tests/test_llamacpp_options.py` + a profile-cap assertion.
- **rc-5 (watchdog false "stuck"):** a turn preempted by newest-input-wins had
  `asr_final` but no `llm_first_token` and no skip-stamp → false "llm stuck" in
  every bundle (the `stuck:1` noise). NB the held/merged theory was wrong —
  `ASR_FINAL` is stamped at *dispatch* (post-hold). New `SUPERSEDED` metric +
  `MetricsRecorder.mark_superseded_turn()` (stamps `_completed[-1]`, the turn the
  new final's `ASR_FINAL` just banked) called right after `cancel_all()`; the
  watchdog skips `SUPERSEDED` turns. Tests in `test_metrics.py` + `test_watchdog.py`.
- **Invariant hardening:** the per-profile guardrail now also forbids a profile
  disabling `sherpa.speaker_gate_input` or `agent_brain.require_owner_verified`.

Suite after Wave 2: **1937 passed, 25 skipped**. The adversarial review **caught a
real blocker**: the `_num_predict_comment` I put inside `llm.options` would have
leaked through the normalizer into llama.cpp's strict-signature
`create_chat_completion` → `TypeError` on the first on-device turn (the lenient
`**kwargs` test fakes hid it). Fixed by dropping `_`-prefixed keys in
`_normalize_llamacpp_options` (config.json's comment convention) + a
strict-signature regression test. Also applied the review's comment-accuracy fix
(on the sherpa engine `SPEECH_END`, not `ASR_FINAL`, is the turn-banker) and
float-coerced the cap.

## Wave 3 — control-plane-3: EWMA-scaled adaptive watchdog deadlines (headless)

Scouted the control-plane/routing cluster first; most candidates were done or
not-headless (`routing-cascade-7` already shipped; `routing-cascade-6`'s costly
redundancy is already killed by `LLMCapabilityRouter`'s built-in cache;
`routing-cascade-4` embedding-router + `asr-tts-3` RTF-guard need real
models/audio; `routing-cascade-1` live-routing is a deferred policy call). The one
clean, fully-headless win — and a direct extension of the rc-5 work — was
**control-plane-3**:

- `core/metrics.py`: the recorder now folds a **second** rolling EWMA — the TTS
  first-audio latency (`llm_first_token → tts_first_audio`), mirroring the existing
  TTFT EWMA. New `recent_tts_ms()`; folded in `mark()` on `TTS_FIRST_AUDIO`.
- `core/watchdog.py`: the "llm stuck" / "tts stuck" deadlines are no longer fixed —
  `_adaptive_deadline(recent_ms, base, mult, floor, ceil) = clamp(mult·recent,
  floor, ceil)`, falling back to the static base (LLM 10s / TTS 5s) at cold start.
  So a snappy desktop (TTFT ~0.3s) surfaces a real stall at the 4s floor instead of
  waiting 10s, and a slow phone (TTFT ~8s) stops false-flagging an honest 12s turn
  (deadline rises to the 30s ceil). The watchdog still only *diagnoses* (warnings →
  bundle); task reaping is separate. The `stuck_hints` substring matcher
  (`core/runlog.py`) is intact (messages still start `llm stuck:` / `tts stuck:`).

Tests: `test_metrics.py` (TTS-EWMA fold + reset) + `test_watchdog.py` (clamp helper;
deadline tightens on a fast device, loosens on a slow one; TTS path; toggle pin).
Suite **1945**.

Adversarial review (correctness + design lenses, verified) = **0 refuted, all nits
+ 1 LOW, no blocker**. Applied the worthwhile findings: the LLM floor is **6s** (not
4s) because the TTFT EWMA *blends* fast/main tiers — a fast-tier-dominated average
must not false-flag an honest occasional heavy-tier turn; added an
`ADAPTIVE_DEADLINES` toggle to pin the legacy fixed deadlines; switched the EWMA
guards to `math.isfinite` (reject `+inf`); and documented that the supervisor
**task-reap** (not this warning) is the real hang safety-bound, so a looser
slow-device deadline only delays the *log line*, never the cancellation.

## Wave 4 — low-value headless cleanup: llm-inference-9 + control-plane-2

Scouted the remaining headless backlog; landed the two that are clean AND
fully testable, and deferred two with precise blockers.

- **llm-inference-9 (KV-cache quantization):** `LlamaCppLLM` never passed
  `type_k`/`type_v`, so the on-device KV cache used the f16 default. Added
  `_resolve_kv_cache_type` (friendly name → ggml int; `q8_0`→8) + plumbed
  `type_k`/`type_v` through `llm_factory` from config. `phone`/`phone_lite` quantize
  the **K cache only** to `q8_0` (V left at f16) — the review flagged that V-cache
  quant requires flash-attention (not validated on this hardware), so K-only is the
  safe headless win (~halves the K-cache memory, no flash-attn dependency). The
  `Llama()` forwarding **fails soft** (an old lib lacking the kwargs degrades to f16
  instead of crashing the first turn). `n_ctx` bounding was already per-profile.
  Tests: resolver, constructor forwarding + the fail-soft path (fake `llama_cpp`),
  config→factory wiring (`type_k==8`, `type_v` None).
- **control-plane-2 (load-elastic admission):** under sustained system load the
  supervisor now tightens its concurrent-task ceiling to 1 (a 2nd turn queues
  rather than thrash a saturated CPU/GPU); the first turn always admits, and it's
  inert without a load reader. Threaded an **ungated** `load_fraction` reader
  app→runtime→supervisor (deliberately separate from the `live_routing`-gated
  `load_snapshot`, so it can't enable the deferred routing nudge). Fully mockable
  test (`test_load_elastic_admission.py`).

**Deferred (with reasons, recorded in memory):**
- **llm-inference-6** (in-process llama.cpp reap timeout): a per-call timeout would
  let the *current* turn give up but `LlamaCppLLM` holds a single-context lock
  across the blocking native call, so every subsequent turn would then block on
  lock acquisition unbounded. A correct fix needs a context-pool / lock-release
  refactor — disproportionate for a low-value item the supervisor's 25s task-reap
  already backstops.
- **llm-inference-7** (Q3/Q4 K-quant ladder): the real value is flipping
  `phone_lite` to a Q3 GGUF, a model-quality change (1B models are
  quant-sensitive) that needs on-device validation — not headless-safe to ship.

Suite after Wave 4: **1956**. Adversarially reviewed (1 refuted, 8 nits + 1 LOW,
no blocker): applied the LOW (K-only KV quant, above), the fail-soft constructor
guard, and a cp2 comment/test fix (the queue drains event-driven on task
completion + a no-starvation test); the cp2 correctness findings all verified
"correct as written".

## Environment (i9-13980HX, `.venv` ACTIVATED in the owner's shell)
Anything touching models/audio/LLM must use `.venv/bin/python` (a fresh shell's
`python3` is bare). Models present (sherpa ASR/VAD/TTS + CAM++ speaker-ID +
SenseVoice, ~600 MB); Ollama up; `doctor=READY`. `python` (no `3`) is NOT on PATH
in a fresh shell — use `.venv/bin/python`.
