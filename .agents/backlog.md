# Improvement backlog

Priority queue. `[ ]` = open, `[x]` = shipped (see `changelog.md`).
P0 = correctness/blocker, P1 = high value, P2 = nice-to-have.

> Refreshed 2026-06-02 during the architecture/doc unification pass. Durable
> design rationale now lives in [`docs/unified_architecture.md`](../docs/unified_architecture.md);
> this file tracks only OPEN work. The session-bootstrap helper
> (`python -m tools.session_bootstrap`) reads the OPEN P0 items below.

## P0 — correctness / blocker
- _(none open — full logic suite green: 1205 passed, 10 skipped)_

## P1 — voice / audio (migrated from session_2026-06-01 handoff)
- [ ] **Enable + validate AEC on real hardware** (needs the mic). `config.local.json`
      → `sherpa`: `aec_enabled=true`, start `aec_backend="nlms"`. Calibrate
      `aec_ref_delay_ms` with `tools/echo_probe.py`. Confirm no self-interrupt AND a
      real interrupt still cuts through; then optionally try `aec_backend="dtln"`.
- [ ] **Extend `tools/echo_probe.py`** to print post-AEC **ERLE (dB)** and auto-suggest
      `aec_ref_delay_ms` via cross-correlation (no mic needed to write it).
- [ ] **Validate the Smart Turn v3 endpoint on hardware** (the prosody detector +
      `tools/turn_detect_check` real-voice validation tool + an adaptive
      confidence-tiered endpoint floor all LANDED on main from the voice batch below;
      what remains is the on-hardware A/B). Run
      `python -m tools.live_session --all --inject --smart-endpoint`; diff ON finals
      vs lexical/acoustic. (Default-off; on in the desktop profile.)
- [ ] **DTLN follow-ups:** smaller 256/128 size for phone profiles; clock-drift over
      long utterances; consider LiveKit AEC3 if the runtime ever moves to ≥3.11.
- [ ] **Move coherence ingest off the audio callback (real-time hardening).** The
      callback-`OutputStream` rewrite tees the played block into the AEC far ring,
      the level EWMA, AND `EchoCoherenceDetector.note_playback` from `_audio_cb`
      (the PortAudio thread). `note_playback` takes the detector lock that the
      capture thread also holds while `decide()` concatenates the reference ring —
      the only contended lock on the audio thread. Bounded + harmless at the
      default `coherence_ring_ms` (~38 KB / sub-100µs concat), but it MUST move
      off the audio thread (feed coherence from a lock-free SPSC stage drained on
      the capture/worker thread, like `FarEndRing`) before `coherence_ring_ms` is
      raised materially. Documented inline in `_audio_cb`'s docstring.

## P1 — desktop / 4090 fit
- [ ] Adopt `desktop_gpu_4090` profile on this machine (currently `device=desktop`;
      4090 profile raises `num_ctx` 4096→8192, `num_predict`→512, enables both gates).
- [ ] Measure real end-to-end ASR→LLM→TTS latency on the 4090 (`tools.bench --real`),
      calibrate `tools/specsim/specs.py` against it.
- [ ] Confirm sherpa runs the CUDA provider (config `sherpa.provider="cpu"` today) —
      evaluate GPU ASR/TTS on the 4090 vs auto-tuned CPU threads (32 logical).

## P1 — architecture / cross-platform
- [ ] **Mobile convergence onto the `AgentEvent` contract.** `mobile/lib/assistant.dart`
      is a parallel Dart loop re-deriving core behavior; align the Dart supervisor with
      the Python brain so the contract duplication disappears. See unified doc §10 / §12.
- [ ] **SQLite + sqlite-vec memory backend** for mobile (the `Memory` protocol makes it
      a drop-in for the Postgres adapter). See unified doc §6.

## Gemma 4 — ADOPTED 2026-06-05 (gemma4:12b)
- [x] **Adopted gemma4:12b** as the model (config.local.json, machine-local).
      Required updating **Ollama 0.24.0 -> 0.30.5** (0.30.4 412s on the gemma4
      manifest; 0.30.5 was the gemma4-capable release, GitHub installer since
      winget lagged). Measured head-to-head via `tools.model_probe` on the 16GB
      box: **gemma4:12b = 8.1GB VRAM (all GPU, ~7GB headroom), text 4/4, multimodal
      YES ('Red'), 256K-capable (num_ctx 8192)** -- a clean upgrade from gemma3:4b.
      gemma4:e4b = tiny (3.3GB VRAM) + 4/4 text but vision NOT wired in Ollama
      ('Please provide the image...') -> text-only. End-to-end VERIFIED: a host
      frame via `runtime.set_current_frame()` reaches gemma4 through the capability
      (image turn -> main tier -> 'Red'); text-only -> fast tier. Suite 1366 green.
      OPTIONAL next: two-tier gemma4:12b main + gemma4:e4b fast (~11.4GB, fits) for
      a real fast/main split; bump num_ctx toward 256K if memory/context needs it.
- [ ] _(superseded eval notes)_ **Evaluate + adopt Gemma 4** (Google, Apache-2.0, ~Mar 2026, actively updated).
      Ollama tags: `e2b`(7.2GB,+audio,128K), `e4b`(9.6GB,+audio,128K),
      **`12b`(7.6GB, image, 256K)** ← smaller than gemma3:12b's ~10GB, the best
      16GB fit; `26b`(18GB, MoE 3.8B active) + `31b`(20GB) too big. All multimodal
      (image); E2B/E4B/12B add native AUDIO + VIDEO in. The swap is **config-only**
      (`OllamaLLM` is model-name-agnostic; the `images=` path already works):
      `config.local.json` → `device_profiles.desktop_gpu_4090.llm`:
      `main_model`/`fast_model` → `gemma4:12b`, bump `options.num_ctx` (256K-capable).
      Harness ready: **`python -m tools.model_probe gemma4:12b gemma4:e4b --pull`**
      measures VRAM + text quality/TTFT + multimodal, vs the gemma3:4b baseline
      (4.4GB VRAM, 4/4 text, sees-image yes). BLOCKED on disk: C: had **1.8GB free**
      (need ~17GB for the 12b+e4b head-to-head). VERIFY when disk is freed:
      (a) Ollama 0.24.0 loads the gemma4 architecture (else update Ollama),
      (b) real VRAM on 16GB, (c) multimodal through the pipeline. Follow-ups:
      audio/video IN = new plumbing; mobile needs a LiteRT/.task Gemma 4 for
      flutter_gemma. See tools/model_probe.py.

## P1 — capability / testing (from 2026-06-05 test-unification pass)
- [x] **Multimodal image plumbing wired (2026-06-05).** The capability layer now
      forwards images: `core/capabilities.py::assistant()` reads `context['images']`
      (per-turn) or an ambient `image_provider()`, forwards them as
      `stream(images=…)`, forces the **main/multimodal** tier (the fast 1b can't
      see images), and floats sensitivity to PRIVATE (a screen capture never rides
      a public cloud chain, §9.7). `VoiceRuntime.set_current_frame(image)` /
      `clear_current_frame()` let a host machine feed the current frame ambiently
      to every assistant turn. Tests in `tests/test_core_multimodal.py`
      (per-turn/ambient/override/runtime + text-only-carries-none).
      A screen-capture SOURCE now exists too (`core/screen_capture.py`,
      `ScreenFrameFeed` + `build_screen_feed`, wired in `core/app.py`), **OFF by
      default** (`config.screen_capture.enabled`); when on it grabs the screen
      every `interval_sec` (mss + optional Pillow) and feeds `set_current_frame`.
      **REMAINING:** on-hardware validation (enable it on a live `--engine sherpa`
      run and confirm a frame reaches the multimodal model + the latency cost is
      acceptable), and any non-screen source (camera/app) if wanted.

## P2
- [ ] Wire `tools/swarm/harness.py perf --real` into `.github/workflows/perf.yml` parity.

## Shipped this session (2026-06-02)
- [x] **Landed the unification refactor on `main`** (merge `d215a31`): merged
      `feat/aec-dtln` (`unified_architecture.md` + `session_bootstrap` + Windows landing
      doc) onto the diverged `origin/main`. Clean merge, full suite green
      (1283 passed, 13 skipped). Branch deleted.
- [x] Architecture/doc unification: `docs/unified_architecture.md` absorbs ~14 dated
      docs; merged docs banner-linked; stale session logs archived.
- [x] Removed dead `always_on_agent/snapshots.py`; renamed `core/agent.py` private
      `AgentEvent` → `AgentBrainEvent` (collision with the public contract).
- [x] Added `tools/session_bootstrap.py` + CLAUDE.md "Session bootstrap" section.
- [x] Relocated the accidentally-nested `social_media_activities_app/` out of the repo;
      removed the `UsersPaul` junk file; added `.gitignore` guards.

## Landed on `main` from the other machine (origin/main voice batch, 2026-06-02)
> Merged into `main` here; recorded so the next session knows this is already on `main`.
> `docs/unified_architecture.md` predates this batch and needs a refresh pass to cover it.
- [x] **SenseVoice two-pass final ASR**, shipped as the DEFAULT (run-on speech fix);
      pinned to English (was mis-detecting Chinese).
- [x] **Smart Turn v3 prosody turn-completion detector** + `tools/turn_detect_check`
      real-voice validation tool + adaptive confidence-tiered endpoint floor (~-110ms).
- [x] **Multi-signal barge-in stack:** loudness fallback (embedder-unreliable setups),
      scale-invariant reference-coherence detector (volume-independent, zero-setup),
      self-calibrating trigger margin (EWMA control chart, no per-room tuning).
- [x] **Enrollment hardening:** pin `capture_samplerate` (AT2020 self-mute fix),
      loudness rescue + VAD-trimmed enrollment; speaker-gated barge-in calibration doc.
- [x] `tools/echo_probe.py` added; `live_session` per-capability latency + denoise A/B
      + barge-in crash guard + response-quality grading.
