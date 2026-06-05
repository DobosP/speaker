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

## P1 — capability / testing (from 2026-06-05 test-unification pass)
- [ ] **Multimodal capability drops images (found while writing `test_multimodal_e2e.py`).**
      The LLM clients forward `images=` faithfully (the routed/raced
      `SensitivityRouterLLM`/`HedgeLLM` chain is now tested), but the
      capability/runtime layer **never reads `context['images']`** —
      `core/capabilities.py::assistant()`/`research_synth()` call
      `model.stream(query, system=...)` with no images, and nothing populates an
      image into the turn context (grep for `images` in `core/runtime.py`/
      `core/capabilities.py`/`always_on_agent/` is empty). So a "describe this
      image" turn silently loses the image — the vision model is reachable but
      unfed. Fix: thread an image source → `turn context['images']` →
      `stream(..., images=...)` in the assistant/main-tier capability, then extend
      `test_multimodal_e2e.py` with a `registry.invoke('assistant.answer', …)` case.

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
