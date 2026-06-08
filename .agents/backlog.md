# Improvement backlog

Priority queue. `[ ]` = open, `[x]` = shipped (see `changelog.md`).
P0 = correctness/blocker, P1 = high value, P2 = nice-to-have.

> Refreshed 2026-06-02 during the architecture/doc unification pass. Durable
> design rationale now lives in [`docs/unified_architecture.md`](../docs/unified_architecture.md);
> this file tracks only OPEN work. The session-bootstrap helper
> (`python -m tools.session_bootstrap`) reads the OPEN P0 items below.

## P0 — correctness / blocker
- [~] **★ WINDOWS-side self-interrupt (live 2026-06-08e, run-20260608-181250) —
      CASCADE HALF FIXED 2026-06-08f.** The runaway "two outputs back-to-back" is now
      broken on ANY device (merged `fix/echo-final-cascade` → main, 1416 green;
      handoff `docs/session_2026-06-08_echo_final_cascade_fix.md`). Three
      device-adaptive, additive, off-switchable layers — NO fixed magic-number
      thresholds (every bar RELATIVE to a LEARNED per-device floor): **L1**
      `_final_above_floor` drops a final at/near `max(_ambient_rms, _playback_floor_rms)`
      (a dB-above-LEARNED-floor margin via `loudness_admits`, never an absolute RMS),
      wired at the final-dispatch seam; fails OPEN until a floor is learned, so it only
      bites on AEC/open-speaker configs. **L2** `_in_post_speaking_refractory` (per-reply,
      stamped at both `_speaking→clear` sites) suppresses a re-fired barge on the
      cancelled echo tail; on the barge debounce ONLY, never the final seam, so a real
      barged-in request is never dropped. **L3** `agreement_guard` (core/asr_text.py)
      demotes short-clip SenseVoice hallucinations (`BEING`→`I.`) back to the streaming
      final. +30 tests; off-switch parity (`final_floor_margin_db=0`,
      `barge_in_refractory_sec=0`). **STILL OPEN — the FIRST self-interrupt itself
      (needs the mic):** the Windows AEC mis-calibration (stale pre-FIFO
      `aec_ref_delay_ms`). Run `python -m tools.echo_probe` (echo-only) on the Windows
      box → pick the ERLE-max `aec_ref_delay_ms` (don't assume 0), raise `dtd_k` if
      echo-only D approaches K=5.0; target self_interruptions=0. Replay
      run-20260608-181250.wav through echo_probe to iterate without re-talking. The
      cascade fix makes this miscalibration NON-catastrophic until it's done. Then
      live-tune `final_floor_margin_db` (6.0) / `barge_in_refractory_sec` (0.5) — watch
      the `echo_floor_rejected_final` metric in run bundles. Original analysis:
      docs/session_2026-06-08_live_session_self_interrupt.md (think=false latency +
      prosody endpointing both validated live; re-assess turn-taking after the mic fix).

## P1 — voice / audio (migrated from session_2026-06-01 handoff)
- [x] **★ HARD REQUIREMENT (owner): open-speaker barge-in WITHOUT headphones —
      DONE + LIVE-VALIDATED 2026-06-08** on the bare ALC285 laptop mic+speaker (no
      premium mic). Owner: "barge feels good now" — a NORMAL-volume talk-over
      interrupts reliably, no shout, no self-interrupt. Solution: device-adaptive
      fused z-score double-talk detector `AdaptiveDTD` (core/engines/_dtd.py) — three
      features (raw energy / post-AEC residual / coherence) each a self-calibrated
      upward z-score from its OWN echo-only control chart; fire on the weighted SUM
      D > dimensionless K. NO fixed margin (the prior fixed-margin attempts all
      failed: self-interrupt, or rejected normal talk-over, or needed a shout). The
      decisive fix was the FIRING LOGIC, not the physics: per-frame fire
      (dtd_confirm_frames=1) + the capture-loop LEAKY integrator, because a real
      talk-over scored D=90-130 but flickered, so the old 3-consecutive rule
      discarded it. Tuned (SherpaConfig defaults): weights (raw 0.2, resid 1.0, coh
      0.0) -- z_resid is the discriminator (user voice isn't in the reference -> AEC
      can't cancel it -> lands in the residual); dtd_chart_rel_floor 0.4 (echo-leak
      precision). Commits fe0617b→5cd7f60 (coherence-primary audit) then
      71fd4ec/bbc4e01 (DTD + live tuning). Handoff:
      docs/session_2026-06-08_device_adaptive_barge_in.md. tools/echo_probe.py logs
      per-frame D for re-calibration on any machine. ref_delay stays 0 (FIFO already
      aligns the far-ref; do NOT set 260ms).
- [ ] **★ Stream the TTS for long answers (owner 2026-06-08).** A long story feels
      like it waits for the whole LLM answer before speaking. `_stream_and_speak`
      (core/capabilities.py) already emits sentence-by-sentence, so investigate why
      long answers feel un-streamed on the sherpa path: likely gemma4:12b main-tier
      first-token latency for a story, and/or confirm the per-sentence emit reaches
      TTS playback incrementally end-to-end. Goal: first audio after sentence 1, not
      after the whole story.
- [ ] **★ Smarter endpointing / turn-taking — don't barge in on the user's pauses
      (owner 2026-06-08).** When the user speaks with small mid-thought pauses, the
      assistant replies too early. It should use context to tell "still talking" from
      "done" and respond only with HIGH CONFIDENCE the turn is complete. Lever exists:
      Smart Turn v3 PROSODY endpoint detector on disk
      (pretrained_models/sherpa/smart_turn/), but endpoint_detector='lexical'. Switch
      to 'prosody' (needs onnxruntime+transformers), live-tune the confidence floor.
      See core/endpointing.py (adaptive confidence-tiered floor).
- [ ] **SenseVoice 2nd-pass agreement-guard (STT quality).** Re-enable sense_voice
      but accept it only when it AGREES with / clearly improves the streaming final
      (kills the short-clip hallucination 'I'->'Okay.'). New core/asr_text.py
      token-agreement helper + _final_transcribe word-count gate. (Currently reverted
      to streaming-only in config.local.json because the unguarded 2nd pass
      hallucinated.)
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

## Smart routing — phase-2 audit (2026-06-08, 6-dimension fan-out: 28 findings, 19 confirmed)
> Verdict: smart routing is largely HEALTHY + fail-safe (no P0, no active §9.7
> boundary leak, every risky path double-bounded toward the more-private/conservative
> choice). Through-line = DORMANT intelligence (live_routing/cost_order/capability_router
> built+tested but off in most profiles) + one narrow PII gap. Adversarial verify
> dropped 9 plausible-but-wrong findings (e.g. "screen captures ride US cloud" is
> §9.7-authorized; several "boundary leak" claims were post-ASR-text egress, which
> the boundary permits). Landed in `feat/smart-routing-phase2-hardening` -> main.
- [x] **PII fail-safe for lowercased ASR (core/sensitivity.py).** Name+money rule was
      case-SENSITIVE; lowercase ASR ("what is john salary") slipped to PUBLIC. Added a
      case-insensitive comp-word rule (salary/wage/income/paycheck/pay stub/net worth/
      bonus) -> PRIVATE. The single fail-UNSAFE path, now closed.
- [x] **Host-aware cost ordering (core/routing.py `_preset_cost_key`).** Added host_rank
      as the OUTERMOST sort key so CN sorts after US/unknown (cost optimizes within a
      jurisdiction, not across it); fixes the CN-floats-ahead + OpenRouter-sinks-below-CN
      bugs. Latent until cost_order is enabled -> fixed before enablement.
- [x] **capability_router ON for desktop_gpu_4090 (config.json).** The shipped device
      inherited enabled=false while base 'desktop' had it on (most-capable profile, least
      routing intelligence). Mirrored desktop's block.
- [ ] **Activate dormant cost/latency levers on the CLOUD profiles + add measured evidence
      (the audit's #4, deferred).** Set `live_routing:true` (llm block) + `cloud.cost_order:true`
      on `cpu_laptop` / `phone_lite` (optionally `macbook_m_series`) where local is slow and
      cloud is on; keep desktop/4090 OFF (local 12b is fast). The cost_order fix above
      unblocks this. PAIR with a `tools/bench` (or replay) smoke that emits the chosen chain
      order + asserts cost_order lowers TTFT and the live nudge shortens the hedge under a
      high-load snapshot -- the first measured proof these levers help. Needs cloud keys to
      validate -> do on a cloud-enabled device. Files: config.json device_profiles
      (cpu_laptop/phone_lite/macbook), tools/bench/{runner,report}.py, docs/unified_architecture.md §5.
- [x] **P2 routing polish (6 of 7 shipped 2026-06-08d via the `p2-routing-polish` fan-out;
      merged to main).**
      (a) DONE -- HedgeLLM.shutdown() now WARNs (`speaker.llm.hedge`) with the survivor count
      when worker threads outlive the join budget (core/llm.py); the leak is visible in the
      run bundle instead of silent.
      (b) DONE -- WINNER_SELECT_BUDGET_FLOOR 30s -> 10s (core/llm.py:687).
      (c) DONE -- tier markers are now `\b`-anchored regexes ('show me the time' no longer
      hits 'how'; multi-word markers still match) + added 'compose'/'draft'/'write an'
      (core/routing.py `_compile_markers`). No route flips (nudges only).
      (d) DONE -- the ESCALATED (ReAct) + research.local paths now `_enrich_context` + publish
      `capability_context` (set/reset in try/finally, no cross-turn leak) so SensitivityRouterLLM
      picks the right cloud chain on those turns too (core/capabilities.py).
      (f) DONE -- LearnedRouter build-path test (backend='learned' raises RuntimeError when
      torch absent; tests/test_core_routing.py).
      (g) DONE -- doc-truth fixes (config.json cost_order comment; docs/unified_architecture.md
      Cost Order test-file citation -> tests/test_core_routing.py + host_rank note).
      (e) DEFERRED (coupled to audit #4) -- tier-aware load_fraction + shorter SystemMonitor
      cadence when live_routing on (core/sysinfo.py). The headroom signal mis-attributes CPU
      STT/TTS vs GPU LLM and lags 10s vs 1-3s turns, but it only matters once live_routing is
      enabled (the deferred #4), so do it WITH that work.

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

## Barge-in coherence-primary — v2 AND-gate LANDED + self-interrupt VALIDATED 2026-06-07
- [~] Coherence-on-raw-mic is the barge trigger (fe0617b); v2 (377d10e) requires a
      coherence "user" verdict to ALSO clear the post-AEC residual floor (orthogonal
      signals: AEC kills echo ENERGY not its incoherence; a real talk-over is incoherent
      AND loud). **LIVE-MEASURED on the open ALC285: 0 self-interrupts across 5 runs at
      full volume** (coherence-alone self-interrupted; echo raw-mic incoherent ~0.88
      overlaps real voice). **REMAINING (needs a human, can't be machine-tested):**
      confirm a REAL talk-over STILL fires — `python -m core --engine sherpa`, talk over
      a long reply; if missed lower `barge_in_residual_margin_db` (10.0). Re-confirm on
      the AT2020 USB mic (was unplugged this session). Handoff:
      docs/session_2026-06-07_barge_in_coherence_primary.md.
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
- [ ] **`web.search` egress vs the floated turn sensitivity (review nit, 2026-06-08g).**
      `core/websearch.py` gates egress on the raw tool `arg` via `may_leave_device(arg)`,
      independent of `context["sensitivity"]`. Now that the ReAct planner also sees the
      recent-conversation block (2026-06-08g), it has more material it *could* phrase into
      a search arg. `_is_personal` re-classifies that exact arg so literal PII still fails
      closed, but consider ALSO gating the search arg on the turn's floated sensitivity
      (defence-in-depth). Orthogonal to the 2026-06-08g change; not a regression.

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
