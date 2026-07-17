Valid until: superseded by the next accepted architecture/roadmap decision — then treat as history.

# Performance roadmap — reaching big-tech voice smoothness (2026-07-17)

Produced by a 25-agent review (4 repo auditors, 5 SOTA researchers, merge, per-item
adversarial verification against the live box, completeness critic). Every claim
below was verified against repo code, run bundles, or the installed venv unless
marked *hypothesis*. Verdicts: 13 recommendations confirmed (most with minor
corrections), 1 refuted, 6 gaps added by the critic.

## 1. Verdict: this is NOT a compute-only problem

The owner question was: "is the lag and accuracy a compute-only problem?" No.
Three verified facts settle it:

1. **The smart model never runs.** `config.local.json` (TEMP 2026-07-02
   "disk full" note, lines 6-15) pins `main_model` AND `fast_model` to
   `gemma4:e4b` — every answer in every live session came from the ~4B fast
   model. Tier routing, escalation, and threshold tuning are currently cosmetic.
2. **The measured latency is policy, not inference.** Across the 2026-07-02 live
   turns: end-of-speech→first-audio median ~2.2 s, of which endpoint silence
   wait is ~1.3 s (a local 1.1 s override of the validated 0.7 s floor) and
   turn_merge holds add 1.2-1.45 s. LLM final→first-token was only ~0.29 s warm.
   The GPU is mostly idle during the perceived lag.
3. **Accuracy failures are pipeline, not model.** The 07-04 A/B replay proved
   raw pre-APM audio transcribes cleanly — the live path shredded it. Windows
   mic captured active speech 30-70x below normal level (avg_rms ~0.0007);
   32% of finals were gate-dropped; 54% of finals in run-223217 were ≤2-word
   fragments, most answered literally ("Bray" → "Did you mean to say 'bra'?").

A 16 GB RTX 4090 laptop has more local compute than any phone. Compute is not
the ceiling; configuration rot, missing streaming glue, and two real model
swaps are.

## 2. Why Google/OpenAI phone voice feels smooth (and what transfers)

Their phone smoothness is **not** on-device compute. Gemini Live / ChatGPT
voice stream compressed audio to datacenter models; the phone contributes
hardware AEC/DSP and capture. What makes them feel instant:

| Technique | Their implementation | Local-first equivalent (this repo) |
|---|---|---|
| Speech-native duplex models | GPT-4o realtime, Gemini Live: audio-in/audio-out, no cascade seams | **Rejected for 16 GB** (verified): Moshi ~24 GB + weak reasoning; Qwen3-Omni Talker is cloud-only. Kyutai themselves shipped cascaded Unmute for intelligence. Keep the cascade. |
| Streaming everything | Token→speech begins before the sentence completes | Sentence-streaming exists; add ack layer + speculative fast-tier start on stable partials |
| Semantic turn-taking | Model decides end-of-turn, not silence timers | Smart Turn v3.1 already integrated; restore the validated 0.7 s floor it was meant to enable |
| Instant acknowledgment | Fillers/preambles mask thinking (ConvFill: 2.9-7.2 s → <1 s perceived) | No ack mechanism exists today — highest-leverage UX add |
| Interruption truncation | History truncated to audio actually heard (OpenAI Realtime pattern) | **Already implemented** (playback receipts → safe_text_prefix); verified, no work needed |
| Hardware/OS echo cancel | Phone DSP + AEC | OS voice-comm route (ADR-0013); the Windows leg is broken (ADR-0019) — Phase 2 |
| Memory across sessions | ChatGPT memory (the "feels smart" benchmark) | Fully built in-repo, config-gated OFF |

Proof the target is reachable locally: RealtimeVoiceChat demonstrates ~500 ms
voice-to-voice fully local with a 24B via Ollama; Kyutai Unmute runs a smooth
cascaded stack. Industry budget: ≤800 ms first audible response feels human;
>1.5 s reads as broken.

**Ground-up redesign: not needed.** The 2026-07-12 parity audit stands — the
control-plane architecture matches LiveKit/Pipecat concepts. What's missing is
(a) reverting dev-box config rot, (b) streaming glue + acks, (c) three model
upgrades, (d) the Windows barge route.

## 3. Refuted / settled negatives (do not re-litigate)

- **r04 REFUTED — Windows Communications ducking is not attenuating TTS.**
  Verified on-box: sounddevice 0.5.5 has no `communications` kwarg, the
  constructor raises TypeError, no comm-role stream ever opens, so Windows
  never starts a ducking session. (Same fact makes r14 the real work.)
- **No local speech-to-speech migration** (Moshi/Qwen3-Omni class) — see table.
- **Skip Canary-Qwen-2.5B / Whisper-family for finals** — Parakeet v2 dominates
  on English WER and has a sherpa-onnx int8 path; the others don't fit or lose.
- **Skip speculative decoding if the main tier becomes MoE** — measured
  net-negative on A3B-class targets (3B-active decode is already draft-speed).
- **Duck-then-confirm caveat:** ADR-0011 already implemented duck+confirm and
  ADR-0013 measured the duck itself pumping on the nonlinear open speaker
  (6/7 false ducks at -18 dB). Confirm-before-cancel: yes. Reintroducing an
  acoustic duck: only with evidence.

## 4. Baseline (2026-07-02 bundles) and acceptance targets

| Metric | Baseline | Target |
|---|---|---|
| End-of-speech → first audible audio (p50) | ~2.2 s (max 4.85) | <0.8 s (ack counts), full answer start <1.5 s |
| Endpoint silence-to-commit | ~1.3 s | ~0.7 s (validated floor), later tune accuracy at 0.2-0.3 s |
| Cold-start first LLM call | 18.79 s | <3 s (pre-warm both tiers before capture opens) |
| Finals dropped by gates | 32% (23/71) | <5% with gates ON |
| ≤2-word fragment finals answered | ~13/42 in worst run | ~0 (min-content gate) |
| Missed talk-overs / false ducks / self-interrupts | 17 / 9 / 1 per 5 sessions | 0 self-cuts; talk-over cut <0.5 s (0.35 s already proven once) |
| Sentence loudness swing | 3.6 dB | <1 dB (leveler validation) |

**Acceptance policy (critic gap):** every roadmap item below defines done as a
measured live A/B against these baselines using the tools that already exist
(`tools/live_audio_ab.py`, `tools/calibration_suite.py`, `tools.live_session
--inject`) — not a config flip. The leveler flip and ADR-0013 Phase-B both
show what happens otherwise: flipped, never measured, complaint persists.

## 5. Roadmap

### Phase 0 — Un-rot the box + instrument (days, config/small edits)

1. **Fix the Windows OS mic level** (root STT garbler). Raise input level
   toward ~-20 dBFS active median; verify with `tools/calibration_suite.py`;
   A/B denoise on/off at the corrected level; then re-enroll speaker-ID
   through the current front end (denoise flip shifted embeddings) and lower
   `speaker_threshold` toward 0.4. Make sustained avg_rms<1e-3 fail readiness
   instead of warn (`core/engines/sherpa.py:4653`).
2. **Restore the two-tier LLM split.** Free disk, `ollama pull gemma4:12b`,
   restore main=12b / fast=gemma3:4b; `keep_alive=-1`, `OLLAMA_MAX_LOADED_MODELS=2`.
3. **Finish the Kokoro swap (ADR-0010, ~90% done).** Free ~1 GB, `python -m
   tools.setup_models --kokoro`, set voices/tokens/lexicon + `tts_output_lowpass_hz≈7000`.
   Before the validation run (critic gap): route playback resampling through the
   existing soxr `AudioResampler` instead of `_resample_linear` (sherpa.py:1301 —
   Kokoro is 24 kHz, a mismatched device rate would alias), and add an
   inter-sentence dry-gap counter next to the underrun diagnostics. Then one
   live pass: leveler RMS steadiness, gap check, echo_probe + barge-gate
   re-check (new voice = new echo signature).
4. **Prompt content fixes (critic gap, minutes).** Set `assistant.name` + one-line
   persona; relax "one or two sentences" to "short by default, expand for
   substantive questions"; make the hardcoded "no web access" line
   (core/persona.py:40) follow actual web/cloud state; add a variety rule.
5. **Context budgets.** Add a `memory` override to the desktop profile:
   `recall_recent_reserve_tokens` 320 → ~1500-2000 (the 320 was sized for the
   1536-ctx phone profile; it binds inside an 8192 ctx). Static-first prompt
   order so Ollama prefix-KV makes it near-free. (Verified: barged replies are
   already truncated to heard prefix — only add an optional `[interrupted]` marker.)
6. **Per-stage latency instrumentation.** Timestamps at endpoint-fired,
   final-repass-done, LLM-first-token, first-TTS-sample; p50/p95 in run bundles;
   regressions gate on it.

### Phase 1 — Perceived latency + smartness (1-2 weeks)

7. **Endpoint re-tune (validated-value restore).** `endpoint_min_silence_sec`
   1.1 → 0.7 and re-enable `endpoint_high_confidence_floor` 0.6 — both were
   validated 2026-05/06 under Smart Turn; the 1.1 override compensated for the
   broken mic. Pre-warm both tiers before capture opens.
8. **Instant-acknowledgment layer (Talker/Reasoner).** Pre-synthesized rotating
   ack pool (zero marginal latency, barge-interruptible) at the tier-router
   decision point; optional fast-tier one-liners for named tasks. This is the
   single cheapest "feels like GPT-voice" change (ConvFill evidence).
9. **Stop answering fragments/noise.** Restore committed `input_gate`/`cleanup`/
   `speaker_gate_input`; add a min-content gate (word count + ASR confidence)
   before LLM dispatch; extend addressing gate output to
   {answer, backchannel, stay-silent, defer} (Gemini Live proactive pattern);
   gate cleanup rewrites on raw-length/agreement so it cannot fabricate.
10. **Turn on the memory stack (built, tested, config-gated OFF).**
    `memory.backend=sqlite`, `procedural_enabled` first, then `recall_enabled`
    with recall fired at endpoint time so retrieval overlaps the gate call.
    One proactive recalled line on session start/topic match. (Correction from
    verify: `profile_enabled` semantics differ slightly from the audit — check
    core/capabilities.py:814-935 wiring while enabling.)
11. **Routing honesty.** Word-boundary + curated escalation markers (reuse
    `core/routing.py:_compile_markers`); ACT → honest one-shot "can't do that
    yet" until real capabilities exist; ASR-confidence gate on high-confidence
    research routes; planner off until real search exists.
12. **Real search + cloud thinking tier (§9.7-compliant).** Stand up self-hosted
    SearXNG (config already wired, `web_search.enabled=true`); enable
    `llm.cloud.enabled=true` + one key (e.g. `$GROQ_API_KEY`) so HedgeLLM races
    a 120B-class model against local main for escalated turns,
    `default_chain=private` keeps sensitive turns local; `think=true` on
    main-tier escalated turns only.

### Phase 2 — Open-speaker barge-in on Windows (the hard P0)

13. **Ship real WASAPI communications capture** (ADR-0019 unblock): small
    ctypes/pybind helper opening the capture with `AudioCategory_Communications`,
    satisfying `verify_required_os_echo_route` with a hardware-verifiable proof.
    **Contingency (critic gap):** if not verified within one session, flip the
    dormant calibrated fallback — `aec_enabled=true`, `aec_backend=apm`
    (WebRTC AEC3, installed 2026-07-02; `aec_ref_delay_ms=105` echo-probe
    calibrated, 30.3 dB ERLE) — and run the barge stress harness. Treat OS-route
    vs AEC3 as an A/B, not a bet.
14. **Guaranteed interrupt floor:** microWakeWord-style "stop/wait/hold on"
    spotter active during playback (HA Voice PE pattern; ONNX/CPU infra exists).
    Satisfies ADR-0008 as a floor even while statistical gates improve.
15. **Fix structural word-cut misses.** First, log word-cut speaker-similarity
    distributions from live bundles (line exists at sherpa.py:7618) — owner's
    own voice scored ~0.15 in the echo domain vs the 0.30 accept threshold, so
    every non-STOP talk-over may be silently rejected. Then a 2-3-word
    interjection cut class gated by speaker-accept + energy. Until a route
    exists: reuse `_splice_word_cut_preroll` so playback-time speech becomes
    the next turn's input ("deferred barge") instead of vanishing
    (today it is wholly discarded, sherpa.py:4733/:4799).
16. Re-run the ADR-0013 live gate (talk-over batch, bare-"stop", silent
    control) with the Kokoro voice.

### Phase 3 — Model upgrades (verified in-budget)

17. **ASR finals: parakeet-tdt-0.6b-v2 int8** (6.05% avg WER vs SenseVoice's
    weaker English; native punctuation/casing into the LLM; CC-BY-4.0;
    sherpa-onnx offline-transducer tarball exists; installed sherpa-onnx 1.13.2
    ≥ required 1.12.9; CPU second pass, zero VRAM contention). Add a
    `nemo_transducer` backend to `core/engines/_sherpa_models.py`; keep
    SenseVoice as fallback, Zipformer for partials/hotwords. Do after the mic
    fix so the A/B isolates the model delta.
18. **Main tier: Qwen3.6-35B-A3B MoE via llama.cpp `--n-cpu-moe`** (Apache-2.0,
    3B active, GPQA-D 86.0, ~30 tok/s reported at 6 GB VRAM). Verified caveat:
    Q4 GGUF is 22.1 GB and this box has 30 GB RAM — needs a Q3-class quant
    (~17-18 GB) or partial expert offload; benchmark on-box behind the existing
    OpenAI-compatible seam before adopting. Pure-Ollama fallback: `gpt-oss:20b`
    (14 GB MXFP4, o3-mini-class, strong function calling) at the cost of
    dropping/CPU-hosting the fast tier. If MoE lands, optionally consolidate
    gate/cleanup onto it and free the ~3.3 GB fast-tier VRAM.
19. **Optional spike (critic gap): audio-native understanding.** Check whether
    the resident gemma4:e4b (Gemma 3n lineage) exposes its audio encoder via
    Ollama; prototype "audio + draft transcript → cleaned intent" as a
    garble-robust front end. §9.7-safe (on-device).

### Phase 4 — Watchlist (do not adopt now)

- Streaming-text TTS (Kyutai TTS 1.6B, CosyVoice2/3-0.5B) — only if sentence
  TTFA becomes the measured bottleneck or VRAM frees.
- nemotron-speech-streaming-en-0.6b ONNX (would delete the two-pass ASR) —
  watch sherpa-onnx issues #2918/#3573.
- Kyutai stt-1b (streaming STT + semantic VAD in one net) — time-boxed spike only.
- Mobile: Moonshine v2 streaming (MIT, ONNX) as the `mobile/` ASR candidate;
  Kokoro int8 via sherpa_onnx Flutter (VoxSherpa proves it); KittenTTS nano fallback.
- LiveKit Turn Detector v1: license forbids use outside LiveKit Agents —
  usable in `remote/` only.

## 6. Expected end state

Phase 0-1 alone should move end-of-speech→first-audible from ~2.2 s to
<0.8 s perceived (ack) / <1.5 s full answer, with a 12B (later 35B-MoE or
hedged-cloud) brain, persistent memory, a modern voice, and honest routing —
i.e. the Google/OpenAI feel for everything except mid-reply interruption.
Phase 2 closes the barge gap (the one genuinely hard problem, and the one
big tech solves with hardware DSP). Phase 3 is the accuracy ceiling raise.

Full agent reports, verdicts, and sources: session workflow `wf_c9a86a70-555`
(run bundle paths in the session transcript; key sources cited inline in the
verdict notes).
