# Improvement Plan — Local-First Voice Assistant

> ⚠️ **Historical record (point-in-time).** Superseded by [`docs/unified_architecture.md`](unified_architecture.md) and the current [`.agents/backlog.md`](../.agents/backlog.md). Kept for history. (2026-06-02 consolidation.)

Driven by the live session **`logs/runs/run-20260529-212103`** (user's own voice, this machine). Three reported failures, all root-caused against the artifacts: **barge-in didn't function**, **STT was wrong most of the time**, **answers were low-quality**. Every fix below is file-mapped and validated by replaying `/home/dobo/work/speaker/logs/runs/run-20260529-212103.wav` via `python -m core --engine replay --replay-dir logs/runs` plus targeted tests.

> **One claim was REFUTED and dropped from the plan:** "swap the 20M ASR model for zipformer-en-2023-06-26" is a no-op — the box is **already** running the full-size 2023-06-26 model (verified via param count and `tools/bench/models.py:22`). Look to the input chain (clipping, endpointing, hotwords, confidence) for STT, not a same-model swap. A genuinely different architecture (NeMo/Parakeet/Moonshine offline-chunked engine) is a roadmap item, not a same-day fix.

---

## The three reported problems → highest-leverage fix

### Problem 1 — Barge-in does not function (0/16 talk-overs)
**Root cause (verified):** `core/engines/sherpa.py:979` `_looks_like_user` gates barge-in on `gate is None or not gate.is_enrolled` and otherwise calls `gate.accept()` (cosine ≥ 0.5 against the enrolled CAM++ reference) — it **ignores `self.config.speaker_gate_input`**, which is `false` in `config.local.json`. The sibling `_should_act_on_final` (sherpa.py:1045) **does** honor the flag. The user's enrolled reference does not clear threshold for the short (0.1s), echo-contaminated barge-in clips, so every interrupt is vetoed → `barge_in_latency=null` on all 23 turns. The output-margin fallback exists but is only reached on the unenrolled branch, which is bypassed because the gate IS enrolled. **(Rank 1)**

**Fix:** In `_looks_like_user`, mirror `_should_act_on_final` — when `speaker_gate_input` is False, skip identity and fall through to `passes_output_margin(rms(samples), self._playback_level, margin_db)`. When it's True, accept on `(margin_ok OR id_ok)` so a mismatched voice-print can never make the assistant un-interruptible. Keep `barge_in_output_margin_db=6.0` as the no-AEC self-echo guard.

### Problem 2 — STT is wrong most of the time
**Root cause (verified, NOT the model):** `input_gain=8` hard-clips loud phonemes into flat-topped distortion — measured on the recorded WAV: **0.282% of samples pinned at ±1.0, peak exactly 1.0** (post-clamp), the only gain stage being `np.clip` at `sherpa.py:624` and `enroll.py:202`. Compounded by `asr_rule2_min_trailing_silence=0.8` fragmenting one utterance into tiny garbage finals, empty `asr_hotwords` (no biasing), and **no ASR-confidence signal** so garbage finals reach the LLM unfiltered. **(Ranks 3, 5, 6; resampler rank 10 measured small)**

**Highest-leverage fix:** Replace the hard clip with a soft limiter / one-pole AGC (both call sites) and add a clip-rate metric; then raise rule2, populate hotwords, and surface `ys_probs` confidence to gate garbage.

### Problem 3 — Answers are low-quality
**Root cause (verified):** `gemma3:4b` answered all 23 turns; the 12b never ran. `HeuristicRouter.score` (routing.py:222-262, threshold 0.5) scores short spoken queries ~0.0-0.18; `IntentKind.ASSISTANT` contributes 0; the ReAct `escalate` path is dead (`agent.planner.enabled=false`); `DEFAULT_SYSTEM` permits chatty confabulation ('Bob, that's a friendly name!', 'Darklift is a self-driving car…'); and `memory.recall_enabled=false` so the model has zero grounding. **(Ranks 2, 4, 7)**

**Highest-leverage fix:** Rewrite `DEFAULT_SYSTEM` to abstain on garbled/ambiguous input and stop persona chatter (zero-latency), then lower the router threshold + lean `ASSISTANT` to main, and turn on session recall.

---

## Phased roadmap (sequenced for biggest felt improvement first)

### Phase A — Restore the dead feature + cheap wins (do first; days)
1. **(Rank 1, S) Decouple barge-in from speaker-ID** — `core/engines/sherpa.py`. Restores barge-in from 0→working with zero new deps. *Validate: new test (enrolled + `speaker_gate_input=false` + loud over playback ⇒ fires); update `test_barge_in_suppression.py:156`; replay WAV.*
2. **(Rank 2, S) Harden `DEFAULT_SYSTEM` + make it config-overridable** — `core/capabilities.py`, `core/app.py`, `core/runtime.py`, `config.json`. Abstain on garbled input, kill persona chatter, bias temp 0.3-0.5. Verify Gemma-3 system text actually reaches both Ollama and llama.cpp. *Validate: prompt test; replay WAV — 'Bob'/'Six'/'darklift' no longer get filler/confabulation.*
3. **(Rank 3, M) Soft-limiter / AGC replacing the hard clip** — `core/engines/sherpa.py:624`, `core/enroll.py:202`, `core/metrics.py`, persist `input_gain` to `config.local.json`. *Validate: numpy THD/crest test; clip-fraction metric → ~0; WER harness before/after.*

### Phase B — Intelligence routing + STT signal (next; ~1 week)
4. **(Rank 4, M) Make tier escalation fire** — `core/routing.py` (ASSISTANT base score / phrasing rule), `config.json` (threshold ~0.3 desktop), decouple main-tier choice from the planner, give fast tier its own `num_predict` in `core/llm_factory.py`. *Validate: `tests/test_core_routing.py`; replay shows 12b on reasoning turns.*
5. **(Rank 5, S) Hotwords + raise rule2** — `config.json`, `config.local.json`, `core/engines/_sherpa_models.py`; add a startup ASR-model log line. *Validate: WER harness; pin rule2 in `test_device_profiles.py`.*
6. **(Rank 6, M) ASR-confidence (`ys_probs`) + low-content clarify gate** — `core/engines/sherpa.py`, `core/runtime.py` `_on_final`, `core/intents.py`, `core/metrics.py`, `core/runlog.py`. Reject <2-word / mostly-non-dictionary / low-min-token finals with a canned 'didn't catch that' (no LLM). *Validate: lexical-gate unit test against actual transcript finals; calibrate threshold on the WAV.*
7. **(Rank 7, S) Enable session recall + feed cleaned text** — `config.json`, `core/capabilities.py`, `core/runtime.py`, `always_on_agent/memory.py`. Stopword-aware overlap scoring; gate PII out of cloud per §9.7. *Validate: two-turn name-recall test; log inject-vs-empty.*

### Phase C — Observability + measurement backbone (parallel; ~1 week)
8. **(Rank 8, M) Quality stuck_hints + barge-in/ASR diagnostics + true SPEECH_END** — `core/runlog.py`, `core/engines/sherpa.py`, `core/metrics.py`, `core/watchdog.py`. Three new hints (barge-in-never-fired, fast-tier-only, endpoint_latency==0); log barge-in rejection reasons; route swallowed metric strings to a counter. *Validate: re-run THIS summary.json through `RunSummary` and assert all three hints fire.*
9. **(Rank 9, L) WER regression harness** — `core/wer.py`, `tools/bench/runner.py`, `tools/bench/report.py`, `tests/test_replay_recorded.py`, `tests/fixture_audio/` sidecar refs seeded from this session. Gates every STT change. *Validate: produces a WER number on the WAV; pinned-ceiling pytest.*

### Phase D — Hardening + structural (later)
10. **(Rank 10, M, low impact) Anti-aliased resample** — `core/engines/sherpa.py:202`. Correctness/hardening only (measured ~1 dB); gate behind the WER harness. *Validate: aliased-tone unit test; no WER regression.*
11. **(Rank 11, L) Move ASR decode + addressing/cleaner off the capture thread; greedy fallback on overrun; surface recorder drops** — `core/engines/sherpa.py`, `core/runtime.py`, `core/recorder.py`, `core/runlog.py`. Prevents PortAudio overflow dropping mic audio mid-utterance. *Validate: concurrency test injecting a slow fast-LLM call; 0 drops in summary.*
12. **(Rank 12, M) CLI escape hatches** — `core/app.py`, `core/enroll.py`, `core/engines/sherpa.py`: `--no-gate`, `--speaker-threshold`, `--check-enrollment` (prints live cosine) + startup gate self-check WARNING. Turns a silent multi-turn failure into a 10-second diagnosis. *Validate: CLI test; manual `--check-enrollment` prints cosine.*

### Roadmap-only (not scheduled — needs an architecture change)
- **Stronger ASR via offline-chunked engine** (NVIDIA streaming Parakeet/Nemotron-0.6B ~8.2% WER, or Moonshine v2 ~6.65% medium-streaming). sherpa-onnx has **no true-streaming Parakeet** today (k2-fsa issues #2918/#3454); requires an `OfflineRecognizer` + VAD-chunked `AudioEngine` that breaks the current `on_partial`/barge-in/KWS-alongside-ASR design. Defer until Phase A-C prove the front-end is exhausted — gate any swap behind the WER harness (rank 9).
- **On-device AEC** (DTLN-aec loopback, 1.8M params ONNX, consumes the `_note_playback_level` reference) so `barge_in_output_margin_db` can drop ~6→2-3 dB and recover sensitivity to soft interrupts. The durable full-duplex fix for the no-AEC open-speaker setup.
- **Pipecat Smart Turn v3** semantic end-of-turn (8M ONNX, ~12ms CPU) layered on Silero VAD to trim the dead-air tail without truncating mid-thought (complements, does not replace, barge-in).
- **§9.7 egress gate on the cloud LLM path** — add a `may_leave_device(query, mode, intent)` check before streaming a recall/PII-injected prompt to a cloud chain (web.search already gates correctly; the LLM path does not).

---

## Sequencing rationale
Rank 1 restores a **completely dead feature** at S effort with no dependencies — the single biggest felt improvement for the barge-in complaint. Ranks 2-3 are cheap and attack the two most visible remaining symptoms (confident nonsense + clipped audio) before any measurement infrastructure exists. The WER harness (rank 9, Phase C) is intentionally *not* first because the Phase A/B fixes (clip, hotwords, rule2, confidence gate) are independently justified by the recorded artifacts and don't need a WER number to ship — but every STT change from rank 5 onward should be re-measured through it. The model swap was refuted, so effort goes to the input chain instead.
