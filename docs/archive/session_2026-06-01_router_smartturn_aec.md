# Session 2026-06-01 — unified capability router + Smart Turn v3 + on-device AEC

> ⚠️ **Historical record (point-in-time).** Superseded by [`docs/unified_architecture.md`](unified_architecture.md) and the current [`.agents/backlog.md`](../.agents/backlog.md). Kept for history. (2026-06-02 consolidation.)

Cross-machine handoff (this session ran on a **Windows 11** box; continuation may be
on a different machine). Written in-repo because per-user Claude memory does NOT travel
between machines — this doc does. All code below is committed on branch
**`feat/aec-dtln`** (pushed to `origin`); `origin/main` was NOT updated by the session
(the harness blocks direct pushes to `main` — finish via a PR `feat/aec-dtln → main`,
or `git push origin main` from a normal terminal).

## Branch / commit map

`feat/aec-dtln` contains the **entire** history (it descends from the two earlier
merges), so checking it out gives you everything:

| Commit | What |
|--------|------|
| `2c2ad40` (+ merge `32c89ab`) | Unified **capability router** + **Smart Turn v3** endpoint |
| `a7176ab` (+ merge `ec1991a`) | **AEC** front-end — NumPy FDAF (`nlms` backend) |
| `24027dc` (+ merge `d1a1554` on local main) | **DTLN-aec** deep ONNX tier (`dtln` backend) |

Local `main` has all three merged (`d1a1554`); that merge bubble is local-only —
`origin/main` catches up when the PR is merged.

## What landed (all OFF by default unless noted)

1. **Unified capability router** — `core/capability_router.py` (the "middle layer").
   One decision per turn: CONTROL / SIMPLE / RESEARCH / ACT + fast/main tier. Heuristic
   floor (reuses `routing.HeuristicRouter` + `react.should_escalate` + `is_stop_command`)
   + optional fast-LLM disambiguation on low-confidence turns (memoized, ≤1 LLM call/turn).
   Backs the existing tier `Router` + `escalate` predicate via adapters → no
   `always_on_agent` changes. Config `capability_router` block; **ENABLED in the
   `desktop` profile**, off elsewhere. Tests: `tests/test_capability_router.py`.

2. **Smart Turn v3 semantic endpoint** — `core/endpointing.py::SmartTurnCompletionDetector`
   (Whisper log-mel + sigmoid ONNX) on the existing `TurnCompletionDetector` seam.
   Dual-gated `sherpa.smart_turn_enabled` + `smart_turn_model`; lazy onnxruntime +
   transformers; fallback to lexical/acoustic. Fetch: `python -m tools.setup_models
   --smart-turn-model`. Model downloaded on the Windows box (gitignored). Tests:
   `tests/test_smart_turn_endpoint.py`.

3. **On-device AEC** — `core/engines/_aec.py`, inserted in `sherpa.py` after resample /
   before denoise. `FarEndRing` (played-TTS ring, fed by the playback thread, read at
   `aec_ref_delay_ms`) → `EchoCanceller` (passthrough-on-error). Two backends:
   - **`nlms`** (default): a dependency-free NumPy frequency-domain adaptive filter
     (FDAF) with double-talk freeze. ~20 dB ERLE on synthetic single-talk.
   - **`dtln`**: the DTLN-aec deep tier under onnxruntime (two-stage, 512/128, LSTM
     state carried block-to-block). ~31 dB ERLE on synthetic single-talk. Converted
     from `breizhn/DTLN-aec` tflite via `tools.setup_models --aec-model` (needs tf2onnx
     + tensorflow-cpu at convert time; runtime needs only onnxruntime).
   When AEC is on, the barge-in gate uses `aec_relaxed_margin_db` (3 dB) instead of the
   6 dB no-AEC guard. Tests: `tests/test_aec_seam.py`.

Full suite at session end: **1202 passed, 10 skipped.**

## Environment on the Windows box (won't fully transfer)

- venv (`.venv`, pyenv py3.10): added `soxr`, `onnxruntime` (1.23.2; pin `>=1.20,<1.24`
  for cp310), `transformers` (Smart Turn / DTLN), and **dev-time** `tf2onnx` +
  `tensorflow-cpu` (only for the DTLN tflite→onnx conversion; NOT a runtime dep).
- Machine-local, gitignored, regenerate per machine:
  - Smart Turn model: `python -m tools.setup_models --smart-turn-model`
  - DTLN-aec ONNX: `python -m tools.setup_models --aec-model` (downloads tflite +
    converts; needs tf2onnx + tensorflow-cpu).
- Speaker-ID is **still not enrolled** on the Windows box — `python -m core --enroll`.

## Next steps (pick up here)

1. **Enable + validate AEC on real hardware** (needs the mic). In `config.local.json`
   → `sherpa`: `aec_enabled=true`, start with `aec_backend="nlms"`. **Calibrate
   `aec_ref_delay_ms`** (speaker→mic delay) with `tools/echo_probe.py`. Live barge-in
   run: confirm the assistant no longer self-interrupts AND a real interrupt still cuts
   through. Then optionally flip to `aec_backend="dtln"` (+ `aec_model=<repo>/
   pretrained_models/sherpa/aec`) for the stronger cancellation.
2. **Extend `tools/echo_probe.py`** to print post-AEC **ERLE (dB)** and auto-suggest
   `aec_ref_delay_ms` via cross-correlation — makes calibration one command. (Designed,
   not yet built — the natural next code piece, no mic needed to write it.)
3. **Validate the Smart Turn endpoint** before enabling: `python -m tools.live_session
   --all --inject --smart-endpoint`, diff ON finals vs lexical/acoustic.
4. **DTLN follow-ups:** a smaller size (256/128) for phone profiles; clock-drift handling
   over long utterances; consider LiveKit's bundled AEC3 if the runtime ever moves to
   ≥3.11 (the classic-DSP route the research flagged as gated behind 3.11).

Prior context: `docs/archive/session_2026-05-31_acoustic_real_voice.md`,
`docs/archive/voice_improvement_plan_2026-05.md`. The capability-router/AEC research reports
(with the production-systems survey + verified library findings) were produced via
workflows this session; their conclusions are reflected in the code + this doc.
