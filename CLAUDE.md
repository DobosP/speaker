# CLAUDE.md

Guidance for Claude Code working in this repo. Read this first every session.

## What this project is

A fully-local, real-time voice assistant (`ASR → LLM → TTS`) with barge-in.
Today it runs on desktop (Linux/Windows/macOS) in Python. The **goal** is a
fully on-device, fully-local, always-listening, mode-based assistant that also
runs on Android and iOS.

> Direction: the refactor is underway. The target architecture and the
> keep/replace/delete plan live in **`docs/target_architecture.md`** — read it
> before proposing structural changes. Short version: the hand-rolled audio
> stack has been **removed** in favour of `sherpa-onnx`; the `always_on_agent/`
> "brain" is kept and made real; the old `main.py` monolith is **deleted**.
>
> **The runtime now lives in `core/`** (`VoiceRuntime`): a swappable
> `AudioEngine` (sherpa-onnx for production, a scripted engine for tests) wired
> to the `always_on_agent` brain with real LLM-backed, cancellable capabilities.
> Try it without audio: `python -m core --engine console --llm echo`.

## Layout

- `core/` — **the runtime (all new work goes here).** `engine.py` (the
  `AudioEngine` seam), `engines/sherpa.py` (production, on-device; CPU STT/TTS
  with auto-tuned threads + explicit `provider`), `engines/scripted.py`
  (tests/console), `engines/speaker_gate.py` (speaker-ID barge-in gate),
  `llm.py` (the `LLMClient` protocol + `EchoLLM` fake, `OllamaLLM` for desktop
  GPU, `LlamaCppLLM` for on-device GGUF; all accept optional `images=` for
  multimodal Gemma 3), `capabilities.py` (LLM-backed cancellable providers;
  two-model split — fast model answers, main/multimodal model researches),
  `runtime.py` (`VoiceRuntime` orchestrator), `app.py` (CLI; builds models from
  the `llm` config block and applies the selected device profile). Run:
  `python -m core --engine console --llm echo`.
- `always_on_agent/` — the **control-plane "brain"** (modes, priority event bus,
  supervisor, cancellable threaded tasks, intent analyzer). The keeper. See its
  `README.md` and `docs/always_on_agent_layer.md`.
- `utils/memory.py` (+ `memory_writer.py`, `memory_config.py`) — Postgres-backed
  smart memory (the only surviving `utils/` modules). See `MEMORY.md`. Keep;
  will move to SQLite on mobile.
- `tests/` — pytest. `tests/sandbox/` is the device-simulation harness
  (latency/LLM-weight profiles + simulated engine/LLM) for middle-layer tests;
  `test_core_runtime.py` is fast logic; `test_sandbox_middle_layer.py` is
  realistic-timing/concurrency. No audio/model deps.
- `tools/` — dev tooling (no app code). `run_tests.py` + `testing/` (staged
  pytest runner with reports under `test-reports/`); `specsim/` (machine-spec
  simulator that renders an HTML capability report — see Conventions).
- `config.json` — runtime config. `docs/` — architecture and subsystem docs.

> The legacy stack (`main.py`, `utils/audio.py`, the hand-rolled STT/TTS/LLM
> plumbing, `benchmarks/`, `scripts/`, and their tests) was deleted in the
> refactor. Don't try to import them.

## Conventions

- Python, standard `pytest`. Run tests: `python -m pytest tests -q`. For staged
  runs with structured reports (per-stage + a tabular run summary under
  `test-reports/`), use `python tools/run_tests.py list|core|sandbox|memory|full`.
- Run the app: `python -m core --engine console --llm echo` (no audio/models);
  `python -m core --engine sherpa` for on-device audio.
- LLM/device config (`config.json`): the `llm` block selects a `backend`
  (`ollama` desktop-GPU, or `llamacpp` on-device GGUF) plus a `main_model`
  (large/multimodal) and `fast_model` (snappy replies). `device_profiles`
  (`desktop`, `phone`, …) are shallow-merged over the base per section; pick one
  with `--device <name>` (default from `config.device`). Desktop runs
  gemma3:12b + 4b on Ollama/GPU; phone runs small Gemma (4b/1b) GGUF on
  llama.cpp with STT/TTS threads dialed down. Ollama is desktop-only — mobile
  must use `llamacpp`.
- Simulate specs without hardware: `python -m tools.specsim` renders
  `test-reports/specsim/index.html` (model-fit + responsiveness matrix + per-
  device ASR→LLM→TTS timelines across 4090/Mac/Windows/phone/web). Numbers are
  modelled estimates, not measurements — calibrate `tools/specsim/specs.py` from
  real runs before trusting absolutes.
- Keep new control-plane logic in `always_on_agent/`, typed and testable, not in `main.py`.
- Prefer replay/transcript tests over tests that require live audio devices.
- Fully-local is a hard product requirement: no cloud STT/LLM/TTS by default.

## Environment / git

- `main` is the integration branch and holds the latest work. Do feature work
  on a short-lived branch and merge back to `main`.
- Web sessions run in an ephemeral container; commit anything worth keeping.
- CI secrets: the repo has an Actions secret **`HF_TOKEN`** (a HuggingFace read
  token, Gemma license accepted) used only by `.github/workflows/publish-model.yml`
  to fetch the gated Gemma 3 model and republish it to the public `gemma-model`
  release that the phone app downloads. The token value lives only in GitHub
  Actions secrets — never commit it to the repo or paste it into files.
- NOTE: pushes may be blocked if the session was provisioned read-only
  (`403 Permission denied`). If so, surface it — it's an environment permission,
  not a code problem.

## When unsure

Ask clarifying questions before large changes. `docs/PROJECT_KICKOFF.md` is the
running list of product decisions; check it for current intent and open items.
