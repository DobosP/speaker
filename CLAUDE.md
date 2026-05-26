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
  `AudioEngine` seam), `engines/sherpa.py` (production, on-device),
  `engines/scripted.py` (tests/console), `engines/speaker_gate.py` (speaker-ID
  barge-in gate), `llm.py` (Ollama client + fake), `capabilities.py` (LLM-backed,
  cancellable providers), `runtime.py` (`VoiceRuntime` orchestrator), `app.py`
  (CLI). Run: `python -m core --engine console --llm echo`.
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
- `config.json` — runtime config. `docs/` — architecture and subsystem docs.

> The legacy stack (`main.py`, `utils/audio.py`, the hand-rolled STT/TTS/LLM
> plumbing, `benchmarks/`, `scripts/`, and their tests) was deleted in the
> refactor. Don't try to import them.

## Conventions

- Python, standard `pytest`. Run tests: `python -m pytest tests -q`.
- Run the app: `python main.py --profile mid` (needs `ollama serve` + a pulled model).
- Keep new control-plane logic in `always_on_agent/`, typed and testable, not in `main.py`.
- Prefer replay/transcript tests over tests that require live audio devices.
- Fully-local is a hard product requirement: no cloud STT/LLM/TTS by default.

## Environment / git

- This repo is developed on branch `claude/nice-planck-ZDr90`.
- Web sessions run in an ephemeral container; commit anything worth keeping.
- NOTE: pushes may be blocked if the session was provisioned read-only
  (`403 Permission denied`). If so, surface it — it's an environment permission,
  not a code problem.

## When unsure

Ask clarifying questions before large changes. `docs/PROJECT_KICKOFF.md` is the
running list of product decisions; check it for current intent and open items.
