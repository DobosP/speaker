# CLAUDE.md

Guidance for Claude Code working in this repo. Read this first every session.

## What this project is

A fully-local, real-time voice assistant (`ASR → LLM → TTS`) with barge-in.
Today it runs on desktop (Linux/Windows/macOS) in Python. The **goal** is a
fully on-device, fully-local, always-listening, mode-based assistant that also
runs on Android and iOS.

> Direction: we are mid-refactor. The target architecture and the
> keep/replace/delete plan live in **`docs/target_architecture.md`** — read it
> before proposing structural changes. Short version: the hand-rolled audio
> stack is being replaced by `sherpa-onnx`; the `always_on_agent/` "brain" is
> being kept and made real; `main.py` shrinks to a thin adapter.

## Layout

- `main.py` — current monolithic orchestrator (`VoiceAssistant`, ~2,900 lines).
  Being replaced by a thin runtime adapter. Don't add features here.
- `utils/audio.py` — hand-rolled real-time DSP (NLMS AEC, VAD gate, barge-in).
  ~3,000 lines. **Scheduled for replacement** by `sherpa-onnx`. Don't extend it.
- `utils/stt.py`, `utils/llm.py`, `utils/tts*` — STT/LLM/TTS plumbing.
- `utils/memory.py` (+ `memory_writer.py`, `memory_config.py`) — Postgres-backed
  smart memory. See `MEMORY.md`. Keep; will move to SQLite on mobile.
- `always_on_agent/` — the **control-plane "brain"** (modes, priority event bus,
  supervisor, cancellable tasks, intent analyzer). This is the keeper. See its
  `README.md` and `docs/always_on_agent_layer.md`.
- `tests/` — pytest, heavily replay/transcript-driven. `config.json` — runtime
  config (note: most `barge_in_*`/`aec_*` knobs go away post-refactor).
- `docs/` — architecture and subsystem docs.

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
