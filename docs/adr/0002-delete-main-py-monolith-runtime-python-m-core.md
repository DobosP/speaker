# ADR-0002: Legacy `main.py` monolith deleted; runtime is `python -m core`

Date: 2026-05-26
Status: accepted

## Decision
Delete the legacy `main.py` monolith and its hand-rolled stack (`utils/audio.py`,
the bespoke STT/TTS/LLM plumbing, `benchmarks/`, `scripts/`, and their tests).
The desktop runtime is `core/` (`VoiceRuntime`) on sherpa-onnx, launched with
`python -m core` (`--engine {console,sherpa,replay,livekit}`,
`--llm {echo,ollama,llamacpp}`). Never import or re-create the legacy modules.

## Context / why
The refactor (landed in commit `ec7dd31`, 2026-05-26) replaced the hand-rolled
audio stack with sherpa-onnx behind the swappable `AudioEngine` seam and made
the `always_on_agent/` brain real. Keeping the monolith alongside `core/`
would have meant two divergent runtimes and doubled test surface. Why not keep
`main.py` as a wrapper: its CLI flags (`--no-memory`, `--db-url`) were
superseded by `config.json` + the `DATABASE_URL` env var, so a shim would
preserve a dead interface.

## Consequences
- Docs or handoffs that say `python main.py` are stale by definition — fix
  them on sight (SETUP.md/MEMORY.md were cleaned 2026-07-02).
- Gotcha: `python -m core` prunes `logs/runs/` bundles to keep=20, including
  committed ones — see STATUS.md.
- This decision existed only in git history until 2026-07-02; it is now
  written down so no agent resurrects the monolith from old prose.
