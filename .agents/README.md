> **SUPERSEDED (2026-07-02):** the 2026-05-28 swarm loop below — including its
> "commit + push green versions to `main`" protocol — is retired by the
> 2026-06-24 fleet git policy (`AGENTS.md`, `docs/adr/0007`): never push, merge
> to `main`, or delete branches without Paul's explicit ask.
> `.agents/backlog.md` remains the **live work queue** and the harness commands
> still work; the roles/loop text is historical.

# Speaker improvement swarm

A small set of role-specialized agents + a self-improving loop that builds,
tests, and tunes the **desktop voice runtime** (`core/`) so it runs well on
**this machine** (i9-13980HX · 32 threads · 30 GiB · RTX 4090 Laptop 16 GB ·
`desktop_gpu_4090` profile). Driven by a human-supervised burst first, then
optionally promoted to a standing `/loop`.

## The engine (`tools/swarm/harness.py`)

One command gives a machine-readable verdict on a code version:

```bash
python -m tools.swarm.harness test    # hermetic CI-parity suite (the green gate)
python -m tools.swarm.harness perf     # latency smoke (--real for the 4090 pipeline)
python -m tools.swarm.harness all      # both; writes .agents/last_report.json
```

`last_report.json.green == true` ⇔ the version is safe. The harness is hermetic
(`SPEAKER_NO_LOCAL_CONFIG=1`) so real model paths in `config.local.json` never
make `--engine sherpa` start the live loop and hang the suite.

## Roles

| Agent | Mandate | Writes? |
|-------|---------|---------|
| **Orchestrator** (the live session) | Picks the next backlog item, spawns agents, integrates results, runs the harness, commits + pushes green versions to `main`. | yes |
| **Architect** | Audits `core/` + `always_on_agent/` against `docs/target_architecture.md` §9; files findings into `backlog.md`. | backlog only |
| **Builder/Fixer** | Keeps the build green: deps, import/build breakage, hermeticity. | code |
| **Tester** | Triages failures, hardens tests, turns real runs into replay regressions. | tests |
| **Perf-tuner** | Measures end-to-end ASR→LLM→TTS latency on the 4090; tunes `desktop_gpu_4090`. | config/perf |

## Loop protocol (one iteration = one "version")

1. Pick the highest-priority unchecked item in `backlog.md`.
2. Implement it (smallest correct change).
3. `python -m tools.swarm.harness all` → must be `green`.
4. If green: commit to `main`, push, append to `changelog.md`, check the item off.
   If red: revert or fix; never leave `main` red.
5. Update `status.json`; repeat.

## Files

- `backlog.md` — prioritized improvement queue (the loop's work source).
- `changelog.md` — one entry per shipped version.
- `status.json` — current iteration, last verdict, machine facts.
- `last_report.json` / `history.jsonl` — harness output (gitignored churn ok to commit `last_report`).
