# Improvement backlog

Priority queue for the swarm loop. `[ ]` = open, `[x]` = shipped (see changelog).
P0 = correctness/blocker, P1 = high value, P2 = nice-to-have.

## P0 — correctness / makes it work on this machine
- [x] **Hermetic test suite** — `config.local.json` real model paths made
  `--engine sherpa` start the live loop and hang the suite. Added
  `SPEAKER_NO_LOCAL_CONFIG` guard in `core/app._load_config`, set session-wide
  in `tests/conftest.py`. *(v1)*

## P1 — desktop / 4090 fit (pending Perf + Architect agents)
- [ ] Adopt `desktop_gpu_4090` profile on this machine (currently `device=desktop`;
      4090 profile raises `num_ctx` 4096→8192, `num_predict`→512, enables both gates).
- [ ] Measure real end-to-end ASR→LLM→TTS latency on the 4090 (`tools.bench --real`),
      calibrate `tools/specsim/specs.py` against it.
- [ ] Confirm sherpa runs CUDA provider (config `sherpa.provider="cpu"` today) —
      evaluate GPU ASR/TTS on the 4090 vs auto-tuned CPU threads (32 logical).

## P1 — architecture (to be filled by the Architect agent)
- [ ] _audit findings land here_

## P2
- [ ] Wire `tools/swarm/harness.py perf --real` into `.github/workflows/perf.yml` parity.
