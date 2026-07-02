> **SUPERSEDED (2026-07-02):** historical — dormant since its single 2026-05-28
> entry; the swarm loop it logged is retired (see `.agents/README.md` banner,
> `docs/adr/0007`). The live ledger is `.agents/backlog.md`; shipped work is
> recorded in `STATUS.md` + `docs/session_*.md` handoffs.

# Swarm changelog

One entry per shipped (green + pushed) version.

## v1 — 2026-05-28 — Hermetic test suite + swarm scaffold
- **Fix (P0):** test suite hung on this machine because `config.local.json`
  supplies real sherpa model paths, so `test_sherpa_without_models_fails_fast`
  started the live capture loop instead of failing fast. Added a
  `SPEAKER_NO_LOCAL_CONFIG` guard to `core/app._load_config` and set it
  session-wide in `tests/conftest.py`; the merge unit-test opts out so it still
  exercises the overlay. Result: **613 passed, 7 skipped** with the real
  `config.local.json` in place (was: hang).
- **New:** `tools/swarm/harness.py` (build/test/perf engine) + `.agents/`
  coordination ledger (README, backlog, changelog, status).
- Baseline harness verdict recorded in `.agents/last_report.json`.
