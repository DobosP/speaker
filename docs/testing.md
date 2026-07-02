# Testing guide — one taxonomy, four tiers

Every test is categorized by **one** mechanism: **pytest markers** (declared in
`pytest.ini`, `--strict-markers` on). CI, the staged runner
(`tools/run_tests.py`), and the opt-in real-output tier all select off those
markers — there is no separate path-list / blocklist / `--ignore` scheme to keep
in sync.

A test belongs to **at most one tier**. No tier marker = **Tier 0**, the default,
CI-safe, logic-only set (the bulk of the suite).

| Tier | Marker | What it is | Sound? | Needs models? | In CI? | Run it |
|---|---|---|---|---|---|---|
| **0 unit/logic** | *(none)* | pure in-process logic + fakes | no | no | **yes — the gate** | `python tools/run_tests.py unit` *(alias `fast`)* |
| **1 integration/sim** | `slow`, `e2e` | realistic-timing sandbox; `e2e` subprocesses real `python -m core` | no | no | yes (full run) | `… sandbox` / `… e2e` |
| **2 real-model** | `real_model` (+`recorded`) | real weights over **fixtures, no sound card** (sherpa ASR/TTS, two-pass final, Smart-Turn, DTLN, replay); self-skips without models | no | **yes** | `perf.yml` only | `python tools/run_tests.py real_model` |
| **3 live-output** | `live_output` | **real speakers/mic** — the only tier that makes sound | **yes** | yes | **never** | `python tools/run_tests.py live` |

Cross-cutting markers (orthogonal to tiers): `postgres` (needs `--postgres`),
plus the legacy `network`/`llm`/`backend`/`hardware`/`smoke`/`dev`/`audio`
categories used by the Tier-0 selector.

## How the gates work (so nothing runs by accident)

- **Tier 0 is the default.** A bare `pytest tests` runs only logic needing no
  models/audio/services. Heavier tests `pytest.importorskip(...)` their deps and
  self-skip when absent — that is why CI (which installs no models) stays green.
- **`real_model` self-skips** when the sherpa models aren't on disk, so the tier
  is safe to *collect* anywhere; it only *runs* where models are configured (a
  dev box with `config.local.json`, or `perf.yml` after `tools.setup_models`).
- **`live_output` is double-locked** so it never makes sound by accident:
  (1) the `live_output` marker, and (2) a `conftest.py` skip unless
  **`SPEAKER_LIVE=1`** is in the env. A bare `pytest`, CI, and the `unit`/`fast`
  stages therefore never play audio. `tools/run_tests.py live` is the intended
  entry — it preflights `tools.live_session --check` (models + audio ready), sets
  `SPEAKER_LIVE=1`, then runs `-m live_output`.

## The single entry point — `tools/run_tests.py`

```
python tools/run_tests.py list          # every stage + its purpose
python tools/run_tests.py unit          # Tier 0 — the TDD loop + CI-safe set (alias: fast)
python tools/run_tests.py real_model    # Tier 2 — real weights over fixtures (no sound)
python tools/run_tests.py live          # Tier 3 — REAL speakers/mic (opt-in; preflights + SPEAKER_LIVE=1)
python tools/run_tests.py e2e           # Tier 1 — subprocess the real CLI
python tools/run_tests.py full          # everything (real_model self-skips, live_output stays gated)
python tools/run_tests.py core|sandbox|memory|cloud|imports     # focused subsystem stages
python tools/run_tests.py all-stages    # every stage in order, with per-stage reports
```

Each run writes structured reports under `test-reports/<run_id>/` (per-stage
`summary.json` / `failures.json` / `llm-summary.md` + a run rollup). Plain
`pytest` also logs to `logs/tests/`.

The `imports` stage is a whole-tree import smoke that catches syntax errors and
missing optional libs across `core/`/`always_on_agent/`/`remote/`/`tools/`
before any logic test runs — use it as a "does the code compile and are the
libraries present" preflight.

### Fast loop / full run / parallel

```
python tools/run_tests.py unit          # everyday TDD (Tier 0), a few seconds
pytest tests                            # the default suite, serial
pytest -n auto                          # parallel (pytest-xdist), same pass count, ~4x faster
python tools/run_tests.py full --pytest-arg=-n --pytest-arg=auto
```

Parallel is deliberately opt-in (not in `addopts`) — a serial run is easier to
debug.

### Hang guard

`pytest.ini` sets `timeout=60 timeout_method=thread`: any single test over 60s is
failed (with a traceback) rather than wedging the run; `thread` tolerates the
runtime's app threads. Mark a legitimately long test `slow` and/or raise
`--timeout=N`.

### Postgres integration tests

`@pytest.mark.postgres` tests (memory integration) are collected but **skipped by
default**; they need a real PostgreSQL + pgvector (`pg_ctl` on PATH). Opt in:

```
pytest --postgres
python tools/run_tests.py memory --pytest-arg=--postgres
```

## CI

- **`tests.yml`** (every push/PR) — the Tier-0 gate: `pytest tests` (livekit
  files ignored). Tiers 2/3 self-skip / stay gated, so they never run here.
- **`perf.yml`** (push to `main`, manual, or the `perf` PR label) — the `bench`
  job (real-model latency) **and** a `real_model_tests` job that downloads the
  sherpa models (`tools.setup_models --sense-voice --turn-model --aec-model`) and
  runs `-m "real_model or recorded"`, so ASR/TTS accuracy regressions are caught
  post-merge. Independent of `bench`/`deploy` so it never blocks the perf publish.

## Heavier validators (separate CLIs, run by hand)

The rich real-output drivers — used deliberately, not part of the pytest tiers
(the `live` tier preflights one of them; the `live_output` smoke covers the basic
real-speaker path):

- `python -m tools.live_session --inject|--suite …` — drives the **real**
  ASR→LLM→TTS pipeline over a scripted mock user (`--inject` = no sound card;
  acoustic = over the air).
- `python -m tools.real_usage` — replays recorded run bundles through the **real**
  `OutputStream` (the shutdown-hang regression catcher).
- `python -m tools.bench` — measured real-model latency over fixtures.

## Adding a test

- Logic/fakes only? Add it under `tests/` with **no tier marker** (Tier 0).
- Needs real weights (no speakers)? `@pytest.mark.real_model` — make it self-skip
  when the model is missing (`importorskip` / a `path.exists()` skip).
- Makes sound on real hardware? `@pytest.mark.live_output` — it is gated behind
  `SPEAKER_LIVE=1` automatically by conftest.
- `--strict-markers` is on, so any new marker must be declared in `pytest.ini`.
