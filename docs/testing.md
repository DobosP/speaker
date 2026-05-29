# Testing guide

How to run the suite, which subset to run when, and what the markers mean. The
suite is pure logic + device simulation -- no audio hardware, no models, no
network -- so it runs anywhere `pip install -r requirements-dev.txt` succeeds.

## Install the test deps

```
pip install -r requirements-dev.txt   # pytest + pytest-xdist + pytest-timeout + psycopg
```

This is the dev/test extra on top of the lean runtime `requirements.txt`; you do
not need it to run the assistant, only to run the tests + `tools/run_tests.py`.

## The fast loop (everyday TDD, ~2-4s)

```
python tools/run_tests.py fast        # the `fast` stage
# or, plain pytest:
pytest -m "not slow and not network and not llm and not backend"
```

This skips the timing/concurrency simulations (the slow ones live in
`tests/test_sandbox_middle_layer.py` and `tests/test_hedge_chain_advanced.py`)
and anything needing a model/network/backend. Use it while iterating.

## The full run

```
pytest tests                          # serial, ~35s
pytest -n auto                        # parallel, ~9s on a many-core box
```

The suite is **parallel-safe** -- `-n auto` (pytest-xdist) gives the same pass
count as serial, roughly 4x faster. Parallel is deliberately *opt-in*: it is not
baked into `addopts`, because a serial run is far easier to debug. Via the staged
runner, forward it through `--pytest-arg` (one flag per token):

```
python tools/run_tests.py full --pytest-arg=-n --pytest-arg=auto
```

### Hang guard

`pytest.ini` sets `--timeout=60 --timeout-method=thread`. Any single test that
runs longer than 60s is failed (with a traceback) instead of wedging the run --
the `thread` method is chosen because it tolerates the runtime's app threads. If
you have a legitimately long test, mark it and/or pass a higher `--timeout=N`.

## The deep / end-to-end run

```
python tools/run_tests.py e2e         # only @pytest.mark.e2e (subprocess the real CLI)
python tools/run_tests.py full        # everything
python tools/run_tests.py all-stages  # every stage in order, with per-stage reports
```

`e2e` tests subprocess the real `python -m core` and assert on its behaviour.
They are bounded by the global per-test timeout so they never hang CI. The staged
runner writes per-stage `summary.json` / `failures.json` / `llm-summary.md` plus a
run-level summary under `test-reports/<run_id>/` (with a `latest` pointer).

## Postgres integration tests

The memory integration tests (`tests/test_memory_postgres_integration.py`, marked
`@pytest.mark.postgres`) are collected but **skipped by default**; they need a
real PostgreSQL with pgvector (`pg_ctl` on PATH). Opt in:

```
pytest --postgres
python tools/run_tests.py memory --pytest-arg=--postgres
```

The driver itself (`psycopg[binary,pool]`) ships in both `requirements.txt` and
`requirements-dev.txt`.

## Staged runner

`tools/run_tests.py` groups the suite into named stages with structured reports:

```
python tools/run_tests.py list   # show every stage + its purpose
```

| Stage | What it covers |
| --- | --- |
| `core` | Runtime, action brain, brain logic (fast, no models). |
| `sandbox` | Realistic-timing / concurrency middle-layer tests (the slow sims). |
| `memory` | Smart-memory save/writer logic + pool contract (`--postgres` for integration). |
| `cloud` | Cloud LLM middle layer: providers, hedge chain, routing, integration. |
| `imports` | Whole-tree import smoke -- does the code compile and are libs present. |
| `fast` | Quick dev subset: tree minus `slow`/`network`/`llm`/`backend`. |
| `e2e` | Full end-to-end CLI/process tests (subprocess the real `python -m core`). |
| `full` | The entire suite. |

## Marker taxonomy

Markers are registered in `pytest.ini` (`--strict-markers` is on, so an
unregistered marker is an error). Select with `-m`, e.g. `-m "e2e"` or
`-m "not slow"`.

| Marker | Meaning |
| --- | --- |
| `smoke` | Fastest import/config/schema checks. |
| `dev` | Critical fast tests for everyday TDD. |
| `audio` | Deterministic audio, VAD, barge-in, replay, conversation tests. |
| `recorded` | Recorded-session replay tests. |
| `discovery` | Failure-discovery tests that may intentionally fail until a bug is fixed. |
| `backend` | Optional backend/model integration tests. |
| `slow` | Tests noticeably slower than the dev stage (timing/concurrency sims). |
| `hardware` | May require real audio hardware or host devices. |
| `network` | May require network access or downloaded assets. |
| `llm` | Require a local or remote LLM service. |
| `e2e` | Full end-to-end CLI/process tests (subprocess the real `python -m core`). |
| `postgres` | Integration tests needing a real PostgreSQL with pgvector (`--postgres`). |
