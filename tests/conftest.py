"""pytest session configuration.

Ensures the repository root is importable so tests can ``import core``,
``always_on_agent``, ``utils.memory``, and ``tests.sandbox``.

It also writes a committable log of every local test run under
``logs/tests/`` -- a full DEBUG ``.txt`` capturing all ``speaker.*`` logging
during the session, plus a ``.summary.json`` digest (counts, failures, and the
slowest tests). Push those files and they show exactly what happened.
Disable with ``SPEAKER_TEST_LOG=0``.

Custom flags:

- ``--postgres``: enable the ``postgres`` marker so
  ``tests/test_memory_postgres_integration.py`` actually runs (requires
  ``pg_ctl`` on PATH + pgvector installed). Without it, those tests are
  collected but skip with a clear reason -- so a developer without a local
  Postgres setup still sees what they're missing.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ENABLED = os.environ.get("SPEAKER_TEST_LOG", "1") != "0"
_LOG_DIR = Path(__file__).resolve().parent.parent / "logs" / "tests"
_RESULTS: list[dict] = []
_STATE: dict = {}


def pytest_addoption(parser):
    parser.addoption(
        "--postgres",
        action="store_true",
        default=False,
        help="run tests marked @pytest.mark.postgres (needs pg_ctl + pgvector)",
    )


def pytest_collection_modifyitems(config, items):
    """Gate side-effectful / opt-in test tiers behind an explicit switch:

    - ``postgres`` tests need ``--postgres`` (pg_ctl + pgvector).
    - ``live_output`` tests MAKE SOUND on the real speakers/mic (Tier 3). They
      are skipped unless ``SPEAKER_LIVE=1`` is in the env, so a bare ``pytest``,
      CI's ``tests.yml``, and the ``unit``/``fast`` stages NEVER play audio --
      even if a ``live_output`` test is collected. Run them on purpose with
      ``python tools/run_tests.py live`` (which sets the env after a preflight).
    """
    want_pg = config.getoption("postgres")
    want_live = os.environ.get("SPEAKER_LIVE", "").strip().lower() not in ("", "0", "false", "no")
    skip_pg = pytest.mark.skip(reason="requires --postgres (pg_ctl + pgvector)")
    skip_live = pytest.mark.skip(
        reason="requires SPEAKER_LIVE=1 + real speakers/mic (run: python tools/run_tests.py live)"
    )
    for item in items:
        if not want_pg and "postgres" in item.keywords:
            item.add_marker(skip_pg)
        if not want_live and "live_output" in item.keywords:
            item.add_marker(skip_live)


def pytest_configure(config):
    # Hermetic tests: ignore the machine-local config.local.json overlay so a
    # dev box with real model paths behaves like CI (empty sherpa paths ->
    # `--engine sherpa` fails fast instead of starting the live loop and
    # hanging). Honoured by core.app._load_config. Override with =0 to opt in.
    os.environ.setdefault("SPEAKER_NO_LOCAL_CONFIG", "1")
    # Register the marker so the strict-markers mode doesn't complain.
    config.addinivalue_line(
        "markers",
        "postgres: integration tests that need a real PostgreSQL with pgvector",
    )
    if not _ENABLED:
        return
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    txt_path = _LOG_DIR / f"tests-{run_id}.txt"
    handler = logging.FileHandler(txt_path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)-5s %(name)s | %(message)s", "%H:%M:%S")
    )
    speaker = logging.getLogger("speaker")
    speaker.setLevel(logging.DEBUG)
    speaker.addHandler(handler)
    _STATE.update(
        run_id=run_id,
        txt_path=str(txt_path),
        summary_path=str(_LOG_DIR / f"tests-{run_id}.summary.json"),
        handler=handler,
        started=time.time(),
    )


def pytest_runtest_logreport(report):
    # One report per phase; record the call phase (and setup failures).
    if not _ENABLED:
        return
    if report.when == "call" or (report.when == "setup" and report.outcome == "failed"):
        _RESULTS.append(
            {
                "test": report.nodeid,
                "outcome": report.outcome,
                "duration_sec": round(report.duration, 3),
                "longrepr": str(report.longrepr)[:2000] if report.failed else None,
            }
        )


def pytest_sessionfinish(session, exitstatus):
    if not _ENABLED or "run_id" not in _STATE:
        return
    passed = sum(1 for r in _RESULTS if r["outcome"] == "passed")
    failed = [r for r in _RESULTS if r["outcome"] == "failed"]
    skipped = sum(1 for r in _RESULTS if r["outcome"] == "skipped")
    slowest = sorted(_RESULTS, key=lambda r: r["duration_sec"], reverse=True)[:10]
    summary = {
        "run_id": _STATE["run_id"],
        "log_path": _STATE["txt_path"],
        "exit_status": int(exitstatus),
        "duration_sec": round(time.time() - _STATE["started"], 2),
        "counts": {
            "total": len(_RESULTS),
            "passed": passed,
            "failed": len(failed),
            "skipped": skipped,
        },
        "failures": [
            {"test": r["test"], "duration_sec": r["duration_sec"], "longrepr": r["longrepr"]}
            for r in failed
        ],
        "slowest": [{"test": r["test"], "duration_sec": r["duration_sec"]} for r in slowest],
    }
    Path(_STATE["summary_path"]).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.getLogger("speaker").removeHandler(_STATE["handler"])
    _STATE["handler"].close()
    print(f"\n[test-log] {_STATE['txt_path']}")
    print(f"[test-log] {_STATE['summary_path']}")
