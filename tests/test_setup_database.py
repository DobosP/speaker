"""Contract tests for the thinned ``setup_database.py`` wrapper.

`setup_database.py` is no longer the schema owner (Locked Decision 5 / design
step 9-10 / R4). It is reduced to:

- import cleanly with ONLY the declared requirements (psycopg3 — NOT psycopg2),
- create the database (``--create-db``) and verify it (``--verify-only``),
- defer the schema step to ``python -m tools.migrate apply`` via
  ``tools.migrate.main(['apply', '--database-url', ...])``, wrapped in a
  try/except that prints a friendly hint (never a raw traceback / never a
  password) when the optional deps are missing or the migration fails.

These tests are hermetic: no live PostgreSQL, no yoyo-migrations required.
They patch ``tools.migrate.main`` so the wiring is exercised without a DB.
"""
from __future__ import annotations

import importlib
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import pytest

# setup_database imports psycopg3 at module top; skip (not error) where the
# optional driver is absent (CI installs only pytest+numpy).
pytest.importorskip("psycopg", reason="setup_database requires psycopg3 (optional dep)")

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture()
def setup_database():
    """Import the root-level ``setup_database`` module fresh."""
    if "setup_database" in sys.modules:
        del sys.modules["setup_database"]
    return importlib.import_module("setup_database")


# --- R4: import smoke + no psycopg2 -------------------------------------------


def test_imports_with_only_declared_deps(setup_database):
    """Module must import with the declared psycopg3 (no psycopg2)."""
    assert setup_database is not None


def test_no_psycopg2_dependency():
    """psycopg2 was removed from requirements.txt — the module must not use it."""
    src = (_REPO_ROOT / "setup_database.py").read_text(encoding="utf-8")
    assert "psycopg2" not in src
    assert "import psycopg" in src


def test_schema_owning_pieces_deleted(setup_database):
    """The schema SQL constants + setup_tables() must be gone (migrate owns schema)."""
    for gone in (
        "CREATE_TABLES_SQL",
        "CREATE_TEXT_ONLY_TABLES_SQL",
        "CREATE_VECTOR_INDEXES_SQL",
        "MIGRATE_MESSAGES_SQL",
        "setup_tables",
    ):
        assert not hasattr(setup_database, gone), f"{gone} should have been deleted"


def test_retained_helpers_present(setup_database):
    """create_database + verify_setup are retained (ported to psycopg3)."""
    assert callable(setup_database.create_database)
    assert callable(setup_database.verify_setup)
    assert callable(setup_database._redact_db_url)


# --- redaction: never echo the password --------------------------------------


def test_redact_db_url_masks_password(setup_database):
    redacted = setup_database._redact_db_url(
        "postgresql://dobo:s3cret@localhost:5432/voice_assistant"
    )
    assert "s3cret" not in redacted
    assert "***" in redacted
    assert "dobo" in redacted
    assert "localhost" in redacted


def test_redact_db_url_passthrough_when_no_password(setup_database):
    url = "postgresql:///voice_assistant"
    assert setup_database._redact_db_url(url) == url


# --- main() defers the schema step to tools.migrate --------------------------


def _run_main(setup_database, argv):
    """Run main() with a patched argv, capturing stdout."""
    buf = io.StringIO()
    with mock.patch.object(sys, "argv", ["setup_database.py", *argv]):
        with redirect_stdout(buf):
            setup_database.main()
    return buf.getvalue()


def test_main_shells_schema_to_tools_migrate(setup_database):
    """The schema step calls tools.migrate.main(['apply', '--database-url', url])."""
    fake = mock.Mock(return_value=0)
    with mock.patch("tools.migrate.main", fake), mock.patch.object(
        setup_database, "verify_setup", return_value=True
    ):
        out = _run_main(setup_database, ["--db-url", "postgresql:///voice_assistant"])
    fake.assert_called_once()
    call_args = fake.call_args.args[0]
    assert call_args[0] == "apply"
    assert "--database-url" in call_args
    assert "postgresql:///voice_assistant" in call_args
    assert "tools.migrate apply" in out  # canonical path advertised in banner


def test_main_never_echoes_password(setup_database):
    """A password in --db-url must never reach stdout (redacted everywhere)."""
    fake = mock.Mock(return_value=0)
    secret_url = "postgresql://dobo:s3cret@localhost/voice_assistant"
    with mock.patch("tools.migrate.main", fake), mock.patch.object(
        setup_database, "verify_setup", return_value=True
    ):
        out = _run_main(setup_database, ["--db-url", secret_url])
    assert "s3cret" not in out
    # But the migrate call itself receives the real URL (it needs to connect).
    assert secret_url in fake.call_args.args[0]


def test_main_friendly_message_on_missing_yoyo(setup_database):
    """A ModuleNotFoundError from the lazy yoyo import => friendly hint, no traceback."""
    boom = mock.Mock(side_effect=ModuleNotFoundError("No module named 'yoyo'"))
    boom.side_effect.name = "yoyo"
    with mock.patch("tools.migrate.main", boom), mock.patch.object(
        setup_database, "verify_setup", return_value=True
    ) as verify:
        out = _run_main(setup_database, ["--db-url", "postgresql:///voice_assistant"])
    assert "yoyo-migrations" in out
    # On migration failure we bail before verifying.
    verify.assert_not_called()


def test_main_friendly_message_on_migration_failure(setup_database):
    """A generic migrate failure prints a friendly message + the canonical hint."""
    boom = mock.Mock(side_effect=RuntimeError("connection refused"))
    with mock.patch("tools.migrate.main", boom), mock.patch.object(
        setup_database, "verify_setup", return_value=True
    ):
        out = _run_main(setup_database, ["--db-url", "postgresql:///voice_assistant"])
    assert "Schema migration failed" in out
    assert "python -m tools.migrate apply" in out
    # And no raw traceback leaked.
    assert "Traceback" not in out


def test_verify_only_skips_migrate(setup_database):
    """--verify-only must NOT touch tools.migrate at all."""
    fake = mock.Mock(return_value=0)
    with mock.patch("tools.migrate.main", fake), mock.patch.object(
        setup_database, "verify_setup", return_value=True
    ) as verify:
        _run_main(setup_database, ["--db-url", "postgresql:///voice_assistant", "--verify-only"])
    fake.assert_not_called()
    verify.assert_called_once()
