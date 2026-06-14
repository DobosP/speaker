"""Tests for tools/migrate.py -- focus on DSN redaction (no DB, no yoyo).

The ``status`` command used to print the raw ``--database-url``, which can carry
a password (``postgresql://user:pass@host/db``). These tests pin that the
password never reaches stdout while the rest of the DSN stays visible.
"""
from __future__ import annotations

import contextlib
import io

import pytest

from tools import migrate


def test_redact_masks_password():
    out = migrate._redact_db_url("postgresql://dobo:s3cret@localhost:5432/voice_assistant")
    assert "s3cret" not in out
    assert "***" in out
    # Non-secret parts stay readable for operators.
    assert "dobo" in out
    assert "localhost" in out
    assert "voice_assistant" in out


def test_redact_passthrough_when_no_password():
    url = "postgresql:///voice_assistant"
    assert migrate._redact_db_url(url) == url


def test_redact_masks_query_param_password():
    # libpq also accepts the password as a query parameter.
    out = migrate._redact_db_url("postgresql://dobo@localhost/db?password=s3cret&sslmode=require")
    assert "s3cret" not in out
    assert "***" in out
    assert "sslmode=require" in out


def test_redact_never_raises_on_garbage():
    # Redaction must never throw -- a malformed DSN should degrade, not crash.
    assert isinstance(migrate._redact_db_url("not a url"), str)


class _FakeBackend:
    """Minimal stand-in for a yoyo backend: a lock() context manager and an
    empty to_apply() so _cmd_status runs without a real database."""

    @contextlib.contextmanager
    def lock(self):
        yield

    def to_apply(self, _all):
        return []


def test_status_never_prints_password(monkeypatch):
    monkeypatch.setattr(migrate, "_backend", lambda url: _FakeBackend())
    monkeypatch.setattr(migrate, "_read_migrations", lambda _dir: [])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = migrate.main(
            ["--database-url", "postgresql://dobo:s3cret@localhost/voice_assistant", "status"]
        )
    out = buf.getvalue()
    assert rc == 0
    assert "s3cret" not in out
    assert "***" in out
