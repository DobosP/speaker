"""lm-2 cross-session continuity (Postgres tier).

A fake pool routes canned rows by SQL so no real Postgres is needed; but
utils.memory imports psycopg (dict_row) at module load, so these self-skip
without it. Default-OFF neutrality is the key contract: with the flag off, a
fresh process loads NOTHING (byte-identical to the prior behavior).
"""
from __future__ import annotations

import contextlib

import pytest

pytest.importorskip("numpy")
pytest.importorskip("psycopg")

from utils.memory import MemoryManager


class _FakeCursor:
    def __init__(self, responses):
        self._responses = responses
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=()):
        self._rows = []
        for pred, rows in self._responses:
            if pred(sql):
                self._rows = list(rows)
                return

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _FakeConn:
    def __init__(self, responses):
        self._responses = responses

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self, *, row_factory=None):
        return _FakeCursor(self._responses)


class _FakePool:
    def __init__(self, responses, **kw):
        self._responses = responses
        self.closed = False

    @contextlib.contextmanager
    def connection(self):
        yield _FakeConn(self._responses)

    def close(self):
        self.closed = True


def _manager(responses, *, continuity):
    return MemoryManager(
        db_url="postgresql://fake",
        enable_embeddings=False,
        smart_save=False,  # no writer thread in tests
        pool_factory=lambda **kw: _FakePool(responses),
        cross_session_continuity=continuity,
    )


_PRIOR_SUMMARY = [{"summary": "PRIOR SESSION DIGEST"}]
_CROSS_ROWS = [  # newest-first, as the DESC query returns
    {"role": "user", "content": "my dog is rex", "timestamp": 2.0},
    {"role": "user", "content": "i live in berlin", "timestamp": 1.0},
]


def _is_summary(s):
    return "FROM summaries" in s


def _is_session_scoped(s):
    return "session_id = %s" in s and "role = 'user'" in s


def _is_cross_session(s):
    return "role = 'user'" in s and "session_id" not in s


def test_continuity_on_seeds_summary_head_and_loads_cross_session():
    responses = [
        (_is_summary, _PRIOR_SUMMARY),
        (_is_session_scoped, []),       # fresh process: no current-session rows
        (_is_cross_session, _CROSS_ROWS),
    ]
    m = _manager(responses, continuity=True)
    try:
        assert m._summary_head == "PRIOR SESSION DIGEST"  # Wire 1
        contents = {msg.content for msg in m.recent_messages}  # Wire 2
        assert contents == {"my dog is rex", "i live in berlin"}
    finally:
        m.close()


def test_continuity_off_is_byte_identical_default():
    """Default OFF: a fresh process seeds nothing and loads nothing -- the
    opening-turn context is unchanged from before lm-2."""
    responses = [
        (_is_summary, _PRIOR_SUMMARY),
        (_is_session_scoped, []),
        (_is_cross_session, _CROSS_ROWS),
    ]
    m = _manager(responses, continuity=False)
    try:
        assert m._summary_head == ""             # Wire 1 not run
        assert m.recent_messages == []           # Wire 2 fallback not run
    finally:
        m.close()


def test_continuity_on_prefers_current_session_when_present():
    """The cross-session fallback fires ONLY when the current session is empty;
    a live session's own rows take precedence (no stale mixing)."""
    responses = [
        (_is_summary, []),
        (_is_session_scoped, [{"role": "user", "content": "current turn", "timestamp": 3.0}]),
        (_is_cross_session, [{"role": "user", "content": "OLD cross row", "timestamp": 1.0}]),
    ]
    m = _manager(responses, continuity=True)
    try:
        contents = [msg.content for msg in m.recent_messages]
        assert contents == ["current turn"]
        assert "OLD cross row" not in contents
    finally:
        m.close()
