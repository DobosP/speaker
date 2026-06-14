"""Postgres-tier producers for the continuity-completion slice.

A fake pool routes canned rows by SQL and records every ``execute`` so no real
PostgreSQL is needed; ``utils.memory`` imports psycopg/numpy at load, so these
self-skip without them. Covers:

* **lm-5** -- ``persist_assistant`` folds 'assistant' into ``persist_roles``;
  an assistant final is persisted (``source='assistant_final'``) and admitted to
  ``search_memory`` (``role = ANY``) ONLY when enabled (default OFF = no write,
  no read -> recall byte-identical).
* **lm-2 Wire 3** -- ``last_session_summary()`` returns the SEEDED head, immune
  to later rolling-summary mutation, and '' when continuity is OFF.
* **Recall-B** -- ``get_profile_context()`` renders the profile block alone, and
  '' when profiles are disabled.
"""
from __future__ import annotations

import contextlib

import pytest

pytest.importorskip("numpy")
pytest.importorskip("psycopg")

import numpy as np

from utils.memory import MemoryManager


class _FakeCursor:
    def __init__(self, responses, log):
        self._responses = responses
        self._log = log
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=()):
        self._log.append((sql, params))
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
    def __init__(self, responses, log):
        self._responses = responses
        self._log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self, *, row_factory=None):
        return _FakeCursor(self._responses, self._log)


class _FakePool:
    def __init__(self, responses=(), *, log=None, **kw):
        self._responses = list(responses)
        self.log = log if log is not None else []
        self.closed = False

    @contextlib.contextmanager
    def connection(self):
        yield _FakeConn(self._responses, self.log)

    def close(self):
        self.closed = True


def _manager(responses=(), *, log=None, **kw):
    pool = _FakePool(responses, log=log)
    m = MemoryManager(
        db_url="postgresql://fake",
        enable_embeddings=False,
        smart_save=False,  # no writer thread
        pool_factory=lambda **_: pool,
        **kw,
    )
    return m, pool


# --- lm-5: persist_roles wiring ---------------------------------------------


def test_persist_assistant_folds_into_persist_roles():
    m, _ = _manager(persist_assistant=True)
    try:
        assert "assistant" in m.persist_roles
    finally:
        m.close()


def test_persist_assistant_default_off_user_only():
    m, _ = _manager()  # default
    try:
        assert "assistant" not in m.persist_roles
        assert "user" in m.persist_roles
    finally:
        m.close()


# --- lm-5: assistant final persistence --------------------------------------


def test_assistant_final_persisted_when_enabled():
    log: list = []
    m, _ = _manager(log=log, persist_assistant=True)
    m._schedule_background = lambda fn: fn()  # run inline for determinism
    try:
        m.add_message("assistant", "the answer is 42")
        inserts = [(s, p) for (s, p) in log if "INSERT INTO messages" in s]
        assert inserts, "assistant final was not persisted"
        _sql, params = inserts[0]
        assert "assistant" in params       # role column
        assert "assistant_final" in params  # source column
        assert "the answer is 42" in params
    finally:
        m.close()


def test_assistant_final_not_persisted_by_default():
    log: list = []
    m, _ = _manager(log=log)  # persist_assistant default OFF
    m._schedule_background = lambda fn: fn()
    try:
        m.add_message("assistant", "the answer is 42")
        inserts = [s for (s, _p) in log if "INSERT INTO messages" in s]
        assert not inserts, "assistant final persisted while disabled"
    finally:
        m.close()


# --- lm-5: search admits assistant rows only when enabled -------------------


def _prep_search(m):
    """Make search_memory runnable without a real embedder."""
    m._embeddings_available = True
    m._get_embedding = lambda text: np.zeros(384, dtype="float32")
    m._check_embedding_dim = lambda *a, **k: None


def test_search_memory_admits_assistant_when_enabled():
    log: list = []
    m, _ = _manager(log=log, persist_assistant=True)
    _prep_search(m)
    try:
        m.search_memory("anything", limit=3)
        msg_q = [p for (s, p) in log if "FROM messages" in s and "role = ANY" in s]
        assert msg_q, "messages query (role = ANY) not issued"
        # param order: (qemb, embedder_id, dim, recall_roles, qemb, limit)
        assert msg_q[0][3] == ["user", "assistant"]
    finally:
        m.close()


def test_search_memory_user_only_by_default():
    log: list = []
    m, _ = _manager(log=log)  # default OFF
    _prep_search(m)
    try:
        m.search_memory("anything", limit=3)
        msg_q = [p for (s, p) in log if "FROM messages" in s and "role = ANY" in s]
        assert msg_q, "messages query not issued"
        assert msg_q[0][3] == ["user"]
    finally:
        m.close()


# --- lm-2 Wire 3: last_session_summary snapshot -----------------------------


def _is_summary(s):
    return "FROM summaries" in s


def test_last_session_summary_snapshot_immune_to_rolling_summary():
    m, _ = _manager(
        [(_is_summary, [{"summary": "PRIOR SESSION DIGEST"}])],
        cross_session_continuity=True,
    )
    try:
        assert m.last_session_summary() == "PRIOR SESSION DIGEST"
        # A mid-session rolling-summary write overwrites _summary_head; the
        # one-shot recap snapshot must NOT follow it.
        with m._summary_lock:
            m._summary_head = "NEW MID-SESSION SUMMARY"
        assert m.last_session_summary() == "PRIOR SESSION DIGEST"
    finally:
        m.close()


def test_last_session_summary_empty_when_continuity_off():
    m, _ = _manager([(_is_summary, [{"summary": "PRIOR SESSION DIGEST"}])])  # off
    try:
        assert m.last_session_summary() == ""
    finally:
        m.close()


# --- Recall-B: get_profile_context ------------------------------------------


def _is_profile(s):
    return "FROM user_profile" in s


_PROFILE_ROWS = [{"key": "name", "value": "Alice"}, {"key": "city", "value": "Berlin"}]


def test_get_profile_context_renders_block_when_enabled():
    m, _ = _manager([(_is_profile, _PROFILE_ROWS)], profile_enabled=True)
    try:
        block = m.get_profile_context()
        assert "=== User Profile ===" in block
        assert "name: Alice" in block
        assert "city: Berlin" in block
    finally:
        m.close()


def test_get_profile_context_empty_when_disabled():
    m, _ = _manager([(_is_profile, _PROFILE_ROWS)])  # profile_enabled default OFF
    try:
        assert m.get_profile_context() == ""
    finally:
        m.close()
