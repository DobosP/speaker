"""Tests for the P2b memory producers in ``utils.memory`` (R2 + R8).

The logic-suite tests here need NEITHER a real PostgreSQL NOR a live LLM:

* **R2 off-the-bus-thread guard (the headline test).** ``_check_and_summarize``
  runs on the single bus thread (via the supervisor's ``add_message``); an LLM
  summarizer there would stall TTS/barge-in. We assert that ``add_message`` /
  ``queue_user_utterance`` RETURN PROMPTLY and that the (slow, sleeping)
  summarizer is invoked on a DIFFERENT thread -- never inline on the caller.
* **Keyword fallback.** With ``summarizer=None`` the rolling summary still gets
  produced via the legacy topic body, so the layer-2 path exercises end-to-end.
* **Profile gate.** ``get_context_for_llm`` must NOT emit the
  ``=== User Profile ===`` block when ``profile_enabled`` is False (default).

Live-Postgres summary/profile/retention assertions are marked with the existing
``postgres`` marker so they are excluded from the logic run (they need
``--postgres`` + ``pg_ctl`` + pgvector).

A null connection pool (no real DB I/O) is injected so the producer code paths
run without a Postgres -- mirroring ``tests/test_memory_contract.py`` and
``tests/test_memory_pool.py``.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager

import pytest

pytest.importorskip("numpy")

from utils.memory import MemoryManager


# --- a minimal no-DB connection pool ----------------------------------------
#
# With ``enable_embeddings=False`` the manager never runs a real query: search
# short-circuits and the only DB touches (demo-schema bootstrap, warm-start
# SELECT, summary/profile INSERTs) are satisfied by an empty cursor. This lets
# the producer logic run with ``_db_available=True`` but zero real I/O.


class _NullCursor:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=()):
        return None

    def fetchall(self):
        return []

    def fetchone(self):
        return None

    @property
    def rowcount(self):
        return 0


class _NullConn:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self, *, row_factory=None):
        return _NullCursor()


class _NullPool:
    def __init__(self, conninfo=None, *, min_size, max_size, kwargs=None, open=True):
        self.closed = False

    @contextmanager
    def connection(self):
        yield _NullConn()

    def close(self):
        self.closed = True


def _null_pool_factory(*, conninfo, min_size, max_size, kwargs):
    return _NullPool(conninfo, min_size=min_size, max_size=max_size, kwargs=kwargs)


def _make_manager(**overrides) -> MemoryManager:
    """A no-DB manager: null pool + embeddings off + smart_save off, so the
    producer paths run without a real Postgres or background writer."""
    kwargs = dict(
        db_url="postgresql://fake",
        enable_embeddings=False,
        smart_save=False,
        max_context_tokens=1,   # force _check_and_summarize to trigger easily
        max_recent_messages=200,
        pool_min_size=1,
        pool_max_size=5,
        pool_factory=_null_pool_factory,
    )
    kwargs.update(overrides)
    return MemoryManager(**kwargs)


def _fill_until_summary(mgr: MemoryManager, *, sentence: str, count: int = 14) -> None:
    """Add enough user turns to trip the summary threshold (>10 messages and
    over the token budget)."""
    for i in range(count):
        mgr.add_message("user", f"{sentence} number {i} with several extra words here")


# --- R2: rolling summary runs OFF the caller (bus) thread -------------------


def test_add_message_returns_promptly_without_inline_summarizer():
    """R2 key guard: the summarizer is slow (sleeps), yet ``add_message`` must
    return immediately and NEVER invoke the summarizer on the caller's thread --
    the job is scheduled onto a background thread instead."""
    caller_thread = threading.get_ident()
    ran_on: list[int] = []
    summarized = threading.Event()

    def slow_summarizer(text: str) -> str:
        ran_on.append(threading.get_ident())
        time.sleep(0.5)  # would stall the bus thread if called inline
        summarized.set()
        return "rolled summary"

    mgr = _make_manager(summarizer=slow_summarizer)
    try:
        t0 = time.monotonic()
        _fill_until_summary(mgr, sentence="remember my dentist appointment")
        elapsed = time.monotonic() - t0
        # The caller never blocks on the 0.5s summarizer: all 14 adds combined
        # finish well under a single summarizer sleep.
        assert elapsed < 0.4, f"add_message blocked on the summarizer ({elapsed:.3f}s)"
        # The summarizer eventually runs -- but on the background thread.
        assert summarized.wait(timeout=3.0), "summarizer was never scheduled"
        assert ran_on, "summarizer never ran"
        assert all(tid != caller_thread for tid in ran_on), (
            "summarizer ran on the caller (bus) thread -- R2 violated"
        )
    finally:
        mgr.close()


def test_queue_user_utterance_does_not_summarize_inline():
    """Same R2 guard via the other ingest entrypoint (``queue_user_utterance``
    -> ``add_message``): no inline summarizer call on the caller's thread."""
    caller_thread = threading.get_ident()
    ran_on: list[int] = []
    done = threading.Event()

    def summarizer(text: str) -> str:
        ran_on.append(threading.get_ident())
        done.set()
        return "rolled"

    mgr = _make_manager(summarizer=summarizer)
    try:
        for i in range(14):
            mgr.queue_user_utterance(f"please remember fact {i} about my schedule today")
        assert done.wait(timeout=3.0)
        assert ran_on and all(tid != caller_thread for tid in ran_on)
    finally:
        mgr.close()


def test_rolling_summary_folds_prior_head_in():
    """R2: the prior summary head is fed back into the next summarize call so the
    layer-2 record accumulates (rolling, not fragmented).

    Driven deterministically: a pre-seeded head must show up as ``Summary so
    far:`` in the next scheduled summarize input (no dependence on summary
    re-trigger timing)."""
    seen_inputs: list[str] = []
    called = threading.Event()

    def summarizer(text: str) -> str:
        seen_inputs.append(text)
        called.set()
        return "ROLLED HEAD"

    mgr = _make_manager(summarizer=summarizer)
    try:
        # Pre-seed a prior head as if an earlier segment was already summarized.
        mgr._summary_head = "Earlier: the user planned a hiking trip."
        _fill_until_summary(mgr, sentence="now talk about packing the gear list")
        assert called.wait(timeout=3.0), "summarizer was never scheduled"
        # The summarize input folds the prior head in ("Summary so far:") --
        # proving accumulation rather than fragmentation.
        assert seen_inputs
        assert "Summary so far:" in seen_inputs[0]
        assert "hiking trip" in seen_inputs[0]
        # And the head advances to the new rolled value.
        deadline = time.monotonic() + 3.0
        while mgr._summary_head != "ROLLED HEAD" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert mgr._summary_head == "ROLLED HEAD"
    finally:
        mgr.close()


# --- keyword fallback when no summarizer is injected ------------------------


def test_keyword_fallback_when_summarizer_is_none():
    """With ``summarizer=None`` the rolling summary still produces a head via the
    legacy topic body -- no LLM, no crash."""
    mgr = _make_manager(summarizer=None)
    try:
        assert mgr._summary_head == ""
        _fill_until_summary(mgr, sentence="planning a trip to the mountains")
        deadline = time.monotonic() + 3.0
        while mgr._summary_head == "" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert mgr._summary_head, "keyword fallback produced no summary head"
        assert "Conversation about:" in mgr._summary_head
    finally:
        mgr.close()


def test_summarize_text_keyword_fallback_unit():
    """Direct unit on ``_summarize_text``: with no summarizer it folds the prior
    head onto the keyword body."""
    mgr = _make_manager(summarizer=None)
    try:
        first = mgr._summarize_text("", "user: I love hiking trails and camping gear")
        assert first.startswith("Conversation about:")
        rolled = mgr._summarize_text(first, "user: also kayaking on alpine lakes")
        assert rolled.startswith(first)
        assert "Conversation about:" in rolled[len(first):]
    finally:
        mgr.close()


def test_summarizer_exception_falls_back_to_keywords():
    """A summarizer that raises must not crash the background thread -- it falls
    back to the keyword body."""
    def boom(text: str) -> str:
        raise RuntimeError("model offline")

    mgr = _make_manager(summarizer=boom)
    try:
        out = mgr._summarize_text("", "user: tell me about the weather forecast today")
        assert out.startswith("Conversation about:")
    finally:
        mgr.close()


# --- R8: profile gate on get_context_for_llm --------------------------------


def test_profile_block_absent_when_profile_disabled():
    """R8 gate: the ``=== User Profile ===`` block is NOT injected when
    ``profile_enabled`` is False (the default), even if profile rows exist."""
    mgr = _make_manager(profile_enabled=False)
    try:
        # Force a populated profile so a leak would be visible.
        mgr.get_user_profile = lambda: {"name": "Ada", "location": "London"}
        ctx = mgr.get_context_for_llm("what is my name")
        assert "=== User Profile ===" not in ctx
        assert "Ada" not in ctx
    finally:
        mgr.close()


def test_profile_block_present_when_profile_enabled():
    """When ``profile_enabled`` is True the block IS emitted (the gate is the
    only thing that differs from the disabled case)."""
    mgr = _make_manager(profile_enabled=True)
    try:
        mgr.get_user_profile = lambda: {"name": "Ada"}
        ctx = mgr.get_context_for_llm("what is my name")
        assert "=== User Profile ===" in ctx
        assert "name: Ada" in ctx
    finally:
        mgr.close()


def test_profile_extractor_noop_when_disabled():
    """The ingest-time extractor is a no-op when ``profile_enabled`` is False --
    no ``update_user_profile`` write happens."""
    mgr = _make_manager(profile_enabled=False)
    try:
        writes: list[tuple] = []
        mgr.update_user_profile = lambda key, value, confidence=1.0: writes.append(
            (key, value, confidence)
        )
        mgr._extract_profile("my name is Grace and I live in Oxford")
        assert writes == []
    finally:
        mgr.close()


def test_profile_regex_extracts_high_signal_facts():
    """The deterministic regex pass maps high-signal phrases to profile writes at
    a confidence floor >= 0.9 (R8). Runs only when enabled + DB-available."""
    mgr = _make_manager(profile_enabled=True)
    try:
        writes: list[tuple] = []
        mgr.update_user_profile = lambda key, value, confidence=1.0: writes.append(
            (key, value, confidence)
        )
        mgr._extract_profile("my name is Grace")
        mgr._extract_profile("I live in Oxford.")
        mgr._extract_profile("call me Gracie")
        mgr._extract_profile("I prefer dark roast coffee, please")
        keyed = {k: (v, c) for k, v, c in writes}
        assert keyed["name"][0] in ("Grace", "Gracie")
        assert keyed["location"][0] == "Oxford"
        assert keyed["preference"][0] == "dark roast coffee"
        # Confidence floor: every deterministic write is >= 0.9.
        assert all(c >= 0.9 for _, _, c in writes)
    finally:
        mgr.close()


def test_profile_extractor_ignores_non_facts():
    """No high-signal phrase -> no profile write."""
    mgr = _make_manager(profile_enabled=True)
    try:
        writes: list[tuple] = []
        mgr.update_user_profile = lambda key, value, confidence=1.0: writes.append(
            (key, value, confidence)
        )
        mgr._extract_profile("what time is the meeting tomorrow")
        assert writes == []
    finally:
        mgr.close()


# --- retention: no-DB no-op -------------------------------------------------


def test_apply_retention_is_noop_without_db():
    """``apply_retention`` returns 0 and does nothing when there is no live DB
    (guarded by ``_db_available``)."""
    mgr = _make_manager()
    mgr._db_available = False
    try:
        assert mgr.apply_retention() == 0
    finally:
        mgr.close()


def test_init_wires_p2b_params():
    """The PINNED CONTRACT __init__ params are stored verbatim."""
    def s(text: str) -> str:
        return text

    mgr = _make_manager(
        summarizer=s,
        profile_enabled=True,
        episodic_ttl_days=30,
        summary_ttl_days=180,
    )
    try:
        assert mgr._summarizer is s
        assert mgr.profile_enabled is True
        assert mgr.episodic_ttl_days == 30
        assert mgr.summary_ttl_days == 180
    finally:
        mgr.close()


def test_create_memory_manager_forwards_p2b_params():
    """``create_memory_manager`` forwards the new keyword params through to the
    manager (the seam both groups implement)."""
    from utils.memory import create_memory_manager

    def s(text: str) -> str:
        return text

    mgr = create_memory_manager(
        db_url="postgresql://fake",
        enable_embeddings=False,
        smart_save=False,
        pool_factory=_null_pool_factory,
        summarizer=s,
        profile_enabled=True,
        episodic_ttl_days=7,
        summary_ttl_days=42,
    )
    try:
        assert mgr._summarizer is s
        assert mgr.profile_enabled is True
        assert mgr.episodic_ttl_days == 7
        assert mgr.summary_ttl_days == 42
    finally:
        mgr.close()


# --- live-Postgres producer/retention assertions (marked, excluded by logic) -
#
# Self-contained pytest-postgresql plumbing so a single-file ``--postgres`` run
# works without depending on fixtures defined in the integration module.


try:
    from pytest_postgresql import factories as _pg_factories  # type: ignore
    _PGSQL_FIXTURE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PGSQL_FIXTURE_AVAILABLE = False


if _PGSQL_FIXTURE_AVAILABLE:
    postgresql_proc = _pg_factories.postgresql_proc(port=None, unixsocketdir="/tmp")
    postgresql = _pg_factories.postgresql("postgresql_proc")


@pytest.fixture
def db_url(postgresql):  # pragma: no cover - only runs under --postgres
    with postgresql.cursor() as cur:
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except Exception as exc:
            pytest.skip(f"pgvector not installed on this PG: {exc}")
    postgresql.commit()
    info = postgresql.info
    return (
        f"postgresql://{info.user}:@{info.host}:{info.port}/{info.dbname}"
        f"?sslmode=disable"
    )


@pytest.mark.postgres
class TestProducersWithRealDB:
    """Summary persistence, profile recall, and retention against a real
    ephemeral PostgreSQL. Excluded from the logic suite (needs ``--postgres``)."""

    @staticmethod
    def _pg_manager(db_url, **overrides):
        kwargs = dict(
            db_url=db_url,
            enable_embeddings=False,
            smart_save=False,
        )
        kwargs.update(overrides)
        return MemoryManager(**kwargs)

    def test_profile_recall_only_when_enabled(self, db_url):  # pragma: no cover
        mgr = self._pg_manager(db_url, profile_enabled=True)
        try:
            mgr._extract_profile("my name is Grace")
            assert mgr.get_user_profile().get("name") == "Grace"
            assert "=== User Profile ===" in mgr.get_context_for_llm("who am I")
        finally:
            mgr.close()

    def test_rolling_summary_persisted(self, db_url):  # pragma: no cover
        mgr = self._pg_manager(
            db_url,
            summarizer=lambda text: "ROLLED: " + text[:40],
            max_context_tokens=1,
        )
        try:
            _fill_until_summary(mgr, sentence="persist this summary to postgres")
            deadline = time.monotonic() + 3.0
            while mgr._summary_head == "" and time.monotonic() < deadline:
                time.sleep(0.01)
            with mgr._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM summaries")
                    assert cur.fetchone()[0] >= 1
        finally:
            mgr.close()

    def test_retention_evicts_old_episodic_and_summaries(self, db_url):  # pragma: no cover
        from datetime import datetime, timedelta

        mgr = self._pg_manager(
            db_url,
            summarizer=lambda text: "ROLLED",
            episodic_ttl_days=90,
            summary_ttl_days=365,
        )
        try:
            old_ts = datetime.now() - timedelta(days=400)
            with mgr._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO messages (session_id, role, content, timestamp, saved_at) "
                        "VALUES (%s, 'user', 'ancient', %s, %s)",
                        (mgr.session_id, old_ts, old_ts),
                    )
                    cur.execute(
                        "INSERT INTO summaries (session_id, summary, created_at) "
                        "VALUES (%s, 'stale', %s)",
                        (mgr.session_id, old_ts),
                    )
                    cur.execute(
                        "INSERT INTO user_profile (key, value) VALUES ('name', 'Grace')"
                    )
            removed = mgr.apply_retention()
            assert removed >= 1
            with mgr._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM messages WHERE content = 'ancient'")
                    assert cur.fetchone()[0] == 0
                    cur.execute("SELECT COUNT(*) FROM summaries WHERE summary = 'stale'")
                    assert cur.fetchone()[0] == 0
                    # user_profile is NEVER TTL'd.
                    cur.execute("SELECT COUNT(*) FROM user_profile WHERE key = 'name'")
                    assert cur.fetchone()[0] == 1
        finally:
            mgr.close()
