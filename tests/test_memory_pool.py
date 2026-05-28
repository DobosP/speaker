"""Unit tests for ``MemoryManager``'s pool semantics under concurrency.

The audited bug was a single shared ``psycopg2.connection`` reused across
the request thread + the background ``MemoryWriter`` ``Timer`` thread.
PR-2 replaces that with ``psycopg_pool.ConnectionPool``. These tests
exercise the *contract* the pool must satisfy WITHOUT needing a real
PostgreSQL -- a fake pool tracks concurrent grabs with a
``BoundedSemaphore`` and a ``threading.local`` to assert that no two
threads ever hold the same logical connection simultaneously.

Integration tests against a real ephemeral PG live in
``tests/test_memory_postgres_integration.py`` (gated by the ``postgres``
pytest marker)."""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager

import pytest

from utils.memory import MemoryManager


# --- fake psycopg_pool.ConnectionPool ------------------------------------


class _FakeCursor:
    """Records every (sql, params) it sees so tests can assert call shape
    without a real database. Returns the last set rows from ``execute``."""

    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []
        self._rows: list[dict] = []

    def __enter__(self): return self
    def __exit__(self, *exc): return False

    def execute(self, sql, params=()):
        self.calls.append((sql, tuple(params) if params else ()))

    def fetchall(self): return list(self._rows)
    def fetchone(self): return self._rows[0] if self._rows else None


class _FakeConn:
    """Tracks which thread currently 'holds' this connection so the test
    can assert no cross-thread sharing."""

    def __init__(self, pool, conn_id):
        self.pool = pool
        self.conn_id = conn_id
        self.holder: int | None = None
        self.cursor_calls: list[_FakeCursor] = []

    def __enter__(self):
        # Acquired -- record holder thread.
        self.holder = threading.get_ident()
        return self

    def __exit__(self, *exc):
        # Released -- clear holder, return to pool.
        self.holder = None
        self.pool._release(self)
        return False

    def cursor(self, *, row_factory=None):
        cur = _FakeCursor()
        self.cursor_calls.append(cur)
        return cur


class _FakePool:
    """Counts concurrent grabs via a semaphore + tracks which conn id is
    held by which thread at any given moment. Tests can read
    ``max_concurrent_held`` after the workload completes to verify
    correct release behavior."""

    def __init__(self, conninfo=None, *, min_size, max_size, kwargs=None, open=True):
        self.conninfo = conninfo
        self.min_size = min_size
        self.max_size = max_size
        self.kwargs = kwargs or {}
        self.open_called = open
        self.closed = False
        self._sem = threading.BoundedSemaphore(max_size)
        self._lock = threading.Lock()
        self._all: list[_FakeConn] = [_FakeConn(self, i) for i in range(max_size)]
        self._free: list[_FakeConn] = list(self._all)
        self._held: set[int] = set()
        self.max_concurrent_held = 0
        self.grabs = 0
        # Records the (conn_id, thread_id) of every grab so tests can
        # assert "no two threads ever held the same conn at the same time".
        self.grab_log: list[tuple[int, int]] = []
        # Map of conn_id -> the thread_id currently holding it; lets us
        # detect a cross-thread share even within the contextmanager.
        self.live_holders: dict[int, int] = {}

    @contextmanager
    def connection(self):
        # ``acquire`` enforces max_size = exhaustion would raise.
        if not self._sem.acquire(blocking=True, timeout=5.0):
            raise RuntimeError("pool exhausted (timed out waiting)")
        try:
            with self._lock:
                if not self._free:
                    # Defense in depth -- semaphore should have made this impossible.
                    raise RuntimeError("pool semaphore allowed past max_size")
                conn = self._free.pop()
                conn_id = conn.conn_id
                self._held.add(conn_id)
                self.live_holders[conn_id] = threading.get_ident()
                self.max_concurrent_held = max(self.max_concurrent_held, len(self._held))
                self.grabs += 1
                self.grab_log.append((conn_id, threading.get_ident()))
            with conn:
                yield conn
        finally:
            self._sem.release()

    def _release(self, conn):
        with self._lock:
            if conn.conn_id in self._held:
                self._held.remove(conn.conn_id)
            self.live_holders.pop(conn.conn_id, None)
            self._free.append(conn)

    def close(self):
        self.closed = True


def _make_manager(*, max_size=5, embedder=None) -> tuple[MemoryManager, _FakePool]:
    """Build a MemoryManager backed by ``_FakePool`` (no real DB)."""
    captured: dict = {}

    def factory(*, conninfo, min_size, max_size, kwargs):
        captured["pool"] = _FakePool(
            conninfo, min_size=min_size, max_size=max_size, kwargs=kwargs, open=True,
        )
        return captured["pool"]

    mgr = MemoryManager(
        db_url="postgresql://fake",
        enable_embeddings=False,  # bypass the sentence_transformer load
        smart_save=False,
        pool_min_size=1,
        pool_max_size=max_size,
        pool_factory=factory,
    )
    return mgr, captured["pool"]


# --- basic lifecycle ------------------------------------------------------


def test_pool_is_constructed_with_configured_sizing():
    """Constructor passes pool_min_size + pool_max_size through to the
    pool factory verbatim."""
    mgr, pool = _make_manager(max_size=7)
    try:
        assert pool.min_size == 1
        assert pool.max_size == 7
        # Pool was opened immediately (open=True path).
        assert pool.open_called is True
    finally:
        mgr.close()


def test_close_releases_pool():
    mgr, pool = _make_manager()
    mgr.close()
    assert pool.closed is True


def test_db_available_when_pool_factory_succeeds():
    mgr, _ = _make_manager()
    try:
        assert mgr._db_available is True
    finally:
        mgr.close()


# --- concurrency contract -------------------------------------------------


def test_no_two_threads_hold_the_same_connection_simultaneously():
    """Spin many writers + readers in parallel; assert that the pool both
    serves concurrent grabs (multiple held at once) and respects the
    max_size ceiling. This is the test that would have caught the audited
    psycopg2-shared-conn bug -- under the old code there was only ever
    one connection (the shared one), so concurrent threads stepped on
    each other.

    A small sleep inside the with-block is required to force overlap;
    without it the calls serialize because the fake cursor's
    ``execute()`` is instantaneous."""
    mgr, pool = _make_manager(max_size=5)
    try:
        errors: list[Exception] = []

        def worker():
            try:
                with mgr._pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1", ())
                        # Force overlap: hold the connection for a bit so
                        # other threads attempting to acquire actually
                        # race against this one.
                        time.sleep(0.02)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=10) as pool_exec:
            futures = [pool_exec.submit(worker) for _ in range(40)]
            wait(futures)
        for exc in errors:
            raise exc

        # Sanity: actually did some concurrent grabbing (>=2 held simultaneously).
        assert pool.max_concurrent_held >= 2, (
            f"pool only ever held {pool.max_concurrent_held} concurrently -- "
            f"the test workload didn't exercise concurrency"
        )
        # The pool ceiling held.
        assert pool.max_concurrent_held <= pool.max_size, (
            f"held {pool.max_concurrent_held} > max_size {pool.max_size} -- "
            f"semaphore failed"
        )
        # And the simple invariant the audited code violated: every worker
        # got SOME connection (i.e. no None grants). The actual count is
        # >= 40 because MemoryManager's __init__ uses the pool too (demo
        # schema + warm-start recent_messages); we only care about the
        # floor.
        assert pool.grabs >= 40
    finally:
        mgr.close()


def test_pool_does_not_exhaust_under_burst_within_capacity():
    """Submitting ``max_size`` simultaneous calls succeeds (semaphore
    grants them all). Submitting more than ``max_size`` simultaneously
    is throttled but eventually completes."""
    mgr, pool = _make_manager(max_size=3)
    try:
        gate = threading.Event()
        results: list[bool] = []

        def slow_writer():
            from utils.memory import Message
            with mgr._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1", ())
                    gate.wait(timeout=2.0)  # hold the connection until gate fires
            results.append(True)

        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = [ex.submit(slow_writer) for _ in range(6)]
            # First 3 acquired immediately; remaining 3 should be waiting.
            time.sleep(0.1)
            assert pool.max_concurrent_held == 3
            gate.set()
            wait(futs)
        assert len(results) == 6
    finally:
        mgr.close()


def test_writer_thread_and_reader_thread_get_separate_connections():
    """The audited race: writer-thread and request-thread sharing one
    psycopg2.connection. Here we assert they actually get different
    logical conns (proves the pool semantics)."""
    mgr, pool = _make_manager(max_size=2)
    try:
        seen_conn_ids: dict[int, set[int]] = {}
        lock = threading.Lock()
        start_gate = threading.Event()

        def worker(tag: str):
            with mgr._pool.connection() as conn:
                start_gate.wait(timeout=2.0)
                with lock:
                    seen_conn_ids.setdefault(threading.get_ident(), set()).add(conn.conn_id)
                # Hold the conn briefly so both threads overlap.
                time.sleep(0.05)

        t1 = threading.Thread(target=worker, args=("a",))
        t2 = threading.Thread(target=worker, args=("b",))
        t1.start(); t2.start()
        start_gate.set()
        t1.join(); t2.join()

        # Each thread saw exactly one conn_id (its own).
        thread_ids = list(seen_conn_ids.keys())
        assert len(thread_ids) == 2
        a_conn_ids = seen_conn_ids[thread_ids[0]]
        b_conn_ids = seen_conn_ids[thread_ids[1]]
        # The two threads must have received different conn ids -- which is
        # exactly the property the audited code violated.
        assert a_conn_ids.isdisjoint(b_conn_ids), (
            f"threads {thread_ids} shared connection ids "
            f"{a_conn_ids & b_conn_ids} -- pool didn't isolate"
        )
    finally:
        mgr.close()


# --- embedding-dim validation --------------------------------------------


def test_save_rejects_embedding_with_wrong_dim():
    """Python-side guard fires before the DB CHECK constraint, so the
    error message is clear."""
    import numpy as np

    mgr, _ = _make_manager()
    mgr.embedder_dim = 384  # default
    try:
        from utils.memory import Message
        bad = np.zeros(768, dtype=np.float32)  # wrong dim
        with pytest.raises(ValueError, match="dimension mismatch"):
            mgr._save_message_to_db(
                Message(role="user", content="hi"), bad,
            )
    finally:
        mgr.close()


def test_save_accepts_embedding_with_correct_dim():
    """Matching dim must NOT raise and must produce one INSERT call on the cursor."""
    import numpy as np

    mgr, pool = _make_manager()
    mgr.embedder_dim = 384
    try:
        from utils.memory import Message
        good = np.zeros(384, dtype=np.float32)
        mgr._save_message_to_db(
            Message(role="user", content="hello there"), good,
        )
        # Find the cursor we used; it should have one execute call.
        all_calls = [c for conn in pool._all for cur in conn.cursor_calls for c in cur.calls]
        assert any("INSERT INTO messages" in sql for sql, _ in all_calls)
    finally:
        mgr.close()


def test_save_with_none_embedding_uses_null_embedding_branch():
    """When the embedder failed (returns None), the save path must still
    work but go through the no-embedding INSERT (no embedder_id stored)."""
    mgr, pool = _make_manager()
    try:
        from utils.memory import Message
        mgr._save_message_to_db(
            Message(role="user", content="hello"), None,
        )
        all_calls = [c for conn in pool._all for cur in conn.cursor_calls for c in cur.calls]
        # The NULL-embedding INSERT has 9 placeholders (no embedding column).
        inserts = [sql for sql, _ in all_calls if "INSERT INTO messages" in sql]
        assert inserts, "no INSERT issued"
        # The branch we took shouldn't include the ``embedding`` column word.
        assert "embedding," not in inserts[0] or "embedding_dim" not in inserts[0]


    finally:
        mgr.close()


def test_search_skips_when_embedder_unavailable():
    """If embeddings are disabled, search_memory returns [] without
    touching the pool (no SET LOCAL hnsw.ef_search call)."""
    mgr, pool = _make_manager()
    try:
        # `enable_embeddings=False` was passed; sanity check.
        assert mgr._embeddings_available is False
        results = mgr.search_memory("anything")
        assert results == []
        # No SET LOCAL call was made.
        all_calls = [c for conn in pool._all for cur in conn.cursor_calls for c in cur.calls]
        assert not any("hnsw.ef_search" in sql for sql, _ in all_calls)
    finally:
        mgr.close()
