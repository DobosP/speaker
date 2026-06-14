"""Tests for the persistent SQLite memory backend (Decision D6).

Stdlib-only (sqlite3 + the pure-Python recall layer) -- no numpy, psycopg, or
models -- so this runs in the thin CI env. Covers: Memory-protocol conformance,
tag fidelity, cross-session PERSISTENCE (the headline value), byte-identical
recall PARITY with SessionMemory, the optional pure-Python cosine path with a
fake embedder, and TTL pruning.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from always_on_agent.memory import Memory, MemoryItem, SessionMemory
from always_on_agent.sqlite_memory import SqliteVecMemory


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "mem.db")


# --- protocol conformance ---------------------------------------------------


def test_isinstance_memory_and_all_verbs():
    mem = SqliteVecMemory(":memory:")
    try:
        assert isinstance(mem, Memory)
        mem.add("hello world this is a test", tags=("user",))
        assert isinstance(mem.search("hello"), list)
        assert isinstance(mem.all(), list)
        assert isinstance(mem.context_for_llm("hello"), str)
        assert mem.prune() == 0  # no TTL configured -> nothing pruned
    finally:
        mem.close()


def test_all_preserves_tags():
    """R3: all() returns the raw (text, tags) handed to add()."""
    mem = SqliteVecMemory(":memory:")
    try:
        mem.add("HE MURMURED HIS MURDERING", tags=("ingested",))
        items = mem.all()
        assert any(
            isinstance(i, MemoryItem)
            and i.text == "HE MURMURED HIS MURDERING"
            and "ingested" in i.tags
            for i in items
        )
    finally:
        mem.close()


def test_empty_and_blank_adds_are_ignored():
    mem = SqliteVecMemory(":memory:")
    try:
        mem.add("   ", tags=("user",))
        mem.add("", tags=("user",))
        assert mem.all() == []
    finally:
        mem.close()


# --- persistence (the headline) ---------------------------------------------


def test_rows_persist_across_reopen(tmp_db):
    a = SqliteVecMemory(tmp_db)
    a.add("my favorite color is teal", tags=("user",))
    a.add("we planned the trip to japan", tags=("user",))
    a.close()

    b = SqliteVecMemory(tmp_db)
    try:
        texts = [i.text for i in b.all()]
        assert "my favorite color is teal" in texts
        assert "we planned the trip to japan" in texts
        # And recall works against the persisted rows on the fresh process.
        block = b.context_for_llm("what is my favorite color")
        assert "favorite color is teal" in block
    finally:
        b.close()


# --- byte-identical recall parity with SessionMemory ------------------------


def test_recall_block_byte_identical_to_session_memory():
    """The keyword path must emit a recall block byte-identical to SessionMemory
    for the same stored data (both go through the shared candidate_for_item ->
    build_block)."""
    facts = [
        "my favorite color is teal",
        "we planned the trip to japan in spring",
        "you wanted a dark roast coffee",
        "the meeting is scheduled for next tuesday",
    ]
    q = "what is my favorite color and the japan trip"

    ram = SessionMemory()
    sql = SqliteVecMemory(":memory:")
    try:
        for f in facts:
            ram.add(f, tags=("user",))
            sql.add(f, tags=("user",))
        assert sql.context_for_llm(q) == ram.context_for_llm(q)
        assert sql.context_for_llm(q).startswith("=== Past Conversations ===")
    finally:
        sql.close()


def test_current_utterance_is_not_recalled_back():
    mem = SqliteVecMemory(":memory:")
    try:
        mem.add("what is my favorite color", tags=("user",))
        # The only stored item IS the query -> nothing to recall (no echo).
        assert mem.context_for_llm("what is my favorite color") == ""
    finally:
        mem.close()


# --- optional pure-Python cosine path (with a fake embedder) ----------------


# A SYNONYM-axis embedder: distinct vocabularies map to the same axis, so a query
# can match a row it shares NO literal word with -- isolating cosine from keyword
# overlap. No model.
_AXES = {
    "cat": 0, "dog": 0, "pet": 0,           # axis 0: animals
    "car": 1, "truck": 1, "vehicle": 1,      # axis 1: vehicles
    "pizza": 2, "sushi": 2, "meal": 2,       # axis 2: food
}


def _axis_embedder(text: str):
    v = [0.0, 0.0, 0.0]
    for w in text.lower().split():
        if w in _AXES:
            v[_AXES[w]] += 1.0
    return v


def test_cosine_path_ranks_by_embedding_not_keyword_overlap():
    """The query shares NO literal token with the winning row, so only cosine
    (not keyword overlap) can pick it -- isolating the embedding path."""
    mem = SqliteVecMemory(":memory:", embedder=_axis_embedder)
    try:
        mem.add("my dog is friendly", tags=("user",))   # axis 0
        mem.add("the truck is fast", tags=("user",))    # axis 1
        mem.add("i ate sushi today", tags=("user",))    # axis 2
        # 'vehicle' overlaps no row's words but maps to the vehicle axis.
        results = mem.search("vehicle", limit=3)
        assert results, "cosine path returned nothing"
        assert "truck" in results[0].text
    finally:
        mem.close()
    # Control: the keyword path (no embedder) finds NOTHING for 'vehicle' (zero
    # overlap), proving the win above came from cosine, not overlap.
    kw = SqliteVecMemory(":memory:")
    try:
        kw.add("the truck is fast", tags=("user",))
        assert kw.search("vehicle") == []
    finally:
        kw.close()


def test_add_time_embedder_failure_is_survived():
    """A throwing embedder at add() must not abort the ingest: the row is still
    stored (with no embedding) and recall over the other rows still works."""
    def flaky(text: str):
        if "boom" in text:
            raise RuntimeError("embed failed")
        return _axis_embedder(text)

    mem = SqliteVecMemory(":memory:", embedder=flaky)
    try:
        mem.add("a boom dog row", tags=("user",))   # embed raises -> stored, emb=None
        mem.add("the truck is fast", tags=("user",))
        assert any("boom" in i.text for i in mem.all())   # stored despite embed failure
        results = mem.search("vehicle", limit=3)
        assert results and "truck" in results[0].text     # other rows still recallable
    finally:
        mem.close()


def test_query_embedder_failure_degrades_to_no_recall():
    """A throwing embedder on the QUERY degrades to empty recall, never raises."""
    def flaky(text: str):
        if text == "explode":
            raise RuntimeError("query embed failed")
        return _axis_embedder(text)

    mem = SqliteVecMemory(":memory:", embedder=flaky)
    try:
        mem.add("the truck is fast", tags=("user",))
        assert mem.context_for_llm("explode") == ""
    finally:
        mem.close()


def test_double_close_is_idempotent():
    """close() must be idempotent (matches SessionMemory; the runtime may
    double-stop)."""
    mem = SqliteVecMemory(":memory:")
    mem.add("a fact", tags=("user",))
    mem.close()
    mem.close()  # must not raise


def test_recall_parity_holds_beyond_the_working_window():
    """Parity holds at ANY size: with a small working window, both backends
    consider only the last `max_items` (SessionMemory evicts; SQLite scans the
    same window), so the recall block stays byte-identical even after many adds."""
    ram = SessionMemory(max_items=10)
    sql = SqliteVecMemory(":memory:", max_items=10)
    try:
        for i in range(40):  # 4x the window
            fact = f"note {i} about apple banana cherry topic {i % 5}"
            ram.add(fact, tags=("user",))
            sql.add(fact, tags=("user",))
        q = "apple banana cherry topic 3"
        assert sql.context_for_llm(q) == ram.context_for_llm(q)
        assert sql.context_for_llm(q)  # non-empty (something recalled)
    finally:
        sql.close()


def test_concurrent_add_and_recall_is_thread_safe(tmp_db):
    """The shared connection + lock must tolerate concurrent writers/readers."""
    import threading

    mem = SqliteVecMemory(tmp_db)
    errors: list = []

    def writer(n):
        try:
            for i in range(100):
                mem.add(f"row {n}-{i} apple", tags=("user",))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def reader():
        try:
            for _ in range(100):
                mem.context_for_llm("apple")
                mem.all()
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(0,)),
               threading.Thread(target=writer, args=(1,)),
               threading.Thread(target=reader),
               threading.Thread(target=reader)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    try:
        assert errors == [], f"concurrent access raised: {errors[:3]}"
    finally:
        mem.close()


def test_cosine_path_used_only_when_embedder_present(tmp_db):
    # Without an embedder, recall is keyword overlap (no embeddings consulted).
    mem = SqliteVecMemory(tmp_db)
    try:
        mem.add("alpha beta gamma delta", tags=("user",))
        # 'zzz' shares no word -> keyword path finds nothing.
        assert mem.context_for_llm("zzz") == ""
    finally:
        mem.close()


# --- TTL prune --------------------------------------------------------------


def test_prune_drops_rows_older_than_ttl(tmp_db):
    mem = SqliteVecMemory(tmp_db, ttl_days=1)
    try:
        mem.add("recent fact", tags=("user",))
        # Backdate one row well past the TTL by editing ts directly.
        import time as _t
        mem._conn.execute(
            "INSERT INTO items (text, tags, ts, embedding) VALUES (?, ?, ?, NULL)",
            ("ancient fact", "[\"user\"]", _t.time() - 5 * 86400),
        )
        mem._conn.commit()
        removed = mem.prune()
        assert removed == 1
        texts = [i.text for i in mem.all()]
        assert "recent fact" in texts and "ancient fact" not in texts
    finally:
        mem.close()


def test_prune_noop_without_ttl():
    mem = SqliteVecMemory(":memory:")  # ttl_days=None
    try:
        mem.add("a fact", tags=("user",))
        assert mem.prune() == 0
    finally:
        mem.close()
