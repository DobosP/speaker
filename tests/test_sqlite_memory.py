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


def _fake_embedder(text: str):
    """Deterministic 3-dim 'embedding': counts of three marker words. No model."""
    t = text.lower()
    return [float(t.count("alpha")), float(t.count("beta")), float(t.count("gamma"))]


def test_cosine_path_ranks_by_embedding_similarity():
    mem = SqliteVecMemory(":memory:", embedder=_fake_embedder)
    try:
        mem.add("the alpha alpha report", tags=("user",))   # vec ~ (2,0,0)
        mem.add("a beta beta summary", tags=("user",))      # vec ~ (0,2,0)
        mem.add("gamma notes here", tags=("user",))         # vec ~ (0,0,1)
        # Query aligned with the beta axis -> the beta row ranks first, even
        # though it shares NO keyword with the query (pure cosine, not overlap).
        results = mem.search("beta", limit=3)
        assert results, "cosine path returned nothing"
        assert "beta" in results[0].text
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
