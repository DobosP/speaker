"""Persistent on-device memory backend on stdlib ``sqlite3`` (Decision D6).

The third :class:`always_on_agent.memory.Memory` backend, alongside the in-RAM
:class:`SessionMemory` and the Postgres :class:`MemoryManagerAdapter`. It is
"``SessionMemory`` but persistent": the same six-verb protocol and the SAME
shared recall selection (:func:`always_on_agent.recall.build_block` via the
shared :func:`always_on_agent.memory.candidate_for_item`), so it emits a
byte-identical recall block for identical stored data -- but rows survive a
process restart, giving cross-session continuity on a desktop WITHOUT Postgres,
and serving as the reference for the Dart/mobile SQLite tier.

Dependency-light on purpose: stdlib ``sqlite3`` only (no ``sqlite-vec`` native
extension, no numpy). Recall defaults to keyword overlap -- exactly the path
desktop-without-Postgres uses today. When an ``embedder`` callable is injected,
embeddings are stored as float BLOBs and recall ranks by a pure-Python cosine
over the recent pool (the pool is small, so brute force is trivially cheap and
ports cleanly to Dart). The native ``sqlite-vec`` ANN path is a future accel.
"""
from __future__ import annotations

import array
import json
import math
import os
import sqlite3
import threading
import time
from typing import Callable, Optional, Sequence

from .memory import DEFAULT_MAX_ITEMS, MemoryItem, candidate_for_item
from .recall import Candidate, RecallBudget, build_block
from .text import keywords, normalize_text

Embedder = Callable[[str], Sequence[float]]


def _pack(vec: Sequence[float]) -> bytes:
    return array.array("f", [float(x) for x in vec]).tobytes()


def _unpack(blob: bytes) -> list[float]:
    a = array.array("f")
    a.frombytes(blob)
    return list(a)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class SqliteVecMemory:
    """Persistent SQLite-backed :class:`always_on_agent.memory.Memory`.

    Conforms to the six-verb protocol (add/search/all/context_for_llm/prune/
    close). Thread-safe: one connection guarded by a lock (the runtime touches
    memory from the bus thread on ``add`` and the worker thread on
    ``context_for_llm``).
    """

    def __init__(
        self,
        path: str = ":memory:",
        *,
        max_items: int = DEFAULT_MAX_ITEMS,
        budget: Optional[RecallBudget] = None,
        embedder: Optional[Embedder] = None,
        ttl_days: Optional[int] = None,
    ) -> None:
        self._path = path
        self._max_items = max(1, int(max_items))
        self._budget = budget or RecallBudget()
        self._embedder = embedder
        self._ttl_days = ttl_days if (ttl_days and ttl_days > 0) else None
        self._lock = threading.Lock()
        self._closed = False
        # check_same_thread=False + our own lock: the connection is shared across
        # the bus/worker threads, serialized by self._lock.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        # Persisted post-ASR text (and, with visual memory on, screen OCR/caption
        # rows) is PRIVATE (§9.7). The default umask leaves a new sqlite file
        # group/other-readable; lock the file + its dir to owner-only so a
        # co-tenant on a multi-user host can't read it off disk. Best-effort.
        if path != ":memory:":
            try:
                os.chmod(path, 0o600)
                parent = os.path.dirname(os.path.abspath(path))
                if parent:
                    os.chmod(parent, 0o700)
            except OSError:
                pass
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS items (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                text      TEXT NOT NULL,
                tags      TEXT NOT NULL,
                ts        REAL NOT NULL,
                embedding BLOB
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_items_ts ON items(ts)")
        self._conn.commit()

    # --- Memory protocol ----------------------------------------------------

    def add(self, text: str, tags: tuple[str, ...] = ()) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        # Mirror SessionMemory.add: default tags to the derived keywords so the
        # two backends store -- and later recall -- identical (text, tags).
        stored_tags = tuple(tags) if tags else tuple(keywords(cleaned))
        emb: Optional[bytes] = None
        if self._embedder is not None:
            try:
                emb = _pack(self._embedder(cleaned))
            except Exception:  # noqa: BLE001 - embedding is best-effort, never fatal
                emb = None
        with self._lock:
            self._conn.execute(
                "INSERT INTO items (text, tags, ts, embedding) VALUES (?, ?, ?, ?)",
                (cleaned, json.dumps(list(stored_tags)), time.time(), emb),
            )
            self._conn.commit()

    def _recent_rows(self) -> list[tuple[str, tuple[str, ...], float, Optional[bytes]]]:
        # The recall candidate pool is the SAME window all() exposes and that the
        # in-RAM SessionMemory effectively scans (it evicts beyond max_items, so
        # its candidate set IS the last max_items). Tracking max_items here -- not
        # a separate constant -- keeps recall byte-identical with SessionMemory at
        # any DB size and consistent with all().
        with self._lock:
            cur = self._conn.execute(
                "SELECT text, tags, ts, embedding FROM items ORDER BY id DESC LIMIT ?",
                (self._max_items,),
            )
            rows = cur.fetchall()
        out: list[tuple[str, tuple[str, ...], float, Optional[bytes]]] = []
        for text, tags_json, ts, emb in rows:
            try:
                tags = tuple(json.loads(tags_json))
            except Exception:  # noqa: BLE001 - tolerate a malformed tag blob
                tags = ()
            out.append((text, tags, float(ts), emb))
        out.reverse()  # oldest -> newest, matching SessionMemory list order
        return out

    def _candidates(self, query: str) -> list[Candidate]:
        q_norm = normalize_text(query)
        q_words = set(q_norm.split())
        if not q_words:
            return []
        rows = self._recent_rows()
        if self._embedder is not None:
            return self._cosine_candidates(query, q_norm, rows)
        # Keyword path: SAME shared candidate builder as SessionMemory -> the
        # rendered recall block is byte-identical for identical stored data.
        out: list[Candidate] = []
        for text, tags, ts, _emb in rows:
            cand = candidate_for_item(MemoryItem(text, tags, ts), q_words, q_norm)
            if cand is not None:
                out.append(cand)
        return out

    def _cosine_candidates(self, query, q_norm, rows) -> list[Candidate]:
        try:
            qvec = list(self._embedder(query))  # type: ignore[misc]
        except Exception:  # noqa: BLE001 - fall back to no recall on an embed failure
            return []
        out: list[Candidate] = []
        for text, tags, ts, emb in rows:
            if normalize_text(text) == q_norm:
                continue  # never recall the current utterance back
            if emb is None:
                continue
            score = _cosine(qvec, _unpack(emb))
            if score <= 0.0:
                continue
            role = "user" if "user" in tags else ("assistant" if "assistant_output" in tags else None)
            if "vision" in tags:
                kind = "vision"
            elif "summary" in tags:
                kind = "summary"
            else:
                kind = "message"
            out.append(Candidate(text, float(score), kind=kind, role=role, timestamp=ts, tags=tags))
        return out

    def search(self, query: str, limit: int = 5) -> list[MemoryItem]:
        cands = self._candidates(query)
        cands.sort(key=lambda c: (c.score, c.timestamp), reverse=True)
        return [MemoryItem(c.text, c.tags, c.timestamp) for c in cands[:limit]]

    def all(self) -> list[MemoryItem]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT text, tags, ts FROM items ORDER BY id DESC LIMIT ?",
                (self._max_items,),
            )
            rows = cur.fetchall()
        items: list[MemoryItem] = []
        for text, tags_json, ts in rows:
            try:
                tags = tuple(json.loads(tags_json))
            except Exception:  # noqa: BLE001
                tags = ()
            items.append(MemoryItem(text, tags, float(ts)))
        items.reverse()  # oldest -> newest, matching SessionMemory.all()
        return items

    def context_for_llm(self, query: str) -> str:
        return build_block(self._candidates(query), query, self._budget)

    def prune(self) -> int:
        """Age-TTL retention: drop rows older than ``ttl_days``. Returns rows
        removed (0 when no TTL is configured)."""
        if self._ttl_days is None:
            return 0
        cutoff = time.time() - self._ttl_days * 86400
        with self._lock:
            cur = self._conn.execute("DELETE FROM items WHERE ts < ?", (cutoff,))
            self._conn.commit()
            return cur.rowcount or 0

    def close(self) -> None:
        # Idempotent (matches SessionMemory): a second close() -- or the runtime's
        # double stop() path -- must not raise on a closed connection.
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._conn.commit()
            finally:
                self._conn.close()
