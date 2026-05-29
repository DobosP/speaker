from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable, Optional, Protocol, Sequence, runtime_checkable

from .text import keywords, normalize_text


@dataclass(frozen=True)
class MemoryItem:
    text: str
    tags: tuple[str, ...] = ()
    timestamp: float = field(default_factory=time.time)


@runtime_checkable
class Memory(Protocol):
    """The one backend-neutral memory seam.

    Verbs only -- no ``db_url``, embeddings, ``session_id``, pool,
    ``role``/``content`` dicts, pgvector, or SQL. Both the in-RAM
    :class:`SessionMemory` and the Postgres :class:`MemoryManagerAdapter`
    conform so the runtime/supervisor/capabilities can type against this and
    stay Postgres-free."""

    def add(self, text: str, tags: tuple[str, ...] = ()) -> None: ...        # ingest (tag = neutral channel)

    def search(self, query: str, limit: int = 5) -> Sequence[MemoryItem]: ... # recall -> neutral items

    def all(self) -> Sequence[MemoryItem]: ...                               # recent window (tag-faithful)

    def context_for_llm(self, query: str) -> str: ...                        # ready-to-prepend block or ''

    def prune(self) -> int: ...                                              # retention/eviction

    def close(self) -> None: ...                                             # flush + release


# Default working-window cap (Layer-1 RAM). Keeps the in-RAM store from growing
# unbounded over a long session; the oldest items fall off the front.
DEFAULT_MAX_ITEMS = 200

# Relevance gate for the keyword recall: a candidate must share at least this
# many normalized words with the query before it is injected. Mirrors the
# Postgres ``similarity > 0.6`` self-gate (R11) so the in-RAM injection volume
# stays comparable to the Postgres top-3.
_RECALL_MIN_OVERLAP = 2
_RECALL_MAX_ITEMS = 3


class SessionMemory:
    """In-memory session store used by tests and the local prototype.

    Conforms to :class:`Memory`. The desktop default when ``DATABASE_URL`` is
    unset."""

    def __init__(self, max_items: int = DEFAULT_MAX_ITEMS):
        self._items: list[MemoryItem] = []
        self._max_items = max(1, int(max_items))

    def add(self, text: str, tags: tuple[str, ...] = ()) -> None:
        cleaned = text.strip()
        if cleaned:
            self._items.append(MemoryItem(cleaned, tags or keywords(cleaned)))
            # Working-window cap: drop the oldest once over the limit so a long
            # session can't grow RAM without bound.
            if len(self._items) > self._max_items:
                self._items = self._items[-self._max_items:]

    def search(self, query: str, limit: int = 5) -> list[MemoryItem]:
        q_words = set(normalize_text(query).split())
        scored: list[tuple[int, MemoryItem]] = []
        for item in self._items:
            haystack = set(normalize_text(item.text).split()) | set(item.tags)
            score = len(q_words & haystack)
            if score:
                scored.append((score, item))
        scored.sort(key=lambda pair: (pair[0], pair[1].timestamp), reverse=True)
        return [item for _, item in scored[:limit]]

    def all(self) -> list[MemoryItem]:
        return list(self._items)

    def context_for_llm(self, query: str) -> str:
        """Keyword recall as a ready-to-prepend block, or ``''`` on no hit.

        A relevance gate (min keyword overlap on the message *text* + an item
        cap, R11) keeps the injection volume close to the Postgres top-3 and
        means an irrelevant query costs nothing -- the empty string leaves the
        prompt unchanged."""
        q_words = set(normalize_text(query).split())
        if not q_words:
            return ""
        scored: list[tuple[int, MemoryItem]] = []
        for item in self._items:
            overlap = len(q_words & set(normalize_text(item.text).split()))
            if overlap >= _RECALL_MIN_OVERLAP:
                scored.append((overlap, item))
        if not scored:
            return ""
        scored.sort(key=lambda pair: (pair[0], pair[1].timestamp), reverse=True)
        lines = ["=== Past Conversations ==="]
        for _, item in scored[:_RECALL_MAX_ITEMS]:
            lines.append(item.text[:150])
        return "\n".join(lines)

    def prune(self) -> int:
        # Age-TTL retention is P2b; the working-window cap in add() is the only
        # eviction this cycle. Nothing to do at close-time.
        return 0

    def close(self) -> None:
        return None


class MemoryManagerAdapter:
    """Thin :class:`Memory` over the existing Postgres ``MemoryManager``.

    Lazily imports :mod:`utils.memory` so ``always_on_agent`` stays
    Postgres-free for shells that never touch a DB. All Postgres-isms
    (roles, embeddings, dicts, SQL) stay inside this class.

    Relies on ``MemoryManager``'s graceful no-DB degradation -- it is safe to
    construct without a live database (the pool just never opens and every DB
    call no-ops)."""

    def __init__(
        self,
        *,
        summarizer: Optional[Callable[[str], str]] = None,
        profile_enabled: bool = False,
        episodic_ttl_days: int = 90,
        summary_ttl_days: int = 365,
        **manager_kwargs,
    ):
        from utils.memory import create_memory_manager  # lazy: keep the brain DB-free

        # P2b knobs forwarded to the engine: the rolling-summary LLM callable
        # (keyword fallback inside the manager when None), the default-off
        # Postgres-only profile producer, and the episodic/summary age-TTLs that
        # prune() -> apply_retention() enforces. user_profile is never TTL'd.
        self._manager = create_memory_manager(
            summarizer=summarizer,
            profile_enabled=profile_enabled,
            episodic_ttl_days=episodic_ttl_days,
            summary_ttl_days=summary_ttl_days,
            **manager_kwargs,
        )
        # Our own small in-RAM ring buffer of the raw (text, tags) handed to
        # add() (R3). MemoryManager.recent_messages keeps only role + junk-
        # filters, so all() reads back from here to preserve tag fidelity --
        # both backends then behave identically (test_addressing relies on the
        # 'ingested' tag surviving).
        self._ring: list[MemoryItem] = []
        self._max_items = int(getattr(self._manager, "max_recent_messages", 0)) or DEFAULT_MAX_ITEMS

    def add(self, text: str, tags: tuple[str, ...] = ()) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        # Keep the raw (text, tags) regardless of how it is routed/persisted so
        # all() stays tag-faithful.
        self._ring.append(MemoryItem(cleaned, tags or keywords(cleaned)))
        if len(self._ring) > self._max_items:
            self._ring = self._ring[-self._max_items:]
        # Meeting notes are RAM-only by default (R7, §9.7 privacy): never hand
        # them to the persisting manager unless meeting_persist is enabled.
        if "meeting" in tags and not getattr(self._manager, "meeting_persist", False):
            return
        # Tag-routed: assistant output is RAM-only context; user/ingested speech
        # is queued for debounced persistence.
        if "assistant_output" in tags:
            self._manager.add_message("assistant", cleaned)
        else:
            self._manager.queue_user_utterance(cleaned)

    def search(self, query: str, limit: int = 5) -> list[MemoryItem]:
        results = self._manager.search_memory(query, limit=limit)
        items: list[MemoryItem] = []
        for d in results:
            # Defensive tags (R9): summary rows have no 'role'; build the tag
            # tuple from whatever identity fields are present.
            tags = tuple(t for t in (d.get("type"), d.get("role")) if t)
            items.append(MemoryItem(str(d.get("content", "")), tags))
        return items

    def all(self) -> list[MemoryItem]:
        # Return the adapter's own ring buffer (R3) -- NOT recent_messages,
        # which drops tags.
        return list(self._ring)

    def context_for_llm(self, query: str) -> str:
        return self._manager.get_context_for_llm(query)

    def prune(self) -> int:
        # Age-TTL retention (P2b): episodic messages older than
        # episodic_ttl_days are summarized-then-evicted, summaries older than
        # summary_ttl_days are dropped, user_profile is never TTL'd. The manager
        # no-ops + returns 0 without a live DB. Invoked from VoiceRuntime.stop()
        # at close-time (R6). Returns rows removed.
        return self._manager.apply_retention()

    def close(self) -> None:
        self._manager.close()
