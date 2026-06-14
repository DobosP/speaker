from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable, Optional, Protocol, Sequence, runtime_checkable

from .recall import Candidate, RecallBudget, build_block
from .text import keywords, normalize_text


@dataclass(frozen=True)
class MemoryItem:
    text: str
    tags: tuple[str, ...] = ()
    timestamp: float = field(default_factory=time.time)


def candidate_for_item(item: "MemoryItem", q_words: set[str], q_norm: str) -> "Candidate | None":
    """Build a recall :class:`Candidate` from a stored item under keyword scoring.

    The SINGLE source of truth shared by every keyword-overlap backend
    (:class:`SessionMemory` and the persistent :class:`SqliteVecMemory`) so they
    produce byte-identical candidates -- and therefore byte-identical recall
    blocks via :func:`build_block` -- for the same stored data. Score is the
    integer overlap between the query words and the item's words *plus its tags*.
    The current utterance is EXCLUDED (it was just ingested by the answer path;
    echoing the live query back is noise that also distorts the adaptive cutoff).
    Returns ``None`` for the current utterance or a zero-overlap item.
    """
    if normalize_text(item.text) == q_norm:
        return None  # the current utterance itself -- never recall it back
    words = set(normalize_text(item.text).split())
    score = len(q_words & (words | set(item.tags)))
    if not score:
        return None
    role = "user" if "user" in item.tags else ("assistant" if "assistant_output" in item.tags else None)
    if "vision" in item.tags:
        kind = "vision"
    elif "summary" in item.tags:
        kind = "summary"
    else:
        kind = "message"
    return Candidate(
        item.text, float(score), kind=kind, role=role, timestamp=item.timestamp, tags=item.tags
    )


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

    def profile_block(self) -> str: ...                                      # durable profile facts alone, or '' (Recall-B)

    def last_session_summary(self) -> str: ...                              # one-shot prior-session head, or '' (lm-2 Wire 3)

    def prune(self) -> int: ...                                              # retention/eviction

    def close(self) -> None: ...                                             # flush + release


# Default working-window cap (Layer-1 RAM). Keeps the in-RAM store from growing
# unbounded over a long session; the oldest items fall off the front.
DEFAULT_MAX_ITEMS = 200


class SessionMemory:
    """In-memory session store used by tests and the local prototype.

    Conforms to :class:`Memory`. The desktop default when ``DATABASE_URL`` is
    unset.

    Recall (``context_for_llm``) gathers keyword-overlap candidates and hands
    them to the shared :func:`always_on_agent.recall.build_block`, so the in-RAM
    path and the Postgres path apply the *same* token-budgeted, deduped,
    adaptive-cutoff selection and emit a byte-identical block for identical
    candidates. The legacy fixed ``overlap >= 2`` gate, ``top-3`` cap, and
    ``[:150]`` per-item char slice are gone -- the budget + adaptive cutoff
    replace them (see the module docstring)."""

    def __init__(self, max_items: int = DEFAULT_MAX_ITEMS, *, budget: RecallBudget | None = None):
        self._items: list[MemoryItem] = []
        self._max_items = max(1, int(max_items))
        self._budget = budget or RecallBudget()

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

    def _candidates(self, query: str) -> list[Candidate]:
        """Keyword-overlap candidates via the shared :func:`candidate_for_item`
        (the single source of truth that keeps the RAM and SQLite backends
        byte-identical at the seam)."""
        q_norm = normalize_text(query)
        q_words = set(q_norm.split())
        if not q_words:
            return []
        out: list[Candidate] = []
        for item in self._items:
            cand = candidate_for_item(item, q_words, q_norm)
            if cand is not None:
                out.append(cand)
        return out

    def context_for_llm(self, query: str) -> str:
        """Token-budgeted recall block, or ``''`` on no hit. See class docstring."""
        return build_block(self._candidates(query), query, self._budget)

    def profile_block(self) -> str:
        # The in-RAM store has no durable user-profile tier (that is a
        # Postgres-only producer), so the decoupled profile-injection path is a
        # no-op here -- byte-identical to today.
        return ""

    def last_session_summary(self) -> str:
        # No rolling-summary head in the in-RAM store; the one-shot "Last session"
        # block is Postgres-only (lm-2 Wire 3).
        return ""

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
        recall_budget: Optional[RecallBudget] = None,
        working_window: Optional[int] = None,
        cross_session_continuity: bool = False,
        persist_assistant: bool = False,
        **manager_kwargs,
    ):
        from utils.memory import create_memory_manager  # lazy: keep the brain DB-free

        # P2b knobs forwarded to the engine: the rolling-summary LLM callable
        # (keyword fallback inside the manager when None), the default-off
        # Postgres-only profile producer, and the episodic/summary age-TTLs that
        # prune() -> apply_retention() enforces. user_profile is never TTL'd.
        # recall_budget is the shared token budget so the Postgres recall block
        # is bounded by the SAME contract as the in-RAM SessionMemory.
        self._manager = create_memory_manager(
            summarizer=summarizer,
            profile_enabled=profile_enabled,
            episodic_ttl_days=episodic_ttl_days,
            summary_ttl_days=summary_ttl_days,
            recall_budget=recall_budget,
            cross_session_continuity=cross_session_continuity,
            # lm-5: persist + recall assistant finals (default OFF) so the Postgres
            # tier matches the in-RAM/SQLite backends, which already store and
            # recall assistant_output items.
            persist_assistant=persist_assistant,
            **manager_kwargs,
        )
        # Our own small in-RAM ring buffer of the raw (text, tags) handed to
        # add() (R3). MemoryManager.recent_messages keeps only role + junk-
        # filters, so all() reads back from here to preserve tag fidelity --
        # both backends then behave identically (test_addressing relies on the
        # 'ingested' tag surviving).
        self._ring: list[MemoryItem] = []
        # lm-7 window parity: size the all() ring off working_window so it matches
        # SessionMemory and SqliteVecMemory (which both use working_window). Falls
        # back to the manager's max_recent_messages, then DEFAULT_MAX_ITEMS, so a
        # caller that doesn't pass working_window keeps the prior behavior.
        self._max_items = (
            int(working_window) if working_window
            else (int(getattr(self._manager, "max_recent_messages", 0)) or DEFAULT_MAX_ITEMS)
        )

    def add(self, text: str, tags: tuple[str, ...] = ()) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        # Keep the raw (text, tags) regardless of how it is routed/persisted so
        # all() stays tag-faithful.
        self._ring.append(MemoryItem(cleaned, tags or keywords(cleaned)))
        if len(self._ring) > self._max_items:
            self._ring = self._ring[-self._max_items:]
        # Meeting notes are RAM-only (R7, §9.7 privacy): never hand them to the
        # persisting manager. (The old meeting_persist toggle was never wired --
        # the getattr always read False -- so RAM-only is now a hard guarantee.)
        if "meeting" in tags:
            return
        # Tag-routed: assistant output is RAM-only context; a VISUAL (screen)
        # observation persists via the dedicated add_observation path -- NOT
        # queue_user_utterance, whose _extract_profile would run the "my name is X"
        # regex over OCR'd screen text and write garbage profile rows; user/ingested
        # speech is queued for debounced persistence.
        if "assistant_output" in tags:
            self._manager.add_message("assistant", cleaned)
        elif "vision" in tags:
            add_observation = getattr(self._manager, "add_observation", None)
            if callable(add_observation):
                add_observation(cleaned)
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

    def profile_block(self) -> str:
        # Recall-B: durable profile facts alone, so they inject even when episodic
        # recall is OFF. '' unless the manager has profile_enabled + facts.
        return self._manager.get_profile_context()

    def last_session_summary(self) -> str:
        # lm-2 Wire 3: the prior-session head seeded at process start, or '' when
        # cross_session_continuity is OFF / there was no prior summary.
        return self._manager.last_session_summary()

    def prune(self) -> int:
        # Age-TTL retention (P2b): episodic messages older than
        # episodic_ttl_days are summarized-then-evicted, summaries older than
        # summary_ttl_days are dropped, user_profile is never TTL'd. The manager
        # no-ops + returns 0 without a live DB. Invoked from VoiceRuntime.stop()
        # at close-time (R6). Returns rows removed.
        return self._manager.apply_retention()

    def close(self) -> None:
        self._manager.close()
