from __future__ import annotations

from dataclasses import dataclass, field
import time

from .text import keywords, normalize_text


@dataclass(frozen=True)
class MemoryItem:
    text: str
    tags: tuple[str, ...] = ()
    timestamp: float = field(default_factory=time.time)


class SessionMemory:
    """In-memory session store used by tests and the local prototype."""

    def __init__(self):
        self._items: list[MemoryItem] = []

    def add(self, text: str, tags: tuple[str, ...] = ()) -> None:
        cleaned = text.strip()
        if cleaned:
            self._items.append(MemoryItem(cleaned, tags or keywords(cleaned)))

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
