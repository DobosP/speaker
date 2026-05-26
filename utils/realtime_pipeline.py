"""
Lightweight message types and queue helpers for the realtime voice pipeline.

Used to keep turn/speak-session invalidation and bounded queues explicit without
a full actor framework.
"""
from __future__ import annotations

import queue
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar

T = TypeVar("T")


class ControlKind(str, Enum):
    STOP = "stop"
    PAUSE = "pause"
    CANCEL = "cancel"
    QUIT = "quit"


@dataclass(frozen=True)
class AudioFrame:
    """Mic frame metadata (raw samples live in numpy elsewhere)."""
    samples: int
    turn_generation: int


@dataclass(frozen=True)
class PartialText:
    text: str
    turn_generation: int


@dataclass(frozen=True)
class FinalText:
    text: str
    turn_generation: int


@dataclass(frozen=True)
class ControlIntent:
    kind: ControlKind
    source: str = "router"


@dataclass(frozen=True)
class LlmChunk:
    text: str
    speak_session: int


@dataclass(frozen=True)
class TtsItem:
    text: str
    speak_session: int


@dataclass(frozen=True)
class CancelTurn:
    reason: str = "user"
    turn_generation: int = 0


def flush_queue(q: "queue.Queue[T]", max_items: int = 10_000) -> int:
    """Drop up to max_items pending tasks; each get is paired with task_done()."""
    n = 0
    while n < max_items:
        try:
            q.get_nowait()
        except queue.Empty:
            break
        q.task_done()
        n += 1
    return n
