from __future__ import annotations

from queue import PriorityQueue
from threading import Event, Thread
from typing import Callable
import itertools

from .events import AgentEvent


EventHandler = Callable[[AgentEvent], None]


class EventBus:
    """Small priority event bus for the prototype supervisor."""

    def __init__(self):
        self._queue: PriorityQueue[tuple[int, int, AgentEvent]] = PriorityQueue()
        self._counter = itertools.count()
        self._handlers: list[EventHandler] = []
        self._stop = Event()
        self._thread: Thread | None = None

    def subscribe(self, handler: EventHandler) -> None:
        self._handlers.append(handler)

    def publish(self, event: AgentEvent) -> None:
        self._queue.put((event.priority, next(self._counter), event))

    def drain_once(self) -> bool:
        if self._queue.empty():
            return False
        _, _, event = self._queue.get_nowait()
        try:
            for handler in list(self._handlers):
                handler(event)
        finally:
            self._queue.task_done()
        return True

    def drain(self) -> int:
        count = 0
        while self.drain_once():
            count += 1
        return count

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.drain_once()
