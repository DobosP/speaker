from __future__ import annotations

import logging
from queue import PriorityQueue
from threading import Event, Thread
from typing import Callable
import itertools

from .events import AgentEvent

log = logging.getLogger("speaker.event_bus")


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
                try:
                    handler(event)
                except Exception:  # noqa: BLE001 - one bad handler must not stop the drain
                    log.exception("event handler raised on %s; dropping it", event.kind)
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
        from queue import Empty

        while not self._stop.is_set():
            try:
                _, _, event = self._queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                for handler in list(self._handlers):
                    try:
                        handler(event)
                    except Exception:  # noqa: BLE001
                        # A handler bug must degrade to a dropped event, never
                        # silently kill the single bus thread (which would make
                        # the whole assistant go dead -- no transcripts, no TTS,
                        # no task lifecycle -- until restart). Log and carry on.
                        log.exception("event handler raised on %s; dropping it", event.kind)
            finally:
                self._queue.task_done()
