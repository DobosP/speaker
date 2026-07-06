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

    # Soft high-water mark: the queue stays unbounded (publish must never block
    # a capture/playback-adjacent thread), but a backlog this deep means the
    # dispatch thread is stalled or was never started -- WARN once per crossing
    # (re-armed at half) so the run bundle shows it instead of silent growth.
    _HIGH_WATER = 512

    def __init__(self):
        self._queue: PriorityQueue[tuple[int, int, AgentEvent]] = PriorityQueue()
        self._counter = itertools.count()
        self._handlers: list[EventHandler] = []
        self._stop = Event()
        self._thread: Thread | None = None
        self._high_water_warned = False

    def subscribe(self, handler: EventHandler) -> None:
        self._handlers.append(handler)

    def publish(self, event: AgentEvent) -> None:
        self._queue.put((event.priority, next(self._counter), event))
        depth = self._queue.qsize()
        if depth >= self._HIGH_WATER:
            if not self._high_water_warned:
                self._high_water_warned = True
                log.warning(
                    "event bus backlog %d >= %d -- dispatch thread stalled or "
                    "not started?", depth, self._HIGH_WATER,
                )
        elif self._high_water_warned and depth < self._HIGH_WATER // 2:
            self._high_water_warned = False

    def idle(self) -> bool:
        """True iff every published event has been dispatched AND handled.

        ``unfinished_tasks`` (the task_done bookkeeping) covers the window where
        the bus thread has popped an event but a handler is still running --
        ``empty()`` alone would call that idle. Lock-free read; callers poll."""
        return self._queue.unfinished_tasks == 0

    def drain_once(self) -> bool:
        from queue import Empty

        # get_nowait (not an empty() pre-check) so a concurrent consumer -- e.g.
        # wait_idle polling while the bus thread runs (rc-1) -- degrades to a
        # clean False instead of an uncaught queue.Empty TOCTOU crash.
        try:
            _, _, event = self._queue.get_nowait()
        except Empty:
            return False
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

    def stop(self, *, drain: bool = False) -> None:
        """Stop the dispatch thread. Still-queued events are DISCARDED by default.

        The bus thread exits between ``get`` timeouts, so anything published
        before ``stop()`` but not yet popped stays queued. Discarding is the
        correct shutdown default: the queue may hold ACTION-producing events
        (e.g. a ``TTS_REQUEST`` whose handler starts speaking), and executing
        those after the runtime decided to stop is worse than dropping them
        (codex-review 2026-07-06 caught exactly that on ``VoiceRuntime.stop``).
        The loss is visible, not silent: ``idle()`` stays False and the
        high-water warn covers pathological backlogs. Pass ``drain=True`` to
        instead dispatch the leftovers on the caller's thread through the same
        exception-guarded path as live dispatch -- only for buses whose
        handlers are pure/log-only."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if drain:
            self.drain()

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
