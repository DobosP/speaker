"""One exact native output teardown, bounded for every caller.

A timeout/failed or ambiguous launch is permanently uncertain for this owner.
The task and handle remain retained; another caller never starts another closer.
No platform audio imports or transcript data belong here.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True, slots=True)
class OutputCleanupSnapshot:
    closed: bool
    active: bool
    poisoned: bool


class OutputCleanupOwner:
    def __init__(self, stream: object, *, thread_factory: Callable = threading.Thread):
        self.stream = stream
        self._thread_factory = thread_factory
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._thread = None
        self._entered = False
        self._succeeded = False
        self._poisoned = False

    def _run(self) -> None:
        stop_ok = close_ok = False
        try:
            try:
                self.stream.stop()
                stop_ok = True
            except BaseException:
                pass
            try:
                self.stream.close()
                close_ok = True
            except BaseException:
                pass
        finally:
            with self._lock:
                self._succeeded = stop_ok and close_ok
                if not self._succeeded:
                    self._poisoned = True
            self._done.set()

    def snapshot(self) -> OutputCleanupSnapshot:
        with self._lock:
            thread = self._thread
            active = bool(thread is not None and thread.is_alive())
            closed = bool(self._done.is_set() and not active and self._succeeded and not self._poisoned)
            return OutputCleanupSnapshot(closed, active, self._poisoned)

    def close(self, timeout: float = 1.0) -> bool:
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError("output cleanup timeout must be finite and nonnegative")
        deadline = time.monotonic() + min(1.0, timeout)
        with self._lock:
            if self._poisoned:
                return False
            if not self._entered:
                self._entered = True
                try:
                    self._thread = self._thread_factory(
                        target=self._run, name="sherpa-output-cleanup", daemon=True
                    )
                    self._thread.start()
                except BaseException as error:
                    # start can throw after OS-thread admission. Retain that exact
                    # task/stream forever; a second closer is never authorized.
                    self._poisoned = True
                    if not isinstance(error, Exception):
                        raise
                    return False
            thread = self._thread
        if thread is threading.current_thread():
            return False
        try:
            self._done.wait(max(0.0, deadline - time.monotonic()))
            if thread is not None and self._done.is_set():
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
        except BaseException:
            with self._lock:
                self._poisoned = True
            raise
        snapshot = self.snapshot()
        if snapshot.closed:
            return True
        with self._lock:
            self._poisoned = True
        return False
