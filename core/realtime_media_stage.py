"""A bounded, route-neutral handoff for process-local media work.

The stage owns admission and lifecycle bookkeeping, not the payload's concrete
type. Callers transfer one opaque payload with :meth:`offer_nowait`; retirement
returns the exact item so its caller can publish a terminal outcome before
recycling any resource. Payloads are never copied, inspected, serialized, or
included in diagnostic representations.

Capture scopes are exact identities, not an ordering policy.  Retiring one
scope fences only items with that exact ``(capture_epoch, capture_generation)``
pair; a later offer with a numerically older scope is still admissible.  This
lets the route owner, rather than a generic queue, decide when a capture
generation is stale.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from threading import Condition, Event, Lock, TIMEOUT_MAX
from typing import Generic, TypeVar


PayloadT = TypeVar("PayloadT")


def _non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _positive_int(name: str, value: int) -> int:
    result = _non_negative_int(name, value)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _bounded_timeout(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("timeout must be a finite non-negative number or None")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout < 0.0 or timeout > TIMEOUT_MAX:
        raise ValueError("timeout must be a finite non-negative number or None")
    return timeout


@dataclass(frozen=True, slots=True)
class CaptureScope:
    """Exact capture continuity identity carried by a stage item."""

    capture_epoch: int
    capture_generation: int

    def __post_init__(self) -> None:
        _non_negative_int("capture_epoch", self.capture_epoch)
        _non_negative_int("capture_generation", self.capture_generation)


@dataclass(frozen=True, slots=True)
class StageItem(Generic[PayloadT]):
    """One admitted payload and its independently observable cancel fence.

    The dataclass is immutable, but the opaque payload may itself be mutable.
    Ownership, copying, and mutation policy remain the caller's responsibility.
    ``cancel_event`` is set when overflow, explicit scope retirement, or close
    invalidates the item.  An in-flight consumer must still call ``finish``.
    """

    sequence: int
    scope: CaptureScope
    payload: PayloadT = field(repr=False, compare=False)
    cancel_event: Event = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _positive_int("sequence", self.sequence)
        if not isinstance(self.scope, CaptureScope):
            raise TypeError("scope must be a CaptureScope")
        if not isinstance(self.cancel_event, Event):
            raise TypeError("cancel_event must be a threading.Event")

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()


@dataclass(frozen=True, slots=True)
class StageRetirement(Generic[PayloadT]):
    """Exact queued items plus metadata from an explicit retirement.

    Retired items are hidden from representations because they retain opaque
    payloads.  The caller knows the operation's reason and can terminalize each
    exact item after stage locking has ended.  In-flight payloads remain with
    their consumers and are deliberately represented only by a count.
    """

    queued_released: int
    in_flight_cancelled: int
    retired: tuple[StageItem[PayloadT], ...] = field(
        default=(),
        repr=False,
        compare=False,
    )

    @property
    def retired_sequences(self) -> tuple[int, ...]:
        return tuple(item.sequence for item in self.retired)


@dataclass(frozen=True, slots=True)
class StageOfferResult(Generic[PayloadT]):
    """Exact outcome of one non-blocking offer.

    Item fields are omitted from the representation because they retain opaque
    payloads.  A closed-stage rejection still returns a newly sequenced,
    cancellation-fenced item so its producer can publish an exact terminal
    outcome.  The producer retains ownership of that rejected payload.
    """

    accepted: bool
    item: StageItem[PayloadT] = field(repr=False, compare=False)
    retired: tuple[StageItem[PayloadT], ...] = field(
        default=(),
        repr=False,
        compare=False,
    )

    @property
    def retired_sequences(self) -> tuple[int, ...]:
        return tuple(item.sequence for item in self.retired)


@dataclass(frozen=True, slots=True)
class StageSnapshot:
    """Small metadata-only diagnostic view of stage state."""

    max_queued: int
    closed: bool
    queued: int
    in_flight: int
    offered: int
    accepted: int
    rejected_closed: int
    overflow_released: int
    scope_released: int
    scope_in_flight_cancelled: int
    close_released: int
    close_in_flight_cancelled: int
    finished: int


class RealtimeMediaStage(Generic[PayloadT]):
    """Physically bounded FIFO handoff with exact-scope cancellation.

    Queue capacity applies to waiting items.  ``take`` transfers an item to the
    in-flight set, where it remains until its exact object is passed to
    ``finish``.  A route normally has a fixed, small number of consumers, so
    in-flight concurrency is bounded by those consumers rather than by growing
    storage inside this class.

    Overflow, scope retirement, and close return exact queued items after stage
    ownership ends. No cleanup callback runs implicitly: the recipient must
    publish any terminal outcome before recycling or mutating the payload.
    """

    def __init__(self, max_queued: int) -> None:
        self._max_queued = _positive_int("max_queued", max_queued)
        self._condition = Condition(Lock())
        self._queue: deque[StageItem[PayloadT]] = deque()
        self._in_flight: dict[int, StageItem[PayloadT]] = {}
        self._next_sequence = 1
        self._closed = False

        self._offered = 0
        self._accepted = 0
        self._rejected_closed = 0
        self._overflow_released = 0
        self._scope_released = 0
        self._scope_in_flight_cancelled = 0
        self._close_released = 0
        self._close_in_flight_cancelled = 0
        self._finished = 0

    def offer_nowait(
        self,
        scope: CaptureScope,
        payload: PayloadT,
    ) -> StageOfferResult[PayloadT]:
        """Admit immediately, retiring the oldest queued item at capacity.

        The method never waits for capacity.  A rejected result means the stage
        was already closed and the caller retains payload ownership.  An
        accepted result transfers ownership. Any overflow-retired predecessor
        is returned intact for caller-owned terminalization and cleanup.
        """

        if not isinstance(scope, CaptureScope):
            raise TypeError("scope must be a CaptureScope")

        retired: StageItem[PayloadT] | None = None
        with self._condition:
            self._offered += 1
            item = StageItem(
                sequence=self._next_sequence,
                scope=scope,
                payload=payload,
                cancel_event=Event(),
            )
            self._next_sequence += 1
            if self._closed:
                self._rejected_closed += 1
                item.cancel_event.set()
                return StageOfferResult(accepted=False, item=item)

            if len(self._queue) >= self._max_queued:
                retired = self._queue.popleft()
                retired.cancel_event.set()
                self._overflow_released += 1
            self._queue.append(item)
            self._accepted += 1
            self._condition.notify()

        return StageOfferResult(
            accepted=True,
            item=item,
            retired=(() if retired is None else (retired,)),
        )

    def take(self, timeout: float | None = None) -> StageItem[PayloadT] | None:
        """Move the oldest queued item to in-flight, waiting only as requested."""

        timeout = _bounded_timeout(timeout)
        with self._condition:
            if timeout is None:
                while not self._queue and not self._closed:
                    self._condition.wait()
            else:
                self._condition.wait_for(
                    lambda: bool(self._queue) or self._closed,
                    timeout=timeout,
                )
            if not self._queue:
                return None
            item = self._queue.popleft()
            self._in_flight[item.sequence] = item
            return item

    def finish(self, item: StageItem[PayloadT]) -> bool:
        """Release one exact in-flight item.

        Returns false for a queued, foreign, or already-finished item.  Object
        identity is intentional: payload equality may be expensive, ambiguous,
        or undefined for a caller-owned media type.
        """

        if not isinstance(item, StageItem):
            raise TypeError("item must be a StageItem")
        with self._condition:
            current = self._in_flight.get(item.sequence)
            if current is not item:
                return False
            del self._in_flight[item.sequence]
            self._finished += 1

        return True

    def retire_scope(self, scope: CaptureScope) -> StageRetirement[PayloadT]:
        """Fence and retire items with one exact scope.

        Matching queued items leave stage ownership immediately and are returned
        after unlocking. Matching in-flight items remain owned by their
        consumers, receive cancellation once, and leave ownership only through
        ``finish``. Unrelated items retain FIFO order.
        """

        if not isinstance(scope, CaptureScope):
            raise TypeError("scope must be a CaptureScope")

        queued: list[StageItem[PayloadT]] = []
        in_flight_cancelled = 0
        with self._condition:
            retained: deque[StageItem[PayloadT]] = deque()
            while self._queue:
                item = self._queue.popleft()
                if item.scope == scope:
                    item.cancel_event.set()
                    queued.append(item)
                else:
                    retained.append(item)
            self._queue = retained

            for item in self._in_flight.values():
                if item.scope == scope and not item.cancel_event.is_set():
                    item.cancel_event.set()
                    in_flight_cancelled += 1

            self._scope_released += len(queued)
            self._scope_in_flight_cancelled += in_flight_cancelled

        return StageRetirement(
            queued_released=len(queued),
            in_flight_cancelled=in_flight_cancelled,
            retired=tuple(queued),
        )

    def close(self) -> StageRetirement[PayloadT]:
        """Fence the stage, wake consumers, and retire all queued ownership."""

        queued: tuple[StageItem[PayloadT], ...]
        in_flight_cancelled = 0
        with self._condition:
            if self._closed:
                return StageRetirement(0, 0)
            self._closed = True
            queued = tuple(self._queue)
            self._queue.clear()
            for item in queued:
                item.cancel_event.set()
            for item in self._in_flight.values():
                if not item.cancel_event.is_set():
                    item.cancel_event.set()
                    in_flight_cancelled += 1
            self._close_released += len(queued)
            self._close_in_flight_cancelled += in_flight_cancelled
            self._condition.notify_all()

        return StageRetirement(
            queued_released=len(queued),
            in_flight_cancelled=in_flight_cancelled,
            retired=queued,
        )

    def snapshot(self) -> StageSnapshot:
        """Return counters and depths without exposing payloads or item objects."""

        with self._condition:
            return StageSnapshot(
                max_queued=self._max_queued,
                closed=self._closed,
                queued=len(self._queue),
                in_flight=len(self._in_flight),
                offered=self._offered,
                accepted=self._accepted,
                rejected_closed=self._rejected_closed,
                overflow_released=self._overflow_released,
                scope_released=self._scope_released,
                scope_in_flight_cancelled=self._scope_in_flight_cancelled,
                close_released=self._close_released,
                close_in_flight_cancelled=self._close_in_flight_cancelled,
                finished=self._finished,
            )


__all__ = [
    "CaptureScope",
    "RealtimeMediaStage",
    "StageItem",
    "StageOfferResult",
    "StageRetirement",
    "StageSnapshot",
]
