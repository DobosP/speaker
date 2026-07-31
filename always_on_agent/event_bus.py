from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import itertools
import logging
from queue import Empty
from threading import Condition, Event, Lock, Thread
import time
from typing import Callable

from .acoustic import AcousticLineage
from .events import AgentEvent, EventKind

log = logging.getLogger("speaker.event_bus")


EventHandler = Callable[[AgentEvent], None]


class MailboxLane(str, Enum):
    """Admission lane, independent from an event's dispatch priority."""

    DATA = "data"
    CONTROL = "control"


@dataclass(frozen=True)
class EventMailboxPolicy:
    lane: MailboxLane
    coalesce_key: tuple[object, ...] | None = None
    revision: int | None = None
    span_keys: frozenset[tuple[object, ...]] = frozenset()
    retire_span_keys: frozenset[tuple[object, ...]] = frozenset()
    preemptive: bool = False
    ordered_output: bool = False
    reject_unscoped_control: bool = False


_EVENT_LANES: dict[EventKind, MailboxLane] = {
    EventKind.STT_PARTIAL: MailboxLane.DATA,
    EventKind.STT_FINAL: MailboxLane.CONTROL,
    EventKind.STT_ABORTED: MailboxLane.CONTROL,
    EventKind.SPEECH_OBSERVATION: MailboxLane.DATA,
    EventKind.INTENT_DECISION: MailboxLane.DATA,
    EventKind.CONTROL_STOP: MailboxLane.CONTROL,
    EventKind.CONTROL_MODE: MailboxLane.CONTROL,
    EventKind.CONTROL_CONFIRM: MailboxLane.CONTROL,
    EventKind.CONTROL_DENY: MailboxLane.CONTROL,
    EventKind.TASK_STARTED: MailboxLane.CONTROL,
    EventKind.TASK_PROGRESS: MailboxLane.DATA,
    EventKind.TASK_COMPLETED: MailboxLane.CONTROL,
    EventKind.TASK_CANCELLED: MailboxLane.CONTROL,
    EventKind.TASK_FAILED: MailboxLane.CONTROL,
    EventKind.TTS_REQUEST: MailboxLane.CONTROL,
    EventKind.TTS_STREAM_END: MailboxLane.CONTROL,
    EventKind.MEMORY_COMMIT: MailboxLane.CONTROL,
    EventKind.FOLLOWUP_TICK: MailboxLane.DATA,
    EventKind.CAPTURE_STATE: MailboxLane.DATA,
    EventKind.MAILBOX_FAULT: MailboxLane.CONTROL,
}

_ORDERED_OUTPUT_KINDS = frozenset(
    {EventKind.TTS_REQUEST, EventKind.TTS_STREAM_END}
)

if frozenset(_EVENT_LANES) != frozenset(EventKind):
    raise RuntimeError("every EventKind requires an explicit mailbox lane")


def classified_event_kinds() -> frozenset[EventKind]:
    """Kinds with an explicit admission policy.

    Kept public so contract tests fail as soon as a new event kind is added
    without a bounded-delivery decision.
    """

    return frozenset(_EVENT_LANES)


def _acoustic_key(event: AgentEvent) -> tuple[object, ...] | None:
    lineage = AcousticLineage.try_from_payload(event.payload.get("acoustic"))
    if lineage is None:
        return None
    return tuple(span.key for span in lineage.spans)


def _event_revision(event: AgentEvent) -> int | None:
    value = event.payload.get("revision")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _partial_is_exact_control(event: AgentEvent) -> bool:
    # This pure classifier is already the shared fail-closed fence used by
    # upstream transcript selection. Import lazily to keep event schema loading
    # independent from the analyzer module.
    from .speech_analyzer import exact_control_class

    control = exact_control_class(str(event.payload.get("text", "")))
    return control is not None and control[0] in {"stop", "mode_switch"}


def event_mailbox_policy(event: AgentEvent) -> EventMailboxPolicy:
    """Return the typed admission policy for one event.

    Admission protects effectful/lifecycle events so a transcript/progress
    storm cannot consume their reserved capacity. Non-output control work also
    preempts data at dispatch; ordered output retains priority/FIFO with a
    bounded age promotion.
    """

    lane = _EVENT_LANES[event.kind]
    if event.kind == EventKind.STT_PARTIAL:
        acoustic_key = _acoustic_key(event)
        revision = _event_revision(event)
        exact_control = _partial_is_exact_control(event)
        if acoustic_key is not None and revision is not None:
            return EventMailboxPolicy(
                (
                    MailboxLane.CONTROL
                    if exact_control
                    else MailboxLane.DATA
                ),
                coalesce_key=(EventKind.STT_PARTIAL, *acoustic_key),
                revision=revision,
                span_keys=frozenset(acoustic_key),
                preemptive=exact_control,
            )
        if exact_control:
            # An actionable partial without valid acoustic identity cannot be
            # corrected safely across participants. Text-only callers must wait
            # for a final or publish an explicit control event.
            return EventMailboxPolicy(
                MailboxLane.CONTROL,
                preemptive=True,
                reject_unscoped_control=True,
            )
    if event.kind in {EventKind.STT_FINAL, EventKind.STT_ABORTED}:
        acoustic_key = _acoustic_key(event)
        return EventMailboxPolicy(
            lane,
            revision=_event_revision(event),
            retire_span_keys=(
                frozenset(acoustic_key)
                if acoustic_key is not None
                else frozenset()
            ),
            preemptive=True,
        )
    if (
        event.kind == EventKind.CAPTURE_STATE
        and event.payload.get("state") == "fatal"
    ):
        return EventMailboxPolicy(MailboxLane.CONTROL, preemptive=True)
    return EventMailboxPolicy(
        lane,
        preemptive=(
            lane == MailboxLane.CONTROL
            and event.kind not in _ORDERED_OUTPUT_KINDS
        ),
        ordered_output=event.kind in _ORDERED_OUTPUT_KINDS,
    )


@dataclass(frozen=True)
class EventBusSnapshot:
    max_events: int
    control_reserve: int
    max_output_bypass: int
    pending: int
    pending_data: int
    pending_control: int
    outstanding: int
    peak_pending: int
    offered: int
    admitted: int
    handled: int
    coalesced: int
    evicted_data: int
    retired_partials: int
    rejected_stale: int
    rejected_data: int
    rejected_unscoped_control: int
    forced_output_dispatches: int
    mailbox_faults: int
    fault_discarded: int
    rejected_after_fault: int
    faulted: bool


@dataclass(frozen=True)
class _MailboxItem:
    event: AgentEvent
    sequence: int
    policy: EventMailboxPolicy
    admitted_dispatch: int

    @property
    def order(self) -> tuple[int, int]:
        return (self.event.priority, self.sequence)


class _Admission(str, Enum):
    ENQUEUED = "enqueued"
    COALESCED = "coalesced"
    EVICTED = "evicted"
    REJECTED_STALE = "rejected_stale"
    REJECTED_DATA = "rejected_data"
    REJECTED_UNSCOPED_CONTROL = "rejected_unscoped_control"
    FAULTED = "faulted"
    REJECTED_FAULTED = "rejected_faulted"


class _BoundedEventMailbox:
    """Finite thread-safe storage with exact Queue-style task accounting."""

    def __init__(
        self,
        *,
        max_events: int,
        control_reserve: int,
        max_output_bypass: int,
    ):
        if (
            isinstance(max_events, bool)
            or not isinstance(max_events, int)
            or max_events < 2
        ):
            raise ValueError("max_events must be an integer >= 2")
        if (
            isinstance(control_reserve, bool)
            or not isinstance(control_reserve, int)
            or control_reserve < 1
            or control_reserve >= max_events
        ):
            raise ValueError(
                "control_reserve must be an integer in [1, max_events)"
            )
        if (
            isinstance(max_output_bypass, bool)
            or not isinstance(max_output_bypass, int)
            or max_output_bypass < 1
        ):
            raise ValueError("max_output_bypass must be an integer >= 1")
        self.max_events = max_events
        self.control_reserve = control_reserve
        self.max_output_bypass = max_output_bypass
        self.data_capacity = max_events - control_reserve
        self._changed = Condition()
        self._pending: list[_MailboxItem] = []
        self._pending_data = 0
        self._pending_control = 0
        self._unfinished_tasks = 0
        self._peak_pending = 0
        self._offered = 0
        self._admitted = 0
        self._handled = 0
        self._coalesced = 0
        self._evicted_data = 0
        self._retired_partials = 0
        self._rejected_stale = 0
        self._rejected_data = 0
        self._rejected_unscoped_control = 0
        self._forced_output_dispatches = 0
        self._dispatch_count = 0
        self._mailbox_faults = 0
        self._fault_discarded = 0
        self._rejected_after_fault = 0
        self._faulted = False

    @property
    def unfinished_tasks(self) -> int:
        with self._changed:
            return self._unfinished_tasks

    def qsize(self) -> int:
        with self._changed:
            return len(self._pending)

    def empty(self) -> bool:
        return self.qsize() == 0

    def _remove_pending(self, index: int) -> _MailboxItem:
        item = self._pending.pop(index)
        if item.policy.lane == MailboxLane.DATA:
            self._pending_data -= 1
        else:
            self._pending_control -= 1
        return item

    def _retire_covered_partials(
        self,
        terminal_span_keys: frozenset[tuple[object, ...]],
        terminal_revision: int | None,
    ) -> bool:
        matches = [
            index
            for index, item in enumerate(self._pending)
            if (
                item.policy.coalesce_key is not None
                and bool(item.policy.span_keys)
                and item.policy.span_keys.issubset(terminal_span_keys)
            )
        ]
        if not matches:
            return True
        if terminal_revision is None or any(
            self._pending[index].policy.revision is None
            or self._pending[index].policy.revision >= terminal_revision
            for index in matches
        ):
            # A final/abort must be strictly newer than an already queued
            # hypothesis. Upstream lineage gates normally reject non-increasing
            # revisions, but the mailbox is the last serialized boundary.
            self._rejected_stale += 1
            return False
        for index in reversed(matches):
            self._remove_pending(index)
            self._unfinished_tasks -= 1
            self._retired_partials += 1
        return True

    def _coalesce(
        self,
        *,
        event: AgentEvent,
        sequence: int,
        policy: EventMailboxPolicy,
    ) -> _Admission | None:
        if policy.coalesce_key is None:
            return None
        for index, item in enumerate(self._pending):
            if item.policy.coalesce_key != policy.coalesce_key:
                continue
            if (
                item.policy.revision is None
                or policy.revision is None
                or policy.revision <= item.policy.revision
            ):
                self._rejected_stale += 1
                return _Admission.REJECTED_STALE
            old_lane = item.policy.lane
            admission = _Admission.COALESCED
            if (
                old_lane == MailboxLane.CONTROL
                and policy.lane == MailboxLane.DATA
                and self._pending_data >= self.data_capacity
            ):
                victim = self._data_victim(event)
                if victim is None:
                    # The correction cannot take a data slot, but the older
                    # exact-control hypothesis is now known stale. Retire it
                    # rather than letting it act after its correction.
                    self._remove_pending(index)
                    self._unfinished_tasks -= 1
                    self._rejected_data += 1
                    return _Admission.REJECTED_DATA
                self._remove_pending(victim)
                self._unfinished_tasks -= 1
                self._evicted_data += 1
                admission = _Admission.EVICTED
                if victim < index:
                    index -= 1
            # A newer revision replaces its queued snapshot but retains its real
            # publication order. This prevents post-STOP updates from inheriting
            # a pre-STOP sequence and leapfrogging the control.
            self._pending[index] = _MailboxItem(
                event=event,
                sequence=sequence,
                policy=policy,
                admitted_dispatch=self._dispatch_count,
            )
            if old_lane != policy.lane:
                if old_lane == MailboxLane.DATA:
                    self._pending_data -= 1
                    self._pending_control += 1
                else:
                    self._pending_control -= 1
                    self._pending_data += 1
            self._admitted += 1
            self._coalesced += 1
            self._changed.notify()
            return admission
        return None

    def _data_victim(
        self,
        incoming: AgentEvent | None = None,
    ) -> int | None:
        candidates = [
            (index, item)
            for index, item in enumerate(self._pending)
            if (
                item.policy.lane == MailboxLane.DATA
                and (
                    incoming is None
                    or item.event.priority >= incoming.priority
                )
            )
        ]
        if not candidates:
            return None
        # Retain more urgent events (smaller numeric priority). Among equally
        # least-urgent candidates, evict the oldest for deterministic freshness.
        return max(
            candidates,
            key=lambda pair: (pair[1].event.priority, -pair[1].sequence),
        )[0]

    def _install_fault(
        self,
        *,
        rejected: AgentEvent,
        sequence: int,
    ) -> _Admission:
        discarded = len(self._pending)
        discarded_kinds: dict[str, int] = {}
        for item in self._pending:
            kind = item.event.kind.value
            discarded_kinds[kind] = discarded_kinds.get(kind, 0) + 1
        self._pending.clear()
        self._pending_data = 0
        self._pending_control = 0
        self._unfinished_tasks -= discarded
        self._fault_discarded += discarded
        self._mailbox_faults += 1
        self._faulted = True
        fault = AgentEvent(
            EventKind.MAILBOX_FAULT,
            {
                "reason": "critical_capacity_exhausted",
                "rejected_kind": rejected.kind.value,
                "discarded_pending": discarded,
                "discarded_kinds": dict(sorted(discarded_kinds.items())),
                "capacity": self.max_events,
            },
            priority=-100,
        )
        self._pending.append(
            _MailboxItem(
                event=fault,
                sequence=sequence,
                policy=EventMailboxPolicy(
                    MailboxLane.CONTROL,
                    preemptive=True,
                ),
                admitted_dispatch=self._dispatch_count,
            )
        )
        self._pending_control = 1
        self._unfinished_tasks += 1
        self._admitted += 1
        self._changed.notify_all()
        return _Admission.FAULTED

    def put(
        self,
        event: AgentEvent,
        sequence: int,
        policy: EventMailboxPolicy,
    ) -> _Admission:
        with self._changed:
            self._offered += 1
            if self._faulted:
                self._rejected_after_fault += 1
                return _Admission.REJECTED_FAULTED
            if policy.reject_unscoped_control:
                self._rejected_unscoped_control += 1
                return _Admission.REJECTED_UNSCOPED_CONTROL
            if (
                policy.retire_span_keys
                and not self._retire_covered_partials(
                    policy.retire_span_keys,
                    policy.revision,
                )
            ):
                return _Admission.REJECTED_STALE

            coalesced = self._coalesce(
                event=event,
                sequence=sequence,
                policy=policy,
            )
            if coalesced is not None:
                return coalesced

            admission = _Admission.ENQUEUED
            if policy.lane == MailboxLane.DATA:
                if (
                    self._pending_data >= self.data_capacity
                    or len(self._pending) >= self.max_events
                ):
                    victim = self._data_victim(event)
                    if victim is None:
                        self._rejected_data += 1
                        return _Admission.REJECTED_DATA
                    self._remove_pending(victim)
                    self._unfinished_tasks -= 1
                    self._evicted_data += 1
                    admission = _Admission.EVICTED
            elif len(self._pending) >= self.max_events:
                # A critical event may displace bounded diagnostic/provisional
                # data after consuming the explicit reserve. It must never
                # displace another critical event.
                victim = self._data_victim()
                if victim is None:
                    return self._install_fault(
                        rejected=event,
                        sequence=sequence,
                    )
                self._remove_pending(victim)
                self._unfinished_tasks -= 1
                self._evicted_data += 1
                admission = _Admission.EVICTED

            self._pending.append(
                _MailboxItem(
                    event=event,
                    sequence=sequence,
                    policy=policy,
                    admitted_dispatch=self._dispatch_count,
                )
            )
            if policy.lane == MailboxLane.DATA:
                self._pending_data += 1
            else:
                self._pending_control += 1
            self._unfinished_tasks += 1
            self._admitted += 1
            self._peak_pending = max(self._peak_pending, len(self._pending))
            self._changed.notify()
            return admission

    def _get_locked(self) -> AgentEvent:
        candidates = range(len(self._pending))
        priority_index = min(
            candidates,
            key=lambda candidate: self._pending[candidate].order,
        )
        preemptive = [
            candidate
            for candidate in candidates
            if self._pending[candidate].policy.preemptive
        ]
        if preemptive:
            index = min(
                preemptive,
                key=lambda candidate: self._pending[candidate].order,
            )
        else:
            index = priority_index
            ordered_output = [
                candidate
                for candidate in candidates
                if self._pending[candidate].policy.ordered_output
            ]
            if ordered_output:
                oldest_output = min(
                    ordered_output,
                    key=lambda candidate: self._pending[candidate].sequence,
                )
                waited = (
                    self._dispatch_count
                    - self._pending[oldest_output].admitted_dispatch
                )
                if (
                    waited >= self.max_output_bypass
                    and oldest_output != priority_index
                ):
                    index = oldest_output
                    self._forced_output_dispatches += 1
        # Advance before invoking handlers so re-entrant publications receive
        # the correct admission epoch.
        self._dispatch_count += 1
        return self._remove_pending(index).event

    def get_nowait(self) -> AgentEvent:
        with self._changed:
            if not self._pending:
                raise Empty
            return self._get_locked()

    def get(self, timeout: float | None = None) -> AgentEvent:
        with self._changed:
            deadline = None if timeout is None else time.monotonic() + timeout
            while not self._pending:
                if deadline is None:
                    self._changed.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise Empty
                self._changed.wait(remaining)
            return self._get_locked()

    def task_done(self) -> None:
        with self._changed:
            if self._unfinished_tasks <= 0:
                raise ValueError("task_done() called too many times")
            self._unfinished_tasks -= 1
            self._handled += 1
            self._changed.notify_all()

    def snapshot(self) -> EventBusSnapshot:
        with self._changed:
            return EventBusSnapshot(
                max_events=self.max_events,
                control_reserve=self.control_reserve,
                max_output_bypass=self.max_output_bypass,
                pending=len(self._pending),
                pending_data=self._pending_data,
                pending_control=self._pending_control,
                outstanding=self._unfinished_tasks,
                peak_pending=self._peak_pending,
                offered=self._offered,
                admitted=self._admitted,
                handled=self._handled,
                coalesced=self._coalesced,
                evicted_data=self._evicted_data,
                retired_partials=self._retired_partials,
                rejected_stale=self._rejected_stale,
                rejected_data=self._rejected_data,
                rejected_unscoped_control=self._rejected_unscoped_control,
                forced_output_dispatches=self._forced_output_dispatches,
                mailbox_faults=self._mailbox_faults,
                fault_discarded=self._fault_discarded,
                rejected_after_fault=self._rejected_after_fault,
                faulted=self._faulted,
            )


class EventBus:
    """Bounded priority event bus for the prototype supervisor.

    Data may use only ``max_events - control_reserve`` slots. Control and
    lifecycle events may borrow every unused slot, so a data storm cannot
    consume the reserve. Preemptive lifecycle/control events win dispatch;
    ordered output otherwise follows global ``(priority, publish order)`` with
    bounded age promotion over data.
    """

    _MAX_EVENTS = 512
    _CONTROL_RESERVE = 64
    _MAX_OUTPUT_BYPASS = 8

    def __init__(
        self,
        *,
        max_events: int = _MAX_EVENTS,
        control_reserve: int = _CONTROL_RESERVE,
        max_output_bypass: int = _MAX_OUTPUT_BYPASS,
    ):
        self._queue = _BoundedEventMailbox(
            max_events=max_events,
            control_reserve=control_reserve,
            max_output_bypass=max_output_bypass,
        )
        self._counter = itertools.count()
        self._handlers: list[EventHandler] = []
        self._stop = Event()
        self._thread: Thread | None = None
        self._pressure_lock = Lock()
        self._data_pressure_warned = False
        self._fault_warned = False
        self._unscoped_control_warned = False

    def subscribe(self, handler: EventHandler) -> None:
        self._handlers.append(handler)

    def publish(self, event: AgentEvent) -> bool:
        admission = self._queue.put(
            event,
            next(self._counter),
            event_mailbox_policy(event),
        )
        if admission == _Admission.FAULTED:
            snapshot = self.stats()
            with self._pressure_lock:
                if not self._fault_warned:
                    self._fault_warned = True
                    log.critical(
                        "event mailbox fault: rejected_kind=%s discarded=%d "
                        "max=%d; runtime shutdown required",
                        event.kind.value,
                        snapshot.fault_discarded,
                        snapshot.max_events,
                    )
        if admission == _Admission.REJECTED_UNSCOPED_CONTROL:
            snapshot = self.stats()
            with self._pressure_lock:
                if not self._unscoped_control_warned:
                    self._unscoped_control_warned = True
                    log.warning(
                        "rejecting actionable STT partial without valid "
                        "acoustic identity; count=%d",
                        snapshot.rejected_unscoped_control,
                    )
        if admission in {_Admission.EVICTED, _Admission.REJECTED_DATA}:
            snapshot = self.stats()
            with self._pressure_lock:
                if not self._data_pressure_warned:
                    self._data_pressure_warned = True
                    log.warning(
                        "event mailbox data pressure: outcome=%s pending_data=%d "
                        "data_capacity=%d",
                        admission.value,
                        snapshot.pending_data,
                        snapshot.max_events - snapshot.control_reserve,
                    )
        return admission in {
            _Admission.ENQUEUED,
            _Admission.COALESCED,
            _Admission.EVICTED,
        }

    @property
    def outstanding_count(self) -> int:
        """Accepted events still queued or executing inside a handler."""

        return self._queue.unfinished_tasks

    def stats(self) -> EventBusSnapshot:
        return self._queue.snapshot()

    @property
    def faulted(self) -> bool:
        return self.stats().faulted

    def idle(self) -> bool:
        """True iff every accepted event has been dispatched and handled."""

        return self.outstanding_count == 0

    def _rearm_pressure_if_recovered(self) -> None:
        snapshot = self.stats()
        data_capacity = snapshot.max_events - snapshot.control_reserve
        with self._pressure_lock:
            if snapshot.pending_data <= data_capacity // 2:
                self._data_pressure_warned = False

    def _dispatch(self, event: AgentEvent) -> None:
        try:
            for handler in list(self._handlers):
                try:
                    handler(event)
                except Exception:  # noqa: BLE001 - isolate every subscriber
                    log.exception(
                        "event handler raised on %s; dropping it", event.kind
                    )
        finally:
            self._queue.task_done()
            self._rearm_pressure_if_recovered()

    def drain_once(self) -> bool:
        # get_nowait (not an empty() pre-check) so concurrent consumers degrade
        # to a clean False instead of an uncaught queue.Empty TOCTOU crash.
        try:
            event = self._queue.get_nowait()
        except Empty:
            return False
        self._dispatch(event)
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
        """Stop dispatch; pending action events remain undispatched by default."""

        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if drain:
            self.drain()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                event = self._queue.get(timeout=0.1)
            except Empty:
                continue
            self._dispatch(event)
