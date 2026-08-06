"""Transcript-free evidence for deterministic async-final worker delivery.

This private evaluator helper does not model production startup or shutdown.
Capture first quiesces with one fresh :class:`RealtimeMediaStage`; a persistent
diagnostic thread then invokes the real Sherpa final worker and closes the
already-empty stage only after every synchronous terminal callback has
returned.  Acoustic identities and thread identifiers exist only inside one
open observer and are erased before its counter-only snapshot is exposed.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from threading import Condition, Lock, Thread, get_ident
import time
from typing import Mapping, Sequence

from always_on_agent.acoustic import AcousticLineage
from core.engine import FinalTranscript, TranscriptAbort, TranscriptAbortReason
from core.realtime_media_stage import (
    RealtimeMediaStage,
    StageRetirement,
    StageSnapshot,
)


_REASONS = tuple(reason.value for reason in TranscriptAbortReason)
_MAX_LINEAGE_SPANS = 64
_MAX_TERMINALS = 256
_TRUE_CAPABILITIES = {
    "accepted_handoff_to_worker_lineage",
    "dedicated_worker_execution",
    "same_lineage_worker_callback_delivery",
    "successful_callbacks_off_capture_thread",
    "clean_stage_drain_before_observer_close",
}
_FALSE_CAPABILITIES = {
    "standalone_raw_to_accepted_handoff_lineage",
    "capture_worker_concurrency",
    "async_inline_selected_text_parity",
    "queue_overflow_backpressure_coverage",
    "stop_shutdown_cancellation_parity",
    "final_dispatcher_delivery",
    "supervisor_delivery",
    "runtime_task_delivery",
    "production_default_change",
    "authority_change",
    "device_microphone_evidence",
    "latency_evidence",
    "live_evidence",
    "quality_improvement_evidence",
}


class AsyncFinalDeliveryError(RuntimeError):
    """Detail-free async-final evidence failure."""


def _count(value: object) -> int:
    if type(value) is not int or value < 0:
        raise AsyncFinalDeliveryError()
    return value


def _timeout(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise AsyncFinalDeliveryError()
    return float(value)


def _identity(
    acoustic: object,
    revision: object,
) -> tuple[tuple[tuple[str, int, int, str], ...], int]:
    if (
        not isinstance(acoustic, AcousticLineage)
        or len(acoustic.spans) > _MAX_LINEAGE_SPANS
        or type(revision) is not int
        or revision < 0
    ):
        raise AsyncFinalDeliveryError()
    return tuple(span.key for span in acoustic.spans), revision


def _empty_reason_counts() -> dict[str, int]:
    return {reason: 0 for reason in _REASONS}


def _reason_tuple(counts: Mapping[str, int]) -> tuple[int, ...]:
    if set(counts) != set(_REASONS):
        raise AsyncFinalDeliveryError()
    return tuple(_count(counts[reason]) for reason in _REASONS)


def _reason_report(values: Sequence[int]) -> dict[str, int]:
    if len(values) != len(_REASONS):
        raise AsyncFinalDeliveryError()
    return {reason: _count(value) for reason, value in zip(_REASONS, values)}


@dataclass(frozen=True, slots=True)
class AsyncFinalDeliverySnapshot:
    """One closed row's counter-only worker-delivery evidence."""

    worker_bound: bool
    worker_distinct_from_capture: bool
    worker_session_completed: bool
    clean_drain: bool
    stage_closed: bool
    observer_closed: bool
    worker_items_started: int
    worker_items_finished: int
    worker_selected_finals: int
    inline_selected_finals: int
    worker_abort_reason_counts: tuple[int, ...]
    inline_abort_reason_counts: tuple[int, ...]
    pre_drain_queued: int
    pre_drain_in_flight: int
    pre_drain_finished: int
    final_queued: int
    final_in_flight: int
    offered: int
    accepted: int
    rejected_closed: int
    overflow_released: int
    scope_released: int
    scope_in_flight_cancelled: int
    close_released: int
    close_in_flight_cancelled: int
    finished: int
    close_retired_queued: int
    close_cancelled_in_flight: int
    foreign_callbacks: int
    late_callbacks: int
    duplicate_terminals: int
    conflicting_terminals: int
    missing_worker_terminals: int
    finish_failures: int
    internal_errors: int
    exact: bool

    def __post_init__(self) -> None:
        boolean_fields = {
            "worker_bound",
            "worker_distinct_from_capture",
            "worker_session_completed",
            "clean_drain",
            "stage_closed",
            "observer_closed",
            "exact",
        }
        reason_fields = {
            "worker_abort_reason_counts",
            "inline_abort_reason_counts",
        }
        for definition in fields(self):
            value = getattr(self, definition.name)
            if definition.name in boolean_fields:
                if type(value) is not bool:
                    raise AsyncFinalDeliveryError()
            elif definition.name in reason_fields:
                if not isinstance(value, tuple) or len(value) != len(_REASONS):
                    raise AsyncFinalDeliveryError()
                for count in value:
                    _count(count)
            else:
                _count(value)


class AsyncFinalDeliveryObserver:
    """Correlate one row's worker item and typed terminal in memory only."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._capture_thread: int | None = get_ident()
        self._worker_thread: int | None = None
        self._closed = False
        self._resolved: dict[object, str] = {}
        self._worker_handoffs: set[object] = set()
        self._worker_sequences: set[int] = set()
        self._active_sequence: int | None = None
        self._active_identity: object | None = None
        self._active_terminal = False

        self._worker_items_started = 0
        self._worker_items_finished = 0
        self._worker_selected = 0
        self._inline_selected = 0
        self._worker_aborts = _empty_reason_counts()
        self._inline_aborts = _empty_reason_counts()
        self._foreign_callbacks = 0
        self._late_callbacks = 0
        self._duplicate_terminals = 0
        self._conflicting_terminals = 0
        self._missing_worker_terminals = 0
        self._finish_failures = 0
        self._internal_errors = 0

        self._worker_session_completed = False
        self._clean_drain = False
        self._pre: StageSnapshot | None = None
        self._closed_stage: StageSnapshot | None = None
        self._close_retirement: StageRetirement[object] | None = None

    def note_internal_error(self) -> None:
        with self._lock:
            self._internal_errors += 1

    def bind_worker_thread(self) -> None:
        current = get_ident()
        with self._lock:
            if (
                self._closed
                or self._worker_thread is not None
                or self._capture_thread is None
                or current == self._capture_thread
            ):
                self._internal_errors += 1
                return
            self._worker_thread = current

    def begin_worker_item(
        self,
        sequence: object,
        acoustic: object,
        revision: object,
    ) -> None:
        try:
            identity = _identity(acoustic, revision)
        except AsyncFinalDeliveryError:
            self.note_internal_error()
            return
        current = get_ident()
        with self._lock:
            if self._closed:
                self._late_callbacks += 1
                self._internal_errors += 1
                return
            if (
                type(sequence) is not int
                or sequence <= 0
                or self._worker_thread is None
                or current != self._worker_thread
                or self._active_sequence is not None
                or self._worker_items_started >= _MAX_TERMINALS
            ):
                self._internal_errors += 1
                return
            if sequence in self._worker_sequences:
                self._conflicting_terminals += 1
                self._internal_errors += 1
            self._worker_sequences.add(sequence)
            if identity in self._worker_handoffs or identity in self._resolved:
                self._conflicting_terminals += 1
                self._internal_errors += 1
            self._worker_handoffs.add(identity)
            self._active_sequence = sequence
            self._active_identity = identity
            self._active_terminal = False
            self._worker_items_started += 1

    def _observe_terminal(
        self,
        acoustic: object,
        revision: object,
        *,
        kind: str,
        reason: TranscriptAbortReason | None,
    ) -> None:
        try:
            identity = _identity(acoustic, revision)
        except AsyncFinalDeliveryError:
            self.note_internal_error()
            return
        current = get_ident()
        with self._lock:
            if self._closed:
                self._late_callbacks += 1
                self._internal_errors += 1
                return
            on_worker = (
                self._worker_thread is not None and current == self._worker_thread
            )
            on_capture = (
                self._capture_thread is not None and current == self._capture_thread
            )
            if not on_worker and not on_capture:
                self._foreign_callbacks += 1
                self._internal_errors += 1
                return
            previous = self._resolved.get(identity)
            if previous is None and len(self._resolved) >= _MAX_TERMINALS:
                self._internal_errors += 1
                return
            if previous is not None:
                if previous == kind:
                    self._duplicate_terminals += 1
                else:
                    self._conflicting_terminals += 1
                self._internal_errors += 1
                return
            if on_worker:
                if (
                    self._active_sequence is None
                    or self._active_identity != identity
                    or self._active_terminal
                ):
                    self._conflicting_terminals += 1
                    self._internal_errors += 1
                    return
                self._active_terminal = True
                if kind == "selected":
                    self._worker_selected += 1
                else:
                    assert reason is not None
                    self._worker_aborts[reason.value] += 1
            else:
                if self._active_sequence is not None:
                    self._foreign_callbacks += 1
                    self._internal_errors += 1
                    return
                if kind == "selected":
                    self._inline_selected += 1
                else:
                    assert reason is not None
                    self._inline_aborts[reason.value] += 1
            self._resolved[identity] = kind

    def observe_selected(self, result: object) -> None:
        if not isinstance(result, FinalTranscript):
            self.note_internal_error()
            return
        self._observe_terminal(
            result.acoustic,
            result.revision,
            kind="selected",
            reason=None,
        )

    def observe_abort(self, result: object) -> None:
        if not isinstance(result, TranscriptAbort) or not isinstance(
            result.reason,
            TranscriptAbortReason,
        ):
            self.note_internal_error()
            return
        self._observe_terminal(
            result.acoustic,
            result.revision,
            kind="abort",
            reason=result.reason,
        )

    def finish_worker_item(
        self,
        sequence: object,
        finish_succeeded: object,
    ) -> None:
        current = get_ident()
        with self._lock:
            if self._closed:
                self._late_callbacks += 1
                self._internal_errors += 1
                return
            if (
                self._worker_thread is None
                or current != self._worker_thread
                or type(sequence) is not int
                or sequence != self._active_sequence
            ):
                self._internal_errors += 1
                return
            if finish_succeeded is not True:
                self._finish_failures += 1
                self._internal_errors += 1
            if not self._active_terminal:
                self._missing_worker_terminals += 1
                self._internal_errors += 1
            self._worker_items_finished += 1
            self._active_sequence = None
            self._active_identity = None
            self._active_terminal = False

    def record_clean_drain(
        self,
        pre_drain: StageSnapshot,
        drained: StageSnapshot,
        closed: StageSnapshot,
        retirement: StageRetirement[object],
    ) -> None:
        with self._lock:
            if (
                self._closed
                or self._worker_session_completed
                or not isinstance(pre_drain, StageSnapshot)
                or not isinstance(drained, StageSnapshot)
                or not isinstance(closed, StageSnapshot)
                or not isinstance(retirement, StageRetirement)
            ):
                self._internal_errors += 1
                return
            self._pre = pre_drain
            self._closed_stage = closed
            self._close_retirement = retirement
            self._worker_session_completed = True
            self._clean_drain = bool(
                not pre_drain.closed
                and pre_drain.queued == pre_drain.accepted
                and pre_drain.in_flight == 0
                and pre_drain.finished == 0
                and pre_drain.offered == pre_drain.accepted
                and drained.queued == 0
                and drained.in_flight == 0
                and drained.finished == drained.accepted
                and not drained.closed
                and closed.closed
                and closed.queued == 0
                and closed.in_flight == 0
                and retirement.queued_released == 0
                and retirement.in_flight_cancelled == 0
            )
            if not self._clean_drain:
                self._internal_errors += 1

    def close(self) -> None:
        with self._lock:
            if (
                self._closed
                or self._capture_thread is None
                or get_ident() != self._capture_thread
                or self._active_sequence is not None
                or not self._worker_session_completed
            ):
                self._internal_errors += 1
            self._closed = True
            self._capture_thread = None
            self._worker_thread = None
            self._active_sequence = None
            self._active_identity = None
            self._active_terminal = False
            self._resolved.clear()
            self._worker_handoffs.clear()
            self._worker_sequences.clear()

    def snapshot(self) -> AsyncFinalDeliverySnapshot:
        with self._lock:
            if not self._closed or self._pre is None or self._closed_stage is None:
                raise AsyncFinalDeliveryError()
            retirement = self._close_retirement
            if retirement is None:
                raise AsyncFinalDeliveryError()
            pre = self._pre
            stage = self._closed_stage
            worker_abort_values = _reason_tuple(self._worker_aborts)
            inline_abort_values = _reason_tuple(self._inline_aborts)
            health_total = (
                self._foreign_callbacks
                + self._late_callbacks
                + self._duplicate_terminals
                + self._conflicting_terminals
                + self._missing_worker_terminals
                + self._finish_failures
                + self._internal_errors
            )
            worker_abort_total = sum(worker_abort_values)
            exact = bool(
                self._worker_session_completed
                and self._clean_drain
                and stage.closed
                and self._worker_items_started == stage.accepted
                and self._worker_items_finished == stage.finished
                and stage.finished == self._worker_selected + worker_abort_total
                and self._inline_selected == 0
                and stage.rejected_closed == 0
                and stage.overflow_released == 0
                and stage.scope_released == 0
                and stage.scope_in_flight_cancelled == 0
                and stage.close_released == 0
                and stage.close_in_flight_cancelled == 0
                and retirement.queued_released == 0
                and retirement.in_flight_cancelled == 0
                and health_total == 0
            )
            return AsyncFinalDeliverySnapshot(
                worker_bound=True,
                worker_distinct_from_capture=True,
                worker_session_completed=self._worker_session_completed,
                clean_drain=self._clean_drain,
                stage_closed=stage.closed,
                observer_closed=self._closed,
                worker_items_started=self._worker_items_started,
                worker_items_finished=self._worker_items_finished,
                worker_selected_finals=self._worker_selected,
                inline_selected_finals=self._inline_selected,
                worker_abort_reason_counts=worker_abort_values,
                inline_abort_reason_counts=inline_abort_values,
                pre_drain_queued=pre.queued,
                pre_drain_in_flight=pre.in_flight,
                pre_drain_finished=pre.finished,
                final_queued=stage.queued,
                final_in_flight=stage.in_flight,
                offered=stage.offered,
                accepted=stage.accepted,
                rejected_closed=stage.rejected_closed,
                overflow_released=stage.overflow_released,
                scope_released=stage.scope_released,
                scope_in_flight_cancelled=stage.scope_in_flight_cancelled,
                close_released=stage.close_released,
                close_in_flight_cancelled=stage.close_in_flight_cancelled,
                finished=stage.finished,
                close_retired_queued=retirement.queued_released,
                close_cancelled_in_flight=retirement.in_flight_cancelled,
                foreign_callbacks=self._foreign_callbacks,
                late_callbacks=self._late_callbacks,
                duplicate_terminals=self._duplicate_terminals,
                conflicting_terminals=self._conflicting_terminals,
                missing_worker_terminals=self._missing_worker_terminals,
                finish_failures=self._finish_failures,
                internal_errors=self._internal_errors,
                exact=exact,
            )


class AsyncFinalWorkerHarness:
    """One persistent evaluator thread that drains fresh stages sequentially."""

    def __init__(self, engine: object) -> None:
        if not callable(getattr(engine, "_final_worker", None)):
            raise AsyncFinalDeliveryError()
        self._engine = engine
        self._condition = Condition(Lock())
        self._pending: (
            tuple[RealtimeMediaStage[object], AsyncFinalDeliveryObserver] | None
        ) = None
        self._active = False
        self._started = False
        self._done = False
        self._worker_error = False
        self._closed = False
        self._thread = Thread(
            target=self._run,
            name="capture-replay-final-delivery",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._pending is not None or self._closed
                )
                if self._pending is None and self._closed:
                    return
                request = self._pending
                self._pending = None
                self._active = True
            assert request is not None
            stage, observer = request
            worker_error = False
            try:
                observer.bind_worker_thread()
                with self._condition:
                    self._started = True
                    self._condition.notify_all()
                self._engine._final_worker(
                    stage,
                    final_delivery_observer=observer,
                )
            except BaseException:  # noqa: BLE001 - guarded evaluator fails closed
                worker_error = True
                observer.note_internal_error()
            finally:
                with self._condition:
                    self._worker_error = worker_error
                    self._active = False
                    self._done = True
                    self._condition.notify_all()

    @staticmethod
    def _remaining(deadline: float) -> float:
        return max(0.0, deadline - time.monotonic())

    def drain(
        self,
        stage: object,
        observer: object,
        *,
        timeout: float,
    ) -> None:
        limit = _timeout(timeout)
        if not isinstance(stage, RealtimeMediaStage) or not isinstance(
            observer,
            AsyncFinalDeliveryObserver,
        ):
            raise AsyncFinalDeliveryError()
        pre = stage.snapshot()
        deadline = time.monotonic() + limit
        try:
            with self._condition:
                if self._closed or self._pending is not None or self._active:
                    raise AsyncFinalDeliveryError()
                self._started = False
                self._done = False
                self._worker_error = False
                self._engine._running.set()
                self._pending = (stage, observer)
                self._condition.notify_all()
                if (
                    not self._condition.wait_for(
                        lambda: self._started or self._done,
                        timeout=self._remaining(deadline),
                    )
                    or not self._started
                ):
                    raise AsyncFinalDeliveryError()
            if not stage.wait_idle(timeout=self._remaining(deadline)):
                raise AsyncFinalDeliveryError()
            drained = stage.snapshot()
            retirement = stage.close()
            closed = stage.snapshot()
            with self._condition:
                if not self._condition.wait_for(
                    lambda: self._done,
                    timeout=self._remaining(deadline),
                ):
                    raise AsyncFinalDeliveryError()
                worker_error = self._worker_error
            self._engine._running.clear()
            if worker_error:
                raise AsyncFinalDeliveryError()
            observer.record_clean_drain(pre, drained, closed, retirement)
        except Exception:
            observer.note_internal_error()
            try:
                stage.close()
            except Exception:
                pass
            self._engine._running.clear()
            with self._condition:
                self._condition.wait_for(
                    lambda: not self._active and self._pending is None,
                    timeout=self._remaining(deadline),
                )
            raise AsyncFinalDeliveryError() from None

    def close(self, *, timeout: float) -> None:
        limit = _timeout(timeout)
        deadline = time.monotonic() + limit
        with self._condition:
            if self._closed:
                raise AsyncFinalDeliveryError()
            if self._pending is not None or self._active:
                if not self._condition.wait_for(
                    lambda: self._pending is None and not self._active,
                    timeout=self._remaining(deadline),
                ):
                    raise AsyncFinalDeliveryError()
            self._closed = True
            self._condition.notify_all()
        self._thread.join(timeout=self._remaining(deadline))
        if self._thread.is_alive():
            raise AsyncFinalDeliveryError()


def _sum_field(
    snapshots: Sequence[AsyncFinalDeliverySnapshot],
    field: str,
) -> int:
    return sum(_count(getattr(snapshot, field)) for snapshot in snapshots)


def aggregate_async_final_delivery(
    snapshots: Sequence[AsyncFinalDeliverySnapshot],
) -> dict[str, object]:
    rows = tuple(snapshots)
    if not rows or any(
        not isinstance(snapshot, AsyncFinalDeliverySnapshot) for snapshot in rows
    ):
        raise AsyncFinalDeliveryError()
    worker_reasons = [0] * len(_REASONS)
    inline_reasons = [0] * len(_REASONS)
    for snapshot in rows:
        for index, value in enumerate(snapshot.worker_abort_reason_counts):
            worker_reasons[index] += _count(value)
        for index, value in enumerate(snapshot.inline_abort_reason_counts):
            inline_reasons[index] += _count(value)
    lifecycle = {
        "rows": len(rows),
        "worker_sessions_started": sum(int(row.worker_bound) for row in rows),
        "worker_sessions_completed": sum(
            int(row.worker_session_completed) for row in rows
        ),
        "worker_threads_distinct_from_capture": sum(
            int(row.worker_distinct_from_capture) for row in rows
        ),
        "clean_drains": sum(int(row.clean_drain) for row in rows),
        "stages_closed": sum(int(row.stage_closed) for row in rows),
        "observers_closed": sum(int(row.observer_closed) for row in rows),
    }
    stage = {
        field: _sum_field(rows, field)
        for field in (
            "pre_drain_queued",
            "pre_drain_in_flight",
            "pre_drain_finished",
            "final_queued",
            "final_in_flight",
            "offered",
            "accepted",
            "rejected_closed",
            "overflow_released",
            "scope_released",
            "scope_in_flight_cancelled",
            "close_released",
            "close_in_flight_cancelled",
            "finished",
            "close_retired_queued",
            "close_cancelled_in_flight",
        )
    }
    callbacks = {
        "worker_items_started": _sum_field(rows, "worker_items_started"),
        "worker_items_finished": _sum_field(rows, "worker_items_finished"),
        "worker_selected_finals": _sum_field(rows, "worker_selected_finals"),
        "inline_selected_finals": _sum_field(rows, "inline_selected_finals"),
        "worker_typed_aborts": sum(worker_reasons),
        "inline_typed_aborts": sum(inline_reasons),
        "worker_abort_reason_counts": _reason_report(worker_reasons),
        "inline_abort_reason_counts": _reason_report(inline_reasons),
    }
    health = {
        field: _sum_field(rows, field)
        for field in (
            "foreign_callbacks",
            "late_callbacks",
            "duplicate_terminals",
            "conflicting_terminals",
            "missing_worker_terminals",
            "finish_failures",
            "internal_errors",
        )
    }
    exact = bool(
        all(row.exact for row in rows)
        and all(value == len(rows) for value in lifecycle.values())
        and stage["pre_drain_queued"] == stage["accepted"]
        and stage["pre_drain_in_flight"] == 0
        and stage["pre_drain_finished"] == 0
        and stage["final_queued"] == 0
        and stage["final_in_flight"] == 0
        and stage["offered"] == stage["accepted"] == stage["finished"]
        and all(
            stage[name] == 0
            for name in (
                "rejected_closed",
                "overflow_released",
                "scope_released",
                "scope_in_flight_cancelled",
                "close_released",
                "close_in_flight_cancelled",
                "close_retired_queued",
                "close_cancelled_in_flight",
            )
        )
        and callbacks["worker_items_started"]
        == callbacks["worker_items_finished"]
        == stage["accepted"]
        and stage["finished"]
        == callbacks["worker_selected_finals"] + callbacks["worker_typed_aborts"]
        and callbacks["worker_selected_finals"] > 0
        and callbacks["inline_selected_finals"] == 0
        and all(value == 0 for value in health.values())
    )
    report = {
        "schema_version": 1,
        "lifecycle": lifecycle,
        "stage": stage,
        "callbacks": callbacks,
        "health": health,
        "capabilities": {
            **{name: True for name in sorted(_TRUE_CAPABILITIES)},
            **{name: False for name in sorted(_FALSE_CAPABILITIES)},
        },
        "exact": exact,
    }
    validate_async_final_delivery_report(report)
    return report


def _require_mapping(value: object, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise AsyncFinalDeliveryError()
    return value


def validate_async_final_delivery_report(report: object) -> None:
    top = _require_mapping(
        report,
        {
            "schema_version",
            "lifecycle",
            "stage",
            "callbacks",
            "health",
            "capabilities",
            "exact",
        },
    )
    if top["schema_version"] != 1 or top["exact"] is not True:
        raise AsyncFinalDeliveryError()
    lifecycle = _require_mapping(
        top["lifecycle"],
        {
            "rows",
            "worker_sessions_started",
            "worker_sessions_completed",
            "worker_threads_distinct_from_capture",
            "clean_drains",
            "stages_closed",
            "observers_closed",
        },
    )
    stage_keys = {
        "pre_drain_queued",
        "pre_drain_in_flight",
        "pre_drain_finished",
        "final_queued",
        "final_in_flight",
        "offered",
        "accepted",
        "rejected_closed",
        "overflow_released",
        "scope_released",
        "scope_in_flight_cancelled",
        "close_released",
        "close_in_flight_cancelled",
        "finished",
        "close_retired_queued",
        "close_cancelled_in_flight",
    }
    stage = _require_mapping(top["stage"], stage_keys)
    callbacks = _require_mapping(
        top["callbacks"],
        {
            "worker_items_started",
            "worker_items_finished",
            "worker_selected_finals",
            "inline_selected_finals",
            "worker_typed_aborts",
            "inline_typed_aborts",
            "worker_abort_reason_counts",
            "inline_abort_reason_counts",
        },
    )
    health = _require_mapping(
        top["health"],
        {
            "foreign_callbacks",
            "late_callbacks",
            "duplicate_terminals",
            "conflicting_terminals",
            "missing_worker_terminals",
            "finish_failures",
            "internal_errors",
        },
    )
    capabilities = _require_mapping(
        top["capabilities"],
        _TRUE_CAPABILITIES | _FALSE_CAPABILITIES,
    )
    rows = _count(lifecycle["rows"])
    if rows <= 0 or any(_count(value) != rows for value in lifecycle.values()):
        raise AsyncFinalDeliveryError()
    for value in stage.values():
        _count(value)
    for name in (
        "worker_items_started",
        "worker_items_finished",
        "worker_selected_finals",
        "inline_selected_finals",
        "worker_typed_aborts",
        "inline_typed_aborts",
    ):
        _count(callbacks[name])
    worker_reasons = _require_mapping(
        callbacks["worker_abort_reason_counts"], set(_REASONS)
    )
    inline_reasons = _require_mapping(
        callbacks["inline_abort_reason_counts"], set(_REASONS)
    )
    if (
        sum(_count(value) for value in worker_reasons.values())
        != callbacks["worker_typed_aborts"]
        or sum(_count(value) for value in inline_reasons.values())
        != callbacks["inline_typed_aborts"]
        or any(_count(value) != 0 for value in health.values())
        or stage["pre_drain_queued"] != stage["accepted"]
        or stage["pre_drain_in_flight"] != 0
        or stage["pre_drain_finished"] != 0
        or stage["final_queued"] != 0
        or stage["final_in_flight"] != 0
        or stage["offered"] != stage["accepted"]
        or stage["accepted"] != stage["finished"]
        or any(
            stage[name] != 0
            for name in stage_keys
            - {"pre_drain_queued", "offered", "accepted", "finished"}
        )
        or callbacks["worker_items_started"] != stage["accepted"]
        or callbacks["worker_items_finished"] != stage["finished"]
        or stage["finished"]
        != callbacks["worker_selected_finals"] + callbacks["worker_typed_aborts"]
        or callbacks["worker_selected_finals"] <= 0
        or callbacks["inline_selected_finals"] != 0
        or any(capabilities[name] is not True for name in _TRUE_CAPABILITIES)
        or any(capabilities[name] is not False for name in _FALSE_CAPABILITIES)
    ):
        raise AsyncFinalDeliveryError()


def validate_async_final_delivery_against_sections(
    *,
    async_delivery: object,
    raw_shadow: object,
    terminal_lineage: object,
    metrics: object,
) -> None:
    validate_async_final_delivery_report(async_delivery)
    try:
        report = async_delivery
        assert isinstance(report, Mapping)
        callbacks = report["callbacks"]
        stage = report["stage"]
        if not isinstance(callbacks, Mapping) or not isinstance(stage, Mapping):
            raise AsyncFinalDeliveryError()
        if (
            not isinstance(raw_shadow, Mapping)
            or not isinstance(terminal_lineage, Mapping)
            or not isinstance(metrics, Mapping)
        ):
            raise AsyncFinalDeliveryError()
        raw_parity = raw_shadow["parity"]
        outcomes = terminal_lineage["outcomes"]
        lineage = terminal_lineage["lineage"]
        if not all(
            isinstance(value, Mapping) for value in (raw_parity, outcomes, lineage)
        ):
            raise AsyncFinalDeliveryError()
        if "streaming" in metrics:
            streaming = metrics["streaming"]
            if not isinstance(streaming, Mapping):
                raise AsyncFinalDeliveryError()
            selected = _count(streaming["final_events"])
            metric_reasons = streaming["abort_reasons"]
            declared_abort_total = None
            declared_terminal_events = None
        else:
            turns = metrics["turn_integrity"]
            aborts = metrics["transcript_aborts"]
            if not isinstance(turns, Mapping) or not isinstance(aborts, Mapping):
                raise AsyncFinalDeliveryError()
            selected = _count(turns["finals"])
            metric_reasons = aborts["reason_counts"]
            declared_abort_total = _count(aborts["total"])
            declared_terminal_events = _count(aborts["terminal_events"])
        if not isinstance(metric_reasons, Mapping) or not set(metric_reasons).issubset(
            _REASONS
        ):
            raise AsyncFinalDeliveryError()
        expected_reasons = {
            reason: _count(metric_reasons.get(reason, 0)) for reason in _REASONS
        }
        expected_abort_total = sum(expected_reasons.values())
        worker_reasons = callbacks["worker_abort_reason_counts"]
        inline_reasons = callbacks["inline_abort_reason_counts"]
        terminal_reasons = outcomes["abort_reason_counts"]
        matched_reasons = lineage["matched_abort_reason_counts"]
        if not all(
            isinstance(value, Mapping)
            for value in (
                worker_reasons,
                inline_reasons,
                terminal_reasons,
                matched_reasons,
            )
        ):
            raise AsyncFinalDeliveryError()
        combined_reasons = {
            reason: _count(worker_reasons[reason]) + _count(inline_reasons[reason])
            for reason in _REASONS
        }
        raw_commits = _count(lineage["raw_commits"])
        inline_total = _count(callbacks["inline_typed_aborts"])
        worker_selected = _count(callbacks["worker_selected_finals"])
        if (
            _count(raw_parity["paired_terminals"]) != raw_commits
            or raw_commits != _count(stage["accepted"]) + inline_total
            or raw_commits
            != worker_selected + inline_total + _count(callbacks["worker_typed_aborts"])
            or worker_selected != selected
            or _count(outcomes["selected_finals"]) != selected
            or _count(lineage["matched_selected_finals"]) != selected
            or _count(outcomes["typed_aborts"]) != expected_abort_total
            or _count(lineage["matched_typed_aborts"]) != expected_abort_total
            or (
                declared_abort_total is not None
                and declared_abort_total != expected_abort_total
            )
            or (
                declared_terminal_events is not None
                and declared_terminal_events != selected + expected_abort_total
            )
            or dict(terminal_reasons) != expected_reasons
            or dict(matched_reasons) != expected_reasons
            or combined_reasons != expected_reasons
        ):
            raise AsyncFinalDeliveryError()
    except (AssertionError, KeyError, TypeError, ValueError, AsyncFinalDeliveryError):
        raise AsyncFinalDeliveryError() from None


__all__ = [
    "AsyncFinalDeliveryError",
    "AsyncFinalDeliveryObserver",
    "AsyncFinalDeliverySnapshot",
    "AsyncFinalWorkerHarness",
    "aggregate_async_final_delivery",
    "validate_async_final_delivery_against_sections",
    "validate_async_final_delivery_report",
]
