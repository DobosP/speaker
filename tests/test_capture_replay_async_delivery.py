"""Deterministic contracts for transcript-free async-final evidence."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from threading import Event
import time

import pytest

from always_on_agent.acoustic import AcousticLineage, AcousticSpan
from core.engine import FinalTranscript, TranscriptAbort, TranscriptAbortReason
from core.realtime_media_stage import CaptureScope, RealtimeMediaStage
from tools.capture_replay.async_delivery import (
    AsyncFinalDeliveryError,
    AsyncFinalDeliveryObserver,
    AsyncFinalDeliverySnapshot,
    AsyncFinalWorkerHarness,
    aggregate_async_final_delivery,
    validate_async_final_delivery_against_sections,
    validate_async_final_delivery_report,
)


def _lineage(index: int) -> AcousticLineage:
    return AcousticLineage.single(
        AcousticSpan(
            stream_id="PRIVATE-STREAM-MUST-BE-ERASED",
            utterance_id=f"PRIVATE-UTTERANCE-{index}",
            capture_epoch=7,
            capture_generation=3,
        )
    )


@dataclass(frozen=True)
class _Work:
    acoustic: AcousticLineage
    revision: int = 1
    abort: TranscriptAbortReason | None = None


class _FakeFinalEngine:
    def __init__(
        self,
        *,
        delay: float = 0.0,
        reuse_sequence: bool = False,
    ) -> None:
        self._running = Event()
        self._delay = delay
        self._reuse_sequence = reuse_sequence

    def _final_worker(self, stage, *, final_delivery_observer=None) -> None:
        assert final_delivery_observer is not None
        while self._running.is_set():
            item = stage.take(timeout=0.01)
            if item is None:
                if stage.snapshot().closed:
                    return
                continue
            work = item.payload
            final_delivery_observer.begin_worker_item(
                1 if self._reuse_sequence else item.sequence,
                work.acoustic,
                work.revision,
            )
            if self._delay:
                time.sleep(self._delay)
            if work.abort is None:
                final_delivery_observer.observe_selected(
                    FinalTranscript(
                        "PRIVATE TRANSCRIPT MUST NOT PERSIST",
                        acoustic=work.acoustic,
                        revision=work.revision,
                    )
                )
            else:
                final_delivery_observer.observe_abort(
                    TranscriptAbort(
                        work.acoustic,
                        work.revision,
                        work.abort,
                    )
                )
            succeeded = stage.finish(item)
            final_delivery_observer.finish_worker_item(
                1 if self._reuse_sequence else item.sequence,
                succeeded,
            )


def _stage(*works: _Work) -> RealtimeMediaStage[_Work]:
    stage = RealtimeMediaStage[_Work](8)
    scope = CaptureScope(capture_epoch=7, capture_generation=3)
    for work in works:
        assert stage.offer_nowait(scope, work).accepted
    return stage


def _closed_row(
    harness: AsyncFinalWorkerHarness,
    works: tuple[_Work, ...],
    *,
    inline_abort: TranscriptAbortReason | None = None,
    inline_index: int = 99,
):
    observer = AsyncFinalDeliveryObserver()
    if inline_abort is not None:
        observer.observe_abort(TranscriptAbort(_lineage(inline_index), 1, inline_abort))
    harness.drain(_stage(*works), observer, timeout=2.0)
    observer.close()
    return observer.snapshot()


def _reason_counts(**updates: int) -> dict[str, int]:
    result = {reason.value: 0 for reason in TranscriptAbortReason}
    result.update(updates)
    return result


def _sections(*, selected: int, input_rejected: int, internal_error: int):
    reasons = _reason_counts(
        input_rejected=input_rejected,
        internal_error=internal_error,
    )
    raw = selected + input_rejected + internal_error
    return (
        {"parity": {"paired_terminals": raw}},
        {
            "outcomes": {
                "selected_finals": selected,
                "typed_aborts": input_rejected + internal_error,
                "abort_reason_counts": reasons,
            },
            "lineage": {
                "raw_commits": raw,
                "matched_selected_finals": selected,
                "matched_typed_aborts": input_rejected + internal_error,
                "matched_abort_reason_counts": reasons,
            },
        },
        {
            "streaming": {
                "final_events": selected,
                "abort_reasons": {
                    key: value for key, value in reasons.items() if value
                },
            }
        },
    )


def test_persistent_harness_drains_two_rows_and_cross_validates_exact_lineage() -> None:
    harness = AsyncFinalWorkerHarness(_FakeFinalEngine())
    try:
        first = _closed_row(
            harness,
            (
                _Work(_lineage(1)),
                _Work(_lineage(2), abort=TranscriptAbortReason.INTERNAL_ERROR),
            ),
            inline_abort=TranscriptAbortReason.INPUT_REJECTED,
        )
        second = _closed_row(harness, (_Work(_lineage(3)),))
    finally:
        harness.close(timeout=2.0)

    report = aggregate_async_final_delivery((first, second))
    raw, terminal, metrics = _sections(
        selected=2,
        input_rejected=1,
        internal_error=1,
    )
    validate_async_final_delivery_against_sections(
        async_delivery=report,
        raw_shadow=raw,
        terminal_lineage=terminal,
        metrics=metrics,
    )

    assert report["exact"] is True
    assert report["lifecycle"]["rows"] == 2
    assert report["stage"]["accepted"] == 3
    assert report["callbacks"]["worker_selected_finals"] == 2
    assert report["callbacks"]["worker_typed_aborts"] == 1
    assert report["callbacks"]["inline_typed_aborts"] == 1
    assert report["capabilities"]["capture_worker_concurrency"] is False
    assert report["capabilities"]["async_inline_selected_text_parity"] is False

    serialized = repr(report)
    assert "PRIVATE TRANSCRIPT" not in serialized
    assert "PRIVATE-STREAM" not in serialized
    assert "PRIVATE-UTTERANCE" not in serialized


@pytest.mark.parametrize(
    "mutation",
    (
        "balanced_abort_reason_swap",
        "accepted_inline_balance",
        "forged_concurrency",
        "forged_text_parity",
        "forged_live",
        "vacuous_selected",
    ),
)
def test_balanced_and_capability_forgeries_fail_closed(mutation: str) -> None:
    harness = AsyncFinalWorkerHarness(_FakeFinalEngine())
    try:
        snapshot = _closed_row(harness, (_Work(_lineage(1)),))
    finally:
        harness.close(timeout=2.0)
    report = aggregate_async_final_delivery((snapshot,))
    raw, terminal, metrics = _sections(
        selected=1,
        input_rejected=0,
        internal_error=0,
    )
    forged = deepcopy(report)
    if mutation == "balanced_abort_reason_swap":
        forged["callbacks"]["worker_selected_finals"] = 0
        forged["callbacks"]["worker_typed_aborts"] = 1
        forged["callbacks"]["worker_abort_reason_counts"]["internal_error"] = 1
    elif mutation == "accepted_inline_balance":
        forged["stage"]["accepted"] = 0
        forged["stage"]["offered"] = 0
        forged["stage"]["finished"] = 0
        forged["stage"]["pre_drain_queued"] = 0
        forged["callbacks"]["worker_items_started"] = 0
        forged["callbacks"]["worker_items_finished"] = 0
        forged["callbacks"]["worker_selected_finals"] = 0
        forged["callbacks"]["inline_typed_aborts"] = 1
        forged["callbacks"]["inline_abort_reason_counts"]["input_rejected"] = 1
    elif mutation == "forged_concurrency":
        forged["capabilities"]["capture_worker_concurrency"] = True
    elif mutation == "forged_text_parity":
        forged["capabilities"]["async_inline_selected_text_parity"] = True
    elif mutation == "forged_live":
        forged["capabilities"]["live_evidence"] = True
    else:
        forged["callbacks"]["worker_selected_finals"] = 0
        forged["stage"]["accepted"] = 0
        forged["stage"]["offered"] = 0
        forged["stage"]["finished"] = 0
        forged["stage"]["pre_drain_queued"] = 0
        forged["callbacks"]["worker_items_started"] = 0
        forged["callbacks"]["worker_items_finished"] = 0

    with pytest.raises(AsyncFinalDeliveryError):
        validate_async_final_delivery_report(forged)
    with pytest.raises(AsyncFinalDeliveryError):
        validate_async_final_delivery_against_sections(
            async_delivery=forged,
            raw_shadow=raw,
            terminal_lineage=terminal,
            metrics=metrics,
        )


def test_duplicate_worker_identity_and_missing_terminal_cannot_aggregate() -> None:
    duplicate = _lineage(1)
    harness = AsyncFinalWorkerHarness(_FakeFinalEngine())
    observer = AsyncFinalDeliveryObserver()
    try:
        harness.drain(
            _stage(_Work(duplicate), _Work(duplicate)),
            observer,
            timeout=2.0,
        )
        observer.close()
        snapshot = observer.snapshot()
    finally:
        harness.close(timeout=2.0)

    assert snapshot.exact is False
    assert snapshot.conflicting_terminals > 0
    with pytest.raises(AsyncFinalDeliveryError):
        aggregate_async_final_delivery((snapshot,))


def test_reused_stage_sequence_is_rejected_even_for_distinct_lineage() -> None:
    harness = AsyncFinalWorkerHarness(_FakeFinalEngine(reuse_sequence=True))
    observer = AsyncFinalDeliveryObserver()
    try:
        harness.drain(
            _stage(_Work(_lineage(1)), _Work(_lineage(2))),
            observer,
            timeout=2.0,
        )
        observer.close()
        snapshot = observer.snapshot()
    finally:
        harness.close(timeout=2.0)

    assert snapshot.exact is False
    assert snapshot.conflicting_terminals > 0
    with pytest.raises(AsyncFinalDeliveryError):
        aggregate_async_final_delivery((snapshot,))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("worker_bound", 1),
        ("exact", 1),
        ("worker_items_started", True),
        ("worker_abort_reason_counts", ()),
        ("inline_abort_reason_counts", (0,)),
    ),
)
def test_snapshot_rejects_boolean_impostors_and_malformed_reason_tuples(
    field: str,
    value: object,
) -> None:
    harness = AsyncFinalWorkerHarness(_FakeFinalEngine())
    try:
        valid = _closed_row(harness, (_Work(_lineage(1)),))
    finally:
        harness.close(timeout=2.0)
    values = asdict(valid)
    values[field] = value
    with pytest.raises(AsyncFinalDeliveryError):
        AsyncFinalDeliverySnapshot(**values)


@pytest.mark.parametrize(
    "mutation",
    (
        "outcome_abort_total",
        "matched_abort_total",
        "edacc_abort_total",
        "edacc_terminal_events",
    ),
)
def test_cross_section_declared_terminal_totals_fail_closed(mutation: str) -> None:
    harness = AsyncFinalWorkerHarness(_FakeFinalEngine())
    try:
        snapshot = _closed_row(
            harness,
            (_Work(_lineage(1)),),
            inline_abort=TranscriptAbortReason.INPUT_REJECTED,
        )
    finally:
        harness.close(timeout=2.0)
    report = aggregate_async_final_delivery((snapshot,))
    raw, terminal, _metrics = _sections(
        selected=1,
        input_rejected=1,
        internal_error=0,
    )
    metrics = {
        "turn_integrity": {"finals": 1},
        "transcript_aborts": {
            "total": 1,
            "terminal_events": 2,
            "reason_counts": _reason_counts(input_rejected=1),
        },
    }
    if mutation == "outcome_abort_total":
        terminal["outcomes"]["typed_aborts"] = 0
    elif mutation == "matched_abort_total":
        terminal["lineage"]["matched_typed_aborts"] = 0
    elif mutation == "edacc_abort_total":
        metrics["transcript_aborts"]["total"] = 0
    else:
        metrics["transcript_aborts"]["terminal_events"] = 1
    with pytest.raises(AsyncFinalDeliveryError):
        validate_async_final_delivery_against_sections(
            async_delivery=report,
            raw_shadow=raw,
            terminal_lineage=terminal,
            metrics=metrics,
        )


@pytest.mark.parametrize("overflow", ("lineage_spans", "terminal_items"))
def test_observer_enforces_private_identity_memory_bounds(overflow: str) -> None:
    observer = AsyncFinalDeliveryObserver()
    if overflow == "lineage_spans":
        oversized = AcousticLineage(
            tuple(
                AcousticSpan(
                    stream_id="bounded-stream",
                    utterance_id=f"bounded-{index}",
                )
                for index in range(65)
            )
        )
        observer.observe_abort(
            TranscriptAbort(oversized, 1, TranscriptAbortReason.INPUT_REJECTED)
        )
    else:
        for index in range(257):
            observer.observe_abort(
                TranscriptAbort(
                    _lineage(index + 1000),
                    1,
                    TranscriptAbortReason.INPUT_REJECTED,
                )
            )
    harness = AsyncFinalWorkerHarness(_FakeFinalEngine())
    try:
        harness.drain(
            _stage(_Work(_lineage(999))),
            observer,
            timeout=2.0,
        )
        observer.close()
        snapshot = observer.snapshot()
    finally:
        harness.close(timeout=2.0)
    assert snapshot.exact is False
    assert snapshot.internal_errors > 0
    with pytest.raises(AsyncFinalDeliveryError):
        aggregate_async_final_delivery((snapshot,))


def test_drain_timeout_fails_without_a_publishable_snapshot() -> None:
    harness = AsyncFinalWorkerHarness(_FakeFinalEngine(delay=0.05))
    observer = AsyncFinalDeliveryObserver()
    try:
        with pytest.raises(AsyncFinalDeliveryError):
            harness.drain(
                _stage(_Work(_lineage(1))),
                observer,
                timeout=0.005,
            )
        time.sleep(0.08)
        with pytest.raises(AsyncFinalDeliveryError):
            observer.snapshot()
    finally:
        harness.close(timeout=2.0)
