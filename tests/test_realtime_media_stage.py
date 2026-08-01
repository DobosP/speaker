"""Deterministic contracts for the route-neutral bounded media stage."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict
import math
from threading import Barrier, Event, Thread

import pytest

from core.realtime_media_stage import (
    CaptureScope,
    RealtimeMediaStage,
    StageOfferResult,
    StageRetirement,
)


class _OpaquePayload:
    def __init__(self, name: str) -> None:
        self.name = name
        self.values: list[int] = []

    def __repr__(self) -> str:
        return f"PRIVATE-PAYLOAD-{self.name}"

    def __eq__(self, other: object) -> bool:
        raise AssertionError("the stage must not compare opaque payloads")


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("capture_epoch", True, TypeError),
        ("capture_epoch", 1.5, TypeError),
        ("capture_epoch", -1, ValueError),
        ("capture_generation", False, TypeError),
        ("capture_generation", "1", TypeError),
        ("capture_generation", -1, ValueError),
    ],
)
def test_capture_scope_is_immutable_and_strict(field, value, error) -> None:
    values = {"capture_epoch": 1, "capture_generation": 2}
    values[field] = value
    with pytest.raises(error):
        CaptureScope(**values)

    scope = CaptureScope(capture_epoch=1, capture_generation=2)
    with pytest.raises(FrozenInstanceError):
        scope.capture_epoch = 3  # type: ignore[misc]


def test_stage_rejects_invalid_capacity_scope_item_and_timeout() -> None:
    with pytest.raises(TypeError, match="max_queued"):
        RealtimeMediaStage(True)
    with pytest.raises(ValueError, match="positive"):
        RealtimeMediaStage(0)
    stage: RealtimeMediaStage[object] = RealtimeMediaStage(1)
    with pytest.raises(TypeError, match="CaptureScope"):
        stage.offer_nowait(object(), object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="CaptureScope"):
        stage.retire_scope(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="StageItem"):
        stage.finish(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="timeout"):
        stage.take(timeout=True)
    with pytest.raises(ValueError, match="timeout"):
        stage.take(timeout=-0.1)
    with pytest.raises(ValueError, match="timeout"):
        stage.take(timeout=math.inf)


def test_stage_item_is_immutable_but_payload_ownership_stays_with_caller() -> None:
    stage = RealtimeMediaStage[_OpaquePayload](2)
    scope = CaptureScope(capture_epoch=3, capture_generation=4)
    payload = _OpaquePayload("owned")

    offered = stage.offer_nowait(scope, payload)
    assert offered.accepted
    assert offered.retired == ()
    assert offered.item.payload is payload
    payload.values.append(7)

    taken = stage.take(timeout=0.0)
    assert taken is offered.item
    assert taken is not None and taken.payload.values == [7]
    with pytest.raises(FrozenInstanceError):
        taken.sequence = 99  # type: ignore[misc]

    assert stage.finish(taken)
    assert not stage.finish(taken)


def test_hard_queue_bound_drops_oldest_and_preserves_retained_fifo() -> None:
    scope = CaptureScope(capture_epoch=1, capture_generation=1)
    stage = RealtimeMediaStage[str](2)

    first = stage.offer_nowait(scope, "first")
    second = stage.offer_nowait(scope, "second")
    third = stage.offer_nowait(scope, "third")

    assert all(result.accepted for result in (first, second, third))
    assert first.item.cancelled
    assert not second.item.cancelled
    assert not third.item.cancelled
    assert third.retired == (first.item,)
    assert third.retired_sequences == (first.item.sequence,)
    snapshot = stage.snapshot()
    assert snapshot.max_queued == 2
    assert snapshot.queued == 2
    assert snapshot.overflow_released == 1

    retained_second = stage.take(timeout=0.0)
    retained_third = stage.take(timeout=0.0)
    assert retained_second is second.item
    assert retained_third is third.item
    assert stage.take(timeout=0.0) is None
    assert stage.finish(retained_second)
    assert stage.finish(retained_third)


def test_overflow_returns_exact_unmodified_payload_for_caller_cleanup() -> None:
    stage = RealtimeMediaStage[_OpaquePayload](1)
    scope = CaptureScope(capture_epoch=9, capture_generation=12)
    first_payload = _OpaquePayload("first")
    first_payload.values.append(7)
    second_payload = _OpaquePayload("second")

    first = stage.offer_nowait(scope, first_payload)
    second = stage.offer_nowait(scope, second_payload)

    assert second.retired == (first.item,)
    assert first.item.cancelled
    assert second.retired[0].payload is first_payload
    assert second.retired[0].payload.values == [7]
    assert stage.snapshot().queued == 1
    assert stage.snapshot().accepted == 2


def test_scope_retirement_fences_popped_item_without_generation_ordering() -> None:
    old_scope = CaptureScope(capture_epoch=8, capture_generation=3)
    other_scope = CaptureScope(capture_epoch=8, capture_generation=4)
    numerically_older_scope = CaptureScope(
        capture_epoch=2,
        capture_generation=1,
    )
    stage = RealtimeMediaStage[str](4)

    popped_offer = stage.offer_nowait(old_scope, "popped-old")
    popped = stage.take(timeout=0.0)
    queued_old = stage.offer_nowait(old_scope, "queued-old")
    queued_other = stage.offer_nowait(other_scope, "queued-other")
    assert popped is popped_offer.item

    retirement = stage.retire_scope(old_scope)

    assert retirement == StageRetirement(
        queued_released=1,
        in_flight_cancelled=1,
    )
    assert retirement.retired == (queued_old.item,)
    assert retirement.retired_sequences == (queued_old.item.sequence,)
    assert popped is not None and popped.cancelled
    assert queued_old.item.cancelled
    assert not queued_other.item.cancelled
    assert stage.retire_scope(old_scope) == StageRetirement(0, 0)

    remaining = stage.take(timeout=0.0)
    assert remaining is queued_other.item
    assert stage.finish(popped)
    assert stage.finish(remaining)

    # Scope numbers are identities only.  Retiring a higher-valued scope does
    # not install a high-water mark that rejects a later lower-valued one.
    lower = stage.offer_nowait(numerically_older_scope, "lower-is-valid")
    assert lower.accepted
    assert not lower.item.cancelled
    assert stage.take(timeout=0.0) is lower.item
    assert stage.finish(lower.item)


def test_close_releases_queued_cancels_in_flight_and_rejects_exact_late_item() -> None:
    scope = CaptureScope(capture_epoch=4, capture_generation=5)
    stage = RealtimeMediaStage[str](2)
    active_offer = stage.offer_nowait(scope, "active")
    active = stage.take(timeout=0.0)
    queued = stage.offer_nowait(scope, "queued")
    assert active is active_offer.item

    retirement = stage.close()

    assert retirement.queued_released == 1
    assert retirement.in_flight_cancelled == 1
    assert retirement.retired == (queued.item,)
    assert retirement.retired_sequences == (queued.item.sequence,)
    assert active is not None and active.cancelled
    assert queued.item.cancelled
    assert stage.take(timeout=0.0) is None

    late = stage.offer_nowait(scope, "late-caller-owned")
    assert not late.accepted
    assert late.item.cancelled
    assert late.retired == ()
    assert late.item.sequence > queued.item.sequence
    assert stage.snapshot().rejected_closed == 1

    assert stage.finish(active)
    assert stage.close() == StageRetirement(0, 0)


def test_offer_close_race_has_one_exact_retirement_owner() -> None:
    scope = CaptureScope(capture_epoch=10, capture_generation=20)
    for _attempt in range(25):
        stage = RealtimeMediaStage[str](1)
        barrier = Barrier(3)
        offers = []
        closes = []

        def offer() -> None:
            barrier.wait()
            offers.append(stage.offer_nowait(scope, "payload"))

        def close() -> None:
            barrier.wait()
            closes.append(stage.close())

        producer = Thread(target=offer)
        closer = Thread(target=close)
        producer.start()
        closer.start()
        barrier.wait()
        producer.join(timeout=1.0)
        closer.join(timeout=1.0)

        assert not producer.is_alive() and not closer.is_alive()
        offered = offers[0]
        retired = closes[0]
        assert offered.item.cancelled
        assert retired.retired == ((offered.item,) if offered.accepted else ())
        assert stage.snapshot().closed
        assert stage.snapshot().queued == 0


def test_take_close_race_keeps_queued_and_in_flight_ownership_distinct() -> None:
    scope = CaptureScope(capture_epoch=11, capture_generation=21)
    for _attempt in range(25):
        stage = RealtimeMediaStage[str](1)
        offered = stage.offer_nowait(scope, "payload")
        barrier = Barrier(3)
        taken = []
        closes = []

        def take() -> None:
            barrier.wait()
            taken.append(stage.take(timeout=0.0))

        def close() -> None:
            barrier.wait()
            closes.append(stage.close())

        consumer = Thread(target=take)
        closer = Thread(target=close)
        consumer.start()
        closer.start()
        barrier.wait()
        consumer.join(timeout=1.0)
        closer.join(timeout=1.0)

        assert not consumer.is_alive() and not closer.is_alive()
        retired = closes[0]
        assert offered.item.cancelled
        if taken[0] is offered.item:
            assert retired.retired == ()
            assert retired.in_flight_cancelled == 1
            assert stage.finish(offered.item)
        else:
            assert taken[0] is None
            assert retired.retired == (offered.item,)
            assert retired.in_flight_cancelled == 0


def test_close_wakes_blocked_consumer() -> None:
    stage: RealtimeMediaStage[object] = RealtimeMediaStage(1)
    entered = Event()
    returned: list[object] = []

    def consume() -> None:
        entered.set()
        returned.append(stage.take())

    consumer = Thread(target=consume)
    consumer.start()
    assert entered.wait(timeout=1.0)

    stage.close()
    consumer.join(timeout=1.0)

    assert not consumer.is_alive()
    assert returned == [None]


def test_payload_never_appears_in_item_result_retirement_or_snapshot_repr() -> None:
    marker = "PRIVATE-PAYLOAD-do-not-log"
    payload = _OpaquePayload("do-not-log")
    scope = CaptureScope(capture_epoch=1, capture_generation=2)
    stage = RealtimeMediaStage[_OpaquePayload](1)
    offered = stage.offer_nowait(scope, payload)
    snapshot = stage.snapshot()
    retirement = stage.close()

    assert marker not in repr(offered.item)
    assert marker not in repr(offered)
    assert marker not in repr(snapshot)
    assert marker not in repr(retirement)
    assert marker not in repr(asdict(snapshot))
    assert not any("payload" in key for key in asdict(snapshot))
    assert offered.item in retirement.retired


def test_public_result_types_are_immutable() -> None:
    scope = CaptureScope(capture_epoch=0, capture_generation=0)
    stage = RealtimeMediaStage[str](1)
    offered: StageOfferResult[str] = stage.offer_nowait(scope, "value")
    with pytest.raises(FrozenInstanceError):
        offered.accepted = False  # type: ignore[misc]

    retired: StageRetirement[str] = stage.close()
    with pytest.raises(FrozenInstanceError):
        retired.queued_released = 99  # type: ignore[misc]
