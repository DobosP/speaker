from __future__ import annotations

import threading

import numpy as np
import pytest

from core.media_session import (
    CaptureBlockContext,
    CaptureControlLane,
    CaptureGapReason,
    CaptureMailbox,
    CaptureMediaSession,
)


def _block(value: float, samples: int = 100) -> np.ndarray:
    return np.full(samples, value, dtype="float32")


def test_mailbox_preserves_fifo_and_owns_source_storage() -> None:
    mailbox = CaptureMailbox(
        buffer_ms=300.0,
        max_frames=8,
        source_generation=4,
        source_sample_rate_hz=1000,
        clock=lambda: 12.5,
    )
    source = _block(1.0)

    first = mailbox.offer(
        source,
        sample_rate_hz=1000,
        source_generation=4,
        capture_epoch=7,
    )
    mailbox.offer(
        _block(2.0),
        sample_rate_hz=1000,
        source_generation=4,
        capture_epoch=7,
    )
    source.fill(99.0)

    one = mailbox.take(timeout=0.0)
    assert one is not None
    assert mailbox.finish(one)
    two = mailbox.take(timeout=0.0)

    assert first.accepted
    assert two is not None
    assert (one.sequence, two.sequence) == (1, 2)
    assert one.capture_epoch == 7
    assert one.capture_generation == 4
    assert one.captured_at == 12.5
    assert one.gap_before is None
    assert np.all(one.pcm == 1.0)
    assert np.all(two.pcm == 2.0)
    assert mailbox.depth == 0
    assert mailbox.buffered_ms == 0.0


def test_backpressure_flushes_stale_pcm_and_coalesces_one_gap() -> None:
    mailbox = CaptureMailbox(
        buffer_ms=300.0,
        max_frames=8,
        source_generation=1,
        source_sample_rate_hz=1000,
    )

    for value in range(1, 8):
        result = mailbox.offer(
            _block(float(value)),
            sample_rate_hz=1000,
            source_generation=1,
        )
        assert result.depth <= 3
        assert result.buffered_ms <= 300.0 + 1e-6

    newest = mailbox.take(timeout=0.0)

    assert newest is not None
    assert newest.sequence == 7
    assert np.all(newest.pcm == 7.0)
    assert newest.gap_before is not None
    assert newest.gap_before.reason is CaptureGapReason.MEDIA_BACKPRESSURE
    assert newest.gap_before.prior_generation == 1
    assert newest.gap_before.generation == 3
    assert newest.gap_before.first_dropped_sequence == 1
    assert newest.gap_before.last_dropped_sequence == 6
    assert newest.gap_before.dropped_frames == 6
    assert newest.gap_before.dropped_samples == 600


def test_frame_cap_is_hard_even_when_time_budget_has_room() -> None:
    mailbox = CaptureMailbox(
        buffer_ms=1000.0,
        max_frames=2,
        source_generation=0,
        source_sample_rate_hz=1000,
    )
    for value in (1.0, 2.0, 3.0):
        mailbox.offer(
            _block(value, samples=10),
            sample_rate_hz=1000,
            source_generation=0,
        )

    newest = mailbox.take(timeout=0.0)

    assert mailbox.depth == 0
    assert newest is not None and newest.sequence == 3
    assert newest.gap_before is not None
    assert newest.gap_before.dropped_frames == 2


def test_oversize_block_is_rejected_and_gap_waits_for_admissible_pcm() -> None:
    mailbox = CaptureMailbox(
        buffer_ms=100.0,
        max_frames=8,
        source_generation=2,
        source_sample_rate_hz=1000,
    )

    rejected = mailbox.offer(
        _block(1.0, samples=101),
        sample_rate_hz=1000,
        source_generation=2,
    )
    accepted = mailbox.offer(
        _block(2.0, samples=50),
        sample_rate_hz=1000,
        source_generation=2,
    )
    current = mailbox.take(timeout=0.0)

    assert not rejected.accepted
    assert accepted.accepted
    assert current is not None and current.sequence == 2
    assert current.gap_before is not None
    assert current.gap_before.reason is CaptureGapReason.MEDIA_BACKPRESSURE
    assert current.gap_before.first_dropped_sequence == 1
    assert current.gap_before.last_dropped_sequence == 1
    assert current.gap_before.dropped_frames == 1
    assert current.gap_before.dropped_samples == 101


def test_device_overflow_and_source_recovery_rotate_distinct_generations() -> None:
    mailbox = CaptureMailbox(
        buffer_ms=300.0,
        max_frames=8,
        source_generation=4,
        source_sample_rate_hz=1000,
    )
    mailbox.offer(
        _block(1.0),
        sample_rate_hz=1000,
        source_generation=4,
        device_overflow=True,
    )
    overflow = mailbox.take(timeout=0.0)
    assert overflow is not None
    mailbox.offer(
        _block(2.0),
        sample_rate_hz=2000,
        source_generation=5,
    )
    assert not mailbox.finish(overflow)
    recovery = mailbox.take(timeout=0.0)

    assert overflow.gap_before is not None
    assert overflow.gap_before.reason is CaptureGapReason.DEVICE_OVERFLOW
    assert overflow.capture_generation == 5
    assert recovery is not None and recovery.gap_before is not None
    assert recovery.gap_before.reason is CaptureGapReason.CAPTURE_RECOVERY
    assert recovery.capture_generation == 6
    assert recovery.source_generation == 5
    assert recovery.sample_rate_hz == 2000


def test_close_wakes_blocked_take_and_rejects_late_offer() -> None:
    mailbox = CaptureMailbox()
    entered = threading.Event()
    result: list[object] = []

    def _take() -> None:
        entered.set()
        result.append(mailbox.take())

    thread = threading.Thread(target=_take)
    thread.start()
    assert entered.wait(1.0)
    mailbox.close()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert result == [None]
    late = mailbox.offer(
        _block(1.0),
        sample_rate_hz=1000,
        source_generation=0,
    )
    assert not late.accepted
    assert mailbox.depth == 0


def test_close_retires_buffered_pcm_instead_of_draining_it() -> None:
    mailbox = CaptureMailbox()
    mailbox.offer(
        _block(1.0),
        sample_rate_hz=1000,
        source_generation=0,
    )

    mailbox.close()

    assert mailbox.depth == 0
    assert mailbox.buffered_ms == 0.0
    assert mailbox.take(timeout=0.0) is None


class _FastThenBlockedSource:
    actual_samplerate = 1000
    generation = 1

    def __init__(self, count: int = 20) -> None:
        self.count = count
        self.reads = 0
        self.produced = threading.Event()
        self.release = threading.Event()

    def read(self, frames: int):
        self.reads += 1
        if self.reads > self.count:
            self.produced.set()
            self.release.wait(2.0)
        return _block(float(self.reads), frames), False


def test_reader_keeps_native_capture_moving_while_consumer_is_stalled() -> None:
    source = _FastThenBlockedSource()
    session = CaptureMediaSession(
        source,
        block_seconds=0.1,
        buffer_ms=300.0,
        max_frames=8,
    )
    session.start()
    try:
        assert source.produced.wait(1.0)
        assert source.reads >= 21
        assert session.mailbox.depth <= 3
        assert session.mailbox.buffered_ms <= 300.0 + 1e-6

        drained = []
        while True:
            block = session.take(timeout=0.0)
            if block is None:
                break
            drained.append(block)
            session.finish(block)
        assert drained
        assert drained[-1].sequence == 20
        gaps = [block.gap_before for block in drained if block.gap_before]
        assert len(gaps) == 1
        assert gaps[0].reason is CaptureGapReason.MEDIA_BACKPRESSURE
    finally:
        session.request_stop()
        source.release.set()
        session.join(timeout=1.0)
    assert not session.is_alive


def test_backpressure_cancels_unclaimed_inflight_and_counts_whole_loss() -> None:
    mailbox = CaptureMailbox(
        buffer_ms=1000.0,
        max_frames=2,
        source_generation=1,
        source_sample_rate_hz=1000,
    )
    mailbox.offer(_block(1.0), sample_rate_hz=1000, source_generation=1)
    in_flight = mailbox.take(timeout=0.0)
    assert in_flight is not None and in_flight.effect_lease is not None

    mailbox.offer(_block(2.0), sample_rate_hz=1000, source_generation=1)
    mailbox.offer(_block(3.0), sample_rate_hz=1000, source_generation=1)
    overloaded = mailbox.offer(
        _block(4.0),
        sample_rate_hz=1000,
        source_generation=1,
    )

    assert overloaded.in_flight_cancelled == 1
    assert overloaded.dropped_frames == 3
    assert overloaded.dropped_samples == 300
    assert in_flight.effect_lease.cancelled
    assert not mailbox.finish(in_flight)
    newest = mailbox.take(timeout=0.0)
    assert newest is not None and newest.sequence == 4
    assert newest.gap_before is not None
    assert newest.gap_before.in_flight_cancelled == 1
    assert newest.gap_before.first_dropped_sequence == 1
    assert newest.gap_before.last_dropped_sequence == 3


def test_cancelled_recovery_carrier_preserves_reopen_reason_and_lineage() -> None:
    mailbox = CaptureMailbox(
        buffer_ms=100.0,
        max_frames=1,
        source_generation=1,
        source_sample_rate_hz=1000,
    )
    mailbox.offer(
        _block(1.0),
        sample_rate_hz=1000,
        source_generation=2,
    )
    carrier = mailbox.take(timeout=0.0)
    assert carrier is not None and carrier.gap_before is not None
    assert carrier.gap_before.reason is CaptureGapReason.CAPTURE_RECOVERY
    assert carrier.gap_before.prior_generation == 1
    assert carrier.capture_generation == 2

    mailbox.offer(
        _block(2.0),
        sample_rate_hz=1000,
        source_generation=2,
    )
    cancelled = mailbox.offer(
        _block(3.0),
        sample_rate_hz=1000,
        source_generation=2,
    )

    assert cancelled.in_flight_cancelled == 1
    assert not mailbox.finish(carrier)
    replacement = mailbox.take(timeout=0.0)
    assert replacement is not None and replacement.gap_before is not None
    assert replacement.capture_generation == 3
    assert replacement.gap_before.reason is CaptureGapReason.CAPTURE_RECOVERY
    assert replacement.gap_before.prior_generation == 1
    assert replacement.gap_before.generation == 3
    assert replacement.gap_before.in_flight_cancelled == 1
    assert replacement.gap_before.dropped_frames == 2
    assert replacement.gap_before.dropped_samples == 200
    assert replacement.gap_before.first_dropped_sequence == 1
    assert replacement.gap_before.last_dropped_sequence == 2


def test_claimed_inflight_effect_wins_before_reader_rotation() -> None:
    mailbox = CaptureMailbox(
        buffer_ms=300.0,
        max_frames=2,
        source_generation=1,
        source_sample_rate_hz=1000,
    )
    mailbox.offer(_block(1.0), sample_rate_hz=1000, source_generation=1)
    captured = mailbox.take(timeout=0.0)
    assert captured is not None and captured.effect_lease is not None
    assert captured.effect_lease.claim()

    rotated = mailbox.offer(
        _block(2.0),
        sample_rate_hz=1000,
        source_generation=1,
        device_overflow=True,
    )

    assert rotated.in_flight_cancelled == 0
    assert captured.effect_lease.cancelled
    assert captured.effect_lease.claimed_by_current_thread
    assert captured.effect_lease.claim()
    captured.effect_lease.release()
    captured.effect_lease.release()
    assert not mailbox.finish(captured)


def test_effect_lease_cancel_wins_concurrently_once_and_finish_advances() -> None:
    mailbox = CaptureMailbox(
        buffer_ms=300.0,
        max_frames=2,
        source_generation=1,
        source_sample_rate_hz=1000,
    )
    mailbox.offer(_block(1.0), sample_rate_hz=1000, source_generation=1)
    captured = mailbox.take(timeout=0.0)
    assert captured is not None and captured.effect_lease is not None

    rotations_done = threading.Event()
    outcomes: dict[str, object] = {}
    errors: list[BaseException] = []

    def reader() -> None:
        try:
            outcomes["first_rotation"] = mailbox.offer(
                _block(2.0),
                sample_rate_hz=1000,
                source_generation=1,
                device_overflow=True,
            )
            outcomes["second_rotation"] = mailbox.offer(
                _block(3.0),
                sample_rate_hz=1000,
                source_generation=1,
                device_overflow=True,
            )
        except BaseException as exc:
            errors.append(exc)
        finally:
            rotations_done.set()

    def consumer() -> None:
        try:
            outcomes["rotations_observed"] = rotations_done.wait(1.0)
            outcomes["claimed"] = captured.effect_lease.claim()
            outcomes["finish_result"] = mailbox.finish(captured)
            newest = mailbox.take(timeout=1.0)
            outcomes["newest"] = newest
            if newest is not None:
                outcomes["newest_finish_result"] = mailbox.finish(newest)
        except BaseException as exc:
            errors.append(exc)

    consumer_thread = threading.Thread(target=consumer, name="lease-consumer")
    reader_thread = threading.Thread(target=reader, name="lease-reader")
    consumer_thread.start()
    reader_thread.start()
    reader_thread.join(timeout=1.0)
    consumer_thread.join(timeout=1.0)

    assert not reader_thread.is_alive()
    assert not consumer_thread.is_alive()
    assert errors == []
    assert outcomes["rotations_observed"] is True
    first = outcomes["first_rotation"]
    second = outcomes["second_rotation"]
    assert outcomes["claimed"] is False
    assert int(bool(outcomes["claimed"])) + first.in_flight_cancelled == 1
    assert first.in_flight_cancelled == 1
    assert first.dropped_frames == 1
    assert first.dropped_samples == 100
    assert second.in_flight_cancelled == 0
    assert second.dropped_frames == 1
    assert second.dropped_samples == 100
    assert outcomes["finish_result"] is False
    newest = outcomes["newest"]
    assert newest is not None and newest.sequence == 3
    assert newest.gap_before is not None
    assert newest.gap_before.in_flight_cancelled == 1
    assert newest.gap_before.dropped_frames == 2
    assert newest.gap_before.dropped_samples == 200
    assert newest.gap_before.first_dropped_sequence == 1
    assert newest.gap_before.last_dropped_sequence == 2
    assert outcomes["newest_finish_result"] is True


def test_effect_lease_claim_wins_concurrently_once_and_finish_advances() -> None:
    mailbox = CaptureMailbox(
        buffer_ms=300.0,
        max_frames=2,
        source_generation=1,
        source_sample_rate_hz=1000,
    )
    mailbox.offer(_block(1.0), sample_rate_hz=1000, source_generation=1)
    captured = mailbox.take(timeout=0.0)
    assert captured is not None and captured.effect_lease is not None

    claim_done = threading.Event()
    rotations_done = threading.Event()
    outcomes: dict[str, object] = {}
    errors: list[BaseException] = []

    def consumer() -> None:
        claimed = False
        try:
            claimed = captured.effect_lease.claim()
            outcomes["claimed"] = claimed
            claim_done.set()
            outcomes["rotations_observed"] = rotations_done.wait(1.0)
            outcomes["cancel_requested"] = captured.effect_lease.cancelled
            outcomes["claim_still_owned"] = (
                captured.effect_lease.claimed_by_current_thread
            )
            if claimed:
                captured.effect_lease.release()
            outcomes["finish_result"] = mailbox.finish(captured)
            newest = mailbox.take(timeout=1.0)
            outcomes["newest"] = newest
            if newest is not None:
                outcomes["newest_finish_result"] = mailbox.finish(newest)
        except BaseException as exc:
            errors.append(exc)
        finally:
            claim_done.set()

    def reader() -> None:
        try:
            outcomes["claim_observed"] = claim_done.wait(1.0)
            outcomes["first_rotation"] = mailbox.offer(
                _block(2.0),
                sample_rate_hz=1000,
                source_generation=1,
                device_overflow=True,
            )
            outcomes["second_rotation"] = mailbox.offer(
                _block(3.0),
                sample_rate_hz=1000,
                source_generation=1,
                device_overflow=True,
            )
        except BaseException as exc:
            errors.append(exc)
        finally:
            rotations_done.set()

    consumer_thread = threading.Thread(target=consumer, name="lease-consumer")
    reader_thread = threading.Thread(target=reader, name="lease-reader")
    reader_thread.start()
    consumer_thread.start()
    consumer_thread.join(timeout=1.0)
    reader_thread.join(timeout=1.0)

    assert not consumer_thread.is_alive()
    assert not reader_thread.is_alive()
    assert errors == []
    assert outcomes["claim_observed"] is True
    assert outcomes["rotations_observed"] is True
    first = outcomes["first_rotation"]
    second = outcomes["second_rotation"]
    assert outcomes["claimed"] is True
    assert int(bool(outcomes["claimed"])) + first.in_flight_cancelled == 1
    assert outcomes["cancel_requested"] is True
    assert outcomes["claim_still_owned"] is True
    assert first.in_flight_cancelled == 0
    assert first.dropped_frames == 0
    assert first.dropped_samples == 0
    assert second.in_flight_cancelled == 0
    assert second.dropped_frames == 1
    assert second.dropped_samples == 100
    assert outcomes["finish_result"] is False
    newest = outcomes["newest"]
    assert newest is not None and newest.sequence == 3
    assert newest.gap_before is not None
    assert newest.gap_before.in_flight_cancelled == 0
    assert newest.gap_before.dropped_frames == 1
    assert newest.gap_before.dropped_samples == 100
    assert newest.gap_before.first_dropped_sequence == 2
    assert newest.gap_before.last_dropped_sequence == 2
    assert outcomes["newest_finish_result"] is True


def test_mailbox_pcm_is_owned_readonly_and_redacted_from_repr() -> None:
    mailbox = CaptureMailbox(
        buffer_ms=300.0,
        source_generation=1,
        source_sample_rate_hz=1000,
    )
    source = np.arange(100, dtype="float32").reshape(10, 10)
    mailbox.offer(source, sample_rate_hz=1000, source_generation=1)
    captured = mailbox.take(timeout=0.0)

    assert captured is not None
    assert captured.pcm.shape == (100,)
    assert captured.pcm.flags.c_contiguous
    assert not captured.pcm.flags.writeable
    with pytest.raises(ValueError):
        captured.pcm.setflags(write=True)
    assert "array" not in repr(captured)
    assert "pcm=" not in repr(captured)


def test_captured_context_and_reference_audio_are_redacted_from_repr() -> None:
    class _PrivateDomain:
        def __repr__(self) -> str:
            raise AssertionError("private capture domain repr was evaluated")

    mailbox = CaptureMailbox(
        buffer_ms=300.0,
        source_generation=1,
        source_sample_rate_hz=1000,
    )
    mailbox.offer(
        _block(1.0),
        sample_rate_hz=1000,
        source_generation=1,
        context=CaptureBlockContext(
            source_domain=_PrivateDomain(),
            far_reference_zero=_block(7.0),
            far_reference_delayed=_block(8.0),
            coherence_reference=_block(9.0),
        ),
    )
    captured = mailbox.take(timeout=0.0)

    assert captured is not None
    rendered = repr(captured)
    assert "context=" not in rendered
    assert "array" not in rendered


def test_second_take_fails_fast_until_exact_inflight_block_finishes() -> None:
    mailbox = CaptureMailbox(
        buffer_ms=300.0,
        source_generation=1,
        source_sample_rate_hz=1000,
    )
    mailbox.offer(_block(1.0), sample_rate_hz=1000, source_generation=1)
    mailbox.offer(_block(2.0), sample_rate_hz=1000, source_generation=1)
    first = mailbox.take(timeout=0.0)

    assert first is not None
    with pytest.raises(RuntimeError, match="one exact consumer"):
        mailbox.take(timeout=0.0)
    assert mailbox.finish(first)
    second = mailbox.take(timeout=0.0)
    assert second is not None
    assert mailbox.finish(second)
    with pytest.raises(RuntimeError, match="exact in-flight owner"):
        mailbox.finish(second)


class _FailingSource:
    actual_samplerate = 16000
    generation = 1

    def read(self, _frames: int):
        raise RuntimeError("scripted native failure")


def test_reader_failure_is_retained_and_closes_consumer_side() -> None:
    session = CaptureMediaSession(_FailingSource(), block_seconds=0.1)
    session.start()
    session.join(timeout=1.0)

    assert not session.is_alive
    assert isinstance(session.error, RuntimeError)
    assert session.take(timeout=0.0) is None


def test_unstarted_media_session_has_no_possible_reader_owner() -> None:
    session = CaptureMediaSession(_FailingSource(), block_seconds=0.1)

    assert not session.start_observed
    assert not session.may_start_or_is_alive
    assert session.join(timeout=0.0)


def test_interrupted_reader_launch_is_retained_until_delayed_owner_exits() -> None:
    allow_launch = threading.Event()
    launchers: list[threading.Thread] = []
    original_start = threading.Thread.start

    class _AmbiguousReaderThread(threading.Thread):
        def start(self) -> None:
            def launch_later() -> None:
                assert allow_launch.wait(1.0)
                original_start(self)

            launcher = threading.Thread(
                target=launch_later,
                name="delayed-reader-launcher",
            )
            launchers.append(launcher)
            launcher.start()
            raise RuntimeError("synthetic reader launch interruption")

    session = CaptureMediaSession(
        _FailingSource(),
        block_seconds=0.1,
        thread_factory=_AmbiguousReaderThread,
    )

    with pytest.raises(RuntimeError, match="synthetic reader launch"):
        session.start()
    assert not session.start_observed
    assert session.may_start_or_is_alive
    assert not session.join(timeout=0.0)

    session.request_stop()
    allow_launch.set()
    for launcher in launchers:
        launcher.join(1.0)
        assert not launcher.is_alive()
    assert session.join(timeout=1.0)
    assert session.start_observed
    assert not session.may_start_or_is_alive
    assert session.error is None


def test_media_session_rejects_budget_smaller_than_one_native_block() -> None:
    with pytest.raises(ValueError, match="at least one"):
        CaptureMediaSession(
            _FailingSource(),
            block_seconds=0.1,
            buffer_ms=99.0,
        )


class _BlockedSource:
    actual_samplerate = 16000
    generation = 1

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def read(self, frames: int):
        self.entered.set()
        self.release.wait(2.0)
        return np.ones(frames, dtype="float32"), False


def test_reader_does_not_publish_a_native_block_returned_after_stop() -> None:
    source = _BlockedSource()
    session = CaptureMediaSession(source, block_seconds=0.1)
    session.start()
    assert source.entered.wait(1.0)

    session.request_stop()
    source.release.set()
    session.join(timeout=1.0)

    assert not session.is_alive
    assert session.error is None
    assert session.mailbox.depth == 0
    assert session.take(timeout=0.0) is None


class _StampedSource:
    actual_samplerate = 1000
    generation = 4
    actual_device = "mic-a"

    def __init__(self) -> None:
        self.reads = 0
        self.release = threading.Event()

    def read(self, frames: int):
        self.reads += 1
        if self.reads > 1:
            self.release.wait(2.0)
        return _block(float(self.reads), frames), False


def test_reader_binds_source_clock_device_and_immutable_context() -> None:
    source = _StampedSource()
    stamps = []

    def context_provider(stamp):
        stamps.append(stamp)
        return CaptureBlockContext(
            speaking=True,
            speak_generation=8,
            playback_generation=3,
        )

    session = CaptureMediaSession(
        source,
        block_seconds=0.1,
        clock=lambda: 10.0,
        context_provider=context_provider,
    )
    session.start()
    try:
        block = session.take(timeout=1.0)
        assert block is not None
        assert block.captured_at == 10.0
        assert block.captured_started_at == 9.9
        assert block.source_device == "mic-a"
        assert (block.source_sample_start, block.source_sample_end) == (0, 100)
        assert block.context is not None
        assert block.context.speaking
        assert block.context.playback_generation == 3
        assert len(stamps) == 1  # provider ran exactly once for the published block
        assert stamps[0].source_device == "mic-a"
        assert stamps[0].source_sample_end == 100
    finally:
        session.request_stop()
        source.release.set()
        session.join(timeout=1.0)


def test_control_lane_is_bounded_and_retains_recovery_priority() -> None:
    lane = CaptureControlLane(max_events=2, clock=lambda: 1.0)
    lane.offer(
        "open",
        "initial",
        capture_epoch=2,
        source_generation=1,
    )
    lane.offer(
        "open",
        "duplicate",
        capture_epoch=2,
        source_generation=1,
    )
    lane.offer(
        "recovering",
        "lost",
        capture_epoch=2,
        source_generation=1,
    )
    lane.offer(
        "fatal",
        "gone",
        capture_epoch=2,
        source_generation=1,
    )

    events = lane.drain()

    assert lane.depth == 0
    assert [event.state for event in events] == ["recovering", "fatal"]
    assert [event.sequence for event in events] == [3, 4]


def test_control_lane_rejects_lower_priority_offer_when_saturated() -> None:
    lane = CaptureControlLane(max_events=2, clock=lambda: 1.0)
    assert lane.offer(
        "fatal",
        "device gone",
        capture_epoch=2,
        source_generation=1,
    )
    assert lane.offer(
        "recovering",
        "retrying",
        capture_epoch=2,
        source_generation=1,
    )

    assert not lane.offer(
        "open",
        "late open",
        capture_epoch=2,
        source_generation=2,
    )
    events = lane.drain()

    assert [event.state for event in events] == ["fatal", "recovering"]
    assert [event.sequence for event in events] == [1, 2]


def test_control_lane_never_replaces_fatal_with_recovering() -> None:
    lane = CaptureControlLane(max_events=1, clock=lambda: 1.0)
    assert lane.offer(
        "fatal",
        "device gone",
        capture_epoch=2,
        source_generation=1,
    )

    assert not lane.offer(
        "recovering",
        "retrying",
        capture_epoch=2,
        source_generation=2,
    )
    assert [event.state for event in lane.drain()] == ["fatal"]
