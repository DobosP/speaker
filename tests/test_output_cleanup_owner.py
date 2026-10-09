"""Native-free exact output close ownership, including hostile driver timing."""
import threading
import time

import pytest

from core.engines._output_cleanup import OutputCleanupOwner


class Stream:
    def __init__(self, *, stop_error=False, close_error=False, blocked=None):
        self.stop_calls = self.close_calls = 0
        self.stop_error = stop_error
        self.close_error = close_error
        self.blocked = blocked
        self.entered = threading.Event()
        self.release = threading.Event()

    def stop(self):
        self.stop_calls += 1
        if self.blocked == "stop":
            self.entered.set()
            assert self.release.wait(timeout=2)
        if self.stop_error:
            raise RuntimeError()

    def close(self):
        self.close_calls += 1
        if self.blocked == "close":
            self.entered.set()
            assert self.release.wait(timeout=2)
        if self.close_error:
            raise RuntimeError()


def test_concurrent_close_callers_share_one_exact_native_task():
    stream = Stream()
    owner = OutputCleanupOwner(stream)
    results = []
    callers = [threading.Thread(target=lambda: results.append(owner.close())) for _ in range(8)]
    for caller in callers:
        caller.start()
    for caller in callers:
        caller.join(timeout=1)
        assert not caller.is_alive()
    assert results == [True] * 8
    assert stream.stop_calls == stream.close_calls == 1
    assert owner.snapshot().closed
    assert owner.stream is stream


@pytest.mark.parametrize("operation", ["stop", "close"])
def test_failed_native_operation_is_poisoned_and_never_retried(operation):
    stream = Stream(stop_error=operation == "stop", close_error=operation == "close")
    owner = OutputCleanupOwner(stream)
    assert not owner.close()
    assert not owner.close()
    assert owner.snapshot().poisoned
    assert not owner.snapshot().closed
    assert stream.stop_calls == stream.close_calls == 1


@pytest.mark.parametrize("operation", ["stop", "close"])
def test_timed_out_native_owner_is_retained_and_late_return_cannot_upgrade(operation):
    stream = Stream(blocked=operation)
    owner = OutputCleanupOwner(stream)
    started = time.monotonic()
    try:
        assert not owner.close(timeout=0.02)
        assert time.monotonic() - started < 0.5
        assert stream.entered.is_set()
        assert owner.snapshot().active
        assert owner.snapshot().poisoned
        assert not owner.close(timeout=0.02)
        assert owner.stream is stream
    finally:
        stream.release.set()
        owner._thread.join(timeout=1)
    assert not owner.snapshot().closed
    assert not owner.close()
    assert stream.stop_calls == stream.close_calls == 1


@pytest.mark.parametrize("after_start", [False, True])
def test_ambiguous_cleanup_launch_keeps_exact_owner_and_never_launches_again(after_start):
    stream = Stream(blocked="stop" if after_start else None)
    created = []

    def factory(**kwargs):
        thread = threading.Thread(**kwargs)
        original_start = thread.start
        def ambiguous_start():
            if after_start:
                original_start()
            raise RuntimeError()
        thread.start = ambiguous_start
        created.append(thread)
        return thread

    owner = OutputCleanupOwner(stream, thread_factory=factory)
    try:
        assert not owner.close(timeout=0.1)
        assert not owner.close(timeout=0.1)
        assert len(created) == 1
        assert owner.snapshot().poisoned
    finally:
        stream.release.set()
        if after_start:
            created[0].join(timeout=1)
    assert not owner.snapshot().closed


def test_cleanup_never_joins_itself():
    results = []
    stream = Stream()
    owner = OutputCleanupOwner(stream)
    original_stop = stream.stop
    def reentrant_stop():
        original_stop()
        results.append(owner.close())
    stream.stop = reentrant_stop
    assert owner.close()
    assert results == [False]
    assert owner.snapshot().closed


@pytest.mark.parametrize("budget", [-1, float("nan"), float("inf"), True])
def test_invalid_budget_cannot_start_native_cleanup(budget):
    stream = Stream()
    owner = OutputCleanupOwner(stream)
    with pytest.raises(ValueError):
        owner.close(timeout=budget)
    assert stream.stop_calls == stream.close_calls == 0


def test_interrupted_wait_retains_poison_and_propagates_interrupt(monkeypatch):
    stream = Stream(blocked="stop")
    owner = OutputCleanupOwner(stream)
    def interrupted_wait(_timeout):
        raise KeyboardInterrupt()
    monkeypatch.setattr(owner._done, "wait", interrupted_wait)
    try:
        with pytest.raises(KeyboardInterrupt):
            owner.close()
        assert owner.snapshot().poisoned
        assert not owner.close()
    finally:
        stream.release.set()
        owner._thread.join(timeout=1)
    assert not owner.snapshot().closed
    assert stream.stop_calls == stream.close_calls == 1
