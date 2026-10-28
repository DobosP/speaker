"""Deterministic local endpoint native-owner lifetime barriers; no models."""
from types import SimpleNamespace
import threading
import time

import pytest

from core.engines import _endpoint_lease as module
from core.engines._endpoint_lease import (EndpointModelLease, EndpointModelRetired,
                                        retire_endpoint_models, require_endpoint_models_retired)


@pytest.fixture(autouse=True)
def isolated_quarantine(monkeypatch):
    monkeypatch.setattr(module, "_UNCERTAIN_LEASES", [])
    monkeypatch.setattr(module, "_RETIRED_LEASES", {})


def join(thread):
    thread.join(2)
    assert not thread.is_alive()


def test_retire_during_entered_warm_rejects_later_entries_and_releases_once():
    entered, finish = threading.Event(), threading.Event()
    released, operations, errors = [], [], []
    native = object()
    def load():
        entered.set()
        assert finish.wait(2)
        return native
    lease = EndpointModelLease(load, lambda value: released.append(value))
    def warm():
        try:
            lease.run(lambda value: operations.append(value))
        except EndpointModelRetired:
            errors.append("retired")
    thread = threading.Thread(target=warm)
    thread.start()
    assert entered.wait(1)
    assert lease.retire() is False
    with pytest.raises(EndpointModelRetired):
        lease.run(lambda value: None)
    assert not lease.closed and released == []
    finish.set()
    join(thread)
    assert operations == [] and errors == ["retired"]
    assert released == [native] and lease.closed
    assert lease.retire() is True and released == [native]


def test_entered_decode_owner_waits_for_real_free_success_before_closed():
    decoding, decode_done, freeing, free_done = [threading.Event() for _ in range(4)]
    release_threads = []
    def release(native):
        release_threads.append(threading.get_ident())
        freeing.set()
        assert free_done.wait(2)
    lease = EndpointModelLease(object, release)
    lease.run(lambda value: None)
    def decode(native):
        decoding.set()
        assert decode_done.wait(2)
        return "copied text"
    thread = threading.Thread(target=lambda: lease.run(decode))
    thread.start()
    assert decoding.wait(1)
    assert lease.retire() is False
    decode_done.set()
    assert freeing.wait(1)
    assert not lease.closed and thread.is_alive()
    free_done.set()
    join(thread)
    assert lease.closed and release_threads == [thread.ident]


def test_idle_free_is_nonblocking_and_runs_outside_owner_tracker_lock():
    freeing, free_done = threading.Event(), threading.Event()
    owner = SimpleNamespace()
    def release(native):
        retire_endpoint_models(owner)  # Re-entry must not deadlock cleanup.
        freeing.set()
        assert free_done.wait(2)
    lease = EndpointModelLease(object, release)
    voter = SimpleNamespace(_endpoint_lease=lease)
    lease.run(lambda native: None)
    began = time.monotonic()
    retire_endpoint_models(owner, voter)
    assert time.monotonic() - began < .5
    assert freeing.wait(1)
    assert not lease.closed
    with pytest.raises(RuntimeError, match="pending"):
        require_endpoint_models_retired(owner)
    free_done.set()
    assert lease.wait_closed(1)
    require_endpoint_models_retired(owner)
    assert owner._retired_endpoint_models == []


def test_failed_free_is_quarantined_without_retry_or_replacement():
    freed = threading.Event()
    calls = []
    native = object()
    def release(value):
        calls.append(value)
        freed.set()
        raise RuntimeError("private native failure")
    lease = EndpointModelLease(lambda: native, release)
    lease.run(lambda value: None)
    assert lease.retire() is False
    assert freed.wait(1)
    lease._cleanup_thread.join(1)
    assert not lease.closed and lease._failed_native is native
    assert lease.retire() is False and calls == [native]
    with pytest.raises(RuntimeError, match="uncertain"):
        EndpointModelLease(object, lambda value: None)


def test_ambiguous_cleanup_launch_cannot_authorize_replacement(monkeypatch):
    release_started, finish = threading.Event(), threading.Event()
    real_thread = threading.Thread
    class Ambiguous(real_thread):
        def start(self):
            super().start()
            raise RuntimeError("ambiguous launch")
    def release(native):
        release_started.set()
        assert finish.wait(2)
    lease = EndpointModelLease(object, release)
    lease.run(lambda native: None)
    monkeypatch.setattr(module.threading, "Thread", Ambiguous)
    assert lease.retire() is False
    assert release_started.wait(1)
    finish.set()
    lease._cleanup_thread.join(1)
    assert not lease.closed
    with pytest.raises(RuntimeError, match="uncertain"):
        EndpointModelLease(object, lambda native: None)


def test_unknown_legacy_close_method_is_never_called():
    owner = SimpleNamespace()
    def forbidden():
        raise AssertionError("legacy close changed")
    retire_endpoint_models(owner, SimpleNamespace(close=forbidden))
    require_endpoint_models_retired(owner)


def test_no_native_entry_retry_when_provider_raises_typeerror():
    calls = []
    def load():
        calls.append("entered")
        raise TypeError("provider failure")
    lease = EndpointModelLease(load, lambda value: None)
    with pytest.raises(TypeError):
        lease.run(lambda native: None)
    assert calls == ["entered"]
    assert lease.retire() is True



def test_temporary_probe_retirement_survives_its_owner_until_native_free():
    started, finish = threading.Event(), threading.Event()
    def release(native):
        started.set()
        assert finish.wait(2)
    lease = EndpointModelLease(object, release)
    lease.run(lambda native: None)
    lease.retire()
    assert started.wait(1)
    with pytest.raises(RuntimeError, match="pending"):
        EndpointModelLease(object, lambda native: None)
    finish.set()
    assert lease.wait_closed(1)
    successor = EndpointModelLease(object, lambda native: None)
    assert successor.retire()



def test_idle_publication_drops_last_borrow_before_cleanup_can_attest_closed():
    paused, resume, destroyed = threading.Event(), threading.Event(), threading.Event()
    operation_done = threading.Event()
    worker_ident = []
    class Native:
        def __del__(self):
            destroyed.set()
    lease = EndpointModelLease(Native, lambda native: None)
    real = lease._state
    class PauseAfterIdle:
        def __enter__(self):
            real.acquire()
            return self
        def __exit__(self, *args):
            pause = (worker_ident and threading.get_ident() == worker_ident[0]
                     and operation_done.is_set() and not lease._active and not lease._retired)
            real.release()
            if pause:
                paused.set()
                assert resume.wait(2)
    lease._state = PauseAfterIdle()
    def run():
        worker_ident.append(threading.get_ident())
        lease.run(lambda native: operation_done.set())
    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert paused.wait(1)
        lease.retire()
        assert lease.wait_closed(1)
        assert destroyed.is_set()
    finally:
        resume.set()
        join(thread)
