"""Natural terminal delivery must wake without weakening provider ownership."""

from __future__ import annotations

import asyncio
import queue
import threading

import pytest

import core.llm as llm_module
from tests.test_ollama_async_cancel import (
    Consumer,
    FakeAsyncChunks,
    FakeAsyncClient,
    FakeAsyncClientFactory,
    ProviderBoom,
    _async_llm,
)


@pytest.mark.parametrize("pieces", [(), ("public",), tuple(str(i) for i in range(64))])
def test_known_completed_stream_never_enters_another_blocking_queue_get(
    monkeypatch, pieces
):
    response = FakeAsyncChunks(*pieces)
    factory = FakeAsyncClientFactory(response)
    stream = _async_llm(factory).stream("Public prompt.")
    stream._ensure_started()
    assert stream._producer.done.wait(1.0)
    assert factory.clients[0].closed.is_set()
    assert stream._producer.items.qsize() <= 64
    real_get = stream._producer.items.get

    def require_nonblocking(block=True, timeout=None):
        assert not block, "known terminal waited for the token poll timeout"
        return real_get(block=block, timeout=timeout)

    monkeypatch.setattr(stream._producer.items, "get", require_nonblocking)
    try:
        assert list(stream) == list(pieces)
        assert response.aclose_calls == factory.clients[0].close_calls == 1
    finally:
        stream.close()


class _GatedCloseClient(FakeAsyncClient):
    def __init__(self, response):
        super().__init__(response)
        self.cleanup_entered = threading.Event()
        self.release_cleanup = threading.Event()

    async def close(self):
        self.cleanup_entered.set()
        while not self.release_cleanup.is_set():
            await asyncio.sleep(0.005)
        await super().close()


def test_natural_terminal_wakes_a_waiting_get_only_after_client_cleanup(monkeypatch):
    client = _GatedCloseClient(FakeAsyncChunks())
    stream = _async_llm(lambda **_kwargs: client).stream("Public prompt.")
    waiting_get = threading.Event()
    real_get = stream._producer.items.get

    def long_wait(block=True, timeout=None):
        if block:
            waiting_get.set()
            # A stretched fallback poll proves completion actually notifies the
            # queue; this is synchronization, not a host performance threshold.
            timeout = 5.0
        return real_get(block=block, timeout=timeout)

    monkeypatch.setattr(stream._producer.items, "get", long_wait)
    consumer = Consumer(stream)
    consumer.start()
    try:
        assert waiting_get.wait(1.0) and client.cleanup_entered.wait(1.0)
        assert not stream._producer.done.is_set()
        assert consumer.thread.is_alive()
        client.release_cleanup.set()
        consumer.join()
        assert consumer.pieces == [] and consumer.error is None
        assert client.closed.is_set() and client.close_calls == 1
    finally:
        client.release_cleanup.set()
        if consumer.thread.is_alive():
            stream._producer.items.put_nowait(llm_module._OLLAMA_STREAM_TERMINAL)
        consumer.thread.join(1.0)
        stream.close()


class _ReleasedChunks(FakeAsyncChunks):
    def __init__(self, *, error=None):
        super().__init__(
            *([] if error is not None else ["last public token"]), error=error
        )
        self.release = threading.Event()

    async def __anext__(self):
        while not self.release.is_set():
            await asyncio.sleep(0.001)
        return await super().__anext__()


@pytest.mark.parametrize("provider_error", [False, True])
def test_completion_racing_empty_get_rechecks_final_token_or_error(
    monkeypatch, provider_error
):
    error = ProviderBoom("public provider failure") if provider_error else None
    response = _ReleasedChunks(error=error)
    factory = FakeAsyncClientFactory(response)
    stream = _async_llm(factory).stream("Public prompt.")
    real_get = stream._producer.items.get
    raced = False

    def racing_get(block=True, timeout=None):
        nonlocal raced
        if block and not raced:
            raced = True
            response.release.set()
            assert stream._producer.done.wait(1.0)
            # Emulate the timed get's earlier empty observation becoming stale
            # just as the producer publishes its last item/error and terminal.
            raise queue.Empty
        return real_get(block=block, timeout=timeout)

    monkeypatch.setattr(stream._producer.items, "get", racing_get)
    try:
        if provider_error:
            with pytest.raises(ProviderBoom) as raised:
                next(stream)
            assert raised.value is error
        else:
            assert list(stream) == ["last public token"]
        assert raced and factory.clients[0].closed.is_set()
    finally:
        response.release.set()
        stream.close()


def test_full_data_queue_cannot_hold_cancellation_cleanup_for_terminal_space():
    client = _GatedCloseClient(FakeAsyncChunks(*(str(i) for i in range(64))))
    stream = _async_llm(lambda **_kwargs: client).stream("Public prompt.")
    stream._ensure_started()
    assert client.cleanup_entered.wait(1.0)
    assert stream._producer.items.qsize() == 64
    closer = threading.Thread(target=stream.close, daemon=True)
    closer.start()
    try:
        assert not stream._producer.done.is_set()
        client.release_cleanup.set()
        closer.join(1.0)
        assert not closer.is_alive()
        assert stream._producer.done.is_set() and client.closed.is_set()
        assert stream._producer.items.qsize() <= 64
        assert list(stream) == []  # pre-cancel buffered data remains revoked
    finally:
        client.release_cleanup.set()
        closer.join(1.0)
        stream.close()


def test_private_loop_failure_is_queued_before_terminal_notification(monkeypatch):
    original_run = asyncio.run
    error = ProviderBoom("public loop shutdown failure")

    def fail_after_cleanup(coro):
        original_run(coro)
        raise error

    monkeypatch.setattr(llm_module.asyncio, "run", fail_after_cleanup)
    factory = FakeAsyncClientFactory(FakeAsyncChunks())
    stream = _async_llm(factory).stream("Public prompt.")
    stream._ensure_started()
    stream._thread.join(1.0)
    assert not stream._thread.is_alive() and stream._producer.done.is_set()
    try:
        with pytest.raises(ProviderBoom) as raised:
            next(stream)
        assert raised.value is error
        assert factory.clients[0].closed.is_set()
    finally:
        stream.close()
