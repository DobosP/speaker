"""Deterministic lifecycle tests for Ollama's async-to-sync stream bridge.

The production voice path consumes a synchronous token iterator, while Ollama's
``AsyncClient`` gives us the cancellation primitive needed to interrupt a read
*before* its first token.  These fakes model that provider boundary without a
daemon, model, socket, or timing-dependent sleeps.
"""
from __future__ import annotations

import asyncio
import gc
import threading
import weakref
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from always_on_agent.events import Mode
from always_on_agent.tasks import DEFAULT_MAX_ACTIVE_TASKS

from core.engines.scripted import ScriptedEngine
from core.llm import HedgeLLM, OllamaLLM, capability_context
from core.runtime import VoiceRuntime


class ProviderBoom(RuntimeError):
    """Distinct provider failure so the bridge cannot silently wrap it."""


class FakeAsyncChunks(AsyncIterator[dict[str, dict[str, str]]]):
    """Scripted async response that can remain blocked after its last chunk."""

    def __init__(
        self,
        *pieces: str,
        block_after_chunks: bool = False,
        error: BaseException | None = None,
    ) -> None:
        self._pieces = list(pieces)
        self._block_after_chunks = block_after_chunks
        self._error = error
        self._index = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._release_gate: asyncio.Event | None = None
        self._cancelled = False

        self.waiting = threading.Event()
        self.task_cancelled = threading.Event()
        self.unwound = threading.Event()
        self.closed = threading.Event()
        self.aclose_calls = 0

    def __aiter__(self) -> "FakeAsyncChunks":
        return self

    async def __anext__(self) -> dict[str, dict[str, str]]:
        try:
            if self._index < len(self._pieces):
                piece = self._pieces[self._index]
                self._index += 1
                return {"message": {"content": piece}}

            if self._error is not None:
                error, self._error = self._error, None
                self.closed.set()
                raise error

            if not self._block_after_chunks:
                self.closed.set()
                raise StopAsyncIteration

            self._loop = asyncio.get_running_loop()
            self._release_gate = asyncio.Event()
            self.waiting.set()
            await self._release_gate.wait()
            if self._cancelled:
                self.unwound.set()
            self.closed.set()
            raise StopAsyncIteration
        except asyncio.CancelledError:
            # This is the essential assertion seam: cancelling the public sync
            # iterator must reach the provider task that is awaiting __anext__.
            self.task_cancelled.set()
            self.unwound.set()
            self.closed.set()
            raise

    async def aclose(self) -> None:
        self.aclose_calls += 1
        if self.closed.is_set():
            return
        self._cancelled = True
        self.unwound.set()
        self.closed.set()
        if self._loop is not None and self._release_gate is not None:
            self._loop.call_soon_threadsafe(self._release_gate.set)

    def release_naturally(self) -> None:
        """Let a blocked iterator exhaust without cancelling it."""

        assert self.waiting.wait(1.0), "provider never entered its blocking read"
        assert self._loop is not None and self._release_gate is not None
        self._loop.call_soon_threadsafe(self._release_gate.set)


class FakeAsyncClient:
    """Small subset of ``ollama.AsyncClient`` touched by the bridge."""

    def __init__(self, response: FakeAsyncChunks) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []
        self.close_calls = 0
        self.closed = threading.Event()

    async def chat(self, **kwargs: Any) -> FakeAsyncChunks:
        self.calls.append(kwargs)
        return self.response

    async def close(self) -> None:
        self.close_calls += 1
        # Real AsyncClient.close() tears down the active HTTP response.  Model
        # that wake-up as well as task.cancel(), so either valid implementation
        # order deterministically releases the fake provider read.
        if not self.response.closed.is_set():
            await self.response.aclose()
        self.closed.set()


class FakeSyncClient:
    """Existing ``client=`` path retained for non-streaming requests."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> dict[str, dict[str, str]]:
        self.calls.append(kwargs)
        return {"message": {"content": "sync reply"}}


class FakeAsyncClientFactory:
    """Returns one independently owned client for each stream invocation."""

    def __init__(self, *responses: FakeAsyncChunks) -> None:
        self._responses = list(responses)
        self.clients: list[FakeAsyncClient] = []
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self._lock = threading.Lock()

    def __call__(self, *args: Any, **kwargs: Any) -> FakeAsyncClient:
        with self._lock:
            self.calls.append((args, kwargs))
            index = len(self.clients)
            if index >= len(self._responses):
                raise AssertionError("async client factory called too many times")
            client = FakeAsyncClient(self._responses[index])
            self.clients.append(client)
            return client


class GatedAsyncClientFactory(FakeAsyncClientFactory):
    """Pause client construction so task cancellation can win that race."""

    def __init__(self, *responses: FakeAsyncChunks) -> None:
        super().__init__(*responses)
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, *args: Any, **kwargs: Any) -> FakeAsyncClient:
        self.entered.set()
        self.release.wait()
        return super().__call__(*args, **kwargs)


class SingleAsyncClientFactory:
    """Return one prebuilt client while retaining constructor-call evidence."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self._claimed = False
        self._lock = threading.Lock()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            if self._claimed:
                raise AssertionError("async client factory called too many times")
            self._claimed = True
            self.calls.append((args, kwargs))
        return self.client


class BlockingChatClient:
    """Block while awaiting chat(), before an async token iterator exists."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.waiting = threading.Event()
        self.task_cancelled = threading.Event()
        self.closed = threading.Event()
        self.close_calls = 0

    async def chat(self, **kwargs: Any) -> FakeAsyncChunks:
        self.calls.append(kwargs)
        self.waiting.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.task_cancelled.set()
            raise
        raise AssertionError("unreachable")

    async def close(self) -> None:
        self.close_calls += 1
        self.closed.set()


class RaisingAcloseChunks(FakeAsyncChunks):
    """Provider failure followed by a broken response-cleanup hook."""

    def __init__(self, close_error: BaseException) -> None:
        super().__init__(error=ProviderBoom("original provider failure"))
        self.close_error = close_error

    async def aclose(self) -> None:
        self.aclose_calls += 1
        raise self.close_error


class RaisingCloseClient(FakeAsyncClient):
    """Provider failure followed by a broken client-cleanup hook."""

    def __init__(self, response: FakeAsyncChunks, close_error: BaseException) -> None:
        super().__init__(response)
        self.close_error = close_error

    async def close(self) -> None:
        self.close_calls += 1
        self.closed.set()
        raise self.close_error


class GatedAcloseChunks(FakeAsyncChunks):
    """Hold response cleanup so Hedge shutdown join semantics are observable."""

    def __init__(self) -> None:
        super().__init__(block_after_chunks=True)
        self.cleanup_waiting = threading.Event()
        self._release_cleanup = threading.Event()

    async def aclose(self) -> None:
        self.aclose_calls += 1
        self.cleanup_waiting.set()
        while not self._release_cleanup.is_set():
            await asyncio.sleep(0.005)

    def release_cleanup(self) -> None:
        assert self.cleanup_waiting.wait(1.0)
        self._release_cleanup.set()


@dataclass
class Consumer:
    stream: Any
    pieces: list[str] = field(default_factory=list)
    error: BaseException | None = None
    thread: threading.Thread = field(init=False)

    def __post_init__(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        try:
            self.pieces.extend(self.stream)
        except BaseException as exc:  # captured for an assertion on the caller
            self.error = exc

    def start(self) -> None:
        self.thread.start()

    def join(self) -> None:
        self.thread.join(timeout=1.0)
        assert not self.thread.is_alive(), "sync stream consumer stayed blocked"


def _async_llm(factory: FakeAsyncClientFactory, **kwargs: Any) -> OllamaLLM:
    return OllamaLLM(
        model="minicpm5-1b:q8",
        async_client_factory=factory,
        **kwargs,
    )


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_context_cancel_before_first_token_unwinds_provider_and_wakes_consumer():
    response = FakeAsyncChunks(block_after_chunks=True)
    factory = FakeAsyncClientFactory(response)
    cancel = threading.Event()

    # Reset before another OS thread consumes the iterator: this proves stream()
    # captured the task Event instead of consulting that thread's empty Context.
    token = capability_context.set({"cancel_event": cancel})
    try:
        stream = _async_llm(factory).stream("hello")
    finally:
        capability_context.reset(token)

    consumer = Consumer(stream)
    consumer.start()
    assert response.waiting.wait(1.0), "provider never began its first-token read"

    cancel.set()  # the same operation AgentTask.cancel() performs on barge-in
    consumer.join()

    assert consumer.pieces == []
    assert consumer.error is None
    assert response.task_cancelled.wait(1.0), (
        "request Task never received CancelledError"
    )
    assert response.unwound.wait(1.0), "cancel never reached the async provider"
    assert response.closed.is_set()
    assert factory.clients[0].closed.wait(1.0)
    assert factory.clients[0].close_calls == 1


def test_explicit_cancel_after_a_token_keeps_prefix_and_unwinds_provider():
    response = FakeAsyncChunks("first ", block_after_chunks=True)
    factory = FakeAsyncClientFactory(response)
    stream = _async_llm(factory).stream("hello")

    delivered = next(stream)
    assert response.waiting.wait(1.0)

    stream.cancel()
    tail = list(stream)

    assert delivered == "first "
    assert tail == []
    assert response.task_cancelled.wait(1.0)
    assert response.unwound.wait(1.0)
    assert factory.clients[0].closed.wait(1.0)
    assert factory.clients[0].close_calls == 1


def test_cancel_and_close_are_idempotent():
    response = FakeAsyncChunks(block_after_chunks=True)
    factory = FakeAsyncClientFactory(response)
    stream = _async_llm(factory).stream("hello")
    consumer = Consumer(stream)
    consumer.start()
    assert response.waiting.wait(1.0)

    stream.close()
    assert factory.clients[0].closed.wait(1.0), (
        "owning close returned before provider cleanup"
    )
    stream.close()
    stream.cancel()
    stream.cancel()
    consumer.join()

    assert consumer.error is None
    assert response.unwound.is_set()
    assert factory.clients[0].close_calls == 1


def test_close_before_iteration_never_starts_a_client_or_request():
    response = FakeAsyncChunks(block_after_chunks=True)
    factory = FakeAsyncClientFactory(response)
    stream = _async_llm(factory).stream("hello")

    stream.close()

    assert list(stream) == []
    assert factory.calls == []
    assert factory.clients == []
    assert not response.waiting.is_set()


def test_context_cancel_during_client_factory_prevents_late_chat_start():
    response = FakeAsyncChunks(block_after_chunks=True)
    factory = GatedAsyncClientFactory(response)
    cancel = threading.Event()
    token = capability_context.set({"cancel_event": cancel})
    try:
        stream = _async_llm(factory).stream("hello")
    finally:
        capability_context.reset(token)

    consumer = Consumer(stream)
    consumer.start()
    try:
        assert factory.entered.wait(1.0), "async client factory never started"
        cancel.set()
    finally:
        # The factory is deliberately synchronous, matching AsyncClient's
        # constructor. Release only after cancellation has won in the test.
        factory.release.set()
    consumer.join()

    assert consumer.error is None
    assert len(factory.clients) == 1
    assert factory.clients[0].closed.wait(1.0)
    assert factory.clients[0].calls == [], (
        "chat started after the captured task Event was already cancelled"
    )


def test_explicit_cancel_reaches_task_blocked_while_awaiting_chat():
    client = BlockingChatClient()
    factory = SingleAsyncClientFactory(client)
    stream = _async_llm(factory).stream("hello")
    consumer = Consumer(stream)
    consumer.start()
    assert client.waiting.wait(1.0), "chat() never reached its blocking await"

    stream.cancel()
    consumer.join()

    assert consumer.pieces == []
    assert consumer.error is None
    assert client.task_cancelled.wait(1.0), (
        "request Task cancellation did not interrupt chat()"
    )
    assert client.closed.wait(1.0)
    assert client.close_calls == 1


def test_cancel_discards_tokens_already_buffered_by_async_producer():
    response = FakeAsyncChunks(
        "delivered", " buffered-one", " buffered-two", block_after_chunks=True
    )
    factory = FakeAsyncClientFactory(response)
    stream = _async_llm(factory).stream("hello")

    assert next(stream) == "delivered"
    # Reaching the next blocking read proves every preceding chunk was enqueued.
    assert response.waiting.wait(1.0)
    stream.cancel()
    tail = list(stream)

    assert factory.clients[0].closed.wait(1.0)
    assert response.task_cancelled.is_set()
    assert tail == [], "cancelled stream yielded pre-cancel queue backlog"


def test_provider_error_is_reraised_by_sync_iterator():
    response = FakeAsyncChunks(error=ProviderBoom("provider read failed"))
    factory = FakeAsyncClientFactory(response)

    with pytest.raises(ProviderBoom, match="provider read failed"):
        list(_async_llm(factory).stream("hello"))

    assert factory.clients[0].closed.wait(1.0)
    assert factory.clients[0].close_calls == 1
    assert response.closed.is_set()
    assert response.aclose_calls == 1
    assert not response.unwound.is_set()


@pytest.mark.parametrize(
    "cleanup_error",
    [RuntimeError("aclose failed"), asyncio.CancelledError()],
    ids=["exception", "cancelled-error"],
)
def test_response_cleanup_failure_preserves_original_provider_error(cleanup_error):
    response = RaisingAcloseChunks(cleanup_error)
    factory = FakeAsyncClientFactory(response)
    stream = _async_llm(factory).stream("hello")

    with pytest.raises(ProviderBoom, match="original provider failure"):
        list(stream)

    assert stream._producer.done.is_set()  # noqa: SLF001 - lifecycle assertion
    assert response.aclose_calls == 1
    assert factory.clients[0].close_calls == 1
    assert factory.clients[0].closed.is_set()


@pytest.mark.parametrize(
    "cleanup_error",
    [RuntimeError("client close failed"), asyncio.CancelledError()],
    ids=["exception", "cancelled-error"],
)
def test_client_cleanup_failure_preserves_original_provider_error(cleanup_error):
    response = FakeAsyncChunks(error=ProviderBoom("original provider failure"))
    client = RaisingCloseClient(response, cleanup_error)
    stream = _async_llm(SingleAsyncClientFactory(client)).stream("hello")

    with pytest.raises(ProviderBoom, match="original provider failure"):
        list(stream)

    assert stream._producer.done.is_set()  # noqa: SLF001 - lifecycle assertion
    assert response.aclose_calls == 1
    assert client.close_calls == 1
    assert client.closed.is_set()


def test_concurrent_stream_cancellation_is_isolated():
    first = FakeAsyncChunks(block_after_chunks=True)
    second = FakeAsyncChunks(block_after_chunks=True)
    factory = FakeAsyncClientFactory(first, second)
    llm = _async_llm(factory)

    consumer_one = Consumer(llm.stream("one"))
    consumer_one.start()
    assert first.waiting.wait(1.0)
    consumer_two = Consumer(llm.stream("two"))
    consumer_two.start()
    assert second.waiting.wait(1.0)

    consumer_one.stream.cancel()
    consumer_one.join()

    assert first.unwound.is_set()
    assert consumer_two.thread.is_alive()  # still blocked, not cancelled
    assert not second.unwound.is_set()
    assert factory.clients[0].closed.wait(1.0)
    assert factory.clients[0].close_calls == 1
    assert factory.clients[1].close_calls == 0

    second.release_naturally()
    consumer_two.join()
    assert consumer_two.error is None
    assert not second.unwound.is_set()
    assert factory.clients[1].closed.wait(1.0)
    assert factory.clients[1].close_calls == 1


def test_natural_completion_yields_content_and_closes_owned_client():
    response = FakeAsyncChunks("hello", "", " world")
    factory = FakeAsyncClientFactory(response)

    out = list(_async_llm(factory).stream("hello"))

    assert out == ["hello", " world"]
    assert factory.clients[0].closed.wait(1.0)
    assert response.closed.is_set()
    assert response.aclose_calls == 1
    assert not response.unwound.is_set()
    assert factory.clients[0].close_calls == 1


def test_request_kwargs_are_preserved_and_llm_is_reusable_after_completion():
    first = FakeAsyncChunks("one")
    second = FakeAsyncChunks("two")
    factory = FakeAsyncClientFactory(first, second)
    sync_client = FakeSyncClient()
    history = [
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "before"},
    ]
    llm = _async_llm(
        factory,
        options={"num_ctx": 8192, "temperature": 0.7},
        keep_alive="30m",
        think=False,
        host="http://ollama.test:11434",
        timeout=2.5,
        client=sync_client,
    )

    assert list(
        llm.stream(
            "first prompt",
            system="be concise",
            images=[b"image"],
            history=history,
        )
    ) == ["one"]
    assert list(llm.stream("second prompt")) == ["two"]
    assert llm.generate("non-streaming prompt") == "sync reply"

    assert len(factory.clients) == 2
    assert factory.calls == [
        ((), {"host": "http://ollama.test:11434", "timeout": 2.5}),
        ((), {"host": "http://ollama.test:11434", "timeout": 2.5}),
    ]
    assert [client.close_calls for client in factory.clients] == [1, 1]
    assert sync_client.calls == [
        {
            "model": "minicpm5-1b:q8",
            "messages": [
                {"role": "user", "content": "non-streaming prompt"},
            ],
            "stream": False,
            "options": {"num_ctx": 8192, "temperature": 0.7},
            "keep_alive": "30m",
            "think": False,
        }
    ]
    first_call = factory.clients[0].calls[0]
    assert first_call == {
        "model": "minicpm5-1b:q8",
        "messages": [
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "before"},
            {"role": "user", "content": "first prompt", "images": [b"image"]},
        ],
        "stream": True,
        "options": {"num_ctx": 8192, "temperature": 0.7},
        "keep_alive": "30m",
        "think": False,
    }
    assert factory.clients[1].calls[0] == {
        "model": "minicpm5-1b:q8",
        "messages": [{"role": "user", "content": "second prompt"}],
        "stream": True,
        "options": {"num_ctx": 8192, "temperature": 0.7},
        "keep_alive": "30m",
        "think": False,
    }


def test_abandoned_full_queue_self_cancels_and_closes_transport():
    response = FakeAsyncChunks(*(["x"] * 200))
    factory = FakeAsyncClientFactory(response)
    stream = _async_llm(factory).stream("hello")
    stream._QUEUE_FULL_TIMEOUT_SEC = 0.1  # noqa: SLF001 - deterministic fault budget

    assert next(stream) == "x"
    # Deliberately abandon without close(): bounded backpressure must self-heal
    # instead of leaving its loop/client thread alive forever.
    assert factory.clients[0].closed.wait(1.0)
    assert response.unwound.is_set()
    assert factory.clients[0].close_calls == 1


def test_dropped_sparse_iterator_cancels_detached_producer_and_client():
    response = FakeAsyncChunks("first", block_after_chunks=True)
    factory = FakeAsyncClientFactory(response)
    stream = _async_llm(factory).stream("hello")

    assert next(stream) == "first"
    assert response.waiting.wait(1.0)
    producer_done = stream._producer.done  # noqa: SLF001 - lifecycle assertion
    producer_thread = stream._thread  # noqa: SLF001 - thread leak assertion
    stream_ref = weakref.ref(stream)
    del stream
    gc.collect()

    assert stream_ref() is None, "producer retained its abandoned public iterator"
    assert response.task_cancelled.wait(1.0)
    assert factory.clients[0].closed.wait(1.0)
    assert producer_done.wait(1.0)
    assert producer_thread is not None
    producer_thread.join(timeout=1.0)
    assert not producer_thread.is_alive()


def test_hedge_cancels_pretoken_ollama_loser_before_winner_finishes():
    response = FakeAsyncChunks(block_after_chunks=True)
    factory = FakeAsyncClientFactory(response)
    local = _async_llm(factory)

    class GatedWinner:
        def __init__(self) -> None:
            self.release = threading.Event()

        def generate(self, prompt, *, system=None, images=None) -> str:
            return "".join(self.stream(prompt, system=system, images=images))

        def stream(self, prompt, *, system=None, images=None):
            yield "winner"
            self.release.wait()
            yield " tail"

    winner = GatedWinner()
    hedge = HedgeLLM(local=local, cloud=winner, hedge_delay_ms=100)
    stream = hedge.stream("race")
    try:
        assert next(stream) == "winner"
        assert response.waiting.wait(1.0), "local loser never entered pre-token read"
        assert response.unwound.wait(1.0), (
            "Ollama loser kept generating until the winning answer completed"
        )
        assert not winner.release.is_set(), "test winner must still be in flight"
        assert factory.clients[0].closed.wait(1.0)
    finally:
        winner.release.set()
        stream.close()


def test_hedge_close_waits_for_owned_ollama_loser_cleanup():
    response = GatedAcloseChunks()
    factory = FakeAsyncClientFactory(response)

    class ImmediateWinner:
        def stream(self, prompt, *, system=None, images=None):
            assert response.waiting.wait(1.0)
            yield "winner"

    hedge = HedgeLLM(
        local=_async_llm(factory),
        cloud=ImmediateWinner(),
        hedge_delay_ms=0,
    )
    stream = hedge.stream("race")
    assert next(stream) == "winner"
    assert response.cleanup_waiting.wait(1.0)

    close_started = threading.Event()
    close_finished = threading.Event()

    def close_hedge() -> None:
        close_started.set()
        stream.close()
        close_finished.set()

    closer = threading.Thread(target=close_hedge, daemon=True)
    closer.start()
    assert close_started.wait(1.0)
    try:
        assert not close_finished.wait(0.1), (
            "Hedge released its outer provider before Ollama cleanup"
        )
    finally:
        response.release_cleanup()
    assert close_finished.wait(1.0)
    closer.join(timeout=1.0)
    assert not closer.is_alive()
    assert factory.clients[0].closed.wait(1.0)


def test_barge_storm_reuses_provider_slots_without_manual_release():
    blocked = [
        FakeAsyncChunks(block_after_chunks=True)
        for _ in range(DEFAULT_MAX_ACTIVE_TASKS + 2)
    ]
    healthy = FakeAsyncChunks("Healthy after native cancellations.")
    factory = FakeAsyncClientFactory(*blocked, healthy)
    llm = _async_llm(factory)
    engine = ScriptedEngine()
    runtime = VoiceRuntime(engine, llm, start_mode=Mode.ASSISTANT, stream_tts=True)
    runtime.start(run_bus=True)
    attempts = runtime.supervisor.tasks.max_active_tasks + 2
    try:
        for index in range(attempts):
            engine.final(f"block before token {index}")
            assert _wait_until(lambda: len(factory.clients) == index + 1)
            response = blocked[index]
            assert response.waiting.wait(1.0)

            engine.barge_in()

            assert response.unwound.wait(1.0)
            assert factory.clients[index].closed.wait(1.0)
            assert _wait_until(
                lambda: runtime.supervisor.tasks.active_count == 0
                and not runtime.supervisor.state.active_tasks
            )

        # No test-owned release gate is touched. Native request cancellation
        # itself returned every bulkhead slot, so the next turn starts and talks.
        engine.final("healthy turn")
        assert _wait_until(lambda: len(factory.clients) == attempts + 1)
        assert runtime.wait_idle(timeout=2.0)
        assert engine.spoken == ["Healthy after native cancellations."]
    finally:
        runtime.stop()
