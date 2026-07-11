from __future__ import annotations

import asyncio
import inspect
import logging
import os
import queue
import re
import threading
import time
import weakref
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import (
    Callable,
    Iterator,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

from .llm_threads import DEFAULT_LLM_THREAD_RESERVE, resolve_llamacpp_thread_pair

# Per-turn context published by the capability layer so the LLM stack can
# pick a routing chain (sensitivity / intent_kind / mode) without changing
# the LLMClient protocol. Default is an empty mapping so callers that don't
# set it always land on the safe-default chain.
capability_context: ContextVar[Mapping[str, object]] = ContextVar(
    "speaker_capability_context", default={}
)

_ollama_log = logging.getLogger("speaker.llm.ollama")
_hedge_log = logging.getLogger("speaker.llm.hedge")
_llamacpp_log = logging.getLogger("speaker.llm.llamacpp")


def _log_llm_request(
    log: logging.Logger,
    model: str,
    prompt: str,
    system: Optional[str],
    *,
    dt: float,
    out_chars: int,
    tokens: Optional[int] = None,
    ttft: Optional[float] = None,
    streamed: bool,
    cancelled: bool = False,
) -> None:
    """One structured line per LLM call -> feeds the run summary via ``extra``.

    The full prompt goes to the DEBUG file (so a committed log shows exactly what
    was asked); the INFO line carries timings + a short preview."""
    preview = " ".join((prompt or "").split())[:200]
    req = {
        "model": model,
        "prompt_chars": len(prompt or ""),
        "system_chars": len(system or ""),
        "prompt_preview": preview,
        "duration_sec": round(dt, 3),
        "ttft_sec": round(ttft, 3) if ttft is not None else None,
        "out_chars": out_chars,
        "tokens": tokens,
        "streamed": streamed,
        "cancelled": cancelled,
    }
    log.debug("ollama %s full prompt: %s", model, prompt)
    log.info(
        "ollama %s: %.2fs ttft=%s out=%dch tokens=%s%s | %r",
        model, dt,
        f"{ttft:.2f}s" if ttft is not None else "-",
        out_chars, tokens if tokens is not None else "-",
        " CANCELLED" if cancelled else "",
        preview,
        extra={"llm_request": req},
    )

# An "image" is either a path to an image file or raw image bytes. The Ollama
# client accepts both for multimodal models (e.g. Gemma 3 4b/12b/27b). Text-only
# models such as gemma3:1b ignore images.
ImageInput = str | bytes

# A prior conversation turn for multi-turn chat: ``{"role": "user"|"assistant",
# "content": str}``. Optional on every LLMClient call; ``None`` (the default)
# keeps the single-turn prompt path byte-identical. When provided, an
# implementation inserts these BETWEEN the system prompt and the CURRENT user
# turn, so the model sees real chat history (R11) instead of history pasted as
# text into the system string -- which the small answering model handles far
# better ("continue the story" actually continues it).
HistoryTurn = Mapping[str, str]


def _history_messages(history: "Optional[Sequence[HistoryTurn]]") -> list[dict]:
    """Normalize prior turns to chat messages, dropping malformed/empty entries.

    Shared by the Ollama and OpenAI-style message builders so both runtimes turn
    the same ``[{'role','content'}, ...]`` into the same message list."""
    out: list[dict] = []
    for turn in history or ():
        role = str(turn.get("role", "")).strip().lower()
        content = str(turn.get("content", "")).strip()
        if role in ("user", "assistant") and content:
            out.append({"role": role, "content": content})
    return out


@runtime_checkable
class LLMClient(Protocol):
    """Minimal local-LLM contract used by the runtime's capabilities.

    ``images`` is optional so the same client serves text-only and multimodal
    callers; implementations backed by a text-only model simply ignore it.
    ``history`` is optional prior-turn context (``HistoryTurn`` list); ``None``
    keeps the single-turn path byte-identical.
    """

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[ImageInput]] = None,
        history: Optional[Sequence[HistoryTurn]] = None,
    ) -> str: ...

    def stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[ImageInput]] = None,
        history: Optional[Sequence[HistoryTurn]] = None,
    ) -> Iterator[str]: ...


class LLMCallCancelled(RuntimeError):
    """Control-flow signal for a retired cancellable LLM invocation."""


class LLMProviderOutputError(RuntimeError):
    """Provider completed without a safe user-visible answer."""


class _CombinedCancelEvent:
    """Event-like OR over explicit and inherited cancellation sources."""

    def __init__(self, *events: object) -> None:
        self._events = tuple(event for event in events if event is not None)

    def is_set(self) -> bool:
        for event in self._events:
            check = getattr(event, "is_set", None)
            if not callable(check):
                continue
            try:
                if check():
                    return True
            except Exception:
                continue
        return False


def collect_llm_text(
    llm: LLMClient,
    prompt: str,
    *,
    system: Optional[str] = None,
    cancel_event: object | None = None,
) -> str:
    """Collect a stream while preserving the provider's cancellation seam.

    Pre-task classifiers historically called blocking ``generate()``. This
    helper keeps their string-returning API but uses ``stream()`` so Ollama can
    cancel before token one. Cancellation is distinct from provider failure and
    never returns a partial classification/rewrite.
    """
    inherited = capability_context.get()
    inherited_cancel = inherited.get("cancel_event")
    if cancel_event is None:
        effective_cancel = inherited_cancel
    elif inherited_cancel is None or inherited_cancel is cancel_event:
        effective_cancel = cancel_event
    else:
        effective_cancel = _CombinedCancelEvent(
            inherited_cancel,
            cancel_event,
        )
    context = dict(inherited)
    if effective_cancel is not None:
        context["cancel_event"] = effective_cancel
    context_token = capability_context.set(context)
    cancel_is_set = _CombinedCancelEvent(effective_cancel).is_set
    stream = None
    try:
        if cancel_is_set():
            raise LLMCallCancelled("LLM call cancelled before stream start")
        stream = llm.stream(prompt, system=system)
        pieces: list[str] = []
        for piece in stream:
            if cancel_is_set():
                raise LLMCallCancelled("LLM call cancelled during stream")
            if piece:
                pieces.append(str(piece))
        # Cancellation wins a simultaneous natural completion.
        if cancel_is_set():
            raise LLMCallCancelled("LLM call cancelled at stream completion")
        return "".join(pieces)
    except (Exception, asyncio.CancelledError) as exc:
        # A provider commonly surfaces its own cancellation as an exception.
        # If our retirement signal won the race, preserve the dedicated control
        # flow instead of letting a gate mistake it for an ordinary failure and
        # cache/fall back from a stale call.
        if cancel_is_set() and not isinstance(exc, LLMCallCancelled):
            raise LLMCallCancelled("LLM call cancelled during provider failure") from exc
        raise
    finally:
        try:
            closer = getattr(stream, "close", None)
            if callable(closer):
                try:
                    closer()
                except (Exception, asyncio.CancelledError):
                    pass
        finally:
            capability_context.reset(context_token)


class EchoLLM:
    """Deterministic fake LLM for tests and the offline console demo."""

    def __init__(self, reply: Optional[str] = None):
        self._reply = reply

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[ImageInput]] = None,
        history: Optional[Sequence[HistoryTurn]] = None,
    ) -> str:
        if self._reply is not None:
            return self._reply
        suffix = f" [+{len(images)} image(s)]" if images else ""
        turns = _history_messages(history)
        hist = f" [+{len(turns)} prior turn(s)]" if turns else ""
        return f"You said: {prompt}{suffix}{hist}"

    def stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[ImageInput]] = None,
        history: Optional[Sequence[HistoryTurn]] = None,
    ) -> Iterator[str]:
        yield self.generate(prompt, system=system, images=images, history=history)


@dataclass(frozen=True)
class _OllamaStreamFailure:
    error: BaseException


class _OllamaAsyncStreamProducer:
    """Own one async request without retaining its public sync iterator."""

    def __init__(
        self,
        owner: "OllamaLLM",
        prompt: str,
        system: Optional[str],
        kwargs: dict,
        cancel_event: object | None,
        *,
        queue_capacity: int,
    ) -> None:
        self.owner = owner
        self.prompt = prompt
        self.system = system
        self.kwargs = kwargs
        self.cancel_event = cancel_event
        self.items: "queue.Queue[object]" = queue.Queue(maxsize=queue_capacity)
        self.lock = threading.Lock()
        self.cancel_requested = False
        self.done = threading.Event()
        self.cancel_poll_sec = 0.01
        self.queue_poll_sec = 0.005
        self.queue_full_timeout_sec = 5.0

    @staticmethod
    def event_is_set(event: object | None) -> bool:
        check = getattr(event, "is_set", None)
        if not callable(check):
            return False
        try:
            return bool(check())
        except Exception:
            return False

    def is_cancelled(self) -> bool:
        with self.lock:
            return self.cancel_requested

    def cancel(self) -> None:
        """Request cancellation without waiting for provider teardown."""
        with self.lock:
            self.cancel_requested = True

    def sync_external_cancel(self) -> bool:
        if self.event_is_set(self.cancel_event):
            self.cancel()
        return self.is_cancelled()

    async def enqueue(self, item: object) -> bool:
        """Bounded async-to-sync put that keeps cancellation schedulable."""
        full_since: Optional[float] = None
        while True:
            if self.is_cancelled():
                return False
            try:
                self.items.put_nowait(item)
                return True
            except queue.Full:
                if full_since is None:
                    full_since = time.monotonic()
                elif time.monotonic() - full_since >= self.queue_full_timeout_sec:
                    self.cancel()
                    return False
                # Never block the event loop in Queue.put: its cancellation
                # watcher must stay runnable while a consumer applies pressure.
                await asyncio.sleep(self.queue_poll_sec)

    def thread_main(self) -> None:
        try:
            asyncio.run(self.produce())
        except BaseException as exc:
            # asyncio.run itself should only fail outside the guarded producer.
            try:
                self.items.put_nowait(_OllamaStreamFailure(exc))
            except queue.Full:
                pass
            finally:
                self.done.set()

    async def watch_cancel(self, request_task: asyncio.Task) -> None:
        while True:
            if self.sync_external_cancel():
                # Cancel only the child that owns chat and async iteration. The
                # parent stays alive to close the response and owned client.
                request_task.cancel()
                return
            await asyncio.sleep(self.cancel_poll_sec)

    @staticmethod
    async def close_resource(resource: object, method: str) -> None:
        closer = getattr(resource, method, None)
        if not callable(closer):
            return
        try:
            result = closer()
            if inspect.isawaitable(result):
                await result
        except (Exception, asyncio.CancelledError):
            # Cleanup failure must not suppress the original provider failure or
            # prevent later client-close and done signaling.
            pass

    async def produce(self) -> None:
        t0 = time.perf_counter()
        ttft: Optional[float] = None
        token_count = 0
        out_chars = 0
        natural = False
        failure: Optional[BaseException] = None
        client = None
        stream_holder: dict[str, object] = {}
        watcher: Optional[asyncio.Task] = None

        async def consume() -> None:
            nonlocal ttft, token_count, out_chars
            # Let the watcher observe an already-set task Event before chat.
            await asyncio.sleep(0)
            if self.sync_external_cancel():
                raise asyncio.CancelledError
            response = client.chat(**self.kwargs)
            sdk_stream = await response if inspect.isawaitable(response) else response
            stream_holder["stream"] = sdk_stream
            async for chunk in sdk_stream:
                if self.sync_external_cancel():
                    raise asyncio.CancelledError
                piece = chunk.get("message", {}).get("content", "")
                if not piece:
                    continue
                if ttft is None:
                    ttft = time.perf_counter() - t0
                token_count += 1
                out_chars += len(piece)
                if not await self.enqueue(piece):
                    raise asyncio.CancelledError

        try:
            if self.sync_external_cancel():
                return
            client = self.owner._make_async_client()
            # Client construction is request-free but synchronous. Recheck both
            # cancellation sources so a slow factory cannot start a retired turn.
            if self.sync_external_cancel():
                return
            request_task = asyncio.create_task(consume())
            watcher = asyncio.create_task(self.watch_cancel(request_task))
            await request_task
            natural = True
        except asyncio.CancelledError:
            pass
        except BaseException as exc:
            failure = exc
        finally:
            if watcher is not None:
                watcher.cancel()
                try:
                    await watcher
                except (Exception, asyncio.CancelledError):
                    pass
            sdk_stream = stream_holder.get("stream")
            if sdk_stream is not None:
                await self.close_resource(sdk_stream, "aclose")
            if client is not None:
                await self.close_resource(client, "close")
            try:
                _log_llm_request(
                    _ollama_log,
                    self.owner.model,
                    self.prompt,
                    self.system,
                    dt=time.perf_counter() - t0,
                    out_chars=out_chars,
                    tokens=token_count,
                    ttft=ttft,
                    streamed=True,
                    cancelled=not natural,
                )
            except Exception:
                pass
            try:
                if failure is not None:
                    await self.enqueue(_OllamaStreamFailure(failure))
            finally:
                self.done.set()


class _OllamaAsyncTokenStream:
    """Sync iterator backed by one cancellable async Ollama request.

    The synchronous Ollama generator cannot be closed safely while a different
    thread executes its pre-first-token next call. The official async client can
    be cancelled there by cancelling the Task that owns its async iteration.
    Each stream owns a short-lived daemon loop thread and client; dropping or
    closing the public iterator cancels its detached producer.
    """

    _QUEUE_CAPACITY = 64
    _CANCEL_POLL_SEC = 0.01
    _QUEUE_POLL_SEC = 0.005
    _QUEUE_FULL_TIMEOUT_SEC = 5.0

    # Nominal opt-in consumed by HedgeLLM before invoking cancel from another
    # thread. A cancel method by itself is not a thread-safety contract.
    _cross_thread_cancel_safe = True

    def __init__(
        self,
        owner: "OllamaLLM",
        prompt: str,
        system: Optional[str],
        kwargs: dict,
        cancel_event: object | None,
    ) -> None:
        self._producer = _OllamaAsyncStreamProducer(
            owner,
            prompt,
            system,
            kwargs,
            cancel_event,
            queue_capacity=self._QUEUE_CAPACITY,
        )
        self._lock = threading.Lock()
        self._started = False
        self._terminal_seen = False
        self._thread: Optional[threading.Thread] = None
        # The thread owns the producer, not this iterator. Dropping a partially
        # consumed iterator therefore cancels even when its queue never fills.
        self._finalizer = weakref.finalize(self, self._producer.cancel)

    def __iter__(self) -> "_OllamaAsyncTokenStream":
        return self

    def _ensure_started(self) -> None:
        with self._lock:
            if self._terminal_seen or self._started:
                return
            if self._producer.sync_external_cancel():
                return
            self._producer.cancel_poll_sec = self._CANCEL_POLL_SEC
            self._producer.queue_poll_sec = self._QUEUE_POLL_SEC
            self._producer.queue_full_timeout_sec = self._QUEUE_FULL_TIMEOUT_SEC
            self._started = True
            thread = threading.Thread(
                target=self._producer.thread_main,
                name=f"ollama-stream-{self._producer.owner.model}",
                daemon=True,
            )
            self._thread = thread
            try:
                thread.start()
            except BaseException:
                self._started = False
                self._thread = None
                raise

    def _mark_terminal(self) -> None:
        with self._lock:
            self._terminal_seen = True

    def __next__(self) -> str:
        self._ensure_started()
        with self._lock:
            if self._terminal_seen or not self._started:
                raise StopIteration
        while True:
            if self._producer.sync_external_cancel():
                # Discard queued pre-cancel output, but hold the provider slot
                # until the request/client cleanup has really completed.
                if self._producer.done.wait(timeout=0.05):
                    self._mark_terminal()
                    raise StopIteration
                continue
            try:
                item = self._producer.items.get(timeout=0.05)
            except queue.Empty:
                if self._producer.done.is_set():
                    self._mark_terminal()
                    raise StopIteration
                continue
            if self._producer.sync_external_cancel():
                continue
            if isinstance(item, _OllamaStreamFailure):
                self._mark_terminal()
                raise item.error
            return str(item)

    def cancel(self) -> None:
        self._producer.cancel()

    def close(self) -> None:
        self.cancel()
        # close() is the owning-consumer teardown path: do not let that provider
        # invocation release its bulkhead slot before cooperative SDK cleanup.
        # Hedge's cross-thread path calls the deliberately nonblocking cancel().
        with self._lock:
            started = self._started
            producer_thread = self._thread
        if started and threading.current_thread() is not producer_thread:
            self._producer.done.wait()


class OllamaLLM:
    """Local LLM via Ollama (GPU-accelerated for Gemma 3 on a CUDA host).

    The ``ollama`` package is imported lazily so the rest of the runtime (and
    the test suite) works in environments without it installed. Pass ``options``
    to tune the model server-side, e.g. ``{"num_ctx": 4096}`` for context size
    or ``{"num_gpu": 999}`` to force full GPU offload.

    Multimodal: pass ``images`` (file paths or bytes) to ``generate``/``stream``
    with a vision-capable model (gemma3:4b / 12b / 27b).

    ``think`` controls Ollama's reasoning-model "thinking" phase. A reasoning
    model (e.g. gemma4) streams a silent chain-of-thought into a SEPARATE
    ``thinking`` field BEFORE any ``content`` token -- which our stream only
    yields content from, so for voice that thinking is pure dead air: measured
    ~9 s of silence before the first spoken word of a *story* on gemma4:12b.
    ``think=False`` skips it (first content token ~1.9 s instead), ``True``
    forces it on, ``None`` (default here) leaves the model's own default. The
    voice factory (:func:`core.llm_factory.build_llms`) defaults it to ``False``
    -- thinking's multi-second latency is unacceptable for a real-time voice
    turn. Passed only when not ``None`` so non-reasoning models / older Ollama
    builds are unaffected.
    """

    def __init__(
        self,
        model: str = "gemma3:12b",
        host: Optional[str] = None,
        *,
        options: Optional[dict] = None,
        keep_alive: Optional[str | int] = None,
        timeout: Optional[float] = 60.0,
        think: Optional[bool] = None,
        client=None,
        async_client_factory=None,
    ):
        self.model = model
        self._host = host
        self._options = dict(options) if options else None
        # Reasoning-model "thinking" toggle (see the class docstring). Sent as a
        # top-level chat arg only when not None, so the default path is unchanged.
        self._think = think
        # How long Ollama keeps the model resident after a request. A long value
        # (e.g. "30m") or -1 (forever) avoids a cold reload on the next turn --
        # the single biggest win for a snappy first token on a warm box.
        self._keep_alive = keep_alive
        # Socket/read timeout (seconds) for the underlying httpx client. Without
        # it a hung Ollama daemon would wedge the turn forever instead of
        # raising (which the Hedge chain treats as a dead source and advances
        # past). ``None`` disables the timeout (the old behaviour).
        self._timeout = timeout
        self._client = client
        self._client_injected = client is not None
        self._async_client_factory = async_client_factory
        # Hedge may use this eager marker before stream construction completes.
        # The injected sync compatibility path has no cross-thread cancel seam.
        self._stream_cross_thread_cancel_safe = not (
            self._client_injected and self._async_client_factory is None
        )

    def _client_kwargs(self) -> dict:
        kwargs: dict = {}
        if self._host:
            kwargs["host"] = self._host
        if self._timeout is not None:
            kwargs["timeout"] = self._timeout
        return kwargs

    def _ensure(self):
        if self._client is None:
            import ollama  # lazy

            # Always build an explicit Client so the read timeout is applied;
            # the bare ``ollama`` module client has no timeout (hangs forever
            # on a stalled connection).
            self._client = ollama.Client(**self._client_kwargs())
        return self._client

    def _make_async_client(self):
        factory = self._async_client_factory
        if factory is None:
            import ollama  # lazy

            factory = ollama.AsyncClient
        return factory(**self._client_kwargs())

    def _messages(
        self, prompt: str, system: Optional[str], images: Optional[Sequence[ImageInput]],
        history: Optional[Sequence[HistoryTurn]] = None,
    ) -> list[dict]:
        msgs: list[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(_history_messages(history))  # prior turns between system + current
        user: dict = {"role": "user", "content": prompt}
        if images:
            user["images"] = list(images)
        msgs.append(user)
        return msgs

    def _chat_kwargs(self, prompt, system, images, *, stream: bool, history=None) -> dict:
        kwargs = {
            "model": self.model,
            "messages": self._messages(prompt, system, images, history),
            "stream": stream,
        }
        if self._options:
            kwargs["options"] = self._options
        if self._keep_alive is not None:
            kwargs["keep_alive"] = self._keep_alive
        if self._think is not None:
            kwargs["think"] = self._think
        return kwargs

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[ImageInput]] = None,
        history: Optional[Sequence[HistoryTurn]] = None,
    ) -> str:
        t0 = time.perf_counter()
        resp = self._ensure().chat(
            **self._chat_kwargs(prompt, system, images, stream=False, history=history)
        )
        out = resp["message"]["content"]
        _log_llm_request(
            _ollama_log, self.model, prompt, system,
            dt=time.perf_counter() - t0, out_chars=len(out or ""), streamed=False,
        )
        return out

    def stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[ImageInput]] = None,
        history: Optional[Sequence[HistoryTurn]] = None,
    ) -> Iterator[str]:
        # Preserve injected sync-client compatibility for existing embedders and
        # deterministic tests. Production (or an explicitly injected async
        # factory) uses the cancellable official AsyncClient bridge.
        if self._client_injected and self._async_client_factory is None:
            return self._stream_sync(
                prompt,
                system=system,
                images=images,
                history=history,
            )
        context = capability_context.get()
        return _OllamaAsyncTokenStream(
            self,
            prompt,
            system,
            self._chat_kwargs(prompt, system, images, stream=True, history=history),
            context.get("cancel_event"),
        )

    def _stream_sync(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[ImageInput]] = None,
        history: Optional[Sequence[HistoryTurn]] = None,
    ) -> Iterator[str]:
        t0 = time.perf_counter()
        ttft: Optional[float] = None
        tokens = 0
        out_chars = 0
        cancelled = True  # flipped to False only on natural exhaustion
        try:
            for chunk in self._ensure().chat(
                **self._chat_kwargs(prompt, system, images, stream=True, history=history)
            ):
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    tokens += 1
                    out_chars += len(piece)
                    yield piece
            cancelled = False
        finally:
            # Runs even if the consumer stops early (barge-in / cancel), so a
            # cut-off generation is still recorded -- useful for "where it stuck".
            _log_llm_request(
                _ollama_log, self.model, prompt, system,
                dt=time.perf_counter() - t0, out_chars=out_chars, tokens=tokens,
                ttft=ttft, streamed=True, cancelled=cancelled,
            )


def _normalize_llamacpp_options(options: Optional[dict]) -> dict:
    """Translate Ollama-vocabulary generation options to llama.cpp's request API.

    A device_profile's ``llm.options`` is written in Ollama's names (``num_ctx`` /
    ``num_predict``), but llama.cpp's ``create_chat_completion`` takes the OUTPUT
    cap as ``max_tokens`` and the context size at CONSTRUCTION (``n_ctx``), not per
    request. Passed through verbatim, ``num_predict`` was silently ignored -> the
    on-device model generated to the context limit on exactly the weakest hardware
    (llm-inference-3), and ``num_ctx`` is not a valid per-request kwarg. So:

    - ``num_predict`` -> ``max_tokens`` (the real on-device output cap), unless an
      explicit ``max_tokens`` is already present (that wins);
    - ``num_ctx`` / ``keep_alive`` are dropped (constructor- / daemon-only);
    - ``_``-prefixed keys are dropped (config.json's human-comment convention --
      llama.cpp's ``create_chat_completion`` has a strict signature and would
      ``TypeError`` on a stray ``_num_predict_comment``);
    - everything else (``temperature``, ``top_p``, ...) passes through unchanged.
    """
    if not options:
        return {}
    out: dict = {}
    num_predict = None
    for key, value in options.items():
        if key.startswith("_") or key in ("num_ctx", "keep_alive"):
            continue
        if key == "num_predict":
            num_predict = value
            continue
        out[key] = value
    if num_predict is not None and "max_tokens" not in out:
        # float() first so a numeric string / float config ("384", 384.0) still
        # bounds output instead of crashing model construction.
        out["max_tokens"] = int(float(num_predict))
    return out


# ggml type enum values (serialization constants in ggml.h -- STABLE across
# llama.cpp versions, since they are part of the on-disk/KV format). Maps the
# friendly KV-cache-quant names a device_profile uses to the int llama.cpp wants.
_KV_CACHE_GGML_TYPES = {
    "f32": 0, "f16": 1, "q8_0": 8, "q5_1": 7, "q5_0": 6, "q4_1": 3, "q4_0": 2,
}


def _resolve_kv_cache_type(value):
    """Resolve a KV-cache dtype (llm-inference-9) to the ggml int llama.cpp wants.

    Accepts a friendly name (``"q8_0"`` -> 8) or a raw int (passed through). An
    unknown string or ``None`` returns ``None`` (-> use llama.cpp's default,
    typically f16), so a typo silently keeps the safe default instead of crashing
    model construction."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    return _KV_CACHE_GGML_TYPES.get(str(value).strip().lower())


LLAMACPP_PINNED_VERSION = "0.3.33"


@dataclass(frozen=True)
class _LlamaCppAbortAPI:
    """Audited v0.3.33 low-level symbols used by the CPU abort boundary."""

    callback_type: Callable[..., object]
    set_callback: Callable[..., object]
    get_memory: Callable[..., object]
    clear_memory: Callable[..., object]


def _resolve_llamacpp_abort_api(module: object) -> _LlamaCppAbortAPI:
    """Verify the exact llama-cpp-python ABI used by phone cancellation.

    The low-level ctypes surface tracks a rapidly changing vendored llama.cpp.
    Silent drift here is worse than refusing to start: the callback object and
    context-memory functions cross a native ABI boundary.  The on-device
    requirements and real-model workflows pin the same audited release.
    """

    version = str(getattr(module, "__version__", "") or "")
    if version != LLAMACPP_PINNED_VERSION:
        raise RuntimeError(
            "llamacpp backend requires llama-cpp-python=="
            f"{LLAMACPP_PINNED_VERSION} for verified CPU cancellation; "
            f"found {version or 'unknown'}"
        )
    names = (
        "ggml_abort_callback",
        "llama_set_abort_callback",
        "llama_get_memory",
        "llama_memory_clear",
    )
    missing = [name for name in names if not callable(getattr(module, name, None))]
    if missing:
        raise RuntimeError(
            "llama-cpp-python cancellation ABI missing: " + ", ".join(missing)
        )
    return _LlamaCppAbortAPI(
        callback_type=getattr(module, "ggml_abort_callback"),
        set_callback=getattr(module, "llama_set_abort_callback"),
        get_memory=getattr(module, "llama_get_memory"),
        clear_memory=getattr(module, "llama_memory_clear"),
    )


def verify_llamacpp_abort_runtime(module: object | None = None) -> _LlamaCppAbortAPI:
    """Import and verify the selected production llama.cpp binding."""

    if module is None:
        import llama_cpp as module  # type: ignore[no-redef]  # lazy optional dep

    return _resolve_llamacpp_abort_api(module)


def _llamacpp_cancelled(event: object | None) -> bool:
    check = getattr(event, "is_set", None)
    if not callable(check):
        return False
    try:
        return bool(check())
    except BaseException:
        # A broken foreign Event must not throw through a ctypes callback.
        return False


def _llamacpp_native_context(client: object) -> object | None:
    """Return the raw ``llama_context *`` from the audited high-level client.

    llama-cpp-python keeps it at ``Llama._ctx.ctx``.  The direct ``ctx``
    fallback is a deliberately small test seam for injected clients; production
    clients are still version/symbol checked before this is called.
    """

    wrapper = getattr(client, "_ctx", None)
    native = getattr(wrapper, "ctx", None)
    if native is not None:
        return native
    return getattr(client, "ctx", None)


class _ReasoningTagFilter:
    """Incrementally remove model-visible reasoning blocks from spoken output.

    MiniCPM's supported no-think template should prevent these from being
    generated. This parser is the output-boundary safety net: drifted, nested,
    or malformed reasoning blocks must never send hidden content to TTS. Markers
    may be split across arbitrary native stream chunks.
    """

    _HTML_OPEN = "<think"
    _HTML_CLOSE = "</think"
    _SPECIAL_OPEN = "<|thought_begin|>"
    _SPECIAL_CLOSE = "<|thought_end|>"
    _MAX_NESTING = 64
    _MAX_PARTIAL_MARKER = 64
    _HTML_OPEN_RE = re.compile(r"<\s*think(?=\s|>)", re.IGNORECASE)
    _HTML_CLOSE_RE = re.compile(r"<\s*/\s*think(?=\s|>)", re.IGNORECASE)
    _SPECIAL_START_RE = re.compile(r"<\s*\|\s*", re.IGNORECASE)
    _DEFINITIONS = (
        (_HTML_OPEN, _HTML_CLOSE),
        (_HTML_CLOSE, None),
        (_SPECIAL_OPEN, _SPECIAL_CLOSE),
        (_SPECIAL_CLOSE, None),
    )

    def __init__(self, *, initial_reasoning_closer: str | None = None) -> None:
        self._buffer = ""
        self._active_closers: list[str] = (
            [initial_reasoning_closer] if initial_reasoning_closer else []
        )
        self.suppressed_chars = 0
        self.suppressed_blocks = 1 if initial_reasoning_closer else 0
        self.saw_markup = bool(initial_reasoning_closer)
        self.safe_nonspace = False
        self.malformed = False

    @classmethod
    def _partial_marker_suffix(cls, text: str) -> tuple[int, str]:
        """Return start and bounded canonical text for a partial marker."""

        candidates: list[tuple[int, str]] = []
        for marker in (cls._SPECIAL_OPEN, cls._SPECIAL_CLOSE):
            upper = min(len(text), len(marker) - 1)
            for size in range(upper, 0, -1):
                if text.lower().endswith(marker[:size].lower()):
                    candidates.append((len(text) - size, text[-size:]))
                    break

        # HTML-like markers permit whitespace around ``/`` and ``think``.
        # Canonicalize an incomplete suffix so arbitrary whitespace cannot make
        # retained stream state grow without bound.
        start = text.rfind("<")
        if start >= 0:
            tail = text[start:]

            # Also retain canonical or whitespace-deformed prefixes of the
            # special-token markers. Once ``<|thought_`` is present, any
            # truncated/divergent continuation is control markup, not speech.
            index = 1
            while index < len(tail) and tail[index].isspace():
                index += 1
            if index < len(tail) and tail[index] == "|":
                index += 1
                while index < len(tail) and tail[index].isspace():
                    index += 1
                special_tail = tail[index:].lower()
                stem = "thought_"
                valid_special_prefix = (
                    len(special_tail) <= len(stem)
                    and stem.startswith(special_tail)
                )
                if special_tail.startswith(stem):
                    suffix = special_tail[len(stem):]
                    valid_special_prefix = any(
                        target.startswith(suffix)
                        or suffix in (target, target + "|")
                        for target in ("begin", "end")
                    )
                if valid_special_prefix:
                    retained = (
                        tail
                        if len(tail) <= cls._MAX_PARTIAL_MARKER
                        else "<|" + special_tail
                    )
                    candidates.append((start, retained))

            index = 1
            while index < len(tail) and tail[index].isspace():
                index += 1
            closing = index < len(tail) and tail[index] == "/"
            if closing:
                index += 1
                while index < len(tail) and tail[index].isspace():
                    index += 1
            letters = tail[index:]
            if len(letters) <= len("think") and "think".startswith(letters.lower()):
                canonical = "</" if closing else "<"
                canonical += letters.lower()
                retained = (
                    tail if len(tail) <= cls._MAX_PARTIAL_MARKER else canonical
                )
                candidates.append((start, retained))

        if not candidates:
            return len(text), ""
        return min(candidates, key=lambda candidate: candidate[0])

    @classmethod
    def _marker_at(
        cls,
        text: str,
        marker: str,
    ) -> tuple[int, int, int] | None:
        """Return ``(start, end, prefix_end)`` for a supported marker.

        A negative ``end`` means a complete HTML prefix was observed but its
        final ``>`` has not arrived. ``prefix_end`` lets the parser enter hidden
        state without retaining unbounded attributes/content.
        """

        if marker == cls._HTML_OPEN:
            match = cls._HTML_OPEN_RE.search(text)
        elif marker == cls._HTML_CLOSE:
            match = cls._HTML_CLOSE_RE.search(text)
        else:
            # Special-token tags have no attributes. Reserve the whole
            # ``<|thought_`` namespace so truncated variants cannot become TTS
            # text merely because native chunking split before ``|>``.
            target = "begin" if marker == cls._SPECIAL_OPEN else "end"
            for special_match in cls._SPECIAL_START_RE.finditer(text):
                remainder = text[special_match.end():]
                if not remainder:
                    return None
                lowered = remainder.lower()
                stem = "thought_"
                stem_common = 0
                while (
                    stem_common < len(lowered)
                    and stem_common < len(stem)
                    and lowered[stem_common] == stem[stem_common]
                ):
                    stem_common += 1
                if stem_common < min(len(lowered), len(stem)):
                    if marker == cls._SPECIAL_OPEN:
                        return (
                            special_match.start(),
                            -1,
                            special_match.end() + stem_common,
                        )
                    continue
                if len(lowered) < len(stem):
                    return None
                mode = lowered[len(stem):]
                mode_start = special_match.end() + len(stem)
                if not mode:
                    return None
                if not mode.startswith(target[0]):
                    if marker == cls._SPECIAL_OPEN and not mode.startswith("e"):
                        return special_match.start(), -1, mode_start
                    continue
                common = 0
                while (
                    common < len(mode)
                    and common < len(target)
                    and mode[common] == target[common]
                ):
                    common += 1
                if common < min(len(mode), len(target)):
                    return special_match.start(), -1, mode_start + common
                if len(mode) < len(target):
                    return None
                prefix_end = mode_start + len(target)
                if prefix_end == len(text) or text[prefix_end:] == "|":
                    return None
                if text.startswith("|>", prefix_end):
                    end = prefix_end + 2
                    return special_match.start(), end, end
                return special_match.start(), -1, prefix_end
            return None
        if match is None:
            return None
        end = text.find(">", match.end())
        nested_start = text.find("<", match.end())
        if nested_start >= 0 and (end < 0 or nested_start < end):
            # Do not steal a later closing tag's ``>`` as this tag's terminator.
            # Enter the normal fail-closed hidden path at the confirmed prefix.
            return match.start(), -1, match.end()
        return match.start(), end + 1 if end >= 0 else -1, match.end()

    def _emit(self, text: str, visible: list[str]) -> None:
        if not text:
            return
        if self.malformed:
            self.suppressed_chars += len(text)
            return
        if not self.safe_nonspace:
            # A closed reasoning preamble commonly leaves ``\n\n`` before the
            # answer. It must not stamp first-token/TTS timing by itself.
            text = text.lstrip()
            if not text:
                return
            self.safe_nonspace = True
        visible.append(text)

    def _hide_or_emit(self, text: str, visible: list[str]) -> None:
        if self._active_closers or self.malformed:
            self.suppressed_chars += len(text)
        else:
            self._emit(text, visible)

    def feed(self, text: object) -> list[str]:
        self._buffer += str(text or "")
        visible: list[str] = []
        while self._buffer:
            candidates: list[tuple[int, int, int, str, str | None]] = []
            for marker, closer in self._DEFINITIONS:
                found = self._marker_at(self._buffer, marker)
                if found is not None:
                    candidates.append((*found, marker, closer))
            if candidates:
                index, end, prefix_end, marker, closer = min(
                    candidates,
                    key=lambda item: item[0],
                )
                self._hide_or_emit(self._buffer[:index], visible)
                self.saw_markup = True

                if end < 0:
                    if closer is not None:
                        # An incomplete opener is already unsafe. Enter hidden
                        # mode now and scan the remainder for its eventual close.
                        if len(self._active_closers) >= self._MAX_NESTING:
                            self.malformed = True
                        else:
                            self._active_closers.append(closer)
                            self.suppressed_blocks += 1
                        self._buffer = self._buffer[prefix_end:]
                        continue
                    if self._active_closers and self._active_closers[-1] != marker:
                        self.malformed = True
                        self._buffer = ""
                        break
                    if marker == self._SPECIAL_CLOSE:
                        self.malformed = True
                        self.suppressed_chars += len(self._buffer) - prefix_end
                        self._buffer = ""
                        break
                    # Keep only a fixed canonical prefix while waiting for ``>``.
                    self._buffer = marker + " "
                    break

                self._buffer = self._buffer[end:]
                if closer is not None:
                    # Depth/closer tracking prevents nested reasoning from
                    # becoming visible after the first inner close.
                    if len(self._active_closers) >= self._MAX_NESTING:
                        self.malformed = True
                    else:
                        self._active_closers.append(closer)
                        self.suppressed_blocks += 1
                elif self._active_closers:
                    if self._active_closers[-1] == marker:
                        self._active_closers.pop()
                    else:
                        # Mismatched control markup is never recoverable into a
                        # trustworthy spoken answer during this invocation.
                        self.malformed = True
                # A stray closing marker outside reasoning is discarded.
                continue

            start, suffix = self._partial_marker_suffix(self._buffer)
            self._hide_or_emit(self._buffer[:start], visible)
            self._buffer = suffix
            break
        return visible

    def finish(self) -> list[str]:
        if self._active_closers:
            # Everything after an opener remains reasoning until all nested
            # blocks close; max-token cuts therefore fail closed.
            self.suppressed_chars += len(self._buffer)
            self._buffer = ""
            self._active_closers.clear()
            self.malformed = True
            return []
        if self._buffer:
            partial_at, _canonical = self._partial_marker_suffix(self._buffer)
            incomplete_html = any(
                (found := self._marker_at(self._buffer, marker)) is not None
                and found[1] < 0
                for marker in (self._HTML_OPEN, self._HTML_CLOSE)
            )
            if partial_at == 0 or incomplete_html:
                # Never turn a truncated control opener (e.g. ``< thi``) into
                # speech at the native stream boundary.
                self._buffer = ""
                self.saw_markup = True
                self.malformed = True
                return []
        tail = self._buffer
        self._buffer = ""
        visible: list[str] = []
        self._emit(tail, visible)
        return visible


def _without_reasoning_tags(
    text: object,
    *,
    initial_reasoning_closer: str | None = None,
) -> tuple[str, _ReasoningTagFilter]:
    parser = _ReasoningTagFilter(
        initial_reasoning_closer=initial_reasoning_closer,
    )
    visible = parser.feed(text)
    visible.extend(parser.finish())
    return "".join(visible), parser


def _require_safe_reasoning_output(parser: _ReasoningTagFilter) -> None:
    if parser.malformed:
        raise LLMProviderOutputError(
            "llama.cpp reasoning output ended in malformed/unclosed markup"
        )
    if parser.saw_markup and not parser.safe_nonspace:
        raise LLMProviderOutputError(
            "llama.cpp reasoning output contained no final answer"
        )


class LlamaCppLLM:
    """On-device LLM via llama.cpp (the mobile/no-Ollama path).

    Ollama is a desktop daemon and does not exist on Android/iOS, so on phone we
    run a quantized GGUF directly through ``llama-cpp-python`` -- same process,
    no server. Use a compact GGUF such as MiniCPM5-1B: on a 12 GB
    phone with no dedicated VRAM, that small model is the intelligence tier.

    The ``llama_cpp`` package is imported lazily so the runtime and tests work
    without the native lib. Pass ``client`` to inject a fake in tests.

    Production uses the exact CPU-only binding audited in ADR-0030. A task-owned
    cancellation event reaches native prompt evaluation/generation and the
    interrupted context is cleared before reuse. Model construction has no
    context yet and therefore remains outside that cooperative abort boundary.

    Vision: llama.cpp needs a separate projector (``mmproj``) + a chat handler
    for image input. When ``images`` are passed we format multimodal chat
    content; without a vision-capable build the underlying model ignores them.
    """

    def __init__(
        self,
        model_path: str,
        *,
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
        n_threads_batch: Optional[int] = None,
        n_gpu_layers: int = 0,
        chat_format: Optional[str] = None,
        options: Optional[dict] = None,
        type_k=None,
        type_v=None,
        think: Optional[bool] = False,
        client=None,
        abort_api: Optional[_LlamaCppAbortAPI] = None,
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        thread_pair = resolve_llamacpp_thread_pair(n_threads, n_threads_batch)
        self.n_threads = thread_pair.n_threads
        self.n_threads_batch = thread_pair.n_threads_batch
        self._available_cpus = thread_pair.available_cpus
        self._n_threads_source = "auto" if n_threads is None else "explicit"
        if n_threads_batch is not None:
            self._n_threads_batch_source = "explicit"
        elif n_threads is None:
            self._n_threads_batch_source = "auto"
        else:
            self._n_threads_batch_source = "paired"
        headroom_ceiling = max(
            1,
            self._available_cpus - DEFAULT_LLM_THREAD_RESERVE,
        )
        if max(self.n_threads, self.n_threads_batch) > headroom_ceiling:
            _llamacpp_log.warning(
                "llama.cpp explicit thread pair %d/%d exceeds the %d-thread "
                "voice-headroom ceiling for %d CPUs visible to the calling thread",
                self.n_threads,
                self.n_threads_batch,
                headroom_ceiling,
                self._available_cpus,
            )
        self.n_gpu_layers = n_gpu_layers
        self.chat_format = chat_format
        # Same voice-path contract as OllamaLLM: reasoning is explicitly off by
        # default. True opts into deliberate reasoning; None is accepted only
        # for non-reasoning templates because an implicit prompt prefill cannot
        # be made safe at the output seam. Raw markup is always suppressed.
        self._think = think
        # KV-cache quantization (llm-inference-9): q8_0 roughly halves the KV
        # memory vs the f16 default at near-lossless quality, so a longer context
        # fits in a phone's RAM budget. None -> llama.cpp's default. n_ctx (above)
        # already bounds the context per profile.
        self.type_k = _resolve_kv_cache_type(type_k)
        self.type_v = _resolve_kv_cache_type(type_v)
        # Options are translated to llama.cpp's request vocabulary ONCE here, so
        # an Ollama-shaped profile still bounds on-device output (llm-inference-3).
        self._options = _normalize_llamacpp_options(options)
        self._client = client
        self._client_injected = client is not None
        self._abort_api = abort_api
        self._abort_callback: object | None = None
        self._abort_registered_client: object | None = None
        self._active_cancel_event: object | None = None
        self._context_poisoned = False
        self.last_reasoning_chars = 0
        self.last_reasoning_blocks = 0
        self.last_finish_reason: str | None = None
        # Set only after this instance installs a supported template handler
        # whose prompt actually ends inside a reasoning block. Merely asking a
        # non-reasoning/injected model for ``think=True`` must not hide its answer.
        self._reasoning_prefill_closer: str | None = None
        # A single llama.cpp context can't run two inferences at once, and the
        # lazy build must not double-construct. This serializes both: the startup
        # warm pass and a concurrent first live turn (or two tasks) share one
        # context safely instead of racing into the native lib.
        self._lock = threading.Lock()

    @staticmethod
    def _context_cancel_event() -> object | None:
        return capability_context.get().get("cancel_event")

    def _acquire_model_lock(self, cancel_event: object | None) -> None:
        if cancel_event is None:
            self._lock.acquire()
            return
        while True:
            if _llamacpp_cancelled(cancel_event):
                raise LLMCallCancelled("llama.cpp call cancelled while waiting")
            if self._lock.acquire(timeout=0.01):
                if _llamacpp_cancelled(cancel_event):
                    self._lock.release()
                    raise LLMCallCancelled(
                        "llama.cpp call cancelled at model-lock admission"
                    )
                return

    def _configure_chat_template(self, client: object) -> str | None:
        """Apply MiniCPM's supported think mode to the selected GGUF handler.

        v0.3.33's direct ``create_chat_completion`` signature cannot accept
        chat-template kwargs. Its bundled HTTP server implements them by
        wrapping the effective handler after model construction; mirror that
        audited mechanism for the in-process phone path.
        """

        metadata = getattr(client, "metadata", None)
        template = (
            metadata.get("tokenizer.chat_template")
            if isinstance(metadata, Mapping)
            else None
        )
        if not isinstance(template, str) or not template:
            return None  # A non-template/non-reasoning GGUF needs no control.
        supports_flag = "enable_thinking" in template
        lowered_template = template.lower()
        has_html_reasoning = bool(re.search(r"<\s*think\b", lowered_template))
        has_special_reasoning = "<|thought_begin|>" in lowered_template
        has_reasoning_markup = has_html_reasoning or has_special_reasoning
        if self._think is None:
            if has_reasoning_markup or supports_flag:
                raise RuntimeError(
                    "GGUF reasoning templates require an explicit think boolean"
                )
            return None
        if not supports_flag:
            if has_reasoning_markup:
                raise RuntimeError(
                    "GGUF reasoning template lacks the required "
                    "enable_thinking control"
                )
            return None

        selected_format = getattr(client, "chat_format", None)
        if callable(getattr(client, "chat_handler", None)) or (
            selected_format != "chat_template.default"
        ):
            raise RuntimeError(
                "GGUF hybrid reasoning requires its embedded "
                "chat_template.default handler"
            )
        handlers = getattr(client, "_chat_handlers", None)
        base_handler = (
            handlers.get("chat_template.default")
            if isinstance(handlers, Mapping)
            else None
        )
        if not callable(base_handler):
            raise RuntimeError(
                "llama.cpp could not resolve the GGUF chat-template handler"
            )

        enable_thinking = bool(self._think)

        def chat_handler_with_think_mode(*args, **kwargs):
            return base_handler(
                *args,
                **{"enable_thinking": enable_thinking, **kwargs},
            )

        try:
            setattr(client, "chat_handler", chat_handler_with_think_mode)
        except Exception as exc:
            raise RuntimeError(
                "llama.cpp could not install the voice think-mode handler"
            ) from exc
        if not enable_thinking:
            return None
        if has_html_reasoning:
            return _ReasoningTagFilter._HTML_CLOSE
        if has_special_reasoning:
            return _ReasoningTagFilter._SPECIAL_CLOSE
        return None

    def _ensure(self, cancel_event: object | None = None):
        if self._client is None:
            self._acquire_model_lock(cancel_event)
            try:
                if self._client is None:  # double-checked: build exactly once
                    if self.n_gpu_layers != 0:
                        raise RuntimeError(
                            "verified llama.cpp cancellation requires CPU-only "
                            "n_gpu_layers=0"
                        )
                    import llama_cpp  # lazy optional dependency

                    self._abort_api = verify_llamacpp_abort_runtime(llama_cpp)
                    Llama = llama_cpp.Llama

                    _llamacpp_log.info(
                        "llama.cpp CPU threads generation=%d (%s) batch=%d (%s) "
                        "available=%d",
                        self.n_threads,
                        self._n_threads_source,
                        self.n_threads_batch,
                        self._n_threads_batch_source,
                        self._available_cpus,
                    )
                    kwargs = dict(
                        model_path=self.model_path,
                        n_ctx=self.n_ctx,
                        n_gpu_layers=self.n_gpu_layers,
                        verbose=False,
                    )
                    if self.n_threads:
                        kwargs["n_threads"] = self.n_threads
                    if self.n_threads_batch:
                        kwargs["n_threads_batch"] = self.n_threads_batch
                    if self.chat_format:
                        kwargs["chat_format"] = self.chat_format
                    if self.type_k is not None:
                        kwargs["type_k"] = self.type_k
                    if self.type_v is not None:
                        kwargs["type_v"] = self.type_v
                    try:
                        candidate = Llama(**kwargs)
                    except TypeError:
                        # An old llama-cpp-python without the type_k/type_v kwargs
                        # would hard-crash the first on-device turn. Degrade to the
                        # f16 KV default instead (matches _resolve_kv_cache_type's
                        # fail-soft intent). Only rescues the KV-quant case -- any
                        # other TypeError re-raises on the retry.
                        if "type_k" not in kwargs and "type_v" not in kwargs:
                            raise
                        kwargs.pop("type_k", None)
                        kwargs.pop("type_v", None)
                        _llamacpp_log.warning(
                            "llama.cpp build lacks KV-cache-quant kwargs; using the f16 default"
                        )
                        candidate = Llama(**kwargs)
                    try:
                        reasoning_prefill_closer = self._configure_chat_template(
                            candidate
                        )
                    except BaseException:
                        closer = getattr(candidate, "close", None)
                        if callable(closer):
                            try:
                                closer()
                            except Exception:
                                _llamacpp_log.debug(
                                    "failed to close llama.cpp after chat-template setup error",
                                    exc_info=True,
                                )
                        raise
                    self._reasoning_prefill_closer = reasoning_prefill_closer
                    self._client = candidate
            finally:
                self._lock.release()
        return self._client

    def _register_abort_locked(self, client: object) -> bool:
        if self._abort_registered_client is client:
            return True
        api = self._abort_api
        if api is None:
            if self._client_injected:
                return False
            raise RuntimeError("verified llama.cpp abort API was not initialized")
        if self.n_gpu_layers != 0:
            raise RuntimeError(
                "llama.cpp native abort is CPU-only; n_gpu_layers must be 0"
            )
        ctx = _llamacpp_native_context(client)
        reset = getattr(client, "reset", None)
        if ctx is None or not callable(reset):
            if self._client_injected:
                return False
            raise RuntimeError("llama.cpp client lacks ctx/reset cancellation seams")

        owner_ref = weakref.ref(self)

        def should_abort(_data) -> bool:
            owner = owner_ref()
            return bool(
                owner is not None
                and _llamacpp_cancelled(owner._active_cancel_event)
            )

        try:
            callback = api.callback_type(should_abort)
            api.set_callback(ctx, callback, None)
        except Exception as exc:
            raise RuntimeError("failed to install llama.cpp CPU abort callback") from exc
        # ctypes does not keep this callback alive for us. Retain it for exactly
        # as long as the native context can call it; the active event changes
        # only while the same inference lock is owned.
        self._abort_callback = callback
        self._abort_registered_client = client
        return True

    def _begin_inference(self, cancel_event: object | None):
        client = self._ensure(cancel_event)
        self._acquire_model_lock(cancel_event)
        try:
            if self._context_poisoned:
                raise RuntimeError(
                    "llama.cpp context is poisoned after failed abort cleanup; restart required"
                )
            self._register_abort_locked(client)
            if _llamacpp_cancelled(cancel_event):
                raise LLMCallCancelled("llama.cpp call cancelled before inference")
            self.last_reasoning_chars = 0
            self.last_reasoning_blocks = 0
            self.last_finish_reason = None
            self._active_cancel_event = cancel_event
            return client
        except BaseException:
            self._active_cancel_event = None
            self._lock.release()
            raise

    def _record_reasoning_filter(self, parser: _ReasoningTagFilter) -> None:
        self.last_reasoning_chars = parser.suppressed_chars
        self.last_reasoning_blocks = parser.suppressed_blocks
        if parser.suppressed_blocks:
            _llamacpp_log.info(
                "llama.cpp suppressed %d reasoning block(s), %d chars",
                parser.suppressed_blocks,
                parser.suppressed_chars,
            )

    def _recover_cancelled_locked(self, client: object) -> None:
        """Clear partial ubatches before the shared context is reusable."""

        api = self._abort_api
        reset = getattr(client, "reset", None)
        try:
            if api is not None and self._abort_registered_client is client:
                ctx = _llamacpp_native_context(client)
                if ctx is None:
                    raise RuntimeError("llama.cpp client lost its native context")
                memory = api.get_memory(ctx)
                if memory is None:
                    raise RuntimeError("llama.cpp context returned no memory handle")
                api.clear_memory(memory, True)
            if callable(reset):
                reset()
            elif not self._client_injected:
                raise RuntimeError("llama.cpp client has no reset()")
        except (Exception, asyncio.CancelledError):
            self._context_poisoned = True
            _llamacpp_log.exception(
                "llama.cpp abort cleanup failed; context is now fail-closed"
            )

    def _finish_inference(self, client: object, cancel_event: object | None) -> bool:
        cancelled = _llamacpp_cancelled(cancel_event)
        try:
            if cancelled:
                self._recover_cancelled_locked(client)
        finally:
            self._active_cancel_event = None
            self._lock.release()
        return cancelled

    def _messages(
        self, prompt: str, system: Optional[str], images: Optional[Sequence[ImageInput]],
        history: Optional[Sequence[HistoryTurn]] = None,
    ) -> list[dict]:
        msgs: list[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(_history_messages(history))  # prior turns between system + current
        if images:
            content: list[dict] = [{"type": "text", "text": prompt}]
            for img in images:
                url = img if isinstance(img, str) else _to_data_uri(img)
                content.append({"type": "image_url", "image_url": {"url": url}})
            msgs.append({"role": "user", "content": content})
        else:
            msgs.append({"role": "user", "content": prompt})
        return msgs

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[ImageInput]] = None,
        history: Optional[Sequence[HistoryTurn]] = None,
    ) -> str:
        cancel_event = self._context_cancel_event()
        client = self._begin_inference(cancel_event)
        content: str | None = None
        error: BaseException | None = None
        try:
            resp = client.create_chat_completion(
                messages=self._messages(prompt, system, images, history),
                stream=False, **self._options
            )
            choice = resp["choices"][0]
            finish_reason = choice.get("finish_reason")
            if finish_reason is not None:
                self.last_finish_reason = str(finish_reason)
            raw_content = choice["message"]["content"]
            content, parser = _without_reasoning_tags(
                raw_content,
                initial_reasoning_closer=self._reasoning_prefill_closer,
            )
            self._record_reasoning_filter(parser)
            _require_safe_reasoning_output(parser)
        except BaseException as exc:
            error = exc
        finally:
            cancelled = self._finish_inference(client, cancel_event)

        if cancelled and (
            error is None
            or isinstance(error, (Exception, asyncio.CancelledError))
        ):
            cancelled_error = LLMCallCancelled(
                "llama.cpp call cancelled during inference"
            )
            if error is not None:
                raise cancelled_error from error
            raise cancelled_error
        if error is not None:
            raise error.with_traceback(error.__traceback__)
        if content is None:
            raise RuntimeError("llama.cpp returned no chat-completion content")
        return content

    def stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[ImageInput]] = None,
        history: Optional[Sequence[HistoryTurn]] = None,
    ) -> Iterator[str]:
        # Capture the ContextVar now: callers may construct the iterator in a
        # capability context and consume it later from a worker thread.
        cancel_event = self._context_cancel_event()
        return self._stream_captured(
            prompt,
            system=system,
            images=images,
            history=history,
            cancel_event=cancel_event,
        )

    def _stream_captured(
        self,
        prompt: str,
        *,
        system: Optional[str],
        images: Optional[Sequence[ImageInput]],
        history: Optional[Sequence[HistoryTurn]],
        cancel_event: object | None,
    ) -> Iterator[str]:
        client = self._begin_inference(cancel_event)
        native_stream = None
        parser = _ReasoningTagFilter(
            initial_reasoning_closer=self._reasoning_prefill_closer,
        )
        filter_recorded = False
        error: BaseException | None = None
        try:
            native_stream = client.create_chat_completion(
                messages=self._messages(prompt, system, images, history),
                stream=True, **self._options
            )
            iterator = iter(native_stream)
            while True:
                if _llamacpp_cancelled(cancel_event):
                    raise LLMCallCancelled("llama.cpp stream cancelled")
                try:
                    chunk = next(iterator)
                except StopIteration:
                    break
                if _llamacpp_cancelled(cancel_event):
                    raise LLMCallCancelled("llama.cpp stream cancelled")
                choice = chunk["choices"][0]
                finish_reason = choice.get("finish_reason")
                if finish_reason is not None:
                    self.last_finish_reason = str(finish_reason)
                piece = choice.get("delta", {}).get("content")
                if piece:
                    for visible in parser.feed(piece):
                        if _llamacpp_cancelled(cancel_event):
                            raise LLMCallCancelled("llama.cpp stream cancelled")
                        if visible:
                            yield visible
            if _llamacpp_cancelled(cancel_event):
                raise LLMCallCancelled("llama.cpp stream cancelled at completion")
            tail = parser.finish()
            self._record_reasoning_filter(parser)
            filter_recorded = True
            _require_safe_reasoning_output(parser)
            for visible in tail:
                if _llamacpp_cancelled(cancel_event):
                    raise LLMCallCancelled("llama.cpp stream cancelled at completion")
                if visible:
                    yield visible
        except BaseException as exc:
            error = exc
        finally:
            try:
                if not filter_recorded:
                    parser.finish()
                    self._record_reasoning_filter(parser)
                closer = getattr(native_stream, "close", None)
                if callable(closer):
                    try:
                        closer()
                    except (Exception, asyncio.CancelledError):
                        _llamacpp_log.debug(
                            "llama.cpp stream close failed during teardown",
                            exc_info=True,
                        )
            finally:
                # Even control-flow BaseExceptions from a foreign iterator's
                # close() must not strand the sole native-context lock.
                cancelled = self._finish_inference(client, cancel_event)

        if cancelled and (
            error is None
            or isinstance(error, (Exception, asyncio.CancelledError))
        ):
            if isinstance(error, LLMCallCancelled):
                raise error.with_traceback(error.__traceback__)
            cancelled_error = LLMCallCancelled(
                "llama.cpp stream cancelled during native inference"
            )
            if error is not None:
                raise cancelled_error from error
            raise cancelled_error
        if error is not None:
            raise error.with_traceback(error.__traceback__)


def _redact_messages_for_egress(messages: list[dict]) -> list[dict]:
    """Scrub high-confidence PII (cards/SSN/keys/email/phone/secrets) from outbound
    cloud messages -- a §9.7 last-line net INDEPENDENT of the regex sensitivity
    classifier. If a credit card / SSN / API key slips past PRIVATE classification
    (garbled ASR, PII phrased outside the pattern set) and a turn is sent to a
    third-party cloud, this still removes it. Conservative (``redact_pii`` only
    touches Luhn-checked cards, SSNs, known key formats, etc.), so ordinary queries
    are untouched. Applied ONLY to cloud egress -- never to a local model. Text-only
    redaction; image parts (data URIs) pass through (vision egress is gated upstream
    by sensitivity + the local-only captioning rule). Applied to every cloud-chain
    member (OpenAICompatLLM with the flag set); the HedgeLLM local safety-net member
    (Ollama / llama.cpp) is never redacted."""
    from always_on_agent.untrusted import redact_pii  # stdlib-only; core->aoa is allowed

    # force=True: the §9.7 outbound-cloud net is mandatory and must NOT share the
    # SPEAKER_DISABLE_REDACT kill-switch with the durable-record redactor -- an
    # operator disabling local-record scrubbing must never silently send PII to a
    # third-party cloud.
    out: list[dict] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            out.append({**m, "content": redact_pii(content, force=True)})
        elif isinstance(content, list):
            parts = [
                ({**p, "text": redact_pii(p["text"], force=True)}
                 if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str)
                 else p)
                for p in content
            ]
            out.append({**m, "content": parts})
        else:
            out.append(m)
    return out


def _openai_messages(
    prompt: str, system: Optional[str], images: Optional[Sequence[ImageInput]],
    history: Optional[Sequence[HistoryTurn]] = None,
) -> list[dict]:
    """OpenAI-style chat messages (also used by llama-server, Groq, etc.)."""
    msgs: list[dict] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.extend(_history_messages(history))  # prior turns between system + current
    if images:
        content: list[dict] = [{"type": "text", "text": prompt}]
        for img in images:
            url = img if isinstance(img, str) else _to_data_uri(img)
            content.append({"type": "image_url", "image_url": {"url": url}})
        msgs.append({"role": "user", "content": content})
    else:
        msgs.append({"role": "user", "content": prompt})
    return msgs


@dataclass(frozen=True)
class ProviderProfile:
    """Per-provider quirks layered on top of the generic OpenAI-compat shape.

    Each major cloud sharing the ``/v1/chat/completions`` endpoint still has
    small departures that, ignored, cause silent failures: Moonshot rejects
    custom temperature; Cerebras requires non-standard params via
    ``extra_body=``; DeepSeek's reasoning models stream the chain-of-thought
    in a separate ``delta.reasoning_content`` field that the generic loop
    drops on the floor; Groq's gpt-oss-120b puts reasoning in ``delta.reasoning``
    and rejects ``n != 1``. This dataclass captures those quirks declaratively
    so :class:`OpenAICompatLLM` consumes them uniformly.

    Names are looked up via :data:`PROVIDER_PROFILES` from a string tag
    stored on the cloud preset in ``config.json`` (e.g. ``"profile":
    "deepseek_reasoning"``).
    """

    name: str
    # Param keys stripped from the generic kwargs dict before calling .create().
    forbidden_params: frozenset[str] = field(default_factory=frozenset)
    # Param keys routed through ``extra_body={...}`` instead of as top-level
    # kwargs (the OpenAI SDK rejects unknown top-level keys).
    extra_body_keys: frozenset[str] = field(default_factory=frozenset)
    # Name of the delta field that carries reasoning tokens (CoT), if any.
    # ``"reasoning_content"`` for DeepSeek V4-Pro; ``"reasoning"`` for Groq
    # gpt-oss-120b; ``None`` for plain chat models.
    reasoning_field: Optional[str] = None
    # When True, reasoning tokens are observed (for metrics) but NOT yielded
    # to the consumer -- the voice assistant shouldn't speak the CoT.
    suppress_reasoning_in_stream: bool = True
    # Hard cap on ``max_tokens`` enforced before sending (Cerebras free tier
    # rejects > 8192).
    max_tokens_cap: Optional[int] = None


# Pre-defined profiles for the cloud providers we ship presets for.
# ``"openai_compat"`` is the safe default for any unrecognized endpoint.
PROVIDER_PROFILES: dict[str, "ProviderProfile"] = {
    "openai_compat": ProviderProfile(name="openai_compat"),
    # Cerebras: free tier caps max_tokens at 8192. Non-OpenAI params (e.g. GLM's
    # ``clear_thinking``, ``reasoning_effort``) must go in extra_body=.
    "cerebras": ProviderProfile(
        name="cerebras",
        extra_body_keys=frozenset({"clear_thinking", "reasoning_effort"}),
        max_tokens_cap=8192,
    ),
    # Groq: ``n`` is fixed at 1 (any other value 400s). gpt-oss-120b streams
    # reasoning in delta.reasoning (separate from delta.content).
    "groq": ProviderProfile(
        name="groq",
        forbidden_params=frozenset({"n"}),
        reasoning_field="reasoning",
    ),
    # DeepSeek non-reasoning models (V4-Flash) -- plain chat.
    "deepseek": ProviderProfile(name="deepseek"),
    # DeepSeek V4-Pro (reasoning): streams delta.reasoning_content ahead of
    # delta.content. The API also rejects echoing reasoning_content back on
    # the next turn -- callers must strip it from prior assistant messages.
    "deepseek_reasoning": ProviderProfile(
        name="deepseek_reasoning",
        reasoning_field="reasoning_content",
    ),
    # Moonshot Kimi: temperature, top_p, n are server-fixed (any value 400s).
    "moonshot": ProviderProfile(
        name="moonshot",
        forbidden_params=frozenset({"temperature", "top_p", "n"}),
    ),
}


class OpenAICompatLLM:
    """Streaming LLM over any OpenAI-compatible ``/v1/chat/completions`` endpoint.

    One client covers Groq, Together, Fireworks, SambaNova, Cerebras, OpenAI and
    a local llama.cpp ``llama-server`` -- they differ only in ``base_url``,
    ``model`` and api key. This is the optional "intelligence from a streaming
    source" tier: it is built only when ``llm.cloud.enabled`` is set, so the
    fully-local default is preserved. Rank providers by time-to-first-token.

    Provider quirks are layered via ``profile`` (a :class:`ProviderProfile`
    instance or a string key into :data:`PROVIDER_PROFILES`): forbidden
    params are stripped, extra-body keys are routed correctly, and
    reasoning-field streaming (DeepSeek ``reasoning_content``, Groq
    ``reasoning``) is consumed and metric-tracked without being yielded to
    the speaker (the assistant shouldn't speak the CoT).

    The ``openai`` package is imported lazily so the runtime and test suite work
    without it; pass ``client`` to inject a fake. ``api_key_env`` names the env
    var holding the key (so secrets never live in config.json).
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_env: Optional[str] = None,
        timeout: float = 30.0,
        max_tokens: Optional[int] = None,
        options: Optional[dict] = None,
        profile: "ProviderProfile | str | None" = None,
        client=None,
        redact_pii_outbound: bool = False,
    ):
        self.model = model
        self._base_url = base_url
        # §9.7 last-line net: when True (set by the cloud-client factory), scrub
        # high-confidence PII from the outbound prompt before it leaves the device.
        # Default False so a LOCAL OpenAICompat endpoint (llama-server) and existing
        # callers/tests are byte-identical; only cloud providers enable it.
        self._redact_pii_outbound = bool(redact_pii_outbound)
        self._api_key = api_key or (os.environ.get(api_key_env) if api_key_env else None)
        # Socket/read timeout (seconds) handed to the OpenAI client. A small
        # value (BR1) reaps a losing cloud worker still blocked in the first-
        # token read fast -- the HTTP hard-close in stream() is deterministic
        # only after tokens flow, so a pre-first-token loser otherwise holds the
        # socket + billing until this timeout. 30.0 keeps the prior behaviour.
        self._timeout = timeout
        # Optional per-turn output ceiling (BR4). Injected into the merged
        # request kwargs BEFORE the profile max_tokens cap so the profile cap
        # stays authoritative via min() composition (e.g. Cerebras free tier).
        self._max_tokens = max_tokens
        self._options = dict(options) if options else {}
        self._client = client
        if profile is None:
            self.profile = PROVIDER_PROFILES["openai_compat"]
        elif isinstance(profile, str):
            self.profile = PROVIDER_PROFILES.get(profile, PROVIDER_PROFILES["openai_compat"])
        else:
            self.profile = profile
        # Last-call observability: bytes seen on the reasoning field, exposed
        # so HedgeLLM / capabilities can log a "thought N chars before
        # answering" metric without inspecting the raw stream.
        self.last_reasoning_chars = 0

    def _ensure(self):
        if self._client is None:
            from openai import OpenAI  # lazy

            self._client = OpenAI(
                base_url=self._base_url,
                api_key=self._api_key or "not-needed",
                timeout=self._timeout,
            )
        return self._client

    def _create_kwargs(self, prompt, system, images, *, stream: bool, history=None) -> dict:
        messages = _openai_messages(prompt, system, images, history)
        if self._redact_pii_outbound:
            messages = _redact_messages_for_egress(messages)
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }
        # Merge caller-supplied options; the profile may then strip / reroute.
        merged: dict = dict(self._options)
        extra_body: dict = {}
        for key in list(merged):
            if key in self.profile.forbidden_params:
                merged.pop(key, None)
            elif key in self.profile.extra_body_keys:
                extra_body[key] = merged.pop(key)
        # Per-turn output ceiling (BR4): inject the configured max_tokens into
        # merged BEFORE the profile-cap block below so the profile cap stays
        # authoritative -- the cap's min()-style composition then keeps whichever
        # is smaller (e.g. a 100-token turn ceiling under an 8192 Cerebras cap
        # stays 100; a 20000 ceiling is clamped to 8192). A caller-supplied
        # options["max_tokens"] still wins (it's already in merged).
        if self._max_tokens is not None and merged.get("max_tokens") is None:
            merged["max_tokens"] = self._max_tokens
        # max_tokens cap (Cerebras free tier).
        cap = self.profile.max_tokens_cap
        if cap is not None:
            requested = merged.get("max_tokens")
            if requested is None or int(requested) > cap:
                merged["max_tokens"] = cap
        kwargs.update(merged)
        if extra_body:
            # Merge with any caller-supplied extra_body rather than clobbering.
            existing = kwargs.get("extra_body") or {}
            existing.update(extra_body)
            kwargs["extra_body"] = existing
        return kwargs

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[ImageInput]] = None,
        history: Optional[Sequence[HistoryTurn]] = None,
    ) -> str:
        resp = self._ensure().chat.completions.create(
            **self._create_kwargs(prompt, system, images, stream=False, history=history)
        )
        return resp.choices[0].message.content or ""

    def stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[ImageInput]] = None,
        history: Optional[Sequence[HistoryTurn]] = None,
    ) -> Iterator[str]:
        self.last_reasoning_chars = 0
        reasoning_field = self.profile.reasoning_field
        suppress = self.profile.suppress_reasoning_in_stream
        # Bind the SDK Stream INSIDE the generator body so a HTTP hard-close on
        # cancel/barge-in (GeneratorExit) runs the finally below, telling the
        # provider to stop streaming + billing instead of leaking the socket
        # until GC (ported from tools/cloudchat.py:181-185). Binding here (not
        # before the loop's caller resumes) keeps a close() BEFORE the first
        # token a no-op: sdk_stream is unbound, so the getattr guard returns
        # None and finally does nothing rather than raising AttributeError (BR6).
        try:
            sdk_stream = self._ensure().chat.completions.create(
                **self._create_kwargs(prompt, system, images, stream=True, history=history)
            )
            for chunk in sdk_stream:
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = choices[0].delta
                # Reasoning-channel tokens (DeepSeek reasoning_content / Groq
                # gpt-oss reasoning). Count for metrics; yield only if not
                # suppressed (default: suppressed -- assistant shouldn't speak CoT).
                if reasoning_field:
                    reasoning_piece = getattr(delta, reasoning_field, None)
                    if reasoning_piece:
                        self.last_reasoning_chars += len(reasoning_piece)
                        if not suppress:
                            yield reasoning_piece
                piece = getattr(delta, "content", None)
                if piece:
                    yield piece
        finally:
            # Hard-close the HTTP stream on natural exhaustion AND on early
            # consumer close (cancel). Guarded so a pre-first-token close (where
            # .create() never returned, sdk_stream unbound) is a NO-OP, and so a
            # fake/iterator stream without .close() doesn't crash the worker.
            closer = getattr(locals().get("sdk_stream"), "close", None)
            if closer is not None:
                try:
                    closer()
                except Exception:
                    pass


class HedgeLLM:
    """Race a local LLM against an optional cloud chain for the lowest latency.

    Strategies (all keep local as the safety net -- any cloud error/timeout
    falls through to the next cloud, finally to local, honoring the fully-
    local requirement):

    - ``hedge`` (default): start local now; if it produces no token within
      ``hedge_delay_ms``, also start the first cloud and stream whichever
      yields the FIRST token. Caps cloud spend/exposure while still racing
      when local is slow. ``hedge_delay_ms=0`` makes it a full race.
    - ``fallback``: start the first cloud with a ``ttft_deadline_ms`` first-
      token deadline; on timeout/error, advance to the next cloud; after the
      chain is exhausted, fall back to local with no deadline.

    The ``cloud`` parameter accepts either a single :class:`LLMClient` (back-
    compat) or a list of them (a failover chain). When any cloud in the chain
    finishes without producing a token (error or empty stream), HedgeLLM
    advances to the next cloud and races it against local with the same
    rules.

    Cancellation: losing workers are signalled to stop between tokens; the
    brain's ``cancel_event`` still cuts the winner's stream in the
    capability layer, so barge-in works unchanged.
    """

    # How long the final-drain ``q.get()`` waits between tokens from a winner
    # that has already produced its first token before giving up and ending the
    # stream cleanly. Without a bound, a winner whose connection stalls
    # mid-stream (TCP black-hole) would wedge the whole turn forever. Generous
    # enough not to truncate a healthy-but-slow generation; the per-token gap on
    # any working stream is far below this.
    DRAIN_IDLE_TIMEOUT = 30.0
    # Wall-clock budget for the PRE-first-token winner-selection wait, derived
    # from ``ttft_deadline`` (see ``_winner_select_budget``). Distinct from
    # DRAIN_IDLE_TIMEOUT, which is the POST-first-token between-token bound: this
    # one caps the time spent waiting for the *first* token from *any* source so
    # a hung source that never yields, errors, or completes cannot wedge the turn
    # before a single token is produced. The hedge code only ever waits on
    # ``deadline`` (hedge_delay / ttft_deadline) when a worker is still live, and
    # in the hedge strategy that becomes ``inf`` once everything is launched -- so
    # without this budget a stalled local source (notably the in-process
    # ``LlamaCppLLM`` tier, whose cooperative CPU abort cannot recover a native
    # deadlock and has no socket/read timeout) would block the
    # ``q.get(timeout=None)`` forever. This wall-clock budget remains that hard
    # outer bound for the llama.cpp tier. The multiplier is generous: each
    # launched source may
    # legitimately consume up to ~``ttft_deadline`` of pre-first-token wait
    # (a fallback chain retires each slow cloud only at its deadline), so the
    # budget scales with the chain length; it must never be unbounded.
    WINNER_SELECT_TTFT_BUDGET_MULT = 4
    # Floor for the winner-selection budget so a tiny configured ttft_deadline
    # (or a zero hedge_delay) still leaves a sane wall-clock window before a hung
    # source is reaped. Capped at a real-time voice-turn budget: this is the max
    # wait for a FIRST token before we give up and end the turn, so it must stay
    # within a conversational latency budget (a 30s floor far exceeds any voice
    # turn and would wedge the turn on a hung source for half a minute).
    WINNER_SELECT_BUDGET_FLOOR = 10.0
    # Total budget the generator's cleanup spends joining worker threads once
    # they have been told to stop. A live worker stops at its next token
    # boundary (sub-ms once signalled) or when its socket read timeout fires;
    # this short bound reaps the common fast-exit case without wedging the turn
    # on a loser still blocked in an uncancellable pre-first-token read (those
    # are daemon threads that their socket timeout reaps shortly after). It must
    # never be unbounded.
    WORKER_JOIN_TIMEOUT = 0.5

    def __init__(
        self,
        *,
        local: "LLMClient",
        cloud: "Optional[LLMClient | Sequence[LLMClient]]",
        strategy: str = "hedge",
        hedge_delay_ms: int = 150,
        ttft_deadline_ms: int = 1200,
    ):
        self.local = local
        if cloud is None:
            self._clouds: list["LLMClient"] = []
        elif isinstance(cloud, (list, tuple)):
            self._clouds = [c for c in cloud if c is not None]
        else:
            self._clouds = [cloud]
        self.strategy = strategy
        self.hedge_delay = max(0.0, hedge_delay_ms / 1000.0)
        self.ttft_deadline = max(0.0, ttft_deadline_ms / 1000.0)
        # Egress receipt (provenance): which source served the LAST stream --
        # "local" or "cloud_<i>" (the chain index), or None before any call. Lets a
        # caller record/surface whether an answer actually came from the device or a
        # cloud provider (HedgeLLM races, so the winner isn't knowable a priori).
        self.last_source: Optional[str] = None

    @property
    def cloud(self) -> "Optional[LLMClient]":
        """Back-compat: the first cloud in the chain (or ``None``)."""
        return self._clouds[0] if self._clouds else None

    @property
    def clouds(self) -> list["LLMClient"]:
        """The full cloud failover chain (empty list when no cloud is set)."""
        return list(self._clouds)

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[ImageInput]] = None,
        history: Optional[Sequence[HistoryTurn]] = None,
    ) -> str:
        return "".join(
            self.stream(prompt, system=system, images=images, history=history)
        ).strip()

    @staticmethod
    def _is_cross_thread_cancel_safe(stream: object) -> bool:
        """Return whether ``stream.cancel()`` may run from the coordinator.

        The marker is deliberately nominal: an incidental ``cancel`` method on
        a third-party iterator does not advertise thread safety or nonblocking
        behavior. Known-safe implementations must opt in explicitly.
        """
        return (
            getattr(stream, "_cross_thread_cancel_safe", False) is True
            and callable(getattr(stream, "cancel", None))
        )

    @staticmethod
    def _worker(
        client,
        tag,
        prompt,
        system,
        images,
        history,
        q,
        stop,
        context,
        active_streams,
        cancellable_tags,
        active_streams_lock,
    ) -> None:
        # Hold the underlying stream so a stop can close it at the next token
        # boundary -- that propagates GeneratorExit into the client's stream(),
        # running its ``finally`` (socket close, metric log) promptly instead
        # of leaving a half-read HTTP body dangling until GC.
        # ``history`` is forwarded ONLY when present (like ``hedge_delay_ms``), so
        # a wrapped client / test fake that predates the multi-turn param is still
        # called with the byte-identical single-turn signature by default.
        context_token = capability_context.set(context)
        stream = None
        try:
            if stop.is_set():
                q.put((tag, "done", None))
                return
            kw: dict = {"system": system, "images": images}
            if history:
                kw["history"] = history
            stream = client.stream(prompt, **kw)
            # Only iterators that explicitly advertise a thread-safe,
            # nonblocking cancel seam may be touched by Hedge.shutdown from a
            # different thread. Arbitrary generator.close() while next() is
            # executing raises and is unsafe (notably sync Ollama pre-TTFT).
            if HedgeLLM._is_cross_thread_cancel_safe(stream):
                with active_streams_lock:
                    active_streams[tag] = stream
                    cancellable_tags.add(tag)
            if stop.is_set():
                if HedgeLLM._is_cross_thread_cancel_safe(stream):
                    stream.cancel()
                q.put((tag, "done", None))
                return
            for token in stream:
                if stop.is_set():
                    break
                q.put((tag, "tok", token))
            q.put((tag, "done", None))
        except Exception as exc:  # cloud down / rate-limited -> chain advances
            q.put((tag, "err", str(exc)))
        finally:
            with active_streams_lock:
                active_streams.pop(tag, None)
            # Best-effort close: list/iterator fakes lack .close(); a generator
            # raising on close shouldn't take the worker down.
            closer = getattr(stream, "close", None)
            if closer is not None:
                try:
                    closer()
                except Exception:
                    pass
            capability_context.reset(context_token)

    def _winner_select_budget(self) -> float:
        """Bounded wall-clock window for the pre-first-token wait.

        Derived from ``ttft_deadline`` and scaled by the number of launched
        sources (local + every cloud) so a healthy fallback chain that retires
        each slow cloud at its own ``ttft_deadline`` never trips it, while a
        hung source still gets reaped. Always finite (never ``inf``)."""
        sources = len(self._clouds) + 1  # + local
        return max(
            self.ttft_deadline * self.WINNER_SELECT_TTFT_BUDGET_MULT * sources,
            self.WINNER_SELECT_BUDGET_FLOOR,
        )

    def stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[ImageInput]] = None,
        history: Optional[Sequence[HistoryTurn]] = None,
        hedge_delay_ms: Optional[int] = None,
    ) -> Iterator[str]:
        # ``stream`` used to be a generator, which delayed this ContextVar read
        # until first iteration. Bind it at call time so an iterator created in a
        # task context keeps that turn's cancellation/routing metadata when a
        # different thread consumes it after the caller resets its ContextVar.
        turn_context = dict(capability_context.get())
        return self._stream_captured(
            prompt,
            system=system,
            images=images,
            history=history,
            hedge_delay_ms=hedge_delay_ms,
            turn_context=turn_context,
        )

    def _stream_captured(
        self,
        prompt: str,
        *,
        system: Optional[str],
        images: Optional[Sequence[ImageInput]],
        history: Optional[Sequence[HistoryTurn]],
        hedge_delay_ms: Optional[int],
        turn_context: Mapping[str, object],
    ) -> Iterator[str]:
        # Per-turn hedge-delay override (PINNED CONTRACT). ``None`` keeps the
        # constructor's ``self.hedge_delay`` so default behaviour is byte-
        # identical; an int overrides the local-vs-cloud start gap for THIS turn
        # only (the hook for dynamic hedge timing from the routing layer). Only
        # the ``hedge`` strategy uses a hedge delay; ``fallback`` is unaffected.
        hedge_delay = (
            self.hedge_delay
            if hedge_delay_ms is None
            else max(0.0, hedge_delay_ms / 1000.0)
        )
        if not self._clouds:
            self.last_source = "local"  # egress receipt: served on-device, no cloud
            local_kw: dict = {"system": system, "images": images}
            if history:
                local_kw["history"] = history
            context_token = capability_context.set(turn_context)
            try:
                yield from self.local.stream(prompt, **local_kw)
            finally:
                capability_context.reset(context_token)
            return

        # Reset the egress receipt at the START of the race so a no-winner turn
        # (every source dead/empty) reports None rather than a stale prior source.
        self.last_source = None
        q: "queue.Queue[tuple[str, str, object]]" = queue.Queue()
        cloud_tags = [f"cloud_{i}" for i in range(len(self._clouds))]
        stops: dict[str, threading.Event] = {
            "local": threading.Event(),
            **{tag: threading.Event() for tag in cloud_tags},
        }
        clients: dict[str, "LLMClient"] = {
            "local": self.local,
            **dict(zip(cloud_tags, self._clouds)),
        }
        started: set[str] = set()
        dead: set[str] = set()
        threads: dict[str, threading.Thread] = {}
        active_streams: dict[str, Iterator[str]] = {}
        # Sticky after registration: a worker removes its active stream before
        # owning close(), but shutdown must still wait for that cleanup.
        cancellable_tags: set[str] = set()
        active_streams_lock = threading.Lock()

        def launch(tag: str) -> None:
            if tag in started:
                return
            started.add(tag)
            t = threading.Thread(
                target=self._worker,
                args=(
                    clients[tag], tag, prompt, system, images, history, q, stops[tag],
                    turn_context, active_streams, cancellable_tags,
                    active_streams_lock,
                ),
                daemon=True,
            )
            threads[tag] = t
            t.start()

        def cancel_active(tags: Optional[set[str]] = None) -> None:
            """Cancel explicitly thread-safe streams by tag."""
            with active_streams_lock:
                cancellable = [
                    stream
                    for tag, stream in active_streams.items()
                    if tags is None or tag in tags
                ]
            for stream in cancellable:
                cancel_stream = getattr(stream, "cancel", None)
                if callable(cancel_stream):
                    try:
                        cancel_stream()
                    except Exception:
                        pass

        def shutdown() -> None:
            """Signal every worker and share one bounded join budget.

            Explicitly cancellable streams unwind promptly; an arbitrary sync
            worker can outlive the join until its next token or timeout. This
            runs on natural completion and early generator close alike.
            """
            for ev in stops.values():
                ev.set()
            # Native cancellation for streams that explicitly guarantee
            # cross-thread cancellation (Ollama's async bridge). Snapshot
            # outside callbacks:
            # cancel() is nonblocking, but it may complete the worker and mutate
            # this registry immediately.
            cancel_active()
            with active_streams_lock:
                cleanup_owned = set(cancellable_tags)
            # Ollama advertises the stream contract eagerly, closing the narrow
            # race in which shutdown arrives before its worker registers the
            # returned iterator.
            cleanup_owned.update(
                tag
                for tag, client in clients.items()
                if getattr(client, "_stream_cross_thread_cancel_safe", False) is True
            )
            # A cooperative provider slot must not be released while its owned
            # request/client is still cleaning up. If an advertised provider
            # violates that contract and hangs, ADR-0021 intentionally keeps the
            # outer bulkhead slot occupied instead of spawning unbounded work.
            for tag in cleanup_owned:
                worker = threads.get(tag)
                if worker is not None and worker.is_alive():
                    worker.join()
            join_deadline = time.monotonic() + self.WORKER_JOIN_TIMEOUT
            for tag, t in threads.items():
                if tag in cleanup_owned:
                    continue
                if t.is_alive():
                    remaining = join_deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    t.join(timeout=remaining)
            # Observability for the known leak: WORKER_JOIN_TIMEOUT is below real
            # reap latency (a cloud worker can take up to ~5s; Hedge's private
            # loser-stop event is not yet bridged into LlamaCppLLM's task-owned
            # native abort), so a loser can outlive the join and leak a live
            # thread/socket. A barge-in storm accumulates them. Surface the
            # survivor count in the run bundle so the leak is visible instead of
            # silent. Best-effort: never raise
            # from cleanup (this runs in a generator finally, including on
            # GeneratorExit), so guard the whole check.
            try:
                leaked = sum(1 for t in threads.values() if t.is_alive())
                if leaked:
                    _hedge_log.warning(
                        "HedgeLLM shutdown: %d worker thread(s) still alive after "
                        "%.3fs join budget; may leak thread/socket until reaped",
                        leaked,
                        self.WORKER_JOIN_TIMEOUT,
                    )
            except Exception:
                pass

        try:
            cloud_iter = iter(cloud_tags)

            def next_cloud() -> Optional[str]:
                try:
                    return next(cloud_iter)
                except StopIteration:
                    return None

            current_cloud = next_cloud()
            if self.strategy == "fallback":
                # Cloud-first: launch the first cloud with a ttft_deadline; on
                # error/timeout, advance the chain; finally fall back to local.
                if current_cloud is not None:
                    launch(current_cloud)
                deadline = time.monotonic() + self.ttft_deadline
                local_pending = True
            else:  # hedge
                # Local-first: kick local now; after hedge_delay also launch the
                # first cloud. On cloud error/finish-without-tokens, advance the
                # chain and keep racing against local. ``hedge_delay`` is the
                # per-call override (or the constructor default when None).
                launch("local")
                deadline = time.monotonic() + hedge_delay
                local_pending = False

            winner: Optional[str] = None
            buffered: list[str] = []
            # Bounded wall-clock budget for the whole pre-first-token wait. The
            # per-iteration ``deadline`` becomes ``inf`` in hedge once every
            # source is launched (and is None-timeout in the q.get below), so a
            # source that hangs without ever yielding/erroring/completing -- e.g.
            # an in-process LlamaCppLLM native call, which has no socket timeout
            # to reap it -- would otherwise block here forever. This caps that
            # wait; on expiry we stop the workers and end the stream cleanly with
            # whatever (nothing) was produced, identical to an all-dead chain.
            select_deadline = time.monotonic() + self._winner_select_budget()

            def kick_chain() -> bool:
                """Bring up whatever is next in line. Returns True if launched."""
                nonlocal current_cloud, deadline, local_pending
                if self.strategy == "hedge":
                    if current_cloud is not None and current_cloud not in started:
                        launch(current_cloud)
                        deadline = float("inf")
                        return True
                else:  # fallback
                    if current_cloud is not None and current_cloud not in started:
                        launch(current_cloud)
                        deadline = time.monotonic() + self.ttft_deadline
                        return True
                if local_pending:
                    launch("local")
                    local_pending = False
                    deadline = float("inf")
                    return True
                return False

            while winner is None:
                live = started - dead
                if not live:
                    if not kick_chain():
                        break
                    continue
                # Wall-clock budget guard: a hung source (no token, no error, no
                # completion) must not wedge the turn pre-first-token. Reap and
                # end cleanly once the budget is spent.
                select_remaining = select_deadline - time.monotonic()
                if select_remaining <= 0:
                    break
                # Clamp the per-iteration wait to the remaining budget so an
                # ``inf`` deadline (hedge: everything launched) still wakes to
                # re-check it instead of blocking forever.
                step = (
                    select_remaining
                    if deadline == float("inf")
                    else min(max(0.0, deadline - time.monotonic()), select_remaining)
                )
                try:
                    tag, kind, val = q.get(timeout=step)
                except queue.Empty:
                    # Either the per-source deadline or the wall-clock budget
                    # elapsed. If the budget is spent, the top-of-loop guard ends
                    # the turn next iteration. Otherwise this is a per-source
                    # deadline -- hedge: kick the current cloud; fallback: retire
                    # the current cloud and advance the chain.
                    if time.monotonic() >= select_deadline:
                        continue
                    if self.strategy == "fallback" and current_cloud is not None:
                        stops[current_cloud].set()
                        cancel_active({current_cloud})
                        dead.add(current_cloud)
                        current_cloud = next_cloud()
                    kick_chain()
                    continue
                if kind == "tok":
                    # A fallback source can pass the worker's stop check just
                    # before its deadline, then enqueue after the coordinator
                    # retires it. Never resurrect that cancelled source as the
                    # winner from a late queue message.
                    if tag in dead:
                        continue
                    winner = tag
                    self.last_source = tag  # egress receipt: this source served the turn
                    buffered.append(str(val))
                else:  # this source died (error or empty stream)
                    dead.add(tag)
                    if tag == current_cloud:
                        current_cloud = next_cloud()
                        kick_chain()
                    elif tag == "local":
                        # Local crashed; nothing to do but ride the cloud chain.
                        pass

            if winner is None:
                return  # nothing produced (every source errored or was empty)
            # Stop the losers now so they stop billing/streaming promptly;
            # the winner keeps streaming and is joined in shutdown().
            for tag, ev in stops.items():
                if tag != winner:
                    ev.set()
            cancel_active(set(stops) - {winner})
            for token in buffered:
                yield token
            while True:
                try:
                    # Bounded idle wait: a winner whose connection stalls
                    # mid-stream must not wedge the turn forever. On timeout
                    # we stop the winner and end the stream cleanly with what
                    # we already delivered.
                    tag, kind, val = q.get(timeout=self.DRAIN_IDLE_TIMEOUT)
                except queue.Empty:
                    stops[winner].set()
                    cancel_active({winner})
                    break
                if tag != winner:
                    continue  # drain the loser's late tokens
                if kind == "tok":
                    yield str(val)
                else:
                    break
        finally:
            shutdown()


class SensitivityRouterLLM:
    """Dispatch ``generate``/``stream`` to one of several backing LLMs based
    on the data-sensitivity tag of the current turn.

    Each backing LLM is typically a :class:`HedgeLLM` (local + cloud
    failover chain). The selection happens at call time using a context
    selector that reads :data:`capability_context` -- a ``ContextVar`` set
    by the capability layer before invoking the LLM. This keeps the
    LLMClient protocol unchanged (no extra parameter on ``stream``) while
    letting the routing decision flow from the brain's per-turn context.
    """

    def __init__(
        self,
        chains: Mapping[str, "LLMClient"],
        selector,
        *,
        default_chain: str = "private",
    ):
        if not chains:
            raise ValueError("SensitivityRouterLLM requires at least one chain")
        if default_chain not in chains:
            raise ValueError(
                f"default_chain {default_chain!r} not in chains {sorted(chains)}"
            )
        self.chains: dict[str, "LLMClient"] = dict(chains)
        self.selector = selector
        self.default_chain = default_chain

    def _pick(self) -> "LLMClient":
        ctx = capability_context.get()
        name = self.selector.choose_chain(ctx) if self.selector is not None else self.default_chain
        return self.chains.get(name, self.chains[self.default_chain])

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[ImageInput]] = None,
        history: Optional[Sequence[HistoryTurn]] = None,
    ) -> str:
        # Pick happens eagerly so any sensitivity-driven routing is logged
        # at the call site (via _pick) rather than at first-yield.
        impl = self._pick()
        kw: dict = {"system": system, "images": images}
        if history:  # forward only when present -> default call is byte-identical
            kw["history"] = history
        return impl.generate(prompt, **kw)

    def stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[ImageInput]] = None,
        history: Optional[Sequence[HistoryTurn]] = None,
        hedge_delay_ms: Optional[int] = None,
    ) -> Iterator[str]:
        impl = self._pick()
        # Transparent dispatch to the chosen per-chain backend. ``hedge_delay_ms``
        # and ``history`` are forwarded ONLY when set (PINNED CONTRACT), so the
        # plain ``LLMClient`` protocol stream() of any other backing client is
        # called unchanged -- keeping default (None) behaviour byte-identical.
        kw: dict = {"system": system, "images": images}
        if history:
            kw["history"] = history
        if hedge_delay_ms is not None and isinstance(impl, HedgeLLM):
            return impl.stream(prompt, hedge_delay_ms=hedge_delay_ms, **kw)
        return impl.stream(prompt, **kw)


def _to_data_uri(raw: bytes) -> str:
    import base64

    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
