"""Deterministic headless coverage for the llama.cpp CPU abort boundary.

The fake client below behaves like a native call: it knows only the callback
installed on its context, never the task's ``cancel_event``.  Event-controlled
plans make cancellation/completion races reproducible without a model, native
library, audio device, or timing sleeps.
"""
from __future__ import annotations

import gc
import sys
import threading
import time
import weakref
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from typing import Callable

import pytest

from always_on_agent.events import EventKind, Mode
from always_on_agent.supervisor import _TIMEOUT_APOLOGY
from core.engines.scripted import ScriptedEngine
from core.llm import (
    LLAMACPP_PINNED_VERSION,
    LLMCallCancelled,
    LlamaCppLLM,
    _LlamaCppAbortAPI,
    _resolve_llamacpp_abort_api,
    capability_context,
)
from core.runtime import VoiceRuntime


_WAIT_SECONDS = 2.0


@dataclass
class _Outcome:
    value: object | None = None
    error: BaseException | None = None


@dataclass
class _CallPlan:
    kind: str
    content: str = "ok"
    entered: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)
    abort_seen: threading.Event = field(default_factory=threading.Event)
    stream_closed: threading.Event = field(default_factory=threading.Event)


@dataclass
class _FakeContext:
    memory: object = field(default_factory=object)
    callback: Callable[[object | None], bool] | None = None
    callback_data: object | None = None


class _AbortHarness:
    """Python stand-in for the audited low-level llama.cpp ctypes functions."""

    def __init__(self, *, fail_clear: bool = False) -> None:
        self.fail_clear = fail_clear
        self.set_calls = 0
        self.get_memory_calls = 0
        self.clear_calls: list[tuple[object, bool]] = []

    @staticmethod
    def callback_type(callback):
        return callback

    def set_callback(self, ctx: _FakeContext, callback, data) -> None:
        self.set_calls += 1
        ctx.callback = callback
        ctx.callback_data = data

    def get_memory(self, ctx: _FakeContext) -> object:
        self.get_memory_calls += 1
        return ctx.memory

    def clear_memory(self, memory: object, clear_data: bool) -> None:
        if self.fail_clear:
            raise RuntimeError("synthetic memory-clear failure")
        self.clear_calls.append((memory, clear_data))

    def api(self) -> _LlamaCppAbortAPI:
        return _LlamaCppAbortAPI(
            callback_type=self.callback_type,
            set_callback=self.set_callback,
            get_memory=self.get_memory,
            clear_memory=self.clear_memory,
        )


class _BlockingNativeIterator:
    def __init__(self, ctx: _FakeContext, plan: _CallPlan) -> None:
        self._ctx = ctx
        self._plan = plan

    def __iter__(self):
        return self

    def __next__(self):
        self._plan.entered.set()
        while True:
            callback = self._ctx.callback
            if callback is not None and callback(self._ctx.callback_data):
                self._plan.abort_seen.set()
                raise RuntimeError("synthetic native stream abort")
            if self._plan.release.wait(0.01):
                raise StopIteration

    def close(self) -> None:
        self._plan.stream_closed.set()


class _SyntheticCloseControlFlow(BaseException):
    """Foreign iterator control flow that ordinary Exception guards miss."""


class _CloseControlFlowIterator:
    def __init__(self, plan: _CallPlan) -> None:
        self._plan = plan
        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        self._plan.entered.set()
        return {"choices": [{"delta": {"content": self._plan.content}}]}

    def close(self) -> None:
        self._plan.stream_closed.set()
        raise _SyntheticCloseControlFlow("synthetic close control flow")


class _FakeNativeClient:
    """Scripted high-level client whose blocking paths see only ``ctx.callback``."""

    def __init__(self, *plans: _CallPlan, fail_reset: bool = False) -> None:
        self.ctx = _FakeContext()
        self._plans = list(plans)
        self.fail_reset = fail_reset
        self.calls = 0
        self.reset_calls = 0

    def create_chat_completion(self, *, messages, stream=False, **options):
        del messages, options
        if self.calls >= len(self._plans):
            raise AssertionError("unexpected fake llama.cpp call")
        plan = self._plans[self.calls]
        self.calls += 1

        if stream:
            if plan.kind == "stream_abort":
                return _BlockingNativeIterator(self.ctx, plan)
            if plan.kind == "stream_healthy":
                plan.entered.set()
                return iter(
                    [{"choices": [{"delta": {"content": plan.content}}]}]
                )
            if plan.kind == "stream_close_control_flow":
                return _CloseControlFlowIterator(plan)
            raise AssertionError(f"unsupported stream plan: {plan.kind}")

        plan.entered.set()
        if plan.kind == "healthy":
            return {"choices": [{"message": {"content": plan.content}}]}
        if plan.kind == "complete_on_release":
            if not plan.release.wait(_WAIT_SECONDS):
                raise AssertionError("test never released native completion")
            return {"choices": [{"message": {"content": plan.content}}]}
        if plan.kind == "poll_abort_or_release":
            while True:
                callback = self.ctx.callback
                if callback is not None and callback(self.ctx.callback_data):
                    plan.abort_seen.set()
                    raise RuntimeError("synthetic llama_decode abort")
                if plan.release.wait(0.01):
                    return {"choices": [{"message": {"content": plan.content}}]}
        raise AssertionError(f"unsupported generate plan: {plan.kind}")

    def reset(self) -> None:
        self.reset_calls += 1
        if self.fail_reset:
            raise RuntimeError("synthetic reset failure")


class _ObservedCancelEvent:
    """Event-like object that exposes when lock admission first polls it."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self.checked = threading.Event()

    def is_set(self) -> bool:
        self.checked.set()
        return self._event.is_set()

    def set(self) -> None:
        self._event.set()


def _spawn_generate(
    llm: LlamaCppLLM,
    cancel_event: object | None,
) -> tuple[threading.Thread, _Outcome]:
    outcome = _Outcome()

    def run() -> None:
        token = capability_context.set(
            {"cancel_event": cancel_event} if cancel_event is not None else {}
        )
        try:
            outcome.value = llm.generate("hello")
        except BaseException as exc:  # preserve control-flow failures for assertions
            outcome.error = exc
        finally:
            capability_context.reset(token)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, outcome


def _spawn_collect(stream) -> tuple[threading.Thread, _Outcome]:
    outcome = _Outcome()

    def run() -> None:
        try:
            outcome.value = list(stream)
        except BaseException as exc:  # preserve control-flow failures for assertions
            outcome.error = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, outcome


def _join(thread: threading.Thread) -> None:
    thread.join(_WAIT_SECONDS)
    assert not thread.is_alive(), "fake llama.cpp invocation did not terminate"


def _wait_until(predicate, timeout: float = _WAIT_SECONDS) -> bool:
    deadline = time.monotonic() + timeout
    pulse = threading.Event()
    while time.monotonic() < deadline:
        if predicate():
            return True
        pulse.wait(0.005)
    return bool(predicate())


def _verified_module(**overrides):
    values = {
        "__version__": LLAMACPP_PINNED_VERSION,
        "ggml_abort_callback": lambda callback: callback,
        "llama_set_abort_callback": lambda ctx, callback, data: None,
        "llama_get_memory": lambda ctx: object(),
        "llama_memory_clear": lambda memory, clear_data: None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_exact_version_and_abort_symbols_are_accepted() -> None:
    module = _verified_module()

    api = _resolve_llamacpp_abort_api(module)

    assert api.callback_type is module.ggml_abort_callback
    assert api.set_callback is module.llama_set_abort_callback
    assert api.get_memory is module.llama_get_memory
    assert api.clear_memory is module.llama_memory_clear


@pytest.mark.parametrize("version", [None, "", "0.3.32", "0.3.33.post1"])
def test_non_exact_llamacpp_version_is_rejected(version) -> None:
    module = _verified_module(__version__=version)

    with pytest.raises(RuntimeError, match=r"llama-cpp-python==0\.3\.33"):
        _resolve_llamacpp_abort_api(module)


@pytest.mark.parametrize(
    "symbol",
    [
        "ggml_abort_callback",
        "llama_set_abort_callback",
        "llama_get_memory",
        "llama_memory_clear",
    ],
)
def test_missing_or_noncallable_abort_symbol_is_rejected(symbol: str) -> None:
    module = _verified_module(**{symbol: None})

    with pytest.raises(RuntimeError, match=symbol):
        _resolve_llamacpp_abort_api(module)


def test_production_gpu_configuration_fails_closed_before_model_load() -> None:
    llm = LlamaCppLLM("/unreachable.gguf", n_gpu_layers=1)

    with pytest.raises(RuntimeError, match=r"CPU-only.*n_gpu_layers=0"):
        llm.generate("hello")

    assert llm._client is None


def test_pretoken_native_abort_is_driven_by_context_event_and_recovers() -> None:
    plan = _CallPlan("poll_abort_or_release")
    harness = _AbortHarness()
    client = _FakeNativeClient(plan)
    llm = LlamaCppLLM("/fake.gguf", client=client, abort_api=harness.api())
    cancel = threading.Event()
    thread, outcome = _spawn_generate(llm, cancel)
    try:
        assert plan.entered.wait(_WAIT_SECONDS), "native inference never started"
        assert client.ctx.callback is not None

        cancel.set()
        assert plan.abort_seen.wait(_WAIT_SECONDS), "native callback did not observe cancel"
        _join(thread)
    finally:
        plan.release.set()
        thread.join(_WAIT_SECONDS)

    assert isinstance(outcome.error, LLMCallCancelled)
    assert outcome.value is None
    assert harness.set_calls == 1
    assert harness.get_memory_calls == 1
    assert harness.clear_calls == [(client.ctx.memory, True)]
    assert client.reset_calls == 1


def test_cancelled_lock_waiter_never_starts_or_aborts_current_owner() -> None:
    owner_plan = _CallPlan("poll_abort_or_release", content="owner")
    impossible_waiter_plan = _CallPlan("healthy", content="stale")
    harness = _AbortHarness()
    client = _FakeNativeClient(owner_plan, impossible_waiter_plan)
    llm = LlamaCppLLM("/fake.gguf", client=client, abort_api=harness.api())
    owner_cancel = threading.Event()
    owner_thread, owner_outcome = _spawn_generate(llm, owner_cancel)
    assert owner_plan.entered.wait(_WAIT_SECONDS), "owner never acquired model context"

    waiter_cancel = _ObservedCancelEvent()
    waiter_thread, waiter_outcome = _spawn_generate(llm, waiter_cancel)
    try:
        assert waiter_cancel.checked.wait(_WAIT_SECONDS), "waiter never reached admission"
        waiter_cancel.set()
        _join(waiter_thread)

        assert isinstance(waiter_outcome.error, LLMCallCancelled)
        assert client.calls == 1, "cancelled waiter started stale native inference"
        assert not owner_plan.abort_seen.is_set(), "waiter cancellation aborted owner"
        assert owner_thread.is_alive(), "owner was disturbed before its release"
    finally:
        owner_plan.release.set()
        owner_thread.join(_WAIT_SECONDS)
        waiter_thread.join(_WAIT_SECONDS)

    assert not owner_thread.is_alive()
    assert owner_outcome.error is None
    assert owner_outcome.value == "owner"
    assert client.reset_calls == 0
    assert harness.clear_calls == []


def test_cancellation_wins_native_completion_race() -> None:
    plan = _CallPlan("complete_on_release", content="must-not-escape")
    harness = _AbortHarness()
    client = _FakeNativeClient(plan)
    llm = LlamaCppLLM("/fake.gguf", client=client, abort_api=harness.api())
    cancel = threading.Event()
    thread, outcome = _spawn_generate(llm, cancel)
    try:
        assert plan.entered.wait(_WAIT_SECONDS), "native completion never started"
        cancel.set()
        plan.release.set()
        _join(thread)
    finally:
        plan.release.set()
        thread.join(_WAIT_SECONDS)

    assert isinstance(outcome.error, LLMCallCancelled)
    assert outcome.value is None
    assert client.reset_calls == 1
    assert harness.clear_calls == [(client.ctx.memory, True)]


def test_cleanup_failure_poisons_context_and_next_call_fails_closed() -> None:
    aborted = _CallPlan("poll_abort_or_release")
    would_be_healthy = _CallPlan("healthy")
    harness = _AbortHarness(fail_clear=True)
    client = _FakeNativeClient(aborted, would_be_healthy)
    llm = LlamaCppLLM("/fake.gguf", client=client, abort_api=harness.api())
    cancel = threading.Event()
    thread, outcome = _spawn_generate(llm, cancel)
    try:
        assert aborted.entered.wait(_WAIT_SECONDS), "native inference never started"
        cancel.set()
        assert aborted.abort_seen.wait(_WAIT_SECONDS), "native callback did not abort"
        _join(thread)
    finally:
        aborted.release.set()
        thread.join(_WAIT_SECONDS)

    assert isinstance(outcome.error, LLMCallCancelled)
    assert llm._context_poisoned is True

    with pytest.raises(RuntimeError, match=r"context is poisoned.*restart required"):
        llm.generate("healthy turn")

    assert client.calls == 1, "poisoned context admitted another native call"


def test_reset_failure_poisons_context_and_next_call_fails_closed() -> None:
    aborted = _CallPlan("poll_abort_or_release")
    would_be_healthy = _CallPlan("healthy")
    harness = _AbortHarness()
    client = _FakeNativeClient(aborted, would_be_healthy, fail_reset=True)
    llm = LlamaCppLLM("/fake.gguf", client=client, abort_api=harness.api())
    cancel = threading.Event()
    thread, outcome = _spawn_generate(llm, cancel)
    try:
        assert aborted.entered.wait(_WAIT_SECONDS), "native inference never started"
        cancel.set()
        assert aborted.abort_seen.wait(_WAIT_SECONDS), "native callback did not abort"
        _join(thread)
    finally:
        aborted.release.set()
        thread.join(_WAIT_SECONDS)

    assert isinstance(outcome.error, LLMCallCancelled)
    assert harness.clear_calls == [(client.ctx.memory, True)]
    assert client.reset_calls == 1
    assert llm._context_poisoned is True
    with pytest.raises(RuntimeError, match=r"context is poisoned.*restart required"):
        llm.generate("must remain fenced")
    assert client.calls == 1


def test_early_stream_cancellation_closes_native_iterator() -> None:
    plan = _CallPlan("stream_abort")
    harness = _AbortHarness()
    client = _FakeNativeClient(plan)
    llm = LlamaCppLLM("/fake.gguf", client=client, abort_api=harness.api())
    cancel = threading.Event()

    token = capability_context.set({"cancel_event": cancel})
    try:
        stream = llm.stream("hello")
    finally:
        capability_context.reset(token)

    thread, outcome = _spawn_collect(stream)
    try:
        assert plan.entered.wait(_WAIT_SECONDS), "native iterator never awaited token one"
        cancel.set()
        assert plan.abort_seen.wait(_WAIT_SECONDS), "stream callback did not abort"
        _join(thread)
    finally:
        plan.release.set()
        thread.join(_WAIT_SECONDS)

    assert isinstance(outcome.error, LLMCallCancelled)
    assert outcome.value is None
    assert plan.stream_closed.is_set()
    assert client.reset_calls == 1
    assert harness.clear_calls == [(client.ctx.memory, True)]


def test_immediate_healthy_call_succeeds_after_native_abort() -> None:
    aborted = _CallPlan("poll_abort_or_release")
    healthy = _CallPlan("healthy", content="recovered")
    harness = _AbortHarness()
    client = _FakeNativeClient(aborted, healthy)
    llm = LlamaCppLLM("/fake.gguf", client=client, abort_api=harness.api())
    cancel = threading.Event()
    thread, outcome = _spawn_generate(llm, cancel)
    try:
        assert aborted.entered.wait(_WAIT_SECONDS), "native inference never started"
        cancel.set()
        assert aborted.abort_seen.wait(_WAIT_SECONDS), "native callback did not abort"
        _join(thread)
    finally:
        aborted.release.set()
        thread.join(_WAIT_SECONDS)

    assert isinstance(outcome.error, LLMCallCancelled)
    assert llm.generate("next turn") == "recovered"
    assert client.calls == 2
    assert client.reset_calls == 1
    assert harness.set_calls == 1, "callback should stay retained on the same context"


def test_native_callback_does_not_retain_discarded_llm_wrapper() -> None:
    healthy = _CallPlan("healthy", content="done")
    harness = _AbortHarness()
    client = _FakeNativeClient(healthy)
    llm = LlamaCppLLM("/fake.gguf", client=client, abort_api=harness.api())

    assert llm.generate("register callback") == "done"
    wrapper_ref = weakref.ref(llm)
    del llm
    gc.collect()

    assert wrapper_ref() is None
    # The fake deliberately retains the Python callback (a real C context keeps
    # only its raw pointer); its weak closure still must not pin the test wrapper.
    assert client.ctx.callback is not None


def test_stream_close_baseexception_cannot_leak_model_lock() -> None:
    broken_close = _CallPlan("stream_close_control_flow", content="first")
    healthy = _CallPlan("healthy", content="second")
    harness = _AbortHarness()
    client = _FakeNativeClient(broken_close, healthy)
    llm = LlamaCppLLM("/fake.gguf", client=client, abort_api=harness.api())

    with pytest.raises(_SyntheticCloseControlFlow):
        list(llm.stream("close must not leak"))

    assert broken_close.stream_closed.is_set()
    assert llm.generate("next call") == "second"
    assert client.calls == 2


def test_runtime_barge_storm_aborts_native_calls_and_reuses_all_capacity() -> None:
    """Barge-in alone must unwind native pre-token work and free both bulkheads."""

    # More turns than TaskRuntime's provider/coordinator ceiling prove that
    # native exits return the provider semaphore too; merely retiring the
    # visible coordinator would stall before every plan can enter.
    engine = ScriptedEngine()
    harness = _AbortHarness()
    placeholder = _CallPlan("stream_healthy")
    client = _FakeNativeClient(placeholder)
    llm = LlamaCppLLM("/fake.gguf", client=client, abort_api=harness.api())
    runtime = VoiceRuntime(
        engine,
        llm,
        start_mode=Mode.ASSISTANT,
        stream_tts=True,
    )
    attempts = runtime.supervisor.tasks.max_active_tasks + 2
    blocked = [_CallPlan("stream_abort") for _ in range(attempts)]
    healthy = _CallPlan(
        "stream_healthy",
        content="Healthy after native cancellations.",
    )
    # Install the final script before runtime threads start. The placeholder
    # avoids coupling this test to a second fake-client constructor shape.
    client._plans = [*blocked, healthy]
    runtime.start(run_bus=True)
    try:
        for index, plan in enumerate(blocked):
            engine.final(f"native pre-token block {index}")
            assert plan.entered.wait(_WAIT_SECONDS), (
                f"native call {index} never entered; provider capacity leaked"
            )

            engine.barge_in()

            assert plan.abort_seen.wait(_WAIT_SECONDS), (
                f"barge-in {index} never reached the native abort callback"
            )
            assert plan.stream_closed.wait(_WAIT_SECONDS), (
                f"barge-in {index} did not close the native iterator"
            )
            assert _wait_until(
                lambda: runtime.supervisor.tasks.active_count == 0
                and not runtime.supervisor.state.active_tasks
                and not runtime.supervisor.state.queued_tasks
            ), f"barge-in {index} leaked coordinator capacity"
            assert not plan.release.is_set(), (
                "production cancellation must not depend on a test release gate"
            )
            assert engine.spoken == [], "cancelled native reply reached speech"

        engine.final("healthy turn after native abort storm")
        assert healthy.entered.wait(_WAIT_SECONDS), (
            "healthy native turn could not reclaim provider capacity"
        )
        assert runtime.wait_idle(timeout=_WAIT_SECONDS)

        assert client.calls == attempts + 1
        assert client.reset_calls == attempts
        assert len(harness.clear_calls) == attempts
        assert harness.set_calls == 1
        assert engine.spoken == ["Healthy after native cancellations."]
        assert runtime.supervisor.tasks.active_count == 0
    finally:
        # These gates are only a broken-test escape hatch. Every passing-path
        # assertion above happens while all of them remain untouched.
        for plan in blocked:
            plan.release.set()
        runtime.stop()


def test_watchdog_deadline_aborts_native_stream_and_releases_capacity() -> None:
    """The watchdog's task Event must drive the same native abort as barge-in."""

    blocked = _CallPlan("stream_abort", content="STALE timeout reply.")
    healthy = _CallPlan(
        "stream_healthy",
        content="Healthy after watchdog cancellation.",
    )
    harness = _AbortHarness()
    client = _FakeNativeClient(blocked, healthy)
    llm = LlamaCppLLM("/fake.gguf", client=client, abort_api=harness.api())
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        llm,
        start_mode=Mode.ASSISTANT,
        stream_tts=True,
        task_timeouts={"assistant": 60.0},
    )
    # One provider slot makes the healthy turn a direct proof that the timed-out
    # provider returned capacity, rather than merely freeing its coordinator.
    runtime.supervisor.tasks._invocation_slots = threading.BoundedSemaphore(1)
    runtime.start(run_bus=True)
    try:
        engine.final("expire inside native decode")
        assert blocked.entered.wait(_WAIT_SECONDS), "native timeout call never entered"
        assert _wait_until(
            lambda: len(runtime.supervisor.state.active_tasks) == 1
        ), "timed task was not registered"
        task_id, task = next(iter(runtime.supervisor.state.active_tasks.items()))

        # Drive the production watchdog/reaper path without a wall-clock sleep.
        task.deadline_at = time.monotonic() - 1.0
        runtime._watchdog.tick()

        assert blocked.abort_seen.wait(_WAIT_SECONDS), (
            "watchdog cancellation did not reach the native callback"
        )
        assert blocked.stream_closed.wait(_WAIT_SECONDS), (
            "watchdog cancellation did not close the native iterator"
        )
        assert _wait_until(
            lambda: runtime.supervisor.tasks.active_count == 0
            and not runtime.supervisor.state.active_tasks
            and not runtime.supervisor.state.queued_tasks
        ), "watchdog cancellation leaked coordinator capacity"
        assert runtime.wait_idle(timeout=_WAIT_SECONDS)
        assert not blocked.release.is_set()
        assert "STALE timeout reply." not in engine.spoken
        assert engine.spoken == [_TIMEOUT_APOLOGY]
        terminal = [
            event
            for event in runtime.supervisor.state.event_log
            if event.payload.get("task_id") == task_id
            and event.kind
            in {
                EventKind.TASK_CANCELLED,
                EventKind.TASK_COMPLETED,
                EventKind.TASK_FAILED,
            }
        ]
        assert [event.kind for event in terminal] == [EventKind.TASK_CANCELLED]
        assert terminal[0].payload.get("reaped") is True

        engine.final("healthy after timeout")
        assert healthy.entered.wait(_WAIT_SECONDS), (
            "timed-out native provider did not return its sole capacity slot"
        )
        assert runtime.wait_idle(timeout=_WAIT_SECONDS)
        assert engine.spoken == [
            _TIMEOUT_APOLOGY,
            "Healthy after watchdog cancellation.",
        ]
        assert client.calls == 2
        assert client.reset_calls == 1
    finally:
        blocked.release.set()
        runtime.stop()


def test_runtime_stop_aborts_pretoken_native_stream_before_returning() -> None:
    blocked = _CallPlan("stream_abort", content="STALE shutdown reply.")
    harness = _AbortHarness()
    client = _FakeNativeClient(blocked)
    llm = LlamaCppLLM("/fake.gguf", client=client, abort_api=harness.api())
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        llm,
        start_mode=Mode.ASSISTANT,
        stream_tts=True,
    )
    stopped = False
    runtime.start(run_bus=True)
    try:
        engine.final("stop inside native decode")
        assert blocked.entered.wait(_WAIT_SECONDS), "native shutdown call never entered"

        runtime.stop()
        stopped = True

        assert blocked.abort_seen.is_set(), (
            "runtime.stop returned before native cancellation reached the callback"
        )
        assert blocked.stream_closed.is_set(), (
            "runtime.stop returned before the native iterator closed"
        )
        assert runtime.supervisor.tasks.active_count == 0
        assert not runtime.supervisor.state.active_tasks
        assert not runtime.supervisor.state.queued_tasks
        assert not blocked.release.is_set()
        assert engine.spoken == []
        assert client.reset_calls == 1
        assert harness.clear_calls == [(client.ctx.memory, True)]
    finally:
        blocked.release.set()
        if not stopped:
            runtime.stop()
        _wait_until(lambda: runtime.supervisor.tasks.active_count == 0)


def test_cancel_during_model_construction_stops_before_inference(
    monkeypatch,
) -> None:
    """Model loading is non-preemptible, but cancellation still fences decode.

    ``llama_cpp.Llama(...)`` exposes no audited cancellation seam before it has
    produced a context, so the constructor is deliberately outside this slice's
    hard-abort guarantee. Once it returns, lock admission must notice retirement
    before installing a callback or calling ``create_chat_completion``.
    """

    constructor_entered = threading.Event()
    constructor_release = threading.Event()
    constructor_finished = threading.Event()
    instances: list[object] = []
    callback_installs: list[object] = []

    class _GatedProductionLlama:
        def __init__(self, **kwargs) -> None:
            del kwargs
            self.ctx = _FakeContext()
            self.create_calls = 0
            self.reset_calls = 0
            instances.append(self)
            constructor_entered.set()
            if not constructor_release.wait(_WAIT_SECONDS):
                raise AssertionError("test never released model construction")
            constructor_finished.set()

        def create_chat_completion(self, **kwargs):
            del kwargs
            self.create_calls += 1
            return {"choices": [{"message": {"content": "must not run"}}]}

        def reset(self) -> None:
            self.reset_calls += 1

    fake_module = ModuleType("llama_cpp")
    fake_module.__version__ = LLAMACPP_PINNED_VERSION
    fake_module.Llama = _GatedProductionLlama
    fake_module.ggml_abort_callback = lambda callback: callback

    def set_callback(ctx, callback, data) -> None:
        del callback, data
        callback_installs.append(ctx)

    fake_module.llama_set_abort_callback = set_callback
    fake_module.llama_get_memory = lambda ctx: ctx.memory
    fake_module.llama_memory_clear = lambda memory, clear_data: None
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)

    llm = LlamaCppLLM("/fake-production.gguf")
    cancel = threading.Event()
    thread, outcome = _spawn_generate(llm, cancel)
    try:
        assert constructor_entered.wait(_WAIT_SECONDS), "model constructor never entered"
        cancel.set()

        # Cancellation cannot force-kill an in-progress model load: the thread
        # remains inside the gated constructor until that native call returns.
        assert thread.is_alive()
        assert not constructor_finished.is_set()
        assert len(instances) == 1
        assert instances[0].create_calls == 0

        constructor_release.set()
        _join(thread)
    finally:
        constructor_release.set()
        thread.join(_WAIT_SECONDS)

    assert isinstance(outcome.error, LLMCallCancelled)
    assert outcome.value is None
    assert constructor_finished.is_set()
    assert instances[0].create_calls == 0
    assert instances[0].reset_calls == 0
    assert callback_installs == []
