"""Capability-provider exceptions stay detail-free at every public seam.

All providers, planners, and model clients in this module are deterministic
fakes.  The tests perform no network, model, audio, or device work.
"""

from __future__ import annotations

import gc
import logging
from threading import Event
import time
import weakref

from always_on_agent.capabilities import (
    CAPABILITY_PROVIDER_FAILED,
    CapabilityRegistry,
    CapabilityResult,
    CapabilitySpec,
)
from always_on_agent.events import EventKind, Mode
from always_on_agent.models import IntentDecision, IntentKind
from always_on_agent.planner_steps import PlannerExchange
from always_on_agent.react import FINAL_SYSTEM, ReactPlanner
from always_on_agent.supervisor import AgentSupervisor, _FAILURE_APOLOGY
from core.capabilities import attach_llm_capabilities
from core.llm import (
    LLAMACPP_TOOL_FORMAT_MINICPM5,
    LlamaCppToolCompletion,
    capability_context,
)
from core.minicpm_tools import MiniCPMXmlPlannerBackend


_CANARY = "SYNTHETIC_PRIVATE_CAPABILITY_EXCEPTION_CANARY"
_EXPLICIT_SAFE_ERROR = "explicit_safe_provider_error"


def _hostile_exception_type(
    base: type[BaseException],
    hooks: list[str],
    *,
    weakrefable: bool = False,
) -> type[BaseException]:
    """Build an exception whose presentation/type-name hooks must stay inert."""

    class TrapMeta(type):
        def __getattribute__(cls, name: str):
            if name in {"__name__", "__qualname__"}:
                hooks.append(f"type.{name}")
                raise AssertionError(_CANARY)
            return type.__getattribute__(cls, name)

        def __repr__(cls) -> str:
            hooks.append("type.__repr__")
            raise AssertionError(_CANARY)

    namespace: dict[str, object] = {
        "__module__": __name__,
        "__str__": lambda self: _raise_hook(hooks, "exception.__str__"),
        "__repr__": lambda self: _raise_hook(hooks, "exception.__repr__"),
    }
    if weakrefable:
        namespace["__slots__"] = ("__weakref__",)
    return TrapMeta("HostileProviderFailure", (base,), namespace)


def _raise_hook(hooks: list[str], name: str) -> str:
    hooks.append(name)
    raise AssertionError(_CANARY)


def _raising_provider(error_type: type[BaseException]):
    def provider(_query: str, _context: dict[str, object]) -> CapabilityResult:
        raise error_type(_CANARY)

    return provider


class _TextPlannerLLM:
    def __init__(self) -> None:
        self._steps = ["TOOL flaky.tool: needle", "FINAL: recovered"]
        self.plan_prompts: list[str] = []

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return next(self.stream(prompt, system=system))

    def stream(self, prompt: str, *, system: str | None = None):
        if system == FINAL_SYSTEM:
            yield "safe synthesis"
            return
        self.plan_prompts.append(prompt)
        yield self._steps.pop(0)


class _NativePlannerLLM:
    tool_format = LLAMACPP_TOOL_FORMAT_MINICPM5

    def __init__(self) -> None:
        self.completions = [
            LlamaCppToolCompletion(
                '<function name="flaky.tool"><param name="query">needle</param></function>',
                "stop",
            ),
            LlamaCppToolCompletion("recovered", "stop"),
        ]
        self.calls: list[dict[str, object]] = []

    def complete_minicpm_tool_chat(
        self,
        *,
        messages,
        tools,
        first_token_hook=None,
        cancel_event=None,
    ) -> LlamaCppToolCompletion:
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "cancel_event": cancel_event,
            }
        )
        if first_token_hook is not None:
            first_token_hook()
        return self.completions.pop(0)


class _RecordingNativeBackend(MiniCPMXmlPlannerBackend):
    def __init__(self, llm: _NativePlannerLLM) -> None:
        super().__init__(llm)  # type: ignore[arg-type]
        self.exchanges: list[tuple[PlannerExchange, ...]] = []

    def next_step(self, **kwargs):
        self.exchanges.append(tuple(kwargs["exchanges"]))
        return super().next_step(**kwargs)


class _UnusedFinalLLM:
    def generate(self, _prompt: str, *, system: str | None = None) -> str:
        return "safe synthesis"

    def stream(self, _prompt: str, *, system: str | None = None):
        yield "safe synthesis"


def _planner_registry(provider) -> CapabilityRegistry:
    registry = CapabilityRegistry()
    registry.register(
        "flaky.tool",
        provider,
        spec=CapabilitySpec("flaky.tool", "test tool", planner_tool=True),
    )
    return registry


def _run_native(provider):
    registry = _planner_registry(provider)
    native_llm = _NativePlannerLLM()
    backend = _RecordingNativeBackend(native_llm)
    result = ReactPlanner(
        _UnusedFinalLLM(),
        registry,
        tools=("flaky.tool",),
        step_backend=backend,
    ).run("research needle", {})
    return result, backend, native_llm


def test_registry_exception_is_one_fixed_result_without_presentation_hooks() -> None:
    hooks: list[str] = []
    error_type = _hostile_exception_type(Exception, hooks)
    registry = CapabilityRegistry()
    registry.register("hostile.tool", _raising_provider(error_type))

    result = registry.invoke("hostile.tool", "safe query")

    assert result == CapabilityResult(
        False,
        "",
        error=CAPABILITY_PROVIDER_FAILED,
    )
    assert hooks == []


def test_observer_receipt_closes_with_only_the_fixed_failure() -> None:
    hooks: list[str] = []
    error_type = _hostile_exception_type(Exception, hooks)
    registry = CapabilityRegistry()
    registry.register("hostile.tool", _raising_provider(error_type))
    receipts = []
    registry.observe_invocations(receipts.append)

    result = registry.invoke("hostile.tool", "safe query", {"task_id": "task-1"})

    assert result.error == CAPABILITY_PROVIDER_FAILED
    assert [receipt.phase for receipt in receipts] == ["started", "finished"]
    assert {receipt.invocation_id for receipt in receipts} == {1}
    assert receipts[0].result is None
    assert receipts[1].result is not None
    assert receipts[1].result.error == CAPABILITY_PROVIDER_FAILED
    assert receipts[1].result.text == ""
    assert dict(receipts[1].result.data) == {}
    assert hooks == []


def test_observer_closes_then_reraises_the_same_direct_baseexception() -> None:
    hooks: list[str] = []
    error_type = _hostile_exception_type(BaseException, hooks)
    thrown: list[BaseException] = []

    def provider(_query: str, _context: dict[str, object]) -> CapabilityResult:
        error = error_type(_CANARY)
        thrown.append(error)
        raise error

    registry = CapabilityRegistry()
    registry.register("abort.tool", provider)
    receipts = []
    registry.observe_invocations(receipts.append)
    propagated: BaseException | None = None

    try:
        registry.invoke("abort.tool", "safe query", {"task_id": "task-2"})
    except BaseException as error:
        propagated = error

    assert propagated is thrown[0]
    assert [receipt.phase for receipt in receipts] == ["started", "finished"]
    assert {receipt.invocation_id for receipt in receipts} == {1}
    assert receipts[0].result is None
    assert receipts[1].result is not None
    assert receipts[1].result.ok is False
    assert receipts[1].result.text == ""
    assert dict(receipts[1].result.data) == {}
    assert receipts[1].result.citations == ()
    assert receipts[1].result.error == CAPABILITY_PROVIDER_FAILED
    assert hooks == []


def test_invalid_result_proxy_and_subclass_fail_before_attribute_hooks() -> None:
    hooks: list[str] = []

    class HostileProxy:
        def __getattribute__(self, name: str):
            hooks.append(f"proxy.{name}")
            raise AssertionError(_CANARY)

        def __str__(self) -> str:
            hooks.append("proxy.__str__")
            raise AssertionError(_CANARY)

        def __repr__(self) -> str:
            hooks.append("proxy.__repr__")
            raise AssertionError(_CANARY)

    class HostileSubclass(CapabilityResult):
        def __getattribute__(self, name: str):
            if name in {"ok", "text", "data", "citations", "error"}:
                hooks.append(f"subclass.{name}")
                raise AssertionError(_CANARY)
            return super().__getattribute__(name)

        def __str__(self) -> str:
            hooks.append("subclass.__str__")
            raise AssertionError(_CANARY)

        def __repr__(self) -> str:
            hooks.append("subclass.__repr__")
            raise AssertionError(_CANARY)

    invalid_results = (
        HostileProxy(),
        HostileSubclass(True, "unsafe", error=_CANARY),
    )
    registry = CapabilityRegistry()
    current = iter(invalid_results)
    registry.register("invalid.tool", lambda _query, _context: next(current))
    receipts = []
    registry.observe_invocations(receipts.append)

    first = registry.invoke("invalid.tool", "safe query")
    second = registry.invoke("invalid.tool", "safe query")

    expected = CapabilityResult(False, "", error=CAPABILITY_PROVIDER_FAILED)
    assert first == expected
    assert second == expected
    assert first is not second
    assert [receipt.phase for receipt in receipts] == [
        "started",
        "finished",
        "started",
        "finished",
    ]
    assert [
        receipt.result.error for receipt in receipts if receipt.result is not None
    ] == [CAPABILITY_PROVIDER_FAILED, CAPABILITY_PROVIDER_FAILED]
    assert hooks == []


def test_escalated_failure_restores_the_outer_capability_context() -> None:
    hooks: list[str] = []
    error_type = _hostile_exception_type(Exception, hooks)
    registry = CapabilityRegistry()
    registry.register("agent.react", _raising_provider(error_type))
    attach_llm_capabilities(
        registry,
        _UnusedFinalLLM(),
        escalate=lambda _query, _context: True,
    )
    outer = {"sensitivity": "outer-context"}
    token = capability_context.set(outer)
    try:
        result = registry.invoke(
            "assistant.answer",
            "compare these options",
            {"mode": Mode.ASSISTANT.value},
        )
        assert result.error == CAPABILITY_PROVIDER_FAILED
        assert capability_context.get() is outer
    finally:
        capability_context.reset(token)

    assert hooks == []


def test_text_react_sees_only_the_fixed_failure_code() -> None:
    hooks: list[str] = []
    error_type = _hostile_exception_type(Exception, hooks)
    llm = _TextPlannerLLM()
    result = ReactPlanner(
        llm,
        _planner_registry(_raising_provider(error_type)),
        tools=("flaky.tool",),
    ).run("research needle", {})

    assert result.ok is True
    assert result.text == "recovered"
    assert len(llm.plan_prompts) == 2
    assert f"flaky.tool failed: {CAPABILITY_PROVIDER_FAILED}" in llm.plan_prompts[1]
    assert _CANARY not in llm.plan_prompts[1]
    assert hooks == []


def test_native_react_exchange_and_model_messages_are_detail_free() -> None:
    hooks: list[str] = []
    error_type = _hostile_exception_type(Exception, hooks)
    result, backend, native_llm = _run_native(_raising_provider(error_type))

    assert result.ok is True
    assert result.text == "recovered"
    assert len(backend.exchanges) == 2
    assert len(backend.exchanges[1]) == 1
    exchange = backend.exchanges[1][0]
    assert exchange.ok is False
    assert exchange.result == f"Tool failed: {CAPABILITY_PROVIDER_FAILED}"
    second_messages = native_llm.calls[1]["messages"]
    tool_messages = [
        message["content"] for message in second_messages if message["role"] == "tool"
    ]
    assert tool_messages == [f"Tool failed: {CAPABILITY_PROVIDER_FAILED}"]
    assert all(_CANARY not in text for text in tool_messages)
    assert hooks == []


def test_explicit_returned_error_stays_exact_through_observer_and_planners() -> None:
    explicit = CapabilityResult(
        False,
        "",
        data={"safe_code": 7},
        citations=("safe-ref",),
        error=_EXPLICIT_SAFE_ERROR,
    )

    registry = CapabilityRegistry()
    registry.register("explicit.tool", lambda _query, _context: explicit)
    receipts = []
    registry.observe_invocations(receipts.append)
    assert registry.invoke("explicit.tool", "safe query") is explicit
    assert receipts[-1].result is not None
    assert receipts[-1].result.error == _EXPLICIT_SAFE_ERROR
    assert dict(receipts[-1].result.data) == {"safe_code": 7}
    assert receipts[-1].result.citations == ("safe-ref",)

    text_llm = _TextPlannerLLM()
    text_result = ReactPlanner(
        text_llm,
        _planner_registry(lambda _query, _context: explicit),
        tools=("flaky.tool",),
    ).run("research needle", {})
    assert text_result.text == "recovered"
    assert f"flaky.tool failed: {_EXPLICIT_SAFE_ERROR}" in text_llm.plan_prompts[1]

    native_result, backend, native_llm = _run_native(lambda _query, _context: explicit)
    assert native_result.text == "recovered"
    assert backend.exchanges[1][0].result == (f"Tool failed: {_EXPLICIT_SAFE_ERROR}")
    second_messages = native_llm.calls[1]["messages"]
    assert [
        message["content"] for message in second_messages if message["role"] == "tool"
    ] == [f"Tool failed: {_EXPLICIT_SAFE_ERROR}"]


def test_missing_capability_and_action_authority_contracts_are_unchanged() -> None:
    registry = CapabilityRegistry()
    invoked: list[str] = []
    registry.register(
        "danger.tool",
        lambda query, _context: (
            invoked.append(query) or CapabilityResult(True, "executed")
        ),
        spec=CapabilitySpec(
            "danger.tool",
            "dangerous test action",
            side_effecting=True,
            authority="direct_live",
        ),
    )

    assert registry.invoke("missing.tool", "query") == CapabilityResult(
        False,
        "",
        error="missing capability: missing.tool",
    )
    assert registry.invoke("danger.tool", "do it") == CapabilityResult(
        True,
        "I can't perform that action from this request.",
        data={"executed": False, "blocked": "action_authority"},
    )
    assert invoked == []


def _wait_for_failed_task(
    supervisor: AgentSupervisor,
    task_id: str,
) -> list:
    deadline = time.monotonic() + 2.0
    terminal = []
    while time.monotonic() < deadline:
        supervisor.drain()
        terminal = [
            event
            for event in supervisor.state.event_log
            if event.kind == EventKind.TASK_FAILED
            and event.payload.get("task_id") == task_id
        ]
        if (
            terminal
            and task_id not in supervisor.state.active_tasks
            and supervisor.tasks.active_count == 0
        ):
            break
        Event().wait(0.005)
    supervisor.drain()
    return terminal


def _run_task_failure(provider, caplog):
    registry = CapabilityRegistry()
    registry.register(
        "assistant.answer",
        provider,
        spec=CapabilitySpec("assistant.answer", "answer"),
    )
    supervisor = AgentSupervisor(capabilities=registry)
    task = supervisor.tasks.create_task(
        IntentDecision(
            IntentKind.ASSISTANT,
            1.0,
            "safe task query",
            "test",
            mode=Mode.ASSISTANT,
        )
    )
    try:
        with caplog.at_level(logging.DEBUG):
            assert supervisor._start_task(task)  # noqa: SLF001
            terminal = _wait_for_failed_task(supervisor, task.task_id)
        assert len(terminal) == 1
        assert terminal[0].payload["error"] == CAPABILITY_PROVIDER_FAILED
        assert list(supervisor.state.failures) == [CAPABILITY_PROVIDER_FAILED]
        assert list(supervisor.state.spoken_outputs) == [_FAILURE_APOLOGY]
        assert _CANARY not in caplog.text
        task_records = [
            record for record in caplog.records if record.name == "speaker.tasks"
        ]
        assert task_records
        assert all(record.exc_info is None for record in task_records)
    finally:
        supervisor.shutdown()


def test_task_failure_event_receipt_and_logs_hide_ordinary_exception(
    caplog,
) -> None:
    hooks: list[str] = []
    error_type = _hostile_exception_type(Exception, hooks)
    _run_task_failure(_raising_provider(error_type), caplog)
    assert hooks == []


def test_task_runtime_contains_and_releases_provider_baseexception(caplog) -> None:
    hooks: list[str] = []
    error_type = _hostile_exception_type(
        BaseException,
        hooks,
        weakrefable=True,
    )
    retained: list[weakref.ReferenceType[BaseException]] = []

    def provider(_query: str, _context: dict[str, object]) -> CapabilityResult:
        error = error_type(_CANARY)
        retained.append(weakref.ref(error))
        raise error

    # Direct registry calls preserve process-control BaseException propagation.
    direct = CapabilityRegistry()
    direct.register("abort.tool", provider)
    propagated = False
    try:
        direct.invoke("abort.tool", "safe query")
    except BaseException as error:
        propagated = isinstance(error, error_type)
    assert propagated is True
    assert hooks == []
    retained.clear()

    _run_task_failure(provider, caplog)
    for _ in range(3):
        gc.collect()
    assert len(retained) == 1
    assert retained[0]() is None
    assert hooks == []
