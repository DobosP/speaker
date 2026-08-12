"""Fake-only contracts for the retained-context cloud-LLM egress fence.

The canaries in this file are deliberately synthetic and contain no PII-shaped
text.  These tests therefore exercise the admission fence itself, independently
of the outbound redactor and sensitivity classifier.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator

import pytest

from always_on_agent.capabilities import (
    CapabilityRegistry,
    CapabilityResult,
    CapabilitySpec,
    create_default_capabilities,
)
from always_on_agent.events import Mode
from always_on_agent.memory import MemoryItem, SessionMemory
from always_on_agent.models import (
    CLOUD_EGRESS_SCOPE_CONTEXT_KEY,
    RETAINED_PROMPT_CONTEXT_METADATA_KEY,
    SYNTHETIC_RESUME_TAIL_METADATA_KEY,
    CloudEgressScope,
    IntentKind,
)
from always_on_agent.recall import VISION_LABEL
from always_on_agent.react import ReactPlanner
from core import llm as llm_module
from core import llm_factory
from core import websearch as websearch_module
from core.capabilities import RecallConfig, attach_llm_capabilities
from core.cleanup import ScriptedTranscriptCleaner
from core.conversation import RecentContextConfig
from core.engines.scripted import ScriptedEngine
from core.engine import FinalTranscript, OwnerVerification
from core.llm import (
    HedgeLLM,
    SensitivityRouterLLM,
    capability_context,
)
from core.resume import DEFAULT_RESUME_TEMPLATE
from core.routing import FAST, MAIN
from core.runtime import VoiceRuntime
from core.websearch import WebSearchConfig, attach_web_search_capability


_CANARY = "retained-canary-kappa-731"
_RAW_CANARY = "raw-turn-canary-omega-284"
_SYNTHETIC_RESUME_PROMPT = DEFAULT_RESUME_TEMPLATE.format(
    query="Describe the synthetic cobalt lattice",
    spoken_tail=f"retained assistant tail {_CANARY}",
)


class _ProbeLLM:
    """Deterministic provider fake that records entry before yielding."""

    def __init__(
        self,
        replies: tuple[str, ...] = ("ok",),
        *,
        error: str | None = None,
    ) -> None:
        self._replies = list(replies)
        self.error = error
        self.calls: list[dict[str, object]] = []
        self._lock = threading.Lock()

    def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        images=None,
        history=None,
    ) -> Iterator[str]:
        with self._lock:
            reply = self._replies.pop(0) if self._replies else "ok"
            current_context = capability_context.get()
            recorded_context = (
                dict.copy(current_context)
                if type(current_context) is dict
                else current_context
            )
            self.calls.append(
                {
                    "prompt": prompt,
                    "system": system,
                    "images": images,
                    "history": history,
                    "context": recorded_context,
                }
            )
        if self.error is not None:
            raise RuntimeError(self.error)
        if reply:
            yield reply

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        images=None,
        history=None,
    ) -> str:
        return "".join(
            self.stream(
                prompt,
                system=system,
                images=images,
                history=history,
            )
        )


class _FixedRouter:
    def __init__(self, tier: str) -> None:
        self.tier = tier

    def choose(self, query: str, context) -> str:
        return self.tier


class _Memory:
    """Small duck-typed memory whose one retained source is configurable."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.items: list[MemoryItem] = []
        if source in {"recent_text", "recent_roles"}:
            self.items.extend(
                (
                    MemoryItem(f"Earlier question {_CANARY}", ("user",)),
                    MemoryItem("Earlier synthetic answer", ("assistant_output",)),
                )
            )

    def add(self, text: str, tags: tuple[str, ...] = ()) -> None:
        self.items.append(MemoryItem(text, tags))

    def all(self):
        return list(self.items)

    def search(self, query: str, limit: int = 5):
        return []

    def context_for_llm(self, query: str) -> str:
        if self.source == "recall":
            return f"Recall: {_CANARY}"
        if self.source == "screen_recall":
            return f"{VISION_LABEL} synthetic screen {_CANARY}"
        return ""

    def profile_block(self) -> str:
        return f"Profile: {_CANARY}" if self.source == "profile" else ""

    def last_session_summary(self) -> str:
        return f"Prior session {_CANARY}" if self.source == "last_session" else ""

    def procedural_rules(self):
        return (
            [f"Use the synthetic style {_CANARY}"]
            if self.source == "procedural"
            else []
        )

    def prune(self) -> int:
        return 0

    def close(self) -> None:
        return None


def _install_cloud_fakes(
    monkeypatch: pytest.MonkeyPatch,
    clouds: dict[str, _ProbeLLM],
) -> None:
    monkeypatch.setattr(
        llm_factory,
        "OpenAICompatLLM",
        lambda *args, **kwargs: clouds["single"],
    )
    monkeypatch.setattr(
        llm_factory,
        "_build_cloud_client",
        lambda name, preset, **kwargs: clouds[name],
    )


def _factory_managed(
    monkeypatch: pytest.MonkeyPatch,
    *,
    topology: str,
    strategy: str,
    local: _ProbeLLM,
    cloud: _ProbeLLM,
):
    clouds = {"single": cloud, "provider_a": cloud}
    _install_cloud_fakes(monkeypatch, clouds)
    cloud_cfg = {
        "enabled": True,
        "strategy": strategy,
        "model": "fake-cloud",
        "hedge_delay_ms": 0,
        "ttft_deadline_ms": 50,
    }
    config: dict[str, object] = {"cloud": cloud_cfg}
    if topology == "multi":
        config.update(
            {
                "cloud_providers": {"provider_a": {"model": "fake-cloud"}},
                "cloud_chains": {"private": ["provider_a"]},
                "cloud_routing": {
                    "default_chain": "private",
                    "sensitivity_to_chain": {"private": "private"},
                },
            }
        )
    wrapped = llm_factory._wrap_cloud(local, config)
    if topology == "single":
        assert isinstance(wrapped, HedgeLLM)
    else:
        assert isinstance(wrapped, SensitivityRouterLLM)
    return wrapped


def _call_with_scope(model, scope: object, *, generate: bool = False) -> str:
    token = capability_context.set(
        {
            "sensitivity": "private",
            CLOUD_EGRESS_SCOPE_CONTEXT_KEY: scope,
        }
    )
    try:
        if generate:
            return model.generate("synthetic query")
        return "".join(model.stream("synthetic query"))
    finally:
        capability_context.reset(token)


@pytest.mark.parametrize("topology", ("single", "multi"))
@pytest.mark.parametrize("strategy", ("hedge", "fallback"))
def test_factory_managed_local_only_never_starts_cloud(
    monkeypatch: pytest.MonkeyPatch,
    topology: str,
    strategy: str,
) -> None:
    local = _ProbeLLM(("local",))
    cloud = _ProbeLLM(("cloud",))
    model = _factory_managed(
        monkeypatch,
        topology=topology,
        strategy=strategy,
        local=local,
        cloud=cloud,
    )

    assert _call_with_scope(model, CloudEgressScope.LOCAL_ONLY) == "local"
    assert len(local.calls) == 1
    assert cloud.calls == []


@pytest.mark.parametrize("topology", ("single", "multi"))
@pytest.mark.parametrize("strategy", ("hedge", "fallback"))
def test_factory_managed_current_turn_only_preserves_cloud(
    monkeypatch: pytest.MonkeyPatch,
    topology: str,
    strategy: str,
) -> None:
    # An empty local stream makes the cloud winner deterministic even for hedge.
    local = _ProbeLLM(("",))
    cloud = _ProbeLLM(("cloud",))
    model = _factory_managed(
        monkeypatch,
        topology=topology,
        strategy=strategy,
        local=local,
        cloud=cloud,
    )

    assert _call_with_scope(model, CloudEgressScope.CURRENT_TURN_ONLY) == "cloud"
    assert len(cloud.calls) == 1


def test_multi_chain_local_only_never_starts_any_cloud_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = _ProbeLLM(("local",))
    cloud_a = _ProbeLLM(("cloud-a",), error="cloud-a started")
    cloud_b = _ProbeLLM(("cloud-b",), error="cloud-b started")
    _install_cloud_fakes(
        monkeypatch,
        {
            "single": cloud_a,
            "provider_a": cloud_a,
            "provider_b": cloud_b,
        },
    )
    model = llm_factory._wrap_cloud(
        local,
        {
            "cloud": {
                "enabled": True,
                "strategy": "hedge",
                "hedge_delay_ms": 0,
                "ttft_deadline_ms": 50,
            },
            "cloud_providers": {
                "provider_a": {"model": "fake-cloud-a"},
                "provider_b": {"model": "fake-cloud-b"},
            },
            "cloud_chains": {"private": ["provider_a", "provider_b"]},
            "cloud_routing": {
                "default_chain": "private",
                "sensitivity_to_chain": {"private": "private"},
            },
        },
    )
    assert isinstance(model, SensitivityRouterLLM)

    assert _call_with_scope(model, CloudEgressScope.LOCAL_ONLY) == "local"
    assert len(local.calls) == 1
    assert cloud_a.calls == []
    assert cloud_b.calls == []


@pytest.mark.parametrize("strategy", ("hedge", "fallback"))
@pytest.mark.parametrize("local_reply,error", (("", None), ("unused", "local failed")))
def test_local_failure_or_empty_under_veto_never_falls_through_to_cloud(
    monkeypatch: pytest.MonkeyPatch,
    strategy: str,
    local_reply: str,
    error: str | None,
) -> None:
    local = _ProbeLLM((local_reply,), error=error)
    cloud = _ProbeLLM(("cloud",))
    model = _factory_managed(
        monkeypatch,
        topology="single",
        strategy=strategy,
        local=local,
        cloud=cloud,
    )

    if error is None:
        assert _call_with_scope(model, CloudEgressScope.LOCAL_ONLY) == ""
    else:
        with pytest.raises(RuntimeError, match="local failed"):
            _call_with_scope(model, CloudEgressScope.LOCAL_ONLY)
    assert cloud.calls == []


def test_generate_inherits_local_only_veto(monkeypatch: pytest.MonkeyPatch) -> None:
    local = _ProbeLLM(("local",))
    cloud = _ProbeLLM(("cloud",))
    model = _factory_managed(
        monkeypatch,
        topology="single",
        strategy="fallback",
        local=local,
        cloud=cloud,
    )

    assert (
        _call_with_scope(
            model,
            CloudEgressScope.LOCAL_ONLY,
            generate=True,
        )
        == "local"
    )
    assert cloud.calls == []


def test_direct_hedge_without_enforcement_retains_missing_scope_compatibility() -> None:
    local = _ProbeLLM(("",))
    cloud = _ProbeLLM(("cloud",))
    model = HedgeLLM(
        local=local,
        cloud=cloud,
        strategy="fallback",
        ttft_deadline_ms=50,
    )
    token = capability_context.set({})
    try:
        assert "".join(model.stream("q")) == "cloud"
    finally:
        capability_context.reset(token)
    assert len(cloud.calls) == 1


def test_direct_hedge_without_enforcement_still_honors_exact_local_only() -> None:
    local = _ProbeLLM(("local",))
    cloud = _ProbeLLM(("cloud",))
    model = HedgeLLM(
        local=local,
        cloud=cloud,
        strategy="fallback",
        ttft_deadline_ms=50,
    )

    assert _call_with_scope(model, CloudEgressScope.LOCAL_ONLY) == "local"
    assert len(local.calls) == 1
    assert cloud.calls == []


class _HostileDict(dict):
    def __init__(self) -> None:
        dict.__init__(
            self,
            {CLOUD_EGRESS_SCOPE_CONTEXT_KEY: CloudEgressScope.CURRENT_TURN_ONLY},
        )
        self.hooks = 0

    def _touch(self, *args, **kwargs):
        self.hooks += 1
        raise AssertionError("virtual mapping hook invoked")

    __iter__ = _touch
    __getitem__ = _touch
    get = _touch
    items = _touch
    keys = _touch
    copy = _touch


class _HostileValue:
    def __init__(self) -> None:
        self.comparisons = 0

    def __eq__(self, other):
        self.comparisons += 1
        raise AssertionError("scope value equality hook invoked")


class _NonExactKey(str):
    pass


@pytest.mark.parametrize(
    "context_factory",
    (
        lambda: {},
        lambda: {CLOUD_EGRESS_SCOPE_CONTEXT_KEY: None},
        lambda: {CLOUD_EGRESS_SCOPE_CONTEXT_KEY: "local_only"},
        lambda: {CLOUD_EGRESS_SCOPE_CONTEXT_KEY: "current_turn_only"},
        lambda: {CLOUD_EGRESS_SCOPE_CONTEXT_KEY: "unknown"},
        lambda: {
            _NonExactKey(
                CLOUD_EGRESS_SCOPE_CONTEXT_KEY
            ): CloudEgressScope.CURRENT_TURN_ONLY
        },
    ),
    ids=(
        "missing",
        "none",
        "plain-local-string",
        "plain-current-string",
        "unknown-string",
        "nonexact-key",
    ),
)
@pytest.mark.parametrize("topology", ("single", "multi"))
@pytest.mark.parametrize("strategy", ("hedge", "fallback"))
def test_missing_malformed_or_nonexact_scope_fails_local(
    monkeypatch: pytest.MonkeyPatch,
    context_factory,
    topology: str,
    strategy: str,
) -> None:
    local = _ProbeLLM(("local",))
    cloud = _ProbeLLM(("cloud",))
    model = _factory_managed(
        monkeypatch,
        topology=topology,
        strategy=strategy,
        local=local,
        cloud=cloud,
    )
    context = context_factory()
    token = capability_context.set(context)
    try:
        assert "".join(model.stream("q")) == "local"
    finally:
        capability_context.reset(token)
    assert cloud.calls == []


@pytest.mark.parametrize("topology", ("single", "multi"))
@pytest.mark.parametrize("strategy", ("hedge", "fallback"))
def test_veto_precedes_queue_thread_and_cloud_stream_construction(
    monkeypatch: pytest.MonkeyPatch,
    topology: str,
    strategy: str,
) -> None:
    local = _ProbeLLM(("local",))
    cloud = _ProbeLLM(("cloud",))
    model = _factory_managed(
        monkeypatch,
        topology=topology,
        strategy=strategy,
        local=local,
        cloud=cloud,
    )

    class _ConstructionTrap:
        def __getattr__(self, name: str):
            raise AssertionError(f"cloud orchestration constructed {name}")

    monkeypatch.setattr(llm_module, "queue", _ConstructionTrap())
    monkeypatch.setattr(llm_module, "threading", _ConstructionTrap())

    assert _call_with_scope(model, CloudEgressScope.LOCAL_ONLY) == "local"
    assert len(local.calls) == 1
    assert cloud.calls == []


def test_hostile_context_and_scope_value_fail_local_without_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = _ProbeLLM(("local", "local"))
    cloud = _ProbeLLM(("cloud", "cloud"))
    model = _factory_managed(
        monkeypatch,
        topology="single",
        strategy="fallback",
        local=local,
        cloud=cloud,
    )

    hostile_context = _HostileDict()
    token = capability_context.set(hostile_context)
    try:
        assert "".join(model.stream("q")) == "local"
    finally:
        capability_context.reset(token)
    assert hostile_context.hooks == 0

    hostile_value = _HostileValue()
    token = capability_context.set({CLOUD_EGRESS_SCOPE_CONTEXT_KEY: hostile_value})
    try:
        assert "".join(model.stream("q")) == "local"
    finally:
        capability_context.reset(token)
    assert hostile_value.comparisons == 0
    assert cloud.calls == []


def test_eager_snapshot_cannot_be_upgraded_before_cross_thread_consumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = _ProbeLLM(("local",))
    cloud = _ProbeLLM(("cloud",))
    model = _factory_managed(
        monkeypatch,
        topology="single",
        strategy="fallback",
        local=local,
        cloud=cloud,
    )
    context = {CLOUD_EGRESS_SCOPE_CONTEXT_KEY: CloudEgressScope.LOCAL_ONLY}
    token = capability_context.set(context)
    try:
        stream = model.stream("q")
    finally:
        capability_context.reset(token)

    # Mutating the caller-owned dict after stream() returns cannot upgrade the
    # exact call-time snapshot captured for a different consumer thread.
    context[CLOUD_EGRESS_SCOPE_CONTEXT_KEY] = CloudEgressScope.CURRENT_TURN_ONLY
    result: list[str] = []
    worker = threading.Thread(target=lambda: result.append("".join(stream)))
    worker.start()
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert result == ["local"]
    assert cloud.calls == []


def test_concurrent_scopes_on_reused_factory_wrapper_are_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = _ProbeLLM(("local", "local"))
    cloud = _ProbeLLM(("cloud", "cloud"))
    model = _factory_managed(
        monkeypatch,
        topology="single",
        strategy="fallback",
        local=local,
        cloud=cloud,
    )
    barrier = threading.Barrier(2)
    results: dict[str, str] = {}

    def run(label: str, scope: CloudEgressScope) -> None:
        token = capability_context.set(
            {CLOUD_EGRESS_SCOPE_CONTEXT_KEY: scope, "sensitivity": "private"}
        )
        try:
            barrier.wait()
            results[label] = "".join(model.stream(label))
        finally:
            capability_context.reset(token)

    local_thread = threading.Thread(
        target=run,
        args=("vetoed", CloudEgressScope.LOCAL_ONLY),
    )
    cloud_thread = threading.Thread(
        target=run,
        args=("raw", CloudEgressScope.CURRENT_TURN_ONLY),
    )
    local_thread.start()
    cloud_thread.start()
    local_thread.join(timeout=2.0)
    cloud_thread.join(timeout=2.0)

    assert not local_thread.is_alive() and not cloud_thread.is_alive()
    assert results == {"vetoed": "local", "raw": "cloud"}
    assert len(cloud.calls) == 1


def _flatten_call(call: dict[str, object]) -> str:
    return " ".join(
        (
            str(call.get("prompt") or ""),
            str(call.get("system") or ""),
            repr(call.get("history")),
        )
    )


@pytest.mark.parametrize(
    "source",
    (
        "recent_text",
        "recent_roles",
        "recall",
        "screen_recall",
        "profile",
        "last_session",
        "procedural",
    ),
)
def test_assistant_retained_context_sources_veto_cloud_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    local = _ProbeLLM(("local answer",))
    cloud = _ProbeLLM(("cloud answer",))
    main = _factory_managed(
        monkeypatch,
        topology="single",
        strategy="fallback",
        local=local,
        cloud=cloud,
    )
    registry = CapabilityRegistry()
    memory = _Memory(source)
    attach_llm_capabilities(
        registry,
        main,
        router=_FixedRouter(MAIN),
        memory=memory,
        recall=RecallConfig(
            enabled=source in {"recall", "screen_recall"},
            procedural_enabled=source == "procedural",
        ),
        recent_context=RecentContextConfig(
            enabled=source in {"recent_text", "recent_roles"},
            as_messages=source == "recent_roles",
        ),
    )
    context = {"mode": Mode.ASSISTANT.value}

    result = registry.invoke("assistant.answer", "Explain the cobalt lattice", context)

    assert result.ok and result.text == "local answer"
    assert cloud.calls == []
    assert len(local.calls) == 1
    assert _CANARY in _flatten_call(local.calls[0])
    assert context[CLOUD_EGRESS_SCOPE_CONTEXT_KEY] is CloudEgressScope.LOCAL_ONLY
    assert capability_context.get() == {}


def test_prepublished_legacy_recent_block_vetoes_cloud() -> None:
    local = _ProbeLLM(("local answer",))
    cloud = _ProbeLLM(("cloud answer",))
    main = HedgeLLM(
        local=local,
        cloud=cloud,
        strategy="fallback",
        ttft_deadline_ms=50,
        enforce_cloud_egress_scope=True,
    )
    registry = CapabilityRegistry()
    attach_llm_capabilities(
        registry,
        main,
        router=_FixedRouter(MAIN),
        memory=None,
        recent_context=RecentContextConfig(enabled=False),
    )
    context = {
        "mode": Mode.ASSISTANT.value,
        "recent_conversation": f"Earlier synthetic turn {_CANARY}",
    }

    result = registry.invoke("assistant.answer", _RAW_CANARY, context)

    assert result.ok and result.text == "local answer"
    assert cloud.calls == []
    assert context[CLOUD_EGRESS_SCOPE_CONTEXT_KEY] is CloudEgressScope.LOCAL_ONLY


@pytest.mark.parametrize(
    "initial_scope",
    (CloudEgressScope.LOCAL_ONLY, "current_turn_only", "unknown"),
    ids=("local-only", "raw-current-string", "unknown-string"),
)
def test_capability_entry_never_upgrades_a_present_scope(initial_scope: object) -> None:
    local = _ProbeLLM(("local answer",))
    cloud = _ProbeLLM(("cloud answer",))
    main = HedgeLLM(
        local=local,
        cloud=cloud,
        strategy="fallback",
        ttft_deadline_ms=50,
        enforce_cloud_egress_scope=True,
    )
    registry = CapabilityRegistry()
    attach_llm_capabilities(
        registry,
        main,
        router=_FixedRouter(MAIN),
        memory=None,
        recent_context=RecentContextConfig(enabled=False),
    )
    context = {
        "mode": Mode.ASSISTANT.value,
        CLOUD_EGRESS_SCOPE_CONTEXT_KEY: initial_scope,
    }

    result = registry.invoke("assistant.answer", _RAW_CANARY, context)

    assert result.ok and result.text == "local answer"
    assert cloud.calls == []
    assert context[CLOUD_EGRESS_SCOPE_CONTEXT_KEY] is initial_scope


def test_fast_failure_main_retry_retains_memory_veto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = _ProbeLLM(("main local answer",))
    cloud = _ProbeLLM(("main cloud answer",))
    main = _factory_managed(
        monkeypatch,
        topology="single",
        strategy="fallback",
        local=local,
        cloud=cloud,
    )
    fast = _ProbeLLM(error="fast failed")
    registry = CapabilityRegistry()
    attach_llm_capabilities(
        registry,
        main,
        fast_llm=fast,
        router=_FixedRouter(FAST),
        memory=_Memory("profile"),
        recent_context=RecentContextConfig(enabled=False),
    )
    context = {"mode": Mode.ASSISTANT.value}

    result = registry.invoke("assistant.answer", "Explain the cobalt lattice", context)

    assert result.ok and result.text == "main local answer"
    assert len(fast.calls) == 1
    assert (
        fast.calls[0]["context"][CLOUD_EGRESS_SCOPE_CONTEXT_KEY]
        is CloudEgressScope.LOCAL_ONLY
    )
    assert cloud.calls == []
    assert context[CLOUD_EGRESS_SCOPE_CONTEXT_KEY] is CloudEgressScope.LOCAL_ONLY


def test_reused_assistant_context_is_never_upgraded_from_local_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = _ProbeLLM(("local first", "local second"))
    cloud = _ProbeLLM(("cloud",))
    main = _factory_managed(
        monkeypatch,
        topology="single",
        strategy="fallback",
        local=local,
        cloud=cloud,
    )
    registry = CapabilityRegistry()
    memory = _Memory("profile")
    attach_llm_capabilities(
        registry,
        main,
        router=_FixedRouter(MAIN),
        memory=memory,
        recent_context=RecentContextConfig(enabled=False),
    )
    reused = {"mode": Mode.ASSISTANT.value, "sensitivity": "private"}

    first = registry.invoke("assistant.answer", "Explain the cobalt lattice", reused)
    assert first.ok and first.text == "local first"
    assert reused[CLOUD_EGRESS_SCOPE_CONTEXT_KEY] is CloudEgressScope.LOCAL_ONLY

    # The same caller-owned context is deliberately conservative on reuse: even
    # after its retained source disappears, entry must not upgrade the old veto.
    memory.source = "none"
    second = registry.invoke("assistant.answer", _RAW_CANARY, reused)

    assert second.ok and second.text == "local second"
    assert cloud.calls == []
    assert reused[CLOUD_EGRESS_SCOPE_CONTEXT_KEY] is CloudEgressScope.LOCAL_ONLY


def test_raw_current_private_turn_without_injected_context_remains_cloud_eligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = _ProbeLLM(("local",))
    cloud = _ProbeLLM(("cloud",))
    main = _factory_managed(
        monkeypatch,
        topology="single",
        strategy="fallback",
        local=local,
        cloud=cloud,
    )
    registry = CapabilityRegistry()
    attach_llm_capabilities(
        registry,
        main,
        router=_FixedRouter(MAIN),
        memory=None,
        recent_context=RecentContextConfig(enabled=False),
    )
    context = {"mode": Mode.ASSISTANT.value, "sensitivity": "private"}

    result = registry.invoke("assistant.answer", _RAW_CANARY, context)

    assert result.ok and result.text == "cloud"
    assert len(cloud.calls) == 1
    assert cloud.calls[0]["prompt"] == _RAW_CANARY
    assert (
        cloud.calls[0]["context"][CLOUD_EGRESS_SCOPE_CONTEXT_KEY]
        is CloudEgressScope.CURRENT_TURN_ONLY
    )
    assert context[CLOUD_EGRESS_SCOPE_CONTEXT_KEY] is CloudEgressScope.CURRENT_TURN_ONLY


def test_current_turn_image_remains_cloud_eligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = _ProbeLLM(("local",))
    cloud = _ProbeLLM(("cloud",))
    main = _factory_managed(
        monkeypatch,
        topology="single",
        strategy="fallback",
        local=local,
        cloud=cloud,
    )
    registry = CapabilityRegistry()
    attach_llm_capabilities(
        registry,
        main,
        router=_FixedRouter(MAIN),
        memory=None,
        recent_context=RecentContextConfig(enabled=False),
    )
    current_image = b"synthetic-current-turn-image"
    context = {
        "mode": Mode.ASSISTANT.value,
        "sensitivity": "private",
        "images": [current_image],
    }

    result = registry.invoke("assistant.answer", _RAW_CANARY, context)

    assert result.ok and result.text == "cloud"
    assert local.calls == []
    assert len(cloud.calls) == 1
    assert cloud.calls[0]["images"] == [current_image]
    assert context[CLOUD_EGRESS_SCOPE_CONTEXT_KEY] is CloudEgressScope.CURRENT_TURN_ONLY


def _response_only_context(*, synthetic_resume_tail: bool) -> dict[str, object]:
    metadata: dict[str, object] = {
        "post_barge_response_only": True,
        "skip_user_memory": True,
    }
    if synthetic_resume_tail:
        metadata[SYNTHETIC_RESUME_TAIL_METADATA_KEY] = True
    return {
        "mode": Mode.ASSISTANT.value,
        "sensitivity": "private",
        "metadata": metadata,
    }


def test_synthetic_resume_tail_vetoes_collapsed_guarded_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = _ProbeLLM(("local continuation",))
    cloud = _ProbeLLM(("cloud continuation",))
    main = _factory_managed(
        monkeypatch,
        topology="single",
        strategy="fallback",
        local=local,
        cloud=cloud,
    )
    registry = CapabilityRegistry()
    attach_llm_capabilities(
        registry,
        main,
        memory=None,
        recent_context=RecentContextConfig(enabled=False),
    )
    context = _response_only_context(synthetic_resume_tail=True)

    result = registry.invoke("assistant.answer", _SYNTHETIC_RESUME_PROMPT, context)

    assert result.ok and result.text == "local continuation"
    assert cloud.calls == []
    assert len(local.calls) == 1
    assert _CANARY in str(local.calls[0]["prompt"])
    assert context[CLOUD_EGRESS_SCOPE_CONTEXT_KEY] is CloudEgressScope.LOCAL_ONLY


def test_synthetic_resume_tail_survives_fast_failure_main_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = _ProbeLLM(("local continuation",))
    cloud = _ProbeLLM(("cloud continuation",))
    main = _factory_managed(
        monkeypatch,
        topology="single",
        strategy="fallback",
        local=local,
        cloud=cloud,
    )
    fast = _ProbeLLM(error="fast failed")
    registry = CapabilityRegistry()
    attach_llm_capabilities(
        registry,
        main,
        fast_llm=fast,
        memory=None,
        recent_context=RecentContextConfig(enabled=False),
    )
    context = _response_only_context(synthetic_resume_tail=True)

    result = registry.invoke("assistant.answer", _SYNTHETIC_RESUME_PROMPT, context)

    assert result.ok and result.text == "local continuation"
    assert len(fast.calls) == 1
    assert (
        fast.calls[0]["context"][CLOUD_EGRESS_SCOPE_CONTEXT_KEY]
        is CloudEgressScope.LOCAL_ONLY
    )
    assert cloud.calls == []
    assert len(local.calls) == 1
    assert _CANARY in str(local.calls[0]["prompt"])
    assert context[CLOUD_EGRESS_SCOPE_CONTEXT_KEY] is CloudEgressScope.LOCAL_ONLY


def test_generic_response_only_without_resume_tail_marker_remains_eligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = _ProbeLLM(("",))
    cloud = _ProbeLLM(("cloud response",))
    main = _factory_managed(
        monkeypatch,
        topology="single",
        strategy="fallback",
        local=local,
        cloud=cloud,
    )
    registry = CapabilityRegistry()
    attach_llm_capabilities(
        registry,
        main,
        memory=None,
        recent_context=RecentContextConfig(enabled=False),
    )
    context = _response_only_context(synthetic_resume_tail=False)

    result = registry.invoke("assistant.answer", _RAW_CANARY, context)

    assert result.ok and result.text == "cloud response"
    assert len(cloud.calls) == 1
    assert (
        cloud.calls[0]["context"][CLOUD_EGRESS_SCOPE_CONTEXT_KEY]
        is CloudEgressScope.CURRENT_TURN_ONLY
    )
    assert context[CLOUD_EGRESS_SCOPE_CONTEXT_KEY] is CloudEgressScope.CURRENT_TURN_ONLY


def _retained_prompt_context(
    *,
    mode: Mode = Mode.ASSISTANT,
    marker: object = True,
) -> dict[str, object]:
    return {
        "mode": mode.value,
        "sensitivity": "public",
        "metadata": {RETAINED_PROMPT_CONTEXT_METADATA_KEY: marker},
    }


@pytest.mark.parametrize("marker", (True, False, "true", None))
def test_retained_prompt_marker_vetoes_collapsed_guarded_main(
    monkeypatch: pytest.MonkeyPatch,
    marker: object,
) -> None:
    local = _ProbeLLM(("local cleaned answer",))
    cloud = _ProbeLLM(("cloud leak",))
    main = _factory_managed(
        monkeypatch,
        topology="single",
        strategy="fallback",
        local=local,
        cloud=cloud,
    )
    registry = CapabilityRegistry()
    attach_llm_capabilities(
        registry,
        main,
        memory=None,
        recent_context=RecentContextConfig(enabled=False),
    )
    context = _retained_prompt_context(marker=marker)

    result = registry.invoke("assistant.answer", _RAW_CANARY, context)

    assert result.ok and result.text == "local cleaned answer"
    assert cloud.calls == []
    assert len(local.calls) == 1
    assert (
        local.calls[0]["context"][CLOUD_EGRESS_SCOPE_CONTEXT_KEY]
        is CloudEgressScope.LOCAL_ONLY
    )
    assert context[CLOUD_EGRESS_SCOPE_CONTEXT_KEY] is CloudEgressScope.LOCAL_ONLY


def test_retained_prompt_marker_survives_fast_failure_main_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = _ProbeLLM(("local retry answer",))
    cloud = _ProbeLLM(("cloud leak",))
    main = _factory_managed(
        monkeypatch,
        topology="single",
        strategy="fallback",
        local=local,
        cloud=cloud,
    )
    fast = _ProbeLLM(error="fast failed")
    registry = CapabilityRegistry()
    attach_llm_capabilities(
        registry,
        main,
        fast_llm=fast,
        router=_FixedRouter(FAST),
        memory=None,
        recent_context=RecentContextConfig(enabled=False),
    )
    context = _retained_prompt_context()

    result = registry.invoke("assistant.answer", _RAW_CANARY, context)

    assert result.ok and result.text == "local retry answer"
    assert len(fast.calls) == 1
    assert (
        fast.calls[0]["context"][CLOUD_EGRESS_SCOPE_CONTEXT_KEY]
        is CloudEgressScope.LOCAL_ONLY
    )
    assert cloud.calls == []
    assert len(local.calls) == 1


def test_retained_prompt_marker_vetoes_research_after_public_web_finding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = _ProbeLLM(("local synthesis",))
    cloud = _ProbeLLM(("cloud leak",))
    main = _factory_managed(
        monkeypatch,
        topology="single",
        strategy="fallback",
        local=local,
        cloud=cloud,
    )
    registry = CapabilityRegistry()
    attach_llm_capabilities(registry, main, memory=None)
    context = _retained_prompt_context(mode=Mode.RESEARCH)
    context["previous_steps"] = [
        {
            "text": "public web finding",
            "data": {"sensitivity": "public", "egress": True},
        }
    ]

    result = registry.invoke("research.local", "summarize the finding", context)

    assert result.ok and result.text == "local synthesis"
    assert cloud.calls == []
    assert len(local.calls) == 1
    assert context[CLOUD_EGRESS_SCOPE_CONTEXT_KEY] is CloudEgressScope.LOCAL_ONLY


def test_contextful_unicode_cleaner_rewrite_is_local_and_marker_is_one_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = SessionMemory()
    prior = f"prior user context 秘密 {_CANARY}"
    memory.add(prior, tags=("user",))
    raw = "tell me um about mars"
    cleaned = f"{raw} 秘密"
    cleaner = ScriptedTranscriptCleaner({raw: cleaned})
    local = _ProbeLLM(("local guarded answer",))
    cloud = _ProbeLLM(("cloud next answer",))
    main = _factory_managed(
        monkeypatch,
        topology="single",
        strategy="fallback",
        local=local,
        cloud=cloud,
    )
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        main,
        memory=memory,
        recall_config=RecallConfig(enabled=False),
        recent_context_config=RecentContextConfig(enabled=False),
        cleaner=cleaner,
    )
    runtime.start(run_bus=False)
    try:
        engine.final_result(
            FinalTranscript(
                raw,
                owner_verification=OwnerVerification.VERIFIED,
                origin="live_audio",
            )
        )
        assert runtime.wait_idle(timeout=2.0)

        assert cloud.calls == []
        assert len(local.calls) == 1
        assert cleaner.calls == [(raw, (prior,))]
        assert cleaned in str(local.calls[0]["prompt"])
        first_context = local.calls[0]["context"]
        assert (
            first_context[CLOUD_EGRESS_SCOPE_CONTEXT_KEY] is CloudEgressScope.LOCAL_ONLY
        )
        first_metadata = first_context["metadata"]
        assert type(first_metadata) is dict
        assert first_metadata[RETAINED_PROMPT_CONTEXT_METADATA_KEY] is True
        assert first_context["owner_verified"] is False
        assert first_context["direct_user_instruction"] is False
        assert (
            RETAINED_PROMPT_CONTEXT_METADATA_KEY
            not in runtime.supervisor.state.turn_metadata
        )

        engine.final("what is the weather in Berlin")
        assert runtime.wait_idle(timeout=2.0)

        assert len(local.calls) == 1
        assert len(cloud.calls) == 1
        second_context = cloud.calls[0]["context"]
        assert (
            second_context[CLOUD_EGRESS_SCOPE_CONTEXT_KEY]
            is CloudEgressScope.CURRENT_TURN_ONLY
        )
        second_metadata = second_context["metadata"]
        assert type(second_metadata) is dict
        assert RETAINED_PROMPT_CONTEXT_METADATA_KEY not in second_metadata
    finally:
        runtime.stop()


def test_contextful_case_spacing_and_ascii_punctuation_cleanup_stays_eligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = SessionMemory()
    memory.add(f"prior user context {_CANARY}", tags=("user",))
    raw = "HELLO, spaced WORLD!"
    cleaned = "hello spaced world"
    local = _ProbeLLM(("",))
    cloud = _ProbeLLM(("cloud answer",))
    main = _factory_managed(
        monkeypatch,
        topology="single",
        strategy="fallback",
        local=local,
        cloud=cloud,
    )
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        main,
        memory=memory,
        recall_config=RecallConfig(enabled=False),
        recent_context_config=RecentContextConfig(enabled=False),
        cleaner=ScriptedTranscriptCleaner({raw: cleaned}),
    )
    runtime.start(run_bus=False)
    try:
        engine.final(raw)
        assert runtime.wait_idle(timeout=2.0)

        assert len(cloud.calls) == 1
        context = cloud.calls[0]["context"]
        assert (
            context[CLOUD_EGRESS_SCOPE_CONTEXT_KEY]
            is CloudEgressScope.CURRENT_TURN_ONLY
        )
        metadata = context["metadata"]
        assert type(metadata) is dict
        assert RETAINED_PROMPT_CONTEXT_METADATA_KEY not in metadata
    finally:
        runtime.stop()


def test_context_free_token_changing_cleanup_stays_cloud_eligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = "tell me about mars mars"
    cleaned = "tell me about mars"
    local = _ProbeLLM(("",))
    cloud = _ProbeLLM(("cloud answer",))
    main = _factory_managed(
        monkeypatch,
        topology="single",
        strategy="fallback",
        local=local,
        cloud=cloud,
    )
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        main,
        recall_config=RecallConfig(enabled=False),
        recent_context_config=RecentContextConfig(enabled=False),
        cleaner=ScriptedTranscriptCleaner({raw: cleaned}),
    )
    runtime.start(run_bus=False)
    try:
        engine.final(raw)
        assert runtime.wait_idle(timeout=2.0)

        assert len(cloud.calls) == 1
        context = cloud.calls[0]["context"]
        assert (
            context[CLOUD_EGRESS_SCOPE_CONTEXT_KEY]
            is CloudEgressScope.CURRENT_TURN_ONLY
        )
        metadata = context["metadata"]
        assert type(metadata) is dict
        assert RETAINED_PROMPT_CONTEXT_METADATA_KEY not in metadata
    finally:
        runtime.stop()


def test_cleaner_derived_search_canary_never_reaches_web_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _WebTripwire:
        def __init__(self) -> None:
            self.calls = 0

        def search(self, query: str):
            self.calls += 1
            raise AssertionError(f"web backend received {query!r}")

    backend = _WebTripwire()
    monkeypatch.setattr(
        websearch_module,
        "SearxngBackend",
        lambda *args, **kwargs: backend,
    )
    canary = "cleaner-canary-kappa731"
    memory = SessionMemory()
    memory.add(f"prior user supplied {canary}", tags=("user",))
    raw = "search web for weather in placeholder"
    cleaned = f"search web for weather in {canary}"
    cleaner = ScriptedTranscriptCleaner({raw: cleaned})
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        _ProbeLLM(("unused",)),
        memory=memory,
        recall_config=RecallConfig(enabled=False),
        recent_context_config=RecentContextConfig(enabled=False),
        cleaner=cleaner,
        web_search_config=WebSearchConfig(
            enabled=True,
            base_url="http://synthetic.invalid",
        ),
    )
    runtime.start(run_bus=False)
    try:
        engine.final(raw)
        assert runtime.wait_idle(timeout=2.0)

        assert runtime.supervisor.state.decisions[-1].kind is IntentKind.SEARCH
        assert cleaner.calls == [(raw, (f"prior user supplied {canary}",))]
        assert backend.calls == 0
        assert (
            RETAINED_PROMPT_CONTEXT_METADATA_KEY
            not in runtime.supervisor.state.turn_metadata
        )
    finally:
        runtime.stop()


def test_react_retained_recent_scope_blocks_public_web_tool_egress() -> None:
    class _WebTripwire:
        def __init__(self) -> None:
            self.calls = 0

        def search(self, query: str):
            self.calls += 1
            raise AssertionError(f"web backend received {query!r}")

    backend = _WebTripwire()
    registry = create_default_capabilities()
    attach_web_search_capability(
        registry,
        WebSearchConfig(enabled=True, base_url="http://synthetic.invalid"),
        backend=backend,
    )
    planner_model = _ProbeLLM(
        (
            "TOOL web.search: weather in Berlin",
            "FINAL: local corpus conclusion",
        )
    )
    planner = ReactPlanner(
        planner_model,
        registry,
        max_steps=2,
        tools=("web.search",),
    )
    context = {
        "sensitivity": "public",
        "recent_conversation": f"retained recent {_CANARY}",
        CLOUD_EGRESS_SCOPE_CONTEXT_KEY: CloudEgressScope.LOCAL_ONLY,
    }
    token = capability_context.set(context)
    try:
        result = planner.run("research public weather", context)
    finally:
        capability_context.reset(token)

    assert result.ok and result.text == "local corpus conclusion"
    assert result.data["steps"] == ["web.search"]
    assert backend.calls == 0
    assert len(planner_model.calls) == 2
    assert _CANARY in str(planner_model.calls[0]["prompt"])


@pytest.mark.parametrize("tool_name", ("search.local", "vault.search"))
def test_react_private_local_finding_vetoes_the_next_model_call(
    monkeypatch: pytest.MonkeyPatch,
    tool_name: str,
) -> None:
    # First plan step is cloud-eligible and asks for a local tool. The private
    # tool result must make the second planning call local-only.
    local = _ProbeLLM(("FINAL: local synthesis",))
    cloud = _ProbeLLM((f"TOOL {tool_name}: synthetic topic",))
    model = _factory_managed(
        monkeypatch,
        topology="single",
        strategy="fallback",
        local=local,
        cloud=cloud,
    )
    registry = CapabilityRegistry()
    registry.register(
        tool_name,
        lambda query, context: CapabilityResult(
            True,
            f"local finding {_CANARY}",
            data={"sensitivity": "private", "egress": False},
        ),
        spec=CapabilitySpec(tool_name, "local search", planner_tool=True),
    )
    planner = ReactPlanner(model, registry, max_steps=2, tools=(tool_name,))
    context = {
        "sensitivity": "private",
        CLOUD_EGRESS_SCOPE_CONTEXT_KEY: CloudEgressScope.CURRENT_TURN_ONLY,
    }
    planner_query = (
        "search my vault for the synthetic topic"
        if tool_name == "vault.search"
        else "research the synthetic topic"
    )
    token = capability_context.set(context)
    try:
        result = planner.run(planner_query, context)
    finally:
        capability_context.reset(token)

    assert result.ok and result.text == "local synthesis"
    assert len(cloud.calls) == 1
    assert len(local.calls) == 1
    assert _CANARY in str(local.calls[0]["prompt"])
    assert (
        local.calls[0]["context"][CLOUD_EGRESS_SCOPE_CONTEXT_KEY]
        is CloudEgressScope.LOCAL_ONLY
    )
    assert context[CLOUD_EGRESS_SCOPE_CONTEXT_KEY] is CloudEgressScope.LOCAL_ONLY


def test_research_private_local_finding_vetoes_synthesis_cloud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = _ProbeLLM(("local synthesis",))
    cloud = _ProbeLLM(("cloud synthesis",))
    main = _factory_managed(
        monkeypatch,
        topology="single",
        strategy="fallback",
        local=local,
        cloud=cloud,
    )
    registry = CapabilityRegistry()
    attach_llm_capabilities(registry, main, memory=None)
    context = {
        "mode": Mode.RESEARCH.value,
        "previous_steps": [
            {
                "text": f"local finding {_CANARY}",
                "data": {"sensitivity": "private", "egress": False},
            }
        ],
    }

    result = registry.invoke("research.local", "summarize the finding", context)

    assert result.ok and result.text == "local synthesis"
    assert cloud.calls == []
    assert _CANARY in str(local.calls[0]["prompt"])
    assert (
        local.calls[0]["context"][CLOUD_EGRESS_SCOPE_CONTEXT_KEY]
        is CloudEgressScope.LOCAL_ONLY
    )
    assert context[CLOUD_EGRESS_SCOPE_CONTEXT_KEY] is CloudEgressScope.LOCAL_ONLY


def test_research_public_web_finding_does_not_downgrade_current_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = _ProbeLLM(("local synthesis",))
    cloud = _ProbeLLM(("cloud synthesis",))
    main = _factory_managed(
        monkeypatch,
        topology="single",
        strategy="fallback",
        local=local,
        cloud=cloud,
    )
    registry = CapabilityRegistry()
    attach_llm_capabilities(registry, main, memory=None)
    context = {
        "mode": Mode.RESEARCH.value,
        "previous_steps": [
            {
                "text": f"public web finding {_CANARY}",
                "data": {"sensitivity": "public", "egress": True},
            }
        ],
    }

    result = registry.invoke("research.local", "summarize the finding", context)

    assert result.ok and result.text == "cloud synthesis"
    assert local.calls == []
    assert len(cloud.calls) == 1
    assert context[CLOUD_EGRESS_SCOPE_CONTEXT_KEY] is CloudEgressScope.CURRENT_TURN_ONLY
