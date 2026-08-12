"""Deterministic contracts for process-wide Hedge source ownership.

These tests use only inert clients and controllable fake threads.  They never
construct a provider SDK client, open a socket, load a model, or touch an audio
device.  Timing-sensitive Hedge behavior is driven with scripted queues rather
than sleeps.
"""

from __future__ import annotations

from contextvars import ContextVar
import dataclasses
import gc
import queue
import weakref

import pytest

import core._hedge_source_owner as owner_module
import core.llm as llm_module
import core.llm_factory as llm_factory
from always_on_agent.models import (
    CLOUD_EGRESS_SCOPE_CONTEXT_KEY,
    CloudEgressScope,
)
from core._hedge_source_owner import (
    FACTORY_HEDGE_SOURCE_REGISTRY,
    FACTORY_LOCAL_MAIN_SOURCE,
    FACTORY_SINGLE_CLOUD_SOURCE,
    HedgeSourceKey,
    HedgeSourceLease,
    HedgeSourceLeaseState,
    HedgeSourceOwner,
    HedgeSourceOwnerState,
    HedgeSourceRegistry,
    HedgeSourceWork,
    factory_named_cloud_source,
    try_claim_hedge_source_lease,
    try_claim_hedge_source_owner,
)
from core.llm import HedgeLLM, SensitivityRouterLLM, capability_context


class _Payload(list):
    """Weak-referenceable sequence used to prove request alias clearing."""


class _Marker:
    pass


class _RecordingClient:
    def __init__(self, tokens=()) -> None:
        self.tokens = tuple(tokens)
        self.calls = 0
        self.seen: list[tuple[object, object, object, object]] = []

    def stream(self, prompt, *, system=None, images=None, history=None):
        self.calls += 1
        self.seen.append((prompt, system, images, history))
        yield from self.tokens


class _CloseFailStream:
    def __init__(self, tokens=("answer",)) -> None:
        self._tokens = iter(tokens)
        self.close_calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._tokens)

    def close(self) -> None:
        self.close_calls += 1
        raise RuntimeError("close failed")


class _CloseFailClient:
    def __init__(self) -> None:
        self.calls = 0
        self.streams: list[_CloseFailStream] = []

    def stream(self, prompt, *, system=None, images=None, history=None):
        self.calls += 1
        stream = _CloseFailStream()
        self.streams.append(stream)
        return stream


class _ControllableThread:
    """A thread-shaped object whose exact target is run only when requested."""

    def __init__(
        self,
        *,
        target,
        args,
        name,
        daemon,
        run_on_start: bool = False,
        start_error: BaseException | None = None,
        raise_after_run: BaseException | None = None,
        alive_error: BaseException | None = None,
        join_error: BaseException | None = None,
        on_start=None,
    ) -> None:
        self.target = target
        self.args = args
        self.name = name
        self.daemon = daemon
        self.run_on_start = run_on_start
        self.start_error = start_error
        self.raise_after_run = raise_after_run
        self.alive_error = alive_error
        self.join_error = join_error
        self.on_start = on_start
        self.start_calls = 0
        self.join_calls: list[float | None] = []
        self.alive = False

    def start(self) -> None:
        self.start_calls += 1
        if self.on_start is not None:
            self.on_start(self)
        if self.start_error is not None:
            raise self.start_error
        self.alive = True
        if self.run_on_start:
            self.run_target()
        if self.raise_after_run is not None:
            raise self.raise_after_run

    def run_target(self, *, remain_alive: bool = False) -> None:
        self.alive = True
        try:
            self.target(*self.args)
        finally:
            self.alive = remain_alive

    def is_alive(self) -> bool:
        if self.alive_error is not None:
            raise self.alive_error
        return self.alive

    def join(self, timeout=None) -> None:
        self.join_calls.append(timeout)
        if self.join_error is not None:
            raise self.join_error


class _ThreadFactory:
    def __init__(self, **thread_options) -> None:
        self.thread_options = thread_options
        self.threads: list[_ControllableThread] = []

    def __call__(self, **kwargs):
        options = dict(self.thread_options)
        on_create = options.pop("on_create", None)
        thread = _ControllableThread(**kwargs, **options)
        self.threads.append(thread)
        if on_create is not None:
            on_create(thread)
        return thread


class _FactoryFailure:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        raise RuntimeError("thread construction failed")


def _work(
    client=None,
    *,
    prompt="prompt",
    system="system",
    images=None,
    history=None,
    output=None,
    stop=None,
    context_var=None,
    turn_context=None,
    tag="local",
) -> HedgeSourceWork:
    return HedgeSourceWork(
        client if client is not None else _RecordingClient(),
        prompt,
        system,
        images,
        history,
        output if output is not None else queue.Queue(),
        stop if stop is not None else owner_module.threading.Event(),
        {} if turn_context is None else turn_context,
        context_var
        if context_var is not None
        else ContextVar(f"hedge-test-context-{id(client)}", default={}),
        tag,
    )


def _claim_immediate(
    registry: HedgeSourceRegistry,
    key: HedgeSourceKey,
    work: HedgeSourceWork | None = None,
) -> tuple[HedgeSourceOwner, _ControllableThread]:
    factory = _ThreadFactory(run_on_start=True)
    owner = try_claim_hedge_source_owner(
        key,
        _work() if work is None else work,
        registry=registry,
        thread_factory=factory,
    )
    assert owner is not None
    owner.start()
    return owner, factory.threads[0]


def _current_turn_context():
    return {
        CLOUD_EGRESS_SCOPE_CONTEXT_KEY: CloudEgressScope.CURRENT_TURN_ONLY,
    }


def _local_only_context():
    return {
        CLOUD_EGRESS_SCOPE_CONTEXT_KEY: CloudEgressScope.LOCAL_ONLY,
    }


# Exact identity and construction contracts ---------------------------------


def test_source_keys_require_exact_nonempty_strings_and_pin_namespaces():
    class StringSubclass(str):
        pass

    with pytest.raises(TypeError):
        HedgeSourceKey(StringSubclass("factory"), "LOCAL_MAIN")
    with pytest.raises(TypeError):
        HedgeSourceKey("factory", StringSubclass("LOCAL_MAIN"))
    with pytest.raises(TypeError):
        HedgeSourceKey("", "source")
    with pytest.raises(TypeError):
        factory_named_cloud_source(StringSubclass("named"))

    assert FACTORY_LOCAL_MAIN_SOURCE == HedgeSourceKey("factory", "LOCAL_MAIN")
    assert FACTORY_SINGLE_CLOUD_SOURCE == HedgeSourceKey("factory", "SINGLE_CLOUD")
    assert factory_named_cloud_source("named") == HedgeSourceKey(
        "factory.named_cloud", "named"
    )
    assert factory_named_cloud_source(" named ").source == " named "


def test_registry_and_owner_reject_key_work_and_registry_subclasses():
    class KeySubclass(HedgeSourceKey):
        pass

    class WorkSubclass(HedgeSourceWork):
        pass

    class RegistrySubclass(HedgeSourceRegistry):
        pass

    registry = HedgeSourceRegistry()
    with pytest.raises(TypeError):
        registry.current(KeySubclass("n", "s"))
    with pytest.raises(TypeError):
        HedgeSourceOwner(
            HedgeSourceKey("n", "s"),
            WorkSubclass(*dataclasses.astuple(_work())),
            registry=registry,
        )
    with pytest.raises(TypeError):
        HedgeSourceOwner(
            HedgeSourceKey("n", "s"),
            _work(),
            registry=RegistrySubclass(),
        )


def test_worker_payload_is_finite_and_thread_target_retains_only_owner():
    expected_fields = {
        "client",
        "prompt",
        "system",
        "images",
        "history",
        "output",
        "stop",
        "turn_context",
        "context_var",
        "tag",
    }
    assert {
        field.name for field in dataclasses.fields(HedgeSourceWork)
    } == expected_fields

    factory = _ThreadFactory()
    registry = HedgeSourceRegistry()
    key = HedgeSourceKey("test", "static-target")
    owner = try_claim_hedge_source_owner(
        key,
        _work(prompt="private prompt", tag="source_7"),
        registry=registry,
        thread_factory=factory,
    )
    assert owner is not None
    thread = factory.threads[0]
    assert thread.target is owner_module._run_hedge_source_owner
    assert thread.target.__closure__ is None
    assert thread.args == (owner,)
    assert thread.name == "speaker-hedge-source-source_7"
    assert thread.daemon is True


def test_provider_descriptor_is_first_touched_after_exact_owner_publication():
    registry = HedgeSourceRegistry()
    key = HedgeSourceKey("test", "hostile-client")

    class PublishedOnlyClient:
        def __init__(self) -> None:
            self.expected_owner = None
            self.stream_lookups = 0

        def __getattribute__(self, name):
            if name == "stream":
                expected = object.__getattribute__(self, "expected_owner")
                assert expected is not None
                assert registry.current(key) is expected
                object.__setattr__(
                    self,
                    "stream_lookups",
                    object.__getattribute__(self, "stream_lookups") + 1,
                )
            return object.__getattribute__(self, name)

        def stream(self, prompt, *, system=None, images=None, history=None):
            yield "safe"

    client = PublishedOnlyClient()
    output = queue.Queue()
    factory = _ThreadFactory(run_on_start=True)
    # Work validation and owner construction must not execute the descriptor.
    work = _work(client, output=output)
    owner = try_claim_hedge_source_owner(
        key,
        work,
        registry=registry,
        thread_factory=factory,
    )
    assert owner is not None
    assert client.stream_lookups == 0
    client.expected_owner = owner
    owner.start()
    assert client.stream_lookups == 1
    assert output.get_nowait() == ("local", "tok", "safe")
    assert output.get_nowait() == ("local", "done", None)
    assert owner.try_reap() is True


@pytest.mark.parametrize(
    ("failure", "error_name"),
    [
        (RuntimeError("provider secret detail"), "RuntimeError"),
        (KeyboardInterrupt("provider secret detail"), "KeyboardInterrupt"),
    ],
)
def test_provider_failures_publish_only_type_and_release_after_clean_return(
    failure,
    error_name,
):
    class ExplodingClient:
        def stream(self, prompt, *, system=None, images=None, history=None):
            raise failure

    registry = HedgeSourceRegistry()
    key = HedgeSourceKey("test", f"provider-{error_name}")
    output = queue.Queue()
    owner, _thread = _claim_immediate(
        registry,
        key,
        _work(ExplodingClient(), output=output),
    )
    assert output.get_nowait() == ("local", "err", error_name)
    assert owner.snapshot().state is HedgeSourceOwnerState.RETURNED
    assert owner.try_reap() is True
    assert registry.current(key) is None


def test_owner_is_published_before_start_is_attempted():
    registry = HedgeSourceRegistry()
    key = HedgeSourceKey("test", "publish-first")

    def assert_published(thread):
        assert registry.current(key) is thread.args[0]

    factory = _ThreadFactory(
        run_on_start=True,
        on_create=assert_published,
        on_start=assert_published,
    )
    owner = try_claim_hedge_source_owner(
        key,
        _work(),
        registry=registry,
        thread_factory=factory,
    )
    assert owner is not None
    assert registry.current(key) is owner
    owner.start()
    assert owner.try_reap() is True
    assert registry.current(key) is None


def test_owner_constructor_failure_cancels_reservation(monkeypatch):
    registry = HedgeSourceRegistry()
    key = HedgeSourceKey("test", "constructor-failure")

    def fail_condition(*args, **kwargs):
        raise RuntimeError("owner construction failed")

    monkeypatch.setattr(owner_module.threading, "Condition", fail_condition)

    with pytest.raises(RuntimeError, match="owner construction failed"):
        try_claim_hedge_source_owner(
            key,
            _work(),
            registry=registry,
            thread_factory=_ThreadFactory(),
        )

    assert registry.current(key) is None
    lease = try_claim_hedge_source_lease(key, registry=registry)
    assert lease is not None
    assert lease.finish(references_cleared=True) is True


def test_thread_factory_failure_after_publication_retains_exact_owner():
    registry = HedgeSourceRegistry()
    key = HedgeSourceKey("test", "thread-install-failure")
    factory = _FactoryFailure()

    with pytest.raises(RuntimeError, match="thread construction failed"):
        try_claim_hedge_source_owner(
            key,
            _work(),
            registry=registry,
            thread_factory=factory,
        )

    assert factory.calls == 1
    retained = registry.current(key)
    assert type(retained) is HedgeSourceOwner
    snapshot = retained.snapshot()
    assert snapshot.state is HedgeSourceOwnerState.RETAINED
    assert snapshot.start_attempted is False
    assert snapshot.start_returned is False
    assert snapshot.worker_returned is False
    assert snapshot.references_cleared is False
    assert snapshot.error_type == "RuntimeError"
    successor_factory = _ThreadFactory(run_on_start=True)
    assert (
        try_claim_hedge_source_owner(
            key,
            _work(),
            registry=registry,
            thread_factory=successor_factory,
        )
        is None
    )
    assert successor_factory.threads == []


# Start ambiguity and exact release proof ------------------------------------


def test_start_failure_without_spawn_remains_owned_and_blocks_successor():
    registry = HedgeSourceRegistry()
    key = HedgeSourceKey("test", "no-spawn")
    factory = _ThreadFactory(start_error=RuntimeError("start failed"))
    owner = try_claim_hedge_source_owner(
        key,
        _work(),
        registry=registry,
        thread_factory=factory,
    )
    assert owner is not None

    with pytest.raises(RuntimeError, match="start failed"):
        owner.start()

    snapshot = owner.snapshot()
    assert snapshot.state is HedgeSourceOwnerState.RETAINED
    assert snapshot.start_attempted is True
    assert snapshot.start_returned is False
    assert snapshot.worker_returned is False
    assert snapshot.references_cleared is False
    assert snapshot.error_type == "RuntimeError"
    assert owner.try_reap() is False

    successor_factory = _ThreadFactory(run_on_start=True)
    successor = try_claim_hedge_source_owner(
        key,
        _work(),
        registry=registry,
        thread_factory=successor_factory,
    )
    assert successor is None
    assert successor_factory.threads == []
    assert registry.current(key) is owner


def test_start_failure_after_exact_target_return_can_be_reaped():
    registry = HedgeSourceRegistry()
    key = HedgeSourceKey("test", "spawned-then-raised")
    factory = _ThreadFactory(
        run_on_start=True,
        raise_after_run=RuntimeError("ambiguous start"),
    )
    owner = try_claim_hedge_source_owner(
        key,
        _work(),
        registry=registry,
        thread_factory=factory,
    )
    assert owner is not None

    with pytest.raises(RuntimeError, match="ambiguous start"):
        owner.start()

    snapshot = owner.snapshot()
    assert snapshot.worker_returned is True
    assert snapshot.references_cleared is True
    assert snapshot.start_returned is False
    assert snapshot.error_type == "RuntimeError"
    assert owner.try_reap() is True
    assert registry.current(key) is None


def test_release_requires_return_reference_clear_death_and_join_zero():
    registry = HedgeSourceRegistry()
    key = HedgeSourceKey("test", "proof")
    factory = _ThreadFactory()
    owner = try_claim_hedge_source_owner(
        key,
        _work(),
        registry=registry,
        thread_factory=factory,
    )
    assert owner is not None
    thread = factory.threads[0]

    assert owner.try_reap() is False
    owner.start()
    assert owner.try_reap() is False
    thread.run_target(remain_alive=True)
    snapshot = owner.snapshot()
    assert snapshot.worker_returned is True
    assert snapshot.references_cleared is True
    assert snapshot.thread_alive is True
    assert owner.try_reap() is False
    assert thread.join_calls == []

    thread.alive = False
    assert owner.try_reap() is True
    assert thread.join_calls == [0.0]
    snapshot = owner.snapshot()
    assert snapshot.state is HedgeSourceOwnerState.CLOSED
    assert snapshot.reaped is True
    assert registry.current(key) is None


def test_is_alive_failure_refuses_reap_without_calling_join():
    registry = HedgeSourceRegistry()
    key = HedgeSourceKey("test", "alive-failure")
    factory = _ThreadFactory(
        run_on_start=True,
        alive_error=RuntimeError("is_alive failed"),
    )
    owner = try_claim_hedge_source_owner(
        key,
        _work(),
        registry=registry,
        thread_factory=factory,
    )
    assert owner is not None
    owner.start()

    assert owner.try_reap() is False
    assert factory.threads[0].join_calls == []
    assert registry.current(key) is owner


def test_is_alive_attribute_failure_refuses_reap_without_escape():
    class AliveLookupFailThread(_ControllableThread):
        def __getattribute__(self, name):
            if name == "is_alive":
                raise KeyboardInterrupt("is_alive lookup failed")
            return super().__getattribute__(name)

    threads = []

    def factory(**kwargs):
        thread = AliveLookupFailThread(**kwargs, run_on_start=True)
        threads.append(thread)
        return thread

    registry = HedgeSourceRegistry()
    key = HedgeSourceKey("test", "alive-lookup-failure")
    owner = try_claim_hedge_source_owner(
        key,
        _work(),
        registry=registry,
        thread_factory=factory,
    )
    assert owner is not None
    owner.start()

    assert owner.try_reap() is False
    assert owner.try_reap() is False
    assert threads[0].join_calls == []
    assert registry.current(key) is owner


def test_join_failure_retains_exact_owner_and_error_type():
    registry = HedgeSourceRegistry()
    key = HedgeSourceKey("test", "join-failure")
    factory = _ThreadFactory(
        run_on_start=True,
        join_error=RuntimeError("join failed"),
    )
    owner = try_claim_hedge_source_owner(
        key,
        _work(),
        registry=registry,
        thread_factory=factory,
    )
    assert owner is not None
    owner.start()

    assert owner.try_reap() is False
    assert factory.threads[0].join_calls == [0.0]
    snapshot = owner.snapshot()
    assert snapshot.state is HedgeSourceOwnerState.RETAINED
    assert snapshot.error_type == "RuntimeError"
    assert registry.current(key) is owner


def test_join_attribute_failure_retains_owner_without_escape():
    class JoinLookupFailThread(_ControllableThread):
        def __getattribute__(self, name):
            if name == "join":
                raise KeyboardInterrupt("join lookup failed")
            return super().__getattribute__(name)

    def factory(**kwargs):
        return JoinLookupFailThread(**kwargs, run_on_start=True)

    registry = HedgeSourceRegistry()
    key = HedgeSourceKey("test", "join-lookup-failure")
    owner = try_claim_hedge_source_owner(
        key,
        _work(),
        registry=registry,
        thread_factory=factory,
    )
    assert owner is not None
    owner.start()

    assert owner.try_reap() is False
    snapshot = owner.snapshot()
    assert snapshot.state is HedgeSourceOwnerState.RETAINED
    assert snapshot.error_type == "KeyboardInterrupt"
    assert registry.current(key) is owner


@pytest.mark.parametrize("release_mode", ["false", "raise"])
def test_registry_release_failure_retains_exact_owner(monkeypatch, release_mode):
    registry = HedgeSourceRegistry()
    key = HedgeSourceKey("test", f"release-{release_mode}")
    owner, thread = _claim_immediate(registry, key)

    if release_mode == "false":
        monkeypatch.setattr(registry, "release", lambda claimed_key, claimed: False)
        expected = "RegistryReleaseError"
    else:

        def fail_release(claimed_key, claimed):
            raise LookupError("release failed")

        monkeypatch.setattr(registry, "release", fail_release)
        expected = "LookupError"

    assert owner.try_reap() is False
    assert thread.join_calls == [0.0]
    snapshot = owner.snapshot()
    assert snapshot.state is HedgeSourceOwnerState.RETAINED
    assert snapshot.error_type == expected
    assert registry.current(key) is owner


def test_stream_close_failure_clears_aliases_but_permanently_retains_owner():
    registry = HedgeSourceRegistry()
    key = HedgeSourceKey("test", "worker-close-failure")
    client = _CloseFailClient()
    owner, _thread = _claim_immediate(registry, key, _work(client))

    snapshot = owner.snapshot()
    assert snapshot.worker_returned is True
    assert snapshot.references_cleared is True
    assert snapshot.state is HedgeSourceOwnerState.RETAINED
    assert snapshot.error_type == "RuntimeError"
    assert client.streams[0].close_calls == 1
    assert owner.try_reap() is False
    assert registry.current(key) is owner


def test_worker_return_clears_payload_aliases_before_publication_and_reap():
    registry = HedgeSourceRegistry()
    key = HedgeSourceKey("test", "weakrefs")
    client = _RecordingClient()
    images = _Payload([object()])
    history = _Payload([{"role": "user", "content": "private"}])
    marker = _Marker()
    turn_context = {"private": marker}
    client_ref = weakref.ref(client)
    images_ref = weakref.ref(images)
    history_ref = weakref.ref(history)
    marker_ref = weakref.ref(marker)
    factory = _ThreadFactory(run_on_start=True)
    work = _work(
        client,
        images=images,
        history=history,
        turn_context=turn_context,
    )
    owner = try_claim_hedge_source_owner(
        key,
        work,
        registry=registry,
        thread_factory=factory,
    )
    assert owner is not None
    del work, client, images, history, marker, turn_context

    owner.start()
    assert owner.snapshot().references_cleared is True
    gc.collect()
    assert client_ref() is None
    assert images_ref() is None
    assert history_ref() is None
    assert marker_ref() is None
    assert owner.try_reap() is True


def test_context_reset_failure_clears_payload_but_retains_owner(monkeypatch):
    class ResetFailContext:
        def set(self, value):
            return object()

        def reset(self, token):
            raise RuntimeError("reset failed")

    registry = HedgeSourceRegistry()
    key = HedgeSourceKey("test", "reset-failure")
    monkeypatch.setattr(owner_module, "ContextVar", ResetFailContext)
    owner, _thread = _claim_immediate(
        registry,
        key,
        _work(context_var=ResetFailContext()),
    )
    snapshot = owner.snapshot()
    assert snapshot.references_cleared is True
    assert snapshot.worker_returned is True
    assert snapshot.state is HedgeSourceOwnerState.RETAINED
    assert snapshot.error_type == "RuntimeError"
    assert owner.try_reap() is False


# Registry identity / ABA contracts -----------------------------------------


def test_forged_lease_cannot_release_current_exact_owner():
    registry = HedgeSourceRegistry()
    key = HedgeSourceKey("test", "forged")
    real = try_claim_hedge_source_lease(key, registry=registry)
    assert real is not None
    forged = HedgeSourceLease(key, registry)

    assert forged.finish(references_cleared=True) is False
    assert forged.snapshot().state is HedgeSourceLeaseState.RETAINED
    assert registry.current(key) is real
    assert registry.release(key, object()) is False
    assert registry.current(key) is real
    assert real.finish(references_cleared=True) is True


def test_stale_owner_cannot_release_aba_successor():
    registry = HedgeSourceRegistry()
    key = HedgeSourceKey("test", "aba")
    stale, _thread = _claim_immediate(registry, key)
    assert registry.release(key, stale) is True
    successor = try_claim_hedge_source_lease(key, registry=registry)
    assert successor is not None

    assert stale.try_reap() is False
    assert stale.snapshot().state is HedgeSourceOwnerState.RETAINED
    assert registry.current(key) is successor
    assert successor.finish(references_cleared=True) is True


def test_direct_lease_releases_only_after_exact_successful_cleanup():
    key = HedgeSourceKey("test", "lease-cleanup")

    clean_registry = HedgeSourceRegistry()
    clean = try_claim_hedge_source_lease(key, registry=clean_registry)
    assert clean is not None
    assert clean.finish(references_cleared=True) is True
    assert clean.finish(references_cleared=True) is True
    assert clean.snapshot().state is HedgeSourceLeaseState.CLOSED
    assert clean_registry.current(key) is None

    retained_registry = HedgeSourceRegistry()
    retained = try_claim_hedge_source_lease(key, registry=retained_registry)
    assert retained is not None
    assert retained.finish(references_cleared=False) is False
    assert retained.finish(references_cleared=True) is False
    snapshot = retained.snapshot()
    assert snapshot.state is HedgeSourceLeaseState.RETAINED
    assert snapshot.references_cleared is False
    assert snapshot.error_type == "ReferencesNotCleared"
    assert retained_registry.current(key) is retained


def test_direct_lease_release_failure_is_retained(monkeypatch):
    registry = HedgeSourceRegistry()
    key = HedgeSourceKey("test", "lease-release-failure")
    lease = try_claim_hedge_source_lease(key, registry=registry)
    assert lease is not None
    monkeypatch.setattr(registry, "release", lambda claimed_key, owner: False)

    assert lease.finish(references_cleared=True) is False
    snapshot = lease.snapshot()
    assert snapshot.state is HedgeSourceLeaseState.RETAINED
    assert snapshot.references_cleared is True
    assert snapshot.error_type == "RegistryReleaseError"
    assert registry.current(key) is lease


# Hedge integration: no wait, failover, and direct-call cleanup --------------


def test_local_only_same_key_busy_never_calls_client_or_constructs_thread():
    registry = HedgeSourceRegistry()
    local_key = HedgeSourceKey("shared", "local")
    busy = try_claim_hedge_source_lease(local_key, registry=registry)
    assert busy is not None
    client = _RecordingClient(["must-not-run"])
    factory = _ThreadFactory(run_on_start=True)
    hedge = HedgeLLM(
        local=client,
        cloud=None,
        source_owner_registry=registry,
        local_source_key=local_key,
        cloud_source_keys=[],
        thread_factory=factory,
    )

    assert list(hedge.stream("private")) == []
    assert hedge.last_source is None
    assert client.calls == 0
    assert factory.threads == []
    assert busy.finish(references_cleared=True) is True


def test_successful_local_only_direct_call_releases_for_immediate_reuse():
    registry = HedgeSourceRegistry()
    local_key = HedgeSourceKey("shared", "direct-success")
    client = _RecordingClient(["ok"])
    hedge = HedgeLLM(
        local=client,
        cloud=None,
        source_owner_registry=registry,
        local_source_key=local_key,
        cloud_source_keys=[],
    )

    assert "".join(hedge.stream("one")) == "ok"
    assert registry.current(local_key) is None
    assert "".join(hedge.stream("two")) == "ok"
    assert client.calls == 2
    assert registry.current(local_key) is None


def test_direct_preparation_failure_releases_lease_without_calling_client():
    class RaisingHistory:
        def __bool__(self):
            raise RuntimeError("private history detail")

    registry = HedgeSourceRegistry()
    local_key = HedgeSourceKey("shared", "direct-preparation")
    client = _RecordingClient(["must-not-run"])
    hedge = HedgeLLM(
        local=client,
        cloud=None,
        source_owner_registry=registry,
        local_source_key=local_key,
        cloud_source_keys=[],
    )

    with pytest.raises(RuntimeError, match="private history detail"):
        list(hedge.stream("one", history=RaisingHistory()))
    assert client.calls == 0
    assert registry.current(local_key) is None
    assert "".join(hedge.stream("two")) == "must-not-run"


def test_direct_stream_close_failure_retains_key_and_blocks_next_client_call():
    registry = HedgeSourceRegistry()
    local_key = HedgeSourceKey("shared", "direct-close")
    client = _CloseFailClient()
    hedge = HedgeLLM(
        local=client,
        cloud=None,
        source_owner_registry=registry,
        local_source_key=local_key,
        cloud_source_keys=[],
    )

    assert "".join(hedge.stream("one")) == "answer"
    retained = registry.current(local_key)
    assert type(retained) is HedgeSourceLease
    snapshot = retained.snapshot()
    assert snapshot.state is HedgeSourceLeaseState.RETAINED
    assert snapshot.references_cleared is True
    assert snapshot.error_type == "RuntimeError"
    assert list(hedge.stream("two")) == []
    assert client.calls == 1


def test_direct_stream_close_attribute_failure_resets_context_and_retains_key():
    class CloseLookupFailStream:
        def __init__(self) -> None:
            self._tokens = iter(("answer",))

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._tokens)

        def __getattribute__(self, name):
            if name == "close":
                raise KeyboardInterrupt("private close detail")
            return object.__getattribute__(self, name)

    class Client:
        def __init__(self) -> None:
            self.calls = 0

        def stream(self, prompt, *, system=None, images=None, history=None):
            self.calls += 1
            return CloseLookupFailStream()

    registry = HedgeSourceRegistry()
    local_key = HedgeSourceKey("shared", "direct-close-lookup")
    client = Client()
    hedge = HedgeLLM(
        local=client,
        cloud=None,
        source_owner_registry=registry,
        local_source_key=local_key,
        cloud_source_keys=[],
    )
    prior = {"outer": _Marker()}
    token = capability_context.set(prior)
    try:
        assert "".join(hedge.stream("one")) == "answer"
        assert capability_context.get() is prior
        retained = registry.current(local_key)
        assert type(retained) is HedgeSourceLease
        assert retained.snapshot().error_type == "KeyboardInterrupt"
        assert list(hedge.stream("two")) == []
        assert client.calls == 1
    finally:
        capability_context.reset(token)


def test_direct_context_reset_failure_retains_key(monkeypatch):
    class ResetFailContext:
        def get(self):
            return {}

        def set(self, value):
            return object()

        def reset(self, token):
            raise RuntimeError("reset failed")

    registry = HedgeSourceRegistry()
    local_key = HedgeSourceKey("shared", "direct-reset")
    client = _RecordingClient(["ok"])
    hedge = HedgeLLM(
        local=client,
        cloud=None,
        source_owner_registry=registry,
        local_source_key=local_key,
        cloud_source_keys=[],
    )
    monkeypatch.setattr(llm_module, "capability_context", ResetFailContext())

    assert "".join(hedge.stream("one")) == "ok"
    retained = registry.current(local_key)
    assert type(retained) is HedgeSourceLease
    assert retained.snapshot().error_type == "RuntimeError"
    assert list(hedge.stream("two")) == []
    assert client.calls == 1


def test_direct_call_clears_images_and_history_aliases_before_release():
    registry = HedgeSourceRegistry()
    local_key = HedgeSourceKey("shared", "direct-aliases")
    client = _RecordingClient([])
    hedge = HedgeLLM(
        local=client,
        cloud=None,
        source_owner_registry=registry,
        local_source_key=local_key,
        cloud_source_keys=[],
    )
    images = _Payload([object()])
    history = _Payload([{"role": "user", "content": "private"}])
    images_ref = weakref.ref(images)
    history_ref = weakref.ref(history)

    stream = hedge.stream("prompt", images=images, history=history)
    del images, history
    assert list(stream) == []
    del stream
    # The fake's observation is test-owned; remove it before checking only the
    # Hedge frame/lease aliases.
    client.seen.clear()
    gc.collect()
    assert images_ref() is None
    assert history_ref() is None
    assert registry.current(local_key) is None


def test_busy_local_race_advances_to_different_cloud_without_local_thread():
    registry = HedgeSourceRegistry()
    local_key = HedgeSourceKey("shared", "busy-local")
    cloud_key = HedgeSourceKey("shared", "available-cloud")
    busy = try_claim_hedge_source_lease(local_key, registry=registry)
    assert busy is not None
    local = _RecordingClient(["local"])
    cloud = _RecordingClient(["cloud"])
    factory = _ThreadFactory(run_on_start=True)
    hedge = HedgeLLM(
        local=local,
        cloud=[cloud],
        strategy="hedge",
        hedge_delay_ms=0,
        enforce_cloud_egress_scope=True,
        source_owner_registry=registry,
        local_source_key=local_key,
        cloud_source_keys=[cloud_key],
        thread_factory=factory,
    )
    token = capability_context.set(_current_turn_context())
    try:
        assert "".join(hedge.stream("q")) == "cloud"
    finally:
        capability_context.reset(token)

    assert local.calls == 0
    assert cloud.calls == 1
    assert len(factory.threads) == 1
    assert hedge.last_source == "cloud_0"
    assert registry.current(cloud_key) is None
    assert busy.finish(references_cleared=True) is True


def test_busy_first_fallback_cloud_advances_immediately_to_next_key():
    registry = HedgeSourceRegistry()
    local_key = HedgeSourceKey("shared", "fallback-local")
    cloud_zero_key = HedgeSourceKey("shared", "busy-cloud")
    cloud_one_key = HedgeSourceKey("shared", "fresh-cloud")
    busy = try_claim_hedge_source_lease(cloud_zero_key, registry=registry)
    assert busy is not None
    local = _RecordingClient(["local"])
    cloud_zero = _RecordingClient(["must-not-run"])
    cloud_one = _RecordingClient(["fresh"])
    factory = _ThreadFactory(run_on_start=True)
    hedge = HedgeLLM(
        local=local,
        cloud=[cloud_zero, cloud_one],
        strategy="fallback",
        ttft_deadline_ms=10_000,
        enforce_cloud_egress_scope=True,
        source_owner_registry=registry,
        local_source_key=local_key,
        cloud_source_keys=[cloud_zero_key, cloud_one_key],
        thread_factory=factory,
    )
    token = capability_context.set(_current_turn_context())
    try:
        assert "".join(hedge.stream("q")) == "fresh"
    finally:
        capability_context.reset(token)

    assert cloud_zero.calls == 0
    assert cloud_one.calls == 1
    assert local.calls == 0
    assert len(factory.threads) == 1
    assert hedge.last_source == "cloud_1"
    assert busy.finish(references_cleared=True) is True


def test_different_source_key_can_run_while_another_key_is_busy():
    registry = HedgeSourceRegistry()
    busy_key = HedgeSourceKey("shared", "busy")
    available_key = HedgeSourceKey("shared", "available")
    busy = try_claim_hedge_source_lease(busy_key, registry=registry)
    assert busy is not None

    owner, thread = _claim_immediate(registry, available_key)
    assert owner.snapshot().worker_returned is True
    assert owner.try_reap() is True
    assert thread.join_calls == [0.0]
    assert registry.current(busy_key) is busy
    assert busy.finish(references_cleared=True) is True


@pytest.mark.parametrize(
    "turn_context",
    [
        _local_only_context(),
        {},
        {CLOUD_EGRESS_SCOPE_CONTEXT_KEY: "malformed"},
    ],
)
def test_worker_to_local_only_or_malformed_collision_uses_same_local_key(
    turn_context,
):
    registry = HedgeSourceRegistry()
    local_key = HedgeSourceKey("shared", "cross-direction")
    factory = _ThreadFactory()
    owner = try_claim_hedge_source_owner(
        local_key,
        _work(),
        registry=registry,
        thread_factory=factory,
    )
    assert owner is not None
    owner.start()
    local = _RecordingClient(["must-not-run"])
    hedge = HedgeLLM(
        local=local,
        cloud=[_RecordingClient(["cloud-must-not-run"])],
        enforce_cloud_egress_scope=True,
        source_owner_registry=registry,
        local_source_key=local_key,
        cloud_source_keys=[HedgeSourceKey("shared", "cloud")],
    )
    token = capability_context.set(turn_context)
    try:
        assert list(hedge.stream("private")) == []
    finally:
        capability_context.reset(token)
    assert local.calls == 0
    assert registry.current(local_key) is owner


def test_retained_source_blocks_rebuilt_wrapper_without_new_thread_or_client():
    registry = HedgeSourceRegistry()
    local_key = HedgeSourceKey("shared", "rebuild-local")
    retained_factory = _ThreadFactory(start_error=RuntimeError("ambiguous"))
    retained = try_claim_hedge_source_owner(
        local_key,
        _work(),
        registry=registry,
        thread_factory=retained_factory,
    )
    assert retained is not None
    with pytest.raises(RuntimeError, match="ambiguous"):
        retained.start()

    rebuilt_client = _RecordingClient(["must-not-run"])
    rebuilt_factory = _ThreadFactory(run_on_start=True)
    rebuilt = HedgeLLM(
        local=rebuilt_client,
        cloud=None,
        source_owner_registry=registry,
        local_source_key=local_key,
        cloud_source_keys=[],
        thread_factory=rebuilt_factory,
    )
    assert list(rebuilt.stream("q")) == []
    assert rebuilt_client.calls == 0
    assert rebuilt_factory.threads == []
    assert registry.current(local_key) is retained


def test_late_queue_token_cannot_resurrect_retired_fallback_source(monkeypatch):
    class ScriptedQueue:
        def __init__(self):
            self.actions = [
                queue.Empty,
                ("cloud_0", "tok", "late"),
                ("cloud_1", "tok", "fresh"),
                ("cloud_1", "done", None),
            ]
            self.puts = []

        def put(self, item):
            self.puts.append(item)

        def get(self, timeout=None):
            action = self.actions.pop(0)
            if action is queue.Empty:
                raise queue.Empty
            return action

    registry = HedgeSourceRegistry()
    factory = _ThreadFactory()
    monkeypatch.setattr(llm_module.queue, "Queue", ScriptedQueue)
    hedge = HedgeLLM(
        local=_RecordingClient(["local"]),
        cloud=[_RecordingClient(["late"]), _RecordingClient(["fresh"])],
        strategy="fallback",
        ttft_deadline_ms=0,
        source_owner_registry=registry,
        local_source_key=HedgeSourceKey("scripted", "local"),
        cloud_source_keys=[
            HedgeSourceKey("scripted", "cloud-zero"),
            HedgeSourceKey("scripted", "cloud-one"),
        ],
        thread_factory=factory,
    )
    hedge.WORKER_JOIN_TIMEOUT = 0.0

    assert "".join(hedge.stream("q")) == "fresh"
    assert hedge.last_source == "cloud_1"
    assert len(factory.threads) == 2


# Factory and direct-wrapper key topology -----------------------------------


def test_factory_multi_chain_keys_are_shared_by_exact_preset_across_rebuilds(
    monkeypatch,
):
    clients = {
        "alpha": _RecordingClient(["a"]),
        "beta": _RecordingClient(["b"]),
    }

    def fake_build(name, preset, **kwargs):
        return clients[name]

    monkeypatch.setattr(llm_factory, "_build_cloud_client", fake_build)
    config = {
        "cloud": {"enabled": True},
        "cloud_providers": {
            "alpha": {"model": "alpha-model"},
            "beta": {"model": "beta-model"},
        },
        "cloud_chains": {
            "private": ["alpha"],
            "public": ["alpha", "beta"],
        },
    }
    local = _RecordingClient(["local"])
    first = llm_factory._wrap_cloud(local, config)
    rebuilt = llm_factory._wrap_cloud(local, config)
    assert type(first) is SensitivityRouterLLM
    assert type(rebuilt) is SensitivityRouterLLM

    first_private = first.chains["private"]
    first_public = first.chains["public"]
    rebuilt_private = rebuilt.chains["private"]
    assert first_private._source_owner_registry is FACTORY_HEDGE_SOURCE_REGISTRY
    assert first_public._source_owner_registry is FACTORY_HEDGE_SOURCE_REGISTRY
    assert rebuilt_private._source_owner_registry is FACTORY_HEDGE_SOURCE_REGISTRY
    assert first_private._local_source_key is FACTORY_LOCAL_MAIN_SOURCE
    assert first_public._local_source_key is FACTORY_LOCAL_MAIN_SOURCE
    assert rebuilt_private._local_source_key is FACTORY_LOCAL_MAIN_SOURCE
    alpha = factory_named_cloud_source("alpha")
    beta = factory_named_cloud_source("beta")
    assert first_private._cloud_source_keys == (alpha,)
    assert first_public._cloud_source_keys == (alpha, beta)
    assert rebuilt_private._cloud_source_keys == (alpha,)


def test_factory_single_cloud_uses_fixed_local_and_cloud_keys(monkeypatch):
    cloud = _RecordingClient(["cloud"])
    monkeypatch.setattr(llm_factory, "OpenAICompatLLM", lambda **kwargs: cloud)
    local = _RecordingClient(["local"])
    wrapped = llm_factory._wrap_cloud(
        local,
        {"cloud": {"enabled": True, "model": "single"}},
    )
    assert type(wrapped) is HedgeLLM
    assert wrapped._source_owner_registry is FACTORY_HEDGE_SOURCE_REGISTRY
    assert wrapped._local_source_key is FACTORY_LOCAL_MAIN_SOURCE
    assert wrapped._cloud_source_keys == (FACTORY_SINGLE_CLOUD_SOURCE,)


def test_direct_wrappers_keep_private_registries_and_stable_member_keys():
    shared_client = _RecordingClient(["ok"])
    first = HedgeLLM(local=shared_client, cloud=None)
    second = HedgeLLM(local=shared_client, cloud=None)
    assert first._source_owner_registry is not second._source_owner_registry
    assert first._local_source_key == HedgeSourceKey("direct", "LOCAL")
    assert second._local_source_key == HedgeSourceKey("direct", "LOCAL")
    assert first._cloud_source_keys == ()
    assert second._cloud_source_keys == ()
    chained = HedgeLLM(
        local=shared_client,
        cloud=[_RecordingClient(), _RecordingClient()],
    )
    assert chained._cloud_source_keys == (
        HedgeSourceKey("direct", "CLOUD_0"),
        HedgeSourceKey("direct", "CLOUD_1"),
    )

    busy = try_claim_hedge_source_lease(
        first._local_source_key,
        registry=first._source_owner_registry,
    )
    assert busy is not None
    assert list(first.stream("blocked")) == []
    assert "".join(second.stream("independent")) == "ok"
    assert shared_client.calls == 1
    assert busy.finish(references_cleared=True) is True


def test_hedge_constructor_requires_complete_exact_shared_key_bundle():
    registry = HedgeSourceRegistry()
    local = _RecordingClient()
    cloud = _RecordingClient()
    key = HedgeSourceKey("shared", "local")

    with pytest.raises(TypeError):
        HedgeLLM(local=local, cloud=[cloud], source_owner_registry=registry)
    with pytest.raises(TypeError):
        HedgeLLM(
            local=local,
            cloud=[cloud],
            source_owner_registry=registry,
            local_source_key=key,
            cloud_source_keys=None,
        )
    with pytest.raises(ValueError):
        HedgeLLM(
            local=local,
            cloud=[cloud],
            source_owner_registry=registry,
            local_source_key=key,
            cloud_source_keys=[],
        )
