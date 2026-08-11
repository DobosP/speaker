from __future__ import annotations

from dataclasses import fields
import gc
import threading
import time
import weakref

import numpy as np
import pytest

import core.engines._kws_speaker_inference_owner as owner_module
from core.engines._kws_speaker_inference_owner import (
    KwsSpeakerInferenceOutcome,
    KwsSpeakerInferenceOwnerState,
    KwsSpeakerInferenceRegistry,
    require_kws_speaker_inference_idle,
    try_claim_kws_speaker_inference_owner,
)
from core.engines.speaker_gate import (
    RuntimeSpeakerInferencePermit,
    RuntimeSpeakerInferencePermitSnapshot,
    SpeakerGate,
    SpeakerSimilarityBatchReceipt,
)


def _owned_clip(value: float = 0.25) -> np.ndarray:
    clip = np.full(160, value, dtype="float32")
    clip.setflags(write=False)
    assert clip.flags.owndata
    return clip


def _enrolled_gate(embed_fn) -> SpeakerGate:
    gate = SpeakerGate(threshold=0.5, embed_fn=embed_fn)
    gate.enroll_embedding([1.0, 0.0])
    return gate


def _future_deadline() -> float:
    return time.monotonic() + 60.0


class _InlineThread:
    def __init__(
        self,
        *,
        target,
        args,
        name,
        daemon,
        before_start=None,
    ) -> None:
        self.target = target
        self.args = args
        self.name = name
        self.daemon = daemon
        self.before_start = before_start
        self.start_calls = 0
        self.join_calls: list[float | None] = []
        self._alive = False

    def start(self) -> None:
        self.start_calls += 1
        if self.before_start is not None:
            self.before_start()
        self._alive = True
        try:
            self.target(*self.args)
        finally:
            self._alive = False

    def join(self, timeout=None) -> None:
        self.join_calls.append(timeout)

    def is_alive(self) -> bool:
        return self._alive


class _DeferredThread(_InlineThread):
    def start(self) -> None:
        self.start_calls += 1
        if self.before_start is not None:
            self.before_start()
        self._alive = True

    def run_target(self) -> None:
        assert self._alive
        try:
            self.target(*self.args)
        finally:
            self._alive = False


class _AmbiguousStartThread(_DeferredThread):
    def start(self) -> None:
        self.start_calls += 1
        if self.before_start is not None:
            self.before_start()
        self._alive = True
        raise RuntimeError("injected ambiguous Thread.start failure")


class _ObservedNativeThread:
    def __init__(self, *, target, args, name, daemon) -> None:
        self.join_calls: list[float | None] = []
        self._thread = threading.Thread(
            target=target,
            args=args,
            name=name,
            daemon=daemon,
        )

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout=None) -> None:
        self.join_calls.append(timeout)
        self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def await_return(self) -> None:
        self._thread.join(timeout=2.0)
        assert not self._thread.is_alive()


class _BlockingReceiveJoinThread(_InlineThread):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.receive_join_entered = threading.Event()
        self.release_receive_join = threading.Event()

    def join(self, timeout=None) -> None:
        self.join_calls.append(timeout)
        if timeout is not None and timeout > 0.0:
            self.receive_join_entered.set()
            assert self.release_receive_join.wait(timeout=2.0)


class _JoinInterruptionInlineThread(_InlineThread):
    def __init__(self, *, raise_on_receive_join: bool, **kwargs) -> None:
        super().__init__(**kwargs)
        self.raise_on_receive_join = raise_on_receive_join
        self.on_receive_join = None
        self.force_alive = False

    def join(self, timeout=None) -> None:
        self.join_calls.append(timeout)
        if timeout is None or timeout <= 0.0:
            return
        assert self.on_receive_join is not None
        self.on_receive_join()
        if self.raise_on_receive_join:
            raise RuntimeError("injected receive join failure")
        self.force_alive = True

    def is_alive(self) -> bool:
        return self.force_alive or super().is_alive()


class _Clock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class _OneShotReleaseFailurePermit(RuntimeSpeakerInferencePermit):
    def __init__(self) -> None:
        super().__init__()
        self.release_calls = 0

    def release(self, lease) -> bool:
        self.release_calls += 1
        if self.release_calls == 1:
            raise KeyboardInterrupt("injected permit release BaseException")
        return super().release(lease)


def test_claim_is_process_single_flight_before_start_and_payload_is_minimal() -> None:
    permit = RuntimeSpeakerInferencePermit()
    registry = KwsSpeakerInferenceRegistry()
    threads: list[_DeferredThread] = []
    gate = _enrolled_gate(lambda _samples, _sample_rate: [1.0, 0.0])
    clip = _owned_clip()

    def thread_factory(**kwargs):
        thread = _DeferredThread(**kwargs)
        threads.append(thread)
        return thread

    owner = try_claim_kws_speaker_inference_owner(
        gate,
        (clip,),
        16_000,
        deadline=_future_deadline(),
        runtime_permit=permit,
        registry=registry,
        thread_factory=thread_factory,
    )

    assert owner is not None
    assert registry.current() is owner
    assert len(threads) == 1
    assert threads[0].start_calls == 0
    assert permit.snapshot() == RuntimeSpeakerInferencePermitSnapshot(
        active=True,
        acquisitions=1,
        releases=0,
    )
    snapshot = owner.snapshot()
    assert snapshot.state is KwsSpeakerInferenceOwnerState.NEW
    assert not snapshot.start_attempted
    assert snapshot.runtime_lease_current

    work = owner._work
    assert work is not None
    assert tuple(field.name for field in fields(type(work))) == (
        "gate",
        "clips",
        "sample_rate",
        "ticket",
    )
    assert work.gate is gate
    assert work.clips == (clip,)
    assert threads[0].args == (owner,)
    assert getattr(threads[0].target, "__self__", None) is None

    second_factory_calls = 0

    def second_factory(**kwargs):
        nonlocal second_factory_calls
        second_factory_calls += 1
        return _DeferredThread(**kwargs)

    assert (
        try_claim_kws_speaker_inference_owner(
            _enrolled_gate(lambda _samples, _sample_rate: [1.0, 0.0]),
            (_owned_clip(0.5),),
            16_000,
            deadline=_future_deadline(),
            runtime_permit=permit,
            registry=registry,
            thread_factory=second_factory,
        )
        is None
    )
    assert second_factory_calls == 0
    assert permit.snapshot().acquisitions == 1
    with pytest.raises(RuntimeError, match="rebuild or start"):
        registry.acquire_mutation()

    assert owner.abandon(KwsSpeakerInferenceOutcome.STOPPED)
    assert owner.snapshot().reaped
    assert registry.current() is None
    assert permit.snapshot() == RuntimeSpeakerInferencePermitSnapshot(
        active=False,
        acquisitions=1,
        releases=1,
    )


def test_registry_mutation_fences_reservation_without_acquiring_runtime_permit() -> (
    None
):
    permit = RuntimeSpeakerInferencePermit()
    registry = KwsSpeakerInferenceRegistry()
    mutation = registry.acquire_mutation()
    try:
        owner = try_claim_kws_speaker_inference_owner(
            _enrolled_gate(lambda _samples, _sample_rate: [1.0, 0.0]),
            (_owned_clip(),),
            16_000,
            deadline=_future_deadline(),
            runtime_permit=permit,
            registry=registry,
        )
        assert owner is None
        assert permit.snapshot() == RuntimeSpeakerInferencePermitSnapshot(
            active=False,
            acquisitions=0,
            releases=0,
        )
    finally:
        assert registry.release_mutation(mutation)

    observed = []

    def factory(**kwargs):
        with pytest.raises(RuntimeError, match="rebuild or start"):
            registry.acquire_mutation()
        observed.append("reservation-fenced")
        return _DeferredThread(**kwargs)

    owner = try_claim_kws_speaker_inference_owner(
        _enrolled_gate(lambda _samples, _sample_rate: [1.0, 0.0]),
        (_owned_clip(),),
        16_000,
        deadline=_future_deadline(),
        runtime_permit=permit,
        registry=registry,
        thread_factory=factory,
    )
    assert owner is not None
    assert observed == ["reservation-fenced"]
    with pytest.raises(ValueError, match="abandon outcome"):
        owner.abandon("stopped")
    assert not owner.snapshot().abandoned
    assert owner.abandon()


def test_definitive_thread_factory_failure_releases_permit_and_reservation() -> None:
    permit = RuntimeSpeakerInferencePermit()
    registry = KwsSpeakerInferenceRegistry()

    def fail_factory(**_kwargs):
        raise RuntimeError("injected thread construction failure")

    with pytest.raises(RuntimeError, match="thread construction failure"):
        try_claim_kws_speaker_inference_owner(
            _enrolled_gate(lambda _samples, _sample_rate: [1.0, 0.0]),
            (_owned_clip(),),
            16_000,
            deadline=_future_deadline(),
            runtime_permit=permit,
            registry=registry,
            thread_factory=fail_factory,
        )

    assert registry.current() is None
    assert permit.snapshot() == RuntimeSpeakerInferencePermitSnapshot(
        active=False,
        acquisitions=1,
        releases=1,
    )


def test_ambiguous_thread_start_failure_retains_exact_owner_and_permit() -> None:
    permit = RuntimeSpeakerInferencePermit()
    registry = KwsSpeakerInferenceRegistry()
    threads = []

    def factory(**kwargs):
        thread = _AmbiguousStartThread(**kwargs)
        threads.append(thread)
        return thread

    owner = try_claim_kws_speaker_inference_owner(
        _enrolled_gate(lambda _samples, _sample_rate: [1.0, 0.0]),
        (_owned_clip(),),
        16_000,
        deadline=_future_deadline(),
        runtime_permit=permit,
        registry=registry,
        thread_factory=factory,
    )
    assert owner is not None

    with pytest.raises(RuntimeError, match="ambiguous Thread.start failure"):
        owner.start()

    snapshot = owner.snapshot()
    assert snapshot.state is KwsSpeakerInferenceOwnerState.RETAINED
    assert snapshot.start_attempted
    assert not snapshot.start_returned
    assert snapshot.abandoned
    assert snapshot.thread_alive
    assert snapshot.runtime_lease_current
    assert snapshot.error_type == "RuntimeError"
    result = owner.receive(owner.ticket)
    assert result.outcome is KwsSpeakerInferenceOutcome.ERROR
    assert result.error_type == "RuntimeError"
    assert threads[0].join_calls == []
    assert not owner.try_reap()
    with pytest.raises(RuntimeError, match="still be live"):
        require_kws_speaker_inference_idle(registry=registry)
    assert (
        try_claim_kws_speaker_inference_owner(
            _enrolled_gate(lambda _samples, _sample_rate: [1.0, 0.0]),
            (_owned_clip(),),
            16_000,
            deadline=_future_deadline(),
            runtime_permit=permit,
            registry=registry,
        )
        is None
    )
    assert permit.snapshot() == RuntimeSpeakerInferencePermitSnapshot(
        active=True,
        acquisitions=1,
        releases=0,
    )


def test_completed_raw_receipt_is_claimed_before_exact_reap() -> None:
    permit = RuntimeSpeakerInferencePermit()
    registry = KwsSpeakerInferenceRegistry()
    threads = []
    calls = []
    owner_box = []

    def before_start() -> None:
        calls.append("start")
        assert permit.snapshot().active
        assert registry.current() is owner_box[0]

    def factory(**kwargs):
        thread = _InlineThread(
            **kwargs,
            before_start=before_start,
        )
        threads.append(thread)
        return thread

    gate = _enrolled_gate(
        lambda samples, _sample_rate: calls.append(tuple(samples)) or [1.0, 0.0]
    )
    owner = try_claim_kws_speaker_inference_owner(
        gate,
        (_owned_clip(),),
        16_000,
        deadline=_future_deadline(),
        runtime_permit=permit,
        registry=registry,
        thread_factory=factory,
    )
    assert owner is not None
    owner_box.append(owner)

    ticket = owner.start()
    ready = owner.snapshot()
    assert ready.state is KwsSpeakerInferenceOwnerState.RESULT_READY
    assert ready.worker_returned
    assert ready.native_steps == 1
    assert ready.runtime_lease_current
    assert not ready.reaped
    assert permit.snapshot().releases == 0

    assert (
        try_claim_kws_speaker_inference_owner(
            gate,
            (_owned_clip(0.5),),
            16_000,
            deadline=_future_deadline(),
            runtime_permit=permit,
            registry=registry,
        )
        is None
    )
    result = owner.receive(ticket)

    assert result.outcome is KwsSpeakerInferenceOutcome.COMPLETED
    assert type(result.receipt) is SpeakerSimilarityBatchReceipt
    assert result.receipt.similarities == (1.0,)
    assert owner.snapshot().state is KwsSpeakerInferenceOwnerState.CLOSED
    assert owner.snapshot().reaped
    assert registry.current() is None
    assert permit.snapshot() == RuntimeSpeakerInferencePermitSnapshot(
        active=False,
        acquisitions=1,
        releases=1,
    )
    assert threads[0].join_calls


def test_worker_clears_gate_and_clip_refs_before_publishing_raw_receipt() -> None:
    permit = RuntimeSpeakerInferencePermit()
    registry = KwsSpeakerInferenceRegistry()
    gate = _enrolled_gate(lambda _samples, _sample_rate: [1.0, 0.0])
    clip = _owned_clip()
    gate_ref = weakref.ref(gate)
    clip_ref = weakref.ref(clip)
    owner = try_claim_kws_speaker_inference_owner(
        gate,
        (clip,),
        16_000,
        deadline=_future_deadline(),
        runtime_permit=permit,
        registry=registry,
        thread_factory=lambda **kwargs: _InlineThread(**kwargs),
    )
    assert owner is not None
    del gate
    del clip

    ticket = owner.start()
    gc.collect()

    ready = owner.snapshot()
    assert ready.state is KwsSpeakerInferenceOwnerState.RESULT_READY
    assert ready.worker_returned
    assert ready.runtime_lease_current
    assert not ready.reaped
    assert gate_ref() is None
    assert clip_ref() is None
    result = owner.receive(ticket)
    assert result.outcome is KwsSpeakerInferenceOutcome.COMPLETED
    assert type(result.receipt) is SpeakerSimilarityBatchReceipt
    assert permit.snapshot() == RuntimeSpeakerInferencePermitSnapshot(
        active=False,
        acquisitions=1,
        releases=1,
    )


def test_second_receiver_before_reap_cannot_claim_or_abandon_ready_result() -> None:
    permit = RuntimeSpeakerInferencePermit()
    registry = KwsSpeakerInferenceRegistry()
    threads = []

    def factory(**kwargs):
        thread = _BlockingReceiveJoinThread(**kwargs)
        threads.append(thread)
        return thread

    owner = try_claim_kws_speaker_inference_owner(
        _enrolled_gate(lambda _samples, _sample_rate: [1.0, 0.0]),
        (_owned_clip(),),
        16_000,
        deadline=_future_deadline(),
        runtime_permit=permit,
        registry=registry,
        thread_factory=factory,
    )
    assert owner is not None
    ticket = owner.start()
    assert owner.snapshot().state is KwsSpeakerInferenceOwnerState.RESULT_READY

    first_results = []
    first_receiver = threading.Thread(
        target=lambda: first_results.append(owner.receive(ticket))
    )
    first_receiver.start()
    assert threads[0].receive_join_entered.wait(timeout=2.0)

    before_second = owner.snapshot()
    assert before_second.worker_returned
    assert before_second.result_claimed
    assert not before_second.reaped
    assert before_second.runtime_lease_current
    second = owner.receive(ticket)

    assert second.outcome is KwsSpeakerInferenceOutcome.ERROR
    assert second.error_type == "InferenceResultAlreadyClaimed"
    after_second = owner.snapshot()
    assert after_second.state is KwsSpeakerInferenceOwnerState.RESULT_READY
    assert after_second.result_claimed
    assert not after_second.abandoned
    assert not after_second.reaped
    assert permit.snapshot().releases == 0

    threads[0].release_receive_join.set()
    first_receiver.join(timeout=2.0)
    assert not first_receiver.is_alive()
    assert len(first_results) == 1
    assert first_results[0].outcome is KwsSpeakerInferenceOutcome.COMPLETED
    closed = owner.snapshot()
    assert closed.state is KwsSpeakerInferenceOwnerState.CLOSED
    assert closed.reaped
    assert not closed.abandoned
    assert registry.current() is None
    assert permit.snapshot() == RuntimeSpeakerInferencePermitSnapshot(
        active=False,
        acquisitions=1,
        releases=1,
    )


@pytest.mark.parametrize("raise_on_receive_join", (False, True))
def test_stop_wins_after_result_claim_during_receive_join(
    raise_on_receive_join,
) -> None:
    permit = RuntimeSpeakerInferencePermit()
    registry = KwsSpeakerInferenceRegistry()
    threads = []

    def factory(**kwargs):
        thread = _JoinInterruptionInlineThread(
            **kwargs,
            raise_on_receive_join=raise_on_receive_join,
        )
        threads.append(thread)
        return thread

    owner = try_claim_kws_speaker_inference_owner(
        _enrolled_gate(lambda _samples, _sample_rate: [1.0, 0.0]),
        (_owned_clip(),),
        16_000,
        deadline=_future_deadline(),
        runtime_permit=permit,
        registry=registry,
        thread_factory=factory,
    )
    assert owner is not None
    ticket = owner.start()
    threads[0].on_receive_join = lambda: owner.abandon(
        KwsSpeakerInferenceOutcome.STOPPED
    )

    result = owner.receive(ticket)

    assert result.outcome is KwsSpeakerInferenceOutcome.STOPPED
    assert owner.snapshot().abandoned
    assert owner.snapshot().runtime_lease_current
    assert registry.current() is owner
    assert permit.snapshot().releases == 0

    threads[0].force_alive = False
    assert owner.try_reap()
    repeated = owner.receive(ticket)
    assert repeated.outcome is KwsSpeakerInferenceOutcome.STOPPED
    assert registry.current() is None
    assert permit.snapshot() == RuntimeSpeakerInferencePermitSnapshot(
        active=False,
        acquisitions=1,
        releases=1,
    )


def test_permit_release_base_exception_retains_exact_owner_for_retry() -> None:
    permit = _OneShotReleaseFailurePermit()
    registry = KwsSpeakerInferenceRegistry()
    owner = try_claim_kws_speaker_inference_owner(
        _enrolled_gate(lambda _samples, _sample_rate: [1.0, 0.0]),
        (_owned_clip(),),
        16_000,
        deadline=_future_deadline(),
        runtime_permit=permit,
        registry=registry,
        thread_factory=lambda **kwargs: _InlineThread(**kwargs),
    )
    assert owner is not None

    result = owner.receive(owner.start())

    assert result.outcome is KwsSpeakerInferenceOutcome.ERROR
    assert result.error_type == "WorkerReapError"
    retained = owner.snapshot()
    assert retained.state is KwsSpeakerInferenceOwnerState.RETAINED
    assert retained.error_type == "KeyboardInterrupt"
    assert retained.runtime_lease_current
    assert not retained.reaped
    assert registry.current() is owner
    assert permit.snapshot() == RuntimeSpeakerInferencePermitSnapshot(
        active=True,
        acquisitions=1,
        releases=0,
    )

    assert owner.try_reap()
    assert permit.release_calls == 2
    assert registry.current() is None
    assert permit.snapshot() == RuntimeSpeakerInferencePermitSnapshot(
        active=False,
        acquisitions=1,
        releases=1,
    )


def test_timeout_before_worker_entry_forbids_every_native_step(monkeypatch) -> None:
    clock = _Clock()
    monkeypatch.setattr(owner_module.time, "monotonic", clock)
    permit = RuntimeSpeakerInferencePermit()
    registry = KwsSpeakerInferenceRegistry()
    threads = []
    native_calls = []

    def factory(**kwargs):
        thread = _DeferredThread(**kwargs)
        threads.append(thread)
        return thread

    owner = try_claim_kws_speaker_inference_owner(
        _enrolled_gate(
            lambda _samples, _sample_rate: native_calls.append("native") or [1.0, 0.0]
        ),
        (_owned_clip(), _owned_clip(0.5)),
        16_000,
        deadline=1.0,
        runtime_permit=permit,
        registry=registry,
        thread_factory=factory,
    )
    assert owner is not None
    ticket = owner.start()
    clock.value = 1.0

    result = owner.receive(ticket)
    assert result.outcome is KwsSpeakerInferenceOutcome.TIMEOUT
    assert native_calls == []
    assert owner.snapshot().native_steps == 0
    assert permit.snapshot().active

    threads[0].run_target()
    assert native_calls == []
    assert owner.snapshot().worker_returned
    assert owner.try_reap()
    assert not permit.snapshot().active
    assert registry.current() is None


@pytest.mark.parametrize(
    ("abandon_with", "expected_outcome"),
    [
        ("deadline", KwsSpeakerInferenceOutcome.TIMEOUT),
        ("stop", KwsSpeakerInferenceOutcome.STOPPED),
    ],
)
def test_abandon_between_words_forbids_second_native_step(
    monkeypatch,
    abandon_with,
    expected_outcome,
) -> None:
    clock = _Clock()
    monkeypatch.setattr(owner_module.time, "monotonic", clock)
    permit = RuntimeSpeakerInferencePermit()
    registry = KwsSpeakerInferenceRegistry()
    owner_box = []
    native_calls = []

    def embed(_samples, _sample_rate):
        native_calls.append(len(native_calls) + 1)
        if abandon_with == "deadline":
            clock.value = 1.0
        else:
            owner_box[0].abandon(KwsSpeakerInferenceOutcome.STOPPED)
        return [1.0, 0.0]

    owner = try_claim_kws_speaker_inference_owner(
        _enrolled_gate(embed),
        (_owned_clip(), _owned_clip(0.5)),
        16_000,
        deadline=1.0,
        runtime_permit=permit,
        registry=registry,
        thread_factory=lambda **kwargs: _InlineThread(**kwargs),
    )
    assert owner is not None
    owner_box.append(owner)

    ticket = owner.start()
    result = owner.receive(ticket)

    assert result.outcome is expected_outcome
    assert native_calls == [1]
    assert owner.snapshot().native_steps == 1
    assert owner.try_reap()
    assert not permit.snapshot().active


def test_stop_wakes_without_join_and_late_native_result_releases_inputs() -> None:
    native_entered = threading.Event()
    release_native = threading.Event()
    native_threads = []
    permit = RuntimeSpeakerInferencePermit()
    registry = KwsSpeakerInferenceRegistry()

    def embed(_samples, _sample_rate):
        native_entered.set()
        assert release_native.wait(timeout=2.0)
        return [1.0, 0.0]

    def factory(**kwargs):
        thread = _ObservedNativeThread(**kwargs)
        native_threads.append(thread)
        return thread

    gate = _enrolled_gate(embed)
    clip = _owned_clip()
    gate_ref = weakref.ref(gate)
    clip_ref = weakref.ref(clip)
    owner = try_claim_kws_speaker_inference_owner(
        gate,
        (clip,),
        16_000,
        deadline=_future_deadline(),
        runtime_permit=permit,
        registry=registry,
        thread_factory=factory,
    )
    assert owner is not None
    ticket = owner.start()
    assert native_entered.wait(timeout=2.0)
    del gate
    del clip

    assert owner.abandon(KwsSpeakerInferenceOutcome.STOPPED)
    result = owner.receive(ticket)
    assert result.outcome is KwsSpeakerInferenceOutcome.STOPPED
    assert native_threads[0].join_calls == []
    assert owner.snapshot().thread_alive
    assert owner.snapshot().runtime_lease_current
    assert not owner.try_reap()
    assert gate_ref() is not None
    assert clip_ref() is not None
    assert (
        try_claim_kws_speaker_inference_owner(
            _enrolled_gate(lambda _samples, _sample_rate: [1.0, 0.0]),
            (_owned_clip(0.5),),
            16_000,
            deadline=_future_deadline(),
            runtime_permit=permit,
            registry=registry,
        )
        is None
    )

    release_native.set()
    native_threads[0].await_return()
    assert owner.snapshot().worker_returned
    gc.collect()
    assert gate_ref() is None
    assert clip_ref() is None
    assert permit.snapshot().active
    assert owner.try_reap()

    assert registry.current() is None
    assert permit.snapshot() == RuntimeSpeakerInferencePermitSnapshot(
        active=False,
        acquisitions=1,
        releases=1,
    )
