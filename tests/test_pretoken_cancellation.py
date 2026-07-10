"""Deterministic regressions for cancellation before an LLM's first token.

The fake model deliberately blocks *inside* ``stream().__next__``.  That is the
important failure shape: setting an ``AgentTask.cancel_event`` cannot wake the
thread that is waiting for the provider's first token.  Barge-in still has to
retire the task worker promptly, without waiting for the provider gate, and any
token the abandoned provider produces later must be unable to reach TTS.

These are headless control-plane tests.  They exercise no microphone, speaker,
model server, network, or recorded audio.
"""

from __future__ import annotations

import threading
import time
from typing import Iterator, Optional, Sequence

from always_on_agent.capabilities import CapabilityRegistry, CapabilityResult
from always_on_agent.events import EventKind, Mode
from always_on_agent.tasks import AgentTask, TaskRuntime

from core.engines.scripted import ScriptedEngine
from core.metrics import LLM_FIRST_TOKEN
from core.runtime import VoiceRuntime


_HEALTHY_PROMPT = "healthy turn after the interruption storm"
_HEALTHY_REPLY = "The healthy turn completed."


class _BlockedCall:
    """One provider invocation, controlled independently by the test thread."""

    def __init__(self, prompt: str):
        self.prompt = prompt
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()


class _PreTokenBlockingLLM:
    """Current-contract LLM fake with a provider-style first-token block.

    Blocked calls ignore task cancellation because the existing ``LLMClient``
    contract does not pass a cancel event to providers.  Once explicitly
    released, each abandoned call adversarially yields a complete stale
    sentence.  A correct cancellation layer has already detached its consumer,
    so that sentence cannot be emitted as speech.
    """

    def __init__(self):
        self._calls: list[_BlockedCall] = []
        self._lock = threading.Lock()

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[object]] = None,
        history: Optional[Sequence[object]] = None,
    ) -> str:  # pragma: no cover - the answering capability uses stream()
        return _HEALTHY_REPLY

    def stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[object]] = None,
        history: Optional[Sequence[object]] = None,
    ) -> Iterator[str]:
        call = _BlockedCall(prompt)
        with self._lock:
            self._calls.append(call)
        call.started.set()
        try:
            if prompt != _HEALTHY_PROMPT:
                # A long finite bound keeps a broken test from leaving a daemon
                # around forever.  Normal cleanup releases every call explicitly.
                if not call.release.wait(timeout=15.0):
                    raise TimeoutError("test provider gate was never released")
                yield f"STALE reply from {prompt}."
                return
            yield _HEALTHY_REPLY
        finally:
            call.finished.set()

    def calls(self) -> list[_BlockedCall]:
        with self._lock:
            return list(self._calls)

    def release_all(self) -> None:
        for call in self.calls():
            call.release.set()


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _call_count(llm: _PreTokenBlockingLLM) -> int:
    return len(llm.calls())


def _task_workers_are_gone(runtime: VoiceRuntime) -> bool:
    return (
        not runtime.supervisor.state.active_tasks
        and not runtime.supervisor.state.queued_tasks
        and runtime.supervisor.tasks.active_count == 0
    )


def _run_cancellable(
    runtime: TaskRuntime,
    task: AgentTask,
    invoke,
    outcome: dict[str, object],
) -> None:
    """Call the private coordinator seam and retain its terminal outcome."""

    try:
        outcome["result"] = runtime._invoke_cancellable(task, invoke)  # noqa: SLF001
    except BaseException as exc:  # cancellation is the expected control flow
        outcome["error"] = exc


def test_cancel_before_child_atomic_claim_skips_provider_and_reuses_slot():
    """Cancellation before the child claim must not consume provider capacity.

    The provider child is paused immediately before ``claim_invocation_start``.
    This pins the narrow window after ``Thread.start()`` transferred semaphore
    ownership but before the child linearized the side-effecting invocation.
    """

    runtime = TaskRuntime(
        lambda _event: None,
        CapabilityRegistry(),
        max_active_tasks=1,
    )
    cancelled_task = AgentTask(Mode.ASSISTANT, "cancel before atomic claim")
    claim_entered = threading.Event()
    allow_claim = threading.Event()
    provider_calls: list[str] = []
    cancelled_outcome: dict[str, object] = {}
    replacement_outcome: dict[str, object] = {}
    replacement_task = AgentTask(Mode.ASSISTANT, "reuse released slot")

    original_claim = cancelled_task.claim_invocation_start

    def gated_claim() -> bool:
        claim_entered.set()
        if not allow_claim.wait(timeout=5.0):
            raise TimeoutError("test did not release the atomic-claim gate")
        return original_claim()

    # Instance-level seam: only this task's provider child is paused.
    cancelled_task.claim_invocation_start = gated_claim  # type: ignore[method-assign]

    def cancelled_provider() -> CapabilityResult:
        provider_calls.append("cancelled")
        return CapabilityResult(True, "should never run")

    cancelled_coordinator = threading.Thread(
        target=_run_cancellable,
        args=(runtime, cancelled_task, cancelled_provider, cancelled_outcome),
        name="cancel-before-claim-coordinator",
        daemon=True,
    )
    replacement_coordinator = threading.Thread(
        target=_run_cancellable,
        args=(
            runtime,
            replacement_task,
            lambda: (
                provider_calls.append("replacement")
                or CapabilityResult(True, "slot reused")
            ),
            replacement_outcome,
        ),
        name="replacement-coordinator",
        daemon=True,
    )

    try:
        cancelled_coordinator.start()
        assert claim_entered.wait(timeout=2.0), (
            "provider child never reached the pre-claim race gate"
        )

        cancelled_task.cancel()
        allow_claim.set()
        cancelled_coordinator.join(timeout=2.0)
        assert not cancelled_coordinator.is_alive()
        assert provider_calls == [], "cancelled child invoked its provider late"
        assert cancelled_outcome.get("error", object()).__class__.__name__ == "_TaskCancelled"

        # If the claim-false path forgot to release its semaphore slot, this
        # coordinator cannot finish.  A successful call proves the slot is live.
        replacement_coordinator.start()
        replacement_coordinator.join(timeout=2.0)
        assert not replacement_coordinator.is_alive(), (
            "the cancelled pre-claim child leaked its provider slot"
        )
        assert provider_calls == ["replacement"]
        result = replacement_outcome.get("result")
        assert isinstance(result, CapabilityResult)
        assert result.text == "slot reused"
    finally:
        cancelled_task.cancel()
        replacement_task.cancel()
        allow_claim.set()
        if cancelled_coordinator.ident is not None:
            cancelled_coordinator.join(timeout=2.0)
        if replacement_coordinator.ident is not None:
            replacement_coordinator.join(timeout=2.0)


def test_cancelled_overcap_waiter_cannot_start_after_provider_slot_releases():
    """A cancelled semaphore waiter must not invoke after capacity returns."""

    max_slots = 2
    runtime = TaskRuntime(
        lambda _event: None,
        CapabilityRegistry(),
        max_active_tasks=max_slots,
    )
    overcap_thread_name = "over-cap-coordinator"

    class _ObservedSlots:
        def __init__(self) -> None:
            self._semaphore = threading.BoundedSemaphore(max_slots)
            self.overcap_waiting = threading.Event()
            self.allow_overcap_acquire = threading.Event()

        def acquire(self, timeout=None):
            if threading.current_thread().name == overcap_thread_name:
                self.overcap_waiting.set()
                if not self.allow_overcap_acquire.wait(timeout=timeout):
                    return False
            return self._semaphore.acquire(timeout=timeout)

        def release(self) -> None:
            self._semaphore.release()

    slots = _ObservedSlots()
    runtime._invocation_slots = slots  # noqa: SLF001 - observe the bulkhead seam
    # Keep the over-cap acquire blocked until the test explicitly releases a
    # provider.  This removes the normal 10 ms polling timeout from the race.
    runtime._CANCEL_POLL_SEC = 5.0  # noqa: SLF001

    blocker_tasks = [
        AgentTask(Mode.ASSISTANT, f"occupy slot {index}")
        for index in range(max_slots)
    ]
    blocker_started = [threading.Event() for _ in range(max_slots)]
    blocker_release = [threading.Event() for _ in range(max_slots)]
    invocations: list[str] = []
    invocations_lock = threading.Lock()
    blocker_outcomes = [dict() for _ in range(max_slots)]
    blocker_coordinators: list[threading.Thread] = []

    def blocker(index: int):
        def invoke() -> CapabilityResult:
            with invocations_lock:
                invocations.append(f"blocker-{index}")
            blocker_started[index].set()
            if not blocker_release[index].wait(timeout=5.0):
                raise TimeoutError("test did not release saturated provider")
            return CapabilityResult(True, f"released {index}")

        return invoke

    for index, task in enumerate(blocker_tasks):
        coordinator = threading.Thread(
            target=_run_cancellable,
            args=(runtime, task, blocker(index), blocker_outcomes[index]),
            name=f"slot-holder-{index}",
            daemon=True,
        )
        blocker_coordinators.append(coordinator)

    overcap_task = AgentTask(Mode.ASSISTANT, "must remain uninvoked")
    overcap_outcome: dict[str, object] = {}
    late_started = threading.Event()

    def late_provider() -> CapabilityResult:
        with invocations_lock:
            invocations.append("late-overcap")
        late_started.set()
        return CapabilityResult(True, "too late")

    overcap_coordinator = threading.Thread(
        target=_run_cancellable,
        args=(runtime, overcap_task, late_provider, overcap_outcome),
        name=overcap_thread_name,
        daemon=True,
    )

    try:
        for coordinator in blocker_coordinators:
            coordinator.start()
        assert all(event.wait(timeout=2.0) for event in blocker_started)
        with invocations_lock:
            assert len(invocations) == max_slots, (
                "test did not saturate exactly the configured provider slots"
            )

        overcap_coordinator.start()
        assert slots.overcap_waiting.wait(timeout=2.0), (
            "over-cap coordinator never blocked on provider capacity"
        )

        # Cancellation lands while acquire() is blocked.  Releasing exactly one
        # occupied provider makes capacity available; once acquire may return,
        # the post-acquire guard must give the slot back without invoking late.
        overcap_task.cancel()
        blocker_release[0].set()
        blocker_coordinators[0].join(timeout=2.0)
        assert not blocker_coordinators[0].is_alive(), (
            "the selected provider did not release its occupied slot"
        )
        slots.allow_overcap_acquire.set()
        overcap_coordinator.join(timeout=2.0)

        assert not overcap_coordinator.is_alive()
        assert overcap_outcome.get("error", object()).__class__.__name__ == "_TaskCancelled"
        assert not late_started.is_set()
        with invocations_lock:
            assert sorted(invocations) == ["blocker-0", "blocker-1"]
        assert not blocker_release[1].is_set(), (
            "the second saturated provider must remain gated during the assertion"
        )
    finally:
        overcap_task.cancel()
        slots.allow_overcap_acquire.set()
        for task in blocker_tasks:
            task.cancel()
        for event in blocker_release:
            event.set()
        for coordinator in blocker_coordinators:
            if coordinator.ident is not None:
                coordinator.join(timeout=2.0)
        if overcap_coordinator.ident is not None:
            overcap_coordinator.join(timeout=2.0)


def test_barge_retires_task_worker_while_provider_is_still_blocked_pre_token():
    """Cancellation must not wait for the provider's first-token read.

    ``active_tasks`` alone is insufficient evidence: ``cancel_all`` could merely
    hide a still-running task.  The TaskRuntime thread registry must also reach
    zero while the provider gate remains closed and its iterator unfinished.
    """

    llm = _PreTokenBlockingLLM()
    engine = ScriptedEngine()
    runtime = VoiceRuntime(engine, llm, start_mode=Mode.ASSISTANT, stream_tts=True)
    runtime.start(run_bus=True)
    try:
        engine.final("blocked before first token")
        assert _wait_until(lambda: _call_count(llm) == 1), "LLM stream never started"
        call = llm.calls()[0]
        assert call.started.is_set()
        assert not call.release.is_set()
        assert runtime.supervisor.tasks.active_count == 1
        task_id = next(iter(runtime.supervisor.state.active_tasks))
        task_runtime = runtime.supervisor.tasks
        with task_runtime._threads_lock:  # noqa: SLF001 - capture the real worker
            coordinator = task_runtime._threads[task_id]  # noqa: SLF001

        engine.barge_in()

        coordinator.join(timeout=2.0)
        assert not coordinator.is_alive(), (
            "the task registry was reaped but its coordinator thread stayed alive"
        )
        assert _wait_until(lambda: _task_workers_are_gone(runtime)), (
            "barge-in left the task worker blocked in the provider's pre-token read"
        )
        assert not call.release.is_set(), "the test, not production, owns this gate"
        assert not call.finished.is_set(), (
            "the provider should still be blocked; task cleanup must not depend on it"
        )

        # Let the abandoned provider produce a stale sentence.  It must have no
        # remaining path to the streaming emitter or engine.speak().
        call.release.set()
        assert _wait_until(call.finished.is_set)
        assert _wait_until(runtime.bus.idle)
        assert engine.spoken == []
        terminal = {
            event.kind
            for event in runtime.supervisor.state.event_log
            if event.payload.get("task_id") == task_id
        }
        assert EventKind.TASK_CANCELLED in terminal
        assert EventKind.TASK_COMPLETED not in terminal
        assert EventKind.TASK_FAILED not in terminal
    finally:
        llm.release_all()
        _wait_until(lambda: all(call.finished.is_set() for call in llm.calls()))
        runtime.stop()


def test_cancelled_late_first_token_cannot_stamp_replacement_turn_metrics():
    """A stale first token must not make a newer turn look model-responsive.

    MetricsRecorder tracks the currently open turn rather than a task ID.  This
    ordering therefore keeps the replacement provider blocked while the
    cancelled provider wakes: its late token must neither stamp
    ``LLM_FIRST_TOKEN`` on the replacement nor seed the local TTFT EWMA.  Only
    the replacement's own token may do those things.
    """

    llm = _PreTokenBlockingLLM()
    engine = ScriptedEngine()
    runtime = VoiceRuntime(engine, llm, start_mode=Mode.ASSISTANT, stream_tts=True)
    runtime.start(run_bus=True)
    try:
        engine.final("cancelled metrics turn")
        assert _wait_until(lambda: _call_count(llm) == 1)
        cancelled_call = llm.calls()[0]

        engine.barge_in()
        assert _wait_until(lambda: _task_workers_are_gone(runtime))

        replacement_prompt = "replacement still blocked before first token"
        engine.final(replacement_prompt)
        assert _wait_until(lambda: _call_count(llm) == 2)
        replacement_call = llm.calls()[1]
        assert not replacement_call.release.is_set()
        assert LLM_FIRST_TOKEN not in runtime.metrics.records()[-1].stamps
        assert runtime.metrics.recent_ttft_ms() is None

        # Wake only the abandoned provider.  Its first token belongs to the
        # cancelled task even though the replacement is now the open metric turn.
        cancelled_call.release.set()
        assert _wait_until(cancelled_call.finished.is_set)
        assert not replacement_call.finished.is_set()
        assert LLM_FIRST_TOKEN not in runtime.metrics.records()[-1].stamps
        assert runtime.metrics.recent_ttft_ms() is None

        # The replacement's own token is authoritative for its metrics.
        replacement_call.release.set()
        assert runtime.wait_idle(timeout=2.0)
        assert replacement_call.finished.is_set()
        assert LLM_FIRST_TOKEN in runtime.metrics.records()[-1].stamps
        assert runtime.metrics.recent_ttft_ms() is not None
        assert engine.spoken == [f"STALE reply from {replacement_prompt}."]
    finally:
        llm.release_all()
        _wait_until(lambda: all(call.finished.is_set() for call in llm.calls()))
        runtime.stop()


def test_repeated_pretoken_barges_do_not_exhaust_workers_or_poison_next_turn():
    """A barge storm beyond the worker cap must remain recoverable.

    Provider invocations that are already blocked may outlive their cancelled
    task coordinators, but their own concurrency must remain bounded.  Once that
    provider budget is full, later coordinators may wait for a slot; barge-in
    must still cancel those coordinators promptly.  Releasing the abandoned
    streams then tries to inject stale speech; finally, a normal turn proves the
    answering path still admits work and produces TTS.
    """

    llm = _PreTokenBlockingLLM()
    engine = ScriptedEngine()
    runtime = VoiceRuntime(engine, llm, start_mode=Mode.ASSISTANT, stream_tts=True)
    runtime.start(run_bus=True)
    task_runtime = runtime.supervisor.tasks
    attempts = task_runtime.max_active_tasks + 2
    try:
        for index in range(attempts):
            calls_before = _call_count(llm)
            engine.final(f"blocked turn {index}")
            assert _wait_until(
                lambda: bool(runtime.supervisor.state.active_tasks)
                or _call_count(llm) > calls_before
            ), (
                f"turn {index} was neither admitted as a task nor entered "
                "into the provider"
            )

            # Give an available provider slot a deterministic chance to enter
            # the fake.  Once the bounded provider budget is full this times out
            # harmlessly: that coordinator is waiting for capacity and is the
            # exact over-cap cancellation path under test.
            _wait_until(lambda: _call_count(llm) > calls_before, timeout=0.25)
            assert task_runtime.active_count <= task_runtime.max_active_tasks

            engine.barge_in()
            assert _wait_until(lambda: _task_workers_are_gone(runtime)), (
                f"pre-token barge {index} did not reclaim its task worker"
            )
            provider_calls = llm.calls()
            assert len(provider_calls) <= task_runtime.max_active_tasks, (
                "uncooperative pre-token provider calls exceeded the bounded "
                f"worker budget: {len(provider_calls)} > "
                f"{task_runtime.max_active_tasks}"
            )
            assert all(not call.finished.is_set() for call in provider_calls), (
                "provider completion accidentally, rather than cancellation, "
                "reclaimed a task worker"
            )

        blocked_calls = llm.calls()
        assert blocked_calls, "the storm never exercised a blocked provider call"
        assert len(blocked_calls) <= task_runtime.max_active_tasks
        assert all(not call.release.is_set() for call in blocked_calls)

        # Every abandoned stream now yields a complete sentence.  Their task
        # epochs/consumers are stale, so none may reach the scripted speaker.
        llm.release_all()
        assert _wait_until(
            lambda: all(call.finished.is_set() for call in blocked_calls)
        )
        assert _wait_until(runtime.bus.idle)
        assert engine.spoken == []

        # Cancellation storms must not poison admission or the model/TTS path.
        calls_before_healthy = _call_count(llm)
        engine.final(_HEALTHY_PROMPT)
        assert _wait_until(lambda: _call_count(llm) == calls_before_healthy + 1)
        assert runtime.wait_idle(timeout=2.0)
        assert task_runtime.active_count == 0
        assert engine.spoken == [_HEALTHY_REPLY]
    finally:
        llm.release_all()
        _wait_until(lambda: all(call.finished.is_set() for call in llm.calls()))
        runtime.stop()
