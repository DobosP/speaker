"""Process-wide single-flight owner for ambiguous KWS speaker inference.

The native speaker extractor has no cancellation seam in the pinned Sherpa
wrapper.  One entered call may therefore outlive the capture run that requested
it.  This module makes that failure finite and visible:

* one exact may-start/running request exists process-wide;
* its runtime permit is acquired before ``Thread.start``;
* timeout and stop abandon authority but never release an entered call;
* no successor is admitted until the exact worker returned and was reaped; and
* the worker owns only a gate, one/two bounded identity clips, and its ticket.

The capture owner remains responsible for policy reduction, source/playback
authority, the action deadline, and callback publication.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import wraps
import math
import threading
import time
from typing import Callable, Optional

import numpy as np

from .speaker_gate import (
    RuntimeSpeakerInferenceLease,
    RuntimeSpeakerInferencePermit,
    SpeakerGate,
    SpeakerSimilarityBatchReceipt,
    runtime_speaker_inference_permit,
)


class KwsSpeakerInferenceOutcome(str, Enum):
    """Closed capture-side result for one exact request."""

    COMPLETED = "completed"
    BUSY = "busy"
    TIMEOUT = "timeout"
    STOPPED = "stopped"
    ERROR = "error"


class KwsSpeakerInferenceOwnerState(str, Enum):
    """Diagnostic lifecycle state; authority uses exact object identity."""

    NEW = "new"
    STARTING = "starting"
    RUNNING = "running"
    RESULT_READY = "result_ready"
    ABANDONED = "abandoned"
    RETAINED = "retained"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True, eq=False)
class KwsSpeakerInferenceTicket:
    """Opaque exact ticket for one process-global request."""

    owner_token: object
    token: object


@dataclass(frozen=True, slots=True)
class KwsSpeakerInferenceResult:
    """One capture-owned terminal observation."""

    outcome: KwsSpeakerInferenceOutcome
    receipt: Optional[SpeakerSimilarityBatchReceipt] = None
    error_type: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, KwsSpeakerInferenceOutcome):
            raise TypeError("outcome must be a KwsSpeakerInferenceOutcome")
        if self.outcome is KwsSpeakerInferenceOutcome.COMPLETED:
            if type(self.receipt) is not SpeakerSimilarityBatchReceipt:
                raise TypeError("completed inference requires an exact receipt")
            if self.error_type:
                raise ValueError("completed inference cannot carry an error")
            return
        if self.receipt is not None:
            raise ValueError("non-completed inference cannot carry a receipt")
        if self.outcome is KwsSpeakerInferenceOutcome.ERROR:
            if type(self.error_type) is not str or not self.error_type:
                raise ValueError("error inference requires an exact error type")
        elif self.error_type:
            raise ValueError("non-error inference cannot carry an error type")


@dataclass(frozen=True, slots=True)
class KwsSpeakerInferenceOwnerSnapshot:
    """Payload-free diagnostic state for tests and lifecycle preflight."""

    state: KwsSpeakerInferenceOwnerState
    start_attempted: bool
    start_returned: bool
    worker_returned: bool
    result_claimed: bool
    abandoned: bool
    native_steps: int
    thread_alive: bool
    runtime_lease_current: bool
    reaped: bool
    error_type: str


@dataclass(slots=True)
class _KwsSpeakerInferenceWork:
    """The complete worker payload; never contains engine/callback authority."""

    gate: SpeakerGate
    clips: tuple[np.ndarray, ...]
    sample_rate: int
    ticket: KwsSpeakerInferenceTicket


class KwsSpeakerInferenceRegistry:
    """Injectable exact process-owner registry; production uses one singleton."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owner: Optional["KwsSpeakerInferenceOwner"] = None
        self._reservation: Optional[object] = None
        self._mutation_thread: Optional[int] = None
        self._mutation_leases: list[object] = []

    def current(self) -> Optional["KwsSpeakerInferenceOwner"]:
        with self._lock:
            return self._owner

    def admission_is_reserved_for_other(self) -> bool:
        with self._lock:
            return bool(
                self._reservation is not None
                or (
                    self._mutation_leases
                    and self._mutation_thread != threading.get_ident()
                )
            )

    def try_reserve(self) -> Optional[object]:
        reservation = object()
        with self._lock:
            if (
                self._owner is not None
                or self._reservation is not None
                or self._mutation_leases
            ):
                return None
            self._reservation = reservation
            return reservation

    def publish(
        self,
        reservation: object,
        owner: "KwsSpeakerInferenceOwner",
    ) -> bool:
        if (
            not isinstance(owner, KwsSpeakerInferenceOwner)
            or getattr(owner, "_registry", None) is not self
        ):
            return False
        with self._lock:
            if self._reservation is not reservation or self._owner is not None:
                return False
            self._reservation = None
            self._owner = owner
            return True

    def cancel_reservation(self, reservation: object) -> bool:
        with self._lock:
            if self._reservation is not reservation:
                return False
            self._reservation = None
            return True

    def acquire_mutation(self) -> object:
        """Fence KWS task admission across one rebuild/start mutation."""

        thread_id = threading.get_ident()
        while True:
            owner = None
            with self._lock:
                if self._mutation_leases:
                    if self._mutation_thread != thread_id:
                        raise RuntimeError(
                            "another rebuild/start mutation is already active"
                        )
                    lease = object()
                    self._mutation_leases.append(lease)
                    return lease
                if self._reservation is not None:
                    raise RuntimeError(
                        "KWS speaker inference may start during rebuild/start"
                    )
                owner = self._owner
                if owner is None:
                    lease = object()
                    self._mutation_thread = thread_id
                    self._mutation_leases.append(lease)
                    return lease
            if not owner.try_reap():
                raise RuntimeError(
                    "cannot rebuild or start while KWS speaker inference "
                    "may still be live"
                )

    def release_mutation(self, lease: object) -> bool:
        with self._lock:
            if (
                self._mutation_thread != threading.get_ident()
                or not self._mutation_leases
                or self._mutation_leases[-1] is not lease
            ):
                return False
            self._mutation_leases.pop()
            if not self._mutation_leases:
                self._mutation_thread = None
            return True

    def release(self, owner: "KwsSpeakerInferenceOwner") -> bool:
        if (
            not isinstance(owner, KwsSpeakerInferenceOwner)
            or getattr(owner, "_registry", None) is not self
        ):
            return False
        with self._lock:
            if self._owner is not owner:
                return False
            self._owner = None
            return True


_PROCESS_OWNER_REGISTRY = KwsSpeakerInferenceRegistry()


def guard_kws_speaker_inference_mutation(method):
    """Run one lifecycle mutation under the process registry's exact guard."""

    @wraps(method)
    def guarded(*args, **kwargs):
        lease = _PROCESS_OWNER_REGISTRY.acquire_mutation()
        try:
            return method(*args, **kwargs)
        finally:
            if not _PROCESS_OWNER_REGISTRY.release_mutation(lease):
                raise RuntimeError("KWS speaker-inference mutation guard was lost")

    return guarded


def _thread_alive(thread: object) -> bool:
    if thread is None:
        return False
    alive = getattr(thread, "is_alive", None)
    if not callable(alive):
        return True
    try:
        return bool(alive())
    except BaseException:
        return True


def _run_kws_speaker_inference_owner(owner: "KwsSpeakerInferenceOwner") -> None:
    """Static target: the thread retains only the deliberately minimal owner."""

    owner._run_worker()


class KwsSpeakerInferenceOwner:
    """Exact per-request worker retained through native return and reaping."""

    def __init__(
        self,
        gate: SpeakerGate,
        clips: tuple[np.ndarray, ...],
        sample_rate: int,
        *,
        deadline: float,
        runtime_permit: RuntimeSpeakerInferencePermit,
        registry: KwsSpeakerInferenceRegistry,
        thread_factory: Optional[Callable[..., object]] = None,
    ) -> None:
        if not isinstance(gate, SpeakerGate):
            raise TypeError("gate must be a SpeakerGate")
        if type(clips) is not tuple or not 0 < len(clips) <= 2:
            raise ValueError("clips must contain one or two owned arrays")
        total_samples = 0
        for clip in clips:
            if type(clip) is not np.ndarray:
                raise TypeError("each clip must be an exact ndarray")
            if (
                clip.ndim != 1
                or clip.dtype != np.dtype("float32")
                or not clip.flags.c_contiguous
                or not clip.flags.owndata
                or clip.flags.writeable
                or clip.size <= 0
                or not bool(np.all(np.isfinite(clip)))
            ):
                raise ValueError(
                    "each clip must be finite, owned, read-only C-float32 PCM"
                )
            count = int(clip.size)
            if total_samples > np.iinfo(np.intp).max - count:
                raise OverflowError("speaker clip sample count overflow")
            total_samples += count
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
            raise TypeError("sample_rate must be an integer")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not isinstance(runtime_permit, RuntimeSpeakerInferencePermit):
            raise TypeError("runtime_permit must be a RuntimeSpeakerInferencePermit")
        if isinstance(deadline, bool) or not isinstance(deadline, (int, float)):
            raise TypeError("deadline must be numeric")
        deadline = float(deadline)
        if not math.isfinite(deadline) or deadline < 0.0:
            raise ValueError("deadline must be finite and non-negative")
        if not isinstance(registry, KwsSpeakerInferenceRegistry):
            raise TypeError("registry must be a KwsSpeakerInferenceRegistry")
        if thread_factory is not None and not callable(thread_factory):
            raise TypeError("thread_factory must be callable or None")

        self._condition = threading.Condition()
        self._owner_token = object()
        self._ticket = KwsSpeakerInferenceTicket(self._owner_token, object())
        self._state = KwsSpeakerInferenceOwnerState.NEW
        self._deadline = deadline
        self._runtime_permit = runtime_permit
        self._registry = registry
        self._runtime_lease: Optional[RuntimeSpeakerInferenceLease] = None
        self._permit_released = False
        self._work: Optional[_KwsSpeakerInferenceWork] = _KwsSpeakerInferenceWork(
            gate,
            clips,
            sample_rate,
            self._ticket,
        )
        self._pending_result: Optional[KwsSpeakerInferenceResult] = None
        self._start_attempted = False
        self._start_returned = False
        self._worker_returned = False
        self._result_claimed = False
        self._abandoned = False
        self._abandon_outcome: Optional[KwsSpeakerInferenceOutcome] = None
        self._native_steps = 0
        self._native_step_limit = len(clips)
        self._reaped = False
        self._reaping = False
        self._error_type = ""
        self._thread_factory = (
            threading.Thread if thread_factory is None else thread_factory
        )
        self._thread = None

    def _install_thread(self) -> None:
        with self._condition:
            if self._thread is not None or self._start_attempted or self._reaped:
                raise RuntimeError("inference worker thread was already installed")
            factory = self._thread_factory
        thread = factory(
            target=_run_kws_speaker_inference_owner,
            args=(self,),
            name="speaker-kws-speaker-inference-owner",
            daemon=True,
        )
        with self._condition:
            if self._thread is not None or self._start_attempted or self._reaped:
                raise RuntimeError("inference worker thread installation raced")
            self._thread = thread
            self._thread_factory = None

    @property
    def ticket(self) -> KwsSpeakerInferenceTicket:
        return self._ticket

    def _install_runtime_lease(self, lease: RuntimeSpeakerInferenceLease) -> None:
        if not isinstance(lease, RuntimeSpeakerInferenceLease):
            raise TypeError("lease must be a RuntimeSpeakerInferenceLease")
        with self._condition:
            if self._runtime_lease is not None or self._start_attempted:
                raise RuntimeError("runtime lease was already installed")
            if not self._runtime_permit.is_current(lease):
                raise RuntimeError("runtime lease is not current")
            self._runtime_lease = lease

    def start(self) -> KwsSpeakerInferenceTicket:
        """Start once; any escaping ``BaseException`` keeps ownership retained."""

        with self._condition:
            if self._runtime_lease is None:
                raise RuntimeError("runtime lease was not installed")
            if self._state is not KwsSpeakerInferenceOwnerState.NEW:
                raise RuntimeError("inference owner was already started")
            if self._thread is None:
                raise RuntimeError("inference worker thread was not installed")
            if self._abandoned or self._reaping or self._reaped:
                raise RuntimeError("inference owner was stopped before start")
            self._start_attempted = True
            self._state = KwsSpeakerInferenceOwnerState.STARTING
            thread = self._thread
        try:
            thread.start()
        except BaseException as exc:
            with self._condition:
                self._error_type = type(exc).__name__
                self._state = KwsSpeakerInferenceOwnerState.RETAINED
                # If the child has not taken the payload, it must not do so
                # after an ambiguous parent-side launch failure. A prior stop
                # or timeout remains the immutable first terminal outcome.
                self._abandon_locked(KwsSpeakerInferenceOutcome.ERROR)
            raise
        with self._condition:
            self._start_returned = True
            self._condition.notify_all()
        return self._ticket

    def enter_native_step(self, ticket: KwsSpeakerInferenceTicket) -> bool:
        """Linearize one word's logical native entry against stop/timeout."""

        with self._condition:
            if time.monotonic() >= self._deadline and not self._abandoned:
                self._abandon_locked(KwsSpeakerInferenceOutcome.TIMEOUT)
            if (
                ticket is not self._ticket
                or self._abandoned
                or self._worker_returned
                or self._state is not KwsSpeakerInferenceOwnerState.RUNNING
                or self._native_steps >= self._native_step_limit
            ):
                return False
            self._native_steps += 1
            return True

    def abandon(
        self,
        outcome: KwsSpeakerInferenceOutcome = KwsSpeakerInferenceOutcome.STOPPED,
    ) -> bool:
        """Abandon effects and wake capture; never release a possibly-entered call."""

        if type(outcome) is not KwsSpeakerInferenceOutcome or outcome not in (
            KwsSpeakerInferenceOutcome.STOPPED,
            KwsSpeakerInferenceOutcome.TIMEOUT,
            KwsSpeakerInferenceOutcome.ERROR,
        ):
            raise ValueError("abandon outcome must be stopped, timeout, or error")
        close_unstarted = False
        with self._condition:
            if self._reaped:
                return False
            changed = not self._abandoned
            self._abandon_locked(outcome)
            close_unstarted = not self._start_attempted
            if close_unstarted:
                self._worker_returned = True
        if close_unstarted:
            self.try_reap()
        return changed

    def receive(
        self,
        ticket: KwsSpeakerInferenceTicket,
    ) -> KwsSpeakerInferenceResult:
        """Wait boundedly, claim one ready result, then require exact reaping."""

        if ticket is not self._ticket:
            raise RuntimeError("inference ticket is not current")

        claimed: Optional[KwsSpeakerInferenceResult] = None
        with self._condition:
            if self._reaped:
                if self._abandoned:
                    return self._abandoned_result_locked()
                return KwsSpeakerInferenceResult(
                    KwsSpeakerInferenceOutcome.ERROR,
                    error_type="InferenceOwnerAlreadyReaped",
                )
            if self._result_claimed:
                return KwsSpeakerInferenceResult(
                    KwsSpeakerInferenceOutcome.ERROR,
                    error_type="InferenceResultAlreadyClaimed",
                )
            if self._abandoned:
                return self._abandoned_result_locked()
            if not self._start_returned:
                raise RuntimeError("inference owner has not started successfully")
            while True:
                if self._reaped:
                    if self._abandoned:
                        return self._abandoned_result_locked()
                    return KwsSpeakerInferenceResult(
                        KwsSpeakerInferenceOutcome.ERROR,
                        error_type="InferenceOwnerAlreadyReaped",
                    )
                if self._abandoned:
                    return self._abandoned_result_locked()
                if self._worker_returned and self._pending_result is None:
                    return KwsSpeakerInferenceResult(
                        KwsSpeakerInferenceOutcome.ERROR,
                        error_type="InferenceResultAlreadyClaimed",
                    )
                remaining = self._deadline - time.monotonic()
                if remaining <= 0.0:
                    self._abandon_locked(KwsSpeakerInferenceOutcome.TIMEOUT)
                    return KwsSpeakerInferenceResult(KwsSpeakerInferenceOutcome.TIMEOUT)
                if self._worker_returned and self._pending_result is not None:
                    self._result_claimed = True
                    claimed = self._pending_result
                    break
                self._condition.wait(remaining)

        remaining = max(0.0, self._deadline - time.monotonic())
        try:
            thread = self._thread
            if thread is None:
                raise RuntimeError("inference worker thread disappeared")
            thread.join(remaining)
        except BaseException as exc:
            with self._condition:
                self._result_claimed = False
                self._error_type = type(exc).__name__
                self._state = KwsSpeakerInferenceOwnerState.RETAINED
                self._abandon_locked(KwsSpeakerInferenceOutcome.ERROR)
                return self._abandoned_result_locked()
        if _thread_alive(thread):
            with self._condition:
                self._result_claimed = False
                self._abandon_locked(KwsSpeakerInferenceOutcome.TIMEOUT)
                return self._abandoned_result_locked()

        with self._condition:
            if self._abandoned:
                self._result_claimed = False
                return self._abandoned_result_locked()
            if time.monotonic() >= self._deadline:
                self._result_claimed = False
                self._abandon_locked(KwsSpeakerInferenceOutcome.TIMEOUT)
                return KwsSpeakerInferenceResult(KwsSpeakerInferenceOutcome.TIMEOUT)
            if claimed is None or self._pending_result is not claimed:
                self._result_claimed = False
                return KwsSpeakerInferenceResult(
                    KwsSpeakerInferenceOutcome.ERROR,
                    error_type="ResultOwnershipError",
                )
            self._pending_result = None
            self._result_claimed = False
        if not self.try_reap():
            return KwsSpeakerInferenceResult(
                KwsSpeakerInferenceOutcome.ERROR,
                error_type="WorkerReapError",
            )
        return claimed

    def try_reap(self) -> bool:
        """Release a proven-finished exact owner and its process registry slot."""

        with self._condition:
            if self._reaped:
                return True
            if self._reaping or not self._worker_returned or self._result_claimed:
                return False
            if self._pending_result is not None and not self._abandoned:
                if time.monotonic() < self._deadline:
                    return False
                self._abandon_locked(KwsSpeakerInferenceOutcome.TIMEOUT)
            self._reaping = True
            thread = self._thread
            lease_to_release = None if self._permit_released else self._runtime_lease
            self._work = None
            if self._abandoned:
                self._pending_result = None
        if _thread_alive(thread):
            with self._condition:
                self._reaping = False
            return False
        if thread is not None and self._start_attempted:
            join = getattr(thread, "join", None)
            if not callable(join):
                with self._condition:
                    self._reaping = False
                    self._error_type = "WorkerJoinUnavailable"
                    self._state = KwsSpeakerInferenceOwnerState.RETAINED
                return False
            try:
                join(0.0)
            except BaseException as exc:
                with self._condition:
                    self._reaping = False
                    self._error_type = type(exc).__name__
                    self._state = KwsSpeakerInferenceOwnerState.RETAINED
                return False
        if lease_to_release is not None:
            try:
                permit_released = self._runtime_permit.release(lease_to_release)
            except BaseException as exc:
                permit_released = False
                permit_error = type(exc).__name__
            else:
                permit_error = "RuntimePermitReleaseError"
            if not permit_released:
                with self._condition:
                    self._reaping = False
                    self._error_type = permit_error
                    self._state = KwsSpeakerInferenceOwnerState.RETAINED
                return False
            with self._condition:
                if self._runtime_lease is lease_to_release:
                    self._runtime_lease = None
                self._permit_released = True
        try:
            registry_released = self._registry.release(self)
        except BaseException as exc:
            registry_released = False
            registry_error = type(exc).__name__
        else:
            registry_error = "RegistryReleaseError"
        with self._condition:
            self._reaping = False
            if not registry_released:
                self._error_type = registry_error
                self._state = KwsSpeakerInferenceOwnerState.RETAINED
                return False
            self._reaped = True
            self._state = KwsSpeakerInferenceOwnerState.CLOSED
            self._condition.notify_all()
            return True

    def snapshot(self) -> KwsSpeakerInferenceOwnerSnapshot:
        with self._condition:
            lease = self._runtime_lease
            state = self._state
            start_attempted = self._start_attempted
            start_returned = self._start_returned
            worker_returned = self._worker_returned
            result_claimed = self._result_claimed
            abandoned = self._abandoned
            native_steps = self._native_steps
            reaped = self._reaped
            error_type = self._error_type
        # Diagnostic observation only: never hold the lifecycle Condition while
        # acquiring the independently injectable process-permit lock.
        runtime_lease_current = bool(
            lease is not None and self._runtime_permit.is_current(lease)
        )
        return KwsSpeakerInferenceOwnerSnapshot(
            state=state,
            start_attempted=start_attempted,
            start_returned=start_returned,
            worker_returned=worker_returned,
            result_claimed=result_claimed,
            abandoned=abandoned,
            native_steps=native_steps,
            thread_alive=_thread_alive(self._thread),
            runtime_lease_current=runtime_lease_current,
            reaped=reaped,
            error_type=error_type,
        )

    def _take_work(self) -> Optional[_KwsSpeakerInferenceWork]:
        with self._condition:
            if self._state is KwsSpeakerInferenceOwnerState.STARTING:
                self._state = KwsSpeakerInferenceOwnerState.RUNNING
            if self._abandoned or self._work is None:
                self._work = None
                return None
            work = self._work
            self._work = None
            return work

    def _abandon_locked(self, outcome: KwsSpeakerInferenceOutcome) -> None:
        if not self._abandoned:
            self._abandoned = True
            self._abandon_outcome = outcome
            if self._state is not KwsSpeakerInferenceOwnerState.RETAINED:
                self._state = KwsSpeakerInferenceOwnerState.ABANDONED
        self._pending_result = None
        self._work = None
        self._condition.notify_all()

    def _run_worker(self) -> None:
        work = None
        gate = None
        clips: tuple[np.ndarray, ...] = ()
        receipt = None
        outcome = KwsSpeakerInferenceOutcome.ERROR
        error_type = "MissingWork"
        try:
            work = self._take_work()
            if work is not None:
                gate = work.gate
                clips = work.clips
                lease = self._runtime_lease
                if lease is None:
                    raise RuntimeError("runtime lease disappeared")
                receipt = gate._try_similarity_batch_with_runtime_lease(
                    clips,
                    work.sample_rate,
                    runtime_permit=self._runtime_permit,
                    runtime_lease=lease,
                    enter_native_step=lambda: self.enter_native_step(self._ticket),
                )
                if receipt is None:
                    outcome = KwsSpeakerInferenceOutcome.BUSY
                    error_type = ""
                elif type(receipt) is SpeakerSimilarityBatchReceipt:
                    outcome = KwsSpeakerInferenceOutcome.COMPLETED
                    error_type = ""
                else:
                    raise TypeError("speaker batch receipt has an invalid type")
        except BaseException as exc:
            outcome = KwsSpeakerInferenceOutcome.ERROR
            error_type = type(exc).__name__
            receipt = None
        finally:
            # Drop every native-input reference before making the outcome
            # observable to the capture owner or releasing the process permit.
            work = None
            gate = None
            clips = ()
            result = None
            if outcome is KwsSpeakerInferenceOutcome.COMPLETED and receipt is not None:
                result = KwsSpeakerInferenceResult(outcome, receipt=receipt)
            elif outcome is KwsSpeakerInferenceOutcome.ERROR:
                result = KwsSpeakerInferenceResult(outcome, error_type=error_type)
            elif outcome is KwsSpeakerInferenceOutcome.BUSY:
                result = KwsSpeakerInferenceResult(outcome)
            receipt = None
            with self._condition:
                if not self._abandoned and time.monotonic() >= self._deadline:
                    self._abandon_locked(KwsSpeakerInferenceOutcome.TIMEOUT)
                if not self._abandoned:
                    self._pending_result = result
                    self._state = KwsSpeakerInferenceOwnerState.RESULT_READY
                result = None
                self._worker_returned = True
                self._condition.notify_all()

    def _abandoned_result_locked(self) -> KwsSpeakerInferenceResult:
        outcome = self._abandon_outcome or KwsSpeakerInferenceOutcome.STOPPED
        if outcome is KwsSpeakerInferenceOutcome.ERROR:
            return KwsSpeakerInferenceResult(
                outcome,
                error_type=self._error_type or "InferenceOwnerError",
            )
        return KwsSpeakerInferenceResult(outcome)


def current_kws_speaker_inference_owner(
    *,
    registry: Optional[KwsSpeakerInferenceRegistry] = None,
) -> Optional[KwsSpeakerInferenceOwner]:
    """Return the exact registered process owner for diagnostics/preflight."""

    selected = _PROCESS_OWNER_REGISTRY if registry is None else registry
    if not isinstance(selected, KwsSpeakerInferenceRegistry):
        raise TypeError("registry must be a KwsSpeakerInferenceRegistry")
    return selected.current()


def try_claim_kws_speaker_inference_owner(
    gate: SpeakerGate,
    clips: tuple[np.ndarray, ...],
    sample_rate: int,
    *,
    deadline: float,
    runtime_permit: Optional[RuntimeSpeakerInferencePermit] = None,
    registry: Optional[KwsSpeakerInferenceRegistry] = None,
    thread_factory: Optional[Callable[..., object]] = None,
) -> Optional[KwsSpeakerInferenceOwner]:
    """Claim one exact process owner and native permit without waiting."""

    permit = (
        runtime_speaker_inference_permit() if runtime_permit is None else runtime_permit
    )
    if not isinstance(permit, RuntimeSpeakerInferencePermit):
        raise TypeError("runtime_permit must be a RuntimeSpeakerInferencePermit")
    selected_registry = _PROCESS_OWNER_REGISTRY if registry is None else registry
    if not isinstance(selected_registry, KwsSpeakerInferenceRegistry):
        raise TypeError("registry must be a KwsSpeakerInferenceRegistry")

    current = selected_registry.current()
    if current is not None and not current.try_reap():
        return None
    reservation = selected_registry.try_reserve()
    if reservation is None:
        return None
    lease = None
    transferred = False
    try:
        lease = permit.try_acquire()
        if lease is None:
            return None
        candidate = KwsSpeakerInferenceOwner(
            gate,
            clips,
            sample_rate,
            deadline=deadline,
            runtime_permit=permit,
            registry=selected_registry,
            thread_factory=thread_factory,
        )
        candidate._install_runtime_lease(lease)
        if not selected_registry.publish(reservation, candidate):
            raise RuntimeError("KWS speaker-inference reservation was lost")
        transferred = True
        try:
            candidate._install_thread()
        except BaseException:
            candidate.abandon(KwsSpeakerInferenceOutcome.ERROR)
            raise
    except BaseException:
        if not transferred and lease is not None and permit.is_current(lease):
            permit.release(lease)
        raise
    finally:
        selected_registry.cancel_reservation(reservation)
    return candidate


def require_kws_speaker_inference_idle(
    *,
    registry: Optional[KwsSpeakerInferenceRegistry] = None,
) -> None:
    """Refuse rebuild/start while any exact may-start/running owner remains."""

    selected = _PROCESS_OWNER_REGISTRY if registry is None else registry
    if not isinstance(selected, KwsSpeakerInferenceRegistry):
        raise TypeError("registry must be a KwsSpeakerInferenceRegistry")
    if selected.admission_is_reserved_for_other():
        raise RuntimeError(
            "cannot rebuild or start while KWS speaker inference admission is reserved"
        )
    owner = selected.current()
    if owner is None:
        return
    if owner.try_reap():
        return
    raise RuntimeError(
        "cannot rebuild or start while KWS speaker inference may still be live"
    )


__all__ = (
    "KwsSpeakerInferenceOutcome",
    "KwsSpeakerInferenceOwner",
    "KwsSpeakerInferenceRegistry",
    "KwsSpeakerInferenceOwnerSnapshot",
    "KwsSpeakerInferenceOwnerState",
    "KwsSpeakerInferenceResult",
    "KwsSpeakerInferenceTicket",
    "current_kws_speaker_inference_owner",
    "guard_kws_speaker_inference_mutation",
    "require_kws_speaker_inference_idle",
    "try_claim_kws_speaker_inference_owner",
)
