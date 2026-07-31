from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from threading import BoundedSemaphore, Event, Lock, Thread
import time
from typing import Callable
import uuid

log = logging.getLogger("speaker.tasks")

from .capabilities import CapabilityRegistry, CapabilityResult
from .events import AgentEvent, EventKind, Mode
from .models import IntentDecision, IntentKind
from .planner import TaskPlan, TaskPlanner
from .session_actor import (
    AdmissionRejected,
    ProviderAdmission,
    SessionActor,
    TaskIdentity,
    ToolAdmission,
    TurnHandle,
    TurnIdentity,
)


class TaskState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_FOR_CONFIRMATION = "waiting_for_confirmation"
    SPEAKING = "speaking"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class AgentTask:
    mode: Mode
    input_text: str
    intent: IntentKind | None = None
    capability: str = ""
    plan: TaskPlan | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    # Bound by SessionActor before execution.  Direct test/adapter construction
    # may leave this unset temporarily; TaskRuntime binds it before admission,
    # and no lifecycle/provider/TTS work starts without a concrete identity.
    identity: TaskIdentity | None = None
    turn_handle: TurnHandle | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    state: TaskState = TaskState.QUEUED
    priority: int = 100
    created_at: float = field(default_factory=time.time)
    cancel_event: Event = field(default_factory=Event)
    output_text: str = ""
    # Speech epoch captured when the task is started (see AgentSupervisor).
    # Stamped on every streaming TTS_REQUEST this task emits so a later barge-in
    # (which advances the supervisor epoch) drops the now-stale sentences without
    # depending on the task still being "active" -- TASK_COMPLETED is dequeued
    # before the trailing TTS_REQUESTs (realtime-concurrency-1).
    speech_epoch: int = 0
    # Set True the moment this task has spoken its first streamed sentence. The
    # ADD-ON / continuation gate reads it (on the bus thread) to choose between
    # *merging* a follow-up into a not-yet-spoken turn (cancel + restart with the
    # combined prompt) and *continuing* after a turn that is already talking
    # (queue a context-carrying follow-up). Written once, from the worker thread,
    # in the streaming emitter; the bool write is atomic under the GIL and any
    # narrow read race is covered by the epoch gate (see _maybe_continue).
    started_speaking: bool = False
    # Set True when this task's ack_then_think latency acknowledgement has been
    # voiced. Deliberately SEPARATE from ``started_speaking``: the ack is generic
    # filler, not answer audio, so an add-on landing after the ack but before the
    # first real sentence must still take the MERGE branch of the continuation
    # gate (cancel + one combined prompt), not queue behind an answer the user
    # has not heard yet.
    ack_spoken: bool = False
    # Monotonic wall-clock deadline (time.monotonic) stamped when the task is
    # started; 0.0 means "no deadline". The supervisor's periodic reap cancels +
    # removes any active task past its deadline so a capability that blocks
    # uninterruptibly (a hung generate / network read) can never leave the
    # controller waiting on it forever. See AgentSupervisor.reap_overdue_tasks.
    deadline_at: float = 0.0
    # Serializes the two control-plane linearization points that must not race:
    # provider-start vs cancellation, and the single terminal lifecycle event.
    # Excluded from dataclass init/repr/equality; it is strictly runtime state.
    _control_lock: Lock = field(
        default_factory=Lock,
        init=False,
        repr=False,
        compare=False,
    )
    _terminal_claimed: bool = field(
        default=False,
        init=False,
        repr=False,
        compare=False,
    )
    _next_tool_call: int = field(
        default=0,
        init=False,
        repr=False,
        compare=False,
    )

    def cancel(self) -> None:
        with self._control_lock:
            if self.turn_handle is not None:
                self.turn_handle.cancel()
            self.cancel_event.set()
            # A terminal transition that already won remains authoritative.
            # The Event is still set so epoch/emitter gates suppress stale audio.
            if not self._terminal_claimed:
                self.state = TaskState.CANCELLED

    def attach_turn_handle(self, handle: TurnHandle) -> None:
        """Attach the one actor scope before this task can execute."""

        with self._control_lock:
            if self.turn_handle is not None and self.turn_handle is not handle:
                raise RuntimeError("task is already bound to another turn handle")
            if self.identity is not None and self.identity != handle.identity:
                raise RuntimeError("task identity disagrees with turn handle")
            was_cancelled = self.cancel_event.is_set()
            self.turn_handle = handle
            self.identity = handle.identity
            self.cancel_event = handle.cancel_event
            if was_cancelled:
                handle.cancel()

    def claim_invocation_start(self) -> bool:
        """Atomically linearize provider start before or after cancellation."""
        with self._control_lock:
            return not self.cancel_event.is_set() and not self._terminal_claimed

    def mark_running(self) -> bool:
        """Enter RUNNING unless cancellation already won the start race."""
        with self._control_lock:
            if self.cancel_event.is_set() or self._terminal_claimed:
                return False
            self.state = TaskState.RUNNING
            return True

    def claim_terminal(self, desired: TaskState) -> TaskState | None:
        """Reserve this task's one terminal lifecycle event.

        Cancellation wins when its Event was set before this claim.  Once a
        terminal state is claimed, later cancellation can still suppress audio
        but cannot rewrite the already-linearized lifecycle result.
        """
        with self._control_lock:
            if self._terminal_claimed:
                return None
            actual = (
                TaskState.CANCELLED
                if self.cancel_event.is_set() or desired == TaskState.CANCELLED
                else desired
            )
            if actual == TaskState.CANCELLED:
                if self.turn_handle is not None:
                    self.turn_handle.cancel()
                else:
                    self.cancel_event.set()
            self._terminal_claimed = True
            self.state = actual
            return actual

    def cancel_and_claim_terminal(
        self,
        fence: Callable[[], object] | None = None,
    ) -> bool:
        """Atomically fence dependent work, cancel, and claim terminality.

        The callback must be a short in-memory transition. It runs only when
        cancellation wins and while ``_control_lock`` is held, establishing the
        task→session-actor lock order used by timeout reaping.
        """

        with self._control_lock:
            if self._terminal_claimed:
                return False
            if fence is not None:
                fence()
            if self.turn_handle is not None:
                self.turn_handle.cancel()
            else:
                self.cancel_event.set()
            self._terminal_claimed = True
            self.state = TaskState.CANCELLED
            return True

    def allocate_tool_identity(self, capability: str) -> tuple[str, str]:
        """Allocate generic call/idempotency IDs for one mutating plan step."""

        identity = self.identity
        if identity is None:
            raise RuntimeError("task has no session identity")
        with self._control_lock:
            self._next_tool_call += 1
            ordinal = self._next_tool_call
        explicit_call = self.metadata.get("tool_call_id")
        explicit_key = self.metadata.get("idempotency_key")
        tool_call_id = (
            str(explicit_call)
            if ordinal == 1 and explicit_call
            else f"tool-{self.task_id}-{identity.binding_id}-{ordinal}"
        )
        idempotency_key = (
            str(explicit_key)
            if ordinal == 1 and explicit_key
            else (
                f"{identity.session_id}:{identity.turn_id}:"
                f"{identity.revision}:{capability}:{ordinal}"
            )
        )
        return tool_call_id, idempotency_key


# Hard ceiling on concurrently running task threads. RESEARCH already has its
# own per-mode cap (``ModePolicy.research_parallel_tasks``); this is the global
# backstop across *all* modes so a burst of assistant/search turns can't spawn
# unbounded daemon threads. Overflow is reported via ``at_capacity()`` so the
# supervisor queues it the same way a capped RESEARCH task is queued.
DEFAULT_MAX_ACTIVE_TASKS = 6


class _TaskCancelled(Exception):
    """Control-flow signal used when a task is cancelled during an invocation."""


class TaskRuntime:
    """Runs cancellable tasks and emits lifecycle events."""

    # Arbitrary synchronous capability providers cannot be force-killed safely
    # in Python.  Run them behind a cancellable coordinator and poll at a short
    # interval so barge-in retires the *task* promptly even if the provider is
    # still blocked before its first token.  The semaphore is deliberately held
    # by the provider thread until it really exits: abandoned native/network
    # calls therefore stay bounded instead of accumulating without limit.
    _CANCEL_POLL_SEC = 0.01

    def __init__(
        self,
        publish,
        capabilities: CapabilityRegistry,
        *,
        stream_tts: bool = False,
        max_active_tasks: int = DEFAULT_MAX_ACTIVE_TASKS,
        session_actor: SessionActor | None = None,
    ):
        self._publish = publish
        self._capabilities = capabilities
        self._planner = TaskPlanner()
        # ``_threads`` is mutated from the bus thread (``start``) and from the
        # worker threads themselves (lifecycle reaping). Guard it with a lock
        # that is only ever held for the dict mutation -- never across a thread
        # join, engine I/O, or ``_publish`` -- so the supervisor can't deadlock.
        self._threads: dict[str, Thread] = {}
        self._thread_identities: dict[str, TaskIdentity] = {}
        self._threads_lock = Lock()
        self._stream_tts = stream_tts
        self._max_active_tasks = max(1, int(max_active_tasks))
        self._session = session_actor or SessionActor(
            max_active_tasks=self._max_active_tasks,
            max_active_tools=self._max_active_tasks,
        )
        self._invocation_slots = BoundedSemaphore(self._max_active_tasks)

    @property
    def session_actor(self) -> SessionActor:
        return self._session

    def _make_emitter(self, task: "AgentTask"):
        """A callback the speaking capability uses to play each sentence as it
        is generated. Stops emitting once the task is cancelled (barge-in)."""

        def emit(sentence: str) -> None:
            if task.cancel_event.is_set() or not sentence:
                return
            identity = self._require_identity(task)
            # Runtime admission, not queue publication, marks started_speaking:
            # a continuation can still cancel this unheard sentence while the
            # TTS event waits behind higher-priority bus work.
            self._publish(
                AgentEvent(
                    EventKind.TTS_REQUEST,
                    {
                        "task_id": task.task_id,
                        "text": sentence,
                        "epoch": task.speech_epoch,
                        "followup": bool(task.metadata.get("followup")),
                        "input_generation": task.metadata.get("input_generation"),
                        "metrics_turn_token": task.metadata.get(
                            "metrics_turn_token"
                        ),
                        # Explicit reply-stream identity.  Completion TTS uses a
                        # single whole reply and omits this flag; the runtime
                        # uses it to scope voice continuity without guessing
                        # from task lifecycle races.
                        "streaming": True,
                        **identity.to_payload(),
                    },
                )
            )

        return emit

    def create_task(
        self,
        decision: IntentDecision,
        *,
        turn_identity: TurnIdentity | None = None,
    ) -> AgentTask:
        plan = self._planner.plan(decision)
        controller_metadata = {
            key: decision.metadata[key]
            for key in ("device_tool", "prepared_device_command")
            if key in decision.metadata
        }
        task = AgentTask(
            mode=plan.mode,
            input_text=decision.text,
            intent=decision.kind,
            capability=plan.steps[-1].capability if plan.steps else "",
            plan=plan,
            priority=plan.priority,
            metadata={
                "confidence": decision.confidence,
                "reason": decision.reason,
                "requires_confirmation": plan.requires_confirmation,
                "speak": plan.speak_final,
                "plan": [
                    {"name": step.name, "capability": step.capability}
                    for step in plan.steps
                ],
                "tags": plan.tags,
                **controller_metadata,
            },
        )
        turn = turn_identity or self._session.open_turn(revision=0)
        task.attach_turn_handle(self._session.bind_turn(task.task_id, turn))
        return task

    def admit_task(self, task: AgentTask) -> bool:
        """Reserve actor task capacity before any acknowledgement is emitted."""

        identity = self._ensure_identity(task)
        with self._threads_lock:
            self._reap_dead_locked()
            if task.task_id in self._threads:
                return False
        if self._session.task_is_admitted(task.task_id, identity):
            return True
        return self._session.admit_task(task.task_id, identity)

    def retire_task_admission(self, task: AgentTask) -> None:
        identity = self._require_identity(task)
        self._session.finish_task(task.task_id, identity)

    def release_task_binding(
        self,
        task_id: str,
        identity: TaskIdentity,
    ) -> bool:
        """Declare controller ownership terminal for one bound task."""

        return self._session.release_task_binding(task_id, identity)

    def start(self, task: AgentTask) -> bool:
        identity = self._ensure_identity(task)
        try:
            thread = Thread(target=self._run_task, args=(task,), daemon=True)
        except BaseException:
            self._session.finish_task(task.task_id, identity)
            task.cancel()
            raise
        with self._threads_lock:
            # Opportunistically drop any threads that have already exited but
            # whose lifecycle event we never saw, so the dict can't creep.
            self._reap_dead_locked()
            if task.task_id in self._threads:
                return False
            if (
                not self._session.task_is_admitted(task.task_id, identity)
                and not self._session.admit_task(task.task_id, identity)
            ):
                return False
            self._threads[task.task_id] = thread
            self._thread_identities[task.task_id] = identity
            # Start while the registry lock is held. A just-created Thread is
            # not alive yet; exposing it before start let active_count reap it
            # and undercount a provider that then began running untracked.
            try:
                thread.start()
            except BaseException:
                self._threads.pop(task.task_id, None)
                self._thread_identities.pop(task.task_id, None)
                self._session.finish_task(task.task_id, identity)
                task.cancel()
                raise
        return True

    @property
    def active_count(self) -> int:
        """Number of task threads currently registered as running."""
        with self._threads_lock:
            self._reap_dead_locked()
            return len(self._threads)

    @property
    def max_active_tasks(self) -> int:
        return self._max_active_tasks

    def at_capacity(self) -> bool:
        """True when starting another task would exceed the global cap.

        The supervisor consults this to overflow excess turns into
        ``queued_tasks`` -- the same path a capped RESEARCH task takes.
        """
        return bool(
            self.active_count >= self._max_active_tasks
            or self._session.active_task_count >= self._max_active_tasks
        )

    def cancel_tool_call(
        self,
        tool_call_id: str,
        identity: TaskIdentity,
    ) -> bool:
        """Explicit tool cancellation; unrelated speech cancellation is separate."""

        return self._session.cancel_tool(tool_call_id, identity)

    def fence_task_tool_calls(
        self,
        task_id: str,
        identity: TaskIdentity,
    ) -> int:
        """Close a task to future tools and cancel its unstarted admissions."""

        return self._session.fence_task_tools(task_id, identity)

    def _ensure_identity(self, task: AgentTask) -> TaskIdentity:
        if task.turn_handle is not None:
            if task.identity != task.turn_handle.identity:
                raise RuntimeError("task identity disagrees with turn handle")
            if not self._session.binding_matches(
                task.task_id,
                task.turn_handle.identity,
            ):
                raise RuntimeError("task turn handle is no longer actor-owned")
            return task.turn_handle.identity
        if task.identity is None:
            turn = self._session.open_turn(revision=0)
            task.attach_turn_handle(self._session.bind_turn(task.task_id, turn))
        else:
            handle = self._session.handle_for_task(task.task_id, task.identity)
            if handle is None:
                raise RuntimeError("task identity has no open actor turn handle")
            task.attach_turn_handle(handle)
        assert task.identity is not None
        return task.identity

    def _ensure_turn_handle(self, task: AgentTask) -> TurnHandle:
        self._ensure_identity(task)
        handle = task.turn_handle
        if handle is None:
            raise RuntimeError("task has no actor turn handle")
        return handle

    @staticmethod
    def _require_identity(task: AgentTask) -> TaskIdentity:
        identity = task.identity
        if identity is None:
            raise RuntimeError("task has no admitted session identity")
        return identity

    def _reap_dead_locked(self) -> None:
        """Drop threads that are no longer alive. Caller must hold the lock.

        Only inspects ``Thread.is_alive`` -- never joins -- so this is safe to
        call from the bus thread without blocking on a worker.
        """
        dead = [tid for tid, thr in self._threads.items() if not thr.is_alive()]
        for tid in dead:
            self._threads.pop(tid, None)
            identity = self._thread_identities.pop(tid, None)
            if identity is not None:
                self._session.finish_task(tid, identity)

    def _reap(self, task: AgentTask) -> None:
        """Remove a finished task's thread from the registry.

        Called from the worker thread on COMPLETED/CANCELLED/FAILED. It must
        not join (that would be a self-join / deadlock) -- the daemon thread is
        already on its way out; we just stop tracking it.
        """
        identity = self._require_identity(task)
        owned = False
        with self._threads_lock:
            if self._thread_identities.get(task.task_id) == identity:
                self._threads.pop(task.task_id, None)
                self._thread_identities.pop(task.task_id, None)
                owned = True
        if owned:
            self._session.finish_task(task.task_id, identity)

    def _run_task(self, task: AgentTask) -> None:
        try:
            identity = self._require_identity(task)
            if not task.mark_running():
                self._publish_cancelled(task)
                return
            log.info(
                "task %s started: mode=%s capability=%s input=%r",
                task.task_id, task.mode.value, task.capability, task.input_text,
            )
            self._publish(
                AgentEvent(
                    EventKind.TASK_STARTED,
                    {
                        "task_id": task.task_id,
                        "mode": task.mode.value,
                        "capability": task.capability,
                        **identity.to_payload(),
                    },
                    priority=40,
                )
            )

            try:
                self._run_plan(task)
            except _TaskCancelled:
                self._publish_cancelled(task)
            except Exception as exc:  # noqa: BLE001
                # Without this, a capability raising (e.g. Ollama unreachable)
                # would kill the daemon thread silently and strand the turn.
                log.exception(
                    "task %s raised in capability %r",
                    task.task_id,
                    task.capability,
                )
                self._publish_failed(task, f"{type(exc).__name__}: {exc}")
        finally:
            # Coordinator ownership ends only after terminal publication and
            # the worker epilogue. A cancelled provider may remain registered
            # independently until its own thread actually exits.
            self._reap(task)

    def _run_plan(self, task: AgentTask) -> None:
        plan = task.plan
        if plan is None:
            self._publish_failed(task, "task has no plan")
            return
        step_results: list[dict[str, object]] = []
        final_result: CapabilityResult | None = None
        for step in plan.steps:
            if task.cancel_event.is_set():
                self._publish_cancelled(task)
                return
            self._publish(
                AgentEvent(
                    EventKind.TASK_PROGRESS,
                    {
                        "task_id": task.task_id,
                        "step": step.name,
                        "capability": step.capability,
                        **self._require_identity(task).to_payload(),
                    },
                    priority=70,
                )
            )
            time.sleep(0.01)
            extra: dict[str, object] = {"previous_steps": step_results}
            if (
                (self._stream_tts or task.metadata.get("stream_tts"))
                and step.speak_result
                and task.metadata.get("speak", True)
            ):
                extra["emit_speech"] = self._make_emitter(task)
            result = self._invoke(task, step.capability, extra)
            if not result.ok:
                self._publish_failed(task, result.error)
                return
            step_results.append(
                {
                    "step": step.name,
                    "capability": step.capability,
                    "text": result.text,
                    "data": result.data,
                    "citations": result.citations,
                }
            )
            if step.speak_result or final_result is None:
                final_result = result
        if final_result is None:
            self._publish_failed(task, "plan produced no result")
            return
        if task.cancel_event.is_set():
            # Cancelled while the final step was running (e.g. user barged in
            # during LLM generation): drop the now-stale result instead of
            # speaking it.
            self._publish_cancelled(task)
            return
        task.output_text = final_result.text
        task.metadata["step_results"] = step_results
        task.metadata["result_data"] = final_result.data
        task.metadata["citations"] = final_result.citations
        self._publish_completed(task)

    @staticmethod
    def _renew_deadline(task: AgentTask, secs: float) -> None:
        """Extend a running task's reap deadline (used by long capabilities).

        Only pushes an EXISTING deadline further out; if the mode disabled the
        deadline (``deadline_at == 0``) it stays disabled. A plain float write,
        atomic under the GIL, read by the watchdog-thread reap."""
        if task.deadline_at:
            try:
                task.deadline_at = max(task.deadline_at, time.monotonic() + float(secs))
            except (TypeError, ValueError):
                pass

    def _invoke(
        self,
        task: AgentTask,
        capability: str,
        extra_context: dict[str, object] | None = None,
    ) -> CapabilityResult:
        identity = self._require_identity(task)
        spec = self._capabilities.spec(capability)
        tool_admission: ToolAdmission | None = None
        if spec is not None and spec.side_effecting:
            tool_call_id, idempotency_key = task.allocate_tool_identity(capability)
            try:
                tool_admission = self._session.admit_tool(
                    task_id=task.task_id,
                    identity=identity,
                    tool_call_id=tool_call_id,
                    idempotency_key=idempotency_key,
                )
            except AdmissionRejected as exc:
                return CapabilityResult(False, "", error=str(exc))
            operation_cancel = tool_admission.cancel_event

            def claim_start() -> bool:
                return self._session.claim_tool_start(tool_call_id, identity)
        else:
            operation_cancel = task.cancel_event

            def claim_start() -> bool:
                return bool(
                    task.claim_invocation_start()
                    and self._session.task_effects_allowed(
                        task.task_id,
                        identity,
                    )
                )
        context: dict[str, object] = {
            "task_id": task.task_id,
            "mode": task.mode.value,
            "metadata": task.metadata,
            "cancel_event": operation_cancel,
            # Lets a capability atomically linearize a nested provider start
            # (notably cross-tier retry) against task cancellation.
            "claim_provider_start": claim_start,
            # Lets a long-running capability (e.g. the multi-step ReAct planner,
            # which runs under the short ASSISTANT budget) push its own wall-clock
            # deadline out so the supervisor's reap doesn't kill a turn that is
            # legitimately still working. Extends only -- never shortens, never
            # creates a deadline where the mode disabled it.
            "renew_deadline": lambda secs: self._renew_deadline(task, secs),
            **identity.to_payload(),
        }
        if tool_admission is not None:
            context["tool_call_id"] = tool_admission.tool_call_id
            context["idempotency_key"] = tool_admission.idempotency_key
        # Forward the IntentKind so downstream routers (HeuristicRouter,
        # SensitivityRouterLLM) can factor it into tier / cloud-chain
        # choice -- e.g. RESEARCH -> main tier, COMMAND -> private chain.
        if task.intent is not None:
            context["intent_kind"] = task.intent.value
        if extra_context:
            context.update(extra_context)
        # Causal identity and cancellation controls are controller-owned.  An
        # adapter's extra context may add provider hints, but cannot substitute
        # a different turn, generation, or tool call.
        context.update(identity.to_payload())
        context["cancel_event"] = operation_cancel
        context["claim_provider_start"] = claim_start
        if tool_admission is not None:
            context["tool_call_id"] = tool_admission.tool_call_id
            context["idempotency_key"] = tool_admission.idempotency_key
        # Forward the turn's provenance to the capability's common authority
        # boundary. Fail-closed: a task without stamped direct/owner/confirmation
        # verdicts receives none of them. Low-risk typed tools may require direct
        # live speech plus confirmation; sensitive actions can additionally require
        # an owner-verified speaker.
        # Stamped AFTER extra_context so a caller's extra_context can never override
        # the trust verdict (defense-in-depth: the trust seam stays authoritative).
        context["owner_verified"] = task.metadata.get("owner_verified") is True
        context["origin"] = str(task.metadata.get("origin", "unknown"))
        context["direct_user_instruction"] = (
            task.metadata.get("direct_user_instruction") is True
        )
        context["confirmed"] = task.metadata.get("confirmed") is True
        return self._invoke_cancellable(
            task,
            lambda: self._capabilities.invoke(
                capability,
                task.input_text,
                context,
            ),
            cancel_event=operation_cancel,
            claim_provider_start=claim_start,
            tool_admission=tool_admission,
        )

    def _invoke_cancellable(
        self,
        task: AgentTask,
        invoke: Callable[[], CapabilityResult],
        *,
        cancel_event: Event | None = None,
        claim_provider_start: Callable[[], bool] | None = None,
        tool_admission: ToolAdmission | None = None,
    ) -> CapabilityResult:
        """Run one synchronous provider behind a cancellable task coordinator.

        A read/generation call uses ``AgentTask.cancel_event``. A mutating call
        uses its actor-owned tool Event. Speech interruption cancels it until
        provider start is claimed; after that, only explicit tool cancellation
        or terminal shutdown sets the Event. Neither Event can interrupt
        arbitrary Python/native work blocked before returning. The registered
        task thread therefore coordinates a daemon provider thread instead of
        becoming that provider thread itself. ``_invocation_slots`` remains
        owned by any abandoned provider until it truly exits, bounding
        uncooperative calls.
        """
        handle = self._ensure_turn_handle(task)
        identity = handle.identity
        operation_cancel = cancel_event or task.cancel_event
        claim_start = claim_provider_start or task.claim_invocation_start
        acquired = False
        while not acquired:
            if operation_cancel.is_set():
                if tool_admission is not None:
                    self._session.finish_tool(
                        tool_admission.tool_call_id,
                        tool_admission.identity,
                    )
                raise _TaskCancelled
            acquired = self._invocation_slots.acquire(timeout=self._CANCEL_POLL_SEC)

        # Cancellation wins the acquire race.  A coordinator cancelled while
        # waiting for capacity must never launch a stale provider invocation.
        if operation_cancel.is_set():
            self._invocation_slots.release()
            if tool_admission is not None:
                self._session.finish_tool(
                    tool_admission.tool_call_id,
                    tool_admission.identity,
                )
            raise _TaskCancelled

        provider_admission: ProviderAdmission | None = None
        while provider_admission is None:
            if (
                operation_cancel.is_set()
                or handle.cancelled
                or not self._session.task_effects_allowed(
                    task.task_id,
                    identity,
                )
            ):
                self._invocation_slots.release()
                if tool_admission is not None:
                    self._session.finish_tool(
                        tool_admission.tool_call_id,
                        tool_admission.identity,
                    )
                raise _TaskCancelled
            provider_admission = self._session.admit_provider(
                task_id=task.task_id,
                identity=identity,
            )
            if provider_admission is None:
                # The previous provider releases its semaphore slot before its
                # actor lease so drain never claims quiescence while capacity
                # is still owned. Bridge that tiny handoff without converting
                # it into a spurious task failure.
                operation_cancel.wait(self._CANCEL_POLL_SEC)

        done = Event()
        outcome: dict[str, object] = {}

        def run_provider() -> None:
            try:
                # Definitive start/cancel linearization.  The coordinator's
                # post-acquire check is only an early exit; cancellation can
                # still land before this new thread is scheduled.  In that
                # case no provider (especially no side effect) may be admitted late.
                if not claim_start() or (
                    tool_admission is None
                    and not self._session.provider_start_allowed(
                        provider_admission
                    )
                ):
                    outcome["error"] = _TaskCancelled()
                    return
                outcome["result"] = invoke()
            except BaseException as exc:  # preserve the provider's normal failure
                outcome["error"] = exc
            finally:
                if tool_admission is not None:
                    self._session.finish_tool(
                        tool_admission.tool_call_id,
                        tool_admission.identity,
                    )
                self._invocation_slots.release()
                self._session.finish_provider(
                    provider_admission.provider_admission_id
                )
                done.set()

        try:
            provider_thread = Thread(
                target=run_provider,
                name=f"speaker-capability-{task.task_id}",
                daemon=True,
            )
            provider_thread.start()
        except BaseException:
            # Ownership only transfers to ``run_provider`` after a successful
            # start; otherwise release synchronously to avoid losing capacity.
            self._invocation_slots.release()
            self._session.finish_provider(
                provider_admission.provider_admission_id
            )
            if tool_admission is not None:
                self._session.finish_tool(
                    tool_admission.tool_call_id,
                    tool_admission.identity,
                )
            raise

        while True:
            if operation_cancel.is_set():
                raise _TaskCancelled
            if done.wait(self._CANCEL_POLL_SEC):
                break

        # Give cancellation precedence over a provider result that completed in
        # the same scheduling window.  _run_plan performs a final guard too.
        if operation_cancel.is_set():
            raise _TaskCancelled
        error = outcome.get("error")
        if isinstance(error, BaseException):
            if isinstance(error, Exception):
                raise error
            raise RuntimeError(f"capability provider aborted: {type(error).__name__}")
        result = outcome.get("result")
        if not isinstance(result, CapabilityResult):
            raise TypeError("capability provider returned no CapabilityResult")
        return result

    def _publish_completed(self, task: AgentTask) -> None:
        terminal = task.claim_terminal(TaskState.COMPLETED)
        if terminal is None:
            return
        if terminal == TaskState.CANCELLED:
            self._emit_cancelled_claimed(task)
            return
        log.info(
            "task %s completed in %.2fs (%d chars)",
            task.task_id, time.time() - task.created_at, len(task.output_text or ""),
        )
        result_data = task.metadata.get("result_data", {})
        streamed = bool(
            isinstance(result_data, dict) and result_data.get("streamed")
        )
        self._publish(
            AgentEvent(
                EventKind.TASK_COMPLETED,
                {
                    "task_id": task.task_id,
                    "mode": task.mode.value,
                    "capability": task.capability,
                    "text": task.output_text,
                    "speak": bool(task.metadata.get("speak", True)),
                    "followup": bool(task.metadata.get("followup")),
                    "data": result_data,
                    "citations": task.metadata.get("citations", ()),
                    # Carry the epoch stamped when this task started so the
                    # supervisor can drop a completion whose turn was superseded
                    # (e.g. a continuation merge bumped the epoch after the task
                    # finished but before its TASK_COMPLETED was dequeued).
                    "epoch": task.speech_epoch,
                    "input_generation": task.metadata.get("input_generation"),
                    "metrics_turn_token": task.metadata.get(
                        "metrics_turn_token"
                    ),
                    **self._require_identity(task).to_payload(),
                },
                priority=60,
            )
        )
        if streamed:
            # All sentence TTS events were published before completion.  This
            # lower-priority marker drains after them and releases the completed
            # stream's queued-audio ownership in the supervisor.
            self._publish(
                AgentEvent(
                    EventKind.TTS_STREAM_END,
                    {
                        "task_id": task.task_id,
                        "epoch": task.speech_epoch,
                        **self._require_identity(task).to_payload(),
                    },
                    priority=110,
                )
            )

    def _publish_cancelled(self, task: AgentTask) -> None:
        terminal = task.claim_terminal(TaskState.CANCELLED)
        if terminal is None:
            # The watchdog can reserve + publish the one cancellation event.
            # The coordinator's outer finally still owns registry removal.
            return
        self._emit_cancelled_claimed(task)

    def _emit_cancelled_claimed(self, task: AgentTask) -> None:
        """Publish cancellation after this caller reserved the terminal event."""
        log.info("task %s cancelled after %.2fs", task.task_id, time.time() - task.created_at)
        self._publish(
            AgentEvent(
                EventKind.TASK_CANCELLED,
                {
                    "task_id": task.task_id,
                    "mode": task.mode.value,
                    **self._require_identity(task).to_payload(),
                },
                priority=20,
            )
        )

    def _publish_failed(self, task: AgentTask, error: str) -> None:
        terminal = task.claim_terminal(TaskState.FAILED)
        if terminal is None:
            return
        if terminal == TaskState.CANCELLED:
            self._emit_cancelled_claimed(task)
            return
        log.error("task %s FAILED after %.2fs: %s", task.task_id,
                  time.time() - task.created_at, error)
        self._publish(
            AgentEvent(
                EventKind.TASK_FAILED,
                {
                    "task_id": task.task_id,
                    "mode": task.mode.value,
                    "error": error,
                    # Carry the same speak/followup/epoch fields as
                    # _publish_completed so the supervisor can gate + epoch-stamp
                    # the spoken failure apology (sr-2).
                    "speak": bool(task.metadata.get("speak", True)),
                    "followup": bool(task.metadata.get("followup")),
                    "epoch": task.speech_epoch,
                    "input_epoch": task.metadata.get("input_epoch"),
                    "input_generation": task.metadata.get("input_generation"),
                    **self._require_identity(task).to_payload(),
                },
                priority=25,
            )
        )
