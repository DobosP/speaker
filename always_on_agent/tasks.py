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

    def cancel(self) -> None:
        with self._control_lock:
            self.cancel_event.set()
            # A terminal transition that already won remains authoritative.
            # The Event is still set so epoch/emitter gates suppress stale audio.
            if not self._terminal_claimed:
                self.state = TaskState.CANCELLED

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
                self.cancel_event.set()
            self._terminal_claimed = True
            self.state = actual
            return actual

    def cancel_and_claim_terminal(self) -> bool:
        """Atomically cancel and reserve the cancellation lifecycle event."""
        with self._control_lock:
            self.cancel_event.set()
            if self._terminal_claimed:
                return False
            self._terminal_claimed = True
            self.state = TaskState.CANCELLED
            return True


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
    ):
        self._publish = publish
        self._capabilities = capabilities
        self._planner = TaskPlanner()
        # ``_threads`` is mutated from the bus thread (``start``) and from the
        # worker threads themselves (lifecycle reaping). Guard it with a lock
        # that is only ever held for the dict mutation -- never across a thread
        # join, engine I/O, or ``_publish`` -- so the supervisor can't deadlock.
        self._threads: dict[str, Thread] = {}
        self._threads_lock = Lock()
        self._stream_tts = stream_tts
        self._max_active_tasks = max(1, int(max_active_tasks))
        self._invocation_slots = BoundedSemaphore(self._max_active_tasks)

    def _make_emitter(self, task: "AgentTask"):
        """A callback the speaking capability uses to play each sentence as it
        is generated. Stops emitting once the task is cancelled (barge-in)."""

        def emit(sentence: str) -> None:
            if task.cancel_event.is_set() or not sentence:
                return
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
                    },
                )
            )

        return emit

    def create_task(self, decision: IntentDecision) -> AgentTask:
        plan = self._planner.plan(decision)
        return AgentTask(
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
            },
        )

    def start(self, task: AgentTask) -> None:
        thread = Thread(target=self._run_task, args=(task,), daemon=True)
        with self._threads_lock:
            # Opportunistically drop any threads that have already exited but
            # whose lifecycle event we never saw, so the dict can't creep.
            self._reap_dead_locked()
            self._threads[task.task_id] = thread
            # Start while the registry lock is held. A just-created Thread is
            # not alive yet; exposing it before start let active_count reap it
            # and undercount a provider that then began running untracked.
            try:
                thread.start()
            except BaseException:
                self._threads.pop(task.task_id, None)
                raise

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
        return self.active_count >= self._max_active_tasks

    def _reap_dead_locked(self) -> None:
        """Drop threads that are no longer alive. Caller must hold the lock.

        Only inspects ``Thread.is_alive`` -- never joins -- so this is safe to
        call from the bus thread without blocking on a worker.
        """
        dead = [tid for tid, thr in self._threads.items() if not thr.is_alive()]
        for tid in dead:
            self._threads.pop(tid, None)

    def _reap(self, task_id: str) -> None:
        """Remove a finished task's thread from the registry.

        Called from the worker thread on COMPLETED/CANCELLED/FAILED. It must
        not join (that would be a self-join / deadlock) -- the daemon thread is
        already on its way out; we just stop tracking it.
        """
        with self._threads_lock:
            self._threads.pop(task_id, None)

    def _run_task(self, task: AgentTask) -> None:
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
                },
                priority=40,
            )
        )

        try:
            self._run_plan(task)
        except _TaskCancelled:
            self._publish_cancelled(task)
        except Exception as exc:  # noqa: BLE001
            # Without this, a capability raising (e.g. Ollama unreachable) would
            # kill the daemon thread silently and the task would sit "active"
            # forever -- the app looks hung. Turn it into a visible failure.
            log.exception("task %s raised in capability %r", task.task_id, task.capability)
            self._publish_failed(task, f"{type(exc).__name__}: {exc}")

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
        context: dict[str, object] = {
            "task_id": task.task_id,
            "mode": task.mode.value,
            "metadata": task.metadata,
            "cancel_event": task.cancel_event,
            # Lets a capability atomically linearize a nested provider start
            # (notably cross-tier retry) against task cancellation.
            "claim_provider_start": task.claim_invocation_start,
            # Lets a long-running capability (e.g. the multi-step ReAct planner,
            # which runs under the short ASSISTANT budget) push its own wall-clock
            # deadline out so the supervisor's reap doesn't kill a turn that is
            # legitimately still working. Extends only -- never shortens, never
            # creates a deadline where the mode disabled it.
            "renew_deadline": lambda secs: self._renew_deadline(task, secs),
        }
        # Forward the IntentKind so downstream routers (HeuristicRouter,
        # SensitivityRouterLLM) can factor it into tier / cloud-chain
        # choice -- e.g. RESEARCH -> main tier, COMMAND -> private chain.
        if task.intent is not None:
            context["intent_kind"] = task.intent.value
        if extra_context:
            context.update(extra_context)
        # Forward the turn's speaker-ID trust to the capability's action chokepoint
        # (always_on_agent.origin). Fail-closed: a task without a stamped verdict is
        # treated as not-owner-verified / unknown origin, so a side-effecting
        # capability (command.stage) refuses unless the turn was owner-verified.
        # Stamped AFTER extra_context so a caller's extra_context can never override
        # the trust verdict (defense-in-depth: the trust seam stays authoritative).
        context["owner_verified"] = bool(task.metadata.get("owner_verified", False))
        context["origin"] = str(task.metadata.get("origin", "unknown"))
        return self._invoke_cancellable(
            task,
            lambda: self._capabilities.invoke(
                capability,
                task.input_text,
                context,
            ),
        )

    def _invoke_cancellable(
        self,
        task: AgentTask,
        invoke: Callable[[], CapabilityResult],
    ) -> CapabilityResult:
        """Run one synchronous provider behind a cancellable task coordinator.

        ``AgentTask.cancel_event`` cannot interrupt an arbitrary Python/native
        call that is blocked before returning a first token.  The registered
        task thread therefore coordinates a daemon provider thread instead of
        becoming that provider thread itself.  Barge-in can retire the
        coordinator immediately, while ``_invocation_slots`` remains owned by
        any abandoned provider until it truly exits.  This is a bulkhead: at
        most ``max_active_tasks`` uncooperative provider calls can survive a
        cancellation storm.
        """
        acquired = False
        while not acquired:
            if task.cancel_event.is_set():
                raise _TaskCancelled
            acquired = self._invocation_slots.acquire(timeout=self._CANCEL_POLL_SEC)

        # Cancellation wins the acquire race.  A coordinator cancelled while
        # waiting for capacity must never launch a stale provider invocation.
        if task.cancel_event.is_set():
            self._invocation_slots.release()
            raise _TaskCancelled

        done = Event()
        outcome: dict[str, object] = {}

        def run_provider() -> None:
            try:
                # Definitive start/cancel linearization.  The coordinator's
                # post-acquire check is only an early exit; cancellation can
                # still land before this new thread is scheduled.  In that
                # case no provider (especially no side effect) may be admitted late.
                if not task.claim_invocation_start():
                    outcome["error"] = _TaskCancelled()
                    return
                outcome["result"] = invoke()
            except BaseException as exc:  # preserve the provider's normal failure
                outcome["error"] = exc
            finally:
                self._invocation_slots.release()
                done.set()

        provider_thread = Thread(
            target=run_provider,
            name=f"speaker-capability-{task.task_id}",
            daemon=True,
        )
        try:
            provider_thread.start()
        except BaseException:
            # Ownership only transfers to ``run_provider`` after a successful
            # start; otherwise release synchronously to avoid losing capacity.
            self._invocation_slots.release()
            raise

        while True:
            if task.cancel_event.is_set():
                raise _TaskCancelled
            if done.wait(self._CANCEL_POLL_SEC):
                break

        # Give cancellation precedence over a provider result that completed in
        # the same scheduling window.  _run_plan performs a final guard too.
        if task.cancel_event.is_set():
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
            self._reap(task.task_id)
            return
        if terminal == TaskState.CANCELLED:
            self._emit_cancelled_claimed(task)
            return
        log.info(
            "task %s completed in %.2fs (%d chars)",
            task.task_id, time.time() - task.created_at, len(task.output_text or ""),
        )
        self._reap(task.task_id)
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
                    {"task_id": task.task_id, "epoch": task.speech_epoch},
                    priority=110,
                )
            )

    def _publish_cancelled(self, task: AgentTask) -> None:
        terminal = task.claim_terminal(TaskState.CANCELLED)
        if terminal is None:
            # The watchdog can reserve + publish the one cancellation event.
            # The coordinator still owns removal from the thread registry.
            self._reap(task.task_id)
            return
        self._emit_cancelled_claimed(task)

    def _emit_cancelled_claimed(self, task: AgentTask) -> None:
        """Publish cancellation after this caller reserved the terminal event."""
        log.info("task %s cancelled after %.2fs", task.task_id, time.time() - task.created_at)
        self._reap(task.task_id)
        self._publish(
            AgentEvent(
                EventKind.TASK_CANCELLED,
                {"task_id": task.task_id, "mode": task.mode.value},
                priority=20,
            )
        )

    def _publish_failed(self, task: AgentTask, error: str) -> None:
        terminal = task.claim_terminal(TaskState.FAILED)
        if terminal is None:
            self._reap(task.task_id)
            return
        if terminal == TaskState.CANCELLED:
            self._emit_cancelled_claimed(task)
            return
        log.error("task %s FAILED after %.2fs: %s", task.task_id,
                  time.time() - task.created_at, error)
        self._reap(task.task_id)
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
                },
                priority=25,
            )
        )
