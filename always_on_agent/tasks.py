from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from threading import Event, Thread
import time
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

    def cancel(self) -> None:
        self.cancel_event.set()
        self.state = TaskState.CANCELLED


class TaskRuntime:
    """Runs cancellable tasks and emits lifecycle events."""

    def __init__(self, publish, capabilities: CapabilityRegistry, *, stream_tts: bool = False):
        self._publish = publish
        self._capabilities = capabilities
        self._planner = TaskPlanner()
        self._threads: dict[str, Thread] = {}
        self._stream_tts = stream_tts

    def _make_emitter(self, task: "AgentTask"):
        """A callback the speaking capability uses to play each sentence as it
        is generated. Stops emitting once the task is cancelled (barge-in)."""

        def emit(sentence: str) -> None:
            if task.cancel_event.is_set() or not sentence:
                return
            self._publish(
                AgentEvent(EventKind.TTS_REQUEST, {"task_id": task.task_id, "text": sentence})
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
        self._threads[task.task_id] = thread
        thread.start()

    def _run_task(self, task: AgentTask) -> None:
        log.info(
            "task %s started: mode=%s capability=%s input=%r",
            task.task_id, task.mode.value, task.capability, task.input_text,
        )
        task.state = TaskState.RUNNING
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
                self._stream_tts
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
        task.state = TaskState.COMPLETED
        self._publish_completed(task)

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
        }
        # Forward the IntentKind so downstream routers (HeuristicRouter,
        # SensitivityRouterLLM) can factor it into tier / cloud-chain
        # choice -- e.g. RESEARCH -> main tier, COMMAND -> private chain.
        if task.intent is not None:
            context["intent_kind"] = task.intent.value
        if extra_context:
            context.update(extra_context)
        return self._capabilities.invoke(
            capability,
            task.input_text,
            context,
        )

    def _publish_completed(self, task: AgentTask) -> None:
        log.info(
            "task %s completed in %.2fs (%d chars)",
            task.task_id, time.time() - task.created_at, len(task.output_text or ""),
        )
        self._publish(
            AgentEvent(
                EventKind.TASK_COMPLETED,
                {
                    "task_id": task.task_id,
                    "mode": task.mode.value,
                    "text": task.output_text,
                    "speak": bool(task.metadata.get("speak", True)),
                    "followup": bool(task.metadata.get("followup")),
                    "data": task.metadata.get("result_data", {}),
                    "citations": task.metadata.get("citations", ()),
                },
                priority=60,
            )
        )

    def _publish_cancelled(self, task: AgentTask) -> None:
        log.info("task %s cancelled after %.2fs", task.task_id, time.time() - task.created_at)
        task.state = TaskState.CANCELLED
        self._publish(
            AgentEvent(
                EventKind.TASK_CANCELLED,
                {"task_id": task.task_id, "mode": task.mode.value},
                priority=20,
            )
        )

    def _publish_failed(self, task: AgentTask, error: str) -> None:
        log.error("task %s FAILED after %.2fs: %s", task.task_id,
                  time.time() - task.created_at, error)
        task.state = TaskState.FAILED
        self._publish(
            AgentEvent(
                EventKind.TASK_FAILED,
                {
                    "task_id": task.task_id,
                    "mode": task.mode.value,
                    "error": error,
                },
                priority=25,
            )
        )
