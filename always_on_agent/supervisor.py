from __future__ import annotations

from dataclasses import dataclass, field
from threading import Timer

from .capabilities import CapabilityRegistry, create_default_capabilities
from .event_bus import EventBus
from .events import AgentEvent, EventKind, Mode
from .followups import FollowupConfig, FollowupState
from .memory import SessionMemory
from .models import IntentDecision, IntentKind, SpeechObservation
from .speech_analyzer import LiveSpeechAnalyzer
from .tasks import AgentTask, TaskRuntime


@dataclass
class SupervisorState:
    mode: Mode = Mode.PASSIVE
    last_partial: str = ""
    transcript_log: list[str] = field(default_factory=list)
    observations: list[SpeechObservation] = field(default_factory=list)
    decisions: list[IntentDecision] = field(default_factory=list)
    active_tasks: dict[str, AgentTask] = field(default_factory=dict)
    queued_tasks: list[AgentTask] = field(default_factory=list)
    pending_confirmations: dict[str, AgentTask] = field(default_factory=dict)
    spoken_outputs: list[str] = field(default_factory=list)
    event_log: list[AgentEvent] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


class AgentSupervisor:
    """High-level mode and task coordinator for always-on voice events."""

    def __init__(
        self,
        bus: EventBus | None = None,
        analyzer: LiveSpeechAnalyzer | None = None,
        capabilities: CapabilityRegistry | None = None,
        memory: SessionMemory | None = None,
        stream_tts: bool = False,
        followup_config: FollowupConfig | None = None,
    ):
        self.bus = bus or EventBus()
        self.state = SupervisorState()
        self.memory = memory or SessionMemory()
        self.capabilities = capabilities or create_default_capabilities(self.memory)
        self.analyzer = analyzer or LiveSpeechAnalyzer()
        self.tasks = TaskRuntime(self.bus.publish, self.capabilities, stream_tts=stream_tts)
        self.followups = followup_config or FollowupConfig()
        self._followup_state = FollowupState(
            markers=self.followups.markers, max_followups=self.followups.max_followups
        )
        self._followup_timer: Timer | None = None
        self.bus.subscribe(self.handle_event)

    def publish(self, event: AgentEvent) -> None:
        self.bus.publish(event)

    def handle_event(self, event: AgentEvent) -> None:
        self.state.event_log.append(event)

        if event.kind == EventKind.STT_PARTIAL:
            self._handle_speech(str(event.payload.get("text", "")), is_final=False)
            return
        if event.kind == EventKind.STT_FINAL:
            self._handle_speech(str(event.payload.get("text", "")), is_final=True)
            return
        if event.kind == EventKind.FOLLOWUP_TICK:
            self._emit_followup()
            return
        if event.kind == EventKind.CONTROL_STOP:
            self.cancel_all()
            return
        if event.kind == EventKind.CONTROL_MODE:
            self._set_mode(str(event.payload.get("mode", "")))
            return
        if event.kind == EventKind.CONTROL_CONFIRM:
            self._confirm_next()
            return
        if event.kind == EventKind.CONTROL_DENY:
            self._deny_all()
            return
        if event.kind == EventKind.TASK_COMPLETED:
            self._handle_task_completed(event)
            return
        if event.kind == EventKind.TASK_CANCELLED:
            self.state.active_tasks.pop(str(event.payload.get("task_id", "")), None)
            self._start_queued_tasks()
            return
        if event.kind == EventKind.TASK_FAILED:
            self.state.active_tasks.pop(str(event.payload.get("task_id", "")), None)
            self.state.failures.append(str(event.payload.get("error", "")))
            self._start_queued_tasks()

    def drain(self) -> int:
        return self.bus.drain()

    def shutdown(self) -> None:
        """Cancel background timers (called on runtime stop)."""
        self._cancel_followup()

    def cancel_all(self) -> None:
        for task in list(self.state.active_tasks.values()):
            task.cancel()
        self.state.queued_tasks.clear()
        self.state.pending_confirmations.clear()
        self._reset_followups()
        self.state.spoken_outputs.append("[cancelled]")

    # --- proactive follow-ups ------------------------------------------------
    def _schedule_followup(self) -> None:
        """Arm a timer that, after a silent delay, nudges the conversation."""
        if not self.followups.enabled or not self._followup_state.can_continue():
            return
        self._cancel_followup()
        timer = Timer(self.followups.delay_sec, self._tick_followup)
        timer.daemon = True
        self._followup_timer = timer
        timer.start()

    def _cancel_followup(self) -> None:
        timer = self._followup_timer
        if timer is not None:
            timer.cancel()
            self._followup_timer = None

    def _reset_followups(self) -> None:
        self._cancel_followup()
        self._followup_state.reset()

    def _tick_followup(self) -> None:
        """Timer-thread callback: hand control back to the bus thread."""
        self.publish(AgentEvent(EventKind.FOLLOWUP_TICK, priority=95))

    def _emit_followup(self) -> None:
        """Bus-thread handler: start a follow-up task if still warranted."""
        if not self.followups.enabled or not self._followup_state.can_continue():
            return
        if self.state.active_tasks or self.state.queued_tasks:
            return
        marker = self._followup_state.next_marker()
        decision = IntentDecision(
            kind=IntentKind.ASSISTANT,
            confidence=1.0,
            text=marker,
            reason="followup",
            mode=Mode.ASSISTANT,
            speak=True,
        )
        task = self.tasks.create_task(decision)
        task.metadata["followup"] = True
        self._start_task(task)

    def _handle_speech(self, text: str, *, is_final: bool) -> None:
        observation = self.analyzer.observe(text, is_final=is_final)
        self.state.observations.append(observation)
        if not observation.normalized:
            return
        # The user spoke: stop any pending proactive follow-up and restart the
        # silence cadence from zero.
        self._reset_followups()
        if is_final:
            self.state.transcript_log.append(text)
        else:
            self.state.last_partial = text

        decision = self.analyzer.decide(observation, self.state.mode)
        self.state.decisions.append(decision)
        self.publish(
            AgentEvent(
                EventKind.INTENT_DECISION,
                {
                    "kind": decision.kind.value,
                    "confidence": decision.confidence,
                    "reason": decision.reason,
                    "text": decision.text,
                },
                priority=55,
            )
        )
        self._execute_decision(decision)

    def _execute_decision(self, decision: IntentDecision) -> None:
        if decision.kind == IntentKind.IGNORE:
            return
        if decision.kind == IntentKind.STOP:
            self.publish(AgentEvent.stop(decision.reason))
            return
        if decision.kind == IntentKind.CONFIRM:
            self.publish(AgentEvent.confirm())
            return
        if decision.kind == IntentKind.DENY:
            self.publish(AgentEvent.deny())
            return
        if decision.kind == IntentKind.MODE_SWITCH and decision.target_mode is not None:
            self.publish(AgentEvent.mode(decision.target_mode))
            return
        if not decision.starts_task:
            return
        task = self.tasks.create_task(decision)
        if task.metadata.get("requires_confirmation"):
            self.state.pending_confirmations[task.task_id] = task
            self.state.spoken_outputs.append(f"Confirm command: {task.input_text}")
            return
        if self._should_queue(task):
            self.state.queued_tasks.append(task)
            self.state.spoken_outputs.append(f"Queued {task.mode.value}: {task.input_text}")
            return
        self._start_task(task)

    def _start_task(self, task: AgentTask) -> None:
        self.state.active_tasks[task.task_id] = task
        self.tasks.start(task)

    def _should_queue(self, task: AgentTask) -> bool:
        if task.mode != Mode.RESEARCH:
            return False
        active_research = sum(
            1 for active in self.state.active_tasks.values() if active.mode == Mode.RESEARCH
        )
        return active_research >= self.analyzer.policy.research_parallel_tasks

    def _start_queued_tasks(self) -> None:
        remaining: list[AgentTask] = []
        for task in self.state.queued_tasks:
            if self._should_queue(task):
                remaining.append(task)
                continue
            self._start_task(task)
        self.state.queued_tasks = remaining

    def _confirm_next(self) -> None:
        if not self.state.pending_confirmations:
            self.state.spoken_outputs.append("Nothing to confirm.")
            return
        task_id, task = next(iter(self.state.pending_confirmations.items()))
        self.state.pending_confirmations.pop(task_id, None)
        self._start_task(task)

    def _deny_all(self) -> None:
        if not self.state.pending_confirmations:
            self.state.spoken_outputs.append("Nothing to cancel.")
            return
        self.state.pending_confirmations.clear()
        self.state.spoken_outputs.append("Command cancelled.")

    def _handle_task_completed(self, event: AgentEvent) -> None:
        task_id = str(event.payload.get("task_id", ""))
        self.state.active_tasks.pop(task_id, None)
        self._start_queued_tasks()
        text = str(event.payload.get("text", "")).strip()
        speak = bool(event.payload.get("speak", True))
        data = event.payload.get("data", {})
        streamed = bool(isinstance(data, dict) and data.get("streamed"))
        is_followup = bool(event.payload.get("followup"))
        # Proactive follow-ups are conversational filler, not facts: keep them
        # out of long-term memory.
        if text and not is_followup:
            self.memory.add(text, tags=("assistant_output",))
        if text and speak:
            self.state.spoken_outputs.append(text)
            # When the capability streamed sentence-by-sentence, the audio has
            # already been spoken during the task; don't re-speak the whole text.
            if not streamed:
                self.publish(
                    AgentEvent(EventKind.TTS_REQUEST, {"task_id": task_id, "text": text})
                )
        # Arm (or continue) the silence cadence once the assistant has spoken.
        if speak and text:
            self._schedule_followup()

    def _set_mode(self, mode_value: str) -> None:
        try:
            self.state.mode = Mode(mode_value)
        except ValueError:
            return
        self._reset_followups()
        self.state.spoken_outputs.append(f"Mode: {self.state.mode.value}")
