from __future__ import annotations

import time
from dataclasses import dataclass

from .diagnostics import summarize
from .events import AgentEvent, Mode
from .supervisor import AgentSupervisor


@dataclass(frozen=True)
class RuntimeSnapshot:
    mode: str
    last_partial: str
    transcripts: tuple[str, ...]
    outputs: tuple[str, ...]
    active_tasks: tuple[str, ...]
    queued_tasks: tuple[str, ...]
    pending_confirmations: tuple[str, ...]
    failures: tuple[str, ...]
    event_count: int


class AlwaysOnAgentRuntime:
    """
    Public facade for feeding live STT events into the agent layer.

    Existing audio code should use this class instead of reaching into
    `AgentSupervisor` directly.
    """

    def __init__(self, supervisor: AgentSupervisor | None = None):
        self.supervisor = supervisor or AgentSupervisor()

    @property
    def mode(self) -> Mode:
        return self.supervisor.state.mode

    def ingest_partial(self, text: str) -> None:
        self.supervisor.publish(AgentEvent.partial(text))
        self.supervisor.drain()

    def ingest_final(self, text: str) -> None:
        self.supervisor.publish(AgentEvent.final(text))
        self.supervisor.drain()

    def stop(self, reason: str = "external") -> None:
        self.supervisor.publish(AgentEvent.stop(reason))
        self.supervisor.drain()

    def wait_idle(self, timeout: float = 2.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.supervisor.drain()
            if (
                not self.supervisor.state.active_tasks
                and not self.supervisor.state.queued_tasks
            ):
                self.supervisor.drain()
                return True
            time.sleep(0.01)
        self.supervisor.drain()
        return (
            not self.supervisor.state.active_tasks
            and not self.supervisor.state.queued_tasks
        )

    def snapshot(self) -> RuntimeSnapshot:
        state = self.supervisor.state
        return RuntimeSnapshot(
            mode=state.mode.value,
            last_partial=state.last_partial,
            transcripts=tuple(state.transcript_log),
            outputs=tuple(state.spoken_outputs),
            active_tasks=tuple(sorted(state.active_tasks)),
            queued_tasks=tuple(task.task_id for task in state.queued_tasks),
            pending_confirmations=tuple(sorted(state.pending_confirmations)),
            failures=tuple(state.failures),
            event_count=len(state.event_log),
        )

    def diagnostics(self) -> dict[str, object]:
        return summarize(self.supervisor)
