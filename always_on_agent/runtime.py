from __future__ import annotations

import itertools
from threading import Lock
import time
from dataclasses import dataclass

from .acoustic import AcousticLineage, AcousticSource, AcousticSpan
from .diagnostics import summarize
from .events import AgentEvent, Mode
from .supervisor import AgentSupervisor


_RUNTIME_STREAM_IDS = itertools.count(1)


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
        self._ingress_lock = Lock()
        self._ingress_stream_id = (
            f"always-on-text-{next(_RUNTIME_STREAM_IDS)}"
        )
        self._ingress_turn = 0
        self._ingress_lineage: AcousticLineage | None = None
        self._ingress_revision = -1

    def _next_ingress_identity_locked(
        self,
        *,
        final: bool,
    ) -> tuple[AcousticLineage, int]:
        if self._ingress_lineage is None:
            self._ingress_turn += 1
            self._ingress_lineage = AcousticLineage.single(
                AcousticSpan(
                    stream_id=self._ingress_stream_id,
                    utterance_id=f"u{self._ingress_turn}",
                    turn_index=self._ingress_turn,
                    source=AcousticSource.UNKNOWN,
                )
            )
            self._ingress_revision = -1
        self._ingress_revision += 1
        identity = (self._ingress_lineage, self._ingress_revision)
        if final:
            self._ingress_lineage = None
            self._ingress_revision = -1
        return identity

    def _abandon_ingress_turn_locked(self) -> None:
        self._ingress_lineage = None
        self._ingress_revision = -1

    @property
    def mode(self) -> Mode:
        return self.supervisor.state.mode

    def ingest_partial(self, text: str) -> None:
        with self._ingress_lock:
            acoustic, revision = self._next_ingress_identity_locked(
                final=False
            )
            self.supervisor.publish(
                AgentEvent.partial(
                    text,
                    acoustic=acoustic,
                    revision=revision,
                )
            )
            self.supervisor.drain()

    def ingest_final(self, text: str) -> None:
        with self._ingress_lock:
            acoustic, revision = self._next_ingress_identity_locked(
                final=True
            )
            self.supervisor.publish(
                AgentEvent.final(
                    text,
                    acoustic=acoustic,
                    revision=revision,
                )
            )
            self.supervisor.drain()

    def stop(self, reason: str = "external") -> None:
        with self._ingress_lock:
            self._abandon_ingress_turn_locked()
            self.supervisor.publish(AgentEvent.stop(reason))
            self.supervisor.drain()

    def wait_idle(self, timeout: float = 2.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._ingress_lock:
                self.supervisor.drain()
                if (
                    not self.supervisor.state.active_tasks
                    and not self.supervisor.state.queued_tasks
                ):
                    self.supervisor.drain()
                    return True
            time.sleep(0.01)
        with self._ingress_lock:
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
