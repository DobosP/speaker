from __future__ import annotations

import itertools
from threading import Lock
import time
from dataclasses import dataclass

from .acoustic import AcousticLineage, AcousticSource, AcousticSpan
from .diagnostics import summarize
from .events import AgentEvent, EventKind, Mode
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
        self.supervisor.bus.subscribe(self._consume_text_output)

    def _consume_text_output(self, event: AgentEvent) -> None:
        """Terminalize speech ownership when this facade has no audio sink."""
        if event.kind != EventKind.TTS_REQUEST:
            return
        if event.payload.get("auxiliary_tts", False):
            # The historical method name means the one-shot auxiliary output
            # reached its terminal consumer; this facade does not claim that
            # physical playback happened.
            self.supervisor.note_aux_tts_admitted(
                str(event.payload.get("aux_tts_id", ""))
            )
            return
        self.supervisor.note_tts_rejected(
            str(event.payload.get("task_id", "")),
            event.payload,
        )

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
        def _quiet() -> bool:
            actor = self.supervisor.session_actor
            return bool(
                self.supervisor.bus.idle()
                and not self.supervisor.state.active_tasks
                and not self.supervisor.state.pending_audio_tasks
                and not self.supervisor.state.pending_aux_tts
                and not self.supervisor.state.queued_tasks
                and self.supervisor.pending_output_count == 0
                and self.supervisor.tasks.active_count == 0
                and actor.active_task_count == 0
                and actor.active_provider_count == 0
                and actor.active_tool_count == 0
                and actor.active_playback_count == 0
            )

        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._ingress_lock:
                self.supervisor.drain()
                if _quiet():
                    self.supervisor.drain()
                    if _quiet():
                        return True
            time.sleep(0.01)
        with self._ingress_lock:
            self.supervisor.drain()
            return _quiet()

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
