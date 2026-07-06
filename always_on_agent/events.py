from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Mapping


class EventKind(str, Enum):
    STT_PARTIAL = "stt.partial"
    STT_FINAL = "stt.final"
    SPEECH_OBSERVATION = "speech.observation"
    INTENT_DECISION = "intent.decision"
    CONTROL_STOP = "control.stop"
    CONTROL_MODE = "control.mode"
    CONTROL_CONFIRM = "control.confirm"
    CONTROL_DENY = "control.deny"
    TASK_STARTED = "task.started"
    TASK_PROGRESS = "task.progress"
    TASK_COMPLETED = "task.completed"
    TASK_CANCELLED = "task.cancelled"
    TASK_FAILED = "task.failed"
    TTS_REQUEST = "tts.request"
    MEMORY_COMMIT = "memory.commit"
    FOLLOWUP_TICK = "followup.tick"
    # Engine -> brain: input-stream lifecycle. ``payload`` carries
    # ``{"state": "open"|"recovering"|"fatal", "message": "..."}``. Lets
    # the brain surface "I'm reconnecting" feedback and stops the
    # watchdog from misattributing reopen gaps as "stuck turn".
    CAPTURE_STATE = "capture.state"


class Mode(str, Enum):
    PASSIVE = "passive"
    ASSISTANT = "assistant"
    COMMAND = "command"
    SEARCH = "search"
    RESEARCH = "research"
    DICTATION = "dictation"
    MEETING = "meeting"


@dataclass(frozen=True)
class AgentEvent:
    kind: EventKind
    payload: dict[str, Any] = field(default_factory=dict)
    priority: int = 100
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def partial(cls, text: str) -> "AgentEvent":
        return cls(EventKind.STT_PARTIAL, {"text": text, "is_final": False}, priority=90)

    @classmethod
    def final(
        cls,
        text: str,
        *,
        owner_verified: bool = False,
        origin: str = "unknown",
        metadata: Mapping[str, Any] | None = None,
    ) -> "AgentEvent":
        # owner_verified/origin carry the speaker-ID trust of this utterance for the
        # action chokepoint (always_on_agent.origin). Default FAIL-CLOSED (not the
        # owner, unknown origin) so a final published without a verdict can never
        # authorize a real action -- only an explicit owner-verified live-audio final
        # may. Set by the runtime from the SpeakerGate.
        payload = {
            "text": text,
            "is_final": True,
            "owner_verified": bool(owner_verified),
            "origin": origin,
        }
        if metadata:
            payload["metadata"] = dict(metadata)
        return cls(
            EventKind.STT_FINAL,
            payload,
            priority=50,
        )

    @classmethod
    def stop(cls, reason: str = "voice") -> "AgentEvent":
        return cls(EventKind.CONTROL_STOP, {"reason": reason}, priority=0)

    @classmethod
    def mode(cls, mode: Mode, source: str = "voice") -> "AgentEvent":
        return cls(
            EventKind.CONTROL_MODE,
            {"mode": mode.value, "source": source},
            priority=10,
        )

    @classmethod
    def confirm(cls, source: str = "voice", *, owner_verified: bool = False) -> "AgentEvent":
        # A confirmation is itself a consequential action -- approving a staged
        # command. owner_verified defaults FAIL-CLOSED so an ambient/leaked "yes"
        # cannot approve a side-effecting task; only an owner-verified confirm can.
        return cls(
            EventKind.CONTROL_CONFIRM,
            {"source": source, "owner_verified": bool(owner_verified)},
            priority=5,
        )

    @classmethod
    def deny(cls, source: str = "voice") -> "AgentEvent":
        return cls(EventKind.CONTROL_DENY, {"source": source}, priority=5)
