from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any


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
    def final(cls, text: str) -> "AgentEvent":
        return cls(EventKind.STT_FINAL, {"text": text, "is_final": True}, priority=50)

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
    def confirm(cls, source: str = "voice") -> "AgentEvent":
        return cls(EventKind.CONTROL_CONFIRM, {"source": source}, priority=5)

    @classmethod
    def deny(cls, source: str = "voice") -> "AgentEvent":
        return cls(EventKind.CONTROL_DENY, {"source": source}, priority=5)
