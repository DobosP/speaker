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
    TTS_STREAM_END = "tts.stream_end"
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
            "owner_verified": owner_verified is True,
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
    def stop(
        cls,
        reason: str = "voice",
        *,
        already_cancelled: bool = False,
        input_generation: int | None = None,
        input_epoch: int | None = None,
    ) -> "AgentEvent":
        payload: dict[str, Any] = {
            "reason": reason,
            "already_cancelled": bool(already_cancelled),
        }
        if input_generation is not None:
            payload["input_generation"] = int(input_generation)
        if input_epoch is not None:
            payload["input_epoch"] = int(input_epoch)
        return cls(
            EventKind.CONTROL_STOP,
            payload,
            priority=0,
        )

    @classmethod
    def mode(
        cls,
        mode: Mode,
        source: str = "voice",
        *,
        input_generation: int | None = None,
        input_epoch: int | None = None,
    ) -> "AgentEvent":
        payload: dict[str, Any] = {"mode": mode.value, "source": source}
        if input_generation is not None:
            payload["input_generation"] = int(input_generation)
        if input_epoch is not None:
            payload["input_epoch"] = int(input_epoch)
        return cls(
            EventKind.CONTROL_MODE,
            payload,
            priority=10,
        )

    @classmethod
    def confirm(
        cls,
        source: str = "voice",
        *,
        owner_verified: bool = False,
        origin: str = "unknown",
        direct_user_instruction: bool = False,
        input_generation: int | None = None,
        input_epoch: int | None = None,
    ) -> "AgentEvent":
        # A confirmation is itself a consequential action -- approving a staged
        # command. owner_verified defaults FAIL-CLOSED so an ambient/leaked "yes"
        # cannot approve a side-effecting task; only an owner-verified confirm can.
        payload: dict[str, Any] = {
            "source": source,
            "owner_verified": owner_verified is True,
            "origin": str(origin),
            "direct_user_instruction": direct_user_instruction is True,
        }
        if input_generation is not None:
            payload["input_generation"] = int(input_generation)
        if input_epoch is not None:
            payload["input_epoch"] = int(input_epoch)
        return cls(EventKind.CONTROL_CONFIRM, payload, priority=5)

    @classmethod
    def deny(
        cls,
        source: str = "voice",
        *,
        input_generation: int | None = None,
        input_epoch: int | None = None,
    ) -> "AgentEvent":
        payload: dict[str, Any] = {"source": source}
        if input_generation is not None:
            payload["input_generation"] = int(input_generation)
        if input_epoch is not None:
            payload["input_epoch"] = int(input_epoch)
        return cls(EventKind.CONTROL_DENY, payload, priority=5)
