from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time

from .events import Mode


class IntentKind(str, Enum):
    IGNORE = "ignore"
    STOP = "stop"
    CONFIRM = "confirm"
    DENY = "deny"
    MODE_SWITCH = "mode_switch"
    ASSISTANT = "assistant"
    SEARCH = "search"
    RESEARCH = "research"
    COMMAND = "command"
    DICTATION = "dictation"
    MEETING_NOTE = "meeting_note"


@dataclass(frozen=True)
class SpeechObservation:
    text: str
    normalized: str
    is_final: bool
    language: str = "unknown"
    stability: float = 0.0
    activation_score: float = 0.0
    keywords: tuple[str, ...] = ()
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class IntentDecision:
    kind: IntentKind
    confidence: float
    text: str
    reason: str
    mode: Mode | None = None
    target_mode: Mode | None = None
    requires_confirmation: bool = False
    speak: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def starts_task(self) -> bool:
        return self.kind in {
            IntentKind.ASSISTANT,
            IntentKind.SEARCH,
            IntentKind.RESEARCH,
            IntentKind.COMMAND,
            IntentKind.DICTATION,
            IntentKind.MEETING_NOTE,
        }
