"""
Deterministic conversation routing before LLM generation.

The router is intentionally cheap and explainable: it handles high-priority
control intents and simple local capabilities before falling back to the LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any


class RouteAction(str, Enum):
    IGNORE = "ignore"
    STOP_OUTPUT = "stop_output"
    SHUTDOWN = "shutdown"
    CAPABILITY = "capability"
    LLM = "llm"


@dataclass(frozen=True)
class RouteContext:
    transcript: str
    assistant_speaking: bool = False
    barge_in_active: bool = False
    is_partial: bool = False
    mode: str = "asr"
    available_capabilities: tuple[str, ...] = ()


@dataclass(frozen=True)
class RouteDecision:
    action: RouteAction
    reason: str
    normalized_text: str = ""
    capability: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


_FILLER_PHRASES = {"", ".", "uh", "um", "erm", "hmm", "mm"}
_STOP_OUTPUT_PHRASES = {
    "stop",
    "stop talking",
    "cancel",
    "cancel that",
    "enough",
    "thats enough",
    "that is enough",
    "be quiet",
    "silence",
    "pause",
}
_SHUTDOWN_PHRASES = {
    "quit",
    "exit",
    "shutdown",
    "shut down",
    "goodbye",
    "stop assistant",
    "close assistant",
}


def normalize_transcript(text: str | None) -> str:
    """Normalize ASR text for routing without making fuzzy unsafe matches."""
    if not text:
        return ""
    cleaned = text.lower().strip()
    cleaned = cleaned.replace("'", "")
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    words = cleaned.split()
    collapsed: list[str] = []
    for word in words:
        if not collapsed or collapsed[-1] != word:
            collapsed.append(word)
    return " ".join(collapsed)


class ConversationRouter:
    """Route user utterances to control, capability, or LLM paths."""

    def __init__(
        self,
        stop_phrases: tuple[str, ...] = ("stop", "quit", "exit"),
        stop_mode: str = "exact",
    ):
        self.stop_mode = stop_mode
        self._configured_stop_phrases = {
            normalize_transcript(phrase) for phrase in stop_phrases
        }

    def route(self, ctx: RouteContext) -> RouteDecision:
        text = normalize_transcript(ctx.transcript)
        if text in _FILLER_PHRASES:
            return RouteDecision(RouteAction.IGNORE, "empty_or_filler", text)

        shutdown = self._matches_any(text, _SHUTDOWN_PHRASES)
        if shutdown:
            return RouteDecision(RouteAction.SHUTDOWN, "shutdown_phrase", text)

        stop_output = self._matches_stop_output(text)
        if stop_output:
            return RouteDecision(RouteAction.STOP_OUTPUT, "stop_output_phrase", text)

        capability = self._route_capability(text, ctx.available_capabilities)
        if capability is not None:
            name, payload, reason = capability
            return RouteDecision(
                RouteAction.CAPABILITY,
                reason,
                text,
                capability=name,
                payload=payload,
            )

        if ctx.is_partial:
            return RouteDecision(RouteAction.IGNORE, "partial_not_control", text)
        return RouteDecision(RouteAction.LLM, "llm_fallback", text)

    def route_partial(self, ctx: RouteContext) -> RouteDecision:
        """Only allow high-confidence realtime control on partial transcripts."""
        text = normalize_transcript(ctx.transcript)
        if not text:
            return RouteDecision(RouteAction.IGNORE, "empty_partial", text)
        if self._matches_any(text, _SHUTDOWN_PHRASES):
            return RouteDecision(RouteAction.SHUTDOWN, "partial_shutdown_phrase", text)
        if self._matches_any(text, _STOP_OUTPUT_PHRASES):
            return RouteDecision(RouteAction.STOP_OUTPUT, "partial_stop_phrase", text)
        return RouteDecision(RouteAction.IGNORE, "partial_not_control", text)

    def _matches_stop_output(self, text: str) -> bool:
        if self._matches_any(text, _STOP_OUTPUT_PHRASES):
            return True
        configured = self._configured_stop_phrases - _SHUTDOWN_PHRASES
        if self.stop_mode == "prefix":
            return any(
                phrase and (text == phrase or text.startswith(f"{phrase} "))
                for phrase in configured
            )
        return text in configured

    @staticmethod
    def _matches_any(text: str, phrases: set[str]) -> bool:
        return any(
            phrase and (text == phrase or text.startswith(f"{phrase} "))
            for phrase in phrases
        )

    @staticmethod
    def _route_capability(
        text: str,
        available: tuple[str, ...],
    ) -> tuple[str, dict[str, Any], str] | None:
        if "system.time" in available and (
            "what time" in text
            or text in {"time", "current time"}
            or text.startswith("tell me the time")
        ):
            return "system.time", {"query": text}, "capability_time"
        if "debug.echo" in available and text.startswith("debug echo"):
            return "debug.echo", {"text": text.removeprefix("debug echo").strip()}, "capability_debug_echo"
        return None
