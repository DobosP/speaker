"""Cheap conversation evidence before an ASR final may replace active work.

This is a candidate gate, never speaker identity or tool authority. Explicit
questions/requests need no wake word. A heard question opens a short answer
window; arbitrary room fragments cannot open or extend that window themselves.
The learned addressing verdict remains a separate, later decision.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from threading import Lock
import time
from typing import Callable, Mapping

from .acoustic import AcousticLineage

_WORDS = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:[.,]\d+)?", re.I)
_REQUEST = re.compile(
    r"^(?:please\s+)?(?:"
    r"(?:what|where|when|who|why|how|which)\s+(?:is|are|was|were|do|does|did|can|could|would|will|should|have|has|about|else|many|much|long|far|old|time|day|date|happened|next|if)\b|"
    r"(?:can|could|would|will|do|did|are|have)\s+you\b|"
    r"(?:is|are|was|were|does|did|should|may|can|could|would|will|have|has)\s+(?:the|a|an|it|this|that|there|i|we|my|our)\b|"
    r"(?:tell|show|give|help|explain|teach|describe|summarize|translate|calculate|compare|convert|define|list|find|search|research|remember|remind|repeat|say|write|read|play|pause|resume|continue|stop|cancel|open|launch|close|set|reset|start|switch|change|make|turn|look|create|delete|add|remove|draft|generate|plan|schedule|explain|spell|check|review|fix|solve|recommend)\s+\S|"
    r"i\s+(?:want|need|would\s+like|was\s+wondering|wonder)\b"
    r")",
    re.I,
)
_FILLERS = frozenset({"uh", "um", "hmm", "hm", "ah", "er", "oh"})

_FOLLOWUP_REQUESTS = frozenset({
    "tell me more",
    "go on",
    "start again",
    "try again",
    "anything else",
    "and then",
    "why",
    "how come",
    "hello",
    "hello there",
    "hi",
    "hey",
    "thank you",
    "thanks",
    "any updates",
    "any news",
})

# Preserve the controller's explicit non-assistant request prefixes that the
# conversational verb list does not already include (speech_analyzer.py).
# Match raw leading syntax so a quoted command is not manufactured by _WORDS.
_CANONICAL_REQUEST = re.compile(
    r"^(?:please\s+)?(?:run|execute|dictate|cerceteaza|cauta|scrie|browse|consult|query)\s+\S|"
    r"^(?:please\s+)?go\s+(?:in|into|inside(?:\s+of)?|through|to|within)\s+\S", re.I
)
_GREETING_PREFIX = re.compile(r"^(?:hey|hello|hi|okay|ok)[\s,:!]+", re.I)
_ASSISTANT_PREFIX = re.compile(r"^(?:assistant|computer|jarvis|asistent)[\s,:]+", re.I)
_POLITE_PREFIX = re.compile(r"^(?:please|kindly)\s+", re.I)
_QUOTE_STARTS = ('"', "'", "“", "‘")


@dataclass(frozen=True)
class ConversationAdmissionConfig:
    enabled: bool = True
    answer_window_sec: float = 8.0
    assistant_name: str = ""

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("conversation admission enabled must be boolean")
        if (
            isinstance(self.answer_window_sec, bool)
            or not isinstance(self.answer_window_sec, (int, float))
            or not math.isfinite(self.answer_window_sec)
            or not 0 <= self.answer_window_sec <= 30
        ):
            raise ValueError(
                "answer window must be finite and between 0 and 30 seconds"
            )
        if not isinstance(self.assistant_name, str) or len(self.assistant_name) > 80:
            raise ValueError("assistant name must be at most 80 characters")

    @classmethod
    def from_dict(
        cls, data: Mapping[str, object] | None, *, assistant_name: str = ""
    ) -> ConversationAdmissionConfig:
        data = data if isinstance(data, Mapping) else {}
        return cls(
            enabled=data.get("enabled", True),
            answer_window_sec=data.get("answer_window_sec", 8.0),
            assistant_name=assistant_name,
        )


@dataclass(frozen=True)
class ConversationCue:
    candidate: bool
    reason: str


class ConversationAdmission:
    def __init__(
        self,
        config: ConversationAdmissionConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.config = config
        self._clock = clock
        self._name = tuple(_WORDS.findall(config.assistant_name.lower()))
        self._lock = Lock()
        self._admitted: tuple[int, int] | None = None
        self._answer_until = 0.0
        self._latest_arrival: tuple[int, int] | None = None
        self._partial_cue: tuple[int, frozenset, float, bool] | None = None

    def observe(
        self, text: str, *, input_epoch: int, acoustic=None, partial: bool = False
    ) -> ConversationCue:
        """Freeze a positive cue only for the same acoustic utterance's final."""
        cue = self.inspect(text, input_epoch=input_epoch)
        keys = (
            frozenset(span.key for span in acoustic.spans)
            if acoustic is not None
            else None
        )
        if cue.reason in {"no_content", "oversized", "invalid_type"}:
            if acoustic is not None:
                self.abandon(acoustic)
            return cue
        now = self._clock()
        with self._lock:
            ticket = self._partial_cue
            carried = bool(
                ticket is not None
                and keys is not None
                and ticket[0] == input_epoch
                and ticket[1] == keys
                and now < ticket[2]
            )
            if ticket is not None and (ticket[0] != input_epoch or now >= ticket[2]):
                self._partial_cue = None
            if cue.reason == "quoted_text" and not (carried and ticket[3]):
                # Idle quoted instructions cannot inherit a request cue. Only
                # this exact utterance's prior heard-answer cue survives the
                # arrival-time window closure; it adds no action authority.
                if ticket is not None and ticket[1] == keys:
                    self._partial_cue = None
                return cue
            if carried:
                cue = ConversationCue(True, "same_utterance")
            if partial and cue.candidate and keys and not carried:
                self._partial_cue = (
                    input_epoch, keys, now + 30.0, cue.reason == "heard_question"
                )
            elif not partial and cue.candidate:
                self._partial_cue = None
        return cue

    def note_arrival(self, *, input_epoch: int, input_generation: int) -> None:
        with self._lock:
            self._latest_arrival = (input_epoch, input_generation)
            self._answer_until = 0.0

    def abandon(self, acoustic: AcousticLineage) -> None:
        keys = frozenset(span.key for span in acoustic.spans)
        with self._lock:
            if self._partial_cue is not None and self._partial_cue[1] == keys:
                self._partial_cue = None

    def inspect(self, text: str, *, input_epoch: int) -> ConversationCue:
        if not self.config.enabled:
            return ConversationCue(True, "disabled")
        if type(text) is not str:
            return ConversationCue(False, "invalid_type")
        if len(text) > 8192:
            return ConversationCue(False, "oversized")
        words = tuple(_WORDS.findall(text.lower()))
        if not words or all(word in _FILLERS for word in words):
            return ConversationCue(False, "no_content")
        raw = text.strip()
        # Prefix removal is cue inspection only. The immutable input passed to
        # addressing, the controller and provenance checks is never rewritten.
        request_text = _GREETING_PREFIX.sub("", raw, count=1)
        request_text = _POLITE_PREFIX.sub("", request_text, count=1)
        request_text = _ASSISTANT_PREFIX.sub("", request_text, count=1)
        request_text = _POLITE_PREFIX.sub("", request_text, count=1)
        quoted = raw.startswith(_QUOTE_STARTS) or request_text.startswith(_QUOTE_STARTS)
        original_normalized = " ".join(words)
        normalized = (
            original_normalized if request_text == raw
            else " ".join(_WORDS.findall(request_text.lower()))
        )
        normalized = re.sub(
            r"\b(what|where|when|who|why|how)'s\b", r"\1 is", normalized
        )
        normalized = normalized.replace("i'd ", "i would ")
        aux_question = bool(
            re.match(
                r"^(?:is|are|was|were|do|does|did|can|could|would|will|should|may|have|has)\s+\S+\s+\S",
                normalized,
            )
        )
        # Open wh-questions need no wake name; exclamations and quoted-room
        # statements still go through the ambient path.
        wh_question = bool(
            re.match(r"^(?:who|where|when|why|how|which)\s+\S", normalized)
        )
        what_question = bool(re.match(r"^what\s+(?!a\b|an\b)\S+\s+\S", normalized))
        named = words[1:] if words[0] in {"hey", "hello", "hi", "okay", "ok"} else words
        if not quoted and self._name and named[: len(self._name)] == self._name:
            return ConversationCue(True, "addressed_name")
        if not quoted and (
            _CANONICAL_REQUEST.match(request_text)
            or wh_question or what_question or aux_question or _REQUEST.search(normalized)
        ):
            return ConversationCue(True, "request")
        # Simple elliptical follow-up requests are still requests, even without
        # an open answer window or a model inventing engagement from memory.
        if not quoted and (
            normalized.startswith(("weather in ", "weather for ", "a recipe for ")) or (
                normalized.endswith(" please") and len(words) >= 3
            )
        ):
            return ConversationCue(True, "elliptical_request")
        if not quoted and (
            normalized in _FOLLOWUP_REQUESTS or original_normalized in _FOLLOWUP_REQUESTS
        ):
            return ConversationCue(True, "followup_request")
        with self._lock:
            if (
                self._admitted is not None
                and self._admitted[0] == input_epoch
                and self._clock() < self._answer_until
            ):
                return ConversationCue(True, "heard_question")
        return ConversationCue(False, "quoted_text" if quoted else "no_engagement")

    def note_admitted(self, *, input_epoch: int, input_generation: int) -> None:
        with self._lock:
            self._admitted = (input_epoch, input_generation)
            self._latest_arrival = self._admitted
            self._answer_until = 0.0
            self._partial_cue = None

    def transfer_admitted(
        self, *, input_epoch: int, previous_generation: int, input_generation: int
    ) -> None:
        """Retarget only a proven restored, already-admitted unheard input."""
        with self._lock:
            if self._admitted == (
                input_epoch,
                previous_generation,
            ) and self._latest_arrival == (input_epoch, input_generation):
                self._admitted = self._latest_arrival

    def note_rendered(
        self, text: str, *, input_epoch: int, input_generation: int
    ) -> None:
        # Only exact current reply receipts can open a window. A queued,
        # cancelled, unplayed or unrelated background notification cannot.
        if not text.rstrip().endswith("?"):
            return
        with self._lock:
            if (
                self._admitted
                == self._latest_arrival
                == (input_epoch, input_generation)
            ):
                self._answer_until = self._clock() + self.config.answer_window_sec

    def invalidate(self) -> None:
        with self._lock:
            self._admitted = None
            self._latest_arrival = None
            self._partial_cue = None
            self._answer_until = 0.0
