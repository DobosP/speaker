"""
Dialogue controller for streaming ASR + LLM + TTS.
Mediates barge-in decisions and ignores noise/partial speech.
"""
import time
from dataclasses import dataclass


@dataclass
class BargeInInfo:
    rms: float
    threshold: float
    voiced: bool
    duration_sec: float
    timestamp: float
    echo: bool = False


class DialogueController:
    """
    Heuristic dialogue controller to improve reliability on devices.
    """

    def __init__(
        self,
        min_interrupt_delay_sec: float = 0.2,
        min_barge_in_sec: float = 0.2,
        min_partial_chars: int = 3,
        max_partial_age_sec: float = 1.5,
        echo_similarity_threshold: float = 0.7,
        allow_rms_fallback: bool = False,
        require_partial_for_barge_in: bool = True,
        strong_voiced_multiplier: float = 2.5,
        ignore_phrases: tuple[str, ...] = ("", ".", "uh", "um"),
    ):
        self.min_interrupt_delay_sec = min_interrupt_delay_sec
        self.min_barge_in_sec = min_barge_in_sec
        self.min_partial_chars = min_partial_chars
        self.max_partial_age_sec = max_partial_age_sec
        self.echo_similarity_threshold = echo_similarity_threshold
        self.allow_rms_fallback = allow_rms_fallback
        self.require_partial_for_barge_in = require_partial_for_barge_in
        self.strong_voiced_multiplier = strong_voiced_multiplier
        self.ignore_phrases = ignore_phrases

        self._assistant_speaking = False
        self._tts_start_time = 0.0
        self._last_partial_text = ""
        self._last_partial_time = 0.0
        self._last_assistant_text = ""
        self._partial_is_echo = False
        self._last_reason = "init"

    def on_tts_start(self):
        self._assistant_speaking = True
        self._tts_start_time = time.time()

    def on_tts_end(self):
        self._assistant_speaking = False
        self._partial_is_echo = False

    def on_assistant_text(self, text: str):
        if text:
            self._last_assistant_text = text.strip().lower()
        else:
            self._last_assistant_text = ""

    def on_partial_transcript(self, text: str):
        if text is None:
            return
        cleaned = text.strip().lower()
        if cleaned in self.ignore_phrases:
            return
        self._last_partial_text = cleaned
        self._last_partial_time = time.time()
        self._partial_is_echo = self._is_echo(cleaned)

    def _is_echo(self, partial: str) -> bool:
        if not partial or not self._last_assistant_text:
            return False
        if partial in self._last_assistant_text:
            return True
        # Simple similarity check to catch partial repeats
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(None, partial, self._last_assistant_text).ratio()
        return ratio >= self.echo_similarity_threshold

    def should_stop_speaking(self, info: BargeInInfo) -> bool:
        if not self._assistant_speaking:
            self._last_reason = "assistant_not_speaking"
            return False

        now = time.time()
        if now - self._tts_start_time < self.min_interrupt_delay_sec:
            self._last_reason = "interrupt_delay"
            return False

        if info.duration_sec < self.min_barge_in_sec:
            self._last_reason = "barge_in_too_short"
            return False

        if info.echo:
            self._last_reason = "echo_blocked_info"
            return False

        if info.voiced:
            if self._partial_is_echo:
                self._last_reason = "echo_blocked_voiced"
                return False
            if self.require_partial_for_barge_in:
                if info.rms < info.threshold * self.strong_voiced_multiplier:
                    self._last_reason = "voiced_too_weak_no_partial"
                    return False
            self._last_reason = "voiced"
            return True

        if self._last_partial_text:
            if len(self._last_partial_text) >= self.min_partial_chars:
                if now - self._last_partial_time <= self.max_partial_age_sec:
                    if self._partial_is_echo:
                        self._last_reason = "echo_blocked_partial"
                        return False
                    self._last_reason = "partial_text"
                    return True
        if self.require_partial_for_barge_in:
            self._last_reason = "no_partial_text"
            return False

        # Fallback: strong energy spike above threshold (optional)
        if self.allow_rms_fallback and not self._partial_is_echo:
            if info.rms > (info.threshold * 1.25):
                self._last_reason = "rms_fallback"
                return True
        self._last_reason = "no_rule_matched"
        return False

    def last_reason(self) -> str:
        return self._last_reason

    def should_ignore_transcript(self, text: str) -> bool:
        if not text:
            return True
        cleaned = text.strip().lower()
        if cleaned in self.ignore_phrases:
            return True
        if len(cleaned) < self.min_partial_chars:
            return True
        return False

