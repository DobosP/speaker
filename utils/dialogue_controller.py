"""
Dialogue controller for streaming ASR + LLM + TTS.

Simplified version that trusts NLMS AEC + Silero VAD for echo rejection.
The controller focuses on turn-taking decisions and transcript filtering.
"""
import time
import json
import os
import threading
from dataclasses import dataclass, field
from enum import Enum

from utils import tts_debug
from utils.interruption_policies import (
    InterruptionContext,
    TimingPolicy,
    DurationPolicy,
    EchoPolicy,
    TranscriptConstraintPolicy,
    VoicedEnergyPolicy,
)


@dataclass
class BargeInInfo:
    rms: float
    threshold: float
    voiced: bool
    duration_sec: float
    timestamp: float
    echo: bool = False


class TurnState(Enum):
    IDLE = "idle"
    ASSISTANT_SPEAKING = "assistant_speaking"
    USER_TAKEOVER = "user_takeover"
    RECOVER = "recover"


class InterruptionStrategy(Enum):
    STRICT_ECHO_PROTECT = "strict_echo_protect"
    BALANCED = "balanced"
    AGGRESSIVE_USER_TAKEOVER = "aggressive_user_takeover"


class DialogueController:
    """
    Dialogue controller for barge-in decisions.

    With proper NLMS AEC + Silero VAD handling echo cancellation, the
    controller can be much simpler.  It mainly enforces:

    1. Minimum delay before allowing barge-in (avoid reacting to TTS onset)
    2. Minimum speech duration (avoid reacting to clicks/bumps)
    3. Voiced speech or partial transcript required (avoid noise)
    4. Transcript filtering (ignore filler words)
    """

    def __init__(
        self,
        min_interrupt_delay_sec: float = 0.2,
        min_barge_in_sec: float = 0.2,
        min_partial_chars: int = 3,
        max_partial_age_sec: float = 1.5,
        echo_similarity_threshold: float = 0.7,
        allow_rms_fallback: bool = False,
        require_partial_for_barge_in: bool = False,
        strong_voiced_multiplier: float = 2.0,
        ignore_phrases: tuple = ("", ".", "uh", "um"),
        interruption_strategy: str = "balanced",
        diagnostics_log_path: str | None = None,
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
        self.interruption_strategy = InterruptionStrategy(interruption_strategy)

        self._assistant_speaking = False
        self._state = TurnState.IDLE
        self._tts_start_time = 0.0
        self._last_partial_text = ""
        self._last_partial_time = 0.0
        self._last_assistant_text = ""
        self._partial_is_echo = False
        self._last_reason = "init"
        self._recovery_until = 0.0
        self._last_rms = 0.0
        self._vad_trend = 0.0
        self._energy_slope = 0.0
        self.diagnostics_log_path = diagnostics_log_path
        self._diag_lock = threading.Lock()
        self._policies = [
            TimingPolicy(min_interrupt_delay_sec=self.min_interrupt_delay_sec),
            DurationPolicy(min_barge_in_sec=self.min_barge_in_sec),
            EchoPolicy(),
            TranscriptConstraintPolicy(),
            VoicedEnergyPolicy(
                strong_voiced_multiplier=self.strong_voiced_multiplier,
                allow_rms_fallback=self.allow_rms_fallback,
            ),
        ]

    # ── TTS state ─────────────────────────────────────────────────────────

    def _diag_log(self, event: str, component: str = "policy", **payload):
        if not self.diagnostics_log_path:
            return
        record = {"ts": time.time(), "event": event, "component": component}
        record.update(payload)
        try:
            with self._diag_lock:
                parent = os.path.dirname(self.diagnostics_log_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with open(self.diagnostics_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=True) + "\n")
        except Exception:
            pass

    def on_tts_start(self):
        self._assistant_speaking = True
        self._tts_start_time = time.time()
        self._state = TurnState.ASSISTANT_SPEAKING
        self._diag_log("policy_tts_start", component="listener_state")
        tts_debug.log_tts(
            "dialogue_on_tts_start",
            state=self._state.value,
        )

    def on_tts_end(self):
        prev = self._state
        self._assistant_speaking = False
        self._partial_is_echo = False
        if self._state == TurnState.USER_TAKEOVER:
            self._state = TurnState.RECOVER
            self._recovery_until = time.time() + 0.25
        else:
            self._state = TurnState.IDLE
        self._diag_log("policy_tts_end", component="listener_state", state=self._state.value)
        tts_debug.log_tts(
            "dialogue_on_tts_end",
            prev_state=prev.value,
            state=self._state.value,
        )

    def on_assistant_text(self, text: str):
        self._last_assistant_text = text.strip().lower() if text else ""

    # ── Partial transcript ────────────────────────────────────────────────

    def on_partial_transcript(self, text: str):
        if text is None:
            return
        cleaned = text.strip().lower()
        if cleaned in self.ignore_phrases:
            return
        self._last_partial_text = cleaned
        self._last_partial_time = time.time()
        self._partial_is_echo = self._is_echo(cleaned)
        self._diag_log(
            "policy_partial",
            component="policy_decision",
            partial_len=len(cleaned),
            partial_is_echo=self._partial_is_echo,
        )
        # crude partial stability heuristic: repeated prefix implies stability
        if cleaned and self._last_partial_text:
            if cleaned.startswith(self._last_partial_text[: max(1, len(self._last_partial_text) // 2)]):
                self._vad_trend = min(1.0, self._vad_trend + 0.1)

    def _strategy_threshold(self) -> float:
        if self.interruption_strategy == InterruptionStrategy.STRICT_ECHO_PROTECT:
            return 3.0
        if self.interruption_strategy == InterruptionStrategy.AGGRESSIVE_USER_TAKEOVER:
            return 1.8
        return 2.5

    def _confidence_fusion_score(
        self,
        info: BargeInInfo,
        partial_recent: bool,
    ) -> float:
        score = 0.0
        if info.voiced:
            score += 2.0
        if info.rms > info.threshold:
            score += 1.0
        if info.rms > info.threshold * self.strong_voiced_multiplier:
            score += 1.0
        if partial_recent:
            score += 1.0
        if self._partial_is_echo:
            score -= 3.0
        elif partial_recent:
            score += 0.5
        # confidence fusion extensions
        score += max(-0.5, min(self._vad_trend, 0.5))
        score += max(-0.5, min(self._energy_slope * 2.0, 0.5))
        return score

    def _is_echo(self, partial: str) -> bool:
        """Check if partial transcript is an echo of the assistant's speech."""
        if not partial or not self._last_assistant_text:
            return False
        if partial in self._last_assistant_text:
            return True
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(
            None, partial, self._last_assistant_text
        ).ratio()
        return ratio >= self.echo_similarity_threshold

    # ── Barge-in decision ─────────────────────────────────────────────────

    def should_stop_speaking(self, info: BargeInInfo) -> bool:
        """
        Decide whether to stop TTS and yield the floor to the user.

        With NLMS AEC + Silero VAD, voiced=True on echo-cancelled audio
        is strong evidence of real user speech.  The controller adds
        timing and duration gates but mostly trusts the VAD signal.
        """
        if self._state == TurnState.RECOVER and time.time() < self._recovery_until:
            self._last_reason = "recovering"
            self._diag_log("policy_blocked", component="policy_decision", reason=self._last_reason)
            tts_debug.log_tts(
                "dialogue_barge_blocked",
                reason=self._last_reason,
                state=self._state.value,
            )
            return False
        if self._state == TurnState.RECOVER and time.time() >= self._recovery_until:
            self._state = TurnState.IDLE

        if not self._assistant_speaking or self._state != TurnState.ASSISTANT_SPEAKING:
            self._last_reason = "assistant_not_speaking"
            self._diag_log("policy_blocked", component="policy_decision", reason=self._last_reason)
            tts_debug.log_tts(
                "dialogue_barge_blocked",
                reason=self._last_reason,
                state=self._state.value,
            )
            return False

        now = time.time()

        # Update fusion signals
        self._energy_slope = info.rms - self._last_rms
        self._last_rms = info.rms
        if info.voiced:
            self._vad_trend = min(1.0, self._vad_trend + 0.2)
        else:
            self._vad_trend = max(-1.0, self._vad_trend - 0.1)

        partial_recent = (
            bool(self._last_partial_text)
            and len(self._last_partial_text) >= self.min_partial_chars
            and (now - self._last_partial_time <= self.max_partial_age_sec)
        )
        ctx = InterruptionContext(
            now=now,
            tts_elapsed_sec=now - self._tts_start_time,
            partial_recent=partial_recent,
            partial_is_echo=self._partial_is_echo,
            require_partial=self.require_partial_for_barge_in,
        )
        score = 0.0
        for policy in self._policies:
            allowed, reason, delta = policy.evaluate(info, ctx)
            if not allowed:
                self._last_reason = reason
                self._diag_log(
                    "policy_blocked",
                    component="policy_decision",
                    reason=reason,
                    rms=info.rms,
                    threshold=info.threshold,
                    voiced=info.voiced,
                    duration_sec=info.duration_sec,
                )
                tts_debug.log_tts(
                    "dialogue_barge_blocked",
                    reason=reason,
                    policy=type(policy).__name__,
                    rms=info.rms,
                    voiced=info.voiced,
                )
                return False
            score += delta
        # blend historical confidence signals
        score += max(-0.5, min(self._vad_trend, 0.5))
        score += max(-0.5, min(self._energy_slope * 2.0, 0.5))

        if score >= self._strategy_threshold():
            self._state = TurnState.USER_TAKEOVER
            self._last_reason = f"score={score:.2f}"
            self._diag_log(
                "policy_allowed",
                component="policy_decision",
                reason=self._last_reason,
                score=score,
            )
            tts_debug.log_tts(
                "dialogue_barge_allowed",
                reason=self._last_reason,
                score=score,
                console_kind="cancelled",
                console_detail="(barge-in)",
            )
            return True

        self._last_reason = f"score={score:.2f}_blocked"
        self._diag_log(
            "policy_blocked",
            component="policy_decision",
            reason=self._last_reason,
            score=score,
        )
        tts_debug.log_tts(
            "dialogue_barge_blocked",
            reason=self._last_reason,
            score=score,
        )
        return False

    def last_reason(self) -> str:
        return self._last_reason

    def state(self) -> str:
        return self._state.value

    # ── Transcript filtering ──────────────────────────────────────────────

    def should_ignore_transcript(self, text: str) -> bool:
        if not text:
            return True
        cleaned = text.strip().lower()
        if cleaned in self.ignore_phrases:
            return True
        if len(cleaned) < self.min_partial_chars:
            return True
        return False
