"""
Layered interruption policies inspired by production realtime stacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple


@dataclass
class InterruptionContext:
    now: float
    tts_elapsed_sec: float
    partial_recent: bool
    partial_is_echo: bool
    require_partial: bool


class InterruptionPolicy(Protocol):
    def evaluate(self, info, ctx: InterruptionContext) -> Tuple[bool, str, float]:
        """Return (allowed_to_continue, reason, score_delta)."""


class TimingPolicy:
    def __init__(self, min_interrupt_delay_sec: float):
        self.min_interrupt_delay_sec = min_interrupt_delay_sec

    def evaluate(self, info, ctx: InterruptionContext):
        if ctx.tts_elapsed_sec < self.min_interrupt_delay_sec:
            return False, "interrupt_delay", 0.0
        return True, "timing_ok", 0.0


class DurationPolicy:
    def __init__(self, min_barge_in_sec: float):
        self.min_barge_in_sec = min_barge_in_sec

    def evaluate(self, info, ctx: InterruptionContext):
        if info.duration_sec < self.min_barge_in_sec:
            return False, "barge_in_too_short", 0.0
        return True, "duration_ok", 0.0


class EchoPolicy:
    def evaluate(self, info, ctx: InterruptionContext):
        if info.echo:
            return False, "echo_blocked_info", 0.0
        if ctx.partial_is_echo:
            return False, "echo_blocked_partial", 0.0
        return True, "echo_ok", 0.0


class TranscriptConstraintPolicy:
    def evaluate(self, info, ctx: InterruptionContext):
        if ctx.require_partial and not ctx.partial_recent:
            return False, "missing_partial", 0.0
        return True, "partial_ok", 1.5 if ctx.partial_recent else 0.0


class VoicedEnergyPolicy:
    def __init__(self, strong_voiced_multiplier: float, allow_rms_fallback: bool):
        self.strong_voiced_multiplier = strong_voiced_multiplier
        self.allow_rms_fallback = allow_rms_fallback

    def evaluate(self, info, ctx: InterruptionContext):
        score = 0.0
        if info.voiced:
            score += 2.0
        if info.rms > info.threshold:
            score += 1.0
        if info.rms > info.threshold * self.strong_voiced_multiplier:
            score += 1.0
        if (
            self.allow_rms_fallback
            and not ctx.partial_is_echo
            and info.rms > info.threshold * 1.25
        ):
            score += 0.5
        return True, "energy_ok", score
