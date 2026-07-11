"""VAD-backed audio ownership for one streaming ASR utterance.

The recognizer stream is intentionally long-lived between endpoints, but the
offline finalizer must not inherit all of that idle audio.  ``ASRSegment`` keeps
only a short model-lookback window until real speech starts, then owns the whole
utterance through its endpoint.  It also turns VAD transitions into a real
trailing-silence clock; decoder token cadence is not an acoustic clock.

This module is pure state + NumPy array bookkeeping.  The live capture loop and
headless tests can therefore exercise the exact segment semantics without an
audio device or native model.
"""
from __future__ import annotations

from collections import deque
from typing import Optional


class ASRSegment:
    """Own audio/timing state for one endpointed utterance.

    ``vad_available=False`` preserves the legacy fail-open final-admission
    contract.  It deliberately disables semantic *early* endpointing because a
    changing decoder hypothesis alone cannot prove acoustic silence.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        pre_roll_sec: float,
        max_utterance_sec: float,
        vad_available: bool,
        block_sec: float,
    ) -> None:
        self.sample_rate = max(1, int(sample_rate))
        self.pre_roll_samples = max(0, int(round(self.sample_rate * pre_roll_sec)))
        self.max_samples = max(
            self.pre_roll_samples + 1,
            int(round(self.sample_rate * max_utterance_sec)),
        )
        self.vad_available = bool(vad_available)
        self.block_sec = max(0.0, float(block_sec))
        self.primary: deque = deque()
        self.alternate: deque = deque()
        self.samples = 0
        self.speech_seen = False
        self.vad_active = False
        self.first_speech_at: Optional[float] = None
        self.last_speech_at: Optional[float] = None
        self.last_text_at: Optional[float] = None
        self._quiet_started_at: Optional[float] = None
        # A speaker-authorized playback-time handoff may recover an utterance
        # with the offline recognizer even when BOTH streaming recognizers stay
        # empty.  This is decoding authority, not owner/action attestation.
        self.offline_recovery_authorized = False
        self.owner_lineage_intact = True
        self._has_word_cut_prefix = False

    def observe_vad(self, active: bool, now: float) -> Optional[float]:
        """Observe one post-front-end VAD verdict.

        Returns a completed mid-utterance quiet gap only when speech resumes.
        End-of-turn silence is never returned, so it cannot contaminate the
        session pause learner.
        """
        if not self.vad_available:
            return None
        active = bool(active)
        pause = None
        if active:
            if self._has_word_cut_prefix:
                # The final may still be decoded and answered, but a fresh
                # post-cut voice block was not part of the cut-time identity
                # decision.  Do not let an aggregate final embedding bless the
                # whole mixed span for owner-authorized actions.
                self.owner_lineage_intact = False
            if not self.speech_seen:
                self.speech_seen = True
                self.first_speech_at = float(now)
            elif not self.vad_active and self._quiet_started_at is not None:
                pause = max(0.0, float(now) - self._quiet_started_at)
            self.last_speech_at = float(now)
            self._quiet_started_at = None
        elif self.vad_active:
            self._quiet_started_at = float(now)
        self.vad_active = active
        return pause

    def observe_text(self, now: float) -> None:
        """Record a decoder advance; speech evidence only when VAD is absent."""
        self.last_text_at = float(now)
        if self.vad_available:
            return
        if not self.speech_seen:
            self.speech_seen = True
            self.first_speech_at = float(now)
        self.last_speech_at = float(now)

    def append(self, primary, alternate=None) -> None:
        """Append one processed capture block and enforce the ownership bound."""
        import numpy as np

        block = np.asarray(primary, dtype="float32").reshape(-1).copy()
        alt = (
            np.asarray(alternate, dtype="float32").reshape(-1).copy()
            if alternate is not None
            else None
        )
        if block.size:
            self.primary.append(block)
            if alt is not None:
                self.alternate.append(alt)
            self.samples += int(block.size)
        # With no VAD there is no acoustic speech-onset signal. Decoder text may
        # arrive seconds after the user began, so applying the idle pre-roll cap
        # before the first partial would silently discard the utterance head.
        # Keep the configured full (still bounded) window in the fail-open
        # fallback; only a real VAD may switch between idle and utterance bounds.
        limit = (
            self.max_samples
            if (not self.vad_available or self.speech_seen)
            else self.pre_roll_samples
        )
        self._trim_to(limit)

    def prepend(
        self,
        blocks,
        alternate_blocks=None,
        *,
        speech_at: Optional[float] = None,
        speech_end_at: Optional[float] = None,
        offline_recovery_authorized: bool = False,
    ) -> int:
        """Prepend confirmed playback-time user PCM and mark it as speech.

        Used by word-cut handoff: SenseVoice, floor gating and speaker-ID must
        see the same pre-cut audio that seeded the normal streaming recognizer.
        """
        import numpy as np

        primary = [np.asarray(b, dtype="float32").reshape(-1).copy() for b in blocks]
        alternates = (
            [np.asarray(b, dtype="float32").reshape(-1).copy() for b in alternate_blocks]
            if alternate_blocks is not None
            else [b.copy() for b in primary]
        )
        n = sum(int(b.size) for b in primary)
        for block in reversed(primary):
            self.primary.appendleft(block)
        for block in reversed(alternates):
            self.alternate.appendleft(block)
        self.samples += n
        if n:
            self.offline_recovery_authorized = bool(
                offline_recovery_authorized
            )
            self._has_word_cut_prefix = True
            at = float(speech_at) if speech_at is not None else self.last_speech_at
            if at is None:
                at = 0.0
            end_at = float(speech_end_at) if speech_end_at is not None else at
            end_at = max(at, end_at)
            self.speech_seen = True
            self.first_speech_at = (
                at if self.first_speech_at is None else min(self.first_speech_at, at)
            )
            self.last_speech_at = (
                end_at
                if self.last_speech_at is None
                else max(self.last_speech_at, end_at)
            )
        self._trim_to(self.max_samples)
        return n

    def _trim_to(self, limit: int) -> None:
        limit = max(0, int(limit))
        while self.samples > limit and self.primary:
            excess = self.samples - limit
            first = self.primary[0]
            if first.size <= excess:
                self.primary.popleft()
                if self.alternate:
                    self.alternate.popleft()
                self.samples -= int(first.size)
                continue
            self.primary[0] = first[excess:].copy()
            if self.alternate:
                alt = self.alternate[0]
                self.alternate[0] = alt[min(excess, alt.size):].copy()
            self.samples -= int(excess)

    def trailing_silence(self, now: float) -> float:
        if not self.speech_seen:
            return 0.0
        if self.vad_available and self.vad_active:
            return 0.0
        stamp = (
            self.last_speech_at
            if self.last_speech_at is not None
            else self.last_text_at
        )
        return max(0.0, float(now) - stamp) if stamp is not None else 0.0

    @property
    def early_endpoint_allowed(self) -> bool:
        return self.vad_available and self.speech_seen and not self.vad_active

    @property
    def final_admitted(self) -> bool:
        return (not self.vad_available) or self.speech_seen

    @property
    def speech_end_at(self) -> Optional[float]:
        return (
            self.last_speech_at
            if self.last_speech_at is not None
            else self.last_text_at
        )

    @property
    def speech_duration_sec(self) -> Optional[float]:
        # Decoder-change timestamps are not acoustic boundaries. Without VAD,
        # let the finalizer fall back to the owned PCM duration so a delayed
        # first/only partial cannot misclassify a multi-second utterance as a
        # 100 ms clip and skip the configured second pass.
        if not self.vad_available:
            return None
        if self.first_speech_at is None or self.speech_end_at is None:
            return None
        return max(
            self.block_sec,
            self.speech_end_at - self.first_speech_at + self.block_sec,
        )

    def arrays(self):
        """Return owned primary/alternate arrays without resetting state."""
        import numpy as np

        primary = np.concatenate(tuple(self.primary)) if self.primary else np.zeros(0, dtype="float32")
        alternate = (
            np.concatenate(tuple(self.alternate))
            if self.alternate
            else None
        )
        return primary, alternate

    def reset(self) -> None:
        self.primary.clear()
        self.alternate.clear()
        self.samples = 0
        self.speech_seen = False
        self.vad_active = False
        self.first_speech_at = None
        self.last_speech_at = None
        self.last_text_at = None
        self._quiet_started_at = None
        self.offline_recovery_authorized = False
        self.owner_lineage_intact = True
        self._has_word_cut_prefix = False
