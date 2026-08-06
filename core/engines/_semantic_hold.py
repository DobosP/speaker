"""Pure state for API-compliant native endpoint reset-and-accumulate.

Sherpa's online endpoint is a native stream boundary.  A semantic HOLD may
override that boundary, but the native stream must then be reset before it can
decode resumed speech.  This module keeps only the stripped native hypotheses
needed to compose the logical turn and recreates the two backstops whose native
clocks restart with the stream.

The state is capture-scope bound and deliberately contains no logging or
serialization hooks.  In particular, its repr never exposes transcript text.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional


def _scope_value(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _duration(name: str, value: float, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (positive and result <= 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


@dataclass(frozen=True, slots=True)
class SemanticEndpointBoundary:
    """Aggregate-safe effective endpoint inputs for one capture tick."""

    native_acoustic_endpoint: bool
    semantic_hold_active: bool
    max_silence_backstop: bool
    hard_boundary: bool
    utterance_elapsed_sec: float

    @property
    def effective_acoustic_endpoint(self) -> bool:
        return bool(
            self.native_acoustic_endpoint
            or self.max_silence_backstop
            or self.hard_boundary
        )


def resolve_semantic_endpoint_boundary(
    *,
    enabled: bool,
    native_acoustic_endpoint: bool,
    semantic_hold_active: bool,
    vad_active: bool,
    trailing_silence_sec: float,
    utterance_elapsed_sec: float,
    max_silence_sec: float,
    rule3_min_utterance_length_sec: float,
) -> SemanticEndpointBoundary:
    """Restore total-silence and logical-rule3 bounds after a native reset.

    A prior native endpoint is never carried into resumed active speech.  Only
    the current native verdict, the current VAD-backed total quiet clock, or the
    current VAD-active logical-turn age can make the effective endpoint true.
    """

    for name, value in (
        ("enabled", enabled),
        ("native_acoustic_endpoint", native_acoustic_endpoint),
        ("semantic_hold_active", semantic_hold_active),
        ("vad_active", vad_active),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a boolean")
    trailing = _duration("trailing_silence_sec", trailing_silence_sec)
    elapsed = _duration("utterance_elapsed_sec", utterance_elapsed_sec)
    max_silence = _duration("max_silence_sec", max_silence_sec, positive=True)
    rule3 = _duration(
        "rule3_min_utterance_length_sec",
        rule3_min_utterance_length_sec,
        positive=True,
    )
    active = bool(enabled and semantic_hold_active)
    max_backstop = bool(active and not vad_active and trailing >= max_silence)
    hard_boundary = bool(active and vad_active and elapsed >= rule3)
    return SemanticEndpointBoundary(
        native_acoustic_endpoint=native_acoustic_endpoint,
        semantic_hold_active=active,
        max_silence_backstop=max_backstop,
        hard_boundary=hard_boundary,
        utterance_elapsed_sec=elapsed,
    )


@dataclass(slots=True)
class HeldASRText:
    """Capture-scoped native hypotheses retained across semantic HOLD resets."""

    enabled: bool = False
    capture_epoch: Optional[int] = None
    capture_generation: Optional[int] = None
    _parts: list[str] = field(default_factory=list, init=False, repr=False)
    _reset_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a boolean")

    @property
    def active(self) -> bool:
        return bool(self.enabled and self._reset_count)

    @property
    def reset_count(self) -> int:
        return self._reset_count

    def bind(self, *, capture_epoch: int, capture_generation: int) -> None:
        """Bind to one capture scope, clearing text on any scope rotation."""

        epoch = _scope_value("capture_epoch", capture_epoch)
        generation = _scope_value("capture_generation", capture_generation)
        if (self.capture_epoch, self.capture_generation) != (epoch, generation):
            self.clear()
            self.capture_epoch = epoch
            self.capture_generation = generation

    def _require_scope(self, capture_epoch: int, capture_generation: int) -> None:
        if not self.enabled:
            return
        epoch = _scope_value("capture_epoch", capture_epoch)
        generation = _scope_value("capture_generation", capture_generation)
        if (self.capture_epoch, self.capture_generation) != (epoch, generation):
            raise RuntimeError("semantic HOLD text used outside its capture scope")

    @staticmethod
    def _piece(text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("ASR text must be a string")
        return text.strip()

    def compose(
        self,
        current_native_text: str,
        *,
        capture_epoch: int,
        capture_generation: int,
    ) -> str:
        """Compose retained pieces and the current stream with one ASCII space."""

        if not self.enabled:
            # Preserve the established path exactly, including native whitespace.
            return current_native_text
        self._require_scope(capture_epoch, capture_generation)
        current = self._piece(current_native_text)
        if not current:
            return " ".join(self._parts)
        return " ".join((*self._parts, current))

    def hold(
        self,
        current_native_text: str,
        *,
        capture_epoch: int,
        capture_generation: int,
    ) -> int:
        """Retain one native result and return its one-based reset ordinal."""

        if not self.enabled:
            raise RuntimeError("semantic HOLD reset is disabled")
        logical = self.compose(
            current_native_text,
            capture_epoch=capture_epoch,
            capture_generation=capture_generation,
        )
        if not logical:
            raise ValueError("semantic HOLD requires non-empty logical ASR text")
        piece = self._piece(current_native_text)
        if piece:
            self._parts.append(piece)
        self._reset_count += 1
        return self._reset_count

    def consume(
        self,
        current_native_text: str,
        *,
        capture_epoch: int,
        capture_generation: int,
    ) -> str:
        """Return the complete logical final and erase retained text exactly once."""

        result = self.compose(
            current_native_text,
            capture_epoch=capture_epoch,
            capture_generation=capture_generation,
        )
        self.clear()
        return result

    def clear(self) -> None:
        for index in range(len(self._parts)):
            self._parts[index] = ""
        self._parts.clear()
        self._reset_count = 0
