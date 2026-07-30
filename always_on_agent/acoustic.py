"""Typed, PII-free lineage for microphone and replay transcript events.

The objects in this module are part of the cross-platform ``AgentEvent``
contract.  They intentionally contain identifiers and timing facts only:
transcript text, PCM, embeddings, and authorization decisions do not belong in
acoustic lineage.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Iterable, Mapping


_MAX_OPAQUE_ID_LENGTH = 128


class AcousticSource(str, Enum):
    UNKNOWN = "unknown"
    LIVE_CAPTURE = "live_capture"
    REMOTE_TRACK = "remote_track"
    FILE_REPLAY = "file_replay"
    SCRIPTED = "scripted"


class EndpointReason(str, Enum):
    UNKNOWN = "unknown"
    ASR = "asr"
    VAD_SILENCE = "vad_silence"
    SEMANTIC = "semantic"
    MAX_WAIT = "max_wait"
    COMMAND = "command"
    ABORTED = "aborted"


def _opaque_id(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or len(value) > _MAX_OPAQUE_ID_LENGTH:
        raise ValueError(f"{name} must contain 1-{_MAX_OPAQUE_ID_LENGTH} characters")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise ValueError(f"{name} cannot contain control characters")
    return value


def _non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _optional_non_negative_int(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    return _non_negative_int(name, value)


def _optional_clock(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative clock value")
    return result


@dataclass(frozen=True)
class AcousticSpan:
    """One capture-domain span contributing to a transcript.

    Monotonic timestamps are comparable only inside the same ``stream_id``.
    ``owned_sample_count`` includes model lookback and endpoint tail; consumers
    must not reinterpret it as voiced-speech duration.
    """

    stream_id: str
    utterance_id: str
    turn_index: int | None = None
    capture_epoch: int = 0
    capture_generation: int = 0
    source: AcousticSource = AcousticSource.UNKNOWN
    speech_start_at: float | None = None
    speech_end_at: float | None = None
    endpoint_committed_at: float | None = None
    emitted_at: float | None = None
    sample_rate_hz: int | None = None
    owned_sample_count: int | None = None
    endpoint_reason: EndpointReason = EndpointReason.UNKNOWN

    def __post_init__(self) -> None:
        _opaque_id("stream_id", self.stream_id)
        _opaque_id("utterance_id", self.utterance_id)
        _optional_non_negative_int("turn_index", self.turn_index)
        _non_negative_int("capture_epoch", self.capture_epoch)
        _non_negative_int("capture_generation", self.capture_generation)
        if not isinstance(self.source, AcousticSource):
            raise TypeError("source must be an AcousticSource")
        if not isinstance(self.endpoint_reason, EndpointReason):
            raise TypeError("endpoint_reason must be an EndpointReason")
        start = _optional_clock("speech_start_at", self.speech_start_at)
        end = _optional_clock("speech_end_at", self.speech_end_at)
        committed = _optional_clock("endpoint_committed_at", self.endpoint_committed_at)
        emitted = _optional_clock("emitted_at", self.emitted_at)
        sample_rate = _optional_non_negative_int("sample_rate_hz", self.sample_rate_hz)
        _optional_non_negative_int("owned_sample_count", self.owned_sample_count)
        if sample_rate == 0:
            raise ValueError("sample_rate_hz must be positive")
        if start is not None and end is not None and end < start:
            raise ValueError("speech_end_at cannot precede speech_start_at")
        if start is not None and committed is not None and committed < start:
            raise ValueError("endpoint_committed_at cannot precede speech_start_at")
        if end is not None and committed is not None and committed < end:
            raise ValueError("endpoint_committed_at cannot precede speech_end_at")
        if committed is not None and emitted is not None and emitted < committed:
            raise ValueError("emitted_at cannot precede endpoint_committed_at")

    @property
    def key(self) -> tuple[str, int, int, str]:
        return (
            self.stream_id,
            self.capture_epoch,
            self.capture_generation,
            self.utterance_id,
        )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "stream_id": self.stream_id,
            "utterance_id": self.utterance_id,
            "capture_epoch": self.capture_epoch,
            "capture_generation": self.capture_generation,
            "source": self.source.value,
            "endpoint_reason": self.endpoint_reason.value,
        }
        if self.turn_index is not None:
            payload["turn_index"] = self.turn_index
        for name in (
            "speech_start_at",
            "speech_end_at",
            "endpoint_committed_at",
            "emitted_at",
            "sample_rate_hz",
            "owned_sample_count",
        ):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "AcousticSpan":
        if not isinstance(payload, Mapping):
            raise TypeError("acoustic span payload must be a mapping")
        try:
            source = AcousticSource(str(payload.get("source", "unknown")))
            reason = EndpointReason(str(payload.get("endpoint_reason", "unknown")))
            return cls(
                stream_id=payload["stream_id"],
                utterance_id=payload["utterance_id"],
                turn_index=payload.get("turn_index"),
                capture_epoch=payload.get("capture_epoch", 0),
                capture_generation=payload.get("capture_generation", 0),
                source=source,
                speech_start_at=payload.get("speech_start_at"),
                speech_end_at=payload.get("speech_end_at"),
                endpoint_committed_at=payload.get("endpoint_committed_at"),
                emitted_at=payload.get("emitted_at"),
                sample_rate_hz=payload.get("sample_rate_hz"),
                owned_sample_count=payload.get("owned_sample_count"),
                endpoint_reason=reason,
            )
        except KeyError as exc:
            raise ValueError(f"missing acoustic span field: {exc.args[0]}") from exc


@dataclass(frozen=True)
class AcousticLineage:
    """Ordered acoustic spans contributing to one transcript."""

    spans: tuple[AcousticSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.spans, tuple):
            raise TypeError("spans must be a tuple")
        if not self.spans:
            raise ValueError("acoustic lineage requires at least one span")
        if any(not isinstance(span, AcousticSpan) for span in self.spans):
            raise TypeError("every acoustic lineage item must be an AcousticSpan")
        keys = [span.key for span in self.spans]
        if len(keys) != len(set(keys)):
            raise ValueError("acoustic lineage cannot contain duplicate spans")

    @classmethod
    def single(cls, span: AcousticSpan) -> "AcousticLineage":
        return cls((span,))

    @classmethod
    def combine(
        cls, lineages: Iterable["AcousticLineage | None"]
    ) -> "AcousticLineage | None":
        """Combine spans in submission order, replacing same-turn snapshots.

        A later snapshot of an existing span replaces the older one in place.
        This lets a final close the same utterance observed by partials without
        manufacturing two acoustic turns.
        """

        spans: list[AcousticSpan] = []
        positions: dict[tuple[str, int, int, str], int] = {}
        for lineage in lineages:
            if lineage is None:
                continue
            if not isinstance(lineage, cls):
                raise TypeError("lineages must contain AcousticLineage or None")
            for span in lineage.spans:
                position = positions.get(span.key)
                if position is None:
                    positions[span.key] = len(spans)
                    spans.append(span)
                else:
                    spans[position] = span
        return cls(tuple(spans)) if spans else None

    def to_payload(self) -> dict[str, Any]:
        return {"spans": [span.to_payload() for span in self.spans]}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "AcousticLineage":
        if not isinstance(payload, Mapping):
            raise TypeError("acoustic lineage payload must be a mapping")
        spans = payload.get("spans")
        if not isinstance(spans, list):
            raise ValueError("acoustic lineage spans must be a list")
        return cls(tuple(AcousticSpan.from_payload(span) for span in spans))

    @classmethod
    def try_from_payload(cls, payload: object) -> "AcousticLineage | None":
        """Parse untrusted adapter payloads without granting behavior/trust."""

        try:
            return cls.from_payload(payload)  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError):
            return None
