"""Capture-thread-owned builder for typed acoustic transcript lineage."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from always_on_agent.acoustic import (
    AcousticLineage,
    AcousticSource,
    AcousticSpan,
    EndpointReason,
)


class AcousticTurnTracker:
    """Allocate opaque utterance IDs and monotonic transcript revisions.

    One capture/ASR owner calls this object. It contains metadata only and never
    stores PCM or transcript text.
    """

    def __init__(
        self,
        stream_id: str,
        *,
        source: AcousticSource = AcousticSource.LIVE_CAPTURE,
    ) -> None:
        self.stream_id = stream_id
        self.source = source
        self._next_utterance = 0
        self._capture_epoch = 0
        self._capture_generation = 0
        self._utterance_id: Optional[str] = None
        self._speech_start_at: Optional[float] = None
        self._revision = -1
        self._last_emitted_at: Optional[float] = None

    def rotate_capture(self, *, capture_epoch: int, capture_generation: int) -> None:
        self.abandon()
        self._capture_epoch = max(0, int(capture_epoch))
        self._capture_generation = max(0, int(capture_generation))

    def ensure_started(self, speech_start_at: Optional[float] = None) -> None:
        if self._utterance_id is None:
            self._next_utterance += 1
            self._utterance_id = f"u{self._next_utterance}"
            self._revision = -1
        if speech_start_at is not None:
            stamp = max(0.0, float(speech_start_at))
            self._speech_start_at = (
                stamp
                if self._speech_start_at is None
                else min(self._speech_start_at, stamp)
            )

    @property
    def active(self) -> bool:
        return self._utterance_id is not None

    @property
    def last_emitted_at(self) -> Optional[float]:
        return self._last_emitted_at

    def partial(
        self,
        *,
        emitted_at: float,
        speech_start_at: Optional[float] = None,
    ) -> tuple[AcousticLineage, int]:
        return self.advance(
            emitted_at=emitted_at,
            speech_start_at=speech_start_at,
        )

    def advance(
        self,
        *,
        emitted_at: float,
        speech_start_at: Optional[float] = None,
    ) -> tuple[AcousticLineage, int]:
        """Advance one typed observation without closing the acoustic turn."""
        self.ensure_started(speech_start_at)
        stamp = max(0.0, float(emitted_at))
        self._revision += 1
        self._last_emitted_at = stamp
        return (
            AcousticLineage.single(
                self._span(
                    emitted_at=stamp,
                    endpoint_reason=EndpointReason.UNKNOWN,
                )
            ),
            self._revision,
        )

    def terminal(
        self,
        *,
        emitted_at: float,
    ) -> tuple[AcousticLineage, int]:
        """Close a consumed command turn at its engine-owned publication seam."""
        stamp = max(0.0, float(emitted_at))
        self.ensure_started(stamp)
        self._revision += 1
        self._last_emitted_at = stamp
        terminal = (
            AcousticLineage.single(
                self._span(
                    endpoint_committed_at=stamp,
                    emitted_at=stamp,
                    endpoint_reason=EndpointReason.COMMAND,
                )
            ),
            self._revision,
        )
        self.abandon()
        return terminal

    def close(
        self,
        *,
        speech_end_at: Optional[float],
        endpoint_committed_at: float,
        emitted_at: Optional[float] = None,
        sample_rate_hz: Optional[int] = None,
        owned_sample_count: Optional[int] = None,
        endpoint_reason: EndpointReason = EndpointReason.UNKNOWN,
        speech_start_at: Optional[float] = None,
    ) -> tuple[AcousticLineage, int]:
        self.ensure_started(speech_start_at)
        self._revision += 1
        lineage = AcousticLineage.single(
            self._span(
                speech_end_at=speech_end_at,
                endpoint_committed_at=endpoint_committed_at,
                emitted_at=emitted_at,
                sample_rate_hz=sample_rate_hz,
                owned_sample_count=owned_sample_count,
                endpoint_reason=endpoint_reason,
            )
        )
        revision = self._revision
        self.abandon()
        return lineage, revision

    def abort(
        self,
        *,
        emitted_at: float,
    ) -> tuple[AcousticLineage, int] | None:
        """Close an active turn without transcript text."""
        if not self.active:
            return None
        self._revision += 1
        lineage = AcousticLineage.single(
            self._span(
                endpoint_committed_at=emitted_at,
                emitted_at=emitted_at,
                endpoint_reason=EndpointReason.ABORTED,
            )
        )
        revision = self._revision
        self.abandon()
        return lineage, revision

    @staticmethod
    def with_emitted_at(
        lineage: AcousticLineage,
        emitted_at: float,
    ) -> AcousticLineage:
        return AcousticLineage(
            tuple(
                replace(span, emitted_at=max(0.0, float(emitted_at)))
                for span in lineage.spans
            )
        )

    def current(
        self, *, emitted_at: Optional[float] = None
    ) -> Optional[AcousticLineage]:
        if self._utterance_id is None:
            return None
        return AcousticLineage.single(
            self._span(
                emitted_at=emitted_at,
                endpoint_reason=EndpointReason.UNKNOWN,
            )
        )

    def abandon(self) -> None:
        self._utterance_id = None
        self._speech_start_at = None
        self._revision = -1
        self._last_emitted_at = None

    def _span(
        self,
        *,
        speech_end_at: Optional[float] = None,
        endpoint_committed_at: Optional[float] = None,
        emitted_at: Optional[float] = None,
        sample_rate_hz: Optional[int] = None,
        owned_sample_count: Optional[int] = None,
        endpoint_reason: EndpointReason,
    ) -> AcousticSpan:
        if self._utterance_id is None:
            raise RuntimeError("acoustic turn has not started")
        return AcousticSpan(
            stream_id=self.stream_id,
            utterance_id=self._utterance_id,
            turn_index=self._next_utterance,
            capture_epoch=self._capture_epoch,
            capture_generation=self._capture_generation,
            source=self.source,
            speech_start_at=self._speech_start_at,
            speech_end_at=speech_end_at,
            endpoint_committed_at=endpoint_committed_at,
            emitted_at=emitted_at,
            sample_rate_hz=sample_rate_hz,
            owned_sample_count=owned_sample_count,
            endpoint_reason=endpoint_reason,
        )
