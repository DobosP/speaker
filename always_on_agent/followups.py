"""Proactive ("always-listening") follow-up policy.

Ported from Vocalis (Apache-2.0, github.com/Lex-au/Vocalis): after the assistant
finishes speaking and the user stays silent, the assistant gently continues the
conversation. A cycling marker ([silent] -> [no response] -> [still waiting]) is
fed to the model, which a system prompt teaches to interpret as "they went
quiet, keep going naturally". Follow-ups are bounded and are not persisted to
memory so they don't pollute long-term history.

This module holds only the pure policy (timing-free, thread-free) so it is
trivially testable; the supervisor owns the timer that drives it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

DEFAULT_MARKERS = ("[silent]", "[no response]", "[still waiting]")


@dataclass
class FollowupConfig:
    enabled: bool = False
    delay_sec: float = 3.0
    max_followups: int = 3
    markers: tuple[str, ...] = DEFAULT_MARKERS

    @classmethod
    def from_dict(cls, data: Mapping[str, object] | None) -> "FollowupConfig":
        data = data or {}
        markers = data.get("markers")
        return cls(
            enabled=bool(data.get("enabled", False)),
            delay_sec=float(data.get("delay_sec", 3.0)),
            max_followups=int(data.get("max_followups", 3)),
            markers=tuple(markers) if markers else DEFAULT_MARKERS,
        )


@dataclass
class FollowupState:
    """Tracks how many consecutive follow-ups have fired since the user spoke."""

    markers: tuple[str, ...] = DEFAULT_MARKERS
    max_followups: int = 3
    count: int = field(default=0)

    def reset(self) -> None:
        """User responded (or context changed): start the cadence over."""
        self.count = 0

    def can_continue(self) -> bool:
        return self.count < self.max_followups

    def next_marker(self) -> str:
        marker = self.markers[self.count % len(self.markers)]
        self.count += 1
        return marker
