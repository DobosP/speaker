"""Early stale-event rejection for typed acoustic transcript updates."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from always_on_agent.acoustic import AcousticLineage


@dataclass(frozen=True)
class _TurnState:
    revision: int
    closed: bool


class AcousticRevisionGate:
    """Accept strictly increasing revisions and exactly one final per turn.

    The gate is intentionally independent from owner verification and runtime
    response generations. It rejects stale recognizer output before that output
    can allocate preprocessing work; it never upgrades trust.
    """

    def __init__(self, *, max_turns: int = 1024) -> None:
        self._max_turns = max(1, int(max_turns))
        self._turns: OrderedDict[tuple[str, int, int, str], _TurnState] = OrderedDict()
        self._stream_high_water: OrderedDict[str, int] = OrderedDict()

    def accept(
        self,
        acoustic: AcousticLineage | None,
        revision: int,
        *,
        final: bool,
    ) -> bool:
        if acoustic is None:
            return True
        if isinstance(revision, bool) or not isinstance(revision, int):
            return False
        if revision < 0:
            return False
        pending: list[tuple[tuple[str, int, int, str], _TurnState]] = []
        pending_high_water: dict[str, int] = {}
        for span in acoustic.spans:
            key = span.key
            if span.turn_index is not None:
                high_water = pending_high_water.get(
                    span.stream_id,
                    self._stream_high_water.get(span.stream_id, -1),
                )
                if span.turn_index < high_water:
                    return False
                pending_high_water[span.stream_id] = max(high_water, span.turn_index)
            previous = self._turns.get(key)
            if previous is not None and (
                previous.closed or revision <= previous.revision
            ):
                return False
            pending.append((key, _TurnState(revision, bool(final))))
        for key, state in pending:
            self._turns[key] = state
            self._turns.move_to_end(key)
        for stream_id, high_water in pending_high_water.items():
            self._stream_high_water[stream_id] = high_water
            self._stream_high_water.move_to_end(stream_id)
        while len(self._turns) > self._max_turns:
            self._turns.popitem(last=False)
        while len(self._stream_high_water) > self._max_turns:
            self._stream_high_water.popitem(last=False)
        return True

    def clear(self) -> None:
        self._turns.clear()
        self._stream_high_water.clear()
