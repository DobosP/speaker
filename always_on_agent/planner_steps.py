"""Typed model-specific step seam for the bounded ReAct controller.

The normal planner remains a plain ``TOOL ...`` / ``FINAL ...`` text protocol.
Models with a verified native tool grammar may instead supply one
``PlannerStepBackend``.  The backend is deliberately *not* an executor: tool
allowlisting, cancellation races, registry invocation, step limits, and
untrusted-result handling stay in :mod:`always_on_agent.react`.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from typing import Callable, Optional, Protocol, Sequence


@dataclass(frozen=True)
class PlannerTool:
    """One read-only capability offered to a native planning model."""

    name: str
    description: str


@dataclass(frozen=True)
class PlannerCall:
    """The controller's current single-string capability contract."""

    name: str
    query: str


@dataclass(frozen=True)
class PlannerExchange:
    """Canonical call/result history for one bounded planner run."""

    call: PlannerCall
    result: str
    ok: bool
    untrusted: bool = False


@dataclass(frozen=True)
class PlannerStep:
    """Exactly one parsed native planner decision.

    ``call`` and ``final`` are mutually exclusive.  An empty ``final`` is valid
    and asks the controller to run its ordinary final-synthesis call.  Invalid,
    truncated, or ambiguous provider output is represented by ``malformed``;
    raw provider text is intentionally never carried toward TTS.
    """

    call: PlannerCall | None = None
    final: str | None = None
    malformed: bool = False

    def __post_init__(self) -> None:
        selected = int(self.call is not None) + int(self.final is not None)
        if self.malformed:
            if selected:
                raise ValueError("a malformed planner step cannot be actionable")
        elif selected != 1:
            raise ValueError("a planner step needs exactly one call or final")


class PlannerStepBackend(Protocol):
    """Optional, model-specific planning protocol.

    Implementations may call a provider synchronously so they can preserve a
    provider-specific token boundary and return only a fully parsed decision.
    ``first_token_hook`` must be fired on the provider's first native output
    chunk.  The controller checks ``cancel`` before and after this call; the
    provider must also use it for in-flight cancellation.
    """

    name: str

    def next_step(
        self,
        *,
        query: str,
        recent: str,
        tools: Sequence[PlannerTool],
        exchanges: Sequence[PlannerExchange],
        reminder: bool,
        cancel: Optional[Event],
        first_token_hook: Optional[Callable[[], None]],
    ) -> PlannerStep: ...

    def validate_final(self, text: str) -> str | None:
        """Return safe spoken text, or ``None`` for protocol residue."""
        ...
