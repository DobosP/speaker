"""Pure, device-neutral ownership of one pre-task user turn.

Recognizer endpoints are evidence, not conversational commits. This module
owns the small state machine between capture/ASR evidence and the existing
post-commit SessionActor. It deliberately has no clocks, threads, I/O, model
calls, or callbacks: callers provide explicit boundary evidence and perform
any waiting outside the state machine.

Transcript text is retained only inside the active turn and the explicit
LogicalTurnCommit result. Snapshots and transition records are transcript-free
so they are safe for diagnostics.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum


class LogicalTurnRejected(RuntimeError):
    """Stale, duplicate, malformed, or out-of-order evidence was rejected."""


class LogicalTurnPhase(str, Enum):
    COLLECTING = "collecting"
    INFERENCE_READY = "inference_ready"


class LogicalTurnTransitionKind(str, Enum):
    STARTED = "started"
    INFERENCE_READY = "inference_ready"
    COMMITTED = "committed"
    ABORTED = "aborted"


class LogicalTurnBoundary(str, Enum):
    """Why a caller judged the accumulated turn ready to progress."""

    NATIVE_FINAL = "native_final"
    SEMANTIC_COMPLETE = "semantic_complete"
    SILENCE_LIMIT = "silence_limit"
    HARD_LIMIT = "hard_limit"


class LogicalTurnAbortReason(str, Enum):
    TRANSCRIPT_ABORT = "transcript_abort"
    CONTROL = "control"
    STOP = "stop"
    SHUTDOWN = "shutdown"


def _non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _positive_int(name: str, value: int) -> int:
    result = _non_negative_int(name, value)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True)
class LogicalTurnTransition:
    """Transcript-free lifecycle evidence safe for logs and replay receipts."""

    kind: LogicalTurnTransitionKind
    turn_id: int
    sequence: int
    native_epoch_count: int
    native_final_count: int
    evidence_sequence: int
    readiness_token: int | None
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, LogicalTurnTransitionKind):
            raise TypeError("kind must be a LogicalTurnTransitionKind")
        _positive_int("turn_id", self.turn_id)
        _positive_int("sequence", self.sequence)
        _positive_int("native_epoch_count", self.native_epoch_count)
        _non_negative_int("native_final_count", self.native_final_count)
        _non_negative_int("evidence_sequence", self.evidence_sequence)
        if self.readiness_token is not None:
            _positive_int("readiness_token", self.readiness_token)
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be a non-empty string")
        ready_terminal = self.kind in {
            LogicalTurnTransitionKind.INFERENCE_READY,
            LogicalTurnTransitionKind.COMMITTED,
        }
        if ready_terminal != (self.readiness_token is not None):
            raise ValueError("readiness token does not match transition kind")


@dataclass(frozen=True)
class LogicalTurnCommit:
    """The only API that releases accumulated user text downstream."""

    transition: LogicalTurnTransition
    text: str = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.transition, LogicalTurnTransition):
            raise TypeError("transition must be a LogicalTurnTransition")
        if self.transition.kind is not LogicalTurnTransitionKind.COMMITTED:
            raise ValueError("commit requires a committed transition")
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("commit text must be non-empty")

    @property
    def turn_id(self) -> int:
        return self.transition.turn_id


@dataclass(frozen=True)
class LogicalTurnSnapshot:
    """Small transcript-free view of coordinator state."""

    active_turn_id: int | None
    phase: LogicalTurnPhase | None
    native_epoch_count: int
    native_final_count: int
    evidence_sequence: int | None
    readiness_token: int | None
    transition_sequence: int
    last_terminal_turn_id: int


@dataclass
class _NativeEpoch:
    final_sequence: int = -1
    text: str | None = field(default=None, repr=False)


@dataclass
class _ActiveTurn:
    turn_id: int
    phase: LogicalTurnPhase
    epochs: OrderedDict[int, _NativeEpoch]
    evidence_sequence: int
    native_final_count: int = 0
    readiness_token: int | None = None
    readiness_reason: LogicalTurnBoundary | None = None


class LogicalTurnCoordinator:
    """Exactly-one-terminal owner for pre-task user-turn lifecycle.

    Native epochs may be revised or followed by later reset epochs, but they
    remain part of one logical turn until this coordinator emits COMMITTED or
    ABORTED. A terminal clears the active turn; any later evidence carrying its
    old turn_id fails closed. This object is intentionally not thread-safe:
    one runtime owner must serialize every call.
    """

    def __init__(
        self,
        *,
        max_native_epochs: int = 64,
        max_native_finals: int = 256,
        max_text_chars: int = 32_768,
    ) -> None:
        self._max_native_epochs = _positive_int("max_native_epochs", max_native_epochs)
        self._max_native_finals = _positive_int("max_native_finals", max_native_finals)
        self._max_text_chars = _positive_int("max_text_chars", max_text_chars)
        self._next_turn_id = 0
        self._transition_sequence = 0
        self._last_terminal_turn_id = 0
        self._active: _ActiveTurn | None = None

    @property
    def active_turn_id(self) -> int | None:
        return self._active.turn_id if self._active is not None else None

    def snapshot(self) -> LogicalTurnSnapshot:
        active = self._active
        return LogicalTurnSnapshot(
            active_turn_id=active.turn_id if active is not None else None,
            phase=active.phase if active is not None else None,
            native_epoch_count=len(active.epochs) if active is not None else 0,
            native_final_count=(active.native_final_count if active is not None else 0),
            evidence_sequence=(
                active.evidence_sequence if active is not None else None
            ),
            readiness_token=(active.readiness_token if active is not None else None),
            transition_sequence=self._transition_sequence,
            last_terminal_turn_id=self._last_terminal_turn_id,
        )

    def start(
        self,
        *,
        native_epoch: int,
        evidence_sequence: int,
    ) -> LogicalTurnTransition:
        epoch = _non_negative_int("native_epoch", native_epoch)
        evidence = _non_negative_int("evidence_sequence", evidence_sequence)
        if self._active is not None:
            raise LogicalTurnRejected("a logical turn is already active")
        self._next_turn_id += 1
        active = _ActiveTurn(
            turn_id=self._next_turn_id,
            phase=LogicalTurnPhase.COLLECTING,
            epochs=OrderedDict(((epoch, _NativeEpoch()),)),
            evidence_sequence=evidence,
        )
        self._active = active
        return self._transition(
            LogicalTurnTransitionKind.STARTED,
            active,
            reason="speech",
        )

    def resume_speech(
        self,
        turn_id: int,
        *,
        native_epoch: int,
        evidence_sequence: int,
    ) -> None:
        """Return an inference-ready turn to collection without uncommitting it."""

        active = self._require_active(turn_id)
        epoch = _non_negative_int("native_epoch", native_epoch)
        evidence = self._fresh_evidence(active, evidence_sequence)
        self._admit_epoch(active, epoch)
        active.phase = LogicalTurnPhase.COLLECTING
        active.evidence_sequence = evidence
        active.readiness_token = None
        active.readiness_reason = None

    def observe_final(
        self,
        turn_id: int,
        *,
        native_epoch: int,
        native_sequence: int,
        evidence_sequence: int,
        text: str,
    ) -> None:
        """Admit one native final/revision without making it public.

        A higher sequence replaces the text for the same native epoch. A later
        native epoch contributes another segment. This avoids appending a
        recognizer correction while still accumulating text across native
        stream resets.
        """

        active = self._require_active(turn_id)
        epoch = _non_negative_int("native_epoch", native_epoch)
        sequence = _non_negative_int("native_sequence", native_sequence)
        evidence = self._fresh_evidence(active, evidence_sequence)
        if active.phase is LogicalTurnPhase.INFERENCE_READY:
            raise LogicalTurnRejected(
                "final after readiness requires explicit resumed speech"
            )
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        retained = text.strip()
        if not retained:
            raise LogicalTurnRejected("empty native final")
        if epoch != next(reversed(active.epochs)):
            raise LogicalTurnRejected("native final is not for the current epoch")
        native = active.epochs[epoch]
        if sequence <= native.final_sequence:
            raise LogicalTurnRejected("stale or duplicate native final")
        if active.native_final_count >= self._max_native_finals:
            raise LogicalTurnRejected("native final limit reached")
        retained_parts = [
            retained if item_epoch == epoch else item.text
            for item_epoch, item in active.epochs.items()
            if (retained if item_epoch == epoch else item.text)
        ]
        retained_chars = len(" ".join(retained_parts))
        if retained_chars > self._max_text_chars:
            raise LogicalTurnRejected("logical turn text limit reached")
        native.final_sequence = sequence
        native.text = retained
        active.native_final_count += 1
        active.evidence_sequence = evidence

    def mark_inference_ready(
        self,
        turn_id: int,
        *,
        reason: LogicalTurnBoundary,
        expected_evidence_sequence: int,
    ) -> LogicalTurnTransition:
        active = self._require_active(turn_id)
        if not isinstance(reason, LogicalTurnBoundary):
            raise TypeError("reason must be a LogicalTurnBoundary")
        expected = _non_negative_int(
            "expected_evidence_sequence",
            expected_evidence_sequence,
        )
        if expected != active.evidence_sequence:
            raise LogicalTurnRejected("readiness evidence is stale")
        if active.phase is LogicalTurnPhase.INFERENCE_READY:
            raise LogicalTurnRejected("turn is already inference-ready")
        if not any(epoch.text for epoch in active.epochs.values()):
            raise LogicalTurnRejected("inference-ready requires a native final")
        active.phase = LogicalTurnPhase.INFERENCE_READY
        transition = self._transition(
            LogicalTurnTransitionKind.INFERENCE_READY,
            active,
            reason=reason.value,
        )
        active.readiness_token = transition.readiness_token
        active.readiness_reason = reason
        return transition

    def commit(
        self,
        turn_id: int,
        *,
        readiness_token: int,
    ) -> LogicalTurnCommit:
        active = self._require_active(turn_id)
        token = _positive_int("readiness_token", readiness_token)
        if active.phase is not LogicalTurnPhase.INFERENCE_READY:
            raise LogicalTurnRejected("commit requires inference-ready state")
        if token != active.readiness_token:
            raise LogicalTurnRejected("stale readiness token")
        if active.readiness_reason is None:
            raise LogicalTurnRejected("readiness reason is missing")
        text = " ".join(epoch.text for epoch in active.epochs.values() if epoch.text)
        if not text:
            raise LogicalTurnRejected("commit requires retained text")
        transition = self._terminal(
            LogicalTurnTransitionKind.COMMITTED,
            active,
            reason=active.readiness_reason.value,
            readiness_token=token,
        )
        return LogicalTurnCommit(transition=transition, text=text)

    def abort(
        self,
        turn_id: int,
        *,
        reason: LogicalTurnAbortReason,
        evidence_sequence: int,
    ) -> LogicalTurnTransition:
        active = self._require_active(turn_id)
        if not isinstance(reason, LogicalTurnAbortReason):
            raise TypeError("reason must be a LogicalTurnAbortReason")
        active.evidence_sequence = self._fresh_evidence(
            active,
            evidence_sequence,
        )
        return self._terminal(
            LogicalTurnTransitionKind.ABORTED,
            active,
            reason=reason.value,
        )

    def _require_active(self, turn_id: int) -> _ActiveTurn:
        candidate = _non_negative_int("turn_id", turn_id)
        active = self._active
        if active is None:
            raise LogicalTurnRejected("no logical turn is active")
        if candidate != active.turn_id:
            raise LogicalTurnRejected("stale or foreign logical turn")
        return active

    def _admit_epoch(
        self,
        active: _ActiveTurn,
        native_epoch: int,
    ) -> _NativeEpoch:
        last_epoch = next(reversed(active.epochs))
        if native_epoch < last_epoch:
            raise LogicalTurnRejected("native epochs must be monotonic")
        existing = active.epochs.get(native_epoch)
        if existing is not None:
            return existing
        if native_epoch < next(reversed(active.epochs)):
            raise LogicalTurnRejected("native epochs must be monotonic")
        if len(active.epochs) >= self._max_native_epochs:
            raise LogicalTurnRejected("native epoch limit reached")
        native = _NativeEpoch()
        active.epochs[native_epoch] = native
        return native

    @staticmethod
    def _fresh_evidence(active: _ActiveTurn, value: int) -> int:
        sequence = _non_negative_int("evidence_sequence", value)
        if sequence <= active.evidence_sequence:
            raise LogicalTurnRejected("stale or duplicate logical-turn evidence")
        return sequence

    def _transition(
        self,
        kind: LogicalTurnTransitionKind,
        active: _ActiveTurn,
        *,
        reason: str,
        readiness_token: int | None = None,
    ) -> LogicalTurnTransition:
        self._transition_sequence += 1
        token = (
            self._transition_sequence
            if kind is LogicalTurnTransitionKind.INFERENCE_READY
            else readiness_token
        )
        return LogicalTurnTransition(
            kind=kind,
            turn_id=active.turn_id,
            sequence=self._transition_sequence,
            native_epoch_count=len(active.epochs),
            native_final_count=active.native_final_count,
            evidence_sequence=active.evidence_sequence,
            readiness_token=token,
            reason=reason,
        )

    def _terminal(
        self,
        kind: LogicalTurnTransitionKind,
        active: _ActiveTurn,
        *,
        reason: str,
        readiness_token: int | None = None,
    ) -> LogicalTurnTransition:
        transition = self._transition(
            kind,
            active,
            reason=reason,
            readiness_token=readiness_token,
        )
        self._last_terminal_turn_id = active.turn_id
        for epoch in active.epochs.values():
            epoch.text = None
        active.epochs.clear()
        active.readiness_token = None
        active.readiness_reason = None
        self._active = None
        return transition


__all__ = [
    "LogicalTurnAbortReason",
    "LogicalTurnBoundary",
    "LogicalTurnCommit",
    "LogicalTurnCoordinator",
    "LogicalTurnPhase",
    "LogicalTurnRejected",
    "LogicalTurnSnapshot",
    "LogicalTurnTransition",
    "LogicalTurnTransitionKind",
]
