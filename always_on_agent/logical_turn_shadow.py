"""Transcript-free evidence for Sherpa native logical-turn composition.

This diagnostic-only observer feeds one terminal result from each native ASR
stream through LogicalTurnCoordinator, compares the in-memory commit with
Sherpa's existing raw composed final, and immediately reduces the comparison
to closed counters.

No report or snapshot contains transcript text, hashes, lengths, acoustic
identities, paths, exception strings, or per-event rows. The scope ends before
final-model selection, dispatcher merging, supervisor continuation, and task
commit.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, fields
from enum import Enum
from threading import Lock
from types import MappingProxyType
from typing import Iterable, Mapping

from core.engine import TranscriptAbortReason

from .acoustic import AcousticLineage
from .logical_turn import (
    LogicalTurnAbortReason,
    LogicalTurnBoundary,
    LogicalTurnCoordinator,
    LogicalTurnRejected,
)


class LogicalTurnShadowScope(str, Enum):
    """Closed name for the only supported comparison seam."""

    SHERPA_RAW_COMPOSITION = "sherpa_raw_composition_v1"


class LogicalTurnTerminalLineageScope(str, Enum):
    """Closed name for raw-commit to downstream-terminal evidence."""

    SHERPA_SELECTED_TERMINAL = "sherpa_selected_terminal_lineage_v1"


_CAPABILITIES: Mapping[str, bool] = MappingProxyType(
    {
        "native_epoch_composition_parity": True,
        "raw_composed_final_text_parity": True,
        "abort_terminal_parity": False,
        "selected_final_parity": False,
        "async_final_selection_parity": False,
        "final_dispatcher_parity": False,
        "supervisor_continuation_parity": False,
        "runtime_stop_shutdown_parity": False,
        "production_configuration_coverage": False,
        "runtime_authority": False,
        "live_evidence": False,
    }
)

_TERMINAL_CAPABILITIES: Mapping[str, bool] = MappingProxyType(
    {
        "raw_commit_terminal_lineage": True,
        "selected_final_lineage": True,
        "typed_abort_lineage": True,
        "typed_abort_reason_lineage": True,
        "selected_final_text_parity": False,
        "async_final_selection_parity": False,
        "final_dispatcher_parity": False,
        "supervisor_continuation_parity": False,
        "runtime_stop_shutdown_parity": False,
        "production_configuration_coverage": False,
        "runtime_authority": False,
        "live_evidence": False,
    }
)

_TRANSCRIPT_ABORT_REASONS = tuple(TranscriptAbortReason)
_TerminalLineageKey = tuple[tuple[tuple[str, int, int, str], ...], int]
_MAX_TERMINAL_COMMITS = 256
_MAX_TERMINAL_SPANS = 64


def _non_negative_int(name: str, value: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _positive_int(name: str, value: int) -> int:
    result = _non_negative_int(name, value)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True, slots=True)
class LogicalTurnShadowSnapshot:
    """One transcript-free observer snapshot safe for aggregate reports."""

    scope: LogicalTurnShadowScope
    closed: bool
    started: int
    native_epochs: int
    native_finals: int
    empty_native_epochs: int
    held_native_finals: int
    readiness: int
    committed: int
    multi_epoch_commits: int
    aborted: int
    active: bool
    legacy_terminals: int
    paired_terminals: int
    extra_legacy_terminals: int
    missing_legacy_terminals: int
    composition_matches: int
    composition_mismatches: int
    boundary_native_final: int
    boundary_semantic_complete: int
    boundary_silence_limit: int
    boundary_hard_limit: int
    abort_transcript: int
    abort_control: int
    abort_stop: int
    abort_shutdown: int
    idle_aborts: int
    coordinator_rejections: int
    bounds_overflows: int
    internal_errors: int
    late_observations: int

    def __post_init__(self) -> None:
        if type(self.scope) is not LogicalTurnShadowScope:
            raise TypeError("scope must be a LogicalTurnShadowScope")
        if type(self.closed) is not bool or type(self.active) is not bool:
            raise TypeError("closed and active must be booleans")
        for item in fields(self):
            if item.name in {"scope", "closed", "active"}:
                continue
            _non_negative_int(item.name, getattr(self, item.name))
        if self.closed and self.active:
            raise ValueError("closed shadow cannot retain an active turn")
        if self.started != self.committed + self.aborted + int(self.active):
            raise ValueError("logical-turn lifecycle totals do not reconcile")
        if self.native_epochs != self.native_finals + self.empty_native_epochs:
            raise ValueError("native epoch totals do not reconcile")
        if self.native_epochs and not self.started:
            raise ValueError("native epochs require a started turn")
        if self.native_epochs and not self.native_finals:
            raise ValueError("native epochs require a non-empty native final")
        if self.empty_native_epochs and not self.held_native_finals:
            raise ValueError("empty native epochs require held native evidence")
        if self.held_native_finals > self.native_finals:
            raise ValueError("held native finals exceed native finals")
        unheld_native_finals = self.native_finals - self.held_native_finals
        if unheld_native_finals > self.started:
            raise ValueError("unheld native finals exceed started turns")
        if unheld_native_finals < self.committed - self.multi_epoch_commits:
            raise ValueError("single-epoch commits exceed unheld native finals")
        if self.committed > self.native_finals:
            raise ValueError("commits exceed native finals")
        if self.committed > self.readiness:
            raise ValueError("commits exceed readiness transitions")
        if self.readiness > self.legacy_terminals:
            raise ValueError("readiness transitions exceed legacy terminals")
        if self.readiness > self.native_finals:
            raise ValueError("readiness transitions exceed native finals")
        if self.held_native_finals + self.readiness > self.native_epochs:
            raise ValueError("held and ready transitions exceed native epochs")
        if self.readiness - self.committed > self.abort_control:
            raise ValueError("failed readiness transitions exceed control aborts")
        if self.multi_epoch_commits > self.committed:
            raise ValueError("multi-epoch commits exceed commits")
        if self.multi_epoch_commits > self.held_native_finals:
            raise ValueError("multi-epoch commits exceed held native finals")
        if 2 * self.multi_epoch_commits > self.native_epochs:
            raise ValueError("multi-epoch commits exceed native epoch evidence")
        if self.readiness != (
            self.boundary_native_final
            + self.boundary_semantic_complete
            + self.boundary_silence_limit
            + self.boundary_hard_limit
        ):
            raise ValueError("readiness boundary totals do not reconcile")
        if self.aborted != (
            self.abort_transcript
            + self.abort_control
            + self.abort_stop
            + self.abort_shutdown
        ):
            raise ValueError("abort reason totals do not reconcile")
        if self.paired_terminals != self.committed:
            raise ValueError("paired terminals must equal coordinator commits")
        if self.legacy_terminals != (
            self.paired_terminals + self.extra_legacy_terminals
        ):
            raise ValueError("legacy terminal totals do not reconcile")
        if self.paired_terminals != (
            self.composition_matches + self.composition_mismatches
        ):
            raise ValueError("composition parity totals do not reconcile")
        if self.missing_legacy_terminals > self.abort_shutdown:
            raise ValueError("missing terminals exceed shutdown aborts")
        failure_slack = self.coordinator_rejections + self.internal_errors
        if self.started > self.native_finals + failure_slack:
            raise ValueError("started turns exceed native finals plus failures")
        if self.started + self.multi_epoch_commits > (
            self.native_epochs + failure_slack
        ):
            raise ValueError("started and multi-epoch turns exceed epoch evidence")


@dataclass(frozen=True, slots=True)
class LogicalTurnTerminalLineageSnapshot:
    """Closed counters for exact raw-commit to typed-terminal correlation."""

    scope: LogicalTurnTerminalLineageScope
    closed: bool
    raw_commits: int
    claimed_raw_commits: int
    bound_raw_commits: int
    selected_finals: int
    typed_aborts: int
    matched_selected_finals: int
    matched_typed_aborts: int
    unscoped_selected_finals: int
    unscoped_typed_aborts: int
    duplicate_selected_finals: int
    duplicate_typed_aborts: int
    missing_downstream_terminals: int
    unbound_raw_commits: int
    binding_rejections: int
    terminal_rejections: int
    internal_errors: int
    late_observations: int
    abort_reason_counts: tuple[int, ...]
    matched_abort_reason_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.scope) is not LogicalTurnTerminalLineageScope:
            raise TypeError("scope must be a LogicalTurnTerminalLineageScope")
        if type(self.closed) is not bool:
            raise TypeError("closed must be a boolean")
        for item in fields(self):
            if item.name in {
                "scope",
                "closed",
                "abort_reason_counts",
                "matched_abort_reason_counts",
            }:
                continue
            _non_negative_int(item.name, getattr(self, item.name))
        for name, counts in (
            ("abort_reason_counts", self.abort_reason_counts),
            ("matched_abort_reason_counts", self.matched_abort_reason_counts),
        ):
            if type(counts) is not tuple or len(counts) != len(
                _TRANSCRIPT_ABORT_REASONS
            ):
                raise TypeError(f"{name} must cover every typed abort reason")
            for index, count in enumerate(counts):
                _non_negative_int(f"{name}[{index}]", count)
        matched = self.matched_selected_finals + self.matched_typed_aborts
        if self.raw_commits != matched + self.missing_downstream_terminals:
            raise ValueError("raw commit terminal totals do not reconcile")
        if self.claimed_raw_commits > self.raw_commits:
            raise ValueError("claimed raw commits exceed raw commits")
        if self.bound_raw_commits > self.claimed_raw_commits:
            raise ValueError("bound raw commits exceed claimed raw commits")
        if self.unbound_raw_commits > self.missing_downstream_terminals:
            raise ValueError("unbound raw commits exceed missing terminals")
        if self.bound_raw_commits != matched + (
            self.missing_downstream_terminals - self.unbound_raw_commits
        ):
            raise ValueError("bound raw commit totals do not reconcile")
        if self.selected_finals != (
            self.matched_selected_finals
            + self.unscoped_selected_finals
            + self.duplicate_selected_finals
        ):
            raise ValueError("selected final totals do not reconcile")
        if self.typed_aborts != (
            self.matched_typed_aborts
            + self.unscoped_typed_aborts
            + self.duplicate_typed_aborts
        ):
            raise ValueError("typed abort totals do not reconcile")
        if sum(self.abort_reason_counts) != self.typed_aborts:
            raise ValueError("typed abort reason totals do not reconcile")
        if sum(self.matched_abort_reason_counts) != self.matched_typed_aborts:
            raise ValueError("matched abort reason totals do not reconcile")
        if any(
            matched_count > observed_count
            for matched_count, observed_count in zip(
                self.matched_abort_reason_counts,
                self.abort_reason_counts,
                strict=True,
            )
        ):
            raise ValueError("matched abort reasons exceed observed aborts")


class LogicalTurnCompositionShadow:
    """Serialized, fail-contained raw-composition observer.

    The capture seam supplies exactly one terminal result per one-based native
    epoch, always with native sequence zero. A held result proves that Sherpa
    reset the stream and permits exactly one successor epoch. An unheld result
    makes the current epoch terminal-ready and requires an immediate legacy
    comparison. Empty successor epochs carry the same disposition explicitly.
    """

    def __init__(
        self,
        *,
        max_native_epochs: int = 64,
        max_native_finals: int = 256,
        max_text_chars: int = 32_768,
        terminal_lineage: bool = False,
    ) -> None:
        if type(terminal_lineage) is not bool:
            raise TypeError("terminal_lineage must be a boolean")
        self._max_native_epochs = _positive_int("max_native_epochs", max_native_epochs)
        self._max_native_finals = _positive_int("max_native_finals", max_native_finals)
        self._max_text_chars = _positive_int("max_text_chars", max_text_chars)
        self._lock = Lock()
        self._coordinator = self._new_coordinator()
        self._evidence_sequence = 0
        self._current_epoch: int | None = None
        self._successor_allowed = False
        self._terminal_ready = False
        self._active_native_epochs = 0
        self._active_native_finals = 0
        self._active_nonempty_epochs = 0
        self._active_retained_chars = 0
        self._closed = False
        self._started = 0
        self._native_epochs = 0
        self._native_finals = 0
        self._empty_native_epochs = 0
        self._held_native_finals = 0
        self._readiness = 0
        self._committed = 0
        self._multi_epoch_commits = 0
        self._aborted = 0
        self._legacy_terminals = 0
        self._paired_terminals = 0
        self._extra_legacy_terminals = 0
        self._missing_legacy_terminals = 0
        self._composition_matches = 0
        self._composition_mismatches = 0
        self._boundary_counts = {boundary: 0 for boundary in LogicalTurnBoundary}
        self._abort_counts = {reason: 0 for reason in LogicalTurnAbortReason}
        self._idle_aborts = 0
        self._coordinator_rejections = 0
        self._bounds_overflows = 0
        self._internal_errors = 0
        self._late_observations = 0
        self._terminal_lineage = terminal_lineage
        self._terminal_raw_commits = 0
        self._terminal_sequence = 0
        self._terminal_untracked_commits = 0
        self._terminal_claimable: deque[int] = deque()
        self._terminal_claimed: set[int] = set()
        self._terminal_claimed_raw_commits = 0
        self._terminal_pending: dict[
            int,
            tuple[tuple[tuple[str, int, int, str], ...], int] | None,
        ] = {}
        self._terminal_by_lineage: dict[
            tuple[tuple[tuple[str, int, int, str], ...], int],
            int,
        ] = {}
        self._terminal_completed_lineages: set[
            tuple[tuple[tuple[str, int, int, str], ...], int]
        ] = set()
        self._terminal_selected_finals = 0
        self._terminal_typed_aborts = 0
        self._terminal_matched_selected_finals = 0
        self._terminal_matched_typed_aborts = 0
        self._terminal_unscoped_selected_finals = 0
        self._terminal_unscoped_typed_aborts = 0
        self._terminal_duplicate_selected_finals = 0
        self._terminal_duplicate_typed_aborts = 0
        self._terminal_bound_raw_commits = 0
        self._terminal_binding_rejections = 0
        self._terminal_rejections = 0
        self._terminal_internal_errors = 0
        self._terminal_late_observations = 0
        self._terminal_abort_counts = {
            reason: 0 for reason in _TRANSCRIPT_ABORT_REASONS
        }
        self._terminal_matched_abort_counts = {
            reason: 0 for reason in _TRANSCRIPT_ABORT_REASONS
        }
        self._terminal_missing_at_close: int | None = None
        self._terminal_unbound_at_close: int | None = None

    def _new_coordinator(self) -> LogicalTurnCoordinator:
        return LogicalTurnCoordinator(
            max_native_epochs=self._max_native_epochs,
            max_native_finals=self._max_native_finals,
            max_text_chars=self._max_text_chars,
        )

    def _next_evidence(self) -> int:
        self._evidence_sequence += 1
        return self._evidence_sequence

    @property
    def terminal_lineage_enabled(self) -> bool:
        """Whether this observer may collect downstream terminal evidence."""

        return self._terminal_lineage

    @staticmethod
    def _terminal_lineage_key(
        acoustic: AcousticLineage,
        revision: int,
    ) -> _TerminalLineageKey:
        if not isinstance(acoustic, AcousticLineage):
            raise TypeError("acoustic must be an AcousticLineage")
        selected_revision = _non_negative_int("revision", revision)
        if len(acoustic.spans) > _MAX_TERMINAL_SPANS:
            raise ValueError("acoustic lineage has too many spans")
        keys: list[tuple[str, int, int, str]] = []
        for span in acoustic.spans:
            if (
                type(span.stream_id) is not str
                or type(span.capture_epoch) is not int
                or type(span.capture_generation) is not int
                or type(span.utterance_id) is not str
            ):
                raise TypeError("acoustic lineage keys must use built-in scalars")
            keys.append(
                (
                    span.stream_id,
                    span.capture_epoch,
                    span.capture_generation,
                    span.utterance_id,
                )
            )
        return (tuple(keys), selected_revision)

    def _mint_terminal_token(self) -> None:
        if not self._terminal_lineage:
            return
        self._terminal_raw_commits += 1
        if self._terminal_sequence >= _MAX_TERMINAL_COMMITS:
            self._terminal_untracked_commits += 1
            self._terminal_rejections += 1
            return
        self._terminal_sequence += 1
        token = self._terminal_sequence
        self._terminal_pending[token] = None
        self._terminal_claimable.append(token)

    def claim_terminal_token(self) -> int | None:
        """Claim the next opaque raw-commit token for acoustic binding."""

        with self._lock:
            if not self._terminal_lineage:
                return None
            if self._closed:
                self._terminal_late_observations += 1
                return None
            if not self._terminal_claimable:
                self._terminal_rejections += 1
                return None
            token = self._terminal_claimable.popleft()
            self._terminal_claimed.add(token)
            self._terminal_claimed_raw_commits += 1
            return token

    def bind_terminal_lineage(
        self,
        token: int,
        acoustic: AcousticLineage,
        revision: int,
    ) -> bool:
        """Bind one opaque raw commit to its stable in-memory acoustic key."""

        try:
            selected_token = _positive_int("terminal token", token)
            lineage_key = self._terminal_lineage_key(acoustic, revision)
        except (TypeError, ValueError):
            with self._lock:
                if self._terminal_lineage:
                    self._terminal_binding_rejections += 1
            return False
        with self._lock:
            if not self._terminal_lineage:
                return False
            if self._closed:
                self._terminal_late_observations += 1
                return False
            if (
                selected_token not in self._terminal_claimed
                or selected_token not in self._terminal_pending
                or self._terminal_pending[selected_token] is not None
                or lineage_key in self._terminal_by_lineage
                or lineage_key in self._terminal_completed_lineages
            ):
                self._terminal_binding_rejections += 1
                return False
            self._terminal_pending[selected_token] = lineage_key
            self._terminal_by_lineage[lineage_key] = selected_token
            self._terminal_bound_raw_commits += 1
            return True

    def _observe_downstream_terminal(
        self,
        acoustic: AcousticLineage,
        revision: int,
        *,
        abort_reason: TranscriptAbortReason | None,
    ) -> bool:
        try:
            lineage_key = self._terminal_lineage_key(acoustic, revision)
            if abort_reason is not None and type(abort_reason) is not TranscriptAbortReason:
                raise TypeError("abort_reason must be a TranscriptAbortReason")
        except (TypeError, ValueError):
            with self._lock:
                if self._terminal_lineage:
                    self._terminal_rejections += 1
            return False
        with self._lock:
            if not self._terminal_lineage:
                return False
            if self._closed:
                self._terminal_late_observations += 1
                return False
            is_abort = abort_reason is not None
            if is_abort:
                self._terminal_typed_aborts += 1
                self._terminal_abort_counts[abort_reason] += 1
            else:
                self._terminal_selected_finals += 1
            token = self._terminal_by_lineage.pop(lineage_key, None)
            if token is None:
                if lineage_key in self._terminal_completed_lineages:
                    if is_abort:
                        self._terminal_duplicate_typed_aborts += 1
                    else:
                        self._terminal_duplicate_selected_finals += 1
                elif is_abort:
                    self._terminal_unscoped_typed_aborts += 1
                else:
                    self._terminal_unscoped_selected_finals += 1
                return True
            self._terminal_pending.pop(token)
            self._terminal_completed_lineages.add(lineage_key)
            if is_abort:
                self._terminal_matched_typed_aborts += 1
                self._terminal_matched_abort_counts[abort_reason] += 1
            else:
                self._terminal_matched_selected_finals += 1
            return True

    def observe_selected_final(
        self,
        acoustic: AcousticLineage,
        revision: int,
    ) -> bool:
        """Reduce one selected-final lineage match to counters."""

        return self._observe_downstream_terminal(
            acoustic,
            revision,
            abort_reason=None,
        )

    def observe_typed_abort(
        self,
        acoustic: AcousticLineage,
        revision: int,
        reason: TranscriptAbortReason,
    ) -> bool:
        """Reduce one typed-abort lineage and closed reason to counters."""

        return self._observe_downstream_terminal(
            acoustic,
            revision,
            abort_reason=reason,
        )

    def _note_rejection(self, *, bounds: bool = False) -> None:
        if bounds:
            self._bounds_overflows += 1
        else:
            self._coordinator_rejections += 1

    def _reject_external(self, *, bounds: bool = False) -> bool:
        with self._lock:
            if self._closed:
                self._late_observations += 1
            else:
                self._note_rejection(bounds=bounds)
        return False

    def _clear_active_bookkeeping(self) -> None:
        self._current_epoch = None
        self._successor_allowed = False
        self._terminal_ready = False
        self._active_native_epochs = 0
        self._active_native_finals = 0
        self._active_nonempty_epochs = 0
        self._active_retained_chars = 0

    def _force_abort(self, reason: LogicalTurnAbortReason) -> None:
        turn_id = self._coordinator.active_turn_id
        if turn_id is None:
            self._clear_active_bookkeeping()
            return
        try:
            self._coordinator.abort(
                turn_id,
                reason=reason,
                evidence_sequence=self._next_evidence(),
            )
        except Exception:
            self._internal_errors += 1
            self._coordinator = self._new_coordinator()
        self._aborted += 1
        self._abort_counts[reason] += 1
        self._clear_active_bookkeeping()

    def _preflight_epoch(self, epoch: int) -> bool:
        turn_id = self._coordinator.active_turn_id
        if turn_id is None:
            return epoch == 1
        return bool(
            self._current_epoch is not None
            and self._successor_allowed
            and epoch == self._current_epoch + 1
        )

    def note_internal_error(self) -> None:
        """Record an integration failure and erase active retained text."""

        with self._lock:
            self._internal_errors += 1
            if self._terminal_lineage:
                self._terminal_internal_errors += 1
            self._force_abort(LogicalTurnAbortReason.CONTROL)

    def observe_native_final(
        self,
        *,
        native_epoch: int,
        native_sequence: int,
        text: str,
        held: bool,
    ) -> bool:
        """Observe one terminal, non-empty result from a native stream."""

        try:
            epoch = _positive_int("native_epoch", native_epoch)
            sequence = _non_negative_int("native_sequence", native_sequence)
            if sequence != 0:
                raise ValueError("native_sequence must be zero at this seam")
            if type(held) is not bool:
                raise TypeError("held must be a boolean")
            if type(text) is not str:
                raise TypeError("native final must be a built-in string")
            retained = text.strip()
            if not retained:
                raise ValueError("native final must be non-empty")
        except (TypeError, ValueError):
            return self._reject_external()

        with self._lock:
            if self._closed:
                self._late_observations += 1
                return False
            if not self._preflight_epoch(epoch):
                self._note_rejection()
                return False
            if (
                self._active_native_epochs >= self._max_native_epochs
                or self._active_native_finals >= self._max_native_finals
            ):
                self._note_rejection(bounds=True)
                return False
            projected_chars = (
                self._active_retained_chars
                + len(retained)
                + int(self._active_nonempty_epochs > 0)
            )
            if projected_chars > self._max_text_chars:
                self._note_rejection(bounds=True)
                return False
            try:
                turn_id = self._coordinator.active_turn_id
                if turn_id is None:
                    started = self._coordinator.start(
                        native_epoch=epoch,
                        evidence_sequence=self._next_evidence(),
                    )
                    turn_id = started.turn_id
                    self._started += 1
                else:
                    self._coordinator.resume_speech(
                        turn_id,
                        native_epoch=epoch,
                        evidence_sequence=self._next_evidence(),
                    )
                self._coordinator.observe_final(
                    turn_id,
                    native_epoch=epoch,
                    native_sequence=sequence,
                    evidence_sequence=self._next_evidence(),
                    text=retained,
                )
            except LogicalTurnRejected:
                self._note_rejection()
                self._force_abort(LogicalTurnAbortReason.CONTROL)
                return False
            except Exception:
                self._internal_errors += 1
                self._force_abort(LogicalTurnAbortReason.CONTROL)
                return False
            self._current_epoch = epoch
            self._successor_allowed = held
            self._terminal_ready = not held
            self._active_native_epochs += 1
            self._active_native_finals += 1
            self._active_nonempty_epochs += 1
            self._active_retained_chars = projected_chars
            self._native_epochs += 1
            self._native_finals += 1
            self._held_native_finals += int(held)
            return True

    def observe_empty_native_epoch(
        self,
        *,
        native_epoch: int,
        held: bool,
    ) -> bool:
        """Observe one empty successor stream and its reset disposition."""

        try:
            epoch = _positive_int("native_epoch", native_epoch)
            if type(held) is not bool:
                raise TypeError("held must be a boolean")
        except (TypeError, ValueError):
            return self._reject_external()
        with self._lock:
            if self._closed:
                self._late_observations += 1
                return False
            if self._coordinator.active_turn_id is None or not self._preflight_epoch(
                epoch
            ):
                self._note_rejection()
                return False
            if self._active_native_epochs >= self._max_native_epochs:
                self._note_rejection(bounds=True)
                return False
            try:
                turn_id = self._coordinator.active_turn_id
                if turn_id is None:
                    raise LogicalTurnRejected("empty epoch requires an active turn")
                self._coordinator.resume_speech(
                    turn_id,
                    native_epoch=epoch,
                    evidence_sequence=self._next_evidence(),
                )
            except LogicalTurnRejected:
                self._note_rejection()
                self._force_abort(LogicalTurnAbortReason.CONTROL)
                return False
            except Exception:
                self._internal_errors += 1
                self._force_abort(LogicalTurnAbortReason.CONTROL)
                return False
            self._current_epoch = epoch
            self._successor_allowed = held
            self._terminal_ready = not held
            self._active_native_epochs += 1
            self._native_epochs += 1
            self._empty_native_epochs += 1
            return True

    def compare_legacy_composition(
        self,
        *,
        boundary: LogicalTurnBoundary,
        legacy_text: str,
    ) -> bool:
        """Commit once and immediately reduce exact equality to counters."""

        try:
            if type(boundary) is not LogicalTurnBoundary:
                raise TypeError("boundary must be a LogicalTurnBoundary")
            if type(legacy_text) is not str:
                raise TypeError("legacy composition must be a built-in string")
            if not legacy_text.strip():
                raise ValueError("legacy composition must be non-empty")
        except (TypeError, ValueError):
            return self._reject_external()
        if len(legacy_text) > self._max_text_chars:
            return self._reject_external(bounds=True)
        with self._lock:
            if self._closed:
                self._late_observations += 1
                return False
            self._legacy_terminals += 1
            turn_id = self._coordinator.active_turn_id
            if turn_id is None or not self._terminal_ready:
                self._extra_legacy_terminals += 1
                self._note_rejection()
                if turn_id is not None:
                    self._force_abort(LogicalTurnAbortReason.CONTROL)
                return False
            try:
                coordinator_snapshot = self._coordinator.snapshot()
                if coordinator_snapshot.evidence_sequence is None:
                    raise LogicalTurnRejected("readiness evidence is missing")
                ready = self._coordinator.mark_inference_ready(
                    turn_id,
                    reason=boundary,
                    expected_evidence_sequence=(coordinator_snapshot.evidence_sequence),
                )
                self._readiness += 1
                self._boundary_counts[boundary] += 1
                if ready.readiness_token is None:
                    raise LogicalTurnRejected("readiness token is missing")
                committed = self._coordinator.commit(
                    turn_id,
                    readiness_token=ready.readiness_token,
                )
            except LogicalTurnRejected:
                self._extra_legacy_terminals += 1
                self._note_rejection()
                self._force_abort(LogicalTurnAbortReason.CONTROL)
                return False
            except Exception:
                self._extra_legacy_terminals += 1
                self._internal_errors += 1
                self._force_abort(LogicalTurnAbortReason.CONTROL)
                return False
            self._committed += 1
            self._paired_terminals += 1
            self._multi_epoch_commits += int(self._active_native_epochs > 1)
            if committed.text == legacy_text:
                self._composition_matches += 1
            else:
                self._composition_mismatches += 1
            self._mint_terminal_token()
            self._clear_active_bookkeeping()
            return True

    def abort(self, reason: LogicalTurnAbortReason) -> bool:
        """Abort an active composition turn; idle aborts remain telemetry."""

        if type(reason) is not LogicalTurnAbortReason:
            return self._reject_external()
        with self._lock:
            if self._closed:
                self._late_observations += 1
                return False
            if self._coordinator.active_turn_id is None:
                self._idle_aborts += 1
                return True
            self._force_abort(reason)
            return True

    def close(self) -> None:
        """Erase retained text, terminalize once, and reject later input."""

        with self._lock:
            if self._closed:
                return
            if self._coordinator.active_turn_id is not None:
                self._missing_legacy_terminals += 1
                self._force_abort(LogicalTurnAbortReason.SHUTDOWN)
            if self._terminal_lineage:
                self._terminal_missing_at_close = (
                    len(self._terminal_pending) + self._terminal_untracked_commits
                )
                self._terminal_unbound_at_close = (
                    self._terminal_untracked_commits
                    + sum(
                        lineage is None
                        for lineage in self._terminal_pending.values()
                    )
                )
                self._terminal_claimable.clear()
                self._terminal_claimed.clear()
                self._terminal_pending.clear()
                self._terminal_by_lineage.clear()
                self._terminal_completed_lineages.clear()
            self._closed = True

    def snapshot(self) -> LogicalTurnShadowSnapshot:
        """Return only counters; no transcript-bearing object escapes."""

        with self._lock:
            return LogicalTurnShadowSnapshot(
                scope=LogicalTurnShadowScope.SHERPA_RAW_COMPOSITION,
                closed=self._closed,
                started=self._started,
                native_epochs=self._native_epochs,
                native_finals=self._native_finals,
                empty_native_epochs=self._empty_native_epochs,
                held_native_finals=self._held_native_finals,
                readiness=self._readiness,
                committed=self._committed,
                multi_epoch_commits=self._multi_epoch_commits,
                aborted=self._aborted,
                active=self._coordinator.active_turn_id is not None,
                legacy_terminals=self._legacy_terminals,
                paired_terminals=self._paired_terminals,
                extra_legacy_terminals=self._extra_legacy_terminals,
                missing_legacy_terminals=self._missing_legacy_terminals,
                composition_matches=self._composition_matches,
                composition_mismatches=self._composition_mismatches,
                boundary_native_final=self._boundary_counts[
                    LogicalTurnBoundary.NATIVE_FINAL
                ],
                boundary_semantic_complete=self._boundary_counts[
                    LogicalTurnBoundary.SEMANTIC_COMPLETE
                ],
                boundary_silence_limit=self._boundary_counts[
                    LogicalTurnBoundary.SILENCE_LIMIT
                ],
                boundary_hard_limit=self._boundary_counts[
                    LogicalTurnBoundary.HARD_LIMIT
                ],
                abort_transcript=self._abort_counts[
                    LogicalTurnAbortReason.TRANSCRIPT_ABORT
                ],
                abort_control=self._abort_counts[LogicalTurnAbortReason.CONTROL],
                abort_stop=self._abort_counts[LogicalTurnAbortReason.STOP],
                abort_shutdown=self._abort_counts[LogicalTurnAbortReason.SHUTDOWN],
                idle_aborts=self._idle_aborts,
                coordinator_rejections=self._coordinator_rejections,
                bounds_overflows=self._bounds_overflows,
                internal_errors=self._internal_errors,
                late_observations=self._late_observations,
            )

    def terminal_lineage_snapshot(self) -> LogicalTurnTerminalLineageSnapshot:
        """Return aggregate-safe terminal counters with no retained identities."""

        with self._lock:
            if not self._terminal_lineage:
                raise RuntimeError("terminal lineage evidence is not enabled")
            missing = (
                self._terminal_missing_at_close
                if self._terminal_missing_at_close is not None
                else len(self._terminal_pending) + self._terminal_untracked_commits
            )
            unbound = (
                self._terminal_unbound_at_close
                if self._terminal_unbound_at_close is not None
                else self._terminal_untracked_commits
                + sum(lineage is None for lineage in self._terminal_pending.values())
            )
            return LogicalTurnTerminalLineageSnapshot(
                scope=(
                    LogicalTurnTerminalLineageScope.SHERPA_SELECTED_TERMINAL
                ),
                closed=self._closed,
                raw_commits=self._terminal_raw_commits,
                claimed_raw_commits=self._terminal_claimed_raw_commits,
                bound_raw_commits=self._terminal_bound_raw_commits,
                selected_finals=self._terminal_selected_finals,
                typed_aborts=self._terminal_typed_aborts,
                matched_selected_finals=(
                    self._terminal_matched_selected_finals
                ),
                matched_typed_aborts=self._terminal_matched_typed_aborts,
                unscoped_selected_finals=(
                    self._terminal_unscoped_selected_finals
                ),
                unscoped_typed_aborts=self._terminal_unscoped_typed_aborts,
                duplicate_selected_finals=(
                    self._terminal_duplicate_selected_finals
                ),
                duplicate_typed_aborts=(
                    self._terminal_duplicate_typed_aborts
                ),
                missing_downstream_terminals=missing,
                unbound_raw_commits=unbound,
                binding_rejections=self._terminal_binding_rejections,
                terminal_rejections=self._terminal_rejections,
                internal_errors=self._terminal_internal_errors,
                late_observations=self._terminal_late_observations,
                abort_reason_counts=tuple(
                    self._terminal_abort_counts[reason]
                    for reason in _TRANSCRIPT_ABORT_REASONS
                ),
                matched_abort_reason_counts=tuple(
                    self._terminal_matched_abort_counts[reason]
                    for reason in _TRANSCRIPT_ABORT_REASONS
                ),
            )


_SUM_FIELDS = tuple(
    item.name
    for item in fields(LogicalTurnShadowSnapshot)
    if item.name not in {"scope", "closed", "active"}
)


def aggregate_logical_turn_shadows(
    snapshots: Iterable[LogicalTurnShadowSnapshot],
) -> dict[str, object]:
    """Build one strict, closed, aggregate-only shadow report."""

    items = tuple(snapshots)
    if not items or any(type(item) is not LogicalTurnShadowSnapshot for item in items):
        raise ValueError("logical-turn shadow snapshots are missing or invalid")
    if any(not item.closed or item.active for item in items):
        raise ValueError("logical-turn shadow snapshots must be closed")
    if any(
        item.scope is not LogicalTurnShadowScope.SHERPA_RAW_COMPOSITION
        for item in items
    ):
        raise ValueError("logical-turn shadow scopes do not match")
    totals = {name: sum(getattr(item, name) for item in items) for name in _SUM_FIELDS}
    health_ok = not any(
        totals[name]
        for name in (
            "coordinator_rejections",
            "bounds_overflows",
            "internal_errors",
            "late_observations",
        )
    )
    evidence_sufficient = bool(
        totals["paired_terminals"] and totals["multi_epoch_commits"]
    )
    parity_exact = bool(
        evidence_sufficient
        and health_ok
        and not any(
            totals[name]
            for name in (
                "composition_mismatches",
                "extra_legacy_terminals",
                "missing_legacy_terminals",
            )
        )
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "scope": LogicalTurnShadowScope.SHERPA_RAW_COMPOSITION.value,
        "capabilities": dict(_CAPABILITIES),
        "coverage": {
            "runs": len(items),
            "closed_runs": len(items),
            "active_at_report": 0,
            "complete": True,
        },
        "lifecycle": {
            name: totals[name]
            for name in (
                "started",
                "native_epochs",
                "native_finals",
                "empty_native_epochs",
                "held_native_finals",
                "readiness",
                "committed",
                "aborted",
                "idle_aborts",
            )
        },
        "boundaries": {
            "native_final": totals["boundary_native_final"],
            "semantic_complete": totals["boundary_semantic_complete"],
            "silence_limit": totals["boundary_silence_limit"],
            "hard_limit": totals["boundary_hard_limit"],
        },
        "abort_reasons": {
            "transcript_abort": totals["abort_transcript"],
            "control": totals["abort_control"],
            "stop": totals["abort_stop"],
            "shutdown": totals["abort_shutdown"],
        },
        "parity": {
            "legacy_terminals": totals["legacy_terminals"],
            "paired_terminals": totals["paired_terminals"],
            "extra_legacy_terminals": totals["extra_legacy_terminals"],
            "missing_legacy_terminals": totals["missing_legacy_terminals"],
            "composition_matches": totals["composition_matches"],
            "composition_mismatches": totals["composition_mismatches"],
            "multi_epoch_commits": totals["multi_epoch_commits"],
            "evidence_sufficient": evidence_sufficient,
            "exact": parity_exact,
        },
        "health": {
            "coordinator_rejections": totals["coordinator_rejections"],
            "bounds_overflows": totals["bounds_overflows"],
            "internal_errors": totals["internal_errors"],
            "late_observations": totals["late_observations"],
            "ok": health_ok,
        },
    }
    validate_logical_turn_shadow_report(report)
    return report


def validate_logical_turn_shadow_report(report: object) -> None:
    """Reject unknown fields, insufficient claims, and broken counter algebra."""

    if type(report) is not dict or set(report) != {
        "schema_version",
        "scope",
        "capabilities",
        "coverage",
        "lifecycle",
        "boundaries",
        "abort_reasons",
        "parity",
        "health",
    }:
        raise ValueError("invalid logical-turn shadow report shape")
    if (
        type(report["schema_version"]) is not int
        or report["schema_version"] != 1
        or type(report["scope"]) is not str
        or report["scope"] != LogicalTurnShadowScope.SHERPA_RAW_COMPOSITION.value
    ):
        raise ValueError("invalid logical-turn shadow report identity")
    capabilities = report["capabilities"]
    if type(capabilities) is not dict or set(capabilities) != set(_CAPABILITIES):
        raise ValueError("invalid logical-turn shadow capabilities")
    if any(
        type(capabilities[name]) is not bool
        or capabilities[name] is not expected
        for name, expected in _CAPABILITIES.items()
    ):
        raise ValueError("invalid logical-turn shadow capabilities")
    expected = {
        "coverage": {"runs", "closed_runs", "active_at_report", "complete"},
        "lifecycle": {
            "started",
            "native_epochs",
            "native_finals",
            "empty_native_epochs",
            "held_native_finals",
            "readiness",
            "committed",
            "aborted",
            "idle_aborts",
        },
        "boundaries": {
            "native_final",
            "semantic_complete",
            "silence_limit",
            "hard_limit",
        },
        "abort_reasons": {"transcript_abort", "control", "stop", "shutdown"},
        "parity": {
            "legacy_terminals",
            "paired_terminals",
            "extra_legacy_terminals",
            "missing_legacy_terminals",
            "composition_matches",
            "composition_mismatches",
            "multi_epoch_commits",
            "evidence_sufficient",
            "exact",
        },
        "health": {
            "coordinator_rejections",
            "bounds_overflows",
            "internal_errors",
            "late_observations",
            "ok",
        },
    }
    boolean_names = {"complete", "evidence_sufficient", "exact", "ok"}
    for section, names in expected.items():
        value = report.get(section)
        if type(value) is not dict or set(value) != names:
            raise ValueError("invalid logical-turn shadow report section")
        for name, item in value.items():
            if name in boolean_names:
                if type(item) is not bool:
                    raise TypeError("logical-turn shadow verdict must be boolean")
            else:
                _non_negative_int(name, item)
    coverage = report["coverage"]
    lifecycle = report["lifecycle"]
    boundaries = report["boundaries"]
    aborts = report["abort_reasons"]
    parity = report["parity"]
    health = report["health"]
    if (
        coverage["runs"] == 0
        or coverage["closed_runs"] != coverage["runs"]
        or coverage["active_at_report"] != 0
        or coverage["complete"] is not True
    ):
        raise ValueError("logical-turn shadow coverage is not closed")
    if lifecycle["started"] != lifecycle["committed"] + lifecycle["aborted"]:
        raise ValueError("aggregate lifecycle totals do not reconcile")
    if lifecycle["native_epochs"] != (
        lifecycle["native_finals"] + lifecycle["empty_native_epochs"]
    ):
        raise ValueError("aggregate native epoch totals do not reconcile")
    if lifecycle["native_epochs"] and not lifecycle["started"]:
        raise ValueError("aggregate native epochs require a started turn")
    if lifecycle["native_epochs"] and not lifecycle["native_finals"]:
        raise ValueError("aggregate native epochs require a non-empty final")
    if lifecycle["empty_native_epochs"] and not lifecycle["held_native_finals"]:
        raise ValueError("aggregate empty epochs require held native evidence")
    if lifecycle["held_native_finals"] > lifecycle["native_finals"]:
        raise ValueError("aggregate held native finals exceed finals")
    unheld_native_finals = (
        lifecycle["native_finals"] - lifecycle["held_native_finals"]
    )
    if unheld_native_finals > lifecycle["started"]:
        raise ValueError("aggregate unheld finals exceed started turns")
    if unheld_native_finals < (
        lifecycle["committed"] - parity["multi_epoch_commits"]
    ):
        raise ValueError("aggregate single-epoch commits exceed unheld finals")
    if lifecycle["committed"] > lifecycle["native_finals"]:
        raise ValueError("aggregate commits exceed native finals")
    if lifecycle["committed"] > lifecycle["readiness"]:
        raise ValueError("aggregate commits exceed readiness transitions")
    if lifecycle["readiness"] > lifecycle["native_finals"]:
        raise ValueError("aggregate readiness exceeds native finals")
    if (
        lifecycle["held_native_finals"] + lifecycle["readiness"]
        > lifecycle["native_epochs"]
    ):
        raise ValueError("aggregate held and ready transitions exceed native epochs")
    if lifecycle["readiness"] != sum(boundaries.values()):
        raise ValueError("aggregate boundary totals do not reconcile")
    if lifecycle["aborted"] != sum(aborts.values()):
        raise ValueError("aggregate abort totals do not reconcile")
    if parity["paired_terminals"] != lifecycle["committed"]:
        raise ValueError("aggregate paired terminals do not equal commits")
    if parity["legacy_terminals"] != (
        parity["paired_terminals"] + parity["extra_legacy_terminals"]
    ):
        raise ValueError("aggregate legacy terminals do not reconcile")
    if lifecycle["readiness"] > parity["legacy_terminals"]:
        raise ValueError("aggregate readiness exceeds legacy terminals")
    if (
        lifecycle["readiness"] - lifecycle["committed"]
        > aborts["control"]
    ):
        raise ValueError("aggregate failed readiness exceeds control aborts")
    if parity["paired_terminals"] != (
        parity["composition_matches"] + parity["composition_mismatches"]
    ):
        raise ValueError("aggregate parity totals do not reconcile")
    if parity["multi_epoch_commits"] > lifecycle["committed"]:
        raise ValueError("aggregate multi-epoch commits exceed commits")
    if parity["multi_epoch_commits"] > lifecycle["held_native_finals"]:
        raise ValueError("aggregate multi-epoch commits exceed held native finals")
    if 2 * parity["multi_epoch_commits"] > lifecycle["native_epochs"]:
        raise ValueError("aggregate multi-epoch commits exceed native epoch evidence")
    if parity["missing_legacy_terminals"] > aborts["shutdown"]:
        raise ValueError("aggregate missing terminals exceed shutdowns")
    failure_slack = health["coordinator_rejections"] + health["internal_errors"]
    if lifecycle["started"] > lifecycle["native_finals"] + failure_slack:
        raise ValueError("aggregate started turns exceed finals plus failures")
    if lifecycle["started"] + parity["multi_epoch_commits"] > (
        lifecycle["native_epochs"] + failure_slack
    ):
        raise ValueError("aggregate started and multi-epoch turns exceed epochs")
    expected_sufficient = bool(
        parity["paired_terminals"] and parity["multi_epoch_commits"]
    )
    if parity["evidence_sufficient"] is not expected_sufficient:
        raise ValueError("logical-turn shadow sufficiency is inconsistent")
    expected_health = not any(
        health[name]
        for name in (
            "coordinator_rejections",
            "bounds_overflows",
            "internal_errors",
            "late_observations",
        )
    )
    if health["ok"] is not expected_health:
        raise ValueError("logical-turn shadow health verdict is inconsistent")
    expected_exact = bool(
        expected_sufficient
        and expected_health
        and not any(
            parity[name]
            for name in (
                "composition_mismatches",
                "extra_legacy_terminals",
                "missing_legacy_terminals",
            )
        )
    )
    if parity["exact"] is not expected_exact:
        raise ValueError("logical-turn shadow parity verdict is inconsistent")


_TERMINAL_SUM_FIELDS = tuple(
    item.name
    for item in fields(LogicalTurnTerminalLineageSnapshot)
    if item.name
    not in {
        "scope",
        "closed",
        "abort_reason_counts",
        "matched_abort_reason_counts",
    }
)


def _terminal_abort_report(counts: tuple[int, ...]) -> dict[str, int]:
    return {
        reason.value: counts[index]
        for index, reason in enumerate(_TRANSCRIPT_ABORT_REASONS)
    }


def aggregate_logical_turn_terminal_lineage(
    snapshots: Iterable[LogicalTurnTerminalLineageSnapshot],
) -> dict[str, object]:
    """Build one strict aggregate without transcript or acoustic identities."""

    items = tuple(snapshots)
    if not items or any(
        type(item) is not LogicalTurnTerminalLineageSnapshot for item in items
    ):
        raise ValueError("terminal lineage snapshots are missing or invalid")
    if any(
        not item.closed
        or item.scope
        is not LogicalTurnTerminalLineageScope.SHERPA_SELECTED_TERMINAL
        for item in items
    ):
        raise ValueError("terminal lineage snapshots must be closed and scoped")
    totals = {
        name: sum(getattr(item, name) for item in items)
        for name in _TERMINAL_SUM_FIELDS
    }
    abort_counts = tuple(
        sum(item.abort_reason_counts[index] for item in items)
        for index in range(len(_TRANSCRIPT_ABORT_REASONS))
    )
    matched_abort_counts = tuple(
        sum(item.matched_abort_reason_counts[index] for item in items)
        for index in range(len(_TRANSCRIPT_ABORT_REASONS))
    )
    health_ok = not any(
        totals[name]
        for name in (
            "binding_rejections",
            "terminal_rejections",
            "internal_errors",
            "late_observations",
        )
    )
    evidence_sufficient = totals["raw_commits"] > 0
    exact = bool(
        evidence_sufficient
        and health_ok
        and totals["claimed_raw_commits"] == totals["raw_commits"]
        and totals["bound_raw_commits"] == totals["raw_commits"]
        and not any(
            totals[name]
            for name in (
                "unscoped_selected_finals",
                "unscoped_typed_aborts",
                "duplicate_selected_finals",
                "duplicate_typed_aborts",
                "missing_downstream_terminals",
                "unbound_raw_commits",
            )
        )
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "scope": (
            LogicalTurnTerminalLineageScope.SHERPA_SELECTED_TERMINAL.value
        ),
        "capabilities": dict(_TERMINAL_CAPABILITIES),
        "coverage": {
            "runs": len(items),
            "closed_runs": len(items),
            "terminal_observations": (
                totals["selected_finals"] + totals["typed_aborts"]
            ),
            "complete": True,
        },
        "outcomes": {
            "selected_finals": totals["selected_finals"],
            "typed_aborts": totals["typed_aborts"],
            "abort_reason_counts": _terminal_abort_report(abort_counts),
        },
        "lineage": {
            "raw_commits": totals["raw_commits"],
            "claimed_raw_commits": totals["claimed_raw_commits"],
            "bound_raw_commits": totals["bound_raw_commits"],
            "matched_selected_finals": totals["matched_selected_finals"],
            "matched_typed_aborts": totals["matched_typed_aborts"],
            "matched_abort_reason_counts": _terminal_abort_report(
                matched_abort_counts
            ),
            "unscoped_selected_finals": totals["unscoped_selected_finals"],
            "unscoped_typed_aborts": totals["unscoped_typed_aborts"],
            "duplicate_selected_finals": totals["duplicate_selected_finals"],
            "duplicate_typed_aborts": totals["duplicate_typed_aborts"],
            "missing_downstream_terminals": totals[
                "missing_downstream_terminals"
            ],
            "unbound_raw_commits": totals["unbound_raw_commits"],
            "evidence_sufficient": evidence_sufficient,
            "exact": exact,
        },
        "health": {
            "binding_rejections": totals["binding_rejections"],
            "terminal_rejections": totals["terminal_rejections"],
            "internal_errors": totals["internal_errors"],
            "late_observations": totals["late_observations"],
            "ok": health_ok,
        },
    }
    validate_logical_turn_terminal_lineage_report(report)
    return report


def validate_logical_turn_terminal_lineage_report(report: object) -> None:
    """Reject unknown fields, forged totals, and unsupported capability claims."""

    if type(report) is not dict or set(report) != {
        "schema_version",
        "scope",
        "capabilities",
        "coverage",
        "outcomes",
        "lineage",
        "health",
    }:
        raise ValueError("invalid terminal lineage report shape")
    if (
        type(report["schema_version"]) is not int
        or report["schema_version"] != 1
        or type(report["scope"]) is not str
        or report["scope"]
        != LogicalTurnTerminalLineageScope.SHERPA_SELECTED_TERMINAL.value
    ):
        raise ValueError("invalid terminal lineage report identity")
    capabilities = report["capabilities"]
    if type(capabilities) is not dict or set(capabilities) != set(
        _TERMINAL_CAPABILITIES
    ):
        raise ValueError("invalid terminal lineage capabilities")
    if any(
        type(capabilities[name]) is not bool
        or capabilities[name] is not expected
        for name, expected in _TERMINAL_CAPABILITIES.items()
    ):
        raise ValueError("invalid terminal lineage capabilities")
    expected = {
        "coverage": {
            "runs",
            "closed_runs",
            "terminal_observations",
            "complete",
        },
        "outcomes": {"selected_finals", "typed_aborts", "abort_reason_counts"},
        "lineage": {
            "raw_commits",
            "claimed_raw_commits",
            "bound_raw_commits",
            "matched_selected_finals",
            "matched_typed_aborts",
            "matched_abort_reason_counts",
            "unscoped_selected_finals",
            "unscoped_typed_aborts",
            "duplicate_selected_finals",
            "duplicate_typed_aborts",
            "missing_downstream_terminals",
            "unbound_raw_commits",
            "evidence_sufficient",
            "exact",
        },
        "health": {
            "binding_rejections",
            "terminal_rejections",
            "internal_errors",
            "late_observations",
            "ok",
        },
    }
    for section, names in expected.items():
        value = report.get(section)
        if type(value) is not dict or set(value) != names:
            raise ValueError("invalid terminal lineage report section")
    coverage = report["coverage"]
    outcomes = report["outcomes"]
    lineage = report["lineage"]
    health = report["health"]
    for name, item in coverage.items():
        if name == "complete":
            if type(item) is not bool:
                raise TypeError("coverage verdict must be boolean")
        else:
            _non_negative_int(name, item)
    for name in ("selected_finals", "typed_aborts"):
        _non_negative_int(name, outcomes[name])
    for name, item in lineage.items():
        if name in {
            "matched_abort_reason_counts",
            "evidence_sufficient",
            "exact",
        }:
            continue
        _non_negative_int(name, item)
    for name, item in health.items():
        if name == "ok":
            if type(item) is not bool:
                raise TypeError("health verdict must be boolean")
        else:
            _non_negative_int(name, item)
    for name in ("evidence_sufficient", "exact"):
        if type(lineage[name]) is not bool:
            raise TypeError("lineage verdict must be boolean")
    reason_names = {reason.value for reason in _TRANSCRIPT_ABORT_REASONS}
    abort_counts = outcomes["abort_reason_counts"]
    matched_abort_counts = lineage["matched_abort_reason_counts"]
    for value in (abort_counts, matched_abort_counts):
        if type(value) is not dict or set(value) != reason_names:
            raise ValueError("invalid terminal abort reason counts")
        for name, item in value.items():
            _non_negative_int(name, item)
    matched = (
        lineage["matched_selected_finals"]
        + lineage["matched_typed_aborts"]
    )
    if (
        coverage["runs"] == 0
        or coverage["closed_runs"] != coverage["runs"]
        or coverage["complete"] is not True
        or coverage["terminal_observations"]
        != outcomes["selected_finals"] + outcomes["typed_aborts"]
        or lineage["raw_commits"]
        != matched + lineage["missing_downstream_terminals"]
        or lineage["claimed_raw_commits"] > lineage["raw_commits"]
        or lineage["bound_raw_commits"] > lineage["claimed_raw_commits"]
        or lineage["unbound_raw_commits"]
        > lineage["missing_downstream_terminals"]
        or lineage["bound_raw_commits"]
        != matched
        + (
            lineage["missing_downstream_terminals"]
            - lineage["unbound_raw_commits"]
        )
        or outcomes["selected_finals"]
        != lineage["matched_selected_finals"]
        + lineage["unscoped_selected_finals"]
        + lineage["duplicate_selected_finals"]
        or outcomes["typed_aborts"]
        != lineage["matched_typed_aborts"]
        + lineage["unscoped_typed_aborts"]
        + lineage["duplicate_typed_aborts"]
        or sum(abort_counts.values()) != outcomes["typed_aborts"]
        or sum(matched_abort_counts.values())
        != lineage["matched_typed_aborts"]
        or any(
            matched_abort_counts[name] > abort_counts[name]
            for name in reason_names
        )
    ):
        raise ValueError("terminal lineage totals do not reconcile")
    expected_health = not any(
        health[name]
        for name in (
            "binding_rejections",
            "terminal_rejections",
            "internal_errors",
            "late_observations",
        )
    )
    if health["ok"] is not expected_health:
        raise ValueError("terminal lineage health verdict is inconsistent")
    expected_sufficient = lineage["raw_commits"] > 0
    if lineage["evidence_sufficient"] is not expected_sufficient:
        raise ValueError("terminal lineage sufficiency is inconsistent")
    expected_exact = bool(
        expected_sufficient
        and expected_health
        and lineage["claimed_raw_commits"] == lineage["raw_commits"]
        and lineage["bound_raw_commits"] == lineage["raw_commits"]
        and not any(
            lineage[name]
            for name in (
                "unscoped_selected_finals",
                "unscoped_typed_aborts",
                "duplicate_selected_finals",
                "duplicate_typed_aborts",
                "missing_downstream_terminals",
                "unbound_raw_commits",
            )
        )
    )
    if lineage["exact"] is not expected_exact:
        raise ValueError("terminal lineage exactness verdict is inconsistent")


__all__ = [
    "LogicalTurnCompositionShadow",
    "LogicalTurnShadowScope",
    "LogicalTurnShadowSnapshot",
    "LogicalTurnTerminalLineageScope",
    "LogicalTurnTerminalLineageSnapshot",
    "aggregate_logical_turn_terminal_lineage",
    "aggregate_logical_turn_shadows",
    "validate_logical_turn_terminal_lineage_report",
    "validate_logical_turn_shadow_report",
]
