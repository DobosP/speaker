"""Effect-free ownership and guidance for exact local STT evidence capture.

This module intentionally has no runtime, event-bus, memory, LLM, or capability
imports.  :class:`CaptureOnlySession` owns one already-constructed local audio
engine directly and binds only typed acoustic callbacks.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from threading import Event, Lock, get_ident
import time
from typing import Callable, Protocol

from always_on_agent.acoustic import AcousticSource

from .engine import (
    AcousticSignal,
    AudioEngine,
    CommandDetection,
    EngineCallbacks,
    FinalTranscript,
    OwnerVerification,
    PartialTranscript,
    TranscriptAbort,
)
from .engines.sherpa import SherpaExecutionPurpose
from .guided_stt_plan import GuidedSttCase, GuidedSttPlan


_CASE_TIMEOUT_SEC = 45.0
_POST_TERMINAL_QUIET_SEC = 1.0


class CaptureOnlyError(RuntimeError):
    """A detail-free capture protocol or fatal media failure."""


class CaptureGuide(Protocol):
    """Human guidance boundary; implementations must not consume ASR text."""

    def acknowledge_geometry(self, geometry: str) -> None:
        """Block until the requested physical geometry is ready."""

    def prepare_case(
        self,
        case: GuidedSttCase,
        *,
        ordinal: int,
        total: int,
    ) -> None:
        """Show one fixed plan phrase and block until the owner is ready."""

    def case_armed(self, *, ordinal: int, total: int) -> None:
        """Signal that exactly one selected final may now arrive."""


class TerminalCaptureGuide:
    """Visual, no-audio guide used by the physical ``./live.sh`` workflow."""

    def __init__(
        self,
        *,
        write: Callable[[str], None] = print,
        read: Callable[[str], str] = input,
    ) -> None:
        self._write = write
        self._read = read

    def acknowledge_geometry(self, geometry: str) -> None:
        if geometry == "close":
            instruction = "Use the normal close-speaking position."
        elif geometry == "far":
            instruction = "Move to the planned far-field speaking position."
        else:
            raise CaptureOnlyError("capture plan geometry is invalid")
        self._read(
            f"[capture] {instruction} Press Enter only when that geometry is ready: "
        )

    def prepare_case(
        self,
        case: GuidedSttCase,
        *,
        ordinal: int,
        total: int,
    ) -> None:
        # This is immutable project-owned reference text, never recognized text.
        self._write(f"[capture] case {ordinal}/{total}: {case.expected_text}")
        self._read("[capture] Press Enter when ready to arm this one phrase: ")

    def case_armed(self, *, ordinal: int, total: int) -> None:
        self._write(f"[capture] armed {ordinal}/{total}; speak the phrase once now.")


@dataclass(frozen=True, slots=True)
class CaptureOnlyResult:
    plan_id: str
    plan_sha256: str
    planned_cases: int
    accepted_finals: int
    capture_state: str
    completed: bool

    def summary_facts(self) -> dict[str, object]:
        """Return aggregate-only facts safe for a private run summary."""

        return {
            "capture_protocol": "guided-stt-capture-v1",
            "capture_plan_id": self.plan_id,
            "capture_plan_sha256": self.plan_sha256,
            "capture_planned_cases": self.planned_cases,
            "capture_accepted_finals": self.accepted_finals,
            "capture_state": self.capture_state,
            "capture_completed": self.completed,
            "effects_enabled": False,
        }


@dataclass(frozen=True, slots=True)
class _ArmedCase:
    index: int
    armed_at: float


class CaptureOnlySession:
    """Own one capture-purpose engine without constructing an assistant.

    A case is armed only after an explicit terminal acknowledgement.  Exactly
    one typed selected-final callback consumes that arm.  The following fixed
    quiet guard keeps the session unarmed, so a late second/split final fails
    before the next case can be acknowledged.  Sherpa exposes no authoritative
    "all pending final work is empty" callback here; the bounded quiet guard and
    explicit next-case Enter are therefore detection controls, not invented
    proof that endpoint work cannot remain pending.
    """

    def __init__(
        self,
        engine: AudioEngine,
        plan: GuidedSttPlan | None = None,
        *,
        guide: CaptureGuide | None = None,
        case_timeout_sec: float = _CASE_TIMEOUT_SEC,
        post_terminal_quiet_sec: float = _POST_TERMINAL_QUIET_SEC,
    ) -> None:
        if not isinstance(engine, AudioEngine):
            raise TypeError("capture-only engine must implement AudioEngine")
        if (
            getattr(engine, "execution_purpose", None)
            is not SherpaExecutionPurpose.STT_CAPTURE_ONLY
        ):
            raise TypeError("capture-only session requires a capture-purpose engine")
        if plan is not None and not isinstance(plan, GuidedSttPlan):
            raise TypeError("plan must be a GuidedSttPlan or None")
        for name, value, allow_zero in (
            ("case_timeout_sec", case_timeout_sec, False),
            ("post_terminal_quiet_sec", post_terminal_quiet_sec, True),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
                or (not allow_zero and float(value) == 0.0)
            ):
                raise ValueError(f"{name} must be finite and positive")

        self.engine = engine
        self.plan = plan
        self.guide = guide or TerminalCaptureGuide()
        self.case_timeout_sec = float(case_timeout_sec)
        self.post_terminal_quiet_sec = float(post_terminal_quiet_sec)

        self._state_lock = Lock()
        self._started = False
        self._armed_case: _ArmedCase | None = None
        self._case_terminal: Event | None = None
        self._accepted_finals = 0
        self._capture_state = "starting"
        self._capture_identity: tuple[str, int, int] | None = None
        self._seen_span_keys: set[tuple[str, int, int, str]] = set()
        self._failure: CaptureOnlyError | None = None
        self._failure_event = Event()

        self._stop_lock = Lock()
        self._stop_started = False
        self._stop_owner: int | None = None
        self._stop_done = Event()
        self._stop_error: BaseException | None = None

    @property
    def accepted_finals(self) -> int:
        with self._state_lock:
            return self._accepted_finals

    @property
    def capture_state(self) -> str:
        with self._state_lock:
            return self._capture_state

    @property
    def stopped(self) -> bool:
        return self._stop_done.is_set()

    def _fail_locked(self, message: str) -> None:
        if self._failure is None:
            self._failure = CaptureOnlyError(message)
        self._armed_case = None
        terminal = self._case_terminal
        self._failure_event.set()
        if terminal is not None:
            terminal.set()

    def _raise_failure(self) -> None:
        with self._state_lock:
            failure = self._failure
        if failure is not None:
            raise failure

    @staticmethod
    def _ignore_partial(_result: PartialTranscript) -> None:
        pass

    @staticmethod
    def _ignore_barge(_result: AcousticSignal) -> None:
        pass

    @staticmethod
    def _ignore_command(_result: CommandDetection) -> None:
        pass

    def _on_final(self, result: FinalTranscript) -> None:
        # Deliberately never inspect, retain, render, or log ``result.text``.
        with self._state_lock:
            if not isinstance(result, FinalTranscript):
                self._fail_locked("capture delivered a non-typed final")
                return
            if self._capture_state != "open":
                self._fail_locked("capture final arrived while media was not open")
                return
            armed = self._armed_case
            expected = self._accepted_finals
            if armed is None:
                self._fail_locked("capture delivered an unarmed or extra final")
                return
            if armed.index != expected:
                self._fail_locked("capture final ordering is invalid")
                return
            if result.owner_verification is not OwnerVerification.UNKNOWN:
                self._fail_locked("capture final carried unexpected speaker authority")
                return
            acoustic = result.acoustic
            if acoustic is None:
                self._fail_locked("capture final has no acoustic lineage")
                return
            identities = {
                (span.stream_id, span.capture_epoch, span.capture_generation)
                for span in acoustic.spans
            }
            if (
                len(identities) != 1
                or any(
                    span.source is not AcousticSource.LIVE_CAPTURE
                    for span in acoustic.spans
                )
                or any(
                    span.speech_start_at is None
                    or span.speech_start_at < armed.armed_at
                    for span in acoustic.spans
                )
            ):
                self._fail_locked("capture final lineage does not match its arm")
                return
            identity = next(iter(identities))
            if (
                self._capture_identity is not None
                and identity != self._capture_identity
            ):
                self._fail_locked("capture identity changed during the fixed plan")
                return
            span_keys = {span.key for span in acoustic.spans}
            if len(span_keys) != len(acoustic.spans):
                self._fail_locked("capture final contains duplicate acoustic lineage")
                return
            if span_keys & self._seen_span_keys:
                self._fail_locked("capture final reused terminal acoustic lineage")
                return
            terminal = self._case_terminal
            if terminal is None:
                self._fail_locked("capture case has no terminal owner")
                return
            self._capture_identity = identity
            self._seen_span_keys.update(span_keys)
            self._accepted_finals += 1
            self._armed_case = None
            terminal.set()

    def _on_abort(self, _abort: TranscriptAbort) -> None:
        with self._state_lock:
            self._fail_locked("guided capture received a transcript abort")

    def _on_capture_state(self, state: str, _message: str) -> None:
        with self._state_lock:
            if state not in {"open", "recovering", "fatal"}:
                self._fail_locked("capture published an invalid media state")
                return
            self._capture_state = state
            if state == "recovering":
                self._fail_locked("capture device recovery invalidated the fixed plan")
            elif state == "fatal":
                self._fail_locked("capture device entered a fatal state")

    def start(self) -> None:
        with self._state_lock:
            if self._started:
                raise RuntimeError("capture-only session already started")
            if self._stop_started:
                raise RuntimeError("capture-only session is already stopping")
            self._started = True
        self.engine.start(
            EngineCallbacks(
                on_partial_result=self._ignore_partial,
                on_final_result=self._on_final,
                on_barge_in_result=self._ignore_barge,
                on_command_result=self._ignore_command,
                on_transcript_abort=self._on_abort,
                on_capture_state=self._on_capture_state,
            )
        )
        warm_stt = getattr(self.engine, "warm_stt", None)
        if not callable(warm_stt):
            raise CaptureOnlyError("capture engine has no STT warm boundary")
        warm_stt()

    def _arm_case(self, index: int) -> Event:
        with self._state_lock:
            if self._failure is not None:
                raise self._failure
            if self._capture_state != "open":
                raise CaptureOnlyError("capture is not open for a new case")
            if self._armed_case is not None or self._case_terminal is not None:
                raise CaptureOnlyError("capture case ownership is already active")
            if index != self._accepted_finals:
                raise CaptureOnlyError("capture case ordering is invalid")
            terminal = Event()
            self._case_terminal = terminal
            self._armed_case = _ArmedCase(
                index=index,
                armed_at=time.perf_counter(),
            )
            return terminal

    def _finish_case(self, terminal: Event) -> None:
        with self._state_lock:
            if self._case_terminal is not terminal:
                self._fail_locked("capture case terminal ownership changed")
            self._case_terminal = None
            failure = self._failure
        if failure is not None:
            raise failure

    def _wait_for_case(self, terminal: Event) -> None:
        if not terminal.wait(timeout=self.case_timeout_sec):
            with self._state_lock:
                self._fail_locked("capture case timed out")
        self._finish_case(terminal)

    def _guard_post_terminal_quiet(self) -> None:
        if self._failure_event.wait(timeout=self.post_terminal_quiet_sec):
            self._raise_failure()

    def _result(self, *, completed: bool) -> CaptureOnlyResult:
        plan = self.plan
        return CaptureOnlyResult(
            plan_id=(plan.plan_id if plan is not None else "none"),
            plan_sha256=(plan.sha256 if plan is not None else ""),
            planned_cases=(plan.case_count if plan is not None else 0),
            accepted_finals=self.accepted_finals,
            capture_state=self.capture_state,
            completed=completed,
        )

    def run(self) -> CaptureOnlyResult:
        """Run the exact guide and always release the engine once."""

        try:
            self.start()
            plan = self.plan
            if plan is None:
                # Diagnostic callers may inspect an unplanned capture manually,
                # but there is no evidence-complete success condition.
                while not self._failure_event.wait(timeout=0.1):
                    pass
                self._raise_failure()
                raise CaptureOnlyError("capture-only run has no completion plan")

            prior_geometry: str | None = None
            for index, case in enumerate(plan.cases):
                self._raise_failure()
                if case.geometry != prior_geometry:
                    self.guide.acknowledge_geometry(case.geometry)
                    self._raise_failure()
                    prior_geometry = case.geometry
                ordinal = index + 1
                self.guide.prepare_case(
                    case,
                    ordinal=ordinal,
                    total=plan.case_count,
                )
                self._raise_failure()
                terminal = self._arm_case(index)
                self.guide.case_armed(ordinal=ordinal, total=plan.case_count)
                self._wait_for_case(terminal)
                self._guard_post_terminal_quiet()

            self._raise_failure()
            completed = self.accepted_finals == plan.case_count
            if not completed or self.capture_state != "open":
                raise CaptureOnlyError("capture plan did not complete")
        finally:
            self.stop()
        # Engine.stop() is the drain boundary. A late final/abort/fatal emitted
        # during teardown must defeat the provisional 16/16 count.
        self._raise_failure()
        if (
            self.plan is None
            or self.accepted_finals != self.plan.case_count
            or self.capture_state != "open"
        ):
            raise CaptureOnlyError("capture plan did not complete cleanly")
        return self._result(completed=True)

    def stop(self) -> None:
        """Release the directly owned engine exactly once."""

        caller = get_ident()
        with self._stop_lock:
            if not self._stop_started:
                self._stop_started = True
                self._stop_owner = caller
                owns_stop = True
            elif self._stop_owner == caller and not self._stop_done.is_set():
                return
            else:
                owns_stop = False

        if owns_stop:
            try:
                self.engine.stop()
            except BaseException as exc:
                with self._stop_lock:
                    self._stop_error = exc
                raise
            finally:
                with self._stop_lock:
                    self._stop_owner = None
                self._stop_done.set()
            return

        self._stop_done.wait()
        with self._stop_lock:
            error = self._stop_error
        if error is not None:
            raise error


__all__ = [
    "CaptureGuide",
    "CaptureOnlyError",
    "CaptureOnlyResult",
    "CaptureOnlySession",
    "TerminalCaptureGuide",
]
