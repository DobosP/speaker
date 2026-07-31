"""Publisher-bound LiveKit Agents adapter for Speaker's trusted runtime.

The optional LiveKit SDK is deliberately not imported here.  A remote
composition layer must create an ``AgentSession`` with one exact publisher,
local/audited speech components, ``llm=None``, ``tools=[]``, and recording
disabled, then expose the small, lease-tracked protocol below. Raw
``AgentSession`` and ``AgentOutput`` objects must not escape that wrapper. This
adapter translates the media session into :class:`core.engine.AudioEngine`;
``VoiceRuntime`` and ``always_on_agent`` remain the only LLM, tool, task, and
authority plane.

Streaming STT ``is_final`` events are segment boundaries, not complete user
turns.  A bridge ``Agent.on_user_turn_completed`` must call the generation-
bound committer returned by :meth:`LiveKitAgentsEngine.user_turn_committer`.
"""

from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
import logging
import threading
import time
from typing import Any, Callable, Optional, Protocol
import uuid

from always_on_agent.acoustic import AcousticSource, EndpointReason

from core.engine import (
    AcousticSignal,
    AudioEngine,
    EngineCallbacks,
    FinalTranscript,
    OwnerVerification,
    PartialTranscript,
    PlaybackCapabilities,
    PlaybackOutcome,
    PlaybackReceipt,
    TrackedSpeech,
    TranscriptAbort,
    TranscriptAbortReason,
)

from ._acoustic_turn import AcousticTurnTracker


log = logging.getLogger("speaker.livekit_agents")

_CALL_TIMEOUT_SEC = 2.0
_RECENT_TURN_IDS = 128
_PLAYBACK_CAPABILITIES = PlaybackCapabilities(
    tracked_terminal=True,
    # AgentSession's public AudioOutput playback_started event has no speech ID.
    # A correlated sink wrapper is required before this may become true.
    exact_started=False,
    sample_counts=False,
    speech_style_hints=False,
)


class SpeechHandleLike(Protocol):
    """Public LiveKit 1.6.7 ``SpeechHandle`` surface used here."""

    @property
    def interrupted(self) -> bool: ...

    def exception(self) -> BaseException | None: ...

    def interrupt(self, *, force: bool = False) -> "SpeechHandleLike": ...


@dataclass(frozen=True)
class PublisherSessionPolicy:
    """Fail-closed construction facts owned by the future trusted wrapper.

    This is an explicit configuration contract, not identity or cryptographic
    proof. Production composition must accept only Speaker's concrete wrapper,
    which owns the raw SDK objects and derives these facts while constructing
    them; arbitrary caller implementations are test doubles only.
    """

    publisher_scoped: bool
    recording_disabled: bool
    llm_disabled: bool
    agent_tools_disabled: bool
    session_tools_disabled: bool
    mcp_disabled: bool
    remote_inference_disabled: bool
    preemptive_generation_disabled: bool
    output_leases: bool

    @property
    def is_safe(self) -> bool:
        return all(
            value is True
            for value in (
                self.publisher_scoped,
                self.recording_disabled,
                self.llm_disabled,
                self.agent_tools_disabled,
                self.session_tools_disabled,
                self.mcp_disabled,
                self.remote_inference_disabled,
                self.preemptive_generation_disabled,
                self.output_leases,
            )
        )


class PublisherSessionLike(Protocol):
    """Trusted publisher wrapper around LiveKit Agents 1.6.7.

    The wrapper records the exact ``RoomOptions.participant_identity``, owns all
    raw output mutations, and never exposes the raw session/output. Its atomic
    ``say_tracked`` installs ``on_done`` before returning and either returns a
    handle plus opaque output lease or raises only when no output was admitted.
    The lease is invalidated by every sink/tail replacement, enable transition,
    detach, or close. ``close`` is forwarded only after SDK playout drains and
    output detaches. ``close_on_playout_failure`` must immediately fence wrapper
    admission and schedule a non-draining SDK close; its later ``close`` event is
    the disposal proof for unresolved handles.
    """

    @property
    def participant_identity(self) -> str: ...

    @property
    def policy(self) -> PublisherSessionPolicy: ...

    def on(self, event: str, callback: Callable[[Any], None]) -> Any: ...

    def off(self, event: str, callback: Callable[[Any], None]) -> Any: ...

    def say_tracked(
        self,
        text: str,
        *,
        allow_interruptions: bool,
        add_to_chat_ctx: bool,
        on_done: Callable[[SpeechHandleLike], None],
    ) -> tuple[SpeechHandleLike, object]: ...

    def output_lease_valid(self, lease: object) -> bool: ...

    def close_on_playout_failure(self) -> None: ...


@dataclass(eq=False)
class _PlaybackCall:
    speech: TrackedSpeech
    playback_generation: int
    lifecycle_generation: int
    on_terminal: Callable[[PlaybackReceipt], None]
    on_started: Optional[Callable[[str], None]] = None
    handle: Optional[SpeechHandleLike] = None
    output_lease: object | None = None
    output_active: bool = False
    stop_requested: bool = False
    terminal: bool = False


@dataclass(frozen=True)
class _SessionHandlers:
    transcript: Callable[[Any], None]
    user_state: Callable[[Any], None]
    agent_state: Callable[[Any], None]
    false_interruption: Callable[[Any], None]
    close: Callable[[Any], None]


@dataclass(frozen=True)
class _InterruptionCandidate:
    episode: int
    call: _PlaybackCall


@dataclass
class _TurnLifecycle:
    generation: int
    tracker: AcousticTurnTracker
    episode: int = 0
    user_speaking: bool = False
    candidate: _InterruptionCandidate | None = None
    barge_emitted: bool = False


class LiveKitAgentsEngine(AudioEngine):
    """Adapt one explicitly publisher-bound session to ``AudioEngine``.

    Participant identity proves only transport scoping. It is never copied into
    acoustic lineage and never mints owner or action authority. Remote finals
    use ``origin="remote_audio"`` (which the current authority enum coerces to
    untrusted UNKNOWN) and :class:`OwnerVerification.UNKNOWN`.

    The future composition owns SDK start/close ordering. ``start`` here binds
    wrapper events; ``stop`` unbinds input events and interrupts only handles
    captured before its generation fence. A retained ``close`` observer resolves
    otherwise-unobservable output only after SDK drain/detach. Every session
    operation is marshalled onto ``loop``.
    """

    def __init__(
        self,
        session: PublisherSessionLike,
        *,
        participant_identity: str,
        loop: asyncio.AbstractEventLoop,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        identity = self._opaque_id("participant identity", participant_identity)
        bound_identity = self._opaque_id(
            "session participant identity",
            getattr(session, "participant_identity", ""),
        )
        if bound_identity != identity:
            raise ValueError("session is not bound to the requested publisher")
        policy = getattr(session, "policy", None)
        if not isinstance(policy, PublisherSessionPolicy) or not policy.is_safe:
            raise ValueError("session does not satisfy the safe publisher policy")
        if loop is None:
            raise TypeError("loop is required")

        self._session = session
        self._participant_identity = identity
        self._loop = loop
        self._clock = clock
        self._callbacks = EngineCallbacks()
        self._lock = threading.RLock()
        self._running = False
        self._session_closed = False
        self._fatal_close_requested = False
        self._lifecycle_generation = 0
        self._turn_state: _TurnLifecycle | None = None
        self._session_handlers: _SessionHandlers | None = None
        self._retained_close_handlers: list[_SessionHandlers] = []
        self._playback_generation = 0
        self._pending: list[_PlaybackCall] = []
        self._agent_output_active = False
        self._recent_turn_ids: deque[tuple[int, str]] = deque()
        self._recent_turn_id_set: set[tuple[int, str]] = set()
        self._stream_id = f"livekit-agent-{uuid.uuid4().hex}"

    @staticmethod
    def _opaque_id(name: str, value: object) -> str:
        result = str(value or "").strip()
        if (
            not result
            or len(result) > 128
            or any(ord(char) < 0x20 or ord(char) == 0x7F for char in result)
        ):
            raise ValueError(f"{name} must be a bounded opaque string")
        return result

    @property
    def participant_identity(self) -> str:
        """Exact transport binding; never an owner-authority assertion."""

        return self._participant_identity

    @property
    def playback_capabilities(self) -> PlaybackCapabilities:
        return _PLAYBACK_CAPABILITIES

    @property
    def is_speaking(self) -> bool:
        with self._lock:
            return any(not call.terminal for call in self._pending)

    @property
    def lifecycle_generation(self) -> int:
        with self._lock:
            return self._lifecycle_generation

    def start(self, callbacks: EngineCallbacks) -> None:
        with self._lock:
            if self._running:
                raise RuntimeError("LiveKit Agents engine is already started")
            if self._session_closed:
                raise RuntimeError("LiveKit Agents session is closed")
            if self._fatal_close_requested:
                raise RuntimeError("LiveKit Agents session close is pending")
            if self._pending:
                raise RuntimeError("previous playback is not terminal")
            self._lifecycle_generation += 1
            generation = self._lifecycle_generation
            state = _TurnLifecycle(
                generation=generation,
                tracker=AcousticTurnTracker(
                    f"{self._stream_id}-g{generation}",
                    source=AcousticSource.REMOTE_TRACK,
                ),
            )
            handlers = _SessionHandlers(
                transcript=lambda event: self._on_transcribed(event, state),
                user_state=lambda event: self._on_user_state(event, state),
                agent_state=lambda event: self._on_agent_state(event, state),
                false_interruption=lambda event: self._on_false_interruption(
                    event, state
                ),
                close=lambda event: self._on_session_close(event, state, handlers),
            )
            self._callbacks = callbacks
            self._turn_state = state
            self._session_handlers = handlers
            self._running = True
            self._agent_output_active = False
        try:
            self._on_loop(lambda: self._bind_on_loop(handlers), wait=True)
        except BaseException:
            with self._lock:
                if (
                    self._session_handlers is handlers
                    and self._turn_state is state
                    and self._lifecycle_generation == generation
                ):
                    self._running = False
                    self._lifecycle_generation += 1
                    state.tracker.abandon()
                    self._turn_state = None
                    self._session_handlers = None
            try:
                self._on_loop(lambda: self._unbind_on_loop(handlers), wait=True)
            except Exception:
                log.debug("failed to roll back LiveKit callback binding", exc_info=True)
            raise

    def stop(self) -> None:
        with self._lock:
            if not self._running and not self._pending:
                return
            handlers = self._session_handlers
            state = self._turn_state
            self._session_handlers = None
            self._turn_state = None
            self._running = False
            self._lifecycle_generation += 1
            self._playback_generation += 1
            calls = tuple(call for call in self._pending if not call.terminal)
            for call in calls:
                call.stop_requested = True
            if state is not None:
                state.candidate = None
                state.barge_emitted = False
                state.tracker.abandon()
            self._agent_output_active = False
        try:
            self._on_loop(
                lambda: self._stop_on_loop(handlers, calls),
                wait=True,
            )
        except Exception:
            log.warning("LiveKit Agents cleanup could not reach its loop", exc_info=True)
            # A handle may still be playing when its loop disappears. Never
            # release that ownership without a real handle terminal or the
            # wrapper's post-drain close event. Calls not yet handed to the
            # wrapper are safe to drop.
            for call in calls:
                if call.handle is None:
                    self._terminalize(call, PlaybackOutcome.DROPPED)

    def speak(
        self,
        text: str,
        on_done: Optional[Callable[[], None]] = None,
    ) -> None:
        speech = TrackedSpeech(
            fragment_id=f"legacy-{uuid.uuid4().hex}",
            text=text,
        )

        def terminal(_receipt: PlaybackReceipt) -> None:
            if on_done is not None:
                on_done()

        self.speak_tracked(speech, on_terminal=terminal)

    def speak_tracked(
        self,
        speech: TrackedSpeech,
        *,
        on_terminal: Callable[[PlaybackReceipt], None],
        on_started: Optional[Callable[[str], None]] = None,
    ) -> None:
        if not isinstance(speech, TrackedSpeech):
            raise TypeError("speech must be TrackedSpeech")
        with self._lock:
            call = _PlaybackCall(
                speech=speech,
                playback_generation=self._playback_generation,
                lifecycle_generation=self._lifecycle_generation,
                on_terminal=on_terminal,
                on_started=on_started,
            )
            if self._running:
                self._pending.append(call)
                reject = False
            else:
                reject = True
        if reject:
            self._terminalize(call, PlaybackOutcome.DROPPED)
            return
        try:
            self._on_loop(lambda: self._say_on_loop(call), wait=False)
        except Exception:
            self._terminalize(call, PlaybackOutcome.FAILED)

    def stop_speaking(self) -> None:
        with self._lock:
            self._playback_generation += 1
            calls = tuple(call for call in self._pending if not call.terminal)
            for call in calls:
                call.stop_requested = True
            if self._turn_state is not None:
                self._turn_state.candidate = None
        try:
            # Per-handle interruption cannot cancel speech admitted after this
            # generation fence; AgentSession.interrupt() is deliberately not used.
            self._on_loop(lambda: self._interrupt_calls_on_loop(calls), wait=False)
        except Exception:
            log.warning("LiveKit Agents interruption scheduling failed", exc_info=True)
            for call in calls:
                if call.handle is None:
                    self._terminalize(call, PlaybackOutcome.DROPPED)

    def user_turn_committer(self) -> Callable[..., bool]:
        """Return a completed-turn callback bound to this start generation."""

        with self._lock:
            state = self._turn_state
            if not self._running or state is None:
                raise RuntimeError("LiveKit Agents engine is not started")

        def commit(text: str, *, turn_id: str | None = None) -> bool:
            return self._commit_user_turn(
                text,
                turn_id=turn_id,
                state=state,
            )

        return commit

    def commit_user_turn(self, text: str, *, turn_id: str | None = None) -> bool:
        """Commit on the current lifecycle; tests/legacy bridges may use this.

        New composition should capture :meth:`user_turn_committer` once and pass
        ``new_message.id`` as ``turn_id`` so callbacks are both generation-bound
        and idempotent.
        """

        with self._lock:
            state = self._turn_state
        if state is None:
            return False
        return self._commit_user_turn(
            text,
            turn_id=turn_id,
            state=state,
        )

    def _commit_user_turn(
        self,
        text: str,
        *,
        turn_id: str | None,
        state: _TurnLifecycle,
    ) -> bool:
        clean = str(text or "").strip()
        if not clean:
            self._abort_user_turn(
                TranscriptAbortReason.EMPTY_FINAL,
                state=state,
            )
            return False
        claimed_turn_id = (
            None if turn_id is None else self._opaque_id("turn_id", turn_id)
        )

        stamp = max(0.0, float(self._clock()))
        with self._lock:
            if not self._is_current_state_locked(state):
                return False
            if claimed_turn_id is not None and not self._claim_turn_id_locked(
                claimed_turn_id, state.generation
            ):
                return False
            barge = (
                self._make_barge_locked(state, stamp)
                if self._candidate_valid_locked(state)
                and not state.barge_emitted
                else None
            )
            lineage, revision = state.tracker.close(
                speech_end_at=None,
                endpoint_committed_at=stamp,
                emitted_at=stamp,
                endpoint_reason=EndpointReason.UNKNOWN,
            )
            state.candidate = None
            state.barge_emitted = False
            state.user_speaking = False

        if barge is not None and not self._publish_barge(
            barge, state
        ):
            return False
        result = FinalTranscript(
            text=clean,
            owner_verification=OwnerVerification.UNKNOWN,
            origin="remote_audio",
            acoustic=lineage,
            revision=revision,
        )
        return self._publish_final(result, state)

    def abort_user_turn(
        self,
        reason: TranscriptAbortReason = TranscriptAbortReason.ABANDONED,
    ) -> bool:
        with self._lock:
            state = self._turn_state
        if state is None:
            return False
        return self._abort_user_turn(reason, state=state)

    def _abort_user_turn(
        self,
        reason: TranscriptAbortReason,
        *,
        state: _TurnLifecycle,
    ) -> bool:
        if not isinstance(reason, TranscriptAbortReason):
            raise TypeError("reason must be a TranscriptAbortReason")
        stamp = max(0.0, float(self._clock()))
        with self._lock:
            if not self._is_current_state_locked(state):
                return False
            aborted = state.tracker.abort(emitted_at=stamp)
            state.barge_emitted = False
            state.candidate = None
            state.user_speaking = False
            if aborted is None:
                return False
            lineage, revision = aborted
            event = TranscriptAbort(
                acoustic=lineage,
                revision=revision,
                reason=reason,
            )
            callback = self._callbacks.on_transcript_abort
            if callback is not None:
                callback(event)
            return callback is not None

    # --- LiveKit loop event handling ---
    def _bind_on_loop(self, handlers: _SessionHandlers) -> None:
        self._session.on("user_input_transcribed", handlers.transcript)
        self._session.on("user_state_changed", handlers.user_state)
        self._session.on("agent_state_changed", handlers.agent_state)
        self._session.on(
            "agent_false_interruption", handlers.false_interruption
        )
        self._session.on("close", handlers.close)

    def _unbind_on_loop(
        self,
        handlers: _SessionHandlers,
        *,
        keep_close: bool = False,
    ) -> None:
        pairs = [
            ("user_input_transcribed", handlers.transcript),
            ("user_state_changed", handlers.user_state),
            ("agent_state_changed", handlers.agent_state),
            ("agent_false_interruption", handlers.false_interruption),
        ]
        if not keep_close:
            pairs.append(("close", handlers.close))
        for event, callback in pairs:
            try:
                self._session.off(event, callback)
            except Exception:
                log.debug("failed to remove LiveKit %s callback", event, exc_info=True)

    def _stop_on_loop(
        self,
        handlers: _SessionHandlers | None,
        calls: tuple[_PlaybackCall, ...],
    ) -> None:
        with self._lock:
            active = tuple(call for call in calls if not call.terminal)
        if handlers is not None:
            keep_close = any(call.handle is not None for call in active)
            self._unbind_on_loop(handlers, keep_close=keep_close)
            if keep_close:
                with self._lock:
                    self._retained_close_handlers.append(handlers)
        self._interrupt_calls_on_loop(active)

    def _on_session_close(
        self,
        event: Any,
        _state: _TurnLifecycle,
        handlers: _SessionHandlers,
    ) -> None:
        reason_value = getattr(getattr(event, "reason", None), "value", None)
        reason = str(reason_value or getattr(event, "reason", "unknown"))
        reason = reason if reason in {
            "error",
            "job_shutdown",
            "participant_disconnected",
            "user_initiated",
            "task_completed",
        } else "unknown"
        stamp = max(0.0, float(self._clock()))
        with self._lock:
            if self._session_closed:
                return
            self._session_closed = True
            was_running = self._running or self._fatal_close_requested
            was_output_active = self._agent_output_active
            current_state = self._turn_state
            callbacks = self._callbacks
            self._running = False
            self._fatal_close_requested = False
            self._lifecycle_generation += 1
            self._playback_generation += 1
            self._turn_state = None
            self._session_handlers = None
            self._agent_output_active = False
            aborted = None
            if current_state is not None:
                aborted = current_state.tracker.abort(emitted_at=stamp)
                current_state.candidate = None
                current_state.barge_emitted = False
            calls = tuple(call for call in self._pending if not call.terminal)
            bound_handlers = tuple(
                dict.fromkeys(
                    (handlers, *self._retained_close_handlers)
                )
            )
            self._retained_close_handlers.clear()

        # LiveKit emits close only after current speech drains/interrupts and
        # output.audio is detached. Any impossible leftover lease is therefore
        # safe to resolve with no text, even when an interrupt request failed.
        if was_output_active:
            try:
                callbacks.on_speech_end()
            except Exception:
                log.exception("LiveKit speech-end callback raised during close")
        if aborted is not None and callbacks.on_transcript_abort is not None:
            lineage, revision = aborted
            try:
                callbacks.on_transcript_abort(
                    TranscriptAbort(
                        acoustic=lineage,
                        revision=revision,
                        reason=TranscriptAbortReason.SHUTDOWN,
                    )
                )
            except Exception:
                log.exception("LiveKit transcript-abort callback raised during close")
        for call in calls:
            try:
                interrupted = bool(call.handle is not None and call.handle.interrupted)
            except Exception:
                interrupted = False
            if call.stop_requested or interrupted:
                outcome = (
                    PlaybackOutcome.INTERRUPTED
                    if call.output_active
                    else PlaybackOutcome.DROPPED
                )
            else:
                outcome = PlaybackOutcome.FAILED
            self._terminalize(call, outcome)
        if was_running:
            try:
                callbacks.on_capture_state(
                    "fatal",
                    f"livekit_session_closed:{reason}",
                )
            except Exception:
                log.exception("LiveKit capture-state callback raised during close")
        for bound in bound_handlers:
            self._unbind_on_loop(bound)

    def _on_user_state(self, event: Any, state: _TurnLifecycle) -> None:
        new_state = str(getattr(event, "new_state", ""))
        with self._lock:
            if not self._is_current_state_locked(state):
                return
            if new_state != "speaking":
                state.user_speaking = False
                return
            if state.user_speaking:
                return
            state.user_speaking = True
            state.episode += 1
            # A new VAD episode always invalidates an unconfirmed older one.
            state.candidate = None
            call = next(
                (
                    item
                    for item in self._pending
                    if not item.terminal
                    and not item.stop_requested
                    and item.output_active
                    and item.lifecycle_generation == state.generation
                    and item.playback_generation == self._playback_generation
                ),
                None,
            )
            if self._agent_output_active and call is not None:
                # Provisional only. A transcript segment or completed user turn
                # must confirm it before controller cancellation.
                state.candidate = _InterruptionCandidate(state.episode, call)

    def _on_agent_state(self, event: Any, state: _TurnLifecycle) -> None:
        new_state = str(getattr(event, "new_state", ""))
        with self._lock:
            if not self._is_current_state_locked(state):
                return
            was_active = self._agent_output_active
            self._agent_output_active = new_state == "speaking"
            if self._agent_output_active:
                # With llm=None/tools=[] every assistant speech belongs to this
                # adapter. Mark only the oldest owned handle as output-active;
                # later queued fragments cannot manufacture a barge candidate.
                call = next(
                    (
                        item
                        for item in self._pending
                        if not item.terminal
                        and not item.stop_requested
                        and item.handle is not None
                        and item.lifecycle_generation == state.generation
                        and item.playback_generation == self._playback_generation
                    ),
                    None,
                )
                if call is not None:
                    call.output_active = True
            if not was_active and self._agent_output_active:
                self._callbacks.on_speech_start()
            elif was_active and not self._agent_output_active:
                # This is assistant playout quiescence, not user speech EOT.
                self._callbacks.on_speech_end()

    def _on_false_interruption(
        self, _event: Any, state: _TurnLifecycle
    ) -> None:
        with self._lock:
            if self._is_current_state_locked(state):
                state.candidate = None

    def _on_transcribed(self, event: Any, state: _TurnLifecycle) -> None:
        clean = str(getattr(event, "transcript", "") or "").strip()
        if not clean:
            return
        stamp = max(0.0, float(self._clock()))
        with self._lock:
            if not self._is_current_state_locked(state):
                return
            if bool(getattr(event, "is_final", False)):
                if self._candidate_valid_locked(state) and not state.barge_emitted:
                    self._publish_barge(
                        self._make_barge_locked(state, stamp),
                        state,
                    )
                return

            lineage, revision = state.tracker.partial(emitted_at=stamp)
            result = PartialTranscript(
                text=clean,
                acoustic=lineage,
                revision=revision,
            )
            callback = self._callbacks.on_partial_result
            if callback is not None:
                callback(result)
            else:
                self._callbacks.on_partial(clean)

    def _candidate_valid_locked(self, state: _TurnLifecycle) -> bool:
        candidate = state.candidate
        if candidate is None or candidate.episode != state.episode:
            return False
        call = candidate.call
        interrupted = bool(call.handle is not None and call.handle.interrupted)
        return bool(
            self._is_current_state_locked(state)
            and call.lifecycle_generation == state.generation
            and call.playback_generation == self._playback_generation
            and not call.stop_requested
            and call.output_active
            and (not call.terminal or interrupted)
        )

    def _make_barge_locked(
        self,
        state: _TurnLifecycle,
        stamp: float,
    ) -> AcousticSignal:
        lineage, revision = state.tracker.advance(emitted_at=stamp)
        state.barge_emitted = True
        return AcousticSignal(
            acoustic=lineage,
            revision=revision,
            detected_at=stamp,
        )

    def _publish_barge(
        self, signal: AcousticSignal, state: _TurnLifecycle
    ) -> bool:
        with self._lock:
            if not self._is_current_state_locked(state):
                return False
            callback = self._callbacks.on_barge_in_result
            if callback is not None:
                callback(signal)
            else:
                self._callbacks.on_barge_in()
            return True

    def _publish_final(
        self, result: FinalTranscript, state: _TurnLifecycle
    ) -> bool:
        with self._lock:
            if not self._is_current_state_locked(state):
                return False
            callback = self._callbacks.on_final_result
            if callback is not None:
                callback(result)
            else:
                self._callbacks.on_final(result.text)
            return True

    def _claim_turn_id_locked(
        self,
        turn_id: str,
        lifecycle_generation: int,
    ) -> bool:
        key = (lifecycle_generation, turn_id)
        if key in self._recent_turn_id_set:
            return False
        if len(self._recent_turn_ids) >= _RECENT_TURN_IDS:
            expired = self._recent_turn_ids.popleft()
            self._recent_turn_id_set.remove(expired)
        self._recent_turn_ids.append(key)
        self._recent_turn_id_set.add(key)
        return True

    # --- tracked output ---
    def _say_on_loop(self, call: _PlaybackCall) -> None:
        with self._lock:
            current = bool(
                self._running
                and call.lifecycle_generation == self._lifecycle_generation
                and call.playback_generation == self._playback_generation
                and not call.terminal
            )
        if not current:
            self._terminalize(call, PlaybackOutcome.DROPPED)
            return

        try:
            handle, lease = self._session.say_tracked(
                call.speech.text,
                allow_interruptions=True,
                add_to_chat_ctx=False,
                on_done=lambda completed: self._handle_done(call, completed),
            )
            with self._lock:
                call.handle = handle
                call.output_lease = lease
                stale = bool(
                    call.stop_requested
                    or call.playback_generation != self._playback_generation
                    or call.lifecycle_generation != self._lifecycle_generation
                )
            if stale:
                self._interrupt_calls_on_loop((call,))
        except Exception:
            # The trusted wrapper contract guarantees this exception means no
            # output escaped and no handle exists. It owns callback installation
            # atomically with SDK admission.
            log.warning("LiveKit publisher wrapper admission failed", exc_info=True)
            self._terminalize(call, PlaybackOutcome.FAILED)

    def _interrupt_calls_on_loop(
        self, calls: tuple[_PlaybackCall, ...]
    ) -> None:
        for call in calls:
            with self._lock:
                if call.terminal:
                    continue
                handle = call.handle
            if handle is None:
                self._terminalize(call, PlaybackOutcome.DROPPED)
                continue
            try:
                handle.interrupt(force=True)
            except Exception:
                log.warning("LiveKit speech-handle interruption failed", exc_info=True)
                # The sink may still be audible. Retain playback ownership until
                # the real handle terminal or post-drain session close arrives.
                self._request_playout_failure_close_on_loop()

    def _request_playout_failure_close_on_loop(self) -> None:
        with self._lock:
            if self._session_closed or self._fatal_close_requested:
                return
            self._fatal_close_requested = True
            handlers = self._session_handlers
            state = self._turn_state
            callbacks = self._callbacks
            self._session_handlers = None
            self._turn_state = None
            self._running = False
            self._lifecycle_generation += 1
            self._playback_generation += 1
            aborted = None
            if state is not None:
                aborted = state.tracker.abort(
                    emitted_at=max(0.0, float(self._clock()))
                )
                state.candidate = None
                state.barge_emitted = False
            if handlers is not None:
                self._unbind_on_loop(handlers, keep_close=True)
                self._retained_close_handlers.append(handlers)

        if aborted is not None and callbacks.on_transcript_abort is not None:
            lineage, revision = aborted
            try:
                callbacks.on_transcript_abort(
                    TranscriptAbort(
                        acoustic=lineage,
                        revision=revision,
                        reason=TranscriptAbortReason.INTERNAL_ERROR,
                    )
                )
            except Exception:
                log.exception("LiveKit abort callback raised during playout failure")
        try:
            callbacks.on_capture_state(
                "recovering",
                "livekit_playout_close_requested",
            )
        except Exception:
            log.exception("LiveKit capture-state callback raised during playout failure")
        try:
            self._session.close_on_playout_failure()
        except Exception:
            # New admissions remain fenced even if close scheduling itself fails.
            log.exception("LiveKit fatal session close request failed")

    def _handle_done(
        self,
        call: _PlaybackCall,
        handle: SpeechHandleLike,
    ) -> None:
        try:
            self._on_loop(
                lambda: self._handle_done_on_loop(call, handle),
                wait=False,
            )
        except Exception:
            # A vanished loop is not evidence that the remote sink is quiet.
            # Retain ownership for the close/disposal path.
            log.warning("LiveKit handle terminal could not reach its loop", exc_info=True)

    def _handle_done_on_loop(
        self,
        call: _PlaybackCall,
        handle: SpeechHandleLike,
    ) -> None:
        try:
            error = handle.exception()
        except Exception as exc:
            error = exc
        interrupted = bool(handle.interrupted)
        with self._lock:
            state = self._turn_state
            if (
                interrupted
                and not call.stop_requested
                and call.output_active
                and call.playback_generation == self._playback_generation
                and state is not None
                and self._is_current_state_locked(state)
                and call.lifecycle_generation == state.generation
            ):
                # Preserve the candidate even if LiveKit removes the handle
                # before emitting the first final transcript segment.
                if state.candidate is None:
                    state.episode += 1
                    state.candidate = _InterruptionCandidate(state.episode, call)
                elif state.candidate.call is call:
                    state.candidate = _InterruptionCandidate(
                        state.candidate.episode,
                        call,
                    )
            elif (
                state is not None
                and state.candidate is not None
                and state.candidate.call is call
                and not interrupted
                and not state.barge_emitted
            ):
                state.candidate = None
            generation_current = bool(
                call.playback_generation == self._playback_generation
                and call.lifecycle_generation == self._lifecycle_generation
            )

        try:
            lease_valid = bool(
                call.output_lease is not None
                and self._session.output_lease_valid(call.output_lease)
            )
        except Exception:
            log.warning("LiveKit output-lease validation failed", exc_info=True)
            lease_valid = False
        if error is not None:
            outcome = PlaybackOutcome.FAILED
        elif interrupted or not generation_current:
            outcome = (
                PlaybackOutcome.INTERRUPTED
                if call.output_active
                else PlaybackOutcome.DROPPED
            )
        elif not lease_valid:
            outcome = PlaybackOutcome.FAILED
        else:
            outcome = PlaybackOutcome.COMPLETED
        self._terminalize(call, outcome)

    def _terminalize(
        self,
        call: _PlaybackCall,
        outcome: PlaybackOutcome,
    ) -> None:
        with self._lock:
            if call.terminal:
                return
            call.terminal = True
            try:
                self._pending.remove(call)
            except ValueError:
                pass
            retired = (
                tuple(self._retained_close_handlers)
                if (
                    not self._running
                    and not self._pending
                    and not self._fatal_close_requested
                )
                else ()
            )
            if retired:
                self._retained_close_handlers.clear()
        for handlers in retired:
            try:
                self._on_loop(
                    lambda bound=handlers: self._unbind_on_loop(bound),
                    wait=False,
                )
            except Exception:
                log.debug("failed to remove retained close callback", exc_info=True)
        receipt = PlaybackReceipt(
            fragment_id=call.speech.fragment_id,
            outcome=outcome,
            safe_text_prefix=(
                call.speech.text if outcome is PlaybackOutcome.COMPLETED else ""
            ),
        )
        try:
            call.on_terminal(receipt)
        except Exception:
            log.exception("tracked playback terminal callback failed")

    # --- lifecycle helpers ---
    def _is_current_state_locked(self, state: _TurnLifecycle) -> bool:
        return bool(
            self._running
            and self._turn_state is state
            and state.generation == self._lifecycle_generation
        )

    def _on_loop(self, action: Callable[[], None], *, wait: bool) -> None:
        if self._loop.is_closed() or not self._loop.is_running():
            raise RuntimeError("LiveKit Agents event loop is not running")
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            action()
            return

        if not wait:
            def invoke_unobserved() -> None:
                try:
                    action()
                except Exception:
                    log.exception("LiveKit Agents loop action failed")

            self._loop.call_soon_threadsafe(invoke_unobserved)
            return

        completed: Future[None] = Future()

        def invoke() -> None:
            try:
                action()
            except BaseException as exc:
                completed.set_exception(exc)
            else:
                completed.set_result(None)

        self._loop.call_soon_threadsafe(invoke)
        completed.result(timeout=_CALL_TIMEOUT_SEC)


__all__ = [
    "LiveKitAgentsEngine",
    "PublisherSessionLike",
    "PublisherSessionPolicy",
    "SpeechHandleLike",
]
