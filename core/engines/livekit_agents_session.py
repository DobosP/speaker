"""Pinned, non-escaping LiveKit Agents publisher-session composition.

This optional module owns raw LiveKit Agents session, agent, RoomIO, and output
objects. Importing it does not import the optional SDK; construction performs a
lazy exact-version check. Speaker's ``LiveKitAgentsEngine`` remains the only
bridge into the application control plane.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import threading
from types import SimpleNamespace
from typing import Any, Callable

from .livekit_agents import PublisherSessionPolicy, SpeechHandleLike


log = logging.getLogger("speaker.livekit_agents_session")

_CLOSE_TIMEOUT_SEC = 2.0
_READY_TIMEOUT_SEC = 5.0
_LIVEKIT_AGENTS_VERSION = "1.6.7"
_LIVEKIT_RTC_VERSION = "1.1.14"
_SDK_EVENTS = (
    "user_input_transcribed",
    "user_state_changed",
    "agent_state_changed",
    "agent_false_interruption",
    "close",
)


class LiveKitAgentsDependencyError(RuntimeError):
    """The pinned optional LiveKit Agents runtime is absent or incompatible."""


def _bounded_opaque_id(name: str, value: object) -> str:
    result = str(value or "").strip()
    if (
        not result
        or len(result) > 128
        or any(ord(char) < 0x20 or ord(char) == 0x7F for char in result)
    ):
        raise ValueError(f"{name} must be a bounded opaque string")
    return result


@dataclass(frozen=True)
class AuditedLocalSpeechComponents:
    """Explicit local STT/VAD/TTS objects admitted to the media-only session.

    LiveKit model-name strings and omitted models may resolve through hosted
    inference.  The concrete wrapper therefore accepts only already-built
    objects plus a bounded audit identifier.  The identifier is a reviewed
    configuration fact, not cryptographic proof; production assembly remains
    responsible for supplying Speaker-owned local implementations.
    """

    stt: object
    vad: object
    tts: object
    audit_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "audit_id",
            _bounded_opaque_id("local speech audit_id", self.audit_id),
        )
        for name in ("stt", "vad", "tts"):
            value = getattr(self, name)
            if value is None or isinstance(value, (str, bytes, bool, int, float)):
                raise TypeError(
                    f"{name} must be an explicit audited local component object"
                )


def _load_livekit_agents_sdk() -> SimpleNamespace:
    """Import exactly the reviewed optional LiveKit Agents release lazily."""

    try:
        import livekit.agents as agents
        from livekit import rtc
    except ImportError as exc:
        raise LiveKitAgentsDependencyError(
            "trusted LiveKit media needs optional livekit==1.1.14 and "
            "livekit-agents==1.6.7; "
            "install requirements-remote.txt in an isolated environment"
        ) from exc

    room_io = getattr(agents, "room_io", None)
    cli = getattr(agents, "cli", None)
    track_source = getattr(rtc, "TrackSource", None)
    connection_state = getattr(rtc, "ConnectionState", None)
    return SimpleNamespace(
        agents_version=str(getattr(agents, "__version__", "")),
        rtc_version=str(getattr(rtc, "__version__", "")),
        Agent=getattr(agents, "Agent", None),
        AgentSession=getattr(agents, "AgentSession", None),
        RoomIO=getattr(agents, "RoomIO", None),
        RoomOptions=getattr(room_io, "RoomOptions", None),
        AudioInputOptions=getattr(room_io, "AudioInputOptions", None),
        AudioOutputOptions=getattr(room_io, "AudioOutputOptions", None),
        AgentsConsole=getattr(cli, "AgentsConsole", None),
        TrackPublishOptions=getattr(rtc, "TrackPublishOptions", None),
        microphone_source=getattr(track_source, "SOURCE_MICROPHONE", None),
        connection_connected=getattr(connection_state, "CONN_CONNECTED", None),
    )


def _validated_livekit_agents_sdk() -> Any:
    result = _load_livekit_agents_sdk()
    agents_version = str(getattr(result, "agents_version", ""))
    if agents_version != _LIVEKIT_AGENTS_VERSION:
        raise LiveKitAgentsDependencyError(
            "trusted LiveKit media requires reviewed livekit-agents=="
            f"{_LIVEKIT_AGENTS_VERSION}, got {agents_version or 'unknown'}"
        )
    rtc_version = str(getattr(result, "rtc_version", ""))
    if rtc_version != _LIVEKIT_RTC_VERSION:
        raise LiveKitAgentsDependencyError(
            "trusted LiveKit media requires reviewed livekit=="
            f"{_LIVEKIT_RTC_VERSION}, got {rtc_version or 'unknown'}"
        )
    required = (
        "Agent",
        "AgentSession",
        "RoomIO",
        "RoomOptions",
        "AudioInputOptions",
        "AudioOutputOptions",
        "AgentsConsole",
        "TrackPublishOptions",
        "microphone_source",
        "connection_connected",
    )
    missing = [name for name in required if getattr(result, name, None) is None]
    if missing:
        raise LiveKitAgentsDependencyError(
            "LiveKit Agents binding is missing reviewed surfaces: " + ", ".join(missing)
        )
    constructors = required[:-2]
    invalid = [name for name in constructors if not callable(getattr(result, name))]
    if invalid:
        raise LiveKitAgentsDependencyError(
            "LiveKit Agents binding has invalid reviewed surfaces: "
            + ", ".join(invalid)
        )
    return result


@dataclass(frozen=True)
class _PublisherOutputLease:
    owner: object
    session_generation: int
    route_generation: int


class _TrustedSpeechHandle:
    """Non-escaping tracked proxy around one SDK ``SpeechHandle``."""

    def __init__(
        self,
        owner: "TrustedLiveKitPublisherSession",
        on_done: Callable[[SpeechHandleLike], None],
    ) -> None:
        # Save the terminal callback before the SDK is allowed to admit speech.
        self._owner = owner
        self._on_done = on_done
        self._lock = threading.RLock()
        self._sdk_handle: Any | None = None
        self._done = False
        self._notification_queued = False
        self._interrupted = False
        self._interrupt_requested = False
        self._error: BaseException | None = None
        self._admission_error: BaseException | None = None

    @property
    def interrupted(self) -> bool:
        with self._lock:
            return self._interrupted

    @property
    def interrupt_requested(self) -> bool:
        with self._lock:
            return self._interrupt_requested

    @property
    def admission_error(self) -> BaseException | None:
        with self._lock:
            return self._admission_error

    def exception(self) -> BaseException | None:
        with self._lock:
            if not self._done:
                raise asyncio.InvalidStateError
            return self._error

    def interrupt(self, *, force: bool = False) -> "_TrustedSpeechHandle":
        self._owner._interrupt_tracked_handle(self, force=force)
        return self

    def _bind_sdk_handle(self, handle: object) -> None:
        add_done_callback = getattr(handle, "add_done_callback", None)
        if not callable(add_done_callback):
            raise TypeError("LiveKit SpeechHandle lacks add_done_callback()")
        with self._lock:
            if self._sdk_handle is not None:
                raise RuntimeError("LiveKit SpeechHandle is already bound")
            if self._done:
                return
            self._sdk_handle = handle
        add_done_callback(self._on_sdk_done)

    def _sdk_handle_for_interrupt(self) -> Any | None:
        with self._lock:
            if self._done:
                return None
            self._interrupt_requested = True
            return self._sdk_handle

    def _mark_interrupt_result(self, *, interrupted: bool) -> None:
        with self._lock:
            self._interrupted = self._interrupted or interrupted

    def _mark_admission_error(self, error: BaseException) -> None:
        with self._lock:
            self._admission_error = error

    def _on_sdk_done(self, handle: object) -> None:
        try:
            interrupted = bool(getattr(handle, "interrupted"))
        except Exception:
            interrupted = self.interrupt_requested
        try:
            exception = getattr(handle, "exception")
            error = exception() if callable(exception) else None
        except BaseException as exc:
            error = exc
        self._finish(interrupted=interrupted, error=error)

    def _finish(
        self,
        *,
        interrupted: bool,
        error: BaseException | None,
    ) -> bool:
        with self._lock:
            if self._done:
                return False
            self._done = True
            self._interrupted = self._interrupted or interrupted
            self._error = error
        self._owner._tracked_handle_terminal(self)
        return True

    def _finish_for_session_close(self) -> None:
        with self._lock:
            requested = self._interrupt_requested
            error = self._admission_error
        self._finish(
            interrupted=requested,
            error=(
                error
                if error is not None
                else (
                    None
                    if requested
                    else RuntimeError(
                        "LiveKit session closed before tracked speech terminal"
                    )
                )
            ),
        )

    def _queue_notification(self) -> bool:
        with self._lock:
            if self._notification_queued:
                return False
            self._notification_queued = True
            return True

    def _deliver_notification(self) -> None:
        try:
            self._on_done(self)
        except Exception:
            log.exception("trusted LiveKit tracked callback raised")


class TrustedLiveKitPublisherSession:
    """Own a pinned, media-only LiveKit Agents session for one publisher.

    The raw ``AgentSession``, media ``Agent``, ``RoomIO``, and output objects
    never leave this class.  The caller supplies one room plus explicitly
    audited local STT/VAD/TTS implementations.  SDK construction disables every
    framework LLM, tool, MCP, recording, hosted turn detector, and preemptive
    generation surface.  Public output mutation methods version a private route
    lease before touching the SDK.

    ``start`` and SDK-mutating methods must run on ``loop``. ``aclose`` is
    caller-bounded and reports wrapper-terminal proof only after manual RoomIO
    disposal plus SDK close when ``AgentSession.start`` completed. If pinned
    SDK start fails before its private ``_started`` flag, shutdown emits no
    close; the wrapper proves only its own media-route disposal and never raw
    AgentSession cleanup. A timeout leaves admission fenced and playback owned.
    """

    policy = PublisherSessionPolicy(
        publisher_scoped=True,
        recording_disabled=True,
        llm_disabled=True,
        agent_tools_disabled=True,
        session_tools_disabled=True,
        mcp_disabled=True,
        remote_inference_disabled=True,
        preemptive_generation_disabled=True,
        output_leases=True,
    )

    def __init__(
        self,
        *,
        room: object,
        participant_identity: str,
        speech: AuditedLocalSpeechComponents,
        loop: asyncio.AbstractEventLoop,
        close_timeout: float = _CLOSE_TIMEOUT_SEC,
        ready_timeout: float = _READY_TIMEOUT_SEC,
    ) -> None:
        if not isinstance(speech, AuditedLocalSpeechComponents):
            raise TypeError("speech must be AuditedLocalSpeechComponents")
        if loop is None:
            raise TypeError("loop is required")
        on = getattr(room, "on", None)
        off = getattr(room, "off", None)
        if not callable(on) or not callable(off):
            raise TypeError("room must provide on() and off()")
        timeout = float(close_timeout)
        if not 0.0 < timeout <= 30.0:
            raise ValueError("close_timeout must be in (0, 30] seconds")
        ready_limit = float(ready_timeout)
        if not 0.0 < ready_limit <= 30.0:
            raise ValueError("ready_timeout must be in (0, 30] seconds")

        self._sdk = _validated_livekit_agents_sdk()
        self._console = self._disabled_console()
        self._room = room
        self._participant_identity = _bounded_opaque_id(
            "participant identity", participant_identity
        )
        self._speech = speech
        self._loop = loop
        self._close_timeout = timeout
        self._ready_timeout = ready_limit
        self._lock = threading.RLock()
        self._handlers: dict[str, list[Callable[[Any], None]]] = {
            event: [] for event in _SDK_EVENTS
        }
        self._turn_committer: Callable[..., bool] | None = None
        self._started = False
        self._starting = False
        self._closing = False
        self._closed = False
        self._admission_fenced = False
        self._start_mutated = False
        self._media_start_inflight = False
        self._room_io_start_attempted = False
        self._room_io_started = False
        self._room_io_ready = False
        self._room_io_closed = False
        self._room_io_resources_closed = False
        self._session_start_attempted = False
        self._session_start_inflight = False
        self._session_started = False
        self._shutdown_requested = False
        self._sdk_close_seen = False
        self._sdk_close_payload: Any | None = None
        self._require_sdk_close = False
        self._cleanup_before_sdk_close = False
        self._close_reason = "user_initiated"
        self._session_generation = 0
        self._route_generation = 0
        self._lease_owner = object()
        self._audio_output: object | None = None
        self._output_attached = False
        self._output_enabled = False
        self._active_input_track_id: str | None = None
        self._connection_state: object | None = getattr(
            room,
            "connection_state",
            None,
        )
        self._tracked_handles: set[_TrustedSpeechHandle] = set()
        self._closed_event = asyncio.Event()
        self._sdk_close_event = asyncio.Event()
        self._media_start_settled_event = asyncio.Event()
        self._media_start_settled_event.set()
        self._session_start_settled_event = asyncio.Event()
        self._session_start_settled_event.set()
        self._finalizer_task: asyncio.Task[None] | None = None
        self._sdk_handlers: dict[str, Callable[[Any], None]] = {}
        self._room_handlers: dict[str, Callable[..., None]] = {}

        turn_handling = self._turn_handling()
        self._session = self._sdk.AgentSession(
            stt=speech.stt,
            vad=speech.vad,
            llm=None,
            tts=speech.tts,
            tools=[],
            mcp_servers=[],
            max_tool_steps=0,
            turn_handling=turn_handling,
            tts_text_transforms=None,
            use_tts_aligned_transcript=False,
            ivr_detection=False,
            user_away_timeout=None,
            aec_warmup_duration=None,
            loop=loop,
        )
        self._agent = self._build_media_agent()
        publish_options = self._sdk.TrackPublishOptions(
            source=self._sdk.microphone_source
        )
        self._room_options = self._sdk.RoomOptions(
            text_input=False,
            audio_input=self._sdk.AudioInputOptions(
                pre_connect_audio=False,
                pre_connect_audio_timeout=3.0,
                auto_gain_control=False,
            ),
            video_input=False,
            audio_output=self._sdk.AudioOutputOptions(
                track_publish_options=publish_options,
            ),
            text_output=False,
            participant_identity=self._participant_identity,
            close_on_disconnect=True,
            delete_room_on_close=False,
        )
        self._room_io = self._sdk.RoomIO(
            agent_session=self._session,
            room=self._room,
            options=self._room_options,
        )
        try:
            self._bind_sdk_events()
            self._bind_room_events()
        except BaseException:
            self._unbind_room_events()
            self._unbind_sdk_events()
            raise

    def _disabled_console(self, *, expected: object | None = None) -> object:
        get_instance = getattr(self._sdk.AgentsConsole, "get_instance", None)
        if not callable(get_instance):
            raise LiveKitAgentsDependencyError(
                "LiveKit Agents binding lacks AgentsConsole.get_instance()"
            )
        console = get_instance()
        if expected is not None and console is not expected:
            raise RuntimeError("LiveKit Agents console singleton changed")
        if not hasattr(console, "enabled") or not hasattr(console, "record"):
            raise LiveKitAgentsDependencyError(
                "LiveKit Agents console lacks reviewed enabled/record state"
            )
        if bool(console.enabled) or bool(console.record):
            raise RuntimeError(
                "trusted LiveKit publisher session requires AgentsConsole disabled"
            )
        return console

    def _assert_console_disabled(self) -> None:
        self._disabled_console(expected=self._console)

    def _assert_exact_manual_io(self) -> object:
        expected_input = getattr(self._room_io, "audio_input", None)
        expected_output = getattr(self._room_io, "audio_output", None)
        session_input = getattr(getattr(self._session, "input", None), "audio", None)
        output_owner = getattr(self._session, "output", None)
        output = getattr(output_owner, "audio", None)
        output_enabled = bool(getattr(output_owner, "audio_enabled", False))
        if (
            expected_input is None
            or expected_output is None
            or session_input is not expected_input
            or output is not expected_output
            or not output_enabled
        ):
            raise RuntimeError(
                "LiveKit manual RoomIO did not retain exact enabled media I/O"
            )
        return expected_output

    @staticmethod
    def _turn_handling() -> dict[str, object]:
        return {
            "turn_detection": "vad",
            "interruption": {
                "enabled": True,
                "mode": "vad",
            },
            "preemptive_generation": {"enabled": False},
        }

    def _build_media_agent(self) -> object:
        owner = self
        base = self._sdk.Agent

        class _PublisherMediaAgent(base):
            async def on_user_turn_completed(
                self,
                _turn_ctx: object,
                new_message: object,
            ) -> None:
                owner._commit_completed_turn(new_message)

        return _PublisherMediaAgent(
            instructions=(
                "Media-only bridge. Never answer, generate, call tools, or "
                "change application state."
            ),
            llm=None,
            tools=[],
            mcp_servers=[],
            turn_handling=self._turn_handling(),
        )

    @property
    def participant_identity(self) -> str:
        return self._participant_identity

    @property
    def local_speech_audit_id(self) -> str:
        return self._speech.audit_id

    @property
    def active_input_track_id(self) -> str | None:
        """Opaque diagnostic only; never speaker identity or authority."""

        with self._lock:
            return self._active_input_track_id

    @property
    def session_generation(self) -> int:
        with self._lock:
            return self._session_generation

    @property
    def route_generation(self) -> int:
        with self._lock:
            return self._route_generation

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def on(self, event: str, callback: Callable[[Any], None]) -> None:
        if event not in self._handlers:
            raise ValueError(f"unsupported LiveKit event: {event}")
        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._lock:
            if self._closing or self._closed:
                raise RuntimeError("trusted LiveKit publisher session is closing")
            if callback not in self._handlers[event]:
                self._handlers[event].append(callback)

    def off(self, event: str, callback: Callable[[Any], None]) -> None:
        if event not in self._handlers:
            return
        with self._lock:
            try:
                self._handlers[event].remove(callback)
            except ValueError:
                pass

    def bind_user_turn_committer(
        self,
        callback: Callable[..., bool],
    ) -> None:
        """Bind the engine's generation-scoped whole-turn committer."""

        if not callable(callback):
            raise TypeError("completed-turn committer must be callable")
        with self._lock:
            if self._closing or self._closed:
                raise RuntimeError("trusted LiveKit publisher session is closing")
            if self._turn_committer not in (None, callback):
                raise RuntimeError("completed-turn committer is already bound")
            self._turn_committer = callback

    def unbind_user_turn_committer(
        self,
        callback: Callable[..., bool],
    ) -> None:
        with self._lock:
            if self._turn_committer is callback:
                self._turn_committer = None

    async def start(self) -> None:
        """Start manual RoomIO without activating SDK room/control hosting."""

        self._require_loop()
        with self._lock:
            if self._closed or self._closing or self._admission_fenced:
                raise RuntimeError("trusted LiveKit publisher session is closed")
        self._assert_console_disabled()
        with self._lock:
            if self._closed or self._closing or self._admission_fenced:
                raise RuntimeError("trusted LiveKit publisher session is closed")
            if self._started or self._starting:
                raise RuntimeError(
                    "trusted LiveKit publisher session is already started"
                )
            if self._turn_committer is None:
                raise RuntimeError(
                    "completed-turn committer must be bound before media start"
                )
            if self._connection_state != self._sdk.connection_connected:
                raise RuntimeError(
                    "trusted LiveKit room must be connected before media start"
                )
            self._starting = True
            self._start_mutated = True
            self._media_start_inflight = True
            self._media_start_settled_event.clear()
            self._room_io_start_attempted = True
        try:
            await self._room_io.start()
            with self._lock:
                self._room_io_started = True
                if (
                    self._closing
                    or self._closed
                    or self._connection_state != self._sdk.connection_connected
                ):
                    raise RuntimeError(
                        "LiveKit room disconnected while media was starting"
                    )
            # AgentSession.start() consults this singleton synchronously before
            # its first await and can replace our manual I/O or create control
            # and recorder paths when enabled. Re-check the exact object as the
            # final synchronous guard before entering that public method.
            self._assert_console_disabled()
            with self._lock:
                self._session_start_attempted = True
                self._session_start_inflight = True
                self._session_start_settled_event.clear()
            await self._session.start(
                self._agent,
                record=False,
                capture_run=False,
            )
            with self._lock:
                self._session_start_inflight = False
                self._session_started = True
                self._require_sdk_close = True
                self._session_start_settled_event.set()
                if (
                    self._closing
                    or self._closed
                    or self._connection_state != self._sdk.connection_connected
                ):
                    raise RuntimeError(
                        "LiveKit room disconnected while media was starting"
                    )
            # Detect console or output mutation performed inside the SDK start
            # call before waiting for readiness or admitting any speech.
            self._assert_console_disabled()
            self._assert_exact_manual_io()
            await asyncio.wait_for(
                self._room_io.wait_for_ready(),
                timeout=self._ready_timeout,
            )
            self._assert_console_disabled()
            output = self._assert_exact_manual_io()
        except BaseException:
            with self._lock:
                self._session_start_inflight = False
                self._session_start_settled_event.set()
            await self._abort_start()
            raise

        with self._lock:
            invalidated = bool(
                self._closed
                or self._closing
                or self._connection_state != self._sdk.connection_connected
            )
            if not invalidated:
                self._room_io_ready = True
                self._audio_output = output
                self._output_attached = True
                self._output_enabled = True
                self._started = True
                self._starting = False
                self._media_start_inflight = False
                self._media_start_settled_event.set()
                self._session_generation += 1
                self._route_generation += 1
        if invalidated:
            await self._abort_start()
            raise RuntimeError("LiveKit session closed while it was starting")

    async def aclose(
        self,
        *,
        drain: bool = True,
        timeout: float | None = None,
    ) -> bool:
        """Fence and await shared wrapper disposal proof.

        A successful SDK start additionally requires its close event. A start
        exception before the pinned SDK marks itself started cannot produce
        that event, so completion then proves only that Speaker detached and
        closed its manual RoomIO; no speech was admitted in that state.
        """

        self._require_loop()
        limit = self._close_timeout if timeout is None else float(timeout)
        if not 0.0 < limit <= 30.0:
            raise ValueError("close timeout must be in (0, 30] seconds")
        with self._lock:
            if self._closed:
                return True
            self._fence_locked()
            self._require_sdk_close = self._require_sdk_close or self._session_started
        self._request_shutdown(drain=bool(drain))
        finalizer = self._ensure_finalizer()
        try:
            await asyncio.wait_for(
                asyncio.shield(finalizer),
                timeout=limit,
            )
        except Exception:
            return False
        with self._lock:
            return self._closed

    def say_tracked(
        self,
        text: str,
        *,
        allow_interruptions: bool,
        add_to_chat_ctx: bool,
        on_done: Callable[[SpeechHandleLike], None],
    ) -> tuple[SpeechHandleLike, object]:
        self._require_loop()
        clean = str(text or "")
        if not clean.strip():
            raise ValueError("tracked speech text must not be empty")
        if add_to_chat_ctx:
            raise ValueError("trusted media speech may not enter SDK chat context")
        if allow_interruptions is not True:
            raise ValueError("trusted media speech must remain interruptible")
        if not callable(on_done):
            raise TypeError("on_done must be callable")
        with self._lock:
            if not self._admission_allowed_locked():
                raise RuntimeError("LiveKit audio output is unavailable")
            lease = _PublisherOutputLease(
                self._lease_owner,
                self._session_generation,
                self._route_generation,
            )
            proxy = _TrustedSpeechHandle(self, on_done)
            self._tracked_handles.add(proxy)

        try:
            sdk_handle = self._session.say(
                clean,
                allow_interruptions=True,
                add_to_chat_ctx=False,
            )
            proxy._bind_sdk_handle(sdk_handle)
        except BaseException as exc:
            # Once SDK admission was attempted, absence of a returned handle is
            # not proof that the sink stayed quiet. Retain the proxy until the
            # wrapper's SDK-plus-RoomIO disposal event instead of raising it out
            # of ownership.
            proxy._mark_admission_error(exc)
            self._begin_failure_close()
        return proxy, lease

    def output_lease_valid(self, lease: object) -> bool:
        with self._lock:
            return bool(
                isinstance(lease, _PublisherOutputLease)
                and lease.owner is self._lease_owner
                and lease.session_generation == self._session_generation
                and lease.route_generation == self._route_generation
                and self._admission_allowed_locked()
            )

    def replace_audio_tail(self, sink: object) -> None:
        """Reject tail replacement that would displace the manual RoomIO sink.

        In the pinned SDK, ``AgentOutput.replace_audio_tail()`` falls back to
        replacing ``output.audio`` when no internal proxy is present. This
        composition deliberately has no recorder/transcript proxy, so invoking
        it would remove the exact participant-scoped RoomIO publisher.
        """

        self._require_loop()
        if sink is None:
            raise TypeError("audio tail sink is required")
        with self._lock:
            if not self._route_owned_locked():
                raise RuntimeError("LiveKit audio output is unavailable")
            # Conservatively expire handles whose callers requested a route
            # change, even though the unsafe SDK mutation is never invoked.
            self._route_generation += 1
        raise RuntimeError(
            "LiveKit audio-tail replacement is unsafe for exact manual RoomIO"
        )

    def set_audio_enabled(self, enabled: bool) -> None:
        """Own every enable transition so a disable/re-enable cannot reuse a lease."""

        self._require_loop()
        value = bool(enabled)
        output_owner = getattr(self._session, "output", None)
        with self._lock:
            if not self._route_owned_locked():
                raise RuntimeError("LiveKit audio output is unavailable")
            if value == self._output_enabled:
                return
            self._route_generation += 1
            expected_output = self._audio_output
        try:
            setter = getattr(output_owner, "set_audio_enabled", None)
            if not callable(setter):
                raise RuntimeError("LiveKit AgentOutput lacks set_audio_enabled()")
            setter(value)
            if (
                getattr(output_owner, "audio", None) is not expected_output
                or bool(getattr(output_owner, "audio_enabled", not value)) is not value
            ):
                raise RuntimeError("LiveKit audio enable transition was not exact")
        except BaseException:
            self._begin_failure_close()
            raise
        with self._lock:
            self._output_enabled = value

    def detach_audio_output(self) -> None:
        """Detach SDK audio without exposing the raw ``AgentOutput``."""

        self._require_loop()
        output_owner = getattr(self._session, "output", None)
        with self._lock:
            if not self._route_owned_locked():
                raise RuntimeError("LiveKit audio output is unavailable")
            self._route_generation += 1
        try:
            if output_owner is None:
                raise RuntimeError("LiveKit AgentOutput is unavailable")
            setattr(output_owner, "audio", None)
            if getattr(output_owner, "audio", None) is not None:
                raise RuntimeError("LiveKit audio output did not detach")
        except BaseException:
            self._begin_failure_close()
            raise
        with self._lock:
            self._output_attached = False
            self._output_enabled = False
            self._audio_output = None

    def close_on_playout_failure(self) -> None:
        """Fence admission, clear queued media, and request non-draining close."""

        self._require_loop()
        self._begin_failure_close()

    def _admission_allowed_locked(self) -> bool:
        return bool(
            self._route_owned_locked()
            and self._output_enabled
            and bool(
                getattr(getattr(self._session, "output", None), "audio_enabled", False)
            )
        )

    def _input_events_admitted_locked(self) -> bool:
        """Whether SDK input/state events may cross the wrapper boundary."""

        return bool(
            self._started
            and not self._starting
            and self._room_io_ready
            and self._connection_state == self._sdk.connection_connected
            and not self._admission_fenced
            and not self._closing
            and not self._closed
        )

    def _route_owned_locked(self) -> bool:
        try:
            output_owner = getattr(self._session, "output", None)
            exact_output = getattr(output_owner, "audio", None) is self._audio_output
        except Exception:
            return False
        return bool(
            self._started
            and not self._starting
            and not self._closing
            and not self._closed
            and not self._admission_fenced
            and self._connection_state == self._sdk.connection_connected
            and self._room_io_ready
            and self._output_attached
            and self._audio_output is not None
            and exact_output
        )

    def _fence_locked(self) -> None:
        first_fence = not self._admission_fenced
        self._admission_fenced = True
        self._closing = not self._closed
        self._started = False
        self._starting = False
        self._room_io_ready = False
        self._output_attached = False
        self._output_enabled = False
        self._audio_output = None
        self._active_input_track_id = None
        if first_fence:
            self._session_generation += 1
            self._route_generation += 1

    async def _abort_start(self) -> None:
        with self._lock:
            self._close_reason = "error"
            self._cleanup_before_sdk_close = True
            self._require_sdk_close = self._require_sdk_close or self._session_started
            self._fence_locked()
            self._media_start_inflight = False
            self._media_start_settled_event.set()
        self._request_shutdown(drain=False)
        finalizer = self._ensure_finalizer()
        try:
            await asyncio.wait_for(
                asyncio.shield(finalizer),
                timeout=self._close_timeout,
            )
        except Exception:
            log.warning("LiveKit rejected-start cleanup remains pending")

    def _request_shutdown(self, *, drain: bool) -> None:
        with self._lock:
            if (
                not self._session_start_attempted
                or self._session_start_inflight
                or self._sdk_close_seen
                or self._shutdown_requested
            ):
                return
            self._shutdown_requested = True
        try:
            self._session.shutdown(drain=drain)
        except Exception:
            with self._lock:
                if not self._sdk_close_seen:
                    self._shutdown_requested = False
            log.exception("LiveKit session shutdown request failed")

    def _ensure_finalizer(self) -> asyncio.Task[None]:
        with self._lock:
            task = self._finalizer_task
            if task is None or (task.done() and not self._closed):
                task = self._loop.create_task(self._finalize_close())
                task.add_done_callback(self._observe_finalizer)
                self._finalizer_task = task
            return task

    @staticmethod
    def _observe_finalizer(task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        try:
            error = task.exception()
        except asyncio.CancelledError:
            return
        if error is not None:
            log.warning(
                "LiveKit manual RoomIO finalizer failed: %s",
                type(error).__name__,
            )

    async def _finalize_close(self) -> None:
        with self._lock:
            media_start_inflight = self._media_start_inflight
            session_start_inflight = self._session_start_inflight
        if media_start_inflight:
            await self._media_start_settled_event.wait()
        if session_start_inflight:
            await self._session_start_settled_event.wait()
        with self._lock:
            cleanup_first = self._cleanup_before_sdk_close
            start_mutated = self._start_mutated
        if start_mutated and cleanup_first:
            await self._close_manual_room_io()
        with self._lock:
            require_sdk_close = self._require_sdk_close
            sdk_close_seen = self._sdk_close_seen
        if require_sdk_close and not sdk_close_seen:
            await self._sdk_close_event.wait()
        if start_mutated:
            await self._close_manual_room_io()
        self._publish_terminal_close()

    async def _close_manual_room_io(self) -> None:
        with self._lock:
            if self._room_io_closed:
                return
            resources_closed = self._room_io_resources_closed
        errors: list[BaseException] = []
        session_input = getattr(self._session, "input", None)
        session_output = getattr(self._session, "output", None)
        try:
            if getattr(session_input, "audio", None) is not None:
                setattr(session_input, "audio", None)
        except BaseException as exc:
            errors.append(exc)
        try:
            if getattr(session_output, "audio", None) is not None:
                setattr(session_output, "audio", None)
        except BaseException as exc:
            errors.append(exc)
        if not resources_closed:
            try:
                await self._room_io.aclose()
            except BaseException as exc:
                errors.append(exc)
            else:
                with self._lock:
                    self._room_io_resources_closed = True
        try:
            if (
                getattr(session_input, "audio", None) is not None
                or getattr(session_output, "audio", None) is not None
            ):
                raise RuntimeError("LiveKit manual RoomIO remained attached")
        except BaseException as exc:
            errors.append(exc)
        if errors:
            raise RuntimeError("LiveKit manual RoomIO disposal failed") from errors[0]
        with self._lock:
            self._room_io_closed = True
            self._room_io_ready = False

    def _publish_terminal_close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._start_mutated and not self._room_io_closed:
                raise RuntimeError("manual RoomIO disposal proof is incomplete")
            if self._require_sdk_close and not self._sdk_close_seen:
                raise RuntimeError("LiveKit SDK close proof is incomplete")
            self._closed = True
            self._closing = False
            self._started = False
            self._starting = False
            proxies = tuple(self._tracked_handles)
            callbacks = tuple(self._handlers["close"])
            event = self._sdk_close_payload or SimpleNamespace(
                reason=SimpleNamespace(value=self._close_reason)
            )
        for proxy in proxies:
            proxy._finish_for_session_close()
        for callback in callbacks:
            try:
                callback(event)
            except Exception:
                log.exception("trusted LiveKit close callback raised")
        self._unbind_room_events()
        self._unbind_sdk_events()
        self._closed_event.set()

    def _interrupt_tracked_handle(
        self,
        proxy: _TrustedSpeechHandle,
        *,
        force: bool,
    ) -> None:
        self._require_loop()
        sdk_handle = proxy._sdk_handle_for_interrupt()
        if sdk_handle is None:
            return
        interrupt_error: BaseException | None = None
        interrupted = False
        try:
            result = sdk_handle.interrupt(force=force)
            interrupted = result is not None or bool(
                getattr(sdk_handle, "interrupted", False)
            )
        except BaseException as exc:
            interrupt_error = exc
            try:
                interrupted = bool(getattr(sdk_handle, "interrupted", False))
            except Exception:
                interrupted = False
        clear_error: BaseException | None = None
        try:
            self._clear_playout_queue()
        except BaseException as exc:
            clear_error = exc
        proxy._mark_interrupt_result(interrupted=interrupted)
        failure = interrupt_error or clear_error
        if failure is not None:
            self._begin_failure_close()
            raise failure

    def _clear_playout_queue(self) -> None:
        with self._lock:
            output = self._audio_output
        self._clear_output_queue(output)

    @staticmethod
    def _clear_output_queue(output: object | None) -> None:
        if output is None:
            return
        clear = getattr(output, "clear_buffer", None)
        if not callable(clear):
            raise RuntimeError("LiveKit audio output lacks clear_buffer()")
        clear()

    def _begin_failure_close(self) -> None:
        self._require_loop()
        with self._lock:
            first_request = not self._closing and not self._closed
            output = self._audio_output
            if first_request:
                self._close_reason = "error"
                self._require_sdk_close = (
                    self._require_sdk_close or self._session_started
                )
                self._fence_locked()
        clear_error: BaseException | None = None
        try:
            self._clear_output_queue(output)
        except BaseException as exc:
            clear_error = exc
        if first_request:
            self._request_shutdown(drain=False)
            self._ensure_finalizer()
        if clear_error is not None:
            log.warning(
                "LiveKit queued playout clear failed: %s",
                type(clear_error).__name__,
            )

    def _tracked_handle_terminal(self, proxy: _TrustedSpeechHandle) -> None:
        with self._lock:
            self._tracked_handles.discard(proxy)
        if not proxy._queue_notification():
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        try:
            if running is self._loop:
                self._loop.call_soon(proxy._deliver_notification)
            else:
                self._loop.call_soon_threadsafe(proxy._deliver_notification)
        except RuntimeError:
            log.warning("LiveKit tracked terminal lost its closed event loop")

    def _commit_completed_turn(self, message: object) -> None:
        text = getattr(message, "text_content", None)
        if callable(text):
            text = text()
        try:
            turn_id = _bounded_opaque_id(
                "completed turn id",
                getattr(message, "id", ""),
            )
        except ValueError:
            log.warning("LiveKit completed turn lacked a stable bounded ID")
            return
        with self._lock:
            committer = self._turn_committer
            admitted = self._input_events_admitted_locked()
        if committer is None or not admitted:
            return
        try:
            committer(str(text or ""), turn_id=turn_id)
        except Exception:
            log.exception("LiveKit completed-turn commit failed")

    def _bind_sdk_events(self) -> None:
        for event in _SDK_EVENTS:
            if event == "close":
                callback = self._on_sdk_close
            else:
                callback = self._sdk_event_forwarder(event)
            self._sdk_handlers[event] = callback
            self._session.on(event, callback)

    def _sdk_event_forwarder(self, event: str) -> Callable[[Any], None]:
        def forward(payload: Any) -> None:
            self._emit_sdk_event(event, payload)

        return forward

    def _unbind_sdk_events(self) -> None:
        handlers = tuple(self._sdk_handlers.items())
        self._sdk_handlers.clear()
        for event, callback in handlers:
            try:
                self._session.off(event, callback)
            except Exception:
                log.debug("failed to unbind SDK %s callback", event, exc_info=True)

    def _bind_room_events(self) -> None:
        handlers: dict[str, Callable[..., None]] = {
            "track_subscribed": self._on_track_subscribed,
            "track_unpublished": self._on_track_unpublished,
            "participant_disconnected": self._on_participant_disconnected,
            "connection_state_changed": self._on_connection_state_changed,
        }
        for event, callback in handlers.items():
            self._room_handlers[event] = callback
            self._room.on(event, callback)

    def _unbind_room_events(self) -> None:
        handlers = tuple(self._room_handlers.items())
        self._room_handlers.clear()
        for event, callback in handlers:
            try:
                self._room.off(event, callback)
            except Exception:
                log.debug("failed to unbind room %s callback", event, exc_info=True)

    def _emit_sdk_event(self, event: str, payload: Any) -> None:
        with self._lock:
            if not self._input_events_admitted_locked():
                return
            callbacks = tuple(self._handlers[event])
        for callback in callbacks:
            try:
                callback(payload)
            except Exception:
                log.exception("trusted LiveKit %s callback raised", event)

    def _on_sdk_close(self, event: Any) -> None:
        with self._lock:
            if self._closed:
                return
            if not self._sdk_close_seen:
                self._sdk_close_seen = True
                self._sdk_close_payload = event
                self._sdk_close_event.set()
            self._fence_locked()
        self._ensure_finalizer()

    def _on_track_subscribed(
        self,
        track: object,
        publication: object,
        participant: object,
    ) -> None:
        if str(getattr(participant, "identity", "")) != self._participant_identity:
            return
        if getattr(publication, "source", None) != self._sdk.microphone_source:
            return
        raw_track_id = getattr(publication, "sid", None) or getattr(track, "sid", None)
        try:
            track_id = _bounded_opaque_id("LiveKit input track id", raw_track_id)
        except ValueError:
            return
        with self._lock:
            if self._closed or self._closing or track_id == self._active_input_track_id:
                return
            self._active_input_track_id = track_id
            self._session_generation += 1

    def _on_track_unpublished(
        self,
        publication: object,
        participant: object,
    ) -> None:
        if str(getattr(participant, "identity", "")) != self._participant_identity:
            return
        if getattr(publication, "source", None) != self._sdk.microphone_source:
            return
        try:
            track_id = _bounded_opaque_id(
                "LiveKit input track id",
                getattr(publication, "sid", None),
            )
        except ValueError:
            return
        with self._lock:
            if self._closed or track_id != self._active_input_track_id:
                return
            self._active_input_track_id = None
            self._session_generation += 1

    def _on_participant_disconnected(self, participant: object) -> None:
        """Make every post-mutation bound-publisher disconnect terminal."""

        if str(getattr(participant, "identity", "")) != self._participant_identity:
            return
        output: object | None = None
        trigger_close = False
        with self._lock:
            if self._closed or self._closing or not self._start_mutated:
                return
            output = self._audio_output
            self._close_reason = "participant_disconnected"
            self._require_sdk_close = self._require_sdk_close or self._session_started
            self._fence_locked()
            trigger_close = True
        if trigger_close:
            self._schedule_transport_loss_close(output)

    def _on_connection_state_changed(self, state: object) -> None:
        output: object | None = None
        trigger_close = False
        with self._lock:
            if self._closed or state == self._connection_state:
                return
            self._connection_state = state
            self._active_input_track_id = None
            self._session_generation += 1
            if (
                state != self._sdk.connection_connected
                and self._start_mutated
                and not self._closing
            ):
                output = self._audio_output
                self._close_reason = "error"
                self._require_sdk_close = (
                    self._require_sdk_close or self._session_started
                )
                self._fence_locked()
                trigger_close = True
        if trigger_close:
            self._schedule_transport_loss_close(output)

    def _schedule_transport_loss_close(self, output: object | None) -> None:
        def close_failed_transport() -> None:
            try:
                self._clear_output_queue(output)
            except Exception:
                log.warning("LiveKit disconnected output queue did not clear")
            self._request_shutdown(drain=False)
            self._ensure_finalizer()

        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        try:
            if running is self._loop:
                close_failed_transport()
            else:
                self._loop.call_soon_threadsafe(close_failed_transport)
        except RuntimeError:
            log.warning("LiveKit transport-loss close could not reach its loop")

    def _require_loop(self) -> None:
        try:
            running = asyncio.get_running_loop()
        except RuntimeError as exc:
            raise RuntimeError(
                "trusted LiveKit session operation must run on its job loop"
            ) from exc
        if running is not self._loop:
            raise RuntimeError(
                "trusted LiveKit session operation must run on its job loop"
            )


__all__ = [
    "AuditedLocalSpeechComponents",
    "LiveKitAgentsDependencyError",
    "TrustedLiveKitPublisherSession",
]
