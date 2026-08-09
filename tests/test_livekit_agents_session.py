"""Headless adversarial tests for the trusted LiveKit Agents SDK owner."""

from __future__ import annotations

import asyncio
from collections import defaultdict
import inspect
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from core.engine import EngineCallbacks, PlaybackOutcome, TrackedSpeech
from core.engines.livekit_agents import LiveKitAgentsEngine
import core.engines.livekit_agents_session as publisher_session
from core.engines.livekit_agents_session import (
    AuditedLocalSpeechComponents,
    LiveKitAgentsDependencyError,
    TrustedLiveKitPublisherSession,
)


_MICROPHONE = 2
_CAMERA = 1
_CONNECTED = "connected"
_DISCONNECTED = "disconnected"


class _Emitter:
    def __init__(self) -> None:
        self.handlers = defaultdict(list)
        self.fail_on_event = None

    def on(self, event, callback):
        self.handlers[event].append(callback)
        if event == self.fail_on_event:
            raise RuntimeError(f"partial bind failed: {event}")

    def off(self, event, callback):
        try:
            self.handlers[event].remove(callback)
        except ValueError:
            pass

    def emit(self, event, *args):
        for callback in tuple(self.handlers[event]):
            callback(*args)


class _Options:
    def __init__(self, **kwargs) -> None:
        vars(self).update(kwargs)


class _AudioSink:
    def __init__(self) -> None:
        self.clear_calls = 0
        self.clear_error = None

    def clear_buffer(self) -> None:
        self.clear_calls += 1
        if self.clear_error is not None:
            raise self.clear_error


class _AgentInput:
    def __init__(self) -> None:
        self.audio = None


class _AgentOutput:
    def __init__(self) -> None:
        self._audio = None
        self.audio_enabled = True
        self.audio_set_error = None
        self.audio_set_partial = False
        self.tail_replacements = []
        self.enabled_error = None
        self.enabled_partial = False
        self.enabled_changes = []

    @property
    def audio(self):
        return self._audio

    @audio.setter
    def audio(self, value) -> None:
        error = self.audio_set_error
        if error is not None and not self.audio_set_partial:
            raise error
        self._audio = value
        if error is not None:
            raise error

    def replace_audio_tail(self, sink) -> None:
        self.tail_replacements.append(sink)
        # Exact livekit-agents 1.6.8 behavior with no output proxy: replacing
        # the "tail" falls back to replacing the AgentOutput head.
        self.audio = sink

    def set_audio_enabled(self, enabled) -> None:
        value = bool(enabled)
        error = self.enabled_error
        if error is not None and not self.enabled_partial:
            raise error
        self.audio_enabled = value
        self.enabled_changes.append(value)
        if error is not None:
            raise error


class _SpeechHandle:
    def __init__(self) -> None:
        self.interrupted = False
        self.error = None
        self.done = False
        self.callbacks = []
        self.interrupt_calls = []
        self.interrupt_error = None

    def add_done_callback(self, callback) -> None:
        self.callbacks.append(callback)
        if self.done:
            callback(self)

    def interrupt(self, *, force=False):
        self.interrupt_calls.append(force)
        if self.interrupt_error is not None:
            raise self.interrupt_error
        self.interrupted = True
        return self

    def exception(self):
        if not self.done:
            raise asyncio.InvalidStateError
        return self.error

    def finish(self, *, interrupted=False, error=None) -> None:
        if self.done:
            return
        self.done = True
        self.interrupted = self.interrupted or bool(interrupted)
        self.error = error
        self.notify_done_again()

    def notify_done_again(self) -> None:
        for callback in tuple(self.callbacks):
            callback(self)


class _Room(_Emitter):
    def __init__(self, state=_CONNECTED) -> None:
        super().__init__()
        self.connection_state = state

    def emit(self, event, *args):
        if event == "connection_state_changed":
            self.connection_state = args[0]
        super().emit(event, *args)


def _fake_sdk():
    captured = SimpleNamespace(
        sessions=[],
        agents=[],
        room_ios=[],
        operations=[],
        forbidden=[],
        session_fail_on=None,
        console=SimpleNamespace(enabled=False, record=False),
        console_toggle_after_room_start=False,
        console_toggle_during_session_start=False,
        replace_output_during_session_start=False,
    )

    class AgentsConsole:
        @classmethod
        def get_instance(cls):
            return captured.console

    class Agent:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            captured.agents.append(self)

    class AgentSession(_Emitter):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            self.fail_on_event = captured.session_fail_on
            self.kwargs = kwargs
            self.input = _AgentInput()
            self.output = _AgentOutput()
            self.start_calls = []
            self.start_error = None
            self.start_gate = None
            self.partial_start_entered = False
            self.started_agent = None
            self.started = False
            self.say_calls = []
            self.say_error = None
            self.handles = []
            self.immediate_done = False
            self.shutdown_calls = []
            self.shutdown_error = None
            self.auto_close = True
            self.close_events = 0
            self.participant_link_calls = []
            captured.sessions.append(self)

        def _on_room_io_participant_linked(self, participant) -> None:
            self.participant_link_calls.append(participant)

        async def start(self, agent, **kwargs) -> None:
            captured.operations.append("session.start")
            self.started_agent = agent
            self.start_calls.append(kwargs)
            if captured.console_toggle_during_session_start:
                captured.console.enabled = True
            if "room" in kwargs or "room_options" in kwargs:
                captured.forbidden.extend(["RoomSessionTransport", "SessionHost"])
            if captured.console.enabled:
                captured.forbidden.extend(["AgentsConsole", "SessionHost"])
            if kwargs.get("record") is not False or (
                captured.console.enabled and captured.console.record
            ):
                captured.forbidden.append("RecorderIO")
            # Model the pinned SDK's internal mutations and awaited activity
            # setup before its private _started flag becomes true.
            self.partial_start_entered = True
            if self.start_gate is not None:
                await self.start_gate.wait()
            if self.start_error is not None:
                raise self.start_error
            self.started = True
            if captured.replace_output_during_session_start:
                self.output.audio = object()

        def say(self, text, **kwargs):
            self.say_calls.append((text, kwargs))
            if self.say_error is not None:
                raise self.say_error
            handle = _SpeechHandle()
            self.handles.append(handle)
            if self.immediate_done:
                handle.finish()
            return handle

        def shutdown(self, *, drain=True) -> None:
            self.shutdown_calls.append(bool(drain))
            if self.shutdown_error is not None:
                raise self.shutdown_error
            if self.started and self.auto_close:
                self.emit_close()

        def emit_close(self, reason="user_initiated") -> None:
            self.close_events += 1
            self.started = False
            self.input.audio = None
            self.output._audio = None
            self.emit(
                "close",
                SimpleNamespace(reason=SimpleNamespace(value=reason)),
            )

    class RoomIO:
        def __init__(self, agent_session, room, *, options) -> None:
            self.agent_session = agent_session
            self.room = room
            self.options = options
            self.audio_input = object()
            self.audio_output = _AudioSink()
            self.start_calls = 0
            self.start_error = None
            self.start_gate = None
            self.auto_ready = True
            self.ready_error = None
            self.ready_event = asyncio.Event()
            self.wait_calls = 0
            self.close_calls = 0
            self.close_errors = []
            self.close_gate = None
            self.closed = False
            captured.room_ios.append(self)

        async def start(self) -> None:
            captured.operations.append("room_io.start")
            self.start_calls += 1
            if self.options.audio_input.pre_connect_audio:
                captured.forbidden.append("PreConnectAudioHandler")
            self.agent_session.input.audio = self.audio_input
            self.agent_session.output.audio = self.audio_output
            self.agent_session._on_room_io_participant_linked(
                SimpleNamespace(identity=self.options.participant_identity)
            )
            if self.start_gate is not None:
                await self.start_gate.wait()
            if self.start_error is not None:
                raise self.start_error
            if self.auto_ready:
                self.ready_event.set()
            if captured.console_toggle_after_room_start:
                captured.console.enabled = True

        async def wait_for_ready(self) -> None:
            captured.operations.append("room_io.wait_for_ready")
            self.wait_calls += 1
            if self.ready_error is not None:
                raise self.ready_error
            await self.ready_event.wait()
            if self.ready_error is not None:
                raise self.ready_error

        async def aclose(self) -> None:
            captured.operations.append("room_io.aclose")
            self.close_calls += 1
            if self.close_gate is not None:
                await self.close_gate.wait()
            if self.close_errors:
                raise self.close_errors.pop(0)
            self.closed = True

    sdk = SimpleNamespace(
        agents_version="1.6.8",
        rtc_version="1.1.14",
        api_version="1.2.0",
        protocol_version="1.1.21",
        Agent=Agent,
        AgentSession=AgentSession,
        RoomIO=RoomIO,
        RoomOptions=_Options,
        AudioInputOptions=_Options,
        AudioOutputOptions=_Options,
        AgentsConsole=AgentsConsole,
        TrackPublishOptions=_Options,
        microphone_source=_MICROPHONE,
        connection_connected=_CONNECTED,
        captured=captured,
    )
    return sdk


def _speech() -> AuditedLocalSpeechComponents:
    return AuditedLocalSpeechComponents(
        stt=object(),
        vad=object(),
        tts=object(),
        audit_id="speaker-local-test",
    )


def _wrapper(
    monkeypatch,
    sdk,
    room,
    loop,
    *,
    close_timeout=0.05,
    ready_timeout=0.05,
) -> TrustedLiveKitPublisherSession:
    monkeypatch.setattr(publisher_session, "_load_livekit_agents_sdk", lambda: sdk)
    return TrustedLiveKitPublisherSession(
        room=room,
        participant_identity="publisher-7",
        speech=_speech(),
        loop=loop,
        close_timeout=close_timeout,
        ready_timeout=ready_timeout,
    )


def _bind_committer(wrapper) -> None:
    wrapper.bind_user_turn_committer(lambda _text, *, turn_id=None: bool(turn_id))


def _participant(identity="publisher-7", *, disconnect_reason=None):
    return SimpleNamespace(identity=identity, disconnect_reason=disconnect_reason)


def _publication(sid, source=_MICROPHONE):
    return SimpleNamespace(sid=sid, source=source)


async def _emit_input_event_burst(sdk, label: str) -> None:
    session = sdk.captured.sessions[0]
    agent = sdk.captured.agents[0]
    session.emit("agent_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit("user_state_changed", SimpleNamespace(new_state="speaking"))
    session.emit(
        "user_input_transcribed",
        SimpleNamespace(transcript=f"{label} partial", is_final=False),
    )
    session.emit(
        "user_input_transcribed",
        SimpleNamespace(transcript=f"{label} final", is_final=True),
    )
    session.emit("agent_false_interruption", SimpleNamespace(resumed=True))
    await agent.on_user_turn_completed(
        None,
        SimpleNamespace(id=f"{label}-message", text_content=f"{label} final"),
    )


def test_builds_exact_manual_room_io_without_implicit_control_paths(monkeypatch):
    async def scenario():
        sdk, room = _fake_sdk(), _Room()
        wrapper = _wrapper(monkeypatch, sdk, room, asyncio.get_running_loop())
        session = sdk.captured.sessions[0]
        room_io = sdk.captured.room_ios[0]
        agent = sdk.captured.agents[0]
        _bind_committer(wrapper)

        assert room_io.agent_session is session
        assert room_io.room is room
        assert room_io.options.participant_identity == "publisher-7"
        assert room_io.options.audio_input.pre_connect_audio is False
        assert room_io.options.audio_input.auto_gain_control is False
        assert room_io.options.text_input is False
        assert room_io.options.video_input is False
        assert room_io.options.text_output is False
        assert room_io.options.audio_output.track_publish_options.source == _MICROPHONE
        assert session.kwargs["aec_warmup_duration"] is None

        await wrapper.start()
        assert len(session.participant_link_calls) == 1
        assert session.participant_link_calls[0].identity == "publisher-7"
        assert session.start_calls == [{"record": False, "capture_run": False}]
        assert session.started_agent is agent
        assert session.input.audio is room_io.audio_input
        assert session.output.audio is room_io.audio_output
        assert session.output.audio_enabled is True
        assert sdk.captured.operations[:3] == [
            "room_io.start",
            "session.start",
            "room_io.wait_for_ready",
        ]
        assert sdk.captured.forbidden == []
        assert await wrapper.aclose()
        assert room_io.closed

    asyncio.run(scenario())


def test_constructor_has_no_public_arbitrary_sdk_injection():
    assert "sdk" not in inspect.signature(TrustedLiveKitPublisherSession).parameters


@pytest.mark.parametrize("missing_metadata", [False, True])
def test_loader_binds_or_rejects_api_and_protocol_distribution_metadata(
    monkeypatch, missing_metadata
):
    livekit = ModuleType("livekit")
    livekit.__path__ = []
    agents = ModuleType("livekit.agents")
    agents.__version__ = "1.6.8"
    rtc = ModuleType("livekit.rtc")
    rtc.__version__ = "1.1.14"
    livekit.agents = agents
    livekit.rtc = rtc
    monkeypatch.setitem(sys.modules, "livekit", livekit)
    monkeypatch.setitem(sys.modules, "livekit.agents", agents)
    monkeypatch.setitem(sys.modules, "livekit.rtc", rtc)

    versions = {"livekit-api": "1.2.0", "livekit-protocol": "1.1.21"}

    def version(name):
        if missing_metadata:
            raise publisher_session.importlib_metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(publisher_session.importlib_metadata, "version", version)
    if missing_metadata:
        with pytest.raises(LiveKitAgentsDependencyError, match="requirements-remote"):
            publisher_session._load_livekit_agents_sdk()
        return

    loaded = publisher_session._load_livekit_agents_sdk()
    assert loaded.agents_version == "1.6.8"
    assert loaded.rtc_version == "1.1.14"
    assert loaded.api_version == "1.2.0"
    assert loaded.protocol_version == "1.1.21"


def test_remote_requirements_pin_the_reviewed_sdk_closure():
    requirements_path = Path(__file__).resolve().parents[1] / "requirements-remote.txt"
    pins = {}
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        requirement = raw_line.partition("#")[0].strip()
        name, separator, version = requirement.partition("==")
        if name not in {
            "livekit",
            "livekit-agents",
            "livekit-api",
            "livekit-protocol",
        }:
            continue
        assert separator == "=="
        assert name not in pins
        pins[name] = version

    assert publisher_session._LIVEKIT_RTC_VERSION == "1.1.14"
    assert publisher_session._LIVEKIT_AGENTS_VERSION == "1.6.8"
    assert publisher_session._LIVEKIT_API_VERSION == "1.2.0"
    assert publisher_session._LIVEKIT_PROTOCOL_VERSION == "1.1.21"
    assert pins == {
        "livekit": publisher_session._LIVEKIT_RTC_VERSION,
        "livekit-agents": publisher_session._LIVEKIT_AGENTS_VERSION,
        "livekit-api": publisher_session._LIVEKIT_API_VERSION,
        "livekit-protocol": publisher_session._LIVEKIT_PROTOCOL_VERSION,
    }


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("agents_version", "1.6.7", "livekit-agents==1.6.8"),
        ("rtc_version", "1.1.10", "livekit==1.1.14"),
        ("api_version", "1.1.1", "livekit-api==1.2.0"),
        ("protocol_version", "1.1.20", "livekit-protocol==1.1.21"),
        ("RoomIO", None, "reviewed surfaces"),
        ("AgentsConsole", None, "reviewed surfaces"),
    ],
)
def test_rejects_unreviewed_pair_and_missing_public_surfaces(
    monkeypatch, field, value, match
):
    sdk = _fake_sdk()
    setattr(sdk, field, value)
    monkeypatch.setattr(publisher_session, "_load_livekit_agents_sdk", lambda: sdk)
    loop = asyncio.new_event_loop()
    try:
        with pytest.raises(LiveKitAgentsDependencyError, match=match):
            TrustedLiveKitPublisherSession(
                room=_Room(),
                participant_identity="publisher-7",
                speech=_speech(),
                loop=loop,
            )
    finally:
        loop.close()


@pytest.mark.parametrize(
    ("enabled", "record"),
    [(True, False), (True, True), (False, True)],
)
def test_console_state_is_rejected_before_sdk_construction(
    monkeypatch, enabled, record
):
    sdk = _fake_sdk()
    sdk.captured.console.enabled = enabled
    sdk.captured.console.record = record
    monkeypatch.setattr(publisher_session, "_load_livekit_agents_sdk", lambda: sdk)
    loop = asyncio.new_event_loop()
    try:
        with pytest.raises(RuntimeError, match="AgentsConsole disabled"):
            TrustedLiveKitPublisherSession(
                room=_Room(),
                participant_identity="publisher-7",
                speech=_speech(),
                loop=loop,
            )
        assert sdk.captured.sessions == []
        assert sdk.captured.room_ios == []
        assert sdk.captured.forbidden == []
    finally:
        loop.close()


@pytest.mark.parametrize("failure_owner", ["session", "room"])
def test_constructor_partial_event_binding_rolls_back(monkeypatch, failure_owner):
    sdk, room = _fake_sdk(), _Room()
    if failure_owner == "session":
        sdk.captured.session_fail_on = "agent_state_changed"
    else:
        room.fail_on_event = "participant_disconnected"
    loop = asyncio.new_event_loop()
    try:
        with pytest.raises(RuntimeError, match="partial bind failed"):
            _wrapper(monkeypatch, sdk, room, loop)
        session = sdk.captured.sessions[0]
        assert all(not callbacks for callbacks in session.handlers.values())
        assert all(not callbacks for callbacks in room.handlers.values())
    finally:
        loop.close()


def test_start_requires_bound_committer_and_observed_connected_room(monkeypatch):
    async def scenario():
        sdk, room = _fake_sdk(), _Room()
        wrapper = _wrapper(monkeypatch, sdk, room, asyncio.get_running_loop())
        room_io = sdk.captured.room_ios[0]
        with pytest.raises(RuntimeError, match="committer"):
            await wrapper.start()
        assert room_io.start_calls == 0

        _bind_committer(wrapper)
        room.emit("connection_state_changed", _DISCONNECTED)
        with pytest.raises(RuntimeError, match="must be connected"):
            await wrapper.start()
        assert room_io.start_calls == 0

        room.emit("connection_state_changed", _CONNECTED)
        await wrapper.start()
        assert await wrapper.aclose()

    asyncio.run(scenario())


def test_start_rechecks_console_before_any_media_mutation(monkeypatch):
    async def scenario():
        sdk, room = _fake_sdk(), _Room()
        wrapper = _wrapper(monkeypatch, sdk, room, asyncio.get_running_loop())
        _bind_committer(wrapper)
        sdk.captured.console.enabled = True
        with pytest.raises(RuntimeError, match="AgentsConsole disabled"):
            await wrapper.start()
        assert sdk.captured.room_ios[0].start_calls == 0
        assert sdk.captured.sessions[0].start_calls == []
        sdk.captured.console.enabled = False
        assert await wrapper.aclose()

    asyncio.run(scenario())


def test_start_rejects_replaced_console_singleton_before_media_mutation(monkeypatch):
    async def scenario():
        sdk, room = _fake_sdk(), _Room()
        wrapper = _wrapper(monkeypatch, sdk, room, asyncio.get_running_loop())
        _bind_committer(wrapper)
        sdk.captured.console = SimpleNamespace(enabled=False, record=False)

        with pytest.raises(RuntimeError, match="console singleton changed"):
            await wrapper.start()

        assert sdk.captured.room_ios[0].start_calls == 0
        assert sdk.captured.sessions[0].start_calls == []
        assert await wrapper.aclose()

    asyncio.run(scenario())


def test_console_toggle_after_room_io_start_disposes_before_outward_close(monkeypatch):
    async def scenario():
        sdk, room = _fake_sdk(), _Room()
        wrapper = _wrapper(monkeypatch, sdk, room, asyncio.get_running_loop())
        session = sdk.captured.sessions[0]
        room_io = sdk.captured.room_ios[0]
        outward_proofs = []
        wrapper.on(
            "close",
            lambda _event: outward_proofs.append(
                (room_io.closed, session.input.audio, session.output.audio)
            ),
        )
        _bind_committer(wrapper)
        sdk.captured.console_toggle_after_room_start = True

        with pytest.raises(RuntimeError, match="AgentsConsole disabled"):
            await wrapper.start()

        assert session.start_calls == []
        assert session.say_calls == []
        assert session.shutdown_calls == []
        assert room_io.closed
        assert wrapper.closed
        assert outward_proofs == [(True, None, None)]

    asyncio.run(scenario())


def test_console_toggle_inside_sdk_start_fails_before_admission_and_disposes(
    monkeypatch,
):
    async def scenario():
        sdk, room = _fake_sdk(), _Room()
        wrapper = _wrapper(monkeypatch, sdk, room, asyncio.get_running_loop())
        session = sdk.captured.sessions[0]
        room_io = sdk.captured.room_ios[0]
        outward_proofs = []
        wrapper.on(
            "close",
            lambda _event: outward_proofs.append(
                (room_io.closed, session.input.audio, session.output.audio)
            ),
        )
        _bind_committer(wrapper)
        sdk.captured.console_toggle_during_session_start = True

        with pytest.raises(RuntimeError, match="AgentsConsole disabled"):
            await wrapper.start()

        assert session.start_calls == [{"record": False, "capture_run": False}]
        assert session.say_calls == []
        assert session.shutdown_calls == [False]
        assert room_io.closed
        assert wrapper.closed
        assert outward_proofs == [(True, None, None)]

    asyncio.run(scenario())


def test_sdk_start_output_takeover_fails_before_admission_and_disposes(monkeypatch):
    async def scenario():
        sdk, room = _fake_sdk(), _Room()
        wrapper = _wrapper(monkeypatch, sdk, room, asyncio.get_running_loop())
        session = sdk.captured.sessions[0]
        room_io = sdk.captured.room_ios[0]
        _bind_committer(wrapper)
        sdk.captured.replace_output_during_session_start = True

        with pytest.raises(RuntimeError, match="exact enabled media I/O"):
            await wrapper.start()

        assert session.say_calls == []
        assert session.shutdown_calls == [False]
        assert room_io.closed
        assert session.output.audio is None
        assert wrapper.closed

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure_mode",
    ["room_start", "session_start", "ready_error", "ready_timeout"],
)
def test_start_and_ready_failures_detach_and_dispose_manual_io(
    monkeypatch, failure_mode
):
    async def scenario():
        sdk, room = _fake_sdk(), _Room()
        wrapper = _wrapper(
            monkeypatch,
            sdk,
            room,
            asyncio.get_running_loop(),
            ready_timeout=0.01,
        )
        session = sdk.captured.sessions[0]
        room_io = sdk.captured.room_ios[0]
        _bind_committer(wrapper)
        if failure_mode == "room_start":
            room_io.start_error = RuntimeError("room start failed")
        elif failure_mode == "session_start":
            session.start_error = RuntimeError("session start failed")
        elif failure_mode == "ready_error":
            room_io.ready_error = RuntimeError("ready failed")
        else:
            room_io.auto_ready = False

        with pytest.raises((RuntimeError, TimeoutError)):
            await wrapper.start()
        assert room_io.closed
        assert room_io.close_calls == 1
        assert session.input.audio is None
        assert session.output.audio is None
        assert wrapper.closed
        if failure_mode == "room_start":
            assert session.shutdown_calls == []
        else:
            assert session.shutdown_calls == [False]

    asyncio.run(scenario())


def test_cancelled_partial_sdk_start_proves_only_manual_room_io_disposal(
    monkeypatch,
):
    async def scenario():
        sdk, room = _fake_sdk(), _Room()
        wrapper = _wrapper(monkeypatch, sdk, room, asyncio.get_running_loop())
        session = sdk.captured.sessions[0]
        room_io = sdk.captured.room_ios[0]
        session.start_gate = asyncio.Event()
        outward_proofs = []
        wrapper.on(
            "close",
            lambda event: outward_proofs.append(
                (room_io.closed, session.close_events, event.reason.value)
            ),
        )
        _bind_committer(wrapper)
        start_task = asyncio.create_task(wrapper.start())
        while not session.start_calls:
            await asyncio.sleep(0)
        assert session.partial_start_entered
        assert not session.started

        start_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await start_task

        assert session.say_calls == []
        assert session.shutdown_calls == [False]
        assert session.close_events == 0
        assert room_io.closed
        assert session.input.audio is None
        assert session.output.audio is None
        assert wrapper.closed
        assert outward_proofs == [(True, 0, "error")]

    asyncio.run(scenario())


def test_aclose_during_room_io_start_waits_for_start_to_settle(monkeypatch):
    async def scenario():
        sdk, room = _fake_sdk(), _Room()
        wrapper = _wrapper(monkeypatch, sdk, room, asyncio.get_running_loop())
        room_io = sdk.captured.room_ios[0]
        room_io.start_gate = asyncio.Event()
        _bind_committer(wrapper)
        start_task = asyncio.create_task(wrapper.start())
        await asyncio.sleep(0)
        assert room_io.start_calls == 1

        assert not await wrapper.aclose(timeout=0.01)
        assert room_io.close_calls == 0
        room_io.start_gate.set()
        with pytest.raises(RuntimeError, match="disconnected|closed"):
            await start_task
        assert await wrapper.aclose(timeout=0.2)
        assert room_io.close_calls == 1
        assert wrapper.closed

    asyncio.run(scenario())


def test_close_waits_for_manual_room_io_and_timeout_retry_keeps_ownership(
    monkeypatch,
):
    async def scenario():
        sdk, room = _fake_sdk(), _Room()
        wrapper = _wrapper(monkeypatch, sdk, room, asyncio.get_running_loop())
        session = sdk.captured.sessions[0]
        room_io = sdk.captured.room_ios[0]
        room_io.close_gate = asyncio.Event()
        outward = []
        wrapper.on("close", outward.append)
        _bind_committer(wrapper)
        await wrapper.start()

        assert not await wrapper.aclose(timeout=0.01)
        assert room_io.close_calls == 1
        assert not wrapper.closed
        assert outward == []
        with pytest.raises(RuntimeError, match="closing"):
            wrapper.on("close", lambda _event: None)
        with pytest.raises(RuntimeError, match="closing"):
            wrapper.bind_user_turn_committer(lambda _text, **_kwargs: True)
        with pytest.raises(RuntimeError, match="closed"):
            await wrapper.start()

        room_io.close_gate.set()
        assert await wrapper.aclose(timeout=0.2)
        assert wrapper.closed
        assert len(outward) == 1
        assert session.shutdown_calls == [True]

    asyncio.run(scenario())


def test_concurrent_aclose_callers_share_one_shutdown_and_finalizer(monkeypatch):
    async def scenario():
        sdk, room = _fake_sdk(), _Room()
        wrapper = _wrapper(monkeypatch, sdk, room, asyncio.get_running_loop())
        session = sdk.captured.sessions[0]
        room_io = sdk.captured.room_ios[0]
        room_io.close_gate = asyncio.Event()
        _bind_committer(wrapper)
        await wrapper.start()

        first = asyncio.create_task(wrapper.aclose(timeout=0.2))
        second = asyncio.create_task(wrapper.aclose(timeout=0.2))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert room_io.close_calls == 1
        assert session.shutdown_calls == [True]
        room_io.close_gate.set()
        assert await asyncio.gather(first, second) == [True, True]

    asyncio.run(scenario())


def test_failed_manual_io_finalizer_is_retryable_and_never_claims_closed(
    monkeypatch,
):
    async def scenario():
        sdk, room = _fake_sdk(), _Room()
        wrapper = _wrapper(monkeypatch, sdk, room, asyncio.get_running_loop())
        room_io = sdk.captured.room_ios[0]
        room_io.close_errors.append(RuntimeError("first close failed"))
        outward = []
        wrapper.on("close", outward.append)
        _bind_committer(wrapper)
        await wrapper.start()

        assert not await wrapper.aclose(timeout=0.1)
        assert not wrapper.closed
        assert outward == []
        assert room_io.close_calls == 1
        assert await wrapper.aclose(timeout=0.1)
        assert wrapper.closed
        assert room_io.close_calls == 2
        assert len(outward) == 1

    asyncio.run(scenario())


def test_close_before_start_mutation_is_the_only_immediate_terminal_path(
    monkeypatch,
):
    async def scenario():
        sdk, room = _fake_sdk(), _Room()
        wrapper = _wrapper(monkeypatch, sdk, room, asyncio.get_running_loop())
        outward = []
        wrapper.on("close", outward.append)
        _bind_committer(wrapper)
        assert await wrapper.aclose()
        assert wrapper.closed
        assert len(outward) == 1
        assert sdk.captured.room_ios[0].close_calls == 0
        assert sdk.captured.sessions[0].shutdown_calls == []

    asyncio.run(scenario())


def test_disconnect_is_terminal_and_disposes_manual_io_before_outward_close(
    monkeypatch,
):
    async def scenario():
        sdk, room = _fake_sdk(), _Room()
        wrapper = _wrapper(monkeypatch, sdk, room, asyncio.get_running_loop())
        session = sdk.captured.sessions[0]
        room_io = sdk.captured.room_ios[0]
        outward = []
        wrapper.on("close", outward.append)
        _bind_committer(wrapper)
        await wrapper.start()
        _, lease = wrapper.say_tracked(
            "connected",
            allow_interruptions=True,
            add_to_chat_ctx=False,
            on_done=lambda _handle: None,
        )

        room.emit("connection_state_changed", _DISCONNECTED)
        assert not wrapper.output_lease_valid(lease)
        assert await wrapper.aclose(timeout=0.2)
        assert session.shutdown_calls == [False]
        assert room_io.closed
        assert wrapper.closed
        assert len(outward) == 1
        room.emit("connection_state_changed", _CONNECTED)
        with pytest.raises(RuntimeError, match="closed"):
            await wrapper.start()

    asyncio.run(scenario())


def test_bound_participant_disconnect_is_terminal_for_every_reason(monkeypatch):
    async def scenario():
        sdk, room = _fake_sdk(), _Room()
        loop = asyncio.get_running_loop()
        wrapper = _wrapper(monkeypatch, sdk, room, loop)
        session = sdk.captured.sessions[0]
        room_io = sdk.captured.room_ios[0]
        effects = []

        # Before any media mutation, participant churn cannot dispose an
        # otherwise reusable wrapper.
        room.emit(
            "participant_disconnected",
            _participant(disconnect_reason="duplicate_identity"),
        )
        engine = LiveKitAgentsEngine(
            wrapper,
            participant_identity="publisher-7",
            loop=loop,
            clock=lambda: 1.0,
        )
        engine.start(
            EngineCallbacks(
                on_partial_result=lambda item: effects.append(("partial", item.text)),
                on_final_result=lambda item: effects.append(("final", item.text)),
            )
        )
        await wrapper.start()
        _, lease = wrapper.say_tracked(
            "connected",
            allow_interruptions=True,
            add_to_chat_ctx=False,
            on_done=lambda _handle: None,
        )

        room.emit(
            "participant_disconnected",
            _participant("somebody-else", disconnect_reason="duplicate_identity"),
        )
        assert wrapper.output_lease_valid(lease)

        room.emit(
            "participant_disconnected",
            _participant(disconnect_reason="duplicate_identity"),
        )
        assert not wrapper.output_lease_valid(lease)
        with pytest.raises(RuntimeError, match="unavailable"):
            wrapper.say_tracked(
                "must not reopen",
                allow_interruptions=True,
                add_to_chat_ctx=False,
                on_done=lambda _handle: None,
            )

        # A replacement connection using the same identity cannot restore the
        # route or release buffered input into the engine/control plane.
        room.emit(
            "track_subscribed",
            SimpleNamespace(sid="replacement-mic"),
            _publication("replacement-mic"),
            _participant(),
        )
        room.emit("connection_state_changed", _CONNECTED)
        await _emit_input_event_burst(sdk, "replacement")
        assert wrapper.active_input_track_id is None
        assert effects == []

        assert await wrapper.aclose(timeout=0.2)
        assert session.shutdown_calls == [False]
        assert room_io.closed
        assert wrapper.closed

    asyncio.run(scenario())


@pytest.mark.parametrize("mutation", ["toggle", "detach"])
def test_successful_output_mutations_invalidate_route_leases(monkeypatch, mutation):
    async def scenario():
        sdk, room = _fake_sdk(), _Room()
        wrapper = _wrapper(monkeypatch, sdk, room, asyncio.get_running_loop())
        _bind_committer(wrapper)
        await wrapper.start()
        _, lease = wrapper.say_tracked(
            "route-bound",
            allow_interruptions=True,
            add_to_chat_ctx=False,
            on_done=lambda _handle: None,
        )
        assert wrapper.output_lease_valid(lease)
        if mutation == "toggle":
            wrapper.set_audio_enabled(False)
            wrapper.set_audio_enabled(True)
        else:
            wrapper.detach_audio_output()
        assert not wrapper.output_lease_valid(lease)
        assert await wrapper.aclose()

    asyncio.run(scenario())


def test_tail_replacement_can_never_displace_exact_manual_room_io(monkeypatch):
    async def scenario():
        sdk, room = _fake_sdk(), _Room()
        wrapper = _wrapper(monkeypatch, sdk, room, asyncio.get_running_loop())
        session = sdk.captured.sessions[0]
        room_io = sdk.captured.room_ios[0]
        _bind_committer(wrapper)
        await wrapper.start()
        _, lease = wrapper.say_tracked(
            "route-bound",
            allow_interruptions=True,
            add_to_chat_ctx=False,
            on_done=lambda _handle: None,
        )

        with pytest.raises(RuntimeError, match="unsafe for exact manual RoomIO"):
            wrapper.replace_audio_tail(object())

        assert session.output.tail_replacements == []
        assert session.output.audio is room_io.audio_output
        assert not wrapper.output_lease_valid(lease)
        _, new_lease = wrapper.say_tracked(
            "still exact",
            allow_interruptions=True,
            add_to_chat_ctx=False,
            on_done=lambda _handle: None,
        )
        assert wrapper.output_lease_valid(new_lease)
        assert await wrapper.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("mutation", ["enable", "detach"])
def test_throwing_partially_mutating_output_operations_failure_close(
    monkeypatch, mutation
):
    async def scenario():
        sdk, room = _fake_sdk(), _Room()
        wrapper = _wrapper(monkeypatch, sdk, room, asyncio.get_running_loop())
        session = sdk.captured.sessions[0]
        _bind_committer(wrapper)
        await wrapper.start()
        _, lease = wrapper.say_tracked(
            "uncertain-route",
            allow_interruptions=True,
            add_to_chat_ctx=False,
            on_done=lambda _handle: None,
        )
        if mutation == "enable":
            session.output.enabled_partial = True
            session.output.enabled_error = RuntimeError("toggle changed then failed")
        else:
            session.output.audio_set_partial = True
            session.output.audio_set_error = RuntimeError("detach changed then failed")

        with pytest.raises(RuntimeError):
            if mutation == "enable":
                wrapper.set_audio_enabled(False)
            else:
                wrapper.detach_audio_output()
        assert not wrapper.output_lease_valid(lease)
        with pytest.raises(RuntimeError, match="unavailable"):
            wrapper.say_tracked(
                "must not overlap",
                allow_interruptions=True,
                add_to_chat_ctx=False,
                on_done=lambda _handle: None,
            )
        assert session.shutdown_calls == [False]
        assert await wrapper.aclose(timeout=0.2)

    asyncio.run(scenario())


def test_wrong_publisher_and_non_microphone_tracks_never_bind(monkeypatch):
    async def scenario():
        sdk, room = _fake_sdk(), _Room()
        wrapper = _wrapper(monkeypatch, sdk, room, asyncio.get_running_loop())
        _bind_committer(wrapper)
        await wrapper.start()
        _, lease = wrapper.say_tracked(
            "hello",
            allow_interruptions=True,
            add_to_chat_ctx=False,
            on_done=lambda _handle: None,
        )
        room.emit(
            "track_subscribed",
            SimpleNamespace(sid="wrong-track"),
            _publication("wrong-track"),
            _participant("somebody-else"),
        )
        room.emit(
            "track_subscribed",
            SimpleNamespace(sid="camera-track"),
            _publication("camera-track", _CAMERA),
            _participant(),
        )
        assert wrapper.active_input_track_id is None
        assert wrapper.output_lease_valid(lease)

        room.emit(
            "track_subscribed",
            SimpleNamespace(sid="mic-1"),
            _publication("mic-1"),
            _participant(),
        )
        assert wrapper.active_input_track_id == "mic-1"
        assert not wrapper.output_lease_valid(lease)
        assert await wrapper.aclose()

    asyncio.run(scenario())


def test_track_replacement_and_unpublish_invalidate_session_generation(
    monkeypatch,
):
    async def scenario():
        sdk, room = _fake_sdk(), _Room()
        wrapper = _wrapper(monkeypatch, sdk, room, asyncio.get_running_loop())
        _bind_committer(wrapper)
        await wrapper.start()
        room.emit(
            "track_subscribed",
            SimpleNamespace(sid="mic-1"),
            _publication("mic-1"),
            _participant(),
        )
        _, first = wrapper.say_tracked(
            "first",
            allow_interruptions=True,
            add_to_chat_ctx=False,
            on_done=lambda _handle: None,
        )
        room.emit(
            "track_subscribed",
            SimpleNamespace(sid="mic-2"),
            _publication("mic-2"),
            _participant(),
        )
        assert not wrapper.output_lease_valid(first)
        _, second = wrapper.say_tracked(
            "second",
            allow_interruptions=True,
            add_to_chat_ctx=False,
            on_done=lambda _handle: None,
        )
        room.emit(
            "track_unpublished",
            _publication("mic-2", _CAMERA),
            _participant(),
        )
        room.emit(
            "track_unpublished",
            _publication("different-mic"),
            _participant(),
        )
        assert wrapper.active_input_track_id == "mic-2"
        assert wrapper.output_lease_valid(second)
        room.emit("track_unpublished", _publication("mic-2"), _participant())
        assert wrapper.active_input_track_id is None
        assert not wrapper.output_lease_valid(second)
        assert await wrapper.aclose()

    asyncio.run(scenario())


def test_input_events_wait_for_successful_manual_room_io_admission(monkeypatch):
    async def scenario():
        sdk, room = _fake_sdk(), _Room()
        loop = asyncio.get_running_loop()
        wrapper = _wrapper(monkeypatch, sdk, room, loop)
        room_io = sdk.captured.room_ios[0]
        room_io.auto_ready = False
        effects = []
        engine = LiveKitAgentsEngine(
            wrapper,
            participant_identity="publisher-7",
            loop=loop,
            clock=lambda: 1.0,
        )
        engine.start(
            EngineCallbacks(
                on_partial_result=lambda item: effects.append(("partial", item.text)),
                on_final_result=lambda item: effects.append(("final", item.text)),
                on_barge_in_result=lambda _item: effects.append(("barge", None)),
                on_speech_start=lambda: effects.append(("state", "assistant-start")),
                on_speech_end=lambda: effects.append(("state", "assistant-end")),
            )
        )
        start_task = asyncio.create_task(wrapper.start())
        while room_io.wait_calls == 0:
            await asyncio.sleep(0)

        await _emit_input_event_burst(sdk, "pre-ready")
        assert effects == []

        room_io.ready_event.set()
        await start_task
        engine.speak_tracked(
            TrackedSpeech("ready", "owned output"),
            on_terminal=lambda _receipt: None,
        )
        await _emit_input_event_burst(sdk, "post-ready")
        assert effects == [
            ("state", "assistant-start"),
            ("partial", "post-ready partial"),
            ("barge", None),
            ("final", "post-ready final"),
        ]
        assert await wrapper.aclose()

    asyncio.run(scenario())


def test_input_events_never_escape_when_manual_room_io_readiness_fails(monkeypatch):
    async def scenario():
        sdk, room = _fake_sdk(), _Room()
        loop = asyncio.get_running_loop()
        wrapper = _wrapper(monkeypatch, sdk, room, loop)
        room_io = sdk.captured.room_ios[0]
        room_io.auto_ready = False
        effects = []
        engine = LiveKitAgentsEngine(
            wrapper,
            participant_identity="publisher-7",
            loop=loop,
            clock=lambda: 1.0,
        )
        engine.start(
            EngineCallbacks(
                on_partial_result=lambda item: effects.append(("partial", item.text)),
                on_final_result=lambda item: effects.append(("final", item.text)),
                on_barge_in_result=lambda _item: effects.append(("barge", None)),
                on_speech_start=lambda: effects.append(("state", "assistant-start")),
                on_speech_end=lambda: effects.append(("state", "assistant-end")),
            )
        )
        start_task = asyncio.create_task(wrapper.start())
        while room_io.wait_calls == 0:
            await asyncio.sleep(0)

        await _emit_input_event_burst(sdk, "rejected")
        assert effects == []
        room_io.ready_error = RuntimeError("publication failed")
        room_io.ready_event.set()
        with pytest.raises(RuntimeError, match="publication failed"):
            await start_task

        assert wrapper.closed
        assert room_io.closed
        assert effects == []
        await _emit_input_event_burst(sdk, "after-failure")
        assert effects == []

    asyncio.run(scenario())


def test_completed_message_id_reaches_generation_bound_engine_once(monkeypatch):
    async def scenario():
        sdk, room = _fake_sdk(), _Room()
        loop = asyncio.get_running_loop()
        wrapper = _wrapper(monkeypatch, sdk, room, loop)
        finals = []
        engine = LiveKitAgentsEngine(
            wrapper,
            participant_identity="publisher-7",
            loop=loop,
            clock=lambda: 1.0,
        )
        engine.start(EngineCallbacks(on_final_result=finals.append))
        await wrapper.start()
        agent = sdk.captured.sessions[0].started_agent
        message = SimpleNamespace(id="message-17", text_content="whole turn")
        await agent.on_user_turn_completed(None, message)
        await agent.on_user_turn_completed(None, message)
        await agent.on_user_turn_completed(
            None,
            SimpleNamespace(id="", text_content="must not invent an ID"),
        )
        assert [(item.text, item.revision) for item in finals] == [("whole turn", 0)]
        engine.stop()
        assert await wrapper.aclose()

    asyncio.run(scenario())


def test_callback_is_owned_before_sdk_admission_and_deferred_until_return(
    monkeypatch,
):
    async def scenario():
        sdk, room = _fake_sdk(), _Room()
        wrapper = _wrapper(monkeypatch, sdk, room, asyncio.get_running_loop())
        session = sdk.captured.sessions[0]
        session.immediate_done = True
        _bind_committer(wrapper)
        await wrapper.start()
        terminals = []
        handle, lease = wrapper.say_tracked(
            "instant",
            allow_interruptions=True,
            add_to_chat_ctx=False,
            on_done=terminals.append,
        )
        assert terminals == []
        assert wrapper.output_lease_valid(lease)
        await asyncio.sleep(0)
        assert terminals == [handle]
        assert handle.exception() is None
        assert await wrapper.aclose()

    asyncio.run(scenario())


def test_sdk_playout_error_produces_failed_receipt_without_safe_text(
    monkeypatch,
):
    async def scenario():
        sdk, room = _fake_sdk(), _Room()
        loop = asyncio.get_running_loop()
        wrapper = _wrapper(monkeypatch, sdk, room, loop)
        session = sdk.captured.sessions[0]
        receipts = []
        engine = LiveKitAgentsEngine(
            wrapper,
            participant_identity="publisher-7",
            loop=loop,
        )
        engine.start(EngineCallbacks())
        await wrapper.start()
        engine.speak_tracked(
            TrackedSpeech("failed", "must not be remembered"),
            on_terminal=receipts.append,
        )
        session.handles[0].finish(error=RuntimeError("playout failed"))
        await asyncio.sleep(0)
        assert [(item.outcome, item.safe_text_prefix) for item in receipts] == [
            (PlaybackOutcome.FAILED, "")
        ]
        engine.stop()
        assert await wrapper.aclose()

    asyncio.run(scenario())


def test_interrupt_and_sdk_double_clear_notify_terminal_once(monkeypatch):
    async def scenario():
        sdk, room = _fake_sdk(), _Room()
        wrapper = _wrapper(monkeypatch, sdk, room, asyncio.get_running_loop())
        session = sdk.captured.sessions[0]
        _bind_committer(wrapper)
        await wrapper.start()
        sink = session.output.audio
        terminals = []
        handle, _lease = wrapper.say_tracked(
            "interrupt me",
            allow_interruptions=True,
            add_to_chat_ctx=False,
            on_done=terminals.append,
        )
        handle.interrupt(force=True)
        assert session.handles[0].interrupt_calls == [True]
        assert sink.clear_calls == 1
        sink.clear_buffer()
        session.handles[0].finish(interrupted=True)
        session.handles[0].notify_done_again()
        await asyncio.sleep(0)
        assert sink.clear_calls == 2
        assert terminals == [handle]
        assert handle.interrupted
        assert await wrapper.aclose()

    asyncio.run(scenario())


def test_failed_queue_clear_fences_engine_until_manual_io_close_proof(
    monkeypatch,
):
    async def scenario():
        sdk, room = _fake_sdk(), _Room()
        loop = asyncio.get_running_loop()
        wrapper = _wrapper(monkeypatch, sdk, room, loop)
        session = sdk.captured.sessions[0]
        session.auto_close = False
        receipts = []
        engine = LiveKitAgentsEngine(
            wrapper,
            participant_identity="publisher-7",
            loop=loop,
        )
        engine.start(EngineCallbacks())
        await wrapper.start()
        sink = session.output.audio
        engine.speak_tracked(
            TrackedSpeech("active", "still queued"),
            on_terminal=receipts.append,
        )
        session.emit("agent_state_changed", SimpleNamespace(new_state="speaking"))
        sink.clear_error = RuntimeError("clear failed")

        engine.stop_speaking()
        assert receipts == []
        assert session.shutdown_calls == [False]
        assert sink.clear_calls == 2
        rejected = []
        engine.speak_tracked(
            TrackedSpeech("new", "must not overlap"),
            on_terminal=rejected.append,
        )
        assert rejected[0].outcome is PlaybackOutcome.DROPPED

        session.emit_close(reason="error")
        assert await wrapper.aclose(timeout=0.2)
        await asyncio.sleep(0)
        assert [(item.outcome, item.safe_text_prefix) for item in receipts] == [
            (PlaybackOutcome.INTERRUPTED, "")
        ]

    asyncio.run(scenario())


def test_engine_restart_rejects_wrapper_closed_while_handlers_were_unbound(
    monkeypatch,
):
    async def scenario():
        sdk, room = _fake_sdk(), _Room()
        loop = asyncio.get_running_loop()
        wrapper = _wrapper(monkeypatch, sdk, room, loop)
        engine = LiveKitAgentsEngine(
            wrapper,
            participant_identity="publisher-7",
            loop=loop,
        )
        engine.start(EngineCallbacks())
        await wrapper.start()
        engine.stop()
        assert await wrapper.aclose()
        with pytest.raises(RuntimeError, match="session is closed"):
            engine.start(EngineCallbacks())

    asyncio.run(scenario())


def test_rejects_non_object_speech_components():
    with pytest.raises(TypeError, match="explicit audited local"):
        AuditedLocalSpeechComponents(
            stt="hosted-model",
            vad=object(),
            tts=object(),
            audit_id="bad",
        )
