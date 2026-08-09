"""Exact installed LiveKit 1.6.8 headless lifecycle contract.

This backend test uses the real published SDK and its in-memory RTC audio
objects.  The room boundary is a deterministic fake: no network, audio device,
model, credential, worker, recorder, or hosted inference path is available.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from importlib import metadata as importlib_metadata
import os
from typing import Any, Callable

import pytest


_EXPECTED_DISTRIBUTIONS = {
    "livekit": "1.1.14",
    "livekit-agents": "1.6.8",
    "livekit-api": "1.2.0",
    "livekit-protocol": "1.1.21",
}


def _require_exact_installed_contract() -> None:
    if os.environ.get("SPEAKER_LIVEKIT_INSTALLED_CONTRACT") != "1":
        pytest.skip(
            "requires SPEAKER_LIVEKIT_INSTALLED_CONTRACT=1 in a private runtime",
            allow_module_level=True,
        )
    observed: dict[str, str] = {}
    try:
        for distribution in _EXPECTED_DISTRIBUTIONS:
            observed[distribution] = importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError as exc:
        raise AssertionError(
            "active installed-package gate is missing the exact LiveKit closure"
        ) from exc
    assert observed == _EXPECTED_DISTRIBUTIONS


_require_exact_installed_contract()

from livekit import rtc  # noqa: E402
from livekit.agents import stt, tts, vad  # noqa: E402
from livekit.agents.types import (  # noqa: E402
    APIConnectOptions,
    DEFAULT_API_CONNECT_OPTIONS,
)

from core.engines.livekit_agents_session import (  # noqa: E402
    AuditedLocalSpeechComponents,
    TrustedLiveKitPublisherSession,
)


pytestmark = pytest.mark.backend


class _SilentRecognizeStream(stt.RecognizeStream):
    async def _run(self) -> None:
        async for _item in self._input_ch:
            pass


class _SilentSTT(stt.STT):
    def __init__(self) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
            )
        )

    async def _recognize_impl(
        self,
        _buffer: Any,
        *,
        language: Any,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        raise AssertionError("headless contract must not invoke offline STT")

    def stream(
        self,
        *,
        language: Any = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.RecognizeStream:
        del language
        return _SilentRecognizeStream(stt=self, conn_options=conn_options)


class _SilentVADStream(vad.VADStream):
    async def _main_task(self) -> None:
        async for _item in self._input_ch:
            pass


class _SilentVAD(vad.VAD):
    def __init__(self) -> None:
        super().__init__(capabilities=vad.VADCapabilities(update_interval=0.1))

    def stream(self) -> vad.VADStream:
        return _SilentVADStream(self)


class _OneFrameChunkedStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id="headless-request",
            sample_rate=24_000,
            num_channels=1,
            mime_type="audio/pcm",
            frame_size_ms=20,
        )
        output_emitter.push(b"\0" * 960)


class _OneFrameTTS(tts.TTS):
    def __init__(self) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=24_000,
            num_channels=1,
        )

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return _OneFrameChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
        )


class _FakePublication:
    async def wait_for_subscription(self) -> None:
        return None


class _FakeLocalParticipant:
    def __init__(self) -> None:
        self.identity = "headless-agent"
        self.published: list[tuple[object, object]] = []
        self.attribute_updates: list[dict[str, str]] = []

    async def publish_track(self, track: object, options: object) -> object:
        self.published.append((track, options))
        return _FakePublication()

    async def set_attributes(self, attributes: dict[str, str]) -> None:
        self.attribute_updates.append(dict(attributes))


class _FakeRemoteParticipant:
    identity = "headless-publisher"
    kind = rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD
    attributes: dict[str, str] = {}
    track_publications: dict[str, object] = {}


class _FakeConnectedRoom:
    def __init__(self) -> None:
        self.name = "headless-room"
        self.connection_state = rtc.ConnectionState.CONN_CONNECTED
        self.local_participant = _FakeLocalParticipant()
        participant = _FakeRemoteParticipant()
        self.remote_participants = {participant.identity: participant}
        self._handlers: dict[str, list[Callable[..., None]]] = defaultdict(list)

    def on(self, event: str, callback: Callable[..., None]) -> Callable[..., None]:
        self._handlers[event].append(callback)
        return callback

    def off(self, event: str, callback: Callable[..., None]) -> None:
        try:
            self._handlers[event].remove(callback)
        except ValueError:
            pass

    def isconnected(self) -> bool:
        return True

    async def connect(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("headless contract must not connect to a room")

    def register_text_stream_handler(self, *_args: object) -> None:
        raise AssertionError("headless contract must not register text transport")

    def unregister_text_stream_handler(self, *_args: object) -> None:
        raise AssertionError("headless contract must not register text transport")


def test_exact_installed_manual_roomio_start_and_close_are_headless() -> None:
    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        room = _FakeConnectedRoom()
        wrapper = TrustedLiveKitPublisherSession(
            room=room,
            participant_identity="headless-publisher",
            speech=AuditedLocalSpeechComponents(
                stt=_SilentSTT(),
                vad=_SilentVAD(),
                tts=_OneFrameTTS(),
                audit_id="installed-livekit-headless-v1",
            ),
            loop=loop,
            ready_timeout=2.0,
            close_timeout=2.0,
        )
        closed = False
        try:
            wrapper.bind_user_turn_committer(lambda *_args, **_kwargs: True)

            await asyncio.wait_for(wrapper.start(), timeout=3.0)

            assert wrapper.policy.is_safe
            assert wrapper._session.options.aec_warmup_duration is None
            assert wrapper._session._room_io is None
            assert wrapper._session._recorder_io is None
            assert wrapper._session._session_host is None
            assert wrapper._session.input.audio is wrapper._room_io.audio_input
            assert wrapper._session.output.audio is wrapper._room_io.audio_output
            assert (
                wrapper._room_io.linked_participant
                is room.remote_participants["headless-publisher"]
            )
            assert len(room.local_participant.published) == 1
            _track, publish_options = room.local_participant.published[0]
            assert publish_options.source == rtc.TrackSource.SOURCE_MICROPHONE

            done = asyncio.Event()
            callbacks: list[object] = []

            def on_done(handle: object) -> None:
                callbacks.append(handle)
                done.set()

            handle, lease = wrapper.say_tracked(
                "headless lifecycle check",
                allow_interruptions=True,
                add_to_chat_ctx=False,
                on_done=on_done,
            )
            await asyncio.wait_for(done.wait(), timeout=3.0)
            assert callbacks == [handle]
            assert handle.exception() is None
            assert handle.interrupted is False
            assert wrapper.output_lease_valid(lease) is True

            closed = await wrapper.aclose(drain=False, timeout=3.0)
        finally:
            if not closed:
                await wrapper.aclose(drain=False, timeout=3.0)

        assert closed is True
        assert wrapper.closed is True
        assert wrapper._room_io_resources_closed is True
        assert wrapper._room_io_closed is True
        assert wrapper._session.input.audio is None
        assert wrapper._session.output.audio is None
        assert wrapper._room_io.audio_input is not None
        assert wrapper._room_io.audio_output is not None
        assert all(not callbacks for callbacks in room._handlers.values())

    asyncio.run(scenario())


def test_exact_installed_token_ttl_and_in_process_health_are_headless(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx
    import jwt

    from remote.token_server import create_access_token, create_app

    monkeypatch.setenv("LIVEKIT_API_KEY", "headless-test-key")
    monkeypatch.setenv(
        "LIVEKIT_API_SECRET",
        "headless-test-secret-with-at-least-thirty-two-bytes",
    )
    token = create_access_token(
        "headless-publisher",
        "headless-room",
        ttl_seconds=3_600,
    )
    claims = jwt.decode(
        token,
        options={"verify_signature": False, "verify_aud": False},
    )
    assert claims["exp"] - claims["nbf"] == 3_600
    assert claims["sub"] == "headless-publisher"
    assert claims["video"]["room"] == "headless-room"
    assert claims["video"]["roomJoin"] is True
    assert claims["video"]["canPublish"] is True
    assert claims["video"]["canSubscribe"] is True
    assert claims["video"]["canPublishData"] is True

    monkeypatch.delenv("SPEAKER_REMOTE_ALLOW_NOAUTH", raising=False)
    monkeypatch.delenv("SPEAKER_REMOTE_TOKEN", raising=False)

    async def get_health() -> httpx.Response:
        transport = httpx.ASGITransport(app=create_app({}))
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://headless.invalid",
        ) as client:
            return await client.get("/healthz")

    response = asyncio.run(get_health())
    assert response.status_code == 200
    assert response.json() == {"ok": True}
