"""Typed deployment plan for every public voice-session entry path.

The control plane is deliberately absent from this module.  A session plan may
choose where audio is captured and transported, but it can never select an LLM,
register tools, or grant action authority.  Those remain owned by
``VoiceRuntime`` and ``always_on_agent`` regardless of the media adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import ipaddress
import re
from threading import Event, Lock, get_ident
from typing import Mapping
from urllib.parse import urlsplit


class VoiceSessionError(ValueError):
    """A requested media topology is invalid or not explicitly authorized."""


class SessionMode(str, Enum):
    """User-facing session choices exposed by the single ``python -m core`` CLI."""

    CONSOLE = "console"
    LOCAL = "local"
    TRUSTED_LAN = "trusted-lan"
    REPLAY = "replay"


class SessionTopology(str, Enum):
    """Physical audio boundary selected by a session mode."""

    DEVICE_LOCAL = "device_local"
    TRUSTED_LAN = "trusted_lan"


class AudioEgress(str, Enum):
    """Maximum audio boundary granted by machine-local setup."""

    DEVICE_ONLY = "device_only"
    TRUSTED_LAN = "trusted_lan"


_ENGINE_BY_MODE = {
    SessionMode.CONSOLE: "console",
    SessionMode.LOCAL: "sherpa",
    # This name is intentionally not understood by ``core.app._build_engine``.
    # The public topology stays fail-closed until ADR-0096's concrete,
    # publisher-bound LiveKit Agents wrapper exists.
    SessionMode.TRUSTED_LAN: "livekit_agents",
    SessionMode.REPLAY: "replay",
}
_MODE_BY_LEGACY_ENGINE = {
    "console": SessionMode.CONSOLE,
    "sherpa": SessionMode.LOCAL,
    "livekit": SessionMode.TRUSTED_LAN,
    "replay": SessionMode.REPLAY,
}
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


def engine_for_session(value: str | SessionMode | None) -> str:
    """Resolve a CLI session name without consulting machine-local policy."""

    try:
        mode = SessionMode(value or SessionMode.CONSOLE)
    except ValueError as exc:
        choices = ", ".join(item.value for item in SessionMode)
        raise VoiceSessionError(f"session must be one of: {choices}") from exc
    return _ENGINE_BY_MODE[mode]


def _safe_identifier(name: str, value: object, default: str) -> str:
    result = str(value or default).strip()
    if not _SAFE_ID.fullmatch(result):
        raise VoiceSessionError(
            f"{name} must be 1-128 letters, digits, '_' or '-', starting "
            "with a letter or digit"
        )
    return result


def _audio_egress(config: Mapping[str, object]) -> AudioEgress:
    block = config.get("voice_session", {})
    if block is None:
        block = {}
    if not isinstance(block, Mapping):
        raise VoiceSessionError("voice_session configuration must be an object")
    value = str(block.get("audio_egress", AudioEgress.DEVICE_ONLY.value) or "")
    try:
        return AudioEgress(value)
    except ValueError as exc:
        choices = ", ".join(item.value for item in AudioEgress)
        raise VoiceSessionError(f"voice_session.audio_egress must be one of: {choices}") from exc


@dataclass(frozen=True)
class VoiceSessionPlan:
    """Validated media/session selection consumed by the runtime assembler.

    ``publisher_identity`` scopes future LiveKit input subscription.  It is a
    transport binding only and must never be interpreted as speaker identity or
    owner/action authority.
    """

    mode: SessionMode
    engine: str
    audio_egress: AudioEgress
    room: str
    agent_identity: str
    publisher_identity: str

    @property
    def topology(self) -> SessionTopology:
        if self.mode is SessionMode.TRUSTED_LAN:
            return SessionTopology.TRUSTED_LAN
        return SessionTopology.DEVICE_LOCAL

    @property
    def sends_audio_over_network(self) -> bool:
        return self.topology is SessionTopology.TRUSTED_LAN

    @property
    def is_selectable(self) -> bool:
        """Whether the chosen adapter has a production composition today."""

        return self.engine != "livekit_agents"

    def require_selectable(self) -> None:
        """Refuse a declared topology whose trusted wrapper has not landed."""

        if not self.is_selectable:
            raise VoiceSessionError(
                "trusted-LAN audio is authorized but its publisher-bound "
                "LiveKit Agents wrapper is not installed yet; keep using a "
                "device-local session until that adapter passes live A/B"
            )

    @classmethod
    def resolve(
        cls,
        *,
        config: Mapping[str, object],
        requested_session: str | None = None,
        legacy_engine: str | None = None,
        room: str | None = None,
        agent_identity: str | None = None,
        publisher_identity: str | None = None,
    ) -> "VoiceSessionPlan":
        if requested_session is not None and legacy_engine is not None:
            raise VoiceSessionError("choose --session or the legacy --engine option, not both")

        if requested_session is not None:
            try:
                mode = SessionMode(requested_session)
            except ValueError as exc:
                choices = ", ".join(item.value for item in SessionMode)
                raise VoiceSessionError(f"session must be one of: {choices}") from exc
        else:
            engine = legacy_engine or "console"
            try:
                mode = _MODE_BY_LEGACY_ENGINE[engine]
            except KeyError as exc:
                choices = ", ".join(sorted(_MODE_BY_LEGACY_ENGINE))
                raise VoiceSessionError(f"legacy engine must be one of: {choices}") from exc

        audio_egress = _audio_egress(config)
        if mode is SessionMode.TRUSTED_LAN and audio_egress is not AudioEgress.TRUSTED_LAN:
            raise VoiceSessionError(
                "trusted-LAN audio is not authorized on this device; run "
                "`python -m tools.setup_assistant --allow-trusted-lan-audio` "
                "after reviewing the privacy boundary"
            )

        # Unused remote configuration must not make a device-local assistant
        # unavailable. It becomes startup-critical only when raw audio is
        # actually allowed onto the trusted LAN.
        remote: Mapping[str, object] = {}
        if mode is SessionMode.TRUSTED_LAN:
            configured_remote = config.get("remote", {})
            if configured_remote is None:
                configured_remote = {}
            if not isinstance(configured_remote, Mapping):
                raise VoiceSessionError("remote configuration must be an object")
            remote = configured_remote

        return cls(
            mode=mode,
            engine=(legacy_engine if legacy_engine is not None else _ENGINE_BY_MODE[mode]),
            audio_egress=audio_egress,
            room=_safe_identifier("room", room or remote.get("room"), "assistant"),
            agent_identity=_safe_identifier(
                "agent identity",
                agent_identity or remote.get("agent_identity"),
                "assistant-agent",
            ),
            publisher_identity=_safe_identifier(
                "publisher identity",
                publisher_identity or remote.get("publisher_identity"),
                "user",
            ),
        )


class VoiceSession:
    """Own exactly one already-assembled runtime for one media session.

    This object deliberately receives the runtime instead of constructing it:
    tools, models, authority, and ``AgentSupervisor`` remain assembled once by
    ``core.app.build_runtime``. Its narrow job is to make teardown ownership
    explicit while the legacy engine is split into replaceable media stages.
    """

    def __init__(self, plan: VoiceSessionPlan, runtime: object) -> None:
        if runtime is None or not callable(getattr(runtime, "stop", None)):
            raise TypeError("voice session runtime must provide stop()")
        self.plan = plan
        self.runtime = runtime
        self._stop_lock = Lock()
        self._stop_started = False
        self._stop_owner: int | None = None
        self._stop_done = Event()
        self._stop_error: BaseException | None = None

    @property
    def stopped(self) -> bool:
        return self._stop_done.is_set()

    def stop(self) -> None:
        """Release the owned runtime at most once, including failure paths."""

        caller = get_ident()
        with self._stop_lock:
            if not self._stop_started:
                self._stop_started = True
                self._stop_owner = caller
                owns_stop = True
            elif self._stop_owner == caller and not self._stop_done.is_set():
                # Avoid deadlock if runtime.stop() re-enters its session owner.
                return
            else:
                owns_stop = False

        if owns_stop:
            try:
                self.runtime.stop()
            except BaseException as exc:
                with self._stop_lock:
                    self._stop_error = exc
                raise
            finally:
                with self._stop_lock:
                    self._stop_owner = None
                self._stop_done.set()
            return

        # A second lifecycle caller observes completed teardown (or the same
        # failure) instead of returning while devices are still being released.
        self._stop_done.wait()
        with self._stop_lock:
            error = self._stop_error
        if error is not None:
            raise error

    close = stop

    def __enter__(self) -> "VoiceSession":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.stop()


def validate_livekit_url(url: object) -> str:
    """Validate a self-hosted trusted-LAN LiveKit URL without DNS or network I/O.

    Plain ``ws://`` is accepted only for the loopback development server.  LAN
    traffic must use TLS, and public/cloud hosts are rejected: granting trusted
    LAN audio must not silently become a general raw-audio egress permission.
    """

    value = str(url or "").strip()
    if not value:
        raise VoiceSessionError("trusted-LAN mode requires LIVEKIT_URL")
    parsed = urlsplit(value)
    if parsed.scheme not in {"ws", "wss"} or not parsed.hostname:
        raise VoiceSessionError("LIVEKIT_URL must be a ws:// or wss:// URL")
    if parsed.username is not None or parsed.password is not None:
        raise VoiceSessionError("LIVEKIT_URL must not contain credentials")
    host = parsed.hostname.rstrip(".").lower()

    local_host = host == "localhost"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None:
        local_host = bool(address.is_loopback)
        trusted_host = bool(address.is_private or address.is_loopback or address.is_link_local)
    else:
        trusted_host = bool(
            local_host
            or host.endswith(".local")
            or host.endswith(".home.arpa")
        )

    if not trusted_host:
        raise VoiceSessionError(
            "LIVEKIT_URL must resolve by configuration to a loopback or trusted-LAN host"
        )
    if parsed.scheme == "ws" and not local_host:
        raise VoiceSessionError("non-loopback trusted-LAN audio requires wss:// encryption")
    return value
