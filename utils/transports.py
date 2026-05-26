"""
Transport abstraction inspired by Wyoming-style local services and
LiveKit-style realtime sessions.

This module intentionally keeps transport adapters lightweight so the core
assistant can run with no external infra. In production, these classes can be
replaced with concrete protocol backends.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
import queue
import threading
import time

from utils import tts_debug


class TransportMode(str, Enum):
    LOCAL_LAN = "local_lan"
    WEBRTC = "webrtc"
    HYBRID = "hybrid"


@dataclass
class SessionEnvelope:
    session_id: str
    event_type: str
    payload: Dict[str, Any]
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp <= 0:
            self.timestamp = time.time()


class BaseTransport:
    """Queue-backed mock transport interface."""

    def __init__(self, name: str):
        self.name = name
        self._started = False
        self._inbound: queue.Queue[SessionEnvelope] = queue.Queue()
        self._outbound: queue.Queue[SessionEnvelope] = queue.Queue()

    def start(self):
        self._started = True

    def stop(self):
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    def push_inbound(self, envelope: SessionEnvelope):
        if self._started:
            self._inbound.put(envelope)

    def pop_inbound(self, timeout: float = 0.0) -> Optional[SessionEnvelope]:
        if not self._started:
            return None
        try:
            return self._inbound.get(timeout=timeout)
        except queue.Empty:
            return None

    def send(self, envelope: SessionEnvelope):
        if self._started:
            self._outbound.put(envelope)

    def pop_outbound(self, timeout: float = 0.0) -> Optional[SessionEnvelope]:
        if not self._started:
            return None
        try:
            return self._outbound.get(timeout=timeout)
        except queue.Empty:
            return None


class LocalLanTransport(BaseTransport):
    def __init__(self):
        super().__init__("local_lan")


class WebRtcTransport(BaseTransport):
    def __init__(self):
        super().__init__("webrtc")
        self._data_sink = None

    def attach_data_sink(self, sink) -> None:
        """Set a ``callable(dict)`` that publishes envelopes to a live LiveKit
        data channel. When unset (default/tests) this stays a plain queue."""
        self._data_sink = sink

    def send(self, envelope: "SessionEnvelope"):
        super().send(envelope)  # keep queue semantics (used by tests / pollers)
        sink = self._data_sink
        if sink is not None and self._started:
            try:
                sink(
                    {
                        "session_id": envelope.session_id,
                        "event_type": envelope.event_type,
                        "payload": envelope.payload,
                        "timestamp": envelope.timestamp,
                    }
                )
            except Exception:
                pass


class SessionMux:
    """Routes envelopes between enabled transports and core runtime."""

    def __init__(self, mode: TransportMode = TransportMode.LOCAL_LAN):
        self.mode = mode
        self.local = LocalLanTransport()
        self.webrtc = WebRtcTransport()
        self._lock = threading.Lock()

    def start(self):
        with self._lock:
            if self.mode in (TransportMode.LOCAL_LAN, TransportMode.HYBRID):
                self.local.start()
            if self.mode in (TransportMode.WEBRTC, TransportMode.HYBRID):
                self.webrtc.start()

    def stop(self):
        with self._lock:
            self.local.stop()
            self.webrtc.stop()

    def active_transports(self) -> List[str]:
        active: List[str] = []
        if self.local.started:
            active.append("local_lan")
        if self.webrtc.started:
            active.append("webrtc")
        return active

    def broadcast(self, envelope: SessionEnvelope):
        if envelope.event_type in (
            "assistant_sentence",
            "user_transcript",
            "speech_detected",
        ):
            tts_debug.log_tts(
                "transport_broadcast",
                mode=self.mode.value,
                event_type=envelope.event_type,
                local_playback_skipped=False,
                note="local_lan queues events only; speakers still play locally",
            )
        if self.local.started:
            self.local.send(envelope)
        if self.webrtc.started:
            self.webrtc.send(envelope)
