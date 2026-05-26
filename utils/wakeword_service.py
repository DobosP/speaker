"""
Wakeword detection service boundary with local and process modes.

This provides a Wyoming-style separation where wakeword detection can run in
its own process while AudioRecorder consumes only detection events.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from utils.voice_gate import OpenWakeWordGate


@dataclass
class WakewordEvent:
    detected: bool
    score: float
    label: Optional[str]
    timestamp: float


class BaseWakewordService:
    mode: str = "base"

    @property
    def available(self) -> bool:
        return False

    @property
    def labels(self) -> list[str]:
        return []

    @property
    def last_score(self) -> float:
        return 0.0

    def start(self):
        return None

    def stop(self):
        return None

    def submit_audio(self, audio_chunk: np.ndarray, sample_rate: int):
        return None

    def poll_event(self) -> Optional[WakewordEvent]:
        return None


class LocalWakewordService(BaseWakewordService):
    mode = "local"

    def __init__(self, gate: OpenWakeWordGate):
        self._gate = gate
        self._events: queue.Queue = queue.Queue(maxsize=8)

    @property
    def available(self) -> bool:
        return self._gate.available

    @property
    def labels(self) -> list[str]:
        return list(getattr(self._gate, "available_labels", []) or [])

    @property
    def last_score(self) -> float:
        return float(getattr(self._gate, "last_score", 0.0))

    def submit_audio(self, audio_chunk: np.ndarray, sample_rate: int):
        if not self.available:
            return
        try:
            detected = self._gate.detect(audio_chunk, sample_rate)
        except Exception:
            detected = False
        if detected:
            evt = WakewordEvent(
                detected=True,
                score=self.last_score,
                label=getattr(self._gate, "wakeword", None),
                timestamp=time.time(),
            )
            try:
                self._events.put_nowait(evt)
            except queue.Full:
                pass

    def poll_event(self) -> Optional[WakewordEvent]:
        try:
            return self._events.get_nowait()
        except queue.Empty:
            return None


def _worker_loop(
    in_q: mp.Queue,
    out_q: mp.Queue,
    stop_q: mp.Queue,
    wakeword: Optional[str],
    threshold: float,
    model_path: Optional[str],
):
    gate = OpenWakeWordGate(
        wakeword=wakeword,
        threshold=threshold,
        model_path=model_path,
    )
    # Send runtime capabilities to parent.
    out_q.put(
        {
            "type": "meta",
            "available": gate.available,
            "labels": getattr(gate, "available_labels", []),
        }
    )
    while True:
        try:
            stop_q.get_nowait()
            break
        except Exception:
            pass
        try:
            payload = in_q.get(timeout=0.1)
        except Exception:
            continue
        if payload is None:
            break
        audio = payload.get("audio")
        sample_rate = payload.get("sample_rate", 16000)
        if audio is None:
            continue
        detected = False
        try:
            detected = gate.detect(audio, sample_rate)
        except Exception:
            detected = False
        if detected:
            out_q.put(
                {
                    "type": "event",
                    "detected": True,
                    "score": float(getattr(gate, "last_score", 0.0)),
                    "label": wakeword,
                    "timestamp": time.time(),
                }
            )


class ProcessWakewordService(BaseWakewordService):
    mode = "process"

    def __init__(
        self,
        wakeword: Optional[str],
        threshold: float,
        model_path: Optional[str],
    ):
        self._wakeword = wakeword
        self._threshold = threshold
        self._model_path = model_path
        self._in_q: mp.Queue = mp.Queue(maxsize=32)
        self._out_q: mp.Queue = mp.Queue(maxsize=32)
        self._stop_q: mp.Queue = mp.Queue(maxsize=1)
        self._process: Optional[mp.Process] = None
        self._available = False
        self._labels: list[str] = []
        self._last_score = 0.0

    @property
    def available(self) -> bool:
        return self._available

    @property
    def labels(self) -> list[str]:
        return self._labels

    @property
    def last_score(self) -> float:
        return self._last_score

    def start(self):
        if self._process and self._process.is_alive():
            return
        self._process = mp.Process(
            target=_worker_loop,
            args=(
                self._in_q,
                self._out_q,
                self._stop_q,
                self._wakeword,
                self._threshold,
                self._model_path,
            ),
            daemon=True,
        )
        self._process.start()
        # Wait briefly for metadata.
        deadline = time.time() + 2.0
        while time.time() < deadline:
            try:
                message = self._out_q.get(timeout=0.1)
            except Exception:
                continue
            if message.get("type") == "meta":
                self._available = bool(message.get("available", False))
                self._labels = list(message.get("labels", []) or [])
                return

    def stop(self):
        try:
            self._stop_q.put_nowait(True)
        except Exception:
            pass
        try:
            self._in_q.put_nowait(None)
        except Exception:
            pass
        if self._process and self._process.is_alive():
            self._process.join(timeout=1.0)

    def submit_audio(self, audio_chunk: np.ndarray, sample_rate: int):
        if not self._process or not self._process.is_alive():
            return
        try:
            self._in_q.put_nowait(
                {
                    "audio": audio_chunk.flatten().astype(np.float32),
                    "sample_rate": int(sample_rate),
                }
            )
        except Exception:
            pass

    def poll_event(self) -> Optional[WakewordEvent]:
        try:
            message = self._out_q.get_nowait()
        except Exception:
            return None
        if message.get("type") != "event":
            return None
        self._last_score = float(message.get("score", 0.0))
        return WakewordEvent(
            detected=bool(message.get("detected", False)),
            score=self._last_score,
            label=message.get("label"),
            timestamp=float(message.get("timestamp", time.time())),
        )


def build_wakeword_service(
    mode: str,
    wakeword: Optional[str],
    threshold: float,
    model_path: Optional[str],
    detector_override=None,
) -> BaseWakewordService:
    """Create wakeword service with graceful fallback to local mode."""
    if detector_override is not None:
        return LocalWakewordService(detector_override)
    normalized = (mode or "local").strip().lower()
    if normalized == "process":
        service = ProcessWakewordService(
            wakeword=wakeword,
            threshold=threshold,
            model_path=model_path,
        )
        service.start()
        if service.available:
            return service
    gate = OpenWakeWordGate(
        wakeword=wakeword,
        threshold=threshold,
        model_path=model_path,
    )
    return LocalWakewordService(gate)
