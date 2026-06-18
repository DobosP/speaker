"""Acoustic-path backends for the voice tier -- how the assistant's TTS reaches
the engine's mic, and where injected 'user' utterances go.

* :class:`CableAcoustics` -- a single PipeWire null sink (digital loopback,
  ~tens-of-ms delay). Silent, fast, parallel-safe. Default.
* :class:`DelayAcoustics` -- two null sinks bridged by a ``module-loopback``
  with ``latency_msec`` ~= the configured acoustic delay (260 ms), so the AEC
  reference aligns the way it does on a real open speaker. Engine output -> far
  sink; far.monitor --(delayed)--> mic sink; engine capture <- mic.monitor;
  user clips are injected into the mic sink (no extra delay, i.e. near-field).
  Silent. Fixes the cable mode's delay-mismatch confound without real hardware.
* :class:`SpeakerAcoustics` -- no virtual devices: the engine runs on the real
  default speaker + mic, and user clips play out the real speaker. TRUE
  over-the-air (real ~260 ms acoustic delay, real speaker/room coloring) -- the
  genuine open-speaker condition. Makes audible sound and records the real mic.

Each is a context manager that loads/unloads its PipeWire modules and exposes:
  ``needs_routing``   -- whether engine streams must be moved (False for speaker)
  ``inject_target``   -- pactl sink to paplay user clips into (None = default)
  ``route(pid)``      -- move the engine's streams onto this rig
  ``capture_ready(pid)`` -- engine capture is attached to the right source
  ``uses_real_device``-- engine should run on default devices (speaker) vs the
                         ``pipewire`` ALSA bridge (cable/delay)
"""
from __future__ import annotations

import contextlib
import subprocess
from typing import Iterator, Optional

from . import audio


class CableAcoustics:
    """Two null sinks: the engine PLAYS to a dead ``play`` sink (discarded) and
    CAPTURES a separate ``cap`` sink where clips are injected. The assistant's
    TTS therefore never reaches the mic -> NO echo -> clean, reproducible STT +
    round-trip (digital injection = a perfect near-field user). It does NOT test
    self-interrupt or barge-in -- both need the echo/talk-over relationship; use
    ``delay`` (silent) or ``speaker`` (real over-the-air)."""

    needs_routing = True
    uses_real_device = False
    has_echo = False     # playback -> dead sink, so STT only (no self-interrupt/barge)
    inject_gain = 100        # digital full-level injection; no echo to out-shout

    def __init__(self, prefix: str = "cc_autotest"):
        self._play = f"{prefix}_play"
        self._cap = f"{prefix}_cap"
        self._mods: list[str] = []

    @property
    def inject_target(self) -> Optional[str]:
        return self._cap

    @property
    def capture_source(self) -> str:
        return f"{self._cap}.monitor"

    @contextlib.contextmanager
    def session(self) -> Iterator["CableAcoustics"]:
        def load(name: str) -> None:
            mid = subprocess.run(
                ["pactl", "load-module", "module-null-sink", f"sink_name={name}",
                 f"sink_properties=device.description={name}"],
                capture_output=True, text=True, check=True).stdout.strip()
            if not mid.isdigit():
                raise RuntimeError(f"load-module failed: {name} -> {mid!r}")
            self._mods.append(mid)
            subprocess.run(["pactl", "set-sink-volume", name, "100%"], capture_output=True)

        try:
            load(self._play)
            load(self._cap)
            yield self
        finally:
            for mid in reversed(self._mods):
                subprocess.run(["pactl", "unload-module", mid], capture_output=True)
            self._mods.clear()

    def route(self, pid: int) -> None:
        audio.route_streams(pid, self._play, f"{self._cap}.monitor")

    def capture_ready(self, pid: int) -> bool:
        return audio.capture_on(pid, f"{self._cap}.monitor")


class DelayAcoustics:
    needs_routing = True
    uses_real_device = False
    has_echo = True
    inject_gain = 300        # the loopback rig is lossy -> out-shout the echo

    def __init__(self, latency_ms: int = 260, prefix: str = "cc_autotest"):
        self.latency_ms = latency_ms
        self._far = f"{prefix}_far"
        self._mic = f"{prefix}_mic"
        self._mods: list[str] = []

    @property
    def inject_target(self) -> Optional[str]:
        return self._mic        # near-field: into the mic sink, no extra delay

    @property
    def capture_source(self) -> str:
        return f"{self._mic}.monitor"

    @contextlib.contextmanager
    def session(self) -> Iterator["DelayAcoustics"]:
        def load(args: list[str]) -> str:
            mid = subprocess.run(["pactl", "load-module", *args],
                                 capture_output=True, text=True, check=True).stdout.strip()
            if not mid.isdigit():
                raise RuntimeError(f"load-module failed: {args} -> {mid!r}")
            self._mods.append(mid)
            return mid

        try:
            load(["module-null-sink", f"sink_name={self._far}",
                  f"sink_properties=device.description={self._far}"])
            load(["module-null-sink", f"sink_name={self._mic}",
                  f"sink_properties=device.description={self._mic}"])
            subprocess.run(["pactl", "set-sink-volume", self._far, "100%"], capture_output=True)
            subprocess.run(["pactl", "set-sink-volume", self._mic, "100%"], capture_output=True)
            # the air gap: far.monitor --(delayed)--> mic, pinned so the engine's
            # own moves don't drag the bridge endpoints around.
            load(["module-loopback", f"source={self._far}.monitor", f"sink={self._mic}",
                  f"latency_msec={self.latency_ms}", "source_dont_move=true",
                  "sink_dont_move=true"])
            yield self
        finally:
            for mid in reversed(self._mods):
                subprocess.run(["pactl", "unload-module", mid], capture_output=True)
            self._mods.clear()

    def route(self, pid: int) -> None:
        audio.route_streams(pid, self._far, f"{self._mic}.monitor")

    def capture_ready(self, pid: int) -> bool:
        return audio.capture_on(pid, f"{self._mic}.monitor")


class SpeakerAcoustics:
    """Real over-the-air: engine on the default speaker+mic; clips play out the
    speaker. No virtual devices, no stream moving -- just real sound."""

    needs_routing = False
    uses_real_device = True
    has_echo = True
    inject_gain = 170             # far-field over the real speaker -> boost
    inject_target = None          # default sink == the real speaker
    capture_source = ""

    @contextlib.contextmanager
    def session(self) -> Iterator["SpeakerAcoustics"]:
        yield self

    def route(self, pid: int) -> None:  # nothing to move
        pass

    def capture_ready(self, pid: int) -> bool:
        return True                # the engine already opened the real mic


def make_acoustics(mode: str, *, latency_ms: int = 260):
    if mode == "cable":
        return CableAcoustics()
    if mode == "delay":
        return DelayAcoustics(latency_ms=latency_ms)
    if mode == "speaker":
        return SpeakerAcoustics()
    raise ValueError(f"unknown acoustics mode: {mode!r}")
