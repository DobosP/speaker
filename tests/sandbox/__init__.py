"""Device-behavior simulation sandbox.

Unlike the static, synchronous tests in ``test_core_runtime.py`` (which prove
*logic*), this sandbox models *timing*: incremental STT partials, endpoint
delay, LLM time-to-first-token and per-token streaming (by model weight), and
TTS playback duration. It drives the real threaded brain so it exercises the
middle-layer decisions (stop / continue / speak) under realistic concurrency.

Swap the simulated engine/LLM for the real ``SherpaOnnxEngine`` / ``OllamaLLM``
to run the same scenarios on-device.
"""

from .profiles import DESKTOP_HIGH, DESKTOP_MID, PHONE_LOW, DeviceProfile
from .scenario import Sandbox
from .sim_engine import SimulatedEngine
from .sim_llm import SimulatedLLM

__all__ = [
    "DeviceProfile",
    "PHONE_LOW",
    "DESKTOP_MID",
    "DESKTOP_HIGH",
    "SimulatedEngine",
    "SimulatedLLM",
    "Sandbox",
]
