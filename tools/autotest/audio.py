"""PipeWire/PulseAudio plumbing for the autonomous ``voice`` tier.

Everything here is reversible and scoped: we create a throwaway ``module-null-
sink`` (a virtual audio cable), route **only the runtime's own** playback +
capture streams onto it via ``pactl move-sink-input`` / ``move-source-output``
(matched by the runtime's PID), and unload the module on exit. The system
default sink/source is never changed, so other apps and the user's audio are
untouched.

No third-party deps -- shells out to ``pactl`` / ``paplay`` (PipeWire's pulse
compat layer, confirmed present on the dev box). TTS synthesis reuses the same
sherpa-onnx VITS model the engine speaks with, so the injected "user" voice is
real speech the recognizer can transcribe.
"""
from __future__ import annotations

import contextlib
import re
import subprocess
import time
import wave
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# null sink (the virtual cable)
# --------------------------------------------------------------------------- #
@dataclass
class NullSink:
    """A loaded ``module-null-sink``: play into ``name``, capture ``monitor``."""

    name: str
    monitor: str
    module_id: str


@contextlib.contextmanager
def null_sink(name: str = "cc_autotest_sink") -> Iterator[NullSink]:
    """Load a null sink, yield it, unload on exit (even on error)."""
    mid = subprocess.run(
        [
            "pactl", "load-module", "module-null-sink",
            f"sink_name={name}",
            f"sink_properties=device.description={name}",
        ],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if not mid.isdigit():
        raise RuntimeError(f"could not load null sink: {mid!r}")
    # Unity volume so the loopback level matches what the engine actually emits
    # (PipeWire otherwise applies a per-stream gain that hot-clips the monitor).
    subprocess.run(["pactl", "set-sink-volume", name, "100%"], capture_output=True)
    try:
        yield NullSink(name=name, monitor=f"{name}.monitor", module_id=mid)
    finally:
        subprocess.run(["pactl", "unload-module", mid], capture_output=True)


# --------------------------------------------------------------------------- #
# stream routing (move only the runtime's own streams onto the cable)
# --------------------------------------------------------------------------- #
# The sherpa engine reaches PipeWire through PortAudio's ALSA backend, so its
# streams arrive as ALSA-PipeWire *bridge* nodes -- which (unlike libpulse
# clients) carry NO ``application.process.id``. We therefore identify them by
# their bridge node names (``alsa_capture.*`` / ``alsa_playback.*``, tagged
# ``PipeWire ALSA [python...]``). Safe for an autonomous run: the runtime
# subprocess is the only thing driving the ALSA-PipeWire bridge during a test
# (synthesis writes WAVs; ``paplay`` is a native pulse client, not an ALSA one).
_PLAY_NODE = "alsa_playback"
_CAP_NODE = "alsa_capture"


def _streams(kind: str) -> list[tuple[str, str, str]]:
    """Return ``[(index, target_node_index, node_name)]`` for ``sink-inputs``
    (playback) or ``source-outputs`` (capture)."""
    txt = subprocess.run(["pactl", "list", kind], capture_output=True, text=True).stdout
    out: list[tuple[str, str, str]] = []
    for block in txt.split("\n\n"):
        m = re.search(r"#(\d+)", block)
        if not m:
            continue
        idx = m.group(1)
        tgt = re.search(r"(?:Sink|Source):\s*(\d+)", block)
        node = re.search(r'node\.name\s*=\s*"([^"]+)"', block)
        app = re.search(r'application\.name\s*=\s*"([^"]+)"', block)
        ident = (node.group(1) if node else "") + "|" + (app.group(1) if app else "")
        out.append((idx, tgt.group(1) if tgt else "", ident))
    return out


def _is_engine_stream(ident: str, node_prefix: str) -> bool:
    return node_prefix in ident or ("ALSA" in ident and "python" in ident.lower())


def route_process_streams(pid: int, sink: NullSink) -> tuple[int, int]:
    """Move the engine's bridge playback/capture streams onto ``sink`` /
    ``sink.monitor``. Idempotent -- safe to call repeatedly from a poll loop.
    ``pid`` is accepted for API symmetry but bridge streams expose no PID, so
    matching is by node name. Returns ``(playback_moved, capture_moved)``."""
    moved_play = moved_cap = 0
    for idx, _tgt, ident in _streams("sink-inputs"):
        if _is_engine_stream(ident, _PLAY_NODE):
            r = subprocess.run(
                ["pactl", "move-sink-input", idx, sink.name], capture_output=True
            )
            moved_play += int(r.returncode == 0)
    for idx, _tgt, ident in _streams("source-outputs"):
        if _is_engine_stream(ident, _CAP_NODE):
            r = subprocess.run(
                ["pactl", "move-source-output", idx, sink.monitor], capture_output=True
            )
            moved_cap += int(r.returncode == 0)
    return moved_play, moved_cap


def capture_is_routed(pid: int, sink: NullSink) -> bool:
    """True once the engine's capture stream is pulling from ``sink.monitor``."""
    want = None
    # resolve the monitor source's numeric index once
    short = subprocess.run(
        ["pactl", "list", "short", "sources"], capture_output=True, text=True
    ).stdout
    for line in short.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1] == sink.monitor:
            want = parts[0]
            break
    if want is None:
        return False
    for _idx, tgt, ident in _streams("source-outputs"):
        if _is_engine_stream(ident, _CAP_NODE) and tgt == want:
            return True
    return False


def inject(sink: NullSink, wav_path: str) -> None:
    """Play ``wav_path`` into the cable (a synthesized 'user' utterance)."""
    subprocess.run(["paplay", f"--device={sink.name}", wav_path], capture_output=True)


# --------------------------------------------------------------------------- #
# TTS synthesis of the injected utterances (same model the engine speaks with)
# --------------------------------------------------------------------------- #
def synth_to_wav(text: str, out_path: str, *, sherpa_cfg: dict, speed: float = 1.0) -> float:
    """Render ``text`` to a 16-bit mono WAV with the engine's VITS voice.

    Returns the clip duration in seconds. ``sherpa_cfg`` is the merged
    ``config["sherpa"]`` block (for the tts model paths)."""
    import sherpa_onnx

    cfg = sherpa_onnx.OfflineTtsConfig()
    cfg.model.vits.model = sherpa_cfg["tts_model"]
    cfg.model.vits.tokens = sherpa_cfg["tts_tokens"]
    if sherpa_cfg.get("tts_data_dir"):
        cfg.model.vits.data_dir = sherpa_cfg["tts_data_dir"]
    cfg.model.num_threads = 2
    tts = sherpa_onnx.OfflineTts(cfg)
    audio = tts.generate(text, sid=0, speed=speed)
    samples = np.asarray(audio.samples, dtype=np.float32).reshape(-1)
    sr = int(audio.sample_rate)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    with wave.open(out_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return len(samples) / sr if sr else 0.0


# --------------------------------------------------------------------------- #
# wav analysis (verify a capture actually carried audio)
# --------------------------------------------------------------------------- #
def wav_rms(path: str) -> tuple[float, float, int]:
    """Return ``(rms, peak, n_samples)`` for a WAV file (0,0,0 if unreadable)."""
    try:
        with wave.open(path, "rb") as w:
            n = w.getnframes()
            sw = w.getsampwidth()
            raw = w.readframes(n)
        dt = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
        a = np.frombuffer(raw, dtype=dt).astype(np.float32)
        if a.size:
            a /= float(np.iinfo(dt).max)
        rms = float(np.sqrt(np.mean(a ** 2))) if a.size else 0.0
        peak = float(np.max(np.abs(a))) if a.size else 0.0
        return rms, peak, a.size
    except Exception:
        return 0.0, 0.0, 0
