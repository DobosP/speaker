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
import os
import re
import subprocess
import tempfile
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


def _source_index(name: str) -> Optional[str]:
    """Numeric index of a source by name (e.g. ``cc_sink.monitor``)."""
    short = subprocess.run(
        ["pactl", "list", "short", "sources"], capture_output=True, text=True
    ).stdout
    for line in short.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1] == name:
            return parts[0]
    return None


def route_streams(pid: int, play_sink: str, capture_source: str) -> tuple[int, int]:
    """Move the engine's bridge playback streams onto ``play_sink`` and its
    capture streams onto ``capture_source``. Idempotent -- safe to call from a
    poll loop. ``pid`` is accepted for API symmetry but bridge streams expose no
    PID, so matching is by node name. Returns ``(playback_moved, capture_moved)``."""
    moved_play = moved_cap = 0
    for idx, _tgt, ident in _streams("sink-inputs"):
        if _is_engine_stream(ident, _PLAY_NODE):
            r = subprocess.run(
                ["pactl", "move-sink-input", idx, play_sink], capture_output=True
            )
            moved_play += int(r.returncode == 0)
    for idx, _tgt, ident in _streams("source-outputs"):
        if _is_engine_stream(ident, _CAP_NODE):
            r = subprocess.run(
                ["pactl", "move-source-output", idx, capture_source], capture_output=True
            )
            moved_cap += int(r.returncode == 0)
    return moved_play, moved_cap


def capture_on(pid: int, source_name: str) -> bool:
    """True once the engine's capture stream is pulling from ``source_name``."""
    want = _source_index(source_name)
    if want is None:
        return False
    for _idx, tgt, ident in _streams("source-outputs"):
        if _is_engine_stream(ident, _CAP_NODE) and tgt == want:
            return True
    return False


def _with_lead_in(wav_path: str, lead_in_ms: int) -> str:
    """Write a temp copy of ``wav_path`` with ``lead_in_ms`` of leading silence,
    returning its path. The caller removes it.

    Two failure modes this fixes: (1) a Bluetooth sink resuming from idle drops
    the first audio of a freshly-opened stream while the A2DP link spins up, and
    (2) the engine's VAD needs a clean onset or it truncates the first word(s).
    Silence padding in the SAME stream covers both -- the link is live and the
    VAD has settled by the time real speech starts."""
    with wave.open(wav_path, "rb") as w:
        nch, sw, sr = w.getnchannels(), w.getsampwidth(), w.getframerate()
        frames = w.readframes(w.getnframes())
    pad = b"\x00" * (int(sr * lead_in_ms / 1000) * nch * sw)
    fd, out = tempfile.mkstemp(prefix="cc_inject_", suffix=".wav")
    os.close(fd)
    with wave.open(out, "wb") as w:
        w.setnchannels(nch)
        w.setsampwidth(sw)
        w.setframerate(sr)
        w.writeframes(pad + frames)
    return out


def inject(
    target_sink: Optional[str], wav_path: str, *, volume_pct: int = 100,
    lead_in_ms: int = 0,
) -> None:
    """Play ``wav_path`` into ``target_sink`` (a 'user' utterance). ``None`` ->
    the system default sink (real over-the-air through the speaker).

    ``volume_pct`` boosts playback (100 = unity, 65536 PA units): a far-field
    injection (real speaker, or the lossy delay rig) must out-shout the
    assistant's echo to clear the engine's echo-floor gate, the way a near-field
    user naturally does.

    ``lead_in_ms`` prepends that much silence to the played clip (real over-the-
    air, esp. Bluetooth) so the sink is live + the engine's VAD has settled
    before the first word -- otherwise the clip's opening gets truncated."""
    play_path = wav_path
    tmp: Optional[str] = None
    if lead_in_ms > 0:
        tmp = _with_lead_in(wav_path, lead_in_ms)
        play_path = tmp
    try:
        cmd = ["paplay"]
        if target_sink:
            cmd.append(f"--device={target_sink}")
        if volume_pct != 100:
            cmd.append(f"--volume={int(65536 * volume_pct / 100)}")
        cmd.append(play_path)
        subprocess.run(cmd, capture_output=True)
    finally:
        if tmp:
            with contextlib.suppress(OSError):
                os.remove(tmp)


# --------------------------------------------------------------------------- #
# TTS synthesis of the injected utterances (same model the engine speaks with)
# --------------------------------------------------------------------------- #
def synth_to_wav(text: str, out_path: str, *, sherpa_cfg: dict, speed: float = 1.0) -> float:
    """Render ``text`` to a 16-bit mono WAV with the engine's configured voice.

    Returns the clip duration in seconds. ``sherpa_cfg`` is the merged
    ``config["sherpa"]`` block. The TTS is built through the SAME
    :func:`core.engines._sherpa_models.build_tts` the runtime uses, so it tracks
    whichever family is configured -- Piper/VITS or Kokoro (ADR-0010) -- rather
    than assuming VITS. (Wiring a Kokoro model into the VITS slot aborts natively:
    "Not a model using characters as modeling unit. Please provide --vits-lexicon".)"""
    from core.engines._sherpa_models import build_tts
    from core.engines.sherpa import SherpaConfig

    tts = build_tts(SherpaConfig.from_dict(sherpa_cfg))
    if tts is None:
        raise RuntimeError(
            "autotest clip synth: TTS failed to build from the sherpa config -- "
            "see the speaker.sherpa warning above (e.g. a Kokoro package with "
            "missing model/voices/tokens files). Fix the tts_* paths in "
            "config.local.json, or clear tts_voices to use the Piper/VITS voice."
        )
    audio = tts.generate(text, sid=0, speed=speed)
    samples = np.asarray(audio.samples, dtype=np.float32).reshape(-1)
    sr = int(audio.sample_rate)
    # peak-normalize to a loud, consistent level so an injected "user" clip sits
    # clearly above the engine's learned echo/quiet floor (a near-field user is
    # louder than the assistant's open-speaker echo; a quiet clip gets dropped as
    # "echo/ambient, not speech").
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 1e-4:
        samples = samples * (0.95 / peak)
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
