"""Canonical over-the-air autotest acoustic rig -- the "real conversation" setup.

OWNER DECISION (2026-06-19), LOCKED -- do not silently change:
to simulate a real conversation between the owner and the assistant,

    * the ASSISTANT (LLM / TTS)        -> the BARE LAPTOP SPEAKER @ 50%   (the device)
    * the USER's injected voice        -> the JBL Flip 5         @ 20%   (the person)
    * capture                          -> the laptop mic
    * AEC reference delay              -> 40 ms (laptop-speaker->mic; the assistant
                                          is NOT on Bluetooth in this setup)

Why this split: the agent speaks from the laptop (where the mic is, the real
open-speaker / self-interrupt condition), and the user's voice arrives from a
separate speaker (the person across the table). The JBL is physically loud, so it
sits at 20%; the laptop assistant at 50%.

The built-in mic's analog ADC gain (`amixer -c 1 Capture`) defaults to +30 dB and
PipeWire RE-APPLIES it on every source suspend/resume, which CLIPS the capture at
the converter before any digital volume can help. So the rig also holds the ADC
down with a background gain-pinner (:func:`gain_pinner`) for the duration of a run.

CLI:  .venv/bin/python -m tools.autotest.ota_setup          # apply + print
The barge-in stress harness applies this automatically (see barge_stress.py).
"""
from __future__ import annotations

import contextlib
import re
import subprocess
import threading
import time
from typing import Optional

# --- the owner's rig (machine-specific device IDs; this laptop) ------------- #
LAPTOP_SPEAKER = "alsa_output.pci-0000_00_1f.3.analog-stereo"
JBL = "bluez_output.D8_37_3B_19_CF_03.1"
LAPTOP_MIC = "alsa_input.pci-0000_00_1f.3.analog-stereo"
JBL_MAC = "D8:37:3B:19:CF:03"

# --- the locked levels ------------------------------------------------------ #
ASSISTANT_SINK = LAPTOP_SPEAKER     # the agent / LLM output (default sink)
ASSISTANT_VOLUME_PCT = 75
USER_INJECT_SINK = JBL              # the injected user voice (--inject-sink)
USER_VOLUME_PCT = 20
CAPTURE_SOURCE = LAPTOP_MIC
MIC_SOURCE_VOLUME_PCT = 100         # PipeWire digital source volume (post-ADC)
MIC_ADC_CARD = 1                    # PCH card holding the Capture/Boost controls
MIC_ADC_CAPTURE_PCT = 52           # held by the gain-pinner (defeats the +30 dB reset)
AEC_REF_DELAY_MS = 40               # laptop-speaker -> mic (assistant is NOT on BT here)


def _pactl(*args) -> None:
    subprocess.run(["pactl", *args], capture_output=True)


def _set_adc(pct: int) -> None:
    subprocess.run(["amixer", "-c", str(MIC_ADC_CARD), "sset", "Capture", f"{pct}%"],
                   capture_output=True)
    subprocess.run(["amixer", "-c", str(MIC_ADC_CARD), "sset", "Internal Mic Boost", "0"],
                   capture_output=True)


def _amixer_raw(control: str) -> Optional[str]:
    """Read an amixer control's raw (settable) first-channel value, e.g. '33' from
    'Front Left: Capture 33 [52%] [7.50dB] [on]' -- used to SAVE the pre-run level
    so :func:`gain_pinner` can restore it exactly (raw avoids %-rounding drift)."""
    out = subprocess.run(["amixer", "-c", str(MIC_ADC_CARD), "sget", control],
                         capture_output=True, text=True).stdout
    m = re.search(r":\s*(?:Capture\s+)?(\d+)\s*\[", out)
    return m.group(1) if m else None


def _source_pct() -> Optional[str]:
    """The PipeWire source volume as an integer percent string, or None."""
    out = subprocess.run(["pactl", "get-source-volume", CAPTURE_SOURCE],
                         capture_output=True, text=True).stdout
    m = re.search(r"/\s*(\d+)%", out)
    return m.group(1) if m else None


def ensure_jbl_connected(timeout_s: float = 8.0) -> bool:
    """Make sure the JBL sink is present (reconnect if it idled out)."""
    def present() -> bool:
        out = subprocess.run(["pactl", "list", "short", "sinks"],
                             capture_output=True, text=True).stdout
        return "bluez" in out
    if present():
        return True
    subprocess.run(["bluetoothctl", "connect", JBL_MAC], capture_output=True, text=True)
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if present():
            return True
        time.sleep(0.5)
    return present()


def apply() -> dict:
    """Apply the locked rig: assistant->laptop@50%, user->JBL@20%, mic, ADC down.
    Returns a dict describing what was set (for logging)."""
    ensure_jbl_connected()
    _pactl("set-default-sink", ASSISTANT_SINK)
    _pactl("set-sink-volume", ASSISTANT_SINK, f"{ASSISTANT_VOLUME_PCT}%")
    _pactl("set-sink-volume", JBL, f"{USER_VOLUME_PCT}%")
    _pactl("set-default-source", CAPTURE_SOURCE)
    _pactl("set-source-volume", CAPTURE_SOURCE, f"{MIC_SOURCE_VOLUME_PCT}%")
    _set_adc(MIC_ADC_CAPTURE_PCT)
    return {
        "assistant_sink": ASSISTANT_SINK, "assistant_volume_pct": ASSISTANT_VOLUME_PCT,
        "user_inject_sink": USER_INJECT_SINK, "user_volume_pct": USER_VOLUME_PCT,
        "capture_source": CAPTURE_SOURCE, "mic_adc_capture_pct": MIC_ADC_CAPTURE_PCT,
        "aec_ref_delay_ms": AEC_REF_DELAY_MS,
    }


@contextlib.contextmanager
def gain_pinner(period_s: float = 0.5):
    """Hold the mic ADC gain down AND the source volume up for the duration of a
    run (PipeWire resets the ADC to +30 dB *and* restores a saved source volume on
    every suspend/resume, so without pinning the source a run can capture far too
    quiet -- the failure observed 2026-06-21 where a stale 13% source survived a
    barge_stress run and crippled the next live session).

    Crucially, RESTORE the pre-run ADC + source on exit so a run is non-destructive
    -- it must not leave the mic dialled to the rig's levels for the next app."""
    orig_cap = _amixer_raw("Capture")
    orig_boost = _amixer_raw("Internal Mic Boost")
    orig_src = _source_pct()
    stop = threading.Event()

    def _loop() -> None:
        while not stop.is_set():
            _set_adc(MIC_ADC_CAPTURE_PCT)
            _pactl("set-source-volume", CAPTURE_SOURCE, f"{MIC_SOURCE_VOLUME_PCT}%")
            stop.wait(period_s)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join(timeout=2.0)
        # best-effort restore of the pre-run capture levels
        if orig_cap is not None:
            subprocess.run(["amixer", "-c", str(MIC_ADC_CARD), "sset", "Capture", orig_cap],
                           capture_output=True)
        if orig_boost is not None:
            subprocess.run(["amixer", "-c", str(MIC_ADC_CARD), "sset",
                            "Internal Mic Boost", orig_boost], capture_output=True)
        if orig_src is not None:
            _pactl("set-source-volume", CAPTURE_SOURCE, f"{orig_src}%")


def main() -> int:
    info = apply()
    print("applied the LOCKED 'real conversation' OTA rig:")
    for k, v in info.items():
        print(f"  {k:22s} = {v}")
    print("  (mic ADC is held by the gain-pinner only DURING a run; barge_stress does this)")
    print(f"  reminder: set sherpa.aec_ref_delay_ms = {AEC_REF_DELAY_MS} in config.local.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
