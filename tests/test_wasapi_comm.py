"""Device-free tests for the native WASAPI communications capture (ADR-0082).

Pure-logic layer only: WAVEFORMATEX parsing, downmix, effects verdict, the
sample ring's blocking/overflow/error contract, and the enroll route-probe
seam. The COM layer is validated on real Windows hardware (the probe run is
part of the ADR-0082 evidence); nothing here touches a device.
"""
from __future__ import annotations

import struct
import threading
import time

import numpy as np
import pytest

from core.engines._wasapi_comm import (
    GUID_EFFECT_AEC,
    GUID_EFFECT_NS,
    MixFormat,
    SampleRing,
    downmix_to_mono_float32,
    effects_verdict,
    parse_waveformatex,
)
from core.enroll import EnrollmentCaptureError, verify_required_os_echo_route


# --- WAVEFORMATEX parsing ----------------------------------------------------


def _wfx_plain(tag, channels, rate, bits):
    align = channels * bits // 8
    return struct.pack("<HHIIHH", tag, channels, rate, rate * align, align, bits)


def _wfx_extensible(channels, rate, bits, subformat_guid: str):
    base = _wfx_plain(0xFFFE, channels, rate, bits)
    d = subformat_guid.replace("-", "")
    sub = (
        struct.pack("<IHH", int(d[0:8], 16), int(d[8:12], 16), int(d[12:16], 16))
        + bytes.fromhex(d[16:20])
        + bytes.fromhex(d[20:32])
    )
    # cbSize(22) + Samples + dwChannelMask + SubFormat
    return base + struct.pack("<HHI", 22, bits, 0x3) + sub


def test_parse_extensible_float32():
    raw = _wfx_extensible(2, 48000, 32, "00000003-0000-0010-8000-00aa00389b71")
    fmt = parse_waveformatex(raw)
    assert fmt == MixFormat(48000, 2, 32, True)


def test_parse_plain_pcm16():
    fmt = parse_waveformatex(_wfx_plain(0x0001, 1, 16000, 16))
    assert fmt == MixFormat(16000, 1, 16, False)


def test_parse_rejects_unknown_subformat_and_short_blobs():
    with pytest.raises(ValueError):
        parse_waveformatex(
            _wfx_extensible(2, 48000, 32, "12345678-0000-0010-8000-00aa00389b71")
        )
    with pytest.raises(ValueError):
        parse_waveformatex(b"\x00" * 8)


# --- downmix -----------------------------------------------------------------


def test_downmix_stereo_float32_averages_channels():
    fmt = MixFormat(48000, 2, 32, True)
    interleaved = np.array([1.0, 0.0, 0.5, -0.5, -1.0, 1.0], dtype="<f4")
    mono = downmix_to_mono_float32(interleaved.tobytes(), fmt)
    np.testing.assert_allclose(mono, [0.5, 0.0, 0.0], atol=1e-7)


def test_downmix_pcm16_scales_to_unit_range():
    fmt = MixFormat(16000, 1, 16, False)
    raw = np.array([32767, -32768, 0], dtype="<i2").tobytes()
    mono = downmix_to_mono_float32(raw, fmt)
    np.testing.assert_allclose(mono, [32767 / 32768.0, -1.0, 0.0], atol=1e-6)


# --- effects verdict ---------------------------------------------------------


def test_effects_verdict_requires_aec_present_and_on():
    on = {"id": GUID_EFFECT_AEC, "state": 1, "canSetState": False}
    off = {"id": GUID_EFFECT_AEC, "state": 0, "canSetState": True}
    ns = {"id": GUID_EFFECT_NS, "state": 1, "canSetState": False}
    assert effects_verdict([on, ns])["aec_active"] is True
    assert effects_verdict([off, ns])["aec_active"] is False
    assert effects_verdict([ns])["aec_active"] is False
    assert effects_verdict([])["aec_active"] is False
    # Brace/case-insensitive GUID comparison (comtypes str() uses braces).
    braced = {"id": GUID_EFFECT_AEC.lower(), "state": 1}
    assert effects_verdict([braced])["aec_active"] is True


# --- SampleRing --------------------------------------------------------------


def test_ring_take_reassembles_exact_frames_across_chunks():
    ring = SampleRing(1.0, 16000)
    ring.put(np.arange(5, dtype="float32"))
    ring.put(np.arange(5, 10, dtype="float32"))
    out, ovf = ring.take(7, timeout=0.5, fatal=lambda: None)
    assert out.shape == (7, 1)
    np.testing.assert_array_equal(out[:, 0], np.arange(7, dtype="float32"))
    assert ovf is False
    out2, _ = ring.take(3, timeout=0.5, fatal=lambda: None)
    np.testing.assert_array_equal(out2[:, 0], np.arange(7, 10, dtype="float32"))


def test_ring_overflow_drops_oldest_and_flags_once():
    ring = SampleRing(10 / 16000, 16000)  # capacity 10 samples
    ring.put(np.zeros(8, dtype="float32"))
    ring.put(np.ones(8, dtype="float32"))  # overflows: oldest chunk dropped
    out, ovf = ring.take(8, timeout=0.5, fatal=lambda: None)
    assert ovf is True
    np.testing.assert_array_equal(out[:, 0], np.ones(8, dtype="float32"))
    ring.put(np.zeros(4, dtype="float32"))
    _, ovf2 = ring.take(4, timeout=0.5, fatal=lambda: None)
    assert ovf2 is False  # flag reset after being reported


def test_ring_discontinuity_flag_reports_as_overflow():
    ring = SampleRing(1.0, 16000)
    ring.put(np.zeros(4, dtype="float32"), discontinuity=True)
    _, ovf = ring.take(4, timeout=0.5, fatal=lambda: None)
    assert ovf is True


def test_ring_timeout_raises_portaudio_shaped_error():
    ring = SampleRing(1.0, 16000)
    t0 = time.monotonic()
    with pytest.raises(Exception) as ei:
        ring.take(10, timeout=0.2, fatal=lambda: None)
    assert time.monotonic() - t0 < 2.0
    assert len(ei.value.args) >= 2 and isinstance(ei.value.args[1], int)


def test_ring_fatal_exception_propagates_from_take():
    boom = RuntimeError("pump died", -9985)
    ring = SampleRing(1.0, 16000)
    with pytest.raises(RuntimeError, match="pump died"):
        ring.take(10, timeout=1.0, fatal=lambda: boom)


def test_ring_close_unblocks_a_waiting_take():
    ring = SampleRing(1.0, 16000)
    got: list = []

    def _reader():
        try:
            ring.take(10, timeout=5.0, fatal=lambda: None)
        except Exception as exc:  # noqa: BLE001
            got.append(exc)

    t = threading.Thread(target=_reader)
    t.start()
    time.sleep(0.05)
    ring.close()
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert got and len(got[0].args) >= 2


# --- enroll route probe seam (ADR-0082 replaces "wasapi-pending") ------------


def test_enroll_windows_route_verified_by_aec_probe():
    mode = verify_required_os_echo_route(
        {"capture_voice_comm": True},
        platform="win32",
        wasapi_probe=lambda: {
            "aec_active": True,
            "ns_active": True,
            "effect_count": 3,
            "build": 26200,
        },
    )
    assert mode.startswith("wasapi-communications-aec:")
    assert "ns=on" in mode and "build=26200" in mode


def test_enroll_windows_route_fails_closed_without_aec():
    with pytest.raises(EnrollmentCaptureError, match="could not be verified"):
        verify_required_os_echo_route(
            {"capture_voice_comm": True},
            platform="win32",
            wasapi_probe=lambda: {"aec_active": False, "error": "no APO"},
        )


def test_enroll_non_voice_comm_windows_stays_none():
    assert (
        verify_required_os_echo_route({}, platform="win32", wasapi_probe=lambda: {})
        == "none"
    )
