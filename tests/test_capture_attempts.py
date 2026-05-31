"""Unit tests for the mic-open attempt ladder (`_capture_attempts`).

The crux of the AT2020USB-X fix: when the capture rate is PINNED, the engine must
open at exactly that rate and never probe 48000 first (a non-native open
reconfigures the USB device and trips its touch-mute). Pure logic, no hardware.
"""

from core.engines.sherpa import _capture_attempts


def _rates(attempts, *, device="MIC"):
    """Sample rates attempted on the preferred device, in order."""
    return [a.samplerate for a in attempts if a.device == device]


def test_pinned_opens_only_native_rate_first():
    attempts = _capture_attempts(
        "MIC",
        preferred_sr=16000,
        dev_sr_in=44100,
        pinned_sr=44100,
        clean_rates=(48000, 32000, 96000),
        supports=lambda d, r: True,  # everything "supported" — must still be skipped
    )
    # First open is the pinned native rate...
    assert attempts[0].device == "MIC"
    assert attempts[0].samplerate == 44100
    # ...and 48000 (the rate that re-mutes the AT2020) is NEVER attempted on the mic.
    assert 48000 not in _rates(attempts)
    assert 16000 not in _rates(attempts)  # the rejected/native-reconfiguring rate skipped too


def test_pinned_adds_native_backstop_when_distinct():
    attempts = _capture_attempts(
        "MIC", preferred_sr=16000, dev_sr_in=48000, pinned_sr=44100,
    )
    mic_rates = _rates(attempts)
    assert mic_rates[0] == 44100  # pinned first
    assert 48000 in mic_rates  # device-native added as a backstop (distinct from pinned)
    assert mic_rates.index(44100) < mic_rates.index(48000)


def test_auto_mode_prefers_clean_then_native():
    attempts = _capture_attempts(
        "MIC",
        preferred_sr=16000,
        dev_sr_in=44100,
        pinned_sr=0,
        clean_rates=(48000, 32000, 96000),
        supports=lambda d, r: r == 48000,  # only 48000 is supported among clean rates
    )
    mic_rates = _rates(attempts)
    assert mic_rates[0] == 16000  # preferred first
    assert 48000 in mic_rates  # clean integer-ratio rate
    assert mic_rates[-1] == 44100  # native last on the preferred device
    assert 32000 not in mic_rates  # unsupported clean rate skipped


def test_auto_mode_skips_unsupported_clean_rates():
    attempts = _capture_attempts(
        "MIC", preferred_sr=16000, dev_sr_in=44100, pinned_sr=0,
        clean_rates=(48000,), supports=lambda d, r: False,
    )
    assert 48000 not in _rates(attempts)


def test_system_default_backstop_present():
    attempts = _capture_attempts(
        "MIC", preferred_sr=16000, dev_sr_in=44100, pinned_sr=44100,
    )
    # A None-device backstop exists so a mid-session reopen can still recover.
    assert any(a.device is None for a in attempts)
