"""WebRTC APM echo-cancel backend (core.engines._apm + build_aec("apm")).

The APM (AEC3+RES+NS+AGC2+HPF via the optional ``livekit`` package) is the
production echo canceller that tolerates a nonlinear open speaker. These tests
self-skip when ``livekit`` is absent (like the real_model tier), and assert the
build seam + fail-open behaviour with no dependency at all.
"""
import importlib.util

import numpy as np
import pytest

from core.engines.sherpa import SherpaConfig
from core.engines._aec import build_aec

_HAS_LIVEKIT = importlib.util.find_spec("livekit") is not None
_needs_livekit = pytest.mark.skipif(not _HAS_LIVEKIT, reason="livekit not installed")


def test_apm_disabled_when_aec_off():
    # aec_enabled False -> no canceller regardless of backend (byte-identical path).
    assert build_aec(SherpaConfig.from_dict({"aec_backend": "apm"})) is None


def test_apm_fails_open_without_livekit(monkeypatch):
    # Force the import to fail and confirm build_aec returns None (no crash).
    import core.engines._apm as apm_mod

    def _boom(*a, **k):
        raise ImportError("simulated missing livekit")

    monkeypatch.setattr(apm_mod, "_WebRTCAPM", _boom)
    c = SherpaConfig.from_dict({"aec_enabled": True, "aec_backend": "apm"})
    assert build_aec(c) is None


@_needs_livekit
def test_apm_build_carries_flags():
    c = SherpaConfig.from_dict({
        "aec_enabled": True, "aec_backend": "apm",
        "apm_always_on": True, "apm_noise_suppression": True,
    })
    ec = build_aec(c)
    assert ec is not None
    assert getattr(ec, "always_on") is True
    assert getattr(ec, "suppresses_noise") is True


@_needs_livekit
def test_apm_cancels_echo():
    from core.engines._apm import _WebRTCAPM

    sr, blk = 16000, 1600
    rng = np.random.default_rng(0)
    far = (0.5 * np.convolve(rng.standard_normal(3 * sr), np.ones(8) / 8, "same")).astype("float32")
    echo = np.zeros_like(far)
    echo[80:] = 0.6 * far[:-80]          # 5 ms delayed, attenuated speaker echo
    apm = _WebRTCAPM(echo_cancellation=True, noise_suppression=False,
                     high_pass_filter=False, gain_control=False)
    out = np.concatenate([apm.process(echo[i:i + blk], far[i:i + blk])
                          for i in range(0, len(far), blk)])

    def rms(x):
        return float(np.sqrt(np.mean(x.astype("float64") ** 2))) if len(x) else 0.0

    half = len(out) // 2
    erle_db = 20 * np.log10(rms(echo[-half:]) / max(rms(out[-half:]), 1e-9))
    assert erle_db > 20.0                # AEC3 crushes the echo (NLMS managed ~0)


@_needs_livekit
def test_apm_idle_passthrough():
    # When the assistant is silent (far == zeros) the user's voice must survive.
    from core.engines._apm import _WebRTCAPM

    sr, blk = 16000, 1600
    rng = np.random.default_rng(1)
    user = (0.2 * np.convolve(rng.standard_normal(2 * sr), np.ones(8) / 8, "same")).astype("float32")
    far = np.zeros_like(user)
    apm = _WebRTCAPM(echo_cancellation=True, noise_suppression=False,
                     high_pass_filter=False, gain_control=False)
    out = np.concatenate([apm.process(user[i:i + blk], far[i:i + blk])
                          for i in range(0, len(user), blk)])

    def rms(x):
        return float(np.sqrt(np.mean(x.astype("float64") ** 2)))

    half = len(out) // 2
    assert rms(out[-half:]) / rms(user[-half:]) > 0.7   # >70% retained when idle


@_needs_livekit
def test_apm_cancels_through_echocanceller_seam():
    # Guards the impl contract: EchoCanceller calls impl.process(near, far) (NOT
    # process_16k). If the APM method is misnamed the wrapper silently passes the
    # raw echo through (0 dB) instead of crashing -- so assert real cancellation
    # via the exact object the engine holds (build_aec -> EchoCanceller).
    c = SherpaConfig.from_dict({
        "aec_enabled": True, "aec_backend": "apm",
        "apm_noise_suppression": False, "apm_high_pass_filter": False,
    })
    ec = build_aec(c)
    assert ec is not None
    sr, blk = 16000, 1600
    rng = np.random.default_rng(0)
    far = (0.5 * np.convolve(rng.standard_normal(3 * sr), np.ones(8) / 8, "same")).astype("float32")
    echo = np.zeros_like(far)
    echo[80:] = 0.6 * far[:-80]
    out = np.concatenate([ec.process_16k(echo[i:i + blk], far[i:i + blk])
                          for i in range(0, len(far), blk)])

    def rms(x):
        return float(np.sqrt(np.mean(x.astype("float64") ** 2))) if len(x) else 0.0

    half = len(out) // 2
    erle_db = 20 * np.log10(rms(echo[-half:]) / max(rms(out[-half:]), 1e-9))
    assert erle_db > 20.0                # not the 0 dB error-passthrough


@_needs_livekit
def test_apm_output_length_framed():
    from core.engines._apm import _WebRTCAPM

    apm = _WebRTCAPM()
    # A sub-frame block (<160 samples) buffers and emits nothing yet.
    assert apm.process(np.zeros(100, "float32"), np.zeros(100, "float32")).size == 0
    # The next block completes a 10 ms frame (160 @ 16 kHz) -> one frame out.
    out = apm.process(np.zeros(100, "float32"), np.zeros(100, "float32"))
    assert out.size == 160
