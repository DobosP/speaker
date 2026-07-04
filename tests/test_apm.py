"""WebRTC APM echo-cancel backend (core.engines._apm + build_aec("apm")).

The APM (AEC3+RES+NS+AGC2+HPF via the optional ``livekit`` package) is the
production echo canceller that tolerates a nonlinear open speaker. These tests
self-skip when ``livekit`` is absent (like the real_model tier), and assert the
build seam + fail-open behaviour with no dependency at all.
"""
import importlib.util
import sys
import types

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
def test_apm_ns_override_forces_ns_off_tap():
    """fix 2: build_aec(ns_override=False) drops the ML noise-suppressor while
    echo cancellation stays on -- the parallel recognizer tap that lets near-end
    words survive. ns_override=None follows apm_noise_suppression (unchanged)."""
    c = SherpaConfig.from_dict({
        "aec_enabled": True, "aec_backend": "apm",
        "apm_always_on": True, "apm_noise_suppression": True,
    })
    assert getattr(build_aec(c, ns_override=None), "suppresses_noise") is True
    ec_off = build_aec(c, ns_override=False)
    assert ec_off is not None
    assert getattr(ec_off, "suppresses_noise") is False   # NS dropped for ASR
    assert getattr(ec_off, "always_on") is True           # still an always-on APM


def test_asr_relax_tap_absent_off_the_apm_owns_ns_path():
    """No APM / not _apm_owns_ns -> the engine's relaxed-NS ASR tap is None, so the
    recognizer reads the NS-on samples (byte-identical to before fix 2)."""
    from core.engines.sherpa import SherpaOnnxEngine

    assert SherpaOnnxEngine(SherpaConfig())._aec_asr is None


def test_nlms_is_not_resid_blind():
    """A genuinely LINEAR canceller (NLMS freezes adaptation on double-talk) leaves
    the near-end user in the residual, so it is NOT flagged suppresses_nearend --
    the DTD keeps reading the post-AEC residual (byte-identical)."""
    ec = build_aec(SherpaConfig.from_dict({"aec_enabled": True, "aec_backend": "nlms"}))
    assert ec is not None
    assert not getattr(ec, "suppresses_nearend", False)


@_needs_livekit
def test_apm_ns_sets_suppresses_nearend():
    """APM+NS masks the near-end user during double-talk -> suppresses_nearend True
    (the DTD reads the raw mic); NS off -> False."""
    on = build_aec(SherpaConfig.from_dict({
        "aec_enabled": True, "aec_backend": "apm",
        "apm_always_on": True, "apm_noise_suppression": True}))
    assert getattr(on, "suppresses_nearend", False) is True
    off = build_aec(SherpaConfig.from_dict({
        "aec_enabled": True, "aec_backend": "apm", "apm_noise_suppression": False}))
    assert getattr(off, "suppresses_nearend", True) is False


def test_dtln_sets_suppresses_nearend_if_model_present():
    """DTLN is a spectral-MASKING canceller (not linear) -> flagged suppresses_nearend
    so the DTD reads the raw mic. This is the live run-20260704-143112 barge miss:
    the DTLN residual pinned at 0 on a real talk-over. Skips when the ONNX is absent."""
    import os

    aec_model = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "pretrained_models", "sherpa", "aec")
    if not os.path.exists(os.path.join(aec_model, "dtln_aec_stage1.onnx")):
        pytest.skip("DTLN ONNX not present")
    ec = build_aec(SherpaConfig.from_dict(
        {"aec_enabled": True, "aec_backend": "dtln", "aec_model": aec_model}))
    if ec is None:
        pytest.skip("DTLN build failed (onnxruntime unavailable?)")
    assert getattr(ec, "suppresses_nearend", False) is True


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


def test_divergence_guard_exempts_amplifying_impl():
    # The "louder than input = diverged -> passthrough" guard must NOT fire for an
    # amplifying canceller (APM + AGC2 boosts quiet blocks by design). A non-amp
    # canceller still discards a >2x-louder output; an amplifying one keeps it.
    from core.engines._aec import EchoCanceller

    class _Loud:                       # returns the input scaled 3x (>2x guard)
        def process(self, near, far):
            return np.asarray(near, dtype="float32") * 3.0

    quiet = np.full(160, 0.05, dtype="float32")
    discarded = EchoCanceller(_Loud(), amplifies=False).process_16k(quiet, None)
    kept = EchoCanceller(_Loud(), amplifies=True).process_16k(quiet, None)

    def rms(x):
        return float(np.sqrt(np.mean(np.asarray(x, dtype="float64") ** 2)))

    assert rms(discarded) <= rms(quiet) * 1.01      # guard fired -> raw passthrough
    assert rms(kept) > rms(quiet) * 2.5             # amplified output preserved


def test_divergence_guard_still_catches_nonfinite_when_amplifying():
    # Even an amplifying impl must be discarded if it goes non-finite (NaN/inf).
    from core.engines._aec import EchoCanceller

    class _Nan:
        def process(self, near, far):
            out = np.asarray(near, dtype="float32").copy()
            out[0] = np.inf
            return out

    near = np.full(160, 0.1, dtype="float32")
    out = EchoCanceller(_Nan(), amplifies=True).process_16k(near, None)
    assert np.all(np.isfinite(out))                 # discarded -> raw near returned


@_needs_livekit
def test_apm_output_length_framed():
    from core.engines._apm import _WebRTCAPM

    apm = _WebRTCAPM()
    # A sub-frame block (<160 samples) buffers and emits nothing yet.
    assert apm.process(np.zeros(100, "float32"), np.zeros(100, "float32")).size == 0
    # The next block completes a 10 ms frame (160 @ 16 kHz) -> one frame out.
    out = apm.process(np.zeros(100, "float32"), np.zeros(100, "float32"))
    assert out.size == 160


def test_apm_process_orders_render_delay_then_capture(monkeypatch):
    from core.engines._apm import _WebRTCAPM

    calls = []
    livekit = types.ModuleType("livekit")
    rtc = types.SimpleNamespace()

    class _Frame:
        def __init__(self, data, sample_rate, num_channels, samples_per_channel):
            self.data = data
            self.sample_rate = sample_rate
            self.num_channels = num_channels
            self.samples_per_channel = samples_per_channel

    class _APM:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def process_reverse_stream(self, frame):
            calls.append(("reverse", frame.samples_per_channel))

        def set_stream_delay_ms(self, delay_ms):
            calls.append(("delay", delay_ms))

        def process_stream(self, frame):
            calls.append(("capture", frame.samples_per_channel))

    rtc.AudioFrame = _Frame
    rtc.AudioProcessingModule = _APM
    livekit.rtc = rtc
    monkeypatch.setitem(sys.modules, "livekit", livekit)

    apm = _WebRTCAPM(stream_delay_ms=37, sample_rate=16000)
    out = apm.process(np.full(160, 0.1, dtype="float32"), np.full(160, 0.2, dtype="float32"))

    assert out.size == 160
    assert calls == [
        (
            "init",
            {
                "echo_cancellation": True,
                "noise_suppression": True,
                "high_pass_filter": True,
                "auto_gain_control": False,
            },
        ),
        ("reverse", 160),
        ("delay", 37),
        ("capture", 160),
    ]
