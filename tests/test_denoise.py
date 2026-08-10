"""Speech-denoise front-end: config gating, the wrapper, and the capture hook.

No ONNX model and no audio device: the denoiser is dependency-injected (a fake
that records its input and returns a transformed block), and the capture loop is
driven for exactly ONE block over a fake input stream + fake recognizer so we can
assert what ``accept_waveform`` actually saw.

Two load-bearing invariants:
* DISABLED (no denoiser built) -> the block reaching ``accept_waveform`` is
  byte-identical to the captured block (the path is unchanged from no-denoise).
* ENABLED (fake denoiser injected) -> the block reaching ``accept_waveform`` is
  the denoiser's OUTPUT, not the raw input (and the recorder sees it too).
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

from core.engines._denoiser import Denoiser, build_denoiser
from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine
from core.engine import EngineCallbacks


# --- SherpaConfig fields default off -----------------------------------------


def test_denoise_fields_default_off():
    c = SherpaConfig()
    assert c.denoise_enabled is False
    assert c.denoise_model == ""


def test_denoise_fields_parse_from_dict():
    c = SherpaConfig.from_dict({"denoise_enabled": True, "denoise_model": "/m/gtcrn.onnx"})
    assert c.denoise_enabled is True
    assert c.denoise_model == "/m/gtcrn.onnx"


def test_shipped_config_has_denoise_on():
    # Fleet-default DECISION 2026-07-08 (owner): denoise ships ON after it won the
    # round-2 capture-calibration double-talk test on the Linux ROG box (STATUS.md +
    # calib_runs/20260708-010952/GRADES.md). A model path ships too so it is live on
    # a machine that has fetched the 523KB GTCRN model; build_denoiser FAILS OPEN, so
    # a machine WITHOUT the model degrades to no-denoise (see the fail-open tests
    # below), never a crash.
    import json
    import os

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    sherpa = cfg["sherpa"]
    assert sherpa["denoise_enabled"] is True
    assert sherpa["denoise_model"].endswith("gtcrn_simple.onnx")


# --- build_denoiser factory (no model -> None) -------------------------------


def test_build_denoiser_none_when_disabled():
    # Even with a model path, disabled -> None (build nothing).
    assert build_denoiser(SherpaConfig(denoise_enabled=False, denoise_model="/m/x.onnx")) is None


def test_build_denoiser_none_when_no_model():
    # Enabled but no path -> None (nothing to load).
    assert build_denoiser(SherpaConfig(denoise_enabled=True, denoise_model="")) is None


def test_build_denoiser_fails_open_on_bad_path(monkeypatch):
    # Enabled + a path, but the eager ctor RAISES (bad path) -> None, not a crash.
    import sys
    import types

    mod = types.ModuleType("sherpa_onnx")

    class _Cfg:
        def __init__(self):
            self.model = types.SimpleNamespace(
                gtcrn=types.SimpleNamespace(model=None), num_threads=None, provider=None
            )

    def _ctor(config):
        raise RuntimeError("no such file: bad.onnx")

    mod.OnlineSpeechDenoiserConfig = _Cfg
    mod.OnlineSpeechDenoiser = _ctor
    monkeypatch.setitem(sys.modules, "sherpa_onnx", mod)

    assert build_denoiser(SherpaConfig(denoise_enabled=True, denoise_model="bad.onnx")) is None


def test_build_denoiser_wires_config(monkeypatch):
    import sys
    import types

    captured = {}
    mod = types.ModuleType("sherpa_onnx")

    class _Cfg:
        def __init__(self):
            self.model = types.SimpleNamespace(
                gtcrn=types.SimpleNamespace(model=None), num_threads=None, provider=None
            )

    class _Impl:
        def __init__(self, config):
            captured["model"] = config.model.gtcrn.model
            captured["num_threads"] = config.model.num_threads
            captured["provider"] = config.model.provider

    mod.OnlineSpeechDenoiserConfig = _Cfg
    mod.OnlineSpeechDenoiser = _Impl
    monkeypatch.setitem(sys.modules, "sherpa_onnx", mod)

    d = build_denoiser(
        SherpaConfig(
            denoise_enabled=True, denoise_model="gtcrn.onnx", provider="cpu", asr_num_threads=3
        )
    )
    assert isinstance(d, Denoiser)
    assert captured == {"model": "gtcrn.onnx", "num_threads": 3, "provider": "cpu"}


# --- Denoiser wrapper --------------------------------------------------------


class _FakeImpl:
    """Duck-typed denoiser: scales the block (a visible, invertible transform)."""

    def __init__(self, scale=0.5):
        self.scale = scale
        self.seen = []
        self.reset_calls = 0

    def run(self, samples, sample_rate):
        self.seen.append((np.asarray(samples, dtype="float32").copy(), sample_rate))
        out = np.asarray(samples, dtype="float32") * self.scale
        return type("D", (), {"samples": out, "sample_rate": sample_rate})()

    def reset(self):
        self.reset_calls += 1


def test_process_16k_applies_impl():
    d = Denoiser(_FakeImpl(scale=0.5), sample_rate=16000)
    block = np.array([1.0, 2.0, 4.0], dtype="float32")
    out = d.process_16k(block)
    assert np.allclose(out, [0.5, 1.0, 2.0])
    # Ran at 16 kHz.
    assert d._impl.seen[0][1] == 16000


def test_denoise_process_16k_passthrough_on_error():
    class _Boom:
        def run(self, samples, sample_rate):
            raise RuntimeError("onnx exploded")

    d = Denoiser(_Boom())
    block = np.array([1.0, 2.0, 3.0], dtype="float32")
    out = d.process_16k(block)
    # The ORIGINAL block comes back untouched -- the capture thread never dies.
    assert out is block


def test_reset_forwards_and_is_best_effort():
    impl = _FakeImpl()
    d = Denoiser(impl)
    d.reset()
    assert impl.reset_calls == 1
    # An impl without reset() must not raise.
    Denoiser(object()).reset()


# --- capture-loop hook: byte-identical disabled / applied when injected -------


class _FakeStream:
    """One block then stop the loop (avoids an infinite capture thread)."""

    def __init__(self, block, engine):
        self._block = block
        self._engine = engine
        self._served = False

    def read(self, n):
        if self._served:
            # Second read -> end the loop deterministically.
            self._engine._running.clear()
            return np.zeros(0, dtype="float32"), False
        self._served = True
        return self._block.copy(), False


class _FakeBlockStream:
    """Serve a finite sequence of blocks, then stop the loop."""

    def __init__(self, blocks, engine):
        self._blocks = list(blocks)
        self._engine = engine
        self._idx = 0

    def read(self, n):
        if self._idx >= len(self._blocks):
            self._engine._running.clear()
            return np.zeros(0, dtype="float32"), False
        block = self._blocks[self._idx]
        self._idx += 1
        if self._idx >= len(self._blocks):
            self._engine._running.clear()
        return np.asarray(block, dtype="float32").copy(), False


class _FakeASRStream:
    """Records every block handed to accept_waveform (the loop calls it on the
    STREAM, not the recognizer)."""

    def __init__(self, sink):
        self._sink = sink

    def accept_waveform(self, sample_rate, samples):
        self._sink.append(np.asarray(samples, dtype="float32").copy())


class _FakeRecognizer:
    """Hands out a stream that records blocks; never reports ready/endpoint."""

    def __init__(self):
        self.blocks = []

    def create_stream(self):
        return _FakeASRStream(self.blocks)

    def is_ready(self, stream):
        return False

    def get_result(self, stream):
        return ""

    def is_endpoint(self, stream):
        return False

    def reset(self, stream):
        pass


def _run_one_block(engine, block):
    """Drive _capture_loop for exactly one captured block."""
    rec = _FakeRecognizer()
    engine._recognizer = rec
    engine._cb = EngineCallbacks()
    engine._stream_in = _FakeStream(block, engine)
    engine._running.set()
    t = threading.Thread(target=engine._capture_loop)
    t.start()
    t.join(timeout=5.0)
    assert not t.is_alive(), "capture loop did not exit"
    return rec


def _run_blocks(engine, blocks):
    """Drive _capture_loop over a finite sequence of captured blocks."""
    rec = _FakeRecognizer()
    engine._recognizer = rec
    engine._cb = EngineCallbacks()
    engine._stream_in = _FakeBlockStream(blocks, engine)
    engine._running.set()
    t = threading.Thread(target=engine._capture_loop)
    t.start()
    t.join(timeout=5.0)
    assert not t.is_alive(), "capture loop did not exit"
    return rec


def test_capture_hook_byte_identical_when_disabled():
    eng = SherpaOnnxEngine(SherpaConfig())
    assert eng._denoiser is None  # nothing built
    block = np.array([0.1, -0.2, 0.3, -0.4], dtype="float32")
    rec = _run_one_block(eng, block)
    assert rec.blocks, "accept_waveform was never called"
    # accept_waveform saw the captured block UNCHANGED (byte-identical path).
    np.testing.assert_array_equal(rec.blocks[0], block)


def test_capture_hook_applies_injected_denoiser():
    eng = SherpaOnnxEngine(SherpaConfig())
    impl = _FakeImpl(scale=0.5)
    eng._denoiser = Denoiser(impl, sample_rate=16000)
    block = np.array([0.2, 0.4, 0.8, 1.0], dtype="float32")
    rec = _run_one_block(eng, block)
    assert rec.blocks, "accept_waveform was never called"
    # accept_waveform saw the DENOISER'S OUTPUT, not the raw block.
    np.testing.assert_allclose(rec.blocks[0], block * 0.5)
    # And the denoiser actually ran on the captured block at 16 kHz.
    assert impl.seen and impl.seen[0][1] == 16000
    np.testing.assert_array_equal(impl.seen[0][0], block)


def test_always_on_apm_feeds_asr_cleaned_block_and_skips_denoiser():
    """The apm_always_on capture path must (a) feed ASR the AEC OUTPUT (not the raw
    mic) on EVERY block, and (b) skip the GTCRN denoiser when the APM owns NS --
    on a realistic 1600-sample block, with finite non-empty output. No livekit: a
    fake AEC stands in for the real WebRTC APM."""
    from core.engines._aec import FarEndRing

    class _FakeAEC:
        always_on = True
        suppresses_noise = True

        def __init__(self):
            self.calls = 0

        def process_16k(self, near, far):
            self.calls += 1
            return np.asarray(near, dtype="float32") * 0.5

    eng = SherpaOnnxEngine(SherpaConfig())
    aec = _FakeAEC()
    eng._aec = aec
    eng._apm_always_on = True
    eng._apm_owns_ns = True
    eng._far_ref = FarEndRing()
    den_impl = _FakeImpl(scale=0.25)
    eng._denoiser = Denoiser(den_impl, sample_rate=16000)
    block = np.full(1600, 0.2, dtype="float32")
    rec = _run_one_block(eng, block)
    assert rec.blocks, "accept_waveform was never called"
    out = rec.blocks[0]
    np.testing.assert_allclose(out, block * 0.5)        # ASR saw the APM output, not raw
    assert aec.calls >= 1                               # the APM ran on the block
    assert not den_impl.seen                            # denoiser SKIPPED (apm owns NS)
    assert out.size > 0 and np.all(np.isfinite(out))


def _stub_optional_sherpa_builders(monkeypatch, sherpa_mod) -> None:
    for name in (
        "build_recognizer",
        "build_final_recognizer",
        "build_final_verifier",
        "build_vad",
        "build_tts",
        "build_denoiser",
        "build_keyword_spotter",
        "build_punctuation",
    ):
        monkeypatch.setattr(sherpa_mod, name, lambda c: None)


@pytest.mark.parametrize(
    (
        "always_on",
        "suppresses_noise",
        "suppresses_nearend",
        "relax_enabled",
        "expected_owns_ns",
        "expected_resid_blind",
        "expected_relaxed_tap",
    ),
    (
        (False, False, False, True, False, False, False),
        (False, False, True, True, False, True, False),
        (False, True, False, True, False, False, False),
        (False, True, True, True, False, True, False),
        (True, False, False, True, False, False, False),
        (True, False, True, True, False, True, False),
        (True, True, False, True, True, True, True),
        (True, True, True, True, True, True, True),
        (True, True, False, False, True, True, False),
    ),
)
def test_apm_ownership_is_derived_from_current_primary_aec(
    monkeypatch,
    always_on,
    suppresses_noise,
    suppresses_nearend,
    relax_enabled,
    expected_owns_ns,
    expected_resid_blind,
    expected_relaxed_tap,
):
    import core.engines.sherpa as sherpa_mod

    class _FakeAEC:
        def __init__(self):
            self.always_on = always_on
            self.suppresses_noise = suppresses_noise
            self.suppresses_nearend = suppresses_nearend

    primary = _FakeAEC()
    relaxed = object()
    build_calls = []

    def _build_aec(_config, *, ns_override=None):
        build_calls.append(ns_override)
        return relaxed if ns_override is False else primary

    _stub_optional_sherpa_builders(monkeypatch, sherpa_mod)
    monkeypatch.setattr(sherpa_mod, "build_aec", _build_aec)

    eng = SherpaOnnxEngine(
        SherpaConfig.from_dict(
            {
                "aec_enabled": True,
                "aec_auto_delay": False,
                "coherence_barge_in_enabled": False,
                "dtd_enabled": False,
                "apm_asr_relax_ns": relax_enabled,
            }
        )
    )
    eng._build()

    assert eng._apm_always_on is always_on
    assert eng._apm_owns_ns is expected_owns_ns
    assert eng._resid_blind is expected_resid_blind
    assert (eng._aec_asr is relaxed) is expected_relaxed_tap
    assert build_calls == ([None, False] if expected_relaxed_tap else [None])


def test_rebuild_clears_optional_audio_state_after_primary_aec_fails_open(
    monkeypatch,
):
    import core.engines.sherpa as sherpa_mod

    class _PrimaryAEC:
        always_on = True
        suppresses_noise = True
        suppresses_nearend = True

    class _Detector:
        def __init__(self, available):
            self.available = available

    primary = _PrimaryAEC()
    relaxed = object()
    first_detector = _Detector(True)
    unavailable_detector = _Detector(False)
    first_dtd = object()
    primary_builds = 0
    detectors = iter((first_detector, unavailable_detector))

    def _build_aec(_config, *, ns_override=None):
        nonlocal primary_builds
        if ns_override is False:
            return relaxed
        primary_builds += 1
        return primary if primary_builds == 1 else None

    _stub_optional_sherpa_builders(monkeypatch, sherpa_mod)
    monkeypatch.setattr(sherpa_mod, "build_aec", _build_aec)
    monkeypatch.setattr(
        sherpa_mod,
        "EchoCoherenceDetector",
        lambda *_args, **_kwargs: next(detectors),
    )
    monkeypatch.setattr(
        sherpa_mod,
        "AdaptiveDTD",
        lambda **_kwargs: first_dtd,
    )

    eng = SherpaOnnxEngine(
        SherpaConfig.from_dict(
            {
                "aec_enabled": True,
                "aec_auto_delay": True,
                "coherence_barge_in_enabled": True,
                "dtd_enabled": True,
                "apm_asr_relax_ns": True,
            }
        )
    )
    eng._build()

    first_playback_ref = eng._playback_ref
    assert eng._aec is primary
    assert eng._echo_coherence is first_detector
    assert eng._dtd is first_dtd
    assert eng._apm_owns_ns is True
    assert eng._resid_blind is True
    assert eng._aec_asr is relaxed
    assert eng._aec_delay_cal is not None
    assert eng._aec_ref_delay > 0

    eng._build()

    assert eng._aec is None
    assert eng._playback_ref is not first_playback_ref
    assert eng._far_ref is None
    assert eng._echo_coherence is None
    assert eng._dtd is None
    assert eng._apm_always_on is False
    assert eng._apm_owns_ns is False
    assert eng._resid_blind is False
    assert eng._aec_asr is None
    assert eng._aec_delay_cal is None
    assert eng._aec_ref_delay == 0


def test_rebuild_drops_disabled_relaxed_tap_and_auto_delay(monkeypatch):
    import core.engines.sherpa as sherpa_mod

    class _PrimaryAEC:
        always_on = True
        suppresses_noise = True
        suppresses_nearend = True

    primary = _PrimaryAEC()
    relaxed = object()
    build_calls = []

    def _build_aec(_config, *, ns_override=None):
        build_calls.append(ns_override)
        return relaxed if ns_override is False else primary

    _stub_optional_sherpa_builders(monkeypatch, sherpa_mod)
    monkeypatch.setattr(sherpa_mod, "build_aec", _build_aec)

    eng = SherpaOnnxEngine(
        SherpaConfig.from_dict(
            {
                "aec_enabled": True,
                "aec_auto_delay": True,
                "coherence_barge_in_enabled": False,
                "dtd_enabled": False,
                "apm_asr_relax_ns": True,
            }
        )
    )
    eng._build()
    assert eng._aec_asr is relaxed
    assert eng._aec_delay_cal is not None

    eng.config.apm_asr_relax_ns = False
    eng.config.aec_auto_delay = False
    eng._build()

    assert eng._aec is primary
    assert eng._apm_owns_ns is True
    assert eng._resid_blind is True
    assert eng._aec_asr is None
    assert eng._aec_delay_cal is None
    assert eng._aec_ref_delay == int(
        eng.config.sample_rate * eng.config.aec_ref_delay_ms / 1000
    )
    assert build_calls == [None, False, None]


def test_aec_auto_delay_wires_calibrator_into_ref_delay():
    """The capture loop feeds AecDelayCalibrator(mic_raw, far0) and adopts its
    measured delay as the operating ``_aec_ref_delay`` -- replacing the old
    coherence-median feedback (un-normalized, no accept-gate, reset every reply)
    that never converged on the open speaker. The calibrator's own convergence
    and correlation-gating are unit-tested in tests/test_aec_seam.py; here we
    assert only the engine wiring."""
    from core.engines._aec import FarEndRing

    class _FakeAEC:
        def process_16k(self, near, far):
            return np.asarray(near, dtype="float32")

    class _FakeCalibrator:
        def __init__(self):
            self.observed = 0
            self.delay = 333

        def observe(self, mic_block, far0_block):
            self.observed += 1

        def current_delay_samples(self):
            return self.delay

    eng = SherpaOnnxEngine(SherpaConfig.from_dict({"aec_auto_delay": True}))
    eng._aec = _FakeAEC()
    eng._far_ref = FarEndRing()
    eng._speaking.set()
    cal = _FakeCalibrator()
    eng._aec_delay_cal = cal
    eng._aec_ref_delay = 80
    block = np.full(1600, 0.05, dtype="float32")

    _run_blocks(eng, [block] * 3)
    assert cal.observed >= 1           # the capture loop fed the calibrator
    assert eng._aec_ref_delay == 333   # and adopted its measured operating delay

    # No calibrator (auto-delay off / not built) -> the seed is left untouched.
    eng._aec_delay_cal = None
    eng._aec_ref_delay = 80
    _run_blocks(eng, [block] * 3)
    assert eng._aec_ref_delay == 80


def test_capture_hook_records_aligned_pre_dsp_and_denoised_blocks():
    eng = SherpaOnnxEngine(SherpaConfig())
    eng._denoiser = Denoiser(_FakeImpl(scale=0.25), sample_rate=16000)

    class _Recorder:
        def __init__(self):
            self.written = []

        def write(self, samples):
            self.written.append(np.asarray(samples, dtype="float32").copy())

    eng._recorder = _Recorder()
    eng._pre_dsp_recorder = _Recorder()
    block = np.array([0.4, 0.8, 1.2], dtype="float32")
    _run_one_block(eng, block)
    assert eng._recorder.written, "recorder never wrote"
    assert eng._pre_dsp_recorder.written, "pre-DSP recorder never wrote"
    # The diagnostic sidecar owns the model-rate capture before application
    # gain/AEC/GTCRN; it can therefore isolate front-end damage from ASR policy.
    np.testing.assert_allclose(eng._pre_dsp_recorder.written[0], block)
    # The recorder writes the DENOISED block (so a recording is already denoised).
    np.testing.assert_allclose(eng._recorder.written[0], block * 0.25)
    assert (
        eng._pre_dsp_recorder.written[0].shape
        == eng._recorder.written[0].shape
    )
