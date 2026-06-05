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


def test_shipped_config_has_denoise_off():
    import json
    import os

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    sherpa = cfg["sherpa"]
    assert sherpa["denoise_enabled"] is False
    assert sherpa["denoise_model"] == ""


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


def test_capture_hook_recorder_sees_denoised_block():
    eng = SherpaOnnxEngine(SherpaConfig())
    eng._denoiser = Denoiser(_FakeImpl(scale=0.25), sample_rate=16000)

    class _Recorder:
        def __init__(self):
            self.written = []

        def write(self, samples):
            self.written.append(np.asarray(samples, dtype="float32").copy())

    eng._recorder = _Recorder()
    block = np.array([0.4, 0.8, 1.2], dtype="float32")
    _run_one_block(eng, block)
    assert eng._recorder.written, "recorder never wrote"
    # The recorder writes the DENOISED block (so a recording is already denoised).
    np.testing.assert_allclose(eng._recorder.written[0], block * 0.25)
