"""Tests for the on-device AEC (core/engines/_aec.py). Pure NumPy, no audio device
and no model: the build gate, the passthrough-on-error seam, the far-end ring
alignment math, and the adaptive filter's single-talk cancellation + double-talk
freeze + reset."""
from __future__ import annotations

import os

import numpy as np
import pytest

from core.engines._aec import (
    EchoCanceller,
    FarEndRing,
    _DTLNEchoCanceller,
    _FDAFAdaptiveFilter,
    _resolve_dtln_paths,
    build_aec,
)
from core.engines.sherpa import SherpaConfig


# --- build gate --------------------------------------------------------------


def test_build_returns_none_when_disabled():
    assert build_aec(SherpaConfig.from_dict({})) is None
    assert build_aec(SherpaConfig.from_dict({"aec_enabled": False})) is None


def test_build_nlms_when_enabled():
    aec = build_aec(SherpaConfig.from_dict({"aec_enabled": True, "aec_backend": "nlms"}))
    assert isinstance(aec, EchoCanceller)


def test_build_dtln_and_unknown_fail_open_to_none():
    # The deep ONNX tier is deferred -> no-op (NOT a crash); unknown backend too.
    assert build_aec(SherpaConfig.from_dict({"aec_enabled": True, "aec_backend": "dtln"})) is None
    assert build_aec(SherpaConfig.from_dict({"aec_enabled": True, "aec_backend": "bogus"})) is None


# --- passthrough-on-error seam ----------------------------------------------


class _RaisingImpl:
    def process(self, near, far):
        raise RuntimeError("boom")


def test_process_16k_passthrough_on_error():
    ec = EchoCanceller(_RaisingImpl())
    near = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = ec.process_16k(near, np.zeros(3, dtype=np.float32))
    assert np.array_equal(out, near)  # never crash the capture daemon


def test_reset_is_safe_on_impl_without_reset():
    EchoCanceller(_RaisingImpl()).reset()  # no reset() on impl -> no-op, no raise


# --- far-end ring alignment math --------------------------------------------


def test_ring_reads_most_recent_window_at_zero_delay():
    ring = FarEndRing(capacity=100)
    ring.push(np.arange(50, dtype=np.float32))
    np.testing.assert_array_equal(ring.read(10, 0), np.arange(40, 50))


def test_ring_delay_offsets_the_window():
    ring = FarEndRing(capacity=100)
    ring.push(np.arange(50, dtype=np.float32))
    np.testing.assert_array_equal(ring.read(10, 5), np.arange(35, 45))


def test_ring_zeros_before_anything_played():
    assert np.all(FarEndRing(100).read(10, 0) == 0.0)


def test_ring_eviction_returns_recent_and_zeros_for_evicted():
    ring = FarEndRing(capacity=100)
    ring.push(np.arange(200, dtype=np.float32))  # only the last 100 survive
    np.testing.assert_array_equal(ring.read(10, 0), np.arange(190, 200))
    assert np.all(ring.read(10, 150) == 0.0)  # window fell off the back -> zeros


def test_ring_clear():
    ring = FarEndRing(100)
    ring.push(np.arange(50, dtype=np.float32))
    ring.clear()
    assert np.all(ring.read(10, 0) == 0.0)


# --- adaptive filter: single-talk cancellation -------------------------------


def _echo(far, fir):
    return np.convolve(far, fir)[: len(far)].astype(np.float32)


def _run(filt, near, far, block):
    out = []
    for i in range(0, len(near) - block, block):
        out.append(filt.process(near[i : i + block], far[i : i + block]))
    return np.concatenate(out)


def test_fdaf_cancels_linear_echo_single_talk():
    rng = np.random.default_rng(0)
    far = (rng.standard_normal(16000 * 2) * 0.3).astype(np.float32)  # 2 s
    fir = np.zeros(200, dtype=np.float64)
    fir[5], fir[40], fir[120] = 0.6, 0.3, 0.15  # sparse decaying room response
    near = _echo(far, fir)  # single-talk: mic hears only the echo
    e = _run(_FDAFAdaptiveFilter(frame=512), near, far, 512)
    d = near[: len(e)]
    half = len(e) // 2  # measure ERLE on the converged tail
    erle = 10 * np.log10(np.sum(d[half:] ** 2) / (np.sum(e[half:] ** 2) + 1e-12))
    assert np.isfinite(erle) and erle > 20.0  # >=20 dB cancellation


def test_fdaf_no_far_is_passthrough():
    # Nothing playing (silent far) -> no echo to cancel -> near returned ~unchanged.
    rng = np.random.default_rng(1)
    near = (rng.standard_normal(2048) * 0.2).astype(np.float32)
    far = np.zeros_like(near)
    e = _run(_FDAFAdaptiveFilter(frame=512), near, far, 512)
    np.testing.assert_allclose(e, near[: len(e)], atol=1e-5)


# --- adaptive filter: double-talk freeze + reset -----------------------------


def test_fdaf_freezes_on_double_talk_and_does_not_diverge():
    rng = np.random.default_rng(2)
    fir = np.zeros(200, dtype=np.float64)
    fir[5], fir[40] = 0.6, 0.25
    filt = _FDAFAdaptiveFilter(frame=512, doubletalk_freeze=True, warmup_frames=4)
    # Converge on single-talk first.
    far = (rng.standard_normal(512 * 24) * 0.3).astype(np.float32)
    near = _echo(far, fir)
    _run(filt, near, far, 512)
    w_before = filt.W.copy()
    # Now one DOUBLE-TALK frame: echo + a loud near-end voice.
    f = (rng.standard_normal(512) * 0.3).astype(np.float32)
    echo = _echo(np.concatenate([far[-512:], f]), fir)[-512:]
    speech = (rng.standard_normal(512) * 1.5).astype(np.float32)  # loud near-end
    e = filt.process(echo + speech, f)
    # Froze -> filter weights essentially unchanged (didn't adapt onto the voice)...
    assert np.linalg.norm(filt.W - w_before) < 1e-6
    # ...and the near-end speech survives in the output (not cancelled away).
    assert np.sqrt(np.mean(e**2)) > 0.5


def test_fdaf_reset_clears_state():
    filt = _FDAFAdaptiveFilter(frame=256)
    rng = np.random.default_rng(3)
    far = (rng.standard_normal(2048) * 0.3).astype(np.float32)
    _run(filt, _echo(far, np.array([0.0, 0.5, 0.2])), far, 256)
    assert filt.count > 0 and np.any(filt.W != 0)
    filt.reset()
    assert filt.count == 0 and np.all(filt.W == 0)


# --- DTLN ONNX backend (fake sessions: no model, no onnxruntime) -------------


class _FakeIn:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class _FakeStage:
    """Mimics a DTLN-aec ONNX stage: inputs [primary, state(4-D), reference];
    outputs [main, state(4-D)]. Returns a passthrough main (ones mask for stage 1,
    the primary block for stage 2) and increments the LSTM state by 1 so a test can
    confirm the state is carried block-to-block."""

    def __init__(self, feat_dim: int, *, passthrough_primary: bool):
        self._feat = feat_dim
        self._pass = passthrough_primary
        self._ins = [
            _FakeIn("primary", [1, 1, feat_dim]),
            _FakeIn("state", [1, 2, 512, 2]),
            _FakeIn("reference", [1, 1, feat_dim]),
        ]
        self._outs = [_FakeIn("main", [1, 1, feat_dim]), _FakeIn("state_out", [1, 2, 512, 2])]
        self.calls = 0

    def get_inputs(self):
        return self._ins

    def get_outputs(self):
        return self._outs

    def run(self, out_names, feed):
        self.calls += 1
        state = np.asarray(feed["state"], dtype=np.float32)
        main = (
            np.asarray(feed["primary"], dtype=np.float32)
            if self._pass
            else np.ones((1, 1, self._feat), dtype=np.float32)
        )
        return [main, state + 1.0]


def _fake_dtln():
    return _DTLNEchoCanceller(
        sessions=(_FakeStage(257, passthrough_primary=False), _FakeStage(512, passthrough_primary=True))
    )


def test_dtln_io_mapping_by_shape():
    aec = _fake_dtln()
    assert aec._m1["primary"] == "primary" and aec._m1["reference"] == "reference"
    assert aec._m1["state_in"] == "state" and aec._m1["state_shape"] == (1, 2, 512, 2)
    assert aec._m1["main_out"] == "main" and aec._m1["state_out"] == "state_out"
    assert aec._m2["primary"] == "primary" and aec._m2["state_shape"] == (1, 2, 512, 2)


def test_dtln_streams_in_128_hops_and_carries_state():
    aec = _fake_dtln()
    out = aec.process(np.zeros(1600, dtype=np.float32), np.zeros(1600, dtype=np.float32))
    # 1600 / 128 = 12 full hops -> 12 block-steps -> 12*128 output samples.
    assert aec._s1.calls == 12 and aec._s2.calls == 12
    assert out.shape[0] == 12 * 128
    # State carried: the fake adds 1 each step, so after 12 steps it reads 12.
    assert np.allclose(aec.st1, 12.0) and np.allclose(aec.st2, 12.0)


def test_dtln_reset_clears_state_and_buffers():
    aec = _fake_dtln()
    aec.process(np.zeros(1600, dtype=np.float32), np.zeros(1600, dtype=np.float32))
    assert np.any(aec.st1 != 0)
    aec.reset()
    assert np.all(aec.st1 == 0) and np.all(aec.st2 == 0)
    assert aec._nq.shape[0] == 0 and aec._fq.shape[0] == 0


# --- DTLN model-path resolver ------------------------------------------------


def test_resolve_dtln_paths_directory(tmp_path):
    (tmp_path / "dtln_aec_stage1.onnx").write_bytes(b"x")
    (tmp_path / "dtln_aec_stage2.onnx").write_bytes(b"x")
    s1, s2 = _resolve_dtln_paths(str(tmp_path))
    assert s1.endswith("dtln_aec_stage1.onnx") and s2.endswith("dtln_aec_stage2.onnx")


def test_resolve_dtln_paths_missing_returns_none(tmp_path):
    assert _resolve_dtln_paths("") is None
    assert _resolve_dtln_paths(str(tmp_path)) is None  # empty dir


def test_resolve_dtln_paths_stage1_file_derives_stage2(tmp_path):
    (tmp_path / "dtln_aec_stage1.onnx").write_bytes(b"x")
    (tmp_path / "dtln_aec_stage2.onnx").write_bytes(b"x")
    s1, s2 = _resolve_dtln_paths(str(tmp_path / "dtln_aec_stage1.onnx"))
    assert s1.endswith("stage1.onnx") and s2.endswith("stage2.onnx")


# --- DTLN real inference (skips unless the converted model + onnxruntime exist)


_DTLN1 = os.path.join("pretrained_models", "sherpa", "aec", "dtln_aec_stage1.onnx")


@pytest.mark.skipif(
    not os.path.exists(_DTLN1),
    reason="DTLN-aec ONNX not present (run tools.setup_models --aec-model)",
)
def test_dtln_real_inference_reduces_echo():
    pytest.importorskip("onnxruntime")
    aec = _DTLNEchoCanceller(_DTLN1, _DTLN1.replace("stage1", "stage2"))
    rng = np.random.default_rng(0)
    raw = rng.standard_normal(16000 * 2).astype(np.float32)
    far = (np.convolve(raw, np.ones(8) / 8)[: len(raw)] * 0.3).astype(np.float32)
    fir = np.zeros(150, dtype=np.float64)
    fir[10], fir[60] = 0.5, 0.25
    mic = np.convolve(far, fir)[: len(far)].astype(np.float32)
    out = np.concatenate([aec.process(mic[i : i + 1600], far[i : i + 1600]) for i in range(0, len(mic) - 1600, 1600)])
    assert np.all(np.isfinite(out))
    tail = len(out) // 2
    erle = 10 * np.log10(np.sum(mic[tail:len(out)] ** 2) / (np.sum(out[tail:] ** 2) + 1e-12))
    assert erle > 15.0  # deep canceller -> strong reduction on single-talk echo
