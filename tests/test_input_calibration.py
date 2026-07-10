"""Startup ambient calibration (core.audio_frontend.compute_input_calibration).

The device-generic operating-point step: measure THIS mic's quiet floor at startup
and set the AGC noise gate just above it -- no per-machine hand tuning. Pure logic,
no audio device.
"""
import threading

import numpy as np

from core.audio_frontend import compute_input_calibration, InputAGC
from core.enroll import CaptureResolution
from core.engine import EngineCallbacks
from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine


def _block(rms, n=1600, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype("float32")
    x *= rms / float(np.sqrt(np.mean(x ** 2)))
    return x


def test_floor_tracks_quiet_level_with_headroom():
    # A quiet room ~0.01 RMS -> floor ~ headroom * 0.01, clamped into range.
    blocks = [_block(0.01, seed=i) for i in range(15)]
    cal = compute_input_calibration(blocks, headroom=3.0)
    assert cal["n_blocks"] == 15
    assert abs(cal["ambient_rms"] - 0.01) < 0.003
    assert 0.02 < cal["noise_floor_rms"] < 0.04          # ~3x ambient
    assert cal["clipping_fraction"] == 0.0


def test_low_percentile_is_robust_to_a_word_during_calibration():
    # Mostly quiet, but a few loud (speech) blocks slipped in -> the floor must
    # follow the QUIET level, not be dragged up by the loud blocks.
    blocks = [_block(0.008, seed=i) for i in range(12)] + [_block(0.3, seed=99 + i) for i in range(3)]
    cal = compute_input_calibration(blocks, headroom=3.0)
    assert cal["noise_floor_rms"] < 0.05                 # not pulled toward 0.3


def test_floor_is_clamped():
    # Near-silent -> clamped to min_floor; very loud floor -> clamped to max_floor.
    assert compute_input_calibration([_block(1e-5)], min_floor=0.004)["noise_floor_rms"] == 0.004
    loud = compute_input_calibration([_block(0.5, seed=i) for i in range(5)], max_floor=0.08)
    assert loud["noise_floor_rms"] == 0.08


def test_clipping_fraction_detected():
    railed = np.ones(1600, dtype="float32")              # fully railed
    cal = compute_input_calibration([railed])
    assert cal["clipping_fraction"] > 0.9
    assert cal["peak"] >= 1.0


def test_empty_input_is_safe():
    cal = compute_input_calibration([])
    assert cal["n_blocks"] == 0
    assert cal["noise_floor_rms"] == 0.004
    assert cal["clipping_fraction"] == 0.0


def test_calibration_feeds_the_agc_floor():
    # End-to-end: the measured floor is the value the engine assigns to InputAGC.
    blocks = [_block(0.02, seed=i) for i in range(10)]
    cal = compute_input_calibration(blocks)
    agc = InputAGC(noise_floor_rms=0.004)
    agc.noise_floor_rms = cal["noise_floor_rms"]
    assert agc.noise_floor_rms == cal["noise_floor_rms"]
    assert agc.noise_floor_rms > 0.004                   # adapted up from the default


def test_calibration_restarts_in_recovered_stream_domain(monkeypatch):
    """A transparent reopen cannot mix old-rate room tone into provenance."""

    class _ReopeningInput:
        generation = 1
        actual_samplerate = 16000

        def __init__(self):
            self.read_sizes = []

        def read(self, frames):
            self.read_sizes.append(frames)
            if len(self.read_sizes) == 1:
                # One block from the retired capture domain.
                return np.full(frames, 0.7, dtype="float32"), False
            if len(self.read_sizes) == 2:
                # Mirror _RecoveringInputStream: this read began with the old
                # frame count, then its internal retry returned a full 100 ms
                # block from the newly opened 48 kHz stream.
                self.generation = 2
                self.actual_samplerate = 48000
                return np.full(4800, 0.02, dtype="float32"), False
            return np.full(frames, 0.02, dtype="float32"), False

    class _Resampler:
        def __init__(self, src_sr, dst_sr, *, quality):
            self.src_sr = src_sr
            self.dst_sr = dst_sr

        def process(self, samples):
            # Deterministic 48k -> 16k stand-in; sufficient to prove the block
            # is processed only after the capture domain has been rebound.
            return np.asarray(samples, dtype="float32")[::3]

    seen = []

    def _compute(blocks):
        seen.extend(np.asarray(block).copy() for block in blocks)
        return compute_input_calibration(blocks)

    monkeypatch.setattr("core.engines.sherpa.AudioResampler", _Resampler)
    monkeypatch.setattr("core.engines.sherpa.compute_input_calibration", _compute)

    engine = SherpaOnnxEngine(
        SherpaConfig(
            block_sec=0.1,
            input_calibrate=True,
            input_calibrate_sec=0.3,
        )
    )
    stream = _ReopeningInput()
    engine._stream_in = stream
    engine._capture_sr = 16000
    engine._resampler = None

    engine._calibrate_input()

    # The second read was initiated in the old domain. Calibration then
    # restarted, and every subsequent request uses the recovered 48 kHz rate.
    assert stream.read_sizes == [1600, 1600, 4800, 4800]
    assert engine._capture_sr == 48000
    assert engine._resampler is not None
    assert engine._resampler.src_sr == 48000
    assert len(seen) == 3
    assert all(block.shape == (1600,) for block in seen)
    assert all(np.allclose(block, 0.02) for block in seen)
    assert engine._last_calibration["n_blocks"] == 3


def test_calibration_rebinds_when_recovered_retry_raises(monkeypatch):
    """A successful reopen is authoritative even if its first retry fails."""

    class _ReopenedThenFailedInput:
        generation = 1
        actual_samplerate = 16000

        def __init__(self):
            self.calls = 0

        def read(self, frames):
            self.calls += 1
            if self.calls == 1:
                return np.full(frames, 0.7, dtype="float32"), False
            self.generation = 2
            self.actual_samplerate = 48000
            raise RuntimeError("recovered stream's first read failed")

    computed = []
    monkeypatch.setattr(
        "core.engines.sherpa.compute_input_calibration",
        lambda blocks: computed.append(blocks),
    )

    engine = SherpaOnnxEngine(
        SherpaConfig(
            block_sec=0.1,
            input_calibrate=True,
            input_calibrate_sec=0.2,
        )
    )
    engine._stream_in = _ReopenedThenFailedInput()
    engine._capture_sr = 16000
    engine._resampler = None

    engine._calibrate_input()

    assert engine._capture_sr == 48000
    assert engine._resampler is not None
    assert engine._resampler.src_sr == 48000
    assert computed == []
    assert engine._last_calibration is None


def test_same_capture_domain_reopen_preserves_calibrated_agc_floor():
    class _Stream:
        actual_samplerate = 16000
        actual_device = "same-mic"

    engine = SherpaOnnxEngine(
        SherpaConfig(
            input_agc=True,
            input_calibrate=True,
            input_calibrate_sec=0.3,
        )
    )
    resolution = CaptureResolution(
        route="same-route",
        capture_sample_rate=16000,
        model_sample_rate=16000,
        resampler="identity",
        voice_comm="none",
        input_agc_noise_floor_rms=0.012,
    )
    calibration = {
        "ambient_rms": 0.004,
        "noise_floor_rms": 0.012,
        "peak": 0.02,
        "clipping_fraction": 0.0,
        "n_blocks": 3,
    }
    engine._stream_in = _Stream()
    engine._capture_sr = 16000
    engine._capture_resolution = resolution
    engine._last_calibration = calibration
    engine._input_agc.noise_floor_rms = 0.012
    restores = []

    def resolve(_sd, _selector, **kwargs):
        restore = kwargs.get("restore_authority", True)
        restores.append(restore)
        engine._capture_resolution = resolution
        engine._word_cut_route_verified = restore
        return True

    engine._resolve_capture_domain = resolve
    engine._reset_capture_frontends_after_reopen()

    assert restores == [False, True]
    assert engine._last_calibration is calibration
    assert engine._input_agc.noise_floor_rms == 0.012
    assert engine._recovery_calibration_target == 0


def test_changed_capture_domain_recalibrates_before_restoring_authority():
    class _Stream:
        actual_samplerate = 48000
        actual_device = "fallback-mic"

    engine = SherpaOnnxEngine(
        SherpaConfig(
            input_agc=True,
            input_calibrate=True,
            input_calibrate_sec=0.3,
            block_sec=0.1,
        )
    )
    old_resolution = CaptureResolution(
        route="old-route",
        capture_sample_rate=16000,
        model_sample_rate=16000,
        resampler="identity",
        voice_comm="none",
        input_agc_noise_floor_rms=0.012,
    )
    new_resolution = CaptureResolution(
        route="fallback-route",
        capture_sample_rate=48000,
        model_sample_rate=16000,
        resampler="soxr",
        voice_comm="none",
        input_agc_noise_floor_rms=0.004,
    )
    engine._stream_in = _Stream()
    engine._capture_sr = 16000
    engine._capture_resolution = old_resolution
    engine._last_calibration = {"noise_floor_rms": 0.012}
    engine._input_agc.noise_floor_rms = 0.012
    restores = []

    def resolve(_sd, _selector, **kwargs):
        restore = kwargs.get("restore_authority", True)
        restores.append(restore)
        engine._capture_resolution = new_resolution
        engine._word_cut_route_verified = restore
        return True

    engine._resolve_capture_domain = resolve
    engine._reset_capture_frontends_after_reopen()

    assert restores == [False]
    assert engine._last_calibration is None
    assert engine._input_agc.noise_floor_rms == 0.004
    assert engine._recovery_calibration_target == 3
    assert not engine._word_cut_route_verified

    for index in range(3):
        engine._observe_recovery_calibration(
            _block(0.006, seed=100 + index), vad_active=False
        )

    assert restores == [False, True]
    assert engine._recovery_calibration_target == 0
    assert engine._last_calibration["n_blocks"] == 3
    assert engine._input_agc.noise_floor_rms > 0.004
    assert engine._word_cut_route_verified


def test_capture_loop_recovery_calibration_samples_before_input_agc():
    raw = _block(0.006, seed=212)

    class _Input:
        generation = 1
        actual_device = "fallback-mic"

        def __init__(self, engine):
            self.engine = engine

        def read(self, _frames):
            self.engine._running.clear()
            return raw.copy(), False

    class _Vad:
        def accept_waveform(self, _samples):
            pass

        def is_speech_detected(self):
            return False

        def reset(self):
            pass

    class _Recognizer:
        class _Stream:
            def accept_waveform(self, _sample_rate, _samples):
                pass

        def create_stream(self):
            return self._Stream()

        def is_ready(self, _stream):
            return False

        def decode_stream(self, _stream):  # pragma: no cover - never ready
            pass

        def get_result(self, _stream):
            return ""

        def is_endpoint(self, _stream):
            return False

    engine = SherpaOnnxEngine(
        SherpaConfig(
            input_agc=True,
            input_calibrate=True,
            input_calibrate_sec=0.1,
        )
    )
    engine._stream_in = _Input(engine)
    engine._capture_sr = 16000
    engine._recognizer = _Recognizer()
    engine._vad = _Vad()
    engine._cb = EngineCallbacks()
    engine._recovery_calibration_target = 1
    engine._resolve_capture_domain = lambda *_args, **_kwargs: False
    gains = []
    process = engine._input_agc.process

    def tracked_process(samples):
        output = process(samples)
        gains.append(engine._input_agc.gain)
        return output

    engine._input_agc.process = tracked_process

    engine._running.set()
    thread = threading.Thread(target=engine._capture_loop)
    thread.start()
    thread.join(timeout=3.0)

    assert not thread.is_alive()
    expected = compute_input_calibration([raw])["noise_floor_rms"]
    assert gains and gains[0] > 1.0
    assert engine._last_calibration["noise_floor_rms"] == expected
