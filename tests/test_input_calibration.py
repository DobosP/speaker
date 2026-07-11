"""Startup ambient calibration (core.audio_frontend.compute_input_calibration).

The device-generic operating-point step: measure THIS mic's quiet floor at startup
and set the AGC noise gate just above it -- no per-machine hand tuning. Pure logic,
no audio device.
"""
import json
import threading
from pathlib import Path

import numpy as np
import pytest

from core.audio_frontend import compute_input_calibration, InputAGC
from core.enroll import CaptureResolution
from core.engine import EngineCallbacks
from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine


def _block(rms, n=1600, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype("float32")
    x *= rms / float(np.sqrt(np.mean(x ** 2)))
    return x


class _CalibrationInput:
    generation = 1
    actual_samplerate = 16000

    def __init__(self, blocks):
        self.blocks = [np.asarray(block, dtype="float32") for block in blocks]
        self.read_count = 0

    def read(self, frames):
        self.read_count += 1
        if not self.blocks:
            raise RuntimeError("synthetic calibration input exhausted")
        block = self.blocks.pop(0)
        assert block.shape == (frames,)
        return block.copy(), False


def test_device_adaptive_startup_calibration_is_shipped_on_but_library_safe():
    # Programmatic callers that construct the engine directly retain the
    # conservative no-capture default.  The normal application path explicitly
    # opts into device adaptation in the committed configuration.
    assert SherpaConfig().input_calibrate is False
    shipped = json.loads(
        (Path(__file__).resolve().parents[1] / "config.json").read_text(
            encoding="utf-8"
        )
    )["sherpa"]
    assert shipped["input_calibrate"] is True
    assert shipped["final_speech_evidence_enabled"] is True


def _impulsive_block(*, quiet_rms=0.0002, peak=0.818, seed=0):
    block = _block(quiet_rms, seed=seed)
    block[0] = peak
    return block


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

    agc.gain = 12.0
    below_calibrated = _block(0.03, seed=100)
    below_out = agc.process(below_calibrated)
    np.testing.assert_array_equal(below_out, below_calibrated)
    assert agc.gain == 12.0
    assert agc.last_above_floor is False
    assert agc.last_applied_gain == 1.0

    above_calibrated = _block(0.07, seed=101)
    above_out = agc.process(above_calibrated)
    assert agc.last_above_floor is True
    assert agc.last_applied_gain > 1.0
    assert np.sqrt(np.mean(above_out.astype("float64") ** 2)) == pytest.approx(
        0.12, abs=0.005
    )


def test_stable_stationary_window_keeps_single_pass_behavior(caplog):
    """A high absolute peak is harmless when its crest is stationary."""
    blocks = [_block(0.04, seed=i) for i in range(3)]
    expected = compute_input_calibration(blocks)
    stream = _CalibrationInput(blocks)
    metrics = []
    engine = SherpaOnnxEngine(
        SherpaConfig(
            block_sec=0.1,
            input_agc=True,
            input_calibrate=True,
            input_calibrate_sec=0.3,
        )
    )
    engine._stream_in = stream
    engine._capture_sr = 16000
    engine._cb = EngineCallbacks(on_metric=lambda name, **_kw: metrics.append(name))

    engine._calibrate_input()

    assert stream.read_count == 3
    assert engine._last_calibration == expected
    assert engine._input_agc.noise_floor_rms == expected["noise_floor_rms"]
    assert engine._speech_evidence_profile is not None
    assert engine._speech_evidence_profile.ambient_rms == pytest.approx(
        expected["ambient_rms"]
    )
    assert metrics == []
    assert "suspicious transient crest" not in caplog.text


def test_suspicious_startup_crest_retries_once_and_uses_fresh_window(caplog):
    first = [_impulsive_block(seed=i) for i in range(3)]
    replacement = [_block(0.0002, seed=20 + i) for i in range(3)]
    expected = compute_input_calibration(replacement)
    stream = _CalibrationInput(first + replacement)
    metrics = []
    engine = SherpaOnnxEngine(
        SherpaConfig(
            block_sec=0.1,
            input_agc=True,
            input_calibrate=True,
            input_calibrate_sec=0.3,
            input_agc_noise_floor_rms=0.012,
        )
    )
    engine._stream_in = stream
    engine._capture_sr = 16000
    engine._cb = EngineCallbacks(on_metric=lambda name, **_kw: metrics.append(name))

    engine._calibrate_input()

    assert stream.read_count == 6
    assert engine._last_calibration == expected
    assert engine._input_agc.noise_floor_rms == expected["noise_floor_rms"] == 0.004
    assert engine._speech_evidence_profile is not None
    assert engine._speech_evidence_profile.ambient_rms == pytest.approx(
        expected["ambient_rms"]
    )
    assert metrics == ["input_calibration_transient_retry"]
    assert "suspicious transient crest" in caplog.text
    assert "retaining configured" not in caplog.text


def test_suspicious_replacement_exhausts_once_and_retains_configured_floor(caplog):
    stream = _CalibrationInput(
        [_impulsive_block(seed=i) for i in range(6)]
        + [_block(0.0002, seed=50 + i) for i in range(6)]
    )
    metrics = []
    engine = SherpaOnnxEngine(
        SherpaConfig(
            block_sec=0.1,
            input_agc=True,
            input_calibrate=True,
            input_calibrate_sec=0.3,
            input_agc_noise_floor_rms=0.007,
        )
    )
    engine._stream_in = stream
    engine._capture_sr = 16000
    engine._last_calibration = {"noise_floor_rms": 0.03}
    engine._input_agc.noise_floor_rms = 0.03
    engine._cb = EngineCallbacks(on_metric=lambda name, **_kw: metrics.append(name))

    engine._calibrate_input()

    # Exactly one replacement window is attempted even though the existing
    # four-window read budget and additional synthetic blocks remain available.
    assert stream.read_count == 6
    assert engine._last_calibration is None
    assert engine._speech_evidence_profile is None
    assert engine._input_agc.noise_floor_rms == 0.007
    assert metrics == [
        "input_calibration_transient_retry",
        "input_calibration_unstable",
    ]
    assert "retaining configured noise_floor=0.0070" in caplog.text


def test_clipped_complete_calibration_retries_then_abstains():
    railed = np.ones(1600, dtype="float32")
    stream = _CalibrationInput([railed.copy() for _ in range(4)])
    metrics: list[str] = []
    engine = SherpaOnnxEngine(
        SherpaConfig(
            block_sec=0.1,
            input_calibrate=True,
            input_calibrate_sec=0.2,
        )
    )
    engine._stream_in = stream
    engine._capture_sr = 16000
    engine._cb = EngineCallbacks(
        on_metric=lambda name, **_kwargs: metrics.append(name)
    )

    engine._calibrate_input()

    assert stream.read_count == 4
    assert engine._last_calibration is None
    assert engine._speech_evidence_profile is None
    assert metrics == [
        "input_clipping",
        "input_calibration_contaminated_retry",
        "input_calibration_unstable",
    ]


def test_speech_contaminated_calibration_retries_with_quiet_replacement():
    class _CalibrationVad:
        def __init__(self):
            self.speech = False
            self.resets = 0

        def accept_waveform(self, samples):
            self.speech = float(np.sqrt(np.mean(samples * samples))) > 0.01

        def is_speech_detected(self):
            return self.speech

        def reset(self):
            self.resets += 1
            self.speech = False

    first = [_block(0.02, seed=i) for i in range(2)]
    replacement = [_block(0.001, seed=10 + i) for i in range(2)]
    stream = _CalibrationInput(first + replacement)
    metrics: list[str] = []
    engine = SherpaOnnxEngine(
        SherpaConfig(
            block_sec=0.1,
            input_calibrate=True,
            input_calibrate_sec=0.2,
        )
    )
    engine._stream_in = stream
    engine._capture_sr = 16000
    engine._vad = _CalibrationVad()
    engine._cb = EngineCallbacks(
        on_metric=lambda name, **_kwargs: metrics.append(name)
    )

    engine._calibrate_input()

    assert engine._last_calibration == compute_input_calibration(replacement)
    assert engine._speech_evidence_profile is not None
    assert metrics == ["input_calibration_contaminated_retry"]
    assert engine._vad.resets >= 4


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
                return _block(0.02, n=4800, seed=420), False
            return _block(0.02, n=frames, seed=420 + len(self.read_sizes)), False

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
    assert engine._pre_gain_resampler is not None
    assert engine._pre_gain_resampler.src_sr == 48000
    assert len(seen) == 3
    assert all(block.shape == (1600,) for block in seen)
    assert all(
        float(np.sqrt(np.mean(block.astype("float64") ** 2)))
        == pytest.approx(0.02, rel=0.1)
        for block in seen
    )
    assert engine._last_calibration["n_blocks"] == 3
    assert engine._speech_evidence_profile is not None


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
    assert engine._speech_evidence_profile is None


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
    engine._install_speech_evidence_profile(
        calibration,
        [_block(0.004, seed=40), _block(0.004, seed=41)],
    )
    old_evidence_profile = engine._speech_evidence_profile
    assert old_evidence_profile is not None
    engine._input_agc.noise_floor_rms = 0.012
    engine._input_agc.gain = 9.0
    engine._input_agc.process(_block(0.001, seed=44))
    assert engine._input_agc.last_input_rms > 0.0
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
    assert engine._speech_evidence_profile is old_evidence_profile
    assert engine._input_agc.noise_floor_rms == 0.012
    assert engine._input_agc.gain == 1.0
    assert engine._input_agc.last_input_rms == 0.0
    assert engine._input_agc.last_applied_gain == 1.0
    assert engine._input_agc.last_above_floor is False
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
    engine._last_calibration = {
        "ambient_rms": 0.004,
        "noise_floor_rms": 0.012,
    }
    engine._install_speech_evidence_profile(
        engine._last_calibration,
        [_block(0.004, seed=80), _block(0.004, seed=81)],
    )
    old_evidence_generation = (
        engine._speech_evidence_profile.calibration_generation
    )
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
    assert engine._speech_evidence_profile is None
    assert engine._input_agc.noise_floor_rms == 0.004
    assert engine._recovery_calibration_target == 3
    assert not engine._word_cut_route_verified

    engine._input_agc.gain = 8.0
    engine._input_agc.process(_block(0.001, seed=88))
    assert engine._input_agc.last_input_rms > 0.0
    for index in range(3):
        engine._observe_recovery_calibration(
            _block(0.006, seed=100 + index), speech_epoch_open=False
        )

    assert restores == [False, True]
    assert engine._recovery_calibration_target == 0
    assert engine._last_calibration["n_blocks"] == 3
    assert engine._speech_evidence_profile is not None
    assert (
        engine._speech_evidence_profile.calibration_generation
        > old_evidence_generation
    )
    assert engine._speech_evidence_profile.domain.route == "fallback-route"
    assert engine._input_agc.noise_floor_rms > 0.004
    assert engine._input_agc.gain == 1.0
    assert engine._input_agc.last_input_rms == 0.0
    assert engine._input_agc.last_applied_gain == 1.0
    assert engine._input_agc.last_above_floor is False
    assert engine._word_cut_route_verified


def test_changed_domain_recalibrates_speech_evidence_without_input_agc():
    class _Stream:
        actual_samplerate = 48000
        actual_device = "fallback-mic"

    engine = SherpaOnnxEngine(
        SherpaConfig(
            input_agc=False,
            input_calibrate=True,
            input_calibrate_sec=0.2,
            block_sec=0.1,
            final_speech_evidence_enabled=True,
        )
    )
    old_resolution = CaptureResolution(
        route="old-route",
        capture_sample_rate=16000,
        model_sample_rate=16000,
        resampler="identity",
        voice_comm="none",
    )
    new_resolution = CaptureResolution(
        route="fallback-route",
        capture_sample_rate=48000,
        model_sample_rate=16000,
        resampler="soxr",
        voice_comm="none",
    )
    engine._stream_in = _Stream()
    engine._capture_sr = 16000
    engine._capture_resolution = old_resolution
    engine._last_calibration = {
        "ambient_rms": 0.001,
        "noise_floor_rms": 0.004,
    }
    engine._install_speech_evidence_profile(
        engine._last_calibration,
        [_block(0.001, seed=280), _block(0.001, seed=281)],
    )

    def resolve(_sd, _selector, **kwargs):
        engine._capture_resolution = new_resolution
        engine._word_cut_route_verified = kwargs.get("restore_authority", True)
        return True

    engine._resolve_capture_domain = resolve
    engine._reset_capture_frontends_after_reopen()

    assert engine._input_agc is None
    assert engine._speech_evidence_profile is None
    assert engine._recovery_calibration_target == 2
    engine._observe_recovery_calibration(
        _block(0.2, seed=299), speech_epoch_open=True
    )
    assert engine._recovery_calibration_target == 2
    assert engine._speech_evidence_profile is None
    engine._observe_recovery_calibration(
        _block(0.001, seed=300), speech_epoch_open=False
    )
    assert engine._speech_evidence_profile is None
    engine._observe_recovery_calibration(
        _block(0.001, seed=301), speech_epoch_open=False
    )

    assert engine._recovery_calibration_target == 0
    assert engine._speech_evidence_profile is not None
    assert engine._speech_evidence_profile.domain.route == "fallback-route"
    assert engine._word_cut_route_verified


def test_unstable_recovery_calibration_keeps_speech_evidence_fail_open():
    class _Stream:
        actual_samplerate = 16000
        actual_device = "test-mic"

    engine = SherpaOnnxEngine(
        SherpaConfig(
            input_agc=False,
            input_calibrate=True,
            final_speech_evidence_enabled=True,
        )
    )
    engine._stream_in = _Stream()
    engine._capture_resolution = CaptureResolution(
        route="test-route",
        capture_sample_rate=16000,
        model_sample_rate=16000,
        resampler="identity",
        voice_comm="none",
    )
    engine._recovery_calibration_target = 1
    engine._resolve_capture_domain = lambda *_args, **_kwargs: True
    metrics: list[str] = []
    engine._cb = EngineCallbacks(
        on_metric=lambda name, **_kwargs: metrics.append(name)
    )

    engine._observe_recovery_calibration(
        _impulsive_block(seed=400),
        speech_epoch_open=False,
    )

    assert engine._recovery_calibration_target == 1
    assert engine._last_calibration is None
    assert engine._speech_evidence_profile is None
    assert "speech_evidence_calibration_retry" in metrics

    engine._observe_recovery_calibration(
        _impulsive_block(seed=401),
        speech_epoch_open=False,
    )

    assert engine._recovery_calibration_target == 0
    assert engine._last_calibration is None
    assert engine._speech_evidence_profile is None
    assert "speech_evidence_calibration_unstable" in metrics


def test_open_speech_epoch_discards_partial_recovery_window():
    engine = SherpaOnnxEngine(
        SherpaConfig(
            input_agc=False,
            input_calibrate=True,
            input_calibrate_sec=0.2,
            block_sec=0.1,
        )
    )
    engine._capture_resolution = CaptureResolution(
        route="new-route",
        capture_sample_rate=16000,
        model_sample_rate=16000,
        resampler="identity",
        voice_comm="none",
    )
    engine._recovery_calibration_target = 2
    engine._resolve_capture_domain = lambda *_args, **_kwargs: True

    engine._observe_recovery_calibration(
        _block(0.001, seed=500), speech_epoch_open=False
    )
    assert len(engine._recovery_calibration_blocks) == 1
    engine._observe_recovery_calibration(
        _block(0.02, seed=501), speech_epoch_open=True
    )
    assert engine._recovery_calibration_blocks == []

    engine._observe_recovery_calibration(
        _block(0.001, seed=502), speech_epoch_open=False
    )
    assert engine._recovery_calibration_target == 2
    assert engine._speech_evidence_profile is None
    engine._observe_recovery_calibration(
        _block(0.001, seed=503), speech_epoch_open=False
    )

    assert engine._recovery_calibration_target == 0
    assert engine._speech_evidence_profile is not None


def test_recovery_replays_candidate_window_through_vad_before_arming():
    class _SpeechVad:
        def __init__(self):
            self.speech = False

        def reset(self):
            self.speech = False

        def accept_waveform(self, samples):
            self.speech = self.speech or float(
                np.sqrt(np.mean(np.asarray(samples, dtype="float64") ** 2))
            ) > 0.01

        def is_speech_detected(self):
            return self.speech

    engine = SherpaOnnxEngine(
        SherpaConfig(input_agc=False, input_calibrate=True)
    )
    engine._vad = _SpeechVad()
    engine._recovery_calibration_target = 2
    engine._word_cut_route_verified = False
    resolves: list[bool] = []
    engine._resolve_capture_domain = (
        lambda *_args, **_kwargs: resolves.append(True) or True
    )

    for seed in (510, 511):
        engine._observe_recovery_calibration(
            _block(0.02, seed=seed), speech_epoch_open=False
        )

    assert engine._recovery_calibration_target == 2
    assert engine._recovery_calibration_retried is True
    assert engine._speech_evidence_profile is None
    assert resolves == []

    for seed in (512, 513):
        engine._observe_recovery_calibration(
            _block(0.02, seed=seed), speech_epoch_open=False
        )

    assert engine._recovery_calibration_target == 0
    assert engine._speech_evidence_profile is None
    assert engine._word_cut_route_verified is False
    assert resolves == []


def test_degenerate_recovery_profile_retries_without_restoring_authority():
    engine = SherpaOnnxEngine(
        SherpaConfig(input_agc=False, input_calibrate=True)
    )
    engine._recovery_calibration_target = 2
    engine._word_cut_route_verified = False
    resolves: list[bool] = []
    engine._resolve_capture_domain = (
        lambda *_args, **_kwargs: resolves.append(True) or True
    )
    dc = np.full(1600, 0.003, dtype="float32")

    for _ in range(2):
        engine._observe_recovery_calibration(
            dc, speech_epoch_open=False
        )

    assert engine._recovery_calibration_target == 2
    assert engine._recovery_calibration_retried is True
    assert engine._last_calibration is None
    assert engine._speech_evidence_profile is None
    assert resolves == []

    for _ in range(2):
        engine._observe_recovery_calibration(
            dc, speech_epoch_open=False
        )

    assert engine._recovery_calibration_target == 0
    assert engine._last_calibration is None
    assert engine._speech_evidence_profile is None
    assert engine._word_cut_route_verified is False
    assert resolves == []


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
            final_speech_evidence_enabled=False,
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
    # This test isolates the AGC ordering; evidence is disabled because its
    # profile correctly requires at least 200 ms of ambient PCM.
    assert engine._speech_evidence_profile is None
