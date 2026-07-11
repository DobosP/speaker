"""Pure calibrated pre-gain speech-pattern evidence regressions."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from core.audio_frontend import compute_input_calibration
from core.engines._speech_evidence import (
    PreGainCaptureDomain,
    SpeechEvidenceDisposition,
    build_speech_evidence_profile,
)


SR = 16000
FRAME = 320
AMBIENT_RMS = 0.001
DOMAIN = PreGainCaptureDomain(
    route="test-mic",
    capture_sample_rate=SR,
    model_sample_rate=SR,
    resampler="identity",
    voice_comm="none",
)


def _normalize(values, target_rms: float):
    values = np.asarray(values, dtype="float32")
    level = float(np.sqrt(np.mean(values.astype("float64") ** 2)))
    return (values * (target_rms / level)).astype("float32")


def _ambient_frame(rms: float = AMBIENT_RMS):
    t = np.arange(FRAME, dtype="float64") / SR
    signal = (
        np.sin(2.0 * np.pi * 5100.0 * t)
        + 0.7 * np.sin(2.0 * np.pi * 6200.0 * t + 0.3)
        + 0.4 * np.sin(2.0 * np.pi * 7100.0 * t + 0.8)
    )
    return _normalize(signal, rms)


def _voice_frame(rms: float = 0.0022):
    t = np.arange(FRAME, dtype="float64") / SR
    envelope = np.sin(np.pi * (np.arange(FRAME) + 0.5) / FRAME) ** 0.2
    signal = envelope * (
        np.sin(2.0 * np.pi * 180.0 * t)
        + 0.55 * np.sin(2.0 * np.pi * 360.0 * t + 0.2)
        + 0.25 * np.sin(2.0 * np.pi * 540.0 * t + 0.5)
    )
    return _normalize(signal, rms)


def _calibration_pcm():
    return [np.tile(_ambient_frame(), 5), np.tile(_ambient_frame(), 5)]


def _dynamic_voice_pattern(rms: float = 0.0022):
    t = np.arange(FRAME * 5, dtype="float64") / SR
    frame_index = np.arange(FRAME * 5) // FRAME
    second = np.asarray((0.2, 0.75, 0.35, 0.8, 0.25))[frame_index]
    third = np.asarray((0.6, 0.2, 0.7, 0.25, 0.65))[frame_index]
    signal = (
        np.sin(2.0 * np.pi * 180.0 * t)
        + second * np.sin(2.0 * np.pi * 360.0 * t + 0.2)
        + third * np.sin(2.0 * np.pi * 540.0 * t + 0.5)
    )
    return np.concatenate(
        [
            _normalize(signal[index * FRAME : (index + 1) * FRAME], rms)
            for index in range(5)
        ]
    )


def _profile(**overrides):
    profile = build_speech_evidence_profile(
        {
            "ambient_rms": AMBIENT_RMS,
            "clipping_fraction": 0.0,
            "n_blocks": 2,
        },
        _calibration_pcm(),
        domain=DOMAIN,
        calibration_generation=7,
        sample_rate=SR,
        **overrides,
    )
    assert profile is not None
    return profile


def test_profile_is_relative_spectral_frozen_and_safety_minima_are_hard():
    profile = _profile(
        margin_db=0.0,
        min_qualified_sec=0.02,
        min_contiguous_sec=0.02,
    )

    assert profile.frame_samples == FRAME
    assert profile.margin_db == 6.0
    assert profile.required_qualified_frames == 4
    assert profile.required_consecutive_frames == 4
    assert profile.threshold_rms == pytest.approx(0.001995262, rel=1e-6)
    assert profile.spectral_distance_threshold >= 0.12
    with pytest.raises(FrozenInstanceError):
        profile.threshold_rms = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("calibration", "pcm"),
    [
        ({}, []),
        ({"ambient_rms": 0.0}, _calibration_pcm()),
        ({"ambient_rms": float("nan")}, _calibration_pcm()),
        ({"ambient_rms": float("inf")}, _calibration_pcm()),
        ({"ambient_rms": "invalid"}, _calibration_pcm()),
        (
            {"ambient_rms": AMBIENT_RMS, "clipping_fraction": 0.03},
            _calibration_pcm(),
        ),
        (
            {"ambient_rms": AMBIENT_RMS, "clipping_fraction": 0.0},
            [np.tile(_ambient_frame(), 5)],
        ),
        (
            {"ambient_rms": AMBIENT_RMS, "clipping_fraction": 0.0},
            [np.full(FRAME * 10, AMBIENT_RMS, dtype="float32")],
        ),
    ],
)
def test_invalid_clipped_or_short_calibration_abstains(calibration, pcm):
    assert build_speech_evidence_profile(
        calibration,
        pcm,
        domain=DOMAIN,
        calibration_generation=1,
        sample_rate=SR,
    ) is None


def test_one_frame_impulse_is_insufficient():
    profile = _profile()
    tracker = profile.accumulator(capture_generation=11)
    block = np.tile(_ambient_frame(), 5)
    block[FRAME // 2] = 0.5

    tracker.observe(block, epoch_open=True)
    snapshot = tracker.snapshot()

    assert snapshot.disposition is SpeechEvidenceDisposition.INSUFFICIENT
    assert snapshot.observed_frames == 5
    assert snapshot.qualified_frames <= 1
    assert snapshot.longest_qualified_run <= 1
    assert snapshot.admitted is False


def test_energetic_dc_with_degenerate_spectrum_is_insufficient():
    tracker = _profile().accumulator(capture_generation=11)

    tracker.observe(
        np.full(FRAME * 5, 0.003, dtype="float32"),
        epoch_open=True,
    )

    snapshot = tracker.snapshot()
    assert snapshot.energy_frames == 5
    assert snapshot.qualified_frames == 0
    assert snapshot.disposition is SpeechEvidenceDisposition.INSUFFICIENT


@pytest.mark.parametrize("frequency", [50.0, 60.0, 90.0, 300.0])
def test_stationary_one_capture_read_tone_needs_longer_fallback(frequency):
    tracker = _profile().accumulator(capture_generation=11)
    t = np.arange(FRAME * 5, dtype="float64") / SR
    tone = _normalize(np.sin(2.0 * np.pi * frequency * t), 0.003)

    tracker.observe(tone, epoch_open=True)

    snapshot = tracker.snapshot()
    assert snapshot.qualified_frames == 5
    assert snapshot.longest_qualified_run == 5
    assert snapshot.dynamic_frames == 0
    assert snapshot.steady_fallback_frames == 6
    assert snapshot.disposition is SpeechEvidenceDisposition.INSUFFICIENT


def test_one_capture_read_broadband_noise_is_insufficient():
    tracker = _profile().accumulator(capture_generation=11)
    noise = _normalize(
        np.random.default_rng(91).standard_normal(FRAME * 5),
        0.003,
    )

    tracker.observe(noise, epoch_open=True)

    snapshot = tracker.snapshot()
    assert snapshot.energy_frames == 5
    assert snapshot.qualified_frames == 0
    assert snapshot.disposition is SpeechEvidenceDisposition.INSUFFICIENT


def test_one_capture_read_low_pass_colored_noise_sweep_is_insufficient():
    profile = _profile()
    seeds = tuple(range(10)) + (20, 43, 97)
    for frame_count in (5, 6, 10):
        for coefficient in (0.9, 0.95, 0.98, 0.999):
            for seed in seeds:
                tracker = profile.accumulator(capture_generation=11)
                excitation = np.random.default_rng(seed).standard_normal(
                    FRAME * frame_count
                )
                colored = np.empty_like(excitation)
                colored[0] = excitation[0]
                for index in range(1, colored.size):
                    colored[index] = (
                        coefficient * colored[index - 1] + excitation[index]
                    )

                tracker.observe(_normalize(colored, 0.003), epoch_open=True)

                snapshot = tracker.snapshot()
                assert snapshot.disposition is (
                    SpeechEvidenceDisposition.INSUFFICIENT
                ), (frame_count, coefficient, seed, snapshot)


def test_periodic_and_dynamic_artifacts_cannot_compose_across_runs():
    profile = _profile()
    t = np.arange(FRAME * 3, dtype="float64") / SR
    tone = _normalize(
        np.sin(2.0 * np.pi * 300.0 * t)
        + 0.5 * np.sin(2.0 * np.pi * 600.0 * t),
        0.003,
    )
    excitation = np.random.default_rng(92).standard_normal(FRAME * 4)
    colored = np.empty_like(excitation)
    colored[0] = excitation[0]
    for index in range(1, colored.size):
        colored[index] = 0.9 * colored[index - 1] + excitation[index]
    tracker = profile.accumulator(capture_generation=12)

    tracker.observe(
        np.concatenate(
            [tone, _ambient_frame(), _normalize(colored, 0.003)]
        ),
        epoch_open=True,
    )

    snapshot = tracker.snapshot()
    assert snapshot.qualified_frames >= 4
    assert snapshot.longest_periodic_run < snapshot.steady_fallback_frames
    assert snapshot.longest_joint_run < snapshot.required_consecutive_frames
    assert snapshot.disposition is SpeechEvidenceDisposition.INSUFFICIENT


@pytest.mark.parametrize(
    "fundamental_hz",
    [
        50.0,
        60.0,
        70.0,
        80.0,
        90.0,
        100.0,
        110.0,
        120.0,
        140.0,
        160.0,
        180.0,
        220.0,
        300.0,
        400.0,
    ],
)
def test_steady_harmonic_voice_fundamental_grid_uses_fallback(fundamental_hz):
    profile = _profile()
    t = np.arange(FRAME * 6, dtype="float64") / SR
    voice = _normalize(
        np.sin(2.0 * np.pi * fundamental_hz * t)
        + 0.5 * np.sin(4.0 * np.pi * fundamental_hz * t + 0.2)
        + 0.25 * np.sin(6.0 * np.pi * fundamental_hz * t + 0.4),
        0.0022,
    )
    tracker = profile.accumulator(capture_generation=12)

    tracker.observe(voice, epoch_open=True)

    snapshot = tracker.snapshot()
    assert snapshot.observed_frames == 6
    assert snapshot.qualified_frames >= 4
    assert snapshot.longest_periodic_run >= 4
    assert snapshot.periodic_frames >= 1
    assert snapshot.disposition is SpeechEvidenceDisposition.SATISFIED


def test_steady_voiced_pattern_has_a_120ms_fallback():
    tracker = _profile().accumulator(capture_generation=11)

    tracker.observe(np.tile(_voice_frame(), 6), epoch_open=True)

    snapshot = tracker.snapshot()
    assert snapshot.longest_qualified_run == 6
    assert snapshot.dynamic_frames == 0
    assert snapshot.disposition is SpeechEvidenceDisposition.SATISFIED


def test_80ms_stationary_tone_plus_ambient_cannot_use_steady_fallback():
    tracker = _profile().accumulator(capture_generation=11)
    t = np.arange(FRAME * 4, dtype="float64") / SR
    tone = _normalize(np.sin(2.0 * np.pi * 300.0 * t), 0.003)

    tracker.observe(
        np.concatenate([tone, _ambient_frame(), _ambient_frame()]),
        epoch_open=True,
    )

    snapshot = tracker.snapshot()
    assert snapshot.observed_frames == 6
    assert snapshot.longest_qualified_run == 4
    assert snapshot.longest_periodic_run == 4
    assert snapshot.disposition is SpeechEvidenceDisposition.INSUFFICIENT


def test_ambient_then_80ms_stationary_tone_cannot_mint_fast_dynamics():
    tracker = _profile().accumulator(capture_generation=11)
    t = np.arange(FRAME * 4, dtype="float64") / SR
    tone = _normalize(np.sin(2.0 * np.pi * 300.0 * t), 0.003)

    tracker.observe(
        np.concatenate([_ambient_frame(), _ambient_frame(), tone]),
        epoch_open=True,
    )

    snapshot = tracker.snapshot()
    assert snapshot.qualified_frames <= 4
    assert snapshot.dynamic_frames == 0
    assert snapshot.longest_joint_run == 0
    assert snapshot.disposition is SpeechEvidenceDisposition.INSUFFICIENT


def test_scaled_low_band_ambient_cannot_seed_tone_run_dynamics():
    t = np.arange(FRAME, dtype="float64") / SR
    fan = _normalize(
        np.sin(2.0 * np.pi * 180.0 * t)
        + 0.7 * np.sin(2.0 * np.pi * 360.0 * t + 0.3)
        + 0.4 * np.sin(2.0 * np.pi * 540.0 * t + 0.8),
        AMBIENT_RMS,
    )
    profile = build_speech_evidence_profile(
        {"ambient_rms": AMBIENT_RMS, "clipping_fraction": 0.0},
        [np.tile(fan, 5), np.tile(fan, 5)],
        domain=DOMAIN,
        calibration_generation=9,
        sample_rate=SR,
    )
    assert profile is not None
    tone_t = np.arange(FRAME * 4, dtype="float64") / SR
    tone = _normalize(
        np.sin(2.0 * np.pi * 300.0 * tone_t)
        + 0.5 * np.sin(2.0 * np.pi * 600.0 * tone_t),
        0.003,
    )
    tracker = profile.accumulator(capture_generation=22)

    tracker.observe(
        np.concatenate([_normalize(fan, 0.003), tone]),
        epoch_open=True,
    )

    snapshot = tracker.snapshot()
    assert snapshot.dynamic_frames == 0
    assert snapshot.longest_joint_run == 0
    assert snapshot.disposition is SpeechEvidenceDisposition.INSUFFICIENT


def test_short_voiced_pattern_satisfies_total_and_contiguous_requirements():
    profile = _profile()
    tracker = profile.accumulator(capture_generation=12)
    speech = _dynamic_voice_pattern()

    for chunk in (speech[:123], speech[123:901], speech[901:]):
        tracker.observe(chunk, epoch_open=True)
    snapshot = tracker.snapshot()

    assert snapshot.disposition is SpeechEvidenceDisposition.SATISFIED
    assert snapshot.qualified_frames == 5
    assert snapshot.longest_joint_run >= 4
    assert snapshot.capture_generation == 12
    assert snapshot.calibration_generation == 7
    assert snapshot.domain == DOMAIN


def test_four_noncontiguous_voice_frames_are_insufficient():
    profile = _profile()
    tracker = profile.accumulator(capture_generation=1)
    quiet = _ambient_frame()
    voiced = _voice_frame(0.003)

    tracker.observe(
        np.concatenate([voiced, voiced, quiet, voiced, voiced]),
        epoch_open=True,
    )

    snapshot = tracker.snapshot()
    assert snapshot.qualified_frames == 4
    assert snapshot.longest_qualified_run == 2
    assert snapshot.disposition is SpeechEvidenceDisposition.INSUFFICIENT


def test_scaled_stationary_ambient_is_not_speech_pattern():
    profile = _profile()
    tracker = profile.accumulator(capture_generation=2)

    tracker.observe(
        np.tile(_ambient_frame(rms=0.003), 6),
        epoch_open=True,
    )

    snapshot = tracker.snapshot()
    assert snapshot.energy_frames == 6
    assert snapshot.qualified_frames == 0
    assert snapshot.disposition is SpeechEvidenceDisposition.INSUFFICIENT


def test_dynamic_quiet_voice_over_overlapping_low_band_fan_is_admitted():
    t = np.arange(FRAME, dtype="float64") / SR
    fan = _normalize(
        np.sin(2.0 * np.pi * 180.0 * t)
        + 0.7 * np.sin(2.0 * np.pi * 360.0 * t + 0.3)
        + 0.4 * np.sin(2.0 * np.pi * 540.0 * t + 0.8),
        AMBIENT_RMS,
    )
    profile = build_speech_evidence_profile(
        {"ambient_rms": AMBIENT_RMS, "clipping_fraction": 0.0},
        [np.tile(fan, 5), np.tile(fan, 5)],
        domain=DOMAIN,
        calibration_generation=8,
        sample_rate=SR,
    )
    assert profile is not None
    tracker = profile.accumulator(capture_generation=4)
    frames = []
    for index, shift in enumerate((8.0, -7.0, 12.0, -10.0, 6.0)):
        voice = _normalize(
            np.sin(2.0 * np.pi * (180.0 + shift) * t + index * 0.2)
            + 0.5 * np.sin(2.0 * np.pi * (360.0 + shift * 2.0) * t)
            + 0.3 * np.sin(2.0 * np.pi * (540.0 + shift * 3.0) * t),
            0.0022,
        )
        frames.append((fan + voice).astype("float32"))

    tracker.observe(np.concatenate(frames), epoch_open=True)

    snapshot = tracker.snapshot()
    assert snapshot.dynamic_frames >= 1
    assert snapshot.disposition is SpeechEvidenceDisposition.SATISFIED


def test_pre_epoch_pcm_and_nonfinite_frame_cannot_authorize():
    profile = _profile()
    tracker = profile.accumulator(capture_generation=3)
    tracker.observe(np.tile(_voice_frame(), 6), epoch_open=False)
    invalid = _voice_frame()
    invalid[0] = np.nan
    tracker.observe(invalid, epoch_open=True)

    snapshot = tracker.snapshot()
    assert snapshot.observed_frames == 1
    assert snapshot.qualified_frames == 0
    assert snapshot.disposition is SpeechEvidenceDisposition.INSUFFICIENT


def test_failure_discovery_replay_admits_short_controls_not_transients():
    fixture_root = (
        Path(__file__).resolve().parent
        / "fixture_audio"
        / "failure_discovery"
    )
    ambient = np.load(
        fixture_root / "background_white_noise_machine.npy"
    ).astype("float32")
    profile = build_speech_evidence_profile(
        compute_input_calibration([ambient]),
        [ambient],
        domain=DOMAIN,
        calibration_generation=9,
        sample_rate=SR,
    )
    assert profile is not None

    expected = {
        "user_short_yes_command": SpeechEvidenceDisposition.SATISFIED,
        "user_short_no_command": SpeechEvidenceDisposition.SATISFIED,
        "user_short_stop_command": SpeechEvidenceDisposition.SATISFIED,
        "user_laptop_fan_and_speech": SpeechEvidenceDisposition.SATISFIED,
        "background_mic_bump": SpeechEvidenceDisposition.INSUFFICIENT,
        "background_chair_scrape": SpeechEvidenceDisposition.INSUFFICIENT,
        "background_door_slam_tail": SpeechEvidenceDisposition.INSUFFICIENT,
        "background_low_bass_music": SpeechEvidenceDisposition.INSUFFICIENT,
        "background_alarm_beep_sequence": SpeechEvidenceDisposition.INSUFFICIENT,
        "background_white_noise_machine": SpeechEvidenceDisposition.INSUFFICIENT,
    }
    for name, disposition in expected.items():
        tracker = profile.accumulator(capture_generation=21)
        tracker.observe(
            np.load(fixture_root / f"{name}.npy").astype("float32"),
            epoch_open=True,
        )
        assert tracker.snapshot().disposition is disposition, name
