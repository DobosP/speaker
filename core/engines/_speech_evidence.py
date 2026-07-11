"""Calibrated pre-gain acoustic evidence for ordinary ASR turns.

Post-front-end VAD/ASR can be fooled by application gain, denoising, or noise.
This module independently requires a short energetic, spectrally novel,
periodic pattern in the current VAD epoch, measured before gain in the startup-
calibration model band. It is a conservative acoustic heuristic, not identity.

It is not an identity or command classifier. A satisfied snapshot may admit an
ordinary turn, but cannot grant owner verification, barge authority, or action
permission.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Mapping, Optional, Sequence


_FRAME_SEC = 0.02
_MIN_MARGIN_DB = 6.0
_MIN_QUALIFIED_SEC = 0.08
_MIN_CONTIGUOUS_SEC = 0.08
_STEADY_FALLBACK_SEC = 0.12
_MIN_CALIBRATION_SEC = 0.20
_SPECTRAL_BANDS = 20
_SPECTRAL_DISTANCE_MARGIN = 0.05
_MIN_SPECTRAL_DISTANCE = 0.12
_MIN_DYNAMIC_DISTANCE = 0.05
_MAX_STABLE_AMBIENT_DISTANCE = 0.75
_MIN_DYNAMIC_ZERO_CROSSING_RATE = 0.015
_MAX_SPEECH_ZERO_CROSSING_RATE = 0.45
_PERIODICITY_WINDOW_FRAMES = 3
_MIN_PERIODICITY = 0.65
_MIN_PERIODIC_FRAMES_PER_RUN = 2


class SpeechEvidenceDisposition(str, Enum):
    """Typed result of one VAD epoch's independent acoustic check."""

    SATISFIED = "satisfied"
    INSUFFICIENT = "insufficient"
    UNAVAILABLE = "unavailable"
    BYPASSED = "bypassed"


@dataclass(frozen=True)
class PreGainCaptureDomain:
    """Physical/model transform that produced pre-gain evidence PCM."""

    route: str
    capture_sample_rate: int
    model_sample_rate: int
    resampler: str
    voice_comm: str


@dataclass(frozen=True)
class SpeechEvidenceProfile:
    """Immutable energy/spectrum operating point from stable ambient PCM."""

    domain: PreGainCaptureDomain
    calibration_generation: int
    sample_rate: int
    frame_samples: int
    ambient_rms: float
    threshold_rms: float
    ambient_spectrum: tuple[float, ...]
    spectral_distance_threshold: float
    required_qualified_frames: int
    required_consecutive_frames: int
    steady_fallback_frames: int
    margin_db: float

    def accumulator(self, *, capture_generation: int) -> "SpeechEvidenceAccumulator":
        return SpeechEvidenceAccumulator(
            self,
            capture_generation=int(capture_generation),
        )


@dataclass(frozen=True)
class SpeechEvidenceSnapshot:
    """Immutable admission evidence captured before an ASR segment resets."""

    disposition: SpeechEvidenceDisposition
    reason: str
    domain: Optional[PreGainCaptureDomain] = None
    calibration_generation: int = 0
    capture_generation: int = 0
    observed_frames: int = 0
    energy_frames: int = 0
    qualified_frames: int = 0
    longest_qualified_run: int = 0
    dynamic_frames: int = 0
    periodic_frames: int = 0
    longest_periodic_run: int = 0
    longest_joint_run: int = 0
    max_periodic_frames_in_run: int = 0
    required_qualified_frames: int = 0
    required_consecutive_frames: int = 0
    steady_fallback_frames: int = 0
    threshold_rms: float = 0.0
    spectral_distance_threshold: float = 0.0
    peak_rms: float = 0.0
    peak_spectral_distance: float = 0.0
    peak_periodicity: float = 0.0

    @property
    def admitted(self) -> bool:
        """Unavailable/bypassed evidence abstains; only insufficiency rejects."""
        return self.disposition is not SpeechEvidenceDisposition.INSUFFICIENT

    @classmethod
    def unavailable(cls, reason: str) -> "SpeechEvidenceSnapshot":
        return cls(SpeechEvidenceDisposition.UNAVAILABLE, str(reason))

    @classmethod
    def bypassed(cls, reason: str) -> "SpeechEvidenceSnapshot":
        return cls(SpeechEvidenceDisposition.BYPASSED, str(reason))


@lru_cache(maxsize=16)
def _spectral_geometry(sample_rate: int, frame_samples: int):
    """Return immutable FFT window + logarithmic band slices for one format."""
    import numpy as np

    window = np.hanning(frame_samples).astype("float64")
    frequencies = np.fft.rfftfreq(frame_samples, d=1.0 / float(sample_rate))
    nyquist = sample_rate / 2.0
    # A separate DC/very-low band plus log bands resolve vowel harmonics while
    # remaining robust to scaled copies of a stationary fan/noise spectrum.
    positive_edges = np.geomspace(50.0, max(51.0, nyquist), _SPECTRAL_BANDS)
    edges = np.concatenate(([0.0], positive_edges))
    slices = []
    for index in range(_SPECTRAL_BANDS):
        lo = edges[index]
        hi = edges[index + 1]
        bins = np.flatnonzero(
            (frequencies >= lo)
            & (
                frequencies <= hi
                if index == _SPECTRAL_BANDS - 1
                else frequencies < hi
            )
        )
        slices.append(tuple(int(item) for item in bins))
    return tuple(float(item) for item in window), tuple(slices)


def _normalized_spectrum(
    frame,
    sample_rate: int,
) -> Optional[tuple[float, ...]]:
    """Amplitude-invariant coarse spectrum, or ``None`` when degenerate.

    A numeric all-zero vector is not a neutral spectrum: its distance from a
    normalized ambient profile is 0.5 and would make energetic DC look novel.
    Keep invalidity typed so offset-only/zero/non-finite frames cannot qualify.
    """
    import numpy as np

    values = np.asarray(frame, dtype="float64").reshape(-1)
    if not values.size or not np.all(np.isfinite(values)):
        return None
    values = values - float(np.mean(values))
    ac_power = float(np.mean(values * values))
    if not math.isfinite(ac_power) or ac_power <= 1e-20:
        return None
    window_values, slices = _spectral_geometry(sample_rate, values.size)
    window = np.asarray(window_values, dtype="float64")
    power = np.abs(np.fft.rfft(values * window)) ** 2
    bands = np.asarray(
        [float(np.sum(power[list(items)])) if items else 0.0 for items in slices],
        dtype="float64",
    )
    total = float(np.sum(bands))
    if not math.isfinite(total) or total <= 1e-20:
        return None
    return tuple(float(item) for item in bands / total)


def _spectral_distance(left: Sequence[float], right: Sequence[float]) -> float:
    """Total-variation distance between normalized spectra (range 0..1)."""
    return 0.5 * sum(abs(float(a) - float(b)) for a, b in zip(left, right))


def _normalized_periodicity(samples, sample_rate: int) -> float:
    """YIN-style periodicity score over 50..400 Hz pitch lags.

    ``1 - min(cumulative-mean-normalized difference)`` rewards a repeated
    waveform, not merely the smooth high autocorrelation of low-pass/AR noise.
    The 60 ms window spans at least three periods at the 50 Hz lower bound.
    """
    import numpy as np

    values = np.asarray(samples, dtype="float64").reshape(-1)
    if values.size < 3 or not np.all(np.isfinite(values)):
        return 0.0
    values = values - float(np.mean(values))
    energy = float(np.dot(values, values))
    if not math.isfinite(energy) or energy <= 1e-20:
        return 0.0
    min_lag = max(1, int(math.ceil(float(sample_rate) / 400.0)))
    max_lag = min(
        values.size - 2,
        int(math.floor(float(sample_rate) / 50.0)),
    )
    if max_lag < min_lag:
        return 0.0
    correlation = np.correlate(values, values, mode="full")[values.size - 1 :]
    squared = values * values
    prefix = np.concatenate(([0.0], np.cumsum(squared)))
    lags = np.arange(1, max_lag + 1, dtype="int64")
    left_energy = prefix[values.size - lags]
    right_energy = prefix[values.size] - prefix[lags]
    difference = np.maximum(
        0.0,
        left_energy + right_energy - 2.0 * correlation[lags],
    )
    cumulative = np.cumsum(difference)
    valid = cumulative > 1e-20
    if not np.any(valid[min_lag - 1 :]):
        return 0.0
    cmnd = np.ones(lags.size, dtype="float64")
    cmnd[valid] = difference[valid] * lags[valid] / cumulative[valid]
    minimum = float(np.min(cmnd[min_lag - 1 :]))
    score = max(0.0, min(1.0, 1.0 - minimum))
    return score if math.isfinite(score) else 0.0


def build_speech_evidence_profile(
    calibration: Mapping[str, object],
    calibration_pcm: Sequence[object],
    *,
    domain: PreGainCaptureDomain,
    calibration_generation: int,
    sample_rate: int,
    margin_db: float = _MIN_MARGIN_DB,
    min_qualified_sec: float = _MIN_QUALIFIED_SEC,
    min_contiguous_sec: float = _MIN_CONTIGUOUS_SEC,
) -> Optional[SpeechEvidenceProfile]:
    """Build from caller-attested complete, stable, speech-free ambient PCM.

    Invalid, clipped, too-short, or spectrally unstable calibration returns
    ``None`` so the live seam fails open. Safety minima cannot be configured
    downward. No absolute RMS floor or AGC-clamped floor is used.
    """
    import numpy as np

    try:
        ambient_rms = float(calibration.get("ambient_rms", 0.0) or 0.0)
        clipping = float(calibration.get("clipping_fraction", 0.0) or 0.0)
        sr = int(sample_rate)
        margin = max(_MIN_MARGIN_DB, float(margin_db))
        qualified_sec = max(_MIN_QUALIFIED_SEC, float(min_qualified_sec))
        contiguous_sec = max(_MIN_CONTIGUOUS_SEC, float(min_contiguous_sec))
    except (TypeError, ValueError):
        return None
    if (
        sr <= 0
        or not math.isfinite(ambient_rms)
        or ambient_rms <= 0.0
        or not math.isfinite(clipping)
        or clipping > 0.02
        or not math.isfinite(margin)
        or not math.isfinite(qualified_sec)
        or not math.isfinite(contiguous_sec)
    ):
        return None
    blocks = [np.asarray(item, dtype="float32").reshape(-1) for item in calibration_pcm]
    blocks = [item for item in blocks if item.size]
    if not blocks:
        return None
    samples = np.concatenate(blocks)
    frame_samples = max(1, int(round(sr * _FRAME_SEC)))
    n_frames = int(samples.size // frame_samples)
    min_calibration_frames = int(math.ceil(_MIN_CALIBRATION_SEC / _FRAME_SEC))
    if n_frames < min_calibration_frames:
        return None
    frames = samples[: n_frames * frame_samples].reshape(n_frames, frame_samples)
    frame_spectra = [_normalized_spectrum(frame, sr) for frame in frames]
    if any(shape is None for shape in frame_spectra):
        return None
    spectra = np.asarray(frame_spectra, dtype="float64")
    center = np.median(spectra, axis=0)
    center_total = float(np.sum(center))
    if center_total > 1e-20:
        center = center / center_total
    else:
        center = np.zeros(_SPECTRAL_BANDS, dtype="float64")
    distances = np.asarray(
        [_spectral_distance(shape, center) for shape in spectra],
        dtype="float64",
    )
    ambient_distance = float(np.percentile(distances, 99.0))
    if (
        not math.isfinite(ambient_distance)
        or ambient_distance >= _MAX_STABLE_AMBIENT_DISTANCE
    ):
        return None
    spectral_threshold = max(
        _MIN_SPECTRAL_DISTANCE,
        ambient_distance + _SPECTRAL_DISTANCE_MARGIN,
    )
    threshold = ambient_rms * (10.0 ** (margin / 20.0))
    if not math.isfinite(threshold) or threshold <= 0.0:
        return None
    required_qualified = max(
        1,
        int(math.ceil(qualified_sec / _FRAME_SEC - 1e-12)),
    )
    required_contiguous = max(
        1,
        int(math.ceil(contiguous_sec / _FRAME_SEC - 1e-12)),
    )
    steady_fallback = max(
        required_qualified,
        required_contiguous,
        int(math.ceil(_STEADY_FALLBACK_SEC / _FRAME_SEC - 1e-12)),
    )
    return SpeechEvidenceProfile(
        domain=domain,
        calibration_generation=max(0, int(calibration_generation)),
        sample_rate=sr,
        frame_samples=frame_samples,
        ambient_rms=ambient_rms,
        threshold_rms=threshold,
        ambient_spectrum=tuple(float(item) for item in center),
        spectral_distance_threshold=spectral_threshold,
        required_qualified_frames=required_qualified,
        required_consecutive_frames=required_contiguous,
        steady_fallback_frames=steady_fallback,
        margin_db=margin,
    )


class SpeechEvidenceAccumulator:
    """Mutable, capture-thread-only tracker for one opened VAD epoch."""

    def __init__(
        self,
        profile: SpeechEvidenceProfile,
        *,
        capture_generation: int,
    ) -> None:
        import numpy as np

        self.profile = profile
        self.capture_generation = int(capture_generation)
        self.observed_frames = 0
        self.energy_frames = 0
        self.qualified_frames = 0
        self.longest_qualified_run = 0
        self.dynamic_frames = 0
        self.periodic_frames = 0
        self.longest_periodic_run = 0
        self.longest_joint_run = 0
        self.max_periodic_frames_in_run = 0
        self._current_qualified_run = 0
        self._current_run_has_dynamic = False
        self._current_run_periodic_frames = 0
        self.peak_rms = 0.0
        self.peak_spectral_distance = 0.0
        self.peak_periodicity = 0.0
        self._pending = np.zeros(0, dtype="float32")
        self._previous_qualified_spectrum: Optional[tuple[float, ...]] = None
        self._level_pcm = np.zeros(0, dtype="float32")
        self._periodicity_pcm = np.zeros(0, dtype="float32")

    def observe(self, samples, *, epoch_open: bool) -> None:
        """Consume pre-gain PCM after VAD has opened this speech epoch.

        VAD may flicker after onset; every following frame is assessed until the
        ASR endpoint. Its own energy/spectrum breaks or extends the run. The
        owning :class:`ASRSegment` excludes whole blocks before the first
        VAD-positive capture block; that onset block is evaluated in full.
        """
        import numpy as np

        if not epoch_open:
            self._pending = np.zeros(0, dtype="float32")
            self._current_qualified_run = 0
            self._current_run_has_dynamic = False
            self._current_run_periodic_frames = 0
            self._previous_qualified_spectrum = None
            self._level_pcm = np.zeros(0, dtype="float32")
            self._periodicity_pcm = np.zeros(0, dtype="float32")
            return
        block = np.asarray(samples, dtype="float32").reshape(-1)
        if not block.size:
            return
        if self._pending.size:
            block = np.concatenate((self._pending, block))
        frame_samples = self.profile.frame_samples
        n_frames = int(block.size // frame_samples)
        if n_frames <= 0:
            self._pending = block.copy()
            return
        framed_samples = n_frames * frame_samples
        frames = block[:framed_samples].reshape(n_frames, frame_samples)
        self._pending = block[framed_samples:].copy()
        profile = self.profile
        for frame in frames:
            values = frame.astype("float64")
            level = (
                float(np.sqrt(np.mean(values * values)))
                if np.all(np.isfinite(values))
                else float("nan")
            )
            centered = values - float(np.mean(values))
            spectrum = _normalized_spectrum(frame, profile.sample_rate)
            distance = (
                _spectral_distance(spectrum, profile.ambient_spectrum)
                if spectrum is not None
                else float("nan")
            )
            self._level_pcm = np.concatenate(
                (self._level_pcm, frame)
            )[-profile.frame_samples * _PERIODICITY_WINDOW_FRAMES :]
            zero_crossing_rate = (
                float(
                    np.mean(
                        np.signbit(centered[1:])
                        != np.signbit(centered[:-1])
                    )
                )
                if spectrum is not None and centered.size > 1
                else float("nan")
            )
            self.observed_frames += 1
            finite_level = math.isfinite(level)
            if finite_level:
                self.peak_rms = max(self.peak_rms, level)
            if math.isfinite(distance):
                self.peak_spectral_distance = max(
                    self.peak_spectral_distance,
                    distance,
                )
            gate_level = level
            # Up to 60 ms of trailing raw PCM makes the +6 dB decision robust to
            # 20 ms frame phase at low fundamentals. It grows 20→40→60 ms at
            # onset, so short commands do not wait for a separate warm-up.
            if self._level_pcm.size and np.all(np.isfinite(self._level_pcm)):
                gate_values = self._level_pcm.astype("float64")
                gate_level = float(
                    np.sqrt(np.mean(gate_values * gate_values))
                )
            energetic = (
                math.isfinite(gate_level)
                and gate_level > profile.threshold_rms
            )
            if energetic:
                self.energy_frames += 1
            speech_band = (
                spectrum is not None
                and math.isfinite(zero_crossing_rate)
                and zero_crossing_rate <= _MAX_SPEECH_ZERO_CROSSING_RATE
            )
            candidate = energetic and speech_band
            dynamic_distance = (
                _spectral_distance(
                    spectrum,
                    self._previous_qualified_spectrum,
                )
                if (
                    candidate
                    and spectrum is not None
                    and self._previous_qualified_spectrum is not None
                )
                else 0.0
            )
            dynamic = (
                candidate
                and math.isfinite(dynamic_distance)
                and dynamic_distance > _MIN_DYNAMIC_DISTANCE
                # Below ~120 Hz, a 20 ms FFT spans too few cycles: stationary
                # mains/hum phase leakage can look like frame-to-frame motion.
                # Such low-rate frames retain the 120 ms steady fallback but
                # cannot unlock the short dynamic path by themselves.
                and zero_crossing_rate >= _MIN_DYNAMIC_ZERO_CROSSING_RATE
            )
            if energetic and dynamic:
                self.dynamic_frames += 1
            novel = (
                math.isfinite(distance)
                and distance > profile.spectral_distance_threshold
            ) or dynamic
            qualified = (
                energetic
                and speech_band
                and novel
            )
            if qualified:
                # Dynamics may only carry inside one uninterrupted qualified
                # run.  An energetic speech-band frame that is not novel is a
                # run break too; retaining it here could let a later steady
                # artifact inherit motion from an unrelated precursor.
                self._previous_qualified_spectrum = spectrum
                self.qualified_frames += 1
                self._periodicity_pcm = np.concatenate(
                    (self._periodicity_pcm, frame)
                )[-profile.frame_samples * _PERIODICITY_WINDOW_FRAMES :]
                periodicity = (
                    _normalized_periodicity(
                        self._periodicity_pcm,
                        profile.sample_rate,
                    )
                    if self._periodicity_pcm.size
                    >= profile.frame_samples * _PERIODICITY_WINDOW_FRAMES
                    else 0.0
                )
                self.peak_periodicity = max(
                    self.peak_periodicity,
                    periodicity,
                )
                if periodicity > _MIN_PERIODICITY:
                    self.periodic_frames += 1
                    self._current_run_periodic_frames += 1
                    self.max_periodic_frames_in_run = max(
                        self.max_periodic_frames_in_run,
                        self._current_run_periodic_frames,
                    )
                self._current_qualified_run += 1
                self._current_run_has_dynamic = (
                    self._current_run_has_dynamic or dynamic
                )
                self.longest_qualified_run = max(
                    self.longest_qualified_run,
                    self._current_qualified_run,
                )
                if (
                    self._current_run_periodic_frames
                    >= _MIN_PERIODIC_FRAMES_PER_RUN
                ):
                    self.longest_periodic_run = max(
                        self.longest_periodic_run,
                        self._current_qualified_run,
                    )
                if (
                    self._current_run_has_dynamic
                    and self._current_run_periodic_frames
                    >= _MIN_PERIODIC_FRAMES_PER_RUN
                ):
                    self.longest_joint_run = max(
                        self.longest_joint_run,
                        self._current_qualified_run,
                    )
            else:
                self._previous_qualified_spectrum = None
                self._current_qualified_run = 0
                self._current_run_has_dynamic = False
                self._current_run_periodic_frames = 0
                self._periodicity_pcm = np.zeros(0, dtype="float32")

    def snapshot(self) -> SpeechEvidenceSnapshot:
        profile = self.profile
        fast_pattern = (
            self.qualified_frames >= profile.required_qualified_frames
            and self.longest_joint_run
            >= profile.required_consecutive_frames
            and self.max_periodic_frames_in_run
            >= _MIN_PERIODIC_FRAMES_PER_RUN
        )
        steady_pattern = (
            self.longest_periodic_run >= profile.steady_fallback_frames
            and self.max_periodic_frames_in_run
            >= _MIN_PERIODIC_FRAMES_PER_RUN
        )
        satisfied = fast_pattern or steady_pattern
        return SpeechEvidenceSnapshot(
            disposition=(
                SpeechEvidenceDisposition.SATISFIED
                if satisfied
                else SpeechEvidenceDisposition.INSUFFICIENT
            ),
            reason=(
                "calibrated_pre_gain_speech_pattern"
                if satisfied
                else "insufficient_pre_gain_speech_pattern"
            ),
            domain=profile.domain,
            calibration_generation=profile.calibration_generation,
            capture_generation=self.capture_generation,
            observed_frames=self.observed_frames,
            energy_frames=self.energy_frames,
            qualified_frames=self.qualified_frames,
            longest_qualified_run=self.longest_qualified_run,
            dynamic_frames=self.dynamic_frames,
            periodic_frames=self.periodic_frames,
            longest_periodic_run=self.longest_periodic_run,
            longest_joint_run=self.longest_joint_run,
            max_periodic_frames_in_run=self.max_periodic_frames_in_run,
            required_qualified_frames=profile.required_qualified_frames,
            required_consecutive_frames=profile.required_consecutive_frames,
            steady_fallback_frames=profile.steady_fallback_frames,
            threshold_rms=profile.threshold_rms,
            spectral_distance_threshold=profile.spectral_distance_threshold,
            peak_rms=self.peak_rms,
            peak_spectral_distance=self.peak_spectral_distance,
            peak_periodicity=self.peak_periodicity,
        )
