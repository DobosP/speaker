"""Pure noise / signal generators + an SNR mixer (numpy only, no models/audio).

Every generator is deterministic given a seeded ``numpy.random.Generator`` so a
run is reproducible and the math is unit-testable with no sherpa / sounddevice.

The single stress knob is **SNR in dB, defined relative to the mock user's voice
RMS** (:func:`mix_at_snr`): the noise is scaled so

    20 * log10( rms(signal) / rms(scaled_noise) ) == snr_db

i.e. a higher dB means the user's voice is louder *relative to* the noise.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def rms(samples: Sequence[float]) -> float:
    """Root-mean-square level of a mono float block (0.0 for empty / silent).

    Mirrors ``core.engines.speaker_gate.rms`` / ``report``-style RMS so the SNR
    here is defined in the same units the engine and grader use."""
    x = np.asarray(samples, dtype="float64").reshape(-1)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x * x)))


def white(n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` samples of zero-mean white (flat-spectrum) Gaussian noise.

    Normalized to unit RMS so the SNR mixer's scaling is well-conditioned
    regardless of length (the raw standard normal already has ~unit RMS, but we
    normalize so a short block isn't biased)."""
    n = max(0, int(n))
    if n == 0:
        return np.zeros(0, dtype="float32")
    x = rng.standard_normal(n).astype("float32")
    r = rms(x)
    if r > 0:
        x = (x / r).astype("float32")
    return x


def pink(n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` samples of pink (1/f) noise via FFT spectral shaping.

    Pink noise has a power spectral density proportional to 1/f, so the
    *amplitude* per frequency bin scales as 1/sqrt(f). We synthesize white
    Gaussian noise, take its real FFT, scale bin ``k`` by ``1/sqrt(k)`` (bin 0 /
    DC left at its 1/sqrt(1) weight to avoid a divide-by-zero and a runaway DC
    term), and invert. Normalized to unit RMS.

    Pink (vs white) puts more energy in the low band where speech formants live,
    so it is a meaner STT stressor at the same SNR -- which the sweep surfaces."""
    n = max(0, int(n))
    if n == 0:
        return np.zeros(0, dtype="float32")
    x = rng.standard_normal(n)
    spectrum = np.fft.rfft(x)
    k = np.arange(spectrum.size, dtype="float64")
    k[0] = 1.0  # leave DC at unit weight (no 1/0); it carries negligible energy
    scaling = 1.0 / np.sqrt(k)
    shaped = np.fft.irfft(spectrum * scaling, n=n)
    y = shaped.astype("float32")
    r = rms(y)
    if r > 0:
        y = (y / r).astype("float32")
    return y


def babble(
    target_len: int,
    sr: int,
    intruder_clips: Sequence[Sequence[float]],
    rng: np.random.Generator,
) -> np.ndarray:
    """Overlapping 'cocktail-party' babble: several intruder-TTS clips summed at
    staggered offsets and looped to ``target_len`` samples, normalized to unit
    RMS.

    This is *voiced* energy (real speech), not flat noise -- it both degrades STT
    (broadband, no denoiser to remove it) AND, as competing speech from non-
    enrolled speakers, is the kind of thing the speaker-ID gate is meant to
    reject when it produces a final. Unintelligible as any single speaker by
    construction (multiple talkers overlapped at random offsets)."""
    target_len = max(0, int(target_len))
    if target_len == 0 or not intruder_clips:
        return np.zeros(target_len, dtype="float32")
    out = np.zeros(target_len, dtype="float64")
    for clip in intruder_clips:
        c = np.asarray(clip, dtype="float64").reshape(-1)
        if c.size == 0:
            continue
        # Tile the clip to cover the target, then roll it by a random offset so
        # the talkers don't line up phase-wise (true overlap, not a chorus).
        reps = -(-target_len // c.size)  # ceil
        tiled = np.tile(c, reps)[:target_len]
        offset = int(rng.integers(0, max(1, c.size)))
        out += np.roll(tiled, offset)
    y = out.astype("float32")
    r = rms(y)
    if r > 0:
        y = (y / r).astype("float32")
    return y


def mix_at_snr(
    signal: Sequence[float], noise: Sequence[float], snr_db: float
) -> np.ndarray:
    """Return ``signal + scaled_noise`` where the noise is scaled to sit
    ``snr_db`` dB below the signal's RMS:

        20 * log10( rms(signal) / rms(scaled_noise) ) == snr_db

    The noise is looped/truncated to the signal length first. A silent signal or
    silent noise is returned without scaling (no meaningful SNR). The result is
    soft-clipped into [-1, 1] with ``tanh`` only if it would otherwise exceed the
    range, so the SNR is preserved for the common in-range case (the unit test
    asserts the measured SNR hits the target) while a pathological loud mix can't
    wrap."""
    s = np.asarray(signal, dtype="float64").reshape(-1)
    if s.size == 0:
        return s.astype("float32")
    nz = _fit_length(np.asarray(noise, dtype="float64").reshape(-1), s.size)
    s_rms = rms(s)
    n_rms = rms(nz)
    if s_rms == 0.0 or n_rms == 0.0:
        return s.astype("float32")
    # target noise RMS so that signal/noise == 10^(snr/20)
    target_n_rms = s_rms / (10.0 ** (snr_db / 20.0))
    scaled = nz * (target_n_rms / n_rms)
    mixed = s + scaled
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 1.0:
        # Only engage the soft limiter when we'd otherwise clip; this keeps the
        # measured SNR exact for the in-range case the tests pin.
        mixed = np.tanh(mixed)
    return mixed.astype("float32")


def measured_snr_db(signal: Sequence[float], noise: Sequence[float]) -> float:
    """The achieved SNR (dB) of a signal vs an (already-scaled) noise block --
    used by the unit test to confirm :func:`mix_at_snr` hit its target, and by
    the report to record the mix it actually fed."""
    s_rms = rms(signal)
    n_rms = rms(noise)
    if s_rms == 0.0 or n_rms == 0.0:
        return math.inf
    return 20.0 * math.log10(s_rms / n_rms)


def scaled_noise_for_snr(
    signal: Sequence[float], noise: Sequence[float], snr_db: float
) -> np.ndarray:
    """The noise block alone, scaled+length-fitted to the target SNR vs the
    signal (so callers can mix it with the signal AND measure it independently
    for the report). Shares the exact scaling of :func:`mix_at_snr`."""
    s = np.asarray(signal, dtype="float64").reshape(-1)
    nz = _fit_length(np.asarray(noise, dtype="float64").reshape(-1), s.size)
    s_rms = rms(s)
    n_rms = rms(nz)
    if s_rms == 0.0 or n_rms == 0.0:
        return nz.astype("float32")
    target_n_rms = s_rms / (10.0 ** (snr_db / 20.0))
    return (nz * (target_n_rms / n_rms)).astype("float32")


def _fit_length(x: np.ndarray, n: int) -> np.ndarray:
    """Loop or truncate ``x`` to exactly ``n`` samples (empty -> zeros)."""
    if n <= 0:
        return np.zeros(0, dtype=x.dtype)
    if x.size == 0:
        return np.zeros(n, dtype=x.dtype)
    if x.size >= n:
        return x[:n]
    reps = -(-n // x.size)  # ceil
    return np.tile(x, reps)[:n]


def spectral_slope(samples: Sequence[float], sr: int) -> float:
    """Least-squares slope of log10(power) vs log10(frequency) -- a sanity check
    on a noise color. White noise ~ 0 (flat); pink ~ -1 (power ∝ 1/f). Used by
    the unit test to confirm :func:`pink` is actually pink. Stdlib+numpy only."""
    x = np.asarray(samples, dtype="float64").reshape(-1)
    if x.size < 8:
        return 0.0
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / float(sr))
    # Drop DC and zero/near-zero power bins before taking logs.
    mask = (freqs > 0) & (spectrum > 0)
    if mask.sum() < 4:
        return 0.0
    lf = np.log10(freqs[mask])
    lp = np.log10(spectrum[mask])
    slope, _ = np.polyfit(lf, lp, 1)
    return float(slope)
