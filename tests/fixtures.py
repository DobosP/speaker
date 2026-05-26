"""
Audio fixtures for barge-in scenario tests.

Two layers of fixtures
----------------------
1. **Synthetic** — pure-NumPy, zero dependencies, zero network, always available.
   Useful for gating / noise / echo tests where exact spectral content does not
   matter.

2. **Real speech** — WAV files generated at session start (see ``conftest.py``)
   using edge-tts or gTTS.  These are acoustically realistic and make Silero
   VAD's voiced-detection path exercise-able in tests.  When fixtures are
   unavailable (offline CI) the ``real_speech()`` and ``real_tts_echo()``
   functions transparently fall back to synthetic.

Usage::

    from tests.fixtures import (
        voiced_speech, silence, tv_noise, tts_echo, mix,
        real_speech, real_tts_echo, REAL_SPEECH_AVAILABLE,
    )

    # Always works — synthetic
    quiet  = silence(0.5)
    noise  = tv_noise(2.0, amplitude=0.04)
    echo   = tts_echo(1.5)

    # Realistic (TTS-generated) when available, synthetic fallback otherwise
    speech = real_speech(0.8)             # voiced phrase from the "user"
    ref    = real_tts_echo(1.5)          # phrase from the "assistant" (for AEC ref)
"""

from __future__ import annotations

import os
import warnings

import numpy as np

SR = 16_000  # default sample rate used throughout tests

# ── Real-speech fixture paths ─────────────────────────────────────────────────

_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixture_audio")
_BARGE_IN_WAV = os.path.join(_FIXTURE_DIR, "barge_in.wav")
_TTS_ECHO_WAV = os.path.join(_FIXTURE_DIR, "tts_echo.wav")
_WAKEWORD_SPEECH_WAV = os.path.join(_FIXTURE_DIR, "wakeword_speech.wav")


def _wav_exists(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 1024


REAL_SPEECH_AVAILABLE: bool = (
    _wav_exists(_BARGE_IN_WAV)
    or _wav_exists(_TTS_ECHO_WAV)
    or _wav_exists(_WAKEWORD_SPEECH_WAV)
)


def _load_wav(path: str, target_sr: int = SR) -> np.ndarray:
    """Load a WAV file, resample to *target_sr* and return float32 mono."""
    try:
        import soundfile as sf  # noqa: PLC0415
        audio, sr = sf.read(path)
        audio = np.array(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            try:
                import librosa  # noqa: PLC0415
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            except ImportError:
                pass  # skip resampling if librosa unavailable
        return audio.astype(np.float32)
    except Exception as exc:
        warnings.warn(f"Could not load fixture WAV '{path}': {exc}")
        return np.array([], dtype=np.float32)


def _fit_to_duration(
    audio: np.ndarray,
    duration_sec: float,
    sr: int = SR,
    amplitude: float = 0.15,
) -> np.ndarray:
    """
    Crop or loop *audio* to exactly *duration_sec* seconds, then normalise
    to the requested *amplitude*.
    """
    n_target = int(duration_sec * sr)
    if len(audio) == 0:
        return np.zeros(n_target, dtype=np.float32)

    # Normalise first
    rms = float(np.sqrt(np.mean(audio ** 2))) or 1.0
    audio = (audio * amplitude / rms).astype(np.float32)

    if len(audio) >= n_target:
        return audio[:n_target]

    # Loop-pad if source is shorter than requested duration
    repeats = (n_target // len(audio)) + 1
    return np.tile(audio, repeats)[:n_target].astype(np.float32)


# ── Primitives ────────────────────────────────────────────────────────────────

def silence(duration_sec: float, sr: int = SR) -> np.ndarray:
    """Return a zero-filled array of the requested length."""
    return np.zeros(int(duration_sec * sr), dtype=np.float32)


def voiced_speech(
    duration_sec: float,
    sr: int = SR,
    pitch_hz: float = 150.0,
    amplitude: float = 0.12,
    syllable_rate: float = 4.0,
) -> np.ndarray:
    """
    Synthetic voiced speech: harmonic stack with a syllabic amplitude envelope.

    The signal has a fundamental at *pitch_hz* plus 4 harmonics, modulated by a
    slow (~4 Hz) syllabic envelope so energy rises and falls like natural speech.
    This produces a recognisably 'voiced' pattern for VAD and echo-similarity
    heuristics without needing a real speaker.

    Args:
        duration_sec: Length of the output signal in seconds.
        sr: Sample rate in Hz.
        pitch_hz: Fundamental frequency (typical male voice: ~120 Hz,
                  female: ~200 Hz).
        amplitude: Peak amplitude (0–1 range; 0.12 is moderately loud).
        syllable_rate: Rate of syllabic envelope modulation in Hz.
    """
    n = int(duration_sec * sr)
    t = np.linspace(0.0, duration_sec, n, dtype=np.float32)

    # Harmonic stack: fundamental + 4 overtones (decreasing amplitude)
    sig = np.zeros(n, dtype=np.float32)
    for k in range(1, 6):
        sig += np.sin(2.0 * np.pi * pitch_hz * k * t).astype(np.float32) / k

    # Syllabic amplitude envelope (rectified sine)
    envelope = (0.4 + 0.6 * np.abs(np.sin(np.pi * syllable_rate * t))).astype(np.float32)

    return (sig * envelope * amplitude).astype(np.float32)


def tv_noise(
    duration_sec: float,
    sr: int = SR,
    amplitude: float = 0.04,
    seed: int = 42,
) -> np.ndarray:
    """
    Flat-spectrum broadband noise that resembles TV background audio.

    The noise is stationary (Gaussian white noise) — VAD classifiers will
    correctly label it as *unvoiced* because it lacks periodic harmonic structure.

    Args:
        duration_sec: Length in seconds.
        sr: Sample rate in Hz.
        amplitude: RMS amplitude; 0.04 is clearly audible but below typical speech.
        seed: RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_sec * sr)
    noise = rng.standard_normal(n).astype(np.float32)
    # Normalise to target RMS
    rms = float(np.sqrt(np.mean(noise ** 2))) or 1.0
    return (noise * amplitude / rms).astype(np.float32)


def tts_echo(
    duration_sec: float,
    sr: int = SR,
    pitch_hz: float = 220.0,
    amplitude: float = 0.09,
) -> np.ndarray:
    """
    Simulated TTS audio leaking from speaker back into microphone.

    Uses a different pitch than :func:`voiced_speech` (220 Hz vs 150 Hz) to
    represent a synthetic voice.  When the same signal is also fed to the AEC
    as reference audio, the NLMS filter learns the transfer function and
    produces a very high echo-similarity score, causing barge-in to be blocked.

    For tests that rely only on the echo-similarity path of :class:`BargeInDetector`
    the exact content does not matter as long as the same buffer is used for
    both the AEC reference and the injected mic frames.
    """
    return voiced_speech(
        duration_sec,
        sr=sr,
        pitch_hz=pitch_hz,
        amplitude=amplitude,
        syllable_rate=3.0,
    )


def click_burst(
    duration_sec: float = 0.05,
    sr: int = SR,
    amplitude: float = 0.35,
    seed: int = 7,
) -> np.ndarray:
    """
    Very short, high-amplitude burst (< 50 ms) that should NOT trigger barge-in.

    Used to verify that the minimum-duration gate rejects transient noises.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_sec * sr)
    return (rng.standard_normal(n) * amplitude).astype(np.float32)


# ── Combination helpers ───────────────────────────────────────────────────────

def mix(
    primary: np.ndarray,
    secondary: np.ndarray,
    snr_db: float = 10.0,
) -> np.ndarray:
    """
    Mix *primary* and *secondary* signals at the requested SNR (dB).

    If the arrays have different lengths *secondary* is cropped or zero-padded
    to match *primary*.  Returns a float32 array the same length as *primary*.

    Args:
        primary: The target signal (e.g. voiced speech).
        secondary: The interfering signal (e.g. TV noise).
        snr_db: Signal-to-noise ratio in dB.  Higher values make the primary
                signal louder relative to secondary.
    """
    n = len(primary)
    if len(secondary) < n:
        secondary = np.pad(secondary, (0, n - len(secondary)))
    else:
        secondary = secondary[:n]

    p_rms = float(np.sqrt(np.mean(primary ** 2))) or 1e-9
    s_rms = float(np.sqrt(np.mean(secondary ** 2))) or 1e-9
    scale = p_rms / (s_rms * (10.0 ** (snr_db / 20.0)))
    return (primary + secondary * scale).astype(np.float32)


def apply_gain_db(audio: np.ndarray, gain_db: float) -> np.ndarray:
    """Apply deterministic gain in decibels."""
    return (audio.astype(np.float32) * (10.0 ** (gain_db / 20.0))).astype(np.float32)


def hard_clip(audio: np.ndarray, limit: float = 0.95) -> np.ndarray:
    """Simulate input clipping from an overdriven microphone path."""
    return np.clip(audio.astype(np.float32), -limit, limit).astype(np.float32)


def sample_rate_roundtrip(
    audio: np.ndarray,
    source_sr: int = SR,
    device_sr: int = 48_000,
) -> np.ndarray:
    """
    Simulate capture hardware running at a different sample rate.

    The output is back at ``source_sr`` so it can be fed into the 16 kHz test
    harness while still carrying interpolation artifacts.
    """
    audio = audio.astype(np.float32)
    if device_sr == source_sr or len(audio) == 0:
        return audio.copy()

    def _resample_linear(signal: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
        out_len = max(1, int(round(len(signal) * to_sr / from_sr)))
        x_old = np.linspace(0.0, 1.0, len(signal), endpoint=False)
        x_new = np.linspace(0.0, 1.0, out_len, endpoint=False)
        return np.interp(x_new, x_old, signal).astype(np.float32)

    device_audio = _resample_linear(audio, source_sr, device_sr)
    return _resample_linear(device_audio, device_sr, source_sr)


def near_far_mix(
    near_speech: np.ndarray,
    far_audio: np.ndarray,
    near_db: float = 0.0,
    far_db: float = -12.0,
) -> np.ndarray:
    """Mix near-field user speech with quieter far-field audio such as TV."""
    n = max(len(near_speech), len(far_audio))
    near = apply_gain_db(pad_to(near_speech, n), near_db)
    far = apply_gain_db(pad_to(far_audio, n), far_db)
    return (near + far).astype(np.float32)


def pad_to(audio: np.ndarray, n_samples: int) -> np.ndarray:
    """Zero-pad *audio* to exactly *n_samples* samples."""
    if len(audio) >= n_samples:
        return audio[:n_samples]
    return np.pad(audio, (0, n_samples - len(audio))).astype(np.float32)


def rms(audio: np.ndarray) -> float:
    """Return the RMS amplitude of *audio*."""
    return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))


# ── Real speech fixtures (TTS-generated, or synthetic fallback) ───────────────

def real_speech(
    duration_sec: float,
    sr: int = SR,
    amplitude: float = 0.15,
    fallback_amplitude: float = 0.35,
) -> np.ndarray:
    """
    Return realistic human speech audio.

    Loads ``tests/fixture_audio/barge_in.wav`` (generated by edge-tts or gTTS
    the first time the test suite runs — see ``conftest.py``).  If the file is
    not available, falls back to the synthetic ``voiced_speech()`` generator.

    The fallback uses a higher amplitude (``fallback_amplitude``) to ensure
    the energy path fires even without Silero's voiced=True signal.

    Args:
        duration_sec: Target length of the returned array in seconds.
        sr: Sample rate (default 16 000 Hz).
        amplitude: Normalised peak amplitude for real speech (0–1).
        fallback_amplitude: Amplitude used when falling back to synthetic.
    """
    global REAL_SPEECH_AVAILABLE

    for wav_path in (_BARGE_IN_WAV, _WAKEWORD_SPEECH_WAV):
        if _wav_exists(wav_path):
            audio = _load_wav(wav_path, target_sr=sr)
            if len(audio) > 0:
                REAL_SPEECH_AVAILABLE = True
                return _fit_to_duration(audio, duration_sec, sr=sr, amplitude=amplitude)

    # Fallback: synthetic voiced signal
    return voiced_speech(
        duration_sec, sr=sr, pitch_hz=150, amplitude=fallback_amplitude
    )


def real_tts_echo(
    duration_sec: float,
    sr: int = SR,
    amplitude: float = 0.10,
    fallback_amplitude: float = 0.09,
) -> np.ndarray:
    """
    Return realistic TTS-generated speech that represents assistant output.

    Used as the AEC echo reference — inject the same array as both the AEC
    reference *and* the mic input to simulate the assistant's voice leaking
    back into the microphone.

    Falls back to ``tts_echo()`` (synthetic) when the fixture WAV is missing.
    """
    global REAL_SPEECH_AVAILABLE

    if _wav_exists(_TTS_ECHO_WAV):
        audio = _load_wav(_TTS_ECHO_WAV, target_sr=sr)
        if len(audio) > 0:
            REAL_SPEECH_AVAILABLE = True
            return _fit_to_duration(audio, duration_sec, sr=sr, amplitude=amplitude)

    return tts_echo(duration_sec, sr=sr, amplitude=fallback_amplitude)


def human_voice(
    duration_sec: float,
    sr: int = SR,
    amplitude: float = 0.08,
    speaker: str = "jackson",
    digit: int = 1,
) -> np.ndarray:
    """
    Load a real human voice recording from the FSDD dataset.

    The Free Spoken Digit Dataset (FSDD) contains short recordings of human
    speakers saying digits 0–9.  Each clip is ~0.25–0.55 s at 8 kHz, which
    we resample to 16 kHz.

    The default amplitude (0.08) is deliberately chosen so that the
    **energy-only path cannot fire** (RMS ≈ 0.04–0.08, which is < threshold
    * 3.0 = 0.03).  Only Silero's voiced classifier can push the
    ``BargeInDetector`` score to ≥ 2.0.  This makes human_voice() the right
    input for tests that must verify the Silero path specifically.

    Args:
        duration_sec: Target length.  If the clip is shorter, it is looped.
        sr:           Target sample rate (default 16 000 Hz).
        amplitude:    Normalised peak RMS (default 0.08 — below energy path).
        speaker:      FSDD speaker name ("jackson", "nicolas", "george").
        digit:        Digit to load (1, 2, or 3).

    Returns:
        float32 array of length ``int(duration_sec * sr)``; falls back to
        ``voiced_speech()`` if the FSDD file is not available.
    """
    _voice_dir = os.path.join(os.path.dirname(__file__), "voice_samples")
    fname = f"{digit}_{speaker}_0.wav"
    path = os.path.join(_voice_dir, fname)

    if os.path.exists(path) and os.path.getsize(path) > 200:
        audio = _load_wav(path, target_sr=sr)
        if len(audio) > 0:
            return _fit_to_duration(audio, duration_sec, sr=sr, amplitude=amplitude)

    warnings.warn(
        f"FSDD sample '{fname}' not found — falling back to synthetic. "
        "Run tests with network access to download voice samples."
    )
    return voiced_speech(duration_sec, sr=sr, amplitude=amplitude)


def human_voice_concat(
    duration_sec: float,
    sr: int = SR,
    amplitude: float = 0.08,
) -> np.ndarray:
    """
    Load the pre-built concatenation of FSDD clips (6 clips, multiple speakers).

    This gives a longer sample (>1.0 s) with natural inter-clip pauses,
    useful for wakeword-callback tests that require the captured buffer to
    exceed the 0.5 s minimum in ``_finish_recording``.

    Falls back to ``voiced_speech()`` if the concat file is unavailable.
    """
    _voice_dir = os.path.join(os.path.dirname(__file__), "voice_samples")
    path = os.path.join(_voice_dir, "human_voice_concat.wav")

    if os.path.exists(path) and os.path.getsize(path) > 4096:
        audio = _load_wav(path, target_sr=sr)
        if len(audio) > 0:
            return _fit_to_duration(audio, duration_sec, sr=sr, amplitude=amplitude)

    warnings.warn("human_voice_concat.wav not found — falling back to voiced_speech")
    return voiced_speech(duration_sec, sr=sr, amplitude=amplitude)


# ── Acoustic realism helpers ──────────────────────────────────────────────────

def apply_room_delay(
    signal: np.ndarray,
    delay_ms: float = 35.0,
    attenuation_db: float = -6.0,
    sr: int = SR,
) -> np.ndarray:
    """
    Apply a single early reflection: signal arrives at the mic *delay_ms* later,
    attenuated by *attenuation_db* dB.

    This simulates the most important real-world echo problem: when TTS audio
    travels from the speaker to the microphone across a room, it arrives with a
    delay proportional to distance.  EchoGuard's zero-lag cosine similarity
    computation fails to recognise the delayed signal as echo → self-barge-in.

    Args:
        signal:         Source audio (e.g. TTS reference).
        delay_ms:       Propagation delay in milliseconds (1 m ≈ 3 ms at 340 m/s;
                        a 2-metre speaker-to-mic distance ≈ 6 ms; room reflections
                        add 20–80 ms on top).
        attenuation_db: Energy loss in dB (negative value).
        sr:             Sample rate.

    Returns:
        float32 array of the same length as *signal*, containing only the
        delayed/attenuated reflection (direct path is NOT included — add it
        yourself if you want a mixed signal).
    """
    delay_samples = int(delay_ms * sr / 1000.0)
    n = len(signal)
    gain = 10.0 ** (attenuation_db / 20.0)
    out = np.zeros(n, dtype=np.float32)
    if delay_samples < n:
        out[delay_samples:] = signal[: n - delay_samples] * gain
    return out


def reverberant_echo(
    tts_signal: np.ndarray,
    direct_delay_ms: float = 5.0,
    rt60_ms: float = 250.0,
    direct_gain: float = 0.75,
    sr: int = SR,
    seed: int = 77,
) -> np.ndarray:
    """
    Simulate how TTS audio sounds at the microphone after bouncing around a room.

    Models three components:
    • Direct path: arrives ~5 ms late with mild attenuation.
    • First reflection (wall ~6 m away): +35 ms, −6 dB.
    • Second reflection (opposite wall): +70 ms, −12 dB.
    • Late reverb: exponential noise tail starting at 80 ms, decaying to −60 dB
      by *rt60_ms*.

    Args:
        tts_signal:     The clean TTS audio (what ``set_echo_reference`` sees).
        direct_delay_ms: Propagation delay of the direct path in ms.
        rt60_ms:        Room reverberation time (time for energy to fall 60 dB).
        direct_gain:    Amplitude of the direct path (< 1 due to distance loss).
        sr:             Sample rate.
        seed:           RNG seed for the reverb diffuse noise.

    Returns:
        float32 array: what the microphone actually picks up when only the room
        echo is present (no user speech).
    """
    rng = np.random.default_rng(seed)
    n = len(tts_signal)
    mic = np.zeros(n, dtype=np.float32)

    def _add_reflection(delay_ms: float, gain: float) -> None:
        d = int(delay_ms * sr / 1000.0)
        if d < n:
            mic[d:] += tts_signal[: n - d] * gain

    _add_reflection(direct_delay_ms, direct_gain)
    _add_reflection(35.0, 0.50)
    _add_reflection(70.0, 0.25)
    _add_reflection(110.0, 0.12)

    # Diffuse reverb tail
    tail_start = int(0.080 * sr)
    if tail_start < n:
        length = n - tail_start
        t = np.arange(length, dtype=np.float32) / sr
        decay = np.exp(-6.91 / (rt60_ms / 1000.0) * t)
        reverb = rng.standard_normal(length).astype(np.float32) * direct_gain * 0.10 * decay
        mic[tail_start:] += reverb

    return mic.astype(np.float32)


def babble_noise(
    duration_sec: float,
    amplitude: float = 0.05,
    num_speakers: int = 3,
    sr: int = SR,
) -> np.ndarray:
    """
    Mix multiple FSDD speakers to produce realistic multi-talker background noise.

    Unlike white noise, babble has voiced characteristics (periodic harmonics,
    formant structure) that can fool Silero VAD into returning voiced=True.
    This is what makes babble a harder test than stationary white noise: the
    system must rely on calibrated noise floor (not just Silero) to reject it.

    Falls back to pink-noise-coloured white noise if FSDD files are not present.

    Args:
        duration_sec:  Length of the output signal.
        amplitude:     RMS amplitude of the mixed result.
        num_speakers:  How many FSDD voices to mix (2–5).
        sr:            Sample rate.
    """
    _voice_dir = os.path.join(os.path.dirname(__file__), "voice_samples")
    candidates = [
        ("george", 1), ("jackson", 2), ("nicolas", 3),
        ("theo", 1), ("george", 2), ("nicolas", 1),
    ]
    mixed = np.zeros(int(duration_sec * sr), dtype=np.float32)
    loaded = 0
    for speaker, digit in candidates[:num_speakers + 1]:
        fname = f"{digit}_{speaker}_0.wav"
        path = os.path.join(_voice_dir, fname)
        if os.path.exists(path) and os.path.getsize(path) > 200:
            clip = _load_wav(path, target_sr=sr)
            if len(clip) > 0:
                loop = _fit_to_duration(
                    clip, duration_sec, sr=sr, amplitude=amplitude / max(num_speakers, 1)
                )
                # Stagger each speaker slightly so they don't overlap perfectly
                offset = int(loaded * sr * 0.11) % int(duration_sec * sr)
                roll = np.roll(loop, offset)
                mixed += roll
                loaded += 1
                if loaded >= num_speakers:
                    break

    if loaded == 0:
        # Fallback: pink-ish noise (1/f shaped white noise)
        rng = np.random.default_rng(123)
        n = int(duration_sec * sr)
        white = rng.standard_normal(n).astype(np.float32)
        # Simple 1/f approximation via cumulative sum of noise
        pink = np.cumsum(rng.standard_normal(n).astype(np.float32))
        pink -= pink.mean()
        mixed = (0.5 * white + 0.5 * pink).astype(np.float32)

    if len(mixed) > 0:
        rms_val = float(np.sqrt(np.mean(mixed ** 2))) or 1e-9
        mixed = (mixed * amplitude / rms_val).astype(np.float32)
    return mixed


def nonstationary_noise(
    duration_sec: float,
    quiet_amplitude: float = 0.01,
    loud_amplitude: float = 0.08,
    transition_at: float = 0.5,
    sr: int = SR,
    seed: int = 55,
) -> np.ndarray:
    """
    Noise that suddenly increases mid-stream (e.g. HVAC kicks in, TV turns on).

    The system calibrates its noise floor during the *quiet* portion.  When the
    level jumps to *loud_amplitude*, the stale calibration is no longer valid
    and the system starts making incorrect gate decisions.

    Args:
        duration_sec:    Total signal length.
        quiet_amplitude: RMS during the first ``transition_at`` fraction.
        loud_amplitude:  RMS after the transition.
        transition_at:   Fraction of the signal where the level jump occurs.
        sr:              Sample rate.
        seed:            RNG seed.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_sec * sr)
    white = rng.standard_normal(n).astype(np.float32)
    env = np.full(n, quiet_amplitude, dtype=np.float32)
    split = int(transition_at * n)
    env[split:] = loud_amplitude
    ramp = int(0.05 * sr)  # 50 ms smooth ramp
    if split + ramp < n and split - ramp >= 0:
        env[split - ramp : split + ramp] = np.linspace(
            quiet_amplitude, loud_amplitude, 2 * ramp, dtype=np.float32
        )
    white_rms = float(np.sqrt(np.mean(white ** 2))) or 1e-9
    return (white * env / white_rms).astype(np.float32)


def music_noise(
    duration_sec: float,
    amplitude: float = 0.05,
    beat_hz: float = 2.0,
    sr: int = SR,
    seed: int = 88,
) -> np.ndarray:
    """
    Simulate background music: broadband noise with a rhythmic energy envelope.

    Music has two properties that make it harder to reject than white noise:
    1. Rhythmic energy bursts at *beat_hz* can accumulate barge-in score
       during loud beats.
    2. Harmonic content in music can push Silero toward voiced=True.

    Args:
        duration_sec: Length of signal.
        amplitude:    Peak RMS.
        beat_hz:      Tempo in beats per second (2 Hz ≈ 120 bpm).
        sr:           Sample rate.
        seed:         RNG seed.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_sec * sr)
    t = np.arange(n, dtype=np.float32) / sr

    # Broadband noise base
    noise = rng.standard_normal(n).astype(np.float32)

    # Rhythmic amplitude envelope (rectified cosine at beat frequency)
    beat_env = (0.3 + 0.7 * np.abs(np.cos(np.pi * beat_hz * t))).astype(np.float32)

    # Add a weak harmonic "melody" line at 440 Hz to push Silero toward voiced
    melody = (
        0.3 * np.sin(2.0 * np.pi * 440.0 * t)
        + 0.15 * np.sin(2.0 * np.pi * 880.0 * t)
        + 0.10 * np.sin(2.0 * np.pi * 660.0 * t)
    ).astype(np.float32)

    mixed = (noise * beat_env + melody * beat_env).astype(np.float32)
    rms_val = float(np.sqrt(np.mean(mixed ** 2))) or 1e-9
    return (mixed * amplitude / rms_val).astype(np.float32)


def clipped_speech(
    duration_sec: float,
    sr: int = SR,
    clip_level: float = 0.5,
    amplitude: float = 0.25,
    speaker: str = "jackson",
    digit: int = 1,
) -> np.ndarray:
    """
    Real human speech with microphone clipping/saturation.

    Clipping occurs when the user speaks too close to the mic, or the mic gain
    is set too high.  The clipped waveform has heavy harmonic distortion, which
    can confuse both Silero VAD (more harmonics → sometimes more "voiced") and
    the echo similarity metric (distorted signal looks different from the clean
    TTS reference).

    The clip is applied AFTER normalising to *clip_level* (so everything above
    *clip_level* is hard-limited), then the result is scaled to *amplitude*.

    Args:
        clip_level: Hard clip threshold before normalisation (0.0–1.0).
        amplitude:  Final RMS target amplitude.
    """
    audio = human_voice(duration_sec, sr=sr, amplitude=clip_level * 2.0,
                         speaker=speaker, digit=digit)
    clipped = np.clip(audio, -clip_level, clip_level).astype(np.float32)
    rms_val = float(np.sqrt(np.mean(clipped ** 2))) or 1e-9
    return (clipped * amplitude / rms_val).astype(np.float32)


def plosive_burst(
    count: int = 5,
    burst_ms: float = 25.0,
    gap_ms: float = 120.0,
    amplitude: float = 0.30,
    sr: int = SR,
    seed: int = 33,
) -> np.ndarray:
    """
    A sequence of plosive-like noise bursts (simulating P/B/T consonants or
    breath pops against the microphone).

    Each burst is a very short broadband noise event at high amplitude, separated
    by near-silence gaps.  With the soft-decay fix in the noise gate, repeated
    bursts that pass the gate can accumulate barge-in score; this fixture tests
    whether that accumulation eventually reaches the fire threshold.

    Args:
        count:     Number of plosive bursts.
        burst_ms:  Duration of each burst in milliseconds.
        gap_ms:    Silence between bursts in milliseconds.
        amplitude: Peak amplitude of each burst.
        sr:        Sample rate.
        seed:      RNG seed.
    """
    rng = np.random.default_rng(seed)
    burst_n = int(burst_ms * sr / 1000.0)
    gap_n = int(gap_ms * sr / 1000.0)
    total = count * (burst_n + gap_n)
    out = np.zeros(total, dtype=np.float32)
    for i in range(count):
        start = i * (burst_n + gap_n)
        burst = rng.standard_normal(burst_n).astype(np.float32)
        burst *= amplitude / (float(np.sqrt(np.mean(burst ** 2))) + 1e-9)
        out[start : start + burst_n] = burst
    return out


def snr_mix(
    speech: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """
    Mix *speech* with *noise* at a precise signal-to-noise ratio.

    Measures the actual RMS of each signal and scales noise so the output
    achieves exactly *snr_db* dB.  Positive SNR = speech is louder.
    """
    n = max(len(speech), len(noise))
    s = np.zeros(n, dtype=np.float32)
    s[: len(speech)] = speech
    nz = np.zeros(n, dtype=np.float32)
    nz[: len(noise)] = noise
    speech_rms = float(np.sqrt(np.mean(s ** 2))) or 1e-9
    noise_rms = float(np.sqrt(np.mean(nz ** 2))) or 1e-9
    desired_noise_rms = speech_rms / (10.0 ** (snr_db / 20.0))
    scale = desired_noise_rms / noise_rms
    return (s + nz * scale).astype(np.float32)


def HUMAN_VOICE_AVAILABLE() -> bool:
    """Return True if FSDD voice samples are present in tests/voice_samples/."""
    _voice_dir = os.path.join(os.path.dirname(__file__), "voice_samples")
    return any(
        os.path.exists(os.path.join(_voice_dir, f"{d}_{s}_0.wav"))
        for s in ("jackson", "nicolas", "george")
        for d in (1, 2, 3)
    )


def speech_fixture_info() -> dict:
    """
    Return a summary of which fixture files are available, for diagnostics.

    Example output::

        {
          "barge_in": True,
          "tts_echo": False,
          "wakeword_speech": True,
          "any_available": True,
          "source": "edge-tts or gTTS (cached WAVs)",
        }
    """
    info = {
        "barge_in": _wav_exists(_BARGE_IN_WAV),
        "tts_echo": _wav_exists(_TTS_ECHO_WAV),
        "wakeword_speech": _wav_exists(_WAKEWORD_SPEECH_WAV),
    }
    info["any_available"] = any(info.values())
    info["source"] = (
        "cached WAVs in tests/fixture_audio/"
        if info["any_available"]
        else "synthetic fallback (no WAV files found)"
    )
    return info
