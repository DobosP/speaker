"""
Cross-platform audio input/output module.
Works on Windows, Mac, and Linux without any OS-specific dependencies.

Features:
- NLMS adaptive echo cancellation (pure NumPy, no native deps)
- Silero VAD neural network voice detection (fallback: WebRTC / RMS heuristic)
- Multiple TTS backends: Kokoro (local ONNX) > Supertonic > edge-tts > gTTS
- Sentence-chunked TTS playback for low-latency and natural interruption
- Automatic sample rate conversion for Whisper compatibility
"""
import logging
import numpy as np
import sounddevice as sd
import threading
import queue
import time
import subprocess
import re
import json
from dataclasses import dataclass
from typing import Callable, Optional, List, Any
import tempfile
import os
from contextlib import nullcontext
from collections import deque
from enum import Enum

from utils.voice_gate import SpeechBrainSpeakerVerifier
from utils.wakeword_service import build_wakeword_service, BaseWakewordService
from utils import tts_debug

# ── Resampling ────────────────────────────────────────────────────────────────
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# ── Audio playback ────────────────────────────────────────────────────────────
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# ── TTS backends ──────────────────────────────────────────────────────────────
try:
    import edge_tts
    import asyncio
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    from supertonic import TTS as SupertonicTTS
    import soundfile as sf
    SUPERTONIC_AVAILABLE = True
except ImportError:
    SUPERTONIC_AVAILABLE = False

# Kokoro TTS – lightweight on-device ONNX TTS (~82M params, Apache-2.0)
try:
    from kokoro_onnx import Kokoro as _KokoroEngine
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False

# Piper TTS – ultra-fast ONNX TTS, 30+ languages, real-time on CPU / Raspberry Pi
try:
    from piper import PiperVoice as _PiperVoice
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False

# MeloTTS – VITS2 distill, 6 languages (EN/ZH/JP/KR/ES/FR), ~600 MB, CPU-friendly
try:
    from melo.api import TTS as _MeloEngine
    MELOTTS_AVAILABLE = True
except ImportError:
    MELOTTS_AVAILABLE = False

# ── VAD backends ──────────────────────────────────────────────────────────────
# Silero VAD – neural network, dramatically better than WebRTC GMM
try:
    from silero_vad import load_silero_vad as _load_silero_vad
    import torch as _torch
    SILERO_VAD_AVAILABLE = True
except ImportError:
    SILERO_VAD_AVAILABLE = False

# Legacy WebRTC VAD (GMM-based, low accuracy – used as fallback only)
try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    WEBRTCVAD_AVAILABLE = False

# soundfile (needed for wav I/O)
try:
    import soundfile as sf
except ImportError:
    pass  # already imported via supertonic or will fail gracefully


# ═══════════════════════════════════════════════════════════════════════════════
#  NLMS Adaptive Echo Canceller
# ═══════════════════════════════════════════════════════════════════════════════

ECHO_FLOOR_MULTIPLIER = 4.0
STRONG_BARGE_IN_ECHO_MULTIPLIER = 4.0
STRONG_BARGE_IN_MIN_RMS = 0.075
STRONG_BARGE_IN_MIN_SEC = 0.12

# Uncalibrated-room barge-in: track slowly-moving ambient RMS so steady TV/babble
# does not accumulate score; real users produce an excursion above the running mean.
BARGE_AMBIENT_WARMUP_CHUNKS = 12
BARGE_AMBIENT_EMA_ALPHA = 0.055
BARGE_UNCALIBRATED_NOISE_FLOOR_FRAC = 0.34
# Below this coefficient-of-variation we merge bootstrap into _noise_floor for adapt().
BARGE_STATIONARY_CV_MAX = 0.26
# Only soften bootstrap when ambient is quiet (whisper-style); loud steady babble/TV
# keeps the full synthetic gate so RMS stays below noise_floor × ratio.
BARGE_BOOTSTRAP_SOFT_MAX_RMS = 0.038

# Raise an existing calibrated noise_floor during TTS when the environment gets louder.
TTS_NOISE_FLOOR_ADAPT_CAP = 0.28

class NLMSEchoCanceller:
    """
    Normalized Least Mean Squares adaptive echo canceller.

    Same core algorithm used by Speex AEC, implemented in pure NumPy using
    block processing with stride tricks for efficiency (no Python per-sample
    loops).

    Processes mic frames against a TTS reference signal and outputs echo-
    cancelled audio.  The adaptive filter learns the acoustic transfer
    function between speaker and microphone automatically.

    Typical usage::

        aec = NLMSEchoCanceller(filter_length=800, step_size=0.3)
        aec.feed_reference(tts_audio)          # before / as TTS plays
        cleaned = aec.process_frame(mic_frame)  # per mic callback
    """

    def __init__(
        self,
        filter_length: int = 800,
        step_size: float = 0.3,
        leakage: float = 1e-5,
    ):
        """
        Args:
            filter_length: Adaptive filter taps.  50 ms at 16 kHz = 800.
            step_size: NLMS mu (0–1).  Lower = more stable, higher = faster.
            leakage: Weight decay factor to prevent divergence.
        """
        self.L = filter_length
        self.mu = step_size
        self.leakage = leakage
        # Filter taps (chronological: oldest tap = index 0)
        self.h = np.zeros(filter_length, dtype=np.float64)
        # Reference history (chronological: oldest = index 0)
        self.ref_history = np.zeros(filter_length, dtype=np.float64)
        # Buffered reference audio from TTS (consumed as mic frames arrive)
        self._ref_buffer = np.zeros(0, dtype=np.float32)
        self._active = False  # True when reference is available

    # ── public API ────────────────────────────────────────────────────────

    def feed_reference(self, audio: np.ndarray):
        """Queue TTS reference audio (will be consumed per mic frame)."""
        audio = audio.flatten().astype(np.float32)
        self._ref_buffer = np.concatenate([self._ref_buffer, audio])
        self._active = True

    @property
    def active(self) -> bool:
        return self._active and len(self._ref_buffer) > 0

    def process_frame(self, mic_frame: np.ndarray) -> np.ndarray:
        """
        Cancel echo from *mic_frame* and return cleaned audio.

        If no reference is available the frame is returned unchanged.
        """
        if not self._active:
            return mic_frame

        N = len(mic_frame)
        d = mic_frame.astype(np.float64)
        ref = self._get_reference(N)

        # Concatenate history with new reference (chronological order)
        chronological = np.concatenate([self.ref_history, ref])
        # Length = L + N

        # Build reference matrix X  (N × L)  via stride tricks – zero copy.
        # X[i, j] = chronological[i + j + 1]  (oldest-first reference window)
        stride = chronological.strides[0]
        base = chronological[1:]          # skip element 0
        X = np.lib.stride_tricks.as_strided(
            base,
            shape=(N, self.L),
            strides=(stride, stride),
        )
        # X is a VIEW – do not write to chronological until done.

        # Echo estimate  y = X h
        y = X @ self.h                    # (N,)

        # Error (echo-cancelled signal)
        e = d - y                         # (N,)

        # Block NLMS weight update with per-row power normalisation
        row_power = np.einsum("ij,ij->i", X, X) + 1e-8   # (N,)
        gradient = X.T @ (e / row_power)                    # (L,)
        self.h *= (1.0 - self.leakage)
        self.h += self.mu * gradient / max(N, 1)

        # Store last L reference samples for next frame
        self.ref_history = chronological[N : N + self.L].copy()

        return e.astype(np.float32)

    def reset(self):
        """Reset filter state completely (weights + buffer). Use only on hard resets."""
        self.h[:] = 0
        self.ref_history[:] = 0
        self._ref_buffer = np.zeros(0, dtype=np.float32)
        self._active = False

    def clear_buffer(self):
        """Clear reference buffer but preserve learned filter weights.

        The room acoustic transfer function (encoded in self.h) does not change
        between TTS turns, so preserving the weights means the next TTS turn
        benefits from the already-converged AEC immediately rather than starting
        from zero and needing hundreds of frames to re-learn.
        """
        self._ref_buffer = np.zeros(0, dtype=np.float32)
        self._active = False

    # ── private ───────────────────────────────────────────────────────────

    def _get_reference(self, n: int) -> np.ndarray:
        """Dequeue *n* samples of reference audio (zero-pad if exhausted)."""
        if len(self._ref_buffer) >= n:
            ref = self._ref_buffer[:n].astype(np.float64)
            self._ref_buffer = self._ref_buffer[n:]
            return ref
        ref = np.zeros(n, dtype=np.float64)
        avail = len(self._ref_buffer)
        if avail > 0:
            ref[:avail] = self._ref_buffer.astype(np.float64)
            self._ref_buffer = np.zeros(0, dtype=np.float32)
        if avail == 0:
            self._active = False
        return ref


# ═══════════════════════════════════════════════════════════════════════════════
#  Silero VAD Detector
# ═══════════════════════════════════════════════════════════════════════════════

class SileroVADDetector:
    """
    Neural-network VAD using Silero VAD (ONNX, ~2 MB).
    Much more accurate than WebRTC VAD at distinguishing speech from noise or
    echo residuals.  Singleton – model is loaded once.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        if not SILERO_VAD_AVAILABLE:
            raise RuntimeError("silero-vad not installed")
        self._model = _load_silero_vad()
        self._initialized = True

    def reset_states(self) -> None:
        """Reset GRU hidden states.

        Call this when starting a new independent audio stream so that
        accumulated context from prior noise-heavy audio does not suppress
        speech probability at the beginning of the new stream.
        """
        if hasattr(self._model, "reset_states"):
            self._model.reset_states()

    def is_speech(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int = 16000,
        threshold: float = 0.5,
    ) -> bool:
        """Return True if the chunk contains speech above *threshold*."""
        prob = self.speech_probability(audio_chunk, sample_rate)
        return prob >= threshold

    def speech_probability(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int = 16000,
    ) -> float:
        """Return average speech probability for *audio_chunk*."""
        chunk = audio_chunk.flatten().astype(np.float32)

        # Silero VAD only supports 16 kHz (and 8 kHz)
        if sample_rate != 16000:
            if LIBROSA_AVAILABLE:
                chunk = librosa.resample(
                    chunk, orig_sr=sample_rate, target_sr=16000
                )
            else:
                return 0.0

        window_size = 512  # 32 ms @ 16 kHz
        if len(chunk) < window_size:
            chunk = np.pad(chunk, (0, window_size - len(chunk)))

        tensor = _torch.from_numpy(chunk)
        probs: list[float] = []
        for start in range(0, len(chunk) - window_size + 1, window_size):
            window = tensor[start : start + window_size]
            prob = self._model(window, 16000).item()
            probs.append(prob)

        return float(np.mean(probs)) if probs else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Sentence splitter (for chunked TTS)
# ═══════════════════════════════════════════════════════════════════════════════

def split_sentences(text: str) -> List[str]:
    """Split *text* into sentences for chunked TTS playback."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences: List[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Merge very short fragments with previous sentence
        if sentences and len(sentences[-1]) < 8:
            sentences[-1] += " " + part
        else:
            sentences.append(part)
    return sentences if sentences else [text]


# ═══════════════════════════════════════════════════════════════════════════════
#  Timestamped realtime components
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FrameFeatures:
    timestamp: float
    rms: float
    threshold: float
    voiced: bool
    echo_similarity: float
    raw_rms: float
    noise_floor_calibrated: bool = False
    # RMS of the TTS reference audio at the current playback position.
    # Non-zero only when a TTS reference has been set (real playback scenario).
    # Used for adaptive echo threshold and to suppress the calibrated energy
    # path during loud TTS playback (prevents self-interruption).
    reference_rms: float = 0.0


class PlaybackReferenceClock:
    """Tracks playback reference position in samples for frame-accurate sync."""

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self._position_samples = 0
        self._reference_set_time = 0.0

    def reset(self):
        self._position_samples = 0
        self._reference_set_time = 0.0

    def set_reference_time(self):
        self._position_samples = 0
        self._reference_set_time = time.time()

    def consume(self, sample_count: int):
        self._position_samples += max(0, int(sample_count))

    @property
    def position_samples(self) -> int:
        return self._position_samples

    @property
    def reference_set_time(self) -> float:
        return self._reference_set_time


class EchoGuard:
    """
    Encapsulates AEC + echo-similarity logic with a shared reference clock.
    """

    def __init__(
        self,
        sample_rate: int,
        nlms: Optional[NLMSEchoCanceller],
        max_ref_sec: float = 20.0,
        reference_delay_sec: float = 0.12,
    ):
        self.sample_rate = sample_rate
        self.nlms = nlms
        self.max_ref_sec = max_ref_sec
        self.reference_delay_sec = reference_delay_sec
        self.clock = PlaybackReferenceClock(sample_rate)
        self._reference: Optional[np.ndarray] = None

    def set_reference(self, audio: np.ndarray):
        max_samples = int(self.max_ref_sec * self.sample_rate)
        ref = audio.flatten().astype(np.float32)
        if max_samples > 0 and len(ref) > max_samples:
            ref = ref[:max_samples]
        delay_samples = int(self.reference_delay_sec * self.sample_rate)
        delayed_ref = np.concatenate(
            [np.zeros(delay_samples, dtype=np.float32), ref]
        )
        self._reference = delayed_ref
        self.clock.set_reference_time()
        if self.nlms is not None:
            self.nlms.feed_reference(delayed_ref)

    def clear_reference(self):
        if self.nlms is not None:
            self.nlms.clear_buffer()
        self._reference = None
        self.clock.reset()

    def process(self, mic_chunk: np.ndarray) -> np.ndarray:
        if self.nlms is None:
            return mic_chunk
        return self.nlms.process_frame(mic_chunk)

    def reference_rms_at_position(self, chunk_len: int) -> float:
        """RMS of the reference window at the *current* clock position.

        Called BEFORE similarity() advances the clock, so it returns the level
        of the TTS audio that was playing during this mic frame.  Returns 0.0
        when no reference has been set.
        """
        if self._reference is None:
            return 0.0
        idx = self.clock.position_samples
        start = max(0, idx - chunk_len)
        end = min(idx + chunk_len, len(self._reference))
        if end <= start:
            return 0.0
        window = self._reference[start:end]
        if len(window) == 0:
            return 0.0
        return float(np.sqrt(np.mean(window.astype(np.float64) ** 2)))

    def similarity(self, raw_mic_chunk: np.ndarray) -> float:
        """
        Return the maximum cosine similarity between *raw_mic_chunk* and the
        playback reference over a lag window of 0 – 80 ms.

        Zero-lag comparison fails in real rooms: TTS audio travels from the
        speaker to the microphone with a propagation delay (≈3 ms/metre) plus
        early reflections (20–80 ms).  Comparing mic[t] with ref[t] gives near-
        zero similarity for any delayed echo → the echo gate passes → self-
        barge-in fires.  Checking the maximum across all plausible lags detects
        the delayed echo and correctly blocks it.
        """
        if self._reference is None:
            return 0.0
        chunk_len = len(raw_mic_chunk)
        idx = self.clock.position_samples
        self.clock.consume(chunk_len)

        if idx >= len(self._reference):
            return 0.0

        mic_norm = float(np.linalg.norm(raw_mic_chunk))
        if mic_norm < 1e-6:
            return 0.0

        # Search up to (reference_delay + 50 ms) to cover the prepended delay
        # that EchoGuard adds to the reference (default 120 ms).  Searching
        # only 0–80 ms would miss echoes whose propagation delay equals or
        # exceeds the reference pre-delay, resulting in near-zero similarity
        # for signals that are obviously the same audio.
        lag_max = int(max(0.150, self.reference_delay_sec + 0.050) * self.sample_rate)
        lag_step = max(1, int(0.002 * self.sample_rate))

        best_sim = 0.0
        for lag in range(0, min(lag_max, idx) + lag_step, lag_step):
            ref_start = max(0, idx - lag)
            ref_end = min(ref_start + chunk_len, len(self._reference))
            if ref_end <= ref_start:
                continue
            ref_chunk = self._reference[ref_start:ref_end]
            if len(ref_chunk) < chunk_len:
                ref_chunk = np.pad(ref_chunk, (0, chunk_len - len(ref_chunk)))
            ref_norm = float(np.linalg.norm(ref_chunk))
            if ref_norm < 1e-6:
                continue
            sim = abs(float(np.dot(raw_mic_chunk, ref_chunk)) / (mic_norm * ref_norm))
            if sim > best_sim:
                best_sim = sim

        return best_sim


class SpeechGate:
    """Centralizes thresholding and VAD gating decisions."""

    def __init__(
        self,
        vad_threshold: float,
        barge_in_rms_ratio: float,
        barge_in_min_rms_ratio: float,
    ):
        self.vad_threshold = vad_threshold
        self.barge_in_rms_ratio = barge_in_rms_ratio
        self.barge_in_min_rms_ratio = barge_in_min_rms_ratio

    def barge_threshold(self, noise_floor: Optional[float]) -> float:
        noise = noise_floor if noise_floor is not None else self.vad_threshold * 0.5
        return max(self.vad_threshold, noise * self.barge_in_rms_ratio)

    def passes_noise_gate(self, rms: float, noise_floor: Optional[float]) -> bool:
        if noise_floor is None:
            return True
        min_rms = noise_floor * max(self.barge_in_min_rms_ratio, 1.0)
        return rms >= min_rms


class BargeInDetector:
    """Score-based barge-in detector to reduce self-triggering."""

    def __init__(
        self,
        sample_rate: int,
        min_speech_sec: float,
        echo_corr_threshold: float,
    ):
        self.sample_rate = sample_rate
        self.min_speech_samples = int(min_speech_sec * sample_rate)
        self.echo_corr_threshold = echo_corr_threshold
        self._above_samples = 0

    def reset(self):
        self._above_samples = 0

    @property
    def above_samples(self) -> int:
        return self._above_samples

    def score(self, features: FrameFeatures) -> float:
        # ── Adaptive echo suppression ─────────────────────────────────────
        # When a TTS reference is active (reference_rms > 0.01), the echo block
        # threshold scales down from echo_corr_threshold toward 0.20 as TTS
        # gets louder.  This catches room echoes whose cross-correlation would
        # not reach the static threshold but are still clearly TTS bleed.
        #
        # Crucially: if Silero confirms voiced=True, the adaptive block is
        # overridden.  Real user speech can legitimately correlate with TTS at
        # moderate levels; Silero's neural VAD discriminates where echo_sim alone
        # cannot.  The unconditional static block (no TTS reference) is kept.
        tts_active = features.reference_rms > 0.01
        if tts_active:
            t = min(1.0, (features.reference_rms - 0.01) / (0.10 - 0.01))
            effective_echo_threshold = self.echo_corr_threshold * (1.0 - t) + 0.20 * t
            effective_echo_threshold = max(0.20, effective_echo_threshold)
        else:
            effective_echo_threshold = self.echo_corr_threshold
        echo_blocked = features.echo_similarity >= effective_echo_threshold
        # When TTS reference is active and Silero confirms real speech,
        # override the adaptive echo block for MODERATE similarity only.
        # High similarity (≥ 0.50) indicates the mic is receiving predominantly
        # the TTS reference signal itself — even if Silero detects speech
        # phonetics, it must be the TTS output, not the user.  Mixing user
        # speech with TTS bleed substantially reduces the cross-correlation
        # (typically below 0.30), so bona fide user speech passes the gate.
        # Note: the echo-floor gate (2b, below) is the primary defence against
        # moderate-similarity (0.20–0.45) echo; the hard cap here is a last
        # resort for very-high-correlation bleed that still slips past the floor.
        _HARD_ECHO_SIM = 0.50
        if tts_active and echo_blocked and features.voiced:
            if features.echo_similarity < _HARD_ECHO_SIM:
                echo_blocked = False
        if echo_blocked:
            tts_debug.log_echo_gate_blocked(
                echo_similarity=round(features.echo_similarity, 4),
                threshold=round(effective_echo_threshold, 4),
                reference_rms=round(features.reference_rms, 4),
                voiced=features.voiced,
            )
            return -10.0

        score = 0.0
        if features.voiced:
            score += 2.0
        if features.rms > features.threshold:
            score += 1.0
        if features.rms > (features.threshold * 1.5):
            score += 0.5
        # Strong-energy boost rules:
        #   1. Silero confirms voiced speech → always grant boost.
        #   2. Noise floor calibrated AND no TTS reference active → grant boost
        #      (quiet idle environment; loud sounds are probably user speech).
        #   3. TTS reference active → require Silero confirmation only; the
        #      calibrated energy path is suppressed because TTS echo bleeds
        #      into the mic at high RMS and must not trigger self-interruption.
        energy_calibrated = features.noise_floor_calibrated and not tts_active
        if features.rms > (features.threshold * 3.0) and (
            features.voiced or energy_calibrated
        ):
            score += 1.0
        if features.raw_rms <= 1e-6:
            score -= 0.5
        return score

    def soft_decay(self, frame_len: int) -> None:
        """
        Reduce the accumulator by half a frame without a full reset.

        Used for frames that fail the noise gate (word-boundary pauses, brief
        silence between syllables).  A hard reset would force the user to
        produce 4 consecutive uninterrupted voiced frames, which is too strict
        for natural conversational speech with normal micro-pauses.
        """
        self._above_samples = max(0, self._above_samples - frame_len // 2)

    def update(self, features: FrameFeatures, frame_len: int) -> bool:
        frame_score = self.score(features)
        if frame_score >= 2.0:
            self._above_samples += frame_len
        else:
            # Hard reset on any low-score frame that PASSES the noise gate.
            # (Frames that FAIL the noise gate never reach update() — the
            # caller uses soft_decay() for those instead, so word-boundary
            # pauses in calibrated environments do not zero the accumulator.)
            self._above_samples = 0
        return self._above_samples >= self.min_speech_samples


class ListenerState(Enum):
    IDLE = "idle"
    ARMED = "armed"
    LISTENING = "listening"
    ASSISTANT_SPEAKING = "assistant_speaking"
    RECOVER = "recover"

# ═══════════════════════════════════════════════════════════════════════════════
#  Device helpers
# ═══════════════════════════════════════════════════════════════════════════════

def list_audio_devices():
    """List all available audio devices with their indices."""
    devices = sd.query_devices()
    print("\n=== Available Audio Devices ===")
    for i, device in enumerate(devices):
        device_type = []
        if device["max_input_channels"] > 0:
            device_type.append("INPUT")
        if device["max_output_channels"] > 0:
            device_type.append("OUTPUT")
        type_str = "/".join(device_type) if device_type else "UNKNOWN"
        print(f"  [{i}] {device['name']} ({type_str})")

    try:
        default_input = sd.query_devices(kind="input")
    except Exception:
        default_input = None
    try:
        default_output = sd.query_devices(kind="output")
    except Exception:
        default_output = None

    if default_input:
        print(f"\nDefault Input:  {default_input['name']}")
    else:
        print("\nDefault Input:  (none)")
    if default_output:
        print(f"Default Output: {default_output['name']}")
    else:
        print("Default Output: (none)")
    print("=" * 35)
    return devices


def get_default_input_device() -> int:
    """Get the default input device index."""
    return sd.default.device[0]


def get_default_output_device() -> int:
    """Get the default output device index."""
    return sd.default.device[1]


# ═══════════════════════════════════════════════════════════════════════════════
#  AudioRecorder
# ═══════════════════════════════════════════════════════════════════════════════

class AudioRecorder:
    """
    Cross-platform audio recorder using sounddevice.

    Uses Voice Activity Detection (VAD) to detect when the user is speaking.
    Automatically handles sample rate conversion for Whisper compatibility.
    Supports barge-in: detects voice even while assistant is speaking.

    Echo cancellation pipeline:
        mic → NLMS AEC (subtract echo) → Silero VAD → barge-in decision
    """

    def __init__(
        self,
        callback: Callable[[np.ndarray], None],
        target_sample_rate: int = 16000,
        vad_threshold: float = 0.01,
        silence_duration: float = 1.5,
        barge_in_pre_roll_sec: float = 0.3,
        barge_in_min_speech_sec: float = 0.2,
        min_utterance_sec: float = 0.15,
        barge_in_rms_ratio: float = 2.0,
        barge_in_cooldown_sec: float = 0.5,
        use_webrtcvad: bool = True,
        partial_callback: Optional[Callable[[np.ndarray], None]] = None,
        partial_interval_sec: float = 1.0,
        adaptive_vad: bool = False,
        vad_noise_multiplier: float = 2.5,
        vad_noise_floor_min: float = 0.003,
        aec_enabled: bool = True,
        aec_strength: float = 0.3,
        aec_filter_ms: float = 80.0,
        aec_max_ref_sec: float = 20.0,
        simple_voiced_fallback: bool = True,
        barge_in_min_delay_sec: float = 0.5,
        barge_in_min_delay_after_ref_sec: float = 0.7,
        barge_in_min_rms_ratio: float = 3.0,
        echo_corr_threshold: float = 0.45,
        wakeword_enabled: bool = False,
        wakeword: Optional[str] = None,
        wakeword_threshold: float = 0.5,
        wakeword_timeout_sec: float = 5.0,
        wakeword_model_path: Optional[str] = None,
        wakeword_service_mode: str = "local",
        wakeword_policy: str = "strict_required",
        wakeword_miss_limit: int = 80,
        wakeword_recovery_window_sec: float = 3.0,
        speaker_verify_enabled: bool = False,
        speaker_enrollment_wav: Optional[str] = None,
        speaker_verify_threshold: float = 0.55,
        diagnostics_log_path: Optional[str] = None,
        diagnostics_log_frames: bool = False,
        wakeword_detector: Optional[Any] = None,
        speaker_verifier: Optional[Any] = None,
        device: Optional[int] = None,
        on_interrupt: Optional[Callable[..., None]] = None,
        console_print_lock: Optional[threading.Lock] = None,
    ):
        self.callback = callback
        self.on_interrupt = on_interrupt
        self.target_sample_rate = target_sample_rate
        self.vad_threshold = vad_threshold
        self.silence_duration = silence_duration
        self.barge_in_pre_roll_sec = barge_in_pre_roll_sec
        self.barge_in_min_speech_sec = barge_in_min_speech_sec
        self.min_utterance_sec = min_utterance_sec
        self.barge_in_rms_ratio = barge_in_rms_ratio
        self.barge_in_cooldown_sec = barge_in_cooldown_sec
        self.use_webrtcvad = use_webrtcvad and WEBRTCVAD_AVAILABLE
        self.partial_callback = partial_callback
        self.partial_interval_sec = partial_interval_sec
        self.adaptive_vad = adaptive_vad
        self.vad_noise_multiplier = vad_noise_multiplier
        self.vad_noise_floor_min = vad_noise_floor_min
        self.aec_enabled = aec_enabled
        self.aec_strength = aec_strength
        self.aec_filter_ms = aec_filter_ms
        self.aec_max_ref_sec = aec_max_ref_sec
        self.simple_voiced_fallback = simple_voiced_fallback
        self.barge_in_min_delay_sec = barge_in_min_delay_sec
        self.barge_in_min_delay_after_ref_sec = barge_in_min_delay_after_ref_sec
        self.barge_in_min_rms_ratio = barge_in_min_rms_ratio
        self.echo_corr_threshold = echo_corr_threshold
        self.wakeword_enabled = wakeword_enabled
        self.wakeword = wakeword
        self.wakeword_threshold = wakeword_threshold
        self.wakeword_timeout_sec = wakeword_timeout_sec
        self.wakeword_model_path = wakeword_model_path
        self.wakeword_service_mode = wakeword_service_mode
        self.wakeword_policy = wakeword_policy
        self.wakeword_miss_limit = max(1, int(wakeword_miss_limit))
        self.wakeword_recovery_window_sec = max(0.0, float(wakeword_recovery_window_sec))
        self.speaker_verify_enabled = speaker_verify_enabled
        self.speaker_enrollment_wav = speaker_enrollment_wav
        self.speaker_verify_threshold = speaker_verify_threshold
        self.diagnostics_log_path = diagnostics_log_path
        self.diagnostics_log_frames = diagnostics_log_frames
        self.device = device
        self._console_print_lock = console_print_lock
        self._diag_lock = threading.Lock()
        self._diag_warned = False
        self._tts_debug_gate_log_ts = 0.0

        # Resolve device and sample rate
        self.device_sample_rate = self._get_device_sample_rate()
        self.needs_resampling = self.device_sample_rate != target_sample_rate

        if self.needs_resampling:
            if not LIBROSA_AVAILABLE:
                print(
                    f"Warning: Device uses {self.device_sample_rate}Hz but "
                    "librosa not available for resampling."
                )
                self.target_sample_rate = self.device_sample_rate
                self.needs_resampling = False
            else:
                print(
                    f"Device sample rate: {self.device_sample_rate}Hz "
                    f"-> resampling to {target_sample_rate}Hz"
                )

        # ── NLMS AEC ─────────────────────────────────────────────────────
        filter_len = int(self.aec_filter_ms / 1000.0 * self.device_sample_rate)
        filter_len = max(filter_len, 160)
        self._nlms_aec = NLMSEchoCanceller(
            filter_length=filter_len,
            step_size=self.aec_strength,
        ) if self.aec_enabled else None
        self._echo_guard = EchoGuard(
            sample_rate=self.device_sample_rate,
            nlms=self._nlms_aec,
            max_ref_sec=self.aec_max_ref_sec,
            reference_delay_sec=0.12,
        )
        self._speech_gate = SpeechGate(
            vad_threshold=self.vad_threshold,
            barge_in_rms_ratio=self.barge_in_rms_ratio,
            barge_in_min_rms_ratio=self.barge_in_min_rms_ratio,
        )
        self._barge_detector = BargeInDetector(
            sample_rate=self.device_sample_rate,
            min_speech_sec=self.barge_in_min_speech_sec,
            echo_corr_threshold=self.echo_corr_threshold,
        )
        self._wakeword_service: BaseWakewordService = build_wakeword_service(
            mode=self.wakeword_service_mode,
            wakeword=self.wakeword,
            threshold=self.wakeword_threshold,
            model_path=self.wakeword_model_path,
            detector_override=(
                wakeword_detector
                if wakeword_detector is not None
                else None
            ),
        )
        if self.wakeword_enabled and not getattr(self._wakeword_service, "available", True):
            print("Warning: wakeword gate enabled but OpenWakeWord is unavailable; disabling wakeword gate.")
            self._diag_log(
                "wakeword_disabled_unavailable",
                wakeword=self.wakeword,
                wakeword_model_path=self.wakeword_model_path,
                wakeword_service_mode=self.wakeword_service_mode,
            )
            self.wakeword_enabled = False

        self._speaker_verifier = (
            speaker_verifier
            if speaker_verifier is not None
            else (
                SpeechBrainSpeakerVerifier(
                    enrollment_wav=self.speaker_enrollment_wav,
                    threshold=self.speaker_verify_threshold,
                )
                if self.speaker_enrollment_wav
                else None
            )
        )
        if (
            self.speaker_verify_enabled
            and (self._speaker_verifier is None or not getattr(self._speaker_verifier, "available", True))
        ):
            print(
                "Warning: speaker verification enabled but SpeechBrain or enrollment audio is unavailable; "
                "disabling speaker verification."
            )
            self._diag_log(
                "speaker_verify_disabled_unavailable",
                speaker_enrollment_wav=self.speaker_enrollment_wav,
            )
            self.speaker_verify_enabled = False

        # ── Silero VAD (best) or WebRTC VAD (fallback) ───────────────────
        self._silero_vad: Optional[SileroVADDetector] = None
        if SILERO_VAD_AVAILABLE:
            try:
                self._silero_vad = SileroVADDetector()
                print("VAD: Silero neural network (high accuracy)")
            except Exception as e:
                print(f"Warning: Silero VAD init failed ({e}), using fallback")

        self._vad = None
        if self._silero_vad is None and self.use_webrtcvad:
            self._vad = webrtcvad.Vad(2)
            print("VAD: WebRTC GMM (legacy fallback)")
        elif self._silero_vad is None:
            print("VAD: RMS energy + zero-crossing heuristic")

        # ── Recording state ──────────────────────────────────────────────
        self.is_recording = False
        self.assistant_is_speaking = False
        self._stop_event = threading.Event()
        self._barge_in_active = False
        self._last_barge_in_time = 0.0

        # VAD state
        self._audio_buffer = np.array([], dtype=np.float32)
        self._is_speaking = False
        self._silence_start = None
        self._interrupt_triggered = False
        self._barge_in_above_samples = 0
        self._strong_barge_in_samples = 0
        self._noise_floor = None
        self._rms_history: deque = deque(maxlen=12)
        self._pre_roll: deque = deque()
        self._pre_roll_samples = 0
        self._last_partial_time = 0.0
        self._calibrated = False
        self._tts_start_time = 0.0
        self._aec_ref_set_time = 0.0
        # Echo-floor: RMS of mic during the gate window (only TTS echo, no user
        # speech) is used after the gate to block frames that are at or below the
        # observed echo level.  Reset each TTS turn so it reflects current conditions.
        self._echo_floor_sum: float = 0.0
        self._echo_floor_count: int = 0
        self._echo_floor_baseline: Optional[float] = None
        # Room echo-attenuation ratio: mic_echo_rms / ref_rms.  Learned from the
        # first gate window; reused on subsequent turns to set the echo-floor
        # instantly without waiting for the full gate window again.  This halves
        # barge-in latency after the first TTS interaction in a session.
        self._room_echo_ratio: Optional[float] = None
        # Slow ambient tracker for barge-in when noise_floor was never calibrated.
        self._barge_ambient_ema: Optional[float] = None
        self._barge_ambient_chunks: int = 0
        self._barge_rms_ring: deque = deque(maxlen=BARGE_AMBIENT_WARMUP_CHUNKS)
        self._ambient_bootstrap_evaluated: bool = False
        # Optional hook for session recording; set by SessionRecorder.start()
        self._session_recorder = None
        self._wakeword_armed_until = 0.0
        self._wakeword_miss_count = 0
        self._listener_state = (
            ListenerState.ARMED if self.wakeword_enabled else ListenerState.IDLE
        )
        self._recover_until = 0.0

        # Echo reference tracking (legacy attrs kept for tests/backward compatibility)
        self._aec_ref = None
        self._aec_ref_idx = 0
        self._ref_samples_consumed = 0  # tracks position in ref during TTS

        # Audio queue
        self._audio_queue: queue.Queue = queue.Queue()
        self._worker_thread = None
        self._diag_log(
            "recorder_init",
            device=self.device,
            device_sample_rate=self.device_sample_rate,
            target_sample_rate=self.target_sample_rate,
            adaptive_vad=self.adaptive_vad,
            vad_threshold=self.vad_threshold,
            noise_multiplier=self.vad_noise_multiplier,
            barge_in_min_delay_sec=self.barge_in_min_delay_sec,
            barge_in_min_delay_after_ref_sec=self.barge_in_min_delay_after_ref_sec,
            barge_in_min_rms_ratio=self.barge_in_min_rms_ratio,
            echo_corr_threshold=self.echo_corr_threshold,
            wakeword_enabled=self.wakeword_enabled,
            wakeword=self.wakeword,
            wakeword_threshold=self.wakeword_threshold,
            wakeword_timeout_sec=self.wakeword_timeout_sec,
            wakeword_model_path=self.wakeword_model_path,
            wakeword_service_mode=self.wakeword_service_mode,
            wakeword_policy=self.wakeword_policy,
            wakeword_miss_limit=self.wakeword_miss_limit,
            wakeword_recovery_window_sec=self.wakeword_recovery_window_sec,
            wakeword_available=getattr(self._wakeword_service, "available", None),
            wakeword_labels=getattr(self._wakeword_service, "labels", []),
            speaker_verify_enabled=self.speaker_verify_enabled,
            speaker_verify_threshold=self.speaker_verify_threshold,
            speaker_enrollment_wav=self.speaker_enrollment_wav,
            speaker_verify_available=(
                None
                if self._speaker_verifier is None
                else getattr(self._speaker_verifier, "available", None)
            ),
        )

    def _tts_debug_gate(self, gate: str, **fields) -> None:
        """Rate-limited EchoGuard / SpeechGate diagnostics (live TTS debug)."""
        if not tts_debug.is_enabled():
            return
        now = time.time()
        if now - self._tts_debug_gate_log_ts < 2.0:
            return
        self._tts_debug_gate_log_ts = now
        if gate == "echo":
            tts_debug.log_echo_gate_blocked(**fields)
            tts_debug.console("blocked", "EchoGuard")
        else:
            tts_debug.log_speech_gate_blocked(**fields)
            tts_debug.console("blocked", "SpeechGate")

    def _diag_log(self, event: str, component: str = "audio", **payload):
        if not self.diagnostics_log_path:
            return
        record = {"ts": time.time(), "event": event, "component": component}
        record.update(payload)
        try:
            with self._diag_lock:
                parent = os.path.dirname(self.diagnostics_log_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with open(self.diagnostics_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=True) + "\n")
        except Exception:
            # Diagnostics must never break capture loop.
            if not self._diag_warned:
                print("Warning: could not write diagnostics log file")
                self._diag_warned = True

    def _set_listener_state(self, new_state: ListenerState, reason: str):
        if self._listener_state == new_state:
            return
        old = self._listener_state
        self._listener_state = new_state
        self._diag_log(
            "listener_state",
            from_state=old.value,
            to_state=new_state.value,
            reason=reason,
        )

    # ── Echo reference ────────────────────────────────────────────────────

    def set_echo_reference(self, audio: np.ndarray, sample_rate: int):
        """Set TTS output as echo reference for the AEC."""
        if audio is None or len(audio) == 0:
            return
        audio = audio.flatten().astype(np.float32)
        # Resample to device rate
        if sample_rate != self.device_sample_rate and LIBROSA_AVAILABLE:
            audio = librosa.resample(
                audio, orig_sr=sample_rate, target_sr=self.device_sample_rate
            )
        self._echo_guard.set_reference(audio)
        # Keep legacy reference mirrors for existing tests
        self._aec_ref = self._echo_guard._reference
        self._aec_ref_idx = 0
        self._ref_samples_consumed = 0
        self._aec_ref_set_time = time.time()
        # Reset echo-floor accumulators for this TTS turn
        self._echo_floor_sum = 0.0
        self._echo_floor_count = 0
        # If we already learned the room's echo attenuation ratio from a previous
        # gate window, pre-set the baseline instantly from the new reference audio.
        # This means the gate window is only needed on the very first TTS turn;
        # all subsequent turns get zero-latency echo floor protection.
        if self._room_echo_ratio is not None:
            ref_rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
            self._echo_floor_baseline = ref_rms * self._room_echo_ratio
        else:
            self._echo_floor_baseline = None

    def clear_echo_reference(self):
        """Clear echo reference (called when TTS stops)."""
        self._echo_guard.clear_reference()
        self._aec_ref = None
        self._aec_ref_idx = 0
        self._ref_samples_consumed = 0
        self._aec_ref_set_time = 0.0

    # ── AEC / VAD internals ───────────────────────────────────────────────

    def _apply_aec(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Apply echo cancellation to *audio_chunk*."""
        if self._nlms_aec is not None:
            return self._echo_guard.process(audio_chunk)
        # Fallback: correlation-based subtraction (legacy)
        return self._apply_aec_legacy(audio_chunk)

    def _apply_aec_legacy(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Legacy correlation-based AEC (kept as fallback)."""
        if self._aec_ref is None:
            return audio_chunk
        ref = self._aec_ref
        if self._aec_ref_idx >= len(ref):
            return audio_chunk
        end = min(self._aec_ref_idx + len(audio_chunk), len(ref))
        ref_chunk = ref[self._aec_ref_idx : end]
        self._aec_ref_idx = end
        if len(ref_chunk) < len(audio_chunk):
            ref_chunk = np.pad(ref_chunk, (0, len(audio_chunk) - len(ref_chunk)))
        denom = float(np.dot(ref_chunk, ref_chunk)) + 1e-6
        scale = float(np.dot(audio_chunk, ref_chunk)) / denom
        scale = max(0.0, min(scale, 1.5)) * 0.8
        return audio_chunk - (ref_chunk * scale)

    def _echo_similarity(self, audio_chunk: np.ndarray) -> float:
        """Normalised correlation between raw mic and TTS reference.

        Uses _ref_samples_consumed to track the current position within
        the reference audio, advancing by one frame length per call.
        """
        if self._echo_guard._reference is not None:
            sim = self._echo_guard.similarity(audio_chunk)
            self._ref_samples_consumed = self._echo_guard.clock.position_samples
            return sim

        # Backward-compatible path for tests that assign _aec_ref directly.
        if self._aec_ref is None:
            return 0.0
        idx = self._ref_samples_consumed
        if idx >= len(self._aec_ref):
            return 0.0
        end = min(idx + len(audio_chunk), len(self._aec_ref))
        ref_chunk = self._aec_ref[idx:end]
        self._ref_samples_consumed = end
        if len(ref_chunk) < len(audio_chunk):
            ref_chunk = np.pad(ref_chunk, (0, len(audio_chunk) - len(ref_chunk)))
        mic_norm = float(np.linalg.norm(audio_chunk))
        ref_norm = float(np.linalg.norm(ref_chunk))
        if mic_norm < 1e-6 or ref_norm < 1e-6:
            return 0.0
        return abs(float(np.dot(audio_chunk, ref_chunk)) / (mic_norm * ref_norm))

    def _is_voiced(
        self,
        audio_chunk: np.ndarray,
        threshold: float,
        barge_in_mode: bool = False,
    ) -> bool:
        """Check if *audio_chunk* contains speech using best available VAD.

        When *barge_in_mode* is True the Silero confidence threshold is raised
        above idle (0.50) so residual echo after AEC doesn't false-trigger, but
        stays below 0.80 so very quiet user speech can still register.
        """
        # Priority: Silero > WebRTC > simple heuristic
        if self._silero_vad is not None:
            silero_thresh = 0.52 if barge_in_mode else 0.50
            return self._silero_vad.is_speech(
                audio_chunk, self.device_sample_rate, threshold=silero_thresh
            )
        if self._vad is not None:
            return self._webrtcvad_voiced(audio_chunk)
        return self._simple_voiced(audio_chunk, threshold)

    def _webrtcvad_voiced(self, audio_chunk: np.ndarray) -> bool:
        """Check if chunk contains speech using WebRTC VAD (16 kHz only)."""
        if not self._vad or self.device_sample_rate != 16000:
            return False
        pcm16 = (
            np.clip(audio_chunk * 32768.0, -32768, 32767)
            .astype(np.int16)
            .tobytes()
        )
        frame_ms = 20
        frame_bytes = int(16000 * frame_ms / 1000) * 2
        if len(pcm16) < frame_bytes:
            return False
        voiced_frames = 0
        total_frames = 0
        for i in range(0, len(pcm16) - frame_bytes + 1, frame_bytes):
            frame = pcm16[i : i + frame_bytes]
            total_frames += 1
            if self._vad.is_speech(frame, 16000):
                voiced_frames += 1
        if total_frames == 0:
            return False
        return voiced_frames / total_frames >= 0.6

    def _simple_voiced(self, audio_chunk: np.ndarray, threshold: float) -> bool:
        """Lightweight voiced heuristic using energy + zero-crossing rate."""
        if not self.simple_voiced_fallback:
            return False
        if len(audio_chunk) < 16:
            return False
        rms = float(np.sqrt(np.mean(audio_chunk**2)))
        if rms < threshold:
            return False
        signs = np.sign(audio_chunk)
        zcr = np.mean(signs[:-1] * signs[1:] < 0)
        return 0.01 <= zcr <= 0.25

    # ── Backward compat ──────────────────────────────────────────────────

    @property
    def is_paused(self) -> bool:
        return self.assistant_is_speaking

    @is_paused.setter
    def is_paused(self, value: bool):
        self.assistant_is_speaking = value

    # ── Device helpers ───────────────────────────────────────────────────

    def _get_device_sample_rate(self) -> int:
        """Get the native sample rate for the recording device."""
        try:
            if self.device is None:
                default_input_idx = sd.default.device[0]
                if default_input_idx is not None and default_input_idx >= 0:
                    self.device = default_input_idx
                else:
                    devices = sd.query_devices()
                    for i, dev in enumerate(devices):
                        if dev["max_input_channels"] > 0:
                            self.device = i
                            break
            device_info = sd.query_devices(self.device)
            if isinstance(device_info, (list, tuple)):
                for dev in device_info:
                    if (
                        isinstance(dev, dict)
                        and dev.get("max_input_channels", 0) > 0
                    ):
                        return int(dev.get("default_samplerate", 44100))
                return 44100
            return int(device_info["default_samplerate"])
        except Exception as e:
            print(f"Warning: Could not query device sample rate: {e}")
            try:
                devices = sd.query_devices()
                for i, dev in enumerate(devices):
                    if dev["max_input_channels"] > 0:
                        self.device = i
                        return int(dev["default_samplerate"])
            except Exception:
                pass
            return 44100

    def _resample_audio(self, audio: np.ndarray) -> np.ndarray:
        """Resample audio from device rate to target rate."""
        if not self.needs_resampling:
            return audio
        return librosa.resample(
            audio, orig_sr=self.device_sample_rate, target_sr=self.target_sample_rate
        )

    @property
    def sample_rate(self) -> int:
        """Return the output sample rate (for backward compatibility)."""
        return self.target_sample_rate

    # ── Noise calibration ────────────────────────────────────────────────

    def _update_noise_floor(self, rms: float):
        if rms <= 0:
            return
        if self._noise_floor is None:
            self._noise_floor = rms
        else:
            alpha = 0.05
            self._noise_floor = (1 - alpha) * self._noise_floor + alpha * rms

    def _update_noise_floor_from_rms(self, rms_values: list) -> float:
        if not rms_values:
            return self._noise_floor or self.vad_noise_floor_min
        values = sorted(v for v in rms_values if v > 0)
        if not values:
            return self._noise_floor or self.vad_noise_floor_min
        median = values[len(values) // 2]
        self._noise_floor = max(median, self.vad_noise_floor_min)
        return self._noise_floor

    def _adapt_noise_floor_during_tts(self, rms: float) -> None:
        """Raise calibrated noise_floor during TTS when ambient energy ramps (TV/HVAC)."""
        if self._noise_floor is None or not self._calibrated:
            return
        nf = self._noise_floor
        cap = TTS_NOISE_FLOOR_ADAPT_CAP
        if rms <= nf * 1.08:
            return
        if rms > nf * 2.2:
            alpha = 0.14
        elif rms > nf * 1.15:
            alpha = 0.07
        else:
            alpha = 0.035
        blended = (1.0 - alpha) * nf + alpha * min(rms, cap)
        self._noise_floor = float(min(max(nf, blended), cap))

    def _bootstrap_uncalibrated_noise_floor_from_ambient(self, rms: float) -> None:
        """After warmup, derive noise_floor from median RMS (handles babble/TV without calibrate())."""
        if self._noise_floor is not None or self._ambient_bootstrap_evaluated:
            return
        if self._barge_ambient_chunks < BARGE_AMBIENT_WARMUP_CHUNKS:
            return
        if len(self._barge_rms_ring) < BARGE_AMBIENT_WARMUP_CHUNKS:
            return
        arr = np.array(self._barge_rms_ring, dtype=np.float64)
        s = np.sort(arr)
        if len(s) > 6:
            core = s[3:-3]
        elif len(s) > 2:
            core = s[1:-1]
        else:
            core = s
        med = float(np.median(core))
        if med <= 1e-9:
            return
        mean_r = float(arr.mean())
        cv = float(arr.std() / mean_r) if mean_r > 1e-9 else 1.0
        frac = BARGE_UNCALIBRATED_NOISE_FLOOR_FRAC
        boot = max(med * frac, self.vad_noise_floor_min)
        boot = float(min(boot, TTS_NOISE_FLOOR_ADAPT_CAP))
        self._ambient_bootstrap_evaluated = True
        # Speech-heavy *quiet* segments need a softer synthetic gate. Loud mixes
        # (TV/babble med RMS already high) must keep the full gate.
        if cv >= BARGE_STATIONARY_CV_MAX and med < BARGE_BOOTSTRAP_SOFT_MAX_RMS:
            self._noise_floor = boot * 0.72
        else:
            self._noise_floor = boot

    def _track_barge_ambient_ema(self, rms: float) -> None:
        """Running EMA of mic RMS during assistant speech (uncalibrated bootstrap)."""
        self._barge_ambient_chunks += 1
        self._barge_rms_ring.append(float(rms))
        if self._barge_ambient_ema is None:
            self._barge_ambient_ema = rms
        else:
            a = BARGE_AMBIENT_EMA_ALPHA
            self._barge_ambient_ema = (1.0 - a) * self._barge_ambient_ema + a * rms

    def calibrate(self, duration_sec: float = 2.5) -> dict:
        """Calibrate ambient noise floor."""
        if duration_sec <= 0:
            return {"calibrated": False, "reason": "duration<=0"}
        rms_samples: list = []
        done = threading.Event()

        def _cb(indata, frames, time_info, status):
            audio_data = indata.flatten().astype(np.float32)
            rms = float(np.sqrt(np.mean(audio_data**2)))
            rms_samples.append(rms)
            if time.time() - start_time >= duration_sec:
                done.set()

        try:
            start_time = time.time()
            with sd.InputStream(
                device=self.device,
                channels=1,
                samplerate=self.device_sample_rate,
                dtype=np.float32,
                callback=_cb,
                blocksize=1024,
            ):
                while not done.is_set():
                    time.sleep(0.05)
        except Exception as e:
            return {"calibrated": False, "reason": f"sounddevice failed: {e}"}

        noise_floor = self._update_noise_floor_from_rms(rms_samples)
        self._calibrated = True
        return {
            "calibrated": True,
            "noise_floor": noise_floor,
            "samples": len(rms_samples),
            "duration_sec": duration_sec,
        }

    def _get_vad_threshold(self) -> float:
        if not self.adaptive_vad:
            return self.vad_threshold
        noise = self._noise_floor or self.vad_noise_floor_min
        noise = max(noise, self.vad_noise_floor_min)
        return max(self.vad_threshold * 0.5, noise * self.vad_noise_multiplier)

    # ── Pre-roll / partial ───────────────────────────────────────────────

    def _push_pre_roll(self, audio_chunk: np.ndarray):
        if self.barge_in_pre_roll_sec <= 0:
            return
        self._pre_roll.append(audio_chunk.copy())
        self._pre_roll_samples += len(audio_chunk)
        max_samples = int(self.barge_in_pre_roll_sec * self.device_sample_rate)
        while self._pre_roll_samples > max_samples and self._pre_roll:
            removed = self._pre_roll.popleft()
            self._pre_roll_samples -= len(removed)

    def _drain_pre_roll(self) -> np.ndarray:
        if not self._pre_roll:
            return np.array([], dtype=np.float32)
        chunks = list(self._pre_roll)
        self._pre_roll.clear()
        self._pre_roll_samples = 0
        return np.concatenate(chunks)

    def _get_barge_in_threshold(self) -> float:
        return self._speech_gate.barge_threshold(self._noise_floor)

    def _update_wakeword_state(self, audio_chunk: np.ndarray):
        if not self.wakeword_enabled or self._is_speaking or self.assistant_is_speaking:
            return
        self._wakeword_service.submit_audio(audio_chunk, self.device_sample_rate)
        event = self._wakeword_service.poll_event()
        if event and event.detected:
            self._wakeword_armed_until = time.time() + self.wakeword_timeout_sec
            self._wakeword_miss_count = 0
            self._set_listener_state(ListenerState.ARMED, reason="wakeword_detected")
            print("\nWakeword detected: input gate opened")
            self._diag_log(
                "wakeword_detected",
                wakeword=self.wakeword,
                score=getattr(self._wakeword_service, "last_score", None),
                label=event.label,
                armed_until=self._wakeword_armed_until,
            )

    def _wakeword_gate_open(self) -> bool:
        if self.wakeword_policy == "legacy_compatible":
            return True
        if not self.wakeword_enabled:
            return True
        return time.time() <= self._wakeword_armed_until

    def _begin_barge_in_capture(self, audio_chunk: np.ndarray):
        self.assistant_is_speaking = False
        self._barge_in_active = True
        self._is_speaking = True
        self._silence_start = None
        pre_roll = self._drain_pre_roll()
        if len(pre_roll) > 0:
            self._audio_buffer = np.concatenate([pre_roll, audio_chunk])
        else:
            self._audio_buffer = audio_chunk.copy()

    def _emit_partial(self):
        if not self.partial_callback:
            return
        # Skip partial ASR while assistant plays TTS (saves CPU); barge-in capture
        # clears assistant_is_speaking and sets _barge_in_active instead.
        if self.assistant_is_speaking and not self._barge_in_active:
            return
        if len(self._audio_buffer) == 0:
            return
        now = time.time()
        if now - self._last_partial_time < self.partial_interval_sec:
            return
        self._last_partial_time = now
        resampled = self._resample_audio(self._audio_buffer.copy())
        self.partial_callback(resampled)

    # ── Audio capture callback ───────────────────────────────────────────

    def _audio_callback(self, indata, frames, time_info, status):
        if status and "overflow" not in str(status):
            print(f"Audio status: {status}")
        audio_data = indata.flatten().astype(np.float32)
        self._audio_queue.put(audio_data)

    # ── Main processing thread ───────────────────────────────────────────

    def _process_audio(self):
        """Worker thread – processes audio from the queue."""
        while not self._stop_event.is_set():
            try:
                audio_chunk = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            now = time.time()
            if self._listener_state == ListenerState.RECOVER and now >= self._recover_until:
                self._set_listener_state(
                    ListenerState.ARMED if self.wakeword_enabled else ListenerState.IDLE,
                    reason="recover_timeout",
                )

            raw_chunk = audio_chunk.copy()
            self._update_wakeword_state(raw_chunk)

            # --- Echo cancellation (only while assistant speaks) ----------
            if self.assistant_is_speaking and self.aec_enabled:
                audio_chunk = self._apply_aec(audio_chunk)

            # --- RMS energy -----------------------------------------------
            rms = float(np.sqrt(np.mean(audio_chunk**2)))
            self._rms_history.append(rms)
            self._push_pre_roll(audio_chunk)

            # Display audio level (serialize with assistant/latency console lines)
            level_bar = "█" * int(min(rms * 500, 20))
            status = "🔊" if self.assistant_is_speaking else "🎤"
            cm = (
                self._console_print_lock
                if self._console_print_lock is not None
                else nullcontext()
            )
            with cm:
                print(f"\r{status} [{level_bar:<20}] {rms:.4f}", end="", flush=True)

            # ═════ BARGE-IN MODE (assistant speaking) ═════════════════════
            if self.assistant_is_speaking:
                # 1) Timing gates
                now = time.time()
                if now - self._tts_start_time < self.barge_in_min_delay_sec:
                    # Still consume echo_similarity position so it stays in sync
                    self._echo_similarity(raw_chunk)
                    if self.diagnostics_log_frames:
                        self._diag_log(
                            "barge_gate_timing_tts",
                            rms=rms,
                            delta_tts_sec=now - self._tts_start_time,
                            min_delay_sec=self.barge_in_min_delay_sec,
                        )
                    continue
                if (
                    self._aec_ref_set_time
                    and now - self._aec_ref_set_time
                    < self.barge_in_min_delay_after_ref_sec
                ):
                    # Accumulate echo-floor baseline during this gate window.
                    # All audio here is TTS echo (no user speech yet), so the
                    # average RMS gives us the expected echo level in this room.
                    self._echo_floor_sum += rms
                    self._echo_floor_count += 1
                    self._echo_similarity(raw_chunk)
                    if self.diagnostics_log_frames:
                        self._diag_log(
                            "barge_gate_timing_ref",
                            rms=rms,
                            delta_ref_sec=now - self._aec_ref_set_time,
                            min_delay_after_ref_sec=self.barge_in_min_delay_after_ref_sec,
                        )
                    continue
                # Freeze echo-floor baseline the first frame after the gate expires.
                # Also learn the room_echo_ratio so future turns need no gate.
                if self._echo_floor_baseline is None and self._echo_floor_count > 0:
                    self._echo_floor_baseline = (
                        self._echo_floor_sum / self._echo_floor_count
                    )
                    if self._room_echo_ratio is None:
                        ref_rms = self._echo_guard.reference_rms_at_position(0)
                        if ref_rms > 0.001:
                            self._room_echo_ratio = self._echo_floor_baseline / ref_rms
                if now - self._last_barge_in_time < self.barge_in_cooldown_sec:
                    self._echo_similarity(raw_chunk)
                    if self.diagnostics_log_frames:
                        self._diag_log(
                            "barge_gate_cooldown",
                            rms=rms,
                            cooldown_elapsed_sec=now - self._last_barge_in_time,
                            cooldown_sec=self.barge_in_cooldown_sec,
                        )
                    continue

                # 2) Echo similarity on RAW mic (before AEC). This check
                #    advances _ref_samples_consumed so it stays time-aligned.
                #    Compute reference RMS BEFORE similarity() moves the clock.
                threshold = self._get_barge_in_threshold()
                ref_rms = self._echo_guard.reference_rms_at_position(len(raw_chunk))
                echo_sim = self._echo_similarity(raw_chunk)

                # Feed session recorder (if attached) with this barge-in-mode chunk
                if self._session_recorder is not None:
                    self._session_recorder.on_mic_chunk(raw_chunk, rms, echo_sim)
                if (
                    self._echo_floor_baseline is None
                    and echo_sim >= self.echo_corr_threshold * 0.95
                ):
                    self._barge_in_above_samples = 0
                    self._strong_barge_in_samples = 0
                    if self.diagnostics_log_frames:
                        self._diag_log(
                            "barge_gate_echo_blocked_no_floor",
                            rms=rms,
                            raw_rms=float(np.sqrt(np.mean(raw_chunk**2))),
                            echo_similarity=echo_sim,
                            echo_corr_threshold=self.echo_corr_threshold,
                        )
                    continue
                strong_threshold = max(
                    threshold * 6.0,
                    STRONG_BARGE_IN_MIN_RMS,
                    (
                        self._echo_floor_baseline * STRONG_BARGE_IN_ECHO_MULTIPLIER
                        if self._echo_floor_baseline is not None
                        else threshold * 6.0
                    ),
                )
                if echo_sim >= self.echo_corr_threshold:
                    strong_voiced = (
                        rms >= strong_threshold
                        and self._is_voiced(audio_chunk, threshold, barge_in_mode=True)
                    )
                    if strong_voiced:
                        self._strong_barge_in_samples += len(audio_chunk)
                        if (
                            self._strong_barge_in_samples
                            >= int(STRONG_BARGE_IN_MIN_SEC * self.device_sample_rate)
                        ):
                            echo_sim = min(echo_sim, self.echo_corr_threshold - 1e-6)
                        else:
                            continue
                    else:
                        self._strong_barge_in_samples = 0
                        # High correlation with TTS reference → it's echo
                        self._barge_in_above_samples = 0
                        if self.diagnostics_log_frames:
                            self._diag_log(
                                "barge_gate_echo_blocked",
                                rms=rms,
                                raw_rms=float(np.sqrt(np.mean(raw_chunk**2))),
                                echo_similarity=echo_sim,
                                echo_corr_threshold=self.echo_corr_threshold,
                            )
                        self._tts_debug_gate(
                            "echo",
                            rms=rms,
                            echo_similarity=echo_sim,
                            echo_corr_threshold=self.echo_corr_threshold,
                        )
                        continue

                # 2b) Echo-floor gate: mic energy at/below observed echo level during TTS.
                #     Without an actual playback reference (tests / missing echo API),
                #     skip — baseline RMS reflects ambient babble/speech, not echo leak.
                if (
                    ref_rms > 0.008
                    and self._echo_floor_baseline is not None
                    and rms < self._echo_floor_baseline * ECHO_FLOOR_MULTIPLIER
                ):
                    self._barge_detector.soft_decay(len(audio_chunk))
                    self._barge_in_above_samples = self._barge_detector.above_samples
                    self._strong_barge_in_samples = 0
                    if self.diagnostics_log_frames:
                        self._diag_log(
                            "barge_gate_echo_floor",
                            rms=rms,
                            echo_floor_baseline=self._echo_floor_baseline,
                            echo_floor_threshold=(
                                self._echo_floor_baseline * ECHO_FLOOR_MULTIPLIER
                            ),
                        )
                    continue

                # 2c) Environment drift for calibrated installs (TV on mid-session).
                self._adapt_noise_floor_during_tts(rms)

                # 2d) Uncalibrated installs: after warmup, optionally bootstrap noise_floor
                # from *stationary* RMS (TV/babble); speech-like variance skips bootstrap.
                if self._noise_floor is None:
                    self._track_barge_ambient_ema(rms)
                    self._bootstrap_uncalibrated_noise_floor_from_ambient(rms)
                    if self._barge_ambient_chunks < BARGE_AMBIENT_WARMUP_CHUNKS:
                        self._barge_detector.soft_decay(len(audio_chunk))
                        self._barge_in_above_samples = self._barge_detector.above_samples
                        self._strong_barge_in_samples = 0
                        continue

                # 3) RMS-based noise floor gate.
                # During TTS playback the echo-floor gate (2b) has already
                # blocked pure echo.  A full 3× noise-floor ratio prevents soft
                # speakers from interrupting because their voice mixed with TTS
                # bleed sits below the gate.  We therefore relax the ratio to
                # 1.5× while TTS is active; the echo-floor + echo-similarity
                # layers provide the false-positive protection instead.
                tts_noise_ratio = 1.5 if (ref_rms > 0.01) else self.barge_in_min_rms_ratio
                effective_noise_floor = (
                    self._noise_floor * max(tts_noise_ratio, 1.0)
                    if self._noise_floor is not None
                    else None
                )
                if effective_noise_floor is not None and rms < effective_noise_floor:
                    self._barge_detector.soft_decay(len(audio_chunk))
                    self._barge_in_above_samples = self._barge_detector.above_samples
                    self._strong_barge_in_samples = 0
                    if self.diagnostics_log_frames:
                        self._diag_log(
                            "barge_gate_noise_blocked",
                            rms=rms,
                            threshold=threshold,
                            noise_floor=self._noise_floor,
                        )
                    self._tts_debug_gate(
                        "noise",
                        gate="barge_gate_noise_blocked",
                        rms=rms,
                        noise_floor=self._noise_floor,
                    )
                    continue

                # 4) VAD on echo-cancelled audio (barge_in_mode raises
                #    the Silero confidence threshold to 0.80)
                voiced = self._is_voiced(
                    audio_chunk, threshold, barge_in_mode=True
                )

                frame = FrameFeatures(
                    timestamp=now,
                    rms=rms,
                    threshold=threshold,
                    voiced=voiced,
                    echo_similarity=echo_sim,
                    raw_rms=float(np.sqrt(np.mean(raw_chunk**2))),
                    noise_floor_calibrated=(
                        self._calibrated
                        or (
                            self._noise_floor is not None
                            and not self._ambient_bootstrap_evaluated
                        )
                    ),
                    reference_rms=ref_rms,
                )
                confirmed = self._barge_detector.update(frame, len(audio_chunk))
                self._barge_in_above_samples = self._barge_detector.above_samples
                score = self._barge_detector.score(frame)
                if frame.voiced and rms >= strong_threshold:
                    self._strong_barge_in_samples += len(audio_chunk)
                else:
                    self._strong_barge_in_samples = 0
                if (
                    self._strong_barge_in_samples
                    >= int(STRONG_BARGE_IN_MIN_SEC * self.device_sample_rate)
                ):
                    confirmed = True
                if self.diagnostics_log_frames:
                    self._diag_log(
                        "barge_frame",
                        rms=rms,
                        raw_rms=frame.raw_rms,
                        threshold=threshold,
                        voiced=voiced,
                        echo_similarity=echo_sim,
                        score=score,
                        above_samples=self._barge_in_above_samples,
                        strong_above_samples=self._strong_barge_in_samples,
                        strong_threshold=strong_threshold,
                        min_samples=self._barge_detector.min_speech_samples,
                        confirmed=confirmed,
                    )

                if confirmed and not self._interrupt_triggered:
                    self._last_barge_in_time = now
                    interrupt_samples = max(
                        self._barge_in_above_samples,
                        self._strong_barge_in_samples,
                    )
                    info = {
                        "rms": rms,
                        "threshold": threshold,
                        "voiced": voiced,
                        "duration_sec": interrupt_samples / max(self.device_sample_rate, 1),
                        "timestamp": self._last_barge_in_time,
                        "echo": False,  # passed echo gate above
                    }
                    should_interrupt = True
                    if self.on_interrupt:
                        try:
                            result = self.on_interrupt(info)
                            if result is False:
                                should_interrupt = False
                        except TypeError:
                            self.on_interrupt()
                    if should_interrupt:
                        print(
                            "\r" + " " * 60 + "\r", end="", flush=True
                        )
                        print(
                            f"BARGE-IN! (RMS: {rms:.4f} > {threshold:.4f}"
                            f"  voiced={voiced}  echo_sim={echo_sim:.2f})"
                        )
                        self._interrupt_triggered = True
                        self._begin_barge_in_capture(audio_chunk)
                        self._set_listener_state(
                            ListenerState.LISTENING, reason="barge_in_capture"
                        )
                        self._diag_log(
                            "barge_in_confirmed",
                            rms=rms,
                            raw_rms=frame.raw_rms,
                            threshold=threshold,
                            voiced=voiced,
                            echo_similarity=echo_sim,
                            score=score,
                            above_samples=self._barge_in_above_samples,
                        )
                    else:
                        self._diag_log(
                            "barge_in_ignored_by_callback",
                            rms=rms,
                            threshold=threshold,
                            voiced=voiced,
                            echo_similarity=echo_sim,
                            score=score,
                        )
                continue

            # ═════ NORMAL RECORDING MODE ══════════════════════════════════
            if self._is_speaking:
                if rms > self._get_vad_threshold() * 0.3:
                    self._audio_buffer = np.append(self._audio_buffer, audio_chunk)
                    self._emit_partial()

                if rms < self._get_vad_threshold():
                    if self._silence_start is None:
                        self._silence_start = time.time()
                    elif time.time() - self._silence_start > self.silence_duration:
                        self._finish_recording()
                else:
                    self._silence_start = None
            else:
                # Skip new speech onset during RECOVER.  The reverb tail from
                # the just-ended TTS lingers for up to RT60 (≈250 ms) and its
                # RMS easily exceeds the VAD threshold.  Blocking onset here
                # prevents a spurious recording from starting.  Pre-roll audio
                # continues to accumulate, so any speech the user uttered during
                # the RECOVER window is captured once the state transitions to
                # IDLE/ARMED and the next voiced frame arrives.
                if self._listener_state == ListenerState.RECOVER:
                    self._update_noise_floor(rms)
                    continue

                # Use Silero / VAD for speech onset detection too
                if rms > self._get_vad_threshold():
                    if not self._wakeword_gate_open():
                        self._wakeword_miss_count += 1
                        if (
                            self.wakeword_policy == "hybrid_recovery"
                            and self._wakeword_miss_count >= self.wakeword_miss_limit
                        ):
                            self._wakeword_armed_until = (
                                time.time() + self.wakeword_recovery_window_sec
                            )
                            self._wakeword_miss_count = 0
                            self._set_listener_state(
                                ListenerState.ARMED,
                                reason="hybrid_recovery_window",
                            )
                            self._diag_log(
                                "wakeword_hybrid_recovery_opened",
                                component="wakeword",
                                recovery_window_sec=self.wakeword_recovery_window_sec,
                            )
                            continue
                        if self.diagnostics_log_frames:
                            self._diag_log(
                                "wakeword_gate_blocked",
                                component="wakeword",
                                rms=rms,
                                vad_threshold=self._get_vad_threshold(),
                                armed_until=self._wakeword_armed_until,
                                now=time.time(),
                                miss_count=self._wakeword_miss_count,
                            )
                        continue
                    voiced_start = self._is_voiced(audio_chunk, self._get_vad_threshold())
                    if voiced_start or rms > self._get_vad_threshold() * 2:
                        print("\r" + " " * 50 + "\r", end="", flush=True)
                        self._is_speaking = True
                        self._set_listener_state(
                            ListenerState.LISTENING, reason="speech_onset"
                        )
                        if self.wakeword_enabled:
                            # Consume wakeword token; require a fresh one next turn.
                            self._wakeword_armed_until = 0.0
                        self._diag_log(
                            "speech_start",
                            rms=rms,
                            vad_threshold=self._get_vad_threshold(),
                            voiced_start=voiced_start,
                        )
                        self._silence_start = None
                        self._audio_buffer = audio_chunk.copy()
                else:
                    self._update_noise_floor(rms)

    # ── Recording lifecycle ──────────────────────────────────────────────

    def _finish_recording(self):
        print("\r" + " " * 50 + "\r", end="", flush=True)
        if len(self._audio_buffer) == 0:
            self._diag_log("recording_skipped_empty")
            self._reset_recording_state()
            return
        avg_energy = float(np.sqrt(np.mean(self._audio_buffer**2)))
        duration = len(self._audio_buffer) / self.device_sample_rate
        if avg_energy < self.vad_threshold * 0.5:
            print(f"Skipped {duration:.1f}s (too quiet: {avg_energy:.4f})")
            self._diag_log(
                "recording_skipped_too_quiet",
                duration_sec=duration,
                avg_energy=avg_energy,
                min_energy=self.vad_threshold * 0.5,
            )
            self._reset_recording_state()
            return
        if duration < self.min_utterance_sec:
            print(f"Skipped {duration:.1f}s (too short, min {self.min_utterance_sec:.2f}s)")
            self._diag_log(
                "recording_skipped_too_short",
                duration_sec=duration,
                min_duration_sec=self.min_utterance_sec,
                avg_energy=avg_energy,
            )
            self._reset_recording_state()
            return
        if self.speaker_verify_enabled and self._speaker_verifier is not None:
            verified = False
            try:
                verified = bool(
                    self._speaker_verifier.verify(
                        self._audio_buffer.copy(),
                        self.device_sample_rate,
                    )
                )
            except Exception:
                verified = False
            if not verified:
                print(f"Skipped {duration:.1f}s (speaker verification failed)")
                self._diag_log(
                    "recording_skipped_speaker_verify_failed",
                    duration_sec=duration,
                    avg_energy=avg_energy,
                    speaker_score=getattr(self._speaker_verifier, "last_score", None),
                    speaker_threshold=self.speaker_verify_threshold,
                )
                self._reset_recording_state()
                return
            self._diag_log(
                "speaker_verify_passed",
                duration_sec=duration,
                speaker_score=getattr(self._speaker_verifier, "last_score", None),
                speaker_threshold=self.speaker_verify_threshold,
            )
        print(f"Processing {duration:.1f}s of audio (energy: {avg_energy:.4f})...")
        self._diag_log(
            "recording_processing",
            duration_sec=duration,
            avg_energy=avg_energy,
            samples=len(self._audio_buffer),
        )
        resampled = self._resample_audio(self._audio_buffer.copy())
        try:
            self.callback(resampled)
        except Exception as exc:
            import traceback as _tb
            print(f"\n[AudioRecorder] callback raised {type(exc).__name__}: {exc}")
            _tb.print_exc()
            self._diag_log(
                "callback_exception",
                exception_type=type(exc).__name__,
                exception_msg=str(exc),
                duration_sec=duration,
            )
            # Fall through: reset state so the system can handle the next utterance.
        self._reset_recording_state()
        self._barge_in_active = False
        if self.wakeword_enabled:
            self._set_listener_state(ListenerState.ARMED, reason="utterance_finished")
        else:
            self._set_listener_state(ListenerState.IDLE, reason="utterance_finished")

    def _reset_recording_state(self):
        self._audio_buffer = np.array([], dtype=np.float32)
        self._is_speaking = False
        self._silence_start = None
        self._barge_in_above_samples = 0
        self._strong_barge_in_samples = 0
        self._barge_detector.reset()
        self._barge_in_active = False
        self._last_partial_time = 0.0

    def start(self):
        if self.is_recording:
            return
        self.is_recording = True
        self._stop_event.clear()
        self._use_parec = False
        self._parec_process = None

        # Reset Silero GRU state so prior audio streams (e.g., earlier tests
        # or previous conversation turns) do not suppress voiced probability
        # at the start of this new recording session.
        if self._silero_vad is not None:
            self._silero_vad.reset_states()

        self._worker_thread = threading.Thread(
            target=self._process_audio, daemon=True
        )
        self._worker_thread.start()

        try:
            self._stream = sd.InputStream(
                device=self.device,
                channels=1,
                samplerate=self.device_sample_rate,
                dtype=np.float32,
                callback=self._audio_callback,
                blocksize=1024,
            )
            self._stream.start()
            print(
                f"Audio recording started at {self.device_sample_rate}Hz (sounddevice)"
            )
        except Exception:
            print("sounddevice failed, using parec (PipeWire)...")
            self._use_parec = True
            self._start_parec()

    def _start_parec(self):
        self.device_sample_rate = 16000
        self.needs_resampling = False
        self._parec_process = subprocess.Popen(
            ["parec", "--rate=16000", "--channels=1", "--format=float32le"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._parec_thread = threading.Thread(
            target=self._read_parec, daemon=True
        )
        self._parec_thread.start()
        print("Audio recording started at 16000Hz (parec/PipeWire)")

    def _read_parec(self):
        chunk_size = 1024 * 4
        while (
            not self._stop_event.is_set()
            and self._parec_process.poll() is None
        ):
            data = self._parec_process.stdout.read(chunk_size)
            if data:
                audio = np.frombuffer(data, dtype=np.float32)
                self._audio_queue.put(audio)

    def stop(self):
        if not self.is_recording:
            return
        self.is_recording = False
        self._stop_event.set()
        if hasattr(self, "_stream") and self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        if hasattr(self, "_parec_process") and self._parec_process:
            try:
                self._parec_process.terminate()
                self._parec_process.wait(timeout=1)
            except Exception:
                pass
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        try:
            self._wakeword_service.stop()
        except Exception:
            pass
        print("\nAudio recording stopped")

    def set_assistant_speaking(self, speaking: bool):
        if speaking:
            self._set_listener_state(
                ListenerState.ASSISTANT_SPEAKING, reason="tts_start"
            )
            self.assistant_is_speaking = True
            self._interrupt_triggered = False
            self._barge_in_active = False
            self._barge_in_above_samples = 0
            self._strong_barge_in_samples = 0
            self._barge_detector.reset()
            self._barge_ambient_ema = None
            self._barge_ambient_chunks = 0
            self._barge_rms_ring.clear()
            self._ambient_bootstrap_evaluated = False
            self._tts_start_time = time.time()
            self._is_speaking = False
            self._silence_start = None
            self._audio_buffer = np.array([], dtype=np.float32)
            print(
                f"\nSpeaking... (interrupt by talking, threshold > "
                f"{self.vad_threshold:.3f})"
            )
            self._diag_log("assistant_speaking_start", vad_threshold=self.vad_threshold)
        else:
            self.assistant_is_speaking = False
            self._interrupt_triggered = False
            self._barge_in_above_samples = 0
            self._strong_barge_in_samples = 0
            self._barge_detector.reset()
            self._tts_start_time = 0.0
            self.clear_echo_reference()
            if not self._barge_in_active:
                self._is_speaking = False
                self._silence_start = None
                self._audio_buffer = np.array([], dtype=np.float32)
            self._recover_until = time.time() + 0.25
            self._set_listener_state(ListenerState.RECOVER, reason="tts_end")
            print("\nReady for input...")
            self._diag_log("assistant_speaking_end")

    def listener_state(self) -> str:
        return self._listener_state.value

    def pause(self):
        self.set_assistant_speaking(True)

    def resume(self):
        self.set_assistant_speaking(False)

    def is_barge_in_active(self) -> bool:
        return self._barge_in_active


# Backends that write PCM WAV (not MP3).
_WAV_TTS_BACKENDS = frozenset({"supertonic", "kokoro", "piper", "melotts"})


def _tts_file_suffix(backend: Optional[str]) -> str:
    return ".wav" if backend in _WAV_TTS_BACKENDS else ".mp3"


def _resolve_playback_device_label(output_device: Optional[int]) -> str:
    if output_device is None:
        try:
            idx = sd.default.device[1]
            info = sd.query_devices(idx)
            return f"default:{info.get('name', idx)}"
        except Exception:
            return "default"
    try:
        info = sd.query_devices(output_device)
        return f"{output_device}:{info.get('name', '?')}"
    except Exception as e:
        return f"{output_device}:<{e}>"


def resolve_output_device(
    input_device: Optional[int],
    output_device: Optional[int],
) -> Optional[int]:
    """Pick a playback device that matches the microphone when unset.

    On many laptops the PortAudio default output is an HDMI/monitor sink while the
    built-in mic uses the analog codec on another index — TTS then plays to a
    display the user is not listening to.
    """
    if output_device is not None:
        return output_device

    try:
        if input_device is not None:
            info = sd.query_devices(input_device)
            if info.get("max_output_channels", 0) > 0:
                return input_device
            hostapi = info.get("hostapi")
            for i, dev in enumerate(sd.query_devices()):
                if dev.get("max_output_channels", 0) <= 0:
                    continue
                if dev.get("hostapi") != hostapi:
                    continue
                name = (dev.get("name") or "").lower()
                if "analog" in name or "speaker" in name or "headphone" in name:
                    return i
            return input_device

        in_idx = sd.default.device[0]
        out_idx = sd.default.device[1]
        if in_idx is None or out_idx is None or in_idx == out_idx:
            return None
        in_info = sd.query_devices(in_idx)
        out_info = sd.query_devices(out_idx)
        in_name = (in_info.get("name") or "").lower()
        out_name = (out_info.get("name") or "").lower()
        if not any(k in in_name for k in ("analog", "pch", "built-in", "internal")):
            return None
        if not any(k in out_name for k in ("hdmi", "nvidia", "display", "monitor", "benq")):
            return None
        hostapi = in_info.get("hostapi")
        for i, dev in enumerate(sd.query_devices()):
            if dev.get("max_output_channels", 0) <= 0:
                continue
            if dev.get("hostapi") != hostapi:
                continue
            name = (dev.get("name") or "").lower()
            if "analog" in name or "speaker" in name:
                return i
        if in_info.get("max_output_channels", 0) > 0:
            return in_idx
    except Exception:
        pass
    return None


def _resolve_pygame_device_name(output_device: Optional[int]) -> Optional[str]:
    """Map a sounddevice output index to an SDL playback device name."""
    try:
        from pygame import _sdl2 as sdl2
    except ImportError:
        return None
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        names = list(sdl2.audio.get_audio_device_names(False))
    except Exception:
        return None
    if not names:
        return None
    if output_device is None:
        return names[0]

    try:
        info = sd.query_devices(output_device)
        raw = (info.get("name") or "").lower()
    except Exception:
        return None

    tokens = []
    for key in ("analog", "built-in", "speaker", "headphone", "pch", "alc"):
        if key in raw:
            tokens.append(key)
    for sdl_name in names:
        low = sdl_name.lower()
        if tokens and any(t in low for t in tokens):
            return sdl_name
    if "hdmi" in raw:
        for sdl_name in names:
            if "hdmi" in sdl_name.lower():
                return sdl_name
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  AudioPlayer
# ═══════════════════════════════════════════════════════════════════════════════

class AudioPlayer:
    """
    Cross-platform audio player with multiple TTS backends.

    Priority (local-first):
        Kokoro (ONNX, local) > Supertonic (local) > edge-tts (online) > gTTS

    Supports:
    - Sentence-chunked playback (lower latency, natural interruption points)
    - Interruption for barge-in
    - Echo reference callback for AEC
    """

    EDGE_VOICES = {
        "en-US": "en-US-AriaNeural",
        "en-US-male": "en-US-GuyNeural",
        "en-GB": "en-GB-SoniaNeural",
        "en-AU": "en-AU-NatashaNeural",
    }

    SUPERTONIC_VOICES = {
        "en-US": "M1",
        "en-US-male": "M1",
        "en-US-female": "F1",
    }

    KOKORO_VOICES = {
        "en-US": "af_heart",
        "en-US-male": "am_adam",
        "en-US-female": "af_heart",
        "en-GB": "bf_emma",
        "en-GB-male": "bm_george",
    }

    # Piper: locale → ONNX voice model stem (downloaded on first use)
    # Full list: https://huggingface.co/rhasspy/piper-voices
    PIPER_VOICES = {
        "en-US":    "en_US-lessac-medium",
        "en-GB":    "en_GB-alan-medium",
        "de-DE":    "de_DE-thorsten-medium",
        "es-ES":    "es_ES-mls-medium",
        "fr-FR":    "fr_FR-mls-medium",
        "it-IT":    "it_IT-riccardo-x_low",
        "pt-BR":    "pt_BR-faber-medium",
        "zh-CN":    "zh_CN-huayan-medium",
        "ja-JP":    "ja_JP-kokoro-medium",
        "ru-RU":    "ru_RU-irina-medium",
        "nl-NL":    "nl_NL-mls-medium",
        "pl-PL":    "pl_PL-mls-medium",
    }

    # MeloTTS: locale → (language_code, speaker_id)
    MELOTTS_VOICES = {
        "en-US":    ("EN", "EN-US"),
        "en-BR":    ("EN", "EN-BR"),
        "en-IN":    ("EN", "EN-India"),
        "en-AU":    ("EN", "EN-AU"),
        "en-GB":    ("EN", "EN-Default"),
        "zh-CN":    ("ZH", "ZH"),
        "ja-JP":    ("JP", "JP"),
        "ko-KR":    ("KR", "KR"),
        "es-ES":    ("ES", "ES"),
        "fr-FR":    ("FR", "FR"),
    }

    def __init__(
        self,
        output_device: Optional[int] = None,
        voice: str = "en-US",
        prefer_local: bool = True,
        tts_backend: Optional[str] = None,
        tts_model: Optional[str] = None,
        playback_backend: str = "auto",
    ):
        pb = (playback_backend or "auto").lower()
        if pb not in ("auto", "sounddevice", "pygame"):
            print(f"Warning: unknown playback_backend '{playback_backend}', using auto")
            pb = "auto"
        if not PYGAME_AVAILABLE and pb == "pygame":
            raise RuntimeError(
                "pygame is required for playback_backend=pygame. "
                "Install with: pip install pygame"
            )

        self.output_device = output_device
        self.playback_backend = pb
        self._playback_engine: Optional[str] = None
        self._is_playing = False
        self._current_file = None
        self.voice = voice
        self._supertonic_tts = None
        self._supertonic_style = None
        self._kokoro_engine = None
        self._kokoro_voice = None
        self._piper_voice_obj = None
        self._melo_engine = None
        self._melo_speaker_id = None
        self._melo_language = None
        # tts_model allows the caller to override the default voice/model per backend
        self._tts_model = tts_model

        # ── Backend selection ────────────────────────────────────────────
        if tts_backend:
            # Explicit backend requested
            self.tts_backend = tts_backend
            self._init_explicit_backend(tts_backend, voice)
        elif prefer_local:
            self._init_local_first(voice)
        else:
            self._init_online_first(voice)

        self._init_playback()

    def _init_playback(self):
        """Initialize playback (sounddevice and/or pygame) on the chosen output device."""
        label = _resolve_playback_device_label(self.output_device)
        if self.playback_backend in ("auto", "sounddevice"):
            self._playback_engine = "sounddevice"
        if self.playback_backend == "pygame" and PYGAME_AVAILABLE:
            devname = _resolve_pygame_device_name(self.output_device)
            try:
                if devname:
                    pygame.mixer.pre_init(44100, -16, 2, devicename=devname)
                pygame.mixer.init(
                    frequency=44100, size=-16, channels=2, devicename=devname
                )
                self._playback_engine = "pygame"
            except Exception as e:
                print(
                    f"Warning: pygame mixer init failed ({e}); "
                    "falling back to sounddevice for playback."
                )
                try:
                    pygame.mixer.quit()
                except Exception:
                    pass
                self._playback_engine = "sounddevice"
        if self._playback_engine:
            print(f"TTS playback: {self._playback_engine} ({label})")
        else:
            print("Warning: no TTS playback engine available")

    def _init_explicit_backend(self, backend: str, voice: str):
        """Initialize an explicitly requested backend."""
        if backend == "kokoro" and KOKORO_AVAILABLE:
            self._init_kokoro(voice)
        elif backend == "supertonic" and SUPERTONIC_AVAILABLE:
            self._init_supertonic(voice)
        elif backend == "piper":
            self._init_piper(voice)
        elif backend == "melotts":
            self._init_melotts(voice)
        else:
            print(
                f"Warning: Requested TTS backend '{backend}' not available "
                "or not allowed in local-only mode; auto-selecting..."
            )
            self._init_local_first(voice)

    def _init_local_first(self, voice: str):
        """Auto-select best local backend."""
        if KOKORO_AVAILABLE:
            self._init_kokoro(voice)
            if self.tts_backend == "kokoro":
                return
        if SUPERTONIC_AVAILABLE:
            self.tts_backend = "supertonic"
            self._init_supertonic(voice)
            if self.tts_backend == "supertonic":
                return
        self.tts_backend = None
        print("Warning: No local open-source TTS backend available!")

    def _init_online_first(self, voice: str):
        """Deprecated path: local-only mode routes to local-first."""
        self._init_local_first(voice)

    def _init_kokoro(self, voice: str):
        """Initialize Kokoro TTS (ONNX, ~82M params, Apache-2.0)."""
        try:
            from huggingface_hub import hf_hub_download

            print("TTS: Downloading Kokoro ONNX model...")
            model_path = hf_hub_download(
                "fastrtc/kokoro-onnx", "kokoro-v1.0.onnx"
            )
            voices_path = hf_hub_download(
                "fastrtc/kokoro-onnx", "voices-v1.0.bin"
            )
            self._kokoro_engine = _KokoroEngine(model_path, voices_path)
            self._kokoro_voice = self.KOKORO_VOICES.get(voice, "af_heart")
            self.tts_backend = "kokoro"
            print(
                f"TTS: Kokoro ONNX (voice: {self._kokoro_voice}, "
                "~80MB, local)"
            )
        except Exception as e:
            print(f"Warning: Kokoro TTS init failed: {e}")
            self.tts_backend = None  # will be picked up by fallback

    def _init_supertonic(self, voice: str = "en-US"):
        """Initialize Supertonic TTS engine."""
        try:
            print("Initializing Supertonic TTS (fast, on-device)...")
            self._supertonic_tts = SupertonicTTS(auto_download=True)
            voice_name = self.SUPERTONIC_VOICES.get(voice, "M1")
            self._supertonic_style = self._supertonic_tts.get_voice_style(
                voice_name=voice_name
            )
            self.tts_backend = "supertonic"
            print(f"TTS: Supertonic (voice: {voice_name}, ~10x realtime)")
        except Exception as e:
            print(f"Warning: Supertonic init failed: {e}")
            self.tts_backend = None

    def _init_piper(self, voice: str):
        """Initialize Piper TTS (ONNX, CPU-first, 30+ languages).

        Voice model files are downloaded on first use via piper's built-in
        downloader.  The model stem can be overridden with ``tts_model``.
        """
        if not PIPER_AVAILABLE:
            print(
                "Warning: piper-tts not installed "
                "(pip install piper-tts  or  pip install -r requirements.txt); "
                "falling back."
            )
            self._init_local_first(voice)
            return
        try:
            from huggingface_hub import hf_hub_download

            voice_stem = self._tts_model or self.PIPER_VOICES.get(
                voice, "en_US-lessac-medium"
            )
            # voice_stem = "{locale}-{voice_name}-{quality}" e.g. en_US-lessac-medium
            quality = voice_stem.rsplit("-", 1)[-1]
            rest = voice_stem[: -(len(quality) + 1)]
            dash = rest.rfind("-")
            lang_prefix = rest[:dash]
            vname = rest[dash + 1 :]
            lang_code = lang_prefix.split("_", 1)[0]
            rel_dir = f"{lang_code}/{lang_prefix}/{vname}/{quality}"
            onnx_name = f"{voice_stem}.onnx"
            json_name = f"{voice_stem}.onnx.json"
            print(f"TTS: Downloading Piper voice '{voice_stem}' ...")
            onnx_path = hf_hub_download(
                "rhasspy/piper-voices", f"{rel_dir}/{onnx_name}"
            )
            _ = hf_hub_download("rhasspy/piper-voices", f"{rel_dir}/{json_name}")
            self._piper_voice_obj = _PiperVoice.load(onnx_path)
            self.tts_backend = "piper"
            print(f"TTS: Piper ONNX (voice: {voice_stem}, ~80MB, 30+ languages)")
        except Exception as e:
            print(f"Warning: Piper TTS init failed: {e}")
            self.tts_backend = None
            self._init_local_first(voice)

    def _init_melotts(self, voice: str):
        """Initialize MeloTTS (VITS2 distill, 6 languages, CPU-friendly).

        Models are downloaded automatically by the melo library on first use
        (~150 MB per language).  The language/speaker can be overridden with
        ``tts_model`` as ``"EN-US"``, ``"ZH"``, ``"JP"``, etc.
        """
        if not MELOTTS_AVAILABLE:
            print("Warning: melotts not installed (pip install melotts); falling back.")
            self._init_local_first(voice)
            return
        try:
            lang_spk = self._tts_model or None
            if lang_spk:
                lang_code = lang_spk.split("-")[0].upper()
                speaker_id = lang_spk.upper()
            else:
                lang_code, speaker_id = self.MELOTTS_VOICES.get(voice, ("EN", "EN-US"))
            print(f"TTS: Loading MeloTTS ({lang_code} / {speaker_id}) ...")
            import torch as _torch
            device = "cuda" if _torch.cuda.is_available() else "cpu"
            self._melo_engine   = _MeloEngine(language=lang_code, device=device)
            self._melo_speaker_id = self._melo_engine.hps.data.spk2id[speaker_id]
            self._melo_language  = lang_code
            self.tts_backend = "melotts"
            print(f"TTS: MeloTTS ({lang_code}/{speaker_id}, ~600MB, 6 languages)")
        except Exception as e:
            print(f"Warning: MeloTTS init failed: {e}")
            self.tts_backend = None
            self._init_local_first(voice)

    # ── Synthesis ─────────────────────────────────────────────────────────

    def _kokoro_synthesize(self, text: str, output_file: str):
        """Synthesize using Kokoro ONNX."""
        audio, sr = self._kokoro_engine.create(
            text, voice=self._kokoro_voice, speed=1.0, lang="en-us"
        )
        audio = audio.flatten().astype(np.float32)
        sf.write(output_file, audio, sr)

    def _supertonic_synthesize(self, text: str, output_file: str):
        audio, _ = self._supertonic_tts.synthesize(
            text, voice_style=self._supertonic_style
        )
        audio = audio.flatten()
        sf.write(output_file, audio, 44100)

    async def _edge_tts_synthesize(self, text: str, output_file: str):
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(output_file)

    def _piper_synthesize(self, text: str, output_file: str):
        """Synthesize using Piper ONNX (writes 16-kHz mono WAV)."""
        import wave
        with wave.open(output_file, "wb") as wav_file:
            self._piper_voice_obj.synthesize_wav(text, wav_file)

    def _melotts_synthesize(self, text: str, output_file: str):
        """Synthesize using MeloTTS (writes 44.1-kHz WAV via soundfile)."""
        import tempfile, os
        # melo writes to a file; use a temp path then rename to avoid partial reads
        tmp = output_file + ".melo.tmp"
        self._melo_engine.tts_to_file(
            text,
            self._melo_speaker_id,
            tmp,
            speed=1.0,
        )
        os.replace(tmp, output_file)

    def _synthesize_speech(self, text: str, output_file: str):
        t0 = time.perf_counter()
        preview = (text or "").strip()
        tts_debug.log_audio(
            "synth_start",
            backend=self.tts_backend,
            text_len=len(preview),
            text_preview=preview[:60] if preview else "",
            console_kind="synth_start",
            console_detail=f"({self.tts_backend})",
        )
        try:
            self._synthesize_speech_inner(text, output_file)
        except Exception as e:
            tts_debug.log_audio(
                "synth_error",
                level=logging.ERROR,
                backend=self.tts_backend,
                error=str(e),
                duration_ms=int((time.perf_counter() - t0) * 1000),
                console_kind="failed",
                console_detail=f"synthesis: {e}",
            )
            raise
        duration_ms = int((time.perf_counter() - t0) * 1000)
        samples = 0
        sr = 0
        try:
            data, sr = self._load_audio_data(output_file)
            samples = len(data)
        except Exception:
            pass
        tts_debug.log_audio(
            "synth_done",
            backend=self.tts_backend,
            duration_ms=duration_ms,
            samples=samples,
            sample_rate=sr,
            console_kind="synth_done",
            console_detail=f"{duration_ms}ms, {samples} samples",
        )

    def _synthesize_speech_inner(self, text: str, output_file: str):
        if self.tts_backend == "kokoro":
            self._kokoro_synthesize(text, output_file)
        elif self.tts_backend == "supertonic":
            self._supertonic_synthesize(text, output_file)
        elif self.tts_backend == "piper":
            self._piper_synthesize(text, output_file)
        elif self.tts_backend == "melotts":
            self._melotts_synthesize(text, output_file)
        elif self.tts_backend == "edge-tts":
            asyncio.run(self._edge_tts_synthesize(text, output_file))
        elif self.tts_backend == "gtts":
            tts = gTTS(text=text, lang="en")
            tts.save(output_file)
        else:
            raise RuntimeError("No TTS backend available")

    # ── Playback ──────────────────────────────────────────────────────────

    def speak(
        self,
        text: str,
        on_start: Callable = None,
        on_end: Callable = None,
        chunked: bool = False,
    ) -> bool:
        """
        Synthesize and play *text*.

        Args:
            text: Text to speak.
            on_start: Called when playback starts ``(audio_data, sample_rate)``.
            on_end: Called when playback ends.
            chunked: If True, split into sentences and play each chunk
                     separately (lower first-audio latency, natural pause
                     points for interruption).

        Returns:
            True if completed, False if interrupted.
        """
        if self.tts_backend is None:
            print(f"[TTS not available] {text}")
            tts_debug.log_tts("speak_skipped", reason="no_backend")
            return True

        if chunked:
            return self._speak_chunked(text, on_start, on_end)
        return self._speak_full(text, on_start, on_end)

    def prepare_speech_file(self, text: str) -> str:
        """Synthesize *text* to a temp path; caller must delete after playback."""
        if self.tts_backend is None:
            raise RuntimeError("No TTS backend available")
        fd, path = tempfile.mkstemp(suffix=_tts_file_suffix(self.tts_backend))
        os.close(fd)
        self._synthesize_speech(text, path)
        return path

    def _play_audio_blocking(self, path: str) -> bool:
        """Play *path* until done or ``stop()``; returns True if playback completed."""
        audio_data, sample_rate = self._load_audio_data(path)
        duration_sec = len(audio_data) / max(sample_rate, 1)
        rms = float(np.sqrt(np.mean(audio_data**2))) if len(audio_data) else 0.0
        if rms < 1e-5:
            print(
                f"⚠ TTS output appears silent (rms={rms:.6f}, backend={self.tts_backend})"
            )
            tts_debug.log_audio(
                "playback_silent_wav",
                rms=rms,
                backend=self.tts_backend,
                console_kind="failed",
                console_detail=f"silent wav rms={rms:.6f}",
            )
            return True

        device_label = _resolve_playback_device_label(self.output_device)
        snippet = os.path.basename(path)
        engine = self._choose_playback_engine()
        print(
            f"🔊 Playing ({self.tts_backend}, {engine}, {duration_sec:.2f}s, "
            f"{device_label}): {snippet}"
        )
        tts_debug.log_audio(
            "playback_start",
            backend=self.tts_backend,
            engine=engine,
            device=self.output_device,
            device_label=device_label,
            sample_rate=sample_rate,
            samples=len(audio_data),
            duration_sec=round(duration_sec, 3),
            rms=round(rms, 6),
            path=snippet,
            console_kind="playing",
            console_detail=f"({len(audio_data)} samples, {engine})",
        )

        self._is_playing = True
        t0 = time.perf_counter()
        try:
            if engine == "sounddevice":
                ok = self._play_with_sounddevice(audio_data, sample_rate, duration_sec)
            else:
                ok = self._play_with_pygame(path, duration_sec)
        except Exception as e:
            tts_debug.log_audio(
                "playback_error",
                level=logging.ERROR,
                engine=engine,
                error=str(e),
                duration_ms=int((time.perf_counter() - t0) * 1000),
                console_kind="failed",
                console_detail=str(e),
            )
            print(f"❌ TTS playback failed ({engine}): {e}")
            if engine == "sounddevice" and PYGAME_AVAILABLE:
                try:
                    print("   Retrying with pygame...")
                    ok = self._play_with_pygame(path, duration_sec)
                    tts_debug.log_audio(
                        "playback_pygame_fallback",
                        completed=ok,
                        duration_ms=int((time.perf_counter() - t0) * 1000),
                    )
                    return ok
                except Exception as e2:
                    tts_debug.log_audio(
                        "playback_error",
                        level=logging.ERROR,
                        engine="pygame",
                        error=str(e2),
                        console_kind="failed",
                        console_detail=f"pygame fallback: {e2}",
                    )
                    print(f"❌ TTS pygame fallback failed: {e2}")
            return False
        else:
            tts_debug.log_audio(
                "playback_end",
                engine=engine,
                completed=ok,
                interrupted=not ok,
                duration_ms=int((time.perf_counter() - t0) * 1000),
                console_kind="played" if ok else "cancelled",
                console_detail=f"({engine})",
            )
            return ok
        finally:
            self._is_playing = False

    def _choose_playback_engine(self) -> str:
        if self.playback_backend == "sounddevice":
            return "sounddevice"
        if self.playback_backend == "pygame":
            return "pygame"
        # auto: prefer sounddevice so output_device matches mic routing (PortAudio)
        return "sounddevice"

    def _playback_sample_rate_for_device(
        self, preferred_sr: int
    ) -> int:
        """Pick a sample rate the output device accepts (Kokoro often emits 24 kHz)."""
        candidates = [preferred_sr]
        try:
            if self.output_device is not None:
                dev_sr = int(
                    sd.query_devices(self.output_device).get("default_samplerate", 0)
                )
            else:
                dev_sr = int(sd.query_devices(sd.default.device[1]).get("default_samplerate", 0))
            if dev_sr > 0:
                candidates.append(dev_sr)
        except Exception:
            pass
        for sr in (48000, 44100, 32000, 24000, 22050, 16000):
            if sr not in candidates:
                candidates.append(sr)
        for sr in candidates:
            try:
                sd.check_output_settings(device=self.output_device, samplerate=sr)
                if sr != preferred_sr:
                    print(
                        f"   Resampling TTS {preferred_sr} Hz → {sr} Hz for output device"
                    )
                return sr
            except Exception:
                continue
        return preferred_sr

    def _play_with_sounddevice(
        self, audio_data: np.ndarray, sample_rate: int, duration_sec: float
    ) -> bool:
        play_sr = self._playback_sample_rate_for_device(sample_rate)
        if play_sr != sample_rate:
            if not LIBROSA_AVAILABLE:
                raise RuntimeError(
                    f"output device does not accept {sample_rate} Hz and librosa "
                    "is not installed for resampling"
                )
            audio_data = librosa.resample(
                audio_data, orig_sr=sample_rate, target_sr=play_sr
            )
            sample_rate = play_sr
            duration_sec = len(audio_data) / sample_rate

        sd.play(audio_data, sample_rate, device=self.output_device)
        deadline = time.perf_counter() + duration_sec + 0.35
        stream = sd.get_stream()
        while self._is_playing and time.perf_counter() < deadline:
            if stream is None or not stream.active:
                break
            time.sleep(0.05)
        if not self._is_playing:
            sd.stop()
            return False
        return True

    def _play_with_pygame(self, path: str, duration_sec: float) -> bool:
        if not PYGAME_AVAILABLE:
            raise RuntimeError("pygame not available")
        audio_data, sample_rate = self._load_audio_data(path)
        channels = 1 if audio_data.ndim == 1 else min(2, audio_data.shape[1])
        devname = _resolve_pygame_device_name(self.output_device)
        try:
            pygame.mixer.quit()
        except Exception:
            pass
        pygame.mixer.init(
            frequency=int(sample_rate),
            size=-16,
            channels=channels,
            devicename=devname,
        )
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        deadline = time.perf_counter() + duration_sec + 0.35
        while self._is_playing and time.perf_counter() < deadline:
            if not pygame.mixer.music.get_busy():
                break
            time.sleep(0.05)
        if not self._is_playing:
            pygame.mixer.music.stop()
            return False
        return not pygame.mixer.music.get_busy()

    def play_prepared_file(self, path: str, on_start=None, on_end=None) -> bool:
        """Play audio from *path* then delete it (same semantics as ``_speak_full``)."""
        completed = False
        try:
            self._current_file = path
            if on_start:
                try:
                    audio_data, sample_rate = self._load_audio_data(path)
                except Exception:
                    audio_data, sample_rate = None, None
                try:
                    on_start(audio_data, sample_rate)
                except TypeError:
                    on_start()
            completed = self._play_audio_blocking(path)
        except Exception as e:
            print(f"❌ Error in play_prepared_file: {e}")
            completed = False
        finally:
            if on_end:
                on_end()
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
            self._current_file = None
        return completed

    def _speak_full(self, text: str, on_start, on_end) -> bool:
        """Synthesize the full text at once, then play."""
        if self.tts_backend is None:
            print(f"[TTS not available] {text}")
            return True
        try:
            path = self.prepare_speech_file(text)
        except Exception as e:
            print(f"Error in speak: {e}")
            return False
        return self.play_prepared_file(path, on_start, on_end)

    def _speak_chunked(self, text: str, on_start, on_end) -> bool:
        """
        Split *text* into sentences, synthesize and play each one.

        Benefits:
        - First audio plays much sooner (only wait for one sentence)
        - Natural pause points between sentences for clean interruption
        - Less memory for long responses
        """
        sentences = split_sentences(text)
        completed = True
        first_chunk = True

        for sentence in sentences:
            if not self._is_playing and not first_chunk:
                # Was interrupted between chunks
                completed = False
                break

            chunk_file = None
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=_tts_file_suffix(self.tts_backend)
                ) as fp:
                    chunk_file = fp.name

                self._synthesize_speech(sentence, chunk_file)

                # Provide echo reference for this chunk
                if on_start:
                    try:
                        audio_data, sample_rate = self._load_audio_data(
                            chunk_file
                        )
                    except Exception:
                        audio_data, sample_rate = None, None
                    try:
                        on_start(audio_data, sample_rate)
                    except TypeError:
                        on_start()

                first_chunk = False
                chunk_completed = self._play_audio_blocking(chunk_file)
            except Exception as e:
                print(f"❌ Error in chunked speak: {e}")
                chunk_completed = False
            finally:
                if chunk_file and os.path.exists(chunk_file):
                    try:
                        os.remove(chunk_file)
                    except Exception:
                        pass

            if not chunk_completed:
                completed = False
                break

        self._is_playing = False
        if on_end:
            on_end()
        return completed

    def stop(self):
        if self._is_playing:
            self._is_playing = False
            try:
                sd.stop()
            except Exception:
                pass
            if PYGAME_AVAILABLE:
                try:
                    pygame.mixer.music.stop()
                except Exception:
                    pass
            print("--- Speech interrupted ---")

    def is_playing(self) -> bool:
        if not self._is_playing:
            return False
        try:
            stream = sd.get_stream()
            if stream is not None and stream.active:
                return True
        except Exception:
            pass
        if PYGAME_AVAILABLE:
            try:
                return pygame.mixer.music.get_busy()
            except Exception:
                pass
        return self._is_playing

    def cleanup(self):
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.quit()
            except Exception:
                pass

    def _load_audio_data(self, file_path: str):
        """Load audio data for echo cancellation reference."""
        if file_path.endswith(".wav"):
            data, sr = sf.read(file_path, dtype="float32")
        else:
            data, sr = librosa.load(file_path, sr=None)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        return data.astype(np.float32), int(sr)


# ═══════════════════════════════════════════════════════════════════════════════
#  Quick test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing cross-platform audio module...")
    list_audio_devices()

    def on_speech(audio):
        print(f"\n[Got {len(audio) / 16000:.1f}s of audio]")

    recorder = AudioRecorder(callback=on_speech)
    recorder.start()

    try:
        input("Press Enter to stop...\n")
    except KeyboardInterrupt:
        pass

    recorder.stop()
