"""Acoustic Echo Cancellation (AEC) front-end for the capture hot path.

On open speakers with no AEC the assistant's own TTS leaks into the mic and looks
exactly like a user barging in -- it self-interrupts (see
``docs/session_2026-05-31_acoustic_real_voice.md``; today only a 6 dB output-margin
guard + the speaker-ID gate hold it back). AEC is the production-standard fix: it
takes the mic (near-end) block AND the audio we are PLAYING (far-end reference),
models the loudspeaker->mic echo path, and subtracts the echo BEFORE the recognizer,
the VAD, the speaker embedder, and the barge-in gate ever see it. Placed right after
the resampler and before the denoiser (so the canceller sees the raw linear echo and
the denoiser then mops up the residual + ambient noise).

This module ships the **dependency-free NumPy backend** (a frequency-domain block
adaptive filter, FDAF / "fast LMS"): zero new packages, cross-platform, real-time.
A deep-learning tier (DTLN-aec) can drop in behind the same :class:`EchoCanceller`
seam later -- it needs a tflite->ONNX conversion, so it is deferred; ``build_aec``
fails open to NO AEC for ``aec_backend='dtln'`` until that lands.

Three pieces, all unit-testable with no audio device:

* :class:`FarEndRing` -- a small thread-safe ring of recently-PLAYED 16 kHz samples.
  The playback thread ``push``es; the capture thread ``read``s the window that was
  played ``delay`` ago (the echo reference for the current near-end block).
* :class:`_FDAFAdaptiveFilter` -- the linear adaptive canceller (overlap-save,
  constrained, normalized) with a double-talk freeze so the filter doesn't diverge
  onto the user's voice when both talk at once.
* :class:`EchoCanceller` -- the seam the engine holds: ``process_16k(near, far)``,
  **passthrough-on-error** (any failure returns the near-end unchanged so it can
  never crash the capture daemon), and ``reset()``.

Limits (honest): a linear filter cannot remove loudspeaker nonlinearity, so real
open-speaker ERLE lands ~10-20 dB (not the 20-30 dB lab figure) -- good for
headset / near-field / moderate rooms (lets ``barge_in_output_margin_db`` relax
6 -> ~3 dB), insufficient for loud cheap speakers (that's the DTLN tier's job).
"""
from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .sherpa import SherpaConfig

log = logging.getLogger("speaker.sherpa")


def _next_pow2(n: int, *, minimum: int = 256) -> int:
    """Smallest power of two >= ``n`` (>= ``minimum``). FFT sizes want pow2."""
    n = max(int(n), minimum)
    return 1 << (n - 1).bit_length()


class FarEndRing:
    """Thread-safe ring of recently-played far-end (16 kHz mono) samples.

    Single-producer (playback thread ``push``) / single-consumer (capture thread
    ``read``); a short lock guards each numpy copy (microseconds -- the capture
    thread never blocks meaningfully). ``read(n, delay)`` returns the ``n`` samples
    that were played ``delay`` samples before the current write head -- the echo
    reference time-aligned to a near-end block. Positions not yet played or already
    evicted come back as zeros (so the canceller is a no-op there)."""

    def __init__(self, capacity: int = 32000):  # ~2 s at 16 kHz
        self._cap = int(capacity)
        self._buf = np.zeros(self._cap, dtype=np.float32)
        self._written = 0  # total samples ever pushed (monotonic)
        self._lock = threading.Lock()

    def push(self, samples) -> None:
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        n = x.shape[0]
        if n == 0:
            return
        with self._lock:
            if n >= self._cap:
                self._buf[:] = x[-self._cap :]
                self._written += n
                return
            start = self._written % self._cap
            end = start + n
            if end <= self._cap:
                self._buf[start:end] = x
            else:
                k = self._cap - start
                self._buf[start:] = x[:k]
                self._buf[: end - self._cap] = x[k:]
            self._written += n

    def read(self, n: int, delay: int) -> np.ndarray:
        n = int(n)
        delay = max(0, int(delay))
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        out = np.zeros(n, dtype=np.float32)
        with self._lock:
            written = self._written
            hi = written - delay      # newest reference sample for "now"
            lo = hi - n
            if hi <= 0:
                return out            # nothing played within the delay yet
            avail_lo = max(0, written - self._cap)
            idx = np.arange(lo, hi)
            valid = (idx >= avail_lo) & (idx < written)
            if valid.any():
                out[valid] = self._buf[idx[valid] % self._cap]
        return out

    def clear(self) -> None:
        with self._lock:
            self._buf[:] = 0.0
            self._written = 0


class _FDAFAdaptiveFilter:
    """Frequency-domain block adaptive echo canceller (constrained, normalized).

    Processes fixed-size frames of ``F`` samples (the modeled echo tail) via 50 %
    overlap-save: ``W`` is the per-bin frequency-domain filter, updated by the
    normalized gradient with a causal (gradient) constraint. Inputs of any/variable
    length are buffered into ``F``-frames, so the output length can differ from the
    input -- every consumer (recognizer, VAD, embedder) accepts that, exactly like
    the resampler/denoiser. Pure NumPy; real-time.

    Double-talk: a linear filter will diverge if it adapts while the user talks over
    the assistant (it tries to model the user's voice as echo). After a short warmup
    we FREEZE adaptation (keep filtering, stop learning) when the near-end clearly
    exceeds the predicted echo -- a near-vs-echo energy test in the consistent mic
    scale. With nothing playing (silent far-end) we also don't adapt."""

    def __init__(
        self,
        frame: int = 512,
        *,
        mu: float = 0.5,
        leak: float = 1.0,
        power_smooth: float = 0.9,
        eps: float = 1e-6,
        doubletalk_freeze: bool = True,
        dt_factor: float = 2.5,
        warmup_frames: int = 8,
    ) -> None:
        self.F = int(frame)
        self.N = 2 * self.F
        self.mu = float(mu)
        self.leak = float(leak)
        self.lam = float(power_smooth)
        self.eps = float(eps)
        self.dt_freeze = bool(doubletalk_freeze)
        self.dt_factor = float(dt_factor)
        self.warmup = int(warmup_frames)
        self._reset_state()

    def _reset_state(self) -> None:
        self.W = np.zeros(self.N, dtype=np.complex128)
        self.x_prev = np.zeros(self.F, dtype=np.float64)
        self.P = np.full(self.N, self.eps, dtype=np.float64)
        self._nbuf = np.zeros(0, dtype=np.float64)
        self._fbuf = np.zeros(0, dtype=np.float64)
        self.count = 0

    def reset(self) -> None:
        self._reset_state()

    def process(self, near, far) -> np.ndarray:
        near = np.asarray(near, dtype=np.float64).reshape(-1)
        far = np.asarray(far, dtype=np.float64).reshape(-1)
        # Pair the two streams sample-for-sample; the ring may hand back a slightly
        # different far count, so match it to the near block.
        if far.shape[0] < near.shape[0]:
            far = np.concatenate([far, np.zeros(near.shape[0] - far.shape[0])])
        elif far.shape[0] > near.shape[0]:
            far = far[: near.shape[0]]
        self._nbuf = np.concatenate([self._nbuf, near])
        self._fbuf = np.concatenate([self._fbuf, far])
        chunks = []
        F = self.F
        while self._nbuf.shape[0] >= F:
            d = self._nbuf[:F]
            x = self._fbuf[:F]
            self._nbuf = self._nbuf[F:]
            self._fbuf = self._fbuf[F:]
            chunks.append(self._process_frame(d, x))
        if chunks:
            return np.concatenate(chunks).astype(np.float32)
        return np.zeros(0, dtype=np.float32)

    def _process_frame(self, d: np.ndarray, x: np.ndarray) -> np.ndarray:
        F = self.F
        xcat = np.concatenate([self.x_prev, x])      # length N = 2F
        X = np.fft.fft(xcat)
        self.P = self.lam * self.P + (1.0 - self.lam) * (np.abs(X) ** 2)
        y = np.fft.ifft(X * self.W).real
        yhat = y[F:]                                  # overlap-save valid output
        e = d - yhat                                  # cancelled near-end

        adapt = True
        if float(np.dot(x, x)) < self.eps:
            adapt = False                             # nothing playing -> no echo
        elif self.dt_freeze and self.count >= self.warmup:
            d_rms = float(np.sqrt(np.mean(d * d) + self.eps))
            yhat_rms = float(np.sqrt(np.mean(yhat * yhat) + self.eps))
            if d_rms > self.dt_factor * yhat_rms:
                adapt = False                         # near-end speech -> freeze
        if adapt:
            E = np.fft.fft(np.concatenate([np.zeros(F), e]))
            dW = (self.mu / (self.P + self.eps)) * np.conj(X) * E
            phi = np.fft.ifft(dW).real
            phi[F:] = 0.0                             # gradient (causality) constraint
            self.W = self.leak * self.W + np.fft.fft(phi)

        self.x_prev = x.copy()
        self.count += 1
        return e


class EchoCanceller:
    """The seam the engine holds. ``impl`` is duck-typed (the NumPy FDAF now, an
    ONNX DTLN session later): anything with ``process(near, far) -> samples`` and
    an optional ``reset()``. ``process_16k`` is **passthrough-on-error** so a
    transient failure can never crash the capture daemon thread."""

    def __init__(self, impl, *, sample_rate: int = 16000):
        self._impl = impl
        self.sample_rate = int(sample_rate)

    def process_16k(self, near, far):
        """Echo-cancel one mono float32 16 kHz near-end block against the aligned
        far-end block. Returns the cancelled near-end (length may differ -- the
        adaptive filter frames internally). On ANY error returns ``near`` unchanged."""
        try:
            return self._impl.process(near, far)
        except Exception:  # noqa: BLE001 - never let AEC kill the capture loop
            log.debug("AEC block failed; passing the raw near-end through", exc_info=True)
            return near

    def reset(self) -> None:
        """Clear adaptive/LSTM state (call on barge-in/idle + ASR decode-recovery)."""
        reset = getattr(self._impl, "reset", None)
        if callable(reset):
            try:
                reset()
            except Exception:  # noqa: BLE001 - reset is best-effort
                pass


def build_aec(c: "SherpaConfig") -> Optional[EchoCanceller]:
    """Echo canceller, or ``None`` when disabled/unbuildable (path then byte-
    identical to no-AEC). Mirrors :func:`build_denoiser`: returns ``None`` unless
    ``aec_enabled``; fails OPEN (logs a warning, returns ``None``) rather than
    crashing ``start()``.

    ``aec_backend='nlms'`` (default) builds the dependency-free NumPy FDAF.
    ``aec_backend='dtln'`` is reserved for the ONNX deep tier (needs a tflite->ONNX
    conversion) and currently fails open to no-AEC -- use ``'nlms'`` today."""
    if not getattr(c, "aec_enabled", False):
        return None
    backend = str(getattr(c, "aec_backend", "nlms") or "nlms").lower()
    sr = int(getattr(c, "sample_rate", 16000))
    if backend in ("nlms", "fdaf", "numpy"):
        frame = _next_pow2(int(getattr(c, "aec_filter_taps", 512) or 512))
        freeze = bool(getattr(c, "aec_doubletalk_freeze", True))
        log.info("AEC active: NumPy FDAF adaptive filter (frame=%d, doubletalk_freeze=%s)", frame, freeze)
        return EchoCanceller(_FDAFAdaptiveFilter(frame, doubletalk_freeze=freeze), sample_rate=sr)
    if backend == "dtln":
        log.warning(
            "aec_backend='dtln' (deep ONNX tier) is not implemented yet -- continuing "
            "WITHOUT AEC. Set aec_backend='nlms' for the dependency-free adaptive filter."
        )
        return None
    log.warning("unknown aec_backend=%r; continuing WITHOUT AEC", backend)
    return None
