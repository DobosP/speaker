"""Reference-coherence barge-in detector (volume-independent, zero-enrollment).

The assistant always knows EXACTLY what it is playing (its own TTS). So instead
of asking the loudness question -- "is this mic energy louder than playback?",
which self-interrupts on an open speaker and depends on absolute level -- we ask
the structural one:

    "does the mic contain sound the playback reference does NOT explain?"

That is measured by the *magnitude-squared coherence* between the time-aligned
TTS reference ``r`` and the mic ``x`` over the voiced band (300-3400 Hz):

    C(f) = |Sxr(f)|^2 / (Sxx(f) * Srr(f))   in [0, 1]

C is **scale-invariant by construction**: the playback gain (and the capture
gain) cancel in the ratio, so the *same utterance at any volume* produces the
same coherence -- exactly the "at any input I say, with the same volume, it
should work" requirement. The assistant's own TTS is, by definition, fully
explained by the reference, so its incoherent fraction is ~0 regardless of how
loud it is played: the assistant **cannot structurally self-interrupt**. A user
talking over it adds energy the reference does not predict, so coherence drops
and the *incoherent fraction* rises -- that is the barge-in signal.

Zero setup: the reference is the assistant's own output, which the engine
already produces; no enrollment, no voiceprint, no per-user data. Every other
parameter (the echo delay, the room's coherence baseline) is estimated at
runtime from the signal itself.

This is a coherence DETECTOR, not a full AEC (no real AEC library builds in this
environment). It decides *whether* user voice is present during playback -- all
barge-in needs -- it does not hand ASR an echo-free signal (ASR is never fed
while the assistant speaks anyway). Strongly nonlinear/clipping speakers or a
badly mis-aligned reference break the linear coherence model; there the caller
falls back to the legacy level gate (never worse than today). See
``docs/barge_in_coherence_2026-06-02.md``.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from typing import Optional, Sequence, Tuple

import numpy as np

try:  # scipy is a verified dependency here; guard so import never crashes start()
    from scipy.signal import coherence as _sp_coherence
    from scipy.signal import correlate as _sp_correlate
    from scipy.signal import welch as _sp_welch

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - exercised only on a broken install
    _HAVE_SCIPY = False


def _mono_f32(samples: Sequence[float]) -> np.ndarray:
    return np.asarray(samples, dtype="float32").reshape(-1)


def _rms(a: np.ndarray) -> float:
    if a.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(a.astype("float64") ** 2)))


def _resample_to(samples: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Linear resample to the mic rate -- adequate for an alignment/coherence
    estimate (we are not synthesising audio, only comparing spectra)."""
    if samples.size == 0 or src_sr <= 0 or dst_sr <= 0 or src_sr == dst_sr:
        return samples
    n_dst = int(round(samples.size * dst_sr / src_sr))
    if n_dst <= 1:
        return samples[:0]
    x_src = np.linspace(0.0, 1.0, num=samples.size, endpoint=False)
    x_dst = np.linspace(0.0, 1.0, num=n_dst, endpoint=False)
    return np.interp(x_dst, x_src, samples).astype("float32")


class EchoCoherenceDetector:
    """Scale-invariant barge-in detector backed by mic/reference coherence.

    Thread model: :meth:`note_playback` is called from the playback thread (per
    TTS chunk), :meth:`decide` from the capture thread (per mic block). The
    reference ring is guarded by a short-held lock; the diagnostic attributes
    are written only by the capture thread (single writer) and read lock-free by
    tooling, matching the existing ``_playback_level`` pattern.
    """

    def __init__(
        self,
        sample_rate: int,
        *,
        voiced_band: Tuple[float, float] = (300.0, 3400.0),
        ring_ms: float = 600.0,
        max_delay_ms: float = 400.0,
        margin_delta: float = 0.12,
        nperseg: int = 256,
        baseline_alpha_down: float = 0.25,
        baseline_alpha_up: float = 0.02,
        provisional_baseline: float = 0.5,
        min_ref_rms: float = 1e-4,
        delay_history: int = 15,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.voiced_band = (float(voiced_band[0]), float(voiced_band[1]))
        self.margin_delta = float(margin_delta)
        self.nperseg = int(nperseg)
        self.min_ref_rms = float(min_ref_rms)
        self.max_delay = max(1, int(self.sample_rate * max_delay_ms / 1000.0))
        self._ring_max = max(self.max_delay * 2, int(self.sample_rate * ring_ms / 1000.0))
        self._alpha_down = float(baseline_alpha_down)
        self._alpha_up = float(baseline_alpha_up)
        self._baseline = float(provisional_baseline)

        self._ref: "deque[np.ndarray]" = deque()
        self._ref_len = 0
        self._lock = threading.Lock()
        self._delays: "deque[int]" = deque(maxlen=int(delay_history))

        # Diagnostics (read by tools/echo_probe.py and debug logs).
        self.available = _HAVE_SCIPY
        self.last_incoherent_fraction = 0.0
        self.last_delay_ms = 0.0
        self.last_baseline = self._baseline
        self.last_decided: Optional[bool] = None

    # --- reference ingestion (playback thread) ------------------------------
    def note_playback(self, samples: Sequence[float], src_sr: int) -> None:
        """Append a played TTS chunk (at ``src_sr``) to the reference ring."""
        if not self.available:
            return
        a = _mono_f32(samples)
        if a.size == 0:
            return
        a = _resample_to(a, int(src_sr), self.sample_rate)
        if a.size == 0:
            return
        with self._lock:
            self._ref.append(a)
            self._ref_len += a.size
            while self._ref_len > self._ring_max and len(self._ref) > 1:
                self._ref_len -= self._ref.popleft().size

    def reset(self) -> None:
        """Playback stopped -> drop the reference (no echo to explain)."""
        with self._lock:
            self._ref.clear()
            self._ref_len = 0
        self._delays.clear()

    def _snapshot_ref(self) -> np.ndarray:
        with self._lock:
            if not self._ref:
                return np.zeros(0, dtype="float32")
            return np.concatenate(list(self._ref))

    # --- decision (capture thread) ------------------------------------------
    def decide(self, mic_samples: Sequence[float]) -> Optional[bool]:
        """Return True (user barge), False (echo-only) or None (can't decide ->
        the caller should fall back to the legacy level gate)."""
        if not self.available:
            return None
        x = _mono_f32(mic_samples)
        if x.size < 8:
            return None
        ref = self._snapshot_ref()
        if ref.size < x.size + self.max_delay:
            self.last_decided = None
            return None  # not enough reference history yet (session start)
        win = ref[-(x.size + self.max_delay):]
        raw_delay = self._estimate_delay(x, win)
        if raw_delay is None:
            self.last_decided = None
            return None
        # Align with the median of recent *echo-only* delays for stability (a
        # single barge frame can't drag the alignment); bootstrap with this
        # frame's estimate when the history is still empty.
        use_delay = int(np.median(self._delays)) if self._delays else raw_delay
        seg = self._segment(win, x.size, use_delay)
        if seg.size != x.size or _rms(seg) < self.min_ref_rms:
            self.last_decided = None
            return None  # nothing actually playing -> no echo reference to test

        frac = self._incoherent_fraction(x, seg)
        bar = self._baseline + self.margin_delta
        is_user = bool(frac > bar)
        if not is_user:
            # Echo-only frame: trust its delay and let the baseline track the
            # room's residual incoherence (reverb/noise floor). Asymmetric EWMA:
            # fall fast to the floor, rise slowly, so a brief dip never resets it
            # and a sustained barge (not updated here) can't inflate the bar.
            self._delays.append(raw_delay)
            alpha = self._alpha_down if frac < self._baseline else self._alpha_up
            self._baseline = (1.0 - alpha) * self._baseline + alpha * frac

        self.last_incoherent_fraction = frac
        self.last_delay_ms = 1000.0 * use_delay / self.sample_rate
        self.last_baseline = self._baseline
        self.last_decided = is_user
        return is_user

    # --- internals ----------------------------------------------------------
    def _estimate_delay(self, x: np.ndarray, win: np.ndarray) -> Optional[int]:
        """Lag (in samples, 0..max_delay) at which the mic block best matches the
        tail of the reference. Normalised so the peak is scale-invariant; |corr|
        tolerates a polarity-inverting playback path."""
        xz = x - float(x.mean())
        wz = win - float(win.mean())
        if np.linalg.norm(xz) == 0.0 or np.linalg.norm(wz) == 0.0:
            return None
        corr = _sp_correlate(wz, xz, mode="valid")  # length == max_delay + 1
        if corr.size == 0:
            return None
        k = int(np.argmax(np.abs(corr)))
        delay = (win.size - x.size) - k
        return max(0, min(self.max_delay, delay))

    @staticmethod
    def _segment(win: np.ndarray, n: int, delay: int) -> np.ndarray:
        k = (win.size - n) - delay
        k = max(0, min(win.size - n, k))
        return win[k : k + n]

    def _incoherent_fraction(self, x: np.ndarray, r: np.ndarray) -> float:
        """Energy-weighted mean of (1 - coherence) over the voiced band, in
        [0, 1]. Scale-invariant: coherence cancels any gain on x or r, and the
        weights are normalised so they cancel a gain on x too."""
        nper = min(self.nperseg, x.size)
        nover = nper // 2
        f, cxy = _sp_coherence(x, r, fs=self.sample_rate, nperseg=nper, noverlap=nover)
        _, pxx = _sp_welch(x, fs=self.sample_rate, nperseg=nper, noverlap=nover)
        lo, hi = self.voiced_band
        band = (f >= lo) & (f <= hi)
        if not np.any(band):
            return 0.0
        c = np.clip(cxy[band], 0.0, 1.0)
        w = pxx[band].astype("float64")
        wsum = float(w.sum())
        if wsum <= 0.0 or not math.isfinite(wsum):
            return float(np.mean(1.0 - c))
        w = w / wsum
        return float(np.sum(w * (1.0 - c)))
