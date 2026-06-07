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
parameter is estimated at runtime from the signal itself -- the echo *delay*
(cross-correlation, median-tracked), the room's coherence *baseline*, AND the
trigger *margin*: a barge must clear the baseline by max(floor, k * sigma) where
sigma is the learned spread of the echo's own incoherence in this room, so the
detector auto-widens in a reverberant/noisy room and tightens in a clean one
without any per-environment tuning. The configured margin is only a floor.

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
        margin_delta: float = 0.08,
        sigma_k: float = 3.0,
        confirm_frames: int = 1,
        warmup_frames: int = 5,
        nperseg: int = 256,
        baseline_alpha: float = 0.2,
        var_alpha: float = 0.15,
        provisional_baseline: float = 0.5,
        min_ref_rms: float = 1e-4,
        delay_history: int = 15,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.voiced_band = (float(voiced_band[0]), float(voiced_band[1]))
        self.margin_delta = float(margin_delta)  # FLOOR; the live margin can only widen
        self.sigma_k = float(sigma_k)
        # A barge must clear the threshold for this many CONSECUTIVE frames before
        # decide() returns True -- "slower but higher confidence". A single
        # over-threshold frame (e.g. a one-off nonlinear-echo spike from a cheap
        # speaker, or a transient) can no longer fire; only a SUSTAINED outlier,
        # which is what continuous talk-over produces, does. 1 = fire on the first
        # over-threshold frame (the original behaviour). While a run is still
        # building, decide() returns False (echo-only), NOT None, so a not-yet-
        # confirmed moment can never fall through to the legacy level gate.
        self._confirm_frames = max(1, int(confirm_frames))
        self._consec = 0  # consecutive over-threshold frames so far (capture thread)
        # Warm-up seeding (nonlinear-echo robustness). The control chart only LEARNS
        # the room's echo baseline on below-threshold (echo-only) frames; if the
        # echo's incoherent fraction is PERSISTENTLY above the provisional threshold
        # -- which a nonlinear/clipping open speaker can produce -- the learning
        # branch never runs, _baseline stays pinned at ``provisional_baseline``, and
        # every echo frame reads as a barge (self-interrupt that no confirm/sustain
        # can prevent, only delay). To break that starvation, the first
        # ``warmup_frames`` echo-bearing blocks after each reset/new speaking run --
        # which are echo-only by construction (a reply has just started; the user has
        # not begun talking over it) -- seed the baseline UNCONDITIONALLY (running
        # mean), even above the provisional threshold, so the chart learns this
        # room+speaker's TRUE echo floor (be it 0.2 or 0.8). During warm-up the
        # verdict is always echo-only (False): a barge in the first ~Nx0.1 s of a
        # reply is rare, and is only DELAYED, never lost. After warm-up the normal
        # mean+k*sigma control chart runs from the learned floor.
        self._warmup_frames = max(0, int(warmup_frames))
        self._warmup_left = self._warmup_frames
        self._warmup_sum = 0.0
        self._warmup_n = 0
        self.nperseg = int(nperseg)
        self.min_ref_rms = float(min_ref_rms)
        self.max_delay = max(1, int(self.sample_rate * max_delay_ms / 1000.0))
        self._ring_max = max(self.max_delay * 2, int(self.sample_rate * ring_ms / 1000.0))
        self._alpha = float(baseline_alpha)
        self._var_alpha = float(var_alpha)
        # Running MEAN of the echo-only incoherent fraction in THIS room (an EWMA
        # control chart). Starts high so the first ~1 s is conservative, then
        # settles to the room's true echo floor.
        self._baseline = float(provisional_baseline)
        # Variance of the echo-only fraction's UPWARD excursions above that mean
        # (how much it fluctuates in THIS room). The live trigger margin is
        # max(margin_delta, sigma_k * sqrt(var)) -> it auto-widens in a reverberant
        # / noisy room and tightens in a clean one, with no per-environment tuning.
        # Upward-only so the initial settling transient (all downward) and a
        # sustained barge can never inflate it.
        self._resid_var = 0.0

        self._ref: "deque[np.ndarray]" = deque()
        self._ref_len = 0
        self._lock = threading.Lock()
        self._delays: "deque[int]" = deque(maxlen=int(delay_history))

        # Diagnostics (read by tools/echo_probe.py and debug logs).
        self.available = _HAVE_SCIPY
        self.last_incoherent_fraction = 0.0
        self.last_delay_ms = 0.0
        self.last_baseline = self._baseline
        self.last_effective_margin = self.margin_delta
        self.last_decided: Optional[bool] = None
        self.last_consec = 0  # confirmation-run length at the last decision

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
        self._consec = 0  # new turn starts with a fresh confirmation run
        self._warmup_left = self._warmup_frames  # re-seed the echo floor each run
        self._warmup_sum = 0.0
        self._warmup_n = 0

    def _snapshot_ref(self) -> np.ndarray:
        with self._lock:
            if not self._ref:
                return np.zeros(0, dtype="float32")
            return np.concatenate(list(self._ref))

    def measured_delay_samples(self) -> Optional[int]:
        """Median of the recently-measured ECHO-ONLY speaker->mic delays (in
        ``sample_rate`` samples), or ``None`` if none measured yet.

        The engine feeds this to the AEC's far-end read delay to auto-calibrate it
        during the run: both the coherence reference ring (``note_playback``) and
        the AEC far-end ring are teed from the SAME audio callback (true playback
        position), so the delay is directly transferable. Read from the capture
        thread, which is the only writer of ``_delays`` (no lock needed)."""
        if not self._delays:
            return None
        return int(np.median(self._delays))

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
        # Warm-up seeding: for the first ``warmup_frames`` echo-bearing blocks after a
        # reset, fold ``frac`` into the baseline UNCONDITIONALLY (running mean) so the
        # control chart learns this room+speaker's true echo floor instead of
        # starving with _baseline pinned at provisional_baseline (which makes a
        # persistently-over-threshold nonlinear echo self-interrupt). Always echo-only
        # (False) during warm-up. ``raw_delay`` still feeds the alignment history.
        if self._warmup_left > 0:
            self._warmup_left -= 1
            self._warmup_sum += frac
            self._warmup_n += 1
            self._baseline = self._warmup_sum / self._warmup_n
            self._delays.append(raw_delay)
            self._consec = 0
            self.last_incoherent_fraction = frac
            self.last_delay_ms = 1000.0 * use_delay / self.sample_rate
            self.last_baseline = self._baseline
            self.last_effective_margin = self.margin_delta
            self.last_consec = 0
            self.last_decided = False
            return False
        # Self-calibrating EWMA control chart: a barge must clear the room's mean
        # echo incoherence by the LARGER of the configured floor and k standard
        # deviations of that room's own echo fluctuation. No fixed per-room
        # threshold -- the room sets it. A real barge is a large outlier far above
        # mean + k*sigma; echo-only fluctuation stays within it.
        eff_margin = max(self.margin_delta, self.sigma_k * math.sqrt(self._resid_var))
        over = bool(frac > self._baseline + eff_margin)
        if over:
            # Candidate barge frame: extend the confirmation run and DON'T update
            # the chart -- a barge (even an as-yet-unconfirmed one) must never drag
            # the baseline up toward itself. Fire only once the run reaches
            # ``confirm_frames`` consecutive over-threshold frames; a single spike
            # stays below that and is rejected.
            self._consec = min(self._consec + 1, self._confirm_frames)
        else:
            # Echo-only frame: the confirmation run is broken, and we trust this
            # frame's delay and update the mean + upward variance. Variance
            # accumulates only on UPWARD excursions (the ones that could cause a
            # false barge), which also keeps the initial mean-settling transient
            # (all downward) out of the estimate.
            self._consec = 0
            self._delays.append(raw_delay)
            resid = frac - self._baseline
            if resid > 0.0:
                self._resid_var = (
                    (1.0 - self._var_alpha) * self._resid_var + self._var_alpha * resid * resid
                )
            self._baseline = (1.0 - self._alpha) * self._baseline + self._alpha * frac

        # Confirmed barge only after a sustained over-threshold run. With
        # confirm_frames=1 this is exactly the old single-frame decision.
        is_user = self._consec >= self._confirm_frames

        self.last_incoherent_fraction = frac
        self.last_delay_ms = 1000.0 * use_delay / self.sample_rate
        self.last_baseline = self._baseline
        self.last_effective_margin = eff_margin
        self.last_consec = self._consec
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
