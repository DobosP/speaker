"""Capture-path audio front-end: anti-aliased downsampling to the ASR rate.

The mic on many laptops refuses 16 kHz and only opens at 44.1/48 kHz, so every
captured block must be downsampled to the recognizer's 16 kHz. The old path used
``np.interp`` linear interpolation with NO anti-alias low-pass, which folds
content above 8 kHz back into the speech band -- measured ~44000x more in-band
alias energy than a proper polyphase resampler, corrupting the fricative/sibilant
filterbank features the zipformer reads (a real, per-block WER hit).

:class:`AudioResampler` is the fix: a stateful, anti-aliased resampler that
prefers ``soxr.ResampleStream`` (carries its FIR state across the 0.1 s blocks so
there's no per-block seam), falls back to ``scipy.signal.resample_poly`` (a
stateless polyphase filter -- still anti-aliased, fine for offline/one-shot use),
and finally to naive linear only if neither library is importable. Shared by the
live engine (``core.engines.sherpa``) and the enrollment recorder so both feed
the ASR/speaker models the same band-limited signal.

Pair this with capturing at 48 kHz when the device allows it: 48000 -> 16000 is a
clean integer /3 decimation (short, well-conditioned FIR), versus the ugly
160/441 fractional ratio of 44100 -> 16000.
"""
from __future__ import annotations

import math
from math import gcd


def _linear_resample(samples, src_sr: int, dst_sr: int):
    """Last-resort naive linear resample (no anti-alias). Self-contained so this
    module has no import cycle with the engine."""
    import numpy as np

    x = np.asarray(samples, dtype="float32").reshape(-1)
    if src_sr == dst_sr or x.size == 0:
        return x
    n_out = int(round(x.shape[0] * float(dst_sr) / float(src_sr)))
    if n_out <= 0:
        return np.zeros(0, dtype="float32")
    idx = np.linspace(0.0, x.shape[0] - 1, num=n_out)
    return np.interp(idx, np.arange(x.shape[0]), x).astype("float32")


def apply_gain_soft_limit(samples, gain: float):
    """Apply mic gain with a soft-knee limiter instead of a hard clip.

    A hard clip (``np.clip(x*gain, -1, 1)``) flat-tops loud phonemes into
    square-wave distortion that injects in-band harmonics and smears the vowel
    formants the ASR reads (measured: ``gain=8`` pinned ~0.28% of samples at
    full scale, ~9.8% THD on a loud phoneme). The soft knee leaves levels below
    the knee perfectly linear and saturates only the peaks smoothly. ``gain ==
    1.0`` is a no-op. Apply this BEFORE the anti-alias resampler so any residual
    saturation harmonics above 8 kHz are filtered out before the recognizer."""
    import numpy as np

    if gain == 1.0:
        return samples
    x = (np.asarray(samples, dtype="float32") * float(gain)).copy()
    knee = 0.8
    mag = np.abs(x)
    over = mag > knee
    if over.any():
        x[over] = np.sign(x[over]) * (
            knee + (1.0 - knee) * np.tanh((mag[over] - knee) / (1.0 - knee))
        )
    return x.astype("float32")


def normalize_rms(samples, target_rms: float, *, max_gain: float = 20.0):
    """Scale a waveform so its RMS ~= ``target_rms`` (soft-knee limited on peaks).

    Evens out per-sentence TTS output level: an offline VITS model emits a
    DIFFERENT amplitude per sentence (a function of text/phonetics), so on an open
    speaker the played-back echo level swings sentence-to-sentence and the barge-in
    echo floor (``_playback_floor_rms``) can never settle -- which is why the
    open-speaker interrupt both self-fired (floor dipped) and missed real talk-over
    (floor spiked). A fixed RMS target couples a STABLE echo level into the mic (and
    a steady, even output volume for the listener). Reuses ``apply_gain_soft_limit``
    so a loud sentence saturates smoothly instead of hard-clipping. ``target_rms <=
    0`` is a no-op (caller keeps the raw stream); ``max_gain`` caps the boost so a
    near-silent clip is not amplified into noise."""
    import numpy as np

    if target_rms <= 0.0:
        return samples
    x = np.asarray(samples, dtype="float32").reshape(-1)
    if x.size == 0:
        return x
    rms = float(np.sqrt(np.mean(x.astype("float64") ** 2)))
    if rms <= 1e-6:  # silence -> nothing to normalize
        return x
    gain = min(float(target_rms) / rms, float(max_gain))
    return apply_gain_soft_limit(x, gain)


def declick(samples, *, threshold: float = 0.18, max_run: int = 8):
    """Repair isolated impulse samples in a synthesized waveform.

    The on-device sherpa-onnx VITS voice (``en_US-libritts_r-medium``) emits
    occasional sample-level SPIKES on certain text -- deterministic (same text ->
    same spikes), content-correlated, often dozens in a single sentence. On an
    open speaker these are audible clicks/crackle ("strange noise"), and a spike
    teed into the AEC/coherence reference can also nudge a false self-interrupt.
    They are NOT FIFO underruns and NOT chunk-concatenation seams (measured
    2026-06-19): a standalone synth with zero engine load reproduces them.

    Detection is a 3-point median: a sample whose deviation from the median of
    itself and its two neighbours exceeds ``threshold`` is an impulse (a genuine
    fast speech transition moves and STAYS, so it tracks the median and is left
    alone). Short runs (<= ``max_run`` samples) are repaired by linear
    interpolation across the surrounding good samples. Cheap, allocation-light,
    runs on the synthesis (producer) thread -- never the real-time audio callback.

    Safety: a no-op on clean speech (measured corr 1.0000, 0 samples touched on a
    glitch-free clip); ~80-95% of clicks removed on a glitchy clip with speech
    correlation >= 0.996. ``threshold <= 0`` disables it (raw passthrough)."""
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view

    if threshold <= 0.0:
        return samples
    x = np.asarray(samples, dtype="float32").reshape(-1)
    if x.size < 3:
        return x
    med = np.median(sliding_window_view(np.pad(x, 1, mode="edge"), 3), axis=1)
    bad = np.abs(x - med) > float(threshold)
    if not bad.any():
        return x
    y = x.copy()
    n = x.size
    i = 0
    while i < n:
        if bad[i]:
            j = i
            while j < n and bad[j] and j - i < max_run:
                j += 1
            lo, hi = i - 1, j
            if lo >= 0 and hi < n:                     # interior run -> interpolate across it
                y[i:j] = np.linspace(y[lo], y[hi], j - i + 2)[1:-1]
            i = j
        else:
            i += 1
    return y


# --- OUTPUT LEVELER -------------------------------------------------------
# Opt-in TTS output stage (perceptual loudness target + look-ahead true-peak
# limiter), a pure-numpy port of WebRTC's AGC2 adaptive-digital leveler +
# fixed-digital limiter. It REPLACES ``normalize_rms`` when on (which targets a
# LINEAR RMS and would fight the perceptual target); ``declick`` still runs in
# both modes. Runs on the synthesis (producer) thread, never the audio callback
# -- cheap (one RMS reduction, one flat multiply, a 4x peak-only upsample of the
# already-small ~1-3 s clip, and a per-sample envelope pass). Default OFF, so a
# clean clone keeps the legacy ``normalize_rms`` path byte-identical.
#
# Why dBFS referenced to full scale 1.0 (not 32768): the repo audio is float in
# [-1, 1]. The WebRTC dB MATH is scale-invariant for GAINS; only the absolute
# level constants change reference. So we re-reference dBFS to 1.0
# (``dbfs = 20*log10(|x|)``, full scale = 1.0) and keep everything in [-1, 1] --
# 0 dBFS == |x|=1.0. The true-peak ceiling is expressed in dBTP relative to that
# 1.0 full scale (e.g. -1.0 dBTP leaves ~1 dB of inter-sample headroom).


def _rms_dbfs_voiced(x, *, floor_dbfs: float = -50.0):
    """Perceptual speech-level estimate: RMS over the VOICED samples only, in
    dBFS (full scale = 1.0). Gating out near-silence (below ``floor_dbfs``) so
    leading/trailing silence and inter-word gaps don't drag the measured level
    down -- this is the analog of AGC2 weighting the level estimate by speech
    probability (here ~1 everywhere TTS is actually speaking). Returns ``None``
    when the clip is essentially silent (caller leaves loudness unchanged)."""
    import numpy as np

    x = np.asarray(x, dtype="float32").reshape(-1)
    if x.size == 0:
        return None
    floor = 10.0 ** (float(floor_dbfs) / 20.0)
    mag = np.abs(x)
    voiced = x[mag > floor]
    if voiced.size < x.size * 0.02:  # essentially silence -> no estimate
        return None
    r = float(np.sqrt(np.mean(voiced.astype("float64") ** 2)))
    if r <= 1e-6:
        return None
    return 20.0 * math.log10(r)


def _upsampled_abs_per_sample(x, oversample: int):
    """Inter-sample (true) peak magnitude per ORIGINAL sample, via ``oversample``x
    upsampling for the MEASUREMENT only (the upsampled signal is discarded). For
    each original sample we take the max |.| over its ``oversample`` upsampled
    points (and the join to the next sample), so a magnitude that peaks BETWEEN
    two samples -- which a per-sample |x| misses but the DAC reconstruction
    actually emits -- is caught. Prefers ``scipy.signal.resample_poly`` (polyphase,
    same library the capture resampler uses) and falls back to ``np.interp``
    linear upsampling. Length of the returned array == ``x.size``."""
    import numpy as np

    x = np.asarray(x, dtype="float64").reshape(-1)
    n = x.size
    if n == 0:
        return np.zeros(0, dtype="float64")
    os = max(1, int(oversample))
    if os == 1 or n == 1:
        return np.abs(x)
    up = None
    if _has_scipy():
        try:
            from scipy.signal import resample_poly

            up = resample_poly(x, os, 1).astype("float64")
        except Exception:  # noqa: BLE001 - degrade to linear
            up = None
    if up is None:
        idx = np.linspace(0.0, n - 1, num=n * os - (os - 1))
        up = np.interp(idx, np.arange(n), x)
    up_abs = np.abs(up)
    # Map back to one value per ORIGINAL sample: the max over this sample's
    # upsampled span (incl. the rise to the next sample). Build n windows of
    # length `os+1` (clamped at the tail) and take their max -- allocation-light.
    out = np.empty(n, dtype="float64")
    for k in range(os + 1):
        lo = k
        if lo == 0:
            cand = up_abs[0 : n * os : os]
            cand = cand[:n]
        else:
            sl = up_abs[lo : n * os : os]
            cand = np.empty(n, dtype="float64")
            cand[: sl.size] = sl
            cand[sl.size :] = sl[-1] if sl.size else 0.0
        if k == 0:
            out[:] = cand
        else:
            np.maximum(out, cand, out=out)
    return out


def _running_min_forward(g, window: int):
    """Forward look-ahead: replace each sample with the MIN of itself and the
    next ``window`` samples, so a gain reduction lands BEFORE the peak it must
    tame (no overshoot). Vectorized via a strided sliding window (cheap on the
    ~1-3 s clip). ``window <= 1`` is a no-op."""
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view

    g = np.asarray(g, dtype="float64").reshape(-1)
    w = max(1, int(window))
    if w <= 1 or g.size == 0:
        return g
    # Pad the tail with the last value so the final samples have a full window.
    padded = np.concatenate([g, np.full(w - 1, g[-1], dtype="float64")])
    return sliding_window_view(padded, w).min(axis=1)


def _true_peak_gain_envelope(x, ceiling_lin: float, sr: int, *, oversample: int = 4,
                             lookahead_ms: float = 1.5, release_ms: float = 50.0):
    """Per-sample gain-reduction envelope holding the INTER-SAMPLE peak below
    ``ceiling_lin``, with forward look-ahead (gain dips before the peak -> no
    overshoot) + a slow one-pole release (no pumping). Instant attack: the gain
    drops immediately when the look-ahead window sees a peak; it recovers only as
    fast as the release one-pole allows. This is the look-ahead soft-knee true-
    peak limiter -- the envelope is continuous, so it acts as a soft knee."""
    import numpy as np

    x = np.asarray(x, dtype="float32").reshape(-1)
    if x.size == 0:
        return np.ones(0, dtype="float32")
    mag = _upsampled_abs_per_sample(x, oversample)  # len == x.size, float64
    desired = np.ones_like(mag)
    hot = mag > ceiling_lin
    # Guard divide-by-zero (mag>ceiling implies mag>0, but stay defensive).
    desired[hot] = np.where(mag[hot] > 0.0, ceiling_lin / mag[hot], 1.0)
    # Look-ahead: running MIN over a forward window so the gain dips before peak.
    la = max(1, int(sr * lookahead_ms / 1000.0))
    desired = _running_min_forward(desired, la)
    # One-pole release: gain may only RISE this slowly back toward 1.0; a drop
    # (attack) is instantaneous so an incoming peak is always tamed in time.
    rel = math.exp(-1.0 / max(1.0, sr * release_ms / 1000.0))
    g = np.empty_like(desired)
    cur = 1.0
    for i in range(desired.size):  # cheap: a TTS clip is ~1-3 s
        d = desired[i]
        cur = d if d < cur else (cur * rel + d * (1.0 - rel))
        g[i] = cur
    return g.astype("float32")


def output_leveler(samples, *, target_dbfs: float, true_peak_dbtp: float, sr: int,
                   prev_gain_db=None, max_boost_db: float = 18.0,
                   max_cut_db: float = 12.0, max_slew_db_per_s: float = 6.0,
                   oversample: int = 8, lookahead_ms: float = 1.5,
                   release_ms: float = 50.0):
    """Opt-in TTS output stage: a speech-level (loudness) target + look-ahead
    true-peak limiter, in one fused pass. Pure numpy, allocation-light, runs on
    the producer thread (never the audio callback).

    STAGE 1 (loudness): measure the clip's speech level -- broadband dBFS RMS over
    the VOICED samples (a cheap proxy for loudness; not BS.1770 K-weighted LUFS,
    which is overkill for one fixed on-device voice -- libebur128 is the upgrade
    path). Compute ``desired = target_dbfs - measured`` clamped to
    ``[-max_cut_db, +max_boost_db]``, then move toward it from ``prev_gain_db``:

    * ``prev_gain_db is None`` (the FIRST utterance of a session) -> SEED straight
      to ``desired`` so the very first reply is on target (no audible ramp-up);
    * otherwise SLEW, TIME-AWARE like WebRTC AGC2: at most
      ``max_slew_db_per_s * clip_duration_s`` dB this call. This damps the ~2 dB
      sentence-to-sentence swing + the rare large correction WITHOUT the
      reply-length dependence a per-call cap has (a 0.4 s 'Yes.' and an 8 s
      sentence would otherwise both move the same fixed step).

    Applied as a flat linear multiply.

    STAGE 2 (true-peak limit): :func:`_true_peak_gain_envelope` builds a
    CONTINUOUS per-sample gain reduction (= a soft knee, no step discontinuities)
    that holds the inter-sample peak below ``true_peak_dbtp`` (dBTP relative to
    1.0 full scale); a final NaN-safe clip at full scale is the brick-wall safety
    net (a no-op on real signal, since the ceiling is sub-0 dBFS). ``oversample``
    defaults to 8x so the measured true peak converges to within ~0.001 of the
    ceiling even on dense near-Nyquist content (4x under-reads it by ~0.33 dB).

    Returns ``(leveled_float32, applied_gain_db)`` so the caller carries
    ``applied_gain_db`` (a float) as ``prev_gain_db`` into the next sentence.
    Length-preserving (the oversample is for peak MEASUREMENT only and is
    discarded), NaN-safe, silence -> passthrough (loudness gain held, limiter
    inert)."""
    import numpy as np

    x = np.asarray(samples, dtype="float32").reshape(-1)
    # NaN/Inf-safe up front so neither the level estimate nor the multiply trips a
    # RuntimeWarning (TTS never emits these; the function is documented NaN-safe).
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.size == 0:
        return x, (0.0 if prev_gain_db is None else float(prev_gain_db))

    # --- Stage 1: speech-level (loudness) target; seed first, then time-aware slew ---
    meas = _rms_dbfs_voiced(x)
    if meas is None:  # silence -> don't move the loudness gain
        gain_db = 0.0 if prev_gain_db is None else float(prev_gain_db)
    else:
        desired = float(target_dbfs) - meas
        desired = max(-float(max_cut_db), min(float(max_boost_db), desired))
        if prev_gain_db is None:                       # first utterance -> seed to target
            gain_db = desired
        else:                                          # time-aware slew (dB/s * duration)
            step = float(max_slew_db_per_s) * (x.size / float(sr))
            delta = max(-step, min(step, desired - float(prev_gain_db)))
            gain_db = float(prev_gain_db) + delta
    y = x * np.float32(10.0 ** (gain_db / 20.0))

    # --- Stage 2: look-ahead true-peak limiter (always last) ---
    ceiling = 10.0 ** (float(true_peak_dbtp) / 20.0)  # dBTP -> linear, full scale 1.0
    env = _true_peak_gain_envelope(
        y, ceiling, int(sr), oversample=oversample,
        lookahead_ms=lookahead_ms, release_ms=release_ms,
    )
    y = (y * env).astype("float32")
    # Brick-wall safety net: the continuous per-sample envelope IS a soft knee
    # (no step discontinuities), so a hard clip here only ever catches residual
    # inter-sample overshoot BETWEEN the sampled peaks -- and it clips at full
    # scale (1.0), comfortably ABOVE the sub-0-dBFS ceiling the envelope targets,
    # so on real signal it never fires. NaN-safe.
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(y, -1.0, 1.0, out=y)
    return y.astype("float32"), gain_db


def compute_input_calibration(
    blocks,
    *,
    floor_percentile: float = 20.0,
    headroom: float = 3.0,
    min_floor: float = 0.004,
    max_floor: float = 0.08,
    clip_level: float = 0.98,
):
    """Estimate a device's ambient operating point from a few captured blocks.

    The "smart + generic across all devices" calibration: instead of a hand-set
    per-machine gain, measure what THIS mic's quiet-between-sounds level actually
    is at startup and set the AGC's noise floor just above it -- so a deliberately
    low (never-clipping) OS gain still reaches the recognizer, on any hardware,
    with no manual tuning. ``blocks`` is a list of mono float32 arrays already at
    the ASR rate (pre-AGC). Returns a dict the engine applies to :class:`InputAGC`
    and logs into the run bundle.

    ``noise_floor_rms`` = clamp(``headroom`` * the LOW-percentile per-block RMS,
    ``min_floor``, ``max_floor``). The low percentile (not the mean) is the quiet
    floor, robust to a cough or a word spoken during the calibration window; the
    headroom lifts the AGC's "this is signal, not hiss" gate just above it.
    ``clipping_fraction`` flags a too-HOT ADC the boost-only AGC cannot fix (the
    user must lower the OS level). Empty input -> safe defaults (no-op floor)."""
    import numpy as np

    rmss = []
    peak = 0.0
    clipped = 0
    total = 0
    for b in blocks:
        x = np.asarray(b, dtype="float32").reshape(-1)
        if x.size == 0:
            continue
        rmss.append(float(np.sqrt(np.mean(x.astype("float64") ** 2))))
        peak = max(peak, float(np.max(np.abs(x))))
        clipped += int(np.count_nonzero(np.abs(x) >= clip_level))
        total += int(x.size)
    if not rmss:
        return {
            "noise_floor_rms": float(min_floor),
            "ambient_rms": 0.0,
            "peak": 0.0,
            "clipping_fraction": 0.0,
            "n_blocks": 0,
        }
    ambient = float(np.percentile(rmss, floor_percentile))
    floor = min(max(headroom * ambient, float(min_floor)), float(max_floor))
    return {
        "noise_floor_rms": float(floor),
        "ambient_rms": ambient,
        "peak": peak,
        "clipping_fraction": (clipped / total) if total else 0.0,
        "n_blocks": len(rmss),
    }


class InputAGC:
    """Stateful automatic gain control for the capture path.

    Lets the user run a deliberately LOW (never-clipping) OS mic gain: the AGC
    boosts a clean-but-quiet block toward ``target_rms`` so the recognizer hears
    a healthy level -- instead of forcing the user to find the one OS gain that
    is neither clipping (too loud) nor inaudible (too quiet). It is BOOST-ONLY
    (gain >= 1.0): software cannot un-clip a hardware-saturated signal, so it
    never attenuates -- the user lowers the OS gain below the ADC clip point and
    the AGC carries the rest.

    The smoothed gain RISES slowly (``rise``) so the noise floor between words is
    not pumped up, and FALLS fast (``fall``) so a sudden loud phrase is reined in
    before the soft-knee limiter has to work. Blocks below ``noise_floor_rms``
    HOLD the gain rather than amplifying hiss. Applied via
    :func:`apply_gain_soft_limit` so any residual peak saturates smoothly instead
    of hard-clipping. ``gain`` starts at 1.0, so the first blocks are a passthrough.
    """

    def __init__(self, *, target_rms: float = 0.12, max_gain: float = 12.0,
                 noise_floor_rms: float = 0.004, rise: float = 0.08, fall: float = 0.4):
        self.target_rms = float(target_rms)
        self.max_gain = float(max_gain)
        self.noise_floor_rms = float(noise_floor_rms)
        self.rise = float(rise)
        self.fall = float(fall)
        self.gain = 1.0

    def process(self, samples):
        import numpy as np

        x = np.asarray(samples, dtype="float32").reshape(-1)
        if x.size:
            r = float(np.sqrt(np.mean(x.astype("float64") ** 2)))
            if r > self.noise_floor_rms:                 # real signal (not silence/hiss)
                desired = min(self.max_gain, max(1.0, self.target_rms / r))
                # rise slow / fall fast, so the gain doesn't pump the noise floor
                # up between words yet still backs off quickly on a loud phrase.
                rate = self.rise if desired > self.gain else self.fall
                self.gain += rate * (desired - self.gain)
        return apply_gain_soft_limit(x, self.gain)


class AudioResampler:
    """Stateful anti-aliased downsampler for the capture hot path.

    ``process(block)`` accepts a mono float32 block at ``src_sr`` and returns it
    at ``dst_sr``. Output length per call varies (the streaming FIR has warm-up
    delay) -- that's fine, ``sherpa_onnx`` ``accept_waveform`` takes any length.
    Identity (zero-copy passthrough) when ``src_sr == dst_sr``.
    """

    def __init__(self, src_sr: int, dst_sr: int, *, quality: str = "HQ"):
        self.src_sr = int(src_sr)
        self.dst_sr = int(dst_sr)
        self.kind = "identity" if self.src_sr == self.dst_sr else "linear"
        self._stream = None
        if self.src_sr != self.dst_sr:
            try:
                import soxr

                self._stream = soxr.ResampleStream(
                    self.src_sr, self.dst_sr, 1, dtype="float32", quality=quality
                )
                self.kind = "soxr"
            except Exception:  # noqa: BLE001 - soxr missing/old -> polyphase/linear fallback
                self._stream = None
                self.kind = "scipy" if _has_scipy() else "linear"

    def process(self, block, *, last: bool = False):
        import numpy as np

        x = np.asarray(block, dtype="float32").reshape(-1)
        if self.src_sr == self.dst_sr or x.size == 0:
            return x
        if self.kind == "soxr" and self._stream is not None:
            try:
                out = self._stream.resample_chunk(x, last=last)
                return np.asarray(out, dtype="float32").reshape(-1)
            except Exception:  # noqa: BLE001 - degrade rather than drop the block
                pass
        if _has_scipy():
            try:
                from scipy.signal import resample_poly

                g = gcd(self.src_sr, self.dst_sr) or 1
                return resample_poly(x, self.dst_sr // g, self.src_sr // g).astype("float32")
            except Exception:  # noqa: BLE001
                pass
        return _linear_resample(x, self.src_sr, self.dst_sr)

    def reset(self) -> None:
        """Clear filter state (call on stream recovery / endpoint reset)."""
        if self._stream is not None:
            try:
                self._stream.clear()
            except Exception:  # noqa: BLE001
                pass


def _has_scipy() -> bool:
    import importlib.util

    return importlib.util.find_spec("scipy") is not None


# Clean integer-ratio capture rates to prefer over a non-integer-ratio device
# native rate (e.g. 44100). 48000/16000 = 3, 32000/16000 = 2, 96000/16000 = 6.
CLEAN_CAPTURE_RATES = (48000, 32000, 96000)
