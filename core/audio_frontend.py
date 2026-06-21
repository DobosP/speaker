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
