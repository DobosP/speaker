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
