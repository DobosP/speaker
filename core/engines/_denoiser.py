"""Speech-denoise front-end for the capture hot path (sherpa-onnx GTCRN).

The on-device capture loop feeds the SAME 16 kHz block to the recognizer, the
speaker-ID embedder, and the VAD. Broadband noise corrupts all three at once:
it garbles STT *and* drags the user's speaker embedding well below the
enrollment gate (observed 0.30-0.46 vs an 0.5 threshold -> the enrolled user
gets locked out). A denoiser placed right after the resampler -- before any of
those consumers -- cleans the single block once, so every downstream model sees
the de-noised signal.

:class:`Denoiser` wraps sherpa-onnx's streaming ``OnlineSpeechDenoiser`` (the
tiny GTCRN model: ~523 KB, CPU, 16 kHz). It is deliberately small and stateful:

* ``process_16k(block)`` runs one inference on a mono float32 16 kHz block and
  returns the de-noised block (also float32, 16 kHz). The model's output length
  per call can differ from the input (a streaming front-end has warm-up delay) --
  that's fine, every consumer (``accept_waveform``, the embedder buffer, the VAD)
  accepts any length, exactly like :class:`~core.audio_frontend.AudioResampler`.
* It is **passthrough-on-error**: any inference failure returns the original
  block unchanged. The capture loop runs on a daemon thread, and a denoise
  exception that escaped would kill it (the "assistant went silent" failure
  mode). Failing open keeps the pipeline alive (noisy, but listening).
* ``reset()`` clears the streaming state; called on ASR decode-recovery so a
  recovered stream starts the denoiser fresh too.

``sherpa_onnx`` is imported lazily inside :func:`build_denoiser` (mirroring the
other builders in ``_sherpa_models``) so the runtime and the test suite import
without the native package or any model file present.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .sherpa import SherpaConfig

log = logging.getLogger("speaker.sherpa")


class Denoiser:
    """Stateful 16 kHz speech denoiser wrapping a sherpa-onnx denoiser object.

    The ``impl`` is duck-typed: anything exposing ``run(samples, sample_rate)``
    returning an object with a ``.samples`` float32 array (sherpa-onnx's
    ``DenoisedAudio``) and an optional ``reset()``. Injectable so the capture
    path can be unit-tested with a fake denoiser and no ONNX model.
    """

    def __init__(self, impl, *, sample_rate: int = 16000):
        self._impl = impl
        self.sample_rate = int(sample_rate)

    def process_16k(self, block):
        """De-noise one mono float32 16 kHz block; return it at 16 kHz.

        Passthrough-on-error: ANY failure returns the input block unchanged so a
        transient model hiccup can never crash the capture daemon thread."""
        import numpy as np

        try:
            out = self._impl.run(block, self.sample_rate)
            return np.asarray(out.samples, dtype="float32").reshape(-1)
        except Exception:  # noqa: BLE001 - never let denoise kill the capture loop
            log.debug("denoise block failed; passing the raw block through", exc_info=True)
            return block

    def reset(self) -> None:
        """Clear streaming state (call on ASR decode-recovery)."""
        reset = getattr(self._impl, "reset", None)
        if callable(reset):
            try:
                reset()
            except Exception:  # noqa: BLE001 - reset is best-effort
                pass


def build_denoiser(c: "SherpaConfig") -> Optional[Denoiser]:
    """Streaming GTCRN speech denoiser, or ``None`` when disabled/unbuildable.

    Returns ``None`` (so the capture path is byte-identical to today) unless
    BOTH ``denoise_enabled`` is true AND ``denoise_model`` is set. Mirrors
    :func:`~core.engines._sherpa_models.build_vad`: ``sherpa_onnx`` is imported
    lazily here.

    The sherpa-onnx ``OnlineSpeechDenoiser`` constructor RAISES on a bad model
    path, so we catch the eager load and FAIL OPEN (return ``None``) rather than
    crash ``start()`` -- a missing/broken denoise model degrades to no denoise,
    never to a dead engine.
    """
    if not getattr(c, "denoise_enabled", False) or not getattr(c, "denoise_model", ""):
        return None
    try:
        import sherpa_onnx

        config = sherpa_onnx.OnlineSpeechDenoiserConfig()
        config.model.gtcrn.model = c.denoise_model
        config.model.num_threads = c.resolved_asr_threads
        config.model.provider = c.provider
        impl = sherpa_onnx.OnlineSpeechDenoiser(config)
    except Exception as exc:  # noqa: BLE001 - bad path / load error -> fail open
        log.warning("could not build speech denoiser (%s); continuing WITHOUT denoise", exc)
        return None
    log.info("speech denoiser loaded (GTCRN): %s", c.denoise_model)
    return Denoiser(impl, sample_rate=c.sample_rate)
