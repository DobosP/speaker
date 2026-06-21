"""WebRTC Audio Processing Module (APM) echo-cancel/denoise/AGC backend.

This is the production audio front-end that Chrome, Google Meet, and Microsoft
Teams ship: **AEC3** (a multi-delay, partitioned-block echo canceller that
tolerates a *nonlinear* loudspeaker -- exactly where the hand-rolled linear NLMS
filter in :mod:`core.engines._aec` measures ~0 dB ERLE and diverges on an open
laptop speaker), a **residual-echo suppressor**, **ML noise suppression**, an
**AGC2** gain controller, and a **high-pass filter** -- all in one stateful
stage. We reach it through the ``livekit`` package, which re-exports libwebrtc's
``rtc.AudioProcessingModule`` and runs it standalone (no room / no network).

The module conforms to the same impl seam every other canceller uses -- a
``process(near, far)`` method that :class:`core.engines._aec.EchoCanceller`
duck-types and wraps -- so it drops into the capture loop with no new plumbing:
the existing :class:`FarEndRing` provides the time-aligned far-end reference and
the loop already tees the true played block into it.

WebRTC's APM is frame-locked to **10 ms** (160 samples at 16 kHz) and int16, so
this wrapper buffers the variable-length capture blocks into 10 ms near/far frame
pairs, runs ``process_reverse_stream`` (render/far) then ``process_stream``
(capture/near, in place), and returns the processed near samples; a sub-frame
remainder is carried to the next call (output length can differ from input --
every downstream consumer already accepts that, like the resampler and the FDAF
filter).

Fails to import cleanly when ``livekit`` is absent: :func:`build_apm_impl`
returns ``None`` and :func:`core.engines._aec.build_aec` then falls open to
no-AEC, never crashing ``start()``.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger("speaker.aec.apm")

_FRAME_MS = 10  # WebRTC APM is hard-locked to 10 ms frames.


class _WebRTCAPM:
    """Stateful WebRTC APM wrapper exposing ``process_16k(near, far)``.

    ``near``/``far`` are mono float32 arrays in [-1, 1] at ``sample_rate``. The
    far-end is assumed already coarsely time-aligned to the mic (the FarEndRing
    read at the configured speaker->mic delay); ``stream_delay_ms`` is the small
    residual hint AEC3 refines from. Pure in-process; real-time on CPU."""

    def __init__(
        self,
        *,
        echo_cancellation: bool = True,
        noise_suppression: bool = True,
        high_pass_filter: bool = True,
        gain_control: bool = False,
        stream_delay_ms: int = 0,
        sample_rate: int = 16000,
    ) -> None:
        from livekit import rtc  # lazy: optional dependency

        self._rtc = rtc
        self._apm = rtc.AudioProcessingModule(
            echo_cancellation=bool(echo_cancellation),
            noise_suppression=bool(noise_suppression),
            high_pass_filter=bool(high_pass_filter),
            auto_gain_control=bool(gain_control),
        )
        self.sample_rate = int(sample_rate)
        self.frame = int(self.sample_rate * _FRAME_MS / 1000)  # 160 @ 16 kHz
        self.stream_delay_ms = int(stream_delay_ms)
        self._echo = bool(echo_cancellation)
        # Sub-frame carry buffers (kept in lockstep: both grow by the same block
        # length each call, so they stay sample-aligned frame-for-frame).
        self._near_buf = np.zeros(0, dtype=np.float32)
        self._far_buf = np.zeros(0, dtype=np.float32)

    def _frame_int16(self, mono_f32):
        pcm = np.clip(mono_f32, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype("<i2")
        return self._rtc.AudioFrame(
            data=pcm.tobytes(),
            sample_rate=self.sample_rate,
            num_channels=1,
            samples_per_channel=int(pcm.shape[0]),
        )

    def process(self, near, far=None):
        """Echo-cancel/clean one near-end block against the aligned far-end block.
        Named ``process`` (not ``process_16k``) to match the impl contract
        :class:`core.engines._aec.EchoCanceller` duck-types and wraps."""
        near = np.asarray(near, dtype=np.float32).reshape(-1)
        if far is None:
            far = np.zeros_like(near)
        else:
            far = np.asarray(far, dtype=np.float32).reshape(-1)
        # Keep the two streams the same length before buffering so frames pair up.
        if far.shape[0] != near.shape[0]:
            if far.shape[0] < near.shape[0]:
                far = np.concatenate([far, np.zeros(near.shape[0] - far.shape[0], dtype=np.float32)])
            else:
                far = far[: near.shape[0]]
        self._near_buf = np.concatenate([self._near_buf, near])
        self._far_buf = np.concatenate([self._far_buf, far])

        F = self.frame
        n_frames = self._near_buf.shape[0] // F
        if n_frames == 0:
            return np.zeros(0, dtype=np.float32)

        out = np.empty(n_frames * F, dtype=np.float32)
        for k in range(n_frames):
            lo = k * F
            near_fr = self._near_buf[lo : lo + F]
            far_fr = self._far_buf[lo : lo + F]
            if self._echo:
                # Render (far-end) first so the canceller has the reference, then
                # the small delay hint, then the capture pass (modified in place).
                self._apm.process_reverse_stream(self._frame_int16(far_fr))
                self._apm.set_stream_delay_ms(self.stream_delay_ms)
            cap = self._frame_int16(near_fr)
            self._apm.process_stream(cap)
            proc = np.frombuffer(bytes(cap.data), dtype="<i2").astype(np.float32) / 32768.0
            out[lo : lo + F] = proc

        # Carry the sub-frame remainder.
        consumed = n_frames * F
        self._near_buf = self._near_buf[consumed:].copy()
        self._far_buf = self._far_buf[consumed:].copy()
        return out

    def reset(self) -> None:
        """Drop the carry buffers on a barge-in / stream reset. The APM's own
        learned echo model is intentionally KEPT (re-learning per utterance would
        regress cancellation) -- only the unframed remainder is cleared."""
        self._near_buf = np.zeros(0, dtype=np.float32)
        self._far_buf = np.zeros(0, dtype=np.float32)


def build_apm_impl(c) -> Optional["_WebRTCAPM"]:
    """Construct the APM impl from a SherpaConfig, or ``None`` (fail open) when
    ``livekit`` is missing or the module can't be built. Caller wraps it in an
    :class:`core.engines._aec.EchoCanceller`."""
    try:
        impl = _WebRTCAPM(
            echo_cancellation=True,
            noise_suppression=bool(getattr(c, "apm_noise_suppression", True)),
            high_pass_filter=bool(getattr(c, "apm_high_pass_filter", True)),
            gain_control=bool(getattr(c, "apm_gain_control", False)),
            stream_delay_ms=int(getattr(c, "apm_stream_delay_ms", 0) or 0),
            sample_rate=int(getattr(c, "sample_rate", 16000)),
        )
    except ImportError:
        log.warning(
            "aec_backend='apm' needs the `livekit` package (rtc.AudioProcessingModule) "
            "-- it is not installed; continuing WITHOUT AEC. `pip install livekit` to "
            "enable the WebRTC APM, or set aec_backend='nlms'/'dtln'."
        )
        return None
    except Exception as exc:  # noqa: BLE001 - any APM build failure -> fail open
        log.warning("could not build the WebRTC APM (%s); continuing WITHOUT AEC", exc)
        return None
    log.info(
        "AEC active: WebRTC APM (AEC3 + RES%s%s%s, always_on=%s)",
        ", NS" if getattr(c, "apm_noise_suppression", True) else "",
        ", HPF" if getattr(c, "apm_high_pass_filter", True) else "",
        ", AGC2" if getattr(c, "apm_gain_control", False) else "",
        getattr(c, "apm_always_on", False),
    )
    return impl
