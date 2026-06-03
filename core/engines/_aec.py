"""Acoustic Echo Cancellation (AEC) front-end for the capture hot path.

On open speakers with no AEC the assistant's own TTS leaks into the mic and looks
exactly like a user barging in -- it self-interrupts (see
``docs/archive/session_2026-05-31_acoustic_real_voice.md``; today only a 6 dB output-margin
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


class PlaybackFIFO:
    """Thread-safe bounded float32 sample FIFO between the synthesizer and the
    PortAudio output callback.

    Why this exists: the AEC far-end reference must be time-aligned to ACTUAL
    acoustic playback. With a single blocking ``out.write()`` the producer pushed
    a whole multi-second TTS chunk into :class:`FarEndRing` the instant it handed
    it to ``write()`` -- but the device buffers and plays it out over the next
    seconds, so the ring write head raced ahead of real playback by a varying
    output-buffer amount (per-block far->near lag swung 54..1179 ms; DTLN tolerates
    only +/-60 ms). The fix routes audio through this FIFO: the producer
    (``write()`` on the playback thread) ``write``s into it (BLOCKING when full --
    that backpressure paces synthesis to real time), and the PortAudio callback
    DRAINS it frame-for-frame via :meth:`read_into`. The callback -- and only the
    callback -- then tees the exact block it is about to play into the far-end
    ring, so the ring write head == the true playback position and the far->near
    lag becomes small + stable (inside DTLN's tolerance).

    Single-producer (the playback worker's ``write()``) / single-consumer (the
    audio callback's :meth:`read_into`). A short lock guards each numpy copy; the
    consumer's hold is a couple of microsecond slice copies + one non-blocking
    ``notify_all`` (never blocks the high-priority audio thread). The producer may
    block on a ``Condition`` when the ring is full; it is woken by the consumer's
    ``notify_all`` (a drain freed space) OR returns promptly when ``should_abort``
    flips True (barge-in / shutdown), with a finite wait timeout as a backstop
    against a missed notify so it can never deadlock."""

    def __init__(self, capacity: int):
        self._cap = max(1, int(capacity))
        self._buf = np.zeros(self._cap, dtype=np.float32)
        self._r = 0           # read head (mod cap)
        self._w = 0           # write head (mod cap)
        self._count = 0       # queued samples
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def write(self, samples, should_abort) -> None:
        """Producer (playback thread). Append ``samples``, BLOCKING with
        backpressure while the ring is full so synthesis is paced to real
        playback. NEVER called from the audio callback.

        ``should_abort`` is a 0-arg predicate (True on barge-in/shutdown). It
        STOPS this write enqueueing anything further and returns promptly so a
        producer parked on a full ring can never deadlock once the consumer stops
        draining. A long chunk is copied over several lock passes (it can fill the
        ring and backpressure mid-chunk), so an abort that arrives mid-chunk may
        leave an earlier pass's samples already queued -- and a true rollback is
        not possible anyway once the audio callback has already PLAYED part of
        them. The drop of those still-queued samples is therefore the job of
        :meth:`flush`, which the barge-in / shutdown paths always call alongside
        setting the abort predicate (see ``SherpaOnnxEngine.stop_speaking`` /
        ``stop``). CONTRACT: a caller that wants the queue emptied on abort MUST
        pair ``should_abort`` with a :meth:`flush`; ``should_abort`` alone only
        halts further synthesis, it does not retroactively clear the ring."""
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        i = 0
        n = x.shape[0]
        while i < n:
            with self._cond:
                # Wait for space, but bail out (finite timeout backstop) the
                # moment a barge-in/shutdown is requested so we never deadlock a
                # producer that the consumer can no longer drain. flush() (paired
                # with the abort by the caller) clears whatever earlier passes
                # already queued; here we only stop adding more.
                while self._count >= self._cap and not should_abort():
                    self._cond.wait(timeout=0.1)
                if should_abort():
                    return
                free = self._cap - self._count
                take = min(free, n - i)
                start = self._w
                end = start + take
                if end <= self._cap:
                    self._buf[start:end] = x[i : i + take]
                else:
                    k = self._cap - start
                    self._buf[start:] = x[i : i + k]
                    self._buf[: end - self._cap] = x[i + k : i + take]
                self._w = end % self._cap
                self._count += take
                i += take
                # Wake the consumer if it was (in principle) waiting; harmless
                # otherwise. The consumer never blocks, so this is purely to keep
                # the symmetry tidy -- the real wake direction is consumer->producer.

    def read_into(self, out_view) -> int:
        """Consumer (audio callback ONLY). Fill ``out_view`` (the PortAudio mono
        ``outdata[:, 0]`` slice) in place with the next queued samples, zero-fill
        any underrun tail, and return the count of REAL (non-zero-fill) samples
        emitted so the callback tees only those to the echo refs. Hard real-time:
        a single microsecond copy lock + one non-blocking ``notify_all`` to wake a
        blocked producer; never blocks, never allocates a new array."""
        n = int(out_view.shape[0])
        if n <= 0:
            return 0
        with self._lock:
            m = min(n, self._count)
            if m:
                start = self._r
                end = start + m
                if end <= self._cap:
                    out_view[:m] = self._buf[start:end]
                else:
                    k = self._cap - start
                    out_view[:k] = self._buf[start:]
                    out_view[k:m] = self._buf[: end - self._cap]
                self._r = end % self._cap
                self._count -= m
            if m < n:
                out_view[m:] = 0.0          # underrun -> silent zero-fill, never a stall
            self._cond.notify_all()         # a drain freed space: wake a blocked producer
        return m

    def flush(self) -> None:
        """Barge-in cut: drop every queued sample in one short lock (the next
        callback emits silence -- equivalent to discarding the device buffer) and
        wake any producer blocked in :meth:`write`."""
        with self._cond:
            self._r = self._w
            self._count = 0
            self._cond.notify_all()

    def count(self) -> int:
        """Queued-sample count (used by the idle drain-wait to know when the last
        frames have actually played out before tearing down the echo refs)."""
        with self._lock:
            return self._count


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
        mu: float = 0.3,
        leak: float = 0.9999,
        power_smooth: float = 0.9,
        eps: float = 1e-6,
        doubletalk_freeze: bool = True,
        dt_factor: float = 2.5,
        warmup_frames: int = 8,
        diverge_factor: float = 2.0,
        diverge_shrink: float = 0.5,
    ) -> None:
        # mu is the NLMS step (lower = slower but more STABLE); leak < 1.0 adds
        # exponential coefficient forgetting that BOUNDS divergence (the classic
        # leaky-LMS safeguard -- with leak=1.0 the taps can grow without limit on a
        # nonlinear / mis-aligned echo and the output explodes). diverge_* is the
        # hard cap: when the filter AMPLIFIES (predicted echo or output louder than
        # the near-end) it has diverged, so shrink the taps and emit the raw input.
        self.F = int(frame)
        self.N = 2 * self.F
        self.mu = float(mu)
        self.leak = float(leak)
        self.lam = float(power_smooth)
        self.eps = float(eps)
        self.dt_freeze = bool(doubletalk_freeze)
        self.dt_factor = float(dt_factor)
        self.warmup = int(warmup_frames)
        self.diverge_factor = float(diverge_factor)
        self.diverge_shrink = float(diverge_shrink)
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

        d_rms = float(np.sqrt(np.mean(d * d) + self.eps))
        yhat_rms = float(np.sqrt(np.mean(yhat * yhat) + self.eps))
        e_rms = float(np.sqrt(np.mean(e * e) + self.eps))

        # DIVERGENCE RECOVERY (active from frame 0). A stable canceller never
        # AMPLIFIES: the predicted echo and the output both sit at or below the
        # near-end. If either has blown past it the (linear) filter has diverged on
        # a nonlinear / mis-aligned echo -- shrink the taps hard, skip adaptation,
        # and emit the RAW near-end so the output can never explode (and thus can
        # never self-interrupt the barge gate downstream).
        if (
            not np.isfinite(yhat_rms)
            or not np.isfinite(e_rms)
            or yhat_rms > self.diverge_factor * d_rms
            or e_rms > self.diverge_factor * d_rms
        ):
            self.W *= self.diverge_shrink
            self.x_prev = x.copy()
            self.count += 1
            return d.astype(np.float32)

        adapt = True
        if float(np.dot(x, x)) < self.eps:
            adapt = False                             # nothing playing -> no echo
        elif self.dt_freeze and self.count >= self.warmup and d_rms > self.dt_factor * yhat_rms:
            adapt = False                             # near-end speech (double-talk) -> freeze
        if adapt:
            E = np.fft.fft(np.concatenate([np.zeros(F), e]))
            dW = (self.mu / (self.P + self.eps)) * np.conj(X) * E
            phi = np.fft.ifft(dW).real
            phi[F:] = 0.0                             # gradient (causality) constraint
            self.W = self.leak * self.W + np.fft.fft(phi)

        self.x_prev = x.copy()
        self.count += 1
        return e


class _DTLNEchoCanceller:
    """Two-stage DTLN-aec deep echo canceller under onnxruntime (the quality tier).

    Reproduces ``breizhn/DTLN-aec`` ``run_aec.py`` streaming inference exactly:
    block_len 512, block_shift 128 (75 % overlap). Stage 1 predicts a spectral mask
    from the mic + loopback (far-end) MAGNITUDE spectra and applies it to the mic
    complex spectrum; stage 2 refines the masked TIME-domain block using the
    loopback time signal. Both stages carry their LSTM state block-to-block (the
    state tensors are explicit model I/O). Output is overlap-added at the 128 hop.

    I/O is mapped by SHAPE so it survives the tflite->ONNX converter's naming: in
    each stage the 4-D ``[1,2,512,2]`` tensor is the LSTM state, and the two equal
    feature tensors are, in graph-input order, ``[primary, reference]`` (mic-mag &
    lpb-mag for stage 1; masked-block & lpb-block for stage 2) -- matching the
    upstream ``input_details[0]=primary, [1]=states, [2]=reference`` order.

    ``onnxruntime`` is imported lazily in ``__init__`` (so the module imports with
    no native dep); a load error raises and ``build_aec`` fails open to no-AEC. The
    NumPy FDAF (``aec_backend='nlms'``) is the dependency-free default; this is the
    heavier, higher-quality tier for louder / more reverberant rooms."""

    BLOCK = 512
    SHIFT = 128

    def __init__(self, stage1_path: str = "", stage2_path: str = "", *, providers=None,
                 sessions=None):
        # ``sessions`` (a (stage1, stage2) pair of objects with get_inputs/
        # get_outputs/run) is injected in tests so the streaming + state-carry
        # logic runs with no ONNX model and no onnxruntime. Production loads from
        # the paths via a lazily-imported onnxruntime.
        if sessions is not None:
            self._s1, self._s2 = sessions
        else:
            import onnxruntime as ort

            prov = list(providers) if providers else ["CPUExecutionProvider"]
            self._s1 = ort.InferenceSession(stage1_path, providers=prov)
            self._s2 = ort.InferenceSession(stage2_path, providers=prov)
        self._m1 = self._io_map(self._s1)
        self._m2 = self._io_map(self._s2)
        self._reset_state()

    @staticmethod
    def _io_map(sess) -> dict:
        """Classify a stage's I/O: the 4-D tensor is the LSTM state, the two equal
        feature tensors are [primary, reference] in graph order."""
        ins = sess.get_inputs()
        state_in = next(i for i in ins if len(i.shape) == 4)
        feats = [i for i in ins if len(i.shape) != 4]
        outs = sess.get_outputs()
        state_out = next(o for o in outs if len(o.shape) == 4)
        main_out = next(o for o in outs if len(o.shape) != 4)
        return {
            "primary": feats[0].name,
            "reference": feats[1].name,
            "state_in": state_in.name,
            "state_shape": tuple(state_in.shape),
            "main_out": main_out.name,
            "state_out": state_out.name,
        }

    def _reset_state(self) -> None:
        self.in_buf = np.zeros(self.BLOCK, dtype=np.float32)
        self.lpb_buf = np.zeros(self.BLOCK, dtype=np.float32)
        self.out_buf = np.zeros(self.BLOCK, dtype=np.float32)
        self.st1 = np.zeros(self._m1["state_shape"], dtype=np.float32)
        self.st2 = np.zeros(self._m2["state_shape"], dtype=np.float32)
        self._nq = np.zeros(0, dtype=np.float32)
        self._fq = np.zeros(0, dtype=np.float32)

    def reset(self) -> None:
        self._reset_state()

    def process(self, near, far) -> np.ndarray:
        near = np.asarray(near, dtype=np.float32).reshape(-1)
        far = np.asarray(far, dtype=np.float32).reshape(-1)
        if far.shape[0] < near.shape[0]:
            far = np.concatenate([far, np.zeros(near.shape[0] - far.shape[0], dtype=np.float32)])
        elif far.shape[0] > near.shape[0]:
            far = far[: near.shape[0]]
        self._nq = np.concatenate([self._nq, near])
        self._fq = np.concatenate([self._fq, far])
        S = self.SHIFT
        chunks = []
        while self._nq.shape[0] >= S and self._fq.shape[0] >= S:
            mic_shift, self._nq = self._nq[:S], self._nq[S:]
            lpb_shift, self._fq = self._fq[:S], self._fq[S:]
            chunks.append(self._block_step(mic_shift, lpb_shift))
        if chunks:
            return np.concatenate(chunks).astype(np.float32)
        return np.zeros(0, dtype=np.float32)

    def _block_step(self, mic_shift, lpb_shift) -> np.ndarray:
        S, B = self.SHIFT, self.BLOCK
        # Shift the newest 128 samples into the 512 analysis buffers.
        self.in_buf = np.concatenate([self.in_buf[S:], mic_shift])
        self.lpb_buf = np.concatenate([self.lpb_buf[S:], lpb_shift])

        in_fft = np.fft.rfft(self.in_buf)
        in_mag = np.abs(in_fft).astype(np.float32).reshape(1, 1, -1)
        lpb_mag = np.abs(np.fft.rfft(self.lpb_buf)).astype(np.float32).reshape(1, 1, -1)

        m1 = self._m1
        mask, self.st1 = self._s1.run(
            [m1["main_out"], m1["state_out"]],
            {m1["primary"]: in_mag, m1["reference"]: lpb_mag, m1["state_in"]: self.st1},
        )
        estimated = np.fft.irfft(in_fft * mask.reshape(-1)).astype(np.float32)

        m2 = self._m2
        out_block, self.st2 = self._s2.run(
            [m2["main_out"], m2["state_out"]],
            {
                m2["primary"]: estimated.reshape(1, 1, B),
                m2["reference"]: self.lpb_buf.reshape(1, 1, B),
                m2["state_in"]: self.st2,
            },
        )
        # Overlap-add (hop = 128) and emit the oldest 128 samples.
        self.out_buf = np.concatenate([self.out_buf[S:], np.zeros(S, dtype=np.float32)])
        self.out_buf = self.out_buf + np.asarray(out_block, dtype=np.float32).reshape(-1)
        return self.out_buf[:S].copy()


def _resolve_dtln_paths(model: str):
    """From the ``aec_model`` setting, return ``(stage1, stage2)`` ONNX paths or
    ``None``. ``model`` is a directory holding ``dtln_aec_stage1.onnx`` +
    ``dtln_aec_stage2.onnx`` (what ``tools.setup_models --aec-model`` writes), or a
    direct path to the stage-1 file (stage 2 derived by swapping 1->2)."""
    import os
    import re

    if not model:
        return None
    if os.path.isdir(model):
        s1 = os.path.join(model, "dtln_aec_stage1.onnx")
        s2 = os.path.join(model, "dtln_aec_stage2.onnx")
    else:
        s1 = model
        s2 = re.sub(r"(?<!\d)1(\.onnx)$", r"2\1", model)
    if os.path.exists(s1) and os.path.exists(s2):
        return s1, s2
    return None


class EchoCanceller:
    """The seam the engine holds. ``impl`` is duck-typed (the NumPy FDAF now, an
    ONNX DTLN session later): anything with ``process(near, far) -> samples`` and
    an optional ``reset()``. ``process_16k`` is **passthrough-on-error** so a
    transient failure can never crash the capture daemon thread."""

    # An echo canceller must REDUCE echo. If the adaptive filter diverges and the
    # output is LOUDER than the input by this ratio, the result is unusable -- it
    # explodes the downstream level/barge gate and self-interrupts (observed on an
    # open laptop speaker: post-AEC RMS ~7.4 the instant playback began). Above it,
    # discard the AEC output, reset the filter, and pass the raw near-end through.
    _DIVERGENCE_RATIO = 2.0  # +6 dB

    def __init__(self, impl, *, sample_rate: int = 16000):
        self._impl = impl
        self.sample_rate = int(sample_rate)

    def process_16k(self, near, far):
        """Echo-cancel one mono float32 16 kHz near-end block against the aligned
        far-end block. Returns the cancelled near-end (length may differ -- the
        adaptive filter frames internally). On ANY error returns ``near`` unchanged.

        DIVERGENCE GUARD: a linear adaptive filter fed a mis-aligned / under-powered
        reference can diverge and AMPLIFY. A canceller must never make the signal
        louder, so if the output exceeds the input by ``_DIVERGENCE_RATIO`` (or goes
        non-finite) we discard it, reset the adaptive state, and pass the raw
        near-end through -- bounded and safe, never an explosion that self-interrupts."""
        try:
            out = self._impl.process(near, far)
        except Exception:  # noqa: BLE001 - never let AEC kill the capture loop
            log.debug("AEC block failed; passing the raw near-end through", exc_info=True)
            return near
        try:
            if out is None or len(out) == 0:
                return out
            near_arr = np.asarray(near, dtype=np.float64).reshape(-1)
            near_rms = float(np.sqrt(np.mean(near_arr ** 2))) if near_arr.size else 0.0
            out_rms = float(np.sqrt(np.mean(np.asarray(out, dtype=np.float64) ** 2)))
            if not np.isfinite(out_rms) or out_rms > max(near_rms, 1e-6) * self._DIVERGENCE_RATIO:
                log.debug(
                    "AEC diverged (out_rms=%.3f >> near_rms=%.3f); reset + passthrough",
                    out_rms, near_rms,
                )
                self.reset()
                return near_arr.astype(np.float32)
        except Exception:  # noqa: BLE001 - the guard must never crash the capture loop
            return out
        return out

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
        mu = float(getattr(c, "aec_mu", 0.3) or 0.3)
        leak = float(getattr(c, "aec_leak", 0.9999) or 0.9999)
        log.info(
            "AEC active: NumPy FDAF adaptive filter (frame=%d, mu=%.3f, leak=%.4f, doubletalk_freeze=%s)",
            frame, mu, leak, freeze,
        )
        return EchoCanceller(
            _FDAFAdaptiveFilter(frame, mu=mu, leak=leak, doubletalk_freeze=freeze), sample_rate=sr
        )
    if backend == "dtln":
        paths = _resolve_dtln_paths(str(getattr(c, "aec_model", "") or ""))
        if paths is None:
            log.warning(
                "aec_backend='dtln' but the converted ONNX stages weren't found at "
                "aec_model=%r -- continuing WITHOUT AEC. Run "
                "`python -m tools.setup_models --aec-model` to fetch + convert them, "
                "or set aec_backend='nlms' for the dependency-free filter.",
                getattr(c, "aec_model", ""),
            )
            return None
        try:
            ep = "CUDAExecutionProvider" if str(getattr(c, "provider", "cpu")).lower() == "cuda" else "CPUExecutionProvider"
            impl = _DTLNEchoCanceller(paths[0], paths[1], providers=[ep])
        except Exception as exc:  # noqa: BLE001 - bad model / missing onnxruntime -> fail open
            log.warning("could not load DTLN-aec ONNX (%s); continuing WITHOUT AEC", exc)
            return None
        log.info("AEC active: DTLN-aec deep ONNX tier (%s)", paths[0])
        return EchoCanceller(impl, sample_rate=sr)
    log.warning("unknown aec_backend=%r; continuing WITHOUT AEC", backend)
    return None
