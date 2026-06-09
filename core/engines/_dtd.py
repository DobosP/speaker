"""Self-calibrating, device-adaptive double-talk (barge-in) detector.

Fixed dB / coherence margins cannot fit every device: the echo level varies by
speaker, volume and room, so a margin tuned to avoid self-interrupt on a loud
nonlinear speaker forces the user to *shout*, while one tuned to catch a quiet
talk-over self-interrupts on the assistant's own echo. Three live attempts on an
open laptop speaker confirmed this (coherence-alone self-interrupted; coherence +
post-AEC residual rejected normal talk-over because DTLN suppresses the user;
coherence + a fixed +10 dB raw margin still needed a shout over a loud echo).

This detector keeps **no fixed margin in the trigger path**. Each feature is
scored as an *upward z-score* from its OWN echo-only EWMA control chart -- the
spread is MEASURED from this device's echo, so the bar auto-widens on a loud /
nonlinear / reverberant speaker and tightens on a clean one (the "adapt to the
machine + environment" requirement). The features are FUSED by summing the
clamped z-scores: a real talk-over lifts all of them together so the sum crosses
a single dimensionless, device-INDEPENDENT threshold ``K``, while echo-only keeps
every z near 0. That fusion is what escapes the single-feature failure modes --
coherence alone overlaps voice on a nonlinear speaker, residual alone is AEC-
suppressed, raw alone needs a device-specific dB; the *sum* has positive headroom
where each part does not.

Pure numpy-free math (plain floats), so it is cheap on the capture thread and
unit-testable with no audio. See core/engines/sherpa.py ``_looks_like_user`` for
the wiring and tools/echo_probe.py for the per-frame ``D`` diagnostics used to
calibrate ``K`` on a new machine.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional, Sequence


class Chart:
    """Upward-only EWMA control chart for one echo-only feature.

    Learns the feature's mean + (upward) variance from echo-only blocks, after a
    short warm-up that seeds the mean UNCONDITIONALLY (running mean of the first
    few blocks) so a persistently-high feature -- e.g. the nonlinear-echo
    incoherence that sits ~0.88 on a cheap speaker -- cannot starve the chart at
    its provisional value. ``z(x)`` standardizes a sample; the sigma denominator
    is floored RELATIVE to the mean so a very steady feature (tiny measured
    variance) does not make z explode on a small fluctuation.
    """

    def __init__(
        self,
        *,
        warmup: int = 5,
        alpha: float = 0.05,
        var_alpha: float = 0.15,
        rel_floor: float = 0.15,
        provisional: float = 0.0,
    ) -> None:
        self._alpha = float(alpha)
        self._var_alpha = float(var_alpha)
        self._rel_floor = float(rel_floor)  # sigma floor as a fraction of the mean
        self._warmup0 = max(0, int(warmup))
        self._provisional = float(provisional)
        self.reset()

    def reset(self) -> None:
        self._mean = self._provisional
        self._var = 0.0
        self._warmup_left = self._warmup0
        self._warm_sum = 0.0
        self._warm_n = 0

    @property
    def warming(self) -> bool:
        return self._warmup_left > 0

    @property
    def mean(self) -> float:
        return self._mean

    def _sigma(self) -> float:
        # Floor sigma relative to the mean (steady feature can't blow z up), with a
        # tiny absolute floor for the mean~0 case.
        return max(math.sqrt(self._var), self._rel_floor * abs(self._mean), 1e-6)

    @property
    def sigma(self) -> float:
        return self._sigma()

    def z(self, x: float) -> float:
        return (float(x) - self._mean) / self._sigma()

    def seed(self, x: float) -> None:
        """Warm-up: running mean of the first few (echo-only) blocks, unconditional."""
        self._warm_sum += float(x)
        self._warm_n += 1
        self._mean = self._warm_sum / self._warm_n
        self._warmup_left -= 1

    def update_echo(self, x: float) -> None:
        """Post-warm-up echo-only update: EWMA mean + UPWARD-only variance. A barge
        excursion is upward and must NOT inflate the echo variance toward itself, so
        only positive residuals feed the variance (mirrors EchoCoherenceDetector)."""
        x = float(x)
        resid = x - self._mean
        if resid > 0.0:
            self._var = (1.0 - self._var_alpha) * self._var + self._var_alpha * resid * resid
        self._mean = (1.0 - self._alpha) * self._mean + self._alpha * x


class AdaptiveDTD:
    """Fused, self-calibrating double-talk detector.

    Feed it the three echo-only-calibratable features per capture block and it
    returns True (barge), False (echo-only / still warming) -- never a fixed
    margin. ``decide`` sums the clamped upward z-scores into ``D`` and fires when
    ``D > K`` for ``confirm_frames`` consecutive blocks. While a candidate run is
    open the charts are FROZEN (a barge must never drag its own floors up); on an
    echo-only block all charts learn.
    """

    def __init__(
        self,
        *,
        k: float = 5.0,
        weights: Sequence[float] = (1.0, 1.0, 0.5),
        confirm_frames: int = 3,
        warmup_frames: int = 5,
        chart_alpha: float = 0.05,
        chart_var_alpha: float = 0.15,
        chart_rel_floor: float = 0.15,
    ) -> None:
        self.k = float(k)
        w = list(weights) + [0.0, 0.0, 0.0]
        self.w_raw, self.w_resid, self.w_coh = float(w[0]), float(w[1]), float(w[2])
        self._confirm = max(1, int(confirm_frames))

        def _chart(prov: float = 0.0) -> Chart:
            return Chart(
                warmup=warmup_frames,
                alpha=chart_alpha,
                var_alpha=chart_var_alpha,
                rel_floor=chart_rel_floor,
                provisional=prov,
            )

        self._raw = _chart()
        self._resid = _chart()
        self._coh = _chart(0.5)  # incoherence starts mid-scale (matches EchoCoherenceDetector)
        self._consec = 0
        # Diagnostics (read by tools/echo_probe.py + debug logs).
        self.last_z_raw = 0.0
        self.last_z_resid = 0.0
        self.last_z_coh = 0.0
        self.last_D = 0.0
        self.last_consec = 0
        self.last_decided: Optional[bool] = None

    def reset(self) -> None:
        """New speaking run -> re-arm warm-up + clear the confirmation run."""
        self._raw.reset()
        self._resid.reset()
        self._coh.reset()
        self._consec = 0

    def decide(self, raw_rms: float, resid_rms: float, incoherent_fraction: float) -> bool:
        """Return True iff this is a confirmed user talk-over. False = echo-only
        (or still warming up the per-device charts). Never raises; pure floats."""
        charts = (self._raw, self._resid, self._coh)
        vals = (float(raw_rms), float(resid_rms), float(incoherent_fraction))
        # Warm-up: seed all charts on the first few (echo-only-by-construction)
        # blocks of a reply and report echo-only, so a not-yet-calibrated moment
        # can never fire.
        if any(c.warming for c in charts):
            for c, v in zip(charts, vals):
                c.seed(v)
            self._consec = 0
            self.last_z_raw = self.last_z_resid = self.last_z_coh = 0.0
            self.last_D = 0.0
            self.last_consec = 0
            self.last_decided = False
            return False
        z_raw = max(self._raw.z(vals[0]), 0.0)
        z_resid = max(self._resid.z(vals[1]), 0.0)
        z_coh = max(self._coh.z(vals[2]), 0.0)
        D = self.w_raw * z_raw + self.w_resid * z_resid + self.w_coh * z_coh
        if D > self.k:
            # Candidate barge frame: extend the run and FREEZE the charts (don't let
            # the talk-over raise its own bar).
            self._consec += 1
        else:
            # Echo-only frame: the run breaks and every chart learns this device's echo.
            self._consec = 0
            for c, v in zip(charts, vals):
                c.update_echo(v)
        decided = self._consec >= self._confirm
        self.last_z_raw, self.last_z_resid, self.last_z_coh = z_raw, z_resid, z_coh
        self.last_D = D
        self.last_consec = self._consec
        self.last_decided = decided
        return decided


class BargeSustain:
    """Temporal confirmation that turns the per-frame double-talk verdict into a
    barge-in CUT.

    ``AdaptiveDTD`` reports per frame, but a real talk-over's verdict FLICKERS:
    the user breathes/pauses and DTLN suppresses the user's voice mid-double-talk,
    so the detector fires on only *some* of the talk-over blocks. A single eligible
    frame is too trigger-happy (one echo spike self-interrupts), and the old
    capture-loop accumulator -- ``voiced_run += block`` on a fire, ``*= 0.5`` on a
    miss, fire at ``min_speech_sec`` -- could never accumulate intermittent fires:
    the live failure ``run-20260609-203236`` fired the DTD on 3 of the 5 turn-2
    blocks (and only 2 on the turn-3 pre-shout) yet the ``*= 0.5`` decay kept
    ``voiced_run`` below the 0.3s bar, so a NORMAL-volume talk-over never cut and
    the owner had to shout.

    This integrates eligibility over a short TRAILING WINDOW and fires when at
    least ``min_voiced_sec`` worth of the last ``window_sec`` were eligible. Two
    properties fall out:

    * **Responsive to flicker** -- 2 eligible blocks within the window fire, even
      with a miss between them, so a genuine talk-over cuts without a shout.
    * **Echo-safety is STRUCTURAL** -- the window is bounded, so the sporadic
      single-frame echo leaks that survive AEC over a long reply (the recorded run
      had exactly one, idx 34) can NEVER accumulate to a cut; only a sustained
      talk-over packs enough eligible frames into one window. This is what the old
      unbounded "just lower the threshold / never decay" alternatives could not
      guarantee on the open nonlinear speaker.

    Pure ints + a bounded ``deque``, so it is cheap on the capture thread and
    unit-testable with no audio. The caller resets it on a fire (and while the
    barge debounce/refractory window is active); see core/engines/sherpa.py.
    """

    def __init__(
        self,
        *,
        window_sec: float = 0.5,
        block_sec: float = 0.1,
        min_voiced_sec: float = 0.2,
    ) -> None:
        block = max(1e-6, float(block_sec))
        self._maxlen = max(1, round(float(window_sec) / block))
        self._need = max(1, round(float(min_voiced_sec) / block))
        # need can't exceed the window (else it could never fire); clamp defensively.
        self._need = min(self._need, self._maxlen)
        self._window: deque = deque(maxlen=self._maxlen)
        # Diagnostics (parity with AdaptiveDTD; read by debug logs/tests).
        self.last_count = 0
        self.last_fired = False

    @property
    def window_frames(self) -> int:
        return self._maxlen

    @property
    def need_frames(self) -> int:
        return self._need

    def reset(self) -> None:
        """Clear the window -- a new speaking run, a fire, or a suppressed block."""
        self._window.clear()
        self.last_count = 0
        self.last_fired = False

    def update(self, eligible: bool) -> bool:
        """Record this block's eligibility and return True iff a cut should fire
        (>= ``need_frames`` eligible within the trailing ``window_frames``).

        ``update`` never raises and is the only mutator besides ``reset``. The
        count can only reach the threshold on an *eligible* block (a miss appends
        ``False`` and cannot raise the count), so a fire always lands on a real
        talk-over block; the caller should ``reset`` after acting on a True."""
        self._window.append(bool(eligible))
        self.last_count = sum(self._window)
        self.last_fired = self.last_count >= self._need
        return self.last_fired
