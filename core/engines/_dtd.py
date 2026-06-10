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
        z_freeze: float = 0.0,
        robust_seed: bool = False,
        freeze_limit: int = 30,
    ) -> None:
        self._alpha = float(alpha)
        self._var_alpha = float(var_alpha)
        self._rel_floor = float(rel_floor)  # sigma floor as a fraction of the mean
        self._warmup0 = max(0, int(warmup))
        self._provisional = float(provisional)
        # Regime-change backstop for z_freeze: after this many CONSECUTIVE frozen
        # (outlier) updates, absorb anyway. A real talk-over reaches z >> K and
        # opens a candidate run (decide() freezes the charts there) long before
        # this bound; only a SUSTAINED sub-K elevation -- a genuine echo regime
        # change like a volume step-up -- accumulates a frozen run this long, and
        # without the bound a persistent chart would deadlock below it forever
        # (every block an outlier, none absorbed, the bar never rises). <= 0 ->
        # freeze indefinitely.
        self._freeze_limit = int(freeze_limit)
        # Anti-contamination (2026-06-10): ``z_freeze`` > 0 -> an echo-only update
        # whose own z exceeds it is NOT absorbed (it is plausibly the user, and
        # absorbing it raises the chart's own bar -- the recorded miss-feedback
        # loop). ``robust_seed`` -> warm-up seeds from the LOWER HALF of the warm
        # samples instead of their mean, so a user already talking at reply onset
        # cannot seed the baseline at talk-over level (echo gaps dominate the low
        # end). Both default OFF here for construction-compat; AdaptiveDTD turns
        # them ON by default.
        self._z_freeze = float(z_freeze)
        self._robust_seed = bool(robust_seed)
        self.reset()

    def reset(self) -> None:
        self._mean = self._provisional
        self._var = 0.0
        self._warmup_left = self._warmup0
        self._warm_sum = 0.0
        self._warm_n = 0
        self._warm_vals: list[float] = []
        self._frozen_run = 0

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
        """Warm-up seeding from the first few blocks of a run.

        Legacy (``robust_seed=False``): unconditional running mean -- assumes the
        warm-up blocks are echo-only by construction. That assumption FAILS when
        the user is already talking at reply onset (the most common real barge:
        objecting as soon as the wrong answer starts) -- the live 2026-06-10
        misses seeded the baseline AT talk-over level so ``z`` stayed 0 for
        seconds. Robust (``robust_seed=True``): seed from the mean of the LOWER
        HALF of the warm samples -- user speech only ADDS energy, so the low end
        of the warm-up blocks (inter-word gaps, echo-only moments) is the best
        echo estimate under possible double-talk."""
        x = float(x)
        self._warm_sum += x
        self._warm_n += 1
        if self._robust_seed:
            self._warm_vals.append(x)
            low = sorted(self._warm_vals)[: max(1, (len(self._warm_vals) + 1) // 2)]
            self._mean = sum(low) / len(low)
        else:
            self._mean = self._warm_sum / self._warm_n
        self._warmup_left -= 1

    def update_echo(self, x: float) -> None:
        """Post-warm-up echo-only update: EWMA mean + UPWARD-only variance. A barge
        excursion is upward and must NOT inflate the echo variance toward itself, so
        only positive residuals feed the variance (mirrors EchoCoherenceDetector).

        ``z_freeze``: a sample that is an OUTLIER on this chart's own scale
        (``z > z_freeze``) is never absorbed at all -- even when the FUSED ``D``
        stayed under ``K`` (e.g. a talk-over that lifted only the residual). The
        recorded 2026-06-10 miss-feedback loop: each missed talk-over block fed
        the baseline, inflating mean+variance toward the user until ``z_resid``
        pinned at 0. Bounded risk: a genuine step-rise in echo level (volume
        knob) freezes too, but per-sentence TTS normalization keeps the echo
        stable and the residual-floor margin gate still bounds any false fires."""
        x = float(x)
        if self._z_freeze > 0.0 and self.z(x) > self._z_freeze:
            self._frozen_run += 1
            if self._freeze_limit <= 0 or self._frozen_run < self._freeze_limit:
                return
            # Sustained sub-K elevation = a genuine echo regime change (e.g. a
            # volume step-up): absorb so a persistent chart cannot deadlock
            # below the new floor forever.
        self._frozen_run = 0
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
        chart_z_freeze: float = 3.0,
        chart_robust_seed: bool = True,
        chart_freeze_limit: int = 30,
        persistent_charts: bool = True,
    ) -> None:
        self.k = float(k)
        w = list(weights) + [0.0, 0.0, 0.0]
        self.w_raw, self.w_resid, self.w_coh = float(w[0]), float(w[1]), float(w[2])
        self._confirm = max(1, int(confirm_frames))
        # Persistent charts (2026-06-10 contamination fix): the learned echo
        # baselines CARRY ACROSS speaking runs (see new_run). The per-reply cold
        # restart was the shared root of both recorded failures: warm-up re-seeded
        # on whatever played at reply onset (often the USER, talking over the
        # reply's first words -> z_resid pinned at 0 for seconds = the 4s
        # scream-miss), and a fresh near-zero chart made z explode on reply-onset
        # echo (= the self-interrupt). Per-sentence TTS loudness normalization
        # (plan step 1) makes the echo level stable run-to-run, so persistence is
        # correct-by-design now. False -> legacy per-run reset.
        self._persistent = bool(persistent_charts)

        def _chart(prov: float = 0.0) -> Chart:
            return Chart(
                warmup=warmup_frames,
                alpha=chart_alpha,
                var_alpha=chart_var_alpha,
                rel_floor=chart_rel_floor,
                provisional=prov,
                z_freeze=chart_z_freeze,
                robust_seed=chart_robust_seed,
                freeze_limit=chart_freeze_limit,
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
        """FULL reset: re-arm warm-up + clear the confirmation run. For
        construction/config changes and tests; speaking-run boundaries use
        :meth:`new_run` (which preserves the learned charts by default)."""
        self._raw.reset()
        self._resid.reset()
        self._coh.reset()
        self._consec = 0

    def new_run(self) -> None:
        """Speaking-run boundary (silent->speaking / a cut). With persistent
        charts (the default) only the candidate-confirmation run clears -- the
        learned per-device echo baselines survive, so the next reply needs no
        warm-up and a user already talking at its onset cannot poison the seed.
        With ``persistent_charts=False`` this is the legacy full reset."""
        if not self._persistent:
            self.reset()
            return
        self._consec = 0

    def observe_echo(
        self, raw_rms: float, resid_rms: float, incoherent_fraction: float
    ) -> None:
        """Echo-only observation from a block the fire path did NOT evaluate.

        ``decide`` only runs on VAD-speech blocks (the capture loop gates the
        barge check on the VAD), so the charts' learning diet was biased toward
        exactly the blocks most likely to contain the USER. This tap lets the
        capture loop feed the charts the VAD-QUIET playback blocks -- the most
        reliably echo-only samples there are -- so the baselines track the true
        echo. Seeds during warm-up, learns after; never opens, extends, or
        breaks a candidate run (skipped while one is open: charts stay frozen
        during a candidate barge). Never raises; pure floats."""
        if self._consec > 0:
            return
        charts = (self._raw, self._resid, self._coh)
        vals = (float(raw_rms), float(resid_rms), float(incoherent_fraction))
        if any(c.warming for c in charts):
            for c, v in zip(charts, vals):
                c.seed(v)
            return
        for c, v in zip(charts, vals):
            c.update_echo(v)

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
