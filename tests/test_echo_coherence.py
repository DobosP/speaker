"""Offline proof of the reference-coherence barge-in detector (no audio device).

The properties under test are exactly the user's requirements:
  * the assistant's own TTS echo NEVER fires barge-in, at ANY playback gain
    (the 134-self-interrupt scar, gone structurally);
  * a user talking over the assistant DOES fire -- including when the user is
    QUIETER than the echo (the case the loudness gate fails);
  * the decision is INVARIANT to a uniform volume scaling of the whole mix
    ("same utterance at any volume"), with ZERO enrollment.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from core.engines.echo_coherence import EchoCoherenceDetector

SR = 16000
BLOCK = 1600  # 0.1 s, the real capture block
DELAY = 96  # 6 ms echo delay (samples)


def _make_reference(seconds: float, *, seed: int) -> np.ndarray:
    """Broadband reference with energy across the voiced band (a stand-in for
    TTS); white noise gives a sharp cross-correlation peak for the delay test."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(SR * seconds)).astype("float32")


def _push(det: EchoCoherenceDetector, ref: np.ndarray) -> None:
    for i in range(0, ref.size, BLOCK):
        det.note_playback(ref[i : i + BLOCK], SR)


def _echo_block(ref: np.ndarray, *, gain: float) -> np.ndarray:
    """The mic block 'now' is a delayed, scaled copy of the most-recent
    reference -- i.e. the assistant's own voice leaking back in."""
    end = ref.size - DELAY
    return (gain * ref[end - BLOCK : end]).astype("float32")


def _user(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(BLOCK).astype("float32")


def _fresh(**kw) -> EchoCoherenceDetector:
    return EchoCoherenceDetector(SR, **kw)


def _prime_echo_baseline(det: EchoCoherenceDetector, ref: np.ndarray, *, gain: float, n: int = 12):
    """Settle the runtime baseline on echo-only frames, as happens live in the
    first ~1 s of an assistant turn before any barge."""
    last = None
    for _ in range(n):
        last = det.decide(_echo_block(ref, gain=gain))
    return last


def test_echo_only_never_fires_at_any_playback_gain():
    ref = _make_reference(0.8, seed=1)
    for gain in (0.2, 0.5, 1.0, 2.0):
        det = _fresh()
        _push(det, ref)
        verdict = _prime_echo_baseline(det, ref, gain=gain)
        # The assistant's own TTS is fully explained by the reference -> never a barge.
        assert verdict is False, f"self-interrupt at playback gain {gain}"
        assert det.last_incoherent_fraction < det.margin_delta + 0.05


def test_user_over_echo_fires_including_when_quieter_than_echo():
    ref = _make_reference(0.8, seed=2)
    echo_gain = 0.5
    # user_gain / echo_gain ratios: 0.6 (user *quieter* than echo), 1.0, 2.0.
    for ratio in (0.6, 1.0, 2.0):
        det = _fresh()
        _push(det, ref)
        _prime_echo_baseline(det, ref, gain=echo_gain)
        mix = _echo_block(ref, gain=echo_gain) + (echo_gain * ratio) * _user(seed=99)
        verdict = det.decide(mix)
        assert verdict is True, f"missed barge at user/echo ratio {ratio}"


def test_decision_is_invariant_to_uniform_volume_scaling():
    """The literal 'same volume always works' proof: scaling the entire mic mix
    (echo + user) by any factor changes neither the verdict nor the incoherent
    fraction -- because coherence and the energy weights both cancel the gain."""
    ref = _make_reference(0.8, seed=3)
    echo_gain = 0.5
    base_mix = _echo_block(ref, gain=echo_gain) + echo_gain * _user(seed=7)

    fracs = []
    for scale in (0.1, 1.0, 10.0):
        det = _fresh()
        _push(det, ref)
        _prime_echo_baseline(det, ref, gain=echo_gain)  # same unscaled priming
        verdict = det.decide(scale * base_mix)
        assert verdict is True, f"barge lost at scale {scale}"
        fracs.append(det.last_incoherent_fraction)
    assert max(fracs) - min(fracs) < 1e-3, f"not scale-invariant: {fracs}"


def test_self_calibrates_margin_to_a_noisy_reverberant_room():
    """In a noisy/reverberant room the echo's own incoherence fluctuates MORE
    (variable per frame), so a fixed margin would false-fire on the high frames.
    The detector learns that spread (an EWMA control chart) and widens its trigger
    to absorb it -- echo-only is suppressed in steady state, the effective margin
    grows well above the floor, the mean/spread are NOT per-room hand-tuned -- yet
    a genuine user still clears it. This is the 'parameters dynamic at runtime'
    requirement, made reliable."""
    rng = np.random.default_rng(11)
    ref = _make_reference(0.8, seed=10)
    floor = 0.08
    det = _fresh(margin_delta=floor)
    _push(det, ref)
    fired = []
    for _ in range(50):
        # Variable reverb/noise level each frame: the model can't fully explain it,
        # so the incoherent fraction fluctuates -- exactly what the chart must learn.
        amp = 0.05 + 0.25 * rng.random()
        noisy = _echo_block(ref, gain=0.6) + amp * rng.standard_normal(BLOCK).astype("float32")
        fired.append(det.decide(noisy))
    # Steady state: once the spread is learned, echo-only no longer self-interrupts.
    assert not any(fired[-20:]), "noisy echo still self-interrupting after adaptation"
    # It auto-widened the trigger well beyond the configured floor...
    assert det.last_effective_margin > floor * 1.5
    # ...yet a genuine user (a large uncorrelated burst) still clears the wider bar.
    user_mix = _echo_block(ref, gain=0.6) + 0.6 * _user(seed=42)
    assert det.decide(user_mix) is True


def test_estimates_the_echo_delay():
    ref = _make_reference(0.8, seed=4)
    det = _fresh()
    _push(det, ref)
    _prime_echo_baseline(det, ref, gain=0.7)
    assert abs(det.last_delay_ms - 1000.0 * DELAY / SR) < (1000.0 * 2 / SR)


def test_decides_none_without_reference_so_caller_falls_back():
    det = _fresh()
    # No playback pushed -> not enough reference -> abstain (caller uses level gate).
    assert det.decide(_user(seed=5)) is None


def test_abstains_when_reference_is_silent():
    det = _fresh()
    _push(det, np.zeros(SR, dtype="float32"))  # playing, but silence
    assert det.decide(_user(seed=6)) is None


# --- consecutive-confirmation: "a bit slower, higher confidence" --------------
# The interrupt (this enrollment-free, identity-free detector) must clear the
# threshold for confirm_frames CONSECUTIVE frames before firing. That rejects a
# one-off over-threshold spike (cheap-speaker nonlinearity / transient noise)
# while a sustained talk-over still fires. While a run is building, decide()
# returns False (echo-only), never None -- so a not-yet-confirmed moment can
# never fall through to the legacy level gate.


def test_confirm_frames_param_is_stored_and_defaults_to_legacy_one():
    assert _fresh()._confirm_frames == 1  # detector default = original behaviour
    assert _fresh(confirm_frames=4)._confirm_frames == 4


def test_confirm_frames_one_is_the_legacy_single_frame_fire():
    ref = _make_reference(0.8, seed=23)
    echo_gain = 0.5
    det = _fresh(confirm_frames=1)
    _push(det, ref)
    _prime_echo_baseline(det, ref, gain=echo_gain)
    mix = _echo_block(ref, gain=echo_gain) + echo_gain * _user(seed=66)
    assert det.decide(mix) is True  # fires on the first over-threshold frame


def test_confirm_frames_requires_consecutive_over_threshold_to_fire():
    ref = _make_reference(0.8, seed=21)
    echo_gain = 0.5
    det = _fresh(confirm_frames=3)
    _push(det, ref)
    _prime_echo_baseline(det, ref, gain=echo_gain)
    mix = _echo_block(ref, gain=echo_gain) + echo_gain * _user(seed=77)
    # Each call is over threshold (last_consec climbs), but only the 3rd
    # consecutive one fires -- and the unconfirmed frames return False, not None.
    v1 = det.decide(mix)
    assert v1 is False and det.last_consec == 1
    v2 = det.decide(mix)
    assert v2 is False and det.last_consec == 2
    v3 = det.decide(mix)
    assert v3 is True and det.last_consec == 3


def test_confirm_frames_run_resets_on_an_intervening_echo_frame():
    ref = _make_reference(0.8, seed=22)
    echo_gain = 0.5
    det = _fresh(confirm_frames=2)
    _push(det, ref)
    _prime_echo_baseline(det, ref, gain=echo_gain)
    mix = _echo_block(ref, gain=echo_gain) + echo_gain * _user(seed=88)
    assert det.decide(mix) is False and det.last_consec == 1  # 1st over-threshold
    # An echo-only frame breaks the run before it can confirm.
    assert det.decide(_echo_block(ref, gain=echo_gain)) is False and det.last_consec == 0
    # A fresh pair of consecutive over-threshold frames is now needed to fire.
    assert det.decide(mix) is False and det.last_consec == 1
    assert det.decide(mix) is True and det.last_consec == 2


def test_reset_clears_the_confirmation_run():
    ref = _make_reference(0.8, seed=24)
    echo_gain = 0.5
    det = _fresh(confirm_frames=2)
    _push(det, ref)
    _prime_echo_baseline(det, ref, gain=echo_gain)
    mix = _echo_block(ref, gain=echo_gain) + echo_gain * _user(seed=55)
    assert det.decide(mix) is False and det.last_consec == 1  # mid-run
    det.reset()  # playback stopped -> new turn starts fresh
    assert det._consec == 0


def test_measured_delay_samples_median_and_none():
    """The engine feeds this to the AEC far-end read for online delay calibration:
    None until any echo-only delay is measured, then the median of recent ones."""
    det = EchoCoherenceDetector(SR)
    assert det.measured_delay_samples() is None
    det._delays.extend([100, 120, 110])
    assert det.measured_delay_samples() == 110


# --- warm-up baseline seeding (control-chart starvation fix, 2026-06-07) --------
# The control chart learns the echo baseline ONLY on below-threshold frames. A
# nonlinear/clipping open speaker can make the echo's incoherent fraction
# PERSISTENTLY exceed the provisional threshold -> the learning branch never runs,
# _baseline stays pinned at provisional_baseline (0.5), and every echo frame reads
# as a barge (a self-interrupt that confirm_frames/sustain only DELAY). Warm-up
# seeds the baseline from the first N echo-bearing frames (echo-only by
# construction) so the chart learns the true echo floor -- whatever it is.


def _stub_decide_inputs(det, monkeypatch, frac):
    """Drive decide() straight to the control chart with a controlled incoherent
    fraction: a big-enough reference snapshot + a fixed delay; _segment/_rms run
    for real on the ones() window (rms 1.0 > min_ref_rms)."""
    win = np.ones(8000, dtype="float32")
    monkeypatch.setattr(det, "_snapshot_ref", lambda: win)
    monkeypatch.setattr(det, "_estimate_delay", lambda x, w: 0)
    monkeypatch.setattr(det, "_incoherent_fraction", frac)


def test_warmup_seeds_baseline_and_prevents_nonlinear_echo_starvation(monkeypatch):
    det = EchoCoherenceDetector(SR, warmup_frames=5, confirm_frames=2, margin_delta=0.08)
    _stub_decide_inputs(det, monkeypatch, lambda x, r: 0.8)  # persistent HIGH echo incoherence
    mic = np.full(160, 0.1, dtype="float32")
    verdicts = [det.decide(mic) for _ in range(20)]
    # Pre-fix this self-interrupted within confirm_frames (0.8 > 0.5+0.08 forever).
    assert all(v is False for v in verdicts), verdicts        # never self-interrupts
    assert det.last_baseline == pytest.approx(0.8, abs=0.05)  # learned the true echo floor


def test_warmup_then_a_clear_outlier_still_fires(monkeypatch):
    det = EchoCoherenceDetector(SR, warmup_frames=3, confirm_frames=2, margin_delta=0.08)
    fracs = iter([0.2, 0.2, 0.2, 0.9, 0.9, 0.9])  # 3 warm-up echo frames, then a barge
    _stub_decide_inputs(det, monkeypatch, lambda x, r: next(fracs))
    mic = np.full(160, 0.1, dtype="float32")
    v = [det.decide(mic) for _ in range(6)]
    assert v[:3] == [False, False, False]   # warm-up: always echo-only
    assert v[3] is False and v[4] is True   # outlier fires after confirm_frames=2 consecutive


def test_reset_re_arms_warmup(monkeypatch):
    det = EchoCoherenceDetector(SR, warmup_frames=2)
    _stub_decide_inputs(det, monkeypatch, lambda x, r: 0.7)
    mic = np.full(160, 0.1, dtype="float32")
    det.decide(mic)
    assert det._warmup_left == 1   # one warm-up frame consumed
    det.reset()
    assert det._warmup_left == 2   # a new speaking run re-seeds the echo floor


def test_warmup_frames_param_is_stored_and_configurable():
    assert EchoCoherenceDetector(SR)._warmup_frames == 5  # default
    assert EchoCoherenceDetector(SR, warmup_frames=0)._warmup_frames == 0  # disable


def test_building_ref_returns_false_not_none():
    """While the reference ring is BUILDING (some played audio, but < the minimum
    for a confident compare) decide() must return False (echo-only), NEVER None --
    else on an open speaker with no AEC the None falls through to the loud-mic
    level gate and self-interrupts (live run-20260618-004500: post-grace cuts at
    0.4-0.64s). A genuinely EMPTY ring (no playback) still returns None so a real
    talk-over during true TTS silence keeps the level-gate fallback."""
    det = EchoCoherenceDetector(SR)
    mic = np.zeros(BLOCK, dtype="float32")
    assert det.decide(mic) is None                       # empty ring -> None (gate)
    det.note_playback(_make_reference(0.1, seed=7), SR)  # SOME played ref (< 0.5s)
    assert det.decide(mic) is False                      # building -> echo-only, NOT None
