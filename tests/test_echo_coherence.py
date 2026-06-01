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
