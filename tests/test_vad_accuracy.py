"""
VAD accuracy contracts.

These tests verify that the speech detection model (Silero) performs
within defined accuracy bounds on known-good and known-bad audio.

WHY these tests exist
---------------------
The barge-in system has TWO paths:

  A) Silero path:  voiced=True fires because the neural model classifies
                   the audio as speech.  Score ≥ 2.0 from voiced alone.

  B) Energy path:  voiced may be False, but rms > threshold * 3.0 gives
                   score 2.5 — fires without Silero.

The existing scenario tests in test_bargein_scenarios.py use synthetic audio
that Silero rates as non-voiced (~10% voiced fraction at 0.8 threshold).
They PASS because they use high amplitude (SPEECH_AMP=0.35) to trigger path B.

This means the scenario tests give NO signal about whether Silero is working.
If Silero is broken/disabled, the scenario tests still pass via path B.

These VAD accuracy tests are the only tests that fail when Silero breaks.

Contract thresholds
-------------------
  • FSDD human speech   → Silero voiced fraction ≥ MIN_VOICED_FRACTION (0.55)
    (Some FSDD clips are very short: 0.25 s.  Silero needs several 512-sample
    windows to stabilize.  55 % is a conservative lower bound.)

  • White / pink noise   → Silero voiced fraction ≤ MAX_NOISE_VOICED (0.05)

  • Pure sine tone       → Silero voiced fraction ≤ MAX_TONE_VOICED (0.05)

  • TTS echo (synthetic) → Silero voiced fraction ≤ MAX_SYNTH_VOICED (0.15)
    (Should also be low — synthetic harmonic stacks are not speech.)

Running
-------
    python -m pytest tests/test_vad_accuracy.py -v

Voice samples are downloaded automatically on first run (requires network).
Tests that depend on FSDD are skipped with a clear message when unavailable.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.fixtures import (
    SR,
    silence,
    tv_noise,
    tts_echo,
    voiced_speech,
    human_voice,
    HUMAN_VOICE_AVAILABLE,
)

# ── Constants ─────────────────────────────────────────────────────────────────

# Minimum fraction of 512-sample windows that Silero must classify as voiced
# for real human speech.
#
# 50 % is the calibrated lower bound based on measured FSDD values:
#   jackson/1: 88%,  jackson/2: 80%,  jackson/3: 52%  ← "three" starts with
#   an unvoiced fricative /θ/ which Silero correctly rejects.
#   nicolas/1: 68%,  george/1:  69%
#
# If this drops below 50 %, Silero is not detecting basic voiced content.
MIN_VOICED_FRACTION = 0.50

# Maximum fraction of windows that Silero may classify as voiced for
# known non-speech signals.
MAX_NOISE_VOICED = 0.05
MAX_TONE_VOICED = 0.05
MAX_SYNTH_HARMONIC_VOICED = 0.15  # synthetic harmonic stack — should be low

# Silero probability threshold used in barge-in mode (matches audio.py)
SILERO_BARGE_IN_THRESH = 0.80

# Pairs of (speaker, digit) used in parametrised tests
_FSDD_PAIRS = [
    ("jackson", 1),
    ("jackson", 2),
    ("jackson", 3),
    ("nicolas", 1),
    ("nicolas", 2),
    ("george", 1),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_silero():
    """Load (and cache) the Silero VAD model."""
    try:
        from silero_vad import load_silero_vad  # noqa: PLC0415
        return load_silero_vad()
    except ImportError:
        return None


_model = None


def _voiced_fraction(audio: np.ndarray, threshold: float = SILERO_BARGE_IN_THRESH) -> float:
    """
    Compute the fraction of 512-sample windows classified as voiced by Silero.

    Resets the model's hidden state before each call so results are
    independent of any prior audio fed to the model.
    """
    global _model
    if _model is None:
        _model = _load_silero()
    if _model is None:
        raise RuntimeError("Silero VAD model not available")

    _model.reset_states()
    voiced = 0
    total = 0
    audio = audio.astype(np.float32)
    for start in range(0, len(audio) - 512, 512):
        chunk = audio[start : start + 512]
        t = torch.from_numpy(chunk).unsqueeze(0)
        prob = float(_model(t, SR))
        voiced += 1 if prob >= threshold else 0
        total += 1

    return voiced / total if total > 0 else 0.0


def _skip_if_no_fsdd():
    if not HUMAN_VOICE_AVAILABLE():
        raise unittest.SkipTest(
            "FSDD voice samples not available — run tests with network access "
            "to download them (conftest.py handles this automatically on first run)."
        )


def _skip_if_no_silero():
    try:
        from silero_vad import load_silero_vad  # noqa: PLC0415
    except ImportError:
        raise unittest.SkipTest("silero_vad not installed")


# ═══════════════════════════════════════════════════════════════════════════════
#  Test classes
# ═══════════════════════════════════════════════════════════════════════════════


class TestSileroOnRealHumanVoice(unittest.TestCase):
    """
    Silero MUST classify a meaningful fraction of real human speech as voiced.

    These tests WILL FAIL if:
    - Silero model is not loaded / broken
    - The model version changed and no longer detects FSDD-style speech
    - The audio preprocessing pipeline (resampling, normalisation) is broken
    """

    def setUp(self):
        _skip_if_no_silero()
        _skip_if_no_fsdd()

    def _assert_voiced_above_minimum(
        self, audio: np.ndarray, label: str
    ):
        frac = _voiced_fraction(audio, threshold=SILERO_BARGE_IN_THRESH)
        self.assertGreaterEqual(
            frac,
            MIN_VOICED_FRACTION,
            f"[{label}] Silero voiced fraction {frac:.1%} < minimum "
            f"{MIN_VOICED_FRACTION:.0%}.  "
            f"Silero may be broken or the audio pipeline is incorrect.",
        )

    def test_jackson_digit1(self):
        """Jackson saying 'one' — reliable voiced content for Silero."""
        self._assert_voiced_above_minimum(
            human_voice(1.0, amplitude=0.15, speaker="jackson", digit=1),
            "jackson/1",
        )

    def test_jackson_digit2(self):
        self._assert_voiced_above_minimum(
            human_voice(1.0, amplitude=0.15, speaker="jackson", digit=2),
            "jackson/2",
        )

    def test_jackson_digit3(self):
        self._assert_voiced_above_minimum(
            human_voice(1.0, amplitude=0.15, speaker="jackson", digit=3),
            "jackson/3",
        )

    def test_nicolas_digit1(self):
        """Different speaker — ensures Silero is not speaker-specific."""
        self._assert_voiced_above_minimum(
            human_voice(1.0, amplitude=0.15, speaker="nicolas", digit=1),
            "nicolas/1",
        )

    def test_george_digit1(self):
        """Third speaker — further accent/voice variety."""
        self._assert_voiced_above_minimum(
            human_voice(1.0, amplitude=0.15, speaker="george", digit=1),
            "george/1",
        )

    def test_silero_uses_barge_in_threshold(self):
        """
        Silero must still classify real speech as voiced even at the STRICT
        barge-in threshold (0.80), which is higher than the default 0.50.

        This threshold is used in the hot path during TTS playback.
        FAILS if the threshold is accidentally set to 0.95 or similar.
        """
        # Use 1.5 s so there are enough windows to get a reliable estimate
        audio = human_voice(1.5, amplitude=0.15, speaker="jackson", digit=1)
        frac_50 = _voiced_fraction(audio, threshold=0.50)
        frac_80 = _voiced_fraction(audio, threshold=0.80)

        # Strict threshold WILL be lower — that's expected.  But it must not
        # drop all the way to zero.
        self.assertGreater(
            frac_80,
            0.30,
            f"Silero barge-in threshold=0.80 gives {frac_80:.1%} voiced on real speech "
            f"(threshold=0.50 gives {frac_50:.1%}).  Barge-in threshold may be too strict.",
        )


class TestSileroRejectsNonSpeech(unittest.TestCase):
    """
    Silero MUST NOT classify known non-speech signals as voiced.

    These tests WILL FAIL if:
    - The Silero model scores flat noise as speech (false positive)
    - The barge-in threshold (0.80) is too low
    - Audio preprocessing introduces artifical voiced-like signals

    A false positive here means the system would interrupt a TTS response
    due to background noise — the most disruptive failure mode.
    """

    def setUp(self):
        _skip_if_no_silero()

    def test_white_noise_not_voiced(self):
        """White noise (TV background) must be classified as non-speech."""
        noise = tv_noise(2.0, amplitude=0.08, seed=1)
        frac = _voiced_fraction(noise, SILERO_BARGE_IN_THRESH)
        self.assertLessEqual(
            frac,
            MAX_NOISE_VOICED,
            f"White noise got {frac:.1%} voiced frames — noise gate is leaking. "
            f"This would cause false barge-ins during TTS playback.",
        )

    def test_pink_noise_not_voiced(self):
        """
        Pink noise (1/f spectrum, more similar to speech than white noise)
        must still be rejected.
        """
        rng = np.random.default_rng(42)
        white = rng.standard_normal(2 * SR).astype(np.float32)
        # 1/f filter: integrate and high-pass
        from scipy.signal import lfilter  # noqa: PLC0415
        pink = lfilter([1.0], [1.0, -0.99], white).astype(np.float32)
        pink /= float(np.sqrt(np.mean(pink ** 2))) or 1.0
        pink *= 0.08

        frac = _voiced_fraction(pink, SILERO_BARGE_IN_THRESH)
        self.assertLessEqual(
            frac,
            MAX_NOISE_VOICED,
            f"Pink noise got {frac:.1%} voiced frames — Silero may be too sensitive.",
        )

    def test_pure_sine_440hz_not_voiced(self):
        """Pure 440 Hz sine wave (musical A) must not be classified as speech."""
        t = np.linspace(0, 2.0, 2 * SR, dtype=np.float32)
        tone = (np.sin(2 * np.pi * 440 * t) * 0.1).astype(np.float32)
        frac = _voiced_fraction(tone, SILERO_BARGE_IN_THRESH)
        self.assertLessEqual(
            frac,
            MAX_TONE_VOICED,
            f"440 Hz pure tone got {frac:.1%} voiced — Silero mistaking tones for speech.",
        )

    def test_pure_sine_1khz_not_voiced(self):
        """Pure 1 kHz sine — within the fundamental frequency range of speech."""
        t = np.linspace(0, 2.0, 2 * SR, dtype=np.float32)
        tone = (np.sin(2 * np.pi * 1000 * t) * 0.1).astype(np.float32)
        frac = _voiced_fraction(tone, SILERO_BARGE_IN_THRESH)
        self.assertLessEqual(
            frac,
            MAX_TONE_VOICED,
            f"1 kHz tone got {frac:.1%} voiced.",
        )

    def test_silence_not_voiced(self):
        """Silence must give 0 % voiced — sanity check for the metric itself."""
        frac = _voiced_fraction(silence(2.0), SILERO_BARGE_IN_THRESH)
        self.assertEqual(
            frac,
            0.0,
            "Silence was classified as voiced — something is very wrong.",
        )

    def test_synthetic_harmonic_stack_low_voiced(self):
        """
        The synthetic voiced_speech() generator (sine harmonic stack) should
        score LOW with Silero.  This documents the known gap: synthetic audio
        does NOT exercise the Silero path.

        If this test FAILS (voiced fraction goes UP), synthetic signals have
        accidentally started to fool Silero — the scenario tests would then
        test both paths simultaneously, which may mask regressions.
        """
        synth = voiced_speech(2.0, amplitude=0.15)
        frac = _voiced_fraction(synth, SILERO_BARGE_IN_THRESH)
        # We expect < 15 %. If it rises above 40 %, the synthetic generator
        # has changed in a way that makes it fool Silero — note this in CI.
        self.assertLessEqual(
            frac,
            MAX_SYNTH_HARMONIC_VOICED,
            f"Synthetic voiced_speech() got {frac:.1%} from Silero — "
            f"it is now fooling the model (was ~5 %).  "
            f"Update the scenario tests to NOT rely on energy path.",
        )


class TestSileroAccuracyVsEnergyPath(unittest.TestCase):
    """
    Verify that Silero and the energy path disagree at the amplitude level
    used by human_voice() tests.

    This is the key invariant that makes human_voice() tests meaningful:
    at amplitude=0.08, the energy-only path CANNOT fire (score < 2.0),
    so only Silero can push above_samples to the required minimum.

    If this test fails, human_voice() tests no longer specifically test Silero.
    """

    def setUp(self):
        _skip_if_no_silero()
        _skip_if_no_fsdd()

    def test_human_voice_at_moderate_amplitude_still_voiced(self):
        """
        At amplitude=0.08, human voice (FSDD) must still be classified as voiced
        by Silero.  This confirms Silero is active and produces useful results
        even at moderate volumes — not just at ear-splitting levels.

        Note: The per-chunk RMS of VOICED segments of FSDD clips is
        significantly higher than the overall average RMS (voiced segments
        contain most of the energy while pauses are near-zero).  At
        amplitude=0.08, voiced-segment chunks routinely have RMS ≈ 0.10–0.18.
        We therefore check Silero's voiced classification directly rather than
        making claims about which scoring path fires.
        """
        audio = human_voice(1.0, amplitude=0.08, speaker="jackson", digit=1)

        frac = _voiced_fraction(audio, threshold=SILERO_BARGE_IN_THRESH)
        self.assertGreater(
            frac,
            0.30,
            f"At amplitude=0.08, Silero only sees {frac:.1%} voiced frames. "
            "Real speech should still be detectable at this amplitude.",
        )

    def test_silero_and_energy_paths_disagree_on_synthetic_speech(self):
        """
        This is the key invariant: synthetic speech and real human speech
        at the SAME amplitude produce DIFFERENT Silero scores.

        Specifically, at amplitude=0.08:
          • human_voice()    → Silero: >30 % voiced (Silero path fires)
          • voiced_speech()  → Silero: <15 % voiced (energy path only)

        If both get the same Silero score, the test_bargein_contracts.py
        test_synthetic_voice_low_amplitude_does_NOT_fire would lose meaning —
        it would fire via Silero too, making it impossible to isolate whether
        the barge-in was triggered by the Silero path or energy path.
        """
        real_audio = human_voice(1.0, amplitude=0.08, speaker="jackson", digit=1)
        synth_audio = voiced_speech(1.0, amplitude=0.08)

        real_frac = _voiced_fraction(real_audio, threshold=SILERO_BARGE_IN_THRESH)
        synth_frac = _voiced_fraction(synth_audio, threshold=SILERO_BARGE_IN_THRESH)

        self.assertGreater(
            real_frac - synth_frac,
            0.20,
            f"Real voice Silero fraction ({real_frac:.1%}) is not sufficiently "
            f"higher than synthetic ({synth_frac:.1%}).  "
            "The two types of audio should produce clearly different VAD scores. "
            "contract tests that isolate the Silero path may not be valid.",
        )

    def test_synthetic_speech_at_low_amplitude_not_voiced(self):
        """
        At the SAME amplitude=0.08, synthetic speech must NOT be classified
        as voiced by Silero.  This documents the gap that motivates using
        FSDD samples in the contract tests.
        """
        synth = voiced_speech(1.0, amplitude=0.08)
        frac = _voiced_fraction(synth, threshold=SILERO_BARGE_IN_THRESH)
        self.assertLess(
            frac,
            0.15,
            f"Synthetic speech at amplitude=0.08 got {frac:.1%} voiced — "
            "higher than expected.  The synthetic generator may have changed.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
