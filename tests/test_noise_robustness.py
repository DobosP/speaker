"""
Noise robustness tests.

Real deployments face noise conditions that the synthetic test suite completely
ignores.  These tests expose how the system behaves with realistic background
noise scenarios.

CURRENTLY FAILING (bugs exposed)
---------------------------------

  test_babble_noise_does_not_trigger_barge_in_without_calibration
    BUG: Multi-speaker babble (3 FSDD voices mixed) during ASSISTANT_SPEAKING
    has voiced speech characteristics that Silero correctly classifies as
    voiced=True.  With no noise floor calibration (noise_floor=None), the noise
    gate always passes → Silero score = 2.0 → accumulates → barge-in fires.
    This is a false positive from background TV / people talking in another room.
    FIX: Require noise_floor calibration before enabling barge-in, OR implement
    a secondary check that rejects barge-in if the signal was already present
    before TTS started (pre-roll energy check).

  test_environment_change_causes_false_positive
    BUG: System calibrates to a quiet room (noise_floor=0.01).  Then the
    environment changes: someone turns on the TV (babble at 0.08 RMS).  The
    stale noise_floor=0.01 means min_rms=0.03, which the TV speech (0.08 RMS)
    easily exceeds.  Silero classifies TV speech as voiced → barge-in fires.
    FIX: Implement dynamic noise floor tracking (exponential moving average
    over idle periods) so the calibrated floor follows slow environment changes.

  test_music_with_speech_segment_does_not_fire
    BUG: Some music (singing, speech-over-music, podcasts) has harmonic content
    and voiced characteristics that push Silero toward voiced=True.  A rhythm
    track at 120 bpm has energy bursts every 500 ms.  With soft-decay, multiple
    energy bursts can accumulate barge-in score even without user intent.

CURRENTLY PASSING (regression guards)
--------------------------------------

  test_white_noise_never_fires
    Stationary Gaussian noise at any amplitude must not fire barge-in.
    Silero correctly returns voiced=False for white noise.

  test_babble_blocked_when_calibrated_to_babble_level
    When noise_floor is calibrated to the babble RMS, the babble itself is
    below the noise gate → soft_decay → never accumulates.

  test_plosive_bursts_do_not_fire
    5 plosive-like 25 ms bursts separated by 120 ms gaps must not trigger
    barge-in.  The bursts are not classified as voiced by Silero.

  test_user_speech_over_babble_fires
    When the user speaks over babble background, barge-in must fire.
    The user's voice is louder than the babble → Silero + energy prevail.
"""

from __future__ import annotations

import os
import sys
import time
import unittest

import numpy as np
import pytest

pytestmark = [pytest.mark.discovery, pytest.mark.audio, pytest.mark.slow]

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.fixtures import (
    SR,
    silence,
    voiced_speech,
    tv_noise,
    human_voice,
    human_voice_concat,
    babble_noise,
    nonstationary_noise,
    music_noise,
    plosive_burst,
    snr_mix,
    HUMAN_VOICE_AVAILABLE,
)
from tests.harness import AudioHarness, make_recorder


def _skip_if_no_fsdd():
    if not HUMAN_VOICE_AVAILABLE():
        raise unittest.SkipTest("FSDD voice samples not found in tests/voice_samples/")


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Babble noise (multi-speaker background)
# ═══════════════════════════════════════════════════════════════════════════════

class TestBabbleNoise(unittest.TestCase):
    """Multi-speaker background noise (TV, party, open-plan office)."""

    def setUp(self):
        _skip_if_no_fsdd()

    def test_babble_does_not_fire_without_calibration(self):
        """
        3 FSDD speakers mixed at 0.06 RMS during ASSISTANT_SPEAKING, no noise
        floor calibration (noise_floor=None → gate always passes).

        CURRENTLY FAILS: Silero classifies the mixed babble as voiced=True on
        several frames → score = 2.0 → accumulates → barge-in fires.
        This is a false positive from background TV speech.

        FIX: Set a minimum noise_floor based on pre-TTS calibration, or check
        that the barge-in energy represents a CHANGE relative to the pre-TTS
        background, not just an absolute threshold.
        """
        babble = babble_noise(3.0, amplitude=0.06, num_speakers=3)

        interrupts = []
        rec = make_recorder(on_interrupt=lambda info=None: interrupts.append(info))

        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.inject(babble)
            h.drain(timeout=5.0)

        self.assertEqual(
            len(interrupts), 0,
            f"Babble noise (3 FSDD speakers, RMS≈0.06) must NOT trigger barge-in. "
            f"Got {len(interrupts)} false interrupt(s).  "
            f"This is background TV speech — the user is not speaking.  "
            f"BUG: Silero detects voices in the babble mix → score ≥ 2.0 → fires.  "
            f"FIX: Require calibrated noise_floor or a pre-TTS energy baseline.",
        )

    def test_babble_blocked_when_calibrated_to_babble_level(self):
        """
        When noise_floor is calibrated to the babble RMS, the babble fails the
        noise gate (rms < noise_floor * 3) → soft_decay → never fires.
        """
        babble = babble_noise(2.0, amplitude=0.06, num_speakers=3)
        babble_rms = float(np.sqrt(np.mean(babble ** 2)))

        interrupts = []
        rec = make_recorder(on_interrupt=lambda info=None: interrupts.append(info))
        # Calibrate to the babble level so the gate rejects it
        rec._noise_floor = babble_rms

        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.inject(babble)
            h.drain(timeout=4.0)

        self.assertEqual(
            len(interrupts), 0,
            f"Babble calibrated to its own RMS ({babble_rms:.4f}) must be blocked. "
            f"Got {len(interrupts)} interrupt(s).",
        )

    def test_user_speech_over_babble_fires(self):
        """
        User speaks 8× louder than the babble background.
        Calibrated to babble level → noise gate passes user speech → barge-in fires.
        """
        babble = babble_noise(2.0, amplitude=0.04, num_speakers=3)
        babble_rms = float(np.sqrt(np.mean(babble ** 2)))

        user = human_voice_concat(1.5, amplitude=babble_rms * 8)
        # Mix: user speech starts at 0.3 s
        n = len(babble)
        mic = babble.copy()
        offset = int(0.3 * SR)
        end = min(offset + len(user), n)
        mic[offset:end] += user[: end - offset]

        interrupts = []
        rec = make_recorder(on_interrupt=lambda info=None: interrupts.append(info))
        rec._noise_floor = babble_rms

        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.inject(mic)
            h.drain(timeout=5.0)

        self.assertGreater(
            len(interrupts), 0,
            f"User speech at 8× babble RMS (noise_floor={babble_rms:.4f}) must fire "
            f"barge-in.  Got 0 interrupts.  The noise gate or Silero path may be failing.",
        )

    def test_5s_babble_never_fires_false_positive_when_calibrated(self):
        """
        5 seconds of babble, calibrated noise floor.  Zero false positives.
        This is the regression guard for the noise gate + soft-decay combo.
        """
        babble = babble_noise(5.0, amplitude=0.05, num_speakers=3)
        babble_rms = float(np.sqrt(np.mean(babble ** 2)))

        interrupts = []
        rec = make_recorder(on_interrupt=lambda info=None: interrupts.append(info))
        rec._noise_floor = babble_rms

        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.inject(babble)
            h.drain(timeout=8.0)

        self.assertEqual(
            len(interrupts), 0,
            f"5 s of babble (calibrated) must never fire barge-in.  "
            f"Got {len(interrupts)} false interrupt(s).",
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Non-stationary noise (environment changes after calibration)
# ═══════════════════════════════════════════════════════════════════════════════

class TestNonstationaryNoise(unittest.TestCase):
    """
    The system calibrates at startup, then the environment changes.
    The stale noise floor causes incorrect gate decisions.
    """

    def test_environment_spike_with_babble_causes_false_positive(self):
        """
        Calibrated to quiet (noise_floor=0.01).  TV turns on at 0.08 RMS with
        speech content.  Silero detects the TV speech as voiced → barge-in fires.

        CURRENTLY FAILS: this is the most common real-world false-positive.

        FIX: Implement dynamic noise floor update.  During idle periods
        (between TTS turns), continuously update the noise floor as an
        exponential moving average of the measured RMS.
        """
        _skip_if_no_fsdd()
        # Quiet period followed by TV speech (babble) turning on
        noise = nonstationary_noise(
            3.0,
            quiet_amplitude=0.005,
            loud_amplitude=0.08,
            transition_at=0.4,
        )
        # Replace the loud half with real babble speech to trigger Silero
        babble = babble_noise(3.0, amplitude=0.08, num_speakers=3)
        transition = int(0.4 * 3.0 * SR)
        noise[transition:] = babble[transition:]

        interrupts = []
        rec = make_recorder(on_interrupt=lambda info=None: interrupts.append(info))
        rec._noise_floor = 0.01  # calibrated to quiet
        rec._calibrated = True  # enable dynamic floor rise during TTS (see audio._adapt)

        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.inject(noise)
            h.drain(timeout=6.0)

        self.assertEqual(
            len(interrupts), 0,
            f"Environment change (TV turns on mid-stream) must NOT fire barge-in. "
            f"Got {len(interrupts)} false interrupt(s).  "
            f"BUG: noise_floor=0.01 is stale; TV speech at 0.08 RMS passes the gate.  "
            f"FIX: Dynamically update noise_floor during idle / non-speaking periods.",
        )

    def test_quiet_noise_spike_white_noise_does_not_fire(self):
        """
        Non-stationary white noise (HVAC kicks in at 0.06 RMS, below energy-path
        threshold).  Silero returns voiced=False AND energy-only score < 2.0
        (RMS < 3×vad_threshold=0.03... actually 0.06 > 0.03 → same energy bug).

        This version uses a loud_amplitude=0.025 to stay below 3×threshold=0.03.
        Tests that sub-threshold noise spikes do not fire.
        """
        noise = nonstationary_noise(
            3.0, quiet_amplitude=0.003, loud_amplitude=0.025, transition_at=0.4
        )

        interrupts = []
        rec = make_recorder(on_interrupt=lambda info=None: interrupts.append(info))

        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.inject(noise)
            h.drain(timeout=5.0)

        self.assertEqual(
            len(interrupts), 0,
            f"White noise spike (loud_amplitude=0.025 < 3×vad_threshold=0.03) "
            f"must not fire barge-in.  Got {len(interrupts)} interrupt(s).",
        )

    def test_hvac_noise_spike_fires_energy_path_without_calibration(self):
        """
        HVAC fan noise at 0.12 RMS (>> 3×vad_threshold=0.03) fires barge-in
        via the energy-only path, even though it's not speech.

        CURRENTLY FAILS as expected bug: same root cause as high-amplitude
        white noise.  Documents that the system has a hard dependency on
        noise_floor calibration to distinguish loud non-speech from user speech.

        FIX: Require calibrated noise_floor for energy path to score ≥ 2.0.
        """
        noise = nonstationary_noise(
            3.0, quiet_amplitude=0.005, loud_amplitude=0.12, transition_at=0.4
        )

        interrupts = []
        rec = make_recorder(on_interrupt=lambda info=None: interrupts.append(info))

        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.inject(noise)
            h.drain(timeout=5.0)

        self.assertEqual(
            len(interrupts), 0,
            f"HVAC noise spike (0.12 RMS) must NOT fire barge-in.  "
            f"Got {len(interrupts)}.  "
            f"BUG: energy-only path fires for any signal with RMS > 3×vad_threshold "
            f"(=0.03) when noise_floor is not calibrated.  "
            f"FIX: require noise_floor calibration for energy path.",
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Music and rhythmic noise
# ═══════════════════════════════════════════════════════════════════════════════

class TestMusicNoise(unittest.TestCase):
    """Background music with rhythmic energy and harmonic content."""

    def test_background_music_does_not_fire_without_calibration(self):
        """
        Music at 120 bpm (energy peaks every 500 ms) with harmonic 440 Hz content.
        No noise floor calibration.  Silero should NOT classify music as voiced.

        CURRENTLY FAILING in some configurations: the 440 Hz melody and harmonic
        stack push Silero toward voiced=True on beats → accumulates barge-in score.
        """
        music = music_noise(4.0, amplitude=0.06, beat_hz=2.0)

        interrupts = []
        rec = make_recorder(on_interrupt=lambda info=None: interrupts.append(info))

        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.inject(music)
            h.drain(timeout=7.0)

        self.assertEqual(
            len(interrupts), 0,
            f"Background music must NOT trigger barge-in.  "
            f"Got {len(interrupts)} false interrupt(s).  "
            f"BUG: Music harmonic content can push Silero voiced fraction toward voiced.  "
            f"FIX: Add minimum voiced fraction gate or require noise_floor calibration.",
        )

    def test_music_calibrated_does_not_fire(self):
        """Music at its own RMS level as noise_floor — always blocked."""
        music = music_noise(3.0, amplitude=0.05, beat_hz=2.0)
        music_rms = float(np.sqrt(np.mean(music ** 2)))

        interrupts = []
        rec = make_recorder(on_interrupt=lambda info=None: interrupts.append(info))
        rec._noise_floor = music_rms

        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.inject(music)
            h.drain(timeout=5.0)

        self.assertEqual(
            len(interrupts), 0,
            f"Music calibrated to its own level must be blocked. "
            f"Got {len(interrupts)} interrupt(s).",
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  4. Transient sounds (plosives, door slams, keyboard)
# ═══════════════════════════════════════════════════════════════════════════════

class TestTransientSounds(unittest.TestCase):
    """Brief, high-energy non-speech events must not trigger barge-in."""

    def test_plosive_bursts_do_not_fire(self):
        """
        5 plosive-like bursts (25 ms each, 120 ms gap) must not fire barge-in.
        Each burst passes the noise gate but Silero returns voiced=False.
        Hard-reset in update() resets accumulator after each gap.
        """
        bursts = plosive_burst(count=5, burst_ms=25.0, gap_ms=120.0, amplitude=0.30)

        interrupts = []
        rec = make_recorder(on_interrupt=lambda info=None: interrupts.append(info))

        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.inject(bursts)
            h.drain(timeout=4.0)

        self.assertEqual(
            len(interrupts), 0,
            f"5 plosive bursts must not fire barge-in.  Got {len(interrupts)}.  "
            f"Bursts are broadband noise — Silero does not score them as voiced.",
        )

    def test_many_plosives_do_not_accumulate_with_soft_decay(self):
        """
        10 plosive bursts in 2 seconds — stress test for the soft-decay path.

        At amplitude=0.35 (RMS >> 3×vad_threshold), the energy-only path scores
        2.5 per burst.  With hard-reset in update() for passed-gate frames that
        score ≥ 2.0... wait, 2.5 ≥ 2.0 → accumulates!  This DOES fire.

        CURRENTLY FAILS: same energy-path-without-calibration bug.  High-amplitude
        broadband bursts score 2.5 (energy-only) which is ≥ 2.0 → they accumulate
        just like voiced speech would.

        FIX: Require Silero voiced=True for energy score to contribute toward
        the 2.0 threshold, or require noise_floor calibration.
        """
        bursts = plosive_burst(count=10, burst_ms=20.0, gap_ms=80.0, amplitude=0.35)

        interrupts = []
        rec = make_recorder(on_interrupt=lambda info=None: interrupts.append(info))

        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.inject(bursts)
            h.drain(timeout=4.0)

        self.assertEqual(
            len(interrupts), 0,
            f"10 plosive bursts (broadband, not voiced) must not fire barge-in.  "
            f"Got {len(interrupts)}.  "
            f"BUG: energy-only score 2.5 ≥ 2.0 → fires even for non-speech bursts "
            f"when no noise_floor calibration is active.  "
            f"FIX: require Silero vote OR noise_floor calibration for score ≥ 2.0.",
        )

    def test_white_noise_silero_rejects_as_unvoiced(self):
        """
        Silero must not classify white noise as voiced — tested at moderate amplitudes
        where the energy-only path cannot fire (RMS < 3 × vad_threshold = 0.03).
        """
        for amp in (0.005, 0.01, 0.02):
            noise = tv_noise(2.0, amplitude=amp, seed=int(amp * 10000))
            interrupts = []
            rec = make_recorder(on_interrupt=lambda info=None: interrupts.append(info))
            with AudioHarness(rec) as h:
                h.set_tts_speaking()
                h.inject(noise)
                h.drain(timeout=4.0)
            self.assertEqual(
                len(interrupts), 0,
                f"White noise at amplitude={amp} (below 3×vad_threshold=0.03) "
                f"must not fire barge-in.  Got {len(interrupts)}.",
            )

    def test_high_amplitude_noise_fires_energy_path_without_calibration(self):
        """
        White noise at high amplitude (RMS > 3 × vad_threshold = 0.03) fires
        barge-in via the ENERGY-ONLY PATH even though Silero returns voiced=False.

        CURRENTLY FAILS as a real-world bug: without noise_floor calibration,
        score = 2.5 (energy_ratio ≥ 3.0) ≥ 2.0 → accumulates → fires.
        This means the assistant self-triggers on loud audio playback, fan noise,
        or any high-amplitude non-speech signal.

        FIX: Require either voiced=True (Silero) OR a calibrated noise_floor
        before the energy-only path can contribute score ≥ 2.0.  Without
        calibration, raise the energy threshold so score < 2.0 unless Silero
        also votes voiced.
        """
        for amp in (0.10, 0.20, 0.40):
            noise = tv_noise(2.0, amplitude=amp, seed=int(amp * 100))
            interrupts = []
            rec = make_recorder(on_interrupt=lambda info=None: interrupts.append(info))
            with AudioHarness(rec) as h:
                h.set_tts_speaking()
                h.inject(noise)
                h.drain(timeout=4.0)
            self.assertEqual(
                len(interrupts), 0,
                f"White noise at amplitude={amp} (RMS >> 3×vad_threshold) "
                f"must NOT fire barge-in even without noise_floor calibration.  "
                f"Got {len(interrupts)}.  "
                f"BUG: energy-only score = 2.5 ≥ 2.0 → fires without Silero vote.  "
                f"FIX: Require noise_floor calibration for energy path to score ≥ 2.0.",
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  5. SNR boundary tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSNRBoundary(unittest.TestCase):
    """User speech vs. noise at defined SNR levels."""

    def setUp(self):
        _skip_if_no_fsdd()

    def test_speech_at_plus_12db_snr_fires(self):
        """
        Real human speech clearly dominant (+12 dB) over white noise must fire
        barge-in.

        In production the noise floor is calibrated before the user speaks.
        Both the Silero path and the calibrated energy path can accumulate,
        providing redundancy at high SNR.
        """
        speech = human_voice_concat(2.5, amplitude=0.20)
        noise = tv_noise(2.5, amplitude=0.04, seed=1)
        mixed = snr_mix(speech, noise, snr_db=12.0)

        interrupts = []
        rec = make_recorder(on_interrupt=lambda info=None: interrupts.append(info))
        rec._noise_floor = 0.005  # Simulate calibrated environment.

        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.inject(mixed)
            h.drain(timeout=6.0, settle=0.3)

        self.assertGreater(
            len(interrupts), 0,
            "Speech at +12 dB SNR must fire barge-in.",
        )

    def test_speech_at_plus_6db_snr_fires(self):
        """
        Real human speech at +6 dB SNR over white noise must fire barge-in.

        In production the noise floor is calibrated before the user speaks
        (the system runs silently for at least a second during IDLE).  The
        calibrated energy path supplements Silero on noisy frames where
        the neural model takes a few chunks to warm up after a GRU reset.
        """
        speech = human_voice_concat(2.5, amplitude=0.20)
        noise = tv_noise(2.5, amplitude=0.04, seed=2)
        mixed = snr_mix(speech, noise, snr_db=6.0)

        interrupts = []
        rec = make_recorder(on_interrupt=lambda info=None: interrupts.append(info))
        # Simulate calibrated environment: system heard ambient noise before user speaks.
        rec._noise_floor = 0.005

        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.inject(mixed)
            h.drain(timeout=6.0, settle=0.3)

        self.assertGreater(
            len(interrupts), 0,
            "Speech at +6 dB SNR over white noise must fire barge-in via Silero path.",
        )

    def test_noise_only_5s_never_fires_when_calibrated(self):
        """
        5 seconds of white noise, calibrated to that noise level.
        The noise gate blocks all frames → soft_decay → zero false positives.
        This is the correct operational mode (calibration required).
        """
        for amp in (0.04, 0.10, 0.20):
            noise = tv_noise(5.0, amplitude=amp, seed=99)
            noise_rms = float(np.sqrt(np.mean(noise ** 2)))
            interrupts = []
            rec = make_recorder(on_interrupt=lambda info=None: interrupts.append(info))
            rec._noise_floor = noise_rms  # calibrated to noise level
            with AudioHarness(rec) as h:
                h.set_tts_speaking()
                h.inject(noise)
                h.drain(timeout=8.0)
            self.assertEqual(
                len(interrupts), 0,
                f"5 s white noise (calibrated, amp={amp}) must produce zero interrupts. "
                f"Got {len(interrupts)}.",
            )

    def test_noise_only_5s_fires_without_calibration_energy_bug(self):
        """
        Same white noise WITHOUT calibration → fires at high amplitudes.

        CURRENTLY FAILS (real-world bug): documents the hard requirement for
        noise_floor calibration before the system can be used reliably.

        All amplitudes > 3×vad_threshold (=0.03) fire via energy-only path.
        FIX: Require noise_floor to be set before energy path is enabled.
        """
        for amp in (0.04, 0.10, 0.20):
            noise = tv_noise(5.0, amplitude=amp, seed=99)
            interrupts = []
            rec = make_recorder(on_interrupt=lambda info=None: interrupts.append(info))
            # No noise_floor calibration
            with AudioHarness(rec) as h:
                h.set_tts_speaking()
                h.inject(noise)
                h.drain(timeout=8.0)
            self.assertEqual(
                len(interrupts), 0,
                f"5 s white noise at amp={amp} (no calibration) must NOT fire.  "
                f"Got {len(interrupts)}.  "
                f"BUG: RMS {amp:.3f} > 3×vad_threshold=0.03 → score=2.5 → fires.  "
                f"FIX: energy path must require noise_floor calibration.",
            )
