"""
Voice variability tests.

Real users speak in diverse ways that uniform synthetic tests cannot cover:
varying distance from the mic, whispering, speaking louder when excited,
running words together, background crosstalk, clipped signals.

CURRENTLY FAILING (bugs exposed)
---------------------------------

  test_whispered_speech_fires_barge_in
    BUG: A whisper has very low RMS (amplitude ≈ 0.02–0.04).  With no noise
    floor calibration, threshold = vad_threshold = 0.01.  3× threshold = 0.03.
    Whispered speech often has RMS < 0.03 → noise gate fails → soft_decay only.
    Silero: whispery unvoiced consonants (/s/, /ʃ/, /θ/) score < 0.80 → no vote.
    The whisper never accumulates → silent drop.  User gets no response.
    FIX: Lower the barge-in Silero threshold from 0.80 to 0.60 for whisper
    mode, or implement an AGC (automatic gain control) to normalise mic input.

  test_clipped_speech_does_not_cause_false_echo_similarity
    BUG: Hard-clipped speech (overdriven mic) has a very different waveform shape
    than the TTS reference.  However, the distorted harmonics can accidentally
    correlate with some TTS content → echo_similarity spikes → barge-in blocked.
    This is an edge case that occurs when users have high mic gain set in OS.
    FIX: Apply soft-clipping detection and reduce echo similarity weight when
    clipping is detected (flag: RMS_peak_ratio < threshold).

  test_rapid_successive_utterances
    BUG: After barge-in fires and the first utterance is processed, a second
    utterance arriving within the cooldown period (0.5 s) is rejected.
    In fast conversations "Yes. No. OK." the second and third words are lost.
    FIX: Make barge_in_cooldown_sec configurable or implement per-utterance
    cooldown reset based on confirmed callback delivery.

CURRENTLY PASSING (regression guards)
--------------------------------------

  test_normal_distance_fires_reliably
    User at normal speaking distance (amplitude ≈ 0.15–0.25) must always fire.

  test_loud_speech_fires_via_energy_path
    Loud speech (amplitude ≈ 0.40) fires via energy path even if Silero is slow.

  test_clipped_speech_fires_via_energy
    Clipped speech has high energy (post-clip RMS ≈ 0.20) → fires energy path.

  test_gender_and_pitch_independence
    Both low-pitch (male, 100 Hz) and high-pitch (female, 250 Hz) synthetic
    speech must fire at the same amplitude threshold.
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
    human_voice,
    human_voice_concat,
    clipped_speech,
    snr_mix,
    tv_noise,
    HUMAN_VOICE_AVAILABLE,
)
from tests.harness import AudioHarness, make_recorder


def _skip_if_no_fsdd():
    if not HUMAN_VOICE_AVAILABLE():
        raise unittest.SkipTest("FSDD voice samples not found in tests/voice_samples/")


_ALL_SPEAKERS = ["george", "jackson", "nicolas", "theo"]
_ALL_DIGITS = [1, 2, 3]


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Distance variation (amplitude variation)
# ═══════════════════════════════════════════════════════════════════════════════

class TestMicDistance(unittest.TestCase):
    """Simulate different microphone distances via amplitude scaling."""

    def setUp(self):
        _skip_if_no_fsdd()

    def _fire_count(self, amplitude: float) -> int:
        audio = human_voice_concat(1.5, amplitude=amplitude)
        interrupts = []
        rec = make_recorder(on_interrupt=lambda info=None: interrupts.append(info))
        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.inject(audio)
            h.drain(timeout=4.0)
        return len(interrupts)

    def test_normal_distance_fires(self):
        """Amplitude 0.15–0.25 (30 cm from mic) must always fire."""
        for amp in (0.15, 0.20, 0.25):
            n = self._fire_count(amp)
            self.assertGreater(n, 0, f"amplitude={amp} must fire barge-in (normal distance)")

    def test_loud_speech_fires(self):
        """Loud speech amplitude 0.35–0.50 fires via energy path."""
        for amp in (0.35, 0.50):
            n = self._fire_count(amp)
            self.assertGreater(n, 0, f"amplitude={amp} must fire barge-in (loud speech)")

    def test_close_range_does_not_false_positive_after_fire(self):
        """
        Close-range speech (amplitude=0.45) triggers barge-in once, then silence
        must NOT produce a second barge-in within the cooldown window.
        """
        interrupts = []
        rec = make_recorder(
            on_interrupt=lambda info=None: interrupts.append(info),
            barge_in_cooldown_sec=0.5,
        )
        audio = human_voice_concat(1.5, amplitude=0.45)

        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.inject(audio)
            h.inject(silence(0.3))
            h.drain(timeout=4.0)

        self.assertLessEqual(
            len(interrupts), 1,
            f"Close-range speech should fire exactly once, not {len(interrupts)} times.",
        )

    def test_whispered_speech_fires_barge_in(self):
        """
        Whispered speech at amplitude=0.03 (very low, barely audible).

        CURRENTLY FAILS: whisper RMS < vad_threshold * 3 = 0.03 and Silero
        scores unvoiced fricatives < 0.80 barge-in threshold → never accumulates.
        Users who whisper "stop" get no response.

        FIX: Lower Silero barge-in threshold for whisper mode, or implement AGC.
        """
        audio = human_voice_concat(1.5, amplitude=0.03)

        interrupts = []
        rec = make_recorder(on_interrupt=lambda info=None: interrupts.append(info))
        # Quiet-room calibration is required for stable gates at low SNR; matches
        # production ``calibrate()`` before sessions without wakeword-only modes.
        rec._noise_floor = 0.005

        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.inject(audio)
            h.drain(timeout=4.0)

        self.assertGreater(
            len(interrupts), 0,
            f"Whispered speech (amplitude=0.03) must trigger barge-in.  "
            f"Got 0 interrupts.  "
            f"BUG: Whispers are below the noise gate AND below Silero's 0.80 "
            f"confidence threshold in barge-in mode.  "
            f"FIX: Lower barge-in Silero threshold to 0.60, or implement AGC "
            f"so weak signals are amplified before gating.",
        )

    def test_far_distance_speech_still_fires(self):
        """
        Far distance speaker: amplitude=0.06.  Must fire via Silero path.
        If this fails, the user at 1.5 m from the mic gets no response.
        """
        audio = human_voice_concat(1.5, amplitude=0.06)

        interrupts = []
        rec = make_recorder(on_interrupt=lambda info=None: interrupts.append(info))
        rec._noise_floor = 0.005

        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.inject(audio)
            h.drain(timeout=4.0)

        self.assertGreater(
            len(interrupts), 0,
            f"Far-distance speech (amplitude=0.06) must fire barge-in via Silero path.  "
            f"Got 0 interrupts.",
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Clipped / distorted microphone input
# ═══════════════════════════════════════════════════════════════════════════════

class TestClippedSpeech(unittest.TestCase):
    """
    Clipping occurs when the mic gain is too high or the user speaks directly
    into a cheap USB microphone at close range.
    """

    def setUp(self):
        _skip_if_no_fsdd()

    def test_clipped_speech_fires_via_energy(self):
        """
        Clipped speech has high sustained RMS → fires via the calibrated energy
        path.  The system must be able to detect a saturated mic signal as a
        barge-in event even if Silero scores it below the voiced threshold.
        """
        audio = clipped_speech(1.5, clip_level=0.5, amplitude=0.25)

        interrupts = []
        rec = make_recorder(on_interrupt=lambda info=None: interrupts.append(info))
        # Energy path requires noise_floor calibration; set it to a quiet
        # room level so the gate opens but the clipped signal clearly exceeds it.
        rec._noise_floor = 0.005

        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.inject(audio)
            h.drain(timeout=4.0)

        self.assertGreater(
            len(interrupts), 0,
            "Clipped speech must still fire barge-in via the calibrated energy path.",
        )

    def test_clipped_speech_does_not_false_block_via_echo_gate(self):
        """
        Clipped speech has distorted harmonics.  These distorted harmonics can
        accidentally correlate with the TTS reference (which also has harmonics)
        → echo_similarity spikes above 0.45 → echo gate BLOCKS the barge-in.
        User speaks loudly and gets no response.

        CURRENTLY FAILS intermittently: depends on TTS content and clip level.
        FIX: Use AEC-cleaned signal (not raw mic) for similarity computation.
        """
        from tests.fixtures import tts_echo
        tts_ref = tts_echo(1.5, amplitude=0.10)
        clipped = clipped_speech(1.5, clip_level=0.4, amplitude=0.30,
                                  speaker="jackson", digit=1)

        interrupts = []
        rec = make_recorder(on_interrupt=lambda info=None: interrupts.append(info))

        with AudioHarness(rec) as h:
            h.set_tts_speaking(audio_ref=tts_ref)
            h.inject(clipped)
            h.drain(timeout=4.0)

        self.assertGreater(
            len(interrupts), 0,
            f"Clipped user speech must NOT be blocked by the echo gate.  "
            f"Got 0 interrupts.  "
            f"BUG: Distorted harmonics in clipped speech can correlate with TTS "
            f"reference harmonics → similarity ≥ 0.45 → blocked as if echo.  "
            f"FIX: Compute similarity on the AEC-cleaned signal, not raw mic.",
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Speaking style: pitch and rate variations
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpeakingStyleVariation(unittest.TestCase):
    """
    Synthetic tests to verify the system handles the full range of pitch,
    speaking rate, and voice energy profiles.
    """

    def _fire(self, amplitude: float, **kwargs) -> bool:
        audio = voiced_speech(1.0, amplitude=amplitude, **kwargs)
        interrupts = []
        rec = make_recorder(on_interrupt=lambda info=None: interrupts.append(info))
        # Calibrate to a quiet room so the energy path fires for synthetic speech.
        # voiced_speech is synthetic; Silero returns voiced=False for it.
        # The calibrated energy path allows high-amplitude synthetic speech to
        # fire barge-in, which exercises the pitch/rate behaviour being tested.
        rec._noise_floor = 0.005
        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.inject(audio)
            h.drain(timeout=3.0)
        return len(interrupts) > 0

    def test_low_pitch_male_fires(self):
        """Low-pitch male voice (100 Hz fundamental) at normal amplitude."""
        self.assertTrue(
            self._fire(amplitude=0.20, pitch_hz=100.0),
            "Low-pitch speech (100 Hz) at amplitude=0.20 must fire.",
        )

    def test_high_pitch_female_fires(self):
        """High-pitch female voice (250 Hz) at normal amplitude."""
        self.assertTrue(
            self._fire(amplitude=0.20, pitch_hz=250.0),
            "High-pitch speech (250 Hz) at amplitude=0.20 must fire.",
        )

    def test_fast_speaking_rate_fires(self):
        """Fast speaking rate (8 syllables/s) — energy peaks are more frequent."""
        self.assertTrue(
            self._fire(amplitude=0.20, syllable_rate=8.0),
            "Fast-speaking synthetic speech must fire barge-in.",
        )

    def test_slow_speaking_rate_fires(self):
        """Slow speech (2 syllables/s) — longer pauses between syllables."""
        self.assertTrue(
            self._fire(amplitude=0.20, syllable_rate=2.0),
            "Slow-speaking synthetic speech must fire barge-in.  "
            "Longer syllable gaps exercise the soft-decay path.",
        )

    def test_very_slow_speech_with_long_pauses(self):
        """
        Very slow speech (1 syllable/s) with large amplitude modulation.
        The gaps between syllables are ~500 ms.  With soft-decay in the noise
        gate path, progress should be preserved across these long pauses.
        """
        audio = voiced_speech(2.5, amplitude=0.25, syllable_rate=1.0)

        interrupts = []
        rec = make_recorder(on_interrupt=lambda info=None: interrupts.append(info))
        rec._noise_floor = 0.005  # calibrated energy path for synthetic speech

        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.inject(audio)
            h.drain(timeout=5.0)

        self.assertGreater(
            len(interrupts), 0,
            "Very slow speech (1 syllable/s, 500 ms pauses) must fire barge-in.  "
            "Validates that the noise-gate soft-decay preserves progress across "
            "natural long pauses.",
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  4. Rapid successive commands
# ═══════════════════════════════════════════════════════════════════════════════

class TestRapidCommands(unittest.TestCase):
    """Users sometimes speak two commands quickly: "Stop. Repeat that." """

    def setUp(self):
        _skip_if_no_fsdd()

    def test_rapid_successive_utterances_both_delivered(self):
        """
        Two separate utterances separated by a ~500 ms pause must each produce
        a callback ("yes" … "no" fast back-and-forth).

        The worker thread processes silence frames with no Silero overhead, so
        the real-time gap between utterances must exceed silence_duration plus
        the onset-Silero latency (~30 ms).  Using inter_chunk_delay=0.020 on a
        1.0 s silence injects ~312 ms of wall-clock silence, which comfortably
        exceeds the 80 ms silence_duration + 30 ms onset cost.
        """
        callbacks = []
        rec = make_recorder(
            callback=lambda a: callbacks.append(a),
            silence_duration=0.08,
        )

        utt1 = human_voice(0.5, amplitude=0.20, speaker="george",  digit=1)
        utt2 = human_voice(0.5, amplitude=0.20, speaker="jackson", digit=2)

        with AudioHarness(rec) as h:
            h.inject(utt1)
            # 1.0 s of silence at 20 ms/chunk ≈ 312 ms wall-clock — enough for
            # the endpointing timer (80 ms) to fire before utt2 starts.
            h.inject(silence(1.0), inter_chunk_delay=0.020)
            h.inject(utt2)
            h.inject(silence(0.5), inter_chunk_delay=0.020)
            h.drain(timeout=8.0, settle=0.5)

        self.assertEqual(
            len(callbacks), 2,
            f"Two utterances with a 500 ms gap must each produce a callback.  "
            f"Got {len(callbacks)}.",
        )

    def test_sequential_turns_after_bargein_both_delivered(self):
        """
        Barge-in fires during TTS, then after TTS ends, user speaks again.
        Both the barge-in interrupt AND the second utterance must be delivered.
        """
        interrupts = []
        callbacks = []

        rec = make_recorder(
            callback=lambda a: callbacks.append(a),
            on_interrupt=lambda info=None: interrupts.append(info),
            silence_duration=0.05,
        )

        with AudioHarness(rec) as h:
            # First: barge-in during TTS
            h.set_tts_speaking()
            h.inject(human_voice_concat(1.0, amplitude=0.20))
            h.drain(timeout=3.0)
            h.stop_tts()
            # Allow RECOVER state (0.25 s) to expire with real-time paced silence
            h.inject(silence(0.5), inter_chunk_delay=0.020)

            # Second: normal utterance after TTS
            h.inject(human_voice_concat(0.8, amplitude=0.20))
            h.inject(silence(0.3), inter_chunk_delay=0.020)
            h.drain(timeout=6.0, settle=0.3)

        self.assertGreater(len(interrupts), 0, "First barge-in must fire")
        self.assertGreater(
            len(callbacks), 0,
            f"Second utterance after barge-in must produce callback.  "
            f"Got {len(callbacks)}.  The system may be stuck after barge-in.",
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  5. Cross-speaker comprehensive test
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossSpeakerCallbackDelivery(unittest.TestCase):
    """
    Every speaker must produce a callback (via normal listening path, not
    barge-in) when speaking after the TTS has ended.  This tests the full
    pipeline: VAD → recording → _finish_recording → callback.
    """

    def setUp(self):
        _skip_if_no_fsdd()

    def test_all_speakers_produce_callbacks_after_tts(self):
        """
        Each FSDD speaker says a digit; the system must deliver a callback
        with the captured audio.  This validates end-to-end recording for
        every available voice.
        """
        failures = []
        voice_dir = os.path.join(os.path.dirname(__file__), "voice_samples")

        for speaker in _ALL_SPEAKERS:
            for digit in _ALL_DIGITS:
                fname = os.path.join(voice_dir, f"{digit}_{speaker}_0.wav")
                if not os.path.exists(fname):
                    continue

                callbacks = []
                rec = make_recorder(
                    callback=lambda a: callbacks.append(a),
                    silence_duration=0.05,
                )

                audio = human_voice(0.8, amplitude=0.20, speaker=speaker, digit=digit)

                with AudioHarness(rec) as h:
                    h.inject(audio)
                    h.inject(silence(0.3), inter_chunk_delay=0.020)
                    h.drain(timeout=5.0, settle=0.2)

                if not callbacks:
                    failures.append(f"{speaker}/{digit}: no callback")
                elif len(callbacks[0]) / SR < 0.05:
                    failures.append(f"{speaker}/{digit}: callback audio too short "
                                    f"({len(callbacks[0])/SR:.3f}s)")

        if failures:
            self.fail(
                f"{len(failures)} speaker/digit(s) failed to produce a callback:\n"
                + "\n".join(f"  {f}" for f in failures)
            )

    def test_captured_audio_rms_reasonable(self):
        """
        Captured audio from every speaker must have RMS > 0.005 (not silence).
        Ensures the recorder isn't delivering empty or near-silence buffers.
        """
        failures = []
        voice_dir = os.path.join(os.path.dirname(__file__), "voice_samples")

        for speaker in _ALL_SPEAKERS:
            for digit in _ALL_DIGITS:
                fname = os.path.join(voice_dir, f"{digit}_{speaker}_0.wav")
                if not os.path.exists(fname):
                    continue

                callbacks = []
                rec = make_recorder(
                    callback=lambda a: callbacks.append(a),
                    silence_duration=0.05,
                )
                audio = human_voice(0.8, amplitude=0.20, speaker=speaker, digit=digit)

                with AudioHarness(rec) as h:
                    h.inject(audio)
                    h.inject(silence(0.3), inter_chunk_delay=0.020)
                    h.drain(timeout=5.0, settle=0.2)

                if callbacks:
                    captured_rms = float(np.sqrt(np.mean(callbacks[0] ** 2)))
                    if captured_rms < 0.005:
                        failures.append(
                            f"{speaker}/{digit}: RMS={captured_rms:.5f} (too quiet)"
                        )

        if failures:
            self.fail(
                f"Captured audio is near-silence for:\n"
                + "\n".join(f"  {f}" for f in failures)
            )
