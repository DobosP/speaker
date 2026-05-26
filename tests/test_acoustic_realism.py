"""
Acoustic realism tests.

These tests simulate real room acoustics — the single biggest gap between
the synthetic test environment and a real living-room deployment.

CURRENTLY FAILING (bugs exposed)
---------------------------------

  test_room_delayed_echo_does_not_self_bargein
    BUG: EchoGuard.similarity() computes zero-lag cosine similarity between
    the raw mic chunk and the reference at the same timestamp.  When the TTS
    audio travels 2–3 metres to the microphone, it arrives ~7–30 ms late
    (340 m/s), and real rooms add reflections at 30–80 ms.  The zero-lag
    correlation between mic[t] and ref[t] is LOW → echo gate passes →
    self-barge-in fires during TTS playback.
    FIX: Replace zero-lag dot-product with max cross-correlation over a
    ±80 ms lag window.  Confirm the mic is echo by finding ANY lag where
    similarity ≥ threshold, not just at zero lag.

  test_full_reverberant_room_self_barge_in
    BUG: Same root cause as above, tested with a full reverberant room model
    (direct + 3 reflections + late diffuse tail).  The reference clock advances
    in lock-step with wall time, but the mic signal is spread across a 250 ms
    reverb tail.  No single lag shows high similarity → echo completely bypasses
    the gate.

CURRENTLY PASSING (regression guards)
--------------------------------------

  test_zero_delay_echo_blocked_correctly
    Baseline: when echo is injected at exactly the same timestamp as the
    reference (zero delay, as in the existing unit tests), the gate correctly
    blocks it.  This test must keep passing after any echo-gate fix.

  test_barge_in_fires_despite_low_echo_with_user_speech
    When the mic contains BOTH the user's voice and a low-level zero-delay echo,
    the EchoGuard should pass (user speech dominates) and barge-in should fire.

  test_all_speakers_barge_in_during_tts
    All four available FSDD speakers × three digits must trigger barge-in when
    played during TTS.  Tests cross-speaker reliability of the Silero path.

  test_barge_in_latency_per_speaker
    Each FSDD speaker must produce a barge-in event within 500 ms of speech
    onset.  Documents per-speaker latency variance.

  test_reverb_tail_does_not_fire_callback_during_recover
    BUG: The 250 ms room reverb tail after TTS ends has enough energy
    (RMS > vad_threshold) to trigger speech-onset detection during the
    RECOVER window.  A spurious callback is delivered with the reverb tail
    audio.  FIX: Ignore speech onset during RECOVER state, or gate on
    AEC reference being fully exhausted before allowing recording.

  test_reverb_tail_clears_before_listen
    After TTS ends, the system enters RECOVER state.  By the time RECOVER
    expires, the room reverb tail (250 ms RT60) should have decayed below
    the noise floor.  Subsequent user speech must fire a callback.
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
    tts_echo,
    human_voice,
    human_voice_concat,
    apply_room_delay,
    reverberant_echo,
    HUMAN_VOICE_AVAILABLE,
)
from tests.harness import AudioHarness, make_recorder

# ── FSDD availability guard ───────────────────────────────────────────────────

def _skip_if_no_fsdd():
    if not HUMAN_VOICE_AVAILABLE():
        raise unittest.SkipTest(
            "FSDD voice samples not found in tests/voice_samples/. "
            "Run the test suite with network access once to download them."
        )


_SPEAKERS_AND_DIGITS = [
    ("george",  1), ("george",  2), ("george",  3),
    ("jackson", 1), ("jackson", 2), ("jackson", 3),
    ("nicolas", 1), ("nicolas", 2), ("nicolas", 3),
    ("theo",    1), ("theo",    2), ("theo",    3),
]


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Room-delayed echo bypasses EchoGuard (FAILS)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRoomDelayedEcho(unittest.TestCase):
    """
    EchoGuard uses zero-lag cosine similarity.  Any propagation delay causes
    similarity to drop below the 0.45 threshold, allowing the echo to appear
    to the barge-in detector as legitimate user speech.
    """

    def _count_interrupts_with_delayed_echo(
        self, delay_ms: float, attenuation_db: float = -6.0
    ) -> int:
        """Helper: returns number of barge-in interrupts for a given room delay."""
        tts_ref = tts_echo(1.5, amplitude=0.10)
        # Mic receives the TTS echo with room propagation delay
        mic_signal = apply_room_delay(tts_ref, delay_ms=delay_ms,
                                       attenuation_db=attenuation_db)

        interrupts = []
        rec = make_recorder(on_interrupt=lambda info=None: interrupts.append(info))

        with AudioHarness(rec) as h:
            h.set_tts_speaking(audio_ref=tts_ref)
            h.inject(mic_signal)
            h.drain()

        return len(interrupts)

    def test_zero_delay_echo_blocked_correctly(self):
        """Baseline: zero-delay echo must be blocked by EchoGuard (this passes)."""
        tts_ref = tts_echo(1.5, amplitude=0.12)

        interrupts = []
        rec = make_recorder(on_interrupt=lambda info=None: interrupts.append(info))

        with AudioHarness(rec) as h:
            h.set_tts_speaking(audio_ref=tts_ref)
            h.inject(tts_ref)  # exact same signal → similarity = 1.0
            h.drain()

        self.assertEqual(
            len(interrupts), 0,
            "Zero-delay echo must be blocked.  If this fails the echo gate is broken.",
        )

    def test_7ms_room_delay_does_not_self_bargein(self):
        """
        7 ms delay = speaker 2.4 m from mic (340 m/s × 0.007 s).
        A typical living-room setup.

        CURRENTLY FAILS: zero-lag similarity of mic[t] vs ref[t] drops to ~0
        when the echo arrives 112 samples late.  The gate passes → barge-in fires.

        FIX: Use maximum cross-correlation over ±80 ms lag window.
        """
        n = self._count_interrupts_with_delayed_echo(delay_ms=7.0)
        self.assertEqual(
            n, 0,
            f"7 ms room delay (speaker 2.4 m from mic) must NOT trigger self-barge-in. "
            f"Got {n} false interrupt(s).  "
            f"BUG: EchoGuard similarity is zero-lag only; any propagation delay "
            f"drops correlation below 0.45 → echo looks like user speech.  "
            f"FIX: Compute max cross-correlation across ±80 ms lag window.",
        )

    def test_35ms_wall_reflection_does_not_self_bargein(self):
        """
        35 ms = first wall reflection in a 6-metre room.
        This is the dominant real-world failure: living rooms add 20–80 ms
        reflections that completely fool the zero-lag gate.
        """
        n = self._count_interrupts_with_delayed_echo(delay_ms=35.0)
        self.assertEqual(
            n, 0,
            f"35 ms wall reflection must NOT trigger self-barge-in (got {n}).  "
            f"Same root cause as 7 ms: zero-lag similarity is ~0 for delayed echo.",
        )

    def test_full_reverberant_room_does_not_self_bargein(self):
        """
        Full reverberant room model (direct + reflections + diffuse tail, RT60=250ms).
        The TTS reference is set to the clean signal; the mic receives the reverberant
        version.  Self-barge-in must NOT fire.

        CURRENTLY FAILS with high probability because no single time-offset
        produces similarity ≥ 0.45 when the energy is spread across a 250 ms tail.
        """
        tts_ref = tts_echo(2.0, amplitude=0.10)
        mic_signal = reverberant_echo(tts_ref, rt60_ms=250.0)

        interrupts = []
        rec = make_recorder(on_interrupt=lambda info=None: interrupts.append(info))

        with AudioHarness(rec) as h:
            h.set_tts_speaking(audio_ref=tts_ref)
            h.inject(mic_signal)
            h.drain()

        self.assertEqual(
            len(interrupts), 0,
            f"Reverberant room echo (RT60=250ms) must not self-barge-in. "
            f"Got {len(interrupts)} false interrupt(s).  "
            f"This is the exact failure mode users experience in real rooms.",
        )

    def test_delayed_echo_plus_user_speech_fires_barge_in(self):
        """
        User speaks while TTS is playing.  Mic = user_voice + delayed_echo(TTS).
        The barge-in MUST fire even in a reverberant room.

        This is the hardest real-world case: the system must simultaneously
        reject pure TTS echo and pass user_voice+echo.  Currently this may
        work only by accident (user voice raises energy above threshold).
        """
        _skip_if_no_fsdd()
        tts_ref = tts_echo(2.0, amplitude=0.08)
        room_echo = reverberant_echo(tts_ref, rt60_ms=200.0)
        user = human_voice_concat(1.0, amplitude=0.20)

        # Mic = user voice starting 0.2 s in, mixed with room echo
        n = len(room_echo)
        mic = room_echo.copy()
        offset = int(0.2 * SR)
        end = min(offset + len(user), n)
        mic[offset:end] += user[: end - offset]

        interrupts = []
        rec = make_recorder(on_interrupt=lambda info=None: interrupts.append(info))
        # Stable gate: ambient bootstrap from user+echo RMS alone over-estimates floor.
        rec._noise_floor = 0.005

        with AudioHarness(rec) as h:
            h.set_tts_speaking(audio_ref=tts_ref)
            h.inject(mic)
            h.drain()

        self.assertGreater(
            len(interrupts), 0,
            "User speech over TTS with room echo must trigger barge-in.  "
            "If this fails, the echo gate is blocking the user's voice because "
            "the combined (user+echo) signal has too high correlation with ref.",
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Cross-speaker and cross-digit barge-in reliability
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossSpeakerBargein(unittest.TestCase):
    """
    The system must reliably trigger barge-in for ALL available FSDD speakers
    and digits, not just the specific combination used in other tests.

    Failures here expose that the system works for some phonemes/speakers but
    not others — a common real-world problem when voice diversity was not tested.
    """

    def setUp(self):
        _skip_if_no_fsdd()

    def _barge_in_fires(
        self,
        speaker: str,
        digit: int,
        amplitude: float = 0.20,
        noise_floor: float = None,
    ) -> tuple[bool, float]:
        """Returns (fired, latency_ms) for one speaker×digit combination."""
        audio = human_voice(1.0, amplitude=amplitude, speaker=speaker, digit=digit)

        interrupts = []
        t_inject = [0.0]
        rec = make_recorder(on_interrupt=lambda info=None: interrupts.append(time.time()))
        rec._noise_floor = noise_floor if noise_floor is not None else 0.005

        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            t_inject[0] = time.time()
            h.inject(audio)
            h.drain(timeout=3.0)

        fired = len(interrupts) > 0
        latency = (interrupts[0] - t_inject[0]) * 1000.0 if fired else float("inf")
        return fired, latency

    def test_all_available_speakers_and_digits_fire(self):
        """
        Every FSDD speaker × digit that is present in tests/voice_samples/
        must produce a barge-in when injected during TTS playback.

        Failures isolate which speakers/phonemes the Silero path misses.
        """
        failures = []
        for speaker, digit in _SPEAKERS_AND_DIGITS:
            voice_dir = os.path.join(
                os.path.dirname(__file__), "voice_samples", f"{digit}_{speaker}_0.wav"
            )
            if not os.path.exists(voice_dir):
                continue
            fired, latency = self._barge_in_fires(speaker, digit)
            if not fired:
                failures.append(f"{speaker}/{digit}: did not fire")

        if failures:
            self.fail(
                f"{len(failures)} speaker/digit combination(s) did NOT trigger barge-in:\n"
                + "\n".join(f"  {f}" for f in failures)
                + "\n\nThese represent real users whose voice is not reliably detected."
            )

    def test_per_speaker_latency_under_500ms(self):
        """
        Every available speaker must achieve barge-in within 500 ms.

        Fails for speakers whose initial phoneme (e.g. unvoiced fricative /θ/,
        plosive /p/) is not classified as voiced by Silero until the 2nd–3rd frame.
        """
        slow = []
        for speaker, digit in _SPEAKERS_AND_DIGITS:
            voice_dir = os.path.join(
                os.path.dirname(__file__), "voice_samples", f"{digit}_{speaker}_0.wav"
            )
            if not os.path.exists(voice_dir):
                continue
            fired, latency_ms = self._barge_in_fires(speaker, digit)
            if not fired:
                slow.append(f"{speaker}/{digit}: never fired")
            elif latency_ms > 500.0:
                slow.append(f"{speaker}/{digit}: {latency_ms:.0f} ms (SLO=500ms)")

        if slow:
            self.fail(
                f"Barge-in latency SLO violated for {len(slow)} combination(s):\n"
                + "\n".join(f"  {s}" for s in slow)
            )

    def test_low_amplitude_speakers_still_fire(self):
        """
        Speaker amplitude = 0.08 — all speakers must fire barge-in.

        Most fire via Silero (consecutive voiced frames build the accumulator).
        Speakers with predominantly fricative phonemes (e.g. "three") may not
        reach the Silero threshold on every frame; the calibrated energy path
        (noise_floor_calibrated=True) provides a reliable fallback so they
        still accumulate when RMS > 3× threshold.
        """
        failures = []
        for speaker, digit in _SPEAKERS_AND_DIGITS:
            voice_dir = os.path.join(
                os.path.dirname(__file__), "voice_samples", f"{digit}_{speaker}_0.wav"
            )
            if not os.path.exists(voice_dir):
                continue
            fired, _ = self._barge_in_fires(speaker, digit, amplitude=0.08, noise_floor=0.005)
            if not fired:
                failures.append(f"{speaker}/{digit}")

        if failures:
            self.fail(
                f"Low-amplitude barge-in (amplitude=0.08) failed for:\n"
                + "\n".join(f"  {f}" for f in failures)
                + "\n\nIf Silero is working these should all pass.  "
                "Failures indicate Silero is not scoring these voices as voiced."
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Reverb tail after TTS ends
# ═══════════════════════════════════════════════════════════════════════════════

class TestReverbTailAfterTTS(unittest.TestCase):
    """
    After the assistant finishes speaking, the room reverb tail lingers for
    200–400 ms.  The system enters RECOVER state.  During RECOVER, the reverb
    tail must NOT trigger a user-utterance callback (it would send garbage
    audio to the STT engine).
    """

    def test_reverb_tail_does_not_fire_callback_during_recover(self):
        """
        1 s of TTS → TTS stops → 300 ms of room reverb tail injected.
        No user speech.  Callback must NOT fire.

        If it fires, the STT engine receives the reverb tail as a "command".
        """
        tts_ref = tts_echo(1.0, amplitude=0.10)
        reverb_tail = reverberant_echo(tts_ref, rt60_ms=300.0)
        # Isolate just the tail portion (after TTS ends)
        tail_only = reverb_tail.copy()
        tail_only[: len(tts_ref)] = 0.0  # zero out the TTS overlap

        callbacks = []
        rec = make_recorder(
            callback=lambda a: callbacks.append(a),
            silence_duration=0.15,
        )

        with AudioHarness(rec) as h:
            h.set_tts_speaking(audio_ref=tts_ref)
            h.inject(tts_ref)
            h.stop_tts()
            time.sleep(0.05)
            h.inject(tail_only[:int(0.35 * SR)])  # 350 ms of reverb tail
            h.inject(silence(0.3), inter_chunk_delay=0.020)
            h.drain(timeout=4.0, settle=0.2)

        self.assertEqual(
            len(callbacks), 0,
            f"Room reverb tail after TTS must NOT produce a callback.  "
            f"Got {len(callbacks)} spurious callback(s) with "
            + str([round(len(a) / SR, 2) for a in callbacks])
            + " s of audio — reverb noise would be sent to the STT engine.",
        )

    def test_user_speech_after_reverb_tail_fires_callback(self):
        """
        After TTS + reverb tail, real user speech must produce a callback.
        This confirms the system re-opens correctly once the tail decays.
        """
        _skip_if_no_fsdd()
        tts_ref = tts_echo(1.0, amplitude=0.10)
        reverb_tail = reverberant_echo(tts_ref, rt60_ms=300.0)

        callbacks = []
        rec = make_recorder(
            callback=lambda a: callbacks.append(a),
            silence_duration=0.05,
        )

        with AudioHarness(rec) as h:
            h.set_tts_speaking(audio_ref=tts_ref)
            h.inject(tts_ref)
            h.stop_tts()
            # RECOVER state lasts 0.25 s.  Inject 1.5 s of real-time-paced
            # silence (24 chunks × 20 ms = 480 ms wall-clock) to ensure the
            # RECOVER window expires well before user speech starts.
            h.inject(silence(1.5), inter_chunk_delay=0.020)
            # Now the user speaks
            h.inject(human_voice_concat(1.0, amplitude=0.20))
            h.inject(silence(0.5), inter_chunk_delay=0.020)
            h.drain(timeout=10.0, settle=0.5)

        self.assertGreater(
            len(callbacks), 0,
            "User speech after TTS + reverb tail must produce a callback.  "
            "If this fails, the system is stuck in a non-listening state.",
        )
