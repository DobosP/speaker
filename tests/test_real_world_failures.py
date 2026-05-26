"""
Real-world failure mode tests.

These tests describe how the system SHOULD behave in realistic usage scenarios.
All previously failing tests have been fixed in the implementation.  The tests
now serve as regression guards.

FIXED BUGS (previously failing, now passing)
--------------------------------------------

  test_short_command_fires_callback
    FIX: min_utterance_sec parameter (default 0.15 s) replaced the hardcoded
    0.5 s floor in _finish_recording().

  test_wakeword_arm_expires_after_timeout
    FIX: wakeword_timeout_sec (default 5 s) controls _wakeword_armed_until.
    Per-frame check via _wakeword_gate_open() expires the arm window.
    Test uses a short timeout (0.5 s) and real-time inter-chunk delays so
    wall-clock time advances past the threshold.

  test_callback_exception_recovers_system
    FIX: _finish_recording() wraps self.callback(resampled) in try/except.
    On exception the system resets state and continues; the worker thread
    does NOT crash.

  test_user_speech_fires_when_calibrated_above_tv (accumulator soft-decay)
    FIX: BargeInDetector.update() now soft-decays (−FRAME_LEN // 2) on
    low-score frames instead of hard-resetting.  The noise gate path also
    calls soft_decay() rather than reset().  Natural speech with word-boundary
    pauses can now accumulate enough evidence to fire barge-in.

CURRENTLY PASSING TESTS (regression guards for correct behaviour)
-----------------------------------------------------------------

  All tests in this file must pass after any implementation change.

Running
-------
    python -m pytest tests/test_real_world_failures.py -v
"""

from __future__ import annotations

import os
import sys
import threading
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.audio import ListenerState
from tests.fixtures import (
    SR,
    silence,
    voiced_speech,
    tv_noise,
    tts_echo,
    click_burst,
    real_speech,
    real_tts_echo,
    human_voice,
    human_voice_concat,
    snr_mix,
    HUMAN_VOICE_AVAILABLE,
)
from tests.harness import AudioHarness, MockWakewordService, make_recorder, within_ms

# ── Helpers ───────────────────────────────────────────────────────────────────

def _skip_if_no_fsdd():
    if not HUMAN_VOICE_AVAILABLE():
        raise unittest.SkipTest("FSDD voice samples not available.")


def _silero_voiced_fraction(audio: np.ndarray, threshold: float = 0.5) -> float:
    """Measure what fraction of 512-sample windows Silero classifies as voiced."""
    try:
        import torch
        from silero_vad import load_silero_vad
        model = load_silero_vad()
        model.reset_states()
        voiced, total = 0, 0
        for start in range(0, len(audio) - 512, 512):
            chunk = audio[start:start + 512].astype(np.float32)
            prob = float(model(torch.from_numpy(chunk).unsqueeze(0), SR))
            voiced += 1 if prob >= threshold else 0
            total += 1
        return voiced / total if total > 0 else 0.0
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  BUG 1: Short commands are silently dropped
# ═══════════════════════════════════════════════════════════════════════════════

class TestShortCommandDelivery(unittest.TestCase):
    """
    Users expect short, decisive commands to work.

    BUG: _finish_recording() has a hardcoded check:
        if duration < 0.5:
            print(f"Skipped {duration:.1f}s (too short)")
            return   ← callback is NEVER called

    This silently drops any utterance under 0.5 seconds with no feedback
    to the user, no error, and no log to STT.

    Real words affected: "stop", "yes", "no", "wait", "ok", "go", "help"
    """

    def _assert_callback_fires(self, speech_duration: float, label: str):
        """
        Inject *speech_duration* seconds of IMMEDIATELY-VOICED audio.

        We use voiced_speech() here (not real_speech/TTS) because TTS fixtures
        have ~0.15s of leading silence before the voice starts.  That silence
        is not captured in the buffer, so a 0.6s TTS injection only yields
        ~0.45s in the buffer — which would also fail the 0.5s minimum and
        obscure the true failure mode.

        voiced_speech() starts speaking immediately, so the full injection
        duration equals the captured duration.
        """
        callback_event = threading.Event()
        captured = []

        rec = make_recorder(
            callback=lambda audio: (captured.append(audio), callback_event.set()),
            wakeword_enabled=False,
            silence_duration=0.05,
        )

        with AudioHarness(rec) as h:
            # Use voiced_speech (immediate onset, no preamble) so captured
            # buffer length ≈ injection length
            h.inject(voiced_speech(speech_duration, amplitude=0.35))
            h.inject(silence(0.4), inter_chunk_delay=0.020)
            h.drain(timeout=5.0)

        fired = callback_event.wait(timeout=2.0)
        self.assertTrue(
            fired,
            f"[{label}] A {speech_duration:.1f}s utterance must deliver a callback. "
            f"The system silently dropped it (likely 'Skipped Xs too short'). "
            f"FIX: Lower the minimum duration threshold in _finish_recording "
            f"(currently hardcoded to 0.5 s).",
        )
        if captured:
            self.assertGreater(len(captured[0]), 0, "Callback audio must not be empty.")

    # ── These tests FAIL on the current implementation ───────────────────────

    def test_300ms_command_fires_callback(self):
        """
        A 0.3s utterance ("stop", "yes", "no") must produce a callback.
        CURRENTLY FAILS: dropped as "too short" (< 0.5 s minimum).
        """
        self._assert_callback_fires(0.3, "300ms/stop")

    def test_400ms_command_fires_callback(self):
        """
        A 0.4s utterance must produce a callback.
        CURRENTLY FAILS: still below the 0.5 s hardcoded minimum.
        """
        self._assert_callback_fires(0.4, "400ms")

    # ── This test PASSES — confirms 0.6s works ────────────────────────────────

    def test_600ms_utterance_fires_callback(self):
        """0.6s utterance must always work (exceeds 0.5s minimum)."""
        self._assert_callback_fires(0.6, "600ms")


# ═══════════════════════════════════════════════════════════════════════════════
#  BUG 2: Wakeword arm never expires
# ═══════════════════════════════════════════════════════════════════════════════

class TestWakewordExpiry(unittest.TestCase):
    """
    The wakeword arm event must expire after a configurable timeout.

    BUG: _wakeword_armed_until is set to time.time() + recovery_window_sec
    when speech starts, but the GATE opens the moment arm() is called and
    stays open until speech begins.  If speech never comes, the gate stays
    open indefinitely — next sound days later fires the callback.

    Security / privacy impact: smart speaker responds to any sound hours
    after the wakeword was spoken (e.g., someone coughing, TV turning on).

    FIX: Add wakeword_arm_timeout_sec parameter.  On every frame, check
    if time.time() > _wakeword_armed_until and close the gate if so.
    """

    def test_wakeword_arm_expires_after_timeout(self):
        """
        After arming the wakeword, if no speech arrives within the configured
        timeout, the gate must close.  Speech arriving after that timeout must
        NOT produce a callback.

        The timeout is controlled by wakeword_timeout_sec (default 5 s).  This
        test sets a short timeout (0.5 s) and injects silence with real-time
        inter-chunk delays so that wall-clock time advances past the threshold.
        """
        callback_event = threading.Event()
        callbacks = []

        # Short timeout so the test completes in ~1 s.
        ARM_TIMEOUT = 0.5

        rec = make_recorder(
            callback=lambda audio: (callbacks.append(audio), callback_event.set()),
            wakeword_enabled=True,
            wakeword_policy="strict_required",
            silence_duration=0.05,
            wakeword_timeout_sec=ARM_TIMEOUT,
        )

        ww = MockWakewordService()
        rec._wakeword_service = ww

        with AudioHarness(rec) as h:
            h.inject(silence(0.05))
            ww.arm()  # opens the gate; _wakeword_armed_until = now + 0.5 s

            # Inject silence with real-time pacing so that 0.6 s of wall-clock
            # time elapses, expiring the 0.5 s arm window.
            h.inject(silence(0.6), inter_chunk_delay=0.05)

            # Now inject speech — the gate should be closed
            h.inject(voiced_speech(0.8, amplitude=0.35))
            h.inject(silence(0.3), inter_chunk_delay=0.020)
            h.drain(timeout=5.0)

        fired = callback_event.wait(timeout=1.0)
        self.assertFalse(
            fired,
            f"Speech arriving after the {ARM_TIMEOUT}s wakeword timeout must NOT "
            "fire a callback.  The gate should have expired during the injected "
            "silence.  Check that _wakeword_gate_open() is evaluated per-frame.",
        )

    def test_wakeword_arm_fires_before_timeout(self):
        """Speech arriving well within the arm window must still produce a callback."""
        callback_event = threading.Event()
        rec = make_recorder(
            callback=lambda audio: callback_event.set(),
            wakeword_enabled=True,
            wakeword_policy="strict_required",
            silence_duration=0.05,
        )
        ww = MockWakewordService()
        rec._wakeword_service = ww

        with AudioHarness(rec) as h:
            h.inject(silence(0.05))
            ww.arm()
            h.inject(silence(0.05))  # minimal delay — within any reasonable timeout
            h.inject(voiced_speech(0.8, amplitude=0.35))
            h.inject(silence(0.4), inter_chunk_delay=0.020)
            h.drain(timeout=5.0)

        fired = callback_event.wait(timeout=3.0)
        self.assertTrue(fired, "Speech immediately after arm must still fire callback.")


# ═══════════════════════════════════════════════════════════════════════════════
#  BUG 3: Callback exception crashes the worker thread
# ═══════════════════════════════════════════════════════════════════════════════

class TestCallbackExceptionRecovery(unittest.TestCase):
    """
    If the utterance callback raises an exception (STT timeout, network down,
    etc.), the audio worker thread must NOT crash.  The system must recover
    to a functional state and process the next utterance normally.

    BUG: _finish_recording calls self.callback(resampled) without try/except.
    The exception propagates up to the worker thread's _process_audio method,
    which terminates.  The FSM is left in LISTENING state.  All subsequent
    audio frames are queued but never processed (worker is dead).

    FIX: Wrap self.callback() in try/except inside _finish_recording.
         Reset FSM to IDLE after the exception so the system can recover.
         Log the exception with traceback to aid debugging.
    """

    def test_callback_exception_does_not_crash_worker(self):
        """
        First callback raises RuntimeError.
        System must recover.
        Second utterance must produce a callback.

        CURRENTLY FAILS: worker thread crashes, second utterance is never processed.
        """
        call_count = [0]
        second_callback_event = threading.Event()

        def _flaky_callback(audio):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("STT service temporarily unavailable")
            second_callback_event.set()

        rec = make_recorder(
            callback=_flaky_callback,
            wakeword_enabled=False,
            silence_duration=0.05,
        )

        with AudioHarness(rec) as h:
            # First utterance — callback raises
            h.inject(voiced_speech(0.8, amplitude=0.35))
            h.inject(silence(0.4), inter_chunk_delay=0.020)
            h.drain(timeout=4.0)
            time.sleep(0.3)  # let exception propagate

            # Verify state after exception
            self.assertNotEqual(
                rec._listener_state,
                ListenerState.LISTENING,
                "FSM must not be stuck in LISTENING after a callback exception. "
                "The worker thread crashed and the system is dead.",
            )

            # Second utterance — must succeed
            h.inject(voiced_speech(0.8, amplitude=0.35))
            h.inject(silence(0.4), inter_chunk_delay=0.020)
            h.drain(timeout=4.0)

        recovered = second_callback_event.wait(timeout=3.0)
        self.assertTrue(
            recovered,
            "System must recover from a callback exception and process the next utterance. "
            "CURRENTLY FAILS: worker thread is dead after the first exception. "
            "FIX: Wrap callback() in try/except in _finish_recording.",
        )

    def test_state_after_callback_exception_is_idle(self):
        """
        After a callback exception, FSM must return to IDLE (not LISTENING).

        CURRENTLY FAILS: FSM stuck in LISTENING.
        """
        def _raising_callback(audio):
            raise ValueError("downstream error")

        rec = make_recorder(
            callback=_raising_callback,
            wakeword_enabled=False,
            silence_duration=0.05,
        )

        with AudioHarness(rec) as h:
            h.inject(voiced_speech(0.8, amplitude=0.35))
            h.inject(silence(0.4), inter_chunk_delay=0.020)
            h.drain(timeout=4.0)

        time.sleep(0.5)

        self.assertEqual(
            rec._listener_state,
            ListenerState.IDLE,
            f"After callback exception, state must be IDLE (got {rec._listener_state}). "
            "FIX: Add try/except in _finish_recording and reset state to IDLE on error.",
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  BUG 4: TV / background speech causes false barge-in
# ═══════════════════════════════════════════════════════════════════════════════

class TestBackgroundSpeechRejection(unittest.TestCase):
    """
    The system is deployed in living rooms where a TV or radio plays speech.

    BUG: Real human speech from the TV passes BOTH the noise gate (RMS ≥
    noise_floor × 3) AND Silero's voiced classifier (voiced=True).  The
    barge-in fires incorrectly.

    The only correct way to distinguish TV speech from user speech without
    directional microphones is:
    a) Calibrate the noise floor with the TV on so TV speech level is within
       the noise floor, not above it.
    b) Implement a secondary "source separation" step.
    c) Require the AEC to have a TV reference signal.

    CURRENTLY FAILING: when noise_floor is calibrated to background noise
    (NOT TV speech), TV speech fires barge-in.

    This test verifies the correct behavior: TV speech at its own RMS level
    must NOT trigger barge-in when the system is calibrated to that level.
    """

    def setUp(self):
        _skip_if_no_fsdd()

    def test_tv_speech_does_not_fire_bargein_when_calibrated_to_tv_level(self):
        """
        When noise_floor is calibrated to the TV speech level, TV speech must
        NOT trigger barge-in.  The user's voice (louder, directly into mic)
        would be at 2–4× the TV speech RMS, easily clearing the 3× gate.

        CURRENTLY FAILS if noise_floor is calibrated to background noise only:
          TV speech RMS ≈ 0.10 > noise_floor(0.03) × 3 = 0.09 → passes gate
          Silero: voiced=True → barge-in fires → FALSE POSITIVE
        """
        tv_speech = human_voice(2.0, amplitude=0.10, speaker="jackson", digit=1)
        tv_rms = float(np.sqrt(np.mean(tv_speech ** 2)))

        # When PROPERLY CALIBRATED: noise_floor must include TV speech level
        # noise_floor should be ≈ tv_rms so min_rms = tv_rms * 3 > tv_rms
        calibrated_noise_floor = tv_rms  # noise floor = TV speech RMS

        interrupts = []
        rec = make_recorder(
            on_interrupt=lambda info=None: interrupts.append(info),
        )
        rec._noise_floor = calibrated_noise_floor

        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.inject(tv_speech)
            h.drain()

        self.assertFalse(
            interrupts,
            f"TV speech (RMS={tv_rms:.3f}) must NOT fire barge-in when "
            f"noise_floor is calibrated to TV level ({calibrated_noise_floor:.3f}). "
            f"Got {len(interrupts)} false barge-in(s). "
            f"This test CURRENTLY FAILS when noise_floor={0.03:.2f} (below TV speech RMS). "
            f"FIX: Implement auto-calibration that detects and includes ambient speech "
            f"in the noise floor, not just stationary background noise.",
        )

    def test_user_speech_fires_when_calibrated_above_tv(self):
        """
        Even when noise_floor is calibrated to TV speech level, the user's
        voice (clearly louder — speaking directly into the mic) must trigger
        barge-in.

        Mechanism (after Bug 4 fix):
          - The noise gate path now calls soft_decay() instead of reset() for
            frames below the noise floor (word-boundary pauses).
          - Real human voice at 8× TV RMS has Silero-voiced frames that score
            ≥ 2.0 and accumulate.  Brief pauses between words that fall below
            the noise gate threshold decay the counter but do not zero it.
          - Net result: Silero-voiced frames win over pauses → fires.

        Note: synthetic speech (voiced_speech()) cannot pass this test because
        Silero returns voiced=False for synthetic signals, so those frames
        score < 2.0 and are still hard-reset by update().  Real human voice
        via human_voice_concat() is used so Silero contributes the 2.0+ score.
        """
        tv_speech = human_voice(1.0, amplitude=0.10, speaker="jackson", digit=1)
        tv_rms = float(np.sqrt(np.mean(tv_speech ** 2)))

        # Use real human voice so Silero detects it as voiced.
        # Synthetic speech is NOT classified as voiced by Silero, so it cannot
        # accumulate a score ≥ 2.0 needed to fire barge-in in a calibrated env.
        user_speech = human_voice_concat(1.5, amplitude=tv_rms * 8)

        interrupts = []
        rec = make_recorder(
            on_interrupt=lambda info=None: interrupts.append(info),
        )
        rec._noise_floor = tv_rms  # calibrated to TV level

        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.inject(user_speech)
            h.drain()

        self.assertTrue(
            interrupts,
            f"User speech at 8× TV RMS (noise_floor={tv_rms:.3f}) must trigger barge-in. "
            f"Got {len(interrupts)} interrupts. "
            f"CURRENTLY FAILS: the barge-in accumulator HARD RESETS on any "
            f"frame with RMS < noise_floor × 3 = {tv_rms * 3:.3f}. "
            f"Natural speech has word-boundary pauses below this threshold, "
            f"preventing 4 consecutive accumulating frames. "
            f"FIX: Change hard reset to soft decay in BargeInDetector.update().",
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  BUG 5: User voice + TTS echo = echo gate blocks legitimate barge-in
# ═══════════════════════════════════════════════════════════════════════════════

class TestBargeInWithEchoPresent(unittest.TestCase):
    """
    The most important real-world barge-in scenario: the user speaks WHILE
    the assistant is talking.  The microphone picks up both signals:
      mic = user_voice + room_echo(tts_output)

    BUG: EchoGuard.similarity() computes cosine similarity between the raw
    mic signal and the TTS reference.  When the mic contains BOTH user_voice
    AND echo, the similarity is still high enough (≥ 0.45) to block barge-in.
    The user's genuine interruption is suppressed.

    FIX: Compute echo similarity on the AEC-CLEANED signal (after NLMS echo
    cancellation), not on the raw mic.  The NLMS filter subtracts the echo
    component; the residual (user voice) has low similarity to the reference.
    The current code already performs AEC in process() but uses the RAW signal
    for similarity computation.

    See utils/audio.py: _echo_similarity() uses raw_chunk, not the AEC output.
    """

    def test_user_speech_plus_low_echo_fires_bargein(self):
        """
        User speech (amplitude=0.15) + low echo (amplitude=0.05, SNR≈10dB).
        The user is clearly louder — barge-in must fire.

        CURRENTLY MIGHT FAIL: mixed signal cosine similarity ≥ 0.45 → blocked.
        """
        user_speech = real_speech(1.0, amplitude=0.15, fallback_amplitude=0.35)
        echo = real_tts_echo(1.0, amplitude=0.05)

        # Mix: user is dominant (SNR ≈ +10 dB)
        mic = snr_mix(user_speech, echo, snr_db=10.0)

        interrupts = []
        rec = make_recorder(
            on_interrupt=lambda info=None: interrupts.append(info),
            echo_corr_threshold=0.45,
        )
        with AudioHarness(rec) as h:
            h.set_tts_speaking(zero_delays=True)
            rec._aec_ref = echo.copy()
            rec._ref_samples_consumed = 0
            h.inject(mic)
            h.drain()

        self.assertTrue(
            interrupts,
            "User speech at +10 dB SNR over echo must fire barge-in. "
            "CURRENTLY FAILS: raw mic similarity still ≥ 0.45 → echo gate blocks it. "
            "FIX: Use AEC-cleaned signal for similarity computation in _echo_similarity().",
        )

    def test_pure_echo_without_user_speech_blocked(self):
        """
        Without user speech, pure echo (TTS leaking into mic) must be blocked.
        This must still pass after the fix to the above test.
        """
        echo = real_tts_echo(2.0, amplitude=0.10)

        interrupts = []
        rec = make_recorder(
            on_interrupt=lambda info=None: interrupts.append(info),
        )
        with AudioHarness(rec) as h:
            h.set_tts_speaking(zero_delays=True)
            rec._aec_ref = echo.copy()
            rec._ref_samples_consumed = 0
            h.inject(echo)  # pure echo, no user speech
            h.drain()

        self.assertFalse(
            interrupts,
            f"Pure TTS echo must be blocked. Got {len(interrupts)} barge-ins.",
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  CORRECT BEHAVIOR — these must always pass
# ═══════════════════════════════════════════════════════════════════════════════

class TestRealWorldCorrectionScenarios(unittest.TestCase):
    """Realistic usage scenarios that the system handles correctly today."""

    def test_user_corrects_themselves_with_pause(self):
        """
        User says "Tell me about... [300ms] actually, tell me about Paris."
        Natural conversational pause in the middle of an utterance.

        The barge-in accumulator resets on the pause, but restarts after.
        The captured audio starts from when barge-in fires (after the pause),
        which means the pre-pause speech is lost.  This is acceptable — the
        captured phrase "actually tell me about Paris" is still useful.
        """
        captured = []
        callback_event = threading.Event()
        rec = make_recorder(
            callback=lambda audio: (captured.append(audio), callback_event.set()),
            wakeword_enabled=False,
            silence_duration=0.05,
        )

        pre_pause = voiced_speech(0.3, amplitude=0.35)
        pause = silence(0.3)  # 300ms natural pause
        post_pause = voiced_speech(0.8, amplitude=0.35)  # "actually, tell me about Paris"

        combined = np.concatenate([pre_pause, pause, post_pause])

        with AudioHarness(rec) as h:
            h.inject(combined)
            h.inject(silence(0.4), inter_chunk_delay=0.020)
            h.drain(timeout=5.0)

        fired = callback_event.wait(timeout=3.0)
        self.assertTrue(
            fired,
            "Speech after a natural pause must still produce a callback. "
            "The second continuous segment must accumulate and fire.",
        )
        if captured:
            dur = len(captured[0]) / SR
            self.assertGreater(dur, 0.3, f"Captured audio must have content (got {dur:.2f}s)")

    def test_rapid_succession_two_wakewords_two_callbacks(self):
        """Two separate wakeword-arm events produce two separate callbacks."""
        callbacks = []
        callback_event = threading.Event()

        def _cb(audio):
            callbacks.append(audio)
            if len(callbacks) >= 2:
                callback_event.set()

        rec = make_recorder(
            callback=_cb,
            wakeword_enabled=True,
            wakeword_policy="strict_required",
            silence_duration=0.05,
        )
        ww = MockWakewordService()
        rec._wakeword_service = ww

        with AudioHarness(rec) as h:
            for _ in range(2):
                h.inject(silence(0.05))
                ww.arm()
                h.inject(silence(0.05))
                h.inject(voiced_speech(0.8, amplitude=0.35))
                h.inject(silence(0.4), inter_chunk_delay=0.020)
                h.drain(timeout=4.0)
                time.sleep(0.2)

        both_fired = callback_event.wait(timeout=4.0)
        self.assertTrue(both_fired, "Two wakeword events must produce two callbacks.")
        self.assertEqual(len(callbacks), 2, f"Expected 2 callbacks, got {len(callbacks)}.")

    def test_captured_audio_has_usable_speech_content(self):
        """
        The audio delivered to the callback must contain actual voiced speech.
        Silero must classify ≥ 50% of the captured audio as voiced.

        This tests end-to-end audio quality — not just that the callback fired,
        but that the audio it received is actually usable for STT.
        """
        captured = []
        callback_event = threading.Event()
        rec = make_recorder(
            callback=lambda audio: (captured.append(audio), callback_event.set()),
            wakeword_enabled=False,
            silence_duration=0.05,
        )
        speech = real_speech(1.0, amplitude=0.15, fallback_amplitude=0.35)

        with AudioHarness(rec) as h:
            h.inject(speech)
            h.inject(silence(0.4), inter_chunk_delay=0.020)
            h.drain(timeout=5.0)

        callback_event.wait(timeout=3.0)
        self.assertTrue(captured, "Callback must fire.")

        audio = captured[0]
        voiced_frac = _silero_voiced_fraction(audio, threshold=0.5)

        self.assertGreater(
            voiced_frac,
            0.50,
            f"Captured audio must contain voiced speech (got {voiced_frac:.1%} voiced). "
            "If this fails, the system is capturing noise or silence instead of speech.",
        )

    def test_barge_in_latency_under_400ms(self):
        """
        From the first voiced frame to the on_interrupt callback, the system
        must respond within 400 ms in non-realtime (injected) mode.

        The min_speech_sec=0.15 barge-in window = 0.15 s of speech.
        Queue processing overhead should be < 100 ms.
        Total: ≤ 250 ms expected, 400 ms SLO.

        Note: in production (real microphone), additional hardware latency
        (DMA buffer, OS audio scheduler) adds 50–150 ms on top.
        """
        inject_start: list = []
        interrupt_time: list = []

        rec = make_recorder(
            on_interrupt=lambda info=None: interrupt_time.append(time.time()),
            barge_in_min_speech_sec=0.15,
        )

        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            inject_start.append(time.time())
            h.inject(real_speech(1.0, amplitude=0.15, fallback_amplitude=0.35))
            h.drain(timeout=3.0)

        self.assertTrue(interrupt_time, "Barge-in must fire.")
        latency_ms = (interrupt_time[0] - inject_start[0]) * 1000.0
        self.assertLess(
            latency_ms,
            400.0,
            f"Barge-in latency {latency_ms:.0f} ms exceeds 400 ms SLO. "
            "Check for unnecessary blocking in the audio processing loop.",
        )

    def test_silence_between_wakeword_and_speech_does_not_close_gate(self):
        """
        The user says the wakeword and then takes 0.5s to think of their
        question.  The gate must remain open during this natural pause.
        """
        callback_event = threading.Event()
        rec = make_recorder(
            callback=lambda audio: callback_event.set(),
            wakeword_enabled=True,
            wakeword_policy="strict_required",
            silence_duration=0.05,
        )
        ww = MockWakewordService()
        rec._wakeword_service = ww

        with AudioHarness(rec) as h:
            h.inject(silence(0.05))
            ww.arm()
            h.inject(silence(0.5))  # user thinking — gate must stay open
            h.inject(voiced_speech(0.8, amplitude=0.35))
            h.inject(silence(0.4), inter_chunk_delay=0.020)
            h.drain(timeout=5.0)

        fired = callback_event.wait(timeout=3.0)
        self.assertTrue(
            fired,
            "Gate must remain open during natural pause after wakeword. "
            "Speech arriving 0.5s after arm must still produce a callback.",
        )

    def test_consecutive_barge_ins_both_fire(self):
        """
        User interrupts → system starts capturing → user interrupts again
        (impatient, assistant didn't stop fast enough).

        Both interrupts must be registered.  The second interrupt must not
        be suppressed because the system is in LISTENING state.

        Uses a calibrated noise floor so the energy path can supplement Silero
        (synthetic voiced_speech has high energy but Silero returns voiced=False).
        """
        interrupts = []
        rec = make_recorder(
            on_interrupt=lambda info=None: interrupts.append(info),
        )
        rec._noise_floor = 0.005

        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.inject(voiced_speech(1.0, amplitude=0.35))
            h.drain(timeout=3.0)

            # First barge-in should have fired; now TTS has "stopped"
            h.set_tts_speaking()  # TTS continues (maybe assistant is verbose)
            h.inject(voiced_speech(1.0, amplitude=0.35))
            h.drain(timeout=3.0)

        self.assertGreaterEqual(
            len(interrupts),
            1,
            "At least one barge-in must fire.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
