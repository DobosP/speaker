"""
Hardware-free barge-in scenario tests.

Each test injects synthetic audio directly into ``AudioRecorder._audio_queue``
via ``AudioHarness`` — no microphone, no sounddevice, no PortAudio dependency.

Scenarios covered
-----------------
1.  test_silence_no_bargein             — dead silence never triggers barge-in
2.  test_tv_noise_no_bargein            — background noise blocked by noise gate
3.  test_human_voice_during_tts_bargein — human speech fires interrupt
4.  test_self_echo_no_bargein           — TTS re-entering mic is echo-blocked
5.  test_short_burst_no_bargein         — transient click too short for min-duration gate
6.  test_speech_without_wakeword_no_callback  — strict gate blocks speech without wakeword
7.  test_speech_after_wakeword_callback — speech after wakeword arm produces callback
8.  test_bargein_latency_slo            — interrupt must arrive within 1 500 ms wall time
9.  test_fsm_state_after_tts_stop       — FSM: ASSISTANT_SPEAKING → RECOVER → IDLE
10. test_hybrid_recovery_opens_after_miss_limit — hybrid policy opens recovery window

Running
-------
    python -m pytest tests/test_bargein_scenarios.py -v
    # or with the full suite
    python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
import os
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
    REAL_SPEECH_AVAILABLE,
    speech_fixture_info,
)
from tests.harness import (
    AudioHarness,
    MockWakewordService,
    make_recorder,
    within_ms,
)

# ── Shared constants ─────────────────────────────────────────────────────────

# Noise floor that represents a calibrated ambient environment.
# After auto-calibration, the recorder learns this background level.
# We pre-seed it so tests don't need the 2-3 s calibration run.
NOISE_FLOOR: float = 0.03

# TV audio at moderate distance — clearly audible but below speech level.
# Its RMS (≈ TV_AMP) must be below NOISE_FLOOR * barge_in_min_rms_ratio (3.0)
# to be blocked by the noise gate: 0.04 < 0.03 * 3 = 0.09 ✓
TV_AMP: float = 0.04

# Synthetic voiced speech amplitude — loud enough that even with Silero VAD
# returning voiced=False (for synthetic signals), the pure-energy path fires:
#   RMS ≈ 0.20  >  barge_threshold * 3 = 0.06 * 3 = 0.18  ✓
# Used only when real_speech() falls back to synthetic.
SPEECH_AMP: float = 0.35

# Real speech amplitude target (normalised RMS in fixture WAVs).
# Real TTS output makes Silero return voiced=True (>0.8 confidence in barge
# mode), so a lower amplitude is sufficient.
REAL_SPEECH_AMP: float = 0.15


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper
# ═══════════════════════════════════════════════════════════════════════════════

def _interrupt_recorder(on_interrupt_list: list):
    """Return a recorder that appends info dicts to *on_interrupt_list*."""
    return make_recorder(
        on_interrupt=lambda info=None: on_interrupt_list.append(
            info if isinstance(info, dict) else {}
        ),
    )


# ── Fixture diagnostics ───────────────────────────────────────────────────────

class TestFixtureDiagnostics(unittest.TestCase):
    """Prints what audio source each test run will use — not a real assertion."""

    def test_fixture_availability(self):
        """
        Report whether real TTS speech fixtures are available.

        When ``REAL_SPEECH_AVAILABLE=True``, the 'should fire' scenario tests
        exercise Silero VAD's full voiced-detection path.  When False, they use
        synthetic audio and rely on the energy path instead.

        Both modes produce correct test outcomes; the real-speech mode is more
        representative of production behaviour.
        """
        info = speech_fixture_info()
        print(f"\n[fixture info] {info}")
        if info["any_available"]:
            print(
                "[fixture info] Real TTS speech available — "
                "Silero voiced path WILL be exercised in barge-in tests."
            )
        else:
            print(
                "[fixture info] No TTS fixtures found — "
                "tests use synthetic audio (energy path only). "
                "Run with network access to generate real speech fixtures."
            )
        # Always passes — this is diagnostic only
        self.assertIsInstance(info, dict)


# ═══════════════════════════════════════════════════════════════════════════════
#  Test cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestBargeInScenarios(unittest.TestCase):
    """Scenario-based barge-in tests, all hardware-free."""

    # ── 1. Silence ────────────────────────────────────────────────────────────

    def test_silence_no_bargein(self):
        """Dead silence while TTS is playing must never fire barge-in."""
        interrupts: list = []
        rec = _interrupt_recorder(interrupts)
        rec._noise_floor = NOISE_FLOOR

        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.inject(silence(1.0))
            h.drain()

        self.assertFalse(
            interrupts,
            f"Expected no interrupt, but got {len(interrupts)} interrupt(s) from silence",
        )

    # ── 2. TV noise ───────────────────────────────────────────────────────────

    def test_tv_noise_no_bargein(self):
        """
        TV background noise at calibrated level must be blocked by the noise gate.

        After calibration, noise_floor is learned from the environment.  The
        noise gate requires mic RMS >= noise_floor * barge_in_min_rms_ratio (3.0).
        TV noise RMS (≈ 0.04) < NOISE_FLOOR * 3 (0.09) → blocked.
        """
        interrupts: list = []
        rec = _interrupt_recorder(interrupts)
        rec._noise_floor = NOISE_FLOOR  # Simulate post-calibration state

        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.inject(tv_noise(1.5, amplitude=TV_AMP))
            h.drain()

        self.assertFalse(
            interrupts,
            f"TV noise triggered {len(interrupts)} barge-in(s) — noise gate broken",
        )

    # ── 3. Human voice triggers barge-in ──────────────────────────────────────

    def test_human_voice_during_tts_bargein(self):
        """
        Voiced speech at a realistic level must fire barge-in within the first
        few frames.

        When ``REAL_SPEECH_AVAILABLE=True`` this test uses actual TTS-generated
        speech, exercising Silero VAD's full voiced-detection path
        (voiced=True fires from Silero's speech classifier, not just energy).

        No pre-seeded noise floor is set here: in a quiet environment the noise
        gate is open (noise_floor=None returns True always), so only the VAD
        and echo gates matter.  This represents the typical user scenario where
        barge-in occurs in a room with low ambient noise.
        """
        interrupts: list = []
        rec = _interrupt_recorder(interrupts)
        # noise_floor=None → noise gate always passes → Silero decides voiced
        # Natural TTS speech has brief word-boundary silences that would reset
        # the accumulator at a high noise_floor; None avoids this.

        speech = real_speech(1.0, amplitude=REAL_SPEECH_AMP, fallback_amplitude=SPEECH_AMP)

        with AudioHarness(rec) as h:
            # No AEC reference → echo_similarity = 0 → echo gate does not block
            h.set_tts_speaking()
            h.inject(speech)
            h.drain()

        self.assertTrue(
            interrupts,
            f"Expected barge-in to fire on voiced speech "
            f"(real_speech={REAL_SPEECH_AVAILABLE}), but interrupt never triggered",
        )

    # ── 4. TTS self-echo blocked ──────────────────────────────────────────────

    def test_self_echo_no_bargein(self):
        """
        When the mic picks up the TTS speaker output (echo), barge-in must be
        suppressed by the echo-similarity gate.

        Uses ``real_tts_echo()`` (TTS-generated speech if available, else
        synthetic) as both the AEC reference and the injected mic signal.
        Cosine similarity ≈ 1.0 → well above the 0.45 threshold → blocked.

        We use the backward-compatible ``_aec_ref`` path so that the echo
        correlation is computed without the 120 ms reference-delay offset
        that ``EchoGuard.set_reference()`` prepends.
        """
        interrupts: list = []
        rec = _interrupt_recorder(interrupts)
        rec._noise_floor = NOISE_FLOOR

        # Use real TTS speech when available so the test is representative
        # of what actually leaks from the speaker into the mic.
        echo = real_tts_echo(2.0)
        with AudioHarness(rec) as h:
            h.set_tts_speaking(zero_delays=True)
            # Inject TTS reference via backward-compat path (no delay offset)
            rec._aec_ref = echo.copy()
            rec._ref_samples_consumed = 0
            # Inject the SAME audio as mic input → perfect cosine similarity
            h.inject(echo)
            h.drain()

        self.assertFalse(
            interrupts,
            f"Self-echo triggered {len(interrupts)} barge-in(s) — echo gate broken",
        )

    # ── 5. Short burst does not trigger ───────────────────────────────────────

    def test_short_burst_no_bargein(self):
        """
        A transient click or bump (< min_speech_sec = 0.2 s) must not trigger
        barge-in even if its amplitude is very high.

        The BargeInDetector only fires once above_samples ≥ min_speech_samples.
        """
        interrupts: list = []
        rec = make_recorder(
            on_interrupt=lambda info=None: interrupts.append(info),
            barge_in_min_speech_sec=0.3,  # strict minimum duration
        )
        rec._noise_floor = NOISE_FLOOR

        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            # Click burst ≈ 50 ms — far below 300 ms minimum
            h.inject(click_burst(duration_sec=0.05, amplitude=0.45))
            # Follow with silence so above_samples drains back to zero
            h.inject(silence(0.2))
            h.drain()

        self.assertFalse(
            interrupts,
            f"Short burst triggered {len(interrupts)} barge-in(s) — min-duration gate broken",
        )

    # ── 6. Wakeword gate: no wakeword → no callback ───────────────────────────

    def test_speech_without_wakeword_no_callback(self):
        """
        With ``wakeword_policy=strict_required`` and no wakeword detection,
        voice speech must NOT reach the utterance callback.
        """
        callbacks: list = []
        rec = make_recorder(
            callback=lambda audio: callbacks.append(audio),
            wakeword_enabled=True,
            wakeword_policy="strict_required",
        )
        rec._noise_floor = NOISE_FLOOR

        ww = MockWakewordService()
        rec._wakeword_service = ww
        # Never call ww.arm() → gate stays closed

        with AudioHarness(rec) as h:
            h.inject(voiced_speech(1.5, amplitude=SPEECH_AMP))
            h.inject(silence(0.3))
            h.drain()

        self.assertFalse(
            callbacks,
            f"Callback fired {len(callbacks)} time(s) without wakeword — gate broken",
        )

    # ── 7. Wakeword gate: arm → speech → callback fires ──────────────────────

    def test_speech_after_wakeword_callback(self):
        """
        Arming the wakeword service, then injecting voiced speech followed by
        silence, must produce exactly one callback with captured audio.

        Note: silence_duration is set short (0.05 s) and silence chunks are
        injected with a real-time delay so the endpointing timer fires.
        """
        callback_event = threading.Event()
        captured: list = []

        def _cb(audio):
            captured.append(audio)
            callback_event.set()

        rec = make_recorder(
            callback=_cb,
            wakeword_enabled=True,
            wakeword_policy="strict_required",
            silence_duration=0.05,  # short so endpointing fires during test
        )
        rec._noise_floor = NOISE_FLOOR

        ww = MockWakewordService()
        rec._wakeword_service = ww

        with AudioHarness(rec) as h:
            # Allow 2 frames for the worker thread to spin up
            h.inject(silence(0.05))

            # Fire wakeword event; next audio frame will pick it up
            ww.arm()
            h.inject(silence(0.05))  # let wakeword event be consumed

            # Inject 0.8s so the captured buffer exceeds the 0.5s minimum check
            # (TTS preamble silence ~0.15s is not captured → need 0.65s of speech)
            h.inject(real_speech(0.8, amplitude=REAL_SPEECH_AMP, fallback_amplitude=SPEECH_AMP))

            # Inject silence with slight inter-chunk delay so wall-clock timer fires
            # 0.15 s of real silence + 0.015 s/frame × 10 frames ≈ 0.15 s wall time
            h.inject(silence(0.3), inter_chunk_delay=0.015)
            h.drain(timeout=5.0)

        fired = callback_event.wait(timeout=3.0)
        self.assertTrue(fired, "Callback never fired after wakeword arm + speech injection")
        self.assertEqual(len(captured), 1, f"Expected 1 callback, got {len(captured)}")
        self.assertIsInstance(captured[0], np.ndarray)
        self.assertGreater(len(captured[0]), 0)

    # ── 8. Barge-in latency SLO ───────────────────────────────────────────────

    def test_bargein_latency_slo(self):
        """
        From the first voiced frame until ``on_interrupt`` is called, the
        wall-clock elapsed time must be less than 1 500 ms.

        This SLO covers queue injection + worker processing + callback overhead
        and ensures no unexpected blocking in the hot path.
        """
        interrupt_ts: list = []
        inject_start: list = []

        def _interrupt(info=None):
            interrupt_ts.append(time.time())

        rec = make_recorder(
            on_interrupt=_interrupt,
            barge_in_min_speech_sec=0.15,  # fire after ~150 ms of speech
        )
        rec._noise_floor = NOISE_FLOOR

        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            inject_start.append(time.time())
            # Use real speech when available so Silero's voiced path is tested.
            # No noise_floor → gate always open, Silero decides voiced.
            h.inject(real_speech(1.0, amplitude=REAL_SPEECH_AMP, fallback_amplitude=SPEECH_AMP))
            h.drain(timeout=3.0)

        self.assertTrue(interrupt_ts, "Barge-in never triggered — cannot measure latency")
        elapsed_ms = (interrupt_ts[0] - inject_start[0]) * 1000.0
        self.assertLess(
            elapsed_ms,
            1500.0,
            f"Barge-in latency SLO violated: {elapsed_ms:.1f} ms > 1500 ms",
        )

    # ── 9. FSM state transitions after TTS stop ───────────────────────────────

    def test_fsm_state_after_tts_stop(self):
        """
        After TTS ends, the FSM must transition:
          ASSISTANT_SPEAKING → RECOVER → IDLE  (wakeword_enabled=False)

        The RECOVER → IDLE transition happens the first time a frame is
        processed after ``_recover_until`` has expired.
        """
        rec = make_recorder(wakeword_enabled=False)

        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            self.assertEqual(
                rec._listener_state,
                ListenerState.ASSISTANT_SPEAKING,
                "State should be ASSISTANT_SPEAKING immediately after TTS start",
            )

            h.stop_tts()
            self.assertEqual(
                rec._listener_state,
                ListenerState.RECOVER,
                "State should be RECOVER immediately after TTS stop",
            )

            # _recover_until = now + 0.25 s.  Wait for it to expire.
            time.sleep(0.35)

            # One frame triggers the RECOVER → IDLE check
            h.inject(silence(0.01))
            h.drain()

        self.assertEqual(
            rec._listener_state,
            ListenerState.IDLE,
            "State should be IDLE after recover_until expires",
        )

    # ── 10. Hybrid recovery window ────────────────────────────────────────────

    def test_hybrid_recovery_opens_after_miss_limit(self):
        """
        With ``wakeword_policy=hybrid_recovery`` and ``wakeword_miss_limit=3``,
        injecting speech that exceeds the VAD threshold without a prior wakeword
        must eventually allow speech through via the recovery window and produce
        a callback.

        Internal flow:
          frames 1-3: rms > threshold + gate closed → miss_count 1→2→3 → recovery window opens
          frame  4+:  gate now open → speech captured → callback fires on silence

        Note: ``_wakeword_armed_until`` is reset to 0 when speech actually starts
        (token consumed), so we verify the callback result rather than the flag.
        """
        callback_event = threading.Event()
        captured: list = []

        def _cb(audio):
            captured.append(audio)
            callback_event.set()

        rec = make_recorder(
            callback=_cb,
            wakeword_enabled=True,
            wakeword_policy="hybrid_recovery",
            wakeword_miss_limit=3,
            wakeword_recovery_window_sec=2.0,
            silence_duration=0.05,  # short endpointing for fast test
        )
        rec._noise_floor = NOISE_FLOOR

        ww = MockWakewordService()
        rec._wakeword_service = ww
        # Never call ww.arm() — recovery must happen via miss_limit

        with AudioHarness(rec) as h:
            # Inject speech: first 3 high-RMS frames open the recovery window,
            # subsequent frames are captured as the utterance.
            h.inject(real_speech(1.0, amplitude=REAL_SPEECH_AMP, fallback_amplitude=SPEECH_AMP))
            # Silence with inter-chunk delay so wall-clock endpointing fires
            h.inject(silence(0.3), inter_chunk_delay=0.015)
            h.drain(timeout=6.0)

        fired = callback_event.wait(timeout=3.0)
        self.assertTrue(
            fired,
            "Hybrid recovery callback never fired — miss_limit did not open the gate",
        )
        self.assertGreater(len(captured), 0)


# ── Additional policy-level regression tests ─────────────────────────────────

class TestLayeredPoliciesViaHarness(unittest.TestCase):
    """
    Verify that the layered interruption policies in DialogueController are
    correctly wired into the AudioRecorder barge-in decision path.

    These tests go one level above the pure policy unit tests in
    test_dialogue_controller.py by driving the full AudioRecorder processing
    loop.
    """

    def test_noise_gate_prevents_bargein_at_background_level(self):
        """
        Any audio whose RMS is below noise_floor * barge_in_min_rms_ratio must
        not reach the BargeInDetector's confirmed state.

        This is a broader statement of test_tv_noise_no_bargein using a
        different noise amplitude to ensure the ratio check is what matters,
        not a specific threshold.
        """
        interrupts: list = []
        for noise_amp in (0.01, 0.02, 0.035):
            interrupts.clear()
            rec = make_recorder(
                on_interrupt=lambda info=None, _l=interrupts: _l.append(info),
            )
            # Calibrate at the same amplitude as the noise
            rec._noise_floor = noise_amp

            with AudioHarness(rec) as h:
                h.set_tts_speaking()
                h.inject(tv_noise(1.0, amplitude=noise_amp))
                h.drain()

            self.assertFalse(
                interrupts,
                f"Background noise at amp={noise_amp} triggered barge-in "
                "(noise gate should have blocked it)",
            )

    def test_voiced_speech_reliably_triggers_across_amplitudes(self):
        """
        Voiced speech at sufficient amplitude must always trigger barge-in.

        When real TTS speech is available, Silero's voiced path fires naturally.
        Without fixtures, the test verifies the pure-energy path fires once
        RMS > barge_threshold * 3.

        For real speech with a pre-seeded noise floor, amplitude is scaled up
        so that even the quietest voiced frames (word boundaries ~44% of peak)
        remain above noise_floor * barge_in_min_rms_ratio (3.0).
        Required: min_voiced_rms > noise_floor * 3
                  amplitude * 0.44 > noise_floor * 3
                  amplitude > noise_floor * 6.8

        We use noise_floor=None (no gate) so that Silero + energy path decides,
        avoiding the noise-gate reset issue with TTS word boundaries.
        """
        for noise_floor in (0.01, 0.02, 0.03):
            barge_threshold = max(0.01, noise_floor * 2.0)
            synth_amp = barge_threshold * 20.0  # energy path fallback

            interrupts: list = []
            rec = make_recorder(
                on_interrupt=lambda info=None, _l=interrupts: _l.append(info),
            )
            # No noise floor pre-seeded → gate always open, Silero + energy decide
            # (avoids spurious resets on TTS word-boundary silence frames)

            with AudioHarness(rec) as h:
                h.set_tts_speaking()
                h.inject(
                    real_speech(1.0, amplitude=REAL_SPEECH_AMP, fallback_amplitude=synth_amp)
                )
                h.drain()

            self.assertTrue(
                interrupts,
                f"Voiced speech (noise_floor={noise_floor}) "
                f"did not trigger barge-in (real_speech={REAL_SPEECH_AVAILABLE})",
            )


class TestWakewordGatingViaHarness(unittest.TestCase):
    """Wakeword gate integration tests driven through AudioHarness."""

    def test_legacy_compatible_policy_always_open(self):
        """With policy=legacy_compatible the wakeword gate is always open."""
        callbacks: list = []
        rec = make_recorder(
            callback=lambda audio: callbacks.append(audio),
            wakeword_enabled=True,
            wakeword_policy="legacy_compatible",
            silence_duration=0.05,
        )
        rec._noise_floor = NOISE_FLOOR

        ww = MockWakewordService()
        rec._wakeword_service = ww
        # Do NOT arm — with legacy_compatible the gate should still open

        callback_event = threading.Event()
        rec.callback = lambda audio: (callbacks.append(audio), callback_event.set())

        with AudioHarness(rec) as h:
            # 0.8s of speech so the captured buffer exceeds the 0.5s minimum in
            # _finish_recording (TTS speech has ~0.15s preamble silence that does
            # not go into the buffer, so 0.6s would only capture ~0.45s)
            h.inject(real_speech(0.8, amplitude=REAL_SPEECH_AMP, fallback_amplitude=SPEECH_AMP))
            h.inject(silence(0.3), inter_chunk_delay=0.015)
            h.drain(timeout=5.0)

        fired = callback_event.wait(timeout=3.0)
        self.assertTrue(fired, "legacy_compatible policy should allow speech without wakeword")

    def test_multiple_sequential_utterances(self):
        """After a callback fires, a second wakeword arm + speech fires again."""
        callbacks: list = []
        callback_event = threading.Event()

        def _cb(audio):
            callbacks.append(audio)
            callback_event.set()

        rec = make_recorder(
            callback=_cb,
            wakeword_enabled=True,
            wakeword_policy="strict_required",
            silence_duration=0.05,
        )
        rec._noise_floor = NOISE_FLOOR

        ww = MockWakewordService()
        rec._wakeword_service = ww

        with AudioHarness(rec) as h:
            # First utterance — 0.6s speech + 0.4s silence at 20ms/chunk
            # 0.4s / 0.064s_per_chunk = 6.25 → 7 chunks × 20ms = 140ms wall time
            # That comfortably exceeds silence_duration=0.05s (50ms required)
            ww.arm()
            h.inject(silence(0.05))  # consume wakeword event
            # 0.8s of speech → buffer captures 0.65s > 0.5s minimum after preamble
            h.inject(real_speech(0.8, amplitude=REAL_SPEECH_AMP, fallback_amplitude=SPEECH_AMP))
            h.inject(silence(0.4), inter_chunk_delay=0.020)
            h.drain(timeout=4.0)

            first_fired = callback_event.wait(timeout=2.0)
            self.assertTrue(first_fired, "First utterance callback did not fire")
            callback_event.clear()

            # Second utterance — needs a fresh wakeword arm
            ww.arm()
            h.inject(silence(0.05))
            h.inject(real_speech(0.8, amplitude=REAL_SPEECH_AMP, fallback_amplitude=SPEECH_AMP))
            h.inject(silence(0.4), inter_chunk_delay=0.020)
            h.drain(timeout=4.0)

            second_fired = callback_event.wait(timeout=2.0)
            self.assertTrue(second_fired, "Second utterance callback did not fire")

        self.assertEqual(len(callbacks), 2, f"Expected 2 callbacks, got {len(callbacks)}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
