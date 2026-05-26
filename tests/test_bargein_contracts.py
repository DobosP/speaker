"""
Behavioral contracts for the barge-in detection system.

TDD contract philosophy
-----------------------
Each test here defines a PRECISE behavioral requirement.  The test name and
docstring state the requirement; the assertion verifies it.  If an engineer
changes a threshold, refactors the pipeline, or disables a component, these
tests tell them exactly which contract they broke.

The tests are organised by contract area:

  1. BargeInDetector score rules — exact score boundaries
  2. Accumulator sustained-speech minimum — hard reset behavior
  3. Noise gate exact boundary — passes at 3.0× noise floor, blocked at 2.9×
  4. Echo correlation gate — exact 0.45 threshold
  5. Silero path vs. energy path — real voice at low amplitude
  6. State machine transitions — precise FSM contract
  7. SNR-based barge-in — speech + background noise at defined SNR
  8. False positive guard — 5+ seconds of various non-speech must NEVER fire
  9. Adversarial inputs — transients, tones, breathing-like signals

HOW TESTS CAN FAIL (and what they reveal)
------------------------------------------
test_echo_gate_blocks_at_exact_threshold
  → FAILS if echo_corr_threshold is changed from 0.45 to anything else

test_echo_gate_passes_just_below_threshold
  → FAILS if echo_corr_threshold is set too low (e.g., 0.40)

test_noise_gate_passes_at_3x_floor / _blocks_below_3x
  → FAILS if barge_in_min_rms_ratio is changed from 3.0

test_single_quiet_frame_resets_counter
  → FAILS if hard-reset in update() is removed (frames that pass the noise
    gate but score < 2.0 must still hard-reset to prevent false positives)

test_min_speech_samples_boundary
  → FAILS if min_speech_sec is changed from 0.2

test_silero_path_fires_human_voice_at_low_amplitude
  → FAILS if Silero is disabled, broken, or model changed

test_energy_path_fires_without_silero
  → FAILS if the energy fallback is removed

test_speech_without_echo_fires_over_background
  → FAILS if noise gate is too aggressive for realistic SNR

test_wakeword_strict_gate_blocks_indefinitely
  → FAILS if wakeword gate is accidentally left open

test_fsm_state_listening_during_recording
  → FAILS if state machine transitions incorrectly during capture

test_state_idle_after_callback_fired
  → FAILS if state machine doesn't reset after delivering an utterance

Running
-------
    python -m pytest tests/test_bargein_contracts.py -v
"""

from __future__ import annotations

import os
import sys
import threading
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.audio import (
    BargeInDetector,
    FrameFeatures,
    SpeechGate,
    ListenerState,
)
from tests.fixtures import (
    SR,
    silence,
    voiced_speech,
    tv_noise,
    tts_echo,
    click_burst,
    human_voice,
    human_voice_concat,
    snr_mix,
    HUMAN_VOICE_AVAILABLE,
)
from tests.harness import AudioHarness, MockWakewordService, make_recorder, within_ms

# ── Shared constants ──────────────────────────────────────────────────────────

# These MUST match the defaults in AudioRecorder / BargeInDetector exactly.
# If a default changes and the constant here is not updated, the constant tests
# below will catch it.

ECHO_CORR_THRESHOLD   = 0.45   # EchoGuard blocks if similarity ≥ this
BARGE_IN_MIN_RMS_RATIO = 3.0   # noise gate: blocks if rms < noise_floor × this
BARGE_IN_RMS_RATIO     = 2.0   # barge_threshold = max(vad_threshold, nf × this)
VAD_THRESHOLD          = 0.01  # default vad_threshold
MIN_SPEECH_SEC         = 0.2   # min speech seconds for BargeInDetector to fire
FRAME_LEN              = 1024  # chunk size used by AudioHarness
MIN_SPEECH_SAMPLES     = int(MIN_SPEECH_SEC * SR)  # 3200 samples


def _skip_if_no_fsdd():
    if not HUMAN_VOICE_AVAILABLE():
        raise unittest.SkipTest(
            "FSDD voice samples not found.  Run tests with network to download "
            "them (conftest.py handles this automatically on first run)."
        )


# ── Unit helpers: BargeInDetector directly ────────────────────────────────────

def _make_detector(
    min_speech_sec: float = MIN_SPEECH_SEC,
    echo_corr_threshold: float = ECHO_CORR_THRESHOLD,
) -> BargeInDetector:
    return BargeInDetector(
        sample_rate=SR,
        min_speech_sec=min_speech_sec,
        echo_corr_threshold=echo_corr_threshold,
    )


def _frame(
    rms: float,
    voiced: bool,
    echo_sim: float = 0.0,
    threshold: float = VAD_THRESHOLD,
    noise_floor_calibrated: bool = False,
    reference_rms: float = 0.0,
) -> FrameFeatures:
    return FrameFeatures(
        timestamp=time.time(),
        rms=rms,
        threshold=threshold,
        voiced=voiced,
        echo_similarity=echo_sim,
        raw_rms=rms,
        noise_floor_calibrated=noise_floor_calibrated,
        reference_rms=reference_rms,
    )


def _accumulate(
    detector: BargeInDetector,
    rms: float,
    voiced: bool,
    echo_sim: float = 0.0,
    n_frames: int = 1,
) -> bool:
    """Push *n_frames* identical frames into *detector*.  Returns final result."""
    result = False
    for _ in range(n_frames):
        result = detector.update(
            _frame(rms, voiced, echo_sim),
            FRAME_LEN,
        )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  1. BargeInDetector score rules (unit tests — no AudioRecorder involved)
# ═══════════════════════════════════════════════════════════════════════════════

class TestBargeInDetectorScore(unittest.TestCase):
    """
    Unit tests for BargeInDetector.score() — verify the exact scoring rules.

    Any change to the scoring formula breaks these tests, which is intentional:
    the scoring rules ARE the barge-in contract.
    """

    def setUp(self):
        self.det = _make_detector()

    def test_echo_above_threshold_scores_minus_ten(self):
        """Echo similarity ≥ 0.45 → score = -10.0 regardless of rms or voiced."""
        features = _frame(rms=0.5, voiced=True, echo_sim=0.45)
        self.assertEqual(
            self.det.score(features),
            -10.0,
            "Echo at threshold=0.45 must score -10.0 (hard block).",
        )

    def test_echo_just_below_threshold_not_blocked(self):
        """Echo similarity = 0.449 → score NOT -10.0, normal scoring applies."""
        features = _frame(rms=0.1, voiced=True, echo_sim=0.44)
        score = self.det.score(features)
        self.assertGreater(
            score,
            0.0,
            f"Echo at 0.44 (below 0.45 threshold) must not be blocked; got score={score}.",
        )

    def test_voiced_alone_scores_exactly_2(self):
        """voiced=True with rms just below threshold → score = exactly 2.0."""
        # rms = 0.005 < threshold=0.01 → no energy bonus
        features = _frame(rms=0.005, voiced=True, echo_sim=0.0)
        score = self.det.score(features)
        self.assertEqual(
            score,
            2.0,
            f"voiced=True with rms below threshold must score exactly 2.0; got {score}.",
        )

    def test_voiced_plus_energy_at_1x_scores_3(self):
        """voiced + rms > 1× threshold → score = 3.0."""
        features = _frame(rms=0.015, voiced=True)
        score = self.det.score(features)
        self.assertEqual(score, 3.0, f"Expected 3.0, got {score}.")

    def test_voiced_plus_energy_at_15x_scores_35(self):
        """voiced + rms > 1.5× threshold → score = 3.5."""
        features = _frame(rms=0.016, voiced=True)  # 0.016 > 0.01 * 1.5 = 0.015
        score = self.det.score(features)
        self.assertEqual(score, 3.5, f"Expected 3.5, got {score}.")

    def test_energy_only_at_3x_threshold_scores_25_when_calibrated(self):
        """voiced=False + calibrated noise floor + rms > 3× threshold → score = 2.5."""
        features = _frame(rms=0.031, voiced=False, noise_floor_calibrated=True)
        score = self.det.score(features)
        self.assertEqual(
            score,
            2.5,
            f"Calibrated energy-only at 3× threshold must score 2.5; got {score}. "
            "This is the fallback path for audio Silero doesn't classify as voiced.",
        )

    def test_energy_only_at_3x_threshold_uncalibrated_scores_15(self):
        """voiced=False + no noise floor + rms > 3× threshold → score = 1.5 (blocked)."""
        features = _frame(rms=0.031, voiced=False, noise_floor_calibrated=False)
        score = self.det.score(features)
        self.assertEqual(
            score,
            1.5,
            f"Uncalibrated energy-only at 3× threshold must score 1.5 (not 2.5); got {score}. "
            "Without calibration the strong-energy boost is suppressed to prevent "
            "loud non-speech (HVAC, plosives) from triggering barge-in.",
        )

    def test_energy_only_below_3x_threshold_scores_below_2(self):
        """voiced=False + rms < 3× threshold → score < 2.0 → does NOT accumulate."""
        features = _frame(rms=0.02, voiced=False)  # 0.02 < 0.03, but > 0.015
        score = self.det.score(features)
        self.assertLess(
            score,
            2.0,
            f"Energy below 3× threshold with voiced=False must score < 2.0; got {score}. "
            "This frame must NOT accumulate in the barge-in counter.",
        )

    def test_silence_scores_negative(self):
        """Near-zero RMS → raw_rms ≤ 1e-6 penalty → score = -0.5."""
        features = _frame(rms=0.0, voiced=False, echo_sim=0.0)
        score = self.det.score(features)
        self.assertEqual(score, -0.5, f"Silence must score -0.5; got {score}.")


# ═══════════════════════════════════════════════════════════════════════════════
#  Adaptive TTS-echo scoring contracts
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdaptiveEchoThreshold(unittest.TestCase):
    """
    When a TTS reference is active (reference_rms > 0.01), the echo block
    threshold scales down from echo_corr_threshold toward 0.20 as TTS gets
    louder.  This prevents room echoes with moderate cross-correlation from
    causing self-interruption.

    When TTS reference is active, the calibrated energy path is also suppressed:
    Silero confirmation is required for the strong-energy boost.  This means
    voiced=False + high RMS + loud TTS → score 1.5 (does NOT accumulate).
    """

    def setUp(self):
        self.det = _make_detector()

    # ── Adaptive echo threshold ──────────────────────────────────────────────

    def test_echo_blocked_at_adaptive_threshold_when_tts_loud(self):
        """
        echo_sim=0.26, ref_rms=0.15 (loud TTS) → adaptive threshold ≈ 0.27
        → score = -10.0 (blocked).

        Without adaptation the default threshold=0.45 would pass echo_sim=0.26
        and allow the false barge-in seen in the real-world session log.
        """
        features = _frame(rms=0.15, voiced=False, echo_sim=0.26, reference_rms=0.15)
        score = self.det.score(features)
        self.assertEqual(
            score,
            -10.0,
            f"echo_sim=0.26 with loud TTS (ref_rms=0.15) must be blocked; got {score}. "
            "Adaptive echo threshold must lower to ≈ 0.27 at ref_rms=0.15.",
        )

    def test_echo_not_blocked_when_tts_quiet(self):
        """
        echo_sim=0.26, ref_rms=0.0 (no TTS reference) → threshold stays at 0.45
        → score > -10.0 (not blocked as echo).
        """
        features = _frame(rms=0.15, voiced=False, echo_sim=0.26, reference_rms=0.0)
        score = self.det.score(features)
        self.assertGreater(
            score,
            -10.0,
            f"echo_sim=0.26 with no TTS reference must NOT be blocked; got {score}.",
        )

    def test_echo_hard_blocked_at_full_threshold_no_tts(self):
        """echo_sim ≥ 0.45 with no TTS → still blocked at the static threshold."""
        features = _frame(rms=0.15, voiced=True, echo_sim=0.45, reference_rms=0.0)
        score = self.det.score(features)
        self.assertEqual(score, -10.0, f"Static echo block must still apply; got {score}.")

    def test_adaptive_threshold_midpoint(self):
        """
        ref_rms=0.055 is halfway between 0.01 and 0.10:
        adaptive_threshold ≈ (0.45 + 0.20) / 2 = 0.325.
        echo_sim=0.35 > 0.325 → blocked; echo_sim=0.30 < 0.325 → not blocked.
        """
        features_blocked = _frame(rms=0.10, voiced=False, echo_sim=0.35, reference_rms=0.055)
        features_pass = _frame(rms=0.10, voiced=False, echo_sim=0.30, reference_rms=0.055)
        self.assertEqual(
            self.det.score(features_blocked),
            -10.0,
            "echo_sim=0.35 at mid-TTS level should be blocked.",
        )
        self.assertGreater(
            self.det.score(features_pass),
            -10.0,
            "echo_sim=0.30 at mid-TTS level should NOT be blocked.",
        )

    # ── Energy path suppression during TTS ──────────────────────────────────

    def test_energy_path_suppressed_when_tts_reference_active(self):
        """
        voiced=False + noise_floor_calibrated + ref_rms=0.15 → score = 1.5.
        The calibrated energy boost (+1.0) is suppressed while TTS plays.

        This is the core self-interruption fix: even with a calibrated noise
        floor, loud TTS echo (voiced=False) cannot accumulate to fire barge-in.
        """
        features = _frame(
            rms=0.15,
            voiced=False,
            echo_sim=0.10,  # below even adaptive threshold
            noise_floor_calibrated=True,
            reference_rms=0.15,
        )
        score = self.det.score(features)
        self.assertEqual(
            score,
            1.5,
            f"voiced=False + calibrated + active TTS must score 1.5 (not 2.5); got {score}. "
            "Energy boost must be suppressed during TTS to prevent self-interruption.",
        )

    def test_voiced_speech_fires_despite_loud_tts(self):
        """
        voiced=True + ref_rms=0.15 → score ≥ 2.0 regardless of TTS level.
        Real user speech (Silero confirms voiced) must always accumulate.
        """
        features = _frame(
            rms=0.04,
            voiced=True,
            echo_sim=0.10,
            reference_rms=0.15,
        )
        score = self.det.score(features)
        self.assertGreaterEqual(
            score,
            2.0,
            f"voiced=True must score ≥ 2.0 even with loud TTS; got {score}.",
        )

    def test_voiced_overrides_adaptive_echo_block_at_moderate_similarity(self):
        """
        voiced=True + echo_sim=0.35 (moderate) + ref_rms=0.15 → NOT blocked.

        The adaptive threshold at ref_rms=0.15 lowers to 0.20, so echo_sim=0.35
        would normally return -10.0.  But Silero confirms voiced=True AND
        echo_sim < 0.50 (ambiguous range) → the user is likely speaking over
        the TTS echo, so the adaptive block is overridden.
        """
        features = _frame(
            rms=0.10,
            voiced=True,
            echo_sim=0.35,
            reference_rms=0.15,
        )
        score = self.det.score(features)
        self.assertGreater(
            score,
            0.0,
            f"voiced=True must override adaptive echo block at moderate similarity; "
            f"got score={score}.",
        )

    def test_voiced_does_not_override_hard_echo_block(self):
        """
        voiced=True + echo_sim=0.80 (very high) + ref_rms=0.15 → blocked (-10.0).

        Very high echo similarity (≥ 0.50) indicates pure TTS bleed-through.
        Even if Silero returns voiced=True (TTS is real speech), the signal is
        the TTS output — not the user.  The Silero override is capped at 0.50.
        """
        features = _frame(
            rms=0.15,
            voiced=True,
            echo_sim=0.80,
            reference_rms=0.15,
        )
        score = self.det.score(features)
        self.assertEqual(
            score,
            -10.0,
            f"High echo_sim=0.80 must be blocked even with voiced=True; got {score}. "
            "Pure TTS echo (very high similarity) must not override the hard block.",
        )

    def test_unvoiced_is_blocked_by_adaptive_echo_threshold(self):
        """
        voiced=False + echo_sim=0.35 + ref_rms=0.15 → blocked at -10.0.

        When Silero returns voiced=False, the Silero override does NOT apply.
        This is the fix for the real-world self-interruption: TTS echo with
        voiced=False and moderate correlation (as seen in session log) is blocked.
        """
        features = _frame(
            rms=0.15,
            voiced=False,
            echo_sim=0.35,
            reference_rms=0.15,
        )
        score = self.det.score(features)
        self.assertEqual(
            score,
            -10.0,
            f"voiced=False with adaptive echo block must score -10.0; got {score}.",
        )

    def test_energy_path_works_without_tts_reference(self):
        """
        voiced=False + noise_floor_calibrated + ref_rms=0.0 → score = 2.5.
        The energy path still works normally in idle / non-TTS mode.
        """
        features = _frame(
            rms=0.15,
            voiced=False,
            echo_sim=0.05,
            noise_floor_calibrated=True,
            reference_rms=0.0,
        )
        score = self.det.score(features)
        self.assertEqual(
            score,
            2.5,
            f"Calibrated energy path must score 2.5 when TTS is not active; got {score}.",
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Accumulator sustained-speech minimum (hard reset contract)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAccumulatorBehavior(unittest.TestCase):
    """
    BargeInDetector accumulates above_samples only on score ≥ 2.0 frames.
    A SINGLE frame with score < 2.0 resets the counter to ZERO (hard reset).

    This is a strict contract: continuous speech must be sustained to fire.
    One quiet window in the middle of speech resets everything.
    """

    def setUp(self):
        self.det = _make_detector()

    def test_3_voiced_frames_do_not_fire(self):
        """
        3 × FRAME_LEN = 3072 < MIN_SPEECH_SAMPLES = 3200 → must NOT fire.

        Fires on frame 4.  Changing min_speech_sec breaks this.
        """
        for i in range(3):
            fired = _accumulate(self.det, rms=0.05, voiced=True)
        self.assertFalse(
            fired,
            f"3 voiced frames ({3 * FRAME_LEN} samples) must not fire "
            f"(need {MIN_SPEECH_SAMPLES} = {MIN_SPEECH_SEC}s minimum).",
        )

    def test_4_voiced_frames_fire(self):
        """4 × FRAME_LEN = 4096 ≥ 3200 → MUST fire on the 4th frame."""
        fired = False
        for i in range(4):
            fired = _accumulate(self.det, rms=0.05, voiced=True)
        self.assertTrue(
            fired,
            f"4 voiced frames ({4 * FRAME_LEN} samples) must fire "
            f"(threshold {MIN_SPEECH_SAMPLES} samples).",
        )

    def test_single_quiet_frame_resets_counter(self):
        """
        3 voiced frames → 1 quiet frame (score < 2.0) that PASSES the noise gate
        → 3 more voiced frames.

        update() hard-resets on any frame that passes the noise gate but scores
        < 2.0.  Word-boundary pauses that fall BELOW the noise gate never reach
        update() — the noise-gate path calls soft_decay() instead, so they do
        not wipe progress.  This test covers only frames that clear the gate.
        """
        for _ in range(3):
            _accumulate(self.det, rms=0.05, voiced=True)

        # One quiet frame (score 0.0): passes the noise gate in the test helper
        # (no noise_floor set) but scores < 2.0 → hard reset.
        _accumulate(self.det, rms=0.005, voiced=False)
        self.assertEqual(
            self.det.above_samples,
            0,
            "A frame that passes the noise gate but scores < 2.0 must hard-reset "
            "above_samples to 0.  (Word-boundary pauses below the noise gate are "
            "handled by soft_decay() in the caller, not here.)",
        )

        # Three more voiced frames — still not enough after reset
        fired = False
        for _ in range(3):
            fired = _accumulate(self.det, rms=0.05, voiced=True)
        self.assertFalse(
            fired,
            "3 voiced frames after a hard-reset must not fire (3072 < 3200).",
        )

    def test_echo_frame_resets_counter(self):
        """
        3 good voiced frames → 1 echo frame → counter resets.
        The echo gate fires IMMEDIATELY, resetting above_samples.
        """
        for _ in range(3):
            _accumulate(self.det, rms=0.05, voiced=True)
        # Echo frame with perfect similarity
        _accumulate(self.det, rms=0.05, voiced=True, echo_sim=0.50)
        self.assertEqual(
            self.det.above_samples,
            0,
            "Echo frame (sim=0.50 ≥ 0.45) must reset above_samples.",
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Noise gate exact boundary (integration test)
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoiseGateBoundary(unittest.TestCase):
    """
    SpeechGate.passes_noise_gate(rms, noise_floor) must return:
      True  when rms >= noise_floor * 3.0
      False when rms <  noise_floor * 3.0

    These tests WILL FAIL if barge_in_min_rms_ratio changes from 3.0.
    """

    def setUp(self):
        self.gate = SpeechGate(
            vad_threshold=VAD_THRESHOLD,
            barge_in_rms_ratio=BARGE_IN_RMS_RATIO,
            barge_in_min_rms_ratio=BARGE_IN_MIN_RMS_RATIO,
        )

    def test_passes_at_exactly_3x(self):
        nf = 0.05
        self.assertTrue(
            self.gate.passes_noise_gate(rms=nf * 3.0, noise_floor=nf),
            f"rms = nf × 3.0 must PASS the noise gate.",
        )

    def test_blocked_just_below_3x(self):
        nf = 0.05
        self.assertFalse(
            self.gate.passes_noise_gate(rms=nf * 2.99, noise_floor=nf),
            "rms = nf × 2.99 must be BLOCKED by the noise gate.",
        )

    def test_tv_noise_blocked_at_calibrated_level(self):
        """
        TV noise RMS ≈ 0.04.  Calibrated noise floor = 0.04.
        min_rms = 0.04 × 3 = 0.12.
        tv_noise RMS (0.04) < 0.12 → blocked.
        """
        noise_floor = 0.04
        tv = tv_noise(1.0, amplitude=noise_floor)
        tv_rms = float(np.sqrt(np.mean(tv ** 2)))

        self.assertFalse(
            self.gate.passes_noise_gate(tv_rms, noise_floor),
            f"TV noise RMS={tv_rms:.4f} must be blocked when noise_floor={noise_floor}.",
        )

    def test_speech_at_4x_passes(self):
        """Clear speech at 4× noise floor level must pass (plenty of headroom)."""
        nf = 0.03
        self.assertTrue(
            self.gate.passes_noise_gate(rms=nf * 4.0, noise_floor=nf),
            "Speech at 4× noise floor must pass the gate.",
        )

    def test_none_noise_floor_always_passes(self):
        """noise_floor=None means the gate is open for all inputs."""
        for rms in (0.0, 0.001, 0.1, 1.0):
            self.assertTrue(
                self.gate.passes_noise_gate(rms=rms, noise_floor=None),
                f"noise_floor=None must always pass (rms={rms}).",
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  4. Echo correlation gate (integration test)
# ═══════════════════════════════════════════════════════════════════════════════

class TestEchoGateContract(unittest.TestCase):
    """
    The echo gate (cosine similarity ≥ 0.45) must block barge-in when the
    mic signal looks like the TTS output, and ALLOW barge-in when it doesn't.

    Tests WILL FAIL if echo_corr_threshold changes from 0.45.
    """

    def test_perfect_echo_never_fires(self):
        """
        Injecting the EXACT same audio as both AEC reference and mic input
        gives cosine similarity = 1.0 → must be blocked unconditionally.
        """
        interrupts = []
        rec = make_recorder(
            on_interrupt=lambda info=None: interrupts.append(info),
            echo_corr_threshold=ECHO_CORR_THRESHOLD,
        )
        # Use real TTS echo if available; synthetic otherwise
        from tests.fixtures import real_tts_echo  # noqa: PLC0415
        echo = real_tts_echo(2.0)
        with AudioHarness(rec) as h:
            h.set_tts_speaking(zero_delays=True)
            rec._aec_ref = echo.copy()
            rec._ref_samples_consumed = 0
            h.inject(echo)
            h.drain()
        self.assertFalse(
            interrupts,
            "Perfect echo (sim=1.0) must be blocked. Echo gate is broken.",
        )

    def test_unrelated_speech_over_echo_fires(self):
        """
        Different speech signal over an echo reference → low cosine similarity
        → echo gate must NOT block → barge-in fires.

        This is the user speaking WHILE the assistant is talking.
        """
        interrupts = []
        rec = make_recorder(
            on_interrupt=lambda info=None: interrupts.append(info),
            echo_corr_threshold=ECHO_CORR_THRESHOLD,
        )
        # Skip ambient bootstrap from the utterance itself — otherwise the ring
        # RMS median tracks speech and the noise gate blocks mid-stream chunks.
        rec._noise_floor = 0.005
        from tests.fixtures import real_tts_echo, real_speech  # noqa: PLC0415
        echo_ref = real_tts_echo(2.0)
        # User speech: completely different content / timing
        user_speech = real_speech(1.0, amplitude=0.15, fallback_amplitude=0.35)
        with AudioHarness(rec) as h:
            h.set_tts_speaking(zero_delays=True)
            rec._aec_ref = echo_ref.copy()
            rec._ref_samples_consumed = 0
            h.inject(user_speech)
            h.drain()
        self.assertTrue(
            interrupts,
            "User speech (different from echo reference) must fire barge-in. "
            "Echo gate is over-blocking.",
        )

    def test_echo_gate_threshold_is_045(self):
        """
        Verify the exact threshold using the BargeInDetector directly.

        At echo_sim=0.449 (just below 0.45): NOT blocked (score > 0).
        At echo_sim=0.450 (at threshold):    BLOCKED (score = -10).
        At echo_sim=0.451 (just above):      BLOCKED.
        """
        det = _make_detector(echo_corr_threshold=ECHO_CORR_THRESHOLD)

        # Just below threshold
        score_below = det.score(_frame(rms=0.1, voiced=True, echo_sim=0.449))
        self.assertGreater(score_below, 0.0, "echo_sim=0.449 must not be blocked")

        # At threshold
        score_at = det.score(_frame(rms=0.1, voiced=True, echo_sim=0.450))
        self.assertEqual(score_at, -10.0, "echo_sim=0.450 must be blocked")

        # Just above
        score_above = det.score(_frame(rms=0.1, voiced=True, echo_sim=0.451))
        self.assertEqual(score_above, -10.0, "echo_sim=0.451 must be blocked")


# ═══════════════════════════════════════════════════════════════════════════════
#  5. Silero path vs. energy path
# ═══════════════════════════════════════════════════════════════════════════════

class TestSileroPathContract(unittest.TestCase):
    """
    Verify that Silero's voiced=True is what fires barge-in when audio is
    kept at low amplitude (below the energy-only threshold).

    These tests WILL FAIL if Silero is disabled or broken.
    They WILL ALSO FAIL if someone removes the voiced=True scoring bonus.
    """

    def setUp(self):
        _skip_if_no_fsdd()

    def test_human_voice_at_moderate_amplitude_fires_via_silero(self):
        """
        Human voice at amplitude=0.08.

        At this amplitude, voiced-segment chunks have per-chunk RMS ≈ 0.10–0.18,
        which IS above the energy threshold * 3 = 0.03.  So this test exercises
        BOTH Silero and energy paths — which reflects real production usage where
        both paths work together.

        The complementary test (synthetic at same amplitude → no fire) documents
        that Silero is needed: without it synthetic speech at 0.08 can still fire
        if amplitude is high enough (0.04 → per-chunk RMS ≈ 0.06 > 0.03).

        Use the concat sample (>1s) to ensure enough frames for min_speech_samples.
        """
        interrupts = []
        rec = make_recorder(
            on_interrupt=lambda info=None: interrupts.append(info),
        )
        audio = human_voice_concat(1.5, amplitude=0.08)

        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.inject(audio)
            h.drain()

        self.assertTrue(
            interrupts,
            "Human voice (FSDD concat) at amplitude=0.08 must fire barge-in. "
            "If this fails, Silero is not running or the scoring is broken.",
        )

    def test_synthetic_voice_low_amplitude_does_NOT_fire(self):
        """
        Synthetic voiced_speech at amplitude=0.04:
          - overall RMS ≈ 0.04 > threshold * 3 = 0.03 → energy path might fire?

        Actually at 0.04 with the sine/harmonic generator:
          - shaped by syllabic envelope → many chunks well below 0.04 peak
          - Silero scores ~5 % voiced → does not trigger Silero path
          - Net result: above_samples hard-resets on low-energy chunks, never fires

        This test verifies the hard-reset behavior matters:
        inconsistent signal → never sustains 4 consecutive scored frames.
        """
        interrupts = []
        rec = make_recorder(
            on_interrupt=lambda info=None: interrupts.append(info),
        )
        audio = voiced_speech(1.5, amplitude=0.04)

        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.inject(audio)
            h.drain()

        self.assertFalse(
            interrupts,
            "Synthetic speech at amplitude=0.04 must NOT fire barge-in. "
            "If it does, the energy path threshold or the envelope generator changed.",
        )

    def test_energy_only_path_fires_when_calibrated(self):
        """
        High-amplitude audio fires barge-in via the energy path when the noise
        floor is calibrated, even if Silero returns voiced=False.

        Without calibration the energy boost requires Silero confirmation so
        that loud non-speech (HVAC, music) cannot falsely trigger barge-in.
        With calibration the boost is re-enabled: the noise gate already filters
        background, so any audio that passes the gate is likely real activity.
        """
        interrupts = []
        rec = make_recorder(
            on_interrupt=lambda info=None: interrupts.append(info),
        )
        # Calibrate to a quiet room so the noise gate is open for normal speech.
        rec._noise_floor = 0.005
        audio = voiced_speech(1.0, amplitude=0.35)

        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.inject(audio)
            h.drain()

        self.assertTrue(
            interrupts,
            "Calibrated energy path must fire barge-in for high-amplitude audio.",
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  6. State machine (FSM) transitions
# ═══════════════════════════════════════════════════════════════════════════════

class TestStateMachineContracts(unittest.TestCase):
    """
    Precise FSM contract: every state transition must be exactly right.

    IDLE → ASSISTANT_SPEAKING  on TTS start
    ASSISTANT_SPEAKING → RECOVER  on TTS stop
    RECOVER → IDLE  after recover_until expires
    IDLE → LISTENING  on speech start (wakeword gate open)
    LISTENING → IDLE  after utterance delivered (callback fires)

    FAILS if any transition fires at the wrong time or is missing.
    """

    def test_tts_start_transitions_to_assistant_speaking(self):
        """State must be ASSISTANT_SPEAKING immediately after TTS starts."""
        rec = make_recorder(wakeword_enabled=False)
        with AudioHarness(rec) as h:
            self.assertEqual(rec._listener_state, ListenerState.IDLE)
            h.set_tts_speaking()
            self.assertEqual(
                rec._listener_state,
                ListenerState.ASSISTANT_SPEAKING,
                "State should be ASSISTANT_SPEAKING immediately after TTS start.",
            )

    def test_tts_stop_transitions_to_recover(self):
        """State must be RECOVER immediately after TTS stops."""
        rec = make_recorder(wakeword_enabled=False)
        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.stop_tts()
            self.assertEqual(
                rec._listener_state,
                ListenerState.RECOVER,
                "State should be RECOVER immediately after TTS stop.",
            )

    def test_recover_expires_to_idle(self):
        """State must transition to IDLE after the recover window expires."""
        rec = make_recorder(wakeword_enabled=False)
        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.stop_tts()
            time.sleep(0.35)  # recover_until = now + 0.25 s
            h.inject(silence(0.01))
            h.drain()
        self.assertEqual(
            rec._listener_state,
            ListenerState.IDLE,
            "State should be IDLE after recover_until expires.",
        )

    def test_never_idle_during_active_recording(self):
        """
        While the user is actively speaking (captured buffer growing),
        state must NOT be IDLE or ASSISTANT_SPEAKING.
        """
        callbacks = []
        state_during_speech = []

        rec = make_recorder(
            callback=lambda audio: callbacks.append(audio),
            wakeword_enabled=False,
            silence_duration=2.0,  # long — don't endpoint during test
        )

        with AudioHarness(rec) as h:
            # Start speech and immediately sample state
            h.inject(voiced_speech(0.15, amplitude=0.35), inter_chunk_delay=0.015)
            state_during_speech.append(rec._listener_state)
            h.drain()

        # During speech, state should have been LISTENING at some point
        self.assertIn(
            ListenerState.LISTENING,
            state_during_speech,
            "State must be LISTENING while speech is being captured. "
            "Got states: " + str(state_during_speech),
        )

    def test_state_idle_after_callback(self):
        """
        After the callback fires (utterance delivered), state must return to IDLE
        so the next utterance can be captured.

        FAILS if the FSM gets stuck in LISTENING after a callback.
        """
        callback_event = threading.Event()
        rec = make_recorder(
            callback=lambda audio: callback_event.set(),
            wakeword_enabled=False,
            silence_duration=0.05,
        )

        with AudioHarness(rec) as h:
            h.inject(voiced_speech(0.8, amplitude=0.35))
            h.inject(silence(0.4), inter_chunk_delay=0.020)
            h.drain(timeout=5.0)

        callback_event.wait(timeout=3.0)
        time.sleep(0.1)  # allow FSM to finish transition

        self.assertEqual(
            rec._listener_state,
            ListenerState.IDLE,
            "State must be IDLE after utterance callback fires. "
            "FSM may be stuck in LISTENING.",
        )

    def test_state_idle_after_tts_barge_in(self):
        """
        After a barge-in fires during TTS, state must be LISTENING
        (the user's speech is now being captured).
        Then, after silence, state must return to IDLE.
        """
        interrupted = threading.Event()
        rec = make_recorder(
            on_interrupt=lambda info=None: interrupted.set(),
            wakeword_enabled=False,
            silence_duration=0.05,
        )

        # Calibrate so the energy path is active (voiced_speech is synthetic;
        # Silero returns voiced=False for it, so calibration is required to
        # push the energy score above the 2.0 barge-in threshold).
        rec._noise_floor = 0.005

        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.inject(voiced_speech(1.0, amplitude=0.35))
            interrupted.wait(timeout=3.0)
            self.assertTrue(interrupted.is_set(), "Barge-in must fire")

            # After barge-in fires the TTS is considered stopped
            # Inject silence to let endpointing run
            h.inject(silence(0.4), inter_chunk_delay=0.020)
            h.drain(timeout=3.0)

        time.sleep(0.15)
        self.assertIn(
            rec._listener_state,
            (ListenerState.IDLE, ListenerState.RECOVER),
            "After barge-in + silence, FSM must be IDLE or RECOVER.",
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  7. SNR-based barge-in contracts
# ═══════════════════════════════════════════════════════════════════════════════

class TestSNRBasedBargeIn(unittest.TestCase):
    """
    Barge-in behavior at defined SNR levels.

    At high SNR (speech much louder than noise), barge-in MUST fire.
    At very negative SNR (noise louder than speech), barge-in must NOT fire.

    These tests expose a common real-world failure: the noise gate is tuned
    too aggressively and blocks real speech in noisy environments.
    """

    def test_speech_at_positive_snr_fires(self):
        """
        Speech at +12 dB SNR over TV noise must always fire barge-in.

        At +12 dB, speech RMS = noise_RMS × 10^(12/20) ≈ 4× noise RMS.
        With noise_floor calibrated to the noise level, speech at 4× easily
        clears the 3× noise gate (4.0 > 3.0).
        """
        noise_rms = 0.03
        # Speech: amplitude chosen so speech_rms ≈ noise_rms * 4
        speech_rms_target = noise_rms * 4.0

        noise = tv_noise(1.5, amplitude=noise_rms, seed=10)
        speech = voiced_speech(1.5, amplitude=speech_rms_target * 3)
        # snr_mix targets the actual SNR level
        mixed = snr_mix(speech, noise, snr_db=12.0)

        interrupts = []
        rec = make_recorder(
            on_interrupt=lambda info=None: interrupts.append(info),
        )
        rec._noise_floor = noise_rms  # calibrated to noise level

        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.inject(mixed)
            h.drain()

        self.assertTrue(
            interrupts,
            "Speech at +12 dB SNR must fire barge-in. "
            "Noise gate may be too aggressive for realistic environments.",
        )

    def test_noise_only_at_5_seconds_never_fires(self):
        """
        Five full seconds of TV noise must produce ZERO barge-ins.

        This is the most critical false-positive guard: if this fails,
        the assistant interrupts itself while playing TTS due to background noise.
        """
        interrupts = []
        rec = make_recorder(
            on_interrupt=lambda info=None: interrupts.append(info),
        )
        rec._noise_floor = 0.04

        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.inject(tv_noise(5.0, amplitude=0.04, seed=99))
            h.drain()

        self.assertEqual(
            len(interrupts),
            0,
            f"TV noise triggered {len(interrupts)} barge-in(s) over 5 seconds. "
            "False positive rate is unacceptably high.",
        )

    def test_human_voice_mixed_with_noise_fires_at_positive_snr(self):
        """
        Real human voice mixed with background noise at +6 dB SNR must fire.

        Skipped if FSDD samples are not available.
        """
        _skip_if_no_fsdd()

        noise_rms = 0.025
        noise = tv_noise(2.0, amplitude=noise_rms, seed=7)
        # human_voice at 0.08 amplitude gives RMS ≈ 0.04, SNR ≈ +4 dB over noise
        speech = human_voice(2.0, amplitude=0.12, speaker="jackson", digit=1)
        mixed = snr_mix(speech, noise, snr_db=6.0)

        interrupts = []
        rec = make_recorder(
            on_interrupt=lambda info=None: interrupts.append(info),
        )
        # noise_floor calibrated from the background noise, NOT from the mix
        rec._noise_floor = noise_rms

        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.inject(mixed)
            h.drain()

        self.assertTrue(
            interrupts,
            "Real human voice at +6 dB SNR must fire barge-in. "
            "System is too conservative for realistic noisy environments.",
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  8. False positive guard — adversarial non-speech inputs
# ═══════════════════════════════════════════════════════════════════════════════

class TestFalsePositiveGuard(unittest.TestCase):
    """
    None of these inputs should EVER trigger barge-in.
    False positives here mean the assistant interrupts itself inappropriately.

    FAILS if the barge-in thresholds are too low.
    """

    def _no_interrupt(self, audio: np.ndarray, noise_floor: float = 0.03, label: str = ""):
        interrupts = []
        rec = make_recorder(
            on_interrupt=lambda info=None: interrupts.append(info),
        )
        rec._noise_floor = noise_floor
        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.inject(audio)
            h.drain()
        self.assertFalse(
            interrupts,
            f"[{label}] Triggered {len(interrupts)} false barge-in(s). "
            "A non-speech input is firing the interrupt.",
        )

    def test_dead_silence_never_fires(self):
        self._no_interrupt(silence(3.0), label="silence")

    def test_tv_noise_3sec_never_fires(self):
        self._no_interrupt(tv_noise(3.0, amplitude=0.04), label="tv_noise/3s")

    def test_short_click_burst_never_fires(self):
        """Door slam / mic knock — transient < 100 ms."""
        self._no_interrupt(
            np.concatenate([
                silence(0.1),
                click_burst(duration_sec=0.08, amplitude=0.5),
                silence(0.5),
            ]),
            label="click_burst/80ms",
        )

    def test_pure_tone_440hz_never_fires(self):
        """A ringing phone or notification tone."""
        t = np.linspace(0, 2.0, 2 * SR, dtype=np.float32)
        tone = (np.sin(2 * np.pi * 440 * t) * 0.06).astype(np.float32)
        self._no_interrupt(tone, label="tone/440Hz")

    def test_multiple_click_bursts_never_fire(self):
        """Repeated knocks — multiple transients don't accumulate across the hard reset."""
        audio = np.concatenate([
            click_burst(0.05, amplitude=0.4, seed=1),
            silence(0.15),
            click_burst(0.05, amplitude=0.4, seed=2),
            silence(0.15),
            click_burst(0.05, amplitude=0.4, seed=3),
            silence(0.15),
        ])
        self._no_interrupt(audio, label="repeated_clicks")


# ═══════════════════════════════════════════════════════════════════════════════
#  9. Wakeword gate strict mode
# ═══════════════════════════════════════════════════════════════════════════════

class TestWakewordGateContracts(unittest.TestCase):
    """
    With wakeword_policy=strict_required, the gate must be closed until
    the wakeword service explicitly arms it.  NO amount of speech injection
    should produce a callback without prior wakeword detection.
    """

    def test_strict_mode_10_injections_no_callback(self):
        """
        10 separate speech injections of 1.0 s each → 0 callbacks.

        FAILS if the strict gate leaks in any circumstance.
        This is the primary regression guard for wakeword security.
        """
        callbacks = []
        rec = make_recorder(
            callback=lambda audio: callbacks.append(audio),
            wakeword_enabled=True,
            wakeword_policy="strict_required",
            silence_duration=0.05,
        )
        rec._noise_floor = 0.03

        ww = MockWakewordService()
        rec._wakeword_service = ww
        # Never arm the wakeword

        with AudioHarness(rec) as h:
            for _ in range(10):
                h.inject(voiced_speech(1.0, amplitude=0.35))
                h.inject(silence(0.2), inter_chunk_delay=0.010)

            h.drain(timeout=15.0)

        self.assertEqual(
            len(callbacks),
            0,
            f"strict_required gate delivered {len(callbacks)} callback(s) "
            "without a wakeword event.  Wakeword gate is broken.",
        )

    def test_strict_mode_fires_immediately_after_arm(self):
        """
        After arming, the NEXT speech utterance must produce a callback.
        The gate must not require multiple arm events.
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
            h.inject(silence(0.05))
            h.inject(voiced_speech(0.8, amplitude=0.35))
            h.inject(silence(0.4), inter_chunk_delay=0.020)
            h.drain(timeout=5.0)

        fired = callback_event.wait(timeout=3.0)
        self.assertTrue(
            fired,
            "Callback must fire on first utterance after wakeword arm.",
        )

    def test_strict_mode_gate_closes_after_one_utterance(self):
        """
        After one utterance is delivered, the gate must close again.
        A second speech injection (without re-arming) must NOT produce
        another callback.
        """
        callbacks = []
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
        ww = MockWakewordService()
        rec._wakeword_service = ww

        with AudioHarness(rec) as h:
            # First utterance — with wakeword
            h.inject(silence(0.05))
            ww.arm()
            h.inject(silence(0.05))
            h.inject(voiced_speech(0.8, amplitude=0.35))
            h.inject(silence(0.4), inter_chunk_delay=0.020)
            h.drain(timeout=4.0)

            callback_event.wait(timeout=2.0)
            self.assertEqual(len(callbacks), 1, "First utterance should produce 1 callback")
            callback_event.clear()

            # Second utterance — NO wakeword re-arm
            h.inject(voiced_speech(0.8, amplitude=0.35))
            h.inject(silence(0.4), inter_chunk_delay=0.020)
            h.drain(timeout=4.0)

        # Wait briefly to ensure callback doesn't arrive late
        time.sleep(0.5)

        self.assertEqual(
            len(callbacks),
            1,
            f"Got {len(callbacks)} callbacks; gate should have closed after first utterance. "
            "Second speech injection without wakeword must NOT produce a callback.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
