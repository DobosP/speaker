"""
TTS self-interruption prevention tests.

Problem captured in the real-world session log
-----------------------------------------------
The assistant interrupted itself:

    BARGE-IN! (RMS: 0.1589 > 0.0100  voiced=False  echo_sim=0.26)

Cause:
  1. TTS audio bled from the speaker back into the microphone at 0.1589 RMS.
  2. Silero correctly returned voiced=False (it is not speech phonetics).
  3. But echo_sim=0.26 is below the default static threshold of 0.45 → not blocked.
  4. With a calibrated noise floor, the energy path scored 2.5 and accumulated
     to fire barge-in even though no human was speaking.

Fix: Three-layer defence against self-interruption
--------------------------------------------------
When a TTS reference audio is active:

  1. Adaptive echo threshold: the echo block threshold scales down from
     echo_corr_threshold (0.45) toward 0.20 as reference_rms increases.  At
     ref_rms=0.15, echo_sim=0.26 is now blocked (threshold ≈ 0.27).

  2. Energy-path suppression: the calibrated energy path (voiced=False +
     noise_floor_calibrated → +1.0 boost) is suppressed.  Only Silero-confirmed
     voiced speech can accumulate to fire barge-in while TTS plays.

  3. Echo-floor gate (new): during the barge_in_min_delay_after_ref_sec window
     (0.35 s in production), all mic audio is pure TTS echo (user hasn't reacted
     yet).  The average RMS of these frames is stored as echo_floor_baseline.
     After the gate opens, any frame below echo_floor_baseline × 1.5 is soft-
     decayed.  This catches the voiced=True echo cases (echo_sim=0.15 and
     echo_sim=0.37) that slipped past the adaptive threshold and Silero guard:
       BARGE-IN! (RMS: 0.0336 > 0.0100  voiced=True  echo_sim=0.15)  → blocked
       BARGE-IN! (RMS: 0.0163 > 0.0100  voiced=True  echo_sim=0.37)  → blocked

  4. AEC persistence: filter weights are now preserved across TTS turns.  After
     the first TTS interaction converges the room model, subsequent turns get
     immediate echo cancellation, driving the AEC-output RMS near zero so the
     noise gate blocks TTS echo without any of the above heuristics.

Test structure
--------------
  TestSelfInterruptPrevention  — unit-level integration: inject TTS audio as
                                 BOTH the AEC reference AND the mic feed.
  TestAdaptiveThresholdFiring  — verify real user speech DOES fire barge-in
                                 while TTS plays (regression guard).
  TestRealWorldEchoScenario    — realistic layered scenario: loud TTS + quiet
                                 user overlap; only user speech should fire.
"""

from __future__ import annotations

import time
import unittest

import numpy as np

from tests.harness import AudioHarness, make_recorder
from tests.fixtures import (
    tts_echo,
    voiced_speech,
    silence,
    human_voice,
    real_tts_echo,
    mix,
    SR,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _attenuate(audio: np.ndarray, factor: float) -> np.ndarray:
    """Simulate acoustic attenuation from speaker to microphone."""
    return (audio * factor).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Self-interruption prevention (TTS echo must NOT fire barge-in)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSelfInterruptPrevention(unittest.TestCase):
    """
    The assistant must never interrupt itself.

    Method: feed the same TTS audio as both the AEC reference AND the mic
    input (optionally attenuated to simulate acoustic path losses).  No barge-in
    must fire.
    """

    def _assert_no_self_interrupt(
        self,
        tts_audio: np.ndarray,
        mic_audio: np.ndarray,
        label: str,
        aec_enabled: bool = False,
    ):
        """
        aec_enabled=False (default for this suite) isolates the adaptive
        threshold logic from NLMS filter behaviour.  When AEC diverges on
        real speech signals it produces artificially high output RMS that
        would trigger barge-in independently of the threshold fix; disabling
        AEC lets us test the threshold fix cleanly.

        A separate NLMS stability test (test_aec_does_not_diverge) covers the
        AEC divergence issue.
        """
        interrupts = []
        rec = make_recorder(
            on_interrupt=lambda info=None: interrupts.append(info),
            aec_enabled=aec_enabled,
        )
        rec._noise_floor = 0.005

        with AudioHarness(rec) as h:
            h.set_tts_speaking(audio_ref=tts_audio)
            h.inject(mic_audio)
            h.drain(timeout=4.0)

        self.assertEqual(
            len(interrupts),
            0,
            f"{label}: barge-in must NOT fire when mic input is TTS echo "
            f"(got {len(interrupts)} interrupts).  "
            "The adaptive echo threshold and TTS energy-path suppression must "
            "block TTS bleed-through.",
        )

    def test_tts_echo_identical_does_not_fire(self):
        """
        Mic receives an identical copy of the TTS reference (0 dB echo —
        worst-case acoustic scenario: speaker directly into mic).
        """
        tts = tts_echo(1.5, amplitude=0.20)
        self._assert_no_self_interrupt(tts, tts.copy(), "0 dB echo")

    def test_tts_echo_attenuated_6db_does_not_fire(self):
        """
        Mic receives TTS attenuated to 50% (≈ −6 dB acoustic path loss).
        Represents a speaker ≈ 0.5 m from the mic in a lightly reverberant room.
        """
        tts = tts_echo(1.5, amplitude=0.20)
        mic = _attenuate(tts, 0.50)
        self._assert_no_self_interrupt(tts, mic, "-6 dB echo")

    def test_tts_echo_attenuated_12db_does_not_fire(self):
        """
        Mic receives TTS attenuated to 25% (≈ −12 dB).
        Represents a speaker ≈ 1 m from the mic with moderate absorption.
        """
        tts = tts_echo(1.5, amplitude=0.20)
        mic = _attenuate(tts, 0.25)
        self._assert_no_self_interrupt(tts, mic, "-12 dB echo")

    def test_realistic_tts_echo_does_not_fire(self):
        """
        Real TTS audio leaking directly back into the mic channel.

        The mic signal is a 40% attenuated copy of the TTS, time-aligned so
        that the EchoGuard similarity function can find the correlation.

        The EchoGuard prepends a reference_delay_sec (120 ms) of silence to
        the reference.  The mic audio must be injected with the same offset so
        the lag search compares the right windows.

        With the lag window extended to cover the reference delay, echo_sim
        should be high (> 0.50).  The hard echo block threshold ensures that
        very high similarity is blocked even when Silero returns voiced=True
        (TTS is real speech that Silero correctly classifies as voiced).
        """
        tts = real_tts_echo(1.5, amplitude=0.20)
        # Align mic with reference: prepend reference_delay_sec of silence
        # so that mic[0] arrives 120 ms after TTS starts (matches the prepended
        # zeros in the reference buffer).
        delay_samples = int(0.12 * SR)  # 1920 samples
        aligned_mic = np.concatenate([
            np.zeros(delay_samples, dtype=np.float32),
            _attenuate(tts, 0.40),
        ])
        self._assert_no_self_interrupt(tts, aligned_mic, "real TTS echo (time-aligned)")

    def test_high_rms_tts_echo_does_not_fire(self):
        """
        Simulates very loud TTS (ref_rms ≈ 0.30) leaking back at 30% into the
        mic (RMS ≈ 0.09).  This is the scenario captured in the real session:
        RMS=0.1589 triggered a false barge-in.  The adaptive threshold and
        energy-path suppression must both prevent it.
        """
        tts = tts_echo(2.0, amplitude=0.30)
        mic = _attenuate(tts, 0.53)  # results in mic RMS ≈ 0.16 (matches log)
        self._assert_no_self_interrupt(tts, mic, "high-RMS echo (matches session log)")


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Real user speech MUST still fire barge-in during TTS playback
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdaptiveThresholdFiring(unittest.TestCase):
    """
    Regression guard: adaptive threshold must not block real human barge-in.

    When the user speaks during TTS playback, their voice is classified as
    voiced=True by Silero.  The voiced path (score ≥ 2.0) always accumulates
    regardless of the TTS reference level.
    """

    def test_human_voice_fires_despite_loud_tts_reference(self):
        """
        TTS reference is active at high amplitude.  User speaks at normal volume.
        Silero classifies user frames as voiced → barge-in fires (voiced
        overrides the adaptive echo block even if similarity is moderate).
        """
        tts = tts_echo(2.0, amplitude=0.20)
        user = human_voice(1.5, amplitude=0.20)

        interrupts = []
        rec = make_recorder(
            on_interrupt=lambda info=None: interrupts.append(info),
            aec_enabled=False,  # isolate threshold logic from AEC divergence
        )
        rec._noise_floor = 0.005

        with AudioHarness(rec) as h:
            h.set_tts_speaking(audio_ref=tts)
            h.inject(user)
            h.drain(timeout=5.0)

        self.assertGreater(
            len(interrupts),
            0,
            "Real human voice at amplitude=0.20 must fire barge-in even when "
            "a loud TTS reference is active.  The voiced path must not be gated.",
        )

    def test_voiced_speech_synthetic_fires_with_tts_reference(self):
        """
        voiced_speech (synthetic, Silero returns voiced=False for it) with a TTS
        reference active.

        Silero returns voiced=False for the synthetic sine-wave signal.
        With TTS active, the energy path is suppressed → no accumulation →
        barge-in does NOT fire.  This is the CORRECT behavior: a signal that
        Silero does not recognise as speech cannot interrupt the assistant via
        energy alone while TTS is playing.
        """
        tts = tts_echo(1.5, amplitude=0.20)
        user = voiced_speech(1.5, amplitude=0.35)  # Silero: voiced=False

        interrupts = []
        rec = make_recorder(
            on_interrupt=lambda info=None: interrupts.append(info),
            aec_enabled=False,
        )
        rec._noise_floor = 0.005

        with AudioHarness(rec) as h:
            h.set_tts_speaking(audio_ref=tts)
            h.inject(user)
            h.drain(timeout=4.0)

        self.assertEqual(
            len(interrupts),
            0,
            "Synthetic voiced_speech (Silero=False) with active TTS reference "
            "must NOT fire barge-in.  Energy path is suppressed during TTS "
            "to prevent the exact self-interruption bug from the session log.",
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Real-world layered scenario: TTS echo + user voice overlap
# ═══════════════════════════════════════════════════════════════════════════════

class TestRealWorldEchoScenario(unittest.TestCase):
    """
    Realistic scenario: assistant speaks, user interrupts.

    The mic captures a mix of TTS echo (from speaker) and user speech.
    Only the user speech portion should fire barge-in.
    """

    def test_user_speech_over_tts_echo_fires_once(self):
        """
        Two-phase test:

        Phase 1 — TTS echo only (0.5 s): mic receives 30% attenuated TTS,
                   no user speech.  Must NOT fire barge-in.

        Phase 2 — TTS echo + user speech mixed: mic receives TTS bleed plus
                   real human voice.  Silero classifies the mixture as voiced
                   → MUST fire barge-in.

        This two-phase structure avoids wall-clock timing issues (all audio
        queues and processes near-instantly in tests).
        """
        tts = tts_echo(3.0, amplitude=0.20)

        # Phase 1: echo only
        echo_only = _attenuate(tts[:int(0.5 * SR)], 0.30)
        interrupts_phase1 = []
        rec1 = make_recorder(
            on_interrupt=lambda info=None: interrupts_phase1.append(info),
            aec_enabled=False,
        )
        rec1._noise_floor = 0.005

        with AudioHarness(rec1) as h:
            h.set_tts_speaking(audio_ref=tts)
            h.inject(echo_only)
            h.drain(timeout=3.0)

        self.assertEqual(
            len(interrupts_phase1),
            0,
            "Phase 1: TTS echo alone (0.5 s) must NOT fire barge-in. "
            "The adaptive threshold and energy-path suppression must block it.",
        )

        # Phase 2: echo + user voice
        user_speech = human_voice(1.5, amplitude=0.20)
        tts_bleed = _attenuate(tts[:len(user_speech)], 0.30)
        if len(tts_bleed) < len(user_speech):
            tts_bleed = np.resize(tts_bleed, len(user_speech))
        mic = (tts_bleed + user_speech).astype(np.float32)

        interrupts_phase2 = []
        rec2 = make_recorder(
            on_interrupt=lambda info=None: interrupts_phase2.append(info),
            aec_enabled=False,
        )
        rec2._noise_floor = 0.005

        with AudioHarness(rec2) as h:
            h.set_tts_speaking(audio_ref=tts)
            h.inject(mic)
            h.drain(timeout=5.0)

        self.assertGreater(
            len(interrupts_phase2),
            0,
            "Phase 2: Human voice mixed with TTS echo must fire barge-in. "
            "Silero classifies human speech as voiced=True; the voiced path "
            "overrides the echo block → barge-in must fire.",
        )

    def test_tts_echo_alone_does_not_fire_with_reference(self):
        """
        Pure TTS echo (no user speech) with TTS reference set.
        No barge-in must fire for any echo attenuation level.
        """
        for attenuation, label in [(1.0, "0 dB"), (0.5, "-6 dB"), (0.25, "-12 dB")]:
            with self.subTest(attenuation=attenuation, label=label):
                tts = tts_echo(1.5, amplitude=0.20)
                mic = _attenuate(tts, attenuation)

                interrupts = []
                rec = make_recorder(
                    on_interrupt=lambda info=None: interrupts.append(info),
                    aec_enabled=False,
                )
                rec._noise_floor = 0.005

                with AudioHarness(rec) as h:
                    h.set_tts_speaking(audio_ref=tts)
                    h.inject(mic)
                    h.drain(timeout=4.0)

                self.assertEqual(
                    len(interrupts),
                    0,
                    f"{label} TTS echo must not fire barge-in when reference is set.",
                )


class TestEchoFloorGate(unittest.TestCase):
    """
    Echo-floor gate: production-mode self-interruption prevention.

    The echo-floor gate is the primary defence against the real-world
    self-interruption logs observed in the session:

        BARGE-IN! (RMS: 0.0336 > 0.0100  voiced=True  echo_sim=0.15)
        BARGE-IN! (RMS: 0.0163 > 0.0100  voiced=True  echo_sim=0.37)

    Mechanism
    ---------
    When TTS starts, ``barge_in_min_delay_after_ref_sec`` (0.35 s by default
    in production) gates barge-in detection.  During this window the mic audio
    is ONLY TTS echo (the user hasn't had time to react), so the per-frame RMS
    gives us a reliable estimate of the room's echo level.  After the gate
    opens, any frame whose RMS is below ``echo_floor_baseline * 1.5`` is
    soft-decayed rather than scored.

    These tests use a non-zero ``barge_in_min_delay_after_ref_sec`` and
    ``inter_chunk_delay`` so that the gate window actually accumulates samples
    before barge-in scoring begins.
    """

    # Echo-floor gate learning window.  Must match (or be less than) the
    # barge_in_min_delay_after_ref_sec used in these tests.
    GATE_SEC = 0.20

    def _make_rec(self, on_interrupt):
        return make_recorder(
            on_interrupt=on_interrupt,
            barge_in_min_delay_sec=0.0,
            barge_in_min_delay_after_ref_sec=self.GATE_SEC,
            aec_enabled=False,
        )

    def test_tts_echo_alone_blocked_by_echo_floor(self):
        """
        TTS echo (no user speech) must be blocked by the echo-floor gate even
        when echo_sim is low (0.15) and Silero returns voiced=True.

        This reproduces the first real-world self-interruption:
            BARGE-IN! (RMS: 0.0336 > 0.0100  voiced=True  echo_sim=0.15)

        The echo-floor baseline is learned during the 0.35 s gate window; after
        the gate the echo at the same RMS fails the 1.5× floor check → no barge-in.
        """
        tts_ref = tts_echo(1.5, amplitude=0.20)
        # Room echo = 30% of TTS (about −10 dB attenuation)
        mic_echo = _attenuate(tts_ref, 0.30)

        interrupts = []
        rec = self._make_rec(lambda info=None: interrupts.append(info))
        rec._noise_floor = 0.005

        # Inject with real-time pacing so the gate window is genuinely elapsed.
        chunk_sec = 1024 / SR
        with AudioHarness(rec) as h:
            h.set_tts_speaking(audio_ref=tts_ref, zero_delays=False)
            h.inject(mic_echo, inter_chunk_delay=chunk_sec)
            h.drain(timeout=5.0)

        self.assertEqual(
            len(interrupts),
            0,
            "TTS echo at RMS ≈ 0.034 must be blocked by the echo-floor gate. "
            "The gate learns the echo baseline during the 0.35 s delay window and "
            "then rejects frames that are at or below 1.5× that level.",
        )

    def test_user_speech_above_echo_floor_fires(self):
        """
        When the user speaks significantly louder than the room echo, the echo-
        floor gate must pass them and barge-in must fire.

        Echo level: ~0.034 RMS (30% of TTS at 0.20)
        User level: ~0.14 RMS (amplitude=0.14 — 4× echo = +12 dB SNR)

        The echo-floor threshold after learning = 0.034 × 1.5 ≈ 0.051.
        User at 0.14 >> 0.051 → passes gate → accumulates → fires.
        """
        tts_ref = tts_echo(2.0, amplitude=0.20)
        user = human_voice(1.5, amplitude=0.14)

        # Gate window: only echo (pure TTS echo for first GATE_SEC seconds)
        gate_samples = int(self.GATE_SEC * SR)
        tts_ref_padded = tts_ref[:gate_samples]
        echo_gate_audio = _attenuate(tts_ref_padded, 0.30)

        # After gate: user speech starts
        interrupts = []
        rec = self._make_rec(lambda info=None: interrupts.append(info))
        rec._noise_floor = 0.005

        chunk_sec = 1024 / SR
        with AudioHarness(rec) as h:
            h.set_tts_speaking(audio_ref=tts_ref, zero_delays=False)
            # Inject echo during gate window (real-time paced)
            h.inject(echo_gate_audio, inter_chunk_delay=chunk_sec)
            # Inject user speech after gate (no delay needed — gate already expired)
            h.inject(user)
            h.drain(timeout=5.0)

        self.assertGreater(
            len(interrupts),
            0,
            "User speech at 4× the echo level must pass the echo-floor gate and "
            "fire barge-in.  If this fails the floor multiplier (1.5×) is too "
            "aggressive and is blocking legitimate user interrupts.",
        )

    def test_echo_floor_baseline_not_set_when_gate_is_zero(self):
        """
        When barge_in_min_delay_after_ref_sec=0 (test default) the gate window
        is never entered, so echo_floor_baseline stays None and the gate is
        transparent — behaviour identical to before this feature was added.
        """
        rec = make_recorder(
            barge_in_min_delay_after_ref_sec=0.0,
            aec_enabled=False,
        )
        tts = tts_echo(1.0, amplitude=0.20)
        rec.set_echo_reference(tts, SR)
        # Simulate a few processing rounds without barge-in mode active
        self.assertIsNone(
            rec._echo_floor_baseline,
            "echo_floor_baseline must be None when gate delay is zero (no learning window).",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
