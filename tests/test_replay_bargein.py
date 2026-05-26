"""
Replay-style synthetic barge-in reliability tests.
"""
import unittest
import time

from utils.audio import BargeInDetector, FrameFeatures
from utils.dialogue_controller import DialogueController, BargeInInfo


class TestReplayBargeInReliability(unittest.TestCase):
    def test_tv_like_echo_should_not_trigger(self):
        detector = BargeInDetector(
            sample_rate=16000, min_speech_sec=0.2, echo_corr_threshold=0.45
        )
        triggered = False
        for _ in range(12):
            frame = FrameFeatures(
                timestamp=time.time(),
                rms=0.08,
                threshold=0.01,
                voiced=True,
                echo_similarity=0.8,  # strong playback similarity
                raw_rms=0.08,
            )
            if detector.update(frame, frame_len=1024):
                triggered = True
                break
        self.assertFalse(triggered)

    def test_overlapping_user_speech_should_trigger(self):
        detector = BargeInDetector(
            sample_rate=16000, min_speech_sec=0.2, echo_corr_threshold=0.45
        )
        triggered = False
        for _ in range(6):
            frame = FrameFeatures(
                timestamp=time.time(),
                rms=0.05,
                threshold=0.01,
                voiced=True,
                echo_similarity=0.1,  # not echo-like
                raw_rms=0.05,
            )
            if detector.update(frame, frame_len=1024):
                triggered = True
                break
        self.assertTrue(triggered)

    def test_layered_policy_blocks_unvoiced_without_partial(self):
        controller = DialogueController(
            min_interrupt_delay_sec=0.0,
            allow_rms_fallback=False,
            require_partial_for_barge_in=True,
        )
        controller.on_tts_start()
        info = BargeInInfo(
            rms=0.25,
            threshold=0.1,
            voiced=False,
            duration_sec=0.4,
            timestamp=time.time(),
            echo=False,
        )
        self.assertFalse(controller.should_stop_speaking(info))


if __name__ == "__main__":
    unittest.main(verbosity=2)
